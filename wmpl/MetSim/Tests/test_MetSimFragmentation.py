""" Tests of the erosion into grains and of the fragmentation entries of MetSim.

The tests check that:
    - the eroded mass goes into grains in full, with the same distribution of grain masses at any time step,
    - the total mass counts the mass in the mass bins which is not released as grains yet,
    - the erosion coefficient of an "M" entry is kept below its height,
    - "EF" and "D" entries without a mass index take 2.0,
    - the count of active fragments stays right through a disruption,
    - the mass of an entry is limited to 100 %,
    - the constants can be saved after a run with the wake.

Run with pytest:
    python -m pytest wmpl/MetSim/Tests/test_MetSimFragmentation.py -v
"""

from __future__ import print_function, division, absolute_import

import json
import os
import tempfile

import numpy as np

import wmpl.MetSim.MetSimErosion as MetSimErosion
from wmpl.MetSim.GUI import FragmentationEntry, saveConstants



def erodingConstants(m_init):
    """ Make the default eroding constants with the given initial mass.

    Arguments:
        m_init: [float] Initial mass (kg).

    Return:
        const: [Constants object] Constants of the meteoroid.
    """

    const = MetSimErosion.Constants()
    const.erosion_on = True
    const.m_init = m_init

    return const



def runAndTrackGrains(const):
    """ Run the simulation and track the mass eroded into grains, and every grain which was made.

    Arguments:
        const: [Constants object] Constants of the meteoroid.

    Return:
        (eroded_mass, grains, n_active):
            - eroded_mass: [float] Total mass eroded into grains (kg).
            - grains: [dict] Mass of one grain and the number of grains, for every grain fragment ID.
            - n_active: [list] The count of active fragments, and the number of active fragments, after
                every step.
    """

    eroded_mass = [0.0]
    grains = {}
    n_active = []

    erode_into_grains = MetSimErosion.erodeIntoGrains
    ablate_all = MetSimErosion.ablateAll

    def erodeIntoGrains(const, frag, mass):
        eroded_mass[0] += mass
        return erode_into_grains(const, frag, mass)

    def ablateAll(fragments, const, *args, **kwargs):
        results = ablate_all(fragments, const, *args, **kwargs)
        for frag in fragments:
            if frag.grain and (frag.id not in grains):
                grains[frag.id] = (frag.m_init, frag.n_grains)
        n_active.append((const.n_active, sum([frag.active for frag in fragments])))
        return results

    MetSimErosion.erodeIntoGrains = erodeIntoGrains
    MetSimErosion.ablateAll = ablateAll

    try:
        MetSimErosion.runSimulation(const)

    finally:
        MetSimErosion.erodeIntoGrains = erode_into_grains
        MetSimErosion.ablateAll = ablate_all

    return eroded_mass[0], grains, n_active



def testErodedMassGoesIntoGrains():
    """ Check that the grains carry all of the eroded mass, including the fractions of grains left in the mass
        bins when the meteoroid is exhausted.
    """

    eroded_mass, grains, _ = runAndTrackGrains(erodingConstants(1e-7))
    grain_mass = sum([m*n for m, n in grains.values()])

    assert eroded_mass > 0
    assert abs(grain_mass - eroded_mass)/eroded_mass < 1e-9



def testGrainMassesDoNotDependOnTheTimeStep():
    """ Check that the mean mass of the grains, weighted by mass, is the same with a 5 and a 0.5 ms step,
        which it was not when every step rounded the number of grains down and moved the rest to smaller
        grains.
    """

    mean_masses = []
    for dt in [0.005, 0.0005]:

        const = erodingConstants(1e-7)
        const.dt = dt

        _, grains, _ = runAndTrackGrains(const)
        masses = np.array([m for m, _ in grains.values()])
        numbers = np.array([n for _, n in grains.values()])
        mean_masses.append(np.sum(masses**2*numbers)/np.sum(masses*numbers))

    print("Mean grain masses (kg):", mean_masses)

    assert abs(mean_masses[0] - mean_masses[1])/mean_masses[1] < 0.02



def testTotalMassIncludesTheMassBins():
    """ Check that the total mass of the active fragments stays the initial mass while the meteoroid erodes
        and nothing ablates, as the grains are heavier than the mass it erodes in one step, so the eroded mass
        waits in the mass bins.
    """

    const = erodingConstants(1e-7)
    const.sigma = 0.0
    const.erosion_mass_max = 1e-7
    const.erosion_mass_min = 1e-8

    _, results_list, _ = MetSimErosion.runSimulation(const)
    results = np.array(results_list, dtype=np.float64)

    # Total mass of the active fragments while the meteoroid is eroding
    eroding = results[:, 16] > 0
    mass_total = results[eroding, 15]

    print("Total mass, smallest and largest (kg):", np.min(mass_total), np.max(mass_total))

    assert np.allclose(mass_total, const.m_init, rtol=1e-9)



def testMainEntryKeepsTheErosionCoefficient():
    """ Check that the erosion coefficient of an "M" entry is kept below its height, and is not replaced by
        the one of the height in the next step.
    """

    const = erodingConstants(1e-4)
    const.erosion_height_start = 110000.0
    const.erosion_coeff = 0.1e-6
    const.fragmentation_on = True
    const.fragmentation_entries = [FragmentationEntry("M", 95000.0, None, None, None, None, 0.6e-6, None, \
        None, None)]

    coeffs = []
    ablate_all = MetSimErosion.ablateAll

    def ablateAll(fragments, const, *args, **kwargs):
        results = ablate_all(fragments, const, *args, **kwargs)
        if fragments[0].active:
            coeffs.append((fragments[0].h, fragments[0].erosion_coeff))
        return results

    MetSimErosion.ablateAll = ablateAll

    try:
        MetSimErosion.runSimulation(const)

    finally:
        MetSimErosion.ablateAll = ablate_all

    coeffs = np.array(coeffs)
    below = coeffs[:, 0] < 94000.0

    assert np.any(below)
    assert np.allclose(coeffs[below, 1], 0.6e-6)



def testNewFragmentsTakeTheDefaultMassIndex():
    """ Check that "EF" and "D" entries without a mass index take 2.0, as the header of the fragmentation file
        says, and that "M" keeps None, which means no change.
    """

    entries = [FragmentationEntry("EF", 95000.0, 2, 30.0, None, None, 0.3e-6, 1e-10, 5e-10, None), \
        FragmentationEntry("D", 90000.0, None, 20.0, None, None, None, 1e-10, 5e-10, None), \
        FragmentationEntry("M", 85000.0, None, None, None, None, 0.3e-6, None, None, None)]

    assert [entry.mass_index for entry in entries] == [2.0, 2.0, None]

    const = erodingConstants(1e-3)
    const.fragmentation_on = True
    const.fragmentation_entries = entries

    _, results_list, _ = MetSimErosion.runSimulation(const)

    assert np.all(np.isfinite(np.array(results_list, dtype=np.float64)))



def testActiveCountThroughDisruption():
    """ Check that the count of active fragments is the number of active fragments after a disruption, which
        counted the disrupted fragment twice.
    """

    const = erodingConstants(1e-3)
    const.disruption_on = True
    const.compressive_strength = 2000.0

    _, _, n_active = runAndTrackGrains(const)

    assert const.n_active == 0
    assert all([count == active for count, active in n_active])



def testEntryMassIsLimitedTo100Percent():
    """ Check that the mass of an entry is limited to 100 % of the parent, so it does not make mass. """

    entry = FragmentationEntry("F", 90000.0, 1, 150.0, None, None, None, None, None, None)

    assert entry.mass_percent == 100.0



def testConstantsCanBeSavedAfterTheWake():
    """ Check that the wake does not change the PSF weights of the constants, so they can be saved. """

    const = erodingConstants(MetSimErosion.Constants().m_init)
    const.wake_psf_weights = [0.9, 0.1]

    MetSimErosion.runSimulation(const, compute_wake=True)

    assert const.wake_psf_weights == [0.9, 0.1]

    dir_path = tempfile.mkdtemp()
    saveConstants(const, dir_path, 'const.json')

    with open(os.path.join(dir_path, 'const.json')) as f:
        assert json.load(f)['wake_psf_weights'] == [0.9, 0.1]



if __name__ == "__main__":

    testErodedMassGoesIntoGrains()
    testGrainMassesDoNotDependOnTheTimeStep()
    testTotalMassIncludesTheMassBins()
    testMainEntryKeepsTheErosionCoefficient()
    testNewFragmentsTakeTheDefaultMassIndex()
    testActiveCountThroughDisruption()
    testEntryMassIsLimitedTo100Percent()
    testConstantsCanBeSavedAfterTheWake()

    print("All tests passed")
