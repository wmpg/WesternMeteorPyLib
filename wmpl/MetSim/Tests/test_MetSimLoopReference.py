""" Tests of the loop over the fragments of MetSim against its Python reference.

The loop of MetSimErosion must give the same results as the Python implementation in
MetSimErosionReference.py to the last bit, on every path: the erosion into grains, the disruption, the
fragmentation entries, the wake, the atmosphere table and polynomial, the luminous efficiencies and the
substeps of small grains. The tests compare every value of every time step, the final fragments, the outputs
saved in the constants and the outputs of the fragmentation entries.

Run with pytest:
    python -m pytest wmpl/MetSim/Tests/test_MetSimLoopReference.py -v
"""

from __future__ import print_function, division, absolute_import

import contextlib
import copy
import io

import numpy as np

import wmpl.MetSim.MetSimErosion as MetSimErosion
import wmpl.MetSim.Tests.MetSimErosionReference as MetSimErosionReference
from wmpl.MetSim.GUI import FragmentationEntry


# Attributes of the fragments which are compared
FRAGMENT_ATTRIBUTES = ['id', 'm', 'm_init', 'h', 'v', 'vv', 'vh', 'length', 'lum', 'q', 'dyn_press', \
    'n_grains', 'active', 'grain', 'main', 'complex', 'sigma', 'K', 'rho', 'erosion_coeff', \
    'h_grav_drop_total', 'v_grav_drop']

# Outputs of the simulation saved in the constants
CONSTANTS_OUTPUTS = ['total_time', 'n_active', 'total_fragments', 'erosion_beg_vel', 'erosion_beg_mass', \
    'erosion_beg_dyn_press', 'mass_at_erosion_change', 'main_mass_exhaustion_ht', 'main_bottom_ht', \
    'disruption_height']

# Outputs of the fragmentation entries
ENTRY_OUTPUTS = ['done', 'time', 'dyn_pressure', 'velocity', 'parent_mass', 'mass', 'final_mass', \
    'main_time_data', 'main_luminosity', 'main_tau_over_lum', 'grains_time_data', 'grains_luminosity', \
    'grains_tau_over_lum']



def bits(value):
    """ Represent a value by its exact bits, so that two values are equal only if they are the same double.

    Arguments:
        value: [object] A number, None, or a list of them.

    Return:
        [object] The hexadecimal representation of a number, None, or a list of them.
    """

    if value is None:
        return None

    if isinstance(value, (list, tuple, np.ndarray)):
        return [bits(x) for x in value]

    return float(value).hex()



def tableConstants(const):
    """ Make the constants use a tabulated atmosphere, made from their density polynomial.

    Arguments:
        const: [Constants object] Constants, changed in place.

    Return:
        const: [Constants object] Constants with the table.
    """

    ht = np.arange(0, 200001, 500.0)
    log10_rho = np.zeros_like(ht)
    for i, coeff in enumerate(const.dens_co):
        log10_rho += coeff*(ht/1e6)**i

    const.atm_density_model = 'table'
    const.atm_table_ht = ht.tolist()
    const.atm_table_log10_rho = log10_rho.tolist()

    return const



def baseConstants(**kwargs):
    """ Make the default constants of an eroding meteoroid with the legacy polynomial atmosphere, changed by
        the given values.

    Keyword arguments:
        **kwargs: [dict] Names and values of the constants to change.

    Return:
        const: [Constants object] Constants.
    """

    const = MetSimErosion.Constants()
    const.atm_density_model = 'polynomial'
    const.disruption_on = False
    const.m_init = 2e-6
    const.h_kill = 80000.0

    for name, value in kwargs.items():
        setattr(const, name, value)

    return const



def runAndCollect(module, const, compute_wake):
    """ Run the simulation of the given module and collect every output in a comparable form.

    Arguments:
        module: [module] MetSimErosion or MetSimErosionReference.
        const: [Constants object] Constants, copied before the run.
        compute_wake: [bool] Compute the wake.

    Return:
        (out, const, frag_main):
            - out: [dict] Outputs, with the numbers given by their bits.
            - const: [Constants object] Constants after the run.
            - frag_main: [object] Main fragment at the end.
    """

    const = copy.deepcopy(const)

    # Keep the fragments of the last time step
    last_fragments = []
    ablate_all = module.ablateAll

    def ablateAll(fragments, const, *args, **kwargs):
        results = ablate_all(fragments, const, *args, **kwargs)
        last_fragments[:] = results[0]
        return results

    module.ablateAll = ablateAll

    try:

        # Silence the printout of a disruption
        with contextlib.redirect_stdout(io.StringIO()):
            frag_main, results_list, wake_results = module.runSimulation(const, compute_wake=compute_wake)

    finally:
        module.ablateAll = ablate_all


    out = {}
    out['results'] = [bits(row) for row in results_list]
    out['main'] = [bits(getattr(frag_main, name)) for name in FRAGMENT_ATTRIBUTES]
    out['fragments'] = [[bits(getattr(frag, name)) for name in FRAGMENT_ATTRIBUTES] \
        for frag in last_fragments]
    out['constants'] = [bits(getattr(const, name)) for name in CONSTANTS_OUTPUTS]
    out['entries'] = [[bits(getattr(entry, name)) for name in ENTRY_OUTPUTS] + [len(entry.fragments)] \
        for entry in const.fragmentation_entries]

    out['wake'] = []
    for wake in wake_results:
        if wake is None:
            out['wake'].append(None)
        else:
            out['wake'].append([bits(wake.length_array), bits(wake.wake_luminosity_profile), \
                len(wake.frag_list)])

    return out, const, frag_main



def assertSameAsReference(const, compute_wake=False):
    """ Check that MetSimErosion and the reference give the same outputs to the last bit.

    Arguments:
        const: [Constants object] Constants.

    Keyword arguments:
        compute_wake: [bool] Compute the wake. False by default.

    Return:
        (const, frag_main, n_wakes): Outputs of the reference run, to check that it takes the tested path.
            - const: [Constants object] Constants after the run.
            - frag_main: [object] Main fragment at the end.
            - n_wakes: [int] Number of time steps with the wake.
    """

    reference, const_ref, frag_main = runAndCollect(MetSimErosionReference, const, compute_wake)
    tested, _, _ = runAndCollect(MetSimErosion, const, compute_wake)

    for name in reference:
        assert tested[name] == reference[name], "The {:s} differ from the reference".format(name)

    n_wakes = len([wake for wake in reference['wake'] if wake is not None])

    return const_ref, frag_main, n_wakes



def testErosionAndDisruption():
    """ Check an eroding meteoroid which disrupts, with the disrupted fragments eroding further. """

    const = baseConstants(m_init=1e-6, disruption_on=True, compressive_strength=300.0, \
        disruption_erosion_coeff=0.2e-6, erosion_bins_per_10mass=5, h_kill=92000.0)

    const_ref, _, _ = assertSameAsReference(const)

    assert const_ref.disruption_height is not None
    assert const_ref.erosion_beg_vel is not None



def testDisruptionWithoutErosion():
    """ Check a disruption into fragments which do not erode. """

    const = baseConstants(m_init=1e-4, erosion_on=False, disruption_on=True, compressive_strength=1000.0)

    const_ref, _, _ = assertSameAsReference(const)

    assert const_ref.disruption_height is not None



def testTableAtmosphereAndGammaGrains():
    """ Check the tabulated atmosphere, the gamma distribution of grain masses and the erosion change of
        the density and the ablation coefficient.
    """

    const = tableConstants(baseConstants(erosion_grain_distribution='gamma', erosion_mass_index=2.3, \
        erosion_height_change=95000.0, erosion_coeff_change=0.1e-6, erosion_rho_change=3500.0, \
        erosion_sigma_change=0.015e-6, erosion_bins_per_10mass=5, m_init=1e-6, h_kill=92000.0))

    const_ref, _, _ = assertSameAsReference(const)

    assert const_ref.mass_at_erosion_change is not None



def testPowerLawWithMassIndex2():
    """ Check the power law of grain masses with the mass index of 2, which has its own formula. """

    assertSameAsReference(baseConstants(erosion_mass_index=2.0, erosion_bins_per_10mass=5, m_init=1e-6, \
        h_kill=92000.0))



def testFragmentationEntries():
    """ Check all types of fragmentation entries, an upward-only entry and the individual light curves. """

    const = tableConstants(baseConstants(m_init=1e-3, v_init=30000.0, h_kill=88000.0))
    const.fragmentation_on = True
    const.fragmentation_show_individual_lcs = True
    const.fragmentation_entries = [
        FragmentationEntry("M", 100000.0, None, None, 0.03e-6, None, 0.5e-6, 2e-11, 3e-10, 2.2),
        FragmentationEntry("A", 97000.0, None, None, 0.02e-6, 0.9, None, None, None, None),
        FragmentationEntry("F", 95000.0, 3, 20.0, 0.01e-6, None, None, None, None, None),
        FragmentationEntry("EF", 93000.0, 2, 30.0, None, None, 0.3e-6, 1e-10, 5e-10, None),
        FragmentationEntry("D", 90000.0, None, 20.0, None, None, None, 1e-10, 5e-10, 2.3),
        FragmentationEntry("F", "U80", 1, 10.0, None, None, None, None, None, None)]

    const_ref, _, _ = assertSameAsReference(const)

    assert all([entry.done for entry in const_ref.fragmentation_entries[:5]])
    assert len(const_ref.fragmentation_entries[2].fragments) == 3



def testWake():
    """ Check the wake at every time step and at the given heights. """

    const = baseConstants(m_init=1e-7, h_init=110000.0, h_kill=98000.0, dt=0.01, erosion_bins_per_10mass=2)

    _, _, n_wakes = assertSameAsReference(const, compute_wake=True)

    assert n_wakes > 10

    const.wake_heights = [105000.0, 100000.0, 99000.0]
    _, _, n_wakes = assertSameAsReference(const, compute_wake=True)

    assert 0 < n_wakes <= 3



def testSingleBodyWithLengthLimit():
    """ Check a single body without erosion which stops at a given length along the path, with the inputs as
        numpy scalars, as they are after the conversion to the ground.
    """

    const = baseConstants(erosion_on=False, m_init=1.0, v_init=np.float64(18000.5), \
        zenith_angle=np.arccos(np.float64(0.6)), h_kill=20000.0, len_kill=60000.0, lum_eff_type=7)

    _, frag_main, _ = assertSameAsReference(const)

    assert frag_main.length > const.len_kill



def testSubstepsOfFastGrains():
    """ Check fast meteoroids with a long time step, whose small grains are integrated in substeps. """

    const = baseConstants(v_init=65000.0, dt=0.02, erosion_mass_min=1e-12, h_kill=85000.0)

    assertSameAsReference(const)



def testLuminousEfficiencies():
    """ Check every type of the luminous efficiency, below and above 25.372 km/s where some of them change
        their formula.
    """

    for lum_eff_type in range(9):
        for v_init in [15000.0, 40000.0]:

            const = baseConstants(erosion_on=False, m_init=1e-4, v_init=v_init, lum_eff_type=lum_eff_type, \
                h_kill=85000.0)

            assertSameAsReference(const)
