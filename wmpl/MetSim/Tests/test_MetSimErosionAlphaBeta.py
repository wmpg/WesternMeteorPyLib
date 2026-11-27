""" Tests of MetSimErosionAlphaBeta, MetSim's erosion model from the alpha-beta closed forms, against MetSim
at a time step of 1 ms.

The leading fragment is compared while the main body keeps more than 1% of its mass, and the light in 30 fps
frames within 3 magnitudes of the peak. After that, grains and fragments that run out of mass lead and shine,
and there MetSim at 1 ms still changes with its step.

Run with pytest:
    python -m pytest wmpl/MetSim/Tests/test_MetSimErosionAlphaBeta.py -v
"""

from __future__ import print_function, division, absolute_import

import contextlib
import io

import numpy as np
import pytest

import wmpl.MetSim.MetSimErosion as MetSimErosion
from wmpl.MetSim.GUI import FragmentationEntry, SimulationResults
from wmpl.MetSim.MetSimErosion import Constants, atmDensity
from wmpl.MetSim.MetSimErosionAlphaBeta import runSimulation, nbDepositLight, nbLumEff, airDensity, \
    OUTPUT_STEP
from wmpl.MetSim.MetSimErosionCyTools import luminousEfficiency



def scenarioConstants(kind):
    """ Make the constants of a 20 mg meteoroid at 30 km/s, with erosion, a disruption or fragmentation
        entries.

    Arguments:
        kind: [str] 'single', 'erosion', 'disruption' or 'entries'.

    Return:
        const: [Constants object] Constants of the meteoroid.
    """

    # A 20 mg meteoroid at 30 km/s and 45 deg
    const = Constants()
    const.m_init = 2e-5
    const.h_init = 120000.0
    const.v_init = 30000.0
    const.zenith_angle = np.radians(45.0)
    const.rho = 1000.0
    const.sigma = 0.02/1e6

    # Erosion in two phases
    const.erosion_on = kind != 'single'
    const.erosion_height_start = 100000.0
    const.erosion_coeff = 0.1/1e6
    const.erosion_height_change = 88000.0
    const.erosion_coeff_change = 0.05/1e6
    const.erosion_mass_index = 2.0
    const.erosion_mass_min = 1e-8
    const.erosion_mass_max = 1e-7

    # Disruption near 90 km
    const.disruption_on = kind == 'disruption'
    const.compressive_strength = 3000.0
    const.disruption_mass_min_ratio = 1.0/100
    const.disruption_mass_max_ratio = 10.0/100
    const.disruption_mass_grain_ratio = 0.3

    # An eroding fragment, a change of all fragments and a dust release
    const.fragmentation_on = kind == 'entries'
    if kind == 'entries':
        const.fragmentation_entries = [ \
            FragmentationEntry('EF', 95000.0, 2, 30.0, None, None, 0.3e-6, 1e-8, 1e-7, 2.0), \
            FragmentationEntry('A', 90000.0, None, None, 0.01e-6, 0.8, None, None, None, None), \
            FragmentationEntry('D', 88000.0, None, 20.0, None, None, None, 1e-8, 1e-7, 2.0)]

    # Stop the run below 75 km or 3 km/s
    const.h_kill = 75000.0
    const.v_kill = 3000.0

    return const



def metSimResults(const, dt):
    """ Run MetSim at the given time step.

    Arguments:
        const: [Constants object] Simulation constants.
        dt: [float] Time step (s).

    Return:
        [ndarray] The 21 columns of MetSim's results.
    """

    # Run MetSim at the step without its printout
    const.dt = dt
    with contextlib.redirect_stdout(io.StringIO()):
        _, results_list, _ = MetSimErosion.runSimulation(const)

    return np.array([[np.nan if x is None else float(x) for x in row[:21]] for row in results_list])



def frameMagnitudes(t, lum, t_end, fps=30.0):
    """ Compute the magnitudes of the light averaged over the frames of a camera.

    Arguments:
        t: [ndarray] Times of the results (s).
        lum: [ndarray] Light at the times, the mean over the step ending at each (W).
        t_end: [float] End of the last frame (s).

    Keyword arguments:
        fps: [float] Frames per second. 30 by default.

    Return:
        [ndarray] Magnitude of every frame.
    """

    # Frame edges and the energy radiated up to every result
    edges = np.arange(t[0], t_end, 1.0/fps)
    energy = np.concatenate([[0.0], np.cumsum(lum[1:]*np.diff(t))])

    # Magnitude of the mean light of every frame
    with np.errstate(divide='ignore', invalid='ignore'):
        return -2.5*np.log10(np.diff(np.interp(edges, t, energy))*fps/840.0)



def compareResults(ref, res, const):
    """ Compare the results with reference results of MetSim.

    Arguments:
        ref: [ndarray] Reference results.
        res: [ndarray] Results to compare.
        const: [Constants object] Simulation constants.

    Return:
        (lead, light_median, light_p90):
            - lead: [float] Largest difference of the leading fragment while the main body keeps 1% of its
                mass (m).
            - light_median: [float] Median difference of the frames within 3 magnitudes of the peak (mag).
            - light_p90: [float] 90th percentile of the same differences (mag).
    """

    # Leading fragment while the main body keeps 1% of its mass
    t = ref[:, 0]
    body = (ref[:, 16] > 0.01*const.m_init) & (t <= res[-1, 0])
    lead = np.max(np.abs(np.interp(t[body], res[:, 0], res[:, 12]) - ref[body, 12]))

    # Frames above the kill height plus 1 km, within 3 magnitudes of the peak
    above = np.isfinite(ref[:, 11]) & (ref[:, 11] > const.h_kill + 1000.0)
    t_end = min(t[above][-1], res[-1, 0])
    mag_ref = frameMagnitudes(t, ref[:, 1], t_end)
    mag_res = frameMagnitudes(res[:, 0], res[:, 1], t_end)
    ok = np.isfinite(mag_ref) & np.isfinite(mag_res) & (mag_ref < np.nanmin(mag_ref) + 3)
    diff = np.abs(mag_res[ok] - mag_ref[ok])

    return lead, np.median(diff), np.percentile(diff, 90)



def checkAgainstMetSim(kind, lead_max, median_max, p90_max, make=None):
    """ Run a scenario with both accuracy settings and check it against MetSim at 1 ms.

    Arguments:
        kind: [str] Scenario, see scenarioConstants(), or a label when make is given.
        lead_max: [float] Largest difference of the leading fragment allowed (m).
        median_max: [float] Largest median difference of the frames allowed (mag).
        p90_max: [float] Largest 90th percentile of the frame differences allowed (mag).

    Keyword arguments:
        make: [function] Function that makes the constants. scenarioConstants(kind) by default.
    """

    # Make the constants of the scenario by default
    if make is None:
        make = lambda: scenarioConstants(kind)

    # MetSim at 1 ms as the reference
    ref = metSimResults(make(), 0.001)

    # Both accuracy settings against the reference
    for accuracy in ('fast', 'precise'):
        const = make()
        _, results, _ = runSimulation(const, accuracy=accuracy)
        lead, light_median, light_p90 = compareResults(ref, results, const)

        # Leading fragment and light within the limits
        assert lead < lead_max, "{:s} {:s}: leading fragment {:.2f} m off".format(kind, accuracy, lead)
        assert light_median < median_max, "{:s} {:s}: light {:.4f} mag off".format(kind, accuracy, \
            light_median)
        assert light_p90 < p90_max, "{:s} {:s}: light p90 {:.4f} mag off".format(kind, accuracy, light_p90)



def testSingleBodyMatchesMetSim():
    """ A single body without erosion follows MetSim at 1 ms. """

    checkAgainstMetSim('single', 0.5, 0.002, 0.005)



def testErodingBodyMatchesMetSim():
    """ A body eroding in two phases follows MetSim at 1 ms. """

    checkAgainstMetSim('erosion', 2.0, 0.01, 0.03)



def testDisruptionMatchesMetSim():
    """ A disruption follows MetSim at 1 ms. The frames after it fade as the fragments run out of mass, where
        MetSim at 1 ms is still off its converged light by a tenth of a magnitude.
    """

    checkAgainstMetSim('disruption', 0.5, 0.03, 0.2)



def testFragmentationEntriesMatchMetSim():
    """ An eroding fragment, a change of all fragments and a dust release follow MetSim at 1 ms. """

    checkAgainstMetSim('entries', 1.0, 0.02, 0.04)



def dynestyConstants():
    """ Make the constants of the eroding meteoroid with what DynestyMetSim sets: the luminous efficiency of
        type 5, a tabulated atmosphere, and an M and an F fragmentation entry.

    Return:
        const: [Constants object] Constants of the meteoroid.
    """

    # The eroding meteoroid with DynestyMetSim's luminous efficiency
    const = scenarioConstants('erosion')
    const.lum_eff_type = 5

    # An exponential atmosphere as a table
    ht = np.arange(50000.0, 200001.0, 1000.0)
    const.atm_density_model = 'table'
    const.atm_table_ht = list(ht)
    const.atm_table_log10_rho = list(np.log10(1.2*np.exp(-ht/7200.0)))

    # New sigma and grains from 96 km, and two fragments at 92 km
    const.fragmentation_on = True
    const.fragmentation_entries = [ \
        FragmentationEntry('M', 96000.0, None, None, 0.03e-6, None, None, 1e-9, 1e-7, 2.2), \
        FragmentationEntry('F', 92000.0, 2, 20.0, None, None, None, None, None, None)]

    return const



def testDynestySettingsMatchMetSim():
    """ The eroding meteoroid with DynestyMetSim's luminous efficiency, a density table and M and F entries
        follows MetSim at 1 ms.
    """

    checkAgainstMetSim('dynesty settings', 2.0, 0.01, 0.03, make=dynestyConstants)



def slowConstants():
    """ Make the constants of a slow meteoroid at 70 deg whose main body erodes away at 92 km, so that its
        grains lead for seconds.

    Return:
        const: [Constants object] Constants of the meteoroid.
    """

    # A slow meteoroid at 70 deg from 180 km
    const = Constants()
    const.h_init = 180000.0
    const.zenith_angle = np.radians(70.0)
    const.v_init = 12000.0
    const.m_init = 1e-5
    const.rho = 1000.0
    const.sigma = 0.005e-6

    # Erosion strong enough to erode the main body away
    const.erosion_on = True
    const.erosion_height_start = 105000.0
    const.erosion_coeff = 1e-6
    const.erosion_height_change = 90000.0
    const.erosion_coeff_change = 1e-6
    const.erosion_mass_min = 1e-9
    const.erosion_mass_max = 1e-7

    # Stop the run below 60 km or 3 km/s
    const.h_kill = 60000.0
    const.v_kill = 3000.0

    return const



def testGrainsFallAfterTheMainBody():
    """ The grains that lead after the main body has eroded away keep the drop of gravity of MetSim's grains,
        within 15 m in height and 10 m in length 2 s later. MetSim at 5 ms agrees with 1 ms to 1 m there.
    """

    # MetSim at 5 ms and the engine for the slow meteoroid
    ref = metSimResults(slowConstants(), 0.005)
    _, results, _ = runSimulation(slowConstants())

    # End of the main body in MetSim
    t_main = ref[np.where(ref[:, 16] > 0)[0][-1], 0]

    # Height and length 1 and 2 s after the main body ends
    for dt_after in (1.0, 2.0):
        t = t_main + dt_after
        d_h = np.interp(t, results[:, 0], results[:, 11]) - np.interp(t, ref[:, 0], ref[:, 11])
        d_l = np.interp(t, results[:, 0], results[:, 12]) - np.interp(t, ref[:, 0], ref[:, 12])
        assert abs(d_h) < 15.0, "{:.0f} s after the main body: height {:.1f} m off".format(dt_after, d_h)
        assert abs(d_l) < 10.0, "{:.0f} s after the main body: length {:.1f} m off".format(dt_after, d_l)



def ironConstants():
    """ Make the constants of an iron-like meteoroid whose grains follow the gamma distribution of their
        diameters.

    Return:
        const: [Constants object] Constants of the meteoroid.
    """

    # Start from the eroding meteoroid
    const = scenarioConstants('erosion')

    # A dense body at 20 km/s with dense grains
    const.v_init = 20000.0
    const.m_init = 1e-5
    const.rho = 6000.0
    const.rho_grain = 4000.0
    const.sigma = 0.05e-6
    const.lum_eff = 0.3

    # Faster erosion of smaller grains
    const.erosion_height_start = 105000.0
    const.erosion_coeff = 5e-6
    const.erosion_height_change = 90000.0
    const.erosion_coeff_change = 2e-6
    const.erosion_mass_min = 1e-11
    const.erosion_mass_max = 1e-9

    # Gamma distribution of the grain diameters
    const.erosion_grain_distribution = 'gamma'

    return const



def testGammaGrainsMatchMetSim():
    """ The gamma distribution of the grain diameters follows MetSim at 1 ms. """

    checkAgainstMetSim('gamma grains', 2.0, 0.01, 0.03, make=ironConstants)



def entryConstants(entries, sigma=0.02e-6):
    """ Make the constants of the meteoroid without erosion and with fragmentation entries.

    Arguments:
        entries: [list] Fragmentation entries.

    Keyword arguments:
        sigma: [float] Ablation coefficient (s^2/m^2). 0.02e-6 by default.

    Return:
        const: [Constants object] Constants of the meteoroid.
    """

    # A heavier single body with the given entries
    const = scenarioConstants('single')
    const.m_init = 1e-4
    const.sigma = sigma
    const.fragmentation_on = True
    const.fragmentation_entries = entries

    return const



def testEntriesAtOneHeightMatchMetSim():
    """ An "A" entry listed after an "F" entry at the same height, and two "A" entries at one height, apply to
        all the fragments as in MetSim at 1 ms.
    """

    # An A entry after an F entry at the same height
    checkAgainstMetSim('F then A', 2.0, 0.01, 0.05, make=lambda: entryConstants([ \
        FragmentationEntry('F', 95000.0, 2, 40.0, 0.005e-6, None, None, None, None, None), \
        FragmentationEntry('A', 95000.0, None, None, 0.08e-6, 2.0, None, None, None, None)]))

    # Two A entries at one height after an F entry
    checkAgainstMetSim('two A', 2.0, 0.01, 0.05, make=lambda: entryConstants([ \
        FragmentationEntry('F', 100000.0, 2, 40.0, 0.005e-6, None, None, None, None, None), \
        FragmentationEntry('A', 92000.0, None, None, 0.06e-6, 1.0, None, None, None, None), \
        FragmentationEntry('A', 92000.0, None, None, None, 2.0, None, None, None, None)]))



def testEdgeCasesRun():
    """ An entry that takes all of the main body's mass, an "F" entry without fragments and a zero ablation
        coefficient run, and a kill length stops the run.
    """

    # Each edge case runs and gives some finite light
    for label, const in ( \
        ('EF of all the mass', entryConstants([ \
            FragmentationEntry('EF', 90000.0, 1, 100.0, None, None, 0.3e-6, 1e-8, 1e-7, 2.0)])), \
        ('F without fragments', entryConstants([ \
            FragmentationEntry('F', 90000.0, 0, 30.0, None, None, None, None, None, None)])), \
        ('zero sigma', entryConstants([], sigma=0.0))):

        _, results, _ = runSimulation(const)
        assert np.all(np.isfinite(results[:, 1])), "{:s}: the light is not finite".format(label)
        assert np.max(results[:, 1]) > 0, "{:s}: no light".format(label)

    # The kill length
    const = scenarioConstants('single')
    const.len_kill = 40000.0
    _, results, _ = runSimulation(const)
    assert np.nanmax(results[:, 12]) < 40000.0 + 100.0



def testGravityOffMatchesMetSim():
    """ The engine takes the gravity of MetSimErosion.G0, so that both turn it off together. """

    # Turn gravity off in both models and restore it afterwards
    g0 = MetSimErosion.G0
    MetSimErosion.G0 = 0.0
    try:
        ref = metSimResults(scenarioConstants('single'), 0.001)
        const = scenarioConstants('single')
        _, results, _ = runSimulation(const, accuracy='precise')
        lead, light_median, _ = compareResults(ref, results, const)

    finally:
        MetSimErosion.G0 = g0

    # Leading fragment and light as in MetSim without gravity
    assert lead < 0.1
    assert light_median < 0.002



def testResultsLoadAsMetSimResults():
    """ The results load into SimulationResults, every OUTPUT_STEP, with the time step of MetSim kept. """

    # Run the eroding meteoroid and load its results
    const = scenarioConstants('erosion')
    dt = const.dt
    frag_main, results, wake = runSimulation(const)
    sim = SimulationResults(const, frag_main, results, wake)

    # Columns, time step and spacing of the results
    assert results.shape[1] == 21
    assert const.dt == dt
    assert np.allclose(np.diff(sim.time_arr), OUTPUT_STEP)

    # The leading fragment falls while the main body flies
    body = sim.main_mass_arr > 0
    assert np.all(np.diff(sim.leading_frag_height_arr[body]) < 0)



def testLightDepositsConserveEnergy():
    """ The light of grain and body intervals sums exactly to their energy in the bins of the grid. """

    # Random intervals with energies and spreads
    rng = np.random.default_rng(1)
    n = 500
    t_a = rng.uniform(0.1, 0.8, n)
    spread = rng.uniform(0.0, 0.05, n)
    t_b = t_a + spread + rng.uniform(0.0, 0.05, n)
    energy = rng.uniform(0.0, 1.0, n)
    q = rng.uniform(-0.5, 0.5, n)

    # The last fifth as intervals of bodies, with no spread
    spread[400:] = 0.0
    q[400:] = 1.0
    bins = nbDepositLight(t_a, t_b, energy, spread, q, 0.0, 0.001, 1001)

    # All of the energy in the bins and none negative
    assert abs(np.sum(bins) - np.sum(energy)) < 1e-9*np.sum(energy)
    assert np.min(bins) > -1e-12



def testClimbingPathIsRefused():
    """ A grazing path that climbs back up before the kill height is refused, as the engine does not follow
        it back up.
    """

    # The meteoroid at 87 deg, whose lowest point is above the kill height
    const = scenarioConstants('erosion')
    const.zenith_angle = np.radians(87.0)

    with pytest.raises(NotImplementedError):
        runSimulation(const)



def testLuminousEfficiencyAndAirDensityMatchMetSim():
    """ The luminous efficiency of every type and the air density, of the polynomial and of a table, are
        MetSim's.
    """

    # Every type of the luminous efficiency over the velocities and masses of meteors
    for lum_eff_type in range(9):
        for vel in (11000.0, 20000.0, 25371.0, 25373.0, 40000.0, 72000.0):
            for mass in (1e-12, 1e-6, 1e-2, 10.0):
                ref = luminousEfficiency(lum_eff_type, 0.7, vel, mass)
                tau = nbLumEff(lum_eff_type, 0.7, vel, np.log(vel/1000.0), np.log(mass))
                assert abs(tau - ref) <= 1e-12*abs(ref), "type {:d}, {:.0f} m/s, {:.0e} kg".format( \
                    lum_eff_type, vel, mass)

    # The density polynomial and a table between their heights
    ht = np.linspace(60000.0, 180000.0, 97)
    for const in (scenarioConstants('single'), dynestyConstants()):
        ref = np.array([atmDensity(h, const) for h in ht])
        assert np.allclose(airDensity(const, ht), ref, rtol=1e-12, atol=0)
