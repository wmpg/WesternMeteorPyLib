""" Tests for the options of the run back through the atmosphere (wmpl.MetSim.BackwardAtmIntegration): the drawn densities
and Gamma*A, the length of the step, the erosion of the body at each height, what the constants leave out, and the
runs and options it refuses.

Run under pytest, or directly:

    python -m pytest wmpl/MetSim/Tests/test_BackwardAtmOptions.py
"""

import argparse
import copy
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from wmpl.MetSim.BackwardAtmIntegration import BACKWARD_GA, BACKWARD_MAX_DT, _eciToEcef, addBackwardArguments, \
    backwardConstants, backwardState, backwardStates, backwardStatesFromArguments, checkBackHeight, \
    checkBackwardArguments, depthWarnings, dragSpeedGain, groundSpeed, unusedBackwardArguments, DRAW_UNKNOWNS_DENSITY, \
    DRAW_UNKNOWNS_GA, DRAW_UNKNOWNS_MASS_SIGMA, DRAW_UNKNOWNS_SIGMA
from wmpl.MetSim.MetSimErosion import Constants, EARTH_ROTATION_RATE, runSimulation
from wmpl.Utils.AtmosphereDensity import getAtmDensityTable, setAtmosphere
from wmpl.Utils.TrajConversions import date2JD, enu2ECEF, geo2Cartesian


JD = 2460600.25
LAT, LON = np.radians(45.0), np.radians(10.0)


def _stateVector(ht, v_ground, zenith_deg=45.0, azim_deg=90.0, jd=JD, lat=LAT, lon=LON):
    """ ECI state vector, with the velocity to the radiant, of a meteor at lat, lon and ht moving at v_ground relative to
        the ground. """

    pos = np.array(geo2Cartesian(lat, lon, ht, jd))
    azim, elev = np.radians(azim_deg), np.radians(90.0 - zenith_deg)
    rad_enu = [np.cos(elev)*np.sin(azim), np.cos(elev)*np.cos(azim), np.sin(elev)]
    rad_eci = _eciToEcef(jd).T.dot(np.array(enu2ECEF(lat, lon, *rad_enu)))

    return np.concatenate([pos, v_ground*rad_eci - np.cross([0.0, 0.0, EARTH_ROTATION_RATE], pos)])


def _noErosion():
    const = Constants()
    const.erosion_on = const.disruption_on = const.fragmentation_on = False

    return const


def _parser():
    parser = argparse.ArgumentParser()
    addBackwardArguments(parser)

    return parser


def _traj():
    return SimpleNamespace(jdt_ref=JD, v_init_stddev=10.0, uncertainties=None)


def test_gamma_a_defaults_to_the_value_of_this_high_regime():
    """ --ga is 1.21 by default here, on purpose not the 0.55 of DynamicMassFit and AlphaBeta. """

    assert BACKWARD_GA == 1.21
    assert _parser().parse_args(["--mass", "1"]).ga == 1.21


def test_density_and_gamma_a_are_drawn_for_each_realization():
    """ With --density_sigma and --ga_sigma each realization runs with its own values, log-normal around --density and
        --ga. The nominal solution keeps them, the masses and coefficients drawn stay the same, and each realization
        ends as a run with its own values does. """

    sv = _stateVector(70e3, 20000.0)
    state_vects = [sv]*4
    base = ["--mass", "1.0", "--mass_sigma", "0.2", "--ablation_coeff", "0.1", "--ablation_coeff_sigma", "0.02"]

    (_, _, masses_fixed), m_fixed, s_fixed, rhos_fixed, gas_fixed, *_ = backwardStatesFromArguments(_traj(), state_vects,
        _parser().parse_args(base), 180000.0, random_seed=5)
    assert rhos_fixed == [3000.0]*4 and gas_fixed == [1.21]*4

    args = _parser().parse_args(base + ["--density_sigma", "1000", "--ga_sigma", "0.3"])
    (_, states, masses), m_inits, sigmas, rhos, gas, *_ = backwardStatesFromArguments(_traj(), state_vects, args, 180000.0,
        random_seed=5)

    assert rhos[0] == 3000.0 and gas[0] == 1.21 and np.all(np.array(rhos) > 0) and np.all(np.array(gas) > 0)
    assert len(set(rhos)) == 4 and len(set(gas)) == 4
    assert m_inits == m_fixed and sigmas == s_fixed

    # Independent generators: with one of them on, the other does not move
    args_rho = _parser().parse_args(base + ["--density_sigma", "1000"])
    _, _, _, rhos_only, gas_only, *_ = backwardStatesFromArguments(_traj(), state_vects, args_rho, 180000.0, random_seed=5)
    assert rhos_only == rhos and gas_only == gas_fixed

    # A realization ends as a run of the engine with its own values does, for as long as the nominal one
    t = runSimulation(_engineRun(sv, m_inits[0], sigmas[0], rhos[0], gas[0]))[1][-1][0]
    for i in range(1, 4):
        const = _engineRun(sv, m_inits[i], sigmas[i], rhos[i], gas[i])
        const.h_kill, const.t_kill = np.inf, abs(t)
        frag, results, _ = runSimulation(const)
        assert results[-1][0] == t
        assert np.linalg.norm(backwardState(JD, sv, frag, t)[1] - states[i]) < 1e-6
        assert np.isclose(frag.m, masses[i], rtol=1e-12, atol=0)

    # They change the result, which is the point of drawing them
    assert len(set(round(np.linalg.norm(s[3:]), 3) for s in states)) == 4


def _engineRun(sv, m, sigma, rho, ga):
    """ The constants of a run back with these values (sigma in s^2/km^2). """

    const = _noErosion()
    const.rho, const.sigma = rho, sigma/1e6
    const.gamma = ga/const.shape_factor

    return backwardConstants(JD, sv, m, const=const)


def test_negative_spreads_and_sizes_are_command_line_errors():

    parser = _parser()
    for argv in [["--density_sigma", "-1"], ["--ga_sigma", "-0.1"], ["--density", "0"], ["--ga", "0"]]:
        with pytest.raises(SystemExit):
            checkBackwardArguments(parser, parser.parse_args(["--mass", "1"] + argv))


def test_the_step_is_at_most_one_millisecond():
    """ The default step of a run back is 1 ms, not MetSim's 5 ms: a finer one given in the constants is kept and a
        coarser one is not. """

    sv = _stateVector(70e3, 20000.0)

    assert BACKWARD_MAX_DT == 0.001
    assert backwardConstants(JD, sv, 1.0).dt == -0.001

    const = _noErosion()
    assert const.dt == 0.005 and backwardConstants(JD, sv, 1.0, const=const).dt == -0.001

    const.dt = 0.0002
    assert backwardConstants(JD, sv, 1.0, const=const).dt == -0.0002


def test_the_default_step_leaves_the_speed_within_two_metres_per_second_from_a_deep_start():
    """ From 50 km, 1 kg at 20 km/s, with the ablation coefficient that grows it most in this test. The step error is
        of first order, so a step 10 times finer gives the error to about 10%. MetSim's 5 ms leaves about 5 m/s. """

    sv = _stateVector(50e3, 20000.0)

    def run(dt):
        const = _noErosion()
        const.rho, const.sigma, const.dt = 3000.0, 0.323/1e6, dt
        c = backwardConstants(JD, sv, 1.0, const=const)
        c.dt = -dt
        frag, _, _ = runSimulation(c)
        return np.sqrt(frag.vx**2 + frag.vy**2 + frag.vz**2), frag.m

    v_fine, m_fine = run(0.0001)
    v_default, m_default = run(BACKWARD_MAX_DT)
    v_metsim, _ = run(0.005)

    assert abs(v_default - v_fine) < 2.0 and abs(m_default/m_fine - 1) < 0.003
    assert abs(v_metsim - v_fine) > 3.0


def _erodedBody(erosion_height_change=-1.0):
    """ A body run forwards through the real engine from 120 km for 3.6 s, eroding from 90 km, with its grains. Returns
        the constants, where it started, the body and the time it ended at. """

    sv_top = _stateVector(120e3, 20000.0)
    const = _noErosion()
    const.rho, const.sigma, const.m_init = 3000.0, 0.02/1e6, 1.0
    c = backwardConstants(JD, sv_top, 1.0, h_kill=125e3, const=const)
    c.atm_table_ht, c.atm_table_log10_rho = getAtmDensityTable(LAT, LON, 30e3, 125e3, JD)
    c.dens_co = const.dens_co
    c.dt, c.h_kill, c.v_kill = 0.0005, 20e3, 100.0
    c.t_kill = 3.6
    c.erosion_on, c.erosion_body_only = True, False
    c.erosion_height_start, c.erosion_coeff = 90e3, 0.3/1e6
    c.erosion_height_change, c.erosion_coeff_change = erosion_height_change, 0.6/1e6
    c.erosion_rho_change, c.erosion_sigma_change = 1500.0, 0.04/1e6
    c.erosion_mass_min, c.erosion_mass_max = 1e-4, 1e-3
    frag, results, _ = runSimulation(c)

    return c, sv_top, frag, results[-1][0]


def _runBack(c, sv_top, frag, t, sigma=None, body_erosion=False):

    jd_end, state_end = backwardState(JD, sv_top, frag, t)

    const = copy.deepcopy(c)
    const.erosion_on = body_erosion
    const.t_kill = -1
    if sigma is not None:
        const.sigma = sigma

    # At the same step as forwards, and in the same density table
    const.dt = 0.0005
    cb = backwardConstants(jd_end, state_end, frag.m, h_kill=120e3, const=const, body_erosion=body_erosion)
    cb.atm_table_ht, cb.atm_table_log10_rho = c.atm_table_ht, c.atm_table_log10_rho
    cb.t_kill = abs(t)
    frag_back, results_back, _ = runSimulation(cb)
    jd, state = backwardState(jd_end, state_end, frag_back, results_back[-1][0])

    return frag_back, groundSpeed(jd, state), cb


def test_body_erosion_undoes_the_erosion_of_the_engine_at_each_height():
    """ A body that eroded in the real engine, with its grains, from an onset above which it did not, comes back to its
        starting mass and speed with body_erosion. A single effective coefficient does not know the onset: it gets the
        speed as well but overestimates the mass, and sigma alone gets neither (0.58 kg, 10 m/s fast). """

    c, sv_top, frag, t = _erodedBody()
    v_top = c.v_init

    assert frag.h < 70e3 and 0.3 < frag.m < 0.8

    frag_back, v_back, cb = _runBack(c, sv_top, frag, t, body_erosion=True)
    assert cb.erosion_on and cb.erosion_body_only
    assert abs(frag_back.m - 1) < 3e-3 and abs(v_back - v_top) < 0.5

    # No grains were released or followed
    assert cb.total_fragments == 1 and frag_back.erosion_coeff == 0

    frag_eff, v_eff, _ = _runBack(c, sv_top, frag, t, sigma=(0.02 + 0.3)/1e6)
    assert (frag_eff.m - 1) > 5*abs(frag_back.m - 1) and abs(v_eff - v_top) < 0.5

    frag_sigma, v_sigma, _ = _runBack(c, sv_top, frag, t)
    assert frag_sigma.m < 0.9*frag_back.m and (v_sigma - v_top) > abs(v_back - v_top)


def test_body_erosion_restores_the_density_and_coefficient_above_their_change():
    """ A forward run changes the density and the coefficients once below erosion_height_change. Going back they are
        those of the height at each step, from a start already below it. """

    c, sv_top, frag, t = _erodedBody(erosion_height_change=80e3)
    assert frag.h < 80e3

    frag_back, v_back, cb = _runBack(c, sv_top, frag, t, body_erosion=True)

    assert abs(frag_back.m - 1) < 5e-3 and abs(v_back - c.v_init) < 1.0
    assert frag_back.rho == c.rho and frag_back.sigma == c.sigma


def test_erosion_is_refused_in_a_backward_run_unless_it_is_for_the_body_alone():

    const = backwardConstants(JD, _stateVector(70e3, 20000.0), 1.0)
    const.erosion_on = True
    assert not const.erosion_body_only
    with pytest.raises(ValueError):
        runSimulation(const)


def test_constants_warn_of_what_the_run_leaves_out():
    """ A MetSim fit given as constants has its erosion, disruption or fragmentation said to be left out, instead of
        dropped silently; nothing is said when they are off, or for the erosion with body_erosion. """

    sv = _stateVector(70e3, 20000.0)
    const = _noErosion()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        backwardConstants(JD, sv, 1.0)
        backwardConstants(JD, sv, 1.0, const=const)

    const.erosion_on, const.erosion_coeff, const.erosion_height_start = True, 0.3/1e6, 100e3
    with pytest.warns(UserWarning, match="eta = 0.3 s\\^2/km\\^2 from 100.0 km"):
        assert not backwardConstants(JD, sv, 1.0, const=const).erosion_on

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert backwardConstants(JD, sv, 1.0, const=const, body_erosion=True).erosion_on

    const.fragmentation_on = True
    with pytest.warns(UserWarning, match="the fragmentation"):
        assert not backwardConstants(JD, sv, 1.0, const=const, body_erosion=True).fragmentation_on


def test_command_line_constants_do_not_warn():

    args = _parser().parse_args(["--mass", "1.0"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        backwardStatesFromArguments(_traj(), [_stateVector(70e3, 20000.0)], args, 180000.0)


def test_constants_without_the_options_of_the_run_back_run_as_before():
    """ Constants saved before the run back was added, without t_kill, freeze_mass and erosion_body_only,
        still run forwards, and give the same simulation. """

    const = _noErosion()
    frag, results, _ = runSimulation(copy.deepcopy(const))

    for name in ("t_kill", "freeze_mass", "erosion_body_only"):
        delattr(const, name)

    frag_old, results_old, _ = runSimulation(const)

    assert len(results_old) == len(results)
    assert np.array_equal(np.array(results_old, dtype=float), np.array(results, dtype=float), equal_nan=True)


def test_a_run_that_stops_before_its_height_is_an_error():
    """ MetSim stops under v_kill, 3 km/s, after one step: that used to give an end 5 ms from the start. A lower
        v_kill lets it through. """

    sv = _stateVector(30e3, 2900.0)
    with pytest.raises(RuntimeError, match="before h_kill"):
        backwardStates(JD, [sv], 1.0, const=_noErosion())

    const = _noErosion()
    const.v_kill = 100.0
    backwardStates(JD, [sv], 1.0, const=const)


def test_a_height_not_above_the_start_is_a_command_line_error():

    parser = _parser()
    sv = _stateVector(116e3, 60000.0)
    with pytest.raises(SystemExit):
        checkBackHeight(parser, JD, sv, 100e3)
    checkBackHeight(parser, JD, sv, 180e3)


def test_options_of_the_run_back_are_reported_when_given_without_it():
    """ REBOUND refuses these without --back_height instead of ignoring them. --mass and --density also set its
        radiation pressure, so they are not among them. """

    parser = _parser()

    assert unusedBackwardArguments(parser, parser.parse_args(["--mass", "1", "--density", "2000"])) == []

    used = unusedBackwardArguments(parser, parser.parse_args(["--ablation_coeff", "0.1", "--ga", "0.5", "--ga_sigma",
        "0.1", "--density_sigma", "100", "--mass_sigma", "0.1", "--freeze_mass", "--atm", "00", "--atmsw",
        "constant", "--erosion_coeff", "0.3", "--erosion_height", "90"]))
    assert set(used) == {"--ablation_coeff", "--ga", "--ga_sigma", "--density_sigma", "--mass_sigma", "--freeze_mass",
        "--atm", "--atmsw", "--erosion_coeff", "--erosion_height"}


def _erosionConstants(sigma, eta, height, rho=3000.0, ga=1.21):
    """ Constants of a MetSim fit with this erosion, as the command line sets them from its options. """

    const = _noErosion()
    const.rho, const.sigma, const.erosion_on = rho, sigma/1e6, True
    const.gamma = ga/const.shape_factor
    const.erosion_coeff, const.erosion_height_start, const.erosion_height_change = eta/1e6, 1000*height, -1.0

    return const


def test_erosion_options_erode_the_body_below_their_height(capsys):
    """ --erosion_coeff and --erosion_height give the same run as the erosion of constants with body_erosion: more mass
        than the ablation coefficient alone, and less than the same erosion all the way up. """

    sv = _stateVector(60e3, 20000.0)
    argv = ["--mass", "1.0", "--ablation_coeff", "0.02"]

    (_, states, masses), *_ = backwardStatesFromArguments(_traj(), [sv], _parser().parse_args(argv + ["--erosion_coeff",
        "0.3", "--erosion_height", "75"]), 180000.0)
    out = capsys.readouterr().out
    assert "The body erodes with 0.3 s^2/km^2 below 75 km" in out and "Ceplecha" not in out

    _, states_direct, masses_direct = backwardStates(JD, [sv], 1.0, const=_erosionConstants(0.02, 0.3, 75.0),
        body_erosion=True)
    assert np.linalg.norm(states[0] - states_direct[0]) < 1e-6 and masses[0] == masses_direct[0]

    (_, _, masses_sigma), *_ = backwardStatesFromArguments(_traj(), [sv], _parser().parse_args(argv), 180000.0)
    (_, _, masses_all), *_ = backwardStatesFromArguments(_traj(), [sv], _parser().parse_args(["--mass", "1.0",
        "--ablation_coeff", "0.32"]), 180000.0)
    assert masses_sigma[0] < masses[0] < masses_all[0]


def test_erosion_starting_below_the_reference_point_does_not_act(capsys):

    sv = _stateVector(90e3, 20000.0)
    (_, states, masses), *_ = backwardStatesFromArguments(_traj(), [sv], _parser().parse_args(["--mass", "1.0",
        "--ablation_coeff", "0.02", "--erosion_coeff", "0.3", "--erosion_height", "80"]), 180000.0)
    assert "so the body does not erode in this run" in capsys.readouterr().out

    (_, states_none, masses_none), *_ = backwardStatesFromArguments(_traj(), [sv], _parser().parse_args(["--mass", "1.0",
        "--ablation_coeff", "0.02"]), 180000.0)
    assert np.linalg.norm(states[0] - states_none[0]) < 1e-6 and masses[0] == masses_none[0]


def test_realizations_erode_as_the_nominal_solution_does():

    sv = _stateVector(60e3, 20000.0)
    args = _parser().parse_args(["--mass", "1.0", "--mass_sigma", "0.2", "--ablation_coeff", "0.02", "--erosion_coeff",
        "0.3", "--erosion_height", "75"])
    (_, _, masses), m_inits, *_ = backwardStatesFromArguments(_traj(), [sv]*3, args, 180000.0, random_seed=5)

    # Starting with more mass, a body grows more in absolute terms, and with its erosion on it grows far more than 1%
    assert np.all(np.array(masses)/np.array(m_inits) > 1.5)


def test_erosion_options_that_do_not_go_together_are_command_line_errors():

    parser = _parser()
    for argv in [["--erosion_coeff", "0.3"], ["--erosion_height", "90"], ["--erosion_coeff", "-0.3", "--erosion_height",
            "90"], ["--erosion_coeff", "0.3", "--erosion_height", "0"], ["--erosion_coeff", "0.3", "--erosion_height",
            "90", "--freeze_mass"]]:
        with pytest.raises(SystemExit):
            checkBackwardArguments(parser, parser.parse_args(["--mass", "1"] + argv))

    checkBackwardArguments(parser, parser.parse_args(["--mass", "1", "--erosion_coeff", "0.3", "--erosion_height", "90"]))


def test_the_speed_the_drag_adds_matches_a_run_without_drag():
    """ The speed gravity alone gives comes from the energy: a run of the engine without drag ends within 0.5 m/s of it
        at 180 km. The gain is positive, and larger from deeper. """

    gains = []
    for ht in (90e3, 70e3, 55e3):
        sv = _stateVector(ht, 20000.0)
        const = _noErosion()
        const.rho = 3000.0
        jd, (end,), _ = backwardStates(JD, [sv], 1.0, const=const)
        gain = dragSpeedGain(JD, sv, jd, end)
        gains.append(gain)

        no_drag = _noErosion()
        no_drag.rho, no_drag.gamma = 3000.0, 0.0
        jd_free, (end_free,), _ = backwardStates(JD, [sv], 1.0, const=no_drag)
        assert abs(dragSpeedGain(JD, sv, jd_free, end_free)) < 0.5
        assert abs(gain - (groundSpeed(jd, end) - groundSpeed(jd_free, end_free))) < 0.5

    assert 0 < gains[0] < gains[1] < gains[2]


def test_depth_warnings_follow_the_speed_gained_and_the_pressure():
    """ The speed the drag adds warns against the uncertainty of the initial velocity, or 10 m/s without one, whatever
        the pressure; the pressure over 0.04 MPa warns on its own, about the mass; none says the coefficient dominates. """

    assert depthWarnings(0.001e6, 5.0) == []
    assert depthWarnings(0.001e6, 15.0, v_init_stddev=30.0) == []

    weak = depthWarnings(0.001e6, 15.0)
    assert len(weak) == 1 and "adds 15.0 m/s" in weak[0] and "10.0 m/s" in weak[0] and "weaker than a chondrite" in weak[0]
    assert "the 30.00 m/s uncertainty" in depthWarnings(0.001e6, 40.0, v_init_stddev=30.0)[0]

    only_pressure = depthWarnings(0.05e6, 1.0)
    assert len(only_pressure) == 1 and "over the 0.04 MPa" in only_pressure[0] and "lower limit" in only_pressure[0]

    both = depthWarnings(0.1e6, 300.0)
    assert len(both) == 2 and "weaker than a chondrite" not in both[0]

    assert not any("dominates" in text for text in weak + only_pressure + both)


def test_erosion_is_drawn_for_each_realization():
    """ With --erosion_coeff_sigma and --erosion_height_sigma each realization runs with its own values, log-normal
        around the given ones. The nominal solution keeps them, the other draws stay the same, and each realization ends
        as a run of the engine with its own values does. Without erosion, the values are 0 and NaN. """

    sv = _stateVector(60e3, 20000.0)
    state_vects = [sv]*4
    base = ["--mass", "1.0", "--mass_sigma", "0.2", "--ablation_coeff", "0.02", "--density_sigma", "500",
        "--erosion_coeff", "0.3", "--erosion_height", "75"]

    (_, _, _), m_fixed, s_fixed, rhos_fixed, gas_fixed, etas_fixed, hts_fixed = backwardStatesFromArguments(_traj(),
        state_vects, _parser().parse_args(base), 180000.0, random_seed=5)
    assert etas_fixed == [0.3]*4 and hts_fixed == [75.0]*4

    args = _parser().parse_args(base + ["--erosion_coeff_sigma", "0.1", "--erosion_height_sigma", "5"])
    (_, states, masses), m_inits, sigmas, rhos, gas, etas, hts = backwardStatesFromArguments(_traj(), state_vects, args,
        180000.0, random_seed=5)

    assert etas[0] == 0.3 and hts[0] == 75.0 and len(set(etas)) == 4 and len(set(hts)) == 4
    assert np.all(np.array(etas) > 0) and np.all(np.array(hts) > 0)
    assert m_inits == m_fixed and sigmas == s_fixed and rhos == rhos_fixed and gas == gas_fixed

    # A realization ends as a run of the engine with its own values does, for as long as the nominal one
    def engine(i):
        const = _erosionConstants(sigmas[i], etas[i], hts[i], rho=rhos[i], ga=gas[i])
        return backwardConstants(JD, sv, m_inits[i], const=const, body_erosion=True)

    t = runSimulation(engine(0))[1][-1][0]
    for i in range(1, 4):
        const = engine(i)
        const.h_kill, const.t_kill = np.inf, abs(t)
        frag, results, _ = runSimulation(const)
        assert results[-1][0] == t
        assert np.linalg.norm(backwardState(JD, sv, frag, t)[1] - states[i]) < 1e-6
        assert np.isclose(frag.m, masses[i], rtol=1e-12, atol=0)

    *_, etas_none, hts_none = backwardStatesFromArguments(_traj(), [sv]*2, _parser().parse_args(["--mass", "1.0"]),
        180000.0, random_seed=5)
    assert etas_none == [0.0]*2 and np.all(np.isnan(hts_none))


def test_erosion_uncertainties_need_the_erosion():

    parser = _parser()
    for argv in [["--erosion_coeff_sigma", "0.1"], ["--erosion_height_sigma", "5"], ["--erosion_coeff", "0.3",
            "--erosion_height", "90", "--erosion_coeff_sigma", "-0.1"], ["--erosion_coeff", "0.3", "--erosion_height",
            "90", "--erosion_height_sigma", "-5"]]:
        with pytest.raises(SystemExit):
            checkBackwardArguments(parser, parser.parse_args(["--mass", "1"] + argv))

    assert set(unusedBackwardArguments(parser, parser.parse_args(["--erosion_coeff_sigma", "0.1",
        "--erosion_height_sigma", "5"]))) == {"--erosion_coeff_sigma", "--erosion_height_sigma"}


def test_back_time_stops_the_run_and_is_not_an_option_to_refuse_without_a_height():
    """ --back_time is shared by both command lines: it runs back for that long unless the height comes first, must be
        positive, and in REBOUND it starts the run back on its own, so it is not among the ones refused without
        --back_height. """

    parser = _parser()
    assert parser.parse_args([]).back_time == -1
    assert unusedBackwardArguments(parser, parser.parse_args(["--back_time", "2"])) == []

    for value in ("0", "-2"):
        with pytest.raises(SystemExit):
            checkBackwardArguments(parser, parser.parse_args(["--mass", "1", "--back_time", value]))

    sv = _stateVector(70e3, 20000.0)
    args = parser.parse_args(["--mass", "1.0", "--back_time", "1.0"])
    (jd, (state,), _), *_ = backwardStatesFromArguments(_traj(), [sv], args, 180000.0, t_kill=args.back_time)
    assert abs((JD - jd)*86400 - 1.0) < 2*BACKWARD_MAX_DT
    assert groundSpeed(jd, state) > 0

    (jd_top, _, _), *_ = backwardStatesFromArguments(_traj(), [sv], parser.parse_args(["--mass", "1.0"]), 180000.0,
        t_kill=1000.0)
    assert 7.0 < (JD - jd_top)*86400 < 12.0


def test_back_time_on_its_own_starts_the_run_back_where_the_height_is_optional():
    """ As in REBOUND, whose --back_height is None unless given: --back_time then sets the highest height to stop at, and
        a --back_height given too wins, in either order. """

    parser = argparse.ArgumentParser()
    parser.add_argument("--back_height", type=float, nargs="?", const=180.0, default=None)
    addBackwardArguments(parser)

    assert parser.parse_args([]).back_height is None
    assert parser.parse_args(["--back_height"]).back_height == 180.0

    args = parser.parse_args(["--back_time", "2"])
    assert args.back_height == 180.0 and args.back_time == 2.0

    assert parser.parse_args(["--back_time", "2", "--back_height", "150"]).back_height == 150.0
    assert parser.parse_args(["--back_height", "150", "--back_time", "2"]).back_height == 150.0


def test_the_drag_gives_the_speed_difference_published_by_vida_et_al_2018(monkeypatch):
    """ Vida et al. (2018), Fig. 5: 0.1 g entering at 20 km/s, 45 deg from the zenith, of 1510 kg/m^3, sigma 0.1 s^2/km^2
        and Gamma*A = 1.21, is detected at 88.38 km with 19.870 km/s, 130 m/s below its speed out of the atmosphere with
        gravity apart. Their ablation model is another, so the mass left there is not known: 0.1-0.08 g give 125-134
        m/s. The run takes their atmosphere, NRLMSISE-00 with constant solar activity: the default NRLMSIS 2.1
        with the measured activity gives 119 m/s. """

    # Select the atmosphere of the paper, and restore the module and the environment after the test
    module = setAtmosphere.__globals__
    for name in ('MSIS_VERSION', 'MSIS_JD', 'MSIS_SPACE_WEATHER'):
        monkeypatch.setitem(module, name, module[name])
    for name in ('MSIS_VERSION_ENV', 'MSIS_JD_ENV', 'MSIS_SPACE_WEATHER_ENV'):
        monkeypatch.delenv(module[name], raising=False)
    setAtmosphere(argparse.Namespace(atm='00', atmtime=None, atmsw='constant'))

    jd = date2JD(2000, 1, 1, 12, 0, 0)
    state = _stateVector(88380.0, 19870.0, jd=jd, lat=np.radians(45.0), lon=0.0)

    const = _noErosion()
    const.rho, const.sigma, const.gamma = 1510.0, 0.1e-6, 1.21/const.shape_factor
    jd_end, states, _ = backwardStates(jd, [state], 0.9e-4, const=const)

    assert abs(dragSpeedGain(jd, state, jd_end, states[0]) - 130.0) < 5.0


def _drawUnknowns(*argv):
    parser = _parser()
    args = parser.parse_args(["--mass", "2.0"] + list(argv))
    args.mc = 3
    checkBackwardArguments(parser, args)

    return args


def test_draw_unknowns_draws_what_is_left_at_its_default():
    """ It fills what was not given and keeps a pair if either of it was given, so that what is known can be given. """

    args = _drawUnknowns("--draw_unknowns")
    assert (args.ablation_coeff, args.ablation_coeff_sigma) == DRAW_UNKNOWNS_SIGMA
    assert (args.density, args.density_sigma) == DRAW_UNKNOWNS_DENSITY
    assert (args.ga, args.ga_sigma) == DRAW_UNKNOWNS_GA
    assert args.mass_sigma == DRAW_UNKNOWNS_MASS_SIGMA*2.0

    # An ordinary chondrite of known density
    args = _drawUnknowns("--draw_unknowns", "--density", "3400", "--density_sigma", "100", "--ablation_coeff", "0.014",
        "--ablation_coeff_sigma", "0.005")
    assert (args.density, args.density_sigma) == (3400.0, 100.0)
    assert (args.ablation_coeff, args.ablation_coeff_sigma) == (0.014, 0.005)
    assert (args.ga, args.ga_sigma) == DRAW_UNKNOWNS_GA
    assert args.mass_sigma == DRAW_UNKNOWNS_MASS_SIGMA*2.0

    args = _drawUnknowns("--draw_unknowns", "--density", "3300", "--ablation_coeff_sigma", "0.01", "--ga", "0.55",
        "--mass_sigma", "0.1")
    assert (args.density, args.density_sigma) == (3300.0, 0.0)
    assert args.ablation_coeff_sigma == 0.01 and args.ablation_coeff == _parser().get_default("ablation_coeff")
    assert (args.ga, args.ga_sigma) == (0.55, 0.0)
    assert args.mass_sigma == 0.1

    # Without the option nothing changes, and --ga keeps its default
    args = _drawUnknowns()
    assert args.density_sigma == 0.0 and args.ablation_coeff_sigma == 0.0 and args.ga_sigma == 0.0
    assert args.mass_sigma == 0.0 and args.ga == BACKWARD_GA


def test_draw_unknowns_keeps_gamma_a_in_the_range_of_the_high_regime():
    """ 95% of the draws of Gamma*A within 1.00 to 1.21, the usual range high up. """

    mean, stddev = DRAW_UNKNOWNS_GA
    sigma_ln = np.sqrt(np.log(1 + (stddev/mean)**2))
    median = mean*np.exp(-sigma_ln**2/2)

    assert abs(median*np.exp(-1.96*sigma_ln) - 1.00) < 0.005
    assert abs(median*np.exp(1.96*sigma_ln) - 1.21) < 0.005


def test_the_help_gives_the_values_per_kind_of_body():
    """ With a meteorite or a PE type in hand, the help leads to the coefficient and the density. """

    text = " ".join(_parser().format_help().split())

    assert "PE type I, ordinary chondrites, 0.014 and 3700" in text
    assert "PE type II, carbonaceous chondrites, 0.042 and 2000" in text
    assert "PE type IIIA, regular cometary material, 0.10 and 750" in text
    assert "PE type IIIB, soft cometary material, 0.21 and 270" in text


def test_draw_unknowns_reaches_the_density_of_every_pe_type():
    """ 95% of the draws of the density hold types I to IIIB of Ceplecha et al. (1998), Table XVII. """

    mean, stddev = DRAW_UNKNOWNS_DENSITY
    sigma_ln = np.sqrt(np.log(1 + (stddev/mean)**2))
    median = mean*np.exp(-sigma_ln**2/2)

    assert median*np.exp(-1.96*sigma_ln) <= 270.0 and median*np.exp(1.96*sigma_ln) >= 3700.0


def test_draw_unknowns_is_refused_where_it_cannot_work():
    """ With the erosion given apart its coefficients would count it twice; without realizations it draws nothing;
        without a run back through the atmosphere, in REBOUND, it does not apply. """

    with pytest.raises(SystemExit):
        _drawUnknowns("--draw_unknowns", "--erosion_coeff", "0.1", "--erosion_height", "80")

    parser = _parser()
    args = parser.parse_args(["--mass", "1.0", "--draw_unknowns"])
    with pytest.raises(SystemExit):
        checkBackwardArguments(parser, args)

    assert unusedBackwardArguments(parser, args) == ["--draw_unknowns"]


def test_realizations_are_drawn_without_uncertainties_in_the_trajectory(capsys):
    """ A trajectory without uncertainties has no state vector to draw, but what is unknown still is. """

    parser = _parser()
    args = parser.parse_args(["--mass", "1e-3", "--draw_unknowns"])
    args.mc = 3
    checkBackwardArguments(parser, args)

    state = _stateVector(90e3, 20e3)
    (_, states, masses), m_inits, sigmas, rhos, gas, _, _ = backwardStatesFromArguments(_traj(), [state], args, 180e3,
        random_seed=1)

    assert len(states) == 4 and len(set(m_inits)) == 4 and len(set(sigmas)) == 4 and len(set(rhos)) == 4
    assert len(set(gas)) == 4 and len(set(masses)) == 4
    assert m_inits[0] == 1e-3 and sigmas[0] == DRAW_UNKNOWNS_SIGMA[0] and rhos[0] == DRAW_UNKNOWNS_DENSITY[0]
    assert gas[0] == DRAW_UNKNOWNS_GA[0]
    assert "only differ in the parameters drawn" in capsys.readouterr().out

    # Without realizations asked for, the spreads have no effect, as they say
    args = parser.parse_args(["--mass", "1e-3", "--density_sigma", "100"])
    (_, states, _), *_ = backwardStatesFromArguments(_traj(), [state], args, 180e3, random_seed=1)
    assert len(states) == 1 and "--density_sigma has no effect" in capsys.readouterr().out
