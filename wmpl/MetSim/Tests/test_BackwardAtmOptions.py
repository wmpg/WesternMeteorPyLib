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
    checkBackwardArguments, groundSpeed, unusedBackwardArguments
from wmpl.MetSim.MetSimErosion import Constants, EARTH_ROTATION_RATE, runSimulation
from wmpl.Utils.AtmosphereDensity import getAtmDensityTable
from wmpl.Utils.TrajConversions import enu2ECEF, geo2Cartesian


JD = 2460600.25
LAT, LON = np.radians(45.0), np.radians(10.0)


def _stateVector(ht, v_ground, zenith_deg=45.0, azim_deg=90.0):
    """ ECI state vector, with the velocity to the radiant, of a meteor at LAT, LON and ht moving at v_ground relative to
        the ground. """

    pos = np.array(geo2Cartesian(LAT, LON, ht, JD))
    azim, elev = np.radians(azim_deg), np.radians(90.0 - zenith_deg)
    rad_enu = [np.cos(elev)*np.sin(azim), np.cos(elev)*np.cos(azim), np.sin(elev)]
    rad_eci = _eciToEcef(JD).T.dot(np.array(enu2ECEF(LAT, LON, *rad_enu)))

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

    (_, _, masses_fixed), m_fixed, s_fixed, rhos_fixed, gas_fixed = backwardStatesFromArguments(_traj(), state_vects,
        _parser().parse_args(base), 180000.0, random_seed=5)
    assert rhos_fixed == [3000.0]*4 and gas_fixed == [1.21]*4

    args = _parser().parse_args(base + ["--density_sigma", "1000", "--ga_sigma", "0.3"])
    (_, states, masses), m_inits, sigmas, rhos, gas = backwardStatesFromArguments(_traj(), state_vects, args, 180000.0,
        random_seed=5)

    assert rhos[0] == 3000.0 and gas[0] == 1.21 and np.all(np.array(rhos) > 0) and np.all(np.array(gas) > 0)
    assert len(set(rhos)) == 4 and len(set(gas)) == 4
    assert m_inits == m_fixed and sigmas == s_fixed

    # Independent generators: with one of them on, the other does not move
    args_rho = _parser().parse_args(base + ["--density_sigma", "1000"])
    *_, rhos_only, gas_only = backwardStatesFromArguments(_traj(), state_vects, args_rho, 180000.0, random_seed=5)
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
        "0.1", "--density_sigma", "100", "--mass_sigma", "0.1", "--freeze_mass", "--atm", "2.1"]))
    assert set(used) == {"--ablation_coeff", "--ga", "--ga_sigma", "--density_sigma", "--mass_sigma", "--freeze_mass",
        "--atm"}
