""" Tests for running MetSimErosion back up a solved trajectory (wmpl.MetSim.BackwardAtmIntegration).

Run under pytest, or directly:

    python -m wmpl.MetSim.Tests.test_BackwardAtmIntegration
"""

import argparse
import os
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from wmpl.MetSim.BackwardAtmIntegration import addBackwardArguments, backwardConstants, backwardState, \
    backwardStates, backwardStatesFromArguments, checkBackwardArguments, referenceLoading
from wmpl.MetSim.MetSimErosion import Constants, Fragment, runSimulation
from wmpl.Rebound.REBOUND import sampleStateVectors
from wmpl.Utils.Pickling import loadPickle, savePickle
from wmpl.Utils.AtmosphereDensity import fitAtmPoly, getAtmDensityTable
from wmpl.Utils.TrajConversions import altAz2RADec, cartesian2Geo, geo2Cartesian, raDec2ECI


# A solved trajectory shipped with the repository (2019-10-23, four stations, 67 km/s, reference point at 116 km),
#   without uncertainties
EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), \
    "Dynesty", "examples", "20191023_091225")
EXAMPLE_PICKLE = "20191023_091225_trajectory.pickle"


def _exampleStart():
    traj = loadPickle(EXAMPLE_DIR, EXAMPLE_PICKLE)

    return traj, np.concatenate([traj.state_vect_mini, traj.v_init*traj.radiant_eci_mini])


def _angleArcsec(a, b):
    return np.degrees(np.arccos(np.clip(np.dot(a, b)/np.linalg.norm(a)/np.linalg.norm(b), -1, 1)))*3600


def test_state_at_the_start_is_the_solver_state():
    """ Before any step, the ground-relative speed and direction MetSim starts with, taken back to ECI with the
        Earth's rotation, give back the solver's state vector exactly. A wrong sign of omega x r would be 681 m/s
        off here. """

    traj, state_vect = _exampleStart()
    const = backwardConstants(traj.jdt_ref, state_vect, 1e-3)
    frag = Fragment()
    frag.init(const, const.m_init, const.rho, const.v_init, const.sigma, const.gamma, const.zenith_angle, \
        const.erosion_mass_index, const.erosion_mass_min, const.erosion_mass_max)

    jd, state_vect_back = backwardState(traj.jdt_ref, state_vect, frag, 0.0)

    assert jd == traj.jdt_ref
    assert np.linalg.norm(state_vect_back[:3] - state_vect[:3]) < 1e-6
    assert np.linalg.norm(state_vect_back[3:] - state_vect[3:]) < 1e-6


def test_backward_run_ends_at_h_kill_on_the_radiant_line():
    """ Run back from 116 to 180 km, the end point is above h_kill by less than one step, at the same height in
        ECI as in MetSim to within the geoid undulation and the Earth's flattening (0.4 m here), and on the line
        back to the radiant in ECI to within the 7.5 arcsec gravity bends the path by over those 73 km. Turning
        the Earth the wrong way between the two times would put it 750 m, 2100 arcsec, off. """

    traj, state_vect = _exampleStart()
    const = backwardConstants(traj.jdt_ref, state_vect, 1e-3)
    frag, results, _ = runSimulation(const)
    t = results[-1][0]

    jd, state_vect_back = backwardState(traj.jdt_ref, state_vect, frag, t)
    _, _, ht = cartesian2Geo(jd, *state_vect_back[:3])

    assert t < 0 and jd == traj.jdt_ref + t/86400.0
    assert const.h_kill < frag.h < const.h_kill + const.v_init*abs(const.dt)
    assert abs(ht - frag.h) < 1.0
    assert _angleArcsec(state_vect_back[:3] - state_vect[:3], traj.radiant_eci_mini) < 20
    assert _angleArcsec(state_vect_back[3:], traj.radiant_eci_mini) < 20


def test_realizations_end_at_the_nominal_epoch_carrying_their_offsets():
    """ Monte Carlo realizations (50 m and 50 m/s here) are run back for as long as the nominal solution takes to
        reach h_kill, so they all end at its epoch. A realization equal to the nominal one ends exactly where it
        does, and the others keep their offsets from it: over the 1.08 s, the velocity offset changes by less than
        0.03 m/s and the position offset moves by the velocity offset times that time to within 3 cm. Without
        uncertainties there are no realizations, and the nominal solution runs back on its own. """

    traj, state_vect = _exampleStart()
    assert sampleStateVectors(traj, 20, random_seed=1) == []

    traj.uncertainties = SimpleNamespace()
    traj.state_vect_cov = np.diag([50.0**2]*6)
    realizations = sampleStateVectors(traj, 20, random_seed=1)

    jd, states, masses = backwardStates(traj.jdt_ref, [state_vect, state_vect.copy()] + realizations, 1e-3)
    t = (jd - traj.jdt_ref)*86400.0

    assert np.array_equal(states[0], states[1])
    for start, end in zip(realizations, states[2:]):
        offset_start, offset_end = start - state_vect, end - states[0]
        assert np.linalg.norm(offset_end[3:] - offset_start[3:]) < 0.03
        assert np.linalg.norm(offset_end[:3] - (offset_start[:3] - offset_start[3:]*t)) < 0.03

    jd_nominal, states_nominal, masses_nominal = backwardStates(traj.jdt_ref, [state_vect], 1e-3)
    assert jd_nominal == jd and len(states_nominal) == 1 and np.array_equal(states_nominal[0], states[0])
    assert masses_nominal == masses[:1]


def test_backward_run_for_a_time_stops_below_h_kill():
    """ With t_kill, the nominal solution and its realizations run back for that long, within one step, and stop
        below h_kill: 0.5 s from 116 km reaches 146 km. """

    traj, state_vect = _exampleStart()
    traj.uncertainties = SimpleNamespace()
    traj.state_vect_cov = np.diag([50.0**2]*6)

    jd, states, _ = backwardStates(traj.jdt_ref, [state_vect] + sampleStateVectors(traj, 3, random_seed=1), 1e-3,
        t_kill=0.5)

    assert 0.5 <= (traj.jdt_ref - jd)*86400.0 < 0.5 + 0.0051
    for sv in states:
        assert 140000 < cartesian2Geo(jd, *sv[:3])[2] < 150000


def _parseArguments(*argv):
    parser = argparse.ArgumentParser()
    addBackwardArguments(parser)

    return parser.parse_args(list(argv))


def test_command_line_arguments_set_the_mass_and_the_physical_parameters():
    """ The run starts with --mass, which grows back unless --freeze_mass keeps it, less with a lower ablation
        coefficient, and a lower density lets the drag slow the meteoroid down more going forwards, so it comes
        back faster. """

    traj, state_vect = _exampleStart()

    def run(*argv):
        (_, states, masses), m_inits, _ = backwardStatesFromArguments(traj, [state_vect],
            _parseArguments("--mass", "1e-6", *argv), 180000.0)
        return m_inits[0], masses[0], np.linalg.norm(states[0][3:])

    m_init, m_end, _ = run()
    assert m_init == 1e-6 and m_end > m_init
    assert run("--freeze_mass")[1] == m_init

    _, m_low, v_dense = run("--ablation_coeff", "0.005", "--density", "3500")
    _, m_high, v_light = run("--ablation_coeff", "0.05", "--density", "1000")
    assert m_init < m_low < m_high and v_dense < v_light


def test_gamma_a_sets_the_drag_as_metsims_gamma_times_shape_factor():
    """ --ga gives the same run as setting MetSim's gamma and shape_factor to that product directly, and a
        larger one, more drag going forwards, brings the meteoroid back faster. """

    traj, state_vect = _exampleStart()

    (_, states, masses), _, _ = backwardStatesFromArguments(traj, [state_vect],
        _parseArguments("--mass", "1e-6", "--ga", "0.55"), 180000.0)

    const = Constants()
    const.rho, const.gamma, const.shape_factor = 3000.0, 1.0, 0.55
    _, states_direct, masses_direct = backwardStates(traj.jdt_ref, [state_vect], 1e-6, const=const)

    assert np.allclose(states[0], states_direct[0], rtol=1e-12, atol=1e-6)
    assert np.isclose(masses[0], masses_direct[0], rtol=1e-12, atol=0)

    (_, states_more, _), _, _ = backwardStatesFromArguments(traj, [state_vect],
        _parseArguments("--mass", "1e-6", "--ga", "1.21"), 180000.0)
    assert np.linalg.norm(states_more[0][3:]) > np.linalg.norm(states[0][3:])


def test_mass_uncertainty_spreads_the_masses_of_the_realizations():
    """ With --mass_sigma the realizations start with log-normal masses of mean --mass and standard deviation
        --mass_sigma: 2000 realizations of 1 +/- 0.4 g have a mean within 2% and a standard deviation within 5%,
        all positive. The nominal mass is unchanged, and the state vectors are the draws sampleStateVectors makes
        with the same seed, so runs with and without it can be compared realization by realization. """

    traj, state_vect = _exampleStart()
    traj.uncertainties = SimpleNamespace()
    traj.state_vect_cov = np.diag([50.0**2]*6)
    realizations = sampleStateVectors(traj, 2000, random_seed=5)

    (_, states, _), m_inits, _ = backwardStatesFromArguments(traj, [state_vect] + realizations,
        _parseArguments("--mass", "1e-3", "--mass_sigma", "4e-4"), 180000.0, random_seed=5)
    masses = np.array(m_inits[1:])

    assert m_inits[0] == 1e-3 and np.all(masses > 0)
    assert abs(np.mean(masses)/1e-3 - 1) < 0.02 and abs(np.std(masses)/4e-4 - 1) < 0.05
    assert np.array_equal(realizations, sampleStateVectors(traj, 2000, random_seed=5))

    (_, _, _), m_inits_none, _ = backwardStatesFromArguments(traj, [state_vect] + realizations[:3],
        _parseArguments("--mass", "1e-3"), 180000.0, random_seed=5)
    assert m_inits_none == [1e-3]*4


def test_erosion_of_the_body_is_undone_by_running_back_with_sigma_plus_eta():
    """ 1 kg at 20 km/s, 45 deg from the zenith, run forwards from 90 km for 1.3 s by MetSim with sigma = 0.023 and
        erosion eta = 0.3 s^2/km^2 (grains of 1e-4 to 1e-3 kg), keeps 42% of its mass. Run back from where its
        body ended with sigma + eta it comes back to 1 kg and 20 km/s within 8e-4 and 7e-6, the first-order error
        of the 0.5 ms steps, while with sigma alone it comes back with 45% of the mass and 20 m/s faster, as the
        smaller body slows down more. """

    traj, _ = _exampleStart()
    lat, lon = np.radians(45.0), np.radians(15.0)
    jd = traj.jdt_ref
    p0 = np.array(geo2Cartesian(lat, lon, 90e3, jd))
    ra, dec = altAz2RADec(np.radians(90.0), np.radians(45.0), jd, lat, lon)
    state_top = np.r_[p0, 20000.0*np.array(raDec2ECI(ra, dec))]

    const = backwardConstants(jd, state_top, 1.0, h_kill=95e3)
    const.dens_co = fitAtmPoly(lat, lon, 40e3, 95e3, jd)
    const.atm_table_ht, const.atm_table_log10_rho = getAtmDensityTable(lat, lon, 40e3, 95e3, jd)
    const.dt, const.h_kill, const.t_kill, const.v_kill = 0.0005, 20e3, 1.3, 100.0
    const.erosion_on, const.erosion_height_start, const.erosion_coeff = True, 95e3, 0.3e-6
    const.erosion_height_change, const.erosion_mass_min, const.erosion_mass_max = -1.0, 1e-4, 1e-3
    frag, results, _ = runSimulation(const)
    t = results[-1][0]
    jd_end, state_end = backwardState(jd, state_top, frag, t)

    def back(sigma):
        const_back = Constants()
        const_back.sigma, const_back.rho, const_back.dt = sigma, const.rho, -0.0005
        cb = backwardConstants(jd_end, state_end, frag.m, h_kill=95e3, const=const_back)
        cb.dens_co, cb.t_kill = const.dens_co, t
        cb.atm_table_ht, cb.atm_table_log10_rho = const.atm_table_ht, const.atm_table_log10_rho
        frag_back, results_back, _ = runSimulation(cb)
        return frag_back.m, np.linalg.norm(backwardState(jd_end, state_end, frag_back, results_back[-1][0])[1][3:])

    m_eta, v_eta = back(0.323e-6)
    m_sigma, v_sigma = back(0.023e-6)

    assert 0.35 < frag.m < 0.5
    assert abs(m_eta - 1) < 1.5e-3 and abs(v_eta/20000.0 - 1) < 2e-5
    assert m_sigma < 0.5 and (v_sigma - 20000.0) > 10.0


def test_ablation_coefficient_uncertainty_spreads_the_coefficients_of_the_realizations():
    """ With --ablation_coeff_sigma each realization runs with its own coefficient, log-normal with mean
        --ablation_coeff: 2000 of 0.1 +/- 0.05 s^2/km^2 have a mean within 3% and a standard deviation within 6%.
        The nominal coefficient, the masses and the state vectors are those without it, and each realization ends
        as a run with its own coefficient does, with more mass than with --ablation_coeff if its own is larger. """

    traj, state_vect = _exampleStart()
    traj.uncertainties = SimpleNamespace()
    traj.state_vect_cov = np.diag([50.0**2]*6)
    realizations = sampleStateVectors(traj, 2000, random_seed=5)
    argv = ["--mass", "1e-3", "--mass_sigma", "4e-4", "--ablation_coeff", "0.1"]

    (_, _, masses_fixed), m_inits_fixed, sigmas_fixed = backwardStatesFromArguments(traj,
        [state_vect] + realizations[:3], _parseArguments(*argv), 180000.0, random_seed=5)
    assert sigmas_fixed == [0.1]*4

    # Only the first realizations are run; the draws of the others are checked
    args = _parseArguments(*(argv + ["--ablation_coeff_sigma", "0.05"]))
    rng = np.random.default_rng([2, 5])
    sigma_ln = np.sqrt(np.log(1 + 0.5**2))
    draws = 0.1*np.exp(sigma_ln*rng.normal(size=2000) - sigma_ln**2/2)
    assert abs(np.mean(draws)/0.1 - 1) < 0.03 and abs(np.std(draws)/0.05 - 1) < 0.06

    (_, states, masses), m_inits, sigmas = backwardStatesFromArguments(traj, [state_vect] + realizations[:3], args,
        180000.0, random_seed=5)
    assert sigmas[0] == 0.1 and np.allclose(sigmas[1:], draws[:3], rtol=1e-12)
    assert m_inits == m_inits_fixed

    const = Constants()
    const.rho = 3000.0
    _, states_own, masses_own = backwardStates(traj.jdt_ref, [state_vect] + realizations[:3], m_inits,
        const=const, sigmas=[sigma/1e6 for sigma in sigmas])
    assert np.allclose(states, states_own, rtol=1e-12, atol=1e-6) and np.allclose(masses, masses_own, rtol=1e-12)

    for m, m_fixed, sigma in zip(masses[1:], masses_fixed[1:], sigmas[1:]):
        assert (m > m_fixed) == (sigma > 0.1)


def test_command_line_reports_the_run_with_ceplechas_types_when_the_coefficient_is_not_drawn(capsys):
    """ Without --ablation_coeff_sigma, the nominal run is also reported with the apparent coefficients of
        Ceplecha's types I and IIIB, and not with it or with a frozen mass. """

    traj, state_vect = _exampleStart()

    backwardStatesFromArguments(traj, [state_vect], _parseArguments("--mass", "1e-3"), 180000.0)
    out = capsys.readouterr().out
    assert "with 0.014" in out and "with 0.21" in out

    for argv in [["--ablation_coeff_sigma", "0.01"], ["--freeze_mass"]]:
        backwardStatesFromArguments(traj, [state_vect], _parseArguments("--mass", "1e-3", *argv), 180000.0)
        assert "Ceplecha" not in capsys.readouterr().out


def _deepStart(h_start):
    """ A trajectory at 20 km/s, 45 deg from the zenith, whose reference point is at h_start (m). """

    jd = 2460000.6
    lat, lon = np.radians(45.0), np.radians(15.0)
    p0 = np.array(geo2Cartesian(lat, lon, h_start, jd))
    ra, dec = altAz2RADec(np.radians(90.0), np.radians(45.0), jd, lat, lon)
    traj = SimpleNamespace(jdt_ref=jd, v_init_stddev=10.0, uncertainties=None)

    return traj, np.r_[p0, 20000.0*np.array(raDec2ECI(ra, dec))]


def test_reference_loading_gives_the_dynamic_pressure_and_the_received_energy():
    """ The example meteor's reference point, at 116 km and 67 km/s, is under 0.0002 MPa and has received 0.9
        MJ/m^2; a fireball at 20 km/s, 45 deg from the zenith, is under 0.028, 0.12 and 0.41 MPa at 70, 60 and
        50 km, having received 130, 590 and 2300 MJ/m^2. """

    traj, state_vect = _exampleStart()
    pressure, energy = referenceLoading(traj.jdt_ref, state_vect)
    assert pressure < 0.0002e6 and 0.8e6 < energy < 1.0e6

    for h_start, p_expected, e_expected in [(70e3, 0.028e6, 130e6), (60e3, 0.12e6, 590e6), (50e3, 0.41e6, 2300e6)]:
        traj, state_vect = _deepStart(h_start)
        pressure, energy = referenceLoading(traj.jdt_ref, state_vect)
        assert abs(pressure/p_expected - 1) < 0.05 and abs(energy/e_expected - 1) < 0.05


def test_command_line_warns_when_the_reference_point_is_deep(capsys):
    """ From 50 km, over the 0.04 MPa of a first fragmentation, the command line warns that a fragmentation above
        cannot be undone, and that the speeds with Ceplecha's types I and IIIB differ by more than the initial
        velocity's 10 m/s uncertainty; from the example's 116 km it warns of neither. """

    traj, state_vect = _deepStart(50e3)
    backwardStatesFromArguments(traj, [state_vect], _parseArguments("--mass", "1"), 180000.0)
    out = capsys.readouterr().out
    assert "At the reference point: dynamic pressure 0.40" in out
    assert "WARNING: The dynamic pressure at the reference point is over the 0.04 MPa" in out
    assert "WARNING: The nominal run ends" in out and "more than the 10.00 m/s uncertainty" in out

    traj, state_vect = _exampleStart()
    backwardStatesFromArguments(traj, [state_vect], _parseArguments("--mass", "1e-3"), 180000.0)
    out = capsys.readouterr().out
    assert "At the reference point" in out and "WARNING" not in out


def test_command_line_suggests_the_coefficient_of_the_trajectorys_drag_fit(capsys):
    """ A trajectory solved with the drag fit of the initial velocity carries the effective ablation coefficient at
        the start of the observed part: it is suggested for --ablation_coeff when the fit constrains it, said to be
        unconstrained when its uncertainty is as large as it, and nothing is said without a drag fit. """

    traj, state_vect = _exampleStart()
    args = _parseArguments("--mass", "1e-3")

    traj.v_init_drag_fit = SimpleNamespace(sigma=0.04, sigma_stddev=0.01, t_range=(0.0, 0.9),
        ht_range=(60e3, 75e3))
    backwardStatesFromArguments(traj, [state_vect], args, 180000.0)
    out = capsys.readouterr().out
    assert "0.0400 +/- 0.0100 s^2/km^2" in out and "--ablation_coeff 0.0400 --ablation_coeff_sigma 0.0100" in out

    traj.v_init_drag_fit = SimpleNamespace(sigma=0.0, sigma_stddev=5.7, t_range=(0.0, 0.28), ht_range=(70e3, 75e3))
    backwardStatesFromArguments(traj, [state_vect], args, 180000.0)
    out = capsys.readouterr().out
    assert "does not constrain it" in out and "--ablation_coeff_sigma 5" not in out

    traj.v_init_drag_fit = None
    backwardStatesFromArguments(traj, [state_vect], args, 180000.0)
    assert "--vinitdrag" not in capsys.readouterr().out


def test_command_line_refuses_a_mass_the_run_cannot_start_from():
    """ A missing, zero or negative --mass, a negative --mass_sigma, --ablation_coeff or --ablation_coeff_sigma,
        or a spread around a zero coefficient, is a command-line error instead of a division by zero or a complex
        power deep in MetSim. """

    parser = argparse.ArgumentParser()
    addBackwardArguments(parser)

    for argv in [[], ["--mass", "0"], ["--mass", "-1"], ["--mass", "1", "--mass_sigma", "-0.1"],
            ["--mass", "1", "--ablation_coeff", "-0.1"], ["--mass", "1", "--ablation_coeff_sigma", "-0.1"],
            ["--mass", "1", "--ablation_coeff", "0", "--ablation_coeff_sigma", "0.1"]]:
        with pytest.raises(SystemExit):
            checkBackwardArguments(parser, parser.parse_args(argv))

    checkBackwardArguments(parser, parser.parse_args(["--mass", "1", "--mass_sigma", "0.1"]))


def test_command_line_saves_the_nominal_solution_and_its_realizations(tmp_path, monkeypatch):
    """ The command line runs the nominal solution and --mc realizations back and saves one row for each, the
        nominal one first, as backwardStates gives them. """

    traj, state_vect = _exampleStart()
    traj.uncertainties = SimpleNamespace()
    traj.state_vect_cov = np.diag([50.0**2]*6)
    savePickle(traj, str(tmp_path), "traj.pickle")

    monkeypatch.setattr(sys, "argv", ["BackwardAtmIntegration", str(tmp_path/"traj.pickle"), "--mc", "3",
        "--seed", "1", "--mass", "1e-3", "--mass_sigma", "2e-4"])
    runpy.run_module("wmpl.MetSim.BackwardAtmIntegration", run_name="__main__")

    rows = np.loadtxt(str(tmp_path/"traj_backward_atm.txt"))
    const = Constants()
    const.rho = 3000.0
    _, states, masses = backwardStates(traj.jdt_ref, [state_vect] + sampleStateVectors(traj, 3, 1), rows[:, 1],
        const=const)

    assert rows.shape == (4, 14) and rows[0, 1] == 1e-3 and len(set(rows[:, 1])) == 4
    assert np.allclose(rows[:, 7:13], states, rtol=1e-9, atol=1e-6) and np.allclose(rows[:, 6], masses, rtol=1e-9)
    assert np.all(rows[:, 13] == Constants().sigma*1e6)



def test_command_line_runs_in_the_atmosphere_chosen_with_atm(tmp_path, monkeypatch):
    """ --atm and --atmtime, shared with REBOUND's command line, choose the MSIS model the run back goes through,
        which the output names: NRLMSIS 2.1 is less dense than NRLMSISE-00 here, so the meteoroid comes back
        slower and with less mass grown back. """

    from wmpl.Utils import AtmosphereDensity

    # The choice is global to the module and kept in the environment, so restore both afterwards
    monkeypatch.setattr(AtmosphereDensity, "MSIS_VERSION", AtmosphereDensity.MSIS_VERSION)
    monkeypatch.setattr(AtmosphereDensity, "MSIS_JD", AtmosphereDensity.MSIS_JD)
    for env in (AtmosphereDensity.MSIS_VERSION_ENV, AtmosphereDensity.MSIS_JD_ENV):
        monkeypatch.setenv(env, "")
        monkeypatch.delenv(env)

    traj, _ = _exampleStart()
    savePickle(traj, str(tmp_path), "traj.pickle")

    rows, headers = {}, {}
    for atm in ("00", "2.1"):
        monkeypatch.setattr(sys, "argv", ["BackwardAtmIntegration", str(tmp_path/"traj.pickle"), "--mass", "1e-3",
            "--atm", atm])
        runpy.run_module("wmpl.MetSim.BackwardAtmIntegration", run_name="__main__")
        rows[atm] = np.loadtxt(str(tmp_path/"traj_backward_atm.txt"))
        with open(str(tmp_path/"traj_backward_atm.txt")) as f:
            headers[atm] = f.readline()

    assert "Atmosphere: NRLMSISE-00, at the trajectory's time" in headers["00"]
    assert "Atmosphere: NRLMSIS 2.1, at the trajectory's time" in headers["2.1"]
    assert rows["2.1"][5] != rows["00"][5] and rows["2.1"][6] != rows["00"][6]

    monkeypatch.setattr(sys, "argv", ["BackwardAtmIntegration", str(tmp_path/"traj.pickle"), "--mass", "1e-3",
        "--atm", "2.1", "--atmtime", "20200101-000000"])
    runpy.run_module("wmpl.MetSim.BackwardAtmIntegration", run_name="__main__")
    with open(str(tmp_path/"traj_backward_atm.txt")) as f:
        assert "NRLMSIS 2.1, at 2020-01-01 00:00:00 UTC (--atmtime)" in f.readline()


if __name__ == "__main__":
    test_state_at_the_start_is_the_solver_state()
    test_backward_run_ends_at_h_kill_on_the_radiant_line()
    test_realizations_end_at_the_nominal_epoch_carrying_their_offsets()
    test_backward_run_for_a_time_stops_below_h_kill()
    test_command_line_arguments_set_the_mass_and_the_physical_parameters()
    test_gamma_a_sets_the_drag_as_metsims_gamma_times_shape_factor()
    test_mass_uncertainty_spreads_the_masses_of_the_realizations()
    test_erosion_of_the_body_is_undone_by_running_back_with_sigma_plus_eta()
    test_ablation_coefficient_uncertainty_spreads_the_coefficients_of_the_realizations()
    test_reference_loading_gives_the_dynamic_pressure_and_the_received_energy()
    test_command_line_refuses_a_mass_the_run_cannot_start_from()
    print("All BackwardAtmIntegration checks passed.")
