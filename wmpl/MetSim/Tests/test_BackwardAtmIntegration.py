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

from wmpl.MetSim.BackwardAtmIntegration import addBackwardArguments, backwardConstants, backwardState, \
    backwardStates, backwardStatesFromArguments
from wmpl.MetSim.MetSimErosion import Constants, Fragment, runSimulation
from wmpl.Rebound.REBOUND import sampleStateVectors
from wmpl.Utils.Pickling import loadPickle, savePickle
from wmpl.Utils.TrajConversions import cartesian2Geo


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
        (_, states, masses), m_inits = backwardStatesFromArguments(traj, [state_vect],
            _parseArguments("--mass", "1e-6", *argv), 180000.0)
        return m_inits[0], masses[0], np.linalg.norm(states[0][3:])

    m_init, m_end, _ = run()
    assert m_init == 1e-6 and m_end > m_init
    assert run("--freeze_mass")[1] == m_init

    _, m_low, v_dense = run("--ablation_coeff", "0.005", "--density", "3500")
    _, m_high, v_light = run("--ablation_coeff", "0.05", "--density", "1000")
    assert m_init < m_low < m_high and v_dense < v_light


def test_mass_uncertainty_spreads_the_masses_of_the_realizations():
    """ With --mass_sigma the realizations start with log-normal masses of mean --mass and standard deviation
        --mass_sigma: 2000 realizations of 1 +/- 0.4 g have a mean within 2% and a standard deviation within 5%,
        all positive. The nominal mass is unchanged, and the state vectors are the draws sampleStateVectors makes
        with the same seed, so runs with and without it can be compared realization by realization. """

    traj, state_vect = _exampleStart()
    traj.uncertainties = SimpleNamespace()
    traj.state_vect_cov = np.diag([50.0**2]*6)
    realizations = sampleStateVectors(traj, 2000, random_seed=5)

    (_, states, _), m_inits = backwardStatesFromArguments(traj, [state_vect] + realizations,
        _parseArguments("--mass", "1e-3", "--mass_sigma", "4e-4"), 180000.0, random_seed=5)
    masses = np.array(m_inits[1:])

    assert m_inits[0] == 1e-3 and np.all(masses > 0)
    assert abs(np.mean(masses)/1e-3 - 1) < 0.02 and abs(np.std(masses)/4e-4 - 1) < 0.05
    assert np.array_equal(realizations, sampleStateVectors(traj, 2000, random_seed=5))

    (_, _, _), m_inits_none = backwardStatesFromArguments(traj, [state_vect] + realizations[:3],
        _parseArguments("--mass", "1e-3"), 180000.0, random_seed=5)
    assert m_inits_none == [1e-3]*4


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

    assert rows.shape == (4, 13) and rows[0, 1] == 1e-3 and len(set(rows[:, 1])) == 4
    assert np.allclose(rows[:, 7:], states, rtol=1e-9, atol=1e-6) and np.allclose(rows[:, 6], masses, rtol=1e-9)


if __name__ == "__main__":
    test_state_at_the_start_is_the_solver_state()
    test_backward_run_ends_at_h_kill_on_the_radiant_line()
    test_realizations_end_at_the_nominal_epoch_carrying_their_offsets()
    test_backward_run_for_a_time_stops_below_h_kill()
    test_command_line_arguments_set_the_mass_and_the_physical_parameters()
    test_mass_uncertainty_spreads_the_masses_of_the_realizations()
    print("All BackwardAtmIntegration checks passed.")
