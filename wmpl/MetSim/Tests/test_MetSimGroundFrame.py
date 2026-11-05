""" Tests that MetSim, as the GUI, AutoRefineFit and DynestyMetSim run it on a solved trajectory, simulates the
motion relative to the air (the ground) and gives the lengths and speeds back in the trajectory solver's inertial
(ECI) frame, where the observations are (wmpl.MetSim.GUI: rotationSpeedAlongTrajectory(),
constantsRelativeToGround(), loadedConstantsRelativeToGround(), SimulationResults).

Run under pytest, or directly:

    python -m wmpl.MetSim.Tests.test_MetSimGroundFrame
"""

import copy
import json
import math
import os
import tempfile
from types import SimpleNamespace

import numpy as np

from wmpl.MetSim.GUI import SimulationResults, constantsRelativeToGround, loadConstants, \
    loadedConstantsRelativeToGround, rotationSpeedAlongTrajectory, saveConstants
from wmpl.MetSim.MetSimErosion import Constants, runSimulation
from wmpl.Utils.Pickling import loadPickle


# A solved trajectory shipped with the repository (2019-10-23, 67 km/s)
EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), \
    "Dynesty", "examples", "20191023_091225")
EXAMPLE_PICKLE = "20191023_091225_trajectory.pickle"


def _fireball(v_init, zenith_deg):
    """ A 1 kg single-body fireball. """

    const = Constants()
    const.m_init, const.v_init, const.zenith_angle = 1.0, v_init, math.radians(zenith_deg)
    const.sigma, const.rho, const.P_0m = 0.014/1e6, 3500.0, 1500.0
    const.erosion_on = const.disruption_on = const.fragmentation_on = False

    return const


def _onHeights(sr, h_grid):
    """ Magnitude, length and speed of the leading fragment on a height grid, length and time from its top. """

    order = np.argsort(sr.leading_frag_height_arr)
    hs = sr.leading_frag_height_arr[order]
    t = np.interp(h_grid, hs, sr.time_arr[order])
    length = np.interp(h_grid, hs, sr.leading_frag_length_arr[order])
    vel = np.interp(h_grid, hs, sr.leading_frag_vel_arr[order])
    mag = np.interp(h_grid, hs, sr.abs_magnitude[order])

    return mag, length - length[0], vel, t - t[0]


def test_rotation_speed_along_trajectory():
    """ It is the solver's v_init minus v_init_norot, and 0 when they are not there. """

    traj = loadPickle(EXAMPLE_DIR, EXAMPLE_PICKLE)
    v_rotation = rotationSpeedAlongTrajectory(traj)

    assert v_rotation == traj.orbit.v_init - traj.orbit.v_init_norot
    assert 0 < abs(v_rotation) < 465

    assert rotationSpeedAlongTrajectory(None) == 0.0
    assert rotationSpeedAlongTrajectory(SimpleNamespace(orbit=SimpleNamespace(v_init=15000.0,
        v_init_norot=None))) == 0.0


def test_constants_relative_to_ground_keep_the_vertical_speed():

    const = constantsRelativeToGround(_fireball(15300.0, 45.0), 300.0)

    assert const.v_init == 15000.0
    assert const.v_rotation == 300.0
    assert np.isclose(const.v_init*math.cos(const.zenith_angle), 15300.0*math.cos(math.radians(45.0)), rtol=1e-12)

    # Without rotation nothing changes
    const = constantsRelativeToGround(_fireball(15300.0, 45.0), 0.0)
    assert const.v_init == 15300.0 and np.isclose(const.zenith_angle, math.radians(45.0), rtol=1e-12)


def test_simulation_results_are_given_in_the_solver_frame():
    """ With v_rotation in the constants, the lengths get v_rotation*t and the speeds v_rotation; the rest, and the
        whole results without it, are MetSim's. """

    const = _fireball(15000.0, 45.0)
    sr_ground = SimulationResults(const, *runSimulation(copy.deepcopy(const), compute_wake=False))

    const.v_rotation = 300.0
    sr = SimulationResults(const, *runSimulation(copy.deepcopy(const), compute_wake=False))

    assert sr_ground.v_rotation == 0.0 and sr.v_rotation == 300.0
    for name in ("brightest", "leading_frag", "main"):
        assert np.allclose(getattr(sr, name + "_length_arr"), getattr(sr_ground, name + "_length_arr") \
            + 300.0*sr.time_arr, rtol=0, atol=1e-6)
        assert np.allclose(getattr(sr, name + "_vel_arr"), getattr(sr_ground, name + "_vel_arr") + 300.0, rtol=0,
            atol=1e-9)
    for name in ("leading_frag_height_arr", "abs_magnitude", "leading_frag_dyn_press_arr", "mass_total_active_arr"):
        assert np.array_equal(getattr(sr, name), getattr(sr_ground, name))


def test_inertial_values_reproduce_the_meteor_relative_to_the_ground():
    """ A fireball moving at 15 km/s relative to the ground, 300 m/s more in the solver's frame (an eastward one at
        mid latitudes). Started from the solver's speed and zenith angle and turned relative to the ground, MetSim
        gives the same magnitudes, lengths and speeds in the solver's frame. Taking the solver's speed as relative
        to the ground, as before, gives a light curve 0.09 mag brighter: a fit makes up for it with a mass 12% higher
        and an ablation coefficient 24% lower. """

    v_rotation = 300.0
    truth = _fireball(15000.0, 45.0)
    truth.v_rotation = v_rotation
    sr_truth = SimulationResults(truth, *runSimulation(copy.deepcopy(truth), compute_wake=False))

    # The solver's speed and zenith angle: same vertical speed, v_rotation more along the path
    v_eci = 15000.0 + v_rotation
    zenith_eci = math.degrees(math.acos(15000.0*math.cos(math.radians(45.0))/v_eci))

    const = constantsRelativeToGround(_fireball(v_eci, zenith_eci), v_rotation)
    sr_new = SimulationResults(const, *runSimulation(copy.deepcopy(const), compute_wake=False))

    old = _fireball(v_eci, zenith_eci)
    sr_old = SimulationResults(old, *runSimulation(copy.deepcopy(old), compute_wake=False))

    lum_hts = sr_truth.leading_frag_height_arr[sr_truth.abs_magnitude < 0]
    h_grid = np.linspace(lum_hts.max(), lum_hts.min(), 50)
    mag_t, len_t, vel_t, _ = _onHeights(sr_truth, h_grid)
    mag_n, len_n, vel_n, _ = _onHeights(sr_new, h_grid)
    mag_o, _, vel_o, _ = _onHeights(sr_old, h_grid)

    assert np.max(np.abs(mag_n - mag_t)) < 1e-3
    assert np.max(np.abs(len_n - len_t)) < 1.0
    assert np.max(np.abs(vel_n - vel_t)) < 1.0

    assert np.mean(mag_o - mag_t) < -0.08


def test_constants_saved_before_are_turned_relative_to_the_ground():
    """ A constants file without v_rotation has the solver's inertial v_init and zenith angle; one with it is left
        as it is. """

    traj = loadPickle(EXAMPLE_DIR, EXAMPLE_PICKLE)
    v_rotation = rotationSpeedAlongTrajectory(traj)

    with tempfile.TemporaryDirectory() as tmp_dir:

        # A file as saved before: v_init in the solver's frame, no v_rotation
        old = _fireball(traj.orbit.v_init, 50.0)
        saveConstants(old, tmp_dir, "old.json")
        with open(os.path.join(tmp_dir, "old.json")) as f:
            assert "v_rotation" not in json.load(f)

        const, const_json = loadConstants(os.path.join(tmp_dir, "old.json"))
        loadedConstantsRelativeToGround(const, const_json, traj)
        assert np.isclose(const.v_init, traj.orbit.v_init_norot, rtol=0, atol=1e-6)
        assert const.v_rotation == v_rotation

        # Saved again, it is kept relative to the ground
        saveConstants(const, tmp_dir, "new.json")
        const_new, const_json_new = loadConstants(os.path.join(tmp_dir, "new.json"))
        loadedConstantsRelativeToGround(const_new, const_json_new, traj)
        assert const_new.v_init == const.v_init and const_new.v_rotation == v_rotation
        assert np.isclose(const_new.zenith_angle, const.zenith_angle, rtol=0, atol=1e-12)


if __name__ == "__main__":
    test_rotation_speed_along_trajectory()
    test_constants_relative_to_ground_keep_the_vertical_speed()
    test_simulation_results_are_given_in_the_solver_frame()
    test_inertial_values_reproduce_the_meteor_relative_to_the_ground()
    test_constants_saved_before_are_turned_relative_to_the_ground()
    print("All MetSimGroundFrame checks passed.")
