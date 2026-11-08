""" Tests that MetSim, as the GUI, AutoRefineFit and DynestyMetSim run it on a solved trajectory, simulates
    the motion relative to the ground and gives the lengths and speeds back in the inertial frame of the
    solver.

The constants are converted to the ground by constantsRelativeToGround() and
loadedConstantsRelativeToGround(), and SimulationResults adds the Earth's rotation along the path back to the
lengths and speeds.

Run with pytest:
    python -m pytest wmpl/MetSim/Tests/test_MetSimGroundFrame.py -v
"""

from __future__ import print_function, division, absolute_import

import copy
import json
import os
import tempfile

import numpy as np

from wmpl.MetSim.GUI import SimulationResults, loadConstants, saveConstants
from wmpl.MetSim.MetSimErosion import Constants, constantsRelativeToGround, constantsToInertial, \
    loadedConstantsRelativeToGround, runSimulation
from wmpl.Utils.Physics import earthRotationSpeed
from wmpl.Utils.Pickling import loadPickle


# Solved trajectory shipped with the repository (2019-10-23, 67 km/s)
EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), \
    'Dynesty', 'examples', '20191023_091225')
EXAMPLE_PICKLE = '20191023_091225_trajectory.pickle'



def _fireball(v_init, zenith_deg):
    """ Make the constants of a 1 kg single-body fireball.

    Arguments:
        v_init: [float] Initial speed (m/s).
        zenith_deg: [float] Zenith angle (deg).

    Return:
        const: [Constants object] Constants of the fireball.
    """

    const = Constants()
    const.m_init = 1.0
    const.v_init = v_init
    const.zenith_angle = np.radians(zenith_deg)
    const.sigma = 0.014/1e6
    const.rho = 3500.0
    const.P_0m = 1500.0
    const.erosion_on = False
    const.disruption_on = False
    const.fragmentation_on = False

    return const



def _onHeights(sr, h_grid):
    """ Interpolate the magnitude, length and speed of the leading fragment on a height grid.

    Arguments:
        sr: [SimulationResults object] Simulation results.
        h_grid: [ndarray] Heights (m).

    Return:
        mag: [ndarray] Absolute magnitudes.
        length: [ndarray] Lengths from the first height (m).
        vel: [ndarray] Speeds (m/s).
    """

    order = np.argsort(sr.leading_frag_height_arr)
    hts = sr.leading_frag_height_arr[order]

    length = np.interp(h_grid, hts, sr.leading_frag_length_arr[order])
    vel = np.interp(h_grid, hts, sr.leading_frag_vel_arr[order])
    mag = np.interp(h_grid, hts, sr.abs_magnitude[order])

    return mag, length - length[0], vel



def testTheConstantsRelativeToTheGroundKeepTheVerticalSpeed():
    """ Check that the conversion takes the rotation off the speed, keeps the vertical speed and stores the
        rotation, that constantsToInertial() inverts it, and that it changes nothing without rotation.
    """

    const = constantsRelativeToGround(_fireball(15300.0, 45.0), 300.0)

    assert const.v_init == 15000.0
    assert const.v_rotation == 300.0
    assert np.isclose(const.v_init*np.cos(const.zenith_angle), 15300.0*np.cos(np.radians(45.0)), \
        rtol=1e-12)

    # Check that the conversion back to the inertial frame inverts it
    const = constantsToInertial(const)

    assert np.isclose(const.v_init, 15300.0, rtol=0, atol=1e-9)
    assert np.isclose(const.zenith_angle, np.radians(45.0), rtol=0, atol=1e-12)
    assert const.v_rotation == 0.0

    # Check that nothing changes without rotation
    const = constantsRelativeToGround(_fireball(15300.0, 45.0), 0.0)

    assert const.v_init == 15300.0
    assert np.isclose(const.zenith_angle, np.radians(45.0), rtol=1e-12)



def testTheSimulationResultsAreGivenInTheSolverFrame():
    """ Check that with v_rotation in the constants, the lengths gain v_rotation*t and the speeds v_rotation,
        and that the heights, magnitudes, dynamic pressures and masses do not change.
    """

    const = _fireball(15000.0, 45.0)
    sr_ground = SimulationResults(const, *runSimulation(copy.deepcopy(const), compute_wake=False))

    const.v_rotation = 300.0
    sr = SimulationResults(const, *runSimulation(copy.deepcopy(const), compute_wake=False))

    assert sr_ground.v_rotation == 0.0
    assert sr.v_rotation == 300.0

    # Check the lengths and speeds
    for name in ('brightest', 'leading_frag', 'main'):

        assert np.allclose(getattr(sr, name + '_length_arr'), getattr(sr_ground, name + '_length_arr') \
            + 300.0*sr.time_arr, rtol=0, atol=1e-6)
        assert np.allclose(getattr(sr, name + '_vel_arr'), getattr(sr_ground, name + '_vel_arr') + 300.0, \
            rtol=0, atol=1e-9)

    # Check that the rest is unchanged
    for name in ('leading_frag_height_arr', 'abs_magnitude', 'leading_frag_dyn_press_arr', \
        'mass_total_active_arr'):

        assert np.array_equal(getattr(sr, name), getattr(sr_ground, name))



def testTheInertialValuesReproduceTheMeteorRelativeToTheGround():
    """ Check that a fireball at 15 km/s relative to the ground and 300 m/s faster in the solver's frame is
        reproduced from the inertial speed and zenith angle converted to the ground, while simulating the
        inertial values directly gives a light curve 0.09 mag brighter.
    """

    v_rotation = 300.0

    # Simulate the true meteor relative to the ground
    truth = _fireball(15000.0, 45.0)
    truth.v_rotation = v_rotation
    sr_truth = SimulationResults(truth, *runSimulation(copy.deepcopy(truth), compute_wake=False))

    # Compute the inertial speed and zenith angle with the same vertical speed
    v_eci = 15000.0 + v_rotation
    zenith_eci = np.degrees(np.arccos(15000.0*np.cos(np.radians(45.0))/v_eci))

    # Simulate from the inertial values converted to the ground
    const = constantsRelativeToGround(_fireball(v_eci, zenith_eci), v_rotation)
    sr_new = SimulationResults(const, *runSimulation(copy.deepcopy(const), compute_wake=False))

    # Simulate the inertial values directly, as before
    old = _fireball(v_eci, zenith_eci)
    sr_old = SimulationResults(old, *runSimulation(copy.deepcopy(old), compute_wake=False))

    # Compare them on the heights where the meteor is brighter than magnitude 0
    lum_hts = sr_truth.leading_frag_height_arr[sr_truth.abs_magnitude < 0]
    h_grid = np.linspace(lum_hts.max(), lum_hts.min(), 50)
    mag_t, len_t, vel_t = _onHeights(sr_truth, h_grid)
    mag_n, len_n, vel_n = _onHeights(sr_new, h_grid)
    mag_o, _, _ = _onHeights(sr_old, h_grid)

    assert np.max(np.abs(mag_n - mag_t)) < 1e-3
    assert np.max(np.abs(len_n - len_t)) < 1.0
    assert np.max(np.abs(vel_n - vel_t)) < 1.0

    assert np.mean(mag_o - mag_t) < -0.08



def testOldConstantsFilesAreConvertedToTheGround():
    """ Check that a constants file without v_rotation is converted to the ground when loaded, and that one
        saved with v_rotation keeps its ground values and takes the rotation of the trajectory.
    """

    traj = loadPickle(EXAMPLE_DIR, EXAMPLE_PICKLE)
    v_rotation = earthRotationSpeed(traj)

    assert v_rotation == traj.orbit.v_init - traj.orbit.v_init_norot

    with tempfile.TemporaryDirectory() as tmp_dir:

        # Save a file with the inertial v_init and no v_rotation
        old = _fireball(traj.orbit.v_init, 50.0)
        saveConstants(old, tmp_dir, 'old.json')

        with open(os.path.join(tmp_dir, 'old.json')) as f:
            assert 'v_rotation' not in json.load(f)

        # Check that it is converted when loaded
        const, const_json = loadConstants(os.path.join(tmp_dir, 'old.json'))
        loadedConstantsRelativeToGround(const, const_json, traj)

        assert np.isclose(const.v_init, traj.orbit.v_init_norot, rtol=0, atol=1e-6)
        assert const.v_rotation == v_rotation

        # Check that it is not converted again after it is saved
        saveConstants(const, tmp_dir, 'new.json')
        const_new, const_json_new = loadConstants(os.path.join(tmp_dir, 'new.json'))
        loadedConstantsRelativeToGround(const_new, const_json_new, traj)

        assert const_new.v_init == const.v_init
        assert const_new.v_rotation == v_rotation
        assert np.isclose(const_new.zenith_angle, const.zenith_angle, rtol=0, atol=1e-12)

        # Check that a file saved with another rotation takes the rotation of this trajectory
        const.v_rotation = v_rotation + 300.0
        saveConstants(const, tmp_dir, 'other.json')
        const_other, const_json_other = loadConstants(os.path.join(tmp_dir, 'other.json'))
        loadedConstantsRelativeToGround(const_other, const_json_other, traj)

        assert const_other.v_init == const.v_init
        assert const_other.v_rotation == v_rotation
