""" Tests that DynestyMetSim samples v_init and the zenith angle in the inertial frame of the trajectory
    solver, as its observations, and simulates them relative to the ground (constructConstants()).

Run with pytest:
    python -m pytest wmpl/Dynesty/Tests/test_DynestyGroundFrame.py -v
"""

from __future__ import print_function, division, absolute_import

import copy
import math
import os
import tempfile
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from wmpl.Dynesty.DynestyMetSim import ObservationData, constructConstants, dynestyMainRun, \
    groundFrameMarkerPath, replayInTheFrameOfTheRun, runSimulationDynesty
from wmpl.MetSim.MetSimErosion import constantsToInertial
from wmpl.Utils.Pickling import loadPickle


# Solved trajectory shipped with the repository
DYNESTY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_DIR = os.path.join(DYNESTY_DIR, 'examples', '20191023_091225')
EXAMPLE_PICKLE = '20191023_091225_trajectory.pickle'



def _observations():
    """ Load the observations of the example with the run settings that DynestyMetSim gives them before
        sampling.

    Return:
        obs: [ObservationData object] Observations of the example.
    """

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        obs = ObservationData([os.path.join(EXAMPLE_DIR, EXAMPLE_PICKLE)])

    obs.dt = 0.005
    obs.disruption_on = False
    obs.lum_eff_type = 5
    obs.h_kill = np.min([np.min(obs.height_lum), np.min(obs.height_lag)]) - 1000
    obs.v_kill = np.min(obs.velocities) - 5000

    return obs



def testTheObservationsCarryTheRotationAlongThePath():
    """ Check that the observations carry the Earth's rotation along the path of the trajectory.
    """

    obs = _observations()
    traj = loadPickle(EXAMPLE_DIR, EXAMPLE_PICKLE)

    assert np.isclose(obs.v_rotation, traj.orbit.v_init - traj.orbit.v_init_norot, rtol=0, atol=1e-9)



def testTheSampledInertialValuesAreSimulatedRelativeToTheGround():
    """ Check that the constants are relative to the ground, keep the vertical speed and carry v_rotation, and
        that the simulated speed at the start comes back as the sampled one.
    """

    obs = _observations()
    var_names = ['v_init', 'zenith_angle', 'm_init']
    guess = [obs.v_init, obs.zenith_angle, obs.m_init]

    const = constructConstants(guess, obs, var_names, {})

    assert np.isclose(const.v_init, obs.v_init - obs.v_rotation, rtol=0, atol=1e-6)
    assert np.isclose(const.v_init*math.cos(const.zenith_angle), obs.v_init*math.cos(obs.zenith_angle), \
        rtol=1e-12)
    assert const.v_rotation == obs.v_rotation

    # Check the first simulated speed, one step after the start, in the solver's frame
    sr = runSimulationDynesty(guess, obs, var_names, {})

    assert abs(obs.v_rotation) > 10
    assert abs(sr.leading_frag_vel_arr[0] - obs.v_init) < 0.1



def testSavedConstantsAreNotConvertedTwice():
    """ Check that constants saved relative to the ground, converted back to the inertial frame of the
        observations as plotJSONDataVsObs() does, are simulated with the saved speed and zenith angle.
    """

    obs = _observations()
    var_names = ['v_init', 'zenith_angle', 'm_init']

    # Make constants relative to the ground, as they are saved
    saved = constructConstants([obs.v_init, obs.zenith_angle, obs.m_init], obs, var_names, {})

    # Convert a copy back to the inertial frame of the observations and construct the constants again
    inertial = copy.deepcopy(saved)
    inertial.v_rotation = obs.v_rotation
    inertial = constantsToInertial(inertial)
    const = constructConstants([inertial.v_init, inertial.zenith_angle, obs.m_init], obs, var_names, {})

    assert np.isclose(const.v_init, saved.v_init, rtol=0, atol=1e-6)
    assert np.isclose(const.zenith_angle, saved.zenith_angle, rtol=0, atol=1e-12)

    # Check that a model saved with another rotation, as a synthetic meteor, keeps its ground values
    saved.v_rotation = 300.0
    obs.v_rotation = 0.0
    inertial = copy.deepcopy(saved)
    inertial.v_rotation = obs.v_rotation
    inertial = constantsToInertial(inertial)
    const = constructConstants([inertial.v_init, inertial.zenith_angle, obs.m_init], obs, var_names, {})

    assert np.isclose(const.v_init, saved.v_init, rtol=0, atol=1e-6)
    assert np.isclose(const.zenith_angle, saved.zenith_angle, rtol=0, atol=1e-12)



def testRunsMadeBeforeTheGroundFrameAreReplayedWithTheInertialSpeed():
    """ Check that a dynesty run without the ground-frame marker is replayed without rotation and cannot be
        resumed, and that a run with the marker keeps the rotation of the observations.
    """

    with tempfile.TemporaryDirectory() as tmp_dir:

        dynesty_file = os.path.join(tmp_dir, 'run.dynesty')
        open(dynesty_file, 'w').close()

        # Check that a run without the marker is replayed without rotation
        obs = SimpleNamespace(v_rotation=-49.6)
        replayInTheFrameOfTheRun(dynesty_file, obs)

        assert obs.v_rotation == 0.0

        # Check that it cannot be resumed
        with pytest.raises(ValueError):
            dynestyMainRun(dynesty_file, SimpleNamespace(v_rotation=-49.6), [], {}, {})

        assert not os.path.isfile(groundFrameMarkerPath(dynesty_file))

        # Check that a run with the marker keeps the rotation
        open(groundFrameMarkerPath(dynesty_file), 'w').close()
        obs = SimpleNamespace(v_rotation=-49.6)
        replayInTheFrameOfTheRun(dynesty_file, obs)

        assert obs.v_rotation == -49.6
