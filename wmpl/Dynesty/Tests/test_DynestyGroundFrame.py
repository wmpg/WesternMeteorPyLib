""" Tests that DynestyMetSim samples v_init and the zenith angle in the trajectory solver's inertial (ECI) frame, as
its observations, and simulates relative to the ground (constructConstants(), see wmpl.MetSim.GUI's
constantsRelativeToGround()).

Run under pytest, or directly:

    python -m wmpl.Dynesty.Tests.test_DynestyGroundFrame
"""

import math
import os
import warnings

import numpy as np

from wmpl.Dynesty.DynestyMetSim import ObservationData, constructConstants, runSimulationDynesty
from wmpl.Utils.Pickling import loadPickle


DYNESTY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_DIR = os.path.join(DYNESTY_DIR, "examples", "20191023_091225")
EXAMPLE_PICKLE = "20191023_091225_trajectory.pickle"


def _observations():
    """ The example's observations, with the run settings DynestyMetSim gives them before sampling. """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obs = ObservationData([os.path.join(EXAMPLE_DIR, EXAMPLE_PICKLE)])

    obs.dt, obs.disruption_on, obs.lum_eff_type = 0.005, False, 5
    obs.h_kill = np.min([np.min(obs.height_lum), np.min(obs.height_lag)]) - 1000
    obs.v_kill = np.min(obs.velocities) - 5000

    return obs


def test_observations_carry_the_rotation_along_the_path():

    obs = _observations()
    traj = loadPickle(EXAMPLE_DIR, EXAMPLE_PICKLE)

    assert np.isclose(obs.v_rotation, traj.orbit.v_init - traj.orbit.v_init_norot, rtol=0, atol=1e-9)


def test_sampled_inertial_values_are_simulated_relative_to_the_ground():
    """ The constants are relative to the ground, keep the vertical speed, and carry v_rotation; the simulated speed
        at the start comes back as the sampled one. """

    obs = _observations()
    v_init, zenith_angle = obs.v_init, obs.zenith_angle
    var_names, guess = ["v_init", "zenith_angle", "m_init"], [v_init, zenith_angle, obs.m_init]

    const = constructConstants(guess, obs, var_names, {})

    assert np.isclose(const.v_init, v_init - obs.v_rotation, rtol=0, atol=1e-6)
    assert np.isclose(const.v_init*math.cos(const.zenith_angle), v_init*math.cos(zenith_angle), rtol=1e-12)
    assert const.v_rotation == obs.v_rotation

    # MetSim records its first point after one step, where the meteor has slowed by a few cm/s at 180 km; without
    #   the speeds given back in the solver's frame it would be v_rotation lower
    sr = runSimulationDynesty(guess, obs, var_names, {})
    assert abs(sr.leading_frag_vel_arr[0] - v_init) < 0.1
    assert abs(obs.v_rotation) > 10


if __name__ == "__main__":
    test_observations_carry_the_rotation_along_the_path()
    test_sampled_inertial_values_are_simulated_relative_to_the_ground()
    print("All DynestyGroundFrame checks passed.")
