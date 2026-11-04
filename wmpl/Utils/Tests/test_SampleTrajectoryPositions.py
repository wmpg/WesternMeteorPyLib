""" Tests of the sampling of trajectory positions by height and by time in
    wmpl.Utils.SampleTrajectoryPositions, on the example trajectory 20191023_091225.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_SampleTrajectoryPositions.py -v
"""

from __future__ import print_function, division, absolute_import

import os

import numpy as np

import wmpl
from wmpl.Utils.Pickling import loadPickle
from wmpl.Utils.SampleTrajectoryPositions import sampleTrajectory


# Directory and the file name of the example trajectory
TRAJ_DIR = os.path.join(os.path.dirname(wmpl.__file__), 'Dynesty', 'examples', '20191023_091225')
TRAJ_FILE = '20191023_091225_trajectory.pickle'



def _sampleArrays(traj_samples):
    """ Convert the times, heights and velocities of the samples to arrays.

    Arguments:
        traj_samples: [TrajectorySamples object] Sampled positions.

    Return:
        (t_arr, ht_arr, vel_arr):
            - t_arr: [ndarray] Times of the samples (s).
            - ht_arr: [ndarray] Heights of the samples (m).
            - vel_arr: [ndarray] Velocities of the samples (m/s).
    """

    return np.array(traj_samples.t_est, dtype=np.float64), np.array(traj_samples.ht, dtype=np.float64), \
        np.array(traj_samples.vel, dtype=np.float64)



def testHeightModeReturnsTheRequestedHeights():
    """ Check that the samples are at the requested heights above the mean sea level. """

    traj = loadPickle(TRAJ_DIR, TRAJ_FILE)

    traj_samples = sampleTrajectory(traj, 120000.0, 90000.0, 1000.0)
    _, ht_arr, _ = _sampleArrays(traj_samples)

    assert len(ht_arr) == 31
    assert np.allclose(ht_arr, np.arange(120000.0, 89999.0, -1000.0))
    assert np.allclose(traj_samples.ele_geo, ht_arr, atol=0.1)



def testTimeModeReturnsTheRequestedTimes():
    """ Check that the samples are taken at the requested times, also before the beginning and after the
        end of the fireball.
    """

    traj = loadPickle(TRAJ_DIR, TRAJ_FILE)

    t_arr, _, _ = _sampleArrays(sampleTrajectory(traj, -0.2, 0.6, 0.05, by_time=True))

    assert len(t_arr) == 17
    assert np.allclose(t_arr, np.arange(-0.2, 0.6 + 0.025, 0.05), atol=1e-9)



def testTimeModeMatchesTheHeightMode():
    """ Check that sampling by time and by height gives the same motion. """

    traj = loadPickle(TRAJ_DIR, TRAJ_FILE)

    t_fine, ht_fine, _ = _sampleArrays(sampleTrajectory(traj, 130000.0, 80000.0, 10.0))
    t_arr, ht_arr, _ = _sampleArrays(sampleTrajectory(traj, -0.1, 0.5, 0.05, by_time=True))

    sort_ind = np.argsort(t_fine)
    assert np.max(np.abs(np.interp(t_arr, t_fine[sort_ind], ht_fine[sort_ind]) - ht_arr)) < 1.0



def testVelocitiesArePhysical():
    """ Check that the velocity above the beginning stays at the initial velocity, and that it decreases
        after the end.
    """

    traj = loadPickle(TRAJ_DIR, TRAJ_FILE)

    t_arr, _, vel_arr = _sampleArrays(sampleTrajectory(traj, -0.5, 0.8, 0.05, by_time=True))

    # In the 0.5 s above the beginning the velocity changes by less than 0.1%, and it agrees with the
    #   initial velocity of the solution to 0.5%
    vel_beg = vel_arr[np.argmin(np.abs(t_arr))]
    assert np.all(np.abs(vel_arr[t_arr < 0]/vel_beg - 1.0) < 1e-3)
    assert abs(vel_beg/traj.v_init - 1.0) < 5e-3

    # The velocity never increases and stays positive
    assert np.all(np.diff(vel_arr) <= 1.0)
    assert np.all(vel_arr > 0)




if __name__ == "__main__":

    # Run all tests
    test_list = [testHeightModeReturnsTheRequestedHeights, testTimeModeReturnsTheRequestedTimes, \
        testTimeModeMatchesTheHeightMode, testVelocitiesArePhysical]

    for test in test_list:
        test()
        print("PASS {:s}".format(test.__name__))
