""" Tests that a time range excluded from a station up to its end is solved when the last points of the
    station are ignored, given as ignored or rejected as bad picks by the solver.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_ExcludedTimeIgnoredEnd.py -v
"""

from __future__ import print_function, division, absolute_import

import copy

import numpy as np

import matplotlib
matplotlib.use('Agg')

from wmpl.Trajectory.Tests.SyntheticTrajectory import observeStraightMeteor
from wmpl.Utils.Math import angleBetweenVectors


def excludeToTheEnd(ignore_last=False, last_pick_error=0.0):
    """ Observe the straight meteor and exclude the end of the first station from its 31st point on.

    Keyword arguments:
        ignore_last: [bool] Ignore the last point of the station. False by default.
        last_pick_error: [float] Error added to the first coordinate of the last point (rad). 0 by default.

    Return:
        (traj, radiant_eci):
            - traj: [Trajectory object] Trajectory object with the observations, not yet solved.
            - radiant_eci: [ndarray] True radiant unit vector in ECI coordinates.
    """

    traj, _, radiant_eci = observeStraightMeteor(noise=1e-5)

    obs_list = copy.deepcopy(traj.observations)
    obs = obs_list[0]

    # Exclude the time from the 31st point to the last one
    obs.excluded_time = [obs.time_data[30], obs.time_data[-1]]

    # Ignore the last point
    obs.ignore_list = np.zeros(len(obs.time_data), dtype=np.uint8)
    if ignore_last:
        obs.ignore_list[-1] = 1

    # Move the last point off the meteor
    obs.meas1 = np.copy(obs.meas1)
    obs.meas1[-1] += last_pick_error

    # Initialize the observations again
    traj.observations = []
    for obs in obs_list:
        traj.infillWithObs(obs)

    return traj, radiant_eci



def testTheExcludedRangeEndsAtTheLastUsedPoint():
    """ Check that the excluded range of a station whose last point is ignored ends at the last used one. """

    traj, _ = excludeToTheEnd(ignore_last=True)
    obs = traj.observations[0]

    n_used = np.count_nonzero(obs.ignore_list == 0)
    assert obs.excluded_indx_range == [30, n_used - 1]



def testAnIgnoredLastPointIsSolved():
    """ Check that the meteor is solved when the last point of the station with the excluded range is
        ignored.
    """

    traj, radiant_eci = excludeToTheEnd(ignore_last=True)
    traj.run()

    assert np.degrees(angleBetweenVectors(traj.radiant_eci_mini, radiant_eci))*3600 < 10



def testARejectedLastPickIsSolved():
    """ Check that the meteor is solved when the solver rejects the last pick of the station with the excluded
        range as a bad pick.
    """

    traj, radiant_eci = excludeToTheEnd(last_pick_error=0.003)
    traj.run()

    # The last pick was rejected
    assert traj.observations[0].ignore_list[-1] == 1

    assert np.degrees(angleBetweenVectors(traj.radiant_eci_mini, radiant_eci))*3600 < 10



if __name__ == "__main__":

    testTheExcludedRangeEndsAtTheLastUsedPoint()
    testAnIgnoredLastPointIsSolved()
    testARejectedLastPickIsSolved()

    print("All tests passed.")
