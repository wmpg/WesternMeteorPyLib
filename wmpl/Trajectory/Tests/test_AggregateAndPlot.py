""" Tests of the functions in wmpl.Trajectory.AggregateAndPlot.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_AggregateAndPlot.py -v
"""

from __future__ import print_function, division, absolute_import

from types import SimpleNamespace

import numpy as np

from wmpl.Trajectory.AggregateAndPlot import computePeakMagHt



def _observation(abs_mags, heights, ignore_list=None, ignore_station=False):
    """ Make a minimal observation with the attributes used by computePeakMagHt.

    Arguments:
        abs_mags: [list] Absolute magnitudes, NaN for the points without a magnitude.
        heights: [list] Heights of the points (m).

    Keyword arguments:
        ignore_list: [list] 1 for the ignored points. None by default, in which case no point is ignored.
        ignore_station: [bool] Whether the station is ignored. False by default.

    Return:
        [SimpleNamespace] Observation.
    """

    if ignore_list is None:
        ignore_list = [0]*len(abs_mags)

    return SimpleNamespace(absolute_magnitudes=np.array(abs_mags, dtype=np.float64), \
        model_ht=np.array(heights, dtype=np.float64), ignore_list=np.array(ignore_list), \
        ignore_station=ignore_station)



def testPeakSkipsPointsWithoutMagnitudes():
    """ Check that the peak is the brightest point with a magnitude, and that its height is the height of
        that point, also when earlier points are ignored.
    """

    obs_a = _observation([np.nan, 1.0, -2.0, np.nan], [90e3, 85e3, 80e3, 75e3])
    obs_b = _observation([-5.0, np.nan, 0.5, -1.0], [95e3, 88e3, 82e3, 78e3], ignore_list=[1, 0, 0, 0])
    traj = SimpleNamespace(observations=[obs_a, obs_b])

    peak_mag, peak_ht = computePeakMagHt(traj)

    # The -5 point of station B is ignored, so the peak is the -2 of station A
    assert peak_mag == -2.0
    assert peak_ht == 80e3



def testPeakOfIgnoredAndUnmeasuredStations():
    """ Check that ignored stations and stations without magnitudes are skipped, and that the peak is NaN if
        no point has a magnitude.
    """

    obs_ignored = _observation([-8.0, -9.0], [80e3, 70e3], ignore_station=True)
    obs_nan = _observation([np.nan, np.nan], [80e3, 70e3])
    obs_good = _observation([0.0, -1.5], [60e3, 50e3])

    traj = SimpleNamespace(observations=[obs_ignored, obs_nan, obs_good])
    assert computePeakMagHt(traj) == (-1.5, 50e3)

    traj = SimpleNamespace(observations=[obs_ignored, obs_nan])
    assert np.all(np.isnan(computePeakMagHt(traj)))




if __name__ == "__main__":

    # Run all tests
    test_list = [testPeakSkipsPointsWithoutMagnitudes, testPeakOfIgnoredAndUnmeasuredStations]

    for test in test_list:
        test()
        print("PASS {:s}".format(test.__name__))
