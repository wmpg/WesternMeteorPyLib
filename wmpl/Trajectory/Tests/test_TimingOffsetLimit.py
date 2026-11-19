""" Tests that a station whose clock needs a larger time offset than max_toffset is ignored, and that the time
    offsets of the other stations are not pulled towards it.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_TimingOffsetLimit.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np

from wmpl.Trajectory.Tests.SyntheticTrajectory import DragMeteor, STATIONS, observeMeteor


# The synthetic meteor and the largest time offset of the solver (s)
METEOR = DragMeteor(20000.0, 45.0, 40.0, 1e-3, 3e-3, 2.0, 5.0)
MAX_TOFFSET = 10.0

# The three stations of SyntheticTrajectory.py and a fourth one, D, which sees the whole visible part
FOUR_STATIONS = STATIONS + [('D', 45.3, 15.5, 300.0, 0.0, 1.0)]

# The three stations of SyntheticTrajectory.py, all seeing the whole visible part
SAME_STATIONS = [station[:4] + (0.0, 1.0) for station in STATIONS]



def solveWithClockError(station_id, clock_error, stations=FOUR_STATIONS, max_toffset=MAX_TOFFSET):
    """ Solve the trajectory of the synthetic meteor seen from the given stations, with the clock of one of
        them off.

    Arguments:
        station_id: [str] ID of the station whose clock is off.
        clock_error: [float] Seconds by which the clock of the station is ahead.

    Keyword arguments:
        stations: [list] Stations in the form of STATIONS. FOUR_STATIONS by default.
        max_toffset: [float] Largest time offset of the solver (s). MAX_TOFFSET by default.

    Return:
        (traj, time_offsets):
            - traj: [Trajectory object] Solved trajectory.
            - time_offsets: [dict] Final time offset of every station (s).
    """

    traj = observeMeteor(METEOR, noise=0.0, stations=stations, clock_offsets={station_id: clock_error}, \
        max_toffset=max_toffset)
    traj.run()

    time_offsets = dict(zip([obs.station_id for obs in traj.observations], traj.time_diffs_final))

    return traj, time_offsets



def ignoredStations(traj):
    """ Return the IDs of the ignored stations of the trajectory. """

    return [obs.station_id for obs in traj.observations if obs.ignore_station]



def testStationBeyondTheLimitIsIgnored():
    """ A station whose clock is 20 s or 60 s off with max_toffset = 10 s is ignored, the other stations keep
        their exact clocks and the velocity is that of the meteor. The offset of 20 s must not be reached in
        two passes of the solver of 10 s each, and at 60 s stations B and C must not be moved towards D.
    """

    for clock_error in [20.0, 60.0]:

        traj, time_offsets = solveWithClockError('D', clock_error)

        assert ignoredStations(traj) == ['D']

        # Check that the other stations are not pulled towards station D
        for station_id in ['B', 'C']:
            assert abs(time_offsets[station_id]) < 1e-3

        assert traj.timing_stddev < 1e-3
        assert abs(traj.v_init - 20000.0) < 50.0



def testReferenceStationBeyondTheLimitIsIgnored():
    """ When the clock of the reference station (A, the one with the most points) is off by more than
        max_toffset, it is that station which is ignored, not all the others which are at the limit.
    """

    traj, time_offsets = solveWithClockError('A', 60.0)

    assert ignoredStations(traj) == ['A']

    # Check that the clocks of the used stations agree
    used_offsets = [time_offsets[station_id] for station_id in ['B', 'C', 'D']]
    assert np.ptp(used_offsets) < 1e-2



def testStationWithinTheLimitIsKept():
    """ A station whose clock is off by less than max_toffset is kept and its offset is estimated. """

    traj, time_offsets = solveWithClockError('D', 5.0)

    assert ignoredStations(traj) == []
    assert abs(time_offsets['D'] + 5.0) < 1e-3



def testStationOffTheOtherWayIsNotTakenForTheFaultyOne():
    """ With three stations on the same part of the meteor and the clock of C 20 s behind, the minimizer
        holds C at +5 s and pulls B to -5 s, so B and C are equally far from the median clock. C must be
        ignored and B kept.
    """

    traj, time_offsets = solveWithClockError('C', -20.0, stations=SAME_STATIONS, max_toffset=5.0)

    assert ignoredStations(traj) == ['C']
    assert abs(time_offsets['B']) < 1e-3



if __name__ == "__main__":

    # Run all tests
    test_list = [testStationBeyondTheLimitIsIgnored, testReferenceStationBeyondTheLimitIsIgnored, \
        testStationWithinTheLimitIsKept, testStationOffTheOtherWayIsNotTakenForTheFaultyOne]

    for test in test_list:
        test()
        print("PASS {:s}".format(test.__name__))

    print("All tests passed.")
