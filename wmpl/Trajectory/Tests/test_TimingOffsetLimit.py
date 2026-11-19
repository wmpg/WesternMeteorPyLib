""" Tests that a station whose clock needs a larger time offset than max_toffset is ignored, and that the time
    offsets of the other stations are not pulled towards it.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_TimingOffsetLimit.py -v
"""

from __future__ import print_function, division, absolute_import

import copy

import numpy as np

from wmpl.Trajectory.Tests.SyntheticTrajectory import DragMeteor, STATIONS, observeMeteor


# The synthetic meteor and the largest time offset of the solver (s)
METEOR = DragMeteor(20000.0, 45.0, 40.0, 1e-3, 3e-3, 2.0, 5.0)
MAX_TOFFSET = 10.0

# The three stations of SyntheticTrajectory.py and a fourth one, D, which sees the whole visible part
FOUR_STATIONS = STATIONS + [('D', 45.3, 15.5, 300.0, 0.0, 1.0)]

# The three stations of SyntheticTrajectory.py, all seeing the whole visible part
SAME_STATIONS = [station[:4] + (0.0, 1.0) for station in STATIONS]

# Stations A and C see the first 20% of the visible part and D the last 20%, so D overlaps only with B
BRIDGE_STATIONS = [('A', 45.6, 14.6, 200.0, 0.0, 0.2), ('B', 44.5, 15.3, 400.0, 0.0, 1.0), \
    ('C', 45.1, 15.9, 150.0, 0.0, 0.2), ('D', 45.3, 15.5, 300.0, 0.8, 1.0)]

# The bridge stations and a fifth one, E, which sees the last 20% with D, so B is the only overlap between the
#   group of A and C and the group of D and E
TWO_GROUP_STATIONS = BRIDGE_STATIONS + [('E', 44.9, 14.9, 250.0, 0.8, 1.0)]

# Stations A, B and C see the whole visible part, and D and E the middle 20%, inside the middle 60% which A
#   and C exclude in the test
EXCLUDED_MIDDLE_STATIONS = [('A', 45.6, 14.6, 200.0, 0.0, 1.0), ('B', 44.5, 15.3, 400.0, 0.0, 1.0), \
    ('C', 45.1, 15.9, 150.0, 0.0, 1.0), ('D', 45.3, 15.5, 300.0, 0.4, 0.6), \
    ('E', 44.9, 14.9, 250.0, 0.4, 0.6)]



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



def testStationAtTheLimitIsKept():
    """ A station whose clock is off by max_toffset, or by a little less, is at the limit but can be aligned
        within it, so it is kept and its offset is estimated.
    """

    for clock_error in [4.99999, 5.0]:

        traj, time_offsets = solveWithClockError('C', clock_error, stations=SAME_STATIONS, max_toffset=5.0)

        assert ignoredStations(traj) == []
        assert abs(time_offsets['C'] + clock_error) < 1e-3



def testStationOffTheOtherWayIsNotTakenForTheFaultyOne():
    """ With three stations on the same part of the meteor and the clock of C 20 s behind, the minimizer
        holds C at +5 s and pulls B to -5 s, so B and C are equally far from the median clock. C must be
        ignored and B kept.
    """

    traj, time_offsets = solveWithClockError('C', -20.0, stations=SAME_STATIONS, max_toffset=5.0)

    assert ignoredStations(traj) == ['C']
    assert abs(time_offsets['B']) < 1e-3



def testHealthyStationIsNotIgnoredWithTheBridgeStation():
    """ Station D overlaps only with B, whose clock is 20 s ahead. Ignoring B would make D lose its overlap
        and be ignored too, so the solver must not ignore a station for its clock in that case.
    """

    traj, _ = solveWithClockError('B', 20.0, stations=BRIDGE_STATIONS, max_toffset=5.0)

    assert 'D' not in ignoredStations(traj)



def testBridgeBetweenTwoGroupsIsNotIgnored():
    """ Station B, whose clock is 20 s ahead, is the only overlap between A and C at the beginning and D and E
        at the end, whose clocks are 3 s ahead. Without B the clocks of the two groups are not tied together
        and the 3 s would stay in the velocity fit, so B must not be ignored for its clock.
    """

    traj = observeMeteor(METEOR, noise=0.0, stations=TWO_GROUP_STATIONS, \
        clock_offsets={'B': 20.0, 'D': 3.0, 'E': 3.0}, max_toffset=5.0)
    traj.run()

    assert 'B' not in ignoredStations(traj)



def testBridgeAcrossAnExcludedRangeIsNotIgnored():
    """ Stations A and C exclude the middle 60% of the meteor, where D and E see it, so D and E are compared
        only to B, whose clock is 20 s ahead. Without B the clocks of D and E, 3 s ahead, are not tied to
        those of A and C, so B must not be ignored for its clock.
    """

    traj = observeMeteor(METEOR, noise=0.0, stations=EXCLUDED_MIDDLE_STATIONS, \
        clock_offsets={'B': 20.0, 'D': 3.0, 'E': 3.0}, max_toffset=5.0)

    # Ignore and exclude the middle 60% of A and C
    duration = METEOR.t_vis_end - METEOR.t_vis_beg
    obs_list = copy.deepcopy(traj.observations)
    for obs in obs_list:
        if obs.station_id in ['A', 'C']:
            middle = (obs.time_data > 0.2*duration) & (obs.time_data < 0.8*duration)
            obs.ignore_list = middle.astype(np.uint8)
            obs.excluded_time = [0.2*duration, 0.8*duration]

    # Initialize the observations again
    traj.observations = []
    for obs in obs_list:
        traj.infillWithObs(obs)

    traj.run()

    assert 'B' not in ignoredStations(traj)



if __name__ == "__main__":

    # Run all tests
    test_list = [testStationBeyondTheLimitIsIgnored, testReferenceStationBeyondTheLimitIsIgnored, \
        testStationWithinTheLimitIsKept, testStationAtTheLimitIsKept, \
        testStationOffTheOtherWayIsNotTakenForTheFaultyOne, \
        testHealthyStationIsNotIgnoredWithTheBridgeStation, testBridgeBetweenTwoGroupsIsNotIgnored, \
        testBridgeAcrossAnExcludedRangeIsNotIgnored]

    for test in test_list:
        test()
        print("PASS {:s}".format(test.__name__))

    print("All tests passed.")
