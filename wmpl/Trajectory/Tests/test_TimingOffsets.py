""" Tests of the timing offsets between stations, estimated by linear least squares with one used station
    held at 0, and of the time residuals reported in seconds.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_TimingOffsets.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np

import matplotlib
matplotlib.use('Agg')

from wmpl.Trajectory.Trajectory import Trajectory, stationsWithoutTimeOverlap, timingCost, \
    timingDifferences
from wmpl.Trajectory.Tests.SyntheticTrajectory import STRAIGHT_JDT_REF, STRAIGHT_STATIONS, \
    observeStraightMeteor, stationAzimAlt, straightMeteorPoints, timingOffsets


# Clocks ahead of the true time (s)
CLOCK_OFFSETS = {'A': 0.0, 'B': 0.037, 'C': -0.052}

# Noise of the measurements, 30 arcsec (rad)
NOISE = np.radians(30.0/3600)



def solveWithClockOffsets(**kwargs):
    """ Solve the straight meteor of SyntheticTrajectory with clock offsets.

    Keyword arguments:
        kwargs: [dict] Passed to observeStraightMeteor.

    Return:
        [Trajectory object] Solved trajectory.
    """

    kwargs.setdefault('clock_offsets', CLOCK_OFFSETS)

    traj, _, _ = observeStraightMeteor(noise=NOISE, seed=3, **kwargs)

    return traj.run()



def stationWeights(traj):
    """ Take the weights of the stations used by the solver.

    Arguments:
        traj: [Trajectory object] Solved trajectory.

    Return:
        [list] Weight of every station.
    """

    return [obs.weight for obs in traj.observations]



def testCostStatistics():
    """ Check that every point of every pair is averaged once, with the weight of its pair. """

    pairs = [(0, 1, np.array([0.001, -0.002, 0.003]), 1.0), (0, 2, np.array([0.004, 0.001]), 0.5), \
        (1, 2, np.array([-0.001, 0.002, 0.0, 0.001]), 0.25)]
    offsets = np.array([0.0, 0.001, -0.002])

    # Compute the statistics from all points at once
    residuals = [diffs + offsets[i] - offsets[j] for i, j, diffs, _ in pairs]
    weights = [np.zeros_like(res) + weight for res, (_, _, _, weight) in zip(residuals, pairs)]
    residuals, weights = np.concatenate(residuals), np.concatenate(weights)

    mean_square = np.sum(weights*residuals**2)/np.sum(weights)

    assert abs(timingCost(pairs, offsets) - mean_square) < 1e-18
    assert abs(timingCost(pairs, offsets, stat='rms') - np.sqrt(mean_square)) < 1e-15
    assert abs(timingCost(pairs, offsets, stat='mean_abs') \
        - np.sum(weights*np.abs(residuals))/np.sum(weights)) < 1e-15



def testOffsetsAreTheLeastSquaresMinimum():
    """ Check that the clock offsets are found at the least squares minimum, and that the reported residuals
        are their root mean square in seconds.
    """

    traj = solveWithClockOffsets()

    # A clock ahead by dt gets an offset of -dt, relative to station A
    offsets = timingOffsets(traj)
    for station_id, clock_offset in CLOCK_OFFSETS.items():
        assert abs(offsets[station_id] + clock_offset) < 1e-3

    # The offsets are applied, so the residuals at zero offsets are the final ones
    pairs = timingDifferences(traj.observations, weights=stationWeights(traj))
    zero = np.zeros(len(traj.observations))
    assert abs(traj.timing_stddev - timingCost(pairs, zero, stat='rms')) < 1e-12
    assert abs(traj.timing_avg_res - timingCost(pairs, zero, stat='mean_abs')) < 1e-12

    # Shifting the offset of any station from the minimum increases the cost
    base = timingCost(pairs, zero)
    for k in range(len(traj.observations)):
        for shift in [-1e-5, 1e-5]:
            shifted = np.copy(zero)
            shifted[k] = shift
            assert timingCost(pairs, shifted) > base



def testReferenceStationIsAUsedStation():
    """ Check that the station held at a zero offset is not an ignored one. """

    traj, _, _ = observeStraightMeteor(noise=NOISE, seed=3, clock_offsets=CLOCK_OFFSETS)

    # Ignore station A, which has as many points as the others and comes first
    traj.observations[0].ignore_station = True
    traj.observations[0].ignore_list[:] = 1

    traj = traj.run()

    held = [station_id for station_id, value in traj.stations_time_dict.items() \
        if not isinstance(value, bool)]
    assert held == ['B']

    # The offset between B and C is found
    offsets = timingOffsets(traj)
    assert abs((offsets['C'] - offsets['B']) - (CLOCK_OFFSETS['B'] - CLOCK_OFFSETS['C'])) < 1e-3



def testFixedOffsetOfAnAbsentStation():
    """ Check that a fixed offset of a station which did not observe the meteor does not leave the estimated
        offsets without a station held at zero.
    """

    traj = solveWithClockOffsets(fixed_times='X:0.1')

    held = [station_id for station_id, value in traj.stations_time_dict.items() \
        if not isinstance(value, bool)]
    assert len(held) == 1
    assert held[0] in ['A', 'B', 'C']

    offsets = timingOffsets(traj)
    for station_id, clock_offset in CLOCK_OFFSETS.items():
        assert abs(offsets[station_id] + clock_offset) < 1e-3



def observeSeparately(estimate_timing_vel, exclude_end=False):
    """ Observe the straight meteor with stations A and B seeing its first part and station C its last part,
        so that the lengths of C do not overlap with those of A or B.

    Arguments:
        estimate_timing_vel: [bool] Passed to the Trajectory object.

    Keyword arguments:
        exclude_end: [bool] Let station A see the whole meteor with its last part excluded, so that C overlaps
            with A only inside the exclusion zone. False by default.

    Return:
        [Trajectory object] Trajectory object with the observations, not yet solved.
    """

    rng = np.random.default_rng(5)

    times, points, _, _ = straightMeteorPoints()

    traj = Trajectory(STRAIGHT_JDT_REF, meastype=2, max_toffset=1.0, gravity_correction=False, \
        monte_carlo=False, show_plots=False, save_results=False, verbose=False, \
        estimate_timing_vel=estimate_timing_vel)

    for station_id, s_lat, s_lon, ele in STRAIGHT_STATIONS:

        excluded_time = None

        # Take the first part of the meteor for A and B, and the last part for C
        if station_id == 'C':
            sel = slice(26, None)
        else:
            sel = slice(0, 24)

        # Let A see the whole meteor and exclude its last part
        if exclude_end and (station_id == 'A'):
            sel = slice(0, None)
            excluded_time = [times[24], times[-1]]

        # Compute the noisy directions of the points as seen from the station
        jds = STRAIGHT_JDT_REF + times[sel]/86400.0
        azim, elev = stationAzimAlt(points[sel], jds, s_lat, s_lon, ele)
        azim = azim + rng.normal(0, NOISE, len(azim))/np.cos(elev)
        elev = elev + rng.normal(0, NOISE, len(elev))

        traj.infillTrajectory(azim, elev, times[sel], np.radians(s_lat), np.radians(s_lon), ele, \
            station_id=station_id, excluded_time=excluded_time)

    return traj



def testStationWithoutOverlapIsKeptWithoutTimingEstimation():
    """ Check that a station which overlaps with no other one is kept when the timing is not estimated. """

    traj = observeSeparately(estimate_timing_vel=False).run()

    assert not traj.observations[2].ignore_station
    assert np.isfinite(traj.timing_res)



def testStationWithoutOverlapIsIgnoredWithTimingEstimation():
    """ Check that a station which overlaps with no other one is ignored when the timing is estimated. """

    traj = observeSeparately(estimate_timing_vel=True).run()

    assert traj.observations[2].ignore_station
    assert not traj.observations[0].ignore_station
    assert not traj.observations[1].ignore_station



def testOverlapInsideExclusionZoneIsNotCounted():
    """ Check that the points inside the exclusion zone of a station are not counted in its overlap with
        another station.
    """

    traj = observeSeparately(estimate_timing_vel=False, exclude_end=True).run()

    # C overlaps with A only inside the exclusion zone of A
    pairs = timingDifferences(traj.observations)
    assert min(len(diffs) for _, _, diffs, _ in pairs) >= 4
    assert stationsWithoutTimeOverlap(traj.observations, pairs) == [2]



def testStationOverlappingInsideExclusionZoneIsIgnored():
    """ Check that a station which overlaps with another one only inside its exclusion zone is ignored when
        the timing is estimated.
    """

    traj = observeSeparately(estimate_timing_vel=True, exclude_end=True).run()

    assert traj.observations[2].ignore_station
    assert not traj.observations[0].ignore_station
    assert not traj.observations[1].ignore_station




if __name__ == "__main__":

    testCostStatistics()
    testOffsetsAreTheLeastSquaresMinimum()
    testReferenceStationIsAUsedStation()
    testFixedOffsetOfAnAbsentStation()
    testStationWithoutOverlapIsKeptWithoutTimingEstimation()
    testStationWithoutOverlapIsIgnoredWithTimingEstimation()
    testOverlapInsideExclusionZoneIsNotCounted()
    testStationOverlappingInsideExclusionZoneIsIgnored()

    print("All tests passed.")
