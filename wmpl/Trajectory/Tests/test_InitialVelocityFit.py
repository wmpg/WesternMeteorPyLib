""" Tests of the initial velocity fit without the ignored stations and picks, of its standard deviation, and
    of the lengths signed along the motion.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_InitialVelocityFit.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np
import scipy.optimize

import matplotlib
matplotlib.use('Agg')

from wmpl.Trajectory.Trajectory import Trajectory, jacchiaLagFunc
from wmpl.Trajectory.Tests.SyntheticTrajectory import STRAIGHT_JDT_REF, STRAIGHT_STATIONS, \
    STRAIGHT_V_INIT, observeStraightMeteor, stationAzimAlt, straightMeteorPoints
from wmpl.Utils.Math import medianStddev


# Noise of the measurements, 20 arcsec (rad)
NOISE = np.radians(20.0/3600)



def observePoints(points, times, seed=1, edit=None, **kwargs):
    """ Observe the given points from the stations of the straight meteor of SyntheticTrajectory.

    Arguments:
        points: [ndarray] ECI coordinates of the points (m), one row per point.
        times: [ndarray] Times of the points (s).

    Keyword arguments:
        seed: [int] Seed of the noise. 1 by default.
        edit: [function] Called with (station_id, azim, elev, times) for every station, returns them changed.
            None by default.
        kwargs: [dict] Passed to the Trajectory object.

    Return:
        [Trajectory object] Trajectory object with the observations, not yet solved.
    """

    rng = np.random.default_rng(seed)

    traj_kwargs = dict(meastype=2, max_toffset=1.0, gravity_correction=False, monte_carlo=False, \
        show_plots=False, save_results=False, verbose=False)
    traj_kwargs.update(kwargs)
    traj = Trajectory(STRAIGHT_JDT_REF, **traj_kwargs)

    for station_id, s_lat, s_lon, ele in STRAIGHT_STATIONS:

        # Compute the noisy directions of the points as seen from the station
        jds = STRAIGHT_JDT_REF + times/86400.0
        azim, elev = stationAzimAlt(points, jds, s_lat, s_lon, ele)
        azim = azim + rng.normal(0, NOISE, len(azim))/np.cos(elev)
        elev = elev + rng.normal(0, NOISE, len(elev))
        station_times = np.copy(times)

        # Edit the observations of the station
        ignore_list = None
        if edit is not None:
            azim, elev, station_times, ignore_list = edit(station_id, azim, elev, station_times)

        traj.infillTrajectory(azim, elev, station_times, np.radians(s_lat), np.radians(s_lon), ele, \
            station_id=station_id, ignore_list=ignore_list)

    return traj



def testVelocityStddevScalesWithTheNoise():
    """ Check that four times the noise gives four times the standard deviation of the initial velocity. """

    for seed in [1, 2]:

        sigmas = []
        for noise_arcsec in [10.0, 40.0]:

            traj, _, _ = observeStraightMeteor(noise=np.radians(noise_arcsec/3600), seed=seed)
            traj = traj.run()
            sigmas.append(traj.v_init_stddev)

        assert 3.8 < sigmas[1]/sigmas[0] < 4.2



def testIgnoredStationDoesNotSetTheMedianVelocity():
    """ Check that without timing estimation the initial velocity is the median of the used stations only. """

    times, points, _, _ = straightMeteorPoints()

    # Station C runs its clock 30% slow and is ignored
    def editStation(station_id, azim, elev, station_times):

        ignore_list = None
        if station_id == 'C':
            station_times = 1.3*station_times
            ignore_list = np.ones(len(station_times), dtype=np.uint8)

        return azim, elev, station_times, ignore_list

    traj = observePoints(points, times, edit=editStation, estimate_timing_vel=False).run()

    used = [obs for obs in traj.observations if not obs.ignore_station]
    assert [obs.station_id for obs in used] == ['A', 'B']

    assert abs(traj.v_init - np.median([obs.v_init for obs in used])) < 1e-6
    assert abs(traj.v_init_stddev - medianStddev([obs.v_init_stddev for obs in used])) < 1e-9
    assert abs(traj.v_init - STRAIGHT_V_INIT) < 100.0



def testIgnoredPicksDoNotSetTheStationVelocity():
    """ Check that the velocity of a station is fitted without its ignored picks. """

    times, points, _, _ = straightMeteorPoints()

    # Make the edit of the stations, with the moved picks ignored or not
    def corruptPicks(ignore):

        # Move the first 3 picks of station A by 0.3 deg in elevation
        def editStation(station_id, azim, elev, station_times):

            ignore_list = None
            if station_id == 'A':
                elev = np.copy(elev)
                elev[:3] += np.radians(0.3)
                if ignore:
                    ignore_list = np.zeros(len(station_times), dtype=np.uint8)
                    ignore_list[:3] = 1

            return azim, elev, station_times, ignore_list

        return editStation

    # Solve with the clean picks, and with the moved picks ignored
    clean = observePoints(points, times, estimate_timing_vel=False, filter_picks=False).run()
    ignored = observePoints(points, times, edit=corruptPicks(True), estimate_timing_vel=False, \
        filter_picks=False).run()

    v_clean = clean.observations[0].v_init
    v_ignored = ignored.observations[0].v_init

    # The velocity without the 3 picks changes only within the noise
    assert abs(v_ignored - v_clean) < 3*ignored.observations[0].v_init_stddev

    # The Jacchia fit of the station also leaves them out
    obs = ignored.observations[0]
    used = obs.ignore_list == 0
    jacchia_fit, _ = scipy.optimize.curve_fit(jacchiaLagFunc, obs.time_data[used], obs.lag[used], \
        p0=np.zeros(2))
    assert np.allclose(obs.jacchia_fit, np.abs(jacchia_fit), rtol=1e-9, atol=1e-12)



def testLengthsAreSignedAlongTheMotion():
    """ Check that a point which projects before the first point of its station has a negative length. """

    times, points, _, _ = straightMeteorPoints()

    # Swap the directions of the first two points of station A, keeping their times
    def editStation(station_id, azim, elev, station_times):

        if station_id == 'A':
            azim, elev = np.copy(azim), np.copy(elev)
            azim[[0, 1]] = azim[[1, 0]]
            elev[[0, 1]] = elev[[1, 0]]

        return azim, elev, station_times, None

    traj = observePoints(points, times, edit=editStation, estimate_timing_vel=False, filter_picks=False).run()

    obs = traj.observations[0]
    spacing = STRAIGHT_V_INIT*(times[1] - times[0])

    assert obs.length[1] < 0
    assert abs(obs.length[1] + spacing) < 0.2*spacing



def testAverageVelocitiesAreTheLineFitsOfThePreviousPoints():
    """ Check that the average velocity up to each point is the slope of the least squares line through the
        first 4 points, or through all points up to it after the first 4.
    """

    traj, _, _ = observeStraightMeteor(noise=NOISE, seed=1)
    traj = traj.run()

    # Compare with the slopes of the lines fitted by numpy
    for obs in traj.observations:
        for i in range(len(obs.time_data)):
            n = max(i + 1, 4)
            slope = np.polyfit(obs.time_data[:n], obs.length[:n], 1)[0]
            assert abs(obs.velocities_prev_point[i] - slope) < 1e-6*abs(slope)



def testNoTimeBetweenPointsGivesNaNVelocity():
    """ Check that two points at the same time get a NaN point to point velocity instead of a huge one. """

    times, points, _, _ = straightMeteorPoints()

    # Give the 11th point of station B the time of the 10th
    def editStation(station_id, azim, elev, station_times):

        if station_id == 'B':
            station_times = np.copy(station_times)
            station_times[10] = station_times[9]

        return azim, elev, station_times, None

    traj = observePoints(points, times, edit=editStation, estimate_timing_vel=False, filter_picks=False).run()

    velocities = traj.observations[1].velocities
    assert np.isnan(velocities[10])
    assert np.all(np.isfinite(np.delete(velocities, 10)))



def testVelocityAboveHeightWeightsStationsByTheirWeight():
    """ Check that the velocity above a height weights each point by the weight of its station. """

    traj, _, _ = observeStraightMeteor(noise=NOISE, seed=1, v_init_ht=70.0)
    traj = traj.run()

    # Take the used points above the height, with the weights of their stations
    times, dists, weights = [], [], []
    for obs in traj.observations:
        used = (obs.ignore_list == 0) & (obs.model_ht >= 70000.0)
        times.append(obs.time_data[used])
        dists.append(obs.state_vect_dist[used])
        weights.append(np.zeros(np.count_nonzero(used)) + obs.weight)

    times, dists, weights = np.concatenate(times), np.concatenate(dists), np.concatenate(weights)

    # The stations have different weights, so the weighting matters
    assert np.ptp(weights) > 0.5

    # Compare with the weighted line fitted by numpy
    slope = np.polyfit(times, dists, 1, w=np.sqrt(weights))[0]
    assert abs(traj.v_init - slope) < 1e-6*slope



def testVelocityStddevWithoutALineFit():
    """ Check that the standard deviation of the initial velocity is that of the median velocity of the
        stations when no line of the sliding fit is kept.
    """

    times, points, begin, radiant_eci = straightMeteorPoints()

    # Accelerate the meteor, so that every line leaves a lag growing faster than 100 m/s
    lengths = STRAIGHT_V_INIT*times + 0.5*4000.0*times**2
    points = begin - np.outer(lengths, radiant_eci)

    traj = observePoints(points, times).run()

    used = [obs for obs in traj.observations if not obs.ignore_station]
    assert abs(traj.v_init - np.median([obs.v_init for obs in used])) < 1e-6
    assert abs(traj.v_init_stddev - medianStddev([obs.v_init_stddev for obs in used])) < 1e-9




if __name__ == "__main__":

    testVelocityStddevScalesWithTheNoise()
    testIgnoredStationDoesNotSetTheMedianVelocity()
    testIgnoredPicksDoNotSetTheStationVelocity()
    testLengthsAreSignedAlongTheMotion()
    testAverageVelocitiesAreTheLineFitsOfThePreviousPoints()
    testNoTimeBetweenPointsGivesNaNVelocity()
    testVelocityAboveHeightWeightsStationsByTheirWeight()
    testVelocityStddevWithoutALineFit()

    print("All tests passed.")
