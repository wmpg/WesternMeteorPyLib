""" Tests of the compiled loop of the line-of-sight cost (angleSumPoints).

The loop over the points of angleSumMeasurements2Line is compiled with numba, and must give the same sum to
the last bit as the loop in Python. The tests compare the cost with that loop, copied here, on lines around a
meteor seen from three stations, without and with the gravity drop, with a fixed and a free distance of the
beginning, a scaled drop, a vertical velocity, weights and ignored points.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_AngleSumPoints.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np

import matplotlib
matplotlib.use('Agg')

from wmpl.Trajectory.Trajectory import angleSumMeasurements2Line, applyGravityDrop, moveStateVector
from wmpl.Trajectory.Tests.SyntheticTrajectory import observeStraightMeteor
from wmpl.Utils.Math import findClosestPoints, vectNorm, vectMag



def angleSumPython(observations, state_vect, radiant_eci, weights=None, gravity=False, gravity_factor=1.0, \
    v0z=None, r0=None, drop_scale=None):
    """ Compute the cost of angleSumMeasurements2Line with the loop over the points in Python. The arguments
        are those of angleSumMeasurements2Line.

    Return:
        [float] Weighted mean of the angles (radians).
    """

    # Use the stations which are not ignored
    if weights is None:
        weights = np.array([0 if obs.ignore_station else 1.0 for obs in observations])

    state_vect = moveStateVector(state_vect, radiant_eci, observations)
    radiant_eci = np.ascontiguousarray(radiant_eci)
    t0 = min([obs.time_data[0] for obs in observations])

    angle_sum = 0.0
    weights_sum = 1e-10

    for i, obs in enumerate(observations):

        # Compute the factor of the gravity drop of every point
        point_factors = np.full(len(obs.time_data), gravity_factor)
        if gravity and (drop_scale is not None):
            point_factors = gravity_factor*np.interp(obs.time_data - t0, *drop_scale)

        # Go through all used points
        for t, meas_eci, stat_eci, ignore, point_factor in zip(obs.time_data, obs.meas_eci_los, \
            obs.stat_eci_los, obs.ignore_list, point_factors):

            if ignore:
                continue

            _, rad_cpa, _ = findClosestPoints(stat_eci, meas_eci, state_vect, radiant_eci)

            # Take the gravity drop into account
            if gravity:

                t_rel = t - t0

                if v0z is None:
                    v0z = 0.0

                r0_point = r0
                if r0_point is None:
                    r0_point = vectMag(rad_cpa)

                if r0_point == 0:
                    r0_point = 1.0

                rad_cpa_grav = applyGravityDrop(rad_cpa, t_rel, r0_point, point_factor, v0z)
                _, rad_cpa, _ = findClosestPoints(stat_eci, meas_eci, rad_cpa_grav, radiant_eci)

            # Add the angle between the line of sight and the line
            station_ray = vectNorm(rad_cpa - stat_eci)
            cosangle = np.dot(meas_eci, station_ray)
            angle_sum += weights[i]*np.arccos(np.clip(cosangle, -1, 1))
            weights_sum += weights[i]

    return angle_sum/weights_sum



def makeObservations():
    """ Make the observations of a meteor dropped by gravity, seen from three stations, and the true line.

    Return:
        (observations, begin, radiant_eci):
            - observations: [list] A list of ObservedPoints objects.
            - begin: [ndarray] ECI coordinates of the beginning of the meteor (m).
            - radiant_eci: [ndarray] Unit radiant vector.
    """

    traj, begin, radiant_eci = observeStraightMeteor(noise=5e-5, seed=3, duration=4.0, v_init=15000.0, \
        elevation=30.0, gravity_drop=True)

    # Shift the times so that they do not start at 0
    for obs in traj.observations:
        obs.time_data = obs.time_data + 0.37

    return traj.observations, begin, radiant_eci



def makeCases(observations, begin):
    """ Make the keyword arguments of the cost which take every branch of the loop.

    Arguments:
        observations: [list] A list of ObservedPoints objects.
        begin: [ndarray] ECI coordinates of the beginning of the meteor (m).

    Return:
        [list] Dictionaries of keyword arguments.
    """

    r0 = vectMag(begin)
    t_end = max(np.max(obs.time_data) for obs in observations) - min(obs.time_data[0] for obs in observations)
    drop_scale = (np.linspace(0, t_end, 20), np.linspace(1.0, 0.7, 20))

    return [dict(), dict(gravity=True), dict(gravity=True, r0=r0, v0z=0.0, drop_scale=drop_scale), \
        dict(gravity=True, r0=r0, v0z=-7500.0, drop_scale=drop_scale, gravity_factor=0.8), \
        dict(gravity=True, r0=r0, weights=np.array([1.0, 2.5, 0.5]))]



def testCompiledCostIsTheSameAsInPython():
    """ Check that the cost with the compiled loop is the same to the last bit as with the loop in Python,
        around the true line and away from it.
    """

    observations, begin, radiant_eci = makeObservations()
    rng = np.random.default_rng(1)

    for kwargs in makeCases(observations, begin):
        for k in range(20):

            # The true line, then lines moved by up to a few km and a few tenths of a degree
            state_vect = begin + (k > 0)*rng.normal(0, 2000.0, 3)
            radiant = vectNorm(radiant_eci + (k > 0)*rng.normal(0, 3e-3, 3))

            compiled = angleSumMeasurements2Line(observations, state_vect, radiant, **kwargs)
            python = angleSumPython(observations, state_vect, radiant, **kwargs)

            assert compiled == python, (kwargs.keys(), k, compiled, python)



def testIgnoredPointsAndStationsAreLeftOut():
    """ Check that the ignored points and stations are left out of the sum, as in the loop in Python. """

    observations, begin, radiant_eci = makeObservations()

    # Ignore every third point of the first station and the whole last station
    observations[0].ignore_list[::3] = 1
    observations[-1].ignore_station = True

    state_vect = begin + np.array([300.0, -200.0, 100.0])
    radiant = vectNorm(radiant_eci + np.array([1e-3, -2e-3, 5e-4]))

    for kwargs in makeCases(observations, begin):

        compiled = angleSumMeasurements2Line(observations, state_vect, radiant, **kwargs)
        python = angleSumPython(observations, state_vect, radiant, **kwargs)

        assert compiled == python, (kwargs.keys(), compiled, python)

    # Ignoring the points changes the cost
    ignored = angleSumMeasurements2Line(observations, state_vect, radiant)
    observations[0].ignore_list[:] = 0
    assert angleSumMeasurements2Line(observations, state_vect, radiant) != ignored



def testCosinesOutOfRangeAreClippedAsInPython():
    """ Check that the cosines above 1 and below -1 are clipped as np.clip does, and that a NaN stays NaN. """

    observations, begin, radiant_eci = makeObservations()

    # Lines of sight which are not unit vectors give cosines out of the range
    observations[0].meas_eci_los = 1.5*observations[0].meas_eci_los
    observations[1].meas_eci_los = -1.5*observations[1].meas_eci_los

    for kwargs in makeCases(observations, begin):

        compiled = angleSumMeasurements2Line(observations, begin, radiant_eci, **kwargs)
        python = angleSumPython(observations, begin, radiant_eci, **kwargs)

        assert np.isfinite(compiled)
        assert compiled == python, (kwargs.keys(), compiled, python)

    # A radiant line along a line of sight has no closest point, and the cost is not a number
    observations, begin, radiant_eci = makeObservations()
    radiant = observations[0].meas_eci_los[0]

    assert np.isnan(angleSumMeasurements2Line(observations, begin, radiant))
    assert np.isnan(angleSumPython(observations, begin, radiant))



if __name__ == "__main__":

    testCompiledCostIsTheSameAsInPython()
    testIgnoredPointsAndStationsAreLeftOut()
    testCosinesOutOfRangeAreClippedAsInPython()

    print("All tests passed.")
