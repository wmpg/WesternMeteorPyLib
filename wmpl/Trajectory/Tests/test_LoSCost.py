""" Tests of the cost of the line-of-sight fit, the sum of the angles between the lines of sight and the
line, which is computed for all points at once. It is tested against a computation point by point, without
gravity, with the vertical gravity drop and with the drop across the path, with weights, ignored stations
and ignored points. The cost with the angles smoothed near zero, with which the fit starts (see
test_LoSFitSmoothing.py), is tested to tend to the exact cost.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_LoSCost.py -v
"""

from __future__ import print_function, division, absolute_import

import copy

import numpy as np

import matplotlib
matplotlib.use('Agg')

from wmpl.Trajectory.Trajectory import angleSumMeasurements2Line, applyGravityDrop, dropTimeOrigin, \
    gravityDropOrigin, gravityDropVectors, moveStateVector
from wmpl.Trajectory.Tests.SyntheticTrajectory import observeStraightMeteor
from wmpl.Utils.Math import findClosestPoints, vectMag, vectNorm


def _angleSumOfEachPoint(observations, state_vect, radiant_eci, weights=None, gravity=False, \
    gravity_factor=1.0, v0z=None, r0=None, gravity_vel=None):
    """ Compute the cost of the line-of-sight fit point by point, as a reference for
        angleSumMeasurements2Line, whose arguments it takes.

    Return:
        [float] Weighted mean of the angles between the line and the lines of sight (rad).
    """

    # Use unit weights for the used stations if none are given, or if none is larger than 0
    if (weights is None) or (sum(weights) <= 0):
        weights = [0.0 if obs.ignore_station else 1.0 for obs in observations]

    state_vect = moveStateVector(state_vect, radiant_eci, observations)
    t0 = dropTimeOrigin(observations)

    # Compute the drop across the path at the times of the used points
    drops = None
    if gravity and (gravity_vel is not None):

        times = [np.array(obs.time_data)[np.array(obs.ignore_list) == 0] for obs in observations]
        drop_used = gravityDropVectors(gravityDropOrigin(observations, state_vect, radiant_eci, \
            gravity_vel, t0), radiant_eci, np.concatenate(times) - t0, gravity_vel, t0, gravity_factor)
        drops = np.split(drop_used, np.cumsum([len(t) for t in times])[:-1])

    angle_sum = 0.0
    weights_sum = 1e-10
    for i, obs in enumerate(observations):

        k = 0
        for t, meas_eci, stat_eci, ignore in zip(obs.time_data, obs.meas_eci_los, obs.stat_eci_los, \
            obs.ignore_list):

            if ignore:
                continue

            _, rad_cpa, _ = findClosestPoints(stat_eci, meas_eci, state_vect, radiant_eci)

            # Drop the point across the path
            if drops is not None:
                _, rad_cpa, _ = findClosestPoints(stat_eci, meas_eci, rad_cpa + drops[i][k], radiant_eci)

            # Drop the point vertically
            elif gravity:

                r0_point = vectMag(rad_cpa) if r0 is None else r0
                if r0_point == 0:
                    r0_point = 1.0

                rad_cpa_grav = applyGravityDrop(rad_cpa, t - t0, r0_point, gravity_factor, \
                    0.0 if v0z is None else v0z)
                _, rad_cpa, _ = findClosestPoints(stat_eci, meas_eci, rad_cpa_grav, radiant_eci)

            chord = vectMag(vectNorm(rad_cpa - stat_eci) - meas_eci)
            angle_sum += weights[i]*2*np.arcsin(min(chord/2, 1.0))
            weights_sum += weights[i]
            k += 1

    return angle_sum/weights_sum



def _observedMeteor(seed=20261117):
    """ Observe the straight meteor falling under gravity, with the second station ignored, every seventh
        point of the third station ignored, and a fourth station with all points ignored.

    Keyword arguments:
        seed: [int] Seed of the noise. 20261117 by default.

    Return:
        (traj, begin, radiant):
            - traj: [Trajectory object] Trajectory object with the observations, not solved.
            - begin: [ndarray] ECI coordinates of the beginning of the meteor (m).
            - radiant: [ndarray] True radiant unit vector in ECI coordinates.
    """

    traj, begin, radiant = observeStraightMeteor(seed=seed, gravity_drop=True, calc_orbit=False)

    traj.observations[1].ignore_station = True
    traj.observations[2].ignore_list[::7] = 1

    # Add a station with all points ignored
    traj.observations.append(copy.deepcopy(traj.observations[0]))
    traj.observations[-1].ignore_list[:] = 1

    return traj, begin, radiant



def testTheCostOfAllPointsAtOnceEqualsTheCostOfEachPoint():
    """ Check that the cost computed for all points at once equals the cost computed point by point, without
        gravity, with the vertical drop of both drop models and with the drop across the path.
    """

    traj, begin, radiant = _observedMeteor()

    # Take a line 200 m and 0.02 deg off the meteor, starting before it
    side = vectNorm(np.cross(radiant, [0, 0, 1]))
    state_vect = begin + 200.0*side + 5000.0*radiant
    radiant_eci = vectNorm(radiant + np.tan(np.radians(0.02))*np.cross(radiant, side))

    # Speeds of the meteor against time, slowing down
    gravity_vel = (np.linspace(-0.5, 3.0, 50), 30000.0 - 2000.0*np.linspace(-0.5, 3.0, 50)**2)

    weights = [0.3, 0.0, 0.8, 0.5]
    cases = [dict(), dict(weights=weights), dict(weights=[0.0, 0.0, 0.0, 0.0]), dict(gravity=True), \
        dict(gravity=True, r0=6.45e6, v0z=-2000.0, weights=weights), dict(gravity=True, v0z=-2000.0), \
        dict(gravity=True, v0z=50.0, gravity_factor=1.5), dict(gravity=True, r0=0.0), \
        dict(gravity=True, gravity_vel=gravity_vel, weights=weights)]

    for kwargs in cases:

        cost = angleSumMeasurements2Line(traj.observations, state_vect, radiant_eci, **kwargs)
        reference = _angleSumOfEachPoint(traj.observations, state_vect, radiant_eci, **kwargs)

        assert abs(cost - reference) < 1e-12*reference, kwargs

    # The drop changes the cost
    assert abs(angleSumMeasurements2Line(traj.observations, state_vect, radiant_eci, gravity=True) \
        - angleSumMeasurements2Line(traj.observations, state_vect, radiant_eci)) > 1e-3*cost



def testTheCostIsNotFiniteForALineOfSightAlongTheLine():
    """ Check that the cost is NaN, as point by point, when the line is parallel to a line of sight. """

    traj, begin, radiant = _observedMeteor()

    los = traj.observations[0].meas_eci_los[3]
    state_vect = traj.observations[0].stat_eci_los[3] + 1.0e5*los

    for kwargs in (dict(), dict(gravity=True), dict(gravity=True, v0z=-2000.0)):
        assert np.isnan(angleSumMeasurements2Line(traj.observations, state_vect, los, **kwargs))
        assert np.isnan(_angleSumOfEachPoint(traj.observations, state_vect, los, **kwargs))



def testTheSmoothedCostTendsToTheExactCost():
    """ Check that the smoothed cost is below the exact one by less than the smoothing, and tends to it as
        the smoothing decreases.
    """

    traj, begin, radiant = _observedMeteor()

    exact = angleSumMeasurements2Line(traj.observations, begin, radiant, gravity=True)

    differences = []
    for smoothing in np.radians(np.array([10.0, 1.0, 0.1, 0.01, 0.001])/3600):

        smoothed = angleSumMeasurements2Line(traj.observations, begin, radiant, gravity=True, \
            smoothing=smoothing)

        assert 0 < exact - smoothed < smoothing
        differences.append(exact - smoothed)

    assert np.all(np.diff(differences) < 0)



if __name__ == "__main__":

    # Run all tests
    test_list = [testTheCostOfAllPointsAtOnceEqualsTheCostOfEachPoint, \
        testTheCostIsNotFiniteForALineOfSightAlongTheLine, \
        testTheSmoothedCostTendsToTheExactCost]

    for test in test_list:
        test()
        print("PASS {:s}".format(test.__name__))
