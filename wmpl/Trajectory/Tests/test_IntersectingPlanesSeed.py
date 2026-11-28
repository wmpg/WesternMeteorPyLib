""" Tests that the intersecting planes solution fits the plane of a station in any orientation, that the
    ignored picks set neither the beginning, the angular length of a track nor the sign of the radiant, and
    that an ignored pick before the beginning has a negative distance from the state vector.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_IntersectingPlanesSeed.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np

import matplotlib
matplotlib.use('Agg')

from wmpl.Trajectory.Trajectory import ObservedPoints, PlaneIntersection, observedAngularLength
from wmpl.Trajectory.Tests.SyntheticTrajectory import STRAIGHT_DURATION, STRAIGHT_JDT_REF, \
    STRAIGHT_STATIONS, STRAIGHT_V_INIT, observeStraightMeteor, stationAzimAlt
from wmpl.Utils.Math import vectNorm


class FakeLines(object):
    """ Stand-in for the lines of sight of a station. """

    def __init__(self, meas_eci, ignore_list=None):
        """ Make the lines of sight.

        Arguments:
            meas_eci: [ndarray] Unit vectors of the lines of sight, one row per point.

        Keyword arguments:
            ignore_list: [ndarray] 1 for the ignored points. None by default, when all points are used.
        """

        self.meas_eci = meas_eci
        self.ignore_list = np.zeros(len(meas_eci), dtype=np.uint8) if ignore_list is None else ignore_list



def makeArc(axis_a, axis_b, n_points=40, arc=np.radians(15.0)):
    """ Make unit vectors along a great circle spanned by two orthonormal vectors.

    Arguments:
        axis_a: [ndarray] First unit vector of the plane.
        axis_b: [ndarray] Second unit vector of the plane, perpendicular to the first.

    Keyword arguments:
        n_points: [int] Number of points. 40 by default.
        arc: [float] Length of the arc (radians). 15 deg by default.

    Return:
        [ndarray] Unit vectors, one row per point.
    """

    # Space the points evenly along the arc, from an arbitrary start
    phi = 0.3 + np.linspace(0, arc, n_points)

    return np.outer(np.cos(phi), axis_a) + np.outer(np.sin(phi), axis_b)



def testPlaneContainingTheZAxis():
    """ Check that the plane of a track along a great circle through the celestial pole is found. """

    # Take a plane which contains the Z axis
    axis_a = vectNorm(np.array([0.6, 0.8, 0.0]))
    axis_z = np.array([0.0, 0.0, 1.0])
    true_normal = np.cross(axis_a, axis_z)

    # Fit the plane to an arc in it
    normal = ObservedPoints.planeFit(FakeLines(makeArc(axis_a, axis_z)))

    assert abs(abs(np.dot(normal, true_normal)) - 1.0) < 1e-12



def testPlaneInAnyOrientation():
    """ Check that the plane is found for tracks in random orientations. """

    rng = np.random.default_rng(3)

    for _ in range(50):

        # Take a plane in a random orientation
        axis_a = vectNorm(rng.normal(size=3))
        axis_b = vectNorm(np.cross(axis_a, rng.normal(size=3)))
        true_normal = np.cross(axis_a, axis_b)

        # Fit the plane to an arc in it
        normal = ObservedPoints.planeFit(FakeLines(makeArc(axis_a, axis_b)))

        assert abs(abs(np.dot(normal, true_normal)) - 1.0) < 1e-12



def testAngularLengthOfTheUsedPicks():
    """ Check that the angular length of a track is taken between its first and last used picks, and is 0
        rather than NaN for a single direction.
    """

    # Take 11 picks 1 deg apart
    meas_eci = makeArc(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), n_points=11, \
        arc=np.radians(10.0))

    # Ignore the first and the last pick
    ignore_list = np.zeros(11, dtype=np.uint8)
    ignore_list[[0, 10]] = 1
    assert abs(observedAngularLength(FakeLines(meas_eci, ignore_list)) - np.radians(8.0)) < 1e-12

    # Two picks in the same direction
    same = np.repeat(vectNorm(np.array([[0.3, 0.4, 0.5]])), 2, axis=0)
    assert observedAngularLength(FakeLines(same)) == 0.0



def observeWithIgnoredFirstPick(up_dist=None, **kwargs):
    """ Observe the straight meteor with the first pick of station A ignored, and pointed at a point on the
        radiant line if asked.

    Keyword arguments:
        up_dist: [float] Distance of the point of the first pick of station A up the radiant line from the
            beginning (m), negative for a point down the line. None by default, when the pick is not moved.
        **kwargs: [dict] Keyword arguments of the Trajectory object.

    Return:
        (traj, begin, radiant_eci):
            - traj: [Trajectory object] Trajectory object with the observations, not yet solved.
            - begin: [ndarray] ECI coordinates of the beginning of the meteor (m).
            - radiant_eci: [ndarray] True radiant unit vector in ECI coordinates.
    """

    # Observe the straight meteor
    traj, begin, radiant_eci = observeStraightMeteor(noise=np.radians(10.0/3600), seed=2, **kwargs)

    obs = traj.observations[0]

    if up_dist is not None:

        # Point the first pick of station A at the point on the radiant line
        _, s_lat, s_lon, ele = STRAIGHT_STATIONS[0]
        azim, elev = stationAzimAlt(np.array([begin + up_dist*radiant_eci]), \
            np.array([STRAIGHT_JDT_REF + obs.time_data[0]/86400.0]), s_lat, s_lon, ele)
        obs.meas1 = np.array(obs.meas1, dtype=np.float64)
        obs.meas2 = np.array(obs.meas2, dtype=np.float64)
        obs.meas1[0], obs.meas2[0] = azim[0], elev[0]

    # Ignore the first pick of station A
    obs.ignore_list[0] = 1

    # Add the observations again to recompute the lines of sight
    obs_list = traj.observations
    traj.observations = []
    for obs in obs_list:
        traj.infillWithObs(obs)

    return traj, begin, radiant_eci



def testIgnoredFirstPickDoesNotSetTheBeginning():
    """ Check that an ignored first pick which lies before the meteor does not move the state vector, the
        beginning of the trajectory on the radiant line.
    """

    # Solve with the ignored pick moved 5 km up the radiant line and kept in place
    moved = observeWithIgnoredFirstPick(5000.0, estimate_timing_vel=False, filter_picks=False)[0].run()
    kept = observeWithIgnoredFirstPick(estimate_timing_vel=False, filter_picks=False)[0].run()

    assert np.linalg.norm(moved.state_vect_mini - kept.state_vect_mini) < 1.0



def testIgnoredFirstPickDoesNotFlipTheRadiant():
    """ Check that an ignored first pick which lies past the end of the meteor does not flip the radiant of
        the intersecting planes.
    """

    # Point the ignored first pick of station A 20 km down the radiant line from the end
    traj, _, radiant_eci = observeWithIgnoredFirstPick(-(STRAIGHT_V_INIT*STRAIGHT_DURATION + 20000.0))

    # Intersect the planes of stations A and B
    plane_intersection = PlaneIntersection(traj.observations[0], traj.observations[1])

    assert np.dot(plane_intersection.radiant_eci, radiant_eci) > 0.99



def testIgnoredPickBeforeTheStateVectorHasANegativeDistance():
    """ Check that an ignored first pick which lies before the meteor is placed before the state vector in
        the distances from it and in the lag.
    """

    # Solve with the ignored pick moved 5 km up the radiant line
    traj = observeWithIgnoredFirstPick(5000.0)[0].run()

    # Check the distance and the lag of the ignored pick of station A
    obs = traj.observations[0]
    assert abs(obs.state_vect_dist[0] + 5000.0) < 50.0
    assert obs.lag[0] < -4000.0



if __name__ == "__main__":

    testPlaneContainingTheZAxis()
    testPlaneInAnyOrientation()
    testAngularLengthOfTheUsedPicks()
    testIgnoredFirstPickDoesNotSetTheBeginning()
    testIgnoredFirstPickDoesNotFlipTheRadiant()
    testIgnoredPickBeforeTheStateVectorHasANegativeDistance()

    print("All tests passed.")
