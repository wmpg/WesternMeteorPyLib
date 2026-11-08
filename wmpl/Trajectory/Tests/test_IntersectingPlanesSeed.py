""" Tests of the intersecting planes solution which seeds the line of sight fit in wmpl.Trajectory.Trajectory:
    the plane of each station is fitted through it in any orientation, and the ignored picks do not set the
    beginning of the trajectory or the angular length of a track.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_IntersectingPlanesSeed.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np

import matplotlib
matplotlib.use('Agg')

from wmpl.Trajectory.Trajectory import ObservedPoints, observedAngularLength
from wmpl.Trajectory.Tests.SyntheticTrajectory import STRAIGHT_JDT_REF, STRAIGHT_STATIONS, \
    observeStraightMeteor, stationAzimAlt
from wmpl.Utils.Math import vectNorm


class _Lines(object):
    def __init__(self, meas_eci, ignore_list=None):
        """ The attributes of a station which planeFit and observedAngularLength read.

        Arguments:
            meas_eci: [ndarray] Unit vectors of the lines of sight, one row per point.

        Keyword arguments:
            ignore_list: [ndarray] 1 for the ignored points. None by default, when all points are used.
        """

        self.meas_eci = meas_eci
        self.ignore_list = np.zeros(len(meas_eci), dtype=np.uint8) if ignore_list is None else ignore_list



def _arc(axis_a, axis_b, n_points=40, arc=np.radians(15.0)):
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

    phi = 0.3 + np.linspace(0, arc, n_points)

    return np.outer(np.cos(phi), axis_a) + np.outer(np.sin(phi), axis_b)



def testPlaneContainingTheZAxis():
    """ Check that the plane of a track along a great circle through the celestial pole is found. """

    axis_a = vectNorm(np.array([0.6, 0.8, 0.0]))
    axis_z = np.array([0.0, 0.0, 1.0])
    true_normal = np.cross(axis_a, axis_z)

    normal = ObservedPoints.planeFit(_Lines(_arc(axis_a, axis_z)))

    assert abs(abs(np.dot(normal, true_normal)) - 1.0) < 1e-12



def testPlaneInAnyOrientation():
    """ Check that the plane is found for tracks in random orientations. """

    rng = np.random.default_rng(3)

    for _ in range(50):

        axis_a = vectNorm(rng.normal(size=3))
        axis_b = vectNorm(np.cross(axis_a, rng.normal(size=3)))
        true_normal = np.cross(axis_a, axis_b)

        normal = ObservedPoints.planeFit(_Lines(_arc(axis_a, axis_b)))

        assert abs(abs(np.dot(normal, true_normal)) - 1.0) < 1e-12



def testAngularLengthOfTheUsedPicks():
    """ Check that the angular length of a track is taken between its first and last used picks, and is 0
        rather than NaN for a single direction.
    """

    meas_eci = _arc(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), n_points=11, arc=np.radians(10.0))

    ignore_list = np.zeros(11, dtype=np.uint8)
    ignore_list[[0, 10]] = 1
    assert abs(observedAngularLength(_Lines(meas_eci, ignore_list)) - np.radians(8.0)) < 1e-12

    same = np.repeat(vectNorm(np.array([[0.3, 0.4, 0.5]])), 2, axis=0)
    assert observedAngularLength(_Lines(same)) == 0.0



def testIgnoredFirstPickDoesNotSetTheBeginning():
    """ Check that an ignored first pick which lies before the meteor does not move the state vector, the
        beginning of the trajectory on the radiant line.
    """

    def _solve(move_first):

        traj, begin, radiant_eci = observeStraightMeteor(noise=np.radians(10.0/3600), seed=2, \
            estimate_timing_vel=False, filter_picks=False)

        obs = traj.observations[0]

        if move_first:

            # Point the first pick of station A at a point 5 km up the radiant line from the beginning
            _, s_lat, s_lon, ele = STRAIGHT_STATIONS[0]
            azim, elev = stationAzimAlt(np.array([begin + 5000.0*radiant_eci]), \
                np.array([STRAIGHT_JDT_REF + obs.time_data[0]/86400.0]), s_lat, s_lon, ele)
            obs.meas1 = np.array(obs.meas1, dtype=np.float64)
            obs.meas2 = np.array(obs.meas2, dtype=np.float64)
            obs.meas1[0], obs.meas2[0] = azim[0], elev[0]

        obs.ignore_list[0] = 1

        obs_list = traj.observations
        traj.observations = []
        for obs in obs_list:
            traj.infillWithObs(obs)

        return traj.run()

    moved = _solve(True)
    kept = _solve(False)

    assert np.linalg.norm(moved.state_vect_mini - kept.state_vect_mini) < 1.0
