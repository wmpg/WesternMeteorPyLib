""" Tests of the time from which wmpl.Trajectory.Trajectory computes the gravity drop: the first time of the
    stations which are used, so that a station which is not used, e.g. because its clock is far off, does not
    move the drop of the others.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_DropTimeOrigin.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np

import matplotlib
matplotlib.use('Agg')

from wmpl.Trajectory.Trajectory import Trajectory, dropTimeOrigin
from wmpl.Trajectory.Tests.SyntheticTrajectory import observeStraightMeteor
from wmpl.Utils.DragFit import DragFit
from wmpl.Utils.DynamicMassFit import pointOnTrajectory


class _Station(object):
    def __init__(self, time_data, ignore_station):
        """ The attributes of a station which dropTimeOrigin reads.

        Arguments:
            time_data: [ndarray] Times of the points (s).
            ignore_station: [bool] True if the station is not used.
        """

        self.time_data = np.array(time_data)
        self.ignore_station = ignore_station



def testOriginIsTheFirstTimeOfTheUsedStations():
    """ Check that the stations which are not used do not set the origin, unless no station is used. """

    stations = [_Station([0.1, 0.2], False), _Station([-1.0, 0.0], True), _Station([0.05, 0.3], False)]
    assert dropTimeOrigin(stations) == 0.05

    for station in stations:
        station.ignore_station = True

    assert dropTimeOrigin(stations) == -1.0



def _solve(clock_lead):
    """ Solve a noise-free straight meteor dropped by gravity, with a fourth station which is not used and
        whose clock is ahead by the given time.

    Arguments:
        clock_lead: [float] How far ahead the clock of the station which is not used is (s).

    Return:
        (traj, radiant_eci):
            - traj: [Trajectory object] Solved trajectory.
            - radiant_eci: [ndarray] True radiant unit vector in ECI coordinates.
    """

    observed, _, radiant_eci = observeStraightMeteor(noise=0.0, duration=3.0, v_init=15000.0, \
        elevation=30.0, gravity_drop=True)

    traj = Trajectory(observed.jdt_ref, meastype=2, max_toffset=1.0, estimate_timing_vel=False, \
        gravity_correction=True, filter_picks=False, monte_carlo=False, calc_orbit=False, show_plots=False, \
        save_results=False, verbose=False)

    for obs in observed.observations:
        traj.infillTrajectory(obs.azim_data, obs.elev_data, obs.time_data, obs.lat, obs.lon, obs.ele, \
            station_id=obs.station_id)

    # Add a station which is not used, with the picks of station A and its clock ahead
    obs = observed.observations[0]
    traj.infillTrajectory(obs.azim_data, obs.elev_data, obs.time_data - clock_lead, obs.lat + 0.01, obs.lon, \
        obs.ele, station_id='D', ignore_station=True)

    return traj.run(), radiant_eci



def testStationWithAClockFarOffDoesNotMoveTheDrop():
    """ Check that a station which is not used and whose clock is 1 s ahead does not change the solution.
        When its first time set the origin of the drop, the radiant was 115 arcsec off and the residuals of
        the used stations reached 22 m.
    """

    traj, radiant_eci = _solve(1.0)
    reference, _ = _solve(0.0)

    # The solution and the residuals of the used stations are those with the clock set
    assert np.allclose(traj.radiant_eci_mini, reference.radiant_eci_mini, rtol=0, atol=1e-12)
    for obs, ref_obs in zip(traj.observations[:3], reference.observations[:3]):
        assert np.allclose(obs.h_residuals, ref_obs.h_residuals, rtol=0, atol=1e-6)
        assert np.allclose(obs.v_residuals, ref_obs.v_residuals, rtol=0, atol=1e-6)

    # The radiant is found, to the difference between the drop of the synthetic meteor and the solver's
    assert np.allclose(traj.radiant_eci_mini, radiant_eci, rtol=0, atol=1e-5)



def testDragAndDynamicMassFitsDropFromTheSameTime():
    """ Check that the drag fit drops its grid, and the dynamic mass fit its points, as the solver does when
        the clock of the station which is not used is 1 s ahead.
    """

    traj, _ = _solve(1.0)
    reference, _ = _solve(0.0)

    assert np.allclose(DragFit(traj).drop_grid, DragFit(reference).drop_grid, rtol=0, atol=1e-3)

    for length, t in ((20e3, 1.0), (35e3, 2.5)):
        assert np.allclose(pointOnTrajectory(traj, length, t), pointOnTrajectory(reference, length, t), \
            rtol=0, atol=1e-6)
