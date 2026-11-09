""" Tests that a station which is not used, with its clock far off, does not move the scaled gravity drop of
    a decelerating meteor: the scale, the residuals, the lengths on the dropped line, the drag fit and the
    dynamic mass fit take their times from the first time of the used stations (dropTimeOrigin), as the
    line-of-sight fit does, and the drag fit its beginning point from the used stations.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_DropTimeOriginDeceleration.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np

import matplotlib
matplotlib.use('Agg')

from wmpl.Trajectory.Tests.SyntheticTrajectory import BentDragMeteor, observeMeteor
from wmpl.Utils.DragFit import DragFit
from wmpl.Utils.DynamicMassFit import pointOnTrajectory


# Fireball of the tests: 18 km/s at 45 deg, slowing down to about 10 km/s in 3.3 s
FIREBALL = (18000.0, 120.0, 45.0, 0.01, 0.05, 2.7, 6.0)

# Integrated fireball and solutions, shared by the tests
_METEOR = []
_SOLVED = {}



def _solve(clock_lead):
    """ Solve the noise-free fireball with a fourth station which is not used, with the picks of station A and
        its clock ahead by the given time.

    Arguments:
        clock_lead: [float] How far ahead the clock of the station which is not used is (s).

    Return:
        [Trajectory object] Solved trajectory.
    """

    if clock_lead not in _SOLVED:

        if not _METEOR:
            _METEOR.append(BentDragMeteor(*FIREBALL))

        traj = observeMeteor(_METEOR[0], noise=0.0, gravity_correction=True, filter_picks=False, \
            calc_orbit=False)

        obs = traj.observations[0]
        traj.infillTrajectory(obs.azim_data, obs.elev_data, obs.time_data - clock_lead, obs.lat + 0.01, \
            obs.lon, obs.ele, station_id='D', ignore_station=True)

        _SOLVED[clock_lead] = traj.run()

    return _SOLVED[clock_lead]



def testAClockFarOffDoesNotMoveTheScaledDrop():
    """ The scale of the drop and the solution do not change when the clock of the station which is not used
        is 1 s ahead.
    """

    traj, reference = _solve(1.0), _solve(0.0)

    assert reference.gravity_drop_scale is not None
    assert np.allclose(traj.gravity_drop_scale[0], reference.gravity_drop_scale[0], rtol=0, atol=1e-9)
    assert np.allclose(traj.gravity_drop_scale[1], reference.gravity_drop_scale[1], rtol=0, atol=1e-9)

    assert np.allclose(traj.radiant_eci_mini, reference.radiant_eci_mini, rtol=0, atol=1e-12)
    assert np.allclose(traj.state_vect_mini, reference.state_vect_mini, rtol=0, atol=1e-6)

    # The residuals of the used stations
    for obs, obs_ref in zip(traj.observations[:3], reference.observations[:3]):
        assert np.allclose(obs.h_residuals, obs_ref.h_residuals, rtol=0, atol=1e-6)
        assert np.allclose(obs.v_residuals, obs_ref.v_residuals, rtol=0, atol=1e-6)



def testAClockFarOffDoesNotMoveTheDropOfTheDragAndDynamicMassFits():
    """ The drag fit drops its grid, and the dynamic mass fit its points, from the same time when the clock of
        the station which is not used is 1 s ahead.
    """

    traj, reference = _solve(1.0), _solve(0.0)

    drop_grid = DragFit(traj).drop_grid
    drop_grid_ref = DragFit(reference).drop_grid
    assert np.allclose(drop_grid, drop_grid_ref, rtol=0, atol=1e-3)

    for length, t in ((20e3, 1.0), (40e3, 2.5)):
        point = pointOnTrajectory(traj, length, t)
        assert np.allclose(point, pointOnTrajectory(reference, length, t), rtol=0, atol=1e-6)




if __name__ == "__main__":

    testAClockFarOffDoesNotMoveTheScaledDrop()
    testAClockFarOffDoesNotMoveTheDropOfTheDragAndDynamicMassFits()

    print("All tests passed.")
