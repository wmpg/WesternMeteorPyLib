""" Tests that a station which is not used, with its clock far off, does not move the scaled gravity drop of
    a decelerating meteor, nor the drop of the drag fit and the dynamic mass fit.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_DropTimeOriginDeceleration.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np

import matplotlib
matplotlib.use('Agg')

from wmpl.Trajectory.Tests.SyntheticTrajectory import DECELERATING_FIREBALL, bentDragMeteor, observeMeteor
from wmpl.Utils.DragFit import DragFit
from wmpl.Utils.DynamicMassFit import pointOnTrajectory


# Solutions, shared by the tests
SOLVED_CACHE = {}



def solveWithIgnoredStation(clock_lead):
    """ Solve the noise-free fireball with a fourth station which is not used, with the picks of station A and
        its clock ahead by the given time.

    Arguments:
        clock_lead: [float] How far ahead the clock of the station which is not used is (s).

    Return:
        [Trajectory object] Solved trajectory.
    """

    if clock_lead not in SOLVED_CACHE:

        traj = observeMeteor(bentDragMeteor(*DECELERATING_FIREBALL), noise=0.0, gravity_correction=True, \
            filter_picks=False, calc_orbit=False)

        # Add a station which is not used, with the picks of station A and its clock ahead
        obs = traj.observations[0]
        traj.infillTrajectory(obs.azim_data, obs.elev_data, obs.time_data - clock_lead, obs.lat + 0.01, \
            obs.lon, obs.ele, station_id='D', ignore_station=True)

        SOLVED_CACHE[clock_lead] = traj.run()

    return SOLVED_CACHE[clock_lead]



def testAClockFarOffDoesNotMoveTheScaledDrop():
    """ Check that the scale of the drop and the solution do not change when the clock of the station which
        is not used is 1 s ahead.
    """

    traj, reference = solveWithIgnoredStation(1.0), solveWithIgnoredStation(0.0)

    # Check the scale of the drop and the solution
    assert reference.gravity_drop_scale is not None
    assert np.allclose(traj.gravity_drop_scale[0], reference.gravity_drop_scale[0], rtol=0, atol=1e-9)
    assert np.allclose(traj.gravity_drop_scale[1], reference.gravity_drop_scale[1], rtol=0, atol=1e-9)

    assert np.allclose(traj.radiant_eci_mini, reference.radiant_eci_mini, rtol=0, atol=1e-12)
    assert np.allclose(traj.state_vect_mini, reference.state_vect_mini, rtol=0, atol=1e-6)

    # Check the residuals of the used stations
    for obs, obs_ref in zip(traj.observations[:3], reference.observations[:3]):
        assert np.allclose(obs.h_residuals, obs_ref.h_residuals, rtol=0, atol=1e-6)
        assert np.allclose(obs.v_residuals, obs_ref.v_residuals, rtol=0, atol=1e-6)



def testAClockFarOffDoesNotMoveTheDropOfTheDragAndDynamicMassFits():
    """ Check that the drag fit drops its grid, and the dynamic mass fit its points, from the same time when
        the clock of the station which is not used is 1 s ahead.
    """

    traj, reference = solveWithIgnoredStation(1.0), solveWithIgnoredStation(0.0)

    # Check the drop of the drag fit
    drop_grid = DragFit(traj).drop_grid
    drop_grid_ref = DragFit(reference).drop_grid
    assert np.allclose(drop_grid, drop_grid_ref, rtol=0, atol=1e-3)

    # Check the points of the dynamic mass fit
    for length, t in ((20e3, 1.0), (40e3, 2.5)):
        point = pointOnTrajectory(traj, length, t)
        assert np.allclose(point, pointOnTrajectory(reference, length, t), rtol=0, atol=1e-6)




if __name__ == "__main__":

    testAClockFarOffDoesNotMoveTheScaledDrop()
    testAClockFarOffDoesNotMoveTheDropOfTheDragAndDynamicMassFits()

    print("All tests passed.")
