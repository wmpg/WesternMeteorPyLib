""" Tests of the fit of the velocity by the drag equation in wmpl.Utils.DragVelocity, on a synthetic fireball
    which decelerates from 18 to 10 km/s while it is observed, and of its use for the gravity drop of the
    trajectory solver.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_DragVelocity.py -v
"""

from __future__ import print_function, division, absolute_import

import contextlib
import functools
import io

import numpy as np

from wmpl.Trajectory.Tests.SyntheticTrajectory import DragMeteor, JDT_REF, observeMeteor
from wmpl.Utils.DragVelocity import fitDragVelocity



@functools.lru_cache(maxsize=None)
def _meteor():
    """ Make the synthetic fireball (computed once).

    Return:
        [DragMeteor object] Synthetic fireball, whose ballistic coefficient grows five times while it is
            visible, from 86 to 48 km.
    """

    return DragMeteor(18000.0, 120.0, 45.0, 0.01, 0.05, 2.7, 6.0)



def _fitVisiblePart(meteor, fps=100.0):
    """ Fit the drag equation to the exact times and lengths of the visible part of the meteor.

    Arguments:
        meteor: [DragMeteor object] Synthetic meteor.

    Keyword arguments:
        fps: [float] Number of points per second. 100 by default.

    Return:
        (time_data, drag_vel):
            - time_data: [ndarray] Times of the points, relative to the beginning of the visible part (s).
            - drag_vel: [DragVelocity object] Fitted model.
    """

    time_data = np.arange(0.0, meteor.t_vis_end - meteor.t_vis_beg, 1/fps)
    len_data = np.interp(time_data + meteor.t_vis_beg, meteor.time, meteor.length)

    return time_data, fitDragVelocity(time_data, len_data, meteor.top_eci, meteor.radiant_eci, JDT_REF)



def testVelocityOfExactPoints():
    """ Check that the velocity fitted to exact points follows the true velocity, which drops by 8 km/s. """

    meteor = _meteor()
    time_data, drag_vel = _fitVisiblePart(meteor)

    vel_err = drag_vel.velocityAtTime(time_data) - meteor.velocityAtTime(time_data)

    print("Largest velocity error: {:.1f} m/s".format(np.max(np.abs(vel_err))))

    assert meteor.velocityAtTime(time_data[0])[0] - meteor.velocityAtTime(time_data[-1])[0] > 7000
    assert np.max(np.abs(vel_err)) < 20.0



def testJacobianMatchesFiniteDifferences():
    """ Check that the derivatives of the residuals by the parameters match finite differences. """

    _, drag_vel = _fitVisiblePart(_meteor(), fps=25.0)

    params = drag_vel.params + 0.01
    jac = drag_vel._jacobian(params, 2.0)

    # Compute the central differences
    jac_num = np.zeros_like(jac)
    for i in range(len(params)):

        step = 1e-5*max(abs(params[i]), 1.0)
        params_up, params_down = np.copy(params), np.copy(params)
        params_up[i] += step
        params_down[i] -= step

        jac_num[:, i] = (drag_vel._residuals(params_up, 2.0) - drag_vel._residuals(params_down, 2.0))/(2*step)

    # Compare by columns, as the parameters have different units
    col_err = np.max(np.abs(jac - jac_num), axis=0)/np.max(np.abs(jac_num), axis=0)

    assert np.median(col_err) < 1e-6
    assert np.max(col_err) < 0.01



def testSolverSpeedsFollowTheDeceleration():
    """ Check that the speeds with which the trajectory solver computes the gravity drop follow the true
        velocity of the fireball.
    """

    meteor = _meteor()
    traj = observeMeteor(meteor, gravity_correction=True)
    with contextlib.redirect_stdout(io.StringIO()):
        traj = traj.run()

    time_rel, vel = traj.gravity_vel
    vel_true = meteor.velocityAtTime(time_rel + (traj.jdt_ref - JDT_REF)*86400.0)

    print("Largest velocity error: {:.1f} m/s".format(np.max(np.abs(vel - vel_true))))

    assert np.max(np.abs(vel - vel_true)) < 150.0
