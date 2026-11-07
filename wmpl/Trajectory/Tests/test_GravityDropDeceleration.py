""" Tests of the gravity drop of a decelerating meteor in the trajectory solver.

Drag also slows down the velocity which gravity adds across the path, so a decelerating meteor drops less
than one at a constant velocity, which applyGravityDrop assumes. The solver scales the drop with the velocity
of the Jacchia fit of the lag (gravityDropScale, Trajectory.calcGravityDropScale). The tests check the scale
against analytic cases and against a fireball integrated in 3D with drag and gravity, and solve that fireball
with and without the scale.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_GravityDropDeceleration.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np

import matplotlib
matplotlib.use('Agg')

from wmpl.Trajectory.Trajectory import GRAVITY_DROP_MIN_VEL, Trajectory, gravityDropScale, \
    jacchiaVelocityFunc
from wmpl.Trajectory.Tests.SyntheticTrajectory import GM_EARTH, JDT_REF, BentDragMeteor, observeMeteor, \
    observeStraightMeteor


# Fireball of the tests: 18 km/s at 45 deg, slowing down to about 10 km/s in 3.3 s
FIREBALL = (18000.0, 120.0, 45.0, 0.01, 0.05, 2.7, 6.0)

# Integrated fireball, shared by the tests
_METEOR = []



def _fireball():
    """ Integrate the fireball of the tests once.

    Return:
        [BentDragMeteor object] The fireball.
    """

    if not _METEOR:
        _METEOR.append(BentDragMeteor(*FIREBALL))

    return _METEOR[0]



def _perpendicularErrors(traj, meteor):
    """ Compute the distances of the model points of a solution from the true positions, across the path.

    Arguments:
        traj: [Trajectory object] Solved trajectory.
        meteor: [BentDragMeteor object] The true meteor.

    Return:
        [ndarray] Distances across the path, of all points of all stations (m).
    """

    errors = []
    for obs in traj.observations:

        diff = np.array(obs.model_eci) - meteor.eciAtTime(obs.time_data)

        # Remove the component along the path, which is the timing and not the shape
        direction = np.array([meteor.radiantAtTime(t) for t in obs.time_data])
        diff -= np.sum(diff*direction, axis=1)[:, None]*direction

        errors.append(np.linalg.norm(diff, axis=1))

    return np.concatenate(errors)



def _solveFireball(gravity_decel):
    """ Solve the noise-free fireball with the gravity correction.

    Arguments:
        gravity_decel: [bool] Scale the drop for the deceleration.

    Return:
        (traj, radiant_error, perp_errors):
            - traj: [Trajectory object] Solved trajectory.
            - radiant_error: [float] Angle between the solved radiant and the direction of motion at the first
                point (arcsec).
            - perp_errors: [ndarray] Distances of the model points from the truth across the path (m).
    """

    meteor = _fireball()

    traj = observeMeteor(meteor, noise=0.0, gravity_correction=True, gravity_decel=gravity_decel, \
        calc_orbit=False)
    traj.run()

    radiant_error = 3600*np.degrees(np.linalg.norm(traj.radiant_eci_mini - meteor.radiantAtTime(0.0)))

    return traj, radiant_error, _perpendicularErrors(traj, meteor)



def testTheScaleIsOneAtAConstantVelocity():
    """ Check that the drop is not scaled at a constant velocity. """

    time_rel = np.linspace(0.0, 5.0, 1001)
    scale = gravityDropScale(time_rel, np.full_like(time_rel, 15000.0))

    assert np.max(np.abs(scale - 1.0)) < 1e-12



def testTheScaleOfALinearDeceleration():
    """ Check the scale against the analytic drop of a linear deceleration, v = v0*(1 - a*t), which is
        g_perp*((u^2*ln(u)/2 - u^2/4) + 1/4)/a^2 with u = 1 - a*t.
    """

    a = 0.15
    time_rel = np.linspace(0.0, 5.0, 1001)
    scale = gravityDropScale(time_rel, 20000.0*(1 - a*time_rel))

    u = 1 - a*time_rel[1:]
    drop = ((u**2*np.log(u)/2 - u**2/4) + 1/4)/a**2
    expected = drop/(0.5*time_rel[1:]**2)

    assert scale[0] == 1.0
    assert np.max(np.abs(scale[1:] - expected)) < 2e-4
    assert abs(scale[-1] - expected[-1]) < 1e-6
    assert expected[-1] < 0.7



def testTheScaleGivesTheDropOfABentPath():
    """ Check that the scaled drop is the distance across the path of a fireball integrated in 3D with drag
        and gravity, from its tangent at the first point.
    """

    meteor = _fireball()

    time_rel = np.linspace(0.0, meteor.t_vis_end - meteor.t_vis_beg, 1001)
    pos = meteor.eciAtTime(time_rel)

    # Distance across the path from the tangent at the first point
    direction = -meteor.radiantAtTime(0.0)
    rel = pos - pos[0]
    across = np.linalg.norm(rel - np.outer(rel @ direction, direction), axis=1)

    # Gravity across the path at the first point
    g_vec = -GM_EARTH*pos[0]/np.linalg.norm(pos[0])**3
    g_perp = np.linalg.norm(g_vec - (g_vec @ direction)*direction)

    drop_const = 0.5*g_perp*time_rel**2
    drop_scaled = gravityDropScale(time_rel, meteor.velocityAtTime(time_rel))*drop_const

    assert np.max(np.abs(across - drop_scaled)) < 0.5
    assert np.max(np.abs(across - drop_const)) > 3.0



def testTheSolverRecoversABentFireball():
    """ Check that the solver recovers the radiant and the positions of the fireball integrated in 3D with
        the scaled drop, and not with the drop at a constant velocity.
    """

    traj, radiant_error, perp_errors = _solveFireball(True)

    assert traj.gravity_drop_scale is not None
    assert 0.8 < traj.gravity_drop_scale[1][-1] < 0.9
    assert radiant_error < 1.0
    assert np.max(perp_errors) < 1.0

    traj_const, radiant_error_const, perp_errors_const = _solveFireball(False)

    assert traj_const.gravity_drop_scale is None
    assert radiant_error_const > 3.0
    assert np.max(perp_errors_const) > 3.0



def testTheDropFactorIsTheGravityFactorTimesTheScale():
    """ Check the factor of the drop at given times, with and without a scale. """

    traj = Trajectory(JDT_REF, gravity_factor=0.5, show_plots=False, save_results=False, verbose=False)
    assert traj.gravity_decel
    assert traj.gravityDropFactor(1.0) == 0.5

    traj.gravity_drop_scale = (np.array([0.0, 2.0]), np.array([1.0, 0.6]))

    assert np.allclose(traj.gravityDropFactor(np.array([-1.0, 0.0, 1.0, 2.0, 3.0])), \
        0.5*np.array([1.0, 1.0, 0.8, 0.6, 0.6]))



def testNoScaleWithoutTheOptionOrInOldPickles():
    """ Check that the drop is not scaled with gravity_decel off, or in a trajectory pickled before the option
        was added.
    """

    traj, _, _ = observeStraightMeteor(gravity_correction=True)
    traj.run()

    traj.v_init = 20000.0
    jacchia_fit = np.array([10.0, 2.0])

    traj.gravity_decel = False
    traj.calcGravityDropScale(traj.observations, jacchia_fit)
    assert traj.gravity_drop_scale is None

    del traj.gravity_decel
    traj.calcGravityDropScale(traj.observations, jacchia_fit)
    assert traj.gravity_drop_scale is None



def testTheVelocityIsHeldAboveWhereAblationStops():
    """ Check that a Jacchia fit whose velocity falls below zero at the end still scales the drop, with the
        velocity held at GRAVITY_DROP_MIN_VEL.
    """

    traj, _, _ = observeStraightMeteor(gravity_correction=True)
    traj.run()

    t0 = min([obs.time_data[0] for obs in traj.observations])
    t_end = max([obs.time_data[-1] for obs in traj.observations])

    # Choose a2 so that the velocity of the fit is negative at the last point
    traj.v_init = 20000.0
    a2 = 3.0
    a1 = 1.5*traj.v_init/(a2*np.exp(a2*t_end))
    assert jacchiaVelocityFunc(t_end, a1, a2, traj.v_init) < 0

    traj.calcGravityDropScale(traj.observations, np.array([a1, a2]))

    time_rel, scale = traj.gravity_drop_scale
    vel = np.maximum(jacchiaVelocityFunc(t0 + time_rel, a1, a2, traj.v_init), GRAVITY_DROP_MIN_VEL)

    assert np.allclose(scale, gravityDropScale(time_rel, vel))
    assert np.all(np.isfinite(scale))
    assert 0 < scale[-1] < 1



def testAMeteorWithoutDecelerationIsNotScaled():
    """ Check that a straight meteor at a constant velocity keeps the drop at a constant velocity. """

    traj, _, _ = observeStraightMeteor(noise=0.0, gravity_correction=True)
    traj.run()

    assert (traj.gravity_drop_scale is None) or (np.min(traj.gravity_drop_scale[1]) > 0.999)




if __name__ == "__main__":

    # Run all tests
    test_list = [testTheScaleIsOneAtAConstantVelocity, testTheScaleOfALinearDeceleration, \
        testTheScaleGivesTheDropOfABentPath, testTheSolverRecoversABentFireball, \
        testTheDropFactorIsTheGravityFactorTimesTheScale, testNoScaleWithoutTheOptionOrInOldPickles, \
        testTheVelocityIsHeldAboveWhereAblationStops, testAMeteorWithoutDecelerationIsNotScaled]

    for test in test_list:
        test()
        print("PASS {:s}".format(test.__name__))
