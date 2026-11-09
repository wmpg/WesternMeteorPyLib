""" Tests of the displacement of a decelerating meteor by drag against the air, which rotates with the Earth.

The air moves with the rotation of the Earth, up to 465 m/s in ECI coordinates, and drag acts against the
velocity of the meteor relative to the air, so a decelerating meteor is pulled across its initial direction
towards the velocity of the air. The solver moves the points of its line by that displacement
(airRotationShift, Trajectory.calcAirRotationShift). The tests check the displacement against a fireball
integrated in 3D with drag against the rotating air, solve that fireball with and without the displacement,
check its lengths, check that a meteor at a constant velocity and one whose velocity changes by gravity alone
are not moved, and check the points of the dynamic mass fit and of the drag fit.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_AirRotationDrag.py -v
"""

from __future__ import print_function, division, absolute_import

import copy
import sys

import numpy as np
import scipy.optimize

import matplotlib
matplotlib.use('Agg')

from wmpl.Trajectory.Trajectory import EARTH_ROTATION_RATE, airRotationShift, interpAirRotationShift
from wmpl.Trajectory.Tests.SyntheticTrajectory import GM_EARTH, JDT_REF, BentDragMeteor, observeMeteor, \
    observeStraightMeteor
from wmpl.Utils.DragFit import DragFit
from wmpl.Utils.DynamicMassFit import pointOnTrajectory
from wmpl.Utils.Math import angleBetweenVectors


# Fireball of the tests: 18 km/s at 45 deg, slowing down to about 10 km/s in 3.3 s
FIREBALL = (18000.0, 120.0, 45.0, 0.01, 0.05, 2.7, 6.0)

# Integrated fireball and its solutions, shared by the tests
_METEOR = []
_SOLUTIONS = {}



def _fireball():
    """ Integrate the fireball of the tests once, with drag against the rotating air.

    Return:
        [BentDragMeteor object] The fireball.
    """

    if not _METEOR:
        _METEOR.append(BentDragMeteor(*FIREBALL, air_rotation=True))

    return _METEOR[0]



def _solveFireball(air_rotation):
    """ Solve the noise-free fireball with the gravity correction, once for each option.

    Arguments:
        air_rotation: [bool] Move the points by drag against the rotating air.

    Return:
        (traj, radiant_error, perp_errors):
            - traj: [Trajectory object] Solved trajectory.
            - radiant_error: [float] Angle between the solved radiant and the direction of motion at the
                beginning of the solution, its first point (arcsec).
            - perp_errors: [ndarray] Distances of the model points from the truth across the path (m).
    """

    if air_rotation not in _SOLUTIONS:

        meteor = _fireball()

        traj = observeMeteor(meteor, noise=0.0, gravity_correction=True, air_rotation=air_rotation, \
            calc_orbit=False)
        traj.run()

        # The solver moves its reference time to its first point
        t_beg = (traj.jdt_ref - JDT_REF)*86400.0
        radiant_error = 3600*np.degrees(angleBetweenVectors(traj.radiant_eci_mini, \
            meteor.radiantAtTime(t_beg)))

        # Take the distances of the model points from the truth across the path
        perp_errors = []
        for obs in traj.observations:

            diff = np.array(obs.model_eci) - meteor.eciAtTime(obs.time_data + t_beg)
            direction = np.array([meteor.radiantAtTime(t) for t in obs.time_data + t_beg])
            diff -= np.sum(diff*direction, axis=1)[:, None]*direction

            perp_errors.append(np.linalg.norm(diff, axis=1))

        _SOLUTIONS[air_rotation] = (traj, radiant_error, np.concatenate(perp_errors))

    return _SOLUTIONS[air_rotation]



def _acrossPath(meteor, air_rotation=True):
    """ Compute the deviation of the visible part of an integrated meteor from its initial line, and the parts
        of it which drag against the rotating air and gravity cause.

    The velocity across the initial line, q, follows dq/dt = g_perp + (a/w)*(u_perp - q), where a is the
    deceleration by drag, w the speed relative to the air and u_perp the velocity of the air across the line.
    This is linear in q, so the deviation is the sum of the part of gravity and the part of the air. The part
    of gravity is integrated here with the true deceleration of the meteor.

    Arguments:
        meteor: [BentDragMeteor object] Integrated meteor.

    Keyword arguments:
        air_rotation: [bool] The meteor was integrated with drag against the rotating air, so the speed
            relative to the air is taken. True by default.

    Return:
        (time_rel, vel_drag, eci0, radiant_eci, deviation, gravity_part):
            - time_rel: [ndarray] Times from the beginning of the visible part (s).
            - vel_drag: [ndarray] Velocity along the initial line without the speed gravity adds along it
                (m/s).
            - eci0: [ndarray] ECI coordinates of the beginning of the visible part (m).
            - radiant_eci: [ndarray] Initial radiant unit vector.
            - deviation: [ndarray] Deviation of the meteor from the initial line (m), one row per time.
            - gravity_part: [ndarray] Part of the deviation which gravity causes (m), one row per time.
    """

    visible = (meteor.time >= meteor.t_vis_beg) & (meteor.time <= meteor.t_vis_end)
    time_rel = meteor.time[visible] - meteor.t_vis_beg
    pos = meteor.pos[visible]
    vel = meteor.vel[visible]

    eci0 = pos[0]
    radiant_eci = -vel[0]/np.linalg.norm(vel[0])
    motion = -radiant_eci

    # Deviation from the initial line
    length = (pos - eci0) @ motion
    deviation = (pos - eci0) - np.outer(length, motion)

    # Gravity, the air velocity and the drag deceleration along the path
    grav = -GM_EARTH*pos/np.linalg.norm(pos, axis=1)[:, None]**3
    air_vel = np.cross(np.array([0.0, 0.0, EARTH_ROTATION_RATE*air_rotation]), pos)
    speed_air = np.linalg.norm(vel - air_vel, axis=1)
    drag_acc = np.linalg.norm(np.gradient(vel, time_rel, axis=0) - grav, axis=1)

    # Velocity along the line without the speed gravity adds along it
    grav_along = grav @ motion
    grav_speed = np.r_[0.0, np.cumsum(0.5*(grav_along[1:] + grav_along[:-1])*np.diff(time_rel))]
    vel_drag = np.gradient(length, time_rel) - grav_speed

    # Integrate the part of gravity across the line, exactly for coefficients constant over each step
    grav_perp = grav - np.outer(grav @ motion, motion)
    rate = drag_acc/speed_air
    vel_perp = np.zeros_like(pos)
    for k in range(len(time_rel) - 1):

        dt = time_rel[k + 1] - time_rel[k]
        rate_mid = 0.5*(rate[k] + rate[k + 1])
        grav_mid = 0.5*(grav_perp[k] + grav_perp[k + 1])

        vel_perp[k + 1] = grav_mid/rate_mid + (vel_perp[k] - grav_mid/rate_mid)*np.exp(-rate_mid*dt)

    gravity_part = np.zeros_like(pos)
    gravity_part[1:] = np.cumsum(0.5*(vel_perp[1:] + vel_perp[:-1])*np.diff(time_rel)[:, None], axis=0)

    return time_rel, vel_drag, eci0, radiant_eci, deviation, gravity_part



def testTheShiftIsZeroAtAConstantVelocity():
    """ Check that the displacement is exactly zero at a constant velocity, for a meteor going down and for
        one going up.
    """

    time_rel = np.linspace(0.0, 5.0, 1001)
    eci0 = np.array([4.0e6, 2.0e6, 4.5e6])
    up = eci0/np.linalg.norm(eci0)
    east = np.cross([0.0, 0.0, 1.0], up)
    east /= np.linalg.norm(east)

    for elev in (40.0, -10.0):

        radiant_eci = np.sin(np.radians(elev))*up + np.cos(np.radians(elev))*east
        shift = airRotationShift(time_rel, np.full_like(time_rel, 15000.0), eci0, radiant_eci)

        assert np.all(shift == 0.0)



def testTheShiftIsAcrossTheLine():
    """ Check that the interpolated displacement has no part along the given line, so it moves the points
        across the line only.
    """

    meteor = _fireball()
    time_rel, vel_drag, eci0, radiant_eci, _, _ = _acrossPath(meteor)
    air_shift = (time_rel, airRotationShift(time_rel, vel_drag, eci0, radiant_eci))

    # Take a line which is 0.5 deg away from the one the displacement was computed for
    tilted = radiant_eci + 0.0087*np.cross(radiant_eci, eci0/np.linalg.norm(eci0))
    tilted /= np.linalg.norm(tilted)

    shift = interpAirRotationShift(air_shift, time_rel, tilted)

    assert np.max(np.abs(shift @ tilted)) < 1e-9*np.max(np.linalg.norm(shift, axis=1))
    assert np.max(np.linalg.norm(shift, axis=1)) > 50.0



def testTheShiftGoesOnAfterTheLastPoint():
    """ Check that after the last time of the displacement, as at picks after the last one used, the
        displacement goes on with the velocity across the line at that time, instead of stopping there.
    """

    meteor = _fireball()
    time_rel, vel_drag, eci0, radiant_eci, _, _ = _acrossPath(meteor)
    shift_full = airRotationShift(time_rel, vel_drag, eci0, radiant_eci)

    # The displacement up to 0.1 s before the end, evaluated over the last 0.1 s
    used = time_rel <= time_rel[-1] - 0.1
    after = time_rel[~used]
    air_shift = (time_rel[used], shift_full[used])

    truth = interpAirRotationShift((time_rel, shift_full), after, radiant_eci)
    shift = interpAirRotationShift(air_shift, after, radiant_eci)
    held = interpAirRotationShift(air_shift, np.full(len(after), time_rel[used][-1]), radiant_eci)

    assert np.max(np.linalg.norm(shift - truth, axis=1)) < 1.0
    assert np.max(np.linalg.norm(held - truth, axis=1)) > 5.0

    # Before the last time, it is the interpolated displacement
    before = time_rel[used][::50]
    assert np.allclose(interpAirRotationShift(air_shift, before, radiant_eci), \
        interpAirRotationShift((time_rel, shift_full), before, radiant_eci), rtol=0, atol=1e-9)



def testTheShiftGivesTheDeviationOfAnIntegratedFireball():
    """ Check that the displacement, with the part of gravity, gives the deviation from the initial line of a
        fireball integrated in 3D with drag against the rotating air, to 0.03 m out of 84 m. Taking the air
        velocity at the beginning only leaves 0.87 m, and the speed in ECI coordinates instead of the speed
        relative to the air 0.96 m.
    """

    meteor = _fireball()
    time_rel, vel_drag, eci0, radiant_eci, deviation, gravity_part = _acrossPath(meteor)

    shift = airRotationShift(time_rel, vel_drag, eci0, radiant_eci)

    assert np.max(np.linalg.norm(shift, axis=1)) > 80.0
    assert np.max(np.linalg.norm(deviation - gravity_part - shift, axis=1)) < 0.03

    # Without the rotation of the air, the part of gravity alone gives the deviation
    still = BentDragMeteor(*FIREBALL, air_rotation=False)
    _, _, _, _, deviation, gravity_part = _acrossPath(still, air_rotation=False)

    assert np.max(np.linalg.norm(deviation - gravity_part, axis=1)) < 0.03



def testTheSolverRecoversAFireballInTheRotatingAir():
    """ Check that the solver recovers the radiant and the positions of the noise-free fireball which drag
        against the rotating air pulls across its path. Without the displacement, the radiant is 157 arcsec
        off and the positions up to 56 m, with it 0.5 arcsec and 0.09 m.
    """

    traj, radiant_error, perp_errors = _solveFireball(True)
    _, radiant_error_off, perp_errors_off = _solveFireball(False)

    assert traj.air_rotation_shift is not None

    assert radiant_error_off > 100.0
    assert np.max(perp_errors_off) > 30.0

    assert radiant_error < 2.0
    assert np.max(perp_errors) < 0.3



def testTheLengthsAreTakenOnTheMovedLine():
    """ Check that the lengths of the solved fireball are those of the points on its line moved by the drop
        and by the displacement, as moving the line across also moves the points along it, here by 33 m.
    """

    traj, _, _ = _solveFireball(True)

    lengths = np.concatenate([np.asarray(obs.state_vect_dist)[obs.ignore_list == 0] \
        for obs in traj.observations if not obs.ignore_station])

    _, lengths_moved = traj.calcDroppedLengths(traj.observations, traj.state_vect_mini, traj.radiant_eci_mini)

    # Take the lengths on the line moved by the drop only
    air_shift = traj.air_rotation_shift
    traj.air_rotation_shift = None
    _, lengths_dropped = traj.calcDroppedLengths(traj.observations, traj.state_vect_mini, \
        traj.radiant_eci_mini)
    traj.air_rotation_shift = air_shift

    assert np.max(np.abs(lengths - lengths_moved)) < 0.05
    assert np.max(np.abs(lengths - lengths_dropped)) > 10.0



def testTheDragFitContinuesTheShiftBelowItsLastPoint():
    """ Check that the drag fit, below its last point, keeps moving the path by drag against the rotating air:
        the last pick of the meteor, rejected as the 3-sigma filter can do, is then extrapolated by the drag
        fit and still on the model path of the solver to 1 m across the line. Without the displacement there
        it is 4 m off, 0.04 s after the last point.
    """

    traj = copy.deepcopy(_solveFireball(True)[0])
    t0 = min([obs.time_data[0] for obs in traj.observations])

    # Reject the last pick of the station which saw the meteor last
    obs = max(traj.observations, key=lambda o: o.time_data[-1])
    obs.ignore_list = np.array(obs.ignore_list)
    obs.ignore_list[-1] = 1

    drag_fit = DragFit(traj)

    length = (traj.state_vect_mini - np.array(obs.model_eci[-1])) @ traj.radiant_eci_mini
    assert length > drag_fit.len_grid[drag_fit.i_last]

    diff = drag_fit.eciAtLength(np.array([length]))[0] - np.array(obs.model_eci[-1])
    diff -= (diff @ traj.radiant_eci_mini)*traj.radiant_eci_mini

    assert np.linalg.norm(diff) < 1.0

    # The displacement grows there
    shift = traj.airRotationShiftAt(np.array([drag_fit.time_grid[drag_fit.i_last], obs.time_data[-1] - t0]), \
        traj.radiant_eci_mini)
    assert np.linalg.norm(shift[1] - shift[0]) > 2.0



def testAFitWhichRunsOutOfEvaluationsIsContinued():
    """ Check that the solver continues a fit with the displacement which runs out of function evaluations,
        and keeps the displacement, by making the first such fit stop after 50 evaluations.
    """

    traj = observeMeteor(_fireball(), noise=0.0, gravity_correction=True, calc_orbit=False)

    # Limit the evaluations of the first fit with a displacement (wmpl loads every module twice)
    traj_module = sys.modules[type(traj).__module__]
    original = scipy.optimize.minimize
    limited = []

    def _limitedMinimize(fun, x0, args=(), **kwargs):

        if (fun is traj_module.minimizeAngleCost) and (len(args) > 7) and (args[7] is not None) \
            and (not limited):

            limited.append(True)
            kwargs['options'] = {'maxfev': 50}

        return original(fun, x0, args=args, **kwargs)

    scipy.optimize.minimize = _limitedMinimize

    try:
        traj.run()

    finally:
        scipy.optimize.minimize = original

    t_beg = (traj.jdt_ref - JDT_REF)*86400.0

    assert limited
    assert traj.air_rotation_shift is not None
    assert 3600*np.degrees(angleBetweenVectors(traj.radiant_eci_mini, _fireball().radiantAtTime(t_beg))) < 2.0



def testNoShiftWithoutTheOptionOrInOldPickles():
    """ Check that the solver does not move the points without the option, and that a trajectory pickled
        before the option was added is solved as without it.
    """

    traj_off, _, _ = _solveFireball(False)

    assert traj_off.air_rotation_shift is None

    # Remove the option and the displacement, as in an old pickle
    traj = observeMeteor(_fireball(), noise=0.0, gravity_correction=True, calc_orbit=False)
    del traj.air_rotation
    del traj.air_rotation_shift
    traj.run()

    assert traj.air_rotation_shift is None
    assert np.array_equal(traj.radiant_eci_mini, traj_off.radiant_eci_mini)
    assert np.array_equal(traj.state_vect_mini, traj_off.state_vect_mini)



def testAMeteorWithoutDecelerationIsNotMoved():
    """ Check that a straight meteor at a constant velocity, with about 20 arcsec of noise, is not moved, and
        that its solution is the same as without the option.
    """

    for kwargs in (dict(seed=1), dict(seed=2)):

        traj, _, _ = observeStraightMeteor(gravity_correction=True, **kwargs)
        traj.run()

        unmoved, _, _ = observeStraightMeteor(gravity_correction=True, air_rotation=False, **kwargs)
        unmoved.run()

        assert traj.air_rotation_shift is None
        assert np.array_equal(traj.state_vect_mini, unmoved.state_vect_mini)
        assert np.array_equal(traj.radiant_eci_mini, unmoved.radiant_eci_mini)



def testAVelocityChangedByGravityAloneIsNotMoved():
    """ Check that a noise-free meteor dropped by gravity, whose velocity changes along its path by gravity
        alone, is not moved, as only drag pulls it towards the velocity of the air: the lengths of the pass
        with the drop do not hold the speed gravity adds along the path. Without noise, the velocity fit
        follows the small difference between the drop of the solver and that of the meteor, which would move
        the points by less than 0.1 mm, below the displacement at which the fit stops moving the line.
    """

    solutions = []
    for air_rotation in (True, False):

        traj, _, _ = observeStraightMeteor(noise=0.0, duration=5.0, v_init=15000.0, elevation=30.0, \
            gravity_drop=True, estimate_timing_vel=False, gravity_correction=True, \
            air_rotation=air_rotation, filter_picks=False, calc_orbit=False)
        traj.run()

        solutions.append(traj)

    assert solutions[0].air_rotation_shift is None
    assert np.array_equal(solutions[0].radiant_eci_mini, solutions[1].radiant_eci_mini)
    assert np.array_equal(solutions[0].state_vect_mini, solutions[1].state_vect_mini)



def testTheDynamicMassFitPointsFollowTheShift():
    """ Check that the points of the dynamic mass fit and of the drag fit on the solved fireball follow the
        model points of the solver, which drag against the rotating air moves by more than 50 m, to 1 m across
        the line.
    """

    traj, _, _ = _solveFireball(True)
    t0 = min([obs.time_data[0] for obs in traj.observations])

    drag_fit = DragFit(traj)

    shift_max = 0.0
    for obs in traj.observations:

        shift_max = max(shift_max, np.max(np.linalg.norm(traj.airRotationShiftAt(obs.time_data - t0, \
            traj.radiant_eci_mini), axis=1)))

        lengths = (traj.state_vect_mini - np.array(obs.model_eci)) @ traj.radiant_eci_mini

        points = np.array([pointOnTrajectory(traj, length, t) for length, t in zip(lengths, obs.time_data)])
        points_drag = drag_fit.eciAtLength(lengths)

        for pts in (points, points_drag):

            diff = pts - np.array(obs.model_eci)
            diff -= np.outer(diff @ traj.radiant_eci_mini, traj.radiant_eci_mini)

            assert np.max(np.linalg.norm(diff, axis=1)) < 1.0

    assert shift_max > 50.0




if __name__ == "__main__":

    # Run all tests
    test_list = [testTheShiftIsZeroAtAConstantVelocity, testTheShiftIsAcrossTheLine, \
        testTheShiftGoesOnAfterTheLastPoint, \
        testTheShiftGivesTheDeviationOfAnIntegratedFireball, testTheSolverRecoversAFireballInTheRotatingAir, \
        testTheLengthsAreTakenOnTheMovedLine, testAFitWhichRunsOutOfEvaluationsIsContinued, \
        testNoShiftWithoutTheOptionOrInOldPickles, testAMeteorWithoutDecelerationIsNotMoved, \
        testAVelocityChangedByGravityAloneIsNotMoved, testTheDynamicMassFitPointsFollowTheShift]

    for test in test_list:
        test()
        print("PASS {:s}".format(test.__name__))
