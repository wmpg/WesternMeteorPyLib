""" Tests of the displacement of a decelerating meteor by drag against the air, which rotates with the Earth.

Drag acts against the velocity of the meteor relative to the air, which moves up to 465 m/s in ECI
coordinates, so a decelerating meteor is pulled across its initial direction. The solver moves the points of
its line by that displacement (airRotationShift, Trajectory.calcAirRotationShift). The tests check the
displacement against a fireball integrated in 3D with drag against the rotating air, solve that fireball with
and without it, and check its lengths, meteors which are not moved, the drag fit and the dynamic mass fit.

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


# Fireball of the tests, 18 km/s at 45 deg slowing down to about 10 km/s in 3.3 s
FIREBALL = (18000.0, 120.0, 45.0, 0.01, 0.05, 2.7, 6.0)

# Integrated fireball and its solutions, shared by the tests
METEOR_CACHE = []
SOLVED_CACHE = {}



def integrateFireball():
    """ Integrate the fireball of the tests once, with drag against the rotating air.

    Return:
        [BentDragMeteor object] The fireball.
    """

    if not METEOR_CACHE:
        METEOR_CACHE.append(BentDragMeteor(*FIREBALL, air_rotation=True))

    return METEOR_CACHE[0]



def solveFireball(air_rotation):
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

    if air_rotation not in SOLVED_CACHE:

        meteor = integrateFireball()

        traj = observeMeteor(meteor, noise=0.0, gravity_correction=True, air_rotation=air_rotation, \
            calc_orbit=False)
        traj.run()

        # Compare the radiant with the direction of motion at the solver's first point
        t_beg = (traj.jdt_ref - JDT_REF)*86400.0
        radiant_error = 3600*np.degrees(angleBetweenVectors(traj.radiant_eci_mini, \
            meteor.radiantAtTime(t_beg)))

        # Compute the distances of the model points from the truth across the path
        perp_errors = []
        for obs in traj.observations:

            diff = np.array(obs.model_eci) - meteor.eciAtTime(obs.time_data + t_beg)
            direction = np.array([meteor.radiantAtTime(t) for t in obs.time_data + t_beg])
            diff -= np.sum(diff*direction, axis=1)[:, None]*direction

            perp_errors.append(np.linalg.norm(diff, axis=1))

        SOLVED_CACHE[air_rotation] = (traj, radiant_error, np.concatenate(perp_errors))

    return SOLVED_CACHE[air_rotation]



def calcDeviationFromLine(meteor, air_rotation=True):
    """ Compute the deviation of the visible part of an integrated meteor from its initial line, and the part
        of it which gravity causes, integrated with the true deceleration of the meteor.

    Arguments:
        meteor: [BentDragMeteor object] Integrated meteor.

    Keyword arguments:
        air_rotation: [bool] Take the speed relative to the rotating air, as the meteor was integrated with
            drag against it. True by default.

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

    # Take the visible part
    visible = (meteor.time >= meteor.t_vis_beg) & (meteor.time <= meteor.t_vis_end)
    time_rel = meteor.time[visible] - meteor.t_vis_beg
    pos = meteor.pos[visible]
    vel = meteor.vel[visible]

    # Take the initial line
    eci0 = pos[0]
    radiant_eci = -vel[0]/np.linalg.norm(vel[0])
    motion = -radiant_eci

    # Compute the deviation from the initial line
    length = (pos - eci0) @ motion
    deviation = (pos - eci0) - np.outer(length, motion)

    # Compute the gravity, the air velocity and the drag deceleration along the path
    grav = -GM_EARTH*pos/np.linalg.norm(pos, axis=1)[:, None]**3
    air_vel = np.cross(np.array([0.0, 0.0, EARTH_ROTATION_RATE*air_rotation]), pos)
    speed_air = np.linalg.norm(vel - air_vel, axis=1)
    drag_acc = np.linalg.norm(np.gradient(vel, time_rel, axis=0) - grav, axis=1)

    # Compute the velocity along the line without the speed gravity adds along it
    grav_along = grav @ motion
    grav_speed = np.r_[0.0, np.cumsum(0.5*(grav_along[1:] + grav_along[:-1])*np.diff(time_rel))]
    vel_drag = np.gradient(length, time_rel) - grav_speed

    # Integrate the part of gravity across the line, exactly over each step
    grav_perp = grav - np.outer(grav @ motion, motion)
    rate = drag_acc/speed_air
    vel_perp = np.zeros_like(pos)
    for k in range(len(time_rel) - 1):

        dt = time_rel[k + 1] - time_rel[k]
        rate_mid = 0.5*(rate[k] + rate[k + 1])
        grav_mid = 0.5*(grav_perp[k] + grav_perp[k + 1])

        vel_perp[k + 1] = grav_mid/rate_mid + (vel_perp[k] - grav_mid/rate_mid)*np.exp(-rate_mid*dt)

    # Integrate the displacement with the trapezoidal rule
    gravity_part = np.zeros_like(pos)
    gravity_part[1:] = np.cumsum(0.5*(vel_perp[1:] + vel_perp[:-1])*np.diff(time_rel)[:, None], axis=0)

    return time_rel, vel_drag, eci0, radiant_eci, deviation, gravity_part



def testTheShiftIsZeroAtAConstantVelocity():
    """ Check that the displacement is exactly zero at a constant velocity, for a meteor going down and for
        one going up.
    """

    # Take the vertical and the east at the beginning
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

    meteor = integrateFireball()
    time_rel, vel_drag, eci0, radiant_eci, _, _ = calcDeviationFromLine(meteor)
    air_shift = (time_rel, airRotationShift(time_rel, vel_drag, eci0, radiant_eci))

    # Tilt the line by 0.5 deg
    tilted = radiant_eci + 0.0087*np.cross(radiant_eci, eci0/np.linalg.norm(eci0))
    tilted /= np.linalg.norm(tilted)

    shift = interpAirRotationShift(air_shift, time_rel, tilted)

    assert np.max(np.abs(shift @ tilted)) < 1e-9*np.max(np.linalg.norm(shift, axis=1))
    assert np.max(np.linalg.norm(shift, axis=1)) > 50.0



def testTheShiftGoesOnAfterTheLastPoint():
    """ Check that after the last time of the displacement, as at picks after the last one used, the
        displacement goes on with the velocity across the line at that time, instead of stopping there.
    """

    meteor = integrateFireball()
    time_rel, vel_drag, eci0, radiant_eci, _, _ = calcDeviationFromLine(meteor)
    shift_full = airRotationShift(time_rel, vel_drag, eci0, radiant_eci)

    # Cut the displacement 0.1 s before the end
    used = time_rel <= time_rel[-1] - 0.1
    after = time_rel[~used]
    air_shift = (time_rel[used], shift_full[used])

    truth = interpAirRotationShift((time_rel, shift_full), after, radiant_eci)
    shift = interpAirRotationShift(air_shift, after, radiant_eci)
    held = interpAirRotationShift(air_shift, np.full(len(after), time_rel[used][-1]), radiant_eci)

    assert np.max(np.linalg.norm(shift - truth, axis=1)) < 1.0
    assert np.max(np.linalg.norm(held - truth, axis=1)) > 5.0

    # Check that it is the interpolated displacement before the last time
    before = time_rel[used][::50]
    assert np.allclose(interpAirRotationShift(air_shift, before, radiant_eci), \
        interpAirRotationShift((time_rel, shift_full), before, radiant_eci), rtol=0, atol=1e-9)



def testTheShiftGivesTheDeviationOfAnIntegratedFireball():
    """ Check that the displacement, with the part of gravity, gives the deviation from the initial line of a
        fireball integrated in 3D with drag against the rotating air, to 0.03 m out of 84 m.
    """

    meteor = integrateFireball()
    time_rel, vel_drag, eci0, radiant_eci, deviation, gravity_part = calcDeviationFromLine(meteor)

    shift = airRotationShift(time_rel, vel_drag, eci0, radiant_eci)

    assert np.max(np.linalg.norm(shift, axis=1)) > 80.0
    assert np.max(np.linalg.norm(deviation - gravity_part - shift, axis=1)) < 0.03

    # Check that gravity alone gives the deviation in still air
    still = BentDragMeteor(*FIREBALL, air_rotation=False)
    _, _, _, _, deviation, gravity_part = calcDeviationFromLine(still, air_rotation=False)

    assert np.max(np.linalg.norm(deviation - gravity_part, axis=1)) < 0.03



def testTheSolverRecoversAFireballInTheRotatingAir():
    """ Check that the solver recovers the radiant and the positions of the noise-free fireball which drag
        against the rotating air pulls across its path, and not without the displacement.
    """

    traj, radiant_error, perp_errors = solveFireball(True)
    _, radiant_error_off, perp_errors_off = solveFireball(False)

    assert traj.air_rotation_shift is not None

    # Check the solution without the displacement
    assert radiant_error_off > 100.0
    assert np.max(perp_errors_off) > 30.0

    # Check the solution with the displacement
    assert radiant_error < 2.0
    assert np.max(perp_errors) < 0.3



def testTheLengthsAreTakenOnTheMovedLine():
    """ Check that the lengths of the solved fireball are those of the points on its line moved by the drop
        and by the displacement.
    """

    traj, _, _ = solveFireball(True)

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
    """ Check that the drag fit keeps moving the path by drag against the rotating air below its last point,
        so that a rejected last pick stays on the model path of the solver to 1 m across the line.
    """

    traj = copy.deepcopy(solveFireball(True)[0])
    t0 = min([obs.time_data[0] for obs in traj.observations])

    # Reject the last pick of the station which saw the meteor last
    obs = max(traj.observations, key=lambda o: o.time_data[-1])
    obs.ignore_list = np.array(obs.ignore_list)
    obs.ignore_list[-1] = 1

    drag_fit = DragFit(traj)

    # Check the extrapolated pick against the model path
    length = (traj.state_vect_mini - np.array(obs.model_eci[-1])) @ traj.radiant_eci_mini
    assert length > drag_fit.len_grid[drag_fit.i_last]

    diff = drag_fit.eciAtLength(np.array([length]))[0] - np.array(obs.model_eci[-1])
    diff -= (diff @ traj.radiant_eci_mini)*traj.radiant_eci_mini

    assert np.linalg.norm(diff) < 1.0

    # Check that the displacement grows there
    shift = traj.airRotationShiftAt(np.array([drag_fit.time_grid[drag_fit.i_last], obs.time_data[-1] - t0]), \
        traj.radiant_eci_mini)
    assert np.linalg.norm(shift[1] - shift[0]) > 2.0



def testAFitWhichRunsOutOfEvaluationsIsContinued():
    """ Check that the solver continues a fit with the displacement which runs out of function evaluations,
        and keeps the displacement, by making the first such fit stop after 50 evaluations.
    """

    traj = observeMeteor(integrateFireball(), noise=0.0, gravity_correction=True, calc_orbit=False)

    # Limit the evaluations of the first fit with a displacement
    traj_module = sys.modules[type(traj).__module__]
    original = scipy.optimize.minimize
    limited = []

    def limitedMinimize(fun, x0, args=(), **kwargs):
        """ Stop the first fit with a displacement after 50 evaluations. """

        if (fun is traj_module.minimizeAngleCost) and (len(args) > 7) and (args[7] is not None) \
            and (not limited):

            limited.append(True)
            kwargs['options'] = {'maxfev': 50}

        return original(fun, x0, args=args, **kwargs)

    scipy.optimize.minimize = limitedMinimize

    try:
        traj.run()

    finally:
        scipy.optimize.minimize = original

    # Compare the radiant with the direction of motion at the solver's first point
    t_beg = (traj.jdt_ref - JDT_REF)*86400.0
    radiant_error = 3600*np.degrees(angleBetweenVectors(traj.radiant_eci_mini, \
        integrateFireball().radiantAtTime(t_beg)))

    assert limited
    assert traj.air_rotation_shift is not None
    assert radiant_error < 2.0



def testNoShiftWithoutTheOptionOrInOldPickles():
    """ Check that the solver does not move the points without the option, and that a trajectory pickled
        without it is solved as without the option.
    """

    traj_off, _, _ = solveFireball(False)

    assert traj_off.air_rotation_shift is None

    # Remove the option and the displacement, as in an old pickle
    traj = observeMeteor(integrateFireball(), noise=0.0, gravity_correction=True, calc_orbit=False)
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
        alone, is not moved, as only drag pulls it towards the velocity of the air.
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

    traj, _, _ = solveFireball(True)
    t0 = min([obs.time_data[0] for obs in traj.observations])

    drag_fit = DragFit(traj)

    shift_max = 0.0
    for obs in traj.observations:

        # Take the largest displacement of the points
        shift_max = max(shift_max, np.max(np.linalg.norm(traj.airRotationShiftAt(obs.time_data - t0, \
            traj.radiant_eci_mini), axis=1)))

        # Compute the points of both fits at the lengths of the model points
        lengths = (traj.state_vect_mini - np.array(obs.model_eci)) @ traj.radiant_eci_mini

        points = np.array([pointOnTrajectory(traj, length, t) for length, t in zip(lengths, obs.time_data)])
        points_drag = drag_fit.eciAtLength(lengths)

        # Check them against the model points across the line
        for pts in (points, points_drag):

            diff = pts - np.array(obs.model_eci)
            diff -= np.outer(diff @ traj.radiant_eci_mini, traj.radiant_eci_mini)

            assert np.max(np.linalg.norm(diff, axis=1)) < 1.0

    assert shift_max > 50.0



if __name__ == "__main__":

    # Run all tests
    test_list = [testTheShiftIsZeroAtAConstantVelocity, testTheShiftIsAcrossTheLine, \
        testTheShiftGoesOnAfterTheLastPoint, testTheShiftGivesTheDeviationOfAnIntegratedFireball, \
        testTheSolverRecoversAFireballInTheRotatingAir, testTheLengthsAreTakenOnTheMovedLine, \
        testTheDragFitContinuesTheShiftBelowItsLastPoint, testAFitWhichRunsOutOfEvaluationsIsContinued, \
        testNoShiftWithoutTheOptionOrInOldPickles, testAMeteorWithoutDecelerationIsNotMoved, \
        testAVelocityChangedByGravityAloneIsNotMoved, testTheDynamicMassFitPointsFollowTheShift]

    for test in test_list:
        test()
        print("PASS {:s}".format(test.__name__))
