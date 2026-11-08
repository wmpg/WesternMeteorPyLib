""" Tests of the gravity drop of a decelerating meteor in the trajectory solver.

Drag also slows down the velocity which gravity adds across the path, so a decelerating meteor drops less
than one at a constant velocity, which applyGravityDrop assumes. The solver scales the drop with a velocity
fitted to the lengths without a deceleration model such as the Jacchia fit (gravityDropScale,
segmentedVelocity, Trajectory.calcGravityDropScale). The tests check the scale against analytic cases and
against a fireball integrated in 3D with drag and gravity, solve that fireball with and without the scale,
and check the velocity of a fireball which fragments, of a meteor at a constant velocity, of one whose
velocity changes by gravity alone, and of few noisy points.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_GravityDropDeceleration.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np

import matplotlib
matplotlib.use('Agg')

from wmpl.Trajectory.Trajectory import GRAVITY_DROP_MAX_ACC, GRAVITY_DROP_MIN_VEL, Trajectory, \
    gravityDropScale, segmentedVelocity
from wmpl.Trajectory.Tests.SyntheticTrajectory import GM_EARTH, JDT_REF, BentDragMeteor, observeMeteor, \
    observeStraightMeteor
from wmpl.Utils.Math import angleBetweenVectors


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



def _decelerate(traj, v_beg, v_end):
    """ Replace the lengths of the points of a solved trajectory by those of a meteor which slows down
        linearly in time.

    Arguments:
        traj: [Trajectory object] Solved trajectory.
        v_beg: [float] Velocity at the first point (m/s).
        v_end: [float] Velocity at the last point (m/s).

    Return:
        (t0, t_end): [tuple of float] Times of the first and of the last point (s).
    """

    t0 = min([obs.time_data[0] for obs in traj.observations])
    t_end = max([obs.time_data[-1] for obs in traj.observations])
    decel = (v_beg - v_end)/(t_end - t0)

    for obs in traj.observations:
        t_rel = obs.time_data - t0
        obs.state_vect_dist = v_beg*t_rel - 0.5*decel*t_rel**2

    return t0, t_end



def _solveFireball(gravity_decel, jacchia_fit=None):
    """ Solve the noise-free fireball with the gravity correction.

    Arguments:
        gravity_decel: [bool] Scale the drop for the deceleration.

    Keyword arguments:
        jacchia_fit: [ndarray] Parameters a1 and a2 which the Jacchia fit of the lag returns instead of the
            fit, for every station and for all of them. None by default, the fit is done.

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

    if jacchia_fit is None:
        traj.run()

    else:

        # Replace the Jacchia fit on the class of this object (wmpl loads every module twice)
        traj_class = type(traj)
        original = traj_class.fitJacchiaLag

        def _replacedFit(self, observations):
            for obs in observations:
                obs.jacchia_fit = np.array(jacchia_fit)
            return np.array(jacchia_fit)

        traj_class.fitJacchiaLag = _replacedFit

        try:
            traj.run()

        finally:
            traj_class.fitJacchiaLag = original

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

    # The lengths of a decelerating meteor
    _decelerate(traj, 20000.0, 10000.0)

    traj.gravity_decel = True
    traj.calcGravityDropScale(traj.observations)
    assert traj.gravity_drop_scale is not None

    traj.gravity_decel = False
    traj.calcGravityDropScale(traj.observations)
    assert traj.gravity_drop_scale is None

    del traj.gravity_decel
    traj.calcGravityDropScale(traj.observations)
    assert traj.gravity_drop_scale is None



def testTheVelocityIsHeldAboveWhereAblationStops():
    """ Check that a meteor which slows down below GRAVITY_DROP_MIN_VEL at the end scales the drop with the
        velocity held at GRAVITY_DROP_MIN_VEL.
    """

    traj, _, _ = observeStraightMeteor(gravity_correction=True)
    traj.run()

    # The lengths of a meteor which slows down linearly from 20 to 1 km/s
    t0, t_end = _decelerate(traj, 20000.0, 1000.0)

    traj.calcGravityDropScale(traj.observations)
    time_rel, scale = traj.gravity_drop_scale

    vel = 20000.0 - 19000.0*time_rel/(t_end - t0)
    held = gravityDropScale(time_rel, np.maximum(vel, GRAVITY_DROP_MIN_VEL))[-1]
    not_held = gravityDropScale(time_rel, vel)[-1]

    assert np.all(np.isfinite(scale))
    assert abs(scale[-1] - held) < 0.002
    assert abs(held - not_held) > 0.02

    # The scale is the same when the times do not start at 0, as when the first point is not at the
    #   reference time
    for obs in traj.observations:
        obs.time_data = obs.time_data + 0.7

    traj.calcGravityDropScale(traj.observations)
    assert np.allclose(traj.gravity_drop_scale[0], time_rel)
    assert np.allclose(traj.gravity_drop_scale[1], scale)



def _lengths(profile, times, duration):
    """ Integrate a velocity profile to the lengths at the given times.

    Arguments:
        profile: [function] Velocity (m/s) as a function of the time (s).
        times: [ndarray] Times of the points (s).
        duration: [float] Duration of the meteor (s).

    Return:
        [ndarray] Lengths at the times of the points (m).
    """

    time_fine = np.linspace(0.0, duration, 80001)
    vel_fine = profile(time_fine)
    length_fine = np.r_[0.0, np.cumsum(0.5*(vel_fine[1:] + vel_fine[:-1])*np.diff(time_fine))]

    return np.interp(times, time_fine, length_fine)



def _twoStations(duration, fps=25.0):
    """ Times of the points of two stations, one until 70% of the duration and one from 25%, whose frames
        are not simultaneous.

    Arguments:
        duration: [float] Duration of the meteor (s).

    Keyword arguments:
        fps: [float] Frame rate (frames per second). 25 by default.

    Return:
        (times, n_first): [tuple] Times of the points (s), and the number of points of the first station.
    """

    times_first = np.arange(0.0, 0.7*duration, 1.0/fps)
    times_second = np.arange(0.25*duration, duration, 1.0/fps) + 0.37/fps

    return np.r_[times_first, times_second], len(times_first)



def testTheVelocityFollowsAFragmentingFireball():
    """ Check the velocity fitted to noisy lengths of two stations against the true velocity of a fireball
        which fragments, a deceleration which an exponential does not follow.
    """

    duration = 8.0
    time_eval = np.linspace(0.0, duration, 1001)

    # A slow deceleration, and 4 km/s lost in about 1 s at 5 s
    def profile(t):
        return 16000.0 - 300.0*t - 2000.0*(1 + np.tanh((t - 5.0)/0.3))

    vel_true = profile(time_eval)

    for seed in range(3):

        # 50 m of noise, and an offset of 30 m between the stations
        rng = np.random.default_rng(seed)
        times, n_first = _twoStations(duration)
        lengths = _lengths(profile, times, duration) + rng.normal(0.0, 50.0, len(times))
        lengths[n_first:] += 30.0

        vel = segmentedVelocity(times, lengths, time_eval)

        assert np.sqrt(np.mean((vel - vel_true)**2)) < 100.0
        assert abs(gravityDropScale(time_eval, vel)[-1] - gravityDropScale(time_eval, vel_true)[-1]) < 0.002

        # Outside the times of the points, the velocity at the nearest point
        outside = segmentedVelocity(times, lengths, np.array([-1.0, np.min(times), np.max(times), \
            np.max(times) + 1.0]))
        assert (outside[0] == outside[1]) and (outside[2] == outside[3])

    # Fewer than 3 distinct times give no velocity
    assert segmentedVelocity(np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 2.0]), time_eval) is None



def testNoChangeOfVelocityIsFoundAtAConstantVelocity():
    """ Check that noisy lengths of a meteor at a constant velocity give a constant velocity, so its drop is
        not scaled.
    """

    duration = 3.0
    time_eval = np.linspace(0.0, duration, 1001)

    for seed in range(10):

        rng = np.random.default_rng(seed)
        times, n_first = _twoStations(duration)
        lengths = 15000.0*times + rng.normal(0.0, 100.0, len(times))
        lengths[n_first:] += 30.0

        vel = segmentedVelocity(times, lengths, time_eval)

        # The offset between the stations moves the velocity by a few tens of m/s
        assert np.ptp(vel) == 0
        assert abs(vel[0] - 15000.0) < 50.0
        assert abs(gravityDropScale(time_eval, vel)[-1] - 1.0) < 1e-12



def testAChangeOfVelocityByGravityAloneIsFollowed():
    """ Check that the velocity follows a meteor whose velocity changes by gravity alone, up or down by
        8 m/s^2, as it may increase as fast as gravity can speed a meteor up.
    """

    duration = 10.0
    time_eval = np.linspace(0.0, duration, 1001)

    for acc in (8.0, -8.0):

        def profile(t):
            return 11000.0 + acc*t

        scale_true = gravityDropScale(time_eval, profile(time_eval))[-1]
        assert abs(scale_true - 1.0) > 0.002

        for seed in range(5):

            rng = np.random.default_rng(seed)
            times, _ = _twoStations(duration)
            lengths = _lengths(profile, times, duration) + rng.normal(0.0, 30.0, len(times))

            vel = segmentedVelocity(times, lengths, time_eval)

            assert np.all(np.diff(vel)/np.diff(time_eval) <= GRAVITY_DROP_MAX_ACC + 1e-6)
            assert abs(gravityDropScale(time_eval, vel)[-1] - scale_true) < 0.0008



def testFewNoisyPointsDoNotMakeTheVelocityOscillate():
    """ Check that few noisy points of three stations, with gaps and offsets between them, give a velocity
        which does not increase faster than gravity can speed a meteor up, nor leave the range of the true
        one.
    """

    duration = 3.4
    time_eval = np.linspace(0.0, duration, 1001)

    def profile(t):
        return 70000.0 - 3000.0*t

    for seed in range(10):

        # A station which sees the beginning, one with a low frame rate and one near the end
        rng = np.random.default_rng(seed)
        times = [np.arange(0.0, 0.6, 0.04), np.arange(0.3, duration, 0.15) + 0.01, \
            np.arange(2.4, 3.1, 0.04) + 0.02]
        lengths = [_lengths(profile, t, duration) + rng.normal(0.0, 400.0, len(t)) for t in times]
        lengths[1] += 300.0
        lengths[2] -= 300.0

        vel = segmentedVelocity(np.concatenate(times), np.concatenate(lengths), time_eval)

        assert np.all(np.diff(vel)/np.diff(time_eval) <= GRAVITY_DROP_MAX_ACC + 1e-6)
        assert (np.min(vel) > 55000.0) and (np.max(vel) < 75000.0)



def testTheScaleDoesNotUseTheJacchiaFit():
    """ Check that the scale and the solution do not change when the Jacchia fit of the lag is replaced by
        one far from the deceleration of the fireball.
    """

    traj, radiant_error, _ = _solveFireball(True)
    traj_jacchia, radiant_error_jacchia, _ = _solveFireball(True, jacchia_fit=[500.0, 5.0])

    assert np.allclose(traj_jacchia.jacchia_fit, [500.0, 5.0])
    assert np.array_equal(traj.gravity_drop_scale[1], traj_jacchia.gravity_drop_scale[1])
    assert np.array_equal(traj.radiant_eci_mini, traj_jacchia.radiant_eci_mini)
    assert radiant_error == radiant_error_jacchia



def testAMeteorWithoutDecelerationIsNotScaled():
    """ Check that a straight meteor at a constant velocity keeps the drop at a constant velocity, without
        noise and with about 20 arcsec of noise. With noise, the lengths show no change of velocity, and the
        solution is the same as without the scale.
    """

    for kwargs in (dict(noise=0.0), dict(seed=1), dict(seed=2)):

        traj, _, _ = observeStraightMeteor(gravity_correction=True, **kwargs)
        traj.run()

        assert (traj.gravity_drop_scale is None) or (np.min(traj.gravity_drop_scale[1]) > 0.999)

        if 'seed' in kwargs:

            unscaled, _, _ = observeStraightMeteor(gravity_correction=True, gravity_decel=False, **kwargs)
            unscaled.run()

            assert traj.gravity_drop_scale is None
            assert np.array_equal(traj.state_vect_mini, unscaled.state_vect_mini)
            assert np.array_equal(traj.radiant_eci_mini, unscaled.radiant_eci_mini)



def testTheSpeedGravityAddsAlongThePathDoesNotScaleTheDrop():
    """ Check that a noise-free meteor at a constant velocity, dropped by gravity, is solved as well with the
        scale as without it, also when the picks of the first 2 s are ignored. Gravity also speeds the meteor
        up along its path, which does not bend it, and the lengths on the straight line of the first pass
        hold that speed: when they gave the scale, the radiant was 0.10 arcsec off, and 0.71 arcsec with the
        first picks ignored, against 0.004 and 0.003 arcsec without the scale. The second limit leaves room
        for the line-of-sight fit, which can stop away from the minimum with these picks (see #154).
    """

    for ignore_before, radiant_limit in ((0.0, 0.03), (2.0, 0.2)):

        errors = []
        for gravity_decel in (False, True):

            traj, _, radiant_eci = observeStraightMeteor(noise=0.0, duration=5.0, v_init=15000.0, \
                elevation=30.0, gravity_drop=True, estimate_timing_vel=False, gravity_correction=True, \
                filter_picks=False, calc_orbit=False)

            # Solve the observations again, ignoring the picks before the given time
            solved = Trajectory(traj.jdt_ref, meastype=2, max_toffset=1.0, estimate_timing_vel=False, \
                gravity_correction=True, gravity_decel=gravity_decel, filter_picks=False, monte_carlo=False, \
                calc_orbit=False, show_plots=False, save_results=False, verbose=False)

            for obs in traj.observations:
                solved.infillTrajectory(obs.azim_data, obs.elev_data, obs.time_data, obs.lat, obs.lon, \
                    obs.ele, station_id=obs.station_id, \
                    ignore_list=(obs.time_data < ignore_before).astype(np.uint8))

            solved = solved.run()

            errors.append(3600*np.degrees(angleBetweenVectors(solved.radiant_eci_mini, radiant_eci)))

        # Check the radiant with the scale, against the truth and against the solution without it
        assert errors[1] < radiant_limit
        assert errors[1] < errors[0] + 0.01




if __name__ == "__main__":

    # Run all tests
    test_list = [testTheScaleIsOneAtAConstantVelocity, testTheScaleOfALinearDeceleration, \
        testTheScaleGivesTheDropOfABentPath, testTheSolverRecoversABentFireball, \
        testTheDropFactorIsTheGravityFactorTimesTheScale, testNoScaleWithoutTheOptionOrInOldPickles, \
        testTheVelocityIsHeldAboveWhereAblationStops, testTheVelocityFollowsAFragmentingFireball, \
        testNoChangeOfVelocityIsFoundAtAConstantVelocity, testAChangeOfVelocityByGravityAloneIsFollowed, \
        testFewNoisyPointsDoNotMakeTheVelocityOscillate, \
        testTheScaleDoesNotUseTheJacchiaFit, testAMeteorWithoutDecelerationIsNotScaled, \
        testTheSpeedGravityAddsAlongThePathDoesNotScaleTheDrop]

    for test in test_list:
        test()
        print("PASS {:s}".format(test.__name__))
