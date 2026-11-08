""" Tests of the free fall of the meteor from the fitted line in the trajectory solver.

applyGravityDrop drops every point by g*t^2/2 along its own vertical. The meteor falls from the line along the
whole path, while the vertical turns with the curvature of the Earth, so on a fireball of several seconds the
line tilts by a few arcseconds. The solver now drops the points by the free fall from the line under the
gravity of the Earth, a point mass with its oblateness (J2), with the velocity of drag alone (freeFallDrop,
freeFallTable). The tests check the free fall against fireballs integrated in 3D with drag and gravity (with
and without J2), against a dense integration where the
fitted velocity changes its slope and along a long path, and against the scale of a decelerating meteor; they
solve fireballs without drag and with strong drag with and without it, and check the option, old pickles and
the drag fit and the dynamic mass fit, which apply the same drop.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_GravityFreeFall.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np

import matplotlib
matplotlib.use('Agg')

from wmpl.Trajectory.Trajectory import G, applyGravityDrop, freeFallDrop, freeFallMoments, freeFallTable, \
    gravityDropScale
from wmpl.Trajectory.Trajectory import GRAVITY_FREE_FALL_J2, GRAVITY_FREE_FALL_RE
from wmpl.Trajectory.Tests.SyntheticTrajectory import GM_EARTH, JDT_REF, BentDragMeteor, BentDragMeteorJ2, \
    observeMeteor, oblatenessAcceleration
from wmpl.Utils.DragFit import DragFit
from wmpl.Utils.DynamicMassFit import pointOnTrajectory


# Fireballs of the tests: 16 km/s at 30 deg for 6 s, without drag and slowing down to 4.2 km/s
FIREBALL_NO_DRAG = (16000.0, 120.0, 30.0, 0.0, 0.0, 4.0, 10.0)
FIREBALL_DRAG = (16000.0, 120.0, 30.0, 0.02, 0.15, 4.0, 10.0)

# Integrated fireballs, shared by the tests
_METEORS = {}

# Mass of the Earth used by applyGravityDrop and freeFallDrop (kg)
EARTH_MASS = 5.9722e24



def _meteor(params, j2=True):
    """ Integrate a fireball of the tests once.

    Arguments:
        params: [tuple] Arguments of BentDragMeteor.

    Keyword arguments:
        j2: [bool] With the oblateness of the Earth in its gravity (BentDragMeteorJ2). True by default.

    Return:
        [BentDragMeteor object] The fireball.
    """

    if (params, j2) not in _METEORS:
        _METEORS[(params, j2)] = (BentDragMeteorJ2 if j2 else BentDragMeteor)(*params)

    return _METEORS[(params, j2)]



def _dragVelocity(meteor, t_start):
    """ Compute the velocity of drag alone of an integrated fireball in its visible part: the speed which
        decreases at the rate drag slows down the velocity, |a - g|/|v|, without the speed gravity adds.

    Arguments:
        meteor: [BentDragMeteor object] The fireball.
        t_start: [float] Time from the beginning of the visible part at which the times start (s).

    Return:
        (time_rel, pos, vel, drag_vel): [tuple of ndarrays] Times from t_start (s), positions (m), velocities
            (m/s, one row per time) and the velocity of drag alone (m/s).
    """

    use = (meteor.time >= meteor.t_vis_beg + t_start - 1e-9) & (meteor.time <= meteor.t_vis_end + 1e-9)
    time_rel = meteor.time[use] - meteor.t_vis_beg - t_start
    pos, vel = meteor.pos[use], meteor.vel[use]

    acc = np.gradient(vel, time_rel, axis=0)
    grav = -GM_EARTH*pos/np.linalg.norm(pos, axis=1)[:, None]**3
    if getattr(meteor, 'j2', False):
        grav += np.array([oblatenessAcceleration(p) for p in pos])
    rate = np.linalg.norm(acc - grav, axis=1)/np.linalg.norm(vel, axis=1)

    decay = np.r_[0.0, np.cumsum(0.5*(rate[1:] + rate[:-1])*np.diff(time_rel))]
    drag_vel = np.linalg.norm(vel[0])*np.exp(-decay)

    return time_rel, pos, vel, drag_vel



def _denseFreeFall(eci_coord, t, radiant_eci, table, n=200001):
    """ Integrate the free fall of freeFallDrop with the trapezoidal rule on a dense grid.

    Arguments:
        eci_coord: [ndarray] Point of the line at the time t (m).
        t: [float] Time from the beginning of the meteor (s).
        radiant_eci: [ndarray] Unit vector of the radiant of the line.
        table: [tuple] Table of the motion along the line (see freeFallTable).

    Keyword arguments:
        n: [int] Number of points of the grid.

    Return:
        [ndarray] The displacement from the line (m).
    """

    time_tab, vel_tab, len_tab, _ = table

    q = np.linspace(0.0, t, n)
    back = np.interp(t, time_tab, len_tab) - np.interp(q, time_tab, len_tab)
    points = eci_coord[None, :] + back[:, None]*radiant_eci[None, :]
    r2 = np.sum(points**2, axis=1)[:, None]
    oblate = 1.5*GRAVITY_FREE_FALL_J2*GRAVITY_FREE_FALL_RE**2/r2
    polar = 5.0*points[:, 2:3]**2/r2
    grav = -G*EARTH_MASS*points/r2**1.5*(1.0 + oblate*(1.0 - polar))
    grav[:, 2:3] = -G*EARTH_MASS*points[:, 2:3]/r2**1.5*(1.0 + oblate*(3.0 - polar))

    return np.trapz((back/np.interp(q, time_tab, vel_tab))[:, None]*grav, q, axis=0)



def _solve(params, gravity_free_fall, seed=1):
    """ Solve a noise-free fireball of the tests with the gravity correction.

    Arguments:
        params: [tuple] Arguments of BentDragMeteor.
        gravity_free_fall: [bool] Drop the points by the free fall from the line.

    Keyword arguments:
        seed: [int] Seed of the times of the frames.

    Return:
        (traj, radiant_error, perp_errors):
            - traj: [Trajectory object] Solved trajectory.
            - radiant_error: [float] Angle between the solved radiant and the radiant at the solver's first
                frame (arcsec).
            - perp_errors: [ndarray] Distances of the model points from the truth across the path (m).
    """

    meteor = _meteor(params)

    traj = observeMeteor(meteor, noise=0.0, seed=seed, gravity_correction=True, calc_orbit=False, \
        gravity_free_fall=gravity_free_fall)
    traj.run()

    # The solver takes its first frame as the reference time
    t_off = (traj.jdt_ref - JDT_REF)*86400.0
    radiant_error = 3600*np.degrees(np.linalg.norm(traj.radiant_eci_mini - meteor.radiantAtTime(t_off)))

    perp_errors = []
    for obs in traj.observations:

        diff = np.array(obs.model_eci) - meteor.eciAtTime(obs.time_data + t_off)
        directions = np.array([-meteor.radiantAtTime(t + t_off) for t in obs.time_data])
        diff = diff - np.sum(diff*directions, axis=1)[:, None]*directions

        perp_errors.append(np.linalg.norm(diff, axis=1))

    return traj, radiant_error, np.concatenate(perp_errors)



def testTheFreeFallFollowsIntegratedFireballs():
    """ The free fall from the tangent at the first point, with the velocity of drag alone, follows fireballs
        integrated in 3D with the oblateness of the Earth, without drag and with strong drag, while the drop
        along the vertical of the point, scaled for the deceleration, misses them across the path. The same
        fireballs integrated without the oblateness differ from the free fall by more than 0.2 m.
    """

    for params in (FIREBALL_NO_DRAG, FIREBALL_DRAG):

        meteor = _meteor(params)
        time_rel, pos, vel, drag_vel = _dragVelocity(meteor, 0.02)
        table = freeFallTable(time_rel, drag_vel)

        x0, direction = pos[0], vel[0]/np.linalg.norm(vel[0])
        r0, v0z = np.linalg.norm(x0), np.dot(vel[0], x0/np.linalg.norm(x0))

        worst_free_fall, worst_vertical = 0.0, 0.0
        for i in range(0, len(time_rel), 100):

            t = time_rel[i]
            line_point = x0 + direction*np.interp(t, table[0], table[2])

            # The whole displacement, along the line too
            dropped = freeFallDrop(line_point, t, -direction, *table, 1.0)
            worst_free_fall = max(worst_free_fall, np.linalg.norm(pos[i] - dropped))

            # The scaled drop along the vertical, across the line only
            scale = gravityDropScale(time_rel[:i + 1], drag_vel[:i + 1])[-1] if (i > 0) else 1.0
            diff = pos[i] - applyGravityDrop(line_point, t, r0, scale, v0z)
            worst_vertical = max(worst_vertical, np.linalg.norm(diff - np.dot(diff, direction)*direction))

        assert worst_free_fall < 0.03, worst_free_fall
        assert worst_vertical > 0.5, worst_vertical

        # The oblateness of the Earth is in the free fall
        meteor_spherical = _meteor(params, j2=False)
        time_rel, pos, vel, drag_vel = _dragVelocity(meteor_spherical, 0.02)
        table = freeFallTable(time_rel, drag_vel)

        x0, direction = pos[0], vel[0]/np.linalg.norm(vel[0])
        worst_spherical = max(np.linalg.norm(pos[i] - freeFallDrop(x0 + direction*np.interp(time_rel[i], \
            table[0], table[2]), time_rel[i], -direction, *table, 1.0)) for i in range(0, len(time_rel), 100))

        assert worst_spherical > 0.2, worst_spherical



def testTheFreeFallIsExactWhereTheVelocityChangesItsSlope():
    """ The free fall agrees with a dense integration for a velocity which changes its slope, as the fitted
        velocity does, and at a constant velocity along a path of 300 km.
    """

    radiant_eci = np.array([0.3, 0.2, 0.93])
    radiant_eci /= np.linalg.norm(radiant_eci)
    eci_coord = np.array([6.45e6, 1.0e5, 2.0e5])

    # A velocity linear in time over segments, as the fitted one
    time_rel = np.linspace(0.0, 8.0, 1001)
    vel = np.interp(time_rel, [0.0, 2.0, 4.5, 6.5, 8.0], [15000.0, 14800.0, 12000.0, 6000.0, 3500.0])
    table = freeFallTable(time_rel, vel)

    for t in (0.05, 1.3, 4.5, 6.1, 7.9, 9.5):
        drop = freeFallDrop(eci_coord, t, radiant_eci, *table, 1.0) - eci_coord
        assert np.linalg.norm(drop - _denseFreeFall(eci_coord, t, radiant_eci, table)) < 1e-3, t

    # A constant velocity of 20 km/s for 15 s
    table = freeFallTable([0.0], [20000.0])
    for t in (5.0, 10.0, 15.0):
        drop = freeFallDrop(eci_coord, t, radiant_eci, *table, 1.0) - eci_coord
        assert np.linalg.norm(drop - _denseFreeFall(eci_coord, t, radiant_eci, table, n=400001)) < 1e-3, t

    # No drop at the beginning and before it, and the gravity factor multiplies the drop
    assert np.array_equal(freeFallDrop(eci_coord, 0.0, radiant_eci, *table, 1.0), eci_coord)
    assert np.array_equal(freeFallDrop(eci_coord, -1.0, radiant_eci, *table, 1.0), eci_coord)
    half = freeFallDrop(eci_coord, 5.0, radiant_eci, *table, 0.5) - eci_coord
    full = freeFallDrop(eci_coord, 5.0, radiant_eci, *table, 1.0) - eci_coord
    assert np.allclose(half, 0.5*full, rtol=1e-9, atol=0)



def testTheDropOfAUniformFieldIsTheScaledDrop():
    """ The integral which multiplies a uniform gravity in the free fall is the drop at a constant velocity
        times the scale of a decelerating meteor (gravityDropScale), and t^2/2 at a constant velocity.
    """

    # A smooth deceleration from 16 to 7.2 km/s
    time_rel = np.linspace(0.0, 6.0, 1001)
    vel = 16000.0 - 600.0*time_rel**1.5
    table = freeFallTable(time_rel, vel)
    scale = gravityDropScale(time_rel, vel)

    for i in (100, 500, 1000):

        t = time_rel[i]
        length, j0, j1, _, _, _ = freeFallMoments(t, *table)

        assert abs((length*j0 - j1) - scale[i]*t**2/2) < 1e-4*t**2/2, i

    table = freeFallTable([0.0], [16000.0])
    for t in (0.5, 3.0, 6.0):
        length, j0, j1, _, _, _ = freeFallMoments(t, *table)
        assert abs((length*j0 - j1) - t**2/2) < 1e-12*t**2, t



def testTheSolverRecoversAFireballWithoutDrag():
    """ The solver recovers the radiant and the positions of a fireball without drag with the free fall, and
        not with the drop along the vertical of the point.
    """

    traj, radiant_error, perp_errors = _solve(FIREBALL_NO_DRAG, True)

    assert traj.gravityFreeFall() is not None
    assert radiant_error < 0.1, radiant_error
    assert np.max(perp_errors) < 0.1, np.max(perp_errors)

    traj_vertical, radiant_error_vertical, perp_errors_vertical = _solve(FIREBALL_NO_DRAG, False)

    assert traj_vertical.gravityFreeFall() is None
    assert radiant_error_vertical > 2.0, radiant_error_vertical
    assert np.max(perp_errors_vertical) > 0.5, np.max(perp_errors_vertical)



def testTheSolverRecoversAFireballWithStrongDrag():
    """ The solver recovers the radiant and the positions of a fireball which slows down from 16 to 4.2 km/s
        with the free fall at the fitted velocity, better than with the scaled drop along the vertical.
    """

    traj, radiant_error, perp_errors = _solve(FIREBALL_DRAG, True)

    assert traj.gravity_drop_velocity is not None
    assert radiant_error < 0.3, radiant_error
    assert np.max(perp_errors) < 0.25, np.max(perp_errors)

    _, radiant_error_vertical, perp_errors_vertical = _solve(FIREBALL_DRAG, False)

    assert radiant_error_vertical > 1.5, radiant_error_vertical
    assert np.max(perp_errors_vertical) > 0.3, np.max(perp_errors_vertical)



def testTheOptionOffAndOldPicklesDropAlongTheVertical():
    """ Without the option, and in trajectories pickled before it was added, the points are dropped along
        their vertical, scaled for the deceleration, as before.
    """

    traj, _, _ = _solve(FIREBALL_DRAG, True)

    eci_coord = traj.state_vect_mini - 50e3*traj.radiant_eci_mini
    r0, v0z = traj.gravity_r0, traj.v0z

    for remove in (False, True):

        if remove:
            del traj.gravity_free_fall
        else:
            traj.gravity_free_fall = False

        assert traj.gravityFreeFall() is None

        for t in (0.5, 2.0, 5.0):
            dropped = traj.gravityDropPoint(eci_coord, t, traj.radiant_eci_mini, r0, v0z, \
                traj.gravityFreeFall())
            assert np.array_equal(dropped, applyGravityDrop(eci_coord, t, r0, traj.gravityDropFactor(t), v0z))



def testTheDragFitAndTheDynamicMassFitDropAsTheSolver():
    """ The drag fit drops its grid, and the dynamic mass fit its points across the line, by the free fall of
        the solver.
    """

    traj, _, _ = _solve(FIREBALL_DRAG, True)

    t0 = min(obs.time_data[0] for obs in traj.observations)
    free_fall = traj.gravityFreeFall()
    radiant_eci = traj.radiant_eci_mini

    # The drop of the grid of the drag fit down to its last point
    drag_fit = DragFit(traj)
    line_eci = drag_fit._lineEci(drag_fit.len_grid)
    checked = 0
    for i in range(drag_fit.i_last + 1):

        t = drag_fit.time_grid[i]
        if np.isfinite(t) and (t > t0):

            dropped = traj.gravityDropPoint(line_eci[i], t - t0, drag_fit.radiant_eci, None, None, free_fall)
            assert np.allclose(drag_fit.drop_grid[i], line_eci[i] - dropped, rtol=0, atol=1e-4)
            checked += 1

    assert checked > 100

    for length, t in ((20e3, 1.5), (60e3, 4.0)):

        line_point = traj.state_vect_mini - length*radiant_eci
        drop = traj.gravityDropPoint(line_point, t - t0, radiant_eci, None, None, free_fall) - line_point
        drop_across = drop - np.dot(drop, radiant_eci)*radiant_eci

        assert np.allclose(pointOnTrajectory(traj, length, t), line_point + drop_across, rtol=0, atol=1e-6)



if __name__ == "__main__":

    testTheFreeFallFollowsIntegratedFireballs()
    testTheFreeFallIsExactWhereTheVelocityChangesItsSlope()
    testTheDropOfAUniformFieldIsTheScaledDrop()
    testTheSolverRecoversAFireballWithoutDrag()
    testTheSolverRecoversAFireballWithStrongDrag()
    testTheOptionOffAndOldPicklesDropAlongTheVertical()
    testTheDragFitAndTheDynamicMassFitDropAsTheSolver()

    print("All tests passed.")
