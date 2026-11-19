""" Tests of the gravity drop across the path, slowed by the drag, in the trajectory solver.

The meteoroids are integrated exactly in ECI coordinates, under the central gravity of the Earth, the drag
and the mass loss (SyntheticTrajectory.GravityDragMeteor), with the air at rest or rotating with the Earth,
and observed without noise from three stations.
The solver must recover their radiant and their positions.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_GravityDropAcrossPath.py -v
"""

from __future__ import print_function, division, absolute_import

import contextlib
import io

import numpy as np

import matplotlib
matplotlib.use('Agg')

from wmpl.Trajectory.Trajectory import applyGravityDrop, gravityDropVectors, solutionDropTimeOrigin
from wmpl.Trajectory.Tests.SyntheticTrajectory import GravityDragMeteor, observeMeteor, observeStraightMeteor, \
    JDT_REF
from wmpl.Utils.DragFit import DragFit
from wmpl.Utils.DynamicMassFit import pointOnTrajectory
from wmpl.Utils.Math import angleBetweenVectors, vectMag



def _solve(meteor):
    """ Solve the noise-free observations of a synthetic meteor with the gravity correction.

    Arguments:
        meteor: [GravityDragMeteor object] Synthetic meteor.

    Return:
        (traj, radiant_error, position_error):
            - traj: [Trajectory object] Solved trajectory.
            - radiant_error: [float] Angle between the solved radiant and the true direction of the motion at
                the time 0 of the solution (arcsec).
            - position_error: [float] Largest distance of a model point from the true position (m).
    """

    # The pick filtering is off, as on noise-free data it rejects picks by the numerical noise
    traj = observeMeteor(meteor, noise=0.0, gravity_correction=True, filter_picks=False, calc_orbit=False, \
        estimate_timing_vel=False)
    with contextlib.redirect_stdout(io.StringIO()):
        traj = traj.run()

    # The solution moves the time 0 to the first time of the used stations
    t_shift = (traj.jdt_ref - JDT_REF)*86400.0

    radiant_true = -meteor.velocityAtTime(t_shift)[0]
    radiant_error = 3600*np.degrees(angleBetweenVectors(traj.radiant_eci_mini, radiant_true))

    position_error = max(np.max(np.linalg.norm(np.array(obs.model_eci) \
        - meteor.eciAtTime(obs.time_data + t_shift), axis=1)) for obs in traj.observations)

    return traj, radiant_error, position_error



def testDeceleratingFireball():
    """ Check a fireball which slows from 13 to 3.3 km/s in 7 s. Dropped vertically by g*t^2/2 without the
        drag, its radiant was off by 22 arcsec and its positions by 23 m.
    """

    meteor = GravityDragMeteor(13000.0, 210.0, 60.0, 50.0, 10.0, ht_beg=85000.0)
    _, radiant_error, position_error = _solve(meteor)

    print("Radiant error: {:.2f} arcsec, position error {:.2f} m".format(radiant_error, position_error))

    assert radiant_error < 1.0
    assert position_error < 1.0



def testShallowFireball():
    """ Check a fireball which flies for 12 s at 15 deg, over which the vertical turns by 1.2 deg with the
        curvature of the Earth, so the direction of gravity changes along the path. Dropped vertically by
        g*t^2/2, its radiant was off by 9 arcsec and its positions by 27 m.
    """

    meteor = GravityDragMeteor(15000.0, 300.0, 15.0, 100.0, 12.0, ht_beg=95000.0)
    _, radiant_error, position_error = _solve(meteor)

    print("Radiant error: {:.2f} arcsec, position error {:.2f} m".format(radiant_error, position_error))

    assert radiant_error < 1.0
    assert position_error < 3.0



def testDeceleratingFireballInRotatingAir():
    """ Check the decelerating fireball in the air which rotates with the Earth. The drag acts against the
        velocity relative to the air, so it pulls the fireball across its line. With the air at rest in the
        solver, its radiant was off by 525 arcsec and its positions by 284 m.
    """

    meteor = GravityDragMeteor(13000.0, 210.0, 60.0, 50.0, 10.0, ht_beg=85000.0, air_rotation=True)
    _, radiant_error, position_error = _solve(meteor)

    print("Radiant error: {:.2f} arcsec, position error {:.2f} m".format(radiant_error, position_error))

    assert radiant_error < 3.0
    assert position_error < 1.5



def testDragFitDirectionInRotatingAir():
    """ Check that the directions of the radiant of the drag fit follow its path in the air which rotates with
        the Earth, which turns the direction of motion of a decelerating fireball by several degrees at the
        end.
    """

    meteor = GravityDragMeteor(13000.0, 210.0, 60.0, 50.0, 10.0, ht_beg=85000.0, air_rotation=True)
    traj, _, _ = _solve(meteor)

    with contextlib.redirect_stdout(io.StringIO()):
        drag_fit = DragFit(traj)

    # Compare the tangent of the path with the direction of the radiant along the observed part
    max_error = 0.0
    for length in np.linspace(drag_fit.len_first + 1000.0, drag_fit.len_last - 1000.0, 10):

        tangent = drag_fit.eciAtLength(length - 10.0)[0] - drag_fit.eciAtLength(length + 10.0)[0]
        max_error = max(max_error, np.degrees(angleBetweenVectors(tangent, \
            drag_fit.radiantAtLength(length)[0])))

    # Take the turn by the air at the last point, from the direction bent by gravity only
    theta = np.interp(drag_fit.len_last, drag_fit.len_grid, drag_fit.bend_grid)
    air_turn = np.degrees(angleBetweenVectors(drag_fit.radiantAtLength(drag_fit.len_last)[0], \
        drag_fit._bentRadiant(np.array([theta]), drag_fit.eciAtLength(drag_fit.len_last))[0]))

    print("Largest angle between the path and the radiant: {:.4f} deg, turn by the air at the end " \
        "{:.2f} deg".format(max_error, air_turn))

    assert max_error < 0.02
    assert air_turn > 0.5



def testSolutionsWithoutTheSpeedsDropVertically():
    """ Check that a solution without the speeds of the drop, as pickled before they were added, is dropped
        vertically from its line by the dynamic mass fit, as the solver dropped it then.
    """

    meteor = GravityDragMeteor(15000.0, 210.0, 45.0, 1000.0, 6.0)
    traj, _, _ = _solve(meteor)

    traj.gravity_vel = None

    length, t = 30e3, 2.0
    P = traj.state_vect_mini - length*traj.radiant_eci_mini
    drop = applyGravityDrop(P, t - solutionDropTimeOrigin(traj), vectMag(traj.state_vect_mini), 1.0, \
        traj.v0z) - P
    drop -= np.dot(drop, traj.radiant_eci_mini)*traj.radiant_eci_mini

    assert np.allclose(pointOnTrajectory(traj, length, t), P + drop, rtol=0, atol=1e-6)



def testMonteCarloRunsWithTheGravityDrop():
    """ Check that the Monte Carlo runs, which switch between the original and the noisy observations, fit the
        speeds of the drop and solve.
    """

    traj, _, _ = observeStraightMeteor(noise=1e-5, duration=2.5, gravity_drop=True, gravity_correction=True, \
        monte_carlo=True, mc_runs=2, mc_cores=1, mc_runs_max=4, filter_picks=False, estimate_timing_vel=True)
    with contextlib.redirect_stdout(io.StringIO()):
        traj = traj.run()

    assert traj is not None
    assert traj.gravity_vel is not None



def testTheLagOfAMeteorWithoutDragIsZero():
    """ Check that the lag is the deceleration by the air only, without the length gained from gravity along
        the path, which the lengths have, so a meteor moving under gravity without drag has no lag.
    """

    traj, _, _ = observeStraightMeteor(noise=0.0, duration=5.0, v_init=15000.0, elevation=60.0, \
        gravity_drop=True, gravity_correction=True, filter_picks=False, calc_orbit=False)
    with contextlib.redirect_stdout(io.StringIO()):
        traj = traj.run()

    for obs in traj.observations:
        assert np.max(np.abs(obs.lag)) < 1.0



def testTheDropAtItsOriginIsZero():
    """ Check that the drop at the origin alone is zero, as a table of one time cannot be integrated. """

    gravity_vel = (np.array([0.0, 1.0]), np.array([15000.0, 14000.0]))
    drop = gravityDropVectors(np.array([0.0, 0.0, 6.45e6]), np.array([0.0, 0.6, 0.8]), [0.0], gravity_vel, 0.0, \
        1.0)

    assert np.all(drop == 0)



def testNoUsedPointsGiveNoSolution():
    """ Check that a solution in which every station is ignored for too few overlapping points ends without a
        trajectory, as without the gravity drop, rather than failing to fit the speeds of the drop.
    """

    meteor = GravityDragMeteor(13000.0, 210.0, 60.0, 50.0, 10.0, ht_beg=85000.0)
    traj = observeMeteor(meteor, noise=0.0, fps=0.7, seed=0, gravity_correction=True, filter_picks=False, \
        calc_orbit=False, estimate_timing_vel=False)
    with contextlib.redirect_stdout(io.StringIO()):
        traj = traj.run()

    assert traj is None



if __name__ == "__main__":

    testDeceleratingFireball()
    testShallowFireball()
    testSolutionsWithoutTheSpeedsDropVertically()
    testMonteCarloRunsWithTheGravityDrop()
    testTheLagOfAMeteorWithoutDragIsZero()
    testTheDropAtItsOriginIsZero()
    testNoUsedPointsGiveNoSolution()

    print("All tests passed.")
