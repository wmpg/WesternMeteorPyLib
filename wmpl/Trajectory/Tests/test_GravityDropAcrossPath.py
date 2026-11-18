""" Tests of the gravity drop across the path, slowed by the drag, in the trajectory solver.

The meteoroids are integrated exactly in ECI coordinates, under the central gravity of the Earth, the drag
and the mass loss (SyntheticTrajectory.GravityDragMeteor), and observed without noise from three stations.
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

from wmpl.Trajectory.Trajectory import applyGravityDrop, solutionDropTimeOrigin
from wmpl.Trajectory.Tests.SyntheticTrajectory import GravityDragMeteor, observeMeteor, JDT_REF
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



if __name__ == "__main__":

    testDeceleratingFireball()
    testShallowFireball()
    testSolutionsWithoutTheSpeedsDropVertically()

    print("All tests passed.")
