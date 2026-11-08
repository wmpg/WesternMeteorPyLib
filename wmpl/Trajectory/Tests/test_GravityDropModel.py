""" Tests that the solver inverts a trajectory dropped with its own gravity drop model.

applyGravityDrop descends from the beginning of the meteor, at distance r0 from the centre of the Earth,
with the vertical velocity v0z. The line-of-sight fit and the spatial residuals passed the distance of each
point instead, and the fit always ran with v0z=None (no vertical velocity), as calcLLA computes it only after
the pass that applies the drop.

The tests drop the noise-free straight meteor of SyntheticTrajectory with applyGravityDrop, as the solver
models it, observe it from three stations, and solve it with the gravity correction.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_GravityDropModel.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np

import matplotlib
matplotlib.use('Agg')

from wmpl.Trajectory.Trajectory import Trajectory
from wmpl.Trajectory.Tests.SyntheticTrajectory import observeStraightMeteor, worstResidual
from wmpl.Utils.Math import angleBetweenVectors



def _solve(duration, v_init, elevation, estimate_timing_vel=True):
    """ Solve the noise-free straight meteor dropped by gravity.

    Arguments:
        duration: [float] Duration of the meteor (s).
        v_init: [float] Velocity of the meteor (m/s).
        elevation: [float] Elevation of the radiant at the beginning of the meteor (deg).

    Keyword arguments:
        estimate_timing_vel: [bool] Estimate the timing offsets. True by default.

    Return:
        (traj, radiant_error):
            - traj: [Trajectory object] Solved trajectory.
            - radiant_error: [float] Angle between the solved and the true radiant (arcsec).
    """

    # The pick filtering is off, as on noise-free data it rejects picks by the numerical noise
    traj, _, radiant_eci = observeStraightMeteor(noise=0.0, duration=duration, v_init=v_init, \
        elevation=elevation, gravity_drop=True, estimate_timing_vel=estimate_timing_vel, \
        gravity_correction=True, filter_picks=False, calc_orbit=False)
    traj = traj.run()

    # Compute the angle between the solved and the true radiant
    return traj, 3600*np.degrees(angleBetweenVectors(traj.radiant_eci_mini, radiant_eci))



def testASteepFastMeteorInverts():
    """ Check that a meteor of 2.5 s at 30 km/s and 55 deg is inverted. Before, the radiant was 0.25 arcsec
        off with 0.27 m of residual.
    """

    traj, radiant_error = _solve(2.5, 30000.0, 55.0)

    # Check the radiant, the residuals and the initial velocity
    assert radiant_error < 0.05
    assert worstResidual(traj) < 0.01
    assert abs(traj.v_init - 30000.0) < 0.1



def testAShallowSlowMeteorInverts():
    """ Check that a meteor of 5 s at 15 km/s and 30 deg, which drops by about 120 m, is inverted. Before,
        the radiant was 0.94 arcsec off with 0.70 m of residual.
    """

    traj, radiant_error = _solve(5.0, 15000.0, 30.0)

    # Check the radiant, the residuals and the initial velocity
    assert radiant_error < 0.1
    assert worstResidual(traj) < 0.1
    assert abs(traj.v_init - 15000.0) < 0.1



def testAShallowSlowMeteorInvertsWithoutTiming():
    """ Check that the shallow slow meteor is inverted without the timing estimation too. Before, the radiant
        was 0.92 arcsec off with 0.74 m of residual.
    """

    traj, radiant_error = _solve(5.0, 15000.0, 30.0, estimate_timing_vel=False)

    # Check the radiant, the residuals and the initial velocity
    assert radiant_error < 0.1
    assert worstResidual(traj) < 0.1
    assert abs(traj.v_init - 15000.0) < 0.1



def testAnIgnoredStationDoesNotSetTheBeginning():
    """ Check that an ignored station whose first line of sight is far above the meteor does not set the
        beginning of the gravity drop. Before, the radiant was 0.36 arcsec off.
    """

    traj, _, radiant_eci = observeStraightMeteor(noise=0.0, duration=5.0, v_init=15000.0, elevation=30.0, \
        gravity_drop=True, estimate_timing_vel=False, gravity_correction=True, filter_picks=False, \
        calc_orbit=False)

    # Raise the first line of sight of station C by 5 deg and ignore the station
    obs = traj.observations[2]
    obs.elev_data[0] += np.radians(5.0)
    obs.ignore_station = True

    # Solve the observations again with station C ignored
    solved = Trajectory(traj.jdt_ref, meastype=2, max_toffset=1.0, estimate_timing_vel=False, \
        gravity_correction=True, filter_picks=False, monte_carlo=False, calc_orbit=False, show_plots=False, \
        save_results=False, verbose=False)

    for obs in traj.observations:
        solved.infillTrajectory(obs.azim_data, obs.elev_data, obs.time_data, obs.lat, obs.lon, obs.ele, \
            station_id=obs.station_id, ignore_station=obs.ignore_station)

    solved = solved.run()

    # Check the radiant
    assert 3600*np.degrees(angleBetweenVectors(solved.radiant_eci_mini, radiant_eci)) < 0.05




if __name__ == "__main__":

    # Print the radiant error, the worst residual and the initial velocity error of each case
    for duration, v_init, elevation, estimate_timing_vel in ((2.5, 30000.0, 55.0, True), \
        (5.0, 15000.0, 30.0, True), (5.0, 15000.0, 30.0, False)):

        traj, radiant_error = _solve(duration, v_init, elevation, estimate_timing_vel=estimate_timing_vel)

        print("{:.1f} s, {:.0f} m/s, {:.0f} deg, timing {!s:5}: radiant error {:.3f} arcsec, " \
            "worst residual {:.3f} m, initial velocity error {:+.3f} m/s".format(duration, v_init, \
            elevation, estimate_timing_vel, radiant_error, worstResidual(traj), traj.v_init - v_init))
