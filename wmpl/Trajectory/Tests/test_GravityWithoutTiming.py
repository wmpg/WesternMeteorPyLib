""" Tests that the gravity correction is applied when the timing offsets are not estimated.

The line-of-sight fit applies the gravity drop only in its second pass, which has the vertical velocity
from the first. That pass used to run only after a timing estimate, so with estimate_timing_vel=False the
fitted line was exactly the one gravity_correction=False gives.

The tests drop a noise-free constant-velocity meteor by gravity, observe it from three stations, and check
that the solver recovers it. A time range excluded from a station must still be excluded in the gravity pass,
and lines of sight given as RA and Dec must not change in it.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_GravityWithoutTiming.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np

import matplotlib
matplotlib.use('Agg')

from wmpl.Trajectory.Trajectory import Trajectory
from wmpl.Trajectory.Tests.SyntheticTrajectory import observeStraightMeteor
from wmpl.Utils.Math import angleBetweenVectors
from wmpl.Utils.TrajConversions import altAz2RADec_vect


def _solve(duration, v_init, elevation, estimate_timing_vel, gravity_correction=True):
    """ Solve a noise-free meteor dropped by gravity.

    Arguments:
        duration: [float] Duration of the meteor (s).
        v_init: [float] Velocity of the meteor (m/s).
        elevation: [float] Elevation of the radiant (deg).
        estimate_timing_vel: [bool] Estimate the timing offsets between the stations.

    Keyword arguments:
        gravity_correction: [bool] Apply the gravity correction. True by default.

    Return:
        (traj, radiant_eci):
            - traj: [Trajectory object] Solved trajectory.
            - radiant_eci: [ndarray] True radiant unit vector in ECI coordinates.
    """

    traj, _, radiant_eci = observeStraightMeteor(noise=0.0, duration=duration, v_init=v_init, \
        elevation=elevation, gravity_drop=True, estimate_timing_vel=estimate_timing_vel, \
        gravity_correction=gravity_correction, calc_orbit=False)

    return traj.run(), radiant_eci



def _worstResidual(traj):
    """ Compute the largest spatial residual over all stations.

    Arguments:
        traj: [Trajectory object] Solved trajectory.

    Return:
        [float] Largest spatial residual (m).
    """

    return max(np.max(np.hypot(obs.h_residuals, obs.v_residuals)) for obs in traj.observations)



def testGravityIsAppliedWithoutTimingEstimation():
    """ Check that a slow, shallow 5 s meteor is recovered without the timing estimation. Without the
        gravity pass, the worst residual was 67.4 m and the radiant was 295.8 arcsec off.
    """

    traj, radiant_eci = _solve(5.0, 15000.0, 30.0, estimate_timing_vel=False)

    assert _worstResidual(traj) < 1.0
    assert 3600*np.degrees(angleBetweenVectors(traj.radiant_eci_mini, radiant_eci)) < 1.5



def testWithoutTimingTheLineMatchesTheOneWithTiming():
    """ Check that the line without the timing estimation is the one with it, as with exact clocks the
        timing estimate has nothing to correct.
    """

    without_timing, radiant_eci = _solve(2.5, 30000.0, 55.0, estimate_timing_vel=False)
    with_timing, _ = _solve(2.5, 30000.0, 55.0, estimate_timing_vel=True)

    assert 3600*np.degrees(angleBetweenVectors(without_timing.radiant_eci_mini, \
        with_timing.radiant_eci_mini)) < 0.1
    assert 3600*np.degrees(angleBetweenVectors(without_timing.radiant_eci_mini, radiant_eci)) < 0.5



def testTheGravityCorrectionChangesTheLineWithoutTiming():
    """ Check that without the timing estimation, the gravity correction on and off give different lines.
    """

    gravity_on, _ = _solve(5.0, 15000.0, 30.0, estimate_timing_vel=False, gravity_correction=True)
    gravity_off, _ = _solve(5.0, 15000.0, 30.0, estimate_timing_vel=False, gravity_correction=False)

    assert not np.allclose(gravity_on.radiant_eci_mini, gravity_off.radiant_eci_mini, rtol=0, atol=1e-9)



def testAnExcludedTimeRangeIsKeptInTheGravityPass():
    """ Check that a time range excluded from a station which does not start at 0 is still excluded after the
        observations are reinitialized for the gravity pass, as the times are shifted to start at 0.
    """

    traj, _, _ = observeStraightMeteor(noise=0.0, gravity_drop=True, estimate_timing_vel=False, \
        gravity_correction=True, calc_orbit=False)

    # Start the times at 10 s, and exclude 11-11.5 s from the first station
    observations = traj.observations
    traj.observations = []
    for obs in observations:

        obs.time_data = obs.time_data + 10.0

        if obs.station_id == 'A':
            obs.excluded_time = [11.0, 11.5]

        traj.infillWithObs(obs)

    traj = traj.run()

    station = [obs for obs in traj.observations if obs.station_id == 'A'][0]

    assert np.allclose(station.excluded_time, [1.0, 1.5])
    assert station.excluded_indx_range



def testRADecAreKeptInTheGravityPassWhenTheTimesDoNotStartAtZero():
    """ Check that the lines of sight given as RA and Dec do not change when the observations are
        reinitialized for the gravity pass, as the reference JD is moved to the first time.
    """

    straight, _, _ = observeStraightMeteor(noise=0.0, gravity_drop=True)

    traj = Trajectory(straight.jdt_ref, meastype=1, max_toffset=1.0, estimate_timing_vel=False, \
        gravity_correction=True, monte_carlo=False, calc_orbit=False, show_plots=False, save_results=False, \
        verbose=False)

    # Give RA and Dec at the reference JD, with the times starting at 10 s
    for obs in straight.observations:
        ra, dec = altAz2RADec_vect(obs.azim_data, obs.elev_data, straight.jdt_ref, obs.lat, obs.lon)
        traj.infillTrajectory(ra, dec, obs.time_data + 10.0, obs.lat, obs.lon, obs.ele, \
            station_id=obs.station_id)

    lines_of_sight = {obs.station_id: obs.meas_eci_los for obs in traj.observations}

    traj = traj.run()

    for obs in traj.observations:
        assert np.allclose(obs.meas_eci_los, lines_of_sight[obs.station_id], rtol=0, atol=1e-8)




if __name__ == "__main__":

    # Print the residuals and the radiant error of the slow, shallow meteor with and without the timing
    #   estimation
    for estimate_timing_vel in (False, True):

        traj, radiant_eci = _solve(5.0, 15000.0, 30.0, estimate_timing_vel)

        print("estimate_timing_vel={!s:5}  worst residual {:8.3f} m  radiant error {:7.2f} arcsec".format( \
            estimate_timing_vel, _worstResidual(traj), \
            3600*np.degrees(angleBetweenVectors(traj.radiant_eci_mini, radiant_eci))))
