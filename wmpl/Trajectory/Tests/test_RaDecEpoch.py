""" Tests for the time at which R.A./Dec. measurements are converted to azimuth and elevation.

With meastype 1, ObservedPoints converts the R.A./Dec. to azimuth/elevation, and back to R.A./Dec. at the time
of each point. By default the first conversion is done at jdt_ref, which inverts R.A./Dec. that were all
computed at that one time, as CAMS does. R.A./Dec. computed at the time of each point, as RMS does, need
radec_jd: without it, each line of sight is rotated by the Earth's rotation since jdt_ref.

The tests observe the straight meteor of SyntheticTrajectory from three stations without noise, and check that
the solver recovers it.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_RaDecEpoch.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np

import matplotlib
matplotlib.use('Agg')

from wmpl.Trajectory.Trajectory import Trajectory
from wmpl.Trajectory.Tests.SyntheticTrajectory import STRAIGHT_JDT_REF, STRAIGHT_STATIONS, \
    observeStraightMeteor, stationAzimAlt, straightMeteorPoints
from wmpl.Utils.Math import angleBetweenVectors
from wmpl.Utils.TrajConversions import altAz2RADec_vect


# Seconds by which the clock of station B is ahead in the tests of the timing offsets. It is behind, so its
#   first time is before the reference time, and the solver moves the reference time to it
CLOCK_OFFSET = -0.3



def _solve(per_point=True, cams=False, clock_offsets=None, **kwargs):
    """ Solve the noise-free straight meteor from R.A./Dec. measurements.

    Keyword arguments:
        per_point: [bool] Give the solver the time at which each R.A./Dec. was computed (radec_jd). True by
            default.
        cams: [bool] Compute all R.A./Dec. at the reference time, as CAMS does, instead of at the time of each
            point, as RMS does. False by default.
        clock_offsets: [dict] Seconds by which the clock of the station is ahead, per station ID. The camera
            measures the azimuth and elevation, which are converted to R.A./Dec. at the time its clock shows.
            None by default, when all clocks are exact.
        **kwargs: [dict] Keyword arguments of the Trajectory object, which replace the defaults (no timing
            estimation, no gravity correction, no pick filtering, no Monte Carlo, no plots and no saving).

    Return:
        (traj, radiant_eci):
            - traj: [Trajectory object] Solved trajectory.
            - radiant_eci: [ndarray] True radiant unit vector in ECI coordinates.
    """

    if clock_offsets is None:
        clock_offsets = {}

    times, points, _, radiant_eci = straightMeteorPoints()

    # The pick filtering is off, as on noise-free data it rejects picks by the numerical noise
    traj_kwargs = dict(meastype=1, max_toffset=1.0, estimate_timing_vel=False, gravity_correction=False, \
        filter_picks=False, monte_carlo=False, calc_orbit=False, show_plots=False, save_results=False, \
        verbose=False)
    traj_kwargs.update(kwargs)
    traj = Trajectory(STRAIGHT_JDT_REF, **traj_kwargs)

    for station_id, s_lat, s_lon, ele in STRAIGHT_STATIONS:

        # Compute the direction of every point as seen from the station
        azim, elev = stationAzimAlt(points, STRAIGHT_JDT_REF + times/86400.0, s_lat, s_lon, ele)

        # Convert the directions to R.A./Dec. at the time the clock of the station shows
        recorded = times + clock_offsets.get(station_id, 0.0)
        radec_jd = STRAIGHT_JDT_REF + recorded/86400.0

        if cams:
            radec_jd = np.zeros_like(recorded) + STRAIGHT_JDT_REF

        ra, dec = altAz2RADec_vect(azim, elev, radec_jd, np.radians(s_lat), np.radians(s_lon))

        if not per_point:
            radec_jd = None

        traj.infillTrajectory(ra, dec, recorded, np.radians(s_lat), np.radians(s_lon), ele, \
            station_id=station_id, radec_jd=radec_jd)

    return traj.run(), radiant_eci



def _worstResidual(traj):
    """ Compute the largest spatial residual over all stations.

    Arguments:
        traj: [Trajectory object] Solved trajectory.

    Return:
        [float] Largest spatial residual (m).
    """

    return max(np.max(np.hypot(obs.h_residuals, obs.v_residuals)) for obs in traj.observations)



def _radiantError(radiant_eci, radiant_ref):
    """ Compute the angle between two radiants.

    Arguments:
        radiant_eci: [ndarray] Radiant unit vector in ECI coordinates.
        radiant_ref: [ndarray] Reference radiant unit vector in ECI coordinates.

    Return:
        [float] Angle between the radiants (arcsec).
    """

    return 3600*np.degrees(angleBetweenVectors(radiant_eci, radiant_ref))



def _offsetOfB(traj):
    """ Compute the estimated timing offset of station B relative to station A, as the solver may take any
        station as the reference.

    Arguments:
        traj: [Trajectory object] Solved trajectory.

    Return:
        [float] Timing offset of B relative to A (s).
    """

    offsets = dict(zip([obs.station_id for obs in traj.observations], traj.time_diffs_final))

    return offsets['B'] - offsets['A']



def testPerPointRaDecInverts():
    """ Check that R.A./Dec. computed at the time of each point are inverted when radec_jd is given. Without
        it, the worst residual is 22.4 m and the radiant is 52.4 arcsec off.
    """

    traj, radiant_eci = _solve()

    assert _worstResidual(traj) < 0.01
    assert _radiantError(traj.radiant_eci_mini, radiant_eci) < 0.01



def testCamsRaDecStillInvert():
    """ Check that R.A./Dec. all computed at the reference time, as CAMS does, are still inverted when
        radec_jd is not given.
    """

    traj, radiant_eci = _solve(per_point=False, cams=True)

    assert _worstResidual(traj) < 0.01
    assert _radiantError(traj.radiant_eci_mini, radiant_eci) < 0.01



def testACorrectedClockKeepsTheLinesOfSightFixedToTheEarth():
    """ Check that per-point R.A./Dec. of a station whose clock is off solve as the azimuth and elevation they
        were computed from (meastype 2). The timing estimation shifts the times of the station, but the
        azimuth and elevation its camera measured must stay fixed to the Earth. Converting at the shifted
        times instead of radec_jd would hold its lines of sight still in the sky while the station turns. The
        R.A./Dec. must not be recomputed at the new reference time either, which put the radiant 53 arcsec
        off.
    """

    traj, _ = _solve(clock_offsets={'B': CLOCK_OFFSET}, estimate_timing_vel=True)

    reference, _, _ = observeStraightMeteor(noise=0.0, clock_offsets={'B': CLOCK_OFFSET}, \
        estimate_timing_vel=True, filter_picks=False, calc_orbit=False)
    reference = reference.run()

    # The timing estimation recovers the clock offset to 44 microseconds either way
    assert abs(_offsetOfB(traj) + CLOCK_OFFSET) < 1e-4

    assert abs(_offsetOfB(traj) - _offsetOfB(reference)) < 1e-6
    assert _radiantError(traj.radiant_eci_mini, reference.radiant_eci_mini) < 0.05



def testObservationsPickledBeforeRadecJdStillInfill():
    """ Check that observations from older pickles, which have no radec_jd and were all computed at jdt_ref,
        are infilled as before.
    """

    traj, _ = _solve(per_point=False, cams=True)

    obs = traj.observations[0]
    del obs.radec_jd

    new = Trajectory(STRAIGHT_JDT_REF, meastype=1, verbose=False)
    new.infillWithObs(obs)

    assert new.observations[0].radec_jd is None
    assert np.allclose(new.observations[0].azim_data, obs.azim_data, rtol=0, atol=1e-12)




if __name__ == "__main__":

    # Print the worst residual and the radiant error with and without radec_jd
    for per_point in (False, True):
        traj, radiant_eci = _solve(per_point=per_point)
        print("radec_jd given: {!s:5}  worst residual {:8.3f} m  radiant error {:7.2f} arcsec".format(\
            per_point, _worstResidual(traj), _radiantError(traj.radiant_eci_mini, radiant_eci)))
