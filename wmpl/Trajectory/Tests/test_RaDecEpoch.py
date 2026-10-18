""" Tests for the epoch at which R.A./Dec. measurements are converted to azimuth and elevation.

With meastype 1, ObservedPoints converts R.A./Dec. to azimuth/elevation and back to R.A./Dec. at each
point's own time. By default the first conversion is done at jdt_ref, which inverts data whose R.A./Dec.
were all computed at that one time, as CAMS does. Data computed at each point's own time, as RMS does,
needs radec_jd: without it each line of sight is rotated by the Earth's rotation since jdt_ref.

The tests generate a straight, constant-velocity trajectory in ECI, observe it from three stations without
noise, and check that the solver returns it.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_RaDecEpoch.py -v
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")

from wmpl.Trajectory.Trajectory import Trajectory
from wmpl.Utils.TrajConversions import altAz2RADec, eci2RaDec, geo2Cartesian, raDec2AltAz_vect, raDec2ECI


JDT_REF = 2461070.9343128
DURATION = 2.5
N_POINTS = 40
V_INIT = 30000.0

# (station_id, lat, lon, elevation) in degrees and metres
STATIONS = [
    ("A", -43.60, 172.12, 120.0),
    ("B", -42.90, 173.20, 200.0),
    ("C", -43.30, 171.30, 350.0),
]


def _trajectory():
    """ Straight constant-velocity trajectory, ECI coordinates of each point and its relative time. """

    lat, lon, ht = np.radians(-43.20), np.radians(172.30), 85000.0
    begin = np.array(geo2Cartesian(lat, lon, ht, JDT_REF))

    # Radiant at 210 deg azimuth, 55 deg elevation from the beginning point
    ra, dec = altAz2RADec(np.radians(210.0), np.radians(55.0), JDT_REF, lat, lon)
    direction = -np.array(raDec2ECI(ra, dec))

    times = np.linspace(0, DURATION, N_POINTS)

    return begin + np.outer(V_INIT*times, direction), times, -direction


def _radecOfDate(points, jds, lat, lon, ele):
    """ R.A./Dec. of date of each point, each computed at its own time. """

    ra, dec = [], []
    for point, jd in zip(points, jds):
        station = np.array(geo2Cartesian(np.radians(lat), np.radians(lon), ele, jd))
        r, d = eci2RaDec(point - station)
        ra.append(r)
        dec.append(d)

    return np.array(ra), np.array(dec)


def _solve(per_point, clock_offsets=None, estimate_timing_vel=False):
    """ Solve the trajectory from noise-free R.A./Dec. computed at each point's own time.

    Arguments:
        per_point: [bool] Give the solver radec_jd.

    Keyword arguments:
        clock_offsets: [dict] Station ID -> seconds the station's clock is ahead. Its R.A./Dec. are
            computed at the time its clock shows, as an RMS station's would be.
        estimate_timing_vel: [bool] Passed to Trajectory.

    Return:
        (traj, radiant_eci): solved trajectory and the true radiant unit vector.
    """

    points, times, radiant_eci = _trajectory()
    clock_offsets = clock_offsets or {}

    traj = Trajectory(JDT_REF, meastype=1, max_toffset=1.0, estimate_timing_vel=estimate_timing_vel,
        gravity_correction=False, monte_carlo=False, calc_orbit=False, show_plots=False,
        save_results=False, verbose=False)

    for station_id, lat, lon, ele in STATIONS:

        recorded = times + clock_offsets.get(station_id, 0.0)

        ra, dec = _radecOfDate(points, JDT_REF + times/86400.0, lat, lon, ele)

        # The camera measures azimuth and elevation; its clock decides the time they are converted at
        if station_id in clock_offsets:
            azim, elev = raDec2AltAz_vect(ra, dec, JDT_REF + times/86400.0, np.radians(lat), np.radians(lon))
            ra, dec = np.array([altAz2RADec(a, e, JDT_REF + t/86400.0, np.radians(lat), np.radians(lon))
                for a, e, t in zip(azim, elev, recorded)]).T

        radec_jd = JDT_REF + recorded/86400.0 if per_point else None

        traj.infillTrajectory(ra, dec, recorded, np.radians(lat), np.radians(lon), ele, station_id=station_id,
            radec_jd=radec_jd)

    return traj.run(), radiant_eci


def _worstResidual(traj):
    """ Largest spatial residual over all stations, in metres. """

    return max(np.max(np.hypot(obs.h_residuals, obs.v_residuals)) for obs in traj.observations)


def _radiantError(traj, radiant_eci):
    """ Angle between the solved and the true radiant, in arcseconds. """

    return 3600*np.degrees(np.arccos(np.clip(np.dot(traj.radiant_eci_mini, radiant_eci), -1, 1)))


def test_per_point_radec_inverts_exactly():
    """ Without radec_jd the same data leaves 22 m of residual and 53 arcsec of radiant error. """

    traj, radiant_eci = _solve(per_point=True)

    assert _worstResidual(traj) < 0.01
    assert _radiantError(traj, radiant_eci) < 0.01


def test_the_default_still_inverts_data_computed_at_the_reference_time():
    """ CAMS-style data: azimuth/elevation converted to R.A./Dec. all at jdt_ref. """

    points, times, radiant_eci = _trajectory()

    traj = Trajectory(JDT_REF, meastype=1, estimate_timing_vel=False, gravity_correction=False,
        monte_carlo=False, calc_orbit=False, show_plots=False, save_results=False, verbose=False)

    for station_id, lat, lon, ele in STATIONS:
        jds = JDT_REF + times/86400.0
        ra, dec = _radecOfDate(points, jds, lat, lon, ele)
        azim, elev = raDec2AltAz_vect(ra, dec, jds, np.radians(lat), np.radians(lon))
        ra, dec = np.array([altAz2RADec(a, e, JDT_REF, np.radians(lat), np.radians(lon))
            for a, e in zip(azim, elev)]).T
        traj.infillTrajectory(ra, dec, times, np.radians(lat), np.radians(lon), ele, station_id=station_id)

    traj = traj.run()

    assert _worstResidual(traj) < 0.01
    assert _radiantError(traj, radiant_eci) < 0.01


def _solveAzimuthal(clock_offsets):
    """ The same observations given as the azimuth/elevation the cameras measured (meastype 2). """

    points, times, radiant_eci = _trajectory()

    traj = Trajectory(JDT_REF, meastype=2, max_toffset=1.0, estimate_timing_vel=True,
        gravity_correction=False, monte_carlo=False, calc_orbit=False, show_plots=False,
        save_results=False, verbose=False)

    for station_id, lat, lon, ele in STATIONS:
        jds = JDT_REF + times/86400.0
        ra, dec = _radecOfDate(points, jds, lat, lon, ele)
        azim, elev = raDec2AltAz_vect(ra, dec, jds, np.radians(lat), np.radians(lon))
        traj.infillTrajectory(azim, elev, times + clock_offsets.get(station_id, 0.0), np.radians(lat),
            np.radians(lon), ele, station_id=station_id)

    return traj.run(), radiant_eci


def test_a_corrected_clock_moves_the_line_of_sight_with_the_earth():
    """ The timing fit shifts time_data; the azimuth/elevation the camera measured must not move with it.

    So per-point R.A./Dec. must solve exactly as the azimuth/elevation they came from. Converting at the
    shifted times instead of radec_jd would hold station B's lines of sight still in the sky while the
    station turns with the Earth, by 0.3 s.
    """

    clock_offsets = {"B": 0.3}
    traj, radiant_eci = _solve(per_point=True, clock_offsets=clock_offsets, estimate_timing_vel=True)
    reference, _ = _solveAzimuthal(clock_offsets)

    def relative_offset(solved):
        offsets = dict(zip([obs.station_id for obs in solved.observations], solved.time_diffs_final))

        # Relative to A, as the solver may take any station as the reference
        return offsets["B"] - offsets["A"]

    # The timing fit's own precision: it recovers the 0.3 s to 44 microseconds either way
    assert abs(relative_offset(traj) + 0.3) < 1e-4

    assert abs(relative_offset(traj) - relative_offset(reference)) < 1e-6
    assert 3600*np.degrees(np.arccos(np.clip(np.dot(traj.radiant_eci_mini, reference.radiant_eci_mini),
        -1, 1))) < 0.01


def test_observations_pickled_before_radec_jd_still_infill():
    """ An ObservedPoints from an older pickle has no radec_jd, and was computed at jdt_ref. """

    points, times, _ = _trajectory()
    station_id, lat, lon, ele = STATIONS[0]
    ra, dec = _radecOfDate(points, JDT_REF + times/86400.0, lat, lon, ele)

    old = Trajectory(JDT_REF, meastype=1, verbose=False)
    old.infillTrajectory(ra, dec, times, np.radians(lat), np.radians(lon), ele, station_id=station_id)
    obs = old.observations[0]
    del obs.radec_jd

    new = Trajectory(JDT_REF, meastype=1, verbose=False)
    new.infillWithObs(obs)

    assert new.observations[0].radec_jd is None
    assert np.allclose(new.observations[0].azim_data, obs.azim_data, rtol=0, atol=1e-12)


if __name__ == "__main__":

    for per_point in (False, True):
        traj, radiant_eci = _solve(per_point)
        print("radec_jd given: {!s:5}  worst residual {:8.3f} m  radiant error {:7.2f} arcsec".format(
            per_point, _worstResidual(traj), _radiantError(traj, radiant_eci)))
