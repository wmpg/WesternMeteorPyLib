""" Tests that the gravity correction is applied when the timing offsets are not estimated.

The line-of-sight fit applies the gravity drop only in its second pass, which has the vertical velocity
from the first. That pass used to run only after a timing estimate, so with estimate_timing_vel=False the
fitted line was exactly the one gravity_correction=False gives.

The tests drop a noise-free constant-velocity trajectory with applyGravityDrop, observe it from three
stations as azimuth/elevation at each point's own time, and check that the solver returns it.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_GravityWithoutTiming.py -v
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")

from wmpl.Trajectory.Trajectory import Trajectory, applyGravityDrop
from wmpl.Utils.TrajConversions import altAz2RADec, eci2RaDec, geo2Cartesian, raDec2AltAz_vect, raDec2ECI


JDT_REF = 2461070.9343128
N_POINTS = 40

# (station_id, lat, lon, elevation) in degrees and metres
STATIONS = [
    ("A", -43.60, 172.12, 120.0),
    ("B", -42.90, 173.20, 200.0),
    ("C", -43.30, 171.30, 350.0),
]


def _trajectory(duration, v_init, elevation):
    """ Gravity-dropped constant-velocity trajectory: ECI points, relative times, radiant unit vector. """

    lat, lon, ht = np.radians(-43.20), np.radians(172.30), 85000.0
    begin = np.array(geo2Cartesian(lat, lon, ht, JDT_REF))

    ra, dec = altAz2RADec(np.radians(210.0), np.radians(elevation), JDT_REF, lat, lon)
    radiant_eci = np.array(raDec2ECI(ra, dec))

    times = np.linspace(0, duration, N_POINTS)
    straight = begin - np.outer(v_init*times, radiant_eci)

    v0z = -v_init*np.sin(np.radians(elevation))
    points = np.array([applyGravityDrop(point, t, np.linalg.norm(begin), 1.0, v0z)
        for point, t in zip(straight, times)])

    return points, times, radiant_eci


def _solve(points, times, estimate_timing_vel, gravity_correction=True):
    """ Solve from noise-free azimuth/elevation, each at its point's own time (meastype 2). """

    traj = Trajectory(JDT_REF, meastype=2, max_toffset=1.0, estimate_timing_vel=estimate_timing_vel,
        gravity_correction=gravity_correction, monte_carlo=False, calc_orbit=False, show_plots=False,
        save_results=False, verbose=False)

    for station_id, lat, lon, ele in STATIONS:

        jds = JDT_REF + times/86400.0

        ra, dec = [], []
        for point, jd in zip(points, jds):
            station = np.array(geo2Cartesian(np.radians(lat), np.radians(lon), ele, jd))
            r, d = eci2RaDec(point - station)
            ra.append(r)
            dec.append(d)

        azim, elev = raDec2AltAz_vect(np.array(ra), np.array(dec), jds, np.radians(lat), np.radians(lon))

        traj.infillTrajectory(azim, elev, times, np.radians(lat), np.radians(lon), ele, station_id=station_id)

    return traj.run()


def _radiantError(traj, radiant_eci):
    """ Angle between the solved and the true radiant, in arcseconds. """

    return 3600*np.degrees(np.arccos(np.clip(np.dot(traj.radiant_eci_mini, radiant_eci), -1, 1)))


def _worstResidual(traj):
    """ Largest spatial residual over all stations, in metres. """

    return max(np.max(np.hypot(obs.h_residuals, obs.v_residuals)) for obs in traj.observations)


def test_gravity_is_applied_without_timing_estimation():
    """ A slow, shallow 5 s meteor. Before: 67.4 m of residual and 295.8 arcsec of radiant error. """

    points, times, radiant_eci = _trajectory(5.0, 15000.0, 30.0)
    traj = _solve(points, times, estimate_timing_vel=False)

    assert _worstResidual(traj) < 1.0
    assert _radiantError(traj, radiant_eci) < 1.5


def test_without_timing_the_line_matches_the_one_with_timing():
    """ With exact clocks the timing estimate has nothing to correct, so only the gravity pass differs. """

    points, times, radiant_eci = _trajectory(2.5, 30000.0, 55.0)

    without = _solve(points, times, estimate_timing_vel=False)
    with_timing = _solve(points, times, estimate_timing_vel=True)

    angle = 3600*np.degrees(np.arccos(np.clip(np.dot(without.radiant_eci_mini, with_timing.radiant_eci_mini),
        -1, 1)))

    assert angle < 0.1
    assert _radiantError(without, radiant_eci) < 0.5


def test_the_gravity_correction_changes_the_line_without_timing():
    """ The symptom itself: gravity on and off used to give the same fitted line. """

    points, times, _ = _trajectory(5.0, 15000.0, 30.0)

    on = _solve(points, times, estimate_timing_vel=False, gravity_correction=True)
    off = _solve(points, times, estimate_timing_vel=False, gravity_correction=False)

    assert not np.allclose(on.radiant_eci_mini, off.radiant_eci_mini, rtol=0, atol=1e-9)


if __name__ == "__main__":

    points, times, radiant_eci = _trajectory(5.0, 15000.0, 30.0)
    for timing in (False, True):
        traj = _solve(points, times, timing)
        print("estimate_timing_vel={!s:5}  worst residual {:8.3f} m  radiant error {:7.2f} arcsec".format(
            timing, _worstResidual(traj), _radiantError(traj, radiant_eci)))
