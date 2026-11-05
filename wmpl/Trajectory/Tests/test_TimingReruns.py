""" Tests the extra passes after the timing rerun (Trajectory(timing_reruns=...)).

The pass that follows the timing estimate recomputes the solution with the estimated clock offsets and
estimates them again, but the station positions and lines of sight it fits were computed at the times it
began with, and the stations move with the Earth. The first pass fits a straight line, so with gravity its
clock estimate is off; here, for a slow 4 s meteor that one station sees nearly end-on, by 0.4 ms, which
moves the other three stations by 0.14 m.

The tests drop a noise-free trajectory with applyGravityDrop, as the solver models it, observe it from four
stations with their clocks set exactly, as azimuth/elevation at each point's own time, and solve with the
timing estimated.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_TimingReruns.py -v
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")

from wmpl.Trajectory.Trajectory import Trajectory, applyGravityDrop
from wmpl.Utils.TrajConversions import altAz2RADec, eci2RaDec, geo2Cartesian, raDec2AltAz_vect, raDec2ECI


JDT_REF = 2461070.9343128
N_POINTS = 48

# (station_id, lat, lon, elevation) in degrees and metres. A sees the meteor 7 deg off its line of flight
STATIONS = [
    ("A", -42.565, 172.781, 310.0),
    ("B", -42.995, 173.312, 95.0),
    ("C", -43.804, 172.453, 640.0),
    ("D", -42.995, 169.873, 1450.0),
]


def _solve(timing_reruns):
    """ Solve a slow, gravity-dropped trajectory. Returns the solution and how many timing reruns ran. """

    lat, lon, ht = np.radians(-43.20), np.radians(172.30), 92000.0
    duration, v_init, elevation = 4.0, 17500.0, 42.0
    begin = np.array(geo2Cartesian(lat, lon, ht, JDT_REF))
    ra, dec = altAz2RADec(np.radians(210.0), np.radians(elevation), JDT_REF, lat, lon)
    radiant_eci = np.array(raDec2ECI(ra, dec))

    times = np.linspace(0, duration, N_POINTS)
    straight = begin - np.outer(v_init*times, radiant_eci)
    v0z = -v_init*np.sin(np.radians(elevation))
    points = np.array([applyGravityDrop(point, t, np.linalg.norm(begin), 1.0, v0z)
        for point, t in zip(straight, times)])

    traj = Trajectory(JDT_REF, meastype=2, max_toffset=1.0, estimate_timing_vel=True, gravity_correction=True,
        monte_carlo=False, calc_orbit=False, show_plots=False, save_results=False, verbose=False,
        timing_reruns=timing_reruns)

    for station_id, s_lat, s_lon, ele in STATIONS:

        jds = JDT_REF + times/86400.0

        ra_list, dec_list = [], []
        for point, jd in zip(points, jds):
            station = np.array(geo2Cartesian(np.radians(s_lat), np.radians(s_lon), ele, jd))
            r, d = eci2RaDec(point - station)
            ra_list.append(r)
            dec_list.append(d)

        azim, elev = raDec2AltAz_vect(np.array(ra_list), np.array(dec_list), jds, np.radians(s_lat),
            np.radians(s_lon))

        traj.infillTrajectory(azim, elev, times, np.radians(s_lat), np.radians(s_lon), ele,
            station_id=station_id)

    # Count the timing reruns
    reruns = []
    run = traj.run

    def counted(*args, **kwargs):
        reruns.append(kwargs.get("_rerun_timing", False))
        return run(*args, **kwargs)

    traj.run = counted
    traj = traj.run()

    return traj, radiant_eci, sum(reruns)


def _radiantError(traj, radiant_eci):
    """ Angle between the solved and the true radiant, in arcseconds. """

    return 3600*np.degrees(np.arccos(np.clip(np.dot(traj.radiant_eci_mini, radiant_eci), -1, 1)))


def _worstResidual(traj):
    """ Largest spatial residual over all stations, in metres. """

    return max(np.max(np.hypot(obs.h_residuals, obs.v_residuals)) for obs in traj.observations)


def test_the_default_keeps_the_single_pass():
    """ timing_reruns=0 is the solver as it was: one pass after the timing estimate. """

    traj, _, reruns = _solve(0)

    assert reruns == 1
    assert traj.timing_reruns == 0


def test_an_extra_pass_brings_the_geometry_to_the_final_clocks():
    """ Measured 0.127 m of residual and 0.027 arcsec with one pass, 0.013 m and 0.004 arcsec with an extra one.

    What is left is the timing estimate's own: the clocks come out 0.02 ms apart, set exactly.
    """

    single, radiant_eci, _ = _solve(0)
    traj, radiant_eci, reruns = _solve(3)

    assert _worstResidual(traj) < 0.02
    assert _worstResidual(traj) < _worstResidual(single)/5
    assert _radiantError(traj, radiant_eci) < 0.01

    # The extra pass leaves the clocks where it found them, so no further one is run
    assert reruns == 2


if __name__ == "__main__":

    for timing_reruns in (0, 3):
        traj, radiant_eci, reruns = _solve(timing_reruns)
        print("timing_reruns={}: {} reruns, radiant error {:.4f} arcsec, worst residual {:.4f} m".format(
            timing_reruns, reruns, _radiantError(traj, radiant_eci), _worstResidual(traj)))
