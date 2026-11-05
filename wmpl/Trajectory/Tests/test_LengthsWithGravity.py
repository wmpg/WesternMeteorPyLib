""" Tests that the lengths along the trajectory follow the gravity drop the line was fitted with.

In the pass that applies the gravity drop, the line-of-sight fit returns the undropped line, which the
measurements sit below by up to g*t^2/2. calcVelocity projected them onto that line as if they were on
it, so each station's lengths were off by an amount that depends on its viewing geometry: the timing fit
read that as clock offsets, and the initial velocity was off too.

The tests drop a noise-free constant-velocity trajectory with applyGravityDrop and observe it from three
stations as azimuth/elevation at each point's own time.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_LengthsWithGravity.py -v
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")

from wmpl.Trajectory.Trajectory import Trajectory, applyGravityDrop
from wmpl.Utils.TrajConversions import altAz2RADec, eci2RaDec, geo2Cartesian, raDec2AltAz_vect, raDec2ECI


JDT_REF = 2461070.9343128
N_POINTS = 40

# A slow, shallow, long meteor, where the drop reaches about 120 m
DURATION = 5.0
V_INIT = 15000.0
ELEVATION = 30.0

# (station_id, lat, lon, elevation) in degrees and metres
STATIONS = [
    ("A", -43.60, 172.12, 120.0),
    ("B", -42.90, 173.20, 200.0),
    ("C", -43.30, 171.30, 350.0),
]


def _solve(clock_offsets=None):
    """ Solve the dropped trajectory, with the timing estimated.

    Keyword arguments:
        clock_offsets: [dict] Station ID -> seconds the station's clock is ahead.
    """

    clock_offsets = clock_offsets or {}

    lat, lon, ht = np.radians(-43.20), np.radians(172.30), 85000.0
    begin = np.array(geo2Cartesian(lat, lon, ht, JDT_REF))
    ra, dec = altAz2RADec(np.radians(210.0), np.radians(ELEVATION), JDT_REF, lat, lon)
    radiant_eci = np.array(raDec2ECI(ra, dec))

    times = np.linspace(0, DURATION, N_POINTS)
    straight = begin - np.outer(V_INIT*times, radiant_eci)
    v0z = -V_INIT*np.sin(np.radians(ELEVATION))
    points = np.array([applyGravityDrop(point, t, np.linalg.norm(begin), 1.0, v0z)
        for point, t in zip(straight, times)])

    traj = Trajectory(JDT_REF, meastype=2, max_toffset=1.0, estimate_timing_vel=True, gravity_correction=True,
        monte_carlo=False, calc_orbit=False, show_plots=False, save_results=False, verbose=False)

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

        traj.infillTrajectory(azim, elev, times + clock_offsets.get(station_id, 0.0), np.radians(s_lat),
            np.radians(s_lon), ele, station_id=station_id)

    return traj.run()


def _offsets(traj):
    """ Estimated timing offsets relative to station A, in seconds. """

    offsets = dict(zip([obs.station_id for obs in traj.observations], traj.time_diffs_final))

    return {station_id: offset - offsets["A"] for station_id, offset in offsets.items()}


def test_exact_clocks_get_no_offset():
    """ Before: -2.32 ms on B and -0.33 ms on C, with every clock exact. """

    offsets = _offsets(_solve())

    assert abs(offsets["B"]) < 5e-5
    assert abs(offsets["C"]) < 5e-5


def test_a_clock_offset_is_recovered():
    """ B's clock 0.3 s ahead. Before: recovered as 302.32 ms. """

    offsets = _offsets(_solve(clock_offsets={"B": 0.3}))

    assert abs(offsets["B"] + 0.3) < 5e-5


def test_the_initial_velocity_is_recovered():
    """ Before: 12.07 m/s fast. """

    traj = _solve()

    assert abs(traj.v_init - V_INIT) < 0.5


if __name__ == "__main__":

    traj = _solve()
    print("offsets {}  initial velocity error {:+.3f} m/s".format(_offsets(traj), traj.v_init - V_INIT))
