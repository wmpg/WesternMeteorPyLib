""" Tests which points the trajectory's beginning and end are reported from.

Each station reports its first and last used point (ignore_list == 0), and the trajectory takes the station
whose point lies earliest along the track for its beginning and the one whose point lies furthest for its
end. The tests observe a noise-free trajectory from four stations: A sees all of it but its first and last
six points are ignored, and the other three see a shorter stretch with every point used. The beginning and
the end must come from the other three, which saw further in both directions than A's used points.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_BeginEndPoints.py -v
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")

from wmpl.Trajectory.Trajectory import Trajectory, applyGravityDrop
from wmpl.Utils.TrajConversions import altAz2RADec, eci2RaDec, geo2Cartesian, raDec2AltAz_vect, raDec2ECI


JDT_REF = 2461070.9343128
N_POINTS = 48

# Points ignored at each end of A's track, and the first and last point the other stations see
A_IGNORED = 6
OTHERS_FIRST, OTHERS_LAST = 3, 44

# (station_id, lat, lon, elevation) in degrees and metres
STATIONS = [
    ("A", -42.565, 172.781, 310.0),
    ("B", -42.995, 173.312, 95.0),
    ("C", -43.804, 172.453, 640.0),
    ("D", -42.995, 169.873, 1450.0),
]


def _solve():
    """ Solve a gravity-dropped trajectory. Returns the solution and the times of its points. """

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

    # The picks are kept as given: none is rejected on its residual
    traj = Trajectory(JDT_REF, meastype=2, max_toffset=1.0, estimate_timing_vel=False, gravity_correction=True,
        filter_picks=False, monte_carlo=False, calc_orbit=False, show_plots=False, save_results=False,
        verbose=False)

    for station_id, s_lat, s_lon, ele in STATIONS:

        if station_id == "A":
            seen = np.arange(N_POINTS)
            ignore_list = np.zeros(N_POINTS, dtype=np.uint8)
            ignore_list[:A_IGNORED] = 1
            ignore_list[-A_IGNORED:] = 1
        else:
            seen = np.arange(OTHERS_FIRST, OTHERS_LAST + 1)
            ignore_list = None

        jds = JDT_REF + times[seen]/86400.0

        ra_list, dec_list = [], []
        for point, jd in zip(points[seen], jds):
            station = np.array(geo2Cartesian(np.radians(s_lat), np.radians(s_lon), ele, jd))
            r, d = eci2RaDec(point - station)
            ra_list.append(r)
            dec_list.append(d)

        azim, elev = raDec2AltAz_vect(np.array(ra_list), np.array(dec_list), jds, np.radians(s_lat),
            np.radians(s_lon))

        traj.infillTrajectory(azim, elev, times[seen], np.radians(s_lat), np.radians(s_lon), ele,
            station_id=station_id, ignore_list=ignore_list)

    return traj.run(), times


def test_the_ends_come_from_the_points_used():
    """ A's ignored points lie furthest along the track at both ends, and must not choose the station.

    Choosing by all of A's points reported its first and last used ones instead: the beginning 0.26 s late
    and 3.0 km low, the end 0.26 s early and 2.9 km high.
    """

    traj, times = _solve()
    others = [obs for obs in traj.observations if obs.station_id != "A"]

    assert np.isclose((traj.rbeg_jd - JDT_REF)*86400, times[OTHERS_FIRST], atol=1e-6)
    assert np.isclose((traj.rend_jd - JDT_REF)*86400, times[OTHERS_LAST], atol=1e-6)

    # The position is reported from the same point as the time
    assert traj.rbeg_ele in [obs.rbeg_ele for obs in others]
    assert traj.rend_ele in [obs.rend_ele for obs in others]


if __name__ == "__main__":

    traj, times = _solve()
    print("begin {:.3f} s at {:.1f} m, end {:.3f} s at {:.1f} m".format((traj.rbeg_jd - JDT_REF)*86400,
        traj.rbeg_ele, (traj.rend_jd - JDT_REF)*86400, traj.rend_ele))
    for obs in traj.observations:
        print("  {}: first used {:.3f} s at {:.1f} m, last used {:.3f} s at {:.1f} m".format(obs.station_id,
            (obs.rbeg_jd - JDT_REF)*86400, obs.rbeg_ele, (obs.rend_jd - JDT_REF)*86400, obs.rend_ele))
