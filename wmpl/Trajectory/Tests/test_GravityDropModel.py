""" Tests that the solver inverts a trajectory dropped with its own gravity drop model.

applyGravityDrop descends from the beginning of the meteor, at distance r0 from the centre of the Earth,
with the vertical velocity v0z. The line-of-sight fit and the spatial residuals passed the distance of each
point instead, and the fit always ran with v0z=None (no vertical velocity): calcLLA computes it only after
the pass that applies the drop.

The tests drop a noise-free constant-velocity trajectory with applyGravityDrop, as the solver models it,
observe it from three stations as azimuth/elevation at each point's own time, and solve with the timing
estimated.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_GravityDropModel.py -v
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


def _solve(duration, v_init, elevation):
    """ Solve a gravity-dropped trajectory. Returns the solution and the true radiant unit vector. """

    lat, lon, ht = np.radians(-43.20), np.radians(172.30), 85000.0
    begin = np.array(geo2Cartesian(lat, lon, ht, JDT_REF))
    ra, dec = altAz2RADec(np.radians(210.0), np.radians(elevation), JDT_REF, lat, lon)
    radiant_eci = np.array(raDec2ECI(ra, dec))

    times = np.linspace(0, duration, N_POINTS)
    straight = begin - np.outer(v_init*times, radiant_eci)
    v0z = -v_init*np.sin(np.radians(elevation))
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

        traj.infillTrajectory(azim, elev, times, np.radians(s_lat), np.radians(s_lon), ele,
            station_id=station_id)

    return traj.run(), radiant_eci


def _radiantError(traj, radiant_eci):
    """ Angle between the solved and the true radiant, in arcseconds. """

    return 3600*np.degrees(np.arccos(np.clip(np.dot(traj.radiant_eci_mini, radiant_eci), -1, 1)))


def _worstResidual(traj):
    """ Largest spatial residual over all stations, in metres. """

    return max(np.max(np.hypot(obs.h_residuals, obs.v_residuals)) for obs in traj.observations)


def test_a_steep_fast_meteor_inverts():
    """ 2.5 s at 30 km/s, 55 deg. Before: 0.27 arcsec and 0.265 m of residual. """

    traj, radiant_eci = _solve(2.5, 30000.0, 55.0)

    assert _radiantError(traj, radiant_eci) < 0.01
    assert _worstResidual(traj) < 0.01
    assert abs(traj.v_init - 30000.0) < 0.1


def test_a_shallow_slow_meteor_inverts():
    """ 5 s at 15 km/s, 30 deg, where the drop reaches about 120 m. Before: 0.97 arcsec and 0.70 m. """

    traj, radiant_eci = _solve(5.0, 15000.0, 30.0)

    assert _radiantError(traj, radiant_eci) < 0.1
    assert _worstResidual(traj) < 0.1
    assert abs(traj.v_init - 15000.0) < 0.1


if __name__ == "__main__":

    for case in ((2.5, 30000.0, 55.0), (5.0, 15000.0, 30.0)):
        traj, radiant_eci = _solve(*case)
        print("{}: radiant error {:.3f} arcsec, worst residual {:.3f} m, initial velocity error {:+.3f} m/s".format(
            case, _radiantError(traj, radiant_eci), _worstResidual(traj), traj.v_init - case[1]))
