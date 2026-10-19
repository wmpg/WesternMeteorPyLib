""" Tests for the optional restarts of the line-of-sight Nelder-Mead minimization (los_restarts).

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_LoSRestarts.py -v
"""

import numpy as np
import scipy.optimize

import matplotlib
matplotlib.use("Agg")

from wmpl.Trajectory.Trajectory import Trajectory
from wmpl.Utils.TrajConversions import altAz2RADec, eci2RaDec, geo2Cartesian, raDec2AltAz_vect, raDec2ECI


JDT_REF = 2461070.9343128

# (station_id, lat, lon, elevation) in degrees and metres
STATIONS = [
    ("A", -43.60, 172.12, 120.0),
    ("B", -42.90, 173.20, 200.0),
    ("C", -43.30, 171.30, 350.0),
]


def _solve(monkeypatch, **kwargs):
    """ Solve a noisy straight trajectory, recording every Nelder-Mead call of the line-of-sight fit. """

    rng = np.random.default_rng(20261019)

    lat, lon = np.radians(-43.20), np.radians(172.30)
    begin = np.array(geo2Cartesian(lat, lon, 85000.0, JDT_REF))
    ra, dec = altAz2RADec(np.radians(210.0), np.radians(55.0), JDT_REF, lat, lon)
    times = np.linspace(0, 2.5, 40)
    points = begin - np.outer(30000.0*times, np.array(raDec2ECI(ra, dec)))

    calls = []
    minimize = scipy.optimize.minimize

    def recording(fun, x0, args=(), **options):
        result = minimize(fun, x0, args=args, **options)
        if getattr(fun, "__name__", "") == "minimizeAngleCost" and options.get("method") == "Nelder-Mead":
            calls.append((np.array(x0), fun(np.array(x0), *args), result))
        return result

    monkeypatch.setattr(scipy.optimize, "minimize", recording)

    traj = Trajectory(JDT_REF, meastype=2, estimate_timing_vel=False, gravity_correction=False,
        monte_carlo=False, calc_orbit=False, show_plots=False, save_results=False, verbose=False, **kwargs)

    for station_id, s_lat, s_lon, ele in STATIONS:
        jds = JDT_REF + times/86400.0
        ra_list, dec_list = [], []
        for point, jd in zip(points, jds):
            r, d = eci2RaDec(point - np.array(geo2Cartesian(np.radians(s_lat), np.radians(s_lon), ele, jd)))
            ra_list.append(r)
            dec_list.append(d)
        azim, elev = raDec2AltAz_vect(np.array(ra_list), np.array(dec_list), jds, np.radians(s_lat),
            np.radians(s_lon))
        traj.infillTrajectory(azim + rng.normal(0, 1e-4, len(times)), elev + rng.normal(0, 1e-4, len(times)),
            times, np.radians(s_lat), np.radians(s_lon), ele, station_id=station_id)

    return traj.run(), calls


def test_no_restart_by_default(monkeypatch):
    """ One line-of-sight fit, one Nelder-Mead call: the default behaviour is unchanged. """

    _, calls = _solve(monkeypatch)

    assert len(calls) == 1


def test_each_restart_starts_from_the_previous_solution(monkeypatch):
    """ And never ends with a higher cost than it started from. """

    traj, calls = _solve(monkeypatch, los_restarts=2)

    assert len(calls) == 3
    for (_, _, previous), (x0, cost0, result) in zip(calls[:-1], calls[1:]):
        assert np.array_equal(x0, previous.x)
        assert result.fun <= cost0
