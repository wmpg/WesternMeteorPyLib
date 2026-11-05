""" Tests for the two ways the line-of-sight fit lost a good solution: a false alarm of the divergence check, and running
    out of function evaluations.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_LoSFixes.py -v
"""

import sys
from types import SimpleNamespace

import numpy as np
import scipy.optimize

import matplotlib
matplotlib.use("Agg")

from wmpl.Trajectory.Trajectory import Trajectory, stateVectorDiverged
from wmpl.Utils.TrajConversions import altAz2RADec, eci2RaDec, geo2Cartesian, raDec2AltAz_vect, raDec2ECI


JDT_REF = 2461070.9343128

# (station_id, lat, lon, elevation) in degrees and metres
STATIONS = [
    ("A", -43.60, 172.12, 120.0),
    ("B", -42.90, 173.20, 200.0),
    ("C", -43.30, 171.30, 350.0),
]


def _solve(minimize):
    """ Solve a noisy straight trajectory with the given replacement of scipy.optimize.minimize. """

    rng = np.random.default_rng(20261105)

    lat, lon = np.radians(-43.20), np.radians(172.30)
    begin = np.array(geo2Cartesian(lat, lon, 85000.0, JDT_REF))
    ra, dec = altAz2RADec(np.radians(210.0), np.radians(55.0), JDT_REF, lat, lon)
    times = np.linspace(0, 2.5, 40)
    points = begin - np.outer(30000.0*times, np.array(raDec2ECI(ra, dec)))

    original = scipy.optimize.minimize
    scipy.optimize.minimize = minimize

    try:

        traj = Trajectory(JDT_REF, meastype=2, estimate_timing_vel=False, gravity_correction=False,
            monte_carlo=False, calc_orbit=False, show_plots=False, save_results=False, verbose=False)

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

        return traj.run()

    finally:
        scipy.optimize.minimize = original


def _unit(v):

    return np.array(v)/np.linalg.norm(v)


def test_a_small_move_is_not_a_divergence_when_a_coordinate_is_near_zero():

    old = np.array([3.2e6, 150.0, 5.5e6])
    new = old + np.array([100.0, 200.0, -100.0])

    # What the check by coordinate says for the same move
    assert np.max(np.abs(new - old)/old) > 0.1

    assert not stateVectorDiverged(new, old)
    assert stateVectorDiverged(old*1.5, old)


def test_a_large_move_is_a_divergence_when_a_coordinate_is_negative():

    old = np.array([-3.2e6, -2.0e6, -5.5e6])
    new = old*1.5

    # What the check by coordinate says for the same move
    assert np.max(np.abs(new - old)/old) < 0.1

    assert stateVectorDiverged(new, old)


def test_a_line_which_ran_out_of_evaluations_is_continued_and_not_thrown_away():

    original = scipy.optimize.minimize
    results = []

    def limited(fun, x0, args=(), **options):

        # Only the first call of the fit is cut short
        if (getattr(fun, "__name__", "") == "minimizeAngleCost") and (options.get("method") == "Nelder-Mead"):

            if not results:
                options = dict(options, options={'maxfev': 20})

            results.append(original(fun, x0, args=args, **options))

            return results[-1]

        return original(fun, x0, args=args, **options)

    traj = _solve(limited)

    assert not results[0].success
    assert len(results) >= 2
    assert np.degrees(np.arccos(np.clip(np.dot(_unit(traj.radiant_eci_mini), _unit(results[-1].x[3:])), -1, 1))) < 1e-6


def test_a_line_which_wandered_off_along_itself_is_continued_from_the_meteor():
    """ The cost does not change along the line, so the simplex can end billions of metres away, and it does not
        converge from there. """

    original = scipy.optimize.minimize
    starts = []

    def wandering(fun, x0, args=(), **options):

        result = original(fun, x0, args=args, **options)

        if (getattr(fun, "__name__", "") == "minimizeAngleCost") and (options.get("method") == "Nelder-Mead"):

            starts.append(np.array(x0))

            if len(starts) == 1:
                x = np.r_[result.x[:3] + 3e10*_unit(result.x[3:]), result.x[3:]]
                return SimpleNamespace(x=x, fun=fun(x, *args), success=False, nit=1, message="wandered off")

        return result

    _solve(wandering)

    assert len(starts) >= 2

    # It starts from the first point of the meteor again, with a unit radiant
    assert abs(np.linalg.norm(starts[1][:3]) - 6.46e6) < 1e5
    assert abs(np.linalg.norm(starts[1][3:]) - 1) < 1e-9


def test_a_bounded_solution_is_only_taken_if_it_is_better(monkeypatch):

    original = scipy.optimize.minimize
    fitted = []
    bounded = []

    def worse_bounded(fun, x0, args=(), **options):

        result = original(fun, x0, args=args, **options)

        if options.get("method") == "SLSQP":

            bounded.append(True)

            # Far worse than the line, as seen in practice
            x = np.r_[x0[:3] + 20000.0, x0[3:] + 0.05]
            return SimpleNamespace(x=x, fun=fun(x, *args), success=True, nit=1, message="worse")

        if (getattr(fun, "__name__", "") == "minimizeAngleCost") and (options.get("method") == "Nelder-Mead"):
            fitted.append(result)

        return result

    # A false alarm of the check for divergence
    monkeypatch.setattr(sys.modules["wmpl.Trajectory.Trajectory"], "stateVectorDiverged", lambda *args, **kwargs: True)

    traj = _solve(worse_bounded)

    assert bounded
    assert np.degrees(np.arccos(np.clip(np.dot(_unit(traj.radiant_eci_mini), _unit(fitted[-1].x[3:])), -1, 1))) < 1e-6
