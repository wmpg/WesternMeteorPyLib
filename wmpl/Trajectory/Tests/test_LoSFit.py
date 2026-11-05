""" Tests of the line-of-sight fit of the trajectory solver.

The fit lost a good solution in two ways: a false alarm of the divergence check, after which a far worse
bounded solution was taken, and running out of function evaluations, after which the solution was replaced
by the intersecting planes. The optional restarts of the Nelder-Mead minimization (los_restarts) are also
tested. The tests solve a noisy straight trajectory seen from three stations, with scipy.optimize.minimize
replaced to record or alter the minimizations of the fit.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_LoSFit.py -v
"""

from __future__ import print_function, division, absolute_import

import contextlib
import sys
from types import SimpleNamespace

import numpy as np
import scipy.optimize

import matplotlib
matplotlib.use('Agg')

from wmpl.Trajectory.Trajectory import Trajectory, stateVectorDiverged
from wmpl.Trajectory.Tests.SyntheticTrajectory import stationAzimAlt
from wmpl.Utils.TrajConversions import altAz2RADec, geo2Cartesian, raDec2ECI


JDT_REF = 2461070.9343128

# Beginning of the meteor (deg, deg, m)
BEGIN_LAT, BEGIN_LON, BEGIN_HT = -43.20, 172.30, 85000.0

# (station_id, lat, lon, elevation) in degrees and metres
STATIONS = [
    ('A', -43.60, 172.12, 120.0),
    ('B', -42.90, 173.20, 200.0),
    ('C', -43.30, 171.30, 350.0),
]



@contextlib.contextmanager
def _replaced(obj, name, value):
    """ Replace an attribute of an object while the context is open.

    Arguments:
        obj: [object] Object, e.g. a module.
        name: [str] Name of the attribute.
        value: [object] Value of the attribute in the context.
    """

    original = getattr(obj, name)
    setattr(obj, name, value)

    try:
        yield

    finally:
        setattr(obj, name, original)



def _isLoSFit(fun, options):
    """ Tell if a call of scipy.optimize.minimize is a Nelder-Mead minimization of the line-of-sight fit.

    Arguments:
        fun: [function] Minimized function.
        options: [dict] Keyword arguments of the call.

    Return:
        [bool] True if it is.
    """

    return (getattr(fun, '__name__', '') == 'minimizeAngleCost') and (options.get('method') == 'Nelder-Mead')



def _angle(a, b):
    """ Compute the angle between two vectors.

    Arguments:
        a: [ndarray] First vector.
        b: [ndarray] Second vector.

    Return:
        [float] Angle between the vectors (deg).
    """

    a, b = np.array(a)/np.linalg.norm(a), np.array(b)/np.linalg.norm(b)

    return np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1)))



def _solve(minimize, seed=20261105, **kwargs):
    """ Solve a noisy straight trajectory seen from three stations, with scipy.optimize.minimize replaced.

    Arguments:
        minimize: [function] Replacement of scipy.optimize.minimize.

    Keyword arguments:
        seed: [int] Seed of the noise. 20261105 by default.
        **kwargs: [dict] Keyword arguments of the Trajectory object.

    Return:
        (traj, begin):
            - traj: [Trajectory object] Solved trajectory.
            - begin: [ndarray] ECI coordinates of the beginning of the meteor (m).
    """

    rng = np.random.default_rng(seed)

    # Points along the straight line, at 30 km/s
    lat, lon = np.radians(BEGIN_LAT), np.radians(BEGIN_LON)
    begin = np.array(geo2Cartesian(lat, lon, BEGIN_HT, JDT_REF))
    ra, dec = altAz2RADec(np.radians(210.0), np.radians(55.0), JDT_REF, lat, lon)
    times = np.linspace(0, 2.5, 40)
    points = begin - np.outer(30000.0*times, np.array(raDec2ECI(ra, dec)))

    traj = Trajectory(JDT_REF, meastype=2, estimate_timing_vel=False, gravity_correction=False, \
        monte_carlo=False, calc_orbit=False, show_plots=False, save_results=False, verbose=False, **kwargs)

    # Directions of the points from each station, with 20 arcsec of noise
    for station_id, s_lat, s_lon, ele in STATIONS:

        azim, elev = stationAzimAlt(points, JDT_REF + times/86400.0, s_lat, s_lon, ele)

        traj.infillTrajectory(azim + rng.normal(0, 1e-4, len(times)), \
            elev + rng.normal(0, 1e-4, len(times)), times, np.radians(s_lat), np.radians(s_lon), ele, \
            station_id=station_id)

    with _replaced(scipy.optimize, 'minimize', minimize):
        traj = traj.run()

    return traj, begin



def testASmallMoveIsNotADivergenceWhenACoordinateIsNearZero():
    """ Check that a move of a few hundred metres is not a divergence, although it is more than 10% of a
        coordinate near 0.
    """

    old = np.array([3.2e6, 150.0, 5.5e6])
    new = old + np.array([100.0, 200.0, -100.0])

    # The check by coordinate says it diverged
    assert np.max(np.abs(new - old)/old) > 0.1

    assert not stateVectorDiverged(new, old)
    assert stateVectorDiverged(1.5*old, old)



def testALargeMoveIsADivergenceWhenACoordinateIsNegative():
    """ Check that a move by half of the state vector is a divergence when its coordinates are negative. """

    old = np.array([-3.2e6, -2.0e6, -5.5e6])
    new = 1.5*old

    # The check by coordinate says it did not diverge
    assert np.max(np.abs(new - old)/old) < 0.1

    assert stateVectorDiverged(new, old)



def testALineWhichRanOutOfEvaluationsIsContinued():
    """ Check that a fit which runs out of function evaluations is continued, and that its solution is
        taken instead of the intersecting planes.
    """

    original = scipy.optimize.minimize
    results = []

    def limited(fun, x0, args=(), **options):

        if not _isLoSFit(fun, options):
            return original(fun, x0, args=args, **options)

        # Only the first minimization of the fit is cut short
        if not results:
            options = dict(options, options={'maxfev': 20})

        results.append(original(fun, x0, args=args, **options))

        return results[-1]

    traj, _ = _solve(limited)

    assert not results[0].success
    assert len(results) >= 2
    assert _angle(traj.radiant_eci_mini, results[-1].x[3:]) < 1e-6



def testALineWhichWanderedOffAlongItselfIsContinuedFromTheMeteor():
    """ Check that a fit which ends far along the line, where the cost is the same, is continued from the
        beginning of the meteor with a unit radiant, as it does not converge from there.
    """

    original = scipy.optimize.minimize
    starts = []

    def wandering(fun, x0, args=(), **options):

        result = original(fun, x0, args=args, **options)

        if not _isLoSFit(fun, options):
            return result

        starts.append(np.array(x0))

        # The first minimization ends 30 million km along the line
        if len(starts) == 1:
            x = np.r_[result.x[:3] + 3e10*result.x[3:]/np.linalg.norm(result.x[3:]), result.x[3:]]
            return SimpleNamespace(x=x, fun=fun(x, *args), success=False, nit=1, message='Wandered off')

        return result

    _, begin = _solve(wandering)

    assert len(starts) >= 2
    assert np.linalg.norm(starts[1][:3] - begin) < 1000.0
    assert abs(np.linalg.norm(starts[1][3:]) - 1) < 1e-9



def testABoundedSolutionIsOnlyTakenIfItIsBetter():
    """ Check that after a false alarm of the divergence check, a bounded solution which is far worse than
        the line is not taken.
    """

    original = scipy.optimize.minimize
    fitted = []
    bounded = []

    def worseBounded(fun, x0, args=(), **options):

        result = original(fun, x0, args=args, **options)

        # The bounded solution is far worse than the line, as seen on real events
        if options.get('method') == 'SLSQP':
            bounded.append(True)
            x = np.r_[x0[:3] + 20000.0, x0[3:] + 0.05]
            return SimpleNamespace(x=x, fun=fun(x, *args), success=True, nit=1, message='Worse')

        if _isLoSFit(fun, options):
            fitted.append(result)

        return result

    # A false alarm of the divergence check
    solver = sys.modules['wmpl.Trajectory.Trajectory']
    with _replaced(solver, 'stateVectorDiverged', lambda *args, **kwargs: True):
        traj, _ = _solve(worseBounded)

    assert bounded
    assert _angle(traj.radiant_eci_mini, fitted[-1].x[3:]) < 1e-6



def _recordLoSFits(seed=20261019, **kwargs):
    """ Solve the trajectory and record every Nelder-Mead minimization of its line-of-sight fit.

    Keyword arguments:
        seed: [int] Seed of the noise. 20261019 by default.
        **kwargs: [dict] Keyword arguments of the Trajectory object.

    Return:
        [list] (x0, cost0, result) of each minimization, with its starting point and cost and its result.
    """

    original = scipy.optimize.minimize
    calls = []

    def recording(fun, x0, args=(), **options):

        result = original(fun, x0, args=args, **options)

        if _isLoSFit(fun, options):
            calls.append((np.array(x0), fun(np.array(x0), *args), result))

        return result

    _solve(recording, seed=seed, **kwargs)

    return calls



def testNoRestartByDefault():
    """ Check that by default the line-of-sight fit is one Nelder-Mead minimization. """

    assert len(_recordLoSFits()) == 1



def testEachRestartStartsFromThePreviousSolution():
    """ Check that each restart starts from the solution of the previous minimization, and never ends with
        a higher cost than it started from.
    """

    calls = _recordLoSFits(los_restarts=2)

    assert len(calls) == 3

    for (_, _, previous), (x0, cost0, result) in zip(calls[:-1], calls[1:]):
        assert np.array_equal(x0, previous.x)
        assert result.fun <= cost0




if __name__ == "__main__":

    # Run all tests
    test_list = [testASmallMoveIsNotADivergenceWhenACoordinateIsNearZero, \
        testALargeMoveIsADivergenceWhenACoordinateIsNegative, testALineWhichRanOutOfEvaluationsIsContinued, \
        testALineWhichWanderedOffAlongItselfIsContinuedFromTheMeteor, \
        testABoundedSolutionIsOnlyTakenIfItIsBetter, testNoRestartByDefault, \
        testEachRestartStartsFromThePreviousSolution]

    for test in test_list:
        test()
        print("PASS {:s}".format(test.__name__))
