""" Tests of the line-of-sight fit of the trajectory solver.

The fit lost a good solution in two ways: a false alarm of the divergence check, after which a far worse
bounded solution was taken, and running out of function evaluations, after which the solution was replaced
by the intersecting planes. A fit which stops in a distant minimum after the timing was estimated must be
minimized again from the solution of the pass before. The optional restarts of the Nelder-Mead minimization
(los_restarts) are also tested. The tests solve a noisy straight trajectory seen from three stations, with
scipy.optimize.minimize replaced to record or alter the minimizations of the fit. The fit minimizes over the
four degrees of freedom of a line (nelderMeadLine), as with the six parameters of a state vector and a radiant
vector the simplex could collapse and stop short of the minimum with success=True. The frame of the four
coordinates is tested, and that the fit ends at the same line from different starts. The angles from the line
are tested to be precise when they are small, as they are at the minimum.

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

from wmpl.Trajectory.Trajectory import stateVectorDiverged
from wmpl.Trajectory.Trajectory import lineFrame, lineFromFrame, minimizeAngleCost, moveStateVector, \
    nelderMeadLine
from wmpl.Trajectory.Trajectory import angleSumMeasurements2Line
from wmpl.Trajectory.Tests.SyntheticTrajectory import observeStraightMeteor
from wmpl.Utils.Math import angleBetweenVectors, vectNorm


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
    """ Tell if a call of scipy.optimize.minimize is a minimization of the line-of-sight fit.

    Arguments:
        fun: [function] Minimized function.
        options: [dict] Keyword arguments of the call.

    Return:
        [bool] True if it is.
    """

    return (getattr(fun, '__name__', '') == 'minimizeAngleCost') \
        and (getattr(options.get('method'), '__name__', '') == 'nelderMeadLine')



def _solve(minimize, seed=20261105, **kwargs):
    """ Solve the straight meteor of the synthetic tests, with scipy.optimize.minimize replaced.

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

    traj, begin, _ = observeStraightMeteor(seed=seed, estimate_timing_vel=False, calc_orbit=False, **kwargs)

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
    assert np.degrees(angleBetweenVectors(traj.radiant_eci_mini, results[-1].x[3:])) < 1e-6



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
    assert np.degrees(angleBetweenVectors(traj.radiant_eci_mini, fitted[-1].x[3:])) < 1e-6



def testAFailedBoundedSolutionIsNotTaken():
    """ Check that after a false alarm of the divergence check, a bounded solution which did not converge
        is not taken, even with a lower cost than the line.
    """

    original = scipy.optimize.minimize
    fitted = []

    def failedBounded(fun, x0, args=(), **options):

        # The bounded minimization stops at its iteration limit with a lower cost
        if options.get('method') == 'SLSQP':
            return SimpleNamespace(x=np.array(x0), fun=0.0, success=False, nit=100, message='Failed')

        result = original(fun, x0, args=args, **options)

        if _isLoSFit(fun, options):
            fitted.append(result)

        return result

    # A false alarm of the divergence check
    solver = sys.modules['wmpl.Trajectory.Trajectory']
    with _replaced(solver, 'stateVectorDiverged', lambda *args, **kwargs: True):
        traj, _ = _solve(failedBounded)

    assert traj.los_mini_status
    assert np.degrees(angleBetweenVectors(traj.radiant_eci_mini, fitted[-1].x[3:])) < 1e-6



def testAFitInADistantMinimumIsMinimizedAgainFromThePreviousSolution():
    """ Check that when the fit after the timing was estimated stops in a distant minimum, with a higher cost
        than at the solution of the pass before, it is minimized again from that solution.
    """

    original = scipy.optimize.minimize
    starts = []
    results = []

    def distant(fun, x0, args=(), **options):

        result = original(fun, x0, args=args, **options)

        if not _isLoSFit(fun, options):
            return result

        starts.append(np.array(x0))

        # The fit of the second pass stops with the radiant 5 degrees away, as seen on real events
        if len(starts) == 2:
            radiant = result.x[3:] + np.tan(np.radians(5))*np.linalg.norm(result.x[3:]) \
                *vectNorm(np.cross(result.x[3:], [0, 0, 1]))
            x = np.r_[result.x[:3], radiant]
            result = SimpleNamespace(x=x, fun=fun(x, *args), success=True, nit=1, message='Distant')

        results.append(result)

        return result

    traj, _, _ = observeStraightMeteor(seed=20261105, estimate_timing_vel=True, calc_orbit=False)

    with _replaced(scipy.optimize, 'minimize', distant):
        traj = traj.run()

    assert len(starts) >= 3

    # The third minimization starts from the solution of the first pass
    first = results[0].x
    assert np.degrees(angleBetweenVectors(starts[2][3:], first[3:])) < 1e-6

    assert np.degrees(angleBetweenVectors(traj.radiant_eci_mini, results[1].x[3:])) > 4
    assert np.degrees(angleBetweenVectors(traj.radiant_eci_mini, first[3:])) < 0.1



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



def testAFailedRestartDoesNotReplaceAConvergedSolution():
    """ Check that a restart which runs out of function evaluations does not replace the converged solution
        it was started from.
    """

    original = scipy.optimize.minimize
    results = []

    def limitedRestart(fun, x0, args=(), **options):

        if not _isLoSFit(fun, options):
            return original(fun, x0, args=args, **options)

        # Only the restart is cut short
        if len(results) == 1:
            options = dict(options, options={'maxfev': 20})

        results.append(original(fun, x0, args=args, **options))

        return results[-1]

    traj, _ = _solve(limitedRestart, los_restarts=1)

    assert results[0].success
    assert not results[1].success
    assert traj.los_mini_status
    assert np.degrees(angleBetweenVectors(traj.radiant_eci_mini, results[0].x[3:])) < 1e-6



def testTheLineFrameMovesTheLineAtTheEndsOfTheMeteor():
    """ Check that a line frame gives the same line at zero, and that each of its coordinates moves the line
        by its value across the line at the ends of the observed part of the meteor.
    """

    traj, begin, radiant = observeStraightMeteor(seed=20261108, calc_orbit=False)

    # A radiant vector which is not a unit vector, and a point 10 km before the beginning
    frame = lineFrame(begin + 10000.0*radiant, 2.0*radiant, traj.observations)
    middle, half_length, unit_radiant, axes = frame

    # The observed part is the length of the meteor
    assert abs(2*half_length - 2.5*30000.0) < 100.0

    # At zero the line is the same, with a unit radiant
    params = lineFromFrame(np.zeros(4), frame)
    assert np.linalg.norm(np.cross(params[:3] - begin, radiant)) < 1e-6
    assert np.linalg.norm(params[3:] - radiant) < 1e-12

    ends = [middle - half_length*unit_radiant, middle + half_length*unit_radiant]

    for k in range(4):

        # Move the line by 1 m along one coordinate
        coords = np.zeros(4)
        coords[k] = 1.0
        params = lineFromFrame(coords, frame)

        # The distance of both ends from the moved line is 1 m
        for end in ends:
            assert abs(np.linalg.norm(np.cross(end - params[:3], params[3:])) - 1.0) < 1e-6

        # The middle stays on the line in a rotation, and moves by 1 m in a shift
        expected = 1.0 if k >= 2 else 0.0
        distance = np.linalg.norm(np.cross(middle - params[:3], params[3:]))
        assert abs(distance - expected) < 1e-6



def testTheLineFitGivesAUnitRadiantAndTheBeginningOfTheMeteor():
    """ Check that the fit returns a unit radiant and the point of the line at the beginning of the meteor,
        starting 10 km before it, 0.1 deg off and with a radiant vector of length 2.
    """

    traj, begin, radiant = observeStraightMeteor(seed=20261108, calc_orbit=False)

    tilted = vectNorm(radiant + np.tan(np.radians(0.1))*vectNorm(np.cross(radiant, [0, 0, 1])))
    p0 = np.r_[begin + 10000.0*radiant, 2.0*tilted]
    result = scipy.optimize.minimize(minimizeAngleCost, p0, args=(traj.observations,), method=nelderMeadLine)

    assert result.success
    assert abs(np.linalg.norm(result.x[3:]) - 1) < 1e-12
    assert np.linalg.norm(result.x[:3] - moveStateVector(result.x[:3], result.x[3:], traj.observations)) < 1e-6

    # The noise of 20 arcsec per point leaves the radiant within a minute of arc
    assert np.degrees(angleBetweenVectors(result.x[3:], radiant)) < 1/60



def testTheLineFitStopsAtItsLimitOfEvaluations():
    """ Check that the fit does not converge and does not exceed a given limit of evaluations of the cost. """

    traj, begin, radiant = observeStraightMeteor(seed=20261108, calc_orbit=False)

    result = scipy.optimize.minimize(minimizeAngleCost, np.r_[begin, radiant], args=(traj.observations,), \
        method=nelderMeadLine, options={'maxfev': 50})

    assert not result.success
    assert result.nfev <= 50



def testTheLineFitEndsAtTheSameLineFromDifferentStarts():
    """ Check that the fit ends at the same line from the true line and from two lines 300 m and 0.05 deg off
        it. With these seeds, the Nelder-Mead minimization of the six parameters ended 0.2 and 0.4 arcsec
        apart.
    """

    for seed in (13, 14):

        traj, begin, radiant = observeStraightMeteor(seed=seed, calc_orbit=False)

        side = vectNorm(np.cross(radiant, [0, 0, 1]))
        up = np.cross(radiant, side)
        tilt = np.tan(np.radians(0.05))
        starts = [np.r_[begin, radiant], np.r_[begin + 300.0*side, vectNorm(radiant + tilt*up)], \
            np.r_[begin - 300.0*up, vectNorm(radiant - tilt*side)]]

        radiants = []
        for p0 in starts:
            result = scipy.optimize.minimize(minimizeAngleCost, p0, args=(traj.observations,), \
                method=nelderMeadLine)
            radiants.append(result.x[3:])

        for other in radiants[1:]:
            assert 3600*np.degrees(angleBetweenVectors(radiants[0], other)) < 0.05



def testSmallAnglesFromTheLineArePrecise():
    """ Check that the angles between the lines of sight and the line are precise down to 1e-9 rad, where the
        arccos of the dot product has an error of up to 1.5e-8 rad. The minimum of the cost has a few angles
        near zero, where that error makes the cost rough.
    """

    station = np.array([4.0e6, 3.0e6, 3.5e6])
    state_vect = station + np.array([1.0e5, -2.0e4, 6.0e4])
    radiant = vectNorm(np.array([0.3, 0.5, -0.8]))

    # Lines of sight to points on the line, tilted by a known small angle out of the plane of the station
    #   and the line
    angles = np.array([1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 1e-6])
    meas = []
    for i, ang in enumerate(angles):
        ray = vectNorm(state_vect - 5000.0*i*radiant - station)
        meas.append(np.cos(ang)*ray + np.sin(ang)*vectNorm(np.cross(ray, radiant)))

    obs = SimpleNamespace(time_data=np.arange(len(angles))*0.1, meas_eci_los=np.array(meas), \
        stat_eci_los=np.array([station]*len(angles)), ignore_list=np.zeros(len(angles), dtype=int), \
        ignore_station=False)

    angle_sum = angleSumMeasurements2Line([obs], state_vect, radiant)

    assert abs(angle_sum - np.mean(angles))/np.mean(angles) < 1e-6




if __name__ == "__main__":

    # Run all tests
    test_list = [testASmallMoveIsNotADivergenceWhenACoordinateIsNearZero, \
        testALargeMoveIsADivergenceWhenACoordinateIsNegative, testALineWhichRanOutOfEvaluationsIsContinued, \
        testALineWhichWanderedOffAlongItselfIsContinuedFromTheMeteor, \
        testABoundedSolutionIsOnlyTakenIfItIsBetter, testAFailedBoundedSolutionIsNotTaken, \
        testAFitInADistantMinimumIsMinimizedAgainFromThePreviousSolution, testNoRestartByDefault, \
        testEachRestartStartsFromThePreviousSolution, testAFailedRestartDoesNotReplaceAConvergedSolution, \
        testTheLineFrameMovesTheLineAtTheEndsOfTheMeteor, \
        testTheLineFitGivesAUnitRadiantAndTheBeginningOfTheMeteor, testTheLineFitStopsAtItsLimitOfEvaluations, \
        testTheLineFitEndsAtTheSameLineFromDifferentStarts, testSmallAnglesFromTheLineArePrecise]

    for test in test_list:
        test()
        print("PASS {:s}".format(test.__name__))
