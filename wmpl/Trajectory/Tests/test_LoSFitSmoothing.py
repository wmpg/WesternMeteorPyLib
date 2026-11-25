""" Tests of the smoothing continuation of the line-of-sight fit. The sum of the angles between the lines of
sight and the line has a kink where an angle is zero, at which the Nelder-Mead minimization can stop short of
the minimum. The fit therefore first minimizes the cost with the angles smoothed near zero by decreasing
amounts. The fit is tested to end at the same line from different starts in a case where the starts ended up
to 11 arcsec apart without the smoothing, to keep to a limit of evaluations of the cost, and to give the
exact cost.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_LoSFitSmoothing.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np
import scipy.optimize

import matplotlib
matplotlib.use('Agg')

from wmpl.Trajectory.Trajectory import minimizeAngleCost, nelderMeadLine
from wmpl.Trajectory.Tests.SyntheticTrajectory import observeStraightMeteor
from wmpl.Utils.Math import angleBetweenVectors, vectNorm


def testTheLineFitEndsAtTheSameLineFromDifferentStarts():
    """ Check that the fit ends at the same line from the true line and from lines 300 m and 0.05 deg off it,
        for a meteor of 6 s seen from two stations with 60 arcsec of noise. Without the smoothing of the
        angles, the fits from these starts ended up to 11 arcsec apart.
    """

    traj, begin, radiant = observeStraightMeteor(seed=15, noise=3e-4, duration=6.0, calc_orbit=False)
    traj.observations[2].ignore_station = True

    side = vectNorm(np.cross(radiant, [0, 0, 1]))
    up = np.cross(radiant, side)
    tilt = np.tan(np.radians(0.05))
    starts = [np.r_[begin, radiant], np.r_[begin + 300.0*side, vectNorm(radiant + tilt*up)], \
        np.r_[begin - 300.0*up, vectNorm(radiant - tilt*side)]]

    results = [scipy.optimize.minimize(minimizeAngleCost, p0, args=(traj.observations,), \
        method=nelderMeadLine) for p0 in starts]

    for result in results:
        assert result.success
        assert 3600*np.degrees(angleBetweenVectors(results[0].x[3:], result.x[3:])) < 0.05

    # The costs differ by less than 1e-7 of the cost
    costs = [result.fun for result in results]
    assert max(costs) < min(costs)*(1 + 1e-7)




def testTheLineFitKeepsToItsLimitOfEvaluationsAndGivesTheExactCost():
    """ Check that the fit counts all evaluations of the cost, keeps to a given limit of them, and gives the
        exact cost of the line, also when the limit is reached in a smoothed minimization.
    """

    traj, begin, radiant = observeStraightMeteor(seed=20261108, calc_orbit=False)
    p0 = np.r_[begin + 300.0*vectNorm(np.cross(radiant, [0, 0, 1])), radiant]

    for maxfev in (1, 2, 5, 30, 400, None):

        calls = []

        def cost(params, *args, **kwargs):
            calls.append(1)
            return minimizeAngleCost(params, *args, **kwargs)

        result = nelderMeadLine(cost, p0, args=(traj.observations,), maxfev=maxfev)

        assert result.nfev == len(calls)
        assert (maxfev is None) or (result.nfev <= maxfev)
        assert result.success == (maxfev is None)
        assert abs(result.fun - minimizeAngleCost(result.x, traj.observations)) < 1e-9*result.fun
        assert result.fun <= minimizeAngleCost(p0, traj.observations)




if __name__ == "__main__":

    # Run all tests
    test_list = [testTheLineFitEndsAtTheSameLineFromDifferentStarts, \
        testTheLineFitKeepsToItsLimitOfEvaluationsAndGivesTheExactCost]

    for test in test_list:
        test()
        print("PASS {:s}".format(test.__name__))
