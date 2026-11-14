""" Tests of the Monte Carlo run returned by the trajectory solver, the run nearest the mean of the runs
    weighted by the likelihood of their timing residuals.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_MCBestRun.py -v
"""

from __future__ import print_function, division, absolute_import

import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')

from wmpl.Trajectory.Trajectory import timingDifferences, timingWeightedRun
from wmpl.Trajectory.Tests.SyntheticTrajectory import observeStraightMeteor
from wmpl.Utils.Math import vectNorm


# Solved straight meteor of SyntheticTrajectory, computed once
SOLVED_METEOR = {}



def solvedStraightMeteor():
    """ Solve the straight meteor of SyntheticTrajectory with 20 arcsec of noise, once.

    Return:
        (traj, res_count):
            - traj: [Trajectory object] Solved trajectory.
            - res_count: [int] Number of its timing residuals.
    """

    if not SOLVED_METEOR:
        traj, _, _ = observeStraightMeteor(noise=np.radians(20.0/3600), seed=1)
        traj = traj.run()
        SOLVED_METEOR['traj'] = traj
        SOLVED_METEOR['res_count'] = sum([len(diffs) for _, _, diffs, _ \
            in timingDifferences(traj.observations)])

    return SOLVED_METEOR['traj'], SOLVED_METEOR['res_count']



class FakeRun(object):
    """ Stand-in for a Monte Carlo run with the attributes which timingWeightedRun reads. """

    def __init__(self, traj, offset, chi2):
        """ Make a run of the solved meteor with its radiant moved across the line.

        Arguments:
            traj: [Trajectory object] Solved trajectory.
            offset: [tuple] Angles by which the radiant is moved along two directions across it (rad).
            chi2: [float] Chi-square of the timing residuals of the run relative to the original solution.
        """

        _, res_count = solvedStraightMeteor()

        # Move the radiant across the line
        radiant = vectNorm(traj.radiant_eci_mini)
        across1 = vectNorm(np.cross(radiant, [0.0, 0.0, 1.0]))
        across2 = np.cross(radiant, across1)

        self.radiant_eci_mini = vectNorm(radiant + offset[0]*across1 + offset[1]*across2)
        self.state_vect_mini = np.copy(traj.state_vect_mini)
        self.v_init = traj.v_init

        # Set the mean square of the timing residuals which gives this chi-square
        self.timing_res = traj.timing_res*chi2/res_count



def makeRing(traj, chi2=100.0, scale=1e-3):
    """ Make runs at the center and on a ring around it, all with the same timing residuals.

    Arguments:
        traj: [Trajectory object] Solved trajectory.

    Keyword arguments:
        chi2: [float] Chi-square of the timing residuals of all runs. 100 by default.
        scale: [float] Radius of the ring (rad). 1e-3 by default.

    Return:
        [list] The run at the center, then the runs on the ring.
    """

    angles = np.radians(np.arange(0, 360, 30))

    return [FakeRun(traj, (0.0, 0.0), chi2)] + [FakeRun(traj, (scale*np.cos(a), scale*np.sin(a)), chi2) \
        for a in angles]



def testEqualTimingResidualsGiveTheRunNearestTheMean():
    """ Check that with the same timing residuals for all runs the run nearest their mean is chosen. """

    traj, _ = solvedStraightMeteor()

    runs = makeRing(traj)
    assert timingWeightedRun(runs, traj) == 0

    # Moved off center, the mean of the ring stays at the center run
    runs = makeRing(traj)[1:] + [FakeRun(traj, (0.0, 0.0), 100.0)]
    assert timingWeightedRun(runs, traj) == len(runs) - 1



def testRunsAreWeightedByTheLikelihoodOfTheirTimingResiduals():
    """ Check that a run far from the others is chosen only when its timing residuals are so much lower that
        it holds nearly all the likelihood.
    """

    traj, _ = solvedStraightMeteor()

    # Lower by 1 in chi2, a weight e^0.5 times that of each of the 13 other runs
    runs = makeRing(traj) + [FakeRun(traj, (3e-3, 0.0), 99.0)]
    assert timingWeightedRun(runs, traj) == 0

    # Lower by 40 in chi2, a weight e^20 times that of the others
    runs = makeRing(traj) + [FakeRun(traj, (3e-3, 0.0), 60.0)]
    assert timingWeightedRun(runs, traj) == len(runs) - 1



def testRadiantAndSpeedAreScaledByTheirSpread():
    """ Check that the distance to the weighted mean scales the radiant and the initial velocity by their
        spreads over the runs.
    """

    traj, _ = solvedStraightMeteor()

    # Give the runs on the ring speeds 50 m/s above and below that of the solution
    runs = makeRing(traj)
    for k, run in enumerate(runs[1:]):
        run.v_init = traj.v_init + (50.0 if k%2 else -50.0)

    # Move the speed of the run at the center by the spread of the speeds
    runs[0].v_init = traj.v_init + 50.0

    # Add a run at the mean speed, moved by about 3 times the spread of the radiants
    runs.append(FakeRun(traj, (2e-3, 0.0), 100.0))

    assert timingWeightedRun(runs, traj) == 0



def testWithoutAScaleTheLowestTimingResidualsAreChosen():
    """ Check that without the timing residuals of the original solution the run with the lowest timing
        residuals is chosen, and that runs without timing residuals are never chosen.
    """

    traj, _ = solvedStraightMeteor()

    # Give two runs no finite timing residuals
    runs = makeRing(traj) + [FakeRun(traj, (3e-3, 0.0), 99.0)]
    runs[3].timing_res = np.inf
    runs[4].timing_res = None

    assert timingWeightedRun(runs, traj) == 0

    # Take a solution without timing residuals
    class NoScale(object):
        """ Stand-in for a solution without timing residuals. """

        timing_res = None
        observations = traj.observations

    assert timingWeightedRun(runs, NoScale()) == len(runs) - 1

    # Only infinite timing residuals but one
    for run in runs[1:]:
        run.timing_res = np.inf

    assert timingWeightedRun(runs, traj) == 0



def testMoreAccurateThanTheLowestTimingResiduals():
    """ Check on a model of the runs that the chosen run is closer to the truth than the run with the lowest
        timing residuals.
    """

    traj, _ = solvedStraightMeteor()
    rng = np.random.default_rng(7)

    # Uncertainty of the line of sight and of the timing across the radiant (rad)
    sigma = 1e-4

    errors_chosen, errors_lowest = [], []
    for _ in range(300):

        # The truth is at the center, the solutions are noisy
        nominal = rng.normal(0, sigma, 2)
        timing = rng.normal(0, sigma, 2)

        # Scatter the runs around the line-of-sight solution, with chi2 from the timing solution
        offsets = nominal + rng.normal(0, sigma, (60, 2))
        runs = [FakeRun(traj, offset, 50.0 + np.sum((offset - timing)**2)/sigma**2) for offset in offsets]

        # Measure the errors of the chosen run and of the lowest one
        chosen = timingWeightedRun(runs, traj)
        lowest = int(np.argmin([run.timing_res for run in runs]))

        errors_chosen.append(np.linalg.norm(offsets[chosen]))
        errors_lowest.append(np.linalg.norm(offsets[lowest]))

    # Compare the root mean square errors
    rms_chosen = np.sqrt(np.mean(np.array(errors_chosen)**2))
    rms_lowest = np.sqrt(np.mean(np.array(errors_lowest)**2))

    assert rms_chosen < 0.9*rms_lowest



def testMonteCarloReturnsTheChosenRun():
    """ Check that the solver returns the Monte Carlo run chosen by timingWeightedRun. """

    module = sys.modules['wmpl.Trajectory.Trajectory']
    original = module.timingWeightedRun
    chosen = []

    # Record the run chosen by timingWeightedRun
    def recordChoice(mc_results, traj):

        index = original(mc_results, traj)
        chosen.append(mc_results[index])

        return index

    # Solve with Monte Carlo runs, recording the choice
    module.timingWeightedRun = recordChoice
    try:
        traj, _, _ = observeStraightMeteor(noise=np.radians(20.0/3600), seed=2, monte_carlo=True, mc_runs=6, \
            mc_cores=2)
        traj_best = traj.run()

    finally:
        module.timingWeightedRun = original

    assert len(chosen) == 1
    assert traj_best is chosen[0]




if __name__ == "__main__":

    testEqualTimingResidualsGiveTheRunNearestTheMean()
    testRunsAreWeightedByTheLikelihoodOfTheirTimingResiduals()
    testRadiantAndSpeedAreScaledByTheirSpread()
    testWithoutAScaleTheLowestTimingResidualsAreChosen()
    testMoreAccurateThanTheLowestTimingResiduals()
    testMonteCarloReturnsTheChosenRun()

    print("All tests passed.")
