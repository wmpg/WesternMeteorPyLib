""" Tests of the Monte Carlo run returned by the trajectory solver (timingWeightedRun in
    wmpl.Trajectory.Trajectory), the run nearest the mean of the runs weighted by the likelihood of their
    timing residuals.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_MCBestRun.py -v
"""

from __future__ import print_function, division, absolute_import

import contextlib
import io
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')

from wmpl.Trajectory.Trajectory import timingWeightedRun
from wmpl.Trajectory.Tests.SyntheticTrajectory import observeStraightMeteor
from wmpl.Utils.Math import vectNorm


# Solved straight meteor of SyntheticTrajectory, computed once
SOLVED_METEOR = {}



def solvedStraightMeteor():
    """ Solve the straight meteor of SyntheticTrajectory with 20 arcsec of noise, once.

    Return:
        (traj, n_eff):
            - traj: [Trajectory object] Solved trajectory.
            - n_eff: [float] Effective number of independent timing residuals.
    """

    if not SOLVED_METEOR:
        traj, _, _ = observeStraightMeteor(noise=np.radians(20.0/3600), seed=1)
        with contextlib.redirect_stdout(io.StringIO()):
            traj = traj.run()
        SOLVED_METEOR['traj'] = traj
        df = traj.drag_timing_fit
        SOLVED_METEOR['n_eff'] = float(df.n_data)/float(max(getattr(df, 'res_tau_int', 1.0), 1.0))

    return SOLVED_METEOR['traj'], SOLVED_METEOR['n_eff']



class FakeDragFit(object):
    """ Stand-in for DragTimingFit holding res_rms_s, n_data and res_tau_int. """

    def __init__(self, res_rms_s, n_data, res_tau_int=1.0):
        """ Initialize the stand-in drag fit.

        Arguments:
            res_rms_s: [float] Timing RMS residual (s).
            n_data: [int] Number of timing points.

        Keyword arguments:
            res_tau_int: [float] Integrated autocorrelation time. 1.0 by default.
        """

        self.res_rms_s = res_rms_s
        self.n_data = n_data
        self.res_tau_int = res_tau_int



class FakeRun(object):
    """ Stand-in for a Monte Carlo run with the attributes which timingWeightedRun reads. """

    def __init__(self, traj, offset, chi2, use_drag=True):
        """ Make a run of the solved meteor with its radiant moved across the line.

        Arguments:
            traj: [Trajectory object] Solved trajectory.
            offset: [tuple] Angles by which the radiant is moved along two directions across it (rad).
            chi2: [float] Chi-square of the timing residuals of the run relative to the original solution.

        Keyword arguments:
            use_drag: [bool] Populate a stand-in drag_timing_fit matching chi2. True by default.
        """

        _, n_eff = solvedStraightMeteor()

        # Move the radiant across the line
        radiant = vectNorm(traj.radiant_eci_mini)
        across1 = vectNorm(np.cross(radiant, [0.0, 0.0, 1.0]))
        across2 = np.cross(radiant, across1)

        self.radiant_eci_mini = vectNorm(radiant + offset[0]*across1 + offset[1]*across2)
        self.state_vect_mini = np.copy(traj.state_vect_mini)
        self.v_init = traj.v_init

        if use_drag:
            df = traj.drag_timing_fit
            rms_s = float(df.res_rms_s)*np.sqrt(chi2/n_eff)
            self.drag_timing_fit = FakeDragFit(rms_s, df.n_data, getattr(df, 'res_tau_int', 1.0))
            self.timing_res = rms_s
        else:
            self.drag_timing_fit = None
            n_pts = float(sum([int(np.count_nonzero(np.asarray(obs.ignore_list) == 0)) \
                for obs in traj.observations if not obs.ignore_station]))
            self.timing_res = traj.timing_res*chi2/n_pts



def makeRing(traj, chi2=100.0, scale=1e-3, use_drag=True):
    """ Make runs at the center and on a ring around it, all with the same timing residuals.

    Arguments:
        traj: [Trajectory object] Solved trajectory.

    Keyword arguments:
        chi2: [float] Chi-square of the timing residuals of all runs. 100 by default.
        scale: [float] Radius of the ring (rad). 1e-3 by default.
        use_drag: [bool] Populate drag_timing_fit on each run. True by default.

    Return:
        [list] The run at the center, then the runs on the ring.
    """

    angles = np.radians(np.arange(0, 360, 30))

    return [FakeRun(traj, (0.0, 0.0), chi2, use_drag=use_drag)] \
        + [FakeRun(traj, (scale*np.cos(a), scale*np.sin(a)), chi2, use_drag=use_drag) for a in angles]



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



def testAutocorrelationTimeScalesEffectiveChi2():
    """ Check that a larger residual autocorrelation time reduces the effective residual count so moderate
        residual drops do not over-concentrate the likelihood on an outlier run.
    """

    traj, _ = solvedStraightMeteor()

    # Build a distant run with chi2 lower by 10 at tau_int = 1 (wins over the ring)
    runs = makeRing(traj, chi2=100.0) + [FakeRun(traj, (3e-3, 0.0), 90.0)]
    assert timingWeightedRun(runs, traj) == len(runs) - 1

    # Inflating tau_int by 20 scales delta_chi2 down by 20, keeping the choice at the ring center
    class CorrelatedTraj(object):
        """ Stand-in trajectory with elevated residual autocorrelation time. """

        timing_res = traj.timing_res
        observations = traj.observations
        drag_timing_fit = FakeDragFit(traj.drag_timing_fit.res_rms_s, traj.drag_timing_fit.n_data, \
            20.0*getattr(traj.drag_timing_fit, 'res_tau_int', 1.0))

    assert timingWeightedRun(runs, CorrelatedTraj()) == 0



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
    runs[3].drag_timing_fit = None
    runs[3].timing_res = np.inf
    runs[4].drag_timing_fit = None
    runs[4].timing_res = None

    assert timingWeightedRun(runs, traj) == 0

    # Take a solution without drag_timing_fit or timing residuals
    class NoScale(object):
        """ Stand-in for a solution without timing residuals. """

        drag_timing_fit = None
        timing_res = None
        observations = traj.observations

    assert timingWeightedRun(runs, NoScale()) == len(runs) - 1

    # Only infinite timing residuals but one
    for run in runs[1:]:
        run.drag_timing_fit = None
        run.timing_res = np.inf

    assert timingWeightedRun(runs, traj) == 0



def testLegacyFallbackWithoutDragFit():
    """ Check that timingWeightedRun also works on legacy solutions without drag_timing_fit. """

    traj, _ = solvedStraightMeteor()

    class LegacyTraj(object):
        """ Stand-in for a legacy solution without drag_timing_fit. """

        drag_timing_fit = None
        timing_res = traj.timing_res
        observations = traj.observations

    runs = makeRing(traj, use_drag=False) + [FakeRun(traj, (3e-3, 0.0), 99.0, use_drag=False)]
    assert timingWeightedRun(runs, LegacyTraj()) == 0

    runs = makeRing(traj, use_drag=False) + [FakeRun(traj, (3e-3, 0.0), 60.0, use_drag=False)]
    assert timingWeightedRun(runs, LegacyTraj()) == len(runs) - 1



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
        traj, _, _ = observeStraightMeteor(noise=np.radians(20.0/3600), seed=2, monte_carlo=True, \
            mc_runs=6, mc_cores=2)
        with contextlib.redirect_stdout(io.StringIO()):
            traj_best = traj.run()

    finally:
        module.timingWeightedRun = original

    assert len(chosen) == 1
    assert traj_best is chosen[0]




if __name__ == "__main__":

    testEqualTimingResidualsGiveTheRunNearestTheMean()
    testRunsAreWeightedByTheLikelihoodOfTheirTimingResiduals()
    testAutocorrelationTimeScalesEffectiveChi2()
    testRadiantAndSpeedAreScaledByTheirSpread()
    testWithoutAScaleTheLowestTimingResidualsAreChosen()
    testLegacyFallbackWithoutDragFit()
    testMoreAccurateThanTheLowestTimingResiduals()
    testMonteCarloReturnsTheChosenRun()

    print("All tests passed.")
