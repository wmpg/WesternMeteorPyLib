""" Tests of the initial velocity taken from the drag fit (v_init_drag, --vinitdrag): it is the velocity
    without the drag of DragFit with the drag above the first point, the Monte Carlo runs add the correction
    of the main solution to their own line instead of fitting the drag again, and without the option the
    solver keeps the sliding fit.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_InitialVelocityFromDragFit.py -v
"""

from __future__ import print_function, division, absolute_import

import contextlib
import copy
import functools
import io
from types import SimpleNamespace

import numpy as np

import wmpl.Trajectory.Trajectory as TrajectoryModule
from wmpl.Trajectory.Tests.SyntheticTrajectory import DragMeteor, observeMeteor
from wmpl.Utils.DragFit import DragFit


# Synthetic fireball, visible from about 85 km while it slows from 18 to 10 km/s
METEOR = DragMeteor(18000.0, 120.0, 45.0, 0.01, 0.05, 2.7, 6.0)



def _solve(traj):
    """ Solve the trajectory without printing.

    Arguments:
        traj: [Trajectory object] Trajectory object with the observations.

    Return:
        [Trajectory object] Solved trajectory.
    """

    with contextlib.redirect_stdout(io.StringIO()):
        traj.run()

    return traj



def _noDragFit():
    """ Fail if the solver runs the drag fit. """

    raise AssertionError("The drag fit was run")



@functools.lru_cache(maxsize=None)
def _solvedMeteor(v_init_drag):
    """ Observe and solve the synthetic fireball with the orbit, by default without the option and without
        running the drag fit (computed once per option).

    Arguments:
        v_init_drag: [bool] Take the initial velocity from the drag fit.

    Return:
        [Trajectory object] Solved trajectory.
    """

    if v_init_drag:
        return _solve(observeMeteor(METEOR, v_init_drag=True))

    traj = observeMeteor(METEOR)
    traj.calcDragFitVelocity = _noDragFit

    return _solve(traj)



def testNoDragFitByDefault():
    """ Check that by default the solver does not run the drag fit and keeps the initial velocity of the
        sliding fit, also for a trajectory pickled before the option was added.
    """

    traj_line = _solvedMeteor(False)
    assert traj_line.v_init_drag_delta is None

    # An old pickle does not have the attributes of the option
    traj = observeMeteor(METEOR)
    del traj.v_init_drag, traj.v_init_drag_delta, traj.v_init_drag_stddev
    traj.calcDragFitVelocity = _noDragFit
    traj = _solve(traj)

    assert traj.v_init == traj_line.v_init
    assert traj.v_init_stddev == traj_line.v_init_stddev



def testDragFitVelocity():
    """ Check that with the option the initial velocity is the velocity without the drag of DragFit with the
        drag above the first point, and the correction of the line is kept. It is closer to the truth than
        the line, which is lowered by the deceleration in the first part of the trajectory.
    """

    traj = _solvedMeteor(True)

    # DragFit run again on the final solution
    traj_rest = copy.copy(traj)
    traj_rest.orbit = None
    v_free, v_free_std = DragFit(traj_rest, drag_above=True).dragFreeVelocity()
    assert abs(traj.v_init - v_free) < 0.1
    assert abs(traj.v_init_stddev/v_free_std - 1) < 0.01
    assert traj.v_init_drag_stddev == traj.v_init_stddev

    # The lag is computed with the same velocity
    assert traj.velocity_fit[0] == traj.v_init

    # The correction is the difference from the line of the solution without the option
    traj_line = _solvedMeteor(False)
    assert abs(traj.v_init_drag_delta - (traj.v_init - traj_line.v_init)) < 1e-6

    # The truth at the first point
    ht_first = max(np.max(obs.model_ht[obs.ignore_list == 0]) for obs in traj.observations)
    v_free_true = METEOR.dragFreeVelocity(ht_first)

    assert abs(traj.v_init - v_free_true) < abs(traj_line.v_init - v_free_true)



def testSameRotatingAirInEveryPass():
    """ Check that the drag fit takes the same rotating air with and without an orbit, as the first pass has
        none, the same as DragFit takes from the orbit of the solution, and the air at rest if air_rotation is
        False.
    """

    # The synthetic fireball is solved with the air at rest, here it rotates
    traj = copy.copy(_solvedMeteor(True))
    traj.air_rotation = True
    orbit = traj.orbit
    v_orbit = traj.calcDragFitVelocity()

    traj.orbit = None
    v_no_orbit = traj.calcDragFitVelocity()

    # An orbit with an air speed of 300 m/s is not used either
    traj.orbit = SimpleNamespace(v_init=traj.v_init, v_init_norot=traj.v_init - 300.0)
    v_fake_orbit = traj.calcDragFitVelocity()

    assert v_orbit == v_no_orbit == v_fake_orbit

    # DragFit with the orbit of the solution takes the same rotating air
    traj.orbit = orbit
    v_rotating, _ = DragFit(traj, drag_above=True).dragFreeVelocity()
    assert abs(v_rotating - v_orbit[0]) < 1e-6

    # With the air at rest the velocity changes, here by about 0.6 mm/s
    traj.air_rotation = False
    v_at_rest = traj.calcDragFitVelocity()
    assert abs(v_at_rest[0] - v_orbit[0]) > 1e-4



def testMonteCarloRunReusesTheCorrection():
    """ Check that a Monte Carlo run does not fit the drag again, but adds the correction of the main solution
        to its own line, with the standard deviation of the drag fit.
    """

    traj = _solvedMeteor(True)

    # A noisy Monte Carlo copy of the solution, with and without the option
    np.random.seed(0)
    _, traj_mc, observations = next(TrajectoryModule.trajNoiseGenerator(traj, 1.0))
    traj_mc.calcDragFitVelocity = _noDragFit
    traj_mc_line = copy.deepcopy(traj_mc)
    traj_mc_line.v_init_drag = False

    with contextlib.redirect_stdout(io.StringIO()):
        traj_mc.run(_mc_run=True, _orig_obs=copy.deepcopy(observations))
        traj_mc_line.run(_mc_run=True, _orig_obs=copy.deepcopy(observations))

    assert abs(traj_mc.v_init - traj_mc_line.v_init - traj.v_init_drag_delta) < 1e-6
    assert traj_mc.v_init_stddev == traj.v_init_drag_stddev
    assert traj_mc.velocity_fit[0] == traj_mc.v_init



def testMonteCarloSamplesTheDragFitVelocity():
    """ Check that with the option the Monte Carlo samples the velocity of every run within the standard
        deviation of the drag fit, which the spread of the runs does not include. The runs are copies of the
        solution, so their spread is zero.
    """

    sample_orig = TrajectoryModule.sampleMCVelocities
    generator_orig = TrajectoryModule.parallelComputeGenerator
    sampled = []


    def _sampleMCVelocities(traj_list):
        sampled.append(len(traj_list))
        return sample_orig(traj_list)


    def _parallelComputeGenerator(*args, **kwargs):
        return [copy.deepcopy(traj_main) for _ in range(2)]


    TrajectoryModule.sampleMCVelocities = _sampleMCVelocities
    TrajectoryModule.parallelComputeGenerator = _parallelComputeGenerator

    try:

        np.random.seed(0)
        v_init_sigma = {}
        for v_init_drag in (True, False):

            traj_main = copy.deepcopy(_solvedMeteor(v_init_drag))

            with contextlib.redirect_stdout(io.StringIO()):
                _, uncertainties = TrajectoryModule.monteCarloTrajectory(traj_main, mc_runs=2, \
                    plot_results=False)

            v_init_sigma[v_init_drag] = uncertainties.v_init

    finally:
        TrajectoryModule.sampleMCVelocities = sample_orig
        TrajectoryModule.parallelComputeGenerator = generator_orig

    # Only the three runs with the drag velocity are sampled
    assert sampled == [3]
    assert v_init_sigma[True] > 1.0
    assert v_init_sigma[False] < 1e-6


def testNoteWhenTheShortestWindowIsLow():
    """ Check that without the option the solver notes when the shortest window of the sliding fit reaches
        below V_INIT_DRAG_NOTE_HT, and that it does not note it above that height or with the option. The
        shortest window of the synthetic fireball reaches about 75 km.
    """

    note_ht_orig = TrajectoryModule.V_INIT_DRAG_NOTE_HT
    noted = {}

    try:
        for label, note_ht, v_init_drag in (('below', 100e3, False), ('above', 50e3, False), \
            ('option', 100e3, True)):

            TrajectoryModule.V_INIT_DRAG_NOTE_HT = note_ht
            traj = observeMeteor(METEOR, v_init_drag=v_init_drag, verbose=True)
            if not v_init_drag:
                traj.calcDragFitVelocity = _noDragFit

            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                traj.run()

            noted[label] = "NOTE: The shortest window of the sliding fit" in out.getvalue()

    finally:
        TrajectoryModule.V_INIT_DRAG_NOTE_HT = note_ht_orig

    assert noted == {'below': True, 'above': False, 'option': False}




if __name__ == "__main__":

    # Run all tests
    test_list = [testNoDragFitByDefault, testDragFitVelocity, testSameRotatingAirInEveryPass, \
        testMonteCarloRunReusesTheCorrection, testMonteCarloSamplesTheDragFitVelocity, \
        testNoteWhenTheShortestWindowIsLow]

    for test in test_list:
        test()
        print("PASS {:s}".format(test.__name__))
