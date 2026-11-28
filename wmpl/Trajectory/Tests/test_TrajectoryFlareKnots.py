""" Tests of the knots of the ballistic coefficient at the flares in the drag-coupled solver, given as the
    times of the flares of every station (infillTrajectory(flare_times=...)).

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_TrajectoryFlareKnots.py -v
"""

from __future__ import print_function, division, absolute_import

import contextlib
import functools
import io

import numpy as np

import wmpl.Trajectory.Trajectory
from wmpl.Trajectory.Tests.SyntheticTrajectory import DragMeteor, JDT_REF, observeMeteor
from wmpl.Trajectory.Trajectory import Trajectory, trajNoiseGenerator
from wmpl.Utils.TrajConversions import cartesian2Geo


# Beginning and end of the visible part, and time of the jump of K, from the top of the simulation (s)
VIS_BEG = 2.7
VIS_END = 6.0
JUMP_TIME = 5.3



@functools.lru_cache(maxsize=None)
def jumpMeteor():
    """ Simulate a meteor whose ballistic coefficient doubles 0.7 s before the end of its visible part, as at
        a fragmentation.

    Return:
        [DragMeteor object] Synthetic meteor.
    """

    return DragMeteor(18000.0, 120.0, 45.0, 0.008, 0.008, VIS_BEG, VIS_END, k_jumps=[(JUMP_TIME, 2.0)])



def solveJumpMeteor(flare_knots=True, clock_offsets=(('B', 0.4),)):
    """ Solve the meteor with the jump of K, which every station flags as a flare at its nearest frame.

    Keyword arguments:
        flare_knots: [bool] Put knots of K at the flares. True by default.
        clock_offsets: [tuple] Pairs of station ID and its clock offset (s). B ahead by 0.4 s by default.

    Return:
        (traj, output):
            - traj: [Trajectory object] Solved trajectory.
            - output: [str] What the solver printed.
    """

    traj = observeMeteor(jumpMeteor(), noise=10.0, clock_offsets=dict(clock_offsets), \
        flare_time=JUMP_TIME - VIS_BEG, max_toffset=5.0, flare_knots=flare_knots, verbose=True)

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        traj = traj.run()

    return traj, output.getvalue()



@functools.lru_cache(maxsize=None)
def solvedJumpMeteor(flare_knots=True, clock_offsets=(('B', 0.4),)):
    """ Solve the meteor once for the tests which do not change it (see solveJumpMeteor()). """

    return solveJumpMeteor(flare_knots=flare_knots, clock_offsets=clock_offsets)



def endSpeedError(traj):
    """ Compute the error of the speed of the drag fit at its last point.

    Arguments:
        traj: [Trajectory object] Solved trajectory of jumpMeteor().

    Return:
        [float] Speed of the fit minus the true one (m/s).
    """

    drag_fit = traj.drag_timing_fit
    t_last = drag_fit.time_grid[-1] + (traj.jdt_ref - JDT_REF)*86400.0

    return drag_fit.vel_grid[-1] - jumpMeteor().velocityAtTime(t_last)[0]



def report(traj):
    """ Make the report of a solved trajectory without saving it. """

    return traj.saveReport('.', 'report.txt', verbose=False, save_results=False)



def testTheKnotIsAtTheFlareWhateverTheClocksOfTheStations():
    """ Check that the flares of the three stations give one knot at the height of the jump, which does not
        change with the clocks of the stations.
    """

    traj, _ = solvedJumpMeteor()
    traj_clocks, _ = solvedJumpMeteor(clock_offsets=(('B', -0.7), ('C', 0.25)))

    info = traj.flare_knot_info
    assert info['dropped'] is None
    assert info['stations'] == [['A', 'B', 'C']]

    # The flagged frames are within half a frame of the jump, which moves by about 200 m in height
    ht_jump = cartesian2Geo(JDT_REF, *jumpMeteor().eciAtTime(JUMP_TIME - VIS_BEG)[0])[2]
    assert abs(info['hts'][0] - ht_jump) < 250.0

    assert abs(traj_clocks.flare_knot_info['hts'][0] - info['hts'][0]) < 1.0



def testTheKnotImprovesTheSpeedAtTheEndAfterTheJump():
    """ Check that the knot lowers the error of the speed at the last point, leaves the initial velocity
        within its uncertainty, and that the report gives the knot and the fit without it.
    """

    traj, _ = solvedJumpMeteor()
    traj_free, _ = solvedJumpMeteor(flare_knots=False)

    assert abs(endSpeedError(traj)) < 0.5*abs(endSpeedError(traj_free))
    assert abs(traj.v_init - traj_free.v_init) < traj_free.v_init_stddev

    # The fit without the knot is the solution without the knots
    assert abs(traj.flare_knot_info['v_init_free'] - traj_free.v_init) < 0.1*traj_free.v_init_stddev

    report_str = report(traj)
    assert "Knots of K at the flares (where K can jump):" in report_str
    assert "(stations A, B, C)" in report_str
    assert "Without them    = v_init" in report_str

    assert "Knots of K at the flares: not used (disabled)" in report(traj_free)



def testTheKnotsAreDroppedWhenTheyMoveTheInitialVelocity(monkeypatch):
    """ Check that the knots are dropped with a warning, and the trajectory solved again without them, when
        they move the initial velocity by more than allowed, here by anything.
    """

    monkeypatch.setattr(wmpl.Trajectory.Trajectory, 'DRAG_TRAJ_FLARE_MAX_DV_SIGMA', 0.0)

    traj, output = solveJumpMeteor()
    traj_free, _ = solvedJumpMeteor(flare_knots=False)

    assert traj.flare_knot_info['dropped'].startswith("they move the initial velocity")
    assert "WARNING: The knots of K at the flares are dropped" in output
    assert "Updating the solution without the knots of K at the flares..." in output

    assert len(traj.drag_timing_fit.knot_lens) == 0
    assert abs(traj.v_init - traj_free.v_init) < 0.1*traj_free.v_init_stddev

    assert "Knots of K at the flares, not used (they move the initial velocity" in report(traj)



def testTheKnotsAreDroppedWhenTheirFitEndsWithAHigherCost(monkeypatch):
    """ Check that the knots are dropped when the fit with them ends with a higher cost than allowed, here
        with any cost.
    """

    monkeypatch.setattr(wmpl.Trajectory.Trajectory, 'DRAG_TRAJ_FLARE_COST_TOL', -1.0)

    traj, _ = solveJumpMeteor()

    assert traj.flare_knot_info['dropped'].startswith("the fit with them ends with a higher cost")
    assert len(traj.drag_timing_fit.knot_lens) == 0



def testTheKnotsAreDroppedWhenTheFitWithThemFails(monkeypatch):
    """ Check that the knots are dropped with a warning, and the drag fit done without them, when the fit
        with them fails.
    """

    drag_timing_fit = wmpl.Trajectory.Trajectory.DragTimingFit


    def failingWithKnots(*args, **kwargs):
        """ Make the drag fit fail when it has knots. """

        if len(kwargs.get('knot_lens', [])):
            raise ValueError("no fit with knots")

        return drag_timing_fit(*args, **kwargs)


    monkeypatch.setattr(wmpl.Trajectory.Trajectory, 'DragTimingFit', failingWithKnots)

    traj, output = solveJumpMeteor()
    traj_free, _ = solvedJumpMeteor(flare_knots=False)

    assert traj.flare_knot_info['dropped'] == "the fit with them failed (no fit with knots)"
    assert "WARNING: The knots of K at the flares are dropped" in output
    assert abs(traj.v_init - traj_free.v_init) < 0.1*traj_free.v_init_stddev

    assert "Knots of K at the flares, not used (the fit with them failed" in report(traj)



def testTheFlaresAreKeptWhenTheObservationsAreMadeAgain():
    """ Check that the flares of the stations stay at their points when the observations are made again
        from the old ones and in the Monte Carlo runs, and that a flare outside the points is ignored.
    """

    traj, _ = solvedJumpMeteor()
    flare_index = [obs.flare_index for obs in traj.observations]
    assert all(len(index) == 1 for index in flare_index)

    traj_again = Trajectory(traj.jdt_ref, meastype=2, verbose=False)
    for obs in traj.observations:
        traj_again.infillWithObs(obs)

    _, traj_mc, _ = next(trajNoiseGenerator(traj, 1.0))

    for trj in (traj_again, traj_mc):
        assert all(np.array_equal(obs.flare_index, index) for obs, index \
            in zip(trj.observations, flare_index))

    # A flare after the last point of the station is not placed
    obs = traj.observations[0]
    traj_again.infillTrajectory(obs.azim_data, obs.elev_data, obs.time_data, obs.lat, obs.lon, obs.ele, \
        station_id='D', flare_times=[obs.time_data[-1] + 0.1])
    assert len(traj_again.observations[-1].flare_index) == 0



def testTheCostOfTheDragFitIsTheOneOfItsLeastSquares():
    """ Check that DragTimingFit.cost() gives the cost of its last least-squares refinement. """

    drag_fit = solvedJumpMeteor()[0].drag_timing_fit

    assert np.isclose(drag_fit.cost(drag_fit.params), drag_fit.opt_result.cost, rtol=1e-9)
