""" Tests of the fragmentations in the drag model of wmpl.Utils.DragFit, where the ballistic coefficient can
    jump without penalty at given heights. The lengths of the synthetic fireball of SyntheticTrajectory are
    replaced by those of a drag model whose ballistic coefficient only grows in a jump at a known height.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_DragFitFragmentations.py -v
"""

from __future__ import print_function, division, absolute_import

import contextlib
import copy
import functools
import io

import numpy as np

from wmpl.Trajectory.Tests.SyntheticTrajectory import DragMeteor, observeMeteor
from wmpl.Utils.DragFit import DragFit


# Height of the fragmentation (m), 5 km above the last point, the growth of the ballistic coefficient in it,
#   and the noise of the lengths (m)
HT_FRAG = 53000.0
K_GROWTH = 3.0
LEN_NOISE = 15.0



@functools.lru_cache(maxsize=None)
def solvedTrajectory(gravity_correction=False):
    """ Simulate, observe and solve the synthetic fireball, seen from 86 to 48 km (computed once), with or
        without the gravity drop in the solution.
    """

    traj = observeMeteor(DragMeteor(18000.0, 120.0, 45.0, 0.01, 0.05, 2.7, 6.0), \
        gravity_correction=gravity_correction)
    with contextlib.redirect_stdout(io.StringIO()):
        traj = traj.run()

    return traj



def fitDrag(traj, **kwargs):
    """ Fit the drag model without printing. """

    with contextlib.redirect_stdout(io.StringIO()):
        return DragFit(traj, **kwargs)



def lastBallisticCoeff(drag_fit):
    """ Ballistic coefficient at the last point (m^2/kg). """

    return drag_fit.ballisticCoeffAtLength(np.array([drag_fit.len_last]))[0][0]



@functools.lru_cache(maxsize=None)
def fragmentedTrajectory(gravity_correction=False):
    """ The synthetic fireball with the lengths of a drag model whose ballistic coefficient is constant,
        except for a jump at HT_FRAG (computed once).

    Keyword arguments:
        gravity_correction: [bool] Solve the trajectory with the gravity drop. False by default.

    Return:
        (traj, truth):
            - traj: [Trajectory object] Solved trajectory with the new lengths of its points.
            - truth: [DragFit object] Drag model the lengths come from.
    """

    traj = copy.deepcopy(solvedTrajectory(gravity_correction))

    # Keep the ballistic coefficient constant except at the jump
    truth = fitDrag(traj, knot_hts=[HT_FRAG])
    params = np.copy(truth.params)
    params[3:] = 0.0
    params[3 + np.where(truth.pen_weight == 0)[0][0]] = np.log(K_GROWTH)
    truth.params = params
    truth._solveFull()

    # Give every point the length of the model at its time, with noise
    rng = np.random.default_rng(1)
    for obs in traj.observations:
        time_data = np.array(obs.time_data)
        obs.state_vect_dist = truth.lengthAtTime(time_data) + rng.normal(0.0, LEN_NOISE, len(time_data))

    return traj, truth



def testTheJumpInAFragmentationIsRecovered():
    """ The jump at the height of the fragmentation takes its growth of the ballistic coefficient, and the
        ballistic coefficient at the last point is that of the truth. Without the jump, the penalty spreads
        the growth and the ballistic coefficient at the last point comes out more than 10% too large.
    """

    traj, truth = fragmentedTrajectory()
    drag_fit = fitDrag(traj, knot_hts=[HT_FRAG])
    k_true = lastBallisticCoeff(truth)

    knots = np.where(drag_fit.pen_weight == 0)[0]
    assert len(knots) == 1
    assert abs(np.exp(drag_fit.params[3 + knots[0]])/K_GROWTH - 1) < 0.05
    assert abs(lastBallisticCoeff(drag_fit)/k_true - 1) < 0.03
    assert lastBallisticCoeff(fitDrag(traj))/k_true - 1 > 0.10



def testHeightsOutsideTheObservedPartAreIgnored():
    """ Fragmentations above the first or below the last point add nothing, and the fit is the same as
        without fragmentations.
    """

    traj = solvedTrajectory()
    plain = fitDrag(traj)
    outside = fitDrag(traj, knot_hts=[150000.0, 10000.0])

    assert outside.n_knots == plain.n_knots
    assert np.array_equal(outside.params, plain.params)



def testTheFragmentationIsAtItsHeightOnTheDroppedPath():
    """ With the gravity drop in the solution, the jump is at the height of the fragmentation on the dropped
        path, on which the solution gives the heights, and the ballistic coefficient at the last point is
        still that of the truth.
    """

    traj, truth = fragmentedTrajectory(gravity_correction=True)
    drag_fit = fitDrag(traj, knot_hts=[HT_FRAG])
    knot = np.where(drag_fit.pen_weight == 0)[0][0]

    assert drag_fit.gravity_correction
    assert abs(drag_fit.heightAtLength(drag_fit.ramp_beg[knot])[0] - HT_FRAG) < 1.0
    assert abs(lastBallisticCoeff(drag_fit)/lastBallisticCoeff(truth) - 1) < 0.03
