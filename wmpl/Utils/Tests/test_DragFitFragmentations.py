""" Tests of the fragmentations in the drag model of wmpl.Utils.DragFit, where the ballistic coefficient can
    jump without penalty at given heights, and of the scan of how spread the last fragmentation was. The
    lengths of the synthetic fireball of SyntheticTrajectory are replaced by those of a drag model whose
    ballistic coefficient only grows in a fragmentation of a known height and spread.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_DragFitFragmentations.py -v
"""

from __future__ import print_function, division, absolute_import

import contextlib
import copy
import functools
import io

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from wmpl.Trajectory.Tests.SyntheticTrajectory import DragMeteor, observeMeteor
from wmpl.Utils.DragFit import DragFit, plotDragFit, spreadScan


# Height of the fragmentation (m), 5 km above the last point, the growth of the ballistic coefficient in it,
#   and the noise of the lengths (m)
HT_FRAG = 53000.0
K_GROWTH = 3.0
LEN_NOISE = 15.0

# Largest spread in the scans (m)
SPREAD_MAX = 3000.0



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



def lastBallisticCoeffStd(drag_fit):
    """ Standard deviation of the logarithm of the ballistic coefficient at the last point. """

    return drag_fit.ballisticCoeffAtLength(np.array([drag_fit.len_last]))[1][0]



@functools.lru_cache(maxsize=None)
def fragmentedTrajectory(spread=0.0, gravity_correction=False):
    """ The synthetic fireball with the lengths of a drag model whose ballistic coefficient is constant,
        except in a fragmentation at HT_FRAG (computed once for every case).

    Keyword arguments:
        spread: [float] Spread of the fragmentation below its height (m). 0 by default, for a jump.
        gravity_correction: [bool] Solve the trajectory with the gravity drop. False by default.

    Return:
        (traj, truth):
            - traj: [Trajectory object] Solved trajectory with the new lengths of its points.
            - truth: [DragFit object] Drag model the lengths come from.
    """

    traj = copy.deepcopy(solvedTrajectory(gravity_correction))

    # Keep the ballistic coefficient constant except in the fragmentation
    truth = fitDrag(traj, knot_hts=[HT_FRAG], knot_spread=spread)
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
        still that of the truth. A spread fragmentation also ends at its height on the dropped path.
    """

    traj, truth = fragmentedTrajectory(0.0, gravity_correction=True)
    drag_fit = fitDrag(traj, knot_hts=[HT_FRAG])
    knot = np.where(drag_fit.pen_weight == 0)[0][0]

    assert drag_fit.gravity_correction
    assert abs(drag_fit.heightAtLength(drag_fit.ramp_beg[knot])[0] - HT_FRAG) < 1.0
    assert abs(lastBallisticCoeff(drag_fit)/lastBallisticCoeff(truth) - 1) < 0.03

    spread_fit = fitDrag(traj, knot_hts=[HT_FRAG], knot_spread=1500.0)
    knot = np.where(spread_fit.pen_weight == 0)[0][0]
    ramp_end = spread_fit.ramp_beg[knot] + spread_fit.ramp_len[knot]

    assert abs(spread_fit.heightAtLength(ramp_end)[0] - (HT_FRAG - 1500.0)) < 1.0



def testTheScanTellsASpreadFragmentationFromAJump():
    """ A fragmentation spread over 1.5 km is acceptable at that spread, and not as a jump nor at the largest
        spreads. The result is still the fit with a jump, its uncertainty reaches the farthest acceptable
        spread, and the true ballistic coefficient at the last point is within it.
    """

    traj, truth = fragmentedTrajectory(1500.0)
    drag_fit, spreads, delta_chi2, log_k_last, log_k_std = spreadScan(traj, [HT_FRAG], spread_max=SPREAD_MAX)
    acceptable = delta_chi2 <= 4.0
    log_k_fit = lastBallisticCoeffStd(fitDrag(traj, knot_hts=[HT_FRAG]))

    assert 1500.0 in spreads[acceptable]
    assert 0.0 not in spreads[acceptable]
    assert SPREAD_MAX not in spreads[acceptable]
    assert np.array_equal(drag_fit.params, fitDrag(traj, knot_hts=[HT_FRAG]).params)
    assert np.isclose(log_k_std, np.hypot(log_k_fit, np.max(np.abs(log_k_last[acceptable] - log_k_last[0]))))
    assert abs(np.log(lastBallisticCoeff(drag_fit)/lastBallisticCoeff(truth))) < log_k_std
    assert np.isclose(np.interp(drag_fit.len_last, *drag_fit.log_k_spread), \
        np.max(np.abs(log_k_last[acceptable] - log_k_last[0])))
    assert np.isclose(lastBallisticCoeffStd(drag_fit), log_k_std)



def testTheScanOfAJumpFindsNoSpread():
    """ A jump is fitted best without a spread. """

    traj, _ = fragmentedTrajectory()
    _, spreads, delta_chi2, _, _ = spreadScan(traj, [HT_FRAG], spread_max=SPREAD_MAX)

    assert spreads[np.argmin(delta_chi2)] == 0.0



def testTheScanWithoutFragmentationsFitsOnce():
    """ Without fragmentations within the observed part there is nothing to scan, and the uncertainty is that
        of the fit.
    """

    drag_fit, spreads, delta_chi2, _, log_k_std = spreadScan(solvedTrajectory(), [150000.0])

    assert np.array_equal(spreads, [0.0])
    assert np.array_equal(delta_chi2, [0.0])
    assert np.isclose(log_k_std, lastBallisticCoeffStd(drag_fit))




def testThePlotShowsTheUncertaintyFromTheSpread():
    """ The plot and the table of a fit from the scan give the final dynamic mass with the uncertainty from
        the fit and from the spread, and those of a plain fit only with that of the fit.
    """

    traj, _ = fragmentedTrajectory(1500.0)
    drag_fit, _, _, _, log_k_std = spreadScan(traj, [HT_FRAG], spread_max=SPREAD_MAX)
    mass = drag_fit.dynamicMassAtLength(drag_fit.len_last, gamma_a=0.78)[0][0]

    fig = plotDragFit(drag_fit, show=False, gamma_a=0.78)
    title = fig.axes[2].get_title()
    plt.close(fig)

    mass_range = "{:.3g} - {:.3g}".format(mass*np.exp(-3*log_k_std), mass*np.exp(3*log_k_std))
    assert "(" + mass_range + ")" in title
    assert "(1 sigma " + mass_range + " kg)" in drag_fit.summaryTable(gamma_a=0.78)

    plain = fitDrag(traj, knot_hts=[HT_FRAG])
    _, mass_lo, mass_hi = plain.dynamicMassAtLength(plain.len_last, gamma_a=0.78)

    fig = plotDragFit(plain, show=False, gamma_a=0.78)
    title = fig.axes[2].get_title()
    plt.close(fig)

    assert "({:.3g} - {:.3g})".format(mass_lo[0], mass_hi[0]) in title
