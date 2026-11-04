""" Tests of the drag model of the motion along the trajectory in wmpl.Utils.DragFit, on a
    synthetic fireball which decelerates from 18 to 10 km/s while it is observed.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_DragFit.py -v
"""

from __future__ import print_function, division, absolute_import

import copy
import functools
import io
import contextlib
import os

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import wmpl
from wmpl.Utils.DragFit import DragFit, plotDragFit
from wmpl.Trajectory.Tests.SyntheticTrajectory import DragMeteor, observeMeteor
from wmpl.Utils.Physics import ballisticCoeffMass
from wmpl.Utils.Pickling import loadPickle


# Directory and the file name of the example trajectory, solved with the gravity correction
TRAJ_DIR = os.path.join(os.path.dirname(wmpl.__file__), 'Dynesty', 'examples', '20191023_091225')
TRAJ_FILE = '20191023_091225_trajectory.pickle'



@functools.lru_cache(maxsize=None)
def _solvedMeteor():
    """ Simulate, observe and solve the synthetic fireball, and fit the drag model (computed once).

    Return:
        (meteor, traj, drag_fit):
            - meteor: [DragMeteor object] Synthetic fireball (the truth).
            - traj: [Trajectory object] Solved trajectory.
            - drag_fit: [DragFit object] Drag fit of the motion.
    """

    # The ballistic coefficient grows five times while the fireball is visible, from 86 to 48 km
    meteor = DragMeteor(18000.0, 120.0, 45.0, 0.01, 0.05, 2.7, 6.0)

    traj = observeMeteor(meteor)
    with contextlib.redirect_stdout(io.StringIO()):
        traj = traj.run()

    drag_fit = DragFit(traj)

    return meteor, traj, drag_fit



def _alongTrackError(meteor, drag_fit, time_arr):
    """ Compute the error of the modelled positions along the trajectory, against the truth projected on the
        solved line. The median error in the observed part, which comes from the trajectory solution and not
        from the model of the motion, is subtracted.

    Arguments:
        meteor: [DragMeteor object] Synthetic fireball.
        drag_fit: [DragFit object] Drag fit of the motion.
        time_arr: [ndarray] Times (s).

    Return:
        [ndarray] Errors along the trajectory (m).
    """

    def _error(t_arr):
        len_true = (drag_fit.state_vect[None, :] - meteor.eciAtTime(t_arr)).dot(drag_fit.radiant_eci)
        return drag_fit.lengthAtTime(t_arr) - len_true

    # Take the median error in the observed part as the offset of the solution
    offset = np.median(_error(np.linspace(0.0, 3.3, 50)))

    return _error(time_arr) - offset



def testFitsTheData():
    """ Check that the model fits the observed lengths to the noise level. """

    _, _, drag_fit = _solvedMeteor()

    # A noise of 30 arcsec at a range of about 100 km is about 15 m
    assert drag_fit.res_rms < 60.0



def testVelocityInTheObservedPart():
    """ Check the velocity inside the observed part, including the deceleration at the end. """

    meteor, _, drag_fit = _solvedMeteor()

    time_arr = np.linspace(0.0, 3.3, 50)
    vel_err = drag_fit.velocityAtLength(drag_fit.lengthAtTime(time_arr)) - meteor.velocityAtTime(time_arr)

    assert np.max(np.abs(vel_err)) < 300.0
    assert np.abs(vel_err[-1]) < 300.0



def testExtrapolationAboveTheBeginning():
    """ Check that the motion above the beginning continues at the nearly constant initial velocity. """

    meteor, _, drag_fit = _solvedMeteor()

    time_arr = np.linspace(-1.5, -0.05, 20)
    vel_err = drag_fit.velocityAtLength(drag_fit.lengthAtTime(time_arr)) - meteor.velocityAtTime(time_arr)

    assert np.max(np.abs(_alongTrackError(meteor, drag_fit, time_arr))) < 50.0
    assert np.max(np.abs(vel_err)) < 50.0



def testExtrapolationBelowTheEnd():
    """ Check the drag-only flight below the end. """

    meteor, _, drag_fit = _solvedMeteor()

    time_arr = np.linspace(3.35, 4.0, 20)
    vel_err = drag_fit.velocityAtLength(drag_fit.lengthAtTime(time_arr)) - meteor.velocityAtTime(time_arr)

    assert np.max(np.abs(_alongTrackError(meteor, drag_fit, time_arr))) < 300.0
    assert np.max(np.abs(vel_err)) < 300.0



def testBallisticCoefficient():
    """ Check the fitted ballistic coefficient and its uncertainty against the truth, which grows linearly
        with time from 0.01 to 0.05 m^2/kg while the fireball is visible.
    """

    _, _, drag_fit = _solvedMeteor()

    time_arr = np.linspace(0.1, 3.2, 8)
    k_arr, log_k_std = drag_fit.ballisticCoeffAtLength(drag_fit.lengthAtTime(time_arr))
    k_true = 0.01 + 0.04*np.clip(time_arr/3.3, 0.0, 1.0)

    # The truth is within 3 sigma of the fit, and the fit sees the growth
    assert np.all(np.abs(np.log(k_arr/k_true)) < 3*log_k_std)
    assert k_arr[-1]/k_arr[0] > 2.0



def testNoDragAboveTheBeginning():
    """ Check that by default the meteor moves without drag above the beginning, so it only slows down by
        gravity going up, and that the drag can be kept there.
    """

    _, traj, drag_fit = _solvedMeteor()

    len_arr = drag_fit.len_first - np.array([0.0, 1e3, 20e3])

    k_arr, _ = drag_fit.ballisticCoeffAtLength(np.concatenate([[drag_fit.len_first - 1.0], len_arr[1:]]))
    assert np.all(k_arr == 0)

    vel = drag_fit.velocityAtLength(len_arr)
    assert np.all(np.diff(vel) < 0)

    # With the drag kept above the beginning, the meteor was faster above
    drag_fit_kept = DragFit(traj, drag_above=True)
    vel_drag = drag_fit_kept.velocityAtLength(len_arr)
    assert vel_drag[1] > vel_drag[0]
    assert np.all(vel_drag[1:] > vel[1:])



def testTimeAndHeightFunctions():
    """ Check the velocities and heights at given times and heights against the functions of the length. """

    _, _, drag_fit = _solvedMeteor()

    time_arr = np.linspace(-0.5, 3.5, 20)
    len_arr = drag_fit.lengthAtTime(time_arr)

    assert np.allclose(drag_fit.velocityAtTime(time_arr), drag_fit.velocityAtLength(len_arr))
    assert np.allclose(drag_fit.heightAtTime(time_arr), drag_fit.heightAtLength(len_arr))
    vel_at_ht = drag_fit.velocityAtHeight(drag_fit.heightAtLength(len_arr))
    assert np.allclose(vel_at_ht, drag_fit.velocityAtLength(len_arr), rtol=1e-6)

    # Outside the modelled part the values are not defined
    assert np.isnan(drag_fit.velocityAtTime(1e4)[0])
    assert np.isnan(drag_fit.heightAtTime(1e4)[0])



def testSummaryTable():
    """ Check that the table of the model has a row for every segment and finds the growth of K. """

    _, _, drag_fit = _solvedMeteor()

    table = drag_fit.summaryTable()
    rows = [line for line in table.split("\n") if line[:4].strip().isdigit()]

    assert len(rows) == drag_fit.n_knots

    k_ratio, mass_ratio = drag_fit.ballisticCoeffGrowth()
    assert k_ratio > 2.0
    assert np.isclose(mass_ratio, k_ratio**(-3))

    assert "Final dynamic mass" in table



def testDynamicMass():
    """ Check the dynamic mass at the end and its uncertainty against the mass of the true ballistic
        coefficient, and that the assumed shape and density scale it.
    """

    _, _, drag_fit = _solvedMeteor()

    # The true ballistic coefficient at the last point
    time_end = drag_fit.timeAtLength(drag_fit.len_last)[0]
    k_true = 0.01 + 0.04*np.clip(time_end/3.3, 0.0, 1.0)
    mass_true = ballisticCoeffMass(k_true, 3500.0, 0.78)

    mass, mass_lo, mass_hi = drag_fit.dynamicMassAtLength(drag_fit.len_last)
    assert mass_lo[0] < mass[0] < mass_hi[0]

    # The truth is within 3 sigma, and the 1 sigma range is three times the relative uncertainty of K
    log_mass_std = np.log(mass_hi[0]/mass[0])
    assert abs(np.log(mass[0]/mass_true)) < 3*log_mass_std
    _, log_k_std = drag_fit.ballisticCoeffAtLength(drag_fit.len_last)
    assert np.isclose(log_mass_std, 3*log_k_std[0])

    # The mass scales as (Gamma*A)^3/rho_m^2
    mass_alt, _, _ = drag_fit.dynamicMassAtLength(drag_fit.len_last, gamma_a=0.55, bulk_density=7800.0)
    assert np.isclose(mass_alt[0], mass[0]*(0.55/0.78)**3*(3500.0/7800.0)**2)

    # There is no mass above the first point without drag
    assert np.isnan(drag_fit.dynamicMassAtLength(drag_fit.len_first - 1e3)[0][0])



def testVelocityIsPhysical():
    """ Check that the velocity is positive and only increases by gravity. """

    _, _, drag_fit = _solvedMeteor()

    valid = np.isfinite(drag_fit.vel_grid)
    vel = drag_fit.vel_grid[valid]
    time = drag_fit.time_grid[valid]

    assert np.all(vel > 0)
    assert np.all(np.diff(vel) <= 9.81*np.diff(time) + 1e-6)



def testHeightsAreConsistent():
    """ Check the heights against the trajectory solution, and that lengths and heights invert each other. """

    _, traj, drag_fit = _solvedMeteor()

    assert abs(drag_fit.heightAtLength(drag_fit.len_first)[0] - traj.rbeg_ele) < 1.0
    assert abs(drag_fit.heightAtLength(drag_fit.len_last)[0] - traj.rend_ele) < 1.0

    len_arr = np.linspace(drag_fit.len_first - 50e3, drag_fit.len_last + 5e3, 20)
    assert np.allclose(drag_fit.lengthAtHeight(drag_fit.heightAtLength(len_arr)), len_arr, atol=0.1)



def testGravityDropMatchesTheSolution():
    """ Check that with the gravity drop the heights agree with the heights of the trajectory solution, and
        that the drop is zero above the beginning.
    """

    traj = loadPickle(TRAJ_DIR, TRAJ_FILE)
    assert traj.gravity_correction

    drag_fit = DragFit(traj)

    for obs in traj.observations:

        used = np.array(obs.ignore_list) == 0
        ht_diff = drag_fit.heightAtLength(obs.state_vect_dist[used]) - obs.model_ht[used]

        assert np.max(np.abs(ht_diff)) < 5.0


    # The drop is zero above the beginning, where the positions are on the line
    len_arr = drag_fit.len_first - np.array([1e3, 10e3])
    assert np.allclose(drag_fit.eciAtLength(len_arr), drag_fit._lineEci(len_arr), rtol=0.0, atol=1e-6)



def testRadiantBentByGravity():
    """ Check that the radiant is bent by gravity from the first point by the angle theta,
        dtheta/dt = g_perp/v, and that the path below the end follows the bent direction.
    """

    traj = loadPickle(TRAJ_DIR, TRAJ_FILE)
    drag_fit = DragFit(traj)

    def _angle(len_arr):
        return np.arccos(np.clip(drag_fit.radiantAtLength(len_arr).dot(drag_fit.radiant_eci), -1, 1))

    # Above the beginning the path is the line
    assert np.allclose(_angle(drag_fit.len_first - np.array([1e3, 10e3])), 0.0, atol=1e-7)

    # The bending grows along the path and agrees with the integral of g_perp/v over time
    len_valid = drag_fit.len_grid[drag_fit._valid]
    len_arr = np.linspace(drag_fit.len_first, len_valid[-1], 20)
    angle = _angle(len_arr)
    assert np.all(np.diff(angle) >= 0)

    valid = drag_fit._valid & (drag_fit.len_grid >= drag_fit.len_first)
    g_perp = drag_fit.g_perp_grid[valid]
    vel = drag_fit.vel_grid[valid]
    time = drag_fit.time_grid[valid]
    theta = np.concatenate([[0.0], np.cumsum(0.5*(g_perp[1:]/vel[1:] + g_perp[:-1]/vel[:-1])*np.diff(time))])
    theta = np.interp(len_arr, drag_fit.len_grid[valid], theta)

    # Compare where the path is bent by less than 2 deg, as the steps are coarse when the meteor is slow
    bent = theta < np.radians(2.0)
    assert np.allclose(angle[bent], theta[bent], rtol=0.01, atol=1e-6)

    # The modelled part ends before the path has bent by 5 deg
    assert np.degrees(angle[-1]) <= 5.0

    # Below the end the path follows the bent direction, compared where it is bent by less than 2 deg
    len_arr = np.linspace(drag_fit.len_last + 100.0, len_valid[-1] - 100.0, 50)
    len_arr = len_arr[_angle(len_arr) < np.radians(2.0)]
    tangent = drag_fit.eciAtLength(len_arr + 10.0) - drag_fit.eciAtLength(len_arr - 10.0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    assert len(len_arr) > 10
    assert np.allclose(drag_fit.radiantAtLength(len_arr), -tangent, rtol=0.0, atol=1e-3)



def testUnconstrainedCoefficient():
    """ Check that the uncertainty of K is infinite in a direction not constrained by the data, and that the
        table and the plot handle it.
    """

    _, _, drag_fit = _solvedMeteor()

    # All directions are constrained in the synthetic fit
    assert drag_fit.params_null.shape[1] == 0
    _, log_k_std = drag_fit.ballisticCoeffAtLength(np.array([drag_fit.len_first, drag_fit.len_last]))
    assert np.all(np.isfinite(log_k_std))

    # Make K at the first point unconstrained
    drag_fit = copy.copy(drag_fit)
    drag_fit.params_null = np.zeros((len(drag_fit.params), 1))
    drag_fit.params_null[2, 0] = 1.0

    _, log_k_std = drag_fit.ballisticCoeffAtLength(np.array([drag_fit.len_first, drag_fit.len_last]))
    assert np.all(np.isinf(log_k_std))

    assert 'inf' in drag_fit.summaryTable()

    fig = plotDragFit(drag_fit, show=False)
    assert np.all(np.isfinite(fig.axes[2].get_xlim()))
    plt.close(fig)



def testPlotWithALargeUncertainty():
    """ Check that the plot can be made when the uncertainty of K is finite but so large that its
        exponential overflows.
    """

    _, _, drag_fit = _solvedMeteor()

    drag_fit = copy.copy(drag_fit)
    drag_fit.params_cov = drag_fit.params_cov*1e12

    _, log_k_std = drag_fit.ballisticCoeffAtLength(np.array([drag_fit.len_first]))
    assert np.isfinite(log_k_std[0]) and (log_k_std[0] > 1000)

    fig = plotDragFit(drag_fit, show=False, gamma_a=0.78)
    assert np.all(np.isfinite(fig.axes[2].get_xlim()))
    plt.close(fig)



def testNoUsedPoints():
    """ Check that a trajectory without used points raises a clear error. """

    _, traj, _ = _solvedMeteor()

    traj = copy.deepcopy(traj)
    for obs in traj.observations:
        obs.ignore_station = True

    with pytest.raises(ValueError, match="no used points"):
        DragFit(traj)



def testPlotDragFit():
    """ Check that the plots of the drag fit are made. """

    _, _, drag_fit = _solvedMeteor()

    for gamma_a in (None, 0.78):

        fig = plotDragFit(drag_fit, show=False, gamma_a=gamma_a)
        assert len(fig.axes) == 3

        plt.close(fig)




if __name__ == "__main__":

    # Run all tests
    test_list = [testFitsTheData, testVelocityInTheObservedPart, testExtrapolationAboveTheBeginning, \
        testExtrapolationBelowTheEnd, testBallisticCoefficient, testNoDragAboveTheBeginning, \
        testTimeAndHeightFunctions, testSummaryTable, testDynamicMass, testVelocityIsPhysical, \
        testHeightsAreConsistent, testGravityDropMatchesTheSolution, testRadiantBentByGravity, \
        testUnconstrainedCoefficient, testPlotWithALargeUncertainty, testNoUsedPoints, testPlotDragFit]

    for test in test_list:
        test()
        print("PASS {:s}".format(test.__name__))
