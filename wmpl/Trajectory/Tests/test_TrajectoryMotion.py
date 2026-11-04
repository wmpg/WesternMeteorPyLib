""" Tests of the drag model of the motion along the trajectory in wmpl.Trajectory.TrajectoryMotion, on a
    synthetic fireball which decelerates from 18 to 10 km/s while it is observed.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_TrajectoryMotion.py -v
"""

from __future__ import print_function, division, absolute_import

import functools
import io
import contextlib
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import wmpl
from wmpl.Trajectory.TrajectoryMotion import TrajectoryMotion, ceplechaType, plotMotion
from wmpl.Trajectory.Tests.SyntheticTrajectory import DragMeteor, observeMeteor
from wmpl.Utils.Pickling import loadPickle


# Directory and the file name of the example trajectory, solved with the gravity correction
TRAJ_DIR = os.path.join(os.path.dirname(wmpl.__file__), 'Dynesty', 'examples', '20191023_091225')
TRAJ_FILE = '20191023_091225_trajectory.pickle'



@functools.lru_cache(maxsize=None)
def _solvedMeteor():
    """ Simulate, observe and solve the synthetic fireball, and fit the motion model (computed once).

    Return:
        (meteor, traj, motion):
            - meteor: [DragMeteor object] Synthetic fireball (the truth).
            - traj: [Trajectory object] Solved trajectory.
            - motion: [TrajectoryMotion object] Fitted model of the motion.
    """

    # The ballistic coefficient grows five times while the fireball is visible, from 86 to 48 km
    meteor = DragMeteor(18000.0, 120.0, 45.0, 0.01, 0.05, 2.7, 6.0)

    traj = observeMeteor(meteor)
    with contextlib.redirect_stdout(io.StringIO()):
        traj = traj.run()

    motion = TrajectoryMotion(traj)

    return meteor, traj, motion



def _alongTrackError(meteor, motion, time_arr):
    """ Compute the error of the modelled positions along the trajectory, against the truth projected on the
        solved line. The median error in the observed part, which comes from the trajectory solution and not
        from the model of the motion, is subtracted.

    Arguments:
        meteor: [DragMeteor object] Synthetic fireball.
        motion: [TrajectoryMotion object] Fitted model of the motion.
        time_arr: [ndarray] Times (s).

    Return:
        [ndarray] Errors along the trajectory (m).
    """

    def _error(t_arr):
        len_true = (motion.state_vect[None, :] - meteor.eciAtTime(t_arr)).dot(motion.radiant_eci)
        return motion.lengthAtTime(t_arr) - len_true

    # Take the median error in the observed part as the offset of the solution
    offset = np.median(_error(np.linspace(0.0, 3.3, 50)))

    return _error(time_arr) - offset



def testFitsTheData():
    """ Check that the model fits the observed lengths to the noise level. """

    _, _, motion = _solvedMeteor()

    # A noise of 30 arcsec at a range of about 100 km is about 15 m
    assert motion.res_rms < 60.0



def testVelocityInTheObservedPart():
    """ Check the velocity inside the observed part, including the deceleration at the end. """

    meteor, _, motion = _solvedMeteor()

    time_arr = np.linspace(0.0, 3.3, 50)
    vel_err = motion.velocityAtLength(motion.lengthAtTime(time_arr)) - meteor.velocityAtTime(time_arr)

    assert np.max(np.abs(vel_err)) < 300.0
    assert np.abs(vel_err[-1]) < 300.0



def testExtrapolationAboveTheBeginning():
    """ Check that the motion above the beginning continues at the nearly constant initial velocity. """

    meteor, _, motion = _solvedMeteor()

    time_arr = np.linspace(-1.5, -0.05, 20)
    vel_err = motion.velocityAtLength(motion.lengthAtTime(time_arr)) - meteor.velocityAtTime(time_arr)

    assert np.max(np.abs(_alongTrackError(meteor, motion, time_arr))) < 50.0
    assert np.max(np.abs(vel_err)) < 50.0



def testExtrapolationBelowTheEnd():
    """ Check the drag-only flight below the end. """

    meteor, _, motion = _solvedMeteor()

    time_arr = np.linspace(3.35, 4.0, 20)
    vel_err = motion.velocityAtLength(motion.lengthAtTime(time_arr)) - meteor.velocityAtTime(time_arr)

    assert np.max(np.abs(_alongTrackError(meteor, motion, time_arr))) < 300.0
    assert np.max(np.abs(vel_err)) < 300.0



def testBallisticCoefficient():
    """ Check the fitted ballistic coefficient and its uncertainty against the truth, which grows linearly
        with time from 0.01 to 0.05 m^2/kg while the fireball is visible.
    """

    _, _, motion = _solvedMeteor()

    time_arr = np.linspace(0.1, 3.2, 8)
    k_arr, log_k_std = motion.ballisticCoeffAtLength(motion.lengthAtTime(time_arr))
    k_true = 0.01 + 0.04*np.clip(time_arr/3.3, 0.0, 1.0)

    # The truth is within 3 sigma of the fit, and the fit sees the growth
    assert np.all(np.abs(np.log(k_arr/k_true)) < 3*log_k_std)
    assert k_arr[-1]/k_arr[0] > 2.0



def testNoDragAboveTheBeginning():
    """ Check that by default the meteor moves without drag above the beginning, so it only slows down by
        gravity going up, and that the drag can be kept there.
    """

    _, traj, motion = _solvedMeteor()

    len_arr = motion.len_first - np.array([0.0, 1e3, 20e3])

    k_arr, _ = motion.ballisticCoeffAtLength(len_arr[1:])
    assert np.all(k_arr == 0)

    vel = motion.velocityAtLength(len_arr)
    assert np.all(np.diff(vel) < 0)

    # With the drag kept above the beginning, the meteor was faster above
    motion_drag = TrajectoryMotion(traj, drag_above=True)
    vel_drag = motion_drag.velocityAtLength(len_arr)
    assert vel_drag[1] > vel_drag[0]
    assert np.all(vel_drag[1:] > vel[1:])



def testTimeAndHeightFunctions():
    """ Check the velocities and heights at given times and heights against the functions of the length. """

    _, _, motion = _solvedMeteor()

    time_arr = np.linspace(-0.5, 3.5, 20)
    len_arr = motion.lengthAtTime(time_arr)

    assert np.allclose(motion.velocityAtTime(time_arr), motion.velocityAtLength(len_arr))
    assert np.allclose(motion.heightAtTime(time_arr), motion.heightAtLength(len_arr))
    vel_at_ht = motion.velocityAtHeight(motion.heightAtLength(len_arr))
    assert np.allclose(vel_at_ht, motion.velocityAtLength(len_arr), rtol=1e-6)

    # Outside the modelled part the values are not defined
    assert np.isnan(motion.velocityAtTime(1e4)[0])
    assert np.isnan(motion.heightAtTime(1e4)[0])



def testSummaryTable():
    """ Check that the table of the model has a row for every segment and finds the growth of K. """

    _, _, motion = _solvedMeteor()

    table = motion.summaryTable()
    rows = [line for line in table.split("\n") if line[:4].strip().isdigit()]

    assert len(rows) == motion.n_knots

    k_ratio, mass_ratio = motion.ballisticCoeffGrowth()
    assert k_ratio > 2.0
    assert np.isclose(mass_ratio, k_ratio**(-3))



def testApparentAblationCoeff():
    """ Check the apparent ablation coefficient against the truth and the fireball types. """

    meteor, _, motion = _solvedMeteor()

    # Compute the true coefficient from the growth of K and the velocities between the first and the last
    #   observed point
    time_beg, time_end = motion.timeAtLength(np.array([motion.len_first, motion.len_last]))
    k_true = 0.01 + 0.04*np.clip(np.array([time_beg, time_end])/3.3, 0.0, 1.0)
    vel_true = meteor.velocityAtTime(np.array([time_beg, time_end]))
    sigma_true = 6*np.log(k_true[1]/k_true[0])/(vel_true[0]**2 - vel_true[1]**2)

    sigma_app, sigma_app_std = motion.apparentAblationCoeff()
    assert abs(sigma_app - sigma_true) < 3*sigma_app_std

    # The closest types
    assert ceplechaType(0.014)[0] == 'I'
    assert ceplechaType(0.03)[0] == 'II'
    assert ceplechaType(0.5)[0] == 'IIIB'
    assert ceplechaType(-0.01) is None

    assert 'Ceplecha' in motion.ablationReport()



def testVelocityIsPhysical():
    """ Check that the velocity is positive and only increases by gravity. """

    _, _, motion = _solvedMeteor()

    valid = np.isfinite(motion.vel_grid)
    vel = motion.vel_grid[valid]
    time = motion.time_grid[valid]

    assert np.all(vel > 0)
    assert np.all(np.diff(vel) <= 9.81*np.diff(time) + 1e-6)



def testHeightsAreConsistent():
    """ Check the heights against the trajectory solution, and that lengths and heights invert each other. """

    _, traj, motion = _solvedMeteor()

    assert abs(motion.heightAtLength(motion.len_first)[0] - traj.rbeg_ele) < 1.0
    assert abs(motion.heightAtLength(motion.len_last)[0] - traj.rend_ele) < 1.0

    len_arr = np.linspace(motion.len_first - 50e3, motion.len_last + 5e3, 20)
    assert np.allclose(motion.lengthAtHeight(motion.heightAtLength(len_arr)), len_arr, atol=0.1)



def testGravityDropMatchesTheSolution():
    """ Check that with the gravity drop the heights agree with the heights of the trajectory solution, and
        that the drop is zero above the beginning.
    """

    traj = loadPickle(TRAJ_DIR, TRAJ_FILE)
    assert traj.gravity_correction

    motion = TrajectoryMotion(traj)

    for obs in traj.observations:

        used = np.array(obs.ignore_list) == 0
        ht_diff = motion.heightAtLength(obs.state_vect_dist[used]) - obs.model_ht[used]

        assert np.max(np.abs(ht_diff)) < 5.0


    # The drop is zero above the beginning, where the positions are on the line
    len_arr = motion.len_first - np.array([1e3, 10e3])
    assert np.allclose(motion.eciAtLength(len_arr), motion._lineEci(len_arr), rtol=0.0, atol=1e-6)



def testPlotMotion():
    """ Check that the plots of the motion model are made. """

    _, _, motion = _solvedMeteor()

    for show_ablation in (False, True):

        fig = plotMotion(motion, show=False, show_ablation=show_ablation)
        assert len(fig.axes) == 3

        plt.close(fig)




if __name__ == "__main__":

    # Run all tests
    test_list = [testFitsTheData, testVelocityInTheObservedPart, testExtrapolationAboveTheBeginning, \
        testExtrapolationBelowTheEnd, testBallisticCoefficient, testNoDragAboveTheBeginning, \
        testTimeAndHeightFunctions, testSummaryTable, testApparentAblationCoeff, testVelocityIsPhysical, \
        testHeightsAreConsistent, testGravityDropMatchesTheSolution, testPlotMotion]

    for test in test_list:
        test()
        print("PASS {:s}".format(test.__name__))
