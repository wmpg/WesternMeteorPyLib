""" Tests of the displacement by drag against the rotating air which wmpl.Trajectory.StationSystematics
    applies to the line of a solved trajectory, as the solver does: the displacement of the solution
    (Trajectory.airRotationShiftAt) is added to the dropped point, at the times from the first point of all
    observations which the line-of-sight fit takes, and a solution without it keeps its errors.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_StationSystematicsAirRotation.py -v
"""

from __future__ import print_function, division, absolute_import

import copy
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')

from wmpl.Trajectory.StationSystematics import stationSystematicErrors
from wmpl.Trajectory.Tests.SyntheticTrajectory import DragMeteor, observeMeteor
from wmpl.Utils.Math import vectNorm


# Solved trajectory, shared by the tests
_TRAJ = []



def _trajectory():
    """ Solve a decelerating meteor with the gravity drop once.

    Return:
        [Trajectory object] Solved trajectory.
    """

    if not _TRAJ:
        meteor = DragMeteor(18000.0, 120.0, 45.0, 0.01, 0.05, 2.7, 6.0)
        traj = observeMeteor(meteor, gravity_correction=True, calc_orbit=False)
        traj.run()
        _TRAJ.append(traj)

    return _TRAJ[0]



def _across(traj, size):
    """ A displacement of the given size across the radiant line of the trajectory (m). """

    radiant = traj.radiant_eci_mini
    across = np.array([1.0, 0.0, 0.0]) - radiant[0]*radiant

    return size*vectNorm(across)



def _errors(traj, shift_at=None):
    """ Measure the systematic errors of the stations with the given displacement of the solution, or without
        one.
    """

    if shift_at is not None:
        traj.airRotationShiftAt = shift_at

    try:
        return stationSystematicErrors(traj)

    finally:
        if shift_at is not None:
            del traj.airRotationShiftAt



def _residuals(traj, shift_at=None):
    """ Measure the systematic errors of the stations and record the residuals across the line of every
        station.

    Return:
        [list] Residuals of every station (rad).
    """

    # The module object which stationSystematicErrors uses (wmpl loads every module twice)
    module = sys.modules['wmpl.Trajectory.StationSystematics']
    original = module.crossTrackResiduals
    residuals = []

    def _recording(*args, **kwargs):
        result = original(*args, **kwargs)
        residuals.append(result[0])
        return result

    module.crossTrackResiduals = _recording

    try:
        _errors(traj, shift_at)

    finally:
        module.crossTrackResiduals = original

    return residuals



def _fitTimeOrigin(observations):
    """ The time from which the line-of-sight fit of the solver drops the points: the first time of the used
        stations (dropTimeOrigin, #202), or of all observations with a solver without it.
    """

    try:
        from wmpl.Trajectory.Trajectory import dropTimeOrigin

    except ImportError:
        return min(obs.time_data[0] for obs in observations)

    return dropTimeOrigin(observations)



def testAConstantDisplacementGivesTheResidualsOfTheMovedLine():
    """ A constant displacement across the line gives the residuals of the line moved by it, which are not
        those of the line.
    """

    traj = _trajectory()
    shift = _across(traj, 300.0)

    residuals = _residuals(traj)
    residuals_shifted = _residuals(traj, lambda t_rel, radiant_eci: shift)

    state_vect = traj.state_vect_mini
    traj.state_vect_mini = state_vect + shift

    try:
        residuals_moved = _residuals(traj)

    finally:
        traj.state_vect_mini = state_vect

    assert len(residuals) == 3

    for res, res_shifted, res_moved in zip(residuals, residuals_shifted, residuals_moved):

        # The drop of each point is taken from its distance, which differs on the moved line by up to a few
        #   millimetres
        assert np.allclose(res_shifted, res_moved, rtol=0, atol=1e-7)
        assert np.min(np.abs(res_shifted - res)) > 1e-4



def testTheDisplacementIsTakenAtTheTimesOfTheFit():
    """ The displacement and the factor of the drop are taken at the times of the line-of-sight fit, also
        with an ignored station which starts earlier (which does not set them with #202), and on the radiant
        line of the solution.
    """

    traj = _trajectory()

    # An ignored station which starts 0.3 s before the others
    t0 = min(obs.time_data[0] for obs in traj.observations)
    earlier = copy.copy(traj.observations[0])
    earlier.time_data = earlier.time_data - (earlier.time_data[0] - t0) - 0.3
    earlier.ignore_station = True

    shift_calls, factor_calls = [], []

    def shiftAt(t_rel, radiant_eci):
        shift_calls.append(t_rel)
        assert np.array_equal(radiant_eci, traj.radiant_eci_mini)
        return np.zeros(3)

    def dropFactor(t_rel):
        factor_calls.append(t_rel)
        return traj.gravity_factor

    # The factor is that of the drop along the vertical, which solutions without the free fall of the meteor
    #   from the line keep (#195)
    traj.observations.append(earlier)
    traj.gravityDropFactor = dropFactor
    traj.gravityFreeFall = lambda: None

    try:
        _errors(traj, shiftAt)
        first = t0 - _fitTimeOrigin(traj.observations)

    finally:
        traj.observations.pop()
        del traj.gravityDropFactor
        del traj.gravityFreeFall

    assert len(shift_calls) > 0
    assert np.isclose(min(shift_calls), first, rtol=0, atol=1e-9)
    assert np.array_equal(shift_calls, factor_calls)



def testWithoutADisplacementTheErrorsDoNotChange():
    """ A solution without the displacement and one with a displacement of 0 give the same errors. """

    traj = _trajectory()

    assert _errors(traj) == _errors(traj, lambda t_rel, radiant_eci: np.zeros(3))




if __name__ == "__main__":

    # Run all tests
    test_list = [testAConstantDisplacementGivesTheResidualsOfTheMovedLine, \
        testTheDisplacementIsTakenAtTheTimesOfTheFit, testWithoutADisplacementTheErrorsDoNotChange]

    for test in test_list:
        test()
        print("PASS {:s}".format(test.__name__))
