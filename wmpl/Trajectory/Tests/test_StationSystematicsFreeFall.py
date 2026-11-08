""" Tests of the free fall of the meteor from the line which wmpl.Trajectory.StationSystematics applies to the
    line of a solved trajectory, as the solver does: a solution which drops its points by the free fall
    (Trajectory.gravityFreeFall, Trajectory.gravityDropPoint) drops the line the same way, at the times from
    the first point of all observations which the line-of-sight fit takes, and a solution without it keeps its
    errors.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_StationSystematicsFreeFall.py -v
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



def _residuals(traj, **methods):
    """ Measure the systematic errors of the stations with the given methods of the solution and record the
        residuals across the line of every station.

    Keyword arguments:
        methods: [dict] Methods set on the solution while the errors are measured.

    Return:
        (errors, residuals): [tuple] Errors of the stations, and the residuals of every station (rad).
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
    for name, method in methods.items():
        setattr(traj, name, method)

    try:
        errors = stationSystematicErrors(traj)

    finally:
        module.crossTrackResiduals = original
        for name in methods:
            delattr(traj, name)

    return errors, residuals



def testTheFreeFallOfTheSolutionDropsTheLine():
    """ A solution with the free fall drops the line by it: a drop of 300 m across the line gives the
        residuals of the line moved by it, without the drop along the vertical.
    """

    traj = _trajectory()
    drop = _across(traj, 300.0)
    table = object()

    _, residuals_fall = _residuals(traj, gravityFreeFall=lambda: table, \
        gravityDropPoint=lambda point, t_rel, radiant_eci, r0, v0z, free_fall: point + drop)

    # The line moved by the drop, without the gravity correction
    state_vect = traj.state_vect_mini
    traj.state_vect_mini = state_vect + drop
    traj.los_gravity = False

    try:
        _, residuals_moved = _residuals(traj)

    finally:
        traj.state_vect_mini = state_vect
        del traj.los_gravity

    _, residuals = _residuals(traj)

    assert len(residuals) == 3

    for res, res_fall, res_moved in zip(residuals, residuals_fall, residuals_moved):

        assert np.allclose(res_fall, res_moved, rtol=0, atol=1e-9)
        assert np.min(np.abs(res_fall - res)) > 1e-4



def testTheFreeFallIsTakenAtTheTimesOfTheFit():
    """ The free fall is taken at the times from the first point of all observations, also of an ignored
        station which starts earlier, as in the line-of-sight fit, on the radiant line of the solution and
        with its table.
    """

    traj = _trajectory()
    table = object()

    # An ignored station which starts 0.3 s before the others
    t0 = min(obs.time_data[0] for obs in traj.observations)
    earlier = copy.copy(traj.observations[0])
    earlier.time_data = earlier.time_data - (earlier.time_data[0] - t0) - 0.3
    earlier.ignore_station = True

    calls = []

    def gravityDropPoint(point, t_rel, radiant_eci, r0, v0z, free_fall):
        calls.append(t_rel)
        assert np.array_equal(radiant_eci, traj.radiant_eci_mini)
        assert free_fall is table
        return point

    traj.observations.append(earlier)

    try:
        _residuals(traj, gravityFreeFall=lambda: table, gravityDropPoint=gravityDropPoint)

    finally:
        traj.observations.pop()

    assert len(calls) > 0
    assert np.isclose(min(calls), 0.3, rtol=0, atol=1e-9)



def testWithoutAFreeFallTheErrorsDoNotChange():
    """ A solution whose free fall is off gives the errors of the drop along the vertical. """

    traj = _trajectory()

    def gravityDropPoint(point, t_rel, radiant_eci, r0, v0z, free_fall):
        raise AssertionError('The free fall is off')

    # The errors of the drop along the vertical, also where the solver has the free fall (#195)
    errors, _ = _residuals(traj, gravityFreeFall=lambda: None)
    errors_off, _ = _residuals(traj, gravityFreeFall=lambda: None, gravityDropPoint=gravityDropPoint)

    assert errors_off == errors




if __name__ == "__main__":

    # Run all tests
    test_list = [testTheFreeFallOfTheSolutionDropsTheLine, testTheFreeFallIsTakenAtTheTimesOfTheFit, \
        testWithoutAFreeFallTheErrorsDoNotChange]

    for test in test_list:
        test()
        print("PASS {:s}".format(test.__name__))
