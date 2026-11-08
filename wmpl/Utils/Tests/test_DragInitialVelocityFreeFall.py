""" Tests of the free fall of the meteor from the line with which
    wmpl.Utils.DragInitialVelocity.droppedLengths finds the points on the trajectory, where the line-of-sight
    fit of the solver finds them: a solution which drops its points by the free fall
    (Trajectory.gravityFreeFall, Trajectory.gravityDropPoint) drops them the same way at the times of the fit,
    and a solution without it keeps its lengths.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_DragInitialVelocityFreeFall.py -v
"""

from __future__ import print_function, division, absolute_import

from types import SimpleNamespace

import numpy as np

from wmpl.Utils.DragInitialVelocity import droppedLengths
from wmpl.Utils.Math import findClosestPoints, vectNorm



def _trajectory(time_data=(0.0, 0.5, 1.0), **attributes):
    """ Make a minimal solved trajectory with one observation of three points, without a gravity drop.

    Keyword arguments:
        time_data: [tuple] Times of the points (s).
        attributes: [dict] Additional attributes of the trajectory.

    Return:
        traj: [object] Object with the attributes of a Trajectory which droppedLengths uses.
    """

    station = np.array([6371e3, 0.0, 0.0])
    meas = np.array([vectNorm(np.array([0.2, 1.0, 0.1*i])) for i in range(3)])

    obs = SimpleNamespace(time_data=np.array(time_data), stat_eci_los=np.array([station]*3), \
        meas_eci_los=meas, ignore_station=False)

    traj = SimpleNamespace(observations=[obs], v0z=0.0, gravity_factor=0.0, \
        state_vect_mini=np.array([6400e3, 100e3, 0.0]), radiant_eci_mini=vectNorm(np.array([0.3, 0.1, 1.0])))

    for name, value in attributes.items():
        setattr(traj, name, value)

    return traj



def _across(traj, size):
    """ A displacement of the given size across the radiant line of the trajectory (m). """

    radiant = traj.radiant_eci_mini
    across = np.array([1.0, 0.0, 0.0]) - radiant[0]*radiant

    return size*vectNorm(across)



def testTheFreeFallOfTheSolutionDropsThePoints():
    """ A solution with the free fall drops the points of the line by it, at the times from the first point
        of all observations, along its radiant: a drop of 50 m across the line gives the lengths on the line
        moved by it.
    """

    calls = []
    table = object()

    traj = _trajectory(time_data=(2.0, 2.5, 3.0))
    drop = _across(traj, 50.0)

    def gravityDropPoint(point, t_rel, radiant_eci, r0, v0z, free_fall):
        calls.append((t_rel, radiant_eci, free_fall))
        return point + drop

    traj.gravityFreeFall = lambda: table
    traj.gravityDropPoint = gravityDropPoint

    lengths = droppedLengths(traj, traj.observations[0])

    # Times from the first point, the radiant of the solution and its table
    assert np.allclose([c[0] for c in calls], [0.0, 0.5, 1.0], rtol=0, atol=1e-12)
    assert all(np.array_equal(c[1], traj.radiant_eci_mini) and (c[2] is table) for c in calls)

    # The lengths of the points on the moved line
    obs = traj.observations[0]
    moved = traj.state_vect_mini + drop
    expected = [abs(np.dot(traj.state_vect_mini - findClosestPoints(stat, meas, moved, \
        traj.radiant_eci_mini)[1], traj.radiant_eci_mini)) for stat, meas in zip(obs.stat_eci_los, \
        obs.meas_eci_los)]

    assert np.allclose(lengths, expected, rtol=0, atol=1e-6)



def testWithoutAFreeFallTheLengthsDoNotChange():
    """ A solution whose free fall is off keeps the lengths of the drop along the vertical. """

    plain = _trajectory()
    lengths = droppedLengths(plain, plain.observations[0])

    def gravityDropPoint(point, t_rel, radiant_eci, r0, v0z, free_fall):
        raise AssertionError('The free fall is off')

    off = _trajectory(gravityFreeFall=lambda: None, gravityDropPoint=gravityDropPoint)

    assert np.array_equal(droppedLengths(off, off.observations[0]), lengths)



if __name__ == "__main__":

    testTheFreeFallOfTheSolutionDropsThePoints()
    testWithoutAFreeFallTheLengthsDoNotChange()

    print("All tests passed.")
