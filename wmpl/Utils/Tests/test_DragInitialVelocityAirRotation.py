""" Tests of the displacement by drag against the rotating air with which
    wmpl.Utils.DragInitialVelocity.droppedLengths finds the points on the trajectory, where the line-of-sight
    fit of the solver finds them: the displacement of the solution (Trajectory.airRotationShiftAt) is added to
    the dropped point at the time of the fit, and a solution without it keeps its lengths.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_DragInitialVelocityAirRotation.py -v
"""

from __future__ import print_function, division, absolute_import

from types import SimpleNamespace

import numpy as np

from wmpl.Utils.DragInitialVelocity import droppedLengths
from wmpl.Utils.Math import vectNorm



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
        meas_eci_los=meas)

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



def testAConstantDisplacementGivesTheLengthsOfTheMovedLine():
    """ Without a drop, a constant displacement across the line gives the lengths on the line moved by it. """

    traj = _trajectory()
    shift = _across(traj, 300.0)
    lengths = droppedLengths(traj, traj.observations[0])

    # The displacement of the solution
    shifted = _trajectory(airRotationShiftAt=lambda t_rel, radiant_eci: shift)
    lengths_shifted = droppedLengths(shifted, shifted.observations[0])

    # The line moved by the displacement
    moved = _trajectory()
    moved.state_vect_mini = moved.state_vect_mini + shift
    lengths_moved = droppedLengths(moved, moved.observations[0])

    assert np.allclose(lengths_shifted, lengths_moved, rtol=0, atol=1e-6)
    assert np.min(np.abs(lengths_shifted - lengths)) > 5.0



def testTheDisplacementIsTakenAtTheTimesOfTheFit():
    """ The displacement is taken at the times from the first point of all observations, as in the fit, and on
        the radiant line of the solution.
    """

    calls = []

    def shiftAt(t_rel, radiant_eci):
        calls.append((t_rel, radiant_eci))
        return np.zeros(3)

    traj = _trajectory(airRotationShiftAt=shiftAt)

    # A second observation which starts earlier
    earlier = SimpleNamespace(**vars(traj.observations[0]))
    earlier.time_data = traj.observations[0].time_data - 0.2
    traj.observations.append(earlier)

    droppedLengths(traj, traj.observations[0])

    assert np.allclose([t_rel for t_rel, _ in calls], [0.2, 0.7, 1.2], rtol=0, atol=1e-12)
    assert all(np.array_equal(radiant, traj.radiant_eci_mini) for _, radiant in calls)



def testWithoutADisplacementTheLengthsDoNotChange():
    """ A solution without the displacement and one with a displacement of 0 give the same lengths. """

    traj = _trajectory()
    zero = _trajectory(airRotationShiftAt=lambda t_rel, radiant_eci: np.zeros(3))

    assert np.array_equal(droppedLengths(traj, traj.observations[0]), \
        droppedLengths(zero, zero.observations[0]))




if __name__ == "__main__":

    # Run all tests
    test_list = [testAConstantDisplacementGivesTheLengthsOfTheMovedLine, \
        testTheDisplacementIsTakenAtTheTimesOfTheFit, testWithoutADisplacementTheLengthsDoNotChange]

    for test in test_list:
        test()
        print("PASS {:s}".format(test.__name__))
