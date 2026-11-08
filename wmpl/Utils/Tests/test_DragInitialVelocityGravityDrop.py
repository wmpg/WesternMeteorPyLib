""" Tests of the gravity drop with which wmpl.Utils.DragInitialVelocity.droppedLengths finds the points on the
    trajectory, where the line-of-sight fit of the solver finds them: from the beginning point of the drop and
    with the factor of the drop of the solution when it has them (Trajectory.gravity_r0 and
    Trajectory.gravityDropFactor), and from each point's distance with the gravity factor otherwise.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_DragInitialVelocityGravityDrop.py -v
"""

from __future__ import print_function, division, absolute_import

import sys
from types import SimpleNamespace

import numpy as np

from wmpl.Utils.DragInitialVelocity import droppedLengths
from wmpl.Utils.Math import vectNorm



def _trajectory(**attributes):
    """ Make a minimal solved trajectory with one observation of three points.

    Keyword arguments:
        attributes: [dict] Additional attributes of the trajectory.

    Return:
        traj: [object] Object with the attributes of a Trajectory which droppedLengths uses.
    """

    station = np.array([6371e3, 0.0, 0.0])
    meas = np.array([vectNorm(np.array([0.2, 1.0, 0.1*i])) for i in range(3)])

    obs = SimpleNamespace(time_data=np.array([0.0, 0.5, 1.0]), stat_eci_los=np.array([station]*3), \
        meas_eci_los=meas, ignore_station=False)

    return SimpleNamespace(observations=[obs], v0z=-10000.0, gravity_factor=0.8, \
        state_vect_mini=np.array([6400e3, 100e3, 0.0]), radiant_eci_mini=vectNorm(np.array([0.3, 0.1, 1.0])), \
        **attributes)



def _dropArguments(traj):
    """ Compute the dropped lengths and record the distance r0 and the factor of every drop.

    Arguments:
        traj: [object] Trajectory.

    Return:
        (r0, factors): [tuple of ndarray] Distances from the centre of the Earth (m) and factors of the drop.
    """

    # The module object from which droppedLengths imports the drop (wmpl loads every module twice)
    module = sys.modules['wmpl.Trajectory.Trajectory']
    original = module.applyGravityDrop
    r0_list, factors = [], []

    def _recording(eci_coord, t, r0, gravity_factor, vz):
        r0_list.append(r0)
        factors.append(gravity_factor)
        return original(eci_coord, t, r0, gravity_factor, vz)

    module.applyGravityDrop = _recording

    try:
        droppedLengths(traj, traj.observations[0])

    finally:
        module.applyGravityDrop = original

    return np.array(r0_list), np.array(factors)



def testTheDropOfTheFitIsUsed():
    """ Check the beginning point and the factor of the drop, with and without those of the solution. """

    # Without them, the distance of each point and the gravity factor
    r0, factors = _dropArguments(_trajectory())
    assert len(np.unique(r0)) == 3
    assert np.all(factors == 0.8)

    # With them, those of the fit
    traj = _trajectory(gravity_r0=6450e3, gravityDropFactor=lambda t_rel: 0.8*(1.0 - 0.1*t_rel))
    r0, factors = _dropArguments(traj)
    assert np.all(r0 == 6450e3)
    assert np.allclose(factors, [0.8, 0.76, 0.72])




if __name__ == "__main__":

    # Run all tests
    test_list = [testTheDropOfTheFitIsUsed]

    for test in test_list:
        test()
        print("PASS {:s}".format(test.__name__))
