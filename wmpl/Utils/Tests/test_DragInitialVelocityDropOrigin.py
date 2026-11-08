""" Tests that wmpl.Utils.DragInitialVelocity.droppedLengths drops the points from the time from which the
    line-of-sight fit of the solver drops them: dropTimeOrigin of the solver (#202), the first time of the
    used stations, where the solver has it.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_DragInitialVelocityDropOrigin.py -v
"""

from __future__ import print_function, division, absolute_import

import sys
from types import SimpleNamespace

import numpy as np

from wmpl.Utils.DragInitialVelocity import droppedLengths
from wmpl.Utils.Math import vectNorm



def testTheTimesComeFromTheDropOriginOfTheSolver():
    """ With a solver which has dropTimeOrigin, the points are dropped at the times from its origin. """

    station = np.array([6371e3, 0.0, 0.0])
    meas = np.array([vectNorm(np.array([0.2, 1.0, 0.1*i])) for i in range(3)])
    obs = SimpleNamespace(time_data=np.array([2.0, 2.5, 3.0]), stat_eci_los=np.array([station]*3), \
        meas_eci_los=meas, ignore_station=False)

    times = []

    def gravityDropPoint(point, t_rel, radiant_eci, r0, v0z, free_fall):
        times.append(t_rel)
        return point

    radiant_eci = vectNorm(np.array([0.3, 0.1, 1.0]))
    traj = SimpleNamespace(observations=[obs], v0z=0.0, gravity_factor=0.0, \
        state_vect_mini=np.array([6400e3, 100e3, 0.0]), radiant_eci_mini=radiant_eci, \
        gravityFreeFall=lambda: object())
    traj.gravityDropPoint = gravityDropPoint

    # The module object of the solver which droppedLengths imports from (wmpl loads every module twice)
    solver = sys.modules['wmpl.Trajectory.Trajectory']
    missing = object()
    original = getattr(solver, 'dropTimeOrigin', missing)
    solver.dropTimeOrigin = lambda observations: 1.75

    try:
        droppedLengths(traj, obs)

    finally:
        if original is missing:
            del solver.dropTimeOrigin
        else:
            solver.dropTimeOrigin = original

    assert np.allclose(times, [0.25, 0.75, 1.25], rtol=0, atol=1e-12)



if __name__ == "__main__":

    testTheTimesComeFromTheDropOriginOfTheSolver()

    print("All tests passed.")
