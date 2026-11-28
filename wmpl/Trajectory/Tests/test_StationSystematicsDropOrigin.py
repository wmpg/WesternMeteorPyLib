""" Tests that wmpl.Trajectory.StationSystematics drops the line from the time from which the line-of-sight
    fit of the solver drops it: dropTimeOrigin of the solver (#202), the first time of the used stations,
    where the solver has it.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_StationSystematicsDropOrigin.py -v
"""

from __future__ import print_function, division, absolute_import

import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')

from wmpl.Trajectory.StationSystematics import stationSystematicErrors
from wmpl.Trajectory.Tests.SyntheticTrajectory import DragMeteor, observeMeteor



def testTheTimesComeFromTheDropOriginOfTheSolver():
    """ With a solver which has dropTimeOrigin, the points are dropped at the times from its origin. """

    meteor = DragMeteor(18000.0, 120.0, 45.0, 0.01, 0.05, 2.7, 6.0)
    traj = observeMeteor(meteor, gravity_correction=True, calc_orbit=False)
    traj.run()

    # The module object of the solver which StationSystematics imports from (wmpl loads every module twice)
    solver = sys.modules['wmpl.Trajectory.Trajectory']
    missing = object()
    original = getattr(solver, 'dropTimeOrigin', missing)

    origin = min(obs.time_data[0] for obs in traj.observations) - 0.25
    origin_calls, times = [], []

    def dropTimeOrigin(observations):
        origin_calls.append(len(observations))
        return origin

    def gravityDropPoint(point, t_rel, radiant_eci, r0, v0z, free_fall):
        times.append(t_rel)
        return point

    solver.dropTimeOrigin = dropTimeOrigin
    traj.gravityFreeFall = lambda: object()
    traj.gravityDropPoint = gravityDropPoint

    try:
        stationSystematicErrors(traj)

    finally:
        del traj.gravityFreeFall, traj.gravityDropPoint
        if original is missing:
            del solver.dropTimeOrigin
        else:
            solver.dropTimeOrigin = original

    first = min(obs.time_data[0] for obs in traj.observations)

    assert origin_calls == [len(traj.observations)]
    assert np.isclose(min(times), first - origin, rtol=0, atol=1e-9)




if __name__ == "__main__":

    testTheTimesComeFromTheDropOriginOfTheSolver()

    print("All tests passed.")
