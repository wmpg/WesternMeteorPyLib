""" Tests of the gravity drop which wmpl.Trajectory.StationSystematics applies to the line of a solved
    trajectory. It must be the drop of the solver, scaled for a decelerating meteor when the solution has the
    scale (Trajectory.gravityDropFactor), and the drop at a constant velocity otherwise.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_StationSystematicsGravityDrop.py -v
"""

from __future__ import print_function, division, absolute_import

import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')

from wmpl.Trajectory.StationSystematics import stationSystematicErrors
from wmpl.Trajectory.Tests.SyntheticTrajectory import DragMeteor, observeMeteor



def _dropFactors(traj):
    """ Measure the systematic errors of a solved trajectory and record the factors of the gravity drop which
        were applied.

    Arguments:
        traj: [Trajectory object] Solved trajectory.

    Return:
        [ndarray] Factors of the drop, one per call of applyGravityDrop.
    """

    # The module object which stationSystematicErrors uses (wmpl loads every module twice)
    module = sys.modules['wmpl.Trajectory.StationSystematics']
    original = module.applyGravityDrop
    factors = []

    def _recording(eci_coord, t, r0, gravity_factor, vz):
        factors.append(gravity_factor)
        return original(eci_coord, t, r0, gravity_factor, vz)

    module.applyGravityDrop = _recording

    try:
        stationSystematicErrors(traj)

    finally:
        module.applyGravityDrop = original

    return np.array(factors)



def testTheDropHasTheFactorOfTheSolver():
    """ Check that the drop has the gravity factor without a scale, and the factor of the solution with it. """

    meteor = DragMeteor(18000.0, 120.0, 45.0, 0.01, 0.05, 2.7, 6.0)
    traj = observeMeteor(meteor, gravity_correction=True, gravity_factor=0.8, calc_orbit=False)
    traj.run()

    # The factor is that of the drop along the vertical, which solutions without the free fall of the meteor
    #   from the line keep (#195)
    traj.gravityFreeFall = lambda: None

    # Without a scale of the drop, the factor is the gravity factor
    traj.gravity_drop_scale = None
    factors = _dropFactors(traj)
    assert len(factors) > 0
    assert np.all(factors == 0.8)

    # With a scale, the factor of the solution at the time of every point
    traj.gravityDropFactor = lambda t_rel: 0.8*(1.0 - 0.05*t_rel)
    factors = _dropFactors(traj)
    assert np.all(factors <= 0.8)
    assert np.min(factors) < 0.7




if __name__ == "__main__":

    # Run all tests
    test_list = [testTheDropHasTheFactorOfTheSolver]

    for test in test_list:
        test()
        print("PASS {:s}".format(test.__name__))
