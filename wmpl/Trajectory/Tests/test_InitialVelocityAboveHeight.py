""" Tests of the initial velocity estimated as the average velocity above a height (v_init_ht, --vinitht): its
    standard deviation is that of the line fitted to the points above the height, and it is the one the solver
    and the orbit give with the velocity.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_InitialVelocityAboveHeight.py -v
"""

from __future__ import print_function, division, absolute_import

import contextlib
import io

import numpy as np
import scipy.optimize

from wmpl.Trajectory.Tests.SyntheticTrajectory import DragMeteor, observeMeteor
from wmpl.Utils.Math import lineFunc


# The synthetic meteor, and the height above which its initial velocity is estimated (km)
METEOR = DragMeteor(18000.0, 120.0, 45.0, 0.01, 0.05, 2.7, 6.0)
HT_INIT = 80.0



def testTheVelocityAboveAHeightComesWithItsStddev():
    """ The standard deviation of the average velocity above a height is that of the slope of the line fitted
        to the points above it, and the solver and the orbit give it with the velocity, instead of that of the
        default fit of the first part (about 2 m/s here, against about 25 m/s for the 25 points above 80 km).
    """

    traj = observeMeteor(METEOR, v_init_ht=HT_INIT)
    with contextlib.redirect_stdout(io.StringIO()):
        traj = traj.run()

    # The line through the points above the height, with equal weights
    weights = [1.0]*len(traj.observations)
    v_ht, _, v_ht_stddev = traj.calcAvgVelocityAboveHt(traj.observations, 1000*HT_INIT, weights)

    times, lengths = [], []
    for obs in traj.observations:
        used = (obs.ignore_list == 0) & (obs.model_ht >= 1000*HT_INIT)
        times.append(obs.time_data[used])
        lengths.append(obs.state_vect_dist[used])
    params, cov = scipy.optimize.curve_fit(lineFunc, np.concatenate(times), np.concatenate(lengths))

    assert abs(v_ht - params[0]) < 1e-6
    assert abs(v_ht_stddev/np.sqrt(cov[0, 0]) - 1) < 1e-6

    # The solver and the orbit take the standard deviation of the velocity they give
    assert traj.v_init_stddev > 10.0
    assert traj.orbit.v_init_stddev == traj.v_init_stddev
