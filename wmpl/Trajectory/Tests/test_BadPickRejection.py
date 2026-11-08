""" Tests of the rejection of bad picks in wmpl.Trajectory.Trajectory: the used picks whose angular residuals
    are beyond the mean plus N times the root mean square of the kept picks are rejected, again until no more
    are, so that outliers and ignored picks do not raise the limit.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_BadPickRejection.py -v
"""

from __future__ import print_function, division, absolute_import

import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')

from wmpl.Trajectory.Tests.SyntheticTrajectory import observeStraightMeteor
from wmpl.Utils.Math import vectNorm
from wmpl.Utils.TrajConversions import eci2RaDec, raDec2AltAz


# Noise of the measurements (rad), 20 arcsec
NOISE = np.radians(20.0/3600)


def _solve(offsets, ignored=(), seed=1):
    """ Solve the straight meteor of SyntheticTrajectory with picks of station A moved across the track.

    Arguments:
        offsets: [dict] Angle by which to move each pick of station A across the track (rad), by index.

    Keyword arguments:
        ignored: [list] Indices of the picks of station A which are ignored. None by default.
        seed: [int] Seed of the noise. 1 by default.

    Return:
        (traj, infill_count):
            - traj: [Trajectory object] Solved trajectory.
            - infill_count: [int] Number of times the observations were initialized again while solving.
    """

    traj, _, _ = observeStraightMeteor(noise=NOISE, seed=seed)

    # Move the picks along the normal of the plane of the station, i.e. across the track
    obs = traj.observations[0]
    azim, elev = np.array(obs.azim_data), np.array(obs.elev_data)
    for i, offset in offsets.items():
        ra, dec = eci2RaDec(vectNorm(obs.meas_eci_los[i] + offset*obs.plane_N))
        azim[i], elev[i] = raDec2AltAz(ra, dec, obs.JD_data[i], obs.lat, obs.lon)

    obs.meas1, obs.meas2 = azim, elev
    obs.ignore_list[list(ignored)] = 1

    obs_list = traj.observations
    traj.observations = []
    for obs in obs_list:
        traj.infillWithObs(obs)

    # Count how many times the solver initializes the observations again
    module = sys.modules['wmpl.Trajectory.Trajectory']
    original = module.Trajectory.infillWithObs
    calls = []

    def _infillWithObs(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    module.Trajectory.infillWithObs = _infillWithObs
    try:
        traj = traj.run()
    finally:
        module.Trajectory.infillWithObs = original

    return traj, len(calls)



def testLargeOutliersDoNotHideSmallerOnes():
    """ Check that a pick 8 sigma off across the track is rejected once two picks 25 sigma off are, which
        raise the limit of the first pass above it.
    """

    offsets = {8: 25*NOISE, 18: 25*NOISE, 22: 8*NOISE}
    traj, _ = _solve(offsets)

    obs = traj.observations[0]
    assert set(np.flatnonzero(obs.ignore_list)) == set(offsets)

    # The noise of the kept picks is that of the measurements
    assert obs.ang_res_std < 1.3*NOISE



def testIgnoredPicksDoNotRaiseTheLimit():
    """ Check that ignored picks with large residuals do not keep a pick 6 sigma off across the track. """

    traj, _ = _solve({5: 1000*NOISE, 15: 1000*NOISE, 25: 1000*NOISE, 20: 6*NOISE}, ignored=[5, 15, 25])

    assert set(np.flatnonzero(traj.observations[0].ignore_list)) == {5, 15, 20, 25}



def testIgnoredPicksAreNotRejectedAgain():
    """ Check that picks which were ignored from the start are not counted as rejected, which made the solver
        solve the trajectory once more.
    """

    traj, infill_count = _solve({5: 1000*NOISE, 15: 1000*NOISE, 25: 1000*NOISE}, ignored=[5, 15, 25], seed=2)

    assert set(np.flatnonzero(traj.observations[0].ignore_list)) == {5, 15, 25}
    assert all(not np.any(obs.ignore_list) for obs in traj.observations[1:])

    # The observations are only initialized again for the pass with the estimated timing
    assert infill_count == len(traj.observations)
