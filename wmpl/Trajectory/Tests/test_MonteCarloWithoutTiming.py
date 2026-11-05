""" Tests of the Monte Carlo uncertainties in wmpl.Trajectory.Trajectory when the timing offsets are not
    estimated. A straight constant-velocity meteor is observed from three stations, with noise, and the
    gravity correction is off.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_MonteCarloWithoutTiming.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np

import matplotlib
matplotlib.use('Agg')

from wmpl.Trajectory.Tests.SyntheticTrajectory import observeStraightMeteor



def _simulateTrajectory(estimate_timing_vel, monte_carlo, seed=20261018):
    """ Make a trajectory object filled with noisy observations of a straight, constant-velocity meteor.

    Arguments:
        estimate_timing_vel: [bool/str] Passed to the Trajectory object.
        monte_carlo: [bool] Run the Monte Carlo uncertainties.

    Keyword arguments:
        seed: [int] Seed of the observation noise. 20261018 by default.

    Return:
        traj, radiant_eci:
            - traj: [Trajectory object] Trajectory object with observations, not yet solved.
            - radiant_eci: [ndarray] True radiant unit vector in ECI coordinates.
    """

    traj, _, radiant_eci = observeStraightMeteor(seed=seed, estimate_timing_vel=estimate_timing_vel, \
        monte_carlo=monte_carlo, mc_runs=5, mc_cores=1, mc_noise_std=1.0, calc_orbit=True)

    return traj, radiant_eci



def testTimingResidualExistsWithoutTimingEstimation():
    """ Check that the timing residual is computed with the given clocks. """

    traj, _ = _simulateTrajectory(estimate_timing_vel=False, monte_carlo=False)
    traj = traj.run()

    assert traj.timing_res is not None
    assert np.isfinite(traj.timing_res)



def testMonteCarloRunsWithoutTimingEstimation():
    """ Check that the Monte Carlo runs give finite uncertainties without timing estimation. """

    np.random.seed(1)
    traj, _ = _simulateTrajectory(estimate_timing_vel=False, monte_carlo=True)
    traj = traj.run()

    assert traj.uncertainties is not None
    assert np.isfinite(traj.uncertainties.ra_g) and (traj.uncertainties.ra_g > 0)



def testMonteCarloKeepsTheClocksFixed():
    """ Check that the Monte Carlo runs estimate the timing only if the original solution does. """

    np.random.seed(1)
    traj, _ = _simulateTrajectory(estimate_timing_vel=False, monte_carlo=True)
    traj = traj.run()

    assert not traj.estimate_timing_vel
    assert np.all(np.array(traj.time_diffs_final) == 0)



def testMonteCarloVelocityUncertaintyWithoutTimingEstimation():
    """ Check that the velocity uncertainty without timing estimation reflects the scatter of the lengths,
        and that the reported velocity is not perturbed.
    """

    # Solve without the Monte Carlo runs to get the nominal velocity
    traj, _ = _simulateTrajectory(estimate_timing_vel=False, monte_carlo=False)
    traj_nominal = traj.run()

    np.random.seed(1)
    traj, _ = _simulateTrajectory(estimate_timing_vel=False, monte_carlo=True)
    traj.mc_runs = 10
    traj = traj.run()

    # On this meteor the velocity scatters by about 15 m/s between noise realizations
    assert 5.0 < traj_nominal.v_init_stddev < 50.0
    assert 5.0 < traj.uncertainties.v_init < 50.0
    assert 5.0 < traj.uncertainties.v_g < 50.0

    # The noise is not applied to the velocity of the reported solution
    assert abs(traj.v_init - traj_nominal.v_init) < 1.0




if __name__ == "__main__":

    # Solve the simulated meteor and compare the radiant to the true one
    np.random.seed(1)
    traj, radiant_eci = _simulateTrajectory(estimate_timing_vel=False, monte_carlo=True)
    traj = traj.run()

    radiant_err = 3600*np.degrees(np.arccos(np.clip(np.dot(traj.radiant_eci_mini, radiant_eci), -1.0, 1.0)))

    print("Radiant error {:.2f} arcsec, timing offsets {}".format(radiant_err, traj.time_diffs_final))
