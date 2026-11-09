""" Tests of the least squares and Huber line-of-sight fits of the trajectory solver (los_fit), of the signed
    angles which they fit, and of the formal uncertainty of the line.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_LoSLeastSquares.py -v
"""

from __future__ import print_function, division, absolute_import

import copy
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')

from wmpl.Trajectory.Trajectory import Trajectory, angleResidualsMeasurements2Line, \
    angleSumMeasurements2Line, nelderMeadLine, leastSquaresLine
from wmpl.Trajectory.Tests.SyntheticTrajectory import STRAIGHT_JDT_REF, STRAIGHT_STATIONS, \
    observeStraightMeteor, stationAzimAlt, straightMeteorPoints
from wmpl.Utils.Math import vectNorm
from wmpl.Utils.TrajConversions import eci2RaDec, raDec2AltAz



def radiantError(traj, radiant_eci):
    """ Compute the angle between the solved and the true radiant from the chord, precise for small angles.

    Arguments:
        traj: [Trajectory object] Solved trajectory.
        radiant_eci: [ndarray] True radiant unit vector in ECI coordinates.

    Return:
        [float] Angle (arcsec).
    """

    chord = np.linalg.norm(vectNorm(traj.radiant_eci_mini) - radiant_eci)

    return 3600*np.degrees(2*np.arcsin(chord/2))



def moveAcrossTrack(traj, offsets):
    """ Move picks across the track of their station, along the normal of its plane, and initialize the
        observations again.

    Arguments:
        traj: [Trajectory object] Trajectory object with observations, not yet solved.
        offsets: [dict] Angle by which to move each pick (rad), by station index and pick index.
    """

    # Move the picks along the normal of the plane of their station
    for k, obs in enumerate(traj.observations):

        azim, elev = np.array(obs.azim_data), np.array(obs.elev_data)
        for (station, i), offset in offsets.items():
            if station == k:
                ra, dec = eci2RaDec(vectNorm(obs.meas_eci_los[i] + offset*obs.plane_N))
                azim[i], elev[i] = raDec2AltAz(ra, dec, obs.JD_data[i], obs.lat, obs.lon)

        obs.meas1, obs.meas2 = azim, elev

    # Initialize the observations again
    obs_list = traj.observations
    traj.observations = []
    for obs in obs_list:
        traj.infillWithObs(obs)



def testResidualsAreTheAnglesWithASign():
    """ Check that the residuals are the angles of the line-of-sight cost, to the third order, with a sign
        which changes across the line.
    """

    traj, begin, radiant_eci = observeStraightMeteor(noise=np.radians(30.0/3600), seed=1)

    # Take a line a few hundred meters and tens of arcsec off
    state_vect = begin + np.array([180.0, -120.0, 60.0])
    radiant = vectNorm(radiant_eci + np.array([1e-4, -5e-5, 2e-5]))

    residuals = angleResidualsMeasurements2Line(traj.observations, state_vect, radiant)
    cost = angleSumMeasurements2Line(traj.observations, state_vect, radiant)

    # Check the residuals against the angles of the cost
    assert len(residuals) == sum(len(obs.time_data) for obs in traj.observations)
    assert abs(np.mean(np.abs(residuals)) - cost) < 1e-5*cost

    # Check that the residuals of every station are on both sides of the true line
    true_residuals = angleResidualsMeasurements2Line(traj.observations, begin, radiant_eci)
    start = 0
    for obs in traj.observations:
        station = true_residuals[start:start + len(obs.time_data)]
        start += len(obs.time_data)
        assert 0.2 < np.mean(station > 0) < 0.8

    # Check that a line moved far across puts the residuals of a station on one side
    shifted = angleResidualsMeasurements2Line(traj.observations, \
        state_vect + 2000.0*traj.observations[0].plane_N, radiant)
    station_a = shifted[:len(traj.observations[0].time_data)]
    assert np.all(np.sign(station_a) == np.sign(station_a[0]))



def testNoiseFreeLineIsFound():
    """ Check that the least squares fit finds the line of a noise-free meteor. """

    traj, begin, radiant_eci = observeStraightMeteor(noise=0.0, seed=1, los_fit='least_squares')
    traj = traj.run()

    assert radiantError(traj, radiant_eci) < 0.002

    # Check that the state vector is on the true line
    offset = traj.state_vect_mini - begin
    assert np.linalg.norm(offset - np.dot(offset, radiant_eci)*radiant_eci) < 0.01



def testSameLineAsTheAnglesWithGravity():
    """ Check that without noise the least squares fit and the fit of the angles find the same line, also with
        the gravity drop.
    """

    # Solve the meteor with both fits
    radiants = []
    for los_fit in ['angles', 'least_squares']:

        traj, _, _ = observeStraightMeteor(noise=0.0, seed=1, duration=3.0, v_init=15000.0, elevation=30.0, \
            gravity_drop=True, free_fall=True, gravity_correction=True, los_fit=los_fit)
        radiants.append(vectNorm(traj.run().radiant_eci_mini))

    assert 3600*np.degrees(2*np.arcsin(np.linalg.norm(radiants[0] - radiants[1])/2)) < 0.01



def testHuberLossResistsOutliers():
    """ Check that the Huber loss keeps the radiant near the truth with picks 10 sigma off across the track,
        which pull the least squares fit away.
    """

    # Move three picks of every station by 10 sigma across the track
    noise = np.radians(30.0/3600)
    rng = np.random.default_rng(503)
    offsets = {}
    for station in range(3):
        for i in rng.choice(np.arange(2, 38), 3, replace=False):
            offsets[(station, int(i))] = 10*noise*rng.choice([-1, 1])

    # Solve the meteor with both losses
    errors = {}
    for los_fit in ['least_squares', 'huber']:

        traj, _, radiant_eci = observeStraightMeteor(noise=noise, seed=3, los_fit=los_fit, \
            filter_picks=False, estimate_timing_vel=False)
        moveAcrossTrack(traj, offsets)

        errors[los_fit] = radiantError(traj.run(), radiant_eci)

    assert errors['huber'] < errors['least_squares']/3



def observeUnequalNoise(seed, noise_arcsec=(10.0, 40.0, 10.0)):
    """ Observe the straight meteor of SyntheticTrajectory with a different noise at each station.

    Arguments:
        seed: [int] Seed of the noise.

    Keyword arguments:
        noise_arcsec: [tuple] Noise of each station (arcsec). 10, 40 and 10 by default.

    Return:
        (traj, radiant_eci):
            - traj: [Trajectory object] Trajectory object with the observations, not yet solved.
            - radiant_eci: [ndarray] True radiant unit vector in ECI coordinates.
    """

    rng = np.random.default_rng(seed)
    times, points, _, radiant_eci = straightMeteorPoints()

    traj = Trajectory(STRAIGHT_JDT_REF, meastype=2, max_toffset=1.0, gravity_correction=False, \
        monte_carlo=False, show_plots=False, save_results=False, verbose=False, filter_picks=False, \
        los_fit='least_squares')

    # Add the noise of each station to its directions
    noise_rad = np.radians(np.array(noise_arcsec)/3600)
    for (station_id, s_lat, s_lon, ele), noise in zip(STRAIGHT_STATIONS, noise_rad):

        azim, elev = stationAzimAlt(points, STRAIGHT_JDT_REF + times/86400.0, s_lat, s_lon, ele)
        azim = azim + rng.normal(0, noise, len(azim))/np.cos(elev)
        elev = elev + rng.normal(0, noise, len(elev))

        traj.infillTrajectory(azim, elev, times, np.radians(s_lat), np.radians(s_lon), ele, \
            station_id=station_id)

    return traj, radiant_eci



def testFormalUncertaintyMatchesTheScatter():
    """ Check that the formal standard deviation of the radiant is the scatter of the radiants over noise
        seeds, also with stations of unequal precision whose weights are not their inverse variances.
    """

    # Solve the meteor with 30 noise seeds
    errors, sigmas = [], []
    for seed in range(1, 31):

        traj, radiant_eci = observeUnequalNoise(seed)
        traj = traj.run()

        radiant_stddev, position_stddev = traj.losFormalUncertainty()
        errors.append(radiantError(traj, radiant_eci))
        sigmas.append(3600*np.degrees(radiant_stddev))

        assert position_stddev > 0

    # Compare the root mean square of the errors and of the formal deviations
    ratio = np.sqrt(np.mean(np.array(errors)**2))/np.sqrt(np.mean(np.array(sigmas)**2))
    assert 0.6 < ratio < 1.4



def testFewerEvaluationsThanNelderMead():
    """ Check that the least squares fit evaluates the lines of sight far fewer times than Nelder-Mead. """

    module = sys.modules['wmpl.Trajectory.Trajectory']

    # Count the evaluations of the cost of each fit
    counts = {}
    for los_fit, function in [('angles', 'minimizeAngleCost'), ('least_squares', 'minimizeAngleResiduals')]:

        original = getattr(module, function)
        calls = []

        def countCall(*args, **kwargs):
            """ Count the call and pass it to the original function. """

            calls.append(1)
            return original(*args, **kwargs)

        setattr(module, function, countCall)
        try:
            traj, _, _ = observeStraightMeteor(noise=np.radians(10.0/3600), seed=1, los_fit=los_fit)
            traj.run()
        finally:
            setattr(module, function, original)

        counts[los_fit] = len(calls)

    assert counts['least_squares'] < counts['angles']/5



def testReportHasTheFormalUncertainty():
    """ Check that the report gives the formal uncertainty of the least squares fit, not of the angles. """

    for los_fit in ['angles', 'least_squares']:

        traj, _, _ = observeStraightMeteor(noise=np.radians(10.0/3600), seed=1, los_fit=los_fit)
        traj = traj.run()

        report = traj.saveReport(None, None, save_results=False, verbose=False)
        assert ("Formal 1 sigma of the line-of-sight fit" in report) == (los_fit == 'least_squares')



def testHuberIsTheDefault():
    """ Check that the line-of-sight fit is the Huber fit by default, also of trajectories pickled without the
        option, and that the angles are fitted when asked for.
    """

    # Check the default
    traj = Trajectory(2460600.25)
    assert traj.lineOfSightMethod() == (leastSquaresLine, {'loss': 'huber'})

    # Check a trajectory pickled without the option
    del traj.los_fit
    assert traj.lineOfSightMethod() == (leastSquaresLine, {'loss': 'huber'})

    # Check the fit of the angles
    traj = Trajectory(2460600.25, los_fit='angles')
    assert traj.lineOfSightMethod()[0] is nelderMeadLine
    assert traj.losFormalUncertainty() == (None, None)



def testTheMonteCarloKeepsTheFormalUncertaintyOfTheMeasuredLines():
    """ Check that the solution returned by the Monte Carlo, one of its runs, has the formal uncertainty of
        the fit of the measured lines of sight, not of the run, whose lines of sight had noise added.
    """

    module = sys.modules['wmpl.Trajectory.Trajectory']

    traj, _, _ = observeStraightMeteor(noise=np.radians(10.0/3600), seed=1, los_fit='least_squares')
    traj = traj.run()

    def makeRuns(*args, **kwargs):
        """ Make two runs of the same line with the covariance of noisier lines of sight. """

        runs = [copy.deepcopy(traj) for _ in range(2)]
        for run in runs:
            run.los_covariance = 2*traj.los_covariance
        return runs

    # Run the Monte Carlo with the runs made here
    original = module.parallelComputeGenerator
    module.parallelComputeGenerator = makeRuns
    try:
        traj_best, _ = module.monteCarloTrajectory(traj, mc_runs=2, plot_results=False, mc_cores=1)
    finally:
        module.parallelComputeGenerator = original

    # Check that the solution keeps the covariance of the measured lines of sight
    assert traj_best is not traj
    assert np.array_equal(traj_best.los_covariance, traj.los_covariance)
    assert traj_best.losFormalUncertainty() == traj.losFormalUncertainty()



if __name__ == "__main__":

    testResidualsAreTheAnglesWithASign()
    testNoiseFreeLineIsFound()
    testSameLineAsTheAnglesWithGravity()
    testHuberLossResistsOutliers()
    testFormalUncertaintyMatchesTheScatter()
    testFewerEvaluationsThanNelderMead()
    testReportHasTheFormalUncertainty()
    testHuberIsTheDefault()
    testTheMonteCarloKeepsTheFormalUncertaintyOfTheMeasuredLines()

    print("All tests passed.")
