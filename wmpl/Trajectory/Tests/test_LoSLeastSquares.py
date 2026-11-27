""" Tests of the least squares and Huber signed-angle line-of-sight fit of wmpl.Trajectory.Trajectory
    (los_fit='huber' or 'least_squares'): residual equivalence, accuracy with gravity and rotating air,
    outlier and bad-station robustness, M-estimator sandwich covariance calibration, Monte Carlo covariance
    preservation, report output, and default selection.

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
    angleSumMeasurements2Line, leastSquaresLine, nelderMeadLine
from wmpl.Trajectory.Tests.SyntheticTrajectory import JDT_REF, STRAIGHT_JDT_REF, STRAIGHT_STATIONS, \
    GravityDragMeteor, observeMeteor, observeStraightMeteor, stationAzimAlt, straightMeteorPoints
from wmpl.Utils.Math import vectNorm
from wmpl.Utils.TrajConversions import eci2RaDec, raDec2AltAz


def _radiantError(traj, radiant_eci):
    """ Compute the angle between the solved and true radiant from the chord, precise for small angles.

    Arguments:
        traj: [Trajectory object] Solved trajectory.
        radiant_eci: [ndarray] True radiant unit vector in ECI coordinates.

    Return:
        [float] Angle (arcsec).
    """

    chord = np.linalg.norm(vectNorm(traj.radiant_eci_mini) - vectNorm(radiant_eci))

    return 3600.0*np.degrees(2.0*np.arcsin(min(chord/2.0, 1.0)))



def _moveAcrossTrack(traj, offsets):
    """ Move selected picks across the track of their station along its plane normal and reinitialize the
        observations.

    Arguments:
        traj: [Trajectory object] Trajectory object with observations, not yet solved.
        offsets: [dict] Angle offset per (station_index, pick_index) key (rad).
    """

    for k, obs in enumerate(traj.observations):

        azim, elev = np.array(obs.azim_data), np.array(obs.elev_data)
        for (station, i), offset in offsets.items():
            if station == k:
                ra, dec = eci2RaDec(vectNorm(obs.meas_eci_los[i] + offset*obs.plane_N))
                azim[i], elev[i] = raDec2AltAz(ra, dec, obs.JD_data[i], obs.lat, obs.lon)

        obs.meas1, obs.meas2 = azim, elev

    # Reinitialize observations so ECI lines of sight reflect the shifted picks
    obs_list = traj.observations
    traj.observations = []
    for obs in obs_list:
        traj.infillWithObs(obs)



def _observeUnequalNoise(seed, noise_arcsec=(10.0, 40.0, 10.0), los_fit='least_squares'):
    """ Observe the straight meteor of SyntheticTrajectory with station-dependent Gaussian noise.

    Arguments:
        seed: [int] Random seed.

    Keyword arguments:
        noise_arcsec: [tuple] Per-station angular noise standard deviation (arcsec). (10.0, 40.0, 10.0) by
            default.
        los_fit: [str] Line-of-sight fit method. 'least_squares' by default.

    Return:
        (traj, radiant_eci):
            - traj: [Trajectory object] Unsolved trajectory object with noisy observations.
            - radiant_eci: [ndarray] True radiant unit vector in ECI coordinates.
    """

    rng = np.random.default_rng(seed)
    times, points, _, radiant_eci = straightMeteorPoints()

    traj = Trajectory(STRAIGHT_JDT_REF, meastype=2, max_toffset=1.0, gravity_correction=False, \
        monte_carlo=False, show_plots=False, save_results=False, verbose=False, filter_picks=False, \
        calc_orbit=False, drag_fit_timing=False, estimate_timing_vel=False, los_fit=los_fit)

    for (station_id, s_lat, s_lon, ele), noise in zip(STRAIGHT_STATIONS, \
        np.radians(np.array(noise_arcsec)/3600.0)):

        azim, elev = stationAzimAlt(points, STRAIGHT_JDT_REF + times/86400.0, s_lat, s_lon, ele)
        azim = azim + rng.normal(0.0, noise, len(azim))/np.cos(elev)
        elev = elev + rng.normal(0.0, noise, len(elev))

        traj.infillTrajectory(azim, elev, times, np.radians(s_lat), np.radians(s_lon), ele, \
            station_id=station_id)

    return traj, radiant_eci



def testResidualsMatchAngleSumWithSign():
    """ Check that signed across-track residuals match the unsigned angles of angleSumMeasurements2Line to
        third order, both on a straight trajectory and with gravity and rotating air.
    """

    traj, begin, radiant_eci = observeStraightMeteor(noise=np.radians(30.0/3600.0), seed=1, \
        gravity_correction=False, calc_orbit=False, drag_fit_timing=False)

    # Evaluate on a line offset by a few hundred meters and tens of arcseconds
    state_vect = begin + np.array([180.0, -120.0, 60.0])
    radiant = vectNorm(radiant_eci + np.array([1e-4, -5e-5, 2e-5]))

    residuals = angleResidualsMeasurements2Line(traj.observations, state_vect, radiant)
    cost = angleSumMeasurements2Line(traj.observations, state_vect, radiant)

    assert len(residuals) == sum(len(obs.time_data) for obs in traj.observations)
    assert abs(np.mean(np.abs(residuals)) - cost) < 1e-5*cost

    # Verify residuals straddle both sides of the true line at every station
    true_residuals = angleResidualsMeasurements2Line(traj.observations, begin, radiant_eci)
    start = 0
    for obs in traj.observations:
        station = true_residuals[start:start + len(obs.time_data)]
        start += len(obs.time_data)
        assert 0.2 < np.mean(station > 0.0) < 0.8

    # Verify a large cross-track shift puts all residuals of a station on the same side
    shifted = angleResidualsMeasurements2Line(traj.observations, \
        state_vect + 2000.0*traj.observations[0].plane_N, radiant)
    station_a = shifted[:len(traj.observations[0].time_data)]
    assert np.all(np.sign(station_a) == np.sign(station_a[0]))

    # Verify equivalence also holds with gravity drop and rotating-air drag enabled
    meteor = GravityDragMeteor(13000.0, 210.0, 60.0, 50.0, 10.0, ht_beg=85000.0, air_rotation=True)
    traj_g = observeMeteor(meteor, noise=10.0, gravity_correction=True, air_rotation=True, \
        filter_picks=False, calc_orbit=False, estimate_timing_vel=False, los_fit='angles')
    traj_g = traj_g.run()

    res_g = angleResidualsMeasurements2Line(traj_g.observations, traj_g.state_vect_mini, \
        traj_g.radiant_eci_mini, gravity=True, gravity_vel=traj_g.gravity_vel, air_rotation=True)
    cost_g = angleSumMeasurements2Line(traj_g.observations, traj_g.state_vect_mini, \
        traj_g.radiant_eci_mini, gravity=True, gravity_vel=traj_g.gravity_vel, air_rotation=True)
    assert abs(np.mean(np.abs(res_g)) - cost_g) < 1e-5*cost_g



def testNoiseFreeLineIsFound():
    """ Check that least squares and Huber recover a noise-free straight line accurately. """

    for los_fit in ('least_squares', 'huber'):

        traj, begin, radiant_eci = observeStraightMeteor(noise=0.0, seed=1, los_fit=los_fit, \
            calc_orbit=False, drag_fit_timing=False)
        traj = traj.run()

        assert _radiantError(traj, radiant_eci) < 0.002

        # Check the state vector lies on the true line to better than 1 cm
        offset = traj.state_vect_mini - begin
        assert np.linalg.norm(offset - np.dot(offset, radiant_eci)*radiant_eci) < 0.01



def testSameLineAsAnglesWithGravityAndAirRotation():
    """ Check that without noise Huber and least squares recover the bent fireball radiant with drag-damped
        gravity drop and rotating air to within 2 arcsec, matching the legacy angles solver.
    """

    meteor = GravityDragMeteor(13000.0, 210.0, 60.0, 50.0, 10.0, ht_beg=85000.0, air_rotation=True)
    radiants = {}
    for los_fit in ('angles', 'least_squares', 'huber'):
        traj = observeMeteor(meteor, noise=0.0, gravity_correction=True, air_rotation=True, \
            filter_picks=False, calc_orbit=False, estimate_timing_vel=False, los_fit=los_fit)
        sol = traj.run()
        t_shift = (sol.jdt_ref - JDT_REF)*86400.0
        rad_true = -meteor.velocityAtTime(t_shift)[0]
        assert _radiantError(sol, rad_true) < 2.0
        radiants[los_fit] = vectNorm(sol.radiant_eci_mini)

    for method in ('least_squares', 'huber'):
        chord = np.linalg.norm(radiants['angles'] - radiants[method])
        assert 3600.0*np.degrees(2.0*np.arcsin(min(chord/2.0, 1.0))) < 2.0



def testHuberResistsOutliersAndKeepsCalibratedUncertainty():
    """ Check that Huber resists 10-sigma pick outliers better than least squares and keeps its formal
        M-estimator uncertainty calibrated instead of letting outliers blow it up.
    """

    noise = np.radians(30.0/3600.0)
    huber_errs = []
    lsq_errs = []
    huber_sigs = []

    for seed in range(1, 21):

        rng = np.random.default_rng(500 + seed)
        offsets = {}
        for station in range(3):
            for i in rng.choice(np.arange(2, 38), 3, replace=False):
                offsets[(station, int(i))] = 10.0*noise*rng.choice([-1, 1])

        for los_fit in ('least_squares', 'huber'):

            traj, _, radiant_eci = observeStraightMeteor(noise=noise, seed=seed, los_fit=los_fit, \
                filter_picks=False, estimate_timing_vel=False, calc_orbit=False, drag_fit_timing=False)
            _moveAcrossTrack(traj, offsets)
            sol = traj.run()

            err = _radiantError(sol, radiant_eci)
            if los_fit == 'least_squares':
                lsq_errs.append(err)
            else:
                huber_errs.append(err)
                rad_std, _ = sol.losFormalUncertainty()
                huber_sigs.append(3600.0*np.degrees(rad_std))

    # Huber must strongly outperform least squares on 10-sigma outliers
    rms_huber = np.sqrt(np.mean(np.square(huber_errs)))
    rms_lsq = np.sqrt(np.mean(np.square(lsq_errs)))
    assert rms_huber < 0.5*rms_lsq

    # Verify M-estimator sandwich covariance is not inflated by outliers
    cal_ratio = rms_huber/np.sqrt(np.mean(np.square(huber_sigs)))
    assert 0.55 < cal_ratio < 1.45



def testHuberSoftL1SeedResistsMiscalibratedStation():
    """ Check that the soft-L1 Stage-1 seed prevents a grossly miscalibrated fourth station from dragging the
        Huber fit away from three well-calibrated stations.
    """

    four_stations = list(STRAIGHT_STATIONS) + [('S4', 42.3, -81.9, 210.0)]
    times, points, _, radiant_eci = straightMeteorPoints()
    noise_good = np.radians(15.0/3600.0)
    bias_bad = np.radians(2000.0/3600.0)

    errors = {}
    for los_fit in ('least_squares', 'huber'):

        rng = np.random.default_rng(42)
        traj = Trajectory(STRAIGHT_JDT_REF, meastype=2, max_toffset=1.0, gravity_correction=False, \
            monte_carlo=False, show_plots=False, save_results=False, verbose=False, filter_picks=False, \
            calc_orbit=False, drag_fit_timing=False, estimate_timing_vel=False, los_fit=los_fit)

        for idx, (station_id, s_lat, s_lon, ele) in enumerate(four_stations):
            azim, elev = stationAzimAlt(points, STRAIGHT_JDT_REF + times/86400.0, s_lat, s_lon, ele)
            d_az = rng.normal(0.0, noise_good, len(azim)) + (bias_bad if idx == 3 else 0.0)
            d_el = rng.normal(0.0, noise_good, len(elev)) + (0.5*bias_bad if idx == 3 else 0.0)
            traj.infillTrajectory(azim + d_az/np.cos(elev), elev + d_el, times, np.radians(s_lat), \
                np.radians(s_lon), ele, station_id=station_id)

        errors[los_fit] = _radiantError(traj.run(), radiant_eci)

    assert errors['huber'] < 0.35*errors['least_squares']



def testFormalUncertaintyMatchesScatterUnequalNoise():
    """ Check that the formal sandwich standard deviation matches the empirical radiant scatter when stations
        have unequal noise and geometric weights are not inverse variances.
    """

    errors = []
    sigmas = []
    for seed in range(1, 31):

        traj, radiant_eci = _observeUnequalNoise(seed, los_fit='least_squares')
        traj = traj.run()

        radiant_stddev, position_stddev = traj.losFormalUncertainty()
        errors.append(_radiantError(traj, radiant_eci))
        sigmas.append(3600.0*np.degrees(radiant_stddev))

        assert position_stddev > 0.0

    ratio = np.sqrt(np.mean(np.square(errors)))/np.sqrt(np.mean(np.square(sigmas)))
    assert 0.6 < ratio < 1.4



def testFewerEvaluationsThanNelderMead():
    """ Check that the signed-angle least squares fit requires far fewer residual evaluations than
        Nelder-Mead.
    """

    module = sys.modules['wmpl.Trajectory.Trajectory']

    counts = {}
    for los_fit, func_name in [('angles', 'minimizeAngleCost'), ('least_squares', 'minimizeAngleResiduals')]:

        original = getattr(module, func_name)
        calls = []

        def _counted(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        setattr(module, func_name, _counted)
        try:
            traj, _, _ = observeStraightMeteor(noise=np.radians(10.0/3600.0), seed=1, los_fit=los_fit, \
                calc_orbit=False, drag_fit_timing=False)
            traj.run()
        finally:
            setattr(module, func_name, original)

        counts[los_fit] = len(calls)

    assert counts['least_squares'] < counts['angles']/5



def testReportHasFormalUncertainty():
    """ Check that saveReport prints formal line-of-sight uncertainties for huber and least_squares fits,
        and omits them for angles.
    """

    for los_fit in ('angles', 'least_squares', 'huber'):

        traj, _, _ = observeStraightMeteor(noise=np.radians(10.0/3600.0), seed=1, los_fit=los_fit, \
            calc_orbit=True, drag_fit_timing=False)
        traj = traj.run()

        report = traj.saveReport(None, None, save_results=False, verbose=False)
        has_formal = "Formal 1 sigma of the line-of-sight fit" in report
        assert has_formal == (los_fit != 'angles')



def testHuberIsDefaultAndLegacyFallbackWorks():
    """ Check that Huber is the default line-of-sight fit, including on legacy pickles without los_fit, and
        that angles can still be selected explicitly.
    """

    traj = Trajectory(2460600.25)
    assert traj.lineOfSightMethod() == (leastSquaresLine, {'loss': 'huber'})

    del traj.los_fit
    assert traj.lineOfSightMethod() == (leastSquaresLine, {'loss': 'huber'})

    traj_ang = Trajectory(2460600.25, los_fit='angles')
    assert traj_ang.lineOfSightMethod()[0] is nelderMeadLine
    assert traj_ang.losFormalUncertainty() == (None, None)



def testMonteCarloKeepsMeasuredFormalUncertainty():
    """ Check that monteCarloTrajectory preserves the formal covariance of the nominal measured lines rather
        than overwriting it with a noisier Monte Carlo draw.
    """

    module = sys.modules['wmpl.Trajectory.Trajectory']

    traj, _, _ = observeStraightMeteor(noise=np.radians(10.0/3600.0), seed=1, los_fit='huber', \
        calc_orbit=True, drag_fit_timing=False)
    traj = traj.run()

    # Build synthetic MC runs with doubled formal covariance as produced by noisy draws
    def _mockRuns(*args, **kwargs):
        runs = [copy.deepcopy(traj) for _ in range(2)]
        for run in runs:
            run.los_covariance = 2.0*traj.los_covariance
        return runs

    original = module.parallelComputeGenerator
    module.parallelComputeGenerator = _mockRuns
    try:
        traj_best, _ = module.monteCarloTrajectory(traj, mc_runs=2, plot_results=False, mc_cores=1)
    finally:
        module.parallelComputeGenerator = original

    assert traj_best is not traj
    assert np.array_equal(traj_best.los_covariance, traj.los_covariance)
    assert traj_best.losFormalUncertainty() == traj.losFormalUncertainty()
