""" Tests of the drag-coupled trajectory solver, pre-atmospheric air-column velocity correction, light-curve
    drag overlays and gap-aware DragFit handoff integrated into wmpl.Trajectory.Trajectory.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_TrajectoryDragFit.py -v
"""

from __future__ import print_function, division, absolute_import

import contextlib
import copy
import io
import os
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from wmpl.Trajectory.Tests.SyntheticTrajectory import DragMeteor, GravityDragMeteor, JDT_REF, STATIONS, \
    observeMeteor, timingOffsets
from wmpl.Trajectory.Trajectory import DragTimingFit, Trajectory, _MCTrajSolve, _updateDragMCResults, \
    checkMCTrajectories, computeStationCoverage, findObservedWindows, plotDragTrajectory
from wmpl.Utils.DragFit import DragFit
from wmpl.Utils.Math import angleBetweenVectors
from wmpl.Utils.Physics import trajectoryLightCurve
from wmpl.Utils.Pickling import loadPickle, savePickle
from wmpl.Utils.PlotCelestial import CelestialPlot
from wmpl.Utils.SampleTrajectoryPositions import sampleTrajectory
from wmpl.Utils.TrajConversions import altAz2RADec_vect, geo2Cartesian


# Four stations split into two completely non-overlapping groups with a 20% middle gap
TWO_GROUP_GAPPED_STATIONS = [
    ('A', 45.6, 14.6, 200.0, 0.00, 0.40),
    ('B', 44.5, 15.3, 400.0, 0.05, 0.40),
    ('C', 45.1, 15.9, 150.0, 0.60, 1.00),
    ('D', 45.3, 15.5, 300.0, 0.60, 0.95),
]

# Two stations with zero overlap and a 20% unobserved gap in between
TWO_STATION_GAPPED = [
    ('A', 45.6, 14.6, 200.0, 0.00, 0.40),
    ('B', 44.5, 15.3, 400.0, 0.60, 1.00),
]



def testFindObservedWindowsAndStationCoverage():
    """ Check that observed windows, single-station no-overlap spans and unobserved gaps are separated. """

    len_list = [np.linspace(0.0, 25000.0, 30), np.linspace(10000.0, 35000.0, 30), \
        np.linspace(50000.0, 75000.0, 30)]

    windows, gaps = findObservedWindows(len_list)
    assert len(windows) == 2
    assert len(gaps) == 1
    assert abs(gaps[0][0] - 35000.0) < 1.0
    assert abs(gaps[0][1] - 50000.0) < 1.0

    len_grid = np.linspace(0.0, 75000.0, 301)
    n_stat, ov_spans, single_spans, gap_spans = computeStationCoverage(len_grid, len_list)

    assert len(ov_spans) == 1
    assert len(single_spans) == 3
    assert len(gap_spans) == 1
    assert np.all(n_stat[(len_grid >= 11000.0) & (len_grid <= 24000.0)] == 2)
    assert np.all(n_stat[(len_grid >= 51000.0) & (len_grid <= 74000.0)] == 1)
    assert np.all(n_stat[(len_grid >= 36000.0) & (len_grid <= 49000.0)] == 0)



def testDragTimingJacobianMatchesFiniteDifferences():
    """ Check that the analytical Jacobian of DragTimingFit matches central finite differences for both
        drag parameters and free station timing offsets.
    """

    meteor = DragMeteor(18000.0, 120.0, 45.0, 0.01, 0.05, 2.7, 6.0)
    dur = meteor.t_vis_end - meteor.t_vis_beg

    t_a = np.arange(0.0, 0.45*dur, 0.04)
    l_a = np.interp(t_a + meteor.t_vis_beg, meteor.time, meteor.length)

    t_b = np.arange(0.55*dur, dur, 0.04)
    l_b = np.interp(t_b + meteor.t_vis_beg, meteor.time, meteor.length)

    for air_rotation in (False, True):
        fit = DragTimingFit([t_a, t_b - 1.2], [l_a, l_b], [1.0, 0.8], [False, True], meteor.top_eci, \
            meteor.radiant_eci, JDT_REF, air_rotation=air_rotation)

        params = fit.params + 0.005
        jac = fit._jacobian(params, 2.0)

        jac_num = np.zeros_like(jac)
        for i in range(len(params)):
            step = 1e-5*max(abs(params[i]), 1.0)
            p_up = params.copy()
            p_dn = params.copy()
            p_up[i] += step
            p_dn[i] -= step
            jac_num[:, i] = (fit._residuals(p_up, 2.0) - fit._residuals(p_dn, 2.0))/(2.0*step)

        col_err = np.max(np.abs(jac - jac_num), axis=0)/np.maximum(np.max(np.abs(jac_num), axis=0), 1e-12)

        assert np.median(col_err) < 1e-4
        assert np.max(col_err) < 0.02



def testOverlappingMeteorMatchesTruthWithGravityAndRotation():
    """ Check that default Trajectory.run() recovers radiant, velocity and clock offsets on a decelerating
        meteor with gravity drop and rotating air.
    """

    meteor = GravityDragMeteor(20000.0, 120.0, 45.0, 1.0, 4.0, air_rotation=True)
    clock_offsets = {'B': 1.35, 'C': -0.82}

    traj = observeMeteor(meteor, noise=10.0, stations=STATIONS, clock_offsets=clock_offsets, \
        gravity_correction=True, max_toffset=5.0)

    with contextlib.redirect_stdout(io.StringIO()):
        traj = traj.run()

    assert traj is not None
    assert traj.drag_timing_fit is not None

    offs = timingOffsets(traj)
    assert abs(offs['B'] + 1.35) < 0.015
    assert abs(offs['C'] + (-0.82)) < 0.015

    rad_err_arcsec = np.degrees(angleBetweenVectors(traj.radiant_eci_mini, meteor.radiant_eci))*3600.0
    assert rad_err_arcsec < 60.0
    assert abs(traj.v_init - 20000.0) < 250.0



def testZeroOverlapTwoStationsRecoversClockAndTrajectory():
    """ Two stations with zero overlap and a 20% unobserved middle gap fail in legacy pairwise mode
        (drag_fit_timing=False), but default drag-coupled Trajectory.run() recovers clock, radiant and speed.
    """

    meteor = DragMeteor(18000.0, 120.0, 45.0, 0.008, 0.02, 2.7, 6.0, ground_drag=True)
    clock_offsets = {'B': 2.15}

    # Verify legacy pairwise mode fails on this zero-overlap configuration
    traj_legacy = observeMeteor(meteor, noise=10.0, stations=TWO_STATION_GAPPED, \
        clock_offsets=clock_offsets, gravity_correction=False, max_toffset=5.0, drag_fit_timing=False)
    with contextlib.redirect_stdout(io.StringIO()):
        res_legacy = traj_legacy.run()
    assert res_legacy is None

    # Solve the same observations with the default drag-coupled solver
    traj_drag = observeMeteor(meteor, noise=10.0, stations=TWO_STATION_GAPPED, clock_offsets=clock_offsets, \
        gravity_correction=False, max_toffset=5.0)
    with contextlib.redirect_stdout(io.StringIO()):
        traj_drag = traj_drag.run()

    assert traj_drag is not None
    assert len(traj_drag.drag_timing_fit.gaps) == 1
    assert len(traj_drag.drag_timing_fit.single_station_spans) == 2
    assert len(traj_drag.drag_timing_fit.gap_uncertainties) == 1

    offs = timingOffsets(traj_drag)
    assert abs(offs['B'] + 2.15) < 0.05

    rad_err_arcsec = np.degrees(angleBetweenVectors(traj_drag.radiant_eci_mini, meteor.radiant_eci))*3600.0
    assert rad_err_arcsec < 120.0
    assert abs(traj_drag.v_init - 18000.0) < 350.0



def testDisconnectedGroupsWithMonteCarloAndPlots(tmp_path):
    """ Four stations in two disconnected groups with a 20% gap and clock offsets are aligned by Trajectory
        with Monte Carlo uncertainties enabled, and velocity, light-curve and drag-fit plots render cleanly.
    """

    meteor = GravityDragMeteor(22000.0, 60.0, 35.0, 5.0, 5.0, air_rotation=True)
    clock_offsets = {'B': -0.40, 'C': 1.80, 'D': -1.25}

    traj_base = observeMeteor(meteor, noise=15.0, stations=TWO_GROUP_GAPPED_STATIONS, \
        clock_offsets=clock_offsets, gravity_correction=True, max_toffset=5.0)

    # Copy observations into Trajectory with synthetic apparent magnitudes and Monte Carlo enabled
    traj = Trajectory(traj_base.jdt_ref, output_dir=str(tmp_path), meastype=2, max_toffset=5.0, \
        gravity_correction=True, monte_carlo=True, mc_runs=8, mc_cores=2, show_plots=False, \
        save_results=True, verbose=False, air_rotation=True)
    for obs in traj_base.observations:
        mags = -2.0 - 4.0*np.sin(np.linspace(0.1, np.pi - 0.1, len(obs.time_data)))
        traj.infillTrajectory(obs.azim_data.copy(), obs.elev_data.copy(), obs.time_data.copy(), \
            obs.lat, obs.lon, obs.ele, station_id=obs.station_id, magnitudes=mags)

    with contextlib.redirect_stdout(io.StringIO()):
        traj = traj.run()

    assert traj is not None
    assert len(traj.drag_timing_fit.gaps) == 1
    assert traj.uncertainties is not None
    assert traj.uncertainties.v_init > 0.0
    assert traj.drag_timing_fit.mc_vel_std is not None
    assert traj.drag_timing_fit.mc_k_lo is not None
    assert traj.drag_timing_fit.mc_dlog_k_std is not None
    assert traj.drag_timing_fit.mc_time_diffs_std is not None
    assert np.all(traj.drag_timing_fit.mc_time_diffs_std < 0.05)

    offs = timingOffsets(traj)
    assert abs(offs['B'] + (-0.40)) < 0.02
    assert abs(offs['C'] + 1.80) < 0.06
    assert abs(offs['D'] + (-1.25)) < 0.06

    rad_err_arcsec = np.degrees(angleBetweenVectors(traj.radiant_eci_mini, meteor.radiant_eci))*3600.0
    assert rad_err_arcsec < 90.0

    # Verify velocity, light-curve and drag-fit diagnostic plots were saved cleanly
    assert os.path.isfile(os.path.join(str(tmp_path), traj.file_name + '_velocities.png'))
    assert os.path.isfile(os.path.join(str(tmp_path), traj.file_name + '_abs_mag.png'))
    assert os.path.isfile(os.path.join(str(tmp_path), traj.file_name + '_abs_mag_ht.png'))
    assert os.path.isfile(os.path.join(str(tmp_path), traj.file_name + '_drag_fit.png'))

    # Verify plotDragTrajectory keeps 4 clean panels (no gap text inserts or twin magnitude axis) and
    #   trajectoryLightCurve handles partially missing (None) photometry
    obs0 = traj.observations[0]
    obs0.absolute_magnitudes = np.array([None if (k%3 == 0) else -2.0 for k in range(obs0.kmeas)], \
        dtype=object)
    t_lc, mag_lc, _ = trajectoryLightCurve(traj)
    assert (t_lc is not None) and (len(mag_lc) > 0)

    fig_drag = plotDragTrajectory(traj, show_plots=False, save_results=False, ret_fig=True)
    assert fig_drag is not None
    assert len(fig_drag.axes) == 4
    assert len(fig_drag.axes[0].texts) == 0
    plt.close(fig_drag)



def testStationBeyondMaxToffsetIsIgnored():
    """ A station whose clock is off by more than max_toffset is ignored while the remaining stations stay
        accurately aligned.
    """

    meteor = DragMeteor(20000.0, 45.0, 40.0, 1e-3, 3e-3, 2.0, 5.0)
    four_stations = STATIONS + [('D', 45.3, 15.5, 300.0, 0.0, 1.0)]

    traj = observeMeteor(meteor, noise=0.0, stations=four_stations, clock_offsets={'D': 25.0}, \
        max_toffset=5.0, gravity_correction=False)

    with contextlib.redirect_stdout(io.StringIO()):
        traj = traj.run()

    assert traj is not None
    ignored = [obs.station_id for obs in traj.observations if obs.ignore_station]
    assert ignored == ['D']

    offs = timingOffsets(traj)
    assert abs(offs['B']) < 1e-2
    assert abs(offs['C']) < 1e-2



def testPickleRoundTripAndGapAwareHandoff(tmp_path):
    """ Check that a drag-coupled trajectory saved to pickle loads cleanly as a standard Trajectory, and
        hands off unobserved gaps cleanly to DragFit and SampleTrajectoryPositions.
    """

    meteor = DragMeteor(18000.0, 120.0, 45.0, 0.008, 0.02, 2.7, 6.0)
    traj = observeMeteor(meteor, noise=5.0, stations=TWO_STATION_GAPPED, clock_offsets={'B': 1.2}, \
        gravity_correction=False, max_toffset=5.0)

    with contextlib.redirect_stdout(io.StringIO()):
        traj = traj.run()

    out_dir = str(tmp_path)
    savePickle(traj, out_dir, 'test_drag_traj.pickle')
    loaded = loadPickle(out_dir, 'test_drag_traj.pickle')

    assert isinstance(loaded, Trajectory)
    assert loaded.drag_timing_fit is not None
    assert len(loaded.drag_timing_fit.gaps) == 1
    assert abs(loaded.v_init - traj.v_init) < 1e-6

    # Gap-aware DragFit detects the gap and assigns a dedicated gap ramp
    df_handoff = DragFit(loaded)
    assert len(df_handoff.gaps) == 1
    assert int(np.sum(df_handoff.is_gap_ramp)) == 1

    # SampleTrajectoryPositions flags samples inside the unobserved gap
    samples = sampleTrajectory(loaded, -1, -1, 1000.0, show_plots=False)
    assert len(samples.ht) > 10
    assert any(samples.in_gap)
    assert not all(samples.in_gap)



def testMaxK0PreventsHighAltitudeNoiseOverfit():
    """ Check that max_k0 bounds initial K_0 and prevents unphysical hyperbolic speed blow-up on short noisy
        high-altitude faint meteors.
    """

    # Short 0.2 s, 63 km/s meteor from 111 km to 105 km with a noisy first point mimicking faint camera bias
    jd = 2460000.5
    state_vect = np.array(geo2Cartesian(np.radians(45.0), np.radians(15.0), 111000.0, jd))
    radiant_eci = np.array([0.3, 0.4, 0.8660254])
    radiant_eci = radiant_eci/np.linalg.norm(radiant_eci)

    t1 = np.linspace(0.0, 0.16, 5)
    t2 = np.linspace(0.02, 0.18, 5)
    l1 = 63000.0*t1 + np.array([-450.0, 120.0, 80.0, -90.0, -50.0])
    l2 = 63000.0*t2 + np.array([-150.0, 60.0, 40.0, -60.0, -80.0])

    fit_capped = DragTimingFit([t1, t2 + 0.3], [l1, l2], [1.0, 1.0], [False, True], state_vect, radiant_eci, \
        jd, max_toffset=5.0, max_k0=10.0)

    assert np.exp(fit_capped.drag_params[2]) <= 10.0 + 1e-6
    assert abs(fit_capped.vel_grid[0] - 63000.0) < 3500.0
    assert abs(fit_capped.time_offsets[1] + 0.3) < 0.01



def testArtificialCloudGapMatchesFullSolution():
    """ Removing a 25% middle cloud gap from all stations still matches the full-data solution in radiant,
        initial velocity, end velocity and station clock offsets.
    """

    meteor = DragMeteor(18000.0, 120.0, 45.0, 0.008, 0.02, 2.7, 6.0)
    shifts = {'B': 0.9, 'C': -1.4}

    traj_full = observeMeteor(meteor, noise=6.0, stations=STATIONS, clock_offsets=shifts, \
        gravity_correction=False, max_toffset=5.0)
    traj_gap = observeMeteor(meteor, noise=6.0, stations=STATIONS, clock_offsets=shifts, \
        gravity_correction=False, max_toffset=5.0)

    # Cut out [35%, 60%] of the flight from every station to simulate a shared cloud deck
    t_all = np.concatenate([obs.time_data - shifts.get(obs.station_id, 0.0) \
        for obs in traj_gap.observations])
    t_beg, t_end = float(np.min(t_all)), float(np.max(t_all))
    t_cut_lo = t_beg + 0.35*(t_end - t_beg)
    t_cut_hi = t_beg + 0.60*(t_end - t_beg)
    obs_gap_src = list(traj_gap.observations)
    traj_gap.observations = []
    for obs in obs_gap_src:
        sh = shifts.get(obs.station_id, 0.0)
        t_true = obs.time_data - sh
        keep = (t_true <= t_cut_lo) | (t_true >= t_cut_hi)
        mags = np.asarray(obs.magnitudes)[keep] if obs.magnitudes is not None else None
        traj_gap.infillTrajectory(obs.azim_data[keep].copy(), obs.elev_data[keep].copy(), \
            obs.time_data[keep].copy(), obs.lat, obs.lon, obs.ele, station_id=obs.station_id, \
            magnitudes=mags)

    with contextlib.redirect_stdout(io.StringIO()):
        sol_full = traj_full.run()
        sol_gap = traj_gap.run()

    assert (sol_full is not None) and (sol_gap is not None)
    assert len(sol_gap.drag_timing_fit.gaps) == 1

    rad_diff_arcsec = np.degrees(angleBetweenVectors(sol_gap.radiant_eci_mini, \
        sol_full.radiant_eci_mini))*3600.0
    assert rad_diff_arcsec < 30.0
    assert abs(sol_gap.v_init - sol_full.v_init)/sol_full.v_init < 0.01

    v_end_full = sol_full.drag_timing_fit.vel_grid[-1]
    v_end_gap = sol_gap.drag_timing_fit.vel_grid[-1]
    assert abs(v_end_gap - v_end_full)/v_end_full < 0.02

    offs = timingOffsets(sol_gap)
    assert abs(offs['B'] + 0.9) < 0.015
    assert abs(offs['C'] - 1.4) < 0.015



def testSuppliedFixedOffsetsPreservedInTimeDiffsFinal():
    """ Supplied fixed clock offsets from fixed_times must be preserved in time_diffs_final without being
        double-applied to the observations.
    """

    meteor = DragMeteor(20000.0, 45.0, 40.0, 1e-3, 3e-3, 2.0, 5.0)
    traj_src = observeMeteor(meteor, noise=0.0, stations=STATIONS, clock_offsets={'A': -0.35, 'B': 0.80}, \
        gravity_correction=False, max_toffset=5.0)

    # Supply station A's exact +0.35 s correction via fixed_times while leaving B and C free
    traj = Trajectory(traj_src.jdt_ref, meastype=2, max_toffset=5.0, gravity_correction=False, \
        monte_carlo=False, show_plots=False, save_results=False, verbose=False, fixed_times='A:0.35')
    for obs in traj_src.observations:
        traj.infillTrajectory(obs.azim_data.copy(), obs.elev_data.copy(), obs.time_data.copy(), \
            obs.lat, obs.lon, obs.ele, station_id=obs.station_id)

    with contextlib.redirect_stdout(io.StringIO()):
        traj = traj.run()

    assert traj is not None
    raw_offs = {obs.station_id: float(dt) for obs, dt in zip(traj.observations, traj.time_diffs_final)}
    assert abs(raw_offs['A'] - 0.35) < 1e-6
    assert abs(raw_offs['B'] + 0.80) < 0.015
    assert abs(raw_offs['C'] - 0.0) < 0.015
    assert abs(traj.stations_time_dict_copy['A'] - 0.35) < 1e-6



def testReportSummaryPreAtmAndTerminalMassGating(tmp_path):
    """ Verify that the drag fit summary is written into *_report.txt, pre-atmospheric air-column correction
        updates orbit.v_inf on constrained fireballs (and respects pre_atm_correction=False), terminal
        dynamic mass and sigma_eff are gated by combined fit+MC uncertainty, and high-altitude faint meteors
        suppress both terminal dynamic mass and pre-atmospheric orbit correction.
    """

    # 1. Late-detected fireball (starts ~58 km, penetrates below 30 km): measurable pre-atm correction
    meteor_fall = DragMeteor(14500.0, 120.0, 45.0, 0.002, 0.004, 5.8, 11.0)
    traj_fall = observeMeteor(meteor_fall, noise=4.0, stations=STATIONS, gravity_correction=False, \
        max_toffset=5.0, calc_orbit=True)
    traj_fall.output_dir = str(tmp_path)
    traj_fall.save_results = True
    traj_fall.show_plots = False

    with contextlib.redirect_stdout(io.StringIO()):
        traj_fall = traj_fall.run()

    assert traj_fall is not None
    tm_fall = traj_fall.drag_timing_fit.terminalDynamicMass()
    assert tm_fall['mass'] is not None
    assert tm_fall['mass_lo'] <= tm_fall['mass'] <= tm_fall['mass_hi']
    assert 0.05 < tm_fall['mass'] < 50.0

    # Pre-atmospheric air-column correction is applied to orbit.v_inf and improves agreement with 14.5 km/s
    assert traj_fall.pre_atm_info is not None
    assert traj_fall.pre_atm_info['applied']
    assert traj_fall.dv_pre_atm > 10.0
    assert abs(traj_fall.orbit.v_inf - (traj_fall.v_init + traj_fall.dv_pre_atm)) < 1e-6
    assert abs(traj_fall.orbit.v_inf - 14500.0) < abs(traj_fall.v_init - 14500.0)

    # Recomputing the orbit from the saved pickle via Orbit CLI preserves dv_pre_atm when -v is not given,
    #   and resets it when -v overrides the initial velocity
    pkl_path = os.path.join(str(tmp_path), traj_fall.file_name + '_trajectory.pickle')
    out_orb = subprocess.check_output([sys.executable, '-m', 'wmpl.Trajectory.Orbit', pkl_path], \
        text=True, stderr=subprocess.STDOUT)
    assert 'pre-atm drag corr' in out_orb
    assert "{:.5f} km/s".format(traj_fall.orbit.v_inf/1000.0) in out_orb

    out_orb_v = subprocess.check_output([sys.executable, '-m', 'wmpl.Trajectory.Orbit', pkl_path, \
        '-v', '14.5'], text=True, stderr=subprocess.STDOUT)
    assert 'pre-atm drag corr' not in out_orb_v
    assert "14.50000 km/s" in out_orb_v

    # Disabling pre_atm_correction leaves orbit.v_inf equal to v_init
    traj_no_pre = observeMeteor(meteor_fall, noise=4.0, stations=STATIONS, gravity_correction=False, \
        max_toffset=5.0, calc_orbit=True, pre_atm_correction=False)
    with contextlib.redirect_stdout(io.StringIO()):
        traj_no_pre = traj_no_pre.run()
    assert traj_no_pre.dv_pre_atm == 0.0
    assert abs(traj_no_pre.orbit.v_inf - traj_no_pre.v_init) < 1e-6

    report_path = os.path.join(str(tmp_path), traj_fall.file_name + '_report.txt')
    assert os.path.isfile(report_path)
    report_txt = open(report_path, 'r').read()
    assert 'Drag fit\n--------' in report_txt
    assert 'Pre-atmospheric air-column drag correction:' in report_txt
    assert 'Air column      =' in report_txt
    assert 'Velocity corr.  =' in report_txt
    assert 'applied to orbit' in report_txt
    assert 'Peak dyn. pres. =' in report_txt
    assert 'Eff. ablation   =' in report_txt
    assert 'Terminal mass   =' in report_txt
    assert 'not estimated' not in report_txt
    assert 'single-body m end/beg' not in report_txt

    # Meteorite-dropping fireball (h_end < 35 km, m > 10 g) reports full dark-flight initial state
    df_state = traj_fall.drag_timing_fit.terminalDarkFlightState()
    assert df_state is not None
    assert df_state['ht'] < 35000.0
    assert df_state['mass'] > 0.01
    assert 0.0 <= np.degrees(df_state['azim']) < 360.0
    assert 0.0 < np.degrees(df_state['elev']) < 90.0
    assert 1000.0 < df_state['vel_gnd'] < 15000.0
    assert 'Dark-flight initial state (terminal height < 35 km, terminal mass > 10 g):' in report_txt
    assert 'Radiant azim    =' in report_txt
    assert 'Radiant alt     =' in report_txt
    assert 'Terminal speed  =' in report_txt

    peak_p = traj_fall.drag_timing_fit.peakDynamicPressure()
    assert peak_p['dyn_pres'] > 1e5
    assert 15000.0 < peak_p['ht'] < 55000.0

    abl = traj_fall.drag_timing_fit.effectiveAblationCoeff()
    assert abl['sigma_eff'] is not None
    assert 0.001 < abl['sigma_eff'] < 0.2

    # Moderate MC spread on ln(K_end/K_beg) must enlarge sigma_eff_std in quadrature
    df_fall = traj_fall.drag_timing_fit
    df_fall.mc_dlog_k_std = 0.25
    abl_mc = df_fall.effectiveAblationCoeff()
    assert abl_mc['sigma_eff_std'] is not None
    assert abl_mc['sigma_eff_std'] > (abl['sigma_eff_std'] or 0.0)
    df_fall.mc_dlog_k_std = None

    # Wide MC ensemble spread must suppress terminal dynamic mass, sigma_eff and pre-atm correction
    k_end = float(np.exp(df_fall.log_k_grid[-1]))
    df_fall.mc_k_lo = np.full_like(df_fall.len_grid, k_end*np.exp(-2.0))
    df_fall.mc_k_hi = np.full_like(df_fall.len_grid, k_end*np.exp(2.0))
    tm_wide_mc = df_fall.terminalDynamicMass()
    assert tm_wide_mc['mass'] is None
    assert 'poorly constrained' in tm_wide_mc['reason']
    abl_wide_mc = df_fall.effectiveAblationCoeff()
    assert abl_wide_mc['sigma_eff'] is None
    assert 'poorly constrained' in abl_wide_mc['reason']
    pre_wide_mc = df_fall.preAtmosphericVelocity()
    assert not pre_wide_mc['applied']
    assert 'poorly constrained' in pre_wide_mc['reason']

    # Reapplying the pre-atm gate after attaching wide MC uncertainty must revoke dv_pre_atm on the orbit
    _updateDragMCResults(traj_fall, traj_fall, [traj_fall])
    assert traj_fall.dv_pre_atm == 0.0
    assert abs(traj_fall.orbit.v_inf - traj_fall.v_init) < 1e-6
    assert not traj_fall.pre_atm_info['applied']

    # 2. Small high-altitude meteor (100 -> 88 km, 35 km/s): terminal mass and pre-atm must NOT be applied
    jd = 2460000.5
    state_vect = np.array(geo2Cartesian(np.radians(45.0), np.radians(15.0), 100000.0, jd))
    rad_eci = np.array([0.3, 0.4, 0.8660254])
    rad_eci = rad_eci/np.linalg.norm(rad_eci)
    t_small = np.linspace(0.0, 0.4, 15)
    l_small = 35000.0*t_small
    fit_small = DragTimingFit([t_small[:10], t_small[5:]], [l_small[:10], l_small[5:]], [1.0, 1.0], \
        [False, True], state_vect, rad_eci, jd)
    tm_small = fit_small.terminalDynamicMass()
    assert tm_small['mass'] is None
    assert 'non-meteorite-dropping' in tm_small['reason']
    pre_small = fit_small.preAtmosphericVelocity()
    assert not pre_small['applied']

    # Verify saveReport writes the gated-out status cleanly and omits dark-flight initial state
    assert fit_small.terminalDarkFlightState() is None
    traj_fall.drag_timing_fit = fit_small
    traj_fall.pre_atm_info = pre_small
    report_small = traj_fall.saveReport(str(tmp_path), 'small_report.txt', verbose=False, save_results=False)
    assert 'Terminal mass   = not estimated' in report_small
    assert 'Dark-flight initial state' not in report_small

    # 3. Verify Monte Carlo runs on meastype=1 (RA/Dec) and meastype=4 (Met.py) convert noise-free
    #   observations to alt/az cleanly and produce finite, realistic radiant and velocity uncertainties
    meteor_ra = DragMeteor(18000.0, 60.0, 45.0, 1e-3, 3e-3, 2.0, 5.0)
    traj_az = observeMeteor(meteor_ra, noise=10.0, stations=STATIONS, clock_offsets={'B': 0.25, 'C': -0.15}, \
        max_toffset=5.0)
    for mtype in (1, 4):
        mc_dir = os.path.join(str(tmp_path), 'mc_mtype_{:d}'.format(mtype))
        traj_mt = Trajectory(traj_az.jdt_ref, output_dir=mc_dir, meastype=mtype, max_toffset=5.0, \
            gravity_correction=True, monte_carlo=True, mc_runs=4, mc_cores=1, show_plots=False, \
            save_results=(mtype == 4), verbose=False)
        for obs in traj_az.observations:
            if mtype == 1:
                m1, m2 = altAz2RADec_vect(obs.azim_data, obs.elev_data, traj_az.jdt_ref, obs.lat, obs.lon)
            else:
                m1 = (np.pi/2.0 - obs.azim_data)%(2.0*np.pi)
                m2 = np.pi/2.0 - obs.elev_data
            traj_mt.infillTrajectory(m1, m2, obs.time_data, obs.lat, obs.lon, obs.ele, \
                station_id=obs.station_id)
        with contextlib.redirect_stdout(io.StringIO()):
            traj_mt = traj_mt.run()
        assert traj_mt is not None
        assert traj_mt.uncertainties is not None
        assert traj_mt.uncertainties.ra_g > 0.0
        assert traj_mt.uncertainties.v_init < 500.0
        assert abs(traj_mt.v_init - meteor_ra.velocityAtTime(0.0)[0]) < 150.0

    # 4. Verify a failed MC worker is rejected rather than returning a stale clone of the nominal trajectory
    traj_fail = copy.deepcopy(traj_mt)
    for obs in traj_fail.observations:
        obs.ignore_station = True
    with contextlib.redirect_stdout(io.StringIO()):
        mc_failed = _MCTrajSolve([0, traj_fail, None])
        kept_mc = checkMCTrajectories([mc_failed], timing_res=traj_mt.timing_res)
    assert len(kept_mc) == 0

    # 5. Verify stereographic CelestialPlot handles identical zero-span points without crashing
    fig = plt.figure()
    CelestialPlot(np.array([1.0, 1.0, 1.0]), np.array([0.5, 0.5, 0.5]), projection='stere', bgcolor='w')
    plt.close(fig)
