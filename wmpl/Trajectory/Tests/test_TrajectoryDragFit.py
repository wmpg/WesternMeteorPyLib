""" Tests of the drag-coupled trajectory solver in wmpl.Trajectory.TrajectoryDragFit.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_TrajectoryDragFit.py -v
"""

from __future__ import print_function, division, absolute_import

import contextlib
import io

import matplotlib
matplotlib.use('Agg')
import numpy as np

from wmpl.Trajectory.Tests.SyntheticTrajectory import DragMeteor, GravityDragMeteor, JDT_REF, STATIONS, \
    observeMeteor, timingOffsets
from wmpl.Trajectory.TrajectoryDragFit import DragTimingFit, TrajectoryDragFit, computeStationCoverage, \
    findObservedWindows, plotDragTrajectory, plotDragVelocity, solveTrajectoryDragFit
from wmpl.Utils.Math import angleBetweenVectors


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
    """ Check that solveTrajectoryDragFit recovers radiant, velocity and clock offsets on a decelerating
        meteor with gravity drop and rotating air.
    """

    meteor = GravityDragMeteor(20000.0, 120.0, 45.0, 1.0, 4.0, air_rotation=True)
    clock_offsets = {'B': 1.35, 'C': -0.82}

    traj = observeMeteor(meteor, noise=10.0, stations=STATIONS, clock_offsets=clock_offsets, \
        gravity_correction=True, max_toffset=5.0)

    with contextlib.redirect_stdout(io.StringIO()):
        traj = solveTrajectoryDragFit(traj)

    assert traj is not None
    assert traj.drag_timing_fit is not None

    offs = timingOffsets(traj)
    assert abs(offs['B'] + 1.35) < 0.015
    assert abs(offs['C'] + (-0.82)) < 0.015

    rad_err_arcsec = np.degrees(angleBetweenVectors(traj.radiant_eci_mini, meteor.radiant_eci))*3600.0
    assert rad_err_arcsec < 60.0
    assert abs(traj.v_init - 20000.0) < 250.0



def testZeroOverlapTwoStationsRecoversClockAndTrajectory():
    """ Two stations with zero overlap and a 20% unobserved middle gap fail in standard Trajectory.run, but
        solveTrajectoryDragFit recovers the clock offset, radiant and initial velocity.
    """

    meteor = DragMeteor(18000.0, 120.0, 45.0, 0.008, 0.02, 2.7, 6.0, ground_drag=True)
    clock_offsets = {'B': 2.15}

    # Verify the standard solver fails on this zero-overlap configuration
    traj_std = observeMeteor(meteor, noise=10.0, stations=TWO_STATION_GAPPED, clock_offsets=clock_offsets, \
        gravity_correction=False, max_toffset=5.0)
    with contextlib.redirect_stdout(io.StringIO()):
        res_std = traj_std.run()
    assert res_std is None

    # Solve the same observations with the drag-coupled solver
    traj_drag = observeMeteor(meteor, noise=10.0, stations=TWO_STATION_GAPPED, clock_offsets=clock_offsets, \
        gravity_correction=False, max_toffset=5.0)
    with contextlib.redirect_stdout(io.StringIO()):
        traj_drag = solveTrajectoryDragFit(traj_drag)

    assert traj_drag is not None
    assert len(traj_drag.drag_timing_fit.gaps) == 1
    assert len(traj_drag.drag_timing_fit.single_station_spans) == 2
    assert len(traj_drag.drag_timing_fit.gap_uncertainties) == 1

    offs = timingOffsets(traj_drag)
    assert abs(offs['B'] + 2.15) < 0.05

    rad_err_arcsec = np.degrees(angleBetweenVectors(traj_drag.radiant_eci_mini, meteor.radiant_eci))*3600.0
    assert rad_err_arcsec < 120.0
    assert abs(traj_drag.v_init - 18000.0) < 350.0



def testDisconnectedGroupsWithMonteCarloAndPlots():
    """ Four stations in two disconnected groups with a 20% gap and clock offsets are aligned by
        TrajectoryDragFit with Monte Carlo uncertainties enabled, and both diagnostic plots render cleanly.
    """

    meteor = GravityDragMeteor(22000.0, 60.0, 35.0, 5.0, 5.0, air_rotation=True)
    clock_offsets = {'B': -0.40, 'C': 1.80, 'D': -1.25}

    traj_base = observeMeteor(meteor, noise=15.0, stations=TWO_GROUP_GAPPED_STATIONS, \
        clock_offsets=clock_offsets, gravity_correction=True, max_toffset=5.0)

    # Copy observations into TrajectoryDragFit with Monte Carlo enabled
    traj = TrajectoryDragFit(traj_base.jdt_ref, meastype=2, max_toffset=5.0, gravity_correction=True, \
        monte_carlo=True, mc_runs=8, mc_cores=2, show_plots=False, save_results=False, verbose=False, \
        air_rotation=True)
    for obs in traj_base.observations:
        traj.infillWithObs(obs)

    with contextlib.redirect_stdout(io.StringIO()):
        traj = traj.run()

    assert traj is not None
    assert len(traj.drag_timing_fit.gaps) == 1
    assert traj.uncertainties is not None
    assert traj.uncertainties.v_init > 0.0
    assert traj.drag_timing_fit.mc_vel_std is not None
    assert traj.drag_timing_fit.mc_k_lo is not None
    assert traj.drag_timing_fit.mc_time_diffs_std is not None
    assert np.all(traj.drag_timing_fit.mc_time_diffs_std < 0.05)

    offs = timingOffsets(traj)
    assert abs(offs['B'] + (-0.40)) < 0.02
    assert abs(offs['C'] + 1.80) < 0.06
    assert abs(offs['D'] + (-1.25)) < 0.06

    rad_err_arcsec = np.degrees(angleBetweenVectors(traj.radiant_eci_mini, meteor.radiant_eci))*3600.0
    assert rad_err_arcsec < 90.0

    # Verify both the velocity overlay plot and the separate drag-fit plot render cleanly
    fig_vel = plotDragVelocity(traj, show_plots=False, save_results=False)
    fig_drag = plotDragTrajectory(traj, show_plots=False, save_results=False)
    assert fig_vel is not None
    assert fig_drag is not None



def testStationBeyondMaxToffsetIsIgnored():
    """ A station whose clock is off by more than max_toffset is ignored while the remaining stations stay
        accurately aligned.
    """

    meteor = DragMeteor(20000.0, 45.0, 40.0, 1e-3, 3e-3, 2.0, 5.0)
    four_stations = STATIONS + [('D', 45.3, 15.5, 300.0, 0.0, 1.0)]

    traj = observeMeteor(meteor, noise=0.0, stations=four_stations, clock_offsets={'D': 25.0}, \
        max_toffset=5.0, gravity_correction=False)

    with contextlib.redirect_stdout(io.StringIO()):
        traj = solveTrajectoryDragFit(traj)

    assert traj is not None
    ignored = [obs.station_id for obs in traj.observations if obs.ignore_station]
    assert ignored == ['D']

    offs = timingOffsets(traj)
    assert abs(offs['B']) < 1e-2
    assert abs(offs['C']) < 1e-2



def testPickleRoundTrip(tmp_path):
    """ Check that a drag-coupled trajectory saved to pickle loads cleanly as a standard Trajectory. """

    from wmpl.Trajectory.Trajectory import Trajectory
    from wmpl.Utils.Pickling import loadPickle, savePickle

    meteor = DragMeteor(18000.0, 120.0, 45.0, 0.008, 0.02, 2.7, 6.0)
    traj = observeMeteor(meteor, noise=5.0, stations=TWO_STATION_GAPPED, clock_offsets={'B': 1.2}, \
        gravity_correction=False, max_toffset=5.0)

    with contextlib.redirect_stdout(io.StringIO()):
        traj = solveTrajectoryDragFit(traj)

    out_dir = str(tmp_path)
    savePickle(traj, out_dir, 'test_drag_traj.pickle')
    loaded = loadPickle(out_dir, 'test_drag_traj.pickle')

    assert isinstance(loaded, Trajectory)
    assert loaded.drag_timing_fit is not None
    assert len(loaded.drag_timing_fit.gaps) == 1
    assert abs(loaded.v_init - traj.v_init) < 1e-6



def testMaxK0PreventsHighAltitudeNoiseOverfit():
    """ Check that max_k0 bounds initial K_0 and prevents unphysical hyperbolic speed blow-up on short noisy
        high-altitude faint meteors.
    """

    from wmpl.Utils.TrajConversions import geo2Cartesian

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
        initial velocity, end velocity, mid-gap velocity and station clock offsets.
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
    for obs in traj_gap.observations:
        sh = shifts.get(obs.station_id, 0.0)
        t_true = obs.time_data - sh
        keep = (t_true <= t_cut_lo) | (t_true >= t_cut_hi)
        obs.time_data = obs.time_data[keep]
        obs.azim_data = obs.azim_data[keep]
        obs.elev_data = obs.elev_data[keep]
        obs.meas1 = obs.meas1[keep]
        obs.meas2 = obs.meas2[keep]
        obs.ignore_list = obs.ignore_list[keep]
        if obs.magnitudes is not None:
            obs.magnitudes = obs.magnitudes[keep]

    with contextlib.redirect_stdout(io.StringIO()):
        sol_full = solveTrajectoryDragFit(traj_full)
        sol_gap = solveTrajectoryDragFit(traj_gap)

    assert (sol_full is not None) and (sol_gap is not None)
    assert len(sol_gap.drag_timing_fit.gaps) == 1

    # Compare radiant, initial/end/mid-gap velocity and clock offsets against the full solution
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

    from wmpl.Trajectory.Trajectory import Trajectory

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
        traj = solveTrajectoryDragFit(traj)

    assert traj is not None
    raw_offs = {obs.station_id: float(dt) for obs, dt in zip(traj.observations, traj.time_diffs_final)}
    assert abs(raw_offs['A'] - 0.35) < 1e-6
    assert abs(raw_offs['B'] + 0.80) < 0.015
    assert abs(raw_offs['C'] - 0.0) < 0.015
    assert abs(traj.stations_time_dict_copy['A'] - 0.35) < 1e-6



def testReportSummaryAndTerminalMassGating(tmp_path):
    """ Verify that the drag fit summary is appended to *_report.txt, terminal dynamic mass is reported for
        deeply decelerating meteorite-dropping fireballs, and safely suppressed for high-altitude small
        meteors.
    """

    import os
    from wmpl.Utils.TrajConversions import geo2Cartesian

    # 1. Deep meteorite-dropping fireball (14.5 km/s, K = 0.002 -> 0.004 m^2/kg, penetrates below 30 km)
    meteor_fall = DragMeteor(14500.0, 120.0, 45.0, 0.002, 0.004, 3.0, 11.0)
    traj_fall = observeMeteor(meteor_fall, noise=4.0, stations=STATIONS, gravity_correction=False, \
        max_toffset=5.0)
    traj_fall.output_dir = str(tmp_path)
    traj_fall.save_results = True
    traj_fall.show_plots = False

    with contextlib.redirect_stdout(io.StringIO()):
        traj_fall = solveTrajectoryDragFit(traj_fall)

    assert traj_fall is not None
    tm_fall = traj_fall.drag_timing_fit.terminalDynamicMass()
    assert tm_fall['mass'] is not None
    assert tm_fall['mass_lo'] <= tm_fall['mass'] <= tm_fall['mass_hi']
    assert 0.05 < tm_fall['mass'] < 50.0

    report_path = os.path.join(str(tmp_path), traj_fall.file_name + '_report.txt')
    assert os.path.isfile(report_path)
    report_txt = open(report_path, 'r').read()
    assert 'Drag fit summary' in report_txt
    assert 'Terminal dynamic mass:' in report_txt
    assert 'not estimated' not in report_txt

    # Wide MC ensemble spread on K_end must also suppress the terminal dynamic mass estimate
    df_fall = traj_fall.drag_timing_fit
    k_end = float(np.exp(df_fall.log_k_grid[-1]))
    df_fall.mc_k_lo = np.full_like(df_fall.len_grid, k_end*np.exp(-2.0))
    df_fall.mc_k_hi = np.full_like(df_fall.len_grid, k_end*np.exp(2.0))
    tm_wide_mc = df_fall.terminalDynamicMass()
    assert tm_wide_mc['mass'] is None
    assert 'poorly constrained' in tm_wide_mc['reason']

    # 2. Small high-altitude meteor (100 -> 88 km, 35 km/s): terminal mass must NOT be estimated
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
    summary_small = fit_small.formatReportSummary()
    assert 'Terminal dynamic mass: not estimated' in summary_small
