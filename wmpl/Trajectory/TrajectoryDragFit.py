""" Meteor trajectory solver coupled with the physical drag fit.

The standard trajectory solver in wmpl.Trajectory.Trajectory estimates station timing offsets by comparing
pairs of stations over their common length interval, requiring at least four overlapping points per pair.
Stations with no time overlap are dropped, and disconnected groups of stations (as on long Earth-grazers or
cloud-gapped fireballs) have unconstrained relative clocks.

This module replaces pairwise overlap matching with a fast joint fit of the physical atmospheric drag model
(from wmpl.Utils.DragVelocity and wmpl.Utils.DragFit) and the station clock offsets:

    dv/dt = -K(L)*rho(h(L))*v_air**2 + g_along(L)

where rho is the NRLMSISE-00 air density along the trajectory, g_along is the component of gravity along the
motion, v_air = v - v_rot is the speed relative to the co-rotating atmosphere, and ln K(L) is a sum of
non-negative ramps in length L. Because the drag equation bridges unobserved gaps physically, cameras do not
need simultaneous overlap to align in time.

The 3D line-of-sight geometry (pure cross-track angular fit, reusing wmpl.Trajectory.Trajectory) and the
drag-timing fit are iterated to convergence, updating the station ECI coordinates for Earth's rotation, the
gravity and rotating-air drop across the path, and the air density along the dropped path at each pass.
Unobserved gaps (zero stations) are detected automatically, assigned a single regularized ramp of ln K per
gap, bracketed for systematic timing uncertainty from an unobserved change of K, and hatched on both the
velocity plot and the separate drag-fit plot. Monte Carlo uncertainty estimation reuses the standard WMPL
Monte Carlo pipeline, with warm-started drag fits for speed and ensemble uncertainties on the velocity,
ballistic coefficient and timing offsets.

Standalone usage:
    python -m wmpl.Trajectory.TrajectoryDragFit path/to/event_trajectory.pickle [options]
    python -m wmpl.Trajectory.TrajectoryDragFit path/to/*.ecsv [options]
"""

from __future__ import print_function, division, absolute_import

import argparse
import collections
import copy
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

from wmpl.Trajectory.Orbit import calcOrbit
from wmpl.Trajectory.Trajectory import GRAVITY_DROP_POINTS, LEGEND_TEXT_SIZE, LOS_CONTINUE_RUNS, \
    PlaneIntersection, Trajectory, checkWeights, copyUncertainties, dropTimeOrigin, gravityDropOrigin, \
    gravityDropReference, gravityDropVectors, minimizeAngleCost, monteCarloTrajectory, moveStateVector, \
    nelderMeadLine, numStationsNotIgnored, stateVectorDiverged
from wmpl.Utils.AtmosphereDensity import getAtmDensity
from wmpl.Utils.DragVelocity import DRAG_VEL_ATM_POINTS, DRAG_VEL_GRID_POINTS, DRAG_VEL_LOG_K_BOUNDS, \
    DRAG_VEL_LOG_K_SCAN, DRAG_VEL_PHI_MAX, DRAG_VEL_RAMP_MAX, DRAG_VEL_V_BOUNDS, cumulativeTrapz
from wmpl.Utils.Math import lineFunc, vectNorm
from wmpl.Utils.OSTools import mkdirP
from wmpl.Utils.Physics import earthRotationVelocity
from wmpl.Utils.Pickling import loadPickle, savePickle
from wmpl.Utils.Plotting import savePlot, stationColors
from wmpl.Utils.TrajConversions import EARTH, G, altAz2RADec_vect, cartesian2Geo, jd2Date


# Maximum length grid size used when solving the drag equation on very long trajectories
DRAG_TRAJ_MAX_GRID_POINTS = 400

# Target grid spacing along the trajectory (m)
DRAG_TRAJ_GRID_STEP = 500.0

# Minimum unobserved length gap to treat as a separate gap window (m)
DRAG_TRAJ_MIN_GAP = 2000.0

# Factor of the local point spacing above which an intra-station gap splits an observed window
DRAG_TRAJ_GAP_FACTOR = 3.0

# Maximum number of fixed-point iterations for the rotating-air drag scale
DRAG_TRAJ_AIR_MAX_ITER = 10

# Speed convergence tolerance for the rotating-air drag scale (m/s)
DRAG_TRAJ_AIR_TOL = 1e-6

# Default convergence tolerance on incremental station time offsets between coupled passes (s)
DRAG_TRAJ_DT_TOL = 5e-3

# Default maximum number of coupled LoS and drag-timing iterations
DRAG_TRAJ_MAX_ITER = 4

# Default upper bound on initial ballistic coefficient K_0 (m^2/kg) to prevent high-altitude noise overfitting
#   on short faint meteors (ramps still allow K to grow above K_0 during fragmentation)
DRAG_TRAJ_MAX_K0 = 10.0

# Evaluation limits for exploratory cold starts and final / warm-started least-squares refinements
DRAG_TRAJ_EXPLORE_MAX_NFEV = 80
DRAG_TRAJ_REFINE_MAX_NFEV = 250
DRAG_TRAJ_WARM_MAX_NFEV = 100



def _stationSubIntervals(len_arr, gap_factor=DRAG_TRAJ_GAP_FACTOR, min_gap_m=DRAG_TRAJ_MIN_GAP):
    """ Split one station's length array into contiguous observed sub-intervals at large internal gaps.

    Arguments:
        len_arr: [ndarray] Observed lengths along the trajectory for one station (m).

    Keyword arguments:
        gap_factor: [float] Multiple of the station's median length step above which a gap splits the
            interval. DRAG_TRAJ_GAP_FACTOR by default.
        min_gap_m: [float] Minimum internal gap length (m). DRAG_TRAJ_MIN_GAP by default.

    Return:
        [list] List of (len_beg, len_end) tuples (m).
    """

    if len(len_arr) == 0:
        return []

    len_sorted = np.sort(np.asarray(len_arr, dtype=np.float64))
    if len(len_sorted) < 2:
        return [(float(len_sorted[0]), float(len_sorted[0]))]

    diffs = np.diff(len_sorted)
    pos_diffs = diffs[diffs > 0]
    med_step = float(np.median(pos_diffs)) if len(pos_diffs) else 0.0
    gap_thresh = max(min_gap_m, gap_factor*med_step)
    splits = np.where(diffs > gap_thresh)[0]

    intervals = []
    start = 0
    for split_ind in splits:
        intervals.append((float(len_sorted[start]), float(len_sorted[split_ind])))
        start = split_ind + 1
    intervals.append((float(len_sorted[start]), float(len_sorted[-1])))

    return intervals



def findObservedWindows(len_list, gap_factor=DRAG_TRAJ_GAP_FACTOR, min_gap_m=DRAG_TRAJ_MIN_GAP):
    """ Identify contiguous observed length windows and unobserved gaps from station length arrays.

    Arguments:
        len_list: [list] List of 1D ndarrays of observed lengths along the trajectory per station (m).

    Keyword arguments:
        gap_factor: [float] Multiple of a station's median length step above which an internal gap splits
            its coverage into separate windows. DRAG_TRAJ_GAP_FACTOR by default.
        min_gap_m: [float] Minimum gap length (m) considered an unobserved gap. DRAG_TRAJ_MIN_GAP by default.

    Return:
        (windows, gaps):
            - windows: [list] Sorted list of (len_beg, len_end) tuples of observed windows (m).
            - gaps: [list] Sorted list of (gap_beg, gap_end) tuples of unobserved gaps (m).
    """

    intervals = []
    for len_arr in len_list:
        intervals.extend(_stationSubIntervals(len_arr, gap_factor=gap_factor, min_gap_m=min_gap_m))

    if not intervals:
        return [], []

    # Merge overlapping or nearly touching intervals across stations
    intervals.sort()
    merged = [list(intervals[0])]
    for beg, end in intervals[1:]:

        if beg <= merged[-1][1] + 0.25*min_gap_m:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([beg, end])


    windows = [(w[0], w[1]) for w in merged]
    gaps = [(windows[i][1], windows[i + 1][0]) for i in range(len(windows) - 1)]

    return windows, gaps



def _extractMaskSpans(len_grid, mask, min_span_m=0.0):
    """ Extract contiguous (len_beg, len_end) spans from a boolean mask on len_grid.

    Arguments:
        len_grid: [ndarray] Sorted length grid (m).
        mask: [ndarray] Boolean mask on len_grid.

    Keyword arguments:
        min_span_m: [float] Discard spans shorter than this length (m). 0.0 by default.

    Return:
        [list] List of (len_beg, len_end) tuples (m).
    """

    if not np.any(mask):
        return []

    padded = np.r_[False, mask, False]
    starts = np.where(padded[1:] & (~padded[:-1]))[0]
    ends = np.where((~padded[1:]) & padded[:-1])[0] - 1

    spans = []
    for s_ind, e_ind in zip(starts, ends):
        beg = float(len_grid[s_ind])
        end = float(len_grid[e_ind])
        if (end - beg) >= min_span_m:
            spans.append((beg, end))

    return spans



def computeStationCoverage(len_grid, len_list, gap_factor=DRAG_TRAJ_GAP_FACTOR, min_gap_m=DRAG_TRAJ_MIN_GAP):
    """ Count how many stations observe every point on len_grid, and partition the track into overlapping
        (>= 2 stations), single-station (1 station, no overlap) and unobserved gap (0 stations) spans.

    Arguments:
        len_grid: [ndarray] Sorted grid of lengths along the trajectory (m).
        len_list: [list] List of 1D ndarrays of observed lengths per active station (m).

    Keyword arguments:
        gap_factor: [float] Internal gap splitting factor. DRAG_TRAJ_GAP_FACTOR by default.
        min_gap_m: [float] Minimum gap length (m). DRAG_TRAJ_MIN_GAP by default.

    Return:
        (n_stations_grid, overlap_spans, single_spans, gap_spans):
            - n_stations_grid: [ndarray] Integer count of observing stations at each node of len_grid.
            - overlap_spans: [list] (len_beg, len_end) spans with >= 2 stations (m).
            - single_spans: [list] (len_beg, len_end) spans with exactly 1 station (m).
            - gap_spans: [list] (len_beg, len_end) spans with 0 stations (m).
    """

    n_stations_grid = np.zeros(len(len_grid), dtype=np.int64)
    for len_arr in len_list:

        sub_intervals = _stationSubIntervals(len_arr, gap_factor=gap_factor, min_gap_m=min_gap_m)
        covered = np.zeros(len(len_grid), dtype=bool)
        for beg, end in sub_intervals:
            covered |= (len_grid >= beg) & (len_grid <= end)
        n_stations_grid += covered.astype(np.int64)

    min_single = max(0.5*min_gap_m, float(len_grid[1] - len_grid[0]) if len(len_grid) > 1 else 0.0)
    overlap_spans = _extractMaskSpans(len_grid, n_stations_grid >= 2)
    single_spans = _extractMaskSpans(len_grid, n_stations_grid == 1, min_span_m=min_single)
    gap_spans = _extractMaskSpans(len_grid, n_stations_grid == 0)

    return n_stations_grid, overlap_spans, single_spans, gap_spans



def buildRampsWithGaps(len_grid, windows, gaps, n_points_total, n_knots=None):
    """ Place monotonic ramps of ln K inside observed windows and one ramp across each unobserved gap.

    Arguments:
        len_grid: [ndarray] Grid of lengths along the trajectory (m).
        windows: [list] Observed length windows as (len_beg, len_end) tuples (m).
        gaps: [list] Unobserved length gaps as (gap_beg, gap_end) tuples (m).
        n_points_total: [int] Total number of used data points across all stations.

    Keyword arguments:
        n_knots: [int] Target number of observed segments of ln K. None by default, which uses one segment
            per 20 points, clipped to [1, 15].

    Return:
        (ramp_beg, ramp_len, is_gap_ramp):
            - ramp_beg: [ndarray] Start lengths of all ramps (m), sorted.
            - ramp_len: [ndarray] Lengths of all ramps (m).
            - is_gap_ramp: [ndarray] Boolean mask, True for ramps spanning an unobserved gap.
    """

    len_first = float(len_grid[0])
    len_last = float(len_grid[-1])

    if n_knots is None:
        n_knots = int(np.clip(n_points_total/20.0, 1, 15))

    # Without gaps, space ramps uniformly along the whole observed span
    if not gaps:
        ramp_beg = np.linspace(len_first, len_last, n_knots + 1)[:-1]
        ramp_len = np.full(n_knots, max((len_last - len_first)/n_knots, 1.0))
        return ramp_beg, ramp_len, np.zeros(n_knots, dtype=bool)


    # Allocate observed ramps proportionally across observed windows, with at least one per window
    total_obs_len = sum(max(end - beg, 1.0) for beg, end in windows)
    obs_knots = max(n_knots, len(windows))

    ramp_beg = []
    ramp_len = []
    is_gap_ramp = []

    for beg, end in windows:

        if end <= beg:
            continue

        cnt = max(1, int(round(obs_knots*max(end - beg, 1.0)/total_obs_len)))
        rb = np.linspace(beg, end, cnt + 1)[:-1]
        rl = np.full(cnt, (end - beg)/cnt)
        ramp_beg.extend(rb.tolist())
        ramp_len.extend(rl.tolist())
        is_gap_ramp.extend([False]*cnt)


    # Add a single monotonic ramp across each unobserved gap
    for gap_beg, gap_end in gaps:
        ramp_beg.append(gap_beg)
        ramp_len.append(max(gap_end - gap_beg, 1.0))
        is_gap_ramp.append(True)


    ramp_beg = np.array(ramp_beg, dtype=np.float64)
    ramp_len = np.array(ramp_len, dtype=np.float64)
    is_gap_ramp = np.array(is_gap_ramp, dtype=bool)

    order = np.argsort(ramp_beg)

    return ramp_beg[order], ramp_len[order], is_gap_ramp[order]



def lineDensityGravityDropped(state_vect, radiant_eci, len_grid, jd, drop_grid=None, gravity_factor=1.0):
    """ Compute the air density, heights and gravity along the trajectory, optionally on the gravity-dropped
        path. On long shallow trajectories such as Earth-grazers, the gravity drop lowers the path by several
        kilometers, significantly increasing the air density relative to the straight tangent line.

    Arguments:
        state_vect: [ndarray] ECI coordinates of the point on the line from which lengths are measured (m).
        radiant_eci: [ndarray] Unit vector of the radiant in ECI, opposite to the motion.
        len_grid: [ndarray] Sorted lengths along the line in the direction of motion (m).
        jd: [float] Julian date.

    Keyword arguments:
        drop_grid: [ndarray] ECI gravity drop vectors at each point of len_grid (m), shape (N, 3). None by
            default, in which case the straight line is used.
        gravity_factor: [float] Factor by which gravity is multiplied. 1.0 by default.

    Return:
        (rho, g_along, ht_grid):
            - rho: [ndarray] Air density on len_grid (kg/m^3).
            - g_along: [ndarray] Acceleration by gravity along the motion on len_grid (m/s^2).
            - ht_grid: [ndarray] Heights above mean sea level on len_grid (m).
    """

    state_vect = np.asarray(state_vect, dtype=np.float64)
    radiant_eci = vectNorm(np.asarray(radiant_eci, dtype=np.float64))

    # Scale the number of MSIS samples with trajectory length for long Earth-grazers
    span = float(len_grid[-1] - len_grid[0])
    n_atm = int(np.clip(np.ceil(span/10000.0) + 10, DRAG_VEL_ATM_POINTS, 60))

    eci_grid = state_vect[None, :] - len_grid[:, None]*radiant_eci[None, :]
    if drop_grid is not None:
        eci_grid = eci_grid + np.asarray(drop_grid, dtype=np.float64)

    # Sample geographical coordinates along the path for MSIS air density
    atm_ind = np.unique(np.round(np.linspace(0, len(len_grid) - 1, n_atm)).astype(int))
    lla = np.array([cartesian2Geo(jd, *eci_grid[i]) for i in atm_ind])
    len_atm = len_grid[atm_ind]

    rho_atm = getAtmDensity(lla[:, 0], lla[:, 1], np.maximum(lla[:, 2], 0.0), jd, space_weather='constant')
    rho = np.exp(np.interp(len_grid, len_atm, np.log(rho_atm)))
    ht_grid = np.interp(len_grid, len_atm, lla[:, 2])

    # Compute the component of central gravity along the motion
    g_along = gravity_factor*G*EARTH.MASS*eci_grid.dot(radiant_eci)/np.linalg.norm(eci_grid, axis=1)**3

    return rho, g_along, ht_grid



def computeVRotation(state_vect, radiant_eci):
    """ Compute the component of the Earth's atmospheric rotation velocity along the meteor's motion.

    Arguments:
        state_vect: [ndarray] ECI position on the trajectory (m).
        radiant_eci: [ndarray] Unit vector of the radiant in ECI, opposite to the motion.

    Return:
        [float] Speed of the co-rotating air along the direction of motion (m/s).
    """

    motion = -vectNorm(np.asarray(radiant_eci, dtype=np.float64))

    return float(np.dot(earthRotationVelocity(np.asarray(state_vect, dtype=np.float64)), motion))



class DragTimingFit(object):
    """ Fast joint fit of the physical drag propagation model and free station clock offsets.

    The model parameters are:
        [t_first, v_first, ln_K0, delta_1, ..., delta_M, dt_free_1, ..., dt_free_S]
    where delta_m >= 0 are the monotonic ramp increments of ln K(L) and dt_free_s are the timing offsets
    added to the free stations' times.
    """

    def __init__(self, station_times, station_lens, station_weights, free_mask, state_vect, radiant_eci, \
        jd, gravity_factor=1.0, max_toffset=5.0, n_knots=None, penalty=1.0, gap_penalty_factor=1.0, \
        drop_grid_func=None, air_rotation=True, params_init=None, res_scale_init=None, \
        max_k0=DRAG_TRAJ_MAX_K0):
        """ Fit the drag model and free station time offsets to projected lengths and times.

        Arguments:
            station_times: [list] List of 1D ndarrays of used point times per station (s).
            station_lens: [list] List of 1D ndarrays of used point lengths along the line per station (m).
            station_weights: [list] Statistical weight for each station.
            free_mask: [list or ndarray] Boolean mask, True for stations whose clock offset is estimated.
            state_vect: [ndarray] ECI coordinates of the state vector on the trajectory line (m).
            radiant_eci: [ndarray] Unit radiant vector in ECI, opposite to the motion.
            jd: [float] Reference Julian date.

        Keyword arguments:
            gravity_factor: [float] Factor multiplying gravity. 1.0 by default.
            max_toffset: [float] Bound on the magnitude of estimated station time offsets (s). 5.0 by
                default.
            n_knots: [int] Target number of observed ln K segments. None by default (auto).
            penalty: [float] Regularization weight on the growth of ln K. 1.0 by default.
            gap_penalty_factor: [float] Relative penalty multiplier on gap ramps. 1.0 by default.
            drop_grid_func: [callable] Optional function taking len_grid (m) and returning ECI drop vectors
                of shape (N, 3) (m) for evaluating air density on the dropped path. None by default.
            air_rotation: [bool] Account for Earth's rotating atmosphere in the drag speed v_air = v - v_rot.
                True by default.
            params_init: [ndarray] Optional warm-start parameter vector (e.g. from the previous pass or from
                the main solution during Monte Carlo runs). Skips the global scan when its length matches.
                None by default.
            res_scale_init: [float] Optional robust residual scale (m) to use with params_init so the loss
                and penalty scaling stays consistent across passes and Monte Carlo runs. None by default.
            max_k0: [float] Upper bound on the initial ballistic coefficient K_0 (m^2/kg), or None to use
                DRAG_VEL_LOG_K_BOUNDS[1]. DRAG_TRAJ_MAX_K0 by default.
        """

        self.n_stations = len(station_times)
        self.free_mask = np.asarray(free_mask, dtype=bool)
        self.free_indices = np.where(self.free_mask)[0]
        self.n_free = len(self.free_indices)
        self.max_toffset = float(max_toffset)
        self.gravity_factor = float(gravity_factor)
        self.air_rotation = bool(air_rotation)
        self.v_rotation = computeVRotation(state_vect, radiant_eci) if self.air_rotation else 0.0
        self.max_k0 = float(max_k0) if max_k0 is not None else None
        self.max_log_k0 = min(float(np.log(self.max_k0)), DRAG_VEL_LOG_K_BOUNDS[1]) \
            if (self.max_k0 is not None) and (self.max_k0 > 0.0) else DRAG_VEL_LOG_K_BOUNDS[1]

        # Stack all used points across active stations
        t_all, l_all, w_all, s_all = [], [], [], []
        used_lens = []
        for s_idx, (t_arr, l_arr, wt) in enumerate(zip(station_times, station_lens, station_weights)):

            if (len(t_arr) == 0) or (wt <= 0):
                continue

            t_arr = np.asarray(t_arr, dtype=np.float64)
            l_arr = np.asarray(l_arr, dtype=np.float64)
            t_all.append(t_arr)
            l_all.append(l_arr)
            w_all.append(np.full(len(t_arr), np.sqrt(wt), dtype=np.float64))
            s_all.append(np.full(len(t_arr), s_idx, dtype=np.int64))
            used_lens.append(l_arr)

        if not t_all:
            raise ValueError("DragTimingFit requires at least one station with used points")

        self.time_data = np.concatenate(t_all)
        self.len_data = np.concatenate(l_all)
        self.sqrt_w = np.concatenate(w_all)
        self.stat_idx = np.concatenate(s_all)
        self.n_data = len(self.len_data)

        # Map each point to its free timing offset slot (-1 if fixed)
        self.stat_to_free = np.full(self.n_stations, -1, dtype=np.int64)
        for slot, s_idx in enumerate(self.free_indices):
            self.stat_to_free[s_idx] = slot
        self.pt_free_slot = self.stat_to_free[self.stat_idx]

        # Build the length grid scaled to the trajectory length
        self.len_first = float(np.min(self.len_data))
        self.len_last = float(np.max(self.len_data))
        span = max(self.len_last - self.len_first, 1.0)
        n_grid = int(np.clip(np.ceil(span/DRAG_TRAJ_GRID_STEP), DRAG_VEL_GRID_POINTS, \
            DRAG_TRAJ_MAX_GRID_POINTS))
        self.len_grid = np.linspace(self.len_first, self.len_last, n_grid)
        self.dlen = np.diff(self.len_grid)

        # Evaluate air density, gravity and heights along the trajectory (dropped path if provided)
        drop_grid = drop_grid_func(self.len_grid) if drop_grid_func is not None else None
        self.rho, self.g_along, self.ht_grid = lineDensityGravityDropped(state_vect, radiant_eci, \
            self.len_grid, jd, drop_grid=drop_grid, gravity_factor=self.gravity_factor)

        # Detect observed windows, single-station no-overlap regions and unobserved gaps
        self.windows, self.gaps = findObservedWindows(used_lens)
        self.n_stations_grid, self.overlap_spans, self.single_station_spans, _ = computeStationCoverage( \
            self.len_grid, used_lens)

        self.ramp_beg, self.ramp_len, self.is_gap_ramp = buildRampsWithGaps(self.len_grid, self.windows, \
            self.gaps, self.n_data, n_knots=n_knots)
        self.n_knots = len(self.ramp_beg)
        self.ramps = np.clip((self.len_grid[:, None] - self.ramp_beg[None, :])/self.ramp_len[None, :], \
            0.0, 1.0)
        self.pen_weights = np.where(self.is_gap_ramp, gap_penalty_factor, 1.0)

        # Run the joint fit (warm-started when params_init matches)
        sol = self._fit(penalty, params_init=params_init, res_scale_init=res_scale_init)
        self.opt_result = sol
        self.params = sol.x
        self.drag_params = self.params[:3 + self.n_knots]

        self.time_offsets = np.zeros(self.n_stations, dtype=np.float64)
        if self.n_free > 0:
            self.time_offsets[self.free_indices] = self.params[3 + self.n_knots:]

        self.vel_grid, self.time_grid = self._solve(self.drag_params)[:2]
        self.log_k_grid = self.drag_params[2] + self.ramps.dot(self.drag_params[3:])

        # Compute final along-track and timing residuals
        t_corr = self._correctedTimes(self.params)
        len_model = self._lenAtTime(t_corr, self.vel_grid, self.time_grid)
        self.residuals_m = len_model - self.len_data
        self.res_rms_m = float(np.sqrt(np.mean(self.residuals_m**2)))
        self.res_scale_m = float(max(1.4826*np.median(np.abs(self.residuals_m \
            - np.median(self.residuals_m))), 1.0))

        t_model = np.interp(self.len_data, self.len_grid, self.time_grid)
        self.residuals_s = t_model - t_corr
        self.res_rms_s = float(np.sqrt(np.mean(self.residuals_s**2)))
        self.res_mean_abs_s = float(np.mean(np.abs(self.residuals_s)))

        # Estimate parameter covariance and 1-sigma uncertainties from the final Jacobian
        self.params_cov, self.params_null = self._estimateCovariance(sol)
        self.v_init_stddev = float(np.sqrt(max(self.params_cov[1, 1], 0.0))) \
            if self.params_cov is not None else 0.0
        self.log_k_std_grid = self._logKStdOnGrid()

        # Optional Monte Carlo ensemble uncertainty arrays, populated after MC runs
        self.mc_vel_std = None
        self.mc_vel_lo = None
        self.mc_vel_hi = None
        self.mc_k_lo = None
        self.mc_k_hi = None
        self.mc_time_diffs_std = None

        # Compute systematic timing and velocity bounds across unobserved gaps
        self.gap_uncertainties = self.computeGapUncertainties()



    def _airDragScale(self, vel):
        """ Compute the factor by which drag against the rotating air differs from inertial drag.

        Arguments:
            vel: [ndarray] Inertial speeds on the grid (m/s).

        Return:
            [ndarray] Dimensionless drag scaling factor.
        """

        vel_safe = np.maximum(vel, max(50.0, abs(self.v_rotation) + 1.0))
        ratio = 1.0 - self.v_rotation/vel_safe

        return ratio*np.abs(ratio)



    def _correctedTimes(self, params):
        """ Apply the free station timing offsets from params to the stacked point times.

        Arguments:
            params: [ndarray] Full parameter vector.

        Return:
            [ndarray] Time-shifted observation times (s).
        """

        if self.n_free == 0:
            return self.time_data

        dt_free = params[3 + self.n_knots:]
        t_corr = self.time_data.copy()
        free_pts = self.pt_free_slot >= 0
        t_corr[free_pts] += dt_free[self.pt_free_slot[free_pts]]

        return t_corr



    def _solve(self, drag_params, derivatives=False):
        """ Solve the linear energy equation along the length grid, with optional derivatives.

        Arguments:
            drag_params: [ndarray] Drag parameters (t_first, v_first, ln K_0, ramp heights).

        Keyword arguments:
            derivatives: [bool] Compute derivatives of velocity and time with respect to drag_params.
                False by default.

        Return:
            (vel, time_arr, dvel, dtime):
                - vel: [ndarray] Speeds on len_grid (m/s).
                - time_arr: [ndarray] Times on len_grid (s).
                - dvel: [ndarray] Derivatives of vel w.r.t. (v_first, ln K_0, ramps), or None.
                - dtime: [ndarray] Derivatives of time_arr w.r.t. drag_params, or None.
        """

        drag_base = 2.0*np.exp(drag_params[2] + self.ramps.dot(drag_params[3:]))*self.rho
        drag_scale = np.ones_like(drag_base)
        vel_prev = None

        n_iter = DRAG_TRAJ_AIR_MAX_ITER if self.v_rotation != 0.0 else 1
        for i in range(n_iter):

            drag = drag_base*drag_scale
            phi = cumulativeTrapz(drag, self.dlen)
            exp_phi = np.exp(np.clip(phi, -DRAG_VEL_PHI_MAX, DRAG_VEL_PHI_MAX))

            grav = cumulativeTrapz(self.g_along*exp_phi, self.dlen)
            energy = (0.5*drag_params[1]**2 + grav)/exp_phi
            vel = np.sqrt(np.maximum(2.0*energy, 1.0))

            if (vel_prev is not None) and (np.max(np.abs(vel - vel_prev)) < DRAG_TRAJ_AIR_TOL):
                break

            vel_prev = vel
            if n_iter > 1:
                new_scale = self._airDragScale(vel)
                drag_scale = new_scale if i < 2 else 0.5*(drag_scale + new_scale)


        time_arr = drag_params[0] + cumulativeTrapz(1.0/vel, self.dlen)

        if not derivatives:
            return vel, time_arr, None, None

        # Compute exact derivatives of the converged energy ODE
        dlog_k = np.c_[np.ones(len(drag)), self.ramps]
        if self.v_rotation == 0.0:
            dphi = cumulativeTrapz(drag[:, None]*dlog_k, self.dlen)
            dgrav = cumulativeTrapz((self.g_along*exp_phi)[:, None]*dphi, self.dlen)
            denergy = dgrav/exp_phi[:, None] - energy[:, None]*dphi
            dvel = np.c_[drag_params[1]/exp_phi, denergy]/vel[:, None]
        else:
            # Linearize dE/dL = -2*K*rho*s(v)*E + g_along with d(s(v)*E)/dE = 1 - v_rot/v above the floor
            vel_floor = max(50.0, abs(self.v_rotation) + 1.0)
            vel_safe = np.maximum(vel, vel_floor)
            ratio = 1.0 - self.v_rotation/vel_safe
            d_scale_energy = np.where((vel > vel_floor) & (2.0*energy > 1.0), ratio, drag_scale)
            psi = cumulativeTrapz(drag_base*d_scale_energy, self.dlen)
            exp_psi = np.exp(np.clip(psi, -250.0, 250.0))
            q_rhs = -(drag*energy)[:, None]*dlog_k
            denergy_v = drag_params[1]/exp_psi
            denergy_k = cumulativeTrapz(exp_psi[:, None]*q_rhs, self.dlen)/exp_psi[:, None]
            dvel = np.c_[denergy_v, denergy_k]/vel[:, None]
            dvel = np.where((2.0*energy > 1.0)[:, None], dvel, 0.0)

        dtime = np.c_[np.ones(len(vel)), cumulativeTrapz(-dvel/vel[:, None]**2, self.dlen)]

        return vel, time_arr, dvel, dtime



    def _lenAtTime(self, t_corr, vel, time_grid):
        """ Interpolate model lengths at the given times, continuing linearly outside the grid.

        Arguments:
            t_corr: [ndarray] Times (s).
            vel: [ndarray] Model speeds on len_grid (m/s).
            time_grid: [ndarray] Model times on len_grid (s).

        Return:
            [ndarray] Model lengths along the trajectory (m).
        """

        len_model = np.interp(t_corr, time_grid, self.len_grid)
        len_model = np.where(t_corr < time_grid[0], self.len_grid[0] + vel[0]*(t_corr - time_grid[0]), \
            len_model)
        len_model = np.where(t_corr > time_grid[-1], self.len_grid[-1] + vel[-1]*(t_corr - time_grid[-1]), \
            len_model)

        return len_model



    def _residuals(self, params, w_pen):
        """ Compute weighted along-track residuals (m) and ramp growth penalties.

        Arguments:
            params: [ndarray] Full parameter vector.
            w_pen: [float] Penalty weight (m).

        Return:
            [ndarray] Residual vector.
        """

        drag_params = params[:3 + self.n_knots]
        vel, time_grid, _, _ = self._solve(drag_params)
        t_corr = self._correctedTimes(params)
        len_model = self._lenAtTime(t_corr, vel, time_grid)

        data_res = self.sqrt_w*(len_model - self.len_data)
        pen_res = w_pen*self.pen_weights*drag_params[3:]

        return np.concatenate([data_res, pen_res])



    def _jacobian(self, params, w_pen):
        """ Compute the analytical Jacobian of the residuals with respect to all parameters.

        Arguments:
            params: [ndarray] Full parameter vector.
            w_pen: [float] Penalty weight (m).

        Return:
            [ndarray] Jacobian matrix, one row per residual.
        """

        n_drag = 3 + self.n_knots
        drag_params = params[:n_drag]
        vel, time_grid, dvel, dtime = self._solve(drag_params, derivatives=True)
        t_corr = self._correctedTimes(params)

        # Find the grid cell containing each observation time
        ind = np.clip(np.searchsorted(time_grid, t_corr) - 1, 0, len(self.dlen) - 1)
        dtime_seg = time_grid[ind + 1] - time_grid[ind]
        frac = np.clip((t_corr - time_grid[ind])/dtime_seg, 0.0, 1.0)

        slope_at = self.dlen[ind]/dtime_seg
        dtime_at = dtime[ind]*(1.0 - frac)[:, None] + dtime[ind + 1]*frac[:, None]
        jac_drag = -dtime_at*slope_at[:, None]
        dlen_dt = slope_at.copy()

        # Differentiate linear continuation outside the grid time span
        before = t_corr < time_grid[0]
        if np.any(before):
            jac_drag[before] = -vel[0]*dtime[0] \
                + (t_corr[before] - time_grid[0])[:, None]*np.r_[0.0, dvel[0]]
            dlen_dt[before] = vel[0]

        after = t_corr > time_grid[-1]
        if np.any(after):
            jac_drag[after] = -vel[-1]*dtime[-1] \
                + (t_corr[after] - time_grid[-1])[:, None]*np.r_[0.0, dvel[-1]]
            dlen_dt[after] = vel[-1]

        jac_drag *= self.sqrt_w[:, None]

        # Derivatives with respect to free station timing offsets
        jac_dt = np.zeros((self.n_data, self.n_free), dtype=np.float64)
        if self.n_free > 0:
            free_pts = self.pt_free_slot >= 0
            jac_dt[free_pts, self.pt_free_slot[free_pts]] = self.sqrt_w[free_pts]*dlen_dt[free_pts]

        jac_data = np.hstack([jac_drag, jac_dt])
        jac_pen = np.zeros((self.n_knots, n_drag + self.n_free), dtype=np.float64)
        jac_pen[:, 3:n_drag] = np.diag(w_pen*self.pen_weights)

        return np.vstack([jac_data, jac_pen])



    def _scanConstant(self):
        """ Global vectorized scan over constant K using Schur-complement elimination of free station time
            offsets. For fixed K (without gravity), t_model(L) = t_first + F(L)/v_first, so every free
            station's optimal offset is closed-form given (t_first, 1/v_first).

        Return:
            (t_first, v_first, log_k, dt_free, res):
                - t_first: [float] Initial time at len_first (s).
                - v_first: [float] Initial velocity at len_first (m/s).
                - log_k: [float] Logarithm of best constant ballistic coefficient.
                - dt_free: [ndarray] Initial timing offsets of free stations (s).
                - res: [ndarray] Weighted length residuals of the best constant-K fit (m).
        """

        # Compute F(L) for all trial ln K values simultaneously within the allowed K_0 range
        log_k_scan = DRAG_VEL_LOG_K_SCAN[DRAG_VEL_LOG_K_SCAN <= self.max_log_k0 + 1e-9]
        if (len(log_k_scan) == 0) or (log_k_scan[-1] < self.max_log_k0 - 1e-6):
            log_k_scan = np.append(log_k_scan, self.max_log_k0)

        drag = 2.0*np.exp(log_k_scan)[None, :]*self.rho[:, None]
        phi = cumulativeTrapz(drag, self.dlen)
        f_grid = cumulativeTrapz(np.exp(np.clip(0.5*phi, None, DRAG_VEL_PHI_MAX)), self.dlen)

        # Estimate a rough mean speed from internal station slopes (independent of clock offsets)
        v_est_list = []
        for s_idx in range(self.n_stations):
            mask_s = self.stat_idx == s_idx
            if np.count_nonzero(mask_s) >= 2:
                dt_s = float(np.max(self.time_data[mask_s]) - np.min(self.time_data[mask_s]))
                dl_s = float(np.max(self.len_data[mask_s]) - np.min(self.len_data[mask_s]))
                if (dt_s > 1e-4) and (dl_s > 0.0):
                    v_est_list.append(dl_s/dt_s)
        vel_mean = float(np.median(v_est_list)) if v_est_list else 20000.0

        # Precompute point weights and station masks for Schur elimination
        w2 = self.sqrt_w**2
        fixed_mask = self.pt_free_slot < 0
        w_fixed = np.sum(w2[fixed_mask])
        t_fixed_mean = np.sum(w2[fixed_mask]*self.time_data[fixed_mask])/w_fixed if w_fixed > 0 else 0.0

        free_masks = [self.pt_free_slot == slot for slot in range(self.n_free)]
        free_w_sums = [np.sum(w2[m]) for m in free_masks]
        free_t_means = [np.sum(w2[m]*self.time_data[m])/ws if ws > 0 else 0.0 \
            for m, ws in zip(free_masks, free_w_sums)]

        # Precompute centered times for free stations and fixed stations
        t_centered = self.time_data.copy()
        if w_fixed > 0:
            t_centered[fixed_mask] -= t_fixed_mean
        for m, tm in zip(free_masks, free_t_means):
            t_centered[m] -= tm

        # Interpolate grid indices of len_data once for fast evaluation across all K
        grid_ind = np.clip(np.searchsorted(self.len_grid, self.len_data) - 1, 0, len(self.dlen) - 1)
        grid_frac = np.clip((self.len_data - self.len_grid[grid_ind])/self.dlen[grid_ind], 0.0, 1.0)

        best = None
        for i, log_k in enumerate(log_k_scan):

            f_col = f_grid[:, i]
            f_data = f_col[grid_ind]*(1.0 - grid_frac) + f_col[grid_ind + 1]*grid_frac

            # Eliminate t_first (using fixed stations) and all free station offsets analytically
            f_centered = f_data.copy()
            f_fixed_mean = np.sum(w2[fixed_mask]*f_data[fixed_mask])/w_fixed if w_fixed > 0 else 0.0
            if w_fixed > 0:
                f_centered[fixed_mask] -= f_fixed_mean

            free_f_means = []
            for m, ws in zip(free_masks, free_w_sums):
                fm = np.sum(w2[m]*f_data[m])/ws if ws > 0 else 0.0
                free_f_means.append(fm)
                f_centered[m] -= fm

            denom = float(np.sum(w2*f_centered**2))
            if denom <= 0.0:
                continue

            slowness = float(np.sum(w2*f_centered*t_centered))/denom
            if slowness <= 0.0:
                continue

            v_first = 1.0/slowness
            if (v_first < DRAG_VEL_V_BOUNDS[0]) or (v_first > DRAG_VEL_V_BOUNDS[1]):
                continue

            t_first = t_fixed_mean - slowness*f_fixed_mean
            dt_free = np.array([t_first + slowness*fm - tm for fm, tm in zip(free_f_means, free_t_means)], \
                dtype=np.float64)

            # Enforce max_toffset bounds if any free offset exceeds the limit
            if (self.n_free > 0) and np.any(np.abs(dt_free) > self.max_toffset):

                dt_free = np.clip(dt_free, -self.max_toffset, self.max_toffset)
                t_adj = self.time_data.copy()
                free_pts = self.pt_free_slot >= 0
                t_adj[free_pts] += dt_free[self.pt_free_slot[free_pts]]

                w_tot = float(np.sum(w2))
                f_bar = float(np.sum(w2*f_data))/w_tot
                t_bar = float(np.sum(w2*t_adj))/w_tot
                df = f_data - f_bar
                denom2 = float(np.sum(w2*df**2))
                if denom2 <= 0.0:
                    continue

                slowness = float(np.sum(w2*df*(t_adj - t_bar)))/denom2
                if slowness <= 0.0:
                    continue

                v_first = 1.0/slowness
                if (v_first < DRAG_VEL_V_BOUNDS[0]) or (v_first > DRAG_VEL_V_BOUNDS[1]):
                    continue

                t_first = t_bar - slowness*f_bar

            t_pred = t_first + slowness*f_data
            t_corr = self.time_data.copy()
            if self.n_free > 0:
                free_pts = self.pt_free_slot >= 0
                t_corr[free_pts] += dt_free[self.pt_free_slot[free_pts]]

            res = self.sqrt_w*(t_pred - t_corr)*vel_mean
            cost = float(np.sum(res**2))

            if (best is None) or (cost < best[0]):
                best = (cost, t_first, v_first, float(log_k), dt_free, res)


        if best is None:
            raise ValueError("The drag-timing model could not be fitted, lengths do not grow with time")

        return best[1:]



    def _fit(self, penalty, params_init=None, res_scale_init=None):
        """ Fit all drag and timing parameters, warm-starting when params_init matches the parameter size.

        Arguments:
            penalty: [float] Weight of the penalty on the growth of ln K.

        Keyword arguments:
            params_init: [ndarray] Optional warm-start parameter vector. None by default.
            res_scale_init: [float] Optional residual scale (m) used when warm-starting. None by default.

        Return:
            [OptimizeResult] Final solution from scipy.optimize.least_squares.
        """

        n_knots = self.n_knots
        n_total = 3 + n_knots + self.n_free
        lower = np.r_[-np.inf, DRAG_VEL_V_BOUNDS[0], DRAG_VEL_LOG_K_BOUNDS[0], np.zeros(n_knots), \
            np.full(self.n_free, -self.max_toffset)]
        upper = np.r_[np.inf, DRAG_VEL_V_BOUNDS[1], self.max_log_k0, \
            np.full(n_knots, DRAG_VEL_RAMP_MAX), np.full(self.n_free, self.max_toffset)]

        # Fast warm-start path for Pass 2+ and Monte Carlo runs
        if (params_init is not None) and (len(params_init) == n_total):
            p0 = np.clip(np.asarray(params_init, dtype=np.float64), lower + 1e-9, upper - 1e-9)
            if res_scale_init is not None:
                return self._leastSquares(p0, None, penalty, lower, upper, max_nfev=DRAG_TRAJ_WARM_MAX_NFEV, \
                    res_scale=res_scale_init)
            res0 = self._residuals(p0, 0.0)[:self.n_data]
            return self._leastSquares(p0, res0, penalty, lower, upper, max_nfev=DRAG_TRAJ_WARM_MAX_NFEV)

        t_first, v_first, log_k, dt_free, res = self._scanConstant()

        # Start from both constant K and K growing by a factor of 10, capped to avoid slow stalls
        best = None
        for d_init in (1e-3, np.log(10.0)/max(n_knots, 1)):

            p0 = np.r_[t_first, v_first, log_k - n_knots*d_init/2.0, np.full(n_knots, d_init), dt_free]
            p0 = np.clip(p0, lower + 1e-9, upper - 1e-9)

            sol = self._leastSquares(p0, res, penalty, lower, upper, max_nfev=DRAG_TRAJ_EXPLORE_MAX_NFEV)
            if (best is None) or (sol.cost < best.cost):
                best = sol


        # Re-estimate the robust residual scale and refine the winning basin
        res_final = self._residuals(best.x, 0.0)[:self.n_data]

        return self._leastSquares(best.x, res_final, penalty, lower, upper, \
            max_nfev=DRAG_TRAJ_REFINE_MAX_NFEV)



    def _leastSquares(self, params_init, res, penalty, lower, upper, max_nfev=DRAG_TRAJ_REFINE_MAX_NFEV, \
        res_scale=None):
        """ Solve the bounded robust nonlinear least-squares problem with the analytical Jacobian.

        Arguments:
            params_init: [ndarray] Initial parameter vector.
            res: [ndarray] Residuals used to set the robust loss scale (m), ignored if res_scale is given.
            penalty: [float] Weight of the penalty on the growth of ln K.
            lower: [ndarray] Lower parameter bounds.
            upper: [ndarray] Upper parameter bounds.

        Keyword arguments:
            max_nfev: [int] Maximum number of function evaluations. DRAG_TRAJ_REFINE_MAX_NFEV by default.
            res_scale: [float] Explicit residual scale (m). None by default (estimated from res).

        Return:
            [OptimizeResult] Solution from scipy.optimize.least_squares.
        """

        if res_scale is None:
            res_scale = max(1.4826*np.median(np.abs(res - np.median(res))), 1.0)
        else:
            res_scale = max(float(res_scale), 1.0)
        f_scale = 2.0*res_scale
        w_pen = np.sqrt(penalty)*f_scale

        return scipy.optimize.least_squares(self._residuals, params_init, jac=self._jacobian, \
            args=(w_pen,), bounds=(lower, upper), loss='soft_l1', f_scale=f_scale, x_scale='jac', \
            ftol=1e-6, xtol=1e-6, max_nfev=max_nfev)



    def _estimateCovariance(self, sol):
        """ Estimate the parameter covariance matrix and null-space directions from the final Jacobian.

        Arguments:
            sol: [OptimizeResult] Solution from scipy.optimize.least_squares.

        Return:
            (cov, params_null):
                - cov: [ndarray] Parameter covariance matrix, or None if SVD fails.
                - params_null: [ndarray] Null-space basis vectors (unconstrained directions).
        """

        jac = sol.jac
        col_norm = np.linalg.norm(jac, axis=0)
        col_norm[col_norm == 0.0] = 1.0

        try:
            _, sing, vt = np.linalg.svd(jac/col_norm, full_matrices=False)
        except np.linalg.LinAlgError:
            return None, np.zeros((jac.shape[1], 0))

        null = sing <= 1e-8*sing[0]
        vt_good = vt[~null]/col_norm
        cov = self.res_scale_m**2*(vt_good.T/sing[~null]**2).dot(vt_good)

        params_null = (vt[null]/col_norm).T
        if params_null.size:
            params_null /= np.linalg.norm(params_null, axis=0)

        return cov, params_null



    def _logKStdOnGrid(self):
        """ Propagate parameter covariance to the 1-sigma uncertainty of ln K along len_grid.

        Return:
            [ndarray] 1-sigma uncertainty of ln K on len_grid (np.inf where unconstrained).
        """

        if self.params_cov is None:
            return np.full(len(self.len_grid), np.nan)

        n_drag = 3 + self.n_knots
        grad = np.zeros((len(self.len_grid), n_drag + self.n_free), dtype=np.float64)
        grad[:, 2] = 1.0
        grad[:, 3:n_drag] = self.ramps

        std = np.sqrt(np.maximum(np.sum(grad.dot(self.params_cov)*grad, axis=1), 0.0))

        if self.params_null.size:
            grad_norm = np.linalg.norm(grad, axis=1)[:, None]
            unconstrained = np.any(np.abs(grad.dot(self.params_null)) > 1e-6*grad_norm, axis=1)
            std[unconstrained] = np.inf

        return std



    def computeGapUncertainties(self):
        """ Compute physical bounds on transit time and velocity across every unobserved gap by bracketing
            the unknown location of any K growth inside the gap between an early jump (at gap_beg, maximum
            deceleration and longest transit time) and a late jump (at gap_end, minimum deceleration and
            shortest transit time).

        Return:
            [list] List of dictionaries, one per unobserved gap, containing its length, height and time
                bounds, fitted growth of ln K across the gap, early/late transit time offsets relative to the
                nominal model (s), total timing uncertainty span (s), and gap velocity envelopes (m/s).
        """

        gap_info_list = []
        if not self.gaps:
            return gap_info_list

        gap_ramp_indices = np.where(self.is_gap_ramp)[0]

        for gap_idx, (gap_beg, gap_end) in enumerate(self.gaps):

            r_idx = gap_ramp_indices[gap_idx] if gap_idx < len(gap_ramp_indices) else None
            dlog_k = float(self.drag_params[3 + r_idx]) if r_idx is not None else 0.0

            # Nominal transit time across the gap
            t_beg_nom = float(self.timeAtLength(gap_beg)[0])
            t_end_nom = float(self.timeAtLength(gap_end)[0])
            dt_nom = t_end_nom - t_beg_nom

            ht_beg = float(self.heightAtLength(gap_beg)[0])
            ht_end = float(self.heightAtLength(gap_end)[0])

            # Build the early-jump and late-jump ln K profiles across the gap
            sel_gap = (self.len_grid >= gap_beg) & (self.len_grid <= gap_end)
            if (dlog_k > 1e-6) and (r_idx is not None):

                ramps_orig = self.ramps.copy()

                # Early jump: all gap growth occurs at the start of the gap (step at gap_beg)
                self.ramps[:, r_idx] = np.where(self.len_grid >= gap_beg, 1.0, 0.0)
                vel_early, time_early, _, _ = self._solve(self.drag_params)
                dt_early = float(np.interp(gap_end, self.len_grid, time_early) \
                    - np.interp(gap_beg, self.len_grid, time_early))

                # Late jump: all gap growth occurs at the end of the gap (step at gap_end)
                self.ramps[:, r_idx] = np.where(self.len_grid >= gap_end, 1.0, 0.0)
                vel_late, time_late, _, _ = self._solve(self.drag_params)
                dt_late = float(np.interp(gap_end, self.len_grid, time_late) \
                    - np.interp(gap_beg, self.len_grid, time_late))

                self.ramps = ramps_orig

                vel_lo = np.minimum(vel_early[sel_gap], vel_late[sel_gap])
                vel_hi = np.maximum(vel_early[sel_gap], vel_late[sel_gap])

            else:
                dt_early = dt_nom
                dt_late = dt_nom
                vel_lo = self.vel_grid[sel_gap].copy()
                vel_hi = self.vel_grid[sel_gap].copy()

            gap_info_list.append({
                'len_beg': float(gap_beg),
                'len_end': float(gap_end),
                'ht_beg': ht_beg,
                'ht_end': ht_end,
                'time_beg': t_beg_nom,
                'time_end': t_end_nom,
                'dt_nom': dt_nom,
                'dt_early': dt_early,
                'dt_late': dt_late,
                'dt_span': abs(dt_early - dt_late),
                'log_k_growth': dlog_k,
                'len_gap_grid': self.len_grid[sel_gap].copy(),
                'ht_gap_grid': self.ht_grid[sel_gap].copy(),
                'vel_lo': vel_lo,
                'vel_hi': vel_hi,
            })


        return gap_info_list



    def lengthAtTime(self, time_arr):
        """ Compute model lengths along the trajectory at the given times.

        Arguments:
            time_arr: [float or ndarray] Times (s).

        Return:
            [ndarray] Lengths along the trajectory (m).
        """

        return self._lenAtTime(np.atleast_1d(np.asarray(time_arr, dtype=np.float64)), self.vel_grid, \
            self.time_grid)



    def timeAtLength(self, len_arr):
        """ Compute model times at the given lengths along the trajectory.

        Arguments:
            len_arr: [float or ndarray] Lengths along the trajectory (m).

        Return:
            [ndarray] Model times (s).
        """

        return np.interp(np.atleast_1d(np.asarray(len_arr, dtype=np.float64)), self.len_grid, \
            self.time_grid)



    def velocityAtLength(self, len_arr):
        """ Compute model speeds at the given lengths along the trajectory.

        Arguments:
            len_arr: [float or ndarray] Lengths along the trajectory (m).

        Return:
            [ndarray] Speeds (m/s).
        """

        return np.interp(np.atleast_1d(np.asarray(len_arr, dtype=np.float64)), self.len_grid, self.vel_grid)



    def velocityAtTime(self, time_arr):
        """ Compute model speeds at the given times.

        Arguments:
            time_arr: [float or ndarray] Times (s).

        Return:
            [ndarray] Speeds (m/s).
        """

        return np.interp(np.atleast_1d(np.asarray(time_arr, dtype=np.float64)), self.time_grid, \
            self.vel_grid)



    def ballisticCoeffAtLength(self, len_arr):
        """ Compute fitted ballistic coefficients K at the given lengths.

        Arguments:
            len_arr: [float or ndarray] Lengths along the trajectory (m).

        Return:
            [ndarray] Ballistic coefficients K (m^2/kg).
        """

        return np.exp(np.interp(np.atleast_1d(np.asarray(len_arr, dtype=np.float64)), self.len_grid, \
            self.log_k_grid))



    def heightAtLength(self, len_arr):
        """ Compute heights above mean sea level at the given lengths.

        Arguments:
            len_arr: [float or ndarray] Lengths along the trajectory (m).

        Return:
            [ndarray] Heights above mean sea level (m).
        """

        return np.interp(np.atleast_1d(np.asarray(len_arr, dtype=np.float64)), self.len_grid, self.ht_grid)



def _buildStationsTimeDict(traj):
    """ Build the station timing status dictionary following the Trajectory.py convention.

    Arguments:
        traj: [Trajectory object] Trajectory with initialized observations.

    Return:
        (stations_time_dict, stations_time_dict_copy, free_mask):
            - stations_time_dict: [OrderedDict] Maps station ID string to working offset (float) or True.
            - stations_time_dict_copy: [OrderedDict] Maps station ID string to supplied offset (float) or
                True.
            - free_mask: [ndarray] Boolean mask, True for used stations whose timing is estimated.
    """

    stations_time_dict = collections.OrderedDict()
    stations_time_dict_copy = collections.OrderedDict()
    fixed_copy = getattr(traj, 'fixed_time_offsets_copy', traj.fixed_time_offsets)
    station_ids = [str(obs.station_id) for obs in traj.observations]

    for obs, stat_id in zip(traj.observations, station_ids):
        if stat_id in traj.fixed_time_offsets:
            stations_time_dict[stat_id] = traj.fixed_time_offsets[stat_id]
            stations_time_dict_copy[stat_id] = fixed_copy.get(stat_id, traj.fixed_time_offsets[stat_id])
        else:
            stations_time_dict[stat_id] = True
            stations_time_dict_copy[stat_id] = True

    # Fix the used station with the most used points as the reference when no fixed times are given
    if len(traj.fixed_time_offsets) == 0:
        counts = [0 if obs.ignore_station else int(np.count_nonzero(obs.ignore_list == 0)) \
            for obs in traj.observations]
        ref_idx = int(np.argmax(counts))
        stations_time_dict[station_ids[ref_idx]] = 0.0
        stations_time_dict_copy[station_ids[ref_idx]] = 0.0

    free_mask = np.array([(not obs.ignore_station) and isinstance(stations_time_dict[stat_id], bool) \
        for obs, stat_id in zip(traj.observations, station_ids)], dtype=bool)

    return stations_time_dict, stations_time_dict_copy, free_mask



def estimateDragTiming(traj, weights, drop_grid_func=None, penalty=1.0, gap_penalty_factor=1.0, \
    n_knots=None, check_clock_outliers=True, params_init=None, res_scale_init=None, \
    max_k0=DRAG_TRAJ_MAX_K0):
    """ Estimate station timing offsets and initial velocity using the joint physical drag fit.

    Unlike Trajectory.estimateTimingAndVelocity, this does not require time or length overlap between
    stations and never drops a station for lacking pairwise overlap.

    Arguments:
        traj: [Trajectory object] Trajectory with projected lengths (obs.state_vect_dist) computed.
        weights: [list] Statistical weights for each station.

    Keyword arguments:
        drop_grid_func: [callable] Optional callable returning gravity drop vectors on len_grid (m). None
            by default.
        penalty: [float] Regularization weight on ln K growth. 1.0 by default.
        gap_penalty_factor: [float] Relative penalty weight on gap ramps. 1.0 by default.
        n_knots: [int] Target number of observed ln K segments. None by default.
        check_clock_outliers: [bool] If True and more than 2 used stations exist, ignore a station whose
            clock offset hits max_toffset and exceeds 1.1*max_toffset when unconstrained. True by default.
        params_init: [ndarray] Optional warm-start parameter vector for DragTimingFit. None by default.
        res_scale_init: [float] Optional residual scale (m) paired with params_init. None by default.
        max_k0: [float] Upper bound on the initial ballistic coefficient K_0 (m^2/kg), or None.
            DRAG_TRAJ_MAX_K0 by default.

    Return:
        (success, drag_fit, time_diffs, v_init, v_init_stddev, velocity_fit):
            - success: [bool] True if the fit succeeded.
            - drag_fit: [DragTimingFit object] Fitted drag-timing model, or None on failure.
            - time_diffs: [ndarray] Estimated timing offsets for all stations (s).
            - v_init: [float] Estimated initial velocity at the beginning of the trajectory (m/s).
            - v_init_stddev: [float] Estimated 1-sigma uncertainty of v_init (m/s).
            - velocity_fit: [list] [v_init, intercept] linear velocity fit for lag computation.
    """

    weights = checkWeights(traj.observations, weights)
    stations_time_dict, stations_time_dict_copy, free_mask = _buildStationsTimeDict(traj)
    traj.stations_time_dict = stations_time_dict
    traj.stations_time_dict_copy = stations_time_dict_copy

    if not traj.estimate_timing_vel:
        free_mask = np.zeros(len(traj.observations), dtype=bool)

    st_times = [obs.time_data[obs.ignore_list == 0] if not obs.ignore_station else np.array([]) \
        for obs in traj.observations]
    st_lens = [obs.state_vect_dist[obs.ignore_list == 0] if not obs.ignore_station else np.array([]) \
        for obs in traj.observations]

    try:
        drag_fit = DragTimingFit(st_times, st_lens, weights, free_mask, traj.state_vect_mini, \
            traj.radiant_eci_mini, traj.jdt_ref, gravity_factor=traj.gravity_factor, \
            max_toffset=traj.max_toffset, n_knots=n_knots, penalty=penalty, \
            gap_penalty_factor=gap_penalty_factor, drop_grid_func=drop_grid_func, \
            air_rotation=getattr(traj, 'air_rotation', True), params_init=params_init, \
            res_scale_init=res_scale_init, max_k0=max_k0)
    except ValueError as err:
        if traj.verbose:
            print("Drag timing fit failed: {:s}".format(str(err)))
        return False, None, np.zeros(len(traj.observations)), 0.0, 0.0, [0.0, 0.0]


    # Ignore a station whose clock needs a larger offset than max_toffset when > 2 stations are used
    if check_clock_outliers and traj.estimate_timing_vel and (numStationsNotIgnored(traj.observations) > 2):

        at_limit = np.any(np.isclose(np.abs(drag_fit.time_offsets[free_mask]), traj.max_toffset, \
            atol=1e-4))
        if at_limit:

            # Test with twice the limit to see if an offset truly exceeds 1.1*max_toffset
            try:
                wide_fit = DragTimingFit(st_times, st_lens, weights, free_mask, traj.state_vect_mini, \
                    traj.radiant_eci_mini, traj.jdt_ref, gravity_factor=traj.gravity_factor, \
                    max_toffset=2.0*traj.max_toffset, n_knots=n_knots, penalty=penalty, \
                    gap_penalty_factor=gap_penalty_factor, drop_grid_func=drop_grid_func, \
                    air_rotation=getattr(traj, 'air_rotation', True), max_k0=max_k0)
                exceeds = np.any(np.abs(wide_fit.time_offsets[free_mask]) > 1.1*traj.max_toffset)
            except ValueError:
                exceeds = False

            if exceeds:

                best_drop = None
                best_rms = np.inf
                for i, obs in enumerate(traj.observations):

                    if obs.ignore_station or (str(obs.station_id) in traj.fixed_time_offsets):
                        continue

                    trial_weights = np.array(weights, dtype=np.float64)
                    trial_weights[i] = 0.0
                    trial_times = [arr if j != i else np.array([]) for j, arr in enumerate(st_times)]
                    trial_lens = [arr if j != i else np.array([]) for j, arr in enumerate(st_lens)]
                    trial_free = free_mask.copy()
                    trial_free[i] = False

                    # Ensure at least one remaining station is fixed as reference
                    if np.all(trial_free[trial_weights > 0]):
                        rem = [j for j in range(len(traj.observations)) if trial_weights[j] > 0]
                        ref_j = max(rem, key=lambda j: len(trial_times[j]))
                        trial_free[ref_j] = False

                    try:
                        trial_fit = DragTimingFit(trial_times, trial_lens, trial_weights, trial_free, \
                            traj.state_vect_mini, traj.radiant_eci_mini, traj.jdt_ref, \
                            gravity_factor=traj.gravity_factor, max_toffset=traj.max_toffset, \
                            n_knots=n_knots, penalty=penalty, gap_penalty_factor=gap_penalty_factor, \
                            drop_grid_func=drop_grid_func, \
                            air_rotation=getattr(traj, 'air_rotation', True), max_k0=max_k0)
                    except ValueError:
                        continue

                    if np.any(np.isclose(np.abs(trial_fit.time_offsets[trial_free]), traj.max_toffset, \
                        atol=1e-4)):
                        continue

                    if trial_fit.res_rms_m < best_rms:
                        best_rms = trial_fit.res_rms_m
                        best_drop = i

                if best_drop is not None:
                    off_obs = traj.observations[best_drop]
                    off_obs.ignore_station = True
                    off_obs.ignore_list = np.ones(len(off_obs.time_data), dtype=np.uint8)
                    if traj.verbose:
                        print("Station {:s} ignored, its clock is off by more than {:.2f} s".format( \
                            str(off_obs.station_id), traj.max_toffset))
                    return estimateDragTiming(traj, weights, drop_grid_func=drop_grid_func, \
                        penalty=penalty, gap_penalty_factor=gap_penalty_factor, n_knots=n_knots, \
                        check_clock_outliers=True, max_k0=max_k0)


    # Collect timing offsets and update station times
    time_diffs = drag_fit.time_offsets.copy()
    for obs, t_diff in zip(traj.observations, time_diffs):

        if traj.verbose and traj.estimate_timing_vel:
            print("STATION {:s} TIME OFFSET = {:.6f} s".format(str(obs.station_id), t_diff))

        if np.isnan(t_diff) or np.isinf(t_diff):
            continue

        obs.time_data = obs.time_data + t_diff
        if obs.excluded_time is not None:
            obs.excluded_time = [ex_time + t_diff for ex_time in obs.excluded_time]


    # Set initial velocity and linear velocity fit tangent to the beginning of the drag model
    v_init = float(drag_fit.vel_grid[0])
    v_init_stddev = float(drag_fit.v_init_stddev)
    intercept = float(drag_fit.len_grid[0] - v_init*drag_fit.time_grid[0])
    velocity_fit = [v_init, intercept]

    # Compute lag relative to the initial-velocity line for every station
    for obs in traj.observations:
        obs.lag = obs.state_vect_dist - traj.gravityLengthAlongPath(traj.observations, obs.time_data) \
            - lineFunc(obs.time_data, *velocity_fit)
        obs.lag_line = list(velocity_fit)
        obs.v_init = v_init
        obs.v_init_stddev = v_init_stddev

    # Store timing residual metrics consistent with Trajectory.py
    traj.timing_res = drag_fit.res_mean_abs_s
    traj.timing_stddev = drag_fit.res_rms_s

    return True, drag_fit, time_diffs, v_init, v_init_stddev, velocity_fit



def _normalizeTrajectoryTimes(traj):
    """ Shift all observation times so the earliest used point starts at 0 s, updating jdt_ref and
        meastype 1 RA/Dec measurements identically to Trajectory.run.

    Arguments:
        traj: [Trajectory object] Trajectory whose times are normalized in place.
    """

    t0 = min([obs.time_data[0] for obs in traj.observations if (not obs.ignore_station) \
        or (not np.all(obs.ignore_list))])

    if t0 != 0.0:

        for obs in traj.observations:
            obs.time_data = obs.time_data - t0
            if getattr(obs, 'excluded_time', None) is not None:
                obs.excluded_time = [ex_time - t0 for ex_time in obs.excluded_time]

        if getattr(traj, 'gravity_vel', None) is not None:
            traj.gravity_vel = (traj.gravity_vel[0] - t0, traj.gravity_vel[1])

        traj.jdt_ref = traj.jdt_ref + t0/86400.0

        if traj.meastype == 1:
            for obs in traj.observations:
                if getattr(obs, 'radec_jd', None) is not None:
                    continue
                obs.meas1, obs.meas2 = altAz2RADec_vect(obs.azim_data, obs.elev_data, traj.jdt_ref, \
                    obs.lat, obs.lon)

    for i, obs in enumerate(traj.observations):
        if obs.excluded_indx_range:
            continue
        if obs.time_data[0] == 0.0:
            traj.t_ref_station = i
            break



def _fitLineOfSight(traj, weights, use_gravity):
    """ Run line-of-sight angle minimization reusing the exact Nelder-Mead line solver of
        wmpl.Trajectory.Trajectory.

    Arguments:
        traj: [Trajectory object] Trajectory being solved.
        weights: [list] Station weights.
        use_gravity: [bool] Apply gravity drop during the line-of-sight fit.

    Return:
        (state_vect_mini, radiant_eci_mini, los_success):
            - state_vect_mini: [ndarray] Fitted ECI state vector at trajectory beginning (m).
            - radiant_eci_mini: [ndarray] Fitted unit ECI radiant vector.
            - los_success: [bool] True if minimization succeeded.
    """

    p0 = np.r_[traj.state_vect, traj.best_conv_inter.radiant_eci]
    if (traj.state_vect_mini is not None) and (traj.radiant_eci_mini is not None):
        p0 = np.r_[traj.state_vect_mini, traj.radiant_eci_mini]

    minimize_args = (traj.observations, weights, use_gravity, traj.gravity_factor, traj.v0z, \
        traj.gravity_r0, traj.gravity_vel, traj.air_rotation)

    minimize_solution = scipy.optimize.minimize(minimizeAngleCost, p0, args=minimize_args, \
        method=nelderMeadLine)

    los_restarts = getattr(traj, 'los_restarts', 0)
    for _ in range(los_restarts):
        restarted = scipy.optimize.minimize(minimizeAngleCost, minimize_solution.x, args=minimize_args, \
            method=nelderMeadLine)
        if restarted.success or (not minimize_solution.success):
            minimize_solution = restarted

    for _ in range(LOS_CONTINUE_RUNS):
        if minimize_solution.success:
            break
        restart = np.r_[moveStateVector(minimize_solution.x[:3], minimize_solution.x[3:], \
            traj.observations), vectNorm(minimize_solution.x[3:])]
        minimize_solution = scipy.optimize.minimize(minimizeAngleCost, restart, args=minimize_args, \
            method=nelderMeadLine)

    unbounded_solution = minimize_solution
    if stateVectorDiverged(minimize_solution.x[:3], traj.state_vect):

        bounds = [sorted([0.9*val, 1.1*val]) for val in traj.state_vect] \
            + [sorted([0.75*val, 1.25*val]) for val in traj.best_conv_inter.radiant_eci]

        minimize_solution = scipy.optimize.minimize(minimizeAngleCost, \
            np.r_[traj.state_vect, traj.best_conv_inter.radiant_eci], args=minimize_args, bounds=bounds, \
            method='SLSQP')

        if unbounded_solution.success and ((not minimize_solution.success) \
            or (unbounded_solution.fun < minimize_solution.fun)):
            minimize_solution = unbounded_solution

    if minimize_solution.success:
        sv_mini, rad_mini = np.hsplit(minimize_solution.x, 2)
        rad_mini = vectNorm(rad_mini)
        sv_mini = moveStateVector(sv_mini, rad_mini, traj.observations)
    else:
        rad_mini = vectNorm(traj.best_conv_inter.radiant_eci)
        sv_mini = moveStateVector(traj.state_vect, rad_mini, traj.observations)

    return sv_mini, rad_mini, bool(minimize_solution.success)



def _computeIntersectingPlanes(traj):
    """ Compute all pairwise plane intersections, the best convergence seed and station weights.

    Arguments:
        traj: [Trajectory object] Trajectory being solved.

    Return:
        weights: [list] Statistical weights for each station.
    """

    traj.intersection_list = []
    nonignored = [obs for obs in traj.observations if not obs.ignore_station]

    for i, obs1 in enumerate(nonignored):
        for obs2 in nonignored[i + 1:]:
            traj.intersection_list.append(PlaneIntersection(obs1, obs2))

    radiant_sum = np.zeros(3)
    weights_sum = 1e-10
    for inter in traj.intersection_list:
        radiant_sum += inter.weight*inter.radiant_eci
        weights_sum += inter.weight

    traj.avg_radiant = vectNorm(radiant_sum/weights_sum)
    traj.best_conv_inter = max(traj.intersection_list, key=lambda x: x.conv_angle)
    traj.state_vect = moveStateVector(traj.best_conv_inter.cpa_eci, traj.best_conv_inter.radiant_eci, \
        traj.observations)

    traj.incident_angles = traj.calcStationIncidentAngles(traj.state_vect, \
        traj.best_conv_inter.radiant_eci, traj.observations)
    for obs, inc_ang in zip(traj.observations, traj.incident_angles):
        obs.incident_angle = inc_ang

    if numStationsNotIgnored(traj.observations) > 2:
        weights = [np.sin(w)**2 if not obs.ignore_station else 0.0 \
            for w, obs in zip(traj.incident_angles, traj.observations)]
    else:
        weights = [1.0 if not obs.ignore_station else 0.0 for obs in traj.observations]

    for w, obs in zip(weights, traj.observations):
        obs.weight = w

    return weights



def _reinitObservations(traj, weights=None, incident_angles=None):
    """ Reinitialize all ObservedPoints in traj so JD_data, stat_eci_los and meas_eci_los reflect updated
        station times, restoring station weights and incident angles when provided.

    Arguments:
        traj: [Trajectory object] Trajectory whose observations are reinitialized in place.

    Keyword arguments:
        weights: [list] Optional station weights to restore. None by default.
        incident_angles: [list] Optional station incident angles to restore. None by default.
    """

    saved_fixed_copy = copy.deepcopy(getattr(traj, 'fixed_time_offsets_copy', {}))
    temp_obs = copy.deepcopy(traj.observations)
    traj.observations = []
    for obs in temp_obs:
        traj.infillWithObs(obs)

    for station in traj.fixed_time_offsets:
        traj.fixed_time_offsets[station] = 0.0
    traj.fixed_time_offsets_copy = saved_fixed_copy

    if weights is not None:
        for w, obs in zip(weights, traj.observations):
            obs.weight = w

    if incident_angles is not None:
        for inc_ang, obs in zip(incident_angles, traj.observations):
            obs.incident_angle = inc_ang



def _makeDropGridFunc(traj, drag_fit_prev):
    """ Build a callable that evaluates gravity drop vectors along a length grid using the previous drag
        model speeds.

    Arguments:
        traj: [Trajectory object] Trajectory with gravity_vel, state_vect_mini and radiant_eci_mini set.
        drag_fit_prev: [DragTimingFit object] Previous pass drag-timing fit.

    Return:
        [callable] Function mapping len_grid (m) to ECI drop vectors (N, 3) (m), or None.
    """

    if (not traj.gravity_correction) or (traj.gravity_vel is None) or (drag_fit_prev is None):
        return None

    t0 = dropTimeOrigin(traj.observations)
    eci0 = gravityDropOrigin(traj.observations, traj.state_vect_mini, traj.radiant_eci_mini, \
        traj.gravity_vel, t0)

    def _dropOnLenGrid(len_grid):
        times_grid = drag_fit_prev.timeAtLength(len_grid)
        return gravityDropVectors(eci0, traj.radiant_eci_mini, times_grid - t0, traj.gravity_vel, t0, \
            traj.gravity_factor, air_rotation=getattr(traj, 'air_rotation', True))

    return _dropOnLenGrid



def _attachDragMCUncertainties(traj, mc_traj_list):
    """ Compute Monte Carlo ensemble uncertainties on the drag model velocity, ballistic coefficient K and
        station timing offsets, and store them on traj.drag_timing_fit.

    Arguments:
        traj: [Trajectory object] Main solved trajectory with traj.drag_timing_fit.
        mc_traj_list: [list] Successful Monte Carlo Trajectory objects.
    """

    drag_fit = getattr(traj, 'drag_timing_fit', None)
    if (drag_fit is None) or (not mc_traj_list):
        return

    mc_fits = [getattr(t, 'drag_timing_fit', None) for t in mc_traj_list]
    mc_fits = [f for f in mc_fits if f is not None]
    if len(mc_fits) < 2:
        return

    vel_samples = np.array([f.velocityAtLength(drag_fit.len_grid) for f in mc_fits])
    k_samples = np.array([f.ballisticCoeffAtLength(drag_fit.len_grid) for f in mc_fits])

    drag_fit.mc_vel_std = np.std(vel_samples, axis=0)
    drag_fit.mc_vel_lo, drag_fit.mc_vel_hi = np.percentile(vel_samples, [15.865, 84.135], axis=0)
    drag_fit.mc_k_lo, drag_fit.mc_k_hi = np.percentile(k_samples, [15.865, 84.135], axis=0)

    td_list = [t.time_diffs_final for t in mc_traj_list \
        if (t.time_diffs_final is not None) and (len(t.time_diffs_final) == len(traj.observations))]
    if len(td_list) >= 2:
        drag_fit.mc_time_diffs_std = np.std(np.array(td_list), axis=0)



def _asDragSubclass(traj, max_iter, dt_tol, penalty, gap_penalty_factor, n_knots, max_k0):
    """ Convert a plain Trajectory instance into a TrajectoryDragFit view so Trajectory.py's
        monteCarloTrajectory() dispatches each MC run through solveTrajectoryDragFit().

    Arguments:
        traj: [Trajectory object] Solved trajectory.
        max_iter: [int] Maximum coupled iterations.
        dt_tol: [float] Timing convergence tolerance (s).
        penalty: [float] Drag penalty weight.
        gap_penalty_factor: [float] Gap ramp penalty multiplier.
        n_knots: [int] Target observed knot count.
        max_k0: [float] Upper bound on initial ballistic coefficient K_0 (m^2/kg), or None.

    Return:
        [TrajectoryDragFit object] Shallow copy with TrajectoryDragFit class and drag solver settings.
    """

    mc_base = copy.copy(traj)
    mc_base.__class__ = TrajectoryDragFit
    mc_base.drag_max_iter = max_iter
    mc_base.drag_dt_tol = dt_tol
    mc_base.drag_penalty = penalty
    mc_base.gap_penalty_factor = gap_penalty_factor
    mc_base.drag_n_knots = n_knots
    mc_base.drag_max_k0 = max_k0

    return mc_base



def solveTrajectoryDragFit(traj, max_iter=DRAG_TRAJ_MAX_ITER, dt_tol=DRAG_TRAJ_DT_TOL, penalty=1.0, \
    gap_penalty_factor=1.0, n_knots=None, max_k0=DRAG_TRAJ_MAX_K0, _rerun_bad_picks=False, _mc_run=False, \
    _orig_obs=None):
    """ Solve a meteor trajectory by coupling the line-of-sight geometry with the physical drag fit.

    Reuses the geometry, gravity-drop, coordinate, residual, orbit, Monte Carlo, report and plotting
    machinery of wmpl.Trajectory.Trajectory, while replacing pairwise overlap timing matching with
    DragTimingFit so that stations with no time overlap or disconnected station groups can be solved
    together.

    Arguments:
        traj: [Trajectory object] Initialized Trajectory object populated via infillTrajectory().

    Keyword arguments:
        max_iter: [int] Maximum number of coupled LoS and drag-timing iterations. DRAG_TRAJ_MAX_ITER by
            default.
        dt_tol: [float] Stop iterating after the gravity pass once the largest incremental timing offset is
            below this threshold (s). DRAG_TRAJ_DT_TOL by default.
        penalty: [float] Regularization weight on the growth of ln K. 1.0 by default.
        gap_penalty_factor: [float] Relative penalty multiplier on gap ramps. 1.0 by default.
        n_knots: [int] Target number of observed ln K segments. None by default (auto).
        max_k0: [float] Upper bound on the initial ballistic coefficient K_0 (m^2/kg), or None.
            DRAG_TRAJ_MAX_K0 by default.
        _rerun_bad_picks: [bool] Internal flag set on the post-outlier-rejection pass. False by default.
        _mc_run: [bool] Internal flag set during Monte Carlo runs. False by default.
        _orig_obs: [list] Internal noise-free observations used for dynamics/timing in MC runs. None by
            default.

    Return:
        traj: [Trajectory object] Solved trajectory object with traj.drag_timing_fit attached, or None if
            solving failed.
    """

    if numStationsNotIgnored(traj.observations) < 2:
        if traj.verbose:
            print("At least 2 sets of measurements from 2 stations are needed to estimate the trajectory!")
        return None

    # Ensure backward-compatible attributes exist
    if not hasattr(traj, 'gravity_r0'):
        traj.gravity_r0 = None
    if not hasattr(traj, 'gravity_vel'):
        traj.gravity_vel = None
    if not hasattr(traj, 'air_rotation'):
        traj.air_rotation = True

    # Capture warm-start drag parameters and residual scale from the main solution in MC runs
    warm_params = None
    warm_scale = None
    prev_drag = getattr(traj, 'drag_timing_fit', None)
    if _mc_run and (prev_drag is not None):
        warm_params = np.r_[prev_drag.drag_params, np.zeros(prev_drag.n_free)]
        warm_scale = prev_drag.res_scale_m

    _normalizeTrajectoryTimes(traj)
    _reinitObservations(traj)

    # In an MC run, keep a clean copy of the noise-free observations aligned to the current jdt_ref
    orig_obs_work = None
    if _mc_run and (_orig_obs is not None):
        orig_obs_work = copy.deepcopy(_orig_obs)
        for obs_o, obs_n in zip(orig_obs_work, traj.observations):
            obs_o.time_data = np.copy(obs_n.time_data)
            obs_o.ignore_station = obs_n.ignore_station
            obs_o.ignore_list = np.copy(obs_n.ignore_list)

    weights = _computeIntersectingPlanes(traj)

    # Preserve supplied fixed offsets and cumulative input-data timing baselines across bad-pick reruns and
    #   MC runs so report/pickle offsets and MC timing statistics share one consistent baseline
    if (_rerun_bad_picks or _mc_run) and (getattr(traj, 'time_diffs_final', None) is not None) \
        and (len(traj.time_diffs_final) == len(traj.observations)):
        cum_offsets = np.array(traj.time_diffs_final, dtype=np.float64)
    else:
        fixed_copy = getattr(traj, 'fixed_time_offsets_copy', {})
        cum_offsets = np.array([float(fixed_copy.get(str(obs.station_id), 0.0)) \
            for obs in traj.observations], dtype=np.float64)

    # Cap MC runs to 2 passes (straight + gravity) since clocks are already aligned from the main solution
    n_passes = 2 if _mc_run else max_iter

    drag_fit = None
    for it in range(n_passes):

        if it > 0:
            _reinitObservations(traj, weights=weights, incident_angles=traj.incident_angles)

        use_gravity = (it > 0) and traj.gravity_correction

        if use_gravity and (traj.gravity_vel is not None):
            eci0, traj.gravity_r0 = gravityDropReference(traj.observations, traj.state_vect_mini, \
                traj.radiant_eci_mini, v_init=traj.v_init)
            traj.calcV0z(eci0, traj.radiant_eci_mini)

        # Fit the 3D trajectory line (pure cross-track LoS on noisy obs in MC runs)
        traj.state_vect_mini, traj.radiant_eci_mini, traj.los_mini_status = _fitLineOfSight(traj, weights, \
            use_gravity)

        # In MC runs, switch to noise-free observations before projecting lengths and fitting drag/timing
        noisy_obs_backup = None
        if _mc_run and (orig_obs_work is not None):
            noisy_obs_backup = list(traj.observations)
            traj.observations = orig_obs_work
            _reinitObservations(traj, weights=weights, incident_angles=traj.incident_angles)
            orig_obs_work = traj.observations

        # Project all observations onto the fitted trajectory line
        traj.calcVelocity(traj.state_vect_mini, traj.radiant_eci_mini, traj.observations, weights, \
            gravity=use_gravity)

        # Warm-start Pass 2+ from the previous pass's drag parameters with zero incremental dt
        if drag_fit is not None:
            warm_params = np.r_[drag_fit.drag_params, np.zeros(drag_fit.n_free)]
            warm_scale = drag_fit.res_scale_m

        # Fit the physical drag model and station timing offsets jointly
        drop_func = _makeDropGridFunc(traj, drag_fit) if use_gravity else None
        success, drag_fit, dt_step, v_init, v_init_std, vel_fit = estimateDragTiming(traj, weights, \
            drop_grid_func=drop_func, penalty=penalty, gap_penalty_factor=gap_penalty_factor, \
            n_knots=n_knots, check_clock_outliers=((it == 0) and (not _mc_run)), params_init=warm_params, \
            res_scale_init=warm_scale, max_k0=max_k0)

        traj.timing_minimization_successful = success
        if not success:
            if traj.verbose:
                print("Unable to minimize drag-coupled timing")
            return None

        cum_offsets = cum_offsets + dt_step
        traj.time_diffs = dt_step
        traj.v_init = v_init
        traj.v_init_stddev = v_init_std
        traj.velocity_fit = vel_fit

        # Update speeds for the gravity drop across the path
        t_rel = np.linspace(drag_fit.time_grid[0], drag_fit.time_grid[-1], GRAVITY_DROP_POINTS)
        v_rel = np.interp(t_rel, drag_fit.time_grid, drag_fit.vel_grid)
        traj.gravity_vel = (t_rel, v_rel)

        # Zero the weight of any station ignored during clock-outlier rejection
        weights = [w if not obs.ignore_station else 0.0 for w, obs in zip(weights, traj.observations)]
        for w, obs in zip(weights, traj.observations):
            obs.weight = w

        # In MC runs, carry updated timings back to noisy observations for the next LoS pass
        if _mc_run and (noisy_obs_backup is not None):
            for obs_o, obs_n in zip(traj.observations, noisy_obs_backup):
                obs_n.time_data = np.copy(obs_o.time_data)
            orig_obs_work = list(traj.observations)
            traj.observations = noisy_obs_backup

        if numStationsNotIgnored(traj.observations) < 2:
            return None

        # Stop once the gravity pass (or second pass) has run and incremental time shifts converged
        min_passes = 2 if (traj.gravity_correction or traj.estimate_timing_vel) else 1
        if (it + 1 >= min_passes) and (np.max(np.abs(dt_step)) < dt_tol):
            break


    # In MC runs, switch to noise-free observations for final derived quantities and orbit
    if _mc_run and (orig_obs_work is not None):
        traj.observations = orig_obs_work

    # Refresh ObservedPoints with final converged timings and recompute all derived quantities
    _reinitObservations(traj, weights=weights, incident_angles=traj.incident_angles)
    traj.time_diffs_final = cum_offsets
    traj.drag_timing_fit = drag_fit
    traj.gravity_t0 = dropTimeOrigin(traj.observations)

    if traj.gravity_correction:
        eci0, traj.gravity_r0 = gravityDropReference(traj.observations, traj.state_vect_mini, \
            traj.radiant_eci_mini, v_init=traj.v_init)
        traj.calcV0z(eci0, traj.radiant_eci_mini)

    traj.calcVelocity(traj.state_vect_mini, traj.radiant_eci_mini, traj.observations, weights, \
        gravity=traj.gravity_correction)
    traj.calcLag(traj.observations, velocity_fit=traj.velocity_fit)
    traj.jacchia_fit = traj.fitJacchiaLag(traj.observations)
    traj.calcLLA(traj.state_vect_mini, traj.radiant_eci_mini, traj.observations)

    # Optional initial velocity override above a fixed height
    if traj.v_init_ht is not None:
        v_ht_avg, intercept_ht_avg, v_ht_stddev = traj.calcAvgVelocityAboveHt(traj.observations, \
            1000.0*traj.v_init_ht, weights)
        if v_ht_avg is not None:
            traj.v_init = v_ht_avg
            traj.v_init_stddev = v_ht_stddev
            traj.velocity_fit = [traj.v_init, intercept_ht_avg]
            traj.calcLag(traj.observations, velocity_fit=traj.velocity_fit)
            traj.jacchia_fit = traj.fitJacchiaLag(traj.observations)

    traj.calcECIEqAltAz(traj.state_vect_mini, traj.radiant_eci_mini, traj.observations)
    traj.calcAllResiduals(traj.state_vect_mini, traj.radiant_eci_mini, traj.observations)
    traj.calcAbsMagnitudes()

    # Reject angular outlier picks and re-solve once, matching Trajectory.run
    if traj.filter_picks and (not _rerun_bad_picks) and (not _mc_run):

        picks_rejected = 0
        for obs in traj.observations:

            if obs.ignore_station:
                continue

            used = obs.ignore_list == 0
            if np.count_nonzero(used) < 4:
                continue

            good_picks = used & (obs.ang_res < (np.mean(obs.ang_res[used]) \
                + traj.reject_n_sigma_outliers*obs.ang_res_std))
            if np.count_nonzero(good_picks) < 4:
                continue

            rejected = used & (~good_picks)
            if np.any(rejected):
                picks_rejected += int(np.count_nonzero(rejected))
                obs.ignore_list[rejected] = 1

        if picks_rejected:
            if traj.verbose:
                print("Updating the drag-coupled solution after rejecting {:d} bad picks...".format( \
                    picks_rejected))
            return solveTrajectoryDragFit(traj, max_iter=max_iter, dt_tol=dt_tol, penalty=penalty, \
                gap_penalty_factor=gap_penalty_factor, n_knots=n_knots, max_k0=max_k0, \
                _rerun_bad_picks=True)


    # Compute orbit
    if traj.calc_orbit:
        traj.v_avg, traj.state_vect_avg, traj.jd_avg = traj.calcAverages(traj.observations)
        traj.orbit_los_success = bool(traj.los_mini_status)
        traj.orbit = calcOrbit(traj.radiant_eci_mini, traj.v_init, traj.v_avg, traj.state_vect_mini, \
            traj.rbeg_jd, stations_fixed=(not traj.orbit_los_success), \
            reference_init=traj.orbit_los_success, v_init_stddev_direct=traj.v_init_stddev)

        if traj.verbose and (not _mc_run):
            print(traj.orbit.__repr__(v_init_ht=traj.v_init_ht))

    # Return early for Monte Carlo worker runs
    if _mc_run:
        return traj

    # Run Monte Carlo uncertainty estimation reusing Trajectory.py's parallel MC engine
    if traj.monte_carlo:

        mc_base = _asDragSubclass(traj, max_iter, dt_tol, penalty, gap_penalty_factor, n_knots, max_k0)
        traj_best, uncertainties = monteCarloTrajectory(mc_base, mc_runs=traj.mc_runs, \
            mc_pick_multiplier=traj.mc_pick_multiplier, noise_sigma=traj.mc_noise_std, \
            geometric_uncert=traj.geometric_uncert, plot_results=traj.save_results, \
            mc_cores=traj.mc_cores, max_runs=traj.mc_runs_max)

        if uncertainties is not None:
            _attachDragMCUncertainties(traj, uncertainties.mc_traj_list)

            traj_uncer = copy.deepcopy(uncertainties)
            traj_uncer.mc_traj_list = []
            traj_best.uncertainties = traj_uncer
            traj_best.uncertanties = traj_uncer

        traj = copyUncertainties(traj_best, traj)


    # Print coverage and gap summary when verbose
    if traj.verbose:
        if drag_fit.single_station_spans:
            print("Single-station (no overlap) spans:")
            for i, (s_beg, s_end) in enumerate(drag_fit.single_station_spans):
                print("  Span {:d}: L = {:.2f} - {:.2f} km (ht {:.2f} - {:.2f} km)".format(i + 1, \
                    s_beg/1000.0, s_end/1000.0, drag_fit.heightAtLength(s_beg)[0]/1000.0, \
                    drag_fit.heightAtLength(s_end)[0]/1000.0))

        if drag_fit.gap_uncertainties:
            print("Unobserved gaps bridged by the drag model:")
            for i, gap in enumerate(drag_fit.gap_uncertainties):
                print(("  Gap {:d}: L = {:.2f} - {:.2f} km (ht {:.2f} - {:.2f} km), dt = {:.3f} s, " \
                    + "assumed d(ln K) = {:.2f}, early/late jump dt span = {:.1f} ms").format(i + 1, \
                    gap['len_beg']/1000.0, gap['len_end']/1000.0, gap['ht_beg']/1000.0, \
                    gap['ht_end']/1000.0, gap['dt_nom'], gap['log_k_growth'], 1000.0*gap['dt_span']))


    # Save report, pickle, KML and plots if requested
    if traj.save_results:
        mkdirP(traj.output_dir)
        traj.saveReport(traj.output_dir, traj.file_name + '_report.txt', uncertainties=traj.uncertainties, \
            verbose=traj.verbose, save_results=True)
        savePickle(traj, traj.output_dir, traj.file_name + '_trajectory.pickle')

        if getattr(traj, 'save_kml', False):
            from wmpl.Utils.TrajectoryKML import trajectoryToKML
            kml_path = os.path.join(traj.output_dir, traj.file_name + '_3D.kml')
            trajectoryToKML(traj, kml_path, measured=getattr(traj, 'kml_measured', False))

    if traj.save_results or traj.show_plots:
        traj.savePlots(traj.output_dir, traj.file_name, show_plots=traj.show_plots)
        plotDragVelocity(traj, output_dir=traj.output_dir, file_name=traj.file_name, \
            show_plots=traj.show_plots, save_results=traj.save_results)
        plotDragTrajectory(traj, output_dir=traj.output_dir, file_name=traj.file_name, \
            show_plots=traj.show_plots, save_results=traj.save_results)

    return traj



class TrajectoryDragFit(Trajectory):
    """ Drop-in Trajectory subclass whose run() solves the trajectory using solveTrajectoryDragFit(). """

    def __init__(self, jdt_ref, max_iter=DRAG_TRAJ_MAX_ITER, dt_tol=DRAG_TRAJ_DT_TOL, drag_penalty=1.0, \
        gap_penalty_factor=1.0, drag_n_knots=None, max_k0=DRAG_TRAJ_MAX_K0, **kwargs):
        """ Initialize the drag-coupled trajectory solver.

        Arguments:
            jdt_ref: [float] Reference Julian date for the measurements.

        Keyword arguments:
            max_iter: [int] Maximum coupled LoS and drag-timing iterations. DRAG_TRAJ_MAX_ITER by default.
            dt_tol: [float] Convergence tolerance on station timing updates (s). DRAG_TRAJ_DT_TOL by default.
            drag_penalty: [float] Regularization weight on ln K growth. 1.0 by default.
            gap_penalty_factor: [float] Relative penalty multiplier on gap ramps. 1.0 by default.
            drag_n_knots: [int] Target number of observed ln K segments. None by default.
            max_k0: [float] Upper bound on the initial ballistic coefficient K_0 (m^2/kg), or None.
                DRAG_TRAJ_MAX_K0 by default.
            **kwargs: [dict] Keyword arguments forwarded to Trajectory.__init__.
        """

        super(TrajectoryDragFit, self).__init__(jdt_ref, **kwargs)

        self.drag_max_iter = max_iter
        self.drag_dt_tol = dt_tol
        self.drag_penalty = drag_penalty
        self.gap_penalty_factor = gap_penalty_factor
        self.drag_n_knots = drag_n_knots
        self.drag_max_k0 = max_k0
        self.drag_timing_fit = None



    def run(self, _mc_run=False, _orig_obs=None, **kwargs):
        """ Run the drag-coupled trajectory solver.

        Keyword arguments:
            _mc_run: [bool] Internal flag set by monteCarloTrajectory(). False by default.
            _orig_obs: [list] Internal noise-free observations passed by monteCarloTrajectory(). None by
                default.
            **kwargs: [dict] Unused compatibility arguments.

        Return:
            [TrajectoryDragFit object] Solved trajectory object, or None on failure.
        """

        return solveTrajectoryDragFit(self, max_iter=self.drag_max_iter, dt_tol=self.drag_dt_tol, \
            penalty=self.drag_penalty, gap_penalty_factor=self.gap_penalty_factor, \
            n_knots=self.drag_n_knots, max_k0=self.drag_max_k0, _mc_run=_mc_run, _orig_obs=_orig_obs)



def _shadeCoverageRegions(ax, drag_fit, axis='y', coord='height'):
    """ Shade and hatch unobserved gap (zero-station) bands on an axis.

    Arguments:
        ax: [matplotlib Axes] Target axes.
        drag_fit: [DragTimingFit object] Fitted drag-timing model.

    Keyword arguments:
        axis: [str] 'y' for horizontal bands (axhspan) or 'x' for vertical bands (axvspan). 'y' by default.
        coord: [str] 'height' (km) or 'time' (s) coordinate of the shaded axis. 'height' by default.
    """

    span_fn = ax.axhspan if axis == 'y' else ax.axvspan

    # Unobserved gaps (0 stations): orange hatched bands
    for idx, gap in enumerate(drag_fit.gap_uncertainties):
        c0 = gap['ht_beg']/1000.0 if coord == 'height' else gap['time_beg']
        c1 = gap['ht_end']/1000.0 if coord == 'height' else gap['time_end']
        lbl = 'Unobserved gap (assumed drag)' if idx == 0 else None
        span_fn(min(c0, c1), max(c0, c1), facecolor='tab:orange', edgecolor='tab:orange', hatch='xxx', \
            alpha=0.18, linewidth=0.0, label=lbl, zorder=1)



def plotDragVelocity(traj, output_dir='.', file_name=None, show_plots=True, save_results=False):
    """ Plot point-to-point station velocities overlaid with the drag fit curve, MC velocity uncertainty
        band (if available), and hatched unobserved gap regions.

    Saves as <file_name>_velocities.<plot_file_type>, matching Trajectory.savePlots's velocity plot layout.

    Arguments:
        traj: [Trajectory object] Solved trajectory with traj.drag_timing_fit attached.

    Keyword arguments:
        output_dir: [str] Output directory for saving the plot. '.' by default.
        file_name: [str] Base file name. None by default (uses traj.file_name).
        show_plots: [bool] Display the figure interactively. True by default.
        save_results: [bool] Save the figure to disk. False by default.

    Return:
        fig: [matplotlib Figure] Velocity figure, or None if traj has no drag_timing_fit.
    """

    drag_fit = getattr(traj, 'drag_timing_fit', None)
    if drag_fit is None:
        return None

    if file_name is None:
        file_name = traj.file_name

    vel_markers = ['x', '+', '.', '2']
    obs_ht_sorted = sorted(traj.observations, \
        key=lambda x: x.model_ht[0] if x.model_ht is not None else 0.0)
    colors, alpha = stationColors(len(obs_ht_sorted))
    obs_lim_list = [obs for obs in obs_ht_sorted if not obs.ignore_station] or obs_ht_sorted

    fig, ax1 = plt.subplots()

    # Shade and hatch unobserved gap bands in time
    _shadeCoverageRegions(ax1, drag_fit, axis='y', coord='time')

    vel_max = -np.inf
    vel_min = np.inf
    ht_max = -np.inf
    ht_min = np.inf
    t_max = -np.inf
    t_min = np.inf
    first_ignored = True

    for i, obs in enumerate(obs_ht_sorted):

        if np.any(obs.ignore_list):
            ig_t = obs.time_data[1:][obs.ignore_list[1:] > 0]
            ig_v = obs.velocities[1:][obs.ignore_list[1:] > 0]
            lbl = 'Ignored points' if first_ignored else None
            first_ignored = False
            ax1.scatter(ig_v/1000.0, ig_t, facecolors='none', edgecolors='k', zorder=4, s=30, label=lbl)

        ax1.scatter(obs.velocities[1:]/1000.0, obs.time_data[1:], marker=vel_markers[i%len(vel_markers)], \
            c=colors[i].reshape(1, -1), alpha=alpha, label="{:s}".format(str(obs.station_id)), zorder=3)

        if obs not in obs_lim_list:
            continue

        vel_max = max(float(np.max(obs.velocities[1:]/1000.0)), vel_max)
        vel_min = min(float(np.min(obs.velocities[1:]/1000.0)), vel_min)
        if np.any(np.isfinite(obs.meas_ht)):
            ht_max = max(float(np.nanmax(obs.meas_ht)), ht_max)
            ht_min = min(float(np.nanmin(obs.meas_ht)), ht_min)
        t_max = max(float(np.max(obs.time_data)), t_max)
        t_min = min(float(np.min(obs.time_data)), t_min)


    # Overlay the drag fit curve (solid where observed, dashed across unobserved gaps)
    in_gap = drag_fit.n_stations_grid == 0
    for sel, ls, lbl in [(~in_gap, 'solid', 'Drag fit'), (in_gap, 'dashed', 'Drag fit (gap)')]:
        if not np.any(sel):
            continue
        ax1.plot(np.where(sel, drag_fit.vel_grid/1000.0, np.nan), np.where(sel, drag_fit.time_grid, np.nan), \
            color='k', linestyle=ls, linewidth=1.8, label=lbl, zorder=5)

    # Overlay 1-sigma Monte Carlo velocity uncertainty band when available
    if (drag_fit.mc_vel_lo is not None) and (drag_fit.mc_vel_hi is not None):
        ax1.fill_betweenx(drag_fit.time_grid, drag_fit.mc_vel_lo/1000.0, drag_fit.mc_vel_hi/1000.0, \
            color='k', alpha=0.18, label='Drag fit 1 sigma (MC)', zorder=4)

    plt.title('Velocity')
    ax1.set_xlabel('Velocity (km/s)')
    ax1.set_ylabel('Time (s)')
    ax1.legend(prop={'size': LEGEND_TEXT_SIZE})
    ax1.grid()

    vel_min = max(vel_min, -20.0)
    vel_max = min(vel_max, 100.0)
    ax1.set_xlim([vel_min - 3.0, vel_max + 3.0])
    ax1.set_ylim([t_min, t_max])
    ax1.invert_yaxis()

    ax2 = ax1.twinx()
    ax2.set_ylim(ht_min/1000.0, ht_max/1000.0)
    ax2.set_ylabel('Height (km)')

    plt.tight_layout()

    if save_results:
        savePlot(plt, file_name + '_velocities.' + traj.plot_file_type, output_dir)

    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    return fig



def plotDragTrajectory(traj, output_dir='.', file_name=None, show_plots=True, save_results=False):
    """ Plot the separate drag-coupled trajectory diagnostic figure with unobserved gap (zero-station)
        regions clearly hatched and shaded.

    Panels:
        1. Height vs. Time (observations colored by station, solid/dashed drag model, hatched gaps).
        2. Velocity vs. Height (raw velocities, drag model, MC 1-sigma band, gap jump bounds).
        3. Lag vs. Time (raw lags and drag model lag with hatched gaps).
        4. Ballistic coefficient K vs. Height (log scale, solid/dashed K with 1-sigma uncertainty band).

    Arguments:
        traj: [Trajectory object] Solved trajectory with traj.drag_timing_fit attached.

    Keyword arguments:
        output_dir: [str] Output directory for saving the plot. '.' by default.
        file_name: [str] Base file name for saving the plot. None by default (uses traj.file_name).
        show_plots: [bool] Display the figure interactively. True by default.
        save_results: [bool] Save the figure to disk. False by default.

    Return:
        fig: [matplotlib Figure] Diagnostic figure, or None if traj has no drag_timing_fit.
    """

    drag_fit = getattr(traj, 'drag_timing_fit', None)
    if drag_fit is None:
        return None

    if file_name is None:
        file_name = traj.file_name

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax_ht, ax_vel = axes[0]
    ax_lag, ax_k = axes[1]

    # Shade and hatch unobserved gaps on all panels
    _shadeCoverageRegions(ax_ht, drag_fit, axis='x', coord='time')
    _shadeCoverageRegions(ax_vel, drag_fit, axis='y', coord='height')
    _shadeCoverageRegions(ax_lag, drag_fit, axis='y', coord='time')
    _shadeCoverageRegions(ax_k, drag_fit, axis='y', coord='height')

    obs_sorted = sorted(traj.observations, key=lambda x: x.model_ht[0] if x.model_ht is not None else 0.0)
    colors, alpha = stationColors(len(obs_sorted))

    # Plot raw measurements per station
    for i, obs in enumerate(obs_sorted):

        if obs.ignore_station:
            continue

        used = np.array(obs.ignore_list) == 0
        if not np.any(used):
            continue

        label = str(obs.station_id)
        ax_ht.scatter(obs.time_data[used], obs.model_ht[used]/1000.0, s=12, marker='x', color=colors[i], \
            alpha=alpha, label=label, zorder=3)

        used_vel = used.copy()
        used_vel[0] = False
        if np.any(used_vel):
            ax_vel.scatter(obs.velocities[used_vel]/1000.0, obs.model_ht[used_vel]/1000.0, s=12, \
                marker='x', color=colors[i], alpha=alpha, label=label, zorder=3)

        ax_lag.scatter(obs.lag[used], obs.time_data[used], s=12, marker='x', color=colors[i], alpha=alpha, \
            label=label, zorder=3)


    in_gap = drag_fit.n_stations_grid == 0
    t_grid = drag_fit.time_grid
    ht_grid_km = drag_fit.ht_grid/1000.0
    vel_grid_kms = drag_fit.vel_grid/1000.0
    k_grid = np.exp(drag_fit.log_k_grid)
    lag_grid = drag_fit.len_grid - traj.gravityLengthAlongPath(traj.observations, t_grid) \
        - lineFunc(t_grid, *traj.velocity_fit)

    # Plot solid model inside observed windows and dashed model inside unobserved gaps
    for sel, ls, lbl in [(~in_gap, 'solid', 'Drag model (observed)'), \
        (in_gap, 'dashed', 'Drag model (gap)')]:

        if not np.any(sel):
            continue

        t_p = np.where(sel, t_grid, np.nan)
        h_p = np.where(sel, ht_grid_km, np.nan)
        v_p = np.where(sel, vel_grid_kms, np.nan)
        l_p = np.where(sel, lag_grid, np.nan)
        k_p = np.where(sel, k_grid, np.nan)

        ax_ht.plot(t_p, h_p, color='k', linestyle=ls, linewidth=1.6, label=lbl, zorder=4)
        ax_vel.plot(v_p, h_p, color='k', linestyle=ls, linewidth=1.6, label=lbl, zorder=4)
        ax_lag.plot(l_p, t_p, color='k', linestyle=ls, linewidth=1.6, label=lbl, zorder=4)
        ax_k.plot(k_p, h_p, color='k', linestyle=ls, linewidth=1.6, label=lbl, zorder=4)


    # Plot 1-sigma MC velocity uncertainty band if available
    if (drag_fit.mc_vel_lo is not None) and (drag_fit.mc_vel_hi is not None):
        ax_vel.fill_betweenx(ht_grid_km, drag_fit.mc_vel_lo/1000.0, drag_fit.mc_vel_hi/1000.0, \
            color='k', alpha=0.18, label='1 sigma (MC)', zorder=3)

    # Plot 1-sigma K uncertainty band (MC ensemble when available, else Jacobian fit band)
    k_pos = k_grid[k_grid > 0]
    k_lo_lim = float(np.min(k_pos))*0.7 if len(k_pos) else 1e-6
    k_hi_lim = float(np.max(k_pos))*1.4 if len(k_pos) else 1.0
    if k_hi_lim < 2.0*k_lo_lim:
        k_mid = np.sqrt(k_lo_lim*k_hi_lim)
        k_lo_lim = k_mid/1.5
        k_hi_lim = k_mid*1.5

    if (drag_fit.mc_k_lo is not None) and (drag_fit.mc_k_hi is not None):
        ax_k.fill_betweenx(ht_grid_km, np.clip(drag_fit.mc_k_lo, k_lo_lim, k_hi_lim), \
            np.clip(drag_fit.mc_k_hi, k_lo_lim, k_hi_lim), color='0.5', alpha=0.30, \
            label='1 sigma (MC)', zorder=2)
    elif drag_fit.log_k_std_grid is not None:
        log_k_all = drag_fit.log_k_grid
        log_std_all = drag_fit.log_k_std_grid
        with np.errstate(invalid='ignore', over='ignore'):
            k_lo = np.exp(np.clip(log_k_all - log_std_all, np.log(k_lo_lim), np.log(k_hi_lim)))
            k_hi = np.exp(np.clip(log_k_all + log_std_all, np.log(k_lo_lim), np.log(k_hi_lim)))
        ax_k.fill_betweenx(ht_grid_km, k_lo, k_hi, where=(~in_gap), color='0.6', alpha=0.25, \
            label='1 sigma (fit)', zorder=2)

    # Annotate unobserved gaps and show early/late jump velocity envelopes
    for idx, gap in enumerate(drag_fit.gap_uncertainties):

        if (len(gap['ht_gap_grid']) > 1) and (gap['dt_span'] > 1e-6):
            env_lbl = 'Gap jump bound' if idx == 0 else None
            ax_vel.fill_betweenx(gap['ht_gap_grid']/1000.0, gap['vel_lo']/1000.0, gap['vel_hi']/1000.0, \
                color='tab:orange', alpha=0.35, label=env_lbl, zorder=2)

        t_mid = 0.5*(gap['time_beg'] + gap['time_end'])
        h_top_band = float(np.max(ht_grid_km)) - 0.06*(float(np.max(ht_grid_km)) - float(np.min(ht_grid_km)))
        gap_km = (gap['len_end'] - gap['len_beg'])/1000.0
        dt_ms = 1000.0*gap['dt_span']
        if dt_ms >= 0.5:
            gap_txt = "Gap {:.1f} km\n+/-{:.0f} ms".format(gap_km, 0.5*dt_ms)
        else:
            gap_txt = "Gap {:.1f} km".format(gap_km)
        ax_ht.text(t_mid, h_top_band, gap_txt, fontsize=8, ha='center', va='top', \
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))


    ax_ht.set_xlabel('Time (s)')
    ax_ht.set_ylabel('Height (km)')
    ax_ht.set_title("Trajectory height vs. time (RMS = {:.1f} m, {:.1f} ms)".format(drag_fit.res_rms_m, \
        1000.0*drag_fit.res_rms_s))
    ax_ht.grid()
    ax_ht.legend(fontsize=7)

    ax_vel.set_xlabel('Velocity (km/s)')
    ax_vel.set_ylabel('Height (km)')
    ax_vel.set_title("Velocity profile (v_init = {:.2f} km/s)".format(traj.v_init/1000.0))
    ax_vel.grid()
    ax_vel.legend(fontsize=7)

    ax_lag.set_xlabel('Lag (m)')
    ax_lag.set_ylabel('Time (s)')
    ax_lag.set_title('Lag along trajectory')
    ax_lag.invert_yaxis()
    ax_lag.grid()
    ax_lag.legend(fontsize=7)

    ax_k.set_xscale('log')
    ax_k.set_xlim(k_lo_lim, k_hi_lim)
    ax_k.set_xlabel('Ballistic coefficient K (m^2/kg)')
    ax_k.set_ylabel('Height (km)')
    ax_k.set_title('Fitted ballistic coefficient K')
    ax_k.grid(which='both', alpha=0.5)
    ax_k.legend(fontsize=7)

    plt.tight_layout()

    if save_results:
        savePlot(plt, file_name + '_drag_fit.' + traj.plot_file_type, output_dir)

    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    return fig



def _loadTrajectoryFromInputs(input_paths, walk=False, radec_at_ref=False, output_dir=None, **traj_kwargs):
    """ Build an unsolved TrajectoryDragFit instance from either a trajectory .pickle file or ECSV files.

    Arguments:
        input_paths: [list] List of file or directory paths (.pickle or .ecsv).

    Keyword arguments:
        walk: [bool] Recursively find ECSV files under input_paths[0]. False by default.
        radec_at_ref: [bool] Legacy ECSV RA/Dec conversion at jdt_ref. False by default.
        output_dir: [str] Explicit output directory. None by default (auto-created next to inputs).
        **traj_kwargs: [dict] Keyword arguments forwarded to TrajectoryDragFit.__init__.

    Return:
        traj: [Trajectory object] Populated trajectory ready for solveTrajectoryDragFit().
    """

    # Single .pickle input: copy all station observations into a fresh Trajectory
    if (len(input_paths) == 1) and input_paths[0].lower().endswith('.pickle') and (not walk):

        pkl_path = os.path.abspath(input_paths[0])
        dir_path, file_name = os.path.split(pkl_path)
        src_traj = loadPickle(dir_path, file_name)

        if output_dir is None:
            stamp = jd2Date(src_traj.jdt_ref, dt_obj=True).strftime('%Y%m%d-%H%M%S.%f')
            output_dir = os.path.join(dir_path, stamp + '_dragfit')

        traj = Trajectory(src_traj.jdt_ref, output_dir=output_dir, meastype=2, **traj_kwargs)
        for obs in src_traj.observations:
            traj.infillTrajectory(obs.azim_data, obs.elev_data, obs.time_data, obs.lat, obs.lon, obs.ele, \
                station_id=obs.station_id, excluded_time=getattr(obs, 'excluded_time', None), \
                ignore_list=getattr(obs, 'ignore_list', None), magnitudes=getattr(obs, 'magnitudes', None), \
                fov_beg=getattr(obs, 'fov_beg', None), fov_end=getattr(obs, 'fov_end', None), \
                obs_id=getattr(obs, 'obs_id', None), comment=getattr(obs, 'comment', ''), \
                ignore_station=getattr(obs, 'ignore_station', False))

        return traj


    # ECSV inputs: load via wmpl.Formats.ECSV.loadECSVs
    from wmpl.Formats.ECSV import loadECSVs

    ecsv_paths = []
    if walk:
        root_dir = os.path.abspath(input_paths[0])
        seen_names = []
        for dir_name, _, file_names in sorted(os.walk(root_dir), key=lambda x: x[0]):
            if 'REJECT' in dir_name:
                continue
            for fn in sorted(file_names):
                if fn.lower().endswith('.ecsv') and (fn not in seen_names):
                    ecsv_paths.append(os.path.join(dir_name, fn))
                    seen_names.append(fn)
        base_dir = root_dir
    else:
        for pat in input_paths:
            for match in sorted(glob.glob(pat)):
                full_p = os.path.abspath(match)
                if os.path.exists(full_p):
                    ecsv_paths.append(full_p)
        base_dir = os.path.dirname(ecsv_paths[0]) if ecsv_paths else '.'

    if len(ecsv_paths) < 2:
        raise ValueError("At least 2 ECSV files (or 1 trajectory .pickle file) are required")

    jdt_ref, meteor_list = loadECSVs(ecsv_paths, radec_per_point=(not radec_at_ref))

    if output_dir is None:
        stamp = jd2Date(jdt_ref, dt_obj=True).strftime('%Y%m%d-%H%M%S.%f')
        output_dir = os.path.join(base_dir, stamp + '_dragfit')

    traj = Trajectory(jdt_ref, output_dir=output_dir, meastype=1, **traj_kwargs)
    for meteor in meteor_list:
        comment = meteor.ff_name if hasattr(meteor, 'ff_name') else ''
        radec_jd = meteor.jdt_ref + meteor.time_data/86400.0 if meteor.radec_per_point else None
        traj.infillTrajectory(meteor.ra_data, meteor.dec_data, meteor.time_data, meteor.latitude, \
            meteor.longitude, meteor.height, station_id=meteor.station_id, magnitudes=meteor.mag_data, \
            comment=comment, radec_jd=radec_jd)

    return traj



if __name__ == "__main__":

    # Import through the package name so pickled objects never bind to __main__
    from wmpl.Formats.GenericFunctions import addKMLOptions, addSolverOptions
    from wmpl.Trajectory.TrajectoryDragFit import _loadTrajectoryFromInputs as loadFromInputs, \
        solveTrajectoryDragFit as runDragFit

    ### COMMAND LINE ARGUMENTS

    arg_parser = argparse.ArgumentParser(description="Run the drag-coupled trajectory solver on a " \
        + "trajectory .pickle file or 2+ ECSV files. Supports stations with no time overlap and " \
        + "disconnected groups.")

    arg_parser.add_argument('input_paths', nargs='+', metavar='INPUT_PATH', type=str, \
        help='Path to a trajectory .pickle file, 2+ ECSV files (wildcards supported), or a folder with -w.')

    arg_parser = addSolverOptions(arg_parser, skip_velpart=True)
    arg_parser = addKMLOptions(arg_parser)

    arg_parser.add_argument('-w', '--walk', action='store_true', \
        help='Recursively find all ECSV files in the given folder (skipping REJECT folders).')

    arg_parser.add_argument('--outdir', metavar='OUTPUT_DIR', type=str, default=None, \
        help='Output directory for reports, pickle and plots. Auto-created next to inputs by default.')

    arg_parser.add_argument('--nosave', action='store_true', \
        help='Do not save reports, pickle or plots to disk.')

    arg_parser.add_argument('--penalty', type=float, default=1.0, \
        help='Regularization weight on the growth of ln K. 1.0 by default.')

    arg_parser.add_argument('--knots', type=int, default=None, \
        help='Target number of observed segments of ln K. Automatic by default.')

    arg_parser.add_argument('--max_k0', type=float, default=DRAG_TRAJ_MAX_K0, \
        help='Upper bound on initial ballistic coefficient K_0 (m^2/kg). 10.0 by default; <= 0 disables.')

    arg_parser.add_argument('--radec_at_ref', action='store_true', \
        help='Legacy ECSV option: convert R.A./Dec. at jdt_ref instead of each point time.')

    cml_args = arg_parser.parse_args()

    max_toffset = cml_args.maxtoffset[0] if cml_args.maxtoffset else None
    vinitht = cml_args.vinitht[0] if cml_args.vinitht else None
    img_fmt = cml_args.imgformat[0] if isinstance(cml_args.imgformat, list) else cml_args.imgformat
    max_k0 = cml_args.max_k0 if (cml_args.max_k0 is not None) and (cml_args.max_k0 > 0.0) else None

    traj = loadFromInputs(cml_args.input_paths, walk=cml_args.walk, radec_at_ref=cml_args.radec_at_ref, \
        output_dir=cml_args.outdir, max_toffset=max_toffset, monte_carlo=(not cml_args.disablemc), \
        mc_runs=cml_args.mcruns, geometric_uncert=cml_args.uncertgeom, \
        gravity_correction=(not cml_args.disablegravity), air_rotation=(not cml_args.disableairrot), \
        gravity_factor=cml_args.gfact, plot_all_spatial_residuals=cml_args.plotallspatial, \
        plot_file_type=img_fmt, show_plots=(not cml_args.hideplots), save_results=(not cml_args.nosave), \
        v_init_ht=vinitht, show_jacchia=cml_args.jacchia, \
        estimate_timing_vel=(False if cml_args.notimefit is None else cml_args.notimefit), \
        fixed_times=cml_args.fixedtimes, mc_noise_std=cml_args.mcstd, \
        enable_OSM_plot=cml_args.enableOSM, save_kml=(cml_args.kml or cml_args.kmlmeas), \
        kml_measured=cml_args.kmlmeas)

    traj = runDragFit(traj, penalty=cml_args.penalty, n_knots=cml_args.knots, max_k0=max_k0)
    if traj is None:
        sys.exit(1)
