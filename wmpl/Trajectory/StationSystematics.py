""" Systematic errors of the stations of a meteor, measured from the residuals of its solved trajectory.

The Monte Carlo runs of the trajectory solver add independent noise to every point. A station whose
pointing is off by a fixed amount (its calibration, or the centroid of a saturated meteor) has the same
error at all its points, which does not average out with their number, so the uncertainties of a solution
do not include it. With three or more stations the others constrain the line of the trajectory, and the
residuals of each station show its own error across the line, as a constant and a slope along it.

Their size is used for the Monte Carlo runs of a fragment of the same meteor, seen by the same stations, whose
few points give no such redundancy (see wmpl.Formats.ECSV).
"""

from __future__ import print_function, division, absolute_import

import numpy as np

from wmpl.Trajectory.Trajectory import applyGravityDrop
from wmpl.Utils.Math import findClosestPoints, vectMag, vectNorm
from wmpl.Utils.TrajConversions import geo2Cartesian, raDec2ECI


# Minimum number of points of a station to measure its systematic error
MIN_POINTS = 6



def crossTrackResiduals(state_vect, radiant_eci, stat_eci, meas_eci, drop=None):
    """ Compute the signed residuals of lines of sight against a line, as the angle from each line of sight to
        the closest point on the line, in the direction of the normal to both.

    Arguments:
        state_vect: [ndarray] A point on the line (m).
        radiant_eci: [ndarray] Unit vector towards the radiant.
        stat_eci: [ndarray] Positions of the station, one row per line of sight (m).
        meas_eci: [ndarray] Unit vectors of the lines of sight.

    Keyword arguments:
        drop: [function] Takes a point on the line and the index of the line of sight, and returns the point
            dropped by gravity, and moved by drag against the rotating air, as in the trajectory solver. None
            by default, for a straight line.

    Return:
        (residuals, along):
            - residuals: [ndarray] Angles across the line (rad).
            - along: [ndarray] Positions of the closest points along the direction of motion (m).
    """

    direction = -vectNorm(np.array(radiant_eci))
    residuals = np.zeros(len(meas_eci))
    along = np.zeros(len(meas_eci))

    for i, (stat, meas) in enumerate(zip(stat_eci, meas_eci)):

        stat = np.ascontiguousarray(stat)
        meas = np.ascontiguousarray(meas)

        obs_cpa, rad_cpa, _ = findClosestPoints(stat, meas, np.ascontiguousarray(state_vect), \
            np.ascontiguousarray(radiant_eci))
        along[i] = np.dot(rad_cpa - state_vect, direction)

        # Find the closest point again on the line dropped by gravity
        if drop is not None:
            obs_cpa, rad_cpa, _ = findClosestPoints(stat, meas, np.ascontiguousarray(drop(rad_cpa, i)), \
                np.ascontiguousarray(radiant_eci))

        normal = vectNorm(np.cross(meas, direction))
        residuals[i] = np.arctan2(np.dot(rad_cpa - obs_cpa, normal), vectMag(obs_cpa - stat))

    return residuals, along



def stationSystematicErrors(traj):
    """ Measure the systematic error of each station of a solved trajectory across its line, as a robust
        straight line of the residuals against the position along the trajectory.

    Arguments:
        traj: [Trajectory object] Solved trajectory.

    Return:
        [dict] {station ID: (slope (rad/m), constant (rad)) of the straight line}. Empty with fewer than 3
            distinct stations, whose residuals say nothing about the pointing.
    """

    used = [obs for obs in traj.observations if not getattr(obs, 'ignore_station', False)]

    # Several files of the same station are not independent
    if len(set((obs.lat, obs.lon, obs.ele) for obs in used)) < 3:
        return {}

    # Times from the first point of the used stations, as in the line-of-sight fit (dropTimeOrigin, #202),
    #   or of all observations, also of the ignored stations, with a solver which takes them from all
    try:
        from wmpl.Trajectory.Trajectory import dropTimeOrigin
        t0 = dropTimeOrigin(traj.observations)

    except ImportError:
        t0 = min(obs.time_data[0] for obs in traj.observations)
    errors = {}

    # The free fall of the meteor from the line in solutions which have it (Trajectory.gravityFreeFall)
    free_fall = traj.gravityFreeFall() if hasattr(traj, 'gravityFreeFall') else None

    for obs in used:

        # Drop the line by gravity as the solver did, from each point's distance to the centre of the Earth
        drop = None
        if getattr(traj, 'los_gravity', getattr(traj, 'gravity_correction', False)):

            def drop(point, i, obs=obs):

                r0 = getattr(traj, 'gravity_r0', None)

                # The free fall from the line, as the solver drops the points
                if free_fall is not None:
                    dropped = traj.gravityDropPoint(point, obs.time_data[i] - t0, traj.radiant_eci_mini, r0, \
                        traj.v0z, free_fall)

                # Or the drop along the vertical, scaled for a decelerating meteor in solutions which have the
                #   scale
                else:
                    drop_factor = getattr(traj, 'gravityDropFactor', lambda t_rel: traj.gravity_factor)

                    dropped = applyGravityDrop(point, obs.time_data[i] - t0, \
                        vectMag(point) if (r0 is None) else r0, drop_factor(obs.time_data[i] - t0), traj.v0z)

                # The points of a decelerating meteor are also moved by drag against the rotating air in
                #   solutions which have the displacement
                shift_at = getattr(traj, 'airRotationShiftAt', None)
                if shift_at is not None:
                    dropped = dropped + shift_at(obs.time_data[i] - t0, traj.radiant_eci_mini)

                return dropped

        residuals, along = crossTrackResiduals(traj.state_vect_mini, traj.radiant_eci_mini, \
            obs.stat_eci_los, obs.meas_eci_los, drop=drop)

        keep = np.array(obs.ignore_list) == 0
        if np.sum(keep) < MIN_POINTS:
            continue

        residuals, along = residuals[keep], along[keep]

        # Start from the median of the slopes between all pairs of points (Theil-Sen), as a saturated stretch
        #   of the meteor can pull a least squares fit
        i, j = np.triu_indices(len(along), 1)
        d_along = along[j] - along[i]
        valid = np.abs(d_along) > 1e-6
        slope = np.median((residuals[j] - residuals[i])[valid]/d_along[valid])
        fit = np.array([slope, np.median(residuals - slope*along)])

        # Refit a few times without the outliers
        good = np.ones(len(residuals), dtype=bool)
        for _ in range(3):

            err = residuals - np.polyval(fit, along)
            scale = 1.4826*np.median(np.abs(err))
            good = np.abs(err) < 3*max(scale, 1e-9)

            if np.sum(good) < MIN_POINTS:
                good[:] = True
                break

            fit = np.polyfit(along[good], residuals[good], 1)

        errors[str(obs.station_id)] = tuple(np.polyfit(along[good], residuals[good], 1))

    return errors



def systematicErrorSize(traj, errors, meteors):
    """ Compute the typical size of the systematic errors of the stations at the position of a fragment along
        the trajectory, for its Monte Carlo runs.

    Arguments:
        traj: [Trajectory object] Solved trajectory the errors were measured on.
        errors: [dict] Systematic errors of its stations, see stationSystematicErrors().
        meteors: [list] MeteorObservation objects of the fragment, prepared for the solver.

    Return:
        [dict] Standard deviations of the constant across the line, 'sigma_c' (rad), and of its slope along
            the trajectory, 'sigma_g' (rad/m). None without errors.
    """

    if not errors:
        return None

    # Position of the fragment along the trajectory
    along_list = []
    for meteor in meteors:

        meas = np.array(raDec2ECI(meteor.ra_data, meteor.dec_data)).T
        jds = meteor.jdt_ref + np.array(meteor.time_data)/86400.0
        stat = np.array([geo2Cartesian(meteor.latitude, meteor.longitude, meteor.height, jd) for jd in jds])
        along_list.append(crossTrackResiduals(traj.state_vect_mini, traj.radiant_eci_mini, stat, meas)[1])

    along_mid = np.median(np.concatenate(along_list))

    constants = np.array([np.polyval(fit, along_mid) for fit in errors.values()])
    slopes = np.array([fit[0] for fit in errors.values()])

    # The line of the trajectory absorbed 4 of the 2 parameters of every station
    n = len(errors)
    factor = n/max(n - 2, 1)

    return {'sigma_c': np.sqrt(np.mean(constants**2)*factor), 'sigma_g': np.sqrt(np.mean(slopes**2)*factor)}
