""" Drag fit of the motion of a meteor along its solved trajectory.

The ballistic coefficient is fitted from the dynamics alone. This differs from wmpl.Utils.DynamicMassFit,
which computes the mass from the drag equation for an assumed shape and density.

The meteor is assumed to move along the straight line of the trajectory solution and to decelerate by
atmospheric drag:

    dv/dt = -K*rho*v**2 + g_along

where rho is the NRLMSISE-00 air density, g_along the component of gravity along the line and K the ballistic
coefficient (Gamma*A/(m**(1/3)*rho_m**(2/3))). Both ablation and fragmentation increase K, so it is only
allowed to grow along the trajectory, which keeps the velocity positive and decreasing. With the specific
kinetic energy E = v**2/2 and the length L along the line, the drag equation is linear:

    dE/dL = -2*K*rho*E + g_along

and is solved exactly. The logarithm of K is a sum of non-negative ramps in L, which is fitted to the lengths
and times of all used points with a robust loss and a small penalty on the growth of K.

Above the first observed point the meteor is assumed to move without drag, only accelerated by gravity, as
it barely decelerates in the thin air. Optionally, K can be kept at its first value instead. As K can only
have been smaller above, this is the largest possible deceleration, so the model above the beginning is then
only computed until the velocity has grown by a set fraction.

Below the last observed point K is kept at its last value, which is drag-only flight with gravity, as at the
beginning of the dark flight. Ablation essentially stops below about 3 km/s, so below that speed this is
exact. A meteor which ends faster probably still ablates below its end, but letting K grow there by
single-body ablation did not predict the hidden ends of real fireballs better.

Gravity bends the path of the meteor by the angle theta, dtheta/dt = g_perp/v, where g_perp is the component
of gravity perpendicular to the path. Drag damps the velocity gained from gravity in the same way as the rest
of the velocity, so the bending grows quickly as the meteor slows down. The model is only computed until the
path has bent by a few degrees from the first point. Further on, a dark flight model has to be used.

If the trajectory was solved with the gravity correction, the positions in the observed part are dropped from
the line in the same way as in the trajectory solver, and the heights and the air density are computed on the
dropped path. The drop is zero at the time of the first point, and the positions above it are on the line.
The drop of the solver assumes a constant velocity, which overestimates the bending of a decelerating meteor,
so below the last point the path is continued along the direction bent by theta, and the direction of the
radiant is bent by theta everywhere.

The fitted K shows how the meteor changed during its flight. If K is constant, the deceleration is only
driven by the air density and the velocity. For a single body K grows as m**(-1/3) as it ablates, and it
grows faster if the meteor fragments, as the fragments have a larger area per mass.
"""

from __future__ import print_function, division, absolute_import

import copy
import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

from wmpl.Trajectory.Trajectory import applyGravityDrop
from wmpl.Utils.AtmosphereDensity import getAtmDensity_vect
from wmpl.Utils.Math import findClosestPoints, vectMag
from wmpl.Utils.Physics import ballisticCoeffMass
from wmpl.Utils.Pickling import loadPickle
from wmpl.Utils.TrajConversions import EARTH, cartesian2Geo, derotatedRadiantAltAz, ecef2LatLonAlt


# Gravitational acceleration at the surface of the Earth (m/s^2)
G_SURFACE = 9.80665

# Product of the drag coefficient and the shape factor, and the bulk density (kg/m^3), assumed for the
#   dynamic mass by default
GAMMA_A = 0.78
BULK_DENSITY = 3500.0



def usedPoints(traj):
    """ Collect the times and the lengths along the trajectory of all used points.

    Arguments:
        traj: [Trajectory object] Solved trajectory.

    Return:
        (time_data, len_data):
            - time_data: [ndarray] Times relative to the reference Julian date (s).
            - len_data: [ndarray] Lengths along the trajectory from the state vector (m).
    """

    time_data = []
    len_data = []

    for obs in traj.observations:

        # Skip ignored stations
        if obs.ignore_station:
            continue

        # Take only the points which were not ignored
        used = np.array(obs.ignore_list) == 0
        time_data.append(np.array(obs.time_data)[used])
        len_data.append(np.array(obs.state_vect_dist)[used])


    if not time_data:
        raise ValueError("The drag model could not be fitted, the trajectory has no used points")

    time_data = np.concatenate(time_data)
    len_data = np.concatenate(len_data)

    # Sort the points by length
    sort_ind = np.argsort(len_data)

    return time_data[sort_ind], len_data[sort_ind]



class DragFit(object):
    def __init__(self, traj, n_knots=None, h_top=200000.0, penalty=1.0, drag_above=False, max_bending=5.0, \
        max_vel_growth=0.1):
        """ Fit the drag model to the lengths and times of the used points of the given trajectory.

        Arguments:
            traj: [Trajectory object] Solved trajectory.

        Keyword arguments:
            n_knots: [int] Number of segments of the ballistic coefficient. None by default, which uses one
                segment for every 25 points, but no fewer than 3 and no more than 15.
            h_top: [float] Height of the top of the modelled part of the trajectory (m), raised to 100 km
                above the first observed point if that is higher. 200 km by default.
            penalty: [float] Weight of the penalty on the growth of the ballistic coefficient, relative to the
                residuals. 1.0 by default.
            drag_above: [bool] If True, the ballistic coefficient above the first observed point is kept at
                its first value. If False, the meteor moves without drag above the first point. False by
                default.
            max_bending: [float] Largest bending of the path by gravity from the first point (deg). The model
                is not computed further below the end. 5 deg by default.
            max_vel_growth: [float] Largest relative growth of the velocity above the beginning of the
                observed part, only used if drag_above is True. The model is not computed further. 0.1 by
                default.
        """

        self.traj = traj
        self.drag_above = drag_above
        self.gravity_factor = getattr(traj, 'gravity_factor', 1.0)
        self.max_bending = np.radians(max_bending)
        self.max_vel_growth = max_vel_growth

        self.state_vect = np.array(traj.state_vect_mini)
        self.radiant_eci = np.array(traj.radiant_eci_mini)

        # Get the data
        self.time_data, self.len_data = usedPoints(traj)
        self.len_first = np.min(self.len_data)
        self.len_last = np.max(self.len_data)

        # Compute the line geometry and the atmosphere along it
        h_top = max(h_top, self._wgs84HeightEci(self._lineEci(self.len_first))[0] + 100e3)
        self.drop_grid = None
        self._initLine(h_top)

        # Set the ramps of the ballistic coefficient, uniformly spaced between the first and the last point
        if n_knots is None:
            n_knots = int(np.clip(len(self.len_data)/25, 3, 15))

        self.n_knots = n_knots
        self.ramp_beg = np.linspace(self.len_first, self.len_last, n_knots + 1)[:-1]
        self.ramp_len = (self.len_last - self.len_first)/n_knots

        # Fit the model and compute the solution along the whole line
        self._fit(penalty)
        self._checkFit()
        self._solveFull()

        # Apply the gravity drop if the trajectory was solved with it, and fit again on the dropped path
        self.gravity_correction = bool(getattr(traj, 'gravity_correction', False))
        if self.gravity_correction:

            self._computeGravityDrop()
            self._fit(penalty, params_init=self.params)
            self._checkFit()
            self._solveFull()

            # Recompute the drop with the final times, and the solution on the final dropped path
            self._computeGravityDrop()
            self._solveFull()



    def _checkFit(self):
        """ Check that the fit is sensible, as the lengths and times of a bad solution can be inconsistent.
        """

        duration = np.max(self.time_data) - np.min(self.time_data)
        if (not np.isfinite(self.res_rms)) or (abs(self.params[0] - self.time_data[0]) > max(1.0, duration)):
            raise ValueError("The drag model could not be fitted, the lengths and times of the points of " \
                + "the trajectory are not consistent")



    def _initLine(self, h_top, n_seg=(1000, 1000, 1000), n_atm=300):
        """ Compute the heights, the air density and the gravity along the line, from the height h_top to the
            ground (or back to h_top for Earth-grazers).

        Arguments:
            h_top: [float] Height of the top of the modelled part of the trajectory (m).

        Keyword arguments:
            n_seg: [tuple] Number of grid points above, inside and below the observed part.
            n_atm: [int] Number of points along the line at which the air density is computed.
        """

        # Find the range of lengths along the line between h_top and the ground, around the observed part
        len_coarse = np.arange(self.len_first - 3000e3, self.len_last + 3000e3, 2000.0)
        ht_coarse = self._wgs84Height(len_coarse)

        i_first = np.searchsorted(len_coarse, self.len_first)
        i_last = np.searchsorted(len_coarse, self.len_last)

        i_top = i_first
        while (i_top > 0) and (ht_coarse[i_top - 1] < h_top):
            i_top -= 1

        i_bot = i_last
        while (i_bot < len(len_coarse) - 1) and (ht_coarse[i_bot + 1] > 0) and (ht_coarse[i_bot + 1] < h_top):
            i_bot += 1

        len_top = min(len_coarse[i_top], self.len_first - 1.0)
        len_bot = max(len_coarse[i_bot], self.len_last + 1.0)


        # Make a grid with nodes at the first and the last observed point
        self.len_grid = np.concatenate([np.linspace(len_top, self.len_first, n_seg[0]), \
            np.linspace(self.len_first, self.len_last, n_seg[1])[1:], \
            np.linspace(self.len_last, len_bot, n_seg[2])[1:]])
        self.i_first = n_seg[0] - 1
        self.i_last = n_seg[0] + n_seg[1] - 2

        # Compute the geographical coordinates at points along the line. The heights are above the mean sea
        #   level, as everywhere in WMPL
        len_atm = np.linspace(len_top, len_bot, n_atm)
        lat_atm, lon_atm, ht_atm = [], [], []
        for len_val in len_atm:
            lat, lon, ht = cartesian2Geo(self.traj.jdt_ref, *self._lineEci(len_val)[0])
            lat_atm.append(lat)
            lon_atm.append(lon)
            ht_atm.append(ht)

        ht_atm = np.array(ht_atm)

        # Compute the heights on the grid, interpolating the slowly changing geoid height
        self.len_atm = len_atm
        self.geoid_atm = ht_atm - self._wgs84Height(len_atm)
        self.ht_grid = self._wgs84Height(self.len_grid) + np.interp(self.len_grid, len_atm, self.geoid_atm)


        # Compute the air density along the line, and interpolate its logarithm with height
        log_rho = np.log(getAtmDensity_vect(np.array(lat_atm), np.array(lon_atm), np.maximum(ht_atm, 0.0), \
            self.traj.jdt_ref))
        sort_ind = np.argsort(ht_atm)
        self.ht_atm = ht_atm[sort_ind]
        self.log_rho_atm = log_rho[sort_ind]
        self.rho_grid = np.exp(np.interp(self.ht_grid, self.ht_atm, self.log_rho_atm))


        # Compute the components of gravity along the line (positive when descending) and perpendicular to it
        g = G_SURFACE*(EARTH.EQUATORIAL_RADIUS/(EARTH.EQUATORIAL_RADIUS + self.ht_grid))**2
        self.g_grid = -g*np.gradient(self.ht_grid, self.len_grid)
        self.g_perp_grid = np.sqrt(np.maximum(g**2 - self.g_grid**2, 0.0))



    def _logK(self, params, len_arr):
        """ Compute the logarithm of the ballistic coefficient at the given lengths.

        Arguments:
            params: [ndarray] Model parameters (t_first, v_first, ln K at the first point, ramp heights).
            len_arr: [ndarray] Lengths along the trajectory (m).

        Return:
            [ndarray] Logarithm of the ballistic coefficient.
        """

        ramps = np.clip((len_arr[:, None] - self.ramp_beg[None, :])/self.ramp_len, 0.0, 1.0)

        return params[2] + ramps.dot(params[3:])



    def _solveObserved(self, params):
        """ Compute the velocity and the time in the observed part of the trajectory, by the exact solution of
            the linear energy equation.

        Arguments:
            params: [ndarray] Model parameters (t_first, v_first, ln K at the first point, ramp heights).

        Return:
            (vel, time):
                - vel: [ndarray] Velocity at the observed part of the grid (m/s).
                - time: [ndarray] Time at the observed part of the grid (s).
        """

        len_arr = self.len_grid[self.i_first:self.i_last + 1]
        rho = self.rho_grid[self.i_first:self.i_last + 1]
        g = self.g_grid[self.i_first:self.i_last + 1]
        dlen = np.diff(len_arr)

        # Compute the integral of 2*K*rho from the first point
        a = 2*np.exp(self._logK(params, len_arr))*rho
        phi = np.concatenate([[0.0], np.cumsum(0.5*(a[1:] + a[:-1])*dlen)])
        exp_phi = np.exp(np.clip(phi, None, 700.0))

        # Compute the energy, including the work done by gravity
        grav = np.concatenate([[0.0], np.cumsum(0.5*(g[1:]*exp_phi[1:] + g[:-1]*exp_phi[:-1])*dlen)])
        energy = (0.5*params[1]**2 + grav)/exp_phi
        vel = np.sqrt(np.maximum(2*energy, 1.0))

        # Integrate the time
        time = params[0] + np.concatenate([[0.0], np.cumsum(0.5*(1/vel[1:] + 1/vel[:-1])*dlen)])

        return vel, time



    def _residuals(self, params, w_pen):
        """ Compute the residuals of the model, in meters along the trajectory, and the penalty on the growth
            of the ballistic coefficient.

        Arguments:
            params: [ndarray] Model parameters (t_first, v_first, ln K at the first point, ramp heights).
            w_pen: [float] Weight of the penalty.

        Return:
            [ndarray] Residuals.
        """

        vel, time = self._solveObserved(params)
        len_arr = self.len_grid[self.i_first:self.i_last + 1]

        # Compute the model lengths at the observed times, continuing with the end velocities outside the
        #   modelled time range
        len_model = np.interp(self.time_data, time, len_arr)
        len_model = np.where(self.time_data < time[0], len_arr[0] + vel[0]*(self.time_data - time[0]), \
            len_model)
        len_model = np.where(self.time_data > time[-1], len_arr[-1] + vel[-1]*(self.time_data - time[-1]), \
            len_model)

        return np.concatenate([len_model - self.len_data, w_pen*params[3:]])



    def _fit(self, penalty, params_init=None):
        """ Fit the model parameters to the data.

        Arguments:
            penalty: [float] Weight of the penalty on the growth of the ballistic coefficient.

        Keyword arguments:
            params_init: [ndarray] Initial parameters. None by default, in which case they are searched for.
        """

        lower = np.concatenate([[-np.inf, 100.0, -60.0], np.zeros(self.n_knots)])
        upper = np.concatenate([[np.inf, 2e5, 10.0], np.full(self.n_knots, 30.0)])

        # Refine the given parameters only
        if params_init is not None:
            self._fitFinal(np.clip(params_init, lower, upper), penalty, lower, upper)
            return None


        # Estimate the initial time and velocity from a line fit to the first quarter of the points
        n_init = max(int(0.25*len(self.len_data)), 4)
        vel_slope, time_int = np.polyfit(self.len_data[:n_init], self.time_data[:n_init], 1)
        v_first = 1.0/vel_slope if vel_slope > 0 else self.traj.v_init
        t_first = time_int + vel_slope*self.len_first


        # Find the constant ballistic coefficient which fits best, starting from several values
        best = None
        for log_k in (-30.0, -24.0, -18.0, -13.0, -9.0, -5.0):

            p0 = np.concatenate([[t_first, np.clip(v_first, 101.0, 1.9e5), log_k], np.zeros(self.n_knots)])

            res = scipy.optimize.least_squares(lambda p: self._residuals(np.concatenate([p, \
                np.zeros(self.n_knots)]), 0.0)[:len(self.len_data)], p0[:3], bounds=(lower[:3], upper[:3]), \
                loss='soft_l1', f_scale=50.0, x_scale='jac', ftol=1e-6, xtol=1e-6)

            if (best is None) or (res.cost < best.cost):
                best = res


        # Estimate the scale of the residuals and use it for the robust loss and the penalty
        res_const = self._residuals(np.concatenate([best.x, np.zeros(self.n_knots)]), 0.0)
        res_const = res_const[:len(self.len_data)]
        self.res_scale = max(1.4826*np.median(np.abs(res_const - np.median(res_const))), 1.0)
        f_scale = 2*self.res_scale
        w_pen = np.sqrt(penalty)*f_scale


        # Fit the full model, letting the ballistic coefficient grow. Start both from the constant coefficient
        #   and from one which grows by a factor of 10 over the observed part, and keep the better fit
        best_full = None
        for d_init in (1e-3, np.log(10.0)/self.n_knots):

            p0 = np.concatenate([best.x, np.full(self.n_knots, d_init)])
            p0[2] -= self.n_knots*d_init/2
            p0 = np.clip(p0, lower + 1e-9, upper - 1e-9)

            res = scipy.optimize.least_squares(self._residuals, p0, args=(w_pen,), bounds=(lower, upper), \
                loss='soft_l1', f_scale=f_scale, x_scale='jac', ftol=1e-6, xtol=1e-6, max_nfev=1000)

            if (best_full is None) or (res.cost < best_full.cost):
                best_full = res


        self._fitFinal(best_full.x, penalty, lower, upper)



    def _fitFinal(self, params, penalty, lower, upper):
        """ Re-estimate the scale of the residuals with the given parameters and fit the model again with it.

        Arguments:
            params: [ndarray] Initial parameters.
            penalty: [float] Weight of the penalty on the growth of the ballistic coefficient.
            lower: [ndarray] Lower bounds of the parameters.
            upper: [ndarray] Upper bounds of the parameters.
        """

        res_init = self._residuals(params, 0.0)[:len(self.len_data)]
        self.res_scale = max(1.4826*np.median(np.abs(res_init - np.median(res_init))), 1.0)
        f_scale = 2*self.res_scale
        w_pen = np.sqrt(penalty)*f_scale

        res = scipy.optimize.least_squares(self._residuals, params, args=(w_pen,), bounds=(lower, upper), \
            loss='soft_l1', f_scale=f_scale, x_scale='jac', ftol=1e-6, xtol=1e-6, max_nfev=1000)

        self.params = res.x

        # Estimate the covariance of the parameters from the Jacobian and the scale of the residuals. The
        #   columns of the Jacobian are normalized, so the parameters with different units can be compared
        col_norm = np.linalg.norm(res.jac, axis=0)
        col_norm[col_norm == 0] = 1.0
        _, sing, vt = np.linalg.svd(res.jac/col_norm, full_matrices=False)

        # The directions in which the residuals do not change are not constrained by the data
        null = sing <= 1e-8*sing[0]
        vt_good = vt[~null]/col_norm
        self.params_cov = self.res_scale**2*(vt_good.T/sing[~null]**2).dot(vt_good)

        # The unconstrained directions in the parameter space, J*x = 0
        params_null = (vt[null]/col_norm).T
        if params_null.size:
            params_null /= np.linalg.norm(params_null, axis=0)
        self.params_null = params_null

        # Compute the final residuals
        self.residuals = self._residuals(self.params, 0.0)[:len(self.len_data)]
        self.res_rms = np.sqrt(np.mean(self.residuals**2))



    def _paramStd(self, grad):
        """ Compute the standard deviations of linear combinations of the parameters, which are infinite if
            they depend on a direction not constrained by the data.

        Arguments:
            grad: [ndarray] Coefficients of the combinations, one row per combination.

        Return:
            [ndarray] Standard deviations.
        """

        std = np.sqrt(np.maximum(np.sum(grad.dot(self.params_cov)*grad, axis=1), 0.0))

        if self.params_null.size:
            grad_norm = np.linalg.norm(grad, axis=1)[:, None]
            unconstrained = np.any(np.abs(grad.dot(self.params_null)) > 1e-6*grad_norm, axis=1)
            std[unconstrained] = np.inf

        return std



    def _solveFull(self):
        """ Compute the velocity and the time along the whole line, continuing the solution in the observed
            part with a constant ballistic coefficient above and below it.
        """

        n = len(self.len_grid)
        vel = np.full(n, np.nan)
        time = np.full(n, np.nan)

        # Solve the observed part
        vel[self.i_first:self.i_last + 1], time[self.i_first:self.i_last + 1] = \
            self._solveObserved(self.params)

        # Compute the bending of the path by gravity from the first point, dtheta/dL = g_perp/v**2
        bend = np.full(n, np.nan)
        obs_slice = slice(self.i_first, self.i_last + 1)
        bend_rate = self.g_perp_grid[obs_slice]/vel[obs_slice]**2
        bend[obs_slice] = np.concatenate([[0.0], np.cumsum(0.5*(bend_rate[1:] + bend_rate[:-1]) \
            *np.diff(self.len_grid[obs_slice]))])

        # Take the ballistic coefficient at the first and the last point outside the observed part
        log_k_grid = self._logK(self.params, self.len_grid)
        if not self.drag_above:
            log_k_grid[:self.i_first] = -np.inf

        # Go up from the first point and down from the last point, solving the energy equation exactly in
        #   every grid cell
        energy = 0.5*vel[self.i_first]**2
        for i in range(self.i_first, 0, -1):

            dlen = self.len_grid[i] - self.len_grid[i - 1]
            g = 0.5*(self.g_grid[i] + self.g_grid[i - 1])

            # Move without drag above the first point, unless the drag is kept
            if self.drag_above:
                a = 2*np.exp(log_k_grid[i])*0.5*(self.rho_grid[i] + self.rho_grid[i - 1])
            else:
                a = 0.0

            energy = self._energyStep(energy, a, g, -dlen)

            # Stop when the velocity has grown too much, as K above the beginning is not known
            if np.sqrt(max(2*energy, 0.0)) > (1 + self.max_vel_growth)*vel[self.i_first]:
                break

            vel[i - 1] = np.sqrt(max(2*energy, 0.0))
            time[i - 1] = time[i] - dlen*0.5*(1/vel[i] + 1/vel[i - 1])


        energy = 0.5*vel[self.i_last]**2
        bending = bend[self.i_last]
        for i in range(self.i_last, n - 1):

            dlen = self.len_grid[i + 1] - self.len_grid[i]
            a = 2*np.exp(log_k_grid[i])*0.5*(self.rho_grid[i] + self.rho_grid[i + 1])
            g = 0.5*(self.g_grid[i] + self.g_grid[i + 1])

            energy = self._energyStep(energy, a, g, dlen)

            # Stop if the meteor comes to a halt (e.g. an Earth-grazer going up)
            if energy <= 0.5:
                break

            # Stop when gravity has bent the path too far from the line
            bending += 0.5*(self.g_perp_grid[i]/vel[i]**2 + self.g_perp_grid[i + 1]/(2*energy))*dlen
            if bending > self.max_bending:
                break

            vel[i + 1] = np.sqrt(2*energy)
            time[i + 1] = time[i] + dlen*0.5*(1/vel[i] + 1/vel[i + 1])
            bend[i + 1] = bending


        # The path above the first point is not bent
        bend[:self.i_first][np.isfinite(time[:self.i_first])] = 0.0

        self.vel_grid = vel
        self.time_grid = time
        self.log_k_grid = log_k_grid
        self.bend_grid = bend

        # Keep the part of the grid where the solution is defined, for interpolation
        self._valid = np.isfinite(time) & np.isfinite(vel) & (vel > 0)



    @staticmethod
    def _energyStep(energy, a, g, dlen):
        """ Advance the energy equation dE/dL = -a*E + g by dlen, for constant a and g.

        Arguments:
            energy: [float] Specific kinetic energy at the start (J/kg).
            a: [float] Drag term 2*K*rho (1/m).
            g: [float] Component of gravity along the line (m/s^2).
            dlen: [float] Step along the line (m), negative when going up.

        Return:
            [float] Specific kinetic energy at the end of the step (J/kg).
        """

        x = a*dlen

        # Without drag only gravity does work
        if abs(x) < 1e-12:
            return energy + g*dlen

        return energy*np.exp(-x) + (g/a)*(1 - np.exp(-x))



    def _computeGravityDrop(self):
        """ Compute the gravity drop along the line in the same way as the trajectory solver does in the
            observed part, and from the bending by gravity below it, and update the heights and the air
            density on the grid.
        """

        traj = self.traj

        # Take the time of the first point and the largest distance from the centre of the Earth of the first
        #   points of all stations as the reference, as the trajectory solver does
        t0 = min([obs.time_data[0] for obs in traj.observations])
        r0 = max([vectMag(findClosestPoints(obs.stat_eci_los[0], obs.meas_eci_los[0], self.state_vect, \
            self.radiant_eci)[1]) for obs in traj.observations])

        v0z = getattr(traj, 'v0z', None)
        if v0z is None:
            v0z = 0.0

        gravity_factor = getattr(traj, 'gravity_factor', 1.0)

        # Compute the drop vector at every grid point with a defined time. The drop is zero above the first
        #   point, where the positions are on the line
        line_eci = self._lineEci(self.len_grid)
        drop_grid = np.zeros_like(line_eci)
        for i, (eci, t) in enumerate(zip(line_eci, self.time_grid)):
            if np.isfinite(t) and (t > t0):
                drop_grid[i] = eci - applyGravityDrop(eci, t - t0, r0, gravity_factor, v0z)

        # Below the last point, continue the path along the direction bent by gravity
        valid = np.where(np.isfinite(self.time_grid))[0]
        pos = line_eci[self.i_last] - drop_grid[self.i_last]
        for i in range(self.i_last + 1, valid[-1] + 1):

            dlen = self.len_grid[i] - self.len_grid[i - 1]
            theta = 0.5*(self.bend_grid[i - 1] + self.bend_grid[i])

            pos = pos - dlen*self._bentRadiant(np.array([theta]), pos[None, :])[0]
            drop_grid[i] = line_eci[i] - pos

        # Keep the drop constant outside the modelled part, so the heights stay continuous
        drop_grid[:valid[0]] = drop_grid[valid[0]]
        drop_grid[valid[-1] + 1:] = drop_grid[valid[-1]]

        self.drop_grid = drop_grid

        # Update the heights and the air density
        self.ht_grid = self._wgs84HeightEci(line_eci - drop_grid) + np.interp(self.len_grid, self.len_atm, \
            self.geoid_atm)
        self.rho_grid = np.exp(np.interp(self.ht_grid, self.ht_atm, self.log_rho_atm))



    def _lineEci(self, len_arr):
        """ Compute the ECI coordinates of the points on the straight line at the given lengths.

        Arguments:
            len_arr: [float or ndarray] Lengths along the trajectory from the state vector (m).

        Return:
            [ndarray] ECI coordinates (m), one row per point.
        """

        return self.state_vect[None, :] - np.outer(np.atleast_1d(len_arr), self.radiant_eci)



    def eciAtLength(self, len_arr):
        """ Compute the ECI coordinates of the points at the given lengths along the trajectory, including
            the gravity drop if the trajectory was solved with it.

        Arguments:
            len_arr: [float or ndarray] Lengths along the trajectory from the state vector (m).

        Return:
            [ndarray] ECI coordinates (m), one row per point.
        """

        len_arr = np.atleast_1d(len_arr)
        eci = self._lineEci(len_arr)

        # Subtract the interpolated gravity drop
        if self.drop_grid is not None:
            for i in range(3):
                eci[:, i] -= np.interp(len_arr, self.len_grid, self.drop_grid[:, i])

        return eci



    def _bentRadiant(self, theta, eci_arr):
        """ Bend the direction of the radiant upwards by the given angles, as the direction of motion is
            bent downwards by gravity.

        Arguments:
            theta: [ndarray] Bending angles (radians), before the gravity factor is applied.
            eci_arr: [ndarray] ECI coordinates of the points (m), one row per point.

        Return:
            [ndarray] Unit vectors towards the radiant in the ECI frame, one row per point.
        """

        theta = self.gravity_factor*theta

        # Compute the upward direction perpendicular to the radiant
        up = eci_arr/np.linalg.norm(eci_arr, axis=1)[:, None]
        up_perp = up - up.dot(self.radiant_eci)[:, None]*self.radiant_eci[None, :]
        up_perp /= np.linalg.norm(up_perp, axis=1)[:, None]

        return np.cos(theta)[:, None]*self.radiant_eci[None, :] + np.sin(theta)[:, None]*up_perp



    def radiantAtLength(self, len_arr):
        """ Compute the directions of the radiant at the given lengths along the trajectory, opposite to the
            direction of motion. If the trajectory was solved with the gravity correction, the direction is
            bent by gravity from the first point, by the angle theta, dtheta/dt = g_perp/v.

        Arguments:
            len_arr: [float or ndarray] Lengths along the trajectory from the state vector (m).

        Return:
            [ndarray] Unit vectors towards the radiant in the ECI frame, one row per point.
        """

        len_arr = np.atleast_1d(len_arr)

        if not self.gravity_correction:
            return np.tile(self.radiant_eci, (len(len_arr), 1))

        theta = np.interp(len_arr, self.len_grid[self._valid], self.bend_grid[self._valid])

        return self._bentRadiant(theta, self.eciAtLength(len_arr))



    def _wgs84HeightEci(self, eci_arr):
        """ Compute the heights above the WGS84 ellipsoid of the given ECI points.

        Arguments:
            eci_arr: [ndarray] ECI coordinates (m), one row per point.

        Return:
            [ndarray] Heights above the WGS84 ellipsoid (m).
        """

        # The height does not depend on the rotation of the Earth, so the ECI coordinates are used directly
        return np.array([ecef2LatLonAlt(*eci)[2] for eci in np.atleast_2d(eci_arr)])



    def _wgs84Height(self, len_arr):
        """ Compute the heights above the WGS84 ellipsoid of the points on the straight line at the given
            lengths.

        Arguments:
            len_arr: [float or ndarray] Lengths along the trajectory from the state vector (m).

        Return:
            [ndarray] Heights above the WGS84 ellipsoid (m).
        """

        return self._wgs84HeightEci(self._lineEci(len_arr))



    def heightAtLength(self, len_arr):
        """ Interpolate the heights at the given lengths along the trajectory.

        Arguments:
            len_arr: [float or ndarray] Lengths along the trajectory from the state vector (m).

        Return:
            [ndarray] Heights above the mean sea level (m).
        """

        return np.interp(np.atleast_1d(len_arr), self.len_grid, self.ht_grid)



    def lengthAtHeight(self, ht_arr):
        """ Compute the lengths along the trajectory at the given heights, on the descending part of the
            trajectory (for an Earth-grazer, before its lowest point).

        Arguments:
            ht_arr: [float or ndarray] Heights above the mean sea level (m).

        Return:
            [ndarray] Lengths along the trajectory (m), NaN for heights outside the modelled part.
        """

        # Take the descending part of the modelled line
        ht_valid = self.ht_grid[self._valid]
        len_valid = self.len_grid[self._valid]
        i_low = np.argmin(ht_valid)
        ht_desc = ht_valid[:i_low + 1][::-1]
        len_desc = len_valid[:i_low + 1][::-1]

        return np.interp(np.atleast_1d(ht_arr), ht_desc, len_desc, left=np.nan, right=np.nan)



    def timeAtLength(self, len_arr):
        """ Compute the times at the given lengths along the trajectory.

        Arguments:
            len_arr: [float or ndarray] Lengths along the trajectory from the state vector (m).

        Return:
            [ndarray] Times relative to the reference Julian date (s), NaN outside the modelled part.
        """

        return np.interp(np.atleast_1d(len_arr), self.len_grid[self._valid], self.time_grid[self._valid], \
            left=np.nan, right=np.nan)



    def velocityAtLength(self, len_arr):
        """ Compute the velocities at the given lengths along the trajectory.

        Arguments:
            len_arr: [float or ndarray] Lengths along the trajectory from the state vector (m).

        Return:
            [ndarray] Velocities in the ECI frame (m/s), NaN outside the modelled part.
        """

        return np.interp(np.atleast_1d(len_arr), self.len_grid[self._valid], self.vel_grid[self._valid], \
            left=np.nan, right=np.nan)



    def ballisticCoeffAtLength(self, len_arr):
        """ Compute the ballistic coefficients at the given lengths along the trajectory, and their
            uncertainties estimated from the fit.

        Arguments:
            len_arr: [float or ndarray] Lengths along the trajectory from the state vector (m).

        Return:
            (k_arr, log_k_std):
                - k_arr: [ndarray] Ballistic coefficients (m^2/kg), zero above the first point if the meteor
                    moves without drag there, NaN outside the modelled part.
                - log_k_std: [ndarray] Standard deviations of the logarithms of the ballistic coefficients,
                    which are the relative uncertainties.
        """

        len_arr = np.atleast_1d(len_arr)

        # Outside the observed part the coefficient is constant, or zero above the first point without drag
        len_clip = np.clip(len_arr, self.len_first, self.len_last)
        k_arr = np.exp(self._logK(self.params, len_clip))
        if not self.drag_above:
            k_arr[len_arr < self.len_first] = 0.0

        # The coefficient is not defined outside the modelled part
        len_valid = self.len_grid[self._valid]
        k_arr[(len_arr < len_valid[0]) | (len_arr > len_valid[-1]) | np.isnan(len_arr)] = np.nan

        # Compute the uncertainty from the covariance of the parameters, which outside the observed part is
        #   the one at the nearest observed point
        ramps = np.clip((len_clip[:, None] - self.ramp_beg[None, :])/self.ramp_len, 0.0, 1.0)
        grad = np.hstack([np.zeros((len(len_arr), 2)), np.ones((len(len_arr), 1)), ramps])
        log_k_std = self._paramStd(grad)

        # The coefficient is exactly zero above the first point without drag
        log_k_std[k_arr == 0] = 0.0

        return k_arr, log_k_std



    def lengthAtTime(self, time_arr):
        """ Compute the lengths along the trajectory at the given times.

        Arguments:
            time_arr: [float or ndarray] Times relative to the reference Julian date (s).

        Return:
            [ndarray] Lengths along the trajectory from the state vector (m), NaN outside the modelled part.
        """

        return np.interp(np.atleast_1d(time_arr), self.time_grid[self._valid], self.len_grid[self._valid], \
            left=np.nan, right=np.nan)



    def heightAtTime(self, time_arr):
        """ Compute the heights at the given times.

        Arguments:
            time_arr: [float or ndarray] Times relative to the reference Julian date (s).

        Return:
            [ndarray] Heights above the mean sea level (m), NaN outside the modelled part.
        """

        return self.heightAtLength(self.lengthAtTime(time_arr))



    def velocityAtTime(self, time_arr):
        """ Compute the smooth velocities of the model at the given times.

        Arguments:
            time_arr: [float or ndarray] Times relative to the reference Julian date (s).

        Return:
            [ndarray] Velocities in the ECI frame (m/s), NaN outside the modelled part.
        """

        return self.velocityAtLength(self.lengthAtTime(time_arr))



    def velocityAtHeight(self, ht_arr):
        """ Compute the smooth velocities of the model at the given heights, on the descending part of the
            trajectory.

        Arguments:
            ht_arr: [float or ndarray] Heights above the mean sea level (m).

        Return:
            [ndarray] Velocities in the ECI frame (m/s), NaN outside the modelled part.
        """

        return self.velocityAtLength(self.lengthAtHeight(ht_arr))



    def groundVelocityAtLength(self, len_arr):
        """ Compute the velocities relative to the ground at the given lengths along the trajectory.

        Arguments:
            len_arr: [float or ndarray] Lengths along the trajectory from the state vector (m).

        Return:
            [ndarray] Velocities relative to the ground (m/s), NaN outside the modelled part.
        """

        len_arr = np.atleast_1d(len_arr)

        # Compute the modelled motion at the given lengths
        vel = self.velocityAtLength(len_arr)
        time_arr = self.timeAtLength(len_arr)
        eci_arr = self.eciAtLength(len_arr)
        radiant_arr = self.radiantAtLength(len_arr)

        vel_norot = np.full(len(len_arr), np.nan)
        for i in range(len(len_arr)):

            # Skip the points outside the modelled part
            if not (np.isfinite(vel[i]) and np.isfinite(time_arr[i])):
                continue

            # Derotate the velocity, with the ECI coordinates in the epoch of date
            jd = self.traj.jdt_ref + time_arr[i]/86400.0
            lat, lon, _ = cartesian2Geo(jd, *eci_arr[i])
            _, _, vel_norot[i] = derotatedRadiantAltAz(vel[i]*radiant_arr[i], eci_arr[i], jd, lat, lon)


        return vel_norot



    def airDensityAtLength(self, len_arr):
        """ Compute the air density used by the model at the given lengths along the trajectory.

        Arguments:
            len_arr: [float or ndarray] Lengths along the trajectory from the state vector (m).

        Return:
            [ndarray] Air densities (kg/m^3).
        """

        return np.exp(np.interp(self.heightAtLength(len_arr), self.ht_atm, self.log_rho_atm))



    def decelerationAtLength(self, len_arr):
        """ Compute the deceleration K*rho*v**2 - g_along of the model at the given lengths along the
            trajectory.

        Arguments:
            len_arr: [float or ndarray] Lengths along the trajectory from the state vector (m).

        Return:
            [ndarray] Decelerations (m/s^2), positive when the meteor slows down, NaN outside the modelled
                part.
        """

        len_arr = np.atleast_1d(len_arr)

        # Take the drag and the gravity along the path
        k_arr, _ = self.ballisticCoeffAtLength(len_arr)
        vel = self.velocityAtLength(len_arr)
        g_along = np.interp(len_arr, self.len_grid, self.g_grid)

        return k_arr*self.airDensityAtLength(len_arr)*vel**2 - g_along



    def _motionGradients(self, coord_arr, by_time):
        """ Compute the derivatives of the velocity and of the time or the length with respect to the
            parameters by finite differences, with the gravity drop kept fixed.

        Arguments:
            coord_arr: [ndarray] Lengths along the trajectory from the state vector (m), or times (s).
            by_time: [bool] If True, coord_arr are times and the derivatives of the lengths are computed. If
                False, coord_arr are lengths and the derivatives of the times are computed.

        Return:
            (grad_vel, grad_coord):
                - grad_vel: [ndarray] Derivatives of the velocities, one row per point.
                - grad_coord: [ndarray] Derivatives of the times or the lengths, one row per point.
        """

        def motionAtParams(params):

            # Solve the motion on a copy, which keeps this fit unchanged
            drag_fit = copy.copy(self)
            drag_fit.params = params
            drag_fit._solveFull()

            # Interpolate the velocity and the time or the length at the given points
            valid = drag_fit._valid
            if by_time:
                x_arr, y_arr = drag_fit.time_grid[valid], drag_fit.len_grid[valid]
            else:
                x_arr, y_arr = drag_fit.len_grid[valid], drag_fit.time_grid[valid]

            vel = np.interp(coord_arr, x_arr, drag_fit.vel_grid[valid], left=np.nan, right=np.nan)
            coord = np.interp(coord_arr, x_arr, y_arr, left=np.nan, right=np.nan)

            return np.column_stack([vel, coord])


        # Solve the motion with the fitted parameters
        n_params = len(self.params)
        params_std = np.sqrt(np.maximum(np.diag(self.params_cov), 0.0))
        nominal = motionAtParams(self.params)

        grad = np.zeros((len(coord_arr), 2, n_params))
        for j in range(n_params):

            # Step by a tenth of the uncertainty of the parameter
            step = 0.1*params_std[j]
            if not (np.isfinite(step) and (step > 0)):
                step = 1e-6*max(abs(self.params[j]), 1.0)

            params_step = np.zeros(n_params)
            params_step[j] = step
            plus = motionAtParams(self.params + params_step)
            minus = motionAtParams(self.params - params_step)

            # Take central differences, one-sided where a step leaves the modelled part
            grad_j = (plus - minus)/(2*step)
            grad_j = np.where(np.isnan(minus), (plus - nominal)/step, grad_j)
            grad_j = np.where(np.isnan(plus), (nominal - minus)/step, grad_j)
            grad[:, :, j] = grad_j


        return grad[:, 0, :], grad[:, 1, :]



    def motionStdAtLength(self, len_arr):
        """ Compute the uncertainties of the velocity and of the time at the given lengths along the
            trajectory, from the covariance of the parameters (the noise of the points only).

        Arguments:
            len_arr: [float or ndarray] Lengths along the trajectory from the state vector (m).

        Return:
            (vel_std, time_std):
                - vel_std: [ndarray] Standard deviations of the velocities (m/s), NaN outside the modelled
                    part and infinite if they depend on a direction not constrained by the data.
                - time_std: [ndarray] Standard deviations of the times (s).
        """

        grad_vel, grad_time = self._motionGradients(np.atleast_1d(len_arr).astype(np.float64), False)

        return self._paramStd(grad_vel), self._paramStd(grad_time)



    def motionStdAtTime(self, time_arr):
        """ Compute the uncertainties of the velocity and of the length along the trajectory at the given
            times, from the covariance of the parameters (the noise of the points only).

        Arguments:
            time_arr: [float or ndarray] Times relative to the reference Julian date (s).

        Return:
            (vel_std, len_std):
                - vel_std: [ndarray] Standard deviations of the velocities (m/s), NaN outside the modelled
                    part and infinite if they depend on a direction not constrained by the data.
                - len_std: [ndarray] Standard deviations of the lengths along the trajectory (m).
        """

        grad_vel, grad_len = self._motionGradients(np.atleast_1d(time_arr).astype(np.float64), True)

        return self._paramStd(grad_vel), self._paramStd(grad_len)



    def ballisticCoeffGrowth(self):
        """ Compute the growth of the ballistic coefficient between the first and the last observed point,
            and the equivalent mass loss of a single body.

        Return:
            (k_ratio, mass_ratio):
                - k_ratio: [float] Ratio of the ballistic coefficients at the last and the first point.
                - mass_ratio: [float] Ratio of the masses at the last and the first point of a single body
                    of a constant shape, which has K proportional to m**(-1/3).
        """

        k_beg, k_end = self.ballisticCoeffAtLength(np.array([self.len_first, self.len_last]))[0]
        k_ratio = k_end/k_beg

        return k_ratio, k_ratio**(-3)



    def dynamicMassAtLength(self, len_arr, gamma_a=GAMMA_A, bulk_density=BULK_DENSITY):
        """ Compute the dynamic mass of the leading body at the given lengths along the trajectory from the
            fitted ballistic coefficient, m = (Gamma*A/(K*rho_m^(2/3)))^3, for an assumed shape and density.
            The uncertainty of K from the fit is propagated, as sigma(ln m) = 3*sigma(ln K).

        Arguments:
            len_arr: [float or ndarray] Lengths along the trajectory from the state vector (m).

        Keyword arguments:
            gamma_a: [float] Product of the drag coefficient Gamma and the shape factor A. 0.78 by default.
            bulk_density: [float] Bulk density (kg/m^3). 3500 by default.

        Return:
            (mass, mass_lo, mass_hi):
                - mass: [ndarray] Dynamic masses (kg), NaN where the meteor moves without drag or outside
                    the modelled part.
                - mass_lo: [ndarray] Lower 1 sigma limits of the masses (kg).
                - mass_hi: [ndarray] Upper 1 sigma limits of the masses (kg).
        """

        k_arr, log_k_std = self.ballisticCoeffAtLength(len_arr)
        k_arr = np.where(k_arr > 0, k_arr, np.nan)

        mass = ballisticCoeffMass(k_arr, bulk_density, gamma_a)

        # The mass is proportional to K^-3, so its logarithm has three times the uncertainty
        with np.errstate(over='ignore', invalid='ignore'):
            mass_lo = mass*np.exp(-3*log_k_std)
            mass_hi = mass*np.exp(3*log_k_std)

        return mass, mass_lo, mass_hi



    def summaryTable(self, gamma_a=GAMMA_A, bulk_density=BULK_DENSITY):
        """ Make a table of the fitted model, with the ballistic coefficient in every segment of the observed
            part, and the final dynamic mass.

        Keyword arguments:
            gamma_a: [float] Product of the drag coefficient Gamma and the shape factor A, assumed for the
                dynamic mass. 0.78 by default.
            bulk_density: [float] Bulk density assumed for the dynamic mass (kg/m^3). 3500 by default.

        Return:
            [str] Table, ready for printing.
        """

        n_data = len(self.len_data)
        k_ratio, mass_ratio = self.ballisticCoeffGrowth()

        if self.drag_above:
            above_str = "the ballistic coefficient at the first point"
        else:
            above_str = "no drag"

        # Superscript 2 for the units
        sq = "\u00b2"

        out = []
        out.append("Drag model of the motion: dv/dt = -K*rho*v" + sq + " + g")
        out.append(("    Points: {:d}, segments of K: {:d}, residuals RMS = {:.1f} m, robust scale = " \
            + "{:.1f} m").format(n_data, self.n_knots, self.res_rms, self.res_scale))
        out.append("    First point: t = {:.3f} s, v = {:.3f} km/s, h = {:.3f} km".format(self.params[0], \
            self.params[1]/1000, self.heightAtLength(self.len_first)[0]/1000))
        out.append("    Gravity drop: {:s}, above the first point: {:s}".format( \
            "yes" if self.gravity_correction else "no", above_str))
        out.append("    K end/beg = {:.2f}, single-body m end/beg = {:.3g}".format(k_ratio, mass_ratio))

        # Report the dynamic mass at the last point
        mass, mass_lo, mass_hi = self.dynamicMassAtLength(self.len_last, gamma_a=gamma_a, \
            bulk_density=bulk_density)
        out.append(("    Final dynamic mass = {:.3g} kg (1 sigma {:.3g} - {:.3g} kg), for Gamma*A = {:.2f} " \
            + "and bulk density = {:.0f} kg/m\u00b3").format(mass[0], mass_lo[0], mass_hi[0], gamma_a, \
            bulk_density))
        out.append("")

        # Compute the uncertainties of the growth of K in every segment
        d_std = self._paramStd(np.eye(len(self.params))[3:])

        out.append(" Seg, Ht beg, Ht end, Time beg,  Vel beg,  Dyn pres,     K beg,     K end, K std, " \
            + "Growth, Signif")
        out.append(("    ,   (km),   (km),      (s),   (km/s), beg (kPa),   (m{:s}/kg),   (m{:s}/kg), " \
            + "  (%),    (x), (sigma)").format(sq, sq))

        for i in range(self.n_knots):

            len_seg = np.array([self.ramp_beg[i], self.ramp_beg[i] + self.ramp_len])
            ht_seg = self.heightAtLength(len_seg)
            time_seg = self.timeAtLength(len_seg)
            vel_seg = self.velocityAtLength(len_seg)
            k_seg, log_k_std = self.ballisticCoeffAtLength(len_seg)

            # Compute the dynamic pressure at the beginning of the segment
            rho_beg = np.exp(np.interp(ht_seg[0], self.ht_atm, self.log_rho_atm))
            dyn_pres = rho_beg*self.groundVelocityAtLength(len_seg[0])[0]**2

            # Compute the significance of the growth of K in the segment
            if d_std[i] > 0:
                signif = self.params[3 + i]/d_std[i]
            else:
                signif = 0.0

            out.append(("{:4d}, {:6.2f}, {:6.2f}, {:8.3f}, {:8.3f}, {:9.1f}, {:9.2e}, {:9.2e}, {:5.0f}, " \
                + "{:6.2f}, {:6.1f}{:s}").format(i + 1, ht_seg[0]/1000, ht_seg[1]/1000, time_seg[0], \
                vel_seg[0]/1000, dyn_pres/1000, k_seg[0], k_seg[1], 100*log_k_std[1], \
                np.exp(self.params[3 + i]), signif, " *" if signif > 2 else ""))

        out.append("The star * marks the segments where K grows significantly (more than 2 sigma). K std " \
            + "is the relative uncertainty of K at the segment end.")
        out.append("Dyn pres is the dynamic pressure rho*v^2 at the segment beginning, with the velocity " \
            + "relative to the air, assumed at rest relative to the ground.")
        out.append("The final dynamic mass is the mass of the leading body at the last point for the " \
            + "assumed Gamma*A and density, m = (Gamma*A/(K*rho_m^(2/3)))^3. Only the uncertainty of K is " \
            + "propagated.")

        return "\n".join(out)



def plotDragFit(drag_fit, show=True, gamma_a=None, bulk_density=BULK_DENSITY):
    """ Plot the observed heights and velocities together with the drag model, and the fitted ballistic
        coefficient with its uncertainty.

    Arguments:
        drag_fit: [DragFit object] Drag fit of the motion along the trajectory.

    Keyword arguments:
        show: [bool] Show the plot. True by default.
        gamma_a: [float] Product of the drag coefficient Gamma and the shape factor A, assumed for the final
            dynamic mass shown in the plot. None by default, in which case the mass is not shown.
        bulk_density: [float] Bulk density assumed for the final dynamic mass (kg/m^3). 3500 by default.

    Return:
        fig: [matplotlib figure] Figure with the plots.
    """

    traj = drag_fit.traj

    fig, (ax_ht, ax_vel, ax_k) = plt.subplots(ncols=3, sharey=True, figsize=(14, 5))

    # Plot the station data
    for obs in traj.observations:

        if obs.ignore_station:
            continue

        used = np.array(obs.ignore_list) == 0

        ax_ht.scatter(obs.time_data[used], obs.model_ht[used]/1000, label=obs.station_id, marker='x', s=10, \
            zorder=3)

        # The first point of every station has no velocity
        used_vel = used.copy()
        used_vel[0] = False
        ax_vel.scatter(obs.velocities[used_vel]/1000, obs.model_ht[used_vel]/1000, marker='x', s=10, zorder=3)


    # Plot the model, with the extrapolated parts dashed
    k_grid, log_k_std = drag_fit.ballisticCoeffAtLength(drag_fit.len_grid)
    valid = np.isfinite(drag_fit.time_grid)
    observed = valid & (drag_fit.len_grid >= drag_fit.len_first) & (drag_fit.len_grid <= drag_fit.len_last)
    for sel, line_style, label in [(observed, 'solid', 'Model'), \
        (valid & ~observed, 'dashed', 'Extrapolated')]:

        time_plot = np.where(sel, drag_fit.time_grid, np.nan)
        ht_plot = np.where(sel, drag_fit.ht_grid, np.nan)
        vel_plot = np.where(sel, drag_fit.vel_grid, np.nan)
        k_plot = np.where(sel & (k_grid > 0), k_grid, np.nan)

        ax_ht.plot(time_plot, ht_plot/1000, color='k', linestyle=line_style, label=label, zorder=4)
        ax_vel.plot(vel_plot/1000, ht_plot/1000, color='k', linestyle=line_style, zorder=4)
        ax_k.plot(k_plot, ht_plot/1000, color='k', linestyle=line_style, zorder=4)


    # Plot the 1 sigma uncertainty of the ballistic coefficient in the observed part, in the logarithm so
    #   that a large uncertainty does not overflow. The limits of the plot are set by the uncertainties
    #   smaller than a factor of 100, and where K is less constrained the band covers the whole plot
    log_k = np.log(k_grid[observed])
    log_std = log_k_std[observed]

    bounded = log_std < np.log(100.0)
    log_min = np.min(log_k) - np.log(2.0)
    log_max = np.max(log_k) + np.log(2.0)
    if np.any(bounded):
        log_min = min(log_min, np.min(log_k[bounded] - log_std[bounded]) - np.log(2.0))
        log_max = max(log_max, np.max(log_k[bounded] + log_std[bounded]) + np.log(2.0))

    k_min = np.exp(log_min)
    k_max = np.exp(log_max)

    with np.errstate(invalid='ignore'):
        k_lo = np.exp(np.clip(log_k - log_std, log_min, log_max))
        k_hi = np.exp(np.clip(log_k + log_std, log_min, log_max))

    ax_k.fill_betweenx(drag_fit.ht_grid[observed]/1000, k_lo, k_hi, color='0.7', label='1$\\sigma$', zorder=2)


    # Show the growth of the ballistic coefficient in the observed part and the equivalent mass loss of a
    #   single body
    k_ratio, mass_ratio = drag_fit.ballisticCoeffGrowth()

    k_text = "$K_{\\mathrm{end}}/K_{\\mathrm{beg}}$ = " + "{:.2f}".format(k_ratio) + ", single body " \
        + "$m_{\\mathrm{end}}/m_{\\mathrm{beg}}$ = " + "{:.3g}".format(mass_ratio)

    # Add the final dynamic mass with its 1 sigma range
    if gamma_a is not None:

        mass, mass_lo, mass_hi = drag_fit.dynamicMassAtLength(drag_fit.len_last, gamma_a=gamma_a, \
            bulk_density=bulk_density)
        k_text += "\n" + "Final dynamic mass = {:.3g} kg ({:.3g} - {:.3g})".format(mass[0], mass_lo[0], \
            mass_hi[0]) + "\n" + "for $\\Gamma A$ = {:.2f}, ".format(gamma_a) + "$\\rho_m$ = " \
            + "{:.0f} kg/m$^3$".format(bulk_density)

    # Show the text above the plot, so it does not cover the data
    ax_k.set_title(k_text, fontsize=9)


    # Limit the plot to the observed part with a margin
    ht_beg, ht_end = drag_fit.heightAtLength(np.array([drag_fit.len_first, drag_fit.len_last]))
    ht_margin = 0.2*abs(ht_beg - ht_end)
    ax_ht.set_ylim(ht_end/1000 - ht_margin/1000, ht_beg/1000 + ht_margin/1000)

    time_beg, time_end = drag_fit.timeAtLength(np.array([drag_fit.len_first, drag_fit.len_last]))
    time_margin = 0.2*(time_end - time_beg)
    ax_ht.set_xlim(time_beg - time_margin, time_end + time_margin)

    vel_obs = np.concatenate([obs.velocities[1:] for obs in traj.observations if not obs.ignore_station])
    ax_vel.set_xlim(0, 1.2*np.percentile(vel_obs, 95)/1000)

    ax_k.set_xscale('log')
    ax_k.set_xlim(k_min, k_max)

    ax_ht.set_xlabel('Time (s)')
    ax_ht.set_ylabel('Height (km)')
    ax_vel.set_xlabel('Velocity (km/s)')
    ax_k.set_xlabel('Ballistic coefficient $K$ (m$^2$/kg)')

    ax_ht.legend()
    ax_k.legend(loc='upper right')
    ax_ht.grid()
    ax_vel.grid()
    ax_k.grid(which='both', alpha=0.5)

    plt.tight_layout()

    if show:
        plt.show()

    return fig




if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description=""" Fit the drag model of the motion along the given
        trajectory, print the table of the fitted ballistic coefficient and the dynamic mass, and plot the
        model. """)

    arg_parser.add_argument('traj_pickle_file', type=str, help='Path to the trajectory .pickle file.')

    arg_parser.add_argument('-k', '--knots', type=int, \
        help='Number of segments of the ballistic coefficient. One for every 25 points by default, between ' \
        + '3 and 15.')

    arg_parser.add_argument('-a', '--dragabove', action='store_true', \
        help='Compute the drag above the beginning with the ballistic coefficient at the beginning, ' \
        + 'instead of moving without drag.')

    arg_parser.add_argument('-g', '--ga', metavar='GAMMA_A', type=float, default=GAMMA_A, \
        help='The product of the drag coefficient Gamma and the shape coefficient A, used for computing ' \
        + 'the dynamic mass. Default is {:.2f}.'.format(GAMMA_A))

    arg_parser.add_argument('-d', '--dens', metavar='DENS', type=float, default=BULK_DENSITY, \
        help='Bulk density in kg/m^3 used for computing the dynamic mass. Default is {:.0f} kg/m^3.'.format( \
        BULK_DENSITY))

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    ############################


    # Load the trajectory file
    dir_path, file_name = os.path.split(cml_args.traj_pickle_file)
    traj = loadPickle(dir_path, file_name)

    # Fit the drag model
    drag_fit = DragFit(traj, n_knots=cml_args.knots, drag_above=cml_args.dragabove)

    print(drag_fit.summaryTable(gamma_a=cml_args.ga, bulk_density=cml_args.dens))

    plotDragFit(drag_fit, gamma_a=cml_args.ga, bulk_density=cml_args.dens)
