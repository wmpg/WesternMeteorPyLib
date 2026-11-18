""" Fit of the velocity of a meteor along a straight line by the drag equation, fast enough to be done in
every solution of the trajectory.

The meteor is decelerated by the air and accelerated by the component of gravity along the line:

    dv/dt = -K*rho*v**2 + g_along

where rho is the air density and K the ballistic coefficient (Gamma*A/(m**(1/3)*rho_m**(2/3))). With the
specific kinetic energy E = v**2/2 and the length L along the line, the equation is linear,

    dE/dL = -2*K*rho*E + g_along,

and is solved exactly on a grid of lengths. Ablation and fragmentation can only increase K, so the logarithm
of K is a sum of non-negative ramps in L. The model is fitted to the lengths and times of the observed points
with a robust loss and a small penalty on the growth of K, using the derivatives of the solution by the
parameters.

This is the model of wmpl.Utils.DragFit, restricted to the observed part of the trajectory, which the
trajectory solver needs for the gravity drop. Only the shape of the density profile matters, as a constant
factor is taken up by K, so the density is taken from the MSIS model with constant solar and geomagnetic
activity.
"""

from __future__ import print_function, division, absolute_import

import numpy as np
import scipy.optimize

from wmpl.Utils.AtmosphereDensity import getAtmDensity
from wmpl.Utils.TrajConversions import EARTH, G, cartesian2Geo


# Number of points of the grid of lengths on which the drag equation is solved
DRAG_VEL_GRID_POINTS = 200

# Number of points along the line at which the air density is computed
DRAG_VEL_ATM_POINTS = 25

# Logarithms of the constant ballistic coefficients which are searched for the initial fit (K in m^2/kg)
DRAG_VEL_LOG_K_SCAN = np.arange(-25.0, 8.01, 0.25)

# Bounds of the speed at the first point (m/s) and of the logarithm of K
DRAG_VEL_V_BOUNDS = (100.0, 2e5)
DRAG_VEL_LOG_K_BOUNDS = (-60.0, 10.0)

# Largest growth of the logarithm of K in one segment
DRAG_VEL_RAMP_MAX = 30.0

# Largest exponent of the integrated drag, beyond which the meteor has stopped, which keeps the sums finite
DRAG_VEL_PHI_MAX = 300.0



def lineDensityGravity(state_vect, radiant_eci, len_arr, jd, gravity_factor=1.0):
    """ Compute the air density and the component of gravity along a straight line.

    Arguments:
        state_vect: [ndarray] ECI coordinates of the point on the line from which the lengths are measured
            (m).
        radiant_eci: [ndarray] Unit vector of the radiant in ECI, opposite to the motion.
        len_arr: [ndarray] Lengths along the line in the direction of the motion (m), sorted.
        jd: [float] Julian date.

    Keyword arguments:
        gravity_factor: [float] Factor by which gravity is multiplied. 1.0 by default.

    Return:
        (rho, g_along):
            - rho: [ndarray] Air density at the lengths (kg/m^3).
            - g_along: [ndarray] Acceleration by gravity along the motion (m/s^2).
    """

    state_vect = np.array(state_vect, dtype=np.float64)
    radiant_eci = np.array(radiant_eci, dtype=np.float64)

    # Compute the geographical coordinates at a few points along the line
    len_atm = np.linspace(len_arr[0], len_arr[-1], DRAG_VEL_ATM_POINTS)
    lla = np.array([cartesian2Geo(jd, *(state_vect - len_val*radiant_eci)) for len_val in len_atm])

    # Compute the air density with constant solar activity, and interpolate its logarithm
    rho_atm = getAtmDensity(lla[:, 0], lla[:, 1], np.maximum(lla[:, 2], 0.0), jd, space_weather="constant")
    rho = np.exp(np.interp(len_arr, len_atm, np.log(rho_atm)))

    # Compute the component of the central gravity along the motion
    eci = state_vect[None, :] - len_arr[:, None]*radiant_eci[None, :]
    g_along = gravity_factor*G*EARTH.MASS*eci.dot(radiant_eci)/np.linalg.norm(eci, axis=1)**3

    return rho, g_along



def cumulativeTrapz(values, dlen):
    """ Integrate the given values on the grid with the trapezoidal rule, from the first point.

    Arguments:
        values: [ndarray] Values at the grid points, one row per point.
        dlen: [ndarray] Steps of the grid.

    Return:
        [ndarray] Integrals from the first point, with the same shape as the values.
    """

    steps = 0.5*(values[1:] + values[:-1])*dlen.reshape((-1,) + (1,)*(values.ndim - 1))

    return np.concatenate([np.zeros((1,) + values.shape[1:]), np.cumsum(steps, axis=0)])



class DragVelocity(object):
    def __init__(self, time_data, len_data, len_grid, rho, g_along, n_knots=None, penalty=1.0):
        """ Fit the drag equation to the times and lengths of the observed points.

        Arguments:
            time_data: [ndarray] Times of the points (s).
            len_data: [ndarray] Lengths of the points along the line (m).
            len_grid: [ndarray] Grid of lengths from the first to the last point (m).
            rho: [ndarray] Air density on the grid (kg/m^3).
            g_along: [ndarray] Acceleration by gravity along the motion on the grid (m/s^2).

        Keyword arguments:
            n_knots: [int] Number of segments of the ballistic coefficient. None by default, which uses one
                segment for every 25 points, but no fewer than 3 and no more than 15.
            penalty: [float] Weight of the penalty on the growth of the ballistic coefficient, relative to the
                residuals. 1.0 by default.
        """

        # Sort the points by length
        sort_ind = np.argsort(len_data)
        self.time_data = np.array(time_data, dtype=np.float64)[sort_ind]
        self.len_data = np.array(len_data, dtype=np.float64)[sort_ind]

        self.len_grid = np.array(len_grid, dtype=np.float64)
        self.dlen = np.diff(self.len_grid)
        self.rho = rho
        self.g_along = g_along

        # Set the ramps of the ballistic coefficient, uniformly spaced along the grid
        if n_knots is None:
            n_knots = int(np.clip(len(self.len_data)/25, 3, 15))

        self.n_knots = n_knots
        ramp_beg = np.linspace(self.len_grid[0], self.len_grid[-1], n_knots + 1)[:-1]
        ramp_len = (self.len_grid[-1] - self.len_grid[0])/n_knots
        self.ramps = np.clip((self.len_grid[:, None] - ramp_beg[None, :])/ramp_len, 0.0, 1.0)

        # Fit the model and compute the velocity on the grid
        self.params = self._fit(penalty)
        self.vel_grid, self.time_grid = self._solve(self.params)[:2]



    def _solve(self, params, derivatives=False):
        """ Compute the velocity and the time on the grid by the exact solution of the energy equation.

        Arguments:
            params: [ndarray] Model parameters (t_first, v_first, ln K at the first point, ramp heights).

        Keyword arguments:
            derivatives: [bool] Also compute the derivatives by the parameters. False by default.

        Return:
            (vel, time, dvel, dtime):
                - vel: [ndarray] Velocity on the grid (m/s).
                - time: [ndarray] Time on the grid (s).
                - dvel: [ndarray] Derivatives of the velocity by the parameters, None if not computed.
                - dtime: [ndarray] Derivatives of the time by the parameters, None if not computed.
        """

        # Compute the integral of 2*K*rho from the first point
        drag = 2*np.exp(params[2] + self.ramps.dot(params[3:]))*self.rho
        phi = cumulativeTrapz(drag, self.dlen)
        exp_phi = np.exp(np.clip(phi, None, DRAG_VEL_PHI_MAX))

        # Compute the energy, including the work done by gravity
        grav = cumulativeTrapz(self.g_along*exp_phi, self.dlen)
        energy = (0.5*params[1]**2 + grav)/exp_phi
        vel = np.sqrt(np.maximum(2*energy, 1.0))

        # Integrate the time
        time = params[0] + cumulativeTrapz(1/vel, self.dlen)

        if not derivatives:
            return vel, time, None, None


        # Differentiate the integral of the drag by ln K and the ramp heights
        dlog_k = np.c_[np.ones(len(drag)), self.ramps]
        dphi = cumulativeTrapz(drag[:, None]*dlog_k, self.dlen)

        # Differentiate the energy, and the velocity by v_first, ln K and the ramp heights
        dgrav = cumulativeTrapz((self.g_along*exp_phi)[:, None]*dphi, self.dlen)
        denergy = dgrav/exp_phi[:, None] - energy[:, None]*dphi
        dvel = np.c_[params[1]/exp_phi, denergy]/vel[:, None]

        # Differentiate the time by all parameters
        dtime = np.c_[np.ones(len(vel)), cumulativeTrapz(-dvel/vel[:, None]**2, self.dlen)]

        return vel, time, dvel, dtime



    def _residuals(self, params, w_pen):
        """ Compute the residuals of the model in meters along the line, and the penalty on the growth of the
            ballistic coefficient.

        Arguments:
            params: [ndarray] Model parameters (t_first, v_first, ln K at the first point, ramp heights).
            w_pen: [float] Weight of the penalty.

        Return:
            [ndarray] Residuals.
        """

        vel, time, _, _ = self._solve(params)

        # Compute the model lengths at the observed times, continuing with the end velocities outside the
        #   modelled time range
        len_model = np.interp(self.time_data, time, self.len_grid)
        len_model = np.where(self.time_data < time[0], self.len_grid[0] + vel[0]*(self.time_data - time[0]), \
            len_model)
        len_model = np.where(self.time_data > time[-1], \
            self.len_grid[-1] + vel[-1]*(self.time_data - time[-1]), len_model)

        return np.concatenate([len_model - self.len_data, w_pen*params[3:]])



    def _jacobian(self, params, w_pen):
        """ Compute the derivatives of the residuals by the parameters.

        Arguments:
            params: [ndarray] Model parameters (t_first, v_first, ln K at the first point, ramp heights).
            w_pen: [float] Weight of the penalty.

        Return:
            [ndarray] Jacobian, one row per residual.
        """

        vel, time, dvel, dtime = self._solve(params, derivatives=True)

        # Find the segment of the grid of every observed time
        ind = np.clip(np.searchsorted(time, self.time_data) - 1, 0, len(self.dlen) - 1)
        dtime_seg = time[ind + 1] - time[ind]
        frac = np.clip((self.time_data - time[ind])/dtime_seg, 0.0, 1.0)

        # The interpolated length moves against the change of the time, dL/dp = -(dt/dp)/(dt/dL)
        dtime_at = dtime[ind]*(1 - frac)[:, None] + dtime[ind + 1]*frac[:, None]
        jac = -dtime_at*(self.dlen[ind]/dtime_seg)[:, None]

        # Differentiate the continuation with the end velocities outside the modelled time range
        before = self.time_data < time[0]
        jac[before] = -vel[0]*dtime[0] + (self.time_data[before] - time[0])[:, None]*np.r_[0.0, dvel[0]]

        after = self.time_data > time[-1]
        jac[after] = -vel[-1]*dtime[-1] + (self.time_data[after] - time[-1])[:, None]*np.r_[0.0, dvel[-1]]

        # Add the derivatives of the penalty
        jac_pen = np.c_[np.zeros((self.n_knots, 3)), w_pen*np.eye(self.n_knots)]

        return np.r_[jac, jac_pen]



    def _scanConstant(self):
        """ Find the constant ballistic coefficient which fits best, without gravity. For a given K, the time
            is t_first + F(L)/v_first, with F the integral of exp(K*int(rho dL)), which is linear in t_first
            and 1/v_first.

        Return:
            (t_first, v_first, log_k, res):
                - t_first: [float] Time of the first point of the grid (s).
                - v_first: [float] Velocity at the first point of the grid (m/s).
                - log_k: [float] Logarithm of the ballistic coefficient.
                - res: [ndarray] Residuals of the fit (m).
        """

        # Compute F for all coefficients at once
        drag = 2*np.exp(DRAG_VEL_LOG_K_SCAN)[None, :]*self.rho[:, None]
        phi = cumulativeTrapz(drag, self.dlen)
        f_grid = cumulativeTrapz(np.exp(np.clip(0.5*phi, None, DRAG_VEL_PHI_MAX)), self.dlen)

        # Convert the residuals in time to lengths with the mean velocity
        duration = np.max(self.time_data) - np.min(self.time_data)
        vel_mean = (self.len_data[-1] - self.len_data[0])/max(duration, 1e-6)

        best = None
        for i, log_k in enumerate(DRAG_VEL_LOG_K_SCAN):

            # Fit the time of the first point and the inverse of the velocity
            f_data = np.interp(self.len_data, self.len_grid, f_grid[:, i])
            design = np.c_[np.ones_like(f_data), f_data]
            (t_first, slowness), _, _, _ = np.linalg.lstsq(design, self.time_data, rcond=None)

            if slowness <= 0:
                continue

            res = (design.dot([t_first, slowness]) - self.time_data)*vel_mean
            cost = np.sum(res**2)

            if (best is None) or (cost < best[0]):
                best = (cost, t_first, 1/slowness, log_k, res)


        if best is None:
            raise ValueError("The drag model could not be fitted, the lengths do not grow with time")

        return best[1:]



    def _fit(self, penalty):
        """ Fit the model parameters to the data.

        Arguments:
            penalty: [float] Weight of the penalty on the growth of the ballistic coefficient.

        Return:
            [ndarray] Fitted parameters.
        """

        n_knots = self.n_knots
        lower = np.r_[-np.inf, DRAG_VEL_V_BOUNDS[0], DRAG_VEL_LOG_K_BOUNDS[0], np.zeros(n_knots)]
        upper = np.r_[np.inf, DRAG_VEL_V_BOUNDS[1], DRAG_VEL_LOG_K_BOUNDS[1], \
            np.full(n_knots, DRAG_VEL_RAMP_MAX)]

        # Start from the best constant ballistic coefficient
        t_first, v_first, log_k, res = self._scanConstant()

        # Fit the full model, starting both from the constant coefficient and from one which grows by a factor
        #   of 10 over the observed part, and keep the better fit
        best = None
        for d_init in (1e-3, np.log(10.0)/n_knots):

            params_init = np.r_[t_first, v_first, log_k - n_knots*d_init/2, np.full(n_knots, d_init)]
            params_init = np.clip(params_init, lower + 1e-9, upper - 1e-9)

            sol = self._leastSquares(params_init, res, penalty, lower, upper)
            if (best is None) or (sol.cost < best.cost):
                best = sol


        # Fit again with the scale of the residuals of the fitted model, which sets the robust loss and the
        #   penalty
        res = self._residuals(best.x, 0.0)[:len(self.len_data)]

        return self._leastSquares(best.x, res, penalty, lower, upper).x



    def _leastSquares(self, params_init, res, penalty, lower, upper):
        """ Fit the model with a robust loss, whose scale is taken from the given residuals.

        Arguments:
            params_init: [ndarray] Initial parameters.
            res: [ndarray] Residuals from which the scale is estimated (m).
            penalty: [float] Weight of the penalty on the growth of the ballistic coefficient.
            lower: [ndarray] Lower bounds of the parameters.
            upper: [ndarray] Upper bounds of the parameters.

        Return:
            [OptimizeResult] Solution of scipy.optimize.least_squares.
        """

        res_scale = max(1.4826*np.median(np.abs(res - np.median(res))), 1.0)
        f_scale = 2*res_scale
        w_pen = np.sqrt(penalty)*f_scale

        return scipy.optimize.least_squares(self._residuals, params_init, jac=self._jacobian, args=(w_pen,), \
            bounds=(lower, upper), loss='soft_l1', f_scale=f_scale, x_scale='jac', ftol=1e-6, xtol=1e-6, \
            max_nfev=1000)



    def velocityAtTime(self, time_arr):
        """ Compute the velocity at the given times, kept at the end values outside the observed part.

        Arguments:
            time_arr: [ndarray] Times (s).

        Return:
            [ndarray] Velocities (m/s).
        """

        return np.interp(time_arr, self.time_grid, self.vel_grid)



def fitDragVelocity(time_data, len_data, state_vect, radiant_eci, jd, gravity_factor=1.0):
    """ Fit the velocity of the meteor by the drag equation to the times and lengths of the observed points.

    Arguments:
        time_data: [ndarray] Times of the points (s).
        len_data: [ndarray] Lengths of the points along the line from the state vector (m).
        state_vect: [ndarray] ECI coordinates of the state vector (m).
        radiant_eci: [ndarray] Unit vector of the radiant in ECI.
        jd: [float] Julian date.

    Keyword arguments:
        gravity_factor: [float] Factor by which gravity is multiplied. 1.0 by default.

    Return:
        [DragVelocity object] Fitted model.
    """

    len_grid = np.linspace(np.min(len_data), np.max(len_data), DRAG_VEL_GRID_POINTS)
    rho, g_along = lineDensityGravity(state_vect, radiant_eci, len_grid, jd, gravity_factor=gravity_factor)

    return DragVelocity(time_data, len_data, len_grid, rho, g_along)
