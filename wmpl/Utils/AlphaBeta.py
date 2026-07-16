""" Defines functions for the alpha-beta fireball characterization by Gritsevich 2012.
Adapted from: https://github.com/desertfireballnetwork/alpha_beta_modules

Besides the dynamics-only fit (fitAlphaBeta) and the mass-constrained fit (fitAlphaBetaMass), this
module also fits the alpha-beta model jointly to the dynamics AND the light curve
(fitAlphaBetaLightCurve), following Gritsevich & Koschny (2011) for the luminosity - see that
function's docstring for the full mathematical formulation. Its light curve amplitude K can be
turned into a luminous efficiency tau via alphaBetaLuminousEfficiency(), a separate post-processing
step (it needs a density/shape/drag assumption fitAlphaBetaLightCurve() itself does not make).
"""


import sys
import numpy as np
import scipy.special
import scipy.optimize


from wmpl.Utils.Math import meanAngle
from wmpl.Utils.Physics import dynamicPressure, dynamicMass
from wmpl.Utils.AtmosphereDensity import getAtmDensity_vect


# Scale height
HT_NORM_CONST = 7160.0

# Power of a zero absolute magnitude meteor (W), WMPL convention. Used to turn the
# fitted light curve amplitude (a magnitude offset) into the physical amplitude K of
# Gritsevich & Koschny (2011) Eq. (13): I = K*f(v), K = P_0M*10^(-0.4*mag_offset).
P_0M = 840.0

# (alpha, beta) bounds shared by every alpha-beta optimizer in this module (minimizeAlphaBeta()
# and fitAlphaBetaLightCurve()), so Stage-0 (Q4) and joint fits are always constrained the same way.
ALPHA_BETA_BOUNDS = ((0.001, 10000.0), (0.00001, 50.0))



# Example input data
# # Height (m), velocity (m/s)
# input_data = np.array([
#     [83681.0215156, 17640.9079043],
#     [83115.2043356, 21738.210588],
#     [82667.2323107, 17229.3030977],
#     [82205.16874, 17788.5556192],
#     [81720.7954369, 18666.3386775],
#     [81311.5861921, 15784.8581581],
#     [80795.4457445, 19929.4459506],
#     [80346.1029557, 17368.2836897],
#     [79929.9709969, 16099.6372199],
#     [79440.9610398, 18937.7398444],
#     [78989.9618378, 17483.5240729],
#     [78520.2057116, 18228.8946485],
#     [78028.4348476, 19103.1785038],
#     [77593.5820248, 16909.2582178],
#     [77160.1939563, 16868.3210314],
#     [76703.7753714, 17782.0583169],
#     [76228.0916978, 18551.6220006],
#     [75780.0116806, 17492.87328],
#     [75311.9978669, 18289.5704834],
#     [74865.7275078, 17457.5077072],
#     [74397.1073002, 18350.4138997],
#     [73935.5087991, 18094.1542407],
#     [73494.4889556, 17304.8853494],
#     [73054.6105159, 17277.0823985],
#     [72600.2502069, 17863.7516712],
#     [72140.2465104, 18104.1833689],
#     [71696.7922973, 17470.5832886],
#     [71241.2214444, 17966.1236285],
#     [70790.6254124, 17788.1234402],
#     [70325.6546885, 18374.6185794],
#     [69907.8932431, 16525.5475434],
#     [69463.9945446, 17576.6748037],
#     [69018.9109104, 17641.449356],
#     [68586.5036082, 17156.1850336],
#     [68105.0959818, 19120.3036409],
#     [67693.6798044, 16357.1580997],
#     [67244.9463149, 17858.47373],
#     [66802.5003205, 17626.3326231],
#     [66339.897621, 18448.6111149],
#     [65929.2353315, 16393.7833404],
#     [65488.9457817, 17593.8687598],
#     [65047.6052004, 17653.948825],
#     [64633.6975273, 16573.1150336],
#     [64207.3219617, 17089.0929273],
#     [63771.9106074, 17468.8511397],
#     [63319.0503533, 18187.8418027],
#     [62888.5377557, 17308.272412],
#     [62457.658419, 17340.6031692],
#     [62044.0753169, 16661.1234526],
#     [61628.6904629, 16750.1269718],
#     [61195.5424224, 17483.9985421],
#     [60773.3635024, 17058.5606553],
#     [60332.8187156, 17818.9521018],
#     [59928.3247354, 16377.2972957],
#     [59529.6086453, 16158.8789266],
#     [59120.1592054, 16609.9572187],
#     [58699.1780828, 17094.8049249],
#     [58296.9717338, 16348.6122603],
#     [57907.1198554, 15861.5876386],
#     [57495.2725451, 16772.7496274],
#     [57101.1348098, 16067.1997136],
#     [56703.7176405, 16216.4670862],
#     [56317.3483881, 15780.7046376],
#     [55918.7425435, 16296.0825456],
#     [55509.251914, 16757.6221384],
#     [55156.7129653, 14440.4520845],
#     [54764.5437031, 16078.4475594],
#     [54392.840789, 15253.6728233],
#     [54034.4700671, 14719.7902665],
#     [53633.6075146, 16480.5484838],
#     [53309.5933381, 13333.0525322],
#     [52939.3494293, 15248.5028132],
#     [52616.7653049, 13297.0678182],
#     [52248.4094022, 15196.8595596],
#     [51917.2901026, 13672.5442947],
#     [51572.4695619, 14250.318074],
#     [51232.9315025, 14044.0294958],
#     [50951.8414222, 11635.5414157],
#     [50605.9098419, 14330.8905801],
#     [50359.1027647, 10232.0885526],
#     [50016.7389018, 14204.2047434],
#     [49720.5878298, 12296.8024016],
#     [49432.4897207, 11971.2633929],
#     [49141.4612413, 12101.897341],
#     [48893.4022872, 10322.1392339],
#     [48625.757255, 11144.4432081],
#     [48419.7730391, 8582.12152838],
#     [48139.5273914, 11683.3788851]])



def rescaleHeightToExponentialAtmosphere(lat, lon, ht_data, jd):
    """ Given observed heights, rescale them from the real NRLMSISE model to the a simplified exponential
        atmosphere model used by the Alpha-Beta procedure.
    
    Arguments:
        lat: [ndarray] Latitude in radians.
        lon: [ndarray] Longitude in radians.
        ht_data: [ndarray] Height in meters.
        jd: [float] Julian date.

    Return:
        rescaled_ht_data
    """

    def _expAtmosphere(ht_data, rho_atm_0=1.0):
        """ Compute the atmosphere mass density using a simple exponential model and a scale height. 
    
        Arguments:
            ht_data: [ndarray] Height in meters.

        Keyword arguments: 
            rho_atm_0: [float] Sea-level atmospheric air density in kg/m^3.

        Return:
            [float] Atmospheric mass density in kg/m^3.
        """

        return rho_atm_0*(1/np.e**(ht_data/HT_NORM_CONST))

    def _expAtmosphereHeight(air_density, rho_atm_0=1.225):
        """ Compute the height given the air density and exponential atmosphere assumption. 

        Arguments:
            air_density: [float] Air density in kg/m^3.

        Keyword arguments: 
            rho_atm_0: [float] Sea-level atmospheric air density in kg/m^3.

        Return:
            [float] Height in meters.
        """

        return HT_NORM_CONST*np.log(rho_atm_0/air_density)


    # Get the atmosphere mass density from the NRLMSISE model for the observed heights
    atm_dens = getAtmDensity_vect(lat, lon, ht_data, jd)

    # Get the equivalent heights using the exponential atmosphre model
    ht_rescaled = _expAtmosphereHeight(atm_dens)

    # # Compare the models
    # plt.semilogy(ht_data/1000, atm_dens, label='NRLMSISE')
    # plt.semilogy(ht_data/1000, _expAtmosphere(ht_data), label='Exp')
    # plt.xlabel("Height (km)")
    # plt.ylabel("log air density kg/m3")
    # plt.legend()
    # plt.show()

    # # Compare the heights before and after rescaling
    # plt.scatter(ht_data/1000, ht_data - ht_rescaled)
    # plt.xlabel("Height (km)")
    # plt.ylabel("Height difference (m)")
    # plt.show()
    # sys.exit()

    return ht_rescaled


def expLinearLag(t, a1, a2, t0, decel):
    """ Model the lag by assuming that the deceleration is exponential until a point t0, after which
        the deceleration is constant.
    """

    # Normalize deceleration for faster convergence
    decel = -1000*abs(decel)

    lag = np.zeros_like(t)

    # Initial part computed with exponential deceleration
    lag[t <  t0] = abs(a1) - abs(a1)*np.exp(abs(a2)*t[t < t0])

    # Second part computed with constant deceleration
    lag[t >= t0] = (-abs(a1)*np.exp(abs(a2)*t0) # Continue at the last point
                    - abs(a1*a2)*np.exp(abs(a2)*t0)*((t[t >= t0] - t0)) # Continue with the same velocity
                    + ((t[t >= t0] - t0)**2)*decel/2.0) # Apply constant deceleration

    return lag


def expLinearVelocity(t, v0, a1, a2, t0, decel):

    # Normalize deceleration for faster convergence
    decel = -1000*abs(decel)

    vel = np.zeros_like(t)

    vel += v0
    vel[t <  t0] += -abs(a1*a2)*np.exp(abs(a2)*t[t < t0])
    vel[t >= t0] += -abs(a1*a2)*np.exp(abs(a2)*t0) + (t[t >= t0] - t0)*decel

    return vel


def lagFitVelocity(time_data, lag_data, vel_data, v0):
    """ Fit a smooth model to the lag data, to improve the alpha-beta fit. """


    def _lagMinimization(params, time_data, lag_data, weights):

        # Compute the sum of absolute residuals (more robust than squared residuals)
        cost = np.sum(weights*np.abs(lag_data - expLinearLag(time_data, *params)))

        return cost


    # Guess initial parameters
    a1 = 20
    a2 = 1.5
    t0 = 9/10*np.max(time_data) # The transition to constant deceleration always happens close to the end
    decel = 6 # km/s^2, typical deceleration for meteorite droppers at the end

    # Initial parameters
    p0 = [a1, a2, t0, decel]

    # Fit the lag function
    #fit_params, _ = scipy.optimize.curve_fit(expLinearLag, time_data, lag_data, p0=p0, maxfev=10000)


    # # Use weights such that they linearly increase from 0.5 at and before the first half of the fireball to 
    # #   1.0 at the end
    # # The time is sorted in reverse, so take that into account
    # weights = np.zeros_like(time_data)
    # first_part_indices = np.arange(0, len(weights)/2).astype(int)
    # weights[first_part_indices] = 1.0 - 0.5*first_part_indices/np.max(first_part_indices)
    # weights[~first_part_indices] = 0.5
    # weights /= np.sum(weights)

    # Don't use weights
    weights = np.ones_like(time_data)

    # Use robust fitting
    res = scipy.optimize.basinhopping(_lagMinimization, p0, niter=200, T=2.0,\
        minimizer_kwargs={'args':(time_data, lag_data, weights), 'method':'Nelder-Mead'})
    fit_params = res.x

    # fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
    
    # # Plot the data
    # ax1.scatter(time_data, lag_data)

    # # Plot the fit
    # time_arr = np.linspace(np.min(time_data), np.max(time_data), 100)
    # ax1.plot(time_arr, expLinearLag(time_arr, *fit_params), color='k', zorder=5)

    # # Plot the residuals
    # ax2.scatter(time_data, lag_data - expLinearLag(time_data, *fit_params))


    # # Plot the observed velocity and the velocity fit
    # ax3.scatter(time_data, vel_data/1000)
    # ax3.plot(time_arr, expLinearVelocity(time_arr, v0,  *fit_params)/1000)

    # plt.show()

    # sys.exit()

    # Compute fitted velocity
    vel_fit = expLinearVelocity(time_data, v0,  *fit_params)

    return vel_fit, fit_params


def _alphaBetaResidual(alpha, beta, v_normed, ht_normed):
    """ Compute the residual of equation 7 using the Q4 minimisation given in equation 10 of
        Gritsevich 2007 - 'Validity of the photometric formula for estimating the mass of a fireball
        projectile'.

    Arguments:
        alpha: [float] Ballistic coefficient.
        beta: [float] Mass loss parameter.
        v_normed: [ndarray] Velocity data normalized to the initial velocity.
        ht_normed: [ndarray] Height data normalized to the scale height.

    Return:
        [float] Sum of absolute residuals (more robust than squared residuals).
    """

    return np.sum(np.abs(2*alpha*np.exp(-ht_normed) \
                      - (scipy.special.expi(beta) \
                         - scipy.special.expi(beta*v_normed**2))*np.exp(-beta)))


def _betaFromMasses(mu, mass_initial, mass_final, v_final, v_init):
    """ Analytically compute beta from the initial/final mass ratio and the observed velocity change,
        for a given shape change coefficient mu, by inverting the general alpha-beta mass loss solution
        m_final = m_init*exp(-beta/(1 - mu)*(1 - (v_final/v_init)^2)).

    Arguments:
        mu: [float] Shape change coefficient.
        mass_initial: [float] Initial mass (kg).
        mass_final: [float] Final mass (kg).
        v_final: [float] Final velocity (m/s).
        v_init: [float] Initial velocity (m/s).

    Return:
        [float] Mass loss parameter beta.
    """

    return (1.0 - mu)*np.log(mass_initial/mass_final)/(1.0 - (v_final/v_init)**2)


def _makeBoundChecker(warnings_list, tol=1e-2):
    """ Build a _checkBound(name, value, lower, upper) closure that appends a message to
        warnings_list whenever value lands within a relative tol of lower or upper. Shared by
        fitAlphaBetaMass() and fitAlphaBetaLightCurve() to flag fitted parameters that are
        poorly constrained (landed on an optimization bound rather than an interior optimum).

    Arguments:
        warnings_list: [list] List that warning strings are appended to.

    Keyword arguments:
        tol: [float] Relative distance-to-bound tolerance.

    Return:
        _checkBound: [callable] (name, value, lower, upper) -> None.
    """

    def _checkBound(name, value, lower, upper):

        span = upper - lower

        if (value - lower)/span < tol:
            warnings_list.append(
                "{:s}={:.6g} is close to the fit lower bound ({:.6g})".format(name, value, lower)
            )

        if (upper - value)/span < tol:
            warnings_list.append(
                "{:s}={:.6g} is close to the fit upper bound ({:.6g})".format(name, value, upper)
            )

    return _checkBound


def minimizeAlphaBeta(v_normed, ht_normed):
    """ initiates and calls the Q4 minimisation given in Gritsevich 2007 -
        'Validity of the photometric formula for estimating the mass of a fireball projectile'
    """

    def _alphaBetaMinimization(x, v_normed, ht_normed):
        """minimises equation 7 using Q4 minimisation given in equation 10 of
           Gritsevich 2007 - 'Validity of the photometric formula for estimating
           the mass of a fireball projectile'

        """

        alpha, beta = x

        return _alphaBetaResidual(alpha, beta, v_normed, ht_normed)

    # params = np.vstack((v_normed, ht_normed))

    # Compute initial alpha-beta guess
    b0 = 1.0
    a0 = np.exp(np.min(ht_normed))/(2.0*b0)
    x0 = [a0, b0]

    # Set alpha-beta limits
    xmin = [bound[0] for bound in ALPHA_BETA_BOUNDS]
    xmax = [bound[1] for bound in ALPHA_BETA_BOUNDS]
    bnds = ALPHA_BETA_BOUNDS

    # If the initial guess is outside the bounds, set a middle value
    for i, initial_param in enumerate(x0):
        if (initial_param < xmin[i]) or (initial_param > xmax[i]):
            x0[i] = (xmin[i] + xmax[i])/2

    # Compute best-fit alpha-beta values
    res = scipy.optimize.minimize(_alphaBetaMinimization, x0, args=(v_normed, ht_normed), bounds=bnds, \
        method='Nelder-Mead')

    return res.x


def _logAlphaBetaBounds():
    """ ALPHA_BETA_BOUNDS in log-space, as used by every least_squares-based fit in this module
        (minimizeAlphaBetaRobust(), fitAlphaBetaLightCurve()), which search in
        (ln alpha, ln beta) directly since alpha/beta span several orders of magnitude.

    Return:
        (lower, upper): [tuple of ndarrays] Lower and upper (ln alpha, ln beta) bounds, in the
            format scipy.optimize.least_squares expects.
    """

    return (
        np.log([ALPHA_BETA_BOUNDS[0][0], ALPHA_BETA_BOUNDS[1][0]]),
        np.log([ALPHA_BETA_BOUNDS[0][1], ALPHA_BETA_BOUNDS[1][1]])
    )


def _estimateSigmaV(alpha0, beta0, v_normed, ht_normed, v_init):
    """ Robust (MAD-based) velocity uncertainty (m/s) from the residuals of a Q4 fit
        (alpha0, beta0), floored at 1 m/s. Used by minimizeAlphaBetaRobust()/fitAlphaBeta() and
        fitAlphaBetaLightCurve() whenever sigma_v isn't given explicitly.

        This is an *effective* uncertainty that absorbs both measurement noise and alpha-beta
        model misspecification - it is not a pure instrumental/measurement uncertainty, even
        though it is used as one here.

    Arguments:
        alpha0: [float] Q4-fitted ballistic coefficient.
        beta0: [float] Q4-fitted mass loss parameter.
        v_normed: [ndarray] Normalized velocity in (0, 1).
        ht_normed: [ndarray] Height normalized to HT_NORM_CONST.
        v_init: [float] Initial velocity (m/s), used to convert the residuals to physical units.

    Return:
        sigma_v: [float] Effective velocity uncertainty (m/s), floored at 1 m/s.
    """

    v_model0 = alphaBetaVelocityNormed(ht_normed, alpha0, beta0)
    res0 = (v_model0 - v_normed)*v_init
    sigma_v = 1.4826*np.median(np.abs(res0 - np.median(res0)))

    return max(sigma_v, 1.0)


def _dynResiduals(x, v_normed, ht_normed, sigma_v_normed):
    """ Dynamics residual block: model velocity residuals (v_model - v_obs), sigma-normalized.
        Used both as the dynamics half of _jointResiduals() (fitAlphaBetaLightCurve()) and, on
        its own, by minimizeAlphaBetaRobust().

    Arguments:
        x: [ndarray] Fit parameters (ln alpha, ln beta).
        v_normed: [ndarray] Normalized velocity in (0, 1).
        ht_normed: [ndarray] Height normalized to HT_NORM_CONST.
        sigma_v_normed: [float] Normalized velocity uncertainty.

    Return:
        res: [ndarray] Sigma-normalized velocity residuals.
    """

    alpha, beta = np.exp(x)
    v_model = alphaBetaVelocityNormed(ht_normed, alpha, beta)

    return (v_model - v_normed)/sigma_v_normed


def minimizeAlphaBetaRobust(v_normed, ht_normed, sigma_v_normed, loss='soft_l1', f_scale=2.0,
        x0_alpha_beta=None, return_fit_result=False):
    """ Robust alternative to minimizeAlphaBeta(): least squares (soft-L1 by default) directly on
        the velocity residuals (v_model - v_obs), instead of Q4's transformed residual.

        Q4's residual (_alphaBetaResidual()) is NOT the velocity residual: its exp(-ht_normed)
        factor structurally shrinks the residual near the end of the trajectory, so Q4 implicitly
        under-weights exactly the region where the deceleration/beta signal is concentrated, in
        favor of the near-constant-velocity beginning. Minimizing the velocity residual directly
        (up to sigma_v_normed and the robust loss) weights the whole trajectory much more evenly,
        and tracks the deceleration/"knee" of the v(h) curve more faithfully as a result.

        Seeded from minimizeAlphaBeta()'s own (Q4) result if x0_alpha_beta isn't given: Q4 is
        cheap and gets close enough that the seed itself is never the issue here (contrast with
        fitAlphaBetaMass()'s mass_constraint="final", where a poor seed combined with a
        line-search method walked into a spurious minimum - see that function's comments).

    Arguments:
        v_normed: [ndarray] Normalized velocity in (0, 1).
        ht_normed: [ndarray] Height normalized to HT_NORM_CONST.
        sigma_v_normed: [float] Normalized velocity uncertainty (see _estimateSigmaV() to derive
            this from a physical-units sigma_v).

    Keyword arguments:
        loss: [str] scipy.optimize.least_squares loss ('soft_l1', 'huber', 'linear', ...).
        f_scale: [float] Robust loss scale, in units of sigma_v_normed.
        x0_alpha_beta: [tuple] Optional (alpha0, beta0) initial guess. If None, derived from
            minimizeAlphaBeta().
        return_fit_result: [bool] If True, also return the raw scipy.optimize.least_squares
            result object as a 3rd element, so a caller can reuse its Jacobian/cost for a
            Gauss-Newton covariance (see fitAlphaBeta()'s estimate_errors) instead of re-fitting
            or recomputing the Jacobian from scratch.

    Return:
        (alpha, beta), or (alpha, beta, res) if return_fit_result=True.
    """

    if x0_alpha_beta is None:
        x0_alpha_beta = minimizeAlphaBeta(v_normed, ht_normed)

    log_bounds = _logAlphaBetaBounds()

    # Clip the seed strictly inside the bounds in linear space before taking the log (a
    #   non-positive seed would otherwise produce a NaN that np.clip propagates)
    x0 = np.log(np.clip(x0_alpha_beta, np.exp(log_bounds[0] + 1e-6), np.exp(log_bounds[1] - 1e-6)))

    res = scipy.optimize.least_squares(_dynResiduals, x0, args=(v_normed, ht_normed, sigma_v_normed), \
        bounds=log_bounds, loss=loss, f_scale=f_scale, x_scale='jac')

    if not res.success:
        print("WARNING: Optimizer failed: {:s}".format(res.message))

    if return_fit_result:
        return tuple(np.exp(res.x)) + (res,)

    return tuple(np.exp(res.x))


def _alphaBetaRobustErrors(fit_res, alpha, beta, v_data, ht_data, v_init, v_init_given, sigma_v,
        sigma_v_init, loss, f_scale, ci):
    """ Analytic error estimate for fitAlphaBeta(method='robust', estimate_errors=True).

        Two contributions to the covariance of (ln alpha, ln beta):
        1. The Gauss-Newton local curvature of the robust-loss least_squares fit itself
           (fit_res.jac/.cost/.fun, reused from minimizeAlphaBetaRobust() rather than
           recomputed), treating v_init as if it were exactly known. Same (J^T J)^-1,
           reduced-chi-square-scaled recipe as fitAlphaBetaLightCurve()'s own
           alpha_std_rel/beta_std_rel, with the same caveat: soft-L1 reweights residuals
           nonlinearly, so this is a Gauss-Newton approximation of the local curvature, not the
           inverse Fisher information of Gaussian noise.
        2. A rank-1 correction J_v0 sigma_v_init^2 J_v0^T propagating v_init's OWN uncertainty
           through a numerically estimated sensitivity J_v0 = d(ln alpha, ln beta)/d(v_init) -
           obtained by refitting fitAlphaBeta() itself at v_init +/- a small step, since v_init is
           held fixed during the fit that produced fit_res and so never appears in fit_res.jac.
           Skipped (zero contribution) whenever sigma_v_init is 0 (v_init treated as exact).
        The two contributions are simply added (v_init's estimation error and the conditional
        fit's residuals are treated as independent, even though both ultimately come from the
        same v_data) - a standard plug-in approximation, not a fully joint treatment.

        cov_log/corr_log ((ln alpha, ln beta), the space this model is actually fit in) are
        propagated to linear (alpha, beta) space via the delta method: since
        d(alpha)/d(ln alpha) = alpha, the Jacobian of (alpha, beta) w.r.t. (ln alpha, ln beta) is
        diag(alpha, beta), so cov = diag(alpha, beta) @ cov_log @ diag(alpha, beta). corr comes
        out numerically identical to corr_log (correlation is invariant under this kind of
        positive-diagonal rescaling) - both are kept for convenience.

        alpha_ci_gaussian_lower/upper and beta_ci_gaussian_lower/upper are APPROXIMATE confidence
        intervals based on the local Gaussian approximation in log-space (equivalently: assuming
        alpha/beta are log-normal), at the `ci` level - exp(ln(alpha) +/- z*alpha_std_rel),
        asymmetric around alpha and, unlike alpha +/- z*alpha_std, never negative. The "_gaussian"
        in the name (and "approximate" here) is deliberate: this is NOT the same kind of object as
        wmpl.Utils.Math.confidenceInterval()'s output elsewhere in this library (e.g. Trajectory.py,
        REBOUND.py) - those are empirical percentiles of an actual Monte Carlo ensemble, whereas
        there is no ensemble here, only a parametric approximation. Don't drop the "_gaussian"
        qualifier when reading these out; it is the one thing standing between this and a false
        claim of MC-equivalent rigor.

        Two failure modes are handled explicitly, both printing a WARNING and returning NaN for
        every error field (alpha/beta themselves are still returned - only their errors are NaN):
        - (alpha, beta) landed within 1% (log-space) of an ALPHA_BETA_BOUNDS edge: the fit is
          likely poorly constrained by this data, and the local curvature below isn't meaningful
          around a boundary the optimizer was clipped against rather than converged to.
        - The Jacobian is near- (but not exactly) singular: (J^T J)^-1 comes out with a negative
          diagonal - impossible for a real covariance, and NOT the same thing as the outright
          LinAlgError case below (a near-singular matrix does not raise). This is what a
          bound-pinned fit typically causes, but is checked independently since it is the actual
          numerical symptom regardless of cause.

    Arguments:
        fit_res: [OptimizeResult] From minimizeAlphaBetaRobust(..., return_fit_result=True).
        alpha: [float] Fitted ballistic coefficient.
        beta: [float] Fitted mass loss parameter.
        v_data: [ndarray] Velocity data (m/s), as given to fitAlphaBeta().
        ht_data: [ndarray] Height data (m), as given to fitAlphaBeta().
        v_init: [float] Adopted initial velocity (m/s).
        v_init_given: [bool] Whether v_init was supplied by the caller (vs. derived internally).
        sigma_v: [float] Velocity uncertainty (m/s) used to weight the fit.
        sigma_v_init: [float or None] Caller-supplied v_init uncertainty - only used if
            v_init_given (see fitAlphaBeta()'s docstring).
        loss: [str] As in fitAlphaBeta().
        f_scale: [float] As in fitAlphaBeta().
        ci: [float] Confidence level (0-100, e.g. 95) for alpha_ci_gaussian_lower/upper and
            beta_ci_gaussian_lower/upper - same convention as
            wmpl.Utils.Math.confidenceInterval()'s `ci`.

    Return:
        errors: [dict] 'alpha', 'beta', 'alpha_std_rel', 'beta_std_rel', 'alpha_std', 'beta_std',
            'cov_log', 'corr_log', 'cov', 'corr', 'alpha_ci_gaussian_lower',
            'alpha_ci_gaussian_upper', 'beta_ci_gaussian_lower', 'beta_ci_gaussian_upper', 'ci',
            'sigma_v', 'sigma_v_init' - see fitAlphaBeta()'s docstring for details.
    """

    # Resolve v_init's own uncertainty up front, so it is available in `errors` even if the
    #   covariance itself turns out to be degenerate below (see the bound/near-singular checks).
    if v_init_given:

        if sigma_v_init is None:
            print("WARNING: v_init was given without sigma_v_init - its uncertainty will NOT be "
                "propagated into alpha_std_rel/beta_std_rel (v_init is being treated as exactly "
                "known).")
            sigma_v_init_used = 0.0

        else:
            sigma_v_init_used = sigma_v_init

    else:
        # Standard error of the median (NOT the plain MAD, which measures the dispersion of the
        #   observations themselves, not the precision of the median as an estimator of v_init)
        #   of the same leading points fitAlphaBeta() used to derive v_init.
        max_index = max(int(0.2*len(v_data)), 10)
        v_head = v_data[:max_index]
        mad = np.median(np.abs(v_head - np.median(v_head)))
        sigma_v_init_used = 1.2533*1.4826*mad/np.sqrt(max_index)

    nan_result = {
        'alpha': alpha, 'beta': beta, 'alpha_std_rel': np.nan, 'beta_std_rel': np.nan,
        'alpha_std': np.nan, 'beta_std': np.nan, 'cov_log': np.full((2, 2), np.nan),
        'corr_log': np.nan, 'cov': np.full((2, 2), np.nan), 'corr': np.nan,
        'alpha_ci_gaussian_lower': np.nan, 'alpha_ci_gaussian_upper': np.nan,
        'beta_ci_gaussian_lower': np.nan, 'beta_ci_gaussian_upper': np.nan,
        'ci': ci, 'sigma_v': sigma_v, 'sigma_v_init': sigma_v_init_used,
    }

    # Warn if the fitted (alpha, beta) landed near an optimization bound, using the same
    #   log-space tolerance fitAlphaBetaLightCurve() uses for its own bound check (alpha/beta
    #   span orders of magnitude, so a linear-space check would misfire - see
    #   _makeBoundChecker()). The Gauss-Newton curvature below is only meaningful around a
    #   genuine local optimum, and tends to be degenerate (or numerically nonsensical) once the
    #   fit has been clipped against a box constraint instead of converging to one.
    log_lower, log_upper = _logAlphaBetaBounds()
    bound_tol = 1e-2
    bound_warnings = []

    for name, value, lo, hi in [
            ("alpha", alpha, log_lower[0], log_upper[0]),
            ("beta", beta, log_lower[1], log_upper[1])]:

        log_value = np.log(value)
        span = hi - lo

        if (log_value - lo)/span < bound_tol:
            bound_warnings.append("{:s}={:.6g} is close to the fit lower bound ({:.6g})".format(
                name, value, np.exp(lo)))

        if (hi - log_value)/span < bound_tol:
            bound_warnings.append("{:s}={:.6g} is close to the fit upper bound ({:.6g})".format(
                name, value, np.exp(hi)))

    if bound_warnings:
        print()
        print("WARNING: fitted (alpha, beta) landed within {:.3g}% of an optimization bound - "
            "the fit may be poorly constrained, and the Gauss-Newton error estimate below may "
            "be degenerate (NaN) or unreliable as a result.".format(100*bound_tol))
        for warning in bound_warnings:
            print("  - {:s}".format(warning))
        print()

    # Gauss-Newton covariance of (ln alpha, ln beta), v_init held fixed at its adopted value
    dof = max(len(fit_res.fun) - len(fit_res.x), 1)
    s2 = 2.0*fit_res.cost/dof

    try:
        cov_fit = s2*np.linalg.inv(fit_res.jac.T @ fit_res.jac)
    except np.linalg.LinAlgError:
        return nan_result

    # A near- (but not exactly) singular Jacobian can produce a numerically "valid" (no
    #   LinAlgError) inverse with a negative variance on the diagonal - mathematically impossible
    #   for a real covariance, and NOT caught by the try/except above. Most often seen exactly
    #   when (alpha, beta) is pinned against a bound (see the warning above): fail cleanly with
    #   NaN here instead of letting np.sqrt() silently emit a generic RuntimeWarning below.
    if np.any(np.diag(cov_fit) < 0):
        print("WARNING: the Gauss-Newton covariance has a negative variance on its diagonal "
            "(a near-singular Jacobian, most often from a fit pinned against a bound - see "
            "above). Returning NaN for alpha_std_rel/beta_std_rel/cov/corr/the confidence "
            "interval instead of a meaningless value.")
        return nan_result

    # Rank-1 correction from v_init's uncertainty, via a numerical sensitivity. Skipped if there
    #   is no v_init uncertainty to propagate.
    if sigma_v_init_used > 0:

        dv = 1e-4*v_init

        def _logAlphaBetaAt(v_init_pert):
            _, alpha_p, beta_p = fitAlphaBeta(v_data, ht_data, v_init=v_init_pert, \
                method='robust', sigma_v=sigma_v, loss=loss, f_scale=f_scale)
            return np.log([alpha_p, beta_p])

        jac_v0 = (_logAlphaBetaAt(v_init + dv) - _logAlphaBetaAt(v_init - dv))/(2*dv)

        cov_v0 = np.outer(jac_v0, jac_v0)*sigma_v_init_used**2

    else:
        cov_v0 = np.zeros((2, 2))

    # cov_v0 is PSD (an outer product scaled by a non-negative variance), so it cannot turn the
    #   already-checked non-negative diagonal of cov_fit negative - cov_log's diagonal is
    #   guaranteed non-negative here too.
    cov_log = cov_fit + cov_v0

    alpha_std_rel, beta_std_rel = np.sqrt(np.diag(cov_log))
    corr_log = cov_log[0, 1]/(alpha_std_rel*beta_std_rel)

    # Linear (alpha, beta) space, via the delta method (see the docstring above)
    scale = np.array([alpha, beta])
    cov = np.outer(scale, scale)*cov_log
    alpha_std, beta_std = np.sqrt(np.diag(cov))
    corr = cov[0, 1]/(alpha_std*beta_std)

    # Parametric (log-normal) confidence interval - see the docstring above for how this differs
    #   from wmpl.Utils.Math.confidenceInterval()'s empirical percentiles elsewhere in the library
    z = scipy.special.ndtri(0.5 + ci/200.0)
    alpha_ci_gaussian_lower = alpha*np.exp(-z*alpha_std_rel)
    alpha_ci_gaussian_upper = alpha*np.exp(z*alpha_std_rel)
    beta_ci_gaussian_lower = beta*np.exp(-z*beta_std_rel)
    beta_ci_gaussian_upper = beta*np.exp(z*beta_std_rel)

    return {
        'alpha': alpha,
        'beta': beta,
        'alpha_std_rel': alpha_std_rel,
        'beta_std_rel': beta_std_rel,
        'alpha_std': alpha_std,
        'beta_std': beta_std,
        'cov_log': cov_log,
        'corr_log': corr_log,
        'cov': cov,
        'corr': corr,
        'alpha_ci_gaussian_lower': alpha_ci_gaussian_lower,
        'alpha_ci_gaussian_upper': alpha_ci_gaussian_upper,
        'beta_ci_gaussian_lower': beta_ci_gaussian_lower,
        'beta_ci_gaussian_upper': beta_ci_gaussian_upper,
        'ci': ci,
        'sigma_v': sigma_v,
        'sigma_v_init': sigma_v_init_used,
    }


def fitAlphaBeta(v_data, ht_data, v_init=None, method='q4', sigma_v=None, loss='soft_l1',
        f_scale=2.0, estimate_errors=False, sigma_v_init=None, ci=95.0, verbose=False):
    """ Fit the alpha and beta parameters to the given velocity and height data.

    Two methods are available (see minimizeAlphaBeta()/minimizeAlphaBetaRobust() for the full
    math):
        - method='q4' (default, unchanged from before): the classic Gritsevich (2007) Q4 fit.
          Its residual is NOT the velocity residual v_model - v_obs - it is a transformed,
          height-weighted quantity that implicitly under-weights the end of the trajectory
          (where the deceleration/beta signal actually is) relative to the near-constant-velocity
          beginning.
        - method='robust': least squares directly on the velocity residuals, robustified with
          `loss` (soft-L1 by default), seeded from the Q4 result. Weights the whole trajectory
          much more evenly than Q4 and tracks the deceleration/"knee" more faithfully - this is
          the same approach fitAlphaBetaLightCurve() uses for its dynamics block. Required for
          estimate_errors=True (see below) - Q4's L1/Nelder-Mead fit has no natural Jacobian to
          build an analytic error estimate from.

    Arguments:
        v_data: [ndarray] Velocity data (m/s).
        ht_data: [ndarray] Height data (m), already rescaled to the equivalent simple exponential
            atmosphere the alpha-beta model assumes (see rescaleHeightToExponentialAtmosphere()) -
            NOT raw geometric/NRLMSISE heights. The real atmosphere is not a simple exponential,
            so fitting unrescaled heights directly biases alpha/beta.

    Keyword arguments:
        v_init: [float] Initial velocity (m/s). If None, it will be determined from the first 20% of point
            (or a minimum of 10 points). Note that this assumes the input arrays are ordered from
            the beginning to the end of the trajectory (unlike fitAlphaBetaLightCurve(), which
            sorts its inputs by decreasing height internally).
        method: [str] 'q4' (default) or 'robust'.
        sigma_v: [float] Velocity uncertainty (m/s) used to weight the residuals for
            method='robust', and (if estimate_errors=True) to scale the error estimate too. If
            None, estimated from the MAD of the Q4 fit's residuals (floored at 1 m/s, see
            _estimateSigmaV()) - an *effective* uncertainty that absorbs both measurement noise
            and alpha-beta model misspecification, not a pure instrumental one.
        loss: [str] Only used for method='robust'. scipy.optimize.least_squares loss.
        f_scale: [float] Only used for method='robust'. Robust loss scale, in units of sigma_v.
        estimate_errors: [bool] If True, also return a 4th element (a dict, see Return) with an
            analytic error estimate. Only supported for method='robust' - raises ValueError
            otherwise (see the method argument above). Off by default, so existing
            3-tuple-unpacking call sites (v_init, alpha, beta = fitAlphaBeta(...)) are unaffected.
        sigma_v_init: [float] Uncertainty on v_init (m/s), only used when estimate_errors=True:
              - v_init given (not None) + sigma_v_init given: propagated into
                alpha_std_rel/beta_std_rel/cov_log.
              - v_init given + sigma_v_init=None (default): v_init is treated as exactly known - a
                warning is printed, since this is an easy way to silently understate alpha/beta's
                uncertainty if v_init actually has a non-negligible error of its own.
              - v_init=None (derived internally): this argument is ignored - v_init's uncertainty
                is instead estimated automatically as the standard error of the median of the same
                leading points used to derive it (see the Return section), which is the quantity
                that actually needs to be propagated, not the raw scatter of those points.
        ci: [float] Confidence level (0-100, e.g. 95, the default) for the approximate
            alpha_ci_gaussian_lower/upper and beta_ci_gaussian_lower/upper bounds below - only
            used if estimate_errors=True. Same convention as
            wmpl.Utils.Math.confidenceInterval()'s `ci`.
        verbose: [bool] If True, print the adopted v_init/alpha/beta (and, if
            estimate_errors=True, the error estimate too).

    Return:
        (v_init, alpha, beta):
            - v_init: [float] Input or derived initial velocity (m/s).
            - alpha: [float] Ballistic coefficient.
            - beta: [float] Mass loss.
        If estimate_errors=True, a 4th element `errors` (dict, from _alphaBetaRobustErrors()) is
        also returned:
            'alpha', 'beta': echoed back, so the dict is self-contained.
            'alpha_std_rel', 'beta_std_rel': [float] Relative (fractional) 1-sigma uncertainties
                of alpha/beta in (ln alpha, ln beta) space, combining the Gauss-Newton curvature
                of the fit itself with (if applicable) v_init's own uncertainty - see the caveats
                below.
            'alpha_std', 'beta_std': [float] The same 1-sigma uncertainties propagated to linear
                (alpha, beta) space via the delta method (alpha_std = alpha*alpha_std_rel, since
                d(alpha)/d(ln alpha) = alpha).
            'cov_log': [ndarray] Full 2x2 covariance matrix of (ln alpha, ln beta) - the space
                this model is actually fit in.
            'cov': [ndarray] The same covariance propagated to linear (alpha, beta) space. Use
                'cov'/'cov_log' instead of the marginal std devs above to propagate into any OTHER
                derived quantity: alpha and beta are typically strongly correlated (see 'corr'/
                'corr_log'), so propagating the marginal std devs independently would be wrong.
            'corr_log', 'corr': [float] Correlation coefficient of (ln alpha, ln beta) / of
                (alpha, beta) - numerically identical (correlation is invariant under the
                positive-diagonal rescaling relating the two spaces); both are kept for
                convenience.
            'alpha_ci_gaussian_lower', 'alpha_ci_gaussian_upper',
            'beta_ci_gaussian_lower', 'beta_ci_gaussian_upper': [float] APPROXIMATE confidence
                intervals based on the local Gaussian approximation in log-space, at the `ci`
                level - i.e. alpha/beta are treated as log-normal, so these bounds are asymmetric
                around alpha/beta and, unlike alpha +/- z*alpha_std, can never go negative. The
                "_gaussian" in the name is deliberate: this is NOT the same kind of object as
                wmpl.Utils.Math.confidenceInterval()'s output elsewhere in this library (e.g.
                Trajectory.py, REBOUND.py) - those are empirical percentiles of an actual Monte
                Carlo ensemble, whereas there is no ensemble here, only a parametric
                approximation.
            'ci': [float] The confidence level actually used (echoed back).
            'sigma_v': [float] The velocity uncertainty actually used (see the sigma_v caveat
                above).
            'sigma_v_init': [float] v_init's uncertainty actually used for the propagation above
                (0.0 if v_init is being treated as exactly known).

    Statistical caveats (estimate_errors=True):
        - Both contributions to alpha_std_rel/beta_std_rel/cov_log assume independent,
          diagonal-covariance velocity residuals - the same simplification fitAlphaBetaLightCurve()
          makes (consecutive points from the same station are not really statistically
          independent).
        - The Gauss-Newton contribution is a *local curvature* approximation: soft-L1 reweights
          residuals nonlinearly, so (J^T J)^-1 is not the inverse Fisher information of Gaussian
          noise - same caveat as fitAlphaBetaLightCurve()'s alpha_std_rel/beta_std_rel.
        - The v_init-sensitivity correction and the Gauss-Newton term are simply added
          (cov_log = cov_fit + cov_v_init), which ignores that v_init and the conditional fit's
          residuals both ultimately come from the same v_data - a standard plug-in approximation,
          not a fully joint treatment.
        - alpha_ci_gaussian_lower/upper and beta_ci_gaussian_lower/upper assume normality in LOG
          space, which is only ever as good as the two contributions to cov_log above - they are
          a convenience view of the same Gaussian approximation, not an independently more
          rigorous one.
        - If the fitted (alpha, beta) landed within 1% (log-space) of an ALPHA_BETA_BOUNDS edge,
          or if that produces a near-singular Jacobian (a negative variance on the covariance
          diagonal - impossible for a real covariance), a WARNING is printed and every error field
          is NaN (alpha/beta themselves are still returned). Both mean the same thing: the fit is
          likely poorly constrained by this data, and the local curvature approximation isn't
          meaningful around a boundary the optimizer was clipped against rather than converged to.
    """

    method = method.lower()

    if estimate_errors and method != 'robust':
        raise ValueError("estimate_errors=True is only supported for method='robust' - Q4's "
            "L1/Nelder-Mead fit has no natural Jacobian to build an analytic covariance from.")

    v_init_given = v_init is not None

    # Compute the initial velocity, if it wasn't given already
    if not v_init_given:

        max_index = int(0.2*len(v_data))
        if max_index < 10:
            max_index = 10

        v_init = np.median(v_data[:max_index])

    # Normalize the velocity and height
    v_normed = v_data/v_init
    ht_normed = ht_data/HT_NORM_CONST

    if method == 'q4':
        alpha, beta = minimizeAlphaBeta(v_normed, ht_normed)

    elif method == 'robust':
        alpha0, beta0 = minimizeAlphaBeta(v_normed, ht_normed)

        if sigma_v is None:
            sigma_v = _estimateSigmaV(alpha0, beta0, v_normed, ht_normed, v_init)

        if estimate_errors:
            alpha, beta, fit_res = minimizeAlphaBetaRobust(v_normed, ht_normed, sigma_v/v_init, \
                loss=loss, f_scale=f_scale, x0_alpha_beta=(alpha0, beta0), return_fit_result=True)
        else:
            alpha, beta = minimizeAlphaBetaRobust(v_normed, ht_normed, sigma_v/v_init, loss=loss, \
                f_scale=f_scale, x0_alpha_beta=(alpha0, beta0))

    else:
        raise ValueError("method must be 'q4' or 'robust', got '{:s}'.".format(method))

    if estimate_errors:
        errors = _alphaBetaRobustErrors(fit_res, alpha, beta, v_data, ht_data, v_init, \
            v_init_given, sigma_v, sigma_v_init, loss, f_scale, ci)

    if verbose:
        print()
        print("--- fitAlphaBeta ({:s}) ---".format(method))
        print("v_init               = {:.3f} m/s".format(v_init))

        if estimate_errors:
            print("alpha                = {:.6f} (+/-{:.1%})  [{:.3g}% gaussian CI: {:.6f} - "
                "{:.6f}]".format(alpha, errors['alpha_std_rel'], ci,
                errors['alpha_ci_gaussian_lower'], errors['alpha_ci_gaussian_upper']))
            print("beta                 = {:.6f} (+/-{:.1%})  [{:.3g}% gaussian CI: {:.6f} - "
                "{:.6f}]".format(beta, errors['beta_std_rel'], ci,
                errors['beta_ci_gaussian_lower'], errors['beta_ci_gaussian_upper']))
            print("corr(alpha, beta)    = {:.3f}".format(errors['corr']))
            print("sigma_v              = {:.3f} m/s".format(errors['sigma_v']))
            print("sigma_v_init         = {:.3f} m/s".format(errors['sigma_v_init']))
        else:
            print("alpha                = {:.6f}".format(alpha))
            print("beta                 = {:.6f}".format(beta))

    if estimate_errors:
        return v_init, alpha, beta, errors

    return v_init, alpha, beta


def alphaBetaHeightNormed(v_normed, alpha, beta):
    """ Normalized height y(v) of the alpha-beta model, Eq. (7) of Gritsevich (2009):

            y = ln(2 alpha) + beta - ln(Delta),   Delta = Ei(beta) - Ei(beta v^2)

        This is the dimensionless core of alphaBetaHeight()/alphaBetaVelocity() below, and is
        also used directly (in v, y space) by fitAlphaBetaLightCurve().

    Arguments:
        v_normed: [float or ndarray] Velocity normalized to v_init, in (0, 1).
        alpha: [float] Ballistic coefficient.
        beta: [float] Mass loss parameter.

    Return:
        y: [float or ndarray] Height normalized to HT_NORM_CONST.
    """

    delta = scipy.special.expi(beta) - scipy.special.expi(beta*np.asarray(v_normed)**2)

    return np.log(2*alpha) + beta - np.log(delta)


def alphaBetaHeight(vel_data, alpha, beta, v_init):
    """ Compute the height given the velocity and alpha, beta parameters.

    Arguments:
        vel_data: [ndarray] Velocity data (m/s).
        alpha: [float] Ballistic coefficient.
        beta: [float] Mass loss.
        v_init: [float] Input or derived initial velocity (m/s).

    Return:
        ht_data: [ndarray] Height data (m).
    """

    # Normalize the velocity
    vel_normed = vel_data/v_init

    # Compute the normalized height and convert it to meters
    return alphaBetaHeightNormed(vel_normed, alpha, beta)*HT_NORM_CONST


def alphaBetaVelocityNormed(ht_normed, alpha, beta, v_eps=1e-10):
    """ Invert Eq. (7): normalized velocity v(y) for given (alpha, beta). Unfortunately there is no
        analytical inverse to the exponential integral, so the solution is found numerically.

        The root is unique because dy/dv = 2 exp(beta v^2)/(v Delta) > 0 on (0, 1), so it can be
        bracketed and found with Brent's method instead of a general-purpose minimizer - this is
        both faster and more precise than minimizing (y_guess - y_target)^2 over v. Heights above
        y(1 - v_eps) are clipped to v = 1 - v_eps, heights below y(v_eps) to v = v_eps.

    Arguments:
        ht_normed: [float or ndarray] Height normalized to HT_NORM_CONST.
        alpha: [float] Ballistic coefficient.
        beta: [float] Mass loss parameter.

    Keyword arguments:
        v_eps: [float] Bracket margin around (0, 1).

    Return:
        v_normed: [ndarray] Normalized model velocity at each height.
    """

    y_arr = np.atleast_1d(np.asarray(ht_normed, dtype=np.float64))
    v_out = np.empty_like(y_arr)

    v_lo, v_hi = v_eps, 1.0 - v_eps

    def _g(v, y_target):
        return alphaBetaHeightNormed(v, alpha, beta) - y_target

    for i, y_t in enumerate(y_arr):

        # Clip outside the invertible range
        if _g(v_hi, y_t) <= 0:
            v_out[i] = v_hi

        elif _g(v_lo, y_t) >= 0:
            v_out[i] = v_lo

        else:
            v_out[i] = scipy.optimize.brentq(_g, v_lo, v_hi, args=(y_t,), xtol=1e-12)

    return v_out


def alphaBetaVelocity(ht_data, alpha, beta, v_init, v_eps=1e-10):
    """ Compute the velocity given the height and alpha, beta parameters. Unfortunately there is no
        analytical inverse to the exponential integral, so the solution is found numerically.

    Arguments:
        ht_data: [ndarray or float] Height data (m).
        alpha: [float] Ballistic coefficient.
        beta: [float] Mass loss.
        v_init: [float] Input or derived initial velocity (m/s).

    Keyword arguments:
        v_eps: [float] Bracket margin around (0, 1) normalized velocity, passed to
            alphaBetaVelocityNormed().

    Return:
        vel_data: [ndarray or float] Velocity data (m/s).
    """

    # Allow both scalar and array height inputs
    scalar_input = np.isscalar(ht_data)
    if scalar_input:
        ht_data = np.array([ht_data])

    # Normalize the height and numerically invert Eq. (7) for the normalized velocity
    ht_normed = ht_data/HT_NORM_CONST
    vel_normed = alphaBetaVelocityNormed(ht_normed, alpha, beta, v_eps=v_eps)

    # Compute the velocity in m/s
    vel_data = vel_normed*v_init

    # Return a scalar if the input height was a scalar
    if scalar_input:
        return vel_data[0]

    return vel_data


def alphaBetaMasses(
        alpha,
        beta,
        slope,
        mu=0,
        dens=3500,
        shape_coeff=0.55,
        gamma=1.0,
        vel_init=None,
        vel_end=None,
        verbose=False):
    """ Compute the initial and final mass from alpha-beta parameters and assumed physical properties.
    
    Arguments:
        alpha: [float]
        beta: [float]
        slope: [float] Fireball entry angle (radians).

    Keyword arguments:
        mu: [float] Shape change coefficient. 0 for no spin, and 2/3 for sufficient spin to equally ablate
            the whole surface. Default value is 0.
        dens: [float] Bulk density in kg/m^3.
        shape_coeff: [float] Shape coefficient. 1.21 for sphere, 1.55 for brick. As shape_coeff and Gamma are 
            factored together, we use the empirical value of gamma*A = 0.55 by default.
        gamma: [float] Drag parameter Γ (= C_D /2).
        vel_init: [float] Initial velocity in m/s. If both vel_init and vel_end are given, the final
            mass is computed using the full alpha-beta solution. If either is None, the simple
            approximation assuming vel_end << vel_init is used.
        vel_end: [float] Final velocity in m/s. Used together with vel_init to compute the final mass
            using the full alpha-beta solution.
        verbose: [bool] If True, print the parameters used in the computation and the resulting
            initial and final mass estimates.
    Return:
        (m_init, m_final): [tuple of floats] Initial and final mass in kg.
    """

    if alpha <= 0:
        raise ValueError("alpha must be positive")

    if beta <= 0:
        raise ValueError("beta must be positive")

    if not (0 < slope <= np.pi/2):
        raise ValueError("slope must be between 0 and pi/2 radians")

    if dens <= 0:
        raise ValueError("dens must be positive")

    if gamma <= 0:
        raise ValueError("gamma must be positive")

    if shape_coeff <= 0:
        raise ValueError("shape_coeff must be positive")

    if vel_init is not None:
        if vel_init <= 0:
            raise ValueError("vel_init must be positive")

    if vel_end is not None:
        if vel_end < 0:
            raise ValueError("vel_end must be non-negative")

    if (vel_init is not None) and (vel_end is not None):
        if vel_end > vel_init:
            raise ValueError("vel_end cannot exceed vel_init")

    if not (0 <= mu <= 2/3):
        raise ValueError("mu must be between 0 and 2/3")

    rho_atm_0 = 1.225

    # Compute the M0_star parameter
    m0s = (gamma*shape_coeff*rho_atm_0*HT_NORM_CONST/(dens**(2/3.0)))**3

    # Compute the initial mass
    m_init = m0s/((alpha*np.sin(slope))**3)


    # Compute the final mass
    if (vel_init is None) or (vel_end is None):
        # Simple alpha-beta approximation
        m_final = m_init*np.exp(-beta/(1 - mu))

    else:
        # General alpha-beta solution
        m_final = m_init*np.exp(
            -beta/(1 - mu)
            *(1 - (vel_end/vel_init)**2)
        )
        
    if verbose:

        print("Alpha-beta mass estimate")
        print("------------------------")

        print(f"alpha        = {alpha:.3f}")
        print(f"beta         = {beta:.3f}")
        print(f"slope        = {np.degrees(slope):.2f} deg")
        print(f"mu           = {mu:.3f}")
        print(f"density      = {dens:.1f} kg/m^3")
        print(f"gamma*A      = {gamma*shape_coeff:.3f}")

        if (vel_init is not None) and (vel_end is not None):
            print(f"v_init       = {vel_init/1000:.3f} km/s")
            print(f"v_end        = {vel_end/1000:.3f} km/s")
            print("solution     = full alpha-beta")
        else:
            print("solution     = asymptotic approximation")

        print()

        print(f"m_init       = {m_init:.3e} kg")
        print(f"m_final      = {m_final:.3e} kg")

    return m_init, m_final


def fitAlphaBetaMass(
        v_data,
        ht_data,
        slope,
        mass,
        mass_constraint="initial",
        shape_coeff=0.55,
        gamma=1.0,
        density=None,
        v_init=None,
        v_final=None,
        verbose=True,
        plot=False):
    """ Fit alpha-beta model parameters under initial, final, or both mass constraints,
    following Peña-Asensio & Gritsevich (2025, 2026). Unlike the simultaneous
    optimization proposed in those papers, this implementation solves a reduced
    optimization problem in which only the parameters required by the selected mass
    constraint are fitted, while the remaining quantities are derived analytically.
    This reduces the dimensionality of the optimization but does not guarantee full
    self-consistency of the recovered solution. In general, use
    mass_constraint="initial" with a photometric initial mass or
    mass_constraint="final" with a dynamic final mass. The bulk density may
    optionally be fixed, and either or both of the initial and final velocities
    may be provided.

    Arguments:
        v_data: [ndarray] Velocity data (m/s).
        ht_data: [ndarray] Height data (m).
        slope: [float] Entry angle (radians).
        mass: [float or tuple] Initial and/or final mass (kg), depending on mass_constraint.
                If mass_constraint="initial", mass must be a float containing the initial mass.
                If mass_constraint="final", mass must be a float containing the final mass.
                If mass_constraint="both", mass must be a tuple: (mass_initial, mass_final)

    Keyword arguments:
        mass_constraint: [str] Mass constraint to use: "initial" ("i"), "final" ("f"), or "both" ("b").
        shape_coeff: [float] Shape coefficient. 1.21 for sphere, 1.55 for brick. As shape_coeff and Gamma
            are factored together, we use the empirical value of gamma*A = 0.55 by default.
        gamma: [float] Drag parameter Γ (= C_D /2).
        density: [float or None] If given, the bulk density (kg/m^3) is held fixed at this
            value instead of being fitted, removing it as a free parameter (e.g. a recovered
            meteorite of known/assumed density, or a synthetic trajectory generated at a known
            density). The optimization problem is reduced accordingly: only beta is fitted
            under mass_constraint="initial" or "final" (plus mu for their best-fit μ search),
            and under mass_constraint="both" with μ=0 or μ=2/3 there are zero free parameters
            left (beta is already analytic), so alpha and beta are computed directly without
            calling the optimizer. Default is None (density is fitted, as before).
        v_init: [float or None] If given, used as-is. If None, derived as the median velocity
            of the top-20% highest-altitude points.
        v_final: [float or None] If given, used as-is, EXCEPT under mass_constraint="initial"
            (see Notes below), where it is only used to validate v_final < v_init and is then
            discarded: the actual v_final is always re-derived from the fitted alpha/beta.
            If None, derived as the asymptotic minimum of a preliminary alpha-beta velocity
            model (or, under mass_constraint="initial", of the final fitted model).
        verbose: [bool] If True, print the parameters used in the computation and the resulting
            fitted parameters, clearly separating which of density/v_init/v_final were given
            as input from those that were derived/fitted internally.
        plot: [bool] If True, show a height vs. velocity plot with the observed data and the
            fitted alpha-beta curve(s): one for μ=0, one for μ=2/3, and (only when
            mass_constraint is "final" or "both", where mu_best is computed) a third one
            for the best-fit μ. Input parameters (as given
            by the caller) are shown in the title, fitted parameters (per μ branch) in the
            legend. Blocks execution until the plot window is closed.

    Notes:
        If mass_constraint="initial", any v_final passed in is NOT used as a fit input: it is
        only checked against v_init and then overwritten with the value derived from the fitted
        alpha/beta, since v_final is not a free input under this constraint. Pass verbose=True
        to confirm whether v_init/v_final/density were used as given or derived.
        If mass_constraint="initial", a single alpha-beta fit is performed because the
        alpha-beta velocity solution is independent of μ. In this case, the fitted
        density, alpha, and beta are identical for μ = 0 and μ = 2/3, and only the
        derived final masses differ. Since μ has no effect whatsoever on the fit
        residual in this case, no data-driven best-fit μ is computed
        (mu_best and its associated outputs are returned as None).
        If mass_constraint="final", two independent fits are performed, one for μ = 0
        and one for μ = 2/3, because reconstructing the initial mass from the final
        mass explicitly depends on μ. Since μ does affect the fit residual here, an
        additional joint fit over (density, beta, μ) is performed, with μ free within
        [0, 2/3], to find the data-driven μ that minimizes the fit residual (mu_best).
        If mass_constraint="both", the initial and final masses are both
        imposed. In this case beta is computed analytically from the mass
        ratio and the observed velocity change, and only the density is fitted.
        Since beta (and hence the residual) still depends on μ through that analytic
        relation, an additional joint fit over (density, μ) is performed, with μ free
        within [0, 2/3], to find the μ that minimizes the fit residual (mu_best).
        The mu_best solution is the value of μ that minimizes the fit residual under
        the selected mass constraint and is not guaranteed to be physically meaningful.
        It tends to pull the constrained fit toward the unconstrained alpha-beta
        solution, and may land on 0 or 2/3 anyway. Treat it as a third, informative
        data point alongside the μ = 0 and μ = 2/3 physical bounds.
        Because the optimization is performed in a reduced, partially decoupled parameter
        space rather than as a simultaneous fit of all quantities, the fitted α-β solution
        is not guaranteed to reproduce exactly the v_final used to impose the mass constraint.
        Consequently, under mass_constraint="final" or "both", the resulting solution may not
        be fully self-consistent. The α-β formulation is a single-body model and is therefore
        intended for events without significant fragmentation, or where a single dominant body
        governs the observed dynamics. Reliable parameter estimation also requires measurable
        deceleration along the observed trajectory.

    Return:
        (
            v_init,
            v_final,
            density_mu0,
            alpha_mu0,
            beta_mu0,
            m_initial_mu0,
            m_final_mu0,
            density_mu23,
            alpha_mu23,
            beta_mu23,
            m_initial_mu23,
            m_final_mu23,
            mu_best,
            density_mu_best,
            alpha_mu_best,
            beta_mu_best,
            m_initial_mu_best,
            m_final_mu_best
        ):
            - v_init: [float] Input or derived initial velocity (m/s).
            - v_final: [float] Input or derived final velocity (m/s).
            - density_mu0: [float] Best-fit bulk density (kg/m^3) assuming μ = 0.
            - alpha_mu0: [float] Derived ballistic coefficient assuming μ = 0.
            - beta_mu0: [float] Derived mass loss parameter assuming μ = 0.
            - m_initial_mu0: [float] Initial mass (input or reconstructed from the fitted
                parameters) assuming μ = 0.
            - m_final_mu0: [float] Final mass (input or reconstructed from the fitted parameters)
                assuming μ = 0.
            - density_mu23: [float] Best-fit bulk density (kg/m^3) assuming μ = 2/3.
            - alpha_mu23: [float] Derived ballistic coefficient assuming μ = 2/3.
            - beta_mu23: [float] Derived mass loss parameter assuming μ = 2/3.
            - m_initial_mu23: [float] Initial mass (input or reconstructed from the fitted
                parameters) assuming μ = 2/3.
            - m_final_mu23: [float] Final mass (input or reconstructed from the fitted parameters)
                assuming μ = 2/3.
            - mu_best: [float or None] Data-driven μ (within [0, 2/3]) that minimizes the fit
                residual. None if mass_constraint="initial".
            - density_mu_best: [float or None] Best-fit bulk density (kg/m^3) at μ = mu_best.
            - alpha_mu_best: [float or None] Derived ballistic coefficient at μ = mu_best.
            - beta_mu_best: [float or None] Derived mass loss parameter at μ = mu_best.
            - m_initial_mu_best: [float or None] Initial mass at μ = mu_best.
            - m_final_mu_best: [float or None] Final mass at μ = mu_best.
    """

    mass_constraint = mass_constraint.lower()

    if mass_constraint in ["initial", "i"]:
        mass_constraint = "initial"

        if not np.isscalar(mass):
            raise ValueError("For mass_constraint='initial', mass must be a single number (kg).")

        mass_initial = mass
        mass_final = None

        if mass_initial <= 0:
            raise ValueError("mass_initial must be positive.")

    elif mass_constraint in ["final", "f"]:
        mass_constraint = "final"

        if not np.isscalar(mass):
            raise ValueError("For mass_constraint='final', mass must be a single number (kg).")

        mass_initial = None
        mass_final = mass

        if mass_final <= 0:
            raise ValueError("mass_final must be positive.")

    elif mass_constraint in ["both", "b"]:
        mass_constraint = "both"

        if not isinstance(mass, (tuple, list)) or len(mass) != 2:
            raise ValueError("For mass_constraint='both', mass must be (mass_initial, mass_final)")

        mass_initial, mass_final = mass

        if mass_initial <= 0:
            raise ValueError("mass_initial must be positive.")

        if mass_final <= 0:
            raise ValueError("mass_final must be positive.")

        if mass_final >= mass_initial:
            raise ValueError("mass_final must be smaller than mass_initial.")

    else:
        raise ValueError("mass_constraint must be 'initial' ('i'), 'final' ('f'), or 'both' ('b')")

    if len(v_data) != len(ht_data):
        raise ValueError("v_data and ht_data must have the same length.")
    if not (0 < slope <= np.pi/2):
        raise ValueError("slope must be between 0 and pi/2 radians.")
    if shape_coeff <= 0:
        raise ValueError("shape_coeff must be > 0.")
    if gamma <= 0:
        raise ValueError("gamma must be > 0.")
    if (density is not None) and (density <= 0):
        raise ValueError("density must be > 0.")

    fit_density = density is None

    # Remember what the caller actually provided, before v_init/v_final are
    # possibly overwritten below with derived values, so verbose=True can report
    # which quantities are inputs and which are derived/fitted.
    v_init_given = v_init is not None
    v_final_given = v_final is not None
    v_final_input = v_final

    rho_atm_0 = 1.225

    # Compute the initial velocity as the median velocity of the top-20% highest-altitude points
    #   (or a minimum of 10 points), if it wasn't given already
    if v_init is None:

        # Sort by decreasing height, regardless of input ordering
        order_desc = np.argsort(-ht_data)
        v_data_desc = v_data[order_desc]

        max_index = int(0.2*len(v_data_desc))
        if max_index < 10:
            max_index = 10

        v_init = np.median(v_data_desc[:max_index])

    if (v_final is not None) and (v_final >= v_init):
        raise ValueError("v_final must be smaller than v_init.")

    # Run a preliminary unconstrained alpha-beta fit when the final velocity has to be derived from
    #   the data, and always under the final-mass constraint, where the fitted beta is needed to
    #   seed the constrained optimization (see below). It is skipped under mass_constraint="initial"
    #   when v_final is not used by the fit at all - there v_final is always re-derived from the
    #   fitted alpha/beta after the constrained fit
    beta_prelim = None
    if (mass_constraint == "final") or ((v_final is None) and (mass_constraint == "both")):

        # Fit the unconstrained alpha-beta model
        _, alpha_prelim, beta_prelim = fitAlphaBeta(v_data, ht_data, v_init=v_init)

        # Evaluate the model velocity at the observed heights and take the minimum as the final
        #   velocity, if it wasn't given already
        if v_final is None:
            vel_model = alphaBetaVelocity(ht_data, alpha_prelim, beta_prelim, v_init)
            v_final = np.min(vel_model)


    # Normalize velocity
    v_normed = v_data/v_init

    # Normalize height
    ht_normed = ht_data/HT_NORM_CONST

    def _alphaFromDensityAndMass(mass_init, dens):
        """ Analytically compute alpha from the bulk density and the initial mass, by inverting the
            initial mass solution used in alphaBetaMasses().
        """

        return gamma*shape_coeff*rho_atm_0*HT_NORM_CONST \
            /(dens**(2/3)*np.sin(slope)*mass_init**(1/3))

    def _massesAt(alpha, beta, mu, dens):
        """ alphaBetaMasses() bound to this fit's slope/shape_coeff/gamma/v_init/v_final, so each
            mu branch below just supplies the (alpha, beta, mu, dens) it solved for. v_final is read
            at call time, so this stays correct even where v_final is only finalized right before
            the first call (mass_constraint="initial").
        """

        return alphaBetaMasses(alpha, beta, slope, mu=mu, dens=dens, shape_coeff=shape_coeff, \
            gamma=gamma, vel_init=v_init, vel_end=v_final)

    def _getDensity(x):
        """ Split the optimizer's parameter vector into (density, remaining free params).
            If density is set, density isn't part of x at all and x is returned
            unchanged as the remaining params; otherwise density is x[0] and the rest
            follow. Every objective below starts by calling this, so fixing the density
            just shrinks x by one slot without changing any downstream physics.
        """

        if fit_density:
            return x[0], x[1:]

        return density, x

    def _densityBetaMinimizationInitialMass(x, v_normed, ht_normed):
        """ Fit objective for mass_constraint="initial": alpha follows analytically from the fixed
            initial mass and the density, so only (density, beta) are free.
        """

        dens, rest = _getDensity(x)
        beta = rest[0]

        alpha = _alphaFromDensityAndMass(mass_initial, dens)

        return _alphaBetaResidual(alpha, beta, v_normed, ht_normed)

    def _reconstructInitialMassAndAlpha(dens, beta, mu):
        """ Reconstruct the initial mass from the fixed final mass by inverting the general
            alpha-beta mass loss solution, then compute alpha from the density and that mass.
        """

        m_init_fit = mass_final*np.exp(beta/(1.0 - mu)*(1.0 - (v_final/v_init)**2))

        alpha = _alphaFromDensityAndMass(m_init_fit, dens)

        return m_init_fit, alpha

    def _densityBetaMinimizationFinalMass(x, v_normed, ht_normed, mu):
        """ Fit objective for mass_constraint="final": the initial mass (and with it alpha) is
            reconstructed from the fixed final mass, so only (density, beta) are free.
        """

        dens, rest = _getDensity(x)
        beta = rest[0]

        _, alpha = _reconstructInitialMassAndAlpha(dens, beta, mu)

        return _alphaBetaResidual(alpha, beta, v_normed, ht_normed)

    def _densityBetaMuMinimizationFinalMass(x, v_normed, ht_normed):
        """ Same as _densityBetaMinimizationFinalMass, but with mu as a free parameter instead of
            fixed, so the residual-minimizing mu can be found jointly with density and beta.
        """

        dens, rest = _getDensity(x)
        beta, mu = rest

        _, alpha = _reconstructInitialMassAndAlpha(dens, beta, mu)

        return _alphaBetaResidual(alpha, beta, v_normed, ht_normed)

    def _solveBothMasses(dens, mu):
        """ Analytic beta from the mass ratio and mu, and alpha from density and the fixed initial
            mass, for the mass_constraint="both" case. Shared by the optimization objectives and
            the post-fit derivation of alpha/beta at mu=0, mu=2/3 and mu=mu_best so both stay in
            sync if the relations change.
        """

        beta = _betaFromMasses(mu, mass_initial, mass_final, v_final, v_init)

        alpha = _alphaFromDensityAndMass(mass_initial, dens)

        return alpha, beta

    def _densityMinimizationBothMasses(x, v_normed, ht_normed, mu):
        """ Fit objective for mass_constraint="both": alpha and beta both follow analytically from
            the two fixed masses, so only the density is free. Only called when the density is
            fitted.
        """

        dens = x[0]

        alpha, beta = _solveBothMasses(dens, mu)

        return _alphaBetaResidual(alpha, beta, v_normed, ht_normed)

    def _densityMuMinimizationBothMasses(x, v_normed, ht_normed):
        """ Same as _densityMinimizationBothMasses, but with mu as a free parameter instead of
            fixed, so the residual-minimizing mu can be found jointly with density (beta is still
            derived analytically from mu and the two masses).
        """

        dens, rest = _getDensity(x)
        mu = rest[0]

        alpha, beta = _solveBothMasses(dens, mu)

        return _alphaBetaResidual(alpha, beta, v_normed, ht_normed)


    # Initial guess
    dens0 = 2500
    beta0 = 1

    # Bounds
    xmin = [100, 0.001]
    xmax = [9000, 50]

    # Seed beta with the preliminary unconstrained fit when available. This is essential under the
    #   final-mass constraint: its reduced objective has a spurious minimum at large beta, where the
    #   reconstructed initial mass diverges, alpha goes to 0, and the residual vanishes for any
    #   data, so the optimizer needs to start near the physical solution
    if beta_prelim is not None:
        beta0 = float(np.clip(beta_prelim, xmin[1], xmax[1]))

    # Free parameters are (density, beta) if the density is fitted, or just (beta,) if it is set.
    #   Used for mass_constraint="initial" and the mu=0/mu=2/3 fits of mass_constraint="final",
    #   which all share this same 2-parameter (or 1-parameter) shape
    if fit_density:
        x0 = [dens0, beta0]
        bnds = ((xmin[0], xmax[0]), (xmin[1], xmax[1]))
    else:
        x0 = [beta0]
        bnds = ((xmin[1], xmax[1]),)

    if mass_constraint == "initial":

        # Fit (density, beta), with alpha following analytically from the fixed initial mass
        res = scipy.optimize.minimize(_densityBetaMinimizationInitialMass, x0, \
            args=(v_normed, ht_normed), bounds=bnds, method='Nelder-Mead')

        if not res.success:
            print(f"WARNING: Optimizer failed: {res.message}")

        # Don't overwrite the input density variable - _getDensity() and the plotting code below
        #   read it to distinguish the fixed input value from the fitted one
        density_fit, rest = _getDensity(res.x)
        beta = rest[0]

        # The fit residual is independent of mu under the initial-mass constraint, so the fitted
        #   density, alpha, and beta are shared by both mu branches
        density_mu0, density_mu23 = density_fit, density_fit
        beta_mu0, beta_mu23 = beta, beta

        alpha = _alphaFromDensityAndMass(mass_initial, density_fit)

        alpha_mu0, alpha_mu23 = alpha, alpha

        # Derive the final velocity from the fitted model (v_final is not a fit input under this
        #   constraint)
        vel_model = alphaBetaVelocity(ht_data, alpha, beta, v_init)
        v_final = np.min(vel_model)

        m_initial_mu0, m_final_mu0 = _massesAt(alpha, beta, 0, density_fit)

        m_initial_mu23, m_final_mu23 = _massesAt(alpha, beta, 2/3, density_fit)

        # mu has zero effect on the fit residual under an initial-mass constraint (see docstring
        #   notes), so no data-driven best-fit mu exists here
        mu_best = None
        density_mu_best = None
        alpha_mu_best = None
        beta_mu_best = None
        m_initial_mu_best = None
        m_final_mu_best = None


    elif mass_constraint == "final":

        # Fit (density, beta) for mu = 0, with the initial mass reconstructed from the fixed
        #   final mass
        res_mu0 = scipy.optimize.minimize(_densityBetaMinimizationFinalMass, x0, \
            args=(v_normed, ht_normed, 0), bounds=bnds, method='Nelder-Mead')

        if not res_mu0.success:
            print(f"WARNING: Optimizer failed for mu=0: {res_mu0.message}")

        density_mu0, rest = _getDensity(res_mu0.x)
        beta_mu0 = rest[0]

        # Only alpha is needed here - the masses are computed below with _massesAt(), which
        #   reproduces the same initial mass by construction
        _, alpha_mu0 = _reconstructInitialMassAndAlpha(density_mu0, beta_mu0, 0)

        m_initial_mu0, m_final_mu0 = _massesAt(alpha_mu0, beta_mu0, 0, density_mu0)

        # Fit (density, beta) for mu = 2/3
        res_mu23 = scipy.optimize.minimize(_densityBetaMinimizationFinalMass, x0, \
            args=(v_normed, ht_normed, 2/3), bounds=bnds, method='Nelder-Mead')

        if not res_mu23.success:
            print(f"WARNING: Optimizer failed for mu=2/3: {res_mu23.message}")

        density_mu23, rest = _getDensity(res_mu23.x)
        beta_mu23 = rest[0]

        _, alpha_mu23 = _reconstructInitialMassAndAlpha(density_mu23, beta_mu23, 2/3)

        m_initial_mu23, m_final_mu23 = _massesAt(alpha_mu23, beta_mu23, 2/3, density_mu23)

        # Data-driven best-fit mu: jointly fit (density, beta, mu) with mu free within [0, 2/3],
        #   since mu affects the residual here (see docstring notes)
        if fit_density:
            x0_mu_best = [dens0, beta0, 1/3]
            bnds_mu_best = ((xmin[0], xmax[0]), (xmin[1], xmax[1]), (0.0, 2/3))
        else:
            x0_mu_best = [beta0, 1/3]
            bnds_mu_best = ((xmin[1], xmax[1]), (0.0, 2/3))

        res_mu_best = scipy.optimize.minimize(_densityBetaMuMinimizationFinalMass, x0_mu_best, \
            args=(v_normed, ht_normed), bounds=bnds_mu_best, method='Nelder-Mead')

        if not res_mu_best.success:
            print(f"WARNING: Optimizer failed for best-fit mu: {res_mu_best.message}")

        density_mu_best, rest = _getDensity(res_mu_best.x)
        beta_mu_best, mu_best = rest
        mu_best = float(np.clip(mu_best, 0.0, 2/3))

        _, alpha_mu_best = _reconstructInitialMassAndAlpha(density_mu_best, beta_mu_best, mu_best)

        m_initial_mu_best, m_final_mu_best = _massesAt(alpha_mu_best, beta_mu_best, mu_best, \
            density_mu_best)

    elif mass_constraint == "both":

        # Fit the density for mu = 0 (alpha and beta both follow analytically from the two masses)
        if fit_density:
            res_mu0 = scipy.optimize.minimize(_densityMinimizationBothMasses, [dens0], \
                args=(v_normed, ht_normed, 0), bounds=[(xmin[0], xmax[0])], method='Nelder-Mead')

            if not res_mu0.success:
                print(f"WARNING: Optimizer failed for mu=0: {res_mu0.message}")

            density_mu0 = res_mu0.x[0]

        else:
            # Both masses and the density are fixed: beta and alpha follow directly, nothing left
            #   to optimize
            density_mu0 = density

        alpha_mu0, beta_mu0 = _solveBothMasses(density_mu0, 0.0)

        m_initial_mu0, m_final_mu0 = _massesAt(alpha_mu0, beta_mu0, 0, density_mu0)

        # Fit the density for mu = 2/3
        if fit_density:
            res_mu23 = scipy.optimize.minimize(_densityMinimizationBothMasses, [dens0], \
                args=(v_normed, ht_normed, 2/3), bounds=[(xmin[0], xmax[0])], method='Nelder-Mead')

            if not res_mu23.success:
                print(f"WARNING: Optimizer failed for mu=2/3: {res_mu23.message}")

            density_mu23 = res_mu23.x[0]

        else:
            density_mu23 = density

        alpha_mu23, beta_mu23 = _solveBothMasses(density_mu23, 2.0/3.0)

        m_initial_mu23, m_final_mu23 = _massesAt(alpha_mu23, beta_mu23, 2/3, density_mu23)

        # Data-driven best-fit mu: jointly fit (density, mu) with mu free within [0, 2/3] (or just
        #   mu, if the density is set); beta is still derived analytically from mu and the two
        #   given masses
        if fit_density:
            x0_mu_best = [dens0, 1/3]
            bnds_mu_best = [(xmin[0], xmax[0]), (0.0, 2/3)]
        else:
            x0_mu_best = [1/3]
            bnds_mu_best = [(0.0, 2/3)]

        res_mu_best = scipy.optimize.minimize(_densityMuMinimizationBothMasses, x0_mu_best, \
            args=(v_normed, ht_normed), bounds=bnds_mu_best, method='Nelder-Mead')

        if not res_mu_best.success:
            print(f"WARNING: Optimizer failed for best-fit mu: {res_mu_best.message}")

        density_mu_best, rest = _getDensity(res_mu_best.x)
        mu_best = rest[0]
        mu_best = float(np.clip(mu_best, 0.0, 2/3))

        alpha_mu_best, beta_mu_best = _solveBothMasses(density_mu_best, mu_best)

        m_initial_mu_best, m_final_mu_best = _massesAt(alpha_mu_best, beta_mu_best, mu_best, density_mu_best)

    if verbose:

        print("\n===== Alpha-Beta Mass Fit =====\n")

        print("--- Inputs (as given by the caller) ---")
        print(f"mass_constraint      = {mass_constraint}")
        if mass_constraint == "initial":
            print(f"mass_init            = {mass:.3f} kg  (input)")
        elif mass_constraint == "final":
            print(f"mass_final           = {mass:.3f} kg  (input)")
        elif mass_constraint == "both":
            print(f"mass_init            = {mass_initial:.3f} kg  (input)")
            print(f"mass_final           = {mass_final:.3f} kg  (input)")
        print(f"GAMMA_A              = {shape_coeff*gamma:.3f}  (input)")

        if fit_density:
            print("density              = not given -> fitted below, separately for each mu")
        else:
            print(f"density              = {density:.1f} kg/m^3  (input, held fixed - not fitted)")

        if v_init_given:
            print(f"v_init               = {v_init/1000:.3f} km/s  (input)")
        else:
            print(f"v_init               = {v_init/1000:.3f} km/s  (DERIVED: median of the "
                  "top-20% highest-altitude points)")

        if mass_constraint == "initial":
            if v_final_given:
                print(f"v_final              = {v_final_input/1000:.3f} km/s  (input, but IGNORED: "
                      "under mass_constraint='initial' v_final is always re-derived from the "
                      "fitted alpha/beta - see 'DERIVED' value below)")
            print(f"v_final              = {v_final/1000:.3f} km/s  (DERIVED: minimum velocity "
                  "predicted by the fitted alpha-beta model)")
        elif v_final_given:
            print(f"v_final              = {v_final/1000:.3f} km/s  (input)")
        else:
            print(f"v_final              = {v_final/1000:.3f} km/s  (DERIVED: minimum velocity "
                  "predicted by a preliminary alpha-beta fit)")

        density_note = "  (= fixed input density)" if not fit_density else ""

        print("\n--- mu = 0 ---")
        print(f"density_mu0          = {density_mu0:.1f} kg/m^3{density_note}")
        print(f"alpha_mu0            = {alpha_mu0:.6f}")
        print(f"beta_mu0             = {beta_mu0:.6f}")
        print(f"m_initial_mu0        = {m_initial_mu0:.3f} kg")
        print(f"m_final_mu0          = {m_final_mu0:.3f} kg")
        print("\n--- mu = 2/3 ---")
        print(f"density_mu23         = {density_mu23:.1f} kg/m^3{density_note}")
        print(f"alpha_mu23           = {alpha_mu23:.6f}")
        print(f"beta_mu23            = {beta_mu23:.6f}")
        print(f"m_initial_mu23       = {m_initial_mu23:.3f} kg")
        print(f"m_final_mu23         = {m_final_mu23:.3f} kg")
        if mass_constraint == "initial":
            print("\n--- Best-fit mu ---")
            print("not applicable: mu has no effect on the fit under an initial-mass constraint")
        else:
            print("\n--- Best-fit mu ---")
            print(f"mu_best              = {mu_best:.3f}")
            print(f"density_mu_best      = {density_mu_best:.1f} kg/m^3{density_note}")
            print(f"alpha_mu_best        = {alpha_mu_best:.6f}")
            print(f"beta_mu_best         = {beta_mu_best:.6f}")
            print(f"m_initial_mu_best    = {m_initial_mu_best:.3f} kg")
            print(f"m_final_mu_best      = {m_final_mu_best:.3f} kg")


    ### Fit-quality diagnostics ###

    # Relative distance-to-bound tolerance
    bound_tol = 1e-2
    warnings_list = []

    _checkBound = _makeBoundChecker(warnings_list, tol=bound_tol)

    # Check the fitted densities against the optimization bounds
    if fit_density:
        _checkBound("density_mu0", density_mu0, xmin[0], xmax[0])
        _checkBound("density_mu23", density_mu23, xmin[0], xmax[0])

    # Check the fitted betas (under "both" beta is analytic, not fitted, so it's skipped)
    if mass_constraint in ["initial", "final"]:
        _checkBound("beta_mu0", beta_mu0, xmin[1], xmax[1])
        _checkBound("beta_mu23", beta_mu23, xmin[1], xmax[1])

    if fit_density and mass_constraint in ["final", "both"]:
        _checkBound("density_mu_best", density_mu_best, xmin[0], xmax[0])

    if mass_constraint == "final":
        _checkBound("beta_mu_best", beta_mu_best, xmin[1], xmax[1])

    def _checkVelocityRecovery(name, alpha, beta):
        """ Under mass_constraint="final"/"both", beta is derived analytically assuming a
            particular v_final (given or derived from a preliminary unconstrained fit), but
            alpha/beta are then fitted to best match the v-h curve shape. Nothing else guarantees
            that plugging the fitted alpha/beta back into alphaBetaVelocity() actually reproduces
            that same v_final, so check it explicitly here.
        """

        v_final_model = np.min(alphaBetaVelocity(ht_data, alpha, beta, v_init))
        rel_diff = abs(v_final_model - v_final)/v_final

        if rel_diff > bound_tol:
            warnings_list.append(f"{name}: model v_final from the fitted alpha/beta "
                f"({v_final_model:.1f} m/s) differs from the v_final used in the mass constraint "
                f"({v_final:.1f} m/s) by {100*rel_diff:.2f}% (> {100*bound_tol:.3g}%)")

    # Check that the fitted solutions reproduce the v_final used to impose the mass constraint
    if mass_constraint in ["final", "both"]:
        _checkVelocityRecovery("v_final_mu0", alpha_mu0, beta_mu0)
        _checkVelocityRecovery("v_final_mu23", alpha_mu23, beta_mu23)
        _checkVelocityRecovery("v_final_mu_best", alpha_mu_best, beta_mu_best)

    # Print all collected fit-quality warnings
    if warnings_list:

        print()
        print(f"WARNING: One or more fit-quality checks exceeded the {100*bound_tol:.3g}% "
            "tolerance.")
        print(f"The fit may be poorly constrained (a fitted parameter landed within "
            f"{100*bound_tol:.3g}% of an optimization bound) or may not satisfy the imposed "
            f"constraints self-consistently (the model's recovered v_final differs from the "
            f"assumed v_final by more than {100*bound_tol:.3g}%).")
        for warning in warnings_list:
            print(f"  - {warning}")
        print()

    ### ###

    if plot:

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 6))

        # Observed data
        ax.scatter(v_data/1000, ht_data/1000, s=10, color='0.5', alpha=0.6, zorder=1, \
            label="Observed data")

        # Height grid used to draw the fitted curves
        ht_arr = np.linspace(np.min(ht_data), np.max(ht_data), 200)

        def _fitLabel(mu_label, dens_val, alpha_val, beta_val, m_init_val, m_final_val, v_final_model):
            """ One parameter per line, so each value can be read on its own instead of
                several being crammed onto the same line.
            """

            lines = [mu_label]

            if fit_density:
                lines.append("$\\rho$ = {:.0f} kg/m$^3$".format(dens_val))

            lines.append("$\\alpha$ = {:.3f}".format(alpha_val))
            lines.append("$\\beta$ = {:.3f}".format(beta_val))
            lines.append("$m_i$ = {:.2e} kg".format(m_init_val))
            lines.append("$m_f$ = {:.2e} kg".format(m_final_val))
            lines.append("$v_f$ = {:.3f} km/s".format(v_final_model/1000))

            return "\n".join(lines)

        # (label, density, alpha, beta, m_init, m_final, color) per fitted branch
        curves = [
            ("$\\mu$ = 0",   density_mu0,  alpha_mu0,  beta_mu0,  m_initial_mu0,  m_final_mu0,  'b'),
            ("$\\mu$ = 2/3", density_mu23, alpha_mu23, beta_mu23, m_initial_mu23, m_final_mu23, 'g'),
        ]

        if mu_best is not None:
            curves.append((
                "$\\mu$ = {:.2f}".format(mu_best),
                density_mu_best, alpha_mu_best, beta_mu_best, m_initial_mu_best, m_final_mu_best,
                'r'
            ))

        for mu_label, dens_val, alpha_val, beta_val, m_init_val, m_final_val, color in curves:

            vel_arr = alphaBetaVelocity(ht_arr, alpha_val, beta_val, v_init)

            # Computed the same way as in _checkVelocityRecovery() above (min over the
            # actual ht_data points, not over the ht_arr plotting grid), so the number
            # shown in the legend always matches what that check evaluates.
            v_final_model = np.min(alphaBetaVelocity(ht_data, alpha_val, beta_val, v_init))

            ax.plot(vel_arr/1000, ht_arr/1000, color=color, \
                label=_fitLabel(mu_label, dens_val, alpha_val, beta_val, m_init_val, m_final_val,
                    v_final_model))

        # Build the input-parameter summary shown in the title (as given by the caller)
        if mass_constraint == "initial":
            mass_line = "mass_init = {:.3g} kg (input)".format(mass_initial)
        elif mass_constraint == "final":
            mass_line = "mass_final = {:.3g} kg (input)".format(mass_final)
        else:
            mass_line = "mass_init = {:.3g} kg, mass_final = {:.3g} kg (input)".format(
                mass_initial, mass_final)

        v_init_tag = "input" if v_init_given else "derived"
        v_final_tag = "input" if (v_final_given and mass_constraint != "initial") else "derived"

        density_line = "\ndensity = {:.1f} kg/m$^3$ (fixed input)".format(density) \
            if not fit_density else ""

        title = (
            "mass_constraint = '{:s}'   slope = {:.2f}°   $\\Gamma A$ = {:.3f}\n"
            "{:s}\n"
            "$v_0$ = {:.2f} km/s ({:s})   $v_f$ = {:.2f} km/s ({:s})"
            "{:s}"
        ).format(
            mass_constraint, np.degrees(slope), gamma*shape_coeff,
            mass_line,
            v_init/1000, v_init_tag, v_final/1000, v_final_tag,
            density_line
        )

        ax.set_title(title, fontsize=9, loc='left')
        ax.set_xlabel("Velocity (km/s)")
        ax.set_ylabel("Height (km)")

        ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5))

        fig.tight_layout()

        plt.show()

    return (
        v_init,
        v_final,

        density_mu0,
        alpha_mu0,
        beta_mu0,
        m_initial_mu0,
        m_final_mu0,

        density_mu23,
        alpha_mu23,
        beta_mu23,
        m_initial_mu23,
        m_final_mu23,

        mu_best,
        density_mu_best,
        alpha_mu_best,
        beta_mu_best,
        m_initial_mu_best,
        m_final_mu_best
    )


### JOINT DYNAMICS + LIGHT CURVE FIT (Gritsevich & Koschny 2011) ###


def _profiledMagOffset(mag_obs, mag_model_zero, sigma_mag=1.0):
    """ L1-optimal additive magnitude offset (profiled amplitude).

        The M0 minimizing sum(|mag_obs - (mag_model_zero + M0)|/sigma_mag) is the
        1/sigma_mag-weighted median of (mag_obs - mag_model_zero) - for a scalar sigma_mag this
        reduces to the plain median. Weighting matters whenever a per-point sigma_mag is given:
        an unweighted median would let noisy points (e.g. the faint ends of the light curve) pull
        M0 - and hence K and any derived luminous efficiency - as hard as the precise ones.

        Note that this is the exact conditional optimum of the sigma-normalized L1 cost, and only
        an approximation to that of the soft-L1 cost the (alpha, beta) search descends - see
        fitAlphaBetaLightCurve() for the caveat.

    Arguments:
        mag_obs: [ndarray] Observed magnitudes.
        mag_model_zero: [ndarray] Model magnitudes computed with a zero offset (M0 = 0).

    Keyword arguments:
        sigma_mag: [float or ndarray] Magnitude uncertainty, a scalar or per-point. A scalar
            (default 1.0) weights all points equally.

    Return:
        mag_offset: [float] The profiled magnitude offset M0.
    """

    diffs = mag_obs - mag_model_zero

    # Uniform weights - the weighted median reduces to the plain median
    if np.ndim(sigma_mag) == 0:
        return np.median(diffs)

    # Weighted median: sort the differences and find where the cumulative weight crosses half of
    #   the total weight
    weights = 1.0/np.asarray(sigma_mag, dtype=np.float64)

    sort_indices = np.argsort(diffs)
    diffs_sorted = diffs[sort_indices]
    cum_weights = np.cumsum(weights[sort_indices])

    half_weight = 0.5*cum_weights[-1]
    index = np.searchsorted(cum_weights, half_weight)

    # If the half-weight falls exactly on a point boundary, average the two straddling values
    #   (this makes the uniform-weight case agree exactly with np.median)
    if (index < len(diffs_sorted) - 1) and np.isclose(cum_weights[index], half_weight):
        return 0.5*(diffs_sorted[index] + diffs_sorted[index + 1])

    return diffs_sorted[index]


def _lightCurveResiduals(x, mu, ht_normed_lc, mag_lc, sigma_mag):
    """ Light curve residual block: model magnitude residuals with the amplitude (magnitude
        offset) profiled out analytically at every call (see _profiledMagOffset()). Used both
        as the light curve half of _jointResiduals() and, on its own, for the purely photometric
        alpha-beta fit shown when plot=True in fitAlphaBetaLightCurve() (alpha/beta constrained
        by the light curve shape alone, ignoring the dynamics entirely).

    Arguments:
        x: [ndarray] Fit parameters (ln alpha, ln beta).
        mu: [float] Shape change coefficient (fixed).
        ht_normed_lc: [ndarray] Light curve heights normalized to HT_NORM_CONST.
        mag_lc: [ndarray] Observed absolute magnitudes.
        sigma_mag: [float or ndarray] Magnitude uncertainty, a scalar or per-point.

    Return:
        res: [ndarray] Sigma-normalized magnitude residuals.
    """

    alpha, beta = np.exp(x)

    mag_zero, _ = alphaBetaModelMagnitude(ht_normed_lc, alpha, beta, mu)
    mag_offset = _profiledMagOffset(mag_lc, mag_zero, sigma_mag)

    return (mag_zero + mag_offset - mag_lc)/sigma_mag


def _jointResiduals(x, mu, v_normed, ht_normed, ht_normed_lc, mag_lc, sigma_v_normed, sigma_mag,
        dyn_weight, lc_weight):
    """ Concatenated residual vector used by fitAlphaBetaLightCurve(): dynamics (_dynResiduals(),
        shared with minimizeAlphaBetaRobust()) + light curve (_lightCurveResiduals()).
        x = (ln alpha, ln beta); the magnitude offset (amplitude) is profiled out analytically at
        every call, so it never enters x and the nonlinear search stays 2-dimensional.

        dyn_weight/lc_weight multiply their respective (already sigma-normalized) block, so the
        two blocks' relative contribution to the fit can be rebalanced independently of how many
        points each one has - see the dyn_weight/lc_weight docs on fitAlphaBetaLightCurve().

    Arguments:
        x: [ndarray] Fit parameters (ln alpha, ln beta).
        mu: [float] Shape change coefficient (fixed).
        v_normed: [ndarray] Normalized velocity in (0, 1).
        ht_normed: [ndarray] Dynamics heights normalized to HT_NORM_CONST.
        ht_normed_lc: [ndarray] Light curve heights normalized to HT_NORM_CONST.
        mag_lc: [ndarray] Observed absolute magnitudes.
        sigma_v_normed: [float] Normalized velocity uncertainty.
        sigma_mag: [float or ndarray] Magnitude uncertainty, a scalar or per-point.
        dyn_weight: [float] Multiplier on the dynamics residual block.
        lc_weight: [float] Multiplier on the light curve residual block.

    Return:
        res: [ndarray] Concatenated (dynamics, light curve) sigma-normalized residuals.
    """

    res_dyn = dyn_weight*_dynResiduals(x, v_normed, ht_normed, sigma_v_normed)
    res_lc = lc_weight*_lightCurveResiduals(x, mu, ht_normed_lc, mag_lc, sigma_mag)

    return np.concatenate([res_dyn, res_lc])


def _lightCurveResidualsFreeMu(x, ht_normed_lc, mag_lc, sigma_mag):
    """ Same as _lightCurveResiduals(), but with mu folded into x as a 3rd free parameter
        (x = (ln alpha, ln beta, mu)) instead of held fixed. Used only to derive an effective
        sigma_mag for the optional free-mu fit in fitAlphaBetaLightCurve() (fit_free_mu=True),
        mirroring how _lightCurveResiduals() is used to derive sigma_mag at each fixed mu.

    Arguments:
        x: [ndarray] Fit parameters (ln alpha, ln beta, mu).
        ht_normed_lc: [ndarray] Light curve heights normalized to HT_NORM_CONST.
        mag_lc: [ndarray] Observed absolute magnitudes.
        sigma_mag: [float or ndarray] Magnitude uncertainty, a scalar or per-point.

    Return:
        res: [ndarray] Sigma-normalized magnitude residuals.
    """

    return _lightCurveResiduals(x[:2], x[2], ht_normed_lc, mag_lc, sigma_mag)


def _jointResidualsFreeMu(x, v_normed, ht_normed, ht_normed_lc, mag_lc, sigma_v_normed, sigma_mag,
        dyn_weight, lc_weight):
    """ Same as _jointResiduals(), but with mu folded into x as a 3rd free parameter
        (x = (ln alpha, ln beta, mu)) instead of held fixed at one of mu_values. Used only by the
        optional free-mu fit in fitAlphaBetaLightCurve() (fit_free_mu=True), which lets the
        shape-change coefficient itself be estimated from the data instead of only compared at a
        handful of fixed values.

    Arguments:
        x: [ndarray] Fit parameters (ln alpha, ln beta, mu).
        v_normed: [ndarray] Normalized velocity in (0, 1).
        ht_normed: [ndarray] Dynamics heights normalized to HT_NORM_CONST.
        ht_normed_lc: [ndarray] Light curve heights normalized to HT_NORM_CONST.
        mag_lc: [ndarray] Observed absolute magnitudes.
        sigma_v_normed: [float] Normalized velocity uncertainty.
        sigma_mag: [float or ndarray] Magnitude uncertainty, a scalar or per-point.
        dyn_weight: [float] Multiplier on the dynamics residual block.
        lc_weight: [float] Multiplier on the light curve residual block.

    Return:
        res: [ndarray] Concatenated (dynamics, light curve) sigma-normalized residuals.
    """

    return _jointResiduals(x[:2], x[2], v_normed, ht_normed, ht_normed_lc, mag_lc, sigma_v_normed,
        sigma_mag, dyn_weight, lc_weight)


def alphaBetaLuminosityF(v_normed, beta, mu):
    """ Dimensionless luminosity function f(v), Eq. (14) of Gritsevich & Koschny (2011):

            f(v) = v^3 Delta (beta v^2/(1-mu) + 1) exp(beta (mu v^2 - 1)/(1-mu)),
            Delta = Ei(beta) - Ei(beta v^2)

        Together with an amplitude K, this gives the luminous intensity I = K*f(v), Eq. (13)
        of the same paper. Note: at mu = 0 the exponential factor reduces to exp(-beta),
        independent of v.

    Arguments:
        v_normed: [ndarray] Normalized velocity in (0, 1).
        beta: [float] Mass loss parameter.
        mu: [float] Shape change coefficient. mu must be < 1 for Eq. (14) to be defined; the
            physically meaningful range is [0, 2/3] (see alphaBetaMasses()).

    Return:
        f: [ndarray] f(v) >= 0.
    """

    if mu >= 1.0:
        raise ValueError("mu must be < 1 (mu = 1 requires a separate treatment).")

    v = np.asarray(v_normed, dtype=np.float64)

    delta = scipy.special.expi(beta) - scipy.special.expi(beta*v**2)

    return v**3*delta*(beta*v**2/(1.0 - mu) + 1.0)*np.exp(beta*(mu*v**2 - 1.0)/(1.0 - mu))


def alphaBetaModelMagnitude(ht_normed, alpha, beta, mu, mag_offset=0.0):
    """ Model absolute magnitude at the given normalized heights (up to an additive offset).

            mag(y) = mag_offset - 2.5 log10( f(v(y)) ),   I = P_0M 10^(-0.4 mag)

    Arguments:
        ht_normed: [ndarray] Height normalized to HT_NORM_CONST.
        alpha: [float] Ballistic coefficient.
        beta: [float] Mass loss parameter.
        mu: [float] Shape change coefficient, see alphaBetaLuminosityF().

    Keyword arguments:
        mag_offset: [float] Additive magnitude offset M0 (amplitude term, see
            fitAlphaBetaLightCurve()).

    Return:
        (mag, v_model): [tuple] Model absolute magnitudes and model normalized velocities.
    """

    v_model = alphaBetaVelocityNormed(ht_normed, alpha, beta)

    f_val = alphaBetaLuminosityF(v_model, beta, mu)

    # Floor to avoid log of zero at v -> 1 (model is arbitrarily faint there)
    f_val = np.clip(f_val, 1e-300, None)

    return mag_offset - 2.5*np.log10(f_val), v_model


def fitAlphaBetaLightCurve(
        v_data,
        ht_data,
        ht_lc_data,
        mag_abs_data,
        v_init=None,
        mu_values=(0.0, 2.0/3.0),
        dyn_method='robust',
        sigma_v=None,
        sigma_mag=None,
        dyn_weight=1.0,
        lc_weight=1.0,
        f_scale=2.0,
        p0m=P_0M,
        fit_free_mu=False,
        verbose=True,
        plot=False):
    """ Simultaneous fit of the alpha-beta model to the dynamics (height vs. velocity) AND the
        light curve (height vs. absolute magnitude), following Gritsevich (2007, 2009) for the
        dynamics and Gritsevich & Koschny (2011) for the luminosity.

    Dynamics, dimensionless v = V/V_e, y = h/HT_NORM_CONST (see alphaBetaHeightNormed()):

        y = ln(2 alpha) + beta - ln(Delta),   Delta = Ei(beta) - Ei(beta v^2)          (7)

    Eq. (7) does not depend on mu: the trajectory h(V) alone constrains only (alpha, beta). mu
    and the amplitude enter only through the luminosity (see alphaBetaLuminosityF()):

        I = tau M_e V_e^3 sin(gamma)/(2 h0) * f(v) = K * f(v)                          (13)
        f(v) = v^3 Delta (beta v^2/(1-mu) + 1) exp(beta (mu v^2 - 1)/(1 - mu))         (14)

    Fitting strategy:
        1. Stage 0: classic Q4 minimization (minimizeAlphaBeta(), dynamics alone)
           -> initial (alpha_0, beta_0).
        2. Joint stage, for each fixed mu in mu_values: robust least squares (soft-L1) over
           x = (ln alpha, ln beta) with residuals

               r_dyn,i = dyn_weight * (v_model(y_i) - v_obs,i) / sigma_v_normed
               r_lc,j  = lc_weight * (mag_model(y_j) - mag_obs,j) / sigma_mag[j]

           where v_model(y) = alphaBetaVelocityNormed(y) (the unique root of Eq. (7), since
           dy/dv = 2 exp(beta v^2)/(v Delta) > 0), and
           mag_model(y) = M0 - 2.5 log10(f(v_model(y))). The additive magnitude offset M0 (i.e.
           the amplitude K = p0m*10^(-0.4*M0)) is profiled out analytically at every iteration
           as the 1/sigma_mag-weighted median of the magnitude differences (the exact conditional
           optimum of the sigma-normalized L1 cost, and a close approximation to that of the
           soft-L1 cost actually descended - see _profiledMagOffset()), so the nonlinear search
           is only 2-dimensional per mu.
        3. Optional (fit_free_mu=True): the same joint problem again, but with mu itself as a 3rd
           free coordinate (bounded to [0, 2/3]) instead of fixed at each mu_values entry - a
           genuine continuous best-fit mu, as opposed to 'best_fixed_mu' below, which is only an argmin
           over the (typically 2) fixed values in mu_values.

    Statistical caveats (see dyn_weight/lc_weight, sigma_v, sigma_mag below for the mitigations
    this function offers):
        - Both blocks assume independent, diagonal-covariance residuals. In practice, consecutive
          dynamics/photometry points from the same station are not statistically independent
          (shared systematics between frames); this correlation is not modeled here, which is a
          standard simplification in alpha-beta fitting but worth keeping in mind.
        - With unit weights and per-point sigma, the total influence of each block scales with
          its point count, so a light curve with many more samples than the dynamics track (or
          vice versa) can dominate the fit even if every individual sigma is correctly calibrated;
          see dyn_weight/lc_weight to rebalance this.
        - alpha_std_rel/beta_std_rel (in the returned 'fits') are local curvature estimates from
          the Jacobian of the *robustified* (soft-L1) cost, not rigorous statistical covariances -
          soft-L1 reweights residuals nonlinearly, so (J^T J)^-1 is only a Gauss-Newton
          approximation, not the inverse Fisher information of Gaussian noise. The profiled
          (weighted-median) M0 also makes the residual vector only piecewise-smooth in
          (alpha, beta), which further degrades the finite-difference Jacobian these estimates
          are computed from. Treat them as approximate/relative uncertainties and use bootstrap
          resampling for publication-grade error bars.
        - When sigma_mag is auto-derived (the default, sigma_mag=None), each mu (and the free-mu
          fit, if fit_free_mu=True) gets its OWN independently-derived sigma_mag from its OWN
          residuals. A mu that fits the light curve worse ends up with a larger self-derived
          sigma_mag, which shrinks its normalized residuals and thus its 'cost' - i.e. a worse mu
          can look artificially cost-competitive. This means 'cost' is only a fair comparison
          ACROSS DIFFERENT mu (best_fixed_mu; free-mu vs. a fixed mu) when every branch shares the same
          sigma_mag, which requires passing it explicitly (scalar or per-point array) instead of
          leaving it to be auto-derived. Comparisons of alpha/beta/M0 *within* a single mu's fit
          are unaffected by this - it only concerns comparing 'cost' (or anything derived from it)
          across different mu.

    Conventions follow WMPL: absolute magnitude (100 km), heights in m, velocities in m/s.

    Like fitAlphaBeta()/fitAlphaBetaMass(), this function does no atmosphere modeling itself: it
    takes ht_data/ht_lc_data exactly as given (both for the fit AND for the plot=True scatter
    points) and treats y = ht/HT_NORM_CONST as if HT_NORM_CONST were a literal exponential
    atmosphere scale height. That is only valid if ht_data/ht_lc_data have ALREADY been rescaled
    to the equivalent simple exponential atmosphere (see rescaleHeightToExponentialAtmosphere(),
    used this way in this module's __main__) - the real atmosphere is not a simple exponential, so
    fitting raw geometric/NRLMSISE heights directly biases the fitted alpha/beta/mu.

    Arguments:
        v_data: [ndarray] Velocity data (m/s).
        ht_data: [ndarray] Height data for the dynamics (m), already rescaled to the equivalent
            exponential atmosphere (see above) - NOT raw geometric/NRLMSISE heights.
        ht_lc_data: [ndarray] Height data of the photometric points (m), same rescaling
            requirement as ht_data. May differ from ht_data (different sampling/stations);
            non-finite points are dropped independently of ht_data.
        mag_abs_data: [ndarray] Absolute magnitudes (100 km) at ht_lc_data.

    Keyword arguments:
        v_init: [float] Initial velocity (m/s). If None, median of the first 20% of points
            (min 10 points), same convention as fitAlphaBeta() - except that here the inputs are
            first sorted by decreasing height internally, so they don't have to be time-ordered.
        mu_values: [tuple] Values of the shape change coefficient to fit (fixed per fit). Must be
            in [0, 1); the physically meaningful range is [0, 2/3] (see alphaBetaMasses()) and a
            warning is printed for values above it. Default fits both physical bounds.
        dyn_method: [str] 'robust' (default) or 'q4' - which dynamics-only fit becomes the
            'alpha_dyn'/'beta_dyn' reference and the seed for the joint stage below. 'q4' is the
            classic Gritsevich (2007) fit (minimizeAlphaBeta()); 'robust' additionally refines it
            with minimizeAlphaBetaRobust() (least squares directly on the velocity residuals,
            robustified with soft-L1 and f_scale below), which tracks the deceleration/"knee" of
            the trajectory more faithfully than Q4's own transformed residual - see
            fitAlphaBeta()'s docstring for why. Since the joint stage re-fits (alpha, beta)
            regardless, dyn_method mainly affects the seed/reference quality, not the final
            per-mu results - but a better seed can matter on noisier or sparser trajectories.
        sigma_v: [float] Velocity uncertainty (m/s) used to weight the dynamics residuals. If
            None, estimated from the MAD of the Q4 fit's velocity residuals (floored at 1 m/s) -
            always from the Q4 fit specifically, even if dyn_method='robust', to avoid the
            circularity of a fit weighting itself by its own residuals. Note that this MAD-derived
            value is an *effective* uncertainty that absorbs both measurement noise and
            alpha-beta model misspecification - it is not a pure instrumental/measurement
            uncertainty, even though it is used as one here.
        sigma_mag: [float, ndarray or None] Photometric uncertainty (mag). Three modes:
              - None (default): DERIVED separately for each mu, from the MAD of the residuals of
                an unweighted, soft-L1 photometry-only fit at that mu (mirrors how sigma_v is
                derived from Stage 0 - but per mu here, since unlike the dynamics, Eq. (7), the
                light curve model depends on mu; see alphaBetaLuminosityF()). Floored at 0.01 mag.
                The value(s) actually used end up in the returned 'fits'[mu]['sigma_mag'].
              - a single value: applied to every point, for every mu.
              - a per-point array the same length as mag_abs_data (before the non-finite
                filtering below), applied for every mu.
            A per-point array (from e.g. RMS/SNR-based photometric error estimates) is the most
            correct option whenever available: photometric uncertainty is typically much larger
            near the detection threshold (~0.5-1 mag) than near peak brightness (~0.05 mag), and
            neither a constant value nor the MAD-derived default capture that - they only avoid
            the alternative of an arbitrary hardcoded constant.
        dyn_weight: [float] Extra multiplicative weight on the (already sigma-normalized)
            dynamics residual block, on top of whatever weight its point count already gives it.
            Default 1 (no rebalancing beyond sigma_v/sigma_mag). Use together with lc_weight if
            one block's point count is swamping the other one's contribution to the fit; e.g.
            lc_weight = sqrt(len(v_data)/len(mag_abs_data)) with dyn_weight = 1 makes the two
            *blocks* contribute equally in total, rather than each individual point contributing
            equally regardless of which block it is in.
        lc_weight: [float] Extra multiplicative weight on the light curve residual block. See
            dyn_weight.
        f_scale: [float] soft-L1 scale of scipy.optimize.least_squares (residuals are
            pre-normalized by sigma [and dyn_weight/lc_weight], so f_scale ~ 2 means the loss
            transitions at ~2 sigma).
        p0m: [float] Power of a zero-magnitude meteor (W).
        fit_free_mu: [bool] If True, additionally run the joint fit with mu itself as a 3rd free
            parameter (see 'Fitting strategy' step 3 above) and return it as 'mu_free_fit'. Off
            by default: it is an extra nonlinear fit (small added cost) and, with alpha/beta/mu
            all free at once, is more prone to being poorly constrained with sparse or noisy
            light curves than the fixed-mu fits - check 'mu_free_fit'['success'] and the
            fit-quality warnings before trusting it.
        verbose: [bool] If True, print the adopted inputs, the Stage-0 result, and, for every mu,
            the fitted alpha/beta/M0/K and any fit-quality warnings.
        plot: [bool] If True, show a two-panel plot (height vs. velocity; height vs. magnitude)
            with the observed data and the fitted curve for every mu (plus the free-mu fit, in a
            distinct color, if fit_free_mu=True). Blocks execution until the plot window closed.

    Return:
        results: [dict] With keys:
            'v_init': adopted initial velocity (m/s),
            'sigma_v': adopted velocity uncertainty (m/s, see the sigma_v caveat above),
            'alpha_dyn', 'beta_dyn': Stage-0 (dynamics only, Q4) values,
            'fits': {mu: {'alpha', 'beta',
                          'alpha_std_rel', 'beta_std_rel' (approximate, curvature-based - see
                              the statistical caveats above, not rigorous statistical errors),
                          'mag_offset',
                          'K' (= tau M_e V_e^3 sin(slope)/(2 h0), W - see
                              alphaBetaLuminousEfficiency() to recover tau from K, given an
                              assumed density/shape/drag and the entry slope),
                          'sigma_mag' (the value(s) actually used for this mu - only
                              informative when the sigma_mag argument was None, since it is then
                              derived independently for each mu; otherwise just echoes the input),
                          'cost', 'success'}},
            'best_fixed_mu': mu (from mu_values) with the lowest robust cost - an argmin over the fixed
                mu_values grid, NOT a continuous optimum (see fit_free_mu for that).
            'mu_free_fit': None unless fit_free_mu=True, in which case a dict with the same shape
                as one 'fits'[mu] entry, plus 'mu' (the fitted shape-change coefficient itself)
                and 'mu_std' (its approximate/relative uncertainty, same Gauss-Newton caveat as
                alpha_std_rel/beta_std_rel).
    """

    v_data = np.asarray(v_data, dtype=np.float64)
    ht_data = np.asarray(ht_data, dtype=np.float64)
    ht_lc_data = np.asarray(ht_lc_data, dtype=np.float64)
    mag_abs_data = np.asarray(mag_abs_data, dtype=np.float64)

    # sigma_mag may be None (derived per mu inside the fit loop below - see there and the
    # docstring), a scalar (applied to every point, for every mu), or a per-point array (also
    # applied for every mu - track which, so it can be carried through the same
    # filtering/sorting as mag_abs_data below and so verbose printing can format it appropriately.
    sigma_mag_given = sigma_mag is not None
    sigma_mag_per_point = sigma_mag_given and np.ndim(sigma_mag) > 0
    if sigma_mag_per_point:
        sigma_mag = np.asarray(sigma_mag, dtype=np.float64)
        if len(sigma_mag) != len(mag_abs_data):
            raise ValueError(
                "sigma_mag must be None, a scalar, or an array the same length as mag_abs_data."
            )

    if len(v_data) != len(ht_data):
        raise ValueError("v_data and ht_data must have the same length.")
    if len(ht_lc_data) != len(mag_abs_data):
        raise ValueError("ht_lc_data and mag_abs_data must have the same length.")

    # Validate the shape-change coefficients up front (the luminosity model diverges as mu -> 1,
    #   see alphaBetaLuminosityF())
    for mu in mu_values:
        if (mu < 0) or (mu >= 1):
            raise ValueError("All mu_values must be in the range [0, 1), got {:g}.".format(mu))
        if mu > 2.0/3.0:
            print("WARNING: mu = {:g} is outside the physically expected range [0, 2/3] "
                "(see alphaBetaMasses()).".format(mu))

    # Filter out non-finite points, dynamics and light curve independently (they may come from
    # different stations/instruments and need not share a sampling)
    dyn_mask = np.isfinite(v_data) & np.isfinite(ht_data)
    v_data, ht_data = v_data[dyn_mask], ht_data[dyn_mask]

    lc_mask = np.isfinite(mag_abs_data) & np.isfinite(ht_lc_data)
    if sigma_mag_per_point:
        lc_mask &= np.isfinite(sigma_mag)
    ht_lc_data, mag_abs_data = ht_lc_data[lc_mask], mag_abs_data[lc_mask]
    if sigma_mag_per_point:
        sigma_mag = sigma_mag[lc_mask]

    if len(v_data) < 2:
        raise ValueError("At least 2 finite dynamics points are required.")
    if len(ht_lc_data) < 2:
        raise ValueError("At least 2 finite light curve points are required.")

    # Dynamics: sort by decreasing height (beginning -> end of trajectory)
    order = np.argsort(ht_data)[::-1]
    ht_data, v_data = ht_data[order], v_data[order]

    # Light curve: sort by decreasing height
    order = np.argsort(ht_lc_data)[::-1]
    ht_lc_data, mag_abs_data = ht_lc_data[order], mag_abs_data[order]
    if sigma_mag_per_point:
        sigma_mag = sigma_mag[order]

    # Compute the initial velocity if not given (WMPL convention, matches fitAlphaBeta())
    v_init_given = v_init is not None
    if not v_init_given:

        max_index = int(0.2*len(v_data))
        if max_index < 10:
            max_index = 10

        v_init = np.median(v_data[:max_index])

    # Normalize observables
    v_normed = v_data/v_init
    ht_normed = ht_data/HT_NORM_CONST
    ht_normed_lc = ht_lc_data/HT_NORM_CONST

    dyn_method = dyn_method.lower()
    if dyn_method not in ('q4', 'robust'):
        raise ValueError("dyn_method must be 'q4' or 'robust', got '{:s}'.".format(dyn_method))

    # --- Stage 0: dynamics-only fit, used as the seed for the joint stage below and (unless
    # overridden) as the 'alpha_dyn'/'beta_dyn' reference. Always starts from the classic Q4 fit
    # (minimizeAlphaBeta()), exactly like fitAlphaBeta() does.
    alpha0, beta0 = minimizeAlphaBeta(v_normed, ht_normed)

    # Estimate sigma_v from the Q4 residuals if not given (robust MAD estimator, shared with
    # fitAlphaBeta(method='robust')). Always derived from the Q4 fit specifically, even when
    # dyn_method='robust' below, to avoid the circularity of a fit weighting itself by its own
    # residuals.
    sigma_v_given = sigma_v is not None
    if not sigma_v_given:
        sigma_v = _estimateSigmaV(alpha0, beta0, v_normed, ht_normed, v_init)

    sigma_v_normed = sigma_v/v_init

    if dyn_method == 'robust':
        # Refine the Q4 seed into the robust dynamics-only fit (see fitAlphaBeta()'s docstring
        # for why this tracks the trajectory - especially its end, where the deceleration/beta
        # signal actually is - more faithfully than Q4's own transformed residual).
        alpha0, beta0 = minimizeAlphaBetaRobust(v_normed, ht_normed, sigma_v_normed, \
            f_scale=f_scale, x0_alpha_beta=(alpha0, beta0))

    if verbose:
        print()
        print("===== Alpha-Beta + Light Curve Joint Fit =====")
        print()
        print("--- Inputs ---")
        print("v_init               = {:.3f} km/s  ({:s})".format(v_init/1000,
            "input" if v_init_given else "DERIVED: median of the top-20% highest points"))
        print("sigma_v              = {:.1f} m/s  ({:s})".format(sigma_v,
            "input" if sigma_v_given else "DERIVED: MAD of Stage-0 velocity residuals (effective, "
            "absorbs model misspecification too - not a pure measurement uncertainty)"))
        if not sigma_mag_given:
            print("sigma_mag            = not given -> DERIVED per μ below (MAD of an unweighted "
                "photometry-only fit at that μ)")
        elif sigma_mag_per_point:
            print("sigma_mag            = per-point array (min={:.3f}, median={:.3f}, "
                "max={:.3f}) mag".format(np.min(sigma_mag), np.median(sigma_mag), np.max(sigma_mag)))
        else:
            print("sigma_mag            = {:.3f} mag (constant for every point)".format(sigma_mag))
        print("n_dynamics           = {:d} points".format(len(v_data)))
        print("n_light_curve        = {:d} points".format(len(ht_lc_data)))
        if dyn_weight != 1.0 or lc_weight != 1.0:
            print("dyn_weight           = {:.3g}".format(dyn_weight))
            print("lc_weight            = {:.3g}".format(lc_weight))
        print()
        print("--- Dynamics only ({:s}) ---".format('Q4' if dyn_method == 'q4' else 'robust, Q4-seeded'))
        print("alpha_dyn            = {:.6f}".format(alpha0))
        print("beta_dyn             = {:.6f}".format(beta0))

    # --- Joint stage: robust least squares per fixed mu, same (alpha, beta) bounds as Stage 0 ---
    log_bounds = _logAlphaBetaBounds()

    # Clip the seed strictly inside the bounds in linear space before taking the log (a
    #   non-positive seed would otherwise produce a NaN that np.clip propagates)
    x0 = np.log(np.clip([alpha0, beta0], np.exp(log_bounds[0] + 1e-6), np.exp(log_bounds[1] - 1e-6)))

    # Shared by every scipy.optimize.least_squares() call below (sigma_mag derivation, the
    # per-mu joint fits, the optional free-mu fit, and the plot=True photometry-only diagnostic)
    lsq_kwargs = dict(loss='soft_l1', f_scale=f_scale, x_scale='jac')

    # Relative distance-to-bound tolerance, checked in the same log-space the optimizer
    # searches in. Unlike e.g. density in fitAlphaBetaMass(), alpha and beta span several
    # orders of magnitude between their bounds, so a *linear* check (_makeBoundChecker()) would
    # flag almost any physically reasonable value as "close to the lower bound"; log-space is
    # where "close to a bound" is actually meaningful here.
    bound_tol = 1e-2
    warnings_list = []

    def _checkLogBound(name, value, log_value, log_lower, log_upper):

        span = log_upper - log_lower

        if (log_value - log_lower)/span < bound_tol:
            warnings_list.append(
                "{:s}={:.6g} is close to the fit lower bound ({:.6g})".format(
                    name, value, np.exp(log_lower))
            )

        if (log_upper - log_value)/span < bound_tol:
            warnings_list.append(
                "{:s}={:.6g} is close to the fit upper bound ({:.6g})".format(
                    name, value, np.exp(log_upper))
            )

    fits = {}
    for mu in mu_values:

        if sigma_mag_given:
            sigma_mag_mu = sigma_mag
        else:
            # No sigma_mag given: derive one *for this mu*. Unlike sigma_v (which uses the single
            # mu-independent Stage-0 dynamics fit), the light curve model depends on mu (see
            # alphaBetaLuminosityF()), so there's no single mu-independent "Stage 0" to reuse here
            # - instead, run an unweighted (sigma_mag=1) preliminary photometry-only fit at this
            # mu (same soft-L1 robustness as the final fit, so a few bad points don't skew the
            # scale estimate) and take the MAD of its raw magnitude residuals, exactly mirroring
            # how sigma_v is derived from the Stage-0 residuals.
            res_prelim = scipy.optimize.least_squares(_lightCurveResiduals, x0, \
                args=(mu, ht_normed_lc, mag_abs_data, 1.0), bounds=log_bounds, **lsq_kwargs)

            sigma_mag_mu = 1.4826*np.median(np.abs(res_prelim.fun - np.median(res_prelim.fun)))

            # Floor at 0.01 mag
            sigma_mag_mu = max(sigma_mag_mu, 0.01)

        res = scipy.optimize.least_squares(_jointResiduals, x0, args=(mu, v_normed, ht_normed, \
            ht_normed_lc, mag_abs_data, sigma_v_normed, sigma_mag_mu, dyn_weight, lc_weight),
            bounds=log_bounds, **lsq_kwargs)

        if not res.success:
            print("WARNING: Optimizer failed for μ={:.3f}: {:s}".format(mu, res.message))

        alpha, beta = np.exp(res.x)

        # Recompute the profiled amplitude at the solution
        mag_zero, _ = alphaBetaModelMagnitude(ht_normed_lc, alpha, beta, mu)
        mag_offset = _profiledMagOffset(mag_abs_data, mag_zero, sigma_mag_mu)

        # Amplitude K = tau M_e V_e^3 sin(gamma)/(2 h0) in W
        lum_amplitude = p0m*10**(-0.4*mag_offset)

        # Local curvature ("covariance") of (ln alpha, ln beta) from the Jacobian of the
        # *robustified* soft-L1 cost - only a Gauss-Newton approximation of the curvature at the
        # optimum, NOT a rigorous statistical covariance: soft_l1 reweights residuals nonlinearly,
        # so (J^T J)^-1 here is not the inverse Fisher information of Gaussian noise. Treat
        # alpha_std_rel/beta_std_rel as approximate/relative uncertainties; use bootstrap
        # resampling for publication-grade errors.
        try:
            jtj = res.jac.T @ res.jac
            dof = max(len(res.fun) - len(res.x), 1)
            s2 = 2.0*res.cost/dof
            cov_log = s2*np.linalg.inv(jtj)
            alpha_std_rel, beta_std_rel = np.sqrt(np.diag(cov_log))
        except np.linalg.LinAlgError:
            alpha_std_rel = beta_std_rel = np.nan

        fits[mu] = {
            'alpha': alpha,
            'beta': beta,
            'alpha_std_rel': alpha_std_rel,
            'beta_std_rel': beta_std_rel,
            'mag_offset': mag_offset,
            'K': lum_amplitude,
            'sigma_mag': sigma_mag_mu,
            'cost': res.cost,
            'success': res.success,
        }

        _checkLogBound("alpha(μ={:.3f})".format(mu), alpha, res.x[0], log_bounds[0][0], log_bounds[1][0])
        _checkLogBound("beta(μ={:.3f})".format(mu), beta, res.x[1], log_bounds[0][1], log_bounds[1][1])

        if verbose:
            print()
            print("--- μ = {:.3f} ---".format(mu))
            if not sigma_mag_given:
                print("sigma_mag            = {:.3f} mag  (DERIVED for this μ: MAD of an "
                    "unweighted photometry-only fit)".format(sigma_mag_mu))
            print("alpha                = {:.6f} (±{:.1%})".format(alpha, alpha_std_rel))
            print("beta                 = {:.6f} (±{:.1%})".format(beta, beta_std_rel))
            print("M0                   = {:.3f} mag".format(mag_offset))
            print("K                    = {:.3e} W".format(lum_amplitude))
            print("cost                 = {:.4f}".format(res.cost))

    # --- Optional: fit mu itself, jointly with (alpha, beta), instead of only comparing it at
    # the fixed values in mu_values. Unlike the sigma_jit-style "fit the noise scale" idea some
    # domains use, this is a low-risk extension of the exact same soft-L1 least_squares problem
    # above: mu just becomes a 3rd free coordinate, bounded to [0, 2/3] (see alphaBetaMasses()),
    # not a new optimization paradigm - it directly shapes the light curve residuals, so unlike
    # an unconstrained scale/jitter parameter it is well identified without needing a separate
    # penalty term. See fitAlphaBetaMass()'s mu_best for the analogous idea applied there.
    mu_free_fit = None
    if fit_free_mu:

        mu_lo, mu_hi = 0.0, 2.0/3.0
        x0_free = np.append(x0, 0.5*(mu_lo + mu_hi))
        bounds_free = (np.append(log_bounds[0], mu_lo), np.append(log_bounds[1], mu_hi))

        if sigma_mag_given:
            sigma_mag_free = sigma_mag
        else:
            # Same MAD-of-an-unweighted-fit idea as the per-mu derivation above, but with mu
            # free here too (see _lightCurveResidualsFreeMu()).
            res_prelim_free = scipy.optimize.least_squares(_lightCurveResidualsFreeMu, x0_free, \
                args=(ht_normed_lc, mag_abs_data, 1.0), bounds=bounds_free, **lsq_kwargs)

            sigma_mag_free = 1.4826*np.median(
                np.abs(res_prelim_free.fun - np.median(res_prelim_free.fun)))

            # Floor at 0.01 mag
            sigma_mag_free = max(sigma_mag_free, 0.01)

        res_free = scipy.optimize.least_squares(_jointResidualsFreeMu, x0_free, args=(v_normed, \
            ht_normed, ht_normed_lc, mag_abs_data, sigma_v_normed, sigma_mag_free, dyn_weight,
            lc_weight), bounds=bounds_free, **lsq_kwargs)

        if not res_free.success:
            print("WARNING: Optimizer failed for the free-μ fit: {:s}".format(res_free.message))

        alpha_free, beta_free = np.exp(res_free.x[:2])
        mu_free = float(np.clip(res_free.x[2], mu_lo, mu_hi))

        mag_zero_free, _ = alphaBetaModelMagnitude(ht_normed_lc, alpha_free, beta_free, mu_free)
        mag_offset_free = _profiledMagOffset(mag_abs_data, mag_zero_free, sigma_mag_free)
        lum_amplitude_free = p0m*10**(-0.4*mag_offset_free)

        # Same Gauss-Newton local-curvature caveat as the fixed-mu fits above, now for
        # (ln alpha, ln beta, mu) jointly.
        try:
            jtj = res_free.jac.T @ res_free.jac
            dof = max(len(res_free.fun) - len(res_free.x), 1)
            s2 = 2.0*res_free.cost/dof
            cov_free = s2*np.linalg.inv(jtj)
            alpha_std_rel_free, beta_std_rel_free, mu_std_free = np.sqrt(np.diag(cov_free))
        except np.linalg.LinAlgError:
            alpha_std_rel_free = beta_std_rel_free = mu_std_free = np.nan

        mu_free_fit = {
            'mu': mu_free,
            'alpha': alpha_free,
            'beta': beta_free,
            'alpha_std_rel': alpha_std_rel_free,
            'beta_std_rel': beta_std_rel_free,
            'mu_std': mu_std_free,
            'mag_offset': mag_offset_free,
            'K': lum_amplitude_free,
            'sigma_mag': sigma_mag_free,
            'cost': res_free.cost,
            'success': res_free.success,
        }

        # mu's bounds are narrow and linear (unlike alpha/beta's), so the plain linear-space
        # checker used by fitAlphaBetaMass is the appropriate one here, not _checkLogBound().
        _checkLinearBound = _makeBoundChecker(warnings_list, tol=bound_tol)
        _checkLinearBound("mu(free fit)", mu_free, mu_lo, mu_hi)
        _checkLogBound("alpha(free fit)", alpha_free, res_free.x[0], log_bounds[0][0], log_bounds[1][0])
        _checkLogBound("beta(free fit)", beta_free, res_free.x[1], log_bounds[0][1], log_bounds[1][1])

        if verbose:
            print()
            print("--- Free-μ fit (μ estimated jointly with α, β) ---")
            if not sigma_mag_given:
                print("sigma_mag            = {:.3f} mag  (DERIVED: MAD of an unweighted "
                    "photometry-only fit with μ also free)".format(sigma_mag_free))
            print("mu                   = {:.3f} (±{:.3f})".format(mu_free, mu_std_free))
            print("alpha                = {:.6f} (±{:.1%})".format(alpha_free, alpha_std_rel_free))
            print("beta                 = {:.6f} (±{:.1%})".format(beta_free, beta_std_rel_free))
            print("M0                   = {:.3f} mag".format(mag_offset_free))
            print("K                    = {:.3e} W".format(lum_amplitude_free))
            print("cost                 = {:.4f}".format(res_free.cost))

    # Pick the lowest-cost mu, preferring converged fits over failed ones (only fall back to a
    #   failed fit if every mu failed)
    successful_mus = [mu for mu in fits if fits[mu]['success']]
    best_fixed_mu = min(successful_mus if successful_mus else fits, key=lambda k: fits[k]['cost'])

    if verbose:
        print()
        print("Best-fit μ (lowest robust cost among mu_values) = {:.3f}".format(best_fixed_mu))

    # When sigma_mag is auto-derived (None), each mu (and the free-mu fit, if any) gets its OWN
    # independently-derived sigma_mag from its OWN residuals (see the sigma_mag docs). A mu that
    # fits the light curve worse ends up with larger residuals -> larger self-derived sigma_mag
    # -> its normalized residuals (and hence 'cost') are shrunk accordingly, which can make a
    # worse-fitting mu look artificially cost-competitive against a better one. So 'cost' (and
    # anything derived from comparing it, like best_fixed_mu, or a free-mu vs. fixed-mu comparison) is
    # only a fair comparison across mu when every branch shares the same sigma_mag - i.e. when it
    # was given explicitly (scalar or per-point array), not auto-derived.
    if (not sigma_mag_given) and (len(mu_values) > 1 or fit_free_mu):
        print()
        print(
            "NOTE: sigma_mag was auto-derived independently for each μ (and for the free-μ fit, "
            "if requested), so their 'cost' values are NOT on equal footing - comparisons across "
            "μ (best_fixed_mu, or free-μ vs. fixed-μ cost) should be read qualitatively, not as a "
            "rigorous likelihood comparison. Pass an explicit sigma_mag (scalar or per-point "
            "array) if you need directly comparable costs across μ."
        )

    if warnings_list:
        print()
        print(
            "WARNING: One or more fitted (alpha, beta) landed within {:.3g}% of an "
            "optimization bound - the fit may be poorly constrained.".format(100*bound_tol)
        )
        for warning in warnings_list:
            print("  - {:s}".format(warning))
        print()

    if plot:

        import matplotlib.pyplot as plt

        fig, (ax_dyn, ax_lc) = plt.subplots(ncols=2, figsize=(13, 6), sharey=True)

        # Observed data
        ax_dyn.scatter(v_data/1000, ht_data/1000, s=10, color='0.5', alpha=0.6, zorder=1, \
            label="Observed dynamics")
        ax_lc.scatter(mag_abs_data, ht_lc_data/1000, s=10, color='0.5', alpha=0.6, zorder=1, \
            label="Observed light curve")

        # Height grid spanning both datasets, used to draw the fitted curves
        ht_all = np.concatenate([ht_data, ht_lc_data])
        ht_arr = np.linspace(np.min(ht_all), np.max(ht_all), 200)
        ht_arr_normed = ht_arr/HT_NORM_CONST

        colors = plt.cm.viridis(np.linspace(0, 0.85, len(mu_values)))

        # Every legend entry below is deliberately kept to a single line and names its own role
        # ("Joint fit"/"Dynamics-only"/"Photometry-only") instead of relying on the reader to
        # infer it from line style alone - line style (solid/dashed/dotted) and color (mu) are
        # still used consistently so same-fit-type and same-mu curves can be matched at a
        # glance, but the label text alone should always say what the curve is.

        # Purely dynamic fit (Stage 0, dynamics alone). Eq. (7) doesn't depend on mu, so this is
        # a single reference curve on the velocity panel - not one per mu.
        vel_arr_dyn = alphaBetaVelocityNormed(ht_arr_normed, alpha0, beta0)*v_init
        ax_dyn.plot(vel_arr_dyn/1000, ht_arr/1000, color='k', linestyle='--', linewidth=1.5, \
            label="Dynamics-only ({:s}): $\\alpha$={:.2f}, $\\beta$={:.2f}".format(
                'Q4' if dyn_method == 'q4' else 'robust', alpha0, beta0))

        for mu, color in zip(mu_values, colors):

            fit = fits[mu]
            alpha, beta = fit['alpha'], fit['beta']

            # --- Joint fit (dynamics + light curve), the main result for this mu ---
            vel_arr = alphaBetaVelocityNormed(ht_arr_normed, alpha, beta)*v_init
            mag_arr, _ = alphaBetaModelMagnitude(ht_arr_normed, alpha, beta, mu, \
                mag_offset=fit['mag_offset'])

            ax_dyn.plot(vel_arr/1000, ht_arr/1000, color=color, \
                label="Joint fit, $\\mu$={:.2f}: $\\alpha$={:.2f}, $\\beta$={:.2f}".format(
                    mu, alpha, beta))
            ax_lc.plot(mag_arr, ht_arr/1000, color=color, \
                label="Joint fit, $\\mu$={:.2f}: $M_0$={:.2f}, $K$={:.2e} W".format(
                    mu, fit['mag_offset'], fit['K']))

            # --- Dynamics-only fit (alpha0, beta0 from above), projected onto the light curve:
            # same shape for every mu, only M0 is re-profiled per mu since f(v) depends on mu.
            # Shows how well the dynamics alone would have predicted the light curve shape.
            mag_zero_dyn, _ = alphaBetaModelMagnitude(ht_normed_lc, alpha0, beta0, mu)
            mag_offset_dyn = _profiledMagOffset(mag_abs_data, mag_zero_dyn, fits[mu]['sigma_mag'])
            mag_arr_dyn, _ = alphaBetaModelMagnitude(ht_arr_normed, alpha0, beta0, mu, \
                mag_offset=mag_offset_dyn)

            ax_lc.plot(mag_arr_dyn, ht_arr/1000, color=color, linestyle='--', linewidth=1.5, \
                label="Dynamics-only, $\\mu$={:.2f}: $M_0$={:.2f}".format(mu, mag_offset_dyn))

            # --- Purely photometric fit: alpha/beta constrained by the light curve alone
            # (dynamics ignored entirely), for comparison with the joint and dynamics-only fits.
            res_phot = scipy.optimize.least_squares(_lightCurveResiduals, x0, \
                args=(mu, ht_normed_lc, mag_abs_data, fits[mu]['sigma_mag']), bounds=log_bounds,
                **lsq_kwargs)

            alpha_phot, beta_phot = np.exp(res_phot.x)
            mag_zero_phot, _ = alphaBetaModelMagnitude(ht_normed_lc, alpha_phot, beta_phot, mu)
            mag_offset_phot = _profiledMagOffset(mag_abs_data, mag_zero_phot, fits[mu]['sigma_mag'])

            vel_arr_phot = alphaBetaVelocityNormed(ht_arr_normed, alpha_phot, beta_phot)*v_init
            mag_arr_phot, _ = alphaBetaModelMagnitude(ht_arr_normed, alpha_phot, beta_phot, mu, \
                mag_offset=mag_offset_phot)

            ax_dyn.plot(vel_arr_phot/1000, ht_arr/1000, color=color, linestyle=':', \
                linewidth=1.5, label="Photometry-only, $\\mu$={:.2f}: $\\alpha$={:.2f}, "
                    "$\\beta$={:.2f}".format(mu, alpha_phot, beta_phot))
            ax_lc.plot(mag_arr_phot, ht_arr/1000, color=color, linestyle=':', \
                linewidth=1.5, label="Photometry-only, $\\mu$={:.2f}: $M_0$={:.2f}".format(
                    mu, mag_offset_phot))

        # --- Free-mu fit (mu estimated jointly with alpha, beta - see fit_free_mu above), drawn
        # in a color deliberately outside the mu_values color family so it reads as "its own
        # thing" rather than being mistaken for one more mu_values branch.
        if mu_free_fit is not None:

            vel_arr_free = alphaBetaVelocityNormed(ht_arr_normed, mu_free_fit['alpha'],
                mu_free_fit['beta'])*v_init
            mag_arr_free, _ = alphaBetaModelMagnitude(ht_arr_normed, mu_free_fit['alpha'],
                mu_free_fit['beta'], mu_free_fit['mu'], mag_offset=mu_free_fit['mag_offset'])

            ax_dyn.plot(vel_arr_free/1000, ht_arr/1000, color='crimson', linewidth=2.5, zorder=4, \
                label="Free-$\\mu$ fit: $\\mu$={:.2f}$\\pm${:.2f}, $\\alpha$={:.2f}, "
                    "$\\beta$={:.2f}".format(mu_free_fit['mu'], mu_free_fit['mu_std'],
                    mu_free_fit['alpha'], mu_free_fit['beta']))
            ax_lc.plot(mag_arr_free, ht_arr/1000, color='crimson', linewidth=2.5, zorder=4, \
                label="Free-$\\mu$ fit: $\\mu$={:.2f}$\\pm${:.2f}, $M_0$={:.2f}".format(
                    mu_free_fit['mu'], mu_free_fit['mu_std'], mu_free_fit['mag_offset']))

        # Clip the view to the observed data range (with padding): the photometry-only fit at a
        # mu far from the true value can be poorly constrained and run away outside the data
        # range (see _lightCurveResiduals()), which would otherwise blow up the axis limits and
        # squash every other curve into an unreadable sliver.
        v_pad = 0.1*(np.max(v_data) - np.min(v_data))
        ax_dyn.set_xlim((np.min(v_data) - v_pad)/1000, (np.max(v_data) + v_pad)/1000)

        mag_pad = 0.1*(np.max(mag_abs_data) - np.min(mag_abs_data))
        ax_lc.set_xlim(np.max(mag_abs_data) + mag_pad, np.min(mag_abs_data) - mag_pad)

        ax_dyn.set_xlabel("Velocity (km/s)")
        ax_dyn.set_ylabel("Height (km)")
        ax_dyn.legend(fontsize=8, loc='best')

        ax_lc.set_xlabel("Absolute magnitude")
        ax_lc.legend(fontsize=8, loc='best')

        title = "Best $\\mu$ = {:.2f}   $v_0$ = {:.2f} km/s".format(
            best_fixed_mu, v_init/1000)
        if mu_free_fit is not None:
            title += "   |   Free-$\\mu$ fit: $\\mu$ = {:.2f}$\\pm${:.2f}".format(
                mu_free_fit['mu'], mu_free_fit['mu_std'])

        fig.suptitle(title)
        fig.tight_layout()

        plt.show()

    return {
        'v_init': v_init,
        'sigma_v': sigma_v,
        'alpha_dyn': alpha0,
        'beta_dyn': beta0,
        'fits': fits,
        'best_fixed_mu': best_fixed_mu,
        'mu_free_fit': mu_free_fit,
    }


def alphaBetaLuminousEfficiency(K, alpha, beta, slope, v_init, mu=0.0, dens=3500.0,
        shape_coeff=0.55, gamma=1.0, v_final=None, verbose=False):
    """ Recover the (dimensionless) luminous efficiency tau from a light curve amplitude K (e.g.
        fits[mu]['K'] returned by fitAlphaBetaLightCurve()), by combining it with the initial
        mass M_e that alphaBetaMasses() computes from the SAME alpha/beta/mu under an assumed
        density/shape/drag, via Eq. (13) of Gritsevich & Koschny (2011):

            K = tau * M_e * V_e^3 * sin(slope) / (2 * HT_NORM_CONST)
            =>  tau = K * 2 * HT_NORM_CONST / (M_e * V_e^3 * sin(slope))

        This is deliberately a separate post-processing step, not part of fitAlphaBetaLightCurve()
        itself: M_e depends on density/shape_coeff/gamma/slope, which that fit knows nothing about
        (see its docstring) - so tau is only as good as those assumptions, on top of whatever
        alpha/beta/mu/K/v_init were already fitted with. m_final is computed alongside m_init by
        the same alphaBetaMasses() call (at no extra cost) and returned too, purely as
        complementary mass context - it plays no role in the tau formula above.

        Naming note: the entry angle is 'slope' here (matching alphaBetaMasses()'s naming), NOT
        the 'gamma' used for it in the module's Eq. (13) - this function's 'gamma' argument, like
        alphaBetaMasses()'s, is the drag coefficient Gamma = Cd/2, not the entry angle. That
        clash in symbol reuse comes from the original papers; 'slope' is used throughout this
        module for the entry angle specifically to avoid it.

    Arguments:
        K: [float] Light curve amplitude (W), e.g. fits[mu]['K'] from fitAlphaBetaLightCurve().
        alpha: [float] Ballistic coefficient (from the same fit branch as K).
        beta: [float] Mass loss parameter (from the same fit branch as K).
        slope: [float] Entry angle (radians).
        v_init: [float] Initial velocity (m/s), e.g. the 'v_init' from fitAlphaBetaLightCurve().

    Keyword arguments:
        mu: [float] Shape change coefficient (from the same fit branch as K); see
            alphaBetaMasses().
        dens: [float] Assumed bulk density (kg/m^3).
        shape_coeff: [float] Shape coefficient; see alphaBetaMasses().
        gamma: [float] Drag coefficient Gamma = Cd/2; see alphaBetaMasses() (NOT the entry angle).
        v_final: [float] Final velocity (m/s). If given (together with v_init), m_final is
            computed from the full alpha-beta solution; if None (default), alphaBetaMasses()
            falls back to the simple v_final << v_init approximation. Has no effect on tau.
        verbose: [bool] If True, print the inputs and the derived M_e, m_final and tau.

    Return:
        (tau, m_init, m_final): [tuple]
            - tau: [float] Dimensionless luminous efficiency (fraction of the kinetic energy loss
                rate radiated in the photometric passband defined by P_0M).
            - m_init: [float] Initial mass M_e (kg) used to derive tau, from alphaBetaMasses()
                under the given dens/shape_coeff/gamma/slope - returned for reference, e.g. to
                cross-check against an independently known initial mass.
            - m_final: [float] Final mass (kg), from the same alphaBetaMasses() call - reference
                mass-loss context only, not used in the tau computation.
    """

    if K <= 0:
        raise ValueError("K must be positive.")
    if v_init <= 0:
        raise ValueError("v_init must be positive.")
    if np.sin(slope) <= 0:
        raise ValueError("slope must satisfy sin(slope) > 0.")

    # alphaBetaMasses() validates alpha/beta/dens/gamma/shape_coeff/mu itself
    m_init, m_final = alphaBetaMasses(alpha, beta, slope, mu=mu, dens=dens,
        shape_coeff=shape_coeff, gamma=gamma, vel_init=v_init, vel_end=v_final)

    tau = K*2.0*HT_NORM_CONST/(m_init*v_init**3*np.sin(slope))

    if verbose:
        print()
        print("===== Luminous Efficiency from K =====")
        print()
        print("--- Inputs ---")
        print("K                    = {:.3e} W".format(K))
        print("alpha                = {:.6f}".format(alpha))
        print("beta                 = {:.6f}".format(beta))
        print("mu                   = {:.3f}".format(mu))
        print("slope                = {:.2f} deg".format(np.degrees(slope)))
        print("v_init               = {:.3f} km/s".format(v_init/1000))
        print("v_final              = {:s}".format(
            "{:.3f} km/s".format(v_final/1000) if v_final is not None
            else "not given (simple v_final << v_init approximation used for m_final)"))
        print("density              = {:.1f} kg/m^3".format(dens))
        print("shape_coeff          = {:.3f}".format(shape_coeff))
        print("gamma (drag, Cd/2)   = {:.3f}".format(gamma))
        print()
        print("--- Result ---")
        print("M_e (initial mass)   = {:.3e} kg".format(m_init))
        print("m_final              = {:.3e} kg".format(m_final))
        print("tau (luminous eff.)  = {:.3e}  ({:.3f}%)".format(tau, 100*tau))

    return tau, m_init, m_final



if __name__ == "__main__":

    import os
    import argparse

    import matplotlib.pyplot as plt

    from wmpl.Utils.Pickling import loadPickle



    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Fit the alpha-beta model to the trajectory.")

    arg_parser.add_argument('traj_path', nargs="?", metavar='TRAJ_PATH', type=str, \
        help="Path to the trajectory pickle file.")

    arg_parser.add_argument('-v', '--obsvel', action="store_true", \
        help="""Fit alpha-beta on the observed velocity instead of the lag-smoothed model. """
        )

    arg_parser.add_argument('-d', '--dens', metavar='DENS', \
        help='Bulk density in kg/m^3 used to compute the final dynamic mass. Default is 3500 kg/m^3.', \
        type=float, default=3500)

    arg_parser.add_argument('-g', '--ga', metavar='GAMMA_A', \
        help='The product of the drag coefficient Gamma and the shape coefficient A. Used for computing the dynamic mass. Default is 0.55.', \
        type=float, default=0.55)

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    # If the trajectory pickle was given, load the orbital elements from it
    if cml_args.traj_path is not None:

        # Load the trajectory pickle
        traj = loadPickle(*os.path.split(cml_args.traj_path))

        # Dictionary for storing time, height, and velocity data per station
        ht_data_dict = {}

        # Construct an input data array
        ht_data = []
        time_data = []
        lat_data = []
        lon_data = []
        vel_data = []
        lag_data = []
        for obs in traj.observations:
            
            if obs.ignore_station:
                continue

            filter_mask = (obs.ignore_list == 0) & (obs.velocities != 0)

            ht_data += obs.model_ht[filter_mask].tolist()
            time_data += obs.time_data[filter_mask].tolist()
            lat_data += obs.model_lat[filter_mask].tolist()
            lon_data += obs.model_lon[filter_mask].tolist()
            vel_data += obs.velocities[filter_mask].tolist()
            lag_data += obs.lag[filter_mask].tolist()

            # Store the data for each station
            ht_data_dict[obs.station_id] = [obs.time_data[filter_mask], obs.model_ht[filter_mask], 
                                            obs.velocities[filter_mask]]


        ht_data = np.array(ht_data)
        time_data = np.array(time_data)
        lat_data = np.array(lat_data)
        lon_data = np.array(lon_data)
        vel_data = np.array(vel_data)
        lag_data = np.array(lag_data)


        # Print time, height, velocity for each station
        for station_id, data in ht_data_dict.items():

            t, h, v = data

            print()
            print("Station:", station_id)
            print("Time (s)   Height (km)   Velocity (km/s)")
            for i in range(len(t)):
                print("{:8.3f}   {:11.3f}   {:15.2f}".format(t[i], h[i]/1000, v[i]/1000))


        # Sort by height
        sorted_indices = np.argsort(ht_data)
        vel_data = vel_data[sorted_indices]
        lag_data  = lag_data[sorted_indices]
        time_data = time_data[sorted_indices]
        lat_data = lat_data[sorted_indices]
        lon_data = lon_data[sorted_indices]
        ht_data  = ht_data[sorted_indices]


        # Rescale the heights to the exponential atmosphere used by alpha-beta
        ht_data_rescaled = rescaleHeightToExponentialAtmosphere(lat_data, lon_data, ht_data, traj.jdt_ref)

        # Fit a functional model to the lag and use that for the alpha-beta fit instead of the noisy
        #   point-to-point velocity measurements
        print("Fitting lag function...")
        vel_data_smooth, lag_fit_params = lagFitVelocity(time_data, lag_data, vel_data, traj.v_init)

        # Choose which data will be used for alpha-beta fitting
        if cml_args.obsvel:
            vel_input = vel_data
        else:
            vel_input = vel_data_smooth

        # Estimate the alpha, beta parameters
        v_init, alpha, beta = fitAlphaBeta(vel_input, ht_data_rescaled, v_init=traj.v_init)

        # Estimate the final velocity from the fitted alpha-beta solution
        vel_end = alphaBetaVelocity(ht_data_rescaled[0], alpha, beta, v_init)

        # Compute initial and final mass
        m_init_mu0, m_final_mu0 = alphaBetaMasses(alpha, beta, traj.orbit.elevation_apparent_norot, \
            mu=0, dens=cml_args.dens, shape_coeff=cml_args.ga, gamma=1.0, vel_init=traj.v_init, vel_end=vel_end)
        m_init_mu23, m_final_mu23 = alphaBetaMasses(alpha, beta, traj.orbit.elevation_apparent_norot, \
            mu=2/3, dens=cml_args.dens, shape_coeff=cml_args.ga, gamma=1.0, vel_init=traj.v_init, vel_end=vel_end)


        print()
        print("Initial velocity = {:.2f} km/s".format(traj.v_init/1000))
        print()
        print("Alpha-beta analysis")
        print("-------------------")
        print("Alpha = {:.3f}".format(alpha))
        print("Beta  = {:.3f}".format(beta))
        print("-------------------")
        print("ln(beta)             = {:.2f}".format(np.log(beta)))
        print("ln(alpha*sin(slope)) = {:.2f}".format(np.log(alpha*np.sin(traj.orbit.elevation_apparent_norot))))
        print("-------------------")
        print("Masses with dens = {:d} kg/m^3, sphere:".format(int(cml_args.dens)))
        print(" * - note that the initial masses are usually 4-10x underestimated!")
        print("  mu = 0:")
        print("    Initial = {:.2e} kg".format(m_init_mu0))
        print("    Final   = {:.2e} kg".format(m_final_mu0))
        print("  mu = 2/3:")
        print("    Initial = {:.2e} kg".format(m_init_mu23))
        print("    Final   = {:.2e} kg".format(m_final_mu23))

        print("*********************************************")
        print()
        print("Lag fit:")
        print("-------------------")
        print("    - a1    = {:.3f}".format(abs(lag_fit_params[0])))
        print("    - a2    = {:.3f}".format(abs(lag_fit_params[1])))
        print("    - t0    = {:.3f} s".format(abs(lag_fit_params[2])))
        print("    - decel = {:.3f} km/s^2".format(abs(lag_fit_params[3])))



        # Predict velocity from height
        ht_end = traj.rend_ele - 5000
        if ht_end < 10000:
            ht_end = 10000
        elif (ht_end > 20000) and (ht_end < 35000):
            ht_end = 20000
        ht_arr = np.linspace(ht_end, traj.rbeg_ele + 5000, 200)
        vel_arr = alphaBetaVelocity(ht_arr, alpha, beta, v_init)



        ### Alpha-beta plot ###
        fig, (ax_ab, ax_vel, ax_lag, ax_lag_res) = plt.subplots(ncols=4, sharey=True, figsize=(14, 6))

        # Plot the data rescaled to an exponential atmosphere
        ax_ab.scatter(vel_data/1000, ht_data_rescaled/1000, s=5, label="Rescaled height to exp. atm")

        # Plot the smoothed velocity
        ax_ab.scatter(vel_data_smooth/1000, ht_data_rescaled/1000, color='r', s=1, \
            label="Lag-based velocity smoothing")

        # Plot the alpha-beta fit
        ax_ab.plot(vel_arr/1000, ht_arr/1000, color='k', \
            label="$v_0$ = {:.2f} km/s\n$\\alpha$ = {:.2f}\n$\\beta$ = {:.2f}".format(v_init/1000, alpha, \
                beta))

        ax_ab.set_xlabel("Velocity (km/s)")
        ax_ab.set_ylabel("Height (km)")

        ax_ab.legend(loc='upper left')


        ### Plot the lag fit ###

        # Plot the original data
        ax_vel.scatter(vel_data/1000, ht_data/1000, s=5, label="Observed heights")

        # Plot the smoothed velocity
        ax_vel.scatter(vel_data_smooth/1000, ht_data/1000, color='r', s=1, \
            label="Lag-based velocity smoothing")

        if not cml_args.obsvel:

            # If the exponental to linear transition point was used by the fit, plot it
            t0 = lag_fit_params[2]
            decel = lag_fit_params[3]
            if t0 < np.max(time_data):

                # Find the height closest to t0
                v_t0 = expLinearVelocity(t0, traj.v_init, *lag_fit_params)
                t0_index = np.argmin(np.abs(vel_data_smooth - v_t0))
                h_t0 = ht_data[t0_index]
                h_rescaled_t0 = ht_data_rescaled[t0_index]

                # Plot the t0 point
                ax_ab.scatter([v_t0/1000], [h_rescaled_t0/1000], label='t0, decel = {:.2f} km/s$^2$'.format(abs(decel)),\
                    color='r')
                ax_vel.scatter([v_t0/1000], [h_t0/1000], label='t0, decel = {:.2f} km/s$^2$'.format(abs(decel)),\
                    color='r')


                ### Compute the dynamic mass at the end ###

                # Compute the values at the point that is 1/4 before the end and t0
                midpoint_index = int(round((t0_index + 0)*1/4)) # Sorted by increasing height!
                ht_dyn = ht_data[midpoint_index]
                t_dyn = time_data[midpoint_index]
                v_dyn = expLinearVelocity(t_dyn, traj.v_init, *lag_fit_params)

                # Compute the dynamic mass
                dyn_mass = dynamicMass(cml_args.dens, traj.rend_lat, traj.rend_lon, ht_dyn, traj.jdt_ref, \
                    v_dyn, 1000*abs(decel), gamma=1.0, shape_factor=cml_args.ga)

                # Plot the point where the dynamic mass is estimated
                ax_vel.scatter([v_dyn/1000], [ht_dyn/1000], label='Dynamic mass = {:.3f} kg'.format(dyn_mass),\
                    color='k')


                print()
                print("Lag fit dynamic mass:")
                print("---------------------")
                print("Bulk density = {:5d} kg/m^3".format(int(cml_args.dens)))
                print("Height       = {:5.2f} km".format(ht_dyn/1000))
                print("Velocity     = {:5.2f} km/s".format(v_dyn/1000))
                print("Deceleration = {:5.2f} km/s^2".format(abs(decel)))
                print("Gamma*A      = {:5.2f}".format(cml_args.ga))
                print()
                print("Dynamic mass = {:5.3f} kg".format(dyn_mass))
                print("---------------------")

                ### ###

        ###

        ax_vel.set_xlabel("Velocity (km/s)")
        
        ax_vel.legend(loc='upper left')


        # Plot the lag and the lag fit
        ax_lag.scatter(lag_data/1000, ht_data/1000, s=5)
        ax_lag.plot(expLinearLag(time_data, *lag_fit_params)/1000, ht_data/1000, color='r', \
            label="Lag-based velocity smoothing")
        ax_lag.set_xlabel("Lag (km)")


        # Plot the lag fit residuals
        ax_lag_res.scatter(lag_data/1000 - expLinearLag(time_data, *lag_fit_params)/1000, ht_data/1000, s=5)
        ax_lag_res.set_xlabel("Lag fit residuals (km)")

        plt.tight_layout()

        plt.subplots_adjust(wspace=0)

        plt.show()



        ### PLOT METEORITE DROPPING POSSIBILITY

        # define x values
        x_mu = np.arange(0, 10, 0.00005)

        # function for mu = 0, 50 g possible meteorite:
        fun_50g_mu0 = lambda x_mu:np.log(13.2 - 3*x_mu)
        y_50g_mu0 = [fun_50g_mu0(i) for i in x_mu]

        # function for mu = 2/3, 50 g possible meteorite:
        fun_50g_mu23 = lambda x_mu:np.log(4.4 - x_mu)
        y_50g_mu23 = [fun_50g_mu23(i) for i in x_mu]

        # function for mu = 0, 1 kg possible meteorite:
        fun_1kg_mu0 = lambda x_mu:np.log(10.21 - 3*x_mu)
        y_1kg_mu0 = [fun_1kg_mu0(i) for i in x_mu]

        # function for mu = 2/3, 1 kg possible meteorite:
        fun_1kg_mu23 = lambda x_mu:np.log(3.4 - x_mu)
        y_1kg_mu23 = [fun_1kg_mu23(i) for i in x_mu]

        # plot mu0, mu2/3 lines and your poit:
        plt.plot(x_mu, y_50g_mu0, color='grey', label=r"50 g meteorite, $\mu = 0$", linestyle='dashed')
        plt.plot(x_mu, y_50g_mu23, color='k',   label=r"50 g meteorite, $\mu = 2/3$", linestyle='dashed')
        plt.plot(x_mu, y_1kg_mu0, color='grey', label=r"1 kg meteorite, $\mu = 0$")
        plt.plot(x_mu, y_1kg_mu23, color='k',   label=r"1 kg meteorite, $\mu = 2/3$")
        plt.scatter(
            [np.log(alpha*np.sin(traj.orbit.elevation_apparent_norot))],
            [np.log(beta)],
            color='r',
            label=(
            f"$m_{{final}}(\\mu=0)={m_final_mu0:.2e}\\,\\mathrm{{kg}}$\n"
            f"$m_{{final}}(\\mu=2/3)={m_final_mu23:.2e}\\,\\mathrm{{kg}}$"
            )
        )

        # defite plot parameters
        plt.xlim((-1, 8))
        plt.ylim((-5, 4))
        plt.xlabel("ln(alpha*sin(slope))")
        plt.ylabel("ln(beta)")
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()



        ### Plot dynamic pressure ###

        # Take mean meteor lat/lon as reference for the atmosphere model
        lat_mean = np.mean([traj.rbeg_lat, traj.rend_lat])
        lon_mean = meanAngle([traj.rbeg_lon, traj.rend_lon])

        # Compute the dynamic pressure using alpha and beta
        dyn_pressure = dynamicPressure(lat_mean, lon_mean, ht_arr, traj.jdt_ref, vel_arr)

        # Compute the dynamic pressure with the lag fit
        dyn_pressure_lag = dynamicPressure(lat_mean, lon_mean, ht_data_rescaled, traj.jdt_ref, 
            vel_data_smooth)
        

        # Print height, model velocity (from alpha-beta and lag), and the dynamic pressure computed from both
        print()
        print("AlphaBeta model:")
        print("Height (km)  Vel (km/s)  DynPress (MPa)")
        for i in range(len(ht_arr)):

            # Compute the index from reverse
            i_rev = len(ht_arr) - i - 1
            print("{:11.2f}  {:10.4f}  {:14.5f}".format(
                ht_arr[i_rev]/1000, 
                vel_arr[i_rev]/1000, 
                dyn_pressure[i_rev]/1e6
                )
                )

        print()
        print("Lag fit:")
        print("Height (km)  Vel (km/s)  DynPress (MPa)")
        for i in range(len(ht_data_rescaled)):

            # Compute the index from reverse
            i_rev = len(ht_data_rescaled) - i - 1

            print("{:11.2f}  {:10.4f}  {:14.5f}".format(
                ht_data_rescaled[i_rev]/1000, 
                vel_data_smooth[i_rev]/1000,
                dyn_pressure_lag[i_rev]/1e6
                )
                )


        # Plot dyn pressure
        plt.plot(dyn_pressure/1e6, ht_arr/1000, color='k', label='AlphaBeta')
        plt.plot(dyn_pressure_lag/1e6, ht_data_rescaled/1000, color='r', label='Lag fit')

        def findPeakDynPressure(dyn_pressure, ht_arr):
            """Find the peak dynamic pressure. """
            peak_dyn_pressure_index = np.argmax(dyn_pressure)
            peak_dyn_pressure = dyn_pressure[peak_dyn_pressure_index]/1e6
            peak_dyn_pressure_ht = ht_arr[peak_dyn_pressure_index]/1000
            return peak_dyn_pressure, peak_dyn_pressure_ht

        # Compute and mark alpha-beta dyn pressure peak on the graph
        peak_dyn_pressure, peak_dyn_pressure_ht = findPeakDynPressure(dyn_pressure, ht_arr)
        plt.scatter(peak_dyn_pressure, peak_dyn_pressure_ht, c='k', \
            label="Peak P = {:.2f} MPa\nHt = {:.2f} km".format(peak_dyn_pressure, peak_dyn_pressure_ht))

        # Compute and mark lag fit dyn pressure peak on the graph
        peak_dyn_pressure_lag, peak_dyn_pressure_ht_lag = findPeakDynPressure(dyn_pressure_lag, 
            ht_data_rescaled)
        plt.scatter(peak_dyn_pressure_lag, peak_dyn_pressure_ht_lag, c='r', \
            label="Peak P = {:.2f} MPa\nHt = {:.2f} km".format(peak_dyn_pressure_lag, peak_dyn_pressure_ht_lag))


        plt.legend()

        plt.ylabel("Height (km)")
        plt.xlabel("Dynamic pressure (MPa)")

        plt.show()


        ### ###


        ### Plot magnitude vs dynamic pressure ###

        for obs in traj.observations:

            if obs.absolute_magnitudes is not None:

                # Don't show magnitudes fainter than mag +8
                mag_filter = obs.absolute_magnitudes < 5

                if np.any(mag_filter):

                    # Get the model velocities at the observed heights
                    vel_model_obs = alphaBetaVelocity(obs.model_ht, alpha, beta, v_init)

                    # Compute the dynamic pressure
                    dyn_pres_station = dynamicPressure(lat_mean, lon_mean, obs.model_ht, traj.jdt_ref, vel_model_obs)

                    # Plot the magnitude
                    plt.plot(dyn_pres_station[mag_filter]/1e6, obs.absolute_magnitudes[mag_filter], label=obs.station_id)




        plt.xlabel("Dynamic pressure (MPa)")
        plt.ylabel("Absolute magnitude")
        plt.gca().invert_yaxis()

        plt.legend()

        plt.show()



        ###