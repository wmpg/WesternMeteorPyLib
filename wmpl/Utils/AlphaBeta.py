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
import scipy.stats
import scipy.interpolate


from wmpl.Utils.Math import meanAngle, vectNorm, integratedAutocorrTime
from wmpl.Utils.Physics import GAMMA_A, dynamicPressure, dynamicMass, earthRotationVelocity
from wmpl.Utils.Plotting import stationColors
from wmpl.Utils.AtmosphereDensity import getAtmDensity_vect, addAtmosphereArguments, setAtmosphere
from wmpl.Utils.TrajConversions import EARTH, G


# Scale height
HT_NORM_CONST = 7160.0

# Sea-level air density of the exponential atmosphere of the alpha-beta model (kg/m^3)
RHO_ATM_0 = 1.225

# Standard gravity (m/s^2) and mean Earth radius (m)
G0 = 9.81
R_EARTH = 6371008.7714

# Largest z of the median residual of a height band before the residuals are flagged as a trend
RESIDUAL_TREND_SIGMA = 2.0

# Smallest fraction of the entry velocity lost by the end of the fit to the lag for the dynamics to constrain
#   alpha, beta and the masses (only 1 of 61 such fits of GMN and UWO fireballs was constrained)
MIN_VELOCITY_LOSS = 0.1

# Bounds of a fitted initial velocity, as fractions of the median speed of the highest points
V_INIT_FIT_BOUNDS = (0.8, 1.5)

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



# Top of the air column integrated to rescale the heights (m)
RESCALE_COLUMN_TOP = 1000e3

# Steps of the column integration (m), fine below RESCALE_FINE_TOP (m) and coarse above
RESCALE_STEP_FINE, RESCALE_STEP_COARSE, RESCALE_FINE_TOP = 100.0, 5000.0, 200e3



def exponentialAtmosphereHeights(ht_data, profile_hts, profile_dens, method='column'):
    """ Map heights in a real atmosphere, given by a density profile, to the exponential atmosphere of the
        alpha-beta model, rho = RHO_ATM_0*exp(-h/HT_NORM_CONST).

    The alpha-beta solution depends on the height only through the air column above it, so by default a
    height is mapped to the one with the same column above. The 'density' method maps it to the height with
    the same local density, which is the same only where the local scale height is HT_NORM_CONST.

    Arguments:
        ht_data: [ndarray] Heights to rescale (m).
        profile_hts: [ndarray] Increasing heights of the density profile (m), from below the lowest of ht_data
            up to RESCALE_COLUMN_TOP.
        profile_dens: [ndarray] Air density at profile_hts (kg/m^3).

    Keyword arguments:
        method: [str] 'column' or 'density'. 'column' by default.

    Return:
        [ndarray] Rescaled heights (m).
    """

    profile_hts = np.asarray(profile_hts, dtype=np.float64)
    profile_dens = np.asarray(profile_dens, dtype=np.float64)
    log_dens = np.log(profile_dens)

    # Map the heights by the local density
    if method == 'density':
        return HT_NORM_CONST*np.log(RHO_ATM_0/np.exp(np.interp(ht_data, profile_hts, log_dens)))

    if method != 'column':
        raise ValueError("method must be 'column' or 'density', got {!r}.".format(method))

    # Compute the column above each profile height, integrating each interval as an exponential
    dh = np.diff(profile_hts)
    ratio = profile_dens[:-1]/profile_dens[1:]
    seg = np.where(np.abs(ratio - 1) > 1e-12, dh*(profile_dens[:-1] - profile_dens[1:])/np.log(ratio), \
        dh*profile_dens[:-1])
    column = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])

    # Add the column above the profile, with the scale height of its last interval
    scale_top = dh[-1]/np.log(ratio[-1])
    column += profile_dens[-1]*scale_top

    # Map the heights to those with the same column in the exponential atmosphere
    column_data = np.exp(np.interp(ht_data, profile_hts, np.log(column)))

    return HT_NORM_CONST*np.log(RHO_ATM_0*HT_NORM_CONST/column_data)



def exponentialAtmosphereDensityRatio(ht_data, profile_hts, profile_dens, method='column'):
    """ Compute the ratio of the real air density at the given heights to the density of the exponential
        atmosphere at their rescaled heights (see exponentialAtmosphereHeights()).

    The luminosity of the alpha-beta model is proportional to the local air density, which the column
    mapping gets wrong by this ratio, so the model magnitudes are too faint by 2.5 log10(ratio). The ratio
    is 1 with the density mapping.

    Arguments:
        ht_data: [ndarray] Heights in the real atmosphere (m).
        profile_hts: [ndarray] Increasing heights of the density profile (m).
        profile_dens: [ndarray] Air density at profile_hts (kg/m^3).

    Keyword arguments:
        method: [str] 'column' or 'density'. 'column' by default.

    Return:
        [ndarray] Real density over the model density at ht_data.
    """

    profile_hts = np.asarray(profile_hts, dtype=np.float64)
    log_dens = np.log(np.asarray(profile_dens, dtype=np.float64))

    # Compare the real and the model densities in log space
    ht_rescaled = exponentialAtmosphereHeights(ht_data, profile_hts, profile_dens, method=method)
    log_rho_real = np.interp(ht_data, profile_hts, log_dens)
    log_rho_model = np.log(RHO_ATM_0) - ht_rescaled/HT_NORM_CONST

    return np.exp(log_rho_real - log_rho_model)



def airDensityProfile(lat, lon, ht_data, jd, ht_min=None):
    """ Compute the air density profile of a meteor at the mean location of its points, from below its lowest
        point up to RESCALE_COLUMN_TOP, for exponentialAtmosphereHeights() and AlphaBetaSlantColumn.

    Arguments:
        lat: [ndarray] Latitude of the points (rad).
        lon: [ndarray] Longitude of the points (rad).
        ht_data: [ndarray] Height of the points (m).
        jd: [float] Julian date.

    Keyword arguments:
        ht_min: [float] Lowest height of the profile (m), e.g. to evaluate the model below the last point. One
            step of the profile below the lowest point by default.

    Return:
        (profile_hts, profile_dens):
            - profile_hts: [ndarray] Increasing heights of the profile (m).
            - profile_dens: [ndarray] Air density at the heights of the profile (kg/m^3).
    """

    # Take the density profile at the mean location of the points
    lat_mean = np.arctan2(np.mean(np.sin(lat)), np.mean(np.cos(lat)))
    lon_mean = meanAngle(np.atleast_1d(lon))

    if ht_min is None:
        ht_min = np.min(ht_data) - RESCALE_STEP_FINE

    profile_hts = np.concatenate([np.arange(ht_min, RESCALE_FINE_TOP, RESCALE_STEP_FINE), \
        np.arange(RESCALE_FINE_TOP, RESCALE_COLUMN_TOP + 1, RESCALE_STEP_COARSE)])
    profile_dens = getAtmDensity_vect(lat_mean, lon_mean, profile_hts, jd)

    return profile_hts, profile_dens



def rescaleHeightToExponentialAtmosphere(lat, lon, ht_data, jd, method='column', return_density_ratio=False):
    """ Given observed heights, rescale them from the real MSIS atmosphere to the simplified exponential
        atmosphere model used by the Alpha-Beta procedure (see exponentialAtmosphereHeights()).

    Arguments:
        lat: [ndarray] Latitude in radians.
        lon: [ndarray] Longitude in radians.
        ht_data: [ndarray] Height in meters.
        jd: [float] Julian date.

    Keyword arguments:
        method: [str] 'column' for the height with the same air column above, or 'density' for the height
            with the same local density. 'column' takes the density profile at the mean location of the
            points. 'column' by default.
        return_density_ratio: [bool] Also return the real density over the model density at the points (see
            exponentialAtmosphereDensityRatio()), for fitAlphaBetaLightCurve(lc_density_ratio=...). False by
            default.

    Return:
        ht_rescaled: [ndarray] Rescaled heights (m).
        density_ratio: [ndarray] Real density over the model density, only returned with
            return_density_ratio.
    """

    ht_data = np.asarray(ht_data, dtype=np.float64)

    # Map the heights by the local density
    if method == 'density':
        atm_dens = getAtmDensity_vect(lat, lon, ht_data, jd)
        ht_rescaled = HT_NORM_CONST*np.log(RHO_ATM_0/atm_dens)
        return (ht_rescaled, np.ones_like(ht_rescaled)) if return_density_ratio else ht_rescaled

    # Take the density profile at the mean location of the points
    profile_hts, profile_dens = airDensityProfile(lat, lon, ht_data, jd)

    # Map the heights by the air column above them
    ht_rescaled = exponentialAtmosphereHeights(ht_data, profile_hts, profile_dens, method=method)
    if return_density_ratio:
        return ht_rescaled, exponentialAtmosphereDensityRatio(ht_data, profile_hts, profile_dens, \
            method=method)

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


def lagFitVelocity(time_data, lag_data, vel_data, v0, seed=0):
    """ Fit a smooth model to the lag data, to improve the alpha-beta fit.

    Keyword arguments:
        seed: [int or None] Random seed for the basinhopping global optimizer, so that repeated runs
            on the same data give the same fit. None draws a fresh seed on every call.
    """


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
        minimizer_kwargs={'args':(time_data, lag_data, weights), 'method':'Nelder-Mead'}, seed=seed)
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


def _makeBoundChecker(warnings_list, tol=1e-2, log=False):
    """ Build a _checkBound(name, value, lower, upper) closure that appends a message to
        warnings_list whenever value lands within a relative tol of lower or upper. Shared by
        fitAlphaBetaMass() and fitAlphaBetaLightCurve() to flag fitted parameters that are
        poorly constrained (landed on an optimization bound rather than an interior optimum).

    Arguments:
        warnings_list: [list] List that warning strings are appended to.

    Keyword arguments:
        tol: [float] Relative distance-to-bound tolerance.
        log: [bool] If True, measure the distance to the bounds in log space instead of linear
            space. Appropriate for parameters whose bounds span several orders of magnitude
            (alpha, beta): a linear check would flag almost any physically reasonable value as
            "close to the lower bound" (e.g. beta=0.33 is only 0.7% of the way across the linear
            (1e-5, 50) span, but sits comfortably in the middle of it in log space). Requires
            value, lower and upper to be positive.

    Return:
        _checkBound: [callable] (name, value, lower, upper) -> None.
    """

    def _checkBound(name, value, lower, upper):

        if log:
            check_value, check_lower, check_upper = np.log(value), np.log(lower), np.log(upper)
        else:
            check_value, check_lower, check_upper = value, lower, upper

        span = check_upper - check_lower

        if (check_value - check_lower)/span < tol:
            warnings_list.append(
                "{:s}={:.6g} is close to the fit lower bound ({:.6g})".format(name, value, lower)
            )

        if (check_upper - check_value)/span < tol:
            warnings_list.append(
                "{:s}={:.6g} is close to the fit upper bound ({:.6g})".format(name, value, upper)
            )

    return _checkBound


def alphaBetaBoundWarnings(alpha, beta, tol=1e-2):
    """ Check if alpha or beta are at the bounds of the fit (ALPHA_BETA_BOUNDS), which means that the
        data did not constrain them.

    Arguments:
        alpha: [float] Ballistic coefficient.
        beta: [float] Mass loss parameter.

    Keyword arguments:
        tol: [float] Distance to a bound in the log space, relative to the span of the bounds.

    Return:
        [list] Warnings, one for each parameter at a bound. Empty if neither is.
    """

    bound_warnings = []

    _checkBound = _makeBoundChecker(bound_warnings, tol=tol, log=True)
    _checkBound("alpha", alpha, *ALPHA_BETA_BOUNDS[0])
    _checkBound("beta", beta, *ALPHA_BETA_BOUNDS[1])

    return bound_warnings



def _alphaBetaVelocityGrid(v_normed, ht_normed, n_grid=13):
    """ Find the alpha and beta with the smallest sum of squared velocity residuals on a grid evenly
        spaced in the log space of ALPHA_BETA_BOUNDS. Used as the starting point of a velocity fit
        when the Q4 fit has no minimum inside the bounds.

    Arguments:
        v_normed: [ndarray] Velocity data normalized to the initial velocity.
        ht_normed: [ndarray] Height data normalized to the scale height.

    Keyword arguments:
        n_grid: [int] Number of grid values of each parameter.

    Return:
        (alpha, beta): [tuple] The grid values with the smallest residuals.
    """

    alpha_grid = np.geomspace(*ALPHA_BETA_BOUNDS[0], n_grid)
    beta_grid = np.geomspace(*ALPHA_BETA_BOUNDS[1], n_grid)

    best_cost = np.inf
    best_alpha_beta = (alpha_grid[n_grid//2], beta_grid[n_grid//2])
    for alpha in alpha_grid:
        for beta in beta_grid:

            # The fast inversion is precise enough to find the starting point
            cost = np.sum((alphaBetaVelocityNormed(ht_normed, alpha, beta, fast=True) - v_normed)**2)

            if cost < best_cost:
                best_cost = cost
                best_alpha_beta = (alpha, beta)

    return best_alpha_beta



def minimizeAlphaBeta(v_normed, ht_normed, fast=False):
    """ initiates and calls the Q4 minimisation given in Gritsevich 2007 -
        'Validity of the photometric formula for estimating the mass of a fireball projectile'

    If the Q4 fit ends at the upper bound of beta, the velocity residuals are fitted instead (see
    minimizeAlphaBetaRobust()), starting from a grid search. The Q4 residual has no minimum inside
    the bounds when no velocity exceeds the initial velocity, as for velocities smoothed by a fit to
    the lag: both of its terms shrink as 1/beta, so the residual falls as beta grows to the bound
    while alpha falls to match, whatever the shape of the deceleration. The lower bound of beta (no
    mass loss) is a physical limit of the model, so a fit which ends there is kept.

    Keyword arguments:
        fast: [bool] Passed to alphaBetaVelocityNormed() in the velocity fit - see its docstring.
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

    alpha, beta = res.x

    # If beta ended at its upper bound (within 1% of the span of the bounds in the log space, as in
    #   alphaBetaBoundWarnings()), fit the velocity residuals instead
    beta_lower, beta_upper = ALPHA_BETA_BOUNDS[1]
    if np.log(beta_upper/beta) < 1e-2*np.log(beta_upper/beta_lower):

        alpha0, beta0 = _alphaBetaVelocityGrid(v_normed, ht_normed)

        # Robust (MAD) scatter of the velocity residuals of the grid solution, as in _estimateSigmaV()
        v_res = alphaBetaVelocityNormed(ht_normed, alpha0, beta0, fast=fast) - v_normed
        sigma_v_normed = max(1.4826*np.median(np.abs(v_res - np.median(v_res))), 1e-6)

        alpha_vel, beta_vel = minimizeAlphaBetaRobust(v_normed, ht_normed, sigma_v_normed, \
            x0_alpha_beta=(alpha0, beta0), fast=fast)

        print("NOTE: The Q4 alpha-beta fit ended at the upper bound of beta, so the velocity residuals "
            "were fitted instead: alpha = {:.3f}, beta = {:.4f} (Q4: alpha = {:.3f}, beta = {:.4f})".format( \
            alpha_vel, beta_vel, alpha, beta))

        alpha, beta = alpha_vel, beta_vel

    return np.array([alpha, beta])


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


def _estimateSigmaV(alpha0, beta0, v_normed, ht_normed, v_init, fast=False):
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

    Keyword arguments:
        fast: [bool] Passed through to alphaBetaVelocityNormed() - see its docstring.

    Return:
        sigma_v: [float] Effective velocity uncertainty (m/s), floored at 1 m/s.
    """

    v_model0 = alphaBetaVelocityNormed(ht_normed, alpha0, beta0, fast=fast)
    res0 = (v_model0 - v_normed)*v_init
    sigma_v = 1.4826*np.median(np.abs(res0 - np.median(res0)))

    return max(sigma_v, 1.0)


def _dynResiduals(x, v_normed, ht_normed, sigma_v_normed, fast=False):
    """ Dynamics residual block: model velocity residuals (v_model - v_obs), sigma-normalized.
        Used both as the dynamics half of _jointResiduals() (fitAlphaBetaLightCurve()) and, on
        its own, by minimizeAlphaBetaRobust(). This is the innermost per-iteration hot path of
        every 'robust' fit (called once per scipy.optimize.least_squares iteration), so fast=True
        here is where alphaBetaVelocityNormedLUT() actually pays off.

    Arguments:
        x: [ndarray] Fit parameters (ln alpha, ln beta).
        v_normed: [ndarray] Normalized velocity in (0, 1).
        ht_normed: [ndarray] Height normalized to HT_NORM_CONST.
        sigma_v_normed: [float] Normalized velocity uncertainty.

    Keyword arguments:
        fast: [bool] Passed through to alphaBetaVelocityNormed() - see its docstring.

    Return:
        res: [ndarray] Sigma-normalized velocity residuals.
    """

    alpha, beta = np.exp(x)
    v_model = alphaBetaVelocityNormed(ht_normed, alpha, beta, fast=fast)

    return (v_model - v_normed)/sigma_v_normed


def minimizeAlphaBetaRobust(v_normed, ht_normed, sigma_v_normed, loss='soft_l1', f_scale=2.0,
        x0_alpha_beta=None, return_fit_result=False, fast=False):
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
        fast: [bool] Passed through to _dynResiduals()/alphaBetaVelocityNormed() - see the latter's
            docstring. This is the residual scipy.optimize.least_squares calls every iteration, so
            fast=True is where the LUT speedup actually matters for this fit.

    Return:
        (alpha, beta), or (alpha, beta, res) if return_fit_result=True.
    """

    if x0_alpha_beta is None:
        x0_alpha_beta = minimizeAlphaBeta(v_normed, ht_normed, fast=fast)

    log_bounds = _logAlphaBetaBounds()

    # Clip the seed strictly inside the bounds in linear space before taking the log (a
    #   non-positive seed would otherwise produce a NaN that np.clip propagates)
    x0 = np.log(np.clip(x0_alpha_beta, np.exp(log_bounds[0] + 1e-6), np.exp(log_bounds[1] - 1e-6)))

    res = scipy.optimize.least_squares(_dynResiduals, x0, \
        args=(v_normed, ht_normed, sigma_v_normed, fast), \
        bounds=log_bounds, loss=loss, f_scale=f_scale, x_scale='jac')

    if not res.success:
        print("WARNING: Optimizer failed: {:s}".format(res.message))

    if return_fit_result:
        return tuple(np.exp(res.x)) + (res,)

    return tuple(np.exp(res.x))


def _alphaBetaRobustErrors(fit_res, alpha, beta, v_data, ht_data, v_init, v_init_given, sigma_v,
        sigma_v_init, loss, f_scale, ci, fast=False):
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

        alpha_ci_wald_lower/upper and beta_ci_wald_lower/upper are APPROXIMATE confidence
        intervals based on the local Gaussian approximation in log-space (equivalently: assuming
        alpha/beta are log-normal), at the `ci` level - exp(ln(alpha) +/- z*alpha_std_rel),
        asymmetric around alpha and, unlike alpha +/- z*alpha_std, never negative. The "_wald"
        in the name (and "approximate" here) is deliberate: this is NOT the same kind of object as
        wmpl.Utils.Math.confidenceInterval()'s output elsewhere in this library (e.g. Trajectory.py,
        REBOUND.py) - those are empirical percentiles of an actual Monte Carlo ensemble, whereas
        there is no ensemble here, only a parametric approximation built on a local quadratic
        (Wald-type) expansion of the cost function around the optimum - see profileAlphaBeta()
        for a rigorous, non-quadratic-assuming alternative when that assumption is in doubt (e.g.
        beta_std_rel large - see the Statistical caveats below). Don't drop the "_wald" qualifier
        when reading these out; it is the one thing standing between this and a false claim of
        likelihood-equivalent (let alone MC-equivalent) rigor.

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
        ci: [float] Confidence level (0-100, e.g. 95) for alpha_ci_wald_lower/upper and
            beta_ci_wald_lower/upper - same convention as
            wmpl.Utils.Math.confidenceInterval()'s `ci`.
        fast: [bool] Passed through to the recursive fitAlphaBeta() calls used for the
            v_init-sensitivity correction - see alphaBetaVelocityNormed()'s docstring.

    Return:
        errors: [dict] 'alpha', 'beta', 'v_init', 'alpha_std_rel', 'beta_std_rel', 'alpha_std',
            'beta_std', 'cov_fit', 'cov_log', 'corr_log', 'cov', 'corr',
            'alpha_ci_wald_lower', 'alpha_ci_wald_upper', 'beta_ci_wald_lower',
            'beta_ci_wald_upper', 'ci', 'sigma_v', 'sigma_v_init' - see fitAlphaBeta()'s
            docstring for details.
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
        if sigma_v_init is not None:
            print("WARNING: sigma_v_init was given but v_init is None (derived internally) - "
                "sigma_v_init is IGNORED in this case; v_init's uncertainty is instead "
                "auto-derived as the standard error of its median (see below).")

        # Standard error of the median (NOT the plain MAD, which measures the dispersion of the
        #   observations themselves, not the precision of the median as an estimator of v_init)
        #   of the same leading points fitAlphaBeta() used to derive v_init. This captures the
        #   RANDOM error of that estimator, but not its systematic one: the median of a still-
        #   decelerating leading window is itself biased low relative to the true v_init (see
        #   fitAlphaBeta()'s docstring caveat on this - no variance term can correct for a bias).
        max_index = max(int(0.2*len(v_data)), 10)
        v_head = v_data[:max_index]
        mad = np.median(np.abs(v_head - np.median(v_head)))
        sigma_v_init_used = 1.2533*1.4826*mad/np.sqrt(len(v_head))

    nan_result = {
        'alpha': alpha, 'beta': beta, 'v_init': v_init, 'alpha_std_rel': np.nan,
        'beta_std_rel': np.nan, 'alpha_std': np.nan, 'beta_std': np.nan,
        'cov_fit': np.full((2, 2), np.nan), 'cov_log': np.full((2, 2), np.nan),
        'corr_log': np.nan, 'cov': np.full((2, 2), np.nan), 'corr': np.nan,
        'alpha_ci_wald_lower': np.nan, 'alpha_ci_wald_upper': np.nan,
        'beta_ci_wald_lower': np.nan, 'beta_ci_wald_upper': np.nan,
        'ci': ci, 'sigma_v': sigma_v, 'sigma_v_init': sigma_v_init_used,
    }

    # Warn if the fitted (alpha, beta) landed near an optimization bound, reusing the same
    #   log-space _makeBoundChecker() fitAlphaBetaMass()/fitAlphaBetaLightCurve() already share
    #   (alpha/beta span orders of magnitude, so a linear-space check would misfire). The
    #   Gauss-Newton curvature below is only meaningful around a genuine local optimum, and tends
    #   to be degenerate (or numerically nonsensical) once the fit has been clipped against a box
    #   constraint instead of converging to one.
    bound_tol = 1e-2
    bound_warnings = []
    check_bound = _makeBoundChecker(bound_warnings, tol=bound_tol, log=True)

    check_bound("alpha", alpha, *ALPHA_BETA_BOUNDS[0])
    check_bound("beta", beta, *ALPHA_BETA_BOUNDS[1])

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
                method='robust', sigma_v=sigma_v, loss=loss, f_scale=f_scale, fast=fast)
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
    alpha_ci_wald_lower = alpha*np.exp(-z*alpha_std_rel)
    alpha_ci_wald_upper = alpha*np.exp(z*alpha_std_rel)
    beta_ci_wald_lower = beta*np.exp(-z*beta_std_rel)
    beta_ci_wald_upper = beta*np.exp(z*beta_std_rel)

    return {
        'alpha': alpha,
        'beta': beta,
        'v_init': v_init,
        'alpha_std_rel': alpha_std_rel,
        'beta_std_rel': beta_std_rel,
        'alpha_std': alpha_std,
        'beta_std': beta_std,
        'cov_fit': cov_fit,
        'cov_log': cov_log,
        'corr_log': corr_log,
        'cov': cov,
        'corr': corr,
        'alpha_ci_wald_lower': alpha_ci_wald_lower,
        'alpha_ci_wald_upper': alpha_ci_wald_upper,
        'beta_ci_wald_lower': beta_ci_wald_lower,
        'beta_ci_wald_upper': beta_ci_wald_upper,
        'ci': ci,
        'sigma_v': sigma_v,
        'sigma_v_init': sigma_v_init_used,
    }



def topSpeed(v_data, ht_data, fraction=0.1, n_min=10):
    """ Compute the median speed of the highest points.

    Arguments:
        v_data: [ndarray] Velocities (m/s).
        ht_data: [ndarray] Heights (m).

    Keyword arguments:
        fraction: [float] Fraction of the points to take from the top. 0.1 by default.
        n_min: [int] Smallest number of points to take. 10 by default.

    Return:
        [float] Median speed of the highest points (m/s).
    """

    order = np.argsort(ht_data)[::-1]
    n = min(len(order), max(n_min, int(fraction*len(order))))

    return np.median(np.asarray(v_data)[order[:n]])



def fitAlphaBetaFreeVinit(v_data, ht_data, v_init0, alpha0, beta0, sigma_v, loss, f_scale, fast=False):
    """ Fit the initial velocity together with alpha and beta by least squares on the velocity residuals,
        starting from a fit at a fixed initial velocity (see fitAlphaBeta(fit_v_init=True)).

    Arguments:
        v_data: [ndarray] Velocities (m/s).
        ht_data: [ndarray] Heights rescaled to the exponential atmosphere (m).
        v_init0: [float] Initial velocity of the fit at a fixed initial velocity (m/s).
        alpha0: [float] Alpha of the fit at a fixed initial velocity.
        beta0: [float] Beta of the fit at a fixed initial velocity.
        sigma_v: [float] Velocity uncertainty (m/s).
        loss: [str] scipy.optimize.least_squares loss.
        f_scale: [float] Robust loss scale, in units of sigma_v.

    Keyword arguments:
        fast: [bool] Use the LUT-accelerated alphaBetaVelocityNormedLUT(). False by default.

    Return:
        (v_init, alpha, beta, fit_res):
            - v_init: [float] Fitted initial velocity (m/s).
            - alpha: [float] Fitted alpha.
            - beta: [float] Fitted beta.
            - fit_res: [OptimizeResult] scipy.optimize.least_squares result for (v_init, ln alpha, ln beta),
                on the residuals normalized by sigma_v.
    """

    ht_normed = ht_data/HT_NORM_CONST
    log_bounds = _logAlphaBetaBounds()

    def velocityResiduals(x):
        """ Compute the velocity residuals normalized by sigma_v for x = (v_init, ln alpha, ln beta). """

        v_init, ln_alpha, ln_beta = x

        # Give a large residual where the model fails
        try:
            v_model = v_init*alphaBetaVelocityNormed(ht_normed, np.exp(ln_alpha), np.exp(ln_beta), fast=fast)
        except (ValueError, RuntimeError, OverflowError, ZeroDivisionError):
            return np.full_like(v_data, 1e6)

        res = (v_model - v_data)/sigma_v

        return np.where(np.isfinite(res), res, 1e6)


    # Bound the initial velocity by the median speed of the highest points
    v_top = topSpeed(v_data, ht_data)
    lower = [V_INIT_FIT_BOUNDS[0]*v_top, log_bounds[0][0], log_bounds[0][1]]
    upper = [V_INIT_FIT_BOUNDS[1]*v_top, log_bounds[1][0], log_bounds[1][1]]

    def clipToBounds(x):
        """ Move the parameters just inside their bounds. """

        return np.clip(x, np.array(lower) + 1e-9*np.abs(lower), np.array(upper) - 1e-9*np.abs(upper))


    # Start from a few points around the solution at the fixed initial velocity
    x0 = np.array([np.clip(v_init0, lower[0], upper[0]), np.log(alpha0), np.log(beta0)])
    starts = [x0, x0 + [0.02*x0[0], 0.0, 0.0], x0 + [0.05*x0[0], -0.5, 0.5]]

    # Keep the best fit
    best = None
    for start in starts:
        fit_res = scipy.optimize.least_squares(velocityResiduals, clipToBounds(start), \
            bounds=(lower, upper), loss=loss, f_scale=f_scale, x_scale=[0.01*x0[0], 0.1, 0.1])
        if (best is None) or (fit_res.cost < best.cost):
            best = fit_res

    v_init, ln_alpha, ln_beta = best.x

    # Warn if the fit did not converge
    if best.status <= 0:
        print("WARNING: the fit of the initial velocity with alpha and beta did not converge " \
            "({:s}).".format(str(best.message)))

    # Warn if the initial velocity is at its bound
    if (v_init < lower[0]*(1 + 1e-3)) or (v_init > upper[0]*(1 - 1e-3)):
        print("WARNING: the fitted initial velocity, {:.1f} m/s, is at its bound ({:.1f} to {:.1f} m/s, " \
            "from the median speed of the highest points, {:.1f} m/s): the fit does not constrain " \
            "it.".format(v_init, lower[0], upper[0], v_top))

    return v_init, np.exp(ln_alpha), np.exp(ln_beta), best



def alphaBetaFreeVinitErrors(fit_res, alpha, beta, v_init, sigma_v, ci):
    """ Estimate the Gauss-Newton errors of fitAlphaBeta(fit_v_init=True).

    Arguments:
        fit_res: [OptimizeResult] Least squares result of fitAlphaBetaFreeVinit().
        alpha: [float] Fitted alpha.
        beta: [float] Fitted beta.
        v_init: [float] Fitted initial velocity (m/s).
        sigma_v: [float] Velocity uncertainty (m/s).
        ci: [float] Confidence level of the Wald intervals (0-100).

    Return:
        [dict] The keys of _alphaBetaRobustErrors(), with 'v_init_std' and 'cov_full', the 3x3 covariance
            of (v_init, ln alpha, ln beta). 'cov_log' is its (ln alpha, ln beta) block, and 'cov_fit' the
            same with v_init held fixed.
    """

    # Return NaN uncertainties if the covariance cannot be computed
    nan_cov = np.full((2, 2), np.nan)
    result = {'alpha': alpha, 'beta': beta, 'v_init': v_init, 'v_init_std': np.nan, 'alpha_std_rel': np.nan, \
        'beta_std_rel': np.nan, 'alpha_std': np.nan, 'beta_std': np.nan, 'cov_fit': nan_cov, \
        'cov_log': nan_cov, 'cov_full': np.full((3, 3), np.nan), 'corr_log': np.nan, 'cov': nan_cov, \
        'corr': np.nan, 'alpha_ci_wald_lower': np.nan, 'alpha_ci_wald_upper': np.nan, \
        'beta_ci_wald_lower': np.nan, 'beta_ci_wald_upper': np.nan, 'ci': ci, 'sigma_v': sigma_v, \
        'sigma_v_init': np.nan}

    # Compute the covariance of (v_init, ln alpha, ln beta)
    dof = max(len(fit_res.fun) - len(fit_res.x), 1)
    s2 = 2.0*fit_res.cost/dof

    try:
        cov_full = s2*np.linalg.inv(fit_res.jac.T @ fit_res.jac)
    except np.linalg.LinAlgError:
        return result

    if np.any(np.diag(cov_full) < 0):
        print("WARNING: the Gauss-Newton covariance of (v_init, alpha, beta) has a negative variance on " \
            "its diagonal (a near-singular Jacobian). Returning NaN for the uncertainties.")
        return result

    cov_log = cov_full[1:, 1:]
    v_init_std = np.sqrt(cov_full[0, 0])

    # Compute the (ln alpha, ln beta) covariance with v_init held fixed
    jtj = fit_res.jac.T @ fit_res.jac
    try:
        cov_fit = s2*np.linalg.inv(jtj[1:, 1:])
    except np.linalg.LinAlgError:
        cov_fit = nan_cov

    # Compute the uncertainties of alpha and beta
    alpha_std_rel, beta_std_rel = np.sqrt(np.diag(cov_log))
    corr_log = cov_log[0, 1]/(alpha_std_rel*beta_std_rel)

    scale = np.array([alpha, beta])
    cov = np.outer(scale, scale)*cov_log
    alpha_std, beta_std = np.sqrt(np.diag(cov))
    corr = cov[0, 1]/(alpha_std*beta_std)

    # Compute the Wald confidence intervals
    z = scipy.special.ndtri(0.5 + ci/200.0)

    result.update({'v_init_std': v_init_std, 'alpha_std_rel': alpha_std_rel, 'beta_std_rel': beta_std_rel, \
        'alpha_std': alpha_std, 'beta_std': beta_std, 'cov_fit': cov_fit, 'cov_log': cov_log, \
        'cov_full': cov_full, 'corr_log': corr_log, 'cov': cov, 'corr': corr, \
        'alpha_ci_wald_lower': alpha*np.exp(-z*alpha_std_rel), \
        'alpha_ci_wald_upper': alpha*np.exp(z*alpha_std_rel), \
        'beta_ci_wald_lower': beta*np.exp(-z*beta_std_rel), \
        'beta_ci_wald_upper': beta*np.exp(z*beta_std_rel), 'sigma_v_init': v_init_std})

    return result



def alphaBetaResidualTrend(v_data, ht_data, v_init, alpha, beta, n_bands=4, ht_label=None):
    """ Compute the median velocity residual of an alpha-beta fit in height bands, and its z, to tell a
        meteor the model does not describe, e.g. one that fragments, from noise.

    Give it the observed velocities, as the residuals of smoothed ones trend even for a good fit. The
    point-to-point velocities are averages over the intervals between the points, which differ from the
    model velocity at the points for a strongly decelerating meteor, so the residuals of a good fit of such
    a meteor trend too.

    Arguments:
        v_data: [ndarray] Observed velocities (m/s).
        ht_data: [ndarray] Heights of the velocities, rescaled to the exponential atmosphere (m).
        v_init: [float] Fitted initial velocity (m/s).
        alpha: [float] Fitted alpha.
        beta: [float] Fitted beta.

    Keyword arguments:
        n_bands: [int] Number of height bands with about the same number of points. 4 by default.
        ht_label: [ndarray] Heights to give the ranges of the bands in, e.g. the real ones (m). None by
            default, for ht_data.

    Return:
        (bands, significant):
            - bands: [list] (ht_hi, ht_lo, n, med, z) of each band from the highest down, with the median
                residual med (m/s) and its z, med over 1.2533 times the robust scatter of all residuals
                over sqrt(n).
            - significant: [bool] True if any |z| exceeds RESIDUAL_TREND_SIGMA.
    """

    v_data, ht_data = np.asarray(v_data), np.asarray(ht_data)
    ht_label = ht_data if ht_label is None else np.asarray(ht_label)

    # Compute the residuals and their robust scatter
    res = alphaBetaVelocity(ht_data, alpha, beta, v_init) - v_data
    scatter = 1.4826*np.median(np.abs(res - np.median(res)))

    # Compute the median residual of each band, from the highest down
    order = np.argsort(ht_data)[::-1]
    bands = []
    for idx in np.array_split(order, n_bands):
        if len(idx) == 0:
            continue
        med = np.median(res[idx])
        z = med/(1.2533*scatter/np.sqrt(len(idx))) if scatter > 0 else np.nan
        bands.append((ht_label[idx].max(), ht_label[idx].min(), len(idx), med, z))

    significant = any(np.isfinite(b[4]) and (abs(b[4]) > RESIDUAL_TREND_SIGMA) for b in bands)

    return bands, significant



def alphaBetaEntryVelocityWithGravity(v_init, ht_first, ht_top=180e3):
    """ Compute the speed at ht_top of a meteoroid with the alpha-beta initial velocity v_init, with the
        gravity the alpha-beta model leaves out.

    The fit puts the speed gained by gravity above the observed part into v_init. By energy, the speed at
    ht_top is sqrt(v_init^2 - 2 g (ht_top - ht_first)), with g at the mid height.

    Arguments:
        v_init: [float] Alpha-beta initial velocity (m/s).
        ht_first: [float] Real height of the highest fitted point (m).

    Keyword arguments:
        ht_top: [float] Height of the speed (m). 180 km by default.

    Return:
        [float] Speed at ht_top (m/s).
    """

    # Gravity at the mid height
    g_mid = G0*(R_EARTH/(R_EARTH + (ht_top + ht_first)/2))**2

    return np.sqrt(v_init**2 - 2*g_mid*(ht_top - ht_first))



def fitAlphaBeta(v_data, ht_data, v_init=None, method='q4', sigma_v=None, loss='soft_l1',
        f_scale=2.0, estimate_errors=False, sigma_v_init=None, ci=95.0, verbose=False, fast=False,
        fit_v_init=False):
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
              - v_init=None (derived internally): this argument is IGNORED (a warning is printed
                if it was given a value) - v_init's uncertainty is instead estimated automatically
                as the standard error of the median of the same leading points used to derive it
                (see the Return section), which is the quantity that actually needs to be
                propagated, not the raw scatter of those points. IMPORTANT: this covers only the
                RANDOM error of that median - it is also systematically biased low by the
                deceleration trend across the leading window itself (a fixed offset, not
                something a variance term can correct for), which can meaningfully under-cover
                alpha_ci_wald_lower/upper - see the Statistical caveats below.
        ci: [float] Confidence level (0-100, e.g. 95, the default) for the approximate
            alpha_ci_wald_lower/upper and beta_ci_wald_lower/upper bounds below - only
            used if estimate_errors=True. Same convention as
            wmpl.Utils.Math.confidenceInterval()'s `ci`.
        verbose: [bool] If True, print the adopted v_init/alpha/beta (and, if
            estimate_errors=True, the error estimate too).
        fit_v_init: [bool] Fit the initial velocity together with alpha and beta on the velocity residuals
            (method='robust' only), starting from v_init if given, and bounded by V_INIT_FIT_BOUNDS times the
            median speed of the highest points. Give it the observed velocities, not smoothed ones. The
            errors then also hold 'v_init_std' and the 3x3 covariance 'cov_full' of (v_init, ln alpha,
            ln beta). False by default.
        fast: [bool] Only affects method='robust' (Q4's residual never inverts velocity, so it is
            unaffected). If True, use the experimental LUT-accelerated alphaBetaVelocityNormedLUT()
            instead of brentq everywhere this fit inverts velocity - see its docstring for the
            accuracy/speed tradeoff and the shared beta >~ 30 precision ceiling. Default False
            keeps today's exact behavior.

    Return:
        (v_init, alpha, beta):
            - v_init: [float] Input or derived initial velocity (m/s).
            - alpha: [float] Ballistic coefficient.
            - beta: [float] Mass loss.
        If estimate_errors=True, a 4th element `errors` (dict, from _alphaBetaRobustErrors()) is
        also returned:
            'alpha', 'beta', 'v_init': echoed back, so the dict is self-contained.
            'alpha_std_rel', 'beta_std_rel': [float] Relative (fractional) 1-sigma uncertainties
                of alpha/beta in (ln alpha, ln beta) space, combining the Gauss-Newton curvature
                of the fit itself with (if applicable) v_init's own uncertainty - see the caveats
                below.
            'alpha_std', 'beta_std': [float] The same 1-sigma uncertainties propagated to linear
                (alpha, beta) space via the delta method (alpha_std = alpha*alpha_std_rel, since
                d(alpha)/d(ln alpha) = alpha).
            'cov_fit': [ndarray] The Gauss-Newton contribution to 'cov_log' ALONE (i.e. v_init
                treated as exactly known, before any sigma_v_init correction is added) - lets a
                caller (e.g. plotAlphaBeta()) show how much of the total uncertainty in 'cov_log'
                comes from the fit itself vs. from propagating v_init's own uncertainty.
            'cov_log': [ndarray] Full 2x2 covariance matrix of (ln alpha, ln beta) - the space
                this model is actually fit in. 'cov_fit' plus the sigma_v_init rank-1 correction
                (see _alphaBetaRobustErrors()'s docstring).
            'cov': [ndarray] The same covariance propagated to linear (alpha, beta) space. Use
                'cov'/'cov_log' instead of the marginal std devs above to propagate into any OTHER
                derived quantity: alpha and beta are typically strongly correlated (see 'corr'/
                'corr_log'), so propagating the marginal std devs independently would be wrong.
            'corr_log', 'corr': [float] Correlation coefficient of (ln alpha, ln beta) / of
                (alpha, beta) - numerically identical (correlation is invariant under the
                positive-diagonal rescaling relating the two spaces); both are kept for
                convenience.
            'alpha_ci_wald_lower', 'alpha_ci_wald_upper',
            'beta_ci_wald_lower', 'beta_ci_wald_upper': [float] APPROXIMATE confidence
                intervals based on the local Gaussian (Wald-type) approximation in log-space, at
                the `ci` level - i.e. alpha/beta are treated as log-normal, so these bounds are
                asymmetric around alpha/beta and, unlike alpha +/- z*alpha_std, can never go
                negative. The "_wald" in the name is deliberate: this is NOT the same kind of
                object as wmpl.Utils.Math.confidenceInterval()'s output elsewhere in this library
                (e.g. Trajectory.py, REBOUND.py) - those are empirical percentiles of an actual
                Monte Carlo ensemble, whereas there is no ensemble here, only a local quadratic
                approximation of the cost function around the optimum. See profileAlphaBeta()
                for a profile-based alternative when that approximation is in doubt (see the
                Statistical caveats below).
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
        - alpha_ci_wald_lower/upper and beta_ci_wald_lower/upper assume normality in LOG
          space, which is only ever as good as the two contributions to cov_log above - they are
          a convenience view of the same Gaussian approximation, not an independently more
          rigorous one.
        - If the fitted (alpha, beta) landed within 1% (log-space) of an ALPHA_BETA_BOUNDS edge,
          or if that produces a near-singular Jacobian (a negative variance on the covariance
          diagonal - impossible for a real covariance), a WARNING is printed and every error field
          is NaN (alpha/beta themselves are still returned). Both mean the same thing: the fit is
          likely poorly constrained by this data, and the local curvature approximation isn't
          meaningful around a boundary the optimizer was clipped against rather than converged to.
        - With v_init=None (derived internally), alpha_ci_wald_lower/upper can under-cover the
          true alpha/beta by a lot more than the other approximations above: the leading window
          used to derive v_init is still decelerating, so its median is not just noisy but
          systematically biased LOW relative to the true v_init (-11 m/s, deterministic, on this
          module's own synthetic test geometry - a fixed offset, not something sigma_v_init's
          variance term can correct for). Measured on that same geometry (300-realization Monte
          Carlo): at the noise level used elsewhere in this test suite, the nominal 95%
          alpha_ci_wald_lower/upper actually covers the true alpha/beta only ~65-71% of the
          time with a derived v_init - even though alpha_std_rel/beta_std_rel themselves stay
          accurate - recovering to ~90% at 3x that noise as the fixed bias shrinks relative to the
          growing scatter. With v_init supplied externally instead (no derivation, hence no
          bias), coverage is a nominal ~96%. The bias is in the pre-existing v_init derivation,
          not introduced by estimate_errors - but the confidence interval is what turns a
          previously-inconsequential offset into a miscalibrated claim. Prefer an external v_init
          (+ sigma_v_init if it has its own uncertainty) over the internally-derived one whenever
          interval coverage matters, not just the point estimate.
        - alpha_ci_wald_lower/upper and beta_ci_wald_lower/upper are derived from a LOCAL
          quadratic approximation (cov_log) of the cost function around the optimum - this is
          exactly what covariances/correlations/std devs are for, and remains reliable for that.
          But when a parameter is only weakly constrained, the reported interval can extend far
          enough from the optimum that the cost function is no longer well described by that
          local quadratic shape - and in that regime, the WALD interval can substantially
          OVERESTIMATE the interval the actual cost surface supports, particularly in the
          direction the real cost rises fastest. Measured systematically (8 synthetic scenarios
          spanning well- to weakly-constrained): the discrepancy between the Wald interval and the
          properly-profiled interval (see profileAlphaBeta()) is negligible
          (~1x) whenever the relevant *_std_rel is small, and grows sharply (as much as ~140x on
          the upper bound, occasionally unbounded) once it is large - NOT a one-off pathological
          case, but a systematic property of this regime. Critically, this affects beta far more
          readily than alpha at comparable *_std_rel magnitudes (e.g. alpha_std_rel=88.6% gave
          only a 1.8x discrepancy, while beta_std_rel=111.4% - barely bigger - gave 5.0x): beta
          enters the model inside Ei(beta*v^2), directly shaping the curve, while alpha enters as
          a near-linear log-scale shift (ln(2*alpha)) - a structurally more nonlinear role for
          beta that a local quadratic expansion is expected to describe far less faithfully. This
          limitation is specific to the CONFIDENCE INTERVAL, not to cov_log/corr_log/alpha_std_rel/
          beta_std_rel themselves (those keep describing the local curvature correctly regardless
          - only the extrapolation of that curvature into a "what values are plausible" statement
          is what breaks down). When beta_std_rel (or alpha_std_rel) is large, prefer
          profileAlphaBeta() over alpha_ci_wald_lower/upper - beta_ci_wald_lower/upper for
          inference about which parameter values are actually compatible with the data.
    """

    method = method.lower()

    # Fit the initial velocity only with the robust fit
    if fit_v_init and method != 'robust':
        raise ValueError("fit_v_init=True is only supported for method='robust', which fits the velocity " \
            "residuals the initial velocity enters.")

    if estimate_errors and method != 'robust':
        raise ValueError("estimate_errors=True is only supported for method='robust' - Q4's "
            "L1/Nelder-Mead fit has no natural Jacobian to build an analytic covariance from.")

    if estimate_errors and not (0 < ci < 100):
        raise ValueError("ci must be strictly between 0 and 100, got {!r}. Out-of-range values "
            "degrade silently: ci=100 gives z=inf (0/inf bounds), ci>100 gives z=NaN, ci<0 "
            "inverts which bound is 'lower' vs 'upper'.".format(ci))

    v_init_given = v_init is not None

    # Compute the initial velocity, if it wasn't given already
    if not v_init_given:

        max_index = int(0.2*len(v_data))
        if max_index < 10:
            max_index = 10

        # Use the highest points when the initial velocity is fitted
        if fit_v_init:
            v_init = np.median(v_data[np.argsort(ht_data)[::-1][:max_index]])
        else:
            v_init = np.median(v_data[:max_index])

    # Normalize the velocity and height
    v_normed = v_data/v_init
    ht_normed = ht_data/HT_NORM_CONST

    if method == 'q4':
        alpha, beta = minimizeAlphaBeta(v_normed, ht_normed, fast=fast)

    elif method == 'robust':
        alpha0, beta0 = minimizeAlphaBeta(v_normed, ht_normed, fast=fast)

        if sigma_v is None:
            sigma_v = _estimateSigmaV(alpha0, beta0, v_normed, ht_normed, v_init, fast=fast)

        if fit_v_init:

            # Fit at the starting initial velocity, then with it free
            alpha1, beta1 = minimizeAlphaBetaRobust(v_normed, ht_normed, sigma_v/v_init, loss=loss, \
                f_scale=f_scale, x0_alpha_beta=(alpha0, beta0), fast=fast)
            v_init, alpha, beta, fit_res = fitAlphaBetaFreeVinit(v_data, ht_data, v_init, alpha1, beta1, \
                sigma_v, loss, f_scale, fast=fast)

        elif estimate_errors:
            alpha, beta, fit_res = minimizeAlphaBetaRobust(v_normed, ht_normed, sigma_v/v_init, \
                loss=loss, f_scale=f_scale, x0_alpha_beta=(alpha0, beta0), return_fit_result=True, \
                fast=fast)
        else:
            alpha, beta = minimizeAlphaBetaRobust(v_normed, ht_normed, sigma_v/v_init, loss=loss, \
                f_scale=f_scale, x0_alpha_beta=(alpha0, beta0), fast=fast)

    else:
        raise ValueError("method must be 'q4' or 'robust', got '{:s}'.".format(method))

    if estimate_errors and fit_v_init:
        errors = alphaBetaFreeVinitErrors(fit_res, alpha, beta, v_init, sigma_v, ci)

    elif estimate_errors:
        errors = _alphaBetaRobustErrors(fit_res, alpha, beta, v_data, ht_data, v_init, \
            v_init_given, sigma_v, sigma_v_init, loss, f_scale, ci, fast=fast)

    if verbose:
        print()
        print("--- fitAlphaBeta ({:s}) ---".format(method))
        if estimate_errors and fit_v_init:
            print("v_init               = {:.3f} +/- {:.3f} m/s (fitted)".format(v_init, \
                errors['v_init_std']))
        else:
            print("v_init               = {:.3f} m/s{:s}".format(v_init, " (fitted)" if fit_v_init else ""))

        if estimate_errors:
            print("alpha                = {:.6f} (+/-{:.1%})  [{:.3g}% Wald CI: {:.6f} - "
                "{:.6f}]".format(alpha, errors['alpha_std_rel'], ci,
                errors['alpha_ci_wald_lower'], errors['alpha_ci_wald_upper']))
            print("beta                 = {:.6f} (+/-{:.1%})  [{:.3g}% Wald CI: {:.6f} - "
                "{:.6f}]".format(beta, errors['beta_std_rel'], ci,
                errors['beta_ci_wald_lower'], errors['beta_ci_wald_upper']))
            print("corr(alpha, beta)    = {:.3f}".format(errors['corr']))
            print("sigma_v              = {:.3f} m/s".format(errors['sigma_v']))
            print("sigma_v_init         = {:.3f} m/s".format(errors['sigma_v_init']))
        else:
            print("alpha                = {:.6f}".format(alpha))
            print("beta                 = {:.6f}".format(beta))

    if estimate_errors:
        return v_init, alpha, beta, errors

    return v_init, alpha, beta


class AlphaBetaPath(object):
    def __init__(self, length_data, ht_data, margin=0.3, n_grid=800):
        """ Path of a meteor for integrating the alpha-beta velocity over time: the height along the path,
            on a grid of distances which extends beyond the observed points on both sides, so that the
            distance at a given time is defined also for a model which is faster or slower than the data.

        Arguments:
            length_data: [ndarray] Distance along the path of every point, from a common reference (m),
                e.g. ObservedPoints.state_vect_dist.
            ht_data: [ndarray] Height of every point, rescaled to the exponential atmosphere of the
                alpha-beta model (m), see rescaleHeightToExponentialAtmosphere().

        Keyword arguments:
            margin: [float] Extension of the grid beyond the first and the last point, relative to the
                length of the observed path. 0.3 by default.
            n_grid: [int] Number of grid points between the first and the last point. 800 by default.
        """

        # Sort the points along the path and make the height decrease monotonically, as the rescaled
        #   heights of points at almost the same distance can be out of order
        sort_indices = np.argsort(length_data)
        length_sorted = length_data[sort_indices]
        ht_sorted = np.minimum.accumulate(ht_data[sort_indices])

        # Extrapolate the height linearly beyond the first and the last point, from the slope of the
        #   points at the ends, but not above the first or below the last height
        span = length_sorted[-1] - length_sorted[0]
        n_end = max(5, len(length_sorted)//10)
        top_coeffs = np.polyfit(length_sorted[:n_end], ht_sorted[:n_end], 1)
        bottom_coeffs = np.polyfit(length_sorted[-n_end:], ht_sorted[-n_end:], 1)

        length_top = np.linspace(length_sorted[0] - margin*span, length_sorted[0], 50, endpoint=False)
        length_mid = np.linspace(length_sorted[0], length_sorted[-1], n_grid)
        length_bottom = np.linspace(length_sorted[-1], length_sorted[-1] + margin*span, 51)[1:]

        ht_top = np.maximum(np.polyval(top_coeffs, length_top), ht_sorted[0])
        ht_mid = np.interp(length_mid, length_sorted, ht_sorted)
        ht_bottom = np.minimum(np.polyval(bottom_coeffs, length_bottom), ht_sorted[-1])

        # Store the grid of distances and the normalized heights, which must not be negative
        self.length_grid = np.concatenate([length_top, length_mid, length_bottom])
        ht_grid = np.minimum.accumulate(np.concatenate([ht_top, ht_mid, ht_bottom]))
        self.ht_normed_grid = np.maximum(ht_grid, 0.0)/HT_NORM_CONST



    def times(self, alpha, beta, v_init, fast=True):
        """ Time at which the alpha-beta model reaches every distance of the grid, from the first one.

        Arguments:
            alpha: [float] Ballistic coefficient.
            beta: [float] Mass loss parameter.
            v_init: [float] Initial velocity (m/s).

        Keyword arguments:
            fast: [bool] Passed to alphaBetaVelocityNormed(). True by default.

        Return:
            time_grid: [ndarray] Time at every distance of the grid (s).
        """

        # Velocity along the path, at least 1 m/s, as the meteor never stops in the model
        vel_grid = v_init*alphaBetaVelocityNormed(self.ht_normed_grid, alpha, beta, fast=fast)
        vel_grid = np.maximum(vel_grid, 1.0)

        # Integrate dt = dl/v along the path with the trapezoidal rule
        dt = 0.5*(1.0/vel_grid[1:] + 1.0/vel_grid[:-1])*np.diff(self.length_grid)

        return np.concatenate([[0.0], np.cumsum(dt)])



    def lengths(self, time_data, alpha, beta, v_init, t0=None, length_data=None, fast=True):
        """ Distance along the path of the alpha-beta model at the given times.

        Arguments:
            time_data: [ndarray] Times (s).
            alpha: [float] Ballistic coefficient.
            beta: [float] Mass loss parameter.
            v_init: [float] Initial velocity (m/s).

        Keyword arguments:
            t0: [float] Time at the first distance of the grid (s). It is a shift of the whole model in
                time, so it is found directly from the observed distances (length_data) if not given, as
                the mean difference between the observed times and the model times at the observed
                distances, weighted by the squared model velocity. This minimizes the squared distance
                residuals to first order, and changes smoothly with the parameters of the model, which the
                fit and its profiles need. None by default.
            length_data: [ndarray] Observed distances at the times time_data (m), used if t0 is None. None
                by default.
            fast: [bool] Passed to alphaBetaVelocityNormed(). True by default.

        Return:
            (length_model, t0):
                - length_model: [ndarray] Model distance at every time (m).
                - t0: [float] Time at the first distance of the grid (s).
        """

        time_grid = self.times(alpha, beta, v_init, fast=fast)

        # Find the time of the first grid point from the observed distances, weighting the time differences
        #   by the squared velocity as a time difference dt gives a distance residual of v*dt
        if t0 is None:
            vel_weights = np.interp(length_data, self.length_grid, \
                np.gradient(self.length_grid, time_grid))**2
            t0 = np.sum(vel_weights*(time_data - np.interp(length_data, self.length_grid, time_grid))) \
                /np.sum(vel_weights)

        time_rel = time_data - t0
        length_model = np.interp(time_rel, time_grid, self.length_grid)

        # Beyond the ends of the grid, continue with the velocity at the ends
        vel_top = (self.length_grid[1] - self.length_grid[0])/(time_grid[1] - time_grid[0])
        vel_bottom = (self.length_grid[-1] - self.length_grid[-2])/max(time_grid[-1] - time_grid[-2], 1e-12)
        length_model = np.where(time_rel < time_grid[0], \
            self.length_grid[0] + vel_top*(time_rel - time_grid[0]), length_model)
        length_model = np.where(time_rel > time_grid[-1], \
            self.length_grid[-1] + vel_bottom*(time_rel - time_grid[-1]), length_model)

        return length_model, t0



    def velocityDeceleration(self, length, alpha, beta, v_init, fast=True):
        """ Velocity and deceleration of the alpha-beta model at the given distances along the path, e.g.
            to compute the dynamic mass at the end of the trajectory.

        Arguments:
            length: [float or ndarray] Distance along the path (m), on the scale of the observed distances.
            alpha: [float] Ballistic coefficient.
            beta: [float] Mass loss parameter.
            v_init: [float] Initial velocity (m/s).

        Keyword arguments:
            fast: [bool] Passed to alphaBetaVelocityNormed(). True by default.

        Return:
            (vel, decel):
                - vel: [float or ndarray] Velocity (m/s).
                - decel: [float or ndarray] Deceleration, positive when the meteor slows down (m/s^2).
        """

        # Velocity on the grid
        vel_grid = v_init*alphaBetaVelocityNormed(self.ht_normed_grid, alpha, beta, fast=fast)

        # The deceleration is -dv/dt = -v*dv/dl along the path
        decel_grid = -vel_grid*np.gradient(vel_grid, self.length_grid)

        vel = np.interp(length, self.length_grid, vel_grid)
        decel = np.interp(length, self.length_grid, decel_grid)

        return vel, decel


def fitAlphaBetaLag(time_data, length_data, ht_data, v_init, x0_alpha_beta=None, fit_v_init=True, \
    v_init_range=0.1, n_starts=5, estimate_errors=False, ci=95.0, fast=True):
    """ Fit the alpha-beta model directly to the distance along the path in time, i.e. to the lag, instead
        of to velocities.

        The alpha-beta velocity is integrated along the path (AlphaBetaPath) to give the distance at every
        observed time, and alpha, beta and the initial velocity are fitted by least squares to the
        observed distances. The residuals are those of the lag, as the lag is the distance minus the
        initial velocity times the time, so the deceleration is fitted from the data directly, without
        a smoothing model of the lag (lagFitVelocity()) or noisy point-to-point velocities. The initial
        velocity is fitted by default, as a slightly wrong one would leave a linear trend in the lag
        which the model could only follow by decelerating. The time offset of the model is found
        directly at every step (AlphaBetaPath.lengths()).

        The fit starts from the best points of a grid in the log space of ALPHA_BETA_BOUNDS and from the
        given initial guesses, and keeps the best result, as the fit has local minima.

    Arguments:
        time_data: [ndarray] Time of every point (s), on the same time scale for all stations.
        length_data: [ndarray] Distance along the path of every point, from a common reference (m),
            e.g. ObservedPoints.state_vect_dist. The distance gained from gravity along the path should be
            taken out (Trajectory.gravityLengthAlongPath()), as the model has no gravity.
        ht_data: [ndarray] Height of every point, rescaled to the exponential atmosphere of the alpha-beta
            model (m), see rescaleHeightToExponentialAtmosphere().
        v_init: [float] Initial velocity (m/s). The starting point of the fit if fit_v_init is True.

    Keyword arguments:
        x0_alpha_beta: [list] Additional (alpha, beta) starting points, e.g. the results of fitAlphaBeta(),
            or (alpha, beta, v_init) to also start from another initial velocity. None by default.
        fit_v_init: [bool] Fit the initial velocity. True by default.
        v_init_range: [float] Range of the fitted initial velocity, relative to v_init. 0.1 by default.
        n_starts: [int] Number of the best grid points from which the fit is started. 5 by default.
        estimate_errors: [bool] Also return the uncertainties of alpha and beta, from the covariance of
            the least squares fit, in the format of fitAlphaBeta(estimate_errors=True), so they can be
            passed to alphaBetaMasses() and plotAlphaBetaSurvivalDiagram(), with 'v_init_std' and the 3x3
            covariance 'cov_full' of (v_init, ln alpha, ln beta) if the initial velocity is fitted. NaN if
            alpha or beta end at a bound of the fit. False by default.
        ci: [float] Confidence level of the Wald intervals in the uncertainties (percent). 95 by default.
        fast: [bool] Passed to alphaBetaVelocityNormed(). True by default, as the model is integrated
            many times in the fit. The grid search always uses the fast inversion.

    Return:
        (v_init, alpha, beta, length_residuals), or (v_init, alpha, beta, length_residuals, errors) if
            estimate_errors is True:
            - v_init: [float] Fitted (or the given) initial velocity (m/s).
            - alpha: [float] Ballistic coefficient.
            - beta: [float] Mass loss parameter.
            - length_residuals: [ndarray] Observed minus model distance at every point (m), equal to the
                lag residuals.
            - errors: [dict] See estimate_errors.
    """

    path = AlphaBetaPath(length_data, ht_data)

    def _lengthResiduals(x, fast):
        """ Residuals of the distances (m) for the parameters (ln alpha, ln beta[, v_init]). """

        alpha, beta = np.exp(x[:2])
        v_init_fit = x[2] if fit_v_init else v_init

        length_model, _ = path.lengths(time_data, alpha, beta, v_init_fit, length_data=length_data, \
            fast=fast)

        return length_data - length_model


    # Bounds of the fitted parameters
    log_bounds = _logAlphaBetaBounds()
    lower_bounds = list(log_bounds[0])
    upper_bounds = list(log_bounds[1])
    if fit_v_init:
        lower_bounds.append((1.0 - v_init_range)*v_init)
        upper_bounds.append((1.0 + v_init_range)*v_init)

    # Compute the residuals on a grid of alpha and beta with the given initial velocity
    grid = []
    for alpha in np.geomspace(*ALPHA_BETA_BOUNDS[0], 13):
        for beta in np.geomspace(*ALPHA_BETA_BOUNDS[1], 13):

            x = [np.log(alpha), np.log(beta)] + ([v_init] if fit_v_init else [])
            grid.append((np.sum(_lengthResiduals(x, True)**2), alpha, beta))

    # Start from the best grid points and the given starting points
    grid.sort(key=lambda entry: entry[0])
    starts = [entry[1:] for entry in grid[:n_starts]]
    if x0_alpha_beta is not None:
        starts += [tuple(x0) for x0 in x0_alpha_beta]

    # Run the fit from every starting point and keep the best one
    best_res = None
    for start in starts:

        # Keep the starting point strictly inside the bounds
        x0 = list(np.log(np.clip(start[:2], np.exp(log_bounds[0] + 1e-6), np.exp(log_bounds[1] - 1e-6))))
        if fit_v_init:
            x0.append(np.clip(start[2], lower_bounds[2], upper_bounds[2]) if len(start) > 2 else v_init)

        res = scipy.optimize.least_squares(_lengthResiduals, x0, args=(fast,), \
            bounds=(lower_bounds, upper_bounds), x_scale='jac')

        if (best_res is None) or (res.cost < best_res.cost):
            best_res = res

    alpha, beta = np.exp(best_res.x[:2])
    v_init_fit = best_res.x[2] if fit_v_init else v_init
    length_residuals = best_res.fun

    if not estimate_errors:
        return v_init_fit, alpha, beta, length_residuals


    ### Estimate the uncertainties from the covariance of the least squares fit ###

    n_params = len(best_res.x)
    errors = {'alpha': alpha, 'beta': beta, 'v_init': v_init_fit, 'ci': ci, 'v_init_std': np.nan, \
        'cov_log': np.full((2, 2), np.nan), 'cov': np.full((2, 2), np.nan), \
        'cov_full': np.full((3, 3), np.nan)}
    for key in ['alpha_std_rel', 'beta_std_rel', 'alpha_std', 'beta_std', 'corr_log', 'corr', \
            'alpha_ci_wald_lower', 'alpha_ci_wald_upper', 'beta_ci_wald_lower', 'beta_ci_wald_upper']:
        errors[key] = np.nan

    # The Gaussian approximation is not valid at a bound, nor without more points than parameters
    if alphaBetaBoundWarnings(alpha, beta) or (len(length_residuals) <= n_params):
        return v_init_fit, alpha, beta, length_residuals, errors

    # Covariance of the parameters, scaled by the variance of the residuals
    jac = best_res.jac
    residual_var = np.sum(length_residuals**2)/(len(length_residuals) - n_params)
    try:
        cov_params = residual_var*np.linalg.inv(jac.T.dot(jac))
    except np.linalg.LinAlgError:
        return v_init_fit, alpha, beta, length_residuals, errors

    # The block of (ln alpha, ln beta), which includes the uncertainty of the fitted initial velocity
    cov_log = cov_params[:2, :2]
    if (not np.all(np.isfinite(cov_log))) or np.any(np.diag(cov_log) <= 0):
        return v_init_fit, alpha, beta, length_residuals, errors

    alpha_std_rel, beta_std_rel = np.sqrt(np.diag(cov_log))

    # Linear (alpha, beta) space, via the delta method
    scale = np.array([alpha, beta])
    cov = np.outer(scale, scale)*cov_log

    # Log-normal Wald intervals
    z = scipy.special.ndtri(0.5 + ci/200.0)

    errors.update({
        'alpha_std_rel': alpha_std_rel,
        'beta_std_rel': beta_std_rel,
        'alpha_std': alpha*alpha_std_rel,
        'beta_std': beta*beta_std_rel,
        'cov_log': cov_log,
        'corr_log': cov_log[0, 1]/(alpha_std_rel*beta_std_rel),
        'cov': cov,
        'corr': cov[0, 1]/np.sqrt(cov[0, 0]*cov[1, 1]),
        'alpha_ci_wald_lower': alpha*np.exp(-z*alpha_std_rel),
        'alpha_ci_wald_upper': alpha*np.exp(z*alpha_std_rel),
        'beta_ci_wald_lower': beta*np.exp(-z*beta_std_rel),
        'beta_ci_wald_upper': beta*np.exp(z*beta_std_rel),
        'v_init_std': np.sqrt(cov_params[2, 2]) if fit_v_init else 0.0
        })

    # The full covariance of (v_init, ln alpha, ln beta), as fitAlphaBeta(fit_v_init=True) gives it
    if fit_v_init:
        order = [2, 0, 1]
        errors['cov_full'] = cov_params[np.ix_(order, order)]

    return v_init_fit, alpha, beta, length_residuals, errors



def _lagFitFixed(path, time_data, length_data, x_start, fixed, lower_bounds, upper_bounds, v_init, \
    fit_v_init, fast=True):
    """ Fit the lag with some of the parameters (ln alpha, ln beta[, v_init]) of fitAlphaBetaLag() held
        fixed, for the profiles in profileAlphaBetaLag().

    Arguments:
        path: [AlphaBetaPath object] Path of the meteor.
        time_data: [ndarray] Time of every point (s).
        length_data: [ndarray] Distance along the path of every point (m).
        x_start: [list] Starting values of all parameters (ln alpha, ln beta[, v_init]).
        fixed: [dict] Index of every fixed parameter and its value.
        lower_bounds: [list] Lower bounds of all parameters.
        upper_bounds: [list] Upper bounds of all parameters.
        v_init: [float] Initial velocity (m/s), used if it is not fitted.
        fit_v_init: [bool] The initial velocity is the third parameter.

    Keyword arguments:
        fast: [bool] Passed to AlphaBetaPath.lengths(). True by default.

    Return:
        (cost, x):
            - cost: [float] Sum of the squared lag residuals (m^2).
            - x: [ndarray] All parameters at the best fit.
    """

    x_all = np.array(x_start, dtype=np.float64)
    for i, value in fixed.items():
        x_all[i] = value

    free_indices = [i for i in range(len(x_all)) if i not in fixed]

    def _residuals(x_free):
        """ Lag residuals for the free parameters. """

        x = x_all.copy()
        x[free_indices] = x_free

        alpha, beta = np.exp(x[:2])
        v_init_fit = x[2] if fit_v_init else v_init

        length_model, _ = path.lengths(time_data, alpha, beta, v_init_fit, length_data=length_data, \
            fast=fast)

        return length_data - length_model


    # Without free parameters only evaluate the cost
    if not free_indices:
        residuals = _residuals(np.array([]))
        return np.sum(residuals**2), x_all

    # Keep the starting point strictly inside the bounds
    lower = np.array(lower_bounds)[free_indices]
    upper = np.array(upper_bounds)[free_indices]
    x0 = np.clip(x_all[free_indices], lower + 1e-6*(upper - lower), upper - 1e-6*(upper - lower))

    res = scipy.optimize.least_squares(_residuals, x0, bounds=(lower, upper), x_scale='jac')

    x_all[free_indices] = res.x

    return 2*res.cost, x_all



def profileAlphaBetaLag(time_data, length_data, ht_data, v_init, alpha, beta, fit_v_init=True, \
    v_init_start=None, v_init_range=0.1, ci=95.0, step_min=0.01, step_max=0.5, n_starts=5, max_points=200, \
    n_grid=40, fast=True):
    """ Profile likelihood intervals of alpha and beta of the fit to the lag (fitAlphaBetaLag()), and the
        likelihood on a grid of alpha and beta.

        The Wald intervals of fitAlphaBetaLag(estimate_errors=True) assume that the cost is a parabola
        around the minimum, which does not hold when a parameter is weakly constrained, e.g. a beta which
        the lag only bounds from above. Here one parameter is fixed on a grid and the others (the other
        one of alpha and beta, and the initial velocity if it is fitted) are fitted again at every point.
        The cost rise, scaled by the variance of the residuals at the best fit, is compared to the
        chi-squared threshold. The residuals are assumed to be independent, so the intervals are too
        narrow if they are correlated, e.g. by a systematic offset of a station.

        The 2D grid covers the region of the profiles within the 3 sigma level of two parameters, with only
        the initial velocity fitted at every point.

    Arguments:
        time_data: [ndarray] Time of every point (s).
        length_data: [ndarray] Distance along the path of every point (m), as given to fitAlphaBetaLag().
        ht_data: [ndarray] Height of every point, rescaled to the exponential atmosphere (m).
        v_init: [float] Initial velocity of the best fit (m/s).
        alpha: [float] Alpha of the best fit.
        beta: [float] Beta of the best fit.

    Keyword arguments:
        fit_v_init: [bool] The initial velocity was fitted. True by default.
        v_init_start: [float] Initial velocity given to fitAlphaBetaLag() as the starting point (m/s), on
            which the bounds of the fitted initial velocity are centered, so the profiles stay within the
            bounds of the fit. v_init by default.
        v_init_range: [float] Range of the fitted initial velocity, relative to v_init_start. 0.1 by
            default.
        ci: [float] Confidence level of the intervals (percent). 95 by default.
        step_min: [float] First step of the profiles in ln alpha and ln beta. The step grows where the
            profile is flat, and shrinks where it is steep or when the first steps already cross the 3 sigma
            level. 0.01 by default.
        step_max: [float] Largest step of the profiles in ln alpha and ln beta. 0.5 by default.
        n_starts: [int] Number of additional starting values of the other parameter at every point of the
            profiles, spread over its bounds. 5 by default.
        max_points: [int] Largest number of points of each side of a profile. A side which reaches it without
            crossing the threshold is taken as not constrained. 200 by default.
        n_grid: [int] Number of grid points along each axis of the 2D grid. 40 by default. 0 skips the
            grid.
        fast: [bool] Passed to AlphaBetaPath.lengths(). True by default.

    Return:
        profile: [dict] With the keys:
            - ci: [float] Confidence level (percent).
            - threshold: [float] Chi-squared rise at the confidence level, for one parameter.
            - sigma2: [float] Variance of the lag residuals at the best fit (m^2).
            - alpha_lower, alpha_upper, beta_lower, beta_upper: [float] Profile intervals. A limit at a
                bound of the fit (ALPHA_BETA_BOUNDS) means that the data do not constrain that side.
            - alpha_lower_at_bound, alpha_upper_at_bound, beta_lower_at_bound, beta_upper_at_bound:
                [bool] The interval reaches the bound of the fit.
            - alpha_profile, beta_profile: [tuple of ndarrays] Values of the parameter and the
                chi-squared rise at them, sorted by the value.
            - alpha_profile_models, beta_profile_models: [ndarray] Alpha, beta and the initial velocity at
                every point of the profiles, one row per point, in the same order.
            - alpha_edge_models, beta_edge_models: [ndarray] Alpha, beta, the initial velocity and the
                chi-squared rise at the points of the profiles next to the crossings of the threshold.
            - grid_alpha, grid_beta: [ndarray] Values of the 2D grid along each axis.
            - grid_delta_chi2: [ndarray] Chi-squared rise on the 2D grid, rows along beta.
            - grid_v_init: [ndarray] Fitted initial velocity on the 2D grid (m/s), rows along beta.
            - corr_log: [float] Correlation of ln alpha and ln beta, weighted by the likelihood on the 2D
                grid. Biased if the likelihood is cut off by a bound of the fit.
            - min_delta_chi2: [float] Lowest chi-squared rise found on the profiles and the grid. Negative if
                they found a better fit than the given one, which then ended in a local minimum.
            - min_model: [tuple] Alpha, beta and the initial velocity at that point.
    """

    path = AlphaBetaPath(length_data, ht_data)

    # Bounds of the parameters, as in fitAlphaBetaLag()
    log_bounds = _logAlphaBetaBounds()
    lower_bounds = list(log_bounds[0])
    upper_bounds = list(log_bounds[1])
    x_best = [np.log(alpha), np.log(beta)]
    if fit_v_init:
        if v_init_start is None:
            v_init_start = v_init
        lower_bounds.append((1.0 - v_init_range)*v_init_start)
        upper_bounds.append((1.0 + v_init_range)*v_init_start)
        x_best.append(v_init)

    # Cost and the variance of the residuals at the best fit
    cost_best, _ = _lagFitFixed(path, time_data, length_data, x_best, {i: x for i, x in enumerate(x_best)}, \
        lower_bounds, upper_bounds, v_init, fit_v_init, fast=fast)
    sigma2 = cost_best/max(len(time_data) - len(x_best), 1)

    threshold = scipy.stats.chi2.ppf(ci/100.0, 1)

    # Walk far enough to also show the 3 sigma level of two parameters
    walk_threshold = max(threshold, scipy.stats.chi2.ppf(0.9973, 2))

    profile = {'ci': ci, 'threshold': threshold, 'sigma2': sigma2}

    for idx, name in enumerate(['alpha', 'beta']):

        values = [x_best[idx]]
        delta_chi2 = [0.0]
        x_values = [list(x_best)]

        # Starting values of the other parameter, spread over its bounds
        other_starts = np.linspace(lower_bounds[1 - idx], upper_bounds[1 - idx], n_starts + 2)[1:-1]

        # Walk from the best fit to each bound, starting every fit from the previous one
        for direction in [-1, 1]:

            x_prev = list(x_best)
            log_value = x_best[idx]
            bound = lower_bounds[idx] if direction < 0 else upper_bounds[idx]
            step = step_min
            step_floor = step_min
            delta_prev = 0.0
            n_points = 0

            while (direction*(bound - log_value) > 0) and (n_points < max_points):

                log_value_new = log_value + direction*step
                if direction*(log_value_new - bound) > 0:
                    log_value_new = bound

                # Start from the previous point, from values of the other parameter across its bounds, as the
                #   profile can follow a flat valley the previous point does not lead into, and from the
                #   initial velocity of the best fit
                x_starts = [list(x_prev)]
                for log_other in other_starts:
                    x_starts.append(list(x_prev))
                    x_starts[-1][1 - idx] = log_other
                if fit_v_init:
                    x_starts.append(list(x_prev[:2]) + [x_best[2]])

                cost, x_new = np.inf, None
                for x_start in x_starts:

                    cost_start, x_fit = _lagFitFixed(path, time_data, length_data, x_start, \
                        {idx: log_value_new}, lower_bounds, upper_bounds, v_init, fit_v_init, fast=fast)

                    if cost_start < cost:
                        cost, x_new = cost_start, x_fit

                delta_new = (cost - cost_best)/sigma2

                # If the first steps already cross the 3 sigma level, the interval is narrower than the
                #   step, so take smaller steps
                if (delta_new > walk_threshold) and (n_points < 4) and (step > 1e-6):
                    step /= 4.0
                    step_floor = min(step_floor, step)
                    continue

                log_value = log_value_new
                x_prev = list(x_new)
                n_points += 1

                values.append(log_value)
                delta_chi2.append(delta_new)
                x_values.append(x_prev)

                if delta_new > walk_threshold:
                    break

                # Take larger steps where the profile is flat and smaller ones where it is steep
                rise = abs(delta_new - delta_prev)
                delta_prev = delta_new
                if rise < 0.2:
                    step = min(1.5*step, step_max)
                elif rise > 1.0:
                    step = max(0.5*step, step_floor)

        # Sort the profile by the value
        order = np.argsort(values)
        values = np.array(values)[order]
        delta_chi2 = np.array(delta_chi2)[order]
        x_values = [x_values[i] for i in order]

        # Find where the profile crosses the threshold on each side of the minimum
        i_min = np.argmin(delta_chi2)
        limits = []
        at_bound = []
        edge_models = []
        for side in [-1, 1]:

            indices = np.arange(i_min, -1, -1) if side < 0 else np.arange(i_min, len(values))
            crossing = None
            for j_prev, j in zip(indices[:-1], indices[1:]):
                if delta_chi2[j] > threshold:

                    # Narrow down the crossing by bisection, then interpolate linearly in the log space
                    inner, outer = values[j_prev], values[j]
                    delta_inner, delta_outer = delta_chi2[j_prev], delta_chi2[j]
                    x_inner = x_values[j_prev]
                    for _ in range(8):

                        middle = 0.5*(inner + outer)
                        cost, x_middle = _lagFitFixed(path, time_data, length_data, x_inner, \
                            {idx: middle}, lower_bounds, upper_bounds, v_init, fit_v_init, fast=fast)
                        delta_middle = (cost - cost_best)/sigma2

                        if delta_middle > threshold:
                            outer, delta_outer = middle, delta_middle
                        else:
                            inner, delta_inner, x_inner = middle, delta_middle, list(x_middle)

                    frac = (threshold - delta_inner)/(delta_outer - delta_inner)
                    crossing = inner + frac*(outer - inner)

                    # Keep the innermost point next to the crossing, for the intervals of derived quantities
                    edge_models.append([np.exp(x_inner[0]), np.exp(x_inner[1]), \
                        (x_inner[2] if fit_v_init else v_init), delta_inner])
                    break

            # The profile reached the bound without crossing the threshold
            if crossing is None:
                crossing = values[indices[-1]]
                at_bound.append(True)
            else:
                at_bound.append(False)

            limits.append(np.exp(crossing))

        profile[name + '_lower'], profile[name + '_upper'] = limits
        profile[name + '_lower_at_bound'], profile[name + '_upper_at_bound'] = at_bound
        profile[name + '_profile'] = (np.exp(values), delta_chi2)

        # Alpha, beta and the initial velocity of every point of the profile
        profile[name + '_profile_models'] = np.array([[np.exp(x[0]), np.exp(x[1]), \
            (x[2] if fit_v_init else v_init)] for x in x_values])

        # Alpha, beta, the initial velocity and the chi-squared rise at the points next to the crossings
        profile[name + '_edge_models'] = np.array(edge_models).reshape(-1, 4)


    ### Compute the chi-squared rise on a 2D grid of alpha and beta ###

    profile['grid_alpha'] = np.array([])
    profile['grid_beta'] = np.array([])
    profile['grid_delta_chi2'] = np.zeros((0, 0))
    profile['grid_v_init'] = np.zeros((0, 0))
    profile['corr_log'] = np.nan

    if n_grid > 0:

        # Cover the extent of the profiles with a margin, within the bounds of the fit
        log_grids = []
        for idx, name in enumerate(['alpha', 'beta']):
            log_lower, log_upper = np.log(profile[name + '_profile'][0][[0, -1]])
            margin = 0.15*(log_upper - log_lower)
            log_grids.append(np.linspace(max(log_lower - margin, lower_bounds[idx]), \
                min(log_upper + margin, upper_bounds[idx]), n_grid))

        log_alpha_grid, log_beta_grid = log_grids

        grid_delta_chi2 = np.zeros((n_grid, n_grid))
        grid_v_init = np.full((n_grid, n_grid), v_init)
        for i, log_beta in enumerate(log_beta_grid):

            # Start every row from the best fit, and every point from the previous point of the row and from
            #   the same point of the previous row, as the initial velocity has local minima
            x_prev = list(x_best)
            for j, log_alpha in enumerate(log_alpha_grid):

                starts = [x_prev]
                if fit_v_init and (i > 0):
                    starts.append(list(x_prev[:2]) + [grid_v_init[i - 1, j]])

                cost = np.inf
                for x_start in starts:
                    cost_start, x_fit = _lagFitFixed(path, time_data, length_data, x_start, \
                        {0: log_alpha, 1: log_beta}, lower_bounds, upper_bounds, v_init, fit_v_init, \
                        fast=fast)
                    if cost_start < cost:
                        cost, x_new = cost_start, list(x_fit)

                x_prev = x_new

                grid_delta_chi2[i, j] = (cost - cost_best)/sigma2
                if fit_v_init:
                    grid_v_init[i, j] = x_prev[2]

        # Correlation of ln alpha and ln beta, weighted by the likelihood
        weights = np.exp(-0.5*(grid_delta_chi2 - np.min(grid_delta_chi2)))
        log_alpha_mesh, log_beta_mesh = np.meshgrid(log_alpha_grid, log_beta_grid)
        cov_log = np.cov(np.vstack([log_alpha_mesh.ravel(), log_beta_mesh.ravel()]), \
            aweights=weights.ravel())

        profile['grid_alpha'] = np.exp(log_alpha_grid)
        profile['grid_beta'] = np.exp(log_beta_grid)
        profile['grid_delta_chi2'] = grid_delta_chi2
        profile['grid_v_init'] = grid_v_init
        profile['corr_log'] = cov_log[0, 1]/np.sqrt(cov_log[0, 0]*cov_log[1, 1])

        # Extend the intervals over the points of the grid within them, as the walk of a profile stops at the
        #   first crossing of the threshold and the likelihood can have another valley beyond it
        region_alpha, region_beta, _ = profileRegionAlphaBetaLag(profile, ci=ci, n_params=1)
        for idx, name, values in [(0, 'alpha', region_alpha), (1, 'beta', region_beta)]:

            if np.min(values) < profile[name + '_lower']:
                profile[name + '_lower'] = np.min(values)
                profile[name + '_lower_at_bound'] = np.log(np.min(values)) <= lower_bounds[idx] + 1e-6

            if np.max(values) > profile[name + '_upper']:
                profile[name + '_upper'] = np.max(values)
                profile[name + '_upper_at_bound'] = np.log(np.max(values)) >= upper_bounds[idx] - 1e-6

    ### ###


    # Find the lowest point of the profiles and the grid, which is below the given fit if it ended in a local
    #   minimum
    candidates = [(0.0, (alpha, beta, v_init))]
    for name in ['alpha', 'beta']:
        i_min = np.argmin(profile[name + '_profile'][1])
        candidates.append((profile[name + '_profile'][1][i_min], \
            tuple(profile[name + '_profile_models'][i_min])))

    if profile['grid_delta_chi2'].size:
        i_min, j_min = np.unravel_index(np.argmin(profile['grid_delta_chi2']), \
            profile['grid_delta_chi2'].shape)
        candidates.append((profile['grid_delta_chi2'][i_min, j_min], (profile['grid_alpha'][j_min], \
            profile['grid_beta'][i_min], profile['grid_v_init'][i_min, j_min])))

    candidates.sort(key=lambda candidate: candidate[0])
    profile['min_delta_chi2'], profile['min_model'] = candidates[0]

    return profile



def profileRegionAlphaBetaLag(profile, ci=95.0, n_params=2):
    """ Alpha, beta and the initial velocity at the points of the 2D grid of profileAlphaBetaLag() within the
        region at the given confidence level, e.g. to draw the band of the models which fit the lag within the
        region of alpha and beta, or to find the interval of a quantity derived from them, e.g. a mass, as the
        range of its values within the region of one parameter.

    Arguments:
        profile: [dict] Result of profileAlphaBetaLag(), with the 2D grid.

    Keyword arguments:
        ci: [float] Confidence level of the region (percent). 95 by default.
        n_params: [int] Number of parameters of the chi-squared threshold of the region: 2 for the joint
            region of alpha and beta, 1 for the interval of a single quantity derived from them. 2 by default.

    Return:
        (alpha, beta, v_init): [tuple of ndarrays] Parameters of the points of the grid and of the profiles
            within the region. Empty if there is no grid.
    """

    if profile['grid_delta_chi2'].size == 0:
        return np.array([]), np.array([]), np.array([])

    threshold = scipy.stats.chi2.ppf(ci/100.0, n_params)

    inside = profile['grid_delta_chi2'] <= threshold
    alpha_mesh, beta_mesh = np.meshgrid(profile['grid_alpha'], profile['grid_beta'])
    models = [np.column_stack([alpha_mesh[inside], beta_mesh[inside], profile['grid_v_init'][inside]])]

    # Add the points of the profiles within the region, which reach its edges more closely than the grid
    for name in ['alpha', 'beta']:
        models.append(profile[name + '_profile_models'][profile[name + '_profile'][1] <= threshold])
        edge_models = profile[name + '_edge_models']
        models.append(edge_models[edge_models[:, 3] <= threshold, :3])

    models = np.vstack(models)

    return models[:, 0], models[:, 1], models[:, 2]



### Fit of the entry lag in slant-column coordinates ###

# Power-law nodes of the normalized velocity from 0.02, of which those below 0.9 are nodes of the entry lag
ENTRY_LAG_POWER_NODES = 0.02 + (1.0 - 1e-13 - 0.02)*np.linspace(0.0, 1.0, 1200)**1.5

# Nodes of the normalized velocity at which the entry lag is integrated, uniform in ln(1 - v) near entry
ENTRY_LAG_V_NODES = np.unique(np.concatenate([ENTRY_LAG_POWER_NODES[ENTRY_LAG_POWER_NODES < 0.9], \
    1.0 - np.logspace(-1, -13, 1000)]))

# Top of the path along which the slant air column is accumulated (m)
SLANT_COLUMN_TOP = 120e3

# Smallest number of points of a station for its own estimate of the noise and of the autocorrelation time
ENTRY_LAG_MIN_STATION_POINTS = 10

# Smallest noise of the distances of a station (m)
ENTRY_LAG_MIN_NOISE = 0.01

# Number of steps of a side of a profile of the fit of the entry lag after which the step is doubled at every
#   step, up to the largest step
ENTRY_LAG_PROFILE_SLOW_STEPS = 20

# Largest fraction of the points beyond the heights reached by a model of the fit of the entry lag
ENTRY_LAG_MAX_OUTSIDE = 0.02

# Largest number of fits of the entry lag with the weights from the previous fit
ENTRY_LAG_MAX_PASSES = 10

# Change of ln alpha and ln beta between two fits of the entry lag below which the fits have converged
ENTRY_LAG_PASS_TOL = 1e-3



class AlphaBetaSlantColumn(object):
    def __init__(self, length_path, ht_path, profile_hts, profile_dens, sin_slope_ref, sin_slope_top=None):
        """ Air column traversed by a meteor along its path, the slant column, as the height coordinate of the
            alpha-beta model. In this coordinate the model holds in any atmosphere and on a path whose slope
            changes, e.g. with the curvature of the Earth. The column is accumulated along the path from the
            column above its first point, taken at a constant slope, integrating the density over every
            interval between the points of the path as an exponential, and its normalized height

                y = -ln(eta*sin(slope_ref)/(RHO_ATM_0*HT_NORM_CONST))

            is the usual h/HT_NORM_CONST on a straight path of the reference slope in the exponential
            atmosphere of the model.

        Arguments:
            length_path: [ndarray] Distance flown through the air along the path (m), increasing along the
                motion.
            ht_path: [ndarray] Height of the path at these distances (m).
            profile_hts: [ndarray] Increasing heights of the density profile (m), from below the lowest of
                ht_path up to RESCALE_COLUMN_TOP, see airDensityProfile().
            profile_dens: [ndarray] Air density at profile_hts (kg/m^3).
            sin_slope_ref: [float] Sine of the reference slope. It only sets the unit of alpha, as
                alpha*sin(slope_ref) does not depend on it, so the masses from alphaBetaMasses() must be
                computed with the same slope.

        Keyword arguments:
            sin_slope_top: [float] Sine of the slope of the path above its first point. sin_slope_ref by
                default.
        """

        if sin_slope_top is None:
            sin_slope_top = sin_slope_ref

        self.sin_slope_ref = sin_slope_ref
        self.profile_hts = np.asarray(profile_hts, dtype=np.float64)
        self.profile_dens = np.asarray(profile_dens, dtype=np.float64)

        # Sort the path by the distance
        sort_indices = np.argsort(length_path)
        self.length_tab = np.asarray(length_path, dtype=np.float64)[sort_indices]
        self.ht_tab = np.asarray(ht_path, dtype=np.float64)[sort_indices]

        # Air density along the path, interpolated in log space
        dens_tab = np.exp(np.interp(self.ht_tab, self.profile_hts, np.log(self.profile_dens)))

        # Vertical column above every point of the path, from its height in the exponential atmosphere
        ht_vertical_tab = exponentialAtmosphereHeights(self.ht_tab, self.profile_hts, self.profile_dens)
        column_vertical_top = RHO_ATM_0*HT_NORM_CONST*np.exp(-ht_vertical_tab[0]/HT_NORM_CONST)

        # Accumulate the slant column along the path, integrating every interval as an exponential, which is
        #   exact where the density falls exponentially along the path
        length_steps = np.diff(self.length_tab)
        log_ratio = np.log(dens_tab[:-1]/dens_tab[1:])
        equal_dens = np.abs(log_ratio) < 1e-12
        column_steps = np.where(equal_dens, length_steps*dens_tab[:-1], \
            length_steps*(dens_tab[:-1] - dens_tab[1:])/np.where(equal_dens, 1.0, log_ratio))
        column = column_vertical_top/sin_slope_top + np.concatenate([[0.0], np.cumsum(column_steps)])

        # Normalized height of the column and the length of the path per unit of it
        self.ht_normed_tab = -np.log(column*sin_slope_ref/(RHO_ATM_0*HT_NORM_CONST))
        self.length_scale_tab = column/dens_tab

        # Difference from the normalized height of the vertical column, for the column beyond the ends of the
        #   path
        self.offset_tab = self.ht_normed_tab - ht_vertical_tab/HT_NORM_CONST



    def htNormed(self, length_data):
        """ Normalized column height at the given distances along the path.

        Arguments:
            length_data: [ndarray] Distance flown through the air (m), on the scale of the path.

        Return:
            [ndarray] Normalized column height.
        """

        return np.interp(length_data, self.length_tab, self.ht_normed_tab)



    def lengthScale(self, ht_normed):
        """ Length of the path per unit of the normalized column height, eta/rho, at the given normalized
            heights, for alphaBetaEntryLag(). Beyond the ends of the path it is that of the ends.

        Arguments:
            ht_normed: [ndarray] Normalized column height.

        Return:
            [ndarray] Length of the path per unit of the normalized column height (m).
        """

        # The normalized height decreases along the path
        return np.interp(ht_normed, self.ht_normed_tab[::-1], self.length_scale_tab[::-1])



    def htNormedAtHeights(self, ht_data):
        """ Normalized column height at the given heights of the path, e.g. to draw the model against the
            height. Beyond the ends of the path the column is the vertical one times its ratio at the end.

        Arguments:
            ht_data: [ndarray] Height (m).

        Return:
            [ndarray] Normalized column height.
        """

        # Add the difference from the vertical column at the same height, which is constant beyond the ends
        ht_vertical = exponentialAtmosphereHeights(np.atleast_1d(ht_data), self.profile_hts, \
            self.profile_dens)/HT_NORM_CONST

        return ht_vertical + np.interp(ht_data, self.ht_tab[::-1], self.offset_tab[::-1])



def slantColumnFromPoints(time_data, length_data, ht_data, profile_hts, profile_dens, sin_slope_ref, \
    air_speed=0.0):
    """ Compute the slant air column along the path of a meteor from the points of a trajectory solution.

        The heights of the points are smoothed by fitting them with

            h = p0 + p1*l + p2*l^2 + p3*t^2 + p4*t^3,

        where l is the distance flown through the air, the quadratic term in l is the curvature of the Earth
        below the straight line of the solution and the terms in the time t are the gravity drop. Above the
        first point the path follows the quadratic in l up to SLANT_COLUMN_TOP. The distances along the line
        of a solution in an inertial frame exceed those flown through the air, which turns with the Earth, by
        air_speed*(t - t1), where t1 is the time of the first point along the path.

    Arguments:
        time_data: [ndarray] Time of every point (s).
        length_data: [ndarray] Distance along the path of every point (m), e.g.
            ObservedPoints.state_vect_dist.
        ht_data: [ndarray] Height of every point (m), e.g. ObservedPoints.model_ht.
        profile_hts: [ndarray] Increasing heights of the density profile (m), see airDensityProfile().
        profile_dens: [ndarray] Air density at profile_hts (kg/m^3).
        sin_slope_ref: [float] Sine of the reference slope, which sets the unit of alpha, see
            AlphaBetaSlantColumn.

    Keyword arguments:
        air_speed: [float] Speed of the air along the path (m/s), see airSpeedAlongPath(). 0 by default, for
            distances which are already relative to the air.

    Return:
        (slant_column, length_air):
            - slant_column: [AlphaBetaSlantColumn object] Slant column along the path.
            - length_air: [ndarray] Distance flown through the air of every point (m), at which the column is
                read, e.g. slant_column.htNormed(length_air).
    """

    time_data = np.asarray(time_data, dtype=np.float64)
    length_data = np.asarray(length_data, dtype=np.float64)
    ht_data = np.asarray(ht_data, dtype=np.float64)

    # Distances flown through the air
    first_index = np.argmin(length_data)
    length_air = length_data - air_speed*(time_data - time_data[first_index])

    # Fit the heights with the curvature of the Earth and the gravity drop
    design = np.column_stack([np.ones_like(length_air), length_air, length_air**2, time_data**2, \
        time_data**3])
    path_coeffs = np.linalg.lstsq(design, ht_data, rcond=None)[0]

    # Slope of the path at the first point
    length_first = length_air[first_index]
    sin_slope_first = -(path_coeffs[1] + 2*path_coeffs[2]*length_first)
    if sin_slope_first <= 0:
        raise ValueError("The path does not descend at its first point, sin(slope) = {:.3f}".format( \
            sin_slope_first))

    # Extend the path above the first point up to the top of the column
    length_top = np.linspace(length_first - (SLANT_COLUMN_TOP - np.max(ht_data))/sin_slope_first, \
        length_first, 400)[:-1]
    ht_top = path_coeffs[0] + path_coeffs[1]*length_top + path_coeffs[2]*length_top**2

    slant_column = AlphaBetaSlantColumn(np.concatenate([length_top, length_air]), \
        np.concatenate([ht_top, design.dot(path_coeffs)]), profile_hts, profile_dens, sin_slope_ref, \
        sin_slope_top=sin_slope_first)

    return slant_column, length_air



def airSpeedAlongPath(state_vect, radiant_eci):
    """ Compute the speed of the air along the path of a meteor, from the rotation of the Earth, by which the
        distances along the line of a trajectory solution in an inertial frame exceed those flown through the
        air. It is the same along the whole straight line.

    Arguments:
        state_vect: [ndarray] ECI position of a point of the line (m), e.g. Trajectory.state_vect_mini.
        radiant_eci: [ndarray] ECI direction of the radiant, e.g. Trajectory.radiant_eci_mini.

    Return:
        [float] Speed of the air along the direction of the motion (m/s).
    """

    # The meteor moves away from the radiant
    motion = -vectNorm(np.asarray(radiant_eci, dtype=np.float64))

    return float(np.dot(earthRotationVelocity(state_vect), motion))



def gravityLengthAlongLine(time_data, length_data, state_vect, radiant_eci, gravity_factor=1.0, \
    time_ref=None):
    """ Compute the distance added by gravity along the line of a trajectory solution, relative to a companion
        of the meteor which falls freely from the reference time, with gravity and without drag. The entry lag
        is referred to that companion. The acceleration along the line changes with the slope of the line to
        the local horizontal, so it is computed at every point, fitted with a quadratic in time and integrated
        twice,

            d(t) = int int g*sin(slope) dt dt,   d(t_ref) = d'(t_ref) = 0.

    Arguments:
        time_data: [ndarray] Time of every point (s).
        length_data: [ndarray] Distance of every point along the line from the state vector (m), e.g.
            ObservedPoints.state_vect_dist.
        state_vect: [ndarray] ECI position of the state vector (m), e.g. Trajectory.state_vect_mini.
        radiant_eci: [ndarray] ECI direction of the radiant, e.g. Trajectory.radiant_eci_mini.

    Keyword arguments:
        gravity_factor: [float] Factor of the gravity, e.g. Trajectory.gravity_factor. 1.0 by default.
        time_ref: [float] Reference time (s). The time of the first point by default.

    Return:
        [ndarray] Distance added by gravity along the line at every point (m).
    """

    time_data = np.asarray(time_data, dtype=np.float64)
    length_data = np.asarray(length_data, dtype=np.float64)

    # Position of every point on the line, as the meteor moves away from the radiant
    motion = -vectNorm(np.asarray(radiant_eci, dtype=np.float64))
    eci_pos = np.asarray(state_vect, dtype=np.float64)[None, :] + length_data[:, None]*motion[None, :]

    # Acceleration of gravity along the motion at every point
    r_mag = np.linalg.norm(eci_pos, axis=1)
    accel_along = -gravity_factor*G*EARTH.MASS*eci_pos.dot(motion)/r_mag**3

    # Fit the acceleration with a quadratic in time and integrate it twice from the reference time
    if time_ref is None:
        time_ref = np.min(time_data)

    time_rel = time_data - time_ref
    accel_coeffs = np.polyfit(time_rel, accel_along, 2)

    return accel_coeffs[0]*time_rel**4/12 + accel_coeffs[1]*time_rel**3/6 + accel_coeffs[2]*time_rel**2/2



def stationLengthNoise(time_data, length_data, station_data, n_min=ENTRY_LAG_MIN_STATION_POINTS):
    """ Estimate the noise of the distances along the path of every station, from the second differences of
        the residuals r of its distances about a quartic in time,

            sigma = 1.4826*MAD(r[j+1] - 2*r[j] + r[j-1])/sqrt(6),

        where MAD is the median absolute deviation. The estimate also contains the effect of the timing
        errors on the distances. A station with fewer than n_min points takes the median of the estimates of
        the other stations, and if no station has n_min points, the second differences of the residuals of
        all stations about a quadratic in time are pooled. A station with an estimate below
        ENTRY_LAG_MIN_NOISE, e.g. from quantized distances, also takes the median of the others.

    Arguments:
        time_data: [ndarray] Time of every point (s).
        length_data: [ndarray] Distance along the path of every point (m).
        station_data: [ndarray] Station of every point.

    Keyword arguments:
        n_min: [int] Smallest number of points of a station with its own estimate.
            ENTRY_LAG_MIN_STATION_POINTS by default.

    Return:
        [ndarray] Noise of the distance of every point (m). A ValueError is raised if no station has the
            points to estimate the noise from.
    """

    time_data = np.asarray(time_data, dtype=np.float64)
    length_data = np.asarray(length_data, dtype=np.float64)
    station_data = np.asarray(station_data)

    def _robustSigma(second_diffs):
        """ Standard deviation of the noise from its second differences, with the MAD. """

        return 1.4826*np.median(np.abs(second_diffs - np.median(second_diffs)))/np.sqrt(6)


    # Estimate the noise of every station, or keep its second differences to pool them
    station_sigma = {}
    pooled_diffs = []
    for station_id in np.unique(station_data):

        # Points of the station in time order
        indices = np.where(station_data == station_id)[0]
        indices = indices[np.argsort(time_data[indices])]
        station_lengths = length_data[indices]

        if len(indices) >= n_min:

            # Second differences of the residuals about a quartic in time
            time_rel = time_data[indices] - np.mean(time_data[indices])
            residuals = station_lengths - np.polyval(np.polyfit(time_rel, station_lengths, 4), time_rel)
            station_sigma[station_id] = _robustSigma(residuals[2:] - 2*residuals[1:-1] + residuals[:-2])

        elif len(indices) >= 4:

            # Keep the second differences of the residuals about a quadratic in time to pool them, so the
            #   deceleration is not taken as noise
            time_rel = time_data[indices] - np.mean(time_data[indices])
            residuals = station_lengths - np.polyval(np.polyfit(time_rel, station_lengths, 2), time_rel)
            second_diffs = residuals[2:] - 2*residuals[1:-1] + residuals[:-2]
            pooled_diffs.append(second_diffs - np.median(second_diffs))

    # Use only the estimates above the smallest noise, as a zero noise would give an infinite weight
    station_sigma = {station_id: sigma for station_id, sigma in station_sigma.items() \
        if sigma > ENTRY_LAG_MIN_NOISE}

    # The noise of the stations with too few points
    if station_sigma:
        sigma_fill = np.median(list(station_sigma.values()))
    elif pooled_diffs:
        sigma_fill = _robustSigma(np.concatenate(pooled_diffs))
    else:
        sigma_fill = np.nan

    if not (sigma_fill > ENTRY_LAG_MIN_NOISE):
        raise ValueError("The noise of the distances cannot be estimated, as no station has enough points")

    return np.array([station_sigma.get(station_id, sigma_fill) for station_id in station_data])



def alphaBetaEntryLag(alpha, beta, length_scale):
    """ Compute the entry lag of the alpha-beta model, the distance by which the meteor trails a companion
        which keeps moving at the entry velocity, as a function of the normalized column height y:

            l(y) = int_y^inf H(y') (1/v(y') - 1) dy',

        where H is the length of the path per unit of the normalized column height, at the height of the
        model. Near entry l = H*(1 - v). The integral is computed with the trapezoidal rule over the nodes
        ENTRY_LAG_V_NODES of the normalized velocity, at which the height of the model follows from
        alphaBetaHeightNormed(), so the model never has to be inverted, with dy/dv = exp(y - beta*(1 - v^2))/
        (alpha*v).

    Arguments:
        alpha: [float] Ballistic coefficient.
        beta: [float] Mass loss parameter.
        length_scale: [function or float] Length of the path per unit of the normalized column height (m) at
            the given normalized heights, e.g. AlphaBetaSlantColumn.lengthScale, or a constant, e.g.
            HT_NORM_CONST/sin(slope) for a straight path in the exponential atmosphere.

    Return:
        (ht_normed, lag, v_normed):
            - ht_normed: [ndarray] Normalized column height of the model at the nodes, increasing.
            - lag: [ndarray] Entry lag at the nodes (m).
            - v_normed: [ndarray] Normalized velocity at the nodes.
    """

    v_nodes = ENTRY_LAG_V_NODES

    # Height of the model at the nodes and its derivative by the velocity
    ht_normed = alphaBetaHeightNormed(v_nodes, alpha, beta)
    dy_dv = np.exp(ht_normed - beta*(1.0 - v_nodes**2))/(alpha*v_nodes)

    # Length of the path per unit of the column height at the height of the model
    if callable(length_scale):
        length_scale_nodes = length_scale(ht_normed)
    else:
        length_scale_nodes = np.full_like(ht_normed, length_scale)

    # Integrate the lag from entry down to every node
    integrand = length_scale_nodes*(1.0/v_nodes - 1.0)*dy_dv
    cumulative = np.concatenate([[0.0], np.cumsum(0.5*(integrand[1:] + integrand[:-1])*np.diff(v_nodes))])
    lag = cumulative[-1] - cumulative

    # Sort by the height
    order = np.argsort(ht_normed)

    return ht_normed[order], lag[order], v_nodes[order]



def _entryLagModel(alpha, beta, ht_normed, length_scale):
    """ Model lag of the fit of the entry lag, the negative entry lag, at the column heights of the points.

    Arguments:
        alpha: [float] Ballistic coefficient.
        beta: [float] Mass loss parameter.
        ht_normed: [ndarray] Normalized column height of every point.
        length_scale: [function or float] See alphaBetaEntryLag().

    Return:
        (lag_model, v_model, outside):
            - lag_model: [ndarray] Model lag at every point (m). Below the lowest height reached by the
                model, the entry lag continues with its slope at the lowest node, H*(1/v - 1) per unit of the
                column height, so the residuals of the points which the model does not reach grow
                continuously with their distance beyond its end.
            - v_model: [ndarray] Normalized velocity of the model at every point, for the weights of the fit.
                It is 1 at the points below the lowest height reached by the model, where the model has no
                velocity.
            - outside: [float] Fraction of the points outside the heights reached by the model.
    """

    ht_nodes, lag_nodes, v_nodes = alphaBetaEntryLag(alpha, beta, length_scale)

    below = ht_normed < ht_nodes[0]
    outside = np.mean(below | (ht_normed > ht_nodes[-1]))

    lag_model = -np.interp(ht_normed, ht_nodes, lag_nodes)

    # Continue the entry lag below the lowest height of the model with its slope there
    if np.any(below):

        if callable(length_scale):
            length_scale_lowest = length_scale(ht_nodes[:1])[0]
        else:
            length_scale_lowest = length_scale

        lag_slope = length_scale_lowest*(1.0/v_nodes[0] - 1.0)
        lag_model[below] = -(lag_nodes[0] + lag_slope*(ht_nodes[0] - ht_normed[below]))

    # The points which the model never reaches are weighted as at entry, so their weights do not vanish
    v_model = np.where(below, 1.0, np.interp(ht_normed, ht_nodes, v_nodes))

    return lag_model, v_model, outside



def _entryLagLinear(lag_diff, time_data, weights, station_index, n_stations):
    """ Fit the correction to the slope of the reference line and the offsets of the stations of the fit of
        the entry lag in closed form, by weighted least squares.

    Arguments:
        lag_diff: [ndarray] Observed lag minus the model lag at every point (m).
        time_data: [ndarray] Time of every point (s).
        weights: [ndarray] Weight of every point (1/m^2).
        station_index: [ndarray] Index of the station of every point.
        n_stations: [int] Number of stations.

    Return:
        (dv, offsets, residuals):
            - dv: [float] Correction to the slope of the reference line (m/s).
            - offsets: [ndarray] Offset of every station (m).
            - residuals: [ndarray] Residual of every point (m).
    """

    # Weighted means of the time and of the lag of every station
    weight_sums = np.bincount(station_index, weights, n_stations)
    time_means = np.bincount(station_index, weights*time_data, n_stations)/weight_sums
    lag_means = np.bincount(station_index, weights*lag_diff, n_stations)/weight_sums

    # Fit the slope to the deviations from the means of the stations
    time_dev = time_data - time_means[station_index]
    lag_dev = lag_diff - lag_means[station_index]
    dv = np.sum(weights*time_dev*lag_dev)/np.sum(weights*time_dev**2)

    return dv, lag_means - dv*time_means, lag_dev - dv*time_dev



def _entryLagTauInt(time_data, residuals_normed, station_index, n_stations, \
    min_points=ENTRY_LAG_MIN_STATION_POINTS):
    """ Integrated autocorrelation time of the normalized residuals of the fit of the entry lag, from the
        residuals of every station with at least min_points points in time order, averaged over the stations
        with the numbers of their points as weights. It is about 1 for independent errors and at least 1, and
        1 if no station has min_points points.

    Arguments:
        time_data: [ndarray] Time of every point (s).
        residuals_normed: [ndarray] Residuals times the square roots of their weights.
        station_index: [ndarray] Index of the station of every point.
        n_stations: [int] Number of stations.

    Keyword arguments:
        min_points: [int] Smallest number of points of a station which is used.
            ENTRY_LAG_MIN_STATION_POINTS by default.

    Return:
        [float] Integrated autocorrelation time.
    """

    tau_sum = 0.0
    n_sum = 0
    for i in range(n_stations):

        station_mask = station_index == i
        n_points = np.count_nonzero(station_mask)
        if n_points < min_points:
            continue

        # Autocorrelation time of the residuals of the station in time order
        tau = integratedAutocorrTime(residuals_normed[station_mask][np.argsort(time_data[station_mask])])
        if not np.isfinite(tau):
            continue

        tau_sum += n_points*tau
        n_sum += n_points

    if n_sum == 0:
        return 1.0

    return max(tau_sum/n_sum, 1.0)



def fitAlphaBetaEntryLag(time_data, lag_data, ht_normed, station_data, length_scale, sigma_length, v_ref, \
    max_pass=ENTRY_LAG_MAX_PASSES, x0_alpha_beta=None, estimate_errors=False, ci=95.0, air_speed=0.0):
    """ Fit the alpha-beta model to the entry lag in slant-column coordinates, with an offset and a noise
        level for every station (the full fit of the lag).

        The lag of a point relative to a reference line in time is the negative entry lag of the model at its
        column height (alphaBetaEntryLag()) plus a straight line in time,

            lag_i = -l(y_i; alpha, beta) + dv*t_i + K_S(i) + r_i,

        where dv is the entry velocity minus the slope of the reference line, which is too low if the meteor
        already decelerates where the line is fitted, and K_S is an offset of the station S of the point,
        which absorbs a constant offset of its clock. If the lag is formed from the distances along the line
        of a solution in an inertial frame and the column heights from the distances flown through the air
        (slantColumnFromPoints()), the slope of the lag is the inertial entry velocity, which exceeds the
        entry velocity relative to the air, the velocity of the model, by the speed of the air along the path,
        so the entry velocity v_ref + dv - air_speed relative to the air is returned. For given alpha and
        beta, dv and the offsets follow in closed form by weighted least squares, so only ln alpha and ln beta
        are fitted, and the velocities are not needed. An error of the distance of a point moves the lag of
        the model at its column height too, so the residual is the error divided by the normalized velocity
        v_i of the model at the point, and the point is weighted by 1/(sigma_S/v_i)^2, with sigma_S the noise
        of its station. Then v_i*r_i is the residual of the distance at the time of the point. The velocity is
        1 in the first fit and that of the model of the previous fit in the next ones, so the weights are
        fixed during a fit, and the fits are repeated until alpha and beta change by less than
        ENTRY_LAG_PASS_TOL in ln, or max_pass fits. If they do not converge, the fit of which the model fits
        the distances best is kept. Every fit starts from the best of a grid in ln alpha and ln beta, the
        given starting points and the previous fit, with its weights. A model may leave at most
        ENTRY_LAG_MAX_OUTSIDE of the points beyond the heights which it reaches, which are weighted as if its
        velocity was 1 there, and alpha and beta are bound by ALPHA_BETA_BOUNDS. The fits start only from
        models which reach the points, and a ValueError is raised if no starting point does.

    Arguments:
        time_data: [ndarray] Time of every point (s).
        lag_data: [ndarray] Lag of every point (m): its distance along the path minus v_ref times its time,
            the same line for all stations, without the distance added by gravity along the path, see
            gravityLengthAlongLine().
        ht_normed: [ndarray] Normalized slant column height of every point, see
            AlphaBetaSlantColumn.htNormed().
        station_data: [ndarray] Station of every point.
        length_scale: [function or float] Length of the path per unit of the normalized column height (m),
            e.g. AlphaBetaSlantColumn.lengthScale.
        sigma_length: [ndarray] Noise of the distance of every point (m), see stationLengthNoise().
        v_ref: [float] Slope of the reference line of the lag (m/s), e.g. Trajectory.v_init.

    Keyword arguments:
        max_pass: [int] Largest number of fits. ENTRY_LAG_MAX_PASSES by default.
        x0_alpha_beta: [list] Additional (alpha, beta) starting points of the fits, e.g. the results of
            fitAlphaBeta(). None by default.
        estimate_errors: [bool] Also return the Wald uncertainties of the fit, in the format of
            fitAlphaBetaLag(estimate_errors=True), so they can be passed to alphaBetaMasses(), with the
            covariance scaled by the reduced chi-squared and the integrated autocorrelation time of the
            residuals. NaN if alpha or beta end at a bound of the fit. False by default.
        ci: [float] Confidence level of the Wald intervals (percent). 95 by default.
        air_speed: [float] Speed of the air along the path (m/s), by which the slope of the lag exceeds the
            entry velocity of the model, see airSpeedAlongPath() and slantColumnFromPoints(). 0 by default,
            for distances which are already relative to the air.

    Return:
        (v_init, alpha, beta, lag_residuals, entry_fit), or (v_init, alpha, beta, lag_residuals, entry_fit,
            errors) if estimate_errors is True:
            - v_init: [float] Entry velocity relative to the air, v_ref + dv - air_speed (m/s), by which the
                normalized velocities of the model are scaled.
            - alpha: [float] Ballistic coefficient.
            - beta: [float] Mass loss parameter.
            - lag_residuals: [ndarray] Residual of the lag of every point (m).
            - entry_fit: [dict] The fit, for profileAlphaBetaEntryLag(), with the keys 'v_ref', 'dv',
                'air_speed', 'offsets' (offset of every station, m), 'sigma_station' (noise of every station,
                m), 'weights', 'v_model' (normalized velocity of the model at every point, 1 beyond the
                heights which it reaches, by which the lag residuals are multiplied to give the residuals of
                the distances), 'cost' (weighted sum of the squared residuals), 'dof' (degrees of freedom),
                'chi2_nu' (reduced chi-squared), 'tau_int' (integrated autocorrelation time of the normalized
                residuals), 'scale' (chi2_nu*tau_int, the variance factor of the uncertainties), 'jac'
                (Jacobian of the weighted residuals by ln alpha and ln beta), 'n_pass' (number of fits) and
                'converged' (False if the fits did not converge in max_pass fits).
            - errors: [dict] See estimate_errors.
    """

    time_data = np.asarray(time_data, dtype=np.float64)
    lag_data = np.asarray(lag_data, dtype=np.float64)
    ht_normed = np.asarray(ht_normed, dtype=np.float64)
    sigma_length = np.asarray(sigma_length, dtype=np.float64)

    # Index of the station of every point
    station_ids, station_index = np.unique(np.asarray(station_data), return_inverse=True)
    n_stations = len(station_ids)

    log_bounds = _logAlphaBetaBounds()

    # Starting points of the fits, a grid in ln alpha and ln beta and the given starting points
    starts = [(alpha, beta) for alpha in np.geomspace(1e-2, 1e3, 21) for beta in np.geomspace(1e-3, 40.0, 21)]
    if x0_alpha_beta is not None:
        starts += [tuple(x0[:2]) for x0 in x0_alpha_beta]
    log_starts = [np.clip(np.log(start), log_bounds[0], log_bounds[1]) for start in starts]

    def _reachesPoints(x):
        """ Check that the model for (ln alpha, ln beta) reaches the column heights of the points. """

        return _entryLagModel(np.exp(x[0]), np.exp(x[1]), ht_normed, length_scale)[2] <= ENTRY_LAG_MAX_OUTSIDE


    v_model = np.ones_like(time_data)
    x_best = None
    passes = []
    converged = False
    for _ in range(max_pass):

        # Weights of the fit, from the velocity of the model of the previous fit
        weights = (v_model/sigma_length)**2

        # Residual of the models which do not reach the points, set above the cost of the start of the fit
        penalty = {'residual': 1e4}

        def _residuals(x, weights=weights):
            """ Weighted residuals of the lag for (ln alpha, ln beta), with dv and the offsets fitted. """

            alpha, beta = np.exp(x)
            lag_model, _, outside = _entryLagModel(alpha, beta, ht_normed, length_scale)

            # Reject models which do not reach the column heights of the points
            if outside > ENTRY_LAG_MAX_OUTSIDE:
                return np.full(len(time_data), penalty['residual'])

            _, _, residuals = _entryLagLinear(lag_data - lag_model, time_data, weights, station_index, \
                n_stations)

            return np.sqrt(weights)*residuals


        # Start from the best of the grid, the given starting points and the previous fit, with these weights,
        #   of the models which reach the points
        candidates = [candidate for candidate in log_starts + ([x_best] if x_best is not None else []) \
            if _reachesPoints(candidate)]
        if not candidates:
            raise ValueError("No starting point of the fit of the entry lag gives a model which reaches " \
                "the points")

        costs = [np.sum(_residuals(candidate)**2) for candidate in candidates]
        x_start = candidates[int(np.argmin(costs))]

        # Make the rejected models cost much more than the start, so the fit never steps into them
        penalty['residual'] = max(1e4, 10*np.sqrt(np.min(costs)/len(time_data)))

        res = scipy.optimize.least_squares(_residuals, x_start, bounds=log_bounds, x_scale='jac', \
            max_nfev=300)

        # Also continue the previous fit if the fit started elsewhere, and keep the better one
        if (x_best is not None) and np.any(x_start != x_best):
            res_previous = scipy.optimize.least_squares(_residuals, x_best, bounds=log_bounds, \
                x_scale='jac', max_nfev=300)
            if res_previous.cost < res.cost:
                res = res_previous

        x_previous = x_best
        x_best = res.x if _reachesPoints(res.x) else x_start

        # Velocity of the model at the points, for the weights of the next fit
        alpha, beta = np.exp(x_best)
        lag_model, v_model, _ = _entryLagModel(alpha, beta, ht_normed, length_scale)
        v_model = np.clip(v_model, ENTRY_LAG_V_NODES[0], 1.0)

        # Residuals of the distances of the model, by which the fits are compared if they do not converge
        _, _, lag_residuals = _entryLagLinear(lag_data - lag_model, time_data, weights, station_index, \
            n_stations)
        passes.append({'x': x_best, 'weights': weights, 'v_model': v_model, \
            'cost_length': np.sum((v_model*lag_residuals/sigma_length)**2)})

        # Stop when alpha and beta do not change any more
        if (x_previous is not None) and (np.max(np.abs(x_best - x_previous)) < ENTRY_LAG_PASS_TOL):
            converged = True
            break

    # If the fits did not converge, keep the one of which the model fits the distances best
    best_pass = passes[-1] if converged else min(passes, key=lambda fit_pass: fit_pass['cost_length'])
    weights = best_pass['weights']
    v_model = best_pass['v_model']
    alpha, beta = np.exp(best_pass['x'])
    lag_model, _, _ = _entryLagModel(alpha, beta, ht_normed, length_scale)

    # Entry velocity, offsets and residuals of the kept fit
    dv, offsets, lag_residuals = _entryLagLinear(lag_data - lag_model, time_data, weights, station_index, \
        n_stations)
    v_init = v_ref + dv - air_speed

    # Reduced chi-squared and the autocorrelation time of the normalized residuals
    cost = np.sum(weights*lag_residuals**2)
    dof = max(len(time_data) - 3 - n_stations, 1)
    chi2_nu = cost/dof
    tau_int = _entryLagTauInt(time_data, np.sqrt(weights)*lag_residuals, station_index, n_stations)

    # Derivatives of the model lag by ln alpha and ln beta, by central differences of the model, so they do
    #   not cross the rejection of the models which do not reach the points
    log_step = 1e-4
    lag_derivs = []
    for log_factors in [(log_step, 0.0), (0.0, log_step)]:

        lag_plus, _, _ = _entryLagModel(alpha*np.exp(log_factors[0]), beta*np.exp(log_factors[1]), \
            ht_normed, length_scale)
        lag_minus, _, _ = _entryLagModel(alpha*np.exp(-log_factors[0]), beta*np.exp(-log_factors[1]), \
            ht_normed, length_scale)
        lag_derivs.append((lag_plus - lag_minus)/(2*log_step))

    # Jacobian of the weighted residuals, with dv and the offsets fitted, which are linear in the lag
    jac = np.column_stack([np.sqrt(weights)*_entryLagLinear(-lag_deriv, time_data, weights, station_index, \
        n_stations)[2] for lag_deriv in lag_derivs])

    entry_fit = {
        'v_ref': v_ref,
        'dv': dv,
        'air_speed': air_speed,
        'offsets': dict(zip(station_ids, offsets)),
        'sigma_station': {station_id: sigma_length[station_index == i][0] \
            for i, station_id in enumerate(station_ids)},
        'weights': weights,
        'v_model': v_model,
        'cost': cost,
        'dof': dof,
        'chi2_nu': chi2_nu,
        'tau_int': tau_int,
        'scale': chi2_nu*tau_int,
        'jac': jac,
        'n_pass': len(passes),
        'converged': converged
        }

    if not estimate_errors:
        return v_init, alpha, beta, lag_residuals, entry_fit


    ### Estimate the uncertainties from the linearized fit ###

    errors = {'alpha': alpha, 'beta': beta, 'v_init': v_init, 'ci': ci, 'v_init_std': np.nan, \
        'cov_log': np.full((2, 2), np.nan), 'cov': np.full((2, 2), np.nan), \
        'cov_full': np.full((3, 3), np.nan)}
    for key in ['alpha_std_rel', 'beta_std_rel', 'alpha_std', 'beta_std', 'corr_log', 'corr', \
            'alpha_ci_wald_lower', 'alpha_ci_wald_upper', 'beta_ci_wald_lower', 'beta_ci_wald_upper']:
        errors[key] = np.nan

    # The Gaussian approximation is not valid at a bound, nor without more points than parameters
    if alphaBetaBoundWarnings(alpha, beta) or (len(time_data) <= 3 + n_stations):
        return v_init, alpha, beta, lag_residuals, entry_fit, errors

    # Covariance of ln alpha, ln beta, dv and the offsets of the stations
    design = np.column_stack(lag_derivs + [time_data, np.eye(n_stations)[station_index]])
    design_weighted = design*np.sqrt(weights)[:, None]
    try:
        cov_params = entry_fit['scale']*np.linalg.inv(design_weighted.T.dot(design_weighted))
    except np.linalg.LinAlgError:
        return v_init, alpha, beta, lag_residuals, entry_fit, errors

    cov_log = cov_params[:2, :2]
    if (not np.all(np.isfinite(cov_log))) or np.any(np.diag(cov_log) <= 0):
        return v_init, alpha, beta, lag_residuals, entry_fit, errors

    alpha_std_rel, beta_std_rel = np.sqrt(np.diag(cov_log))

    # Linear (alpha, beta) space, via the delta method
    param_scale = np.array([alpha, beta])
    cov = np.outer(param_scale, param_scale)*cov_log

    # Log-normal Wald intervals
    z = scipy.special.ndtri(0.5 + ci/200.0)

    errors.update({
        'alpha_std_rel': alpha_std_rel,
        'beta_std_rel': beta_std_rel,
        'alpha_std': alpha*alpha_std_rel,
        'beta_std': beta*beta_std_rel,
        'cov_log': cov_log,
        'corr_log': cov_log[0, 1]/(alpha_std_rel*beta_std_rel),
        'cov': cov,
        'corr': cov[0, 1]/np.sqrt(cov[0, 0]*cov[1, 1]),
        'alpha_ci_wald_lower': alpha*np.exp(-z*alpha_std_rel),
        'alpha_ci_wald_upper': alpha*np.exp(z*alpha_std_rel),
        'beta_ci_wald_lower': beta*np.exp(-z*beta_std_rel),
        'beta_ci_wald_upper': beta*np.exp(z*beta_std_rel),
        'v_init_std': np.sqrt(cov_params[2, 2]),
        'cov_full': cov_params[np.ix_([2, 0, 1], [2, 0, 1])]
        })

    return v_init, alpha, beta, lag_residuals, entry_fit, errors



def stationJackknifeAlphaBetaEntryLag(time_data, lag_data, ht_normed, station_data, length_scale, \
    sigma_length, v_ref, alpha, beta):
    """ Fit the entry lag again without every station in turn (see fitAlphaBetaEntryLag()). The profile
        intervals only contain the random noise of the distances, so a station with a systematic error, e.g.
        of its calibration, shows as a fit without it which is far outside the intervals.

    Arguments:
        time_data: [ndarray] Time of every point (s).
        lag_data: [ndarray] Lag of every point (m), see fitAlphaBetaEntryLag().
        ht_normed: [ndarray] Normalized slant column height of every point.
        station_data: [ndarray] Station of every point.
        length_scale: [function or float] Length of the path per unit of the normalized column height (m).
        sigma_length: [ndarray] Noise of the distance of every point (m).
        v_ref: [float] Slope of the reference line of the lag (m/s).
        alpha: [float] Alpha of the fit with all stations, from which the fits start.
        beta: [float] Beta of the fit with all stations, from which the fits start.

    Return:
        [list] (station, alpha, beta) of the fit without every station, for the stations without which
            at least two stations are left.
    """

    station_data = np.asarray(station_data)
    station_ids = np.unique(station_data)

    jackknife = []
    for station_id in station_ids:

        # Keep the points of the other stations, if at least two of them are left
        keep = station_data != station_id
        if len(np.unique(station_data[keep])) < 2:
            continue

        # Skip the station if no model reaches the points of the others
        try:
            fit_result = fitAlphaBetaEntryLag(time_data[keep], lag_data[keep], ht_normed[keep], \
                station_data[keep], length_scale, sigma_length[keep], v_ref, x0_alpha_beta=[[alpha, beta]])
        except ValueError:
            continue

        jackknife.append((station_id, fit_result[1], fit_result[2]))

    return jackknife



def profileAlphaBetaEntryLag(time_data, lag_data, ht_normed, station_data, length_scale, alpha, beta, \
    entry_fit, ci=95.0, n_grid=40):
    """ Profile likelihood intervals of alpha and beta of the fit of the entry lag (fitAlphaBetaEntryLag()),
        and the likelihood on a grid of alpha and beta, in the format of profileAlphaBetaLag().

        The cost is the weighted sum of the squared residuals with the weights of the fit, minimized over dv
        and the offsets of the stations in closed form. One of ln alpha and ln beta is fixed at a time and
        the cost is minimized over the other one: over ln alpha within a bracket which follows the floor of
        the valley of the cost, and over ln beta from a grid refined by a bounded minimization, as the cost
        is nearly flat along the valley at a fixed alpha. The rise of the cost is divided by the reduced
        chi-squared of the fit times the integrated autocorrelation time of its residuals, which is about 1
        for independent errors and widens the intervals if the errors are correlated in time, and compared
        with the chi-squared threshold of one parameter. The intervals contain the random noise of the
        distances, also correlated in time, but not the systematic errors of the stations, see
        stationJackknifeAlphaBetaEntryLag(). Each profile starts with steps of half the Wald
        error, which double after ENTRY_LAG_PROFILE_SLOW_STEPS steps where the cost rises slower than
        expected, and the crossings of the threshold are refined with Brent's method. A side which reaches a
        bound of the fit gives only a limit.

    Arguments:
        time_data: [ndarray] Time of every point (s), as given to fitAlphaBetaEntryLag().
        lag_data: [ndarray] Lag of every point (m), as given to fitAlphaBetaEntryLag().
        ht_normed: [ndarray] Normalized slant column height of every point.
        station_data: [ndarray] Station of every point.
        length_scale: [function or float] Length of the path per unit of the normalized column height (m).
        alpha: [float] Alpha of the fit.
        beta: [float] Beta of the fit.
        entry_fit: [dict] The fit, from fitAlphaBetaEntryLag().

    Keyword arguments:
        ci: [float] Confidence level of the intervals (percent). 95 by default.
        n_grid: [int] Number of points of the 2D grid along each axis, which covers the profiles. 40 by
            default. 0 skips the grid.

    Return:
        profile: [dict] With the keys of profileAlphaBetaLag(), where the entry velocity relative to the air
            takes the place of the fitted initial velocity and 'sigma2' is the variance factor of the rise of
            the cost (entry_fit['scale']), and also 'chi2_nu' and 'tau_int' of the fit.
    """

    time_data = np.asarray(time_data, dtype=np.float64)
    lag_data = np.asarray(lag_data, dtype=np.float64)
    ht_normed = np.asarray(ht_normed, dtype=np.float64)

    # Index of the station of every point, as in the fit
    station_index = np.unique(np.asarray(station_data), return_inverse=True)[1]
    n_stations = int(np.max(station_index)) + 1

    weights = entry_fit['weights']
    cost_best = entry_fit['cost']
    scale = entry_fit['scale']
    lower_bounds, upper_bounds = _logAlphaBetaBounds()

    def _cost(log_alpha, log_beta):
        """ Cost and the entry velocity of a model, with dv and the offsets of the stations fitted. """

        lag_model, _, outside = _entryLagModel(np.exp(log_alpha), np.exp(log_beta), ht_normed, length_scale)
        if outside > ENTRY_LAG_MAX_OUTSIDE:
            return np.inf, np.nan

        dv, _, residuals = _entryLagLinear(lag_data - lag_model, time_data, weights, station_index, \
            n_stations)

        return np.sum(weights*residuals**2), entry_fit['v_ref'] + dv - entry_fit['air_speed']


    threshold = scipy.stats.chi2.ppf(ci/100.0, 1)

    # Walk far enough to also show the 3 sigma level of two parameters
    walk_threshold = max(threshold, scipy.stats.chi2.ppf(0.9973, 2))

    profile = {'ci': ci, 'threshold': threshold, 'sigma2': scale, 'chi2_nu': entry_fit['chi2_nu'], \
        'tau_int': entry_fit['tau_int']}

    # Steps of half the Wald errors of ln alpha and ln beta
    try:
        wald_std = np.sqrt(np.diag(np.linalg.inv(entry_fit['jac'].T.dot(entry_fit['jac'])))*scale)
    except np.linalg.LinAlgError:
        wald_std = np.full(2, np.nan)

    steps = [float(np.clip(0.5*std, 1e-3, 0.5)) if (np.isfinite(std) and (std > 0)) else 0.1 \
        for std in wald_std]

    x_best = [np.log(alpha), np.log(beta)]
    v_init_best = entry_fit['v_ref'] + entry_fit['dv'] - entry_fit['air_speed']

    # The profile of ln beta follows the floor of the valley in ln alpha from the best fit
    valley = {'log_alpha': x_best[0]}

    def _profilePoint(idx, log_value):
        """ Minimize the cost over the other parameter with ln alpha (idx 0) or ln beta (idx 1) fixed, and
            return the cost, ln alpha, ln beta and the entry velocity there.
        """

        if idx == 1:

            def _costAlpha(log_alpha):
                """ Cost of the model with the fixed ln beta at the given ln alpha. """

                return _cost(log_alpha, log_value)[0]


            # Minimize over ln alpha within a bracket around the previous point of the valley, which fails if
            #   no model around it reaches the points
            try:
                log_alpha = scipy.optimize.minimize_scalar(_costAlpha, bracket=(valley['log_alpha'] - 0.05, \
                    valley['log_alpha'] + 0.05), tol=1e-6).x
            except (ValueError, RuntimeError):
                log_alpha = np.nan

            # If the valley leaves the bounds of alpha of the fit, minimize between the previous point and
            #   the bound which it crosses
            if (log_alpha < lower_bounds[0]) or (log_alpha > upper_bounds[0]):

                edge = lower_bounds[0] if (log_alpha < lower_bounds[0]) else upper_bounds[0]
                inner = float(np.clip(valley['log_alpha'], lower_bounds[0], upper_bounds[0]))
                log_alpha = edge

                if inner != edge:
                    res = scipy.optimize.minimize_scalar(_costAlpha, method='bounded', \
                        bounds=(min(inner, edge), max(inner, edge)), options={'xatol': 1e-6})
                    log_alpha = res.x

            # If the minimization did not find a model which reaches the points, search for one over the
            #   bounds of alpha and refine the best one between its neighbours
            if (not np.isfinite(log_alpha)) or (not np.isfinite(_costAlpha(log_alpha))):

                log_alpha_grid = np.linspace(lower_bounds[0], upper_bounds[0], 61)
                grid_costs = np.array([_costAlpha(log_alpha_grid_point) for log_alpha_grid_point \
                    in log_alpha_grid])

                if np.any(np.isfinite(grid_costs)):

                    k = int(np.argmin(grid_costs))
                    res = scipy.optimize.minimize_scalar(_costAlpha, method='bounded', \
                        bounds=(log_alpha_grid[max(k - 1, 0)], log_alpha_grid[min(k + 1, \
                        len(log_alpha_grid) - 1)]), options={'xatol': 1e-6})
                    log_alpha = res.x if (res.fun < grid_costs[k]) else log_alpha_grid[k]

                else:

                    # No alpha gives a model which reaches the points, so the cost stays infinite
                    log_alpha = valley['log_alpha']

            valley['log_alpha'] = log_alpha
            log_beta = log_value

        else:

            # Minimize over ln beta from a grid, refined between the neighbours of its best point
            log_beta_grid = np.linspace(lower_bounds[1], upper_bounds[1], 25)
            grid_costs = np.array([_cost(log_value, log_beta)[0] for log_beta in log_beta_grid])
            k = int(np.argmin(grid_costs))
            res = scipy.optimize.minimize_scalar(lambda log_beta: _cost(log_value, log_beta)[0], \
                method='bounded', bounds=(log_beta_grid[max(k - 1, 0)], \
                log_beta_grid[min(k + 1, len(log_beta_grid) - 1)]), options={'xatol': 1e-4})
            log_alpha = log_value
            log_beta = res.x if (res.fun < grid_costs[k]) else log_beta_grid[k]

        cost, v_init = _cost(log_alpha, log_beta)

        return cost, log_alpha, log_beta, v_init


    for idx, name in enumerate(['alpha', 'beta']):

        points = [(cost_best, x_best[0], x_best[1], v_init_best)]
        limits = []
        at_bound = []
        edge_models = []

        # Walk from the best fit to each side, until the rise passes walk_threshold or the bound is reached
        for direction in [-1, 1]:

            valley['log_alpha'] = x_best[0]
            bound = lower_bounds[idx] if direction < 0 else upper_bounds[idx]

            side_values = [x_best[idx]]
            side_points = [points[0]]
            step = steps[idx]
            while ((side_points[-1][0] - cost_best)/scale < walk_threshold) \
                and (direction*(bound - side_values[-1]) > 0):

                # Grow the step if the cost rises much slower than expected from the Wald error
                if len(side_values) > ENTRY_LAG_PROFILE_SLOW_STEPS:
                    step = min(2*step, 0.5)

                log_value = float(np.clip(side_values[-1] + direction*step, lower_bounds[idx], \
                    upper_bounds[idx]))
                side_values.append(log_value)
                side_points.append(_profilePoint(idx, log_value))

            points += side_points[1:]

            # The profile reached the bound without crossing the threshold
            delta_chi2 = (np.array([point[0] for point in side_points]) - cost_best)/scale
            above = np.where(delta_chi2 > threshold)[0]
            if len(above) == 0:
                limits.append(np.exp(side_values[-1]))
                at_bound.append(True)
                continue

            # Refine the crossing of the threshold with Brent's method
            k = above[0]
            valley['log_alpha'] = x_best[0]
            try:
                crossing = scipy.optimize.brentq(lambda log_value: (_profilePoint(idx, log_value)[0] \
                    - cost_best)/scale - threshold, min(side_values[k - 1], side_values[k]), \
                    max(side_values[k - 1], side_values[k]), xtol=1e-5)

            except ValueError:
                crossing = side_values[k - 1] + (threshold - delta_chi2[k - 1])*(side_values[k] \
                    - side_values[k - 1])/(delta_chi2[k] - delta_chi2[k - 1])

            limits.append(np.exp(crossing))
            at_bound.append(False)

            # Keep the model at the crossing, or just inside it, for the intervals of derived quantities, and
            #   the inner point of the walk if neither is within the threshold
            edge_point, edge_delta = side_points[k - 1], delta_chi2[k - 1]
            for log_edge in [crossing, crossing - direction*1e-4]:

                point = _profilePoint(idx, log_edge)
                delta_edge = (point[0] - cost_best)/scale
                if delta_edge <= threshold:
                    edge_point, edge_delta = point, delta_edge
                    break

            edge_models.append([np.exp(edge_point[1]), np.exp(edge_point[2]), edge_point[3], edge_delta])

        # Sort the points of the profile by the value of the parameter
        points.sort(key=lambda point: point[idx + 1])

        profile[name + '_lower'], profile[name + '_upper'] = limits
        profile[name + '_lower_at_bound'], profile[name + '_upper_at_bound'] = at_bound
        profile[name + '_profile'] = (np.exp([point[idx + 1] for point in points]), \
            (np.array([point[0] for point in points]) - cost_best)/scale)

        # Alpha, beta and the entry velocity of every point of the profile
        profile[name + '_profile_models'] = np.array([[np.exp(point[1]), np.exp(point[2]), point[3]] \
            for point in points])

        # Alpha, beta, the entry velocity and the chi-squared rise at the points next to the crossings
        profile[name + '_edge_models'] = np.array(edge_models).reshape(-1, 4)


    ### Compute the chi-squared rise on a 2D grid of alpha and beta ###

    profile['grid_alpha'] = np.array([])
    profile['grid_beta'] = np.array([])
    profile['grid_delta_chi2'] = np.zeros((0, 0))
    profile['grid_v_init'] = np.zeros((0, 0))
    profile['corr_log'] = np.nan

    if n_grid > 0:

        # Cover the extent of the profiles with a margin, within the bounds of the fit
        log_grids = []
        for idx, name in enumerate(['alpha', 'beta']):
            log_lower, log_upper = np.log(profile[name + '_profile'][0][[0, -1]])
            margin = 0.15*(log_upper - log_lower)
            log_grids.append(np.linspace(max(log_lower - margin, lower_bounds[idx]), \
                min(log_upper + margin, upper_bounds[idx]), n_grid))

        log_alpha_grid, log_beta_grid = log_grids

        # The offsets and dv are fitted in closed form, so every point is a single evaluation
        grid_delta_chi2 = np.zeros((n_grid, n_grid))
        grid_v_init = np.zeros((n_grid, n_grid))
        for i, log_beta in enumerate(log_beta_grid):
            for j, log_alpha in enumerate(log_alpha_grid):

                cost, v_init = _cost(log_alpha, log_beta)
                grid_delta_chi2[i, j] = (cost - cost_best)/scale
                grid_v_init[i, j] = v_init

        # Correlation of ln alpha and ln beta, weighted by the likelihood
        weights_grid = np.exp(-0.5*(grid_delta_chi2 - np.min(grid_delta_chi2)))
        log_alpha_mesh, log_beta_mesh = np.meshgrid(log_alpha_grid, log_beta_grid)
        cov_log = np.cov(np.vstack([log_alpha_mesh.ravel(), log_beta_mesh.ravel()]), \
            aweights=weights_grid.ravel())

        profile['grid_alpha'] = np.exp(log_alpha_grid)
        profile['grid_beta'] = np.exp(log_beta_grid)
        profile['grid_delta_chi2'] = grid_delta_chi2
        profile['grid_v_init'] = grid_v_init
        profile['corr_log'] = cov_log[0, 1]/np.sqrt(cov_log[0, 0]*cov_log[1, 1])

        # Extend the intervals over the points of the grid within them, as the walk of a profile stops at the
        #   first crossing of the threshold and the likelihood can have another valley beyond it
        region_alpha, region_beta, _ = profileRegionAlphaBetaLag(profile, ci=ci, n_params=1)
        for idx, name, values in [(0, 'alpha', region_alpha), (1, 'beta', region_beta)]:

            if np.min(values) < profile[name + '_lower']:
                profile[name + '_lower'] = np.min(values)
                profile[name + '_lower_at_bound'] = np.log(np.min(values)) <= lower_bounds[idx] + 1e-6

            if np.max(values) > profile[name + '_upper']:
                profile[name + '_upper'] = np.max(values)
                profile[name + '_upper_at_bound'] = np.log(np.max(values)) >= upper_bounds[idx] - 1e-6

    ### ###


    # Find the lowest point of the profiles and the grid, which is below the given fit if it ended in a local
    #   minimum
    candidates = [(0.0, (alpha, beta, v_init_best))]
    for name in ['alpha', 'beta']:
        i_min = np.argmin(profile[name + '_profile'][1])
        candidates.append((profile[name + '_profile'][1][i_min], \
            tuple(profile[name + '_profile_models'][i_min])))

    if profile['grid_delta_chi2'].size:
        i_min, j_min = np.unravel_index(np.argmin(profile['grid_delta_chi2']), \
            profile['grid_delta_chi2'].shape)
        candidates.append((profile['grid_delta_chi2'][i_min, j_min], (profile['grid_alpha'][j_min], \
            profile['grid_beta'][i_min], profile['grid_v_init'][i_min, j_min])))

    candidates.sort(key=lambda candidate: candidate[0])
    profile['min_delta_chi2'], profile['min_model'] = candidates[0]

    return profile



def _profileCostAt(fixed_value, fixed_idx, free_log_guess, v_normed, ht_normed, sigma_v_normed,
        free_bounds, loss, f_scale, fast=False):
    """ Cost at a fixed alpha or beta value, re-optimizing the other (profiled-out) parameter -
        the building block profileAlphaBeta() walks its grid with.

    Arguments:
        fixed_value: [float] The value (linear space) the profiled parameter is held at.
        fixed_idx: [int] 0 if fixed_value is alpha (profiling out beta), 1 if it is beta
            (profiling out alpha) - matches _dynResiduals()'s x = (ln alpha, ln beta) ordering.
        free_log_guess: [float] Starting guess (ln space) for the other, re-optimized parameter.
        v_normed: [ndarray] Normalized velocity in (0, 1).
        ht_normed: [ndarray] Height normalized to HT_NORM_CONST.
        sigma_v_normed: [float] Normalized velocity uncertainty.
        free_bounds: [tuple] (log_lower, log_upper) bounds for the re-optimized parameter.
        loss: [str] scipy.optimize.least_squares loss.
        f_scale: [float] Robust loss scale.

    Keyword arguments:
        fast: [bool] Passed through to _dynResiduals()/alphaBetaVelocityNormed() - see the
            latter's docstring. This is called once per re-optimization at every profile grid
            point, itself iterated by scipy.optimize.least_squares - one of the largest
            beneficiaries of fast=True in this module (profileAlphaBeta() calls this O(n_grid)
            times, each a full least_squares run).

    Return:
        (cost, log_free_at_optimum): [tuple] res.cost, and the optimized log-parameter so the
            next grid point can warm-start from it.
    """

    def _resid(log_free):
        x = np.empty(2)
        x[fixed_idx] = np.log(fixed_value)
        x[1 - fixed_idx] = log_free[0]
        return _dynResiduals(x, v_normed, ht_normed, sigma_v_normed, fast=fast)

    res = scipy.optimize.least_squares(_resid, [free_log_guess], \
        bounds=([free_bounds[0]], [free_bounds[1]]), loss=loss, f_scale=f_scale)

    return res.cost, res.x[0]


def _costAtOptimum(alpha, beta, v_normed, ht_normed, sigma_v_normed, log_bounds, loss, f_scale,
        fast=False):
    """ Cost at the already-known joint optimum (alpha, beta) - profileAlphaBeta()'s cost_best
        reference point every grid point's delta_cost is measured against.

        Direct residual evaluation for loss='soft_l1' (the default, and the only loss this module
        itself ever actually exercises), via scipy's own published cost formula for that loss:
        F = 0.5*sum(f_scale**2 * 2*(sqrt(1 + (f/f_scale)**2) - 1)) - verified numerically against
        scipy.optimize.least_squares's own reported .cost (matched to solver tolerance, ~1e-7).
        This avoids re-optimizing to reach an answer that is already known: (alpha, beta) being
        the joint optimum means fixing beta at its own value and re-optimizing alpha must
        converge back to alpha itself, so a fresh optimizer call here only ever confirms, never
        discovers, its own starting point.

        Falls back to re-optimizing (via _profileCostAt(), fixing beta and re-optimizing alpha
        from its own value - the same mechanism every grid point already uses) for any other
        loss, including a custom callable - not worth hand-rolling scipy's other built-in loss
        formulas, or trying to support arbitrary callables generically, just to skip what is, for
        those, already a near-free optimizer call starting exactly at its own answer.
    """

    if loss == 'soft_l1':
        residuals = _dynResiduals(np.log([alpha, beta]), v_normed, ht_normed, sigma_v_normed, \
            fast=fast)
        z = (residuals/f_scale)**2
        return 0.5*np.sum(f_scale**2 * 2.0*(np.sqrt(1.0 + z) - 1.0))

    cost, _ = _profileCostAt(beta, 1, np.log(alpha), v_normed, ht_normed, sigma_v_normed,
        (log_bounds[0][0], log_bounds[1][0]), loss, f_scale, fast=fast)
    return cost


def _profileOneParameter(fixed_idx, value_hat, other_hat, cost_best, s2, v_normed, ht_normed,
        sigma_v_normed, log_bounds, ci, n_grid, loss, f_scale, fast=False):
    """ Profile scan and ci-crossing search for one parameter (0=alpha, 1=beta), profiling out the
        other. See profileAlphaBeta() for the full picture.

    Keyword arguments:
        fast: [bool] Passed through to _profileCostAt() - see its docstring.

    Return:
        profile: [dict] 'ci_lower', 'ci_upper' (float or None - None if the search grid, bounded
            by ALPHA_BETA_BOUNDS, never crosses the threshold on that side), 'grid',
            'delta_cost', 'delta_threshold', 'value_hat', 'cost_best', 's2', 'ci', 'other_hat',
            'other_profile' (see profileAlphaBeta()'s Return section for the last three - they
            exist so a caller (e.g. plotProfileAlphaBeta()) can reconstruct trajectories from the
            profile without re-running the sweep).
    """

    # Grid spans ALPHA_BETA_BOUNDS in LOG space (matching the space the model/optimizer/cov_log
    #   all actually work in), via linspace + exp rather than converting to linear bounds first and
    #   letting geomspace re-derive that same log-space grid internally - mathematically identical
    #   (geomspace IS exp(linspace(log(...)))), just avoids a redundant log/exp round trip and
    #   keeps the log-space nature of the computation explicit.
    log_lo = log_bounds[0][fixed_idx] + np.log(1.001)
    log_hi = log_bounds[1][fixed_idx] + np.log(0.999)
    log_grid = np.linspace(log_lo, log_hi, n_grid)
    grid = np.exp(log_grid)

    other_bounds = (log_bounds[0][1 - fixed_idx], log_bounds[1][1 - fixed_idx])

    # chi2(df=1) quantile at `ci`: exact closed form via the standard normal quantile (same
    #   scipy.stats-avoiding trick as _gaussianEllipsePoints()/alpha_ci_wald_lower):
    #   chi2.ppf(p, df=1) = ndtri(0.5 + p/2)^2
    chi2_1 = scipy.special.ndtri(0.5 + ci/200.0)**2

    delta = np.empty_like(grid)

    # The other (profiled-out) parameter's re-optimized value at each grid point, in LINEAR
    #   space - free (already computed by the walk below to warm-start the next grid point, and
    #   previously discarded once used). Lets a caller (e.g. plotProfileAlphaBeta()) reconstruct
    #   trajectories along the profile ridge without re-optimizing - see profileAlphaBeta()'s
    #   Statistical caveats for exactly what that ridge does and does not represent.
    other_profile = np.empty_like(grid)

    center_i = np.searchsorted(grid, value_hat)

    # Walk outward from value_hat in both directions, warm-starting each step from the last
    log_other = np.log(other_hat)
    for i in range(center_i, len(grid)):
        cost, log_other = _profileCostAt(grid[i], fixed_idx, log_other, v_normed, ht_normed,
            sigma_v_normed, other_bounds, loss, f_scale, fast=fast)
        delta[i] = 2.0*(cost - cost_best)/s2
        other_profile[i] = np.exp(log_other)

    log_other = np.log(other_hat)
    for i in range(center_i - 1, -1, -1):
        cost, log_other = _profileCostAt(grid[i], fixed_idx, log_other, v_normed, ht_normed,
            sigma_v_normed, other_bounds, loss, f_scale, fast=fast)
        delta[i] = 2.0*(cost - cost_best)/s2
        other_profile[i] = np.exp(log_other)

    below, above = grid < value_hat, grid > value_hat

    # Split the grid into the two sides of value_hat, with (value_hat, 0.0) prepended as an exact
    #   anchor (delta_cost is 0 there by construction - cost_best IS the cost at value_hat with the
    #   other parameter re-optimized). The anchor is the inner end of the root-finding bracket
    #   below: it is guaranteed to sit below the threshold (delta=0 < chi2_1), so it always pairs
    #   with the first grid point above the threshold to give a valid sign-changing bracket, even
    #   when the whole CI is narrower than one grid cell and no grid point falls inside it (the
    #   common case for a well-constrained parameter, since n_grid is fixed across the full
    #   ALPHA_BETA_BOUNDS span).
    log_value_hat = np.log(value_hat)
    log_grid_above = np.concatenate(([log_value_hat], log_grid[above]))
    delta_above = np.concatenate(([0.0], delta[above]))

    log_grid_below = np.concatenate(([log_value_hat], log_grid[below][::-1]))
    delta_below = np.concatenate(([0.0], delta[below][::-1]))

    # Locate each CI edge by root-finding the crossing 2*(cost - cost_best)/s2 = chi2_1 in LOG
    #   space, rather than linearly interpolating between the two bracketing grid points. Linear
    #   interpolation of the (locally near-quadratic) delta across one grid cell systematically
    #   MIS-estimates the crossing whenever the true CI is narrower than the grid spacing - and
    #   because n_grid is fixed across the full ALPHA_BETA_BOUNDS span, that is the common case for
    #   a well-constrained parameter (its CI can be many times narrower than one cell), where the
    #   only bracket available is [value_hat (delta=0), first grid point (delta>>chi2_1)] and the
    #   linear estimate comes out roughly 2x too narrow. A bracketed root-find between the
    #   value_hat anchor (delta=0 < chi2_1 by construction) and the first grid point above the
    #   threshold re-optimizes the profiled-out parameter at each trial value, so the reported edge
    #   is accurate to the solver tolerance regardless of grid resolution (n_grid then only sets
    #   the resolution of the returned 'grid'/'delta_cost' arrays and the initial bracketing scan).
    def _delta_minus_threshold(log_value):
        cost, _ = _profileCostAt(np.exp(log_value), fixed_idx, np.log(other_hat), v_normed,
            ht_normed, sigma_v_normed, other_bounds, loss, f_scale, fast=fast)
        return 2.0*(cost - cost_best)/s2 - chi2_1

    def _crossing(log_side, delta_side):
        # First grid point (index i, i >= 1 since delta_side[0] is the anchor's 0.0) whose delta
        #   exceeds the threshold; None if the grid never crosses (the data does not constrain
        #   this side within ALPHA_BETA_BOUNDS). The root is bracketed by the value_hat anchor
        #   (delta 0, so _delta_minus_threshold < 0) and that grid point.
        idx = np.where(delta_side > chi2_1)[0]
        if not len(idx):
            return None

        i = idx[0]
        try:
            log_ci = scipy.optimize.brentq(_delta_minus_threshold, log_value_hat, log_side[i],
                xtol=1e-8)
        except ValueError:
            # Re-optimizing inside _delta_minus_threshold warm-starts from other_hat rather than
            #   from the swept trajectory, so it can land marginally below the threshold right at
            #   log_side[i] and break the bracket sign. Fall back to interpolating the (ascending)
            #   swept deltas - never worse than the original linear estimate.
            log_ci = np.interp(chi2_1, [delta_side[i - 1], delta_side[i]],
                [log_side[i - 1], log_side[i]])
        return np.exp(log_ci)

    ci_upper = _crossing(log_grid_above, delta_above)
    ci_lower = _crossing(log_grid_below, delta_below)

    return {
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'grid': grid,
        'delta_cost': delta,
        'delta_threshold': chi2_1,
        'value_hat': value_hat,
        'cost_best': cost_best,
        's2': s2,
        'ci': ci,
        'other_hat': other_hat,
        'other_profile': other_profile,
    }


def profileAlphaBeta(v_data, ht_data, fit, sigma_v=None, ci=95.0, param='both', n_grid=250,
        loss='soft_l1', f_scale=2.0, verbose=False, fast=False):
    """ Profile-based confidence interval(s) for alpha and/or beta - commonly called a "profile
        likelihood" interval (the name this module's own function/parameter names use, since
        that's the term to search for), though see the Statistical caveats below for the precise
        sense in which that name applies: what is actually profiled by default is the robust
        (soft-L1) fitting COST, not a Gaussian log-likelihood. A more direct alternative to
        fitAlphaBeta(..., estimate_errors=True)'s alpha_ci_wald_lower/upper -
        beta_ci_wald_lower/upper for when a parameter is weakly constrained (see the Statistical
        caveats on those fields) and the local quadratic (Wald) approximation those are built on
        no longer describes the cost surface out at the interval's edge.

        Deliberately a SEPARATE function from estimate_errors=True, not a replacement for it and
        not an additional field mixed into the same `errors` dict: they answer different
        questions and mixing them would make it much harder to tell which quantities describe a
        local approximation and which describe the real cost surface. estimate_errors=True's
        cov_log/corr_log/alpha_std_rel/beta_std_rel describe the LOCAL curvature of the cost
        function at the optimum - correct and useful for propagating small uncertainties
        regardless of anything below (see fitAlphaBeta()'s docstring). This function instead asks
        "what parameter values are actually compatible with the data" by walking the real
        (soft-L1) cost function directly: for a grid of values of the profiled parameter, the
        OTHER parameter is re-optimized at each point (not held at its joint-fit value), and the
        interval is read off wherever the resulting, appropriately-scaled cost rise crosses the
        chi-squared(df=1) threshold for `ci` - the same likelihood-ratio-style construction as a
        true profile likelihood, applied to a cost surface rather than a log-likelihood (see the
        Statistical caveats below), not an extrapolated quadratic. It makes no assumption that
        the region it reports on is well-described by a quadratic bowl, which is exactly the
        assumption that breaks down for a weakly-constrained
        beta in this model (see fitAlphaBeta()'s docstring for the measured extent of this - up
        to ~140x on the upper bound in the scenarios checked, growing sharply and systematically
        once beta_std_rel is large, more so than for alpha given how much more nonlinearly beta
        enters the model).

        Unlike estimate_errors=True, this does NOT require the original fit to have used
        method='robust' - it re-derives its own cost from `fit`'s (alpha, beta) directly via the
        same velocity-residual definition minimizeAlphaBetaRobust() uses, regardless of whether
        that point estimate itself came from method='q4' or 'robust'.

        Not free: this is an optimization sweep (n_grid re-fits per profiled parameter), not the
        few extra refits estimate_errors=True's sigma_v_init correction costs.

    Arguments:
        v_data: [ndarray] Velocity data (m/s), as given to fitAlphaBeta().
        ht_data: [ndarray] Height data (m), as given to fitAlphaBeta().
        fit: [tuple] (v_init, alpha, beta), exactly as returned by fitAlphaBeta(). v_init is held
            fixed at this value throughout - it is not itself profiled (matching
            estimate_errors=True's own treatment of a given v_init as exactly known unless
            sigma_v_init says otherwise; there is no equivalent of that correction here).

    Keyword arguments:
        sigma_v: [float] Velocity uncertainty (m/s), same role as fitAlphaBeta()'s sigma_v. If
            None, estimated the same way (_estimateSigmaV(), using `fit`'s own alpha/beta as the
            reference model) - the same *effective*, not purely instrumental, uncertainty
            fitAlphaBeta() itself would derive. NOTE: to compare this interval against
            fitAlphaBeta(estimate_errors=True)'s Wald interval on the identical noise scaling (s2),
            pass the SAME sigma_v (and loss/f_scale) here that produced that fit - left to default,
            the re-estimated sigma_v can differ from the fit's, giving a different s2 and hence a
            differently-scaled interval.
        ci: [float] Confidence level (0-100, e.g. 95, the default).
        param: [str] 'alpha', 'beta' (most relevant given beta is the parameter this matters for
            in this model - see above), or 'both' (default).
        n_grid: [int] Number of grid points spanning ALPHA_BETA_BOUNDS for the profiled
            parameter - determines resolution/cost, not a statistical stability budget like
            plotAlphaBeta()'s band_samples (this is a deterministic sweep, not a sampling one).
        loss: [str] scipy.optimize.least_squares loss - should match what the original fit used,
            for the comparison against its cost to be meaningful.
        f_scale: [float] Robust loss scale - should match the original fit's.
        verbose: [bool] If True, print the profiled parameter(s) and their profile-based CI,
            mirroring fitAlphaBeta(..., verbose=True)'s style.
        fast: [bool] Passed through everywhere this sweep inverts velocity - see
            alphaBetaVelocityNormed()'s docstring. This sweep runs O(n_grid) full
            scipy.optimize.least_squares fits per profiled parameter, each iterating
            _dynResiduals() many times, so fast=True is one of the largest speedups available in
            this module.

    Return:
        profile: [dict] Keyed by whichever of 'alpha'/'beta' was requested via `param`, each a
            dict with:
                'ci_lower', 'ci_upper': [float or None] The profile-based interval at `ci` - None
                    if the grid (bounded by ALPHA_BETA_BOUNDS) never reaches where the cost
                    crosses the threshold on that side - the data does not constrain that side at
                    all within the model's own physical bounds, which is itself informative.
                'grid': [ndarray] The parameter grid the profile was evaluated on.
                'delta_cost': [ndarray] 2*(cost - cost_best)/s2 at each grid point - compare
                    directly against 'delta_threshold' (or recompute at a different `ci` via
                    ndtri(0.5 + ci/200)**2) to reproduce ci_lower/ci_upper, or re-derive the
                    interval at a different `ci` without re-running the sweep.
                'delta_threshold': [float] The chi2(df=1) quantile at `ci` that 'delta_cost' is
                    compared against - exactly ndtri(0.5 + ci/200)**2, exposed directly so a
                    caller can e.g. plt.axhline(profile['beta']['delta_threshold']) without
                    recomputing it.
                'value_hat': [float] The point estimate this parameter was profiled around
                    (`fit`'s alpha or beta, echoed back so this dict is self-contained).
                'cost_best', 's2': [float] The same reduced-chi-square scale
                    fitAlphaBeta(estimate_errors=True) uses for cov_log, so this comparison uses
                    an identical noise-scaling convention - any difference from the Wald interval
                    is due to the cost surface's actual shape, not a different assumption about
                    how much to trust sigma_v.
                'ci': [float] Echoes the `ci` argument, so this dict is self-contained.
                'other_hat': [float] The OTHER parameter's value at the joint optimum (`fit`'s
                    beta if this is the alpha profile, alpha if this is the beta profile) - the
                    (value_hat, other_hat) point, with no re-optimization needed to get it.
                'other_profile': [ndarray] The other parameter, re-optimized at each point of
                    'grid' (same index alignment) - (grid, other_profile) traces the
                    profile-likelihood "ridge" through (alpha, beta) space. Free: already
                    computed by the sweep to warm-start each grid step, previously discarded
                    once used. Lets a caller (e.g. plotProfileAlphaBeta()) reconstruct
                    trajectories at any grid point, including inside [ci_lower, ci_upper],
                    without re-optimizing - see the Statistical caveats below for what that
                    reconstruction does and does not represent.

    Statistical caveats:
        - On the name: despite "profile likelihood" being the term used throughout this function
          and its helpers (it's the standard, searchable name for this class of method - fix one
          parameter, re-optimize the rest, threshold the objective's rise), what is actually
          profiled by default is a robust soft-L1 fitting COST, not a Gaussian log-likelihood.
          There is no claim anywhere in this module that soft_l1's cost equals -2*ln(L) for any
          actual noise model - it doesn't, for any noise model. The chi2(df=1) threshold used to
          read off ci_lower/ci_upper is exactly Wilks'-theorem-justified only for a genuine
          Gaussian-noise least-squares cost (loss='linear'); under the default soft-L1 loss it is
          the same well-motivated but not perfectly theoretically clean comparison this module
          already accepts elsewhere for its Gauss-Newton covariances (e.g.
          fitAlphaBetaLightCurve()'s alpha_std_rel/beta_std_rel) - a useful, standard engineering
          approximation, not a literal likelihood-ratio test. There is a second, milder
          approximation on top of that: because the delta is scaled by the estimated s2 (below),
          the self-consistent threshold is the F(1, dof) quantile rather than chi2(df=1); the two
          coincide only in the large-dof limit (chi2(df=1) = dof*F(1, dof) as dof -> inf), so for
          the point counts typical here (dof in the tens to hundreds) the difference is small, but
          the interval is very slightly optimistic at small dof.
        - Not a replacement for estimate_errors=True's cov_log/corr_log/alpha_std_rel/
          beta_std_rel - those remain the right tool for LOCAL error propagation (e.g. via the
          delta method into some other derived quantity). This answers a different question
          (which parameter values are compatible with the data at all) and does not itself
          provide a covariance or support sampling a band/fan/ellipse - plotAlphaBeta() still
          needs cov_log for that, regardless of what this function reports.
        - Uses the same reduced-chi-square (s2) rescaling as estimate_errors=True, which assumes
          sigma_v is an effective, self-derived scale rather than a true instrumental one (see
          fitAlphaBeta()'s sigma_v caveat).
        - 'other_profile' lets a caller reconstruct trajectories along the profile ridge, e.g.
          plotProfileAlphaBeta()'s trajectories panel. That ridge, restricted to
          [ci_lower, ci_upper], does stay inside the JOINT (alpha, beta) confidence region at the
          same `ci`: a point with 1-dof delta_cost <= chi2(ci, df=1) automatically also satisfies
          the looser 2-dof threshold chi2(ci, df=2) that a true joint region would use (e.g.
          3.841 vs 5.991 at ci=95). But it is NOT that joint region's boundary or an envelope of
          it - it is the 1-dimensional locus of the other parameter's re-optimized value, clipped
          to where THIS parameter alone stays within its own marginal (df=1) profile CI. Do not
          describe the resulting trajectory family as "the `ci`% confidence region for the
          trajectory" - it is narrower than that, and shows how far the fit trajectory moves if
          this parameter is pushed to the edge of what's marginally still compatible with the
          data, not a joint-probability statement about (alpha, beta) together.
    """

    v_init, alpha, beta = fit

    if param not in ('alpha', 'beta', 'both'):
        raise ValueError("param must be 'alpha', 'beta', or 'both', got {!r}.".format(param))

    if not (0 < ci < 100):
        raise ValueError("ci must be strictly between 0 and 100, got {!r}.".format(ci))

    v_normed = v_data/v_init
    ht_normed = ht_data/HT_NORM_CONST

    if sigma_v is None:
        sigma_v = _estimateSigmaV(alpha, beta, v_normed, ht_normed, v_init, fast=fast)

    sigma_v_normed = sigma_v/v_init
    log_bounds = _logAlphaBetaBounds()

    # Reference cost at the joint optimum - see _costAtOptimum()'s docstring for why this is a
    #   direct evaluation rather than a redundant re-optimization for the default loss.
    cost_best = _costAtOptimum(alpha, beta, v_normed, ht_normed, sigma_v_normed, log_bounds, loss,
        f_scale, fast=fast)

    dof = max(len(v_data) - 2, 1)
    s2 = 2.0*cost_best/dof

    profile = {}

    if param in ('alpha', 'both'):
        profile['alpha'] = _profileOneParameter(0, alpha, beta, cost_best, s2, v_normed,
            ht_normed, sigma_v_normed, log_bounds, ci, n_grid, loss, f_scale, fast=fast)

    if param in ('beta', 'both'):
        profile['beta'] = _profileOneParameter(1, beta, alpha, cost_best, s2, v_normed,
            ht_normed, sigma_v_normed, log_bounds, ci, n_grid, loss, f_scale, fast=fast)

    # cost_best is the cost at the passed-in (alpha, beta), used as the profile's reference
    #   minimum. If a profiled-out re-optimization along the grid ever reaches a LOWER cost
    #   (delta_cost < 0), then (alpha, beta) was not actually the joint minimum of this loss -
    #   which happens when `fit` came from a different objective than `loss` (e.g. a method='q4'
    #   point estimate profiled under the default soft_l1): the reference is then off-minimum, so
    #   the interval is measured from an inflated baseline and can be biased/mis-centered. Warn
    #   rather than silently reporting it (pass a robust `fit` with matching loss/f_scale to avoid).
    min_delta = min(float(np.min(p['delta_cost'])) for p in profile.values())
    if min_delta < -1e-6:
        print("WARNING: a profiled re-optimization reached a lower cost than the reference "
            "(min delta_cost = {:.3g} < 0), so the supplied (alpha, beta) is not the joint "
            "minimum of loss='{:s}'. The profile CI is measured from an off-minimum reference and "
            "may be biased - pass a fit from fitAlphaBeta(method='robust') with the same "
            "loss/f_scale.".format(min_delta, loss))

    if verbose:
        print()
        print("--- profileAlphaBeta ---")

        for name, value in [('alpha', alpha), ('beta', beta)]:

            if name not in profile:
                continue

            p = profile[name]
            lo = "{:.6f}".format(p['ci_lower']) if p['ci_lower'] is not None else "<= lower bound"
            hi = "{:.6f}".format(p['ci_upper']) if p['ci_upper'] is not None else ">= upper bound"

            print("{:<21s}= {:.6f}  [{:.3g}% profile CI: {:s} - {:s}]".format(
                name, value, ci, lo, hi))

    return profile


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


def alphaBetaVelocityNormed(ht_normed, alpha, beta, v_eps=1e-10, fast=False):
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
        fast: [bool] If True, use the experimental LUT-accelerated closed-form inversion
            (alphaBetaVelocityNormedLUT()) instead of the brentq bracket below. Much faster in
            Monte Carlo/optimizer-heavy contexts (see test_AlphaBeta.py::testAlphaBetaVelocityNormedLUT
            for accuracy/speed numbers), but inherits its beta >~ 30 precision ceiling - a limit
            shared with this same brentq path (see the module comment above
            alphaBetaVelocityNormedLUT() for why). Default False keeps today's exact behavior.

    Return:
        v_normed: [ndarray] Normalized model velocity at each height.
    """

    if fast:
        return alphaBetaVelocityNormedLUT(ht_normed, alpha, beta, v_eps=v_eps)

    y_arr = np.atleast_1d(np.asarray(ht_normed, dtype=np.float64))
    v_out = np.empty_like(y_arr)

    v_lo, v_hi = v_eps, 1.0 - v_eps

    def _g(v, y_target):
        return alphaBetaHeightNormed(v, alpha, beta) - y_target

    # alphaBetaHeightNormed(v_hi/v_lo, alpha, beta) doesn't depend on y_target, so it's the same
    #   value on every iteration below - compute it once here (2 Ei() calls total) instead of
    #   inside the loop (2 Ei() calls per height point, most of which just get thrown away on a
    #   clip). _g(v_hi, y_t) <= 0 is exactly y_t >= y_hi (and symmetrically for v_lo) - only the
    #   comparison itself needs to happen per point now.
    y_hi = alphaBetaHeightNormed(v_hi, alpha, beta)
    y_lo = alphaBetaHeightNormed(v_lo, alpha, beta)

    for i, y_t in enumerate(y_arr):

        # Clip outside the invertible range
        if y_t >= y_hi:
            v_out[i] = v_hi

        elif y_t <= y_lo:
            v_out[i] = v_lo

        else:
            v_out[i] = scipy.optimize.brentq(_g, v_lo, v_hi, args=(y_t,), xtol=1e-12)

    return v_out


def alphaBetaVelocity(ht_data, alpha, beta, v_init, v_eps=1e-10, fast=False):
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
        fast: [bool] Passed through to alphaBetaVelocityNormed() - see its docstring.

    Return:
        vel_data: [ndarray or float] Velocity data (m/s).
    """

    # Allow both scalar and array height inputs
    scalar_input = np.isscalar(ht_data)
    if scalar_input:
        ht_data = np.array([ht_data])

    # Normalize the height and numerically invert Eq. (7) for the normalized velocity
    ht_normed = ht_data/HT_NORM_CONST
    vel_normed = alphaBetaVelocityNormed(ht_normed, alpha, beta, v_eps=v_eps, fast=fast)

    # Compute the velocity in m/s
    vel_data = vel_normed*v_init

    # Return a scalar if the input height was a scalar
    if scalar_input:
        return vel_data[0]

    return vel_data



def alphaBetaVelocityInterp(ht_data, alpha, beta, v_init):
    """ Compute the velocity given the height and alpha, beta parameters, as alphaBetaVelocity(), but by
        interpolating the height of the model at the nodes ENTRY_LAG_V_NODES of the normalized velocity
        (extended down to 1e-10), which is much faster for many heights. The normalized velocity is accurate
        to about 1e-5.

    Arguments:
        ht_data: [ndarray] Height data (m).
        alpha: [float] Ballistic coefficient.
        beta: [float] Mass loss parameter.
        v_init: [float] Initial velocity (m/s).

    Return:
        [ndarray] Velocity at the given heights (m/s).
    """

    # Nodes of the normalized velocity, extended down to almost zero
    v_nodes = np.concatenate([np.geomspace(1e-10, ENTRY_LAG_V_NODES[0], 200)[:-1], ENTRY_LAG_V_NODES])

    # Height of the model at the nodes, sorted by the height
    ht_nodes = alphaBetaHeightNormed(v_nodes, alpha, beta)
    finite = np.isfinite(ht_nodes)
    order = np.argsort(ht_nodes[finite])

    return v_init*np.interp(np.asarray(ht_data)/HT_NORM_CONST, ht_nodes[finite][order], \
        v_nodes[finite][order])



# ============================================================================================
# EXPERIMENTAL: LUT-accelerated inverse of Ei() for alphaBetaVelocityNormed()
#
# alphaBetaVelocityNormed() above finds v by bracketing scipy.optimize.brentq on the *forward*
# model alphaBetaHeightNormed() - each brentq call costs ~30-50 iterations, each evaluating Ei()
# twice, all inside a per-point Python loop. Eq. (7) can instead be inverted in closed form:
#
#   y = ln(2 alpha) + beta - ln(Delta),  Delta = Ei(beta) - Ei(beta v^2)
#   =>  v = sqrt( Ei^-1( Ei(beta) - 2 alpha exp(beta - y) ) / beta )
#
# The only missing piece is Ei^-1, which has no closed form. _InverseEiLUT below tabulates it once,
# in a log(x) <-> signed-log1p(Ei(x)) space where a modest number of monotone cubic (PCHIP) knots
# is enough to represent its huge dynamic range accurately, and every subsequent call just
# evaluates that small interpolator - turning O(iterations) transcendental root-finds per point
# into a couple of vectorized array ops for the whole height array at once. See
# test_AlphaBeta.py::testAlphaBetaVelocityNormedLUT for the accuracy/speed validation this
# experimental status is based on.
#
# Known limitation, shared with alphaBetaVelocityNormed(): for beta >~ 30, Ei(beta) is so much
# larger than Ei(beta v^2) (Ei(50) ~ 1e20) that Delta = Ei(beta) - Ei(beta v^2) loses precision to
# direct float64 cancellation. That happens in the *forward* model itself, so brentq is not a
# reliable reference there either - both implementations degrade together above this threshold.
# This is a limitation of the double-precision formulation, not of either root-finding strategy.
# ============================================================================================


def _signedLog1p(u):
    """ log(1 + |u|) with the sign of u restored - compresses Ei()'s huge dynamic range into
        something a small number of interpolation knots can represent.
    """

    return np.sign(u)*np.log1p(np.abs(u))


class _InverseEiLUT:
    """ Tabulates the inverse of scipy.special.expi(), for fast (experimental) vectorized use by
        alphaBetaVelocityNormedLUT(). See the module comment above for the motivation.
    """

    def __init__(self, x_min=1e-10, x_max=55.0, n_points=4097, n_dense=200_000):
        """
        Keyword arguments:
            x_min: [float] Lower end of the tabulated Ei() domain.
            x_max: [float] Upper end of the tabulated Ei() domain. 55.0 gives margin above the
                largest beta allowed by ALPHA_BETA_BOUNDS (50.0), since Ei^-1 is only ever
                evaluated at values of beta*v_normed**2 <= beta.
            n_points: [int] Number of PCHIP knots. 4097 keeps the max error in recovered v below
                ~2e-6 across the ALPHA_BETA_BOUNDS domain (excluding the beta >~ 30 cancellation
                regime described above, where that error is dominated by float64 precision, not
                table resolution).
            n_dense: [int] Resolution of the dense ground-truth grid the knots are fit against.
        """

        logx_dense = np.linspace(np.log(x_min), np.log(x_max), n_dense)
        ei_dense = scipy.special.expi(np.exp(logx_dense))
        s_dense = _signedLog1p(ei_dense)

        self.s_min = s_dense[0]
        self.s_max = s_dense[-1]

        s_grid = np.linspace(self.s_min, self.s_max, n_points)
        logx_grid = np.interp(s_grid, s_dense, logx_dense)

        self._interp = scipy.interpolate.PchipInterpolator(s_grid, logx_grid)

    def __call__(self, u):
        """ x = Ei^-1(u), vectorized. Values of u outside the tabulated range are clipped. """

        s = np.clip(_signedLog1p(u), self.s_min, self.s_max)

        return np.exp(self._interp(s))


# Ei^-1 doesn't depend on alpha/beta/y, only on the value being inverted, so a single table
#   (covering the full ALPHA_BETA_BOUNDS beta range) can be built once and reused by every
#   alphaBetaVelocityNormedLUT() call regardless of the (alpha, beta) it's invoked with.
_default_inverse_ei_lut = None


def getDefaultInverseEiLUT():
    """ Lazily build (once) and return the process-wide default _InverseEiLUT used by
        alphaBetaVelocityNormedLUT()/every fast=True call in this module.

        Public so a caller running many fast=True fits (e.g. a Monte Carlo/dynesty driver) can
        call this once up front to pay the ~5-6 ms build cost before the timing-sensitive part of
        a run starts, instead of it landing on whichever fast=True call happens to be first.
        Calling it is entirely optional otherwise - every fast=True code path in this module
        already calls it lazily on first use and reuses the same table afterwards, within one
        process. Each process builds (and caches) its own copy - e.g. under multiprocessing, every
        worker builds its table once independently, not once for the whole pool.
    """

    global _default_inverse_ei_lut

    if _default_inverse_ei_lut is None:
        _default_inverse_ei_lut = _InverseEiLUT()

    return _default_inverse_ei_lut


def alphaBetaVelocityNormedLUT(ht_normed, alpha, beta, v_eps=1e-10, lut=None):
    """ EXPERIMENTAL - LUT-accelerated alternative to alphaBetaVelocityNormed(). Inverts Eq. (7)
        in closed form using a tabulated inverse of Ei() instead of bracketing
        alphaBetaHeightNormed() with brentq per point - see the module comment above for the
        derivation and known limitations. Used by every fast=True code path in this module
        (alphaBetaVelocityNormed(..., fast=True) dispatches here); kept as a separate function,
        alongside the brentq-based alphaBetaVelocityNormed(), for direct benchmarking/validation
        (test_AlphaBeta.py::testAlphaBetaVelocityNormedLUT).

    Arguments:
        ht_normed: [float or ndarray] Height normalized to HT_NORM_CONST.
        alpha: [float] Ballistic coefficient.
        beta: [float] Mass loss parameter.

    Keyword arguments:
        v_eps: [float] Bracket margin around (0, 1), same convention as alphaBetaVelocityNormed().
        lut: [_InverseEiLUT] Table to use. Defaults to the lazily-built, process-wide table.

    Return:
        v_normed: [ndarray] Normalized model velocity at each height.
    """

    if lut is None:
        lut = getDefaultInverseEiLUT()

    y = np.atleast_1d(np.asarray(ht_normed, dtype=np.float64))

    v_lo, v_hi = v_eps, 1.0 - v_eps
    y_hi = alphaBetaHeightNormed(v_hi, alpha, beta)
    y_lo = alphaBetaHeightNormed(v_lo, alpha, beta)

    ei_beta = scipy.special.expi(beta)

    with np.errstate(over='ignore'):

        delta = 2*alpha*np.exp(beta - y)

        # Beta >= 10 is where Ei(beta) can be large enough that a tiny Delta (v close to 1) gets
        #   lost when computing u = Ei(beta) - Delta directly; a local Taylor expansion of Ei
        #   around beta avoids that cancellation for v close to 1 specifically. This is a
        #   different (and fixable) corner from the beta >~ 30 cancellation described in the
        #   module comment, which affects the whole formulation and both implementations equally.
        if beta >= 10.0:
            mask_asymptotic = delta < 1e-8*ei_beta
        else:
            mask_asymptotic = np.zeros_like(y, dtype=bool)

        x = np.empty_like(y)

        if np.any(mask_asymptotic):
            x[mask_asymptotic] = beta - delta[mask_asymptotic]*beta*np.exp(-beta)

        mask_lut = ~mask_asymptotic
        if np.any(mask_lut):
            x[mask_lut] = lut(ei_beta - delta[mask_lut])

        v_out = np.sqrt(np.clip(x, 0.0, None)/beta)

    # Clip outside the invertible range, exactly like alphaBetaVelocityNormed()
    v_out = np.where(y >= y_hi, v_hi, v_out)
    v_out = np.where(y <= y_lo, v_lo, v_out)

    return v_out


def _gaussianEllipsePoints(mean, cov, p, n_points=200):
    """ Points tracing the boundary of the 2D Gaussian ellipse enclosing probability mass p.
        Factored out as a standalone helper so other plotting functions (e.g. a future
        plotAlphaBetaSurvivalDiagram()) can reuse it for any 2-parameter covariance, not just
        (ln alpha, ln beta).

        The Mahalanobis radius follows the chi-squared distribution with 2 degrees of freedom,
        which has an exact closed form here (chi2(df=2) is Exponential(scale=2)):
        chi2.ppf(p, df=2) = -2*ln(1 - p) - used instead of scipy.stats.chi2.ppf() to avoid a new
        scipy submodule dependency.

    Arguments:
        mean: [ndarray] (2,) Center of the ellipse.
        cov: [ndarray] (2, 2) Covariance matrix.
        p: [float] Probability mass enclosed by the ellipse (0-1) - e.g. 0.6827 for the 2D
            "1-sigma" ellipse (NOT the same as a unit Mahalanobis distance - that would only
            enclose ~39% of the mass in 2D, a common point of confusion), or ci/100 for a ci%
            ellipse.

    Keyword arguments:
        n_points: [int] Number of points tracing the boundary.

    Return:
        (x, y): [tuple of ndarrays] Points tracing the ellipse boundary, in the same units as
            mean/cov.
    """

    radius = np.sqrt(-2.0*np.log(1.0 - p))

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0.0, None)  # guard tiny negative eigenvalues from roundoff

    # eigh() already returns ascending eigenvalues (a documented numpy contract, not just this
    #   function's current behavior), so this doesn't change the resulting ellipse - reordered
    #   descending anyway so the axis order is explicit in this function's own code rather than
    #   relying on the reader knowing that contract.
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    theta = np.linspace(0.0, 2*np.pi, n_points)
    circle = np.vstack([np.cos(theta), np.sin(theta)])

    transform = eigvecs @ np.diag(np.sqrt(eigvals))
    ellipse = np.asarray(mean)[:, None] + radius*(transform @ circle)

    return ellipse[0], ellipse[1]


def plotAlphaBeta(v_data, ht_data, fit, errors=None, band=True, fan=True, band_samples=1000,
        fan_curves=100, ellipse=True, ellipse_scatter=True, ellipse_scatter_points=300,
        residuals=False, ci=None, seed=None, axes=None, fast=False):
    """ Plot a fitAlphaBeta() result: observed (velocity, height) data and the fitted model
        curve, plus - if `errors` (from estimate_errors=True) is given - its propagated parameter
        uncertainty, optionally alongside a (ln alpha, ln beta) ellipse panel and a residuals
        panel.

        Standalone from fitAlphaBeta() on purpose: re-plotting an already-computed (fit, errors)
        pair never needs to re-run the fit, and estimate_errors=True's own sensitivity refits
        (for sigma_v_init's rank-1 correction) are not free - bundling plotting into the fit
        would force redoing that work just to redraw a figure.

        Naming discipline: the shaded region in the main panel, and the words used for it
        throughout this function, are "parameter uncertainty band"/"propagated parameter
        uncertainty", NEVER "confidence band". It is a Monte Carlo propagation of the analytic
        (ln alpha, ln beta) covariance through the deterministic forward model - not a classical
        confidence band, and not the same kind of object as wmpl.Utils.Math.confidenceInterval()'s
        empirical percentiles elsewhere in this library either, even though the same percentile
        formula is used once the samples exist. Same discipline as the existing "_wald"
        suffix on alpha_ci_wald_lower/upper.

    Arguments:
        v_data: [ndarray] Velocity data (m/s), as given to fitAlphaBeta().
        ht_data: [ndarray] Height data (m), as given to fitAlphaBeta().
        fit: [tuple] (v_init, alpha, beta), exactly as returned by fitAlphaBeta().

    Keyword arguments:
        errors: [dict or None] The 4th element returned by fitAlphaBeta(..., estimate_errors=
            True), or None. If None, or if its covariance fields are NaN (e.g. the fit landed on
            an ALPHA_BETA_BOUNDS edge - see fitAlphaBeta()'s docstring), the band/fan/ellipse
            content degrades gracefully to an annotation instead of raising.
        band: [bool] Draw the parameter uncertainty band in the main panel.
        fan: [bool] Draw individual propagated-uncertainty realizations (a subsample of the same
            draws used for the band) in the main panel.
        band_samples: [int] Number of (alpha, beta) draws from the (ln alpha, ln beta) Gaussian
            approximation (covariance cov_log). Set this by how stable the tail percentiles of the
            band need to be, but note the cost is NOT negligible: each draw evaluates the forward
            model (a per-height numerical velocity inversion) once, un-vectorized, so runtime grows
            roughly linearly with band_samples (order a few seconds at the default 1000 for a
            typical trajectory - reduce it for quick interactive plots).
        fan_curves: [int] Number of individual curves drawn for `fan`, subsampled from the same
            `band_samples` draws (capped at however many landed inside ALPHA_BETA_BOUNDS).
        ellipse: [bool] Draw the (ln alpha, ln beta) ellipse panel.
        ellipse_scatter: [bool] Overlay the sampled (ln alpha, ln beta) cloud on the ellipse
            panel - a visual illustration of the sampled Gaussian approximation, NOT a
            consistency check of anything: the samples and the ellipses both come from the same
            N(mean, cov_fit/cov_log), so they cannot meaningfully disagree except via a bug -
            this just renders the same distribution two ways. Default True; see
            ellipse_scatter_points to keep it from being too busy.
        ellipse_scatter_points: [int] Number of points drawn for `ellipse_scatter`, subsampled
            from the same pool as band/fan if those also drew samples (band_samples-sized),
            otherwise drawn fresh at this size directly - the ellipse cloud is a visual
            illustration, not a percentile computation, so it doesn't need band_samples'
            stability budget.
        residuals: [bool] Draw the (v_obs - v_model) vs height residuals panel. Off by default -
            opt-in, since it is a diagnostic view rather than part of the main result. Only needs
            `fit` and the raw data, so it still draws (without the sigma_v reference lines/
            down-weighted marking) even if errors is None - e.g. for a method='q4' point
            estimate, which can never have estimate_errors=True.
        ci: [float] Confidence level (0-100) for the band and the outer ellipse contour. If None
            (default), uses errors['ci'] (the level estimate_errors was computed at) if errors is
            given, else 95.0.
        seed: [int, Generator or None] Forwarded to numpy.random.default_rng() for reproducible
            sampling (an existing Generator is returned unaltered) - set this explicitly for any
            figure meant to be regenerated identically (e.g. for a paper).
        axes: [Axes] An existing matplotlib Axes to draw the main panel into, instead of creating
            a new figure. Only accepted when exactly one panel is being drawn (ellipse=False,
            residuals=False) - raises ValueError otherwise, since a single Axes can't hold a
            multi-panel layout. When given, plt.show() is NOT called (the caller owns the
            enclosing figure); when None, plotAlphaBeta() creates its own figure and calls
            plt.show() before returning, exactly like fitAlphaBetaLightCurve(..., plot=True).
        fast: [bool] Passed through everywhere this function inverts velocity - see
            alphaBetaVelocityNormed()'s docstring. The band_samples Monte Carlo loop (one velocity
            curve per sample, band_samples=1000 by default) is the largest beneficiary here.

    Return:
        (fig, axes): [tuple] The Figure (axes.figure if `axes` was given, otherwise newly
            created) and the Axes used, in (main, [ellipse], [residuals]) order - a bare Axes
            (not wrapped in a tuple) if only one panel was drawn.

    Statistical caveats:
        - The parameter uncertainty band/fan/ellipse are only as good as errors['cov_log']
          itself - see fitAlphaBeta()'s own Statistical caveats (local Gauss-Newton curvature,
          the v_init bias if derived internally, etc.). This function visualizes that
          approximation; it does not make it any more or less rigorous.
        - Samples landing outside ALPHA_BETA_BOUNDS are excluded from the band/fan/ellipse-overlay
          (a report is printed, never silently absorbed). If that excluded fraction exceeds 5%,
          the band/fan/ellipse-overlay are suppressed entirely (an annotation explains why, in
          place of the content) rather than drawn from what remains: past that point, the
          surviving sample no longer represents the (ln alpha, ln beta) Gaussian approximation
          itself, it represents that Gaussian CONDITIONED on being physically valid - a quietly
          different quantity that would be misleading to show under the same "propagated
          parameter uncertainty" label.
        - The main panel's x-axis (velocity) is clipped to the observed v_data range (+10%
          padding), same convention fitAlphaBetaLightCurve(..., plot=True) uses for its own
          velocity panel. v(h) evaluated over the observed HEIGHT range is NOT thereby
          constrained to the observed VELOCITY range - a fitted or (especially) sampled (alpha,
          beta) can still swing toward v=0 within that same height span, in exactly the regime
          where this model is steepest/least stable. Unclipped, that excursion would dominate the
          view and read as "the fit/band is very uncertain" when it is really just the model
          being asked to extrapolate into territory the data says nothing about. This only
          affects the view (ax.set_xlim) - the underlying curve/band/fan data is untouched, so a
          caller can always call ax.set_xlim()/set_ylim() again afterwards to see the full extent.
        - The residuals panel's down-weighted-point marker and its +/-f_scale*sigma_v reference
          lines use a hardcoded f_scale=2.0 (this module's own default elsewhere) - errors
          doesn't carry the f_scale an existing fit actually used, so these lines may not exactly
          match a fit's real soft-L1 transition point if it was fit with a non-default f_scale.
    """

    v_init, alpha, beta = fit

    have_errors = errors is not None
    have_cov = (have_errors and np.all(np.isfinite(errors['cov_log']))
        and np.all(np.isfinite(errors['cov_fit'])))

    if ci is None:
        ci = errors['ci'] if have_errors else 95.0

    if not (0 < ci < 100):
        raise ValueError("ci must be strictly between 0 and 100, got {!r}.".format(ci))

    # --- Decide which panels to draw, and enforce the axes/panel-count contract
    panel_names = ['main']
    if ellipse:
        panel_names.append('ellipse')
    if residuals:
        panel_names.append('residuals')

    n_panels = len(panel_names)

    if axes is not None and n_panels > 1:
        raise ValueError("axes is only accepted for a single-panel plotAlphaBeta() call (here "
            "ellipse={!r}, residuals={!r} would need {:d} panels) - pass axes=None and let "
            "plotAlphaBeta() build its own multi-panel figure instead.".format(
            ellipse, residuals, n_panels))

    import matplotlib.pyplot as plt

    if axes is not None:
        fig = axes.figure
        ax_map = {'main': axes}
    else:
        figsize = (9, 6) if n_panels == 1 else (6.5*n_panels, 6)
        fig, created_axes = plt.subplots(ncols=n_panels, figsize=figsize)
        if n_panels == 1:
            created_axes = [created_axes]
        ax_map = dict(zip(panel_names, created_axes))

    rng = np.random.default_rng(seed)
    log_mean = np.log([alpha, beta])

    # --- Draw (alpha, beta) samples from the (ln alpha, ln beta) Gaussian approximation once, if
    #   anything below needs them. Sized at band_samples if band/fan need it (their forward-model
    #   evaluation, the expensive part, is done separately below, only when actually needed) - or,
    #   if only the ellipse panel's optional scatter wants samples, sized at the smaller
    #   ellipse_scatter_points directly (that overlay is a visual illustration, not a
    #   percentile computation, so it doesn't need band_samples' stability budget).
    alpha_valid = beta_valid = v0_valid = None
    band_fan_suppressed = False

    # Draw a fitted initial velocity together with alpha and beta
    cov_full = errors.get('cov_full') if have_errors else None
    joint_v0 = (cov_full is not None) and np.all(np.isfinite(cov_full))

    need_velocity_samples = have_cov and (band or fan)
    need_ellipse_scatter = have_cov and ellipse and ellipse_scatter

    if need_velocity_samples or need_ellipse_scatter:

        n_draw = band_samples if need_velocity_samples else ellipse_scatter_points

        if joint_v0:
            joint_samples = rng.multivariate_normal(np.r_[v_init, log_mean], cov_full, size=n_draw)
            v0_samples, log_samples = joint_samples[:, 0], joint_samples[:, 1:]
        else:
            log_samples = rng.multivariate_normal(log_mean, errors['cov_log'], size=n_draw)
            v0_samples = np.full(n_draw, v_init)
        alpha_samples, beta_samples = np.exp(log_samples[:, 0]), np.exp(log_samples[:, 1])

        (a_lo, a_hi), (b_lo, b_hi) = ALPHA_BETA_BOUNDS
        is_valid = ((alpha_samples >= a_lo) & (alpha_samples <= a_hi) &
            (beta_samples >= b_lo) & (beta_samples <= b_hi))
        n_valid = int(np.sum(is_valid))

        if n_valid < n_draw:

            n_rejected = n_draw - n_valid
            rejected_frac = n_rejected/n_draw

            msg = "{:d}/{:d} sampled (alpha, beta) draws fell outside ALPHA_BETA_BOUNDS and " \
                "were excluded".format(n_rejected, n_draw)

            if rejected_frac > 0.05:
                band_fan_suppressed = True
                print("WARNING: {:s} ({:.1%}) - the local Gaussian approximation is likely "
                    "unreliable this close to a bound. Suppressing the parameter uncertainty "
                    "band/fan/ellipse-overlay rather than drawing them from what "
                    "remains.".format(msg, rejected_frac))
            else:
                print("{:s}.".format(msg))

        alpha_valid, beta_valid = alpha_samples[is_valid], beta_samples[is_valid]
        v0_valid = v0_samples[is_valid]

    # ================= Main panel: data + fit + parameter uncertainty =================

    ax_main = ax_map['main']

    ax_main.scatter(v_data/1000, ht_data/1000, s=10, color='0.5', alpha=0.6, zorder=3, \
        label="Observed")

    ht_grid = np.linspace(np.min(ht_data), np.max(ht_data), 150)
    v_fit_grid = alphaBetaVelocity(ht_grid, alpha, beta, v_init, fast=fast)

    ax_main.plot(v_fit_grid/1000, ht_grid/1000, color='k', linewidth=1.5, zorder=4, \
        label=r"Fit: $\alpha$={:.3g}, $\beta$={:.3g}".format(alpha, beta))

    if have_cov:
        # A separate (invisible) legend entry rather than folding into the "Fit:" label above,
        #   so the curve's own label stays short and this can be read independently
        ax_main.plot([], [], linestyle='none', \
            label=r"{:.3g}% CI: $\alpha$=[{:.3g}, {:.3g}], $\beta$=[{:.3g}, {:.3g}]".format(
            ci, errors['alpha_ci_wald_lower'], errors['alpha_ci_wald_upper'],
            errors['beta_ci_wald_lower'], errors['beta_ci_wald_upper']))

    if band or fan:

        if not have_cov:
            ax_main.text(0.5, 0.03, "Parameter uncertainty band unavailable\n(no errors given, "
                "or the fit landed on a bound)", transform=ax_main.transAxes, ha='center', \
                va='bottom', fontsize=8, color='0.4')

        elif band_fan_suppressed:
            ax_main.text(0.5, 0.03, "Parameter uncertainty band suppressed\n(too many samples "
                "landed outside ALPHA_BETA_BOUNDS - see console warning)", \
                transform=ax_main.transAxes, ha='center', va='bottom', fontsize=8, color='0.4')

        else:
            v_samples = np.array([alphaBetaVelocity(ht_grid, a, b, v0, fast=fast) \
                for v0, a, b in zip(v0_valid, alpha_valid, beta_valid)])

            if band:
                lo_pct, hi_pct = 50.0 - ci/2.0, 50.0 + ci/2.0
                v_lo, v_hi = np.percentile(v_samples, [lo_pct, hi_pct], axis=0)

                ax_main.fill_betweenx(ht_grid/1000, v_lo/1000, v_hi/1000, color='tab:blue', \
                    alpha=0.25, zorder=2, \
                    label="{:.3g}% propagated parameter uncertainty".format(ci))

            if fan:
                n_fan = min(fan_curves, len(alpha_valid))
                fan_idx = rng.choice(len(alpha_valid), size=n_fan, replace=False)

                for i in fan_idx:
                    ax_main.plot(v_samples[i]/1000, ht_grid/1000, color='tab:blue', alpha=0.05, \
                        linewidth=0.8, zorder=1)

                # One proxy artist for the legend - n_fan individually-labeled lines would flood it
                ax_main.plot([], [], color='tab:blue', alpha=0.4, linewidth=0.8, \
                    label="{:d} propagated-uncertainty realizations".format(n_fan))

    # Clip the view to the observed data range (with padding), same convention
    #   fitAlphaBetaLightCurve(..., plot=True) uses for its own velocity panel: v(h) evaluated
    #   over the observed HEIGHT range is not thereby constrained to the observed VELOCITY range
    #   - a fitted or sampled (alpha, beta) can still swing toward v=0 within that same height
    #   span, precisely in the regime where the model is steepest/least stable. Left unclipped,
    #   that excursion dominates the view and reads as "the fit/band is very uncertain" when it
    #   is really just the model being asked to extrapolate into territory the data says nothing
    #   about.
    v_pad = 0.1*(np.max(v_data) - np.min(v_data))
    ax_main.set_xlim((np.min(v_data) - v_pad)/1000, (np.max(v_data) + v_pad)/1000)

    ax_main.set_xlabel("Velocity (km/s)")
    ax_main.set_ylabel("Height (km)")
    ax_main.legend(fontsize=8, loc='best')

    # ================= Ellipse panel: (ln alpha, ln beta) =================

    if ellipse:

        ax_ell = ax_map['ellipse']

        if not have_cov:
            ax_ell.text(0.5, 0.5, "Error estimate unavailable\n(no errors given, or the fit "
                "landed on a bound)", transform=ax_ell.transAxes, ha='center', va='center', \
                fontsize=9, color='0.4')
            ax_ell.set_xticks([])
            ax_ell.set_yticks([])

        else:
            for cov_key, color, label_prefix in [
                    ('cov_fit', 'tab:orange', 'Fit only'),
                    ('cov_log', 'tab:blue', r'Fit + $v_0$')]:

                cov = errors[cov_key]

                for p, linestyle, level_label in [
                        (0.6827, '-', r"1$\sigma$"), (ci/100.0, '--', "{:.3g}%".format(ci))]:

                    x, y = _gaussianEllipsePoints(log_mean, cov, p)

                    ax_ell.plot(x, y, color=color, linestyle=linestyle, linewidth=1.5, \
                        label="{:s}, {:s}".format(label_prefix, level_label))

            if ellipse_scatter and not band_fan_suppressed and alpha_valid is not None:
                n_ell = min(ellipse_scatter_points, len(alpha_valid))
                ell_idx = rng.choice(len(alpha_valid), size=n_ell, replace=False)

                ax_ell.scatter(np.log(alpha_valid[ell_idx]), np.log(beta_valid[ell_idx]), \
                    s=10, color='0.6', alpha=0.15, zorder=0)

            ax_ell.scatter([log_mean[0]], [log_mean[1]], color='k', marker='x', zorder=5, \
                label=r"Fitted (ln $\alpha$, ln $\beta$)")

            ax_ell.set_xlabel(r"ln $\alpha$")
            ax_ell.set_ylabel(r"ln $\beta$")
            ax_ell.legend(fontsize=7, loc='best')

    # ================= Residuals panel (opt-in) =================

    if residuals:

        ax_res = ax_map['residuals']

        v_model_data = alphaBetaVelocity(ht_data, alpha, beta, v_init, fast=fast)
        resid = v_data - v_model_data

        if not have_errors:
            ax_res.scatter(resid, ht_data/1000, s=12, color='0.3', label="Residual")
            ax_res.axvline(0, color='k', linewidth=0.8)
            ax_res.text(0.5, 0.03, "sigma_v reference lines unavailable\n(no errors given)", \
                transform=ax_res.transAxes, ha='center', va='bottom', fontsize=8, color='0.4')

        else:
            sigma_v = errors['sigma_v']

            # This module's own default f_scale elsewhere (fitAlphaBeta(),
            #   minimizeAlphaBetaRobust()) - errors doesn't carry the f_scale an existing fit
            #   actually used, so these reference lines may not exactly match a fit's real
            #   soft-L1 transition point if it was fit with a non-default f_scale
            f_scale_ref = 2.0
            is_downweighted = np.abs(resid/sigma_v) > f_scale_ref

            ax_res.scatter(resid[~is_downweighted], ht_data[~is_downweighted]/1000, s=12, \
                color='0.3', label="Residual")
            ax_res.scatter(resid[is_downweighted], ht_data[is_downweighted]/1000, s=24, \
                color='crimson', marker='x', label="Down-weighted (soft-L1)")

            for k, style in [(1.0, ':'), (f_scale_ref, '--')]:
                ax_res.axvline(k*sigma_v, color='0.4', linestyle=style, linewidth=1)
                ax_res.axvline(-k*sigma_v, color='0.4', linestyle=style, linewidth=1)

            ax_res.axvline(0, color='k', linewidth=0.8)

        ax_res.set_xlabel(r"$v_{obs} - v_{model}$ (m/s)")
        ax_res.set_ylabel("Height (km)")
        ax_res.legend(fontsize=8, loc='best')

    fig.tight_layout()

    if axes is None:
        plt.show()

    result_axes = tuple(ax_map[name] for name in panel_names)
    if len(result_axes) == 1:
        result_axes = result_axes[0]

    return fig, result_axes


def plotProfileAlphaBeta(v_data, ht_data, fit, profile, delta_cost=True, degeneracy=True,
        trajectories=True, n_traj_curves=8, axes=None, fast=False):
    """ Plot a profileAlphaBeta() result: up to three panels per profiled parameter, built
        entirely from `profile` (plus the original data/fit for the trajectories panel) - no
        re-optimization.

        Panel 1 (`delta_cost`): the standard profile-likelihood plot - 'delta_cost' vs the
        parameter (log-x), 'delta_threshold' as a horizontal line, 'value_hat'/'ci_lower'/
        'ci_upper' as vertical lines. This alone already contains the full statistical content
        of the profile.

        Panel 2 (`degeneracy`): the (parameter, other_profile) curve - the other parameter's
        best re-optimized value as this one is swept. Shows the alpha/beta degeneracy directly:
        how much one has to move to compensate for the other and still fit the data comparably
        well.

        Panel 3 (`trajectories`): reconstructed (velocity, height) trajectories at 'ci_lower',
        'ci_upper', and `n_traj_curves` points in between (log-spaced in the profiled
        parameter), each paired with its own re-optimized 'other_profile' value via the same
        loss/f_scale the original fit used. If exactly one side of the profile CI is open (a
        weakly-constrained parameter's cost never crosses the threshold before hitting
        ALPHA_BETA_BOUNDS on that side - itself informative, see profileAlphaBeta()'s Return
        section), the sweep runs from 'value_hat' out to whichever side IS closed instead of
        showing nothing, with a note identifying the open side. Degrades to an annotation
        (rather than raising) only when BOTH sides are open, since there is then no finite
        interval left to reconstruct across. See the Statistical caveats below - and
        profileAlphaBeta()'s own - for exactly what this trajectory family does and does not
        represent: it is NOT the joint (alpha, beta) confidence region for the trajectory.

    Arguments:
        v_data: [ndarray] Velocity data (m/s), as given to fitAlphaBeta()/profileAlphaBeta().
        ht_data: [ndarray] Height data (m), as given to fitAlphaBeta()/profileAlphaBeta().
        fit: [tuple] (v_init, alpha, beta), exactly as returned by fitAlphaBeta() - should be the
            same fit `profile` was computed from (alpha/beta here are only used for the "Joint
            fit" curve/marker; they are not re-derived from `profile`).
        profile: [dict] profileAlphaBeta()'s return value for 'alpha' and/or 'beta' - must be
            from a profileAlphaBeta() new enough to include 'other_hat'/'other_profile'/'ci' in
            each per-parameter dict.

    Keyword arguments:
        delta_cost: [bool] Draw the delta_cost-vs-parameter panel.
        degeneracy: [bool] Draw the (parameter, other_profile) panel.
        trajectories: [bool] Draw the reconstructed-trajectories panel.
        n_traj_curves: [int] Number of intermediate trajectories drawn between 'ci_lower' and
            'ci_upper' (log-spaced in the profiled parameter), in addition to the two boundary
            curves and the joint-fit curve.
        axes: [Axes] An existing matplotlib Axes to draw into, instead of creating a new figure -
            only accepted when exactly one panel (one profiled parameter, one panel type) is
            being drawn, same contract as plotAlphaBeta()'s `axes`.
        fast: [bool] Passed through everywhere this function inverts velocity (the trajectories
            panel) - see alphaBetaVelocityNormed()'s docstring.

    Return:
        (fig, axes): [tuple] The Figure (axes.figure if `axes` was given, otherwise newly
            created) and either a bare Axes (if exactly one panel was drawn) or a nested dict
            {param_name: {panel_type: Axes}} covering every panel actually drawn.

    Statistical caveats:
        - See profileAlphaBeta()'s own Statistical caveats for the precise sense in which the
          trajectories panel's family is, and is not, a confidence region: it is the profile
          ridge (grid, other_profile) clipped to [ci_lower, ci_upper], which stays inside but
          does not trace the boundary of the joint (alpha, beta) region at the same `ci`.
        - The boundary curves at 'ci_lower'/'ci_upper' use 'other_profile' interpolated
          (log-log, matching how ci_lower/ci_upper themselves were located) rather than a fresh
          optimizer call exactly at those points - consistent with how those bounds were found,
          and at the same resolution (profileAlphaBeta()'s n_grid), not exact to machine
          precision.
    """

    import matplotlib.pyplot as plt

    param_names = [name for name in ('alpha', 'beta') if name in profile]
    if not param_names:
        raise ValueError("profile must contain 'alpha' and/or 'beta'.")

    panel_types = []
    if delta_cost:
        panel_types.append('delta_cost')
    if degeneracy:
        panel_types.append('degeneracy')
    if trajectories:
        panel_types.append('trajectories')

    if not panel_types:
        raise ValueError("At least one of delta_cost, degeneracy, trajectories must be True.")

    n_rows, n_cols = len(param_names), len(panel_types)
    n_panels = n_rows*n_cols

    if axes is not None and n_panels > 1:
        raise ValueError("axes is only accepted for a single-panel plotProfileAlphaBeta() call "
            "(here {:d} profiled parameter(s) x {:d} panel type(s) = {:d} panels) - pass "
            "axes=None and let plotProfileAlphaBeta() build its own multi-panel figure "
            "instead.".format(n_rows, n_cols, n_panels))

    if axes is not None:
        fig = axes.figure
        ax_map = {param_names[0]: {panel_types[0]: axes}}
    else:
        fig, created_axes = plt.subplots(nrows=n_rows, ncols=n_cols, squeeze=False,
            figsize=(5.5*n_cols, 4.5*n_rows))
        ax_map = {name: {} for name in param_names}
        for row, name in enumerate(param_names):
            for col, panel in enumerate(panel_types):
                ax_map[name][panel] = created_axes[row][col]

    v_init, alpha, beta = fit
    ht_grid = np.linspace(np.min(ht_data), np.max(ht_data), 150)
    v_pad = 0.1*(np.max(v_data) - np.min(v_data))

    other_param_name = {'alpha': 'beta', 'beta': 'alpha'}
    symbol = {'alpha': r'$\alpha$', 'beta': r'$\beta$'}

    for name in param_names:

        p = profile[name]
        other_name = other_param_name[name]

        # ================= Panel 1: delta_cost vs parameter =================

        if delta_cost:

            ax = ax_map[name]['delta_cost']

            ax.plot(p['grid'], p['delta_cost'], color='tab:blue', linewidth=1.5)
            ax.axhline(p['delta_threshold'], color='0.4', linestyle='--', linewidth=1, \
                label="{:.3g}% threshold (chi2, df=1)".format(p['ci']))
            ax.axvline(p['value_hat'], color='k', linewidth=1, \
                label="{:s} = {:.3g}".format(symbol[name], p['value_hat']))

            for bound, lbl in [(p['ci_lower'], 'CI lower'), (p['ci_upper'], 'CI upper')]:
                if bound is not None:
                    ax.axvline(bound, color='tab:orange', linestyle=':', linewidth=1.2, \
                        label="{:s} = {:.3g}".format(lbl, bound))

            ax.set_xscale('log')
            ax.set_xlabel(symbol[name])
            ax.set_ylabel(r"$\Delta$cost = 2(cost $-$ cost$_{best}$)/s$^2$")
            ax.set_title("{:s} profile".format(name))
            ax.legend(fontsize=7, loc='best')

        # ================= Panel 2: degeneracy curve =================

        if degeneracy:

            ax = ax_map[name]['degeneracy']

            ax.plot(p['grid'], p['other_profile'], color='tab:blue', linewidth=1.2, \
                label="Profile ridge")
            ax.scatter([p['value_hat']], [p['other_hat']], color='k', marker='x', zorder=5, \
                label="Joint fit")

            if p['ci_lower'] is not None or p['ci_upper'] is not None:

                for bound in (p['ci_lower'], p['ci_upper']):
                    if bound is not None:
                        other_at_bound = np.exp(np.interp(np.log(bound), np.log(p['grid']), \
                            np.log(p['other_profile'])))
                        ax.scatter([bound], [other_at_bound], color='tab:orange', marker='o', \
                            s=25, zorder=4)

                ax.scatter([], [], color='tab:orange', marker='o', s=25, label="CI edges")

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(symbol[name])
            ax.set_ylabel(symbol[other_name])
            ax.set_title("{:s}-{:s} degeneracy".format(name, other_name))
            ax.legend(fontsize=7, loc='best')

        # ================= Panel 3: reconstructed trajectories =================

        if trajectories:

            ax = ax_map[name]['trajectories']

            ax.scatter(v_data/1000, ht_data/1000, s=10, color='0.5', alpha=0.6, zorder=3, \
                label="Observed")

            v_fit_grid = alphaBetaVelocity(ht_grid, alpha, beta, v_init, fast=fast)
            ax.plot(v_fit_grid/1000, ht_grid/1000, color='k', linewidth=1.5, zorder=4, \
                label="Joint fit")

            # (alpha, beta) pair for a given value of THIS parameter, with the other one
            #   interpolated (log-log) from the already-computed profile ridge
            def _pairAt(target_value, name=name, p=p):
                other_value = np.exp(np.interp(np.log(target_value), np.log(p['grid']), \
                    np.log(p['other_profile'])))
                return (target_value, other_value) if name == 'alpha' \
                    else (other_value, target_value)

            bounds_available = [b for b in (p['ci_lower'], p['ci_upper']) if b is not None]

            if not bounds_available:
                ax.text(0.5, 0.03, "Trajectory family unavailable\n(profile CI open on both "
                    "sides - see the delta_cost panel)", transform=ax.transAxes, \
                    ha='center', va='bottom', fontsize=8, color='0.4')

            else:

                # An open side has no finite edge to sweep to, so the interior sweep and the
                #   boundary curve are only drawn for whichever side(s) are actually closed -
                #   the open side's own reconstruction would require extrapolating past where
                #   the data constrains anything at all. Sweeping FROM value_hat (rather than
                #   omitting that side's contribution entirely) still shows the full family
                #   compatible with the closed side, instead of discarding it just because the
                #   other side happens to be open.
                sweep_lo = p['ci_lower'] if p['ci_lower'] is not None else p['value_hat']
                sweep_hi = p['ci_upper'] if p['ci_upper'] is not None else p['value_hat']

                interior = np.geomspace(sweep_lo, sweep_hi, n_traj_curves + 2)[1:-1]

                for target_value in interior:
                    a, b = _pairAt(target_value)
                    v_curve = alphaBetaVelocity(ht_grid, a, b, v_init, fast=fast)
                    ax.plot(v_curve/1000, ht_grid/1000, color='tab:blue', alpha=0.2, \
                        linewidth=0.8, zorder=1)

                if len(interior):
                    ax.plot([], [], color='tab:blue', alpha=0.6, linewidth=0.8, \
                        label="{:d} profile CI trajectories".format(len(interior)))

                for bound, lbl in [(p['ci_lower'], "CI lower"), (p['ci_upper'], "CI upper")]:
                    if bound is None:
                        continue
                    a, b = _pairAt(bound)
                    v_curve = alphaBetaVelocity(ht_grid, a, b, v_init, fast=fast)
                    ax.plot(v_curve/1000, ht_grid/1000, color='tab:blue', linewidth=1.3, \
                        zorder=2, label="{:s} ({:s}={:.3g})".format(lbl, symbol[name], bound))

                if len(bounds_available) == 1:
                    open_side = "Lower" if p['ci_lower'] is None else "Upper"
                    ax.text(0.5, 0.03, "{:s} side open (unconstrained within "
                        "ALPHA_BETA_BOUNDS) - only the closed side's family is "
                        "shown".format(open_side), transform=ax.transAxes, ha='center', \
                        va='bottom', fontsize=7, color='0.4')

            ax.set_xlim((np.min(v_data) - v_pad)/1000, (np.max(v_data) + v_pad)/1000)
            ax.set_xlabel("Velocity (km/s)")
            ax.set_ylabel("Height (km)")
            ax.set_title("{:s} profile CI trajectories".format(name))
            ax.legend(fontsize=7, loc='best')

    fig.tight_layout()

    if axes is None:
        plt.show()

    if n_panels == 1:
        return fig, ax_map[param_names[0]][panel_types[0]]

    return fig, ax_map


def profileIntervalText(profile, name, fmt="{:.3g}", notes=True):
    """ Text of a profile interval from profileAlphaBetaLag(), which says if a side of it is not
        constrained by the data. An interval of beta which reaches the lower bound of the fit starts at 0, as
        beta is then consistent with zero (no mass loss), which is a physical limit and not a missing one.

    Arguments:
        profile: [dict] Result of profileAlphaBetaLag().
        name: [str] 'alpha' or 'beta'.

    Keyword arguments:
        fmt: [str] Format of the limits. "{:.3g}" by default.
        notes: [bool] Say which side of the interval is not constrained, or that beta is consistent with
            zero. True by default.

    Return:
        [str] The interval, e.g. "49.6 - 60.2", "< 0.482 (no lower limit)" for alpha, or
            "0 - 1.31 (consistent with zero)" for beta.
    """

    lower = fmt.format(profile[name + '_lower'])
    upper = fmt.format(profile[name + '_upper'])
    lower_open = profile[name + '_lower_at_bound']
    upper_open = profile[name + '_upper_at_bound']

    if lower_open and upper_open:
        return "not constrained"

    # Beta at the lower bound of the fit is consistent with zero
    if lower_open and (name == 'beta'):
        return "0 - {:s}{:s}".format(upper, " (consistent with zero)" if notes else "")

    if lower_open:
        return "< {:s}{:s}".format(upper, " (no lower limit)" if notes else "")

    if upper_open:
        return "> {:s}{:s}".format(lower, " (no upper limit)" if notes else "")

    return "{:s} - {:s}".format(lower, upper)



def plotAlphaBetaLagProfile(profile, alpha, beta, errors=None, title=None, fig=None, subplot_spec=None):
    """ Plot the uncertainties of alpha and beta of the fit to the lag: the likelihood on the 2D grid of alpha
        and beta with its 1, 2 and 3 sigma regions, and the profiles of alpha and beta on the sides.

    Arguments:
        profile: [dict] Result of profileAlphaBetaLag().
        alpha: [float] Alpha of the best fit.
        beta: [float] Beta of the best fit.

    Keyword arguments:
        errors: [dict] Uncertainties from fitAlphaBetaLag(estimate_errors=True). If given and finite, the
            Wald region of the same confidence level as the profile intervals is drawn for comparison, unless
            it reaches the bounds of the fit, where it is not meaningful. None by default.
        title: [str] Title of the plot. None by default.
        fig: [matplotlib figure] Figure to draw into, e.g. a figure with several plots. A new figure if None
            (default).
        subplot_spec: [SubplotSpec] Part of fig to draw into. The whole figure if None (default).

    Return:
        fig: [matplotlib figure] The figure.
    """

    import matplotlib.pyplot as plt

    own_figure = fig is None
    if own_figure:
        fig = plt.figure(figsize=(10, 8.5))

    if subplot_spec is None:
        grid_spec = fig.add_gridspec(2, 2, width_ratios=[3, 1.2], height_ratios=[1, 3])
    else:
        grid_spec = subplot_spec.subgridspec(2, 2, width_ratios=[3, 1.2], height_ratios=[1, 3], wspace=0.05, \
            hspace=0.05)
    ax_main = fig.add_subplot(grid_spec[1, 0])
    ax_alpha = fig.add_subplot(grid_spec[0, 0], sharex=ax_main)
    ax_beta = fig.add_subplot(grid_spec[1, 1], sharey=ax_main)
    ax_legend = fig.add_subplot(grid_spec[0, 1])
    ax_legend.axis('off')

    ci = profile['ci']
    threshold = profile['threshold']

    grid_x = profile['grid_alpha']
    grid_y = profile['grid_beta']
    grid_delta_chi2 = profile['grid_delta_chi2']

    # Plot the 3 sigma region of the grid with a margin, or the extent of the profiles if the grid has fewer
    #   than two points in it along an axis (or no grid)
    x_range = tuple(profile['alpha_profile'][0][[0, -1]])
    y_range = tuple(profile['beta_profile'][0][[0, -1]])
    if grid_delta_chi2.size:
        inside = grid_delta_chi2 <= scipy.stats.chi2.ppf(0.9973, 2)
        x_mesh, y_mesh = np.meshgrid(grid_x, grid_y)
        if np.any(inside):
            x_min, x_max = np.min(x_mesh[inside]), np.max(x_mesh[inside])
            y_min, y_max = np.min(y_mesh[inside]), np.max(y_mesh[inside])
            if x_max > x_min:
                x_range = (x_min - 0.15*(x_max - x_min), x_max + 0.15*(x_max - x_min))
            if y_max > y_min:
                y_range = (max(y_min - 0.15*(y_max - y_min), 0.0), y_max + 0.15*(y_max - y_min))

    # Mark the best fit
    handles = [ax_main.scatter([alpha], [beta], color='k', marker='+', s=80, zorder=4)]
    labels = ["Best fit"]

    ax_main.set_xlabel("Alpha")
    ax_main.set_ylabel("Beta")


    ### Plot the likelihood with the 1, 2 and 3 sigma regions of two parameters ###

    if grid_delta_chi2.size:

        likelihood = np.exp(-0.5*(grid_delta_chi2 - np.min(grid_delta_chi2)))
        ax_main.contourf(grid_x, grid_y, likelihood, levels=np.linspace(0.05, 1.0, 11), cmap='Blues', \
            alpha=0.7, zorder=1)

        levels = [scipy.stats.chi2.ppf(p, 2) for p in [0.6827, 0.9545, 0.9973]]
        contours = ax_main.contour(grid_x, grid_y, grid_delta_chi2, levels=levels, colors='navy', \
            linestyles=['-', '--', ':'], linewidths=1.0, zorder=2)
        handles += list(contours.legend_elements()[0])
        labels += ["1 sigma (2 params)", "2 sigma (2 params)", "3 sigma (2 params)"]

    # Draw the Wald region of the same confidence level for comparison
    if (errors is not None) and np.all(np.isfinite(errors['cov_log'])):

        radius = np.sqrt(scipy.stats.chi2.ppf(ci/100.0, 2))
        angles = np.linspace(0, 2*np.pi, 200)
        chol = np.linalg.cholesky(errors['cov_log'])
        ellipse = np.log([alpha, beta])[:, None] + radius*chol.dot(np.vstack([np.cos(angles), \
            np.sin(angles)]))

        # Draw the region only if it is within the bounds of the fit, as it is not meaningful for a weak
        #   constraint
        log_bounds = _logAlphaBetaBounds()
        ellipse_alpha, ellipse_beta = np.exp(ellipse)
        in_range = (np.min(ellipse_alpha) >= x_range[0]) and (np.max(ellipse_alpha) <= x_range[1]) \
            and (np.min(ellipse_beta) >= y_range[0]) and (np.max(ellipse_beta) <= y_range[1])
        if np.all(ellipse >= log_bounds[0][:, None]) and np.all(ellipse <= log_bounds[1][:, None]) \
            and in_range:
            handles += ax_main.plot(np.exp(ellipse[0]), np.exp(ellipse[1]), color='tab:red', lw=1.0, zorder=3)
            labels.append("Wald {:.3g}% (2 params)".format(ci))

    ax_main.set_xlim(x_range)
    ax_main.set_ylim(y_range)

    ax_legend.legend(handles, labels, loc='center', fontsize=7)


    ### Plot the profiles ###

    for name, ax, horizontal in [('alpha', ax_alpha, True), ('beta', ax_beta, False)]:

        values, delta_chi2 = profile[name + '_profile']
        lower = profile[name + '_lower']
        upper = profile[name + '_upper']

        if horizontal:
            ax.plot(values, delta_chi2, color='k', lw=1.0)
            ax.axhline(threshold, color='0.5', lw=0.8, ls='--')
            ax.axvspan(lower, upper, color='tab:blue', alpha=0.15, lw=0)
            ax.set_ylabel("Delta chi2")
            ax.set_ylim(0, 1.5*threshold)
            ax.tick_params(axis='x', which='both', labelbottom=False)

        else:
            ax.plot(delta_chi2, values, color='k', lw=1.0)
            ax.axvline(threshold, color='0.5', lw=0.8, ls='--')
            ax.axhspan(lower, upper, color='tab:blue', alpha=0.15, lw=0)
            ax.set_xlabel("Delta chi2")
            ax.set_xlim(0, 1.5*threshold)
            ax.tick_params(axis='y', which='both', labelleft=False)


    ### Print the values in the title ###

    lines = ["alpha = {:.3g} ({:s})".format(alpha, profileIntervalText(profile, 'alpha', notes=False)), \
        "beta = {:.3g} ({:s})".format(beta, profileIntervalText(profile, 'beta', notes=False)), \
        "{:.3g}% profile intervals".format(ci)]

    # The correlation of ln alpha and ln beta is not meaningful when beta reaches zero
    if np.isfinite(profile['corr_log']) and (not profile['beta_lower_at_bound']):
        lines.append("corr(ln alpha, ln beta) = {:.2f}".format(profile['corr_log']))

    # One line in the title of the figure, several lines over the plot in a figure with several plots
    if own_figure and (subplot_spec is None):
        text = ", ".join(lines)
        fig.suptitle(text if title is None else title + "\n" + text, fontsize=10)
        fig.tight_layout()
    else:
        text = "\n".join(lines)
        ax_alpha.set_title(text if title is None else title + "\n" + text, fontsize=8, loc='left')

    return fig



def _alphaBetaMassErrors(alpha, beta, slope, mu, dens, m_init, m_final, vel_init, vel_end,
        cov_log, sigma_slope, sigma_dens, sigma_vel_init, sigma_vel_end, ci):
    """ Analytic error estimate for alphaBetaMasses(estimate_errors=True).

        alphaBetaMasses() is, up to the mu/gamma/shape_coeff constants (treated as exact), a
        LOG-LINEAR function of alpha, sin(slope) and dens, plus one genuinely nonlinear term
        (m_final's dependence on beta itself, not ln(beta)) and, for the full alpha-beta solution
        (vel_init and vel_end both given), on vel_init/vel_end through r = vel_end/vel_init:

            ln(m_init)  = C - 2*ln(dens) - 3*ln(alpha) - 3*ln(sin(slope))
            ln(m_final) = ln(m_init) - beta*k,   k = (1 - r^2)/(1 - mu)   [r = 0 for the simple
                approximation, i.e. vel_init/vel_end not both given]

        This means the delta method (first-order Taylor propagation) is EXACT for every
        contribution except the beta/velocity ones - not an approximation on top of an
        approximation, unlike e.g. fitAlphaBeta()'s own Gauss-Newton curvature.

        The (alpha, beta) contribution is propagated via the same cov_log 2x2 covariance of
        (ln alpha, ln beta) that fitAlphaBeta(method='robust', estimate_errors=True) returns - the
        Jacobian rows below are dotted into it (J . cov_log . J^T), exactly the "use the
        covariance matrix" approach: this is required (not just quadrature-summing alpha_std and
        beta_std separately) because alpha and beta are typically strongly correlated, and because
        m_final depends on BOTH (m_init depends only on alpha - beta never appears in it at all).

        slope and dens are treated as independent of (alpha, beta) and of each other (a
        modeling/measurement assumption fed in by the caller, not something the dynamics fit
        constrains) - their contribution is simply added. Because they enter ln(m_init) and
        ln(m_final) through the EXACT SAME additive term (-3*ln(sin slope) - 2*ln(dens)), that
        contribution also appears in full in the m_init/m_final cross-covariance, not just in each
        variance separately.

        A clean consistency check of this whole derivation: Var(ln(m_final/m_init)) collapses to
        just (k*beta*beta_std_rel)^2 PLUS the vel_init/vel_end contribution (through r inside k,
        for the full solution) - it never depends on alpha/slope/dens at all, because those enter
        ln(m_init) and ln(m_final) identically and cancel out of the difference. It DOES still
        depend on vel_init/vel_end when those carry their own uncertainty, since - unlike
        alpha/slope/dens - they only ever appear in ln(m_final), never in ln(m_init), so their
        contribution has nothing to cancel against. This matches the physics: the fraction of mass
        ablated doesn't care about the absolute mass scale, only about how much velocity was lost.

    Arguments:
        alpha, beta, slope, mu, dens: [float] As given to alphaBetaMasses().
        m_init, m_final: [float] Already-computed point estimates.
        vel_init, vel_end: [float or None] As given to alphaBetaMasses().
        cov_log: [ndarray] 2x2 covariance matrix of (ln alpha, ln beta) - fitAlphaBeta(...,
            estimate_errors=True)'s errors['cov_log'].
        sigma_slope, sigma_dens: [float or None] 1-sigma uncertainty on slope (radians) / dens
            (kg/m^3). None treats that input as exactly known (zero contribution).
        sigma_vel_init, sigma_vel_end: [float or None] 1-sigma uncertainty on vel_init/vel_end
            (m/s). Only propagated when both vel_init and vel_end were given to alphaBetaMasses()
            (the full alpha-beta solution) - ignored (with a warning) otherwise, since the simple
            approximation doesn't use either velocity.
        ci: [float] Confidence level (0-100) for the Wald bounds.

    Return:
        errors: [dict] See alphaBetaMasses()'s docstring for the full field list.
    """

    sigma_slope_used = sigma_slope if sigma_slope is not None else 0.0
    sigma_dens_used = sigma_dens if sigma_dens is not None else 0.0

    full_solution = (vel_init is not None) and (vel_end is not None)

    if not full_solution:

        if (sigma_vel_init is not None) or (sigma_vel_end is not None):
            print("WARNING: sigma_vel_init/sigma_vel_end were given but vel_init and vel_end "
                "were not both supplied to alphaBetaMasses() - the asymptotic mass-loss "
                "approximation doesn't depend on either velocity, so sigma_vel_init/sigma_vel_end "
                "are IGNORED.")

        sigma_vel_init_used = 0.0
        sigma_vel_end_used = 0.0

    else:
        sigma_vel_init_used = sigma_vel_init if sigma_vel_init is not None else 0.0
        sigma_vel_end_used = sigma_vel_end if sigma_vel_end is not None else 0.0

    nan_result = {
        'm_init': m_init, 'm_final': m_final,
        'm_init_std_rel': np.nan, 'm_final_std_rel': np.nan,
        'm_init_std': np.nan, 'm_final_std': np.nan,
        'cov_log': np.full((2, 2), np.nan), 'corr_log': np.nan,
        'm_init_ci_wald_lower': np.nan, 'm_init_ci_wald_upper': np.nan,
        'm_final_ci_wald_lower': np.nan, 'm_final_ci_wald_upper': np.nan,
        'ci': ci,
        'sigma_slope': sigma_slope_used, 'sigma_dens': sigma_dens_used,
        'sigma_vel_init': sigma_vel_init_used, 'sigma_vel_end': sigma_vel_end_used,
    }

    cov_log = np.asarray(cov_log, dtype=float)

    if (cov_log.shape != (2, 2)) or (not np.all(np.isfinite(cov_log))):
        # Mirrors fitAlphaBeta()'s own bound-pinned failure mode: a NaN cov_log (e.g. the
        #   dynamics fit landed on an ALPHA_BETA_BOUNDS edge) must propagate to NaN here too,
        #   cleanly, rather than raising or emitting a raw numpy warning downstream.
        return nan_result

    cov00, cov01, cov11 = cov_log[0, 0], cov_log[0, 1], cov_log[1, 1]

    # d ln(m)/d ln(alpha) = -3, identical for m_init and m_final (the shared m0s/(alpha*sin
    #   slope)^3 term) - beta's own coefficient is 0 for m_init (it never appears there) and
    #   -k*beta for m_final (chain rule: d ln(m_final)/d(ln beta) = beta*d ln(m_final)/d(beta))
    a_coef = -3.0

    r = vel_end/vel_init if full_solution else 0.0
    k = (1.0 - r**2)/(1.0 - mu)
    b_coef_final = -k*beta

    # d ln(m)/d(slope) = -3*cot(slope), d ln(m)/d(dens) = -2/dens - identical for m_init/m_final
    s_coef = -3.0/np.tan(slope)
    d_coef = -2.0/dens

    # d ln(m_final)/d(vel_init), d ln(m_final)/d(vel_end) - zero for m_init (no velocity
    #   dependence at all) and zero for m_final too under the simple approximation
    if full_solution:
        vinit_coef = -2.0*beta*r**2/((1.0 - mu)*vel_init)
        vend_coef = 2.0*beta*r/((1.0 - mu)*vel_init)
    else:
        vinit_coef = 0.0
        vend_coef = 0.0

    # (alpha, beta) fit contribution: J . cov_log . J^T for each Jacobian row/pair of rows
    var_init_fit = a_coef**2*cov00
    var_final_fit = a_coef**2*cov00 + b_coef_final**2*cov11 + 2.0*a_coef*b_coef_final*cov01
    cov_fit_cross = a_coef**2*cov00 + a_coef*b_coef_final*cov01

    # Independent slope/dens contribution - the EXACT SAME additive random term in ln(m_init) and
    #   ln(m_final), so it adds to the cross-covariance in full too, not just to each variance
    #   separately (see the docstring above)
    var_shared_extra = s_coef**2*sigma_slope_used**2 + d_coef**2*sigma_dens_used**2

    var_m_init = var_init_fit + var_shared_extra
    var_m_final = (var_final_fit + var_shared_extra
        + vinit_coef**2*sigma_vel_init_used**2 + vend_coef**2*sigma_vel_end_used**2)
    cov_m = cov_fit_cross + var_shared_extra

    if (var_m_init < 0) or (var_m_final < 0):
        # Should be impossible: both are quadratic forms of a valid (PSD) cov_log plus
        # non-negative terms. Guarded anyway rather than ever propagating a nonsensical negative
        # variance (e.g. from a caller-supplied cov_log that isn't actually PSD).
        print("WARNING: alphaBetaMasses() mass-error propagation produced a negative variance - "
            "cov_log is likely not a valid (positive-semidefinite) covariance matrix. Returning "
            "NaN instead of a meaningless value.")
        return nan_result

    m_init_std_rel = np.sqrt(var_m_init)
    m_final_std_rel = np.sqrt(var_m_final)

    if (m_init_std_rel > 0) and (m_final_std_rel > 0):
        # Clip to [-1, 1]: for a valid (PSD) cov_log this is already in range, but a
        #   caller-supplied cov_log that isn't actually PSD could push it slightly outside, which
        #   would be a nonsensical correlation coefficient - clamp rather than propagate it.
        corr_log = np.clip(cov_m/(m_init_std_rel*m_final_std_rel), -1.0, 1.0)
    else:
        corr_log = np.nan

    m_init_std = m_init*m_init_std_rel
    m_final_std = m_final*m_final_std_rel

    # Parametric (log-normal) confidence interval - same "_wald" convention as fitAlphaBeta()'s
    #   alpha_ci_wald_lower/upper (see that docstring for what this is and is not)
    z = scipy.special.ndtri(0.5 + ci/200.0)
    m_init_ci_wald_lower = m_init*np.exp(-z*m_init_std_rel)
    m_init_ci_wald_upper = m_init*np.exp(z*m_init_std_rel)
    m_final_ci_wald_lower = m_final*np.exp(-z*m_final_std_rel)
    m_final_ci_wald_upper = m_final*np.exp(z*m_final_std_rel)

    return {
        'm_init': m_init,
        'm_final': m_final,
        'm_init_std_rel': m_init_std_rel,
        'm_final_std_rel': m_final_std_rel,
        'm_init_std': m_init_std,
        'm_final_std': m_final_std,
        'cov_log': np.array([[var_m_init, cov_m], [cov_m, var_m_final]]),
        'corr_log': corr_log,
        'm_init_ci_wald_lower': m_init_ci_wald_lower,
        'm_init_ci_wald_upper': m_init_ci_wald_upper,
        'm_final_ci_wald_lower': m_final_ci_wald_lower,
        'm_final_ci_wald_upper': m_final_ci_wald_upper,
        'ci': ci,
        'sigma_slope': sigma_slope_used,
        'sigma_dens': sigma_dens_used,
        'sigma_vel_init': sigma_vel_init_used,
        'sigma_vel_end': sigma_vel_end_used,
    }


def alphaBetaMasses(
        alpha,
        beta,
        slope,
        mu=0,
        dens=3500,
        shape_coeff=GAMMA_A,
        gamma=1.0,
        vel_init=None,
        vel_end=None,
        estimate_errors=False,
        cov_log=None,
        sigma_slope=None,
        sigma_dens=None,
        sigma_vel_init=None,
        sigma_vel_end=None,
        ci=95.0,
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
            factored together, we use gamma*A = GAMMA_A = 0.78 by default, as for all fireball masses in WMPL.
        gamma: [float] Drag parameter Γ (= C_D /2).
        vel_init: [float] Initial velocity in m/s. If both vel_init and vel_end are given, the final
            mass is computed using the full alpha-beta solution. If either is None, the simple
            approximation assuming vel_end << vel_init is used.
        vel_end: [float] Final velocity in m/s. Used together with vel_init to compute the final mass
            using the full alpha-beta solution.
        estimate_errors: [bool] If True, also return a 3rd element (a dict, see Return) with an
            analytic error estimate propagated from alpha/beta's own covariance plus (optionally)
            slope/density/velocity uncertainties. Off by default, so existing 2-tuple-unpacking
            call sites (m_init, m_final = alphaBetaMasses(...)) are unaffected. mu, gamma and
            shape_coeff are always treated as exactly known (no error is propagated from them).
        cov_log: [ndarray] REQUIRED if estimate_errors=True. The 2x2 covariance matrix of
            (ln alpha, ln beta) - pass fitAlphaBeta(..., estimate_errors=True)'s
            errors['cov_log'] directly. If this is all-NaN (e.g. that fit landed on an
            ALPHA_BETA_BOUNDS edge), every error field below comes back NaN too (alpha/beta
            themselves are unaffected, same graceful-degradation contract as fitAlphaBeta()).
        sigma_slope: [float] 1-sigma uncertainty on slope (radians). Only used if
            estimate_errors=True. None (default) treats slope as exactly known.
        sigma_dens: [float] 1-sigma uncertainty on dens (kg/m^3). Only used if
            estimate_errors=True. None (default) treats dens as exactly known.
        sigma_vel_init: [float] 1-sigma uncertainty on vel_init (m/s). Only used if
            estimate_errors=True, AND only propagated when both vel_init and vel_end are given
            (the full alpha-beta solution) - ignored (with a warning) under the simple
            approximation, since it doesn't depend on either velocity. None (default) treats
            vel_init as exactly known.
        sigma_vel_end: [float] 1-sigma uncertainty on vel_end (m/s). Same conditions as
            sigma_vel_init.
        ci: [float] Confidence level (0-100, e.g. 95, the default) for the approximate
            m_init_ci_wald_lower/upper and m_final_ci_wald_lower/upper bounds below - only used if
            estimate_errors=True. Same convention as wmpl.Utils.Math.confidenceInterval()'s `ci`.
        verbose: [bool] If True, print the parameters used in the computation and the resulting
            initial and final mass estimates (and, if estimate_errors=True, the error estimate
            too).
    Return:
        (m_init, m_final): [tuple of floats] Initial and final mass in kg.
        If estimate_errors=True, a 3rd element `errors` (dict, from _alphaBetaMassErrors()) is
        also returned:
            'm_init', 'm_final': echoed back, so the dict is self-contained.
            'm_init_std_rel', 'm_final_std_rel': [float] Relative (fractional) 1-sigma
                uncertainties of m_init/m_final, in log space - combining alpha/beta's own
                covariance (cov_log) with (if given) sigma_slope/sigma_dens/sigma_vel_init/
                sigma_vel_end.
            'm_init_std', 'm_final_std': [float] The same 1-sigma uncertainties propagated to
                linear mass space via the delta method (m_std = m*m_std_rel).
            'cov_log': [ndarray] 2x2 covariance matrix of (ln m_init, ln m_final) - NOT the same
                object as the `cov_log` argument (that one is alpha/beta's own covariance). Use
                this (not the marginal std devs above) to propagate m_init/m_final jointly into
                any OTHER derived quantity, e.g. the mass-loss fraction m_final/m_init - m_init
                and m_final are typically strongly correlated, since both share the same alpha/
                slope/density dependence.
            'corr_log': [float] Correlation coefficient of (ln m_init, ln m_final).
            'm_init_ci_wald_lower', 'm_init_ci_wald_upper', 'm_final_ci_wald_lower',
            'm_final_ci_wald_upper': [float] APPROXIMATE confidence intervals based on the local
                Gaussian (log-normal) approximation, at the `ci` level - same "_wald" convention
                as fitAlphaBeta()'s alpha_ci_wald_lower/upper (see that docstring for exactly what
                this is and is not).
            'ci': [float] The confidence level actually used (echoed back).
            'sigma_slope', 'sigma_dens', 'sigma_vel_init', 'sigma_vel_end': [float] The
                uncertainties actually used for the propagation above (0.0 for any that were left
                as None, or that were given but ignored - see the sigma_vel_init/sigma_vel_end
                Keyword arguments above).

    Statistical caveats (estimate_errors=True):
        - alpha, slope and dens all enter ln(m_init) (and hence ln(m_final)) EXACTLY log-linearly
          (up to the mu/gamma/shape_coeff constants, treated as exact) - so unlike fitAlphaBeta()'s
          own Gauss-Newton covariance, propagating THEIR contribution via the delta method
          introduces no additional approximation beyond assuming alpha/slope/dens are themselves
          adequately described by cov_log/sigma_slope/sigma_dens (i.e. log-normal for alpha,
          Gaussian for slope/dens).
        - beta (and, for the full solution, vel_init/vel_end) enter ln(m_final) through a
          genuinely nonlinear term (beta itself, and r^2 = (vel_end/vel_init)^2) - so, exactly as
          for fitAlphaBeta()'s own beta_ci_wald_lower/upper, this propagation is a LOCAL
          linearization around the point estimate, and degrades the same way fitAlphaBeta()'s own
          Wald interval does when beta_std_rel is large: see that function's Statistical caveats
          for the size of the effect and profileAlphaBeta() for a non-quadratic-assuming
          alternative to beta's own uncertainty (this function does not have an equivalent
          profile-based option for the propagated mass).
        - slope and dens are assumed independent of each other, of alpha/beta, and (for
          sigma_vel_init/sigma_vel_end) of the velocities - a simplification, not a check: if e.g.
          the assumed density is itself derived from the same fitted alpha/beta, that correlation
          is NOT captured here.
        - s_coef = -3*cot(slope) diverges as slope -> 0: near-grazing trajectories amplify
          sigma_slope's effect on the mass error dramatically (a small slope uncertainty translates
          into a large sin(slope) uncertainty), which is real physics, not a numerical artifact -
          but it does mean this propagation should not be trusted uncritically at very shallow
          entry angles without checking how large the resulting m_init_std_rel actually is.
        - Var(ln(m_final/m_init)) collapses to exactly (k*beta*beta_std_rel)^2 plus the
          vel_init/vel_end contribution (through r inside k) - independent of alpha/slope/dens
          entirely (they cancel out of the difference), but NOT independent of vel_init/vel_end's
          own uncertainty when the full solution is used - a useful identity to sanity-check any
          downstream use of 'cov_log' against.
    """

    if estimate_errors and (cov_log is None):
        raise ValueError("cov_log is required when estimate_errors=True - pass the 'cov_log' "
            "field from fitAlphaBeta(..., estimate_errors=True)'s errors dict.")

    if estimate_errors and not (0 < ci < 100):
        raise ValueError("ci must be strictly between 0 and 100, got {!r}. Out-of-range values "
            "degrade silently: ci=100 gives z=inf (0/inf bounds), ci>100 gives z=NaN, ci<0 "
            "inverts which bound is 'lower' vs 'upper'.".format(ci))

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

    if estimate_errors:
        errors = _alphaBetaMassErrors(alpha, beta, slope, mu, dens, m_init, m_final, vel_init, \
            vel_end, cov_log, sigma_slope, sigma_dens, sigma_vel_init, sigma_vel_end, ci)

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

        if estimate_errors:
            print(f"m_init       = {m_init:.3e} kg (+/-{errors['m_init_std_rel']:.1%})  "
                f"[{ci:.3g}% Wald CI: {errors['m_init_ci_wald_lower']:.3e} - "
                f"{errors['m_init_ci_wald_upper']:.3e}]")
            print(f"m_final      = {m_final:.3e} kg (+/-{errors['m_final_std_rel']:.1%})  "
                f"[{ci:.3g}% Wald CI: {errors['m_final_ci_wald_lower']:.3e} - "
                f"{errors['m_final_ci_wald_upper']:.3e}]")
            print(f"corr(m_init, m_final) = {errors['corr_log']:.3f}")
        else:
            print(f"m_init       = {m_init:.3e} kg")
            print(f"m_final      = {m_final:.3e} kg")

    if estimate_errors:
        return m_init, m_final, errors

    return m_init, m_final



def profileMassesAlphaBetaLag(profile, alpha, beta, v_init, slope, ht_end, mu=0, dens=3500, \
    shape_coeff=GAMMA_A, gamma=1.0, ci=95.0):
    """ Initial and final masses of the best fit and of the models within the profile interval of one
        parameter (see profileRegionAlphaBetaLag()), whose ranges are the profile intervals of the masses.
        The intervals hold also when beta is consistent with zero, where the Wald intervals of
        alphaBetaMasses(estimate_errors=True) from the covariance of ln alpha and ln beta are not meaningful,
        as the masses are finite at beta = 0 (no mass loss).

    Arguments:
        profile: [dict] Result of profileAlphaBetaLag() or profileAlphaBetaEntryLag(), with the 2D grid.
        alpha: [float] Alpha of the best fit.
        beta: [float] Beta of the best fit.
        v_init: [float] Initial velocity of the best fit (m/s).
        slope: [float] Entry angle above the horizon (radians).
        ht_end: [float] Height of the final mass in the exponential atmosphere of the model (m).

    Keyword arguments:
        mu: [float] Shape change coefficient. 0 by default.
        dens: [float] Bulk density (kg/m^3). 3500 by default.
        shape_coeff: [float] Shape coefficient, see alphaBetaMasses(). GAMMA_A by default.
        gamma: [float] Drag parameter, see alphaBetaMasses(). 1.0 by default.
        ci: [float] Confidence level of the intervals (percent). 95 by default.

    Return:
        (m_init, m_final):
            - m_init: [ndarray] Initial masses of the models (kg), empty if the profile has no grid.
            - m_final: [ndarray] Final masses of the models (kg), empty if the profile has no grid.
    """

    region_alpha, region_beta, region_v_init = profileRegionAlphaBetaLag(profile, ci=ci, n_params=1)

    if len(region_alpha) == 0:
        return np.array([]), np.array([])

    # Add the best fit to the models within the interval
    models = list(zip(region_alpha, region_beta, region_v_init)) + [(alpha, beta, v_init)]

    # Compute the masses with the final velocity of every model at the given height
    masses = np.array([alphaBetaMasses(a, b, slope, mu=mu, dens=dens, shape_coeff=shape_coeff, gamma=gamma, \
        vel_init=v0, vel_end=alphaBetaVelocity(ht_end, a, b, v0)) for a, b, v0 in models])

    return masses[:, 0], masses[:, 1]



def fitAlphaBetaMass(
        v_data,
        ht_data,
        slope,
        mass,
        mass_constraint="initial",
        shape_coeff=GAMMA_A,
        gamma=1.0,
        density=None,
        v_init=None,
        v_final=None,
        method='robust',
        sigma_v=None,
        loss='soft_l1',
        f_scale=2.0,
        verbose=True,
        plot=False,
        fast=False):
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
            are factored together, we use gamma*A = GAMMA_A = 0.78 by default, as for all fireball masses in
            WMPL.
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
        method: [str] 'robust' (default) or 'q4' - fitting approach used both for the preliminary
            unconstrained alpha-beta fit (see Notes below) and for the (density, beta[, mu])
            optimization under the selected mass_constraint. NOTE: 'robust' is the default, which
            is a change from this function's original behavior - before the fitting `method`
            argument existed, this function always used the 'q4' path. Existing callers that do
            not pass `method` will therefore get (generally slightly) different fitted
            density/alpha/beta/mass values than before. Pass method='q4' to reproduce the
            pre-change results exactly.
              - 'q4': the classic Gritsevich (2007) Q4 fit throughout (minimizeAlphaBeta() for the
                preliminary fit; scipy.optimize.minimize(method='Nelder-Mead') on
                _alphaBetaResidual()'s transformed, height-weighted residual for the constrained
                optimization) - unchanged from this function's original behavior.
              - 'robust': fitAlphaBeta(method='robust') for the preliminary fit, and
                scipy.optimize.least_squares(loss=loss, f_scale=f_scale) directly on the
                sigma-normalized velocity residuals (_dynResiduals(), the same residual
                minimizeAlphaBetaRobust() uses) for the constrained optimization. This also fixes
                a real pathology of the 'q4' path under mass_constraint="final": _alphaBetaResidual()
                trivially -> 0 as alpha -> 0 (a spurious minimum reached as beta grows and the
                reconstructed initial mass diverges - see the comment above where beta0 is seeded),
                which the velocity residual does not share, since alpha -> 0 flattens the model
                velocity curve instead of vanishing the residual.
        sigma_v: [float] Velocity uncertainty (m/s), only used for method='robust' (weights the
            residuals of every least_squares call this function makes, the preliminary fit
            included). If None, estimated once from the MAD of a Q4 fit's residuals (floored at
            1 m/s, see _estimateSigmaV()) and reused everywhere in this function, rather than
            letting each stage derive its own - the same "avoid circularity" reasoning
            fitAlphaBetaLightCurve() documents for its own sigma_v.
        loss: [str] Only used for method='robust'. scipy.optimize.least_squares loss, forwarded to
            every least_squares call this function makes.
        f_scale: [float] Only used for method='robust'. Robust loss scale (units of sigma_v),
            forwarded to every least_squares call this function makes.
        verbose: [bool] If True, print the parameters used in the computation and the resulting
            fitted parameters, clearly separating which of density/v_init/v_final were given
            as input from those that were derived/fitted internally.
        plot: [bool] If True, show a height vs. velocity plot with the observed data and the
            fitted alpha-beta curve(s): one for μ=0, one for μ=2/3, and (only when
            mass_constraint is "final" or "both", where mu_best is computed) a third one
            for the best-fit μ. Input parameters (as given
            by the caller) are shown in the title, fitted parameters (per μ branch) in the
            legend. Blocks execution until the plot window is closed.
        fast: [bool] Passed through everywhere this function inverts velocity (the preliminary
            fitAlphaBeta() call, the method='robust' (density, beta[, mu]) optimization, and the
            v_final derivation/recovery checks) - see alphaBetaVelocityNormed()'s docstring.

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

    method = method.lower()
    if method not in ('q4', 'robust'):
        raise ValueError("method must be 'q4' or 'robust', got '{:s}'.".format(method))

    # Normalize velocity
    v_normed = v_data/v_init

    # Normalize height
    ht_normed = ht_data/HT_NORM_CONST

    # For method='robust', estimate sigma_v (if not given) once, from a Q4 fit, and reuse it
    #   everywhere below (the preliminary fit and every constrained optimization) - avoids each
    #   stage deriving its own sigma_v from a fit that is itself being weighted by it (same
    #   reasoning as fitAlphaBetaLightCurve()'s sigma_v). Not needed at all for method='q4', which
    #   never weights residuals by sigma_v.
    sigma_v_given = sigma_v is not None
    sigma_v_normed = None
    if method == 'robust':

        alpha0_q4, beta0_q4 = minimizeAlphaBeta(v_normed, ht_normed, fast=fast)

        if not sigma_v_given:
            sigma_v = _estimateSigmaV(alpha0_q4, beta0_q4, v_normed, ht_normed, v_init)

        sigma_v_normed = sigma_v/v_init

    # Run a preliminary unconstrained alpha-beta fit when the final velocity has to be derived from
    #   the data, and always under the final-mass constraint, where the fitted beta is needed to
    #   seed the constrained optimization (see below). It is skipped under mass_constraint="initial"
    #   when v_final is not used by the fit at all - there v_final is always re-derived from the
    #   fitted alpha/beta after the constrained fit
    beta_prelim = None
    if (mass_constraint == "final") or ((v_final is None) and (mass_constraint == "both")):

        # Fit the unconstrained alpha-beta model
        _, alpha_prelim, beta_prelim = fitAlphaBeta(v_data, ht_data, v_init=v_init, method=method, \
            sigma_v=sigma_v, loss=loss, f_scale=f_scale, fast=fast)

        # Evaluate the model velocity at the observed heights and take the minimum as the final
        #   velocity, if it wasn't given already
        if v_final is None:
            vel_model = alphaBetaVelocity(ht_data, alpha_prelim, beta_prelim, v_init, fast=fast)
            v_final = np.min(vel_model)


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

    def _massFitResidual(alpha, beta, v_normed, ht_normed):
        """ Fit residual shared by every (density, beta[, mu]) objective below - only depends on
            the (alpha, beta) each objective derives from its own free parameters.

            method='q4': _alphaBetaResidual()'s scalar, Q4-transformed residual (for
            scipy.optimize.minimize(method='Nelder-Mead')) - unchanged original behavior.

            method='robust': _dynResiduals()'s sigma-normalized velocity residual VECTOR (the same
            residual minimizeAlphaBetaRobust() uses), for scipy.optimize.least_squares(loss=loss,
            f_scale=f_scale) - see _runMassFit(). Unlike the Q4 residual, this does not vanish as
            alpha -> 0, so it doesn't share the spurious-minimum pathology _alphaBetaResidual() has
            under mass_constraint="final" (see the beta0 seeding comment below).
        """

        if method == 'q4':
            return _alphaBetaResidual(alpha, beta, v_normed, ht_normed)

        return _dynResiduals(np.log([alpha, beta]), v_normed, ht_normed, sigma_v_normed, fast=fast)

    def _runMassFit(objective, x0, xmin_free, xmax_free, args):
        """ Runs one (density, beta[, mu]) optimization, dispatching on `method` to match the form
            `objective` returns (see _massFitResidual()): Nelder-Mead over the scalar Q4 cost for
            method='q4' (bounds as a tuple of (lo, hi) pairs), or least_squares over the robust
            residual vector for method='robust' (bounds as a (lo_array, hi_array) pair) - the two
            optimizers' bounds conventions differ, so this is the one place that needs to know
            which is active, instead of every call site duplicating the branch.
        """

        if method == 'q4':
            bnds = tuple(zip(xmin_free, xmax_free))
            return scipy.optimize.minimize(objective, x0, args=args, bounds=bnds, \
                method='Nelder-Mead')

        return scipy.optimize.least_squares(objective, x0, args=args, \
            bounds=(xmin_free, xmax_free), loss=loss, f_scale=f_scale, x_scale='jac')

    def _densityBetaMinimizationInitialMass(x, v_normed, ht_normed):
        """ Fit objective for mass_constraint="initial": alpha follows analytically from the fixed
            initial mass and the density, so only (density, beta) are free.
        """

        dens, rest = _getDensity(x)
        beta = rest[0]

        alpha = _alphaFromDensityAndMass(mass_initial, dens)

        return _massFitResidual(alpha, beta, v_normed, ht_normed)

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

        return _massFitResidual(alpha, beta, v_normed, ht_normed)

    def _densityBetaMuMinimizationFinalMass(x, v_normed, ht_normed):
        """ Same as _densityBetaMinimizationFinalMass, but with mu as a free parameter instead of
            fixed, so the residual-minimizing mu can be found jointly with density and beta.
        """

        dens, rest = _getDensity(x)
        beta, mu = rest

        _, alpha = _reconstructInitialMassAndAlpha(dens, beta, mu)

        return _massFitResidual(alpha, beta, v_normed, ht_normed)

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

        return _massFitResidual(alpha, beta, v_normed, ht_normed)

    def _densityMuMinimizationBothMasses(x, v_normed, ht_normed):
        """ Same as _densityMinimizationBothMasses, but with mu as a free parameter instead of
            fixed, so the residual-minimizing mu can be found jointly with density (beta is still
            derived analytically from mu and the two masses).
        """

        dens, rest = _getDensity(x)
        mu = rest[0]

        alpha, beta = _solveBothMasses(dens, mu)

        return _massFitResidual(alpha, beta, v_normed, ht_normed)


    # Initial guess
    dens0 = 2500
    beta0 = 1

    # Bounds
    xmin = [100, 0.001]
    xmax = [9000, 50]

    # Seed beta with the preliminary unconstrained fit when available. This is especially important
    #   under the final-mass constraint: for method='q4', the reduced objective has a spurious
    #   minimum at large beta, where the reconstructed initial mass diverges, alpha goes to 0, and
    #   _alphaBetaResidual() vanishes for any data, so the optimizer needs to start near the
    #   physical solution (method='robust' doesn't share this failure mode - see
    #   _massFitResidual() - but still benefits from starting close to the answer)
    if beta_prelim is not None:
        beta0 = float(np.clip(beta_prelim, xmin[1], xmax[1]))

    # Free parameters are (density, beta) if the density is fitted, or just (beta,) if it is set.
    #   Used for mass_constraint="initial" and the mu=0/mu=2/3 fits of mass_constraint="final",
    #   which all share this same 2-parameter (or 1-parameter) shape
    if fit_density:
        x0 = [dens0, beta0]
        xmin_free, xmax_free = [xmin[0], xmin[1]], [xmax[0], xmax[1]]
    else:
        x0 = [beta0]
        xmin_free, xmax_free = [xmin[1]], [xmax[1]]

    if mass_constraint == "initial":

        # Fit (density, beta), with alpha following analytically from the fixed initial mass
        res = _runMassFit(_densityBetaMinimizationInitialMass, x0, xmin_free, xmax_free, \
            (v_normed, ht_normed))

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
        vel_model = alphaBetaVelocity(ht_data, alpha, beta, v_init, fast=fast)
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
        res_mu0 = _runMassFit(_densityBetaMinimizationFinalMass, x0, xmin_free, xmax_free, \
            (v_normed, ht_normed, 0))

        if not res_mu0.success:
            print(f"WARNING: Optimizer failed for mu=0: {res_mu0.message}")

        density_mu0, rest = _getDensity(res_mu0.x)
        beta_mu0 = rest[0]

        # Only alpha is needed here - the masses are computed below with _massesAt(), which
        #   reproduces the same initial mass by construction
        _, alpha_mu0 = _reconstructInitialMassAndAlpha(density_mu0, beta_mu0, 0)

        m_initial_mu0, m_final_mu0 = _massesAt(alpha_mu0, beta_mu0, 0, density_mu0)

        # Fit (density, beta) for mu = 2/3
        res_mu23 = _runMassFit(_densityBetaMinimizationFinalMass, x0, xmin_free, xmax_free, \
            (v_normed, ht_normed, 2/3))

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
            xmin_mu_best, xmax_mu_best = [xmin[0], xmin[1], 0.0], [xmax[0], xmax[1], 2/3]
        else:
            x0_mu_best = [beta0, 1/3]
            xmin_mu_best, xmax_mu_best = [xmin[1], 0.0], [xmax[1], 2/3]

        res_mu_best = _runMassFit(_densityBetaMuMinimizationFinalMass, x0_mu_best, xmin_mu_best, \
            xmax_mu_best, (v_normed, ht_normed))

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
            res_mu0 = _runMassFit(_densityMinimizationBothMasses, [dens0], [xmin[0]], [xmax[0]], \
                (v_normed, ht_normed, 0))

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
            res_mu23 = _runMassFit(_densityMinimizationBothMasses, [dens0], [xmin[0]], [xmax[0]], \
                (v_normed, ht_normed, 2/3))

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
            xmin_mu_best, xmax_mu_best = [xmin[0], 0.0], [xmax[0], 2/3]
        else:
            x0_mu_best = [1/3]
            xmin_mu_best, xmax_mu_best = [0.0], [2/3]

        res_mu_best = _runMassFit(_densityMuMinimizationBothMasses, x0_mu_best, xmin_mu_best, \
            xmax_mu_best, (v_normed, ht_normed))

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
        print(f"method               = {method}")
        if method == 'robust':
            print("sigma_v              = {:.1f} m/s  ({:s})".format(sigma_v,
                "input" if sigma_v_given else "DERIVED: MAD of a Q4 fit's velocity residuals "
                "(effective, absorbs model misspecification too - not a pure measurement "
                "uncertainty)"))

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

    # Beta spans several orders of magnitude between its bounds, so its distance to them is only
    #   meaningful in log space (a linear check would flag e.g. beta=0.33 as "close" to the 1e-5
    #   lower bound) - see _makeBoundChecker()
    _checkBoundLog = _makeBoundChecker(warnings_list, tol=bound_tol, log=True)

    # Check the fitted densities against the optimization bounds
    if fit_density:
        _checkBound("density_mu0", density_mu0, xmin[0], xmax[0])
        _checkBound("density_mu23", density_mu23, xmin[0], xmax[0])

    # Check the fitted betas (under "both" beta is analytic, not fitted, so it's skipped)
    if mass_constraint in ["initial", "final"]:
        _checkBoundLog("beta_mu0", beta_mu0, xmin[1], xmax[1])
        _checkBoundLog("beta_mu23", beta_mu23, xmin[1], xmax[1])

    if fit_density and mass_constraint in ["final", "both"]:
        _checkBound("density_mu_best", density_mu_best, xmin[0], xmax[0])

    if mass_constraint == "final":
        _checkBoundLog("beta_mu_best", beta_mu_best, xmin[1], xmax[1])

    def _checkVelocityRecovery(name, alpha, beta):
        """ Under mass_constraint="final"/"both", beta is derived analytically assuming a
            particular v_final (given or derived from a preliminary unconstrained fit), but
            alpha/beta are then fitted to best match the v-h curve shape. Nothing else guarantees
            that plugging the fitted alpha/beta back into alphaBetaVelocity() actually reproduces
            that same v_final, so check it explicitly here.
        """

        v_final_model = np.min(alphaBetaVelocity(ht_data, alpha, beta, v_init, fast=fast))
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

            vel_arr = alphaBetaVelocity(ht_arr, alpha_val, beta_val, v_init, fast=fast)

            # Computed the same way as in _checkVelocityRecovery() above (min over the
            # actual ht_data points, not over the ht_arr plotting grid), so the number
            # shown in the legend always matches what that check evaluates.
            v_final_model = np.min(alphaBetaVelocity(ht_data, alpha_val, beta_val, v_init, fast=fast))

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


def plotAlphaBetaSurvivalDiagram(
        alpha,
        beta,
        slope,
        mass_thresholds=(1.0, 0.05),
        mu_low=0.0,
        mu_high=2/3,
        dens=3500,
        shape_coeff=GAMMA_A,
        gamma=1.0,
        errors=None,
        ci=None,
        shade=True,
        x_range=(0, 7.0),
        y_range=(-4.0, 4.0),
        axes=None,
        profile=None):
    """ Plot the classic Gritsevich ln(alpha*sin(slope)) vs ln(beta) meteorite-survival diagram:
        where a given (alpha, beta, slope) falls relative to the minimum-recoverable-mass
        boundary curves for the two physical shape/spin extremes (mu_low = 0, no rotation;
        mu_high = 2/3, fast rotation ablating the whole surface evenly), shaded into "likely
        fall" / "possible fall" / "unlikely fall" zones for one primary mass threshold.

        Replaces the ad-hoc, hardcoded-constant version of this plot that used to live inline in
        this module's CLI script (the "PLOT METEORITE DROPPING POSSIBILITY" block): those
        constants (e.g. "13.2 - 3*x") were copied in from elsewhere and did NOT track this
        module's own dens/gamma/shape_coeff - they stayed fixed even if a caller passed a
        different --dens/--ga on the command line. Here the boundary curves are instead derived
        analytically, live, from the exact same m0s alphaBetaMasses() itself uses, so they are
        always self-consistent with whatever dens/gamma/shape_coeff is passed in.

        Derivation: alphaBetaMasses()'s asymptotic mass-loss approximation gives
            ln(m_final) = ln(m0s) - 3*x - beta/(1 - mu),   x = ln(alpha*sin(slope))
        Solving for beta at a fixed m_final = mass threshold:
            beta_boundary(x; mu) = (1 - mu)*(ln(m0s/mass_threshold) - 3*x)
        which is exactly linear in x - a straight line in this (x, ln beta) plot for each
        (mu, mass_threshold) pair. Since (1 - mu) is smaller at mu_high than at mu_low, the
        mu_high boundary sits at a LOWER beta than the mu_low boundary for the same threshold (the
        same alpha/beta loses more mass under the fast-rotation assumption) - so for a fixed
        threshold, a point below BOTH lines survives under either shape assumption ("likely
        fall"), a point between them survives only under mu_low ("possible fall", i.e. the
        verdict depends on the shape/spin assumption), and a point above both never survives
        ("unlikely fall").

        Only the asymptotic (v_final << v_init) mass approximation is used here, deliberately -
        not the full alpha-beta solution alphaBetaMasses() also supports - because the boundary
        curves above are only exactly linear in x under that approximation; mixing it with a
        full-solution point estimate would make the plotted point not fall exactly on a boundary
        even when its own m_final exactly equals the threshold.

    Arguments:
        alpha: [float] Ballistic coefficient.
        beta: [float] Mass loss parameter.
        slope: [float] Entry angle (radians).

    Keyword arguments:
        mass_thresholds: [sequence of floats] Mass thresholds (kg) to draw boundary curves for.
            The FIRST entry is the "primary" threshold: its mu_low/mu_high curves get the
            likely/possible/unlikely shading (if shade=True) and the legend's mu=0/mu=2/3 labels.
            Any remaining entries are drawn as thin grey reference lines only, without shading -
            useful to show e.g. both a 50 g and a 1 kg threshold on the same plot without
            overlapping/contradictory shaded regions. Default (1.0, 0.05) kg matches the
            module's previous ad-hoc script.
        mu_low: [float] Lower shape/spin extreme (no rotation). Must satisfy
            0 <= mu_low < mu_high <= 2/3 (alphaBetaMasses()'s own valid mu range).
        mu_high: [float] Upper shape/spin extreme (fast rotation, uniform ablation).
        dens: [float] Bulk density (kg/m^3) - see alphaBetaMasses().
        shape_coeff: [float] Shape coefficient - see alphaBetaMasses().
        gamma: [float] Drag parameter Gamma - see alphaBetaMasses().
        errors: [dict or None] fitAlphaBeta(..., estimate_errors=True)'s errors dict (the one
            with a (ln alpha, ln beta) 'cov_log', NOT alphaBetaMasses()'s own errors dict, which
            is a (ln m_init, ln m_final) covariance instead and isn't the right shape for this
            plot). If given (and its cov_log is finite), draws a 1-sigma and a ci% uncertainty
            ellipse around the point, in the same (x, ln beta) plane (slope is treated as exactly
            known - shifting x by ln(sin(slope)) doesn't change the covariance). If None, or if
            cov_log is all-NaN (e.g. the fit landed on an ALPHA_BETA_BOUNDS edge), only the point
            itself is drawn.
        ci: [float] Confidence level (0-100) for the outer ellipse contour. If None (default),
            uses errors['ci'] if errors is given, else 95.0 - same fallback convention as
            plotAlphaBeta()'s own `ci`.
        shade: [bool] Draw the likely/possible/unlikely fall zone shading for the primary
            threshold. If False, only the boundary lines are drawn (all thresholds treated as
            reference lines), matching the original script's appearance.
        x_range, y_range: [tuple of floats] Axis limits, (min, max). Defaults match the module's
            previous ad-hoc script.
        axes: [Axes] An existing matplotlib Axes to draw into, instead of creating a new figure.
        profile: [dict] Result of profileAlphaBetaLag() with the 2D grid. If given, the 1 sigma and the ci%
            regions of two parameters are drawn from its likelihood instead of the ellipses of errors, as
            they also hold when alpha or beta is weakly constrained. None by default.

    Return:
        (fig, ax): [tuple] The Figure (ax.figure if `axes` was given, otherwise newly created) and
            the Axes used.

    Statistical caveats:
        - The boundary curves assume slope, dens, gamma and shape_coeff are all exactly known -
          the same assumption alphaBetaMasses() itself makes unless its own estimate_errors=True
          uncertainty inputs are used. No slope/density uncertainty is folded into this plot.
        - The uncertainty ellipse (from `errors`) describes (alpha, beta)'s covariance only; it
          does not shift or widen the boundary lines themselves, which remain drawn at the point
          estimates of dens/gamma/shape_coeff.
    """

    if not (0 <= mu_low < mu_high <= 2/3):
        raise ValueError("mu_low and mu_high must satisfy 0 <= mu_low < mu_high <= 2/3, got "
            "mu_low={!r}, mu_high={!r}.".format(mu_low, mu_high))

    if len(mass_thresholds) < 1:
        raise ValueError("mass_thresholds must contain at least one mass (kg).")

    if any(m <= 0 for m in mass_thresholds):
        raise ValueError("mass_thresholds must all be positive, got {!r}.".format(mass_thresholds))

    if ci is None:
        ci = errors['ci'] if errors is not None else 95.0

    if not (0 < ci < 100):
        raise ValueError("ci must be strictly between 0 and 100, got {!r}.".format(ci))

    import matplotlib.pyplot as plt

    if axes is None:
        fig, ax = plt.subplots(figsize=(7.5, 6))
    else:
        fig = axes.figure
        ax = axes

    def _fmtMass(m):
        return "{:.0f} g".format(1000*m) if m < 1.0 else "{:.3g} kg".format(m)

    rho_atm_0 = 1.225
    m0s = (gamma*shape_coeff*rho_atm_0*HT_NORM_CONST/(dens**(2/3.0)))**3

    x_grid = np.linspace(x_range[0], x_range[1], 400)

    def _betaBoundary(mu, mass_threshold):
        """ The beta AT WHICH m_final(x, beta, mu) = mass_threshold - linear in x, but this is
            beta itself, NOT ln(beta) (the plotted y-axis) - callers that plot this must take
            np.log() of it first. Goes negative once x is large enough that m_init(x) alone
            (i.e. even beta=0, no mass loss at all) already undercuts mass_threshold - beyond that
            point there is no physical (positive) beta left that reaches the threshold, so
            np.log() of it is correctly NaN there (see _lnBetaBoundary()).
        """
        return (1.0 - mu)*(np.log(m0s/mass_threshold) - 3.0*x_grid)

    def _lnBetaBoundary(mu, mass_threshold):
        """ ln(beta) of the boundary curve - the actual y-axis quantity to plot. NaN wherever
            _betaBoundary() is non-positive (see its docstring); matplotlib skips NaN points,
            which correctly truncates the curve exactly where it stops being physical, instead of
            extrapolating a meaningless negative-beta boundary.
        """
        with np.errstate(invalid='ignore'):
            return np.log(_betaBoundary(mu, mass_threshold))

    primary_threshold, reference_thresholds = mass_thresholds[0], mass_thresholds[1:]

    # The mu_high curve sits at a LOWER beta than mu_low for the same threshold (see docstring),
    #   so mu_high is the "likely/possible" divide and mu_low is the "possible/unlikely" divide
    ln_boundary_low = _lnBetaBoundary(mu_low, primary_threshold)
    ln_boundary_high = _lnBetaBoundary(mu_high, primary_threshold)

    if shade:

        # Past the point where a boundary curve goes NaN (see _betaBoundary()'s docstring), the
        #   entire column is already "unlikely" for ANY physical beta - fill down to y_range[0]
        #   there instead of leaving a gap, so the "likely"/"possible" bands correctly vanish to
        #   zero width and only "unlikely" remains, rather than matplotlib silently not filling
        #   anything past the NaN.
        y_lo_clip = np.clip(np.where(np.isfinite(ln_boundary_high), ln_boundary_high, \
            y_range[0]), y_range[0], y_range[1])
        y_hi_clip = np.clip(np.where(np.isfinite(ln_boundary_low), ln_boundary_low, \
            y_range[0]), y_range[0], y_range[1])

        ax.fill_between(x_grid, y_hi_clip, y_range[1], color='red', alpha=0.2, zorder=0, \
            label="Unlikely fall ({:s})".format(_fmtMass(primary_threshold)))
        ax.fill_between(x_grid, y_lo_clip, y_hi_clip, color='orange', alpha=0.2, zorder=0, \
            label="Possible fall ({:s})".format(_fmtMass(primary_threshold)))
        ax.fill_between(x_grid, y_range[0], y_lo_clip, color='green', alpha=0.2, zorder=0, \
            label="Likely fall ({:s})".format(_fmtMass(primary_threshold)))

    ax.plot(x_grid, ln_boundary_low, color='k', linestyle='-', linewidth=1.3, \
        label=r"{:s}, $\mu$={:s}".format(_fmtMass(primary_threshold), _formatMu(mu_low)))
    ax.plot(x_grid, ln_boundary_high, color='k', linestyle='--', linewidth=1.3, \
        label=r"{:s}, $\mu$={:s}".format(_fmtMass(primary_threshold), _formatMu(mu_high)))

    for mass_threshold in reference_thresholds:
        ax.plot(x_grid, _lnBetaBoundary(mu_low, mass_threshold), color='0.5', linestyle='-', \
            linewidth=1.0, label=r"{:s}, $\mu$={:s}".format(_fmtMass(mass_threshold), _formatMu(mu_low)))
        ax.plot(x_grid, _lnBetaBoundary(mu_high, mass_threshold), color='0.5', linestyle='--', \
            linewidth=1.0, label=r"{:s}, $\mu$={:s}".format(_fmtMass(mass_threshold), _formatMu(mu_high)))

    # Point estimate, annotated with m_final at both mu extremes for the primary threshold's own
    #   dens/gamma/shape_coeff (asymptotic approximation, matching the boundary curves above)
    x_point = np.log(alpha) + np.log(np.sin(slope))
    y_point = np.log(beta)

    _, m_final_mu_low = alphaBetaMasses(alpha, beta, slope, mu=mu_low, dens=dens, \
        shape_coeff=shape_coeff, gamma=gamma)
    _, m_final_mu_high = alphaBetaMasses(alpha, beta, slope, mu=mu_high, dens=dens, \
        shape_coeff=shape_coeff, gamma=gamma)

    point_label = (r"$m_{{final}}(\mu={:s})={:.2e}$ kg".format(_formatMu(mu_low), m_final_mu_low) + "\n" +
        r"$m_{{final}}(\mu={:s})={:.2e}$ kg".format(_formatMu(mu_high), m_final_mu_high))

    # Draw the regions of the profile likelihood if given
    have_profile = (profile is not None) and (profile['grid_delta_chi2'].size > 0)
    if have_profile:
        grid_x = np.log(profile['grid_alpha']) + np.log(np.sin(slope))
        grid_y = np.log(profile['grid_beta'])
        for p, linestyle, level_label in [
                (0.6827, '-', r"1$\sigma$"), (ci/100.0, '--', "{:.3g}%".format(ci))]:
            ax.contour(grid_x, grid_y, profile['grid_delta_chi2'], levels=[scipy.stats.chi2.ppf(p, 2)], \
                colors='tab:blue', linestyles=linestyle, linewidths=1.3)
            ax.plot([], [], color='tab:blue', linestyle=linestyle, linewidth=1.3, \
                label=r"{:s} ($\alpha$, $\beta$) profile likelihood".format(level_label))

    have_cov = (not have_profile) and (errors is not None) and np.all(np.isfinite(errors['cov_log']))

    if have_cov:

        mean = np.array([x_point, y_point])

        for p, linestyle, level_label in [
                (0.6827, '-', r"1$\sigma$"), (ci/100.0, '--', "{:.3g}%".format(ci))]:

            ex, ey = _gaussianEllipsePoints(mean, errors['cov_log'], p)
            ax.plot(ex, ey, color='tab:blue', linestyle=linestyle, linewidth=1.3, \
                label=r"{:s} ($\alpha$, $\beta$) uncertainty".format(level_label))

    elif (errors is not None) and (not have_profile):
        ax.text(0.02, 0.02, "Uncertainty ellipse unavailable\n(cov_log is NaN)", \
            transform=ax.transAxes, ha='left', va='bottom', fontsize=8, color='0.4')

    if have_cov:

        # Marginal 1-sigma error bars on the point (sqrt of cov_log's diagonal): a quick-read
        #   complement to the ellipse above, which additionally shows the (alpha, beta)
        #   correlation the bars alone cannot. slope is treated as exact, so sigma_x = sigma_ln
        #   alpha and sigma_y = sigma_ln beta directly.
        sigma_x = np.sqrt(errors['cov_log'][0, 0])
        sigma_y = np.sqrt(errors['cov_log'][1, 1])

        ax.errorbar(
            [x_point],
            [y_point],
            xerr=[sigma_x],
            yerr=[sigma_y],
            fmt='o',
            color='blue',
            ecolor='tab:blue',
            elinewidth=1.0,
            capsize=3,
            capthick=1.0,
            zorder=5,
            label=point_label,
        )

    else:
        ax.scatter(
            [x_point],
            [y_point],
            c='blue',
            zorder=5,
            label=point_label,
        )

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel(r"$\ln(\alpha \sin(\mathrm{slope}))$")
    ax.set_ylabel(r"$\ln(\beta)$")
    ax.legend(fontsize=7, loc='best')

    if axes is None:
        fig.tight_layout()

    if axes is None:
        plt.show()

    return fig, ax


def _formatMu(mu):
    """ Format the shape change coefficient, as a fraction for 2/3.

    Arguments:
        mu: [float] Shape change coefficient.

    Return:
        [str] The coefficient, e.g. "0" or "2/3".
    """

    if np.isclose(mu, 2/3.0):
        return "2/3"

    return "{:.2g}".format(mu)



def _formatMass(mass):
    """ Format a mass with a unit which keeps the number readable.

    Arguments:
        mass: [float] Mass (kg).

    Return:
        [str] The mass with the unit, e.g. "142 g" or "2.3 kg".
    """

    if not np.isfinite(mass):
        return "inf" if mass > 0 else "-"

    if mass < 1e-3:
        return "{:.3g} mg".format(1e6*mass)

    if mass < 1.0:
        return "{:.3g} g".format(1e3*mass)

    if mass < 1000.0:
        return "{:.3g} kg".format(mass)

    return "{:.3g} t".format(mass/1000.0)



def plotAlphaBetaMassDiagram(alpha, beta, slope, profile=None, zone_masses=(0.05, 0.1), mu_zones=0.0, \
    mu_other=2/3, dens=3500, shape_coeff=GAMMA_A, gamma=1.0, axes=None):
    """ Plot the meteorite survival diagram with beta on a linear axis, where the lines of a constant final
        mass are straight. With the final mass m_final = m0s/(alpha*sin(slope))^3*exp(-beta/(1 - mu)) (the
        approximation of a final velocity much smaller than the initial one, as in
        plotAlphaBetaSurvivalDiagram()), the final mass is M on the line
        beta = (1 - mu)*(ln(m0s/M) - 3*ln(alpha*sin(slope))). It is the same diagram as
        plotAlphaBetaSurvivalDiagram() with ln(beta) turned into beta, which straightens the lines. The zones
        are colored by the final mass: below the first of zone_masses in red, between them in yellow, and
        above the last one in green.

    Arguments:
        alpha: [float] Ballistic coefficient.
        beta: [float] Mass loss parameter.
        slope: [float] Entry angle above the horizon (radians).

    Keyword arguments:
        profile: [dict] Result of profileAlphaBetaLag() with the 2D grid. If given, its 1 sigma and 95%
            regions of two parameters are drawn. None by default.
        zone_masses: [tuple] Final masses between the zones (kg), in increasing order. (0.05, 0.1) by default.
        mu_zones: [float] Shape change coefficient of the zones and the solid lines. 0 (no rotation) by
            default.
        mu_other: [float] Shape change coefficient of the dashed lines of the same final masses. 2/3 (fast
            rotation) by default.
        dens: [float] Bulk density (kg/m^3), see alphaBetaMasses(). 3500 by default.
        shape_coeff: [float] Shape coefficient, see alphaBetaMasses(). 0.78 by default.
        gamma: [float] Drag parameter, see alphaBetaMasses(). 1.0 by default.
        axes: [Axes] Axes to draw into. A new figure if None (default).

    Return:
        (fig, ax): [tuple] The figure and the axes.
    """

    import matplotlib.ticker
    import matplotlib.pyplot as plt

    if axes is None:
        fig, ax = plt.subplots(figsize=(7.5, 6))
    else:
        fig, ax = axes.figure, axes

    m0s = (gamma*shape_coeff*RHO_ATM_0*HT_NORM_CONST/(dens**(2/3.0)))**3

    x_point = alpha*np.sin(slope)

    # Extent of the plot, covering the best fit, the 95% region and the lines of the zones at beta = 0
    have_region = (profile is not None) and (profile['grid_delta_chi2'].size > 0)
    x_values = [x_point] + [(m0s/mass)**(1/3.0) for mass in zone_masses]
    y_values = [beta, 1.0]
    if have_region:
        region_alpha, region_beta, _ = profileRegionAlphaBetaLag(profile, n_params=2)
        x_values += list(region_alpha*np.sin(slope))
        y_values += list(region_beta)

    x_range = (0.5*np.min(x_values), 2*np.max(x_values))
    y_range = (0.0, min(1.25*np.max(y_values), ALPHA_BETA_BOUNDS[1][1]))

    x_grid = np.geomspace(*x_range, 400)

    def _betaLine(mu, mass):
        """ Beta at which the final mass is the given one, for every alpha*sin(slope) of the grid. """

        return (1.0 - mu)*(np.log(m0s/mass) - 3*np.log(x_grid))

    # Color the zones by the class of the final mass (red below the first zone mass, yellow between them,
    #   green above) for both mu, as mu is not known. Where the classes of the two mu differ, the zone has
    #   the mix of their colors. The final mass is smaller for the larger mu, so its class is never higher
    zone_colors = np.array([[1.0, 0.0, 0.0], [1.0, 0.84, 0.0], [0.0, 0.6, 0.0]])
    zone_names = ["< {:s}".format(_formatMass(zone_masses[0])), \
        "{:s} - {:s}".format(_formatMass(zone_masses[0]), _formatMass(zone_masses[-1])), \
        "> {:s}".format(_formatMass(zone_masses[-1]))]
    mu_small, mu_large = min(mu_zones, mu_other), max(mu_zones, mu_other)
    combinations = [(c_small, c_large) for c_small in range(3) for c_large in range(c_small + 1)]

    y_grid = np.linspace(*y_range, 300)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

    def _massClass(mu):
        """ Class of the final mass on the grid of the plot, 0 below the first zone mass, 1 between them and
            2 above.
        """

        log_mass = np.log(m0s) - 3*np.log(x_mesh) - y_mesh/(1.0 - mu)

        return (log_mass >= np.log(zone_masses[0])).astype(int) \
            + (log_mass >= np.log(zone_masses[-1])).astype(int)

    class_small, class_large = _massClass(mu_small), _massClass(mu_large)
    combination_index = np.zeros_like(class_small)
    for i, (c_small, c_large) in enumerate(combinations):
        combination_index[(class_small == c_small) & (class_large == c_large)] = i

    combination_colors = [0.5*(zone_colors[c_small] + zone_colors[c_large]) \
        for c_small, c_large in combinations]
    ax.contourf(x_mesh, y_mesh, combination_index, levels=np.arange(len(combinations) + 1) - 0.5, \
        colors=combination_colors, alpha=0.25, zorder=0)

    # Legend entries of the combinations which are in the plot, as the final mass for the two mu
    for i, (c_small, c_large) in enumerate(combinations):
        if np.any(combination_index == i):
            ax.fill_between([], [], [], color=combination_colors[i], alpha=0.25, lw=0, \
                label="{:s} / {:s}".format(zone_names[c_small], zone_names[c_large]))

    # Lines of the final masses between the zones
    for mass, color in zip(zone_masses, ['k', '0.4']):
        for mu, linestyle in [(mu_zones, '-'), (mu_other, '--')]:
            ax.plot(x_grid, _betaLine(mu, mass), color=color, linestyle=linestyle, lw=1.2, \
                label=r"$m_{{final}}$ = {:s}, $\mu$ = {:s}".format(_formatMass(mass), _formatMu(mu)))

    # Regions of the profile likelihood
    if have_region:
        for p, linestyle, level_label in [(0.6827, '-', r"1$\sigma$"), (0.95, '--', "95%")]:
            ax.contour(profile['grid_alpha']*np.sin(slope), profile['grid_beta'], \
                profile['grid_delta_chi2'], levels=[scipy.stats.chi2.ppf(p, 2)], colors='tab:blue', \
                linestyles=linestyle, linewidths=1.3)
            ax.plot([], [], color='tab:blue', linestyle=linestyle, lw=1.3, \
                label=r"{:s} ($\alpha$, $\beta$) region".format(level_label))

    # Best fit
    ax.scatter([x_point], [beta], color='blue', zorder=5, label="Best fit", clip_on=False)

    ax.set_xscale('log')
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)

    # Label the ticks with plain numbers, also the minor ticks if the axis spans less than a decade
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: "{:g}".format(x)))
    if x_range[1]/x_range[0] < 30:
        ax.xaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: "{:g}".format(x)))

    ax.set_xlabel(r"$\alpha \sin(\mathrm{slope})$, dens = " + "{:d} kg/m$^3$, ".format(int(dens)) \
        + r"$\Gamma A$ = " + "{:.2f}".format(gamma*shape_coeff))
    ax.set_ylabel(r"$\beta$")
    ax.legend(fontsize=6, loc='best', title_fontsize=6, framealpha=0.6, \
        title="Final mass, $\\mu$ = {:s} / {:s}".format(_formatMu(min(mu_zones, mu_other)), \
        _formatMu(max(mu_zones, mu_other))))

    return fig, ax



def plotAlphaBetaMasses(masses, axes=None, mass_thresholds=(0.05, 0.1), mass_min=0.001, title=None, \
    note=None):
    """ Plot masses with their 95% intervals as points with horizontal bars on a log axis, with the values
        written next to them, over the zones of plotAlphaBetaMassDiagram(): red below the first mass
        threshold, yellow between the thresholds and green above them.

    Arguments:
        masses: [list] One (label, mass, lower, upper, color) tuple per mass (kg). The lower and the upper
            limit can be None if not known, the lower one 0 if there is no lower limit, and the upper one inf
            if there is no upper limit. A mass which is not defined is NaN.

    Keyword arguments:
        axes: [Axes] Axes to draw into. A new figure if None (default).
        mass_thresholds: [tuple] Masses between the zones (kg), in increasing order. (0.05, 0.1) by default.
        mass_min: [float] Lowest mass on the axis (kg), smaller masses and limits are shown with an arrow at
            its left edge. 0.001 (1 g) by default.
        title: [str] Title of the plot. None by default.
        note: [str] Text written in the middle of the plot, e.g. why the masses are not constrained. None by
            default.

    Return:
        (fig, ax): [tuple] The figure and the axes.
    """

    import matplotlib.pyplot as plt

    if axes is None:
        fig, ax = plt.subplots(figsize=(7.5, 4))
    else:
        fig, ax = axes.figure, axes

    # Range of the axis, from all finite masses and limits
    finite = [m for entry in masses for m in entry[1:4] if (m is not None) and np.isfinite(m) and (m > 0)]
    finite += list(mass_thresholds)

    # Keep the axis between the lowest mass and 1000 t, as the masses of degenerate fits can be absurd
    x_lo, x_hi = max(0.2*np.min(finite), mass_min), min(5*np.max(finite), 1e6)

    for i, (label, mass, lower, upper, color) in enumerate(masses):

        y = len(masses) - i

        if (lower is not None) and (upper is not None):

            # Draw the interval within the axis, with an arrow if it goes beyond the axis
            if upper >= x_lo:
                ax.plot([min(max(lower, x_lo), x_hi), min(upper, x_hi)], [y, y], color=color, lw=4, \
                    alpha=0.4, solid_capstyle='butt')
            if upper > x_hi:
                ax.annotate("", xy=(x_hi, y), xytext=(x_hi/3, y), \
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

            # Mark the interval below the axis or without a lower limit with an arrow to the left edge
            if lower < x_lo:
                ax.annotate("", xy=(x_lo, y), xytext=(3*x_lo, y), \
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

            if (lower <= 0) and (not np.isfinite(upper)):
                interval_text = "not constrained"
            elif lower <= 0:
                interval_text = "< {:s}".format(_formatMass(upper))
            elif not np.isfinite(upper):
                interval_text = "> {:s}".format(_formatMass(lower))
            else:
                interval_text = "{:s} - {:s}".format(_formatMass(lower), _formatMass(upper))
        else:
            interval_text = ""

        if np.isfinite(mass) and (mass > 0):
            if x_lo <= mass <= x_hi:
                ax.scatter([mass], [y], color=color, s=40, zorder=3)

            # Mark the mass below the axis without an interval with an arrow to the left edge
            elif (mass < x_lo) and ((lower is None) or (upper is None)):
                ax.annotate("", xy=(x_lo, y), xytext=(3*x_lo, y), \
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

            value_text = _formatMass(mass)
        elif mass == 0:
            value_text = "~0"
        else:
            value_text = "not defined"

        # Write the value and the interval above the bar
        ax.text(0.02, y + 0.22, "{:s}{:s}".format(value_text, "  (95%: {:s})".format(interval_text) \
            if interval_text else ""), transform=ax.get_yaxis_transform(), va='center', fontsize=8)

    # Color the zones of the final mass
    ax.axvspan(x_lo, mass_thresholds[0], color='red', alpha=0.1, lw=0, zorder=0)
    ax.axvspan(mass_thresholds[0], mass_thresholds[-1], color='gold', alpha=0.15, lw=0, zorder=0)
    ax.axvspan(mass_thresholds[-1], x_hi, color='green', alpha=0.1, lw=0, zorder=0)

    ax.set_xscale('log')
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0.4, len(masses) + 0.6)
    ax.set_yticks(np.arange(len(masses), 0, -1))
    ax.set_yticklabels([entry[0] for entry in masses], fontsize=8)
    ax.set_xlabel("Mass (kg)")
    if title is not None:
        ax.set_title(title, fontsize=9)

    if note is not None:
        ax.text(0.5, 0.5, note, transform=ax.transAxes, ha='center', va='center', fontsize=10, \
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='0.5'))

    return fig, ax



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


def _lightCurveResiduals(x, mu, ht_normed_lc, mag_lc, sigma_mag, fast=False):
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

    Keyword arguments:
        fast: [bool] Passed through to alphaBetaModelMagnitude() - see alphaBetaVelocityNormed()'s
            docstring.

    Return:
        res: [ndarray] Sigma-normalized magnitude residuals.
    """

    alpha, beta = np.exp(x)

    mag_zero, _ = alphaBetaModelMagnitude(ht_normed_lc, alpha, beta, mu, fast=fast)
    mag_offset = _profiledMagOffset(mag_lc, mag_zero, sigma_mag)

    return (mag_zero + mag_offset - mag_lc)/sigma_mag


def _jointResiduals(x, mu, v_normed, ht_normed, ht_normed_lc, mag_lc, sigma_v_normed, sigma_mag,
        dyn_weight, lc_weight, fast=False):
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

    Keyword arguments:
        fast: [bool] Passed through to _dynResiduals()/_lightCurveResiduals() - see
            alphaBetaVelocityNormed()'s docstring. This is the residual scipy.optimize.least_squares
            calls every iteration of fitAlphaBetaLightCurve(), so fast=True is where the LUT
            speedup actually matters for this fit.

    Return:
        res: [ndarray] Concatenated (dynamics, light curve) sigma-normalized residuals.
    """

    res_dyn = dyn_weight*_dynResiduals(x, v_normed, ht_normed, sigma_v_normed, fast=fast)
    res_lc = lc_weight*_lightCurveResiduals(x, mu, ht_normed_lc, mag_lc, sigma_mag, fast=fast)

    return np.concatenate([res_dyn, res_lc])


def _lightCurveResidualsFreeMu(x, ht_normed_lc, mag_lc, sigma_mag, fast=False):
    """ Same as _lightCurveResiduals(), but with mu folded into x as a 3rd free parameter
        (x = (ln alpha, ln beta, mu)) instead of held fixed. Used only to derive an effective
        sigma_mag for the optional free-mu fit in fitAlphaBetaLightCurve() (fit_free_mu=True),
        mirroring how _lightCurveResiduals() is used to derive sigma_mag at each fixed mu.

    Arguments:
        x: [ndarray] Fit parameters (ln alpha, ln beta, mu).
        ht_normed_lc: [ndarray] Light curve heights normalized to HT_NORM_CONST.
        mag_lc: [ndarray] Observed absolute magnitudes.
        sigma_mag: [float or ndarray] Magnitude uncertainty, a scalar or per-point.

    Keyword arguments:
        fast: [bool] Passed through to _lightCurveResiduals() - see alphaBetaVelocityNormed()'s
            docstring.

    Return:
        res: [ndarray] Sigma-normalized magnitude residuals.
    """

    return _lightCurveResiduals(x[:2], x[2], ht_normed_lc, mag_lc, sigma_mag, fast=fast)


def _jointResidualsFreeMu(x, v_normed, ht_normed, ht_normed_lc, mag_lc, sigma_v_normed, sigma_mag,
        dyn_weight, lc_weight, fast=False):
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

    Keyword arguments:
        fast: [bool] Passed through to _jointResiduals() - see alphaBetaVelocityNormed()'s
            docstring.

    Return:
        res: [ndarray] Concatenated (dynamics, light curve) sigma-normalized residuals.
    """

    return _jointResiduals(x[:2], x[2], v_normed, ht_normed, ht_normed_lc, mag_lc, sigma_v_normed,
        sigma_mag, dyn_weight, lc_weight, fast=fast)


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


def alphaBetaModelMagnitude(ht_normed, alpha, beta, mu, mag_offset=0.0, fast=False):
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
        fast: [bool] Passed through to alphaBetaVelocityNormed() - see its docstring.

    Return:
        (mag, v_model): [tuple] Model absolute magnitudes and model normalized velocities.
    """

    v_model = alphaBetaVelocityNormed(ht_normed, alpha, beta, fast=fast)

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
        plot=False,
        fast=False,
        lc_density_ratio=None):
    """ Simultaneous fit of the alpha-beta model to the dynamics (height vs. velocity) AND the
        light curve (height vs. absolute magnitude), following Gritsevich (2007, 2009) for the
        dynamics and Gritsevich & Koschny (2011) for the luminosity.

    WARNING: Like the whole alpha-beta formalism, this fit assumes a SINGLE, non-fragmenting
        body ablating quasi-continuously. Most real fireballs fragment, and a fragmenting event's
        light curve (flares, sudden brightenings) does not follow Eq. (13)-(14) below - so this
        joint fit, and especially the photometric outputs derived from it (M0, K, and any
        luminous efficiency computed with alphaBetaLuminousEfficiency()), is NOT expected to be
        valid for the majority of real events. Only apply it to events consistent with
        single-body ablation, and inspect the light curve fit visually (plot=True) before
        trusting the photometric results.

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
        - This function does NOT return parameter error estimates: a Gauss-Newton curvature
          estimate on the *robustified* (soft-L1) joint cost would not be a rigorous statistical
          covariance (soft-L1 reweights residuals nonlinearly, so (J^T J)^-1 is only a local
          approximation, not the inverse Fisher information of Gaussian noise - and the profiled
          weighted-median M0 makes the residual vector only piecewise-smooth in (alpha, beta),
          which further degrades a finite-difference Jacobian). Rather than surface a rough
          estimate that could be mistaken for a rigorous one, get error bars separately: run
          fitAlphaBeta(method='robust', estimate_errors=True) on the dynamics alone, or
          profileAlphaBeta() for likelihood-based confidence intervals.
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
        lc_density_ratio: [ndarray] Real air density over the model density at the light curve points, from
            rescaleHeightToExponentialAtmosphere(..., return_density_ratio=True), needed with the column
            mapping. The observed magnitudes are corrected by +2.5 log10(ratio) for the fit and the plot.
            None by default, for heights rescaled by density.
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
        fast: [bool] Passed through everywhere this function inverts velocity (every joint/
            photometric-only least_squares stage, plus the post-fit velocity/magnitude curves used
            for verbose/plot output) - see alphaBetaVelocityNormed()'s docstring.

    Return:
        results: [dict] With keys:
            'v_init': adopted initial velocity (m/s),
            'sigma_v': adopted velocity uncertainty (m/s, see the sigma_v caveat above),
            'alpha_dyn', 'beta_dyn': Stage-0 (dynamics only, Q4) values,
            'fits': {mu: {'alpha', 'beta',
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
                as one 'fits'[mu] entry, plus 'mu' (the fitted shape-change coefficient itself).
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

    # Correct the observed magnitudes for the real air density at the points
    if lc_density_ratio is not None:
        lc_density_ratio = np.broadcast_to(np.asarray(lc_density_ratio, dtype=np.float64), mag_abs_data.shape)
        mag_abs_data = mag_abs_data + 2.5*np.log10(lc_density_ratio)

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
    alpha0, beta0 = minimizeAlphaBeta(v_normed, ht_normed, fast=fast)

    # Estimate sigma_v from the Q4 residuals if not given (robust MAD estimator, shared with
    # fitAlphaBeta(method='robust')). Always derived from the Q4 fit specifically, even when
    # dyn_method='robust' below, to avoid the circularity of a fit weighting itself by its own
    # residuals.
    sigma_v_given = sigma_v is not None
    if not sigma_v_given:
        sigma_v = _estimateSigmaV(alpha0, beta0, v_normed, ht_normed, v_init, fast=fast)

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
    # searches in (alpha and beta span several orders of magnitude between their bounds - see
    # _makeBoundChecker())
    bound_tol = 1e-2
    warnings_list = []

    _checkLogBound = _makeBoundChecker(warnings_list, tol=bound_tol, log=True)

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
                args=(mu, ht_normed_lc, mag_abs_data, 1.0, fast), bounds=log_bounds, **lsq_kwargs)

            sigma_mag_mu = 1.4826*np.median(np.abs(res_prelim.fun - np.median(res_prelim.fun)))

            # Floor at 0.01 mag
            sigma_mag_mu = max(sigma_mag_mu, 0.01)

        res = scipy.optimize.least_squares(_jointResiduals, x0, args=(mu, v_normed, ht_normed, \
            ht_normed_lc, mag_abs_data, sigma_v_normed, sigma_mag_mu, dyn_weight, lc_weight, fast),
            bounds=log_bounds, **lsq_kwargs)

        if not res.success:
            print("WARNING: Optimizer failed for μ={:.3f}: {:s}".format(mu, res.message))

        alpha, beta = np.exp(res.x)

        # Recompute the profiled amplitude at the solution
        mag_zero, _ = alphaBetaModelMagnitude(ht_normed_lc, alpha, beta, mu, fast=fast)
        mag_offset = _profiledMagOffset(mag_abs_data, mag_zero, sigma_mag_mu)

        # Amplitude K = tau M_e V_e^3 sin(gamma)/(2 h0) in W
        lum_amplitude = p0m*10**(-0.4*mag_offset)

        fits[mu] = {
            'alpha': alpha,
            'beta': beta,
            'mag_offset': mag_offset,
            'K': lum_amplitude,
            'sigma_mag': sigma_mag_mu,
            'cost': res.cost,
            'success': res.success,
        }

        _checkLogBound("alpha(μ={:.3f})".format(mu), alpha, ALPHA_BETA_BOUNDS[0][0], ALPHA_BETA_BOUNDS[0][1])
        _checkLogBound("beta(μ={:.3f})".format(mu), beta, ALPHA_BETA_BOUNDS[1][0], ALPHA_BETA_BOUNDS[1][1])

        if verbose:
            print()
            print("--- μ = {:.3f} ---".format(mu))
            if not sigma_mag_given:
                print("sigma_mag            = {:.3f} mag  (DERIVED for this μ: MAD of an "
                    "unweighted photometry-only fit)".format(sigma_mag_mu))
            print("alpha                = {:.6f}".format(alpha))
            print("beta                 = {:.6f}".format(beta))
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
                args=(ht_normed_lc, mag_abs_data, 1.0, fast), bounds=bounds_free, **lsq_kwargs)

            sigma_mag_free = 1.4826*np.median(
                np.abs(res_prelim_free.fun - np.median(res_prelim_free.fun)))

            # Floor at 0.01 mag
            sigma_mag_free = max(sigma_mag_free, 0.01)

        res_free = scipy.optimize.least_squares(_jointResidualsFreeMu, x0_free, args=(v_normed, \
            ht_normed, ht_normed_lc, mag_abs_data, sigma_v_normed, sigma_mag_free, dyn_weight,
            lc_weight, fast), bounds=bounds_free, **lsq_kwargs)

        if not res_free.success:
            print("WARNING: Optimizer failed for the free-μ fit: {:s}".format(res_free.message))

        alpha_free, beta_free = np.exp(res_free.x[:2])
        mu_free = float(np.clip(res_free.x[2], mu_lo, mu_hi))

        mag_zero_free, _ = alphaBetaModelMagnitude(ht_normed_lc, alpha_free, beta_free, mu_free, \
            fast=fast)
        mag_offset_free = _profiledMagOffset(mag_abs_data, mag_zero_free, sigma_mag_free)
        lum_amplitude_free = p0m*10**(-0.4*mag_offset_free)

        mu_free_fit = {
            'mu': mu_free,
            'alpha': alpha_free,
            'beta': beta_free,
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
        _checkLogBound("alpha(free fit)", alpha_free, ALPHA_BETA_BOUNDS[0][0], ALPHA_BETA_BOUNDS[0][1])
        _checkLogBound("beta(free fit)", beta_free, ALPHA_BETA_BOUNDS[1][0], ALPHA_BETA_BOUNDS[1][1])

        if verbose:
            print()
            print("--- Free-μ fit (μ estimated jointly with α, β) ---")
            if not sigma_mag_given:
                print("sigma_mag            = {:.3f} mag  (DERIVED: MAD of an unweighted "
                    "photometry-only fit with μ also free)".format(sigma_mag_free))
            print("mu                   = {:.3f}".format(mu_free))
            print("alpha                = {:.6f}".format(alpha_free))
            print("beta                 = {:.6f}".format(beta_free))
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
            label="Observed light curve" if lc_density_ratio is None else \
            "Observed light curve, corrected to the model's air density")

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
        vel_arr_dyn = alphaBetaVelocityNormed(ht_arr_normed, alpha0, beta0, fast=fast)*v_init
        ax_dyn.plot(vel_arr_dyn/1000, ht_arr/1000, color='k', linestyle='--', linewidth=1.5, \
            label="Dynamics-only ({:s}): $\\alpha$={:.2f}, $\\beta$={:.2f}".format(
                'Q4' if dyn_method == 'q4' else 'robust', alpha0, beta0))

        for mu, color in zip(mu_values, colors):

            fit = fits[mu]
            alpha, beta = fit['alpha'], fit['beta']

            # --- Joint fit (dynamics + light curve), the main result for this mu ---
            vel_arr = alphaBetaVelocityNormed(ht_arr_normed, alpha, beta, fast=fast)*v_init
            mag_arr, _ = alphaBetaModelMagnitude(ht_arr_normed, alpha, beta, mu, \
                mag_offset=fit['mag_offset'], fast=fast)

            ax_dyn.plot(vel_arr/1000, ht_arr/1000, color=color, \
                label="Joint fit, $\\mu$={:.2f}: $\\alpha$={:.2f}, $\\beta$={:.2f}".format(
                    mu, alpha, beta))
            ax_lc.plot(mag_arr, ht_arr/1000, color=color, \
                label="Joint fit, $\\mu$={:.2f}: $M_0$={:.2f}, $K$={:.2e} W".format(
                    mu, fit['mag_offset'], fit['K']))

            # --- Dynamics-only fit (alpha0, beta0 from above), projected onto the light curve:
            # same shape for every mu, only M0 is re-profiled per mu since f(v) depends on mu.
            # Shows how well the dynamics alone would have predicted the light curve shape.
            mag_zero_dyn, _ = alphaBetaModelMagnitude(ht_normed_lc, alpha0, beta0, mu, fast=fast)
            mag_offset_dyn = _profiledMagOffset(mag_abs_data, mag_zero_dyn, fits[mu]['sigma_mag'])
            mag_arr_dyn, _ = alphaBetaModelMagnitude(ht_arr_normed, alpha0, beta0, mu, \
                mag_offset=mag_offset_dyn, fast=fast)

            ax_lc.plot(mag_arr_dyn, ht_arr/1000, color=color, linestyle='--', linewidth=1.5, \
                label="Dynamics-only, $\\mu$={:.2f}: $M_0$={:.2f}".format(mu, mag_offset_dyn))

            # --- Purely photometric fit: alpha/beta constrained by the light curve alone
            # (dynamics ignored entirely), for comparison with the joint and dynamics-only fits.
            res_phot = scipy.optimize.least_squares(_lightCurveResiduals, x0, \
                args=(mu, ht_normed_lc, mag_abs_data, fits[mu]['sigma_mag'], fast), \
                bounds=log_bounds, **lsq_kwargs)

            alpha_phot, beta_phot = np.exp(res_phot.x)
            mag_zero_phot, _ = alphaBetaModelMagnitude(ht_normed_lc, alpha_phot, beta_phot, mu, \
                fast=fast)
            mag_offset_phot = _profiledMagOffset(mag_abs_data, mag_zero_phot, fits[mu]['sigma_mag'])

            vel_arr_phot = alphaBetaVelocityNormed(ht_arr_normed, alpha_phot, beta_phot, \
                fast=fast)*v_init
            mag_arr_phot, _ = alphaBetaModelMagnitude(ht_arr_normed, alpha_phot, beta_phot, mu, \
                mag_offset=mag_offset_phot, fast=fast)

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
                mu_free_fit['beta'], fast=fast)*v_init
            mag_arr_free, _ = alphaBetaModelMagnitude(ht_arr_normed, mu_free_fit['alpha'],
                mu_free_fit['beta'], mu_free_fit['mu'], mag_offset=mu_free_fit['mag_offset'], \
                fast=fast)

            ax_dyn.plot(vel_arr_free/1000, ht_arr/1000, color='crimson', linewidth=2.5, zorder=4, \
                label="Free-$\\mu$ fit: $\\mu$={:.2f}, $\\alpha$={:.2f}, "
                    "$\\beta$={:.2f}".format(mu_free_fit['mu'], mu_free_fit['alpha'],
                    mu_free_fit['beta']))
            ax_lc.plot(mag_arr_free, ht_arr/1000, color='crimson', linewidth=2.5, zorder=4, \
                label="Free-$\\mu$ fit: $\\mu$={:.2f}, $M_0$={:.2f}".format(
                    mu_free_fit['mu'], mu_free_fit['mag_offset']))

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
            title += "   |   Free-$\\mu$ fit: $\\mu$ = {:.2f}".format(mu_free_fit['mu'])

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
        shape_coeff=GAMMA_A, gamma=1.0, v_final=None, verbose=False):
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
    from wmpl.Trajectory.Trajectory import dropTimeOrigin



    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Fit the alpha-beta model to the lag of the trajectory.")

    arg_parser.add_argument('traj_path', nargs="?", metavar='TRAJ_PATH', type=str, \
        help="Path to the trajectory pickle file.")

    arg_parser.add_argument('--lagsmooth', action="store_true", \
        help="Also fit a smooth model to the lag (exponential, then constant deceleration) and show its "
        "velocities, dynamic mass and dynamic pressure, for comparison with the alpha-beta fit.")

    arg_parser.add_argument('-d', '--dens', metavar='DENS', \
        help='Bulk density in kg/m^3 used to compute the final dynamic mass. Default is 3500 kg/m^3.', \
        type=float, default=3500)

    arg_parser.add_argument('-g', '--ga', metavar='GAMMA_A', \
        help='The product of the drag coefficient Gamma and the shape coefficient A. Used for computing the alpha-beta masses and the dynamic mass. Default is {:.2f}.'.format(GAMMA_A), \
        type=float, default=GAMMA_A)

    arg_parser.add_argument('-e', '--errors', action="store_true", \
        help="Estimate and report alpha/beta and initial/final mass uncertainties, from the covariance "
        "of the fit of the lag. Propagates the fitted (ln alpha, ln beta) covariance into the masses, plots "
        "the profile likelihood of alpha and beta, draws its 1 sigma and 95%% regions on the survival diagram, "
        "and the band of the models within the 95%% region on the velocities and the lag.")

    arg_parser.add_argument('--slopeunc', metavar='SLOPE_UNC', type=float, default=None, \
        help="1-sigma uncertainty on the entry slope, in DEGREES, folded into the mass error "
        "estimate when --errors is set. Default: the slope is treated as exactly known.")

    arg_parser.add_argument('--densunc', metavar='DENS_UNC', type=float, default=None, \
        help="1-sigma uncertainty on the bulk density, in kg/m^3, folded into the mass error "
        "estimate when --errors is set. Default: the density is treated as exactly known.")

    arg_parser.add_argument('--lagfit', choices=['full', 'simple'], default='full', \
        help="Fit of the lag. 'full' (default) fits the entry lag against the slant air column traversed " \
        "along the path, with an offset and a noise level for every station, which gives the entry " \
        "velocity and is unaffected by constant clock offsets between the stations (fitAlphaBetaEntryLag). " \
        "'simple' fits the distances along the path with one offset and one noise level for all stations, " \
        "with the heights mapped by --atmrescale (fitAlphaBetaLag).")

    arg_parser.add_argument('--atmrescale', choices=['column', 'density'], default='column', \
        help="How the heights are mapped to the exponential atmosphere of the alpha-beta model by the " \
        "simple fit of the lag. 'column' (default) maps them to the height with the same air column above, " \
        "and 'density' to the height with the same local density, which biases alpha and beta where the " \
        "scale height is not 7.16 km.")

    arg_parser.add_argument('-x', '--hideplots', action="store_true", \
        help="Don't show the plots on the screen. Combine with --saveplots to only write them to disk.")

    arg_parser.add_argument('-s', '--saveplots', action="store_true", \
        help="Save the plots as PNGs to the trajectory pickle directory, named after the trajectory "
        "timestamp: the summary (e.g. 20191023_091225_alpha_beta_summary.png) and the dynamic pressure.")

    # Add the atmosphere model options
    addAtmosphereArguments(arg_parser)

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    # Apply the atmosphere model options
    setAtmosphere(cml_args)

    #########################


    # If the trajectory pickle was given, load the orbital elements from it
    if cml_args.traj_path is not None:

        # Load the trajectory pickle
        traj = loadPickle(*os.path.split(cml_args.traj_path))

        # Plots are saved next to the trajectory pickle
        dir_path = os.path.dirname(cml_args.traj_path)

        def _finishPlot(fig, suffix):
            """ Save the figure (if --saveplots) and show it (unless --hideplots). """

            if cml_args.saveplots:
                fig.savefig(os.path.join(dir_path, traj.file_name + "_" + suffix + ".png"), dpi=300)

            if cml_args.hideplots:
                plt.close(fig)
            else:
                plt.show()

        # Dictionary for storing time, height, and velocity data per station
        ht_data_dict = {}

        # Construct an input data array
        ht_data = []
        time_data = []
        lat_data = []
        lon_data = []
        vel_data = []
        lag_data = []
        length_data = []
        station_data = []
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

            # Distance along the path from the state vector, the same reference for all stations
            length_data += obs.state_vect_dist[filter_mask].tolist()

            station_data += [str(obs.station_id)]*int(np.count_nonzero(filter_mask))

            # Store the data for each station
            ht_data_dict[obs.station_id] = [obs.time_data[filter_mask], obs.model_ht[filter_mask], 
                                            obs.velocities[filter_mask]]


        ht_data = np.array(ht_data)
        time_data = np.array(time_data)
        lat_data = np.array(lat_data)
        lon_data = np.array(lon_data)
        vel_data = np.array(vel_data)
        lag_data = np.array(lag_data)
        length_data = np.array(length_data)
        station_data = np.array(station_data)

        # Use the colors of the stations in the trajectory plots
        obs_ht_sorted = sorted(traj.observations, key=lambda x: x.model_ht[0])
        station_colors, station_alpha = stationColors(len(obs_ht_sorted))
        station_color_list = station_colors
        station_colors = {}
        station_ids = []
        for obs, color in zip(obs_ht_sorted, station_color_list):
            station_id = str(obs.station_id)
            station_colors.setdefault(station_id, color)
            if (station_id in station_data) and (station_id not in station_ids):
                station_ids.append(station_id)


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
        length_data = length_data[sorted_indices]
        time_data = time_data[sorted_indices]
        station_data = station_data[sorted_indices]
        lat_data = lat_data[sorted_indices]
        lon_data = lon_data[sorted_indices]
        ht_data  = ht_data[sorted_indices]


        # Slope of the trajectory for the masses, which also sets the unit of alpha of the full fit
        slope = traj.orbit.elevation_apparent_norot

        # The apparent elevation of the radiant is negative if the direction of the solution is reversed, or
        #   if the meteor climbs, e.g. an Earth-grazer. Then the local slope of the observed path along the
        #   direction of the motion is used, and if the meteor climbs, the alpha-beta model does not apply
        if slope <= 0:

            motion_sign = np.sign(np.polyfit(time_data, length_data, 1)[0])
            slope = np.arcsin(np.clip(-motion_sign*np.polyfit(length_data, ht_data, 1)[0], -1.0, 1.0))

            if slope <= 0:
                print()
                print("ERROR: The meteor climbs along the observed path (slope {:.1f} deg), e.g. an " \
                    "Earth-grazer or a solution with a wrong timing, and the alpha-beta model needs a " \
                    "descending meteor".format(np.degrees(slope)))
                sys.exit(1)

            print("NOTE: The apparent elevation of the radiant is {:.1f} deg, so the direction of the " \
                "trajectory solution is likely reversed, check its timing. The local slope of the observed " \
                "path, {:.1f} deg, is used instead".format(np.degrees(traj.orbit.elevation_apparent_norot), \
                np.degrees(slope)))

        # Keep the distances on the line of the solution, from which the full fit forms its lags
        length_data_line = np.copy(length_data)

        # Take out the distance and the speed gained from gravity along the path, as the solver does for the
        #   lag, because the alpha-beta model has no gravity
        try:
            length_data = length_data - traj.gravityLengthAlongPath(traj.observations, time_data)
            vel_data_nograv = vel_data - traj.gravityAlongPath()*(time_data \
                - dropTimeOrigin(traj.observations))

        except AttributeError:
            print("NOTE: The trajectory has no gravity information, the gravity along the path is ignored")
            vel_data_nograv = vel_data

        if cml_args.lagfit == 'full':

            try:

                # Normalized heights of the slant air column along the path flown through the air, with the
                #   density profile of the event down to the lowest height of the plots, and alpha in the unit
                #   of the slope of the masses
                profile_hts, profile_dens = airDensityProfile(lat_data, lon_data, ht_data, traj.jdt_ref, \
                    ht_min=min(10000.0, np.min(ht_data) - RESCALE_STEP_FINE))
                air_speed = airSpeedAlongPath(traj.state_vect_mini, traj.radiant_eci_mini)
                slant_column, length_air = slantColumnFromPoints(time_data, length_data_line, ht_data, \
                    profile_hts, profile_dens, np.sin(slope), air_speed=air_speed)
                ht_normed_data = slant_column.htNormed(length_air)
                ht_data_rescaled = HT_NORM_CONST*ht_normed_data

                # Lags of all stations relative to the initial velocity of the solution, without the distance
                #   added by gravity along the line, which is computed at every point
                if getattr(traj, 'gravity_correction', False):
                    gravity_length = gravityLengthAlongLine(time_data, length_data_line, \
                        traj.state_vect_mini, traj.radiant_eci_mini, \
                        gravity_factor=getattr(traj, 'gravity_factor', 1.0))
                else:
                    gravity_length = np.zeros_like(time_data)

                entry_lag_data = length_data_line - traj.v_init*time_data - gravity_length

                # Noise of the distances of every station
                sigma_length = stationLengthNoise(time_data, length_data_line, station_data)

            except ValueError as e:

                # Fall back to the simple fit if the full fit cannot be set up, e.g. if the path does not
                #   descend at its first point
                print("NOTE: {:s}, so the simple fit of the lag is used (--lagfit simple)".format(str(e)))
                cml_args.lagfit = 'simple'

        if cml_args.lagfit == 'simple':

            # Rescale the heights to the exponential atmosphere used by alpha-beta
            ht_data_rescaled = rescaleHeightToExponentialAtmosphere(lat_data, lon_data, ht_data, \
                traj.jdt_ref, method=cml_args.atmrescale)

        # Fit alpha-beta directly to the lag, starting also from the Q4 fit of the observed velocities
        print("Fitting alpha-beta to the lag...")
        x0_alpha_beta = [fitAlphaBeta(vel_data_nograv, ht_data_rescaled, v_init=traj.v_init)[1:]]

        if cml_args.lagfit == 'full':

            try:

                def _fitEntryLag(x0_alpha_beta):
                    """ Fit the entry lag and compute the profile intervals of alpha and beta. """

                    fit_result = fitAlphaBetaEntryLag(time_data, entry_lag_data, ht_normed_data, \
                        station_data, slant_column.lengthScale, sigma_length, traj.v_init, \
                        x0_alpha_beta=x0_alpha_beta, estimate_errors=cml_args.errors, air_speed=air_speed)

                    # Profile intervals of alpha and beta, with the likelihood on a grid for the plot of the
                    #   uncertainties
                    profile = profileAlphaBetaEntryLag(time_data, entry_lag_data, ht_normed_data, \
                        station_data, slant_column.lengthScale, fit_result[1], fit_result[2], fit_result[4], \
                        n_grid=(40 if cml_args.errors else 0))

                    return fit_result, profile


                fit_result, lag_profile = _fitEntryLag(x0_alpha_beta)

                # If the profile found a better fit, the fit ended in a local minimum, so fit again from that
                #   point
                for _ in range(2):

                    if lag_profile['min_delta_chi2'] > -0.1:
                        break

                    print("NOTE: The profile found a better fit (chi-squared lower by {:.1f}), fitting " \
                        "again from it".format(-lag_profile['min_delta_chi2']))

                    x0_alpha_beta.append(lag_profile['min_model'][:2])
                    fit_result, lag_profile = _fitEntryLag(x0_alpha_beta)

                v_init, alpha, beta, lag_residuals, entry_fit = fit_result[:5]
                fit_errors = fit_result[5] if cml_args.errors else None

            except ValueError as e:

                # Fall back to the simple fit if no model of the full fit reaches the points
                print("NOTE: {:s}, so the simple fit of the lag is used (--lagfit simple)".format(str(e)))
                cml_args.lagfit = 'simple'
                ht_data_rescaled = rescaleHeightToExponentialAtmosphere(lat_data, lon_data, ht_data, \
                    traj.jdt_ref, method=cml_args.atmrescale)
                x0_alpha_beta = [fitAlphaBeta(vel_data_nograv, ht_data_rescaled, v_init=traj.v_init)[1:]]

        if cml_args.lagfit == 'simple':

            if cml_args.errors:
                v_init, alpha, beta, lag_residuals, fit_errors = fitAlphaBetaLag(time_data, length_data, \
                    ht_data_rescaled, traj.v_init, x0_alpha_beta=x0_alpha_beta, estimate_errors=True)
            else:
                v_init, alpha, beta, lag_residuals = fitAlphaBetaLag(time_data, length_data, \
                    ht_data_rescaled, traj.v_init, x0_alpha_beta=x0_alpha_beta)
                fit_errors = None

            # Profile intervals of alpha and beta, with the likelihood on a grid for the plot of the
            #   uncertainties
            lag_profile = profileAlphaBetaLag(time_data, length_data, ht_data_rescaled, v_init, alpha, beta, \
                v_init_start=traj.v_init, n_grid=(40 if cml_args.errors else 0))

            # If the profile found a better fit, the fit ended in a local minimum, so fit again from that
            #   point
            for _ in range(2):

                if lag_profile['min_delta_chi2'] > -0.1:
                    break

                print("NOTE: The profile found a better fit (chi-squared lower by {:.1f}), fitting again " \
                    "from it".format(-lag_profile['min_delta_chi2']))

                x0_alpha_beta.append(lag_profile['min_model'])
                if cml_args.errors:
                    v_init, alpha, beta, lag_residuals, fit_errors = fitAlphaBetaLag(time_data, length_data, \
                        ht_data_rescaled, traj.v_init, x0_alpha_beta=x0_alpha_beta, estimate_errors=True)
                else:
                    v_init, alpha, beta, lag_residuals = fitAlphaBetaLag(time_data, length_data, \
                        ht_data_rescaled, traj.v_init, x0_alpha_beta=x0_alpha_beta)

                lag_profile = profileAlphaBetaLag(time_data, length_data, ht_data_rescaled, v_init, alpha, \
                    beta, v_init_start=traj.v_init, n_grid=(40 if cml_args.errors else 0))

        # Speed of the air along the path, which is added to the velocities of the model of the full fit,
        #   relative to the air, to compare them with the inertial velocities of the trajectory
        air_offset = air_speed if (cml_args.lagfit == 'full') else 0.0
        v_init_inertial = v_init + air_offset

        # Residuals of the distances at the times of the points, from the lag residuals of the full fit, which
        #   grow as one over the velocity of the model, or directly from the simple fit
        if cml_args.lagfit == 'full':
            length_residuals = lag_residuals*entry_fit['v_model']
        else:
            length_residuals = lag_residuals

        # Models within the 95% interval of one parameter (with the best fit), whose range of the masses, the
        #   dynamic mass and the peak dynamic pressure gives their 95% intervals (only with --errors, which
        #   computes the 2D grid)
        interval_models = list(zip(*profileRegionAlphaBetaLag(lag_profile, n_params=1)))
        if interval_models:
            interval_models.append((alpha, beta, v_init))

        def _intervalText(values, fmt, no_lower=False, no_upper=False):
            """ Text of the 95% interval of a derived quantity from its values for the models of the interval,
                empty without them, with the sides which are not constrained by the data marked.
            """

            if not len(values):
                return ""

            if no_lower and no_upper:
                return "  [95% profile: not constrained]"

            if no_upper:
                return "  [95% profile: > " + fmt.format(np.min(values)) + " (no upper limit)]"

            if no_lower:
                return "  [95% profile: < " + fmt.format(np.max(values)) + " (no lower limit)]"

            return "  [95% profile: " + fmt.format(np.min(values)) + " - " + fmt.format(np.max(values)) + "]"

        # Fit the smooth model of the lag (exponential, then constant deceleration), if asked for
        if cml_args.lagsmooth:
            print("Fitting lag function...")
            vel_data_smooth, lag_fit_params = lagFitVelocity(time_data, lag_data, vel_data, traj.v_init)

        # Estimate the final velocity from the fitted alpha-beta solution, relative to the air for the masses
        vel_end = alphaBetaVelocity(ht_data_rescaled[0], alpha, beta, v_init)

        # Optional independent slope/density uncertainties (only used with --errors). Velocities
        #   are deliberately left exact: vel_end above is DERIVED from the fitted alpha/beta, so
        #   its uncertainty is already carried by cov_log - passing an independent sigma_vel_*
        #   would double-count it.
        sigma_slope = np.radians(cml_args.slopeunc) \
            if (cml_args.errors and (cml_args.slopeunc is not None)) else None

        # The full fit fixes alpha*sin(slope) by the data, so the slope only sets the unit of alpha and its
        #   uncertainty does not change the masses
        if (cml_args.lagfit == 'full') and (sigma_slope is not None):
            print("NOTE: --slopeunc is not used with the full fit, as the slope does not change its masses")
            sigma_slope = None
        sigma_dens = cml_args.densunc if (cml_args.errors and (cml_args.densunc is not None)) else None

        # The interval of beta reaches zero at the lower bound of the fit, where ln(beta) goes to minus
        #   infinity and the Wald uncertainties from the covariance of ln alpha and ln beta are not meaningful
        beta_at_zero = lag_profile['beta_lower_at_bound']
        if beta_at_zero and ((sigma_dens is not None) or (sigma_slope is not None)):
            print("NOTE: --densunc and --slopeunc are not used, as the Wald uncertainties are not computed " \
                "when beta reaches zero")

        # Compute initial and final mass (with error propagation if --errors was set and beta is not
        #   consistent with zero)
        if cml_args.errors and (not beta_at_zero):
            m_init_mu0, m_final_mu0, mass_err_mu0 = alphaBetaMasses(alpha, beta, slope, \
                mu=0, dens=cml_args.dens, shape_coeff=cml_args.ga, gamma=1.0, \
                vel_init=v_init, vel_end=vel_end, estimate_errors=True, \
                cov_log=fit_errors['cov_log'], sigma_slope=sigma_slope, sigma_dens=sigma_dens)
            m_init_mu23, m_final_mu23, mass_err_mu23 = alphaBetaMasses(alpha, beta, slope, \
                mu=2/3, dens=cml_args.dens, shape_coeff=cml_args.ga, gamma=1.0, \
                vel_init=v_init, vel_end=vel_end, estimate_errors=True, \
                cov_log=fit_errors['cov_log'], sigma_slope=sigma_slope, sigma_dens=sigma_dens)
        else:
            m_init_mu0, m_final_mu0 = alphaBetaMasses(alpha, beta, slope, \
                mu=0, dens=cml_args.dens, shape_coeff=cml_args.ga, gamma=1.0, \
                vel_init=v_init, vel_end=vel_end)
            m_init_mu23, m_final_mu23 = alphaBetaMasses(alpha, beta, slope, \
                mu=2/3, dens=cml_args.dens, shape_coeff=cml_args.ga, gamma=1.0, \
                vel_init=v_init, vel_end=vel_end)
            mass_err_mu0 = None
            mass_err_mu23 = None


        print()
        # Report what was fitted, how the heights were mapped and the velocities involved, as the alpha-beta
        #   entry velocity is the asymptote of the model above the atmosphere without gravity, not the speed at
        #   the first point, which the initial velocity of the trajectory is
        ht_first = np.max(ht_data)
        v_model_first = alphaBetaVelocity(ht_data_rescaled[np.argmax(ht_data)], alpha, beta, v_init) \
            + air_offset

        print("Alpha-beta fit to the lag of {:d} points, {:.1f} to {:.1f} km".format(len(ht_data), \
            ht_first/1000, np.min(ht_data)/1000))
        if cml_args.lagfit == 'full':
            print("  Full fit of the entry lag against the slant air column along the path (--lagfit " \
                "full), with an offset and a noise level for every station")
            print("  Speed of the air along the path from the rotation of the Earth = {:.1f} m/s".format( \
                air_speed))
            print("  Reduced chi-squared = {:.2f}, integrated autocorrelation time of the residuals = " \
                "{:.2f}".format(entry_fit['chi2_nu'], entry_fit['tau_int']))

            # The weights of the points depend on the model, so the fits may not converge
            if not entry_fit['converged']:
                print("NOTE: The fits with the weights from the velocity of the previous model did not " \
                    "converge in {:d} fits, the fit which fits the distances best is used".format( \
                    entry_fit['n_pass']))

            # Noise and offset of every station
            print("  Station      Noise (m)   Offset (m)")
            for station_id in station_ids:
                print("  {:10s} {:9.1f} {:12.1f}".format(station_id, entry_fit['sigma_station'][station_id], \
                    entry_fit['offsets'][station_id]))
        elif cml_args.atmrescale == 'column':
            print("  Heights mapped to the model's exponential atmosphere by the air column above them " \
                "(--atmrescale column)")
        else:
            print("  Heights mapped to the model's exponential atmosphere by local density (--atmrescale " \
                "density, which biases alpha and beta where the scale height is not 7.16 km)")
        print("  Distance residuals RMS = {:.1f} m".format(np.sqrt(np.mean(length_residuals**2))))

        # Print the entry velocity, the model speed at the first point and the trajectory's velocity, in the
        #   inertial frame of the trajectory, and the entry velocity relative to the air of the full fit
        print("Velocities:")
        v_init_std = fit_errors.get('v_init_std', np.nan) if (fit_errors is not None) else np.nan
        print("  V_e, alpha-beta entry velocity (asymptote above the atmosphere, without gravity)")
        print("                                             = {:.3f}{:s} km/s (fitted to the lag)".format(
            v_init_inertial/1000, " +/- {:.3f}".format(v_init_std/1000) if np.isfinite(v_init_std) \
            else ""))
        if cml_args.lagfit == 'full':
            print("  V_e relative to the air                    = {:.3f} km/s".format(v_init/1000))
        print("  Speed at 180 km, with gravity              = {:.3f} km/s".format(
            alphaBetaEntryVelocityWithGravity(v_init_inertial, ht_first)/1000))
        print("  Model speed at the first point, {:5.1f} km   = {:.3f} km/s".format(ht_first/1000,
            v_model_first/1000))
        print("  Trajectory's initial velocity (solver)     = {:.3f} km/s (the model is {:+.0f} m/s from " \
            "it at the first point)".format(traj.v_init/1000, v_model_first - traj.v_init))
        if cml_args.lagfit == 'full':
            print("  The masses, the dynamic mass and the dynamic pressure below use the velocities " \
                "relative to the air.")
        else:
            print("  The masses below use V_e.")

        print()
        print("Alpha-beta analysis")
        print("-------------------")
        if cml_args.errors and (not beta_at_zero):
            print("Alpha = {:.3f} (+/-{:.1%})".format(alpha, fit_errors['alpha_std_rel']))
            print("Beta  = {:.4g} (+/-{:.1%})".format(beta, fit_errors['beta_std_rel']))
            print("corr(alpha, beta) = {:.3f}".format(fit_errors['corr']))
        else:
            print("Alpha = {:.3f}".format(alpha))
            print("Beta  = {:.4g}".format(beta))

        # Warn if alpha or beta are at a bound of the fit, where the data did not constrain them
        for bound_warning in alphaBetaBoundWarnings(alpha, beta):
            print("WARNING: {:s}".format(bound_warning))

        # Print the profile intervals, which hold also where the Wald errors above do not
        if cml_args.lagfit == 'full':
            print("{:.3g}% profile intervals (threshold scaled by the autocorrelation time of the " \
                "residuals):".format(lag_profile['ci']))
        else:
            print("{:.3g}% profile intervals (lag residuals assumed independent):".format(lag_profile['ci']))
        print("  Alpha: {:s}".format(profileIntervalText(lag_profile, 'alpha')))
        print("  Beta:  {:s}".format(profileIntervalText(lag_profile, 'beta')))

        # Beta with only an upper limit is consistent with zero
        if beta_at_zero and (not lag_profile['beta_upper_at_bound']):
            print("NOTE: Beta is consistent with zero, as the lag gives only an upper limit of the mass loss")

        # Without the Wald uncertainties the intervals are taken from the profile region
        if beta_at_zero and cml_args.errors:
            print("NOTE: The intervals of beta and of the quantities derived from it, e.g. the masses, are " \
                "taken from the {:.3g}% profile region, as the Wald uncertainties are not meaningful when " \
                "beta reaches zero".format(lag_profile['ci']))

        # Fit again without every station, as the intervals do not contain the systematic errors of the
        #   stations, and name the stations without which the fit is outside the intervals
        if (cml_args.lagfit == 'full') and (len(np.unique(station_data)) >= 3):

            print("Fit without every station (* outside the {:.3g}% intervals):".format(lag_profile['ci']))
            for station_id, alpha_jack, beta_jack in stationJackknifeAlphaBetaEntryLag(time_data, \
                entry_lag_data, ht_normed_data, station_data, slant_column.lengthScale, sigma_length, \
                traj.v_init, alpha, beta):

                outside = (alpha_jack < lag_profile['alpha_lower']) \
                    or (alpha_jack > lag_profile['alpha_upper']) or (beta_jack < lag_profile['beta_lower']) \
                    or (beta_jack > lag_profile['beta_upper'])
                print("  {:s} Without {:s}: alpha = {:.3g}, beta = {:.3g}".format("*" if outside else " ", \
                    str(station_id), alpha_jack, beta_jack))

        print("-------------------")
        print("ln(beta)             = {:.2f}".format(np.log(beta)))
        print("ln(alpha*sin(slope)) = {:.2f}".format(np.log(alpha*np.sin(slope))))
        print("-------------------")
        print("Masses with dens = {:d} kg/m^3, Gamma*A = {:.2f}:".format(int(cml_args.dens), cml_args.ga))
        print(" * - note that the initial masses are usually 4-10x underestimated!")

        # Masses of the models within the 95% interval, for each mu (initial and final)
        interval_masses = {}
        for mu in [0, 2/3]:
            interval_masses[mu] = np.column_stack(profileMassesAlphaBetaLag(lag_profile, alpha, beta, \
                v_init, slope, ht_data_rescaled[0], mu=mu, dens=cml_args.dens, shape_coeff=cml_args.ga, \
                gamma=1.0))

        def _printMass(label, mass, mass_err, key, masses, no_lower=False, no_upper=False):
            """ Print a single mass line, annotated with its 1-sigma relative error and Wald CI
                if --errors produced a (finite) estimate for it, plain otherwise, and with the 95% profile
                interval if the masses of the models within it are given, with the sides which are not
                constrained by the data marked.
            """
            if (mass_err is not None) and np.isfinite(mass_err[key + '_std_rel']):
                print("    {:s} = {:.3e} kg (+/-{:.1%})  [{:.3g}% CI: {:.3e} - {:.3e}]".format(
                    label, mass, mass_err[key + '_std_rel'], mass_err['ci'],
                    mass_err[key + '_ci_wald_lower'], mass_err[key + '_ci_wald_upper']))
            else:
                print("    {:s} = {:.3e} kg".format(label, mass))

            if len(masses):
                print("      {:s}".format(_intervalText(masses, "{:.3e}", no_lower=no_lower, \
                    no_upper=no_upper).strip()))

        # The initial mass goes as alpha^-3, so it has no upper limit if alpha has no lower limit and the
        #   other way around
        init_no_lower = lag_profile['alpha_upper_at_bound']
        init_no_upper = lag_profile['alpha_lower_at_bound']

        print("  mu = 0:")
        _printMass("Initial", m_init_mu0, mass_err_mu0, 'm_init', interval_masses[0][:, 0], \
            no_lower=init_no_lower, no_upper=init_no_upper)
        _printMass("Final  ", m_final_mu0, mass_err_mu0, 'm_final', interval_masses[0][:, 1])
        print("  mu = 2/3:")
        _printMass("Initial", m_init_mu23, mass_err_mu23, 'm_init', interval_masses[2/3][:, 0], \
            no_lower=init_no_lower, no_upper=init_no_upper)
        _printMass("Final  ", m_final_mu23, mass_err_mu23, 'm_final', interval_masses[2/3][:, 1])
        if interval_models and (mass_err_mu0 is not None):
            print("  The profile intervals include only the uncertainty of the fit, the Wald CIs also " \
                "--densunc{:s}".format(" and --slopeunc" if cml_args.lagfit == 'simple' else ""))
        elif interval_models:
            print("  The profile intervals include only the uncertainty of the fit")

        print("*********************************************")


        ### Compute the dynamic mass at the end from the alpha-beta fit ###

        # Velocity and deceleration of the alpha-beta model at the last point along the path, which is the
        #   path flown through the air with the full fit
        length_path = length_air if (cml_args.lagfit == 'full') else length_data
        ab_path = AlphaBetaPath(length_path, ht_data_rescaled)
        end_index = np.argmax(length_path)
        vel_dyn, decel_dyn = ab_path.velocityDeceleration(length_path[end_index], alpha, beta, v_init)
        ht_dyn = ht_data[end_index]

        # Compute the dynamic mass, which is not defined if the model does not decelerate at the end
        dyn_mass = dynamicMass(cml_args.dens, traj.rend_lat, traj.rend_lon, ht_dyn, traj.jdt_ref, \
            vel_dyn, decel_dyn, gamma=1.0, shape_factor=cml_args.ga) if decel_dyn > 0 else np.nan

        print()
        print("Alpha-beta dynamic mass at the end:")
        print("-----------------------------------")
        print("Bulk density = {:5d} kg/m^3".format(int(cml_args.dens)))
        print("Height       = {:5.2f} km".format(ht_dyn/1000))
        print("Velocity     = {:5.2f} km/s{:s}".format(vel_dyn/1000, " (relative to the air)" \
            if (cml_args.lagfit == 'full') else ""))
        print("Deceleration = {:5.2f} km/s^2".format(decel_dyn/1000))
        print("Gamma*A      = {:5.2f}".format(cml_args.ga))
        # Dynamic masses of the models within the 95% interval
        # Dynamic masses of the models within the 95% interval, which has no upper limit if one of them does
        #   not decelerate at the end
        interval_dyn_masses = []
        no_decel = False
        for a, b, v0 in interval_models:
            vel_model_end, decel_model_end = ab_path.velocityDeceleration(length_path[end_index], a, b, v0)
            if decel_model_end <= 0:
                no_decel = True
                continue
            interval_dyn_masses.append(dynamicMass(cml_args.dens, traj.rend_lat, traj.rend_lon, ht_dyn, \
                traj.jdt_ref, vel_model_end, decel_model_end, gamma=1.0, shape_factor=cml_args.ga))

        # Print the dynamic masses in the scientific notation if they are large
        dyn_values = np.array(interval_dyn_masses + ([dyn_mass] if np.isfinite(dyn_mass) else []))
        dyn_fmt = "{:.3e}" if (len(dyn_values) and (np.max(np.abs(dyn_values)) >= 1e4)) else "{:.3f}"

        dyn_interval_text = _intervalText(interval_dyn_masses, dyn_fmt)
        if no_decel and interval_dyn_masses:
            dyn_interval_text = "  [95% profile: > " + dyn_fmt.format(np.min(interval_dyn_masses)) \
                + " (no upper limit)]"

        print()
        if np.isfinite(dyn_mass):
            print("Dynamic mass = " + dyn_fmt.format(dyn_mass) + " kg" + dyn_interval_text)
        else:
            print("Dynamic mass = not defined, the model does not decelerate at the end{:s}".format( \
                dyn_interval_text))

        # The deceleration of a fit at a bound is not constrained by the data
        if alphaBetaBoundWarnings(alpha, beta):
            print("NOTE: alpha or beta are at a bound of the fit, so the deceleration and the dynamic mass "
                "are not constrained by the data")

        # A fireball which barely decelerates does not constrain alpha, beta and the masses, and it usually
        #   ablates completely
        velocity_loss = 1.0 - vel_dyn/v_init
        little_deceleration = velocity_loss < MIN_VELOCITY_LOSS
        if little_deceleration:
            print("NOTE: The fireball lost only {:.1%} of its velocity by the end at {:.1f} km, so the " \
                "dynamics do not constrain alpha, beta and the masses. A fireball which does not " \
                "decelerate usually ablates completely, so the final mass is ~0.".format(velocity_loss, \
                ht_dyn/1000))

        print("-----------------------------------")

        ### ###


        # Print the parameters of the smooth model of the lag
        if cml_args.lagsmooth:
            print()
            print("Lag smoothing:")
            print("-------------------")
            print("    - a1    = {:.3f}".format(abs(lag_fit_params[0])))
            print("    - a2    = {:.3f}".format(abs(lag_fit_params[1])))
            print("    - t0    = {:.3f} s".format(abs(lag_fit_params[2])))
            print("    - decel = {:.3f} km/s^2".format(abs(lag_fit_params[3])))



        # Heights of the curves of the model, from below the end to above the beginning
        ht_end = traj.rend_ele - 5000
        if ht_end < 10000:
            ht_end = 10000
        elif (ht_end > 20000) and (ht_end < 35000):
            ht_end = 20000
        ht_arr = np.linspace(ht_end, traj.rbeg_ele + 5000, 200)



        ### Alpha-beta plot ###

        # Models within the 95% region of alpha and beta of the profile, which give the bands of the
        #   velocity and of the lag (only with --errors, which computes the 2D grid)
        region_alpha, region_beta, region_v_init = profileRegionAlphaBetaLag(lag_profile)
        have_band = len(region_alpha) > 0
        band_label = "95% region of alpha and beta"

        # Label of the alpha-beta fit, with the profile intervals of alpha and beta
        fit_label = "$v_0$ = {:.2f} km/s\n$\\alpha$ = {:.3g} ({:s})\n$\\beta$ = {:.3g} ({:s})".format( \
            v_init_inertial/1000, alpha, profileIntervalText(lag_profile, 'alpha', notes=False), beta, \
            profileIntervalText(lag_profile, 'beta', notes=False))

        # Label of the dynamic mass
        if little_deceleration:
            dyn_label = "Dynamic mass not constrained\n(velocity loss {:.1%})".format(velocity_loss)
        elif np.isfinite(dyn_mass):
            dyn_label = "Dynamic mass = {:s}\ndecel = {:.2f} km/s$^2$, $\\Gamma A$ = {:.2f}".format( \
                _formatMass(dyn_mass), decel_dyn/1000, cml_args.ga)
        else:
            dyn_label = "Dynamic mass not defined\n(no deceleration at the end)"

        if cml_args.lagfit == 'full':

            # The full fit has an offset for every station, so the model is drawn for every station from its
            #   own points
            common_lag_line = False

            # Band of the distance residuals of the models within the region at the observed points, relative
            #   to the best model, with dv and the offsets of the stations fitted for every model
            if have_band:
                region_models = list(zip(region_alpha, region_beta, region_v_init))
                station_index_fit = np.unique(station_data, return_inverse=True)[1]
                n_stations_fit = int(np.max(station_index_fit)) + 1
                lag_fit_best = entry_lag_data - lag_residuals
                length_band = []
                for a, b, _ in region_models:
                    lag_model_region, _, _ = _entryLagModel(a, b, ht_normed_data, slant_column.lengthScale)
                    _, _, residuals_region = _entryLagLinear(entry_lag_data - lag_model_region, time_data, \
                        entry_fit['weights'], station_index_fit, n_stations_fit)
                    length_band.append((entry_lag_data - residuals_region - lag_fit_best) \
                        *entry_fit['v_model'])

                # Sort the band by the height of the points
                ht_order = np.argsort(ht_data)
                ht_lag_dense = ht_data[ht_order]
                length_band_lo = np.min(length_band, axis=0)[ht_order]
                length_band_hi = np.max(length_band, axis=0)[ht_order]

        else:

            # Lag of the models on a dense grid of times. The lag is the distance minus a line in time,
            #   which is the same for all stations, so the line is fitted to the differences of the distances
            #   and the lag, and the lag is drawn against the height of the best model at every time, which
            #   keeps the curve smooth also when the points of the stations are interleaved in height
            lag_line = np.polyfit(time_data, length_data - lag_data, 1)

            # Older solutions have a separate line for every station, then the model is drawn for every
            #   station from its own points
            lag_line_rms = np.sqrt(np.mean((length_data - lag_data - np.polyval(lag_line, time_data))**2))
            common_lag_line = lag_line_rms < 1.0
            time_dense = np.linspace(np.min(time_data), np.max(time_data), 400)
            _, t0_best = ab_path.lengths(time_data, alpha, beta, v_init, length_data=length_data)
            length_dense, _ = ab_path.lengths(time_dense, alpha, beta, v_init, t0=t0_best)
            lag_dense = length_dense - np.polyval(lag_line, time_dense)
            length_order = np.argsort(length_data)
            ht_along = np.minimum.accumulate(ht_data[length_order])
            ht_lag_dense = np.interp(length_dense, length_data[length_order], ht_along)

            # Band of the distance of the models within the region (each model is shifted in time to fit the
            #   observed distances, as the best one)
            if have_band:
                region_models = list(zip(region_alpha, region_beta, region_v_init))
                length_band = np.array([ab_path.lengths(time_dense, a, b, v0, \
                    t0=ab_path.lengths(time_data, a, b, v0, length_data=length_data)[1])[0] \
                    for a, b, v0 in region_models]) - length_dense
                length_band_lo = np.min(length_band, axis=0)
                length_band_hi = np.max(length_band, axis=0)

        # Limits of the velocity axes which are robust to single outliers of the point-to-point velocities
        vel_axis_lo = min(0.0, max(np.percentile(vel_data, 1), -0.5*v_init_inertial))
        vel_axis_hi = max(1.2*v_init_inertial, min(np.percentile(vel_data, 99), 3*v_init_inertial))

        def _plotFit(ax_vel, ax_lag, ax_lag_res, print_results=True):
            """ Plot the alpha-beta fit: the velocities against the observed heights, with the model from
                above the first point down to below the last one, the lag and the distance residuals, with the
                points of every station in its color.

            Arguments:
                ax_vel: [Axes] Velocities against the observed heights.
                ax_lag: [Axes] Lag.
                ax_lag_res: [Axes] Distance residuals.

            Keyword arguments:
                print_results: [bool] Print the dynamic mass of the lag smoothing. True by default.
            """

            ### Plot the velocities against the observed heights ###

            for station_id in station_ids:
                station_mask = station_data == station_id
                ax_vel.scatter(vel_data[station_mask]/1000, ht_data[station_mask]/1000, s=5, \
                    color=station_colors[station_id], alpha=station_alpha)

            ax_vel.set_title("Velocity", fontsize=10)

            # Plot the alpha-beta velocity from above the first point down to below the last one, at the
            #   heights rescaled to the exponential atmosphere at the mean position, as the dynamic pressure,
            #   with the band of the models within the region, in the inertial frame of the observations
            if have_band:
                ht_arr_rescaled_mean = rescaleRealHeights(ht_arr)
                vel_band = np.array([alphaBetaVelocityInterp(ht_arr_rescaled_mean, a, b, v0) \
                    for a, b, v0 in region_models]) + air_offset
                ax_vel.fill_betweenx(ht_arr/1000, np.min(vel_band, axis=0)/1000, \
                    np.max(vel_band, axis=0)/1000, color='k', alpha=0.15, lw=0, zorder=1, label=band_label)

            ax_vel.plot((vel_arr_real + air_offset)/1000, ht_arr/1000, color='k', zorder=2, label=fit_label)

            # Plot the point where the dynamic mass is estimated
            if np.isfinite(dyn_mass):
                ax_vel.scatter([(vel_dyn + air_offset)/1000], [ht_dyn/1000], color='k', zorder=3, \
                    label=dyn_label)
            else:
                ax_vel.plot([], [], ' ', label=dyn_label)

            # Plot the smoothed velocity and its dynamic mass
            if cml_args.lagsmooth:

                ax_vel.scatter(vel_data_smooth/1000, ht_data/1000, color='r', s=1, \
                    label="Lag-based velocity smoothing")

                # If the exponental to linear transition point was used by the fit, plot it
                t0 = lag_fit_params[2]
                decel = lag_fit_params[3]
                if t0 < np.max(time_data):

                    # Find the height closest to t0
                    v_t0 = expLinearVelocity(t0, traj.v_init, *lag_fit_params)
                    t0_index = np.argmin(np.abs(vel_data_smooth - v_t0))
                    h_t0 = ht_data[t0_index]

                    # Plot the t0 point
                    ax_vel.scatter([v_t0/1000], [h_t0/1000], \
                        label='t0, decel = {:.2f} km/s$^2$'.format(abs(decel)), color='r')


                    ### Compute the dynamic mass at the end from the lag smoothing ###

                    # Compute the values at the point that is 1/4 before the end and t0 (sorted by increasing
                    #   height)
                    midpoint_index = int(round(t0_index*1/4))
                    ht_dyn_smooth = ht_data[midpoint_index]
                    t_dyn = time_data[midpoint_index]
                    v_dyn_smooth = expLinearVelocity(t_dyn, traj.v_init, *lag_fit_params)

                    # Compute the dynamic mass
                    dyn_mass_smooth = dynamicMass(cml_args.dens, traj.rend_lat, traj.rend_lon, \
                        ht_dyn_smooth, traj.jdt_ref, v_dyn_smooth, 1000*abs(decel), gamma=1.0, \
                        shape_factor=cml_args.ga)

                    # Plot the point where the dynamic mass is estimated
                    ax_vel.scatter([v_dyn_smooth/1000], [ht_dyn_smooth/1000], color='r', marker='s', \
                        label='Lag smoothing dynamic mass = {:.3f} kg'.format(dyn_mass_smooth))

                    if print_results:
                        print()
                        print("Lag smoothing dynamic mass:")
                        print("---------------------------")
                        print("Bulk density = {:5d} kg/m^3".format(int(cml_args.dens)))
                        print("Height       = {:5.2f} km".format(ht_dyn_smooth/1000))
                        print("Velocity     = {:5.2f} km/s".format(v_dyn_smooth/1000))
                        print("Deceleration = {:5.2f} km/s^2".format(abs(decel)))
                        print("Gamma*A      = {:5.2f}".format(cml_args.ga))
                        print()
                        print("Dynamic mass = {:5.3f} kg".format(dyn_mass_smooth))
                        print("---------------------------")

                    ### ###

            ax_vel.set_xlim(vel_axis_lo/1000, vel_axis_hi/1000)
            ax_vel.set_xlabel("Velocity (km/s)")
            ax_vel.set_ylabel("Height (km)")

            ax_vel.legend(loc='upper left', fontsize=7)


            ### Plot the lag of every station and the alpha-beta model of it ###

            for station_id in station_ids:
                station_mask = station_data == station_id
                ax_lag.scatter(lag_data[station_mask]/1000, ht_data[station_mask]/1000, s=5, \
                    color=station_colors[station_id], alpha=station_alpha)

            if common_lag_line:
                ax_lag.plot(lag_dense/1000, ht_lag_dense/1000, color='k', label="Alpha-beta fit")
            else:
                for i, station_id in enumerate(station_ids):
                    station_mask = station_data == station_id
                    station_order = np.argsort(time_data[station_mask])
                    ax_lag.plot((lag_data - lag_residuals)[station_mask][station_order]/1000, \
                        ht_data[station_mask][station_order]/1000, color='k', \
                        label=("Alpha-beta fit" if i == 0 else None))
            if have_band and common_lag_line:
                ax_lag.fill_betweenx(ht_lag_dense/1000, (lag_dense + length_band_lo)/1000, \
                    (lag_dense + length_band_hi)/1000, color='k', alpha=0.15, lw=0, zorder=1, \
                    label=band_label)

            # Plot the smooth model of the lag
            if cml_args.lagsmooth:
                ax_lag.plot(expLinearLag(time_data, *lag_fit_params)/1000, ht_data/1000, color='r', \
                    label="Lag-based velocity smoothing")

            ax_lag.set_title("Lag", fontsize=10)
            ax_lag.set_xlabel("Lag (km)")
            ax_lag.legend(loc='lower left', fontsize=8)


            ### Plot the distance residuals of the alpha-beta fit of every station ###

            for station_id in station_ids:
                station_mask = station_data == station_id
                ax_lag_res.scatter(length_residuals[station_mask]/1000, ht_data[station_mask]/1000, s=5, \
                    color=station_colors[station_id], alpha=station_alpha, \
                    label="{:s}, RMS {:.0f} m".format(station_id, \
                    np.sqrt(np.mean(length_residuals[station_mask]**2))))

            ax_lag_res.axvline(0, color='0.5', lw=0.8, zorder=0)
            if have_band:
                ax_lag_res.fill_betweenx(ht_lag_dense/1000, length_band_lo/1000, length_band_hi/1000, \
                    color='k', alpha=0.15, lw=0, zorder=1, label=band_label)
            ax_lag_res.set_title("Alpha-beta fit residuals", fontsize=10)

            # Plot the residuals of the smooth model of the lag
            if cml_args.lagsmooth:
                ax_lag_res.scatter(lag_data/1000 - expLinearLag(time_data, *lag_fit_params)/1000, \
                    ht_data/1000, s=5, color='r', label="Lag-based velocity smoothing")

            ax_lag_res.set_xlabel("Distance residuals (km)")
            ax_lag_res.legend(loc='upper left', fontsize=7 if len(station_ids) <= 8 else 5, \
                ncol=(1 if len(station_ids) <= 8 else 2))





        ### Plot dynamic pressure ###

        # Take mean meteor lat/lon as reference for the atmosphere model
        lat_mean = np.mean([traj.rbeg_lat, traj.rend_lat])
        lon_mean = meanAngle([traj.rbeg_lon, traj.rend_lon])

        def rescaleRealHeights(ht_real):
            """ Map real heights to the heights of the model, by the slant column of the full fit, or as the
                simple fit maps them at the mean position of the meteor.
            """

            if cml_args.lagfit == 'full':
                return HT_NORM_CONST*slant_column.htNormedAtHeights(ht_real)

            n = len(ht_real)

            return rescaleHeightToExponentialAtmosphere(np.full(n, lat_mean), np.full(n, lon_mean), ht_real, \
                traj.jdt_ref, method=cml_args.atmrescale)


        def modelVelocityAtRealHeights(ht_real):
            """ Compute the alpha-beta velocity at real heights, rescaled to the model atmosphere. """

            return alphaBetaVelocity(rescaleRealHeights(ht_real), alpha, beta, v_init)

        # Compute the dynamic pressure using alpha and beta, with the real density at real heights, and the
        #   model velocity at those heights rescaled to its atmosphere
        vel_arr_real = modelVelocityAtRealHeights(ht_arr)
        dyn_pressure = dynamicPressure(lat_mean, lon_mean, ht_arr, traj.jdt_ref, vel_arr_real)
        def findPeakDynPressure(dyn_pressure, ht_arr):
            """ Find the peak dynamic pressure.

            Arguments:
                dyn_pressure: [ndarray] Dynamic pressure (Pa).
                ht_arr: [ndarray] Heights (m).

            Return:
                (peak_dyn_pressure_index, peak_dyn_pressure, peak_dyn_pressure_ht):
                    - peak_dyn_pressure_index: [int] Index of the peak.
                    - peak_dyn_pressure: [float] Peak dynamic pressure (MPa).
                    - peak_dyn_pressure_ht: [float] Height of the peak (km).
            """

            peak_dyn_pressure_index = np.argmax(dyn_pressure)
            peak_dyn_pressure = dyn_pressure[peak_dyn_pressure_index]/1e6
            peak_dyn_pressure_ht = ht_arr[peak_dyn_pressure_index]/1000

            return peak_dyn_pressure_index, peak_dyn_pressure, peak_dyn_pressure_ht


        # Find the alpha-beta dyn pressure peak
        peak_dyn_pressure_index, peak_dyn_pressure, peak_dyn_pressure_ht = findPeakDynPressure( \
            dyn_pressure, ht_arr)

        # Print height, model velocity and the dynamic pressure of the alpha-beta model
        print()
        print("AlphaBeta model:")
        if cml_args.lagfit == 'full':
            print("Velocities relative to the air")
        print("Height (km)  Vel (km/s)  DynPress (MPa)")
        for i in range(len(ht_arr)):

            # Compute the index from reverse
            i_rev = len(ht_arr) - i - 1
            print("{:11.2f}  {:10.4f}  {:14.5f}".format(ht_arr[i_rev]/1000, vel_arr_real[i_rev]/1000, \
                dyn_pressure[i_rev]/1e6))

        # Compute the dynamic pressure with the lag smoothing at the real heights of the points, and find its
        #   peak
        if cml_args.lagsmooth:

            dyn_pressure_lag = dynamicPressure(lat_mean, lon_mean, ht_data, traj.jdt_ref, vel_data_smooth)

            peak_dyn_pressure_index_lag, peak_dyn_pressure_lag, peak_dyn_pressure_ht_lag = \
                findPeakDynPressure(dyn_pressure_lag, ht_data)

            print()
            print("Lag smoothing:")
            print("Height (km)  Vel (km/s)  DynPress (MPa)")
            for i in range(len(ht_data)):

                # Compute the index from reverse
                i_rev = len(ht_data) - i - 1
                print("{:11.2f}  {:10.4f}  {:14.5f}".format(ht_data[i_rev]/1000, vel_data_smooth[i_rev]/1000, \
                    dyn_pressure_lag[i_rev]/1e6))

        # Peak dynamic pressures and their heights of the models within the 95% interval, with the density
        #   computed once as the dynamic pressure is proportional to the squared velocity
        interval_peaks = []
        interval_pressures = []
        if interval_models:
            atm_dens = getAtmDensity_vect(lat_mean, lon_mean, ht_arr, traj.jdt_ref)
            ht_arr_rescaled = rescaleRealHeights(ht_arr)
            for a, b, v0 in interval_models:
                dyn_pressure_model = atm_dens*alphaBetaVelocityInterp(ht_arr_rescaled, a, b, v0)**2
                interval_pressures.append(dyn_pressure_model)
                interval_peaks.append([np.max(dyn_pressure_model)/1e6, \
                    ht_arr[np.argmax(dyn_pressure_model)]/1000])
        interval_peaks = np.array(interval_peaks).reshape(-1, 2)

        print()
        print("Peak dynamic pressure:")
        print("----------------------")
        print("AlphaBeta model = {:6.3f} MPa at {:6.2f} km (v = {:5.2f} km/s)".format(
            peak_dyn_pressure, peak_dyn_pressure_ht, vel_arr_real[peak_dyn_pressure_index]/1000))
        if len(interval_peaks):
            print("  95% profile: {:.3f} - {:.3f} MPa, at {:.2f} - {:.2f} km".format( \
                np.min(interval_peaks[:, 0]), np.max(interval_peaks[:, 0]), np.min(interval_peaks[:, 1]), \
                np.max(interval_peaks[:, 1])))
        if cml_args.lagsmooth:
            print("Lag smoothing   = {:6.3f} MPa at {:6.2f} km (v = {:5.2f} km/s)".format(
                peak_dyn_pressure_lag, peak_dyn_pressure_ht_lag,
                vel_data_smooth[peak_dyn_pressure_index_lag]/1000))
        print("----------------------")


        def _plotDynPressure(ax):
            """ Plot the dynamic pressure of the alpha-beta model along the path, with the band of the models
                within the 95% interval and the peak.

            Arguments:
                ax: [Axes] Axes to draw into.
            """

            if interval_pressures:
                ax.fill_betweenx(ht_arr/1000, np.min(interval_pressures, axis=0)/1e6, \
                    np.max(interval_pressures, axis=0)/1e6, color='k', alpha=0.15, lw=0, label="95% interval")
            ax.plot(dyn_pressure/1e6, ht_arr/1000, color='k', label='AlphaBeta')

            # Mark alpha-beta dyn pressure peak on the graph
            peak_label = "Peak P = {:.2f} MPa\nHt = {:.2f} km".format(peak_dyn_pressure, peak_dyn_pressure_ht)
            if len(interval_peaks):
                peak_label += "\n95%: {:.2f} - {:.2f} MPa".format(np.min(interval_peaks[:, 0]), \
                    np.max(interval_peaks[:, 0]))
            ax.scatter(peak_dyn_pressure, peak_dyn_pressure_ht, c='k', label=peak_label)

            # Plot the dynamic pressure of the lag smoothing and mark its peak
            if cml_args.lagsmooth:
                ax.plot(dyn_pressure_lag/1e6, ht_data/1000, color='r', label='Lag smoothing')
                ax.scatter(peak_dyn_pressure_lag, peak_dyn_pressure_ht_lag, c='r', \
                    label="Peak P = {:.2f} MPa\nHt = {:.2f} km".format(peak_dyn_pressure_lag, \
                    peak_dyn_pressure_ht_lag))

            ax.legend(fontsize=8)
            ax.set_xlabel("Dynamic pressure (MPa)")


        # Plot dyn pressure, with the band of the models within the 95% interval
        fig_dyn, ax_dyn = plt.subplots()
        _plotDynPressure(ax_dyn)
        ax_dyn.set_ylabel("Height (km)")
        ax_dyn.set_title("{:s}, alpha-beta dynamic pressure".format(traj.file_name), fontsize=10)

        _finishPlot(fig_dyn, "alpha_beta_dyn_pressure")



        ### Summary plot of the fit, the pressure, the uncertainties, the mass diagram and the masses ###

        fig_sum = plt.figure(figsize=(22, 13))
        grid_sum = fig_sum.add_gridspec(2, 1, height_ratios=[1, 1.1], hspace=0.25, left=0.05, right=0.89, \
            top=0.93, bottom=0.06)

        # Plot the fit and the dynamic pressure in the top row
        grid_top = grid_sum[0].subgridspec(1, 4, wspace=0)
        ax_sum_vel = fig_sum.add_subplot(grid_top[0])
        sum_axes = [ax_sum_vel] + [fig_sum.add_subplot(grid_top[i], sharey=ax_sum_vel) for i in range(1, 4)]
        _plotFit(*sum_axes[:3], print_results=False)
        _plotDynPressure(sum_axes[3])
        sum_axes[3].set_title("Dynamic pressure", fontsize=10)
        for ax in sum_axes[1:]:
            ax.tick_params(labelleft=False)

        grid_bottom = grid_sum[1].subgridspec(1, 3, width_ratios=[1.3, 1, 1], wspace=0.3)

        # Plot the uncertainties of alpha and beta
        if cml_args.errors:
            plotAlphaBetaLagProfile(lag_profile, alpha, beta, errors=(None if beta_at_zero else fit_errors), \
                fig=fig_sum, subplot_spec=grid_bottom[0])
        else:
            ax_note = fig_sum.add_subplot(grid_bottom[0])
            ax_note.axis('off')
            ax_note.text(0.5, 0.5, "Run with --errors for the uncertainties of alpha and beta", \
                ha='center', va='center', transform=ax_note.transAxes)

        # Plot the survival diagram in the space of the initial mass and beta
        ax_mass_diagram = fig_sum.add_subplot(grid_bottom[1])
        plotAlphaBetaMassDiagram(alpha, beta, slope, profile=(lag_profile if cml_args.errors else None), \
            dens=cml_args.dens, shape_coeff=cml_args.ga, axes=ax_mass_diagram)
        if little_deceleration:
            ax_mass_diagram.text(0.5, 0.5, "Not constrained by the dynamics\n(velocity loss {:.1%})".format( \
                velocity_loss), transform=ax_mass_diagram.transAxes, ha='center', va='center', fontsize=12, \
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='0.5'))

        # Plot the masses with their 95% intervals
        def _massEntry(label, mass, interval_values, color, no_upper_limit=False):
            """ One mass of the plot of the masses, with its interval from the values of the models within
                the 95% interval.
            """

            if not len(interval_values):
                return (label, mass, None, None, color)

            upper = np.inf if no_upper_limit else np.max(interval_values)

            return (label, mass, np.min(interval_values), upper, color)

        # Without a significant deceleration the final masses are ~0
        if little_deceleration:
            masses = [("Final, $\\mu$ = 0", 0.0, None, None, 'tab:blue'), \
                ("Final, $\\mu$ = 2/3", 0.0, None, None, 'tab:orange'), \
                ("Dynamic, end", 0.0, None, None, 'k')]
            masses_note = "Velocity loss of only {:.1%} by the end:\nthe dynamics do not constrain the " \
                "masses, and the\nfireball most likely ablated completely".format(velocity_loss)
        else:
            masses = [
                _massEntry("Final, $\\mu$ = 0", m_final_mu0, interval_masses[0][:, 1], 'tab:blue'),
                _massEntry("Final, $\\mu$ = 2/3", m_final_mu23, interval_masses[2/3][:, 1], 'tab:orange'),
                _massEntry("Dynamic, end", dyn_mass, interval_dyn_masses, 'k', no_upper_limit=no_decel),
                ]
            masses_note = None
        ax_masses = fig_sum.add_subplot(grid_bottom[2])
        plotAlphaBetaMasses(masses, axes=ax_masses, title="Masses with the 95% profile intervals, " \
            "dens = {:d} kg/m$^3$, $\\Gamma A$ = {:.2f}".format(int(cml_args.dens), cml_args.ga), \
            note=masses_note)

        if cml_args.lagfit == 'full':
            fig_sum.suptitle("{:s}, alpha-beta full fit of the entry lag in slant-column " \
                "coordinates".format(traj.file_name))
        else:
            fig_sum.suptitle("{:s}, alpha-beta fit to the lag".format(traj.file_name))

        _finishPlot(fig_sum, "alpha_beta_summary")





        ### ###





        ###