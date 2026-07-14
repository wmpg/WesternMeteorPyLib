""" Defines functions for the alpha-beta fireball characterization by Gritsevich 2012. 
Adapted from: https://github.com/desertfireballnetwork/alpha_beta_modules
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
    """ Compute the Q4 minimisation residual (sum of absolute residuals) given in Gritsevich 2007 
    """

    return np.sum(
        np.abs(
            2*alpha*np.exp(-ht_normed)
            -
            (
                scipy.special.expi(beta)
                -
                scipy.special.expi(beta*v_normed**2)
            )*np.exp(-beta)
        )
    )


def _betaFromMasses(mu, mass_initial, mass_final, v_final, v_init):
    """ Analytically compute beta from the initial/final mass ratio and the observed velocity change,
        for a given mass change coefficient mu.
    """

    return (
        (1.0 - mu)
        * np.log(mass_initial/mass_final)
        /
        (
            1.0
            - (v_final/v_init)**2
        )
    )


def minimizeAlphaBeta(v_normed, ht_normed):
    """ initiates and calls the Q4 minimisation given in Gritsevich 2007 -
    """

    def _alphaBetaMinimization(x, v_normed, ht_normed):
        """minimises using Q4 minimisation given in Gritsevich 2007
        """ 

        alpha, beta = x

        return _alphaBetaResidual(alpha, beta, v_normed, ht_normed)

    # params = np.vstack((v_normed, ht_normed))

    # Compute initial alpha-beta guess
    b0 = 1.0
    a0 = np.exp(np.min(ht_normed))/(2.0*b0)
    x0 = [a0, b0]

    # Set alpha-beta limits
    xmin = [    0.001,  0.00001]
    xmax = [10000.0,   50.0]
    bnds = ((xmin[0], xmax[0]), (xmin[1], xmax[1]))

    # If the initial guess is outside the bounds, set a middle value
    for i, initial_param in enumerate(x0):
        if (initial_param < xmin[i]) or (initial_param > xmax[i]):
            x0[i] = (xmin[i] + xmax[i])/2

    print("Initial guess:", x0)

    # Compute best-fit alpha-beta values
    res = scipy.optimize.minimize(_alphaBetaMinimization, x0, args=(v_normed, ht_normed), bounds=bnds, \
        method='Nelder-Mead')

    return res.x


def fitAlphaBeta(v_data, ht_data, v_init=None):
    """ Fit the alpha and beta parameters to the given velocity and height data. 
    
    Arguments:
        v_data: [ndarray] Velocity data (m/s).
        ht_data: [ndarray] Height data (m).

    Keyword arguments:
        v_init: [float] Initial velocity (m/s). If None, it will be determined from the first 20% of point
            (or a minimum of 10 points).

    Return:
        (v_init, alpha, beta):
            - v_init: [float] Input or derived initial velocity (m/s).
            - alpha: [float] Ballistic coefficient.
            - beta: [float] Mass loss.
    """


    # Compute the initial velocity, if it wasn't given already
    if v_init is None:

        max_index = int(0.2*len(v_data))
        if max_index < 10:
            max_index = 10

        v_init = np.median(v_data[:max_index])



    # Normalize the velocity
    v_normed = v_data/v_init

    # Normalize the height
    ht_normed = ht_data/HT_NORM_CONST


    # Fit alpha and beta
    alpha, beta = minimizeAlphaBeta(v_normed, ht_normed)


    return v_init, alpha, beta


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

    # Compute the normalized height
    ht_normed = np.log(alpha) + beta - np.log((scipy.special.expi(beta) - scipy.special.expi(beta*vel_normed**2))/2)

    # Compute the height in m
    ht_data = ht_normed*HT_NORM_CONST

    return ht_data


def alphaBetaVelocity(ht_data, alpha, beta, v_init):
    """ Compute the velocity given the height and alpha, beta parameters. Unfortunately there is no 
        analytical inverse to the exponential integral, so the solution is found numerically.

    Arguments:
        ht_data: [ndarray or float] Height data (m).
        alpha: [float] Ballistic coefficient.
        beta: [float] Mass loss.
        v_init: [float] Input or derived initial velocity (m/s).

    Return:
        vel_data: [ndarray or float] Velocity data (m/s).
    """

    def _diff(v, alpha, beta, ht_target):
        """ Function to minimize the height. """
        
        # Compute the height using a guess velocity
        ht_guess = np.log(alpha) + beta - np.log((scipy.special.expi(beta) - scipy.special.expi(beta*v**2))/2)
        
        return (ht_guess - ht_target)**2

    # Allow both scalar and array height inputs
    scalar_input = np.isscalar(ht_data)
    if scalar_input:
        ht_data = np.array([ht_data])

    # Normalize the height
    ht_normed = ht_data/HT_NORM_CONST

    # Numerically compute the normalized velocity
    vel_normed = []
    
    v_first_guess = 0.5
    bounds = [(0.0000001, 0.9999999)]
    for ht_n in ht_normed:

        # Minimize the forward function to find the velocity at the given height
        res = scipy.optimize.minimize(_diff, v_first_guess, args=(alpha, beta, ht_n), bounds=bounds)
        vel_normed.append(res.x[0])

    vel_normed = np.array(vel_normed)

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
    """Fit alpha-beta model parameters under initial, final, or both mass constraints,
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
        shape_coeff: [float]
        gamma: [float]
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
            - m_initial_mu0: [float] Initial mass (input or reconstructed from the fitted parameters) assuming μ = 0.
            - m_final_mu0: [float] Final mass (input or reconstructed from the fitted parameters) assuming μ = 0.
            - density_mu23: [float] Best-fit bulk density (kg/m^3) assuming μ = 2/3.
            - alpha_mu23: [float] Derived ballistic coefficient assuming μ = 2/3.
            - beta_mu23: [float] Derived mass loss parameter assuming μ = 2/3.
            - m_initial_mu23: [float] Initial mass (input or reconstructed from the fitted parameters) assuming μ = 2/3.
            - m_final_mu23: [float] Final mass (input or reconstructed from the fitted parameters) assuming μ = 2/3.
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
        mass_initial = mass
        mass_final = None

        if mass_initial <= 0:
            raise ValueError("mass_initial must be positive.")

    elif mass_constraint in ["final", "f"]:
        mass_constraint = "final"
        mass_initial = None
        mass_final = mass

        if mass_final <= 0:
            raise ValueError("mass_final must be positive.")

    elif mass_constraint in ["both", "b"]:
        mass_constraint = "both"

        if not isinstance(mass, (tuple, list)) or len(mass) != 2:
            raise ValueError(
                "For mass_constraint='both', mass must be "
                "(mass_initial, mass_final)"
            )
        
        mass_initial, mass_final = mass
    
        if mass_initial <= 0:
            raise ValueError("mass_initial must be positive.")

        if mass_final <= 0:
            raise ValueError("mass_final must be positive.")

        if mass_final >= mass_initial:
            raise ValueError(
                "mass_final must be smaller than mass_initial."
            )

    else:
        raise ValueError(
            "mass_constraint must be 'initial' ('i'), 'final' ('f'), or 'both' ('b')"
        )

    if len(v_data) != len(ht_data):
        raise ValueError("v_data and ht_data must have the same length.")
    if np.sin(slope) <= 0:
        raise ValueError("slope must satisfy sin(slope) > 0.")
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

    if v_init is None:
        # Take the top-20% highest-altitude points regardless of input ordering
        order_desc = np.argsort(-ht_data)
        v_data_desc = v_data[order_desc]

        max_index = int(0.2 * len(v_data_desc))

        if max_index < 10:
            max_index = 10

        v_init = np.median(v_data_desc[:max_index])

    if v_final is None:
        _, alpha, beta = fitAlphaBeta(v_data, ht_data, v_init=v_init)

        vel_model = alphaBetaVelocity(
            ht_data,
            alpha,
            beta,
            v_init
        )

        # Final velocity
        v_final = np.min(vel_model)
    else:
        if v_final >= v_init:
            raise ValueError(
                "v_final must be smaller than v_init."
            )
        

    # Normalize velocity
    v_normed = v_data/v_init

    # Normalize height
    ht_normed = ht_data/HT_NORM_CONST

    def _alphaFromDensityAndMass(mass_init, dens):

        return (
            gamma*shape_coeff*rho_atm_0*HT_NORM_CONST
            /
            (
                dens**(2/3)
                * np.sin(slope)
                * mass_init**(1/3)
            )
        )

    def _massesAt(alpha, beta, mu, dens):
        """ alphaBetaMasses() bound to this fit's slope/shape_coeff/gamma/v_init/v_final,
            so each μ-branch below just supplies the (alpha, beta, mu, dens) it solved for.
            v_final is read at call time, so this stays correct even where v_final is
            only finalized right before the first call (mass_constraint="initial").
        """

        return alphaBetaMasses(
            alpha,
            beta,
            slope,
            mu=mu,
            dens=dens,
            shape_coeff=shape_coeff,
            gamma=gamma,
            vel_init=v_init,
            vel_end=v_final
        )

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

        dens, rest = _getDensity(x)
        beta = rest[0]

        alpha = _alphaFromDensityAndMass(mass_initial, dens)

        return _alphaBetaResidual(alpha, beta, v_normed, ht_normed)

    def _reconstructInitialMassAndAlpha(dens, beta, mu):

        m_init_fit = (
            mass_final
            * np.exp(
                beta/(1.0 - mu)
                * (
                    1.0
                    - (v_final/v_init)**2
                )
            )
        )

        alpha = _alphaFromDensityAndMass(m_init_fit, dens)

        return m_init_fit, alpha

    def _densityBetaMinimizationFinalMass(x, v_normed, ht_normed, mu):

        dens, rest = _getDensity(x)
        beta = rest[0]

        _, alpha = _reconstructInitialMassAndAlpha(
            dens,
            beta,
            mu
        )

        return _alphaBetaResidual(alpha, beta, v_normed, ht_normed)

    def _densityBetaMuMinimizationFinalMass(x, v_normed, ht_normed):
        """ Same as _densityBetaMinimizationFinalMass, but with μ as a free parameter
            instead of fixed, so the residual-minimizing μ can be found jointly with
            density and beta.
        """

        dens, rest = _getDensity(x)
        beta, mu = rest

        _, alpha = _reconstructInitialMassAndAlpha(
            dens,
            beta,
            mu
        )

        return _alphaBetaResidual(alpha, beta, v_normed, ht_normed)

    def _solveBothMasses(dens, mu):
        """ Analytic beta from the mass ratio and μ, and alpha from density and the
            fixed initial mass, for the mass_constraint="both" case. Shared by the
            optimization objectives and the post-fit derivation of alpha/beta at
            μ=0, μ=2/3 and μ=mu_best so both stay in sync if the relations change.
        """

        beta = _betaFromMasses(mu, mass_initial, mass_final, v_final, v_init)

        alpha = _alphaFromDensityAndMass(mass_initial, dens)

        return alpha, beta

    def _densityMinimizationBothMasses(x, v_normed, ht_normed, mu):

        dens = x[0]

        alpha, beta = _solveBothMasses(dens, mu)

        return _alphaBetaResidual(alpha, beta, v_normed, ht_normed)

    def _densityMuMinimizationBothMasses(x, v_normed, ht_normed):
        """ Same as _densityMinimizationBothMasses, but with μ as a free parameter
            instead of fixed, so the residual-minimizing μ can be found jointly with
            density (beta is still derived analytically from μ and the two masses).
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

    # (density, beta) if density is fitted, or just (beta,) if density is set.
    # Used for mass_constraint="initial" and the μ=0/μ=2/3 fits of mass_constraint="final",
    # which all share this same 2-parameter (or 1-parameter) shape.
    if fit_density:
        x0 = [dens0, beta0]
        bnds = ((xmin[0], xmax[0]), (xmin[1], xmax[1]))
    else:
        x0 = [beta0]
        bnds = ((xmin[1], xmax[1]),)

    if mass_constraint == "initial":
        res = scipy.optimize.minimize(
            _densityBetaMinimizationInitialMass,
            x0,
            args=(v_normed, ht_normed),
            bounds=bnds,
            method='Powell'
        )

        if not res.success:
            print(
                f"WARNING: Optimizer failed: {res.message}"
            )

        density, rest = _getDensity(res.x)
        beta = rest[0]

        density_mu0, density_mu23 = density, density
        beta_mu0, beta_mu23 = beta, beta

        alpha = _alphaFromDensityAndMass(mass_initial, density)

        alpha_mu0, alpha_mu23 = alpha, alpha

        vel_model = alphaBetaVelocity(
            ht_data,
            alpha,
            beta,
            v_init
        )

        vel_end = np.min(vel_model)
        
        v_final = vel_end   # overwrite the unconstrained preliminary v_final for consistency


        m_initial_mu0, m_final_mu0 = _massesAt(alpha, beta, 0, density)

        m_initial_mu23, m_final_mu23 = _massesAt(alpha, beta, 2/3, density)

        # μ has zero effect on the fit residual under an initial-mass constraint
        # (see docstring notes), so no data-driven best-fit μ exists here.
        mu_best = None
        density_mu_best = None
        alpha_mu_best = None
        beta_mu_best = None
        m_initial_mu_best = None
        m_final_mu_best = None


    elif mass_constraint == "final":
        res_mu0 = scipy.optimize.minimize(
            _densityBetaMinimizationFinalMass,
            x0,
            args=(v_normed, ht_normed, 0),
            bounds=bnds,
            method='Powell'
        )

        if not res_mu0.success:
            print(
                f"WARNING: Optimizer failed for μ=0: {res_mu0.message}"
            )

        density_mu0, rest = _getDensity(res_mu0.x)
        beta_mu0 = rest[0]

        m_initial_mu0, alpha_mu0 = \
            _reconstructInitialMassAndAlpha(
                density_mu0,
                beta_mu0,
                0
        )

        m_initial_mu0, m_final_mu0 = _massesAt(alpha_mu0, beta_mu0, 0, density_mu0)

        res_mu23 = scipy.optimize.minimize(
            _densityBetaMinimizationFinalMass,
            x0,
            args=(v_normed, ht_normed, 2/3),
            bounds=bnds,
            method='Powell'
        )

        if not res_mu23.success:
            print(
                f"WARNING: Optimizer failed for μ=2/3: {res_mu23.message}"
            )

        density_mu23, rest = _getDensity(res_mu23.x)
        beta_mu23 = rest[0]

        m_initial_mu23, alpha_mu23 = \
            _reconstructInitialMassAndAlpha(
                density_mu23,
                beta_mu23,
                2/3
        )

        m_initial_mu23, m_final_mu23 = _massesAt(alpha_mu23, beta_mu23, 2/3, density_mu23)

        # Data-driven best-fit μ: jointly fit (density, beta, μ) with μ free within
        # [0, 2/3], since μ affects the residual here (see docstring notes).
        if fit_density:
            x0_mu_best = [dens0, beta0, 1/3]
            bnds_mu_best = ((xmin[0], xmax[0]), (xmin[1], xmax[1]), (0.0, 2/3))
        else:
            x0_mu_best = [beta0, 1/3]
            bnds_mu_best = ((xmin[1], xmax[1]), (0.0, 2/3))

        res_mu_best = scipy.optimize.minimize(
            _densityBetaMuMinimizationFinalMass,
            x0_mu_best,
            args=(v_normed, ht_normed),
            bounds=bnds_mu_best,
            method='Powell'
        )

        if not res_mu_best.success:
            print(
                f"WARNING: Optimizer failed for best-fit μ: {res_mu_best.message}"
            )

        density_mu_best, rest = _getDensity(res_mu_best.x)
        beta_mu_best, mu_best = rest
        mu_best = float(np.clip(mu_best, 0.0, 2/3))

        m_initial_mu_best, alpha_mu_best = \
            _reconstructInitialMassAndAlpha(
                density_mu_best,
                beta_mu_best,
                mu_best
        )

        m_initial_mu_best, m_final_mu_best = _massesAt(alpha_mu_best, beta_mu_best, mu_best, density_mu_best)

    elif mass_constraint == "both":

        if fit_density:
            res_mu0 = scipy.optimize.minimize(
                _densityMinimizationBothMasses,
                [dens0],
                args=(v_normed, ht_normed, 0),
                bounds=[(xmin[0], xmax[0])],
                method="Powell"
            )

            if not res_mu0.success:
                print(
                    f"WARNING: Optimizer failed for μ=0: {res_mu0.message}"
                )

            density_mu0 = res_mu0.x[0]
        else:
            # Both masses and the density are fixed: beta and alpha follow directly,
            # nothing left to optimize.
            density_mu0 = density

        alpha_mu0, beta_mu0 = _solveBothMasses(density_mu0, 0.0)

        m_initial_mu0, m_final_mu0 = _massesAt(alpha_mu0, beta_mu0, 0, density_mu0)

        if fit_density:
            res_mu23 = scipy.optimize.minimize(
                _densityMinimizationBothMasses,
                [dens0],
                args=(v_normed, ht_normed, 2/3),
                bounds=[(xmin[0], xmax[0])],
                method="Powell"
            )

            if not res_mu23.success:
                print(
                    f"WARNING: Optimizer failed for μ=2/3: {res_mu23.message}"
                )

            density_mu23 = res_mu23.x[0]
        else:
            density_mu23 = density

        alpha_mu23, beta_mu23 = _solveBothMasses(density_mu23, 2.0/3.0)

        m_initial_mu23, m_final_mu23 = _massesAt(alpha_mu23, beta_mu23, 2/3, density_mu23)

        # Data-driven best-fit μ: jointly fit (density, μ) with μ free within [0, 2/3]
        # (or just μ, if density is set); beta is still derived analytically from
        # μ and the two given masses.
        if fit_density:
            x0_mu_best = [dens0, 1/3]
            bnds_mu_best = [(xmin[0], xmax[0]), (0.0, 2/3)]
        else:
            x0_mu_best = [1/3]
            bnds_mu_best = [(0.0, 2/3)]

        res_mu_best = scipy.optimize.minimize(
            _densityMuMinimizationBothMasses,
            x0_mu_best,
            args=(v_normed, ht_normed),
            bounds=bnds_mu_best,
            method="Powell"
        )

        if not res_mu_best.success:
            print(
                f"WARNING: Optimizer failed for best-fit μ: {res_mu_best.message}"
            )

        density_mu_best, rest = _getDensity(res_mu_best.x)
        mu_best = rest[0]
        mu_best = float(np.clip(mu_best, 0.0, 2/3))

        alpha_mu_best, beta_mu_best = _solveBothMasses(density_mu_best, mu_best)

        m_initial_mu_best, m_final_mu_best = _massesAt(alpha_mu_best, beta_mu_best, mu_best, density_mu_best)

    if verbose:
        print()
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
            print("density              = not given -> fitted below, separately for each μ")
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

        print("\n--- μ = 0 ---")
        print(f"density_mu0          = {density_mu0:.1f} kg/m^3{density_note}")
        print(f"alpha_mu0            = {alpha_mu0:.6f}")
        print(f"beta_mu0             = {beta_mu0:.6f}")
        print(f"m_initial_mu0        = {m_initial_mu0:.3f} kg")
        print(f"m_final_mu0          = {m_final_mu0:.3f} kg")
        print("\n--- μ = 2/3 ---")
        print(f"density_mu23         = {density_mu23:.1f} kg/m^3{density_note}")
        print(f"alpha_mu23           = {alpha_mu23:.6f}")
        print(f"beta_mu23            = {beta_mu23:.6f}")
        print(f"m_initial_mu23       = {m_initial_mu23:.3f} kg")
        print(f"m_final_mu23         = {m_final_mu23:.3f} kg")
        if mass_constraint == "initial":
            print("\n--- Best-fit μ ---")
            print("not applicable: μ has no effect on the fit under an initial-mass constraint")
        else:
            print("\n--- Best-fit μ ---")
            print(f"mu_best              = {mu_best:.3f}")
            print(f"density_mu_best      = {density_mu_best:.1f} kg/m^3{density_note}")
            print(f"alpha_mu_best        = {alpha_mu_best:.6f}")
            print(f"beta_mu_best         = {beta_mu_best:.6f}")
            print(f"m_initial_mu_best    = {m_initial_mu_best:.3f} kg")
            print(f"m_final_mu_best      = {m_final_mu_best:.3f} kg")


    # Relative distance-to-bound tolerance
    bound_tol = 1e-2
    warnings_list = []

    def _checkBound(name, value, lower, upper):

        span = upper - lower

        if (value - lower)/span < bound_tol:
            warnings_list.append(
                f"{name}={value:.6g} is close to the minimization lower bound ({lower:.6g})"
            )

        if (upper - value)/span < bound_tol:
            warnings_list.append(
                f"{name}={value:.6g} is close to the minimization upper bound ({upper:.6g})"
            )

    if fit_density:

        _checkBound(
            "density_mu0",
            density_mu0,
            xmin[0],
            xmax[0]
        )

        _checkBound(
            "density_mu23",
            density_mu23,
            xmin[0],
            xmax[0]
        )

    if mass_constraint in ["initial", "final"]:

        _checkBound(
            "beta_mu0",
            beta_mu0,
            xmin[1],
            xmax[1]
        )

        _checkBound(
            "beta_mu23",
            beta_mu23,
            xmin[1],
            xmax[1]
        )

    if fit_density and mass_constraint in ["final", "both"]:

        _checkBound(
            "density_mu_best",
            density_mu_best,
            xmin[0],
            xmax[0]
        )

    if mass_constraint == "final":

        _checkBound(
            "beta_mu_best",
            beta_mu_best,
            xmin[1],
            xmax[1]
        )

    def _checkVelocityRecovery(name, alpha, beta):
        """ Under mass_constraint="final"/"both", beta is derived analytically assuming
            a particular v_final (given or derived from a preliminary unconstrained fit),
            but alpha/beta are then fitted to best match the v-h curve shape. Nothing else
            guarantees that plugging the fitted alpha/beta back into alphaBetaVelocity()
            actually reproduces that same v_final, so check it explicitly here.
        """

        v_final_model = np.min(alphaBetaVelocity(ht_data, alpha, beta, v_init))
        rel_diff = abs(v_final_model - v_final)/v_final

        if rel_diff > bound_tol:
            warnings_list.append(
                f"{name}: model v_final from the fitted alpha/beta "
                f"({v_final_model:.1f} m/s) differs from the v_final used in the "
                f"mass constraint ({v_final:.1f} m/s) by {100*rel_diff:.2f}% "
                f"(> {100*bound_tol:.3g}%)"
            )

    if mass_constraint in ["final", "both"]:

        _checkVelocityRecovery("v_final_mu0", alpha_mu0, beta_mu0)
        _checkVelocityRecovery("v_final_mu23", alpha_mu23, beta_mu23)
        _checkVelocityRecovery("v_final_mu_best", alpha_mu_best, beta_mu_best)

    if warnings_list:

        print()
        print(
            f"WARNING: One or more fit-quality checks exceeded the "
            f"{100*bound_tol:.3g}% tolerance."
        )
        print(
            f"The fit may be poorly constrained (a fitted parameter landed within "
            f"{100*bound_tol:.3g}% of an optimization bound) or may not satisfy the "
            f"imposed constraints self-consistently (the model's recovered v_final "
            f"differs from the assumed v_final by more than {100*bound_tol:.3g}%)."
        )
        for warning in warnings_list:
            print(f"  - {warning}")
        print()

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