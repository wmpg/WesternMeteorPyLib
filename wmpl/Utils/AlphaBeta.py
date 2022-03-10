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

    def _expAtmosphereHeight(air_density, rho_atm_0=1.0):
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
    # first_part_indices = np.arange(0, len(weights)/2).astype(np.int)
    # weights[first_part_indices] = 1.0 - 0.5*first_part_indices/np.max(first_part_indices)
    # weights[~first_part_indices] = 0.5
    # weights /= np.sum(weights)

    # Don't use weights
    weights = np.ones_like(time_data)

    # Use robust fitting
    res = scipy.optimize.basinhopping(_lagMinimization, p0, \
        minimizer_kwargs={'args':(time_data, lag_data, weights)})
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

        # Compute the sum of absolute residuals (more robust than squared residuals)
        res = np.sum(np.abs(2*alpha*np.exp(-ht_normed) \
                         - (scipy.special.expi(beta) \
                            - scipy.special.expi(beta*v_normed**2))*np.exp(-beta)))

        return res


    # params = np.vstack((v_normed, ht_normed))

    # Compute initial alpha-beta guess
    b0 = 1.0
    a0 = np.exp(ht_normed[-1])/(2.0*b0)
    x0 = [a0, b0]

    # Set alpha-beta limits
    xmin = [    0.001,  0.00001]
    xmax = [10000.0,   50.0]
    bnds = ((xmin[0], xmax[0]), (xmin[1], xmax[1]))

    # Compute best-fit alpha-beta values
    res = scipy.optimize.minimize(_alphaBetaMinimization, x0, args=(v_normed, ht_normed), bounds=bnds, method='Nelder-Mead')

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
            - alpha: [float] Balistic coefficient.
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
        alpha: [float] Balistic coefficient.
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
        ht_data: [ndarray] Height data (m).
        alpha: [float] Balistic coefficient.
        beta: [float] Mass loss.
        v_init: [float] Input or derived initial velocity (m/s).

    Return:
        vel_data: [ndarray] Velocity data (m/s).
    
    """

    def _diff(v, alpha, beta, ht_target):
        """ Function to minimize the height. """
        
        # Compute the height using a guess velocity
        ht_guess = np.log(alpha) + beta - np.log((scipy.special.expi(beta) - scipy.special.expi(beta*v**2))/2)
        
        return (ht_guess - ht_target)**2

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

    return vel_data



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
        help='The product of the drag coefficient Gamma and the shape coefficient A. Used for computing the dynamic mass. Default is 0.7.', \
        type=float, default=0.7)

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    # If the trajectory pickle was given, load the orbital elements from it
    if cml_args.traj_path is not None:

        # Load the trajectory pickle
        traj = loadPickle(*os.path.split(cml_args.traj_path))


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


        ht_data = np.array(ht_data)
        time_data = np.array(time_data)
        lat_data = np.array(lat_data)
        lon_data = np.array(lon_data)
        vel_data = np.array(vel_data)
        lag_data = np.array(lag_data)

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

        print("Initial velocity:", traj.v_init)
        print("Lag fit:")
        print("    - a1    = {:.3f}".format(lag_fit_params[0]))
        print("    - a2    = {:.3f}".format(lag_fit_params[1]))
        print("    - t0    = {:.3f}".format(lag_fit_params[2]))
        print("    - decel = {:.3f}".format(lag_fit_params[3]))

        # Choose which data will be used for alpha-beta fitting
        if cml_args.obsvel:
            vel_input = vel_data
        else:
            vel_input = vel_data_smooth

        # Estimate the alpha, beta parameters
        v_init, alpha, beta = fitAlphaBeta(vel_input, ht_data_rescaled, v_init=traj.v_init)

        print()
        print("Alpha:", alpha)
        print("Beta:", beta)

        print()

        print("ln(beta)             = {:.2f}".format(np.log(beta)))
        print("ln(alpha*sin(slope)) = {:.2f}".format(np.log(alpha*np.sin(traj.orbit.elevation_apparent_norot))))


        # Predict velocity from height
        ht_end = traj.rend_ele - 5000
        if ht_end < 10000:
            ht_end = 10000
        elif (ht_end > 20000) and (ht_end < 35000):
            ht_end = 20000
        ht_arr = np.linspace(ht_end, traj.rbeg_ele + 5000, 200)
        vel_arr = alphaBetaVelocity(ht_arr, alpha, beta, v_init)




        fig, (ax_ab, ax_vel, ax_lag, ax_lag_res) = plt.subplots(ncols=4, sharey=True, figsize=(14, 6))


        ### Alpha-beta plot ###

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


        ### ###



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
                ax_ab.scatter([v_t0/1000], [h_rescaled_t0/1000], label='t0, decel = {:.2f} km/s^2'.format(abs(decel)),\
                    color='r')
                ax_vel.scatter([v_t0/1000], [h_t0/1000], label='t0, decel = {:.2f} km/s^2'.format(abs(decel)),\
                    color='r')


                ### Compute the dynamic mass at the end ###

                # Compute the values at the point that is 1/4 before the end and t0
                midpoint_index = int(round((t0_index + 0)*1/4)) # Sorted by increasing height!
                ht_dyn = ht_data[midpoint_index]
                t_dyn = time_data[midpoint_index]
                v_dyn = expLinearVelocity(t_dyn, traj.v_init, *lag_fit_params)

                # Compute the dynamic mass
                dyn_mass = dynamicMass(cml_args.dens, traj.rend_lat, traj.rend_lon, ht_dyn, traj.jdt_ref, \
                    v_dyn, 1000*abs(decel), gamma=cml_args.ga, shape_factor=1.0)

                # Plot the point where the dynamic mass is estiamted
                ax_vel.scatter([v_dyn/1000], [ht_dyn/1000], label='Dynamic mass = {:.3f} kg'.format(dyn_mass),\
                    color='k')


                print()
                print("Dynamic mass:")
                print("-------------")
                print("Bulk density = {:5d} kg/m^3".format(int(cml_args.dens)))
                print("Height       = {:5.2f} km".format(ht_dyn/1000))
                print("Velocity     = {:5.2f} km/s".format(v_dyn/1000))
                print("Deceleration = {:5.2f} km/s^3".format(abs(decel)))
                print("Gamma*A      = {:5.2f}".format(cml_args.ga))
                print()
                print("Dynamic mass = {:5.3f} kg".format(dyn_mass))
                print()
                print("--------------")

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
        x_mu = np.arange(0,10, 0.00005)

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
        plt.plot(x_mu, y_50g_mu0, color='grey', label="50 g meteorite, mu = 0", linestyle='dashed')
        plt.plot(x_mu, y_50g_mu23, color='k',   label="50 g meteorite, mu = 2/3", linestyle='dashed')
        plt.plot(x_mu, y_1kg_mu0, color='grey', label="1 kg meteorite, mu = 0")
        plt.plot(x_mu, y_1kg_mu23, color='k',   label="1 kg meteorite, mu = 2/3")
        plt.scatter([np.log(alpha*np.sin(traj.orbit.elevation_apparent_norot))], [np.log(beta)], color='r')

        # defite plot parameters
        plt.xlim((-1, 8))
        plt.ylim((-5, 4))
        plt.xlabel("ln(alpha*sin(slope))")
        plt.ylabel("ln(beta)")
        plt.axes().set_aspect('equal')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()



        ### Plot dynamic pressure ###

        # Take mean meteor lat/lon as reference for the atmosphere model
        lat_mean = np.mean([traj.rbeg_lat, traj.rend_lat])
        lon_mean = meanAngle([traj.rbeg_lon, traj.rend_lon])

        # Compute the dynamic pressure
        dyn_pressure = dynamicPressure(lat_mean, lon_mean, ht_arr, traj.jdt_ref, vel_arr)


        # Plot dyn pressure
        plt.plot(dyn_pressure/1e6, ht_arr/1000, color='k')


        # Compute and mark peak on the graph
        peak_dyn_pressure_index = np.argmax(dyn_pressure)
        peak_dyn_pressure = dyn_pressure[peak_dyn_pressure_index]/1e6
        peak_dyn_pressure_ht = ht_arr[peak_dyn_pressure_index]/1000
        plt.scatter(peak_dyn_pressure, peak_dyn_pressure_ht, \
            label="Peak P = {:.2f} MPa\nHt = {:.2f} km".format(peak_dyn_pressure, peak_dyn_pressure_ht))



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