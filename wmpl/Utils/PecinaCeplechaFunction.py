
import numpy as np
import scipy.optimize
import scipy.special
import scipy.integrate
import scipy.interpolate

from wmpl.Utils.AtmosphereDensity import atmDensPoly, fitAtmPoly


# Define the ceiling height (assumed to be h_inf in terms of the air density)
HT_CEILING = 180



def lenFromHt(h, c, zr):
    """ Compute the length from the height, constant c, and the zenith angle zr. 
    
    Arguments:
        h: [float] Height in km.
        c: [float] Height-length constant (km).
        zr: [float] Zenith angle (radians).

    Return:
        l: [float] Length (km).

    """

    l = c - h/np.cos(zr)

    return l


def _lenFromHtResidual(params, ht_data, len_target):
    """ Residual function usef for finding the constant c and the zenith angle. """

    c, zr = params

    return np.sum((len_target - lenFromHt(ht_data, c, zr))**2)



def htFromLen(l, c, zr):
    """ Compute the height from the length, constant c, and the zenith angle zr. 
    
    Arguments:
        l: [float] Length (km).
        c: [float] Height-length constant (km).
        zr: [float] Zenith angle (radians).

    Return:
        h: [float] Height in km.

    """

    return (c - l)*np.cos(zr)



def velFromHtPhysicalParams(ht_arr, v_inf, m_inf, sigma, zr, K, dens_interp):
    """ For the given height as meteoroid parameters, compute the velocity. 
    
    Arguments:
        ht_arr: [ndarray] Height in meters.
        v_inf: [float] Initial velocity in m/s.
        m_inf: [float] Mass in kg.
        sigma: [float] Ablation coefficient in m^2/s^2.
        zr: [float] Zenith angle (radians).
        K: [float] Shape-density coefficient (m^2/kg^(2/3)).
        dens_interp: [scipy.interpol handle] Interpolation handle for the air mass density in kg/m^3 where input is in meters.

    Return:
        vel_arr: [ndarray] Velocity for every given height (m/s).
    """


    # Convert to km as it keeps the values in the Ei intergral small
    ht_arr = np.array(ht_arr)/1000
    v_inf /= 1000
    sigma *= 1e6


    vel_arr = []

    # Go through the whole array of heights (in km)
    for ht in ht_arr:

        # Integrate the air density (compute in kg/m^3)
        air_dens_integ = scipy.integrate.quad(dens_interp, 1000*ht, 1000*HT_CEILING)[0]

        # Compute the Ei((sigma*v**2)/6) term
        eiv_term = scipy.special.expi((sigma*v_inf**2)/6) - (2*K*np.exp((sigma*v_inf**2)/6))/((m_inf**(1/3.0))*np.cos(zr))*air_dens_integ

        
        ### Numerically invert the velocity from the exponential integral ###

        def _diff(v, sigma, eiv_target):

            # Compute the guess value of the Ei((sigma*v**2)/6) term for the given velocity
            eiv_guess = scipy.special.expi((sigma*v**2)/6)

            # Compute the square residual
            return (eiv_target - eiv_guess)**2


        v_first_guess = v_inf
        v_bounds = [(0.1, 80)]

        res = scipy.optimize.minimize(_diff, v_first_guess, args=(sigma, eiv_term), bounds=v_bounds)

        # print()
        # print("HT:", ht)
        # print("Air integ:", air_dens_integ)
        # print("E_vinf:", scipy.special.expi((sigma*v_inf**2)/6))
        # print("EIV:", eiv_term)
        # print("vel:", res.x[0])

        vel = res.x[0]


        ###

        # Store the velocity in m/s
        vel_arr.append(1000*vel)

        


    return np.array(vel_arr)



def velFromHt(ht_arr, h0, v0, v_inf, sigma, c, zr, dens_interp):
    """ Compute the velocity given the height and parameters as defined by Pecina & Ceplecha (1984) model. 
    
    Arugments:
        ht_arr: [ndarray] Height in km.
        h0: [float] Height of the reference point (seconds).
        v0: [float] Velocity at the reference point (km/s).
        v_inf: [float] Velocity at infinity (km/s).
        sigma: [float] Ablation coefficeint km^2/s^2.
        c: [float] Height-length constant (km).
        zr: [float] Zenith angle (radians).
        dens_interp: [scipy.interpol handle] Interpolation handle for the air mass density in kg/m^3 where input is in meters.

    Return:
        vel_arr: [ndarray] Velocity at the given height (km/s).

    """


    vel_arr = []


    for ht in ht_arr:

        # Integrate the air density from the reference point to infinity (compute in kg/m^3)
        air_dens_integ_h0 = scipy.integrate.quad(dens_interp, 1000*h0, 1000*HT_CEILING)[0]

        # Integrate the air density from the given height to infinity (compute in kg/m^3)
        air_dens_integ_ht = scipy.integrate.quad(dens_interp, 1000*ht, 1000*HT_CEILING)[0]

        # Compute the Ei((sigma*v**2)/6) term
        eiv_term = scipy.special.expi((sigma*v_inf**2)/6) - (scipy.special.expi((sigma*v_inf**2)/6) \
            - scipy.special.expi((sigma*v0**2)/6))*air_dens_integ_ht/air_dens_integ_h0



        ### Numerically invert the velocity from the exponential integral ###

        def _diff(v, sigma, eiv_target):

            # Compute the guess value of the Ei((sigma*v**2)/6) term for the given velocity
            eiv_guess = scipy.special.expi((sigma*v**2)/6)

            # Compute the square residual
            return (eiv_target - eiv_guess)**2


        v_first_guess = v_inf
        v_bounds = [(0.1, 80)]

        res = scipy.optimize.minimize(_diff, v_first_guess, args=(sigma, eiv_term), bounds=v_bounds)


        vel = res.x[0]

        ###

        # Store the velocity in km/s
        vel_arr.append(vel)


    return np.array(vel_arr)



def timeFromLen(len_arr, t0, l0, v0, v_inf, sigma, c, zr, dens_interp):
    """ Compute the time given the length of a Pecina & Ceplecha (1984) model.

    Arugments:
        len_arr: [ndarray] Length in km.
        t0: [float] Time of the reference point (seconds).
        l0: [float] Length of the reference point (km).
        v0: [float] Velocity at the reference point (km/s).
        v_inf: [float] Velocity at infinity (km/s).
        sigma: [float] Ablation coefficeint km^2/s^2.
        c: [float] Height-length constant (km).
        zr: [float] Zenith angle (radians).
        dens_interp: [scipy.interpol handle] Interpolation handle for the air mass density in kg/m^3 where input is in meters.

    Return:
        time_arr: [ndarray] Time at the given length (seconds).

    """

    # Compute the h0 limit
    h0 = htFromLen(l0, c, zr)


    # Compute the height for the given length
    ht_arr = [htFromLen(l, c, zr) for l in len_arr]


    # Compute the velocity from the height
    vel_arr = velFromHt(ht_arr, h0, v0, v_inf, sigma, c, zr, dens_interp)


    # Compute the time from length
    time_arr = []
    for l, vel in zip(len_arr, vel_arr):

        # Interpolate the inverse velocity over length
        inv_vel_interp = scipy.interpolate.CubicSpline(len_arr, 1.0/vel_arr)

        # Integrate the velocity^-1 over length to compute the relative time from t0 
        vel_integ = scipy.integrate.quad(inv_vel_interp, l0, l)[0]

        # Compute the final time
        t = t0 + vel_integ

        time_arr.append(t)


    return np.array(time_arr)



def jacchiaFuncLen(t, a1, a2, a3, a4):
    """ Predict the length from time using the Jacchia exponential function. """

    return a1 + a2*t - np.abs(a3)*np.exp(np.abs(a4)*t)

def jacchiaFuncVel(t, a1, a2, a3, a4):
    """ Predict the velocity from time using the Jacchia exponential function. """

    return a2 - np.abs(a3*a4)*np.exp(np.abs(a4)*t)



def fitPecinaCeplecha84Model(lat, lon, jd, time_data, ht_data, len_data, dens_interp=None, sigma_initial=0.03):
    """ Fit the Pecina & Ceplecha (1984) model to the given data. 
    
    Arguments:
        lat: [float] Latitude (radians).
        Lon: [float] Longitude (radians).
        jd: [float] Julian date of the event.
        time_data: [ndarray] Relative time (seconds).
        ht_data: [ndarray] Height (km).
        len_data: [ndarray] Length (km).

    Keyword arguments:
        dens_interp: [func] Function which takes the height (in METERS!) and return the atmosphere density 
            at the given point in kg/m^3. If not given, it will be computed.
        sigma_initial: [float] Initial ablation coefficient (km^2/s^2). The fit is very dependent on this 
            number and different numbers should be tried to improve the fit. sigma = 0.03 by default.

    Return:
        t0: [float] Time of the reference point (seconds).
        l0: [float] Length of the reference point (km).
        v0: [float] Velocity at the reference point (km/s).
        v_inf: [float] Velocity at infinity (km/s).
        sigma: [float] Ablation coefficeint km^2/s^2.
        c: [float] Height-length constant (km).
        zr: [float] Zenith angle (radians).
        dens_interp: [scipy.interpol handle] Interpolation handle for the air mass density in kg/m^3 where 
            input is in meters.
    """

    ### FIT THE AIR DENSITY MODEL ###

    # Fit a 7th order polynomial to the air mass density from NRL-MSISE from the ceiling height to 3 km below
    #   the fireball - limit the height to 12 km
    ht_min = np.min(ht_data) - 3
    if ht_min < 12:
        ht_min = 12


    if dens_interp is None:

        # Compute the poly fit
        print("Fitting atmosphere polynomial...")
        dens_co = fitAtmPoly(lat, lon, 1000*ht_min, 1000*HT_CEILING, jd)

        # Create a convinience function for compute the density at the given height
        dens_interp = lambda h: atmDensPoly(h, dens_co)

        print("   ... done!")


    ###


    ### FIT THE HEIGHT-LENGTH CONSTANT
    print("Finding height-length constant...")

    # Find the height-length constant and zenith angle
    p0 = [0, np.radians(45)]
    res = scipy.optimize.minimize(_lenFromHtResidual, p0, args=(ht_data, len_data))

    
    # Extracted fitted parameters
    c, zr = res.x
    zr = np.abs(zr)

    print("c  = {:.2f} km".format(c))
    print("zr = {:.2f} deg".format(np.degrees(zr)))


    # # Plot the c, zr fit
    # ht_arr = np.linspace(np.min(ht_data), np.max(ht_data), 100)
    # plt.scatter(ht_data, len_data)
    # plt.plot(ht_arr, lenFromHt(ht_arr, c, zr))
    # plt.xlabel("Height (km)")
    # plt.ylabel("Length (km)")
    # plt.show()

    ###


    def _jacchiaResiduals(params, len_target, time_data):
        return np.sum((len_target - jacchiaFuncLen(time_data, *params))**2)

        #return np.sum(np.abs(len_target - jacchiaFuncLen(time_data, *params)))

    # Fit the Jacchia function to get the initial estimate of the fit parameters
    p0 = [0, 10, 0, 1]
    res = scipy.optimize.minimize(_jacchiaResiduals, p0, args=(len_data, time_data), method='Nelder-Mead')
    a1, a2, a3, a4 = res.x

    # # Show Jacchia fit
    # plt.scatter(time_data, len_data)
    # plt.plot(time_data, jacchiaFuncLen(time_data, a1, a2, a3, a4))
    # plt.show()


    def _residuals(params, t0, c, zr, dens_interp, len_arr, time_target):
        """ Residuals function for the model fit. """

        l0, v0, v_inf, sigma = params

        # Compute the time guess with the given parameters
        time_arr = timeFromLen(len_arr, t0, l0, v0, v_inf, sigma, c, zr, dens_interp)

        # Sum of squared residuals
        cost = np.sum((time_target - time_arr)**2)

        # # Sum of absolute residuals
        # cost = np.sum(np.abs(time_target - time_arr))

        print("Cost = {:16.10f}, guess: l0 = {:7.3f}, v0 = {:6.3f}, vi = {:6.3f}, sigma = {:.5f}".format(cost, *params))

        return cost

            
    # Choose t0 at the 0.77*max_time (converges better if this is at a point where there's deceleration)
    t0 = 0.77*np.max(time_data)

    print("t0 = {:.2f} s".format(t0))
    
    # Construct the initial guess of the fit parameters using the Jacchia function
    l0 = jacchiaFuncLen(t0, a1, a2, a3, a4)
    v0 = jacchiaFuncVel(t0, a1, a2, a3, a4)
    v_inf = a2
    sigma = sigma_initial # km^2/s^2

    # Separate initial guess velocities if they are too close
    if (v_inf - v0) < 1:
        v0 = v_inf - 2

    p0 = [l0, v0, v_inf, sigma]

    print("Initial parameters:", p0)

    # Set the optimization bounds
    bounds = [
        ( 0, np.max(len_data)),  # l0
        ( 0, 80.0),              # v0
        (10, 80.0),              # v_inf  
        (0.0001, 1.0)             # sigma
    ]

    # Set the constraint that v_inf > v0
    constraints = ({'type': 'ineq',
                    'fun': lambda x: x[2] - x[1]})

    # Fit the parameters to the observations
    res = scipy.optimize.minimize(_residuals, p0, args=(t0, c, zr, dens_interp, len_data, time_data), \
        bounds=bounds, constraints=constraints, method='SLSQP')


    # # Default tolerance using by SLSQP
    # ftol = 1e-06

    # # Compute the formal uncertainties
    # # Source: https://stackoverflow.com/a/53489234
    # tmp_i = np.zeros(len(res.x))
    # for i in range(len(res.x)):
    #     tmp_i[i] = 1.0
    #     hess_inv_i = res.hess_inv(tmp_i)[i]
    #     uncertainty_i = np.sqrt(max(1, abs(res.fun))*ftol*hess_inv_i)
    #     tmp_i[i] = 0.0
    #     print('x^{0} = {1:.3f} ± {2:.6f}'.format(i, res.x[i], uncertainty_i))
    

    l0, v0, v_inf, sigma = res.x


    return t0, l0, v0, v_inf, sigma, c, zr, dens_interp




if __name__ == "__main__":

    import os
    import sys
    import argparse

    import matplotlib.pyplot as plt

    from wmpl.Utils.Pickling import loadPickle


    # ### COMMAND LINE ARGUMENTS

    # # Init the command line arguments parser
    # arg_parser = argparse.ArgumentParser(description="""Fit the Pecina & Ceplecha (1984) model to a trajectory in the pickle file.""",
    #     formatter_class=argparse.RawTextHelpFormatter)

    # arg_parser.add_argument('input_file', type=str, help='Path to the .pickle file.')

    # # Parse the command line arguments
    # cml_args = arg_parser.parse_args()

    # ############################


    # # Load the pickle file
    # if not os.path.isfile(cml_args.input_file):

    #     print("Could not find file:", cml_args.input_file)
    #     print("Exiting...")

    #     sys.exit()



    # # Load the trajectory pickle file
    # traj = loadPickle(*os.path.split(cml_args.input_file))

    # # Extract the time, height, and length data
    # time_data = []
    # len_data = []
    # ht_data = []
    # vel_data = []
    # for obs in traj.observations:

    #     # Relative time in seconds
    #     time_obs = obs.time_data[obs.ignore_list == 0]
    #     time_data += time_obs.tolist()

    #     # Height in km
    #     ht_obs = obs.model_ht[obs.ignore_list == 0]/1000
    #     ht_data += ht_obs.tolist()

    #     # Length in km
    #     len_obs = obs.state_vect_dist[obs.ignore_list == 0]/1000
    #     len_data += len_obs.tolist()

    #     # Velocity in km/s
    #     vel_obs = obs.velocities[obs.ignore_list == 0]/1000
    #     vel_data += vel_obs.tolist()


    # # Sort observations by length
    # tmp_arr = np.c_[time_data, ht_data, len_data, vel_data]
    # tmp_arr = tmp_arr[np.argsort(len_data)]
    # time_data, ht_data, len_data, vel_data = tmp_arr.T

    # # # Check data 
    # # plt.scatter(time_data, len_data)
    # # plt.show()

    # # plt.scatter(ht_data, vel_data)
    # # plt.show()


    # # Fit the Pecina & Ceplecha (1984) model to observations
    # t0, l0, v0, v_inf, sigma, c, zr, dens_interp = fitPecinaCeplecha84Model(traj.rend_lat, traj.rend_lon, \
    #     traj.jdt_ref, time_data, ht_data, len_data)


    # print("Solution:")
    # print("    t0    = {:.3f} s".format(t0))
    # print("    l0    = {:.3f} km".format(l0))
    # print("    v0    = {:.3f} km/s".format(v0))
    # print("    v_inf = {:.3f} km/s".format(v_inf))
    # print("    sigma = {:.6f} km^2/s^2".format(sigma))



    # # Compute the h0 limit
    # h0 = htFromLen(l0, c, zr)

    # # Compute the velocity from height and model parameters
    # ht_arr = ht_dens_arr = np.linspace(1000*np.min(ht_data), 1000*np.max(ht_data), 100)
    # vel_arr = 1000*velFromHt(ht_arr/1000, h0, v0, v_inf, sigma, c, zr, dens_interp)

    # # Plot velocity observations vs fit
    # plt.scatter(vel_data[vel_data > 0], ht_data[vel_data > 0])
    # plt.plot(vel_arr/1000, ht_arr/1000)

    # plt.xlabel("Velocity (km/s)")
    # plt.ylabel("Height (km)")

    # plt.show()



    # # Compute the time from height and model parameters
    # len_arr = np.linspace(np.min(len_data), np.max(len_data), 100)
    # time_arr = timeFromLen(len_arr, t0, l0, v0, v_inf, sigma, c, zr, dens_interp)

    # # Plot time vs length observations vs fit
    # plt.scatter(time_data, len_data)
    # plt.plot(time_arr, len_arr)

    # plt.xlabel("Time (s)")
    # plt.ylabel("Length (km)")

    # plt.show()


    # # Plot fit residuals
    # time_residuals = time_data - timeFromLen(len_data, t0, l0, v0, v_inf, sigma, c, zr, dens_interp)
    # plt.scatter(len_data, time_residuals)

    # # Plot the zero line
    # plt.plot(len_arr, np.zeros_like(len_arr), c='k', linestyle='dashed')

    # plt.xlabel("Length (km)")
    # plt.ylabel("Time residuals (s)")

    # max_res = 1.2*np.max(np.abs(time_residuals))
    # plt.ylim(-max_res, max_res)
    # plt.show()




    # sys.exit()

    ### BELOW IS THE EXAMPLE FOR THE ORIGINAL PAPER ###


    # Location data for the PN example event (rough)
    lat = np.radians(50)
    lon = np.radians(-107)
    jd = 2444239.50000


    # Example data from Pecina & Ceplecha (1983) for PN 39 404
    pn_data = np.array([
    # t(s),h (km),l (km)
     [0.00,79.174,0.000],
     [0.05,78.581,0.714],
     [0.10,77.904,1.530],
     [0.15,77.311,2.246],
     [0.25,76.015,3.808],
     [0.30,75.384,4.569],
     [0.40,74.111,6.102],
     [0.45,73.461,6.886],
     [0.50,72.837,7.639],
     [0.55,72.195,8.413],
     [0.60,71.556,9.183],
     [0.65,70.909,9.964],
     [0.70,70.269,10.735],
     [0.75,69.646,11.487],
     [0.90,67.750,13.773],
     [1.00,66.482,15.303],
     [1.05,65.852,16.062],
     [1.10,65.229,16.814],
     [1.15,64.596,17.578],
     [1.20,63.960,18.345],
     [1.25,63.338,19.096],
     [1.30,62.694,19.873],
     [1.35,62.086,20.606],
     [1.40,61.449,21.376],
     [1.45,60.829,22.123],
     [1.55,59.558,23.657],
     [1.60,58.949,24.392],
     [1.70,57.685,25.918],
     [1.75,57.055,26.679],
     [1.80,56.424,27.440],
     [1.85,55.795,28.199],
     [1.90,55.187,28.933],
     [1.95,54.576,29.671],
     [2.00,53.995,30.372],
     [2.05,53.340,31.163],
     [2.20,51.410,33.493],
     [2.30,50.191,34.966],
     [2.35,49.563,35.724],
     [2.40,48.892,36.534],
     [2.45,48.294,37.257],
     [2.50,47.682,37.996],
     [2.55,47.107,38.691],
     [2.60,46.500,39.424],
     [2.65,45.900,40.148],
     [2.70,45.289,40.887],
     [2.75,44.713,41.583],
     [2.85,43.532,43.010],
     [2.90,42.907,43.765],
     [2.95,42.363,44.422],
     [3.05,41.144,45.895],
     [3.10,40.581,46.575],
     [3.15,40.001,47.276],
     [3.20,39.478,47.909],
     [3.25,38.925,48.577],
     [3.30,38.369,49.249],
     [3.35,37.851,49.875],
     [3.50,36.290,51.762],
     [3.60,35.301,52.957],
     [3.65,34.825,53.533],
     [3.70,34.330,54.128],
     [3.75,33.915,54.633],
     [3.80,33.430,55.220],
     [3.85,32.993,55.743],
     [3.90,32.592,56.233],
     [3.95,32.184,56.727],
     [4.00,31.798,57.193],
     [4.05,31.436,57.631],
     [4.15,30.765,58.443],
     [4.20,30.442,58.832],
     [4.25,30.134,59.205],
     [4.35,29.565,59.894],
     [4.40,29.314,60.198],
     [4.45,29.049,60.517],
     [4.50,28.807,60.810],
     [4.55,28.567,61.101],
     [4.60,28.347,61.367]
     ])

    # Extract the example PN data
    time_data, ht_data, len_data = pn_data.T


    # Compute the point-to-point velocity
    len_diff = len_data[1:] - len_data[:-1]
    time_diff = time_data[1:] - time_data[:-1]
    vel_data = len_diff/time_diff



    # Fit the Pecina & Ceplecha (1984) model to observations
    t0, l0, v0, v_inf, sigma, c, zr, dens_interp = fitPecinaCeplecha84Model(lat, lon, jd, time_data, ht_data, len_data)



    # Compute the h0 limit
    h0 = htFromLen(l0, c, zr)

    # Compute the velocity from height and model parameters
    ht_arr = ht_dens_arr = np.linspace(1000*np.min(ht_data), 1000*np.max(ht_data), 100)
    vel_arr = 1000*velFromHt(ht_arr/1000, h0, v0, v_inf, sigma, c, zr, dens_interp)

    # Plot velocity observations vs fit
    plt.scatter(vel_data, ht_data[1:])
    plt.plot(vel_arr/1000, ht_arr/1000)

    plt.xlabel("Velocity (km/s)")
    plt.ylabel("Height (km)")

    plt.show()



    # Compute the time from height and model parameters
    len_arr = np.linspace(np.min(len_data), np.max(len_data), 100)
    time_arr = timeFromLen(len_arr, t0, l0, v0, v_inf, sigma, c, zr, dens_interp)

    # Plot time vs length observations vs fit
    plt.scatter(time_data, len_data)
    plt.plot(time_arr, len_arr)

    plt.xlabel("Time (s)")
    plt.ylabel("Length (km)")

    plt.show()


    sys.exit()



    ### BELOW  IS THE CHECK OF THE FUNCTIONS ON THE ORIGINAL VALUES FROM THE PAPER ###


    ### FIT THE AIR DENSITY MODEL ###

    # Fit a 7th order polynomial to the air mass density from NRL-MSISE from the ceiling height to 3 km below
    #   the fireball - limit the height to 12 km
    ht_min = np.min(ht_data) - 3
    if ht_min < 12:
        ht_min = 12


    # Compute the poly fit
    print("Fitting atmosphere polynomial...")
    dens_co = fitAtmPoly(lat, lon, 1000*ht_min, 1000*HT_CEILING, jd)

    # Create a convinience function for compute the density at the given height
    dens_interp = lambda h: atmDensPoly(h, dens_co)

    print("   ... done!")

    ###



    ### TEST EXAMPLE ###

    # PN 
    v_inf = 15.3456 # km/s
    m_inf = 91.2 # kg
    sigma = 0.0308 # km^2/s^2
    zr = np.radians(34.089)
    K = 1.0*1.2*650**(-2/3.0) # m^2/kg^(2/3)

    t0 = 3.5 # s
    l0 = 51.773 # km
    v0 = 12.281 # km/s


    # # Compute the velocity for every height using K
    # vel_arr = velFromHtPhysicalParams(ht_arr, 1000*v_inf, m_inf, sigma/1e6, zr, K, dens_interp)
    
    # # Plot observations vs fit
    # plt.scatter(ht_data[1:], vel_data)
    # plt.plot(ht_arr/1000, vel_arr/1000)

    # plt.show()

    # sys.exit()


    ###

    print("Finding height-length constant...")

    # Find the height-length constant and zenith angle
    p0 = [0, np.radians(45)]
    res = scipy.optimize.minimize(_lenFromHtResidual, p0, args=(ht_data, len_data))

    
    # Extracted fitted parameters
    c, zr = res.x
    zr = np.abs(zr)

    print("c  = {:.2f} km".format(c))
    print("zr = {:.2f} deg".format(np.degrees(zr)))



    # Compute the h0 limit
    h0 = htFromLen(l0, c, zr)

    # Compute the velocity from height and model parameters
    ht_arr = ht_dens_arr = np.linspace(1000*ht_min, 1000*np.max(ht_data), 100)
    vel_arr = 1000*velFromHt(ht_arr/1000, h0, v0, v_inf, sigma, c, zr, dens_interp)

    # Plot velocity observations vs fit
    plt.scatter(vel_data, ht_data[1:])
    plt.plot(vel_arr/1000, ht_arr/1000)

    plt.xlabel("Velocity (km/s)")
    plt.ylabel("Height (km)")

    plt.show()



    # Compute the time from height and model parameters
    len_arr = np.linspace(np.min(len_data), np.max(len_data), 100)
    time_arr = timeFromLen(len_arr, t0, l0, v0, v_inf, sigma, c, zr, dens_interp)

    # Plot time vs length observations vs fit
    plt.scatter(time_data, len_data)
    plt.plot(time_arr, len_arr)

    plt.xlabel("Time (s)")
    plt.ylabel("Length (km)")

    plt.show()