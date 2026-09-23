""" NRL MSIS atmosphere mass density model, evaluated using pymsis. """

from __future__ import print_function, division, absolute_import

import datetime

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from pymsis import calculate, Variable

from wmpl.Utils.TrajConversions import jd2Date, datetime2JD


# MSIS version used for all atmosphere density evaluations. "00" is NRLMSISE-00, the model WMPL has
#   always used, "2.0" and "2.1" are the newer NRLMSIS 2.x releases which give up to 20% lower
#   densities around 85 and 120 km
MSIS_VERSION = "00"

# Julian date used instead of the one passed to getAtmDensity. Only set when the date cannot be taken
#   from the input data, e.g. when the model is not run on a trajectory pickle
MSIS_JD = None



def addAtmosphereArguments(arg_parser):
    """ Add the atmosphere model options to a command line argument parser. Apply them by calling
        setAtmosphere() on the parsed arguments.

    Arguments:
        arg_parser: [ArgumentParser] Argument parser to add the options to.
    """

    arg_parser.add_argument('--atm', metavar='MSIS_VERSION', type=str, default=MSIS_VERSION, \
        choices=['00', '2.0', '2.1'], \
        help="MSIS atmosphere model: 00 for NRLMSISE-00 (default), 2.0 or 2.1 for NRLMSIS 2.x.")

    arg_parser.add_argument('--atmtime', metavar='ATM_TIME', type=str, default=None, \
        help="UTC date and time at which the atmosphere is evaluated. Format: YYYYMMDD-HHMMSS. By "
        "default the reference time of the input data is used, e.g. of the trajectory pickle.")



def setAtmosphere(cml_args):
    """ Apply the atmosphere options added by addAtmosphereArguments to all density evaluations.

    Arguments:
        cml_args: [Namespace] Parsed command line arguments.
    """

    global MSIS_VERSION, MSIS_JD

    MSIS_VERSION = cml_args.atm

    MSIS_JD = None if cml_args.atmtime is None \
        else datetime2JD(datetime.datetime.strptime(cml_args.atmtime, "%Y%m%d-%H%M%S"))



def atmDensPoly6th(ht, dens_co):
    """ Compute the atmosphere density using a 6th order polynomial. This is used in the ablation simulation
        for faster execution. 

    Arguments:
        ht: [float] Height above sea level (m).
        dens_co: [list] Coeffs of the 6th order polynomial.

    Return: 
        atm_dens: [float] Atmosphere neutral mass density in kg/m^3.
    """

    # Compute the density
    rho_a = 1000*(10**(dens_co[0] 
                     + dens_co[1]*(ht/1000)
                     + dens_co[2]*(ht/1000)**2 
                     + dens_co[3]*(ht/1000)**3 
                     + dens_co[4]*(ht/1000)**4 
                     + dens_co[5]*(ht/1000)**5))

    return rho_a



def atmDensPoly(ht, dens_co):
    """ Compute the atmosphere density using a 7th order polynomial. This is used in the ablation simulation
        for faster execution. 

    Arguments:
        ht: [float] Height above sea level (m).
        dens_co: [list] Coeffs of the 7th order polynomial.

    Return: 
        atm_dens: [float] Atmosphere neutral mass density in kg/m^3. Note that the minimum set density is
            10^-14 kg/m^3.
    """

    # Compute the density (height is scaled to megameters to avoid overflows when raising it to the 6th power)
    rho_a = 10**(dens_co[0] 
               + dens_co[1]*(ht/1e6) 
               + dens_co[2]*(ht/1e6)**2 
               + dens_co[3]*(ht/1e6)**3 
               + dens_co[4]*(ht/1e6)**4 
               + dens_co[5]*(ht/1e6)**5
               + dens_co[6]*(ht/1e6)**6
               )

    # Set a minimum density
    if isinstance(rho_a, np.ndarray):
        rho_a[rho_a == 0] = 1e-14
    else:
        if rho_a == 0:
            rho_a = 1e-14

    return rho_a



def fitAtmPoly(lat, lon, height_min, height_max, jd):
    """ Fits a 7th order polynomial on the atmosphere mass density profile at the given location, time, and 
        for the given height range.

    Arguments:
        lat: [float] Latitude in radians.
        lon: [float] Longitude in radians.
        height_min: [float] Minimum height in meters. E.g. 30000 or 60000 are good values.
        height_max: [float] Maximum height in meters. E.g. 120000 or 180000 are good values.
        jd: [float] Julian date.

    Return:
        dens_co: [list] Coeffs for the 7th order polynomial.
    """

    # Generate a height array
    height_arr = np.linspace(height_min, height_max, 200)

    # Get atmosphere densities from the MSIS model (use log values for the fit)
    atm_densities = getAtmDensity(lat, lon, height_arr, jd)
    atm_densities_log = np.log10(atm_densities)


    def atmDensPolyLog(height_arr, *dens_co):
        return np.log10(atmDensPoly(height_arr, dens_co))

    # Fit the 7th order polynomial
    dens_co, _ = scipy.optimize.curve_fit(atmDensPolyLog, height_arr, atm_densities_log, \
        p0=np.zeros(7), maxfev=10000)

    return dens_co


    


def getAtmDensity(lat, lon, height, jd):
    """ For the given heights, returns the atmospheric density from the MSIS model. The model version is
        given by MSIS_VERSION, see setAtmosphere().

    More info: https://swxtrec.github.io/pymsis/

    Arguments:
        lat: [float or ndarray] Latitude in radians.
        lon: [float or ndarray] Longitude in radians.
        height: [float or ndarray] Height in meters.
        jd: [float] Julian date. Ignored if a date was set using setAtmosphere().

    Return:
        [float or ndarray] Atmosphere mass density in kg/m^3.

    """

    # Take the date given on the command line, if there was one
    if MSIS_JD is not None:
        jd = MSIS_JD

    # Broadcast the inputs to a common shape, so that pymsis evaluates them point by point
    lat, lon, height = np.broadcast_arrays(np.degrees(lat), np.degrees(lon), height)
    dt_arr = np.full(lat.size, np.datetime64(jd2Date(jd, dt_obj=True)))

    # f107, f107A, and ap effects are neither large nor well established below 80 km and these parameters
    #   should be set to 150., 150., and 4. respectively
    f107_arr = np.full(lat.size, 150.0)
    ap_arr = np.full((lat.size, 7), 4.0)

    # Take the total mass density out of the 11 variables that the model returns. Giving all inputs
    #   the same length makes pymsis return one row per point, instead of a grid
    atm_dens = calculate(dt_arr, lon.ravel(), lat.ravel(), height.ravel()/1000, f107_arr, f107_arr, \
        ap_arr, version=MSIS_VERSION)[:, Variable.MASS_DENSITY].astype(np.float64)

    # Return a scalar if only scalars were given
    if lat.ndim == 0:
        return float(atm_dens[0])

    return atm_dens.reshape(lat.shape)



# getAtmDensity handles arrays directly, the alias is kept for backwards compatibility
getAtmDensity_vect = getAtmDensity




if __name__ == "__main__":

    lat = 44.327234
    lon = -81.372350
    jd = datetime2JD(datetime.datetime.now(datetime.timezone.utc))

    # Height range (km)
    height_min = 20
    height_max = 180

    # Density evaluation heights (m)
    heights = np.linspace(height_min, height_max, 100)*1000

    atm_densities = getAtmDensity(np.radians(lat), np.radians(lon), heights, jd)

    plt.semilogx(atm_densities, heights/1000, zorder=3, label="MSIS " + MSIS_VERSION)


    # Fit the 6th order poly model
    dens_co = fitAtmPoly(np.radians(lat), np.radians(lon), 1000*height_min, 1000*height_max, jd)

    print(dens_co)

    # Plot the fitted poly model
    plt.semilogx(atmDensPoly(heights, dens_co), heights/1000, label="Poly fit")

    plt.legend()

    plt.xlabel('Density (kg/m^3)')
    plt.ylabel('Height (km)')

    plt.grid()

    plt.title('MSIS ' + MSIS_VERSION)

    # plt.savefig('atm_dens.png', dpi=300)

    plt.show()