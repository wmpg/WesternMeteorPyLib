""" NRL MSIS atmosphere mass density model, evaluated using pymsis. """

from __future__ import print_function, division, absolute_import

import datetime
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from pymsis import calculate, Variable
import pymsis.utils

from wmpl.Utils.Math import expSegmentIntegral
from wmpl.Utils.TrajConversions import jd2Date, datetime2JD


# setAtmosphere() also keeps the choice in these environment variables, which every process started from this
#   one inherits, as a process started by multiprocessing with "spawn" imports this module afresh
MSIS_VERSION_ENV = "WMPL_MSIS_VERSION"
MSIS_JD_ENV = "WMPL_MSIS_JD"
MSIS_SPACE_WEATHER_ENV = "WMPL_MSIS_SPACE_WEATHER"

# MSIS version used for all atmosphere density evaluations, "2.1" (NRLMSIS 2.1) by default, "2.0" or "00" for
#   NRLMSISE-00
MSIS_VERSION = os.environ.get(MSIS_VERSION_ENV, "2.1")

# Julian date used instead of the one passed to getAtmDensity, only set when the date cannot be taken from the
#   input data
MSIS_JD = float(os.environ[MSIS_JD_ENV]) if os.environ.get(MSIS_JD_ENV) else None

# Solar and geomagnetic activity given to the model, "real" for the measured F10.7 and Ap of the date or
#   "constant" for the values below
MSIS_SPACE_WEATHER = os.environ.get(MSIS_SPACE_WEATHER_ENV, "real")

# Constant solar and geomagnetic activity, the recommended values below 80 km, also used when the measured
#   values are not available
CONSTANT_F107A = 150.0
CONSTANT_F107 = 150.0
CONSTANT_AP = 4.0

# Names of the MSIS versions
MSIS_MODEL_NAMES = {"00": "NRLMSISE-00", "2.0": "NRLMSIS 2.0", "2.1": "NRLMSIS 2.1"}

# Height range and step of the atmosphere density tables of MetSim (m), covering the whole simulation
ATM_TABLE_HT_MIN = 0.0
ATM_TABLE_HT_MAX = 200000.0
ATM_TABLE_STEP = 500.0

# Whether the warning about missing space weather was printed, to print it only once
SPACE_WEATHER_WARNED = [False]



def addAtmosphereArguments(arg_parser):
    """ Add the atmosphere model options to a command line argument parser. Apply them by calling
        setAtmosphere() on the parsed arguments.

    Arguments:
        arg_parser: [ArgumentParser object] Argument parser to add the options to.

    Return:
        arg_parser: [ArgumentParser object] Parser with the options.
    """

    arg_parser.add_argument('--atm', metavar='MSIS_VERSION', type=str, default="2.1", \
        choices=['00', '2.0', '2.1'], \
        help="MSIS atmosphere model: 2.1 for NRLMSIS 2.1 (default), 2.0 for NRLMSIS 2.0 or 00 for " \
        "NRLMSISE-00.")

    arg_parser.add_argument('--atmtime', metavar='ATM_TIME', type=str, default=None, \
        help="UTC date and time at which the atmosphere is evaluated. Format: YYYYMMDD-HHMMSS. By " \
        "default the reference time of the input data is used, e.g. of the trajectory pickle.")

    arg_parser.add_argument('--atmsw', metavar='SPACE_WEATHER', type=str, default="real", \
        choices=['real', 'constant'], \
        help="Solar and geomagnetic activity given to the MSIS model: real for the measured F10.7 and Ap " \
        "of the date (default, downloaded once from CelesTrak), constant for F10.7 = 150 and Ap = 4.")

    return arg_parser



def setAtmosphere(cml_args):
    """ Apply the atmosphere options added by addAtmosphereArguments() to all density evaluations.

    Arguments:
        cml_args: [Namespace] Parsed command line arguments.

    Return:
        None
    """

    global MSIS_VERSION, MSIS_JD, MSIS_SPACE_WEATHER

    MSIS_VERSION = cml_args.atm
    MSIS_SPACE_WEATHER = getattr(cml_args, 'atmsw', "real")

    MSIS_JD = None if cml_args.atmtime is None \
        else datetime2JD(datetime.datetime.strptime(cml_args.atmtime, "%Y%m%d-%H%M%S"))

    # Keep the choice for the processes started from this one
    os.environ[MSIS_VERSION_ENV] = MSIS_VERSION
    os.environ[MSIS_SPACE_WEATHER_ENV] = MSIS_SPACE_WEATHER
    if MSIS_JD is None:
        os.environ.pop(MSIS_JD_ENV, None)
    else:
        os.environ[MSIS_JD_ENV] = repr(MSIS_JD)



def getMSISVersion():
    """ Return the MSIS version used for the atmosphere, e.g. "2.1" or "00". """

    return MSIS_VERSION



def msisModelName():
    """ Return the name of the MSIS model used for the atmosphere, e.g. "NRLMSIS 2.1". """

    return MSIS_MODEL_NAMES.get(MSIS_VERSION, "MSIS " + MSIS_VERSION)



def spaceWeather(jd):
    """ Get the solar and geomagnetic activity given to the MSIS model at the given Julian date.

    With the real space weather, the measured F10.7 of the previous day, its 81-day average and the daily Ap
    are taken from the CelesTrak file that pymsis downloads once. If they are not available, e.g. offline or
    for a future date, the constant values are used with a warning.

    Arguments:
        jd: [float] Julian date.

    Return:
        (f107, f107a, ap, source):
            - f107: [float] F10.7 of the previous day (sfu).
            - f107a: [float] 81-day average of F10.7 (sfu).
            - ap: [float] Daily Ap.
            - source: [str] 'real' or 'constant'.
    """

    if MSIS_SPACE_WEATHER == "real":

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                f107, f107a, ap = pymsis.utils.get_f107_ap(np.datetime64(jd2Date(jd, dt_obj=True)))

            if np.all(np.isfinite(f107)) and np.all(np.isfinite(f107a)) and np.all(np.isfinite(ap)):
                return float(np.ravel(f107)[0]), float(np.ravel(f107a)[0]), float(np.ravel(ap)[0]), 'real'

        except Exception as e:
            if not SPACE_WEATHER_WARNED[0]:
                print("WARNING: The measured F10.7 and Ap are not available ({:s}), so the constant values " \
                    "are used.".format(repr(e)))
                SPACE_WEATHER_WARNED[0] = True

    return CONSTANT_F107, CONSTANT_F107A, CONSTANT_AP, 'constant'



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
    """ Compute the atmosphere density using a 6th degree polynomial (7 coefficients) in log10 of the
        density.

    The polynomial deviates from the MSIS model by up to tens of percent at 80 to 120 km when it is fitted
    over a wide height range, e.g. 14 to 180 km for fireballs. The tabulated profile of getAtmDensityTable()
    follows the model to better than 0.5%.

    Arguments:
        ht: [float] Height above sea level (m).
        dens_co: [list] Coeffs of the 6th degree polynomial.

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
    """ Fits a 6th degree polynomial (7 coefficients) on the log10 of the atmosphere mass density profile at
        the given location, time, and for the given height range. See getAtmDensityTable() for a more
        accurate tabulated profile.

    Arguments:
        lat: [float] Latitude in radians.
        lon: [float] Longitude in radians.
        height_min: [float] Minimum height in meters. E.g. 30000 or 60000 are good values.
        height_max: [float] Maximum height in meters. E.g. 120000 or 180000 are good values.
        jd: [float] Julian date.

    Return:
        dens_co: [list] Coeffs for the 6th degree polynomial.
    """

    # Generate a height array
    height_arr = np.linspace(height_min, height_max, 200)

    # Get atmosphere densities from the MSIS model (use log values for the fit)
    atm_densities = getAtmDensity(lat, lon, height_arr, jd)
    atm_densities_log = np.log10(atm_densities)


    def atmDensPolyLog(height_arr, *dens_co):
        return np.log10(atmDensPoly(height_arr, dens_co))

    # Fit the 6th degree polynomial
    dens_co, _ = scipy.optimize.curve_fit(atmDensPolyLog, height_arr, atm_densities_log, \
        p0=np.zeros(7), maxfev=10000)

    return dens_co



def getAtmDensityTable(lat, lon, height_min, height_max, jd, step=ATM_TABLE_STEP):
    """ Tabulate log10 of the atmosphere mass density of the MSIS model (getAtmDensity()) at the given
        location and time on a uniform height grid.

    The ablation simulation interpolates log10 of the density linearly between the table points (see
    atmDensTable()). With the default step of 500 m this reproduces the model to better than 0.5% in the
    density and 0.02% in the air column above any height between 0 and 180 km.

    Arguments:
        lat: [float] Latitude in radians.
        lon: [float] Longitude in radians.
        height_min: [float] Minimum height in meters.
        height_max: [float] Maximum height in meters.
        jd: [float] Julian date.

    Keyword arguments:
        step: [float] Maximum height step of the table in meters. The range is split into equal steps no
            larger than this. ATM_TABLE_STEP by default.

    Return:
        (table_ht, table_log10_rho):
            - table_ht: [list] Table heights in meters, ascending, from height_min to height_max.
            - table_log10_rho: [list] log10 of the density (kg/m^3) at the table heights.
    """

    if height_max <= height_min:
        raise ValueError("height_max ({:.1f} m) must be above height_min ({:.1f} m)".format(height_max, \
            height_min))

    # Generate a height array with equal steps no larger than the requested step
    n_points = int(np.ceil((height_max - height_min)/step)) + 1
    table_ht = np.linspace(height_min, height_max, n_points)

    # Get atmosphere densities from the MSIS model
    table_log10_rho = np.log10(getAtmDensity(lat, lon, table_ht, jd))

    # Return lists of floats, which are JSON serializable and fast to index in the simulation
    return [float(ht) for ht in table_ht], [float(lr) for lr in table_log10_rho]



def atmDensityTableInfo(lat, lon, jd, height_min=ATM_TABLE_HT_MIN, height_max=ATM_TABLE_HT_MAX, \
    step=ATM_TABLE_STEP):
    """ Describe the atmosphere model and inputs of a density table, to be saved with the simulation
        constants so that the atmosphere of a fit can be reproduced.

    Arguments:
        lat: [float] Latitude in radians.
        lon: [float] Longitude in radians.
        jd: [float] Julian date.

    Keyword arguments:
        height_min: [float] Bottom of the table (m). ATM_TABLE_HT_MIN by default.
        height_max: [float] Top of the table (m). ATM_TABLE_HT_MAX by default.
        step: [float] Maximum height step of the table (m). ATM_TABLE_STEP by default.

    Return:
        [dict] Model, location (deg), Julian date, height range and step (m), and the solar and geomagnetic
            activity given to the model and its source.
    """

    # Take the date given on the command line, if there was one
    if MSIS_JD is not None:
        jd = MSIS_JD

    f107, f107a, ap, sw_source = spaceWeather(jd)

    return {'model': msisModelName(), 'lat': float(np.degrees(lat)), 'lon': float(np.degrees(lon)), \
        'jd': float(jd), 'ht_min': float(height_min), 'ht_max': float(height_max), 'step': float(step), \
        'f107a': float(f107a), 'f107': float(f107), 'ap': float(ap), 'space_weather': sw_source}



def atmDensTable(ht, table_ht, table_log10_rho):
    """ Compute the atmosphere density from a tabulated profile, interpolating log10 of the density linearly
        in height, which is an exponential atmosphere between the table points. Outside the table, the end
        segments are extrapolated. Works on scalars and numpy arrays.

    Arguments:
        ht: [float or ndarray] Height above sea level (m).
        table_ht: [list] Table heights (m), ascending.
        table_log10_rho: [list] log10 of the densities (kg/m^3) at the table heights.

    Return:
        atm_dens: [float or ndarray] Atmosphere neutral mass density in kg/m^3.
    """

    table_ht = np.asarray(table_ht, dtype=np.float64)
    table_log10_rho = np.asarray(table_log10_rho, dtype=np.float64)

    # Find the table segment of every height, using the end segments outside the table
    i = np.searchsorted(table_ht, ht, side='right') - 1
    i = np.clip(i, 0, len(table_ht) - 2)

    frac = (ht - table_ht[i])/(table_ht[i + 1] - table_ht[i])

    return 10**(table_log10_rho[i] + frac*(table_log10_rho[i + 1] - table_log10_rho[i]))



def atmTableSegments(table_ht, table_log10_rho):
    """ Compute the exponential segments of a density table.

    Arguments:
        table_ht: [list] Table heights (m), ascending.
        table_log10_rho: [list] log10 of the densities (kg/m^3) at the table heights.

    Return:
        (table_ht, rho, scale, column_cum):
            - table_ht: [ndarray] Table heights (m).
            - rho: [ndarray] Densities at the table heights (kg/m^3).
            - scale: [ndarray] Exponential scale of every segment (1/m).
            - column_cum: [ndarray] Air column from the bottom of the table up to every table point (kg/m^2).
    """

    table_ht = np.asarray(table_ht, dtype=np.float64)
    ln_rho = np.log(10)*np.asarray(table_log10_rho, dtype=np.float64)
    rho = np.exp(ln_rho)

    scale = np.diff(ln_rho)/np.diff(table_ht)
    column_cum = np.concatenate([[0.0], np.cumsum(expSegmentIntegral(rho[:-1], scale, np.diff(table_ht)))])

    return table_ht, rho, scale, column_cum



def atmTableColumnFromBottom(ht, segments):
    """ Compute the air column from the bottom of the table up to the given height, negative below the
        table.

    Arguments:
        ht: [float or ndarray] Height (m).
        segments: [tuple] Segments of the table from atmTableSegments().

    Return:
        [float or ndarray] Air column (kg/m^2).
    """

    table_ht, rho, scale, column_cum = segments

    i = np.searchsorted(table_ht, ht, side='right') - 1
    i = np.clip(i, 0, len(table_ht) - 2)

    return column_cum[i] + expSegmentIntegral(rho[i], scale[i], ht - table_ht[i])



def atmColumnTable(ht_low, ht_high, table_ht, table_log10_rho):
    """ Compute the air mass column between two heights from a tabulated density profile, the integral of
        atmDensTable() over the height. The profile is exponential between the table points, so the integral
        is exact, and the end segments are extrapolated outside the table.

    Arguments:
        ht_low: [float or ndarray] Lower height (m).
        ht_high: [float or ndarray] Upper height (m).
        table_ht: [list] Table heights (m), ascending.
        table_log10_rho: [list] log10 of the densities (kg/m^3) at the table heights.

    Return:
        column: [float or ndarray] Air mass column in kg/m^2 (negative if ht_low is above ht_high).
    """

    segments = atmTableSegments(table_ht, table_log10_rho)

    return atmTableColumnFromBottom(ht_high, segments) - atmTableColumnFromBottom(ht_low, segments)



class AtmDensityTableInterp(object):
    def __init__(self, table_ht, table_log10_rho):
        """ Callable atmosphere density from a tabulated profile (see atmDensTable()), which also gives the
            exact air column between two heights. Use it in place of a density function when the code
            integrates the density, as the exact column is faster and more accurate than quadrature.

        Arguments:
            table_ht: [list] Table heights (m), ascending.
            table_log10_rho: [list] log10 of the densities (kg/m^3) at the table heights.
        """

        self.table_ht = np.asarray(table_ht, dtype=np.float64)
        self.table_log10_rho = np.asarray(table_log10_rho, dtype=np.float64)

        self.segments = atmTableSegments(self.table_ht, self.table_log10_rho)


    def __call__(self, ht):
        """ Compute the atmosphere density (kg/m^3) at the given heights (m). """

        return atmDensTable(ht, self.table_ht, self.table_log10_rho)


    def column(self, ht_low, ht_high):
        """ Compute the air mass column (kg/m^2) between the given heights (m), see atmColumnTable(). """

        return atmTableColumnFromBottom(ht_high, self.segments) \
            - atmTableColumnFromBottom(ht_low, self.segments)


    



def getMSISVariable(lat, lon, height, jd, variable):
    """ For the given heights, returns one of the variables computed by the MSIS model. The model version
        is given by MSIS_VERSION, see setAtmosphere().

    More info: https://swxtrec.github.io/pymsis/

    Arguments:
        lat: [float or ndarray] Latitude in radians.
        lon: [float or ndarray] Longitude in radians.
        height: [float or ndarray] Height in meters.
        jd: [float or ndarray] Julian date, of all the points or of every point. Ignored if a date was set
            using setAtmosphere().
        variable: [Variable] Model output to return, e.g. Variable.MASS_DENSITY or Variable.TEMPERATURE.
            The model also gives the number densities of N2, O2, O, He, H, Ar, N, anomalous oxygen and NO.

    Return:
        [float or ndarray] The requested variable, in SI units.

    """

    # Take the date given on the command line, if there was one
    if MSIS_JD is not None:
        jd = MSIS_JD

    # Broadcast the inputs to a common shape, so that pymsis evaluates them point by point
    lat, lon, height, jd = np.broadcast_arrays(np.degrees(lat), np.degrees(lon), height, jd)

    # Convert every distinct Julian date to a date once
    jd_unique, jd_index = np.unique(jd.ravel(), return_inverse=True)
    dt_unique = np.array([np.datetime64(jd2Date(jd_value, dt_obj=True)) for jd_value in jd_unique])
    dt_arr = dt_unique[jd_index.ravel()]

    # Take the measured solar and geomagnetic activity of every distinct date, or the constant values
    sw_unique = np.array([spaceWeather(jd_value) for jd_value in jd_unique], dtype=object)
    if all(sw[3] == 'real' for sw in sw_unique):
        f107_arr = f107a_arr = ap_arr = None
    else:
        f107_arr = np.full(lat.size, CONSTANT_F107)
        f107a_arr = np.full(lat.size, CONSTANT_F107A)
        ap_arr = np.full((lat.size, 7), CONSTANT_AP)

    # Take the requested variable out of the 11 that the model returns, giving all inputs the same length
    #   so that pymsis returns one row per point instead of a grid
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        values = calculate(dt_arr, lon.ravel(), lat.ravel(), height.ravel()/1000, f107_arr, f107a_arr, \
            ap_arr, version=MSIS_VERSION)[:, variable].astype(np.float64)

    # Return a scalar if only scalars were given
    if lat.ndim == 0:
        return float(values[0])

    return values.reshape(lat.shape)



def getAtmDensity(lat, lon, height, jd):
    """ For the given heights, returns the atmosphere mass density in kg/m^3. See getMSISVariable(). """

    return getMSISVariable(lat, lon, height, jd, Variable.MASS_DENSITY)



def getAtmTemperature(lat, lon, height, jd):
    """ For the given heights, returns the neutral atmosphere temperature in K. See getMSISVariable(). """

    return getMSISVariable(lat, lon, height, jd, Variable.TEMPERATURE)



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

    plt.semilogx(atm_densities, heights/1000, zorder=3, label="MSIS " + getMSISVersion())


    # Fit the 6th order poly model
    dens_co = fitAtmPoly(np.radians(lat), np.radians(lon), 1000*height_min, 1000*height_max, jd)

    print(dens_co)

    # Plot the fitted poly model
    plt.semilogx(atmDensPoly(heights, dens_co), heights/1000, label="Poly fit")

    # Tabulate the density and plot the interpolated table
    table_ht, table_log10_rho = getAtmDensityTable(np.radians(lat), np.radians(lon), 1000*height_min, \
        1000*height_max, jd)
    plt.semilogx(atmDensTable(heights, table_ht, table_log10_rho), heights/1000, linestyle='dashed', \
        label="Table")

    plt.legend()

    plt.xlabel('Density (kg/m^3)')
    plt.ylabel('Height (km)')

    plt.grid()

    plt.title('MSIS ' + getMSISVersion())

    # plt.savefig('atm_dens.png', dpi=300)

    plt.show()