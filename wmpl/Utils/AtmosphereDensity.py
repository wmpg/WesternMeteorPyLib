""" Plots NRL MSISE atmosphere density model. """

from __future__ import print_function, division, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

from wmpl.PythonNRLMSISE00.nrlmsise_00_header import *
from wmpl.PythonNRLMSISE00.nrlmsise_00 import *
from wmpl.Utils.TrajConversions import jd2Date, jd2LST



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
    """ Compute the atmosphere density using a 6th degree polynomial (7 coefficients) in log10 of the density.
        This is used in the ablation simulation for faster execution. 

        Note that the polynomial deviates from the MSIS model by up to tens of percent at 80 - 120 km when it is
        fitted over a wide height range (e.g. 14 - 180 km for fireballs), use the tabulated profile from
        getAtmDensityTable() instead when accuracy matters.

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

    # Get atmosphere densities from NRLMSISE-00 (use log values for the fit)
    atm_densities = np.array([getAtmDensity(lat, lon, ht, jd) for ht in height_arr])
    atm_densities_log = np.log10(atm_densities)


    def atmDensPolyLog(height_arr, *dens_co):
        return np.log10(atmDensPoly(height_arr, dens_co))

    # Fit the 6th degree polynomial
    dens_co, _ = scipy.optimize.curve_fit(atmDensPolyLog, height_arr, atm_densities_log, \
        p0=np.zeros(7), maxfev=10000)

    return dens_co



def getAtmDensityTable(lat, lon, height_min, height_max, jd, step=500):
    """ Tabulate the log10 of the atmosphere mass density of the MSIS model (as given by getAtmDensity) at the
        given location and time on a uniform height grid. The ablation simulation interpolates log10(density) 
        linearly between the table points (see atmDensTable), which with the default 500 m step reproduces the 
        model to better than 0.5% in density and 0.02% in the air column above any height, between 0 and 
        180 km (measured for NRLMSISE-00 and NRLMSIS 2.0 and 2.1).

    Arguments:
        lat: [float] Latitude in radians.
        lon: [float] Longitude in radians.
        height_min: [float] Minimum height in meters.
        height_max: [float] Maximum height in meters.
        jd: [float] Julian date.

    Keyword arguments:
        step: [float] Maximum height step of the table in meters. The range is split into equal steps no 
            larger than this. 500 m by default.

    Return:
        (table_ht, table_log10_rho):
            - table_ht: [list] Table heights in meters, ascending, from height_min to height_max.
            - table_log10_rho: [list] log10 of the density (kg/m^3) at the table heights.
    """

    if height_max <= height_min:
        raise ValueError("height_max ({:.1f} m) must be above height_min ({:.1f} m)".format(height_max, 
            height_min))

    # Generate a height array with equal steps no larger than the requested step
    n_points = int(np.ceil((height_max - height_min)/step)) + 1
    table_ht = np.linspace(height_min, height_max, n_points)

    # Get atmosphere densities from the MSIS model
    table_log10_rho = np.log10([getAtmDensity(lat, lon, ht, jd) for ht in table_ht])

    # Return plain lists of floats, which are JSON serializable and fast to index in the simulation
    return [float(ht) for ht in table_ht], [float(lr) for lr in table_log10_rho]



def atmDensTable(ht, table_ht, table_log10_rho):
    """ Compute the atmosphere density from a tabulated profile, interpolating log10(density) linearly in 
        height (i.e. an exponential atmosphere between the table points). Outside the table, the first/last
        segment is extrapolated. Works on scalars and numpy arrays.

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



def _expSegmentIntegral(rho_0, scale, dh):
    """ Integral of rho_0*exp(scale*x) for x from 0 to dh, stable when scale*dh is near 0. """

    x = np.asarray(scale*dh, dtype=np.float64)
    x_safe = np.where(x == 0, 1.0, x)

    return rho_0*dh*np.where(x == 0, 1.0, np.expm1(x_safe)/x_safe)



def _atmTableSegments(table_ht, table_log10_rho):
    """ Exponential segments of a density table: heights, densities and exponential scale (1/m) of every
        segment, and the air column (kg/m^2) from the bottom of the table up to every table point.
    """

    table_ht = np.asarray(table_ht, dtype=np.float64)
    ln_rho = np.log(10)*np.asarray(table_log10_rho, dtype=np.float64)
    rho = np.exp(ln_rho)

    scale = np.diff(ln_rho)/np.diff(table_ht)
    column_cum = np.concatenate([[0.0], np.cumsum(_expSegmentIntegral(rho[:-1], scale, np.diff(table_ht)))])

    return table_ht, rho, scale, column_cum



def _atmTableColumnFromBottom(ht, segments):
    """ Air column (kg/m^2) from the bottom of the table up to the given height (negative below the table).
    """

    table_ht, rho, scale, column_cum = segments

    i = np.searchsorted(table_ht, ht, side='right') - 1
    i = np.clip(i, 0, len(table_ht) - 2)

    return column_cum[i] + _expSegmentIntegral(rho[i], scale[i], ht - table_ht[i])



def atmColumnTable(ht_low, ht_high, table_ht, table_log10_rho):
    """ Air mass column (kg/m^2) between two heights from a tabulated density profile, i.e. the integral of
        atmDensTable over the height. The profile is exponential between the table points, so the integral is
        exact (no numerical quadrature), and the end segments are extrapolated outside the table.

    Arguments:
        ht_low: [float or ndarray] Lower height (m).
        ht_high: [float or ndarray] Upper height (m).
        table_ht: [list] Table heights (m), ascending.
        table_log10_rho: [list] log10 of the densities (kg/m^3) at the table heights.

    Return:
        column: [float or ndarray] Air mass column in kg/m^2 (negative if ht_low is above ht_high).
    """

    segments = _atmTableSegments(table_ht, table_log10_rho)

    return _atmTableColumnFromBottom(ht_high, segments) - _atmTableColumnFromBottom(ht_low, segments)



class AtmDensityTableInterp(object):
    def __init__(self, table_ht, table_log10_rho):
        """ Callable atmosphere density from a tabulated profile (see atmDensTable), which also gives the exact
            air column between two heights. Use it in place of a density function when the code integrates 
            the density, as the exact column is much faster and more accurate than numerical quadrature of the
            piecewise profile.

        Arguments:
            table_ht: [list] Table heights (m), ascending.
            table_log10_rho: [list] log10 of the densities (kg/m^3) at the table heights.
        """

        self.table_ht = np.asarray(table_ht, dtype=np.float64)
        self.table_log10_rho = np.asarray(table_log10_rho, dtype=np.float64)

        self.segments = _atmTableSegments(self.table_ht, self.table_log10_rho)


    def __call__(self, ht):
        """ Atmosphere density (kg/m^3) at the given height(s) in meters. """

        return atmDensTable(ht, self.table_ht, self.table_log10_rho)


    def column(self, ht_low, ht_high):
        """ Air mass column (kg/m^2) between the given heights in meters (see atmColumnTable). """

        return _atmTableColumnFromBottom(ht_high, self.segments) \
            - _atmTableColumnFromBottom(ht_low, self.segments)


    


def getAtmDensity(lat, lon, height, jd):
    """ For the given heights, returns the atmospheric density from NRLMSISE-00 model. 
    
    More info: https://github.com/magnific0/nrlmsise-00/blob/master/nrlmsise-00.h

    Arguments:
        lat: [float] Latitude in radians.
        lon: [float] Longitude in radians.
        height: [float] Height in meters.
        jd: [float] Julian date.

    Return:
        [float] Atmosphere density in kg/m^3.

    """


    # Init the input array
    inp = nrlmsise_input()


    # Convert the given Julian date to datetime
    dt = jd2Date(jd, dt_obj=True)

    # Get the day of year
    doy = dt.timetuple().tm_yday

    # Get the second in day
    midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    sec = (dt - midnight).seconds

    # Calculate the Local sidreal time (degrees)
    lst, _ = jd2LST(jd, np.degrees(lon))


    ### INPUT PARAMETERS ###
    ##########################################################################################################
    # Set year (no effect)
    inp.year = 0

    # Day of year
    inp.doy = doy

    # Seconds in a day
    inp.sec = sec

    # Altitude in kilometers
    inp.alt = height/1000.0

    # Geodetic latitude (deg)
    inp.g_lat = np.degrees(lat)

    # Geodetic longitude (deg)
    inp.g_long = np.degrees(lon)

    # Local apparent solar time (hours)
    inp.lst = lst/15


    # f107, f107A, and ap effects are neither large nor well established below 80 km and these parameters 
    # should be set to 150., 150., and 4. respectively.

    # 81 day average of 10.7 cm radio flux (centered on DOY)
    inp.f107A = 150

    # Daily 10.7 cm radio flux for previous day
    inp.f107 = 150

    # Magnetic index (daily)
    inp.ap = 4

    ##########################################################################################################


    # Init the flags array
    flags = nrlmsise_flags()

    # Set output in kilograms and meters
    flags.switches[0] = 1

    # Set all switches to ON
    for i in range(1, 24):
        flags.switches[i] = 1

    
    # Array containing the following magnetic values:
    #   0 : daily AP
    #   1 : 3 hr AP index for current time
    #   2 : 3 hr AP index for 3 hrs before current time
    #   3 : 3 hr AP index for 6 hrs before current time
    #   4 : 3 hr AP index for 9 hrs before current time
    #   5 : Average of eight 3 hr AP indices from 12 to 33 hrs prior to current time
    #   6 : Average of eight 3 hr AP indices from 36 to 57 hrs prior to current time 
    aph = ap_array()

    # Set all AP indices to 100
    for i in range(7):
        aph.a[i] = 100


    # Init the output array
    # OUTPUT VARIABLES:
    #     d[0] - HE NUMBER DENSITY(CM-3)
    #     d[1] - O NUMBER DENSITY(CM-3)
    #     d[2] - N2 NUMBER DENSITY(CM-3)
    #     d[3] - O2 NUMBER DENSITY(CM-3)
    #     d[4] - AR NUMBER DENSITY(CM-3)                       
    #     d[5] - TOTAL MASS DENSITY(GM/CM3) [includes d[8] in td7d]
    #     d[6] - H NUMBER DENSITY(CM-3)
    #     d[7] - N NUMBER DENSITY(CM-3)
    #     d[8] - Anomalous oxygen NUMBER DENSITY(CM-3)
    #     t[0] - EXOSPHERIC TEMPERATURE
    #     t[1] - TEMPERATURE AT ALT
    out = nrlmsise_output()


    # Evaluate the atmosphere with the given parameters
    gtd7(inp, flags, out)


    # Get the total mass density
    atm_density = out.d[5]

    return atm_density



getAtmDensity_vect = np.vectorize(getAtmDensity, excluded=['jd'])




if __name__ == "__main__":

    import datetime
    from wmpl.Utils.TrajConversions import datetime2JD
    
    lat = 44.327234
    lon = -81.372350
    jd = datetime2JD(datetime.datetime.now(datetime.timezone.utc))

    # Height range (km)
    height_min = 20
    height_max = 180

    # Density evaluation heights (m)
    heights = np.linspace(height_min, height_max, 100)*1000

    atm_densities = []
    for height in heights:
        atm_density = getAtmDensity(np.radians(lat), np.radians(lon), height, jd)
        atm_densities.append(atm_density)


    plt.semilogx(atm_densities, heights/1000, zorder=3, label="NRLMSISE-00")


    # Fit the 6th order poly model
    dens_co = fitAtmPoly(np.radians(lat), np.radians(lon), 1000*height_min, 1000*height_max, jd)

    print(dens_co)

    # Plot the fitted poly model
    plt.semilogx(atmDensPoly(heights, dens_co), heights/1000, label="Poly fit")

    # Tabulate the density and plot the interpolated table
    table_ht, table_log10_rho = getAtmDensityTable(np.radians(lat), np.radians(lon), 1000*height_min, 
        1000*height_max, jd)
    plt.semilogx(atmDensTable(heights, table_ht, table_log10_rho), heights/1000, linestyle='dashed', 
        label="Table")

    plt.legend()

    plt.xlabel('Density (kg/m^3)')
    plt.ylabel('Height (km)')

    plt.grid()

    plt.title('NRLMSISE-00')

    # plt.savefig('atm_dens.png', dpi=300)

    plt.show()