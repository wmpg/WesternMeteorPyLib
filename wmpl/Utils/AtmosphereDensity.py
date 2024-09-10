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

    # Get atmosphere densities from NRLMSISE-00 (use log values for the fit)
    atm_densities = np.array([getAtmDensity(lat, lon, ht, jd) for ht in height_arr])
    atm_densities_log = np.log10(atm_densities)


    def atmDensPolyLog(height_arr, *dens_co):
        return np.log10(atmDensPoly(height_arr, dens_co))

    # Fit the 7th order polynomial
    dens_co, _ = scipy.optimize.curve_fit(atmDensPolyLog, height_arr, atm_densities_log, \
        p0=np.zeros(7), maxfev=10000)

    return dens_co


    


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
    #   5 : Average of eight 3 hr AP indicies from 12 to 33 hrs prior to current time
    #   6 : Average of eight 3 hr AP indicies from 36 to 57 hrs prior to current time 
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

    plt.legend()

    plt.xlabel('Density (kg/m^3)')
    plt.ylabel('Height (km)')

    plt.grid()

    plt.title('NRLMSISE-00')

    # plt.savefig('atm_dens.png', dpi=300)

    plt.show()