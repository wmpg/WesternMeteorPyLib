""" Plots NRL MSISE atmosphere density model. """

from __future__ import print_function, division, absolute_import

import numpy as np
import matplotlib.pyplot as plt

from wmpl.PythonNRLMSISE00.nrlmsise_00_header import *
from wmpl.PythonNRLMSISE00.nrlmsise_00 import *
from wmpl.Utils.TrajConversions import jd2Date, jd2LST



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
    jd = datetime2JD(datetime.datetime.now())

    # Density evaluation heights (m)
    heights = np.linspace(70, 120, 100)*1000

    atm_densities = []
    for height in heights:
        atm_density = getAtmDensity(np.radians(lat), np.radians(lon), height, jd)
        atm_densities.append(atm_density)


    plt.semilogx(atm_densities, heights/1000, zorder=3)

    plt.xlabel('Density (kg/m^3)')
    plt.ylabel('Height (km)')

    plt.xlim(xmin=0)

    plt.grid()

    plt.title('NRLMSISE-00')

    # plt.savefig('atm_dens.png', dpi=300)

    plt.show()