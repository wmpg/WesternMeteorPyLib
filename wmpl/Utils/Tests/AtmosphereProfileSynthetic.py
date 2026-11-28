""" Synthetic atmosphere profiles for the tests of --atm_profile, from the MSIS density at the end of the
    Winchcombe fireball, written in the CSV format OpenDarkflight writes for model profiles.
"""

from __future__ import print_function, division, absolute_import

import numpy as np

from wmpl.Utils.AtmosphereDensity import loadAtmosphereProfile, getAtmDensity


# End of the Winchcombe fireball (2021-02-28)
LAT = 51.94011
LON = -2.09633
JD = 2459274.412778

# Air 9.4% denser than NRLMSIS 2.1, as in the ERA5 reanalysis there
DENSER = 1.094

# Heights of an ERA5 profile as OpenDarkflight writes it (m)
PROFILE_HT = np.arange(300.0, 46300.0, 100.0)

# Header of the CSV files OpenDarkflight writes for model profiles
PROFILE_HEADER = "height,temperature,pressure,relative_humidity,wind_horizontal,wind_direction,wind_east," \
    "wind_north,wind_up,density"



def writeProfile(file_path, ht_arr, dens_arr):
    """ Write an atmosphere profile in the CSV format OpenDarkflight writes for model profiles.

    Arguments:
        file_path: [str] Path to the CSV file.
        ht_arr: [ndarray] Heights (m).
        dens_arr: [ndarray] Air densities (kg/m^3).

    Return:
        None
    """

    with open(file_path, 'w') as f:

        f.write(PROFILE_HEADER + "\n")

        for ht, dens in zip(ht_arr, dens_arr):
            f.write("{:.1f},220.0,1000.0,0.0,10.0,270.0,10.0,0.0,0.0,{:.10e}\n".format(ht, dens))



def msisProfile(tmp_path, scale):
    """ Write and load a profile of the MSIS density at the end of Winchcombe, multiplied by a factor.

    Arguments:
        tmp_path: [Path] Directory for the file.
        scale: [float] Factor applied to the MSIS density.

    Return:
        (table_ht, table_log10_rho): [tuple] Density table of the profile.
    """

    file_path = str(tmp_path/"ModelWindProfile_{:.3f}.csv".format(scale))
    writeProfile(file_path, PROFILE_HT, \
        scale*getAtmDensity(np.radians(LAT), np.radians(LON), PROFILE_HT, JD))

    return loadAtmosphereProfile(file_path)
