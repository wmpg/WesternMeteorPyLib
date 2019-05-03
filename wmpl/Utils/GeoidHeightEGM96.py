
from __future__ import print_function, division, absolute_import

import os

import numpy as np
import scipy.interpolate


from wmpl.Config import config


def loadEGM96Data(dir_path, file_name):
    """ Load a file with EGM96 data.
    
    EGM96 data source: http://earth-info.nga.mil/GandG/wgs84/gravitymod/egm96/binary/binarygeoid.html
    """

    # Load the geoid heights
    geoid_heights = np.fromfile(os.path.join(dir_path, file_name), \
        dtype=np.int16).byteswap().astype(np.float64)

    # Reshape the data to 15 min grid
    geoid_heights = geoid_heights.reshape(721, 1440)

    # Compute the height in meters
    geoid_heights /= 100

    return geoid_heights



def interpolateEGM96Data(geoid_heights):
    """ Interpolate geoid heights on a sphere. """

    # Interpolate the data
    lat_points = np.radians(np.linspace(0.25, 179.25, 719))
    lon_points = np.radians(np.linspace(0, 359.75, 1440))

    # Extract pole values
    north_pole_value = geoid_heights[0][0]
    south_pole_value = geoid_heights[-1][0]

    # Remove points on the pole
    geoid_heights = geoid_heights[1:-1]
        
    # Construct an interpolation instance
    geoid_model = scipy.interpolate.RectSphereBivariateSpline(lat_points, lon_points, geoid_heights,
        pole_values=(north_pole_value, south_pole_value))

    return geoid_model



# Load the geoid heights array
GEOID_HEIGHTS = loadEGM96Data(*os.path.split(config.egm96_file))

# Init the interpolated geoid model
GEOID_MODEL = interpolateEGM96Data(GEOID_HEIGHTS)




def mslToWGS84Height(lat, lon, msl_height):
    """ Given the height above sea level (using the EGM96 model), compute the height above the WGS84
        ellipsoid.
    
    Arguments:
        lat: [float] Latitude +N (deg).
        lon: [float] Longitude +E (deg).
        msl_height: [float] Height above sea level (meters).

    Return:
        wgs84_height: [float] Height above the WGS84 ellipsoid.

    """


    # Get the difference between WGS84 and MSL height
    lat_mod = np.radians(90 - lat)
    lon_mod = np.radians(lon%360)
    msl_ht_diff = GEOID_MODEL(lat_mod, lon_mod)[0][0]

    # Compute the WGS84 height
    wgs84_height = msl_height + msl_ht_diff


    return wgs84_height




if __name__ == "__main__":

    import matplotlib.pyplot as plt


    dir_path = '.'
    file_name = "WW15MGH.DAC"


    # Test data
    lat = 43.2640333333
    lon = -80.7721783333
    msl_height = 329.49



    # Compute the WGS84 height
    wgs84_height = mslToWGS84Height(lat, lon, msl_height)



    print('Latitude:', lat)
    print('Longitude', lon)
    print('MSL height (m):', msl_height)
    print('WGS84 height (m):', wgs84_height)

    # plt.imshow(geoid_heights)
    # plt.show()