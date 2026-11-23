""" Tests of the EGM96 geoid heights in wmpl.Utils.GeoidHeightEGM96.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_GeoidHeightEGM96.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np

from wmpl.Utils.GeoidHeightEGM96 import mslToWGS84Height, wgs84toMSLHeight


# Test points of the EGM96 geoid from NGA (latitude and longitude in deg, geoid height in m)
NGA_TEST_POINTS = [
    (38.6281550, 269.7791550, -31.628),
    (-14.6212170, 305.0211140, -2.969),
    (46.8743190, 102.4487290, -43.575),
    (-23.6174460, 133.8747120, 15.871),
    (38.6254730, 359.9995000, 50.066),
    (-0.4667440, 0.0023000, 17.329),
]



def testGeoidHeightsAtNGATestPoints():
    """ Check the geoid heights against the test points of NGA. With the rows of the grid at the wrong
        colatitudes, they were off by up to 2.95 m.
    """

    for lat, lon, geoid_ht in NGA_TEST_POINTS:

        wgs84_ht = mslToWGS84Height(np.radians(lat), np.radians(lon), 0.0)

        print("Lat {:+11.6f}, lon {:11.6f}: geoid height {:8.3f} m, NGA {:8.3f} m".format(lat, lon, wgs84_ht, \
            geoid_ht))

        assert abs(wgs84_ht - geoid_ht) < 0.01



def testHeightConversionsInverse():
    """ Check that the conversion from the height above the ellipsoid to the height above sea level inverts
        the conversion the other way.
    """

    for lat, lon, _ in NGA_TEST_POINTS:

        msl_ht = 1234.5
        wgs84_ht = mslToWGS84Height(np.radians(lat), np.radians(lon), msl_ht)

        assert abs(wgs84toMSLHeight(np.radians(lat), np.radians(lon), wgs84_ht) - msl_ht) < 1e-9
