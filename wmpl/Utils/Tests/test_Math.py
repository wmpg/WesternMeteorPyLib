""" Tests of the functions in wmpl.Utils.Math.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_Math.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np

from wmpl.Utils.Math import medianStddev, circPercentile



def testMedianStddev():
    """ Check the standard deviation of the median against the known values for equal deviations. """

    # The median of one value has the same deviation, and of three about 0.670 of it
    assert np.isclose(medianStddev([10.0]), 10.0, rtol=0.02)
    assert np.isclose(medianStddev([10.0, 10.0, 10.0]), 6.70, rtol=0.02)

    # Values without a valid deviation are taken as exact
    assert medianStddev([None, np.inf]) == 0.0
    assert medianStddev([]) == 0.0



def testCircPercentileOfAnglesFrom0To360():
    """ Check the median of angles from 0 to 360 deg, also across the 0/360 break. """

    assert np.isclose(circPercentile(np.radians([10.0, 20.0, 30.0, 40.0, 50.0]), 50), np.radians(30.0))

    median = circPercentile(np.radians([350.0, 355.0, 0.0, 5.0, 10.0]), 50)
    assert np.isclose(np.cos(median), 1.0)



def testCircPercentileOfAnglesFromMinus180To180():
    """ Check that angles from -180 to 180 deg give the same percentiles as from 0 to 360, also when they
        are more than 180 deg apart and none is above 180 deg, which raised a ValueError (e.g. longitudes
        on both sides of the date line).
    """

    data = np.radians([-20.0, -10.0, 0.0, 10.0, 20.0])
    assert np.isclose(circPercentile(data, 50), circPercentile(data%(2*np.pi), 50))

    data = np.radians([179.8, 179.9, -179.9, -179.8])
    for percentile in [2.5, 50, 97.5]:
        assert np.isclose(circPercentile(data, percentile), circPercentile(data%(2*np.pi), percentile))

    assert 179.8 <= np.degrees(circPercentile(data, 2.5)) <= 180.2




if __name__ == "__main__":

    # Run all tests
    test_list = [testMedianStddev, testCircPercentileOfAnglesFrom0To360, \
        testCircPercentileOfAnglesFromMinus180To180]

    for test in test_list:
        test()
        print("PASS {:s}".format(test.__name__))
