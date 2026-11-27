""" Tests for the air density of MSIS at a Julian date for every point, which the vectorized function of
    NRLMSISE-00 accepted, e.g. for the samples of a trajectory at their own times.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_AtmosphereDensityDates.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np

from wmpl.Utils.AtmosphereDensity import getAtmDensity, getAtmDensity_vect
from wmpl.Utils.Physics import dynamicPressure



def testAJulianDateForEveryPoint():
    """ Check that a Julian date for every point gives the density of every point at its own date, and that
        one date for all points gives the same as that date repeated.
    """

    lat = np.radians(np.array([45.0, 45.0, -30.0]))
    lon = np.radians(np.array([10.0, 10.0, 120.0]))
    height = np.array([80000.0, 80000.0, 40000.0])

    # The same place and height in winter and in summer, which differ in density
    jd = np.array([2460310.5, 2460490.5, 2460400.5])

    density = getAtmDensity_vect(lat, lon, height, jd)
    expected = [getAtmDensity(lat[i], lon[i], height[i], jd[i]) for i in range(len(jd))]

    assert np.allclose(density, expected, rtol=1e-12)
    assert abs(density[0]/density[1] - 1) > 0.05

    assert np.allclose(getAtmDensity_vect(lat, lon, height, jd[0]), \
        getAtmDensity_vect(lat, lon, height, np.full(len(jd), jd[0])), rtol=1e-12)



def testTheDynamicPressureTakesADateForEveryPoint():
    """ Check that the dynamic pressure takes a Julian date for every point, as for the samples of a
        trajectory.
    """

    lat = np.radians(np.array([45.0, 45.1]))
    lon = np.radians(np.array([10.0, 10.1]))
    height = np.array([60000.0, 59000.0])
    jd = 2460310.5 + np.array([0.0, 1.0])/86400
    velocity = np.array([20000.0, 19900.0])

    pressure = dynamicPressure(lat, lon, height, jd, velocity)

    assert np.allclose(pressure, getAtmDensity_vect(lat, lon, height, jd)*velocity**2, rtol=1e-12)
