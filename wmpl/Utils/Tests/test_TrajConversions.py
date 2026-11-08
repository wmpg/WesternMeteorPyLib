""" Tests of the declination and the ecliptic latitude at the poles in wmpl.Utils.TrajConversions.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_TrajConversions.py -v

or standalone (no pytest required):
    python -m wmpl.Utils.Tests.test_TrajConversions
"""

from __future__ import print_function, division, absolute_import

import numpy as np

from wmpl.Utils.Earth import calcTrueObliquity
from wmpl.Utils.TrajConversions import J2000_JD, ecliptic2RaDec, equatorialCoordPrecession, \
    raDec2Ecliptic


# Julian date of 2021-02-28
JD = 2459274.41



def _unitVector(ra, dec):
    """ Compute the unit vector of a direction given in equatorial coordinates.

    Arguments:
        ra: [float or ndarray] Right ascension (radians).
        dec: [float or ndarray] Declination (radians).

    Return:
        [ndarray] Unit vector, the first axis are the x, y, z components.
    """

    return np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])



def _separationArcsec(ra1, dec1, ra2, dec2):
    """ Compute the angle between two directions from the chord between their unit vectors. An arccos
        of the dot product cannot resolve less than about 3 mas.

    Arguments:
        ra1: [float or ndarray] Right ascension of the first direction (radians).
        dec1: [float or ndarray] Declination of the first direction (radians).
        ra2: [float or ndarray] Right ascension of the second direction (radians).
        dec2: [float or ndarray] Declination of the second direction (radians).

    Return:
        [float or ndarray] Angle between the directions (arcsec).
    """

    # Compute the chord between the unit vectors
    chord = np.linalg.norm(_unitVector(ra1, dec1) - _unitVector(ra2, dec2), axis=0)

    return np.degrees(2*np.arcsin(chord/2))*3600



def testPrecessionKeepsTheNorthPole():
    """ Check that the precession keeps the celestial poles in their own hemisphere. """

    jd = J2000_JD.days

    # Without precession the poles stay where they are
    assert np.isclose(equatorialCoordPrecession(jd, jd, 0.0, np.pi/2)[1], np.pi/2, atol=1e-12)
    assert np.isclose(equatorialCoordPrecession(jd, jd, 0.0, -np.pi/2)[1], -np.pi/2, atol=1e-12)

    # 21 years of precession move the pole by about 0.12 deg, not to the other hemisphere
    dec = equatorialCoordPrecession(jd, JD, 0.0, np.pi/2)[1]
    assert np.isclose(np.degrees(dec), 89.883, atol=0.001)



def testEclipticPolesKeepTheirSign():
    """ Check that the ecliptic and the celestial poles keep their sign in the ecliptic conversions. """

    eps = calcTrueObliquity(JD)

    # The north ecliptic pole is at RA = 18 h, Dec = 90 deg - eps, the south one at 6 h, eps - 90 deg
    assert np.isclose(raDec2Ecliptic(JD, 1.5*np.pi, np.pi/2 - eps)[1], np.pi/2, atol=1e-7)
    assert np.isclose(raDec2Ecliptic(JD, 0.5*np.pi, eps - np.pi/2)[1], -np.pi/2, atol=1e-7)

    # The north celestial pole is at L = 90 deg, B = 90 deg - eps, the south one at 270 deg, eps - 90 deg
    assert np.isclose(ecliptic2RaDec(JD, np.pi/2, np.pi/2 - eps)[1], np.pi/2, atol=1e-7)
    assert np.isclose(ecliptic2RaDec(JD, 1.5*np.pi, eps - np.pi/2)[1], -np.pi/2, atol=1e-7)

    # Check the north ecliptic pole in an array
    lat = raDec2Ecliptic(JD, np.array([1.5*np.pi, 0.3]), np.array([np.pi/2 - eps, 0.2]))[1]
    assert np.isclose(lat[0], np.pi/2, atol=1e-7)



def testConversionsInvertEachOtherAndStayFinite():
    """ Check that the conversions invert each other and give no NaNs, also next to the poles. """

    # Generate random directions uniformly distributed on the sky
    rng = np.random.default_rng(3)
    n = 20000
    ra = rng.uniform(0, 2*np.pi, n)
    dec = np.arcsin(rng.uniform(-1, 1, n))

    # Add points right at and next to the poles, where rounding can push an arcsine past +/-1
    ra = np.concatenate([ra, rng.uniform(0, 2*np.pi, 200)])
    dec = np.concatenate([dec, np.pi/2 - np.abs(rng.normal(0, 1e-9, 100)), \
        -np.pi/2 + np.abs(rng.normal(0, 1e-9, 100))])

    # Convert to ecliptic coordinates
    L, B = raDec2Ecliptic(JD, ra, dec)
    assert np.all(np.isfinite(L)) and np.all(np.isfinite(B))
    assert np.all(np.abs(B) <= np.pi/2)

    # Convert back to equatorial coordinates
    ra2, dec2 = ecliptic2RaDec(JD, L, B)
    assert np.all(np.isfinite(dec2))

    # Compare the directions (an arcsine next to a pole only resolves about 3 mas, so those points get
    #   a looser bound)
    sep = _separationArcsec(ra, dec, ra2, dec2)
    assert np.all(sep[:n] < 1e-6)
    assert np.all(sep[n:] < 1e-2)

    # Check that precessing forward and back returns the same direction
    for k in range(0, n, 97):
        ra_p, dec_p = equatorialCoordPrecession(J2000_JD.days, JD, ra[k], dec[k])
        ra_b, dec_b = equatorialCoordPrecession(JD, J2000_JD.days, ra_p, dec_p)
        assert _separationArcsec(ra[k], dec[k], ra_b, dec_b) < 1e-6




if __name__ == "__main__":

    # Run all tests
    test_list = [testPrecessionKeepsTheNorthPole, testEclipticPolesKeepTheirSign, \
        testConversionsInvertEachOtherAndStayFinite]

    for test in test_list:
        test()
        print("PASS {:s}".format(test.__name__))
