""" Tests for the poles in the coordinate conversions of wmpl.Utils.TrajConversions.

equatorialCoordPrecession, raDec2Ecliptic and ecliptic2RaDec compute the declination (or the ecliptic
latitude) with an arcsine or an arccosine, which already returns it in [-pi/2, pi/2]. They then
"wrapped" it with (x + pi/2) % pi - pi/2, which does nothing inside the range but turns +pi/2 into
-pi/2, so a north pole came out as a south pole. The tests check both poles, that nothing else
changes, and that the conversions invert each other.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_TrajConversions.py -v

or standalone (no pytest required):
    python -m wmpl.Utils.Tests.test_TrajConversions
"""

import numpy as np

from wmpl.Utils.Earth import calcTrueObliquity
from wmpl.Utils.TrajConversions import J2000_JD, ecliptic2RaDec, equatorialCoordPrecession, \
    raDec2Ecliptic


JD = 2459274.41  # 2021-02-28


def _separationArcsec(ra1, dec1, ra2, dec2):
    """ Angle between two directions, from the distance between their unit vectors (an arccos of
    the dot product cannot resolve less than ~3 mas). """

    def vect(ra, dec):
        return np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])

    chord = np.linalg.norm(vect(ra1, dec1) - vect(ra2, dec2), axis=0)

    return np.degrees(2*np.arcsin(chord/2))*3600


def testPrecessionKeepsTheNorthPole():

    jd = J2000_JD.days

    # No precession: the poles stay where they are
    assert np.isclose(equatorialCoordPrecession(jd, jd, 0.0, np.pi/2)[1], np.pi/2, atol=1e-12)
    assert np.isclose(equatorialCoordPrecession(jd, jd, 0.0, -np.pi/2)[1], -np.pi/2, atol=1e-12)

    # 21 years of precession move the pole by ~0.12 deg, not to the other hemisphere
    dec = equatorialCoordPrecession(jd, JD, 0.0, np.pi/2)[1]
    assert np.isclose(np.degrees(dec), 89.883, atol=0.001)


def testEclipticPolesKeepTheirSign():

    eps = calcTrueObliquity(JD)

    # The north ecliptic pole is at RA = 18 h, Dec = 90 deg - eps, the south one at 6 h, eps - 90 deg
    assert np.isclose(raDec2Ecliptic(JD, 1.5*np.pi, np.pi/2 - eps)[1], np.pi/2, atol=1e-7)
    assert np.isclose(raDec2Ecliptic(JD, 0.5*np.pi, eps - np.pi/2)[1], -np.pi/2, atol=1e-7)

    # The north celestial pole is at L = 90 deg, B = 90 deg - eps
    assert np.isclose(ecliptic2RaDec(JD, np.pi/2, np.pi/2 - eps)[1], np.pi/2, atol=1e-7)
    assert np.isclose(ecliptic2RaDec(JD, 1.5*np.pi, eps - np.pi/2)[1], -np.pi/2, atol=1e-7)

    # Also in arrays
    lat = raDec2Ecliptic(JD, np.array([1.5*np.pi, 0.3]), np.array([np.pi/2 - eps, 0.2]))[1]
    assert np.isclose(lat[0], np.pi/2, atol=1e-7)


def testConversionsInvertEachOtherAndStayFinite():

    rng = np.random.default_rng(3)
    n = 20000
    ra = rng.uniform(0, 2*np.pi, n)
    dec = np.arcsin(rng.uniform(-1, 1, n))

    # Points right at and next to the poles, where rounding can push an arcsine past +/-1
    ra = np.concatenate([ra, rng.uniform(0, 2*np.pi, 200)])
    dec = np.concatenate([dec, np.pi/2 - np.abs(rng.normal(0, 1e-9, 100)),
                          -np.pi/2 + np.abs(rng.normal(0, 1e-9, 100))])

    L, B = raDec2Ecliptic(JD, ra, dec)
    assert np.all(np.isfinite(L)) and np.all(np.isfinite(B))
    assert np.all(np.abs(B) <= np.pi/2)

    ra2, dec2 = ecliptic2RaDec(JD, L, B)
    assert np.all(np.isfinite(dec2))

    # Angular distance between the original and the converted-back direction. Next to a pole an
    #   arcsine only resolves ~sqrt(machine epsilon), ~3 mas, so those points get a looser bound.
    sep = _separationArcsec(ra, dec, ra2, dec2)
    assert np.all(sep[:n] < 1e-6)
    assert np.all(sep[n:] < 1e-2)

    # Precessing forward and back returns the same direction
    for k in range(0, n, 97):
        ra_p, dec_p = equatorialCoordPrecession(J2000_JD.days, JD, ra[k], dec[k])
        ra_b, dec_b = equatorialCoordPrecession(JD, J2000_JD.days, ra_p, dec_p)
        assert _separationArcsec(ra[k], dec[k], ra_b, dec_b) < 1e-6


if __name__ == "__main__":

    for test in [testPrecessionKeepsTheNorthPole, testEclipticPolesKeepTheirSign,
                 testConversionsInvertEachOtherAndStayFinite]:
        test()
        print("PASS", test.__name__)
