""" Dynamical classification of a single orbit.

These criteria take one orbit and return a dynamical class, rather than comparing two orbits, so
they are kept apart from the dissimilarity criteria in wmpl.Utils.Dcriteria.

All functions accept scalars or numpy arrays and take angles in radians.
"""

from __future__ import print_function, division, absolute_import

import numpy as np


# Semi-major axis of Jupiter [AU]
A_JUPITER = 5.20336

# Inclination above which an orbit is taken to be cometary in the two-parameter criteria of
#   Jopek & Williams (2013) [rad]
JW_INCL_LIMIT = np.radians(75.0)

# Cometary limits of the one-parameter criteria, as adopted by Jopek & Williams (2013)
JW_APHELION_LIMIT = 4.6
JW_KRESAK_P_LIMIT = 2.5


def calcTisserand(a, e, i, a_planet=A_JUPITER):
    """ Calculate the Tisserand parameter of an orbit with respect to a perturbing planet.

        The parameter is very nearly conserved during a close encounter with the planet, which
        makes it the standard discriminant between asteroidal orbits, which have T above about 3
        with respect to Jupiter, and Jupiter-family cometary orbits, which have T below 3.

        Reference: Whipple (1954), AJ 59, 201.

    Arguments:
        a: [float] semi-major axis of the orbit (AU)
        e: [float] num. eccentricity of the orbit
        i: [float] inclination of the orbit (rad)

    Keyword arguments:
        a_planet: [float] semi-major axis of the perturbing planet (AU). Default Jupiter.

    Return:
        [float] Tisserand parameter
    """

    return a_planet/a + 2*np.cos(i)*np.sqrt((a/a_planet)*(1.0 - e**2))


def calcKresakK(a, e):
    """ Calculate the Kresak K criterion of an orbit.

        K is positive for cometary orbits and negative for asteroidal ones.

        Reference: Kresak (1967). The form used here, and the sign of the cometary limit, follow
        Jopek & Williams (2013), MNRAS 430, 2377, eq. 12, doi:10.1093/mnras/stt057.

    Arguments:
        a: [float] semi-major axis of the orbit (AU)
        e: [float] num. eccentricity of the orbit

    Return:
        [float] K value
    """

    return np.log10(a*(1.0 + e)/(1.0 - e)) - 1.0


def calcKresakP(a, e):
    """ Calculate the Kresak P criterion of an orbit.

        P exceeds 2.5 yr for cometary orbits.

        Reference: Kresak (1969). The form used here, and the cometary limit, follow Jopek &
        Williams (2013), MNRAS 430, 2377, eq. 11, doi:10.1093/mnras/stt057.

    Arguments:
        a: [float] semi-major axis of the orbit (AU)
        e: [float] num. eccentricity of the orbit

    Return:
        [float] P value (yr)
    """

    return a**1.5*e


def calcAphelionDistance(a, e):
    """ Calculate the aphelion distance of an orbit.

        The aphelion distance is the Q of the two-parameter Q-i criterion, and exceeds 4.6 AU for
        cometary orbits.

    Arguments:
        a: [float] semi-major axis of the orbit (AU)
        e: [float] num. eccentricity of the orbit

    Return:
        [float] aphelion distance (AU)
    """

    return a*(1.0 + e)


def isCometaryQi(a, e, i):
    """ Classify an orbit as cometary or asteroidal using the two-parameter Q-i criterion.

        An orbit counts as cometary if its aphelion reaches beyond 4.6 AU, which places it under
        the dynamical control of Jupiter, or if it is inclined by more than 75 deg, which no
        collisionally produced asteroid fragment is expected to be. Of the five two-parameter
        criteria examined, Q-i and E-i were found to be the most reliable.

        Reference: Jopek & Williams (2013), MNRAS 430, 2377, eq. 8, doi:10.1093/mnras/stt057;
        Williams & Jopek (2014).

    Arguments:
        a: [float] semi-major axis of the orbit (AU)
        e: [float] num. eccentricity of the orbit
        i: [float] inclination of the orbit (rad)

    Return:
        [bool] True if the orbit is cometary
    """

    return (calcAphelionDistance(a, e) > JW_APHELION_LIMIT) | (np.asarray(i) > JW_INCL_LIMIT)


def isCometaryKi(a, e, i):
    """ Classify an orbit as cometary or asteroidal using the two-parameter K-i criterion.

        Reference: Jopek & Williams (2013), MNRAS 430, 2377, eq. 12, doi:10.1093/mnras/stt057.

    Arguments:
        a: [float] semi-major axis of the orbit (AU)
        e: [float] num. eccentricity of the orbit
        i: [float] inclination of the orbit (rad)

    Return:
        [bool] True if the orbit is cometary
    """

    return (calcKresakK(a, e) > 0.0) | (np.asarray(i) > JW_INCL_LIMIT)


def isCometaryPi(a, e, i):
    """ Classify an orbit as cometary or asteroidal using the two-parameter P-i criterion.

        Reference: Jopek & Williams (2013), MNRAS 430, 2377, eq. 11, doi:10.1093/mnras/stt057.

    Arguments:
        a: [float] semi-major axis of the orbit (AU)
        e: [float] num. eccentricity of the orbit
        i: [float] inclination of the orbit (rad)

    Return:
        [bool] True if the orbit is cometary
    """

    return (calcKresakP(a, e) > JW_KRESAK_P_LIMIT) | (np.asarray(i) > JW_INCL_LIMIT)
