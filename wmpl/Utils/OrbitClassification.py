""" Dynamical classification of a single orbit.

These criteria take one orbit and return a dynamical class, rather than comparing two orbits, so
they are kept apart from the dissimilarity criteria in wmpl.Utils.Dcriteria.

All functions take angles in radians. The single-parameter criteria and the two-parameter
cometary tests accept scalars or numpy arrays; the Tancredi (2014) classifiers return a label and
take scalars only.
"""

from __future__ import print_function, division, absolute_import

import numpy as np


# Semi-major axis of Jupiter [AU]
A_JUPITER = 5.20336

# Inclination above which an orbit is taken to be cometary in the two-parameter criteria of
#   Jopek & Williams (2013) [rad]
JW_INCL_LIMIT = np.radians(75.0)

# Cometary limits of the one-parameter criteria, as adopted by Jopek & Williams (2013)
JW_APHELION_LIMIT = 4.6      # aphelion distance [AU]
JW_KRESAK_P_LIMIT = 2.5      # Kresak P [yr]
JW_ENERGY_LIMIT = -5.28e-5   # orbital energy -k^2/(2a) [AU^2/day^2]

# Square of the Gaussian gravitational constant [AU^3/day^2]
GAUSS_K_SQUARED = 0.01720209895**2

# Semi-major axes of the other giant planets [AU]
A_SATURN = 9.5826
A_URANUS = 19.2018
A_NEPTUNE = 30.0470

# Eccentricity of Jupiter, and the perihelion and aphelion distances it gives [AU]. Tancredi (2014)
#   quotes 4.85 AU for Jupiter's perihelion in the text, 0.1 AU inside the value used here, which
#   moves only objects whose own perihelion falls in that narrow band
E_JUPITER = 0.0489
Q_JUPITER_PERIHELION = A_JUPITER*(1.0 - E_JUPITER)
Q_JUPITER_APHELION = A_JUPITER*(1.0 + E_JUPITER)

# Hill radii of the giant planets [AU], Tancredi (2014) section 2.3
HILL_RADIUS_JUPITER = 0.355
HILL_RADIUS_SATURN = 0.436
HILL_RADIUS_URANUS = 0.469
HILL_RADIUS_NEPTUNE = 0.776

# Tisserand parameter limits of the Tancredi (2014) classification. The upper limit is 3.05 rather
#   than 3 because Jupiter's orbit is not circular and because encounters out to a few Hill radii
#   still perturb an orbit, so objects a little above 3 can still be Jupiter-dominated
TANCREDI_T_LOW = 2.0
TANCREDI_T_HIGH = 3.05

# Mean-motion resonances with Jupiter considered by Tancredi (2014), table 1, as
#   (label, p + q, p, maximum libration in semi-major axis [AU])
TANCREDI_RESONANCES = [
    ('4:1', 4, 1, 0.0075),
    ('3:1', 3, 1, 0.0287),
    ('5:2', 5, 2, 0.0260),
    ('7:3', 7, 3, 0.0215),
    ('2:1', 2, 1, 0.1127),
    ('3:2', 3, 2, 0.1),
    ('4:3', 4, 3, 0.1),
    ('1:1', 1, 1, 0.208),
    ]


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

        An orbit counts as cometary if its aphelion reaches beyond 4.6 AU, which brings it close
        to Jupiter's orbit, or if it is inclined by more than 75 deg. Of the five two-parameter
        criteria the paper examines, Q-i and E-i were the most reliable; E-i is isCometaryEi.

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


def calcOrbitalEnergy(a):
    """ Calculate the orbital energy of an orbit in the units used by Jopek & Williams (2013).

        Reference: Jopek & Williams (2013), MNRAS 430, 2377, eq. 9, doi:10.1093/mnras/stt057.

    Arguments:
        a: [float] semi-major axis of the orbit (AU)

    Return:
        [float] orbital energy (AU^2/day^2)
    """

    return -GAUSS_K_SQUARED/(2*a)


def isCometaryEi(a, e, i):
    """ Classify an orbit as cometary or asteroidal using the two-parameter E-i criterion.

        The eccentricity is not used: the energy depends only on the semi-major axis. It is kept in
        the signature so that the five two-parameter criteria can be called interchangeably.

        Together with Q-i this was the more reliable of the five criteria the paper examines.

        Reference: Jopek & Williams (2013), MNRAS 430, 2377, eq. 9, doi:10.1093/mnras/stt057.

    Arguments:
        a: [float] semi-major axis of the orbit (AU)
        e: [float] num. eccentricity of the orbit, not used
        i: [float] inclination of the orbit (rad)

    Return:
        [bool] True if the orbit is cometary
    """

    return (calcOrbitalEnergy(a) > JW_ENERGY_LIMIT) | (np.asarray(i) > JW_INCL_LIMIT)


def calcHillRadius(a_planet, mass_ratio):
    """ Calculate the Hill radius of a planet.

        Inside this radius the planet's attraction dominates the Sun's tidal pull, so it sets the
        scale on which a close encounter perturbs a heliocentric orbit.

        Reference: Tancredi (2014), Icarus 234, 66, eq. 2, doi:10.1016/j.icarus.2014.02.013.

    Arguments:
        a_planet: [float] semi-major axis of the planet (AU)
        mass_ratio: [float] mass of the planet divided by the mass of the Sun

    Return:
        [float] Hill radius (AU)
    """

    return a_planet*(mass_ratio/(3.0*(1.0 + mass_ratio)))**(1.0/3.0)


def calcResonanceSemiMajorAxis(p, p_plus_q, a_planet=A_JUPITER):
    """ Calculate the semi-major axis at the centre of a mean-motion resonance with a planet.

        A resonance labelled (p + q):p holds when the body completes p + q orbits for every p of
        the planet.

        Reference: Tancredi (2014), Icarus 234, 66, eq. 3, doi:10.1016/j.icarus.2014.02.013.

    Arguments:
        p: [int] second integer of the (p + q):p label
        p_plus_q: [int] first integer of the (p + q):p label

    Keyword arguments:
        a_planet: [float] semi-major axis of the planet (AU). Default Jupiter.

    Return:
        [float] semi-major axis at the centre of the resonance (AU)
    """

    return (float(p)/p_plus_q)**(2.0/3.0)*a_planet


def classifyTancrediComet(a, e, i):
    """ Classify a periodic comet's orbit under the Tancredi (2014) scheme.

        The scheme separates orbits by whether Jupiter can force them: below a Tisserand parameter
        of 2 an encounter is fast and the orbit is driven by secular effects instead, between 2 and
        3.05 encounters are slow and dominate the evolution, and above 3.05 the orbit does not
        reach Jupiter at all.

        Reference: Tancredi (2014), Icarus 234, 66, section 4, doi:10.1016/j.icarus.2014.02.013.

    Arguments:
        a: [float] semi-major axis of the orbit (AU)
        e: [float] num. eccentricity of the orbit
        i: [float] inclination of the orbit (rad)

    Return:
        [str] one of 'halley', 'jupiter family', 'asteroidal orbit', 'centaur', or 'unclassified'
    """

    tisserand = float(calcTisserand(a, e, i))
    q = a*(1.0 - e)

    if (tisserand < TANCREDI_T_LOW) and (a < A_NEPTUNE):
        return 'halley'

    if (TANCREDI_T_LOW < tisserand < TANCREDI_T_HIGH) and (q < Q_JUPITER_APHELION):
        return 'jupiter family'

    if (tisserand > TANCREDI_T_HIGH) and (q < Q_JUPITER_APHELION):
        return 'asteroidal orbit'

    if (tisserand > TANCREDI_T_LOW) and (Q_JUPITER_APHELION < q < A_URANUS):
        return 'centaur'

    return 'unclassified'


def isTancrediResonanceProtected(a, e, moid_jupiter_hill):
    """ Decide whether an orbit sits in a mean-motion resonance with Jupiter that protects it from
        close encounters.

        An orbit can have a small minimum distance to Jupiter's orbit and still never approach the
        planet, because the resonance keeps the two apart in phase. Tancredi (2014) excludes such
        orbits from the cometary classes for that reason.

        The libration widths used here are the maxima tabulated in the paper's table 1. The paper
        computes a width that varies with eccentricity, by a method given in an appendix that was
        not available, so an orbit near the edge of a resonance may be called protected here when
        the paper would not.

        Reference: Tancredi (2014), Icarus 234, 66, section 6, doi:10.1016/j.icarus.2014.02.013.

    Arguments:
        a: [float] semi-major axis of the orbit (AU)
        e: [float] num. eccentricity of the orbit
        moid_jupiter_hill: [float] minimum orbital intersection distance with Jupiter, in units of
            Jupiter's Hill radius

    Return:
        [bool] True if the orbit is protected by a resonance
    """

    aphelion = a*(1.0 + e)

    for label, p_plus_q, p, half_width in TANCREDI_RESONANCES:

        if abs(a - calcResonanceSemiMajorAxis(p, p_plus_q)) > half_width:
            continue

        if label == '1:1':
            return e < 0.35

        if label == '3:2':
            return aphelion < Q_JUPITER_APHELION + HILL_RADIUS_JUPITER/4.0

        if label == '4:3':
            return aphelion < Q_JUPITER_PERIHELION - HILL_RADIUS_JUPITER/2.0

        # The inner resonances, 4:1 through 2:1
        crosses = (moid_jupiter_hill < 1.5) \
            and (aphelion > Q_JUPITER_PERIHELION - HILL_RADIUS_JUPITER/4.0)

        return (aphelion < Q_JUPITER_APHELION + HILL_RADIUS_JUPITER) and not crosses

    return False


def classifyTancrediAsteroid(a, e, i, moid_jupiter_hill, moid_giants_min_hill):
    """ Classify an asteroid's orbit under the Tancredi (2014) scheme, identifying asteroids in
        cometary orbits.

        Applying the cometary Tisserand cut alone to the asteroid population returns thousands of
        candidates, almost all of them on stable orbits. This criterion adds the two conditions
        that separate a genuinely comet-like orbit: that the body is not held away from Jupiter by
        a resonance, and that it actually reaches the planet, expressed through the minimum
        distance between the orbits. The result is a criterion strict enough that the objects it
        selects evolve chaotically like periodic comets.

        The two minimum orbital intersection distances are arguments rather than computed here.
        Computing a MOID is a separate problem, and the paper's method for it is in an appendix
        that was not available.

        Reference: Tancredi (2014), Icarus 234, 66, section 6, doi:10.1016/j.icarus.2014.02.013.

    Arguments:
        a: [float] semi-major axis of the orbit (AU)
        e: [float] num. eccentricity of the orbit
        i: [float] inclination of the orbit (rad)
        moid_jupiter_hill: [float] minimum orbital intersection distance with Jupiter, in units of
            Jupiter's Hill radius
        moid_giants_min_hill: [float] smallest minimum orbital intersection distance with any giant
            planet, each expressed in units of that planet's own Hill radius

    Return:
        [str] one of 'aco jupiter family', 'aco halley', 'centaur', 'transneptunian', or 'asteroid'
    """

    tisserand = float(calcTisserand(a, e, i))
    q = a*(1.0 - e)
    aphelion = a*(1.0 + e)

    if q > A_URANUS:
        return 'transneptunian'

    if tisserand < TANCREDI_T_LOW:
        return 'aco halley'

    if Q_JUPITER_APHELION < q < A_URANUS:
        return 'centaur'

    if not (TANCREDI_T_LOW < tisserand < TANCREDI_T_HIGH and q < Q_JUPITER_APHELION):
        return 'asteroid'

    if isTancrediResonanceProtected(a, e, moid_jupiter_hill):
        return 'asteroid'

    # Very eccentric orbits that barely graze Jupiter's orbit at aphelion without a small MOID
    if (moid_jupiter_hill > 1.0) and (a < calcResonanceSemiMajorAxis(2, 5)):
        return 'asteroid'

    if (moid_jupiter_hill > 2.0) and (a < calcResonanceSemiMajorAxis(3, 7)):
        return 'asteroid'

    # The orbit must cross Jupiter's, or come close enough to it to be perturbed
    if aphelion > Q_JUPITER_PERIHELION:
        reaches = moid_giants_min_hill < 4.0

    elif aphelion > Q_JUPITER_PERIHELION - 1.5*HILL_RADIUS_JUPITER:
        reaches = moid_giants_min_hill < 2.5

    else:
        reaches = False

    return 'aco jupiter family' if reaches else 'asteroid'
