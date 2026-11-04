""" Dynamical classification of a single orbit.

These criteria take one orbit and return a dynamical class, rather than comparing two orbits, so
they are kept apart from the dissimilarity criteria in wmpl.Utils.Dcriteria.

All functions take angles in radians. The single-parameter criteria and the two-parameter
cometary tests accept scalars or numpy arrays; the Tancredi (2014) classifiers return a label and
take scalars only.
"""

from __future__ import print_function, division, absolute_import

import os
import concurrent.futures

import numpy as np
import scipy.integrate
import scipy.optimize
import scipy.special

# Re-exported so that a caller can keep importing them from here
from wmpl.Utils.DynamicsCheckpoint import DynamicsCheckpoint, checkpointSeed, fingerprint
from wmpl.Utils.OrbitConstants import (GAUSS_K, GAUSS_K_SQUARED, A_JUPITER, A_SATURN, A_URANUS,
    A_NEPTUNE)




# Inclination above which an orbit is taken to be cometary in the two-parameter criteria of
#   Jopek & Williams (2013) [rad]
JW_INCL_LIMIT = np.radians(75.0)

# Cometary limits of the one-parameter criteria, as adopted by Jopek & Williams (2013)
JW_APHELION_LIMIT = 4.6      # aphelion distance [AU]
JW_KRESAK_P_LIMIT = 2.5      # Kresak P [yr]
JW_ENERGY_LIMIT = -5.28e-5   # orbital energy -k^2/(2a) [AU^2/day^2]

# Tisserand parameter below which an orbit is taken to be cometary. Jopek & Williams (2013) write
#   the limit as 0.58, but their eq. 10 defines T as 1/a + 2*a_J^-1.5*sqrt(a*(1 - e^2))*cos(i),
#   which is the Tisserand parameter divided by a_J and so carries units of 1/AU. Multiplying by
#   a_J puts the limit in the units calcTisserand returns, and recovers the familiar cut at 3
JW_TISSERAND_LIMIT = 0.58*A_JUPITER





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
#   (label, p + q, p, fixed half width in semi-major axis [AU] or None to compute it). The paper
#   computes the width for the five inner resonances and adopts a fixed value for the three outer
#   ones, where the expansion of the disturbing function is no longer reliable
TANCREDI_RESONANCES = [
    ('4:1', 4, 1, None),
    ('3:1', 3, 1, None),
    ('5:2', 5, 2, None),
    ('7:3', 7, 3, None),
    ('2:1', 2, 1, None),
    ('3:2', 3, 2, 0.1),
    ('4:3', 4, 3, 0.1),
    ('1:1', 1, 1, 0.208),
    ]

# Eccentricity range over which the libration width is evaluated. Outside it the expansion of the
#   disturbing function is not reliable, so the width is held at the value on the nearer edge
RESONANCE_E_MIN = 0.01
RESONANCE_E_MAX = 0.3

# Mass of Jupiter divided by the mass of the Sun
MASS_RATIO_JUPITER = 1.0/1047.3486

# Mean elements of the giant planets at J2000, as
#   (name, a [AU], e, i [rad], node [rad], argument of perihelion [rad], Hill radius [AU])
GIANT_PLANETS = [
    ('jupiter', 5.20288700, 0.04838624, np.radians(1.30439695), np.radians(100.47390909),
        np.radians(14.72847983 - 100.47390909), HILL_RADIUS_JUPITER),
    ('saturn', 9.53667594, 0.05386179, np.radians(2.48599187), np.radians(113.66242448),
        np.radians(92.59887831 - 113.66242448), HILL_RADIUS_SATURN),
    ('uranus', 19.18916464, 0.04725744, np.radians(0.77263783), np.radians(74.01692503),
        np.radians(170.95427630 - 74.01692503), HILL_RADIUS_URANUS),
    ('neptune', 30.06992276, 0.00859048, np.radians(1.77004347), np.radians(131.78422574),
        np.radians(44.96476227 - 131.78422574), HILL_RADIUS_NEPTUNE),
    ]

# Laplace coefficients depend only on the resonance, so they are worth keeping
_LAPLACE_CACHE = {}


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
        [float] Tisserand parameter. nan for an unbound orbit (e >= 1), where the parameter is not
            defined. wmpl.Rebound.REBOUND.tisserandParameterJupiter is the scalar equivalent used by
            the REBOUND report and returns None in that case instead.
    """

    # An unbound orbit takes the square root of a negative number and gives nan, which is the right
    #   answer: the parameter describes bounded motion. The classifiers below test for it rather
    #   than letting every comparison against nan quietly evaluate to False.
    return a_planet/a + 2*np.cos(i)*np.sqrt((a/a_planet)*(1.0 - e**2))


def calcWhippleK(a, e):
    """ Calculate the Whipple K criterion of an orbit.

        K is positive for cometary orbits and negative for asteroidal ones.

        The criterion is empirical and has no dynamical basis, so the sign carries no meaning for
        an orbit that lands near zero. Jopek & Williams (2013) state that it is inconclusive for
        short-period orbits of low eccentricity, and give the Pribram and Neuschwanstein
        meteorites, which are of asteroidal origin, as a case where it returns K ~ 0.08. It
        produced the most exceptions of the five criteria in their reliability test, 16.4 per cent
        among near-Earth asteroids and 13.8 per cent among periodic comets.

        Reference: Whipple (1954), AJ 59, 201, as eq. 3 of Jopek & Williams (2013), MNRAS 430,
        2377, doi:10.1093/mnras/stt057, whose eq. 12 gives the sign of the cometary limit.

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


def isCometaryTi(a, e, i, a_planet=A_JUPITER):
    """ Classify an orbit as cometary or asteroidal using the two-parameter T-i criterion.

        Of the five two-parameter criteria this is the only one with a dynamical basis rather than
        an empirical one, the Tisserand parameter being conserved under an encounter with the
        planet. The paper nevertheless found Q-i and E-i to be the more reliable discriminants.

        The paper restricted its sample to elliptical orbits, and so does the comparison here: an
        unbound orbit gives a Tisserand parameter of nan, which counts as asteroidal.

        Reference: Jopek & Williams (2013), MNRAS 430, 2377, eq. 10, doi:10.1093/mnras/stt057.

    Arguments:
        a: [float] semi-major axis of the orbit (AU)
        e: [float] num. eccentricity of the orbit
        i: [float] inclination of the orbit (rad)

    Keyword arguments:
        a_planet: [float] semi-major axis of the perturbing planet (AU). Default Jupiter.

    Return:
        [bool] True if the orbit is cometary
    """

    return (calcTisserand(a, e, i, a_planet=a_planet) < JW_TISSERAND_LIMIT) \
        | (np.asarray(i) > JW_INCL_LIMIT)


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

    return (calcWhippleK(a, e) > 0.0) | (np.asarray(i) > JW_INCL_LIMIT)


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

        The HILL_RADIUS_* constants are the values the paper quotes rather than values from this
        function, so that the classification reproduces the paper exactly; the two agree to about
        2e-3 AU. Use this function for a planet the module does not tabulate.

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

    # An unbound orbit has no Tisserand parameter, and every comparison against nan is False, so
    #   without this it would reach 'unclassified' for the wrong reason
    if not np.isfinite(tisserand):
        return 'unclassified'

    # The intervals are half open, so an orbit landing exactly on a limit is classified instead of
    #   falling through every branch
    if (tisserand < TANCREDI_T_LOW) and (a < A_NEPTUNE):
        return 'halley'

    if (TANCREDI_T_LOW <= tisserand < TANCREDI_T_HIGH) and (q <= Q_JUPITER_APHELION):
        return 'jupiter family'

    if (tisserand >= TANCREDI_T_HIGH) and (q <= Q_JUPITER_APHELION):
        return 'asteroidal orbit'

    if (tisserand >= TANCREDI_T_LOW) and (Q_JUPITER_APHELION < q < A_URANUS):
        return 'centaur'

    return 'unclassified'


def isTancrediResonanceProtected(a, e, moid_jupiter_hill):
    """ Decide whether an orbit sits in a mean-motion resonance with Jupiter that protects it from
        close encounters.

        An orbit can have a small minimum distance to Jupiter's orbit and still never approach the
        planet, because the resonance keeps the two apart in phase. Tancredi (2014) excludes such
        orbits from the cometary classes for that reason.

        The libration width is computed from the eccentricity by calcResonanceWidth for the five
        inner resonances, as the paper does. The three outer ones keep the fixed widths the paper
        adopts for them, where the expansion of the disturbing function is no longer reliable.

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

    for label, p_plus_q, p, fixed_width in TANCREDI_RESONANCES:

        half_width = fixed_width if fixed_width is not None \
            else calcResonanceWidth(p, p_plus_q, e)

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

        The two minimum orbital intersection distances are arguments, so that a study can supply
        values computed against whatever planetary ephemeris it uses. calcGiantPlanetMOIDs computes
        them from mean elements at J2000, in the Hill radii expected here, as a dictionary keyed by
        planet: pass its 'jupiter' entry and the smallest of its values.

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


def _orbitPosition(ecc_anomaly, a, e, i, node, peri):
    """ Heliocentric position of a point on an orbit, given its eccentric anomaly.

    Arguments:
        ecc_anomaly: [float] eccentric anomaly (rad)
        a: [float] semi-major axis (AU)
        e: [float] num. eccentricity
        i: [float] inclination (rad)
        node: [float] longitude of ascending node (rad)
        peri: [float] argument of perihelion (rad)

    Return:
        [ndarray] position vector (AU)
    """

    # Position in the orbital plane, with the origin at the focus and x towards perihelion
    x = a*(np.cos(ecc_anomaly) - e)
    y = a*np.sqrt(1.0 - e**2)*np.sin(ecc_anomaly)

    cos_w, sin_w = np.cos(peri), np.sin(peri)
    cos_i, sin_i = np.cos(i), np.sin(i)
    cos_O, sin_O = np.cos(node), np.sin(node)

    # Rotate by the argument of perihelion, then by the inclination and the node, i.e. the
    #   standard 3-1-3 Euler sequence written out
    x_peri = x*cos_w - y*sin_w
    y_peri = x*sin_w + y*cos_w

    return np.array([x_peri*cos_O - y_peri*cos_i*sin_O, x_peri*sin_O + y_peri*cos_i*cos_O,
        y_peri*sin_i])


def calcMOID(a1, e1, i1, O1, w1, a2, e2, i2, O2, w2):
    """ Calculate the minimum orbital intersection distance between two orbits.

        Note the element order: this function takes the semi-major axis first, where every
        dissimilarity criterion in wmpl.Utils.Dcriteria takes the perihelion distance. The two
        signatures are otherwise the same shape, so a q passed here is silently accepted.

        The distance between a point on each orbit is minimised over the two eccentric anomalies.
        That surface has more than one local minimum, so the search is repeated from the three
        further starting points the paper prescribes and the smallest of the four results is taken.

        Reference: Tancredi (2014), Icarus 234, 66, appendix A, doi:10.1016/j.icarus.2014.02.013.

    Arguments:
        a1: [float] semi-major axis of the first orbit (AU)
        e1: [float] num. eccentricity of the first orbit
        i1: [float] inclination of the first orbit (rad)
        O1: [float] longitude of ascending node of the first orbit (rad)
        w1: [float] argument of perihelion of the first orbit (rad)
        a2: [float] semi-major axis of the second orbit (AU)
        e2: [float] num. eccentricity of the second orbit
        i2: [float] inclination of the second orbit (rad)
        O2: [float] longitude of ascending node of the second orbit (rad)
        w2: [float] argument of perihelion of the second orbit (rad)

    Return:
        [float] minimum orbital intersection distance (AU)
    """

    orbit1 = (a1, e1, i1, O1, w1)
    orbit2 = (a2, e2, i2, O2, w2)

    def separation(anomalies):
        return float(np.linalg.norm(_orbitPosition(anomalies[0], *orbit1)
            - _orbitPosition(anomalies[1], *orbit2)))

    def refine(start):

        # The tolerances are well below the AU-scale distances being compared, so the result is
        #   limited by which basin the search lands in rather than by how finely it converges
        found = np.atleast_1d(scipy.optimize.fmin_powell(separation, np.asarray(start, dtype=float),
            disp=False, xtol=1e-10, ftol=1e-12))
        return found, separation(found)

    # Powell converges to whichever local minimum it starts nearest, and the separation surface
    #   has several. The paper restarts from three points reflected about the first result, which
    #   between them reach the other basins: the opposite side of both orbits, the opposite side of
    #   the second only, and the two reflections combined.
    first, smallest = refine((0.0, 0.0))

    for start in ((first[0] + np.pi, first[1] + np.pi), (first[0], 2*np.pi - first[1]),
                  (first[0] + np.pi, np.pi - first[1])):

        _, distance = refine(start)
        smallest = min(smallest, distance)

    return smallest


def calcGiantPlanetMOIDs(a, e, i, node, peri):
    """ Calculate the minimum orbital intersection distance of an orbit with each giant planet,
        expressed in units of that planet's Hill radius.

        The Tancredi (2014) criterion compares these distances against a few Hill radii, because
        that is the scale on which an encounter perturbs a heliocentric orbit, so the planet's own
        Hill radius is the natural unit for each.

        The planetary elements used are mean elements at J2000. A study spanning a long time base
        should supply its own.

        classifyTancrediAsteroid takes two scalars rather than this dictionary, so a caller passes
        distances['jupiter'] and min(distances.values()) from the result.

    Arguments:
        a: [float] semi-major axis of the orbit (AU)
        e: [float] num. eccentricity of the orbit
        i: [float] inclination of the orbit (rad)
        node: [float] longitude of ascending node of the orbit (rad)
        peri: [float] argument of perihelion of the orbit (rad)

    Return:
        [dict] minimum orbital intersection distance with each giant planet, in Hill radii
    """

    distances = {}

    for name, a_p, e_p, i_p, node_p, peri_p, hill in GIANT_PLANETS:

        distances[name] = calcMOID(a, e, i, node, peri, a_p, e_p, i_p, node_p, peri_p)/hill

    return distances


def calcMaxPerihelionForTisserand(tisserand):
    """ Calculate the largest perihelion distance an orbit interior to the planet can have at a
        given Tisserand parameter.

        The largest perihelion belongs to the circular orbit, which sets the upper end of the range
        of orbits available at that Tisserand parameter. Distances are in units of the planet's
        semi-major axis, and the orbit is taken to be coplanar with the planet.

        Reference: Tancredi (2014), Icarus 234, 66, appendix A, eq. A.1,
        doi:10.1016/j.icarus.2014.02.013.

    Arguments:
        tisserand: [float] Tisserand parameter, which has to exceed 3

    Return:
        [float] largest perihelion distance, in units of the planet's semi-major axis
    """

    if tisserand < 3.0:
        raise ValueError("The Tisserand parameter has to exceed 3 for a forbidden region to exist.")

    # Eq. A.1 with i = 0, solved directly rather than through the cubic it squares to, which
    #   carries two spurious roots
    return scipy.optimize.brentq(lambda q: 1.0/q + 2*np.sqrt(q) - tisserand, 1e-12, 1.0)


def calcMinMOIDForTisserand(tisserand, n_samples=2000):
    """ Calculate the smallest minimum orbital intersection distance an orbit interior to the
        planet can have at a given Tisserand parameter.

        Above a Tisserand parameter of 3 an orbit cannot cross the planet's, so it cannot approach
        closer than some distance. That bound traces the edge of the forbidden region in the
        Tisserand against MOID plane, where no object can lie.

        Reference: Tancredi (2014), Icarus 234, 66, appendix A, doi:10.1016/j.icarus.2014.02.013.

    Arguments:
        tisserand: [float] Tisserand parameter, which has to exceed 3

    Keyword arguments:
        n_samples: [int] number of perihelion distances sampled in the search for the largest
            aphelion. Default 2000.

    Return:
        [float] smallest possible MOID, in units of the planet's semi-major axis
    """

    q_max = calcMaxPerihelionForTisserand(tisserand)

    def aphelionFor(q):
        # The Tisserand parameter written in q and Q, with i = 0, solved for Q
        return scipy.optimize.brentq(
            lambda Q: 2.0/(q + Q) + 2*np.sqrt(2*q*Q/(q + Q)) - tisserand, q, 1e3)

    perihelia = np.linspace(q_max*1e-6, q_max*(1.0 - 1e-9), n_samples)

    return 1.0 - max(aphelionFor(q) for q in perihelia)


def calcLaplaceCoefficient(alpha, j, s):
    """ Calculate a Laplace coefficient of the expansion of the disturbing function.

        Evaluated from its integral definition rather than the truncated series, so it stays exact
        as alpha approaches 1.

    Arguments:
        alpha: [float] ratio of the two semi-major axes, inner over outer
        j: [int] order of the cosine term
        s: [float] index of the coefficient

    Return:
        [float] Laplace coefficient
    """

    # b_s^(-j) = b_s^(j), so only the magnitude matters and the cache is not split over the sign
    j = abs(int(j))

    # Rounded so that values differing only in the last bits share an entry. The coefficients are
    #   needed repeatedly for the same handful of resonances, and each one costs a quadrature.
    key = (round(float(alpha), 12), j, round(float(s), 6))

    if key not in _LAPLACE_CACHE:

        integrand = lambda theta: np.cos(j*theta)/(1.0 - 2*alpha*np.cos(theta) + alpha**2)**s

        _LAPLACE_CACHE[key] = scipy.integrate.quad(integrand, 0.0, 2*np.pi, limit=200)[0]/np.pi

    return _LAPLACE_CACHE[key]


def calcLaplaceDerivative(alpha, j, s, n):
    """ Calculate a derivative of a Laplace coefficient with respect to alpha.

        Reference: Murray & Dermott (1999), eqs 6.70 and 6.71, as used in appendix B of Tancredi
        (2014).

    Arguments:
        alpha: [float] ratio of the two semi-major axes, inner over outer
        j: [int] order of the cosine term
        s: [float] index of the coefficient
        n: [int] order of the derivative

    Return:
        [float] derivative of the Laplace coefficient
    """

    if n == 0:
        return calcLaplaceCoefficient(alpha, j, s)

    # The first derivative in terms of coefficients of index s + 1, eq. 6.70
    if n == 1:
        return s*(calcLaplaceCoefficient(alpha, j - 1, s + 1)
            - 2*alpha*calcLaplaceCoefficient(alpha, j, s + 1)
            + calcLaplaceCoefficient(alpha, j + 1, s + 1))

    # Higher derivatives by differentiating eq. 6.70 n - 1 more times, eq. 6.71. The last term
    #   comes from differentiating the -2*alpha factor and vanishes for n = 1, which is why that
    #   case is written out separately above rather than folded in here.
    return s*(calcLaplaceDerivative(alpha, j - 1, s + 1, n - 1)
        - 2*alpha*calcLaplaceDerivative(alpha, j, s + 1, n - 1)
        + calcLaplaceDerivative(alpha, j + 1, s + 1, n - 1)
        - 2*(n - 1)*calcLaplaceDerivative(alpha, j, s + 1, n - 2))


def calcDisturbingFunctionTerm(alpha, j, order):
    """ Calculate the direct term of the expansion of the disturbing function for a resonance.

        Reference: Tancredi (2014), Icarus 234, 66, appendix B, table B.1,
        doi:10.1016/j.icarus.2014.02.013.

    Arguments:
        alpha: [float] ratio of the two semi-major axes, inner over outer
        j: [int] first integer of the critical angle, equal to p + q
        order: [int] order of the resonance, from 1 to 4

    Return:
        [float] direct term of the disturbing function
    """

    # The expansion is in the Laplace coefficients of index 1/2 and their first four derivatives
    #   with respect to alpha. All five are computed up front; an order-1 term uses only the first
    #   two, but the cache in calcLaplaceCoefficient makes the unused ones nearly free.
    b = [calcLaplaceDerivative(alpha, j, 0.5, n) for n in range(5)]

    # Every numerical coefficient below is transcribed from table B.1 of the paper. They are the
    #   polynomial in j multiplying each derivative, and they have no separate meaning: an error in
    #   one shows up only as a wrong resonance width, so they are checked against the widths the
    #   paper tabulates (see test_tancredi_resonance_widths_match_table1).
    if order == 1:
        return 0.5*(-2*j*b[0] - alpha*b[1])

    if order == 2:
        return (1.0/8)*((-5*j + 4*j**2)*b[0] + (-2 + 4*j)*alpha*b[1] + alpha**2*b[2])

    if order == 3:
        return (1.0/48)*((-26*j + 30*j**2 - 8*j**3)*b[0] + (-9 + 27*j - 12*j**2)*alpha*b[1]
            + (6 - 6*j)*alpha**2*b[2] - alpha**3*b[3])

    if order == 4:
        return (1.0/384)*((-206*j + 283*j**2 - 120*j**3 + 16*j**4)*b[0]
            + (-64 + 236*j - 168*j**2 + 32*j**3)*alpha*b[1]
            + (48 - 78*j + 24*j**2)*alpha**2*b[2] + (-12 + 8*j)*alpha**3*b[3] + alpha**4*b[4])

    raise ValueError("Only resonances of order 1 to 4 are covered, got {!r}.".format(order))


def calcResonanceWidth(p, p_plus_q, e, mass_ratio=MASS_RATIO_JUPITER, a_planet=A_JUPITER):
    """ Calculate the half width in semi-major axis of a mean-motion resonance.

        The width is the extent over which a body librates about the centre of the resonance, from
        the pendulum model of the resonant part of the disturbing function. It widens with
        eccentricity, so the eccentricity is held to the range over which the expansion is
        reliable, 0.01 to 0.3.

        Reference: Tancredi (2014), Icarus 234, 66, appendix B, eqs B.5 to B.7,
        doi:10.1016/j.icarus.2014.02.013.

    Arguments:
        p: [int] second integer of the (p + q):p resonance label
        p_plus_q: [int] first integer of the (p + q):p resonance label
        e: [float] num. eccentricity of the orbit

    Keyword arguments:
        mass_ratio: [float] mass of the perturbing planet over the mass of the Sun. Default Jupiter.
        a_planet: [float] semi-major axis of the perturbing planet (AU). Default Jupiter.

    Return:
        [float] half width of the resonance in semi-major axis (AU)
    """

    order = p_plus_q - p

    e = min(max(e, RESONANCE_E_MIN), RESONANCE_E_MAX)

    alpha = (float(p)/p_plus_q)**(2.0/3.0)

    # The mean motion in C_r cancels, since only |C_r|/n enters the width
    strength = abs(mass_ratio*alpha*calcDisturbingFunctionTerm(alpha, p_plus_q, order))

    if order >= 2:
        relative_width = np.sqrt(16.0/3.0*strength*e**order)

    else:
        # First order, eq. B.6. The second integer of the critical angle is negative, and carrying
        #   that sign is what reproduces the width tabulated in the paper
        j2 = -p
        relative_width = np.sqrt(16.0/3.0*strength*e)*np.sqrt(1.0 + strength/(27*j2**2*e**3)) \
            - 2.0/(9*j2*e)*strength

    return relative_width*(float(p)/p_plus_q)**(2.0/3.0)*a_planet




# Perihelion and semi-major axis limits of the Jupiter-family dynamics metrics [AU]. A Jupiter-family
#   comet spends most of its time beyond either, because close encounters with Jupiter keep lifting
#   its perihelion, while an asteroid on a comet-like orbit rarely reaches them. Shober et al. (2024),
#   sec. 2.4.1, after Fernandez et al. (2014)
JFC_DYNAMICS_Q_LIMIT = 2.5
JFC_DYNAMICS_A_LIMIT = 7.37

# Fraction of the clones of an object reaching JFC_DYNAMICS_Q_LIMIT above which the object is taken to
#   move like a Jupiter-family comet. Over 90% of 661 known JFCs exceed it over 10 kyr, against ~10% of
#   the fireballs on Jupiter-family orbits, Shober et al. (2024), sec. 3.3.1
JFC_DYNAMICS_P_THRESHOLD = 0.5


def calcKozaiEnergy(e, i):
    """ Calculate the Kozai energy H = sqrt(1 - e^2) cos(i) of an orbit.

        H is the z-component of the orbital angular momentum in units of sqrt(G*M_sun*a), which the
        secular perturbations of an axisymmetric potential conserve. Below about 0.7 the perihelion
        distance and the argument of perihelion of a Jupiter-family orbit cycle together under the
        Kozai mechanism (Fernandez et al. 2014), which can lift the perihelion without any close
        encounter. Its square is the C1 of calcDB in wmpl.Utils.Dcriteria.

        Reference: Shober et al. (2024), A&A 687, A181, sec. 3.3.3,
        doi:10.1051/0004-6361/202449635.

    Arguments:
        e: [float] num. eccentricity of the orbit
        i: [float] inclination of the orbit (rad)

    Return:
        [float] H, negative for a retrograde orbit. nan for an unbound orbit (e > 1).
    """

    return np.sqrt(1.0 - e**2)*np.cos(i)


def _ensembleArrays(t, *series):
    """ Check the output times and per-clone series of an integrated ensemble and shape them.

    Arguments:
        t: [ndarray] output times (yr), shape (n_times,)
        *series: [ndarray] one or more per-clone series, each of shape (n_clones, n_times), or
            (n_times,) for a single clone

    Return:
        (elapsed, arrays):
            elapsed: [ndarray] time elapsed since the epoch at each output (yr), which does not
                depend on whether the integration ran forward or backward
            arrays: [list] the series as 2D float arrays
    """

    t = np.asarray(t, dtype=float)

    if t.ndim != 1:
        raise ValueError("The output times have to be a 1D array, got shape {}.".format(t.shape))

    # Integration order is the order of the outputs only if they move away from the epoch
    elapsed = np.abs(t)
    if np.any(np.diff(elapsed) < 0):
        raise ValueError("The output times have to move away from the epoch monotonically.")

    arrays = []
    for s in series:

        s = np.atleast_2d(np.asarray(s, dtype=float))

        if s.shape[1] != len(t):
            raise ValueError("A series has {:d} samples per clone for {:d} output times.".format(
                s.shape[1], len(t)))

        arrays.append(s)

    return elapsed, arrays


def calcExcursionFractions(t, q, a, q_limit=JFC_DYNAMICS_Q_LIMIT, a_limit=JFC_DYNAMICS_A_LIMIT):
    """ Calculate the fractions of time an ensemble of clones spends with its perihelion distance and
        its semi-major axis beyond the Jupiter-family limits, the f_q and f_a indices.

        f_q = sum_j dt_j/sum_j T_j, where dt_j is the time clone j spends with q > q_limit and T_j the
        time over which it was integrated, and likewise f_a with a > a_limit. With every clone
        integrated over the same span T this is eq. 1 and 2 of Shober et al. (2024), which divide by
        N*T. A clone that ends early, by hitting a body, contributes only the time it was followed,
        so its missing time is neither counted as beyond the limit nor as within it.

        Each output is held until the next one, so the times are exact for a series that changes only
        at the outputs, and otherwise off by at most one output interval per crossing of a limit. A
        clone that ends between two outputs is likewise taken as followed until the second. The first
        output is the state the integration starts from, not a result of it, so the first interval
        takes the state of the second output instead: a clone started beyond a limit that leaves it
        at once would otherwise count one interval beyond it, which is 0.5% of the time with 200
        outputs and more the shorter the run.

        Reference: Fernandez et al. (2014), Icarus 238, 1; Shober et al. (2024), A&A 687, A181,
        sec. 2.4.1, doi:10.1051/0004-6361/202449635.

    Arguments:
        t: [ndarray] output times (yr), shape (n_times,), negative for a backward integration
        q: [ndarray] perihelion distance (AU), shape (n_clones, n_times). nan where a clone has no
            state (after it ended), +inf after it left the solar system.
        a: [ndarray] semi-major axis (AU), same shape and conventions as q. A negative value, which is
            what an unbound orbit has, counts as beyond the limit.

    Keyword arguments:
        q_limit: [float] perihelion limit (AU). Default JFC_DYNAMICS_Q_LIMIT.
        a_limit: [float] semi-major axis limit (AU). Default JFC_DYNAMICS_A_LIMIT.

    Return:
        (f_q, f_a): [tuple of float] the two fractions, nan if no clone has any time with a state
    """

    elapsed, (q, a) = _ensembleArrays(t, q, a)

    # An unbound orbit has no semi-major axis to compare, and it is leaving: beyond any limit
    a = np.where(a < 0, np.inf, a)

    # Each interval between two outputs takes the state of its first output, but for the first
    #   interval, which takes the state of the first output after the start
    dt = np.diff(elapsed)

    def _fraction(x, limit):

        x = np.column_stack((x[:, 1], x[:, 1:-1]))
        has_state = ~np.isnan(x)

        total = np.sum(dt*has_state)
        if total == 0:
            return np.nan

        return np.sum(dt*(x > limit))/total

    return _fraction(q, q_limit), _fraction(a, a_limit)


def countPerihelionReach(q, q_limit=JFC_DYNAMICS_Q_LIMIT):
    """ Count the clones whose perihelion distance exceeds a limit at least once.

        This is the metric Shober et al. (2024) classify by, the percentage of clones reaching
        q > 2.5 AU at least once over the integration, which replaced counting close encounters with
        Jupiter by eye: the two correlate with a Pearson coefficient of 0.86.

        Reference: Shober et al. (2024), A&A 687, A181, secs 2.4.1 and 3.3.1,
        doi:10.1051/0004-6361/202449635.

    Arguments:
        q: [ndarray] perihelion distance (AU), shape (n_clones, n_times). nan where a clone has no
            state, +inf after it left the solar system.

    Keyword arguments:
        q_limit: [float] perihelion limit (AU). Default JFC_DYNAMICS_Q_LIMIT.

    Return:
        (n_reached, n_clones): [tuple of int] clones that exceeded the limit, and clones with at least
            one state, the only ones that carry any information
    """

    q = np.atleast_2d(np.asarray(q, dtype=float))

    has_state = np.any(~np.isnan(q), axis=1)
    reached = np.any(q > q_limit, axis=1)

    return int(np.count_nonzero(reached & has_state)), int(np.count_nonzero(has_state))


def splitPerihelionReachByEncounter(t, q, encounter_times, q_limit=JFC_DYNAMICS_Q_LIMIT):
    """ Split the clones that reach a perihelion limit by whether a close encounter came first.

        The Kozai mechanism can lift the perihelion past the limit with no encounter at all, which
        counts a stable orbit as Jupiter-family (Shober et al. 2024, fig. 7). A clone is counted as
        lifted by an encounter if one of the given encounters happened no later than the first output
        beyond the limit, in the order of the integration; the crossing itself happened somewhere in
        the interval before that output.

    Arguments:
        t: [ndarray] output times (yr), shape (n_times,), negative for a backward integration
        q: [ndarray] perihelion distance (AU), shape (n_clones, n_times)
        encounter_times: [list] for each clone, the times (yr, on the same scale as t) of the
            encounters that count. The caller chooses them, by body and by depth.

    Keyword arguments:
        q_limit: [float] perihelion limit (AU). Default JFC_DYNAMICS_Q_LIMIT.

    Return:
        (n_with, n_without): [tuple of int] clones that reached the limit after an encounter, and
            clones that reached it without one. Together they are the n_reached of
            countPerihelionReach.
    """

    elapsed, (q,) = _ensembleArrays(t, q)

    if len(encounter_times) != q.shape[0]:
        raise ValueError("{:d} lists of encounter times were given for {:d} clones.".format(
            len(encounter_times), q.shape[0]))

    n_with = 0
    n_without = 0

    for q_clone, times in zip(q, encounter_times):

        beyond = np.flatnonzero(q_clone > q_limit)
        if beyond.size == 0:
            continue

        if np.any(np.abs(np.asarray(times, dtype=float)) <= elapsed[beyond[0]]):
            n_with += 1
        else:
            n_without += 1

    return n_with, n_without


def calcBinomialInterval(k, n, confidence=0.95):
    """ Calculate the Wilson score interval of a binomial proportion.

        Unlike the normal approximation, it stays inside [0, 1] and keeps a finite width at k = 0 and
        k = n, which is where the fraction of clones of a clearly stable or clearly chaotic orbit
        falls.

        Reference: Wilson (1927), J. Am. Stat. Assoc. 22, 209.

    Arguments:
        k: [int] number of successes
        n: [int] number of trials, positive

    Keyword arguments:
        confidence: [float] two-sided confidence level. Default 0.95.

    Return:
        (low, high): [tuple of float] bounds of the interval
    """

    if n <= 0:
        raise ValueError("The interval needs at least one trial, got n = {}.".format(n))

    z = scipy.special.ndtri(0.5 + confidence/2.0)
    p = float(k)/n

    denominator = 1.0 + z**2/n
    centre = (p + z**2/(2.0*n))/denominator
    half_width = z*np.sqrt(p*(1.0 - p)/n + z**2/(4.0*n**2))/denominator

    # At k = 0 and k = n the bound is exactly 0 or 1, but the subtraction of two nearly equal terms
    #   leaves a few ulp of rounding in its place
    low = 0.0 if k == 0 else centre - half_width
    high = 1.0 if k == n else centre + half_width

    return low, high


def classifyPerihelionReach(n_reached, n_clones, threshold=JFC_DYNAMICS_P_THRESHOLD,
    confidence=0.95):
    """ Classify the dynamics of an object from the fraction of its clones that reached the
        Jupiter-family perihelion limit.

        Two verdicts are given. The first is the rule of Shober et al. (2024), the fraction above the
        threshold, kept so that a result can be compared with theirs. The second takes the number of
        clones into account: the object is Jupiter-family only if the whole confidence interval of the
        fraction lies above the threshold, not Jupiter-family if it lies below, and undetermined if it
        contains the threshold, in which case more clones can settle it.

        Reference: Shober et al. (2024), A&A 687, A181, sec. 3.3.1,
        doi:10.1051/0004-6361/202449635.

    Arguments:
        n_reached: [int] clones that reached the limit, from countPerihelionReach
        n_clones: [int] clones with a state, from countPerihelionReach

    Keyword arguments:
        threshold: [float] fraction of clones that separates the classes. Default
            JFC_DYNAMICS_P_THRESHOLD.
        confidence: [float] confidence level of the interval. Default 0.95.

    Return:
        [dict] with
            p: [float] fraction of clones that reached the limit
            p_low, p_high: [float] its Wilson interval (see calcBinomialInterval)
            jupiter_family_shober: [bool] p > threshold
            dynamics_class: [str] 'jupiter family', 'not jupiter family' or 'undetermined'
    """

    p_low, p_high = calcBinomialInterval(n_reached, n_clones, confidence=confidence)
    p = float(n_reached)/n_clones

    if p_low > threshold:
        dynamics_class = 'jupiter family'

    elif p_high < threshold:
        dynamics_class = 'not jupiter family'

    else:
        dynamics_class = 'undetermined'

    return {'p': p, 'p_low': p_low, 'p_high': p_high, 'jupiter_family_shober': p > threshold,
        'dynamics_class': dynamics_class}


# Defaults of classifyJupiterFamilyDynamics. The span is that of Shober et al. (2024). Outputs every
#   50 yr take 1/50 of the memory of yearly ones and gave the same fraction reaching q > 2.5 AU on a
#   21P-like orbit
JFC_DYNAMICS_YEARS = 10000.0
JFC_DYNAMICS_OUTPUT_YEARS = 50.0
JFC_DYNAMICS_BATCH = 20
JFC_DYNAMICS_MAX_CLONES = 200

# Depth, in Hill radii of Jupiter, of the encounters that count as lifting a perihelion past the
#   limit (see splitPerihelionReachByEncounter)
JFC_DYNAMICS_ENCOUNTER_HILL = 1.0


def cloneEnsemble(outputs_mc, diagnostics, n_outputs, encounter_hill=JFC_DYNAMICS_ENCOUNTER_HILL):
    """ Turn the lean outputs of the clones of a REBOUND run into the arrays of the Jupiter-family
        metrics.

        A clone that hit a body has no state after its last output (nan). One that left the simulation
        volume of wmpl.Rebound.REBOUND (1000 AU from the Sun) is told apart by its eccentricity:
          - bound at its last output: its aphelion lies beyond the volume, where nothing in the force
            model changes the orbit, so it keeps its last q and a;
          - unbound at its last output, after being bound: it was thrown out, and is beyond every
            limit from then on (+inf), as calcExcursionFractions expects;
          - unbound at every output after the epoch: it was never part of the bound dynamics being
            classified, and is left out of the metrics altogether (nan at every output).
        A clone unbound at every output after the epoch that has not reached the volume by the end of
        the run is left out too: it is the same orbit, only with less time to get there.
        The output at the epoch is not used for this, because it is taken at the Earth, whose pull
        changes the heliocentric orbit: on a Halley-type orbit with a = 10 AU it gave a = 36-108 AU.

    Arguments:
        outputs_mc: [dict] {clone name: {key: array}} from wmpl.Rebound.REBOUND.reboundSimulate with
            lean_outputs=True.
        diagnostics: [dict] {clone name: diagnostics} from the same call (return_diagnostics=True).
        n_outputs: [int] Number of output times the run asked for.

    Keyword arguments:
        encounter_hill: [float] Encounters with Jupiter closer than this many Hill radii are returned.
            Default JFC_DYNAMICS_ENCOUNTER_HILL.

    Return:
        (q, a, encounter_times, left_unbound): q and a [ndarray] of shape (n_clones, n_outputs), in
            AU; for each clone the times (yr, negative before the epoch) of its encounters with
            Jupiter; and [ndarray] whether each clone was unbound at every output after the epoch.
            All in the order of the clone names.
    """

    names = sorted(outputs_mc, key=lambda name: int(name.rsplit('_', 1)[1]))

    q = np.full((len(names), n_outputs), np.nan)
    a = np.full((len(names), n_outputs), np.nan)
    encounter_times = []
    left_unbound = np.zeros(len(names), dtype=bool)

    for k, name in enumerate(names):

        columns = outputs_mc[name]
        n = len(columns['a'])

        q[k, :n] = columns['a']*(1.0 - columns['e'])
        a[k, :n] = columns['a']

        escaped = diagnostics[name]['escaped'] is not None
        e_after_epoch = columns['e'][1:] if n > 1 else columns['e']

        # A clone with a single output that stayed inside the volume hit a body at once, so nothing
        #   is known of its orbit after the epoch
        if (escaped or (n > 1)) and np.all(e_after_epoch >= 1.0):
            left_unbound[k] = True
            q[k] = np.nan
            a[k] = np.nan

        elif escaped:

            if columns['e'][-1] < 1.0:
                q[k, n:] = q[k, n - 1]
                a[k, n:] = a[k, n - 1]

            else:
                q[k, n:] = np.inf
                a[k, n:] = np.inf

        encounter_times.append([enc['time_days']/365.25 for enc in diagnostics[name]['encounters']
            if (enc['body'] == 'Jupiter') and (enc['n_hill'] < encounter_hill)])

    return q, a, encounter_times, left_unbound


def summariseCloneEnsemble(t, q, a, encounter_times, left_unbound, confidence=0.95):
    """ The Jupiter-family metrics of an ensemble of clones, as arranged by cloneEnsemble.

    Arguments:
        t: [ndarray] Output times (yr).
        q, a, encounter_times, left_unbound: The return of cloneEnsemble, for every clone.

    Keyword arguments:
        confidence: [float] Confidence of the interval of the fraction. Default 0.95.

    Return:
        [dict] 'n_clones' (all but the unbound ones), 'f_q', 'f_a', 'n_reached',
            'n_reached_after_encounter', 'n_reached_without_encounter' and how many of those by being
            thrown out of the solar system ('n_reached_without_encounter_by_escape'), 'n_left_unbound',
            'p', 'p_low', 'p_high' and 'jupiter_family_shober' (nan, nan, nan and False if no clone
            is bound).
    """

    n_reached, n_clones = countPerihelionReach(q)
    f_q, f_a = calcExcursionFractions(t, q, a)
    n_with, n_without = splitPerihelionReachByEncounter(t, q, encounter_times)

    if n_clones:
        plain = classifyPerihelionReach(n_reached, n_clones, confidence=confidence)

    else:
        plain = {'p': np.nan, 'p_low': np.nan, 'p_high': np.nan, 'jupiter_family_shober': False}

    # Of the clones that reached the limit without an encounter, those that did it by being thrown out
    #   of the solar system, rather than by a slow rise of the perihelion as the Kozai mechanism gives
    elapsed = np.abs(t)
    n_without_by_escape = 0
    for q_clone, times in zip(q, encounter_times):
        beyond = np.flatnonzero(q_clone > JFC_DYNAMICS_Q_LIMIT)
        if beyond.size and np.isinf(q_clone[beyond[0]]) \
                and not np.any(np.abs(np.asarray(times, dtype=float)) <= elapsed[beyond[0]]):
            n_without_by_escape += 1

    return {'n_clones': n_clones, 'f_q': f_q, 'f_a': f_a, 'n_reached': n_reached,
        'n_reached_after_encounter': n_with, 'n_reached_without_encounter': n_without,
        'n_reached_without_encounter_by_escape': n_without_by_escape,
        'n_left_unbound': int(np.count_nonzero(left_unbound)), 'p': plain['p'],
        'p_low': plain['p_low'], 'p_high': plain['p_high'],
        'jupiter_family_shober': plain['jupiter_family_shober']}


def dynamicsOutputTimes(years, output_years):
    """ The output times of a backward integration of the Jupiter-family metrics.

    Arguments:
        years: [float] Span of the integration.
        output_years: [float] Spacing of the outputs.

    Return:
        [ndarray] The times (yr), from 0 to -years. A span too short to leave an output after the
            start has no metrics to give, so it raises a ValueError.
    """

    n_outputs = int(round(years/output_years)) + 1

    if n_outputs < 2:
        raise ValueError("A span of {:g} yr leaves no output after the start with one every {:g} yr, "
            "so there would be nothing to measure.".format(years, output_years))

    return np.linspace(0.0, -years, n_outputs)


def checkDynamicsInputs(traj, batch):
    """ Raise a ValueError if classifyJupiterFamilyDynamics cannot run on these inputs, before any
        integration is started.

    Arguments:
        traj: [Trajectory] The trajectory.
        batch: [int] Clones per batch.
    """

    if (traj.uncertainties is None) or (traj.state_vect_cov is None):
        raise ValueError("The trajectory has no Monte Carlo uncertainties, so there are no clones to "
            "integrate. Solve it with the Monte Carlo runs on.")

    # reboundSimulate draws clones only for more than one realization
    if batch < 2:
        raise ValueError("A batch needs at least 2 clones, got {}.".format(batch))


def classifyJupiterFamilyDynamics(traj, years=JFC_DYNAMICS_YEARS, batch=JFC_DYNAMICS_BATCH,
    max_clones=JFC_DYNAMICS_MAX_CLONES, confidence=0.95, integrator='ias15', gr='gr', beta=None,
    seed=None, output_years=JFC_DYNAMICS_OUTPUT_YEARS, encounter_hill=JFC_DYNAMICS_ENCOUNTER_HILL,
    n_cpu=None, checkpoint_path=None):
    """ Classify whether a meteoroid moved like a Jupiter-family comet, from clones of its orbit
        integrated backwards.

        The clones are drawn from the covariance of the trajectory's state vector, as
        wmpl.Rebound.REBOUND draws its Monte Carlo realizations, and integrated by it. They are run in
        batches until the fraction of them reaching q > 2.5 AU settles the class or max_clones is
        reached: an orbit that is clearly or clearly not Jupiter-family stops after a few tens of
        clones instead of the fixed 1000 of Shober et al. (2024).

        Each batch is a look at the fraction that could stop the run on a fluke, so each look uses the
        interval at a confidence 1 - (1 - confidence)/n_looks (Bonferroni over the possible looks).
        That keeps the chance of stopping on the wrong class at about 1 - confidence or less, about
        because the coverage of the Wilson interval is itself approximate. The fraction is also given
        with its plain interval at the requested confidence.

        Reference: Shober et al. (2024), A&A 687, A181, doi:10.1051/0004-6361/202449635, adapted:
        clones from the full state-vector covariance, the stopping rule above, and the split of the
        clones that reach the limit without a close encounter (splitPerihelionReachByEncounter).

    Arguments:
        traj: [Trajectory] Trajectory solved with the Monte Carlo uncertainties (traj.state_vect_cov).

    Keyword arguments:
        years: [float] Span of the backward integration. Default JFC_DYNAMICS_YEARS.
        batch: [int] Clones per batch, at least 2. Default JFC_DYNAMICS_BATCH.
        max_clones: [int] Most clones integrated, one more if the last batch would otherwise hold a
            single clone. Default JFC_DYNAMICS_MAX_CLONES.
        confidence: [float] Confidence of the verdict. Default 0.95.
        integrator: [str] Integrator of wmpl.Rebound.REBOUND. Default 'ias15'.
        gr: [str] General relativity model of wmpl.Rebound.REBOUND. Default 'gr'.
        beta: [float] Radiation pressure parameter, None (default) for none.
        seed: [int] Seed of the clones; batch k uses seed + k. None draws one, which is returned.
        output_years: [float] Spacing of the outputs. Default JFC_DYNAMICS_OUTPUT_YEARS.
        encounter_hill: [float] Depth of the encounters with Jupiter, in Hill radii, that count as
            lifting a perihelion. Default JFC_DYNAMICS_ENCOUNTER_HILL.
        n_cpu: [int] Processes for the clones. None uses all cores but one.
        checkpoint_path: [str] File to save each batch to as soon as it ends. If it already holds batches
            of a run with the same parameters, they are read back instead of integrated again, with the
            seed they were drawn with if none is given. A batch cut halfway starts again. The file is
            left in place at the end. None (default) saves nothing. See wmpl.Utils.DynamicsCheckpoint.

    Return:
        [dict] The settings (under 'assumptions'), and: 'n_integrated', 'n_clones' (those that count:
            all but the unbound ones), 'n_batches', 'f_q', 'f_a',
            'n_reached', 'n_reached_after_encounter', 'n_reached_without_encounter' and how many of
            those by being thrown out of the solar system ('n_reached_without_encounter_by_escape'),
            'n_left_unbound' (clones unbound at every output after the epoch, see cloneEnsemble),
            'n_escaped' (clones that left the simulation volume), 'n_impacted', 'p', 'p_low' and
            'p_high' (plain interval), 'look_confidence', 'jupiter_family_shober' and
            'dynamics_class' (from the last look; 'unbound' if every clone was unbound), 'n_failed'
            (clones that failed to integrate and are in none of the counts; a RuntimeError if a whole
            batch did) and 'integrator_warning' (the WHFast warning of wmpl.Rebound.REBOUND, or None).
    """

    from wmpl.Rebound import REBOUND as reb

    if not reb.REBOUND_FOUND:
        raise ImportError(reb._REBOUND_IMPORT_ERROR)

    checkDynamicsInputs(traj, batch)

    t = dynamicsOutputTimes(years, output_years)
    n_outputs = len(t)

    if max_clones < 1:
        raise ValueError("At least one clone is needed, got max_clones = {}.".format(max_clones))

    if (seed is None) and (checkpoint_path is not None):
        seed = checkpointSeed(checkpoint_path)

    if seed is None:
        seed = int(np.random.SeedSequence().entropy%(2**32))

    checkpoint = None
    if checkpoint_path is not None:
        checkpoint = DynamicsCheckpoint(checkpoint_path, {'mode': 'adapted', 'years': years,
            'output_years': output_years, 'epoch_jd': traj.jdt_ref, 'integrator': integrator, 'gr': gr,
            'beta': beta, 'seed': seed, 'batch': batch, 'max_clones': max_clones,
            'trajectory': fingerprint([np.asarray(getattr(traj, key, 0.0), dtype=float).tolist()
                for key in ('state_vect_mini', 'v_init', 'radiant_eci_mini', 'state_vect_cov')])})

    n_looks = int(np.ceil(float(max_clones)/batch))
    look_confidence = 1.0 - (1.0 - confidence)/n_looks

    q_parts, a_parts, encounter_times, unbound_parts = [], [], [], []
    n_integrated = 0
    n_escaped = 0
    n_impacted = 0
    n_failed = 0
    n_resumed = 0
    all_diagnostics = {}

    for look in range(n_looks):

        n_batch = max(2, min(batch, max_clones - look*batch))

        key = 'batch_{:d}'.format(look)

        if (checkpoint is not None) and (key in checkpoint.records):
            outputs_mc, diagnostics = checkpoint.records[key]
            n_resumed += len(outputs_mc)

        else:

            _, outputs_mc, diagnostics = reb.reboundSimulate(None, None, traj=traj, direction='backward',
                sim_days=years*365.25, n_outputs=n_outputs, mc_runs=n_batch, random_seed=seed + look,
                integrator=integrator, gr=gr, beta=beta, lean_outputs=True, return_diagnostics=True,
                show_progress=False, n_cpu=n_cpu, skip_nominal=True)

            if (checkpoint is not None) and outputs_mc:
                checkpoint.add(key, (outputs_mc, diagnostics))

        # reboundSimulate skips a clone that failed. With none left there is nothing to say about the
        #   orbit, which is not the same as every clone being unbound
        if not outputs_mc:
            raise RuntimeError("All the {:d} clones of batch {:d} failed to integrate.".format(n_batch,
                look + 1))

        q_batch, a_batch, encounters_batch, unbound_batch = cloneEnsemble(outputs_mc, diagnostics,
            n_outputs, encounter_hill=encounter_hill)

        q_parts.append(q_batch)
        a_parts.append(a_batch)
        unbound_parts.append(unbound_batch)
        encounter_times += encounters_batch
        n_integrated += len(outputs_mc)
        n_failed += n_batch - len(outputs_mc)
        all_diagnostics.update(('{:d}_{:s}'.format(look, name), diagnostics[name]) for name in outputs_mc)
        n_escaped += sum(diagnostics[name]['escaped'] is not None for name in outputs_mc)
        n_impacted += sum(diagnostics[name]['impact'] is not None for name in outputs_mc)

        q = np.vstack(q_parts)
        n_reached, n_clones = countPerihelionReach(q)

        # Every clone so far was unbound: the orbit is not bound, and more clones will not change that
        if n_clones == 0:
            break

        verdict = classifyPerihelionReach(n_reached, n_clones, confidence=look_confidence)

        if verdict['dynamics_class'] != 'undetermined':
            break

    summary = summariseCloneEnsemble(t, q, np.vstack(a_parts), encounter_times,
        np.concatenate(unbound_parts), confidence=confidence)

    # WHFast does not resolve a close encounter, which is what lifts most perihelia here
    integrator_warning = reb.whfastEncounterWarning(all_diagnostics) if integrator == 'whfast' else None

    return dict(summary,
        assumptions={'mode': 'adapted', 'years': years, 'epoch_jd': traj.jdt_ref,
            'integrator': integrator, 'gr': gr, 'beta': beta, 'seed': seed, 'batch': batch,
            'max_clones': max_clones, 'output_years': output_years, 'q_limit': JFC_DYNAMICS_Q_LIMIT,
            'a_limit': JFC_DYNAMICS_A_LIMIT, 'p_threshold': JFC_DYNAMICS_P_THRESHOLD,
            'encounter_hill': encounter_hill, 'confidence': confidence, 'checkpoint': checkpoint_path},
        n_resumed=n_resumed, n_integrated=n_integrated, n_batches=look + 1, n_escaped=n_escaped, n_impacted=n_impacted,
        n_failed=n_failed, integrator_warning=integrator_warning, look_confidence=look_confidence,
        dynamics_class=verdict['dynamics_class'] if summary['n_clones'] else 'unbound')


# The setup of Shober et al. (2024), Sect. 2.4: 1000 clones drawn independently from the published
#   uncertainties of a, e, i, node and peri, integrated 10 000 yr backward with IAS15 under the
#   gravity of the Sun, the planets and the Moon, with 1000 outputs
SHOBER2024_CLONES = 1000
SHOBER2024_YEARS = 10000.0
SHOBER2024_OUTPUT_YEARS = 10.0

# The 1-sigma uncertainties they assumed for the MORP fireballs, which had none listed, used here for a
#   trajectory without Monte Carlo uncertainties. a in AU, the angles in degrees
SHOBER2024_DEFAULT_SIGMA = {'a': 0.01, 'e': 0.001, 'i': 0.1, 'node': 0.01, 'peri': 0.01}

# Distance from the Earth, in its Hill radii, at which the clones start: where wmpl.Rebound.REBOUND
#   considers that the meteoroid has left the Earth
SHOBER2024_START_HILL = 3.0


def elementUncertainties(traj, draw_q=False):
    """ The 1-sigma uncertainties of a, e, i, node and peri of a trajectory's orbit, as Shober et al.
        (2024) drew their clones from.

    Arguments:
        traj: [Trajectory] The trajectory.

    Keyword arguments:
        draw_q: [bool] Take the uncertainty of q instead of that of e, for a trajectory with Monte Carlo
            uncertainties, which has both. Without them it raises a ValueError: the values assumed for MORP do not
            include one. Default False.

    Return:
        (sigma, source): [dict] {'a': AU, 'e' or 'q': AU, 'i': deg, 'node': deg, 'peri': deg}, and [str] where
            they come from: the Monte Carlo runs of the trajectory or, without them, the values
            Shober et al. assumed for the MORP fireballs (SHOBER2024_DEFAULT_SIGMA).
    """

    unc = traj.uncertainties

    if unc is None:

        if draw_q:
            raise ValueError("Drawing q needs its uncertainty, and this trajectory has no Monte Carlo uncertainties.")

        return dict(SHOBER2024_DEFAULT_SIGMA), "assumed by Shober et al. for MORP"

    sigma = {'a': unc.a, 'i': np.degrees(unc.i), 'node': np.degrees(unc.node), 'peri': np.degrees(unc.peri)}

    # Elements given with the error of q instead of that of e. A trajectory has both, and draws e unless asked
    if draw_q or getattr(unc, 'draw_q', False):
        sigma['q'] = unc.q

    else:
        sigma['e'] = unc.e

    return sigma, getattr(unc, 'source', "Monte Carlo of the trajectory")


def drawElementClones(orbit, sigma, n_clones, rng):
    """ Clones of an orbit with a, e, i, node and peri drawn independently from normal distributions,
        as Shober et al. (2024) drew theirs from the diagonal of the uncertainties.

        The anomaly is not drawn: each clone keeps the argument of latitude (peri + f) of the
        orbit, and so its place relative to the node, where the meteoroid met the Earth.

        With a 1-sigma of q instead of one of e, q is drawn and e = 1 - q/a. A perihelion distance is usually known
        far better than a and e, whose errors are strongly correlated, and drawing a and e independently spreads
        it well beyond its error. This departs from the setup of Shober et al.

    Arguments:
        orbit: [Orbit] The orbit of the trajectory (angles in radians).
        sigma: [dict] 1-sigma of 'a' (AU), 'e' or 'q' (AU), 'i', 'node' and 'peri' (deg), see
            elementUncertainties.
        n_clones: [int] Number of clones.
        rng: [numpy.random.Generator] Source of the draws.

    Return:
        [dict] Arrays 'a' (AU), 'e', 'i', 'node', 'peri' and 'f' (deg) of the clones.
    """

    clones = {'a': rng.normal(orbit.a, sigma['a'], n_clones)}

    # In the place of the draws of e, so that with the same seed the other elements do not change
    if 'q' in sigma:
        q = rng.normal(orbit.q, sigma['q'], n_clones)
        clones['e'] = 1.0 - q/clones['a']

        n_invalid = int(np.count_nonzero((q <= 0) | (clones['a'] <= 0) | (clones['e'] < 0)))
        if n_invalid:
            raise ValueError("{:d} of the {:d} clones drawn have a semi-major axis or a perihelion distance that "
                "make no ellipse: the uncertainty of a is too wide for this q.".format(n_invalid, n_clones))

    else:
        clones['e'] = rng.normal(orbit.e, sigma['e'], n_clones)

    for key in ('i', 'node', 'peri'):
        clones[key] = rng.normal(np.degrees(getattr(orbit, key)), sigma[key], n_clones)

    clones['f'] = np.degrees(orbit.peri + orbit.true_anomaly) - clones['peri']

    # Drawn independently, a and e can disagree on the kind of conic: a > 0 with e >= 1, or a < 0
    #   with e < 1, is no orbit
    n_invalid = int(np.count_nonzero(clones['a']*(1.0 - clones['e']) <= 0))
    if n_invalid:
        raise ValueError("{:d} of the {:d} clones drawn have an a and an e that make no conic: the "
            "uncertainties of a and e are too wide to be drawn independently. Use the adapted "
            "mode, which draws the clones from the state-vector covariance.".format(n_invalid,
            n_clones))

    return clones


def elementCloneStates(clones, dt_days):
    """ Heliocentric states of clones given by their elements, moved dt_days back on their orbits
        around the Sun alone.

    Arguments:
        clones: [dict] Return of drawElementClones.
        dt_days: [float] How far back to move them (days).

    Return:
        [list] [x, y, z, vx, vy, vz] of each clone, in AU and AU/day, ecliptic J2000.
    """

    from wmpl.Rebound import REBOUND as reb

    # Around a Sun of one solar mass, as in the integration
    sim = reb.rb.Simulation()
    sim.add(m=1.0)
    for k in range(len(clones['a'])):
        sim.add(primary=sim.particles[0], a=clones['a'][k], e=clones['e'][k],
            inc=np.radians(clones['i'][k]), Omega=np.radians(clones['node'][k]),
            omega=np.radians(clones['peri'][k]), f=np.radians(clones['f'][k]))
    sim.N_active = 1

    sim.integrate(-dt_days/reb.SIM_TIME_DAYS)

    return [[p.x, p.y, p.z, p.vx/reb.SIM_TIME_DAYS, p.vy/reb.SIM_TIME_DAYS, p.vz/reb.SIM_TIME_DAYS]
        for p in sim.particles[1:]]


def addEarthAttraction(states, planets):
    """ Speed up clones that start near the Earth to the geocentric speed they had there.

        The elements are those of the orbit without the Earth's attraction, so a clone moved back on
        that orbit has, relative to the Earth, the geocentric velocity at infinity v_g. At a distance
        d the meteoroid moved faster, at sqrt(v_g^2 + 2 GM/d); a clone started with v_g instead leaves
        the Earth (backward in time) with v_g^2 - 2 GM/d, which changes its orbit by an amount that
        scales as 1/d. The direction of the geocentric velocity is kept.

    Arguments:
        states: [list] Heliocentric [x, y, z, vx, vy, vz] of the clones, AU and AU/day.
        planets: [tuple] (names, barycentric states, masses) in REBOUND units, as returned by
            wmpl.Rebound.REBOUND.planetStatesAtEpoch at the start.

    Return:
        [list] The states with the velocity relative to the Earth lengthened.
    """

    from wmpl.Rebound import REBOUND as reb

    names, planet_states, masses = planets
    earth = np.array(planet_states[names.index("Earth")]) - np.array(planet_states[names.index("Sun")])
    gm = masses[names.index("Earth")]

    corrected = []
    for state in states:
        r_rel = np.array(state[:3]) - earth[:3]
        v_rel = np.array(state[3:])*reb.SIM_TIME_DAYS - earth[3:]
        v_rel *= np.sqrt(1.0 + 2.0*gm/(np.linalg.norm(r_rel)*np.dot(v_rel, v_rel)))
        corrected.append(list(state[:3]) + list((earth[3:] + v_rel)/reb.SIM_TIME_DAYS))

    return corrected


def checkShober2024Inputs(traj):
    """ Raise a ValueError if classifyJupiterFamilyDynamicsShober2024 cannot run on this trajectory,
        before any integration is started.

    Arguments:
        traj: [Trajectory] The trajectory.
    """

    if (traj.orbit is None) or (traj.orbit.a is None) or (traj.orbit.v_g is None):
        raise ValueError("The trajectory has no heliocentric orbit to draw the clones from.")


def classifyJupiterFamilyDynamicsShober2024(traj, n_clones=SHOBER2024_CLONES, years=SHOBER2024_YEARS,
    output_years=SHOBER2024_OUTPUT_YEARS, confidence=0.95, seed=None,
    encounter_hill=JFC_DYNAMICS_ENCOUNTER_HILL, n_cpu=None, show_progress=False, checkpoint_path=None,
    draw_q=False):
    """ Classify whether a meteoroid moved like a Jupiter-family comet with the setup of Shober et al.
        (2024), Sect. 2.4, so that the result can be compared with theirs.

        The clones are drawn from the uncertainties of a, e, i, node and peri independently
        (elementUncertainties, drawElementClones) and integrated 10 000 yr backward with IAS15 under
        Newtonian gravity: the Sun, the planets and the Moon, with no general relativity, no J2 of the
        Sun and no radiation. The Earth's J2 and J4 stay, as wmpl.Rebound.REBOUND always includes
        them. All the clones are integrated, with no early stop.

        The elements are those of the orbit without the Earth's attraction, at the place where the
        meteoroid met the Earth. A clone put there with the Earth in the simulation would feel that
        attraction a second time, so each clone starts SHOBER2024_START_HILL Hill radii before the
        Earth, at the time the meteoroid took to cover that distance at its geocentric velocity, moved
        back to it on its orbit around the Sun alone and given the geocentric speed it had there
        (addEarthAttraction). Shober et al. start theirs at the beginning of the fireball observation,
        and do not say how they treat the Earth's attraction there.

        Reference: Shober et al. (2024), A&A 687, A181, doi:10.1051/0004-6361/202449635.

    Arguments:
        traj: [Trajectory] The trajectory. Without Monte Carlo uncertainties, the clones are drawn
            from the uncertainties Shober et al. assumed for the MORP fireballs.

    Keyword arguments:
        n_clones: [int] Number of clones. Default SHOBER2024_CLONES.
        years: [float] Span of the backward integration. Default SHOBER2024_YEARS.
        output_years: [float] Spacing of the outputs. Default SHOBER2024_OUTPUT_YEARS.
        confidence: [float] Confidence of the interval of the fraction. Default 0.95.
        seed: [int] Seed of the clones. None draws one, which is returned.
        encounter_hill: [float] Depth of the encounters with Jupiter, in Hill radii, that count as
            lifting a perihelion. Default JFC_DYNAMICS_ENCOUNTER_HILL.
        n_cpu: [int] Processes for the clones. None uses all cores but one.
        show_progress: [bool] Print how many clones have been integrated. Default False.
        checkpoint_path: [str] File to save each clone to as soon as it ends. If it already holds clones of
            a run with the same parameters, they are read back instead of integrated again, with the seed
            they were drawn with if none is given. A clone that failed is tried again. The file is left
            in place at the end. None (default) saves nothing. See wmpl.Utils.DynamicsCheckpoint.
        draw_q: [bool] Draw q from its Monte Carlo uncertainty and take e = 1 - q/a, instead of drawing e, which
            departs from the setup of the paper (see drawElementClones). Default False.

    Return:
        [dict] As classifyJupiterFamilyDynamics, with the mode, the uncertainties and the start of the
            clones under 'assumptions'. The class comes from the interval of the fraction at the
            requested confidence (one look). A clone that fails to integrate does not stop the others:
            it is in none of the counts, 'n_failed' counts it and 'failed_clones' holds its error. If
            they all fail, a RuntimeError is raised.
    """

    from wmpl.Rebound import REBOUND as reb

    if not reb.REBOUND_FOUND:
        raise ImportError(reb._REBOUND_IMPORT_ERROR)

    # Fail early, before the hours of integration, if REBOUNDx cannot attach to a simulation
    reb.checkReboundxUsable()

    checkShober2024Inputs(traj)

    t = dynamicsOutputTimes(years, output_years)
    n_outputs = len(t)

    if n_clones < 1:
        raise ValueError("At least one clone is needed, got n_clones = {}.".format(n_clones))

    if (seed is None) and (checkpoint_path is not None):
        seed = checkpointSeed(checkpoint_path)

    if seed is None:
        seed = int(np.random.SeedSequence().entropy%(2**32))

    sigma, sigma_source = elementUncertainties(traj, draw_q=draw_q)
    clones = drawElementClones(traj.orbit, sigma, n_clones, np.random.default_rng(seed))

    dt_days = SHOBER2024_START_HILL*reb.HILL_RADII_AU["Earth"]*reb.AU_M/traj.orbit.v_g/86400.0
    jd_start = reb.utcToTdb(traj.jdt_ref) - dt_days
    planets = reb.planetStatesAtEpoch(jd_start)
    states = addEarthAttraction(elementCloneStates(clones, dt_days), planets)

    times = list(reb.outputTimes(years*365.25, n_outputs, 'backward'))

    names = ['obj_MC_{:d}'.format(k) for k in range(n_clones)]
    kwargs = dict(integrator='ias15', planets=planets, sun_j2=False, gr='none', lean_outputs=True)

    if n_cpu is None:
        n_cpu = max(1, (os.cpu_count() or 1) - 1)

    # Fails here, before the hours of integration, if the file cannot be written or belongs to another run
    checkpoint = None
    if checkpoint_path is not None:
        checkpoint = DynamicsCheckpoint(checkpoint_path, {'mode': 'shober2024', 'years': years,
            'output_years': output_years, 'epoch_jd': traj.jdt_ref, 'seed': seed, 'n_clones': n_clones,
            'states': fingerprint([jd_start, states])})

    outputs_mc, diagnostics, failed = {}, {}, {}

    def keep(result):
        outputs_mc.update(result['outputs'])
        diagnostics.update(result['diagnostics'])

    def collect(name, get_result):
        """ Keep the result of a clone, or its error: one clone should not lose hours of the others. """

        try:
            result = get_result()

        except Exception as err:
            failed[name] = repr(err)

        # Outside the try: a checkpoint that cannot be written is no failure of the clone
        else:
            keep(result)
            if checkpoint is not None:
                checkpoint.add(name, result)

        done = len(outputs_mc) + len(failed)
        if show_progress and (done%max(1, n_clones//20) == 0):
            print("  {:d}/{:d} clones integrated".format(done, n_clones), flush=True)

    pending = list(zip(names, states))
    n_resumed = 0

    if checkpoint is not None:

        for name, _ in pending:
            if name in checkpoint.records:
                keep(checkpoint.records[name])
                n_resumed += 1

        pending = [(name, state) for name, state in pending if name not in checkpoint.records]

        if show_progress and n_resumed:
            print("  {:d}/{:d} clones read back from {:s}".format(n_resumed, n_clones, checkpoint_path),
                flush=True)

    # One clone per simulation, so that each gets its own adaptive timestep
    if n_cpu == 1:
        for name, state in pending:
            collect(name, lambda: reb.integrateHeliocentricBodies([name], [state], jd_start, times, **kwargs))

    elif pending:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_cpu) as executor:
            futures = {executor.submit(reb.integrateHeliocentricBodies, [name], [state], jd_start, times,
                **kwargs): name for name, state in pending}

            # Leaving the pool waits for all that is queued, which may be hours, if a run that stops here
            #   (an interruption, a full disk) does not cancel it
            try:
                for future in concurrent.futures.as_completed(futures):
                    collect(futures[future], future.result)

            except BaseException:
                for future in futures:
                    future.cancel()
                raise

    if not outputs_mc:
        raise RuntimeError("All the {:d} clones failed to integrate. One of the errors: {:s}".format(n_clones,
            next(iter(failed.values()))))

    q, a, encounter_times, left_unbound = cloneEnsemble(outputs_mc, diagnostics, n_outputs,
        encounter_hill=encounter_hill)
    summary = summariseCloneEnsemble(t, q, a, encounter_times, left_unbound, confidence=confidence)

    if summary['n_clones']:
        dynamics_class = classifyPerihelionReach(summary['n_reached'], summary['n_clones'],
            confidence=confidence)['dynamics_class']

    else:
        dynamics_class = 'unbound'

    return dict(summary,
        assumptions={'mode': 'shober2024', 'years': years, 'epoch_jd': traj.jdt_ref,
            'integrator': 'ias15', 'gr': 'none', 'beta': None, 'seed': seed,
            'n_clones': n_clones, 'sigma': sigma, 'sigma_source': sigma_source,
            'start_hill': SHOBER2024_START_HILL, 'start_days_before': dt_days,
            'node_passage': getattr(traj, 'node_passage', None),
            'output_years': output_years, 'q_limit': JFC_DYNAMICS_Q_LIMIT,
            'a_limit': JFC_DYNAMICS_A_LIMIT, 'p_threshold': JFC_DYNAMICS_P_THRESHOLD,
            'encounter_hill': encounter_hill, 'confidence': confidence, 'checkpoint': checkpoint_path},
        n_resumed=n_resumed, n_integrated=len(outputs_mc), n_batches=1,
        n_escaped=sum(diagnostics[name]['escaped'] is not None for name in outputs_mc),
        n_impacted=sum(diagnostics[name]['impact'] is not None for name in outputs_mc),
        n_failed=len(failed), failed_clones=failed, integrator_warning=None, look_confidence=confidence,
        dynamics_class=dynamics_class)


def formatJupiterFamilyDynamics(result):
    """ Lines of the report of classifyJupiterFamilyDynamics: what was assumed, then what came out.

    Arguments:
        result: [dict] Return of classifyJupiterFamilyDynamics or
            classifyJupiterFamilyDynamicsShober2024.

    Return:
        [list] Lines of text.
    """

    s = result['assumptions']
    span = "{:.0f} yr backward from JD {:.5f}, {:s}".format(s['years'], s['epoch_jd'],
        s['integrator'].upper())
    limits = "q > {:g} AU, a > {:g} AU, threshold {:g}, encounters < {:g} R_H of Jupiter".format(
        s['q_limit'], s['a_limit'], s['p_threshold'], s['encounter_hill'])

    if s['mode'] == 'shober2024':

        sigma = s['sigma']
        size = "q {:.3g} AU".format(sigma['q']) if 'q' in sigma else "e {:.3g}".format(sigma['e'])
        lines = [
            "Jupiter-family dynamics (setup of Shober et al. 2024)",
            "-----------------------------------------------------",
            "  Assumptions",
            "    Integration : {:s}, every clone in its own simulation".format(span),
            "    Forces      : Sun, planets and Moon (DE430), Newtonian; Earth J2/J4 (always in "
                "wmpl.Rebound.REBOUND); no GR, Sun J2, radiation pressure, outgassing or Yarkovsky",
            "    Clones      : {:d}, a, {:s}, i, node, peri drawn independently{:s}, seed {:d}; 1-sigma ({:s}): "
                "a {:.3g} AU, {:s}, i {:.3g} deg, node {:.3g} deg, peri {:.3g} deg; argument of "
                "latitude kept".format(s['n_clones'], 'q' if 'q' in sigma else 'e',
                " and e = 1 - q/a, unlike the paper" if 'q' in sigma else "", s['seed'], s['sigma_source'],
                sigma['a'], size, sigma['i'], sigma['node'], sigma['peri']),
            "    Start       : on each clone's orbit without the Earth's attraction, {:g} Hill radii "
                "before the Earth ({:.2f} d before the epoch), at the geocentric speed it had "
                "there".format(s['start_hill'], s['start_days_before']),
            "    Outputs     : every {:g} yr".format(s['output_years']),
            "    Limits      : {:s}".format(limits),
            "  Result",
            ]

        # What a reader can compare with the paper the elements came from
        if s.get('node_passage') is not None:
            lines.insert(-1, "    Orbit       : from the elements given; at the {:s} node, {:.4f} AU from the Earth, "
                "v_g = {:.1f} km/s".format(s['node_passage']['node'], s['node_passage']['miss_au'],
                s['node_passage']['v_g']/1000.0))

    else:

        gr_text = {'gr': 'GR (Sun)', 'gr_full': 'GR (all bodies)', 'none': 'no GR'}[s['gr']]
        radiation = "no radiation pressure" if s['beta'] is None \
            else "radiation pressure and Poynting-Robertson drag (beta = {:.3e})".format(s['beta'])

        lines = [
            "Jupiter-family dynamics (adapted from Shober et al. 2024)",
            "---------------------------------------------------------",
            "  Assumptions",
            "    Integration : {:s}".format(span),
            "    Forces      : Sun, planets and Moon (DE430); {:s}; Sun J2; Earth J2/J4; {:s}; no "
                "outgassing or Yarkovsky".format(gr_text, radiation),
            "    Clones      : {:d} from the trajectory's state-vector covariance, seed {:d}, in batches "
                "of {:d} (at most {:d})".format(result['n_integrated'], s['seed'], s['batch'],
                s['max_clones']),
            "    Outputs     : every {:g} yr".format(s['output_years']),
            "    Limits      : {:s}".format(limits),
            "  Result",
            ]

    # With no clone bound to the Sun there is no fraction to take and no interval to test
    if result['n_clones'] == 0:
        lines.append("    Verdict: {:s}".format(result['dynamics_class']))

    else:
        lines += [
            "    f_q = {:.3f}   f_a = {:.3f}".format(result['f_q'], result['f_a']),
            "    Clones reaching q > {:g} AU: {:d}/{:d}  ({:d} after a Jupiter encounter; {:d} without "
                "one: {:d} thrown out of the solar system, {:d} possible Kozai)".format(s['q_limit'],
                result['n_reached'], result['n_clones'], result['n_reached_after_encounter'],
                result['n_reached_without_encounter'], result['n_reached_without_encounter_by_escape'],
                result['n_reached_without_encounter'] - result['n_reached_without_encounter_by_escape']),
            "    P = {:.2f}  [{:.2f}, {:.2f}] at {:.0%} (Wilson)".format(result['p'], result['p_low'],
                result['p_high'], s['confidence']),
            "    Rule of Shober et al. (P > {:g}): {:s}".format(s['p_threshold'],
                "yes" if result['jupiter_family_shober'] else "no"),
            ]

        if s['mode'] == 'shober2024':
            lines.append("    Verdict: {:s} (from the interval at {:.0%})".format(result['dynamics_class'],
                s['confidence']))

        else:
            lines.append("    Verdict: {:s} (interval at {:.2%} per look, {:d} look{:s})".format(
                result['dynamics_class'], result['look_confidence'], result['n_batches'],
                "" if result['n_batches'] == 1 else "s"))

    if result['n_escaped'] or result['n_impacted']:
        lines.append("    {:d} clone(s) left the simulation volume (1000 AU) and {:d} hit a body".format(
            result['n_escaped'], result['n_impacted']))

    if result['n_left_unbound']:
        lines.append("    {:d} clone(s) were unbound at every output after the epoch and are left out of "
            "the classification".format(result['n_left_unbound']))

    if s.get('checkpoint') is not None:
        lines.append("    Checkpoint: {:s}, {:d} clone(s) read back from it".format(s['checkpoint'],
            result['n_resumed']))

    if result['n_failed']:
        lines.append("    {:d} clone(s) failed to integrate and are left out of the classification, which "
            "would bias it if the failures depend on the orbit".format(result['n_failed']))

        if result.get('failed_clones'):
            name = sorted(result['failed_clones'])[0]
            lines.append("    The error of {:s}: {:s}".format(name, result['failed_clones'][name]))

    if result['integrator_warning']:
        lines.append("    {:s}".format(result['integrator_warning']))

    return lines


if __name__ == "__main__":

    import sys
    import json
    import argparse

    import datetime

    from wmpl.Utils.OrbitFromElements import ELEMENTS_MAX_NODE_MISS, trajectoryFromElements
    from wmpl.Utils.Pickling import loadPickle
    from wmpl.Utils.TrajConversions import datetime2JD

    # The choices of the dynamics options are those of the integrator, so that they cannot drift apart
    from wmpl.Rebound.REBOUND import GR_MODELS, INTEGRATORS, radiationPressureBeta


    ### COMMAND LINE ARGUMENTS

    arg_parser = argparse.ArgumentParser(description="Classify one orbit dynamically. The orbit "
        "is taken either from a trajectory pickle or from orbital elements given on the command "
        "line. To compare two orbits instead, use wmpl.Utils.Dcriteria.")

    arg_parser.add_argument('traj_path', metavar='TRAJ_PATH', type=str, nargs='?', default=None, \
        help="Path to the trajectory pickle of the orbit. Omit it to give the orbit with --q or "
             "--a, --e and --i.")

    elem = arg_parser.add_argument_group('orbit, given by hand',
        "The size of the orbit is given either as a perihelion distance or as a semi-major axis. "
        "The node and the argument of perihelion are needed only for the distances to the giant "
        "planets, and so for the Tancredi classification of an asteroid.")
    elem.add_argument('-q', '--q', metavar='PERIHELION_DIST', type=float, dest='q', \
        help="Perihelion distance of the orbit in AU.")
    elem.add_argument('-a', '--a', metavar='SEMI_MAJOR_AXIS', type=float, dest='a', \
        help="Semi-major axis of the orbit in AU, in place of the perihelion distance.")
    elem.add_argument('-e', '--e', metavar='ECCENTRICITY', type=float, dest='e', \
        help="Eccentricity of the orbit.")
    elem.add_argument('-i', '--i', metavar='INCLINATION', type=float, dest='i', \
        help="Inclination of the orbit (deg).")
    elem.add_argument('-p', '--peri', metavar='ARG_OF_PERI', type=float, dest='peri', \
        help="Argument of perihelion of the orbit (deg).")
    elem.add_argument('-n', '--node', metavar='ASCENDING_NODE', type=float, dest='node', \
        help="Ascending node of the orbit (deg).")

    output = arg_parser.add_argument_group('output')
    output.add_argument('--quiet', '-Q', action='store_true', \
        help="Print one 'KEY VALUE' line per quantity and nothing else, for scripted use.")

    dyn = arg_parser.add_argument_group('Jupiter-family dynamics',
        "Integrate clones of the orbit backwards with wmpl.Rebound.REBOUND and classify whether it "
        "moved like a Jupiter-family comet: the fraction of clones whose perihelion rises above 2.5 AU, "
        "after Shober et al. (2024), A&A 687, A181. Needs rebound/reboundx and the DE430 ephemeris. The "
        "orbit comes from a trajectory pickle or, with shober2024, from the elements with --date. The adapted mode draws the clones from the Monte Carlo state-vector "
        "covariance and integrates them in batches until the class is settled or --max_clones is "
        "reached. The shober2024 mode repeats the setup of the paper: 1000 clones from the "
        "uncertainties of the elements (or, without Monte Carlo, those the paper assumed for MORP), "
        "Newtonian gravity, no early stop.")
    dyn.add_argument('--dynamics', nargs='?', const='adapted', default=None, \
        choices=('adapted', 'shober2024'), metavar='MODE', \
        help="Run the classification, in the adapted mode (the default) or in shober2024.")
    dyn.add_argument('--years', type=float, default=JFC_DYNAMICS_YEARS, \
        help="Span of the backward integration in years. Default: {:g}.".format(JFC_DYNAMICS_YEARS))
    dyn.add_argument('--batch', type=int, default=None, \
        help="Adapted mode: clones per batch, at least 2. Default: {:d}.".format(JFC_DYNAMICS_BATCH))
    dyn.add_argument('--max_clones', type=int, default=None, \
        help="Most clones integrated in the adapted mode (default: {:d}), or the number of clones in "
        "shober2024 (default: {:d}).".format(JFC_DYNAMICS_MAX_CLONES, SHOBER2024_CLONES))
    dyn.add_argument('--integrator', type=str.lower, default=None, choices=INTEGRATORS, \
        help="Adapted mode: integrator. Default: ias15. On a 21P-like test orbit TRACE showed no "
        "detectable difference in the fraction of 40 clones and took only 12%% less time.")
    dyn.add_argument('--gr', type=str.lower, default=None, choices=GR_MODELS, \
        help="Adapted mode: general relativity. Default: gr, the Sun's post-Newtonian correction: on a "
        "21P-like test orbit it showed no detectable difference from gr_full in the fraction of 40 "
        "clones, at 1/4.6 of the cost. none leaves it out.")
    dyn.add_argument('--beta', type=float, default=None, \
        help="Adapted mode: include radiation pressure and Poynting-Robertson drag with this beta. Off "
        "by default.")
    dyn.add_argument('--radius', type=float, default=None, \
        help="Adapted mode: meteoroid radius in metres, with --density, to compute beta instead of "
        "giving it.")
    dyn.add_argument('--density', type=float, default=3000.0, \
        help="Bulk density in kg/m^3 for --radius. Default: 3000.")
    dyn.add_argument('--seed', type=int, default=None, \
        help="Seed of the clones, to reproduce a run. Drawn and reported if not given.")
    dyn.add_argument('--cores', type=int, default=None, \
        help="Processes for the clones. Default: all cores but one.")
    dyn.add_argument('--dynamics_save', metavar='FILE', type=str, default=None, \
        help="Save the result of the classification, with its assumptions, to this JSON file.")
    dyn.add_argument('--date', metavar='DATE', type=str, default=None, \
        help="With the elements given by hand: UTC time of the fireball, as 'YYYY-MM-DD HH:MM:SS'. It says which "
        "node of the orbit the Earth was at, where the clones start. --q or --a, --e, --i, --peri and --node are "
        "all needed.")
    for name, unit in (('a', 'AU'), ('e', ''), ('i', 'deg'), ('node', 'deg'), ('peri', 'deg')):
        dyn.add_argument('--sigma_' + name, metavar='SIGMA', type=float, default=None, \
            help="With --date: 1-sigma of {:s}{:s}. Give all five, or none for the values Shober et al. assumed for "
            "MORP.".format(name, ' ({:s})'.format(unit) if unit else ''))
    dyn.add_argument('--sigma_q', metavar='SIGMA', type=float, default=None, \
        help="With --date, in place of --sigma_e: 1-sigma of q (AU). q is then drawn and e = 1 - q/a, which respects a "
        "perihelion distance known far better than a and e. This departs from the setup of Shober et al.")
    dyn.add_argument('--draw_q', action='store_true', \
        help="With shober2024 and a trajectory pickle solved with the Monte Carlo uncertainties: draw q from its "
        "uncertainty and take e = 1 - q/a, instead of drawing e, as --sigma_q does for elements given by hand. This "
        "departs from the setup of Shober et al. To measure a fireball, the adapted mode already respects these "
        "correlations, since it draws from the full covariance.")
    dyn.add_argument('--max_node_miss', metavar='AU', type=float, default=None, \
        help="With --date: farthest the node may lie from the Earth, or the elements are refused. Default: 0.05 AU; "
        "the elements of three published fireballs give 0.0004 to 0.012 AU.")
    dyn.add_argument('--dynamics_checkpoint', metavar='FILE', type=str, default=None, \
        help="Save each clone (shober2024) or batch (adapted) to this file as soon as it ends. Run the same "
        "command again to go on from there after an interruption, with the same clones. A file from a run "
        "with other parameters is refused. It takes about 60 kB per clone with the outputs of shober2024.")

    cml_args = arg_parser.parse_args()


    #########################


    def reportError(message):
        """ Print a message to stderr and stop.

        Arguments:
            message: [str] what went wrong
        """

        print(message, file=sys.stderr)
        sys.exit(1)


    # Assemble the orbit, either from a trajectory pickle or from the elements given by hand
    if cml_args.traj_path is not None:

        traj = loadPickle(*os.path.split(cml_args.traj_path))
        orbit = traj.orbit
        q, e, incl, node, peri = orbit.q, orbit.e, orbit.i, orbit.node, orbit.peri

    else:

        if (cml_args.q is not None) and (cml_args.a is not None):
            reportError("Give the size of the orbit either as --q or as --a, not as both.")

        if (cml_args.q is None) and (cml_args.a is None):
            reportError("The orbit needs a size, given either as --q or as --a, or a trajectory "
                "pickle to take it from.")

        if (cml_args.e is None) or (cml_args.i is None):
            reportError("The orbit needs an eccentricity and an inclination.")

        if (cml_args.node is None) != (cml_args.peri is None):
            reportError("The node and the argument of perihelion have to be given together.")

        e = cml_args.e
        incl = np.radians(cml_args.i)

        q = cml_args.q if cml_args.q is not None else cml_args.a*(1.0 - e)

        node = None if cml_args.node is None else np.radians(cml_args.node)
        peri = None if cml_args.peri is None else np.radians(cml_args.peri)

    # Every criterion below is a function of the semi-major axis, which an unbound orbit does not
    #   have. Stopping here says so, instead of dividing by zero or by a negative number and
    #   printing values that look like classifications
    if e >= 1.0:
        reportError("The orbit is not bound, and a dynamical class is defined only for a bound "
            "orbit. The eccentricity given is {:.5f}.".format(e))

    a = q/(1.0 - e)

    verbose = not cml_args.quiet


    if verbose:

        print("Orbit:")

        if cml_args.traj_path is not None:
            print("  from {:s}".format(cml_args.traj_path))

        print("  q = {:.5f} AU".format(q))
        print("  e = {:.5f}".format(e))
        print("  i = {:.5f} deg".format(np.degrees(incl)))

        if peri is not None:
            print("  w = {:.5f} deg".format(np.degrees(peri)))
            print("  O = {:.5f} deg".format(np.degrees(node)))

        print("  a = {:.5f} AU".format(a))


    ### DYNAMICAL PARAMETERS

    # (key for the quiet output, label, value, format). The energy is a few times 1e-5 over the
    #   whole range of interest, so a fixed point format would print it as zero
    parameters = [
        ('T_J', 'Tisserand parameter T_J', float(calcTisserand(a, e, incl)), '{:14.6f}'),
        ('K', 'Whipple K', float(calcWhippleK(a, e)), '{:14.6f}'),
        ('P', 'Kresak P (yr)', float(calcKresakP(a, e)), '{:14.6f}'),
        ('Q', 'Aphelion Q (AU)', float(calcAphelionDistance(a, e)), '{:14.6f}'),
        ('E', 'Orbital energy E (AU^2/day^2)', float(calcOrbitalEnergy(a)), '{:14.6e}'),
        ]

    if verbose:
        print()
        print("Dynamical parameters")
        print("--------------------")

    for key, label, value, fmt in parameters:
        print(("  {:<30s} " + fmt).format(label, value) if verbose \
            else "{:s} {:s}".format(key, fmt.format(value).strip()))


    ### TWO-PARAMETER COMETARY CLASSIFICATION

    if verbose:
        print()
        print("Jopek & Williams (2013) two-parameter classification")
        print("----------------------------------------------------")

    for label, test in (("Q-i", isCometaryQi), ("E-i", isCometaryEi), ("T-i", isCometaryTi),
                        ("P-i", isCometaryPi), ("K-i", isCometaryKi)):

        verdict = "cometary" if bool(test(a, e, incl)) else "asteroidal"

        print("  {:<30s} {:>14s}".format(label, verdict) if verbose \
            else "{:s} {:s}".format(label, verdict))


    ### TANCREDI CLASSIFICATION

    if verbose:
        print()
        print("Tancredi (2014) classification")
        print("------------------------------")

    comet_class = classifyTancrediComet(a, e, incl)

    print("  {:<30s} {:>14s}".format("comet class", comet_class) if verbose \
        else "TANCREDI_COMET {:s}".format(comet_class))

    # The asteroid scheme needs how close the orbit comes to each giant planet, which needs the
    #   orientation of the orbit in its plane
    if peri is not None:

        moids = calcGiantPlanetMOIDs(a, e, incl, node, peri)

        for name, _, _, _, _, _, _ in GIANT_PLANETS:

            print("  {:<30s} {:14.4f}".format("MOID " + name.capitalize() + " (Hill radii)",
                moids[name]) if verbose \
                else "MOID_{:s} {:.4f}".format(name.upper(), moids[name]))

        asteroid_class = classifyTancrediAsteroid(a, e, incl, moids['jupiter'],
            min(moids.values()))

        print("  {:<30s} {:>14s}".format("asteroid class", asteroid_class) if verbose \
            else "TANCREDI_ASTEROID {:s}".format(asteroid_class))

    elif verbose:
        print()
        print("The Tancredi classification of an asteroid was skipped: it needs the distances to")
        print("the giant planets, and so the node and the argument of perihelion, from a")
        print("trajectory pickle or from the --node and --peri arguments.")


    ### JUPITER-FAMILY DYNAMICS

    if cml_args.dynamics is not None:

        element_options = ['date', 'max_node_miss', 'sigma_q'] + ['sigma_' + name for name in SHOBER2024_DEFAULT_SIGMA]
        given = [name for name in element_options if getattr(cml_args, name) is not None]

        if (cml_args.traj_path is not None) and given:

            if given == ['sigma_q']:
                reportError("--sigma_q goes with the elements given by hand, not with a trajectory pickle. With a pickle "
                    "solved with the Monte Carlo uncertainties, --draw_q takes the uncertainty of q from it.")

            reportError("--{:s} goes with the elements given by hand, not with a trajectory pickle.".format(given[0]))

        if cml_args.draw_q and ((cml_args.traj_path is None) or (cml_args.dynamics != 'shober2024')):
            reportError("--draw_q goes with --dynamics shober2024 and a trajectory pickle. With the elements given by "
                "hand, use --sigma_q. The adapted mode already respects the correlations of the orbit.")

        if cml_args.traj_path is None:

            if cml_args.dynamics != 'shober2024':
                reportError("The adapted mode of --dynamics draws the clones from the covariance of a trajectory "
                    "pickle. With the elements given by hand, use --dynamics shober2024.")

            if (peri is None) or (cml_args.date is None):
                reportError("--dynamics with the elements given by hand needs --q or --a, --e, --i, --peri, --node "
                    "and --date.")

            sigma = {name: getattr(cml_args, 'sigma_' + name) for name in SHOBER2024_DEFAULT_SIGMA}

            if cml_args.sigma_q is not None:

                if sigma.pop('e') is not None:
                    reportError("Give either --sigma_e or --sigma_q, not both.")

                sigma['q'] = cml_args.sigma_q

            n_given = sum(value is not None for value in sigma.values())

            if n_given not in (0, len(sigma)):
                reportError("Give all of --sigma_a, --sigma_e (or --sigma_q), --sigma_i, --sigma_node and "
                    "--sigma_peri, or none.")

            try:
                jd_utc = datetime2JD(datetime.datetime.fromisoformat(cml_args.date))
            except ValueError:
                reportError("--date has to be given as 'YYYY-MM-DD HH:MM:SS', got '{:s}'.".format(cml_args.date))

            try:
                traj = trajectoryFromElements(a, e, np.degrees(incl), np.degrees(node), np.degrees(peri), jd_utc,
                    sigma=sigma if n_given else None, max_node_miss=ELEMENTS_MAX_NODE_MISS
                    if cml_args.max_node_miss is None else cml_args.max_node_miss)

            except (ImportError, ValueError, OSError, RuntimeError) as err:
                reportError("--dynamics: {:s}".format(str(err)))

        if cml_args.dynamics == 'shober2024':

            # The paper's setup fixes what these would change
            given = [name for name in ('batch', 'integrator', 'gr', 'beta', 'radius')
                if getattr(cml_args, name) is not None]
            if given:
                reportError("--dynamics shober2024 integrates as Shober et al. did (IAS15, Newtonian, "
                    "no radiation, no batches); leave out {:s}.".format(", ".join("--" + name
                    for name in given)))

            n_clones = SHOBER2024_CLONES if cml_args.max_clones is None else cml_args.max_clones

            try:
                checkShober2024Inputs(traj)
            except ValueError as err:
                reportError("--dynamics: {:s}".format(str(err)))

            if verbose:
                print()
                print("Integrating {:d} clones of {:.0f} yr each for the Jupiter-family dynamics with the "
                    "setup of Shober et al. (2024); this takes hours...".format(n_clones, cml_args.years))

            try:
                result = classifyJupiterFamilyDynamicsShober2024(traj, n_clones=n_clones,
                    years=cml_args.years, seed=cml_args.seed, n_cpu=cml_args.cores,
                    show_progress=verbose, checkpoint_path=cml_args.dynamics_checkpoint,
                    draw_q=cml_args.draw_q)

            except (ImportError, ValueError, OSError, RuntimeError) as err:
                reportError("--dynamics: {:s}".format(str(err)))

        else:

            batch = JFC_DYNAMICS_BATCH if cml_args.batch is None else cml_args.batch
            max_clones = JFC_DYNAMICS_MAX_CLONES if cml_args.max_clones is None else cml_args.max_clones

            if (cml_args.beta is not None) and (cml_args.radius is not None):
                reportError("Give either --beta or --radius (with --density), not both.")

            beta = cml_args.beta
            if cml_args.radius is not None:
                beta = radiationPressureBeta(cml_args.radius, cml_args.density)

            try:
                checkDynamicsInputs(traj, batch)
            except ValueError as err:
                reportError("--dynamics: {:s}".format(str(err)))

            if verbose:
                print()
                print("Integrating clones for the Jupiter-family dynamics, up to {:d} of {:.0f} yr each; "
                    "this takes minutes to hours...".format(max_clones, cml_args.years))

            try:
                result = classifyJupiterFamilyDynamics(traj, years=cml_args.years, batch=batch,
                    max_clones=max_clones,
                    integrator='ias15' if cml_args.integrator is None else cml_args.integrator,
                    gr='gr' if cml_args.gr is None else cml_args.gr, beta=beta, seed=cml_args.seed,
                    n_cpu=cml_args.cores, checkpoint_path=cml_args.dynamics_checkpoint)

            except (ImportError, ValueError, OSError, RuntimeError) as err:
                reportError("--dynamics: {:s}".format(str(err)))

        if verbose:
            print()
            print("\n".join(formatJupiterFamilyDynamics(result)))

        else:
            for key in ('n_clones', 'n_reached', 'p', 'p_low', 'p_high', 'f_q', 'f_a', 'n_failed'):
                print("JFC_DYNAMICS_{:s} {}".format(key.upper(), result[key]))
            print("JFC_DYNAMICS_CLASS {:s}".format(result['dynamics_class']))

        if cml_args.dynamics_save is not None:
            with open(cml_args.dynamics_save, 'w') as f:
                json.dump(result, f, indent=2, default=float)
