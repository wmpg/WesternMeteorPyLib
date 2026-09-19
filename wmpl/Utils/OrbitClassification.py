""" Dynamical classification of a single orbit.

These criteria take one orbit and return a dynamical class, rather than comparing two orbits, so
they are kept apart from the dissimilarity criteria in wmpl.Utils.Dcriteria.

All functions take angles in radians. The single-parameter criteria and the two-parameter
cometary tests accept scalars or numpy arrays; the Tancredi (2014) classifiers return a label and
take scalars only.
"""

from __future__ import print_function, division, absolute_import

import numpy as np
import scipy.integrate
import scipy.optimize


# Semi-major axis of Jupiter [AU]
A_JUPITER = 5.20336

# Inclination above which an orbit is taken to be cometary in the two-parameter criteria of
#   Jopek & Williams (2013) [rad]
JW_INCL_LIMIT = np.radians(75.0)

# Cometary limits of the one-parameter criteria, as adopted by Jopek & Williams (2013)
JW_APHELION_LIMIT = 4.6      # aphelion distance [AU]
JW_KRESAK_P_LIMIT = 2.5      # Kresak P [yr]
JW_ENERGY_LIMIT = -5.28e-5   # orbital energy -k^2/(2a) [AU^2/day^2]

# Gaussian gravitational constant [AU^1.5/day] and its square, the solar gravitational
#   parameter in units of AU, day and solar masses [AU^3/day^2]
GAUSS_K = 0.01720209895
GAUSS_K_SQUARED = GAUSS_K**2

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
