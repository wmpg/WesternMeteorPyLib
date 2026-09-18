""" Tests for the additional orbit dissimilarity criteria, the orbit classification criteria and
    the threshold methods.

Each criterion is checked either against a value tabulated in the paper that defines it, or, where
the paper tabulates none, against an analytic identity that the criterion has to satisfy by
construction: vanishing on identical orbits, symmetry, the triangle inequality for the metrics,
invariance under the rotations the quotient metrics divide out, and the inequalities that relate a
reduced criterion to the full one it comes from.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_DcriteriaAdditional.py -v

or standalone (no pytest required):
    python -m wmpl.Utils.Tests.test_DcriteriaAdditional
"""

import inspect
import math

import numpy as np

from wmpl.Utils.Dcriteria import (calcRho1, calcRho2, calcRho5, calcC, calcDR, calcDB, calcDT,
    calcDX, calcDVJopek, calcDVWeights, calcDACS, calcDSAC, calcDN, calcDD, DV_DISPERSIONS,
    DV_THRESHOLDS, DV_DEFAULT_EPOCH, TC_REFERENCE_A, TC_REFERENCE_E,
    TC_REFERENCE_INCL, TC_REFERENCE_Q, DACS_A_SCALE, DX_W_SOL, DX_W_RA, DX_W_DEC, DX_W_VG)
from wmpl.Utils.OrbitClassification import (calcTisserand, calcKresakK, calcKresakP,
    calcAphelionDistance, calcOrbitalEnergy, isCometaryQi, isCometaryKi, isCometaryPi,
    isCometaryEi, GAUSS_K, calcHillRadius, calcResonanceSemiMajorAxis, classifyTancrediComet,
    classifyTancrediAsteroid, isTancrediResonanceProtected, calcMOID, calcGiantPlanetMOIDs,
    calcResonanceWidth, calcMaxPerihelionForTisserand, calcMinMOIDForTisserand,
    _orbitPosition, A_JUPITER, A_URANUS, JW_INCL_LIMIT, JW_ENERGY_LIMIT, GAUSS_K_SQUARED,
    TANCREDI_RESONANCES, TANCREDI_T_HIGH, Q_JUPITER_APHELION, RESONANCE_E_MIN, RESONANCE_E_MAX)


# Orbits of comet 96P/Machholz 1 and asteroid 2003 EH1 at the 7415 BC epoch, from table 2 of
#   Kholshevnikov et al. (2016). The paper tabulates rho_1 = 0.036, rho_2 = 0.035, rho_5 = 0.016
#   for this pair
KHOLSHEVNIKOV_ORBIT_1 = (0.622, 0.795, math.radians(65.82), math.radians(98.56),
    math.radians(13.28))
KHOLSHEVNIKOV_ORBIT_2 = (0.615, 0.798, math.radians(65.03), math.radians(97.78),
    math.radians(15.55))
KHOLSHEVNIKOV_RHO_1 = 0.036
KHOLSHEVNIKOV_RHO_2 = 0.035
KHOLSHEVNIKOV_RHO_5 = 0.016

# The paper quotes the elements to 3 or 4 significant figures and the metrics to 3 decimals, which
#   together limit the agreement that can be reached. Every plausible transcription error of the
#   formulae was measured to miss by at least 0.0099, five times this tolerance
KHOLSHEVNIKOV_TOL = 0.002

# A metric compared with itself cannot return exactly zero: its square cancels down to the rounding
#   level of about 1e-16 and the square root lifts that to about 1e-8. Measured worst case over
#   5000 random orbits: rho_1 3.7e-08, rho_2 6.0e-08, rho_5 exactly 0
SELF_COMPARISON_TOL = 1e-7


def _randomOrbits(n, random_state=0, e_min=0.01):
    """ Draw random orbits spanning the physical parameter ranges.

    Arguments:
        n: [int] number of orbits

    Keyword arguments:
        random_state: [int] seed
        e_min: [float] smallest eccentricity drawn

    Return:
        [list] tuples of (q, e, i, O, w), angles in radians
    """

    rng = np.random.RandomState(random_state)

    q = rng.uniform(0.05, 1.5, n)
    e = rng.uniform(e_min, 0.99, n)
    i = np.arccos(rng.uniform(-1, 1, n))
    O = rng.uniform(0, 2*np.pi, n)
    w = rng.uniform(0, 2*np.pi, n)

    return list(zip(q, e, i, O, w))


def _randomRadiants(n, random_state=0):
    """ Draw random geocentric radiants and speeds.

    Arguments:
        n: [int] number of radiants

    Keyword arguments:
        random_state: [int] seed

    Return:
        [list] tuples of (ra, dec, sol, vg), angles in radians and vg in km/s
    """

    rng = np.random.RandomState(random_state)

    ra = rng.uniform(0, 2*np.pi, n)
    dec = np.arcsin(rng.uniform(-1, 1, n))
    sol = rng.uniform(0, 2*np.pi, n)
    vg = rng.uniform(11.5, 71.0, n)

    return list(zip(ra, dec, sol, vg))


### Kholshevnikov metrics ###

def test_rho_reproduces_kholshevnikov_table():
    """ The three metrics must reproduce the values tabulated for 96P/Machholz 1 and 2003 EH1 at
        7415 BC in Kholshevnikov et al. (2016).
    """

    args = KHOLSHEVNIKOV_ORBIT_1 + KHOLSHEVNIKOV_ORBIT_2

    for name, func, expected in (("rho_1", calcRho1, KHOLSHEVNIKOV_RHO_1),
                                 ("rho_2", calcRho2, KHOLSHEVNIKOV_RHO_2),
                                 ("rho_5", calcRho5, KHOLSHEVNIKOV_RHO_5)):

        got = float(func(*args))

        assert abs(got - expected) < KHOLSHEVNIKOV_TOL, \
            "{:s} gave {:.5f}, the paper tabulates {:.3f}".format(name, got, expected)


def test_rho_metrics_satisfy_triangle_inequality():
    """ The distinguishing property of the Kholshevnikov metrics: unlike D_SH, D_D and D_H they are
        true metrics, so the triangle inequality must hold for every triple of orbits.
    """

    orbits = _randomOrbits(60, random_state=11)

    for name, func in (("rho_1", calcRho1), ("rho_2", calcRho2), ("rho_5", calcRho5)):

        worst = 0.0

        for a in range(len(orbits)):
            for b in range(len(orbits)):
                for c in range(len(orbits)):

                    d_ab = float(func(*(orbits[a] + orbits[b])))
                    d_bc = float(func(*(orbits[b] + orbits[c])))
                    d_ac = float(func(*(orbits[a] + orbits[c])))

                    # Record the largest violation, allowing for rounding
                    worst = max(worst, d_ac - (d_ab + d_bc))

        assert worst < 1e-9, \
            "{:s} violated the triangle inequality by {:.4e}".format(name, worst)


def test_DD_violates_triangle_inequality():
    """ The control for the test above. D_D is a pseudometric, so a triple of orbits that breaks
        the triangle inequality exists; without showing one, the triangle test on the metrics would
        not demonstrate that they do anything the existing criteria do not.

        The triple below was found by maximising the violation over the physical parameter box,
        restricted to inclinations between 10 and 170 deg and eccentricities between 0.1 and 0.9 so
        that it is not a degenerate orbit. rho_2 satisfies the inequality on the very same triple.
    """

    orbit_1 = (0.955, 0.348, math.radians(168.20), math.radians(184.34), math.radians(29.66))
    orbit_2 = (0.959, 0.441, math.radians(168.33), math.radians(186.31), math.radians(153.67))
    orbit_3 = (0.965, 0.900, math.radians(168.27), math.radians(184.55), math.radians(209.86))

    d_13 = calcDD(*(orbit_1 + orbit_3))
    d_12 = calcDD(*(orbit_1 + orbit_2))
    d_23 = calcDD(*(orbit_2 + orbit_3))

    assert d_13 > d_12 + d_23, \
        "expected D_D to break the triangle inequality on this triple, got {:.6f} <= {:.6f}".format(
            d_13, d_12 + d_23)

    rho_13 = float(calcRho2(*(orbit_1 + orbit_3)))
    rho_12 = float(calcRho2(*(orbit_1 + orbit_2)))
    rho_23 = float(calcRho2(*(orbit_2 + orbit_3)))

    assert rho_13 <= rho_12 + rho_23, \
        "rho_2 broke the triangle inequality on the same triple: {:.6f} > {:.6f}".format(rho_13,
            rho_12 + rho_23)


def test_rho_vanishes_on_identical_orbits_and_is_symmetric():
    """ Any metric must satisfy rho(x, x) = 0 and rho(x, y) = rho(y, x). """

    orbits = _randomOrbits(200, random_state=2)

    for name, func in (("rho_1", calcRho1), ("rho_2", calcRho2), ("rho_5", calcRho5)):

        worst_self = 0.0
        worst_asym = 0.0

        for k in range(len(orbits) - 1):

            worst_self = max(worst_self, abs(float(func(*(orbits[k] + orbits[k])))))

            d_ab = float(func(*(orbits[k] + orbits[k + 1])))
            d_ba = float(func(*(orbits[k + 1] + orbits[k])))
            worst_asym = max(worst_asym, abs(d_ab - d_ba))

        assert worst_self < SELF_COMPARISON_TOL, \
            "{:s} returned {:.4e} for an orbit compared with itself".format(name, worst_self)
        assert worst_asym < 1e-12, \
            "{:s} was asymmetric by {:.4e}".format(name, worst_asym)


def test_rho5_is_a_lower_bound_of_rho2():
    """ rho_5 is by definition the minimum of rho_2 over both nodes and both arguments of
        perihelion, so it can never exceed rho_2 for the same pair of orbits.
    """

    orbits = _randomOrbits(300, random_state=3)

    worst = 0.0

    for k in range(len(orbits) - 1):

        args = orbits[k] + orbits[k + 1]
        worst = max(worst, float(calcRho5(*args)) - float(calcRho2(*args)))

    assert worst < 1e-12, "rho_5 exceeded rho_2 by {:.4e}".format(worst)


def test_rho5_attains_the_minimum_of_rho2():
    """ The same definition, checked the other way: scanning rho_2 over the two nodes and the two
        arguments of perihelion must not find a value below rho_5, and must approach it.
    """

    q1, e1, i1 = 0.6, 0.8, math.radians(65.0)
    q2, e2, i2 = 0.5, 0.7, math.radians(40.0)

    rho5 = float(calcRho5(q1, e1, i1, 0.0, 0.0, q2, e2, i2, 0.0, 0.0))

    grid = np.linspace(0, 2*np.pi, 60, endpoint=False)

    smallest = np.inf
    for O1 in grid:
        for w1 in grid:
            for w2 in grid:
                # Only the node difference matters, so the second node can stay fixed
                smallest = min(smallest,
                    float(calcRho2(q1, e1, i1, O1, w1, q2, e2, i2, 0.0, w2)))

    assert smallest > rho5 - 1e-9, \
        "rho_2 dipped {:.4e} below rho_5".format(rho5 - smallest)
    assert smallest - rho5 < 5e-3, \
        "the rho_2 scan bottomed out at {:.6f}, rho_5 claims {:.6f}".format(smallest, rho5)


def test_rho5_invariant_under_node_and_perihelion_rotation():
    """ rho_5 is a metric on the quotient space in which the node and the argument of perihelion
        are divided out, so changing either must not change it.
    """

    rng = np.random.RandomState(9)
    orbits = _randomOrbits(100, random_state=8)

    worst = 0.0

    for k in range(len(orbits) - 1):

        q1, e1, i1, O1, w1 = orbits[k]
        q2, e2, i2, O2, w2 = orbits[k + 1]

        reference = float(calcRho5(q1, e1, i1, O1, w1, q2, e2, i2, O2, w2))

        rotated = float(calcRho5(q1, e1, i1, rng.uniform(0, 2*np.pi), rng.uniform(0, 2*np.pi),
            q2, e2, i2, rng.uniform(0, 2*np.pi), rng.uniform(0, 2*np.pi)))

        worst = max(worst, abs(rotated - reference))

    assert worst < 1e-12, "rho_5 changed by {:.4e} under a node or apsidal rotation".format(worst)


def test_rho_defined_for_circular_orbits():
    """ The motivation for the metrics: D_D divides by (e1 + e2) and so fails outright for two
        circular orbits, while the metrics stay finite.
    """

    circular_1 = (1.0, 0.0, math.radians(5.0), math.radians(30.0), math.radians(0.0))
    circular_2 = (1.2, 0.0, math.radians(7.0), math.radians(35.0), math.radians(0.0))

    for name, func in (("rho_1", calcRho1), ("rho_2", calcRho2), ("rho_5", calcRho5)):

        value = float(func(*(circular_1 + circular_2)))

        assert np.isfinite(value), "{:s} was not finite for two circular orbits".format(name)

    try:
        calcDD(*(circular_1 + circular_2))
        raise AssertionError("expected D_D to fail for two circular orbits")

    except ZeroDivisionError:
        pass


def test_rho_accepts_arrays():
    """ The new criteria must work on numpy arrays as well as on scalars. """

    orbits = _randomOrbits(50, random_state=4)

    q1, e1, i1, O1, w1 = [np.array(c) for c in zip(*orbits[:25])]
    q2, e2, i2, O2, w2 = [np.array(c) for c in zip(*orbits[25:])]

    for name, func in (("rho_1", calcRho1), ("rho_2", calcRho2), ("rho_5", calcRho5),
                       ("C", calcC)):

        vector = func(q1, e1, i1, O1, w1, q2, e2, i2, O2, w2)

        assert vector.shape == (25,), "{:s} did not return one value per pair".format(name)

        for k in range(25):

            scalar = float(func(q1[k], e1[k], i1[k], O1[k], w1[k], q2[k], e2[k], i2[k], O2[k],
                w2[k]))

            assert abs(vector[k] - scalar) < 1e-12, \
                "{:s} disagreed between array and scalar calls".format(name)


### Neslusan C ###

def test_C_matches_the_angular_momentum_term_of_rho1():
    """ C is the length of the difference of the two angular momentum vectors, which is exactly the
        first term of rho_1. Comparing the two pins down the vector convention in both.
    """

    orbits = _randomOrbits(200, random_state=6)

    worst = 0.0

    for k in range(len(orbits) - 1):

        q1, e1, i1, O1, w1 = orbits[k]
        q2, e2, i2, O2, w2 = orbits[k + 1]

        c_value = float(calcC(q1, e1, i1, O1, w1, q2, e2, i2, O2, w2))

        # Strip the eccentricity vector term out of rho_1 to leave the angular momentum term
        rho1 = float(calcRho1(q1, e1, i1, O1, w1, q2, e2, i2, O2, w2))
        cos_P = (rho1**2 - c_value**2 - e1**2 - e2**2)/(-2*e1*e2)

        worst = max(worst, abs(cos_P) - 1.0)

        expected = math.sqrt(max(rho1**2 - (e1**2 + e2**2 - 2*e1*e2*cos_P), 0.0))

        assert abs(c_value - expected) < 1e-9, \
            "C and the rho_1 angular momentum term disagreed: {:.8f} vs {:.8f}".format(c_value,
                expected)

    assert worst < 1e-9, "the recovered cos P left the valid range by {:.4e}".format(worst)


def test_C_vanishes_on_coplanar_orbits_of_equal_semilatus_rectum():
    """ C compares only the angular momentum vectors, so two orbits sharing a plane and a semi
        latus rectum must be at zero distance however their apsides are oriented.
    """

    q1, e1 = 0.4, 0.6
    p = q1*(1 + e1)

    e2 = 0.9
    q2 = p/(1 + e2)

    i, O = math.radians(23.0), math.radians(77.0)

    value = float(calcC(q1, e1, i, O, math.radians(10.0), q2, e2, i, O, math.radians(250.0)))

    assert value < 1e-12, "C gave {:.4e} for two orbits with the same angular momentum".format(
        value)


### Valsecchi D_R ###

def test_DR_is_a_lower_bound_of_DN():
    """ D_R is D_N with the non-negative angular term dropped, so it can never exceed D_N. """

    radiants = _randomRadiants(400, random_state=7)

    worst = 0.0

    for k in range(len(radiants) - 1):

        args = radiants[k] + radiants[k + 1]

        d_r = float(calcDR(*args))
        d_n = calcDN(*args)

        worst = max(worst, d_r - d_n)

    assert worst < 1e-12, "D_R exceeded D_N by {:.4e}".format(worst)


def test_DR_vanishes_on_identical_radiants():
    """ D_R must be zero for a radiant compared with itself. """

    radiants = _randomRadiants(200, random_state=12)

    worst = 0.0

    for radiant in radiants:
        worst = max(worst, abs(float(calcDR(*(radiant + radiant)))))

    assert worst < 1e-12, "D_R returned {:.4e} for a radiant compared with itself".format(worst)


### Jenniskens D_B and D_T ###

def test_DB_vanishes_on_identical_orbits_and_is_symmetric():
    """ D_B must be zero for an orbit compared with itself, and independent of the order. """

    orbits = _randomOrbits(200, random_state=13)

    worst_self = 0.0
    worst_asym = 0.0

    for k in range(len(orbits) - 1):

        _, e1, i1, O1, w1 = orbits[k]
        _, e2, i2, O2, w2 = orbits[k + 1]

        worst_self = max(worst_self, abs(float(calcDB(e1, i1, O1, w1, e1, i1, O1, w1))))

        d_ab = float(calcDB(e1, i1, O1, w1, e2, i2, O2, w2))
        d_ba = float(calcDB(e2, i2, O2, w2, e1, i1, O1, w1))
        worst_asym = max(worst_asym, abs(d_ab - d_ba))

    assert worst_self < 1e-12, "D_B returned {:.4e} for an orbit compared with itself".format(
        worst_self)
    assert worst_asym < 1e-12, "D_B was asymmetric by {:.4e}".format(worst_asym)


def test_DB_C3_uses_the_smallest_angular_difference():
    """ C3 = omega + Omega is an angle, so a pair straddling 0/360 deg must be compared the short
        way round. Two orbits with C3 of 350 deg and 10 deg are 20 deg apart, not 340 deg.
    """

    e, i = 0.7, math.radians(20.0)

    # Same eccentricity and inclination, so only the C3 term contributes
    d_straddling = float(calcDB(e, i, math.radians(350.0), 0.0, e, i, math.radians(10.0), 0.0))
    d_equivalent = float(calcDB(e, i, math.radians(10.0), 0.0, e, i, math.radians(30.0), 0.0))

    assert abs(d_straddling - d_equivalent) < 1e-12, \
        "D_B gave {:.6f} across the 0/360 deg cut but {:.6f} for the same 20 deg separation".format(
            d_straddling, d_equivalent)

    # 20 deg over the published 14.2 deg normalisation
    expected = 20.0/14.2

    assert abs(d_straddling - expected) < 1e-12, \
        "the C3 term gave {:.6f}, expected {:.6f}".format(d_straddling, expected)


def test_DT_matches_the_tisserand_difference():
    """ D_T is the absolute difference of the two Tisserand parameters. The paper writes the
        parameter in q and e rather than a, so this pins that form against calcTisserand.
    """

    orbits = [(2.215, 0.848, math.radians(11.70)), (2.234, 0.862, math.radians(4.18)),
              (3.037, 0.795, math.radians(65.82)), (1.5, 0.3, math.radians(2.0))]

    for a1, e1, i1 in orbits:
        for a2, e2, i2 in orbits:

            expected = abs(float(calcTisserand(a1, e1, i1)) - float(calcTisserand(a2, e2, i2)))
            got = float(calcDT(a1*(1.0 - e1), e1, i1, a2*(1.0 - e2), e2, i2))

            assert abs(got - expected) < 1e-12, \
                "D_T gave {:.8f}, expected {:.8f}".format(got, expected)


def test_DT_reproduces_the_jenniskens_tisserand_values():
    """ Table 1 of Jenniskens (2008) tabulates the Tisserand parameter of each candidate parent
        body, so the parameter itself is checked against the table and D_T is checked to be the
        difference of two such values.

        D_T is an absolute difference and cannot recover a Tisserand parameter on its own, so the
        values are taken from calcTisserand rather than reconstructed by offsetting D_T from a
        reference, which would only be valid for orbits above the reference.
    """

    # (name, a, e, i in deg, T_J tabulated in the paper)
    cases = [("2P/Encke", 2.215, 0.848, 11.70, 3.03),
             ("3200 Phaethon", 1.2712, 0.8898, 22.26, 4.51)]

    for name, a, e, incl_deg, published in cases:

        got = float(calcTisserand(a, e, math.radians(incl_deg)))

        assert abs(got - published) < 0.01, \
            "T_J of {:s} came out as {:.4f}, table 1 gives {:.2f}".format(name, got, published)

    # D_T between the two must be the difference of the tabulated values
    (_, a1, e1, i1, t1), (_, a2, e2, i2, t2) = cases

    d_value = float(calcDT(a1*(1.0 - e1), e1, math.radians(i1), a2*(1.0 - e2), e2,
        math.radians(i2)))

    assert abs(d_value - abs(t1 - t2)) < 0.01, \
        "D_T between Encke and Phaethon is {:.4f}, the tabulated values differ by {:.2f}".format(
            d_value, abs(t1 - t2))


def test_DT_stays_finite_at_unit_eccentricity():
    """ The reason the paper writes the Tisserand parameter in q and e: at e = 1 the semi-major
        axis diverges and the (a, e, i) form evaluates to nan, while this form does not.
    """

    d_value = float(calcDT(0.5, 1.0, 0.0, 0.5, 0.9, 0.0))

    assert np.isfinite(d_value), "D_T was not finite for a parabolic orbit"

    # The (a, e, i) form gives nan for the same orbit
    assert not np.isfinite(float(calcTisserand(np.inf, 1.0, 0.0))), \
        "expected the semi-major axis form to fail at e = 1"


### Rudawska D_X ###

def test_DX_vanishes_on_identical_radiants():
    """ D_X must be zero for a radiant compared with itself, for any weights. """

    radiants = _randomRadiants(200, random_state=14)

    worst = 0.0

    for ra, dec, sol, vg in radiants:
        worst = max(worst, abs(float(calcDX(ra, dec, sol, vg, ra, dec, sol, vg))))

    assert worst < 1e-12, "D_X returned {:.4e} for a radiant compared with itself".format(worst)


def test_DX_weights_scale_the_terms():
    """ Each weight must scale only its own term, so zeroing three of the four must leave the
        remaining term alone and the whole must be the quadrature sum of the parts.
    """

    args = (math.radians(45.0), math.radians(20.0), math.radians(100.0), 30.0,
            math.radians(50.0), math.radians(25.0), math.radians(105.0), 35.0)

    parts = []
    for k in range(4):
        weights = [0.0, 0.0, 0.0, 0.0]
        weights[k] = 1.0
        parts.append(float(calcDX(*(args + tuple(weights)))))

    total = float(calcDX(*(args + (1.0, 1.0, 1.0, 1.0))))
    expected = math.sqrt(sum(p**2 for p in parts))

    assert abs(total - expected) < 1e-12, \
        "D_X gave {:.8f}, the quadrature sum of its terms is {:.8f}".format(total, expected)

    for k, part in enumerate(parts):
        assert part > 0.0, "term {:d} of D_X vanished for two different radiants".format(k)


# Mean geocentric parameters and their dispersions for three established showers, from table 2 of
#   Rudawska et al. (2015), with the mean D_X the paper quotes for each:
#   (code, sol, d_sol, ra, d_ra, dec, d_dec, vg, d_vg, mean D_X)
RUDAWSKA_SHOWERS = [
    ("GEM", 261.0, 1.8, 112.7, 2.4, 32.4, 1.1, 33.53, 1.47, 0.06),
    ("PER", 139.4, 4.2,  46.6, 6.3, 57.6, 1.8, 58.20, 1.67, 0.09),
    ("ORI", 208.1, 3.2,  95.3, 2.7, 15.6, 1.0, 65.44, 1.40, 0.07),
    ]


def test_DX_weights_are_the_published_values():
    """ The weights are taken from the paper, not chosen here. """

    assert (DX_W_SOL, DX_W_RA, DX_W_DEC, DX_W_VG) == (0.17, 1.20, 1.20, 0.20), \
        "the D_X weights are not the published ones"


def test_DX_reproduces_the_quoted_shower_means():
    """ The paper quotes the mean D_X within the Geminids, Perseids and Orionids. Drawing members
        from each shower's mean and its tabulated dispersions reproduces those means, which checks
        the formula and the overall scale of the weights end to end.

        This is a check on the formula and the scale, not on the four weights individually: unit
        weights are rejected, but some other weight sets of similar magnitude are not.
    """

    for code, sol, d_sol, ra, d_ra, dec, d_dec, vg, d_vg, published in RUDAWSKA_SHOWERS:

        rng = np.random.RandomState(7)

        values = []
        for _ in range(20000):

            values.append(float(calcDX(math.radians(ra), math.radians(dec), math.radians(sol), vg,
                math.radians(ra + rng.normal(0, d_ra)),
                math.radians(dec + rng.normal(0, d_dec)),
                math.radians(sol + rng.normal(0, d_sol)),
                vg + rng.normal(0, d_vg))))

        got = float(np.mean(values))

        assert abs(got - published) < 0.02, \
            "mean D_X of the {:s} came out as {:.3f}, the paper quotes {:.2f}".format(code, got,
                published)


def test_DX_separates_the_taurid_branches():
    """ The paper merges groups only when D_X <= 0.15, and reports that its method keeps the
        Southern and Northern Taurids apart. Their tabulated mean parameters must therefore give a
        D_X above the merge threshold.
    """

    # Table 2: ra, dec, sol, vg for showers 002 STA and 017 NTA
    sta = (48.8, 13.2, 217.0, 27.56)
    nta = (54.4, 22.1, 224.5, 27.98)

    for first, second in ((sta, nta), (nta, sta)):

        d_value = float(calcDX(math.radians(first[0]), math.radians(first[1]),
            math.radians(first[2]), first[3], math.radians(second[0]), math.radians(second[1]),
            math.radians(second[2]), second[3]))

        assert d_value > 0.15, \
            "D_X between the Taurid branches came out as {:.4f}, below the merge threshold".format(
                d_value)



### Jopek D_V ###

def test_DVJopek_vanishes_on_identical_orbits_and_is_symmetric():
    """ D_V must be zero for an orbit compared with itself, and independent of the order. """

    orbits = _randomOrbits(200, random_state=15)

    w_h, w_e, w_E = [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], 1.0

    worst_self = 0.0
    worst_asym = 0.0

    for k in range(len(orbits) - 1):

        a, b = orbits[k], orbits[k + 1]

        worst_self = max(worst_self, abs(float(calcDVJopek(*(a + a), w_h=w_h, w_e=w_e, w_E=w_E))))

        d_ab = float(calcDVJopek(*(a + b), w_h=w_h, w_e=w_e, w_E=w_E))
        d_ba = float(calcDVJopek(*(b + a), w_h=w_h, w_e=w_e, w_E=w_E))
        worst_asym = max(worst_asym, abs(d_ab - d_ba))

    assert worst_self < 1e-12, "D_V returned {:.4e} for an orbit compared with itself".format(
        worst_self)
    assert worst_asym < 1e-12, "D_V was asymmetric by {:.4e}".format(worst_asym)


def test_DVJopek_energy_term_matches_the_semimajor_axis():
    """ The energy is written in terms of q and e to stay finite as e approaches 1. It must equal
        -1/(2a), which is what isolating the energy term of D_V recovers.
    """

    orbits = _randomOrbits(200, random_state=16)

    worst = 0.0

    for k in range(len(orbits) - 1):

        q1, e1, i1, O1, w1 = orbits[k]
        q2, e2, i2, O2, w2 = orbits[k + 1]

        # Zero every weight but the energy one, and undo the published factor of 2
        isolated = float(calcDVJopek(q1, e1, i1, O1, w1, q2, e2, i2, O2, w2,
            w_h=[0.0, 0.0, 0.0], w_e=[0.0, 0.0, 0.0], w_E=0.5))

        a1 = q1/(1.0 - e1)
        a2 = q2/(1.0 - e2)
        expected = GAUSS_K**2*abs(-1.0/(2*a1) + 1.0/(2*a2))

        worst = max(worst, abs(isolated - expected))

    assert worst < 1e-12, "the D_V energy term was off by {:.4e}".format(worst)


# Table 1 of Asher, Clube & Steel (1993): the Earth-crossing asteroids with the smallest D_ACS
#   against the Taurid Complex core, as (name, a [AU], e, observed i [deg], tabulated D)
ACS_TABLE_1 = [
    ("1991 AQ",           2.16, 0.77,  3, 0.05),
    ("(2212) Hephaistos", 2.16, 0.84, 12, 0.06),
    ("1984 KB",           2.22, 0.76,  5, 0.07),
    ("(2101) Adonis",     1.87, 0.76,  1, 0.10),
    ("1991 TB2",          2.40, 0.84,  9, 0.10),
    ("1990 SM",           2.16, 0.78, 12, 0.12),
    ("(2201) Oljato",     2.18, 0.71,  3, 0.12),
    ("(5143) 1991 VL",    1.83, 0.77,  9, 0.13),
    ("(4197) 1982 TA",    2.30, 0.77, 12, 0.14),
    ("1991 BA",           2.24, 0.68,  2, 0.16),
    ("(4341) Poseidon",   1.84, 0.68, 12, 0.16),
    ("(4486) Mithra",     2.20, 0.66,  3, 0.17),
    ("1990 TG1",          2.48, 0.69,  9, 0.18),
    ("1988 VP4",          2.26, 0.65, 12, 0.19),
    ("1991 GO",           1.96, 0.66, 10, 0.19),
    ("(4183) Cuno",       1.98, 0.64,  7, 0.19),
    ("1990 HA",           2.58, 0.69,  4, 0.20),
    ("1983 VA",           2.61, 0.69, 16, 0.21),
    ("1983 LC",           2.63, 0.71,  2, 0.22),
    ("1991 EE",           2.25, 0.62, 10, 0.22),
    ("(4179) Toutatis",   2.51, 0.64,  0, 0.23),
    ("1991 CB1",          1.69, 0.62, 16, 0.24),
    ("6344 P-L",          2.62, 0.64,  5, 0.25),
    ("1937 UB Hermes",    1.64, 0.62,  6, 0.25),
    ("1991 XA",           2.27, 0.57,  5, 0.26),
    ("P/Encke",           2.22, 0.85, 12, 0.04),
    ]


def _dacsSemiMajorAxisTerm(a):
    """ The semi-major axis term of D_ACS against the Taurid Complex reference orbit.

    Arguments:
        a: [float] semi-major axis (AU)

    Return:
        [float] contribution of the semi-major axis term to D_ACS
    """

    return abs(TC_REFERENCE_A - a)/DACS_A_SCALE


def test_DACS_semimajor_axis_term_bounds_table1():
    """ The paper adjusts the inclination, and the eccentricity, by secular perturbation theory
        before evaluating D_ACS, while table 1 reports observed values, so the tabulated D cannot be
        recomputed from the table alone. The semi-major axis is not adjusted, though, and the other
        two terms are non-negative, so its term is a rigorous lower bound on every tabulated D.

        This bounds the scale of the semi-major axis term from below: a smaller scale would break
        the bound.
    """

    for name, a, _, _, d_table in ACS_TABLE_1:

        lower_bound = _dacsSemiMajorAxisTerm(a)

        assert lower_bound <= d_table + 1e-9, \
            "{:s}: the semi-major axis term {:.4f} exceeds the tabulated D of {:.2f}".format(name,
                lower_bound, d_table)


def test_DACS_semimajor_axis_term_bound_is_attained():
    """ The companion of the test above, bounding the scale from above. Two objects, P/Encke and
        1991 TB2, sit exactly on the bound, so their adjusted eccentricity and inclination coincide
        with the reference orbit. A larger scale would leave the bound slack everywhere.

        Together the two tests fix the scale at 3 AU from the table alone, with no fitting.
    """

    ratios = [(_dacsSemiMajorAxisTerm(a)/d_table, name) for name, a, _, _, d_table in ACS_TABLE_1]

    largest = max(ratio for ratio, _ in ratios)

    assert abs(largest - 1.0) < 1e-9, \
        "the largest semi-major axis term to D ratio is {:.4f}, so the bound is not attained".format(
            largest)

    attaining = sorted(name for ratio, name in ratios if abs(ratio - 1.0) < 1e-9)

    assert attaining == ["1991 TB2", "P/Encke"], \
        "expected P/Encke and 1991 TB2 on the bound, found {!s}".format(attaining)


def test_DACS_encke_matches_the_reference_orbit():
    """ P/Encke's tabulated D of 0.04 is exactly its semi-major axis term, so evaluating D_ACS with
        the reference eccentricity and inclination and Encke's own semi-major axis must reproduce
        it. The Taurid Complex core orbit is essentially Encke's.
    """

    d_value = float(calcDACS(TC_REFERENCE_A, TC_REFERENCE_E, TC_REFERENCE_INCL,
        2.22, TC_REFERENCE_E, TC_REFERENCE_INCL))

    assert abs(d_value - 0.04) < 1e-9, \
        "D_ACS for P/Encke came out as {:.6f}, the paper tabulates 0.04".format(d_value)


def test_DACS_table1_is_ordered_by_the_criterion():
    """ Table 1 is sorted by D, and the paper's selection is by D alone, so no row may fall below an
        earlier one. This checks the transcription of the table rather than the criterion.
    """

    tabulated = [d_table for name, _, _, _, d_table in ACS_TABLE_1 if name != "P/Encke"]

    assert tabulated == sorted(tabulated), "table 1 is not in ascending order of D"


def test_DACS_scale_makes_three_au_contribute_unity():
    """ The semi-major axis term is normalised by 3 AU, so that difference alone gives D = 1. """

    d_value = float(calcDACS(2.1, 0.5, 0.0, 2.1 + DACS_A_SCALE, 0.5, 0.0))

    assert abs(d_value - 1.0) < 1e-12, \
        "a difference of {:.1f} AU in a gave D = {:.12f}".format(DACS_A_SCALE, d_value)

    assert DACS_A_SCALE == 3.0, "the published scale is 3 AU"


def test_DACS_vanishes_on_identical_orbits_and_is_symmetric():
    """ D_ACS must be zero for an orbit compared with itself, and independent of the order. """

    orbits = _randomOrbits(200, random_state=17)

    worst_self = 0.0
    worst_asym = 0.0

    for k in range(len(orbits) - 1):

        q1, e1, i1, _, _ = orbits[k]
        q2, e2, i2, _, _ = orbits[k + 1]

        a1 = q1/(1.0 - e1)
        a2 = q2/(1.0 - e2)

        worst_self = max(worst_self, abs(float(calcDACS(a1, e1, i1, a1, e1, i1))))

        d_ab = float(calcDACS(a1, e1, i1, a2, e2, i2))
        d_ba = float(calcDACS(a2, e2, i2, a1, e1, i1))
        worst_asym = max(worst_asym, abs(d_ab - d_ba))

    assert worst_self < 1e-12, "D_ACS returned {:.4e} for an orbit compared with itself".format(
        worst_self)
    assert worst_asym < 1e-12, "D_ACS was asymmetric by {:.4e}".format(worst_asym)


def test_DACS_ignores_the_node_and_perihelion_argument():
    """ D_ACS takes no node or argument of perihelion, by design, so it cannot be affected by them.
        This is what makes it suitable for the Taurid Complex, which is spread in longitude of
        perihelion, and what makes a longitude-bearing criterion unsuitable.
    """

    # The signature admits no angles beyond the inclination, so the property is structural
    try:
        names = list(inspect.signature(calcDACS).parameters)

    except AttributeError:
        names = list(inspect.getargspec(calcDACS).args)

    assert names == ["a1", "e1", "i1", "a2", "e2", "i2"], \
        "D_ACS takes {!s}, expected only a, e and i for both orbits".format(names)


def test_DACS_accepts_arrays():
    """ D_ACS must work elementwise on numpy arrays. """

    a = np.array([a for _, a, _, _, _ in ACS_TABLE_1])
    e = np.array([e for _, _, e, _, _ in ACS_TABLE_1])
    incl = np.radians([i for _, _, _, i, _ in ACS_TABLE_1])

    vector = calcDACS(TC_REFERENCE_A, TC_REFERENCE_E, TC_REFERENCE_INCL, a, e, incl)

    assert vector.shape == a.shape

    for k in range(len(a)):

        scalar = float(calcDACS(TC_REFERENCE_A, TC_REFERENCE_E, TC_REFERENCE_INCL, a[k], e[k],
            incl[k]))

        assert abs(vector[k] - scalar) < 1e-12



def test_DSAC_reference_orbit_agrees_with_the_semimajor_axis_form():
    """ The two forms of the Taurid criterion quote their reference orbit differently, one by
        perihelion distance and one by semi-major axis, but both must describe the same physical
        orbit. This is the only cross-check available for the perihelion form, since the 1993 paper
        tabulates D for the semi-major axis form only and no table exists for this one.
    """

    implied_q = TC_REFERENCE_A*(1.0 - TC_REFERENCE_E)

    assert abs(implied_q - TC_REFERENCE_Q) < 0.005, \
        "a1(1 - e1) = {:.4f} AU but the published q1 is {:.3f} AU".format(implied_q,
            TC_REFERENCE_Q)

    # The agreement is limited by the two significant figures of a1 and the two decimals of e1
    widest = 2.15*(1.0 - 0.815)
    narrowest = 2.05*(1.0 - 0.825)

    assert narrowest <= TC_REFERENCE_Q <= widest, \
        "q1 = {:.3f} AU falls outside [{:.4f}, {:.4f}] AU implied by the quoted a1 and e1".format(
            TC_REFERENCE_Q, narrowest, widest)


def test_DSAC_perihelion_term_is_unscaled():
    """ The perihelion form carries no scale factor, unlike the semi-major axis form, so a
        difference of 1 AU in q contributes exactly 1 to D. This is what distinguishes the two.
    """

    d_value = float(calcDSAC(0.375, 0.5, 0.0, 1.375, 0.5, 0.0))

    assert abs(d_value - 1.0) < 1e-12, \
        "a 1 AU difference in q gave D = {:.12f}".format(d_value)

    # The same difference in a is divided by 3 AU
    d_scaled = float(calcDACS(2.1, 0.5, 0.0, 3.1, 0.5, 0.0))

    assert abs(d_scaled - 1.0/DACS_A_SCALE) < 1e-12, \
        "a 1 AU difference in a gave D = {:.12f}".format(d_scaled)


def test_DSAC_and_DACS_share_their_eccentricity_and_inclination_terms():
    """ The two forms differ only in their first term, so removing it from each must leave the same
        remainder. This pins the shared part of both expressions at once.
    """

    orbits = _randomOrbits(300, random_state=18)

    worst = 0.0

    for k in range(len(orbits) - 1):

        q1, e1, i1, _, _ = orbits[k]
        q2, e2, i2, _, _ = orbits[k + 1]

        a1 = q1/(1.0 - e1)
        a2 = q2/(1.0 - e2)

        d_sac = float(calcDSAC(q1, e1, i1, q2, e2, i2))
        d_acs = float(calcDACS(a1, e1, i1, a2, e2, i2))

        remainder_sac = d_sac**2 - (q1 - q2)**2
        remainder_acs = d_acs**2 - ((a1 - a2)/DACS_A_SCALE)**2

        worst = max(worst, abs(remainder_sac - remainder_acs))

    assert worst < 1e-9, \
        "the shared eccentricity and inclination terms differed by {:.4e}".format(worst)


def test_DSAC_vanishes_on_identical_orbits_and_is_symmetric():
    """ D must be zero for an orbit compared with itself, and independent of the order. """

    orbits = _randomOrbits(200, random_state=19)

    worst_self = 0.0
    worst_asym = 0.0

    for k in range(len(orbits) - 1):

        q1, e1, i1, _, _ = orbits[k]
        q2, e2, i2, _, _ = orbits[k + 1]

        worst_self = max(worst_self, abs(float(calcDSAC(q1, e1, i1, q1, e1, i1))))

        d_ab = float(calcDSAC(q1, e1, i1, q2, e2, i2))
        d_ba = float(calcDSAC(q2, e2, i2, q1, e1, i1))
        worst_asym = max(worst_asym, abs(d_ab - d_ba))

    assert worst_self < 1e-12, "D_SAC returned {:.4e} for an orbit compared with itself".format(
        worst_self)
    assert worst_asym < 1e-12, "D_SAC was asymmetric by {:.4e}".format(worst_asym)


def test_DSAC_takes_only_the_elements_the_command_line_supplies():
    """ The perihelion form works in the same variables as D_SH, D_D and D_H, so it needs nothing
        the rest of the module does not already have.
    """

    try:
        names = list(inspect.signature(calcDSAC).parameters)

    except AttributeError:
        names = list(inspect.getargspec(calcDSAC).args)

    assert names == ["q1", "e1", "i1", "q2", "e2", "i2"], \
        "D_SAC takes {!s}, expected only q, e and i for both orbits".format(names)


def test_DSAC_accepts_arrays():
    """ D_SAC must work elementwise on numpy arrays. """

    q = np.array([a*(1.0 - e) for _, a, e, _, _ in ACS_TABLE_1])
    e = np.array([e for _, _, e, _, _ in ACS_TABLE_1])
    incl = np.radians([i for _, _, _, i, _ in ACS_TABLE_1])

    vector = calcDSAC(TC_REFERENCE_Q, TC_REFERENCE_E, TC_REFERENCE_INCL, q, e, incl)

    assert vector.shape == q.shape

    for k in range(len(q)):

        scalar = float(calcDSAC(TC_REFERENCE_Q, TC_REFERENCE_E, TC_REFERENCE_INCL, q[k], e[k],
            incl[k]))

        assert abs(vector[k] - scalar) < 1e-12



def test_DV_weights_follow_from_the_published_dispersions():
    """ The weights are the reciprocal squared dispersions of table 1, w = (2*sigma)^-2. """

    for epoch in DV_DISPERSIONS:

        sigma_h, sigma_e, sigma_energy = DV_DISPERSIONS[epoch]
        w_h, w_e, w_energy = calcDVWeights(epoch)

        for k in range(3):
            assert abs(w_h[k] - 1.0/(2*sigma_h[k])**2) < 1e-6*w_h[k]
            assert abs(w_e[k] - 1.0/(2*sigma_e[k])**2) < 1e-6*w_e[k]

        assert abs(w_energy - 1.0/(2*sigma_energy)**2) < 1e-6*w_energy

    # The paper used the set for a stream 4000 years after formation
    assert DV_DEFAULT_EPOCH == 4000

    try:
        calcDVWeights(1234)
        raise AssertionError("expected a ValueError for an epoch with no published dispersions")

    except ValueError:
        pass


def test_DV_weights_make_two_sigma_contribute_unity():
    """ What fixes the scale of D_V: a pair differing by twice the dispersion in one element alone
        contributes exactly 1 to the sum. This is what makes the published thresholds meaningful.
    """

    _, _, w_energy = calcDVWeights()
    _, _, sigma_energy = DV_DISPERSIONS[DV_DEFAULT_EPOCH]

    assert abs(w_energy*(2*sigma_energy)**2 - 1.0) < 1e-12


def test_DV_dispersions_grow_as_a_stream_ages():
    """ A stream spreads as it evolves, so every dispersion in table 1 must grow with its age, and
        the sporadic background must be looser still than any of them.
    """

    ages = sorted(k for k in DV_DISPERSIONS if k != 'sporadic')

    for index in range(3):

        series = [DV_DISPERSIONS[age][0][index] for age in ages]
        assert series == sorted(series), "angular momentum dispersion {:d} did not grow".format(index)

        series = [DV_DISPERSIONS[age][1][index] for age in ages]
        assert series == sorted(series), "eccentricity dispersion {:d} did not grow".format(index)

    energies = [DV_DISPERSIONS[age][2] for age in ages]
    assert energies == sorted(energies), "the energy dispersion did not grow"

    sporadic_h, sporadic_e, sporadic_energy = DV_DISPERSIONS['sporadic']
    oldest_h, oldest_e, oldest_energy = DV_DISPERSIONS[ages[-1]]

    assert all(sporadic_h[k] > oldest_h[k] for k in range(3))
    assert all(sporadic_e[k] > oldest_e[k] for k in range(3))
    assert sporadic_energy > oldest_energy


def test_DV_thresholds_sit_just_above_the_spread_within_a_stream():
    """ Table 2 is headed "D_V x 10^-1", so the published thresholds are ten times the printed
        figures. Read that way they sit just above the separation between two members of one
        stream, which is what an association threshold has to do. Read literally they would fall an
        order of magnitude below it and admit nothing.
    """

    sigma_h, sigma_e, sigma_energy = DV_DISPERSIONS[DV_DEFAULT_EPOCH]
    w_h, w_e, w_energy = calcDVWeights()

    rng = np.random.RandomState(0)

    separations = []
    for _ in range(20000):

        # Two members each drawn about the mean orbit, so they differ by sqrt(2) times a dispersion
        d_h = [rng.normal(0, s)*math.sqrt(2) for s in sigma_h]
        d_e = [rng.normal(0, s)*math.sqrt(2) for s in sigma_e]
        d_energy = rng.normal(0, sigma_energy)*math.sqrt(2)

        separations.append(math.sqrt(
            w_h[0]*d_h[0]**2 + w_h[1]*d_h[1]**2 + 1.5*w_h[2]*d_h[2]**2
            + w_e[0]*d_e[0]**2 + w_e[1]*d_e[1]**2 + w_e[2]*d_e[2]**2
            + 2*w_energy*d_energy**2))

    separations = np.array(separations)
    threshold = DV_THRESHOLDS[15]

    assert (separations < threshold).mean() > 0.9, \
        "only {:.1%} of same-stream pairs fall under the threshold".format(
            (separations < threshold).mean())

    assert (separations < threshold/10.0).mean() < 0.01, \
        "the printed figures taken literally would admit {:.1%} of same-stream pairs".format(
            (separations < threshold/10.0).mean())

    # The threshold loosens as the smallest accepted group grows
    sizes = sorted(DV_THRESHOLDS)
    assert [DV_THRESHOLDS[m] for m in sizes] == sorted(DV_THRESHOLDS[m] for m in sizes)


def test_DV_uses_astronomical_units_and_days():
    """ The weights are dimensional, so D_V is only meaningful with the angular momentum in
        AU^2/day and the energy in AU^2/day^2, which means both carry the Gaussian constant.
    """

    q, e, incl = 0.5, 0.7, math.radians(20.0)
    node, peri = math.radians(100.0), math.radians(200.0)

    # Isolate the angular momentum by zeroing the other weights. For two coplanar orbits sharing a
    #   node the difference is along the pole and equals k*(sqrt(p1) - sqrt(p2))
    q2 = 0.8
    isolated = float(calcDVJopek(q, e, 0.0, node, peri, q2, e, 0.0, node, peri,
        w_h=[0.0, 0.0, 1.0/1.5], w_e=[0.0, 0.0, 0.0], w_E=0.0))

    expected = GAUSS_K*abs(math.sqrt(q*(1 + e)) - math.sqrt(q2*(1 + e)))

    assert abs(isolated - expected) < 1e-12, \
        "the angular momentum term gave {:.8e}, expected {:.8e}".format(isolated, expected)



### Orbit classification ###

def test_tisserand_is_three_for_a_planet_crossing_circular_orbit():
    """ For an orbit matching the planet's, T = a_p/a + 2 cos i sqrt((a/a_p)(1 - e^2)) reduces to
        1 + 2 = 3 exactly, which is the boundary the criterion is built around.
    """

    value = float(calcTisserand(A_JUPITER, 0.0, 0.0))

    assert abs(value - 3.0) < 1e-12, "T was {:.12f} for an orbit identical to Jupiter's".format(
        value)


def test_tisserand_reproduces_encke():
    """ 2P/Encke has a Tisserand parameter with respect to Jupiter of about 3.03, which places it
        just on the asteroidal side of the boundary despite being a comet.
    """

    value = float(calcTisserand(2.215, 0.848, math.radians(11.70)))

    assert abs(value - 3.03) < 0.01, "T of 2P/Encke came out as {:.4f}, expected about 3.03".format(
        value)


def test_kresak_and_aphelion_boundaries():
    """ The one-parameter criteria must switch at their published limits: K at zero, which is
        a(1 + e)/(1 - e) = 10, P at 2.5 yr and Q at 4.6 AU.
    """

    # K = 0 exactly when a(1 + e)/(1 - e) = 10
    e = 0.5
    a = 10.0*(1.0 - e)/(1.0 + e)

    assert abs(float(calcKresakK(a, e))) < 1e-12, \
        "K was {:.4e} at its boundary".format(float(calcKresakK(a, e)))

    # P = a^1.5 e
    assert abs(float(calcKresakP(4.0, 0.25)) - 2.0) < 1e-12

    # Q = a(1 + e)
    assert abs(float(calcAphelionDistance(2.0, 0.8)) - 3.6) < 1e-12


def test_jopek_williams_two_parameter_criteria():
    """ Each two-parameter criterion must call an orbit cometary when either its own parameter or
        the inclination crosses the limit, and asteroidal only when neither does.
    """

    low_incl = math.radians(10.0)
    high_incl = JW_INCL_LIMIT + math.radians(1.0)

    # Q = 2.0*(1 + 0.5) = 3.0 AU, below the 4.6 AU limit
    assert not isCometaryQi(2.0, 0.5, low_incl)
    assert isCometaryQi(2.0, 0.5, high_incl)

    # Q = 3.0*(1 + 0.8) = 5.4 AU, above the limit
    assert isCometaryQi(3.0, 0.8, low_incl)

    # P = 1.5^1.5*0.2 = 0.367 yr, below the 2.5 yr limit
    assert not isCometaryPi(1.5, 0.2, low_incl)
    assert isCometaryPi(1.5, 0.2, high_incl)
    assert isCometaryPi(4.0, 0.9, low_incl)

    # K of a low, nearly circular orbit is negative
    assert not isCometaryKi(1.5, 0.1, low_incl)
    assert isCometaryKi(1.5, 0.1, high_incl)
    assert isCometaryKi(3.0, 0.9, low_incl)

    # E-i reduces to a cut on the semi-major axis alone, at k^2/(2*|E_limit|)
    a_limit = GAUSS_K_SQUARED/(-2*JW_ENERGY_LIMIT)

    assert abs(float(calcOrbitalEnergy(a_limit)) - JW_ENERGY_LIMIT) < 1e-18, \
        "the energy at the limiting semi-major axis is not the published limit"

    assert not isCometaryEi(a_limit*0.99, 0.5, low_incl)
    assert isCometaryEi(a_limit*1.01, 0.5, low_incl)
    assert isCometaryEi(a_limit*0.99, 0.5, high_incl)


def test_classification_accepts_arrays():
    """ The classification criteria must work elementwise on numpy arrays. """

    a = np.array([2.0, 3.0, 1.5])
    e = np.array([0.5, 0.8, 0.2])
    i = np.array([math.radians(10.0), math.radians(10.0), JW_INCL_LIMIT + 0.1])

    result = isCometaryQi(a, e, i)

    assert result.tolist() == [False, True, True], "Q-i gave {!s} on arrays".format(result)


# Table 1 of Tancredi (2014): the mean-motion resonances with Jupiter and the semi-major axis at
#   the centre of each, as (label, semi-major axis in AU)
TANCREDI_TABLE_1 = [('4:1', 2.065), ('3:1', 2.502), ('5:2', 2.825), ('7:3', 2.958),
    ('2:1', 3.278), ('3:2', 3.971), ('4:3', 4.295), ('1:1', 5.203)]

# Hill radii of the giant planets quoted in section 2.3, as (name, a in AU, M_sun/M_planet, R_H)
TANCREDI_HILL_RADII = [('Jupiter', 5.203, 1047.3486, 0.355), ('Saturn', 9.5826, 3497.898, 0.436),
    ('Uranus', 19.2018, 22902.98, 0.469), ('Neptune', 30.0470, 19412.24, 0.776)]


def test_tancredi_resonance_semimajor_axes_match_table1():
    """ Eq. 3 must reproduce the semi-major axis at the centre of every resonance in table 1. """

    published = dict(TANCREDI_TABLE_1)

    for label, p_plus_q, p, _ in TANCREDI_RESONANCES:

        got = float(calcResonanceSemiMajorAxis(p, p_plus_q))

        assert abs(got - published[label]) < 1e-3, \
            "the {:s} resonance came out at {:.4f} AU, table 1 gives {:.3f}".format(label, got,
                published[label])


def test_tancredi_hill_radii_match_the_paper():
    """ Eq. 2 must reproduce the Hill radii quoted for the four giant planets. """

    for name, a_planet, inv_mass, published in TANCREDI_HILL_RADII:

        got = float(calcHillRadius(a_planet, 1.0/inv_mass))

        assert abs(got - published) < 2e-3, \
            "the Hill radius of {:s} came out as {:.4f} AU, the paper gives {:.3f}".format(name,
                got, published)


def test_tancredi_classifies_la_sagra_as_a_comet_in_an_asteroidal_orbit():
    """ The paper singles out 233P/La Sagra as a "Comet" in an Asteroidal Orbit it had newly
        identified, and quotes its orbit and Tisserand parameter, so both are checked.
    """

    a, e, incl = 3.037, 0.409, math.radians(10.1)

    tisserand = float(calcTisserand(a, e, incl))

    assert abs(tisserand - 3.086) < 1e-3, \
        "T_Jup of 233P/La Sagra came out as {:.4f}, the paper gives 3.086".format(tisserand)

    assert classifyTancrediComet(a, e, incl) == 'asteroidal orbit', \
        "233P/La Sagra was classified as {!r}".format(classifyTancrediComet(a, e, incl))


def test_tancredi_quasi_hilda_sets_the_upper_tisserand_limit():
    """ The upper limit of 3.05 is justified in the paper by the quasi-Hildas, which it states have
        that Tisserand parameter at a = 4.05 AU, e = 0, i = 0. That orbit must therefore fall just
        inside the Jupiter family class rather than outside it.
    """

    tisserand = float(calcTisserand(4.05, 0.0, 0.0))

    assert abs(tisserand - 3.05) < 5e-3, \
        "T_Jup of the quoted quasi-Hilda orbit is {:.4f}, the paper gives about 3.05".format(
            tisserand)

    assert tisserand < TANCREDI_T_HIGH, "the quasi-Hilda orbit fell outside the Jupiter family cut"

    assert classifyTancrediComet(4.05, 0.0, 0.0) == 'jupiter family'


def test_tancredi_comet_classes_split_at_the_tisserand_limits():
    """ The cometary classes are separated by the Tisserand parameter at 2 and 3.05, with the
        Centaurs split off by perihelion instead.
    """

    # 2P/Encke: T_J just under 3.05, perihelion well inside Jupiter's orbit
    assert classifyTancrediComet(2.215, 0.848, math.radians(11.70)) == 'jupiter family'

    # A retrograde orbit has a low Tisserand parameter
    assert classifyTancrediComet(17.8, 0.967, math.radians(162.3)) == 'halley'

    # 2060 Chiron: perihelion beyond Jupiter's aphelion but inside Uranus
    assert classifyTancrediComet(13.65, 0.379, math.radians(6.9)) == 'centaur'

    q_chiron = 13.65*(1.0 - 0.379)
    assert Q_JUPITER_APHELION < q_chiron < A_URANUS


def test_tancredi_asteroid_needs_to_reach_jupiter():
    """ The condition that separates an asteroid in a cometary orbit from the thousands of stable
        asteroids sharing its Tisserand parameter is that it actually approaches a giant planet.
        The same orbit must classify either way on the minimum orbital intersection distance alone.
    """

    # An orbit with 2 < T_J < 3.05 whose aphelion crosses Jupiter's perihelion
    a, e, incl = 3.6, 0.45, math.radians(8.0)

    tisserand = float(calcTisserand(a, e, incl))
    assert 2.0 < tisserand < TANCREDI_T_HIGH, "test orbit is not in the Jupiter family range"

    assert classifyTancrediAsteroid(a, e, incl, 1.0, 1.0) == 'aco jupiter family'
    assert classifyTancrediAsteroid(a, e, incl, 8.0, 8.0) == 'asteroid'


def test_tancredi_resonance_protection_excludes_the_hildas():
    """ The Hildas sit in the 3:2 resonance and never approach Jupiter despite a Tisserand
        parameter in the cometary range, which is exactly what the resonance filter is for.
    """

    a_hilda = float(calcResonanceSemiMajorAxis(2, 3))

    assert abs(a_hilda - 3.971) < 1e-3

    assert isTancrediResonanceProtected(a_hilda, 0.15, 1.0), \
        "a Hilda orbit was not recognised as resonance protected"

    assert classifyTancrediAsteroid(a_hilda, 0.15, math.radians(5.0), 1.0, 1.0) == 'asteroid'

    # An orbit away from any resonance is not protected
    assert not isTancrediResonanceProtected(3.6, 0.45, 1.0)


def test_tancredi_asteroid_outer_classes():
    """ The asteroid scheme reuses the cometary Centaur definition, applies no semi-major axis cut
        to the Halley type, and hands anything beyond Uranus to the transneptunian region.
    """

    # Retrograde, so the Tisserand parameter is below 2
    assert classifyTancrediAsteroid(8.0, 0.6, math.radians(150.0), 1.0, 1.0) == 'aco halley'

    # Perihelion between Jupiter's aphelion and Uranus
    assert classifyTancrediAsteroid(13.65, 0.379, math.radians(6.9), 9.0, 9.0) == 'centaur'

    # Perihelion beyond Uranus
    assert classifyTancrediAsteroid(45.0, 0.1, math.radians(5.0), 99.0, 99.0) == 'transneptunian'



# Table 1 of Tancredi (2014) also tabulates the maximum libration in semi-major axis of the five
#   inner resonances, computed by the method of its appendix B, as (label, p + q, p, half width AU)
TANCREDI_TABLE_1_WIDTHS = [('4:1', 4, 1, 0.0075), ('3:1', 3, 1, 0.0287), ('5:2', 5, 2, 0.0260),
    ('7:3', 7, 3, 0.0215), ('2:1', 2, 1, 0.1127)]


def test_tancredi_resonance_widths_match_table1():
    """ The libration widths of appendix B must reproduce those tabulated in table 1. The table
        gives the maximum libration, so it is evaluated at the top of the eccentricity range over
        which the paper considers the expansion reliable.
    """

    for label, p_plus_q, p, published in TANCREDI_TABLE_1_WIDTHS:

        got = float(calcResonanceWidth(p, p_plus_q, RESONANCE_E_MAX))

        assert abs(got - published)/published < 0.01, \
            "the {:s} width came out as {:.5f} AU, table 1 gives {:.4f}".format(label, got,
                published)


def test_tancredi_resonance_width_grows_with_eccentricity_and_is_clamped():
    """ A resonance widens with eccentricity. Outside the range where the expansion of the
        disturbing function is reliable the width is held at the value on the nearer edge.
    """

    widths = [float(calcResonanceWidth(1, 2, e)) for e in (0.05, 0.1, 0.2, 0.3)]

    assert widths == sorted(widths), "the 2:1 width did not grow with eccentricity"

    assert abs(float(calcResonanceWidth(1, 2, 0.001))
        - float(calcResonanceWidth(1, 2, RESONANCE_E_MIN))) < 1e-12
    assert abs(float(calcResonanceWidth(1, 2, 0.9))
        - float(calcResonanceWidth(1, 2, RESONANCE_E_MAX))) < 1e-12


def test_moid_matches_a_brute_force_search():
    """ The minimum found by the four-start search must be the global minimum of the distance
        surface, which a dense grid over both eccentric anomalies settles independently.

        The orbits are those of figure A.1 of the paper, chosen there because the surface has a
        second local minimum that a single search can fall into.
    """

    orbit_1 = (1.0, 0.0, 0.0, 0.0, 0.0)
    orbit_2 = (1.2, 0.4, math.radians(30.0), math.radians(330.0), math.radians(30.0))

    got = float(calcMOID(*(orbit_1 + orbit_2)))

    grid = np.linspace(0, 2*np.pi, 600)
    mesh_1, mesh_2 = np.meshgrid(grid, grid, indexing='ij')

    point_1 = _orbitPosition(mesh_1, *orbit_1)
    point_2 = _orbitPosition(mesh_2, *orbit_2)

    brute = float(np.sqrt(sum((point_1[k] - point_2[k])**2 for k in range(3))).min())

    assert got <= brute + 1e-9, \
        "the search returned {:.6f} AU, above the grid minimum of {:.6f}".format(got, brute)
    assert abs(got - brute) < 1e-3, \
        "the search returned {:.6f} AU against a grid minimum of {:.6f}".format(got, brute)


def test_moid_analytic_cases():
    """ Two coplanar circles are separated everywhere by the difference of their radii, and an
        orbit is at zero distance from itself.
    """

    assert abs(float(calcMOID(1.0, 0, 0, 0, 0, 1.2, 0, 0, 0, 0)) - 0.2) < 1e-6

    orbit = (2.5, 0.6, math.radians(12.0), math.radians(40.0), math.radians(200.0))

    assert float(calcMOID(*(orbit + orbit))) < 1e-6


def test_moid_is_symmetric():
    """ The distance between two orbits cannot depend on which is given first. """

    orbit_1 = (2.7, 0.35, math.radians(9.0), math.radians(120.0), math.radians(60.0))
    orbit_2 = (5.2, 0.05, math.radians(1.3), math.radians(100.0), math.radians(275.0))

    forward = float(calcMOID(*(orbit_1 + orbit_2)))
    backward = float(calcMOID(*(orbit_2 + orbit_1)))

    assert abs(forward - backward) < 1e-6, \
        "MOID gave {:.6f} one way and {:.6f} the other".format(forward, backward)


def test_forbidden_region_closes_at_tisserand_three():
    """ Above a Tisserand parameter of 3 an orbit cannot reach the planet, so there is a distance
        it cannot come inside. That bound has to vanish at 3, where the orbit just touches, and
        widen above it.
    """

    assert abs(float(calcMinMOIDForTisserand(3.0))) < 1e-4, \
        "the forbidden region does not close at a Tisserand parameter of 3"

    bounds = [float(calcMinMOIDForTisserand(t)) for t in (3.05, 3.1, 3.2, 3.4)]

    assert bounds == sorted(bounds), "the forbidden region did not widen with T"

    # The largest perihelion belongs to the circular orbit, which is the planet's own at T = 3
    assert abs(float(calcMaxPerihelionForTisserand(3.0)) - 1.0) < 1e-6

    try:
        calcMaxPerihelionForTisserand(2.9)
        raise AssertionError("expected a ValueError below a Tisserand parameter of 3")

    except ValueError:
        pass


def test_giant_planet_moids_close_the_tancredi_criterion():
    """ With the MOIDs computed rather than supplied, the classification runs from orbital elements
        alone, which is what the criterion is for.
    """

    a, e, incl = 3.6, 0.45, math.radians(8.0)
    node, peri = math.radians(100.0), math.radians(50.0)

    distances = calcGiantPlanetMOIDs(a, e, incl, node, peri)

    assert set(distances) == {'jupiter', 'saturn', 'uranus', 'neptune'}
    assert all(v > 0 for v in distances.values())

    # Jupiter is the closest of the four for an orbit of this size
    assert min(distances, key=distances.get) == 'jupiter'

    label = classifyTancrediAsteroid(a, e, incl, distances['jupiter'], min(distances.values()))

    assert label == 'aco jupiter family', "the orbit classified as {!r}".format(label)



if __name__ == "__main__":

    test_functions = [
        test_rho_reproduces_kholshevnikov_table,
        test_rho_metrics_satisfy_triangle_inequality,
        test_DD_violates_triangle_inequality,
        test_rho_vanishes_on_identical_orbits_and_is_symmetric,
        test_rho5_is_a_lower_bound_of_rho2,
        test_rho5_attains_the_minimum_of_rho2,
        test_rho5_invariant_under_node_and_perihelion_rotation,
        test_rho_defined_for_circular_orbits,
        test_rho_accepts_arrays,
        test_C_matches_the_angular_momentum_term_of_rho1,
        test_C_vanishes_on_coplanar_orbits_of_equal_semilatus_rectum,
        test_DR_is_a_lower_bound_of_DN,
        test_DR_vanishes_on_identical_radiants,
        test_DB_vanishes_on_identical_orbits_and_is_symmetric,
        test_DB_C3_uses_the_smallest_angular_difference,
        test_DT_matches_the_tisserand_difference,
        test_DT_reproduces_the_jenniskens_tisserand_values,
        test_DT_stays_finite_at_unit_eccentricity,
        test_DX_vanishes_on_identical_radiants,
        test_DX_weights_scale_the_terms,
        test_DX_weights_are_the_published_values,
        test_DX_reproduces_the_quoted_shower_means,
        test_DX_separates_the_taurid_branches,
        test_DVJopek_vanishes_on_identical_orbits_and_is_symmetric,
        test_DVJopek_energy_term_matches_the_semimajor_axis,
        test_DV_weights_follow_from_the_published_dispersions,
        test_DV_weights_make_two_sigma_contribute_unity,
        test_DV_dispersions_grow_as_a_stream_ages,
        test_DV_thresholds_sit_just_above_the_spread_within_a_stream,
        test_DV_uses_astronomical_units_and_days,
        test_DACS_semimajor_axis_term_bounds_table1,
        test_DACS_semimajor_axis_term_bound_is_attained,
        test_DACS_encke_matches_the_reference_orbit,
        test_DACS_table1_is_ordered_by_the_criterion,
        test_DACS_scale_makes_three_au_contribute_unity,
        test_DACS_vanishes_on_identical_orbits_and_is_symmetric,
        test_DACS_ignores_the_node_and_perihelion_argument,
        test_DACS_accepts_arrays,
        test_DSAC_reference_orbit_agrees_with_the_semimajor_axis_form,
        test_DSAC_perihelion_term_is_unscaled,
        test_DSAC_and_DACS_share_their_eccentricity_and_inclination_terms,
        test_DSAC_vanishes_on_identical_orbits_and_is_symmetric,
        test_DSAC_takes_only_the_elements_the_command_line_supplies,
        test_DSAC_accepts_arrays,
        test_tisserand_is_three_for_a_planet_crossing_circular_orbit,
        test_tisserand_reproduces_encke,
        test_kresak_and_aphelion_boundaries,
        test_jopek_williams_two_parameter_criteria,
        test_classification_accepts_arrays,
        test_tancredi_resonance_semimajor_axes_match_table1,
        test_tancredi_hill_radii_match_the_paper,
        test_tancredi_classifies_la_sagra_as_a_comet_in_an_asteroidal_orbit,
        test_tancredi_quasi_hilda_sets_the_upper_tisserand_limit,
        test_tancredi_comet_classes_split_at_the_tisserand_limits,
        test_tancredi_asteroid_needs_to_reach_jupiter,
        test_tancredi_resonance_protection_excludes_the_hildas,
        test_tancredi_asteroid_outer_classes,
        test_tancredi_resonance_widths_match_table1,
        test_tancredi_resonance_width_grows_with_eccentricity_and_is_clamped,
        test_moid_matches_a_brute_force_search,
        test_moid_analytic_cases,
        test_moid_is_symmetric,
        test_forbidden_region_closes_at_tisserand_three,
        test_giant_planet_moids_close_the_tancredi_criterion,
        ]

    failed = 0
    for test_func in test_functions:

        try:
            test_func()
            print("PASS: {:s}".format(test_func.__name__))

        except Exception as e:
            failed += 1
            print("FAIL: {:s}: {:s}".format(test_func.__name__, str(e)))

    print()
    if failed:
        print("{:d}/{:d} tests failed".format(failed, len(test_functions)))
        raise SystemExit(1)

    print("All {:d} tests passed".format(len(test_functions)))
