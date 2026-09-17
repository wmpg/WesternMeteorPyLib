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

import math

import numpy as np

from wmpl.Utils.Dcriteria import (calcRho1, calcRho2, calcRho5, calcC, calcDR, calcDB, calcDT,
    calcDX, calcDVJopek, calcDACS, calcDN, calcDD, TC_REFERENCE_A, TC_REFERENCE_E,
    TC_REFERENCE_INCL, DACS_A_SCALE)
from wmpl.Utils.OrbitClassification import (calcTisserand, calcKresakK, calcKresakP,
    calcAphelionDistance, isCometaryQi, isCometaryKi, isCometaryPi, A_JUPITER, JW_INCL_LIMIT)


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
    """ D_T is the absolute difference of the two Tisserand parameters. """

    orbits = [(2.215, 0.848, math.radians(11.70)), (2.234, 0.862, math.radians(4.18)),
              (3.037, 0.795, math.radians(65.82)), (1.5, 0.3, math.radians(2.0))]

    for a1, e1, i1 in orbits:
        for a2, e2, i2 in orbits:

            expected = abs(float(calcTisserand(a1, e1, i1)) - float(calcTisserand(a2, e2, i2)))
            got = float(calcDT(a1, e1, i1, a2, e2, i2))

            assert abs(got - expected) < 1e-12, \
                "D_T gave {:.8f}, expected {:.8f}".format(got, expected)


### Rudawska D_X ###

def test_DX_vanishes_on_identical_radiants():
    """ D_X must be zero for a radiant compared with itself, for any weights. """

    radiants = _randomRadiants(200, random_state=14)

    worst = 0.0

    for ra, dec, sol, vg in radiants:
        worst = max(worst, abs(float(calcDX(sol, ra, dec, vg, sol, ra, dec, vg,
            1.0, 1.0, 1.0, 1.0))))

    assert worst < 1e-12, "D_X returned {:.4e} for a radiant compared with itself".format(worst)


def test_DX_weights_scale_the_terms():
    """ Each weight must scale only its own term, so zeroing three of the four must leave the
        remaining term alone and the whole must be the quadrature sum of the parts.
    """

    args = (math.radians(100.0), math.radians(45.0), math.radians(20.0), 30.0,
            math.radians(105.0), math.radians(50.0), math.radians(25.0), 35.0)

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
        expected = abs(-1.0/(2*a1) + 1.0/(2*a2))

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
    import inspect

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


def test_classification_accepts_arrays():
    """ The classification criteria must work elementwise on numpy arrays. """

    a = np.array([2.0, 3.0, 1.5])
    e = np.array([0.5, 0.8, 0.2])
    i = np.array([math.radians(10.0), math.radians(10.0), JW_INCL_LIMIT + 0.1])

    result = isCometaryQi(a, e, i)

    assert result.tolist() == [False, True, True], "Q-i gave {!s} on arrays".format(result)


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
        test_DX_vanishes_on_identical_radiants,
        test_DX_weights_scale_the_terms,
        test_DVJopek_vanishes_on_identical_orbits_and_is_symmetric,
        test_DVJopek_energy_term_matches_the_semimajor_axis,
        test_DACS_semimajor_axis_term_bounds_table1,
        test_DACS_semimajor_axis_term_bound_is_attained,
        test_DACS_encke_matches_the_reference_orbit,
        test_DACS_table1_is_ordered_by_the_criterion,
        test_DACS_scale_makes_three_au_contribute_unity,
        test_DACS_vanishes_on_identical_orbits_and_is_symmetric,
        test_DACS_ignores_the_node_and_perihelion_argument,
        test_DACS_accepts_arrays,
        test_tisserand_is_three_for_a_planet_crossing_circular_orbit,
        test_tisserand_reproduces_encke,
        test_kresak_and_aphelion_boundaries,
        test_jopek_williams_two_parameter_criteria,
        test_classification_accepts_arrays,
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
