""" Tests for the meteoroid orbit dissimilarity criteria in wmpl.Utils.Dcriteria.

The tests cover two properties that hold independently of any tabulated value:

    - The node representation invariance of D_SH and D_H. A pair of orbits is unchanged when the
      second node is written as O2 or as O2 - 360 deg, so a dissimilarity criterion must return the
      same value for both. This requires the Southworth & Hawkins (1963) sign factor rho, which
      selects the negative branch of the Pi_21 arcsine when |O2 - O1| > 180 deg. The convention
      assumes both nodes are given in [0, 360 deg), since the rho test is not 2*pi periodic.

    - The acos domain guard. For identical or near-identical orbits the argument of the inclination
      (and, in D_D, of the angular separation of the perihelion directions) evaluates to slightly
      more than 1.0 in double precision, which makes an unguarded math.acos raise a domain error.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_Dcriteria.py -v

or standalone (no pytest required):
    python -m wmpl.Utils.Tests.test_Dcriteria
"""

import math

import numpy as np

from wmpl.Utils.Dcriteria import calcDSH, calcDD, calcDH


# A self-comparison cannot return exactly zero: acos has an infinite derivative at 1, so the
#   1 ulp error in its argument becomes ~2e-8 in the returned angle
SELF_COMPARISON_TOL = 1e-7


def _orbitPairs():
    """ Generate orbit pairs spanning the physical parameter ranges, including node pairs that
        straddle 0/360 deg.

    Return:
        [list] Tuples of (q1, e1, i1, O1, w1, q2, e2, i2, O2, w2), all angles in radians.
    """

    rng = np.random.RandomState(20250917)
    n = 5000

    q1 = rng.uniform(0.05, 1.5, n)
    q2 = rng.uniform(0.05, 1.5, n)
    e1 = rng.uniform(0.01, 0.99, n)
    e2 = rng.uniform(0.01, 0.99, n)

    # Sample the inclination isotropically rather than uniformly in angle
    i1 = np.arccos(rng.uniform(-1, 1, n))
    i2 = np.arccos(rng.uniform(-1, 1, n))

    O1 = rng.uniform(0, 2*np.pi, n)
    O2 = rng.uniform(0, 2*np.pi, n)
    w1 = rng.uniform(0, 2*np.pi, n)
    w2 = rng.uniform(0, 2*np.pi, n)

    return list(zip(q1, e1, i1, O1, w1, q2, e2, i2, O2, w2))


def _principalNodePair(O1, O2):
    """ Rewrite the second node so that the node difference lies in (-180, 180] deg, which is the
        same pair of orbits written the other way across the 0/360 deg branch cut.

    Arguments:
        O1: [double] longitude of ascending node of the first orbit (rad)
        O2: [double] longitude of ascending node of the second orbit (rad)

    Return:
        [double] equivalent second node (rad)
    """

    return O1 + (O2 - O1 + math.pi)%(2*math.pi) - math.pi


def test_DH_invariant_under_node_representation():
    """ D_H must depend on the nodes only through their true angular separation, so writing the
        second node on the other side of the 0/360 deg branch cut must not change the value.
    """

    pairs = _orbitPairs()

    max_diff = 0.0
    n_straddling = 0

    for q1, e1, i1, O1, w1, q2, e2, i2, O2, w2 in pairs:

        O2_alt = _principalNodePair(O1, O2)

        d_ref = calcDH(q1, e1, i1, O1, w1, q2, e2, i2, O2, w2)
        d_alt = calcDH(q1, e1, i1, O1, w1, q2, e2, i2, O2_alt, w2)

        max_diff = max(max_diff, abs(d_alt - d_ref))

        if abs(O2 - O1) > math.pi:
            n_straddling += 1

    assert n_straddling > 0, "no test pair exercised the |O2 - O1| > 180 deg branch"
    assert max_diff < 1e-12, \
        "D_H changed by {:.4e} when the node was rewritten (max over {:d} pairs)".format(max_diff, \
            len(pairs))


def test_DSH_invariant_under_node_representation():
    """ D_SH must be invariant to the node representation for the same reason as D_H. """

    pairs = _orbitPairs()

    max_diff = 0.0

    for q1, e1, i1, O1, w1, q2, e2, i2, O2, w2 in pairs:

        O2_alt = _principalNodePair(O1, O2)

        d_ref = calcDSH(q1, e1, i1, O1, w1, q2, e2, i2, O2, w2)
        d_alt = calcDSH(q1, e1, i1, O1, w1, q2, e2, i2, O2_alt, w2)

        max_diff = max(max_diff, abs(d_alt - d_ref))

    assert max_diff < 1e-12, \
        "D_SH changed by {:.4e} when the node was rewritten".format(max_diff)


def test_DH_coplanar_limit_recovers_longitude_of_perihelion():
    """ For two coplanar orbits Pi_21 reduces to the difference of the longitudes of perihelion,
        so D_H must reduce to the corresponding closed form. This pins down the sign of the rho
        factor without needing a tabulated value.
    """

    q, e = 0.5, 0.7
    i = 0.0

    max_diff = 0.0

    for O1_deg in range(0, 360, 13):
        for O2_deg in range(0, 360, 11):
            for dw_deg in range(0, 360, 17):

                O1 = math.radians(O1_deg)
                O2 = math.radians(O2_deg)
                w1 = math.radians(20.0)
                w2 = w1 + math.radians(dw_deg)

                d_got = calcDH(q, e, i, O1, w1, q, e, i, O2, w2)

                # Coplanar closed form: I_21 = 0 and Pi_21 = (w2 + O2) - (w1 + O1)
                pi21 = (w2 + O2) - (w1 + O1)
                d_expected = math.sqrt(((e + e)/2.0)**2*(2*math.sin(pi21/2.0))**2)

                max_diff = max(max_diff, abs(d_got - d_expected))

    assert max_diff < 1e-12, \
        "coplanar D_H departed from the longitude of perihelion form by {:.4e}".format(max_diff)


def test_DH_equals_DSH_when_perihelia_sum_to_unity():
    """ D_H differs from D_SH only in normalising the perihelion distance term by (q1 + q2), so
        the two must agree exactly when q1 + q2 = 1. This ties the D_H node sign convention to
        the one in D_SH.
    """

    pairs = _orbitPairs()

    max_diff = 0.0

    for _, e1, i1, O1, w1, _, e2, i2, O2, w2 in pairs:

        # Split unity into the two perihelion distances
        q1 = 0.37
        q2 = 1.0 - q1

        d_sh = calcDSH(q1, e1, i1, O1, w1, q2, e2, i2, O2, w2)
        d_h = calcDH(q1, e1, i1, O1, w1, q2, e2, i2, O2, w2)

        max_diff = max(max_diff, abs(d_sh - d_h))

    assert max_diff < 1e-12, \
        "D_H and D_SH disagreed by {:.4e} for q1 + q2 = 1".format(max_diff)


def test_acos_domain_holds_on_self_comparison():
    """ Comparing an orbit with itself must return zero for every criterion, at every inclination,
        rather than raising a math domain error.
    """

    # Sweep densely enough to hit the inclinations whose cos^2 i + sin^2 i sums above 1.0
    incl_array = np.radians(np.linspace(0.0, 180.0, 18001))

    q, e = 0.5, 0.7
    O = math.radians(123.4)
    w = math.radians(56.7)

    for name, func in (("D_SH", calcDSH), ("D_D", calcDD), ("D_H", calcDH)):

        worst = 0.0
        worst_incl = None

        for incl in incl_array:

            d_value = func(q, e, incl, O, w, q, e, incl, O, w)

            if abs(d_value) > worst:
                worst = abs(d_value)
                worst_incl = incl

        assert worst < SELF_COMPARISON_TOL, \
            "{:s} returned {:.4e} for an orbit compared with itself at i = {:.4f} deg".format(name, \
                worst, math.degrees(worst_incl))


def test_acos_domain_holds_for_near_identical_orbits():
    """ Monte Carlo clones of one orbit differ from it by far less than the acos rounding error, so
        they exercise the same domain overflow as an exact self-comparison.
    """

    rng = np.random.RandomState(7)
    n = 20000

    q, e = 0.8, 0.55
    O = math.radians(200.0)
    w = math.radians(95.0)

    for name, func in (("D_SH", calcDSH), ("D_D", calcDD), ("D_H", calcDH)):

        for _ in range(n):

            incl = math.acos(rng.uniform(-1, 1))

            # Perturb by a few ulp, which is small enough to leave the acos argument above 1.0
            incl_clone = incl*(1 + rng.randint(-4, 5)*np.finfo(float).eps)

            d_value = func(q, e, incl, O, w, q, e, incl_clone, O, w)

            assert d_value < SELF_COMPARISON_TOL, \
                "{:s} returned {:.4e} for a clone at i = {:.6f} deg".format(name, d_value, \
                    math.degrees(incl))


if __name__ == "__main__":

    test_functions = [
        test_DH_invariant_under_node_representation,
        test_DSH_invariant_under_node_representation,
        test_DH_coplanar_limit_recovers_longitude_of_perihelion,
        test_DH_equals_DSH_when_perihelia_sum_to_unity,
        test_acos_domain_holds_on_self_comparison,
        test_acos_domain_holds_for_near_identical_orbits,
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
