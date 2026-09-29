""" Tests for the meteoroid orbit dissimilarity criteria in wmpl.Utils.Dcriteria.

The tests cover two properties that hold independently of any tabulated value:

    - The node representation invariance of D_SH and D_H. A pair of orbits is unchanged when the
      second node is written as O2 or as O2 - 360 deg, so a dissimilarity criterion must return the
      same value for both. This requires the Southworth & Hawkins (1963) sign factor rho, which
      selects the negative branch of the Pi_21 arcsine when |O2 - O1| > 180 deg. The convention
      assumes both nodes are given in [0, 360 deg), since the rho test is not 2*pi periodic.

    - Agreement with the independent vector definition of Pi_21. Jopek & Bronikowska (2017) define
      Pi_21 through the mutual node of the two orbital planes rather than through an arcsine, which
      has no branch cut to get wrong, so it settles the sign convention outright.

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

from wmpl.Utils.Dcriteria import calcDSH, calcDD, calcDH, _clippedAcos


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


def _piFromVectorDefinition(i1, O1, w1, i2, O2, w2):
    """ Compute Pi_21 from the mutual node of the two orbital planes, which needs no arcsine and so
        has no sign branch to choose.

        Jopek & Bronikowska (2017), eqs 4, 5 and 7, define Pi_21 as the difference of the angles
        from the mutual node to each perihelion direction. Their eq. 7 writes those angles as
        arccos(N.e), which is unsigned and so cannot tell which side of the node a perihelion lies
        on; the angles are resolved here within each orbital plane about that orbit's own normal.

    Arguments:
        i1: [float] inclination of the first orbit (rad)
        O1: [float] longitude of ascending node of the first orbit (rad)
        w1: [float] argument of perihelion of the first orbit (rad)
        i2: [float] inclination of the second orbit (rad)
        O2: [float] longitude of ascending node of the second orbit (rad)
        w2: [float] argument of perihelion of the second orbit (rad)

    Return:
        [float] Pi_21 (rad), or None if the orbits are coplanar and the mutual node undefined
    """

    def angularMomentum(incl, node):
        return np.array([math.sin(incl)*math.sin(node), -math.sin(incl)*math.cos(node),
            math.cos(incl)])

    def perihelionDirection(incl, node, peri):
        return np.array([
            math.cos(peri)*math.cos(node) - math.cos(incl)*math.sin(peri)*math.sin(node),
            math.cos(peri)*math.sin(node) + math.cos(incl)*math.sin(peri)*math.cos(node),
            math.sin(incl)*math.sin(peri)])

    h1 = angularMomentum(i1, O1)
    h2 = angularMomentum(i2, O2)

    node_vect = np.cross(h1, h2)
    node_norm = np.linalg.norm(node_vect)

    if node_norm < 1e-8:
        return None

    node_vect = node_vect/node_norm

    e1_vect = perihelionDirection(i1, O1, w1)
    e2_vect = perihelionDirection(i2, O2, w2)

    angle1 = math.atan2(float(np.dot(np.cross(node_vect, e1_vect), h1)),
        float(np.dot(node_vect, e1_vect)))
    angle2 = math.atan2(float(np.dot(np.cross(node_vect, e2_vect), h2)),
        float(np.dot(node_vect, e2_vect)))

    return angle1 - angle2


def test_DH_pi21_matches_the_vector_definition():
    """ D_H must imply the same Pi_21 as the mutual-node construction, which carries no sign
        convention to get wrong. Only sin(Pi_21/2) squared enters D_H, and with equal perihelion
        distances and eccentricities it can be recovered from the returned value, so this tests the
        public result rather than an internal quantity.
    """

    q, e = 0.7, 0.6

    pairs = _orbitPairs()

    worst = 0.0
    n_straddling = 0

    for _, _, i1, O1, w1, _, _, i2, O2, w2 in pairs:

        pi21 = _piFromVectorDefinition(i1, O1, w1, i2, O2, w2)

        if pi21 is None:
            continue

        expected = math.sin(pi21/2.0)**2

        d_value = calcDH(q, e, i1, O1, w1, q, e, i2, O2, w2)

        # With q1 = q2 and e1 = e2 the perihelion and eccentricity terms vanish, leaving
        #   D_H^2 = (2 sin(I_21/2))^2 + e^2 (2 sin(Pi_21/2))^2
        cos_I = np.clip(math.cos(i1)*math.cos(i2) + math.sin(i1)*math.sin(i2)*math.cos(O2 - O1),
            -1.0, 1.0)
        incl_term = 2*math.sin(math.acos(cos_I)/2.0)

        got = (d_value**2 - incl_term**2)/(4*e**2)

        worst = max(worst, abs(got - expected))

        if abs(O2 - O1) > math.pi:
            n_straddling += 1

    assert n_straddling > 0, "no test pair exercised the |O2 - O1| > 180 deg branch"
    assert worst < 1e-9, \
        "D_H implied a Pi_21 differing from the vector definition by {:.4e}".format(worst)


def test_DH_node_sign_worked_example():
    """ The worked example that motivated the fix, pinned so the convention cannot be reverted.

        q = 0.5 AU, e = 0.7, i = 15 deg, O1 = 10 deg, w1 = 150 deg, w2 = 190 deg, with the second
        node written both ways across the branch cut. Before the rho factor the two representations
        gave 0.69872 and 0.26675; both must now give 0.26675.
    """

    common = (0.5, 0.7, math.radians(15.0), math.radians(10.0), math.radians(150.0))

    for node_deg in (350.0, -10.0):

        d_value = calcDH(*(common + (0.5, 0.7, math.radians(15.0), math.radians(node_deg),
            math.radians(190.0))))

        assert abs(d_value - 0.26675) < 1e-5, \
            "D_H with the second node at {:.0f} deg came out as {:.5f}, expected 0.26675".format(
                node_deg, d_value)


def test_rho_convention_holds_only_within_one_turn():
    """ The rho test is on the raw node difference, so it is not 2*pi periodic.

        Adding a full turn to a node pushes |O2 - O1| past 180 deg and selects the other branch,
        which changes the result. This is a property of the Southworth & Hawkins convention rather
        than of this implementation, and it affects D_SH exactly as it affects D_H.

        The behaviour is pinned here deliberately. It is a known limitation, documented in both
        docstrings, and this test exists so that a later change to it is a visible decision rather
        than an accident. The robust alternative is the vector definition of Pi_21 used by
        _piFromVectorDefinition below, which has no branch to select.
    """

    common = (0.5, 0.7, math.radians(15.0), math.radians(10.0), math.radians(150.0))
    second = (0.5, 0.7, math.radians(15.0))

    for criterion in (calcDSH, calcDH):

        within = criterion(*(common + second + (math.radians(350.0), math.radians(190.0))))
        extra_turn = criterion(*(common + second
            + (math.radians(350.0) + 2*math.pi, math.radians(190.0))))

        assert abs(within - extra_turn) > 0.1, \
            "{:s} unexpectedly survived a full turn added to the node; if that is now intended, " \
            "update this test and both docstrings".format(criterion.__name__)


def test_clipped_acos_engages_only_outside_the_domain():
    """ The guard must change nothing that was already computable. """

    # An argument inside the domain is passed through untouched
    for value in (-1.0, -0.5, 0.0, 0.25, 1.0):
        assert _clippedAcos(value) == math.acos(value)

    # One ulp outside, where math.acos raises, is clipped to the endpoint
    just_above = 1.0 + 2*np.spacing(1.0)
    just_below = -1.0 - 2*np.spacing(1.0)

    try:
        math.acos(just_above)
        raise AssertionError("the test argument is not actually outside the domain")
    except ValueError:
        pass

    assert _clippedAcos(just_above) == 0.0
    assert _clippedAcos(just_below) == math.pi


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
        test_DH_pi21_matches_the_vector_definition,
        test_acos_domain_holds_on_self_comparison,
        test_acos_domain_holds_for_near_identical_orbits,
        test_DH_node_sign_worked_example,
        test_rho_convention_holds_only_within_one_turn,
        test_clipped_acos_engages_only_outside_the_domain,
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
