""" Tests for the orbital class labels of wmpl.Utils.OrbitClassification: the near-Earth classes,
    the Arjuna orbital domain, the Tisserand classes and the orbit flags.

The class labels are checked against well-known objects, whose elements are those published for
them, and against orbits built exactly on each limit, to pin which side of a limit belongs to which
class.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_OrbitClassification.py -v

or standalone (no pytest required):
    python -m wmpl.Utils.Tests.test_OrbitClassification
"""

import math

import numpy as np

from wmpl.Utils.OrbitClassification import (classifyNEO, isArjuna, classifyTisserand,
    isSunApproaching, isProgrograde, isRetrograde, isEarthCrossing, calcTisserand,
    EARTH_PERIHELION, EARTH_APHELION, NEO_Q_LIMIT, ARJUNA_A_MIN, ARJUNA_A_MAX, ARJUNA_E_MAX,
    ARJUNA_INCL_MAX, LONG_PERIOD_LIMIT, A_JUPITER, _tisserandFromPerihelion)


def _q(a, e):
    """ Perihelion distance from the semi-major axis and eccentricity. """
    return a*(1.0 - e)


def test_neo_classes_of_known_asteroids():
    """ One asteroid of each class, from its published semi-major axis and eccentricity. """

    # (name, a [AU], e, class)
    objects = [("163693 Atira", 0.7413, 0.3224, 'Atira'),
               ("2062 Aten", 0.9668, 0.1827, 'Aten'),
               ("1862 Apollo", 1.4701, 0.5600, 'Apollo'),
               ("433 Eros", 1.4582, 0.2229, 'Amor'),
               ("1 Ceres", 2.7675, 0.0758, 'non-NEO')]

    for name, a, e, expected in objects:
        assert classifyNEO(_q(a, e), e) == expected, \
            "{:s} came out as {!r}".format(name, classifyNEO(_q(a, e), e))


def test_neo_limits_are_the_earth_perihelion_and_aphelion():
    """ Atira ends at an aphelion of 0.983 AU, Apollo at a perihelion of 1.017 AU and Amor at
        1.3 AU. An orbit exactly on the first limit is an Aten, as that class is defined by
        Q >= 0.983 AU, and one exactly on the other two goes to the next class out.
    """

    e = 0.2

    # a < 1 AU and Q on the limit
    q = EARTH_PERIHELION/(1.0 + e)*(1.0 - e)
    assert classifyNEO(q, e) == 'Aten'
    assert classifyNEO(q*(1.0 - 1e-9), e) == 'Atira'

    # a > 1 AU and q on the limits
    assert classifyNEO(EARTH_APHELION*(1.0 - 1e-9), e) == 'Apollo'
    assert classifyNEO(EARTH_APHELION, e) == 'Amor'
    assert classifyNEO(NEO_Q_LIMIT*(1.0 - 1e-9), e) == 'Amor'
    assert classifyNEO(NEO_Q_LIMIT, e) == 'non-NEO'


def test_neo_class_has_no_gap_at_one_astronomical_unit():
    """ An orbit with a = 1 AU exactly goes with the larger orbits. """

    for e in (0.0, 0.1, 0.5):
        assert classifyNEO(_q(1.0, e), e) in ('Apollo', 'Amor'), \
            "e = {:.1f}, a = 1 AU: {!r}".format(e, classifyNEO(_q(1.0, e), e))


def test_neo_unbound_orbit_is_labelled_unbound():
    """ The classes are for bound orbits, and an unbound one must not fall into one of them. """

    assert classifyNEO(0.5, 1.0) == 'unbound'
    assert classifyNEO(0.5, 1.3) == 'unbound'


def test_arjuna_domain_and_its_limits():
    """ The Arjuna limits are strict, and it takes all three of them. """

    i = np.radians(1.0)

    assert isArjuna(_q(0.9977, 0.0245), 0.0245, np.radians(1.087))     # 2006 RH120

    # Each limit, just inside and just outside
    assert isArjuna(_q(ARJUNA_A_MIN + 1e-6, 0.05), 0.05, i)
    assert not isArjuna(_q(ARJUNA_A_MIN, 0.05), 0.05, i)
    assert isArjuna(_q(ARJUNA_A_MAX - 1e-6, 0.05), 0.05, i)
    assert not isArjuna(_q(ARJUNA_A_MAX, 0.05), 0.05, i)
    assert isArjuna(_q(1.0, ARJUNA_E_MAX - 1e-6), ARJUNA_E_MAX - 1e-6, i)
    assert not isArjuna(_q(1.0, ARJUNA_E_MAX), ARJUNA_E_MAX, i)
    assert isArjuna(_q(1.0, 0.05), 0.05, ARJUNA_INCL_MAX - 1e-6)
    assert not isArjuna(_q(1.0, 0.05), 0.05, ARJUNA_INCL_MAX)

    # A retrograde orbit of the same size is not Earth-like
    assert not isArjuna(_q(1.0, 0.05), 0.05, np.radians(179.0))

    # An unbound orbit has no semi-major axis to be near 1 AU
    assert not isArjuna(0.99, 1.0, i)


def test_arjuna_members_are_also_in_a_neo_class():
    """ Arjuna is a subset of the other classes and not an alternative to them. """

    for a, e in ((0.99, 0.05), (1.0, 0.08), (1.012, 0.02)):
        assert isArjuna(_q(a, e), e, np.radians(2.0))
        assert classifyNEO(_q(a, e), e) in ('Aten', 'Apollo', 'Amor', 'Atira')


def test_tisserand_class_of_known_bodies():
    """ A body of each class, from its published elements (inclination in degrees). """

    # (name, a [AU], e, i [deg], class)
    objects = [("1 Ceres", 2.7675, 0.0758, 10.59, 'asteroidal-like'),
               ("67P/Churyumov-Gerasimenko", 3.4630, 0.6410, 7.04, 'jupiter-family-comet-like'),
               ("1P/Halley", 17.834, 0.96714, 162.26, 'halley-type-comet-like'),
               ("109P/Swift-Tuttle", 26.092, 0.96363, 113.45, 'halley-type-comet-like'),
               ("C/1995 O1 Hale-Bopp", 186.0, 0.9951, 89.43, 'long-period-comet-like')]

    for name, a, e, incl, expected in objects:
        label = classifyTisserand(_q(a, e), e, np.radians(incl))
        assert label == expected, "{:s} came out as {!r}".format(name, label)


def test_tisserand_limits_and_period_limit():
    """ T_J = 3 belongs to the Jupiter family and T_J = 2 to the Jupiter family too, as the table
        gives asteroidal as T_J > 3 and Halley type as T_J < 2. At T_J < 2 the period decides.
    """

    def orbitWithTisserand(tisserand, a):
        # T = a_J/a + 2 sqrt((a/a_J)(1 - e^2)) at i = 0, solved for e
        root = (tisserand - A_JUPITER/a)/2.0
        e = math.sqrt(1.0 - root**2*A_JUPITER/a)
        assert abs(float(calcTisserand(a, e, 0.0)) - tisserand) < 1e-9
        return _q(a, e), e

    q, e = orbitWithTisserand(3.0, 3.0)
    assert classifyTisserand(q, e, 0.0) == 'jupiter-family-comet-like'

    q, e = orbitWithTisserand(3.0 + 1e-6, 3.0)
    assert classifyTisserand(q, e, 0.0) == 'asteroidal-like'

    q, e = orbitWithTisserand(2.0, 3.0)
    assert classifyTisserand(q, e, 0.0) == 'jupiter-family-comet-like'

    q, e = orbitWithTisserand(2.0 - 1e-6, 3.0)
    assert classifyTisserand(q, e, 0.0) == 'halley-type-comet-like'

    # The same T_J < 2 on either side of a 200 yr period, a = 200^(2/3) AU
    a_limit = LONG_PERIOD_LIMIT**(2.0/3.0)

    for a, expected in ((a_limit*(1.0 - 1e-6), 'halley-type-comet-like'),
                        (a_limit, 'halley-type-comet-like'),
                        (a_limit*(1.0 + 1e-6), 'long-period-comet-like')):

        # q = 0.5 AU, i = 0 puts T_J well under 2 at these sizes
        e = 1.0 - 0.5/a
        assert float(calcTisserand(a, e, 0.0)) < 2.0
        assert classifyTisserand(0.5, e, 0.0) == expected, \
            "a = {:.4f} AU came out as {!r}".format(a, classifyTisserand(0.5, e, 0.0))


def test_tisserand_from_perihelion_matches_calcTisserand_for_bound_orbits():
    """ The class is computed from q and e, so that it is defined for an unbound orbit, and that
        form must agree with the semi-major axis form wherever the latter exists.
    """

    for a, e, incl in ((0.8, 0.3, 0.2), (2.5, 0.6, 0.5), (17.8, 0.967, 2.8), (186.0, 0.995, 1.5)):

        assert abs(_tisserandFromPerihelion(_q(a, e), e, incl)
            - float(calcTisserand(a, e, incl))) < 1e-9, "a = {:.1f} AU".format(a)


def test_tisserand_class_of_unbound_orbits():
    """ A parabolic or hyperbolic orbit has an infinite period, so it is long period wherever its
        Tisserand parameter is under 2, and is classified rather than left as nan.
    """

    for e in (1.0, 1.0005, 1.5):
        assert classifyTisserand(0.5, e, np.radians(30.0)) == 'long-period-comet-like'

    # A tight, low-inclination hyperbolic orbit has T_J under 2 only through its first term
    assert classifyTisserand(0.5, 1.0, 0.0) == 'long-period-comet-like'


def test_flags():
    """ Sun approaching, direction and Earth crossing. """

    assert isSunApproaching(0.1) and not isSunApproaching(0.2) and not isSunApproaching(0.5)
    assert isSunApproaching(0.3, q_limit=0.4)

    assert isProgrograde(np.radians(10.0)) and not isRetrograde(np.radians(10.0))
    assert isRetrograde(np.radians(170.0)) and not isProgrograde(np.radians(170.0))

    # Exactly polar is neither
    assert not isProgrograde(np.pi/2) and not isRetrograde(np.pi/2)

    # q <= 1 <= Q
    assert isEarthCrossing(0.5, 0.5)         # Q = 1.5
    assert not isEarthCrossing(1.2, 0.3)     # q above 1
    assert not isEarthCrossing(0.5, 0.2)     # Q = 0.75 < 1
    assert isEarthCrossing(1.0, 0.1)         # touches 1 AU at perihelion
    assert isEarthCrossing(0.5, 1.0)         # parabolic
    assert isEarthCrossing(0.5, 1.3)         # hyperbolic
    assert not isEarthCrossing(1.5, 1.3)


def test_classification_accepts_arrays():
    """ The labels and flags work elementwise, and a scalar still gives a plain str. """

    # An Aten, an Amor on a retrograde orbit, an Earth-like Apollo, and a hyperbolic orbit
    a = np.array([0.8, 1.375, 1.0])
    e = np.array([0.3, 0.2, 0.05])
    q = np.append(_q(a, e), 2.0)
    e = np.append(e, 1.2)
    i = np.radians(np.array([5.0, 150.0, 1.0, 40.0]))

    assert classifyNEO(q, e).tolist() == ['Aten', 'Amor', 'Apollo', 'unbound']
    assert isArjuna(q, e, i).tolist() == [False, False, True, False]
    assert classifyTisserand(q, e, i).tolist() == ['asteroidal-like',
        'jupiter-family-comet-like', 'asteroidal-like', 'long-period-comet-like']
    assert isSunApproaching(q).tolist() == [False, False, False, False]
    assert isProgrograde(i).tolist() == [True, False, True, True]
    assert isRetrograde(i).tolist() == [False, True, False, False]
    assert isEarthCrossing(q, e).tolist() == [True, False, True, False]

    assert type(classifyNEO(0.7, 0.3)) is str
    assert type(classifyTisserand(0.7, 0.3, 0.1)) is str


if __name__ == "__main__":

    test_functions = [
        test_neo_classes_of_known_asteroids,
        test_neo_limits_are_the_earth_perihelion_and_aphelion,
        test_neo_class_has_no_gap_at_one_astronomical_unit,
        test_neo_unbound_orbit_is_labelled_unbound,
        test_arjuna_domain_and_its_limits,
        test_arjuna_members_are_also_in_a_neo_class,
        test_tisserand_class_of_known_bodies,
        test_tisserand_limits_and_period_limit,
        test_tisserand_from_perihelion_matches_calcTisserand_for_bound_orbits,
        test_tisserand_class_of_unbound_orbits,
        test_flags,
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
