""" Tests for the meteoroid orbit dissimilarity criteria in wmpl.Utils.Dcriteria.

The tests cover a property that holds independently of any tabulated value:

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
