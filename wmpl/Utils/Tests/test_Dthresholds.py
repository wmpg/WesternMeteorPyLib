""" Tests for the D criterion threshold methods in wmpl.Utils.Dthresholds.

The scaled Southworth & Hawkins threshold is checked against its anchor value and its fourth-root
scaling. The break-point method is checked on a synthetic sample built from a tight stream on top
of a broad sporadic background, where the break is known by construction. The reliability method is
checked by re-drawing shuffled samples and confirming that the threshold it returns really does
leave them free of associated pairs at the stated rate.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_Dthresholds.py -v

or standalone (no pytest required):
    python -m wmpl.Utils.Tests.test_Dthresholds
"""

import math

import numpy as np

from wmpl.Utils.Dcriteria import calcRho2
from wmpl.Utils.Dthresholds import (thresholdDr, thresholdBreakPoint, thresholdReliability,
    thresholdRandomPairing, RANDOM_PAIRING_COEFFS)
from wmpl.Utils.Tests.test_DcriteriaAdditional import _randomOrbits


# Table 3 of Jopek & Bronikowska (2017): bolide orbits, D_SH, generator method E. Columns are the
#   sample size, the measured threshold at a coincidental pairing probability of 0.01, the paper's
#   own tabulation of its eq. 1, and its own tabulation of its eq. 14
JB_TABLE_3 = [
    (200, 0.0325, 0.2317, 0.0324),
    (300, 0.0274, 0.2093, 0.0272),
    (400, 0.0237, 0.1948, 0.0241),
    (500, 0.0218, 0.1842, 0.0218),
    (600, 0.0203, 0.1760, 0.0202),
    (700, 0.0190, 0.1694, 0.0189),
    (800, 0.0178, 0.1638, 0.0178),
    ]

# Sample sizes of table 7 of Jopek & Bronikowska (2017)
JB_TABLE_7_SIZES = [1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 14000, 16000]

# Table 7, the measured thresholds at a coincidental pairing probability of 0.01
JB_TABLE_7 = {
    ('D_SH', 'NEAs'):  [0.01489, 0.01154, 0.01018, 0.00881, 0.00802, 0.00725, 0.00645, 0.00612,
                        0.00554, 0.00518, 0.00488],
    ('D_SH', 'radar'): [0.01761, 0.01309, 0.01116, 0.00955, 0.00869, 0.00797, 0.00680, 0.00641,
                        0.00575, 0.00541, 0.00518],
    ('D_SH', 'video'): [0.02008, 0.01456, 0.01190, 0.01015, 0.00920, 0.00825, 0.00712, 0.00656,
                        0.00615, 0.00566, 0.00515],
    ('D_H', 'NEAs'):   [0.01273, 0.01018, 0.00880, 0.00774, 0.00727, 0.00642, 0.00571, 0.00532,
                        0.00485, 0.00450, 0.00423],
    ('D_H', 'radar'):  [0.01733, 0.01258, 0.01063, 0.00907, 0.00827, 0.00774, 0.00668, 0.00622,
                        0.00566, 0.00529, 0.00501],
    ('D_H', 'video'):  [0.01967, 0.01443, 0.01199, 0.01007, 0.00895, 0.00826, 0.00718, 0.00680,
                        0.00632, 0.00589, 0.00540],
    ('D_D', 'NEAs'):   [0.00627, 0.00487, 0.00399, 0.00358, 0.00318, 0.00292, 0.00266, 0.00254,
                        0.00224, 0.00219, 0.00203],
    ('D_D', 'radar'):  [0.00756, 0.00545, 0.00473, 0.00415, 0.00388, 0.00348, 0.00305, 0.00279,
                        0.00252, 0.00234, 0.00228],
    ('D_D', 'video'):  [0.00807, 0.00581, 0.00480, 0.00427, 0.00376, 0.00347, 0.00312, 0.00279,
                        0.00264, 0.00251, 0.00230],
    }

# Table 8, the average ratios of the thresholds between criteria
JB_TABLE_8 = {
    'NEAs':  {'D_H': 1.14, 'D_D': 2.44},
    'radar': {'D_H': 1.03, 'D_D': 2.30},
    'video': {'D_H': 0.99, 'D_D': 2.38},
    }

# The table 7 thresholds are the measured points; the eq. 17 to 25 formulae are least-squares fits
#   through them, so they are not expected to pass through every point. Measured worst departure
#   over all nine formulae and all eleven sample sizes is 6.4e-4, or 5.0% in relative terms
JB_FIT_TOL = 1e-3
JB_FIT_REL_TOL = 0.06


def test_Dr_reproduces_jopek_bronikowska_table3():
    """ Formula 1 must reproduce the paper's own tabulation of it, column D_(1) of table 3. """

    worst = 0.0

    for n_orbits, _, d_formula_1, _ in JB_TABLE_3:

        got = float(thresholdDr(n_orbits))
        worst = max(worst, abs(got - d_formula_1))

    # The column is quoted to four decimals, so the rounding floor is 5e-5
    assert worst < 5e-5, \
        "D_r departed from table 3 of Jopek & Bronikowska (2017) by {:.6f}".format(worst)


def test_random_pairing_reproduces_bolide_formula():
    """ The bolide D_SH threshold must reproduce the paper's own tabulation of its eq. 14,
        column D_(14) of table 3.
    """

    worst = 0.0

    for n_orbits, _, _, d_formula_14 in JB_TABLE_3:

        got = float(thresholdRandomPairing(n_orbits, d_criterion='D_SH', population='bolides'))
        worst = max(worst, abs(got - d_formula_14))

    assert worst < 1e-4, \
        "the bolide threshold departed from column D_(14) of table 3 by {:.6f}".format(worst)


def test_random_pairing_reproduces_table7():
    """ Each of the nine large-sample formulae must track the measured thresholds of table 7 to
        within the scatter of the least-squares fit that produced it.
    """

    for (d_criterion, population), measured in sorted(JB_TABLE_7.items()):

        for n_orbits, expected in zip(JB_TABLE_7_SIZES, measured):

            got = float(thresholdRandomPairing(n_orbits, d_criterion=d_criterion,
                population=population))

            assert abs(got - expected) < JB_FIT_TOL, \
                "{:s}/{:s} at N = {:d} gave {:.5f}, table 7 measures {:.5f}".format(d_criterion,
                    population, n_orbits, got, expected)

            assert abs(got - expected)/expected < JB_FIT_REL_TOL, \
                "{:s}/{:s} at N = {:d} was off by {:.1%}".format(d_criterion, population,
                    n_orbits, abs(got - expected)/expected)


def test_random_pairing_ratios_match_table8():
    """ The ratios between the criteria are a published result in their own right, table 8, and
        they follow from the formulae without reference to any single threshold. D_D is about 2.4
        times tighter than D_SH across every population.
    """

    for population, ratios in sorted(JB_TABLE_8.items()):

        d_sh = np.array([float(thresholdRandomPairing(n, 'D_SH', population))
            for n in JB_TABLE_7_SIZES])

        for d_criterion, expected in sorted(ratios.items()):

            other = np.array([float(thresholdRandomPairing(n, d_criterion, population))
                for n in JB_TABLE_7_SIZES])

            got = float(np.mean(d_sh/other))

            assert abs(got - expected) < 0.02, \
                "D_SH/{:s} for {:s} came out as {:.3f}, table 8 gives {:.2f}".format(d_criterion,
                    population, got, expected)


def test_Dr_is_far_looser_than_the_random_pairing_threshold():
    """ The paper's central practical conclusion: the scaled Southworth & Hawkins threshold sits
        where a coincidental pair is certain, so it must come out far above the threshold that
        actually delivers a probability of 0.01.
    """

    for n_orbits, _, _, _ in JB_TABLE_3:

        loose = float(thresholdDr(n_orbits))
        tight = float(thresholdRandomPairing(n_orbits, 'D_SH', 'bolides'))

        assert loose > 5*tight, \
            "at N = {:d} D_r was {:.4f} against {:.4f}, less than the expected gap".format(
                n_orbits, loose, tight)


def test_random_pairing_rejects_an_unpublished_combination():
    """ No threshold was published for the other criteria or populations, so asking for one is an
        error rather than a silently substituted default.
    """

    for bad in (('D_V', 'bolides'), ('D_SH', 'photographic'), ('rho_2', 'NEAs')):

        try:
            thresholdRandomPairing(1000, d_criterion=bad[0], population=bad[1])
            raise AssertionError("expected a ValueError for {!r}".format(bad))

        except ValueError:
            pass


def test_random_pairing_accepts_arrays():
    """ The threshold must work elementwise on numpy arrays. """

    sizes = np.array(JB_TABLE_7_SIZES)

    vector = thresholdRandomPairing(sizes, 'D_H', 'video')

    assert vector.shape == sizes.shape

    for k, n_orbits in enumerate(JB_TABLE_7_SIZES):

        scalar = float(thresholdRandomPairing(n_orbits, 'D_H', 'video'))

        assert abs(vector[k] - scalar) < 1e-12


def test_random_pairing_covers_every_published_combination():
    """ Every coefficient pair in the table must be reachable and return a positive threshold. """

    for d_criterion, population in sorted(RANDOM_PAIRING_COEFFS):

        value = float(thresholdRandomPairing(1000, d_criterion, population))

        assert value > 0.0, "{:s}/{:s} returned {:.4f}".format(d_criterion, population, value)

    assert len(RANDOM_PAIRING_COEFFS) == 12, \
        "expected 12 published combinations, found {:d}".format(len(RANDOM_PAIRING_COEFFS))


def test_Dr_matches_its_anchor_and_scaling():
    """ D_r = 0.20 (360/N)^(1/4) is anchored at 0.20 for a sample of 360 orbits, and shrinks as the
        fourth root of the sample size.
    """

    assert abs(float(thresholdDr(360)) - 0.20) < 1e-12, \
        "D_r was {:.12f} at its anchor point".format(float(thresholdDr(360)))

    # Sixteen times the orbits must halve the threshold
    assert abs(float(thresholdDr(360*16)) - 0.10) < 1e-12

    assert float(thresholdDr(10000)) < float(thresholdDr(1000))


def test_breakpoint_finds_a_known_break():
    """ Given a tight stream on top of a broad sporadic background, the break point must land on
        the edge of the stream, which is the tightest threshold that still takes all of it.
    """

    rng = np.random.RandomState(21)

    stream_edge = 0.05
    stream = rng.uniform(0.0, stream_edge, 400)
    background = rng.uniform(0.30, 1.0, 400)

    d_values = np.concatenate((stream, background))
    n_bins = 100

    break_point = thresholdBreakPoint(d_values, n_bins=n_bins)

    bin_width = np.max(d_values)/n_bins

    assert abs(break_point - stream_edge) <= bin_width, \
        "the break point came out at {:.5f}, more than one bin of {:.5f} from the stream edge " \
        "at {:.5f}".format(break_point, bin_width, stream_edge)

    # It must in any case be usable as a threshold, i.e. separate the two populations
    assert stream_edge - bin_width <= break_point <= 0.30, \
        "the break point at {:.5f} does not separate the two populations".format(break_point)


def test_breakpoint_rejects_too_few_values():
    """ A break point cannot be located from fewer than three values. """

    try:
        thresholdBreakPoint([0.1, 0.2])
        raise AssertionError("expected a ValueError for two D values")

    except ValueError:
        pass


def test_reliability_threshold_excludes_chance_pairs():
    """ The threshold derived from shuffled samples must be small enough that, at the stated
        reliability, the shuffled samples contain no associated pair.
    """

    orbits = np.array(_randomOrbits(30, random_state=22))

    def d_func(q1, e1, i1, O1, w1, q2, e2, i2, O2, w2):
        return calcRho2(q1, e1, i1, O1, w1, q2, e2, i2, O2, w2)

    reliability = 0.9
    threshold = thresholdReliability(orbits, d_func, reliability=reliability, n_trials=60,
        random_state=1)

    assert threshold > 0.0, "the threshold came out non-positive"

    # Re-draw shuffled samples and count how often any pair falls below the threshold
    rng = np.random.RandomState(99)
    n_with_pair = 0
    n_trials = 60

    for _ in range(n_trials):

        synthetic = np.empty_like(orbits)
        for param in range(orbits.shape[1]):
            synthetic[:, param] = rng.permutation(orbits[:, param])

        found = False
        for a in range(len(synthetic) - 1):
            for b in range(a + 1, len(synthetic)):
                if float(d_func(*np.concatenate((synthetic[a], synthetic[b])))) < threshold:
                    found = True
                    break
            if found:
                break

        n_with_pair += found

    empirical = 1.0 - n_with_pair/float(n_trials)

    assert empirical > reliability - 0.15, \
        "only {:.0f}% of shuffled samples were free of pairs, the threshold promised {:.0f}%".format(
            100*empirical, 100*reliability)


def test_reliability_threshold_is_monotonic_in_reliability():
    """ Demanding a higher reliability cannot raise the threshold. """

    orbits = np.array(_randomOrbits(25, random_state=23))

    def d_func(*args):
        return calcRho2(*args)

    low = thresholdReliability(orbits, d_func, reliability=0.5, n_trials=40, random_state=2)
    high = thresholdReliability(orbits, d_func, reliability=0.99, n_trials=40, random_state=2)

    assert high <= low + 1e-12, \
        "the 99% threshold {:.6f} exceeded the 50% threshold {:.6f}".format(high, low)


def test_reliability_threshold_rejects_bad_inputs():
    """ A reliability outside (0, 1) and a sample of fewer than two orbits are both errors. """

    orbits = np.array(_randomOrbits(5, random_state=24))

    for bad in (0.0, 1.0, -0.1, 1.5):
        try:
            thresholdReliability(orbits, calcRho2, reliability=bad, n_trials=2)
            raise AssertionError("expected a ValueError for reliability {!r}".format(bad))
        except ValueError:
            pass

    try:
        thresholdReliability(orbits[:1], calcRho2, n_trials=2)
        raise AssertionError("expected a ValueError for a single orbit")
    except ValueError:
        pass


if __name__ == "__main__":

    test_functions = [
        test_Dr_matches_its_anchor_and_scaling,
        test_Dr_reproduces_jopek_bronikowska_table3,
        test_random_pairing_reproduces_bolide_formula,
        test_random_pairing_reproduces_table7,
        test_random_pairing_ratios_match_table8,
        test_Dr_is_far_looser_than_the_random_pairing_threshold,
        test_random_pairing_rejects_an_unpublished_combination,
        test_random_pairing_accepts_arrays,
        test_random_pairing_covers_every_published_combination,
        test_breakpoint_finds_a_known_break,
        test_breakpoint_rejects_too_few_values,
        test_reliability_threshold_excludes_chance_pairs,
        test_reliability_threshold_is_monotonic_in_reliability,
        test_reliability_threshold_rejects_bad_inputs,
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
