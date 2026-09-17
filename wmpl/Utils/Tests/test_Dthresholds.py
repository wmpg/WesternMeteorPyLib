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
from wmpl.Utils.Dthresholds import thresholdDr, thresholdBreakPoint, thresholdReliability
from wmpl.Utils.Tests.test_DcriteriaAdditional import _randomOrbits


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
