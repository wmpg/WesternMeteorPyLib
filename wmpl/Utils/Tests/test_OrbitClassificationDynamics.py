""" Tests for the Jupiter-family dynamics metrics of an integrated ensemble of clones.

The metrics are checked on synthetic ensembles whose answer is known by construction: series that
change only at the outputs, clones that end early or leave the solar system, and encounter
sequences placed on either side of a crossing. The binomial interval is checked against its closed
form at k = 0 and against its symmetry.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_OrbitClassificationDynamics.py -v

or standalone (no pytest required):
    python -m wmpl.Utils.Tests.test_OrbitClassificationDynamics
"""

import math

import numpy as np

from wmpl.Utils.OrbitClassification import (calcKozaiEnergy, calcExcursionFractions,
    countPerihelionReach, splitPerihelionReachByEncounter, calcBinomialInterval,
    classifyPerihelionReach, JFC_DYNAMICS_Q_LIMIT, JFC_DYNAMICS_A_LIMIT)


# Ten one-year intervals of a backward integration
T_BACKWARD = -np.arange(11, dtype=float)

Q_LOW = 1.0
Q_HIGH = JFC_DYNAMICS_Q_LIMIT + 0.5
A_LOW = 3.0
A_HIGH = JFC_DYNAMICS_A_LIMIT + 1.0


def _raises(exception, func, *args, **kwargs):
    """ Return True if calling func raises the given exception. """

    try:
        func(*args, **kwargs)

    except exception:
        return True

    return False


def test_excursion_fractions_are_exact_for_series_that_change_at_the_outputs():

    # Beyond both limits for the first 3 of 10 intervals, then within them
    q = np.full(11, Q_LOW)
    q[:3] = Q_HIGH
    a = np.full(11, A_LOW)
    a[:3] = A_HIGH

    f_q, f_a = calcExcursionFractions(T_BACKWARD, q, a)
    assert f_q == 0.3
    assert f_a == 0.3

    # A second clone that never leaves halves the ensemble fraction
    f_q, f_a = calcExcursionFractions(T_BACKWARD, [q, np.full(11, Q_LOW)], [a, np.full(11, A_LOW)])
    assert f_q == 0.15
    assert f_a == 0.15


def test_metrics_do_not_depend_on_the_direction_of_the_integration():

    rng = np.random.default_rng(1)
    q = rng.uniform(0.5, 4.0, size=(5, 11))
    a = rng.uniform(2.0, 9.0, size=(5, 11))
    encounters = [rng.uniform(0, 10, size=2) for _ in range(5)]

    t_forward = -T_BACKWARD
    encounters_forward = [-times for times in encounters]

    assert calcExcursionFractions(T_BACKWARD, q, a) == calcExcursionFractions(t_forward, q, a)
    assert splitPerihelionReachByEncounter(-t_forward, q, encounters) \
        == splitPerihelionReachByEncounter(t_forward, q, encounters_forward)


def test_excursion_fraction_is_unchanged_by_finer_sampling_of_the_same_series():

    # A perihelion that steps up at 4 yr, sampled every year and every half year
    def q_of(t):
        return np.where(np.abs(t) >= 4.0, Q_HIGH, Q_LOW)

    t_coarse = -np.arange(0, 10.5, 1.0)
    t_fine = -np.arange(0, 10.25, 0.5)

    f_coarse = calcExcursionFractions(t_coarse, q_of(t_coarse), np.full_like(t_coarse, A_LOW))
    f_fine = calcExcursionFractions(t_fine, q_of(t_fine), np.full_like(t_fine, A_LOW))

    assert f_coarse == f_fine == (0.6, 0.0)


def test_unbound_and_escaped_clones_count_beyond_the_limits():

    q = np.full(11, Q_LOW)
    a = np.full(11, A_LOW)

    # Unbound from 5 yr on: a negative semi-major axis, q still finite and low
    a_unbound = a.copy()
    a_unbound[5:] = -4.0
    assert calcExcursionFractions(T_BACKWARD, q, a_unbound) == (0.0, 0.5)

    # Out of the solar system from 5 yr on
    q_escaped = q.copy()
    q_escaped[5:] = np.inf
    a_escaped = a.copy()
    a_escaped[5:] = np.inf
    assert calcExcursionFractions(T_BACKWARD, q_escaped, a_escaped) == (0.5, 0.5)
    assert countPerihelionReach(q_escaped) == (1, 1)


def test_clone_that_ended_contributes_only_the_time_it_was_followed():

    # Beyond the perihelion limit at its 5 outputs, then gone. Each output is held until the next,
    #   so the clone counts as followed for 5 intervals
    q_ended = np.full(11, np.nan)
    q_ended[:5] = Q_HIGH
    a_ended = np.full(11, np.nan)
    a_ended[:5] = A_LOW

    f_q, _ = calcExcursionFractions(T_BACKWARD, q_ended, a_ended)
    assert f_q == 1.0

    # Pooled with a clone followed for all 10 intervals and never beyond the limit
    f_q, _ = calcExcursionFractions(T_BACKWARD, [q_ended, np.full(11, Q_LOW)],
        [a_ended, np.full(11, A_LOW)])
    assert f_q == 5.0/15.0


def test_clone_without_any_state_is_not_counted():

    q = [np.full(11, Q_HIGH), np.full(11, np.nan)]
    a = [np.full(11, A_LOW), np.full(11, np.nan)]

    assert countPerihelionReach(q) == (1, 1)
    assert calcExcursionFractions(T_BACKWARD, q, a)[0] == 1.0

    f_q, f_a = calcExcursionFractions(T_BACKWARD, np.full(11, np.nan), np.full(11, np.nan))
    assert math.isnan(f_q) and math.isnan(f_a)


def test_reach_counts_a_single_output_beyond_the_limit():

    q = np.full((3, 11), Q_LOW)
    q[0, 7] = Q_HIGH
    q[1, -1] = Q_HIGH

    assert countPerihelionReach(q) == (2, 3)
    assert countPerihelionReach(q, q_limit=Q_HIGH) == (0, 3)


def test_split_by_encounter_uses_the_first_output_beyond_the_limit():

    # Every clone first gets beyond the limit at the 5 yr output
    q = np.full((5, 11), Q_LOW)
    q[:4, 5:] = Q_HIGH

    encounters = [
        [-3.0],         # before the crossing: lifted by the encounter
        [-7.0],         # after it: lifted without one
        [-5.0],         # at the output that first shows the crossing: counted as before
        [],             # no encounter at all
        [-2.0],         # never gets beyond the limit, so it is in neither count
        ]

    n_with, n_without = splitPerihelionReachByEncounter(T_BACKWARD, q, encounters)

    assert (n_with, n_without) == (2, 2)
    assert n_with + n_without == countPerihelionReach(q)[0]


def test_inputs_that_cannot_be_interpreted_are_refused():

    q = np.full((2, 11), Q_LOW)

    # One list of encounters per clone
    assert _raises(ValueError, splitPerihelionReachByEncounter, T_BACKWARD, q, [[]])

    # Outputs that do not move away from the epoch have no integration order
    t_unordered = T_BACKWARD.copy()
    t_unordered[[3, 4]] = t_unordered[[4, 3]]
    assert _raises(ValueError, calcExcursionFractions, t_unordered, q, q)

    # A series sampled at other times
    assert _raises(ValueError, calcExcursionFractions, T_BACKWARD, q[:, :-1], q[:, :-1])

    # An interval needs at least one trial
    assert _raises(ValueError, calcBinomialInterval, 0, 0)


def test_binomial_interval_closed_form_at_zero_successes():

    # At k = 0 the Wilson interval is [0, z^2/(n + z^2)]
    z = 1.959963984540054
    for n in (10, 50, 1000):
        low, high = calcBinomialInterval(0, n)
        assert low == 0.0
        assert abs(high - z**2/(n + z**2)) < 1e-12


def test_binomial_interval_is_symmetric_and_narrows_with_more_trials():

    for k, n in ((0, 50), (7, 50), (25, 50), (85, 100)):
        low, high = calcBinomialInterval(k, n)
        low_mirror, high_mirror = calcBinomialInterval(n - k, n)
        assert abs(low - (1.0 - high_mirror)) < 1e-12
        assert abs(high - (1.0 - low_mirror)) < 1e-12
        assert low <= float(k)/n <= high

    width_50 = np.diff(calcBinomialInterval(25, 50))[0]
    width_1000 = np.diff(calcBinomialInterval(500, 1000))[0]
    assert width_1000 < width_50


def test_classification_gives_both_verdicts():

    stable = classifyPerihelionReach(0, 50)
    assert stable['dynamics_class'] == 'not jupiter family'
    assert not stable['jupiter_family_shober']

    chaotic = classifyPerihelionReach(50, 50)
    assert chaotic['dynamics_class'] == 'jupiter family'
    assert chaotic['jupiter_family_shober']

    # Just above half: the rule of Shober et al. calls it Jupiter-family, the interval cannot tell
    marginal = classifyPerihelionReach(26, 50)
    assert marginal['jupiter_family_shober']
    assert marginal['dynamics_class'] == 'undetermined'

    # Exactly half is not above the threshold
    assert not classifyPerihelionReach(25, 50)['jupiter_family_shober']

    # Enough clones settle a fraction that 50 clones leave open
    assert classifyPerihelionReach(600, 1000)['dynamics_class'] == 'jupiter family'


def test_kozai_energy_analytic_values():

    assert calcKozaiEnergy(0.0, 0.0) == 1.0
    assert abs(calcKozaiEnergy(0.6, np.radians(60.0)) - 0.4) < 1e-12

    # Retrograde orbits have a negative H of the same size as their prograde mirror
    assert abs(calcKozaiEnergy(0.6, np.radians(120.0)) + 0.4) < 1e-12

    # Works on arrays
    h = calcKozaiEnergy(np.array([0.0, 0.6]), np.array([0.0, np.radians(60.0)]))
    assert np.allclose(h, [1.0, 0.4])


if __name__ == "__main__":

    test_functions = [
        test_excursion_fractions_are_exact_for_series_that_change_at_the_outputs,
        test_metrics_do_not_depend_on_the_direction_of_the_integration,
        test_excursion_fraction_is_unchanged_by_finer_sampling_of_the_same_series,
        test_unbound_and_escaped_clones_count_beyond_the_limits,
        test_clone_that_ended_contributes_only_the_time_it_was_followed,
        test_clone_without_any_state_is_not_counted,
        test_reach_counts_a_single_output_beyond_the_limit,
        test_split_by_encounter_uses_the_first_output_beyond_the_limit,
        test_inputs_that_cannot_be_interpreted_are_refused,
        test_binomial_interval_closed_form_at_zero_successes,
        test_binomial_interval_is_symmetric_and_narrows_with_more_trials,
        test_classification_gives_both_verdicts,
        test_kozai_energy_analytic_values,
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
