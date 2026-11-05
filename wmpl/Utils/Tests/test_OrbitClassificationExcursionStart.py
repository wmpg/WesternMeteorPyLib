""" Tests that the state a clone starts from does not count as time beyond the Jupiter-family limits. """

import math

import numpy as np

from wmpl.Utils.OrbitClassification import calcExcursionFractions


Q_LOW, Q_HIGH = 1.0, 3.0
A_LOW, A_HIGH = 3.0, 10.0


def test_clone_started_beyond_the_limit_counts_nothing_if_it_leaves_at_once():

    t = np.linspace(0.0, -10000.0, 201)
    q = np.full(201, Q_LOW)
    a = np.full(201, A_LOW)
    q[0] = Q_HIGH
    a[0] = A_HIGH

    assert calcExcursionFractions(t, q, a) == (0.0, 0.0)


def test_short_run_is_not_dominated_by_its_start():

    # The case that used to give f_a = 0.25: 4 intervals, beyond the limit only at the start
    t = np.linspace(0.0, -200.0, 5)
    a = np.array([A_HIGH, A_LOW, A_LOW, A_LOW, A_LOW])

    assert calcExcursionFractions(t, np.full(5, Q_LOW), a)[1] == 0.0


def test_clone_that_stays_beyond_the_limit_counts_all_the_time():

    t = np.linspace(0.0, -100.0, 11)

    assert calcExcursionFractions(t, np.full(11, Q_HIGH), np.full(11, A_HIGH)) == (1.0, 1.0)


def test_clone_that_rises_at_the_first_output_counts_from_the_start():

    t = np.linspace(0.0, -100.0, 11)
    q = np.full(11, Q_HIGH)
    q[0] = Q_LOW

    assert calcExcursionFractions(t, q, np.full(11, A_LOW))[0] == 1.0


def test_run_with_one_interval_uses_the_output_after_the_start():

    t = np.array([0.0, -50.0])

    assert calcExcursionFractions(t, np.array([Q_HIGH, Q_LOW]), np.array([A_LOW, A_HIGH])) == (0.0, 1.0)


def test_clone_with_no_output_after_the_start_has_no_time():

    t = np.linspace(0.0, -100.0, 11)
    q = np.full(11, np.nan)
    a = np.full(11, np.nan)
    q[0] = Q_HIGH
    a[0] = A_HIGH

    f_q, f_a = calcExcursionFractions(t, q, a)
    assert math.isnan(f_q) and math.isnan(f_a)
