""" Tests for how far ahead of the main fragment the comparison of the fragments of ECSV files puts each one, with a 
    main fragment that slows down.

Run with pytest:
    python -m pytest wmpl/Formats/Tests/test_ECSVFragmentAhead.py -v
"""

from __future__ import print_function, division, absolute_import

import re
from types import SimpleNamespace

import numpy as np

from wmpl.Formats.ECSV import fragmentComparison, localMotionFit


JD0 = 2461000.5
FPS = 25.0
SPEED = 20000.0


def solution(times, along):
    """ A solution along a straight vertical path, with the given distance along it at the given times. """

    model_eci = np.array([[6.5e6 - distance, 0.0, 0.0] for distance in along])
    obs = SimpleNamespace(JD_data=JD0 + times/86400, model_eci=model_eci, time_data=times,
        ignore_list=np.zeros(len(times), dtype=int), ignore_station=False)

    return SimpleNamespace(observations=[obs], radiant_eci_mini=np.array([1.0, 0.0, 0.0]),
        state_vect_mini=np.array([6.5e6, 0.0, 0.0]), orbit=None, uncertainties=None, v_init=SPEED,
        rbeg_ele=1e5, rend_ele=5e4, rbeg_jd=obs.JD_data[0])


def mainAlong(t, lag):

    return SPEED*t - lag*np.exp(2.0*t)


def aheadOf(main, fragment):
    """ How far ahead of the main fragment the comparison says the fragment is, at its first and last points. """

    report = fragmentComparison(main, [(1, fragment)])
    match = re.search(r"Ahead of the main one : ([+-]\d+) m at the fragment's first point, ([+-]\d+) m", report)

    return float(match.group(1)), float(match.group(2))


def test_a_fragment_moving_with_a_main_fragment_slowing_down_stays_with_it():

    t = np.arange(0, 3.0, 1/FPS)
    main = solution(t, mainAlong(t, lag=5.0))

    late = t[t >= 1.4]
    first, last = aheadOf(main, solution(late, mainAlong(late, lag=5.0)))

    assert abs(first) < 5.0
    assert abs(last) < 5.0


def test_a_fragment_falling_behind_is_reported_behind():

    t = np.arange(0, 3.0, 1/FPS)
    main = solution(t, mainAlong(t, lag=5.0))

    late = t[t >= 1.4]
    first, last = aheadOf(main, solution(late, mainAlong(late, lag=5.0) - 200.0*(late - late[0])))

    assert abs(first) < 5.0
    assert abs(last - (-200.0*(late[-1] - late[0]))) < 5.0


def test_a_fragment_with_few_points():

    t = np.arange(0, 1.0, 0.1)
    main = solution(t, mainAlong(t, lag=0.0))

    late = t[-4:]
    first, last = aheadOf(main, solution(late, mainAlong(late, lag=0.0) + 100.0))

    assert abs(first - 100.0) < 1.0
    assert abs(last - 100.0) < 1.0


def test_a_fragment_seen_after_the_main_one_says_it_is_extrapolated():

    t = np.arange(0, 2.0, 1/FPS)
    main = solution(t, mainAlong(t, lag=0.0))

    inside = t[t >= 1.0]
    assert "extrapolated" not in fragmentComparison(main, [(1, solution(inside, mainAlong(inside, lag=0.0)))])

    after = np.arange(1.5, 2.5, 1/FPS)
    assert "seen 0.50 s outside the main one" in fragmentComparison(main, [(1, solution(after, mainAlong(after, lag=0.0)))])


def test_the_local_fit_follows_the_points_and_takes_few_of_them():

    t = np.arange(0, 3.0, 1/FPS)
    along = mainAlong(t, lag=5.0)

    assert abs(localMotionFit(t, along)(2.9) - mainAlong(2.9, lag=5.0)) < 5.0

    # Past the last point it extrapolates from the closest ones
    assert abs(localMotionFit(t, along)(3.1) - mainAlong(3.1, lag=5.0)) < 100.0

    assert np.isclose(localMotionFit(t[:2], along[:2])(0.02), np.interp(0.02, t[:2], along[:2]))
