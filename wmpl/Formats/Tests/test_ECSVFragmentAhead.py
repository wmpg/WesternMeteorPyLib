""" Tests for how far ahead of the main fragment the comparison of the fragments of ECSV files puts each
    one, with a main fragment that slows down, and for how far off the main path.

Run with pytest:
    python -m pytest wmpl/Formats/Tests/test_ECSVFragmentAhead.py -v
"""

from __future__ import print_function, division, absolute_import

import re
from types import SimpleNamespace

import numpy as np

from wmpl.Formats.ECSV import fragmentComparison, localMotionFit


# Julian date of the first point, frame rate (frames per second) and initial speed (m/s)
JD0 = 2461000.5
FPS = 25.0
SPEED = 20000.0



def _solution(times, along):
    """ Make a solution along a straight vertical path, with the given distance along it at the given times.

    Arguments:
        times: [ndarray] Times of the points (s).
        along: [ndarray] Distance along the path of every point (m).

    Return:
        [SimpleNamespace] The solution, with the attributes fragmentComparison() uses.
    """

    model_eci = np.array([[6.5e6 - distance, 0.0, 0.0] for distance in along])
    obs = SimpleNamespace(JD_data=JD0 + times/86400, model_eci=model_eci, time_data=times, \
        ignore_list=np.zeros(len(times), dtype=int), ignore_station=False)

    return SimpleNamespace(observations=[obs], radiant_eci_mini=np.array([1.0, 0.0, 0.0]), \
        state_vect_mini=np.array([6.5e6, 0.0, 0.0]), orbit=None, uncertainties=None, v_init=SPEED, \
        rbeg_ele=1e5, rend_ele=5e4, rbeg_jd=obs.JD_data[0])



def _mainAlong(t, lag):
    """ Compute the distance along the path of a fragment slowing down exponentially.

    Arguments:
        t: [ndarray] Times (s).
        lag: [float] Lag behind the constant speed at the first point (m).

    Return:
        [ndarray] Distance along the path (m).
    """

    return SPEED*t - lag*np.exp(2.0*t)



def _aheadOf(main, fragment):
    """ Read how far ahead of the main fragment the comparison puts the fragment.

    Arguments:
        main: [SimpleNamespace] Solution of the main fragment.
        fragment: [SimpleNamespace] Solution of the fragment.

    Return:
        (first, last): [tuple] Distance ahead (m) at the first and the last point of the fragment.
    """

    report = fragmentComparison(main, [(1, fragment)])
    match = re.search(r"Ahead of the main one : ([+-]\d+) m at the fragment's first point, ([+-]\d+) m", \
        report)

    return float(match.group(1)), float(match.group(2))



def testAFragmentMovingWithAMainFragmentSlowingDownStaysWithIt():
    """ Check that a fragment moving with the main one, which slows down, is not ahead of it. """

    t = np.arange(0, 3.0, 1/FPS)
    main = _solution(t, _mainAlong(t, lag=5.0))

    late = t[t >= 1.4]
    first, last = _aheadOf(main, _solution(late, _mainAlong(late, lag=5.0)))

    assert abs(first) < 5.0
    assert abs(last) < 5.0



def testAFragmentFallingBehindIsReportedBehind():
    """ Check that a fragment falling behind the main one at 200 m/s is reported behind it. """

    t = np.arange(0, 3.0, 1/FPS)
    main = _solution(t, _mainAlong(t, lag=5.0))

    late = t[t >= 1.4]
    first, last = _aheadOf(main, _solution(late, _mainAlong(late, lag=5.0) - 200.0*(late - late[0])))

    assert abs(first) < 5.0
    assert abs(last - (-200.0*(late[-1] - late[0]))) < 5.0



def testAFragmentWithFewPoints():
    """ Check that a fragment with 4 points, 100 m ahead of the main one, is reported 100 m ahead. """

    t = np.arange(0, 1.0, 0.1)
    main = _solution(t, _mainAlong(t, lag=0.0))

    late = t[-4:]
    first, last = _aheadOf(main, _solution(late, _mainAlong(late, lag=0.0) + 100.0))

    assert abs(first - 100.0) < 1.0
    assert abs(last - 100.0) < 1.0



def testAFragmentSeenAfterTheMainOneSaysItIsExtrapolated():
    """ Check that the comparison says when the main fragment is extrapolated to the times of the fragment.
    """

    t = np.arange(0, 2.0, 1/FPS)
    main = _solution(t, _mainAlong(t, lag=0.0))

    inside = t[t >= 1.0]
    report = fragmentComparison(main, [(1, _solution(inside, _mainAlong(inside, lag=0.0)))])
    assert "extrapolated" not in report

    after = np.arange(1.5, 2.5, 1/FPS)
    report = fragmentComparison(main, [(1, _solution(after, _mainAlong(after, lag=0.0)))])
    assert "seen 0.50 s outside the main one" in report



def testTheLocalFitFollowsThePointsAndTakesFewOfThem():
    """ Check that the local fit follows a fragment slowing down, extrapolates past its last point, and
        works with only 2 points.
    """

    t = np.arange(0, 3.0, 1/FPS)
    along = _mainAlong(t, lag=5.0)

    assert abs(localMotionFit(t, along)(2.9) - _mainAlong(2.9, lag=5.0)) < 5.0

    # Past the last point it extrapolates from the last part of the path
    assert abs(localMotionFit(t, along)(3.1) - _mainAlong(3.1, lag=5.0)) < 100.0

    assert np.isclose(localMotionFit(t[:2], along[:2])(0.02), np.interp(0.02, t[:2], along[:2]))



def testTheLocalFitPastTheEndOfSeveralStations():
    """ Check that past the end of a path seen by several stations, the local fit extrapolates from its last
        part, also when the stations share the times of their points.
    """

    # Twelve synchronised stations, with the same frame times
    t = np.tile(np.arange(0, 2.0, 1/FPS), 12)
    assert abs(localMotionFit(t, SPEED*t)(2.5) - SPEED*2.5) < 1.0

    # Five stations with frames at other times, and 10 m of noise
    rng = np.random.default_rng(1)
    t = np.concatenate([np.arange(rng.uniform(0, 1/FPS), 2.0, 1/FPS) for _ in range(5)])
    along = _mainAlong(t, lag=5.0) + rng.normal(0, 10.0, len(t))
    assert abs(localMotionFit(t, along)(t.max() + 0.5) - _mainAlong(t.max() + 0.5, lag=5.0)) < 400.0



def testTheLocalFitTakesDistinctTimes():
    """ Check that the degree of the local fit is limited by the number of distinct times, not of points. """

    # Six stations sharing the times of two frames
    t = np.repeat([0.0, 1/FPS], 6)
    assert np.isclose(localMotionFit(t, SPEED*t)(0.5/FPS), SPEED*0.5/FPS)



def testAFragmentFarOffTheMainPathIsNoted():
    """ Check that the comparison notes a fragment further off the main path than any fragment of the main
        one in simulations, and not one closer to it.
    """

    t = np.arange(0, 2.0, 1/FPS)
    main = _solution(t, _mainAlong(t, lag=0.0))

    reports = []
    for offset in (1000.0, 2000.0):

        later = t[t >= 1.0]
        fragment = _solution(later, _mainAlong(later, lag=0.0))
        fragment.observations[0].model_eci[:, 1] += offset

        reports.append(fragmentComparison(main, [(1, fragment)]))

    assert "off the main path, where no fragment" not in reports[0]
    assert "more than 1.5 km off the main path" in reports[1]
