""" Tests for how far apart along the path of a fragment its stations see it at the same times, which the
    comparison of the fragments of ECSV files gives and notes when the stations seem to have measured
    different fragments.

Run with pytest:
    python -m pytest wmpl/Formats/Tests/test_ECSVFragmentTiming.py -v
"""

from __future__ import print_function, division, absolute_import

from types import SimpleNamespace

import numpy as np

from wmpl.Formats.ECSV import fragmentComparison, projectionMismatch, stationsApart


# Julian date of the reference time, frame rate (frames per second) and speed (m/s)
JD0 = 2461000.5
FPS = 25.0
SPEED = 20000.0



def _solution(tracks):
    """ Make a solution along a straight vertical path, with the points of every station at the given
        distances along it at the given times.

    Arguments:
        tracks: [dict] {station ID: (times (s), distances along the path (m))}.

    Return:
        [SimpleNamespace] The solution, with the attributes fragmentComparison() uses.
    """

    observations = []
    for station_id, (times, along) in sorted(tracks.items()):

        model_eci = np.array([[6.5e6 - distance, 0.0, 0.0] for distance in along])
        observations.append(SimpleNamespace(station_id=station_id, JD_data=JD0 + times/86400, \
            model_eci=model_eci, time_data=times, ignore_list=np.zeros(len(times), dtype=int), \
            ignore_station=False))

    first_jd = min(obs.JD_data[0] for obs in observations)

    return SimpleNamespace(observations=observations, radiant_eci_mini=np.array([1.0, 0.0, 0.0]), \
        state_vect_mini=np.array([6.5e6, 0.0, 0.0]), orbit=None, uncertainties=None, v_init=SPEED, \
        rbeg_ele=1e5, rend_ele=5e4, rbeg_jd=first_jd, jdt_ref=JD0)



def _fragment(shifts, t0=1.0, t1=2.0):
    """ Make a fragment seen by stations at distances along the path shifted by the given amounts, on frames
        of their own.

    Arguments:
        shifts: [dict] {station ID: shift along the path (m)}.

    Keyword arguments:
        t0: [float] Time of the first frame (s). 1 by default.
        t1: [float] Time after the last frame (s). 2 by default.

    Return:
        [SimpleNamespace] The solution.
    """

    tracks = {}
    for i, (station_id, shift) in enumerate(sorted(shifts.items())):
        t = np.arange(t0 + 0.01*i, t1, 1/FPS)
        tracks[station_id] = (t, SPEED*t - 300.0*t**2 + shift)

    return _solution(tracks)



def testTheDistanceAlongThePathOfTwoStationsIsMeasured():
    """ Check that the distance between the positions of two stations along the path is measured despite an
        outlier, and that too few points give none.
    """

    t_a = np.linspace(0.0, 1.0, 20)
    t_b = np.linspace(0.02, 0.98, 25)
    rng = np.random.default_rng(3)

    along_a = 20000*t_a - 1500*t_a**2 + rng.normal(0, 20.0, len(t_a))
    along_b = 20000*t_b - 1500*t_b**2 + 300.0 + rng.normal(0, 20.0, len(t_b))
    along_b[5] += 5000.0

    assert abs(projectionMismatch(t_a, along_a, t_b, along_b) - 300.0) < 20.0
    assert projectionMismatch(t_a[:1], along_a[:1], t_b[:2], along_b[:2]) is None



def testStationsSeeingTheSameFragmentAreTogether():
    """ Check that two stations which see the fragment at the same place at the same times are given close
        together, without a note.
    """

    t = np.arange(0, 3.0, 1/FPS)
    main = _solution({'XX0001': (t, SPEED*t)})
    fragment = _fragment({'XX0001': 0.0, 'XX0002': 0.0})

    # Points not used in the trajectory, far off, do not count
    obs = fragment.observations[1]
    obs.ignore_list[:10] = 1
    obs.model_eci[:10, 0] -= 3000.0

    assert stationsApart(fragment)[('XX0001', 'XX0002')] < 1.0

    report = fragmentComparison(main, [(1, fragment)])
    assert "Stations apart        : XX0001-XX0002 0 m" in report
    assert "km apart" not in report



def testStationsSeeingDifferentFragmentsAreNoted():
    """ Check that two stations which see the fragment 1.5 km apart along the path at the same times are
        noted, and that the time offsets estimated for the fragment, which would absorb it, give nothing.
    """

    t = np.arange(0, 3.0, 1/FPS)
    main = _solution({'XX0001': (t, SPEED*t)})
    fragment = _fragment({'XX0001': 0.0, 'XX0002': 1500.0})

    assert abs(stationsApart(fragment)[('XX0001', 'XX0002')] - 1500.0) < 1.0

    report = fragmentComparison(main, [(1, fragment)])
    assert "Stations apart        : XX0001-XX0002 1500 m" in report
    assert "NOTE: more than 1.0 km apart" in report

    report = fragmentComparison(main, [(1, fragment)], reuse_timing=False)
    assert "Stations apart" not in report
    assert "km apart" not in report



def testEveryTwoOfThreeStationsAreGiven():
    """ Check that every two of three stations are given, the one which saw another fragment far from the
        others.
    """

    t = np.arange(0, 3.0, 1/FPS)
    main = _solution({'XX0001': (t, SPEED*t)})
    fragment = _fragment({'XX0001': 0.0, 'XX0002': 0.0, 'XX0003': -2000.0})

    apart = stationsApart(fragment)
    assert sorted(apart) == [('XX0001', 'XX0002'), ('XX0001', 'XX0003'), ('XX0002', 'XX0003')]
    assert apart[('XX0001', 'XX0002')] < 1.0
    assert abs(apart[('XX0002', 'XX0003')] - 2000.0) < 1.0

    report = fragmentComparison(main, [(1, fragment)])
    assert "XX0001-XX0002 0 m, XX0001-XX0003 2000 m, XX0002-XX0003 2000 m" in report
    assert "NOTE: more than 1.0 km apart" in report

    # A station the solver ignored does not count
    fragment.observations[2].ignore_station = True
    assert sorted(stationsApart(fragment)) == [('XX0001', 'XX0002')]
