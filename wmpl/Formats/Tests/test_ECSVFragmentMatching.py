""" Tests for the matching of the additional fragments of ECSV files between stations, by where their
    projections on the main trajectory are along its path at the same times, whatever their numbers. Noiseless
    events with a known truth, solved with the real solver.

Run with pytest:
    python -m pytest wmpl/Formats/Tests/test_ECSVFragmentMatching.py -v
"""

from __future__ import print_function, division, absolute_import

import datetime
import os
import re
import shutil

import numpy as np

from wmpl.Formats.ECSV import fragmentMatches, fragmentMatchReport
from wmpl.Formats.Tests.test_ECSVUnsolvedFragments import META, MAIN, FRAGMENT, STATIONS, T0, FPS, N_FRAMES, \
    FRAGMENT_START, DIRECTION, azAlt, mainPoint, solveMain



def writeEvent(dir_path, numbers, main_stations=('XX0001', 'XX0002', 'XX0003'), frames=None):
    """ Write the ECSV files of an event without noise: the main fragment, and from the frame FRAGMENT_START
        on, fragments on its path behind it, which every station numbers in its own way.

    Arguments:
        dir_path: [str] Directory of the files.
        numbers: [dict] Fragments of every station, {station code: {number: distance behind the main
            fragment (m)}}.

    Keyword arguments:
        main_stations: [tuple] Stations which see the main fragment. XX0001 to XX0003 by default.
        frames: [dict] First and last frames on which some fragments are seen, {station code: {number:
            (first, last)}}. None by default, for all from FRAGMENT_START to the last frame.

    Return:
        [list] Paths of the files.
    """

    paths = []
    for name in sorted(set(main_stations) | set(numbers)):

        lat, lon, ele = STATIONS[name]
        fragments = numbers.get(name, {})
        seen = (frames or {}).get(name, {})

        rows = []
        for i in range(N_FRAMES):

            when = (T0 + datetime.timedelta(seconds=i/FPS)).strftime("%Y-%m-%dT%H:%M:%S.%f")
            row = ['']*len(MAIN)
            if name in main_stations:
                azim, alt = azAlt(mainPoint(i), name)
                row = [when, '10.0', '-30.0', '{:.8f}'.format(azim), '{:.8f}'.format(alt), '100', '200', \
                    '2.0']

            row += [str(i), 'False', 'True']

            for number in sorted(fragments):

                fragment = ['']*len(FRAGMENT)
                first, last = seen.get(number, (FRAGMENT_START, N_FRAMES - 1))
                if first <= i <= last:
                    azim, alt = azAlt(mainPoint(i) - fragments[number]*DIRECTION, name)
                    fragment = [when, '10.0', '-30.0', '{:.8f}'.format(azim), '{:.8f}'.format(alt), '100', \
                        '210']

                row += fragment

            rows.append(",".join(row))

        header = MAIN + ['frame_number', 'flare', 'trajectory_use'] \
            + [column + str(number) for number in sorted(fragments) for column in FRAGMENT]

        path = os.path.join(str(dir_path), name + '.ecsv')
        with open(path, 'w') as f:
            f.write(META.format(lat=lat, lon=lon, ele=ele, name=name) + ",".join(header) + "\n" \
                + "\n".join(rows) + "\n")

        paths.append(path)

    return paths



def reportRows(report):
    """ Keep the rows of the tables of a report of the matches, without its explanation.

    Arguments:
        report: [str] Report from fragmentMatchReport().

    Return:
        [str] The rows.
    """

    return "\n".join(line for line in report.splitlines() if line.startswith("  "))



def testCrossedNumbersAreMatched(tmp_path):
    """ Check that two fragments numbered the other way round by the second station are matched by where they
        are, and that the report notes it, while a station not in the main trajectory is left out.
    """

    numbers = {'XX0001': {1: 800.0, 2: 2500.0}, 'XX0002': {1: 2500.0, 2: 800.0}, 'XX0004': {1: 800.0}}
    paths = writeEvent(tmp_path, numbers)
    traj = solveMain(paths, tmp_path)

    matches = fragmentMatches(traj, paths, [1, 2])

    assert [match['stations'] for match in matches] == [('XX0001', 'XX0002')]

    pairs = matches[0]['pairs']
    assert [(pair[0], pair[1]) for pair in pairs] == [(1, 2), (2, 1)]
    assert all(pair[2] < 20.0 for pair in pairs)
    assert all(pair[3] is None for pair in pairs)

    # The same numbers are as far apart as the fragments
    assert abs(matches[0]['distance'][0, 0] - 1700.0) < 50.0

    rows = reportRows(fragmentMatchReport(matches))
    assert rows.count("NOTE: different numbers") == 2
    assert "NOTE: different numbers (1 at XX0002: 1.70 km)" in rows
    assert "ambiguous" not in rows



def testRightNumbersCloseToEachOtherAreAmbiguous(tmp_path):
    """ Check that fragments with the same numbers are matched, and that two fragments closer to each other
        than the margin are ambiguous, while two files of one camera are one station.
    """

    numbers = {'XX0001': {1: 800.0, 2: 1100.0}, 'XX0002': {1: 800.0, 2: 1100.0}}
    paths = writeEvent(tmp_path, numbers)

    # A second file of the camera of the first station
    second = str(tmp_path/"XX0001_second.ecsv")
    shutil.copy(paths[0], second)
    paths.append(second)

    matches = fragmentMatches(solveMain(paths, tmp_path), paths, [1, 2])

    assert [match['stations'] for match in matches] == [('XX0001', 'XX0002'), ('XX0001_2', 'XX0002')]

    pairs = matches[0]['pairs']
    assert [(pair[0], pair[1]) for pair in pairs] == [(1, 1), (2, 2)]
    assert all(abs(pair[3] - 300.0) < 20.0 for pair in pairs)

    rows = reportRows(fragmentMatchReport(matches[:1]))
    assert "different numbers" not in rows
    assert rows.count("ambiguous") == 2



def testAFragmentFarFromItsNumberIsNotMatched(tmp_path):
    """ Check that a fragment further than the limit from the one with its number at the other station is not
        matched, and that the report notes it.
    """

    numbers = {'XX0001': {1: 800.0}, 'XX0002': {1: 3000.0, 2: 5000.0}}
    paths = writeEvent(tmp_path, numbers)

    matches = fragmentMatches(solveMain(paths, tmp_path), paths, [1, 2])

    assert matches[0]['pairs'] == []

    rows = reportRows(fragmentMatchReport(matches))
    assert "NOTE: not matched, 1 at XX0002 is 2.20 km away" in rows
    assert "NOTE: not matched, 1 at XX0001 is 2.20 km away" in rows
    assert rows.count("not matched") == 3



def testAFragmentWhoseNumberIsMatchedWithAnotherOneSaysWhich(tmp_path):
    """ Check that a fragment whose number at the other station is close, but matched with another fragment,
        is reported as such and not noted as far.
    """

    numbers = {'XX0001': {1: 800.0, 2: 1000.0}, 'XX0002': {1: 1000.0}}
    paths = writeEvent(tmp_path, numbers)

    matches = fragmentMatches(solveMain(paths, tmp_path), paths, [1, 2])

    assert [(pair[0], pair[1]) for pair in matches[0]['pairs']] == [(2, 1)]

    rows = reportRows(fragmentMatchReport(matches))
    assert re.search(r"not matched, 1 at XX0002 \(\d+ m away\) is matched with 2", rows)
    assert "NOTE: not matched" not in rows
    assert "NOTE: different numbers, ambiguous" in rows



def testFragmentsNotSeenAtTheSameTimesAreNotCompared(tmp_path):
    """ Check that two fragments which two stations see at the same times for less than the least overlap are
        not compared, and that the report says it.
    """

    numbers = {'XX0001': {1: 800.0, 2: 2500.0}, 'XX0002': {1: 800.0, 2: 2500.0}}
    paths = writeEvent(tmp_path, numbers, frames={'XX0001': {1: (FRAGMENT_START, FRAGMENT_START + 10)}, \
        'XX0002': {1: (FRAGMENT_START + 11, N_FRAMES - 1)}})

    matches = fragmentMatches(solveMain(paths, tmp_path), paths, [1, 2])

    assert [(pair[0], pair[1]) for pair in matches[0]['pairs']] == [(2, 2)]
    assert np.isnan(matches[0]['distance'][0, 0])

    rows = reportRows(fragmentMatchReport(matches))
    assert "not matched, 1 at XX0002 is not seen at the same times" in rows
    assert "not matched, 1 at XX0001 is not seen at the same times" in rows
