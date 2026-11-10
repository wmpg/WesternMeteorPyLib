""" Tests of reading ECSV files with loadECSVs(). The GFE standard (Appendix 4) keeps one row per frame, with
    additional fragments in columns with a numeric suffix (e.g. azimuth1), so the main fragment columns are
    empty on the frames on which only an additional fragment was measured, and main fragment points flagged
    with trajectory_use = False are left out of its trajectory. The azimuth and altitude of each row are
    converted to R.A./Dec. at the time of the row, which the solver has to convert back at that time.

Run with pytest:
    python -m pytest wmpl/Formats/Tests/test_ECSV.py -v
"""

from __future__ import print_function, division, absolute_import

import datetime
import os
import tempfile

from types import SimpleNamespace

import numpy as np

import matplotlib
matplotlib.use('Agg')

from wmpl.Formats.ECSV import ecsvFragments, loadECSVs, appliedTimeOffsets, solverStationIDs, \
    originalPicksTrajectory
from wmpl.Formats.GenericFunctions import solveTrajectoryGeneric
from wmpl.Formats.Tests.SyntheticECSV import FRAGMENT, writeFragmentECSV
from wmpl.Trajectory.Tests.SyntheticTrajectory import STRAIGHT_JDT_REF, STRAIGHT_STATIONS, \
    stationAzimAlt, straightMeteorPoints
from wmpl.Utils.Math import angleBetweenVectors
from wmpl.Utils.Pickling import savePickle
from wmpl.Utils.TrajConversions import datetime2JD, jd2Date



def testFragmentsAreFound(tmp_path):
    """ Check that the additional fragments are the suffixes of the azimuth columns, and that files without
        them have none.
    """

    assert ecsvFragments([writeFragmentECSV(tmp_path/"a.ecsv", "XX0001")]) == [1]
    assert ecsvFragments([writeFragmentECSV(tmp_path/"b.ecsv", "XX0002", fragments=False)]) == []



def testMainFragmentSkipsEmptyAndUnusedPoints(tmp_path):
    """ Check that of the 7 rows, the frame with only fragment 1 is skipped, and so is the main fragment
        point flagged as not to be used in the trajectory, leaving 5 points.
    """

    _, meteors = loadECSVs([writeFragmentECSV(tmp_path/"a.ecsv", "XX0001")], no_prepare=True)

    assert len(meteors) == 1
    assert np.allclose(np.degrees(meteors[0].azim_data), [120.0, 120.5, 121.5, 122.0, 122.5])



def testAdditionalFragmentColumns(tmp_path):
    """ Check that fragment 1 is read from its own columns, without the photometry of the main fragment. """

    _, meteors = loadECSVs([writeFragmentECSV(tmp_path/"a.ecsv", "XX0001")], no_prepare=True, fragment=1)

    assert len(meteors) == 1
    assert len(meteors[0].time_data) == 5
    assert np.isclose(np.degrees(min(meteors[0].elev_data)), 29.0 - 0.3*6)
    assert set(meteors[0].mag_data) == {10.0}



def testFilesWithoutTheFragmentAreSkipped(tmp_path):
    """ Check that only the stations whose files describe a fragment are loaded for it. """

    paths = [writeFragmentECSV(tmp_path/"a.ecsv", "XX0001"), \
        writeFragmentECSV(tmp_path/"b.ecsv", "XX0002", fragments=False)]

    _, main = loadECSVs(paths, no_prepare=True)
    _, fragment = loadECSVs(paths, no_prepare=True, fragment=1)

    assert sorted(meteor.station_id for meteor in main) == ['XX0001', 'XX0002']
    assert [meteor.station_id for meteor in fragment] == ['XX0001']



def testPreviousFormatIsUnchanged(tmp_path):
    """ Check that a file without the annotation columns loads all its points, as before. """

    _, meteors = loadECSVs([writeFragmentECSV(tmp_path/"b.ecsv", "XX0002", fragments=False)], \
        no_prepare=True)

    assert len(meteors[0].time_data) == 6
    assert list(meteors[0].mag_data) == [2.0, 1.9, 1.8, 1.7, 1.6, 1.5]



def testMainFragmentWithTheZeroSuffix(tmp_path):
    """ Check that the suffix 0 of the main fragment is optional, as in the GFE standard (azimuth0 is
        azimuth).
    """

    path = writeFragmentECSV(tmp_path/"b.ecsv", "XX0002", fragments=False)
    with open(path) as f:
        lines = f.read().splitlines()

    header = [i for i, line in enumerate(lines) if not line.startswith('#')][0]
    lines[header] = ",".join(name + '0' if name in FRAGMENT else name for name in lines[header].split(','))
    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")

    assert ecsvFragments([path]) == []

    _, meteors = loadECSVs([path], no_prepare=True)
    assert len(meteors[0].time_data) == 6



def testStationWithoutPointsIsSkipped(tmp_path):
    """ Check that a station whose main fragment points are all flagged as not to be used (in any letter
        case) is skipped, and so is a set of files without any points.
    """

    path = writeFragmentECSV(tmp_path/"a.ecsv", "XX0001")
    with open(path) as f:
        text = f.read()
    with open(path, 'w') as f:
        f.write(text.replace(',True,', ',false,'))

    jdt_ref, meteors = loadECSVs([path], no_prepare=True)
    assert (jdt_ref is None) and (meteors == [])

    # Fragment 1 is not flagged, so it still loads
    _, meteors = loadECSVs([path], no_prepare=True, fragment=1)
    assert len(meteors) == 1



def testSolverStationIDs():
    """ Check that a station added again gets the suffix _2, _3, ..., as in Trajectory.infillTrajectory(). """

    meteors = [SimpleNamespace(station_id=station_id) for station_id in ('A', 'B', 'A', 'A', 'B')]

    assert solverStationIDs(meteors) == ['A', 'B', 'A_2', 'A_3', 'B_2']

    # A station whose ID already ends in a suffix takes the next one, as in the solver
    meteors = [SimpleNamespace(station_id=station_id) for station_id in ('A', 'A_2', 'A')]

    assert solverStationIDs(meteors) == ['A', 'A_2', 'A_3']



def testAppliedTimeOffsetsOfAStationWithTwoFiles(tmp_path):
    """ Check that the offset applied to every observation is matched to its own input file, also when a
        station has two (e.g. one per video file), which the solver names XX0001 and XX0001_2.
    """

    path_a = writeFragmentECSV(tmp_path/"a.ecsv", "XX0001", fragments=False)
    path_b = writeFragmentECSV(tmp_path/"b.ecsv", "XX0001", fragments=False)

    # The second file has a point less, at other times
    with open(path_b) as f:
        lines = f.read().splitlines()
    with open(path_b, 'w') as f:
        f.write("\n".join(line.replace('2026-01-30T10:25:36.', '2026-01-30T10:25:37.') \
            for line in lines[:-1]) + "\n")

    _, meteors = loadECSVs([path_a, path_b], no_prepare=True)

    # Solved observations, with the times of the two files shifted by 0.25 s and -0.1 s
    observations = [SimpleNamespace(station_id=station_id, \
        JD_data=meteor.jdt_ref + (np.array(meteor.time_data) + shift)/86400) \
        for station_id, meteor, shift in zip(['XX0001', 'XX0001_2'], meteors, [0.25, -0.1])]

    offsets = appliedTimeOffsets(SimpleNamespace(observations=observations), meteors)

    assert sorted(offsets) == sorted([path_a, path_b])
    assert np.isclose(offsets[path_a], 0.25, atol=1e-4)
    assert np.isclose(offsets[path_b], -0.1, atol=1e-4)

    # A station the solver ignored has no offset to take
    observations[1].ignore_station = True

    assert sorted(appliedTimeOffsets(SimpleNamespace(observations=observations), meteors)) == [path_a]



def testOriginalPicksTrajectory(tmp_path):
    """ Check that with the Monte Carlo runs, for which the solver returns the best run, the solution with
        the original picks is the one saved with the results.
    """

    best = SimpleNamespace(uncertainties=SimpleNamespace(), save_results=True, output_dir=str(tmp_path), \
        file_name="20260130_102535", name="best run")
    savePickle(SimpleNamespace(name="original picks"), str(tmp_path), "20260130_102535_trajectory.pickle")

    assert originalPicksTrajectory(best).name == "original picks"

    # Without the Monte Carlo runs the solver returns the solution with the original picks itself
    no_mc = SimpleNamespace(**dict(best.__dict__, uncertainties=None))
    assert originalPicksTrajectory(no_mc) is no_mc

    # Without the saved results, there is no other solution than the given one
    not_saved = SimpleNamespace(**dict(best.__dict__, save_results=False))
    assert originalPicksTrajectory(not_saved) is not_saved




# Header with the station of the points
HEADER = """# %ECSV 0.9
# ---
# delimiter: ','
# meta: !!omap
# - {obs_latitude: 45.0}
# - {obs_longitude: 15.0}
# - {obs_elevation: 200.0}
# - {camera_id: 'XX0001'}
"""
COLUMNS = ['datetime', 'ra', 'dec', 'azimuth', 'altitude', 'x_image', 'y_image', 'mag_data']



def _point(i):
    """ Make the values of a point on the given frame of a meteor moving at 25 frames per second.

    Arguments:
        i: [int] Frame index.

    Return:
        [list] Values of the columns, as strings.
    """

    return ["2024-01-01T00:00:{:09.6f}".format(10 + i*0.04), "{:.6f}".format(100.0 + i*0.2), \
        "{:+.6f}".format(20.0 + i*0.1), "{:.6f}".format(90.0 + i*0.3), "{:+.6f}".format(40.0 - i*0.2), \
        "{:.3f}".format(300.0 + 5*i), "{:.3f}".format(200.0 + 3*i), "{:+.2f}".format(-2.0 - 0.1*i)]



def _writeECSV(dir_path, n_main, n_fragment_only):
    """ Write an ECSV file with an additional fragment, measured on all frames, and the main fragment,
        measured only on the first frames.

    Arguments:
        dir_path: [str] Directory where the file is written.
        n_main: [int] Number of frames with the main fragment.
        n_fragment_only: [int] Number of the following frames with only the additional fragment.

    Return:
        path: [str] Path to the file.
    """

    header = COLUMNS + [name + '1' for name in COLUMNS[1:]]

    rows = []
    for i in range(n_main + n_fragment_only):

        fragment = _point(i)[1:]

        # Leave the main fragment columns empty after its last frame, except the time
        if i < n_main:
            main = _point(i)
        else:
            main = [_point(i)[0]] + ['']*(len(COLUMNS) - 1)

        rows.append(",".join(main + fragment))

    path = os.path.join(dir_path, "test.ecsv")
    with open(path, "w") as f:
        f.write(HEADER + ",".join(header) + "\n" + "\n".join(rows) + "\n")

    return path



def testRowsWithoutTheMainFragmentAreSkipped():
    """ Check that the rows on which only the additional fragment was measured are skipped, and that the
        points of the main fragment are loaded as they are.
    """

    with tempfile.TemporaryDirectory() as tmp_dir:
        _, meteor_list = loadECSVs([_writeECSV(tmp_dir, 6, 3)], no_prepare=True)

    meteor = meteor_list[0]
    assert len(meteor.azim_data) == 6
    assert np.allclose(np.degrees(meteor.azim_data), [90.0 + i*0.3 for i in range(6)])
    assert np.allclose(np.array(meteor.mag_data, dtype=np.float64), [-2.0 - 0.1*i for i in range(6)])



def testFilesWithoutEmptyRowsAreUnchanged():
    """ Check that all points are loaded from a file in which the main fragment was measured on every frame.
    """

    with tempfile.TemporaryDirectory() as tmp_dir:
        _, meteor_list = loadECSVs([_writeECSV(tmp_dir, 9, 0)], no_prepare=True)

    assert len(meteor_list[0].azim_data) == 9




def _writeStraightMeteorECSVs(dir_path):
    """ Write the noise-free straight meteor of SyntheticTrajectory as seen from each of its stations into an
        ECSV file per station, with the azimuth and altitude at the time of each row.

    Arguments:
        dir_path: [str] Directory where the files are written.

    Return:
        (paths, radiant_eci):
            - paths: [list] Paths to the files.
            - radiant_eci: [ndarray] True radiant unit vector in ECI coordinates.
    """

    times, points, _, radiant_eci = straightMeteorPoints()

    # Compute the times of the rows to the microsecond, as they are written, and their Julian dates
    dt_ref = jd2Date(STRAIGHT_JDT_REF, dt_obj=True)
    dt_list = [dt_ref + datetime.timedelta(microseconds=int(round(1e6*t))) for t in times]
    jds = np.array([datetime2JD(dt) for dt in dt_list])

    paths = []
    for station_id, s_lat, s_lon, ele in STRAIGHT_STATIONS:

        azim, alt = stationAzimAlt(points, jds, s_lat, s_lon, ele)

        # Put the station into the header
        header = HEADER.replace("obs_latitude: 45.0", "obs_latitude: {:.6f}".format(s_lat))
        header = header.replace("obs_longitude: 15.0", "obs_longitude: {:.6f}".format(s_lon))
        header = header.replace("obs_elevation: 200.0", "obs_elevation: {:.1f}".format(ele))
        header = header.replace("XX0001", station_id)

        # Write a row per point, with the azimuth and altitude in degrees
        rows = []
        for i, dt in enumerate(dt_list):
            rows.append(",".join([dt.strftime("%Y-%m-%dT%H:%M:%S.%f"), "0.0", "0.0", \
                "{:.9f}".format(np.degrees(azim[i])), "{:+.9f}".format(np.degrees(alt[i])), "0.0", "0.0", \
                "+0.00"]))

        path = os.path.join(dir_path, "{:s}.ecsv".format(station_id))
        with open(path, "w") as f:
            f.write(header + ",".join(COLUMNS) + "\n" + "\n".join(rows) + "\n")

        paths.append(path)

    return paths, radiant_eci



def _solveStraightMeteor(radec_per_point):
    """ Solve the straight meteor from its ECSV files without noise.

    Arguments:
        radec_per_point: [bool] Passed to loadECSVs.

    Return:
        [float] Angle between the solved and the true radiant (arcsec).
    """

    with tempfile.TemporaryDirectory() as tmp_dir:

        paths, radiant_eci = _writeStraightMeteorECSVs(tmp_dir)
        jdt_ref, meteor_list = loadECSVs(paths, radec_per_point=radec_per_point)

        traj = solveTrajectoryGeneric(jdt_ref, meteor_list, tmp_dir, estimate_timing_vel=False, \
            gravity_correction=False, filter_picks=False, monte_carlo=False, calc_orbit=False, \
            show_plots=False, save_results=False, verbose=False)

    return 3600*np.degrees(angleBetweenVectors(traj.radiant_eci_mini, radiant_eci))



def testTheRaDecOfEachRowAreConvertedAtItsTime():
    """ Check that the straight meteor is recovered from its ECSV files. When the R.A./Dec. were converted to
        azimuth and altitude at the reference time, the radiant was 52 arcsec off.
    """

    assert _solveStraightMeteor(True) < 0.05



def testTheLegacyConversionAtTheReferenceTimeIsKept():
    """ Check that the legacy and incorrect conversion at the reference time, only for reproducing older
        solutions, still rotates the lines of sight.
    """

    assert _solveStraightMeteor(False) > 30.0




if __name__ == "__main__":

    # Run all tests
    test_list = [testSolverStationIDs, testRowsWithoutTheMainFragmentAreSkipped, \
        testFilesWithoutEmptyRowsAreUnchanged, testTheRaDecOfEachRowAreConvertedAtItsTime, \
        testTheLegacyConversionAtTheReferenceTimeIsKept]

    for test in test_list:
        test()
        print("PASS {:s}".format(test.__name__))
