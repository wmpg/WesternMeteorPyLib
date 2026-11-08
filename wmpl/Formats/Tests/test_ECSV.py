""" Tests of reading ECSV files with loadECSVs(). The GFE standard (Appendix 4) keeps one row per frame, with
    additional fragments in columns with a numeric suffix (e.g. azimuth1), so the main fragment columns are
    empty on the frames on which only an additional fragment was measured. The azimuth and altitude of each
    row are converted to R.A./Dec. at the time of the row, which the solver has to convert back at that time.

Run with pytest:
    python -m pytest wmpl/Formats/Tests/test_ECSV.py -v
"""

from __future__ import print_function, division, absolute_import

import datetime
import os
import tempfile

import numpy as np

import matplotlib
matplotlib.use('Agg')

from wmpl.Formats.ECSV import loadECSVs
from wmpl.Formats.GenericFunctions import solveTrajectoryGeneric
from wmpl.Trajectory.Tests.SyntheticTrajectory import STRAIGHT_JDT_REF, STRAIGHT_STATIONS, \
    stationAzimAlt, straightMeteorPoints
from wmpl.Utils.Math import angleBetweenVectors
from wmpl.Utils.TrajConversions import datetime2JD, jd2Date


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
    test_list = [testRowsWithoutTheMainFragmentAreSkipped, testFilesWithoutEmptyRowsAreUnchanged, \
        testTheRaDecOfEachRowAreConvertedAtItsTime, testTheLegacyConversionAtTheReferenceTimeIsKept]

    for test in test_list:
        test()
        print("PASS {:s}".format(test.__name__))
