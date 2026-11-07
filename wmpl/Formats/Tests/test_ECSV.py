""" Tests of reading ECSV files with loadECSVs(). The GFE standard (Appendix 4) keeps one row per frame, with
    additional fragments in columns with a numeric suffix (e.g. azimuth1), so the main fragment columns are
    empty on the frames on which only an additional fragment was measured.

Run with pytest:
    python -m pytest wmpl/Formats/Tests/test_ECSV.py -v
"""

from __future__ import print_function, division, absolute_import

import os
import tempfile

import numpy as np

from wmpl.Formats.ECSV import loadECSVs


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




if __name__ == "__main__":

    # Run all tests
    test_list = [testRowsWithoutTheMainFragmentAreSkipped, testFilesWithoutEmptyRowsAreUnchanged]

    for test in test_list:
        test()
        print("PASS {:s}".format(test.__name__))
