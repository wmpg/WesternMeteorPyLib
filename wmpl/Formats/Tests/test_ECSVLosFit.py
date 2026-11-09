""" Test of the --losfit option of the ECSV command line, which picks the cost of the line-of-sight fit of the
    trajectory solver (los_fit of wmpl.Trajectory.Trajectory).

Run with pytest:
    python -m pytest wmpl/Formats/Tests/test_ECSVLosFit.py -v
"""

from __future__ import print_function, division, absolute_import

import datetime
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np

from wmpl.Trajectory.Tests.SyntheticTrajectory import STRAIGHT_JDT_REF, STRAIGHT_STATIONS, stationAzimAlt, \
    straightMeteorPoints
from wmpl.Utils.TrajConversions import datetime2JD, jd2Date


# Header of the files, with the station
HEADER = """# %ECSV 0.9
# ---
# delimiter: ','
# meta: !!omap
# - {{obs_latitude: {:.6f}}}
# - {{obs_longitude: {:.6f}}}
# - {{obs_elevation: {:.1f}}}
# - {{camera_id: '{:s}'}}
"""
COLUMNS = ['datetime', 'ra', 'dec', 'azimuth', 'altitude', 'x_image', 'y_image', 'mag_data']



def _writeECSVs(dir_path):
    """ Write the noise-free straight meteor of SyntheticTrajectory as seen from each of its stations into an
        ECSV file per station, with the azimuth and altitude at the time of each row.

    Arguments:
        dir_path: [str] Directory where the files are written.

    Return:
        [list] Paths to the files.
    """

    times, points, _, _ = straightMeteorPoints()

    # The times of the rows to the microsecond, as they are written
    dt_ref = jd2Date(STRAIGHT_JDT_REF, dt_obj=True)
    dt_list = [dt_ref + datetime.timedelta(microseconds=int(round(1e6*t))) for t in times]
    jds = np.array([datetime2JD(dt) for dt in dt_list])

    paths = []
    for station_id, s_lat, s_lon, ele in STRAIGHT_STATIONS:

        azim, alt = stationAzimAlt(points, jds, s_lat, s_lon, ele)

        rows = []
        for i, dt in enumerate(dt_list):
            rows.append(",".join([dt.strftime("%Y-%m-%dT%H:%M:%S.%f"), "0.0", "0.0", \
                "{:.9f}".format(np.degrees(azim[i])), "{:+.9f}".format(np.degrees(alt[i])), "0.0", "0.0", \
                "+0.00"]))

        path = os.path.join(dir_path, "{:s}.ecsv".format(station_id))
        with open(path, "w") as f:
            f.write(HEADER.format(s_lat, s_lon, ele, station_id) + ",".join(COLUMNS) + "\n" \
                + "\n".join(rows) + "\n")

        paths.append(path)

    return paths



def _solveReport(los_fit):
    """ Solve the ECSV files of the straight meteor with the ECSV command line, without Monte Carlo runs, and
        read the report.

    Arguments:
        los_fit: [str] Value of --losfit, or None to leave it out.

    Return:
        [str] The report of the solution.
    """

    with tempfile.TemporaryDirectory() as tmp_dir:

        paths = _writeECSVs(tmp_dir)

        command = [sys.executable, "-m", "wmpl.Formats.ECSV"] + paths + ["-d", "-x"]
        if los_fit is not None:
            command += ["--losfit", los_fit]

        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        report_paths = glob.glob(os.path.join(tmp_dir, "*", "*_report.txt"))
        assert len(report_paths) == 1

        with open(report_paths[0]) as f:
            return f.read()



def testLosFitOptionPicksTheFit():
    """ Check that --losfit huber solves the line-of-sight fit with the Huber loss, which the report names
        with its formal uncertainty, and that the fit of the angles stays the default.
    """

    assert "Formal 1 sigma of the line-of-sight fit (huber)" in _solveReport("huber")
    assert "Formal 1 sigma of the line-of-sight fit" not in _solveReport(None)
