""" Tests of solving UWO-style ev files with solveTrajectoryEv(). The azimuth and altitude of each point are
    converted to R.A./Dec. at the time of the point, and to J2000, which prepareObservations takes.

Run with pytest:
    python -m pytest wmpl/Formats/Tests/test_EvUWO.py -v
"""

from __future__ import print_function, division, absolute_import

import os
import tempfile

import numpy as np

import matplotlib
matplotlib.use('Agg')

from wmpl.Formats.EvUWO import solveTrajectoryEv, writeEvFile
from wmpl.Trajectory.Tests.SyntheticTrajectory import STRAIGHT_JDT_REF, STRAIGHT_STATIONS, \
    stationAzimAlt, straightMeteorPoints, worstResidual
from wmpl.Utils.Math import angleBetweenVectors



def _writeStraightMeteorEvFiles(dir_path):
    """ Write the noise-free straight meteor of SyntheticTrajectory as seen from each of its stations into an
        ev file per station.

    Arguments:
        dir_path: [str] Directory where the files are written.

    Return:
        (paths, radiant_eci):
            - paths: [list] Paths to the files.
            - radiant_eci: [ndarray] True radiant unit vector in ECI coordinates.
    """

    times, points, _, radiant_eci = straightMeteorPoints()

    # Compute the directions at the times as they are written, to the millisecond
    times = np.round(times, 3)

    paths = []
    for station_id, s_lat, s_lon, ele in STRAIGHT_STATIONS:

        azim, alt = stationAzimAlt(points, STRAIGHT_JDT_REF + times/86400.0, s_lat, s_lon, ele)

        # Convert to the zenith angle and the azimuth +N of due E of the ev files
        theta_data = np.pi/2.0 - alt
        phi_data = (np.pi/2.0 - azim)%(2*np.pi)

        file_name = "ev_{:s}.txt".format(station_id)
        writeEvFile(dir_path, file_name, STRAIGHT_JDT_REF, station_id, np.radians(s_lat), \
            np.radians(s_lon), ele, times, theta_data, phi_data)

        paths.append(os.path.join(dir_path, file_name))

    return paths, radiant_eci



def testTheStraightMeteorIsRecoveredFromEvFiles():
    """ Check that the straight meteor is recovered from its ev files. The R.A./Dec. of date used to be
        precessed again as if they were J2000, which moved the lines of sight by 1060 arcsec, put the radiant
        396 arcsec off and left 810 m of residual.
    """

    with tempfile.TemporaryDirectory() as tmp_dir:

        paths, radiant_eci = _writeStraightMeteorEvFiles(tmp_dir)

        traj = solveTrajectoryEv(paths, estimate_timing_vel=False, gravity_correction=False, \
            filter_picks=False, monte_carlo=False, calc_orbit=False, show_plots=False, save_results=False, \
            verbose=False)

    radiant_error = 3600*np.degrees(angleBetweenVectors(traj.radiant_eci_mini, radiant_eci))

    # The angles in the ev files are rounded to 1e-5 deg, 0.04 arcsec
    assert radiant_error < 0.2
    assert worstResidual(traj) < 0.5




if __name__ == "__main__":

    testTheStraightMeteorIsRecoveredFromEvFiles()
    print("PASS testTheStraightMeteorIsRecoveredFromEvFiles")
