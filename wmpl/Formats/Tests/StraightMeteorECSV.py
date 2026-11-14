""" The straight meteor of the trajectory tests written into ECSV files, shared by the ECSV tests. """

from __future__ import print_function, division, absolute_import

import datetime
import os

import numpy as np

from wmpl.Trajectory.Tests.SyntheticTrajectory import STRAIGHT_DURATION, STRAIGHT_JDT_REF, STRAIGHT_STATIONS, \
    stationAzimAlt, straightMeteorPoints
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



def writeStraightMeteorECSVs(dir_path, duration=STRAIGHT_DURATION):
    """ Write the noise-free straight meteor of SyntheticTrajectory as seen from each of its stations into an
        ECSV file per station, with the azimuth and altitude at the time of each row.

    Arguments:
        dir_path: [str] Directory where the files are written.

    Keyword arguments:
        duration: [float] Duration of the meteor (s). STRAIGHT_DURATION by default.

    Return:
        (paths, azim_alt):
            - paths: [list] Paths to the files.
            - azim_alt: [dict] Azimuths and altitudes of the rows of each station (rad).
    """

    times, points, _, _ = straightMeteorPoints(duration=duration)

    # Compute the times of the rows to the microsecond, as they are written, and their Julian dates
    dt_ref = jd2Date(STRAIGHT_JDT_REF, dt_obj=True)
    dt_list = [dt_ref + datetime.timedelta(microseconds=int(round(1e6*t))) for t in times]
    jds = np.array([datetime2JD(dt) for dt in dt_list])

    paths = []
    azim_alt = {}
    for station_id, s_lat, s_lon, ele in STRAIGHT_STATIONS:

        # Compute the direction of every row from the station
        azim, alt = stationAzimAlt(points, jds, s_lat, s_lon, ele)
        azim_alt[station_id] = (azim, alt)

        # Write a row per point, with the azimuth and altitude in degrees
        rows = []
        for i, dt in enumerate(dt_list):
            rows.append(",".join([dt.strftime("%Y-%m-%dT%H:%M:%S.%f"), "0.0", "0.0", \
                "{:.9f}".format(np.degrees(azim[i])), "{:+.9f}".format(np.degrees(alt[i])), "0.0", "0.0", \
                "+0.00"]))

        path = os.path.join(dir_path, "{:s}.ecsv".format(station_id))
        with open(path, 'w') as f:
            f.write(HEADER.format(s_lat, s_lon, ele, station_id) + ",".join(COLUMNS) + "\n" \
                + "\n".join(rows) + "\n")

        paths.append(path)

    return paths, azim_alt
