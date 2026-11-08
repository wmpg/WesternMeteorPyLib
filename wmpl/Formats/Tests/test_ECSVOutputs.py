""" Tests of what is made from ECSV files besides the trajectory: the azimuths and altitudes of the prepared
    observations, which go into the MILIG input file, the R.A./Dec. given to the Gural solver, and the ECSV
    files written by saveECSV. The R.A./Dec. of ECSV files are computed at the time of each row, so they have
    to be converted back at that time.

Run with pytest:
    python -m pytest wmpl/Formats/Tests/test_ECSVOutputs.py -v
"""

from __future__ import print_function, division, absolute_import

import datetime
import os
import sys
import tempfile

import numpy as np

import matplotlib
matplotlib.use('Agg')

from wmpl.Formats.ECSV import loadECSVs, saveECSV
from wmpl.Formats.GenericFunctions import solveTrajectoryGeneric, writeMiligInputFileMeteorObservation
from wmpl.Trajectory.Tests.SyntheticTrajectory import STRAIGHT_JDT_REF, STRAIGHT_STATIONS, \
    stationAzimAlt, straightMeteorPoints
from wmpl.Utils.Earth import calcEquationOfEquinoxes
from wmpl.Utils.Math import angleBetweenSphericalCoords
from wmpl.Utils.TrajConversions import datetime2JD, jd2Date, raDec2AltAz_vect


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

# Duration of the meteor (s), in which the Earth turns by 37 arcsec
DURATION = 2.5



def _writeECSVs(dir_path):
    """ Write the noise-free straight meteor of SyntheticTrajectory, lasting DURATION, as seen from each of
        its stations into an ECSV file per station, with the azimuth and altitude at the time of each row.

    Arguments:
        dir_path: [str] Directory where the files are written.

    Return:
        (paths, azim_alt):
            - paths: [list] Paths to the files.
            - azim_alt: [dict] Azimuths and altitudes of the rows of each station (rad).
    """

    times, points, _, _ = straightMeteorPoints(duration=DURATION)

    # Compute the times of the rows to the microsecond, as they are written, and their Julian dates
    dt_ref = jd2Date(STRAIGHT_JDT_REF, dt_obj=True)
    dt_list = [dt_ref + datetime.timedelta(microseconds=int(round(1e6*t))) for t in times]
    jds = np.array([datetime2JD(dt) for dt in dt_list])

    paths = []
    azim_alt = {}
    for station_id, s_lat, s_lon, ele in STRAIGHT_STATIONS:

        azim, alt = stationAzimAlt(points, jds, s_lat, s_lon, ele)
        azim_alt[station_id] = (azim, alt)

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

    return paths, azim_alt



def _angleArcsec(azim1, alt1, azim2, alt2):
    """ Compute the angles between two sets of directions.

    Arguments:
        azim1, alt1: [ndarray] First directions (rad).
        azim2, alt2: [ndarray] Second directions (rad).

    Return:
        [ndarray] Angles (arcsec).
    """

    return 3600*np.degrees(angleBetweenSphericalCoords(alt1, azim1, alt2, azim2))



def testPreparedAzimuthsAreThoseOfTheFile():
    """ Check that the prepared observations keep the azimuth and altitude of every row, rather than the
        direction of its R.A./Dec. at the reference time, which is turned by the rotation of the Earth.
    """

    with tempfile.TemporaryDirectory() as tmp_dir:
        paths, azim_alt = _writeECSVs(tmp_dir)
        _, meteor_list = loadECSVs(paths)

    for meteor in meteor_list:
        azim, alt = azim_alt[meteor.station_id]
        assert np.max(_angleArcsec(meteor.azim_data, meteor.elev_data, azim, alt)) < 0.01



def testMiligFileHasTheAzimuthsOfTheRows():
    """ Check that the MILIG input file has the azimuth and zenith angle of every row of the ECSV files. """

    with tempfile.TemporaryDirectory() as tmp_dir:

        paths, azim_alt = _writeECSVs(tmp_dir)
        jdt_ref, meteor_list = loadECSVs(paths)

        milig_path = os.path.join(tmp_dir, "milig.txt")
        writeMiligInputFileMeteorObservation(jdt_ref, meteor_list, milig_path)

        with open(milig_path) as f:
            lines = f.readlines()

    # Read the points, which follow the line of each station, the last one flagged with 9, until the end flag
    stations = []
    new_station = True
    for line in lines[1:]:

        if line.startswith('-1'):
            break

        if new_station:
            stations.append([])
            new_station = False
            continue

        stations[-1].append((float(line[0:9]), float(line[9:17])))
        if int(line[17:20]) == 9:
            new_station = True

    assert len(stations) == len(STRAIGHT_STATIONS)
    for (station_id, _, _, _), station_points in zip(STRAIGHT_STATIONS, stations):

        azim, alt = azim_alt[station_id]
        station_points = np.array(station_points)

        # The file has the azimuth west of due south and the zenith angle, in degrees
        milig_azim = (np.radians(station_points[:, 0]) + np.pi)%(2*np.pi)
        milig_alt = np.pi/2 - np.radians(station_points[:, 1])

        # The file has 5 decimals of a degree
        assert np.max(_angleArcsec(milig_azim, milig_alt, azim, alt)) < 0.1



class _GuralStub(object):
    def __init__(self, *args, **kwargs):
        """ Record the measurements given to the Gural solver instead of solving the trajectory. """

        self.infilled = []



    def infillTrajectory(self, ra, dec, time_data, lat, lon, ele):
        """ Record the measurements of a station. """

        self.infilled.append((np.array(ra), np.array(dec), lat, lon))



    def run(self):
        """ Return the stub. """

        return self



def testGuralSolverGetsTheRaDecAtTheReferenceTime():
    """ Check that the Gural solver, which converts all R.A./Dec. at the reference time, gets R.A./Dec. which
        point at the azimuth and altitude of every row at that time.
    """

    generic = sys.modules['wmpl.Formats.GenericFunctions']
    original = generic.GuralTrajectory
    generic.GuralTrajectory = _GuralStub

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths, azim_alt = _writeECSVs(tmp_dir)
            jdt_ref, meteor_list = loadECSVs(paths)
            stub = solveTrajectoryGeneric(jdt_ref, meteor_list, tmp_dir, solver='gural')

    finally:
        generic.GuralTrajectory = original

    for (ra, dec, lat, lon), (station_id, _, _, _) in zip(stub.infilled, STRAIGHT_STATIONS):

        # The Gural solver gets the R.A. with the mean sidereal time
        ra = ra + calcEquationOfEquinoxes(jdt_ref)

        azim, alt = raDec2AltAz_vect(ra, dec, jdt_ref, lat, lon)
        assert np.max(_angleArcsec(azim, alt, *azim_alt[station_id])) < 0.01



def testSavedECSVIsReadBack():
    """ Check that the ECSV files written by saveECSV declare the columns they have, so that astropy can read
        them, and that loadECSVs reads back the same directions.
    """

    from astropy.table import Table

    with tempfile.TemporaryDirectory() as tmp_dir:

        paths, azim_alt = _writeECSVs(tmp_dir)
        _, meteor_list = loadECSVs(paths)

        out_dir = os.path.join(tmp_dir, "saved")
        os.makedirs(out_dir)
        saveECSV(out_dir, meteor_list)

        saved_paths = sorted(os.path.join(out_dir, name) for name in os.listdir(out_dir))
        assert len(saved_paths) == len(STRAIGHT_STATIONS)

        for path in saved_paths:
            table = Table.read(path, format='ascii.ecsv')
            assert table.colnames == COLUMNS

        _, saved_meteors = loadECSVs(saved_paths)

    for meteor in saved_meteors:
        azim, alt = azim_alt[meteor.station_id]

        # The files have 6 decimals of a degree
        assert np.max(_angleArcsec(meteor.azim_data, meteor.elev_data, azim, alt)) < 0.01
