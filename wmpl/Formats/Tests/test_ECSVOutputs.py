""" Tests that the MILIG input file and the Gural solver get the direction of every ECSV row at its time, and
    that the ECSV files written by saveECSV are read back.

Run with pytest:
    python -m pytest wmpl/Formats/Tests/test_ECSVOutputs.py -v
"""

from __future__ import print_function, division, absolute_import

import os
import sys
import tempfile

import numpy as np

import matplotlib
matplotlib.use('Agg')

from wmpl.Formats.ECSV import loadECSVs, saveECSV
from wmpl.Formats.GenericFunctions import solveTrajectoryGeneric, writeMiligInputFileMeteorObservation
from wmpl.Formats.Tests.StraightMeteorECSV import COLUMNS, writeStraightMeteorECSVs
from wmpl.Trajectory.Tests.SyntheticTrajectory import STRAIGHT_STATIONS
from wmpl.Utils.Earth import calcEquationOfEquinoxes
from wmpl.Utils.Math import angleBetweenSphericalCoords
from wmpl.Utils.TrajConversions import raDec2AltAz_vect


# Duration of the meteor (s), in which the Earth turns by 37 arcsec
DURATION = 2.5



def angleArcsec(azim1, alt1, azim2, alt2):
    """ Compute the angles between two sets of directions.

    Arguments:
        azim1: [ndarray] Azimuths of the first directions (rad).
        alt1: [ndarray] Altitudes of the first directions (rad).
        azim2: [ndarray] Azimuths of the second directions (rad).
        alt2: [ndarray] Altitudes of the second directions (rad).

    Return:
        [ndarray] Angles (arcsec).
    """

    return 3600*np.degrees(angleBetweenSphericalCoords(alt1, azim1, alt2, azim2))



def testPreparedAzimuthsAreThoseOfTheFile():
    """ Check that the prepared observations keep the azimuth and altitude of every row, rather than the
        direction of its R.A./Dec. at the reference time, which is turned by the rotation of the Earth.
    """

    with tempfile.TemporaryDirectory() as tmp_dir:
        paths, azim_alt = writeStraightMeteorECSVs(tmp_dir, duration=DURATION)
        _, meteor_list = loadECSVs(paths)

    for meteor in meteor_list:
        azim, alt = azim_alt[meteor.station_id]
        assert np.max(angleArcsec(meteor.azim_data, meteor.elev_data, azim, alt)) < 0.01



def testMiligFileHasTheAzimuthsOfTheRows():
    """ Check that the MILIG input file has the azimuth and zenith angle of every row of the ECSV files. """

    with tempfile.TemporaryDirectory() as tmp_dir:

        paths, azim_alt = writeStraightMeteorECSVs(tmp_dir, duration=DURATION)
        jdt_ref, meteor_list = loadECSVs(paths)

        milig_path = os.path.join(tmp_dir, "milig.txt")
        writeMiligInputFileMeteorObservation(jdt_ref, meteor_list, milig_path)

        with open(milig_path) as f:
            lines = f.readlines()

    # Read the points of each station, the last one flagged with 9
    stations = []
    new_station = True
    for line in lines[1:]:

        # Stop at the end flag
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

        # Convert the azimuth west of due south and the zenith angle
        milig_azim = (np.radians(station_points[:, 0]) + np.pi)%(2*np.pi)
        milig_alt = np.pi/2 - np.radians(station_points[:, 1])

        # The file has 5 decimals of a degree
        assert np.max(angleArcsec(milig_azim, milig_alt, azim, alt)) < 0.1



class GuralStub(object):
    """ Stand-in for the Gural solver which records the measurements instead of solving the trajectory. """

    def __init__(self, *args, **kwargs):
        """ Make an empty record. """

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

    # Replace the Gural solver by the stub
    generic = sys.modules['wmpl.Formats.GenericFunctions']
    original = generic.GuralTrajectory
    generic.GuralTrajectory = GuralStub

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths, azim_alt = writeStraightMeteorECSVs(tmp_dir, duration=DURATION)
            jdt_ref, meteor_list = loadECSVs(paths)
            stub = solveTrajectoryGeneric(jdt_ref, meteor_list, tmp_dir, solver='gural')

    finally:
        generic.GuralTrajectory = original

    for (ra, dec, lat, lon), (station_id, _, _, _) in zip(stub.infilled, STRAIGHT_STATIONS):

        # The Gural solver gets the R.A. with the mean sidereal time
        ra = ra + calcEquationOfEquinoxes(jdt_ref)

        azim, alt = raDec2AltAz_vect(ra, dec, jdt_ref, lat, lon)
        assert np.max(angleArcsec(azim, alt, *azim_alt[station_id])) < 0.01



def testSavedECSVIsReadBack():
    """ Check that the ECSV files written by saveECSV declare the columns they have, so that astropy can read
        them, and that loadECSVs reads back the same directions.
    """

    from astropy.table import Table

    with tempfile.TemporaryDirectory() as tmp_dir:

        paths, azim_alt = writeStraightMeteorECSVs(tmp_dir, duration=DURATION)
        _, meteor_list = loadECSVs(paths)

        # Save the loaded meteors into new files
        out_dir = os.path.join(tmp_dir, "saved")
        os.makedirs(out_dir)
        saveECSV(out_dir, meteor_list)

        saved_paths = sorted(os.path.join(out_dir, name) for name in os.listdir(out_dir))
        assert len(saved_paths) == len(STRAIGHT_STATIONS)

        # Check the declared columns with astropy
        for path in saved_paths:
            table = Table.read(path, format='ascii.ecsv')
            assert table.colnames == COLUMNS

        _, saved_meteors = loadECSVs(saved_paths)

    for meteor in saved_meteors:
        azim, alt = azim_alt[meteor.station_id]

        # The files have 6 decimals of a degree
        assert np.max(angleArcsec(meteor.azim_data, meteor.elev_data, azim, alt)) < 0.01



if __name__ == "__main__":

    testPreparedAzimuthsAreThoseOfTheFile()
    testMiligFileHasTheAzimuthsOfTheRows()
    testGuralSolverGetsTheRaDecAtTheReferenceTime()
    testSavedECSVIsReadBack()

    print("All tests passed.")
