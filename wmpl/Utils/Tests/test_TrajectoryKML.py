""" Tests the 3D KML of a solved trajectory (TrajectoryKML.trajectoryToKML()).

The tests solve the straight meteor of SyntheticTrajectory, observed from three stations, and check the
folders and placemarks of its KML.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_TrajectoryKML.py -v
"""

from __future__ import print_function, division, absolute_import

import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

pytest.importorskip('simplekml')

from wmpl.Trajectory.Tests.SyntheticTrajectory import observeStraightMeteor
from wmpl.Utils.TrajectoryKML import kmlColor, trajectoryToKML


# KML namespace
KML_NS = '{http://www.opengis.net/kml/2.2}'



def _solvedKML(ignore_first=0, measured=True):
    """ Solve the straight meteor and parse its KML.

    Keyword arguments:
        ignore_first: [int] Number of first picks of the first station to ignore. 0 by default.
        measured: [bool] Add the measured layers. True by default.

    Return:
        traj: [Trajectory object] Solved trajectory.
        folders: [dict] Top folders of the KML by name, each a dict of its station folders by name.
    """

    traj, _, _ = observeStraightMeteor(noise=0.0, filter_picks=False, calc_orbit=False)
    traj = traj.run()

    # Ignore the first picks of the first station
    traj.observations[0].ignore_list[:ignore_first] = 1

    with tempfile.TemporaryDirectory() as tmp_dir:

        kml_path = trajectoryToKML(traj, os.path.join(tmp_dir, 'test_3D.kml'), measured=measured)
        root = ET.parse(kml_path).getroot()

    # Collect the top folders and their station folders
    folders = {}
    for folder in root.find(KML_NS + 'Document').findall(KML_NS + 'Folder'):
        name = folder.find(KML_NS + 'name').text
        folders[name] = {sub.find(KML_NS + 'name').text: sub for sub in folder.findall(KML_NS + 'Folder')}
        folders[name][None] = folder

    return traj, folders



def testTheKMLHasAPlacemarkForEveryUsedFrame():
    """ Check that the KML with the measurements has the six folders, and per station a projected point, a
        line of sight and a residual for every frame that is not ignored, and a station placemark.
    """

    traj, folders = _solvedKML(ignore_first=3)

    assert sorted(name for name in folders) == ["1. Vertical Plane", "2. Drag Fit Trajectory", \
        "3. Projected Points", "4. Lines of Sight", "5. Residuals", "6. Stations"]

    for obs in traj.observations:

        n_used = int(np.sum(obs.ignore_list == 0))

        for name in ("3. Projected Points", "4. Lines of Sight", "5. Residuals"):
            assert len(folders[name][obs.station_id].findall(KML_NS + 'Placemark')) == n_used, name

    # Check that the ignored picks are skipped
    obs = traj.observations[0]
    assert int(np.sum(obs.ignore_list == 0)) == len(obs.time_data) - 3

    # Check the stations and the plane
    assert len(folders["6. Stations"][None].findall(KML_NS + 'Placemark')) == len(traj.observations)
    assert len(folders["1. Vertical Plane"][None].findall(KML_NS + 'Placemark')) == 1



def testTheDefaultKMLShowsTheTrajectoryOfTheDragFit():
    """ Check that the default KML has the trajectory of the drag fit, sampled every 100 m in height from the
        beginning to the end, without the measured layers.
    """

    traj, folders = _solvedKML(measured=False)

    assert sorted(name for name in folders) == ["1. Vertical Plane", "2. Drag Fit Trajectory", "3. Stations"]

    # Check the number of samples and the line through them
    n_samples = len(np.arange(traj.rend_ele, traj.rbeg_ele + 100.0, 100.0))
    samples = folders["2. Drag Fit Trajectory"]["Samples"].findall(KML_NS + 'Placemark')
    line = folders["2. Drag Fit Trajectory"][None].findall(KML_NS + 'Placemark')

    assert abs(len(samples) - n_samples) <= 1
    assert len(line) == 1

    # Check the speed of the samples of the meteor at a constant 30 km/s
    for placemark in samples:
        description = placemark.find(KML_NS + 'description').text
        vel = float(description.split("<br>v = ")[1].split(" km/s")[0])

        assert abs(vel - traj.v_init/1000) < 0.05



def testTheResidualsAreTheDistancesToTheProjectedPoints():
    """ Check that the residuals of a noise-free meteor are below 1 m, as its lines of sight meet the
        trajectory.
    """

    traj, folders = _solvedKML()

    for obs in traj.observations:
        for placemark in folders["5. Residuals"][obs.station_id].findall(KML_NS + 'Placemark'):

            # The name ends with the residual in meters
            residual = float(placemark.find(KML_NS + 'name').text.split(': ')[-1].split(' m')[0])

            assert residual < 1.0



def testTheKMLColor():
    """ Check that the colour is written as alpha, blue, green and red.
    """

    assert kmlColor((1.0, 0.5, 0.0)) == 'ff0080ff'
    assert kmlColor((0.0, 0.0, 1.0), alpha=130) == '82ff0000'



def testThePointsGiveTheSpeedOfTheDragFit():
    """ Check that the projected points of the meteor at a constant 30 km/s give the speed of the drag fit
        within 50 m/s, and the point-to-point speed.
    """

    traj, folders = _solvedKML()

    for obs in traj.observations:
        for placemark in folders["3. Projected Points"][obs.station_id].findall(KML_NS + 'Placemark'):

            description = placemark.find(KML_NS + 'description').text

            assert "(drag fit)" in description
            assert "v point = " in description

            vel_drag = float(description.split("v = ")[1].split(" km/s (drag fit)")[0])
            assert abs(vel_drag - traj.v_init/1000) < 0.05
