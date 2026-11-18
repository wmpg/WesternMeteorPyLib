""" Tests the 3D KML of a solved trajectory (TrajectoryKML.trajectoryToKML()).

The tests solve the straight meteor of SyntheticTrajectory, observed from three stations, and check the
folders, placemarks, coordinates and speeds of its KML.

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

import wmpl.Utils.TrajectoryKML as TrajectoryKML
from wmpl.Trajectory.Tests.SyntheticTrajectory import observeStraightMeteor
from wmpl.Utils.DragFit import DragFit
from wmpl.Utils.TrajConversions import geo2Cartesian


# KML namespace
KML_NS = '{http://www.opengis.net/kml/2.2}'

# Precision of the KML coordinates, longitude and latitude (deg) and height (m)
COORD_TOL = np.array([1e-6, 1e-6, 1e-3])



def _solve(noise=0.0):
    """ Solve the straight meteor.

    Keyword arguments:
        noise: [float] Noise of the azimuth and elevation (radians). 0 by default.

    Return:
        traj: [Trajectory object] Solved trajectory.
    """

    traj, _, _ = observeStraightMeteor(noise=noise, filter_picks=False, calc_orbit=False)

    return traj.run()



def _parseKML(traj, measured=True):
    """ Save the KML of a trajectory and collect its folders.

    Arguments:
        traj: [Trajectory object] Solved trajectory.

    Keyword arguments:
        measured: [bool] Add the measured layers. True by default.

    Return:
        folders: [dict] Top folders of the KML by name, each a dict of its subfolders by name, with the
            folder itself under None.
    """

    with tempfile.TemporaryDirectory() as tmp_dir:

        kml_path = TrajectoryKML.trajectoryToKML(traj, os.path.join(tmp_dir, 'test_3D.kml'), \
            measured=measured)
        root = ET.parse(kml_path).getroot()

    folders = {}
    for folder in root.find(KML_NS + 'Document').findall(KML_NS + 'Folder'):
        name = folder.find(KML_NS + 'name').text
        folders[name] = {sub.find(KML_NS + 'name').text: sub for sub in folder.findall(KML_NS + 'Folder')}
        folders[name][None] = folder

    return folders



def _coordinates(placemark):
    """ Read the coordinates of a placemark.

    Arguments:
        placemark: [Element] KML placemark.

    Return:
        [ndarray] Longitude (deg), latitude (deg) and height (m) of each point.
    """

    points = []
    for coordinates in placemark.findall('.//' + KML_NS + 'coordinates'):
        points += [[float(x) for x in point.split(',')] for point in coordinates.text.split()]

    return np.array(points)



def testTheDefaultKMLShowsTheTrajectoryOfTheDragFit():
    """ Check that the default KML has the trajectory of the drag fit with 500 samples over the time of the
        picks, and their speeds, without the measured layers.
    """

    traj = _solve()
    folders = _parseKML(traj, measured=False)

    assert sorted(folders) == ["1. Vertical Plane", "2. Drag Fit Trajectory", "3. Stations"]

    samples = folders["2. Drag Fit Trajectory"]["Samples"].findall(KML_NS + 'Placemark')
    line = folders["2. Drag Fit Trajectory"][None].findall(KML_NS + 'Placemark')

    assert len(samples) == 500
    assert len(line) == 1

    # Check that the line starts and ends at the heights of the first and last picks
    line_coords = _coordinates(line[0])
    heights = np.concatenate([obs.model_ht for obs in traj.observations])

    assert abs(line_coords[0, 2] - np.max(heights)) < 100.0
    assert abs(line_coords[-1, 2] - np.min(heights)) < 100.0

    # Check the speed of the samples of the meteor at a constant 30 km/s
    for placemark in samples:
        description = placemark.find(KML_NS + 'description').text
        vel = float(description.split("<br>v = ")[1].split(" km/s")[0])

        assert abs(vel - traj.v_init/1000) < 0.05



def testTheKMLHasAPlacemarkForEveryUsedFrame():
    """ Check that the KML with the measurements has the six folders, and per station a projected point, a
        line of sight and a residual for every frame that is not ignored, and a station placemark.
    """

    traj = _solve()
    traj.observations[0].ignore_list[:3] = 1
    folders = _parseKML(traj)

    assert sorted(folders) == ["1. Vertical Plane", "2. Drag Fit Trajectory", "3. Projected Points", \
        "4. Lines of Sight", "5. Residuals", "6. Stations"]

    for obs in traj.observations:

        n_used = int(np.sum(obs.ignore_list == 0))

        for name in ("3. Projected Points", "4. Lines of Sight", "5. Residuals"):
            assert len(folders[name][obs.station_id].findall(KML_NS + 'Placemark')) == n_used, name

    assert len(folders["6. Stations"][None].findall(KML_NS + 'Placemark')) == len(traj.observations)
    assert len(folders["1. Vertical Plane"][None].findall(KML_NS + 'Placemark')) == 1



def testThePointsAndResidualsAreWhereTheSolverPutThem():
    """ Check on a noisy meteor that the projected points are at the solved positions as longitude,
        latitude and height, and that each residual joins the end of the line of sight to the projected
        point with the distance between them.
    """

    traj = _solve(noise=1e-4)
    folders = _parseKML(traj)

    for obs in traj.observations:

        used = np.where(obs.ignore_list == 0)[0]
        points = folders["3. Projected Points"][obs.station_id].findall(KML_NS + 'Placemark')
        residuals = folders["5. Residuals"][obs.station_id].findall(KML_NS + 'Placemark')

        for k, point, residual in zip(used, points, residuals):

            model = np.array([np.degrees(obs.model_lon[k]), np.degrees(obs.model_lat[k]), obs.model_ht[k]])
            meas = np.array([np.degrees(obs.meas_lon[k]), np.degrees(obs.meas_lat[k]), obs.meas_ht[k]])

            assert np.allclose(_coordinates(point)[0], model, rtol=0, atol=COORD_TOL)

            # Check that the dashed residual runs from the end of the line of sight towards the projected
            #   point, its last dash ending one dash short of it
            res_coords = _coordinates(residual)
            dash_pts = np.linspace(meas, model, 2*TrajectoryKML.N_DASHES + 1)

            assert np.allclose(res_coords[0], meas, rtol=0, atol=COORD_TOL)
            assert np.allclose(res_coords[-1], dash_pts[-2], rtol=0, atol=COORD_TOL)

            # Check the residual distance in the name
            model_eci = np.array(geo2Cartesian(obs.model_lat[k], obs.model_lon[k], obs.model_ht[k], \
                traj.jdt_ref))
            meas_eci = np.array(geo2Cartesian(obs.meas_lat[k], obs.meas_lon[k], obs.meas_ht[k], traj.jdt_ref))
            distance = np.linalg.norm(model_eci - meas_eci)
            name = residual.find(KML_NS + 'name').text

            assert distance > 0
            assert float(name.split(': ')[-1].split(' m')[0]) == pytest.approx(distance, abs=0.06)



def testThePointsGiveTheSpeedOfTheDragFitAtTheirLengths(monkeypatch):
    """ Check that the projected points give the speed of the drag fit at their lengths along the
        trajectory, not at their heights, which on real data differ from the heights of the drag fit, and
        the point-to-point speed except for the first point of a station.
    """

    traj = _solve(noise=1e-4)

    # Make the speeds at heights unavailable
    def failingVelocityAtHeight(self, ht_arr):
        raise AssertionError("velocityAtHeight was used")

    monkeypatch.setattr(DragFit, 'velocityAtHeight', failingVelocityAtHeight)

    folders = _parseKML(traj)
    drag_fit = DragFit(traj)

    for obs in traj.observations:

        used = np.where(obs.ignore_list == 0)[0]
        points = folders["3. Projected Points"][obs.station_id].findall(KML_NS + 'Placemark')
        vel_drag = drag_fit.velocityAtLength(np.array(obs.state_vect_dist)[used])

        for k, point, vel in zip(used, points, vel_drag):

            description = point.find(KML_NS + 'description').text
            vel_kml = float(description.split("v = ")[1].split(" km/s (drag fit)")[0])

            assert vel_kml == pytest.approx(vel/1000, abs=1e-3)

            # Check the point-to-point speed, which the solver sets to 0 for the first point
            if k == 0:
                assert "v point" not in description

            else:
                vel_point = float(description.split("v point = ")[1].split(" km/s")[0])
                assert vel_point == pytest.approx(obs.velocities[k]/1000, abs=1e-3)



def testIgnoredStationsAreSkipped():
    """ Check that an ignored station has no measured layers but keeps its station placemark.
    """

    traj = _solve()
    traj.observations[1].ignore_station = True
    folders = _parseKML(traj)

    station_id = traj.observations[1].station_id

    for name in ("3. Projected Points", "4. Lines of Sight", "5. Residuals"):
        assert station_id not in folders[name]

    assert len(folders["6. Stations"][None].findall(KML_NS + 'Placemark')) == len(traj.observations)



def testWithoutADragFitTheMeasuredLayersAreSaved(monkeypatch):
    """ Check that the measured layers are saved when the drag model cannot be fitted.
    """

    traj = _solve()

    def failingSample(*args, **kwargs):
        raise ValueError("no fit")

    monkeypatch.setattr(TrajectoryKML, 'sampleTrajectory', failingSample)
    folders = _parseKML(traj, measured=False)

    assert sorted(folders) == ["1. Vertical Plane", "2. Projected Points", "3. Lines of Sight", \
        "4. Residuals", "5. Stations"]



def testTheKMLColor():
    """ Check that the colour is written as alpha, blue, green and red. """

    assert TrajectoryKML.kmlColor((1.0, 0.5, 0.0)) == 'ff0080ff'
    assert TrajectoryKML.kmlColor((0.0, 0.0, 1.0), alpha=130) == '82ff0000'



def testTheFragmentsAreDrawnInTheirColoursInSkyFit2():
    """ Check that every additional fragment is a line from its first to its last point, in the colour of
        its number in SkyFit2, which repeat after fragment 8.
    """

    traj = _solve()

    with tempfile.TemporaryDirectory() as tmp_dir:
        kml_path = TrajectoryKML.trajectoryToKML(traj, os.path.join(tmp_dir, 'test_3D.kml'), \
            fragments=[(2, traj), (10, traj)])
        root = ET.parse(kml_path).getroot()

    folders = {folder.find(KML_NS + 'name').text: folder for folder in root.iter(KML_NS + 'Folder')}
    styles = {style.get('id'): style for style in root.iter(KML_NS + 'Style')}
    placemarks = folders["3. Fragments"].findall(KML_NS + 'Placemark')

    assert [placemark.find(KML_NS + 'name').text for placemark in placemarks] == ["Fragment 2", "Fragment 10"]

    # Check the green of fragment 2, and the cyan of fragment 10 as for fragment 1
    colors = [styles[placemark.find(KML_NS + 'styleUrl').text[1:]].find('.//' + KML_NS + 'color').text \
        for placemark in placemarks]

    assert colors == ['ff00ff00', 'ffffff00']

    # Check the first and the last point
    ends = np.array([[np.degrees(traj.rbeg_lon), np.degrees(traj.rbeg_lat), traj.rbeg_ele], \
        [np.degrees(traj.rend_lon), np.degrees(traj.rend_lat), traj.rend_ele]])

    assert np.allclose(_coordinates(placemarks[0]), ends, rtol=0, atol=COORD_TOL)



def testTheFlaresAreStarsInTheColourOfTheirStation():
    """ Check that every flare point is a star at its position and height, in the colour of its station in
        the KML.
    """

    traj = _solve()
    obs = traj.observations[1]
    flare = {'station_id': str(obs.station_id), 't_rel': obs.time_data[5], 'used': True, \
        'lat': obs.model_lat[5], 'lon': obs.model_lon[5], 'ht': obs.model_ht[5]}

    with tempfile.TemporaryDirectory() as tmp_dir:
        kml_path = TrajectoryKML.trajectoryToKML(traj, os.path.join(tmp_dir, 'test_3D.kml'), flares=[flare])
        root = ET.parse(kml_path).getroot()

    folders = {folder.find(KML_NS + 'name').text: folder for folder in root.iter(KML_NS + 'Folder')}
    styles = {style.get('id'): style for style in root.iter(KML_NS + 'Style')}
    placemarks = folders["3. Flares"].findall(KML_NS + 'Placemark')

    assert len(placemarks) == 1

    # Check the star in the orange of the second station
    href = styles[placemarks[0].find(KML_NS + 'styleUrl').text[1:]].find('.//' + KML_NS + 'href').text

    assert href == TrajectoryKML.WEB_ICON + "color=ff7f0e&id=371&scale=4"

    position = np.array([np.degrees(obs.model_lon[5]), np.degrees(obs.model_lat[5]), obs.model_ht[5]])

    assert np.allclose(_coordinates(placemarks[0])[0], position, rtol=0, atol=COORD_TOL)
