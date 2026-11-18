""" Tests that the flare stars of the 3D KML of a solved trajectory say whether the flare was used in the
    trajectory, which sets where the star is.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_TrajectoryKMLFlareUsed.py -v
"""

from __future__ import print_function, division, absolute_import

import os
import tempfile
import xml.etree.ElementTree as ET

import pytest

import matplotlib
matplotlib.use('Agg')

pytest.importorskip('simplekml')

import wmpl.Utils.TrajectoryKML as TrajectoryKML
from wmpl.Utils.Tests.test_TrajectoryKML import KML_NS, _solve



def testTheFlareStarsSayWhetherTheFlareWasUsedInTheTrajectory():
    """ Check that the description of a flare star says whether the flare was used in the trajectory, and so
        whether it is projected from its line of sight or is at the point of the trajectory at its time.
    """

    traj = _solve()
    obs = traj.observations[1]
    flares = [{'station_id': str(obs.station_id), 't_rel': obs.time_data[k], 'used': used, \
        'lat': obs.model_lat[k], 'lon': obs.model_lon[k], 'ht': obs.model_ht[k]} \
        for k, used in [(5, True), (10, False)]]

    with tempfile.TemporaryDirectory() as tmp_dir:
        kml_path = TrajectoryKML.trajectoryToKML(traj, os.path.join(tmp_dir, 'test_3D.kml'), flares=flares)
        root = ET.parse(kml_path).getroot()

    folders = {folder.find(KML_NS + 'name').text: folder for folder in root.iter(KML_NS + 'Folder')}
    descriptions = [placemark.find(KML_NS + 'description').text \
        for placemark in folders["3. Flares"].findall(KML_NS + 'Placemark')]

    assert "Used in the trajectory: yes, projected from its line of sight" in descriptions[0]
    assert "Used in the trajectory: no, at the point of the trajectory at its time" in descriptions[1]
