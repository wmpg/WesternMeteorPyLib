""" Tests for the position of a flare point not used in the trajectory, which is only a time and must take
    the point of the trajectory at that time, not where its line of sight crosses the trajectory.

Run with pytest:
    python -m pytest wmpl/Formats/Tests/test_ECSVFlareUnusedPoint.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np

from wmpl.Formats.ECSV import flareHeights, loadECSVs
from wmpl.Formats.GenericFunctions import solveTrajectoryGeneric
from wmpl.Formats.Tests.test_ECSVUnsolvedFragments import FRAGMENT_START, writeEvent
from wmpl.Utils.TrajConversions import latLonAlt2ECEF



def testAFlareNotUsedInTheTrajectoryTakesThePointOfTheTrajectoryAtItsTime(tmp_path):
    """ Check that a flare point not used in the trajectory, clicked 5 frames further along the path than
        the meteor was at its time, lies on the trajectory where the meteor was at its time.
    """

    frame = FRAGMENT_START + 5
    paths = writeEvent(tmp_path)

    path = [path for path in paths if path.endswith('XX0002.ecsv')][0]
    with open(path) as f:
        lines = f.read().splitlines()

    # Take the azimuth and altitude of 5 frames later for the flare, and leave it out of the trajectory
    rows = {int(line.split(',')[8]): n for n, line in enumerate(lines) \
        if (not line.startswith('#')) and (not line.startswith('datetime'))}
    fields = lines[rows[frame]].split(',')
    fields[3:5] = lines[rows[frame + 5]].split(',')[3:5]
    fields[9:11] = ['True', 'False']
    lines[rows[frame]] = ','.join(fields)

    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")

    # Solve the event
    jdt_ref, meteor_list = loadECSVs(paths)
    traj = solveTrajectoryGeneric(jdt_ref, meteor_list, str(tmp_path), solver='original', monte_carlo=False, \
        save_results=False, show_plots=False, verbose=False, calc_orbit=False, gravity_correction=False, \
        estimate_timing_vel=False)

    flare = flareHeights(traj, paths)[0]
    assert not flare['used']

    # The point of station XX0001 on the same frame, at the same time without clock errors
    obs = [obs for obs in traj.observations if str(obs.station_id) == 'XX0001'][0]
    k = np.argmin(np.abs(obs.JD_data - flare['jd']))
    assert abs(obs.JD_data[k] - flare['jd'])*86400 < 1e-6

    point = np.array(latLonAlt2ECEF(obs.model_lat[k], obs.model_lon[k], obs.model_ht[k]))
    flare_point = np.array(latLonAlt2ECEF(flare['lat'], flare['lon'], flare['ht']))

    assert np.linalg.norm(flare_point - point) < 5.0
