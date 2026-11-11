""" Tests for the projection of the flares of ECSV files on the main trajectory, which must convert their
    R.A./Dec. as the observations given to the solver are.

Run with pytest:
    python -m pytest wmpl/Formats/Tests/test_ECSVFlareProjection.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np

from wmpl.Formats.ECSV import flareHeights
from wmpl.Formats.Tests.test_ECSVFlarePressure import flaggedEvent
from wmpl.Formats.Tests.test_ECSVUnsolvedFragments import FRAGMENT_START



def testAFlareUsedInTheTrajectoryIsProjectedOnItsPointOfTheSolution(tmp_path):
    """ Check that a flare point used in the trajectory is projected where the solver put the same pick,
        which needs the nutation and the time every R.A./Dec. was computed at, as for the solver.
    """

    paths, traj = flaggedEvent(tmp_path, FRAGMENT_START + 5)
    flare = flareHeights(traj, paths)[0]

    assert flare['used']

    obs = [obs for obs in traj.observations if str(obs.station_id) == flare['station_id']][0]
    k = np.argmin(np.abs(obs.JD_data - flare['jd']))

    assert abs(obs.JD_data[k] - flare['jd'])*86400 < 1e-6
    assert np.linalg.norm(np.array(obs.model_eci[k]) - flare['eci']) < 0.01
    assert abs(flare['ht'] - obs.model_ht[k]) < 0.01
