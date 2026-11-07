""" Tests of the plots of wmpl.Trajectory.Trajectory with a station ignored because its clock is far off,
    whose heights are NaN with the gravity correction.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_SavePlotsIgnoredStation.py -v
"""

from __future__ import print_function, division, absolute_import

import os

import numpy as np

import matplotlib
matplotlib.use('Agg')

from wmpl.Trajectory.Trajectory import Trajectory
from wmpl.Trajectory.Tests.SyntheticTrajectory import DragMeteor, JDT_REF, STATIONS, stationAzimAlt


# Error of the clock of the fourth station (s), far more than the largest time offset the solver estimates
CLOCK_ERROR = 600.0



def testThePlotsAreSavedWithAStationWhoseClockIsFarOff(tmp_path):
    """ Check that the solver saves its plots when a station is ignored because its clock is far off, whose
        heights are NaN with the gravity correction, as the drop is computed at times far from the meteor.
    """

    meteor = DragMeteor(20000.0, 45.0, 40.0, 1e-3, 3e-3, 2.0, 5.0)
    traj = Trajectory(JDT_REF, meastype=2, gravity_correction=True, monte_carlo=False, show_plots=False, \
        save_results=True, verbose=False, output_dir=str(tmp_path), calc_orbit=True, max_toffset=10.0)

    rng = np.random.default_rng(1)
    duration = meteor.t_vis_end - meteor.t_vis_beg
    for station_id, lat, lon, ele, frac_beg, frac_end in STATIONS + [('D', 45.3, 15.5, 300.0, 0.0, 1.0)]:

        times = np.arange(frac_beg*duration + rng.uniform(0, 0.04), frac_end*duration, 0.04)
        azim, elev = stationAzimAlt(meteor.eciAtTime(times), JDT_REF + times/86400, lat, lon, ele)

        clock = CLOCK_ERROR if (station_id == 'D') else 0.0
        traj.infillTrajectory(azim, elev, times + clock, np.radians(lat), np.radians(lon), ele, \
            station_id=station_id)

    traj = traj.run()

    ignored = [obs for obs in traj.observations if obs.station_id == 'D'][0]
    assert ignored.ignore_station
    assert not np.any(np.isfinite(ignored.meas_ht))

    assert os.path.isfile(os.path.join(str(tmp_path), traj.file_name + '_velocities.png'))
