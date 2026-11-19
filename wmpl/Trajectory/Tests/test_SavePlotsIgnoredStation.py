""" Tests that the trajectory solver saves its plots when a station has NaN heights, as an ignored station
    whose clock is far off can have with the gravity correction.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_SavePlotsIgnoredStation.py -v
"""

from __future__ import print_function, division, absolute_import

import os
import tempfile

import numpy as np

import matplotlib
matplotlib.use('Agg')

from wmpl.Trajectory.Tests.SyntheticTrajectory import DragMeteor, STATIONS, observeMeteor


# Clock error of station D (s), far beyond the largest estimated time offset
CLOCK_ERROR = 600.0



def makeTrajectory(output_dir, clock_error=None):
    """ Make the trajectory of the synthetic meteor seen from the three stations of SyntheticTrajectory.py
        and a fourth one, D, with the gravity correction and saved results.

    Arguments:
        output_dir: [str] Output directory.

    Keyword arguments:
        clock_error: [float] Clock error of station D (s). None by default, for no error.

    Return:
        [Trajectory object] Trajectory, not yet solved.
    """

    # Make the meteor
    meteor = DragMeteor(20000.0, 45.0, 40.0, 1e-3, 3e-3, 2.0, 5.0)

    # Shift the times of station D by its clock error
    clock_offsets = {}
    if clock_error is not None:
        clock_offsets['D'] = clock_error

    # Observe the meteor from every station
    traj = observeMeteor(meteor, noise=0.0, stations=STATIONS + [('D', 45.3, 15.5, 300.0, 0.0, 1.0)], \
        clock_offsets=clock_offsets, gravity_correction=True, save_results=True, output_dir=output_dir, \
        calc_orbit=True, max_toffset=10.0)

    return traj



def testThePlotsAreSavedWithAStationWhoseClockIsFarOff(tmp_path):
    """ Check that the solver saves its plots when a station is ignored because its clock is far off, which
        can make its heights NaN or hundreds of km off with the gravity correction.

    Arguments:
        tmp_path: [Path] Temporary directory of the test.
    """

    traj = makeTrajectory(str(tmp_path), clock_error=CLOCK_ERROR).run()

    # Check that station D is ignored
    ignored = [obs for obs in traj.observations if obs.station_id == 'D'][0]
    assert ignored.ignore_station

    assert os.path.isfile(os.path.join(str(tmp_path), traj.file_name + '_velocities.png'))



def testThePlotsAreSavedWithNaNHeights(tmp_path):
    """ Check that the plots are saved when the station plotted last has NaN heights.

    Arguments:
        tmp_path: [Path] Temporary directory of the test.
    """

    traj = makeTrajectory(str(tmp_path)).run()

    # Set NaN heights to the station with the highest first height, plotted last
    last = sorted(traj.observations, key=lambda obs: obs.model_ht[0])[-1]
    last.meas_ht = np.full_like(last.meas_ht, np.nan)

    output_dir = os.path.join(str(tmp_path), 'nan')
    traj.savePlots(output_dir, traj.file_name, show_plots=False)

    assert os.path.isfile(os.path.join(output_dir, traj.file_name + '_velocities.png'))



if __name__ == "__main__":

    # Run all tests, each in its own temporary directory
    test_list = [testThePlotsAreSavedWithAStationWhoseClockIsFarOff, testThePlotsAreSavedWithNaNHeights]

    for test in test_list:
        test(tempfile.mkdtemp())
        print("PASS {:s}".format(test.__name__))

    print("All tests passed.")
