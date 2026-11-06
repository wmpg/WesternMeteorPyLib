""" Tests of the sampling of trajectory positions by height and by time in
    wmpl.Utils.SampleTrajectoryPositions, on the example trajectory 20191023_091225.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_SampleTrajectoryPositions.py -v
"""

from __future__ import print_function, division, absolute_import

import contextlib
import io
import os
import tempfile

import numpy as np

import wmpl
from wmpl.Utils.Pickling import loadPickle
from wmpl.Utils.SampleTrajectoryPositions import sampleTrajectory
from wmpl.Utils.SampleTrajectoryPositions import SAMPLE_ATTRIBUTES, TrajectorySamples
from wmpl.Utils.Physics import dynamicPressure


# Directory and the file name of the example trajectory
TRAJ_DIR = os.path.join(os.path.dirname(wmpl.__file__), 'Dynesty', 'examples', '20191023_091225')
TRAJ_FILE = '20191023_091225_trajectory.pickle'



def _sampleArrays(traj_samples):
    """ Convert the times, heights and velocities of the samples to arrays.

    Arguments:
        traj_samples: [TrajectorySamples object] Sampled positions.

    Return:
        (t_arr, ht_arr, vel_arr):
            - t_arr: [ndarray] Times of the samples (s).
            - ht_arr: [ndarray] Heights of the samples (m).
            - vel_arr: [ndarray] Velocities of the samples (m/s).
    """

    return np.array(traj_samples.t_est, dtype=np.float64), np.array(traj_samples.ht, dtype=np.float64), \
        np.array(traj_samples.vel, dtype=np.float64)



def testHeightModeReturnsTheRequestedHeights():
    """ Check that the samples are at the requested heights above the mean sea level. """

    traj = loadPickle(TRAJ_DIR, TRAJ_FILE)

    traj_samples = sampleTrajectory(traj, 120000.0, 90000.0, 1000.0)
    _, ht_arr, _ = _sampleArrays(traj_samples)

    assert len(ht_arr) == 31
    assert np.allclose(ht_arr, np.arange(120000.0, 89999.0, -1000.0))
    assert np.allclose(traj_samples.ele_geo, ht_arr, atol=0.1)



def testTimeModeReturnsTheRequestedTimes():
    """ Check that the samples are taken at the requested times, also before the beginning and after the
        end of the fireball.
    """

    traj = loadPickle(TRAJ_DIR, TRAJ_FILE)

    t_arr, _, _ = _sampleArrays(sampleTrajectory(traj, -0.2, 0.6, 0.05, by_time=True))

    assert len(t_arr) == 17
    assert np.allclose(t_arr, np.arange(-0.2, 0.6 + 0.025, 0.05), atol=1e-9)



def testTimeModeMatchesTheHeightMode():
    """ Check that sampling by time and by height gives the same motion. """

    traj = loadPickle(TRAJ_DIR, TRAJ_FILE)

    t_fine, ht_fine, _ = _sampleArrays(sampleTrajectory(traj, 130000.0, 80000.0, 10.0))
    t_arr, ht_arr, _ = _sampleArrays(sampleTrajectory(traj, -0.1, 0.5, 0.05, by_time=True))

    sort_ind = np.argsort(t_fine)
    assert np.max(np.abs(np.interp(t_arr, t_fine[sort_ind], ht_fine[sort_ind]) - ht_arr)) < 1.0



def testVelocitiesArePhysical():
    """ Check that the velocity above the beginning stays at the initial velocity, and that it decreases
        after the end.
    """

    traj = loadPickle(TRAJ_DIR, TRAJ_FILE)

    t_arr, _, vel_arr = _sampleArrays(sampleTrajectory(traj, -0.5, 0.8, 0.05, by_time=True))

    # In the 0.5 s above the beginning the velocity changes by less than 0.1%, and it agrees with the
    #   initial velocity of the solution to 0.5%
    vel_beg = vel_arr[np.argmin(np.abs(t_arr))]
    assert np.all(np.abs(vel_arr[t_arr < 0]/vel_beg - 1.0) < 1e-3)
    assert abs(vel_beg/traj.v_init - 1.0) < 5e-3

    # The velocity never increases and stays positive
    assert np.all(np.diff(vel_arr) <= 1.0)
    assert np.all(vel_arr > 0)




def testSkippedSamplesAreReported():
    """ Check that the samples outside the modelled part of the trajectory are skipped and reported. """

    traj = loadPickle(TRAJ_DIR, TRAJ_FILE)

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        t_arr, _, _ = _sampleArrays(sampleTrajectory(traj, 0.0, 1000.0, 10.0, by_time=True))

    assert len(t_arr) < 101
    assert "samples were skipped" in out.getvalue()



def testSamplesHaveTheMotionAndTheDynamics():
    """ Check the positions, the uncertainties and the dynamics stored for every sample when sampling by
        height, from above the beginning (116.1 km) to below the end (96.2 km).
    """

    traj = loadPickle(TRAJ_DIR, TRAJ_FILE)

    traj_samples = sampleTrajectory(traj, 118000.0, 94000.0, 2000.0)
    drag_fit = traj_samples.drag_fit
    arrays = traj_samples.toArrays()

    n_samples = len(arrays['t_est'])
    assert n_samples == 13
    assert sorted(arrays.keys()) == sorted(SAMPLE_ATTRIBUTES)
    assert arrays['eci'].shape == (n_samples, 3)
    assert arrays['radiant_eci'].shape == (n_samples, 3)

    # The lengths are at the sampled heights, and the steps between the ECI points are the steps of length,
    #   up to the change of the gravity drop
    assert np.allclose(drag_fit.heightAtLength(arrays['length']), arrays['ht'], atol=0.1)
    eci_step = np.linalg.norm(np.diff(arrays['eci'], axis=0), axis=1)
    assert np.allclose(eci_step, np.diff(arrays['length']), rtol=1e-4)

    # The radiant points back along the path
    path_back = -np.diff(arrays['eci'], axis=0)
    path_back /= np.linalg.norm(path_back, axis=1)[:, None]
    assert np.allclose(np.linalg.norm(arrays['radiant_eci'], axis=1), 1.0)
    assert np.all(np.sum(path_back*arrays['radiant_eci'][1:], axis=1) > np.cos(np.radians(0.01)))

    # The first sample is above the beginning, where the meteor moves without drag, and the last two are
    #   below the end
    assert list(arrays['extrapolated']) == [True] + [False]*10 + [True]*2
    assert arrays['k'][0] == 0
    assert np.all(arrays['k'][1:] > 0)
    assert arrays['decel'][0] < 0
    assert np.all(arrays['decel'][1:] > 0)
    assert np.allclose(arrays['decel'], drag_fit.decelerationAtLength(arrays['length']))

    # The heights are exact, the times and the velocities are uncertain
    assert np.all(arrays['len_std'] == 0)
    assert np.all(np.isfinite(arrays['time_std']) & (arrays['time_std'] > 0))
    assert np.all(np.isfinite(arrays['vel_std']) & (arrays['vel_std'] > 0))
    assert np.all(np.isfinite(arrays['log_k_std'][1:]) & (arrays['log_k_std'][1:] > 0))

    # The velocity is known best in the observed part
    assert np.max(arrays['vel_std'][~arrays['extrapolated']]) < arrays['vel_std'][-1]



def testTimeModeUncertainties():
    """ Check that when sampling by time the times are exact and the lengths are uncertain. """

    traj = loadPickle(TRAJ_DIR, TRAJ_FILE)

    arrays = sampleTrajectory(traj, 0.0, 0.3, 0.05, by_time=True).toArrays()

    assert np.all(arrays['time_std'] == 0)
    assert np.all(np.isfinite(arrays['len_std']) & (arrays['len_std'] > 0))
    assert np.all(np.isfinite(arrays['vel_std']) & (arrays['vel_std'] > 0))



def testDynamicPressureIsRelativeToTheGround():
    """ Check that the dynamic pressure is computed with the velocity relative to the ground, and that the
        air density of the model agrees with the density computed at every sample.
    """

    traj = loadPickle(TRAJ_DIR, TRAJ_FILE)

    traj_samples = sampleTrajectory(traj, 116000.0, 96000.0, 2000.0)
    arrays = traj_samples.toArrays()
    rho = traj_samples.drag_fit.airDensityAtLength(arrays['length'])

    # The velocities in the ECI frame and relative to the ground differ by about 50 m/s
    assert np.all(np.abs(arrays['vel_norot'] - arrays['vel']) > 10.0)
    assert np.allclose(arrays['dyn_pressure'], rho*arrays['vel_norot']**2, rtol=1e-9)

    jd_arr = traj.jdt_ref + arrays['t_est']/86400.0
    dyn_pressure = dynamicPressure(arrays['lat'], arrays['lon'], arrays['ht'], jd_arr, arrays['vel_norot'])
    assert np.allclose(arrays['dyn_pressure'], dyn_pressure, rtol=0.02)



def testSaveCSV():
    """ Check that the CSV file has one row per sample with the sampled values. """

    traj = loadPickle(TRAJ_DIR, TRAJ_FILE)

    traj_samples = sampleTrajectory(traj, 118000.0, 94000.0, 2000.0)
    arrays = traj_samples.toArrays()

    with tempfile.TemporaryDirectory() as dir_path:

        file_path = os.path.join(dir_path, 'samples.csv')
        traj_samples.saveCSV(file_path)

        with open(file_path) as f:
            lines = f.readlines()

        data = np.genfromtxt(file_path, delimiter=',', skip_header=5)


    header = [name.strip() for name in lines[4].split(',')]
    assert data.shape == (len(arrays['t_est']), len(header))

    def _column(name):
        return data[:, header.index(name)]

    assert np.allclose(_column('Time (s)'), arrays['t_est'], atol=1e-6)
    assert np.allclose(_column('Height (m)'), arrays['ht'], atol=0.01)
    assert np.allclose(_column('Lat (deg)'), np.degrees(arrays['lat']), atol=1e-8)
    assert np.allclose(_column('Vel (m/s)'), arrays['vel'], atol=0.01)
    assert np.allclose(_column('Vel std (m/s)'), arrays['vel_std'], atol=0.01)
    assert np.allclose(_column('Vgnd (m/s)'), arrays['vel_norot'], atol=0.01)
    assert np.allclose(_column('Dyn pres (Pa)'), arrays['dyn_pressure'], rtol=1e-6)
    assert np.allclose(_column('Z ECI (m)'), arrays['eci'][:, 2], atol=0.01)
    assert np.array_equal(_column('Extrapolated'), arrays['extrapolated'].astype(int))



def testSamplesWithoutTheNewQuantities():
    """ Check that samples added only with the positions give NaN for the other quantities. """

    traj = loadPickle(TRAJ_DIR, TRAJ_FILE)

    traj_samples = TrajectorySamples(traj)
    traj_samples.addSample(0.1, 100000.0, 0.7, -1.4, 100000.0, 2.8, 1.1)
    arrays = traj_samples.toArrays()

    assert arrays['ht'][0] == 100000.0
    assert np.isnan(arrays['vel'][0]) and np.isnan(arrays['dyn_pressure'][0])
    assert arrays['eci'].shape == (1, 3) and np.all(np.isnan(arrays['eci']))
    assert not arrays['extrapolated'][0]




if __name__ == "__main__":

    # Run all tests
    test_list = [testHeightModeReturnsTheRequestedHeights, testTimeModeReturnsTheRequestedTimes, \
        testTimeModeMatchesTheHeightMode, testVelocitiesArePhysical, testSkippedSamplesAreReported, \
        testSamplesHaveTheMotionAndTheDynamics, testTimeModeUncertainties, \
        testDynamicPressureIsRelativeToTheGround, testSaveCSV, testSamplesWithoutTheNewQuantities]

    for test in test_list:
        test()
        print("PASS {:s}".format(test.__name__))
