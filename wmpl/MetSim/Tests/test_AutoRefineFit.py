""" Tests of the cost function in wmpl.MetSim.AutoRefineFit.

Run with pytest:
    python -m pytest wmpl/MetSim/Tests/test_AutoRefineFit.py -v
"""

from __future__ import print_function, division, absolute_import

from types import SimpleNamespace

import numpy as np

from wmpl.MetSim.AutoRefineFit import costFunc


# Speed of the simulated meteor (m/s) and its rate of descent (m/s)
SIM_VEL = 20000.0
SIM_DESCENT = 10000.0



def _simulation():
    """ Make the simulation results of a meteor moving at a constant speed with a constant magnitude of 0,
        from 100 to 60 km.

    Return:
        [SimpleNamespace] Simulation results.
    """

    time_arr = np.linspace(0, 4, 401)

    return SimpleNamespace(time_arr=time_arr, leading_frag_height_arr=100e3 - SIM_DESCENT*time_arr, \
        leading_frag_length_arr=SIM_VEL*time_arr, leading_frag_vel_arr=np.zeros_like(time_arr) + SIM_VEL, \
        abs_magnitude=np.zeros_like(time_arr))



def _observation(n_points, abs_mag, len_offset):
    """ Make an observation of the simulated meteor with a constant magnitude and a constant offset in
        length.

    Arguments:
        n_points: [int] Number of points.
        abs_mag: [float] Absolute magnitude of all points.
        len_offset: [float] Offset of the observed lengths from the simulated ones (m).

    Return:
        [SimpleNamespace] Observation.
    """

    time_data = np.linspace(0.5, 3.5, n_points)

    return SimpleNamespace(station_id="XX0001", time_data=time_data, \
        model_ht=100e3 - SIM_DESCENT*time_data, state_vect_dist=SIM_VEL*time_data + len_offset, \
        absolute_magnitudes=np.zeros(n_points) + abs_mag)



def testLengthResidualIsAveragedOverThePointsUsed():
    """ Check that the points fainter than +9, which are left out of the length residual, are not counted
        in its average.
    """

    obs_bright = _observation(10, 0.0, 10.0)
    obs_faint = _observation(30, 10.0, 1000.0)
    traj = SimpleNamespace(rbeg_ele=100e3, observations=[obs_bright, obs_faint])

    mag_res, len_res, cost = costFunc(traj, None, _simulation(), 1.0, 1.0)

    assert np.isclose(mag_res, 0.0)
    assert np.isclose(len_res, 10.0)
    assert np.isclose(cost, 10.0)



def testLengthResidualWithoutFaintPoints():
    """ Check that the length residual is the average over all points when none is fainter than +9. """

    traj = SimpleNamespace(rbeg_ele=100e3, observations=[_observation(10, 0.0, 10.0), \
        _observation(30, 2.0, -50.0)])

    _, len_res, _ = costFunc(traj, None, _simulation(), 1.0, 1.0)

    assert np.isclose(len_res, (10*10.0 + 30*50.0)/40)




if __name__ == "__main__":

    # Run all tests
    test_list = [testLengthResidualIsAveragedOverThePointsUsed, testLengthResidualWithoutFaintPoints]

    for test in test_list:
        test()
        print("PASS {:s}".format(test.__name__))
