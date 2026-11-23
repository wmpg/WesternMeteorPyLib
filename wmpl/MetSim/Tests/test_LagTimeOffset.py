""" Tests of the time offset of the simulation from the first observed point, which MetSim fits so that the
simulation is compared with all observed lengths instead of passing exactly through the first point.

Run with pytest:
    python -m pytest wmpl/MetSim/Tests/test_LagTimeOffset.py -v
"""

from __future__ import print_function, division, absolute_import

import copy
from types import SimpleNamespace

import numpy as np

from wmpl.MetSim.AutoRefineFit import costFunc
from wmpl.MetSim.GUI import SimulationResults, MinimizationParameterNormalization, fitResiduals, \
    fitLagTimeOffset, lagTimeOffsetMax
from wmpl.MetSim.MetSimErosion import runSimulation
from wmpl.MetSim.Tests.test_MetSimGravityDrop import bodyConstants


# Initial speed of the simulated meteor (m/s)
V_INIT = 20000.0

# Height of the first observed point (m)
HT_BEG = 90000.0

# Along-track error of the first observed point (m)
FIRST_POINT_ERROR = 300.0



def simulate(v_init=V_INIT):
    """ Simulate a decelerating single body.

    Keyword arguments:
        v_init: [float] Initial speed (m/s). V_INIT by default.

    Return:
        sr: [SimulationResults object] Simulation results.
    """

    const = bodyConstants(0.1, v_init, 45.0, 20000.0)

    return SimulationResults(const, *runSimulation(const))



def observe(sr, first_point_error):
    """ Observe the simulated meteor from two stations at 25 frames per second between 90 and 45 km, without
        noise except for the first point, which is moved along the track. The lengths are measured from the
        first point, so all other lengths are shorter by the error, and the begin height is that of the
        moved point.

    Arguments:
        sr: [SimulationResults object] Simulation results.
        first_point_error: [float] Along-track error of the first point, positive in the direction of
            motion (m).

    Return:
        traj: [SimpleNamespace] Trajectory with the begin height and the observations of both stations.
    """

    time_arr, len_arr, ht_arr = sr.time_arr, sr.leading_frag_length_arr, sr.leading_frag_height_arr

    # Find when the meteor passes the first point
    time_beg = np.interp(-HT_BEG, -ht_arr, time_arr)
    len_beg = np.interp(time_beg, time_arr, len_arr)
    time_end = np.interp(-45000.0, -ht_arr, time_arr)

    observations = []
    for station_id, time_start in [("A", 0.0), ("B", 0.3)]:

        time_data = np.arange(time_start, time_end - time_beg, 0.04)

        # Lengths from the first point, and the heights at them
        len_true = np.interp(time_beg + time_data, time_arr, len_arr) - len_beg
        len_data = len_true - first_point_error
        model_ht = np.interp(len_beg + len_true, len_arr, ht_arr)

        # The first point of the first station is the reference, so its length is zero
        if station_id == "A":
            len_data[0] = 0.0
            model_ht[0] = np.interp(len_beg + first_point_error, len_arr, ht_arr)

        observations.append(SimpleNamespace(station_id=station_id, time_data=time_data, model_ht=model_ht, \
            state_vect_dist=len_data, absolute_magnitudes=None, ignore_list=np.zeros(len(time_data))))

    return SimpleNamespace(rbeg_ele=observations[0].model_ht[0], observations=observations)



def pointCount(traj):
    """ Count the observed points.

    Arguments:
        traj: [SimpleNamespace] Trajectory made by observe().

    Return:
        [int] Number of points of all stations.
    """

    return sum(len(obs.time_data) for obs in traj.observations)



def fitInputData(traj):
    """ Make the input data of the GUI fit, without magnitudes.

    Arguments:
        traj: [SimpleNamespace] Trajectory made by observe().

    Return:
        [ndarray] Rows of the height (m), magnitude, time (s) and length (m) of every point.
    """

    return np.vstack([np.c_[obs.model_ht, np.zeros_like(obs.model_ht) + np.nan, obs.time_data, \
        obs.state_vect_dist] for obs in traj.observations])



def normalizedSimulation(sr, rbeg_ele):
    """ Take the simulated time and length from the point at which the simulation passes the begin height.

    Arguments:
        sr: [SimulationResults object] Simulation results.
        rbeg_ele: [float] Begin height (m).

    Return:
        (norm_sim_time, norm_sim_len, norm_sim_vel): [tuple of ndarrays] Time (s), length (m) and velocity
            (m/s) of the simulation after the begin height.
    """

    ht_arr, len_arr = sr.leading_frag_height_arr, sr.leading_frag_length_arr
    len_beg = np.interp(-rbeg_ele, -ht_arr, len_arr)
    time_beg = np.interp(-rbeg_ele, -ht_arr, sr.time_arr)

    used = len_arr > len_beg

    return sr.time_arr[used] - time_beg, len_arr[used] - len_beg, sr.leading_frag_vel_arr[used]



def refitInitialSpeed(traj, fit_time_offset):
    """ Find the initial speed which minimizes the cost function of AutoRefineFit, with the other parameters
        fixed, on a grid of simulations and a parabola through the smallest values.

    Arguments:
        traj: [SimpleNamespace] Trajectory made by observe().
        fit_time_offset: [bool] Fit the time offset of the simulation.

    Return:
        [float] Initial speed of the smallest cost (m/s).
    """

    v_grid = V_INIT + np.arange(-200.0, 201.0, 25.0)
    cost = np.array([costFunc(traj, None, simulate(v_init), 1.0, 1.0, \
        fit_time_offset=fit_time_offset)[1] for v_init in v_grid])

    i = int(np.clip(np.argmin(cost), 2, len(v_grid) - 3))
    a, b, _ = np.polyfit(v_grid[i - 2:i + 3] - v_grid[i], cost[i - 2:i + 3], 2)

    return v_grid[i] - b/(2*a)



def testTimeOffsetTakesUpFirstPointError():
    """ Check that the fitted offset is the time the meteor takes to move by the error of the first point,
        with both losses, and that only the moved first point is then left with a residual.
    """

    sr = simulate()
    traj = observe(sr, FIRST_POINT_ERROR)

    # Fit the offset with the squared residuals
    norm_sim_time, norm_sim_len, norm_sim_vel = normalizedSimulation(sr, traj.rbeg_ele)
    time_offset_l2 = fitLagTimeOffset(norm_sim_time, norm_sim_len, norm_sim_vel, \
        np.concatenate([obs.time_data for obs in traj.observations]), \
        np.concatenate([obs.state_vect_dist for obs in traj.observations]))

    # Fit the offset with the absolute residuals of AutoRefineFit
    _, len_res, _, time_offset_l1 = costFunc(traj, None, sr, 1.0, 1.0)

    # The simulation reaches the moved first point later by the error over the speed
    assert np.isclose(time_offset_l2, FIRST_POINT_ERROR/V_INIT, rtol=0.01)
    assert np.isclose(time_offset_l1, FIRST_POINT_ERROR/V_INIT, rtol=0.01)
    assert np.isclose(len_res, FIRST_POINT_ERROR/pointCount(traj), rtol=0.05)



def testInitialSpeedIsRecoveredWithTimeOffset():
    """ Check that the initial speed is recovered with the time offset, and that fixing the simulation to the
        moved first point biases it.
    """

    traj = observe(simulate(), FIRST_POINT_ERROR)

    v_offset = refitInitialSpeed(traj, True)
    v_anchor = refitInitialSpeed(traj, False)

    assert abs(v_offset - V_INIT) < 3.0
    assert abs(v_anchor - V_INIT) > 30.0



def testWithoutTimeOffsetTheSimulationPassesTheFirstPoint():
    """ Check that without the offset the simulated length is zero at the height and the time of the first
        point, as before the offset was fitted, and that an exact first point gives a zero offset.
    """

    sr = simulate()

    # With the offset turned off, the residuals are those of the simulation through the first point
    traj = observe(sr, FIRST_POINT_ERROR)
    _, len_res, _, time_offset = costFunc(traj, None, sr, 1.0, 1.0, fit_time_offset=False)

    norm_sim_time, norm_sim_len, _ = normalizedSimulation(sr, traj.rbeg_ele)
    len_res_anchor = np.mean(np.concatenate([np.abs(obs.state_vect_dist \
        - np.interp(obs.time_data, norm_sim_time, norm_sim_len)) for obs in traj.observations]))

    assert time_offset == 0.0
    assert np.isclose(len_res, len_res_anchor, rtol=0.01)
    assert len_res > 100.0

    # With an exact first point, the fitted offset is zero
    _, len_res, _, time_offset = costFunc(observe(sr, 0.0), None, sr, 1.0, 1.0)

    assert abs(time_offset) < 1e-5
    assert len_res < 1.0



def testGUIFitResidualUsesTheTimeOffset():
    """ Check that the residual of the GUI fit is close to zero for the true parameters with the offset,
        and large without it.
    """

    sr = simulate()
    traj = observe(sr, FIRST_POINT_ERROR)
    fit_input_data = fitInputData(traj)

    # Fit the mass, the speed, the density and the ablation coefficient, at the true values
    const = copy.deepcopy(sr.const)
    p0 = [const.m_init, const.v_init, const.rho, const.sigma]
    mini_norm_handle = MinimizationParameterNormalization(p0)
    p0_normed, _ = mini_norm_handle.normalizeBounds([[0.5*x, 2*x] for x in p0])

    residual_offset = fitResiduals(p0_normed, fit_input_data, 'm', const, traj, mini_norm_handle, \
        verbose=False)
    residual_anchor = fitResiduals(p0_normed, fit_input_data, 'm', const, traj, mini_norm_handle, \
        verbose=False, fit_time_offset=False)

    # The residual is the mean squared length residual, with the offset only that of the moved first point
    assert np.isclose(residual_offset, FIRST_POINT_ERROR**2/pointCount(traj), rtol=0.05)
    assert residual_anchor > 100.0**2



def testTimeOffsetIsBounded():
    """ Check that the offset is bounded by two intervals between the points, also when the lengths are far
        from the simulation.
    """

    time_data = [np.arange(0, 1, 0.04), np.arange(0.1, 0.9, 0.04)]
    assert np.isclose(lagTimeOffsetMax(time_data), 0.08)
    assert lagTimeOffsetMax([np.array([0.0])]) == 0.0

    norm_sim_time = np.linspace(-1, 2, 301)
    time_data = np.arange(0, 1, 0.04)
    time_offset = fitLagTimeOffset(norm_sim_time, 20000.0*norm_sim_time, np.zeros_like(norm_sim_time) \
        + 20000.0, time_data, 20000.0*time_data + 5000.0, time_offset_max=0.08)

    assert np.isclose(time_offset, -0.08)




if __name__ == "__main__":

    # Run all tests
    test_list = [testTimeOffsetTakesUpFirstPointError, testInitialSpeedIsRecoveredWithTimeOffset, \
        testWithoutTimeOffsetTheSimulationPassesTheFirstPoint, testGUIFitResidualUsesTheTimeOffset, \
        testTimeOffsetIsBounded]

    for test in test_list:
        test()
        print("PASS {:s}".format(test.__name__))
