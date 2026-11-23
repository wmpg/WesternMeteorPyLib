""" Tests of the time offset of the simulation from the first lag point in the likelihood of the Dynesty fit,
which compares the lengths at the observed times instead of passing the simulation exactly through the first
point.

Run with pytest:
    python -m pytest wmpl/Dynesty/Tests/test_DynestyLagTimeOffset.py -v
"""

from __future__ import print_function, division, absolute_import

import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import wmpl.Dynesty.DynestyMetSim as DynestyMetSim
from wmpl.Dynesty.Tests.test_DynestyGroundFrame import _observations
from wmpl.MetSim.Tests.DisplacedFirstPoint import V_INIT, FIRST_POINT_ERROR, simulate, observe


# Noise of the lengths in the likelihood (m)
NOISE_LAG = 10.0



def dynestyObservations(sr, first_point_error, fit_lag_time_offset=True):
    """ Make a stand-in for ObservationData from the displaced first point observations of the simulated
        meteor, with the light curve of the simulation at the heights of the exact points.

    Arguments:
        sr: [SimulationResults object] Simulation results.
        first_point_error: [float] Along-track error of the first point (m).

    Keyword arguments:
        fit_lag_time_offset: [bool] Fit the time offset of the simulation. True by default.

    Return:
        [SimpleNamespace] The observations, sorted by height from the top as in ObservationData.
    """

    traj = observe(sr, first_point_error)
    traj_exact = observe(sr, 0.0)

    time_lag = np.concatenate([obs.time_data for obs in traj.observations])
    length = np.concatenate([obs.state_vect_dist for obs in traj.observations])
    height_lag = np.concatenate([obs.model_ht for obs in traj.observations])
    height_exact = np.concatenate([obs.model_ht for obs in traj_exact.observations])
    stations_lag = np.concatenate([[obs.station_id]*len(obs.time_data) for obs in traj.observations])

    # Sort the points by height from the top
    order = np.argsort(height_lag)[::-1]
    time_lag, length, height_lag, height_exact, stations_lag = time_lag[order], length[order], \
        height_lag[order], height_exact[order], stations_lag[order]

    # Take the simulated light curve, so only the lengths differ from the simulation
    luminosity = np.interp(height_exact, np.flip(sr.leading_frag_height_arr), np.flip(sr.luminosity_arr))

    return SimpleNamespace(height_lum=height_exact, time_lum=time_lag, luminosity=luminosity, \
        fps_lum=1000.0, P_0m=840.0, height_lag=height_lag, time_lag=time_lag, length=length, \
        stations_lag=stations_lag, lag=length - V_INIT*time_lag, v_init=V_INIT, noise_lum=1.0, \
        noise_lag=NOISE_LAG, fit_lag_time_offset=fit_lag_time_offset)



def logLikelihood(sr, obs):
    """ Compute the log-likelihood of the Dynesty fit for the given simulation.

    Arguments:
        sr: [SimulationResults object] Simulation results.
        obs: [SimpleNamespace] Observations made by dynestyObservations().

    Return:
        [float] Log-likelihood.
    """

    with patch.object(DynestyMetSim, 'runSimulationDynesty', return_value=sr):
        return DynestyMetSim.logLikelihoodDynesty([V_INIT], obs, {'v_init': []}, {})



def testTimeOffsetTakesUpFirstPointError():
    """ Check that the fitted offset is the time the meteor takes to move by the error of the first point, and
        that only the moved first point then lowers the likelihood.
    """

    sr = simulate()
    obs = dynestyObservations(sr, FIRST_POINT_ERROR)

    _, time_offset = DynestyMetSim.simulatedLengthsDynesty(sr, obs)
    assert np.isclose(time_offset, FIRST_POINT_ERROR/V_INIT, rtol=0.01)

    # The moved first point is the only residual, so the likelihood is lower than with an exact first point
    #   by its squared error over twice the variance
    log_l_exact = logLikelihood(sr, dynestyObservations(sr, 0.0))
    log_l_offset = logLikelihood(sr, obs)

    assert np.isclose(log_l_exact - log_l_offset, 0.5*FIRST_POINT_ERROR**2/NOISE_LAG**2, rtol=0.05)



def testWithoutTimeOffsetTheSimulationPassesTheFirstPoint():
    """ Check that without the offset the simulation starts late by the time the meteor takes to move by the
        error of the first point, so the length residuals of the other points are the error scaled by the
        speed.
    """

    sr = simulate()
    obs = dynestyObservations(sr, FIRST_POINT_ERROR, fit_lag_time_offset=False)

    len_sim, time_offset = DynestyMetSim.simulatedLengthsDynesty(sr, obs)
    assert time_offset == 0.0

    vel = np.interp(obs.height_lag, np.flip(sr.leading_frag_height_arr), np.flip(sr.leading_frag_vel_arr))
    assert np.allclose((obs.length - len_sim)[1:], -FIRST_POINT_ERROR*vel[1:]/vel[0], rtol=0.02)

    log_l_anchor = logLikelihood(sr, obs)
    log_l_offset = logLikelihood(sr, dynestyObservations(sr, FIRST_POINT_ERROR))
    assert log_l_offset - log_l_anchor > 0.4*len(obs.length)*FIRST_POINT_ERROR**2/NOISE_LAG**2



def testSimulationStoppingAboveTheFirstPointIsRejected():
    """ Check that a simulation which does not reach the first lag point is rejected. """

    sr = simulate()
    sr_short = simulate(h_kill=95000.0)

    obs = dynestyObservations(sr, FIRST_POINT_ERROR)
    obs.height_lum = np.array([99000.0, 97000.0])
    obs.time_lum = np.array([0.0, 0.1])
    obs.luminosity = np.array([1.0, 1.0])

    len_sim, time_offset = DynestyMetSim.simulatedLengthsDynesty(sr_short, obs)
    assert np.all(np.isnan(len_sim))
    assert time_offset == 0.0
    assert logLikelihood(sr_short, obs) == -np.inf



def testLagNoiseIgnoresTheLinearTrend():
    """ Check that the noise estimated from the lag does not count its linear trend, which comes from the
        velocity the lag is computed with, for a meteor at a constant speed and a decelerating one.
    """

    rng = np.random.RandomState(0)
    time_lag = np.linspace(0, 5, 251)
    noise = rng.normal(0, 1.0, len(time_lag))

    def fitLagNoise(lag):
        """ Estimate the noise of the given lag. """

        return DynestyMetSim.ObservationData.fitLagNoise(SimpleNamespace(lag=lag, time_lag=time_lag))

    # A lag computed with a velocity 100 m/s higher than the initial one has the noise of the lag without
    #   the trend
    deceleration = -50.0*np.clip(time_lag - 3.0, 0, None)**2
    for lag in [noise, deceleration + noise]:
        assert np.isclose(fitLagNoise(lag - 100.0*time_lag), fitLagNoise(lag), rtol=0.02)

    # At a constant speed the noise is recovered
    assert 0.8 < fitLagNoise(noise - 100.0*time_lag) < 1.2



def testExampleObservationsAndSavedTimeOffset():
    """ Check the likelihood of the shipped CAMO example with and without the offset, and that the saved
        constants keep the fitted offset.
    """

    obs = _observations()
    var_names = ['v_init', 'zenith_angle', 'm_init']
    guess = [obs.v_init, obs.zenith_angle, obs.m_init]
    flags_dict = {var_name: [] for var_name in var_names}

    # The lags are computed from the lengths with the velocity of the simulated lag
    assert np.allclose(obs.lag, obs.length - obs.v_init*obs.time_lag)

    # The profiled likelihood is at least that of the simulation through the first point
    obs.fit_lag_time_offset = True
    log_l_offset = DynestyMetSim.logLikelihoodDynesty(list(guess), obs, flags_dict, {})
    obs.fit_lag_time_offset = False
    log_l_anchor = DynestyMetSim.logLikelihoodDynesty(list(guess), obs, flags_dict, {})

    assert np.isfinite(log_l_offset) and np.isfinite(log_l_anchor)
    assert log_l_offset >= log_l_anchor

    # The saved constants keep the setting and the fitted offset
    obs.fit_lag_time_offset = True
    sr = DynestyMetSim.runSimulationDynesty(list(guess), obs, var_names, {})
    _, time_offset = DynestyMetSim.simulatedLengthsDynesty(sr, obs)

    with tempfile.TemporaryDirectory() as dir_path:
        DynestyMetSim.constructConstants(list(guess), obs, var_names, {}, dir_path=dir_path, \
            file_name='test_sim_fit.json')

        with open(os.path.join(dir_path, 'test_sim_fit.json')) as f:
            const_json = json.load(f)

    assert const_json['fit_lag_time_offset'] is True
    assert np.isclose(const_json['lag_time_offset'], time_offset, rtol=0, atol=1e-6)




if __name__ == "__main__":

    # Run all tests
    test_list = [testTimeOffsetTakesUpFirstPointError, \
        testWithoutTimeOffsetTheSimulationPassesTheFirstPoint, \
        testSimulationStoppingAboveTheFirstPointIsRejected, testLagNoiseIgnoresTheLinearTrend, \
        testExampleObservationsAndSavedTimeOffset]

    for test in test_list:
        test()
        print("PASS {:s}".format(test.__name__))
