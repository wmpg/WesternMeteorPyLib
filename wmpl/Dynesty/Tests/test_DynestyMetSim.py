""" Tests that logLikelihoodDynesty() rejects the simulations which do not cover the observations or fail, and
    that the marginal mode of the results is computed from the finite samples only.

The simulations are stand-ins, so the tests run without MetSim or dynesty.

Run with pytest:
    python -m pytest wmpl/Dynesty/Tests/test_DynestyMetSim.py -v
"""

from __future__ import print_function, division, absolute_import

import warnings
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

import wmpl.Dynesty.DynestyMetSim as DynestyMetSim



def makeObs(**overrides):
    """ Make a stand-in for ObservationData with the attributes logLikelihoodDynesty() reads.

    Keyword arguments:
        overrides: [dict] Attributes to replace.

    Return:
        [SimpleNamespace] The observations.
    """

    obs = SimpleNamespace(
        height_lum=np.array([98000.0, 90000.0, 82000.0, 75000.0, 70000.0]),
        time_lum=np.array([0.1, 1.0, 2.0, 3.0, 4.0]),
        luminosity=np.array([100.0, 90.0, 80.0, 70.0, 60.0]),
        fps_lum=30.0,
        P_0m=840.0,
        height_lag=np.array([98000.0, 90000.0, 82000.0]),
        lag=np.array([0.0, 1.0, 2.0]),
        v_init=16000.0,
        noise_lum=1.0,
        noise_lag=1.0,
    )

    for key, value in overrides.items():
        setattr(obs, key, value)

    return obs



def makeSim(height_max=100000.0, height_min=80000.0, n=5, dt=1.0):
    """ Make a stand-in for SimulationResults, a straight descent sampled every time step.

    Keyword arguments:
        height_max: [float] First height (m). 100 km by default.
        height_min: [float] Last height (m). 80 km by default.
        n: [int] Number of samples. 5 by default.
        dt: [float] Time step (s). 1 s by default.

    Return:
        [SimpleNamespace] The simulation.
    """

    time = np.arange(n)*dt

    return SimpleNamespace(
        leading_frag_height_arr=np.linspace(height_max, height_min, n),
        time_arr=time,
        leading_frag_length_arr=time*1000.0,
        luminosity_arr=np.linspace(50.0, 150.0, n),
        const=SimpleNamespace(dt=dt),
    )



def testSimulationEndingAboveTheObservationsIsRejected():
    """ Check that a simulation which ends at 80 km, above observations down to 70 km, is rejected. """

    with patch.object(DynestyMetSim, 'runSimulationDynesty', return_value=makeSim(height_min=80000.0)):
        assert DynestyMetSim.logLikelihoodDynesty([1.0], makeObs(), {'v_init': []}, {}) == -np.inf



def testIntegratedLuminosityOutsideTheSimulationIsRejected():
    """ Check that a simulation whose luminosity, integrated over the frames, does not cover an observed frame
        is rejected.
    """

    # Frames of 1/30 s, longer than the time step, so the luminosity is integrated, and the last frame is
    #   after the end of the simulation
    sim = makeSim(n=500, dt=0.01)
    obs = makeObs(height_lum=np.array([98000.0, 95000.0, 92000.0, 90000.0]), \
        time_lum=np.array([0.5, 1.5, 2.5, 100.0]), luminosity=np.array([100.0, 90.0, 80.0, 70.0]))

    with patch.object(DynestyMetSim, 'runSimulationDynesty', return_value=sim), warnings.catch_warnings():

        # The mean of the empty frame warns
        warnings.simplefilter('ignore', category=RuntimeWarning)

        assert DynestyMetSim.logLikelihoodDynesty([1.0], obs, {'v_init': []}, {}) == -np.inf



def testFailedSimulationIsRejected():
    """ Check that a ZeroDivisionError in the simulation rejects the draw, instead of a nominal simulation
        being scored in its place.
    """

    with patch.object(DynestyMetSim, 'runSimulationDynesty', side_effect=ZeroDivisionError('division')):
        assert DynestyMetSim.logLikelihoodDynesty([1.0], makeObs(), {'v_init': []}, {}) == -np.inf

    with patch.object(DynestyMetSim, 'constructConstants', return_value=SimpleNamespace()), \
            patch.object(DynestyMetSim, 'runSimulation', side_effect=ZeroDivisionError('division')):
        with pytest.raises(ZeroDivisionError):
            DynestyMetSim.runSimulationDynesty([1.0], object(), ['v_init'], {})



def testWeightedHistogramMode():
    """ Check that the weighted histogram of normal samples peaks at their mean. """

    x = np.random.default_rng(0).normal(loc=5.0, scale=1.0, size=5000)
    w = np.full(x.size, 1.0/x.size)

    centers, hist = DynestyMetSim.weightedHistogram(x, w, smooth=0.02)

    assert abs(centers[np.argmax(hist)] - 5.0) < 0.5



if __name__ == "__main__":

    testSimulationEndingAboveTheObservationsIsRejected()
    testIntegratedLuminosityOutsideTheSimulationIsRejected()
    testFailedSimulationIsRejected()
    testWeightedHistogramMode()

    print('All tests passed.')
