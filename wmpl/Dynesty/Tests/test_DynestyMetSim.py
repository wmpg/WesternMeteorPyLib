""" Regression tests for the three logLikelihoodDynesty/runSimulationDynesty bugs found in
DynestyMetSim.py, all of which let a bad model draw leak a finite log-likelihood into the
nested-sampling likelihood surface instead of being rejected with -inf:

    - out-of-range np.interp queries used to clamp to the simulation's edge value instead of
      returning NaN, so a simulation that terminates before covering the observed height range
      slipped past the NaN-count guard that was supposed to reject it
    - the frame-integration branch of the luminosity comparison (the common case, since camera
      frame time is normally longer than the simulation step) never had that NaN-count guard at
      all - only the plain-interpolation branch did
    - a ZeroDivisionError during the simulation used to be swallowed and silently replaced with an
      unrelated nominal Constants() simulation, so dynesty would score a completely different model
      at that live point without any indication it happened
    - the marginal-mode histogram in the results summary table (and, near-identically, in the
      posterior distribution plot) ran on the raw, NaN-containing samples/weights instead of the
      already NaN-masked x_valid/w_valid used for every other statistic, so np.min/np.max silently
      became NaN whenever a parameter had any NaN sample - which np.histogram then turns into a hard
      ValueError, crashing the whole results/plotting stage. Both call sites now share
      _weightedHistogramMode(), which only ever operates on pre-masked input.

Every simulation result here is a hand-built stand-in (SimpleNamespace) with runSimulationDynesty
patched out, so these run in milliseconds instead of paying for a real MetSim/dynesty evaluation.

Run under pytest, or directly:

    python -m wmpl.Dynesty.Tests.test_DynestyMetSim
"""

import warnings
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import wmpl.Dynesty.DynestyMetSim as DynestyMetSim


def _makeObs(**overrides):
    """ Minimal ObservationData-like stand-in with just the attributes logLikelihoodDynesty reads. """
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


def _makeSim(height_max=100000.0, height_min=80000.0, n=5, dt=1.0):
    """ Minimal SimulationResults-like stand-in: a straight descent from height_max to height_min,
    sampled every dt seconds (dt=1.0 by default since most tests never reach the integration branch
    that actually depends on realistic time spacing). """
    height = np.linspace(height_max, height_min, n)  # decreasing, like a real leading_frag_height_arr
    time = np.arange(n)*dt
    return SimpleNamespace(
        leading_frag_height_arr=height,
        time_arr=time,
        leading_frag_length_arr=time*1000.0,
        luminosity_arr=np.linspace(50.0, 150.0, n),
        const=SimpleNamespace(dt=dt),
    )


def test_truncated_simulation_is_rejected():
    """ Bug 1: obs heights below the simulation's lowest reached height (75000/70000, while the sim
    stops at 80000) must make the NaN-count guard fire, since np.interp is no longer allowed to clamp
    them to the simulation's edge time/value. """
    sim = _makeSim(height_min=80000.0)
    obs = _makeObs()

    with patch.object(DynestyMetSim, "runSimulationDynesty", return_value=sim):
        result = DynestyMetSim.logLikelihoodDynesty([1.0], obs, {"v_init": []}, {})

    assert result == -np.inf, "truncated simulation was not rejected (np.interp clamping regression)"


def test_integration_branch_checks_coverage():
    """ Bug 2: force the frame-integration branch (1/fps_lum > dt) with a realistic, dt-spaced time
    grid, and make the last observed frame time (100.0) fall far outside every simulated time sample.
    integrateLuminosity() returns NaN there while obs.luminosity has a real value at that index, so
    the branch must now reject the draw instead of silently nansum-ing over the mismatch. """
    sim = _makeSim(height_max=100000.0, height_min=80000.0, n=500, dt=0.01)  # 1/30 s frame >> 0.01 s dt
    obs = _makeObs(
        height_lum=np.array([98000.0, 95000.0, 92000.0, 90000.0]),
        time_lum=np.array([0.5, 1.5, 2.5, 100.0]),  # last point far beyond any simulated time
        luminosity=np.array([100.0, 90.0, 80.0, 70.0]),
    )

    with patch.object(DynestyMetSim, "runSimulationDynesty", return_value=sim), \
         warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # expected empty-slice mean at i=3
        result = DynestyMetSim.logLikelihoodDynesty([1.0], obs, {"v_init": []}, {})

    assert result == -np.inf, "integration branch let a coverage mismatch through"


def test_zero_division_rejected_not_swallowed():
    """ Bug 3: a ZeroDivisionError raised while running the simulation must reject the point (-inf),
    not silently fall back to an unrelated nominal Constants() simulation. """
    obs = _makeObs()

    with patch.object(DynestyMetSim, "runSimulationDynesty", side_effect=ZeroDivisionError("boom")):
        result = DynestyMetSim.logLikelihoodDynesty([1.0], obs, {"v_init": []}, {})

    assert result == -np.inf, "ZeroDivisionError was not converted to a -inf rejection"


def test_run_simulation_dynesty_propagates_zero_division():
    """ Bug 3, lower level: runSimulationDynesty itself must not swallow ZeroDivisionError into a
    fallback nominal simulation - it must propagate so the caller can decide how to reject the draw. """
    dummy_const = SimpleNamespace()
    with patch.object(DynestyMetSim, "constructConstants", return_value=dummy_const), \
         patch.object(DynestyMetSim, "runSimulation", side_effect=ZeroDivisionError("boom")):
        try:
            DynestyMetSim.runSimulationDynesty([1.0], object(), ["v_init"], {})
        except ZeroDivisionError:
            pass
        else:
            raise AssertionError("runSimulationDynesty swallowed ZeroDivisionError into a fallback")


def test_weighted_histogram_mode_recovers_true_peak():
    """ Bug 4: _weightedHistogramMode (shared by summaryResultsTable's Mode_{Ndim} column and the
    posterior distribution plot) must reproduce a sane weighted mode from pre-masked samples. """
    rng = np.random.default_rng(0)
    x_valid = rng.normal(loc=5.0, scale=1.0, size=5000)
    w_valid = np.full(x_valid.size, 1.0/x_valid.size)

    centers, hist = DynestyMetSim._weightedHistogramMode(x_valid, w_valid, smooth=0.02)
    mode = centers[np.argmax(hist)]

    assert np.isfinite(mode), "mode is not finite on clean, pre-masked input"
    assert abs(mode - 5.0) < 0.5, "recovered mode is far from the true peak at 5.0"


def test_unmasked_nan_input_raises_documenting_why_callers_must_mask():
    """ Bug 4 root cause, pinned down: feeding _weightedHistogramMode raw (unmasked) samples containing
    a NaN reproduces the original crash (np.min/np.max of a NaN-containing array is NaN, and
    np.histogram rejects a NaN range outright). This is why both call sites now mask into
    x_valid/w_valid before calling the helper - if this assertion ever stops raising, np.histogram's
    behavior changed and the masking discipline in the two callers should be re-checked. """
    x = np.array([1.0, 2.0, np.nan, 3.0])
    w = np.full(4, 0.25)

    try:
        DynestyMetSim._weightedHistogramMode(x, w, smooth=0.02)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError from an unmasked NaN range - masking may be needed elsewhere too")


if __name__ == "__main__":
    test_truncated_simulation_is_rejected()
    test_integration_branch_checks_coverage()
    test_zero_division_rejected_not_swallowed()
    test_run_simulation_dynesty_propagates_zero_division()
    test_weighted_histogram_mode_recovers_true_peak()
    test_unmasked_nan_input_raises_documenting_why_callers_must_mask()
    print("All DynestyMetSim log-likelihood regression checks passed.")
