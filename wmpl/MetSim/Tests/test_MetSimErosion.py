""" Regression tests for the MetSimErosion reference engine.

Runs one complex scenario - two-phase erosion + EF/A/D complex fragmentation + a compressive-strength
disruption of the main fragment, all in one flight - and asserts the invariants that the silent
state-overwrite bugfixes restored:

    - mass never goes negative (the "ablate what's left" clamp now floors at exactly 0 instead of
      overshooting into negative mass, which used to propagate as NaN luminosity)
    - no NaN/inf anywhere in the per-tick outputs
    - lum_eroded/tau_eroded are tracked even with fragmentation_show_individual_lcs False (default),
      since that light already flows into lum_total
    - brightest_height is never a spurious 0 while the meteor is still luminous (a fragment's own
      death tick now stays a valid brightest/leading candidate)
    - total active mass is n_grains-weighted, so it is always >= the main fragment's mass alone

Run under pytest, or directly:

    python -m wmpl.MetSim.Tests.test_MetSimErosion         # run the asserts
    python -m wmpl.MetSim.Tests.test_MetSimErosion --plot  # also save a diagnostic figure
"""

import os
import sys

import numpy as np

import wmpl.MetSim.MetSimErosion as MetSimErosion
from wmpl.MetSim.GUI import FragmentationEntry
from wmpl.Utils.AtmosphereDensity import fitAtmPoly
from wmpl.Utils.TrajConversions import date2JD


# Column layout of each row in ablateAll()'s results list
COL_TIME = 0
COL_LUM_TOTAL = 1
COL_LUM_MAIN = 2
COL_LUM_ERODED = 3
COL_BRIGHTEST_HEIGHT = 8
COL_BRIGHTEST_VEL = 10
COL_MASS_TOTAL_ACTIVE = 15
COL_MAIN_MASS = 16


# Shared atmosphere density polynomial (covers the whole descent below h_init=80000.0) - fit once and
# reused by every Constants built below, instead of refitting per call.
H_REF = 220000.0
DENS_CO = fitAtmPoly(np.radians(45.3), np.radians(18.1), 20000, H_REF,
                     date2JD(2020, 4, 20, 16, 15, 0))


def _makeComplexScenarioEntries():
    """ The three FragmentationEntry events used by _makeComplexScenarioConstants() - a fresh list
    every call, since FragmentationEntry objects are mutated during a run (.done/.time/.mass/etc) and
    must never be reused across runs. Heights spread the three events across the flight, all between
    erosion_height_start/erosion_height_change (erosion already active), above where disruption itself
    later triggers (~65.6km in the default-parameter case).

    "A" (74000.0) sits below erosion_height_change (75000.0) on purpose: this is the exact case that
    used to be silently undone one tick later by the pre-fix erosion_height_change block -
    erosion_sigma_change is set equal to the initial sigma below, so the ONLY thing that should change
    frag.sigma at all during this flight is this "A" event. """

    return [
        FragmentationEntry("EF", 76500.0, 2, 30.0, 0.015e-6, 1.0, 0.4e-6, 1e-10, 5e-10, 2.0),
        FragmentationEntry("A", 74000.0, None, None, 0.02e-6, 1.0, None, None, None, None),
        FragmentationEntry("D", 71000.0, None, 15.0, None, None, None, 1e-10, 5e-10, 2.0),
    ]


def _makeComplexScenarioConstants(m_init=0.5, v_init=16000.0, h_init=80000.0, zenith_deg=45.0,
        rho=3300, sigma=0.015e-6, compressive_strength=40000.0):
    """ Build a fresh Constants for the "complex case" scenario: continuous two-phase erosion (rho
    only - erosion_sigma_change/erosion_coeff_change are set equal to their initial values), the three
    _makeComplexScenarioEntries() fragmentation events (EF/A/D), AND a compressive-strength disruption
    of the main fragment, all in one flight.

    compressive_strength=40000.0 pushes disruption down to ~65.6km, below all three EF/A/D events,
    without going so high the fragment never disrupts (dies some other way, e.g. v_kill, first). """

    const = MetSimErosion.Constants()
    const.erosion_on = True
    const.disruption_on = True
    const.fragmentation_on = True
    const.dens_co = DENS_CO
    const.h_init = h_init
    const.zenith_angle = np.radians(zenith_deg)
    const.m_init = m_init
    const.v_init = v_init
    const.rho = rho
    const.sigma = sigma
    const.erosion_height_start = 78000.0
    const.erosion_coeff = 0.3e-6
    const.erosion_height_change = 75000.0
    const.erosion_coeff_change = 0.3e-6
    const.erosion_rho_change = 3700
    const.erosion_sigma_change = sigma
    const.compressive_strength = compressive_strength
    const.disruption_mass_index = 2.0
    const.disruption_mass_min_ratio = 0.01
    const.disruption_mass_max_ratio = 0.1
    const.disruption_mass_grain_ratio = 0.25
    const.fragmentation_entries = _makeComplexScenarioEntries()
    return const


def _runComplexScenario():
    """ Run the complex scenario once and return (const, results) with results as a float ndarray. """
    const = _makeComplexScenarioConstants()
    _, results_list, _ = MetSimErosion.runSimulation(const)
    return const, np.array(results_list, dtype=float)


def test_disruption_is_triggered():
    """ The scenario must actually exercise the disruption path (otherwise the other asserts are
    vacuous). """
    const, _ = _runComplexScenario()
    assert const.disruption_height > 0, "disruption never triggered - scenario is not exercising it"


def test_no_nan_or_inf_outputs():
    """ Mass-clamp fix: negative mass used to propagate as NaN luminosity. Nothing should be
    non-finite. """
    _, results = _runComplexScenario()
    assert np.all(np.isfinite(results)), "non-finite value in per-tick outputs"


def test_mass_never_negative():
    """ Mass-clamp fix: the main mass and the total active mass must never go below zero. """
    _, results = _runComplexScenario()
    assert np.all(results[:, COL_MAIN_MASS] >= 0), "main fragment mass went negative"
    assert np.all(results[:, COL_MASS_TOTAL_ACTIVE] >= 0), "total active mass went negative"


def test_total_active_mass_is_grain_weighted():
    """ n_grains-weighting fix: total active mass (main + grains + daughters) must always be at least
    the main fragment's own mass. Pre-fix it summed one grain per bin and could dip below. """
    _, results = _runComplexScenario()
    assert np.all(results[:, COL_MASS_TOTAL_ACTIVE] >= results[:, COL_MAIN_MASS] - 1e-12), \
        "total active mass fell below the main fragment mass (n_grains weighting lost)"


def test_lum_eroded_tracked_by_default():
    """ lum_eroded gating fix: with fragmentation_show_individual_lcs at its default (False), the
    eroded/disrupted luminosity must still be tracked (it already flows into lum_total). """
    _, results = _runComplexScenario()
    assert np.nanmax(results[:, COL_LUM_ERODED]) > 0, \
        "lum_eroded stayed 0 with the default flag - eroded-light tracking is gated again"


def test_brightest_height_not_spuriously_zero_while_luminous():
    """ Brightest death-tick fix: while the meteor is still producing light, brightest_height must be
    a real height, never a spurious 0 (which happened when a fragment's own death tick was excluded
    from candidacy). """
    _, results = _runComplexScenario()
    luminous = results[:, COL_LUM_TOTAL] > 0
    heights = results[luminous, COL_BRIGHTEST_HEIGHT]
    assert np.all(heights > 0), \
        "brightest_height was 0 on a tick where the meteor was still luminous"


def _savePlot(save_path=None):
    """ New-engine-only diagnostic figure (LC, mass, velocity, height vs time). Optional, for eyeballing. """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, r = _runComplexScenario()
    t = r[:, COL_TIME]
    P_0m = 840.0
    mag = -2.5*np.log10(np.maximum(r[:, COL_LUM_TOTAL], 1e-10)/P_0m)

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    ax[0, 0].plot(t, r[:, COL_LUM_TOTAL], 'b-'); ax[0, 0].set_title("Total luminosity (W)")
    ax[0, 1].plot(t, mag, 'b-'); ax[0, 1].invert_yaxis(); ax[0, 1].set_title("Total magnitude")
    ax[1, 0].plot(t, r[:, COL_MASS_TOTAL_ACTIVE]*1000.0, 'b-', label="total active")
    ax[1, 0].plot(t, r[:, COL_MAIN_MASS]*1000.0, 'r--', label="main"); ax[1, 0].legend()
    ax[1, 0].set_title("Mass (g)")
    ax[1, 1].plot(t[:-1], r[:-1, COL_BRIGHTEST_VEL]/1000.0, 'b-')
    ax[1, 1].set_title("Brightest fragment velocity (km/s)")
    for a in ax.ravel():
        a.set_xlabel("Time (s)"); a.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is None:
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            "complex_scenario.png")
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    print("Saved diagnostic plot to {:s}".format(save_path))
    return save_path


if __name__ == "__main__":
    test_disruption_is_triggered()
    test_no_nan_or_inf_outputs()
    test_mass_never_negative()
    test_total_active_mass_is_grain_weighted()
    test_lum_eroded_tracked_by_default()
    test_brightest_height_not_spuriously_zero_while_luminous()
    print("All MetSimErosion regression checks passed.")

    if "--plot" in sys.argv:
        _savePlot()
