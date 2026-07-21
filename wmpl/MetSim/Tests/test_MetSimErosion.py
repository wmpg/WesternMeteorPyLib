""" Manual/visual inspection tool - runs the SAME complex scenario (two-phase erosion + EF/A/D
fragmentation events + compressive-strength disruption of the main fragment) through both:

    - wmpl.MetSim.MetSimErosion       - the current, fixed reference engine
    - wmpl.MetSim.MetSimErosion_old   - a frozen snapshot of the pre-fix code (see below)

and saves a 3x3 panel figure (2 lines per panel) so the two can be eyeballed against each other.

MetSimErosion_old.py is a plain `git show HEAD:wmpl/MetSim/MetSimErosion.py` snapshot taken before
the bugfixes below were applied - not maintained, just a frozen diffing target:
    - erosion_height_change's rho/sigma change re-applied every tick instead of once, silently
      undoing a later complex-fragmentation "A" event's sigma change one tick after it fired
    - mass_total_active summed frag.m directly instead of frag.m*frag.n_grains (inconsistent with
      how lum/electron_density are already weighted)
    - lum_eroded/tau_eroded stayed 0.0 whenever fragmentation_show_individual_lcs was False (the
      default), even though those fragments' light was already flowing into lum_total
    - brightest_*/leading_frag_* excluded a fragment's own death tick from candidacy, so the final
      row of a run without erosion could report brightest_height=0.0 instead of the real height
    - disruption_erosion_coeff was overwritten by the generic height-based getErosionCoeff() on the
      very next tick after a disruption daughter was assigned it, making the parameter nearly a no-op

Deliberately not a pytest test (no test_* function) - a manual/exploratory tool, run directly:

    python -c "from wmpl.MetSim.Tests.test_MetSimErosion import plotComplexOldVsNewComparison as p; p()"
"""

import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import wmpl.MetSim.MetSimErosion as MetSimErosionNew
import wmpl.MetSim.MetSimErosion_old as MetSimErosionOld
from wmpl.MetSim.GUI import FragmentationEntry
from wmpl.Utils.AtmosphereDensity import fitAtmPoly
from wmpl.Utils.TrajConversions import date2JD


def _timeIt(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


# Shared atmosphere density polynomial (30-85 km covers the whole descent below h_init=80000.0) -
# fit once and reused by every Constants built below, instead of refitting per call/per engine.
H_REF = 220000.0
DENS_CO = fitAtmPoly(np.radians(45.3), np.radians(18.1), 20000, H_REF,
                      date2JD(2020, 4, 20, 16, 15, 0))


def _makeComplexScenarioEntries():
    """ The three FragmentationEntry events used by _makeComplexScenarioConstantsPair() - a fresh
    list every call, since FragmentationEntry objects are mutated during a run (.done/.time/.mass/
    etc) and must never be reused across engines or repeated runs. Heights spread the three events
    across the flight, all between erosion_height_start/erosion_height_change (erosion already
    active), above where disruption itself later triggers (~65.6km in the default-parameter case).

    "A" (74000.0) sits below erosion_height_change (75000.0) on purpose: this is the exact case
    that used to be silently undone one tick later by the pre-fix erosion_height_change block (see
    this module's own top-of-file docstring) - erosion_sigma_change is set equal to the initial
    sigma below, so the ONLY thing that should change frag.sigma at all during this flight is this
    "A" event, making the old-vs-new divergence unambiguous. """

    return [
        FragmentationEntry("EF", 76500.0, 2, 30.0, 0.015e-6, 1.0, 0.4e-6, 1e-10, 5e-10, 2.0),
        FragmentationEntry("A", 74000.0, None, None, 0.02e-6, 1.0, None, None, None, None),
        FragmentationEntry("D", 71000.0, None, 15.0, None, None, None, 1e-10, 5e-10, 2.0),
    ]


def _makeComplexScenarioConstantsPair(m_init=0.5, v_init=16000.0, h_init=80000.0,
        zenith_deg=45.0, rho=3300, sigma=0.015e-6, compressive_strength=40000.0):
    """ Build a fresh (new, old) Constants pair - one per engine module - for the "complex case"
    scenario: continuous two-phase erosion (rho only - see erosion_sigma_change/erosion_coeff_change
    below), the three _makeComplexScenarioEntries() fragmentation events (EF/A/D), AND a
    compressive-strength disruption of the main fragment, all in one flight.

    compressive_strength=40000.0 (rather than a much lower value) is deliberate: disruption height
    is governed by atmosphere/dynamic-pressure physics and lands at essentially the same real
    altitude regardless of h_init, so a low compressive_strength disrupts the main fragment before
    it ever reaches erosion_height_change/the EF/A/D events below - 40000.0 pushes disruption down
    to ~65.6km, below all three events, without going so high the fragment never disrupts at all
    (dies some other way, e.g. v_kill, first). """

    consts = []
    for ConstantsClass in (MetSimErosionNew.Constants, MetSimErosionOld.Constants):
        const = ConstantsClass()
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
        consts.append(const)
    return consts



def plotComplexOldVsNewComparison(m_init=0.5, v_init=16000.0, h_init=80000.0, zenith_deg=45.0,
        rho=3300, sigma=0.015e-6, compressive_strength=40000.0, save_path=None):
    """ Run the complex scenario (see _makeComplexScenarioConstantsPair) through both MetSimErosion.py
    (current, fixed) and MetSimErosion_old.py (frozen pre-fix snapshot) and save a 3x3 panel figure
    (2 lines per panel) to save_path, for visual side-by-side inspection.

    Returns:
        save_path: [str] Where the figure was written.
    """

    def _makePair():
        return _makeComplexScenarioConstantsPair(m_init=m_init, v_init=v_init, h_init=h_init,
            zenith_deg=zenith_deg, rho=rho, sigma=sigma, compressive_strength=compressive_strength)

    def _timeNew():
        const_new, _ = _makePair()
        MetSimErosionNew.runSimulation(const_new)

    def _timeOld():
        _, const_old = _makePair()
        MetSimErosionOld.runSimulation(const_old)

    n_reps = 5
    new_time = min(_timeIt(_timeNew) for _ in range(n_reps))
    old_time = min(_timeIt(_timeOld) for _ in range(n_reps))

    print("Execution time (best of {:d} runs): MetSimErosion.py = {:.3f} ms   "
        "MetSimErosion_old.py = {:.3f} ms".format(n_reps, new_time*1000, old_time*1000))

    const_new, const_old = _makePair()
    _, results_new, _ = MetSimErosionNew.runSimulation(const_new)
    _, results_old, _ = MetSimErosionOld.runSimulation(const_old)

    results_new = np.array(results_new, dtype=float)
    results_old = np.array(results_old, dtype=float)

    print("Disruption height: new={:.1f}m old={:.1f}m".format(
        const_new.disruption_height, const_old.disruption_height))

    # Column layout (ablateAll return tuple, identical across both): 0 time, 1 lum_total, 2 lum_main,
    # 3 lum_eroded, 4 electron_density_total, 5 tau_total, 6 tau_main, 7 tau_eroded, 8 brightest_height,
    # 9 brightest_length, 10 brightest_vel, 11 leading_frag_height, 12 leading_frag_length,
    # 13 leading_frag_vel, 14 leading_frag_dyn_press, 15 mass_total_active, 16 main_mass,
    # 17 main_height, 18 main_length, 19 main_vel, 20 main_dyn_press.
    t_new = results_new[:, 0]
    t_old = results_old[:, 0]

    # main_vel/main_mass/main_height are zeroed for every row AFTER the main fragment's own death
    # (disruption here) - truncate to the pre-death portion (each run's own) before differentiating,
    # to avoid a huge, physically meaningless deceleration spike at the drop-to-zero
    death_new = np.argmax(results_new[:, 16] == 0.0) or len(t_new)
    death_old = np.argmax(results_old[:, 16] == 0.0) or len(t_old)
    a_new = np.gradient(results_new[:death_new, 19], t_new[:death_new])
    a_old = np.gradient(results_old[:death_old, 19], t_old[:death_old])

    fig, axes = plt.subplots(3, 3, figsize=(16, 13))

    _STYLES = (("-", "tab:blue"), ("--", "tab:orange"))

    def _panel(ax, series, ylabel, title):
        for (t, y, label), (ls, color) in zip(series, _STYLES):
            ax.plot(t, y, ls, color=color, label=label)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    def _series(t_n, y_n, t_o, y_o):
        return [(t_n, y_n, "MetSimErosion.py (fixed)"), (t_o, y_o, "MetSimErosion_old.py (pre-fix)")]

    _panel(axes[0, 0], _series(t_new, results_new[:, 1], t_old, results_old[:, 1]),
        "Luminous intensity (W)", "Total luminosity (main + grains + daughters)")
    _panel(axes[0, 1], _series(t_new, results_new[:, 2], t_old, results_old[:, 2]),
        "Luminous intensity (W)", "Main-fragment-only luminosity")
    _panel(axes[0, 2], _series(t_new, results_new[:, 16]*1000.0, t_old, results_old[:, 16]*1000.0),
        "Mass (g)", "Main fragment mass")

    # Total magnitude - clipped away from log(0) at the pre-ablation start
    P_0m = 840.0
    mag_new = -2.5*np.log10(np.maximum(results_new[:, 1], 1e-10)/P_0m)
    mag_old = -2.5*np.log10(np.maximum(results_old[:, 1], 1e-10)/P_0m)
    _panel(axes[1, 0], _series(t_new, mag_new, t_old, mag_old), "Magnitude", "Total magnitude")
    axes[1, 0].invert_yaxis()

    # brightest_vel/leading_frag_*/brightest_height are zeroed on the tick every active fragment dies
    # simultaneously - only ever the very last row of a whole run (fixed in MetSimErosion.py, still
    # present in MetSimErosion_old.py) - excluded here ([:-1]) so the plot doesn't show a spurious
    # drop-to-zero at the very end for EITHER engine
    _panel(axes[1, 1], _series(t_new[:-1], results_new[:-1, 10]/1000.0, t_old[:-1],
        results_old[:-1, 10]/1000.0), "Velocity (km/s)", "Brightest fragment velocity")
    _panel(axes[1, 2], _series(t_new[:death_new], a_new/1000.0, t_old[:death_old], a_old/1000.0),
        "Deceleration (km/s^2)", "Main fragment acceleration")

    _panel(axes[2, 0], _series(t_new, results_new[:, 15]*1000.0, t_old, results_old[:, 15]*1000.0),
        "Mass (g)", "Total active mass (main + grains + daughters)")
    _panel(axes[2, 1], _series(t_new[:-1], results_new[:-1, 12]/1000.0, t_old[:-1],
        results_old[:-1, 12]/1000.0), "Length (km)", "Leading fragment distance travelled")
    _panel(axes[2, 2], _series(t_new[:-1], results_new[:-1, 8]/1000.0, t_old[:-1],
        results_old[:-1, 8]/1000.0), "Height (km)", "Brightest fragment height")

    fig.suptitle(
        "MetSimErosion.py (fixed) vs MetSimErosion_old.py (pre-fix) - complex scenario: 2-phase "
        "erosion + EF/A/D fragmentation + disruption\n"
        "m={:.2f}kg, v_init={:.0f}km/s, h_init={:.0f}km, zenith={:.0f}deg, rho={:.0f}kg/m^3, "
        "sigma={:.3g}s^2/m^2, compressive_strength={:.0f}Pa\n"
        "Execution time (best of {:d} runs): fixed = {:.3f} ms   pre-fix = {:.3f} ms".format(
            m_init, v_init/1000.0, h_init/1000.0, zenith_deg, rho, sigma, compressive_strength,
            n_reps, new_time*1000, old_time*1000), fontsize=11)
    fig.tight_layout()

    if save_path is None:
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            "complex_scenario_old_vs_new.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print("Saved comparison plot to {:s}".format(save_path))

    return save_path


if __name__ == "__main__":
    plotComplexOldVsNewComparison()
