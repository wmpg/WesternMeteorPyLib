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

import copy
import os
import sys

import numpy as np
import pytest

import wmpl.MetSim.MetSimErosion as MetSimErosion
from wmpl.MetSim.GUI import FragmentationEntry
from wmpl.Utils.AtmosphereDensity import fitAtmPoly, atmDensPoly
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


### Winds ###

class _ShearedWind(object):
    """ Wind turning from 250 to 300 deg (the direction it blows from) and growing from 10 to 70 m/s between 20
        and 32 km, constant outside. MetSim only needs its wind(height) method. """

    def __init__(self, scale=1.0):
        self.scale = scale

    def wind(self, height):
        frac = np.clip((np.asarray(height, dtype=float) - 20000.0)/12000.0, 0, 1)
        speed, direction = self.scale*(10 + 60*frac), np.radians(250 + 50*frac)
        return -np.array([speed*np.sin(direction), speed*np.cos(direction)])


def _makeSingleBodyConstants(wind_profile=None, gravity_3d=False):
    """ A single body at the end of a fireball, as DynamicMassFit simulates it. """

    const = MetSimErosion.Constants()
    const.h_kill, const.v_kill = 15000, 3000
    const.m_init, const.v_init, const.h_init, const.rho = 0.112, 6303.0, 30000.0, 3500.0
    const.shape_factor, const.gamma, const.sigma = 1.21, 0.7, 0.005/1e6
    const.zenith_angle = np.radians(90 - 41.4)
    const.erosion_on, const.disruption_on, const.fragmentation_on = False, False, False
    const.dens_co = fitAtmPoly(np.radians(51.94), np.radians(-2.10), const.h_kill, const.h_init, 2459274.41)
    const.wind_profile, const.radiant_azimuth, const.gravity_3d = wind_profile, np.radians(264.0), gravity_3d

    return const


def _motion(const):
    """ Unit vector (east, north, up) of the initial direction of motion. """

    az, zc = const.radiant_azimuth, const.zenith_angle
    return np.array([-np.sin(zc)*np.sin(az), -np.sin(zc)*np.cos(az), -np.cos(zc)])


def _referenceRun(const):
    """ MetSim's single-body equations with the wind evaluated at every step, and gravity towards the Earth's
        centre and the Coriolis acceleration with const.gravity_3d, integrated with RK4 in 0.1 ms steps (converged) in 3D over a spherical
        Earth, as an independent check of MetSim in 3D. Returns the final position (east, north, up) from the
        start, the velocity relative to the ground and the mass. """

    K = const.gamma*const.shape_factor*const.rho**(-2/3.0)
    if const.wind_profile is None:
        wind = lambda h: np.zeros(3)
    else:
        wind = lambda h: np.append(const.wind_profile.wind(h), 0.0)
    centre = np.array([0.0, 0.0, -(const.r_earth + const.h_init)])
    height = lambda x: np.linalg.norm(x - centre) - const.r_earth

    def deriv(s):
        h = height(s[:3])
        u = s[3:6] - wind(h)
        rho = atmDensPoly(h, const.dens_co)
        acc = -K*s[6]**(-1/3.0)*rho*np.linalg.norm(u)*u
        if const.gravity_3d:
            acc = acc - MetSimErosion.G0*(const.r_earth/(const.r_earth + h))**2*(s[:3] - centre) \
                /np.linalg.norm(s[:3] - centre)
            if const.latitude is not None:
                omega = MetSimErosion.EARTH_ROTATION_RATE*np.array([0.0, np.cos(const.latitude), \
                    np.sin(const.latitude)])
                acc = acc - 2*np.cross(omega, s[3:6])
        return np.concatenate([s[3:6], acc, [-K*const.sigma*s[6]**(2/3.0)*rho*np.linalg.norm(u)**3]])

    s, dt = np.concatenate([np.zeros(3), const.v_init*_motion(const), [const.m_init]]), 1e-4
    while np.linalg.norm(s[3:6] - wind(height(s[:3]))) > const.v_kill:
        k1 = deriv(s); k2 = deriv(s + dt/2*k1); k3 = deriv(s + dt/2*k2); k4 = deriv(s + dt*k3)
        s = s + dt/6*(k1 + 2*k2 + 2*k3 + k4)

    return s[:3], s[3:6], s[6]


def _angle(a, b):
    return np.degrees(np.arccos(np.clip(np.dot(a, b)/np.linalg.norm(a)/np.linalg.norm(b), -1, 1)))


def test_winds_in_still_air_reproduce_the_simulation_without_winds():
    frag_still, _, _ = MetSimErosion.runSimulation(_makeSingleBodyConstants(_ShearedWind(scale=0.0)))
    frag_none, _, _ = MetSimErosion.runSimulation(_makeSingleBodyConstants())

    for name in ['m', 'v', 'h', 'length']:
        assert np.isclose(getattr(frag_still, name), getattr(frag_none, name), rtol=1e-9, atol=0), name


def test_winds_match_an_integration_with_the_wind_at_every_step():
    """ With a wind that turns and grows with height, MetSim matches the reference to within what its 5 ms steps
        leave without winds (a few metres and ~0.1% in mass), while ignoring the wind is off by much more. """

    const = _makeSingleBodyConstants(_ShearedWind())
    frag, _, _ = MetSimErosion.runSimulation(const)
    x_ref, v_ref, m_ref = _referenceRun(const)

    vel = np.array([frag.vx, frag.vy, frag.vz])
    assert np.linalg.norm(np.array([frag.px, frag.py, frag.pz]) - x_ref) < 15.0
    assert _angle(vel, v_ref) < 0.01
    assert abs(frag.m/m_ref - 1) < 0.002

    # Without winds the fragment keeps its initial direction, which the wind turns by much more than the above
    assert _angle(_motion(const), v_ref) > 0.1


def test_gravity_3d_matches_an_integration_with_gravity_at_every_step():
    """ With gravity and the Coriolis acceleration in the velocity, with and without winds, MetSim matches the
        reference to within what its 5 ms steps leave (a few metres, 0.001 deg), while gravity itself turns the path
        by 0.1 deg here. """

    for wind in [None, _ShearedWind()]:
        const = _makeSingleBodyConstants(wind, gravity_3d=True)
        const.latitude = np.radians(51.94)
        frag, _, _ = MetSimErosion.runSimulation(const)
        x_ref, v_ref, m_ref = _referenceRun(const)

        vel = np.array([frag.vx, frag.vy, frag.vz])
        assert np.linalg.norm(np.array([frag.px, frag.py, frag.pz]) - x_ref) < 15.0
        assert _angle(vel, v_ref) < 0.005
        assert abs(frag.m/m_ref - 1) < 0.002

        const.gravity_3d = False
        _, v_ref_no_gravity, _ = _referenceRun(const)
        assert _angle(v_ref_no_gravity, v_ref) > 0.05


### Backward runs ###

def _makeFireballConstants(gravity_3d=False):
    """ A single body from 120 km down to 50 km at 20 km/s, which loses 1.2 km/s and 27% of its mass on the way,
        so a backward run has something to undo. """

    const = MetSimErosion.Constants()
    const.h_init, const.h_kill, const.v_kill = 120000.0, 50000.0, 3000.0
    const.m_init, const.v_init, const.rho, const.sigma = 1.0, 20000.0, 3500.0, 0.014/1e6
    const.zenith_angle, const.radiant_azimuth = np.radians(45.0), np.radians(90.0)
    const.erosion_on, const.disruption_on, const.fragmentation_on = False, False, False
    const.dens_co = fitAtmPoly(np.radians(45.0), 0.0, 40000.0, 130000.0, 2460000.5)
    const.gravity_3d = gravity_3d
    if gravity_3d:
        const.latitude = np.radians(45.0)

    return const


def _backwardFrom(const, frag):
    """ Constants that run back up to const.h_init from where a forward run of const stopped, starting with the
        fragment's speed, mass and direction there. """

    back = copy.deepcopy(const)
    back.dt, back.h_kill = -const.dt, const.h_init
    back.h_init, back.v_init, back.m_init = frag.h, frag.v, frag.m

    if const.gravity_3d:

        # Local frame at the end of the forward run, which is where the backward run's frame starts
        up = np.array([frag.px, frag.py, frag.pz + const.r_earth + const.h_init])
        up /= np.linalg.norm(up)
        axis = np.array([0.0, np.cos(const.latitude), np.sin(const.latitude)])
        north = axis - np.dot(axis, up)*up
        north /= np.linalg.norm(north)
        radiant = -np.array([frag.vx, frag.vy, frag.vz])/frag.v
        back.zenith_angle = np.arccos(np.dot(radiant, up))
        back.radiant_azimuth = np.arctan2(np.dot(radiant, np.cross(north, up)), np.dot(radiant, north))
        back.latitude = np.arcsin(np.dot(axis, up))

    # The 2D path is a straight line, whose zenith angle grows going down. The (vv, vh) velocity components do
    #   not follow it, so they cannot give it
    else:
        back.zenith_angle = np.arcsin((const.r_earth + const.h_init)/(const.r_earth + frag.h) \
            *np.sin(const.zenith_angle))

    return back


def _roundTripErrors(gravity_3d, dt):
    """ Run forward, then backward from where the forward run stopped, and return the relative errors in the
        speed and mass recovered at the start, and the difference between the two run times (s). """

    const = _makeFireballConstants(gravity_3d)
    const.dt = dt
    frag, results, _ = MetSimErosion.runSimulation(const)
    frag_back, results_back, _ = MetSimErosion.runSimulation(_backwardFrom(const, frag))

    return frag_back.v/const.v_init - 1, frag_back.m/const.m_init - 1, \
        results[-1][COL_TIME] + results_back[-1][COL_TIME]


def test_backward_run_returns_to_the_start_of_a_forward_run():
    """ In 2D and with gravity in 3D, running back from where a forward run stopped recovers the starting speed
        and mass, the mass growing back from 0.73 to 1 kg. The error is first order in the time step, 10 times
        smaller with 10 times smaller steps, so it comes from the integration and not from the backward run. """

    for gravity_3d in [False, True]:
        errors = [np.abs(_roundTripErrors(gravity_3d, dt)) for dt in [0.005, 0.0005]]
        assert errors[1][0] < 1e-4 and errors[1][1] < 5e-4 and errors[1][2] < 1e-3, errors[1]
        assert np.all(errors[0][:2] > 7*errors[1][:2]), errors


def test_backward_run_stops_at_h_kill_or_t_kill():
    """ Running backwards, h_kill is the height to stop at, and t_kill stops it after that long, both within one
        step. """

    const = _makeFireballConstants()
    frag, _, _ = MetSimErosion.runSimulation(const)
    back = _backwardFrom(const, frag)
    step_height = const.v_init*abs(back.dt)

    frag_back, results_back, _ = MetSimErosion.runSimulation(back)
    assert back.h_kill < frag_back.h < back.h_kill + step_height

    back.t_kill = 1.0
    frag_back, results_back, _ = MetSimErosion.runSimulation(back)
    assert back.t_kill <= -results_back[-1][COL_TIME] < back.t_kill + 1.5*abs(back.dt)
    assert frag_back.h < back.h_kill


def test_freeze_mass_keeps_the_mass_in_both_directions():
    """ With freeze_mass, the mass stays at m_init going forwards and backwards, and the light, only the drag
        term, stays positive. """

    const = _makeFireballConstants()
    const.freeze_mass = True
    frag, results, _ = MetSimErosion.runSimulation(const)
    frag_back, results_back, _ = MetSimErosion.runSimulation(_backwardFrom(const, frag))

    assert frag.m == const.m_init and frag_back.m == const.m_init
    assert min(row[COL_LUM_TOTAL] for row in results + results_back) > 0


def test_backward_run_refuses_what_it_cannot_undo():
    """ Erosion, disruption, fragmentation and the wake cannot be run backwards, and a backward run must have a
        height to stop at above its start. """

    const = _makeFireballConstants()
    const.dt, const.h_init, const.h_kill = -0.005, 50000.0, 120000.0
    for name in ['erosion_on', 'disruption_on', 'fragmentation_on']:
        bad = copy.deepcopy(const)
        setattr(bad, name, True)
        with pytest.raises(ValueError):
            MetSimErosion.runSimulation(bad)

    with pytest.raises(ValueError):
        MetSimErosion.runSimulation(const, compute_wake=True)

    const.h_kill = 40000.0
    with pytest.raises(ValueError):
        MetSimErosion.runSimulation(const)


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
    test_winds_in_still_air_reproduce_the_simulation_without_winds()
    test_winds_match_an_integration_with_the_wind_at_every_step()
    test_gravity_3d_matches_an_integration_with_gravity_at_every_step()
    test_backward_run_returns_to_the_start_of_a_forward_run()
    test_backward_run_stops_at_h_kill_or_t_kill()
    test_freeze_mass_keeps_the_mass_in_both_directions()
    test_backward_run_refuses_what_it_cannot_undo()
    print("All MetSimErosion regression checks passed.")

    if "--plot" in sys.argv:
        _savePlot()
