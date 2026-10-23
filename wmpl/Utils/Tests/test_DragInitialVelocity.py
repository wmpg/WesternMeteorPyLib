""" Tests for the initial velocity from a drag and ablation fit (wmpl.Utils.DragInitialVelocity), and for the
v_init_drag option of the trajectory solver.

The observations are synthetic: a meteoroid decelerated by drag, with ablation, that drops below its straight line
and speeds up along it under gravity, observed by three stations at 25 frames per second with 20 arcsec of noise.

Run under pytest, or directly:

    python -m wmpl.Utils.Tests.test_DragInitialVelocity
"""

import contextlib
import copy
import io
import math
import os
from types import SimpleNamespace

import numpy as np

from wmpl.MetSim import MetSimErosion
from wmpl.Trajectory.Trajectory import Trajectory
from wmpl.Utils.AtmosphereDensity import AtmDensityTableInterp, getAtmDensityTable
from wmpl.Utils.DragInitialVelocity import atmosphereDescription, breakupNotes, fitDragInitialVelocity, \
    fittedTimeLimit
from wmpl.Utils.Pickling import loadPickle
from wmpl.Utils.TrajConversions import altAz2RADec, cartesian2Geo, eci2RaDec, geo2Cartesian, raDec2ECI


JD0 = 2460000.6
LAT0, LON0 = np.radians(45.0), np.radians(15.0)
STATIONS = [(44.6, 14.5), (45.5, 14.6), (44.8, 15.8)]

EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), \
    "Dynesty", "examples", "20191023_091225")
EXAMPLE_PICKLE = "20191023_091225_trajectory.pickle"


def _trueTrack(v0, drag_coeff, sigma, h0, zenith, duration, frag_ht=None, frag_kept=1.0, gravity=True):
    """ Inertial positions against time from the first point, integrated in 3D with RK4 in 1 ms steps: the drag and
        the ablation along the velocity, on the speed relative to the air along the path, which turns with the
        Earth, and the gravity of a point-mass Earth, so the meteoroid drops below its straight line and speeds up
        along it as a real one does. The drag along the velocity relative to the air would also bend the path
        sideways, by about 1% of the speed lost, which the solver's straight line does not model. The mass follows
        the ablation equation, and the density is tabulated over the heights the meteoroid reaches, as the drag fit
        does. If frag_ht is given, the meteoroid keeps the fraction frag_kept of its mass when it goes below that
        height. Without gravity, the path is the straight line. """

    p0 = np.array(geo2Cartesian(LAT0, LON0, h0, JD0))
    ra, dec = altAz2RADec(np.radians(90.0), np.pi/2 - zenith, JD0, LAT0, LON0)
    motion = -np.array(raDec2ECI(ra, dec))
    omega = np.array([0.0, 0.0, 2*np.pi/86164.09053])
    gm = 3.986004418e14 if gravity else 0.0

    def deriv(y, air_density):
        r, v, log_m = y[:3], y[3:6], y[6]
        speed = math.sqrt(v @ v)
        direction = v/speed
        u = speed - np.cross(omega, r) @ direction
        drag = drag_coeff*math.exp(-log_m/3)*air_density(cartesian2Geo(JD0, *r)[2])*u**2
        return np.concatenate([v, -drag*direction - gm*r/(r @ r)**1.5, [-sigma*u*drag]])

    # Integrated twice: the heights reached by the first integration set the density of the second
    ht_reached = [h0 - v0*duration*math.cos(zenith), h0]
    for _ in range(2):
        air_density = AtmDensityTableInterp(*getAtmDensityTable(LAT0, LON0, np.min(ht_reached) - 5000,
            np.max(ht_reached) + 5000, JD0))
        dt = 0.001
        y = np.concatenate([p0, v0*motion, [0.0]])
        positions, ht_reached = [p0], [h0]
        for _ in range(int(round(duration/dt))):
            k1 = deriv(y, air_density)
            k2 = deriv(y + k1*dt/2, air_density)
            k3 = deriv(y + k2*dt/2, air_density)
            k4 = deriv(y + k3*dt, air_density)
            y = y + dt*(k1 + 2*k2 + 2*k3 + k4)/6
            ht = cartesian2Geo(JD0, *y[:3])[2]
            if (frag_ht is not None) and (ht_reached[-1] >= frag_ht) and (ht < frag_ht):
                y[6] += math.log(frag_kept)
            positions.append(y[:3].copy())
            ht_reached.append(ht)

    return p0, motion, dt*np.arange(len(positions)), np.array(positions)


def _solve(v0, drag_coeff, sigma, h0, zenith_deg, v_init_drag, duration=2.0, seed=0, frag_ht=None, frag_kept=1.0,
        gravity=True, noise_arcsec=20.0, **solver_kwargs):
    """ Solve the synthetic observations of the meteoroid with WMPL, with the given solver options. Without
        gravity, the meteoroid has none and the solver applies no gravity correction. """

    _, _, times, positions = _trueTrack(v0, drag_coeff, sigma*1e-6, h0, np.radians(zenith_deg), duration,
        frag_ht=frag_ht, frag_kept=frag_kept, gravity=gravity)
    rng = np.random.default_rng(seed)
    noise = np.radians(noise_arcsec/3600)

    traj = Trajectory(JD0, meastype=1, monte_carlo=False, calc_orbit=False, show_plots=False, save_results=False,
        verbose=False, v_init_drag=v_init_drag, gravity_correction=gravity, **solver_kwargs)

    for k, (lat, lon) in enumerate(STATIONS):
        lat, lon = np.radians(lat), np.radians(lon)
        t_obs = np.arange(0.2*k, duration - 0.05, 0.04)
        ra_list, dec_list = [], []
        for t in t_obs:
            los = positions[int(round(t/(times[1] - times[0])))] - np.array(geo2Cartesian(lat, lon, 300.0,
                JD0 + t/86400))
            ra, dec = eci2RaDec(los/np.linalg.norm(los))
            ra_list.append(ra + rng.normal(0, noise)/np.cos(dec))
            dec_list.append(dec + rng.normal(0, noise))
        traj.infillTrajectory(np.array(ra_list), np.array(dec_list), t_obs, lat, lon, 300.0, \
            station_id="S{:d}".format(k))

    with contextlib.redirect_stdout(io.StringIO()):
        return traj.run()


def test_drag_fit_recovers_the_initial_velocity_of_a_decelerating_fireball():
    """ A fireball first seen at 60 km at 24 km/s (B = 5.3e-3 m^2/kg, about 1 kg of a 3500 kg/m^3 sphere, sigma =
        0.005 s^2/km^2) slows down to 11 km/s over the 2 s it is observed. Over three noise realizations, the
        straight line over the first part gives a velocity 560-615 m/s too low, while the drag fit to all points
        is 7-23 m/s above the true one, at most 2.3 times its uncertainty, and recovers the ablation coefficient
        within 7%. """

    traj_line = _solve(24000.0, 5.3e-3, 0.005, 60e3, 45.0, v_init_drag=False)
    traj_drag = _solve(24000.0, 5.3e-3, 0.005, 60e3, 45.0, v_init_drag=True, v_init_drag_time=np.inf)
    fit = traj_drag.v_init_drag_fit

    assert traj_line.v_init_drag_fit is None
    assert traj_line.v_init < 24000.0 - 300
    assert fit is not None and traj_drag.v_init == fit.v_init
    assert abs(fit.v_init - 24000.0) < 3*fit.v_init_stddev
    assert abs(fit.sigma/0.005 - 1) < 0.2
    assert fit.v_init_linear == traj_line.v_init


def test_drag_fit_needs_its_own_time_offsets():
    """ First seen at 45 km, the solver's time offsets absorb part of the deceleration: fitting all points with
        them, the velocity is 121 m/s too high, 3.7 times its uncertainty, and with its own 60 m/s, 2.4 times. """

    traj = _solve(24000.0, 5.3e-3, 0.005, 45e3, 45.0, v_init_drag=True, v_init_drag_time=np.inf)
    fit = traj.v_init_drag_fit

    traj_fixed = copy.deepcopy(traj)
    traj_fixed.max_toffset = 1e-9
    fit_fixed = fitDragInitialVelocity(traj_fixed, gravity=traj.los_gravity)

    assert abs(fit.v_init - 24000.0) < 3*fit.v_init_stddev
    assert abs(fit_fixed.v_init - 24000.0) > 3*fit_fixed.v_init_stddev


def test_drag_fit_keeps_the_time_offsets_given_to_the_solver():
    """ Time offsets given to the solver as fixed are kept in the fit instead of fitted again, as they do not come
        from its lag model, e.g. for a fragment of a meteor that takes the time offsets of the main one. The
        others are still fitted, and with all kept the velocity stays within its uncertainty of the true one. """

    traj = _solve(24000.0, 5.3e-3, 0.005, 45e3, 45.0, v_init_drag=True, v_init_drag_time=np.inf)
    assert traj.v_init_drag_fit.fixed_stations == []

    traj_fixed = _solve(24000.0, 5.3e-3, 0.005, 45e3, 45.0, v_init_drag=True, v_init_drag_time=np.inf, 
        fixed_times="S1:0.0")
    fit = traj_fixed.v_init_drag_fit

    assert fit.fixed_stations == ["S1"]
    assert (fit.time_offsets["S1"] == 0.0) and (fit.time_offsets["S2"] != 0.0)

    traj_all = _solve(24000.0, 5.3e-3, 0.005, 45e3, 45.0, v_init_drag=True, v_init_drag_time=np.inf, 
        fixed_times="S0:0.0,S1:0.0,S2:0.0")
    fit = traj_all.v_init_drag_fit

    assert sorted(fit.fixed_stations) == ["S0", "S1", "S2"]
    assert abs(fit.v_init - 24000.0) < 3*fit.v_init_stddev


def test_drag_fit_ends_before_a_fragmentation():
    """ The fireball first seen at 60 km keeps 20% of its mass at 45 km, 0.9 s later. The single-body fit to all
        points puts its velocity 288 m/s too high, 11 times its uncertainty; fitting the default first second, or
        the points above 46 km, gives it within its uncertainty. """

    def fit(**solver_kwargs):
        return _solve(24000.0, 5.3e-3, 0.005, 60e3, 45.0, v_init_drag=True, frag_ht=45e3, frag_kept=0.2,
            **solver_kwargs).v_init_drag_fit

    fit_all = fit(v_init_drag_time=np.inf)
    fit_time = fit()
    fit_ht = fit(v_init_drag_ht=46.0)

    assert abs(fit_all.v_init - 24000.0) > 3*fit_all.v_init_stddev
    assert fit_time.t_range[1] < 1.0 and fit_ht.ht_range[0] > 46e3
    for fit_part in (fit_time, fit_ht):
        assert abs(fit_part.v_init - 24000.0) < 3*fit_part.v_init_stddev


def test_drag_fit_of_a_meteoroid_that_ablates_until_it_stops_does_not_depend_on_metsim_step():
    """ With sigma = 0.05 s^2/km^2 the meteoroid first seen at 50 km stops within the 2 s it is observed, with a
        millionth of its mass. Against the fit extrapolated to a zero step, fits with MetSim steps of 5, 1 and 0.5 ms
        alone put the velocity 163, 32 and 16 m/s higher, while the extrapolations from 1 and from 0.5 ms agree
        within 0.02 m/s. The lengths are the true ones, because the solver's radiant for synthetic observations of
        this meteoroid came out 0.18 deg off. """

    p0, motion, times, positions = _trueTrack(24000.0, 5.3e-3, 0.05e-6, 50e3, np.radians(45.0), 2.0)
    rng = np.random.default_rng(0)

    observations = []
    for k in range(len(STATIONS)):
        t_obs = np.arange(0.2*k, 2.0 - 0.05, 0.04)
        points = positions[np.round(t_obs/(times[1] - times[0])).astype(int)]
        lengths = (points - p0) @ motion
        observations.append(SimpleNamespace(ignore_station=False, station_id="S{:d}".format(k), time_data=t_obs,
            state_vect_dist=lengths + rng.normal(0, 10.0, len(t_obs)), ignore_list=np.zeros(len(t_obs), dtype=int),
            model_ht=np.array([cartesian2Geo(JD0, *p)[2] for p in points])))

    all_t = np.concatenate([obs.time_data for obs in observations])
    all_l = np.concatenate([obs.state_vect_dist for obs in observations])
    v_lin, intercept = np.polyfit(all_t[all_t < 0.5], all_l[all_t < 0.5], 1)
    traj = SimpleNamespace(observations=observations, t_ref_station=0, jdt_ref=JD0, rbeg_lat=LAT0, rbeg_lon=LON0,
        state_vect_mini=p0, radiant_eci_mini=-motion, v_init=v_lin, velocity_fit=[v_lin, intercept], max_toffset=1.0)

    fit = fitDragInitialVelocity(traj)
    fit_half = fitDragInitialVelocity(traj, fine_dt=0.0005)

    assert abs(fit.v_init - 24000.0) < 3*fit.v_init_stddev
    assert abs(fit.sigma/0.05 - 1) < 0.1
    assert abs(fit.v_init - fit_half.v_init) < 0.1*fit.v_init_stddev


@contextlib.contextmanager
def _withoutMetSimGravity():
    """ Run MetSim without gravity: its surface gravity G0, and its gravity(lat, h) where it has one. """

    saved = {name: getattr(MetSimErosion, name) for name in ("G0", "gravity") if hasattr(MetSimErosion, name)}
    try:
        MetSimErosion.G0 = 0.0
        if "gravity" in saved:
            MetSimErosion.gravity = lambda lat, ht: 0.0
        yield

    finally:
        for name, value in saved.items():
            setattr(MetSimErosion, name, value)


def test_gravity_does_not_move_the_fitted_velocity():
    """ The solver fits the lines of sight with the gravity drop, and the drag fit takes the lengths where they meet
        the dropped trajectory, keeping the drop, which MetSim's gravity puts in along the path too. Fitting all
        points of a meteoroid at 17.5 km/s first seen at 92 km, observed for 3.5 s without noise, the velocity is
        then 0.27 m/s from that of the same meteoroid without gravity (in it, in the solver and in MetSim), 0.3 times
        its uncertainty. Taking the lengths on the line without the drop put it 9.0 m/s lower, and on the straight
        line, without finding them on the dropped trajectory, 2.6 m/s lower. """

    def solved(gravity):
        traj = _solve(17500.0, 9e-4, 0.005, 92e3, 45.0, v_init_drag=True, v_init_drag_time=np.inf, duration=3.5,
            noise_arcsec=0.0, gravity=gravity)
        assert (traj.v_init_drag_fit is not None) and (traj.los_gravity == gravity)
        return traj

    traj = solved(True)
    with _withoutMetSimGravity():
        traj_none = solved(False)

    assert abs(traj.v_init - traj_none.v_init) < 0.5


def test_drag_fit_keeps_the_velocity_of_a_meteor_that_does_not_decelerate():
    """ Without measurable deceleration (B = 1e-6 m^2/kg) the drag fit gives the straight-line velocity back, both
        within their uncertainties of the true one, and the solver takes it. """

    traj = _solve(30000.0, 1e-6, 0.005, 100e3, 45.0, v_init_drag=True)
    fit = traj.v_init_drag_fit

    assert fit is not None
    assert abs(fit.v_init - 30000.0) < max(3*fit.v_init_stddev, 15.0)
    assert abs(fit.v_init_linear - 30000.0) < 15.0
    assert traj.v_init == fit.v_init and traj.v_init_stddev == fit.v_init_stddev


def test_fitted_points_without_deceleration_keep_the_straight_line():
    """ Points that do not decelerate, here a meteor first seen at 50 km whose lengths are given a slight
        acceleration (e.g. from a scale error of a station), push B down until the model has no drag. The fit is
        then only a straight line over them, so it is not used, and the solver keeps its straight line over the
        first part, saying why in the report. """

    traj = _solve(20000.0, 1e-9, 0.005, 50e3, 45.0, v_init_drag=True)
    assert traj.v_init_drag_fit is not None

    for obs in traj.observations:
        obs.state_vect_dist = obs.state_vect_dist + 0.5*300.0*obs.time_data**2
    fit, reason = fitDragInitialVelocity(traj, return_reason=True)

    assert fit is None
    assert reason.startswith("the fitted points show no deceleration")

    example = loadPickle(EXAMPLE_DIR, EXAMPLE_PICKLE)
    example.v_init_drag, example.v_init_drag_fit, example.v_init_drag_rejection = True, None, reason
    report = example.saveReport(".", "unused.txt", verbose=False, save_results=False)
    assert "was not used: the fitted points show no deceleration" in report
    assert "so the initial velocity is the straight line's" in report


def test_without_the_drag_fit_the_solver_warns_when_the_straight_line_is_low():
    """ With the option off, a parabola over the straight line's points flags the fireball first seen at 60 km,
        whose straight line is 615 m/s low, as 985 +/- 139 m/s, and suggests the drag fit; the meteor without
        measurable deceleration is not flagged. """

    traj = _solve(24000.0, 5.3e-3, 0.005, 60e3, 45.0, v_init_drag=False)
    bias = traj.v_init_line_bias

    assert bias is not None and bias.significant
    assert 300 < bias.bias < 2000 and (24000.0 - traj.v_init) > 300
    assert "consider --vinitdrag" in traj._lineBiasWarning()

    bias = _solve(30000.0, 1e-6, 0.005, 100e3, 45.0, v_init_drag=False).v_init_line_bias
    assert bias is not None and not bias.significant


def test_drag_fit_notes_typical_fragmentation_pressures_and_erosion_energies():
    """ Along the fitted model, the fireball at 24 km/s first seen at 75 km goes from 0.019 to 0.205 MPa and crosses
        0.04 MPa, where ordinary chondritic fireballs start fragmenting, at 70 km, which the note gives as the
        height to fit above; the meteor at 30 km/s first seen at 115 km receives 0.2-6.8 MJ/m^2, across the
        energy at which shower meteoroids start eroding. A note is only given for what the fitted part crosses. """

    fit = _solve(24000.0, 5.3e-3, 0.005, 75e3, 45.0, v_init_drag=True).v_init_drag_fit
    notes = breakupNotes(fit)
    assert 0.01e6 < fit.dyn_pressure_range[0] < 0.04e6 < 0.12e6 < fit.dyn_pressure_range[1] < 0.5e6
    assert 68e3 < fit.first_fragmentation_ht < 72e3 and fit.energy_range[0] > 2e6
    assert len(notes) == 1 and "--vinitdraght {:.1f}".format(fit.first_fragmentation_ht/1000) in notes[0]

    fit = _solve(30000.0, 1e-6, 0.005, 115e3, 45.0, v_init_drag=True).v_init_drag_fit
    notes = breakupNotes(fit)
    assert fit.energy_range[0] < 1e6 and fit.energy_range[1] > 2e6 and fit.dyn_pressure_range[1] < 0.04e6
    assert len(notes) == 1 and "Buccongello" in notes[0]


def test_a_height_limit_alone_is_the_only_limit_and_both_end_at_the_first_reached():
    """ Without limits the fit takes the first second; with only a height limit it reaches that height, here
        40 km after 1.2 s, rather than stopping at the default second; with both it ends at whichever comes first.
        """

    def window(**kwargs):
        fit = _solve(24000.0, 5.3e-3, 0.005, 60e3, 45.0, v_init_drag=True, **kwargs).v_init_drag_fit
        return fit.t_range[1], fit.ht_range[0]

    assert fittedTimeLimit(None, None) == 1.0 and fittedTimeLimit(None, 40e3) is None
    assert fittedTimeLimit(np.inf, None) is None and fittedTimeLimit(0.5, 40e3) == 0.5

    t_end, _ = window()
    assert 0.9 < t_end < 1.0

    t_end, ht_end = window(v_init_drag_ht=40.0)
    assert t_end > 1.2 and 40e3 < ht_end < 41e3

    t_end, ht_end = window(v_init_drag_ht=40.0, v_init_drag_time=0.5)
    assert t_end < 0.5 and ht_end > 45e3


def test_drag_fit_records_the_atmosphere_model_it_used():
    """ The fit records the atmosphere its densities came from, read from the module that evaluates them:
        NRLMSISE-00 by default, or the MSIS version and date set by setAtmosphere() where the model can be chosen
        (pymsis), set here directly in that module, and the report lists it. """

    # The module the fit evaluates its densities with
    atm = atmosphereDescription.__globals__["fitAtmPoly"].__globals__
    saved = {key: atm[key] for key in ("MSIS_VERSION", "MSIS_JD") if key in atm}

    try:
        atm["MSIS_VERSION"], atm["MSIS_JD"] = "00", None
        assert atmosphereDescription() == "NRLMSISE-00, at the trajectory's time"

        atm["MSIS_VERSION"], atm["MSIS_JD"] = "2.1", 2460310.5
        assert atmosphereDescription() == "NRLMSIS 2.1, at 2024-01-01 00:00:00 UTC (--atmtime)"

    finally:
        for key in ("MSIS_VERSION", "MSIS_JD"):
            atm.pop(key, None)
        atm.update(saved)

    fit = _solve(30000.0, 1e-6, 0.005, 100e3, 45.0, v_init_drag=True).v_init_drag_fit
    assert fit.atmosphere == atmosphereDescription()

    example = loadPickle(EXAMPLE_DIR, EXAMPLE_PICKLE)
    example.v_init_drag, example.v_init_drag_fit = True, fit
    assert "  Atmosphere: " + fit.atmosphere in example.saveReport(".", "unused.txt", verbose=False,
        save_results=False)


def test_a_drag_fit_that_is_not_used_is_reported_with_its_reason():
    """ Asked for over the first 0.05 s, the drag fit has too few points: the solver keeps the straight line, says
        why, and the report says so; a trajectory without the option reports nothing about it. """

    traj = _solve(24000.0, 5.3e-3, 0.005, 60e3, 45.0, v_init_drag=True, v_init_drag_time=0.05)
    line = _solve(24000.0, 5.3e-3, 0.005, 60e3, 45.0, v_init_drag=False)

    assert traj.v_init_drag_fit is None and traj.v_init == line.v_init
    assert "points in the fitted part, not more than its" in traj.v_init_drag_rejection
    assert traj.v_init_drag_rejection in traj._dragFitRejectedText()

    example = loadPickle(EXAMPLE_DIR, EXAMPLE_PICKLE)
    assert "--vinitdrag" not in example.saveReport(".", "unused.txt", verbose=False, save_results=False)
    example.v_init_drag, example.v_init_drag_rejection = True, "the fit did not converge"
    report = example.saveReport(".", "unused.txt", verbose=False, save_results=False)
    assert "(--vinitdrag) was not used: the fit did not converge, so the initial velocity is the straight " \
        "line's." in report


def test_option_is_off_by_default_and_in_old_pickles():
    """ The solver does not fit the drag unless asked, and a trajectory pickled before the option existed loads with
        it off, so its report is unchanged. """

    assert Trajectory(JD0).v_init_drag is False

    traj = loadPickle(EXAMPLE_DIR, EXAMPLE_PICKLE)
    assert traj.v_init_drag is False and traj.v_init_drag_fit is None
    assert traj.v_init_line_bias is None
    assert traj.v_init_drag_rejection is None


if __name__ == "__main__":
    test_drag_fit_recovers_the_initial_velocity_of_a_decelerating_fireball()
    test_drag_fit_needs_its_own_time_offsets()
    test_drag_fit_keeps_the_time_offsets_given_to_the_solver()
    test_drag_fit_ends_before_a_fragmentation()
    test_drag_fit_of_a_meteoroid_that_ablates_until_it_stops_does_not_depend_on_metsim_step()
    test_gravity_does_not_move_the_fitted_velocity()
    test_drag_fit_keeps_the_velocity_of_a_meteor_that_does_not_decelerate()
    test_fitted_points_without_deceleration_keep_the_straight_line()
    test_without_the_drag_fit_the_solver_warns_when_the_straight_line_is_low()
    test_drag_fit_notes_typical_fragmentation_pressures_and_erosion_energies()
    test_a_height_limit_alone_is_the_only_limit_and_both_end_at_the_first_reached()
    test_drag_fit_records_the_atmosphere_model_it_used()
    test_a_drag_fit_that_is_not_used_is_reported_with_its_reason()
    test_option_is_off_by_default_and_in_old_pickles()
    print("All DragInitialVelocity checks passed.")
