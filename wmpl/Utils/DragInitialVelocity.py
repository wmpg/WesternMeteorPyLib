""" Initial velocity of a decelerating meteor from a fit of a single-body drag and ablation model to the lengths
    along the first part of its trajectory.

The trajectory solver estimates the initial velocity as the slope of a straight line fitted to the first part
of the meteor (at least the first 25% of the points). A meteor that already decelerates there is slower on
average over that part than at its first point, so the slope underestimates the velocity at the reference
point. For a fireball first seen at 45-60 km this can be hundreds of m/s, comparable to or larger than the
deceleration in the atmosphere above the first point, and it carries into the orbit.

Here the lengths of the points within a time from the first one (1 s by default), and optionally above a height,
are fitted with a MetSim single body, in 3D with gravity and the Coriolis acceleration:
dv/dt = -B rho(h) v^2 (m0/m)^(1/3) plus gravity, where rho is the atmosphere density and the mass ablates at the
rate set by sigma, as m/m0 = exp(sigma (v^2 - v0^2)/2) without gravity. The fitted parameters are the velocity at
t = 0, sigma and B = Gamma A rho_m^(-2/3) m0^(-1/3), so neither the mass nor the bulk density are needed: MetSim's
drag only takes that combination, and the mass it is given follows from B. The measured deceleration constrains B
and sigma, and with them the velocity at the first point. The time offsets of the stations other than the
reference one are fitted again, as the solver's come from its lag model, not from this one; they are only used for
this fit. On the test fireball first seen at 45 km, fitting all points, the velocity was 4-24 m/s off with its own
offsets and 4-33 m/s with the solver's over three noise realizations: no worse, not clearly better. The offsets
the solver was given as fixed are kept, as they do not come from its lag model.

The lengths are taken where the lines of sight meet the trajectory the solver fitted them to: with the gravity drop
when it applied one (droppedLengths), keeping the drop, since MetSim's gravity also speeds the meteoroid up along
its path. The solver may give the lengths of the line without the drop instead, which is what its velocity and lag
describe, and fitting MetSim to those counts the gravity twice along the path. On a synthetic meteoroid at 17.5 km/s
first seen at 92 km at 45 deg from the zenith, without noise, that put the velocity 8.4 m/s low fitting all 3.5 s
of points and 1.1 m/s low fitting 1 s. The lengths on the straight line itself, where the drop shifts the closest
point of each line of sight by an amount that depends on the station's view, put it 4.2 and 2.3 m/s low. Taken on
the dropped trajectory, the velocity is within 0.2 m/s of that of the same meteoroid without gravity.

With the option on, the solver takes the fitted velocity whenever the fit converges, fits better than the
straight line and measures a deceleration. If the fitted points show none, B goes down until the model has no
drag, and the fit is only a straight line over these points, which over a short part, or one seen mostly by one
station, can differ from the solver's straight line over the first part: the solver then keeps the latter. Keeping the straight line when the fit does not measure its bias (when the two velocities are within
sqrt(2) of the fit's uncertainty) was tried and dropped: it rests on the straight line being the more precise, but
the solver's straight-line uncertainty is formal: through the solver, on the synthetic meteoroids below, where that
rule kept the straight line its errors were 5-6 times that uncertainty in RMS. On 26 synthetic meteoroids (12-65
km/s, B from 1e-6 to 1.5e-2 m^2/kg, sigma 0.005-0.03 s^2/km^2, first seen at 55-115 km, two noise realizations each,
52 runs), the RMS velocity error was 195 and 257 m/s for the straight line, 15 and 28 m/s with the option on (18
and 29 m/s where the drag fit was used, in all but the 8 and 10 runs that measured no deceleration), and 19 and 31
m/s with that rule (the first 10 meteoroids, then 16 others); no threshold for it did consistently better than
none. Even where it kept the straight line, the drag fit's RMS error was smaller (22 against 24 and 25 against 34
m/s), and its own uncertainty was calibrated or somewhat large (RMS of the error over it 0.47 and 0.95). With the
option off, estimateLineBias() fits a parabola to the straight line's points and the solver warns when it puts the
velocity at the first point more than 2 sigma above the straight line's. On the 16 meteoroids not used to set that
threshold (32 runs), it warned for 7 of the 11 straight lines over 100 m/s low, none of the 3 that were 50-100 m/s
low, and 1 of the 16 within 50 m/s (that one 2 m/s low); its precision falls with the speed: it did not warn for
either of the two straight lines 62 and 187 m/s low above 35 km/s. Neither check sees a fragmentation or a wake
(below).

In the solver's Monte Carlo the fit is redone for each realization, so the spread of the initial velocity, and of
the orbit, includes it. For the fireball at 24 km/s first seen at 60 km, with 30 realizations (geometric
uncertainties, 8 cores), every realization used the fit, each took 2.0 s against 1.5 s without it, and their
velocities were 0.4 m/s off on average with a spread of 29 m/s, against a formal uncertainty of the nominal fit of
34 m/s; without the option the realizations were 614 m/s low with a spread of 10 m/s, so the Monte Carlo uncertainty
of the straight line does not include its bias.

MetSim erodes mass with the same law as it ablates it, so the erosion of the main body is absorbed by the fitted
sigma, which is then sigma + eta; the eroded grains are not followed. On the true lengths of the main body of the
MetSim erosion model, integrated with 0.1 ms steps, with 10 m of noise and ten noise realizations (20 km/s and 1 kg
first seen at 70 km, 30 km/s and 0.1 kg first seen at 80 km), erosion with eta = 0.1-0.3 s^2/km^2 from above the
first point or starting 3-8 km below it, or changing from 0.05 to 0.5 s^2/km^2 5 km below it, so active over most of
the fitted points, did not bias the velocity: over 170 distinct fits of 1 s or of all points, its error over its
uncertainty had a mean of -0.05, and it differed from the error of the same meteoroid without erosion, over the same
points and noise, by +0.28 of the uncertainty on average and by at most 3.0. Strong erosion makes the uncertainty
somewhat too small: the RMS of the error over the uncertainty was 1.18, and 1.23 with eta = 0.3, against 1.00
without erosion over the same points, with 88% of the fits within 2 sigma against 94%. The fitted sigma was sigma +
eta when the points cover enough of the deceleration, e.g. 0.105 and 0.30 s^2/km^2 for eta = 0.1 and 0.3 at 20 km/s
fitting all points, but 0.18-0.23 for 0.31 over the shorter spans at 30 km/s. Over 1 s sigma is often poorly
constrained: without erosion it ended at one of its bounds in 9-10 of 20 fits, while the velocity uncertainty stayed
calibrated (RMS of the error over it 0.91-1.01), so sigma does not flag the problems below. All these tests use
MetSim both as the truth and as the model: they show the fit absorbs MetSim's erosion of the body, not that a real
meteoroid erodes that way.

A fragmentation is not described, and the fit does not detect it: its RMS stays below the straight line's. Erosion
that starts within the fitted part, after a stretch without it, acts alike. In synthetic tests (20 and 30 km/s,
first seen at 55 and 70 km), the main body losing 50-80% of its mass 10-20 km below the first point biased the
velocity by 40-480 m/s, 5-29 times its uncertainty. With erosion of eta = 0.1 s^2/km^2 starting 10 km below the
first point (truth integrated with 0.1 ms steps, ten noise realizations), fitting all points biased it by -357 m/s,
20 times its uncertainty, for 20 km/s and 1 kg first seen at 55 km and fitted down to 40 km, and by 16 m/s below the
same fit without erosion, 3 times its uncertainty, first seen at 70 km and fitted down to 46 km. The 1 s fits, which
ended within 4.5 km below the start of the erosion, had a mean error over their uncertainty of -0.1 to -0.8 in all
eight cases (eta = 0.1 and 0.3, 20 and 30 km/s, first seen at 55 and 70 km). Fitting only the points above a
fragmentation, all were within their uncertainty. Hence the fitted part ends at a time or a height, which should be
before the first fragmentation, e.g. from the light curve. Without fragmentation, on the true lengths of 35
synthetic meteoroids, the uncertainty of the velocity had a median of 97, 33 and 9 m/s fitting 0.5 s, 1 s and all
points.

The fit therefore reports, along its model over the fitted points, the dynamic pressure rho_air v^2 and the energy
received per unit cross section from the top of the atmosphere, E = int rho_air v^3/2 dt, and notes (breakupNotes)
when the fitted part crosses the 0.04-0.12 or 0.5-5 MPa of the first and second fragmentation phases of ordinary
chondritic fireballs (Borovicka et al. 2020), with the height where it reaches 0.04 MPa, or the 1-2 MJ/m^2 at which
the erosion of cometary shower meteoroids observed by CAMO begins (Buccongello et al. 2024). These only point to the
light curve: Geminids begin to crumble at 1-100 kPa (Henych et al. 2024), and the strength of a given meteoroid can
be far from them. In the test cases, a fireball at 24 km/s first seen at 75 km crosses 0.04 MPa at 70 km, one first
seen at 60 km is already at 0.17 MPa and reaches 0.9 MPa, and a meteor at 30 km/s first seen at 115 km receives
0.2-5.8 MJ/m^2. Through NRLMSISE-00 at 45 deg from the zenith, 1-2 MJ/m^2 are received by 90-94 km at 12 km/s,
96-100 km at 20 km/s, 100-105 km at 30 km/s and 111-117 km at 70 km/s, close to the erosion onset heights CAMO fits
find. So for a meteoroid first seen below about 90 km, the energy criterion puts any erosion before the first point,
active over the whole fitted part, which the fit absorbs; a fragmentation triggered by pressure is the likelier
break within it.

The lengths must follow the body. The solver measures the centroid of the light, which with erosion includes the
wake of the grains slowing down behind the body as they ablate. The fit takes the change of that lag for
deceleration, and neither its uncertainty nor its RMS show it. How large the bias is depends on how the centroid is
measured and on the grains, so the numbers here are from one synthetic model of the measurement, and give its order
and that it can have either sign, not the bias of a given camera: the luminosity-weighted centroid of the body and
of all the grains within 1 km behind it, with no point spread function, saturation or detection threshold, a
constant luminous efficiency, grain masses chosen for the test, and the same two meteoroids integrated with 0.5 ms
steps. Fitting 1 s, eta = 0.1 s^2/km^2 at 30 km/s moved the velocity by about 25 m/s (1 sigma) from the same fit
without erosion with grains of 1e-9 to 1e-7 kg, while the straight line was 465 m/s low, but by 230-280 m/s with
grains of 1e-7 to 1e-5 and 1e-6 to 1e-4 kg; at 20 km/s with grains of 1e-6 to 1e-4 kg the velocity was 400 m/s low,
21-26 times its uncertainty. With the larger grains the errors ranged from -470 to +185 m/s, of either sign
depending on where the erosion started, and the fit was sometimes further off than the straight line. For a grain
that keeps its speed, the lag when it is consumed goes as m^(1/3)/(sigma^2 rho_air v^4), so the wake is longest for
large grains, thin air and slow meteors. If the meteor shows a significant wake within the point spread function,
the fitted velocity should not be trusted; data that follow the leading fragment, as high-resolution tracking does,
are not affected.

Two checks can reveal a wake, neither reliably. In the same model, with two stations whose centroids take 100 m and
1000 m of the wake (a stand-in for different ranges or plate scales), fits of 1 s and of all points and three noise
realizations: fitting each station alone and comparing their velocities flagged (over 3 sigma) 15 of the 31 biased
events and none of the 17 unbiased ones; the mass implied by B, (Gamma A rho_m^(-2/3)/B)^3, outside 0.1-10 times the
true mass flagged 16 of the 31 and 2 of the 17; either, 25 of the 31. The misses were fits of all points 28-110 m/s
off. These rates are optimistic: the 0.1-10 band was chosen after seeing the fits, and the implied mass used the
true Gamma A and bulk density, which enters it squared. Over 1 s, B is poorly constrained even without a wake
(implied mass 0.3-8 times the true one), so against a photometric mass this check only flags gross cases.

Fitting all the points of 50 synthetic meteoroids without fragmentation (15-40 km/s, B from 1.1e-3 to 2.5e-2
m^2/kg, sigma from 0.005 to 0.05 s^2/km^2, first seen at 50 and 70 km, observed while they keep 1e-3 of their mass
and 40% of their speed, with NRLMSISE-00 as the atmosphere) the velocity error over its formal uncertainty had an
RMS of 1.02 and stayed within 2.7. The fitted velocity depends on the shape of the density profile with height,
not on its scale, which B absorbs: in those cases a density 6% higher at 40 km than NRLMSISE-00, growing linearly
from 60 km, moved the velocity by up to 42 m/s, beyond 3 times its uncertainty in 6 of them. MetSim takes the density
tabulated over the observed heights, within 0.5% of the MSIS model (the 6th degree polynomial it took before was
within 2.3% of NRLMSISE-00). The fit also inherits the errors of the solver's lengths: through the solver, the
synthetic fireballs of the tests come out from 6 m/s low to 43 m/s high fitting all their points, within 2.2 times
their uncertainty.
"""

import math

import numpy as np
import scipy.optimize

from wmpl.MetSim.BackwardAtmIntegration import backwardConstants
from wmpl.MetSim.MetSimErosion import atmDensity, runSimulation
from wmpl.Utils.AtmosphereDensity import fitAtmPoly, getAtmDensity, getAtmDensityTable
from wmpl.Utils.Math import findClosestPoints, vectMag
from wmpl.Utils.TrajConversions import jd2Date


# Default time limit of the fitted points from the reference time (s), when no limit is given in time or height
DEFAULT_TIME_LIMIT = 1.0


def fittedTimeLimit(time_limit, ht_limit):
    """ The time limit of the drag fit given the user's limits (see Trajectory's v_init_drag_time): the given one,
        DEFAULT_TIME_LIMIT if neither limit is given, and none (None) if only the height limit is given, so the
        fit then reaches as far down as that height. inf also means no time limit. """

    if time_limit is None:
        return DEFAULT_TIME_LIMIT if (ht_limit is None) else None

    return None if math.isinf(time_limit) else time_limit

# A curvature of the lengths the straight line was fitted to that puts the velocity at the first point this many
#   uncertainties above the straight line's is reported (see the module docstring for how it did)
LINE_BIAS_SIGMA = 2.0

# Dynamic pressures (Pa) of the first and second fragmentation phases of ordinary chondritic fireballs (Borovicka
#   et al. 2020, ApJ 160, 101), and energies received per unit cross section (J/m^2) at which the erosion of
#   cometary shower meteoroids observed by CAMO begins (Buccongello et al. 2024, Icarus 410, 115907). They are
#   indicative: Geminids begin to crumble at 1-100 kPa (Henych et al. 2024), and other populations differ
FIRST_FRAGMENTATION_PRESSURE = (0.04e6, 0.12e6)
SECOND_FRAGMENTATION_PRESSURE = (0.5e6, 5e6)
EROSION_ONSET_ENERGY = (1e6, 2e6)

class DragVelocityFit(object):
    def __init__(self, v_init, v_init_stddev, intercept, drag_coeff, sigma, sigma_stddev, time_offsets, rms,
            v_init_linear, rms_linear, n_points, ht_range, t_range):
        """ Result of fitDragInitialVelocity().

        Arguments:
            v_init: [float] Velocity at the reference time, t = 0 (m/s).
            v_init_stddev: [float] Its formal 1-sigma uncertainty (m/s).
            intercept: [float] Length at t = 0 (m).
            drag_coeff: [float] B = Gamma A rho_m^(-2/3) m0^(-1/3) at t = 0 (m^2/kg).
            sigma: [float] Ablation coefficient (s^2/km^2).
            sigma_stddev: [float] Its formal 1-sigma uncertainty (s^2/km^2). Large when the ablation does not
                change the deceleration enough to be measured, which is when it does not matter for v_init, and
                infinite when it does not change the model at all.
            time_offsets: [dict] Time offsets added to each station's times for this fit (s), by station ID, 0
                for the stations in fixed_stations.
            rms: [float] RMS of the length residuals (m).
            v_init_linear: [float] The solver's straight-line initial velocity (m/s).
            rms_linear: [float] RMS of the length residuals of that straight line (m).
            n_points: [int] Number of fitted points.
            ht_range: [tuple] Lowest and highest height of the fitted points (m).
            t_range: [tuple] Earliest and latest time of the fitted points from the reference time (s).
        """

        self.v_init = v_init
        self.v_init_stddev = v_init_stddev
        self.intercept = intercept
        self.drag_coeff = drag_coeff
        self.sigma = sigma
        self.sigma_stddev = sigma_stddev
        self.time_offsets = time_offsets
        self.rms = rms
        self.v_init_linear = v_init_linear
        self.rms_linear = rms_linear
        self.n_points = n_points
        self.ht_range = ht_range
        self.t_range = t_range

        # Stations whose time offsets were kept instead of fitted, as they were given to the solver as fixed
        self.fixed_stations = []

        # Along the fitted model over the fitted points (see breakupNotes): the range of the dynamic pressure
        #   rho_air v^2 (Pa) and of the energy received per unit cross section from the top of the atmosphere
        #   (J/m^2), and the height where the pressure reaches the first fragmentation phase, if it does there (m)
        self.dyn_pressure_range = None
        self.energy_range = None
        self.first_fragmentation_ht = None

        # The atmosphere model its densities come from (see atmosphereDescription)
        self.atmosphere = atmosphereDescription()


class LineBias(object):
    def __init__(self, bias, bias_stddev, n_points, t_range):
        """ Result of estimateLineBias().

        Arguments:
            bias: [float] Velocity at the reference time from a parabola minus the straight line's (m/s).
            bias_stddev: [float] Its formal 1-sigma uncertainty (m/s).
            n_points: [int] Number of points of the straight-line fit.
            t_range: [tuple] Earliest and latest time of those points from the reference time (s).
        """

        self.bias = bias
        self.bias_stddev = bias_stddev
        self.n_points = n_points
        self.t_range = t_range

        # The straight line measurably underestimates the initial velocity
        self.significant = bias > LINE_BIAS_SIGMA*bias_stddev


def estimateLineBias(traj):
    """ Estimate how much the solver's straight line underestimates the initial velocity, from a parabola fitted
        to the same points.

    A straight line fitted to a meteor decelerating at a over a time T has the velocity at its middle, a*T/2 below
    the one at its start. The parabola measures the deceleration over those points, and its velocity at the
    reference time is compared with the straight line's. It is cheap and needs no model, so the solver gives it
    when the drag fit is off, and reports it when it is significant (see LINE_BIAS_SIGMA). It is a rough estimate:
    through the solver, where it warned for synthetic meteoroids whose straight line was 110-960 m/s low, it gave
    0.8-2.0 times that.

    Arguments:
        traj: [Trajectory] Solved trajectory, with velocity_fit and velocity_fit_t_range.

    Return:
        [LineBias] or None if the straight line's points are unknown or too few.
    """

    if (traj.velocity_fit is None) or (getattr(traj, "velocity_fit_t_range", None) is None):
        return None

    t_lo, t_hi = traj.velocity_fit_t_range
    times, lengths = [], []
    for obs in traj.observations:
        if obs.ignore_station:
            continue
        good = (obs.ignore_list == 0) & (obs.time_data >= t_lo) & (obs.time_data <= t_hi)
        times.append(obs.time_data[good])
        lengths.append(obs.state_vect_dist[good])
    times, lengths = np.concatenate(times), np.concatenate(lengths)
    if len(times) <= 4:
        return None

    # Length = intercept + v*t - a*t^2/2, with the formal uncertainties scaled by the residual variance
    design = np.column_stack([np.ones_like(times), times, -times**2/2])
    coeffs = np.linalg.lstsq(design, lengths, rcond=None)[0]
    res = lengths - design.dot(coeffs)
    cov = np.linalg.pinv(design.T.dot(design))*np.sum(res**2)/(len(times) - 3)

    return LineBias(coeffs[1] - traj.velocity_fit[0], math.sqrt(abs(cov[1, 1])), len(times), (t_lo, t_hi))


def atmosphereDescription():
    """ The atmosphere model the fit's densities come from, as the module that evaluates them is set: the MSIS
        version and date chosen with --atm and --atmtime (wmpl.Utils.AtmosphereDensity.setAtmosphere), or
        NRLMSISE-00 where the version cannot be chosen. It is read from the globals of fitAtmPoly itself, so it is
        the model actually used, in this process and in each Monte Carlo one. """

    atm = fitAtmPoly.__globals__
    version = atm.get("MSIS_VERSION")
    name = "NRLMSISE-00" if version in (None, "00") else "NRLMSIS " + version

    jd = atm.get("MSIS_JD")
    if jd is None:
        return name + ", at the trajectory's time"

    return name + ", at " + jd2Date(jd, dt_obj=True).strftime("%Y-%m-%d %H:%M:%S") + " UTC (--atmtime)"


def _metsimLengths(const, v0, drag_coeff, sigma, t_lo, t_hi, v_rotation, profile=False):
    """ Run MetSim from t = 0 forwards to t_hi, and backwards to t_lo if it is negative, and return the times and
        the lengths along the path from t = 0 in the solver's inertial frame, and with profile also the heights (m)
        and the speeds relative to the ground (m/s).

    MetSim follows the motion relative to the ground, whose speed along the path differs from the inertial one by
    the Earth's rotation, v_rotation, practically constant over the few seconds of a meteor: MetSim starts at
    v0 - v_rotation, and v_rotation*t is added back to its lengths.
    """

    # MetSim's drag is gamma*shape_factor*rho^(-2/3)*m^(-1/3), so the drag coefficient sets the mass
    const.v_init, const.sigma = v0 - v_rotation, sigma
    const.m_init = (const.gamma*const.shape_factor*const.rho**(-2/3.0)/drag_coeff)**3

    times, lengths, heights, speeds = [np.zeros(1)], [np.zeros(1)], [np.array([const.h_init])], \
        [np.array([const.v_init])]
    for sign, t_kill in [(1, t_hi), (-1, -t_lo)]:
        if t_kill <= 0:
            continue
        const.dt, const.t_kill = sign*abs(const.dt), t_kill
        const.h_kill = 1.0 if (sign > 0) else const.h_init + 1e5
        results = np.array(runSimulation(const)[1])
        times.append(results[:, 0])
        lengths.append(results[:, 18])
        heights.append(results[:, 17])
        speeds.append(results[:, 19])

    times, lengths = np.concatenate(times), np.concatenate(lengths)
    order = np.argsort(times)

    if profile:
        return times[order], lengths[order] + v_rotation*times[order], np.concatenate(heights)[order], \
            np.concatenate(speeds)[order]

    return times[order], lengths[order] + v_rotation*times[order]


def _breakupProfile(fit, const, traj, v_rotation, n_top=200):
    """ Fill the dynamic pressure and received energy ranges of a fit (see DragVelocityFit) from its model.

    The energy received per unit cross section above the first fitted point, E = int rho v^3/2 dt, is taken with
    the speed there along a straight path through the MSIS model up to 180 km, rho v^2/2/cos(z) per unit height;
    the deceleration above the first point makes it slightly low.
    """

    t_lo, t_hi = fit.t_range
    times, _, heights, speeds = _metsimLengths(const, fit.v_init, fit.drag_coeff, fit.sigma*1e-6, t_lo, t_hi,
        v_rotation, profile=True)
    inside = (times >= t_lo) & (times <= t_hi)
    times, heights, speeds = times[inside], heights[inside], speeds[inside]
    rho = np.array([atmDensity(h, const) for h in heights])
    pressure = rho*speeds**2

    # Received above the first point, then along the fitted model
    hts_top = np.linspace(heights[0], 180e3, n_top)
    rho_top = np.array([getAtmDensity(traj.rbeg_lat, traj.rbeg_lon, h, traj.jdt_ref) for h in hts_top])
    e_top = speeds[0]**2/2*np.sum(np.diff(hts_top)*(rho_top[1:] + rho_top[:-1])/2)/math.cos(const.zenith_angle)
    energy = e_top + np.concatenate([[0], np.cumsum(np.diff(times)*(rho*speeds**3/2)[1:])])

    fit.dyn_pressure_range = (float(np.min(pressure)), float(np.max(pressure)))
    fit.energy_range = (float(energy[0]), float(energy[-1]))
    crossed = np.nonzero(pressure >= FIRST_FRAGMENTATION_PRESSURE[0])[0]
    if len(crossed) and (crossed[0] > 0):
        fit.first_fragmentation_ht = float(heights[crossed[0]])


def breakupNotes(fit):
    """ Notes on where the fitted part reaches the dynamic pressures at which fireballs typically fragment, or the
        received energy at which the erosion of shower meteoroids typically begins, either of which would bias the
        fit if it happens within the fitted part (see the module docstring). They are indicative and only point to
        the light curve: the strength of a given meteoroid can be far from these values.

    Arguments:
        fit: [DragVelocityFit]

    Return:
        [list] Strings, empty if none applies or the ranges were not computed.
    """

    notes = []

    if fit.dyn_pressure_range is not None:
        p_lo, p_hi = fit.dyn_pressure_range
        for (b_lo, b_hi), phase in [(FIRST_FRAGMENTATION_PRESSURE, "first"), (SECOND_FRAGMENTATION_PRESSURE,
                "second")]:
            if (p_lo <= b_hi) and (p_hi >= b_lo):
                note = ("The fitted part spans dynamic pressures of {:.3f}-{:.3f} MPa, across the {:g}-{:g} MPa of "
                    "the {:s} fragmentation of ordinary chondritic fireballs (Borovicka et al. 2020).").format(
                    p_lo/1e6, p_hi/1e6, b_lo/1e6, b_hi/1e6, phase)
                if (phase == "first") and (fit.first_fragmentation_ht is not None):
                    note += " The pressure reaches {:g} MPa at {:.1f} km: if the light curve flares below, fit " \
                        "above it (--vinitdraght {:.1f}).".format(b_lo/1e6, fit.first_fragmentation_ht/1000,
                        fit.first_fragmentation_ht/1000)
                else:
                    note += " If the light curve flares within the fitted part, end the fit before it."
                notes.append(note)

    if fit.energy_range is not None:
        e_lo, e_hi = fit.energy_range
        b_lo, b_hi = EROSION_ONSET_ENERGY
        if (e_lo <= b_hi) and (e_hi >= b_lo):
            notes.append(("The meteoroid receives {:.2f}-{:.2f} MJ/m^2 over the fitted part, across the {:g}-{:g} "
                "MJ/m^2 at which the erosion of shower meteoroids typically begins (Buccongello et al. 2024): if "
                "the light curve rises there, the erosion starts within the fit.").format(e_lo/1e6, e_hi/1e6,
                b_lo/1e6, b_hi/1e6))

    return notes


def droppedLengths(traj, obs):
    """ Lengths of the points of an observation from the solver's state vector along the radiant line, measured on
        the trajectory with the gravity drop the line of sight fit applied, where angleSumMeasurements2Line finds
        them, and keeping the drop: the motion MetSim follows, with gravity along the path too. The solver's own
        lengths may be those of the line without the drop.

    Arguments:
        traj: [Trajectory] Solved trajectory whose lines of sight were fitted with the gravity drop.
        obs: [ObservedPoints] One of its observations.

    Return:
        [ndarray] Lengths (m), one per point.
    """

    # Imported here, as the trajectory solver imports this module
    from wmpl.Trajectory.Trajectory import applyGravityDrop

    # Same reference time and vertical velocity as the line of sight fit
    t0 = min([o.time_data[0] for o in traj.observations])
    v0z = traj.v0z if traj.v0z is not None else 0.0

    state_vect, radiant_eci = traj.state_vect_mini, traj.radiant_eci_mini
    lengths = []
    for t, stat, meas in zip(obs.time_data, obs.stat_eci_los, obs.meas_eci_los):
        _, rad_cpa, _ = findClosestPoints(stat, meas, state_vect, radiant_eci)
        rad_cpa_grav = applyGravityDrop(rad_cpa, t - t0, vectMag(rad_cpa), traj.gravity_factor, v0z)
        _, rad_cpa_dropped, _ = findClosestPoints(stat, meas, rad_cpa_grav, radiant_eci)
        lengths.append(abs(np.dot(state_vect - rad_cpa_dropped, radiant_eci)))

    return np.array(lengths)


def fitDragInitialVelocity(traj, ht_min=None, t_max=None, fine_dt=0.001, return_reason=False, gravity=False):
    """ Fit the single-body drag and ablation model to the lengths of the non-ignored points of a solved
        trajectory above a height and before a time (see the module docstring).

    Arguments:
        traj: [Trajectory] Solved trajectory, with time_data, state_vect_dist and model_ht of its observations.

    Keyword arguments:
        ht_min: [float] Only points above this height are fitted (m). None by default, for no limit.
        t_max: [float] Only points before this time from the reference time, the first point in a solved
            trajectory, are fitted (s). None by default, for no limit.
        fine_dt: [float] The larger of the two MetSim time steps whose fits are extrapolated to a zero step (s).
            The starting fit uses MetSim's default step.
        return_reason: [bool] Also return why the fit is None, so it can be reported. False by default.
        gravity: [bool] The lines of sight were fitted with the gravity drop, so the lengths are measured on the
            dropped trajectory, keeping the drop (see droppedLengths), instead of taken from state_vect_dist. False
            by default.

    Return:
        [DragVelocityFit] or None if there are not more points than parameters, or the fit did not converge,
            does not fit better than the straight line or measures no deceleration. With return_reason, (fit,
            reason), the reason being None when the fit is returned.
    """

    def rejected(reason):
        return (None, reason) if return_reason else None

    ref_id = traj.observations[traj.t_ref_station].station_id

    observations, times, lengths, heights, station_index = [], [], [], [], []
    for obs in traj.observations:
        good = (obs.ignore_list == 0) & (obs.model_ht > (-np.inf if ht_min is None else ht_min)) \
            & (obs.time_data < (np.inf if t_max is None else t_max))
        if obs.ignore_station or (not np.any(good)):
            continue
        station_index.append(np.full(np.count_nonzero(good), len(observations)))
        observations.append(obs)
        times.append(obs.time_data[good])
        lengths.append((droppedLengths(traj, obs) if gravity else obs.state_vect_dist)[good])
        heights.append(obs.model_ht[good])
    if not observations:
        return rejected("no points in the fitted part")
    times, lengths, heights = np.concatenate(times), np.concatenate(lengths), np.concatenate(heights)
    station_index = np.concatenate(station_index)

    # The time offsets the solver was given as fixed are kept, as they do not come from its lag model
    kept = set(str(station_id) for station_id in getattr(traj, 'fixed_time_offsets', {}))

    # The stations whose time offset is fitted: all but the kept ones, or without them in the fitted part, all but
    #   the reference one, or but the first one if the reference station has no points in the fitted part either,
    #   since a common offset is the intercept
    offset_stations = [i for i, obs in enumerate(observations) if str(obs.station_id) not in kept]
    if len(offset_stations) == len(observations):
        offset_stations = [i for i in offset_stations if observations[i].station_id != ref_id]
    if len(offset_stations) == len(observations):
        offset_stations = offset_stations[1:]
    if len(times) <= 4 + len(offset_stations):
        return rejected("{:d} points in the fitted part, not more than its {:d} parameters".format(len(times),
            4 + len(offset_stations)))

    # MetSim starts from the solver's state vector, relative to the ground, with the air density over the heights
    #   the meteor reaches. The bulk density only scales the mass that B gives, so its value does not matter
    ht_lo, ht_hi = np.min(heights) - 5000.0, np.max(heights) + 5000.0
    const = backwardConstants(traj.jdt_ref, np.concatenate([traj.state_vect_mini, traj.v_init*traj.radiant_eci_mini]),
        1.0, h_kill=ht_hi)
    const.dens_co = fitAtmPoly(traj.rbeg_lat, traj.rbeg_lon, ht_lo, ht_hi, traj.jdt_ref)

    # The tabulated density over the same heights, which MetSim uses instead of the polynomial. It replaces the 
    #   one from backwardConstants(), which only starts at the reference point
    const.atm_table_ht, const.atm_table_log10_rho = getAtmDensityTable(traj.rbeg_lat, traj.rbeg_lon, ht_lo, ht_hi,
        traj.jdt_ref)
    const.v_kill = 100.0
    v_rotation = traj.v_init - const.v_init

    v_lin, intercept_lin = traj.velocity_fit
    res_lin = lengths - (v_lin*times + intercept_lin)
    rms_linear = np.sqrt(np.mean(res_lin**2))

    def residuals(params):
        v0, log_b, intercept, sigma = params[:4]
        offsets = np.zeros(len(observations))
        offsets[offset_stations] = params[-len(offset_stations):] if offset_stations else []
        t_model = times + offsets[station_index]
        try:
            t_grid, l_grid = _metsimLengths(const, v0, math.exp(log_b), sigma*1e-6, np.min(t_model), np.max(t_model),
                v_rotation)
        except (ValueError, OverflowError, ZeroDivisionError):
            return np.full_like(lengths, 1e9)
        res = np.interp(t_model, t_grid, l_grid) + intercept - lengths
        return res if np.all(np.isfinite(res)) else np.full_like(res, 1e9)

    n_off = len(offset_stations)
    # The solver's own offsets can be off by as much as it allows them, so they can be corrected as much
    lb = [0.5*v_lin, -25.0, intercept_lin - 1e4, 0.0] + [-traj.max_toffset]*n_off
    ub = [1.5*v_lin, 3.0, intercept_lin + 1e4, 0.5] + [traj.max_toffset]*n_off
    x_scale = [100.0, 1.0, 10.0, 0.005] + [1e-3]*n_off
    # The straight line's residuals are dominated by the deceleration, so this robust scale is loose and only
    #   downweights gross outliers
    f_scale = max(1.4826*np.median(np.abs(res_lin - np.median(res_lin))), 1.0)

    def fit(p0):
        return scipy.optimize.least_squares(residuals, p0, bounds=(lb, ub), x_scale=x_scale, loss="soft_l1",
            f_scale=f_scale)

    # MetSim advances the speed and the mass one after the other, so its lengths are first order in the time
    #   step, which biases the velocity of a strongly ablating meteoroid high. The fit starts with MetSim's default
    #   step, then fits with fine_dt and half of it are extrapolated to a zero step
    start = fit([v_lin, math.log(1e-3), intercept_lin, 0.01] + [0.0]*n_off)
    const.dt = fine_dt
    full = fit(start.x)
    const.dt = fine_dt/2
    best = fit(full.x)
    params = np.clip(2*best.x - full.x, lb, ub)

    # The straight line is the model without drag, so a worse fit than it means the fit went wrong
    rms = np.sqrt(np.mean(best.fun**2))
    if (best.status <= 0) or (not np.all(np.isfinite(params))):
        return rejected("the fit did not converge")
    if rms > rms_linear:
        return rejected("it fits worse than the straight line (RMS {:.2f} m against {:.2f} m)".format(rms,
            rms_linear))

    # Formal uncertainties from the Jacobian, scaled by the residual variance
    dof = max(len(lengths) - len(params), 1)
    cov = np.linalg.pinv(best.jac.T.dot(best.jac))*np.sum(best.fun**2)/dof
    stddev = np.sqrt(np.abs(np.diag(cov)))

    # When the fitted points show no deceleration, B goes down until MetSim gives no drag at all, so that an
    #   e-fold change of B moves the lengths by less than a millimetre. The fit is then only a straight line over
    #   these points, and the solver's straight line over the first part is kept
    if np.max(np.abs(best.jac[:, 1])) < 1e-3:
        return rejected(("the fitted points show no deceleration (B went down to {:.1e} m^2/kg, where the model has "
            "no drag): it is only a straight line over these {:d} points from {:d} station(s), at {:.0f} +/- "
            "{:.0f} m/s").format(math.exp(params[1]), len(times), len(observations), params[0], stddev[0]))

    # An ablation coefficient that does not change the model lengths is not constrained, although its formal
    #   uncertainty is then 0
    sigma_stddev = np.inf if (np.max(np.abs(best.jac[:, 3]))*(ub[3] - lb[3]) < 1e-3) else stddev[3]

    time_offsets = {obs.station_id: 0.0 for obs in observations}
    for k, i in enumerate(offset_stations):
        time_offsets[observations[i].station_id] = params[4 + k]

    fit_result = DragVelocityFit(params[0], stddev[0], params[2], math.exp(params[1]), params[3], sigma_stddev,
        time_offsets, rms, v_lin, rms_linear, len(times), (np.min(heights), np.max(heights)),
        (np.min(times), np.max(times)))
    fit_result.fixed_stations = [str(obs.station_id) for obs in observations if str(obs.station_id) in kept]

    # Where the fitted part stands against typical fragmentation pressures and erosion onset energies
    try:
        _breakupProfile(fit_result, const, traj, v_rotation)
    except (ValueError, OverflowError, ZeroDivisionError):
        pass

    return (fit_result, None) if return_reason else fit_result
