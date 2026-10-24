""" Run MetSimErosion back up a solved trajectory from its reference point, to before it was observed, and give
    where it ended in the trajectory solver's ECI frame, ready for wmpl.Rebound.REBOUND.reboundSimulate().

    state_vect = np.r_[traj.state_vect_mini, traj.v_init*traj.radiant_eci_mini]
    jd, state_vects, masses = backwardStates(traj.jdt_ref, [state_vect] + realizations, m_init)

From the command line, to a height or for a time, with or without Monte Carlo realizations:

    python -m wmpl.MetSim.BackwardAtmIntegration traj.pickle --mass 0.5 --back_height 180 --mc 100 --atm 2.1

State vectors are [x, y, z, vx, vy, vz] in ECI (true equator and equinox of date), in m and m/s, with the
velocity pointing to the radiant, as the solver gives them and reboundSimulate() takes them.

The run follows a single body, whose mass grows back as dm = sigma m v dv. MetSim erodes the body with the same law
as it ablates it, so its erosion is undone by giving the effective coefficient sigma + eta: only the eroded grains,
which are not followed, cannot be undone. The coefficient is the effective one between the starting point and where
the run ends, which nothing observed measures. An apparent coefficient fitted to the meteoroid's deceleration (e.g.
by DynamicMassFit, AlphaBeta or the trajectory solver's --vinitdrag over all its points) includes its erosion, but
over the observed part, lower down: where the erosion or fragmentation grows with the dynamic pressure, it is larger
there than above, so taking it overestimates the mass growth. The same holds for Ceplecha et al. (1998)'s 0.014,
0.042, 0.10 and 0.21 s^2/km^2 for fireball types I, II, IIIA and IIIB, apparent coefficients over whole trajectories
including their gross fragmentation; single bright meteors range from 0.001 to 0.19 s^2/km^2 (Silber et al. 2015).
It matters when the run starts deep. Run back to 180 km, the speed there relative to a run with a frozen mass, and
the mass there over the starting one, were, for sigma = 0.023 and sigma + eta = 0.123 and 0.323 s^2/km^2:

| Start | Speed change (m/s) | Mass ratio |
|---|---|---|
| 1 g at 30 km/s from 100 km | 0.0, 0.0, -0.1 | 1.01, 1.03, 1.09 |
| 1 kg at 20 km/s from 70 km | -0.4, -2.3, -5.7 | 1.04, 1.2, 1.6 |
| 1 kg at 15 km/s from 60 km, 60 deg from the zenith | -8, -37, -81 | 1.14, 1.9, 4.2 |
| 1 kg at 20 km/s from 50 km | -129, -455, -730 | 1.8, 9.8, 62 |

What matters most is whether the body was already eroding just above the first point. With eta = 0.3 s^2/km^2 only
up to some height above the start, and sigma = 0.023 above it, the share of the speed change that eroding all the
way up gives, from the run without erosion, and the mass at 180 km over the starting one were:

| Eroding up to | 15 km/s from 60 km | 20 km/s from 50 km |
|---|---|---|
| 0.5 km above | 18%, 1.27 | 26%, 2.6 |
| 1 km above | 32%, 1.40 | 44%, 3.6 |
| 2 km above | 53%, 1.65 | 66%, 6.1 |
| 5 km above | 84%, 2.37 | 90%, 15.9 |
| 10 km above | 97%, 3.23 | 98%, 33.7 |
| 180 km | 100% (-73 m/s), 4.15 | 100% (-601 m/s), 61.9 |

The speed is set within the first few km, while the mass keeps growing with any erosion higher up.

Against a truth run forwards from 180 km (chondritic fireballs of 10 kg at 15, 20 and 30 km/s, intrinsic sigma =
0.005 s^2/km^2, eta = 0.1-0.3 s^2/km^2 starting where the dynamic pressure reaches 0.04 MPa or the received energy 1
MJ/m^2, the same atmosphere in every run, reference points at 50-70 km), running back with the right sigma + eta all
the way to 180 km left the speed within 3 m/s and the mass within a factor 1.3, over the truth when the erosion had
started at 0.04 MPa; going back to the intrinsic sigma above either onset gave the speed within 4 m/s and the mass
within a factor 0.73-1.3. A height above which the body stops eroding is therefore not modelled: the thin air there
takes little speed. What sets the result is sigma + eta over the first few km above the reference point. Run back
with the intrinsic sigma or MetSim's 0.023 instead, the speed came out up to 2 m/s off from 70 km, 73 m/s from 60 km
and 660 m/s from 50 km; for a body that did not erode, MetSim's 0.023 put it 7-24 m/s low from 50 km. A
fragmentation above the reference point cannot be undone: with half the mass lost at 0.04 or 0.12 MPa and the right
coefficient, the mass at 180 km came out half the true one and the speed 5-17 or 17-58 m/s high.

So the command lines report the dynamic pressure and the energy received at the reference point (referenceLoading),
warn when the pressure is over the 0.04 MPa of the first fragmentation of chondritic fireballs
(FIRST_FRAGMENTATION_PRESSURE), and report the nominal run with the coefficients of Ceplecha's types I and IIIB, as
a warning when the two speeds differ by more than the uncertainty of the initial velocity. When the coefficient is
uncertain, --ablation_coeff_sigma draws one for each Monte Carlo realization.
"""

import argparse
import copy
import math
import os

import numpy as np

from wmpl.MetSim.MetSimErosion import Constants, EARTH_ROTATION_RATE, runSimulation
from wmpl.Utils import AtmosphereDensity
from wmpl.Utils.AtmosphereDensity import addAtmosphereArguments, fitAtmPoly, getAtmDensity, getAtmDensityTable, \
    getMSISVersion
from wmpl.Utils.TrajConversions import cartesian2Geo, derotatedRadiantAltAz, enu2ECEF, jd2Date, jd2LST


def _eciToEcef(jd):
    """ Rotation matrix from ECI (true equator and equinox of date, as the trajectory solver uses) to ECEF, by the
        Greenwich apparent sidereal time about the rotation axis. """

    gst = np.radians(jd2LST(jd, 0.0)[1])

    return np.array([[np.cos(gst), np.sin(gst), 0.0], [-np.sin(gst), np.cos(gst), 0.0], [0.0, 0.0, 1.0]])


def _startFrom(const, jd_ref, state_vect):
    """ Set the start of a MetSim run in 3D to the given state vector, and return its latitude and longitude. """

    lat, lon, ht = cartesian2Geo(jd_ref, *state_vect[:3])

    # The solver's velocity is in ECI, so it includes the Earth's rotation, while MetSim follows the motion relative
    #   to the ground
    azim, elev, v_norot = derotatedRadiantAltAz(state_vect[3:], state_vect[:3], jd_ref, lat, lon)

    const.v_init, const.zenith_angle, const.radiant_azimuth = v_norot, np.pi/2 - elev, azim
    const.h_init, const.latitude = ht, lat

    return lat, lon


def backwardConstants(jd_ref, state_vect, m_init, h_kill=180000.0, const=None):
    """ Constants for a backward MetSimErosion run, in 3D with gravity and the Coriolis acceleration, from the given
        state vector up to h_kill.

    Arguments:
        jd_ref: [float] Julian date of the state vector.
        state_vect: [ndarray] State vector to start from (see the module docstring), e.g. the solver's at its
            reference point.
        m_init: [float] Mass at the start (kg), e.g. the photometric mass.

    Keyword arguments:
        h_kill: [float] Height to stop at (m), 180 km by default. The atmosphere density is tabulated (and fitted)
            up to it, so to stop after a given time with t_kill instead, keep h_kill above the height that time
            reaches. To run in another atmosphere, replace the table (atm_table_ht, atm_table_log10_rho), which is
            used instead of dens_co.
        const: [Constants] Physical parameters to start from (rho, sigma, gamma, shape_factor, dt, freeze_mass...),
            e.g. from a MetSim fit. It is copied, not changed. Constants() by default.

    Return:
        const: [Constants]
    """

    const = Constants() if const is None else copy.deepcopy(const)

    lat, lon = _startFrom(const, jd_ref, state_vect)

    const.m_init = m_init
    const.gravity_3d = True
    const.dt, const.h_kill = -abs(const.dt), h_kill
    const.erosion_on = const.disruption_on = const.fragmentation_on = False
    const.dens_co = fitAtmPoly(lat, lon, const.h_init, h_kill, jd_ref)

    # Tabulate the density too, the run uses the table instead of the polynomial. A run back from a deep start
    #   spans the mesopause, where a single polynomial misses the MSIS model by tens of percent
    const.atm_table_ht, const.atm_table_log10_rho = getAtmDensityTable(lat, lon, const.h_init, h_kill, jd_ref)

    return const


def backwardState(jd_ref, state_vect, frag, t):
    """ Where a run with the constants from backwardConstants() ended, in the trajectory solver's frame.

    Arguments:
        jd_ref: [float] Julian date the run started from.
        state_vect: [ndarray] State vector the run started from.
        frag: [Fragment] The fragment runSimulation() returned.
        t: [float] Time of its last step (s), negative: the first column of the last row of runSimulation()'s
            results.

    Return:
        (jd, state_vect): Julian date and state vector (see the module docstring) where the run ended.
    """

    lat, lon, _ = cartesian2Geo(jd_ref, *state_vect[:3])

    # MetSim follows the fragment in the east-north-up frame of the start, fixed to the ground
    pos_ecef = _eciToEcef(jd_ref).dot(state_vect[:3]) + np.array(enu2ECEF(lat, lon, frag.px, frag.py, frag.pz))
    vel_ecef = np.array(enu2ECEF(lat, lon, frag.vx, frag.vy, frag.vz))

    jd = jd_ref + t/86400.0
    ecef_to_eci = _eciToEcef(jd).T
    pos = ecef_to_eci.dot(pos_ecef)
    vel = ecef_to_eci.dot(vel_ecef) + np.cross([0.0, 0.0, EARTH_ROTATION_RATE], pos)

    return jd, np.concatenate([pos, -vel])


def backwardStates(jd_ref, state_vects, m_init, h_kill=180000.0, t_kill=-1, const=None, sigmas=None):
    """ Run several state vectors back through the atmosphere to a common epoch, so the nominal solution and its
        Monte Carlo realizations can go into reboundSimulate() together. The first state vector, the nominal one,
        is run up to h_kill, or for t_kill seconds if it is given and h_kill is not reached first, and the others
        for the same time.

    Arguments:
        jd_ref: [float] Julian date of the state vectors.
        state_vects: [list] State vectors (see the module docstring), the nominal one first.
        m_init: [float or list] Mass at the start (kg), for all of them or for each one.

    Keyword arguments:
        h_kill, const: As in backwardConstants().
        t_kill: [float] Time to run back for (s). -1, the default, runs back to h_kill.
        sigmas: [list] Ablation coefficient of each state vector (s^2/m^2, MetSim's units). None by default, for
            const.sigma for all of them.

    Return:
        (jd, state_vects, masses): The common Julian date, and the state vectors and masses (kg) there, in the same
            order.
    """

    m_inits = [m_init]*len(state_vects) if np.ndim(m_init) == 0 else m_init

    const = backwardConstants(jd_ref, state_vects[0], m_inits[0], h_kill=h_kill, const=const)
    if sigmas is None:
        sigmas = [const.sigma]*len(state_vects)
    const.sigma = sigmas[0]
    const.t_kill = t_kill
    frag, results, _ = runSimulation(const)
    t = results[-1][0]
    jd, state_vect = backwardState(jd_ref, state_vects[0], frag, t)
    states, masses = [state_vect], [frag.m]

    # The realizations stop by time alone, after as many steps as the nominal run, since the time adds up the same
    #   steps, and keep its atmosphere fit, made at practically the same place
    const.h_kill, const.t_kill = np.inf, abs(t)

    for sv, m, sigma in zip(state_vects[1:], m_inits[1:], sigmas[1:]):

        const_mc = copy.deepcopy(const)
        const_mc.m_init, const_mc.sigma = m, sigma
        _startFrom(const_mc, jd_ref, sv)
        frag, results, _ = runSimulation(const_mc)

        if results[-1][0] != t:
            raise RuntimeError("A realization ended at t = {:.6f} s instead of the nominal {:.6f} s.".format(
                results[-1][0], t))

        states.append(backwardState(jd_ref, sv, frag, t)[1])
        masses.append(frag.m)

    return jd, states, masses


def atmosphereDescription():
    """ The MSIS model the densities of a run back through the atmosphere come from, as chosen with --atm and
        --atmtime (see wmpl.Utils.AtmosphereDensity.setAtmosphere()). """

    version = getMSISVersion()
    name = "NRLMSISE-00" if version == "00" else "NRLMSIS " + version

    if AtmosphereDensity.MSIS_JD is None:
        return name + ", at the trajectory's time"

    return name + ", at " + jd2Date(AtmosphereDensity.MSIS_JD, dt_obj=True).strftime("%Y-%m-%d %H:%M:%S") \
        + " UTC (--atmtime)"


def addBackwardArguments(arg_parser):
    """ Add the command-line arguments for the mass and the physical parameters of the meteoroid in a run back
        through the atmosphere, shared by this module's command line and REBOUND's, and those of the atmosphere
        model (--atm and --atmtime), which setAtmosphere() applies after parsing. """

    arg_parser.add_argument("--mass", type=float, default=None,
        help="Mass of the meteoroid at the trajectory's reference point in kg, e.g. a photometric mass. In REBOUND "
        "it also sets the radiation pressure (beta) with --density.")

    arg_parser.add_argument("--mass_sigma", type=float, default=0.0,
        help="1-sigma uncertainty of --mass in kg. Each Monte Carlo realization starts with a mass drawn from a "
        "log-normal distribution with --mass as its mean and this standard deviation, which keeps every mass "
        "positive. Default: 0.")

    arg_parser.add_argument("--freeze_mass", action="store_true",
        help="Keep the mass constant instead of growing it back as the ablation is undone.")

    arg_parser.add_argument("--ablation_coeff", type=float, default=Constants().sigma*1e6,
        help="Effective ablation coefficient in s^2/km^2: the mass lost for the kinetic energy lost to the drag "
        "(dm = sigma m v dv), so it sets how fast the mass grows back. It includes the erosion of the body "
        "(sigma + eta) above the starting point. The speed at the end is set by its value over the first few km "
        "above the starting point, so the best is one fitted at the start of the observed part (e.g. by "
        "DynamicMassFit or the trajectory solver's --vinitdrag); farther up it matters little for the speed and "
        "likely overestimates the mass growth. Ceplecha's 0.014, 0.042, 0.10 and 0.21 for fireball types I, II, "
        "IIIA and IIIB are apparent coefficients over whole trajectories, and 0.005 is the intrinsic one of "
        "chondritic fireballs (Borovicka et al. 2020). Default: MetSim's, {:g}.".format(Constants().sigma*1e6))

    arg_parser.add_argument("--ablation_coeff_sigma", type=float, default=0.0,
        help="1-sigma uncertainty of --ablation_coeff in s^2/km^2. Each Monte Carlo realization runs with a "
        "coefficient drawn from a log-normal distribution with --ablation_coeff as its mean and this standard "
        "deviation. Default: 0.")

    arg_parser.add_argument("--density", type=float, default=3000.0,
        help="Bulk density of the meteoroid in kg/m^3, which with the mass sets the drag, and in REBOUND the "
        "radiation pressure with --radius or --mass. Default: 3000.")

    arg_parser.add_argument("-g", "--ga", metavar="GAMMA_A", type=float,
        default=Constants().gamma*Constants().shape_factor,
        help="The product of the drag coefficient Gamma and the shape coefficient A. Default: MetSim's, "
        "{:g}.".format(Constants().gamma*Constants().shape_factor))

    # The MSIS model of the air density
    addAtmosphereArguments(arg_parser)


def checkBackwardArguments(arg_parser, args):
    """ Stop with a command-line error for a mass a run back through the atmosphere cannot start from. """

    if (args.mass is None) or (args.mass <= 0):
        arg_parser.error("the meteoroid's --mass is required, and must be positive.")

    if args.mass_sigma < 0:
        arg_parser.error("--mass_sigma cannot be negative.")

    if args.ablation_coeff < 0:
        arg_parser.error("--ablation_coeff cannot be negative.")

    if args.ablation_coeff_sigma < 0:
        arg_parser.error("--ablation_coeff_sigma cannot be negative.")

    if (args.ablation_coeff_sigma > 0) and (args.ablation_coeff == 0):
        arg_parser.error("--ablation_coeff_sigma needs a positive --ablation_coeff, the mean it is drawn around.")


# Apparent ablation coefficients of Ceplecha's fireball types I and IIIB (s^2/km^2), the range the command lines
#   report the nominal run for when the coefficient is not drawn
CEPLECHA_SIGMA_RANGE = (0.014, 0.21)

# Dynamic pressure (Pa) at which ordinary chondritic fireballs typically fragment a first time, 0.04-0.12 MPa
#   (Borovicka et al. 2020): below it, a fragmentation above the starting point cannot be excluded
FIRST_FRAGMENTATION_PRESSURE = 0.04e6


def referenceLoading(jd_ref, state_vect, n_top=200):
    """ The dynamic pressure on a meteoroid at the given state vector, and the energy per unit cross section it
        received above it.

    The energy, E = int rho_air v^3/2 dt, is taken with the speed at the state vector along a straight path up to
    180 km, rho_air v^2/2/cos(z) per unit height; the deceleration above makes it slightly low.

    Arguments:
        jd_ref: [float] Julian date of the state vector.
        state_vect: [ndarray] State vector (see the module docstring).

    Keyword arguments:
        n_top: [int] Number of heights the energy is integrated over.

    Return:
        (pressure, energy): rho_air v^2 (Pa) and the energy received (J/m^2), with the speed relative to the ground.
    """

    const = Constants()
    lat, lon = _startFrom(const, jd_ref, state_vect)

    hts = np.linspace(const.h_init, 180e3, n_top)
    rho = np.array([getAtmDensity(lat, lon, ht, jd_ref) for ht in hts])
    column = np.sum(np.diff(hts)*(rho[1:] + rho[:-1])/2)/math.cos(const.zenith_angle)

    return rho[0]*const.v_init**2, const.v_init**2/2*column


def _logNormal(rng, mean, stddev, n):
    """ n draws from a log-normal distribution with the given mean and standard deviation, all positive. """

    sigma_ln = np.sqrt(np.log(1 + (stddev/mean)**2))

    return list(mean*np.exp(sigma_ln*rng.normal(size=n) - sigma_ln**2/2))


def dragFitCoefficientNote(drag_fit, ablation_coeff):
    """ Text on the effective ablation coefficient a trajectory's drag fit of the initial velocity found
        (wmpl.Utils.DragInitialVelocity), as a value for --ablation_coeff and --ablation_coeff_sigma when it is
        constrained. It is not used by itself: over a short fit it is often poorly constrained.

    Arguments:
        drag_fit: [DragVelocityFit] The trajectory's v_init_drag_fit, with sigma and sigma_stddev in s^2/km^2,
            t_range in s and ht_range in m.
        ablation_coeff: [float] The --ablation_coeff of this run (s^2/km^2).

    Return:
        [str]
    """

    text = ("The trajectory's initial velocity comes from its drag fit (--vinitdrag) over {:.2f} to {:.2f} s, "
        "{:.1f} to {:.1f} km, which found an effective ablation coefficient of {:.4f} +/- {:.4f} s^2/km^2 at the "
        "start of the observed part, while this run uses {:g}.").format(drag_fit.t_range[0], drag_fit.t_range[1],
        drag_fit.ht_range[1]/1000, drag_fit.ht_range[0]/1000, drag_fit.sigma, drag_fit.sigma_stddev,
        ablation_coeff)

    if (drag_fit.sigma > 0) and (drag_fit.sigma_stddev < drag_fit.sigma):
        return text + " It can be given as --ablation_coeff {:.4f} --ablation_coeff_sigma {:.4f}.".format(
            drag_fit.sigma, drag_fit.sigma_stddev)

    return text + " Its uncertainty is as large as the value, so the fit does not constrain it."


def backwardStatesFromArguments(traj, state_vects, args, h_kill, t_kill=-1, random_seed=None):
    """ backwardStates() from the trajectory's reference point, with the mass and physical parameters given by the
        command-line arguments of addBackwardArguments(). The masses and ablation coefficients of the
        realizations are drawn with random_seed from --mass and --mass_sigma, and from --ablation_coeff and
        --ablation_coeff_sigma. Also returns the starting masses and the ablation coefficients (s^2/km^2).

    If the trajectory was solved with the drag fit of the initial velocity, it prints the effective ablation
    coefficient that fit found (dragFitCoefficientNote). It prints the dynamic pressure and the energy received at
    the reference point, with a warning when the
    pressure is over FIRST_FRAGMENTATION_PRESSURE, below which a fragmentation above the first point, which this
    single-body run cannot undo, cannot be excluded. If the coefficient is not drawn and the mass is not frozen, it
    also prints the speed and mass of the nominal run with the apparent coefficients of Ceplecha's fireball types I
    and IIIB, as a warning when their speeds differ by more than the uncertainty of the initial velocity.
    """

    if (args.mass_sigma > 0) and (len(state_vects) == 1):
        print("--mass_sigma has no effect without Monte Carlo realizations.")

    if (args.ablation_coeff_sigma > 0) and (len(state_vects) == 1):
        print("--ablation_coeff_sigma has no effect without Monte Carlo realizations.")

    const = Constants()
    const.freeze_mass = args.freeze_mass
    const.sigma = args.ablation_coeff/1e6
    const.rho = args.density

    # MetSim's drag only takes the product gamma*shape_factor, so the shape factor keeps its default
    const.gamma = args.ga/const.shape_factor

    # Log-normal with mean args.mass and standard deviation args.mass_sigma, from a generator of its own, so the
    #   state vector draws stay those of sampleStateVectors with the same seed
    rng = np.random.default_rng(None if (random_seed is None) else [1, random_seed])
    m_inits = [args.mass] + _logNormal(rng, args.mass, args.mass_sigma, len(state_vects) - 1)

    # The ablation coefficients likewise, from another generator, so the masses stay the same with or without them
    n_mc = len(state_vects) - 1
    if args.ablation_coeff_sigma > 0:
        rng = np.random.default_rng(None if (random_seed is None) else [2, random_seed])
        sigmas = [args.ablation_coeff] + _logNormal(rng, args.ablation_coeff, args.ablation_coeff_sigma, n_mc)
    else:
        sigmas = [args.ablation_coeff]*len(state_vects)

    result = backwardStates(traj.jdt_ref, state_vects, m_inits, h_kill=h_kill, t_kill=t_kill, const=const,
        sigmas=[sigma/1e6 for sigma in sigmas])

    # A trajectory solved with the drag fit of the initial velocity (--vinitdrag) carries the effective ablation
    #   coefficient at the start of the observed part, the one that sets the speed of this run
    drag_fit = getattr(traj, "v_init_drag_fit", None)
    if drag_fit is not None:
        print(dragFitCoefficientNote(drag_fit, args.ablation_coeff))

    # How deep the reference point is
    pressure, energy = referenceLoading(traj.jdt_ref, state_vects[0])
    print("At the reference point: dynamic pressure {:.4f} MPa, energy received {:.3g} MJ/m^2.".format(
        pressure/1e6, energy/1e6))
    if pressure >= FIRST_FRAGMENTATION_PRESSURE:
        print("WARNING: The dynamic pressure at the reference point is over the {:g} MPa at which ordinary "
            "chondritic fireballs typically fragment a first time (0.04-0.12 MPa, Borovicka et al. 2020). A "
            "fragmentation above the first point cannot be undone by this single-body run: the mass at the end is "
            "then short by the mass that was lost, a lower limit, and the speed comes out high (5-58 m/s in "
            "synthetic tests with half the mass lost at 0.04-0.12 MPa). From this depth the ablation coefficient "
            "also dominates the result: give one fitted at the start of the observed part, with "
            "--ablation_coeff_sigma.".format(FIRST_FRAGMENTATION_PRESSURE/1e6))

    if (args.ablation_coeff_sigma == 0) and (not args.freeze_mass):
        line, speeds = [], []
        for sigma in CEPLECHA_SIGMA_RANGE:
            const_type = copy.deepcopy(const)
            const_type.sigma = sigma/1e6
            _, states_type, masses_type = backwardStates(traj.jdt_ref, state_vects[:1], m_inits[0], h_kill=h_kill,
                t_kill=t_kill, const=const_type)
            speeds.append(np.linalg.norm(states_type[0][3:]))
            line.append("{:.2f} m/s and {:.6g} kg with {:g}".format(speeds[-1], masses_type[0], sigma))

        # The Monte Carlo uncertainty of the initial velocity if there is one, else the solver's formal one
        v_init_stddev = getattr(getattr(traj, "uncertainties", None), "v_init", None)
        if v_init_stddev is None:
            v_init_stddev = getattr(traj, "v_init_stddev", None)
        spread = abs(speeds[1] - speeds[0])
        significant = (v_init_stddev is not None) and (spread > v_init_stddev)

        print(("{:s}The nominal run ends at {:.2f} m/s with {:.6g} kg with --ablation_coeff {:g} s^2/km^2, and at "
            "{:s} (Ceplecha's fireball types I and IIIB){:s}. If the erosion above the first point is uncertain, "
            "give --ablation_coeff_sigma.").format("WARNING: " if significant else "",
            np.linalg.norm(result[1][0][3:]), result[2][0], args.ablation_coeff, " and at ".join(line),
            ": {:.2f} m/s apart, more than the {:.2f} m/s uncertainty of the initial velocity".format(spread,
            v_init_stddev) if significant else ""))

    return result, m_inits, sigmas


if __name__ == "__main__":

    from wmpl.Rebound.REBOUND import sampleStateVectors
    from wmpl.Utils.AtmosphereDensity import setAtmosphere
    from wmpl.Utils.Pickling import loadPickle

    arg_parser = argparse.ArgumentParser(description="Run a trajectory, and optionally its Monte Carlo "
        "realizations, back up through the atmosphere from its reference point with MetSim (single body, drag, "
        "gravity, Coriolis), to a height or for a time. Saves where each one ended next to the pickle.")

    arg_parser.add_argument("pickle_path", type=str, help="Path to the trajectory pickle file.")

    arg_parser.add_argument("--back_height", type=float, default=180.0,
        help="Height in km to run back to. Default: 180.")

    arg_parser.add_argument("--back_time", type=float, default=-1,
        help="Run back for this many seconds instead, unless --back_height is reached first.")

    arg_parser.add_argument("--mc", type=int, default=1,
        help="Number of Monte Carlo realizations drawn from the trajectory's state vector covariance, run back for "
        "as long as the nominal solution. Default: 1, the nominal solution only.")

    arg_parser.add_argument("--seed", type=int, default=None, help="Seed for the Monte Carlo realizations.")

    addBackwardArguments(arg_parser)

    args = arg_parser.parse_args()

    checkBackwardArguments(arg_parser, args)
    setAtmosphere(args)

    # As in REBOUND's command line, draw a seed if none was given and report it, so the run can be reproduced
    random_seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy % (2**32))

    traj = loadPickle(*os.path.split(args.pickle_path))
    state_vect = np.concatenate([traj.state_vect_mini, traj.v_init*traj.radiant_eci_mini])
    state_vects = [state_vect] + sampleStateVectors(traj, args.mc, random_seed)

    (jd, states, masses), m_inits, sigmas = backwardStatesFromArguments(traj, state_vects, args,
        1000*args.back_height, t_kill=args.back_time, random_seed=random_seed)

    rows = []
    for i, (sv, m_ref, m, sigma) in enumerate(zip(states, m_inits, masses, sigmas)):
        lat, lon, ht = cartesian2Geo(jd, *sv[:3])
        rows.append([i, m_ref, np.degrees(lat), np.degrees(lon), ht, np.linalg.norm(sv[3:]), m] + list(sv)
            + [sigma])
    rows = np.array(rows)

    print("Atmosphere: {:s}".format(atmosphereDescription()))
    print("Mass at the reference point: {:.6g} kg{:s}".format(m_inits[0], ", frozen" if args.freeze_mass else ""))
    print("Ran {:d} state vector(s) back {:.4f} s, to JD {:.8f}".format(len(states), (traj.jdt_ref - jd)*86400,
        jd))
    print("Nominal: lat {:.5f} deg, lon {:.5f} deg, height {:.1f} m, speed {:.2f} m/s, mass {:.6g} kg".format(
        *rows[0, 2:7]))
    if len(rows) > 1:
        print("Monte Carlo seed: {:d}".format(random_seed))
        for name, col, unit in [("mass at the reference point", 1, "kg"), ("ablation coefficient", 13, "s^2/km^2"),
                ("height", 4, "m"), ("speed", 5, "m/s"), ("mass", 6, "kg")]:
            print("Realizations {:s}: 2.5/50/97.5 percentiles {:s} {:s}".format(name,
                " / ".join("{:.6g}".format(v) for v in np.percentile(rows[1:, col], [2.5, 50, 97.5])), unit))

    out_path = os.path.splitext(args.pickle_path)[0] + "_backward_atm.txt"
    np.savetxt(out_path, rows, fmt=["%d"] + ["%.10g"]*13, header="JD {:.10f} (UTC), {:.6f} s from the reference "
        "point. Row 0 is the nominal solution, the others its realizations. State vectors in ECI, true equator and "
        "equinox of date, velocity to the radiant. Atmosphere: {:s}.\nrow, mass at the reference point (kg), lat (deg), lon (deg), "
        "height MSL (m), speed (m/s), mass (kg), x (m), y (m), z (m), vx (m/s), vy (m/s), vz (m/s), ablation "
        "coefficient (s^2/km^2)".format(jd,
        (jd - traj.jdt_ref)*86400, atmosphereDescription()))
    print("Saved:", out_path)
