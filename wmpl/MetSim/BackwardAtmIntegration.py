""" Run MetSimErosion back up a solved trajectory from its reference point, to before it was observed, and give
    where it ended in the trajectory solver's ECI frame, ready for wmpl.Rebound.REBOUND.reboundSimulate().

    state_vect = np.r_[traj.state_vect_mini, traj.v_init*traj.radiant_eci_mini]
    jd, state_vects, masses = backwardStates(traj.jdt_ref, [state_vect] + realizations, m_init)

From the command line, to a height or for a time, with or without Monte Carlo realizations:

    python -m wmpl.MetSim.BackwardAtmIntegration traj.pickle --atm_height 180 --mc 100

State vectors are [x, y, z, vx, vy, vz] in ECI (true equator and equinox of date), in m and m/s, with the
velocity pointing to the radiant, as the solver gives them and reboundSimulate() takes them.
"""

import argparse
import copy
import os

import numpy as np

from wmpl.MetSim.MetSimErosion import Constants, EARTH_ROTATION_RATE, runSimulation
from wmpl.Trajectory.AggregateAndPlot import computeMass
from wmpl.Utils.AtmosphereDensity import fitAtmPoly
from wmpl.Utils.Physics import LUM_EFF_MODELS, PANCHROMATIC_LUM_EFF_TYPES
from wmpl.Utils.TrajConversions import cartesian2Geo, derotatedRadiantAltAz, enu2ECEF, jd2LST


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
        h_kill: [float] Height to stop at (m), 180 km by default. The atmosphere density is fitted up to it, so to
            stop after a given time with t_kill instead, keep h_kill above the height that time reaches.
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


def backwardStates(jd_ref, state_vects, m_init, h_kill=180000.0, t_kill=-1, const=None):
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

    Return:
        (jd, state_vects, masses): The common Julian date, and the state vectors and masses (kg) there, in the same
            order.
    """

    m_inits = [m_init]*len(state_vects) if np.ndim(m_init) == 0 else m_init

    const = backwardConstants(jd_ref, state_vects[0], m_inits[0], h_kill=h_kill, const=const)
    const.t_kill = t_kill
    frag, results, _ = runSimulation(const)
    t = results[-1][0]
    jd, state_vect = backwardState(jd_ref, state_vects[0], frag, t)
    states, masses = [state_vect], [frag.m]

    # The realizations stop by time alone, after as many steps as the nominal run, since the time adds up the same
    #   steps, and keep its atmosphere fit, made at practically the same place
    const.h_kill, const.t_kill = np.inf, abs(t)

    for sv, m in zip(state_vects[1:], m_inits[1:]):

        const_mc = copy.deepcopy(const)
        const_mc.m_init = m
        _startFrom(const_mc, jd_ref, sv)
        frag, results, _ = runSimulation(const_mc)

        if results[-1][0] != t:
            raise RuntimeError("A realization ended at t = {:.6f} s instead of the nominal {:.6f} s.".format(
                results[-1][0], t))

        states.append(backwardState(jd_ref, sv, frag, t)[1])
        masses.append(frag.m)

    return jd, states, masses


def photometricMass(traj, lum_eff, P_0m=None):
    """ Photometric mass of the trajectory (kg), from its whole light curve, which is the mass at its reference
        point if it ablated completely.

    Arguments:
        traj: [Trajectory]
        lum_eff: [str] Luminous efficiency: a model name from wmpl.Utils.Physics.LUM_EFF_MODELS (e.g.
            'borovicka2020'), or a constant in percent (e.g. '0.7').

    Keyword arguments:
        P_0m: [float] Power of a zero-magnitude meteor (W). None, the default, takes 1500 W for the panchromatic
            models and 840 W otherwise.
    """

    key = lum_eff.strip().lower()
    tau = key if key in LUM_EFF_MODELS else float(lum_eff)/100

    if P_0m is None:
        P_0m = 1500.0 if (key in LUM_EFF_MODELS) and (LUM_EFF_MODELS[key] in PANCHROMATIC_LUM_EFF_TYPES) else 840.0

    return computeMass(traj, P_0m, tau=tau)


def addBackwardArguments(arg_parser):
    """ Add the command-line arguments for the mass and the physical parameters of the meteoroid in a run back
        through the atmosphere, shared by this module's command line and REBOUND's. """

    arg_parser.add_argument("--mass", type=float, default=None,
        help="Mass at the trajectory's reference point in kg. By default, the photometric mass from the whole "
        "light curve, with --lum_eff and --P_0m.")

    arg_parser.add_argument("--lum_eff", type=str, default="0.7",
        help="Luminous efficiency for the photometric mass: a constant in percent, or a model name: "
        "{:s}. Default: 0.7.".format(", ".join(LUM_EFF_MODELS)))

    arg_parser.add_argument("--P_0m", type=float, default=None,
        help="Power of a zero-magnitude meteor in W for the photometric mass. Default: 1500 for the panchromatic "
        "models (rc2001*, cm1976, borovicka2020, pc1983), 840 otherwise.")

    arg_parser.add_argument("--mag_sigma", type=float, default=0.0,
        help="Uncertainty of the photometric calibration in magnitudes. Each Monte Carlo realization shifts the "
        "whole light curve by a normal draw with this sigma, which multiplies its mass by 10^(-0.4 shift). "
        "Default: 0.")

    arg_parser.add_argument("--freeze_mass", action="store_true",
        help="Keep the mass constant instead of growing it back as the ablation is undone.")

    arg_parser.add_argument("--ablation_coeff", type=float, default=Constants().sigma*1e6,
        help="Ablation coefficient in s^2/km^2, which sets how fast the mass grows back. Default: MetSim's, "
        "{:g}.".format(Constants().sigma*1e6))

    arg_parser.add_argument("--density", type=float, default=3000.0,
        help="Bulk density of the meteoroid in kg/m^3, which with the mass sets the drag, and in REBOUND the "
        "radiation pressure with --radius. Default: 3000.")


def backwardStatesFromArguments(traj, state_vects, args, h_kill, t_kill=-1, random_seed=None):
    """ backwardStates() from the trajectory's reference point, with the mass and physical parameters given by the
        command-line arguments of addBackwardArguments(). The masses of the realizations carry the photometric
        uncertainty --mag_sigma, drawn with random_seed. Also returns the starting masses. """

    const = Constants()
    const.freeze_mass = args.freeze_mass
    const.sigma = args.ablation_coeff/1e6
    const.rho = args.density

    m_init = args.mass if (args.mass is not None) else photometricMass(traj, args.lum_eff, args.P_0m)

    # A generator of its own, so the state vector draws stay those of sampleStateVectors with the same seed
    rng = np.random.default_rng(None if (random_seed is None) else [1, random_seed])
    m_inits = [m_init] + list(m_init*10**(-0.4*rng.normal(0.0, args.mag_sigma, len(state_vects) - 1)))

    return backwardStates(traj.jdt_ref, state_vects, m_inits, h_kill=h_kill, t_kill=t_kill, const=const), m_inits


if __name__ == "__main__":

    from wmpl.Rebound.REBOUND import sampleStateVectors
    from wmpl.Utils.Pickling import loadPickle
    from wmpl.Utils.TrajConversions import cartesian2Geo

    arg_parser = argparse.ArgumentParser(description="Run a trajectory, and optionally its Monte Carlo "
        "realizations, back up through the atmosphere from its reference point with MetSim (single body, drag, "
        "gravity, Coriolis), to a height or for a time. Saves where each one ended next to the pickle.")

    arg_parser.add_argument("pickle_path", type=str, help="Path to the trajectory pickle file.")

    arg_parser.add_argument("--atm_height", type=float, default=180.0,
        help="Height in km to run back to. Default: 180.")

    arg_parser.add_argument("--atm_time", type=float, default=-1,
        help="Run back for this many seconds instead, unless --atm_height is reached first.")

    arg_parser.add_argument("--mc", type=int, default=1,
        help="Number of Monte Carlo realizations drawn from the trajectory's state vector covariance, run back for "
        "as long as the nominal solution. Default: 1, the nominal solution only.")

    arg_parser.add_argument("--seed", type=int, default=None, help="Seed for the Monte Carlo realizations.")

    addBackwardArguments(arg_parser)

    args = arg_parser.parse_args()

    traj = loadPickle(*os.path.split(args.pickle_path))
    state_vect = np.concatenate([traj.state_vect_mini, traj.v_init*traj.radiant_eci_mini])
    state_vects = [state_vect] + sampleStateVectors(traj, args.mc, args.seed)

    (jd, states, masses), m_inits = backwardStatesFromArguments(traj, state_vects, args, 1000*args.atm_height,
        t_kill=args.atm_time, random_seed=args.seed)

    rows = []
    for i, (sv, m_ref, m) in enumerate(zip(states, m_inits, masses)):
        lat, lon, ht = cartesian2Geo(jd, *sv[:3])
        rows.append([i, m_ref, np.degrees(lat), np.degrees(lon), ht, np.linalg.norm(sv[3:]), m] + list(sv))
    rows = np.array(rows)

    print("Mass at the reference point: {:.6g} kg{:s}".format(m_inits[0], ", frozen" if args.freeze_mass else ""))
    print("Ran {:d} state vector(s) back {:.4f} s, to JD {:.8f}".format(len(states), (traj.jdt_ref - jd)*86400,
        jd))
    print("Nominal: lat {:.5f} deg, lon {:.5f} deg, height {:.1f} m, speed {:.2f} m/s, mass {:.6g} kg".format(
        *rows[0, 2:7]))
    if len(rows) > 1:
        for name, col, unit in [("mass at the reference point", 1, "kg"), ("height", 4, "m"),
                ("speed", 5, "m/s"), ("mass", 6, "kg")]:
            print("Realizations {:s}: 2.5/50/97.5 percentiles {:s} {:s}".format(name,
                " / ".join("{:.6g}".format(v) for v in np.percentile(rows[1:, col], [2.5, 50, 97.5])), unit))

    out_path = os.path.splitext(args.pickle_path)[0] + "_backward_atm.txt"
    np.savetxt(out_path, rows, fmt=["%d"] + ["%.10g"]*12, header="JD {:.10f} (UTC), {:.6f} s from the reference "
        "point. Row 0 is the nominal solution, the others its realizations. State vectors in ECI, true equator and "
        "equinox of date, velocity to the radiant.\nrow, mass at the reference point (kg), lat (deg), lon (deg), "
        "height MSL (m), speed (m/s), mass (kg), x (m), y (m), z (m), vx (m/s), vy (m/s), vz (m/s)".format(jd, (jd - traj.jdt_ref)*86400))
    print("Saved:", out_path)
