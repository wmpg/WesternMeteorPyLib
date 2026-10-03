""" Run MetSimErosion back up a solved trajectory from its reference point, to before it was observed, and give
    where it ended in the trajectory solver's ECI frame, ready for wmpl.Rebound.REBOUND.reboundSimulate().

    state_vect = np.r_[traj.state_vect_mini, traj.v_init*traj.radiant_eci_mini]
    jd, state_vects = backwardStates(traj.jdt_ref, [state_vect] + realizations, m_init)

State vectors are [x, y, z, vx, vy, vz] in ECI (true equator and equinox of date), in m and m/s, with the
velocity pointing to the radiant, as the solver gives them and reboundSimulate() takes them.
"""

import copy

import numpy as np

from wmpl.MetSim.MetSimErosion import Constants, EARTH_ROTATION_RATE, runSimulation
from wmpl.Utils.AtmosphereDensity import fitAtmPoly
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


def backwardStates(jd_ref, state_vects, m_init, h_kill=180000.0, const=None):
    """ Run several state vectors back through the atmosphere to a common epoch, so the nominal solution and its
        Monte Carlo realizations can go into reboundSimulate() together. The first state vector, the nominal one,
        is run up to h_kill, and the others for the same time, so they end near h_kill.

    Arguments:
        jd_ref: [float] Julian date of the state vectors.
        state_vects: [list] State vectors (see the module docstring), the nominal one first.
        m_init: [float] Mass at the start (kg), the same for all of them.

    Keyword arguments:
        h_kill, const: As in backwardConstants().

    Return:
        (jd, state_vects): The common Julian date and the state vectors there, in the same order.
    """

    const = backwardConstants(jd_ref, state_vects[0], m_init, h_kill=h_kill, const=const)
    frag, results, _ = runSimulation(const)
    t = results[-1][0]
    jd, state_vect = backwardState(jd_ref, state_vects[0], frag, t)
    states = [state_vect]

    # The realizations stop by time alone, after as many steps as the nominal run, since the time adds up the same
    #   steps, and keep its atmosphere fit, made at practically the same place
    const.h_kill, const.t_kill = np.inf, abs(t)

    for sv in state_vects[1:]:

        const_mc = copy.deepcopy(const)
        _startFrom(const_mc, jd_ref, sv)
        frag, results, _ = runSimulation(const_mc)

        if results[-1][0] != t:
            raise RuntimeError("A realization ended at t = {:.6f} s instead of the nominal {:.6f} s.".format(
                results[-1][0], t))

        states.append(backwardState(jd_ref, sv, frag, t)[1])

    return jd, states
