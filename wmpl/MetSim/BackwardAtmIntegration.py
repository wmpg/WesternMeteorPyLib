""" Run MetSimErosion back up a solved trajectory from its reference point, to before it was observed, and give
    where it ended in the trajectory solver's ECI frame.

    const = backwardConstants(traj, m_init)       # e.g. the photometric mass at the reference point
    frag, results, _ = runSimulation(const)        # optionally set const.t_kill or const.freeze_mass first
    jd, pos, vel = backwardState(traj, frag, results[-1][0])
"""

import copy

import numpy as np

from wmpl.MetSim.MetSimErosion import Constants, EARTH_ROTATION_RATE
from wmpl.Utils.AtmosphereDensity import fitAtmPoly
from wmpl.Utils.TrajConversions import cartesian2Geo, enu2ECEF, jd2LST


def _eciToEcef(jd):
    """ Rotation matrix from ECI (true equator and equinox of date, as the trajectory solver uses) to ECEF, by the
        Greenwich apparent sidereal time about the rotation axis. """

    gst = np.radians(jd2LST(jd, 0.0)[1])

    return np.array([[np.cos(gst), np.sin(gst), 0.0], [-np.sin(gst), np.cos(gst), 0.0], [0.0, 0.0, 1.0]])


def backwardConstants(traj, m_init, h_kill=180000.0, const=None):
    """ Constants for a backward MetSimErosion run, in 3D with gravity and the Coriolis acceleration, from the
        reference point of a solved trajectory (traj.state_vect_mini at traj.jdt_ref) up to h_kill.

    Arguments:
        traj: [Trajectory] Solved trajectory, with its orbit.
        m_init: [float] Mass at the reference point (kg), e.g. the photometric mass.

    Keyword arguments:
        h_kill: [float] Height to stop at (m), 180 km by default. The atmosphere density is fitted up to it, so to
            stop after a given time with t_kill instead, keep h_kill above the height that time reaches.
        const: [Constants] Physical parameters to start from (rho, sigma, gamma, shape_factor, dt, freeze_mass...),
            e.g. from a MetSim fit. It is copied, not changed. Constants() by default.

    Return:
        const: [Constants]
    """

    const = Constants() if const is None else copy.deepcopy(const)

    lat, lon, ht = cartesian2Geo(traj.jdt_ref, *traj.state_vect_mini)

    # The solver's velocity is in ECI, so it includes the Earth's rotation, while MetSim follows the motion relative
    #   to the ground, which the orbit gives at the reference point
    const.v_init = traj.orbit.v_init_norot
    const.zenith_angle = np.pi/2 - traj.orbit.elevation_apparent_norot
    const.radiant_azimuth = traj.orbit.azimuth_apparent_norot

    const.h_init, const.latitude, const.m_init = ht, lat, m_init
    const.gravity_3d = True
    const.dt, const.h_kill = -abs(const.dt), h_kill
    const.erosion_on = const.disruption_on = const.fragmentation_on = False
    const.dens_co = fitAtmPoly(lat, lon, ht, h_kill, traj.jdt_ref)

    return const


def backwardState(traj, frag, t):
    """ Where a run with the constants from backwardConstants() ended, in the trajectory solver's frame.

    Arguments:
        traj: [Trajectory] The trajectory given to backwardConstants().
        frag: [Fragment] The fragment runSimulation() returned.
        t: [float] Time of its last step (s), negative: the first column of the last row of runSimulation()'s
            results.

    Return:
        (jd, pos, vel):
            jd: [float] Julian date.
            pos: [ndarray] ECI position (m), true equator and equinox of date.
            vel: [ndarray] ECI velocity (m/s), along the motion.
    """

    lat, lon, _ = cartesian2Geo(traj.jdt_ref, *traj.state_vect_mini)

    # MetSim follows the fragment in the east-north-up frame of the start, fixed to the ground
    pos_ecef = _eciToEcef(traj.jdt_ref).dot(traj.state_vect_mini) \
        + np.array(enu2ECEF(lat, lon, frag.px, frag.py, frag.pz))
    vel_ecef = np.array(enu2ECEF(lat, lon, frag.vx, frag.vy, frag.vz))

    jd = traj.jdt_ref + t/86400.0
    ecef_to_eci = _eciToEcef(jd).T
    pos = ecef_to_eci.dot(pos_ecef)
    vel = ecef_to_eci.dot(vel_ecef) + np.cross([0.0, 0.0, EARTH_ROTATION_RATE], pos)

    return jd, pos, vel
