""" Tests that the run back through the atmosphere (wmpl.MetSim.BackwardAtmIntegration) handles the Earth's
rotation: it is checked against an integration of the same body in the inertial frame, where drag and ablation act on
the speed relative to the air turning with the ground, v - omega x r, and no rotating frame is involved.

Run under pytest, or directly:

    python -m wmpl.MetSim.Tests.test_BackwardAtmRotation
"""

import numpy as np
from scipy.integrate import solve_ivp

from wmpl.MetSim.BackwardAtmIntegration import _eciToEcef, backwardConstants, backwardStates, groundSpeed
from wmpl.MetSim.MetSimErosion import Constants, EARTH_ROTATION_RATE, atmDensityTable
from wmpl.Utils.TrajConversions import ecef2LatLonAlt, enu2ECEF, geo2Cartesian


JD = 2460600.25

# WGS84 GM and J2, and the equatorial radius J2 refers to
GM = 3.986004418e14
J2 = 1.08262668e-3
R_EQ = 6378137.0


def _stateVector(lat, lon, ht, v_ground, azim, elev):
    """ ECI state vector [position, velocity to the radiant] of a meteor at (lat, lon, ht) moving at v_ground relative
        to the ground, from the ground-fixed radiant (azim, elev). Angles in radians. """

    pos = np.array(geo2Cartesian(lat, lon, ht, JD))
    rad_enu = [np.cos(elev)*np.sin(azim), np.cos(elev)*np.cos(azim), np.sin(elev)]
    rad_eci = _eciToEcef(JD).T.dot(np.array(enu2ECEF(lat, lon, *rad_enu)))

    # The inertial velocity of the motion is the one relative to the ground plus omega x r
    v_motion = -v_ground*rad_eci + np.cross([0.0, 0.0, EARTH_ROTATION_RATE], pos)

    return np.concatenate([pos, -v_motion])


def _inertialRun(state_vect, m_init, const, t_end):
    """ The same body run back to t_end (s, negative) in ECI, with GM/r^2 + J2 gravity, and drag and ablation on the
        speed relative to the air, in MetSim's density table at the geodetic height. """

    K = const.gamma*const.shape_factor*const.rho**(-2/3.0)
    omega = np.array([0.0, 0.0, EARTH_ROTATION_RATE])

    def rhs(t, y):

        r, v, m = y[:3], y[3:6], y[6]
        r_norm = np.sqrt(r @ r)

        # Gravity with J2 about the rotation axis of date (the ECI z axis)
        z2 = (r[2]/r_norm)**2
        j2 = 1.5*J2*(R_EQ/r_norm)**2
        g = -GM*r/r_norm**3*np.array([1 - j2*(5*z2 - 1), 1 - j2*(5*z2 - 1), 1 - j2*(5*z2 - 3)])

        ht = ecef2LatLonAlt(*_eciToEcef(JD + t/86400.0).dot(r))[2]
        rho_atm = atmDensityTable(ht, const.atm_table_ht, const.atm_table_log10_rho)

        v_air = v - np.cross(omega, r)
        v_air_norm = np.sqrt(v_air @ v_air)

        a = g - K*m**(-1/3.0)*rho_atm*v_air_norm*v_air
        dm = -K*const.sigma*m**(2/3.0)*rho_atm*v_air_norm**3

        return np.concatenate([v, a, [dm]])

    y0 = np.concatenate([state_vect[:3], -state_vect[3:], [m_init]])
    y = solve_ivp(rhs, (0.0, t_end), y0, method="DOP853", rtol=1e-10, atol=1e-6).y[:, -1]

    return np.concatenate([y[:3], -y[3:6]]), y[6]


def _angleArcsec(a, b):
    return np.degrees(np.arccos(np.clip(np.dot(a, b)/np.linalg.norm(a)/np.linalg.norm(b), -1, 1)))*3600


def test_start_sets_the_latitude_of_the_gravity():
    """ The run sets const.lat, which MetSim's latitude-dependent gravity reads, and not only const.latitude, which
        sets the Coriolis acceleration: otherwise the gravity is that of 45 deg wherever the meteor is. """

    for lat_deg in (0.0, -30.0, 70.0):
        state_vect = _stateVector(np.radians(lat_deg), np.radians(10.0), 75e3, 15000.0, np.radians(90.0),
            np.radians(20.0))
        const = backwardConstants(JD, state_vect, 0.5)

        assert np.isclose(const.lat, const.latitude, rtol=0, atol=1e-12)
        assert np.isclose(np.degrees(const.lat), lat_deg, rtol=0, atol=1e-6)


def test_ground_speed_takes_out_the_rotation():
    """ At the equator the rotation adds 465 m/s to a meteor moving east, and takes it away from one moving west. """

    for azim_deg in (90.0, 270.0, 0.0):
        state_vect = _stateVector(0.0, np.radians(10.0), 75e3, 15000.0, np.radians(azim_deg), np.radians(20.0))
        assert abs(groundSpeed(JD, state_vect) - 15000.0) < 1e-3

    # Moving west (radiant in the east) the inertial speed is lower, moving east higher
    v_west = np.linalg.norm(_stateVector(0.0, 0.1, 75e3, 15000.0, np.radians(90.0), np.radians(20.0))[3:])
    v_east = np.linalg.norm(_stateVector(0.0, 0.1, 75e3, 15000.0, np.radians(270.0), np.radians(20.0))[3:])
    assert 430 < 15000.0 - v_west < 450
    assert 430 < v_east - 15000.0 < 450


def test_run_back_matches_an_inertial_integration():
    """ Run back from 75 km to 180 km at the equator, where the rotation matters most (a ground speed of 15 km/s is
        441 m/s from the inertial one there), it matches the inertial integration to within a metre per second and a
        few arcseconds, and the mass to 0.1%. Ignoring the rotation (the inertial speed and radiant taken as relative
        to the ground, no Coriolis acceleration) misses it by 22 m/s, 300 arcsec and 9 km. """

    for azim_deg in (90.0, 270.0):

        state_vect = _stateVector(0.0, np.radians(10.0), 75e3, 15000.0, np.radians(azim_deg), np.radians(20.0))

        const = Constants()
        const.sigma = 0.023/1e6
        const.rho = 3500.0

        jd, (state_end,), (mass_end,) = backwardStates(JD, [state_vect], 0.5, const=const)
        const_used = backwardConstants(JD, state_vect, 0.5, const=const)
        state_ref, mass_ref = _inertialRun(state_vect, 0.5, const_used, (jd - JD)*86400.0)

        assert np.linalg.norm(state_end[:3] - state_ref[:3]) < 20.0
        assert np.linalg.norm(state_end[3:] - state_ref[3:]) < 1.0
        assert _angleArcsec(state_end[3:], state_ref[3:]) < 15.0
        assert abs(mass_end/mass_ref - 1) < 1e-3

        # A uniform gravity of 9.81 m/s^2 leaves the radiant 8 arcsec off at the equator, where the gravity felt is
        #   9.78 m/s^2. With the WGS84 gravity at the meteor's latitude (MetSim's const.lat) it matches to 0.1 arcsec,
        #   and to 7 arcsec with the gravity of 45 deg
        if hasattr(Constants(), "lat"):
            assert _angleArcsec(state_end[3:], state_ref[3:]) < 1.0


if __name__ == "__main__":
    test_start_sets_the_latitude_of_the_gravity()
    test_ground_speed_takes_out_the_rotation()
    test_run_back_matches_an_inertial_integration()
    print("All BackwardAtmRotation checks passed.")
