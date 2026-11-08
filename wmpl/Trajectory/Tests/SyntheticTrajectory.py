""" Synthetic meteors for the tests: a meteor which decelerates by atmospheric drag along a straight line,
    observed from three stations.
"""

from __future__ import print_function, division, absolute_import

import numpy as np

from wmpl.Trajectory.Trajectory import Trajectory, applyGravityDrop
from wmpl.Utils.AtmosphereDensity import getAtmDensity_vect
from wmpl.Utils.Math import vectMag
from wmpl.Utils.TrajConversions import altAz2RADec, cartesian2Geo, eci2RaDec, geo2Cartesian, \
    raDec2AltAz_vect, raDec2ECI


# Reference Julian date of the meteor
JDT_REF = 2460600.25

# Geographical coordinates of the point on the line at the top of the simulation (deg, deg, m)
TOP_LAT = 45.0
TOP_LON = 15.0
TOP_HT = 120000.0

# Station ID, latitude (deg), longitude (deg), elevation (m) and the observed part of the visible
#   trajectory (fractions of the visible duration)
STATIONS = [
    ('A', 45.6, 14.6, 200.0, 0.0, 1.0),
    ('B', 44.5, 15.3, 400.0, 0.1, 1.0),
    ('C', 45.1, 15.9, 150.0, 0.0, 0.85),
]



class DragMeteor(object):
    def __init__(self, v_init, azim, elev, k_beg, k_end, t_vis_beg, t_vis_end, dt=0.001, duration=20.0):
        """ Integrate the motion of a meteor decelerated by drag along a straight line. The ballistic
            coefficient grows linearly with time during the visible part of the trajectory.

        Arguments:
            v_init: [float] Velocity at the top of the simulation (m/s).
            azim: [float] Azimuth of the radiant at the top, +E of due N (deg).
            elev: [float] Elevation of the radiant at the top (deg).
            k_beg: [float] Ballistic coefficient before the visible part (m^2/kg).
            k_end: [float] Ballistic coefficient after the visible part (m^2/kg).
            t_vis_beg: [float] Time of the beginning of the visible part, from the top (s).
            t_vis_end: [float] Time of the end of the visible part, from the top (s).

        Keyword arguments:
            dt: [float] Integration step (s). 0.001 s by default.
            duration: [float] Duration of the integration (s). 20 s by default.
        """

        self.t_vis_beg = t_vis_beg
        self.t_vis_end = t_vis_end

        # Compute the top point and the radiant direction
        lat, lon = np.radians(TOP_LAT), np.radians(TOP_LON)
        self.top_eci = np.array(geo2Cartesian(lat, lon, TOP_HT, JDT_REF))
        ra, dec = altAz2RADec(np.radians(azim), np.radians(elev), JDT_REF, lat, lon)
        self.radiant_eci = np.array(raDec2ECI(ra, dec))

        # Tabulate the air density with height along the line
        len_table = np.linspace(0, 400e3, 401)
        ht_table = np.array([cartesian2Geo(JDT_REF, *eci)[2] for eci in self.eciAtLength(len_table)])
        lat_lon = [cartesian2Geo(JDT_REF, *eci)[:2] for eci in self.eciAtLength(len_table)]
        lat_table, lon_table = np.array(lat_lon).T
        rho_table = getAtmDensity_vect(lat_table, lon_table, np.maximum(ht_table, 0.0), JDT_REF)
        log_rho_table = np.log(rho_table)
        g_table = -9.81*np.gradient(ht_table, len_table)


        def _acceleration(t, len_val, vel):

            # Interpolate the ballistic coefficient linearly in the visible part
            frac = np.clip((t - t_vis_beg)/(t_vis_end - t_vis_beg), 0.0, 1.0)
            k = k_beg + frac*(k_end - k_beg)

            rho = np.exp(np.interp(len_val, len_table, log_rho_table))
            g = np.interp(len_val, len_table, g_table)

            return -k*rho*vel**2 + g


        # Integrate with the 4th order Runge-Kutta method
        n = int(duration/dt)
        self.time = np.arange(n + 1)*dt
        self.length = np.zeros(n + 1)
        self.vel = np.zeros(n + 1)
        self.vel[0] = v_init
        for i in range(n):

            t, len_val, vel = self.time[i], self.length[i], self.vel[i]

            k1_len = vel
            k1_vel = _acceleration(t, len_val, vel)
            k2_len = vel + dt/2*k1_vel
            k2_vel = _acceleration(t + dt/2, len_val + dt/2*k1_len, vel + dt/2*k1_vel)
            k3_len = vel + dt/2*k2_vel
            k3_vel = _acceleration(t + dt/2, len_val + dt/2*k2_len, vel + dt/2*k2_vel)
            k4_len = vel + dt*k3_vel
            k4_vel = _acceleration(t + dt, len_val + dt*k3_len, vel + dt*k3_vel)

            self.length[i + 1] = len_val + dt/6*(k1_len + 2*k2_len + 2*k3_len + k4_len)
            self.vel[i + 1] = vel + dt/6*(k1_vel + 2*k2_vel + 2*k3_vel + k4_vel)

            # Stop when the meteor has slowed down
            if self.vel[i + 1] < 500.0:
                self.time = self.time[:i + 2]
                self.length = self.length[:i + 2]
                self.vel = self.vel[:i + 2]
                break



    def eciAtLength(self, len_arr):
        """ Compute the ECI coordinates at the given lengths along the line from the top point.

        Arguments:
            len_arr: [float or ndarray] Lengths from the top point (m).

        Return:
            [ndarray] ECI coordinates (m), one row per point.
        """

        return self.top_eci[None, :] - np.outer(np.atleast_1d(len_arr), self.radiant_eci)



    def eciAtTime(self, time_arr):
        """ Compute the ECI coordinates at the given times.

        Arguments:
            time_arr: [float or ndarray] Times relative to the beginning of the visible part (s).

        Return:
            [ndarray] ECI coordinates (m), one row per point.
        """

        return self.eciAtLength(np.interp(np.atleast_1d(time_arr) + self.t_vis_beg, self.time, self.length))



    def velocityAtTime(self, time_arr):
        """ Compute the velocity at the given times.

        Arguments:
            time_arr: [float or ndarray] Times relative to the beginning of the visible part (s).

        Return:
            [ndarray] Velocities (m/s).
        """

        return np.interp(np.atleast_1d(time_arr) + self.t_vis_beg, self.time, self.vel)



def stationAzimAlt(eci_points, jds, lat, lon, ele):
    """ Compute the directions of points on a trajectory as seen from a station.

    Arguments:
        eci_points: [ndarray] ECI coordinates of the points (m).
        jds: [ndarray] Julian dates of the points.
        lat: [float] Latitude of the station (deg).
        lon: [float] Longitude of the station (deg).
        ele: [float] Elevation of the station (m).

    Return:
        (azim, elev):
            - azim: [ndarray] Azimuths of the points, +E of due N (rad).
            - elev: [ndarray] Elevations of the points (rad).
    """

    ra_list, dec_list = [], []
    for eci, jd in zip(eci_points, jds):
        station_eci = np.array(geo2Cartesian(np.radians(lat), np.radians(lon), ele, jd))
        ra, dec = eci2RaDec(eci - station_eci)
        ra_list.append(ra)
        dec_list.append(dec)

    return raDec2AltAz_vect(np.array(ra_list), np.array(dec_list), jds, np.radians(lat), np.radians(lon))



def observeMeteor(meteor, noise=30.0, fps=25.0, seed=1, **kwargs):
    """ Observe the visible part of the synthetic meteor from three stations and make a trajectory object.
        The times are relative to the beginning of the visible part.

    Arguments:
        meteor: [DragMeteor object] Synthetic meteor.

    Keyword arguments:
        noise: [float] Noise of the azimuth and elevation (arcsec). 30 by default.
        fps: [float] Frame rate of the cameras. 25 by default.
        seed: [int] Seed of the noise. 1 by default.
        kwargs: Passed to the Trajectory object.

    Return:
        traj: [Trajectory object] Trajectory object with the observations, not yet solved.
    """

    rng = np.random.default_rng(seed)

    traj_kwargs = dict(meastype=2, max_toffset=1.0, gravity_correction=False, monte_carlo=False, \
        show_plots=False, save_results=False, verbose=False)
    traj_kwargs.update(kwargs)
    traj = Trajectory(JDT_REF, **traj_kwargs)

    duration = meteor.t_vis_end - meteor.t_vis_beg
    noise = np.radians(noise/3600.0)

    for station_id, lat, lon, ele, frac_beg, frac_end in STATIONS:

        # Sample the frames with a random phase
        times = np.arange(frac_beg*duration + rng.uniform(0, 1/fps), frac_end*duration, 1/fps)
        jds = JDT_REF + times/86400.0

        # Compute the direction of every point as seen from the station
        azim, elev = stationAzimAlt(meteor.eciAtTime(times), jds, lat, lon, ele)

        # Add noise to the measurements
        azim = azim + rng.normal(0, noise, len(azim))/np.cos(elev)
        elev = elev + rng.normal(0, noise, len(elev))

        traj.infillTrajectory(azim, elev, times, np.radians(lat), np.radians(lon), ele, station_id=station_id)


    return traj



# Reference Julian date of the straight meteor
STRAIGHT_JDT_REF = 2461070.9343128

# Geographical coordinates of the beginning of the straight meteor (deg, deg, m)
STRAIGHT_BEG_LAT = -43.20
STRAIGHT_BEG_LON = 172.30
STRAIGHT_BEG_HT = 85000.0

# Duration (s), number of points per station and velocity (m/s) of the straight meteor
STRAIGHT_DURATION = 2.5
STRAIGHT_N_POINTS = 40
STRAIGHT_V_INIT = 30000.0

# Station ID, latitude (deg), longitude (deg) and elevation (m) of the stations seeing the straight meteor
STRAIGHT_STATIONS = [
    ('A', -43.60, 172.12, 120.0),
    ('B', -42.90, 173.20, 200.0),
    ('C', -43.30, 171.30, 350.0),
]



def straightMeteorPoints(duration=STRAIGHT_DURATION, v_init=STRAIGHT_V_INIT, elevation=55.0, \
    gravity_drop=False):
    """ Compute the points of a straight, constant-velocity meteor.

    Keyword arguments:
        duration: [float] Duration of the meteor (s). STRAIGHT_DURATION by default.
        v_init: [float] Velocity of the meteor (m/s). STRAIGHT_V_INIT by default.
        elevation: [float] Elevation of the radiant at the beginning of the meteor (deg). 55 by default.
        gravity_drop: [bool] Drop the meteor by gravity with applyGravityDrop, so that it is not a straight
            line anymore. False by default.

    Return:
        (times, points, begin, radiant_eci):
            - times: [ndarray] Times of the points, from the beginning (s).
            - points: [ndarray] ECI coordinates of the points (m), one row per point.
            - begin: [ndarray] ECI coordinates of the beginning of the meteor (m).
            - radiant_eci: [ndarray] True radiant unit vector in ECI coordinates.
    """

    # Compute the beginning point and the radiant
    lat, lon = np.radians(STRAIGHT_BEG_LAT), np.radians(STRAIGHT_BEG_LON)
    begin = np.array(geo2Cartesian(lat, lon, STRAIGHT_BEG_HT, STRAIGHT_JDT_REF))
    ra, dec = altAz2RADec(np.radians(210.0), np.radians(elevation), STRAIGHT_JDT_REF, lat, lon)
    radiant_eci = np.array(raDec2ECI(ra, dec))

    # Compute the points along the trajectory
    times = np.linspace(0, duration, STRAIGHT_N_POINTS)
    points = begin - np.outer(v_init*times, radiant_eci)

    # Drop the points by gravity, with the vertical component of the velocity
    if gravity_drop:
        v0z = -v_init*np.sin(np.radians(elevation))
        points = np.array([applyGravityDrop(point, t, vectMag(begin), 1.0, v0z) \
            for point, t in zip(points, times)])

    return times, points, begin, radiant_eci



def observeStraightMeteor(noise=1e-4, seed=1, duration=STRAIGHT_DURATION, v_init=STRAIGHT_V_INIT, \
    elevation=55.0, gravity_drop=False, clock_offsets=None, **kwargs):
    """ Observe a straight, constant-velocity meteor from three stations and make a trajectory object.

    Keyword arguments:
        noise: [float] Standard deviation of the noise on the azimuth and elevation (radians). 1e-4 by
            default, about 20 arcsec.
        seed: [int] Seed of the noise. 1 by default.
        duration: [float] Duration of the meteor (s). STRAIGHT_DURATION by default.
        v_init: [float] Velocity of the meteor (m/s). STRAIGHT_V_INIT by default.
        elevation: [float] Elevation of the radiant at the beginning of the meteor (deg). 55 by default.
        gravity_drop: [bool] Drop the meteor by gravity with applyGravityDrop, so that it is not a straight
            line anymore. False by default.
        clock_offsets: [dict] Seconds by which the clock of the station is ahead, per station ID. None by
            default, when all clocks are exact.
        **kwargs: [dict] Keyword arguments of the Trajectory object, which replace the defaults (no timing
            offsets above 1 s, no gravity correction, no Monte Carlo, no plots and no saving).

    Return:
        (traj, begin, radiant_eci):
            - traj: [Trajectory object] Trajectory object with the observations, not yet solved.
            - begin: [ndarray] ECI coordinates of the beginning of the meteor (m).
            - radiant_eci: [ndarray] True radiant unit vector in ECI coordinates.
    """

    rng = np.random.default_rng(seed)

    if clock_offsets is None:
        clock_offsets = {}

    # Compute the points along the trajectory
    times, points, begin, radiant_eci = straightMeteorPoints(duration=duration, v_init=v_init, \
        elevation=elevation, gravity_drop=gravity_drop)

    traj_kwargs = dict(meastype=2, max_toffset=1.0, gravity_correction=False, monte_carlo=False, \
        show_plots=False, save_results=False, verbose=False)
    traj_kwargs.update(kwargs)
    traj = Trajectory(STRAIGHT_JDT_REF, **traj_kwargs)

    for station_id, s_lat, s_lon, ele in STRAIGHT_STATIONS:

        jds = STRAIGHT_JDT_REF + times/86400.0

        # Compute the direction of every point as seen from the station
        azim, elev = stationAzimAlt(points, jds, s_lat, s_lon, ele)

        # Add noise to the measurements
        azim = azim + rng.normal(0, noise, len(times))/np.cos(elev)
        elev = elev + rng.normal(0, noise, len(times))

        traj.infillTrajectory(azim, elev, times + clock_offsets.get(station_id, 0.0), np.radians(s_lat), \
            np.radians(s_lon), ele, station_id=station_id)

    return traj, begin, radiant_eci
