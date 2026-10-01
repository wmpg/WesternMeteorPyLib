""" Dynamic mass of a fireball near the end of its luminous path, and where, in which direction and how fast
    ablation ends, as the starting conditions of a dark flight computation.

The deceleration is fitted to the point velocities of a solved trajectory over a height window, and the drag
equation gives the dynamic mass, m = (Gamma*A*rho_air*v^2/a)^3/rho_m^2. The solver's velocities are inertial
(ECI), so they are first taken relative to the ground, which the air moves with. A single-body MetSim simulation
then follows that mass from the evaluation point down to the speed where ablation is taken to stop, in 3D in a
frame fixed to the ground, with gravity and the Coriolis acceleration in its velocity.

Since the mass goes as the cube of the air density and the sixth power of the speed relative to the air, the air
density and the winds can be taken from an atmosphere profile file, read as OpenDarkflight reads it, so that a
dark flight continues in the same atmosphere. With winds, the speed relative to the air is used for the mass, and
the simulation runs MetSim with the wind of each height. The uncertainty of the trajectory solver can be propagated
with its Monte Carlo realizations (--mc), and the results saved for a dark flight code (--save_pickle).

See wmpl/Docs/DynamicMassFit.md for the full description, the reasons behind each choice, and examples.
"""

import os
import multiprocessing

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from wmpl.Utils.AtmosphereDensity import fitAtmPoly, addAtmosphereArguments, setAtmosphere
from wmpl.Utils.Math import lineFunc, vectMag, rotateVector
from wmpl.Utils.TrajConversions import cartesian2Geo, derotatedRadiantAltAz, jd2LST, enu2ECEF, ecef2ENU
from wmpl.Utils.Physics import dynamicMass
from wmpl.Utils.Pickling import loadPickle
from wmpl.Utils.PyDomainParallelizer import domainParallelizer
from wmpl.Utils.DynamicMassFitExport import ejectionState, buildDynMassFitOutput, saveDynMassFitPickle
from wmpl.Utils.AtmosphereProfile import AtmosphereProfile, PROFILE_TYPES
from wmpl.MetSim.MetSimErosion import Constants, runSimulation, G0, EARTH_ROTATION_RATE
from wmpl.MetSim.GUI import SimulationResults
from wmpl.Trajectory.Trajectory import applyGravityDrop


# Lowest height of the fragment simulation (m)
SIM_HT_MIN = 15000


def _rotationVelocity(eci_pos):
    """ Velocity (m/s) of the ground at the given ECI position(s) (m), from the rotation of the Earth around the
        z axis of the ECI frame of date.
    """

    eci_pos = np.asarray(eci_pos, dtype=float)

    return EARTH_ROTATION_RATE*np.stack([-eci_pos[..., 1], eci_pos[..., 0], np.zeros_like(eci_pos[..., 0])], \
        axis=-1)


def groundSpeed(traj, eci_pos, vel):
    """ Speed relative to the ground (m/s) of a point on the trajectory, from the speed along the fitted line that
        the solver measures.

        The solver works in the ECI frame with moving stations, so its point velocities are inertial: they include
        the component of the Earth's rotation along the motion (217 m/s for an eastward fireball at 52 deg N).
        Drag acts on the speed relative to the air, which moves with the ground, and the mass goes as its sixth
        power, so the dynamic mass and the end of ablation need the ground speed. Relative to the ground the body
        moves along the fixed direction of the apparent ground-fixed radiant, so the speed along the line is that
        speed projected on the line plus the rotation velocity projected on it.

    Arguments:
        traj: [Trajectory] Solved trajectory.
        eci_pos: [ndarray] ECI position(s) of the point(s) (m), shape (3,) or (N, 3).
        vel: [float or ndarray] Speed(s) along the fitted line in the ECI frame (m/s).

    Return:
        [float or ndarray] Speed(s) relative to the ground (m/s).
    """

    motion = -traj.radiant_eci_mini
    direction_ground = traj.v_init*motion - _rotationVelocity(traj.state_vect_mini)
    direction_ground /= np.sqrt(direction_ground @ direction_ground)

    return (vel - _rotationVelocity(eci_pos) @ motion)/np.dot(direction_ground, motion)


def evalPointState(traj, height, ht_vs_time_interp=None):
    """ Where the body is and where it moves, relative to the ground, at the given height of the trajectory.

        The direction is the apparent ground-fixed radiant at that point, as Orbit.calcOrbit() computes it at the
        reference point, steepened by the gravity turn since the point where the fitted radiant is tangent to the
        path: its beginning if the solver modelled the gravity drop, otherwise about its middle. The turn uses the
        average speed relative to the ground, g*cos(elev)*t/v_avg.

    Arguments:
        traj: [Trajectory] Solved trajectory.
        height: [float] Height of the point (m).

    Keyword arguments:
        ht_vs_time_interp: [callable] Time as a function of height, from interpolateHtVsTimeLen(), if it is
            already at hand.

    Return:
        (time, length, eci, jd, lat, lon, azim, elev): [tuple]
            time: [float] Time of the point after traj.jdt_ref (s).
            length: [float] Length from the state vector along the fitted line (m).
            eci: [ndarray] ECI position of the point, gravity drop included (m).
            jd: [float] Julian date of the point.
            lat, lon: [float] Geodetic latitude and longitude of the point (radians).
            azim, elev: [float] Azimuth (+E of due N) and elevation of the apparent ground-fixed radiant at the
                point, gravity turn included (radians).
    """

    if ht_vs_time_interp is None:
        ht_vs_time_interp, _ = interpolateHtVsTimeLen(traj, sample_step=0.1, show_plots=False)

    # Get the time at the point, and its length from the trajectory geometry. Interpolating the length instead
    #   mixes the smoothed heights with the raw lengths, which at shallow entry angles turns a small height
    #   difference into a large length difference
    time = float(ht_vs_time_interp(height))

    # The height along the line falls from the state vector until the point nearest the Earth's centre, at a
    #   length of state_vect.radiant, which brackets the root
    length = scipy.optimize.brentq(lambda l: cartesian2Geo(traj.jdt_ref + time/86400, \
        *pointOnTrajectory(traj, l, time))[2] - height, 0, np.dot(traj.state_vect_mini, traj.radiant_eci_mini))

    eci = pointOnTrajectory(traj, length, time)
    jd = traj.jdt_ref + time/86400
    lat, lon, _ = cartesian2Geo(jd, *eci)

    # Derotate the fitted radiant at the point. The radiant is the tangent of the path at its beginning (the solver
    #   models gravity as a drop from that line), so it is derotated with the initial velocity, as
    #   Orbit.calcOrbit() does. Drag does not rotate the direction of motion relative to the air, so this
    #   ground-fixed direction holds along the path up to the gravity turn
    azim, elev, _ = derotatedRadiantAltAz(traj.v_init*traj.radiant_eci_mini, eci, jd, lat, lon)

    t_obs = np.concatenate([obs.time_data for obs in traj.observations])
    t_turn = 0.0 if getattr(traj, 'gravity_correction', True) else (np.min(t_obs) + np.max(t_obs))/2
    elev += G0/(1 + height/Constants().r_earth)**2*np.cos(elev)*(time - t_turn)/traj.orbit.v_avg_norot

    return time, length, eci, jd, lat, lon, azim, elev


def _airDensity(atm_profile, height):
    """ Air density (kg/m^3) at the given height (m) from the atmosphere profile, or None if there is no
        profile, in which case dynamicMass() computes it from the MSIS model.
    """

    if atm_profile is None:
        return None

    return atm_profile.density(height)


def _msisSettings():
    """ The MSIS options of this process (version and date, see AtmosphereDensity.setAtmosphere()), to hand
        them to the Monte Carlo workers. Where processes are spawned instead of forked (macOS, Windows) the
        workers start from the defaults, and would otherwise use NRLMSISE-00 whatever --atm says.
    """

    module = fitAtmPoly.__globals__

    return module['MSIS_VERSION'], module['MSIS_JD']


def _msisName():
    """ Name of the MSIS model selected with --atm (see AtmosphereDensity.setAtmosphere()). """

    version = _msisSettings()[0]

    return "NRLMSISE-00" if version == "00" else "NRLMSIS " + version


def _setMSISSettings(settings):
    """ Apply the MSIS options returned by _msisSettings() in this process. """

    module = fitAtmPoly.__globals__
    module['MSIS_VERSION'], module['MSIS_JD'] = settings


def _endDecel(sr):
    """ Deceleration at the end of a fragment simulation (m/s^2), or NaN if it took a single step, which
        happens when it starts less than one step above the kill speed.
    """

    if len(sr.time_arr) < 2:
        return np.nan

    return (sr.main_vel_arr[-1] - sr.main_vel_arr[-2])/(sr.time_arr[-1] - sr.time_arr[-2])


def _motionENU(azim, elev):
    """ Unit vector (east, north, up) of the direction of motion, opposite to a radiant at the given azimuth
        (+E of due N) and elevation, in radians.
    """

    return -np.array([np.sin(azim)*np.cos(elev), np.cos(azim)*np.cos(elev), np.sin(elev)])


def _windENU(atm_profile, height):
    """ Wind velocity (east, north, up) in m/s at the given height (m) from the atmosphere profile, or None if
        there is no profile or its winds are switched off.
    """

    if (atm_profile is None) or (not atm_profile.use_winds):
        return None

    return np.append(atm_profile.wind(height), 0.0)


def _airSpeed(atm_profile, height, vel, azim, elev):
    """ Speed relative to the air (m/s), which the drag depends on, for the given speed relative to the ground
        (m/s) at the given height (m), moving opposite to a radiant at the given azimuth and elevation (radians).
        Without winds it is the given speed, and the direction is not used.
    """

    wind = _windENU(atm_profile, height)
    if wind is None:
        return vel

    return vectMag(vel*_motionENU(azim, elev) - wind)


def runFragSim(mass, density, lat, lon, jd, ht_beg, v_init, entry_angle, gamma_a, v_kill=3000, \
    atm_profile=None, radiant_azimuth=0.0):
    """ Run a single-body MetSim simulation (no erosion or fragmentation) from the given point down to the
        speed where ablation is taken to stop, or to SIM_HT_MIN.

        MetSim follows the body in 3D, in the ground-fixed frame of the initial point, with gravity and the
        Coriolis acceleration in its velocity (Constants.gravity_3d) and with the winds of the atmosphere profile
        if there are any, so the gravity turn and drop come out of the same integration as the drag. Its default
        straight path with a negligible drop runs through thinner air than the real, curving one, and on long
        shallow paths ends hundreds of metres further along.

    Arguments:
        mass: [float] Initial mass (kg).
        density: [float] Bulk density (kg/m^3).
        lat: [float] Geodetic latitude of the initial point (deg), for the MSIS atmosphere and the Coriolis
            acceleration.
        lon: [float] Longitude of the initial point for the MSIS atmosphere (deg).
        jd: [float] Julian date for the MSIS atmosphere.
        ht_beg: [float] Initial height (m).
        v_init: [float] Initial speed relative to the ground (m/s). With winds, MetSim works out the speed
            relative to the air.
        entry_angle: [float] Elevation of the radiant at the initial point (deg).
        gamma_a: [float] Not used, Gamma*A is fixed (see below). Kept for the callers.

    Keyword arguments:
        v_kill: [float] Speed relative to the air where the simulation stops (m/s). 3000 by default.
        atm_profile: [AtmosphereProfile] Take the air density, and the winds unless its use_winds is False,
            from this profile. None by default, which uses the MSIS model.
        radiant_azimuth: [float] Azimuth of the radiant at the initial point (deg, +E of due N).

    Return:
        sr: [SimulationResults] sr.frag_main also holds the 3D displacement (px, py, pz) and velocity relative
            to the ground (vx, vy, vz) in the east-north-up frame of the initial point (m, m/s).
    """

    # Init simulation constants
    const = Constants()


    # Set minimum simulation height
    const.h_kill = SIM_HT_MIN

    # Set minimum simulation speed, where ablation is taken to stop (m/s)
    const.v_kill = v_kill


    # Set meteoroid parameters
    const.m_init = mass
    const.v_init = v_init
    const.h_init = ht_beg
    const.rho = density
    #const.gamma = 1.0
    #const.shape_factor = 1.21

    # Gamma*A is fixed on purpose, not taken from gamma_a: with the same Gamma*A as the dynamic mass, the
    #   simulated velocity and path would not depend on gamma_a at all (only the mass scales as Gamma*A^3)
    const.shape_factor = 1.21
    const.gamma = 0.7
    

    # Ablation coeff of chondritic material
    const.sigma = 0.005/1e6

    # Direction of the radiant at the initial point
    const.zenith_angle = np.radians(90 - entry_angle)
    const.radiant_azimuth = np.radians(radiant_azimuth)

    # Follow the body in 3D with gravity and the Coriolis acceleration in its velocity
    const.gravity_3d = True
    const.latitude = np.radians(lat)

    # Use Borovicka 2020 luminous efficiency (not really used here)
    const.lum_eff_type = 7
    const.P_0m = 1210

    # Disable erosion and disruption (single-body only)
    const.erosion_on = False
    const.erosion_coeff = 0
    const.disruption_on = False
    const.fragmentation_on = False
    


    # Fit the atmosphere density polynomial, to the MSIS model or to the given atmosphere profile. The simulation
    #   only descends from its starting height, and a 7th order polynomial fitted up to 180 km misses the
    #   stratosphere by 10-30%, so it is fitted over those heights only
    if atm_profile is None:

        # The location is given in degrees, while fitAtmPoly() takes it in radians
        const.dens_co = fitAtmPoly(np.radians(lat), np.radians(lon), const.h_kill, ht_beg, jd)

    else:
        const.dens_co, _ = atm_profile.fitPoly(const.h_kill, ht_beg)

        # MetSim uses the profile's winds at each height
        if atm_profile.use_winds:
            const.wind_profile = atm_profile

    # Run the simulation
    frag_main, results_list, wake_results = runSimulation(const)

    sr = SimulationResults(const, frag_main, results_list, wake_results)

    return sr




def interpolateHtVsTimeLen(traj, sample_step=0.1, show_plots=False):


    # Set begin and end heights    
    beg_ht = traj.rbeg_ele/1000
    end_ht = traj.rend_ele/1000


    if show_plots:
        fig, (ax_ht, ax_len) = plt.subplots(ncols=2, sharey=True)


    # Convert heights to meters
    beg_ht *= 1000
    end_ht *= 1000
    sample_step *= 1000

    ### Fit time vs. height

    time_data = []
    height_data = []
    len_data = []

    for obs in traj.observations:

        time_data += obs.time_data.tolist()
        height_data += obs.model_ht.tolist()
        len_data += obs.state_vect_dist.tolist()

        if show_plots:
            
            # Plot the station data
            ax_ht.scatter(obs.time_data, obs.model_ht/1000, label=obs.station_id, marker='x', zorder=3)
            ax_len.scatter(obs.state_vect_dist/1000, obs.model_ht/1000, label=obs.station_id, marker='x', zorder=3)


    height_data = np.array(height_data)
    time_data = np.array(time_data)
    len_data = np.array(len_data)

    # Sort the arrays by decreasing time
    arr_sort_indices = np.argsort(time_data)[::-1]
    height_data = height_data[arr_sort_indices]
    len_data = len_data[arr_sort_indices]
    time_data = time_data[arr_sort_indices]


    # Plot the non-smoothed time vs. height
    # if show_plots:
    #   plt.scatter(time_data, height_data/1000, label='Data')


    # Apply Savitzky-Golay to smooth out the height change
    height_data = scipy.signal.savgol_filter(height_data, 21, 5)

    if show_plots:

        ax_ht.scatter(time_data, height_data/1000, label='Savitzky-Golay filtered', marker='+', zorder=3)
        ax_len.scatter(len_data/1000, height_data/1000, label='Savitzky-Golay filtered', marker='+', zorder=3)


    # Sort the arrays by increasing heights (needed for interpolation)
    arr_sort_indices = np.argsort(height_data)
    height_data = height_data[arr_sort_indices]
    len_data = len_data[arr_sort_indices]
    time_data = time_data[arr_sort_indices]


    # Interpolate height vs. time
    ht_vs_time_interp = scipy.interpolate.PchipInterpolator(height_data, time_data)

    # Interpolate height vs. length
    ht_vs_len_interp = scipy.interpolate.PchipInterpolator(height_data, len_data)


    # Plot the interpolation
    if show_plots:

        ht_arr = np.linspace(np.min(height_data), np.max(height_data), 1000)
        time_arr = ht_vs_time_interp(ht_arr)
        len_arr = ht_vs_len_interp(ht_arr)


        ax_ht.plot(time_arr, ht_arr/1000, label='Interpolation', zorder=3)
        ax_len.plot(len_arr/1000, ht_arr/1000, label='Interpolation', zorder=3)


        ax_ht.legend()


        ax_ht.set_xlabel('Time (s)')
        ax_ht.set_ylabel('Height (km)')
        ax_len.set_xlabel('Length (km)')

        ax_ht.grid()
        ax_len.grid()

        plt.show()

    ###

    return ht_vs_time_interp, ht_vs_len_interp



def pointOnTrajectory(traj, length, t):
    """ Compute the ECI coordinates of a point on the fitted trajectory, including the gravity drop.

        The point lies at the given length from the state vector along the radiant, displaced by the gravity
        drop the solver models from the fitted line (see Trajectory.applyGravityDrop). Only the component of
        the drop perpendicular to the line is applied, as in the solver's model points, where the along-track
        part is absorbed by the length.

    Arguments:
        traj: [Trajectory] Solved trajectory.
        length: [float] Length from the state vector along the trajectory (m), positive towards the end.
        t: [float] Time of the point relative to the trajectory reference time (s).

    Return:
        [ndarray] ECI coordinates of the point (m).
    """

    P = traj.state_vect_mini - length*traj.radiant_eci_mini

    # Solutions without the gravity correction have a straight fitted line and nothing to add
    if getattr(traj, 'gravity_correction', True):

        # The solver measures the drop from the first observation of the meteor
        t0 = min(obs.time_data[0] for obs in traj.observations)

        drop = applyGravityDrop(P, t - t0, vectMag(traj.state_vect_mini), getattr(traj, 'gravity_factor', 1.0), \
            getattr(traj, 'v0z', None) or 0.0) - P

        # Keep only the component perpendicular to the fitted line
        P = P + drop - np.dot(drop, traj.radiant_eci_mini)*traj.radiant_eci_mini

    return P


def computeFragEndParams(traj, dyn_mass, density, hend, vend, gamma_a, v_kill=3000, atm_profile=None):
    """ Propagate a fragment of the given dynamic mass from the evaluation point down to the speed where
        ablation is taken to stop (3 km/s by default) with the single-body ablation model, and compute where
        and in which direction it ends.

        The simulation starts at the evaluation point, on the solver's trajectory with its gravity drop, moving
        relative to the ground in the direction evalPointState() gives. MetSim follows it in 3D with gravity in
        its velocity and, with the winds of an atmosphere profile, with the wind of each height, so drag and
        ablation act on the velocity relative to the air. Its final position and velocity, in the east-north-up
        frame of the evaluation point fixed to the ground, are taken to geographic coordinates and to the local
        horizon of the final point.

        The returned final azimuth and elevation are those of the apparent ground-fixed radiant (epoch of date) at
        the final point, as traj.orbit.azimuth_apparent_norot and elevation_apparent_norot are at the reference
        point. They are the inputs a dark flight computation needs.

    Arguments:
        traj: [Trajectory] Solved trajectory.
        dyn_mass: [float] Dynamic mass at the evaluation point (kg).
        density: [float] Bulk density of the meteoroid (kg/m^3).
        hend: [float] Height of the evaluation point (m).
        vend: [float] Speed relative to the ground at the evaluation point (m/s), see groundSpeed().
        gamma_a: [float] Product of the drag coefficient and the shape factor used for the dynamic mass. Not
            used by the simulation itself, see runFragSim().

    Keyword arguments:
        v_kill: [float] Speed relative to the air at which the simulation stops (m/s). 3000 by default.
        atm_profile: [AtmosphereProfile] Atmosphere for the simulation. None by default, which uses the MSIS
            model.

    Return:
        (sr, final_mass, final_lat, final_lon, final_ele, final_azim, final_elev, final_time, final_vel): [tuple]
            sr: [SimulationResults] Results of the fragment simulation.
            final_mass: [float] Mass at the final point (kg).
            final_lat: [float] Latitude of the final point (deg, +N).
            final_lon: [float] Longitude of the final point (deg, +E).
            final_ele: [float] Height of the final point (km).
            final_azim: [float] Azimuth of the ground-fixed radiant at the final point (deg, +E of due N).
            final_elev: [float] Elevation of the ground-fixed radiant at the final point (deg).
            final_time: [float] Time of the final point after traj.jdt_ref (s).
            final_vel: [float] Speed at the final point relative to the ground (m/s).
    """

    meas_time, meas_len, eval_eci, eval_jd, eval_lat, eval_lon, eval_azim, eval_elev = \
        evalPointState(traj, hend)

    # Run the simulation from the evaluation point until ablation stops
    sr = runFragSim(dyn_mass, density, np.degrees(eval_lat), np.degrees(eval_lon), eval_jd, hend, vend, \
        np.degrees(eval_elev), gamma_a, v_kill=v_kill, atm_profile=atm_profile, \
        radiant_azimuth=np.degrees(eval_azim))

    frag = sr.frag_main

    # Extract the final height
    final_ht = 0
    if len(sr.brightest_height_arr) > 2:
        final_ht = sr.brightest_height_arr[-2]

    # Extract the total simulation time
    final_time = np.max(sr.time_arr)
    total_len = meas_len + frag.length
    total_time = meas_time + final_time
    final_jd = eval_jd + final_time/86400


    ### Take MetSim's final state to the final point ###

    # The frame of the simulation is fixed to the ground, so its displacement is added to the evaluation point
    #   where the ground was at the evaluation time, and then carried by the Earth's rotation to the final time.
    #   In the ECI frame of a given time, the east-north-up axes of a point are those of a longitude equal to its
    #   local sidereal time
    eval_lst = np.radians(jd2LST(eval_jd, np.degrees(eval_lon))[0])
    z_axis, earth_turn = np.array([0.0, 0.0, 1.0]), EARTH_ROTATION_RATE*final_time
    final_eci = rotateVector(eval_eci + np.array(enu2ECEF(eval_lat, eval_lst, frag.px, frag.py, frag.pz)), \
        z_axis, earth_turn)
    final_lat, final_lon, final_ele = cartesian2Geo(final_jd, *final_eci)

    # The final velocity relative to the ground, in the local east-north-up frame of the final point
    vel_eci = rotateVector(np.array(enu2ECEF(eval_lat, eval_lst, frag.vx, frag.vy, frag.vz)), z_axis, earth_turn)
    vel_end = np.array(ecef2ENU(final_lat, np.radians(jd2LST(final_jd, np.degrees(final_lon))[0]), *vel_eci))
    final_vel = vectMag(vel_end)
    final_azim = np.arctan2(-vel_end[0], -vel_end[1])%(2*np.pi)
    final_elev = np.arcsin(-vel_end[2]/final_vel)

    ###


    print("  final mass     = {:.3f} kg".format(frag.m))
    print("  final vel      = {:.3f} km/s".format(final_vel/1000))
    print("  final ht (sim) = {:.3f} km".format(final_ht/1000))
    print("  total len      = {:.3f} km".format(total_len/1000))
    print("  total time     = {:.3f} s".format(total_time))
    print("  final lat      = {:.5f} deg".format(np.degrees(final_lat)))
    print("  final lon      = {:.5f} deg".format(np.degrees(final_lon)))
    print("  final ht       = {:.3f} km".format(final_ele/1000))
    print("  final azim     = {:.5f} deg".format(np.degrees(final_azim)))
    print("  final elev     = {:.5f} deg".format(np.degrees(final_elev)))


    return sr, frag.m, np.degrees(final_lat), np.degrees(final_lon), final_ele/1000, \
        np.degrees(final_azim), np.degrees(final_elev), total_time, final_vel



def _robust_linear_fit(x, y, p0=(1.0, 1.0), loss='soft_l1', **kwargs):
    """ Fit a linear model y = m*x + c using robust least-squares optimization.
        Covariance is estimated from the linear model Jacobian and the robust (MAD) residual variance.

    Arguments:
        x: [ndarray] Independent variable.
        y: [ndarray] Dependent variable.
        p0: [tuple(float, float)] Initial guess for slope and intercept.
        loss: [str] Loss function for robust fitting (passed to least_squares).
        **kwargs: Additional arguments passed to least_squares (e.g. f_scale).

    Return:
        popt: [ndarray] Optimal parameters [m, c].
        pcov: [ndarray] 2x2 covariance matrix of the fit.
        perr: [ndarray] Standard deviations of parameters [sigma_m, sigma_c].
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    def residualFunc(params, xx, yy):
        m, c = params
        return yy - lineFunc(xx, m, c)

    res = scipy.optimize.least_squares(
        residualFunc, p0, args=(x, y), loss=loss, jac='2-point', **kwargs
    )
    if not res.success:
        raise RuntimeError("least_squares did not find a solution: " + res.message)

    popt = res.x

    # Use the Jacobian of the linear model, not res.jac, which is reweighted by the robust loss
    J = np.column_stack((x, np.ones_like(x)))

    # The normal matrix is singular when all points share the same time, in which case no slope is defined
    if np.ptp(x) == 0:
        raise RuntimeError("All points have the same time; the velocity slope and its covariance are "
            "undefined.")

    pcov = np.linalg.inv(J.T @ J)

    r_raw = y - lineFunc(x, *popt)

    # Use robust variance estimation (MAD), or the standard deviation if MAD is zero
    mad = np.median(np.abs(r_raw - np.median(r_raw)))
    resid_std = 1.4826*mad if mad > 0 else np.std(r_raw, ddof=1)

    s_sq = resid_std**2
    pcov = pcov*s_sq

    perr = np.sqrt(np.diag(pcov))
    
    return popt, pcov, perr


def fitVelocity(time_data, vel_data, p0=(1.0, 1.0), loss='soft_l1', sigma_clip=3.0):
    """ Perform a robust velocity fit and iterative outlier rejection.
        Uses robust least-squares (soft L1 loss) both before and after sigma-clipping.

    Arguments:
        time_data: [ndarray] Time data points.
        vel_data: [ndarray] Velocity data points.
        p0: [tuple(float, float)] Initial parameter guess (slope, intercept).
        loss: [str] Loss function for robust fitting.
        sigma_clip: [float] Outlier rejection threshold, in units of the robust (MAD) residual standard
            deviation.

    Return:
        popt: [ndarray] Final optimal parameters [m, c].
        pcov: [ndarray] 2x2 covariance matrix of the final fit.
        perr: [ndarray] Standard deviations of parameters [sigma_m, sigma_c].
        mask: [ndarray(bool)] Boolean mask of inlier points used in final fit.
    """

    # Initial robust fit
    popt, pcov, perr = _robust_linear_fit(time_data, vel_data, p0=p0, loss=loss)

    yhat = lineFunc(np.asarray(time_data, dtype=float), *popt)
    resid = np.asarray(vel_data, dtype=float) - yhat

    # Compute robust std dev of residuals using MAD
    mad = np.median(np.abs(resid - np.median(resid)))
    resid_std = 1.4826*mad if mad > 0 else np.std(resid, ddof=1)
    if not np.isfinite(resid_std) or resid_std == 0:
        resid_std = np.finfo(float).eps

    # Outlier rejection - Create mask for points within sigma_clip * resid_std
    mask = np.abs(resid) < sigma_clip*resid_std

    print("Outlier rejection: kept {}/{} points.".format(np.count_nonzero(mask), len(mask)))
    print("Residual std dev: {:.3f}".format(resid_std))
    print(mask)
    print("Residuals:", resid)

    if np.count_nonzero(mask) < 2:
        return popt, pcov, perr, np.ones_like(mask, dtype=bool)

    # Final robust fit on clipped data
    popt2, pcov2, perr2 = _robust_linear_fit(
        np.asarray(time_data)[mask],
        np.asarray(vel_data)[mask],
        p0=popt,
        loss=loss,
        f_scale=resid_std
    )

    return popt2, pcov2, perr2, mask



def _ensureTrajDefaults(traj):
    """ Fill in the trajectory attributes that Pickling.loadPickle() normally back-fills on load, but which
        are not set on the individual per-run trajectories stored inside a Monte Carlo uncertainties pickle
        (loadPickle's post-processing only touches the top-level unpickled object).

    Arguments:
        traj: [Trajectory] Trajectory (e.g. one Monte Carlo realization) to patch in place.

    Return:
        traj: [Trajectory] The same object, with defaults filled in.
    """

    if hasattr(traj, 'orbit'):
        try:
            traj.orbit.fixMissingParameters()
        except Exception:
            pass

    if not hasattr(traj, 'gravity_factor'):
        traj.gravity_factor = 1.0

    if not hasattr(traj, 'v0z'):
        traj.v0z = 0.0

    return traj



def mcUncertaintiesPath(traj_path):
    """ Path of the Monte Carlo uncertainties pickle that the WMPL trajectory solver saves next to the main
        trajectory pickle, as "Monte Carlo/<file_name>_mc_uncertainties.pickle" (see Trajectory.run() in
        wmpl/Trajectory/Trajectory.py). The file may not exist.

    Arguments:
        traj_path: [str] Path to the main "*_trajectory.pickle" file.

    Return:
        [str] Absolute path of the Monte Carlo uncertainties pickle.
    """

    dir_path = os.path.dirname(os.path.abspath(traj_path))
    file_name = os.path.basename(traj_path)

    if file_name.endswith('_trajectory.pickle'):
        file_name_core = file_name[:-len('_trajectory.pickle')]
    else:
        file_name_core = os.path.splitext(file_name)[0]

    return os.path.join(dir_path, 'Monte Carlo', file_name_core + '_mc_uncertainties.pickle')



def findMCUncertaintiesPickle(traj_path):
    """ Locate the Monte Carlo uncertainties pickle that the WMPL trajectory solver saves next to the main
        trajectory pickle (see mcUncertaintiesPath()), and return the individual Monte Carlo trajectory
        realizations it holds.

        Unlike traj.uncertainties on the main pickle, whose mc_traj_list is stripped to save space, this file
        keeps every accepted Monte Carlo run as a full, independently re-solved Trajectory object.

    Arguments:
        traj_path: [str] Path to the main "*_trajectory.pickle" file.

    Return:
        mc_traj_list: [list or None] List of Trajectory objects, one per accepted Monte Carlo run, or None
            if no Monte Carlo uncertainties pickle was found next to the trajectory pickle, or it did not
            contain any runs.
    """

    mc_path = mcUncertaintiesPath(traj_path)

    if not os.path.isfile(mc_path):
        return None

    traj_unc = loadPickle(*os.path.split(mc_path))

    mc_traj_list = getattr(traj_unc, 'mc_traj_list', None)

    if not mc_traj_list:
        return None

    return mc_traj_list



def dynMassFromTraj(traj, ht_max, ht_min, eval_point, bulk_density, gamma_a, max_vel, mass_max, \
    sigma_clip=3.0, atm_profile=None):
    """ Compute the dynamic mass at the evaluation point for a single trajectory solution, following the
        same steps as the dynamic mass fit in __main__ (velocity vs. time fit in the height window, then
        dynamic mass at the evaluation point). Used for the individual Monte Carlo trajectory realizations.

    Arguments:
        traj: [Trajectory] Trajectory (or Monte Carlo realization) to fit.
        ht_max: [float] Top height of the fitting window (km).
        ht_min: [float] Bottom height of the fitting window (km).
        eval_point: [float] Point where to evaluate the dynamic mass (0 = ht_min, 1 = ht_max).
        bulk_density: [float] Bulk density of the meteoroid (kg/m^3).
        gamma_a: [float] Product of the drag coefficient and the shape factor.
        max_vel: [float] Maximum velocity to consider (m/s).
        mass_max: [float] Maximum dynamic mass allowed (kg), used to clip runaway values.
        sigma_clip: [float] Outlier rejection threshold for the velocity fit (see fitVelocity()).
        atm_profile: [AtmosphereProfile] Atmosphere for the dynamic mass. None uses the MSIS model.

    Return:
        [dict or None] None if there isn't enough data in the height window to fit a line. Otherwise a dict
            with keys: decel, decel_std, vel_eval (relative to the ground), ht_eval, time_eval, dyn_mass, popt,
            pcov (the velocity fit [slope, intercept] and its covariance), azim_eval, elev_eval (the
            ground-fixed radiant at the evaluation point, see evalPointState(), or None without winds).
    """

    vel_data, ht_data, time_data = [], [], []

    for obs in traj.observations:

        ignored = obs.ignore_list[1:] > 0

        # Speeds relative to the ground, which the drag depends on, from the solver's ECI speeds
        vel = groundSpeed(traj, obs.model_eci[1:][~ignored], obs.velocities[1:][~ignored])
        ht = obs.meas_ht[1:][~ignored]
        t = obs.time_data[1:][~ignored]

        vel_filter = (vel > 0) & (vel < min(max_vel, 73_000))

        vel_data += vel[vel_filter].tolist()
        ht_data += ht[vel_filter].tolist()
        time_data += t[vel_filter].tolist()

    vel_data = np.array(vel_data)
    ht_data = np.array(ht_data)
    time_data = np.array(time_data)

    # Sort by height, matching the ordering used by the nominal (non-MC) fit
    ht_sort = np.argsort(ht_data)
    vel_data, time_data, ht_data = vel_data[ht_sort], time_data[ht_sort], ht_data[ht_sort]

    ht_filter = (ht_data/1000 >= ht_min) & (ht_data/1000 <= ht_max)
    vel_data = vel_data[ht_filter]
    ht_data = ht_data[ht_filter]
    time_data = time_data[ht_filter]

    if len(vel_data) < 2:
        return None

    # Start the robust fit from the least-squares line; from (1, 1) it often runs out of function evaluations
    try:
        popt, pcov, perr, _ = fitVelocity(time_data, vel_data, p0=np.polyfit(time_data, vel_data, 1), \
            loss='soft_l1', sigma_clip=sigma_clip)
    except RuntimeError:
        return None

    decel = -popt[0]
    decel_std = perr[0]

    time_eval = np.min(time_data) + eval_point*(np.max(time_data) - np.min(time_data))
    vel_eval = lineFunc(time_eval, *popt)

    try:
        ht_vs_time_interp, _ = interpolateHtVsTimeLen(traj)
        ht_eval = scipy.optimize.brentq(lambda h: ht_vs_time_interp(h) - time_eval, \
            *ht_vs_time_interp.x[[0, -1]])

        # The direction of motion is only needed to add the wind
        azim_eval = elev_eval = None
        if _windENU(atm_profile, ht_eval) is not None:
            azim_eval, elev_eval = evalPointState(traj, ht_eval, ht_vs_time_interp=ht_vs_time_interp)[6:]

    except Exception:
        return None

    if decel < 0:
        decel = 0

    dyn_mass = dynamicMass(bulk_density, traj.rend_lat, traj.rend_lon, ht_eval, traj.jdt_ref, \
        _airSpeed(atm_profile, ht_eval, vel_eval, azim_eval, elev_eval), decel, gamma=1.0, \
        shape_factor=gamma_a, atm_dens=_airDensity(atm_profile, ht_eval))
    dyn_mass = np.clip(dyn_mass, 0, mass_max)

    return {
        'decel': decel, 'decel_std': decel_std, 'vel_eval': vel_eval, 'ht_eval': ht_eval, \
        'time_eval': time_eval, 'dyn_mass': dyn_mass, 'popt': popt, 'pcov': pcov, 'azim_eval': azim_eval, \
        'elev_eval': elev_eval
    }



MC_FINAL_KEYS = ['final_mass', 'final_lat', 'final_lon', 'final_ele', 'final_azim', 'final_elev', \
    'final_decel', 'final_vel', 'final_time']


def _mcRealization(traj_mc, traj_index, seed, label, ht_max, ht_min, eval_point, bulk_density, gamma_a, max_vel, \
    mass_max, sigma_clip, run_final_sim, v_kill, v_kill_sigma, density_sigma, atm_profile, msis_settings):
    """ Process one Monte Carlo trajectory realization for runMonteCarloDynMass(). It is a module-level
        function so it can be sent to the worker processes.

    Arguments:
        traj_mc: [Trajectory] Monte Carlo trajectory realization.
        traj_index: [int] Index of the realization in the solver's list of Monte Carlo trajectories.
        seed: [int] Random seed for the draws of this realization.
        label: [str] Realization label used in the printouts.
        ht_max, ht_min, eval_point, bulk_density, gamma_a, max_vel, mass_max, sigma_clip, run_final_sim,
            v_kill, v_kill_sigma, density_sigma, atm_profile: See runMonteCarloDynMass().
        msis_settings: [tuple] MSIS options of the main process, see _msisSettings().

    Return:
        [dict or None] The values of this realization (see runMonteCarloDynMass()), or None if no velocity fit
            could be made.
    """

    traj_mc = _ensureTrajDefaults(traj_mc)

    # Use the MSIS model of the main process
    _setMSISSettings(msis_settings)

    fit_res = dynMassFromTraj(traj_mc, ht_max, ht_min, eval_point, bulk_density, gamma_a, max_vel, \
        mass_max, sigma_clip=sigma_clip, atm_profile=atm_profile)

    if fit_res is None:
        print("MC realization {:s} skipped: no valid velocity fit or evaluation height in the height "
            "window.".format(label))
        return None

    rng = np.random.default_rng(seed)

    # Draw the velocity fit parameters from the fit covariance of this realization
    slope, intercept = rng.multivariate_normal(fit_res['popt'], fit_res['pcov'])
    decel = max(-slope, 0)
    vel_eval = lineFunc(fit_res['time_eval'], slope, intercept)

    # Draw the bulk density, redrawing non-positive values
    density = bulk_density
    if density_sigma > 0:
        density = rng.normal(bulk_density, density_sigma)
        while density <= 0:
            density = rng.normal(bulk_density, density_sigma)

    vel_air = _airSpeed(atm_profile, fit_res['ht_eval'], vel_eval, fit_res['azim_eval'], fit_res['elev_eval'])
    dyn_mass = dynamicMass(density, traj_mc.rend_lat, traj_mc.rend_lon, fit_res['ht_eval'], \
        traj_mc.jdt_ref, vel_air, decel, gamma=1.0, shape_factor=gamma_a, \
        atm_dens=_airDensity(atm_profile, fit_res['ht_eval']))
    dyn_mass = np.clip(dyn_mass, 0, mass_max)

    res = {
        'dyn_mass': dyn_mass, 'decel': decel, 'vel_eval': vel_eval, 'dyn_mass_geom': fit_res['dyn_mass'], \
        'decel_geom': fit_res['decel'], 'ht_eval': fit_res['ht_eval'], 'time_eval': fit_res['time_eval'], \
        'density': density, 'jdt_ref': traj_mc.jdt_ref, 'traj_index': traj_index
    }

    if not run_final_sim:
        return res

    # Draw the speed where ablation stops, redrawing non-positive values
    v_kill_nominal = v_kill
    if v_kill_sigma > 0:
        v_kill = rng.normal(v_kill_nominal, v_kill_sigma)
        while v_kill <= 0:
            v_kill = rng.normal(v_kill_nominal, v_kill_sigma)

    res['v_kill'] = v_kill

    # Keep the realization in the dynamic mass statistics even if its end point cannot be simulated
    final_vals = [np.nan]*len(MC_FINAL_KEYS)

    if vel_air <= v_kill:
        print("MC realization {:s}: no final simulation, the evaluation velocity is already below "
            "{:.2f} km/s.".format(label, v_kill/1000))

    else:
        try:
            sr_mc, final_mass, final_lat, final_lon, final_ele, final_azim, final_elev, final_time, \
                final_vel = \
                computeFragEndParams(traj_mc, dyn_mass, density, fit_res['ht_eval'], vel_eval, gamma_a, \
                    v_kill=v_kill, atm_profile=atm_profile)

            final_decel = _endDecel(sr_mc)

            final_vals = [final_mass, final_lat, final_lon, final_ele, final_azim, final_elev, final_decel, \
                final_vel, final_time]

        except Exception as e:
            print("MC realization {:s}: the final simulation failed ({}).".format(label, e))

    res.update(zip(MC_FINAL_KEYS, final_vals))

    return res



def runMonteCarloDynMass(mc_traj_list, ht_max, ht_min, eval_point, bulk_density, gamma_a, max_vel, \
    mass_max, sigma_clip, n_samples=None, run_final_sim=True, rng_seed=None, cores=None, v_kill=3000, \
    v_kill_sigma=0, density_sigma=0, atm_profile=None):
    """ Propagate the uncertainties through the dynamic mass fit (and, optionally, through the final fragment
        simulation down to the speed where ablation stops) over the WMPL trajectory solver's Monte Carlo
        realizations.

        The solver perturbs only the lines of sight to get each realization's radiant and state vector, and
        then computes the point velocities and heights from the original, un-noised observations (see
        Trajectory.run(), _mc_run). Every realization therefore fits the same measurement scatter, and the
        spread between realizations carries only the geometric (radiant, state vector, timing offset)
        uncertainty. The velocity fit uncertainty is added by drawing the slope and intercept of each
        realization's fit from its covariance, so the propagated distribution holds both terms.

    Arguments:
        mc_traj_list: [list] List of Trajectory objects, one per Monte Carlo realization (see
            findMCUncertaintiesPickle()).
        ht_max, ht_min, eval_point, bulk_density, gamma_a, max_vel, mass_max, sigma_clip: Same meaning as in
            dynMassFromTraj() / computeFragEndParams().
        n_samples: [int or None] Maximum number of realizations to use. If mc_traj_list is larger, a random
            subset of this size is drawn without replacement. None uses all realizations.
        run_final_sim: [bool] Also run the final fragment simulation (down to v_kill) for every realization,
            to get the final mass, position and radiant uncertainty. Slower.
        rng_seed: [int or None] Random seed used for subsampling and for all the draws. Each realization gets
            its own seed from it, so the results do not depend on the number of cores.
        cores: [int or None] Number of processes used to run the realizations in parallel. None uses all
            available cores, 1 runs them serially.
        v_kill: [float] Nominal speed where ablation is taken to stop and the final simulation ends (m/s).
        v_kill_sigma: [float] Standard deviation of v_kill (m/s). If above 0, every realization draws its own
            v_kill from a normal distribution, redrawing non-positive values.
        density_sigma: [float] Standard deviation of the bulk density (kg/m^3). If above 0, every realization
            draws its own density from a normal distribution centred on bulk_density, redrawing non-positive
            values, and uses it for both its dynamic mass and its final simulation.
        atm_profile: [AtmosphereProfile] Atmosphere for the dynamic mass and the final simulation. None uses
            the MSIS model.

    Return:
        results: [dict of ndarray] One entry per successfully fitted realization:
            dyn_mass, decel, vel_eval: with the velocity fit parameters drawn from their covariance.
            density: bulk density of the realization.
            jdt_ref: reference Julian date of the realization's trajectory.
            traj_index: index of the realization in mc_traj_list.
            dyn_mass_geom, decel_geom: with the best-fit parameters and the nominal density, i.e. the
                geometric uncertainty only.
            ht_eval, time_eval: evaluation point of the realization.
            final_mass, final_lat, final_lon, final_ele, final_azim, final_elev, final_decel, final_vel,
                final_time: if run_final_sim is True, NaN where the simulation could not be run. final_vel
                is the speed at the final point (m/s), final_time its time after jdt_ref (s).
            v_kill: if run_final_sim is True, the speed where the simulation of the realization stops.
    """

    rng = np.random.default_rng(rng_seed)

    traj_indices = list(range(len(mc_traj_list)))

    if (n_samples is not None) and (len(mc_traj_list) > n_samples):
        traj_indices = [int(i) for i in rng.choice(len(mc_traj_list), size=n_samples, replace=False)]

    traj_samples = [mc_traj_list[i] for i in traj_indices]

    n_total = len(traj_samples)

    # One seed per realization, so the draws do not depend on how the realizations are split among cores
    seeds = rng.integers(0, 2**63 - 1, size=n_total)

    domain = [[traj_mc, traj_index, seed, "{:d}/{:d}".format(i + 1, n_total), ht_max, ht_min, eval_point, bulk_density, \
        gamma_a, max_vel, mass_max, sigma_clip, run_final_sim, v_kill, v_kill_sigma, \
        density_sigma, atm_profile, _msisSettings()] for i, (traj_mc, traj_index, seed) \
        in enumerate(zip(traj_samples, traj_indices, seeds))]

    # Do not start more processes than there are realizations
    if cores is None:
        cores = multiprocessing.cpu_count()
    cores = max(1, min(cores, n_total))

    print("Running {:d} Monte Carlo realizations on {:d} core(s)...".format(n_total, cores))

    res_list = [res for res in domainParallelizer(domain, _mcRealization, cores=cores) if res is not None]

    keys = ['dyn_mass', 'decel', 'vel_eval', 'dyn_mass_geom', 'decel_geom', 'ht_eval', 'time_eval', 'density', \
        'jdt_ref', 'traj_index']
    if run_final_sim:
        keys += MC_FINAL_KEYS + ['v_kill']

    results = {key: np.array([res[key] for res in res_list], dtype=float) for key in keys}

    print()
    print("Monte Carlo propagation: {:d}/{:d} realizations fitted.".format(len(results['dyn_mass']), n_total))

    if run_final_sim:
        print("Final simulation run for {:d} of them.".format(np.count_nonzero(~np.isnan(results['final_mass']))))

    return results



def printMCPercentiles(results, ci=95.0):
    """ Print the median and confidence interval of every array in a Monte Carlo results dict (see
        runMonteCarloDynMass()).
    """

    lo_pct = (100 - ci)/2
    hi_pct = 100 - lo_pct

    for key, arr in results.items():

        arr = arr[~np.isnan(arr)]

        if len(arr) == 0:
            continue

        med = np.median(arr)
        lo, hi = np.percentile(arr, [lo_pct, hi_pct])

        print("  {:12s} median = {:10.4f}, {:.0f}% CI = [{:10.4f}, {:10.4f}]".format(key, med, ci, lo, hi))



if __name__ == "__main__":

    import argparse

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Compute the final dynamic mass of a fireball by defining the height range of the final portion where the mass is measured. A simulation is run to propagate the fragment down to the speed where ablation stops (3 km/s by default, see --vkill) and estimate the final mass and location.")

    arg_parser.add_argument('traj_path', metavar='TRAJ_PATH', type=str, \
        help="Path to the trajectory pickle file.")

    arg_parser.add_argument('ht_max', metavar='HT_MAX', \
        help='Top height in km taken to compute the dynamic mass (it should be close to the end of the fireball). If negative, e.g. -3, the last 3 km will be taken.', \
        type=float)

    arg_parser.add_argument('ht_min', metavar='HT_MIN', \
        help='Bottom height in km taken to compute the dynamic mass. If set to -1, the last observed point will be taken.', \
        type=float)

    arg_parser.add_argument('-d', '--dens', metavar='DENS', \
        help='Bulk density in kg/m^3 used to compute the final dynamic mass. Default is 3500 kg/m^3.', \
        type=float, default=3500)

    arg_parser.add_argument('--dens_sigma', metavar='DENS_SIGMA', type=float, default=0.0, \
        help='Standard deviation in kg/m^3 of the bulk density, used with --mc: every Monte Carlo realization '
        'draws its own density from a normal distribution centred on --dens (non-positive draws are '
        'redrawn) and uses it for both its dynamic mass and its final simulation. Default is 0, i.e. a fixed '
        'density.')

    arg_parser.add_argument('-g', '--ga', metavar='GAMMA_A', \
        help='The product of the drag coefficient Gamma and the shape coefficient A. Used for computing the dynamic mass. Default is 0.55.', \
        type=float, default=0.55)

    arg_parser.add_argument('-e', '--eval', metavar='EVAL_PT', \
        help='Point where to evaluate the dynamic mass (0 = ht_min, 1 = ht_max). Default is 0.5.', \
        type=float, default=0.5)
    
    arg_parser.add_argument('--sigma_clip', metavar='SIGMA_CLIP', \
        help='Sigma threshold for outlier rejection in the velocity fit. Default is 3.0.', \
        type=float, default=3.0)
    
    arg_parser.add_argument('--maxvel', metavar='MAX_VEL', \
        help='Maximum velocity in km/s to consider in the height window. Used to remove outliers. Default and upper limit is 73 km/s.', \
        type=float, default=None)
    
    arg_parser.add_argument('--maxmass', metavar='MAX_MASS', \
                            help='Maximum mass in kg for the dynamic mass measurements. Used to avoid inf values. Default is 50 kg.', \
                            type=float, default=50)

    arg_parser.add_argument('--mc', action='store_true', \
        help="Propagate the WMPL trajectory Monte Carlo solver's uncertainties through the dynamic mass fit "
        "(and, unless --mc_no_final_sim is given, through the final fragment simulation too), by repeating "
        "the fit on every individual Monte Carlo trajectory realization saved by the solver in the "
        "'Monte Carlo/*_mc_uncertainties.pickle' file next to the trajectory pickle. Requires the trajectory "
        "to have been solved with Monte Carlo error estimation and that file to still be present.")

    arg_parser.add_argument('--mc_samples', metavar='N', type=int, default=None, \
        help='Maximum number of Monte Carlo trajectory realizations used for the uncertainty propagation. '
        'If the solver produced more, a random subset of this size is drawn without replacement. '
        'Default is to use every realization the solver produced (read from the MC uncertainties pickle).')

    arg_parser.add_argument('--mc_no_final_sim', action='store_true', \
        help='When propagating Monte Carlo uncertainties, skip the final fragment ablation simulation (down '
        'to the kill speed) for every realization and only propagate the dynamic mass at the evaluation point. Much '
        'faster, but does not give final mass/position/radiant uncertainties.')

    arg_parser.add_argument('--mc_seed', metavar='SEED', type=int, default=None, \
        help='Random seed for subsampling the Monte Carlo realizations and drawing their velocity fit '
        'parameters. The results do not depend on --mc_cores.')

    arg_parser.add_argument('--vkill', metavar='V_KILL', type=float, default=3.0, \
        help='Speed in km/s, relative to the air, where ablation is taken to stop and the final simulation ends. '
        'Default is 3 km/s.')

    arg_parser.add_argument('--vkill_sigma', metavar='V_KILL_SIGMA', type=float, default=0.0, \
        help='Standard deviation in km/s of the kill speed, used with --mc: every Monte Carlo realization '
        'draws its own kill speed from a normal distribution centred on --vkill (non-positive draws are '
        'redrawn). Default is 0, i.e. a fixed kill speed.')

    arg_parser.add_argument('--atm_profile', metavar='PATH', type=str, default=None, \
        help='Atmosphere profile file to take the air density from, for both the dynamic mass and the final '
        'simulation, instead of the MSIS model. It is read and interpolated as OpenDarkflight does, so a dark '
        'flight run with the same file continues in the same atmosphere. It has to cover the heights from the '
        'top of the fit window down to 15 km. See --atm_profile_type for the formats.')

    arg_parser.add_argument('--atm_profile_type', metavar='TYPE', type=str, default='wrf', \
        choices=PROFILE_TYPES, \
        help="Format of the --atm_profile file, as in OpenDarkflight: 'wrf' (default; the CSV every model "
        "profile is written in), 'wyoming' (University of Wyoming radiosonde) or 'supracenter'.")

    arg_parser.add_argument('--no_winds', action='store_true', \
        help='Ignore the winds of the --atm_profile file. By default they are used: the drag depends on the '
        'velocity relative to the air, both for the dynamic mass and for the final simulation, whose end point '
        'and velocity are given relative to the ground.')

    arg_parser.add_argument('--save_pickle', metavar='PATH', nargs='?', const='', default=None, \
        help='Save the results to a pickle made only of built-in Python types, readable without wmpl, with '
        'the ejection states for a dark flight code (see wmpl.Utils.DynamicMassFitExport). If no path is '
        'given, it is saved next to the trajectory pickle as <name>_dyn_mass_fit.pickle.')

    arg_parser.add_argument('--mc_cores', metavar='CORES', type=int, default=None, \
        help='Number of CPU cores used to process the Monte Carlo realizations in parallel. Default is all '
        'available cores; 1 runs them serially.')

    # Add the atmosphere model options
    addAtmosphereArguments(arg_parser)

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    # Apply the atmosphere model options
    setAtmosphere(cml_args)

    #########################

    # INPUTS

    # Point where to evaluate the dynamic mass (0 = ht_min, 1 = ht_max)
    eval_point = cml_args.eval

    # Meteoroid density (kg/m^3)
    bulk_density = cml_args.dens

    # Gamma*A factor
    gamma_a = cml_args.ga

    # Read the maximum velocity
    max_vel = cml_args.maxvel
    if max_vel is not None:
        max_vel *= 1000
    
    else:
        max_vel = 73_000

    # Speed where ablation stops, and its spread for the Monte Carlo (m/s)
    if cml_args.vkill <= 0:
        raise ValueError("--vkill has to be positive, got {:g} km/s".format(cml_args.vkill))
    if cml_args.vkill_sigma < 0:
        raise ValueError("--vkill_sigma cannot be negative, got {:g} km/s".format(cml_args.vkill_sigma))
    v_kill = 1000*cml_args.vkill
    v_kill_sigma = 1000*cml_args.vkill_sigma

    if cml_args.dens <= 0:
        raise ValueError("--dens has to be positive, got {:g} kg/m^3".format(cml_args.dens))
    if cml_args.dens_sigma < 0:
        raise ValueError("--dens_sigma cannot be negative, got {:g} kg/m^3".format(cml_args.dens_sigma))


    # Load the trajectory
    traj = loadPickle(*os.path.split(os.path.abspath(cml_args.traj_path)))

    dir_path = os.path.dirname(cml_args.traj_path)


    # Top height (if negative, e.g. -3, the last 3 km from the bottom will be taken)
    if cml_args.ht_max < 0:
        ht_max = traj.rend_ele/1000 - cml_args.ht_max
    else:
        ht_max = cml_args.ht_max

    # Bottom height (km, if -1, take the last point)
    ht_min = cml_args.ht_min

    if ht_max < ht_min:
        raise ValueError("The min height has to be lower than the max height! ht_max = {:.2f} km, ht_min = {:.2f} km".format(ht_max, ht_min))

    # The atmosphere profile has to cover the fit window down to the bottom of the simulation
    atm_profile = None
    if cml_args.atm_profile is not None:
        atm_profile = AtmosphereProfile(cml_args.atm_profile, profile_type=cml_args.atm_profile_type)
        atm_profile.checkCoverage(SIM_HT_MIN, 1000*ht_max)
        atm_profile.use_winds = not cml_args.no_winds

        if (cml_args.atm != "00") or (cml_args.atmtime is not None):
            print("Note: --atm and --atmtime select the MSIS model, which is not used with --atm_profile")


    #################


    vel_data = []
    ht_data = []
    time_data = []



    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,6))


    # Possible markers for the plot
    markers = ['x', '+', '.', '2']

    # Generate a list of colors to use for markers
    colors = cm.viridis(np.linspace(0, 0.8, len(traj.observations)))
    
    for i, obs in enumerate(traj.observations):

        ignored = obs.ignore_list[1:] > 0


        # Speeds relative to the ground, which the drag depends on, from the solver's ECI speeds
        vel = groundSpeed(traj, obs.model_eci[1:][~ignored], obs.velocities[1:][~ignored])
        ht = obs.meas_ht[1:][~ignored]
        t = obs.time_data[1:][~ignored]

        # Only take velocities inside a reasonable range (the physical upper limit, or a lower --maxvel)
        vel_filter = (vel > 0) & (vel < min(max_vel, 73_000))

        # Filter out all data
        vel = vel[vel_filter]
        ht = ht[vel_filter]
        t = t[vel_filter]

        # Store all data
        vel_data += vel.tolist()
        ht_data += ht.tolist()
        time_data += t.tolist()

        # Plot all velocities vs height
        ax1.scatter(vel/1000, ht/1000, label=obs.station_id, marker=markers[i%len(markers)], 
                c=colors[i].reshape(1,-1))


    # Mark the range of heights used
    ax1.axhline(y=ht_max, color='k', linestyle='dashed', label='Ht max')
    if ht_min > 0:
        ax1.axhline(y=ht_min, color='k', linestyle='dashed', label='Ht min')

    ax1.set_xlabel("Velocity relative to the ground (km/s)")
    ax1.set_ylabel("Height (km)")


    vel_data = np.array(vel_data)
    time_data = np.array(time_data)
    ht_data = np.array(ht_data)

    # Sort data by height
    ht_sort = np.argsort(ht_data)
    vel_data = vel_data[ht_sort]
    time_data = time_data[ht_sort]
    ht_data = ht_data[ht_sort]


    # Set up a velocity filter to remove outliers
    vel_filter = vel_data < max_vel


    # Only take data in the appropriate range of heights
    ht_filter = (ht_data/1000 >= ht_min) & (ht_data/1000 <= ht_max)
    vel_data = vel_data[ht_filter & vel_filter]
    time_data = time_data[ht_filter & vel_filter]
    ht_data = ht_data[ht_filter & vel_filter]

    # Plot the selected data
    ax2.scatter(vel_data/1000, time_data, s=5, label="Measurements")
    

    # Fit a line to the velocity data in the range, starting from the least-squares line as dynMassFromTraj()
    #   does; from (1, 1) it can run out of function evaluations
    popt_final, pcov_final, perr_final, vel_filter = fitVelocity(
        time_data, vel_data, p0=np.polyfit(time_data, vel_data, 1), loss='soft_l1', \
        sigma_clip=cml_args.sigma_clip
    )

    vel_fit = popt_final
    vel_fit_cov = pcov_final
    vel_fit_std = perr_final
    decel_std = vel_fit_std[0]

    # Compute the standard deviations of the fit
    #vel_fit_std = np.sqrt(np.diag(vel_fit_cov))
    #decel_std = vel_fit_std[0]

    # # Remove 5 sigma outliers from the data and re-fit
    # vel_filter = np.abs(vel_data - lineFunc(time_data, *vel_fit)) < 5*decel_std
    # vel_fit, vel_fit_cov = scipy.optimize.curve_fit(lineFunc, time_data[vel_filter], vel_data[vel_filter])

    # Plot the selected outliers as an empty red circle
    ax2.scatter(vel_data[~vel_filter]/1000, time_data[~vel_filter], s=20, marker='o', facecolors='none', 
        edgecolors='r', label="${:g}\\sigma$ outliers".format(cml_args.sigma_clip))



    print("Velocity fit params:", vel_fit)

    # Fit a line to the height data
    ht_fit, _ = scipy.optimize.curve_fit(lineFunc, time_data, ht_data)

    print("Height fit params:", ht_fit)

    
    # Plot the line on the height plot
    time_arr = np.linspace(np.min(time_data), np.max(time_data), 10)
    ax1.plot(lineFunc(time_arr, *vel_fit)/1000, lineFunc(time_arr, *ht_fit)/1000, label='Dyn mass fit', color='r')


    # Plot the line on the velocity plot
    ax2.plot(lineFunc(time_arr, *vel_fit)/1000, time_arr, label='Dyn mass fit', color='r')

        
    # Compute the evaluation point
    decel = -vel_fit[0]
    time_eval = np.min(time_data) + eval_point*(np.max(time_data) - np.min(time_data))
    vel_eval = lineFunc(time_eval, *vel_fit)

    # Take the height at the evaluation time from the solver's trajectory model (which includes the gravity
    #   drop), as computeFragEndParams() does. A line fitted to the measured heights is biased at the window
    #   centre by the curvature of the decelerating path and by the measurement noise
    ht_vs_time_interp, _ = interpolateHtVsTimeLen(traj)
    ht_eval = scipy.optimize.brentq(lambda h: ht_vs_time_interp(h) - time_eval, \
        *ht_vs_time_interp.x[[0, -1]])

    # The direction of motion is only needed to add the wind
    azim_eval = elev_eval = None
    if _windENU(atm_profile, ht_eval) is not None:
        azim_eval, elev_eval = evalPointState(traj, ht_eval, ht_vs_time_interp=ht_vs_time_interp)[6:]

    # Compute +/- 2 sigma deceleartion
    decel_lo = decel - 2*decel_std
    decel_hi = decel + 2*decel_std

    # Limit the deceleration to a minimum of 0
    if decel < 0:
        decel = 0
    if decel_lo < 0:
        decel_lo = 0
    if decel_hi < 0:
        decel_hi = 0

    # Compute the dynamic mass (and +/- 2 sigma)
    vel_air = _airSpeed(atm_profile, ht_eval, vel_eval, azim_eval, elev_eval)
    dyn_mass = dynamicMass(bulk_density, traj.rend_lat, traj.rend_lon, ht_eval, traj.jdt_ref, \
        vel_air, decel, gamma=1.0, shape_factor=gamma_a, atm_dens=_airDensity(atm_profile, ht_eval))
    dyn_mass_hi = dynamicMass(bulk_density, traj.rend_lat, traj.rend_lon, ht_eval, traj.jdt_ref, \
        vel_air, decel_lo, gamma=1.0, shape_factor=gamma_a, atm_dens=_airDensity(atm_profile, ht_eval))
    dyn_mass_lo = dynamicMass(bulk_density, traj.rend_lat, traj.rend_lon, ht_eval, traj.jdt_ref, \
        vel_air, decel_hi, gamma=1.0, shape_factor=gamma_a, atm_dens=_airDensity(atm_profile, ht_eval))
    

    # Limit the dynamic mass to a 0 - maxmass range
    mass_max = cml_args.maxmass
    dyn_mass = np.clip(dyn_mass, 0, mass_max)
    dyn_mass_hi = np.clip(dyn_mass_hi, 0, mass_max)
    dyn_mass_lo = np.clip(dyn_mass_lo, 0, mass_max)


    final_decel = final_decel_hi = final_decel_lo = 0

    # Final values stay undefined (NaN) if the evaluation velocity is already below the kill speed
    final_mass = final_mass_hi = final_mass_lo = np.nan
    final_lat = final_lat_hi = final_lat_lo = np.nan
    final_lon = final_lon_hi = final_lon_lo = np.nan
    final_ele = final_ele_hi = final_ele_lo = np.nan
    final_azim = final_azim_hi = final_azim_lo = np.nan
    final_elev = final_elev_hi = final_elev_lo = np.nan
    final_time = final_time_hi = final_time_lo = np.nan
    final_vel = final_vel_hi = final_vel_lo = np.nan
    final_sr = final_sr_hi = final_sr_lo = None

    # Run the fragment until the speed where ablation stops, which is a speed relative to the air
    if vel_air > v_kill:

        print()
        print("Running simulation down to {:g} km/s...".format(v_kill/1000))
        print()
        print("  init vel = {:.2f} km/s".format(vel_eval/1000))
        print("  init ht  = {:.2f} km".format(ht_eval/1000))
        print("  decel   = {:.2f} km/s^2".format(decel/1000))
        print("  init mass = {:.3f} kg".format(dyn_mass))
        print()
        final_sr, final_mass, final_lat, final_lon, final_ele, final_azim, final_elev, final_time, final_vel = \
            computeFragEndParams(traj, dyn_mass, bulk_density, ht_eval, vel_eval, gamma_a, v_kill=v_kill, \
                atm_profile=atm_profile)

        print()
        print("Running simulation down to {:g} km/s (+2 sigma mass)...".format(v_kill/1000))
        print()
        print("  decel = {:.2f} km/s^2".format(decel_hi/1000))
        print("  init mass = {:.3f} kg".format(dyn_mass_hi))
        print()
        final_sr_hi, final_mass_hi, final_lat_hi, final_lon_hi, final_ele_hi, final_azim_hi, final_elev_hi, \
            final_time_hi, final_vel_hi = \
            computeFragEndParams(traj, dyn_mass_hi, bulk_density, ht_eval, vel_eval, gamma_a, v_kill=v_kill, \
                atm_profile=atm_profile)

        print()
        print("Running simulation down to {:g} km/s (-2 sigma mass)...".format(v_kill/1000))
        print()
        print("  decel = {:.2f} km/s^2".format(decel_lo/1000))
        print("  init mass = {:.3f} kg".format(dyn_mass_lo))
        print()
        final_sr_lo, final_mass_lo, final_lat_lo, final_lon_lo, final_ele_lo, final_azim_lo, final_elev_lo, \
            final_time_lo, final_vel_lo = \
            computeFragEndParams(traj, dyn_mass_lo, bulk_density, ht_eval, vel_eval, gamma_a, v_kill=v_kill, \
                atm_profile=atm_profile)
        
        print()

        # Get the deceleration in the last point
        final_decel = _endDecel(final_sr)
        final_decel_hi = _endDecel(final_sr_hi)
        final_decel_lo = _endDecel(final_sr_lo)

        # Plot the simulated velocity until the end (time plot)
        # With winds, MetSim's speeds are relative to the air, not to the ground like the measurements
        sim_frame = ", relative to the air" if final_sr.const.wind_profile is not None else ""

        ax2.plot(final_sr_lo.main_vel_arr/1000, final_sr_lo.time_arr + time_eval, label='Simulation (-2sigma{:s})'.format(sim_frame), color='k', linestyle='dashed')
        ax2.plot(final_sr.main_vel_arr/1000, final_sr.time_arr + time_eval, label='Simulation (nominal{:s})'.format(sim_frame), color='k', linestyle='solid')
        ax2.plot(final_sr_hi.main_vel_arr/1000, final_sr_hi.time_arr + time_eval, label='Simulation (+2sigma{:s})'.format(sim_frame), color='k', linestyle='dotted')

        # Plot the simulated velocity until the end (height plot)
        ax1.plot(final_sr.main_vel_arr/1000, final_sr.main_height_arr/1000, label='Simulation (nominal{:s})'.format(sim_frame), color='k', linestyle='solid')



    ### Propagate the WMPL trajectory Monte Carlo solver's uncertainties, if requested ###

    mc_results = None
    mc_realizations = None

    if cml_args.mc:

        print()
        print("Looking for the Monte Carlo uncertainties pickle...")
        mc_traj_list = findMCUncertaintiesPickle(cml_args.traj_path)
        if mc_traj_list is not None:
            mc_realizations = len(mc_traj_list)

        if mc_traj_list is None:
            print("  No 'Monte Carlo/*_mc_uncertainties.pickle' file with Monte Carlo trajectory "
                "realizations was found next to the trajectory pickle. Skipping MC uncertainty propagation.")

        else:
            print("  Found {:d} Monte Carlo trajectory realizations.".format(len(mc_traj_list)))
            print()
            print("Propagating Monte Carlo uncertainties through the dynamic mass fit...")
            print()

            mc_results = runMonteCarloDynMass(mc_traj_list, ht_max, ht_min, eval_point, bulk_density, \
                gamma_a, max_vel, mass_max, cml_args.sigma_clip, n_samples=cml_args.mc_samples, \
                run_final_sim=(not cml_args.mc_no_final_sim), rng_seed=cml_args.mc_seed, \
                cores=cml_args.mc_cores, v_kill=v_kill, v_kill_sigma=v_kill_sigma, \
                density_sigma=cml_args.dens_sigma, atm_profile=atm_profile)

            print()
            print("Monte Carlo uncertainty summary (95% CI):")
            printMCPercentiles(mc_results)

            # Save the full set of Monte Carlo realizations for further analysis
            mc_npz_path = os.path.join(dir_path, traj.file_name + "_dyn_mass_mc.npz")
            np.savez(mc_npz_path, **mc_results)
            print()
            print("Saved Monte Carlo realizations to: {:s}".format(mc_npz_path))

            # Overlay the Monte Carlo evaluation points on the velocity vs. time plot
            if len(mc_results['vel_eval']) > 0:
                ax2.scatter(mc_results['vel_eval']/1000, mc_results['time_eval'], s=8, marker='.', \
                    color='tab:orange', alpha=0.4, zorder=2, label='MC realizations')

    ###



    # Plot the evaluation point
    label_text = "Dyn $m$ = [{:.3f}, {:.3f}, {:.3f}] kg\n"\
        "Final $m$ = [{:.3f}, {:.3f}, {:.3f}] kg\n"\
        "Decel = {:.2f} $\\pm$ {:.2f} km/s$^2$\n"\
        "V = {:.2f} km/s\n"\
        "h = {:.2f} km\n"\
        "$\\rho_m$ = {:d} kg/m$^3$\n"\
        "$\\Gamma A$ = {:.2f}".format( \
            dyn_mass_lo, dyn_mass, dyn_mass_hi,
            final_mass_lo, final_mass, final_mass_hi,
            decel/1000, decel_std/1000,
            vel_eval/1000,
            ht_eval/1000,
            int(bulk_density),
            gamma_a)

    # Number of MC realizations with a velocity fit, and of those with a final simulation
    n_mc_fit = n_mc_final = 0
    if mc_results is not None:
        n_mc_fit = len(mc_results['dyn_mass'])
        if 'final_mass' in mc_results:
            n_mc_final = np.count_nonzero(~np.isnan(mc_results['final_mass']))

    if n_mc_fit > 0:
        dyn_mass_mc_lo, dyn_mass_mc_hi = np.percentile(mc_results['dyn_mass'], [2.5, 97.5])
        label_text += "\nDyn $m$ (MC 95% CI) = [{:.3f}, {:.3f}] kg".format(dyn_mass_mc_lo, dyn_mass_mc_hi)

    if n_mc_final > 0:
        final_mass_mc_lo, final_mass_mc_hi = np.nanpercentile(mc_results['final_mass'], [2.5, 97.5])
        label_text += "\nFinal $m$ (MC 95% CI) = [{:.3f}, {:.3f}] kg".format(final_mass_mc_lo, \
            final_mass_mc_hi)

    ax2.scatter(vel_eval/1000, time_eval, color='g', marker='o', s=50, label=label_text)


    print()
    print("Density = {:d} kg/m^3".format(int(bulk_density)))
    print("Gamma*A = {:.2f}".format(gamma_a))
    if atm_profile is None:
        print("Atmosphere = {:s}{:s}".format(_msisName(), "" if cml_args.atmtime is None \
            else " at {:s} UTC".format(cml_args.atmtime)))
    else:
        print("Atmosphere = {:s} profile {:s}".format(atm_profile.profile_type, atm_profile.path))
        if atm_profile.use_winds:
            wind = atm_profile.wind(ht_eval)
            print("            (wind {:.1f} m/s towards {:.0f} deg at {:.2f} km; speed relative to the air {:.3f} "
                "km/s)".format(np.hypot(*wind), np.degrees(np.arctan2(*wind))%360, ht_eval/1000, vel_air/1000))
        print("            (simulation polynomial within {:.3f}% of it between {:.1f} and {:.1f} km)".format( \
            100*atm_profile.fitPoly(SIM_HT_MIN, ht_eval)[1], SIM_HT_MIN/1000, ht_eval/1000))
    print("Decel = {:.2f} +/- {:.2f} km/s^2".format(decel/1000, decel_std/1000))
    print()
    print("Dynamic mass at {:.2f} km and {:.2f} km/s:".format(ht_eval/1000, vel_eval/1000))
    print("(+/-2 sigma from the deceleration uncertainty only)")
    print("-2sigma = {:.3f} kg".format(dyn_mass_lo))
    print("Nominal = {:.3f} kg".format(dyn_mass))
    print("+2sigma    = {:.3f} kg".format(dyn_mass_hi))

    if n_mc_fit > 0:
        dm_lo, dm_med, dm_hi = np.percentile(mc_results['dyn_mass'], [2.5, 50, 97.5])
        dmg_lo, dmg_med, dmg_hi = np.percentile(mc_results['dyn_mass_geom'], [2.5, 50, 97.5])
        print()
        print("Monte Carlo over {:d} trajectory solver realizations (95% CI):".format(n_mc_fit))
        if cml_args.dens_sigma > 0:
            print("(velocity fit uncertainty + trajectory geometry uncertainty + density uncertainty)")
        else:
            print("(velocity fit uncertainty + trajectory geometry uncertainty)")
        print("MC 2.5%  = {:.3f} kg".format(dm_lo))
        print("MC Median= {:.3f} kg".format(dm_med))
        print("MC 97.5% = {:.3f} kg".format(dm_hi))
        print("(trajectory geometry uncertainty only: {:.3f} [{:.3f}, {:.3f}] kg)".format(dmg_med, dmg_lo, \
            dmg_hi))

        if cml_args.dens_sigma > 0:
            de_lo, de_med, de_hi = np.percentile(mc_results['density'], [2.5, 50, 97.5])
            print("(bulk density drawn: {:.0f} [{:.0f}, {:.0f}] kg/m^3)".format(de_med, de_lo, de_hi))

    print()
    print("Simulation down to {:g} km/s:".format(v_kill/1000))
    print("------------------------------------")
    print("Azim (+E of due N) and Elev: apparent ground-fixed radiant, epoch of date, gravity turn included")
    if n_mc_final > 0:
        print("(-2sigma/nominal/+2sigma below vary only the deceleration on the nominal trajectory; the "
            "MC block after them also includes the trajectory geometry uncertainty)")
    print("Final end coordinates (-2sigma mass)")
    print("Mass      = {:.3f} kg".format(final_mass_lo))
    print("Lat (+N)  = {:.5f} deg".format(final_lat_lo))
    print("Lon (+E)  = {:.5f} deg".format(final_lon_lo))
    print("Ele MSL   = {:.2f} km".format(final_ele_lo))
    print("Azim      = {:.5f} deg".format(final_azim_lo))
    print("Elev      = {:.5f} deg".format(final_elev_lo))
    print("End decel = {:.3f} km/s^2".format(final_decel_lo/1000))
    print("Final end coordinates (nominal mass)")
    print("Mass      = {:.3f} kg".format(final_mass))
    print("Lat (+N)  = {:.5f} deg".format(final_lat))
    print("Lon (+E)  = {:.5f} deg".format(final_lon))
    print("Ele MSL   = {:.2f} km".format(final_ele))
    print("Azim      = {:.5f} deg".format(final_azim))
    print("Elev      = {:.5f} deg".format(final_elev))
    print("End decel = {:.3f} km/s^2".format(final_decel/1000))
    print("Final end coordinates (+2sigma mass)")
    print("Mass      = {:.3f} kg".format(final_mass_hi))
    print("Lat (+N)  = {:.5f} deg".format(final_lat_hi))
    print("Lon (+E)  = {:.5f} deg".format(final_lon_hi))
    print("Ele MSL   = {:.2f} km".format(final_ele_hi))
    print("Azim      = {:.5f} deg".format(final_azim_hi))
    print("Elev      = {:.5f} deg".format(final_elev_hi))
    print("End decel = {:.3f} km/s^2".format(final_decel_hi/1000))

    if n_mc_final > 0:
        pct = [2.5, 50, 97.5]
        mass_lo, mass_med, mass_hi = np.nanpercentile(mc_results['final_mass'], pct)
        lat_lo, lat_med, lat_hi = np.nanpercentile(mc_results['final_lat'], pct)
        lon_lo, lon_med, lon_hi = np.nanpercentile(mc_results['final_lon'], pct)
        ele_lo, ele_med, ele_hi = np.nanpercentile(mc_results['final_ele'], pct)
        azim_lo, azim_med, azim_hi = np.nanpercentile(mc_results['final_azim'], pct)
        elev_lo, elev_med, elev_hi = np.nanpercentile(mc_results['final_elev'], pct)
        fdecel_lo, fdecel_med, fdecel_hi = np.nanpercentile(mc_results['final_decel'], pct)

        print()
        print("Final end coordinates (Monte Carlo, {:d} realizations, 95% CI)".format(n_mc_final))
        print("Mass      = {:.3f} kg     [{:.3f}, {:.3f}]".format(mass_med, mass_lo, mass_hi))
        print("Lat (+N)  = {:.5f} deg    [{:.5f}, {:.5f}]".format(lat_med, lat_lo, lat_hi))
        print("Lon (+E)  = {:.5f} deg    [{:.5f}, {:.5f}]".format(lon_med, lon_lo, lon_hi))
        print("Ele MSL   = {:.2f} km     [{:.2f}, {:.2f}]".format(ele_med, ele_lo, ele_hi))
        print("Azim      = {:.5f} deg    [{:.5f}, {:.5f}]".format(azim_med, azim_lo, azim_hi))
        print("Elev      = {:.5f} deg    [{:.5f}, {:.5f}]".format(elev_med, elev_lo, elev_hi))
        print("End decel = {:.3f} km/s^2 [{:.3f}, {:.3f}]".format(fdecel_med/1000, fdecel_lo/1000, \
            fdecel_hi/1000))

        if v_kill_sigma > 0:
            v_kill_sim = mc_results['v_kill'][~np.isnan(mc_results['final_mass'])]
            vk_lo, vk_med, vk_hi = np.percentile(v_kill_sim, pct)
            print("V kill    = {:.3f} km/s   [{:.3f}, {:.3f}]".format(vk_med/1000, vk_lo/1000, vk_hi/1000))



    # Save the results for a dark flight computation
    if cml_args.save_pickle is not None:

        def _state(sr, vel, mass, lat, lon, ele, azim, elev, t):
            if sr is None:
                return None
            return ejectionState(lat, lon, ele, vel/1000, azim, elev, t, mass, bulk_density)

        fit = {
            'ht_min': ht_min, 'ht_max': ht_max, 'eval_point': eval_point, 'ht_eval': ht_eval/1000, \
            'vel_eval': vel_eval/1000, 'time_eval': time_eval, 'decel': decel/1000, \
            'decel_std': decel_std/1000, 'dyn_mass': dyn_mass, 'dyn_mass_minus_2sigma': dyn_mass_lo, \
            'dyn_mass_plus_2sigma': dyn_mass_hi, 'sigma_clip': cml_args.sigma_clip
        }

        model = {
            'gamma_a': gamma_a, 'density': bulk_density, 'density_sigma': cml_args.dens_sigma, \
            'v_kill': v_kill/1000, 'v_kill_sigma': v_kill_sigma/1000, 'mass_max': mass_max, \
            'atmosphere': '{:s} polynomial fit (wmpl.Utils.AtmosphereDensity.fitAtmPoly)'.format(_msisName()), \
            'msis_version': _msisSettings()[0], 'msis_time': cml_args.atmtime, 'atm_profile': None
        }

        # The atmosphere profile, so a dark flight code can check it uses the same one
        if atm_profile is not None:
            model['atmosphere'] = 'profile file, see atm_profile'
            model['atm_profile'] = {
                'path': os.path.abspath(atm_profile.path), 'type': atm_profile.profile_type, \
                'sha256': atm_profile.sha256, 'ht_min': atm_profile.ht_min/1000, \
                'ht_max': atm_profile.ht_max/1000, \
                'sim_poly_max_rel_err': atm_profile.fitPoly(SIM_HT_MIN, ht_eval)[1], \
                'winds': atm_profile.use_winds
            }

        # Constants of the single-body ablation simulation
        if final_sr is not None:
            model.update({'sim_gamma': final_sr.const.gamma, 'sim_shape_factor': final_sr.const.shape_factor, \
                'sim_ablation_coeff': final_sr.const.sigma*1e6})

        output = buildDynMassFitOutput(traj.jdt_ref, fit, model, \
            nominal=_state(final_sr, final_vel, final_mass, final_lat, final_lon, final_ele, final_azim, final_elev, \
                final_time), \
            minus_2sigma=_state(final_sr_lo, final_vel_lo, final_mass_lo, final_lat_lo, final_lon_lo, final_ele_lo, \
                final_azim_lo, final_elev_lo, final_time_lo), \
            plus_2sigma=_state(final_sr_hi, final_vel_hi, final_mass_hi, final_lat_hi, final_lon_hi, final_ele_hi, \
                final_azim_hi, final_elev_hi, final_time_hi), \
            mc_results=mc_results, mc_realizations=mc_realizations, traj_path=cml_args.traj_path, \
            mc_path=(mcUncertaintiesPath(cml_args.traj_path) if mc_results is not None else None), \
            dmf_args=vars(cml_args))

        pickle_path = cml_args.save_pickle
        if not pickle_path:
            pickle_path = os.path.join(dir_path, traj.file_name + "_dyn_mass_fit.pickle")

        saveDynMassFitPickle(pickle_path, output)

        print()
        print("Saved the results to: {:s}".format(pickle_path))


    ax2.invert_yaxis()
    ax2.set_ylabel("Time (s)")
    ax2.set_xlabel("Velocity relative to the ground (km/s)")
    
    ax1.legend()
    ax2.legend()

    # Add padding to the simulation axis so the legend fits
    ax2_y_min, ax2_y_max = ax2.get_ylim()
    ax2.set_ylim(ax2_y_min, 0.75*ax2_y_max)


    plt.tight_layout()


    # Save plot to pickle directory
    plt.savefig(os.path.join(dir_path, traj.file_name + "_dyn_mass_fit.png"), dpi=300)

    plt.show()