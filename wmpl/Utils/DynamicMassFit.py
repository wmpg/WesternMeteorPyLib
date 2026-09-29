import os

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from wmpl.Utils.AtmosphereDensity import fitAtmPoly
from wmpl.Utils.Math import lineFunc, vectMag
from wmpl.Utils.TrajConversions import cartesian2Geo, derotatedRadiantAltAz
from wmpl.Utils.Physics import dynamicMass
from wmpl.Utils.Pickling import loadPickle
from wmpl.MetSim.MetSimErosion import Constants, runSimulation, G0
from wmpl.MetSim.GUI import SimulationResults
from wmpl.Trajectory.Trajectory import applyGravityDrop


def runFragSim(mass, density, lat, lon, jd, ht_beg, v_init, entry_angle, gamma_a):

    # Init simulation constants
    const = Constants()


    # Set minimum simulation height
    const.h_kill = 15000 # m 

    # Set minimum simulation speed
    const.v_kill = 3000 # m/s


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

    # Zenith angle
    const.zenith_angle = np.radians(90 - entry_angle)

    # Use Borovicka 2020 luminous efficiency (not really used here)
    const.lum_eff_type = 7
    const.P_0m = 1210

    # Disable erosion and disruption (single-body only)
    const.erosion_on = False
    const.erosion_coeff = 0
    const.disruption_on = False
    const.fragmentation_on = False
    


    # Fit the atmosphere density polynomial using NRLMSISE
    ht_min = const.h_kill
    ht_max = 180000
    const.dens_co = fitAtmPoly(lat, lon, ht_min, ht_max, jd)

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


def computeFragEndParams(traj, dyn_mass, density, hend, vend, gamma_a):
    """ Propagate a fragment of the given dynamic mass from the evaluation point down to 3 km/s with the
        single-body ablation model, and compute where and in which direction it ends.

        The returned final azimuth and elevation are those of the apparent ground-fixed radiant (epoch of
        date), as traj.orbit.azimuth_apparent_norot and elevation_apparent_norot, evaluated at the final point
        and with the elevation steepened by the gravity turn along the path. They are the inputs a dark
        flight computation needs.

    Arguments:
        traj: [Trajectory] Solved trajectory.
        dyn_mass: [float] Dynamic mass at the evaluation point (kg).
        density: [float] Bulk density of the meteoroid (kg/m^3).
        hend: [float] Height of the evaluation point (m).
        vend: [float] Velocity at the evaluation point (m/s).
        gamma_a: [float] Product of the drag coefficient and the shape factor used for the dynamic mass. Not
            used by the simulation itself, see runFragSim().

    Return:
        (sr, final_mass, final_lat, final_lon, final_ele, final_azim, final_elev): [tuple]
            sr: [SimulationResults] Results of the fragment simulation.
            final_mass: [float] Mass at the final point (kg).
            final_lat: [float] Latitude of the final point (deg, +N).
            final_lon: [float] Longitude of the final point (deg, +E).
            final_ele: [float] Height of the final point (km).
            final_azim: [float] Azimuth of the ground-fixed radiant at the final point (deg, +E of due N).
            final_elev: [float] Elevation of the ground-fixed radiant at the final point (deg).
    """

    jd = traj.jdt_ref
    lat = np.degrees(traj.rend_lat)
    lon = np.degrees(traj.rend_lon)

    entry_angle = np.degrees(traj.orbit.elevation_apparent_norot)

    # Fit an interpolation function from time to height
    ht_vs_time_interp, _ = interpolateHtVsTimeLen(traj, sample_step=0.1, show_plots=False)

    # # Compute the dynamic mass (upper range)
    # dyn_mass = dynamicMass(density, np.radians(lat), np.radians(lon), hend, jd, vend, decel, \
    #     gamma=1.0, shape_factor=gamma_a)

    # print("  vel        = {:.2f} km/s".format(vend/1000))
    # print("  decel      = {:.2f} km/s^2".format(decel/1000))
    # print("  dyn mass   = {:.3f} kg".format(dyn_mass))


    # Get the time at the observed point, and its length from the trajectory geometry. Interpolating the
    #   length instead mixes the smoothed heights with the raw lengths, which at shallow entry angles turns
    #   a small height difference into a large length difference
    meas_time = ht_vs_time_interp(hend)

    # The height along the line falls from the state vector until the point nearest the Earth's centre, at
    #   a length of state_vect.radiant, which brackets the root
    meas_len = scipy.optimize.brentq(lambda l: cartesian2Geo(traj.jdt_ref + meas_time/86400, \
        *pointOnTrajectory(traj, l, meas_time))[2] - hend, 0, \
        np.dot(traj.state_vect_mini, traj.radiant_eci_mini))
    

    # Run the simulation until ablation stops
    sr = runFragSim(dyn_mass, density, lat, lon, jd, hend, vend, entry_angle, gamma_a)

    # Extract the final height
    final_ht = 0
    if len(sr.brightest_height_arr) > 2:
        final_ht = sr.brightest_height_arr[-2]

    # Extract the total simulation time
    final_time = np.max(sr.time_arr)

    # Compute the total length and time since the first observed point on the trajectory
    total_len = meas_len + sr.frag_main.length
    total_time = meas_time + final_time


    ### Compute the final lat/lon ###

    # Initial 3D ECI vector + total length x direction
    final_eci = pointOnTrajectory(traj, total_len, total_time)

    t_obs = np.concatenate([obs.time_data for obs in traj.observations])

    # Compute exact time of the end
    final_jd = jd + total_time/86400

    # Compute the geo coordinates
    final_lat, final_lon, final_ele = cartesian2Geo(final_jd, *final_eci)

    ###


    ### Compute the final ground-fixed azimuth and elevation (as in Orbit.calcOrbit) ###

    # Derotate the fitted radiant at the final point. The radiant is the tangent of the path at its beginning
    #   (the solver models gravity as a drop from that line), so it is derotated with the initial velocity, as
    #   Orbit.calcOrbit does. Drag does not rotate the direction of motion relative to the air, so this
    #   ground-fixed direction holds along the path up to the gravity turn added below.
    final_azim, final_elev, _ = derotatedRadiantAltAz(traj.v_init*traj.radiant_eci_mini, final_eci, \
        final_jd, final_lat, final_lon)

    # Steepen the elevation by the gravity turn along the path, d(elev)/dt = g*cos(elev)/v, using the average
    #   speed over the observed part and the simulated speeds after it. The turn starts where the fitted radiant
    #   is tangent to the path: its beginning if the solver modelled the gravity drop, otherwise about its middle
    t_turn = 0.0 if getattr(traj, 'gravity_correction', True) else (np.min(t_obs) + np.max(t_obs))/2
    v_sim = sr.main_vel_arr[1:]
    g_final = G0/(1 + final_ele/sr.const.r_earth)**2
    final_elev += g_final*np.cos(final_elev)*((meas_time - t_turn)/traj.orbit.v_avg_norot \
        + np.sum(np.diff(sr.time_arr)[v_sim > 0]/v_sim[v_sim > 0]))

    ###



    print("  final mass     = {:.3f} kg".format(sr.frag_main.m))
    print("  final vel      = {:.3f} km/s".format(sr.frag_main.v/1000))
    print("  final ht (sim) = {:.3f} km".format(final_ht/1000))
    print("  total len      = {:.3f} km".format(total_len/1000))
    print("  total time     = {:.3f} s".format(total_time))
    print("  final lat      = {:.5f} deg".format(np.degrees(final_lat)))
    print("  final lon      = {:.5f} deg".format(np.degrees(final_lon)))
    print("  final ht       = {:.3f} km".format(final_ele/1000))
    print("  final azim     = {:.5f} deg".format(np.degrees(final_azim)))
    print("  final elev     = {:.5f} deg".format(np.degrees(final_elev)))


    return sr, sr.frag_main.m, np.degrees(final_lat), np.degrees(final_lon), final_ele/1000, \
        np.degrees(final_azim), np.degrees(final_elev)



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



def findMCUncertaintiesPickle(traj_path):
    """ Locate the Monte Carlo uncertainties pickle that the WMPL trajectory solver saves next to the main
        trajectory pickle, and return the individual Monte Carlo trajectory realizations it holds.

        The solver saves this as "Monte Carlo/<file_name>_mc_uncertainties.pickle", next to the main
        "<file_name>_trajectory.pickle" (see Trajectory.run() in wmpl/Trajectory/Trajectory.py). Unlike
        traj.uncertainties on the main pickle, whose mc_traj_list is stripped to save space, this file keeps
        every accepted Monte Carlo run as a full, independently re-solved Trajectory object.

    Arguments:
        traj_path: [str] Path to the main "*_trajectory.pickle" file.

    Return:
        mc_traj_list: [list or None] List of Trajectory objects, one per accepted Monte Carlo run, or None
            if no Monte Carlo uncertainties pickle was found next to the trajectory pickle, or it did not
            contain any runs.
    """

    dir_path = os.path.dirname(os.path.abspath(traj_path))
    file_name = os.path.basename(traj_path)

    if file_name.endswith('_trajectory.pickle'):
        file_name_core = file_name[:-len('_trajectory.pickle')]
    else:
        file_name_core = os.path.splitext(file_name)[0]

    mc_dir = os.path.join(dir_path, 'Monte Carlo')
    mc_file = file_name_core + '_mc_uncertainties.pickle'

    if not os.path.isfile(os.path.join(mc_dir, mc_file)):
        return None

    traj_unc = loadPickle(mc_dir, mc_file)

    mc_traj_list = getattr(traj_unc, 'mc_traj_list', None)

    if not mc_traj_list:
        return None

    return mc_traj_list



def dynMassFromTraj(traj, ht_max, ht_min, eval_point, bulk_density, gamma_a, max_vel, mass_max, \
    sigma_clip=3.0):
    """ Compute the dynamic mass at the evaluation point for a single trajectory solution, following the
        same steps as the dynamic mass fit in __main__ (velocity vs. time fit in the height window, then
        dynamic mass at the evaluation point). Used both for individual Monte Carlo trajectory realizations.

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

    Return:
        [dict or None] None if there isn't enough data in the height window to fit a line. Otherwise a dict
            with keys: decel, decel_std, vel_eval, ht_eval, time_eval, dyn_mass.
    """

    vel_data, ht_data, time_data = [], [], []

    for obs in traj.observations:

        ignored = obs.ignore_list[1:] > 0

        vel = obs.velocities[1:][~ignored]
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

    try:
        popt, _, perr, _ = fitVelocity(time_data, vel_data, p0=(1.0, 1.0), loss='soft_l1', \
            sigma_clip=sigma_clip)
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
    except Exception:
        return None

    if decel < 0:
        decel = 0

    dyn_mass = dynamicMass(bulk_density, traj.rend_lat, traj.rend_lon, ht_eval, traj.jdt_ref, vel_eval, \
        decel, gamma=1.0, shape_factor=gamma_a)
    dyn_mass = np.clip(dyn_mass, 0, mass_max)

    return {
        'decel': decel, 'decel_std': decel_std, 'vel_eval': vel_eval, 'ht_eval': ht_eval, \
        'time_eval': time_eval, 'dyn_mass': dyn_mass
    }



def runMonteCarloDynMass(mc_traj_list, ht_max, ht_min, eval_point, bulk_density, gamma_a, max_vel, \
    mass_max, sigma_clip, n_samples=None, run_final_sim=True, rng_seed=None):
    """ Propagate the WMPL trajectory Monte Carlo solver's uncertainties through the dynamic mass fit (and,
        optionally, through the final fragment simulation down to 3 km/s) by repeating dynMassFromTraj() and
        computeFragEndParams() on every individual Monte Carlo trajectory realization.

    Arguments:
        mc_traj_list: [list] List of Trajectory objects, one per Monte Carlo realization (see
            findMCUncertaintiesPickle()).
        ht_max, ht_min, eval_point, bulk_density, gamma_a, max_vel, mass_max, sigma_clip: Same meaning as in
            dynMassFromTraj() / computeFragEndParams().
        n_samples: [int or None] Maximum number of realizations to use. If mc_traj_list is larger, a random
            subset of this size is drawn without replacement. None uses all realizations.
        run_final_sim: [bool] Also run the final fragment simulation (down to 3 km/s) for every realization,
            to get the final mass, position and radiant uncertainty. Slower.
        rng_seed: [int or None] Random seed used for subsampling.

    Return:
        results: [dict of ndarray] Arrays of dyn_mass, decel, vel_eval, ht_eval over the successfully fitted
            realizations, plus final_mass, final_lat, final_lon, final_ele, final_azim, final_elev,
            final_decel if run_final_sim is True.
    """

    traj_samples = mc_traj_list

    if (n_samples is not None) and (len(traj_samples) > n_samples):
        rng = np.random.default_rng(rng_seed)
        indices = rng.choice(len(traj_samples), size=n_samples, replace=False)
        traj_samples = [traj_samples[i] for i in indices]

    keys = ['dyn_mass', 'decel', 'vel_eval', 'ht_eval', 'time_eval']
    if run_final_sim:
        keys += ['final_mass', 'final_lat', 'final_lon', 'final_ele', 'final_azim', 'final_elev', \
            'final_decel']

    results = {key: [] for key in keys}

    n_total = len(traj_samples)

    for i, traj_mc in enumerate(traj_samples):

        traj_mc = _ensureTrajDefaults(traj_mc)

        print("MC realization {:d}/{:d}...".format(i + 1, n_total))

        fit_res = dynMassFromTraj(traj_mc, ht_max, ht_min, eval_point, bulk_density, gamma_a, max_vel, \
            mass_max, sigma_clip=sigma_clip)

        if fit_res is None:
            print("  Skipped: not enough data in the height window.")
            continue

        if run_final_sim:

            if fit_res['vel_eval'] <= 3000:
                print("  Skipped: evaluation velocity already below 3 km/s.")
                continue

            try:
                sr_mc, final_mass, final_lat, final_lon, final_ele, final_azim, final_elev = \
                    computeFragEndParams(traj_mc, fit_res['dyn_mass'], bulk_density, fit_res['ht_eval'], \
                        fit_res['vel_eval'], gamma_a)
            except Exception as e:
                print("  Skipped: fragment simulation failed ({}).".format(e))
                continue

            results['final_mass'].append(final_mass)
            results['final_lat'].append(final_lat)
            results['final_lon'].append(final_lon)
            results['final_ele'].append(final_ele)
            results['final_azim'].append(final_azim)
            results['final_elev'].append(final_elev)
            results['final_decel'].append((sr_mc.main_vel_arr[-1] - sr_mc.main_vel_arr[-2]) \
                /(sr_mc.time_arr[-1] - sr_mc.time_arr[-2]))

        results['dyn_mass'].append(fit_res['dyn_mass'])
        results['decel'].append(fit_res['decel'])
        results['vel_eval'].append(fit_res['vel_eval'])
        results['ht_eval'].append(fit_res['ht_eval'])
        results['time_eval'].append(fit_res['time_eval'])

    results = {key: np.array(val) for key, val in results.items()}

    print()
    print("Monte Carlo propagation: {:d}/{:d} realizations used.".format(len(results['dyn_mass']), n_total))

    return results



def printMCPercentiles(results, ci=95.0):
    """ Print the median and confidence interval of every array in a Monte Carlo results dict (see
        runMonteCarloDynMass()).
    """

    lo_pct = (100 - ci)/2
    hi_pct = 100 - lo_pct

    for key, arr in results.items():

        if len(arr) == 0:
            continue

        med = np.median(arr)
        lo, hi = np.percentile(arr, [lo_pct, hi_pct])

        print("  {:12s} median = {:10.4f}, {:.0f}% CI = [{:10.4f}, {:10.4f}]".format(key, med, ci, lo, hi))



if __name__ == "__main__":

    import argparse

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Compute the final dynamic mass of a fireball by defining the height range of the final portion where the mass is measured. A simulation is run to propagate the fragment to a speed of 3 km/s and estimate the final mass and location.")

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
        'to 3 km/s) for every realization and only propagate the dynamic mass at the evaluation point. Much '
        'faster, but does not give final mass/position/radiant uncertainties.')

    arg_parser.add_argument('--mc_seed', metavar='SEED', type=int, default=None, \
        help='Random seed used when subsampling Monte Carlo realizations with --mc_samples.')

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

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


        vel = obs.velocities[1:][~ignored]
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

    ax1.set_xlabel("Velocity (km/s)")
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
    

    # Fit a line to the velocity data in the range
    popt_final, pcov_final, perr_final, vel_filter = fitVelocity(
        time_data, vel_data, p0=(1.0, 1.0), loss='soft_l1', sigma_clip=cml_args.sigma_clip
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
    dyn_mass = dynamicMass(bulk_density, traj.rend_lat, traj.rend_lon, ht_eval, traj.jdt_ref, \
        vel_eval, decel, gamma=1.0, shape_factor=gamma_a)
    dyn_mass_hi = dynamicMass(bulk_density, traj.rend_lat, traj.rend_lon, ht_eval, traj.jdt_ref, \
        vel_eval, decel_lo, gamma=1.0, shape_factor=gamma_a)
    dyn_mass_lo = dynamicMass(bulk_density, traj.rend_lat, traj.rend_lon, ht_eval, traj.jdt_ref, \
        vel_eval, decel_hi, gamma=1.0, shape_factor=gamma_a)
    

    # Limit the dynamic mass to a 0 - maxmass range
    mass_max = cml_args.maxmass
    dyn_mass = np.clip(dyn_mass, 0, mass_max)
    dyn_mass_hi = np.clip(dyn_mass_hi, 0, mass_max)
    dyn_mass_lo = np.clip(dyn_mass_lo, 0, mass_max)


    final_decel = final_decel_hi = final_decel_lo = 0

    # Final values stay undefined (NaN) if the evaluation velocity is already below 3 km/s
    final_mass = final_mass_hi = final_mass_lo = np.nan
    final_lat = final_lat_hi = final_lat_lo = np.nan
    final_lon = final_lon_hi = final_lon_lo = np.nan
    final_ele = final_ele_hi = final_ele_lo = np.nan
    final_azim = final_azim_hi = final_azim_lo = np.nan
    final_elev = final_elev_hi = final_elev_lo = np.nan

    # Run the fragment until the final velocity of 3 km/s
    if vel_eval > 3000:

        print()
        print("Running simulation down to 3 km/s...")
        print()
        print("  init vel = {:.2f} km/s".format(vel_eval/1000))
        print("  init ht  = {:.2f} km".format(ht_eval/1000))
        print("  decel   = {:.2f} km/s^2".format(decel/1000))
        print("  init mass = {:.3f} kg".format(dyn_mass))
        print()
        final_sr, final_mass, final_lat, final_lon, final_ele, final_azim, final_elev = \
            computeFragEndParams(traj, dyn_mass, bulk_density, ht_eval, vel_eval, gamma_a)

        print()
        print("Running simulation down to 3 km/s (+2 sigma mass)...")
        print()
        print("  decel = {:.2f} km/s^2".format(decel_hi/1000))
        print("  init mass = {:.3f} kg".format(dyn_mass_hi))
        print()
        final_sr_hi, final_mass_hi, final_lat_hi, final_lon_hi, final_ele_hi, final_azim_hi, final_elev_hi = \
            computeFragEndParams(traj, dyn_mass_hi, bulk_density, ht_eval, vel_eval, gamma_a)

        print()
        print("Running simulation down to 3 km/s (-2 sigma mass)...")
        print()
        print("  decel = {:.2f} km/s^2".format(decel_lo/1000))
        print("  init mass = {:.3f} kg".format(dyn_mass_lo))
        print()
        final_sr_lo, final_mass_lo, final_lat_lo, final_lon_lo, final_ele_lo, final_azim_lo, final_elev_lo = \
            computeFragEndParams(traj, dyn_mass_lo, bulk_density, ht_eval, vel_eval, gamma_a)
        
        print()

        # Get the deceleration in the last point
        final_decel = (final_sr.main_vel_arr[-1] - final_sr.main_vel_arr[-2]) \
            /(final_sr.time_arr[-1] - final_sr.time_arr[-2])
        final_decel_hi = (final_sr_hi.main_vel_arr[-1] - final_sr_hi.main_vel_arr[-2]) \
            /(final_sr_hi.time_arr[-1] - final_sr_hi.time_arr[-2])
        final_decel_lo = (final_sr_lo.main_vel_arr[-1] - final_sr_lo.main_vel_arr[-2]) \
            /(final_sr_lo.time_arr[-1] - final_sr_lo.time_arr[-2])

        # Plot the simulated velocity until the end (time plot)
        ax2.plot(final_sr_lo.main_vel_arr/1000, final_sr_lo.time_arr + time_eval, label='Simulation (-2sigma)', color='k', linestyle='dashed')
        ax2.plot(final_sr.main_vel_arr/1000, final_sr.time_arr + time_eval, label='Simulation (nominal)', color='k', linestyle='solid')
        ax2.plot(final_sr_hi.main_vel_arr/1000, final_sr_hi.time_arr + time_eval, label='Simulation (+2sigma)', color='k', linestyle='dotted')

        # Plot the simulated velocity until the end (height plot)
        ax1.plot(final_sr.main_vel_arr/1000, final_sr.main_height_arr/1000, label='Simulation (nominal)', color='k', linestyle='solid')



    ### Propagate the WMPL trajectory Monte Carlo solver's uncertainties, if requested ###

    mc_results = None

    if cml_args.mc:

        print()
        print("Looking for the Monte Carlo uncertainties pickle...")
        mc_traj_list = findMCUncertaintiesPickle(cml_args.traj_path)

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
                run_final_sim=(not cml_args.mc_no_final_sim), rng_seed=cml_args.mc_seed)

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

    if mc_results is not None and len(mc_results['dyn_mass']) > 0:
        dyn_mass_mc_lo, dyn_mass_mc_hi = np.percentile(mc_results['dyn_mass'], [2.5, 97.5])
        label_text += "\nDyn $m$ (MC 95% CI) = [{:.3f}, {:.3f}] kg".format(dyn_mass_mc_lo, dyn_mass_mc_hi)

        if len(mc_results.get('final_mass', [])) > 0:
            final_mass_mc_lo, final_mass_mc_hi = np.percentile(mc_results['final_mass'], [2.5, 97.5])
            label_text += "\nFinal $m$ (MC 95% CI) = [{:.3f}, {:.3f}] kg".format(final_mass_mc_lo, \
                final_mass_mc_hi)

    ax2.scatter(vel_eval/1000, time_eval, color='g', marker='o', s=50, label=label_text)


    print()
    print("Density = {:d} kg/m^3".format(int(bulk_density)))
    print("Gamma*A = {:.2f}".format(gamma_a))
    print("Decel = {:.2f} +/- {:.2f} km/s^2".format(decel/1000, decel_std/1000))
    print()
    print("Dynamic mass at {:.2f} km and {:.2f} km/s:".format(ht_eval/1000, vel_eval/1000))
    if mc_results is not None and len(mc_results['dyn_mass']) > 0:
        print("(+/-2 sigma from the deceleration uncertainty only; local estimate on the nominal "
            "trajectory, superseded by the MC 95% CI below, which also propagates the trajectory "
            "solver's radiant/geometry/timing uncertainty)")
    else:
        print("(+/-2 sigma from the deceleration uncertainty only)")
    print("-2sigma = {:.3f} kg".format(dyn_mass_lo))
    print("Nominal = {:.3f} kg".format(dyn_mass))
    print("+2sigma    = {:.3f} kg".format(dyn_mass_hi))

    if mc_results is not None and len(mc_results['dyn_mass']) > 0:
        dm_lo, dm_med, dm_hi = np.percentile(mc_results['dyn_mass'], [2.5, 50, 97.5])
        print()
        print("(Monte Carlo propagation of the WMPL trajectory solver uncertainties, {:d} "
            "realizations, 95% CI)".format(len(mc_results['dyn_mass'])))
        print("MC 2.5%  = {:.3f} kg".format(dm_lo))
        print("MC Median= {:.3f} kg".format(dm_med))
        print("MC 97.5% = {:.3f} kg".format(dm_hi))

    print()
    print("Simulation down to 3 km/s:")
    print("------------------------------------")
    print("Azim (+E of due N) and Elev: apparent ground-fixed radiant, epoch of date, gravity turn included")
    if mc_results is not None and len(mc_results.get('final_mass', [])) > 0:
        print("(-2sigma/nominal/+2sigma below vary only the dynamic mass on the nominal trajectory; "
            "see the MC block below for the full solver-uncertainty propagation)")
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

    if mc_results is not None and len(mc_results.get('final_mass', [])) > 0:
        n_mc = len(mc_results['final_mass'])
        mass_lo, mass_med, mass_hi = np.percentile(mc_results['final_mass'], [2.5, 50, 97.5])
        lat_lo, lat_med, lat_hi = np.percentile(mc_results['final_lat'], [2.5, 50, 97.5])
        lon_lo, lon_med, lon_hi = np.percentile(mc_results['final_lon'], [2.5, 50, 97.5])
        ele_lo, ele_med, ele_hi = np.percentile(mc_results['final_ele'], [2.5, 50, 97.5])
        azim_lo, azim_med, azim_hi = np.percentile(mc_results['final_azim'], [2.5, 50, 97.5])
        elev_lo, elev_med, elev_hi = np.percentile(mc_results['final_elev'], [2.5, 50, 97.5])
        fdecel_lo, fdecel_med, fdecel_hi = np.percentile(mc_results['final_decel'], [2.5, 50, 97.5])

        print()
        print("Final end coordinates (Monte Carlo, {:d} realizations, 95% CI)".format(n_mc))
        print("Mass      = {:.3f} kg     [{:.3f}, {:.3f}]".format(mass_med, mass_lo, mass_hi))
        print("Lat (+N)  = {:.5f} deg    [{:.5f}, {:.5f}]".format(lat_med, lat_lo, lat_hi))
        print("Lon (+E)  = {:.5f} deg    [{:.5f}, {:.5f}]".format(lon_med, lon_lo, lon_hi))
        print("Ele MSL   = {:.2f} km     [{:.2f}, {:.2f}]".format(ele_med, ele_lo, ele_hi))
        print("Azim      = {:.5f} deg    [{:.5f}, {:.5f}]".format(azim_med, azim_lo, azim_hi))
        print("Elev      = {:.5f} deg    [{:.5f}, {:.5f}]".format(elev_med, elev_lo, elev_hi))
        print("End decel = {:.3f} km/s^2 [{:.3f}, {:.3f}]".format(fdecel_med/1000, fdecel_lo/1000, \
            fdecel_hi/1000))

    elif mc_results is not None:
        # Ran with --mc_no_final_sim: only the dynamic mass at the evaluation point was propagated
        print()
        print("Monte Carlo uncertainty propagation (WMPL trajectory solver uncertainties, 95% CI):")
        print("------------------------------------")
        printMCPercentiles(mc_results)


    ax2.invert_yaxis()
    ax2.set_ylabel("Time (s)")
    ax2.set_xlabel("Velocity (km/s)")
    
    ax1.legend()
    ax2.legend()

    # Add padding to the simulation axis so the legend fits
    ax2_y_min, ax2_y_max = ax2.get_ylim()
    ax2.set_ylim(ax2_y_min, 0.75*ax2_y_max)


    plt.tight_layout()


    # Save plot to pickle directory
    plt.savefig(os.path.join(dir_path, traj.file_name + "_dyn_mass_fit.png"), dpi=300)

    plt.show()