import os

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from wmpl.Utils.AtmosphereDensity import fitAtmPoly
from wmpl.Utils.Math import lineFunc
from wmpl.Utils.TrajConversions import cartesian2Geo
from wmpl.Utils.Physics import dynamicMass
from wmpl.Utils.Pickling import loadPickle
from wmpl.MetSim.MetSimErosion import Constants, runSimulation
from wmpl.MetSim.GUI import SimulationResults


def runFragSim(mass, density, lat, lon, jd, ht_beg, v_init, entry_angle):

    # Init simulation constants
    const = Constants()


    # Set minimum simulation height
    const.h_kill = 15000 # m 

    # Set minimum simuation speed
    const.v_kill = 3000 # m/s


    # Set meteoroid parametrs
    const.m_init = mass
    const.v_init = v_init
    const.h_init = ht_beg
    const.rho = density
    const.gamma = 1.0
    const.shape_factor = 1.21
    

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



def computeFragEndParams(traj, dyn_mass, density, hend, vend, gamma_a):

    jd = traj.jdt_ref
    lat = np.degrees(traj.rend_lat)
    lon = np.degrees(traj.rend_lon)

    entry_angle = np.degrees(traj.orbit.elevation_apparent_norot)

    # Fit an interpolation function from time to height
    ht_vs_time_interp, ht_vs_len_interp = interpolateHtVsTimeLen(traj, sample_step=0.1, show_plots=False)

    # # Compute the dynamic mass (upper range)
    # dyn_mass = dynamicMass(density, np.radians(lat), np.radians(lon), hend, jd, vend, decel, \
    #     gamma=1.0, shape_factor=gamma_a)

    # print("  vel        = {:.2f} km/s".format(vend/1000))
    # print("  decel      = {:.2f} km/s^2".format(decel/1000))
    # print("  dyn mass   = {:.3f} kg".format(dyn_mass))


    # Get the time and length at the observed point
    meas_time = ht_vs_time_interp(hend)
    meas_len  = ht_vs_len_interp(hend)
    

    # Run the simulation until ablation stops
    sr = runFragSim(dyn_mass, density, lat, lon, jd, hend, vend, entry_angle)

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
    final_eci = traj.state_vect_mini - total_len*traj.radiant_eci_mini

    # Compute exact time of the end
    final_jd = jd + total_time/86400

    # Compute the geo coordiantes
    final_lat, final_lon, final_ele = cartesian2Geo(final_jd, *final_eci)

    ###



    print("  final mass     = {:.3f} kg".format(sr.frag_main.m))
    print("  final vel      = {:.3f} km/s".format(sr.frag_main.v/1000))
    print("  final ht (sim) = {:.3f} km".format(final_ht/1000))
    print("  total len      = {:.3f} km".format(total_len/1000))
    print("  total time     = {:.3f} s".format(total_time))
    print("  final lat      = {:.5f} deg".format(np.degrees(final_lat)))
    print("  final lon      = {:.5f} deg".format(np.degrees(final_lon)))
    print("  final ht       = {:.3f} km".format(final_ele/1000))


    return sr, sr.frag_main.m, np.degrees(final_lat), np.degrees(final_lon), final_ele/1000


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
        help='The product of the drag coefficient Gamma and the shape coefficient A. Used for computing the dynamic mass. Default is 0.7.', \
        type=float, default=0.7)

    arg_parser.add_argument('-e', '--eval', metavar='EVAL_PT', \
        help='Point where to evaluate the dynamic mass (0 = ht_min, 1 = ht_max). Default is 0.5.', \
        type=float, default=0.5)

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
    

    # Load the trajectory
    traj = loadPickle(*os.path.split(os.path.abspath(cml_args.traj_path)))

    dir_path = os.path.dirname(cml_args.traj_path)


    # Top height (if nagative, e.g. -3, the last 3 km from the bottom will be taken)
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

        # Only take velocities inside a reasonable range
        vel_filter = (vel > 0) & (vel < 73_000)

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


    # Only take data in the approprite range of heights
    ht_filter = (ht_data/1000 >= ht_min) & (ht_data/1000 <= ht_max)
    vel_data = vel_data[ht_filter]
    time_data = time_data[ht_filter]
    ht_data = ht_data[ht_filter]

    # Plot the selected data
    ax2.scatter(vel_data/1000, time_data, s=5, label="Measurements")
    


    # Fit a line to the velocity data in the range
    vel_fit, vel_fit_cov = scipy.optimize.curve_fit(lineFunc, time_data, vel_data)

    # Compute the standard deviations of the fit
    vel_fit_std = np.sqrt(np.diag(vel_fit_cov))
    decel_std = vel_fit_std[0]

    # Remove 5 sigma outliers from the data and re-fit
    vel_filter = np.abs(vel_data - lineFunc(time_data, *vel_fit)) < 5*decel_std
    vel_fit, vel_fit_cov = scipy.optimize.curve_fit(lineFunc, time_data[vel_filter], vel_data[vel_filter])

    # Plot the selected outliers as an empty red circle
    ax2.scatter(vel_data[~vel_filter]/1000, time_data[~vel_filter], s=20, marker='o', facecolors='none', 
        edgecolors='r', label="$5\\sigma$ outliers")



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
    ht_eval = lineFunc(time_eval, *ht_fit)

    # Compute the dynamic mass (and +/- 2 sigma)
    dyn_mass = dynamicMass(bulk_density, traj.rend_lat, traj.rend_lon, traj.rend_ele, traj.jdt_ref, \
        vel_eval, decel, gamma=1.0, shape_factor=gamma_a)
    dyn_mass_hi = dynamicMass(bulk_density, traj.rend_lat, traj.rend_lon, traj.rend_ele, traj.jdt_ref, \
        vel_eval, decel - 2*decel_std, gamma=1.0, shape_factor=gamma_a)
    dyn_mass_lo = dynamicMass(bulk_density, traj.rend_lat, traj.rend_lon, traj.rend_ele, traj.jdt_ref, \
        vel_eval, decel + 2*decel_std, gamma=1.0, shape_factor=gamma_a)


    final_decel = final_decel_hi = final_decel_lo = 0

    # Run the fragment until the final velocity of 3 km/s
    if vel_eval > 3000:

        final_sr, final_mass, final_lat, final_lon, final_ele = computeFragEndParams(traj, dyn_mass, \
            bulk_density, ht_eval, vel_eval, gamma_a)

        final_sr_hi, final_mass_hi, final_lat_hi, final_lon_hi, final_ele_hi = computeFragEndParams(traj, \
            dyn_mass_hi, bulk_density, ht_eval, vel_eval, gamma_a)

        final_sr_lo, final_mass_lo, final_lat_lo, final_lon_lo, final_ele_lo = computeFragEndParams(traj, \
            dyn_mass_lo, bulk_density, ht_eval, vel_eval, gamma_a)

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



    # Plot the evaluation point
    ax2.scatter(vel_eval/1000, time_eval, color='g', marker='o', s=50, \
        label="Dyn $m$ = [{:.3f}, {:.3f}, {:.3f}] kg\n"
        "Final $m$ = [{:.3f}, {:.3f}, {:.3f}] kg\n"
        "Decel = {:.2f} $\\pm$ {:.2f} km/s$^2$\n"
        "V = {:.2f} km/s\n"
        "h = {:.2f} km\n"
        "$\\rho_m$ = {:d} kg/m$^3$\n"
        "$\\Gamma A$ = {:.2f}".format( \
            dyn_mass_lo, dyn_mass, dyn_mass_hi, 
            final_mass_lo, final_mass, final_mass_hi, 
            decel/1000, decel_std/1000, 
            vel_eval/1000, 
            ht_eval/1000, 
            int(bulk_density), 
            gamma_a)
        )


    print()
    print("Density = {:d} kg/m^3".format(int(bulk_density)))
    print("Gamma*A = {:.2f}".format(gamma_a))
    print("Decel = {:.2f} +/- {:.2f} km/s^2".format(decel/1000, decel_std/1000))
    print()
    print("Dynamic mass at {:.2f} km and {:.2f} km/s:".format(ht_eval/1000, vel_eval/1000))
    print("-2sigma = {:.3f} kg".format(dyn_mass_lo))
    print("Nominal = {:.3f} kg".format(dyn_mass))
    print("+2sigma    = {:.3f} kg".format(dyn_mass_hi))
    print()
    print("Simulation down to 3 km/s:")
    print("------------------------------------")
    print("Final end coordinates (-2sigma mass)")
    print("Mass      = {:.3f} kg".format(final_mass_lo))
    print("Lat (+N)  = {:.5f} deg".format(final_lat_lo))
    print("Lon (+E)  = {:.5f} deg".format(final_lon_lo))
    print("Ele MSL   = {:.2f} km".format(final_ele_lo))
    print("End decel = {:.3f} km/s^2".format(final_decel_lo/1000))
    print("Final end coordinates (nominal mass)")
    print("Mass      = {:.3f} kg".format(final_mass))
    print("Lat (+N)  = {:.5f} deg".format(final_lat))
    print("Lon (+E)  = {:.5f} deg".format(final_lon))
    print("Ele MSL   = {:.2f} km".format(final_ele))
    print("End decel = {:.3f} km/s^2".format(final_decel/1000))
    print("Final end coordinates (+2sigma mass)")
    print("Mass      = {:.3f} kg".format(final_mass_hi))
    print("Lat (+N)  = {:.5f} deg".format(final_lat_hi))
    print("Lon (+E)  = {:.5f} deg".format(final_lon_hi))
    print("Ele MSL   = {:.2f} km".format(final_ele_hi))
    print("End decel = {:.3f} km/s^2".format(final_decel_hi/1000))


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