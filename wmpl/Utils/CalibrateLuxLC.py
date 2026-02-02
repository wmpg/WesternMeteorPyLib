""" Given raw light curve data, this script will take the trajectory and compute the calibrated light curve."""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.signal
import scipy.interpolate
import scipy.integrate

from wmpl.Trajectory.Trajectory import Trajectory
from wmpl.Utils.Math import lineFunc, vectMag, vectNorm, angleBetweenSphericalCoords
from wmpl.Utils.Pickling import loadPickle
from wmpl.Utils.Physics import calcMass, calcRadiatedEnergy
from wmpl.Utils.TrajConversions import jd2Date, geo2Cartesian, cartesian2Geo, eci2RaDec, raDec2AltAz



def atmosphericExtinction(altitude, obs_ht, k=0.2):
    """ Compute the atmospheric extinction given the altitude above horizon.
    
    Arguments:
        mag: [float] Magnitude.
        altitude: [float] Altitude above horizon in degrees.
        obs_ht: [float] Observer height in meters.

    Keyword arguments:
        k: [float] Extinction coefficient. Default is 0.2 (V band).

    Returns:
        [float] Magnitude correction for extinction.
    """

    # Compute the zenith angle
    z = np.radians(90.0 - altitude)

    # Convert height to km
    h = obs_ht/1000

    # Compute the air mass
    x = (np.cos(z) + 0.025*np.exp(-11*np.cos(z)))**(-1)

    # Compute the correction
    return k*x

    




if __name__ == "__main__":

    import argparse


    ### Define the command line arguments ###
    arg_parser = argparse.ArgumentParser(description="Calibrate light curve data collected by the lux meter given a trajectory file.")

    arg_parser.add_argument("light_curve_file", help="The light curve file to calibrate.")

    arg_parser.add_argument("trajectory_file", help="The trajectory file to use for calibration.")

    arg_parser.add_argument("lc_peak_ht", help="The height of the light curve peak in km. This will be used to align the light curve with the trajectory.")

    arg_parser.add_argument("time_range", help="The comma-separated relative time range speficying where the fireball is in the LC data. E.g. 2.5,6.0")

    arg_parser.add_argument("obs_geocoords", help="The comma-separated coordinates of the observer (lat,lon,ele) in degrees and meters. If the latitude is negative, put the coordinates in quotes and have a leading space, e.g. \" -45.0,120.0,100\".")

    arg_parser.add_argument("--pointing", help="The pointing direction of the sensor. Assumed zenith by default. If given, it must be a comma-separated list of (az,el) in degrees (azimuth is +E of due N).", default="0,90")

    arg_parser.add_argument("--timehtfit", type=int, choices=[1, 2, 3], default=None, help="If given, a polynomial fit of the specified order (1, 2, or 3) will be used for the time vs height relationship instead of interpolation.")

    arg_parser.add_argument("--tau", type=float, default=None, help="The luminous efficiency of fireballs in %. If given, it will be used instead of the default value of 5% which is appropraite for low speed fireballs.")


    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    ### ###


    ### CONSTANTS ###

    # Power of a zero magnitude meteor
    P_0M = 1500

    # Luminous efficacy at 5800 K
    LUM_EFFICACY = 0.0079 # 1 lux in W/m^2

    # Luminous efficiency of fireballs at low speeds
    if cml_args.tau is None:
        TAU = 5.0/100
    else:
        TAU = cml_args.tau/100


    ## Sensor parameters

    # Re irradiance responsivity from TSL2591 datasheet, white light on "visible" sensor channel 0
    # The 100 scaling factor is to convert the RE_WHITE_CHANNEL0 from the datsheet units of counts/(μW/cm2) to counts/(W/m2)
    RE_WHITE_CHANNEL0 = 264.1*100

    # High gain factor 428x from https://github.com/adafruit/Adafruit_CircuitPython_TSL2591/blob/main/adafruit_tsl2591.py
    GAIN_HIGH = 428

    ##


    ### ###





    # Check that the light curve file exists
    if not os.path.exists(cml_args.light_curve_file):
        raise ValueError("The light curve file does not exist.")
    
    # Get the directory path of the light curve file
    dir_path = os.path.dirname(os.path.realpath(cml_args.light_curve_file))
    
    # Load the lux sensor data (handle both number and text column)
    columns = ["Date", "Time", "Lux", "Visible", "IR", "Gain", "IntTime"]
    lc_data = pd.read_csv(cml_args.light_curve_file, header=None, delimiter=' ', names=columns)
    print(lc_data)


    # Load the trajectory data
    if not os.path.exists(cml_args.trajectory_file):
        raise ValueError("The trajectory file does not exist.")
    
    traj = loadPickle(*os.path.split(cml_args.trajectory_file))

    # Extract the peak height
    peak_ht = float(cml_args.lc_peak_ht)


    # Get the fireball time range
    if "," not in cml_args.time_range:
        raise ValueError("The time range must be comma-separated.")
    
    time_range = np.array(cml_args.time_range.split(','), dtype=float)

    # Extract the observer coordinates
    if "," not in cml_args.obs_geocoords:
        raise ValueError("The observer coordinates must be comma-separated.")
    
    obs_lat, obs_lon, obs_ht = np.array(cml_args.obs_geocoords.split(','), dtype=float)

    print("Observer coordinates:")
    print(" Lat: {:.5f} deg +N".format(obs_lat))
    print(" Lon: {:.5f} deg +E".format(obs_lon))
    print(" Ele: {:.2f} m".format(obs_ht))


    # Read the sensor pointing
    if "," not in cml_args.pointing:
        raise ValueError("The sensor pointing must be comma-separated.")

    sensor_azim, sensor_alt = np.array(cml_args.pointing.split(','), dtype=float)



    # Extract the relative time and lux values
    # The columns are: "Date", "Time", "Lux", "Visible", "IR", "Gain", "IntTime"
    date_date = lc_data["Date"].values
    time_data = lc_data["Time"].values

    
    ### EXTRACT LUX VALUES ###

    # # Use the computed lux values
    # lux_data = lc_data["Lux"].values

    
    # OR: 

    # Calculate the gain scaling
    # The RE_WHITE_CHANNEL0 measured in the datasheet is measured at high gain, so divide by the GAIN_HIGH factor
    # Note: datasheet says the gain scaling for max gain is 9200/400
    gain_scaling = np.array(lc_data["Gain"].values, dtype=float)/GAIN_HIGH

    # Get watts/m2 from visibale data
    visible_data = np.array(lc_data["Visible"].values, dtype=float)
    watts_per_square_meter = visible_data/(RE_WHITE_CHANNEL0*gain_scaling)

    # Compute the lux from visible data
    lux_data = watts_per_square_meter/LUM_EFFICACY

    ### ###




    # Merge the date and time columns and convert to datetime
    date_time = pd.to_datetime(date_date + ' ' + time_data)

    # Compute relative time in seconds
    time_data = (date_time - date_time[0]).total_seconds().values
    

    ### Fit a line to the lux for background subtraction (use soft l1 for outlier rejection) ###

    # Iteratively fit a line to the background
    # Initial mask includes the faintest 80% of data to exclude the fireball peak
    bg_mask = lux_data < np.percentile(lux_data, 80)

    # Perform 10 iterations
    for _ in range(10):

        # Fit a line to the data
        bg_params = np.polyfit(time_data[bg_mask], lux_data[bg_mask], 1)

        # Compute the model
        bg_model = np.polyval(bg_params, time_data)

        # Compute residuals
        residuals = lux_data - bg_model

        # Compute the standard deviation of the residuals
        sigma = np.std(residuals[bg_mask])

        # Update the mask (reject positive outliers > 2 sigma)
        # We only care about positive outliers (fireball)
        new_bg_mask = residuals < 2*sigma

        # Check if the mask has converged
        if np.sum(new_bg_mask) == np.sum(bg_mask):
            bg_mask = new_bg_mask
            break

        bg_mask = new_bg_mask

    print("Background subtraction parameters: m={:.3e}, k={:.3e} (iterations={:d})".format(bg_params[0], bg_params[1], 10))

    # Convert to the format expected by lineFunc (m, k) -> (x, m, k)
    # np.polyfit returns (m, k)
    # lineFunc takes (x, m, k)
    # So we can just unpack *bg_params in the next step, but be careful of order. 
    # np.polyfit returns highest power first, so [slope, intercept].
    # lineFunc definition is m*x + k. So it matches.


    ### ###

    # Compute background-subtracted lux
    lux_data = lux_data - np.polyval(bg_params, time_data)


    # Only take the data within the specified time range and with lux > 0
    filter_mask = (time_data >= time_range[0]) & (time_data <= time_range[1]) & (lux_data > 0)
    time_data_fireball = time_data[filter_mask]
    lux_data_fireball = lux_data[filter_mask]

    # Identify the peak lux and its time
    peak_lux = np.max(lux_data_fireball)
    peak_time = time_data_fireball[np.where(lux_data_fireball == peak_lux)[0][0]]


    # Plot the extracted lux to check that the fireball data was extracted correctly

    # Plot all data
    plt.scatter(time_data, lux_data, c='k', s=1, label="All data")

    # Plot the fireball data
    plt.plot(time_data_fireball, lux_data_fireball, color='r', linewidth=1, label="Fireball data")

    # Plot the peak lux with an empty red circle
    plt.plot(peak_time, peak_lux, 'ro', markerfacecolor='none', label="Peak\nt = {:.3f} s\nh = {:.2f} km".format(peak_time, peak_ht))

    # Plot a horizontal zero line
    plt.plot([time_data[0], time_data[-1]], [0, 0], 'k--', label='Background')

    plt.xlabel('Time (s)')
    plt.ylabel('Apparent illuminance (lux)')

    plt.legend()

    plot_name = os.path.basename(os.path.realpath(cml_args.light_curve_file)).replace(".csv", "_fireball_lux.png")
    plt.savefig(os.path.join(dir_path, plot_name), dpi=300)

    plt.show()


    ### Interpolate the trajectory time vs height ###

    time_data_traj = np.concatenate([obs.time_data for obs in traj.observations])
    ht_data_traj = np.concatenate([obs.model_ht for obs in traj.observations])


    # Sort the trajectory by time
    sort_idx = np.argsort(time_data_traj)
    time_data_traj, ht_data_traj = time_data_traj[sort_idx], ht_data_traj[sort_idx]


    # Check if a fit should be performed instead of interpolation
    if cml_args.timehtfit is not None:

        print("Fitting a polynomial of order {:d} to time vs height...".format(cml_args.timehtfit))

        # Fit a polynomial to the data
        poly_ht = np.poly1d(np.polyfit(time_data_traj, ht_data_traj, cml_args.timehtfit))

        # Define the height interpolator as a function of time
        ht_interp = lambda t: poly_ht(t)

        # Compute the interpolated data for plotting
        time_data_interp = np.linspace(time_data_traj[0], time_data_traj[-1], 1000)
        ht_data_interp = ht_interp(time_data_interp)

        # Define the inverse interpolator (time as a function of height)
        def time_ht_interp(h):
            
            # Find the roots of the polynomial minus the height
            roots = (poly_ht - h).roots

            # Take the real roots
            roots = roots[np.isreal(roots)].real

            # Take the root that is within the time range
            # If there are multiple, take the one closest to the middle of the time range
            if len(roots) > 0:
                return roots[np.argmin(np.abs(roots - np.mean(time_data_traj)))]
            else:
                return np.nan

    else:

        # Interpolate the trajectory time vs height
        ht_interp = scipy.interpolate.PchipInterpolator(time_data_traj, ht_data_traj)

        # Compute the interpolated data
        time_data_interp = np.linspace(time_data_traj[0], time_data_traj[-1], 1000)
        ht_data_interp = ht_interp(time_data_interp)

        # Smooth the interpolated data
        ht_data_interp = scipy.signal.savgol_filter(ht_data_interp, 21, 3)

        # Interpolate again after smoothing
        ht_interp = scipy.interpolate.PchipInterpolator(time_data_interp, ht_data_interp)

        # Interpolate the inverse, i.e. height vs time (sort the interpolated data by height first)
        sort_idx = np.argsort(ht_data_interp)
        time_ht_interp = scipy.interpolate.PchipInterpolator(ht_data_interp[sort_idx], time_data_interp[sort_idx])


    ### ###


    # Plot time vs height from the trajectory
    for obs in traj.observations:

        plt.scatter(obs.time_data, obs.model_ht/1000, label=obs.station_id, marker='x')

    # Plot the interpolated data
    plt.plot(time_data_interp, ht_data_interp/1000, 'r-', label='Interpolated')

    plt.xlabel('Time (s)')
    plt.ylabel('Height (km)')

    plt.legend()

    plot_name = os.path.basename(os.path.realpath(cml_args.light_curve_file)).replace(".csv", "_time_vs_ht.png")
    plt.savefig(os.path.join(dir_path, plot_name), dpi=300)

    plt.show()


    ### Interpolate the trajectory time vs length ###
    
    time_data_traj = np.concatenate([obs.time_data for obs in traj.observations])
    len_data_traj = np.concatenate([obs.state_vect_dist for obs in traj.observations])

    # Sort the trajectory by time
    sort_idx = np.argsort(time_data_traj)
    time_data_traj, len_data_traj = time_data_traj[sort_idx], len_data_traj[sort_idx]


    # Check if a fit should be performed instead of interpolation
    if cml_args.timehtfit is not None:
        
        print("Fitting a polynomial of order {:d} to time vs length...".format(cml_args.timehtfit))

        # Fit a polynomial to the data
        poly_len = np.poly1d(np.polyfit(time_data_traj, len_data_traj, cml_args.timehtfit))

        # Define the length interpolator as a function of time
        len_interp = lambda t: poly_len(t)

        # Compute the interpolated data for plotting
        time_data_interp = np.linspace(time_data_traj[0], time_data_traj[-1], 1000)
        len_data_interp = len_interp(time_data_interp)

    else:

        # Interpolate the trajectory time vs length
        len_interp = scipy.interpolate.PchipInterpolator(time_data_traj, len_data_traj)

        # Compute the interpolated data
        time_data_interp = np.linspace(time_data_traj[0], time_data_traj[-1], 1000)
        len_data_interp = len_interp(time_data_interp)

        # Smooth the interpolated data
        len_data_interp = scipy.signal.savgol_filter(len_data_interp, 21, 3)

        # Interpolate again after smoothing
        len_interp = scipy.interpolate.PchipInterpolator(time_data_interp, len_data_interp)

    ### ###


    # Plot time vs length from the trajectory
    for obs in traj.observations:

        plt.scatter(obs.time_data, obs.state_vect_dist/1000, label=obs.station_id, marker='x')

    # Plot the interpolated data
    plt.plot(time_data_interp, len_data_interp/1000, 'r-', label='Interpolated')

    plt.xlabel('Time (s)')
    plt.ylabel('Trajectory length (km)')

    plt.legend()

    plot_name = os.path.basename(os.path.realpath(cml_args.light_curve_file)).replace(".csv", "_time_vs_len.png")
    plt.savefig(os.path.join(dir_path, plot_name), dpi=300)

    plt.show()



    ### Compute the corrected light curve ###

    # Find the trajectory time at the peak height
    traj_peak_time = time_ht_interp(1000*peak_ht)

    print("Peak time: {:.3f} s".format(peak_time))
    print("Trajectory peak time: {:.3f} s".format(traj_peak_time))

    # Compute the LC time in the trajectory frame
    lc_time_traj = time_data_fireball + traj_peak_time - peak_time

    # Compute the LC height
    lc_ht = ht_interp(lc_time_traj)

    # Compute the LC length
    lc_len = len_interp(lc_time_traj)


    ## Compute the ECI coordinates of the fireball over time

    lat_fireball_data = []
    lon_fireball_data = []
    height_fireball_data = []
    range_fireball_data = []
    alt_fireball_data = []
    power_data = []
    extinction_data = []
    abs_mag_fireball_data = []

    # Get ECI coordinates of the state vector over time
    for t, lux in zip(lc_time_traj, lux_data_fireball):

        # Compute the Julian date of the point
        jd = traj.jdt_ref + t/86400

        # Compute the distance traveled from the state vector at the given time
        sv_dist = len_interp(t)

        # Compute the ECI coordinates of the fireball at the given time
        fireball_eci = traj.state_vect_mini - sv_dist*traj.radiant_eci_mini

        # Compute the height of the fireball above the ground
        lat_fireball, lon_fireball, ht_fireball = cartesian2Geo(jd, *fireball_eci)


        # Compute ECI coordinates of the observer at the given time
        obs_eci = np.array(geo2Cartesian(np.radians(obs_lat), np.radians(obs_lon), obs_ht, jd))

        # Compute the distance between the fireball and the observer
        range_fireball = vectMag(fireball_eci - obs_eci)

        
        # Compute the vector pointing from the observer to the fireball
        fireball_obs_vect = vectNorm(fireball_eci - obs_eci)

        # Compute the apparent altitude of the fireball
        ra, dec = eci2RaDec(fireball_obs_vect)
        azim, alt = raDec2AltAz(ra, dec, jd, np.radians(obs_lat), np.radians(obs_lon))

        print()
        print("Time: {:.3f} s".format(t))
        print("Fireball range: {:.3f} km".format(range_fireball/1000))
        print("Fireball altitude: {:.3f} deg".format(np.degrees(alt)))


        # Compute the angular distance between the sensor centre and the fireball
        sensor_ang_dist = angleBetweenSphericalCoords(alt, azim, np.radians(sensor_alt), np.radians(sensor_azim))


        # Compute sensor sensitivity correction (assuming a simple cosine response)
        sensor_corr = 1/np.cos(sensor_ang_dist)

        # Compute the corrected lux
        lux_corr = lux*sensor_corr

        # Compute the power over area using an assumed luminous efficacy
        power_area = lux_corr*LUM_EFFICACY # W/m^2

        # Compute the total power emitted by the fireball by applying a range correction
        power = power_area*4*np.pi*range_fireball**2 # W

        # Compute the absolute magnitude of the fireball
        abs_mag = -2.5*np.log10(power/P_0M)

        # Correct the absolute magnitude for extinction
        extinction = atmosphericExtinction(np.degrees(alt), obs_ht)
        abs_mag -= extinction

        # Compute the extinction-corrected power
        power = P_0M*10**(abs_mag/-2.5)

        print("Power: {:.3f} W".format(power))
        print("Extinction correction: {:.3f} mag".format(extinction))
        print("Absolute magnitude: {:.3f}".format(abs_mag))


        # Save the data
        lat_fireball_data.append(lat_fireball)
        lon_fireball_data.append(lon_fireball)
        height_fireball_data.append(ht_fireball)
        range_fireball_data.append(range_fireball)
        alt_fireball_data.append(alt)
        power_data.append(power)
        extinction_data.append(extinction)
        abs_mag_fireball_data.append(abs_mag)





    ##

    # Compute the total radiate energy
    energy = calcRadiatedEnergy(lc_time_traj, np.array(abs_mag_fireball_data), P_0m=P_0M)

    # Compute the photometric mass
    mass = calcMass(lc_time_traj, np.array(abs_mag_fireball_data), traj.orbit.v_avg_norot, tau=TAU, P_0m=P_0M)

    print()
    print("-" * 50)
    print("Peak absolute magnitude: {:.3f}".format(np.min(abs_mag_fireball_data)))
    print("Average velocity: {:.3f} km/s".format(traj.orbit.v_avg_norot/1000))
    print("Radiated energy: {:.3f} J".format(energy))
    print("Assumed luminous efficacy (5800 K): {:.3f} lm/W".format(LUM_EFFICACY))
    print("Assumed luminous efficiency: {:.2f} %".format(100*TAU))
    print("Photometric mass: {:.3f} kg".format(mass))



    # Save the computed parameters to an output file next to the input file
    file_name = os.path.basename(os.path.realpath(cml_args.light_curve_file)).replace(".csv", "_calibrated.csv")

    # Compute the reference time
    ref_dt = jd2Date(traj.jdt_ref, dt_obj=True).strftime("%Y-%m-%d %H:%M:%S.%f")

    with open(os.path.join(dir_path, file_name), 'w') as f:
        f.write("# Reference time: {:s}\n".format(ref_dt))
        f.write("# Time (s), Lat (deg), Lon (deg), Height (m), Range (m), Alt (deg), Power (W), Extinction (mag), Abs mag\n")

        for t, lat, lon, ht, r, alt, p, ext, abs_mag in zip(lc_time_traj, lat_fireball_data, 
                                                            lon_fireball_data, height_fireball_data, 
                                                            range_fireball_data, alt_fireball_data, 
                                                            power_data, extinction_data, 
                                                            abs_mag_fireball_data):
                
                f.write("{:.3f}, {:.6f}, {:.6f}, {:.3f}, {:.3f}, {:.6f}, {:.3f}, {:.6f}, {:.6f}\n".format(t, 
                    np.degrees(lat), np.degrees(lon), ht, r, np.degrees(alt), p, ext, abs_mag))




    fig, (ax_time, ax_ht) = plt.subplots(ncols=2, sharex=True, figsize=(10, 8))

    ax_time.plot(abs_mag_fireball_data, lc_time_traj, color='k')
    ax_time.invert_xaxis()
    ax_time.invert_yaxis()

    ax_time.set_xlabel("Absolute magnitude")
    ax_time.set_ylabel("Time after {:s} (s)".format(ref_dt))


    # Plot the LC in the trajectory frame
    ax_ht.plot(abs_mag_fireball_data, lc_ht/1000, color='k')

    ax_ht.set_ylabel("Height (km)")
    ax_ht.set_xlabel("Absolute magnitude")

    plt.tight_layout()

    # Save the plot
    plot_name = os.path.basename(os.path.realpath(cml_args.light_curve_file)).replace(".csv", "_lc.png")
    plt.savefig(os.path.join(dir_path, plot_name), dpi=300)

    plt.show()



    ### ### 