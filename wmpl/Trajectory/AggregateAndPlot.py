""" Aggregate trajectory results into one results file and generate plots of trajectory results. """

from __future__ import print_function, division, absolute_import


import os
import time
import datetime
import shutil

import numpy as np
import scipy.integrate
import scipy.ndimage
import scipy.stats


from wmpl.Analysis.FitPopulationAndMassIndex import fitSlope, logline
from wmpl.Trajectory.Trajectory import addTrajectoryID
from wmpl.Utils.Math import mergeClosePoints, meanAngle, sphericalPointFromHeadingAndDistance, \
    angleBetweenSphericalCoords, generateMonthyTimeBins
from wmpl.Utils.OSTools import mkdirP
from wmpl.Utils.Physics import calcMass
from wmpl.Utils.Pickling import loadPickle
from wmpl.Utils.PlotCelestial import CelestialPlot
from wmpl.Utils.PlotMap import MapColorScheme
from wmpl.Utils.ShowerAssociation import associateShowerTraj, MeteorShower
from wmpl.Utils.SolarLongitude import jd2SolLonSteyaert, solLon2jdSteyaert
from wmpl.Utils.TrajConversions import jd2Date, datetime2JD




### CONSTANTS ###
    
# Trajectory summary file name
TRAJ_SUMMARY_FILE = "trajectory_summary.txt"


# Output directory for auto-generated data
AUTO_OUTPUT_DATA_DIR = "auto_output"

# Output directory inside AUTO_OUTPUT_DATA_DIR for summary files
AUTO_OUTPUT_SUMMARY_DIR = "traj_summary_data"

# Output directory for plots
AUTO_OUTPUT_PLOT_DIR = "plots"

# Output directory for daily plots
AUTO_OUTPUT_DAILY_DIR = "daily"

# Output directory for monthly plots
AUTO_OUTPUT_MONTHLY_DIR = "monthly"

# Summary file with all orbits
TRAJ_SUMMARY_ALL = "traj_summary_all.txt"

# Auto run frequency (hours)
AUTO_RUN_FREQUENCY = 2


# Minimum number of shower members to mark the shower
MIN_SHOWER_MEMBERS = 3

# Plot shower radius (deg)
PLOT_SHOWER_RADIUS = 3.0

# Power of a zero magnitude meteor
P_0M = 1210

### ###



def computeMass(traj, P_0m):
    """ Compute the mass given the trajectory and a power of a zero magnitude meteor. """


    time_mag_arr = []
    avg_t_diff_max = 0
    for obs in traj.observations:

        # Skip ignored stations
        if obs.ignore_station:
            continue

        # If there are not magnitudes for this site, skip it
        if obs.absolute_magnitudes is None:
            continue

        # Compute average time difference
        avg_t_diff_max = max(avg_t_diff_max, np.median(obs.time_data[1:] - obs.time_data[:-1]))

        # Exclude magnitudes fainter than mag +8
        mag_filter = obs.absolute_magnitudes < 8

        for t, mag in zip(obs.time_data[mag_filter], obs.absolute_magnitudes[mag_filter]):
            if (mag is not None) and (not np.isnan(mag)):
                time_mag_arr.append([t, mag])


    # Compute the mass
    time_mag_arr = np.array(sorted(time_mag_arr, key=lambda x: x[0]))
    time_arr, mag_arr = time_mag_arr.T
    
    # Take the brightest magnitudes, which mitigates saturation effects. Note that this assumes that the
    #   photometric calibration was done well
    time_arr, mag_arr = mergeClosePoints(time_arr, mag_arr, avg_t_diff_max, method='min')

    # Compute the photometric mass
    mass = calcMass(np.array(time_arr), np.array(mag_arr), traj.orbit.v_avg_norot, P_0m=P_0m)


    return mass



def computePeakMagHt(traj):
    """ Compute the peak magnitude and peak height. """

    # Compute peak magnitude from all stations
    peak_mags = [np.min(obs.absolute_magnitudes[obs.ignore_list == 0]) for obs in traj.observations \
        if obs.ignore_station == False]
    peak_mag = np.min(peak_mags)

    # Compute the peak height
    peak_ht = [obs.model_ht[np.argmin(obs.absolute_magnitudes[obs.ignore_list == 0])] \
        for obs in traj.observations if obs.ignore_station == False][np.argmin(peak_mags)]


    return peak_mag, peak_ht




def checkMeteorFOVBegEnd(traj):
    """ Check if the meteor begins or ends inside the FOV of at least one camera. """

    # Check if the meteor begins in the FOV of at least one camera
    fov_beg = None
    fov_beg_list = [obs.fov_beg for obs in traj.observations if (obs.ignore_station == False) \
        and hasattr(obs, "fov_beg")]
    if len(fov_beg_list) > 0:
        fov_beg = np.any(fov_beg_list)


    # Meteor ends inside the FOV
    fov_end = None
    fov_end_list = [obs.fov_end for obs in traj.observations if (obs.ignore_station == False) \
        and hasattr(obs, "fov_end")]
    if len(fov_end_list) > 0:
        fov_end = np.any(fov_end_list)


    return fov_beg, fov_end



def writeOrbitSummaryFile(dir_path, traj_list, traj_summary_file_name=TRAJ_SUMMARY_FILE, P_0m=1210):
    """ Given a list of trajectory files, generate CSV file with the orbit summary. """

    def _uncer(traj, str_format, std_name, multi=1.0, deg=False, max_val=None, max_val_format="{:7.1e}"):
        """ Internal function. Returns the formatted uncertanty, if the uncertanty is given. If not,
            it returns nothing. 

        Arguments:
            traj: [Trajectory instance]
            str_format: [str] String format for the unceertanty.
            std_name: [str] Name of the uncertanty attribute, e.g. if it is 'x', then the uncertanty is 
                stored in uncertainties.x.
    
        Keyword arguments:
            multi: [float] Uncertanty multiplier. 1.0 by default. This is used to scale the uncertanty to
                different units (e.g. from m/s to km/s).
            deg: [bool] Converet radians to degrees if True. False by defualt.
            max_val: [float] Larger number to use the given format. If the value is larger than that, the
                max_val_format is used.
            max_val_format: [str]
            """

        if deg:
            multi *= np.degrees(1.0)

        if traj.uncertainties is not None:
            if hasattr(traj.uncertainties, std_name):

                # Get the value
                val = getattr(traj.uncertainties, std_name)*multi

                # If the value is too big, use scientific notation
                if max_val is not None:
                    if val > max_val:
                        str_format = max_val_format

                return str_format.format(val)

        
        return "None"

    # Sort trajectories by Julian date
    traj_list = sorted(traj_list, key=lambda x: x.jdt_ref)


    delimiter = "; "

    out_str =  ""
    out_str += "# Summary generated on {:s} UTC\n\r".format(str(datetime.datetime.utcnow()))

    header = [" Unique trajectory", "     Beginning      ", "       Beginning          ", "  IAU", " IAU", "  Sol lon ", "  App LST ", "  RAgeo  ", "  +/-  ", "  DECgeo ", "  +/-  ", " LAMgeo  ", "  +/-  ", "  BETgeo ", "  +/-  ", "   Vgeo  ", "   +/- ", " LAMhel  ", "  +/-  ", "  BEThel ", "  +/-  ", "   Vhel  ", "   +/- ", "      a    ", "  +/-  ", "     e    ", "  +/-  ", "     i    ", "  +/-  ", "   peri   ", "   +/-  ", "   node   ", "   +/-  ", "    Pi    ", "  +/-  ", "     b    ", "  +/-  ", "     q    ", "  +/-  ", "     f    ", "  +/-  ", "     M    ", "  +/-  ", "      Q    ", "  +/-  ", "     n    ", "  +/-  ", "     T    ", "  +/-  ", "TisserandJ", "  +/-  ", "  RAapp  ", "  +/-  ", "  DECapp ", "  +/-  ", " Azim +E ", "  +/-  ", "   Elev  ", "  +/-  ", "  Vinit  ", "   +/- ", "   Vavg  ", "   +/- ", "   LatBeg   ", "  +/-  ", "   LonBeg   ", "  +/-  ", "  HtBeg ", "  +/-  ", "   LatEnd   ", "  +/-  ", "   LonEnd   ", "  +/-  ", "  HtEnd ", "  +/-  ", "Duration", " Peak ", " Peak Ht", "  F  ", " Mass kg", "  Qc ", "MedianFitErr", "Beg in", "End in", " Num", "     Participating    "]
    head_2 = ["     identifier   ", "    Julian date     ", "        UTC Time          ", "   No", "code", "    deg   ", "    deg   ", "   deg   ", " sigma ", "   deg   ", " sigma ", "   deg   ", " sigma ", "    deg  ", " sigma ", "   km/s  ", "  sigma", "   deg   ", " sigma ", "    deg  ", " sigma ", "   km/s  ", "  sigma", "     AU    ", " sigma ", "          ", " sigma ", "   deg    ", " sigma ", "    deg   ", "  sigma ", "    deg   ", "  sigma ", "   deg    ", " sigma ", "   deg    ", " sigma ", "    AU    ", " sigma ", "   deg    ", " sigma ", "    deg   ", " sigma ", "     AU    ", " sigma ", "  deg/day ", " sigma ", "   years  ", " sigma ", "          ", " sigma ", "   deg   ", " sigma ", "   deg   ", " sigma ", "of N  deg", " sigma ", "    deg  ", " sigma ", "   km/s  ", "  sigma", "   km/s  ", "  sigma", "   +N deg   ", " sigma ", "   +E deg   ", " sigma ", "    km  ", " sigma ", "   +N deg   ", " sigma ", "   +E deg   ", " sigma ", "    km  ", " sigma ", "  sec   ", "AbsMag", "    km  ", "param", "tau=0.7%", " deg ", "   arcsec   ", "  FOV ", "  FOV ", "stat", "        stations      "]
    out_str += "# {:s}\n\r".format(delimiter.join(header))
    out_str += "# {:s}\n\r".format(delimiter.join(head_2))

    # Add a horizontal line
    out_str += "# {:s}\n\r".format("; ".join(["-"*len(entry) for entry in header]))

    # Write lines of data
    for traj in traj_list:

        line_info = []

        line_info.append("{:20s}".format(traj.traj_id))

        line_info.append("{:20.12f}".format(traj.jdt_ref))
        line_info.append("{:26s}".format(str(jd2Date(traj.jdt_ref, dt_obj=True))))

        # Perform shower association
        shower_obj = associateShowerTraj(traj)
        if shower_obj is None:
            shower_no = -1
            shower_code = '...'
        else:
            shower_no = shower_obj.IAU_no
            shower_code = shower_obj.IAU_code

        line_info.append("{:>5d}".format(shower_no))
        line_info.append("{:>4s}".format(shower_code))


        # Geocentric radiant (equatorial and ecliptic)
        line_info.append("{:>10.6f}".format(np.degrees(traj.orbit.la_sun)))
        line_info.append("{:>10.6f}".format(np.degrees(traj.orbit.lst_ref)))
        line_info.append("{:>9.5f}".format(np.degrees(traj.orbit.ra_g)))
        line_info.append("{:>7s}".format(_uncer(traj, '{:7.4f}', 'ra_g', deg=True, max_val=100.0)))
        line_info.append("{:>+9.5f}".format(np.degrees(traj.orbit.dec_g)))
        line_info.append("{:>7s}".format(_uncer(traj, '{:7.4f}', 'dec_g', deg=True, max_val=100.0)))
        line_info.append("{:>9.5f}".format(np.degrees(traj.orbit.L_g)))
        line_info.append("{:>7s}".format(_uncer(traj, '{:7.4f}', 'L_g', deg=True, max_val=100.0)))
        line_info.append("{:>+9.5f}".format(np.degrees(traj.orbit.B_g)))
        line_info.append("{:>7s}".format(_uncer(traj, '{:7.4f}', 'B_g', deg=True, max_val=100.0)))
        line_info.append("{:>9.5f}".format(traj.orbit.v_g/1000))
        line_info.append("{:>7s}".format(_uncer(traj, '{:7.4f}', 'v_g', multi=1.0/1000)))

        # Ecliptic heliocentric radiant
        line_info.append("{:>9.5f}".format(np.degrees(traj.orbit.L_h)))
        line_info.append("{:>7s}".format(_uncer(traj, '{:7.4f}', 'L_h', deg=True, max_val=100.0)))
        line_info.append("{:>+9.5f}".format(np.degrees(traj.orbit.B_h)))
        line_info.append("{:>7s}".format(_uncer(traj, '{:7.4f}', 'B_h', deg=True, max_val=100.0)))
        line_info.append("{:>9.5f}".format(traj.orbit.v_h/1000))
        line_info.append("{:>7s}".format(_uncer(traj, '{:7.4f}', 'v_h', multi=1.0/1000)))

        # Orbital elements
        if abs(traj.orbit.a) < 1000:
            line_info.append("{:>11.6f}".format(traj.orbit.a))
        else:
            line_info.append("{:>11.2e}".format(traj.orbit.a))
        line_info.append("{:>7s}".format(_uncer(traj, '{:.4f}', 'a', max_val=100.0)))
        line_info.append("{:>10.6f}".format(traj.orbit.e))
        line_info.append("{:>7s}".format(_uncer(traj, '{:.4f}', 'e')))
        line_info.append("{:>10.6f}".format(np.degrees(traj.orbit.i)))
        line_info.append("{:>7s}".format(_uncer(traj, '{:.4f}', 'i', deg=True)))
        line_info.append("{:>10.6f}".format(np.degrees(traj.orbit.peri)))
        line_info.append("{:>8s}".format(_uncer(traj, '{:.4f}', 'peri', deg=True)))
        line_info.append("{:>10.6f}".format(np.degrees(traj.orbit.node)))
        line_info.append("{:>8s}".format(_uncer(traj, '{:.4f}', 'node', deg=True)))
        line_info.append("{:>10.6f}".format(np.degrees(traj.orbit.pi)))
        line_info.append("{:>7s}".format(_uncer(traj, '{:.4f}', 'pi', deg=True)))
        line_info.append("{:>10.6f}".format(np.degrees(traj.orbit.b)))
        line_info.append("{:>7s}".format(_uncer(traj, '{:.4f}', 'b', deg=True)))
        line_info.append("{:>10.6f}".format(traj.orbit.q))
        line_info.append("{:>7s}".format(_uncer(traj, '{:.4f}', 'q')))
        line_info.append("{:>10.6f}".format(np.degrees(traj.orbit.true_anomaly)))
        line_info.append("{:>7s}".format(_uncer(traj, '{:.4f}', 'true_anomaly', deg=True)))
        line_info.append("{:>10.6f}".format(np.degrees(traj.orbit.mean_anomaly)))
        line_info.append("{:>7s}".format(_uncer(traj, '{:.4f}', 'mean_anomaly', deg=True)))
        if abs(traj.orbit.Q) < 1000:
            line_info.append("{:>11.6f}".format(traj.orbit.Q))
        else:
            line_info.append("{:>11.4e}".format(traj.orbit.Q))
        line_info.append("{:>7s}".format(_uncer(traj, '{:.4f}', 'Q', max_val=100.0)))
        line_info.append("{:>10.6f}".format(np.degrees(traj.orbit.n)))
        line_info.append("{:>7s}".format(_uncer(traj, '{:.4f}', 'n', deg=True, max_val=100.0)))
        if traj.orbit.T < 1000:
            line_info.append("{:>10.6f}".format(traj.orbit.T))
        else:
            line_info.append("{:>10.4e}".format(traj.orbit.T))
        line_info.append("{:>7s}".format(_uncer(traj, '{:.4f}', 'T', max_val=100.0)))
        line_info.append("{:>10.6f}".format(traj.orbit.Tj))
        line_info.append("{:>7s}".format(_uncer(traj, '{:.4f}', 'Tj', max_val=100.0)))
        
        # Apparent radiant
        line_info.append("{:>9.5f}".format(np.degrees(traj.orbit.ra_norot)))
        line_info.append("{:>7s}".format(_uncer(traj, '{:7.4f}', 'ra_norot', deg=True, max_val=100.0)))
        line_info.append("{:>+9.5f}".format(np.degrees(traj.orbit.dec_norot)))
        line_info.append("{:>7s}".format(_uncer(traj, '{:7.4f}', 'dec_norot', deg=True, max_val=100.0)))
        line_info.append("{:>9.5f}".format(np.degrees(traj.orbit.azimuth_apparent_norot)))
        line_info.append("{:>7s}".format(_uncer(traj, '{:7.4f}', 'azimuth_apparent', deg=True, max_val=100.0)))
        line_info.append("{:>9.5f}".format(np.degrees(traj.orbit.elevation_apparent_norot)))
        line_info.append("{:>7s}".format(_uncer(traj, '{:7.4f}', 'elevation_apparent', deg=True, max_val=100.0)))
        line_info.append("{:>9.5f}".format(traj.orbit.v_init_norot/1000))
        line_info.append("{:>7s}".format(_uncer(traj, '{:7.4f}', 'v_init', multi=1.0/1000)))
        line_info.append("{:>9.5f}".format(traj.orbit.v_avg_norot/1000))
        line_info.append("{:>7s}".format(_uncer(traj, '{:7.4f}', 'v_avg', multi=1.0/1000)))

        # Begin/end point
        line_info.append("{:>12.6f}".format(np.degrees(traj.rbeg_lat)))
        line_info.append("{:>7s}".format(_uncer(traj, '{:.4f}', 'rbeg_lat', deg=True)))
        line_info.append("{:>12.6f}".format(np.degrees(traj.rbeg_lon)))
        line_info.append("{:>7s}".format(_uncer(traj, '{:.4f}', 'rbeg_lon', deg=True)))
        line_info.append("{:>8.4f}".format(traj.rbeg_ele/1000))
        line_info.append("{:>7s}".format(_uncer(traj, '{:.2f}', 'rbeg_ele', multi=1.0/1000)))
        line_info.append("{:>12.6f}".format(np.degrees(traj.rend_lat)))
        line_info.append("{:>7s}".format(_uncer(traj, '{:.4f}', 'rend_lat', deg=True)))
        line_info.append("{:>12.6f}".format(np.degrees(traj.rend_lon)))
        line_info.append("{:>7s}".format(_uncer(traj, '{:.4f}', 'rend_lon', deg=True)))
        line_info.append("{:>8.4f}".format(traj.rend_ele/1000))
        line_info.append("{:>7s}".format(_uncer(traj, '{:.2f}', 'rend_ele', multi=1.0/1000)))

        
        
        # Compute the duration
        duration = max([np.max(obs.time_data[obs.ignore_list == 0]) for obs in traj.observations \
            if obs.ignore_station == False])
        

        # Compute the peak magnitude and height
        peak_mag, peak_ht = computePeakMagHt(traj)

        # Compute the F parameter
        f_param = (traj.rbeg_ele - peak_ht)/(traj.rbeg_ele - traj.rend_ele)


        # Compute the mass
        mass = computeMass(traj, P_0m=P_0m)



        # Meteor parameters (duration, peak magnitude, integrated intensity, Q angle)
        line_info.append("{:8.2f}".format(duration))
        line_info.append("{:+6.2f}".format(peak_mag))
        line_info.append("{:>8.4f}".format(peak_ht/1000))
        line_info.append("{:>5.3f}".format(f_param))
        line_info.append("{:8.2e}".format(mass))

        # Convergence angle
        line_info.append("{:5.2f}".format(np.degrees(traj.best_conv_inter.conv_angle)))

        # Median fit error in arcsec
        line_info.append("{:12.2f}".format(3600*np.degrees(np.median([obs.ang_res_std for obs \
            in traj.observations if not obs.ignore_station]))))


        # Check if the meteor begins/ends inside the FOV
        fov_beg, fov_end = checkMeteorFOVBegEnd(traj)

        line_info.append("{:>6s}".format(str(fov_beg)))
        line_info.append("{:>6s}".format(str(fov_end)))


        # Participating stations
        participating_stations = sorted([obs.station_id for obs in traj.observations \
            if obs.ignore_station == False])
        line_info.append("{:>4d}".format(len(participating_stations)))
        line_info.append("{:s}".format(",".join(participating_stations)))


        out_str += delimiter.join(line_info) + "\n\r"


    # Save the file to a trajectory summary
    traj_summary_path = os.path.join(dir_path, traj_summary_file_name)
    with open(traj_summary_path, 'w') as f:
        f.write(out_str)

    print("Trajectory summary saved to:", traj_summary_path)





def plotSCE(x_data, y_data, color_data, plot_title, colorbar_title, output_dir, \
    file_name, density_plot=False, low_density=False, high_density=False, cmap=None, \
    cmap_reverse=False, plot_showers=False, shower_obj_list=None, show_sol_range=True, sol_range=None, \
    dt_range=None, import_matplotlib=False):
    """ Plot the given data in Sun-centered ecliptic coordinates.

    Arguments:
        x_data: [ndarray] SCE longitude data (radians).
        y_data: [ndarray] SCE latitude data (radians).
        color_data: [ndarray] Data for colour coding.
        plot_title: [str] Title of the plot.
        colorbar_title: [str] Colour bar title.
        output_dir: [str] Path to the output directory.
        file_name: [str] Name of the image file that will be saved.

    Kewyword arguments:
        density_plot: [bool] Save the SCE density plot. False by default.
        low_density: [bool] When the number of trajectories is low, change the density scale. False by 
            default. high_density cannot be True at the same time.
        high_density: [bool] When the number of trajectories is high, change the density scale. False by 
            default. low_density cannot be true at the same time.
        cmap: [bool] Use a specific colour map. None by default.
        cmap_reverse: [bool] Reverse the colour map. False by default.
        plot_showers: [bool] Mark showers on the plot. False by default.
        shower_obj_list: [list] A list of MeteorShower objects that will be plotted. Needs to be given in 
            plot_showers is True. 
        show_sol_range: [bool] Show the solar longitude range on the plot. True by default.
        sol_range: [tuple] Range of solar longitudes to plot. None by default.
        dt_range: [tuple] Datetimes of the first and last trajectory in the plot.
        import_matplotlib: [bool] Import matplotlib, as it is not imported globally. False by default.
            This needs to be used if the function is used from another script.

    """

    if import_matplotlib:

        import matplotlib
        import matplotlib.pyplot as plt

    # Otherwise, use global variables
    else:
        global matplotlib
        global plt


    ### PLOT SUN-CENTERED GEOCENTRIC ECLIPTIC RADIANTS ###

    fig = plt.figure(figsize=(16, 8), facecolor='k')


    # Init the allsky plot
    celes_plot = CelestialPlot(x_data, y_data, projection='sinu', lon_0=-90, ax=fig.gca())

    # ### Mark the sources ###
    # sources_lg = np.radians([0, 270, 180])
    # sources_bg = np.radians([0, 0, 0])
    # sources_labels = ["Helion", "Apex", "Antihelion"]
    # celes_plot.scatter(sources_lg, sources_bg, marker='x', s=15, c='0.75')

    # # Convert angular coordinates to image coordinates
    # x_list, y_list = celes_plot.m(np.degrees(sources_lg), np.degrees(sources_bg + np.radians(2.0)))
    # for x, y, lbl in zip(x_list, y_list, sources_labels):
    #     plt.text(x, y, lbl, color='w', ha='center', alpha=0.5)

    # ### ###

    fg_color = 'white'

    # Choose the colormap
    if density_plot:
        cmap_name = 'inferno'
        cmap_bottom_cut = 0.0
    else:
        cmap_name = 'viridis'
        cmap_bottom_cut = 0.3


    # Set a specific colour map
    if cmap is not None:
        cmap_name = cmap

    if cmap_reverse:
        cmap_name += "_r"


    # Cut the dark portion of the colormap
    cmap = plt.get_cmap(cmap_name)
    if cmap_reverse:
        colors = cmap(np.linspace(0.0, 1.0 - cmap_bottom_cut, cmap.N))
    else:
        colors = cmap(np.linspace(cmap_bottom_cut, 1.0, cmap.N))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('cut_cmap', colors)


    ### Do a KDE density plot
    if density_plot:

        # Define extent and density
        lon_min = -180
        lon_max = 180
        lat_min = -90
        lat_max = 90
        delta_deg = 0.5

        lon_bins = np.linspace(lon_min, lon_max, int(360/delta_deg))
        lat_bins = np.linspace(lat_min, lat_max, int(180/delta_deg))


        # Determine colorbar range based on whether it's a high or low density plot
        if low_density:

            vmin, vmax = 0.4, 50.0
            colorbar_ticks = [1, 2, 5, 10, 20, 50]

        # High density plot
        elif high_density:
            vmin, vmax = 1.0, 500
            colorbar_ticks = [1, 5, 10, 20, 50, 100, 200, 500]

        # Normal plot
        else:
            vmin, vmax = 1.0, 100
            colorbar_ticks = [1, 5, 10, 20, 50, 100]


        # Rotate all coordinates by 90 deg to make them Sun-centered
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        lon_corr = (np.degrees(x_data) + 90)%360

        # Do a sinus projection
        lon_corr_temp = np.zeros_like(lon_corr)
        lon_corr_temp[lon_corr > 180] = ((180 - lon_corr[lon_corr > 180] + 180)*np.cos(y_data[lon_corr > 180]))
        lon_corr_temp[lon_corr <= 180] = ((180 - lon_corr[lon_corr <= 180] - 180)*np.cos(y_data[lon_corr <= 180]))
        lon_corr = lon_corr_temp

        # Compute the histogram
        data, _, _ = np.histogram2d(lon_corr, 
            np.degrees(np.array(y_data)), bins=(lon_bins, lat_bins))

        # Apply Gaussian filter to it
        data = scipy.ndimage.filters.gaussian_filter(data, 1.0)*4*np.pi

        plt_handle = celes_plot.m.imshow(data.T, origin='lower', extent=[lon_min, lon_max, lat_min, lat_max],\
            #interpolation='gaussian', norm=matplotlib.colors.PowerNorm(gamma=1./2.), cmap=cmap)
            interpolation='gaussian', norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=cmap)

        # Plot the colorbar
        cb = fig.colorbar(plt_handle, ticks=colorbar_ticks, format="%.0f")


    else:
        
        ### Do a scatter plot

        # Compute the dot size which varies by the number of data points
        dot_size = 40*(1.0/np.sqrt(len(x_data)))
        if dot_size > 1:
            dot_size = 1

        # Compute the dot size differently if a high density plot is done
        if high_density:
            dot_size = 2.0*(1.0/np.sqrt(len(x_data)))
            if dot_size > 1:
                dot_size = 1

        # Plot the data
        plt_handle = celes_plot.scatter(x_data, y_data, color_data, s=dot_size, cmap=cmap)
    
        # Plot the colorbar
        cb = fig.colorbar(plt_handle)




    # Plot showers, if given
    if plot_showers and (shower_obj_list is not None):
        for shower_obj in shower_obj_list:

            # Compute the plotting coordinates
            lam = shower_obj.L_g - shower_obj.la_sun
            bet = shower_obj.B_g


            ### Plot a <PLOT_SHOWER_RADIUS> deg radius circle around the shower centre ###
            
            # Generate circle data points
            heading_arr = np.linspace(0, 2*np.pi, 50)
            bet_arr, lam_arr = sphericalPointFromHeadingAndDistance(bet, lam, heading_arr, \
                np.radians(PLOT_SHOWER_RADIUS))

            # If the circle is on the 90 deg boundary, split the arrays into two parts
            lam_arr_check = (lam_arr - np.radians(270))%(2*np.pi)
            if np.any(lam_arr_check < np.pi) and np.any(lam_arr_check >= np.pi):

                temp_arr = np.c_[lam_arr, bet_arr]
                arr_left = temp_arr[lam_arr_check < np.pi]
                arr_right = temp_arr[lam_arr_check > np.pi]

                lam_left_arr, bet_left_arr = arr_left.T
                lam_right_arr, bet_right_arr = arr_right.T

                lam_arr_segments = [lam_left_arr, lam_right_arr]
                bet_arr_segments = [bet_left_arr, bet_right_arr]

            else:
                lam_arr_segments = [lam_arr]
                bet_arr_segments = [bet_arr]


            # Plot the circle segments
            for lam_arr, bet_arr in zip(lam_arr_segments, bet_arr_segments):
                celes_plot.plot(lam_arr, bet_arr, color='w', alpha=0.5)

            ### ###


            #### Plot the name of the shower ###

            # The name orientation is determined by the quadrant, so all names "radiate" from the 
            #   centre of the plot
            heading = 0
            lam_check = (lam - np.radians(270))%(2*np.pi)
            va = 'top'
            if lam_check < np.pi:
                ha = 'right'
                if bet > 0:
                    heading = -np.pi/4
                    va = 'bottom'
                else:
                    heading = -3*np.pi/4
            else:
                ha = 'left'
                if bet > 0:
                    heading = np.pi/4
                    va = 'bottom'
                else:
                    heading = 3*np.pi/4

            # Get the shower name location
            bet_txt, lam_txt = sphericalPointFromHeadingAndDistance(bet, lam, heading, \
                np.radians(PLOT_SHOWER_RADIUS))

            # Plot the shower name
            celes_plot.text(shower_obj.IAU_code, lam_txt, bet_txt, ha=ha, va=va, color='w', alpha=0.5)


            ### ###

    
    # Tweak the colorbar
    cb.set_label(colorbar_title, color=fg_color)
    cb.ax.yaxis.set_tick_params(color=fg_color)
    cb.outline.set_edgecolor(fg_color)
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=fg_color)


    plt.title(plot_title, color=fg_color)


    if show_sol_range and (sol_range is not None) and (dt_range is not None):

        # Plot solar longitude range and count
        sol_min, sol_max = sol_range
        dt_min, dt_max = dt_range

        # plt.annotate(u"$\lambda_{\u2609 min} =$" + u"{:>5.2f}\u00b0".format(sol_min) \
        #     + u"\n$\lambda_{\u2609 max} =$" + u"{:>5.2f}\u00b0".format(sol_max), \
        #     xy=(0, 1), xycoords='axes fraction', color='w', size=12, family='monospace')
        plt.annotate(
              u"Sol min = {:>6.2f}\u00b0 ({:s})\n".format(sol_min, dt_min.strftime("%Y/%m/%d %H:%M")) \
            + u"Sol max = {:>6.2f}\u00b0 ({:s})\n".format(sol_max, dt_max.strftime("%Y/%m/%d %H:%M"))
            +  "Count = {:d}".format(len(x_data)), \
            xy=(0, 1), xycoords='axes fraction', color='w', size=10, family='monospace')


    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, file_name), dpi=100, facecolor=fig.get_facecolor(), \
        edgecolor='none')

    plt.close()

    ### ###



def generateTrajectoryPlots(output_dir, traj_list, plot_name='scecliptic', plot_vg=True, plot_sol=True, \
    plot_density=True, low_density=False, plot_showers=False, time_limited_plot=False):
    """ Given a path with trajectory .pickle files, generate orbits plots. 
    
    Arguments:
        output_dir: [str] Output directory path.
        traj_list: [str] List of trajectory objects to plots.

    Keyword arguments:
        plot_name: [str] Name of the plot to save.
        plot_vg: [bool] Save the SCE Vg plot. True by default.
        plot_sol: [bool] Save the SCE solar longitude plot. True by default
        plot_density: [bool] Save the SCE density plot. True by default.
        low_density: [bool] When the number of trajectories is low, change the density scale. False by 
            default.
        plot_showers: [bool] Mark showers on the plot. False by default.
        time_limited_plot: [bool] Plotted trajectories are within a limited time. False by default.
            If True, this will disable the sol and time text on the image.
    """



    ### Plot Sun-centered geocentric ecliptic plots ###

    lambda_list = []
    beta_list = []
    vg_list = []
    sol_list = []

    shower_no_list = []
    shower_obj_dict = {}

    hypo_count = 0
    jd_min = np.inf
    jd_max = 0
    for traj in traj_list:

        # Reject all hyperbolic orbits
        if traj.orbit.e > 1:
            hypo_count += 1
            continue

        # Compute Sun-centered longitude
        lambda_list.append(traj.orbit.L_g - traj.orbit.la_sun)

        beta_list.append(traj.orbit.B_g)
        vg_list.append(traj.orbit.v_g/1000)
        sol_list.append(np.degrees(traj.orbit.la_sun))

        # Track first and last observation
        jd_min = min(jd_min, traj.jdt_ref)
        jd_max = max(jd_max, traj.jdt_ref)



        if plot_showers:

            # Perform shower association and track the list of all showers
            shower_obj = associateShowerTraj(traj)

            # If the trajectory was associated, sort it to the appropriate shower
            if shower_obj is not None:
                if shower_obj.IAU_no not in shower_no_list:
                    shower_no_list.append(shower_obj.IAU_no)
                    shower_obj_dict[shower_obj.IAU_no] = [shower_obj]
                else:
                    shower_obj_dict[shower_obj.IAU_no].append(shower_obj)



    # Compute mean shower radiant for all associated showers
    shower_obj_list = []
    if plot_showers and shower_obj_dict:
        for shower_no in shower_obj_dict:

            # Check if there are enough shower members for plotting
            if len(shower_obj_dict[shower_no]) < MIN_SHOWER_MEMBERS:
                continue

            la_sun_mean = meanAngle([sh.la_sun for sh in shower_obj_dict[shower_no]])
            L_g_mean = meanAngle([sh.L_g for sh in shower_obj_dict[shower_no]])
            B_g_mean = np.mean([sh.B_g for sh in shower_obj_dict[shower_no]])
            v_g_mean = np.mean([sh.v_g for sh in shower_obj_dict[shower_no]])

            # Init a new shower object
            shower_obj_mean = MeteorShower(la_sun_mean, L_g_mean, B_g_mean, v_g_mean, shower_no)

            shower_obj_list.append(shower_obj_mean)



    print("Hyperbolic percentage: {:.2f}%".format(100*hypo_count/len(traj_list)))

    # Compute the range of solar longitudes
    sol_min = np.degrees(jd2SolLonSteyaert(jd_min))
    sol_max = np.degrees(jd2SolLonSteyaert(jd_max))

    # Compute the time range as datetime objects
    dt_min = jd2Date(jd_min, dt_obj=True)
    dt_max = jd2Date(jd_max, dt_obj=True)

    # Plot SCE vs Vg
    if plot_vg:
        plotSCE(lambda_list, beta_list, vg_list, 
            "Sun-centered geocentric ecliptic coordinates", "$V_g$ (km/s)", output_dir, plot_name + "_vg.png", \
            shower_obj_list=shower_obj_list, plot_showers=plot_showers, show_sol_range=time_limited_plot, \
            sol_range=(sol_min, sol_max), dt_range=(dt_min, dt_max))


    # Plot SCE vs Sol
    if plot_sol:
        plotSCE(lambda_list, beta_list, sol_list, \
            "Sun-centered geocentric ecliptic coordinates", "Solar longitude (deg)", output_dir, \
            plot_name + "_sol.png", shower_obj_list=shower_obj_list, plot_showers=plot_showers, \
            show_sol_range=time_limited_plot, sol_range=(sol_min, sol_max), \
            dt_range=(dt_min, dt_max))
    

    
    # Plot SCE orbit density
    if plot_density:
        plotSCE(lambda_list, beta_list, None, 
            "Sun-centered geocentric ecliptic coordinates", "Count", output_dir, plot_name + "_density.png", \
            density_plot=True, low_density=low_density, shower_obj_list=shower_obj_list, 
            plot_showers=plot_showers, show_sol_range=time_limited_plot, sol_range=(sol_min, sol_max), \
            dt_range=(dt_min, dt_max))




def generateStationPlot(dir_path, traj_list, color_scheme='light'):
    """ Generate a plot of all stations participating in the trajectory estimation. """


    # Choose the color scheme
    cs = MapColorScheme()
    
    if color_scheme == 'light':
        cs.light()

    else:
        cs.dark()


    plt.figure(figsize=(19.2, 10.8))

    # Init the map
    m = Basemap(projection='cyl', resolution='i')

    # Draw the coast boundary and fill the oceans with the given color
    m.drawmapboundary(fill_color=cs.map_background)

    # Fill continents, set lake color same as ocean color
    m.fillcontinents(color=cs.continents, lake_color=cs.lakes, zorder=1)

    # Draw country borders
    m.drawcountries(color=cs.countries)
    m.drawstates(color=cs.states, linestyle='--')



    ### PLOT WORLD MAP ###

    # Group stations into countries
    country_dict = {}
    for traj in traj_list:

        for obs in traj.observations:

            # Extract country code
            country_code = obs.station_id[:2]

            if country_code not in country_dict:
                country_dict[country_code] = {}
            

            if obs.station_id not in country_dict[country_code]:
                country_dict[country_code][obs.station_id] = [obs.lat, obs.lon]



    # Plot stations in all countries
    for country_code in country_dict:

        station_dict = country_dict[country_code]

        # Extract lat/lon
        lat = np.degrees([station_dict[station_id][0] for station_id in station_dict])
        lon = np.degrees([station_dict[station_id][1] for station_id in station_dict])

        # Convert lat/lon to x/y
        x, y = m(lon, lat)

        plt.scatter(x, y, s=0.75, zorder=5, label="{:s}: {:d}".format(country_code, len(lat)))


    plt.legend(loc='lower left')

    plt.tight_layout()

    plt.savefig(os.path.join(dir_path, "world_map.png"), dpi=100)

    plt.close()

    ### ###





def generateShowerPlots(dir_path, traj_list, min_members=30, max_radiant_err=0.5, P_0m=1210):
    """ Generate shower plots of showers with a minimum of min_members members. 
    
    Arguments:
        dir_path: [str] Path where to save the plots.
        traj_list: [list] A list of Trajectory objects.

    Keyword arguments:
        min_members: [int] Minimum number of shower members to plot the shower. 30 by default
        max_radiant_err: [float] Maximum radiant error in degrees. 0.5 deg by default
        P_0m: [float] Power of a zero magnitue meteors (watts) in the given camera bandpass. 1210 W by 
            defualt.

    """


    # Plot parameters
    label_text_size = 6


    # Create a mask to filter out trajectories not satisfying the filter conditions
    reject_indices = []
    for i, traj in enumerate(traj_list):

        if traj.uncertainties is not None:
            if np.degrees(np.hypot(np.cos(traj.orbit.dec_g)*traj.uncertainties.ra_g, \
                traj.uncertainties.dec_g)) > max_radiant_err:

                reject_indices.append(i)



    # Generate a dictionary of showers and trajectories
    shower_no_list = []
    shower_traj_dict = {}
    for i, traj in enumerate(traj_list):

        # Skip low quality orbits
        if i in reject_indices:
            continue

        # Perform shower association and track the list of all showers
        shower_obj = associateShowerTraj(traj)

        # If the trajectory was associated, sort it to the appropriate shower
        if shower_obj is not None:
            if shower_obj.IAU_no not in shower_no_list:
                shower_no_list.append(shower_obj.IAU_no)
                shower_traj_dict[shower_obj.IAU_no] = [traj]
            else:
                shower_traj_dict[shower_obj.IAU_no].append(traj)



    # Extract all data
    ht_beg_all  = np.array([traj.rbeg_ele for i, traj in enumerate(traj_list) if i not in reject_indices])
    ht_end_all  = np.array([traj.rend_ele for i, traj in enumerate(traj_list) if i not in reject_indices])
    v_g_all     = np.array([traj.orbit.v_g for i, traj in enumerate(traj_list) if i not in reject_indices])
    tj_all      = np.array([traj.orbit.Tj for i, traj in enumerate(traj_list) if i not in reject_indices])
    ht_max_all  = np.array([computePeakMagHt(traj)[1] for i, traj in enumerate(traj_list) \
        if i not in reject_indices])


    # Generate shower plots
    for shower_no in sorted(shower_no_list):

        # Extract shower trajectories
        shower_trajs = shower_traj_dict[shower_no]

        # Only take those showers with the minimum number of members
        if len(shower_trajs) < min_members:
            continue


        # Get shower letter code
        shower_code = associateShowerTraj(shower_trajs[0]).IAU_code

        print("Processing shower:", shower_no, shower_code)


        # Init the plot
        fig, ((ax_rad, ax_radnodrift, ax_rayleigh), (ax_mass, ax_ht, ax_tj)) = plt.subplots(nrows=2, ncols=3,\
            figsize=(10, 6))




        ### Plot the non-drift corrected radiants in Sun-centered ecliptic coordinates ###

        # Compute SCE coordinates
        sol_data = np.array([traj.orbit.la_sun for traj in shower_trajs])
        lam_data = np.array([traj.orbit.L_g for traj in shower_trajs])
        bet_data = np.array([traj.orbit.B_g for traj in shower_trajs])
        lam_sol_data = (lam_data - sol_data)%(2*np.pi)

        # Get the errors
        lam_err = np.array([traj.uncertainties.L_g for traj in shower_trajs])
        bet_err = np.array([traj.uncertainties.B_g for traj in shower_trajs])

        # Compute masses (only take trajectories which are completely inside the FOV, otherwise set the 
        #   mass to None)
        mass_data = np.array([computeMass(traj, P_0m) if all(checkMeteorFOVBegEnd(traj)) else None \
            for traj in shower_trajs])

        # Begin/end/peak heights and initial velocities
        ht_beg_shower = np.array([traj.rbeg_ele for traj in shower_trajs])
        ht_end_shower = np.array([traj.rend_ele for traj in shower_trajs])
        ht_max_shower = np.array([computePeakMagHt(traj)[1] if all(checkMeteorFOVBegEnd(traj)) else None \
            for traj in shower_trajs])
        v_g_shower = np.array([traj.orbit.v_g for traj in shower_trajs])

        # Get Tj
        tj_shower = np.array([traj.orbit.Tj for traj in shower_trajs])



        ### Radiant drift functions ###


        def circleXYLineFunc(x, m, k):
            """ Fit a line through circular data, both X and Y. """
            return (m*x%(2*np.pi) + k)%(2*np.pi)


        def circleXLineFunc(x, m, k):
            """ Fit a line through circular data, only X. """
            return m*x%(2*np.pi) + k


        ###



        ## Estimate radiant drift and reject 3 sigma outliers from the mean ##

        n_iter = 3
        for i in range(n_iter):

            # Compute the radiant drift in SCE longitude
            lam_sol_drift_params, _ = scipy.optimize.curve_fit(circleXYLineFunc, sol_data, lam_sol_data)

            # Compute the radiant drift in SCE latitude
            bet_drift_params, _ = scipy.optimize.curve_fit(circleXLineFunc, sol_data, bet_data)


            # Compute SCE coordinates without the radiant drift
            lam_sol_data_nodrift = lam_sol_data - circleXYLineFunc(sol_data, *lam_sol_drift_params)
            bet_data_nodrift = bet_data - circleXLineFunc(sol_data, *bet_drift_params)


            ###
            

            # Compute distances from the drift-corrected mean radiant
            rad_dists = np.array([angleBetweenSphericalCoords(0.0, 0.0, bet_nodrift, lam_sol_nodrift) \
                for lam_sol_nodrift, bet_nodrift in zip(lam_sol_data_nodrift, bet_data_nodrift)])

            # Fit a Rayleigh distribution to radiant distances and get the standard deviation
            ray_params = scipy.stats.rayleigh.fit(rad_dists)
            ray_std = scipy.stats.rayleigh.std(*ray_params)


            # Skip rejecting radiants in the last iteration
            if i == n_iter - 1:
                break

            # Reject all showers outside 3 standard deviations
            filter_indices      = rad_dists < 3*ray_std
            sol_data            = sol_data[filter_indices]
            lam_data            = lam_data[filter_indices]
            bet_data            = bet_data[filter_indices]
            lam_sol_data        = lam_sol_data[filter_indices]
            lam_sol_data_nodrift = lam_sol_data_nodrift[filter_indices]
            bet_data_nodrift    = bet_data_nodrift[filter_indices]
            lam_err             = lam_err[filter_indices]
            bet_err             = bet_err[filter_indices]
            mass_data           = mass_data[filter_indices]
            ht_beg_shower       = ht_beg_shower[filter_indices]
            ht_end_shower       = ht_end_shower[filter_indices]
            ht_max_shower       = ht_max_shower[filter_indices]
            v_g_shower          = v_g_shower[filter_indices]
            tj_shower           = tj_shower[filter_indices]

            ##


        # Compute mean radiant values
        lam_sol_avg = meanAngle(lam_sol_data)%(2*np.pi)
        bet_avg = np.mean(bet_data)


        # Remove None masses
        mass_data_filt = np.array([mass for mass in mass_data if mass is not None])


        ### Plot the radiants (no drift correction) ###

        cp = CelestialPlot(lam_sol_data, bet_data, projection="stere", ax=ax_rad, bgcolor='w', \
            tick_text_size=label_text_size)
        cp.scatter(lam_sol_data, bet_data, c='k', s=1)


        # Plot the mean radiant
        cp.scatter([lam_sol_avg], [bet_avg], c='red', marker='x', alpha=0.5, s=20, \
            label="Mean $\\lambda_{g} - \\lambda_{\\odot} = $" \
                + "{:.2f}$^\\circ$".format(np.degrees(lam_sol_avg)) \
                + "\nMean $B_g = $" + "{:.2f}$^\\circ$".format(np.degrees(bet_avg)))
        
        ax_rad.legend(prop={'size': label_text_size}, loc='upper left')

        ax_rad.set_xlabel("$\\lambda_g - \\lambda_{\\odot}$", labelpad=15, fontsize=label_text_size)
        ax_rad.set_ylabel("$\\beta_g$", labelpad=15, fontsize=label_text_size)


        ### ###


        ### Plot the radiants (drift corrected) ###


        # Plot the errors (scale the X error by the cos of the latitude)
        ax_radnodrift.errorbar(np.degrees(lam_sol_data_nodrift), np.degrees(bet_data_nodrift), \
            xerr=np.degrees(lam_err)*np.cos(bet_data), yerr=np.degrees(bet_err), color='k', ms=0.5, fmt="o", \
            zorder=3, elinewidth=0.5, \
            label="Drift $\\lambda_{g} - \\lambda_{\\odot} = $" \
                + "{:.3f}".format(lam_sol_drift_params[0]) \
                + "\nDrift $B_g = $" + "{:.3f}".format(bet_drift_params[0]))
            
        ax_radnodrift.legend(prop={'size': label_text_size}, loc='upper left')

        ax_radnodrift.set_xlabel("$(\\lambda_g - \\lambda_{\\odot}) - (\\lambda_g - \\lambda_{\\odot})'$", \
            fontsize=label_text_size)
        ax_radnodrift.set_ylabel("$\\beta_g - \\beta'_g$", fontsize=label_text_size)


        ax_radnodrift.grid(linestyle='dashed')

        ## Make the limits rectangular ##
        x_min, x_max = ax_radnodrift.get_xlim()
        y_min, y_max = ax_radnodrift.get_ylim()
        delta_x = abs(x_max - x_min)
        delta_y = abs(y_max - y_min)

        if delta_x > delta_y:

            delta_largest = delta_x
            
            delta_diff = (delta_x - delta_y)/2
            
            y_min -= delta_diff
            y_max += delta_diff

        else:

            delta_largest = delta_y

            delta_diff = (delta_y - delta_x)/2

            x_min -= delta_diff
            x_max += delta_diff


        # Add a buffer around the points
        border_ratio = 0.1
        x_min -= delta_largest*border_ratio
        x_max += delta_largest*border_ratio
        y_min -= delta_largest*border_ratio
        y_max += delta_largest*border_ratio


        # Set the plot limits
        ax_radnodrift.set_xlim([x_min, x_max])
        ax_radnodrift.set_ylim([y_min, y_max])

        # Set equal aspect ratio
        ax_radnodrift.set_aspect("equal")

        ## ##


        # Set smaller tick size
        ax_radnodrift.xaxis.set_tick_params(labelsize=label_text_size)
        ax_radnodrift.yaxis.set_tick_params(labelsize=label_text_size)


        ### ###



        ### Plot the Rayleigh distribution ###

        # Plot the histogram of radiant distances
        nbins = int(np.sqrt(len(rad_dists))) if len(rad_dists) > 100 else 10
        _, bins, _ = ax_rayleigh.hist(np.degrees(rad_dists), density=True, bins=nbins, histtype='step', \
            color='0.5', label="Data")

        # # Get angular residuals that are above and below the median mass
        # mass_median = np.median(mass_data_filt)
        # rad_dists_small = [rdist for rdist, m in zip(rad_dists, mass_data) \
        #     if ((m is not None) and (m <  mass_median))]
        # rad_dists_large = [rdist for rdist, m in zip(rad_dists, mass_data) \
        #     if (m is not None) and (m >= mass_median)]

        # # Plot angular deviations below and above the median mass, to see the dispersion dependence on size    
        # ax_rayleigh.hist(np.degrees(rad_dists_small), density=True, bins=bins, histtype='step', \
        #     color='r', label="m < median mass")
        # ax_rayleigh.hist(np.degrees(rad_dists_large), density=True, bins=bins, histtype='step', \
        #     color='b', label="m >= median mass")


        # Plot the fitted Rayleigh distribution (normalize the integral to 1)
        x_arr = np.linspace(np.min(rad_dists), np.max(rad_dists), 100)
        ray_pdf_values = scipy.stats.rayleigh.pdf(x_arr, *ray_params)
        ray_integ = scipy.integrate.simps(ray_pdf_values, x=np.degrees(x_arr))
        ax_rayleigh.plot(np.degrees(x_arr), ray_pdf_values/ray_integ, color='k', \
            label="Rayleigh, $\\sigma = $" + "{:.2f}".format(np.degrees(ray_std)) + "$^{\\circ}$")

        # Plot the median deviation
        rad_dists_median = np.median(rad_dists)
        y_min, y_max = ax_rayleigh.get_ylim()
        y_arr = np.linspace(y_min, y_max, 10)
        ax_rayleigh.plot(np.degrees(rad_dists_median) + np.zeros_like(y_arr), y_arr, linestyle='dashed', 
            linewidth=0.5, color='k', zorder=3, \
            label="Median deviation = {:.2}".format(np.degrees(rad_dists_median)) + "$^{\\circ}$")


        ax_rayleigh.legend(prop={'size': label_text_size}, loc='upper right')

        ax_rayleigh.set_ylim(y_min, y_max)

        ax_rayleigh.set_xlabel("Angular offset ($^{\\circ}$)", fontsize=label_text_size)
        ax_rayleigh.set_ylabel("Fraction", fontsize=label_text_size)

        # Set smaller tick size
        ax_rayleigh.xaxis.set_tick_params(labelsize=label_text_size)
        ax_rayleigh.yaxis.set_tick_params(labelsize=label_text_size)


        ### ###



        ### Plot the mass distribution ###

        # Plot the cumulative distribution
        ax_mass.hist(np.log10(mass_data_filt), bins=len(mass_data_filt), cumulative=-1, \
            density=True, log=True, histtype='step', color='k', zorder=4)


        # Fit the slope to the data
        params, x_arr, inflection_point, ref_point, slope, slope_report, intercept, lim_point, sign, \
            kstest = fitSlope(np.log10(mass_data_filt), True)


        # Plot the tangential line with the slope
        ax_mass.plot(sign*x_arr, logline(-x_arr, slope, intercept), color='r', \
            label="s = {:.2f} \nKS test D = {:.3f} \nKS test p-value = {:.3f}".format(\
                slope_report, kstest.statistic, kstest.pvalue), zorder=5)

        ax_mass.legend(prop={'size': label_text_size}, loc='lower left')


        ax_mass.set_xlabel("Log10 mass (kg)", fontsize=label_text_size)
        ax_mass.set_ylabel("Cumulative count", fontsize=label_text_size)

        ax_mass.set_ylim(ymax=1.0)
        ax_mass.set_xlim([-8.0, -3.0])


        # Set smaller tick size
        ax_mass.xaxis.set_tick_params(labelsize=label_text_size)
        ax_mass.yaxis.set_tick_params(labelsize=label_text_size)

        ###



        ### Plot beginning and end heights vs initial speed

        ax_ht.grid(alpha=0.2)

        # Plot begin and end heights of all meteors
        ax_ht.scatter(v_g_all/1000, ht_beg_all/1000, s=0.1, c='k', alpha=0.2, zorder=3)
        ax_ht.scatter(v_g_all/1000, ht_end_all/1000, s=0.1, c='k', alpha=0.2, zorder=3)

        # Plot begin and end heights of shower meteors
        ax_ht.scatter(v_g_shower/1000, ht_beg_shower/1000, s=0.5, c='r', alpha=0.25, label='Begin', \
            zorder=4)
        ax_ht.scatter(v_g_shower/1000, ht_end_shower/1000, s=0.5, c='b', alpha=0.25, label='End', zorder=4)

        ax_ht.legend(prop={'size': label_text_size}, loc='lower right')


        ax_ht.set_xlabel("$v_{init}$ (km/s)", fontsize=label_text_size)
        ax_ht.set_ylabel("Height (km)", fontsize=label_text_size)

        ax_ht.set_xlim([9, 71])

        # Set smaller tick size
        ax_ht.xaxis.set_tick_params(labelsize=label_text_size)
        ax_ht.yaxis.set_tick_params(labelsize=label_text_size)


        ###


        ### Plot Tj vs peak height ###

        ax_tj.grid(alpha=0.2)

        # Plot Tj vs peak height of all meteors
        ax_tj.scatter(tj_all, ht_max_all/1000, s=0.1, c='k', alpha=0.2, zorder=3)


        # Remove entries where the peak height is None (i.e. the meteor was not fully observed)
        ht_max_none_arr = np.array([ht_tmp is not None for ht_tmp in ht_max_shower])
        ht_max_shower_filter = ht_max_shower[ht_max_none_arr]
        tj_shower_filter = tj_shower[ht_max_none_arr]


        # Plot the shower data
        ax_tj.scatter(tj_shower_filter, ht_max_shower_filter/1000, s=0.5, c='r', alpha=0.5, \
            zorder=4)


        ax_tj.set_xlabel("$T_J$", fontsize=label_text_size)
        ax_tj.set_ylabel("Peak height (km)", fontsize=label_text_size)

        ax_tj.set_xlim([-1, 6])


        # Plot the divisions by Tisserand
        y_min, y_max = ax_tj.get_ylim()
        y_arr = np.linspace(y_min, y_max, 10)
        ax_tj.plot(np.zeros_like(y_arr) + 2, y_arr, linestyle='dashed', color='k', zorder=3, linewidth=0.5)
        ax_tj.plot(np.zeros_like(y_arr) + 3, y_arr, linestyle='dashed', color='k', zorder=3, linewidth=0.5)
        ax_tj.set_ylim([y_min, y_max])


        # Set smaller tick size
        ax_tj.xaxis.set_tick_params(labelsize=label_text_size)
        ax_tj.yaxis.set_tick_params(labelsize=label_text_size)
    

        ### ###



        # # Set the shower code as the title
        # fig.suptitle("#{:d} - {:s}".format(shower_no, shower_code))


        fig.tight_layout()


        # Save the plot
        fig.savefig(os.path.join(dir_path, "{:04d}{:s}.png".format(shower_no, shower_code)), dpi=300)

        # Close the figure
        plt.clf()
        plt.close()




def inTimeRange(traj_dt, time_beg, time_end):

    # Test the time range
    if time_beg is not None:
        if time_beg >= traj_dt:
            return False

    if time_end is not None:
        if time_end < traj_dt:
            return False


    return True



def loadTrajectoryPickles(dir_path, traj_quality_params, time_beg=None, time_end=None, verbose=False, \
    filter_duplicates=True):
    """ Load trajectory pickle files with the given quality constraints and in the given time range. 
    
    Arguments:
        dir_path: [str] Path to the directory with trajectory directories.
        traj_quality_params: [TrajQualityParams instance]

    Keyword arguments:
        time_beg: [datetime] First time to load. Note that it is assumed that the trajectory is in a folder 
            named in the following format: YYYYMMDD_hhmmss.us_STATION
        time_end: [datetime] Last time to load.
        verbose: [bool] Print how many trajectoories were loaded. False by default.
        filter_duplicates: [bool] Filter duplicate trajectories (starting at the same time and observed
            by the same stations).


    Return:
        traj_list: [list] A list of Trajectory objects.

    """


    traj_list = []
    loaded_trajs_count = 0
    for entry in os.walk(dir_path):

        traj_dir_path, _, file_names = entry

        # Try loading a time if the time limit was given
        time_read_failed_dir = False
        if (time_beg is not None) or (time_end is not None):

            try:

                # Extract the name of the trajectory folder
                traj_dir_name =  os.path.basename(os.path.normpath(traj_dir_path))

                traj_dir_split = traj_dir_name.split("_")

                if len(traj_dir_split) < 2:
                    time_read_failed_dir = True

                else:

                    traj_date = traj_dir_split[0]
                    traj_time = traj_dir_split[1]

                    traj_dt = datetime.datetime.strptime("{:s}-{:s}".format(traj_date, traj_time), \
                        "%Y%m%d-%H%M%S.%f")

                    # Test the time range
                    if not inTimeRange(traj_dt, time_beg, time_end):
                        continue

            except:
                time_read_failed_dir = True

                if verbose:
                    print("Failed to read the time for dir name: {:s}".format(traj_dir_name))


        # Go through all files
        for file_name in file_names:

            # Check if the file is a pickle file
            if file_name.endswith("_trajectory.pickle") or file_name.endswith("_trajectory_sim.pickle"):

                # If reading the time from directory failed, try reading it from the file name
                time_read_failed_file = False
                if time_read_failed_dir:

                    file_name_split = file_name.split("_")
                    traj_date = file_name_split[0]
                    traj_time = file_name_split[1]

                    try:
                        traj_dt = datetime.datetime.strptime("{:s}-{:s}".format(traj_date, traj_time), \
                            "%Y%m%d-%H%M%S")

                        # Test the time range
                        if not inTimeRange(traj_dt, time_beg, time_end):
                            continue

                    except:
                        time_read_failed_file = True

                        if verbose:
                            print("Failed to read the time for file name: {:s}".format(file_name))


                # Load the pickle file
                try:
                    traj = loadPickle(traj_dir_path, file_name)

                except:
                    print("Error opening trajectory file: {:s}".format(traj_dir_path, file_name))
                    continue

                # If reading the time from the file name has failed, read it from the pickle
                if time_read_failed_file:

                    # Compute the trajectory datetime
                    traj_dt = jd2Date(traj.jdt_ref) 

                    # Test the time range
                    if not inTimeRange(traj_dt, time_beg, time_end):
                        continue


                loaded_trajs_count += 1

                # Print loading progress
                if verbose:
                    if loaded_trajs_count%1000 == 0:
                        print("Loaded {:d} trajectories...".format(loaded_trajs_count))


                ### DELETE UNECESSARY OBJECTS AND ARGUMENTS TO CONSERVE MEMORY ###

                # Delete all stored MC runs to conserve memory
                if traj.uncertainties is not None:
                    if hasattr(traj.uncertainties, "mc_traj_list"):
                        del traj.uncertainties.mc_traj_list
                    

                # Delete plane intersections
                if traj.intersection_list:
                    del traj.intersection_list

                # Delete observations with added noise used during MC runs
                if hasattr(traj, "obs_noisy"):
                    del traj.obs_noisy


                # Delete stuff in observations
                for obs in traj.observations:
                    del obs.meas1
                    del obs.meas2
                    del obs.magnitudes
                    del obs.azim_data
                    del obs.elev_data
                    del obs.ra_data
                    del obs.dec_data
                    del obs.velocities
                    del obs.velocities_prev_point
                    del obs.model_ra
                    del obs.model_dec
                    del obs.model_azim
                    del obs.model_elev
                    del obs.model_fit1
                    del obs.model_fit2
                    del obs.length
                    del obs.lag
                    del obs.state_vect_dist
                    del obs.meas_lat
                    del obs.meas_lon
                    del obs.meas_range
                    del obs.model_lat
                    del obs.model_lon
                    del obs.model_range
                    del obs.meas_eci
                    del obs.meas_eci_los
                    del obs.model_eci


                ### ###

                
                # Skip those with no orbit solution
                if traj.orbit.ra_g is None:
                    continue


                ### MINIMUM POINTS
                ### Reject all trajectories with small number of used points ###
                points_count = [len(obs.time_data[obs.ignore_list == 0]) for obs in traj.observations \
                    if obs.ignore_station == False]

                if not points_count:
                    continue

                max_points = max(points_count)

                if max_points < traj_quality_params.min_traj_points:
                    # print("Skipping {:.2f} due to the small number of points...".format(traj.jdt_ref))
                    continue

                ###


                ### CONVERGENCE ANGLE                
                ### Reject all trajectories with a too small convergence angle ###

                if np.degrees(traj.best_conv_inter.conv_angle) < traj_quality_params.min_qc:
                    # print("Skipping {:.2f} due to the small convergence angle...".format(traj.jdt_ref))
                    continue

                ###


                ### MAXIMUM ECCENTRICITY ###

                if traj.orbit.e > traj_quality_params.max_e:
                    continue

                ###


                ### MAXIMUM RADIANT ERROR ###

                if traj.uncertainties is not None:
                    if np.degrees(np.hypot(np.cos(traj.orbit.dec_g)*traj.uncertainties.ra_g, \
                        traj.uncertainties.dec_g)) > traj_quality_params.max_radiant_err:

                        continue



                ### MAXIMUM GEOCENTRIC VELOCITY ERROR ###

                if traj.uncertainties is not None:
                    if traj.uncertainties.v_g > traj.orbit.v_g*traj_quality_params.max_vg_err/100:
                        continue

                ###


                ### HEIGHT FILTER ###

                if traj.rbeg_ele/1000 > traj_quality_params.max_begin_ht:
                    continue

                if traj.rend_ele/1000 < traj_quality_params.min_end_ht:
                    continue

                ###


                # Add the trajectory identified if it's missing
                traj = addTrajectoryID(traj)


                # Add the trajectory to the output list
                traj_list.append(traj)


    # Sort trajectories by time
    traj_list = sorted(traj_list, key=lambda x: x.jdt_ref)

    # Remove duplicate trajectories
    if filter_duplicates:
        
        filtered_traj_list = []
        skipped_indices = []

        for i, traj1 in enumerate(traj_list):

            # Skip already checked duplicates
            if i in skipped_indices:
                continue

            # Set the first trajectory as the candidate
            candidate_traj = traj1

            for traj2 in traj_list[i + 1:]:

                # Check if the trajectories have the same time
                if traj1.jdt_ref == traj2.jdt_ref:

                    # Check if they have the same stations
                    if set([obs.station_id for obs in traj1.observations]) \
                        == set([obs.station_id for obs in traj2.observations]):

                        # If the duplicate has a smaller radiant error, take it instead of the first
                        #   trajectory
                        if hasattr(traj1, 'uncertainties') and hasattr(traj2, 'uncertainties'):
                                
                            if traj1.uncertainties is not None:

                                if traj2.uncertainties is not None:

                                    # Compute the radiant errors
                                    traj1_rad_error = np.hypot(traj1.uncertainties.ra_g, \
                                        traj1.uncertainties.dec_g)
                                    traj2_rad_error = np.hypot(traj2.uncertainties.ra_g, \
                                        traj2.uncertainties.dec_g)

                                    # Take the second candidate if the radiant error is smaller
                                    if traj2_rad_error < traj1_rad_error:
                                        candidate_traj = traj2


                            # If the first candidate doesn't have estimated errors, but the second one does,
                            #   use that one
                            else:
                                if traj2.uncertainties is not None:
                                    candidate_traj = traj2

                        
                        # Add duplicate to the already checked list
                        skipped_indices.append(traj_list.index(traj2))




            filtered_traj_list.append(candidate_traj)


        traj_list = filtered_traj_list



    return traj_list



def generateAutoPlotsAndReports(dir_path, traj_quality_params, prev_sols=10, sol_window=1):
    """ Auto generate plots per degree of solar longitude and put them in the AUTO_OUTPUT_DATA_DIR that is 
        in the parent directory of dir_path. 
    
    Arguments:
        dir_path: [str] Path to folders with trajectories.
        traj_quality_params: [TrajQualityParams instance]

    Keyword arguments:
        prev_sols: [int] Number of previous degrees of solar longitdes to go back for and re-generate plots and reports.
        sol_window: [float] Window of solar longitudes for plots and graphs in degrees.

    """


    # Make path to the output directory
    parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
    output_dir = os.path.join(parent_dir, AUTO_OUTPUT_DATA_DIR)
    mkdirP(output_dir)

    # Make the plots directory
    plots_dir         = os.path.join(output_dir, AUTO_OUTPUT_PLOT_DIR)
    plots_daily_dir   = os.path.join(plots_dir,  AUTO_OUTPUT_DAILY_DIR)
    plots_monthly_dir = os.path.join(plots_dir,  AUTO_OUTPUT_MONTHLY_DIR)
    mkdirP(plots_dir)
    mkdirP(plots_daily_dir)
    mkdirP(plots_monthly_dir)

    # Make the trajectory summary directory
    summary_dir         = os.path.join(output_dir,  AUTO_OUTPUT_SUMMARY_DIR)
    summary_daily_dir   = os.path.join(summary_dir, AUTO_OUTPUT_DAILY_DIR)
    summary_monthly_dir = os.path.join(summary_dir, AUTO_OUTPUT_MONTHLY_DIR)
    mkdirP(summary_dir)
    mkdirP(summary_daily_dir)
    mkdirP(summary_monthly_dir)



    ### Generate daily plots for the last N days ###

    # Compute the time of the next closest integer solar longitude in degrees
    time_now = datetime.datetime.utcnow()
    sol_lon_now = np.degrees(jd2SolLonSteyaert(datetime2JD(time_now)))
    sol_lon_next = np.ceil(sol_lon_now)
    

    # Generate plots for every day
    most_recent_plot_file = None
    yesterday_plot_file = None
    most_recent_summary_file = None
    yesterday_summary_file = None
    for sol_decrement in range(prev_sols):

        sol_lon_end = (sol_lon_next - sol_decrement)%360
        sol_lon_beg = (sol_lon_end - sol_window)%360

        # Compute beg/end dates from solar longitudes
        time_end_est = time_now - datetime.timedelta(days=sol_decrement*365.25/360.0)
        time_end = jd2Date(solLon2jdSteyaert(time_end_est.year, time_end_est.month, \
            np.radians(sol_lon_end)), dt_obj=True)
        time_beg_est = time_end_est - datetime.timedelta(days=sol_window)
        time_beg = jd2Date(solLon2jdSteyaert(time_beg_est.year, time_beg_est.month, \
            np.radians(sol_lon_beg)), dt_obj=True)


        # Load all trajectories within the given time range
        traj_list = loadTrajectoryPickles(dir_path, traj_quality_params, time_beg=time_beg, \
            time_end=time_end, verbose=True)


        # Skip if no trajectories are loaded
        if not traj_list:
            print("No trajectories in sol range: {:.1f} - {:.1f}".format(sol_lon_beg, sol_lon_end))
            continue


        # Plot graphs per solar longitude
        plot_name = "scecliptic_{:4d}{:02d}{:02d}_solrange_{:05.1f}-{:05.1f}".format(time_beg.year, \
            time_beg.month, time_beg.day, sol_lon_beg, sol_lon_end)
        print("Plotting sol range {:05.1f}-{:05.1f}...".format(sol_lon_beg, sol_lon_end))
        generateTrajectoryPlots(plots_daily_dir, traj_list, plot_name=plot_name, plot_sol=False, \
            plot_showers=True, time_limited_plot=True, low_density=True)

        # Store the most recent plot
        if most_recent_plot_file is None:
            most_recent_plot_file = plot_name

        else:
            # Store the yesterday's plot
            if yesterday_plot_file is None:
                yesterday_plot_file = plot_name


        # Write the trajectory summary
        summary_name = "traj_summary_{:4d}{:02d}{:02d}_solrange_{:05.1f}-{:05.1f}.txt".format(time_beg.year, \
            time_beg.month, time_beg.day, sol_lon_beg, sol_lon_end)
        writeOrbitSummaryFile(summary_daily_dir, traj_list, summary_name)

        # Store the most recent summary file name
        if most_recent_summary_file is None:
            most_recent_summary_file = summary_name

        else:
            # Store yesterday's data
            if yesterday_summary_file is None:
                yesterday_summary_file = summary_name


    ### ###


    ### Generate monthly plots ###

    month_dt_beg_est = time_now - datetime.timedelta(days=prev_sols*365.25/360.0)
    month_dt_beg = jd2Date(solLon2jdSteyaert(month_dt_beg_est.year, month_dt_beg_est.month, \
            np.radians(sol_lon_end)), dt_obj=True)
    month_dt_end = time_now

    # Generate pairs of datetimes with edges as the beginning/end of each month which fully encompasses
    #   the given range of times
    monthly_bins =  generateMonthyTimeBins(month_dt_beg, month_dt_end)

    # Plot data and create trajectory summaries for monthly bins
    latest_monthly_plot_file = None
    last_month_plot_file = None
    for dt_beg, dt_end in monthly_bins:

        # Load all trajectories within the given time range
        traj_list = loadTrajectoryPickles(dir_path, traj_quality_params, time_beg=dt_beg, \
            time_end=dt_end, verbose=True)


        # Skip if no trajectories are loaded
        if not traj_list:
            print("No trajectories in month: {:s}".format(dt_beg.strftime("%B %Y")))
            continue


        # Plot graphs per month
        plot_name = "scecliptic_monthly_{:4d}{:02d}".format(dt_beg.year, dt_beg.month)
        print("Plotting month: {:s}".format(dt_beg.strftime("%B %Y")))
        generateTrajectoryPlots(plots_monthly_dir, traj_list, plot_name=plot_name, plot_sol=False, \
            plot_showers=True, time_limited_plot=True, low_density=False)


        # Store this month's plot
        if latest_monthly_plot_file is None:
            latest_monthly_plot_file = plot_name

        else:
            # Store the last month's plot
            if last_month_plot_file is None:
                last_month_plot_file = plot_name


        # Write the trajectory summary
        summary_name = "traj_summary_monthly_{:4d}{:02d}.txt".format(dt_beg.year, dt_beg.month)
        writeOrbitSummaryFile(summary_monthly_dir, traj_list, summary_name)


    ### 



    ### Create one summary file with all data

    # Open the full file for writing (first write to a temp file, then move it to a completed file)
    traj_summary_all_temp = TRAJ_SUMMARY_ALL + ".tmp"
    with open(os.path.join(summary_dir, traj_summary_all_temp), 'w') as f_all:

        f_all.write("# Summary generated on {:s} UTC\n\r".format(str(datetime.datetime.utcnow())))

        # Find all monthly trajectory reports
        header_written = False
        skipped_summary_date = False
        for summary_name in sorted(os.listdir(summary_monthly_dir)):
            if summary_name.startswith("traj_summary_monthly_"):

                # Load the monthly summary file and write entries to the collective file
                with open(os.path.join(summary_monthly_dir, summary_name)) as f_monthly:
                    for line in f_monthly:

                        # Skip the header if it was already written
                        if header_written:
                            if line.startswith("#"):
                                continue

                        else:
                            # If the header was not written, copy it, but skip the summary date at the top
                            if not skipped_summary_date:
                                skipped_summary_date = True
                                continue

                        f_all.write(line)

                # Write the header only once
                header_written = True

    # Change the name of the temp file to the summary file
    if os.path.isfile(os.path.join(summary_dir, traj_summary_all_temp)):
        shutil.copy2(os.path.join(summary_dir, traj_summary_all_temp), 
            os.path.join(summary_dir, TRAJ_SUMMARY_ALL))

        # Remove the temp file
        os.remove(os.path.join(summary_dir, traj_summary_all_temp))

        print("Saved summary file with all orbits: {:s}".format(TRAJ_SUMMARY_ALL))

    else:
        print("The temp file {:s} was not created!".format(traj_summary_all_temp))


    ###



    ### Link to latest files ###

    plot_copy_list = [
        [plots_daily_dir,   most_recent_plot_file,    "scecliptic_latest_daily"], \
        [plots_daily_dir,   yesterday_plot_file,      "scecliptic_yesterday"], \
        [plots_monthly_dir, latest_monthly_plot_file, "scecliptic_latest_monthly"], \
        [plots_monthly_dir, last_month_plot_file,     "scecliptic_last_month"]]

    # Link latest plots per day and month
    for plots_dir, plot_copy_name, plot_name in plot_copy_list:

        # Set the most recent daily plots
        if plot_copy_name is not None:

            print("Copying latest plots...")

            # Set latest plots
            suffix_list = ["_vg.png", "_density.png"]
            for suffix in suffix_list:
                shutil.copy2(os.path.join(plots_dir, plot_copy_name + suffix), \
                    os.path.join(plots_dir, plot_name + suffix))

    
    # Link to latest summary files
    summary_copy_list = [
        [summary_daily_dir, most_recent_summary_file, "traj_summary_latest_daily.txt"], \
        [summary_daily_dir, yesterday_summary_file,   "traj_summary_yesterday.txt"]
        ]

    for summary_dir, summary_copy_name, summary_name in summary_copy_list:
        
        # Set the most recent daily summary file
        if summary_copy_name is not None:

            print("Copying latest report...")

            # Set latest summary file
            shutil.copy2(os.path.join(summary_dir, summary_copy_name), \
                os.path.join(summary_dir, summary_name))


    ### ###






if __name__ == "__main__":

    from wmpl.Utils.OSTools import importBasemap

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    Basemap = importBasemap()

    import argparse

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="""Given a folder with trajectory .pickle files, generate an orbit summary CSV file and orbital graphs.""",
        formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('dir_path', type=str, help='Path to the data directory. Trajectory pickle files are found in all subdirectories.')

    arg_parser.add_argument('-s', '--solstep', metavar='SOL_STEP', \
        help='Step in solar longitude for plotting (degrees). 5 deg by default.', type=float, default=5.0)

    arg_parser.add_argument('-a', '--auto', metavar='PREV_SOLS', type=int, default=None, const=10, \
        nargs='?', \
        help="""Run continously taking the data in the last PREV_SOLS degrees of solar longitudes to generate new plots and reports, and update old ones."""
        )

    arg_parser.add_argument('-f', '--autofirst', metavar='FIRST_PREV_SOLS', type=int, \
        help="""During the first continous run, go back FIRST_PREV_SOLS to generate the plots and report. After that, use PREV_SOLS to run."""
        )

    arg_parser.add_argument('-p', '--plot_all_showers', action="store_true", \
        help="""Plot showers on the maps showing the whole date ramge."""
        )
    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    ############################



    ### FILTERS ###

    class TrajQualityParams(object):
        def __init__(self):

            # Minimum number of points on the trajectory for the station with the most points
            self. min_traj_points = 6

            # Minimum convergence angle (deg)
            self. min_qc = 5.0

            # Maximum eccentricity
            self. max_e = 1.5

            # Maximum radiant error (deg)
            self. max_radiant_err = 2.0

            # Maximum geocentric velocity error (percent)
            self. max_vg_err = 10.0

            # Begin/end height filters (km)
            self. max_begin_ht = 160
            self. min_end_ht = 20


    traj_quality_params = TrajQualityParams()

    ### ###


    # If auto trajectories should run, run in an infinite loop
    if cml_args.auto is not None:

        print("Auto generating plots and reports every {:.1f} hours using the last {:d} deg solar longitudes of data...".format(AUTO_RUN_FREQUENCY, cml_args.auto))

        if cml_args.autofirst is not None:
            print("The first run will use {:d} deg solar longitudes of data!".format(cml_args.autofirst))

        first_run = True
        while True:

            # Clock for measuring script time
            t1 = datetime.datetime.utcnow()


            # Set the number of solar longitudes to use for the auto run
            prev_sols = cml_args.auto

            # If the solar longitude range is different for the first run
            if cml_args.autofirst is not None:
                if first_run:
                    prev_sols = cml_args.autofirst


            # Generate the latest plots
            generateAutoPlotsAndReports(cml_args.dir_path, traj_quality_params, prev_sols=prev_sols)



            # Wait to run AUTO_RUN_FREQUENCY hours after the beginning
            wait_time = (datetime.timedelta(hours=AUTO_RUN_FREQUENCY) \
                - (datetime.datetime.utcnow() - t1)).total_seconds()

            # Compute next run time
            next_run_time = datetime.datetime.utcnow() + datetime.timedelta(seconds=wait_time)

            # Wait to run
            while next_run_time > datetime.datetime.utcnow():
                print("Waiting {:s} to generate plots and summary again...          ".format(str(next_run_time \
                    - datetime.datetime.utcnow())), end='\r')
                time.sleep(2)


            first_run = False


    else:

        # Run once

        # Get a list of paths of all trajectory pickle files
        traj_list = loadTrajectoryPickles(cml_args.dir_path, traj_quality_params, verbose=True)



        # Generate shower plots
        print("Plotting showers...")
        generateShowerPlots(cml_args.dir_path, traj_list, min_members=30, P_0m=P_0M, max_radiant_err=0.5)

        # Generate the orbit summary file
        print("Writing summary file...")
        writeOrbitSummaryFile(cml_args.dir_path, traj_list, P_0m=P_0M)

        # Generate summary plots
        print("Plotting all trajectories...")
        pas = False
        if cml_args.plot_all_showers is not None:
            pas = True
            print('adding shower loci to all maps')
        generateTrajectoryPlots(cml_args.dir_path, traj_list, plot_showers=pas, time_limited_plot=False)

        # Generate station plot
        print("Plotting station plot...")
        generateStationPlot(cml_args.dir_path, traj_list)




        # Generate radiant plots per solar longitude (degrees)
        step = cml_args.solstep
        for sol_min in np.arange(0, 360 + step, step):
            sol_max = sol_min + step

            # Extract only those trajectories with solar longitudes in the given range
            traj_list_sol = [traj_temp for traj_temp in traj_list if \
                (np.degrees(traj_temp.orbit.la_sun) >= sol_min) \
                and (np.degrees(traj_temp.orbit.la_sun) < sol_max)]


            # Skip solar longitudes with no data
            if len(traj_list_sol) == 0:
                continue

            print("Plotting solar longitude range: {:.1f} - {:.1f}".format(sol_min, sol_max))


            # Plot graphs per solar longitude
            generateTrajectoryPlots(cml_args.dir_path, traj_list_sol, \
                plot_name="scecliptic_solrange_{:05.1f}-{:05.1f}".format(sol_min, sol_max), plot_sol=False, \
                plot_showers=True, time_limited_plot=True)
