""" I/O functions for the UWO ev_* file format. """

from __future__ import print_function, division, absolute_import


import os
import shutil

import numpy as np
import scipy.interpolate

from wmpl.Formats.EventUWO import StationData
from wmpl.Formats.GenericFunctions import addSolverOptions, solveTrajectoryGeneric, MeteorObservation, \
    prepareObservations
from wmpl.Trajectory.Trajectory import Trajectory
from wmpl.Trajectory.GuralTrajectory import GuralTrajectory
from wmpl.Utils.OSTools import mkdirP
from wmpl.Utils.TrajConversions import jd2Date, jd2UnixTime, unixTime2JD, altAz2RADec


def writeEvFile(dir_path, file_name, jdt_ref, station_id, lat, lon, ele, time_data, theta_data, phi_data, 
    mag_data=None):
    """ Write a UWO style ev_* file.

    Arguments:
        dir_path: [str] Path to the directory where the file will be saved.
        file_name: [str] Name of the file.
        jdt_ref: [float] Julian date for which the time in time_data is 0.
        station_id: [str] Name of the station
        lat: [float] Latitude +N of the station (radians).
        lon: [float] Longitude +E of the station, (radians).
        ele: [float] Height above sea level (meters).
        time_data: [list of floats] A list of times of observations in seconds, where t = 0s is at jdt_ref.
        theta_data: [list of floats]: A list of zenith angles of observations (radians)
        phi_data: [list of floats] A list of azimuth (+N of due E) of observations (radians).

    """

    # Convert Julian date to date string
    year, month, day, hour, minute, second, millisecond = jd2Date(jdt_ref)
    date_str = "{:4d}{:02d}{:02d} {:02d}:{:02d}:{:02d}.{:03d}".format(year, month, day, hour, minute, second, \
        int(millisecond))

    # Convert JD to unix time
    unix_time = jd2UnixTime(jdt_ref)


    # Check if the magnitude data was given
    if mag_data is None:
        mag_data = np.zeros_like(time_data)


    with open(os.path.join(dir_path, file_name), 'w') as f:

        f.write("#\n")
        f.write("#   version : WMPL\n")
        f.write("#    num_fr : {:d}\n".format(len(time_data)))
        f.write("#    num_tr : 1\n")
        f.write("#      time : {:s} UTC\n".format(date_str))
        f.write("#      unix : {:f}\n".format(unix_time))
        f.write("#       ntp : LOCK 83613 1068718 130\n")
        f.write("#       seq : 0\n")
        f.write("#       mul : 0 [A]\n")
        f.write("#      site : {:s}\n".format(station_id))
        f.write("#    latlon : {:9.6f} {:+10.6f} {:.1f}\n".format(np.degrees(lat), np.degrees(lon), ele))
        f.write("#      text : WMPL generated\n")
        f.write("#    stream : KT\n")
        f.write("#     plate : none\n")
        f.write("#      geom : 0 0\n")
        f.write("#    filter : 0\n")
        f.write("#\n")
        #f.write("#  fr    time    sum     seq       cx       cy     th        phi      lsp     mag  flag   bak    max\n")
        f.write("#  fr    time    sum     seq       cx       cy     th        phi      lsp     mag  flag\n")


        for fr, (t, theta, phi, mag) in enumerate(zip(time_data, theta_data, phi_data, mag_data)):

            sum_ = 0
            seq = 0
            cx = 0.0
            cy = 0.0
            lsp = 0.0
            flag = "0000"
            bak = 0.0
            max_ = 0.0

            f.write("{:5d} ".format(fr))
            f.write("{:7.3f} ".format(t))
            
            f.write("{:6d} ".format(sum_))
            f.write("{:7d} ".format(seq))
            
            f.write("{:8.3f} ".format(cx))
            f.write("{:8.3f} ".format(cy))
            
            f.write("{:9.5f} ".format(np.degrees(theta)))
            f.write("{:10.5f} ".format(np.degrees(phi)))

            f.write("{:7.3f} ".format(lsp))
            f.write("{:6.2f} ".format(mag))

            f.write("{:5s} ".format(flag))

            #f.write("{:6.2} ".format(bak))
            #f.write("{:6.2} ".format(max_))

            f.write("\n")




def readEvFile(dir_path, file_name):
    """ Given the path of the UWO-style event file, read it into the StationData object. 

    Arguments:
        dir_path: [str] Path to the directory with the ev file.
        file_name: [str] Name of the ev file.

    Return:
        [StationData instance]
    """


    with open(os.path.join(dir_path, file_name)) as f:


        jdt_ref = None
        lat = None
        lon = None
        elev = None
        site = None
        stream = None


        time_data = []
        x_data = []
        y_data = []
        theta_data = []
        phi_data = []
        mag_data = []

        for line in f:

            if not line:
                continue

            # Read metadata
            if line.startswith("#"):

                entry = line[1:].split()

                if not entry:
                    continue

                # Read reference time
                if entry[0] == "unix":
                    ts, tu = list(map(int, entry[2].split(".")))
                    jdt_ref = unixTime2JD(ts, tu)

                elif entry[0] == "site":
                    site = entry[2]

                elif entry[0] == "latlon":
                    lat, lon, elev = list(map(float, entry[2:5]))

                elif entry[0] == "stream":
                    stream = entry[2]


            # Read data
            else:

                line = line.split()

                time_data.append(float(line[1]))
                x_data.append(float(line[4]))
                y_data.append(float(line[5]))
                theta_data.append(float(line[6]))
                phi_data.append(float(line[7]))

                # Check if the magnitude is NaN and set None instead
                mag = line[9]
                if 'nan' in mag:
                    mag = None
                else:
                    mag = float(mag)

                mag_data.append(mag)


        # If there is a NaN in the magnitude data, interpolate it
        if None in mag_data:

            # Get a list of clean data
            mag_data_clean = [entry for entry in enumerate(mag_data) if entry[1] is not None]
            clean_indices, clean_mags = np.array(mag_data_clean).T

            # If there aren't at least 2 good points, return None
            if len(clean_indices) < 2:
                return None

            # Interpolate in linear units
            intens_interpol = scipy.interpolate.PchipInterpolator(clean_indices, 10**(clean_mags/(-2.5)))


            # Interpolate missing magnitudes
            for i, mag in enumerate(mag_data):

                # Don't interpolate at the edges if there are NaNs
                if (i < np.min(clean_indices)) or (i > np.max(clean_indices)):
                    mag_data[i] = np.nan
                    continue

                if mag is None:
                    mag_data[i] = -2.5*np.log10(intens_interpol(i))

            # none_index = mag_data.index(None)

            # # If the first magnitude is NaN, take the magnitude of the second point
            # if none_index == 0:
            #     mag_data[none_index] = mag_data[none_index + 1]

            # # If the last magnitude is NaN, use the magnitude of the previous point
            # elif none_index == (len(mag_data) - 1):
            #     mag_data[none_index] = mag_data[none_index - 1]

            # # If the magnitude is in between, interpolate it
            # else:

            #     mag_prev = float(mag_data[none_index - 1])
            #     mag_next = float(mag_data[none_index + 1])

            #     # Interpolate in linear units
            #     intens_prev = 10**(mag_prev/(-2.5))
            #     intens_next = 10**(mag_next/(-2.5))
            #     intens_interpol = (intens_prev + intens_next)/2
            #     mag_interpol = -2.5*np.log10(intens_interpol)

            #     mag_data[none_index] = mag_interpol


        

        # Change the relative time to 0 and update the reference Julian date
        time_data = np.array(time_data)
        jdt_ref += time_data[0]/86400
        time_data -= time_data[0]

        # Init the StationData object
        sd = StationData(jdt_ref, np.radians(lat), np.radians(lon), elev, site + stream)
        sd.time_data = np.array(time_data)
        sd.x_data = np.array(x_data)
        sd.y_data = np.array(y_data)
        sd.theta_data = np.radians(theta_data)
        sd.phi_data = np.radians(phi_data)
        sd.mag_data = np.array(mag_data)


        return sd



def readEvFileIntoMeteorObject(ev_file_path):
    """ Read the UWO-style ev file into a MeteorObservation object.

    Arguments:
        ev_file_path: [str] Path to the ev file.

    Return:
        [MeteorObservation instance]
    """

    # Store the ev file contants into a StationData object
    sd = readEvFile(*os.path.split(ev_file_path))

    # Skip bad ev files
    if sd is None:
        print("Skipping {:s}, bad ev file!".format(ev_file_path))
        return None

    # Estimate the FPS from the time data
    fps = round(1.0/np.median(np.diff(sd.time_data)), 2)

    # Init the meteor object
    meteor = MeteorObservation(sd.jd_ref, sd.station_id, sd.lat, sd.lon, sd.height, fps)

    # Add data to meteor object
    for t_rel, x_centroid, y_centroid, theta, phi, mag in zip(sd.time_data, sd.x_data, sd.y_data,\
        sd.theta_data, sd.phi_data, sd.mag_data):

        # Convert theta, phi to azim, alt
        azim = np.pi/2 - phi
        alt = np.pi/2 - theta

        # Compute the JD of the data point
        jd = sd.jd_ref + t_rel/86400.0

        # Convert azim, alt to RA, Dec
        ra, dec = altAz2RADec(azim, alt, jd, sd.lat, sd.lon)
        
        meteor.addPoint(
            t_rel*fps, 
            x_centroid, y_centroid, 
            np.degrees(azim), np.degrees(alt), 
            np.degrees(ra), np.degrees(dec), 
            mag
            )

    meteor.finish()

    return meteor


def solveTrajectoryEv(ev_file_list, solver='original', **kwargs):
    """ Runs the trajectory solver on UWO style ev file. 

    Arguments:
        ev_file_list: [list] A list of paths to ev files.


    Keyword arguments:
        solver: [str] Trajectory solver to use:
            - 'original' (default) - "in-house" trajectory solver implemented in Python
            - 'gural' - Pete Gural's PSO solver

    Return:
        traj: [Trajectory instance] Solved trajectory
    """


    # Check that there are at least two stations present
    if len(ev_file_list) < 2:
        print('ERROR! The list of ev files does not contain multistation data!')

        return False


    # Load the ev file
    meteor_list = []
    for ev_file_path in ev_file_list:

        # Read the ev file into a MeteorObservation object        
        meteor = readEvFileIntoMeteorObject(ev_file_path)

        if meteor is None:
            continue

        # Check that the observation has a minimum number of points
        if len(meteor.time_data) < 4:
            print("The station {:s} has too few points (<4), skipping: {:s}".format(sd.station_id, ev_file_path))
            continue


        meteor_list.append(meteor)


    # Normalize all observations to the same JD and precess from J2000 to the epoch of date
    jdt_ref, meteor_list = prepareObservations(meteor_list)


    # Check that there are at least two good stations present
    if len(meteor_list) < 2:
        print('ERROR! The list of ev files does not contain at least 2 good ev files!')

        return False


    # Get the base path of these ev files
    root_path = os.path.dirname(ev_file_list[0])

    # Create a new output directory
    dir_path = os.path.join(root_path, jd2Date(jdt_ref, dt_obj=True).strftime("traj_%Y%m%d_%H%M%S.%f"))
    mkdirP(dir_path)

    traj = solveTrajectoryGeneric(jdt_ref, meteor_list, dir_path, solver=solver, **kwargs)


    # Copy the ev files into the output directory
    for ev_file_path in ev_file_list:
        shutil.copy2(ev_file_path, os.path.join(dir_path, os.path.basename(ev_file_path)))



    return traj




if __name__ == '__main__':

    import sys
    import glob
    import argparse


    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Run the trajectory solver on the list of UWO ev files.")

    arg_parser.add_argument('ev_files', metavar='EV_FILES', type=str, nargs='+',\
        help='Full path to ev_*.txt files.')

    # Add other solver options
    arg_parser = addSolverOptions(arg_parser)

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    # Unpack wildcards
    if not isinstance(cml_args.ev_files, list):
        ev_files = glob.glob(cml_args.ev_files)
    else:
        ev_files = cml_args.ev_files
        
        if len(ev_files) == 1:
            ev_files = glob.glob(ev_files[0])

    event_path = os.path.abspath(ev_files[0])

    # Check if the file path exists
    if not os.path.isfile(event_path):
        print('No such file:', event_path)
        sys.exit()

    ev_files = [fn for fn in ev_files if os.path.isfile(fn)]

    print("Using file:")
    for fn in ev_files:
        print(fn)



    max_toffset = None
    if cml_args.maxtoffset:
        max_toffset = cml_args.maxtoffset[0]

    velpart = None
    if cml_args.velpart:
        velpart = cml_args.velpart

    vinitht = None
    if cml_args.vinitht:
        vinitht = cml_args.vinitht[0]



    # Run trajectory solver on the loaded .met file
    solveTrajectoryEv(ev_files, solver=cml_args.solver, max_toffset=max_toffset, \
            monte_carlo=(not cml_args.disablemc), mc_runs=cml_args.mcruns, \
            geometric_uncert=cml_args.uncertgeom, gravity_correction=(not cml_args.disablegravity), 
            gravity_factor=cml_args.gfact,
            plot_all_spatial_residuals=cml_args.plotallspatial, plot_file_type=cml_args.imgformat, \
            show_plots=(not cml_args.hideplots), v_init_part=velpart, v_init_ht=vinitht, \
            show_jacchia=cml_args.jacchia, \
            estimate_timing_vel=(False if cml_args.notimefit is None else cml_args.notimefit), \
            fixed_times=cml_args.fixedtimes, mc_noise_std=cml_args.mcstd)