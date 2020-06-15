""" I/O functions for the UWO ev_* file format. """

from __future__ import print_function, division, absolute_import


import os
import shutil

import numpy as np

from wmpl.Formats.EventUWO import StationData
from wmpl.Formats.GenericArgumentParser import addSolverOptions
from wmpl.Trajectory.Trajectory import Trajectory
from wmpl.Trajectory.GuralTrajectory import GuralTrajectory
from wmpl.Utils.OSTools import mkdirP
from wmpl.Utils.TrajConversions import jd2Date, jd2UnixTime, unixTime2JD


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
            none_index = mag_data.index(None)

            # If the first magnitude is NaN, take the magnitude of the second point
            if none_index == 0:
                mag_data[none_index] = mag_data[none_index + 1]

            # If the last magnitude is NaN, use the magnitude of the previous point
            elif none_index == (len(mag_data) - 1):
                mag_data[none_index] = mag_data[none_index - 1]

            # If the magnitude is in between, interpolate it
            else:

                mag_prev = float(mag_data[none_index - 1])
                mag_next = float(mag_data[none_index + 1])

                # Interpolate in linear units
                intens_prev = 10**(mag_prev/(-2.5))
                intens_next = 10**(mag_next/(-2.5))
                intens_interpol = (intens_prev + intens_next)/2
                mag_interpol = -2.5*np.log10(intens_interpol)

                mag_data[none_index] = mag_interpol


        

        # Change the relative time to 0 and update the reference Julian date
        time_data = np.array(time_data)
        jdt_ref += time_data[0]/86400
        time_data -= time_data[0]

        # Init the StationData object
        sd = StationData(jdt_ref, np.radians(lat), np.radians(lon), elev, site + stream)
        sd.time_data = np.array(time_data)
        #sd.time_data -= sd.time_data[0] # Normalize to 0
        sd.theta_data = np.radians(theta_data)
        sd.phi_data = np.radians(phi_data)
        sd.mag_data = np.array(mag_data)


        return sd




def solveTrajectoryEv(ev_file_list, solver='original', velmodel=3, **kwargs):
        """ Runs the trajectory solver on UWO style ev file. 
    
        Arguments:
            ev_file_list: [list] A list of paths to ev files.


        Keyword arguments:
            solver: [str] Trajectory solver to use:
                - 'original' (default) - "in-house" trajectory solver implemented in Python
                - 'gural' - Pete Gural's PSO solver
            velmodel: [int] Velocity propagation model for the Gural solver
                0 = constant   v(t) = vinf
                1 = linear     v(t) = vinf - |acc1| * t
                2 = quadratic  v(t) = vinf - |acc1| * t + acc2 * t^2
                3 = exponent   v(t) = vinf - |acc1| * |acc2| * exp( |acc2| * t ) (default)


        Return:
            traj: [Trajectory instance] Solved trajectory
        """


        # Check that there are at least two stations present
        if len(ev_file_list) < 2:
            print('ERROR! The list of ev files does not contain multistation data!')

            return False


        # Load the ev file
        station_data_list = []
        for ev_file_path in ev_file_list:
            
            # Store the ev file contants into a StationData object
            sd = readEvFile(*os.path.split(ev_file_path))

            station_data_list.append(sd)


        # Normalize all times to earliest reference Julian date
        jdt_ref = min([sd_temp.jd_ref for sd_temp in station_data_list])
        for sd in station_data_list:
            for i in range(len(sd.time_data)):
                sd.time_data[i] += (sd.jd_ref - jdt_ref)*86400
            
            sd.jd_ref = jdt_ref


        for sd in station_data_list:
            print(sd)


        # Get the base path of these ev files
        root_path = os.path.dirname(ev_file_list[0])

        # Create a new output directory
        dir_path = os.path.join(root_path, jd2Date(jdt_ref, dt_obj=True).strftime("traj_%Y%m%d_%H%M%S.%f"))
        mkdirP(dir_path)


        if solver == 'original':

            # Init the new trajectory solver object
            traj = Trajectory(jdt_ref, output_dir=dir_path, meastype=4, **kwargs)

        elif solver.startswith('gural'):

            # Extract velocity model is given
            try:
                velmodel = int(solver[-1])

            except: 
                # Default to the exponential model
                velmodel = 3

            # Select extra keyword arguments that are present only for the gural solver
            gural_keys = ['max_toffset', 'nummonte', 'meastype', 'verbose', 'show_plots']
            gural_kwargs = {key: kwargs[key] for key in gural_keys if key in kwargs}

            # Init the new Gural trajectory solver object
            traj = GuralTrajectory(len(station_data_list), jdt_ref, velmodel, verbose=1, \
                output_dir=dir_path, meastype=4, **gural_kwargs)


        # Infill trajectories from each site
        for sd in station_data_list:

            # MC solver
            if solver == 'original':

                traj.infillTrajectory(sd.phi_data, sd.theta_data, sd.time_data, sd.lat, sd.lon, sd.height, \
                    station_id=sd.station_id, magnitudes=sd.mag_data)
            
            # Gural solver
            else:
                traj.infillTrajectory(sd.phi_data, sd.theta_data, sd.time_data, sd.lat, sd.lon, sd.height)


        print('Filling done!')


        # Solve the trajectory
        traj.run()


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

    arg_parser.add_argument('ev_files', metavar='EV_FILES', type=str, \
        help='Full path to ev_*.txt files.')

    # Add other solver options
    arg_parser = addSolverOptions(arg_parser)

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    # Unpack wildcards
    ev_files = glob.glob(cml_args.ev_files)
    

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
            plot_all_spatial_residuals=cml_args.plotallspatial, plot_file_type=cml_args.imgformat, \
            show_plots=(not cml_args.hideplots), v_init_part=velpart, v_init_ht=vinitht)