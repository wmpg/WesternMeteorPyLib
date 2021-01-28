""" Loading CAMS file products, FTPDetectInfo and CameraSites files, running the trajectory solver on loaded
    data. 

"""

from __future__ import print_function, division, absolute_import

import os
import sys
import argparse
import numpy as np

from wmpl.Formats.GenericFunctions import addSolverOptions, MeteorObservation, \
    prepareObservations, solveTrajectoryGeneric
from wmpl.Formats.Milig import StationData, writeMiligInputFile
from wmpl.Utils.TrajConversions import date2JD
from wmpl.Utils.Math import mergeClosePoints, angleBetweenSphericalCoords
from wmpl.Utils.Physics import calcMass



def loadCameraTimeOffsets(cameratimeoffsets_file_name):
    """ Loads time offsets in seconds from the CameraTimeOffsets.txt file. 
    

    Arguments:
        camerasites_file_name: [str] Path to the CameraTimeOffsets.txt file.

    Return:
        time_offsets: [dict] (key, value) pairs of (station_id, time_offset) for every station.
    """


    time_offsets = {}

    # If the file was not found, skip it
    if not os.path.isfile(cameratimeoffsets_file_name):
        print('The time offsets file could not be found! ', cameratimeoffsets_file_name)
        return time_offsets

    with open(cameratimeoffsets_file_name) as f:

        # Skip the header
        for i in range(2):
            next(f)

        # Load camera time offsets
        for line in f:

            line = line.replace('\n', '')
            line = line.split()

            station_id = line[0].strip()

            # Try converting station ID to integer
            try:
                station_id = int(station_id)
            except:
                pass

            t_offset = float(line[1])

            time_offsets[station_id] = t_offset


    return time_offsets




def loadCameraSites(camerasites_file_name):
    """ Loads locations of cameras from a CAMS-style CameraSites files. 

    Arguments:
        camerasites_file_name: [str] Path to the CameraSites.txt file.

    Return:
        stations: [dict] A dictionary where the keys are stations IDs, and values are lists of:
            - latitude +N in radians
            - longitude +E in radians
            - height in meters
    """


    stations = {}

    with open(camerasites_file_name) as f:

        # Skip the fist two lines (header)
        for i in range(2):
            next(f)

        # Read in station info
        for line in f:

            # Skip commended out lines
            if line[0] == '#':
                continue

            line = line.replace('\n', '')

            if line:

                line = line.split()

                station_id, lat, lon, height = line[:4]

                station_id = station_id.strip()

                # Try converting station ID to integer
                try:
                    station_id = int(station_id)
                except:
                    pass

                lat, lon, height = map(float, [lat, lon, height])

                stations[station_id] = [np.radians(lat), np.radians(-lon), height*1000]


    return stations




def loadFTPDetectInfo(ftpdetectinfo_file_name, stations, time_offsets=None,
        join_broken_meteors=True):
    """

    Arguments:
        ftpdetectinfo_file_name: [str] Path to the FTPdetectinfo file.
        stations: [dict] A dictionary where the keys are stations IDs, and values are lists of:
            - latitude +N in radians
            - longitude +E in radians
            - height in meters
        

    Keyword arguments:
        time_offsets: [dict] (key, value) pairs of (stations_id, time_offset) for every station. None by 
            default.
        join_broken_meteors: [bool] Join meteors broken across 2 FF files.


    Return:
        meteor_list: [list] A list of MeteorObservation objects filled with data from the FTPdetectinfo file.

    """

    meteor_list = []

    with open(ftpdetectinfo_file_name) as f:

        # Skip the header
        for i in range(11):
            next(f)


        current_meteor = None

        bin_name = False
        cal_name = False
        meteor_header = False

        for line in f:

            line = line.replace('\n', '').replace('\r', '')

            # Skip the line if it is empty
            if not line:
                continue


            if '-----' in line:

                # Mark that the next line is the bin name
                bin_name = True

                # If the separator is read in, save the current meteor
                if current_meteor is not None:
                    current_meteor.finish()
                    meteor_list.append(current_meteor)

                continue


            if bin_name:

                bin_name = False

                # Mark that the next line is the calibration file name
                cal_name = True

                # Save the name of the FF file
                ff_name = line

                # Extract the reference time from the FF bin file name
                line = line.split('_')

                # Count the number of string segments, and determine if it the old or new CAMS format
                if len(line) == 6:
                    sc = 1
                else:
                    sc = 0

                ff_date = line[1 + sc]
                ff_time = line[2 + sc]
                milliseconds = line[3 + sc]

                year = ff_date[:4]
                month = ff_date[4:6]
                day = ff_date[6:8]

                hour = ff_time[:2]
                minute = ff_time[2:4]
                seconds = ff_time[4:6]

                year, month, day, hour, minute, seconds, milliseconds = map(int, [year, month, day, hour, 
                    minute, seconds, milliseconds])

                # Calculate the reference JD time
                jdt_ref = date2JD(year, month, day, hour, minute, seconds, milliseconds)

                continue


            if cal_name:

                cal_name = False

                # Mark that the next line is the meteor header
                meteor_header = True

                continue


            if meteor_header:

                meteor_header = False

                line = line.split()

                # Get the station ID and the FPS from the meteor header
                station_id = line[0].strip()
                fps = float(line[3])

                # Try converting station ID to integer
                try:
                    station_id = int(station_id)
                except:
                    pass

                # If the time offsets were given, apply the correction to the JD
                if time_offsets is not None:

                    if station_id in time_offsets:
                        print('Applying time offset for station {:s} of {:.2f} s'.format(str(station_id), \
                            time_offsets[station_id]))

                        jdt_ref += time_offsets[station_id]/86400.0

                    else:
                        print('Time offset for given station not found!')


                # Get the station data
                if station_id in stations:
                    lat, lon, height = stations[station_id]


                else:
                    print('ERROR! No info for station ', station_id, ' found in CameraSites.txt file!')
                    print('Exiting...')
                    break


                # Init a new meteor observation
                current_meteor = MeteorObservation(jdt_ref, station_id, lat, lon, height, fps, \
                    ff_name=ff_name)

                continue


            # Read in the meteor observation point
            if (current_meteor is not None) and (not bin_name) and (not cal_name) and (not meteor_header):

                line = line.replace('\n', '').split()

                # Read in the meteor frame, RA and Dec
                frame_n = float(line[0])
                x = float(line[1])
                y = float(line[2])
                ra = float(line[3])
                dec = float(line[4])
                azim = float(line[5])
                elev = float(line[6])

                # Read the visual magnitude, if present
                if len(line) > 8:
                    
                    mag = line[8]

                    if mag == 'inf':
                        mag = None

                    else:
                        mag = float(mag)

                else:
                    mag = None


                # Add the measurement point to the current meteor 
                current_meteor.addPoint(frame_n, x, y, azim, elev, ra, dec, mag)


        # Add the last meteor the the meteor list
        if current_meteor is not None:
            current_meteor.finish()
            meteor_list.append(current_meteor)


    ### Concatenate observations across different FF files ###
    if join_broken_meteors:

        # Go through all meteors and compare the next observation
        merged_indices = []
        for i in range(len(meteor_list)):

            # If the next observation was merged, skip it
            if (i + 1) in merged_indices:
                continue


            # Get the current meteor observation
            met1 = meteor_list[i]


            if i >= (len(meteor_list) - 1):
                break


            # Get the next meteor observation
            met2 = meteor_list[i + 1]
            
            # Compare only same station observations
            if met1.station_id != met2.station_id:
                continue


            # Extract frame number
            met1_frame_no = int(met1.ff_name.split("_")[-1].split('.')[0])
            met2_frame_no = int(met2.ff_name.split("_")[-1].split('.')[0])

            # Skip if the next FF is not exactly 256 frames later
            if met2_frame_no != (met1_frame_no + 256):
                continue


            # Check for frame continouty
            if (met1.frames[-1] < 254) or (met2.frames[0] > 2):
                continue


            ### Check if the next frame is close to the predicted position ###

            # Compute angular distance between the last 2 points on the first FF
            ang_dist = angleBetweenSphericalCoords(met1.dec_data[-2], met1.ra_data[-2], met1.dec_data[-1], \
                met1.ra_data[-1])

            # Compute frame difference between the last frame on the 1st FF and the first frame on the 2nd FF
            df = met2.frames[0] + (256 - met1.frames[-1])

            # Skip the pair if the angular distance between the last and first frames is 2x larger than the 
            #   frame difference times the expected separation
            ang_dist_between = angleBetweenSphericalCoords(met1.dec_data[-1], met1.ra_data[-1], \
                met2.dec_data[0], met2.ra_data[0])

            if ang_dist_between > 2*df*ang_dist:
                continue

            ### ###


            ### If all checks have passed, merge observations ###

            # Recompute the frames
            frames = 256.0 + met2.frames

            # Recompute the time data
            time_data = frames/met1.fps

            # Add the observations to first meteor object
            met1.frames = np.append(met1.frames, frames)
            met1.time_data = np.append(met1.time_data, time_data)
            met1.x_data = np.append(met1.x_data, met2.x_data)
            met1.y_data = np.append(met1.y_data, met2.y_data)
            met1.azim_data = np.append(met1.azim_data, met2.azim_data)
            met1.elev_data = np.append(met1.elev_data, met2.elev_data)
            met1.ra_data = np.append(met1.ra_data, met2.ra_data)
            met1.dec_data = np.append(met1.dec_data, met2.dec_data)
            met1.mag_data = np.append(met1.mag_data, met2.mag_data)

            # Sort all observations by time
            met1.finish()

            # Indicate that the next observation is to be skipped
            merged_indices.append(i + 1)

            ### ###


        # Removed merged meteors from the list
        meteor_list = [element for i, element in enumerate(meteor_list) if i not in merged_indices]




    return meteor_list



def cams2MiligInput(meteor_list, file_path):
    """ Writes CAMS data to MILIG input file. 
    
    Arguments:
        meteor_list: [list] A list of MeteorObservation objects.
        file_path: [str] Path to the MILIG input file which will be written.

    Return:
        None
    """

    # Normalize the observations to the same reference Julian date and precess them from J2000 to the 
    # epoch of date
    jdt_ref, meteor_list = prepareObservations(meteor_list)

    ### Convert CAMS MeteorObservation to MILIG StationData object
    milig_list = []
    for i, meteor in enumerate(meteor_list):

        # Check if the station ID is an integer
        try:
            station_id = int(meteor.station_id)

        except ValueError:
            station_id = i + 1

        # Init a new MILIG meteor data container
        milig_meteor = StationData(station_id, meteor.longitude, meteor.latitude, meteor.height)

        # Fill in meteor points
        for azim, elev, t in zip(meteor.azim_data, meteor.elev_data, meteor.time_data):

            # Convert +E of due N azimuth to +W of due S
            milig_meteor.azim_data.append((azim + np.pi)%(2*np.pi))

            # Convert elevation angle to zenith angle
            milig_meteor.zangle_data.append(np.pi/2 - elev)
            milig_meteor.time_data.append(t)


        milig_list.append(milig_meteor)

    ######


    # Write MILIG input file
    writeMiligInputFile(jdt_ref, milig_list, file_path)



def computeAbsoluteMagnitudes(traj, meteor_list):
    """ Given the trajectory, compute the absolute mangitude (visual mangitude @100km). """

    # Go though every observation of the meteor
    for i, meteor_obs in enumerate(meteor_list):

        # Go through all magnitudes and compute absolute mangitudes
        for dist, mag in zip(traj.observations[i].model_range, meteor_obs.mag_data):

            # Skip nonexistent magnitudes
            if mag is not None:
                
                # Compute the range-corrected magnitude
                abs_mag = mag + 5*np.log10((10**5)/dist)

            else:
                abs_mag = None


            meteor_obs.abs_mag_data.append(abs_mag)

        



if __name__ == "__main__":


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Run the trajectory solver on the given FTPdetectinfo file. It is assumed that only one meteor per file is given.")

    arg_parser.add_argument('ftpdetectinfo_path', nargs=1, metavar='FTP_PATH', type=str, \
        help='Path to the FTPdetectinfo file. It is assumed that the CameraSites.txt and CameraTimeOffsets.txt are in the same folder.')

    # Add other solver options
    arg_parser = addSolverOptions(arg_parser, skip_velpart=True)

    arg_parser.add_argument('-p', '--velpart', metavar='VELOCITY_PART', \
        help='Fixed part from the beginning of the meteor on which the initial velocity estimation using the sliding fit will start. Default is 0.4 (40 percent), but for noisier data this might be bumped up to 0.5.', \
        type=float, default=0.4)

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    ### Parse command line arguments ###

    ftpdetectinfo_path = os.path.abspath(cml_args.ftpdetectinfo_path[0])

    dir_path = os.path.dirname(ftpdetectinfo_path)

    # Check if the given directory is OK
    if not os.path.isfile(ftpdetectinfo_path):
        print('No such file:', ftpdetectinfo_path)
        sys.exit()



    max_toffset = None
    if cml_args.maxtoffset:
        max_toffset = cml_args.maxtoffset[0]

    velpart = None
    if cml_args.velpart:
        velpart = cml_args.velpart

    vinitht = None
    if cml_args.vinitht:
        vinitht = cml_args.vinitht[0]

    ### ###


    # Image file type of the plots
    plot_file_type = 'png'

    camerasites_file_name = 'CameraSites.txt'
    cameratimeoffsets_file_name = 'CameraTimeOffsets.txt'


    camerasites_file_name = os.path.join(dir_path, camerasites_file_name)
    cameratimeoffsets_file_name = os.path.join(dir_path, cameratimeoffsets_file_name)

    # Get locations of stations
    stations = loadCameraSites(camerasites_file_name)

    # Get time offsets of cameras
    time_offsets = loadCameraTimeOffsets(cameratimeoffsets_file_name)

    # Get the meteor data
    meteor_list = loadFTPDetectInfo(ftpdetectinfo_path, stations, time_offsets=time_offsets)


    # # Construct lists of observations of the same meteor (Rpi)
    # meteor1 = meteor_list[:2]
    # meteor2 = meteor_list[2:4]
    # meteor3 = meteor_list[4:6]
    # meteor4 = meteor_list[6:8]
    # meteor5 = meteor_list[8:10]
    # meteor6 = meteor_list[10:12]

    # # # Construct lists of observations of the same meteor
    # meteor1 = meteor_list[13:18]
    # meteor2 = meteor_list[2:5]
    # meteor3 = meteor_list[9:13]
    # meteor4 = meteor_list[18:24]
    # meteor5 = meteor_list[:2]
    # meteor6 = meteor_list[5:9]


    # # Pete's meteors
    # meteor1 = meteor_list[:4]
    # meteor2 = meteor_list[4:7]
    # meteor3 = meteor_list[7:10]

    # # RMS leonids
    # meteor1 = meteor_list[:2]
    # meteor2 = meteor_list[2:4]

    
    # Assume all entires in the FTPdetectinfo path should be used for one meteor
    meteor_proc_list = [meteor_list]


    for meteor in meteor_proc_list:

        for met in meteor:
            print('--------------------------')
            print(met)


        # Normalize the observations to the same reference Julian date and precess them from J2000 to the 
        # epoch of date
        jdt_ref, meteor_list = prepareObservations(meteor)

        # Solve the trajectory
        traj = solveTrajectoryGeneric(jdt_ref, meteor_list, dir_path, solver=cml_args.solver, \
            max_toffset=max_toffset, monte_carlo=(not cml_args.disablemc), mc_runs=cml_args.mcruns, \
            geometric_uncert=cml_args.uncertgeom, gravity_correction=(not cml_args.disablegravity), 
            plot_all_spatial_residuals=cml_args.plotallspatial, plot_file_type=cml_args.imgformat, \
            show_plots=(not cml_args.hideplots), v_init_part=velpart, v_init_ht=vinitht, \
            show_jacchia=cml_args.jacchia, estimate_timing_vel=(not cml_args.notimefit))


    # ### PERFORM PHOTOMETRY

    # import matplotlib
    # import matplotlib.pyplot as plt

    # # Override default DPI for saving from the interactive window
    # matplotlib.rcParams['savefig.dpi'] = 300

    # # Compute absolute mangitudes
    # computeAbsoluteMagnitudes(traj, meteor)

    # # List of photometric uncertainties per station
    # photometry_stddevs = [0.3]*len(meteor)
    # # photometry_stddevs = [0.2, 0.3]*len(meteor)

    # time_data_all = []
    # abs_mag_data_all = []

    # # Plot absolute magnitudes for every station
    # for i, (meteor_obs, photometry_stddev) in enumerate(zip(meteor, photometry_stddevs)):

    #     # Take only magnitudes that are not None
    #     good_mag_indices = [j for j, abs_mag in enumerate(meteor_obs.abs_mag_data) if abs_mag is not None]
    #     time_data = traj.observations[i].time_data[good_mag_indices]
    #     abs_mag_data = np.array(meteor_obs.abs_mag_data)[good_mag_indices]

    #     time_data_all += time_data.tolist()
    #     abs_mag_data_all += abs_mag_data.tolist()

    #     # Sort by time
    #     temp_arr = np.c_[time_data, abs_mag_data]
    #     temp_arr = temp_arr[np.argsort(temp_arr[:, 0])]
    #     time_data, abs_mag_data = temp_arr.T

    #     # Plot the magnitude
    #     plt_hande = plt.plot(time_data, abs_mag_data, label=meteor_obs.station_id, zorder=3)

    #     # Plot magnitude errorbars
    #     plt.errorbar(time_data, abs_mag_data, yerr=photometry_stddev, fmt='--o', color=plt_hande[0].get_color())

    
    # plt.legend()

    # plt.xlabel('Time (s)')
    # plt.ylabel('Absolute magnitude')
    
    # plt.gca().invert_yaxis()

    # plt.grid()

    # plt.show()


    # ### Compute the mass

    # # Sort by time
    # temp_arr = np.c_[time_data_all, abs_mag_data_all]
    # temp_arr = temp_arr[np.argsort(temp_arr[:, 0])]
    # time_data_all, abs_mag_data_all = temp_arr.T

    # # Average the close points
    # time_data_all, abs_mag_data_all = mergeClosePoints(time_data_all, abs_mag_data_all, 1/(2*25))

    # time_data_all = np.array(time_data_all)
    # abs_mag_data_all = np.array(abs_mag_data_all)

    # # Compute the mass
    # mass = calcMass(time_data_all, abs_mag_data_all, traj.v_avg, P_0m=1210)

    # print(mass)

    # ######



    ############################


    # Write the MILIG input file
    #cams2MiligInput(meteor6, 'milig_meteor6.txt')

