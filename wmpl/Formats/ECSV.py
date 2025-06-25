""" Solve the trajectory from observations in ECSV files. """

from __future__ import print_function, division, absolute_import

import os
import sys
import glob
import datetime

import numpy as np

from wmpl.Formats.GenericFunctions import addSolverOptions, solveTrajectoryGeneric, MeteorObservation, \
    prepareObservations, writeMiligInputFileMeteorObservation
from wmpl.Utils.TrajConversions import J2000_JD, datetime2JD, altAz2RADec_vect, \
    equatorialCoordPrecession_vect, jd2Date



# Use a fixed FPS (not really important in the grand scheme of things, it's just for combatibility with
#   used functions)
FPS = 15


def loadECSVs(ecsv_paths, no_prepare=False):
    """ Load meteor observations from ECSV files. 
    
    Arguments:
        ecsv_paths: [list] List of paths to ECSV files.

    Keyword arguments:
        no_prepare: [bool] If True, only load the observations, do not prepare them for the solver.
    
    """

    # Init meteor objects
    meteor_list = []
    for ecsv_file in ecsv_paths:

        
        station_lat = None
        station_lon = None
        station_ele = None
        station_id = None
        image_file = ''

        # Load the station information from the ECSV file
        with open(ecsv_file) as f:

            for line in f:
                if line.startswith("#"):
                    line = line.replace('\n', '').replace('\r', '').replace('{', '').replace('}', '')
                    line = line.split(':')
                    if len(line) > 1:

                        if "obs_latitude" in line[0]:
                            station_lat = float(line[1])

                        if "obs_longitude" in line[0]:
                            station_lon = float(line[1])

                        if "obs_elevation" in line[0]:
                            station_ele = float(line[1])

                        if 'image_file' in line[0]:
                            image_file = line[1].strip().strip("'")

                        if "camera_id" in line[0]:
                            station_id = line[1].strip().strip("'")

                        if (station_id is None) and ("dfn_camera_codename" in line[0]):
                            station_id = line[1].strip().strip("'")
                            

            if (station_lat is None) or (station_lon is None) or (station_ele is None) \
                or (station_id is None):

                print("Station info could not be read from file:", ecsv_file, ", skipping...")
                continue


            # Load meteor measurements
            delimiter = ','
            data = np.loadtxt(ecsv_file, comments='#', delimiter=delimiter, dtype=str)

            # Determine the column indices from the header
            header = data[0].tolist()
            dt_indx = header.index('datetime')
            azim_indx = header.index('azimuth')
            alt_indx = header.index('altitude')
            x_indx = header.index('x_image')
            y_indx = header.index('y_image')

            if 'mag_data' in header:
                mag_indx = header.index('mag_data')
            else:
                mag_indx = None


            # Skip the header
            data = data[1:]

            # Unpack data
            dt_data, azim_data, alt_data, x_data, y_data = data[:, dt_indx], data[:, azim_indx], \
                data[:, alt_indx], data[:, x_indx], data[:, y_indx]

            azim_data = azim_data.astype(np.float64)
            alt_data = alt_data.astype(np.float64)
            x_data = x_data.astype(np.float64)
            y_data = y_data.astype(np.float64)

            # Get magnitude data, if any
            if mag_indx is not None:
                mag_data = data[:, mag_indx].astype(np.float64)
            else:
                mag_data = np.zeros_like(azim_data) + 10.0


            # Convert time to JD
            jd_data = []
            for date in dt_data:
                dt = datetime.datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%f")
                jd_data.append(datetime2JD(dt))


            jd_data = np.array(jd_data)

            # Take the first time as reference time
            jdt_ref = jd_data[0]


            # Compute relative time
            time_data = (jd_data - jdt_ref)*86400


            # Compute RA/Dec
            ra_data, dec_data = altAz2RADec_vect(np.radians(azim_data), np.radians(alt_data), jd_data, \
                np.radians(station_lat), np.radians(station_lon))

            # Precess to J2000
            ra_data, dec_data = equatorialCoordPrecession_vect(jdt_ref, J2000_JD.days, ra_data, dec_data)

            comment = '{"ff_name":' + f'"{image_file}"' +'}'


            # Init the meteor object
            meteor = MeteorObservation(jdt_ref, station_id, np.radians(station_lat), \
                np.radians(station_lon), station_ele, FPS, ff_name=comment)

            # Add data to meteor object
            for t_rel, x_centroid, y_centroid, ra, dec, azim, alt, mag in zip(time_data, x_data, y_data, \
                np.degrees(ra_data), np.degrees(dec_data), azim_data, alt_data, mag_data):

                meteor.addPoint(t_rel*FPS, x_centroid, y_centroid, azim, alt, ra, dec, mag)

            meteor.finish()


            # Check that the observation has a minimum number of points
            if len(meteor.time_data) < 4:
                print("The station {:s} has too few points (<4), skipping: {:s}".format(station_id, ecsv_file))
                continue


            meteor_list.append(meteor)


    if no_prepare:
        return jdt_ref, meteor_list

    else:
        
        # Normalize all observations to the same JD and precess from J2000 to the epoch of date
        return prepareObservations(meteor_list)



def saveECSV(dir_path, meteor_observations, 
             network_name='RMS', x_res=None, y_res=None, photom_band=None, img_name=None, calib_stars=None, 
             fov_mid_azim=None, fov_mid_elev=None, fov_mid_rot_horiz=None, fov_horiz=None, fov_vert=None):
    """ Save meteor observations to ECSV files. 
    
    Arguments:
        dir_path: [str] Directory path where the ECSV files will be saved.
        meteor_observations: [list] List of MeteorObservation objects to save.

    Keyword arguments:
        network_name: [str] Name of the network, used in the ECSV file name and header. "RMS" by default.
        x_res: [int] Horizontal resolution of the camera in pixels.
        y_res: [int] Vertical resolution of the camera in pixels.
        photom_band: [str] Photometric band of the star catalogue, e.g. "B", "V", etc.
        img_name: [str] Name of the original image or video file, used in the ECSV header.
        calib_stars: [int] Number of stars used in the astrometric calibration, used in the ECSV header.
        fov_mid_azim: [float] Azimuth of the centre of the field of view in decimal degrees. North = 0, 
            increasing to the East.
        fov_mid_elev: [float] Elevation of the centre of the field of view in decimal degrees. Horizon = 0,
            Zenith = 90.
        fov_mid_rot_horiz: [float] Rotation of the field of view from horizontal, decimal degrees. Clockwise 
            is positive.
        fov_horiz: [float] Horizontal extent of the field of view in decimal degrees.
        fov_vert: [float] Vertical extent of the field of view in decimal degrees.
    
    """

    for meteor in meteor_observations:
        
        # Get the reference datetime
        dt_ref = jd2Date(meteor.jdt_ref, dt_obj=True)

        isodate_format_file = "%Y-%m-%dT%H_%M_%S"
        isodate_format_entry = "%Y-%m-%dT%H:%M:%S.%f"

        # Construct the file name
        # E.g. 2025-06-24T07_55_23_RMS_CA003D.ecsv
        ecsv_name = f"{dt_ref.strftime(isodate_format_file)}_{network_name}_{meteor.station_id}.ecsv"
        ecsv_path = os.path.join(dir_path, ecsv_name)

        
        # Prepare the metadata/header
        meta_dict = {
            'obs_latitude': np.degrees(meteor.latitude),   # Decimal signed latitude (-90 S to +90 N)
            'obs_longitude': np.degrees(meteor.longitude), # Decimal signed longitude (-180 W to +180 E)
            'obs_elevation': meteor.height,                # Altitude in metres above MSL. Note not WGS84
            'origin': 'SkyFit2',                           # The software which produced the data file
            'camera_id': meteor.station_id,                # The code name of the camera, likely to be network-specific
            'cx' : x_res,                                  # Horizontal camera resolution in pixels
            'cy' : y_res,                                  # Vertical camera resolution in pixels
            'photometric_band' : photom_band,              # The photometric band of the star catalogue
            'image_file' : img_name,                       # The name of the original image or video
            'isodate_start_obs': str(dt_ref.strftime(isodate_format_entry)), # The date and time of the start of the video or exposure
            'astrometry_number_stars' : calib_stars,       # The number of stars identified and used in the astrometric calibration
            'mag_label': 'mag_data',                       # The label of the Magnitude column in the Point Observation data
            'no_frags': 1,                                 # The number of meteoroid fragments described in this data
            'obs_az': fov_mid_azim,                        # The azimuth of the centre of the field of view in decimal degrees. North = 0, increasing to the East
            'obs_ev': fov_mid_elev,                        # The elevation of the centre of the field of view in decimal degrees. Horizon =0, Zenith = 90
            'obs_rot': fov_mid_rot_horiz,                  # Rotation of the field of view from horizontal, decimal degrees. Clockwise is positive
            'fov_horiz': fov_horiz,                        # Horizontal extent of the field of view, decimal degrees
            'fov_vert': fov_vert,                          # Vertical extent of the field of view, decimal degrees
           }

                # Write the header
        out_str = """# %ECSV 0.9
# ---
# datatype:
# - {name: datetime, datatype: string}
# - {name: ra, unit: deg, datatype: float64}
# - {name: dec, unit: deg, datatype: float64}
# - {name: azimuth, datatype: float64}
# - {name: altitude, datatype: float64}
# - {name: x_image, unit: pix, datatype: float64}
# - {name: y_image, unit: pix, datatype: float64}
# - {name: integrated_pixel_value, datatype: int64}
# - {name: background_pixel_value, datatype: int64}
# - {name: saturated_pixels, datatype: bool}
# - {name: mag_data, datatype: float64}
# - {name: err_minus_mag, datatype: float64}
# - {name: err_plus_mag, datatype: float64}
# - {name: snr, datatype: float64}
# delimiter: ','
# meta: !!omap
"""
        # Add the meta information
        for key in meta_dict:

            value = meta_dict[key]

            if isinstance(value, str):
                value_str = "'{:s}'".format(value)
            else:
                value_str = str(value)

            out_str += "# - {" + "{:s}: {:s}".format(key, value_str) + "}\n"

        
        out_str += "# schema: astropy-2.0\n"
        out_str += "datetime,ra,dec,azimuth,altitude,x_image,y_image,mag_data\n"


        # Go though the meteor points
        for (
            t_rel, 
            x_centroid, y_centroid, 
            azim, alt, ra, dec, 
            mag
            ) in zip(
            meteor.time_data, 
            meteor.x_data, meteor.y_data, 
            meteor.azim_data, meteor.elev_data, meteor.ra_data, meteor.dec_data, 
            meteor.mag_data
            ):

            # Compute the absolute time
            frame_time = dt_ref + datetime.timedelta(seconds=t_rel)


            # Precess RA/Dec to J2000
            ra_J2000, dec_J2000 = equatorialCoordPrecession_vect(meteor.jdt_ref, J2000_JD.days, ra, dec)


            # Add an entry to the ECSV file
            entry = [
                frame_time.strftime(isodate_format_entry),
                "{:10.6f}".format(np.degrees(ra_J2000)), "{:+10.6f}".format(np.degrees(dec_J2000)),
                "{:10.6f}".format(np.degrees(azim)), "{:+10.6f}".format(np.degrees(alt)),
                "{:9.3f}".format(x_centroid), "{:9.3f}".format(y_centroid),
                "{:+7.2f}".format(mag)
                ]

            out_str += ",".join(entry) + "\n"


        # Write file to disk
        with open(ecsv_path, 'w') as f:
            f.write(out_str)


        print("ESCV file saved to:", ecsv_path)
            




if __name__ == "__main__":

    import argparse


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Run the trajectory solver on DFN ECSV files.")

    arg_parser.add_argument('ecsv_files', nargs="+", metavar='ECSV_PATH', type=str, \
        help="Path to 2 of more ECSV files. Wildcards are supported, so e.g. /path/to/*.ecsv also works.")

    # Add other solver options
    arg_parser = addSolverOptions(arg_parser, skip_velpart=True)

    arg_parser.add_argument('-p', '--velpart', metavar='VELOCITY_PART', \
        help="Fixed part from the beginning of the meteor on which the initial velocity estimation using the sliding fit will start. Default is 0.4 (40 percent), but for noisier data this might be bumped up to 0.5.", \
        type=float, default=0.4)

    arg_parser.add_argument('-w', '--walk', \
        help="Recursively find all ECSV files in the given folder and use them for trajectory estimation. If a directory containing the file contains the word 'REJECT', it will be skipped. ", \
        action="store_true")
    
    arg_parser.add_argument('--writemilig', metavar='MILIG_PATH', type=str, \
        help="Write the observations to a MILIG input file and exit. The MILIG_PATH argument is the path to the output file.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    ### Parse command line arguments ###

    ecsv_paths = []
    print('Using ECSV files:')


    # If the recursive walk option is given, find all ECSV files recursively in the given folder
    if cml_args.walk:

        # Take the dir path as the given path
        dir_path = cml_args.ecsv_files[0]

        # Find all manual reduction ECSV files in the given folder
        ecsv_names = []
        for entry in sorted(os.walk(dir_path), key=lambda x: x[0]):

            dir_name, _, file_names = entry

            # Skip all directories with the word "REJECT" in them
            if "REJECT" in dir_name:
                print("Directory {:s} skipped because it contains 'REJECT'.".format(dir_name))
                continue

            # Add all ECSV files with picks to the processing list
            for fn in file_names:
                if fn.lower().endswith(".ecsv"):

                    # Add ECSV file, but skip duplicates
                    if fn not in ecsv_names:
                        ecsv_paths.append(os.path.join(dir_name, fn))
                        ecsv_names.append(fn)

                        print(fn)


    else:
        for ecsv_p in cml_args.ecsv_files:
            for ecsv_full_p in glob.glob(ecsv_p):
                ecsv_full_path = os.path.abspath(ecsv_full_p)

                # Check that the path exists
                if os.path.exists(ecsv_full_path):
                    ecsv_paths.append(ecsv_full_path)
                    print(ecsv_full_path)
                else:
                    print('File not found:', ecsv_full_path)


        # Extract dir path
        dir_path = os.path.dirname(ecsv_paths[0])


    # Load the observations into container objects
    jdt_ref, meteor_list = loadECSVs(ecsv_paths)


    # Write the observations to a MILIG input file and exit
    if cml_args.writemilig:
        writeMiligInputFileMeteorObservation(jdt_ref, meteor_list, cml_args.writemilig)
        print("MILIG input file written to:", cml_args.writemilig)
        print("Exiting...")
        sys.exit()



    # Check that there are more than 2 ECSV files given
    if len(ecsv_paths) < 2:
        print("At least 2 files are needed for trajectory estimation!")
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


    # Solve the trajectory
    traj = solveTrajectoryGeneric(jdt_ref, meteor_list, dir_path, solver=cml_args.solver, \
        max_toffset=max_toffset, monte_carlo=(not cml_args.disablemc), mc_runs=cml_args.mcruns, \
        geometric_uncert=cml_args.uncertgeom, gravity_correction=(not cml_args.disablegravity), 
        gravity_factor=cml_args.gfact,
        plot_all_spatial_residuals=cml_args.plotallspatial, plot_file_type=cml_args.imgformat, \
        show_plots=(not cml_args.hideplots), v_init_part=velpart, v_init_ht=vinitht, \
        show_jacchia=cml_args.jacchia,
        estimate_timing_vel=(False if cml_args.notimefit is None else cml_args.notimefit), \
        fixed_times=cml_args.fixedtimes, mc_noise_std=cml_args.mcstd, enable_OSM_plot=cml_args.enableOSM)

