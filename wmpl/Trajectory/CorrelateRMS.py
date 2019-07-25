""" Script which automatically pairs meteor observations from different RMS stations and computes
    trajectories. 
"""

from __future__ import print_function, division, absolute_import

import os
import sys
import re
import argparse
import json
import copy
import datetime

import numpy as np
import matplotlib

from wmpl.Formats.CAMS import loadFTPDetectInfo
from wmpl.Trajectory.CorrelateEngine import TrajectoryCorrelator, TrajectoryConstraints
from wmpl.Utils.TrajConversions import jd2Date



### CONSTANTS ###

# Name of the ouput trajectory directory
OUTPUT_TRAJ_DIR = "trajectories"

# Name of json file with the list of processed directories
JSON_DB_NAME = "processed.json"

### ###


class DatabaseJSON(object):
    def __init__(self, db_file_path):

        self.db_file_path = db_file_path

        # List of processed directories
        self.processed_dirs = {}

        # Load the database from a JSON file
        self.load()


    def load(self):
        """ Load the database from a JSON file. """

        if os.path.exists(self.db_file_path):
            with open(self.db_file_path) as f:
                self.__dict__ = json.load(f)


    def save(self):
        """ Save the database of processed meteors to disk. """

        with open(self.db_file_path, 'w') as f:
            self2 = copy.deepcopy(self)
            f.write(json.dumps(self2, default=lambda o: o.__dict__, indent=4, sort_keys=True))


    def addProcessedDir(self, station_name, rel_proc_path):
        """ Add the processed directory to the list. """

        if station_name in self.processed_dirs:
            if not rel_proc_path in self.processed_dirs[station_name]:
                self.processed_dirs[station_name].append(rel_proc_path)



class MeteorPointRMS(object):
    def __init__(self, frame, time_rel, ra, dec, azim, alt, mag):
        """ Container for individual meteor picks. """

        # Frame number since the beginning of the FF file
        self.frame = frame
        
        # Relative time
        self.time_rel = time_rel

        # Image coordinats
        self.x = None
        self.y = None
        
        # Equatorial coordinates (J2000, deg)
        self.ra = ra
        self.dec = dec

        # Horizontal coordinates (J2000, deg), azim is +E of due N
        self.azim = azim
        self.alt = alt

        self.intensity_sum = None

        self.mag = mag


class MeteorObsRMS(object):
    def __init__(self, station_code, reference_dt, platepar, data, rel_proc_path):
        """ Container for meteor observations with the interface compatible with the trajectory correlator
            interface. 
        """

        self.station_code = station_code

        self.reference_dt = reference_dt
        self.platepar = platepar
        self.data = data

        # Path to the directory with data
        self.rel_proc_path = rel_proc_path

        self.processed = False 
        self.paired = False



class PlateparDummy:
    def __init__(self, **entries):
        """ This class takes a platepar dictionary and converts it into an object. """
        self.__dict__.update(entries)



class RMSDataHandle(object):
    def __init__(self, dir_path):
        """ Handles data interfacing between the trajectory correlator and RMS data files on disk. 
    
        Arguments:
            dir_path: [str] Path to the directory with data files. 
        """

        self.dir_path = dir_path

        print("Using directory:", self.dir_path)

        # Load the list of stations
        station_list = self.loadStations()

        # Load database of processed folders
        self.db = DatabaseJSON(os.path.join(self.dir_path, JSON_DB_NAME))

        # Find unprocessed meteor files
        processing_list = self.findUnprocessedFolders(station_list)

        # Load data of unprocessed observations
        self.unprocessed_observations = self.loadUnprocessedObservations(processing_list)


    def loadStations(self):
        """ Load the station names in the processing folder. """

        station_list = []

        for dir_name in os.listdir(self.dir_path):

            # Check if the dir name matches the station name pattern
            if os.path.isdir(os.path.join(self.dir_path, dir_name)):
                if re.match("^[A-Z]{2}[A-Z0-9]{4}$", dir_name):
                    print("Using station:", dir_name)
                    station_list.append(dir_name)
                else:
                    print("Skipping directory:", dir_name)


        return station_list



    def findUnprocessedFolders(self, station_list):
        """ Go through directories and find folders with unprocessed data. """

        processing_list = []

        # Go through all station directories
        for station_name in station_list:

            station_path = os.path.join(self.dir_path, station_name)

            # Add the station name to the database if it doesn't exist
            if station_name not in self.db.processed_dirs:
                self.db.processed_dirs[station_name] = []

            # Go through all directories in stations
            for night_name in os.listdir(station_path):

                night_path = os.path.join(station_path, night_name)

                # If the night path is not in the processed list, add it to the processing list
                if night_path not in self.db.processed_dirs[station_name]:
                    night_path_rel = os.path.join(station_name, night_name)
                    processing_list.append([station_name, night_path_rel, night_path])


        return processing_list



    def initMeteorObs(self, station_code, ftpdetectinfo_path, platepars_recalibrated_dict):
        """ Init meteor observations from the FTPdetectinfo file and recalibrated platepars. """

        # Load station coordinates
        if len(list(platepars_recalibrated_dict.keys())):
            
            pp_dict = platepars_recalibrated_dict[list(platepars_recalibrated_dict.keys())[0]]
            pp = PlateparDummy(**pp_dict)
            stations_dict = {station_code: [np.radians(pp.lat), np.radians(pp.lon), pp.elev]}

            # Load the FTPdetectinfo file
            meteor_list = loadFTPDetectInfo(ftpdetectinfo_path, stations_dict)

        else:
            meteor_list = []


        return meteor_list



    def loadUnprocessedObservations(self, processing_list):
        """ Load unprocessed meteor observations. """

        # Go through folders for processing
        met_obs_list = []
        for station_code, rel_proc_path, proc_path in processing_list:

            ftpdetectinfo_name = None
            platepar_recalibrated_name = None

            # Find FTPdetectinfo and platepar files
            for name in os.listdir(proc_path):
                    
                # Find FTPdetectinfo
                if name.startswith("FTPdetectinfo") and name.endswith('.txt') and \
                    (not "backup" in name) and (not "uncalibrated" in name):
                    ftpdetectinfo_name = name
                    continue

                if name == "platepars_all_recalibrated.json":
                    platepar_recalibrated_name = name
                    continue

            # Skip these observations if no data files were found inside
            if (ftpdetectinfo_name is None) or (platepar_recalibrated_name is None):
                print("Skipping {:s} due to missing data files...".format(rel_proc_path))

                # Add the folder to the list of processed folders
                self.db.addProcessedDir(station_code, rel_proc_path)

                continue


            # Load platepars
            with open(os.path.join(proc_path, platepar_recalibrated_name)) as f:
                platepars_recalibrated_dict = json.load(f)

            # If all files exist, init the meteor container object
            cams_met_obs_list = self.initMeteorObs(station_code, os.path.join(proc_path, \
                ftpdetectinfo_name), platepars_recalibrated_dict)

            # Format the observation object to the one required by the trajectory correlator
            for cams_met_obs in cams_met_obs_list:

                # Get the platepar
                pp_dict = platepars_recalibrated_dict[cams_met_obs.ff_name]
                pp = PlateparDummy(**pp_dict)

                # Init meteor data
                meteor_data = []
                for entry in zip(cams_met_obs.frames, cams_met_obs.time_data, cams_met_obs.azim_data, \
                    cams_met_obs.elev_data, cams_met_obs.ra_data, cams_met_obs.dec_data, \
                    cams_met_obs.mag_data):

                    frame, time_rel, azim, alt, ra, dec, mag = entry

                    met_point = MeteorPointRMS(frame, time_rel, np.degrees(ra), np.degrees(dec), \
                        np.degrees(azim), np.degrees(alt), mag)

                    meteor_data.append(met_point)


                # Init the new meteor observation object
                met_obs = MeteorObsRMS(station_code, jd2Date(cams_met_obs.jdt_ref, dt_obj=True), pp, \
                    meteor_data, rel_proc_path)

                print(station_code, met_obs.reference_dt, rel_proc_path)

                met_obs_list.append(met_obs)


        return met_obs_list



    def getPlatepar(self, met_obs):
        """ Return the platepar of the meteor observation. """

        return met_obs.platepar



    def getUnprocessedObservations(self):
        """ Returns a list of unprocessed meteor observations. """

        return self.unprocessed_observations



    def findTimePairs(self, met_obs, max_toffset):
        """ Finds pairs in time between the given meteor observations and all other observations from 
            different stations. 
        """

        found_pairs = []

        # Compute datetime of the mean time of the meteor
        met_obs_mean_dt = met_obs.reference_dt + datetime.timedelta(seconds=np.mean([entry.time_rel \
            for entry in met_obs.data]))

        # Go through all meteors from other stations
        for met_obs2 in self.unprocessed_observations:

            # Take only observations from different stations
            if met_obs.station_code == met_obs2.station_code:
                continue

            # Compute datetime of the mean time of the meteor
            met_obs2_mean_dt = met_obs2.reference_dt + datetime.timedelta(seconds=np.mean([entry.time_rel \
                for entry in met_obs2.data]))

            # Take observations which are within the given time window
            if abs((met_obs_mean_dt - met_obs2_mean_dt).total_seconds()) <= max_toffset:
                found_pairs.append(met_obs2)


        return found_pairs



    def saveTrajectoryResults(self, traj):
        """ Save trajectory results to the disk. """


        # Generate the name for the output directory (add list of country codes at the end)
        output_dir = os.path.join(self.dir_path, OUTPUT_TRAJ_DIR, \
            jd2Date(traj.jdt_ref, dt_obj=True).strftime("%Y%m%d_%H%M%S.%f")[:-3] + "_" \
            + "_".join(list(set([obs.station_id[:2] for obs in traj.observations]))))

        # Save the report
        traj.saveReport(output_dir, traj.file_name + '_report.txt', uncertanties=traj.uncertanties, 
            verbose=False)

        # Save the plots
        traj.save_results = True
        traj.savePlots(output_dir, traj.file_name, show_plots=False)
        traj.save_results = False



    def markObservationAsProcessed(self, met_obs):
        """ Mark the given meteor observation as processed. """

        self.db.addProcessedDir(met_obs.station_code, met_obs.rel_proc_path)



    def finish(self):
        pass



if __name__ == "__main__":

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="""Automatically compute trajectories from RMS data in the given directory. 
The directory structure needs to be the following, for example:
    ./ # root directory
        ./CameraSites.txt
        ./CameraTimeOffsets.txt
        /HR0001/
            /HR0001_20190707_192835_241084_detected
                ./FTPdetectinfo_HR0001_20190707_192835_241084.txt
                ./platepars_all_recalibrated.json
        /HR0004/
            ./FTPdetectinfo_HR0004_20190707_193044_498581.txt
            ./platepars_all_recalibrated.json
        /...

The root directory has to contain the CAMS-type CameraSites.txt file with the optional CameraTimeOffsets file.
Next, is should contain directories of stations (station codes need to be exact), and these directories should
contain data folders. Data folders should have FTPdetectinfo files together with platepar files.""",
        formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('dir_path', type=str, help='Path to the data directory. Trajectory helper files will be stored here as well.')

    arg_parser.add_argument('-t', '--maxtoffset', metavar='MAX_TOFFSET', \
        help='Maximum time offset between the stations. Default is 5 seconds.', type=float, default=5.0)

    arg_parser.add_argument('-s', '--maxstationdist', metavar='MAX_STATION_DIST', \
        help='Maximum distance (km) between stations of paired meteors. Default is 300 km.', type=float, \
        default=300.0)

    arg_parser.add_argument('-v', '--maxveldiff', metavar='MAX_VEL_DIFF', \
        help='Maximum difference in percent between velocities between two stations. Default is 25%.', \
        type=float, default=25.0)

    arg_parser.add_argument('-p', '--velpart', metavar='VELOCITY_PART', \
        help='Fixed part from the beginning of the meteor on which the initial velocity estimation using the sliding fit will start. Default is 0.25 (25 percent), but for noisier data this might be bumped up to 0.5.', \
        type=float, default=0.25)

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    ############################

    # Init the data handler
    data_handle = RMSDataHandle(cml_args.dir_path)


    # Set matplotlib for headless running
    matplotlib.use('Agg')


    # Init trajectory constraints
    trajectory_constraints = TrajectoryConstraints()
    trajectory_constraints.max_toffset = cml_args.maxtoffset
    trajectory_constraints.max_station_dist = cml_args.maxstationdist
    trajectory_constraints.max_vel_percent_diff = cml_args.maxveldiff

    # Run the trajectory correlator
    tc = TrajectoryCorrelator(data_handle, trajectory_constraints, cml_args.velpart, data_in_j2000=True)
    tc.run()