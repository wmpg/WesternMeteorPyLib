""" Script which automatically pairs meteor observations from different RMS stations and computes
    trajectories. 
"""

from __future__ import print_function, division, absolute_import

import os
import re
import argparse
import json
import copy
import datetime
import shutil
import time
import multiprocessing
import logging
import logging.handlers
import glob
from dateutil.relativedelta import relativedelta
import numpy as np
import sys
import signal
import secrets

from wmpl.Formats.CAMS import loadFTPDetectInfo
from wmpl.Trajectory.CorrelateEngine import TrajectoryCorrelator, TrajectoryConstraints
from wmpl.Utils.Math import generateDatetimeBins
from wmpl.Utils.OSTools import mkdirP
from wmpl.Utils.Pickling import loadPickle, savePickle
from wmpl.Utils.TrajConversions import datetime2JD, jd2Date
from wmpl.Utils.remoteDataHandling import RemoteDataHandler
from wmpl.Trajectory.CorrelateDB import ObservationDatabase, TrajectoryDatabase
from wmpl.Trajectory.Trajectory import Trajectory

from wmpl.Trajectory.CorrelateEngine import MCMODE_CANDS, MCMODE_PHASE1, MCMODE_PHASE2, MCMODE_ALL, MCMODE_BOTH

### CONSTANTS ###

# Name of the ouput trajectory directory
OUTPUT_TRAJ_DIR = "trajectories"

# Name of json file with the list of processed directories
JSON_DB_NAME = "processed_trajectories.json"

# Auto run frequency (hours)
AUTO_RUN_FREQUENCY = 6

### ###

log = logging.getLogger("traj_correlator")


class TrajectoryReduced(object):
    def __init__(self, traj_file_path, json_dict=None, traj_obj=None):
        """ Reduced representation of the Trajectory object which helps to save memory. 

        Arguments:
            traj_file_path: [str] Full path to the trajectory object.

        Keyword arguments:
            json_dict: [dict] Load values from a dictionary instead of a traj pickle file. None by default,
                in which case a traj pickle file will be loaded.
            traj_obj: [Trajectory instance] Load values from a full Trajectory object, instead from the disk.
                None by default.
        """

        # Load values from a JSON pickle file
        if json_dict is None:

            if traj_obj is None:

                # Full path to the trajectory object (saved to it can be loaded later if needed)
                self.traj_file_path = traj_file_path

                # Load the trajectory object
                try:
                    traj = loadPickle(*os.path.split(traj_file_path))
                except EOFError:
                    log.info("Pickle file could not be loaded: " + traj_file_path)
                    return None

                except FileNotFoundError:
                    log.info("Pickle file not found: " + traj_file_path)
                    return None
                
                except:
                    log.info("Pickle file could not be loaded: " + traj_file_path)
                    return None

            else:

                # Load values from a given trajectory file
                traj = traj_obj
                self.traj_file_path = os.path.join(traj.output_dir, traj.file_name + "_trajectory.pickle")

            # Reference Julian date (beginning of the meteor)
            self.jdt_ref = traj.jdt_ref

            # ECI coordinates of the state vector computed through minimization
            if traj.state_vect_mini is None:
                self.state_vect_mini = None
            else:
                self.state_vect_mini = traj.state_vect_mini.tolist()

            # Apparent radiant vector computed through minimization
            if traj.radiant_eci_mini is None:
                self.radiant_eci_mini = None
            else:
                self.radiant_eci_mini = traj.radiant_eci_mini.tolist()

            # Initial and average velocity
            self.v_init = traj.v_init
            self.v_avg = traj.v_avg

            # Coordinates of the first point (observed)
            self.rbeg_lat = traj.rbeg_lat
            self.rbeg_lon = traj.rbeg_lon
            self.rbeg_ele = traj.rbeg_ele
            self.rbeg_jd = traj.rbeg_jd

            # Coordinates of the last point (observed)
            self.rend_lat = traj.rend_lat
            self.rend_lon = traj.rend_lon
            self.rend_ele = traj.rend_ele
            self.rend_jd = traj.rend_jd

            # Save the gravity factor
            self.gravity_factor = traj.gravity_factor

            # Save the vertical velocity v0z
            self.v0z = traj.v0z

            # Stations participating in the solution
            self.participating_stations = sorted([obs.station_id for obs in traj.observations 
                if obs.ignore_station is False])

            # Ignored stations
            self.ignored_stations = sorted([obs.station_id for obs in traj.observations 
                if obs.ignore_station is True])

            if hasattr(traj, 'phase_1_only'):
                self.phase_1_only = traj.phase_1_only
            else:
                self.phase_1_only = False

            if hasattr(traj, 'traj_id'):
                self.traj_id = traj.traj_id

        # Load values from a dictionary
        else:
            self.__dict__ = json_dict



class DatabaseJSON(object):
    def __init__(self, db_file_path, verbose=False):

        self.db_file_path = db_file_path

        # List of paired observations as a part of a trajectory (keys are station codes, values are unique 
        #   observation IDs)
        self.paired_obs = {}

        # List of processed trajectories (keys are trajectory reference julian dates, values are \
        #   TrajectoryReduced objects)
        self.trajectories = {}

        # List of failed trajectories (keys are trajectory reference julian dates, values are \
        #   TrajectoryReduced objects)
        self.failed_trajectories = {}

        # Load the database from a JSON file
        self.load(verbose=verbose)

    def load(self, verbose=False):
        """ Load the database from a JSON file. """

        # location of last backup of the database, in case we need to use it
        db_bak_file_path = self.db_file_path + ".bak"

        if os.path.exists(self.db_file_path):
            db_file_path_saved = self.db_file_path

            # Load the value from the database
            # if there's a problem with the file, try using the last backup
            db_is_ok = False
            try:
                self.__dict__ = json.load(open(self.db_file_path))
                db_is_ok = True

            except Exception:
                db_is_ok = False
                log.warning('trajectory database damaged, trying backup')

                try:
                    if os.path.exists(db_bak_file_path):
                        shutil.copy2(db_bak_file_path, self.db_file_path)
                        self.__dict__ = json.load(open(self.db_file_path))
                        db_is_ok = True

                except Exception:
                    log.warning('unable to find a useable trajectory database')
                    db_is_ok = False

            # Overwrite the database path with the saved one
            self.db_file_path = db_file_path_saved

            # if the trajectories attribute is not present, then the database has been converted to sqlite            
            if db_is_ok and hasattr(self, 'trajectories'):
                # Convert trajectories from JSON to TrajectoryReduced objects
                for traj_dict_str in ["trajectories", "failed_trajectories"]:
                    traj_dict = getattr(self, traj_dict_str)
                    trajectories_obj_dict = {}
                    for traj_json in traj_dict:
                        traj_reduced_tmp = TrajectoryReduced(None, json_dict=traj_dict[traj_json])

                        trajectories_obj_dict[traj_reduced_tmp.jdt_ref] = traj_reduced_tmp

                    # Set the trajectory dictionary
                    setattr(self, traj_dict_str, trajectories_obj_dict)

        # do this here because the object dict is overwritten during the load operation above
        self.verbose = verbose

    def save(self):
        """ Save the database of processed meteors to disk. """

        # Back up the existing data base
        db_bak_file_path = self.db_file_path + ".bak"
        if os.path.exists(self.db_file_path):
            shutil.copy2(self.db_file_path, db_bak_file_path)
        else:
            return 

        # Save the data base
        try:
            with open(self.db_file_path, 'w') as f:
                self2 = copy.deepcopy(self)

                # Convert reduced trajectory objects to JSON objects
                if hasattr(self2,'trajectories'):
                    self2.trajectories = {key: self.trajectories[key].__dict__ for key in self.trajectories}
                if hasattr(self2, 'failed_trajectories'):
                    self2.failed_trajectories = {key: self.failed_trajectories[key].__dict__ 
                    for key in self.failed_trajectories}
                if hasattr(self2, 'phase1Trajectories'):
                    delattr(self2, 'phase1Trajectories')

                f.write(json.dumps(self2, default=lambda o: o.__dict__, indent=4, sort_keys=True))

            # Remove the backup file
            if os.path.exists(db_bak_file_path):
                os.remove(db_bak_file_path)

        except Exception as e:
            log.warning('unable to save the database, likely corrupt data')
            shutil.copy2(db_bak_file_path, self.db_file_path)
            log.warning(e)

    def checkTrajIfFailed(self, traj):
        """ Check if the given trajectory has been computed with the same observations and has failed to be
            computed before.

        """

        # Check if the reference time is in the list of failed trajectories
        if traj.jdt_ref in self.failed_trajectories:

            # Get the failed trajectory object
            failed_traj = self.failed_trajectories[traj.jdt_ref]

            # Check if the same observations participate in the failed trajectory as in the trajectory that
            #   is being tested
            all_match = True
            for obs in traj.observations:
                
                if not ((obs.station_id in failed_traj.participating_stations) or (obs.station_id in failed_traj.ignored_stations)):

                    all_match = False
                    break

            # If the same stations were used, the trajectory estimation failed before
            if all_match:
                return True

        return False

    def addTrajectory(self, traj_reduced, failed=False):
        """ Add a computed trajectory to the list. 
    
        Arguments:
            traj_file_path: [str] Full path the trajectory object.

        Keyword arguments:
            traj_obj: [bool] Instead of loading a traj object from disk, use the given object.
            failed: [bool] Add as a failed trajectory. False by default.
        """

        if traj_reduced is None or not hasattr(traj_reduced, "jdt_ref"):
            return None

        if self.verbose:
            log.info(f' loaded {traj_reduced.traj_file_path}, traj_id {traj_reduced.traj_id}')


        # Choose to which dictionary the trajectory will be added
        if failed:
            traj_dict = self.failed_trajectories

        else:
            traj_dict = self.trajectories


        # Add the trajectory to the list (key is the reference JD)
        if traj_reduced.jdt_ref not in traj_dict:
            traj_dict[traj_reduced.jdt_ref] = traj_reduced
        else:
            traj_dict[traj_reduced.jdt_ref].traj_id = traj_reduced.traj_id

    def removeTrajectory(self, traj_reduced, keepFolder=False):
        """ Remove the trajectory from the data base and disk. """

        # Remove the trajectory data base entry
        if traj_reduced.jdt_ref in self.trajectories:
            del self.trajectories[traj_reduced.jdt_ref]

        # Remove the trajectory folder on the disk
        if not keepFolder and os.path.isfile(traj_reduced.traj_file_path):
            traj_dir = os.path.dirname(traj_reduced.traj_file_path)
            shutil.rmtree(traj_dir, ignore_errors=True)
            if os.path.isfile(traj_reduced.traj_file_path):
                log.info(f'unable to remove {traj_dir}')        


class MeteorPointRMS(object):
    def __init__(self, frame, time_rel, x, y, ra, dec, azim, alt, mag):
        """ Container for individual meteor picks. """

        # Frame number since the beginning of the FF file
        self.frame = frame
        
        # Relative time
        self.time_rel = time_rel

        # Image coordinats
        self.x = x
        self.y = y
        
        # Equatorial coordinates (J2000, deg)
        self.ra = ra
        self.dec = dec

        # Horizontal coordinates (J2000, deg), azim is +E of due N
        self.azim = azim
        self.alt = alt

        self.intensity_sum = None

        self.mag = mag


class MeteorObsRMS(object):
    def __init__(self, station_code, reference_dt, platepar, data, rel_proc_path, ff_name=None):
        """ Container for meteor observations with the interface compatible with the trajectory correlator
            interface. 

            Arguments:
                station_code: [str] RMS station code.
                reference_dt: [datetime] Datetime when the relative time is t = 0.
                platepar: [Platepar object] RMS calibration plate for the given observations.
                data: [list] A list of MeteorPointRMS objects.
                rel_proc_path: [str] Path to the folder with the nighly observations for this meteor.

            Keyword arguments:
                ff_name: [str] Name of the FF file(s) which contains the meteor.
        """

        self.station_code = station_code

        self.reference_dt = reference_dt
        self.platepar = platepar
        self.data = data

        # Path to the directory with data
        self.rel_proc_path = rel_proc_path

        self.ff_name = ff_name

        # Internal flags to control the processing flow
        # NOTE: The processed flag should always be set to False for every observation when the program starts
        self.processed = False 

        # Mean datetime of the observation
        self.mean_dt = self.reference_dt + datetime.timedelta(seconds=np.mean([entry.time_rel 
            for entry in self.data]))

        
        ### Estimate if the meteor begins and ends inside the FOV ###

        self.fov_beg = False
        self.fov_end = False

        half_index = len(data)//2


        # Find angular velocity at the beginning per every axis
        dxdf_beg = (self.data[half_index].x - self.data[0].x)/(self.data[half_index].frame 
            - self.data[0].frame)
        dydf_beg = (self.data[half_index].y - self.data[0].y)/(self.data[half_index].frame 
            - self.data[0].frame)

        # Compute locations of centroids 2 frames before the beginning
        x_pre_begin = self.data[0].x - 2*dxdf_beg
        y_pre_begin = self.data[0].y - 2*dydf_beg

        # If the predicted point is inside the FOV, mark it as such
        if (x_pre_begin > 0) and (x_pre_begin <= self.platepar.X_res) and (y_pre_begin > 0) \
                and (y_pre_begin < self.platepar.Y_res):

            self.fov_beg = True

        # If the starting point is not inside the FOV, exlude the first point
        else:
            self.data = self.data[1:]

            # Recompute the halfway point
            half_index = len(self.data)//2


        # If there is no data or the length is too short, skip the observation
        if (len(self.data) == 0) or (len(self.data) < half_index):
            self.bad_data = True
            return None
            
        else:
            self.bad_data = False


        # Find angular velocity at the ending per every axis
        dxdf_end = (self.data[-1].x - self.data[half_index].x)/(self.data[-1].frame 
            - self.data[half_index].frame)
        dydf_end = (self.data[-1].y - self.data[half_index].y)/(self.data[-1].frame 
            - self.data[half_index].frame)

        # Compute locations of centroids 2 frames after the end
        x_post_end = self.data[-1].x + 2*dxdf_end
        y_post_end = self.data[-1].y + 2*dydf_end

        # If the predicted point is inside the FOV, mark it as such
        if (x_post_end > 0) and (x_post_end <= self.platepar.X_res) and (y_post_end > 0) \
                and (y_post_end <= self.platepar.Y_res):
            
            self.fov_end = True

        # If the ending point is not inside fully inside the FOV, exclude it
        else:
            self.data = self.data[:-1]

        ### ###


        # Generate a unique observation ID, the format is: STATIONID_YYYYMMDD-HHMMSS.us_CHECKSUM
        #  where CHECKSUM is the last four digits of the sum of all observation image X cordinates
        checksum = int(np.sum([entry.x for entry in self.data]) % 10000)
        self.id = "{:s}_{:s}_{:04d}".format(self.station_code, self.mean_dt.strftime("%Y%m%d-%H%M%S.%f"), 
            checksum)



class PlateparDummy:
    def __init__(self, **entries):
        """ This class takes a platepar dictionary and converts it into an object. """

        self.__dict__.update(entries)



class RMSDataHandle(object):
    def __init__(self, dir_path, dt_range=None, db_dir=None, output_dir=None, mcmode=MCMODE_ALL, max_trajs=1000, verbose=False, archivemonths=3):
        """ Handles data interfacing between the trajectory correlator and RMS data files on disk. 
    
        Arguments:
            dir_path: [str] Path to the directory with data files. 

        Keyword arguments:
            dt_range: [list of datetimes] A range of datetimes between which the existing trajectories will be
                loaded.
            db_dir: [str] Path to the directory with the database file. None by default, in which case the
                database file will be loaded from the dir_path.
            output_dir: [str] Path to the directory where the output files will be saved. None by default, in
                which case the output files will be saved in the dir_path.
            mcmode: [int] the operation mode, candidates, phase1 simple solns, mc phase or a combination
            max_trajs: [int] maximum number of phase1 trajectories to load at a time when adding uncertainties. Improves throughput.
        """

        self.mc_mode = mcmode

        self.dir_path = dir_path

        self.dt_range = dt_range

        log.info("Using directory: " + self.dir_path)


        # Set the database directory
        if db_dir is None:
            db_dir = self.dir_path
        self.db_dir = db_dir

        # Create the database directory if it doesn't exist
        mkdirP(self.db_dir)


        # Set the output directory
        if output_dir is None:
            output_dir = self.dir_path
        self.output_dir = output_dir

        # Create the output directory if it doesn't exist
        mkdirP(self.output_dir)

        # Candidate directory, if running in create or load cands modes
        self.candidate_dir = os.path.join(self.output_dir, 'candidates')
        if not self.mc_mode & MCMODE_PHASE2:
            mkdirP(os.path.join(self.candidate_dir, 'processed'))

        # Phase 1 trajectory pickle directory needed to reload previous results.
        self.phase1_dir = os.path.join(self.output_dir, 'phase1')
        if self.mc_mode & MCMODE_PHASE1:
            mkdirP(os.path.join(self.phase1_dir, 'processed'))
            self.purgePhase1ProcessedData(os.path.join(self.phase1_dir, 'processed'))

        self.verbose = verbose

        ############################

        # Load database of processed folders
        database_path = os.path.join(self.db_dir, JSON_DB_NAME)

        log.info("")

        if mcmode != MCMODE_PHASE2:

            # no need to load the legacy JSON file if we already have the sqlite databases
            if not os.path.isfile(os.path.join(db_dir, 'observations.db')) and \
               not os.path.isfile(os.path.join(db_dir, 'trajectories.db')):
                log.info("Loading database: {:s}".format(database_path))
                self.old_db = DatabaseJSON(database_path, verbose=self.verbose)
            else:
                self.old_db = None

            self.observations_db = ObservationDatabase(db_dir)
            if hasattr(self.old_db, 'paired_obs'):
                # move any legacy paired obs data into sqlite
                self.observations_db.moveObsJsonRecords(self.old_db.paired_obs, dt_range)

            self.traj_db = TrajectoryDatabase(db_dir)
            if hasattr(self.old_db, 'failed_trajectories'):
                # move any legacy failed traj data into sqlite
                self.traj_db.moveFailedTrajectories(self.old_db.failed_trajectories, dt_range)

            if archivemonths != 0:
                log.info('Archiving older entries....')
                try:
                    self.archiveOldRecords(older_than=archivemonths)
                except: 
                    pass
                log.info("   ... done!")

            # Load the list of stations
            station_list = self.loadStations()

            # Find unprocessed meteor files
            log.info("")
            log.info("Finding unprocessed data...")
            self.processing_list = self.findUnprocessedFolders(station_list)
            log.info("   ... done!")

            # in phase 1, initialise and collect data second as we load candidates dynamically
            self.initialiseRemoteDataHandling()

        else:
            # in phase 2, initialise and collect data first as we need the phase1 traj on disk already
            self.traj_db = None
            self.observations_db = None
            self.initialiseRemoteDataHandling()

            dt_beg, dt_end = self.loadPhase1Trajectories(max_trajs=max_trajs)
            self.processing_list = None
            self.dt_range=[dt_beg, dt_end]

        ### Define country groups to speed up the proceessing ###

        north_america_group = ["CA", "US", "MX"]

        south_america_group = ["AR", "BO", "BR", "CL", "CO", "EC", "FK", "GF", "GY", "GY", "PY", "PE", "SR", 
            "UY", "VE"]

        europe_group = ["AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE", 
            "IT", "LV", "LT", "LU", "MT", "NL", "PO", "PT", "RO", "SK", "SI", "ES", "SE", "AL", "AD", "AM", 
            "BY", "BA", "FO", "GE", "GI", "IM", "XK", "LI", "MK", "MD", "MC", "ME", "NO", "RU", "SM", "RS", 
            "CH", "TR", "UA", "UK", "VA"]

        new_zealand_group = ["NZ"]

        australia_group = ["AU"]


        self.country_groups = [north_america_group, south_america_group, europe_group, new_zealand_group, 
            australia_group]

        ### ###


    def initialiseRemoteDataHandling(self):
        # Initialise remote data handling, if the config file is present
        remote_cfg = os.path.join(self.db_dir, 'wmpl_remote.cfg')
        if os.path.isfile(remote_cfg):
            log.info('remote data management requested, initialising')
            self.RemoteDatahandler = RemoteDataHandler(remote_cfg)
            if self.RemoteDatahandler.mode == 'child':
                self.RemoteDatahandler.clearStopFlag()
                status = self.getRemoteData(verbose=True)
            else:
                status = self.moveUploadedData(verbose=False)                
            if not status:
                log.info('no remote data yet')
        else:
            self.RemoteDatahandler = None


    def purgePhase1ProcessedData(self, dir_path):
        """ Purge old phase1 processed data if it is older than 90 days. """

        refdt = time.time() - 90*86400
        result = []
        for path, _, files in os.walk(dir_path):

            for file in files:

                file_path = os.path.join(path, file)

                # Check if the file is older than the reference date
                try:
                    file_dt = os.stat(file_path).st_mtime
                except FileNotFoundError:
                    log.warning(f"File not found: {file_path}")
                    continue

                if os.path.exists(file_path) and (file_dt < refdt) and os.path.isfile(file_path):
                    
                    try:
                        os.remove(file_path)
                        result.append(file_path)

                    except FileNotFoundError:
                        log.warning(f"File not found: {file_path}")

                    except Exception as e:
                        log.error(f"Error removing file {file_path}: {e}")
        
        return result


    def archiveOldRecords(self, older_than=3):
        """
        Archive off old records to keep the database size down

        Keyword Arguments:
            older_than: [int] number of months to keep, default 3
        """
        class DummyMetObs():
            def __init__(self, station, obs_id):
                self.station_code = station
                self.id = obs_id

        archdate = datetime.datetime.now(datetime.timezone.utc) - relativedelta(months=older_than)
        archdate_jd = datetime2JD(archdate)
        arch_prefix = archdate.strftime("%Y%m")

        # TODO check if this works
        self.observations_db.archiveObsDatabase(self.db_dir, arch_prefix, archdate_jd)
        self.traj_db.archiveTrajDatabase(self.db_dir, arch_prefix, archdate_jd)

        return 

    def loadStations(self):
        """ Load the station names in the processing folder. """

        station_list = []

        for dir_name in sorted(os.listdir(self.dir_path)):

            # Check if the dir name matches the station name pattern
            if os.path.isdir(os.path.join(self.dir_path, dir_name)):
                if re.match("^[A-Z]{2}[A-Z0-9]{4}$", dir_name):
                    log.info("Using station: " + dir_name)
                    station_list.append(dir_name)
                else:
                    log.info("Skipping directory: " + dir_name)


        return station_list



    def findUnprocessedFolders(self, station_list):
        """ Go through directories and find folders with unprocessed data. """

        processing_list = []

        # skipped_dirs = 0

        # Go through all station directories
        for station_name in station_list:

            station_path = os.path.join(self.dir_path, station_name)

            # Go through all directories in stations
            for night_name in os.listdir(station_path):

                # Extract the date and time of directory, if possible
                try:
                    night_dt = datetime.datetime.strptime("_".join(night_name.split("_")[1:3]), 
                        "%Y%m%d_%H%M%S").replace(tzinfo=datetime.timezone.utc)
                except:
                    log.info(f'Could not parse the date of the night dir: {night_name}')
                    night_dt = None
                    continue
                if self.dt_range is not None:
                    # skip folders more than a day older the requested date range
                    if night_dt < (self.dt_range[0]+ datetime.timedelta(days=-1)).replace(tzinfo=datetime.timezone.utc):
                        continue

                night_path = os.path.join(station_path, night_name)
                night_path_rel = os.path.join(station_name, night_name)

                processing_list.append([station_name, night_path_rel, night_path, night_dt])

                # else:
                #     skipped_dirs += 1


        # if skipped_dirs:
        #     log.info("Skipped {:d} processed directories".format(skipped_dirs))

        return processing_list



    def initMeteorObs(self, station_code, ftpdetectinfo_path, platepars_recalibrated_dict):
        """ Init meteor observations from the FTPdetectinfo file and recalibrated platepars. """

        # Load station coordinates
        if len(list(platepars_recalibrated_dict.keys())):
            
            pp_dict = platepars_recalibrated_dict[list(platepars_recalibrated_dict.keys())[0]]
            pp = PlateparDummy(**pp_dict)
            stations_dict = {station_code: [np.radians(pp.lat), np.radians(pp.lon), pp.elev]}

            # Load the FTPdetectinfo file
            meteor_list = loadFTPDetectInfo(ftpdetectinfo_path, stations_dict, join_broken_meteors=False)

        else:
            meteor_list = []


        return meteor_list



    def loadUnpairedObservations(self, processing_list, dt_range=None):
        """ Load unpaired meteor observations, i.e. observations that are not a part of any trajectory. """

        # Go through folders for processing
        unpaired_met_obs_list = []
        prev_station = None
        station_count = 1
        for station_code, rel_proc_path, proc_path, night_dt in processing_list:

            # Check that the night datetime is within the given range of times, if the range is given
            if (dt_range is not None) and (night_dt is not None):
                dt_beg, dt_end = dt_range

                # Skip all folders which are outside the limits
                if (night_dt < dt_beg) or (night_dt > dt_end):
                    continue



            ftpdetectinfo_name = None
            platepar_recalibrated_name = None

            # Skip files, only take directories
            if os.path.isfile(proc_path):
                continue

            log.info("")
            log.info("Processing station: " + station_code)

            # Find FTPdetectinfo and platepar files
            for name in os.listdir(proc_path):
                    
                # Find FTPdetectinfo
                if name.startswith("FTPdetectinfo") and name.endswith('.txt') and \
                        ("backup" not in name) and ("uncalibrated" not in name) and ("unfiltered" not in name):
                    ftpdetectinfo_name = name
                    continue

                if name == "platepars_all_recalibrated.json":

                    try:
                        # Try loading the recalibrated platepars
                        with open(os.path.join(proc_path, name)) as f:
                            platepars_recalibrated_dict = json.load(f)                            
                            platepar_recalibrated_name = name
                            continue

                    except:
                        pass
    

            # Skip these observations if no data files were found inside
            if (ftpdetectinfo_name is None) or (platepar_recalibrated_name is None):
                log.info("  Skipping {:s} due to missing data files...".format(rel_proc_path))
                continue

            if station_code != prev_station:
                station_count += 1
                prev_station = station_code

            # Load platepars
            with open(os.path.join(proc_path, platepar_recalibrated_name)) as f:
                platepars_recalibrated_dict = json.load(f)

            # If all files exist, init the meteor container object
            cams_met_obs_list = self.initMeteorObs(station_code, os.path.join(proc_path, 
                ftpdetectinfo_name), platepars_recalibrated_dict)

            # Format the observation object to the one required by the trajectory correlator
            added_count = 0
            for cams_met_obs in cams_met_obs_list:

                # Get the platepar
                if cams_met_obs.ff_name in platepars_recalibrated_dict:
                    pp_dict = platepars_recalibrated_dict[cams_met_obs.ff_name]
                else:
                    log.info("    Skipping {:s}, not found in platepar dict".format(cams_met_obs.ff_name))
                    continue

                pp = PlateparDummy(**pp_dict)


                # Skip observations which weren't recalibrated
                if hasattr(pp, "auto_recalibrated"):
                    if not pp.auto_recalibrated:
                        log.info("    Skipping {:s}, not recalibrated!".format(cams_met_obs.ff_name))
                        continue


                # Init meteor data
                meteor_data = []
                for entry in zip(cams_met_obs.frames, cams_met_obs.time_data, cams_met_obs.x_data,
                        cams_met_obs.y_data, cams_met_obs.azim_data, cams_met_obs.elev_data, 
                        cams_met_obs.ra_data, cams_met_obs.dec_data, cams_met_obs.mag_data):

                    frame, time_rel, x, y, azim, alt, ra, dec, mag = entry

                    met_point = MeteorPointRMS(frame, time_rel, x, y, np.degrees(ra), np.degrees(dec), 
                        np.degrees(azim), np.degrees(alt), mag)

                    meteor_data.append(met_point)


                # Init the new meteor observation object
                met_obs = MeteorObsRMS(
                    station_code, 
                    jd2Date(cams_met_obs.jdt_ref, dt_obj=True, tzinfo=datetime.timezone.utc), 
                    pp,
                    meteor_data, 
                    rel_proc_path, 
                    ff_name=cams_met_obs.ff_name)

                # Skip bad observations
                if met_obs.bad_data:
                    continue

                # Add only unpaired observations
                if not self.observations_db.checkObsPaired(met_obs.station_code, met_obs.id):
                    # print(" ", station_code, met_obs.reference_dt, rel_proc_path)
                    added_count += 1
                    unpaired_met_obs_list.append(met_obs)

            log.info("  Added {:d} observations!".format(added_count))


        log.info("")
        log.info("  Finished loading unpaired observations!")

        return unpaired_met_obs_list
    

    def yearMonthDayDirInDtRange(self, dir_name):
        """ Given a directory name which is either YYYY, YYYYMM or YYYYMMDD, check if it is in the given 
            datetime range. 

        Arguments:
            dir_name: [str] Directory name which is either YYYY, YYYYMM or YYYYMMDD.

        Return:
            [bool] True if the directory is in the datetime range, False otherwise.
        """

        # If the date range is not given, then skip the directory
        if self.dt_range is None:
            return True
        
        # Check in which format the directory name is
        if len(dir_name) == 4:
            date_fmt = "%Y"

            # Check if the directory name starts with a year
            if not re.match("^\d{4}", dir_name):   # noqa: W605 
                return False

        elif len(dir_name) == 6:
            date_fmt = "%Y%m"

            # Check if the directory name starts with a year and month
            if not re.match("^\d{6}", dir_name): # noqa: W605 
                return False

        elif len(dir_name) == 8:
            date_fmt = "%Y%m%d"

            # Check if the directory name starts with a year, month and day
            if not re.match("^\d{8}", dir_name): # noqa: W605 
                return False

        else:
            return False
        
        
        # Make a datetime object from the directory name
        dt = datetime.datetime.strptime(dir_name, date_fmt).replace(tzinfo=datetime.timezone.utc)

        dt_beg, dt_end = self.dt_range


        # Check if the date time is in the time range
        if len(dir_name) >= 4:

            # Check if the year is in the range
            if (dt.year >= dt_beg.year) and (dt.year <= dt_end.year):

                # If the month is also given, check that it's within the range
                if len(dir_name) >= 6:
                    
                    # Construct test datetime objects with the first and last times within the given month
                    dt_beg_test = datetime.datetime(dt.year, dt.month, 1, tzinfo=datetime.timezone.utc)
                    dt_end_test = datetime.datetime(dt.year, dt.month, 1, tzinfo=datetime.timezone.utc) \
                        + datetime.timedelta(days=31)

                    # Check if the month is in the range
                    if (dt_end_test >= dt_beg) and (dt_beg_test <= dt_end):

                        # If the day is also given, check that it's within the range
                        if len(dir_name) >= 8:

                            # Construct test datetime objects with the first and last times within the given day
                            dt_beg_test = datetime.datetime(
                                dt.year, dt.month, dt.day, tzinfo=datetime.timezone.utc)
                            dt_end_test = datetime.datetime(
                                dt.year, dt.month, dt.day, tzinfo=datetime.timezone.utc) + datetime.timedelta(days=1)

                            # Check if the day is in the range
                            if (dt_end_test >= dt_beg) and (dt_beg_test <= dt_end):
                                return True

                            else:
                                return False

                        return True
                    
                    else:
                        return False

                return True
            
            else:
                return False
            

    def trajectoryFileInDtRange(self, file_name, dt_range=None):
        """ Check if the trajectory file is in the given datetime range. """

        if dt_range is None:
            dt_beg, dt_end = self.dt_range
        else:
            dt_beg, dt_end = dt_range


        # If the date range is not given, then skip the trajectory
        if dt_beg is None or dt_end is None:
            return True

        # Extract the datetime from the trajectory name
        date_str, time_str = file_name.split('_')[:2]

        # Make a datetime object
        dt = datetime.datetime.strptime(
            "_".join([date_str, time_str]), "%Y%m%d_%H%M%S").replace(tzinfo=datetime.timezone.utc)

        # Check if the date time is in the time range
        if (dt >= dt_beg) and (dt <= dt_end):
            return True

        else:
            return False


    def removeDeletedTrajectories(self):
        """ Purge the database of any trajectories that no longer exist on disk.
            These can arise because the monte-carlo stage may update the data. 
        """

        if not os.path.isdir(self.output_dir):
            return 
        if self.traj_db is None:
            return 
        
        log.info("  Removing deleted trajectories from: " + self.output_dir)
        if self.dt_range is not None:
            log.info("  Datetime range: {:s} - {:s}".format(
                self.dt_range[0].strftime("%Y-%m-%d %H:%M:%S"), 
                self.dt_range[1].strftime("%Y-%m-%d %H:%M:%S")))
            
        jdt_start = datetime2JD(self.dt_range[0]) 
        jdt_end = datetime2JD(self.dt_range[1])

        self.traj_db.removeDeletedTrajectories(self.output_dir, jdt_start, jdt_end)

        return 


    def loadComputedTrajectories(self, dt_range=None):
        """ Load already estimated trajectories from disk within a date range. 

        Arguments:
            dt_range: [datetime, datetime] range of dates to load data for
        """
        traj_dir_path = os.path.join(self.output_dir, OUTPUT_TRAJ_DIR)
        # defend against the case where there are no existing trajectories and traj_dir_path doesn't exist
        if not os.path.isdir(traj_dir_path):
            return

        if self.traj_db is None:
            return 
        
        if dt_range is None:
            dt_beg, dt_end = self.dt_range
        else:
            dt_beg, dt_end = dt_range

        log.info("  Loading found trajectories from: " + traj_dir_path)
        if self.dt_range is not None:
            log.info("  Datetime range: {:s} - {:s}".format(
                dt_beg.strftime("%Y-%m-%d %H:%M:%S"), 
                dt_end.strftime("%Y-%m-%d %H:%M:%S")))

        counter = 0

        # Construct a list of all ddirectory paths to visit. The trajectory directories are sorted in 
        # YYYY/YYYYMM/YYYYMMDD, so visit them in that order to check if they are in the datetime range
        dir_paths = []

        #iterate over the days in the range
        jdt_beg = int(np.floor(datetime2JD(dt_beg)))
        jdt_end = int(np.ceil(datetime2JD(dt_end)))

        yyyy = 0
        mm = 0
        dd = 0
        start_time = datetime.datetime.now()
        for jdt in range(jdt_beg, jdt_end + 1):

            curr_dt = jd2Date(jdt, dt_obj=True)
            if curr_dt.year != yyyy:
                yyyy = curr_dt.year
                #log.info("- year    " + str(yyyy))

            if curr_dt.month != mm:
                mm = curr_dt.month
                yyyymm = f'{yyyy}{mm:02d}'
                #log.info("  - month " + str(yyyymm))

            if curr_dt.day != dd:
                dd = curr_dt.day
                yyyymmdd = f'{yyyy}{mm:02d}{dd:02d}'
                #log.info("    - day " + str(yyyymmdd))

            yyyymmdd_dir_path = os.path.join(traj_dir_path, f'{yyyy}', f'{yyyymm}', f'{yyyymmdd}')

            # catch for folder not existing for some reason
            if os.path.isdir(yyyymmdd_dir_path):

                for traj_dir in sorted(os.listdir(yyyymmdd_dir_path)):

                    # Add the directory to the list of directories to visit
                    full_traj_dir = os.path.join(yyyymmdd_dir_path, traj_dir)
                    if os.path.isdir(full_traj_dir) and (full_traj_dir not in dir_paths):

                        for file_name in glob.glob1(full_traj_dir, '*_trajectory.pickle'):

                            if self.trajectoryFileInDtRange(file_name, dt_range=dt_range):

                                self.traj_db.addTrajectory(TrajectoryReduced(os.path.join(full_traj_dir, file_name)))

                                # Print every 1000th trajectory
                                if counter % 1000 == 0:
                                    log.info(f"  Loaded {counter:6d} trajectories")
                                counter += 1

                        dir_paths.append(full_traj_dir)

        dur = (datetime.datetime.now() - start_time).total_seconds()
        log.info(f"  Loaded {counter:6d} trajectories in {dur:.0f} seconds")
        


    def getComputedTrajectories(self, jd_beg, jd_end):
        """ Returns a list of computed trajectories between the Julian dates.
        """
        json_dicts = self.traj_db.getTrajectories(self.output_dir, jd_beg, jd_end)
        trajs = [TrajectoryReduced(None, json_dict=j) for j in json_dicts]
        return trajs
               

    def getPlatepar(self, met_obs):
        """ Return the platepar of the meteor observation. """

        return met_obs.platepar



    def getUnpairedObservations(self):
        """ Returns a list of unpaired meteor observations. """

        return self.unpaired_observations


    def countryFilter(self, station_code1, station_code2):
        """ Only pair observations if they are in proximity to a given country. """

        # Check that both stations are in the same country group
        for group in self.country_groups:
            if station_code1[:2] in group:
                if station_code2[:2] in group:
                    return True
                else:
                    return False


        # If a given country is not in any of the groups, allow it to be paired
        return True


    def findTimePairs(self, met_obs, unpaired_observations, max_toffset, verbose=False):
        """ Finds pairs in time between the given meteor observations and all other observations from 
            different stations. 

        Arguments:
            met_obs: [MeteorObsRMS] Object containing a meteor observation.
            unpaired_observations: [list] A list of MeteorObsRMS objects which will be paired in time with
                the given object.
            max_toffset: [float] Maximum offset in time (seconds) for pairing.

        Return:
            [list] A list of MeteorObsRMS instances with are offten in time less than max_toffset from 
                met_obs.
        """

        found_pairs = []

        # Go through all meteors from other stations
        for met_obs2 in unpaired_observations:

            if self.observations_db.checkObsPaired(met_obs2.station_code, met_obs2.id, verbose=verbose):
                continue

            # Take only observations from different stations
            if met_obs.station_code == met_obs2.station_code:
                continue

            # Check that the stations are in the same region / group of countres
            if not self.countryFilter(met_obs.station_code, met_obs2.station_code):
                continue

            # Take observations which are within the given time window
            if abs((met_obs.mean_dt - met_obs2.mean_dt).total_seconds()) <= max_toffset:
                found_pairs.append(met_obs2)


        return found_pairs


    def getTrajTimePairs(self, traj_reduced, unpaired_observations, max_toffset):
        """ Find unpaired observations which are close in time to the given trajectory. """

        found_traj_obs_pairs = []

        # Compute the middle time of the trajectory as reference time
        traj_mid_dt = jd2Date((traj_reduced.rbeg_jd + traj_reduced.rend_jd)/2, dt_obj=True, 
                              tzinfo=datetime.timezone.utc)

        # Go through all unpaired observations
        for met_obs in unpaired_observations:

            # Check that the stations are in the same region / group of countres
            if not self.countryFilter(met_obs.station_code, (traj_reduced.participating_stations + traj_reduced.ignored_stations)[0]):
                continue

            # Skip all stations that are already participating in the trajectory solution
            if (met_obs.station_code in traj_reduced.participating_stations) or (met_obs.station_code in traj_reduced.ignored_stations):
                continue


            # Take observations which are within the given time window from the trajectory
            if abs((met_obs.mean_dt - traj_mid_dt).total_seconds()) <= max_toffset:
                found_traj_obs_pairs.append(met_obs)


        return found_traj_obs_pairs


    def generateTrajOutputDirectoryPath(self, traj, make_dirs=False):
        """ Generate a path to the trajectory output directory. 
        
        Keyword arguments:
            make_dirs: [bool] Make the tree of output directories. False by default.
        """

        # Generate a list of station codes
        if isinstance(traj, TrajectoryReduced):
            # If the reducted trajectory object is given
            station_list = traj.participating_stations

        else:
            # If the full trajectory object is given
            station_list = [obs.station_id for obs in traj.observations if obs.ignore_station is False]


        # Datetime of the reference trajectory time
        dt = jd2Date(traj.jdt_ref, dt_obj=True, tzinfo=datetime.timezone.utc)


        # Year directory
        year_dir = dt.strftime("%Y")

        # Month directory
        month_dir = dt.strftime("%Y%m")

        # Date directory
        date_dir = dt.strftime("%Y%m%d")

        # Name of the trajectory directory
        # sort the list of country codes otherwise we can end up with duplicate trajectories
        ctry_list = list(set([stat_id[:2] for stat_id in station_list]))
        ctry_list.sort()
        traj_dir = dt.strftime("%Y%m%d_%H%M%S.%f")[:-3] + "_" + "_".join(ctry_list)


        # Path to the year directory
        out_path = os.path.join(self.output_dir, OUTPUT_TRAJ_DIR, year_dir, month_dir, date_dir, traj_dir)
        if make_dirs:
            mkdirP(out_path)

        return out_path


    def saveTrajectoryResults(self, traj, save_plots, verbose=False):
        """ Save trajectory results to the disk. """


        # Generate the name for the output directory (add list of country codes at the end)
        output_dir = self.generateTrajOutputDirectoryPath(traj, make_dirs=True)

        # Save the report
        traj.saveReport(output_dir, traj.file_name + '_report.txt', uncertainties=traj.uncertainties, 
            verbose=False)

        # Add the trajectory foldername to the saved traj. We may need this later, for example
        # if additional observations are found then the refdt or country list may change quite a bit
        traj.longname = os.path.split(output_dir)[-1]

        if self.mc_mode & MCMODE_PHASE1:
            # The MC phase may change the refdt so save a copy of the the original name.
            traj.pre_mc_longname = traj.longname

            # keep track of when we originally saved a solution, so we can check how long its taking
            traj.save_date = datetime.datetime.now().replace(tzinfo=datetime.timezone.utc)

        # Save the picked trajectory structure
        savePickle(traj, output_dir, traj.file_name + '_trajectory.pickle')
        log.info(f'saved {traj.traj_id} to {output_dir}')

        if self.mc_mode & MCMODE_PHASE1 and not self.mc_mode & MCMODE_PHASE2:
            self.savePhase1Trajectory(traj, traj.pre_mc_longname + '_trajectory.pickle', verbose=verbose)
            
        elif self.mc_mode & MCMODE_PHASE2:
            # the MC phase may alter the trajectory details and if later on 
            # we're including additional observations we need to use the most recent version of the trajectory
            savePickle(traj, os.path.join(self.phase1_dir, 'processed'), traj.pre_mc_longname + '_trajectory.pickle')

        # Save the plots
        if save_plots:
            traj.save_results = True
            try:
                traj.savePlots(output_dir, traj.file_name, show_plots=False)
            except:
                pass
            traj.save_results = False

    def addTrajectory(self, traj, failed_jdt_ref=None, verbose=False):
        """ Add the resulting trajectory to the database. 

        Arguments:
            traj: [Trajectory object]
            failed_jdt_ref: [float] Reference Julian date of the failed trajectory. None by default.
        """

        if self.traj_db is None:
            return 
        # Set the correct output path
        traj.output_dir = self.generateTrajOutputDirectoryPath(traj)

        # Convert the full trajectory object into the reduced trajectory object
        traj_reduced = TrajectoryReduced(None, traj_obj=traj)

        # If the trajectory failed, keep track of the original reference Julian date, as it might have been
        #   changed during trajectory estimation
        if failed_jdt_ref is not None:
            traj_reduced.jdt_ref = failed_jdt_ref

        self.traj_db.addTrajectory(traj_reduced, failed=(failed_jdt_ref is not None), verbose=verbose)



    def removeTrajectory(self, traj_reduced, remove_phase1=False):
        """ Remove the trajectory from the data base and disk. """

        # in mcmode 2 the database isn't loaded but we still need to delete updated trajectories
        if self.mc_mode & MCMODE_PHASE2: 
            if os.path.isfile(traj_reduced.traj_file_path):
                traj_dir = os.path.dirname(traj_reduced.traj_file_path)
                shutil.rmtree(traj_dir, ignore_errors=True)
            elif hasattr(traj_reduced, 'pre_mc_longname'):
                traj_dir = os.path.dirname(traj_reduced.traj_file_path)
                base_dir = os.path.split(traj_dir)[0]
                traj_dir = os.path.join(base_dir, traj_reduced.pre_mc_longname)
                if os.path.isdir(traj_dir):
                    shutil.rmtree(traj_dir, ignore_errors=True)
                else:
                    log.warning(f'unable to find {traj_dir}')
            else:
                log.warning(f'unable to find {traj_reduced.traj_file_path}')

            # remove the processed pickle now we're done with it
            self.cleanupPhase2TempPickle(traj_reduced, True)
            return
        if self.mcmode & MCMODE_PHASE1 and remove_phase1:
            # remove any solution from the phase1 folder
            phase1_traj = os.path.join(self.phase1_dir, os.path.basename(traj_reduced.traj_file_path))
            if os.path.isfile(phase1_traj):
                try:
                    os.remove(phase1_traj)
                except Exception: 
                    pass

        self.traj_db.removeTrajectory(traj_reduced)


    def cleanupPhase2TempPickle(self, traj, success=False):
        """
        At the start of phase 2 monte-carlo sim calculation, the phase1 pickles are renamed to indicate they're being processed.
        Once each one is processed (fail or succeed) we need to clean up the file. If the MC step failed, we still want to keep
        the pickle, because we might later on get new data and it might become solvable. Otherwise, we can just delete the file 
        since the MC solver will have saved an updated one already.
        """
        if not self.mc_mode & MCMODE_PHASE2:
            return 
        fldr_name = os.path.split(self.generateTrajOutputDirectoryPath(traj, make_dirs=False))[-1] 
        pick = os.path.join(self.phase1_dir, fldr_name + '_trajectory.pickle_processing')
        if os.path.isfile(pick):
            os.remove(pick)
        else:
            log.warning(f'unable to find _processing file {pick}')
        if not success:
            # save the pickle in case we get new data later and can solve it
            savePickle(traj, os.path.join(self.phase1_dir, 'processed'), fldr_name + '_trajectory.pickle')
        return 

    def excludeAlreadyFailedCandidates(self, matched_observations, remaining_unpaired, verbose=False):

        # go through the candidates and check if they correspond to already-failed
        candidate_trajectories=[]
        for cand in matched_observations:
            ref_dt = min([met_obs.reference_dt for _, met_obs, _ in cand])
            jdt_ref = datetime2JD(ref_dt)
            traj = Trajectory(jdt_ref, verbose=False)

            # Feed the observations into the trajectory solver
            for obs_temp, met_obs, _ in cand:

                # Normalize the observations to the reference Julian date
                jdt_ref_curr = datetime2JD(met_obs.reference_dt)
                obs_temp.time_data += (jdt_ref_curr - jdt_ref)*86400

                traj.infillWithObs(obs_temp)

            ### Recompute the reference JD and all times so that the first time starts at 0 ###

            # Determine the first relative time from reference JD
            t0 = min([obs.time_data[0] for obs in traj.observations if (not obs.ignore_station) 
                or (not np.all(obs.ignore_list))])

            # If the first time is not 0, normalize times so that the earliest time is 0
            if t0 != 0.0:
                # Recompute the reference JD to corresponds with t0
                traj.jdt_ref = traj.jdt_ref + t0/86400.0

            if self.checkTrajIfFailed(traj):
                log.info(f'Trajectory at {jd2Date(traj.jdt_ref,dt_obj=True).isoformat()} already failed, skipping')
                for _, met_obs_temp, _ in cand:
                    self.observations_db.unpairObs(met_obs_temp.station_code, met_obs_temp.id, met_obs_temp.mean_dt, verbose=verbose)
                    remaining_unpaired -= 1
            else:
                candidate_trajectories.append(cand)
    
        return candidate_trajectories, max(0,remaining_unpaired)

    def checkTrajIfFailed(self, traj):
        """ Check if the given trajectory has been computed with the same observations and has failed to be
            computed before.

        """

        if self.traj_db is None:
            return
        traj_reduced = TrajectoryReduced(None, traj_obj=traj)
        return self.traj_db.checkTrajIfFailed(traj_reduced)



    def loadFullTraj(self, traj_reduced):
        """ Load the full trajectory object. 
    
        Arguments:
            traj_reduced: [TrajectoryReduced object]

        Return:
            traj: [Trajectory object] or [None] if file not found
        """

        # Generate the path to the output directory
        output_dir = self.generateTrajOutputDirectoryPath(traj_reduced)
        phase1_name = os.path.split(output_dir)[-1] + '_trajectory.pickle'

        # Get the file name
        file_name = os.path.basename(traj_reduced.traj_file_path)

        # Try loading a full trajectory, first from the standard output area, then from the phase-1 folders
        full_traj_loc = os.path.join(output_dir, file_name)
        if not os.path.isfile(full_traj_loc):
            full_traj_loc = os.path.join(self.phase1_dir, phase1_name)
        if not os.path.isfile(full_traj_loc):
            full_traj_loc = os.path.join(self.phase1_dir, 'processed', phase1_name)
        if not os.path.isfile(full_traj_loc):
            log.info(f'File {full_traj_loc} not found!')
            return None

        try:
            traj = loadPickle(*os.path.split(full_traj_loc))

            # Check if the traj object as fixed time offsets
            if not hasattr(traj, 'fixed_time_offsets'):
                traj.fixed_time_offsets = {}

            return traj

        except FileNotFoundError:
            log.info(f'File {full_traj_loc} not found!')
            
            return None

    def loadPhase1Trajectories(self, max_trajs=1000):
        """
        Load trajectories calculated by the intersecting-planes phase 1. These trajectories
        do not include uncertainties which are calculated in the Monte-Carlo phase 2

        keyword arguments:
        maxtrajs: [int] maximum number of trajectories to load in each pass, to avoid taking too long per pass.


        returns:
        dt_beg, dt_end: [datetime] The earliest and latest date/time of the loaded trajectories. Used later to set the 
                                    number of time buckets to process data in. 

        """
        pickles = glob.glob1(self.phase1_dir, "*_trajectory.pickle")
        pickles.sort()
        pickles = pickles[:max_trajs]
        self.phase1Trajectories = []
        if len(pickles) == 0:
            return None, None
        dt_beg = datetime.datetime.strptime(pickles[0][:15], '%Y%m%d_%H%M%S').replace(tzinfo=datetime.timezone.utc)
        dt_end = datetime.datetime.strptime(pickles[-1][:15], '%Y%m%d_%H%M%S').replace(tzinfo=datetime.timezone.utc)
        for pick in pickles:
            # Try loading a full trajectory
            try:
                traj = loadPickle(self.phase1_dir, pick)

                traj_dir = self.generateTrajOutputDirectoryPath(traj, make_dirs=False)
                # Add the filepath if not present so we can remove updated trajectories
                if not hasattr(traj, 'traj_file_path'):
                    # stored filename includes the millisecs and countries, to help make it unique
                    # so we need to chop that off again to set up the true pickle name
                    real_pick_name = pick[:15] + '_trajectory.pickle'
                    traj.traj_file_path = os.path.join(traj_dir, real_pick_name)

                if not hasattr(traj, 'longname'):
                    traj.longname = os.path.split(traj_dir)[-1]

                if not hasattr(traj, 'pre_mc_longname'):
                    traj.pre_mc_longname = os.path.split(traj_dir)[-1]

                # Check if the traj object as fixed time offsets
                if not hasattr(traj, 'fixed_time_offsets'):
                    traj.fixed_time_offsets = {}

                # now we've loaded the phase 1 solution, move it to prevent accidental reprocessing
                procfile = os.path.join(self.phase1_dir, pick + '_processing')
                if os.path.isfile(procfile):
                    os.remove(procfile)
                os.rename(os.path.join(self.phase1_dir, pick), procfile)

                self.phase1Trajectories.append(traj)
                log.info(f'loaded {traj.traj_id}')
            except Exception:
                # if the file couldn't be read, then skip it for now - we'll get it in the next pass
                log.info(f'File {pick} skipped for now')
        return dt_beg, dt_end
   
    def moveUploadedData(self, verbose=False):
        """
        Used in 'master' mode: this moves uploaded data to the target locations on the server
        and merges in the databases
        """
        for node in self.RemoteDatahandler.nodes:
            if node.nodename == 'localhost' or self.observations_db is None or self.traj_db is None:
                continue

            # if the remote node upload path doesn't exist skip it
            if not os.path.isdir(os.path.join(node.dirpath,'files')):
                continue

            # merge the databases
            for obsdb_path in glob.glob(os.path.join(node.dirpath,'files','observations*.db')):
                self.observations_db.mergeObsDatabase(obsdb_path)
                os.remove(obsdb_path)

            
            for trajdb_path in glob.glob(os.path.join(node.dirpath,'files','trajectories*.db')):
                self.traj_db.mergeTrajDatabase(trajdb_path)
                os.remove(trajdb_path)

            i = 0
            remote_trajdir = os.path.join(node.dirpath, 'files', 'trajectories')
            if os.path.isdir(remote_trajdir):
                for i,traj in enumerate(os.listdir(remote_trajdir)):
                    if os.path.isdir(os.path.join(remote_trajdir, traj)):
                        targ_path = os.path.join(self.output_dir, 'trajectories', traj[:4], traj[:6], traj[:8], traj)
                        src_path = os.path.join(node.dirpath,'files', 'trajectories', traj)
                        for src_name in os.listdir(src_path):
                            src_name = os.path.join(src_path, src_name)
                            if not os.path.isfile(src_name):
                                log.info(f'{src_name} missing')
                            else:
                                os.makedirs(targ_path, exist_ok=True)
                                shutil.copy(src_name, targ_path)
                        shutil.rmtree(src_path,ignore_errors=True)
            if i > 0:
                log.info(f'moved {i+1} trajectories')

            # if the node was in mode 1 then move any uploaded phase1 solutions
            remote_ph1dir = os.path.join(node.dirpath, 'files', 'phase1')
            if os.path.isdir(remote_ph1dir) and node.mode==1:
                if not os.path.isdir(self.phase1_dir):
                    os.makedirs(self.phase1_dir, exist_ok=True)
                i = 0
                for i, fil in enumerate([x for x in os.listdir(remote_ph1dir) if '.pickle' in x]):
                    full_name = os.path.join(remote_ph1dir, fil)
                    shutil.copy(full_name, self.phase1_dir)
                    os.remove(full_name)

                if i > 0:
                    log.info(f'moved {i+1} phase 1 files from {node.nodename}')
            
        return True

    def getRemoteData(self, verbose=False):
        """
        Used in 'child' mode: this downloads data from the master for local processing. 
        """
        if not self.RemoteDatahandler:
            log.info('remote data handler not initialised')
            return False
        
        # collect candidates or phase1 solutions from the master node
        if self.mc_mode == MCMODE_PHASE1 or self.mc_mode == MCMODE_BOTH:
            status = self.RemoteDatahandler.collectRemoteData('candidates', self.output_dir, verbose=verbose)
        elif mcmode == MCMODE_PHASE2:
            status = self.RemoteDatahandler.collectRemoteData('phase1', self.output_dir, verbose=verbose)
        else:
            status = False
        return status
    
    def saveCandidates(self, candidate_trajectories, verbose=False):
        for matched_observations in candidate_trajectories:
            ref_dt = min([met_obs.reference_dt for _, met_obs, _ in matched_observations])
            ctries = '_'.join(list(set([met_obs.station_code[:2] for _, met_obs, _ in matched_observations])))
            picklename = f'{ref_dt.timestamp():.6f}_{ctries}.pickle'

            # this function can also save a candidate
            self.savePhase1Trajectory(matched_observations, picklename, 'candidates', verbose=verbose)

        log.info("-----------------------")
        log.info(f'Saved {len(candidate_trajectories)} candidates')
        log.info("-----------------------")

    def savePhase1Trajectory(self, traj, file_name, savetype='phase1', verbose=False):
        """
        in mcmode MCMODE_PHASE1 or MCMODE_SIMPLE , save the candidates or phase 1 trajectories
        and distribute as appropriate
    
        """
        if savetype == 'phase1':
            save_dir = self.phase1_dir
            required_mode = 2
        else:
            save_dir = self.candidate_dir
            required_mode = 1

        if self.RemoteDatahandler and self.RemoteDatahandler.mode == 'master':

            # Select a random bucket, check its not already full, and then save the pickle there.
            # Make sure to break out once all buckets have been tested
            # Fallback/default is to use the local phase_1 dir. 
            tested_buckets = []
            bucket_num = -1
            bucket_list = self.RemoteDatahandler.nodes
            bucket_list[-1].dirpath = save_dir

            while bucket_num not in tested_buckets:
                bucket_num = secrets.randbelow(len(bucket_list))
                bucket = bucket_list[bucket_num]
                # if the child isn't in mc mode, skip it
                if bucket.mode != required_mode and bucket.mode != -1:
                    tested_buckets.append(bucket_num)
                    continue
                if bucket.nodename != 'localhost':
                    tmp_save_dir = os.path.join(bucket.dirpath, 'files', savetype)
                else:
                    tmp_save_dir = save_dir
                os.makedirs(tmp_save_dir, exist_ok=True)
                if os.path.isfile(os.path.join(bucket.dirpath, 'files', 'stop')):
                    tested_buckets.append(bucket_num)
                    continue
                if bucket.capacity < 0 or len(glob.glob(os.path.join(tmp_save_dir, '*.pickle'))) < bucket.capacity:
                    if bucket.nodename != 'localhost':
                        save_dir = tmp_save_dir
                    break
                tested_buckets.append(bucket_num)
                
        if verbose:
            log.info(f'saving {file_name} to {save_dir}')
        savePickle(traj, save_dir, file_name)



if __name__ == "__main__":

    # Set matplotlib for headless running
    import matplotlib
    matplotlib.use('Agg')


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="""Automatically compute trajectories from RMS data in the given directory. 
The directory structure needs to be the following, for example:
    ./ # root directory
        /HR0001/
            /HR0001_20190707_192835_241084_detected
                ./FTPdetectinfo_HR0001_20190707_192835_241084.txt
                ./platepars_all_recalibrated.json
        /HR0004/
            ./FTPdetectinfo_HR0004_20190707_193044_498581.txt
            ./platepars_all_recalibrated.json
        /...

In essence, the root directory should contain directories of stations (station codes need to be exact), and these directories should
contain data folders. Data folders should have FTPdetectinfo files together with platepar files.""",
        formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('dir_path', type=str, help='Path to the root data directory. Trajectory helper files will be stored here as well.')

    arg_parser.add_argument('-t', '--maxtoffset', metavar='MAX_TOFFSET', 
        help='Maximum time offset between the stations. Default is 5 seconds.', type=float, default=10.0)

    arg_parser.add_argument('-s', '--maxstationdist', metavar='MAX_STATION_DIST', 
        help='Maximum distance (km) between stations of paired meteors. Default is 600 km.', type=float, 
        default=600.0)

    arg_parser.add_argument('-m', '--minerr', metavar='MIN_ARCSEC_ERR', 
        help="Minimum error in arc seconds below which the station won't be rejected. 30 arcsec by default.", 
        type=float)

    arg_parser.add_argument('-M', '--maxerr', metavar='MAX_ARCSEC_ERR', 
        help="Maximum error in arc seconds, above which the station will be rejected. 180 arcsec by default.", 
        type=float)

    arg_parser.add_argument('-v', '--maxveldiff', metavar='MAX_VEL_DIFF', 
        help='Maximum difference in percent between velocities between two stations. Default is 25 percent.', 
        type=float, default=25.0)

    arg_parser.add_argument('-p', '--velpart', metavar='VELOCITY_PART', 
        help='Fixed part from the beginning of the meteor on which the initial velocity estimation using the sliding fit will start. Default is 0.4 (40 percent), but for noisier data this should be bumped up to 0.5.', 
        type=float, default=0.40)

    arg_parser.add_argument('-d', '--disablemc', 
        help='Disable Monte Carlo.', action="store_true")

    arg_parser.add_argument('-u', '--uncerttime', 
        help="Compute uncertainties by culling solutions with worse value of the time fit than the LoS solution. This may increase the computation time considerably.", 
        action="store_true")

    arg_parser.add_argument('-l', '--saveplots', 
        help='Save plots to disk.', action="store_true")

    arg_parser.add_argument('-r', '--timerange', metavar='TIME_RANGE',
        help="""Only compute the trajectories in the given range of time. The time range should be given in the format: "(YYYYMMDD-HHMMSS,YYYYMMDD-HHMMSS)".""", 
            type=str)

    arg_parser.add_argument('-a', '--auto', metavar='PREV_DAYS', type=float, default=None, const=5.0, 
        nargs='?', 
        help="""Run continously taking the data in the last PREV_DAYS to compute the new trajectories and update the old ones. The default time range is 5 days.""")

    arg_parser.add_argument("--cpucores", type=int, default=-1,
        help="Number of CPU codes to use for computation. -1 to use all cores minus one (default).",)

    arg_parser.add_argument('-o', '--enableOSM', 
        help="Enable OSM based groung plots. Internet connection required.", action="store_true")     

    arg_parser.add_argument("--dbdir", type=str, default=None,
        help="Path to the directory where the trajectory database file will be stored. If not given, the database will be stored in the data directory.")

    arg_parser.add_argument("--outdir", type=str, default=None,
        help="Path to the directory where the trajectory output files will be stored. If not given, the output will be stored in the data directory.")
        
    arg_parser.add_argument("--logdir", type=str, default=None,
        help="Path to the directory where the log files will be stored. If not given, the logs will be stored in the output directory.")
        
    arg_parser.add_argument('-x', '--maxstations', type=int, default=15,
        help="Use best N stations in the solution (default is use 15 stations).")

    arg_parser.add_argument('--mcmode', '--mcmode', type=int, default=0,
        help="Operation mode - see readme. For standalone solving either don't set this or set it to 0")

    arg_parser.add_argument('--archivemonths', '--archivemonths', type=int, default=3,
        help="Months back to archive old data. Default 3. Zero means don't archive (useful in testing).")

    arg_parser.add_argument('--maxtrajs', '--maxtrajs', type=int, default=None,
        help="Max number of trajectories to reload in each pass when doing the Monte-Carlo phase")
    
    arg_parser.add_argument('--autofreq', '--autofreq', type=int, default=360,
        help="Minutes to wait between runs in auto-mode")
    
    arg_parser.add_argument('--verbose', '--verbose', help='Verbose logging.', default=False, action="store_true")

    arg_parser.add_argument('--addlogsuffix', '--addlogsuffix', help='add a suffix to the log to show what stage it is.', default=False, action="store_true")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    ############################

    db_dir = cml_args.dbdir
    if db_dir is None:
        db_dir = cml_args.dir_path 

    # signal handler created inline here as it needs access to db_dir
    def signal_handler(sig, frame):
        signal.signal(sig, signal.SIG_IGN) # ignore additional signals
        log.info('======================================')
        log.info('CTRL-C pressed, exiting gracefully....')
        log.info('======================================')
        remote_cfg = os.path.join(db_dir, 'wmpl_remote.cfg')
        if os.path.isfile(remote_cfg):
            rdh = RemoteDataHandler(remote_cfg)
            if rdh and rdh.mode == 'child':
                rdh.setStopFlag()
        log.info('DONE')
        log.info('======================================')
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    ### Init logging - roll over every day ###

    # Find the log directory
    log_dir = cml_args.logdir
    if log_dir is None:
        log_dir = cml_args.outdir
    if log_dir is None:
        log_dir = cml_args.dir_path

    # Create a log dir if it doesn't exist
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    # Init the logger
    #log = logging.getLogger("traj_correlator")
    log.setLevel(logging.DEBUG)

    # Init the log formatter
    log_formatter = logging.Formatter(
        fmt='%(asctime)s-%(levelname)-5s-%(module)-15s:%(lineno)-5d- %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S')

    # Init the file handler
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"correlate_rms_{timestamp}.log")
    if cml_args.addlogsuffix:
        modestrs = {4:'cands', 1:'simple', 2:'mcphase', 5:'candsimple', 3:'simplemc',7:'full',0:'full'}
        if cml_args.mcmode in modestrs.keys():
            modestr = modestrs[cml_args.mcmode]
            log_file = os.path.join(log_dir, f"correlate_rms_{timestamp}_{modestr}.log")
       
    file_handler = logging.handlers.TimedRotatingFileHandler(log_file, when="midnight", backupCount=7)
    file_handler.setFormatter(log_formatter)
    log.addHandler(file_handler)

    # Init the console handler (i.e. print to console)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    log.addHandler(console_handler)

    ###
    if cml_args.autofreq is not None:
        AUTO_RUN_FREQUENCY = cml_args.autofreq/60

    if cml_args.auto is None:
        log.info("Running trajectory estimation once!")
    else:
        log.info("Auto running trajectory estimation every {:.1f} hours using the last {:.1f} days of data...".format(AUTO_RUN_FREQUENCY, cml_args.auto))

    # Set max stations to use in a solution, minimum 2.
    # The best N will be chosen. -1 means use all. 
    if cml_args.maxstations is not None:
        if cml_args.maxstations == -1:
            # Set to large number, so we can easily test later
            max_stations = 9999
            log.info('Solutions will use all available stations.')
        else:
            max_stations = max(2, cml_args.maxstations)
            log.info('Solutions will use the best {} stations.'.format(max_stations))
    else:
        max_stations = 15
        log.info('Solutions will use the best {} stations.'.format(max_stations))

    # Init trajectory constraints
    trajectory_constraints = TrajectoryConstraints()
    trajectory_constraints.max_toffset = cml_args.maxtoffset
    trajectory_constraints.max_station_dist = cml_args.maxstationdist
    trajectory_constraints.max_vel_percent_diff = cml_args.maxveldiff
    trajectory_constraints.run_mc = not cml_args.disablemc
    trajectory_constraints.save_plots = cml_args.saveplots
    trajectory_constraints.geometric_uncert = not cml_args.uncerttime
    trajectory_constraints.max_stations = max_stations

    if cml_args.minerr is not None:
        trajectory_constraints.min_arcsec_err = cml_args.minerr

    if cml_args.maxerr is not None:
        trajectory_constraints.max_arcsec_err = cml_args.maxerr

    # mcmode values
    # mcmode = 1 -> load candidates and do simple solutions
    # mcmode = 2 -> load simple solns and do MC solutions
    # mcmode = 4 -> find candidates only
    # mcmode = 7 -> do everything
    # mcmode = 0 -> same as mode 7
    # bitwise combinations are permissioble so:
    #   4+1 will find candidates and then run simple solutions to populate "phase1"
    #   1+2 will load candidates from "candidates" and solve them completely
    
    mcmode = MCMODE_ALL if cml_args.mcmode == 0 else cml_args.mcmode
    
    # set the maximum number of trajectories to reprocess when doing the MC uncertainties
    # set a default of 10 for remote processing and 1000 for local processing
    max_trajs = 1000
    if cml_args.maxtrajs is not None:
        max_trajs = int(cml_args.maxtrajs)
        
    if mcmode == MCMODE_PHASE2:
        log.info(f'Reloading at most {max_trajs} phase1 trajectories.')

    # Set the number of CPU cores
    cpu_cores = cml_args.cpucores
    if (cpu_cores < 1) or (cpu_cores > multiprocessing.cpu_count()):
        cpu_cores = multiprocessing.cpu_count()
    trajectory_constraints.mc_cores = cpu_cores
    log.info("Running using {:d} CPU cores.".format(cpu_cores))

    if mcmode == MCMODE_CANDS:
        log.info('Saving Candidates only')
    elif mcmode == MCMODE_PHASE1:
        log.info('Loading Candidates if needed')
    elif mcmode == MCMODE_ALL:
        log.info('Full processing mode')

    # Run processing. If the auto run more is not on, the loop will break after one run
    previous_start_time = None
    while True: 

        # Clock for measuring script time
        t1 = datetime.datetime.now(datetime.timezone.utc)

        # If auto run is enabled, compute the time range to use
        event_time_range = None
        if cml_args.auto is not None:

            # Compute first date and time to use for auto run
            dt_beg = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=cml_args.auto)

            # If the beginning time is later than the beginning of the previous run, use the beginning of the
            # previous run minus two days as the beginning time
            if previous_start_time is not None:
                if dt_beg > previous_start_time:
                    dt_beg = previous_start_time - datetime.timedelta(days=2)


            # Use now as the upper time limit
            dt_end = datetime.datetime.now(datetime.timezone.utc)

            event_time_range = [dt_beg, dt_end]


        # Otherwise check if the time range is given
        else:

            # If the time range to use is given, use it
            if cml_args.timerange is not None:

                # Extract time range
                time_beg, time_end = cml_args.timerange.strip("(").strip(")").split(",")
                dt_beg = datetime.datetime.strptime(
                    time_beg, "%Y%m%d-%H%M%S").replace(tzinfo=datetime.timezone.utc)
                
                dt_end = datetime.datetime.strptime(
                    time_end, "%Y%m%d-%H%M%S").replace(tzinfo=datetime.timezone.utc)

                log.info("Custom time range:")
                log.info("    BEG: {:s}".format(str(dt_beg)))
                log.info("    END: {:s}".format(str(dt_end)))

                event_time_range = [dt_beg, dt_end]
            else:
                # set the timerange to cover all possible dates
                dt_beg = datetime.datetime(2000,1,1,0,0,0).replace(tzinfo=datetime.timezone.utc)
                dt_end = datetime.datetime.now().replace(tzinfo=datetime.timezone.utc)
                event_time_range = [dt_beg, dt_end]

        # Init the data handle
        dh = RMSDataHandle(
            cml_args.dir_path, dt_range=event_time_range, 
            db_dir=cml_args.dbdir, output_dir=cml_args.outdir,
            mcmode=mcmode, max_trajs=max_trajs, verbose=cml_args.verbose, archivemonths=cml_args.archivemonths)
        
        # If there is nothing to process and we're in Candidate mode, stop
        if not dh.processing_list and (mcmode & MCMODE_CANDS):
            log.info("")
            log.info("Nothing to process!")
            log.info("Probably everything is already processed.")
            if cml_args.auto is None:
                break
        else:

            ### GENERATE DAILY TIME BINS ###

            if mcmode != MCMODE_PHASE2:
                # Find the range of datetimes of all folders (take only those after the year 2000)
                proc_dir_dts = [entry[3] for entry in dh.processing_list if entry[3] is not None]
                proc_dir_dts = [dt for dt in proc_dir_dts if dt > datetime.datetime(2000, 1, 1, 0, 0, 0, 
                                                                                    tzinfo=datetime.timezone.utc)]

                # Reject all folders not within the time range of interest +/- 1 day, to reduce the amount of data to be loaded
                if event_time_range is not None:

                    dt_beg, dt_end = event_time_range

                    proc_dir_dts = [dt for dt in proc_dir_dts 
                        if (dt >= dt_beg - datetime.timedelta(days=1)) and (dt <= dt_end + datetime.timedelta(days=1))]
                    
                    # to avoid excluding all possible dates
                    if proc_dir_dts == []: 
                        proc_dir_dts=[dt_beg - datetime.timedelta(days=1), dt_end + datetime.timedelta(days=1)]

                # Determine the limits of data
                proc_dir_dt_beg = min(proc_dir_dts)
                proc_dir_dt_end = max(proc_dir_dts)

                # Split the processing into daily chunks
                dt_bins = generateDatetimeBins(
                    proc_dir_dt_beg, proc_dir_dt_end, 
                    bin_days=1, tzinfo=datetime.timezone.utc, reverse=False)

                # check if we've created an extra bucket (might happen if requested timeperiod is less than 24h)
                if event_time_range is not None:
                    if dt_bins[-1][0] > event_time_range[1]: 
                        dt_bins.pop(-1)
            else:
                # in mcmode 2 we want to process all loaded trajectories so set the bin start/end accordingly
                dt_bins = [(dh.dt_range[0], dh.dt_range[1])]

            if dh.dt_range is not None:
                # there's some data to process
                log.info("")
                log.info("ALL TIME BINS:")
                log.info("----------")
                for bin_beg, bin_end in dt_bins:
                    log.info("{:s}, {:s}".format(str(bin_beg), str(bin_end)))


                ### ###


                # Go through all chunks in time
                for bin_beg, bin_end in dt_bins:

                    log.info("")
                    log.info("PROCESSING TIME BIN:")
                    log.info("{:s}, {:s}".format(str(bin_beg), str(bin_end)))
                    log.info("-----------------------------")
                    log.info("")

                    # Load data of unprocessed observations only if creating candidates
                    if mcmode & MCMODE_CANDS:
                        dh.unpaired_observations = dh.loadUnpairedObservations(dh.processing_list, 
                            dt_range=(bin_beg, bin_end))
                        log.info(f'loaded {len(dh.unpaired_observations)} observations')

                    if mcmode != MCMODE_PHASE2:
                        # remove any trajectories that no longer exist on disk
                        dh.removeDeletedTrajectories()
                        # load computed trajectories from disk into sqlite
                        dh.loadComputedTrajectories(dt_range=(bin_beg, bin_end))
                        # move any legacy failed traj into sqlite


                    # Run the trajectory correlator
                    tc = TrajectoryCorrelator(dh, trajectory_constraints, cml_args.velpart, data_in_j2000=True, enableOSM=cml_args.enableOSM)
                    bin_time_range = [bin_beg, bin_end]
                    num_done = tc.run(event_time_range=event_time_range, mcmode=mcmode, bin_time_range=bin_time_range)

                    if dh.RemoteDatahandler and dh.RemoteDatahandler.mode == 'child' and num_done > 0:
                        log.info('uploading to master node')
                        # close the databases and upload the data to the master node
                        if mcmode != MCMODE_PHASE2:
                            dh.traj_db.closeTrajDatabase()
                            dh.observations_db.closeObsDatabase()

                        dh.RemoteDatahandler.uploadToMaster(dh.output_dir, verbose=False)

                        # truncate the tables here so they are clean for the next run
                        if mcmode != MCMODE_PHASE2:
                            dh.traj_db = TrajectoryDatabase(dh.db_dir, purge_records=True)
                            dh.observations_db = ObservationDatabase(dh.db_dir, purge_records=True)

                if mcmode & MCMODE_CANDS:
                    dh.observations_db.closeObsDatabase()

            else:
                # there were no datasets to process
                log.info('no data to process yet')
            
            log.info("Total run time: {:s}".format(str(datetime.datetime.now(datetime.timezone.utc) - t1)))

            # Store the previous start time
            previous_start_time = copy.deepcopy(t1)



        # Break after one loop if auto mode is not on
        if cml_args.auto is None:
            # clear the remote data ready flag to indicate we're shutting down
            if dh.RemoteDatahandler and dh.RemoteDatahandler.mode == 'child':
                dh.RemoteDatahandler.setStopFlag()
            break

        else:

            # Otherwise wait to run AUTO_RUN_FREQUENCY hours after the beginning
            wait_time = (datetime.timedelta(hours=AUTO_RUN_FREQUENCY) 
                - (datetime.datetime.now(datetime.timezone.utc) - t1)).total_seconds()

            # remove the remote data stop flag to indicate we're open for business
            if dh.RemoteDatahandler and dh.RemoteDatahandler.mode == 'child':
                dh.RemoteDatahandler.clearStopFlag()

            # Run immediately if the wait time has elapsed
            if wait_time < 0:
                continue

            # Otherwise wait to run
            else:

                # Compute next run time
                next_run_time = datetime.datetime.now(datetime.timezone.utc) \
                    + datetime.timedelta(seconds=wait_time)
                
                log.info("Next run time: {:s}, waiting {:d} seconds...".format(str(next_run_time), 
                                                                               int(wait_time)))

                # Wait to run
                while next_run_time > datetime.datetime.now(datetime.timezone.utc):
                    print("Waiting {:s} to run the trajectory solver...          ".format(str(next_run_time 
                        - datetime.datetime.now(datetime.timezone.utc))))
                    time.sleep(10)
