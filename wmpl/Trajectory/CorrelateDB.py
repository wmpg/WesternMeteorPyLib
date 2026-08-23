# The MIT License

# Copyright (c) 2024 Mark McIntyre

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

""" Python scripts to manage the WMPL SQLite databases
"""
import os
import sys
import sqlite3
import logging
import logging.handlers
import argparse
import datetime
import json
import numpy as np
import copy
from time import sleep

from wmpl.Utils.TrajConversions import datetime2JD, jd2Date


log = logging.getLogger("wmpl_logger")

############################################################
# classes to handle the Observation and Trajectory databases
############################################################


class ObservationsDatabase():
    """
    A class to handle the sqlite observations database transparently.
    """

    def __init__(self, db_path, db_name='observations.db', purge_records=False, verbose=False):
        """
        Create an observations database instance

        Parameters:
        db_path         : path to the location of the database
        db_name         : name to use, typically observations.db
        purge_records   : boolean, if true then delete any existing records

        """
        self.db_path = db_path
        self.db_name = db_name
        db_full_name = os.path.join(db_path, f'{db_name}')
        if verbose:
            log.info(f'opening database {db_full_name}')
        con = sqlite3.connect(db_full_name)
        self.dbhandle = con
        con.execute('pragma journal_mode=wal')
        if purge_records:
            con.execute('drop table if exists paired_obs')
        res = con.execute("SELECT name FROM sqlite_master WHERE name='paired_obs'")
        if res.fetchone() is None:
            if verbose:
                log.info('create table paired_obs')
            con.execute('CREATE TABLE paired_obs(obs_id VARCHAR(36) UNIQUE, obs_dt REAL, status INTEGER)')
        self._commitObsDatabase()

    def _commitObsDatabase(self):
        """
        Commit the obs db. This function exists so we can do lazy writes
        """
        self.dbhandle.commit()
        try:
            self.dbhandle.execute('pragma wal_checkpoint(TRUNCATE)')
        except Exception:
            self.dbhandle.execute('pragma wal_checkpoint(PASSIVE)')
        return 

    def closeObsDatabase(self):
        """
        Close the database, making sure we commit any pending updates
        """

        if self.dbhandle:
            self._commitObsDatabase()
            self.dbhandle.close()
            self.dbhandle = None
        return 

    def checkObsPaired(self, obs_id, verbose=False):
        """
        Check if an observation is already marked paired
        return True if there is an observation with the correct obs id and with status = 1 

        Parameters:
        obs_id  : observation ID to check

        Returns: 
            True if paired, False otherwise
        """
        
        paired = True
        cur = self.dbhandle.execute('SELECT obs_id FROM paired_obs WHERE obs_id=? and status=1', (obs_id,))
        if cur.fetchone() is None:
            paired = False
        if verbose:
            log.info(f'{obs_id} is {"Paired" if paired else "Unpaired"}')
        return paired 

    def addPairedObservations(self, obs_ids, jdt_refs, verbose=False):
        """
        Add or update a list of observations paired, setting status = 1

        Parameters:
        obs_ids          : list of observation IDs
        jdt_refs         : list of julian reference dates of the observations
        """

        if obs_ids is None or jdt_refs is None or len(jdt_refs) != len(obs_ids):
            log.warning('malformed observations data')
            return False
        
        vals = []
        for obs_id, jdt_ref in zip(obs_ids, jdt_refs):
            vals.append({"obs_id":obs_id, "jdt_ref":jdt_ref, "status":1})        

        if verbose:
            log.info(f'adding {obs_ids} to paired_obs table')
        for retry in range(10):
            try:
                self.dbhandle.executemany('insert or replace into paired_obs values(:obs_id, :jdt_ref, :status)', vals)
                self.dbhandle.commit()
                return True
            except sqlite3.OperationalError as e:
                log.warning(f'failed add paired_obs  {vals}, try {retry+1}/10')
                log.warning(f'reason: {e}')
                sleep(1)
            except Exception as e:
                log.warning('unable to insert paired ops')
                log.warning(e)
                return False
        # if we get this far, the retries failed but no other error occured
        return  False


    def unpairObs(self, obs_ids, verbose=False):
        """
        Mark an observation unpaired.
        If an entry exists in the database, update the status to 0. 
        ** Currently unused. **

        Parameters:
        met_obs_list    : a list of observation IDs
        """
        if len(obs_ids) < 1:
            log.warning('no obs_ids supplied to unpairObs')
            return False

        qmarks = '?,'*len(obs_ids)

        if verbose:
            log.info(f'unpairing {obs_ids}')
        for retry in range(10):
            try:
                self.dbhandle.execute(f'update paired_obs set status = 0 where obs_id in ({qmarks[:-1]})', obs_ids)
                self.dbhandle.commit()
                return True
            except sqlite3.OperationalError as e:
                log.warning(f'failed to update {obs_ids} in database, try {retry+1}/10')
                log.warning(f'reason: {e}')
                sleep(1)
            except Exception as e:
                log.warning(f'failed to unpair {obs_ids}')
                log.warning(f'reason: {e}')
                return False
        # if we get to here, the retries failed
        return False   

    def getLinkedObservations(self, jdt_ref):
        """
        Return a list of observation IDs linked with a trajectory based on the jdt_ref of the traj

        Parameters
        jdt_ref     : the julian reference date of the trajectory

        """
        cur = self.dbhandle.execute(f"SELECT obs_id FROM paired_obs WHERE obs_dt=? and status=1", (round(jdt_ref,10),))
        return [x[0] for x in cur.fetchall()]

    def safeDetachDatabase(self, dbname):
        try:
            self.dbhandle.execute(f"detach database '{dbname}'")
        except Exception:
            pass

    def archiveObsDatabase(self, db_path, arch_prefix, archdate_jd):
        """
        archive records older than archdate_jd to a database {arch_prefix}_observations.db

        Parameters:
        db_path     : path to the location of the archive database   
        arch_prefix : prefix to apply - typically of the form yyyymm. Set this to None to purge without archiving.
        archdate_jd : julian date before which to archive data. Set this to None to purge anything older than 21 days.
        """

        if archdate_jd is None:
            archdate = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=21)
            archdate_jd = datetime2JD(archdate)

        purge_ok = True
        log.info(f'{"Archiving" if arch_prefix else "Purging"} observations database')

        # create the database if it doesnt exist

        for retry in range(10):
            try:
                if arch_prefix:
                    # open and close to create it if needed, then attach, copy records and delete from original db
                    archdb_name = f'{arch_prefix}_observations.db'
                    archdb = ObservationsDatabase(db_path, archdb_name)
                    archdb.closeObsDatabase()
                    archdb_fullname = os.path.join(db_path, f'{archdb_name}')
                    self.dbhandle.execute(f"attach database '{archdb_fullname}' as archdb")
                    # check the table exists before trying to copy from it
                    res = self.dbhandle.execute("SELECT count(name) FROM archdb.sqlite_master WHERE name='paired_obs'").fetchall()
                    if res[0][0] > 0:
                        self.dbhandle.execute('insert or replace into archdb.paired_obs select * from paired_obs where obs_dt < ?', (archdate_jd,))
                    self.dbhandle.commit()
                    self.dbhandle.execute("detach database 'archdb'")
                return self.purgeObsDatabase(archdate_jd=archdate_jd) 
            except sqlite3.OperationalError as e:
                log.warning(f'failed to archive observations, try {retry+1}/10')
                log.warning(f'reason: {e}')
                self.dbhandle.commit()
                self.safeDetachDatabase('archdb')
                sleep(1)
            except Exception as e:
                log.warning('unable to archive observations database')
                log.warning(f'reason: {e}')
                self.dbhandle.commit()
                self.safeDetachDatabase('archdb')
                return False
        # if we got this far, the retries failed or 
        self.dbhandle.commit()
        self.safeDetachDatabase('archdb')
        return  False

    def purgeObsDatabase(self, archdate_jd=None):
        """
        purge records from before a specified julian date. 

        parameters:
        archdate_jd :    julian date before which to purge. Default None will purge records more than 21 days old
    
        """
        if archdate_jd is None:
            archdate = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=21)
            archdate_jd = datetime2JD(archdate)

        for retry in range(10):
            try:
                res = self.dbhandle.execute('select count(*) from paired_obs where obs_dt < ?', (archdate_jd,)).fetchall()
                log.info(f'  purging {res[0][0]} records from paired_obs')
                self.dbhandle.execute('delete from paired_obs where obs_dt < ?', (archdate_jd,))
                self.dbhandle.commit()
                return True
            except sqlite3.OperationalError as e:
                log.warning(f'failed to purge observations database, try {retry+1}/10')
                log.warning(f'reason: {e}')
                sleep(1)
            except Exception as e:
                log.warning(f'failed to purge observations database')
                log.warning(f'reason: {e}')
                return False
        # if we got this far, the retries failed
        return  False

    def copyObsJsonRecords(self, paired_obs, dt_range):
        """ 
        Copy data from the legacy Json database to the new database between the dates specified in dt_range. 
        Note that copying large date ranges will be extremely slow. 

        Parameters:
        paired_obs  : a json list of paired observations from the old database.
        dt_range    : a date range to operate on.

        """
        # only copy recent observations since 
        dt_end = dt_range[1]
        dt_beg = dt_range[0]

        log.info('-----------------------------')
        log.info('moving recent observations to sqlite - this may take some time....')
        log.info(f'observation date range {dt_beg.isoformat()} to {dt_end.isoformat()}')

        i = 0
        keylist = paired_obs.keys()
        for stat_id in keylist:
            for obs_id in paired_obs[stat_id]:
                try:
                    obs_date = datetime.datetime.strptime(obs_id.split('_')[1], '%Y%m%d-%H%M%S.%f')
                except Exception:
                    obs_date = datetime.datetime(2000,1,1,0,0,0)
                obs_date = obs_date.replace(tzinfo=datetime.timezone.utc)

                if obs_date >= dt_beg and obs_date < dt_end:
                    self.addPairedObservations([obs_id], [datetime2JD(obs_date)])
                    i += 1
                if not i % 100000 and i != 0:
                    log.info(f'moved {i} observations')
        self.dbhandle.commit()
        log.info(f'done - moved {i} observations')
        log.info('-----------------------------')
        return 

    def mergeObsDatabase(self, source_db_path):
        """
        Merge in records from another database 'source_db_path', for example from a remote node

        Parameters:
        source_db_path  : full name and path to the source database to merge from 
        """

        if not os.path.isfile(source_db_path):
            log.warning(f'source database missing: {source_db_path}')
            return False

        for retry in range(10):
            try:
                self.dbhandle.execute(f"attach database '{source_db_path}' as sourcedb")
                rws = self.dbhandle.execute("SELECT count(name) FROM sourcedb.sqlite_master WHERE name='paired_obs'").fetchall()
                if rws[0][0] == 0:
                    # table is missing so nothing to do
                    log.info(f'  no records in {source_db_path} to merge')
                else:
                    self.dbhandle.execute('insert or replace into paired_obs select * from sourcedb.paired_obs')
                self.dbhandle.commit()
                self.dbhandle.execute("detach database 'sourcedb'")
                return True
            except sqlite3.OperationalError as e:
                log.warning(f'failed to merge paired_obs from {source_db_path}, try {retry+1}/10')
                log.warning(f'reason: {e}')
                sleep(1)
                self.dbhandle.commit()
                self.safeDetachDatabase('sourcedb')
            except Exception as e:
                log.warning('unable to archive observations database')
                log.warning(e)
                self.dbhandle.commit()
                self.safeDetachDatabase('sourcedb')
                return False
        # if we got this far, the retries failed
        self.dbhandle.commit()
        self.safeDetachDatabase('sourcedb')
        return  False


############################################################


class TrajectoryDatabase():
    """
    A class to handle the sqlite trajectory database transparently.
    """

    def __init__(self, db_path, db_name='trajectories.db', purge_records=False, verbose=False):
        """
        initialise the trajectory database

        Parameters:
        db_path         : path to the location to store the database
        db_name         : database name
        purge_records   : boolean, if true, delete any existing records
        """
        self.db_path = db_path
        self.db_name = db_name
        db_full_name = os.path.join(db_path, f'{db_name}')
        log.info(f'opening database {db_full_name}')
        con = sqlite3.connect(db_full_name)
        # use write-ahead logging to allow writer and concurrent multiple readers
        con.execute('pragma journal_mode=wal')
        if purge_records:
            con.execute('drop table if exists trajectories')
            con.execute('drop table if exists failed_trajectories')
            con.commit()
        res = con.execute("SELECT name FROM sqlite_master WHERE name='trajectories'")
        if res.fetchone() is None:
            if verbose:
                log.info('create table trajectories')
            con.execute("""CREATE TABLE trajectories(
                        jdt_ref REAL UNIQUE,
                        traj_id VARCHAR UNIQUE,
                        traj_file_path VARCHAR,
                        participating_stations VARCHAR, 
                        ignored_stations VARCHAR,
                        radiant_eci_mini VARCHAR,
                        state_vect_mini VARCHAR,
                        phase_1_only INTEGER,
                        v_init REAL,
                        gravity_factor REAL,
                        v0z REAL,
                        v_avg REAL,
                        rbeg_jd REAL, 
                        rend_jd REAL,
                        rbeg_lat REAL, 
                        rbeg_lon REAL, 
                        rbeg_ele REAL,
                        rend_lat REAL, 
                        rend_lon REAL,
                        rend_ele REAL,
                        obs_ids VARCHAR,
                        ign_obs_ids VARCHAR,
                        status INTEGER) """)

        res = con.execute("SELECT name FROM sqlite_master WHERE name='failed_trajectories'")
        if res.fetchone() is None:
            # note: traj_id not set as unique as some fails will have traj-id None
            if verbose:
                log.info('create table failed_trajectories')
            con.execute("""CREATE TABLE failed_trajectories(
                        jdt_ref REAL UNIQUE,
                        traj_id VARCHAR, 
                        traj_file_path VARCHAR,
                        participating_stations VARCHAR, 
                        ignored_stations VARCHAR,
                        radiant_eci_mini VARCHAR,
                        state_vect_mini VARCHAR,
                        phase_1_only INTEGER,
                        v_init REAL,
                        gravity_factor REAL,
                        obs_ids VARCHAR,
                        ign_obs_ids VARCHAR,
                        status INTEGER) """)
                        
        con.commit()
        self.dbhandle = con
        return 

    def _commitTrajDatabase(self, verbose=False):
        """
        commit the traj db. 
        This function exists so we can do lazy writes 
        """

        if verbose:
            log.info('commit trajdb')
        self.dbhandle.commit()
        try:
            self.dbhandle.execute('pragma wal_checkpoint(TRUNCATE)')
        except Exception:
            try:
                self.dbhandle.execute('pragma wal_checkpoint(PASSIVE)')
            except Exception:
                pass
        return 

    def closeTrajDatabase(self, verbose=False):
        """
        close the database, making sure we commit any pending updates
        """

        if verbose:
            log.info('close trajdb')
        if self.dbhandle:
            for retry in range(10):
                try:
                    self._commitTrajDatabase(verbose=verbose)
                    self.dbhandle.close()
                    self.dbhandle = None
                    return 
                except sqlite3.OperationalError as e:
                    log.warning(f'failed to commit and close {self.db_name}, try {retry+1}/10')
                    log.warning(f'reason: {e}')
                    sleep(1)
                except Exception as e:
                    log.warning(f'unable to close traj database')
                    log.warning(f'reason: {e}')
                    sleep(2)
        # if we got this far, the retries failed
        return 

    def checkTrajIfFailed(self, traj_reduced, verbose=False):
        """
        Check if a Trajectory was marked failed

        Parameters:
        traj_reduced    : a TrajReduced object

        Returns 
        True if there is a failed trajectory with the same traj_id or jdt_ref and matching list of stations

        """

        if not hasattr(traj_reduced, 'jdt_ref') or not hasattr(traj_reduced, 'participating_stations') or not hasattr(traj_reduced, 'ignored_stations'):
            return False
        
        found = False
        station_list = list(set(traj_reduced.participating_stations + traj_reduced.ignored_stations))
        if hasattr(traj_reduced, 'traj_id') and traj_reduced.traj_id is not None and traj_reduced.traj_id != 'None':
            traj_id = traj_reduced.traj_id
            res = self.dbhandle.execute('SELECT traj_id,participating_stations, ignored_stations FROM failed_trajectories WHERE traj_id=? and status=1', (traj_id,))
        else:
            test_jdt = traj_reduced.jdt_ref
            res = self.dbhandle.execute('SELECT traj_id,participating_stations, ignored_stations FROM failed_trajectories WHERE jdt_ref=? and status=1', (test_jdt,))
        row = res.fetchone()
        if row is None:
            found = False
        else:
            traj_stations = list(set(json.loads(row[1]) + json.loads(row[2])))
            found = True if (traj_stations == station_list) else False
        return found

    def addTrajectory(self, traj_reduced, failed=False, force_add=True, verbose=False):
        """
        add or update an entry in the database, setting status = 1

        Parameters:
        traj_reduced    : a TrajReduced object
        failed          : boolean, if true, add the traj to the fails list

        Returns:
            true if the trajectory was added, false if it exists already

        """

        tblname = 'failed_trajectories' if failed else 'trajectories'

        # if force_add is false, don't replace any existing entry
        if not force_add and hasattr(traj_reduced, 'traj_id') and traj_reduced.traj_id is not None:
            rws = self.dbhandle.execute(f'select count(traj_id) from {tblname} where status=1 and traj_id=?', (traj_reduced.traj_id,)).fetchall()
            if rws[0][0] > 0:
                return False
            
        if verbose:
            log.info(f'    adding jdt {traj_reduced.jdt_ref} to {tblname}')

        try:
            vals = copy.deepcopy(traj_reduced.__dict__)

            # remove the output_dir part from the path so that the data are location-independent
            if hasattr(traj_reduced, 'traj_file_path') and 'trajectories' in traj_reduced.traj_file_path:
                vals['traj_file_path'] = traj_reduced.traj_file_path[traj_reduced.traj_file_path.find('trajectories'):].replace('\\','/')
            else:
                vals['traj_file_path'] = ''

            # fixup bad values and convert lists to sqlite-compatible format
            vals['traj_id'] = 'None' if not hasattr(traj_reduced, 'traj_id') or traj_reduced.traj_id is None else traj_reduced.traj_id

            vals['participating_stations'] = json.dumps([]) if not hasattr(traj_reduced, 'participating_stations') or traj_reduced.participating_stations is None else json.dumps(traj_reduced.participating_stations)
            vals['ignored_stations'] = json.dumps([]) if not hasattr(traj_reduced, 'ignored_stations') or traj_reduced.ignored_stations is None else json.dumps(traj_reduced.ignored_stations)
            vals['obs_ids'] = json.dumps([]) if not hasattr(traj_reduced, 'obs_ids') or traj_reduced.obs_ids is None else json.dumps(traj_reduced.obs_ids)
            vals['ign_obs_ids'] = json.dumps([]) if not hasattr(traj_reduced, 'ign_obs_ids') or traj_reduced.ign_obs_ids is None else json.dumps(traj_reduced.ign_obs_ids)

            vals['v_init'] = 0 if not hasattr(traj_reduced, 'v_init') or traj_reduced.v_init is None else traj_reduced.v_init

            vals['radiant_eci_mini'] = json.dumps([0,0,0]) if traj_reduced.radiant_eci_mini is None else json.dumps(traj_reduced.radiant_eci_mini)
            vals['state_vect_mini'] = json.dumps([0,0,0]) if traj_reduced.state_vect_mini is None else json.dumps(traj_reduced.state_vect_mini)
        except Exception as e:
            log.warning('malformed trajectory')
            log.warning(e)
            return False

        for retry in range(10):
            try:
                if failed:
                    self.dbhandle.execute('insert or replace into failed_trajectories values(:jdt_ref,:traj_id,:traj_file_path,:participating_stations,:ignored_stations,' \
                                        ':radiant_eci_mini,:state_vect_mini,:phase_1_only,:v_init,:gravity_factor,' \
                                        ':obs_ids,:ign_obs_ids,1)', \
                                        vals)
                else:
                    self.dbhandle.execute('insert or replace into trajectories values(:jdt_ref,:traj_id,:traj_file_path,:participating_stations,:ignored_stations,' \
                                        ':radiant_eci_mini,:state_vect_mini,:phase_1_only,:v_init,:gravity_factor,' \
                                        ':v0z,:v_avg,:rbeg_jd,:rend_jd,:rbeg_lat,:rbeg_lon,:rbeg_ele,:rend_lat,:rend_lon,:rend_ele,' \
                                        ':obs_ids,:ign_obs_ids,1)', \
                                        vals)
                self.dbhandle.commit()
                return True
            except sqlite3.OperationalError as e:
                log.warning(f'failed to insert {vals["traj_id"]}, try {retry+1}/10')
                log.warning(f'reason: {e}')
                sleep(1)
            except Exception as e:
                log.warning(f'unable to insert {vals["traj_id"]} into traj database')
                log.warning(f'reason: {e}')
                return False
        # if we got this far, the retries failed
        return  False
                
   
    def removeTrajectoryById(self, traj_id, failed=False, verbose=False):
        """
        Mark a trajectory removed, matching on the traj_id
        If an entry exists, update the status to 0. 

        Parameters:
        traj_id         : a trajectory ID
        failed          : boolean, if true then remove from the fails list
        """
        if traj_id is None:
            log.info('not possible to remove if traj_id is None')
            return False

        if verbose:
            log.info(f'removing {traj_id}')

        table_name = 'failed_trajectories' if failed else 'trajectories'
        for retry in range(10):
            try:
                self.dbhandle.execute(f"update {table_name} set status=0 where traj_id=?", (traj_id,))
                self.dbhandle.commit()
                return True
            except sqlite3.OperationalError as e:
                log.warning(f'failed to mark {traj_id} deleted, try {retry+1}/10')
                log.warning(f'reason: {e}')
                sleep(1)
            except Exception as e:
                log.warning(f'failed to mark {traj_id} deleted')
                log.warning(e)
                return False
        # if we got this far, the retries failed
        return  False
    
    def getTrajectories(self, output_dir, jdt_range, failed=False, inc_deleted=False, verbose=False):
        """
        Get a list of trajectories between two julian dates 

        Parameters: 
        output_dir  : output_dir specified when invoking CorrelateRMS - will be prepended to the trajectory path
        jdt_range   : tuple of julian dates to retrieve data between. if the 2nd date is None, retrieve all data to today
        failed      : boolean - if true, retrieve failed traj rather than successful ones
        inc_deleted : include logically-deleted trajectories

        Returns:
        trajs: json list of traj_reduced objects

        """

        jdt_start, jdt_end = jdt_range
        sts_test = 'and status=1' if not inc_deleted else ''

        table_name = 'failed_trajectories' if failed else 'trajectories'
        if verbose:
            log.info(f'getting trajectories between {jd2Date(jdt_start, dt_obj=True).strftime("%Y%m%d_%M%M%S.%f")} and {jd2Date(jdt_end, dt_obj=True).strftime("%Y%m%d_%M%M%S.%f")}')

        if not jdt_end:
            rows = self.dbhandle.execute(f"SELECT * FROM {table_name} WHERE jdt_ref>=? {sts_test}", (jdt_start,)).fetchall()
        else:
            rows = self.dbhandle.execute(f"SELECT * FROM {table_name} WHERE jdt_ref>=? and jdt_ref<=? {sts_test}", (jdt_start,jdt_end,)).fetchall()
        trajs = []
        for rw in rows:
            rw = [np.nan if x == 'NaN' else x for x in rw]   
            if failed:
                json_dict = {'jdt_ref':rw[0], 'traj_id':rw[1], 'traj_file_path':os.path.join(output_dir, rw[2]),
                         'participating_stations': json.loads(rw[3]),
                         'ignored_stations': json.loads(rw[4]),
                         'radiant_eci_mini': json.loads(rw[5]),
                         'state_vect_mini': json.loads(rw[6]),
                         'phase_1_only': rw[7], 'v_init': rw[8],'gravity_factor': rw[9],
                         'obs_ids': json.loads(rw[10]), 'ign_obs_ids': json.loads(rw[11]),
                         }
            else:  
                json_dict = {'jdt_ref':rw[0], 'traj_id':rw[1], 'traj_file_path':os.path.join(output_dir, rw[2]),
                         'participating_stations': json.loads(rw[3]),
                         'ignored_stations': json.loads(rw[4]),
                         'radiant_eci_mini': json.loads(rw[5]),
                         'state_vect_mini': json.loads(rw[6]),
                         'phase_1_only': rw[7], 'v_init': rw[8],'gravity_factor': rw[9],
                         'v0z': rw[10], 'v_avg': rw[11], 
                         'rbeg_jd': rw[12], 'rend_jd': rw[13], 
                         'rbeg_lat': rw[14], 'rbeg_lon': rw[15], 'rbeg_ele': rw[16], 
                         'rend_lat': rw[17], 'rend_lon': rw[18], 'rend_ele': rw[19],
                         'obs_ids': json.loads(rw[20]), 'ign_obs_ids': json.loads(rw[21]),
                         }
            
            trajs.append(json_dict)
        return trajs

    def getTrajBasics(self, output_dir, jdt_range, failed=False, verbose=False):
        """
        Get a list of minimal trajectory details between two dates

        Parameters:
        output_dir  : output_dir specified when invoking CorrelateRMS - will be prepended to the trajectory path
        jdt_range   : tuple of julian dates to retrieve data between
        failed      : boolean, if true retrieve names of fails, otherwise retrieve successful 
    
        Returns:
        trajs: a json list of tuples of {jdt_ref, traj_id, traj_file_path}

        """

        jdt_start, jdt_end = jdt_range
        table_name = 'failed_trajectories' if failed else 'trajectories'
        if not jdt_start:
            rows = self.dbhandle.execute(f"SELECT jdt_ref, traj_id, traj_file_path, obs_ids, ign_obs_ids FROM {table_name} where status=1 order by jdt_ref").fetchall()
        elif not jdt_end:
            rows = self.dbhandle.execute(f"SELECT jdt_ref, traj_id, traj_file_path, obs_ids, ign_obs_ids FROM {table_name} WHERE jdt_ref=? and status=1 order by jdt_ref", (jdt_start,)).fetchall()
        else:
            rows = self.dbhandle.execute(f"SELECT jdt_ref, traj_id, traj_file_path, obs_ids, ign_obs_ids FROM {table_name} WHERE jdt_ref>=? and jdt_ref<=? and status=1 order by jdt_ref", (jdt_start, jdt_end,)).fetchall()
        trajs = []
        for rw in rows:
            trajs.append({'jdt_ref':rw[0], 'traj_id':rw[1], 'traj_file_path':os.path.join(output_dir, rw[2]), 
                'obs_ids':json.loads(rw[3]), 'ign_obs_ids':json.loads(rw[4])})
        return trajs

    def isBeingProcessed(self, traj_id, clear=False):
        """
        Check if a trajectory is already being processed

        Parameters:
        traj_id:    [string]    the trajectory ID
        clear:      [bool]      True if we want to check that a trajectory ISNT being processed

        Returns
        Bool - True if the traj is found and has the expected status, False otherwise 

        """
        try:
            if not clear:
                chk = self.dbhandle.execute('select traj_id from trajectories WHERE traj_id=? and status<>1', (traj_id,)).fetchall()
            else:
                chk = self.dbhandle.execute('select traj_id from trajectories WHERE traj_id=? and status=1', (traj_id,)).fetchall()
            if len(chk) == 0:
                return False
            return True
        except Exception as e:
            log.warning(f'problem checking if {traj_id} is already being processed')
            log.warning(f'reason: {e}')
            return False

    def markBeingProcessed(self, traj_id, clear=False):
        """
        Mark or unmark a trajectory as being processed.
         
        Parameters:
        traj_id:    [string] trajectory ID
        clear:      [bool] clear the 'being processed' marker instead of setting it. Default false

        Returns
        Bool -  True if the trajetory is found and was then marked being processed
                False if there was an error
        """
        for retry in range(10):
            try:
                statuscode = 1 if clear else 2
                self.dbhandle.execute('update trajectories set status=? WHERE traj_id=?', (statuscode, traj_id,))
                self.dbhandle.commit()
                return self.isBeingProcessed(traj_id, clear)
            except sqlite3.OperationalError as e:
                log.warning(f'failed to update {traj_id} to {statuscode}, retry {retry+1}/10')
                sleep(1)
            except Exception:
                log.warning(f'problem marking {traj_id} as being processed')
                log.warning(str(e))
                return False
        # if we get to here, the retries failed
        return False   

    def unmarkBeingProcessed(self, traj_id):
        return self.markBeingProcessed(traj_id, clear=True)

    def clearProcessingFlag(self):
        """
        Clear the processing flag from trajectories by setting status=1
        This is used at startup to ensure any trajectories left behind by a crash are picked up
        """
        for retry in range(10):
            try:
                self.dbhandle.execute('update trajectories set status=1 WHERE status=2')
                self.dbhandle.commit()
            except sqlite3.OperationalError as e:
                log.warning(f'failed to reset processing flag on trajectory database, retry {retry+1}/10')
                log.warning(f'reason: {e}')
                sleep(1)
            except Exception:
                log.warning('failed to reset processing flag on trajectory database')
                log.warning(f'reason: {e}')
                return False
        # if we get to here, the retries failed
        return False   


    def safeDetachDatabase(self, dbname):
        try:
            self.dbhandle.execute("detach database 'archdb'")
        except Exception:
            pass
        return

    def archiveTrajDatabase(self, db_path, arch_prefix, archdate_jd):
        """
        archive records older than archdate_jd to a database {arch_prefix}_trajectories.db

        Parameters:
        db_path     : path to the location of the archive database   
        arch_prefix : prefix to apply - typically of the form yyyymm. Set to None to purge data without archiving.
        archdate_jd : julian date before which to archive data. Default is today - 21 days

        """
        # if no archdate is set, then set it to 21 days
        if archdate_jd is None:
            archdate = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=21)
            archdate_jd = datetime2JD(archdate)

        log.info(f'{"Archiving" if arch_prefix else "Purging"} trajectories database')

        for retry in range(10):
            try:
                archdb_name = f'{arch_prefix}_trajectories.db'
                if arch_prefix:
                    # attach the arch db, copy the records then delete them        
                    archdb = TrajectoryDatabase(db_path, archdb_name)
                    archdb.closeTrajDatabase()
                    archdb_fullname = os.path.join(db_path, f'{archdb_name}')
                    cur = self.dbhandle.execute(f"attach database '{archdb_fullname}' as archdb")
                    for table_name in ['trajectories', 'failed_trajectories']:
                        cur.execute(f'insert or replace into archdb.{table_name} select * from {table_name} where jdt_ref<?',(archdate_jd,))
                    self.dbhandle.commit()
                    self.dbhandle.execute("detach database 'archdb'")
                return self.purgeTrajDatabase(archdate_jd=archdate_jd)
            except sqlite3.OperationalError as e:
                log.warning(f'failed to {"archive" if arch_prefix else "purge"} trajectories, try {retry+1}/10')
                log.warning(f'reason: {e}')
                self.dbhandle.commit()
                self.safeDetachDatabase('archdb')
                sleep(1)
            except Exception as e:
                log.warning(f'failed to {"archive" if arch_prefix else "purge"} trajectories')
                log.warning(f'reason: {e}')
                self.dbhandle.commit()
                self.safeDetachDatabase('archdb')
                return False
        # if we got this far, the retries failed
        self.dbhandle.commit()
        self.safeDetachDatabase('archdb')
        return  False

    
    def purgeTrajDatabase(self, archdate_jd=None):
        """
        purge records from before a specified julian date. 

        parameters:
            archdate_jd:    julian date before which to purge. Default None will purge records more than 21 days old
    
        """
        if archdate_jd is None:
            archdate = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=21)
            archdate_jd = datetime2JD(archdate)

        for retry in range(10):
            try:
                for table_name in ['trajectories', 'failed_trajectories']:
                    res = self.dbhandle.execute(f'select count(*) from {table_name} where jdt_ref<?', (archdate_jd,)).fetchall()
                    log.info(f'  purging {res[0][0]} records from {table_name}')
                    self.dbhandle.execute(f'delete from {table_name} where jdt_ref<?', (archdate_jd,))
                    self.dbhandle.commit()
                    return True
            except sqlite3.OperationalError as e:
                log.warning(f'failed to purge trajectory database, try {retry+1}/10')
                log.warning(f'reason: {e}')
                sleep(1)
            except Exception as e:
                log.warning('unable to purge trajectory database')
                log.warning(f'reason: {e}')
                return False
        # if we got this far, the retries failed
        return  False

    def copyTrajJsonRecords(self, trajectories, dt_range, failed=True, max_days=14):
        """
        Copy trajectories from the old Json database 
        We generally only copy recent records since if we ever run for an historic date
        its likely we will want to reanalyse all available data

        Parameters:

        trajectories    : json list of trajetories extracted from the old Json DB
        dt_range:       : date range to use, at most fourteen days at a time
        failed          : boolean, default true to move failed traj

        """
        jd_end = datetime2JD(dt_range[1])
        jd_beg = max(datetime2JD(dt_range[0]), jd_end - max_days)

        log.info(f'moving recent {"" if failed is False else "failed"} trajectories to sqlite - this may take some time....')
        log.info(f'trajectory date range {jd2Date(jd_beg, dt_obj=True).isoformat()} to {dt_range[1].isoformat()}')

        keylist = [k for k in trajectories.keys() if float(k) >= jd_beg and float(k) <= jd_end]
        i = 0 # just in case there aren't any trajectories to move
        for i,jdt_ref in enumerate(keylist):
            self.addTrajectory(trajectories[jdt_ref], failed=failed)
            i += 1
            if not i % 10000:
                self._commitTrajDatabase()
                log.info(f'moved {i} {"" if failed is False else "failed"} trajectories')
        self._commitTrajDatabase()
        log.info(f'done - moved {i} {"" if failed is False else "failed"} trajectories')

        return 
    
    def mergeTrajDatabase(self, source_db_path):
        """
        merge in records from another database, for example from a remote node

        Parameters:
        source_db_path  : the full name of the source database from which to merge in records

        """

        if not os.path.isfile(source_db_path):
            log.warning(f'source database missing: {source_db_path}')
            return False

        for retry in range(10):
            try:
                self.dbhandle.execute(f"attach database '{source_db_path}' as sourcedb")
                for table_name in ['trajectories', 'failed_trajectories']:
                    self.dbhandle.execute(f'insert or replace into {table_name} select * from sourcedb.{table_name}')
                    self.dbhandle.commit()
                self.dbhandle.execute("detach database 'sourcedb'")
                return True
            except sqlite3.OperationalError as e:
                log.warning(f'failed to merge {source_db_path}, try {retry+1}/10')
                log.warning(f'reason: {e}')
                self.dbhandle.commit()
                self.safeDetachDatabase('sourcedb')
                sleep(1)
            except Exception as e:
                log.warning(f'failed to merge {source_db_path}')
                log.warning(e)
                self.dbhandle.commit()
                self.safeDetachDatabase('sourcedb')
                return False
        # if we got this far, the retries failed
        self.dbhandle.commit()
        self.safeDetachDatabase('sourcedb')
        return  False


############################################################


class CandidateDatabase():
    """
    A class to handle the sqlite candidates database transparently.
    """

    def __init__(self, db_path:str, db_name='candidates.db', keep=21, verbose=False):
        """
        Create a database instance

        Parameters:
        db_path         : path to the location of the database
        db_name         : name to use, typically candidates.db
        keep            : Amount of data to keep. Default 21 days

        """
        self.db_path = db_path
        self.db_name = db_name
        db_full_name = os.path.join(db_path, f'{db_name}')
        if verbose:
            log.info(f'opening database {db_full_name}')
        con = sqlite3.connect(db_full_name)
        con.execute('pragma journal_mode=wal')
        res = con.execute("SELECT name FROM sqlite_master WHERE name='candidates'")
        if res.fetchone() is None:
            if verbose:
                log.info('create table candidates')
            con.execute('CREATE TABLE candidates(cand_id VARCHAR UNIQUE, ref_dt REAL, obs_ids VARCHAR, status INTEGER)')
        con.commit()
        self.dbhandle = con
        if keep > 0:
            keep_dt = datetime.datetime.now().replace(tzinfo=datetime.timezone.utc) - datetime.timedelta(days=keep)
            keep_jd = datetime2JD(keep_dt)
            self.purgeCandDatabase(archdate_jd=keep_jd)

    def _commitCandDatabase(self):
        """
        Commit the db. This function exists so we can do lazy writes
        """
        self.dbhandle.commit()
        try:
            self.dbhandle.execute('pragma wal_checkpoint(TRUNCATE)')
        except Exception:
            self.dbhandle.execute('pragma wal_checkpoint(PASSIVE)')
        return 

    def closeCandDatabase(self):
        """
        Close database, making sure we commit any pending updates
        """
        if self.dbhandle:
            self._commitCandDatabase()
            self.dbhandle.close()
            self.dbhandle = None
        return 

    def checkAndAddCand(self, cand_id:str, ref_dt:float, obs_ids:list, verbose=False):
        """
        Check and add a candidate if its not already there

        Parameters:
        cand_id     : candidate ID
        ref_dt      : reference date as a timestamp
        obs_ids     : list of observation IDs

        Returns: 
            True if added, False if its already present or the insert failed
        """
        
        obs_ids_str = json.dumps(list(set(obs_ids)))
        for retry in range(10):
            try:
                rws = self.dbhandle.execute('SELECT count(*) FROM candidates WHERE cand_id=?', (cand_id,)).fetchall()
                if rws[0][0] == 0:
                    self.dbhandle.execute('insert into candidates values (?,?,?,1)',(cand_id,ref_dt,obs_ids_str,))
                    self.dbhandle.commit()
                    if verbose:
                        log.info(f'{cand_id} was added to the database')
                    return True
                else:
                    if verbose:
                        log.info(f'{cand_id} was already in the database')
                    return False
            except sqlite3.OperationalError as e:
                log.warning(f'failed to insert {cand_id} , try {retry+1}/10')
                log.warning(f'reason: {e}')
                sleep(1)
            except Exception as e:
                log.warning(f'failed to insert {cand_id}')
                log.warning(f'reason: {e}')
                return False
        # if we got this far, the retries failed
        return  False

    def getCandidateObs(self, cand_id:str, verbose=False):
        """
        retrieve a list of observations linked to a candidate

        Parameters:
        cand_id     : candidate ID

        Returns: 
            the observations linked to the candidate

        This function is currently unused
        """
        
        obs_ids = []
        cur = self.dbhandle.execute('SELECT obs_ids FROM candidates WHERE cand_id=? and status=1', (cand_id,))
        rw = cur.fetchone()
        if rw is not None:
            obs_ids= json.loads(rw[0])
        if verbose:
            log.info(f'{cand_id} contains {obs_ids}')
        return obs_ids

    def isBeingProcessed(self, cand_id:str, clear=False):
        """
        Check if a candidate is already being processed

        Parameters:
        cand_id:    [string] either a pickle name or the candidate id

        Returns
        Bool - True if the candidate is found and has != 1 , False otherwise 
        (nb: returns true for logically deleted cands, this is deliberate)

        """
        if cand_id.endswith('.pickle'):
            cand_id = os.path.split(cand_id)[1]
            cand_id = os.path.splitext(cand_id)[0]
        try:
            if not clear:
                chk = self.dbhandle.execute('select count(cand_id) from candidates WHERE cand_id=? and status<>1', (cand_id,)).fetchall()
            else:
                chk = self.dbhandle.execute('select count(cand_id) from candidates WHERE cand_id=? and status=1', (cand_id,)).fetchall()
            if chk[0][0] == 0:
                return False
            return True
        except Exception as e:
            log.warning(f'problem checking if {cand_id} is already being processed')
            return False

    def markBeingProcessed(self, cand_id:str, clear=False):
        """
        Mark or unmark a candidate as being processed.
         
        Parameters:
        cand_id:    [string] either a pickle name or the candidate id
        clear:      [bool] clear the 'being processed' marker instead of setting it. Default false

        Returns
        Bool -  True if the candidate is found and was then marked being processed
                False if there was an error
        """
        if cand_id.endswith('.pickle'):
            cand_id = os.path.split(cand_id)[1]
            cand_id = os.path.splitext(cand_id)[0]
        statuscode = 1 if clear else 2
        for retry in range(10):
            try:
                self.dbhandle.execute('update candidates set status=? WHERE cand_id=?', (statuscode, cand_id,))
                self.dbhandle.commit()
                return self.isBeingProcessed(cand_id, clear)
            except sqlite3.OperationalError as e:
                log.warning(f'failed to set {cand_id} status, try {retry+1}/10')
                log.warning(f'reason: {e}')
                sleep(1)
            except Exception as e:
                log.warning(f'problem marking {cand_id} as being processed')
                log.warning(e)
                return False
        # if we get to here, the retries failed
        return False

    def unmarkBeingProcessed(self, cand_id:str):
        return self.markBeingProcessed(cand_id, clear=True)

    def clearProcessingFlag(self):
        """
        Clear the processing flag from candidates by setting status=1
        This is used at startup to ensure any candidates left behind by a crash are picked up
        """
        for retry in range(10):
            try:
                self.dbhandle.execute('update candidates set status=1 WHERE status=2')
                self.dbhandle.commit()
            except sqlite3.OperationalError as e:
                log.warning(f'failed to reset processing flag on candidate database, retry {retry+1}/10')
                log.warning(f'reason: {e}')
                sleep(1)
            except Exception:
                log.warning('failed to reset processing flag on candidate database')
                log.warning(f'reason: {e}')
                return False
        # if we get to here, the retries failed
        return False   

    def safeDetachDatabase(self, dbname):
        try:
            self.dbhandle.execute(f"detach database 'dbname'")
        except Exception:
            pass
        return 
    
    def archiveCandDatabase(self, db_path, arch_prefix, archdate_jd):
        """
        archive records older than archdate_jd to a database {arch_prefix}_candidates.db

        Parameters:
        db_path     : path to the location of the archive database   
        arch_prefix : prefix to apply - typically of the form yyyymm
        archdate_jd : julian date before which to archive data

        """

        if archdate_jd is None:
            keep_dt = datetime.datetime.now().replace(tzinfo=datetime.timezone.utc) - datetime.timedelta(days=21)
        else:
            keep_dt = jd2Date(archdate_jd,dt_obj=True)

        for retry in range(10):
            try:
                if arch_prefix: 
                    archdb_name = f'{arch_prefix}_candidates.db'
                    archdb = CandidateDatabase(db_path, archdb_name, keep=0)
                    archdb.closeCandDatabase()
                    archdb_fullname = os.path.join(db_path, f'{archdb_name}')
                    self.dbhandle.execute(f"attach database '{archdb_fullname}' as archdb")
                    self.dbhandle.execute('insert or replace into archdb.candidates select * from candidates where ref_dt < ?', (keep_dt.timestamp(),))
                    self.dbhandle.commit()
                    self.dbhandle.execute("detach database 'archdb'")
                return self.purgeCandDatabase(archdate_jd=archdate_jd)
            except sqlite3.OperationalError as e:
                log.warning(f'failed to {"archive" if arch_prefix else "purge"}, try {retry+1}/10')
                log.warning(f'reason: {e}')
                self.dbhandle.commit()
                self.safeDetachDatabase('archdb')
                sleep(1)
            except Exception as e:
                log.warning(f'failed to {"archive" if arch_prefix else "purge"}')
                log.warning(e)
                self.dbhandle.commit()
                self.safeDetachDatabase('archdb')
                return False
        self.dbhandle.commit()
        self.safeDetachDatabase('archdb')
        return  False

    def purgeCandDatabase(self, archdate_jd=None):
        """
        purge old candidates after 'keep' weeks

        Parameters:
        keep    : days to keep data for, default 21
        """
        if archdate_jd is None:
            keep_dt = datetime.datetime.now().replace(tzinfo=datetime.timezone.utc) - datetime.timedelta(days=21)
        else:
            keep_dt = jd2Date(archdate_jd,dt_obj=True)

        log.info(f'purging candidates older than {keep_dt.isoformat()}')
        for retry in range(10):
            try:
                self.dbhandle.execute('delete from candidates where ref_dt < ?', (keep_dt.timestamp(),))
                self.dbhandle.commit()
                return True
            except sqlite3.OperationalError as e:
                log.warning(f'failed to purge cand db, try {retry+1}/10')
                log.warning(f'reason: {e}')
                sleep(1)
            except Exception as e:
                log.warning(f'failed to purge cand db')
                log.warning(e)
                return False
        return  False

    def mergeCandDatabase(self, source_db_path):
        """
        merge in records from another observation database, for example from a remote node

        Parameters:
        source_db_path  : the full name of the source database from which to merge in records

        """

        if not os.path.isfile(source_db_path):
            log.warning(f'source database missing: {source_db_path}')
            return 

        for retry in range(10):
            try:
                self.dbhandle.execute(f"attach database '{source_db_path}' as sourcedb")
                try:
                    self.dbhandle.execute(f'insert or replace into candidates select * from sourcedb.candidates')
                except Exception as e:
                    log.warning(f'skipping  {source_db_path}')
                    log.warning(f'reason: {e}')
                self.dbhandle.commit()
                self.dbhandle.execute("detach database 'sourcedb'")
                return True
            except sqlite3.OperationalError as e:
                log.warning(f'failed to merge records from {source_db_path}, try {retry+1}/10')
                log.warning(f'reason: {e}')
                self.dbhandle.commit()
                self.safeDetachDatabase('sourcedb')
                sleep(1)
            except Exception as e:
                log.warning('unable to merge candidate records')
                log.warning(e)
                self.dbhandle.commit()
                self.safeDetachDatabase('sourcedb')
                return False
        # if we get this far, the retries failed. 
        self.dbhandle.commit()
        self.safeDetachDatabase('sourcedb')
        return  False
    

##################################################################################
# dummy classes for use in the above.
# We can't import from CorrelateRMS as this would create a circular reference 


class DummyTrajReduced():
    """
    a dummy class for handling TrajReduced objects.
    We can't import CorrelateRMS as that would create a circular dependency
    """
    def __init__(self, jdt_ref=None, traj_id=None, traj_file_path=None, json_dict=None):
        if json_dict is None:
            self.jdt_ref = jdt_ref
            self.traj_id = traj_id
            self.traj_file_path = traj_file_path
        else:
            self.__dict__ = json_dict


class dummyDatabaseJSON():
    """
    Dummy class to handle the old Json data format
    We can't import CorrelateRMS as that would create a circular dependency
    """
    def __init__(self, db_dir, dt_range=None):
        self.db_file_path = os.path.join(db_dir, 'processed_trajectories.json')
        self.paired_obs = {}
        self.failed_trajectories = {}
        if os.path.exists(self.db_file_path):
            self.__dict__ = json.load(open(self.db_file_path))
            
            if hasattr(self, 'failed_trajectories'):
                # Convert trajectories from JSON to TrajectoryReduced objects
                traj_dict = getattr(self, "failed_trajectories")
                trajectories_obj_dict = {}
                for traj_json in traj_dict:
                    traj_reduced_tmp = DummyTrajReduced(json_dict=traj_dict[traj_json])
                    trajectories_obj_dict[traj_reduced_tmp.jdt_ref] = traj_reduced_tmp
                setattr(self, "failed_trajectories", trajectories_obj_dict)

            if hasattr(self, 'trajectories'):
                # Convert trajectories from JSON to TrajectoryReduced objects
                traj_dict = getattr(self, "trajectories")
                trajectories_obj_dict = {}
                for traj_json in traj_dict:
                    traj_reduced_tmp = DummyTrajReduced(json_dict=traj_dict[traj_json])
                    trajectories_obj_dict[traj_reduced_tmp.jdt_ref] = traj_reduced_tmp
                setattr(self, "trajectories", trajectories_obj_dict)


##################################################################################


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="""Automatically compute trajectories from RMS data in the given directory.""",
        formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('--dir_path', type=str, default=None, help='Path to the directory containing the databases.')

    arg_parser.add_argument('--database', type=str, default=None, help='Database to process, either observations or trajectories')

    arg_parser.add_argument('--action', type=str, default=None, help='Action to take on the database')

    arg_parser.add_argument('--stmt', type=str, default=None, help='statement to execute eg "select * from paired_obs"')

    arg_parser.add_argument('--logdir', type=str, default=None,
        help="Path to the directory where the log files will be stored. If not given, a logs folder will be created in the database folder")
      
    arg_parser.add_argument('-r', '--timerange', metavar='TIME_RANGE',
        help="""Apply action to this date range in the format: "(YYYYMMDD-HHMMSS,YYYYMMDD-HHMMSS)".""", type=str)
    
    cml_args = arg_parser.parse_args()
    # Find the log directory
    log_dir = cml_args.logdir
    if log_dir is None:
        log_dir = os.path.join(cml_args.dir_path, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log.setLevel(logging.DEBUG)

    # Init the log formatter
    log_formatter = logging.Formatter(
        fmt='%(asctime)s-%(levelname)-5s-%(module)-15s:%(lineno)-5d- %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S')

    # Init the file handler
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'correlate_db_{timestamp}.log')
    file_handler = logging.handlers.TimedRotatingFileHandler(log_file, when='midnight', backupCount=7)
    file_handler.setFormatter(log_formatter)
    log.addHandler(file_handler)

    # Init the console handler (i.e. print to console)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    log.addHandler(console_handler)

    if cml_args.database:
        dbname = cml_args.database.lower()
    action = cml_args.action.lower()

    stmt = cml_args.stmt

    dt_range = None
    if cml_args.timerange is not None:
        time_beg, time_end = cml_args.timerange.strip('(').strip(')').split(',')
        dt_beg = datetime.datetime.strptime(time_beg, "%Y%m%d-%H%M%S").replace(tzinfo=datetime.timezone.utc)
        dt_end = datetime.datetime.strptime(time_end, "%Y%m%d-%H%M%S").replace(tzinfo=datetime.timezone.utc)
        log.info('Custom time range:')
        log.info('    BEG: {:s}'.format(str(dt_beg)))
        log.info('    END: {:s}'.format(str(dt_end)))
        dt_range = [dt_beg, dt_end]


    if action == 'copy':
        if dt_range is None:
            log.info('Date range must be provided for copy operation')
        else:
            dt_range_jd = [datetime2JD(dt_range[0]),datetime2JD(dt_range[1])]
            jsondb = dummyDatabaseJSON(db_dir=cml_args.dir_path)
            obsdb = ObservationsDatabase(cml_args.dir_path)
            obsdb.copyObsJsonRecords(jsondb.paired_obs, dt_range)
            obsdb.closeObsDatabase()
            trajdb = TrajectoryDatabase(cml_args.dir_path)
            trajdb.copyTrajJsonRecords(jsondb.failed_trajectories, dt_range, failed=True)
            trajdb.copyTrajJsonRecords(jsondb.trajectories, dt_range, failed=False)
            trajdb.closeTrajDatabase()
    else:
        if dbname == 'observations':
            obsdb = ObservationsDatabase(cml_args.dir_path)
            if action == 'status':
                cur = obsdb.dbhandle.execute('select * from paired_obs where status=1')
                print(f'there are {len(cur.fetchall())} paired obs')
                cur = obsdb.dbhandle.execute('select * from paired_obs where status=0')
                print(f'and       {len(cur.fetchall())} unpaired obs')
            if action == 'execute':
                print(stmt)
                cur = obsdb.dbhandle.execute(stmt)
                for rw in cur.fetchall():
                    print(rw)
            obsdb.closeObsDatabase()

        elif dbname == 'trajectories':
            trajdb = TrajectoryDatabase(cml_args.dir_path)
            if action == 'status':
                cur = trajdb.dbhandle.execute('select * from trajectories where status=1')
                print(f'there are {len(cur.fetchall())} successful trajectories')
                cur = trajdb.dbhandle.execute('select * from failed_trajectories')
                print(f'and       {len(cur.fetchall())} failed trajectories')
            if action == 'execute':
                print(stmt)
                cur = trajdb.dbhandle.execute(stmt)
                for rw in cur.fetchall():
                    print(rw)
            trajdb.closeTrajDatabase()
        else:
            log.info('valid database not specified')
