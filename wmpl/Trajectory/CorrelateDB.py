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
import sqlite3
import logging
import logging.handlers
import argparse
import datetime
import json
import numpy as np

from wmpl.Utils.TrajConversions import datetime2JD, jd2Date


log = logging.getLogger("traj_correlator")

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
            con.execute("CREATE TABLE paired_obs(obs_id VARCHAR(36) UNIQUE, obs_dt REAL, status INTEGER)")
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
        cur = self.dbhandle.execute(f"SELECT obs_id FROM paired_obs WHERE obs_id='{obs_id}' and status=1")
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
            log.warning(f'malformed observations data')
            return False
        
        vals_str = ','.join(map(str,[(id, float(dt), 1) for id,dt in zip(obs_ids,jdt_refs)]))

        if verbose:
            log.info(f'adding {obs_ids} to paired_obs table')
        try:
            self.dbhandle.execute(f"insert or replace into paired_obs values {vals_str}")
            self.dbhandle.commit()
            return True
        except Exception as e:
            log.warning(f'failed to add {obs_ids} to paired_obs table')
            log.warning(vals_str)
            log.warning(e)
            return False            

        return 

    def addPairedObs(self, obs_id, jdt_ref, verbose=False):
        """
        Add or update a single entry in the database to mark an observation paired, setting status = 1

        Parameters:
        obs_id          : observation ID
        jdt_ref         : julian reference date of the observation
        """

        if verbose:
            log.info(f'adding {obs_id} to paired_obs table')
        try:
            self.dbhandle.execute(f"insert or replace into paired_obs values ('{obs_id}', {jdt_ref}, 1)")
            self.dbhandle.commit()
            return True
        except Exception:
            log.warning(f'failed to add {obs_id} to paired_obs table')
            return False            

    def unpairObs(self, obs_ids, verbose=False):
        """
        Mark an observation unpaired.
        If an entry exists in the database, update the status to 0. 
        ** Currently unused. **

        Parameters:
        met_obs_list    : a list of observation IDs
        """
        obs_ids_str = ','.join(f"'{id}'" for id in obs_ids)

        if verbose:
            log.info(f'unpairing {obs_ids_str}')
        try:
            self.dbhandle.execute(f"update paired_obs set status = 0 where obs_id in ({obs_ids_str})")
            self.dbhandle.commit()
            return True
        except Exception:
            log.warning(f'failed to unpair {obs_ids_str}')
            return False   

    def getLinkedObservations(self, jdt_ref):
        """
        Return a list of observation IDs linked with a trajectory based on the jdt_ref of the traj

        Parameters
        jdt_ref     : the julian reference date of the trajectory

        """
        cur = self.dbhandle.execute(f"SELECT obs_id FROM paired_obs WHERE obs_dt={jdt_ref} and status=1")
        return [x[0] for x in cur.fetchall()]

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
        if arch_prefix:
            # create the database if it doesnt exist
            archdb_name = f'{arch_prefix}_observations.db'
            archdb = ObservationsDatabase(db_path, archdb_name)
            archdb.closeObsDatabase()

            # attach the arch db, copy the records then delete them
            archdb_fullname = os.path.join(db_path, f'{archdb_name}')
            self.dbhandle.execute(f"attach database '{archdb_fullname}' as archdb")
            try:
                self.dbhandle.execute(f'insert or replace into archdb.paired_obs select * from paired_obs where obs_dt < {archdate_jd}')
            except Exception:
                log.warning('unable to archive observations database')
                purge_ok = False
            self.dbhandle.execute("detach database 'archdb'")
        if purge_ok:
            self.purgeObsDatabase(archdate_jd=archdate_jd) 
        return 

    def purgeObsDatabase(self, archdate_jd=None):
        """
        purge records from before a specified julian date. 

        parameters:
        archdate_jd :    julian date before which to purge. Default None will purge records more than 21 days old
    
        """
        if archdate_jd is None:
            archdate = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=21)
            archdate_jd = datetime2JD(archdate)

        cur = self.dbhandle.execute(f'select count(*) from paired_obs where obs_dt < {archdate_jd}')
        res = cur.fetchone()
        count = res[0] if res else 0
        log.info(f'  purging {count} records from paired_obs')
        self.dbhandle.execute(f'delete from paired_obs where obs_dt < {archdate_jd}')
        self.dbhandle.commit()
        return 

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
                    self.addPairedObs(obs_id, datetime2JD(obs_date))
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
            return 
        # attach the other db, copy the records then detach it
        self.dbhandle.execute(f"attach database '{source_db_path}' as sourcedb")
        res = self.dbhandle.execute("SELECT name FROM sourcedb.sqlite_master WHERE name='paired_obs'")
        if res.fetchone() is None:
            # table is missing so nothing to do
            status = True
        else:
            try:
                self.dbhandle.execute('insert or replace into paired_obs select * from sourcedb.paired_obs')
                status = True
            except Exception as e:
                log.info(f'unable to merge child observations from {source_db_path}')
                log.info(e)
                status = False

        self.dbhandle.commit()
        self.dbhandle.execute("detach database 'sourcedb'")
        return status


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

        db_full_name = os.path.join(db_path, f'{db_name}')
        log.info(f'opening database {db_full_name}')
        con = sqlite3.connect(db_full_name)
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
        This function exists so we can do lazy writes in some cases
        """

        if verbose:
            log.info('commit trajdb')
        self.dbhandle.commit()
        return 

    def closeTrajDatabase(self, verbose=False):
        """
        close the database, making sure we commit any pending updates
        """

        if verbose:
            log.info('close trajdb')
        if self.dbhandle:
            self._commitTrajDatabase(verbose=verbose)
            self.dbhandle.close()
            self.dbhandle = None
        return 

    def checkTrajIfFailed(self, traj_reduced, verbose=False):
        """
        Check if a Trajectory was marked failed

        Parameters:
        traj_reduced    : a TrajReduced object

        Returns 
        True if there is a failed trajectory with the same jdt_ref and matching list of stations
        """

        if not hasattr(traj_reduced, 'jdt_ref') or not hasattr(traj_reduced, 'participating_stations') or not hasattr(traj_reduced, 'ignored_stations'):
            return False
        
        found = False
        station_list = list(set(traj_reduced.participating_stations + traj_reduced.ignored_stations))
        res = self.dbhandle.execute(f"SELECT traj_id,participating_stations, ignored_stations FROM failed_trajectories WHERE jdt_ref={traj_reduced.jdt_ref} and status=1")
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
            res = self.dbhandle.execute(f'select traj_id from {tblname} where status = 1 and traj_id = "{traj_reduced.traj_id}"')
            row = res.fetchone()
            if row is not None and row[0] !='None':
                return False
            
        if verbose:
            log.info(f'    adding jdt {traj_reduced.jdt_ref} to {tblname}')

        try:
            # remove the output_dir part from the path so that the data are location-independent
            traj_file_path = traj_reduced.traj_file_path[traj_reduced.traj_file_path.find('trajectories'):]

            # and remove windows-style path separators
            traj_file_path = traj_file_path.replace('\\','/')

            obs_ids = 'None' if not hasattr(traj_reduced, 'obs_ids') or traj_reduced.obs_ids is None else traj_reduced.obs_ids
            ign_obs_ids = 'None' if not hasattr(traj_reduced, 'ign_obs_ids') or traj_reduced.ign_obs_ids is None else traj_reduced.ign_obs_ids

            if failed:
                # fixup possible bad values
                traj_id = 'None' if not hasattr(traj_reduced, 'traj_id') or traj_reduced.traj_id is None else traj_reduced.traj_id
                v_init = 0 if traj_reduced.v_init is None else traj_reduced.v_init
                radiant_eci_mini = [0,0,0] if traj_reduced.radiant_eci_mini is None else traj_reduced.radiant_eci_mini
                state_vect_mini = [0,0,0] if traj_reduced.state_vect_mini is None else traj_reduced.state_vect_mini

                sql_str = (f'insert or replace into failed_trajectories values ('
                            f"{traj_reduced.jdt_ref}, '{traj_id}', '{traj_file_path}',"
                            f"'{json.dumps(traj_reduced.participating_stations)}',"
                            f"'{json.dumps(traj_reduced.ignored_stations)}',"
                            f"'{json.dumps(radiant_eci_mini)}',"
                            f"'{json.dumps(state_vect_mini)}',"
                            f"0,{v_init},{traj_reduced.gravity_factor},"
                            f"'{json.dumps(obs_ids)}',"
                            f"'{json.dumps(ign_obs_ids)}',1)")
            else:
                sql_str = (f'insert or replace into trajectories values ('
                            f"{traj_reduced.jdt_ref}, '{traj_reduced.traj_id}', '{traj_file_path}',"
                            f"'{json.dumps(traj_reduced.participating_stations)}',"
                            f"'{json.dumps(traj_reduced.ignored_stations)}',"
                            f"'{json.dumps(traj_reduced.radiant_eci_mini)}',"
                            f"'{json.dumps(traj_reduced.state_vect_mini)}',"
                            f"{traj_reduced.phase_1_only},{traj_reduced.v_init},{traj_reduced.gravity_factor},"
                            f"{traj_reduced.v0z},{traj_reduced.v_avg},"
                            f"{traj_reduced.rbeg_jd},{traj_reduced.rend_jd},"
                            f"{traj_reduced.rbeg_lat},{traj_reduced.rbeg_lon},{traj_reduced.rbeg_ele},"
                            f"{traj_reduced.rend_lat},{traj_reduced.rend_lon},{traj_reduced.rend_ele},"
                            f"'{json.dumps(obs_ids)}',"
                            f"'{json.dumps(ign_obs_ids)}',1)")

            sql_str = sql_str.replace('nan','"NaN"')
        except Exception as e:
            log.warning('malformed trajectory')
            print(e)
            return False
        try:
            self.dbhandle.execute(sql_str)
        except Exception as e:
            print(e)
            print(sql_str)
            return False
        self.dbhandle.commit()
        return True

    def removeTrajectory(self, traj_reduced, failed=False, verbose=False):
        """
        Mark a trajectory unsolved
        If an entry exists, update the status to 0. 

        Parameters:
        traj_reduced    : a TrajReduced object
        failed          : boolean, if true then remove from the fails list
        """
        if verbose:
            log.info(f'removing {traj_reduced.traj_id}')
        table_name = 'failed_trajectories' if failed else 'trajectories'

        self.dbhandle.execute(f"update {table_name} set status=0 where jdt_ref='{traj_reduced.jdt_ref}'")
        self.dbhandle.commit()

        return True
    
    def removeTrajectoryById(self, traj_id, failed=False, verbose=False):
        """
        Mark a trajectory unsolved
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

        self.dbhandle.execute(f"update {table_name} set status=0 where traj_id='{traj_id}'")
        self.dbhandle.commit()

        return True

    
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
            rows = self.dbhandle.execute(f"SELECT * FROM {table_name} WHERE jdt_ref>={jdt_start} {sts_test}")
        else:
            rows = self.dbhandle.execute(f"SELECT * FROM {table_name} WHERE jdt_ref>={jdt_start} and jdt_ref<={jdt_end} {sts_test}")
        trajs = []
        for rw in rows.fetchall():
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
        jdt_range   : tuple of julian dates to retrieve data betwee
        failed      : boolean, if true retrieve names of fails, otherwise retrieve successful 
    
        Returns:
        trajs: a json list of tuples of {jdt_ref, traj_id, traj_file_path}

        """

        jdt_start, jdt_end = jdt_range
        table_name = 'failed_trajectories' if failed else 'trajectories'
        if not jdt_start:
            cur = self.dbhandle.execute(f"SELECT jdt_ref, traj_id, traj_file_path, obs_ids, ign_obs_ids FROM {table_name} where status=1 order by jdt_ref")
            rows = cur.fetchall()
        elif not jdt_end:
            cur = self.dbhandle.execute(f"SELECT jdt_ref, traj_id, traj_file_path, obs_ids, ign_obs_ids FROM {table_name} WHERE jdt_ref={jdt_start} and status=1 order by jdt_ref")
            rows = cur.fetchall()
        else:
            cur = self.dbhandle.execute(f"SELECT jdt_ref, traj_id, traj_file_path, obs_ids, ign_obs_ids FROM {table_name} WHERE jdt_ref>={jdt_start} and jdt_ref<={jdt_end} and status=1 order by jdt_ref")
            rows = cur.fetchall()
        trajs = []
        for rw in rows:
            trajs.append({'jdt_ref':rw[0], 'traj_id':rw[1], 'traj_file_path':os.path.join(output_dir, rw[2]), 
                'obs_ids':json.loads(rw[3]), 'ign_obs_ids':json.loads(rw[4])})
        return trajs

    def archiveTrajDatabase(self, db_path, arch_prefix, archdate_jd):
        """
        archive records older than archdate_jd to a database {arch_prefix}_trajectories.db

        Parameters:
        db_path     : path to the location of the archive database   
        arch_prefix : prefix to apply - typically of the form yyyymm. Set to None to purge data without archiving.
        archdate_jd : julian date before which to archive data. Default is now-21 dayss

        """
        # if no archdate is set, then set it to 21 days
        if archdate_jd is None:
            archdate = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=21)
            archdate_jd = datetime2JD(archdate)

        log.info(f'{"Archiving" if arch_prefix else "Purging"} trajectories database')

        purge_ok = True
        if arch_prefix:
            # create the archive database if it doesnt exist
            archdb_name = f'{arch_prefix}_trajectories.db'
            archdb = TrajectoryDatabase(db_path, archdb_name)
            archdb.closeTrajDatabase()

            # attach the arch db, copy the records then delete them
            archdb_fullname = os.path.join(db_path, f'{archdb_name}')
            cur = self.dbhandle.execute(f"attach database '{archdb_fullname}' as archdb")
            for table_name in ['trajectories', 'failed_trajectories']:
                try:
                    # bulk-copy if possible
                    cur.execute(f'insert or replace into archdb.{table_name} select * from {table_name} where jdt_ref < {archdate_jd}')
                except Exception:
                    log.warning(f'unable to archive {table_name} in trajectories database')
                    purge_ok = False

            self.dbhandle.execute("detach database 'archdb'")
            self.dbhandle.commit()

        if purge_ok:
            self.purgeTrajDatabase(archdate_jd=archdate_jd)
        return 
    
    def purgeTrajDatabase(self, archdate_jd=None):
        """
        purge records from before a specified julian date. 

        parameters:
            archdate_jd:    julian date before which to purge. Default None will purge records more than 21 days old
    
        """
        if archdate_jd is None:
            archdate = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=21)
            archdate_jd = datetime2JD(archdate)

        for table_name in ['trajectories', 'failed_trajectories']:
            cur = self.dbhandle.execute(f'select count(*) from {table_name} where jdt_ref < {archdate_jd}')
            res = cur.fetchone()
            count = res[0] if res else 0
            log.info(f'  purging {count} records from {table_name}')
            self.dbhandle.execute(f'delete from {table_name} where jdt_ref < {archdate_jd}')
        self.dbhandle.commit()
        return         



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
            return 
        # attach the other db, copy the records then detach it
        cur = self.dbhandle.execute(f"attach database '{source_db_path}' as sourcedb")

        status = True
        for table_name in ['trajectories', 'failed_trajectories']:
            try:
                # bulk-copy if possible
                cur.execute(f'insert or replace into {table_name} select * from sourcedb.{table_name}')
            except Exception:
                log.warning(f'unable to merge data from {source_db_path}')
                status = False
        self.dbhandle.commit()
        cur.execute("detach database 'sourcedb'")
        return status


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
        db_full_name = os.path.join(db_path, f'{db_name}')
        if verbose:
            log.info(f'opening database {db_full_name}')
        con = sqlite3.connect(db_full_name)
        con.execute('pragma journal_mode=wal')
        res = con.execute("SELECT name FROM sqlite_master WHERE name='candidates'")
        if res.fetchone() is None:
            if verbose:
                log.info('create table candidates')
            con.execute("CREATE TABLE candidates(cand_id VARCHAR UNIQUE, ref_dt REAL, obs_ids VARCHAR, status INTEGER)")
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
            True if added, False if its already present
        """
        
        to_be_added = True
        cur = self.dbhandle.execute(f"SELECT * FROM candidates WHERE cand_id='{cand_id}' and status=1")
        if cur.fetchone() is not None:
            to_be_added = False
        else:
            to_be_added = True
            obs_ids_str = json.dumps(list(set(obs_ids)))
            self.dbhandle.execute(f"insert into candidates values ('{cand_id}',{ref_dt},'{obs_ids_str}',1)")
            self.dbhandle.commit()
        if verbose:
            log.info(f'{cand_id} {"was added to the database" if to_be_added else "already present"}')
        return to_be_added

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
        cur = self.dbhandle.execute(f"SELECT obs_ids FROM candidates WHERE cand_id='{cand_id}' and status=1")
        rw = cur.fetchone()
        if rw is not None:
            obs_ids= json.loads(rw[0])
        if verbose:
            log.info(f'{cand_id} contains {obs_ids}')
        return obs_ids

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
        self.dbhandle.execute(f"delete from candidates where ref_dt < {keep_dt.timestamp()}")
        self.dbhandle.commit()
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

        purge_ok = True
        if arch_prefix:
            # create the archive database if it doesnt exist
            archdb_name = f'{arch_prefix}_candidates.db'
            archdb = CandidateDatabase(db_path, archdb_name, keep=0)
            archdb.closeCandDatabase()

            # attach the arch db, copy the records then delete them
            archdb_fullname = os.path.join(db_path, f'{archdb_name}')
            cur = self.dbhandle.execute(f"attach database '{archdb_fullname}' as archdb")
            try:
                cur.execute(f'insert or replace into archdb.candidates select * from candidates where ref_dt < {keep_dt.timestamp()}')
            except Exception:
                log.warning(f'unable to archive candidate database')
                purge_ok = False

        self.dbhandle.execute("detach database 'archdb'")
        if purge_ok:
            self.purgeCandDatabase(archdate_jd=archdate_jd)

        self.dbhandle.commit()
        return 

    def mergeCandDatabase(self, source_db_path):
        """
        merge in records from another observation database, for example from a remote node

        Parameters:
        source_db_path  : the full name of the source database from which to merge in records

        """

        if not os.path.isfile(source_db_path):
            log.warning(f'source database missing: {source_db_path}')
            return 
        # attach the other db, copy the records then detach it
        cur = self.dbhandle.execute(f"attach database '{source_db_path}' as sourcedb")

        status = True
        for table_name in ['candidates']:
            try:
                # bulk-copy if possible
                cur.execute(f'insert or replace into {table_name} select * from sourcedb.{table_name}')
            except Exception:
                log.warning(f'unable to merge data from {source_db_path}')
                status = False
        self.dbhandle.commit()
        cur.execute("detach database 'sourcedb'")
        return status
    

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

    arg_parser.add_argument("--logdir", type=str, default=None,
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
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"correlate_db_{timestamp}.log")
    file_handler = logging.handlers.TimedRotatingFileHandler(log_file, when="midnight", backupCount=7)
    file_handler.setFormatter(log_formatter)
    log.addHandler(file_handler)

    # Init the console handler (i.e. print to console)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    log.addHandler(console_handler)

    if cml_args.database:
        dbname = cml_args.database.lower()
    action = cml_args.action.lower()

    stmt = cml_args.stmt

    dt_range = None
    if cml_args.timerange is not None:
        time_beg, time_end = cml_args.timerange.strip("(").strip(")").split(",")
        dt_beg = datetime.datetime.strptime(time_beg, "%Y%m%d-%H%M%S").replace(tzinfo=datetime.timezone.utc)
        dt_end = datetime.datetime.strptime(time_end, "%Y%m%d-%H%M%S").replace(tzinfo=datetime.timezone.utc)
        log.info("Custom time range:")
        log.info("    BEG: {:s}".format(str(dt_beg)))
        log.info("    END: {:s}".format(str(dt_end)))
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
