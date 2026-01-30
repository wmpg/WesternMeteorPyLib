""" Python scripts to manage the WMPL SQLite databases
"""
import os
import sqlite3
import logging
import logging.handlers
import argparse
import datetime
import json
import shutil

from wmpl.Utils.TrajConversions import datetime2JD, jd2Date


log = logging.getLogger("traj_correlator")

############################################################
# classes to handle the Observation and Trajectory databases
############################################################


class ObservationDatabase():

    # A class to handle the sqlite observations database transparently.

    def __init__(self, db_path, db_name='observations.db', purge_records=False):
        self.dbhandle = self.openObsDatabase(db_path, db_name, purge_records)
        
    def openObsDatabase(self, db_path, db_name='observations.db', purge_records=False):
        # Open the database, creating it and adding the required table if necessary.
        # If purge_records is true, delete any existing records. 

        db_full_name = os.path.join(db_path, f'{db_name}')
        log.info(f'opening database {db_full_name}')
        con = sqlite3.connect(db_full_name)
        cur = con.cursor()
        if purge_records:
            cur.execute('drop table paired_obs')
        cur.execute("SELECT name FROM sqlite_master WHERE name='paired_obs'")
        if cur.fetchone() is None:
            cur.execute("CREATE TABLE paired_obs(station_code VARCHAR(8), obs_id VARCHAR(36) UNIQUE, obs_date REAL, status INTEGER)")
        con.commit()
        cur.close()
        return con

    def commitObsDatabase(self):
        # commit the obs db. This function exists so we can do lazy writes in some cases

        self.dbhandle.commit()
        return 

    def closeObsDatabase(self):
        # close the database, making sure we commit any pending updates

        self.dbhandle.commit()
        self.dbhandle.close()
        self.dbhandle = None
        return 


    def checkObsPaired(self, station_code, obs_id):
        # return True if there is an observation with the correct station code, obs id and with status = 1 
        
        paired = True
        cur = self.dbhandle.cursor()
        cur.execute(f"SELECT obs_id FROM paired_obs WHERE obs_id='{obs_id}' and status=1")
        if cur.fetchone() is None:
            paired = False
        cur.close()
        return paired 


    def addPairedObs(self, station_code, obs_id, obs_date, verbose=False):
        # add or update an entry in the database, setting status = 1
        if verbose:
            log.info(f'adding {obs_id} to paired_obs table')
        cur = self.dbhandle.cursor()
        sqlstr = f"insert or replace into paired_obs values ('{station_code}','{obs_id}', {datetime2JD(obs_date)}, 1)"
        cur.execute(sqlstr)
        cur.close()

        if not self.checkObsPaired(station_code, obs_id):
            log.warning(f'failed to add {obs_id} to paired_obs table')
            return False
        return True


    def unpairObs(self, station_code, obs_id, verbose=False):
        # if an entry exists, update the status to 0. 
        # this allows us to mark an observation paired, then unpair it later if the solution fails
        # or we want to force a rerun. 
        if verbose:
            log.info(f'unpairing {obs_id}')

        cur = self.dbhandle.cursor()
        try:
            cur.execute(f"update paired_obs set status=0 where station_code='{station_code}' and obs_id='{obs_id}'")
            self.dbhandle.commit()
        except Exception:
            # obs wasn't in the database so no need to unpair it
            pass
        
        cur.close()
        return True


    def archiveObsDatabase(self, db_path, arch_prefix, archdate_jd):
        # archive records older than archdate_jd to a database {arch_prefix}_observations.db

        # create the database and table if it doesnt exist
        archdb_name = f'{arch_prefix}_observations.db'
        archdb = self.openObsDatabase(db_path, archdb_name)
        archdb.commit()
        archdb.close()

        # attach the arch db, copy the records then delete them
        cur = self.dbhandle.cursor()
        archdb_fullname = os.path.join(db_path, f'{archdb_name}')
        cur.execute(f"attach database '{archdb_fullname}' as archdb")
        try:
            # bulk-copy if possible
            cur.execute(f'insert or replace into archdb.paired_obs select * from paired_obs where obs_date < {archdate_jd}')
        except Exception:
            # otherwise, one by one 
            cur.execute(f'select * from paired_obs where obs_date < {archdate_jd}')
            for row in cur.fetchall():
                try:
                    cur.execute(f"insert into archdb.paired_obs values('{row[0]}','{row[1]}',{row[2]},{row[3]})")
                except Exception:
                    log.info(f'{row[1]} already exists in target')

        cur.execute(f'delete from paired_obs where obs_date < {archdate_jd}')
        self.commitObsDatabase()
        cur.close()
        return 

    def moveObsJsonRecords(self, paired_obs, dt_range):
        log.info('-----------------------------')
        log.info('moving recent observations to sqlite - this may take some time....')
        i = 0

        # only copy recent observations since if we ever run for an historic date
        # its likely we will want to reanalyse all available obs anyway
        dt_end = dt_range[1]
        dt_beg = max(dt_range[0], dt_end + datetime.timedelta(days=-7))

        keylist = paired_obs.keys()
        for stat_id in keylist:
            for obs_id in paired_obs[stat_id]:
                try:
                    obs_date = datetime.datetime.strptime(obs_id.split('_')[1], '%Y%m%d-%H%M%S.%f')
                except Exception:
                    obs_date = datetime.datetime(2000,1,1,0,0,0)
                obs_date = obs_date.replace(tzinfo=datetime.timezone.utc)
                
                if obs_date >= dt_beg and obs_date < dt_end:
                    self.addPairedObs(stat_id, obs_id, obs_date)
                i += 1
                if not i % 100000:
                    log.info(f'moved {i} observations')
        self.commitObsDatabase()
        log.info(f'done - moved {i} observations')
        log.info('-----------------------------')

        return 

    def mergeObsDatabase(self, source_db_path):
        # merge in records from another observation database, for example from a remote node

        if not os.path.isfile(source_db_path):
            log.warning(f'source database missing: {source_db_path}')
            return 
        # attach the other db, copy the records then detach it
        cur = self.dbhandle.cursor()
        cur.execute(f"attach database '{source_db_path}' as sourcedb")
        try:
            # bulk-copy if possible
            cur.execute('insert or replace into paired_obs select * from sourcedb.paired_obs')
        except Exception:
            # otherwise, one by one 
            log.info('Some records already exist, doing row-wise copy')
            cur.execute('select * from sourcedb.paired_obs')
            for row in cur.fetchall():
                self.addPairedObs(row[0], row[1],row[2])
        self.commitObsDatabase()
        cur.execute("detach database 'sourcedb'")
        cur.close()
        return 


############################################################

class DummyTrajReduced():
    # a dummy class for use in a couple of fuctions in the TrajectoryDatabase
    def __init__(self, jdt_ref, traj_id, traj_file_path):
        self.jdt_ref = jdt_ref
        self.traj_id = traj_id
        self.traj_file_path = traj_file_path


class TrajectoryDatabase():

    # A class to handle the sqlite trajectory database transparently.

    def __init__(self, db_path, db_name='trajectories.db', purge_records=False):
        self.dbhandle = self.openTrajDatabase(db_path, db_name, purge_records)
        
    def openTrajDatabase(self, db_path, db_name='trajectories.db', purge_records=False):
        # Open the database, creating it and adding the required table if necessary.
        # If purge_records is true, delete any existing records. 

        db_full_name = os.path.join(db_path, f'{db_name}')
        log.info(f'opening database {db_full_name}')
        con = sqlite3.connect(db_full_name)
        cur = con.cursor()
        if purge_records:
            cur.execute('drop table trajectories')
            cur.execute('drop table failed_trajectories')
        res = cur.execute("SELECT name FROM sqlite_master WHERE name='trajectories'")
        if res.fetchone() is None:
            cur.execute("""CREATE TABLE trajectories(
                        jdt_ref REAL UNIQUE,
                        traj_id VARCHAR UNIQUE,
                        traj_file_path VARCHAR,
                        participating_stations VARCHAR, 
                        radiant_eci_mini VARCHAR,
                        state_vect_mini VARCHAR,
                        ignored_stations VARCHAR,
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
                        status INTEGER) """)

        res = cur.execute("SELECT name FROM sqlite_master WHERE name='failed_trajectories'")
        if res.fetchone() is None:
            # note: traj_id not unique here as some fails will have traj-id None
            cur.execute("""CREATE TABLE failed_trajectories(
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
                        status INTEGER) """)
                        
        con.commit()
        cur.close()
        return con

    def commitTrajDatabase(self):
        # commit the obs db. This function exists so we can do lazy writes in some cases

        self.dbhandle.commit()
        return 

    def closeTrajDatabase(self):
        # close the database, making sure we commit any pending updates

        self.dbhandle.commit()
        self.dbhandle.close()
        self.dbhandle = None
        return 


    def checkTrajIfFailed(self, traj_reduced, verbose=False):
        # return True if there is an observation with the same jdt_ref and matching list of stations

        if not hasattr(traj_reduced, 'jdt_ref') or not hasattr(traj_reduced, 'participating_stations') or not hasattr(traj_reduced, 'ignored_stations'):
            return False
        
        found = False
        station_list = list(set(traj_reduced.participating_stations + traj_reduced.ignored_stations))
        cur = self.dbhandle.cursor()
        res = cur.execute(f"SELECT traj_id,participating_stations, ignored_stations FROM failed_trajectories WHERE jdt_ref={traj_reduced.jdt_ref} and status=1")
        row = res.fetchone()
        if row is None:
            found = False
        else:
            traj_stations = list(set(json.loads(row[1]) + json.loads(row[2])))
            found = True if (traj_stations == station_list) else False
        cur.close()
        return found

    def addTrajectory(self, traj_reduced, failed=False, verbose=False):
        # add or update an entry in the database, setting status = 1

        if verbose:
            log.info(f'adding jdt {traj_reduced.jdt_ref} to {"failed" if failed else "trajectories"}')
        cur = self.dbhandle.cursor()
        # remove the output_dir part from the path so that the data are location-independent
        traj_file_path = traj_reduced.traj_file_path[traj_reduced.traj_file_path.find('trajectories'):]

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
                        f"0,{v_init},{traj_reduced.gravity_factor},1)")
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
                        f"{traj_reduced.rend_lat},{traj_reduced.rend_lon},{traj_reduced.rend_ele},1)")

        sql_str = sql_str.replace('nan','"NaN"')
        cur.execute(sql_str)
        self.dbhandle.commit()
        cur.close()
        return True

    def removeTrajectory(self, traj_reduced, keepFolder=False, failed=False, verbose=False):
        # if an entry exists, update the status to 0. 
        # this allows us to mark an observation paired, then unpair it later if the solution fails
        # or we want to force a rerun. 
        if verbose:
            log.info(f'removing {traj_reduced.traj_id}')
        table_name = 'failed_trajectories' if failed else 'trajectories'

        cur = self.dbhandle.cursor()
        try:
            cur.execute(f"update {table_name} set status=0 where jdt_ref='{traj_reduced.jdt_ref}'")
            self.dbhandle.commit()
        except Exception:
            # traj wasn't in the database so no action required
            pass
        cur.close()

        # Remove the trajectory folder on the disk
        if not keepFolder and os.path.isfile(traj_reduced.traj_file_path):
            traj_dir = os.path.dirname(traj_reduced.traj_file_path)
            shutil.rmtree(traj_dir, ignore_errors=True)
            if os.path.isfile(traj_reduced.traj_file_path):
                log.info(f'unable to remove {traj_dir}')        

        return True

    
    def getTrajectories(self, output_dir, jdt_start, jdt_end=None, failed=False, verbose=False):

        table_name = 'failed_trajectories' if failed else 'trajectories'
        if verbose:
            log.info(f'getting trajectories between {jd2Date(jdt_start, dt_obj=True).strftime("%Y%m%d_%M%M%S.%f")} and {jd2Date(jdt_end, dt_obj=True).strftime("%Y%m%d_%M%M%S.%f")}')

        cur = self.dbhandle.cursor()
        if not jdt_end:
            cur.execute(f"SELECT * FROM {table_name} WHERE jdt_ref={jdt_start}")
            rows = cur.fetchall()
        else:
            cur.execute(f"SELECT * FROM {table_name} WHERE jdt_ref>={jdt_start} and jdt_ref<={jdt_end}")
            rows = cur.fetchall()
        cur.close()
        trajs = []
        for rw in rows:
            json_dict = {'jdt_ref':rw[0], 'traj_id':rw[1], 'traj_file_path':os.path.join(output_dir, rw[2]),
                         'participating_stations': json.loads(rw[3]),
                         'ignored_stations': json.loads(rw[4]),
                         'radiant_eci_mini': json.loads(rw[5]),
                         'state_vect_mini': json.loads(rw[6]),
                         'phase_1_only': rw[7], 'v_init': rw[8],'gravity_factor': rw[9],
                         'v0z': rw[10], 'v_avg': rw[11], 
                         'rbeg_jd': rw[12], 'rend_jd': rw[13], 
                         'rbeg_lat': rw[14], 'rbeg_lon': rw[15], 'rbeg_ele': rw[16], 
                         'rend_lat': rw[17], 'rend_lon': rw[18], 'rend_ele': rw[19]                        
                         }
            
            trajs.append(json_dict)
        return trajs


    def removeDeletedTrajectories(self, output_dir, jdt_start, jdt_end=None, failed=False, verbose=False):

        table_name = 'failed_trajectories' if failed else 'trajectories'
        if verbose:
            log.info(f'getting trajectories between {jdt_start} and {jdt_end}')

        cur = self.dbhandle.cursor()
        if not jdt_end:
            cur.execute(f"SELECT * FROM {table_name} WHERE jdt_ref={jdt_start}")
            rows = cur.fetchall()
        else:
            cur.execute(f"SELECT * FROM {table_name} WHERE jdt_ref>={jdt_start} and jdt_ref<={jdt_end}")
            rows = cur.fetchall()
        cur.close()
        i = 0 
        for rw in rows:
            if not os.path.isfile(os.path.join(output_dir, rw[2])):
                if verbose:
                    log.info(f'removing traj {jd2Date(rw[0], dt_obj=True).strftime("%Y%m%d_%M%M%S.%f")} from database')
                self.removeTrajectory(DummyTrajReduced(rw[0], rw[1], rw[2]), keepFolder=True)
                i += 1
        log.info(f'removed {i} deleted trajectories')
        return 


    def archiveTrajDatabase(self, db_path, arch_prefix, archdate_jd):
        # archive records older than archdate_jd to a database {arch_prefix}_trajectories.db

        # create the database and table if it doesnt exist
        archdb_name = f'{arch_prefix}_trajectories.db'
        archdb = self.openObsDatabase(db_path, archdb_name)
        archdb.commit()
        archdb.close()

        # attach the arch db, copy the records then delete them
        cur = self.dbhandle.cursor()
        archdb_fullname = os.path.join(db_path, f'{archdb_name}')
        cur.execute(f"attach database '{archdb_fullname}' as archdb")
        for table_name in ['trajectories', 'failed_trajectories']:
            try:
                # bulk-copy if possible
                cur.execute(f'insert or replace into archdb.{table_name} select * from {table_name} where jdt_ref < {archdate_jd}')
                cur.execute(f'delete from {table_name} where jdt_ref < {archdate_jd}')
            except Exception:
                log.warning(f'unable to archive {table_name}')

        self.commitTrajDatabase()
        cur.close()
        return 

    def moveFailedTrajectories(self, failed_trajectories, dt_range):
        log.info('moving recent trajectories to sqlite - this may take some time....')

        # only copy recent records since if we ever run for an historic date
        # its likely we will want to reanalyse all available obs anyway
        jd_end = datetime2JD(dt_range[1])
        jd_beg = max(datetime2JD(dt_range[0]), jd_end - 7)

        keylist = [k for k in failed_trajectories.keys() if float(k) >= jd_beg and float(k) <= jd_end]
        i = 0 # just in case there aren't any trajectories to move
        for i,jdt_ref in enumerate(keylist):
            self.addTrajectory(failed_trajectories[jdt_ref], failed=True)
            i += 1
            if not i % 10000:
                self.commitTrajDatabase()
                log.info(f'moved {i} failed_trajectories')
        self.commitTrajDatabase()
        log.info(f'done - moved {i} failed_trajectories')

        return 
    
    def mergeTrajDatabase(self, source_db_path):
        # merge in records from another observation database, for example from a remote node

        if not os.path.isfile(source_db_path):
            log.warning(f'source database missing: {source_db_path}')
            return 
        # attach the other db, copy the records then detach it
        cur = self.dbhandle.cursor()
        cur.execute(f"attach database '{source_db_path}' as sourcedb")

        # TODO need to correct the traj_file_path to account for server locations

        for table_name in ['trajectories', 'failed_trajectories']:
            try:
                # bulk-copy if possible
                cur.execute(f'insert or replace into {table_name} select * from sourcedb.{table_name}')
            except Exception:
                log.warning(f'unable to merge data from {source_db_path}')
        self.commitTrajDatabase()
        cur.execute("detach database 'sourcedb'")
        cur.close()
        return 


############################################################

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="""Automatically compute trajectories from RMS data in the given directory.""",
        formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('--dir_path', type=str, default=None, help='Path to the directory containing the databases.')

    arg_parser.add_argument('--database', type=str, default=None, help='Database to process, either observations or trajectories')

    arg_parser.add_argument('--action', type=str, default=None, help='Action to take on the database')

    arg_parser.add_argument("--logdir", type=str, default=None,
        help="Path to the directory where the log files will be stored. If not given, a logs folder will be created in the database folder")
      
    arg_parser.add_argument('-r', '--timerange', metavar='TIME_RANGE',
        help="""Apply action to this date range in the format: "(YYYYMMDD-HHMMSS,YYYYMMDD-HHMMSS)".""", type=str)
    
    cml_args = arg_parser.parse_args()
    # Find the log directory
    log_dir = cml_args.logdir
    if log_dir is None:
        log_dir = os.path.join(cml_args.dir_path, 'logs')
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
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

    dbname = cml_args.database.lower()
    action = cml_args.action.lower()

    if dbname == 'observations':
        obsdb = ObservationDatabase(cml_args.dir_path)
        if action == 'read':
            cur = obsdb.dbhandle.cursor()
            cur.execute('select * from paired_obs where status=1')
            print(f'there are {len(cur.fetchall())} paired obs')
            cur.execute('select * from paired_obs where status=0')
            print(f'and       {len(cur.fetchall())} unpaired obs')
        obsdb.closeObsDatabase()
    elif dbname == 'trajectories':
        trajdb = TrajectoryDatabase(cml_args.dir_path)
        if action == 'read':
            cur = trajdb.dbhandle.cursor()
            cur.execute('select * from trajectories where status=1')
            print(f'there are {len(cur.fetchall())} successful trajectories')
            cur.execute('select * from failed_trajectories')
            print(f'and       {len(cur.fetchall())} failed trajectories')
    else:
        log.info('valid database not specified')
