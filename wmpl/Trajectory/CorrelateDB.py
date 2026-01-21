""" Python scripts to manage the WMPL SQLite databases
"""
import os
import sqlite3
import logging
import logging.handlers
import argparse
import datetime

from wmpl.Utils.TrajConversions import datetime2JD

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
        res = cur.execute("SELECT name FROM sqlite_master WHERE name='paired_obs'")
        if res.fetchone() is None:
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
        return 


    def checkObsPaired(self, station_code, obs_id):
        # return True if there is an observation with the correct station code, obs id and with status = 1 
        
        paired = True
        cur = self.dbhandle.cursor()
        res = cur.execute(f"SELECT obs_id FROM paired_obs WHERE station_code='{station_code}' and obs_id='{obs_id}' and status=1")
        if res.fetchone() is None:
            paired = False
        cur.close()
        return paired 


    def addPairedObs(self, station_code, obs_id, obs_date, commitnow=True, verbose=False):
        # add or update an entry in the database, setting status = 1
        cur = self.dbhandle.cursor()
        res = cur.execute(f"SELECT obs_id FROM paired_obs WHERE station_code='{station_code}' and obs_id='{obs_id}'")
        if res.fetchone() is None:
            if verbose:
                log.info(f'adding {obs_id} to paired_obs table')
            sqlstr = f"insert into paired_obs values ('{station_code}','{obs_id}', {datetime2JD(obs_date)}, 1)"
        else:
            if verbose:
                log.info(f'updating {obs_id} in paired_obs table')
            sqlstr = f"update paired_obs set status=1 where station_code='{station_code}' and obs_id='{obs_id}'"
        cur.execute(sqlstr)
        cur.close()

        if commitnow:
            self.dbhandle.commit()
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

    def moveJsonRecords(self, paired_obs):
        log.info('-----------------------------')
        log.info('moving observations to sqlite - this may take some time....')
        i = 0
        keylist = paired_obs.keys()
        for stat_id in keylist:
            for obs_id in paired_obs[stat_id]:
                try:
                    obs_date = datetime.datetime.strptime(obs_id.split('_')[1], '%Y%m%d-%H%M%S.%f')
                except Exception:
                    obs_date = datetime.datetime(2000,1,1,0,0,0)
                self.addPairedObs(stat_id, obs_id, obs_date, commitnow=False)
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

def openTrajDatabase(db_path, db_name='processed_trajectories.db'):
    db_full_name = os.path.join(db_path, f'{db_name}')
    log.info(f'opening database {db_full_name}')
    con = sqlite3.connect(db_full_name)
    cur = con.cursor()
    res = cur.execute("SELECT name FROM sqlite_master WHERE name='failed_trajectories'")
    if res.fetchone() is None:
        cur.execute("CREATE TABLE failed_trajectories()")

    res = cur.execute("SELECT name FROM sqlite_master WHERE name='trajectories'")
    if res.fetchone() is None:
        cur.execute("CREATE TABLE trajectories()")
    con.commit()
    return con


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
        print('hello')
    else:
        log.info('valid database not specified')
