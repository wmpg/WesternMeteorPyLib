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


def openObsDatabase(db_path, db_name='observations'):
    """
    openObsDatabase - open the observations sqlite database and return a database handle
                      
    The database is created if it doesn't exist.

    :param db_path: the path to the database
    :param db_name: the name of the database to open, default 'observations'
    :return: database handle 
    """

    db_full_name = os.path.join(db_path, f'{db_name}.db')
    log.info(f'opening database {db_full_name}')
    con = sqlite3.connect(db_full_name)
    cur = con.cursor()
    res = cur.execute("SELECT name FROM sqlite_master WHERE name='paired_obs'")
    if res.fetchone() is None:
        cur.execute("CREATE TABLE paired_obs(station_code VARCHAR(8), obs_id VARCHAR(36) UNIQUE, obs_date REAL, status INTEGER)")
    con.commit()
    cur.close()
    return con


def commitObsDatabase(dbhandle):
    """ commit the obs db 
    """
    dbhandle.commit()
    return 


def closeObsDatabase(dbhandle):
    dbhandle.commit()
    dbhandle.close()
    return 


def checkObsPaired(dbhandle, station_code, obs_id):
    """
    checkObsPaired - check if an observation is already paired

    :param dbhandle: the database
    :param station_code: the station ID
    :param obs_id; the observation id
    :return: true if matched, false otherwise

    """
    cur = dbhandle.cursor()
    res = cur.execute(f"SELECT obs_id FROM paired_obs WHERE station_code='{station_code}' and obs_id='{obs_id}' and status=1")
    if res.fetchone() is None:
        return False
    cur.close()
    return True


def addPairedObs(dbhandle,station_code, obs_id, obs_date, commitnow=True, verbose=False):
    """
    addPairedObs - add a potentially paired Observation to the database
    
    :param dbhandle: database connection handle
    :param station_code: station code eg UK12345
    :param obs_id: met_obs observation ID
    :param obs_date: observation date/time 
    :param commitnow: boolean true to force commit immediately

    :return: true if successful, false if the object already exists
    :rtype: bool
    """
    cur = dbhandle.cursor()
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
        dbhandle.commit()
    if not checkObsPaired(dbhandle, station_code, obs_id):
        log.warning(f'failed to add {obs_id} to paired_obs table')
        return False
    return True


def unpairObs(dbhandle, station_code, obs_id, verbose=False):
    """
    unpairObs - mark an observation unpaired by setting the status to zero

    :param dbhandle: the database
    :param station_code: the station ID
    :param obs_id; the observation id
    
    """

    cur = dbhandle.cursor()
    if verbose:
        log.info(f'unpairing {obs_id}')
    cur.execute(f"update paired_obs set status=0 where station_code='{station_code}' and obs_id='{obs_id}'")
    dbhandle.commit()
    cur.close()
    return True


def archiveObsDatabase(dbhandle, db_path, arch_prefix, archdate_jd):
    # create the database if it doesnt exist
    archdb_name = f'{arch_prefix}_observations'
    archdb = openObsDatabase(db_path, archdb_name)
    closeObsDatabase(archdb)

    # attach the arch db and copy the records then delete them
    cur = dbhandle.cursor()
    archdb_fullname = os.path.join(db_path, f'{archdb_name}.db')
    cur.execute(f"attach database '{archdb_fullname}' as archdb")
    try:
        cur.execute(f'insert into archdb.paired_obs select * from paired_obs where obs_date < {archdate_jd}')
    except Exception:
        log.info('Some records already exist in archdb, doing row-wise copy')
        cur.execute(f'select * from paired_obs where obs_date < {archdate_jd}')
        for row in cur.fetchall():
            try:
                cur.execute(f"insert into archdb.paired_obs values('{row[0]}','{row[1]}',{row[2]},{row[3]})")
            except Exception:
                log.info(f'{row[1]} already exists in target')

    cur.execute(f'delete from paired_obs where obs_date < {archdate_jd}')
    commitObsDatabase(dbhandle)
    cur.close()
    return 


def openTrajDatabase(db_path, db_name='processed_trajectories'):
    db_full_name = os.path.join(db_path, f'{db_name}.db')
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
        dbhandle = openObsDatabase(cml_args.dir_path)
        if action == 'read':
            cur = dbhandle.cursor()
            cur.execute('select * from paired_obs where status=1')
            print(f'there are {len(cur.fetchall())} paired obs')
            cur.execute('select * from paired_obs where status=0')
            print(f'and       {len(cur.fetchall())} unpaired obs')
        closeObsDatabase(dbhandle)
    elif dbname == 'trajectories':
        print('hello')
    else:
        log.info('valid database not specified')
