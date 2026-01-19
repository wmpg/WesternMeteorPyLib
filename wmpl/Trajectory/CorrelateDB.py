""" Python scripts to manage the WMPL SQLite databases
"""
import os
import sqlite3
import logging
import logging.handlers
import argparse
import datetime
log = logging.getLogger("traj_correlator")


def openObsDatabase(db_path, db_name='observations'):
    db_full_name = os.path.join(db_path, f'{db_name}.db')
    log.info(f'opening database {db_full_name}')
    con = sqlite3.connect(db_full_name)
    cur = con.cursor()
    res = cur.execute("SELECT name FROM sqlite_master WHERE name='paired_obs'")
    if res.fetchone() is None:
        cur.execute("CREATE TABLE paired_obs(station_code, obs_id unique, status)")
    con.commit()
    return con


def closeObsDatabase(dbhandle):
    dbhandle.commit()
    dbhandle.close()
    return 


def checkObsPaired(dbhandle, station_code, obs_id):
    cur = dbhandle.cursor()
    res = cur.execute(f"SELECT obs_id FROM paired_obs WHERE station_code='{station_code}' and obs_id='{obs_id}' and status=1")
    if res.fetchone() is None:
        return False
    return True


def addPairedObs(dbhandle,station_code, obs_id, commitnow=True):
    """
    addPairedObs - add a potentially paired Observation to the database
    
    :param dbhandle: database connection handle
    :param station_code: station code eg UK12345
    :param obs_id: met_obs observation ID
    :return: true if successful, false if the object already exists
    :rtype: bool
    """
    cur = dbhandle.cursor()
    res = cur.execute(f"SELECT obs_id FROM paired_obs WHERE station_code='{station_code}' and obs_id='{obs_id}'")
    if res.fetchone() is None:
        log.info(f'adding {obs_id} to paired_obs table')
        sqlstr = f"insert into paired_obs values ('{station_code}','{obs_id}',1)"
    else:
        log.info(f'updating {obs_id} in paired_obs table')
        sqlstr = f"update paired_obs set status=1 where station_code='{station_code}' and obs_id='{obs_id}'"
    cur.execute(sqlstr)
    if commitnow:
        dbhandle.commit()
    if not checkObsPaired(dbhandle, station_code, obs_id):
        log.info(f'failed to add {obs_id} to paired_obs table')
        return False
    return True


def commitObsDb(dbhandle):
    """ commit the obs db, called only during initialisation    
    """
    dbhandle.commit()
    return 


def unpairObs(dbhandle, station_code, obs_id):
    cur = dbhandle.cursor()
    cur.execute(f"update paired_obs set status=0 where station_code='{station_code}' and obs_id='{obs_id}'")
    dbhandle.commit()
    return True


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
