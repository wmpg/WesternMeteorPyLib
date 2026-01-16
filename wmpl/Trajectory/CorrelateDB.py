""" Python scripts to manage the WMPL SQLite databases
"""
import os
import sqlite3
import logging
import logging.handlers
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


def addPairedObs(dbhandle,station_code, obs_id):
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
    dbhandle.commit()
    if not checkObsPaired(dbhandle, station_code, obs_id):
        log.info(f'failed to add {obs_id} to paired_obs table')
        return False
    return True


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
