# The MIT License

# Copyright (c) 2024 Mark McIntyre

# various tests for the CorrelateDB classes

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

import os
import shutil
import datetime
from wmpl.Trajectory.CorrelateDB import TrajectoryDatabase, CandidateDatabase, ObservationsDatabase
from wmpl.Trajectory.CorrelateRMS import TrajectoryReduced
from wmpl.Utils.TrajConversions import date2JD, datetime2JD
import sqlite3
import time
import threading
from zipfile import ZipFile

dbloc = os.path.split(__file__)[0]
jdt_beg = date2JD(2026,7,2,0,0,0)
jdt_end = date2JD(2026,7,15,0,0,0)
exact_jdt_succ = 2461236.5589184340
exact_jdt_fail = 2461245.5424379872
exact_jdt_fail_1 = 2461228.5865266793
test_traj_id = '20260718225156_h23G2'
failed_traj_id = '20260712222653_4D7Xl'
loaded_traj_path = '20260712_222508_trajectory.pickle'
loaded_traj_id = '20260712222508_r89he'

# extract the test data from the zip archive
with ZipFile(os.path.join(dbloc,'test_dbs.zip'),'r') as zip_ref:
    zip_ref.extractall(dbloc)

###################################################
## Candidate database tests

def test_CandDb():
    cdb = CandidateDatabase(dbloc, 'test_cands.db')
    cand_id = '1784513127.942391_UK'
    ref_dt = 1784513127.942391
    obs_ids = ["UK00A0_20260720-020528.036771_4176", "UK005M_20260720-020528.102589_3959", "UK00AN_20260720-020528.099155_2335"]

    # cand already exists
    assert cdb.checkAndAddCand(cand_id, ref_dt, obs_ids) is False

    obslist = cdb.getCandidateObs(cand_id)
    assert len(obslist) == 3
    assert 'UK00A0_20260720-020528.036771_4176' in obslist

    # new cand
    cand_id = 'abc123456'
    ref_dt = 12345
    obs_ids = ['a','b']
    assert cdb.checkAndAddCand(cand_id, ref_dt, obs_ids) is True

    obslist = cdb.getCandidateObs(cand_id)
    assert len(obslist) == 2
    assert 'a' in obslist
   
    # cleanup
    cdb.dbhandle.execute(f"delete from candidates where cand_id = '{cand_id}'")

    # check nonexistent candidate
    obslist = cdb.getCandidateObs('potato_123')
    assert len(obslist) == 0

    # tests for setting, unsetting and checking the cand's processing status
    real_cand_ids = ['1784502166.566642_UK', '1784499242.627764_UK.pickle', '/tmp/candidates/1784499242.627764_UK.pickle']
    for real_cand_id in real_cand_ids:
        assert cdb.isBeingProcessed(real_cand_id) is False
        
        assert cdb.markBeingProcessed(real_cand_id)
        assert cdb.isBeingProcessed(real_cand_id) is True

        assert cdb.unmarkBeingProcessed(real_cand_id)
        assert  cdb.isBeingProcessed(real_cand_id) is False

    # also test with nonsense ID
    bad_cand_id = r"foo#'%4$#111"

    assert cdb.isBeingProcessed(bad_cand_id) is False
    assert cdb.markBeingProcessed(bad_cand_id) is False
    assert cdb.unmarkBeingProcessed(bad_cand_id) is False

    cdb.closeCandDatabase()

def test_archAndPurgeCandDb():
    shutil.copyfile(os.path.join(dbloc, 'test_cands.db'), os.path.join(dbloc, 'candidates.db'))
    cdb = CandidateDatabase(dbloc, 'candidates.db')
    rws = cdb.dbhandle.execute('select ref_dt from candidates order by ref_dt asc').fetchall()
    arch_dt = (float(rws[-1][0])+float(rws[0][0]))/2.0
    arch_jd = datetime2JD(datetime.datetime.fromtimestamp(arch_dt))
    sts = cdb.archiveCandDatabase(dbloc, 'arch', arch_jd)
    rws = cdb.dbhandle.execute('select count(*) from candidates').fetchall()
    assert rws[0][0] == 52
    cdb.closeCandDatabase()
    cdb = CandidateDatabase(dbloc, 'arch_candidates.db')
    rws = cdb.dbhandle.execute('select count(*) from candidates').fetchall()
    assert rws[0][0] == 48
    cdb.closeCandDatabase()
    os.remove(os.path.join(dbloc, 'candidates.db'))
    os.remove(os.path.join(dbloc, 'arch_candidates.db'))

    shutil.copyfile(os.path.join(dbloc, 'test_cands.db'), os.path.join(dbloc, 'candidates.db'))
    cdb = CandidateDatabase(dbloc, 'candidates.db')
    sts = cdb.archiveCandDatabase(dbloc, None, arch_jd)
    rws = cdb.dbhandle.execute('select count(*) from candidates').fetchall()
    assert rws[0][0] == 52
    cdb.closeCandDatabase()
    os.remove(os.path.join(dbloc, 'candidates.db'))

def test_mergeCandDb():
    shutil.copyfile(os.path.join(dbloc, 'test_cands.db'), os.path.join(dbloc, 'candidates.db'))
    cdb = CandidateDatabase(dbloc, 'candidates.db')
    assert cdb.mergeCandDatabase(os.path.join(dbloc, 'test_mc.db'))
    rws = cdb.dbhandle.execute('select count(*) from candidates').fetchall()
    assert rws[0][0] == 131
    cdb.closeCandDatabase()
    os.remove(os.path.join(dbloc, 'candidates.db'))

###################################################
## Observations database tests

def test_PairinAndUnpairingObs():
    odb = ObservationsDatabase(dbloc, 'test_obs.db')
    # test checking for a nonexistent and real already-paired observation 
    assert odb.checkObsPaired('abcdedgf') is False
    assert odb.checkObsPaired('UK00A5_20260702-225405.573159_1605')

    # special characters, bad values and sql-injection type stuff
    assert odb.checkObsPaired("abcd'edgf") is False
    assert odb.checkObsPaired(r"abcd,edgf") is False
    assert odb.checkObsPaired(r"abcd%edgf") is False
    assert odb.checkObsPaired(r"abcd;drop table banana") is False

    # test unpairing a nonsense value and then a two real observations
    assert odb.unpairObs(['nonsense_value']) 
    assert odb.unpairObs(['UK009K_20260713-022400.679677_6047', 'UK009K_20260712-012935.530139_4333'])

    # confirm the obs got unpaired
    assert odb.checkObsPaired('UK009K_20260713-022400.679677_6047') is False
    assert odb.checkObsPaired('UK009K_20260712-012935.530139_4333') is False

    # now add them back 
    odb.addPairedObservations(['UK009K_20260713-022400.679677_6047', 'UK009K_20260712-012935.530139_4333'], 
                              [2461234.6000050465,2461233.562214929])

    # confirm the obs got re=paired
    assert odb.checkObsPaired('UK009K_20260713-022400.679677_6047') 
    assert odb.checkObsPaired('UK009K_20260712-012935.530139_4333')

    # special characters, bad values and sql-injection type stuff
    assert odb.unpairObs([r"foo'bar"]) 
    assert odb.unpairObs([r"foo,bar"]) 
    assert odb.unpairObs([r"foo%bar"]) 
    assert odb.unpairObs([r"foo;bar"]) 
    assert odb.unpairObs([r"foo;drop table banana"]) 

    # check we can retrieve linked obs with a real jdt
    vals = odb.getLinkedObservations(2461234.6000050465)
    assert len(vals) == 2
    assert 'UK009K_20260713-022400.679677_6047' in vals

    # and a nonexistent observation set
    vals = odb.getLinkedObservations(2461234.6)
    assert len(vals) == 0

    odb.closeObsDatabase()


def test_createObsDatabase():
    odb = ObservationsDatabase(dbloc, 'dummy.db')
    assert os.path.isfile(os.path.join(dbloc, 'dummy.db'))
    odb.closeObsDatabase()
    os.remove(os.path.join(dbloc, 'dummy.db')) 
    return 


def test_ArchAndPurgeObsDb():

    # test archiving half the data
    shutil.copyfile(os.path.join(dbloc, 'test_obs.db'), os.path.join(dbloc, 'observations.db'))
    cdb = ObservationsDatabase(dbloc, 'observations.db')
    rws = cdb.dbhandle.execute('select obs_dt from paired_obs order by obs_dt asc').fetchall()
    arch_jd = (float(rws[-1][0])+float(rws[0][0]))/2.0
    assert cdb.archiveObsDatabase(dbloc, 'arch', arch_jd)
    rws = cdb.dbhandle.execute('select count(*) from paired_obs').fetchall()
    assert rws[0][0] == 74
    cdb.closeObsDatabase()
    cdb = ObservationsDatabase(dbloc, 'arch_observations.db')
    rws = cdb.dbhandle.execute('select count(*) from paired_obs').fetchall()
    assert rws[0][0] == 26
    cdb.closeObsDatabase()
    os.remove(os.path.join(dbloc, 'observations.db'))
    os.remove(os.path.join(dbloc, 'arch_observations.db'))

    # test passing None for arch_prefix which purges without archiving
    shutil.copyfile(os.path.join(dbloc, 'test_obs.db'), os.path.join(dbloc, 'observations.db'))
    cdb = ObservationsDatabase(dbloc, 'observations.db')
    assert cdb.archiveObsDatabase(dbloc, None, arch_jd)
    rws = cdb.dbhandle.execute('select count(*) from paired_obs').fetchall()
    assert rws[0][0] == 74
    cdb.closeObsDatabase()
    os.remove(os.path.join(dbloc, 'observations.db'))

def test_mergeObsDb():
    shutil.copyfile(os.path.join(dbloc, 'test_obs.db'), os.path.join(dbloc, 'observations.db'))
    cdb = ObservationsDatabase(dbloc, 'observations.db')
    assert cdb.mergeObsDatabase(os.path.join(dbloc, 'test_mo.db'))
    rws = cdb.dbhandle.execute('select count(*) from paired_obs').fetchall()
    assert rws[0][0] == 146
    cdb.closeObsDatabase()
    os.remove(os.path.join(dbloc, 'observations.db'))

############################################
# Trajectories Database Tests

def reset_trajdb(trajdb):
    # delete any previously-added or modified traj just in case
    trajdb.dbhandle.execute(f"delete from trajectories where traj_id = '{loaded_traj_id}'")
    trajdb.dbhandle.execute(f"delete from failed_trajectories where traj_id = '{loaded_traj_id}'")
    trajdb.dbhandle.execute(f"update trajectories set status=1 where jdt_ref = {exact_jdt_succ}")
    trajdb.dbhandle.execute(f"update failed_trajectories set status=1 where jdt_ref = {exact_jdt_fail}")

    trajdb.dbhandle.execute(f"delete from failed_trajectories where traj_id = '{loaded_traj_id}'")
    trajdb.dbhandle.execute(f"delete from trajectories where traj_id = '{loaded_traj_id}'")
    trajdb.dbhandle.execute('commit')

def test_createTrajDatabase():
    odb = TrajectoryDatabase(dbloc, 'dummy.db')
    assert os.path.isfile(os.path.join(dbloc, 'dummy.db'))
    odb.closeTrajDatabase()
    os.remove(os.path.join(dbloc, 'dummy.db')) 
    return 

def test_getTrajectories():
    trajdb = TrajectoryDatabase(dbloc, 'test_traj.db')
    reset_trajdb(trajdb)

    # single date, non-failed, not including logically deleted
    trajs = trajdb.getTrajectories('.',[jdt_beg, None], failed=False, inc_deleted=False, verbose=False)
    assert len(trajs) == 45

    # single date, non-failed, including logically deleted
    trajs = trajdb.getTrajectories('.',[jdt_beg, None], failed=False, inc_deleted=True, verbose=False)
    assert len(trajs) == 47

    # two dates, non-failed, not including logically deleted
    trajs = trajdb.getTrajectories('.',[jdt_beg, jdt_end], failed=False, inc_deleted=False, verbose=False)
    assert len(trajs) == 30

    # two dates, non-failed,  including logically deleted
    trajs = trajdb.getTrajectories('.',[jdt_beg, jdt_end], failed=False, inc_deleted=True, verbose=False)
    assert len(trajs) == 31

    # failed traj, one date, ignoring logically deleted (which should never appear in this table)
    trajs = trajdb.getTrajectories('.',[jdt_beg, None], failed=True, inc_deleted=False, verbose=False)
    assert len(trajs) == 48

    # failed traj, two dates, ignoring logically deleted (which never appear in this table)
    trajs = trajdb.getTrajectories('.',[jdt_beg, jdt_end], failed=True, inc_deleted=False, verbose=False)
    assert len(trajs) == 15

    #  try inverted dates
    trajs = trajdb.getTrajectories('.',[jdt_end, jdt_beg], failed=False, inc_deleted=False, verbose=False)
    assert len(trajs) == 0

    #  try two bad dates in the far past
    trajs = trajdb.getTrajectories('.',[1, 2], failed=False, inc_deleted=False, verbose=False)
    assert len(trajs) == 0

    #  try one date in the future
    trajs = trajdb.getTrajectories('.',[5555555, None], failed=False, inc_deleted=False, verbose=False)
    assert len(trajs) == 0

    #  non-numeric and hackery
    trajs = trajdb.getTrajectories('.',["jdt_beg", None], failed=False, inc_deleted=False, verbose=False)
    assert len(trajs) == 0
    trajs = trajdb.getTrajectories('.',["1;drop table foo", None], failed=False, inc_deleted=False, verbose=False)
    assert len(trajs) == 0

    trajdb.closeTrajDatabase()

def test_ProcessingFlags():

    trajdb = TrajectoryDatabase(dbloc, 'test_traj.db')
    reset_trajdb(trajdb)

    # tests for setting, unsetting and checking the cand's processing status
    real_traj_id  = '20260719003629_16xBY'
    assert trajdb.isBeingProcessed(real_traj_id) is False
    
    res = trajdb.markBeingProcessed(real_traj_id)
    assert trajdb.isBeingProcessed(real_traj_id) is True

    res = trajdb.unmarkBeingProcessed(real_traj_id)
    assert  trajdb.isBeingProcessed(real_traj_id) is False

    # also test with nonsense ID
    bad_cand_id = r"foo#'%4$#111"

    assert trajdb.isBeingProcessed(bad_cand_id) is False
    assert trajdb.markBeingProcessed(bad_cand_id) is False
    assert trajdb.unmarkBeingProcessed(bad_cand_id) is False

    trajdb.closeTrajDatabase()

############################################
# tests for getTrajBasics
def test_getTrajBasics():
    trajdb = TrajectoryDatabase(dbloc, 'test_traj.db')
    reset_trajdb(trajdb)
    # retrieve from success table
    rows = trajdb.getTrajBasics('.', [None,None])
    assert len(rows)==47
    rows = trajdb.getTrajBasics('.', [exact_jdt_succ, None])
    assert len(rows)==1
    rows = trajdb.getTrajBasics('.', [jdt_beg, jdt_end])
    assert len(rows)==30

    # retrieve from failed table
    rows = trajdb.getTrajBasics('.', [None,None],failed=True)
    assert len(rows)==48
    rows = trajdb.getTrajBasics('.', [exact_jdt_fail, None],failed=True)
    assert len(rows)==1
    rows = trajdb.getTrajBasics('.', [jdt_beg, jdt_end],failed=True)
    assert len(rows)==15

    # a variety of bad dates, dates backwards, far future, far past
    rows = trajdb.getTrajBasics('.', [jdt_end, jdt_beg])
    assert len(rows)==0
    rows = trajdb.getTrajBasics('.', [5555555, None])
    assert len(rows)==0
    rows = trajdb.getTrajBasics('.', [1, 2])
    assert len(rows)==0

    # and some non-numeric or dangerous values
    rows = trajdb.getTrajBasics('.', ['bob', None])
    assert len(rows)==0
    rows = trajdb.getTrajBasics('.', ['bob', r"12345;delete foo from bar"])
    assert len(rows)==0

    trajdb.closeTrajDatabase()

############################################
# tests for checkTrajIfFailed
def test_checkTrajIfFailed():
    trajdb = TrajectoryDatabase(dbloc, 'test_traj.db')
    reset_trajdb(trajdb)

    # test for a failed traj with status = 1 (not logically deleted)
    trajs = trajdb.getTrajectories('.',[exact_jdt_fail_1, exact_jdt_fail_1], failed=True, inc_deleted=False, verbose=False)
    assert len(trajs) == 1

    traj_reduced = TrajectoryReduced(None, json_dict=trajs[0])
    res = trajdb.checkTrajIfFailed(traj_reduced, verbose=True)
    assert res is True

    # test for a failed traj, including logically deleted
    trajs = trajdb.getTrajectories('.',[exact_jdt_fail, exact_jdt_fail], failed=True, inc_deleted=True, verbose=False)
    assert len(trajs) == 1

    traj_reduced = TrajectoryReduced(None, json_dict=trajs[0])
    res = trajdb.checkTrajIfFailed(traj_reduced, verbose=True)
    assert res is True

    # now test for a non-failed traj
    trajs = trajdb.getTrajectories('.',[exact_jdt_succ, exact_jdt_succ], failed=False, inc_deleted=False, verbose=False)
    assert len(trajs) == 1

    traj_reduced = TrajectoryReduced(None, json_dict=trajs[0])
    res = trajdb.checkTrajIfFailed(traj_reduced, verbose=True)
    assert res is False

    # now test for a nonexistent traj - bad data
    json_data = {'no_data':'none'}
    traj_reduced = TrajectoryReduced(None, json_dict=json_data)
    res = trajdb.checkTrajIfFailed(traj_reduced)
    assert res is False

    # jdt_ref is not in the table
    json_data = {'jdt_ref':exact_jdt_fail-0.0001, 'participating_stations':'', 'ignored_stations':''}
    traj_reduced = TrajectoryReduced(None, json_dict=json_data)
    res = trajdb.checkTrajIfFailed(traj_reduced)
    assert res is False

    # jdt_ref is good but participating or ignored stations dont match
    json_data = {'jdt_ref':exact_jdt_fail, 'participating_stations':'UK0005,UK002F', 'ignored_stations':'UK000S'}
    traj_reduced = TrajectoryReduced(None, json_dict=json_data)
    res = trajdb.checkTrajIfFailed(traj_reduced)
    assert res is False

    trajdb.closeTrajDatabase()

############################################
# tests for addTrajectory
def test_addTrajectory():
    trajdb = TrajectoryDatabase(dbloc, 'test_traj.db')
    reset_trajdb(trajdb)

    # add to successful table
    traj_reduced = TrajectoryReduced(traj_file_path=os.path.join(dbloc, loaded_traj_path))
    assert trajdb.addTrajectory(traj_reduced, failed=False) == True
    
    cur = trajdb.dbhandle.execute(f"select count(*) from trajectories where traj_id = '{traj_reduced.traj_id}'")
    dta = cur.fetchall()
    numrows = dta[0][0]
    assert numrows == 1

    # add malformed trajectory with missing elements
    json_data = {'jdt_ref':exact_jdt_fail, 'participating_stations':['UK0005,UK002F'], 'ignored_stations':['UK000S']}
    traj_reduced = TrajectoryReduced(None, json_dict=json_data)
    assert trajdb.addTrajectory(traj_reduced, failed=False) == False

    # trying add to failed table
    traj_failed = TrajectoryReduced(traj_file_path=os.path.join(dbloc, loaded_traj_path))
    assert trajdb.addTrajectory(traj_failed, failed=True) == True

    res = trajdb.checkTrajIfFailed(traj_failed)
    assert res is True

    # add malformed traj to the table with bad data or hackery
    json_data = {'jdt_ref':exact_jdt_fail, 'participating_stations':['UK0005,UK002F'], 'ignored_stations':['UK000S']}
    traj_reduced = TrajectoryReduced(None, json_dict=json_data)
    assert trajdb.addTrajectory(traj_reduced, failed=False) == False

    json_data = {'jdt_ref':"notadate", 'participating_stations':['UK0005,UK002F'], 'ignored_stations':['UK000S']}
    traj_reduced = TrajectoryReduced(None, json_dict=json_data)
    assert trajdb.addTrajectory(traj_reduced, failed=False) == False

    json_data = {'jdt_ref':"2345678;drop table foo", 'participating_stations':['UK0005,UK002F'], 'ignored_stations':['UK000S']}
    traj_reduced = TrajectoryReduced(None, json_dict=json_data)
    assert trajdb.addTrajectory(traj_reduced, failed=False) == False

    trajdb.closeTrajDatabase()

############################################
# tests for removeTrajectoryById
def test_removeTrajectoryById():
    trajdb = TrajectoryDatabase(dbloc, 'test_traj.db')
    reset_trajdb(trajdb)
    
    # test a real successful trajectory
    trajdb.removeTrajectoryById(f'{test_traj_id}')
    cur = trajdb.dbhandle.execute(f"select count(*) from trajectories where traj_id = '{test_traj_id}' and status=0")
    dta = cur.fetchall()
    numrows = dta[0][0]
    assert numrows == 1

    # and from the failed table
    trajdb.removeTrajectoryById(failed_traj_id, failed=True)
    cur = trajdb.dbhandle.execute(f"select count(*) from failed_trajectories where traj_id = '{failed_traj_id}' and status=0")
    dta = cur.fetchall()
    numrows = dta[0][0]
    assert numrows == 1

    # and passing None - this returns False
    assert trajdb.removeTrajectoryById(None) is False

    # test for removing a nonexistent or illegal ID - this will return True
    assert trajdb.removeTrajectoryById('20260712123456_ZZZZZZZ') 
    assert trajdb.removeTrajectoryById(r'20260712%123456_ZZZZZZZ') 
    assert trajdb.removeTrajectoryById(r'20260712;drop table frobozz') 

    trajdb.closeTrajDatabase()


def test_archAndPurgeTrajDb():
    shutil.copyfile(os.path.join(dbloc, 'test_traj.db'), os.path.join(dbloc, 'trajectories.db'))
    cdb = TrajectoryDatabase(dbloc, 'trajectories.db')
    rws = cdb.dbhandle.execute('select jdt_ref from trajectories order by jdt_ref asc').fetchall()
    arch_jd = (float(rws[-1][0])+float(rws[0][0]))/2.0
    assert cdb.archiveTrajDatabase(dbloc, 'arch', arch_jd)
    rws = cdb.dbhandle.execute('select count(*) from trajectories').fetchall()
    assert rws[0][0] == 34
    cdb.closeTrajDatabase()
    cdb = TrajectoryDatabase(dbloc, 'arch_trajectories.db')
    rws = cdb.dbhandle.execute('select count(*) from trajectories').fetchall()
    assert rws[0][0] == 15
    cdb.closeTrajDatabase()
    os.remove(os.path.join(dbloc, 'trajectories.db'))
    os.remove(os.path.join(dbloc, 'arch_trajectories.db'))

    shutil.copyfile(os.path.join(dbloc, 'test_traj.db'), os.path.join(dbloc, 'trajectories.db'))
    cdb = TrajectoryDatabase(dbloc, 'trajectories.db')
    assert cdb.archiveTrajDatabase(dbloc, None, arch_jd)
    rws = cdb.dbhandle.execute('select count(*) from trajectories').fetchall()
    assert rws[0][0] == 34
    cdb.closeTrajDatabase()
    os.remove(os.path.join(dbloc, 'trajectories.db'))

def test_mergeTrajDb():
    shutil.copyfile(os.path.join(dbloc, 'test_traj.db'), os.path.join(dbloc, 'trajectories.db'))
    cdb = TrajectoryDatabase(dbloc, 'trajectories.db')
    assert cdb.mergeTrajDatabase(os.path.join(dbloc, 'test_mt.db'))
    rws = cdb.dbhandle.execute('select count(*) from trajectories').fetchall()
    assert rws[0][0] == 71
    cdb.closeTrajDatabase()
    os.remove(os.path.join(dbloc, 'trajectories.db'))

###################################################
## test for locking strategies

def worker(db_file):
    with sqlite3.connect(db_file) as conn:
        conn.execute("BEGIN EXCLUSIVE")
        time.sleep(2)  # Simulate long operation
        conn.execute("drop TABLE IF EXISTS tasks")
        conn.execute("CREATE TABLE tasks (id INTEGER)")
        conn.execute("drop TABLE IF EXISTS tasks")
        conn.commit()

def test_lockedTrajDb():
    # launch a thread that locks the database for two seconds
    thread = threading.Thread(target=worker, args=(os.path.join(dbloc, 'test_traj.db'),))
    thread.start()

    trajdb = TrajectoryDatabase(dbloc, 'test_traj.db')
    #reset_trajdb(trajdb)

    # test a real successful trajectory
    trajdb.removeTrajectoryById(f'{test_traj_id}')
    cur = trajdb.dbhandle.execute(f"select count(*) from trajectories where traj_id = '{test_traj_id}' and status=0")
    dta = cur.fetchall()
    numrows = dta[0][0]
    assert numrows == 1
    reset_trajdb(trajdb)

