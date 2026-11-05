""" Tests for the solver options of the additional fragments of ECSV files: they are given to
    solveFragmentTrajectories() by the caller, and nothing about them is kept in the trajectory, so its
    pickles are the same as before.

Run with pytest:
    python -m pytest wmpl/Formats/Tests/test_ECSVFragmentOptions.py -v
"""

from __future__ import print_function, division, absolute_import

import sys
from types import SimpleNamespace

import numpy as np

import wmpl.Formats.ECSV
import wmpl.Formats.GenericFunctions
from wmpl.Formats.Tests.SyntheticECSV import writeFragmentECSV


# The modules themselves, to replace their functions in the tests (importing wmpl sets attributes of the
#   same names on the packages)
ECSV = sys.modules['wmpl.Formats.ECSV']
GENERIC = sys.modules['wmpl.Formats.GenericFunctions']



def _solvedMain(ecsv_paths, offset):
    """ Make a solved main trajectory, whose times are those of the files shifted by the given offset.

    Arguments:
        ecsv_paths: [list] Paths of the ECSV files.
        offset: [float] Time offset of all stations (s).

    Return:
        [SimpleNamespace] The trajectory, with its observations.
    """

    _, meteors = ECSV.loadECSVs(ecsv_paths, no_prepare=True)
    observations = [SimpleNamespace(station_id=meteor.station_id, \
        JD_data=meteor.jdt_ref + (np.array(meteor.time_data) + offset)/86400) for meteor in meteors]

    return SimpleNamespace(observations=observations, uncertainties=None, output_dir='main_dir')



def _fakeSolver(calls):
    """ Make a stand-in for solveTrajectoryGeneric() which keeps what it was called with.

    Arguments:
        calls: [list] List to which the keyword arguments of every call are appended.

    Return:
        [function] The stand-in.
    """

    def solve(jdt_ref, meteor_list, dir_path, solver='original', **kwargs):

        calls.append(dict(kwargs, solver=solver, dir_path=dir_path))

        return SimpleNamespace(uncertainties=None)

    return solve



def _twoStations(tmp_path):
    """ Write the ECSV files of two stations with fragment 1.

    Arguments:
        tmp_path: [Path] Directory of the files.

    Return:
        [list] Paths of the files.
    """

    return [writeFragmentECSV(tmp_path/"a.ecsv", "XX0001"), writeFragmentECSV(tmp_path/"b.ecsv", "XX0002")]



def testFragmentsGetTheOptionsGiven(tmp_path, monkeypatch):
    """ Check that a fragment is solved with the options of the main one, and with its time offsets fixed
        by default.
    """

    calls = []
    monkeypatch.setattr(ECSV, 'solveTrajectoryGeneric', _fakeSolver(calls))

    paths = _twoStations(tmp_path)
    solved = ECSV.solveFragmentTrajectories(_solvedMain(paths, 0.25), paths, solver='original', mc_runs=7, \
        monte_carlo=False, estimate_timing_vel=True)

    assert [fragment for fragment, _ in solved] == [1]
    assert len(calls) == 1

    assert calls[0]['solver'] == 'original'
    assert calls[0]['mc_runs'] == 7
    assert calls[0]['monte_carlo'] is False
    assert calls[0]['dir_path'].endswith('fragment_1')

    # The offsets of the main trajectory are fixed, with the timing estimation on for its residuals
    assert calls[0]['estimate_timing_vel'] is True

    # The times go through Julian dates, which resolve about 40 microseconds
    fixed = dict(entry.split(':') for entry in calls[0]['fixed_times'].split(','))
    assert sorted(fixed) == ['XX0001', 'XX0002']
    assert all(abs(float(offset) - 0.25) < 1e-4 for offset in fixed.values())



def testFragmentsKeepTheTimingOfTheMainOneWithoutTheTimingEstimation(tmp_path, monkeypatch):
    """ Check that without the timing estimation (-e, with or without the offsets of the stations), the
        fragments are solved with the same fixed offsets as the main one, and without the timing estimation.
    """

    paths = _twoStations(tmp_path)

    for estimate_timing_vel in [False, "XX0001:0.25,XX0002:0.25"]:

        calls = []
        monkeypatch.setattr(ECSV, 'solveTrajectoryGeneric', _fakeSolver(calls))

        ECSV.solveFragmentTrajectories(_solvedMain(paths, 0.25), paths, \
            estimate_timing_vel=estimate_timing_vel, fixed_times=None)

        assert calls[0]['estimate_timing_vel'] == estimate_timing_vel
        assert calls[0]['fixed_times'] is None



def testFragmentsCanEstimateTheirOwnOffsets(tmp_path, monkeypatch):
    """ Check that with reuse_timing off, the options are passed on as they were given. """

    calls = []
    monkeypatch.setattr(ECSV, 'solveTrajectoryGeneric', _fakeSolver(calls))

    paths = _twoStations(tmp_path)
    ECSV.solveFragmentTrajectories(_solvedMain(paths, 0.25), paths, reuse_timing=False, \
        estimate_timing_vel=True, fixed_times=None)

    assert calls[0]['estimate_timing_vel'] is True
    assert calls[0]['fixed_times'] is None



def testOtherSolversAreNotSupported(tmp_path, monkeypatch):
    """ Check that only the original solver is supported for the fragments: nothing is solved with the
        others.
    """

    calls = []
    monkeypatch.setattr(ECSV, 'solveTrajectoryGeneric', _fakeSolver(calls))

    paths = _twoStations(tmp_path)

    assert ECSV.solveFragmentTrajectories(_solvedMain(paths, 0.0), paths, solver='gural1') == []
    assert calls == []



def testTheReturnedTrajectoryKeepsNothingAboutTheOptions(tmp_path, monkeypatch):
    """ Check that the generic solver sets no attribute on what it returns, so that nothing new reaches the
        pickles (AutoGUI saves the returned trajectory itself, with savePickle()).
    """

    class FakeTrajectory(object):

        def __init__(self, *args, **kwargs):
            pass

        def infillTrajectory(self, *args, **kwargs):
            pass

        def run(self):
            return self

    monkeypatch.setattr(GENERIC, 'Trajectory', FakeTrajectory)

    meteor = SimpleNamespace(ra_data=[], dec_data=[], time_data=[], latitude=0.0, longitude=0.0, height=0.0, \
        station_id='XX0001', mag_data=[], radec_per_point=False)
    traj = GENERIC.solveTrajectoryGeneric(2461000.5, [meteor], str(tmp_path), solver='original', mc_runs=7)

    assert vars(traj) == {}
