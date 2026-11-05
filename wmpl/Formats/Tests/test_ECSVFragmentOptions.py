""" Tests for the solver options of the additional fragments of ECSV files: they are given to
    solveFragmentTrajectories() by the caller, and nothing about them is kept in the trajectory, so its pickles
    are the same as before.

Run with pytest:
    python -m pytest wmpl/Formats/Tests/test_ECSVFragmentOptions.py -v
"""

from __future__ import print_function, division, absolute_import

import sys
from types import SimpleNamespace

import numpy as np

import wmpl.Formats.ECSV
import wmpl.Formats.GenericFunctions
from wmpl.Formats.Tests.test_ECSV import writeEcsv


def solvedMain(ecsv_paths, offset):
    """ A solved main trajectory, whose times are those of the files shifted by the given offset (s). """

    ecsv = sys.modules['wmpl.Formats.ECSV']

    _, meteors = ecsv.loadECSVs(ecsv_paths, no_prepare=True)
    observations = [SimpleNamespace(station_id=meteor.station_id,
        JD_data=meteor.jdt_ref + (np.array(meteor.time_data) + offset)/86400) for meteor in meteors]

    return SimpleNamespace(observations=observations, uncertainties=None, output_dir='main_dir')


def fakeSolver(calls):
    """ A stand-in for solveTrajectoryGeneric() which keeps what it was called with. """

    def solve(jdt_ref, meteor_list, dir_path, solver='original', **kwargs):

        calls.append(dict(kwargs, solver=solver, dir_path=dir_path))

        return SimpleNamespace(uncertainties=None)

    return solve


def test_fragments_get_the_options_given(tmp_path, monkeypatch):
    """ A fragment is solved with the options of the main one, and with its time offsets fixed by default. """

    ecsv = sys.modules['wmpl.Formats.ECSV']

    calls = []
    monkeypatch.setattr(ecsv, 'solveTrajectoryGeneric', fakeSolver(calls))

    paths = [writeEcsv(tmp_path/"a.ecsv", "XX0001"), writeEcsv(tmp_path/"b.ecsv", "XX0002")]
    solved = ecsv.solveFragmentTrajectories(solvedMain(paths, 0.25), paths, solver='original', mc_runs=7,
        monte_carlo=False, estimate_timing_vel=False)

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


def test_fragments_can_estimate_their_own_offsets(tmp_path, monkeypatch):
    """ With reuse_timing off, the options are passed on as they were given. """

    ecsv = sys.modules['wmpl.Formats.ECSV']

    calls = []
    monkeypatch.setattr(ecsv, 'solveTrajectoryGeneric', fakeSolver(calls))

    paths = [writeEcsv(tmp_path/"a.ecsv", "XX0001"), writeEcsv(tmp_path/"b.ecsv", "XX0002")]
    ecsv.solveFragmentTrajectories(solvedMain(paths, 0.25), paths, reuse_timing=False, estimate_timing_vel=False,
        fixed_times=None)

    assert calls[0]['estimate_timing_vel'] is False
    assert calls[0]['fixed_times'] is None


def test_other_solvers_are_not_supported(tmp_path, monkeypatch):
    """ Only the original solver is supported for the fragments: nothing is solved with the others. """

    ecsv = sys.modules['wmpl.Formats.ECSV']

    calls = []
    monkeypatch.setattr(ecsv, 'solveTrajectoryGeneric', fakeSolver(calls))

    paths = [writeEcsv(tmp_path/"a.ecsv", "XX0001"), writeEcsv(tmp_path/"b.ecsv", "XX0002")]

    assert ecsv.solveFragmentTrajectories(solvedMain(paths, 0.0), paths, solver='gural1') == []
    assert calls == []


def test_the_returned_trajectory_keeps_nothing_about_the_options(tmp_path, monkeypatch):
    """ The generic solver sets no attribute on what it returns, so that nothing new reaches the pickles
        (AutoGUI saves the returned trajectory itself, with savePickle()). """

    class FakeTrajectory(object):
        def __init__(self, *args, **kwargs):
            pass

        def infillTrajectory(self, *args, **kwargs):
            pass

        def run(self):
            return self

    generic = sys.modules['wmpl.Formats.GenericFunctions']
    monkeypatch.setattr(generic, 'Trajectory', FakeTrajectory)

    meteor = SimpleNamespace(ra_data=[], dec_data=[], time_data=[], latitude=0.0, longitude=0.0, height=0.0,
        station_id='XX0001', mag_data=[], radec_per_point=False)
    traj = generic.solveTrajectoryGeneric(2461000.5, [meteor], str(tmp_path), solver='original', mc_runs=7)

    assert vars(traj) == {}
