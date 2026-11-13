""" Tests for keeping in the pickle of a solved trajectory what is computed after the solver, such as the
    solved fragments: saveWithOriginalPicks() sets it on the solution with the original picks and saves that
    solution again.

Run with pytest:
    python -m pytest wmpl/Formats/Tests/test_ECSVSavedSolution.py -v
"""

from __future__ import print_function, division, absolute_import

import os

from wmpl.Formats.ECSV import saveWithOriginalPicks
from wmpl.Utils.Pickling import loadPickle, savePickle


# Name of the pickle of the solution
FILE_NAME = '20200101_000000'



class SolvedTrajectory(object):
    """ Stand-in for a solved trajectory, with what saveWithOriginalPicks() uses. """

    def __init__(self, output_dir, label, monte_carlo=False, save_results=True):

        self.output_dir = output_dir
        self.file_name = FILE_NAME
        self.save_results = save_results
        self.uncertainties = 'uncertainties' if monte_carlo else None
        self.label = label



def savedSolution(output_dir):
    """ Load the pickle of the solution in the given directory. """

    return loadPickle(output_dir, FILE_NAME + '_trajectory.pickle')



def testTheAttributesAreSavedWithTheSolution(tmp_path):
    """ Without the Monte Carlo runs the solver returns the solution with the original picks, which gets the
        attributes and is saved again with them.
    """

    traj = SolvedTrajectory(str(tmp_path), 'original picks')
    savePickle(traj, traj.output_dir, FILE_NAME + '_trajectory.pickle')

    saveWithOriginalPicks(traj, fragment_trajs=[(1, 'fragment 1')])

    assert traj.fragment_trajs == [(1, 'fragment 1')]
    assert savedSolution(traj.output_dir).fragment_trajs == [(1, 'fragment 1')]



def testTheSolutionWithTheOriginalPicksIsSavedWithMonteCarloRuns(tmp_path):
    """ With the Monte Carlo runs the solver returns the best run, but the pickle keeps the solution with the
        original picks, which gets the attributes, and the best run is left as it is.
    """

    savePickle(SolvedTrajectory(str(tmp_path), 'original picks', monte_carlo=True), str(tmp_path), \
        FILE_NAME + '_trajectory.pickle')
    best_run = SolvedTrajectory(str(tmp_path), 'best run', monte_carlo=True)

    traj_orig = saveWithOriginalPicks(best_run, fragment_trajs=[(1, 'fragment 1')])
    saved = savedSolution(str(tmp_path))

    assert traj_orig.label == 'original picks'
    assert saved.label == 'original picks'
    assert saved.fragment_trajs == [(1, 'fragment 1')]
    assert not hasattr(best_run, 'fragment_trajs')



def testNothingIsSavedWithoutSavedResults(tmp_path):
    """ Without saved results there is no pickle to keep the attributes in, and none is written. """

    traj = SolvedTrajectory(str(tmp_path), 'original picks', save_results=False)

    traj_orig = saveWithOriginalPicks(traj, fragment_trajs=[(1, 'fragment 1')])

    assert traj_orig.fragment_trajs == [(1, 'fragment 1')]
    assert os.listdir(str(tmp_path)) == []
