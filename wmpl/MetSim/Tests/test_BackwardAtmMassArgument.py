""" Tests of --mass of the run back through the atmosphere, shared with REBOUND: the mass at the trajectory's
    reference point, and without a value the photometric mass of the trajectory's light curve.

Run with pytest:
    python -m pytest wmpl/MetSim/Tests/test_BackwardAtmMassArgument.py -v
"""

from __future__ import print_function, division, absolute_import

import argparse
import copy
import os

import pytest

from wmpl.MetSim.BackwardAtmIntegration import addBackwardArguments, checkBackwardArguments, massArgument
from wmpl.Trajectory.AggregateAndPlot import computeMass, P_0M
from wmpl.Utils.Pickling import loadPickle


# Example trajectory with magnitudes at all stations
EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Dynesty", "examples",
    "20191023_091225")
EXAMPLE_PICKLE = "20191023_091225_trajectory.pickle"



def _parser():
    parser = argparse.ArgumentParser()
    addBackwardArguments(parser)

    return parser



@pytest.fixture(scope="module")
def traj():
    return loadPickle(EXAMPLE_DIR, EXAMPLE_PICKLE)



def testMassWithoutAValueIsThePhotometricMassOfTheLightCurve(traj):
    """ --mass without a value is the photometric mass with the zero-magnitude power of the trajectory
        summaries, which then passes the checks of the run back. """

    parser = _parser()
    args = parser.parse_args(["--mass"])
    assert args.mass == 0

    massArgument(parser, args, traj)

    assert args.mass == pytest.approx(computeMass(traj, P_0M), rel=1e-12)
    assert args.mass > 0
    checkBackwardArguments(parser, args)



def testAGivenMassIsKept(traj):
    """ A mass given in kg, or none, is left as it is. """

    parser = _parser()

    args = parser.parse_args(["--mass", "0.25"])
    massArgument(parser, args, traj)
    assert args.mass == 0.25

    args = parser.parse_args([])
    massArgument(parser, args, traj)
    assert args.mass is None



def testMassWithoutPhotometryOrNotPositiveIsACommandLineError(traj):
    """ Without magnitudes --mass needs a value, and a negative or non-finite mass is refused. """

    traj_dark = copy.deepcopy(traj)
    for obs in traj_dark.observations:
        obs.absolute_magnitudes = None

    parser = _parser()

    with pytest.raises(SystemExit):
        massArgument(parser, parser.parse_args(["--mass"]), traj_dark)

    for value in ("-1", "nan", "inf"):
        with pytest.raises(SystemExit):
            massArgument(parser, parser.parse_args(["--mass", value]), traj)




if __name__ == "__main__":

    example = loadPickle(EXAMPLE_DIR, EXAMPLE_PICKLE)

    testMassWithoutAValueIsThePhotometricMassOfTheLightCurve(example)
    testAGivenMassIsKept(example)
    testMassWithoutPhotometryOrNotPositiveIsACommandLineError(example)

    print("All tests passed.")
