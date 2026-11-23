""" Tests of the command line options of the trajectory solver, added by addSolverOptions().

Run with pytest:
    python -m pytest wmpl/Formats/Tests/test_GenericFunctions.py -v
"""

from __future__ import print_function, division, absolute_import

import argparse

from wmpl.Formats.GenericFunctions import addSolverOptions



def testInitialVelocityFromDragFitOption():
    """ Check that --vinitdrag is off by default and turned on by the flag. """

    arg_parser = addSolverOptions(argparse.ArgumentParser())

    assert arg_parser.parse_args([]).vinitdrag is False
    assert arg_parser.parse_args(['--vinitdrag']).vinitdrag is True




if __name__ == "__main__":

    # Run all tests
    test_list = [testInitialVelocityFromDragFitOption]

    for test in test_list:
        test()
        print("PASS {:s}".format(test.__name__))
