""" Test of the --losfit command line option through wmpl.Formats.ECSV, verifying that Huber is the default
    and that --losfit angles disables formal least-squares line uncertainties.

Run with pytest:
    python -m pytest wmpl/Formats/Tests/test_ECSVLosFit.py -v
"""

from __future__ import print_function, division, absolute_import

import glob
import os
import subprocess
import sys
import tempfile

from wmpl.Formats.Tests.StraightMeteorECSV import writeStraightMeteorECSVs


def _solveECSVReport(los_fit):
    """ Solve the straight synthetic ECSV files via the ECSV CLI without Monte Carlo and read the report.

    Arguments:
        los_fit: [str or None] Value for --losfit, or None to use the CLI default.

    Return:
        [str] Text contents of the generated trajectory report.
    """

    with tempfile.TemporaryDirectory() as tmp_dir:

        paths, _, _ = writeStraightMeteorECSVs(tmp_dir)

        # Invoke the ECSV command line without Monte Carlo or interactive plots
        command = [sys.executable, '-m', 'wmpl.Formats.ECSV'] + paths + ['-d', '-x']
        if los_fit is not None:
            command += ['--losfit', los_fit]

        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        report_paths = glob.glob(os.path.join(tmp_dir, '*', '*_report.txt'))
        assert len(report_paths) == 1

        with open(report_paths[0]) as f:
            return f.read()



def testECSVLosFitOptionSelectsMethod():
    """ Check that ECSV defaults to Huber (printing formal 1-sigma line uncertainties) and respects
        --losfit least_squares and --losfit angles.
    """

    assert "Formal 1 sigma of the line-of-sight fit (huber)" in _solveECSVReport(None)
    assert "Formal 1 sigma of the line-of-sight fit (least_squares)" in _solveECSVReport('least_squares')
    assert "Formal 1 sigma of the line-of-sight fit" not in _solveECSVReport('angles')
