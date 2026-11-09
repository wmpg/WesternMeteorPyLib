""" Tests that the --losfit option of the ECSV command line picks the line-of-sight fit of the trajectory
    solver (los_fit of wmpl.Trajectory.Trajectory).

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


def solveECSVReport(los_fit):
    """ Solve the ECSV files of the straight meteor with the ECSV command line, without Monte Carlo runs, and
        read the report.

    Arguments:
        los_fit: [str] Value of --losfit, or None to leave it out.

    Return:
        [str] The report of the solution.
    """

    with tempfile.TemporaryDirectory() as tmp_dir:

        paths, _ = writeStraightMeteorECSVs(tmp_dir)

        # Run the ECSV command line without Monte Carlo runs
        command = [sys.executable, "-m", "wmpl.Formats.ECSV"] + paths + ["-d", "-x"]
        if los_fit is not None:
            command += ["--losfit", los_fit]

        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Read the report of the solution
        report_paths = glob.glob(os.path.join(tmp_dir, "*", "*_report.txt"))
        assert len(report_paths) == 1

        with open(report_paths[0]) as f:
            return f.read()



def testLosFitOptionPicksTheFit():
    """ Check that the line-of-sight fit is the Huber fit by default, which the report names with its formal
        uncertainty, and that --losfit angles fits the angles, which have no formal uncertainty.
    """

    assert "Formal 1 sigma of the line-of-sight fit (huber)" in solveECSVReport(None)
    assert "Formal 1 sigma of the line-of-sight fit" not in solveECSVReport("angles")



if __name__ == "__main__":

    testLosFitOptionPicksTheFit()

    print("All tests passed.")
