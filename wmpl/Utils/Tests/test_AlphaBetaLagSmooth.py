""" Tests of the air-relative velocities used by the lag smoothing (--lagsmooth) in the alpha-beta CLI.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_AlphaBetaLagSmooth.py -v
"""

from __future__ import print_function, division, absolute_import

import contextlib
import io
import os
import re
import subprocess
import sys
import tempfile

from wmpl.Trajectory.Tests.SyntheticTrajectory import DragMeteor, observeMeteor
from wmpl.Utils.Pickling import savePickle



def testLagSmoothUsesAirRelativeVelocity():
    """ Check that --lagsmooth with the full fit uses air-relative velocities for its dynamic pressure and
        labels them accordingly, while --lagfit simple stays in the inertial frame.
    """

    # Build a synthetic westward fireball whose air speed along the path is strongly negative
    meteor = DragMeteor(15000.0, 90.0, 45.0, 0.005, 0.005, 2.7, 10.0, ground_drag=True)
    traj = observeMeteor(meteor, noise=0.0, fps=10.0)
    with contextlib.redirect_stdout(io.StringIO()):
        traj = traj.run()

    # Save the solved trajectory to a temporary pickle and run the AlphaBeta CLI in both modes
    with tempfile.TemporaryDirectory() as tmp_dir:

        savePickle(traj, tmp_dir, 'westward_trajectory.pickle')
        pkl_path = os.path.join(tmp_dir, 'westward_trajectory.pickle')

        # Run the default full lag fit with lag smoothing enabled
        res_full = subprocess.run([sys.executable, '-m', 'wmpl.Utils.AlphaBeta', pkl_path, '--lagsmooth', \
            '--hideplots'], capture_output=True, text=True, timeout=300)
        assert res_full.returncode == 0, res_full.stderr

        # Run the simple lag fit, which keeps inertial velocities
        res_simple = subprocess.run([sys.executable, '-m', 'wmpl.Utils.AlphaBeta', pkl_path, '--lagsmooth', \
            '--lagfit', 'simple', '--hideplots'], capture_output=True, text=True, timeout=300)
        assert res_simple.returncode == 0, res_simple.stderr

    # Full fit labels air-relative lag-smoothing velocities while simple fit stays inertial
    assert "Lag smoothing:\nVelocities relative to the air\n" in res_full.stdout
    assert "Lag smoothing:\nVelocities relative to the air\n" not in res_simple.stdout

    # Westward meteor has negative air_speed, so air-relative speed and peak pressure exceed inertial values
    pat = r"Lag smoothing\s*=\s*([\d.]+)\s*MPa at\s*([\d.]+)\s*km \(v =\s*([\d.]+)\s*km/s\)"
    m_full = re.search(pat, res_full.stdout)
    m_simple = re.search(pat, res_simple.stdout)
    assert (m_full is not None) and (m_simple is not None)
    assert float(m_full.group(3)) > float(m_simple.group(3)) + 0.15
    assert float(m_full.group(1)) > float(m_simple.group(1))*1.03



if __name__ == "__main__":

    testLagSmoothUsesAirRelativeVelocity()
    print("All tests passed.")
