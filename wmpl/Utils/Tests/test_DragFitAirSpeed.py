""" Tests that the drag fit decelerates the meteor with the speed relative to the air, which moves with the
    ground, and not with the inertial speed.
"""

from __future__ import print_function, division, absolute_import

import io
import contextlib

import numpy as np

from wmpl.Utils.DragFit import DragFit
from wmpl.Trajectory.Tests.SyntheticTrajectory import DragMeteor, observeMeteor



def testBallisticCoefficientOfASlowWestwardMeteor():
    """ Check the ballistic coefficient of a meteor flying west, which slows from 15 to 4 km/s. The Earth's
        rotation takes 239 m/s off its speed relative to the air, so a drag on the inertial speed would make
        K about 8% too large at the end.
    """

    # Constant ballistic coefficient, with the drag acting on the speed relative to the ground
    k_true = 0.005
    meteor = DragMeteor(15000.0, 90.0, 45.0, k_true, k_true, 2.7, 10.0, ground_drag=True)

    traj = observeMeteor(meteor)
    with contextlib.redirect_stdout(io.StringIO()):
        traj = traj.run()

    drag_fit = DragFit(traj)

    # Compare the fitted coefficient with the truth along the observed part
    time_arr = np.linspace(0.1, meteor.t_vis_end - meteor.t_vis_beg - 0.1, 10)
    k_arr, _ = drag_fit.ballisticCoeffAtLength(drag_fit.lengthAtTime(time_arr))

    assert drag_fit.v_rotation < -200.0
    assert np.max(np.abs(np.log(k_arr/k_true))) < 0.04
