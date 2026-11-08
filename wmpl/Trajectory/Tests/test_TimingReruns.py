""" Tests the extra passes after the timing rerun (Trajectory(timing_reruns=...)).

The pass that follows the timing estimate recomputes the solution with the estimated clock offsets and
estimates them again, but the station positions and lines of sight it fits were computed at the times it
began with, and the stations move with the Earth. The first pass fits a straight line, so with gravity its
clock estimate is off. Here, for a slow 4 s meteor that one station sees nearly end-on, it is off by 0.4 ms,
which moves the other three stations by 0.14 m.

The tests drop the noise-free straight meteor of SyntheticTrajectory with applyGravityDrop, observe it from
four stations with their clocks set exactly, and solve it with the timing estimated.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_TimingReruns.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np

import matplotlib
matplotlib.use('Agg')

from wmpl.Trajectory.Tests.SyntheticTrajectory import observeStraightMeteor, worstResidual
from wmpl.Utils.Math import angleBetweenVectors


# Station ID, latitude (deg), longitude (deg) and elevation (m). A sees the meteor 7 deg off its line of
#   flight
STATIONS = [
    ('A', -42.565, 172.781, 310.0),
    ('B', -42.995, 173.312, 95.0),
    ('C', -43.804, 172.453, 640.0),
    ('D', -42.995, 169.873, 1450.0),
]



def _solve(timing_reruns):
    """ Solve the slow meteor dropped by gravity, and count the passes after the timing estimate.

    Arguments:
        timing_reruns: [int] Passed to Trajectory.

    Return:
        (traj, radiant_error, reruns):
            - traj: [Trajectory object] Solved trajectory.
            - radiant_error: [float] Angle between the solved and the true radiant (arcsec).
            - reruns: [int] Number of passes after the timing estimate.
    """

    traj, _, radiant_eci = observeStraightMeteor(noise=0.0, duration=4.0, v_init=17500.0, elevation=42.0, \
        gravity_drop=True, begin_ht=92000.0, stations=STATIONS, estimate_timing_vel=True, \
        gravity_correction=True, calc_orbit=False, timing_reruns=timing_reruns)

    # Count the passes after the timing estimate
    reruns = []
    run = traj.run

    def countedRun(*args, **kwargs):
        reruns.append(kwargs.get('_rerun_timing', False))
        return run(*args, **kwargs)

    traj.run = countedRun
    traj = traj.run()

    return traj, 3600*np.degrees(angleBetweenVectors(traj.radiant_eci_mini, radiant_eci)), sum(reruns)



def testTheDefaultKeepsTheSinglePass():
    """ Check that timing_reruns=0 runs one pass after the timing estimate, as before the option existed.
    """

    traj, _, reruns = _solve(0)

    assert reruns == 1
    assert traj.timing_reruns == 0



def testAnExtraPassBringsTheGeometryToTheFinalClocks():
    """ Check that an extra pass, run at the clocks of the pass before, removes the residual of the station
        motion. With one pass, 0.13 m of residual and 0.032 arcsec remain, with an extra one 0.008 m and
        0.005 arcsec, which is the precision of the timing estimate itself.
    """

    single, _, _ = _solve(0)
    traj, radiant_error, reruns = _solve(3)

    # Check the residuals and the radiant
    assert worstResidual(traj) < 0.02
    assert worstResidual(traj) < worstResidual(single)/5
    assert radiant_error < 0.01

    # The extra pass leaves the clocks where it found them, so no further one is run
    assert reruns == 2




if __name__ == "__main__":

    # Print the number of passes, the radiant error and the worst residual
    for timing_reruns in (0, 3):
        traj, radiant_error, reruns = _solve(timing_reruns)
        print("timing_reruns={:d}: {:d} reruns, radiant error {:.4f} arcsec, worst residual {:.4f} m".format(\
            timing_reruns, reruns, radiant_error, worstResidual(traj)))
