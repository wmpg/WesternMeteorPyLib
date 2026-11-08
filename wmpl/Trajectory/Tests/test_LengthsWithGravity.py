""" Tests that the lengths along the trajectory follow the gravity drop the line was fitted with.

In the pass that applies the gravity drop, the line-of-sight fit returns the undropped line, which the
measurements sit below by up to g*t^2/2. calcVelocity projected them onto that line as if they were on it,
so the lengths of each station were off by an amount that depends on its viewing geometry: the timing fit
read that as clock offsets, and the initial velocity was off too.

The tests drop a noise-free constant-velocity meteor by gravity, observe it from three stations, and solve
it with the timing estimated.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_LengthsWithGravity.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np

import matplotlib
matplotlib.use('Agg')

from wmpl.Trajectory.Tests.SyntheticTrajectory import observeStraightMeteor
from wmpl.Utils.Math import lineFunc


# A slow, shallow, long meteor, which drops by about 120 m
DURATION = 5.0
V_INIT = 15000.0
ELEVATION = 30.0



def _solve(clock_offsets=None):
    """ Solve the noise-free meteor dropped by gravity, with the timing estimated.

    Keyword arguments:
        clock_offsets: [dict] Seconds by which the clock of the station is ahead, per station ID. None by
            default, when all clocks are exact.

    Return:
        traj: [Trajectory object] Solved trajectory.
    """

    traj, _, _ = observeStraightMeteor(noise=0.0, duration=DURATION, v_init=V_INIT, elevation=ELEVATION, \
        gravity_drop=True, clock_offsets=clock_offsets, estimate_timing_vel=True, gravity_correction=True, \
        calc_orbit=False)

    return traj.run()



def _offsets(traj):
    """ Compute the estimated timing offsets relative to station A.

    Arguments:
        traj: [Trajectory object] Solved trajectory.

    Return:
        [dict] Timing offset per station ID (s).
    """

    offsets = dict(zip([obs.station_id for obs in traj.observations], traj.time_diffs_final))

    return {station_id: offset - offsets['A'] for station_id, offset in offsets.items()}



def testExactClocksGetNoOffset():
    """ Check that no timing offsets are estimated when all clocks are exact. Before the fix, -2.32 ms were
        estimated on B and -0.33 ms on C.
    """

    offsets = _offsets(_solve())

    assert abs(offsets['B']) < 5e-5
    assert abs(offsets['C']) < 5e-5



def testAClockOffsetIsRecovered():
    """ Check that a clock of B 0.3 s ahead is recovered. Before the fix, it was recovered as 302.32 ms.
    """

    offsets = _offsets(_solve(clock_offsets={'B': 0.3}))

    assert abs(offsets['B'] + 0.3) < 5e-5



def testTheInitialVelocityIsRecovered():
    """ Check that the initial velocity is recovered. Before the fix, it was 12.07 m/s too high.
    """

    traj = _solve()

    assert abs(traj.v_init - V_INIT) < 0.5



def testTheLagIsComputedFromTheFittedLengths():
    """ Check that the lag is computed from the lengths the velocity was fitted to, as the lengths with the
        gravity drop depend on the times, which the timing estimation changes. Before the fix, the lengths
        were computed again with the new times and were up to 4.9 cm off.
    """

    traj = _solve(clock_offsets={'B': 0.3})

    for obs in traj.observations:
        lag = obs.state_vect_dist - lineFunc(obs.time_data, *traj.velocity_fit)
        assert np.allclose(obs.lag, lag, rtol=0, atol=1e-9)




if __name__ == "__main__":

    # Print the timing offsets and the initial velocity error with exact clocks
    traj = _solve()
    print("offsets {}  initial velocity error {:+.3f} m/s".format(_offsets(traj), traj.v_init - V_INIT))
