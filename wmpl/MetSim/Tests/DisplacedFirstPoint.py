""" A decelerating single body observed with the first point moved along the track, shared by the tests of the
time offset of the simulation from the first observed point in MetSim and in the Dynesty fit.
"""

from __future__ import print_function, division, absolute_import

from types import SimpleNamespace

import numpy as np

from wmpl.MetSim.GUI import SimulationResults
from wmpl.MetSim.MetSimErosion import runSimulation
from wmpl.MetSim.Tests.test_MetSimGravityDrop import bodyConstants


# Initial speed of the simulated meteor (m/s)
V_INIT = 20000.0

# Height of the first observed point (m)
HT_BEG = 90000.0

# Along-track error of the first observed point (m)
FIRST_POINT_ERROR = 300.0



def simulate(v_init=V_INIT, h_kill=20000.0):
    """ Simulate a decelerating single body.

    Keyword arguments:
        v_init: [float] Initial speed (m/s). V_INIT by default.
        h_kill: [float] Height at which the simulation stops (m). 20000 by default.

    Return:
        sr: [SimulationResults object] Simulation results.
    """

    const = bodyConstants(0.1, v_init, 45.0, h_kill)

    return SimulationResults(const, *runSimulation(const))



def observe(sr, first_point_error):
    """ Observe the simulated meteor from two stations at 25 frames per second between 90 and 45 km, without
        noise except for the first point, which is moved along the track. The lengths are measured from the
        first point, so all other lengths are shorter by the error, and the begin height is that of the
        moved point.

    Arguments:
        sr: [SimulationResults object] Simulation results.
        first_point_error: [float] Along-track error of the first point, positive in the direction of
            motion (m).

    Return:
        traj: [SimpleNamespace] Trajectory with the begin height and the observations of both stations.
    """

    time_arr, len_arr, ht_arr = sr.time_arr, sr.leading_frag_length_arr, sr.leading_frag_height_arr

    # Find when the meteor passes the first point
    time_beg = np.interp(-HT_BEG, -ht_arr, time_arr)
    len_beg = np.interp(time_beg, time_arr, len_arr)
    time_end = np.interp(-45000.0, -ht_arr, time_arr)

    observations = []
    for station_id, time_start in [("A", 0.0), ("B", 0.3)]:

        time_data = np.arange(time_start, time_end - time_beg, 0.04)

        # Lengths from the first point, and the heights at them
        len_true = np.interp(time_beg + time_data, time_arr, len_arr) - len_beg
        len_data = len_true - first_point_error
        model_ht = np.interp(len_beg + len_true, len_arr, ht_arr)

        # The first point of the first station is the reference, so its length is zero
        if station_id == "A":
            len_data[0] = 0.0
            model_ht[0] = np.interp(len_beg + first_point_error, len_arr, ht_arr)

        observations.append(SimpleNamespace(station_id=station_id, time_data=time_data, model_ht=model_ht, \
            state_vect_dist=len_data, absolute_magnitudes=None, ignore_list=np.zeros(len(time_data))))

    return SimpleNamespace(rbeg_ele=observations[0].model_ht[0], observations=observations)



def pointCount(traj):
    """ Count the observed points.

    Arguments:
        traj: [SimpleNamespace] Trajectory made by observe().

    Return:
        [int] Number of points of all stations.
    """

    return sum(len(obs.time_data) for obs in traj.observations)
