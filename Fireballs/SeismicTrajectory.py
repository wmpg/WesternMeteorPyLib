""" Determine the fireball trajectory from seismic data.

Modified method of Pujol et al. (2005).

"""

from __future__ import print_function, division, absolute_import

import numpy as np

from Utils.TrajConversions import raDec2ECI
from Utils.Math import vectMag



def timeResiduals(params, stat_eci_list, arrival_times, v_sound):
    """ Cost function for seismic fireball trajectory optimization. 

    Arguments:
        params: [list] Estimated parameters: x0, t0, z0, t0, v, ra, dec.
        stat_eci_list: [list of ndarrays] A list of station ECI coordinates (x, y, z).
        arrival_times: [list] A list of arrival times of the sound wave to the seismic station (in seconds 
            from some referent time).
        v_sound: [float] Average speed of sound (m/s).

    """

    x0, y0, z0, t0, v, ra, dec = params


    cost_value = 0

    # Go through all arrival times
    for t_obs, stat_eci in zip(arrival_times, stat_eci_list):

        ### Calculate the difference between the observed and the prediced arrival times ###
        ######################################################################################################

        # Calculate the mach angle
        beta = np.arcsin(v_sound/v)

        # Radiant vector
        x, y, z = raDec2ECI(ra, dec)
        u = np.array([x, y, z])

        # Difference from the referent point on the trajectory and the station
        b = stat_eci - np.array([x0, y0, z0])


        # Calculate the distance along the trajectory
        dt = np.abs(np.dot(b, u))

        # Calculate the distance prependicular to the trajectory
        dp = np.sqrt(vectMag(b)**2 - dt**2)

        # Calculate the time of arrival
        ti = t0 - dt/v + (dp*np.cos(beta))/v_sound

        # Calculate the squared difference in time
        cost_value += (t_obs - ti)**2

        ######################################################################################################


    return cost_value





if __name__ == "__main__":


    # Average speed of sound in the atmosphere
    v_sound = 320 # m/s

    