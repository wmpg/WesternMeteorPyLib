""" Determine the fireball trajectory from seismic data.

Modified method of Pujol et al. (2005).

"""

from __future__ import print_function, division, absolute_import

import numpy as np
import scipy.signal

from Formats.CSSseismic import loadCSSseismicData
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

    import matplotlib.pyplot as plt

    ### INPUTS ###
    ##########################################################################################################


    # DATA PATHS
    dir_paths = [
        #"/local4/infrasound/Infrasound/Fireball/15-Sep-2007/seismic/UBh2"
        #"/local4/infrasound/Infrasound/Fireball/15-Sep-2011-SW_USA/is56"
        "/local4/infrasound/Infrasound/Fireball/15-Sep-2011-SW_USA/is57"
    ]

    site_files = [
        #"UBh2.site"
        #"is56.site"
        "is57.site"
        ]

    wfdisc_files = [
        #"UBh2.wfdisc"
        #"is56.wfdisc"
        "is57.wfdisc"
        ]

    # Average speed of sound in the atmosphere
    v_sound = 320 # m/s


    ##########################################################################################################

    seismic_data = []

    # Load seismic data from given files
    for dir_path, site_file, wfdisc_file in zip(dir_paths, site_files, wfdisc_files):

        # Load the seismic data from individual file
        file_data = loadCSSseismicData(dir_path, site_file, wfdisc_file)

        # Add all entries to the global list
        for entry in file_data:
            seismic_data.append(entry)


    # Determine the earliest time from all beginning times
    ref_time = min([w.begin_time for _, w, _, _ in seismic_data])

    # Setup the plotting
    f, axes = plt.subplots(nrows=len(seismic_data), ncols=1, sharex=True)

    for i, entry in enumerate(seismic_data):

        # Select the current axis for plotting
        ax = axes[i]

        # Unpack the loaded seismic data
        site, w, time_data, waveform_data = entry


        # Calculate the difference from the referent time
        t_diff = (w.begin_time - ref_time).total_seconds()

        # Offset the time data to be in accordance with the referent time
        time_data += t_diff


        ax.plot(time_data, waveform_data, zorder=3)

        ax.grid(color='0.9')

    plt.subplots_adjust(hspace=0)
    plt.show()

