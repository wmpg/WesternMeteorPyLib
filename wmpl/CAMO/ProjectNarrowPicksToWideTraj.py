""" Takes measurements from the narrow field and projects them on the trajectory estimated from wide-field 
    data. 
"""

from __future__ import print_function, division, absolute_import

import os
import sys

import numpy as np
import scipy.optimize
import scipy.interpolate
import matplotlib.pyplot as plt


from wmpl.Formats.Met import loadMet
from wmpl.Trajectory.Trajectory import lineFunc
from wmpl.Utils.TrajConversions import raDec2ECI, geo2Cartesian_vect, altAz2RADec_vect, unixTime2JD, cartesian2Geo, jd2Date
from wmpl.Utils.Math import findClosestPoints, vectMag
from wmpl.Utils.AtmosphereDensity import getAtmDensity, getAtmDensity_vect
from wmpl.Utils.Pickling import loadPickle
from wmpl.Utils.Physics import dynamicPressure

# DRAG COEFFICIENT (assume unity)
DRAG_COEFF = 1.0




class FragmentationInfo(object):
    def __init__(self, frag_dict, fragmentation_points):
        """ Container for information about fragments and fragmentation points. """

        self.frag_dict = frag_dict

        #self.frag_dict_rev = dict((v,k) for k,v in frag_dict.iteritems())

        self.fragmentation_points = fragmentation_points





def timeHeightFunc(x, a, b, c, d):
    return a + b*x + c*x**2 + d*x**3


def exponentialDeceleration(t, v, d_t, k, a1, a2):
    """ Model for exponential deceleration. Returns the length at every point. """
    td = t + d_t
    return k + v*td + a1*np.exp(a2*td)



def exponentialDecelerationVel(t, v, d_t, k, a1, a2):
    """ Model for exponential deceleration. Returns the velocity at every point. """
    td = t + d_t
    return v + a1*a2*np.exp(a2*td)



def projectNarrowPicks(dir_path, met, traj, traj_uncert, frag_info):
    """ Projects picks done in the narrow-field to the given trajectory. """


    ### MANUAL ENTRIES ###
    ##########################################################################################################


    ##########################################################################################################


    # Go through picks from all sites (there should only be one)
    for site_no in met.picks:

        # Extract site exact plate
        exact = met.exact_plates[site_no]

        # Extract site picks
        picks = np.array(met.picks[site_no])


        # Skip the site if there are no picks
        if not len(picks):
            continue

        # Find unique fragments
        fragments = np.unique(picks[:, 1])

        # If the fragmentation dictionary is empty, generate one
        if frag_info.frag_dict is None:
            frag_info.frag_dict = {float(i):i+1 for i in range(len(fragments))}

        # A list with results of finding the closest point on the trajectory
        cpa_list = []

        # Go thorugh all fragments and calculate the coordinates of the closest points on the trajectory and 
        # the line of sight
        for frag in fragments:

            # Take only those picks from current fragment
            frag_picks = picks[picks[:, 1] == frag]

            # Sort by frame
            frag_picks = frag_picks[np.argsort(frag_picks[:, 0])]

            # Extract Unix timestamp
            ts = frag_picks[:, 11]
            tu = frag_picks[:, 12]

            # Extract theta, phi
            theta = np.radians(frag_picks[:, 4])
            phi   = np.radians(frag_picks[:, 5])

            # Calculate azimuth +E of N
            azim = (np.pi/2.0 - phi)%(2*np.pi)
            
            # Calculate elevation
            elev = np.pi/2.0 - theta

            # Calculate Julian date from Unix timestamp
            jd_data = np.array([unixTime2JD(s, u) for s, u in zip(ts, tu)])
            
            # Convert azim/elev to RA/Dec
            ra, dec = altAz2RADec_vect(azim, elev, jd_data, exact.lat, exact.lon)

            # Convert RA/Dec to ECI direction vector
            x_eci, y_eci, z_eci = raDec2ECI(ra, dec)

            # Convert station geocoords to ECEF coordinates
            x_stat_vect, y_stat_vect, z_stat_vect = geo2Cartesian_vect(exact.lat, exact.lon, exact.elev, \
                jd_data)


            # Find closest points of aproach for all measurements
            for jd, x, y, z, x_stat, y_stat, z_stat in np.c_[jd_data, x_eci, y_eci, z_eci, x_stat_vect, \
                y_stat_vect, z_stat_vect]:
                
                # Find the closest point of approach of every narrow LoS to the wide trajectory
                obs_cpa, rad_cpa, d = findClosestPoints(np.array([x_stat, y_stat, z_stat]), \
                    np.array([x, y, z]), traj.state_vect_mini, traj.radiant_eci_mini)


                # Calculate the height of each fragment for the given time
                rad_lat, rad_lon, height = cartesian2Geo(jd, *rad_cpa)

                cpa_list.append([frag, jd, obs_cpa, rad_cpa, d, rad_lat, rad_lon, height])



        # Find the coordinates of the first point in time on the trajectory and the first JD
        first_jd_indx = np.argmin([entry[1] for entry in cpa_list])
        jd_ref = cpa_list[first_jd_indx][1]
        rad_cpa_ref = cpa_list[first_jd_indx][3]

        print(jd_ref)
        #print(cpa_list)

        length_list = []
        decel_list = []

        # Go through all fragments and calculate the length from the reference point
        for frag in fragments:

            # Select only the data points of the current fragment
            cpa_data = [entry for entry in cpa_list if entry[0] == frag]

            # Lengths of the current fragment
            length_frag = []

            # Go through all projected points on the trajectory
            for entry in cpa_data:

                jd = entry[1]
                rad_cpa = entry[3]
                rad_lat = entry[5]
                rad_lon = entry[6]
                height = entry[7]

                # Calculate the distance from the first point on the trajectory and the given point
                dist = vectMag(rad_cpa - rad_cpa_ref)

                # Calculate the time in seconds
                time_sec = (jd - jd_ref)*24*3600

                length_frag.append([time_sec, dist, rad_lat, rad_lon, height])
                length_list.append([frag, time_sec, dist, rad_lat, rad_lon, height])
        

            # Fit a line to the first part of the fragment which starts the first
            if length_frag[0][0] == 0:

                length_frag = np.array(length_frag)

                # Take only the first 1/10 of the first fragment's trajectory
                part_size = int(0.1*len(length_frag))
                if part_size < 4:
                    part_size = 4

                # Extract JDs and lengths into individual arrays
                time_data, length_data, lat_data, lon_data, height_data = length_frag[:part_size].T

                # print(time_data, length_data)

                # Fit a line to the first part of the first fragment to get the estimate of the velocity
                lag_line, _ = scipy.optimize.curve_fit(lineFunc, time_data, length_data)


            ### Fit the deceleration model to the length ###
            ##################################################################################################

            length_frag = np.array(length_frag)

            # Extract JDs and lengths into individual arrays
            time_data, length_data, lat_data, lon_data, height_data = length_frag.T

            # First guess of the lag parameters
            p0 = [0, 0, traj.jacchia_fit[0], traj.jacchia_fit[1]]

            # Length residuals function
            def _lenRes(params, time_data, length_data, v_init):
                return np.sum((length_data - exponentialDeceleration(time_data, v_init, *params))**2)


            # Fit an exponential to the data
            res = scipy.optimize.basinhopping(_lenRes, p0, \
                minimizer_kwargs={"method": "Nelder-Mead", 'args':(time_data, length_data, traj.v_init)}, \
                niter=100)
            decel_fit = res.x

            decel_list.append(decel_fit)

            print('---------------')
            print('Fragment', frag_info.frag_dict[frag], 'fit:')
            print(decel_fit)


            # plt.plot(time_data, length_data, label='Observed')
            # plt.plot(time_data, exponentialDeceleration(time_data, traj.v_init, *decel_fit), label='fit')
            # plt.legend()
            # plt.xlabel('Time (s)')
            # plt.ylabel('Length (m)')
            # plt.title('Fragment {:d} fit'.format(frag_info.frag_dict[frag]))
            # plt.show()

            # Plot the residuals
            plt.plot(time_data, length_data - exponentialDeceleration(time_data, traj.v_init, *decel_fit))
            plt.xlabel('Time (s)')
            plt.ylabel('Length O - C (m)')
            plt.title('Fragment {:d} fit residuals'.format(frag_info.frag_dict[frag]))
            plt.show()


            ##################################################################################################



        # Generate a unique color for every fragment
        colors = plt.cm.rainbow(np.linspace(0, 1, len(fragments)))

        # Create a dictionary for every fragment-color pair
        colors_frags = {frag: color for frag, color in zip(fragments, colors)}

        # Make sure lags start at 0
        offset_vel_max = 0


        # Plot the positions of fragments from the beginning to the end
        # Calculate and plot the lag of all fragments
        for frag, decel_fit in zip(fragments, decel_list):

            # Select only the data points of the current fragment
            length_frag = [entry for entry in length_list if entry[0] == frag]

            # Find the last time of the fragment appearance
            last_time = max([entry[1] for entry in length_frag])

            # Extract the observed data
            _, time_data, length_data, lat_data, lon_data, height_data = np.array(length_frag).T

            # Plot the positions of fragments from the first time to the end, using fitted parameters
            # The lag is calculated by subtracting an "average" velocity length from the observed length
            time_array = np.linspace(-1, last_time, 1000)
            plt.plot(exponentialDeceleration(time_array, traj.v_init, *decel_fit) - exponentialDeceleration(time_array, \
                traj.v_init, 0, offset_vel_max, 0, 0), time_array, linestyle='--', color=colors_frags[frag], \
                linewidth=0.75)


            # Plot the observed data
            fake_lag = length_data - exponentialDeceleration(time_data, traj.v_init, 0, offset_vel_max, 0, 0)
            plt.plot(fake_lag, time_data, color=colors_frags[frag], linewidth=0.75)


            # Plot the fragment number at the end of each lag
            plt.text(fake_lag[-1] - 10, time_data[-1] + 0.02, str(frag_info.frag_dict[frag]), color=colors_frags[frag], \
                size=7, va='center', ha='right')


            # Check if the fragment has a fragmentation point and plot it
            if frag_info.frag_dict[frag] in frag_info.fragmentation_points:

                # Get the lag of the fragmentation point
                frag_point_time, fragments_list = frag_info.fragmentation_points[frag_info.frag_dict[frag]]
                frag_point_lag = exponentialDeceleration(frag_point_time, traj.v_init, *decel_fit) \
                    - exponentialDeceleration(frag_point_time, traj.v_init, 0, offset_vel_max, 0, 0)


                fragments_list = map(str, fragments_list)

                # Plot the fragmentation point
                plt.scatter(frag_point_lag, frag_point_time, s=20, zorder=4, color=colors_frags[frag], \
                    edgecolor='k', linewidth=0.5, label='Fragmentation: ' + ",".join(fragments_list))
            


        # Plot reference time
        plt.title('Reference time: ' + str(jd2Date(jd_ref, dt_obj=True)))

        plt.gca().invert_yaxis()
        plt.grid(color='0.9')

        plt.xlabel('Lag (m)')
        plt.ylabel('Time (s)')

        plt.ylim(ymax=-1)

        plt.legend()

        plt.savefig(os.path.join(dir_path, 'fragments_deceleration.png'), dpi=300)

        plt.show()


        time_min =  np.inf
        time_max = -np.inf
        ht_min =  np.inf
        ht_max = -np.inf

        # Calculate the dynamic pressure for the first fragment
        for frag, decel_fit in zip(fragments, decel_list):

            # Select only the data points of the current fragment
            length_frag = [entry for entry in length_list if entry[0] == frag]

            # Extract the observed data
            _, time_data, length_data, lat_data, lon_data, height_data = np.array(length_frag).T


            # Fit a linear dependance of time vs. height
            line_fit, _ = scipy.optimize.curve_fit(timeHeightFunc, time_data, height_data)


            # Get the time and height limits
            time_min = min(time_min, min(time_data))
            time_max = max(time_max, max(time_data))
            ht_min = min(ht_min, min(height_data))
            ht_max = max(ht_max, max(height_data))


            ### CALCULATE OBSERVED DYN PRESSURE

            # # Get the atmospheric densities at every heights
            # atm_dens = getAtmDensity_vect(lat_data, lon_data, height_data, jd_ref)

            # Get the velocity at every point in time
            velocities = exponentialDecelerationVel(time_data, traj.v_init, *decel_fit)

            # # Calculate the dynamic pressure
            # dyn_pressure = atm_dens*DRAG_COEFF*velocities**2

            # Calculate the dynamic pressure
            dyn_pressure = dynamicPressure(lat_data, lon_data, height_data, jd_ref, velocities)

            ###


            # Plot Observed height vs. dynamic pressure
            plt.plot(dyn_pressure/10**3, height_data/1000, color=colors_frags[frag], zorder=3, linewidth=0.75)

            # Plot the fragment number at the end of each lag
            plt.text(dyn_pressure[-1]/10**3, height_data[-1]/1000 - 0.02, str(frag_info.frag_dict[frag]), \
                color=colors_frags[frag], size=7, va='top', zorder=3)


            ### CALCULATE MODELLED DYN PRESSURE

            time_array = np.linspace(-1.0, max(time_data), 1000)

            # Calculate the modelled height
            height_array = timeHeightFunc(time_array, *line_fit)


            # Get the time and height limits
            time_min = min(time_min, min(time_array))
            time_max = max(time_max, max(time_array))
            ht_min = min(ht_min, min(height_array))
            ht_max = max(ht_max, max(height_array))


            # Get the atmospheric densities at every heights
            atm_dens_model = getAtmDensity_vect(np.zeros_like(time_array) + np.mean(lat_data), \
                np.zeros_like(time_array) + np.mean(lon_data), height_array, jd_ref)

            # Get the velocity at every point in time
            velocities_model = exponentialDecelerationVel(time_array, traj.v_init, *decel_fit)

            # Calculate the dynamic pressure
            dyn_pressure_model = atm_dens_model*DRAG_COEFF*velocities_model**2

            ###

            # Plot Modelled height vs. dynamic pressure
            plt.plot(dyn_pressure_model/10**3, height_array/1000, color=colors_frags[frag], zorder=3, \
                linewidth=0.75, linestyle='--')



            # Check if the fragment has a fragmentation point and plot it
            if frag_info.frag_dict[frag] in frag_info.fragmentation_points:

                # Get the lag of the fragmentation point
                frag_point_time, fragments_list = frag_info.fragmentation_points[frag_info.frag_dict[frag]]
                
                # Get the fragmentation height
                frag_point_height = timeHeightFunc(frag_point_time, *line_fit)

                # Calculate the velocity at fragmentation
                frag_point_velocity = exponentialDecelerationVel(frag_point_time, traj.v_init, *decel_fit)

                # Calculate the atm. density at the fragmentation point
                frag_point_atm_dens = getAtmDensity(np.mean(lat_data), np.mean(lon_data), frag_point_height, \
                    jd_ref)

                # Calculate the dynamic pressure at fragmentation in kPa
                frag_point_dyn_pressure = frag_point_atm_dens*DRAG_COEFF*frag_point_velocity**2
                frag_point_dyn_pressure /= 10**3

                # Compute height in km
                frag_point_height_km = frag_point_height/1000


                fragments_list = map(str, fragments_list)

                # Plot the fragmentation point
                plt.scatter(frag_point_dyn_pressure, frag_point_height_km, s=20, zorder=5, \
                    color=colors_frags[frag], edgecolor='k', linewidth=0.5, \
                    label='Fragmentation: ' + ",".join(fragments_list))



                ### Plot the errorbar 

                # Compute the lower veloicty estimate
                stddev_multiplier = 2.0


                # Check if the uncertainty exists
                if traj_uncert.v_init is None:
                    v_init_uncert = 0
                else:
                    v_init_uncert = traj_uncert.v_init


                # Compute the range of velocities
                lower_vel = frag_point_velocity - stddev_multiplier*v_init_uncert
                higher_vel = frag_point_velocity + stddev_multiplier*v_init_uncert

                # Assume the atmosphere density can vary +/- 50%
                lower_atm_dens = 0.5*frag_point_atm_dens
                higher_atm_dens = 1.5*frag_point_atm_dens

                # Compute lower and higher range for dyn pressure in kPa
                lower_frag_point_dyn_pressure = (lower_atm_dens*DRAG_COEFF*lower_vel**2)/10**3
                higher_frag_point_dyn_pressure = (higher_atm_dens*DRAG_COEFF*higher_vel**2)/10**3

                # Compute errors
                lower_error = abs(frag_point_dyn_pressure - lower_frag_point_dyn_pressure)
                higher_error = abs(frag_point_dyn_pressure - higher_frag_point_dyn_pressure)


                print(frag_point_dyn_pressure, frag_point_height_km, [lower_frag_point_dyn_pressure, higher_frag_point_dyn_pressure])

                # Plot the errorbar
                plt.errorbar(frag_point_dyn_pressure, frag_point_height_km, xerr=[[lower_error], [higher_error]], fmt='--', capsize=5, zorder=4, label='+/- 50% $\\rho_{atm}$, 2$\\sigma_v$ ')


                ######





        # Plot reference time
        plt.title('Reference time: ' + str(jd2Date(jd_ref, dt_obj=True)))

        plt.xlabel('Dynamic pressure (kPa)')
        plt.ylabel('Height (km)')

        plt.ylim([ht_min/1000, ht_max/1000])

        plt.legend()
        plt.grid(color='0.9')

        # Create the label for seconds
        ax2 = plt.gca().twinx()
        ax2.set_ylim([time_max, time_min])
        ax2.set_ylabel('Time (s)')

        plt.savefig(os.path.join(dir_path, 'fragments_dyn_pressures.png'), dpi=300)

        plt.show()
        


    pass





if __name__ == "__main__":


    # ### July 21, 2017 event

    # # Main directory
    # dir_path = "../MetalPrepare/20170721_070420_met/"

    # # .met file containing narrow-field picks
    # met_file = 'state_fragment_picks.met'


    # # Trajectory file
    # traj_file = 'Monte Carlo' + os.sep + "20170721_070419_mc_trajectory.pickle"


    # # DICTIONARY WHICH MAPS FRAGMENT IDs to FRAGMENTS TO PLOT
    # frag_dict = {0.0: 10, 1.0: 11, 2.0: 6, 3.0: 9, 4.0: 8, 5.0: 3, 6.0: 2, 7.0: 4, 8.0: 1, 9.0: 5, 10.0: 12, \
    #     11.0: 7}


    # # FRAGMENTATION POINTS (main fragments: time of fragmentation)
    # fragmentation_points = {
    #     2 : [0.0854, [2, 3]], # Fragments 2 and 3
    #     10: [0.066, [10, 11]],  # Fragments 10 and 11
    #     7:  [0.246, [7, 9, 12]],  # Fragments 7, 9, 12
    #     4:  [0.26, [4, 3]],   # 4 fragmented from 3
    #     8:  [0.046, [8, 5]]    # Fragments 8 and 5
    # }

    # frag_info = FragmentationInfo(frag_dict, fragmentation_points)


    # ######

    ### 20180703

    # Main directory
    dir_path = "/mnt/bulk/mirfitprepare/20180703_030839_met"

    # .met file containing narrow-field picks
    met_file = 'state_fragment_picks.met'


    # Trajectory file
    traj_file = 'Monte Carlo' + os.sep + "20180703_030839_mc_trajectory.pickle"

    # Trajectory uncertainties file
    traj_uncert_file = 'Monte Carlo' + os.sep + "20180703_030839_mc_uncertanties.pickle"



    # DICTIONARY WHICH MAPS FRAGMENT IDs to FRAGMENTS TO PLOT
    frag_dict = None


    # FRAGMENTATION POINTS (main fragments: time of fragmentation)
    fragmentation_points = {
        4 : [-0.18, [1, 2, 3, 4]]
    }

    frag_info = FragmentationInfo(frag_dict, fragmentation_points)


    ######

    


    ##########################################################################################################



    # Load the MET file
    met = loadMet(dir_path, met_file, mirfit=True)

    # Load the trajectory
    traj = loadPickle(dir_path, traj_file)

    # Load trajectory uncertainties
    traj_uncert = loadPickle(dir_path, traj_uncert_file)

    print(met)

    
    # Project narrow-field picks to wide-field trajectory
    projectNarrowPicks(dir_path, met, traj, traj_uncert, frag_info)