""" Takes measurements from the narrow field and projects them on the trajectory estimated from wide-field 
    data. 
"""

from __future__ import print_function, division, absolute_import

import os
import argparse
import json
from collections import OrderedDict

import numpy as np
import scipy.optimize
import scipy.interpolate
import matplotlib.pyplot as plt


from wmpl.Formats.Met import loadMet
from wmpl.Trajectory.Trajectory import lineFunc
from wmpl.MetSim.MetalMass import loadMetalMags
from wmpl.Utils.TrajConversions import raDec2ECI, geo2Cartesian_vect, altAz2RADec_vect, unixTime2JD, \
    cartesian2Geo, jd2Date
from wmpl.Utils.Math import findClosestPoints, vectMag
from wmpl.Utils.AtmosphereDensity import getAtmDensity, getAtmDensity_vect
from wmpl.Utils.Pickling import loadPickle
from wmpl.Utils.Physics import dynamicPressure, dynamicMass

# DRAG COEFFICIENT (assume unity)
DRAG_COEFF = 1.0




class NarrowProjectionInputData(object):
    def __init__(self, name, dir_path, met_file, traj_file, traj_uncert_file=None, metal_met_file=None, \
        frag_dict=None, fragmentation_points=None, fit_full_exp_model=False, v_init_adjust=0, 
        bulk_density=3500):
        """ Structure storing input data. 
    
        Arguments:
            name: [str] Name of the event.
            dir_path: [str] Full path to the directory with data.
            met_file: [str] Name of the mirfit .met file in the dir_path directory.
            traj_file: [str] Name of the .pickle trajectory file in the dir_path directory.

        Keyword arguments:
            traj_uncert_file: [str] Path to the MC uncertainties file in the dir_path directory. None by
                default, in which case the file will be tried to found in the same directory where
                the traj_file is.
            metal_met_file: [str] Name of the METAL .met file in the dir_path directory. 'state.met' is used 
                by default.
            frag_dict: [dict] Dictionary that maps the mirfit fragment IDs to desired numbers or names.
                None by default, in which case the fragments will be named from 1 to N.
            fragmentation_points: [dict] Determines where the fragmentation points are and in which points
                the dynamic pressure will be computed. Format:
                    {site_ID: {main fragment: [time of fragmentation, [indices of daughter fragments]],
                               main_fragment 2: ...},
                    #   ...}
            fit_full_exp_model: [bool] If True, the full exponential deceleration model will the fit, 
                including the velocity. If False, the initial velocity will be taken from the given
                trajectory and a simpler (more robust) model will be fitted to the lag.
            v_init_adjust: [float] Sometimes to get a good exponential fit, the initial velocity has to be
                adjusted on the order of +/- 100 m/s. The default value is 0 m/s.
            bulk_density: [float] Bulk density of the meteoroid in kg/m3 used for dynamic mass computation. 
                3500 kg/m3 by default.

        """

        self.name = name
        self.dir_path = dir_path
        self.met_file = met_file
        self.traj_file = traj_file


        if traj_uncert_file is None:

            # Try finding the trajectory undertainty file
            self.traj_uncert_file = self.traj_file.replace('_mc_trajectory', '_mc_uncertainties')

        else:
            self.traj_uncert_file = traj_uncert_file

        # If the file cannot be found, don't use the uncertainties
        if not os.path.isfile(os.path.join(self.dir_path, self.traj_uncert_file)):
            print('The trajectry uncertainties file cannot be found:', traj_uncert_file)
            self.traj_uncert_file = None


        if metal_met_file is None:
            self.metal_met_file = 'state.met'
        else:
            self.metal_met_file = metal_met_file


        # If the file cannot be found, don't use anything
        if not os.path.isfile(os.path.join(self.dir_path, self.metal_met_file)):
            print('The METAL .met file cannot be found:', self.metal_met_file)
            self.metal_met_file = None


        # Init the fragmentation info container
        self.frag_info = FragmentationInfo(frag_dict, fragmentation_points, v_init_adjust=v_init_adjust, \
        fit_full_exp_model=fit_full_exp_model, bulk_density=bulk_density)


        # Load the Mirfit .met file
        self.met = loadMet(self.dir_path, self.met_file)

        # Load the trajectory
        self.traj = loadPickle(self.dir_path, self.traj_file)


        # Load trajectory uncertainties
        if self.traj_uncert_file is not None:
            self.traj_uncert = loadPickle(dir_path, self.traj_uncert_file)
        else:
            self.traj_uncert = None


        # Load magnitudes from the METAL .met file
        if self.metal_met_file is not None:
            self.metal_mags = loadMetalMags(self.dir_path, self.metal_met_file)
        else:
            self.metal_mags = None




class FragmentationInfo(object):
    def __init__(self, frag_dict, fragmentation_points, v_init_adjust=0, fit_full_exp_model=False, 
        bulk_density=3500):
        """ Container for information about fragments and fragmentation points. """

        self.frag_dict = frag_dict

        #self.frag_dict_rev = dict((v,k) for k,v in frag_dict.iteritems())

        self.fragmentation_points = fragmentation_points

        self.v_init_adjust = v_init_adjust

        self.fit_full_exp_model = fit_full_exp_model

        self.bulk_density = bulk_density



def loadNarrowProjectionInputJSON(file_path):
    """ Load the input data from a JSON file. """     

    # Read the JSON file
    with open(file_path) as f:

        json_data = json.load(f)


    # Parse the input data into the input structure
    name = json_data['name']
    dir_path = json_data['dir_path']
    met_file = json_data['met_file']
    traj_file = json_data['traj_file']


    # Parse the fragmentation dictionary
    if 'frag_dict' in json_data:
        frag_dict = {float(key):json_data['frag_dict'][key] for key in json_data['frag_dict']}
    else:
        frag_dict = None



    # Parse the points of fragmentation
    if 'fragmentation_points' in json_data:

        fragmentation_points = {}

        # Convert frag IDs from string to int
        frag_pts = json_data['fragmentation_points']
        for site_id in frag_pts:

            site_frags = {}

            for frag_id in frag_pts[site_id]:
                site_frags[int(frag_id)] = frag_pts[site_id][frag_id]

            fragmentation_points[site_id] = site_frags

    else:
        fragmentation_points = None


    if 'traj_uncert_file' in json_data:
        traj_uncert_file = json_data['traj_uncert_file']
    else:
        traj_uncert_file = None


    if 'metal_met_file' in json_data:
        metal_met_file = json_data['metal_met_file']
    else:
        metal_met_file = None


    if 'v_init_adjust' in json_data:
        v_init_adjust = json_data['v_init_adjust']
    else:
        v_init_adjust = 0


    if 'fit_full_exp_model' in json_data:
        fit_full_exp_model = json_data['fit_full_exp_model']
    else:
        fit_full_exp_model = False



    if 'bulk_density' in json_data:
        bulk_density = json_data['bulk_density']
    else:
        bulk_density = 3500 # kg/m3


    # Init the input structure
    input_data = NarrowProjectionInputData(name, dir_path, met_file, traj_file, \
        traj_uncert_file=traj_uncert_file, metal_met_file=metal_met_file, frag_dict=frag_dict, \
        fragmentation_points=fragmentation_points, fit_full_exp_model=fit_full_exp_model, \
        v_init_adjust=v_init_adjust, bulk_density=bulk_density)


    return input_data




def exponentialDeceleration(t, v, d_t, k, a1, a2):
    """ Model for exponential deceleration. Returns the length at every point. """
    td = t + d_t
    return k + v*td - abs(a1)*np.exp(a2*td)



def exponentialDecelerationVel(t, v, d_t, k, a1, a2):
    """ Model for exponential deceleration. Returns the velocity at every point. """
    td = t + d_t
    return v - abs(a1*a2)*np.exp(a2*td)


def exponentialDecelerationDecel(t, v, d_t, k, a1, a2):
    """ Model for exponential deceleration. Returns the deceleration at every point. """
    td = t + d_t
    return -abs(a1*a2*a2)*np.exp(a2*td)




def projectNarrowPicks(dir_path, met, traj, traj_uncert, metal_mags, frag_info):
    """ Projects picks done in the narrow-field to the given trajectory. """


    # Adjust initial velocity
    frag_v_init = traj.v_init + frag_info.v_init_adjust

    # List for computed values to be stored in a file
    computed_values = []

    # Generate the file name prefix from the time (take from trajectory)
    file_name_prefix = traj.file_name


    # List that holds datetimes of fragmentations, used for the light curve plot
    fragmentations_datetime = []


    # Go through picks from all sites
    for site_no in met.picks:

        # Extract site exact plate
        exact = met.exact_plates[site_no]

        # Extract site picks
        picks = np.array(met.picks[site_no])


        # Skip the site if there are no picks
        if not len(picks):
            continue


        print()
        print('Processing site:', site_no)


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

        # Set the beginning time to the beginning of the widefield trajectory
        ref_beg_time = (traj.jdt_ref - jd_ref)*86400

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


            ### Fit the deceleration model to the length ###
            ##################################################################################################

            length_frag = np.array(length_frag)

            # Extract JDs and lengths into individual arrays
            time_data, length_data, lat_data, lon_data, height_data = length_frag.T


            if frag_info.fit_full_exp_model:

                # Fit the full exp deceleration model

                # First guess of the lag parameters
                p0 = [frag_v_init, 0, 0, traj.jacchia_fit[0], traj.jacchia_fit[1]]

                # Length residuals function
                def _lenRes(params, time_data, length_data):
                    return np.sum((length_data - exponentialDeceleration(time_data, *params))**2)


                # Fit an exponential to the data
                res = scipy.optimize.basinhopping(_lenRes, p0, \
                    minimizer_kwargs={"method": "BFGS", 'args':(time_data, length_data)}, \
                    niter=1000)
                decel_fit = res.x

            else:

                # Fit only the deceleration parameters

                # First guess of the lag parameters
                p0 = [0, 0, traj.jacchia_fit[0], traj.jacchia_fit[1]]

                # Length residuals function
                def _lenRes(params, time_data, length_data, v_init):
                    return np.sum((length_data - exponentialDeceleration(time_data, v_init, *params))**2)


                # Fit an exponential to the data
                res = scipy.optimize.basinhopping(_lenRes, p0, \
                    minimizer_kwargs={"method": "Nelder-Mead", 'args':(time_data, length_data, frag_v_init)}, \
                    niter=100)
                decel_fit = res.x


                # Add the velocity to the deceleration fit
                decel_fit = np.append(np.array([frag_v_init]), decel_fit)




            decel_list.append(decel_fit)

            print('---------------')
            print('Fragment', frag_info.frag_dict[frag], 'fit:')
            print(decel_fit)


            # plt.plot(time_data, length_data, label='Observed')
            # plt.plot(time_data, exponentialDeceleration(time_data, *decel_fit), label='fit')
            # plt.legend()
            # plt.xlabel('Time (s)')
            # plt.ylabel('Length (m)')
            # plt.title('Fragment {:d} fit'.format(frag_info.frag_dict[frag]))
            # plt.show()

            # # Plot the residuals
            # plt.plot(time_data, length_data - exponentialDeceleration(time_data, *decel_fit))
            # plt.xlabel('Time (s)')
            # plt.ylabel('Length O - C (m)')
            # plt.title('Fragment {:d} fit residuals'.format(frag_info.frag_dict[frag]))
            # plt.show()


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
            time_array = np.linspace(ref_beg_time, last_time, 1000)
            plt.plot(exponentialDeceleration(time_array, *decel_fit) - exponentialDeceleration(time_array, \
                frag_v_init, 0, offset_vel_max, 0, 0), time_array, linestyle='--', color=colors_frags[frag], \
                linewidth=0.75)


            # Plot the observed data
            fake_lag = length_data - exponentialDeceleration(time_data, frag_v_init, 0, offset_vel_max, 0, 0)
            plt.plot(fake_lag, time_data, color=colors_frags[frag], linewidth=0.75)


            # Plot the fragment number at the end of each lag
            plt.text(fake_lag[-1] - 10, time_data[-1] + 0.02, str(frag_info.frag_dict[frag]), color=colors_frags[frag], \
                size=7, va='center', ha='right')


            # Check if the fragment has a fragmentation point and plot it
            if site_no in frag_info.fragmentation_points:
                if frag_info.frag_dict[frag] in frag_info.fragmentation_points[site_no]:

                    # Get the lag of the fragmentation point
                    frag_point_time, fragments_list = frag_info.fragmentation_points[site_no][frag_info.frag_dict[frag]]
                    frag_point_lag = exponentialDeceleration(frag_point_time, *decel_fit) \
                        - exponentialDeceleration(frag_point_time, frag_v_init, 0, offset_vel_max, 0, 0)


                    fragments_list = list(map(str, fragments_list))


                    # Save the fragmentation time in the list for light curve plot
                    fragmentations_datetime.append([jd2Date(jd_ref + frag_point_time/86400, dt_obj=True), \
                        fragments_list])

                    # Plot the fragmentation point
                    plt.scatter(frag_point_lag, frag_point_time, s=20, zorder=4, color=colors_frags[frag], \
                        edgecolor='k', linewidth=0.5, label='Fragmentation: ' + ",".join(fragments_list))
            


        # Plot reference time
        plt.title('Reference time: ' + str(jd2Date(jd_ref, dt_obj=True)))

        plt.gca().invert_yaxis()
        plt.grid(color='0.9')

        plt.xlabel('Lag (m)')
        plt.ylabel('Time (s)')

        plt.ylim(ymax=ref_beg_time)

        plt.legend()

        plt.savefig(os.path.join(dir_path, file_name_prefix \
            + '_fragments_deceleration_site_{:s}.png'.format(str(site_no))), dpi=300)

        plt.show()



        time_min =  np.inf
        time_max = -np.inf
        ht_min =  np.inf
        ht_max = -np.inf

        ### PLOT DYNAMIC PRESSURE FOR EVERY FRAGMENT
        for frag, decel_fit in zip(fragments, decel_list):

            # Select only the data points of the current fragment
            length_frag = [entry for entry in length_list if entry[0] == frag]

            # Extract the observed data
            _, time_data, length_data, lat_data, lon_data, height_data = np.array(length_frag).T


            # Fit a linear dependance of time vs. height
            line_fit, _ = scipy.optimize.curve_fit(lineFunc, time_data, height_data)


            # Get the time and height limits
            time_min = min(time_min, min(time_data))
            time_max = max(time_max, max(time_data))
            ht_min = min(ht_min, min(height_data))
            ht_max = max(ht_max, max(height_data))


            ### CALCULATE OBSERVED DYN PRESSURE

            # Get the velocity at every point in time
            velocities = exponentialDecelerationVel(time_data, *decel_fit)

            # Calculate the dynamic pressure
            dyn_pressure = dynamicPressure(lat_data, lon_data, height_data, jd_ref, velocities)

            ###


            # Plot Observed height vs. dynamic pressure
            plt.plot(dyn_pressure/10**3, height_data/1000, color=colors_frags[frag], zorder=3, linewidth=0.75)

            # Plot the fragment number at the end of each lag
            plt.text(dyn_pressure[-1]/10**3, height_data[-1]/1000 - 0.02, str(frag_info.frag_dict[frag]), \
                color=colors_frags[frag], size=7, va='top', zorder=3)


            ### CALCULATE MODELLED DYN PRESSURE

            time_array = np.linspace(ref_beg_time, max(time_data), 1000)

            # Calculate the modelled height
            height_array = lineFunc(time_array, *line_fit)


            # Get the time and height limits
            time_min = min(time_min, min(time_array))
            time_max = max(time_max, max(time_array))
            ht_min = min(ht_min, min(height_array))
            ht_max = max(ht_max, max(height_array))


            # Get the atmospheric densities at every heights
            atm_dens_model = getAtmDensity_vect(np.zeros_like(time_array) + np.mean(lat_data), \
                np.zeros_like(time_array) + np.mean(lon_data), height_array, jd_ref)

            # Get the velocity at every point in time
            velocities_model = exponentialDecelerationVel(time_array, *decel_fit)

            # Calculate the dynamic pressure
            dyn_pressure_model = atm_dens_model*DRAG_COEFF*velocities_model**2

            ###

            # Plot Modelled height vs. dynamic pressure
            plt.plot(dyn_pressure_model/10**3, height_array/1000, color=colors_frags[frag], zorder=3, \
                linewidth=0.75, linestyle='--')



            # Check if the fragment has a fragmentation point and plot it
            if site_no in frag_info.fragmentation_points:
                if frag_info.frag_dict[frag] in frag_info.fragmentation_points[site_no]:

                    # Get the lag of the fragmentation point
                    frag_point_time, fragments_list = frag_info.fragmentation_points[site_no][frag_info.frag_dict[frag]]
                    
                    # Get the fragmentation height
                    frag_point_height = lineFunc(frag_point_time, *line_fit)

                    # Calculate the velocity at fragmentation
                    frag_point_velocity = exponentialDecelerationVel(frag_point_time, *decel_fit)

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

                    # Assume the atmosphere density can vary +/- 50% (Gunther's analysis)
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
                    plt.errorbar(frag_point_dyn_pressure, frag_point_height_km, \
                        xerr=[[lower_error], [higher_error]], fmt='--', capsize=5, zorder=4, \
                        color=colors_frags[frag], label='+/- 50% $\\rho_{atm}$, 2$\\sigma_v$ ')


                    # Save the computed fragmentation values to list
                    # Site, Reference JD, Relative time, Fragment ID, Height, Dyn pressure, Dyn pressure lower \
                    #   bound, Dyn pressure upper bound
                    computed_values.append([site_no, jd_ref, frag_point_time, frag_info.frag_dict[frag], \
                        frag_point_height_km, frag_point_dyn_pressure, lower_frag_point_dyn_pressure, \
                        higher_frag_point_dyn_pressure])


                    ######





        # Plot reference time
        plt.title('Reference time: ' + str(jd2Date(jd_ref, dt_obj=True)))

        plt.xlabel('Dynamic pressure (kPa)')
        plt.ylabel('Height (km)')

        plt.ylim([ht_min/1000, ht_max/1000])


        # Remove repeating labels and plot the legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        
        plt.grid(color='0.9')

        # Create the label for seconds
        ax2 = plt.gca().twinx()
        ax2.set_ylim([time_max, time_min])
        ax2.set_ylabel('Time (s)')

        plt.savefig(os.path.join(dir_path, file_name_prefix \
            + '_fragments_dyn_pressures_site_{:s}.png'.format(str(site_no))), dpi=300)

        plt.show()




        ### PLOT DYNAMICS MASSES FOR ALL FRAGMENTS
        for frag, decel_fit in zip(fragments, decel_list):

            # Select only the data points of the current fragment
            length_frag = [entry for entry in length_list if entry[0] == frag]

            # Extract the observed data
            _, time_data, length_data, lat_data, lon_data, height_data = np.array(length_frag).T


            # Fit a linear dependance of time vs. height
            line_fit, _ = scipy.optimize.curve_fit(lineFunc, time_data, height_data)


            ### CALCULATE OBSERVED DYN MASS

            # Get the velocity at every point in time
            velocities = exponentialDecelerationVel(time_data, *decel_fit)

            decelerations = np.abs(exponentialDecelerationDecel(time_data, *decel_fit))

            # Calculate the dynamic mass
            dyn_mass = dynamicMass(frag_info.bulk_density, lat_data, lon_data, height_data, jd_ref, \
                velocities, decelerations)

            ###


            # Plot Observed height vs. dynamic pressure
            plt.plot(dyn_mass*1000, height_data/1000, color=colors_frags[frag], zorder=3, linewidth=0.75)

            # Plot the fragment number at the end of each lag
            plt.text(dyn_mass[-1]*1000, height_data[-1]/1000 - 0.02, str(frag_info.frag_dict[frag]), \
                color=colors_frags[frag], size=7, va='top', zorder=3)



            ### CALCULATE MODELLED DYN MASS

            time_array = np.linspace(ref_beg_time, max(time_data), 1000)

            # Calculate the modelled height
            height_array = lineFunc(time_array, *line_fit)


            # Get the velocity at every point in time
            velocities_model = exponentialDecelerationVel(time_array, *decel_fit)

            # Get the deceleration
            decelerations_model = np.abs(exponentialDecelerationDecel(time_array, *decel_fit))

            # Calculate the modelled dynamic mass
            dyn_mass_model = dynamicMass(frag_info.bulk_density, 
                np.zeros_like(time_array) + np.mean(lat_data), 
                np.zeros_like(time_array) + np.mean(lon_data), height_array, jd_ref, \
                velocities_model, decelerations_model)

            

            ###

            # Plot Modelled height vs. dynamic mass
            plt.plot(dyn_mass_model*1000, height_array/1000, color=colors_frags[frag], zorder=3, \
                linewidth=0.75, linestyle='--', \
                label='Frag {:d} initial dyn mass = {:.1e} g'.format(frag_info.frag_dict[frag], \
                    1000*dyn_mass_model[0]))



        # Plot reference time
        plt.title('Reference time: ' + str(jd2Date(jd_ref, dt_obj=True)) \
            + ', $\\rho_m = ${:d} $kg/m^3$'.format(frag_info.bulk_density))

        plt.xlabel('Dynamic mass (g)')
        plt.ylabel('Height (km)')

        plt.ylim([ht_min/1000, ht_max/1000])


        # Remove repeating labels and plot the legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        
        plt.grid(color='0.9')

        # Create the label for seconds
        ax2 = plt.gca().twinx()
        ax2.set_ylim([time_max, time_min])
        ax2.set_ylabel('Time (s)')

        plt.savefig(os.path.join(dir_path, file_name_prefix \
            + '_fragments_dyn_mass_site_{:s}.png'.format(str(site_no))), dpi=300)

        plt.show()






    # Plot the light curve if the METAL .met file was given
    if (metal_mags is not None):

        # Make sure there are lightcurves in the data
        if len(metal_mags):

            lc_min = np.inf
            lc_max = -np.inf

            # Plot the lightcurves
            for site_entry in metal_mags:

                site_id, time, mags = site_entry

                # Track the minimum and maximum magnitude
                lc_min = np.min([lc_min, np.min(mags)])
                lc_max = np.max([lc_max, np.max(mags)])

                plt.plot(time, mags, marker='+', label='Site: ' + str(site_id), zorder=4, linewidth=1)


            # Plot times of fragmentation
            for frag_dt, fragments_list in fragmentations_datetime:

                # Plot the lines of fragmentation
                y_arr = np.linspace(lc_min, lc_max, 10)
                x_arr = [frag_dt]*len(y_arr)

                plt.plot(x_arr, y_arr, linestyle='--', zorder=4, \
                    label='Fragmentation: ' + ",".join(fragments_list))


            plt.xlabel('Time (UTC)')
            plt.ylabel('Absolute magnitude (@100km)')

            plt.grid()

            plt.gca().invert_yaxis()

            plt.legend()

            ### Format the X axis datetimes
            import matplotlib

            def formatDT(x, pos=None):

                x = matplotlib.dates.num2date(x)

                # Add date to the first tick
                if pos == 0:
                    fmt = '%D %H:%M:%S.%f'
                else:
                    fmt = '%H:%M:%S.%f'

                label = x.strftime(fmt)[:-3]
                label = label.rstrip("0")
                label = label.rstrip(".")

                return label

            from matplotlib.ticker import FuncFormatter

            plt.gca().xaxis.set_major_formatter(FuncFormatter(formatDT))
            plt.gca().xaxis.set_minor_formatter(FuncFormatter(formatDT))

            ###


            plt.tight_layout()

            # Save the figure
            plt.savefig(os.path.join(dir_path, file_name_prefix + '_fragments_light_curve_comparison.png'), \
                dpi=300)

            plt.show()




    # Save the computed values to file
    with open(os.path.join(dir_path, file_name_prefix + "_fragments_dyn_pressure_info.txt"), 'w') as f:

        # Site, Reference JD, Relative time, Fragment ID, Height, Dyn pressure, Dyn pressure lower \
                #   bound, Dyn pressure upper bound

        # Write the header
        f.write("# Site,               Ref JD,  Rel time, Frag ID, Ht (km),  DP (kPa),   DP low,  DP high\n")

        # Write computed values for every fragment
        for entry in computed_values:
            f.write(" {:>5s}, {:20.12f}, {:+8.6f}, {:7d}, {:7.3f}, {:9.2f}, {:8.2f}, {:8.2f}\n".format(*entry))
        





if __name__ == "__main__":


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="""Choose which input data will be used.""",
        formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('input_file', type=str, help="""Path to JSON file with input data.""")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    ############################


    # Load inputs from a JSON file
    input_data = loadNarrowProjectionInputJSON(cml_args.input_file)


    ##########################################################################################################



    # Project narrow-field picks to wide-field trajectory
    projectNarrowPicks(input_data.dir_path, input_data.met, input_data.traj, input_data.traj_uncert, \
        input_data.metal_mags, input_data.frag_info)


