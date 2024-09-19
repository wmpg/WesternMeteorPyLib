""" Tools for automated trajectory pairing. """

from __future__ import print_function, division, absolute_import

import copy
import datetime
import json
import multiprocessing

import numpy as np

from wmpl.Trajectory.Trajectory import ObservedPoints, PlaneIntersection, Trajectory
from wmpl.Utils.Earth import greatCircleDistance
from wmpl.Utils.Math import vectNorm, vectMag, angleBetweenVectors, vectorFromPointDirectionAndAngle, \
    findClosestPoints, generateDatetimeBins, meanAngle, angleBetweenSphericalCoords
from wmpl.Utils.ShowerAssociation import associateShowerTraj
from wmpl.Utils.TrajConversions import J2000_JD, geo2Cartesian, cartesian2Geo, raDec2AltAz, altAz2RADec, \
    raDec2ECI, datetime2JD, jd2Date, equatorialCoordPrecession_vect




class TrajectoryConstraints(object):
    def __init__(self):
        """ Container for trajectory constraints. """


        # Minimum number of measurement points per observation
        self.min_meas_pts = 4


        # Max time difference between meteor observations
        self.max_toffset = 10.0


        # Minimum distance between stations (km)
        self.min_station_dist = 5.0

        # Maximum distance between stations (km)
        self.max_station_dist = 600.0


        # Minimum convergence angle (deg)
        self.min_qc = 3.0


        # Maximum number of stations included in the trajectory solution (for speed)
        self.max_stations = 8


        ### Velocity filters ###

        # Max difference between velocities from difference stations (percent)
        self.max_vel_percent_diff = 25.0

        # Minimum and maximum average velocities (km/s)
        self.v_avg_min = 3.0
        self.v_avg_max = 73.0

        ### ###


        ### Height filters ###

        # Maximum begin height (km)
        self.max_begin_ht = 150

        # Minimum begin height (km)
        self.min_begin_ht = 50.0

        # Maximum end height (km)
        self.max_end_ht = 130.0

        # Minimum end height (km)
        self.min_end_ht = 20.0

        # Force begin height to the higher than the end height
        self.force_hb_gt_he = True

        ### ###


        ### Magnitude filters ###

        # If the first and last 2 points are N magnitudes fainter than the median magnitude, ignore them
        self.mag_filter_endpoints = 3

        ### ###


        ### Trajectory candidate merging ###

        # Maximum angle between radiants of candidate trajectories estimated using intersecting planes method
        #   for them to be merged into one trajectory (deg)
        self.max_merge_radiant_angle = 15


        ### ###


        ### MC solver settings ###

        # Run Monte Carlo or not
        self.run_mc = True

        # Only compute geometric uncertainties. If False, the solver will cull all MC runs with time residuals
        #   worse than the initial LoS fit
        self.geometric_uncert = True

        # Number of CPU cores to use for parallel processing
        self.mc_cores = multiprocessing.cpu_count() - 2
        if self.mc_cores < 2:
            self.mc_cores = 2

        # MC runs to run for error estimation
        self.error_mc_runs = 10

        # Convergence angle below which more MC runs will be used (deg)
        self.low_qc_threshold = 15.0

        # Number of MC runs to run for low Qc trajectories
        self.low_qc_mc_runs = 20

        # Save plots to disk
        self.save_plots = False

        ### ###


        ### Trajectory filters ###
        ### These filter will be applied after the trajectory has been computed

        # It will remove all stations with residuals larger than this fixed amount in arc seconds
        self.max_arcsec_err = 180.0

        # It will keep all stations with residuals below this error, regardless of other filters
        self.min_arcsec_err = 30.0


        # Bad station filter. If there are 3 or more stations, it will remove those that have angular 
        #   residuals <bad_station_obs_ang_limit> times larger than the mean of all other stations
        self.bad_station_obs_ang_limit = 2.0


        ### ###



class TrajectoryCorrelator(object):
    def __init__(self, data_handle, traj_constraints, v_init_part, data_in_j2000=True):
        """ Correlates meteor trajectories using meteor data given to it through a data handle. A data handle
        is a class instance with a common interface between e.g. files on the disk in various formats or 
        meteors in a database.
        """

        self.dh = data_handle

        self.traj_constraints = traj_constraints

        # Smallest part of the meteor (fraction) used for automated velocity estimation using the sliding fit
        self.v_init_part = v_init_part


        # Indicate that the data is in J2000
        self.data_in_j2000 = data_in_j2000



    def trajectoryRangeCheck(self, traj_reduced, platepar):
        """ Check that the trajectory is within the range limits. 
        
        Arguments:
            traj_reduced: [TrajectoryReducted object]
            platepar: [Platepar object]

        Return:
            [bool] True if either the trajectory beg/end are within the distance limit, False otherwise.
        """

        # Compute distance between station and trajectory beginning
        beg_dist = greatCircleDistance(traj_reduced.rbeg_lat, traj_reduced.rbeg_lon, \
            np.radians(platepar.lat), np.radians(platepar.lon))


        # Compute distance between station and trajectory end
        end_dist = greatCircleDistance(traj_reduced.rend_lat, traj_reduced.rend_lon, \
            np.radians(platepar.lat), np.radians(platepar.lon))


        # Check that the trajectory beg or end is within the limits
        if (beg_dist <= self.traj_constraints.max_station_dist) \
            or (end_dist <= self.traj_constraints.max_station_dist):

            return True

        else:
            print("Distance between station and trajectory too large!")
            return False



    def trajectoryInFOV(self, traj_reduced, platepar):
        """ Check that the trajectory is within the FOV of the camera. 
        
        Arguments:
            traj_reduced: [TrajectoryReducted object]
            platepar: [Platepar object]

        Return:
            [bool] True if the trajectory is within the FOV, False otherwise.
        """

        # Compute the FOV diagonal
        fov = np.radians(np.sqrt(platepar.fov_v**2 + platepar.fov_h**2))

        # Station coordinates in radians and meters
        lat, lon, elev = np.radians(platepar.lat), np.radians(platepar.lon), platepar.elev

        # Compute ECI coordinates of the station at the reference meteor time
        stat_eci = np.array(geo2Cartesian(lat, lon, elev, traj_reduced.jdt_ref))

        # Compute ECI vectors of the FOV centre
        azim, alt = raDec2AltAz(np.radians(platepar.RA_d), np.radians(platepar.dec_d), platepar.JD, lat, lon)
        ra, dec = altAz2RADec(azim, alt, traj_reduced.jdt_ref, lat, lon)
        fov_eci = vectNorm(np.array(raDec2ECI(ra, dec)))

        # Compute the closest point of approach between the middle of the FOV and the trajectory line
        obs_cpa, rad_cpa, d = findClosestPoints(stat_eci, fov_eci, np.array(traj_reduced.state_vect_mini), \
            np.array(traj_reduced.radiant_eci_mini))

        # Convert to unit vectors
        obs_cpa = vectNorm(obs_cpa)
        rad_cpa = vectNorm(rad_cpa)


        # Check that the angle between the closest points of approach is within the FOV
        if angleBetweenVectors(obs_cpa, rad_cpa) <= fov/2:
            return True

        else:
            print("Trajectory not in station FOV!")
            return False

    

    def stationRangeCheck(self, rp, tp):
        """ Check if the two stations are within the set maximum distance. 

        Arguments:
            rp: [Platepar] Reference platepar.
            tp: [Platepar] Test platepar.
        """

        # Compute the distance between stations (km)
        dist = greatCircleDistance(np.radians(rp.lat), np.radians(rp.lon), np.radians(tp.lat), \
            np.radians(tp.lon))

        print("Distance between {:s} and {:s} = {:.1f} km".format(rp.station_code, tp.station_code, dist))

        if (dist < self.traj_constraints.min_station_dist) \
            or (dist > self.traj_constraints.max_station_dist):

            print("Rejecting station combination...")
            return False
        else:
            return True



    def checkFOVOverlap(self, rp, tp):
        """ Check if two stations have overlapping fields of view between heights of 50 to 115 km.
        
        Arguments:
            rp: [Platepar] Reference platepar.
            tp: [Platepar] Test platepar.

        Return:
            [bool] True if FOVs overlap, False otherwise.
        """

        # Compute the FOV diagonals of both stations
        reference_fov = np.radians(np.sqrt(rp.fov_v**2 + rp.fov_h**2))
        test_fov = np.radians(np.sqrt(tp.fov_v**2 + tp.fov_h**2))


        lat1, lon1, elev1 = np.radians(rp.lat), np.radians(rp.lon), rp.elev
        lat2, lon2, elev2 = np.radians(tp.lat), np.radians(tp.lon), tp.elev

        # Compute alt/az of the FOV centre
        azim1, alt1 = raDec2AltAz(np.radians(rp.RA_d), np.radians(rp.dec_d), rp.JD, lat1, lon1)
        azim2, alt2 = raDec2AltAz(np.radians(tp.RA_d), np.radians(tp.dec_d), tp.JD, lat2, lon2)


        # Use now as a reference time for FOV overlap check
        ref_jd = datetime2JD(datetime.datetime.now(datetime.timezone.utc))

        # Compute ECI coordinates of both stations
        reference_stat_eci = np.array(geo2Cartesian(lat1, lon1, elev1, ref_jd))
        test_stat_eci = np.array(geo2Cartesian(lat2, lon2, elev2, ref_jd))

        # Compute ECI vectors of the FOV centre
        ra1, dec1 = altAz2RADec(azim1, alt1, ref_jd, lat1, lon1)
        reference_fov_eci = vectNorm(np.array(raDec2ECI(ra1, dec1)))
        ra2, dec2 = altAz2RADec(azim2, alt2, ref_jd, lat2, lon2)
        test_fov_eci = vectNorm(np.array(raDec2ECI(ra2, dec2)))

        # Compute ECI coordinates at different heights along the FOV line and check for FOV overlap
        # The checked heights are 50, 70, 95, and 115 km (ordered by overlap probability for faster 
        # execution)
        for height_above_ground in [95000, 70000, 115000, 50000]:

            # Compute points in the middle of FOVs of both stations at given heights
            reference_fov_point = reference_stat_eci + reference_fov_eci*(height_above_ground \
                - elev1)/np.sin(alt1)
            test_fov_point = test_stat_eci + test_fov_eci*(height_above_ground - elev2)/np.sin(alt2)

            # Check if the middles of the FOV are in the other camera's FOV
            if (angleBetweenVectors(reference_fov_eci, test_fov_point - reference_stat_eci) <= reference_fov/2) \
                or (angleBetweenVectors(test_fov_eci, reference_fov_point - test_stat_eci) <= test_fov/2):

                return True

            # Compute vectors pointing from one station's point on the FOV line to the other
            reference_to_test = vectNorm(test_fov_point - reference_fov_point)
            test_to_reference = -reference_to_test

            # Compute vectors from the ground to those points
            reference_fov_gnd = reference_fov_point - reference_stat_eci
            test_fov_gnd = test_fov_point - test_stat_eci

            # Compute vectors moved towards the other station by half the FOV diameter
            reference_moved = reference_stat_eci + vectorFromPointDirectionAndAngle(reference_fov_gnd, \
                reference_to_test, reference_fov/2)
            test_moved = test_stat_eci + vectorFromPointDirectionAndAngle(test_fov_gnd, test_to_reference, \
                test_fov/2)

            # Compute the vector pointing from one station to the moved point of the other station
            reference_to_test_moved = vectNorm(test_moved - reference_stat_eci)
            test_to_reference_moved = vectNorm(reference_moved - test_stat_eci)


            # Check if the FOVs overlap
            if (angleBetweenVectors(reference_fov_eci, reference_to_test_moved) <= reference_fov/2) \
                or (angleBetweenVectors(test_fov_eci, test_to_reference_moved) <= test_fov/2):

                return True


        return False


    def initObservationsObject(self, met, pp, ref_dt=None):
        """ Init the observations object which will be fed into the trajectory solver. """

        # If the reference datetime is given, apply a time offset
        if ref_dt is not None:
            # Compute the time offset that will have to be added to the relative time
            time_offset = (met.reference_dt - ref_dt).total_seconds()
        else:
            ref_dt = met.reference_dt
            time_offset = 0

        ra_data = np.array([np.radians(entry.ra) for entry in met.data])
        dec_data = np.array([np.radians(entry.dec) for entry in met.data])
        time_data = np.array([entry.time_rel + time_offset for entry in met.data])
        mag_data = np.array([entry.mag for entry in met.data])

        # If the data is in J2000, precess it to the epoch of date
        if self.data_in_j2000:
            jdt_ref_vect = np.zeros_like(ra_data) + datetime2JD(ref_dt)
            ra_data, dec_data = equatorialCoordPrecession_vect(J2000_JD.days, jdt_ref_vect, ra_data, dec_data)


        # If the two begin and end points are fainter by N magnitudes from the median, ignore them
        ignore_list = np.zeros_like(mag_data)
        if len(mag_data) > 5:

            median_mag = np.median(mag_data)
            indices = [0, 1, -1, -2]

            for ind in indices:
                if mag_data[ind] > (median_mag + self.traj_constraints.mag_filter_endpoints):
                    ignore_list[ind] = 1


        # Set the FF file name as the comment
        comment_dict = {}
        if met.ff_name is not None:
            comment_dict['ff_name'] = met.ff_name


        # Convert the comment dictionary to a JSON string
        comment = json.dumps(comment_dict, sort_keys=True).replace('\n', '').replace('\r', '')


        # Init the observation object
        obs = ObservedPoints(datetime2JD(ref_dt), ra_data, dec_data, time_data, np.radians(pp.lat), \
            np.radians(pp.lon), pp.elev, meastype=1, station_id=pp.station_code, magnitudes=mag_data, \
            ignore_list=ignore_list, fov_beg=met.fov_beg, fov_end=met.fov_end, comment=comment)

        return obs



    def projectPointToTrajectory(self, indx, obs, plane_intersection):
        """ Compute lat, lon and height of given point on the meteor trajectory. """

        meas_vector = obs.meas_eci[indx]
        jd = obs.JD_data[indx]

        # Calculate closest points of approach (observed line of sight to radiant line)
        _, rad_cpa, _ = findClosestPoints(obs.stat_eci, meas_vector, plane_intersection.cpa_eci, \
            plane_intersection.radiant_eci)

        lat, lon, elev = cartesian2Geo(jd, *rad_cpa)

        # Compute lat, lon, elev
        return rad_cpa, lat, lon, elev



    def quickTrajectorySolution(self, obs1, obs2):
        """ Perform an intersecting planes solution and check if it satisfies specified sanity checks. """

        # Do the plane intersection solution
        plane_intersection = PlaneIntersection(obs1, obs2)

        ra_cand, dec_cand = plane_intersection.radiant_eq
        print("Candidate radiant: RA = {:.3f}, Dec = {:+.3f}".format(np.degrees(ra_cand), \
            np.degrees(dec_cand)))
        

        ### Compute meteor begin and end points
        eci1_beg, lat1_beg, lon1_beg, ht1_beg = self.projectPointToTrajectory(0, obs1, plane_intersection)
        eci1_end, lat1_end, lon1_end, ht1_end = self.projectPointToTrajectory(-1, obs1, plane_intersection)
        eci2_beg, lat2_beg, lon2_beg, ht2_beg = self.projectPointToTrajectory(0, obs2, plane_intersection)
        eci2_end, lat2_end, lon2_end, ht2_end = self.projectPointToTrajectory(-1, obs2, plane_intersection)

        # Convert heights to kilometers
        ht1_beg /= 1000
        ht1_end /= 1000
        ht2_beg /= 1000
        ht2_end /= 1000

        ### ###


        ### Check if the meteor begin and end points are within the specified range ###

        # Check the end height is lower than begin height
        if (ht1_end > ht1_beg) or (ht2_end > ht2_beg):
            print("Begin height lower than the end height!")
            return None

        # Check if begin height are within the specified range
        if (ht1_beg > self.traj_constraints.max_begin_ht) \
            or (ht1_beg < self.traj_constraints.min_begin_ht) \
            or (ht2_beg > self.traj_constraints.max_begin_ht) \
            or (ht2_beg < self.traj_constraints.min_begin_ht) \
            or (ht1_end > self.traj_constraints.max_end_ht) \
            or (ht1_end < self.traj_constraints.min_end_ht) \
            or (ht2_end > self.traj_constraints.max_end_ht) \
            or (ht2_end < self.traj_constraints.min_end_ht):

            print("Meteor heights outside allowed range!")
            print("H1_beg: {:.2f}, H1_end: {:.2f}".format(ht1_beg, ht1_end))
            print("H2_beg: {:.2f}, H2_end: {:.2f}".format(ht2_beg, ht2_end))

            return None

        ### ###
        

        ### Check if the velocity is consistent ###

        # Compute the average velocity from both stations (km/s)
        vel1 = vectMag(eci1_end - eci1_beg)/(obs1.time_data[-1] - obs1.time_data[0])/1000
        vel2 = vectMag(eci2_end - eci2_beg)/(obs2.time_data[-1] - obs2.time_data[0])/1000

        # Check if they are within a certain percentage difference
        percent_diff = 100*abs(vel1 - vel2)/max(vel1, vel2)

        if percent_diff > self.traj_constraints.max_vel_percent_diff:

            print("Velocity difference too high: {:.2f} vs {:.2f} km/s".format(vel1/1000, vel2/1000))
            return None


        # Check the velocity range
        v_avg = (vel1 + vel2)/2
        if (v_avg < self.traj_constraints.v_avg_min) or (v_avg > self.traj_constraints.v_avg_max):
            
            print("Average veocity outside velocity bounds: {:.1f} < {:.1f} < {:.1f}".format(self.traj_constraints.v_avg_min, \
                v_avg, self.traj_constraints.v_avg_max))
            return None



        ### ###

        return plane_intersection



    def initTrajectory(self, jdt_ref, mc_runs):
        """ Initialize the Trajectory solver.
        
        Arguments:
            jdt_ref: [datetime] Reference Julian date.
            mc_runs: [int] Number of Monte Carlo runs.

        Return:
            traj: [Trajectory]
        """

        traj = Trajectory(jdt_ref, \
            max_toffset=self.traj_constraints.max_toffset, meastype=1, \
            v_init_part=self.v_init_part, monte_carlo=self.traj_constraints.run_mc, \
            mc_runs=mc_runs, show_plots=False, verbose=False, save_results=False, \
            reject_n_sigma_outliers=2, mc_cores=self.traj_constraints.mc_cores, \
            geometric_uncert=self.traj_constraints.geometric_uncert)

        return traj



    def solveTrajectory(self, traj, mc_runs):
        """ Given an initialized Trajectory object with observation, run the solver and automatically
            reject bad observations.

        Arguments:
            traj: [Trajectory object]
            mc_runs: [int] Number of Monte Carlo runs.

        Return:

        """

        # Reference Julian date (the one in the traj object may change later, after timing offset estimation,
        #   so we keep this one as a "hard" reference)
        jdt_ref = traj.jdt_ref

        # Disable Monte Carlo runs until an initial stable set of observations is found
        traj.monte_carlo = False

        # Run the solver
        try:
            traj_status = traj.run()

        # If solving has failed, stop solving the trajectory
        except ValueError:
            print("Error during trajectory estimation!")
            return False, None


        # Reject bad observations until a stable set is found, but only if there are more than 2    
        #   stations. Only one station will be rejected at one point in time
        successful_traj_fit = False
        skip_trajectory = False
        traj_best = None
        ignored_station_dict = {}
        for _ in range(len(traj.observations)):
        
            # If the trajectory estimation failed, skip this trajectory
            if traj_status is None:
                print("Trajectory estimation failed!")
                skip_trajectory = True
                break

            # Store the "best" trajectory if it is good
            else:

                # If there's no best trajectory, store the current one as best
                if traj_best is None:
                    traj_best = copy.deepcopy(traj_status)

                # Check if the current trajectory has smaller median residuals than the best
                #    trajectory and store it as the best trajectory
                elif np.median([obstmp.ang_res_std for obstmp in traj_status.observations 
                    if not obstmp.ignore_station]) < np.median([obstmp.ang_res_std for obstmp \
                    in traj_best.observations if not obstmp.ignore_station]):

                    traj_best = copy.deepcopy(traj_status)


            # Skip this part if there are less than 3 stations
            if len(traj.observations) < 3:
                break


            # If there are less than 2 stations that are not ignored, skip this solution
            if len([obstmp for obstmp in traj_status.observations if not obstmp.ignore_station]) < 2:
                print("Skipping trajectory solution, not enough good observations...")
                skip_trajectory = True
                break

            print()


            ### Check for bad observations and rerun the solution if necessary ###

            
            any_ignored_toggle = False
            ignore_candidates = {}

            # a) Reject all observations which have angular residuals <bad_station_obs_ang_limit>
            #   times larger than the median of all other observations
            # b) Reject all observations with higher residuals than the fixed limit
            # c) Keep all observations with error inside the minimum error limit, even though they
            #   might have been rejected in a previous iteration
            # d) Only reject a maximum of 50% of stations
            max_rejections_possible = int(np.ceil(0.5*len(traj_status.observations)))
            for i, obs in enumerate(traj_status.observations):

                # Compute the median angular uncertainty of all other non-ignored stations
                ang_res_list = [obstmp.ang_res_std for j, obstmp in \
                    enumerate(traj_status.observations) if (i != j) and not obstmp.ignore_station]

                # If all other stations are ignored, skip this procedure
                if len(ang_res_list) == 0:
                    break

                ang_res_median = np.median(ang_res_list)

                # ### DEBUG PRINT
                # print(obs.station_id, 'ang res:', np.degrees(obs.ang_res_std)*3600, \
                #     np.degrees(ang_res_median)*3600)
                
                # Check if the current observations is larger than the minimum limit, and
                # outside the median limit or larger than the maximum limit
                if (obs.ang_res_std > np.radians(self.traj_constraints.min_arcsec_err/3600)) \
                    and ((obs.ang_res_std \
                        > ang_res_median*self.traj_constraints.bad_station_obs_ang_limit) \
                    or (obs.ang_res_std > np.radians(self.traj_constraints.max_arcsec_err/3600))):

                    # Add an ignore candidate and store its angular error
                    if obs.obs_id not in ignored_station_dict:
                        ignore_candidates[i] = [obs.ang_res_std, ang_res_median]
                        any_ignored_toggle = True

                # If the station is inside the limit
                else:

                    # If the station was ignored, and now it is inside the limit, re-enable it
                    if obs.obs_id in ignored_station_dict:

                        print("Re-enabling the station: {:s}".format(obs.station_id))

                        # Re-enable station and restore the original ignore list
                        traj_status.observations[i].ignore_station = False
                        traj_status.observations[i].ignore_list \
                            = np.array(ignored_station_dict[obs.obs_id])

                        any_ignored_toggle = True

                        # Remove the station from the ignored dictionary
                        del ignored_station_dict[obs.obs_id]



            # If there are any ignored stations, rerun the solution
            if any_ignored_toggle:

                # Stop if too many observations were rejected
                if len(ignore_candidates) >= max_rejections_possible:
                    print("Too many observations ejected!")
                    skip_trajectory = True
                    break


                # If there any candidate observations to ignore
                if len(ignore_candidates):

                    # Choose the observation with the largest error
                    obs_ignore_indx = max(ignore_candidates, \
                        key=lambda x: ignore_candidates.get(x)[0])
                    obs = traj_status.observations[obs_ignore_indx]

                    ### Ignore the observation with the largest error ###

                    # Add the observation to the ignored dictionary and store the ignore list
                    ignored_station_dict[obs.obs_id] = np.array(obs.ignore_list)

                    # Ignore the observation
                    traj_status.observations[obs_ignore_indx].ignore_station = True
                    traj_status.observations[obs_ignore_indx].ignore_list \
                        = np.ones(len(obs.time_data), dtype=np.uint8)

                    ###

                    ang_res_median = ignore_candidates[obs_ignore_indx][1]
                    print("Ignoring station {:s}".format(obs.station_id))
                    print("   obs std: {:.2f} arcsec".format(3600*np.degrees(obs.ang_res_std)))
                    print("   bad lim: {:.2f} arcsec".format(3600*np.degrees(ang_res_median\
                        *self.traj_constraints.bad_station_obs_ang_limit)))
                    print("   max err: {:.2f} arcsec".format(self.traj_constraints.max_arcsec_err))



                print()
                print("Rerunning the trajectory solution...")

                # Init a new trajectory object (make sure to use the new reference Julian date)
                traj = self.initTrajectory(traj_status.jdt_ref, mc_runs)

                # Disable Monte Carlo runs until an initial stable set of observations is found
                traj.monte_carlo = False

                # Reinitialize the observations, rejecting the ignored stations
                for obs in traj_status.observations:
                    if not obs.ignore_station:
                        traj.infillWithObs(obs)

                
                # Re-run the trajectory solution
                try:
                    traj_status = traj.run()

                # If solving has failed, stop solving the trajectory
                except ValueError:
                    print("Error during trajectory estimation!")
                    return False, None


                # If the trajectory estimation failed, skip this trajectory
                if traj_status is None:
                    print("Trajectory estimation failed!")
                    skip_trajectory = True
                    break


            # If there are no ignored observations, stop trying to improve the trajectory
            else:
                break

            ### ###


        # Skip the trajectory if no good solution was found
        if skip_trajectory:

            # Add the trajectory to the list of failed trajectories
            self.dh.addTrajectory(traj, failed_jdt_ref=jdt_ref)

            return False, None

            # # If the trajectory solutions was not done at any point, skip the trajectory completely
            # if traj_best is None:
            #     return False, None

            # # Otherwise, use the best trajectory solution until the solving failed
            # else:
            #     print("Using previously estimated best trajectory...")
            #     traj_status = traj_best


        # If there are only two stations, make sure to reject solutions which have stations with 
        #   residuals higher than the maximum limit
        if len(traj_status.observations) == 2:
            if np.any([(obstmp.ang_res_std > np.radians(self.traj_constraints.max_arcsec_err/3600)) \
                for obstmp in traj_status.observations]):

                print("2 station only solution, one station has an error above the maximum limit, skipping!")

                # Add the trajectory to the list of failed trajectories
                self.dh.addTrajectory(traj_status, failed_jdt_ref=jdt_ref)

                return False, None


        # Use the best trajectory solution
        traj = traj_status


        # Only proceed if the orbit could be computed
        if traj.orbit.ra_g is not None:

            ## Compute uncertainties using Monte Carlo ##

            print("Stable set of observations found, computing uncertainties using Monte Carlo...")

            # Init a new trajectory object (make sure to use the new reference Julian date)
            traj = self.initTrajectory(traj_status.jdt_ref, mc_runs)

            # Enable Monte Carlo
            traj.monte_carlo = True

            # Reinitialize the observations, rejecting ignored stations
            for obs in traj_status.observations:
                if not obs.ignore_station:
                    traj.infillWithObs(obs)


            ### TO DO - improve the logic of choosing stations ###
            
            # If there are more than the maximum number of stations, choose the ones with the smallest residuals
            if len(traj.observations) > self.traj_constraints.max_stations:
                    
                # Sort the observations by residuals (smallest first)
                traj.observations = sorted(traj.observations, key=lambda x: x.ang_res_std)

                # Keep only the first <max_stations> stations with the smallest residuals
                traj.observations = traj.observations[:self.traj_constraints.max_stations]

            ### ###


            # Re-run the trajectory solution
            try:
                traj_status = traj.run()

            # If solving has failed, stop solving the trajectory
            except ValueError:
                print("Error during trajectory estimation!")
                return False, None


            # If the solve failed, stop
            if traj_status is None:

                # Add the trajectory to the list of failed trajectories
                self.dh.addTrajectory(traj, failed_jdt_ref=jdt_ref)

                return False, None


            traj = traj_status
            

            # Check that the average velocity is within the accepted range
            if (traj.orbit.v_avg/1000 < self.traj_constraints.v_avg_min) \
                or (traj.orbit.v_avg/1000 > self.traj_constraints.v_avg_max):

                print("Average velocity outside range: {:.1f} < {:.1f} < {:.1f} km/s, skipping...".format(self.traj_constraints.v_avg_min, \
                    traj.orbit.v_avg/1000, self.traj_constraints.v_avg_max))

                return False, None


            # If one of the observations doesn't have an estimated height, skip this trajectory
            for obs in traj.observations:
                if (obs.rbeg_ele is None) and (not obs.ignore_station):
                    print("Heights from observations failed to be estimated!")
                    return False, None


            # Check that the orbit could be computed
            if traj.orbit.ra_g is None:
                print("The orbit could not be computed!")
                return False, None

            # Set the trajectory fit as successful
            successful_traj_fit = True


            # Update trajectory file name
            traj.generateFileName()

            print()
            print("RA_g  = {:7.3f} deg".format(np.degrees(traj.orbit.ra_g)))
            print("Deg_g = {:+7.3f} deg".format(np.degrees(traj.orbit.dec_g)))
            print("V_g   = {:6.2f} km/s".format(traj.orbit.v_g/1000))
            shower_obj = associateShowerTraj(traj)
            if shower_obj is None:
                shower_code = '...'
            else:
                shower_code = shower_obj.IAU_code
            print("Shower: {:s}".format(shower_code))

        else:
            print("The orbit could not be computed!")

        ###


        return successful_traj_fit, traj



    def run(self, event_time_range=None):
        """ Run meteor corellation using available data. 

        Keyword arguments:
            event_time_range: [list] A list of two datetime objects. These are times between which
                events should be used. None by default, which uses all available events.
        """

        # Get unpaired observations, filter out observations with too little points and sort them by time
        unpaired_observations_all = self.dh.getUnpairedObservations()
        unpaired_observations_all = [mettmp for mettmp in unpaired_observations_all \
            if len(mettmp.data) >= self.traj_constraints.min_meas_pts]
        unpaired_observations_all = sorted(unpaired_observations_all, key=lambda x: x.reference_dt)

        # Remove all observations done prior to 2000, to weed out those with bad time
        unpaired_observations_all = [met_obs for met_obs in unpaired_observations_all \
            if met_obs.reference_dt > datetime.datetime(2000, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)]


        # Normalize all reference times and time data so that the reference time is at t = 0 s
        for met_obs in unpaired_observations_all:

            # Correct the reference time
            t_zero = met_obs.data[0].time_rel
            met_obs.reference_dt = met_obs.reference_dt + datetime.timedelta(seconds=t_zero)

            # Normalize all observation times so that the first time is t = 0 s
            for i in range(len(met_obs.data)):
                met_obs.data[i].time_rel -= t_zero


        
        # If the time range was given, only use the events in that time range
        if event_time_range:
            dt_beg, dt_end = event_time_range
            dt_bin_list = [event_time_range]

        # Otherwise, generate bins of datetimes for faster processing
        # Data will be divided into time bins, so the pairing function doesn't have to go pair many
        #   observations at once and keep all pairs in memory
        else:
            dt_beg = unpaired_observations_all[0].reference_dt
            dt_end = unpaired_observations_all[-1].reference_dt
            dt_bin_list = generateDatetimeBins(dt_beg, dt_end, bin_days=1, utc_hour_break=12)


        print()
        print("---------------------------------")
        print("{}".format(datetime.datetime.now().strftime('%Y-%m-%dZ%H:%M:%S')))
        print("RUNNING TRAJECTORY CORRELATION...")
        print("  TIME BEG: {:s} UTC".format(str(dt_beg)))
        print("  TIME END: {:s} UTC".format(str(dt_end)))
        print("  SPLITTING OBSERVATIONS INTO {:d} BINS".format(len(dt_bin_list)))
        print("---------------------------------")
        print()


        # Go though all time bins and split the list of observations
        for bin_beg, bin_end in dt_bin_list:


            print()
            print("-----------------------------------")
            print("  PAIRING TRAJECTORIES IN TIME BIN:")
            print("    BIN BEG: {:s} UTC".format(str(bin_beg)))
            print("    BIN END: {:s} UTC".format(str(bin_end)))
            print("-----------------------------------")
            print()


            # Select observations in the given time bin
            unpaired_observations = [met_obs for met_obs in unpaired_observations_all \
                if (met_obs.reference_dt >= bin_beg) and (met_obs.reference_dt <= bin_end)]


            # Counter for the total number of solved trajectories in this bin
            traj_solved_count = 0

            ### CHECK FOR PAIRING WITH PREVIOUSLY ESTIMATED TRAJECTORIES ###

            print()
            print("--------------------------------------------------------------------------")
            print("{}".format(datetime.datetime.now().strftime('%Y-%m-%dZ%H:%M:%S')))
            print("    1) CHECKING IF PREVIOUSLY ESTIMATED TRAJECTORIES HAVE NEW OBSERVATIONS")
            print("--------------------------------------------------------------------------")
            print()

            # Get a list of all already computed trajectories within the given time bin
            #   Reducted trajectory objects are returned
            computed_traj_list = self.dh.getComputedTrajectories(datetime2JD(bin_beg), datetime2JD(bin_end))

            # Find all unpaired observations that match already existing trajectories
            for traj_reduced in computed_traj_list:

                # If the trajectory already has more than the maximum number of stations, skip it
                if len(traj_reduced.participating_stations) >= self.traj_constraints.max_stations:
                    
                    print(
                        "Trajectory {:s} has already reached the maximum number of stations, "
                        "skipping...".format(
                            str(jd2Date(traj_reduced.jdt_ref, dt_obj=True, tzinfo=datetime.timezone.utc))
                            )
                        )
                    
                    continue

                # Get all unprocessed observations which are close in time to the reference trajectory
                traj_time_pairs = self.dh.getTrajTimePairs(traj_reduced, unpaired_observations, \
                    self.traj_constraints.max_toffset)

                # Skip trajectory if there are no new obervations
                if not traj_time_pairs:
                    continue


                print()
                print("{}".format(datetime.datetime.now().strftime('%Y-%m-%dZ%H:%M:%S')))
                print("Checking trajectory at {:s} in countries: {:s}".format( \
                    str(jd2Date(traj_reduced.jdt_ref, dt_obj=True, tzinfo=datetime.timezone.utc)), \
                    ", ".join(list(set([stat_id[:2] for stat_id in traj_reduced.participating_stations]))) \
                    ) \
                )
                print("--------")


                # Filter out bad matches and only keep the good ones
                candidate_observations = []
                traj_full = None
                skip_traj_check = False
                for met_obs in traj_time_pairs:

                    print("Candidate observation: {:s}".format(met_obs.station_code))

                    platepar = self.dh.getPlatepar(met_obs)

                    # Check that the trajectory beginning and end are within the distance limit
                    if not self.trajectoryRangeCheck(traj_reduced, platepar):
                        continue


                    # Check that the trajectory is within the field of view
                    if not self.trajectoryInFOV(traj_reduced, platepar):
                        continue


                    # Load the full trajectory object
                    if traj_full is None:
                        traj_full = self.dh.loadFullTraj(traj_reduced)

                        # If the full trajectory couldn't be loaded, skip checking this trajectory
                        if traj_full is None:
                            
                            skip_traj_check = True
                            break


                    ### Do a rough trajectory solution and perform a quick quality control ###

                    # Init observation object using the new meteor observation
                    obs_new = self.initObservationsObject(met_obs, platepar, \
                        ref_dt=jd2Date(traj_reduced.jdt_ref, dt_obj=True, tzinfo=datetime.timezone.utc))


                    # Get an observation from the trajectory object with the maximum convergence angle to
                    #   the reference observations
                    obs_traj_best = None
                    qc_max = 0.0
                    for obs_tmp in traj_full.observations:
                        
                        # Compute the plane intersection between the new and one of trajectory observations
                        pi = PlaneIntersection(obs_new, obs_tmp)

                        # Take the observation with the maximum convergence angle
                        if (obs_traj_best is None) or (pi.conv_angle > qc_max):
                            qc_max = pi.conv_angle
                            obs_traj_best = obs_tmp


                    # Do a quick trajectory solution and perform sanity checks
                    plane_intersection = self.quickTrajectorySolution(obs_traj_best, obs_new)
                    if plane_intersection is None:
                        continue

                    ### ###

                    candidate_observations.append([met_obs, obs_new])


                # Skip the candidate trajectory if it couldn't be loaded from disk
                if skip_traj_check:
                    continue


                # If there are any good new observations, add them to the trajectory and re-run the solution
                if candidate_observations:

                    print("{}".format(datetime.datetime.now().strftime('%Y-%m-%dZ%H:%M:%S')))
                    print("Recomputing trajectory with new observations from stations:")

                    # Add new observations to the trajectory object
                    for _, obs_new in candidate_observations:
                        print(obs_new.station_id)
                        traj_full.infillWithObs(obs_new)


                    # Re-run the trajectory fit
                    successful_traj_fit, traj_new = self.solveTrajectory(traj_full, traj_full.mc_runs)


                    # If the new trajectory solution succeeded, save it
                    if successful_traj_fit:

                        print("Saving the improved trajectory...")

                        # Mark the observations as paired and remove them from the processing list
                        for met_obs_temp, _ in candidate_observations:
                            self.dh.markObservationAsPaired(met_obs_temp)
                            unpaired_observations.remove(met_obs_temp)

                        # Remove the old trajectory
                        self.dh.removeTrajectory(traj_reduced)

                        # Save the new trajectory
                        self.dh.saveTrajectoryResults(traj_new, self.traj_constraints.save_plots)
                        self.dh.addTrajectory(traj_new)

                    else:
                        print("New trajectory solution failed, keeping the old trajectory...")

            ### ###


            print()
            print("-------------------------------------------------")
            print("{}".format(datetime.datetime.now().strftime('%Y-%m-%dZ%H:%M:%S')))
            print("    2) PAIRING OBSERVATIONS INTO NEW TRAJECTORIES")
            print("-------------------------------------------------")
            print()

            # List of all candidate trajectories
            candidate_trajectories = []

            # Go through all unpaired and unprocessed meteor observations
            for met_obs in unpaired_observations:

                # Skip observations that were processed in the meantime
                if met_obs.processed:
                    continue

                # Get station platepar
                reference_platepar = self.dh.getPlatepar(met_obs)
                obs1 = self.initObservationsObject(met_obs, reference_platepar)


                # Keep a list of observations which matched the reference observation
                matched_observations = []

                # Find all meteors from other stations that are close in time to this meteor
                plane_intersection_good = None
                time_pairs = self.dh.findTimePairs(met_obs, unpaired_observations, \
                    self.traj_constraints.max_toffset)
                for met_pair_candidate in time_pairs:

                    print()
                    print("Processing pair:")
                    print("{:s} and {:s}".format(met_obs.station_code, met_pair_candidate.station_code))
                    print("{:s} and {:s}".format(str(met_obs.reference_dt), \
                        str(met_pair_candidate.reference_dt)))
                    print("-----------------------")

                    ### Check if the stations are close enough and have roughly overlapping fields of view ###

                    # Get candidate station platepar
                    candidate_platepar = self.dh.getPlatepar(met_pair_candidate)

                    # Check if the stations are within range
                    if not self.stationRangeCheck(reference_platepar, candidate_platepar):
                        continue

                    # Check the FOV overlap
                    if not self.checkFOVOverlap(reference_platepar, candidate_platepar):
                        print("Station FOV does not overlap: {:s} and {:s}".format(met_obs.station_code, \
                            met_pair_candidate.station_code))
                        continue

                    ### ###



                    ### Do a rough trajectory solution and perform a quick quality control ###

                    # Init observations
                    obs2 = self.initObservationsObject(met_pair_candidate, candidate_platepar, \
                        ref_dt=met_obs.reference_dt)

                    # Do a quick trajectory solution and perform sanity checks
                    plane_intersection = self.quickTrajectorySolution(obs1, obs2)
                    if plane_intersection is None:
                        continue

                    else:
                        plane_intersection_good = plane_intersection

                    ### ###

                    matched_observations.append([obs2, met_pair_candidate, plane_intersection])



                # If there are no matched observations, skip it
                if len(matched_observations) == 0:

                    if len(time_pairs) > 0:
                        print()
                        print(" --- NO MATCH ---")

                    continue

                # Skip if there are not good plane intersections
                if plane_intersection_good is None:
                    continue

                # Add the first observation to matched observations
                matched_observations.append([obs1, met_obs, plane_intersection_good])


                # Mark observations as processed
                for _, met_obs_temp, _ in matched_observations:
                    met_obs_temp.processed = True
                    self.dh.markObservationAsProcessed(met_obs_temp)


                # Store candidate trajectories
                print()
                print(" --- ADDING CANDIDATE ---")
                candidate_trajectories.append(matched_observations)



            ### Merge all candidate trajectories which share the same observations ###
            print()
            print("---------------------------")
            print("{}".format(datetime.datetime.now().strftime('%Y-%m-%dZ%H:%M:%S')))
            print("MERGING BROKEN OBSERVATIONS")
            print("---------------------------")
            merged_candidate_trajectories = []
            merged_indices = []
            for i, traj_cand_ref in enumerate(candidate_trajectories):

                # Skip candidate trajectories that have already been merged
                if i in merged_indices:
                    continue

                
                # Stop the search if the end has been reached
                if (i + 1) == len(candidate_trajectories):
                    merged_candidate_trajectories.append(traj_cand_ref)
                    break


                # Get the mean time of the reference observation
                ref_mean_dt = traj_cand_ref[0][1].mean_dt

                obs_list_ref = [entry[1] for entry in traj_cand_ref]
                merged_candidate = []

                # Compute the mean radiant of the reference solution
                plane_radiants_ref = [entry[2].radiant_eq for entry in traj_cand_ref]
                ra_mean_ref = meanAngle([ra for ra, _ in plane_radiants_ref])
                dec_mean_ref = np.mean([dec for _, dec in plane_radiants_ref])


                # Check for pairs
                found_first_pair = False
                for j, traj_cand_test in enumerate(candidate_trajectories[(i + 1):]):

                    # Skip same observations
                    if traj_cand_ref[0] == traj_cand_test[0]:
                        continue


                    # Get the mean time of the test observation
                    test_mean_dt = traj_cand_test[0][1].mean_dt

                    # Make sure the observations that are being compared are within the time window
                    time_diff = (test_mean_dt - ref_mean_dt).total_seconds()
                    if abs(time_diff) > self.traj_constraints.max_toffset:
                        continue


                    # Break the search if the time went beyond the search. This can be done as observations 
                    #   are ordered in time
                    if time_diff > self.traj_constraints.max_toffset:
                        break



                    # Create a list of observations
                    obs_list_test = [entry[1] for entry in traj_cand_test]

                    # Check if there any any common observations between candidate trajectories and merge them
                    #   if that is the case
                    found_match = False
                    for obs1 in obs_list_ref:
                        if obs1 in obs_list_test:
                            found_match = True
                            break


                    # Compute the mean radiant of the reference solution
                    plane_radiants_test = [entry[2].radiant_eq for entry in traj_cand_test]
                    ra_mean_test = meanAngle([ra for ra, _ in plane_radiants_test])
                    dec_mean_test = np.mean([dec for _, dec in plane_radiants_test])

                    # Skip the mergning attempt if the estimated radiants are too far off
                    if np.degrees(angleBetweenSphericalCoords(dec_mean_ref, ra_mean_ref, dec_mean_test, \
                        ra_mean_test)) > self.traj_constraints.max_merge_radiant_angle:

                        continue


                    # Add the candidate trajectory to the common list if a match has been found
                    if found_match:

                        ref_stations = [obs.station_code for obs in obs_list_ref]

                        # Add observations that weren't present in the reference candidate
                        for entry in traj_cand_test:

                            # Make sure the added observation is not from a station that's already added
                            if entry[1].station_code in ref_stations:
                                continue

                            if entry[1] not in obs_list_ref:

                                # Print the reference and the merged radiants
                                if not found_first_pair:
                                    print("")
                                    print("------")
                                    print("Reference time:", ref_mean_dt)
                                    print("Reference stations: {:s}".format(", ".join(sorted(ref_stations))))
                                    print("Reference radiant: RA = {:.2f}, Dec = {:.2f}".format(np.degrees(ra_mean_ref), np.degrees(dec_mean_ref)))
                                    print("")
                                    found_first_pair = True

                                print("Merging:", entry[1].mean_dt, entry[1].station_code)
                                traj_cand_ref.append(entry)

                                print("Merged radiant:    RA = {:.2f}, Dec = {:.2f}".format(np.degrees(ra_mean_test), np.degrees(dec_mean_test)))

                                


                        # Mark that the current index has been processed
                        merged_indices.append(i + j + 1)


                # Add the reference candidate observations to the list
                merged_candidate += traj_cand_ref


                # Add the merged observation to the final list
                merged_candidate_trajectories.append(merged_candidate)



            candidate_trajectories = merged_candidate_trajectories

            ### ###




            print()
            print("-----------------------")
            print("{}".format(datetime.datetime.now().strftime('%Y-%m-%dZ%H:%M:%S')))
            print("SOLVING {:d} TRAJECTORIES".format(len(candidate_trajectories)))
            print("-----------------------")
            print()

            # Go through all candidate trajectories and compute the complete trajectory solution
            for matched_observations in candidate_trajectories:

                print()
                print("-----------------------")
                print("{}".format(datetime.datetime.now().strftime('%Y-%m-%dZ%H:%M:%S')))


                ### If there are duplicate observations from the same station, take the longer one ###

                # Find unique station counts
                station_counts = np.unique([entry[1].station_code for entry in matched_observations], \
                    return_counts=True)

                # If there are duplicates, choose the longest one
                station_duplicates = np.where(station_counts[1] > 1)[0]
                if len(station_duplicates):

                    # Go through all duplicates
                    for duplicate_station_id in station_counts[0][station_duplicates]:

                        # Find the longest observation of all from the given duplicate station
                        longest_entry = None
                        longest_count = 0
                        for entry in matched_observations:
                            _, met_obs, _ = entry

                            # Skip non-duplicate stations
                            if met_obs.station_code != duplicate_station_id:
                                continue

                            if len(met_obs.data) > longest_count:
                                longest_count = len(met_obs.data)
                                longest_entry = entry

                        # Reject all other shorter entries
                        for _ in range(station_counts[1][station_duplicates][0] - 1):
                            for entry in matched_observations:
                                _, met_obs, _ = entry

                                # Skip non-duplicate stations
                                if met_obs.station_code != duplicate_station_id:
                                    continue

                                # Remove shorter duplicate entries
                                if entry != longest_entry:
                                    print("Rejecting duplicate observation from:", met_obs.station_code)
                                    matched_observations.remove(entry)

                ###


                # Sort observations by station code
                matched_observations = sorted(matched_observations, key=lambda x: str(x[1].station_code))


                # Print info about observations which are being solved
                print()
                print("Observations:")
                for entry in matched_observations:
                    _, met_obs, _ = entry
                    print(met_obs.station_code, met_obs.mean_dt)



                # Check if the maximum convergence angle is large enough
                qc_max = np.degrees(max([entry[2].conv_angle for entry in matched_observations]))
                if qc_max < self.traj_constraints.min_qc:
                    print("Max convergence angle too small: {:.1f} < {:.1f} deg".format(qc_max, \
                        self.traj_constraints.min_qc))

                    continue


                ### Solve the trajectory ###

                print()
                print("Solving the trajectory...")

                # Decide the number of MC runs to use depending on the convergence angle
                if np.degrees(max([entry[2].conv_angle for entry in matched_observations])) \
                    < self.traj_constraints.low_qc_threshold:

                    mc_runs = self.traj_constraints.low_qc_mc_runs
                else:
                    mc_runs = self.traj_constraints.error_mc_runs


                ### ADJUST THE NUMBER OF MC RUNS FOR OPTIMAL USE OF CPU CORES ###

                # Make sure that the number of MC runs is larger or equal to the number of processor cores
                if mc_runs < self.traj_constraints.mc_cores:
                    mc_runs = int(self.traj_constraints.mc_cores)

                # If the number of MC runs is not a multiple of CPU cores, increase it until it is
                #   This will increase the number of MC runs while keeping the processing time the same
                mc_runs = int(np.ceil(mc_runs/self.traj_constraints.mc_cores)*self.traj_constraints.mc_cores)

                ### ###


                # Init the solver (use the earliest date as the reference)
                ref_dt = min([met_obs.reference_dt for _, met_obs, _ in matched_observations])
                jdt_ref = datetime2JD(ref_dt)
                traj = self.initTrajectory(jdt_ref, mc_runs)


                # Feed the observations into the trajectory solver
                for obs_temp, met_obs, _ in matched_observations:

                    # Normalize the observations to the reference Julian date
                    jdt_ref_curr = datetime2JD(met_obs.reference_dt)
                    obs_temp.time_data += (jdt_ref_curr - jdt_ref)*86400

                    traj.infillWithObs(obs_temp)


                # If this trajectory already failed to be computed, don't try to recompute it again unless
                #   new observations are added
                if self.dh.checkTrajIfFailed(traj):
                    print("The same trajectory already failed to be computed in previous runs!")
                    continue


                # Solve the trajectory
                successful_traj_fit, traj = self.solveTrajectory(traj, mc_runs)

                # Save the trajectory if successful
                if successful_traj_fit:
                    self.dh.saveTrajectoryResults(traj, self.traj_constraints.save_plots)
                    self.dh.addTrajectory(traj)

                    # Mark observations as paired in a trajectory if fit successful
                    for _, met_obs_temp, _ in matched_observations:
                        self.dh.markObservationAsPaired(met_obs_temp)


                    traj_solved_count += 1

                    # If 250 new trajectories were computed, save the DB
                    if traj_solved_count%250 == 0:
                        self.dh.saveDatabase()

                


            # Finish the correlation run (update the database with new values)
            self.dh.finish()

            print()
            print("-----------------")
            print("{}".format(datetime.datetime.now().strftime('%Y-%m-%dZ%H:%M:%S')))
            print("SOLVING RUN DONE!")
            print("-----------------")