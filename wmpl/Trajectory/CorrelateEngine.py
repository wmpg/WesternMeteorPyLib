""" Tools for automated trajectory pairing. """

from __future__ import print_function, division, absolute_import

import copy
import datetime
import json
import multiprocessing
import logging
import os
import numpy as np

from wmpl.Trajectory.Trajectory import ObservedPoints, PlaneIntersection, Trajectory, moveStateVector
from wmpl.Utils.Earth import greatCircleDistance
from wmpl.Utils.Math import vectNorm, vectMag, angleBetweenVectors, vectorFromPointDirectionAndAngle, \
    findClosestPoints, generateDatetimeBins, meanAngle, angleBetweenSphericalCoords
from wmpl.Utils.ShowerAssociation import associateShowerTraj
from wmpl.Utils.TrajConversions import J2000_JD, geo2Cartesian, cartesian2Geo, raDec2AltAz, altAz2RADec, \
    raDec2ECI, datetime2JD, jd2Date, equatorialCoordPrecession_vect
from wmpl.Utils.Pickling import loadPickle

MCMODE_NONE = 0
MCMODE_PHASE1 = 1
MCMODE_PHASE2 = 2
MCMODE_CANDS = 4
MCMODE_SIMPLE = MCMODE_CANDS + MCMODE_PHASE1
MCMODE_BOTH = MCMODE_PHASE1 + MCMODE_PHASE2
MCMODE_ALL = MCMODE_CANDS + MCMODE_PHASE1 + MCMODE_PHASE2


# Grab the logger from the main thread
log = logging.getLogger("traj_correlator")


def getMcModeStr(mcmode, strtype=0):
    modestrs = {4:'cands', 1:'simple', 2:'mcphase', 5:'candsimple', 3:'simplemc',7:'full',0:'full'}
    fullmodestrs = {4:'CANDIDATE STAGE', 1:'SIMPLE STAGE', 2:'MONTE CARLO STAGE', 7:'FULL',0:'FULL'}
    if strtype == 0:
        if mcmode in fullmodestrs.keys():
            return fullmodestrs[mcmode]
        else:
            return 'MIXED'
    else:
        if mcmode in modestrs.keys():
            return modestrs[mcmode]
        else:
            return False


def pickBestStations(obslist, max_stns):
    """
    Find the stations with the best statistics
    This is to reduce computation workload and failures in cases where
    many cameras detect the same event. 

    paramters 
    - obslist[]  - list of observations in a candidate
    - max_stns   - max number of stations to include in solution

    each observation in oblist is a tuple of 
        [ObservedPoints, MeteorObsRMS, PlaneIntersection]
    """
    # only filter if more than max_stns entries
    if len(obslist) <= max_stns:
        return obslist

    # mean error in the fit for each station - want the best fits
    # TODO actually not sure this is available here
    fit_errs=[1 for obs in obslist]

    # max magnitude seen by each station. Overbright events may be nearby but
    # may saturate the sensor so should be downweighted
    # TODO decide weighting to apply
    mags = [min(obs[0].magnitudes) for obs in obslist]
    mag_wgts = [1 for x in mags]

    # test if the meteor started and ended in the field of view
    # not sure this is needed
    # in_fovs = [obs[0].fov_beg & obs[0].fov_end for obs in obslist]

    # work out what fraction of the event each camera saw - more is better
    durations = [obs[0].JD_data[-1] - obs[0].JD_data[0] for obs in obslist]
    frac_missed = [1 - d/max(durations) for d in durations]

    # calculate the angle of incidence - nearer 90 degrees is better
    try:
        approx_state_vects = [moveStateVector(obs[2].cpa_eci, obs[2].radiant_eci, [obs[0]]) for obs in obslist]
        ws = [vectNorm(sv - obs[0].stat_eci) for sv, obs in zip(approx_state_vects, obslist)]
        cos_inc_angles = [abs(np.dot(obs[2].radiant_eci, w)) for w,obs in zip(ws, obslist)]
    except Exception:
        cos_inc_angles = [1 for obs in obslist]
        
    # distance from the station - prefer the nearest ones
    # dists = [np.linalg.norm(obs[0].stat_eci - sv)/1000 for sv, obs in zip(approx_state_vects, obslist)] 
    dists = [1 for obs in obslist]
    #
    # Cost function is the product of the values
    #  
    costs = [f*c*d*e*m for f,c,d,e,m in zip(frac_missed, cos_inc_angles, dists, fit_errs, mag_wgts)] 
    #
    #log.info(f'{dists}, {mag_wgts}, {cos_inc_angles}, {in_fovs}, {frac_missed}, {fit_errs}')
    #log.info(f'{costs}')

    # now select the best.
    # there's a tiny chance that two or more stations may have identical costs,
    # and we'll exclude them both but cost is a float so the chance is small
    threshold = sorted(costs)[max_stns]
    for i in range(len(obslist)):
        if costs[i] >= threshold:
            obslist[i][0].ignore_station = True
            # obslist[i][0].ignore_list = np.ones(len(obslist[i][0].time_data), dtype=np.uint8)
            # log.info(f'skipping {obslist[i][0].station_id}, cost {costs[i]:.3f}')
        #else:
        #    log.info(f'selecting {obslist[i][0].station_id}, cost {costs[i]:.3f}')
    return obslist


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
    def __init__(self, data_handle, traj_constraints, v_init_part, data_in_j2000=True, enableOSM=False):
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

        # enable OS style ground maps if true
        self.enableOSM = enableOSM

        self.candidatemode = None

    def trajectoryRangeCheck(self, traj_reduced, platepar):
        """ Check that the trajectory is within the range limits. 
        
        Arguments:
            traj_reduced: [TrajectoryReducted object]
            platepar: [Platepar object]

        Return:
            [bool] True if either the trajectory beg/end are within the distance limit, False otherwise.
        """

        # Compute distance between station and trajectory beginning
        beg_dist = greatCircleDistance(traj_reduced.rbeg_lat, traj_reduced.rbeg_lon, 
            np.radians(platepar.lat), np.radians(platepar.lon))


        # Compute distance between station and trajectory end
        end_dist = greatCircleDistance(traj_reduced.rend_lat, traj_reduced.rend_lon, 
            np.radians(platepar.lat), np.radians(platepar.lon))


        # Check that the trajectory beg or end is within the limits
        if (beg_dist <= self.traj_constraints.max_station_dist) \
                or (end_dist <= self.traj_constraints.max_station_dist):

            return True

        else:
            log.info("Distance between station and trajectory too large!")
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
        obs_cpa, rad_cpa, d = findClosestPoints(stat_eci, fov_eci, np.array(traj_reduced.state_vect_mini), 
            np.array(traj_reduced.radiant_eci_mini))

        # Convert to unit vectors
        obs_cpa = vectNorm(obs_cpa)
        rad_cpa = vectNorm(rad_cpa)


        # Check that the angle between the closest points of approach is within the FOV
        if angleBetweenVectors(obs_cpa, rad_cpa) <= fov/2:
            return True

        else:
            log.info("Trajectory not in station FOV!")
            return False

    

    def stationRangeCheck(self, rp, tp):
        """ Check if the two stations are within the set maximum distance. 

        Arguments:
            rp: [Platepar] Reference platepar.
            tp: [Platepar] Test platepar.
        """

        # Compute the distance between stations (km)
        dist = greatCircleDistance(np.radians(rp.lat), np.radians(rp.lon), np.radians(tp.lat), 
            np.radians(tp.lon))

        log.info("Distance between {:s} and {:s} = {:.1f} km".format(rp.station_code, tp.station_code, dist))

        if (dist < self.traj_constraints.min_station_dist) \
                or (dist > self.traj_constraints.max_station_dist):

            log.info("Rejecting station combination...")
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
            reference_fov_point = reference_stat_eci + reference_fov_eci*(height_above_ground 
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
            reference_moved = reference_stat_eci + vectorFromPointDirectionAndAngle(reference_fov_gnd, 
                reference_to_test, reference_fov/2)
            test_moved = test_stat_eci + vectorFromPointDirectionAndAngle(test_fov_gnd, test_to_reference, 
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
        obs = ObservedPoints(datetime2JD(ref_dt), ra_data, dec_data, time_data, np.radians(pp.lat), 
            np.radians(pp.lon), pp.elev, meastype=1, station_id=pp.station_code, magnitudes=mag_data, 
            ignore_list=ignore_list, fov_beg=met.fov_beg, fov_end=met.fov_end, comment=comment)

        return obs



    def projectPointToTrajectory(self, indx, obs, plane_intersection):
        """ Compute lat, lon and height of given point on the meteor trajectory. """

        meas_vector = obs.meas_eci[indx]
        jd = obs.JD_data[indx]

        # Calculate closest points of approach (observed line of sight to radiant line)
        _, rad_cpa, _ = findClosestPoints(obs.stat_eci, meas_vector, plane_intersection.cpa_eci, 
            plane_intersection.radiant_eci)

        lat, lon, elev = cartesian2Geo(jd, *rad_cpa)

        # Compute lat, lon, elev
        return rad_cpa, lat, lon, elev



    def quickTrajectorySolution(self, obs1, obs2):
        """ Perform an intersecting planes solution and check if it satisfies specified sanity checks. """

        # Do the plane intersection solution
        plane_intersection = PlaneIntersection(obs1, obs2)

        ra_cand, dec_cand = plane_intersection.radiant_eq
        log.info("Candidate radiant: RA = {:.3f}, Dec = {:+.3f}".format(np.degrees(ra_cand), np.degrees(dec_cand)))
        

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
            log.info("Begin height lower than the end height!")
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

            log.info("Meteor heights outside allowed range!")
            log.info("H1_beg: {:.2f}, H1_end: {:.2f}".format(ht1_beg, ht1_end))
            log.info("H2_beg: {:.2f}, H2_end: {:.2f}".format(ht2_beg, ht2_end))

            return None

        ### ###
        

        ### Check if the velocity is consistent ###

        # Compute the average velocity from both stations (km/s)
        vel1 = vectMag(eci1_end - eci1_beg)/(obs1.time_data[-1] - obs1.time_data[0])/1000
        vel2 = vectMag(eci2_end - eci2_beg)/(obs2.time_data[-1] - obs2.time_data[0])/1000

        # Check if they are within a certain percentage difference
        percent_diff = 100*abs(vel1 - vel2)/max(vel1, vel2)

        if percent_diff > self.traj_constraints.max_vel_percent_diff:

            log.info("Velocity difference too high: {:.2f} vs {:.2f} km/s".format(vel1/1000, vel2/1000))
            return None


        # Check the velocity range
        v_avg = (vel1 + vel2)/2
        if (v_avg < self.traj_constraints.v_avg_min) or (v_avg > self.traj_constraints.v_avg_max):
            
            log.info("Average veocity outside velocity bounds: {:.1f} < {:.1f} < {:.1f}".format(self.traj_constraints.v_avg_min, 
                v_avg, self.traj_constraints.v_avg_max))
            return None



        ### ###

        return plane_intersection



    def initTrajectory(self, jdt_ref, mc_runs, verbose=False):
        """ Initialize the Trajectory solver.
       
        Limits the number of maximum MC runs to 2*mc_runs.

        Arguments:
            jdt_ref: [datetime] Reference Julian date.
            mc_runs: [int] Number of Monte Carlo runs.

        Keyword Arguments:
            verbose: [bool] Enable or disable verbose logging

        Return:
            traj: [Trajectory]
        """

        traj = Trajectory(jdt_ref, 
            max_toffset=self.traj_constraints.max_toffset, meastype=1, 
            v_init_part=self.v_init_part, monte_carlo=self.traj_constraints.run_mc, 
            mc_runs=mc_runs, mc_runs_max=2*mc_runs,
            show_plots=False, verbose=verbose, save_results=False, 
            reject_n_sigma_outliers=2, mc_cores=self.traj_constraints.mc_cores, 
            geometric_uncert=self.traj_constraints.geometric_uncert, enable_OSM_plot=self.enableOSM)

        return traj


    def solveTrajectory(self, traj, mc_runs, mcmode=MCMODE_ALL, matched_obs=None, orig_traj=None, verbose=False):
        """ Given an initialized Trajectory object with observation, run the solver and automatically
            reject bad observations.

        Arguments:
            traj: [Trajectory object]
            mc_runs: [int] Number of Monte Carlo runs.

        Keyword Arguments:
            mcmode: [int] whether to run both intersecting-planes and monte carlo (0), just IP (1) or just MC (2)
            matched_obs [list] default none: list of new observations being used in the traj.
            orig_traj [trajectory] default none: original trajectory being updated, required to delete the old data if new soln found. 

        Return:
            successful: [bool] True if successfully solved
            traj: [Trajectory]

        """

        # Reference Julian date (the one in the traj object may change later, after timing offset estimation,
        #   so we keep this one as a "hard" reference)
        jdt_ref = traj.jdt_ref
        saved_traj_id = traj.traj_id
        log.info("")
        log.info(f"Solving the trajectory at {jd2Date(jdt_ref, dt_obj=True, tzinfo=datetime.timezone.utc).strftime('%Y-%m-%dZ%H:%M:%S.%f')}...")

        # make a note of how many observations are already marked ignored.
        initial_ignore_count = len([obs for obs in traj.observations if obs.ignore_station])
        log.info(f'initially ignoring {initial_ignore_count} stations...')
        successful_traj_fit = False

        # run the first phase of the solver if mcmode is MCMODE_PHASE1
        if mcmode & MCMODE_PHASE1: 
            # Disable Monte Carlo runs until an initial stable set of observations is found
            traj.monte_carlo = False

            # Run the solver
            try:
                traj_status = traj.run()

            # If solving has failed, stop solving the trajectory
            except ValueError as e:
                log.info("Error during trajectory estimation!")
                print(e)
                return False


            # Reject bad observations until a stable set is found, but only if there are more than 2    
            #   stations. Only one station will be rejected at one point in time
            successful_traj_fit = False
            skip_trajectory = False
            traj_best = None
            ignored_station_dict = {}
            for _ in range(len(traj.observations)):
            
                # If the trajectory estimation failed, skip this trajectory
                if traj_status is None:
                    log.info("Trajectory estimation failed!")
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
                        if not obstmp.ignore_station]) < np.median([obstmp.ang_res_std for obstmp 
                            in traj_best.observations if not obstmp.ignore_station]):

                        traj_best = copy.deepcopy(traj_status)


                # Skip this part if there are less than 3 stations
                if len(traj.observations) < 3:
                    break


                # If there are less than 2 stations that are not ignored, skip this solution
                if len([obstmp for obstmp in traj_status.observations if not obstmp.ignore_station]) < 2:
                    log.info("Skipping trajectory solution, not enough good observations...")
                    skip_trajectory = True
                    break

                log.info("")


                ### Check for bad observations and rerun the solution if necessary ###

                
                any_ignored_toggle = False
                ignore_candidates = {}

                # a) Reject all observations which have angular residuals <bad_station_obs_ang_limit>
                #   times larger than the median of all other observations
                # b) Reject all observations with higher residuals than the fixed limit
                # c) Keep all observations with error inside the minimum error limit, even though they
                #   might have been rejected in a previous iteration
                # d) Only reject a maximum of 50% of non-ignored stations
                
                max_rejections_possible = int(np.ceil(0.5*len(traj_status.observations))) + initial_ignore_count
                log.info(f'max stations allowed to be rejected is {max_rejections_possible}')
                for i, obs in enumerate(traj_status.observations):
                    if obs.ignore_station:
                        continue
                    # Compute the median angular uncertainty of all other non-ignored stations
                    ang_res_list = [obstmp.ang_res_std for j, obstmp in 
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
                            and ((obs.ang_res_std > ang_res_median*self.traj_constraints.bad_station_obs_ang_limit) 
                            or (obs.ang_res_std > np.radians(self.traj_constraints.max_arcsec_err/3600))):

                        # Add an ignore candidate and store its angular error
                        if obs.obs_id not in ignored_station_dict:
                            ignore_candidates[i] = [obs.ang_res_std, ang_res_median]
                            any_ignored_toggle = True

                    # If the station is inside the limit
                    else:

                        # If the station was ignored, and now it is inside the limit, re-enable it
                        if obs.obs_id in ignored_station_dict:

                            log.info("Re-enabling the station: {:s}".format(obs.station_id))

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
                    remaining_stations = len(traj_status.observations) - len(ignore_candidates)
                    if len(ignore_candidates) > max_rejections_possible or remaining_stations < 2:
                        log.info(f"Too many observations ejected! Only {remaining_stations} left")
                        skip_trajectory = True
                        break


                    # If there any candidate observations to ignore
                    if len(ignore_candidates):

                        # Choose the observation with the largest error
                        obs_ignore_indx = max(ignore_candidates, 
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
                        log.info("Ignoring station {:s}".format(obs.station_id))
                        log.info("   obs std: {:.2f} arcsec".format(3600*np.degrees(obs.ang_res_std)))
                        log.info("   bad lim: {:.2f} arcsec".format(3600*np.degrees(ang_res_median 
                            *self.traj_constraints.bad_station_obs_ang_limit)))
                        log.info("   max err: {:.2f} arcsec".format(self.traj_constraints.max_arcsec_err))




                    # Init a new trajectory object (make sure to use the new reference Julian date)
                    traj = self.initTrajectory(traj_status.jdt_ref, mc_runs, verbose=verbose)

                    # Disable Monte Carlo runs until an initial stable set of observations is found
                    traj.monte_carlo = False

                    # Reinitialize the observations. Note we *include* the ignored obs as they're internally marked ignored
                    # and so will be skipped, but to avoid confusion in the logs we only print the names of the non-ignored ones
                    for obs in traj_status.observations:
                        traj.infillWithObs(obs)
                        if not obs.ignore_station:
                            log.info(f'Adding {obs.station_id}')

                    log.info("")
                    active_stns = len([obs for obs in traj.observations if not obs.ignore_station])
                    if active_stns < 2:
                        log.info(f"Only {active_stns} stations left - trajectory estimation failed!")
                        skip_trajectory = True
                        break

                    log.info(f'Rerunning the trajectory solution with {active_stns} stations...')
                    # Re-run the trajectory solution
                    try:
                        traj_status = traj.run()

                    # If solving has failed, stop solving the trajectory
                    except ValueError as e:
                        log.info("Error during trajectory estimation!")
                        print(e)
                        return False


                    # If the trajectory estimation failed, skip this trajectory
                    if traj_status is None:
                        log.info("Trajectory estimation failed!")
                        skip_trajectory = True
                        break


                # If there are no ignored observations, stop trying to improve the trajectory
                else:
                    break

                ### ###


            # Skip the trajectory if no good solution was found
            if skip_trajectory:
                # Add the trajectory to the list of failed trajectories
                self.dh.addTrajectory(traj, failed_jdt_ref=jdt_ref, verbose=verbose)
                ref_dt = jd2Date(min([met_obs.jdt_ref for met_obs in traj.observations]), dt_obj=True)
                log.info(f"Trajectory at {ref_dt.isoformat()} skipped and added to fails!")

                if matched_obs:
                    for _, met_obs_temp, _ in matched_obs:
                        self.dh.observations_db.unpairObs(met_obs_temp.station_code, met_obs_temp.id, met_obs_temp.mean_dt, verbose=verbose)
                return False

            # If there are only two stations, make sure to reject solutions which have stations with 
            #   residuals higher than the maximum limit
            if len(traj_status.observations) == 2:
                if np.any([(obstmp.ang_res_std > np.radians(self.traj_constraints.max_arcsec_err/3600)) 
                        for obstmp in traj_status.observations]):

                    ref_dt = jd2Date(min([met_obs.jdt_ref for met_obs in traj.observations]), dt_obj=True)
                    log.info("2 station only solution, one station has an error above the maximum limit, skipping!")
                    log.info(f"Trajectory at {ref_dt.isoformat()} skipped and added to fails!")

                    # Add the trajectory to the list of failed trajectories
                    self.dh.addTrajectory(traj_status, failed_jdt_ref=jdt_ref, verbose=verbose)
                    for _, met_obs_temp, _ in matched_obs:
                        self.dh.observations_db.unpairObs(met_obs_temp.station_code, met_obs_temp.id, met_obs_temp.mean_dt, verbose=verbose)
                    return False


            # Use the best trajectory solution
            traj = traj_status

        # if we're only doing the simple solution, then print the results
        if mcmode == MCMODE_PHASE1:
            # Only proceed if the orbit could be computed
            if traj.orbit.ra_g is not None:
                # Update trajectory file name
                traj.generateFileName()

                log.info("")
                log.info("RA_g  = {:7.3f} deg".format(np.degrees(traj.orbit.ra_g)))
                log.info("Deg_g = {:+7.3f} deg".format(np.degrees(traj.orbit.dec_g)))
                log.info("V_g   = {:6.2f} km/s".format(traj.orbit.v_g/1000))
                shower_obj = associateShowerTraj(traj)
                if shower_obj is None:
                    shower_code = '...'
                else:
                    shower_code = shower_obj.IAU_code
                log.info("Shower: {:s}".format(shower_code))

        if mcmode & MCMODE_PHASE1:
            successful_traj_fit = True
            log.info('finished initial solution')

        ##### end of simple soln phase 
        ##### now run the Monte-carlo phase, if the mcmode is 0 (do both) or 2 (mc-only)
        if mcmode & MCMODE_PHASE2: 
            traj_status = traj

            # Only proceed if the orbit could be computed
            if traj.orbit.ra_g is not None:

                ## Compute uncertainties using Monte Carlo ##

                log.info("Stable set of observations found, computing uncertainties using Monte Carlo...")

                # Init a new trajectory object (make sure to use the new reference Julian date)
                traj = self.initTrajectory(traj_status.jdt_ref, mc_runs, verbose=verbose)

                # Enable Monte Carlo
                traj.monte_carlo = True

                # Get all non-ignored observations
                non_ignored_observations = [obs for obs in traj_status.observations if not obs.ignore_station]

                # If there are more than the maximum number of stations, choose the ones with the smallest
                # residuals
                # Don't do this in mc-only mode since phase1 has already selected the stations and we could 
                # create duplicate orbits if we now exclude some stations from the solution
                # TODO should we do this here *at all* ? 
                if len(non_ignored_observations) > self.traj_constraints.max_stations and mcmode != MCMODE_PHASE2:

                    # Sort the observations by residuals (smallest first)
                    # TODO: implement better sorting algorithm
                    log.info('Selecting best {} stations'.format(self.traj_constraints.max_stations))
                    obs_sorted = sorted(non_ignored_observations, key=lambda x: x.ang_res_std)

                    # Keep only the first <max_stations> stations with the smallest residuals
                    obs_selected = obs_sorted[:self.traj_constraints.max_stations]

                    log.info("More than {:d} stations, keeping only the best ones...".format(self.traj_constraints.max_stations))
                    log.info("    Selected stations: {:s}".format(', '.join([obs.station_id for obs in obs_selected])))

                else:
                    obs_selected = non_ignored_observations

                ### ###


                # Reinitialize the observations, rejecting ignored stations
                for obs in obs_selected:
                    if not obs.ignore_station:
                        traj.infillWithObs(obs)


                # Re-run the trajectory solution
                try:
                    traj_status = traj.run()

                # If solving has failed, stop solving the trajectory
                except ValueError as e:
                    log.info("Error during trajectory estimation!")
                    print(e)
                    return False


                # If the solve failed, stop
                if traj_status is None:

                    # Add the trajectory to the list of failed trajectories
                    if mcmode != MCMODE_PHASE2:
                        self.dh.addTrajectory(traj, failed_jdt_ref=jdt_ref, verbose=verbose)
                    log.info(f"Trajectory at {ref_dt.isoformat()} skipped and added to fails!")
                    return False


                traj = traj_status
                

                # Check that the average velocity is within the accepted range
                if (traj.orbit.v_avg/1000 < self.traj_constraints.v_avg_min) or (traj.orbit.v_avg/1000 > self.traj_constraints.v_avg_max):

                    log.info("Average velocity outside range: {:.1f} < {:.1f} < {:.1f} km/s, skipping...".format(self.traj_constraints.v_avg_min, 
                        traj.orbit.v_avg/1000, self.traj_constraints.v_avg_max))

                    return False


                # If one of the observations doesn't have an estimated height, skip this trajectory
                for obs in traj.observations:
                    if (obs.rbeg_ele is None) and (not obs.ignore_station):
                        log.info("Heights from observations failed to be estimated!")
                        return False


                # Check that the orbit could be computed
                if traj.orbit.ra_g is None:
                    log.info("The orbit could not be computed!")
                    return False

                # Set the trajectory fit as successful
                successful_traj_fit = True
                log.info('Monte-carlo phase complete ')

                ### end of the MC phase

                # Update trajectory file name
                traj.generateFileName()

                log.info("")
                log.info("RA_g  = {:7.3f} deg".format(np.degrees(traj.orbit.ra_g)))
                log.info("Deg_g = {:+7.3f} deg".format(np.degrees(traj.orbit.dec_g)))
                log.info("V_g   = {:6.2f} km/s".format(traj.orbit.v_g/1000))
                shower_obj = associateShowerTraj(traj)
                if shower_obj is None:
                    shower_code = '...'
                else:
                    shower_code = shower_obj.IAU_code
                log.info("Shower: {:s}".format(shower_code))

            else:
                log.info("The orbit could not be computed!")
                return False



        # Save the trajectory if successful. 
        if successful_traj_fit:
            # restore the original traj_id so that the phase1 and phase 2 results use the same ID
            if mcmode == MCMODE_PHASE2:
                traj.traj_id = saved_traj_id
                traj.phase_1_only = False

            if mcmode == MCMODE_PHASE1:
                traj.phase_1_only = True

            if orig_traj:
                log.info(f"Removing the previous solution {os.path.dirname(orig_traj.traj_file_path)} ...")
                remove_phase1 = True if abs(round((traj.jdt_ref-orig_traj.jdt_ref)*86400000,0)) > 0 else False
                self.dh.removeTrajectory(orig_traj, remove_phase1)
                traj.pre_mc_longname = os.path.split(self.dh.generateTrajOutputDirectoryPath(orig_traj, make_dirs=False))[-1] 

            log.info('Saving trajectory....')

            self.dh.saveTrajectoryResults(traj, self.traj_constraints.save_plots)
            if mcmode != MCMODE_PHASE2:
                # we do not need to update the database for phase2 
                log.info('Updating database....')
                self.dh.addTrajectory(traj)

        else:
            log.info('unable to fit trajectory')

        return successful_traj_fit


    def run(self, event_time_range=None, bin_time_range=None, mcmode=MCMODE_ALL, verbose=False):
        """ Run meteor corellation using available data. 

        Keyword arguments:
            event_time_range: [list] A list of two datetime objects. These are times between which
                events should be used. None by default, which uses all available events.
            mcmode: [int] flag to indicate whether or not to run monte-carlos
        """

        # a bit of logging to let readers know what we're doing
        mcmodestr = getMcModeStr(mcmode, strtype=1)

        if mcmode != MCMODE_PHASE2:
            if mcmode & MCMODE_CANDS:
                # Get unpaired observations, filter out observations with too little points and sort them by time
                unpaired_observations_all = self.dh.getUnpairedObservations()
                unpaired_observations_all = [mettmp for mettmp in unpaired_observations_all 
                    if len(mettmp.data) >= self.traj_constraints.min_meas_pts]
                unpaired_observations_all = sorted(unpaired_observations_all, key=lambda x: x.reference_dt)

                # Remove all observations done prior to 2000, to weed out those with bad time
                unpaired_observations_all = [met_obs for met_obs in unpaired_observations_all 
                    if met_obs.reference_dt > datetime.datetime(2000, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)]

                # Normalize all reference times and time data so that the reference time is at t = 0 s
                for met_obs in unpaired_observations_all:

                    # Correct the reference time
                    t_zero = met_obs.data[0].time_rel
                    met_obs.reference_dt = met_obs.reference_dt + datetime.timedelta(seconds=t_zero)

                    # Normalize all observation times so that the first time is t = 0 s
                    for i in range(len(met_obs.data)):
                        met_obs.data[i].time_rel -= t_zero
            else: 
                event_time_range = self.dh.dt_range
            
            # If the time range was given, only use the events in that time range
            if event_time_range:
                dt_beg, dt_end = event_time_range
                dt_bin_list = [event_time_range]

            # Otherwise, generate bins of datetimes for faster processing
            # Data will be divided into time bins, so the pairing function doesn't have to go pair many
            #   observations at once and keep all pairs in memory
            else:
                if mcmode & MCMODE_CANDS:
                    dt_beg = unpaired_observations_all[0].reference_dt
                    dt_end = unpaired_observations_all[-1].reference_dt
                else: 
                    dt_beg, dt_end = self.dh.dt_range
                dt_bin_list = generateDatetimeBins(
                    dt_beg, dt_end, 
                    bin_days=1, utc_hour_break=12, tzinfo=datetime.timezone.utc, reverse=False
                )
                
        else:
            dt_beg = self.dh.dt_range[0]
            dt_end = self.dh.dt_range[1]
            dt_bin_list = [(dt_beg, dt_end)]

        log.info("")
        log.info("---------------------------------")
        log.info("{}".format(datetime.datetime.now().strftime('%Y-%m-%dZ%H:%M:%S')))
        log.info("RUNNING TRAJECTORY CORRELATION...")
        log.info("  TIME BEG: {:s} UTC".format(str(dt_beg)))
        log.info("  TIME END: {:s} UTC".format(str(dt_end)))
        log.info("  SPLITTING OBSERVATIONS INTO {:d} BINS".format(len(dt_bin_list)))
        log.info("---------------------------------")
        log.info("")

        log.info(f'mcmode is {mcmodestr}')

        # Go though all time bins and split the list of observations
        for bin_beg, bin_end in dt_bin_list:
            # Counter for the total number of solved trajectories in this bin
            traj_solved_count = 0

            # if we're in MC mode 0 or 1 we have to find the candidate trajectories
            if mcmode != MCMODE_PHASE2:
                ## we are in candidatemode mode 0 or 1 and want to find candidates 
                if mcmode & MCMODE_CANDS:
                    log.info("")
                    log.info("-----------------------------------")
                    log.info("  PAIRING TRAJECTORIES IN TIME BIN:")
                    log.info("    BIN BEG: {:s} UTC".format(str(bin_beg)))
                    log.info("    BIN END: {:s} UTC".format(str(bin_end)))
                    log.info("-----------------------------------")
                    log.info("")


                    # Select observations in the given time bin
                    unpaired_observations = [met_obs for met_obs in unpaired_observations_all 
                        if (met_obs.reference_dt >= bin_beg) and (met_obs.reference_dt <= bin_end)]

                    total_unpaired = len(unpaired_observations)
                    remaining_unpaired = total_unpaired
                    log.info(f'Analysing {total_unpaired} observations in this bucket...')

                    ### CHECK FOR PAIRING WITH PREVIOUSLY ESTIMATED TRAJECTORIES ###

                    log.info("")
                    log.info("--------------------------------------------------------------------------")
                    log.info("    1) CHECKING IF PREVIOUSLY ESTIMATED TRAJECTORIES HAVE NEW OBSERVATIONS")
                    log.info("--------------------------------------------------------------------------")
                    log.info("")

                    # Get a list of all already computed trajectories within the given time bin
                    #   Reducted trajectory objects are returned
                    
                    if bin_time_range:
                        # restrict checks to the bin range supplied to run() plus a day to allow for data upload times
                        log.info(f'Getting computed trajectories for bin {str(bin_time_range[0])} to {str(bin_time_range[1])}')
                        computed_traj_list = self.dh.getComputedTrajectories(datetime2JD(bin_time_range[0]), datetime2JD(bin_time_range[1])+1)
                    else:
                        # use the current bin. 
                        log.info(f'Getting computed trajectories for {str(bin_beg)} to {str(bin_end)}')
                        computed_traj_list = self.dh.getComputedTrajectories(datetime2JD(bin_beg), datetime2JD(bin_end))

                    # Find all unpaired observations that match already existing trajectories
                    for traj_reduced in computed_traj_list:

                        # If the trajectory already has more than the maximum number of stations, skip it
                        if len(traj_reduced.participating_stations) >= self.traj_constraints.max_stations:

                            log.info(
                                "Trajectory {:s} has already reached the maximum number of stations, "
                                "skipping...".format(
                                    str(jd2Date(traj_reduced.jdt_ref, dt_obj=True, tzinfo=datetime.timezone.utc))))

                            # TODO DECIDE WHETHER WE ACTUALLY WANT TO DO THIS
                            # the problem is that we could end up with unpaired observations that form a new trajectory instead of
                            # being added to an existing one
                            continue
                    
                        # Get all unprocessed observations which are close in time to the reference trajectory
                        traj_time_pairs = self.dh.getTrajTimePairs(traj_reduced, unpaired_observations, 
                            self.traj_constraints.max_toffset)

                        # Skip trajectory if there are no new obervations
                        if not traj_time_pairs:
                            continue


                        log.info("")
                        log.info("Checking trajectory at {:s} in countries: {:s}".format( 
                            str(jd2Date(traj_reduced.jdt_ref, dt_obj=True, tzinfo=datetime.timezone.utc)), 
                            ", ".join(list(set([stat_id[:2] for stat_id in traj_reduced.participating_stations])))))
                        log.info("--------")


                        # Filter out bad matches and only keep the good ones
                        candidate_observations = []
                        traj_full = None
                        skip_traj_check = False
                        for met_obs in traj_time_pairs:

                            log.info("Candidate observation: {:s}".format(met_obs.station_code))

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
                            obs_new = self.initObservationsObject(met_obs, platepar, 
                                ref_dt=jd2Date(traj_reduced.jdt_ref, dt_obj=True, tzinfo=datetime.timezone.utc))
                            obs_new.id = met_obs.id
                            obs_new.station_code = met_obs.station_code
                            obs_new.mean_dt = met_obs.mean_dt

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

                            candidate_observations.append([obs_new, met_obs])


                        # Skip the candidate trajectory if it couldn't be loaded from disk
                        if skip_traj_check:
                            continue


                        # If there are any good new observations, add them to the trajectory and re-run the solution
                        if candidate_observations:

                            log.info("Recomputing trajectory with new observations from stations:")

                            # Add new observations to the trajectory object
                            for obs_new, _ in candidate_observations:
                                log.info(obs_new.station_id)
                                traj_full.infillWithObs(obs_new)


                            # Re-run the trajectory fit
                            # pass in orig_traj here so that it can be deleted from disk if the new solution succeeds
                            successful_traj_fit = self.solveTrajectory(traj_full, traj_full.mc_runs, mcmode=mcmode, orig_traj=traj_reduced, verbose=verbose)
                            
                            # If the new trajectory solution succeeded, remove the now-paired observations
                            if successful_traj_fit:

                                log.info("Remove paired observations from the processing list...")
                                for _, met_obs_temp in candidate_observations:
                                    unpaired_observations.remove(met_obs_temp)
                                    if self.dh.observations_db.addPairedObs(met_obs_temp.station_code, met_obs_temp.id, met_obs_temp.mean_dt):
                                        remaining_unpaired -= 1


                            else:
                                for met_obs_temp, _ in candidate_observations:
                                    self.dh.observations_db.unpairObs(met_obs_temp.station_code, met_obs_temp.id, met_obs_temp.mean_dt, verbose=verbose)
                                log.info("New trajectory solution failed, keeping the old trajectory...")

                    ### ###


                    log.info("")
                    log.info("-------------------------------------------------")
                    log.info("    2) PAIRING OBSERVATIONS INTO NEW TRAJECTORIES")
                    log.info("-------------------------------------------------")
                    log.info("")

                    # List of all candidate trajectories
                    candidate_trajectories = []

                    # Go through all unpaired and unprocessed meteor observations
                    for met_obs in unpaired_observations:

                        # Skip observations that were processed in the meantime
                        if met_obs.processed:
                            continue

                        if self.dh.observations_db.checkObsPaired(met_obs.station_code, met_obs.id, verbose=verbose):
                            continue

                        # Get station platepar
                        reference_platepar = self.dh.getPlatepar(met_obs)
                        obs1 = self.initObservationsObject(met_obs, reference_platepar)


                        # Keep a list of observations which matched the reference observation
                        matched_observations = []

                        # Find all meteors from other stations that are close in time to this meteor
                        plane_intersection_good = None
                        time_pairs = self.dh.findTimePairs(met_obs, unpaired_observations, 
                            self.traj_constraints.max_toffset)
                        for met_pair_candidate in time_pairs:

                            log.info("")
                            log.info("Processing pair:")
                            log.info("{:s} and {:s}".format(met_obs.station_code, met_pair_candidate.station_code))
                            log.info("{:s} and {:s}".format(str(met_obs.reference_dt), str(met_pair_candidate.reference_dt)))
                            log.info("-----------------------")

                            ### Check if the stations are close enough and have roughly overlapping fields of view ###

                            # Get candidate station platepar
                            candidate_platepar = self.dh.getPlatepar(met_pair_candidate)

                            # Check if the stations are within range
                            if not self.stationRangeCheck(reference_platepar, candidate_platepar):
                                continue

                            # Check the FOV overlap
                            if not self.checkFOVOverlap(reference_platepar, candidate_platepar):
                                log.info("Station FOV does not overlap: {:s} and {:s}".format(met_obs.station_code, 
                                    met_pair_candidate.station_code))
                                continue

                            ### ###



                            ### Do a rough trajectory solution and perform a quick quality control ###

                            # Init observations
                            obs2 = self.initObservationsObject(met_pair_candidate, candidate_platepar, 
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
                                log.info("")
                                log.info(" --- NO MATCH ---")

                            continue

                        # Skip if there are not good plane intersections
                        if plane_intersection_good is None:
                            continue

                        # Add the first observation to matched observations
                        matched_observations.append([obs1, met_obs, plane_intersection_good])


                        # Mark observations as processed
                        for _, met_obs_temp, _ in matched_observations:
                            met_obs_temp.processed = True
                            if self.dh.observations_db.addPairedObs(met_obs_temp.station_code, met_obs_temp.id, met_obs_temp.mean_dt, verbose=verbose):
                                remaining_unpaired -= 1

                        # Store candidate trajectory group
                        # Note that this will include candidate groups that already failed on previous runs. 
                        # We will exclude these later - we can't do it just yet as if new data has arrived, then 
                        # in the next step, the group might be merged with another group creating a solvable set. 
                        log.info("")
                        ref_dt = min([met_obs.reference_dt for _, met_obs, _ in matched_observations])
                        log.info(f" --- ADDING CANDIDATE at {ref_dt.isoformat()} ---")
                        candidate_trajectories.append(matched_observations)

                    ### Merge all candidate trajectories which share the same observations ###
                    log.info("")
                    log.info("---------------------------")
                    log.info("MERGING BROKEN OBSERVATIONS")
                    log.info("---------------------------")
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
                            test_ids = [x.id for x in obs_list_test]
                            for obs1 in obs_list_ref:
                                if obs1.id in test_ids:
                                    found_match = True
                                    break


                            # Compute the mean radiant of the reference solution
                            plane_radiants_test = [entry[2].radiant_eq for entry in traj_cand_test]
                            ra_mean_test = meanAngle([ra for ra, _ in plane_radiants_test])
                            dec_mean_test = np.mean([dec for _, dec in plane_radiants_test])

                            # Skip the merging attempt if the estimated radiants are too far off
                            if np.degrees(angleBetweenSphericalCoords(dec_mean_ref, ra_mean_ref, dec_mean_test, ra_mean_test)) > self.traj_constraints.max_merge_radiant_angle:
                                continue


                            # Add the candidate trajectory to the common list if a match has been found
                            if found_match:

                                ref_stations = [obs.station_code for obs in obs_list_ref]

                                # Add observations that weren't present in the reference candidate
                                for entry in traj_cand_test:

                                    # Make sure the added observation is not already added
                                    if entry[1] not in obs_list_ref:

                                        # Print the reference and the merged radiants
                                        if not found_first_pair:
                                            log.info("")
                                            log.info("------")
                                            log.info("Reference time: {:s}".format(str(ref_mean_dt)))
                                            log.info("Reference stations: {:s}".format(", ".join(sorted(ref_stations))))
                                            log.info("Reference radiant: RA = {:.2f}, Dec = {:.2f}".format(np.degrees(ra_mean_ref), np.degrees(dec_mean_ref)))
                                            log.info("")
                                            found_first_pair = True

                                        log.info("Merging: {:s} {:s}".format(str(entry[1].mean_dt), str(entry[1].station_code)))
                                        traj_cand_ref.append(entry)

                                        log.info("Merged radiant:    RA = {:.2f}, Dec = {:.2f}".format(np.degrees(ra_mean_test), np.degrees(dec_mean_test)))
                                        log.info(f'Candidate contains {len(traj_cand_ref)} obs')

                                        


                                # Mark that the current index has been processed
                                merged_indices.append(i + j + 1)


                        # Add the reference candidate observations to the list
                        merged_candidate += traj_cand_ref


                        # Add the merged observation to the final list
                        merged_candidate_trajectories.append(merged_candidate)

                    log.info("-----------------------")
                    log.info('CHECKING FOR ALREADY-FAILED CANDIDATES')
                    log.info("-----------------------")

                    # okay now we can remove any already-failed combinations. This wasn't safe to do earlier
                    # because we first needed to see if we could merge any groups. 
                    candidate_trajectories, remaining_unpaired = self.dh.excludeAlreadyFailedCandidates(merged_candidate_trajectories, remaining_unpaired)

                    log.info("-----------------------")
                    log.info(f'There are {remaining_unpaired} remaining unpaired observations in this bucket.')
                    log.info("-----------------------")

                    # in candidate mode we want to save the candidates to disk
                    if mcmode == MCMODE_CANDS: 
                        log.info("-----------------------")
                        log.info('SAVING {} CANDIDATES'.format(len(candidate_trajectories)))
                        log.info("-----------------------")

                        self.dh.saveCandidates(candidate_trajectories, verbose=verbose)
                        return len(candidate_trajectories)
                    else:
                        log.info("-----------------------")
                        log.info('PROCESSING {} CANDIDATES'.format(len(candidate_trajectories)))
                        log.info("-----------------------")

                # end of 'if mcmode & MCMODE_CANDS'
                ### ###
                else:
                    # candidatemode is LOAD so load any available candidates for processing
                    traj_solved_count = 0
                    candidate_trajectories = []
                    log.info("-----------------------")
                    log.info('LOADING CANDIDATES')
                    log.info("-----------------------")

                    save_path = self.dh.candidate_dir
                    for fil in os.listdir(save_path):
                        if '.pickle' not in fil: 
                            continue
                        try:
                            loadedpickle = loadPickle(save_path, fil)
                            candidate_trajectories.append(loadedpickle)
                            # move the loaded file so we don't try to reprocess it on a subsequent pass
                            procpath = os.path.join(save_path, 'processed')  
                            os.makedirs(procpath, exist_ok=True)
                            procfile = os.path.join(procpath, fil)
                            if os.path.isfile(procfile):
                                os.remove(procfile)
                            os.rename(os.path.join(save_path, fil), procfile)
                        except Exception: 
                            print(f'Candidate {fil} went away, probably picked up by another process')
                    log.info("-----------------------")
                    log.info('LOADED {} CANDIDATES'.format(len(candidate_trajectories)))
                    log.info("-----------------------")
                # end of 'self.candidatemode == CANDMODE_LOAD'
            # end of 'if mcmode != MCMODE_PHASE2' 
            else: 
                # mcmode == MCMODE_PHASE2 so we need to load the phase1 solutions
                log.info("-----------------------")
                log.info('LOADING PHASE1 SOLUTIONS')
                log.info("-----------------------")
                candidate_trajectories = self.dh.phase1Trajectories
            # end of "if mcmode == MCMODE_PHASE2"

            num_traj = len(candidate_trajectories)
            log.info("")
            log.info("-----------------------")
            log.info(f'SOLVING {num_traj} TRAJECTORIES {mcmodestr}')
            log.info("-----------------------")
            log.info("")

            # Go through all candidate trajectories and compute the complete trajectory solution
            for i, matched_observations in enumerate(candidate_trajectories):

                log.info("")
                log.info("-----------------------")
                log.info(f'processing {"candidate" if mcmode==MCMODE_PHASE1 else "trajectory"} {i+1}/{num_traj}')


                # if mcmode is not 2, prepare to calculate the intersecting planes solutions
                if mcmode != MCMODE_PHASE2:
                    # Find unique station counts
                    station_counts = np.unique([entry[1].station_code for entry in matched_observations], 
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
                                        log.info("Rejecting duplicate observation from: {:s}".format(str(met_obs.station_code)))
                                        matched_observations.remove(entry)

                    ###


                    # Sort observations by station code
                    matched_observations = sorted(matched_observations, key=lambda x: str(x[1].station_code))

                    # TODO: work out better algorithm here
                    if len(matched_observations) > self.traj_constraints.max_stations:
                        log.info('Selecting best {} stations'.format(self.traj_constraints.max_stations))

                        # pickBestStations selects the best and marks the others "ignored". This keeps
                        # them in the dataset without using them in the solver. Otherwise if they're not markeed as used
                        # on the next pass through the solver they wil be picked up as a different trajectory.
                        matched_observations = pickBestStations(matched_observations, self.traj_constraints.max_stations)

                    # Print info about observations which are being solved
                    log.info("")
                    log.info("Observations:")
                    for entry in matched_observations:
                        obs, met_obs, _ = entry
                        log.info(f'{met_obs.station_code} - {met_obs.mean_dt} - {obs.ignore_station}')



                    # Check if the maximum convergence angle is large enough
                    qc_max = np.degrees(max([entry[2].conv_angle for entry in matched_observations]))
                    if qc_max < self.traj_constraints.min_qc:
                        log.info("Max convergence angle too small: {:.1f} < {:.1f} deg".format(qc_max, 
                            self.traj_constraints.min_qc))

                        # create a traj object to add to the failed database so we don't try to recompute this one again
                        ref_dt = min([met_obs.reference_dt for _, met_obs, _ in matched_observations])
                        jdt_ref = datetime2JD(ref_dt)

                        failed_traj = self.initTrajectory(jdt_ref, 0, verbose=verbose)
                        for obs_temp, met_obs, _ in matched_observations:
                            failed_traj.infillWithObs(obs_temp)

                        t0 = min([obs.time_data[0] for obs in failed_traj.observations if (not obs.ignore_station) 
                            or (not np.all(obs.ignore_list))])
                        if t0 != 0.0:
                            failed_traj.jdt_ref = failed_traj.jdt_ref + t0/86400.0

                        self.dh.addTrajectory(failed_traj, failed_traj.jdt_ref, verbose=verbose)

                        for _, met_obs_temp, _ in matched_observations:
                            self.dh.observations_db.unpairObs(met_obs_temp.station_code, met_obs_temp.id, met_obs_temp.mean_dt, verbose=verbose)
                        log.info(f"Trajectory at {ref_dt.isoformat()} skipped and added to fails!")
                        continue


                    ### Solve the trajectory ###

                    # Decide the number of MC runs to use depending on the convergence angle
                    if np.degrees(max([entry[2].conv_angle for entry in matched_observations])) < self.traj_constraints.low_qc_threshold:

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
                    traj = self.initTrajectory(jdt_ref, mc_runs, verbose=verbose)


                    # Feed the observations into the trajectory solver
                    for obs_temp, met_obs, _ in matched_observations:

                        # Normalize the observations to the reference Julian date
                        jdt_ref_curr = datetime2JD(met_obs.reference_dt)
                        obs_temp.time_data += (jdt_ref_curr - jdt_ref)*86400

                        traj.infillWithObs(obs_temp)

                    ### Recompute the reference JD and all times so that the first time starts at 0 ###

                    # Determine the first relative time from reference JD
                    t0 = min([obs.time_data[0] for obs in traj.observations if (not obs.ignore_station) 
                        or (not np.all(obs.ignore_list))])

                    # If the first time is not 0, normalize times so that the earliest time is 0
                    if t0 != 0.0:

                        # Offset all times by t0
                        for i in range(len(traj.observations)):
                            traj.observations[i].time_data -= t0


                        # Recompute the reference JD to corresponds with t0
                        traj.jdt_ref = traj.jdt_ref + t0/86400.0


                    # If this trajectory already failed to be computed, don't try to recompute it again unless
                    #   new observations are added
                    if self.dh.checkTrajIfFailed(traj):
                        log.info("The same trajectory already failed to be computed in previous runs!")
                        for _, met_obs_temp, _ in matched_observations:
                            self.dh.observations_db.unpairObs(met_obs_temp.station_code, met_obs_temp.id, met_obs_temp.mean_dt, verbose=verbose)
                        continue

                    # pass in matched_observations here so that solveTrajectory can mark them paired if they're used
                    result = self.solveTrajectory(traj, mc_runs, mcmode=mcmode, matched_obs=matched_observations, verbose=verbose)
                    traj_solved_count += int(result)

                    # end of if mcmode != MCMODE_PHASE2
                else:
                    # mcmode is MCMODE_PHASE2 and so we have a list of trajectories that were solved in phase 1
                    # to prepare for monte-carlo solutions

                    traj = matched_observations
                    log.info("")
                    log.info(f"Solving the trajectory {traj.traj_id}...")

                    # Decide the number of MC runs to use depending on the convergence angle
                    if np.degrees(max([entry.conv_angle for entry in matched_observations.intersection_list])) < self.traj_constraints.low_qc_threshold:

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

                    # pass in matched_observations here so that solveTrajectory can mark them unpaired if the solver fails
                    result = self.solveTrajectory(traj, mc_runs, mcmode=mcmode, matched_obs=matched_observations, orig_traj=traj, verbose=verbose)
                    traj_solved_count += int(result)

            # end of "for matched_observations in candidate_trajectories"
            outcomes = [traj_solved_count]

            log.info(f'SOLVED {sum(outcomes)} TRAJECTORIES')

            log.info("")
            log.info("-----------------")
            log.info("SOLVING RUN DONE!")
            log.info("-----------------")

            return sum(outcomes)
