""" PyLIG trajectory solver

Estimates meteor trajectory from given observed points. 

"""

from __future__ import print_function, division, absolute_import

import time
import copy
import sys
import os
import datetime
import pickle
import json
from operator import attrgetter

import numpy as np
import scipy.optimize
import scipy.interpolate
import scipy.stats

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from wmpl.Utils.OSTools import importBasemap
Basemap = importBasemap()


from wmpl.Trajectory.Orbit import calcOrbit
from wmpl.Utils.Math import vectNorm, vectMag, meanAngle, findClosestPoints, RMSD, \
    angleBetweenSphericalCoords, angleBetweenVectors, lineFunc, normalizeAngleWrap, confidenceInterval
from wmpl.Utils.Misc import valueFormat
from wmpl.Utils.OSTools import mkdirP
from wmpl.Utils.Pickling import savePickle
from wmpl.Utils.Plotting import savePlot
from wmpl.Utils.PlotOrbits import plotOrbits
from wmpl.Utils.PlotCelestial import CelestialPlot
from wmpl.Utils.PlotMap import GroundMap
from wmpl.Utils.TrajConversions import EARTH, G, ecef2ENU, enu2ECEF, geo2Cartesian, geo2Cartesian_vect, \
    cartesian2Geo, altAz2RADec_vect, raDec2AltAz, raDec2AltAz_vect, raDec2ECI, eci2RaDec, jd2Date, datetime2JD
from wmpl.Utils.PyDomainParallelizer import parallelComputeGenerator


# Text size of image legends
LEGEND_TEXT_SIZE = 6


class ObservedPoints(object):
    def __init__(self, jdt_ref, meas1, meas2, time_data, lat, lon, ele, meastype, station_id=None, \
        excluded_time=None, ignore_list=None, ignore_station=False, magnitudes=None, fov_beg=None, \
        fov_end=None, obs_id=None, comment=""):
        """ Structure for containing data of observations from invidiual stations.
        
        Arguments:
            jdt_ref: [float] reference Julian date for the measurements. Add provided times should be given
                relative to this number. This is user selectable and can be the time of the first camera, or 
                the first measurement, or some average time for the meteor, but should be close to the time of 
                the meteor. This same reference date/time will be used on all camera measurements for the 
                purposes of computing local sidereal time and making geocentric coordinate transformations, 
                thus it is good that this time corresponds to the beginning of the meteor.
            meas1: [list or ndarray] First measurement array (azimuth or R.A., depending on meastype, see 
                meastype documentation for more information). Measurements should be given in radians.
            meas2: [list or ndarray] Second measurement array (altitude, zenith angle or declination, 
                depending on meastype, see meastype documentation for more information), in radians.
            time_data: [list or ndarray] Time in seconds from the reference Julian date.
            lat: [float] Latitude +N of station in radians.
            lon: [float] Longitude +E of station in radians.
            ele: [float] Elevation of station in meters.
            meastype: [float] Flag indicating the type of angle measurements the user is providing for meas1 
                and meas2 below. The following are all in radians:
                        1 = Right Ascension for meas1, Declination for meas2.
                        2 = Azimuth +east of due north for meas1, Elevation angle
                            above the horizon for meas2
                        3 = Azimuth +west of due south for meas1, Zenith angle for meas2
                        4 = Azimuth +north of due east for meas1, Zenith angle for meas2

        Keyword arguments:
            station_id: [str] Identification of the station. None by default, in which case a number will be
                assigned to the station by the program.
            excluded_time: [list] [excluded_time_min, excluded_time_max] A range of minimum and maximum 
                observation time which should be excluded from the optimization because the measurements are 
                missing in that portion of the time.
            ignore_list: [list or ndarray] A list of 0s and 1s which should be of the equal length as 
                the input data. If a particular data point is to be ignored, number 1 should be put,
                otherwise (if the point should be used) 0 should be used. E.g. the this should could look
                like this: [0, 0, 0, 1, 1, 0, 0], which would mean that the fourth and the fifth points
                will be ignored in trajectory estimation.
            ignore_station: [bool] If True, all data from the given station will not be taken into 
                consideration upon trajectory fitting, but they will still be shown on the graphs.
            magnitudes: [list] A list of apparent magnitudes of the meteor. None by default.
            fov_beg: [bool] True if the meteor began inside the FOV, False otherwise. None by default.
            fov_end: [bool] True if the meteor ended inside the FOV, False otherwise. None by default.
            obs_id: [int] Unique ID of the observation. This is to differentiate different observations from
                the same station.
            comment: [str] A comment about the observations. May be used to store RMS FF file number on which
                the meteor was observed.
        """

        ### INPUT DATA ###
        ######################################################################################################

        self.meas1 = meas1
        self.meas2 = meas2

        # reference Julian date
        self.jdt_ref = jdt_ref

        self.time_data = time_data



        self.ignore_station = ignore_station

        # Set all points to be ignored if the station is ignored
        if self.ignore_station:
            ignore_list = np.ones(len(time_data), dtype=np.uint8)


        # Init the ignore list
        if ignore_list is None:
            self.ignore_list = np.zeros(len(time_data), dtype=np.uint8)

        else:

            self.ignore_list = np.array(ignore_list, dtype=np.uint8)

            # If all points are ignored, set this station as ignored
            if np.all(ignore_list):
                self.ignore_station = True



        # Store the number of measurement
        self.kmeas = len(self.time_data)

        # Calculate JD of each point
        self.JD_data = self.jdt_ref + self.time_data/86400.0

        # Station info
        self.lat = lat
        self.lon = lon
        self.ele = ele
        self.station_id = station_id

        # Observed points
        # - azim_data: azimuth +west of due north
        # - elev_data: elevation angle (altitude)
        self.azim_data = None
        self.elev_data = None

        # Equatorial coordinates
        self.ra_data = None
        self.dec_data = None

        # Apparent magnitude
        self.magnitudes = magnitudes

        # Meteor began/ended inside the FOV flags
        self.fov_beg = fov_beg
        self.fov_end = fov_end

        # Unique observation ID
        self.obs_id = obs_id

        # Observations comment (may be the FF file name)
        self.comment = comment

        ######################################################################################################


        ### CALCULATED DATA ###
        ######################################################################################################

        # Angle between the station, the state vector, and the trajectory
        self.incident_angle = None

        # Weight for the station
        self.weight = None

        # Residuals from the fit
        self.h_residuals = None
        self.h_res_rms = None
        self.v_residuals = None
        self.v_res_rms = None

        # Calculated point to point velocities (in m/s)
        self.velocities = None

        # Average velocities including all previous points up to the current point (for first 4 points the
        #   velocity corresponds to the average velocity through those 4 points)
        self.velocities_prev_point = None

        # Calculated length along the path (meters)
        self.length = None

        # Distance from state vector (meters)
        self.state_vect_dist = None

        # Calculated lag (meters)
        self.lag = None

        # Line parameters used for lag calculation (first element is the line slope, i.e. velocity in m/s)
        self.lag_line = None

        # Initial velocity
        self.v_init = None

        # Jacchia fit parameters for these observations
        self.jacchia_fit = None


        # Modelled RA and Dec
        self.model_ra = None
        self.model_dec = None

        # Modelled azimuth and elevation
        self.model_azim = None
        self.model_elev = None

        # Modelled values for the input type data
        self.model_fit1 = None
        self.model_fit2 = None

        # ECI coordinates of observed CPA to the radiant line, with the station fixed in time at jdt_ref
        self.meas_eci = None

        # ECI vector of observed CPA to the radiant line, with the station moving in time
        self.meas_eci_los = None

        # ECI coordinates of radiant CPA to the observed line of sight
        self.model_eci = None

        # Arrays for geo coords of closest points of approach of observed lines of sight to the radiant line
        #   (i.e. points on the LoS lines)
        self.meas_lat = None
        self.meas_lon = None
        self.meas_ht = None
        self.meas_range = None

        # Arrays for geo coords of closest points of approach of the radiant line to the observed lines of 
        #   sight (i.e. points on the trajectory)
        self.model_lat = None
        self.model_lon = None
        self.model_ht = None
        self.model_range = None

        # Coordinates of the first point (observed)
        self.rbeg_lat = None
        self.rbeg_lon = None
        self.rbeg_ele = None
        self.rbeg_jd = None

        # Coordinates of the last point (observed)
        self.rend_lat = None
        self.rend_lon = None
        self.rend_ele = None
        self.rend_jd = None

        # Absolute magntiudes
        self.absolute_magnitudes = None

        ######################################################################################################


        # If inputs are RA and Dec
        if meastype == 1:
            
            self.ra_data = meas1
            self.dec_data = meas2

            # Calculate azimuthal coordinates
            self.calcAzimuthal()


        # If inputs are azimuth +east of due north, and elevation angle
        elif meastype == 2:
            
            self.azim_data = meas1
            self.elev_data = meas2

        # If inputs are azimuth +west of due south, and zenith angle
        elif meastype == 3:

            self.azim_data = (meas1 + np.pi)%(2*np.pi)
            self.elev_data = np.pi/2.0 - meas2

        # If input are azimuth +north of due east, and zenith angle
        elif meastype == 4:

            self.azim_data = (np.pi/2.0 - meas1)%(2*np.pi)
            self.elev_data = np.pi/2.0 - meas2

        else:

            print("Measurement type 'meastype' =", meastype, 'invalid!')
            sys.exit()

        

        # Calculate equatorial coordinates
        self.calcEquatorial()

        # Calculate the Earth-centered interial coordinates of observed points
        self.calcECI()

        # Calculate position of the station in ECI coordinates (only for the reference JD, used for 
        # intersecting planes solution)
        self.x_stat, self.y_stat, self.z_stat = geo2Cartesian(self.lat, self.lon, self.ele, self.jdt_ref)
        self.stat_eci = np.array([self.x_stat, self.y_stat, self.z_stat])

        # Calculate positions of the station in ECI coordinates, for each JD of individual measurements
        # (used for the lines of sight least squares approach)
        self.stat_eci_los = np.array(geo2Cartesian_vect(self.lat, self.lon, self.ele, self.JD_data)).T

        # Fit a plane through the given points
        self.plane_N = self.planeFit()


        ### EXCLUDED POINTS ###
        ######################################################################################################

        self.excluded_time = excluded_time

        self.excluded_indx_range = []

        # Get the indices of measurements between which there is an excluded part of the trajectory
        if self.excluded_time is not None:

            # Get minimum and maximum excluded times
            excluded_time_min, excluded_time_max = min(self.excluded_time), max(self.excluded_time)


            # Make sure the excluded time is within the observations
            if (excluded_time_min >= np.min(self.time_data)) and (excluded_time_max <= np.max(time_data)):

                excluded_indx_min = 0
                excluded_indx_max = len(self.time_data) - 1

                # Find indices of excluded times, taking the ignored points into account
                for i, t in enumerate(self.time_data[self.ignore_list == 0]):
                    
                    if t <= excluded_time_min:
                        excluded_indx_min = i

                    if t >= excluded_time_max:
                        excluded_indx_max = i
                        break


                self.excluded_indx_range = [excluded_indx_min, excluded_indx_max]

            else:

                print('Excluded time range', self.excluded_time, 'is outside the observation times!')


        ######################################################################################################

        # ### PLOT RESULTS

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # # Plot station position
        # ax.scatter(self.x_stat, self.y_stat, self.z_stat, s=50)

        # # Plot line of sight
        # #ax.scatter(self.x_stat + self.x_eci, self.y_stat + self.y_eci, self.z_stat + self.z_eci, c='red')
        # ax.quiver(self.x_stat, self.y_stat, self.z_stat, self.x_eci, self.y_eci, self.z_eci, length=1.0,
        #         normalize=True, arrow_length_ratio=0.1)

        # # ax.scatter(0, 0, 0, s=50)
        # # ax.scatter(self.x_eci, self.y_eci, self.z_eci, c='red')

        # d = -np.array([self.x_stat, self.y_stat, self.z_stat]).dot(self.plane_N)
        # #d = -np.array([0, 0, 0]).dot(self.plane_N)

        # print('d', d)

        # # create x,y
        # xx, yy = np.meshgrid(np.arange(self.x_stat - 1, self.x_stat + 2), np.arange(self.y_stat - 1, self.y_stat + 2))
        # #xx, yy = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2))

        # # calculate corresponding z
        # z = (-self.plane_N[0]*xx - self.plane_N[1]*yy - d)*1.0/self.plane_N[2]

        # # Plot plane normal
        # ax.scatter(*(self.plane_N + self.stat_eci))

        # print('N:', self.plane_N)

        # print(z)

        # # plot the surface
        # ax.plot_surface(xx, yy, z, color='green', alpha=0.5)

        # ax.set_xlim([-1 + self.x_stat, 1 + self.x_stat])
        # ax.set_ylim([-1 + self.y_stat, 1 + self.y_stat])
        # ax.set_zlim([-1 + self.z_stat, 1 + self.z_stat])

        # plt.show()

        # ###



    def calcAzimuthal(self):
        """ Calculate azimuthal coordinates from right ascension and declination. """

        # Let the JD data be fixed to the reference time - this is done because for CAMS data the azimuthal to 
        # equatorial conversion was done without considering the flow of time during the meteor's appearance.
        # NOTE: If your data does account for the changing time, then jdt_ref_vect should be:
        #   jdt_ref_vect = self.JD_data
        jdt_ref_vect = np.zeros_like(self.ra_data) + self.jdt_ref

        # Calculate azimuth and elevation
        self.azim_data, self.elev_data = raDec2AltAz_vect(self.ra_data, self.dec_data, jdt_ref_vect, self.lat, 
            self.lon)



    def calcEquatorial(self):
        """ Calculates equatorial coordinates from the given azimuthal coordinates. """ 

        # Calculate RA and declination for the plane intersection method
        self.ra_data, self.dec_data = altAz2RADec_vect(self.azim_data, self.elev_data, self.jdt_ref, self.lat, 
            self.lon)

        # Calculate RA and declination for the line of sight method
        self.ra_data_los, self.dec_data_los = altAz2RADec_vect(self.azim_data, self.elev_data, self.JD_data, 
            self.lat, self.lon)



    def calcECI(self):
        """ Calculate Earth-centered intertial coordinates from RA and Dec. """

        # Calculate measurement ECI coordinates for the planes intersection method
        self.meas_eci = np.array(raDec2ECI(self.ra_data, self.dec_data)).T
        self.x_eci, self.y_eci, self.z_eci = self.meas_eci.T

        # Calculate measurement ECI coordinates for the line of sight method
        self.meas_eci_los = np.array(raDec2ECI(self.ra_data_los, self.dec_data_los)).T
        self.x_eci_los, self.y_eci_los, self.z_eci_los = self.meas_eci_los.T



    def planeFit(self):
        """ Fits a plane through station position and observed points. """

        # Add meteor line of sight positions and station positions to single arays.
        #   Only use non-ignored points
        x_data = np.append(self.x_eci[self.ignore_list == 0], 0)
        y_data = np.append(self.y_eci[self.ignore_list == 0], 0)
        z_data = np.append(self.z_eci[self.ignore_list == 0], 0)

        A = np.c_[x_data, y_data, np.ones(x_data.shape[0])]

        # Fit a linear plane through the data points, return plane params (form: aX + bY + d = Z)
        C,_,_,_ = scipy.linalg.lstsq(A, z_data)

        # Calculate the plane normal
        N = np.array([C[0], C[1], -1.0])

        # Norm the normal vector to unit length
        N = vectNorm(N)

        return N



class PlaneIntersection(object):
    def __init__(self, obs1, obs2):
        """ Calculate the plane intersection between two stations. 
            
        Arguments:
            obs1: [ObservedPoints] Observations from the first station.
            obs2: [ObservedPoints] Observations from the second station.

        """

        self.obs1 = obs1
        self.obs2 = obs2

        # Calculate the observed angular length of the track from the first station
        obsangle1 = np.arccos(np.dot(self.obs1.meas_eci[0], self.obs1.meas_eci[-1]))

        # Calculate the observed angular length of the track from the second station
        obsangle2 = np.arccos(np.dot(self.obs2.meas_eci[0], self.obs2.meas_eci[-1]))


        ### Calculate the angle between the pair of planes (convergence angle) ###
        ######################################################################################################
        
        # Calculate the cosine of the convergence angle
        ang_cos = np.dot(self.obs1.plane_N, self.obs2.plane_N)

        # Make sure the cosine is in the proper range
        self.conv_angle = np.arccos(np.abs(np.clip(ang_cos, -1, 1)))

        ######################################################################################################


        # Calculate the plane intersection radiant ECI vector
        self.radiant_eci = np.cross(self.obs1.plane_N, self.obs2.plane_N)
        self.radiant_eci = vectNorm(self.radiant_eci)

        # If the last measurement is closer to the radiant than the first point, reverse signs
        if np.dot(self.obs1.meas_eci[0], self.radiant_eci) < np.dot(self.obs1.meas_eci[-1], self.radiant_eci):
            self.radiant_eci = -self.radiant_eci

        # Calculate the radiant position in RA and Dec
        self.radiant_eq = eci2RaDec(self.radiant_eci)


        ###### Calculate the closest point of approach (CPA) from the stations to the radiant line,
        ###### that is, a vector pointing from each station to the radiant line, which magnitude
        ###### corresponds to the distance to the radiant line

        ### Calculate the unit vector pointing from the 1st station to the radiant line ###
        ######################################################################################################

        self.w1 = np.cross(self.radiant_eci, self.obs1.plane_N)

        # Normalize the vector
        self.w1 = vectNorm(self.w1)

        # Invert vector orientation if pointing towards the station, not the radiant line
        if np.dot(self.w1, self.obs1.meas_eci[0]) < 0:
            self.w1 = -self.w1
        
        ######################################################################################################


        ### Calculate the unit vector pointing from the 2nd station to the radiant line ###
        ######################################################################################################

        self.w2 = np.cross(self.radiant_eci, self.obs2.plane_N)

        # Normalize the vector
        self.w2 = vectNorm(self.w2)

        # Invert vector orientation if pointing towards the station, not the radiant line
        if np.dot(self.w2, self.obs2.meas_eci[0]) < 0:
            self.w2 = -self.w2
        ######################################################################################################


        ### Calculate the range from stations to the radiant line ###
        ######################################################################################################

        # Calculate the difference in position of the two stations
        stat_diff = self.obs1.stat_eci - self.obs2.stat_eci

        # Calculate the angle between the pointings to the radiant line
        stat_cosangle = np.dot(self.w1, self.w2)


        # Calculate the range from the 1st station to the radiant line
        stat_range1 = (stat_cosangle*np.dot(stat_diff, self.w2) - np.dot(stat_diff, self.w1))/(1.0 \
            - stat_cosangle**2)

        # Calculate the CPA vector for the 1st station
        self.rcpa_stat1 = stat_range1*self.w1


        # Calculate the range from the 2nd station to the radiant line
        stat_range2 = (np.dot(stat_diff, self.w2) - stat_cosangle*np.dot(stat_diff, self.w1))/(1.0 \
            - stat_cosangle**2)

        # Calculate the CPA vector for the 2nd station
        self.rcpa_stat2 = stat_range2*self.w2


        # Calculate the position of the CPA with respect to the first camera, in ECI coordinates
        self.cpa_eci = obs1.stat_eci + self.rcpa_stat1

        ######################################################################################################

        # Calculate the statistical weight of the radiant solution
        self.weight = obsangle1*obsangle2*np.sin(self.conv_angle)**2



    def show(self):
        """ Shows the intersection of the two planes in 3D. """

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        observations = [self.obs1, self.obs2]

        # Calculate one point on the meteor trajectory
        traj_point, _, _ = findClosestPoints(self.obs1.stat_eci, self.obs1.meas_eci[0], self.cpa_eci, \
            self.radiant_eci)

        # Calculate the plot limits
        x_min = min([self.obs1.x_stat, self.obs2.x_stat, traj_point[0]])
        x_max = max([self.obs1.x_stat, self.obs2.x_stat, traj_point[0]])
        y_min = min([self.obs1.y_stat, self.obs2.y_stat, traj_point[1]])
        y_max = max([self.obs1.y_stat, self.obs2.y_stat, traj_point[1]])
        z_min = min([self.obs1.z_stat, self.obs2.z_stat, traj_point[2]])
        z_max = max([self.obs1.z_stat, self.obs2.z_stat, traj_point[2]])

        # Normalize the plot limits so they are rectangular
        delta_x = x_max - x_min
        delta_y = y_max - y_min
        delta_z = z_max - z_min
        delta_max = max([delta_x, delta_y, delta_z])

        x_diff = delta_max - delta_x
        x_min -= x_diff/2
        x_max += x_diff/2

        y_diff = delta_max - delta_y
        y_min -= y_diff/2
        y_max += y_diff/2

        z_diff = delta_max - delta_z
        z_min -= z_diff/2
        z_max += z_diff/2


        # Convert meters to km
        x_min /= 1000
        x_max /= 1000
        y_min /= 1000
        y_max /= 1000
        z_min /= 1000
        z_max /= 1000

        # Calculate the quiver arrow length
        arrow_len = 0.2*np.sqrt((x_min - x_max)**2 + (y_min - y_max)**2 + (z_min - z_max)**2)

        # Plot stations and observations
        for obs in observations:

            # Station positions
            ax.scatter(obs.x_stat/1000, obs.y_stat/1000, obs.z_stat/1000, s=50)

            # Lines of sight
            ax.quiver(obs.x_stat/1000, obs.y_stat/1000, obs.z_stat/1000, obs.x_eci/1000, obs.y_eci/1000, \
                obs.z_eci/1000, length=arrow_len, normalize=True, arrow_length_ratio=0.1, color='blue')

            d = -np.array([obs.x_stat/1000, obs.y_stat/1000, obs.z_stat/1000]).dot(obs.plane_N)

            # Create x,y
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))

            # Calculate corresponding z
            z = (-obs.plane_N[0]*xx - obs.plane_N[1]*yy - d)*1.0/obs.plane_N[2]

            # Plot plane normal
            ax.quiver(obs.x_stat/1000, obs.y_stat/1000, obs.z_stat/1000, *obs.plane_N, length=arrow_len/2, 
                normalize=True, arrow_length_ratio=0.1, color='green')

            # Plot the plane
            ax.plot_surface(xx, yy, z, alpha=0.25)


        # Plot the radiant state vector
        rad_x, rad_y, rad_z = -self.radiant_eci/1000
        rst_x, rst_y, rst_z = traj_point/1000
        ax.quiver(rst_x, rst_y, rst_z, rad_x, rad_y, rad_z, length=arrow_len, normalize=True, color='red', \
            arrow_length_ratio=0.1)

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')

        # Change the size of ticks (make them smaller)
        ax.tick_params(axis='both', which='major', labelsize=8)


        plt.show()



def numStationsNotIgnored(observations):
    """ Take a list of ObservedPoints and returns the number of stations that are actually to be used and 
        are not ignored in the solution.

    Arguments: 
        observations: [list] A list of ObservedPoints objects.

    Return:
        [int] Number of stations that are used in the solution.

    """

    return len([obs for obs in observations if obs.ignore_station == False])



def angleSumMeasurements2Line(observations, state_vect, radiant_eci, weights=None, gravity=False):
    """ Sum all angles between the radiant line and measurement lines of sight.

        This function is used as a cost function for the least squares radiant solution of Borovicka et 
        al. (1990). The difference from the original approach is that the distancesfrom the radiant line
        have been replaced with angles.

    Arguments:
        observations: [list] A list of ObservedPoints objects which are containing meteor observations.
        state_vect: [3 element ndarray] Estimated position of the initial state vector in ECI coordinates.
        radiant_eci: [3 element ndarray] Unit 3D vector of the radiant in ECI coordinates.

    Keyword arguments:
        weights: [list] A list of statistical weights for every station. None by default.
        gravity: [bool] If True, the gravity drop will be taken into account.

    Return:
        angle_sum: [float] Sum of angles between the estimated trajectory line and individual lines of sight.

    """

    # If the weights were not given, use 1 for every weight
    if weights is None:
        weights = np.ones(len(observations))

        # Set weights for stations that are not used to 0
        weights = np.array([w if (observations[i].ignore_station == False) else 0 \
            for i, w in enumerate(weights)])

    # Make sure there are weights larger than 0
    if sum(weights) <= 0:
        weights = np.ones(len(observations))

        # Set weights for stations that are not used to 0
        weights = np.array([w if (observations[i].ignore_station == False) else 0 \
            for i, w in enumerate(weights)])



    # Move the state vector to the beginning of the trajectory
    state_vect = moveStateVector(state_vect, radiant_eci, observations)

    # Find the earliest point in time
    t0 = min([obs.time_data[0] for obs in observations])

    angle_sum = 0.0
    weights_sum = 1e-10

    # Go through all observations from all stations
    for i, obs in enumerate(observations):
        
        # Go through all measured positions
        for t, meas_eci, stat_eci, ignore in zip(obs.time_data, obs.meas_eci_los, obs.stat_eci_los, \
            obs.ignore_list):

            # Skip the point if it is to be ignored
            if ignore:
                continue

            # Get the ECI coordinates of the projection of the measurement line of sight on the radiant line
            _, rad_cpa, _ = findClosestPoints(stat_eci, meas_eci, state_vect, radiant_eci)


            # Take the gravity drop into account
            #   Note: here we assume that the acceleration due to gravity is fixed at the given height,
            #   which might cause an offset of a few meters for events longer than 5 seconds
            if gravity:


                # Calculate the time in seconds from the beginning of the meteor
                t_rel = t - t0

                # Compute the model point modified due to gravity, assuming zero vertical velocity
                rad_cpa = applyGravityDrop(rad_cpa, t_rel, vectMag(rad_cpa), 0.0)

                # # Calculate the gravitational acceleration at the given height
                # g = G*EARTH.MASS/(vectMag(rad_cpa)**2)

                # # Determing the sign of the initial time
                # time_sign = np.sign(t_rel)

                # # Calculate the amount of gravity drop from a straight trajectory (handle the case when the time
                # #   can be negative)
                # drop = time_sign*(1.0/2)*g*t_rel**2

                # # Apply gravity drop to ECI coordinates
                # rad_cpa -= drop*vectNorm(rad_cpa)


                # print('-----')
                # print('Station:', obs.station_id)
                # print('t:', t_rel)
                # print('g:', g)

                # print('Drop:', drop)


            # Calculate the unit vector pointing from the station to the point on the trajectory
            station_ray = rad_cpa - stat_eci
            station_ray = vectNorm(station_ray)

            # Calculate the angle between the observed LoS as seen from the station and the radiant line
            cosangle = np.dot(meas_eci, station_ray)

            # Make sure the cosine is within limits and calculate the angle
            angle_sum += weights[i]*np.arccos(np.clip(cosangle, -1, 1))

            weights_sum += weights[i]


    return angle_sum/weights_sum




def minimizeAngleCost(params, observations, weights=None, gravity=False):
    """ A helper function for minimization of angle deviations. """

    state_vect, radiant_eci = np.hsplit(params, 2)
    
    return angleSumMeasurements2Line(observations, state_vect, radiant_eci, weights=weights, gravity=gravity)




def calcSpatialResidual(jdt_ref, jd, state_vect, radiant_eci, stat, meas, gravity=False):
    """ Calculate horizontal and vertical residuals from the radiant line, for the given observed point.

    Arguments:
        jd: [float] Julian date
        state_vect: [3 element ndarray] ECI position of the state vector
        radiant_eci: [3 element ndarray] radiant direction vector in ECI
        stat: [3 element ndarray] position of the station in ECI
        meas: [3 element ndarray] line of sight from the station, in ECI

    Keyword arguments:
        gravity: [bool] Apply the correction for Earth's gravity.

    Return:
        (hres, vres): [tuple of floats] residuals in horitontal and vertical direction from the radiant line

    """


    # Note:
    #   This function has been tested (without the gravity influence part) and it produces good results


    meas = vectNorm(meas)

    # Calculate closest points of approach (observed line of sight to radiant line) from the state vector
    obs_cpa, rad_cpa, d = findClosestPoints(stat, meas, state_vect, radiant_eci)

    # # Apply the gravity drop
    # if gravity:

    #     # Compute the relative time
    #     t_rel = 86400*(jd - jdt_ref)

    #     # Correct the point on the trajectory for gravity
    #     rad_cpa = applyGravityDrop(rad_cpa, t_rel, vectMag(rad_cpa), 0.0)

    #     # ###########################

    #     # # Calculate closest points of approach (observed line of sight to radiant line) from the gravity corrected
    #     # #   point
    #     # obs_cpa, _, d = findClosestPoints(stat, meas, rad_cpa, radiant_eci)

    #     # ##!!!!!


    # Vector pointing from the point on the trajectory to the point on the line of sight
    p = obs_cpa - rad_cpa

    # # Calculate geographical coordinates of the point on the trajectory
    # lat, lon, elev = cartesian2Geo(jd, *rad_cpa)

    # Calculate geographical coordinates of the state vector
    lat, lon, elev = cartesian2Geo(jd, *state_vect)

    # Calculate ENU (East, North, Up) vector at the position of the state vector, and direction of the radiant
    nn = np.array(ecef2ENU(lat, lon, *radiant_eci))

    # Convert the vector to polar coordinates
    theta = np.arctan2(nn[1], nn[0])
    phi = np.arccos(nn[2]/vectMag(nn))

    # Local reference frame unit vectors
    hx = np.array([            -np.cos(theta),              np.sin(theta),         0.0])
    vz = np.array([-np.cos(phi)*np.sin(theta), -np.cos(phi)*np.cos(theta), np.sin(phi)])
    hy = np.array([ np.sin(phi)*np.sin(theta),  np.sin(phi)*np.cos(theta), np.cos(phi)])
    
    # Calculate local reference frame unit vectors in ECEF coordinates
    ehorzx = enu2ECEF(lat, lon, *hx)
    ehorzy = enu2ECEF(lat, lon, *hy)
    evert  = enu2ECEF(lat, lon, *vz)

    ehx = np.dot(p, ehorzx)
    ehy = np.dot(p, ehorzy)

    # Calculate vertical residuals
    vres = np.sign(ehx)*np.hypot(ehx, ehy)

    # Calculate horizontal residuals
    hres = np.dot(p, evert)

    return hres, vres



def lineFuncLS(params, x, y, weights):
    """ Line defined by slope and intercept. Version for least squares.
    
    Arguments:
        params: [list] Line parameters 
        x: [float] Independant variable
        y: [float] Estimated values

    Keyword arguments:
        weight: [float] Weight of the residual.

    Return:
        [float]: line given by (m, k) evaluated at x

    """

    # Compute the residuals and apply weights (sqrt of weights is takes because the value will be squared in
    #   the LS function)
    return (lineFunc(x, *params) - y)*np.sqrt(weights)



def jacchiaLagFunc(t, a1, a2):
    """ Jacchia (1955) model for modeling lengths along the trail of meteors, modified to fit the lag (length 
        along the trail minus the linear part, estimated by fitting a line to the first part of observations, 
        where the length is still linear) instead of the length along the trail. 
    
    Arguments:
        t: [float] time in seconds at which the Jacchia function will be evaluated
        a1: [float] 1st acceleration term
        a2: [float] 2nd acceleration term

    Return:
        [float] Jacchia model defined by a1 and a2, estimated at point in time t

    """

    return -np.abs(a1)*np.exp(np.abs(a2)*t)



def jacchiaLengthFunc(t, a1, a2, v_init, k):
    """ Jacchia (1955) model for modelling lengths along the trail of meteors. 
    
    Arguments:
        t: [float] Time in seconds at which the Jacchia function will be evaluated.
        a1: [float] 1st decelerationn term.
        a2: [float] 2nd deceleration term.
        v_init: [float] Initial velocity in m/s.
        k: [float] Initial offset in length.

    Return:
        [float] Jacchia model defined by a1 and a2, estimated at point in time t.

    """


    return k + v_init*t - np.abs(a1)*np.exp(np.abs(a2)*t)



def jacchiaVelocityFunc(t, a1, a2, v_init):
    """ Derivation of the Jacchia (1955) model, used for calculating velocities from the fitted model. 
    
    Arguments:
        t: [float] Time in seconds at which the Jacchia function will be evaluated.
        a1: [float] 1st decelerationn term.
        a2: [float] 2nd deceleration term.
        v_init: [float] Initial velocity in m/s.
        k: [float] Initial offset in length.

    Return:
        [float] velocity at time t

    """

    return v_init - np.abs(a1*a2)*np.exp(np.abs(a2)*t)



def checkWeights(observations, weights):
    """ Check weight values and make sure they can be used. """

    # If the weights were not given, use 1 for every weight
    if weights is None:
        weights = np.ones(len(observations))

        # Set weights for stations that are not used to 0
        weights = np.array([w if (observations[i].ignore_station == False) else 0 \
            for i, w in enumerate(weights)])

    # Make sure there are weights larger than 0
    if sum(weights) <= 0:
        weights = np.ones(len(observations))

        # Set weights for stations that are not used to 0
        weights = np.array([w if (observations[i].ignore_station == False) else 0 \
            for i, w in enumerate(weights)])


    return weights



def timingResiduals(params, observations, t_ref_station, weights=None, ret_stddev=False):
    """ Calculate the sum of absolute differences between timings of given stations using the length from
        respective stations.
    
    Arguments:
        params: [ndarray] Timing differences from the reference station (NOTE: reference station should NOT be 
            in this list).
        observations: [list] A list of ObservedPoints objects.
        t_ref_station: [int] Index of the reference station.

    Keyword arguments:
        weights: [list] A list of statistical weights for every station.
        ret_stddev: [bool] Returns the standard deviation instead of the cost function.
    
    Return:
        [float] Average absolute difference between the timings from all stations using the length for
            matching.

    """

    # Make sure weight values are OK
    weights = checkWeights(observations, weights)

    stat_count = 0

    state_vect_distances = []

    # Go through observations from all stations
    for i, obs in enumerate(observations):

        # Time difference is 0 for the reference stations
        if i == t_ref_station:
            t_diff = 0

        else:
            # Take the estimated time difference for all other stations
            t_diff = params[stat_count]
            stat_count += 1


        # Calculate the shifted time
        time_shifted = obs.time_data + t_diff

        # Add length to length list
        state_vect_distances.append([time_shifted,  obs.state_vect_dist])



    cost_sum = 0
    cost_point_count = 0
    weights_sum = 1e-10

    # Keep track of stations with confirmed overlaps
    confirmed_overlaps = []

    # Go through all pairs of observations (i.e. stations)
    for i in range(len(observations)):

        # Skip ignored stations
        if observations[i].ignore_station:
            continue

        for j in range(len(observations)):
            
            # Skip ignored stations
            if observations[j].ignore_station:
                continue


            # Skip pairing the same observations again
            if j <= i:
                continue

            # Extract times and lengths from both stations
            time1, len1 = state_vect_distances[i]
            time2, len2 = state_vect_distances[j]

            # Exclude ignored points
            time1 = time1[observations[i].ignore_list == 0]
            len1 = len1[observations[i].ignore_list == 0]
            time2 = time2[observations[j].ignore_list == 0]
            len2 = len2[observations[j].ignore_list == 0]

            # Find common points in length between both stations
            common_pts = np.where((len2 >= np.min(len1)) & (len2 <= np.max(len1)))


            # Continue without fitting the timing is there is no, or almost no overlap
            if len(common_pts[0]) < 4:
                continue


            # Keep track of stations with confirmed overlaps
            confirmed_overlaps.append(observations[i].station_id)
            confirmed_overlaps.append(observations[j].station_id)
            

            # Take only the common points
            time2 = time2[common_pts]
            len2 = len2[common_pts]


            # If there are any excluded points in the reference observations, do not take their
            # pairs from the other site into consideration
            if observations[i].excluded_indx_range:

                # Extract excluded indices
                excluded_indx_min, excluded_indx_max = observations[i].excluded_indx_range

                # Get the range of lengths inside the exclusion zone
                len1_excluded_min = len1[excluded_indx_min]
                len1_excluded_max = len1[excluded_indx_max]

                # Select only those lengths in the other station which are outside the exclusion zone
                temp_arr = np.c_[time2, len2]
                temp_arr = temp_arr[~((temp_arr[:, 1] >= len1_excluded_min) \
                    & (temp_arr[:, 1] <= len1_excluded_max))]

                time2, len2 = temp_arr.T


            # Interpolate the first (i.e. reference length)
            len1_interpol = scipy.interpolate.interp1d(len1, time1)

            # Calculate the residuals using smooth approximation of L1 (absolute value) cost
            z = (len1_interpol(len2) - time2)**2

            # Calculate the cost function sum
            cost_sum += weights[i]*weights[j]*np.sum(2*(np.sqrt(1 + z) - 1))

            # Add the weight sum
            weights_sum += weights[i]*weights[j]

            # Add the total number of points to the cost counter
            cost_point_count += len(z)


    # Exclude stations with no time overlap with other stations
    if (len(observations) > 2):
        confirmed_overlaps = list(set(confirmed_overlaps))
        for obs in observations:
            if obs.station_id not in confirmed_overlaps:
                obs.ignore_station = True
                obs.ignore_list = np.ones(len(obs.time_data), dtype=np.uint8)


    # If no points were compared, return infinite
    if cost_point_count == 0:
        return np.inf

    # Calculate the standard deviation of the fit
    dist_stddev = np.sqrt(cost_sum/weights_sum/cost_point_count)

    if ret_stddev:

        # Returned for reporting the goodness of fit
        return dist_stddev

    else:

        # Returned for minimization
        return cost_sum/weights_sum/cost_point_count



def moveStateVector(state_vect, radiant_eci, observations):
        """ Moves the state vector position along the radiant line until it is before any points which are
            projected on it. This is used to make sure that lengths and lags are properly calculated.
        
        Arguments:
            state_vect: [ndarray] (x, y, z) ECI coordinates of the initial state vector (meters).
            radiant_eci: [ndarray] (x, y, z) components of the unit radiant direction vector.
            observations: [list] A list of ObservationPoints objects which hold measurements from individual
                stations.

        Return:
            rad_cpa_beg: [ndarray] (x, y, z) ECI coordinates of the beginning point of the trajectory.

        """

        rad_cpa_list = []
        radiant_ang_dist_list = []

        # Go through all non-ignored observations from all stations
        nonignored_observations = [obstmp for obstmp in observations if not obstmp.ignore_station]
        for obs in nonignored_observations:

            # Calculate closest points of approach (observed line of sight to radiant line) of the first point
            # on the trajectory across all stations
            _, rad_cpa, _ = findClosestPoints(obs.stat_eci_los[0], obs.meas_eci_los[0], state_vect, 
                radiant_eci)

            rad_cpa_list.append(rad_cpa)


            # Compute angular distance from the first point to the radiant
            rad_ang_dist = angleBetweenVectors(radiant_eci, vectNorm(rad_cpa))
            radiant_ang_dist_list.append(rad_ang_dist)

            

        # # Choose the state vector with the largest height
        # rad_cpa_beg = rad_cpa_list[np.argmax([vectMag(rad_cpa_temp) for rad_cpa_temp in rad_cpa_list])]

        # Choose the state vector as the point of initial observation closest to the radiant
        rad_cpa_beg = rad_cpa_list[np.argmin([rad_ang_dist for rad_ang_dist in radiant_ang_dist_list])]


        return rad_cpa_beg




class MCUncertainties(object):
    def __init__(self, mc_traj_list):
        """ Container for standard deviations and confidence intervals of trajectory parameters calculated 
        using Monte Carlo. 
        """

        # Confidence interval value (95%)
        self.ci = 95

        # A list with all trajectory objects calculated via Monte Carlo
        self.mc_traj_list = mc_traj_list

        # State vector position
        self.state_vect_mini = None
        self.state_vect_mini_ci = None
        self.x = None
        self.x_ci = None
        self.y = None
        self.y_ci = None
        self.z = None
        self.z_ci = None

        # Velocity state vector
        self.vx = None
        self.vx_ci = None
        self.vy = None
        self.vy_ci = None
        self.vz = None
        self.vz_ci = None

        # Radiant vector
        self.radiant_eci_mini = None
        self.radiant_eci_mini_ci = None

        # Beginning/ending points
        self.rbeg_lon = None
        self.rbeg_lon_ci = None
        self.rbeg_lon_m = None
        self.rbeg_lat = None
        self.rbeg_lat_ci = None
        self.rbeg_lat_m = None
        self.rbeg_ele = None
        self.rbeg_ele_ci = None

        self.rend_lon = None
        self.rend_lon_ci = None
        self.rend_lon_m = None
        self.rend_lat = None
        self.rend_lat_ci = None
        self.rend_lat_m = None
        self.rend_ele = None
        self.rend_ele_ci = None

        # Apparent radiant position (radians)
        self.ra = None
        self.ra_ci = None
        self.dec = None
        self.dec_ci = None

        # Apparent azimuth and altitude
        self.azimuth_apparent = None
        self.azimuth_apparent_ci = None
        self.elevation_apparent = None
        self.elevation_apparent_ci = None

        # Estimated average velocity
        self.v_avg = None
        self.v_avg_ci = None

        # Estimated initial velocity
        self.v_init = None
        self.v_init_ci = None

        # Longitude of the reference point on the trajectory (rad)
        self.lon_ref = None
        self.lon_ref_ci = None

        # Latitude of the reference point on the trajectory (rad)
        self.lat_ref = None
        self.lat_ref_ci = None

        # Height of the reference point on the trajectory (meters)
        self.ht_ref = None
        self.ht_ref_ci = None

        # Geocentric latitude of the reference point (rad)
        self.lat_geocentric = None
        self.lat_geocentric_ci = None

        # Apparent zenith angle (before the correction for Earth's gravity)
        self.zc = None
        self.zc_ci = None

        # Zenith distance of the geocentric radiant (after the correction for Earth's gravity)
        self.zg = None
        self.zg_ci = None

        # Velocity at infinity
        self.v_inf = None
        self.v_inf_ci = None

        # Geocentric velocity (m/s)
        self.v_g = None
        self.v_g_ci = None

        # Geocentric radiant position (radians)
        self.ra_g = None
        self.ra_g_ci = None
        self.dec_g = None
        self.dec_g_ci = None

        # Ecliptic coordinates of the radiant (radians)
        self.L_g = None
        self.L_g_ci = None
        self.B_g = None
        self.B_g_ci = None

        # Sun-centered ecliptic rectangular coordinates of the average position on the meteor's trajectory 
        # (in kilometers)
        self.meteor_pos = None
        self.meteor_pos_ci = None

        # Helioventric velocity of the meteor (m/s)
        self.v_h = None
        self.v_h_ci = None

        # Corrected heliocentric velocity vector of the meteoroid using the method of Sato & Watanabe (2014)
        self.v_h_x = None
        self.v_h_x_ci = None
        self.v_h_y = None
        self.v_h_y_ci = None
        self.v_h_z = None
        self.v_h_z_ci = None

        # Corrected ecliptci coordinates of the meteor using the method of Sato & Watanabe (2014)
        self.L_h = None
        self.L_h_ci = None
        self.B_h = None
        self.B_h_ci = None

        # Solar longitude (radians)
        self.la_sun = None
        self.la_sun_ci = None

        # Semi-major axis (AU)
        self.a = None
        self.a_ci = None

        # Eccentricty
        self.e = None
        self.e_ci = None

        # Inclination (radians)
        self.i = None
        self.i_ci = None

        # Argument of perihelion (radians)
        self.peri = None
        self.peri_ci = None

        # Ascending node (radians)
        self.node = None
        self.node_ci = None

        # Longitude of perihelion (radians)
        self.pi = None
        self.pi_ci = None

        # Latitude of perihelion (radians)
        self.b = None
        self.b_ci = None

        # Perihelion distance (AU)
        self.q = None
        self.q_ci = None

        # Aphelion distance (AU)
        self.Q = None
        self.Q_ci = None

        # True anomaly at the moment of contact with Earth (radians)
        self.true_anomaly = None
        self.true_anomaly_ci = None

        # Exxentric anomaly (radians)
        self.eccentric_anomaly = None
        self.eccentric_anomaly_ci = None

        # Mean anomaly (radians)
        self.mean_anomaly = None
        self.mean_anomaly_ci = None

        # Calculate the date and time of the last perihelion passage (datetime object)
        self.last_perihelion = None
        self.last_perihelion_ci = None

        # Mean motion in the orbit (rad/day)
        self.n = None
        self.n_ci = None

        # Orbital period
        self.T = None
        self.T_ci = None

        # Tisserand's parameter with respect to Jupiter
        self.Tj = None
        self.Tj_ci = None

# Preserve compatibility with pickle files genrated before the typo fix
MCUncertanties = MCUncertainties



def calcMCUncertainties(traj_list, traj_best):
    """ Takes a list of trajectory objects and returns the standard deviation of every parameter. 

    Arguments:
        traj_list: [list] A list of Trajectory objects, each is the result of an individual Monte Carlo run.
        traj_best: [Trajectory object] Trajectory which is chosen to the be the best of all MC runs.

    Return:
        un: [MCUncertainties object] Object containing the uncertainty of every calculated parameter.
    """


    # Init a new container for uncertainties
    un = MCUncertainties(traj_list)

    # Initial velocity
    un.v_init = np.std([traj.v_init for traj in traj_list])
    un.v_init_ci = confidenceInterval([traj.v_init for traj in traj_list], un.ci)

    # State vector
    un.x = np.std([traj.state_vect_mini[0] for traj in traj_list])
    un.x_ci = confidenceInterval([traj.state_vect_mini[0] for traj in traj_list], un.ci)
    un.y = np.std([traj.state_vect_mini[1] for traj in traj_list])
    un.y_ci = confidenceInterval([traj.state_vect_mini[1] for traj in traj_list], un.ci)
    un.z = np.std([traj.state_vect_mini[2] for traj in traj_list])
    un.z_ci = confidenceInterval([traj.state_vect_mini[2] for traj in traj_list], un.ci)

    un.state_vect_mini = np.array([un.x, un.y, un.z])
    un.state_vect_mini_ci = np.array([un.x_ci, un.y_ci, un.z_ci])


    rad_x = np.std([traj.radiant_eci_mini[0] for traj in traj_list])
    rad_x_ci = confidenceInterval([traj.radiant_eci_mini[0] for traj in traj_list], un.ci)
    rad_y = np.std([traj.radiant_eci_mini[1] for traj in traj_list])
    rad_y_ci = confidenceInterval([traj.radiant_eci_mini[1] for traj in traj_list], un.ci)
    rad_z = np.std([traj.radiant_eci_mini[2] for traj in traj_list])
    rad_z_ci = confidenceInterval([traj.radiant_eci_mini[2] for traj in traj_list], un.ci)

    un.radiant_eci_mini = np.array([rad_x, rad_y, rad_z])
    un.radiant_eci_mini_ci = np.array([rad_x_ci, rad_y_ci, rad_z_ci])

    # Velocity state vector
    un.vx = abs(traj_best.v_init*traj_best.radiant_eci_mini[0]*(un.v_init/traj_best.v_init
        + rad_x/traj_best.radiant_eci_mini[0]))
    un.vx_ci = confidenceInterval([traj.v_init*traj.radiant_eci_mini[0] for traj in traj_list], un.ci)
    un.vy = abs(traj_best.v_init*traj_best.radiant_eci_mini[1]*(un.v_init/traj_best.v_init
        + rad_y/traj_best.radiant_eci_mini[1]))
    un.vy_ci = confidenceInterval([traj.v_init*traj.radiant_eci_mini[1] for traj in traj_list], un.ci)
    un.vz = abs(traj_best.v_init*traj_best.radiant_eci_mini[2]*(un.v_init/traj_best.v_init
        + rad_z/traj_best.radiant_eci_mini[2]))
    un.vz_ci = confidenceInterval([traj.v_init*traj.radiant_eci_mini[2] for traj in traj_list], un.ci)


    # Beginning/ending points
    N_beg = EARTH.EQUATORIAL_RADIUS/np.sqrt(1.0 - (EARTH.E**2)*np.sin(traj_best.rbeg_lat)**2)
    un.rbeg_lon = scipy.stats.circstd([traj.rbeg_lon for traj in traj_list])
    un.rbeg_lon_ci = confidenceInterval([traj.rbeg_lon for traj in traj_list], un.ci, angle=True)
    un.rbeg_lon_m = np.sin(un.rbeg_lon)*np.cos(traj_best.rbeg_lat)*N_beg
    un.rbeg_lat = np.std([traj.rbeg_lat for traj in traj_list])
    un.rbeg_lat_ci = confidenceInterval([traj.rbeg_lat for traj in traj_list], un.ci)
    un.rbeg_lat_m = np.sin(un.rbeg_lat)*N_beg
    un.rbeg_ele = np.std([traj.rbeg_ele for traj in traj_list])
    un.rbeg_ele_ci = confidenceInterval([traj.rbeg_ele for traj in traj_list], un.ci)

    N_end = EARTH.EQUATORIAL_RADIUS/np.sqrt(1.0 - (EARTH.E**2)*np.sin(traj_best.rend_lat)**2)
    un.rend_lon = scipy.stats.circstd([traj.rend_lon for traj in traj_list])
    un.rend_lon_ci = confidenceInterval([traj.rend_lon for traj in traj_list], un.ci, angle=True)
    un.rend_lon_m = np.sin(un.rend_lon)*np.cos(traj_best.rend_lat)*N_end
    un.rend_lat = np.std([traj.rend_lat for traj in traj_list])
    un.rend_lat_ci = confidenceInterval([traj.rend_lat for traj in traj_list], un.ci)
    un.rend_lat_m = np.sin(un.rend_lat)*N_end
    un.rend_ele = np.std([traj.rend_ele for traj in traj_list])
    un.rend_ele_ci = confidenceInterval([traj.rend_ele for traj in traj_list], un.ci)


    if traj_best.orbit is not None:

        # Apparent ECI
        un.ra = scipy.stats.circstd([traj.orbit.ra for traj in traj_list])
        un.ra_ci = confidenceInterval([traj.orbit.ra for traj in traj_list], un.ci, angle=True)
        un.dec = np.std([traj.orbit.dec for traj in traj_list])
        un.dec_ci = confidenceInterval([traj.orbit.dec for traj in traj_list], un.ci)
        un.v_avg = np.std([traj.orbit.v_avg for traj in traj_list])
        un.v_avg_ci = confidenceInterval([traj.orbit.v_avg for traj in traj_list], un.ci)
        un.v_inf = np.std([traj.orbit.v_inf for traj in traj_list])
        un.v_inf_ci = confidenceInterval([traj.orbit.v_inf for traj in traj_list], un.ci)
        un.azimuth_apparent = scipy.stats.circstd([traj.orbit.azimuth_apparent for traj in traj_list])
        un.azimuth_apparent_ci = confidenceInterval([traj.orbit.azimuth_apparent for traj in traj_list], \
            un.ci, angle=True)
        un.elevation_apparent = np.std([traj.orbit.elevation_apparent for traj in traj_list])
        un.elevation_apparent_ci = confidenceInterval([traj.orbit.elevation_apparent for traj in traj_list], \
            un.ci)

        # Apparent ground-fixed
        un.ra_norot = scipy.stats.circstd([traj.orbit.ra_norot for traj in traj_list])
        un.ra_norot_ci = confidenceInterval([traj.orbit.ra_norot for traj in traj_list], un.ci, angle=True)
        un.dec_norot = np.std([traj.orbit.dec_norot for traj in traj_list])
        un.dec_norot_ci = confidenceInterval([traj.orbit.dec_norot for traj in traj_list], un.ci)
        un.v_avg_norot = np.std([traj.orbit.v_avg_norot for traj in traj_list])
        un.v_avg_norot_ci = confidenceInterval([traj.orbit.v_avg_norot for traj in traj_list], un.ci)
        un.v_init_norot = np.std([traj.orbit.v_init_norot for traj in traj_list])
        un.v_init_norot_ci = confidenceInterval([traj.orbit.v_init_norot for traj in traj_list], un.ci)
        un.azimuth_apparent_norot = scipy.stats.circstd([traj.orbit.azimuth_apparent_norot for traj \
            in traj_list])
        un.azimuth_apparent_norot_ci = confidenceInterval([traj.orbit.azimuth_apparent_norot for traj \
            in traj_list], un.ci, angle=True)
        un.elevation_apparent_norot = np.std([traj.orbit.elevation_apparent_norot for traj in traj_list])
        un.elevation_apparent_norot_ci = confidenceInterval([traj.orbit.elevation_apparent_norot for traj \
            in traj_list], un.ci)

        # Reference point on the meteor trajectory
        un.lon_ref = scipy.stats.circstd([traj.orbit.lon_ref for traj in traj_list])
        un.lon_ref_ci = confidenceInterval([traj.orbit.lon_ref for traj in traj_list], un.ci, angle=True)
        un.lat_ref = np.std([traj.orbit.lat_ref for traj in traj_list])
        un.lat_ref_ci = confidenceInterval([traj.orbit.lat_ref for traj in traj_list], un.ci)
        un.lat_geocentric = np.std([traj.orbit.lat_geocentric for traj in traj_list])
        un.lat_geocentric_ci = confidenceInterval([traj.orbit.lat_geocentric for traj in traj_list], un.ci)
        un.ht_ref = np.std([traj.orbit.ht_ref for traj in traj_list])
        un.ht_ref_ci = confidenceInterval([traj.orbit.ht_ref for traj in traj_list], un.ci)

        # Geocentric
        un.ra_g = scipy.stats.circstd([traj.orbit.ra_g for traj in traj_list])
        un.ra_g_ci = confidenceInterval([traj.orbit.ra_g for traj in traj_list], un.ci, angle=True)
        un.dec_g = np.std([traj.orbit.dec_g for traj in traj_list])
        un.dec_g_ci = confidenceInterval([traj.orbit.dec_g for traj in traj_list], un.ci)
        un.v_g = np.std([traj.orbit.v_g for traj in traj_list])
        un.v_g_ci = confidenceInterval([traj.orbit.v_g for traj in traj_list], un.ci)

        # Meteor position in Sun-centred rectangular coordinates
        meteor_pos_x = np.std([traj.orbit.meteor_pos[0] for traj in traj_list])
        meteor_pos_x_ci = confidenceInterval([traj.orbit.meteor_pos[0] for traj in traj_list], un.ci)
        meteor_pos_y = np.std([traj.orbit.meteor_pos[1] for traj in traj_list])
        meteor_pos_y_ci = confidenceInterval([traj.orbit.meteor_pos[1] for traj in traj_list], un.ci)
        meteor_pos_z = np.std([traj.orbit.meteor_pos[2] for traj in traj_list])
        meteor_pos_z_ci = confidenceInterval([traj.orbit.meteor_pos[2] for traj in traj_list], un.ci)

        un.meteor_pos = np.array([meteor_pos_x, meteor_pos_y, meteor_pos_z])
        un.meteor_pos_ci = np.array([meteor_pos_x_ci, meteor_pos_y_ci, meteor_pos_z_ci])

        # Zenith angles
        un.zc = np.std([traj.orbit.zc for traj in traj_list])
        un.zc_ci = confidenceInterval([traj.orbit.zc for traj in traj_list], un.ci)
        un.zg = np.std([traj.orbit.zg for traj in traj_list])
        un.zg_ci = confidenceInterval([traj.orbit.zg for traj in traj_list], un.ci)


        # Ecliptic geocentric
        un.L_g = scipy.stats.circstd([traj.orbit.L_g for traj in traj_list])
        un.L_g_ci = confidenceInterval([traj.orbit.L_g for traj in traj_list], un.ci, angle=True)
        un.B_g = np.std([traj.orbit.B_g for traj in traj_list])
        un.B_g_ci = confidenceInterval([traj.orbit.B_g for traj in traj_list], un.ci)
        un.v_h = np.std([traj.orbit.v_h for traj in traj_list])
        un.v_h_ci = confidenceInterval([traj.orbit.v_h for traj in traj_list], un.ci)

        # Ecliptic heliocentric
        un.L_h = scipy.stats.circstd([traj.orbit.L_h for traj in traj_list])
        un.L_h_ci = confidenceInterval([traj.orbit.L_h for traj in traj_list], un.ci, angle=True)
        un.B_h = np.std([traj.orbit.B_h for traj in traj_list])
        un.B_h_ci = confidenceInterval([traj.orbit.B_h for traj in traj_list], un.ci)
        un.v_h_x = np.std([traj.orbit.v_h_x for traj in traj_list])
        un.v_h_x_ci = confidenceInterval([traj.orbit.v_h_x for traj in traj_list], un.ci)
        un.v_h_y = np.std([traj.orbit.v_h_y for traj in traj_list])
        un.v_h_y_ci = confidenceInterval([traj.orbit.v_h_y for traj in traj_list], un.ci)
        un.v_h_z = np.std([traj.orbit.v_h_z for traj in traj_list])
        un.v_h_z_ci = confidenceInterval([traj.orbit.v_h_z for traj in traj_list], un.ci)

        # Orbital elements
        un.la_sun = scipy.stats.circstd([traj.orbit.la_sun for traj in traj_list])
        un.la_sun_ci = confidenceInterval([traj.orbit.la_sun for traj in traj_list], un.ci, angle=True)
        un.a = np.std([traj.orbit.a for traj in traj_list])
        un.a_ci = confidenceInterval([traj.orbit.a for traj in traj_list], un.ci)
        un.e = np.std([traj.orbit.e for traj in traj_list])
        un.e_ci = confidenceInterval([traj.orbit.e for traj in traj_list], un.ci)
        un.i = np.std([traj.orbit.i for traj in traj_list])
        un.i_ci = confidenceInterval([traj.orbit.i for traj in traj_list], un.ci)
        un.peri = scipy.stats.circstd([traj.orbit.peri for traj in traj_list])
        un.peri_ci = confidenceInterval([traj.orbit.peri for traj in traj_list], un.ci, angle=True)
        un.node = scipy.stats.circstd([traj.orbit.node for traj in traj_list])
        un.node_ci = confidenceInterval([traj.orbit.node for traj in traj_list], un.ci, angle=True)
        un.pi = scipy.stats.circstd([traj.orbit.pi for traj in traj_list])
        un.pi_ci = confidenceInterval([traj.orbit.pi for traj in traj_list], un.ci, angle=True)
        un.b = np.std([traj.orbit.b for traj in traj_list])
        un.b_ci = confidenceInterval([traj.orbit.b for traj in traj_list], un.ci)
        un.q = np.std([traj.orbit.q for traj in traj_list])
        un.q_ci = confidenceInterval([traj.orbit.q for traj in traj_list], un.ci)
        un.Q = np.std([traj.orbit.Q for traj in traj_list])
        un.Q_ci = confidenceInterval([traj.orbit.Q for traj in traj_list], un.ci)
        un.true_anomaly = scipy.stats.circstd([traj.orbit.true_anomaly for traj in traj_list])
        un.true_anomaly_ci = confidenceInterval([traj.orbit.true_anomaly for traj in traj_list], un.ci, \
            angle=True)
        un.eccentric_anomaly = scipy.stats.circstd([traj.orbit.eccentric_anomaly for traj in traj_list])
        un.eccentric_anomaly_ci = confidenceInterval([traj.orbit.eccentric_anomaly for traj in traj_list], \
            un.ci, angle=True)
        un.mean_anomaly = scipy.stats.circstd([traj.orbit.mean_anomaly for traj in traj_list])
        un.mean_anomaly_ci = confidenceInterval([traj.orbit.mean_anomaly for traj in traj_list], un.ci, \
            angle=True)

        # Last perihelion uncertanty (days)
        last_perihelion_list = [datetime2JD(traj.orbit.last_perihelion) for traj \
            in traj_list if isinstance(traj.orbit.last_perihelion, datetime.datetime)]
        if len(last_perihelion_list):
            un.last_perihelion = np.std(last_perihelion_list)
            un.last_perihelion_ci = confidenceInterval(last_perihelion_list, un.ci)
        else:
            un.last_perihelion = np.nan
            un.last_perihelion_ci = (np.nan, np.nan)
        

        # Mean motion in the orbit (rad/day)
        un.n = np.std([traj.orbit.n for traj in traj_list])
        un.n_ci = confidenceInterval([traj.orbit.n for traj in traj_list], un.ci)

        # Orbital period
        un.T = np.std([traj.orbit.T for traj in traj_list])
        un.T_ci = confidenceInterval([traj.orbit.T for traj in traj_list], un.ci)

        # Tisserand's parameter
        un.Tj = np.std([traj.orbit.Tj for traj in traj_list])
        un.Tj_ci = confidenceInterval([traj.orbit.Tj for traj in traj_list], un.ci)
    

    return un



def calcCovMatrices(mc_traj_list):
    """ Calculate the covariance matrix between orbital elements, and initial state vector using all Monte 
        Carlo trajectories. The covariance matrix is weighted by the timing residuals.

        The orbital covariance matrix is calculated for radians and the inital state vector matrix in meters
        and meters per second.

    Arguments:
        mc_traj_list: [list] A list of Trajectory objects from Monte Carlo runs.


    Return:
        orbit_cov, state_vect_cov: [tuple of ndarrays] Orbital and initial state vector covariance matrices.
    """

    # Filter out those trajectories for which the last perihelion time could not be estimated
    mc_traj_list = [traj for traj in mc_traj_list if traj.orbit.last_perihelion is not None]

    # If there are no good orbits, do not estimate the covariance matrix
    if not mc_traj_list:
        return np.zeros((6, 6)) - 1, np.zeros((6, 6)) - 1

    # Extract timing residuals
    timing_res_list = np.array([traj.timing_res for traj in mc_traj_list])

    # Make sure the timing residual is not 0
    timing_res_list[timing_res_list == 0] = 1e-10

    # Calculate the weights using timing residuals
    weights = np.min(timing_res_list)/timing_res_list
    weights = weights

    # Extract orbit elements
    e_list = np.array([traj.orbit.e for traj in mc_traj_list])
    q_list = np.array([traj.orbit.q for traj in mc_traj_list])
    tp_list = np.array([datetime2JD(traj.orbit.last_perihelion) for traj in mc_traj_list])
    node_list = np.degrees(normalizeAngleWrap(np.array([traj.orbit.node for traj in mc_traj_list])))
    peri_list = np.degrees(normalizeAngleWrap(np.array([traj.orbit.peri for traj in mc_traj_list])))
    i_list = np.degrees(normalizeAngleWrap(np.array([traj.orbit.i for traj in mc_traj_list])))
    

    # Calculate the orbital covariance (angles in degrees)
    orbit_input = np.c_[e_list, q_list, tp_list, node_list, peri_list, i_list].T
    orbit_cov = np.cov(orbit_input, aweights=weights)


    # Extract inital state vectors
    state_vect_list = np.array([traj.state_vect_mini for traj in mc_traj_list])
    initial_vel_vect_list = np.array([traj.v_init*traj.radiant_eci_mini for traj in mc_traj_list])

    # Calculate inital state vector covariance
    state_vect_input = np.hstack([state_vect_list, initial_vel_vect_list]).T
    state_vect_cov = np.cov(state_vect_input, aweights=weights)


    return orbit_cov, state_vect_cov




def trajNoiseGenerator(traj, noise_sigma):
    """ Given a base trajectory object and the observation uncertainly, this generator will generate
        new trajectory objects with noise-added obsevations. 
    
    Arguments:
        traj: [Trajectory] Trajectory instance.
        noise_sigma: [float] Standard deviations of noise to add to the data.

    Yields:
        [counter, traj_mc, traj.observations]:
            - counter: [int] Number of trajectories generated since the generator init.
            - traj_mc: [Trajectory] Trajectory object with added noise
            - traj.observations: [list] A list of original noise-free ObservedPoints.
    """


    counter = 0

    # Do mc_runs Monte Carlo runs
    while True:

        # Make a copy of the original trajectory object
        traj_mc = copy.deepcopy(traj)

        # Set the measurement type to alt/az
        traj_mc.meastype = 2
        
        # Reset the observation points
        traj_mc.observations = []

        # Reinitialize the observations with points sampled using a Gaussian kernel
        for obs in traj.observations:


            azim_noise_list = []
            elev_noise_list = []

            # Go through all ECI unit vectors of measurement LoS, add the noise and calculate alt/az coords
            for jd, rhat in zip(obs.JD_data, obs.meas_eci_los):

                # Unit vector pointing from the station to the meteor observation point in ECI coordinates
                rhat = vectNorm(rhat)

                ### Add noise to simulated coordinates (taken over from Gural solver source)

                zhat = np.array([0.0, 0.0, 1.0])
                uhat = vectNorm(np.cross(rhat, zhat))
                vhat = vectNorm(np.cross(uhat, rhat))

                # # sqrt(2)/2*noise in each orthogonal dimension
                # NOTE: This is a bad way to do it because the estimated fit residuals are already estimated
                #   in the prependicular direction to the trajectory line
                # sigma = noise_sigma*np.abs(obs.ang_res_std)/np.sqrt(2.0)

                # # Make sure sigma is positive, if not set it to 1/sqrt(2) degrees
                # if (sigma < 0) or np.isnan(sigma):
                #     sigma = np.radians(1)/np.sqrt(2)

                # Compute noise level to add to observations
                sigma = noise_sigma*np.abs(obs.ang_res_std)

                # Make sure sigma is positive, if not set it to 1 degree
                if (sigma < 0) or np.isnan(sigma):
                    sigma = np.radians(1)

                # Add noise to observations
                meas_eci_noise = rhat + np.random.normal(0, sigma)*uhat + np.random.normal(0, sigma)*vhat

                # Normalize to a unit vector
                meas_eci_noise = vectNorm(meas_eci_noise)

                ###

                # Calculate RA, Dec for the given point
                ra, dec = eci2RaDec(meas_eci_noise)

                # Calculate azimuth and altitude of this direction vector
                azim, elev = raDec2AltAz(ra, dec, jd, obs.lat, obs.lon)

                azim_noise_list.append(azim)
                elev_noise_list.append(elev)

        
        
            # Fill in the new trajectory object - the time is assumed to be absolute
            traj_mc.infillTrajectory(azim_noise_list, elev_noise_list, obs.time_data, obs.lat, obs.lon, \
                obs.ele, station_id=obs.station_id, excluded_time=obs.excluded_time, \
                ignore_list=obs.ignore_list, magnitudes=obs.magnitudes, fov_beg=obs.fov_beg, \
                fov_end=obs.fov_end, obs_id=obs.obs_id, comment=obs.comment)

            
        # Do not show plots or perform additional optimizations
        traj_mc.verbose = False
        traj_mc.estimate_timing_vel = True
        traj_mc.filter_picks = False
        traj_mc.show_plots = False
        traj_mc.save_results = False

        # Return the modified trajectory object
        yield [counter, traj_mc, traj.observations]

        counter += 1



def checkMCTrajectories(mc_results, timing_res=np.inf, geometric_uncert=False):
    """ Filter out MC computed trajectories and only return successful ones. 
    
    Arguments:
        mc_results: [list] A list of Trajectory objects computed with added noise.

    Keyword arguments:
        timing_res: [float] Timing residual from the original LoS trajectory fit.
        geometric_uncert: [bool] If True, all MC runs will be taken to estimate the uncertainty, not just
            the ones with the better cost function value than the pure geometric solution. Use this when
            the lag is not reliable.

    Returns:
        [list] A filtered list of trajectories.

    """


    if not geometric_uncert:

        # Take only those solutions which have the timing residuals <= than the initial solution
        mc_results = [mc_traj for mc_traj in mc_results if mc_traj.timing_res <= timing_res]

    ##########

    # Reject those solutions for which LoS angle minimization failed
    mc_results = [mc_traj for mc_traj in mc_results if mc_traj.los_mini_status == True]

    # Reject those solutions for which the orbit could not be calculated
    mc_results = [mc_traj for mc_traj in mc_results if (mc_traj.orbit.ra_g is not None) \
        and (mc_traj.orbit.dec_g is not None)]

    print("{:d} successful MC runs done...".format(len(mc_results)))

    return mc_results



def _MCTrajSolve(params):
    """ Internal function. Does a Monte Carlo run of the given trajectory object. Used as a function for
        parallelization. 

    Arguments:
        params: [list]
            - i: [int] Number of MC run to be printed out.
            - traj: [Trajectory object] Trajectory object on which the run will be performed.
            - observations: [list] A list of observations with no noise.

    Return:
        traj: [Trajectory object] Trajectory object with the MC solution.

    """

    i, traj, observations = params

    print('Run No.', i + 1)

    traj.run(_mc_run=True, _orig_obs=observations)

    return traj



def monteCarloTrajectory(traj, mc_runs=None, mc_pick_multiplier=1, noise_sigma=1, geometric_uncert=False, \
    plot_results=True, mc_cores=None):
    """ Estimates uncertanty in the trajectory solution by doing Monte Carlo runs. The MC runs are done 
        in parallel on all available computer cores.

        The uncertanty is taken as the standard deviation of angular measurements. Each point is sampled 
        mc_pick_multiplier times using a symetric 2D Gaussian kernel.

    Arguments:
        traj: [Trajectory object] initial trajectory on which Monte Carlo runs will be performed

    Keyword arguments:
        mc_runs: [int] A fixed number of Monte Carlo simulations. None by default. If it is given, it will
            override mc_pick_multiplier.
        mc_pick_multiplier: [int] Number of MC samples that will be taken for every point. 1 by default.
        noise_sigma: [float] Number of standard deviations to use for adding Gaussian noise to original 
            measurements.
        geometric_uncert: [bool] If True, all MC runs will be taken to estimate the uncertainty, not just
            the ones with the better cost function value than the pure geometric solution. Use this when
            the lag is not reliable.
        plot_results: [bool] Plot the trajectory and orbit spread. True by default.
        mc_cores: [int] Number of CPU cores to use for Monte Carlo parallel procesing. None by default,
            which means that all available cores will be used.
    """



    ### DO MONTE CARLO RUNS ###
    ##########################################################################################################

    # If a fixed number of Monte Carlo simulations is given, use it
    if mc_runs is not None:

        mc_runs = mc_runs

    else:

        # Calculate the total number of Monte Carlo runs, so every point is sampled mc_pick_multiplier times.
        mc_runs = sum([len(obs.time_data) for obs in traj.observations])
        mc_runs = mc_runs*mc_pick_multiplier


    print("Doing", mc_runs, "successful Monte Carlo runs...")


    # Init the trajectory noise generator
    traj_generator = trajNoiseGenerator(traj, noise_sigma)

    
    # Run the MC solutions
    results_check_kwagrs = {"timing_res": traj.timing_res, "geometric_uncert": geometric_uncert}
    mc_results = parallelComputeGenerator(traj_generator, _MCTrajSolve, checkMCTrajectories, mc_runs, \
        results_check_kwagrs=results_check_kwagrs)


    # If there are no MC runs which were successful, recompute using geometric uncertainties
    if len(mc_results) < 2:
        print("No successful MC runs, computing geometric uncertanties...")

        # Run the MC solutions
        geometric_uncert = True
        results_check_kwagrs["geometric_uncert"] = geometric_uncert
        mc_results = parallelComputeGenerator(traj_generator, _MCTrajSolve, checkMCTrajectories, mc_runs, \
            results_check_kwagrs=results_check_kwagrs)


    # Add the original trajectory in the Monte Carlo results, if it is the one which has the best length match
    if traj.orbit.ra_g is not None:
        mc_results.append(traj)

    
    ##########################################################################################################


    # Break the function of there are no trajectories to process
    if len(mc_results) < 2:
        print('!!! Not enough good Monte Carlo runs for uncertaintly estimation!')
        return traj, None


    # Choose the solution with the lowest timing residuals as the best solution
    timing_res_trajs = [traj_tmp.timing_res for traj_tmp in mc_results]
    best_traj_ind = timing_res_trajs.index(min(timing_res_trajs))

    # Choose the best trajectory
    traj_best = mc_results[best_traj_ind]

    # Assign geometric uncertainty flag, if it was changed
    traj_best.geometric_uncert = geometric_uncert

    print('Computing uncertainties...')

    # Calculate the standard deviation of every trajectory parameter
    uncertainties = calcMCUncertainties(mc_results, traj_best)

    print('Computing covariance matrices...')

    # Calculate orbital and inital state vector covariance matrices (angles in degrees)
    traj_best.orbit_cov, traj_best.state_vect_cov = calcCovMatrices(mc_results)


    ### PLOT RADIANT SPREAD (Vg color and length stddev) ###
    ##########################################################################################################

    if (traj.orbit is not None) and plot_results:

        ra_g_list = np.array([traj_temp.orbit.ra_g for traj_temp in mc_results])
        dec_g_list = np.array([traj_temp.orbit.dec_g for traj_temp in mc_results])
        v_g_list = np.array([traj_temp.orbit.v_g for traj_temp in mc_results])/1000
        timing_res_list = np.array([traj_temp.timing_res for traj_temp in mc_results])

        # Color code Vg and length standard deviation
        for plt_flag in ['vg', 'time_res']:

            # Init a celestial plot
            m = CelestialPlot(ra_g_list, dec_g_list, projection='stere', bgcolor='w')

            if plt_flag == 'vg':

                # Plot all MC radiants (geocentric velocities)
                m.scatter(ra_g_list, dec_g_list, c=v_g_list, s=2)

                m.colorbar(label='$V_g$ (km/s)')


                if traj.orbit.ra_g is not None:
                    
                    # Plot original radiant
                    m.scatter(traj.orbit.ra_g, traj.orbit.dec_g, s=20, facecolors='none', edgecolors='r')


                if traj_best.orbit.ra_g is not None:
                    
                    # Plot MC best radiant
                    m.scatter(traj_best.orbit.ra_g, traj_best.orbit.dec_g, s=20, facecolors='none', edgecolors='g')



            elif plt_flag == 'time_res':

                timing_res_list_ms = 1000*timing_res_list

                v_min = np.min(timing_res_list_ms)
                v_max = np.max(timing_res_list_ms)

                # Determine the limits of the colorbar if there are more points
                if len(timing_res_list) > 4:

                    v_max = np.median(timing_res_list_ms) + 2*np.std(timing_res_list_ms)


                # Plot all MC radiants (length fit offsets)
                m.scatter(ra_g_list, dec_g_list, c=timing_res_list_ms, s=2, vmin=v_min, vmax=v_max)

                m.colorbar(label='Time residuals (ms)')


                # Plot original radiant
                m.scatter(traj.orbit.ra_g, traj.orbit.dec_g, s=20, facecolors='none', edgecolors='r')

                # Plot MC best radiant
                m.scatter(traj_best.orbit.ra_g, traj_best.orbit.dec_g, s=20, facecolors='none', edgecolors='g')



            plt.title('Monte Carlo - geocentric radiant')
            # plt.xlabel('$\\alpha_g (\\degree)$')
            # plt.ylabel('$\\delta_g (\\degree)$')

            # plt.tight_layout()

            if traj.save_results:
                savePlot(plt, traj.file_name + '_monte_carlo_eq_' + plt_flag + '.' + traj.plot_file_type, \
                    output_dir=traj.output_dir)


            if traj.show_plots:
                plt.show()

            else:
                plt.clf()
                plt.close()

    ##########################################################################################################



    ### PLOT ORBITAL ELEMENTS SPREAD ###
    ##########################################################################################################

    if (traj.orbit is not None) and plot_results:

        a_list = np.array([traj_temp.orbit.a for traj_temp in mc_results])
        incl_list = np.array([traj_temp.orbit.i for traj_temp in mc_results])
        e_list = np.array([traj_temp.orbit.e for traj_temp in mc_results])
        peri_list = np.array([traj_temp.orbit.peri for traj_temp in mc_results])
        q_list = np.array([traj_temp.orbit.q for traj_temp in mc_results])

        fig = plt.figure()

        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2, sharey=ax1)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4, sharey=ax3)

        # Compute the number of bins
        nbins = int(np.ceil(np.sqrt(len(a_list))))
        if nbins < 10:
            nbins = 10

        # Semimajor axis vs. inclination
        ax1.hist2d(a_list, np.degrees(incl_list), bins=nbins)
        ax1.set_xlabel('a (AU)')
        ax1.set_ylabel('Inclination (deg)')
        plt.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')
        #ax1.get_xaxis().get_major_formatter().set_useOffset(False)
        ax1.ticklabel_format(useOffset=False)

        # Plot the first solution and the MC solution
        if traj.orbit.a is not None:
            ax1.scatter(traj.orbit.a, np.degrees(traj.orbit.i), c='r', linewidth=1, edgecolors='w')

        if traj_best.orbit.a is not None:
            ax1.scatter(traj_best.orbit.a, np.degrees(traj_best.orbit.i), c='g', linewidth=1, edgecolors='w')



        # Plot argument of perihelion vs. inclination
        ax2.hist2d(np.degrees(peri_list), np.degrees(incl_list), bins=nbins)
        ax2.set_xlabel('peri (deg)')
        plt.setp(ax2.get_xticklabels(), rotation=30, horizontalalignment='right')
        #ax2.get_xaxis().get_major_formatter().set_useOffset(False)
        ax2.ticklabel_format(useOffset=False)

        # Plot the first solution and the MC solution
        if traj.orbit.peri is not None:
            ax2.scatter(np.degrees(traj.orbit.peri), np.degrees(traj.orbit.i), c='r', linewidth=1, \
                edgecolors='w')

        if traj_best.orbit.peri is not None:
            ax2.scatter(np.degrees(traj_best.orbit.peri), np.degrees(traj_best.orbit.i), c='g', linewidth=1, \
                edgecolors='w')

        ax2.tick_params(
            axis='y',          # changes apply to the y-axis
            which='both',      # both major and minor ticks are affected
            left='off',        # ticks along the left edge are off
            labelleft='off')   # labels along the left edge are off


        # Plot eccentricity vs. perihelion distance
        ax3.hist2d(e_list, q_list, bins=nbins)
        ax3.set_xlabel('Eccentricity')
        ax3.set_ylabel('q (AU)')
        plt.setp(ax3.get_xticklabels(), rotation=30, horizontalalignment='right')
        #ax3.get_xaxis().get_major_formatter().set_useOffset(False)
        ax3.ticklabel_format(useOffset=False)

        # Plot the first solution and the MC solution
        if traj.orbit.e is not None:
            ax3.scatter(traj.orbit.e, traj.orbit.q, c='r', linewidth=1, edgecolors='w')

        if traj_best.orbit.e is not None:
            ax3.scatter(traj_best.orbit.e, traj_best.orbit.q, c='g', linewidth=1, edgecolors='w')

        # Plot argument of perihelion vs. perihelion distance
        ax4.hist2d(np.degrees(peri_list), q_list, bins=nbins)
        ax4.set_xlabel('peri (deg)')
        plt.setp(ax4.get_xticklabels(), rotation=30, horizontalalignment='right')
        #ax4.get_xaxis().get_major_formatter().set_useOffset(False)
        ax4.ticklabel_format(useOffset=False)

        # Plot the first solution and the MC solution
        if traj.orbit.peri is not None:
            ax4.scatter(np.degrees(traj.orbit.peri), traj.orbit.q, c='r', linewidth=1, edgecolors='w')

        if traj_best.orbit.peri is not None:
            ax4.scatter(np.degrees(traj_best.orbit.peri), traj_best.orbit.q, c='g', linewidth=1, \
                edgecolors='w')
            

        ax4.tick_params(
            axis='y',          # changes apply to the y-axis
            which='both',      # both major and minor ticks are affected
            left='off',        # ticks along the left edge are off
            labelleft='off')   # labels along the left edge are off
        

        plt.tight_layout()
        plt.subplots_adjust(wspace=0)


        if traj.save_results:
            savePlot(plt, traj.file_name + '_monte_carlo_orbit_elems.' + traj.plot_file_type, 
                output_dir=traj.output_dir)


        if traj.show_plots:
            plt.show()

        else:
            plt.clf()
            plt.close()

    ##########################################################################################################


    return traj_best, uncertainties


def copyUncertainties(traj_source, traj_target, copy_mc_traj_instances=False):
    """ Copy uncertainties from one trajectory to the other. 
    
    Arguments:
        traj_source: [Trajectory object] Trajectory object with uncertainties.
        traj_target: [Trajectory object] Trajectory object to which uncertainties will be copied.

    Keyword arguments:
        copy_mc_traj_instances: [bool] Copy all trajectory instances generated during the MC procedure.
            This will make the trajectory pickle file very large. False by default.

    Return:
        traj_target: [Trajectory object] Target trajectory object with copied uncertainties.
    """

    # Copy covariance matrices
    traj_target.orbit_cov, traj_target.state_vect_cov = copy.deepcopy(traj_source.orbit_cov), \
        copy.deepcopy(traj_source.state_vect_cov)

    # Copy uncertainties
    traj_target.uncertainties = copy.deepcopy(traj_source.uncertainties)

    # Handle individual trajectory instances
    if not copy_mc_traj_instances:
        if traj_target.uncertainties is not None:
            del traj_target.uncertainties.mc_traj_list
            traj_target.uncertainties.mc_traj_list = []


    return traj_target





def applyGravityDrop(eci_coord, t, r0, vz):
    """ Given the ECI position of the meteor and the duration of flight, this function calculates the
        drop caused by gravity and returns ECI coordinates of the meteor corrected for gravity drop. As the 
        gravitational acceleration changes with height, the drop is changes too. We assumed that the vertical
        component of the meteor's velocity is constant dervied the modified drop equation.

    Arguments:
        eci_coord: [ndarray] (x, y, z) ECI coordinates of the meteor at the given time t (meters).
        t: [float] Time of meteor since the beginning of the trajectory.
        r0: [float] Distance from the centre of the Earth of the beginning of the meteor.
        vz: [float] Vertical component of the meteor's velocity.

    """

    # Determing the sign of the initial time
    time_sign = np.sign(t)

    # The derived drop function does not work for small vz's, thus the classical drop function is used
    if abs(vz) < 100:

        # Calculate gravitational acceleration at given ECI coordinates
        g = G*EARTH.MASS/r0**2

        # Calculate the amount of gravity drop from a straight trajectory
        drop = time_sign*(1.0/2)*g*t**2


    else:

        # Compute the drop using a drop model with a constant vertical velocity
        drop = time_sign*(G*EARTH.MASS/vz**2)*(r0/(r0 + vz*t) + np.log((r0 + vz*t)/r0) - 1)
    

    # Apply gravity drop to ECI coordinates
    return eci_coord - drop*vectNorm(eci_coord)




class Trajectory(object):
    """ Meteor trajectory solver designed at the University of Western Ontario.

    The solver makes a first estimate using the Ceplecha (1987) plane intersection approach, then refines the 
    solution my miniming the angles between the observed lines of sight and the radiant line. The best 
    solution is found by adding noise to original measurements and doing Monte Carlo runs to find the 
    trajectory whose deceleratioins and velocity profiles match the best, as seen from individual stations.
    The initial velocity is estimated from time vs. length by iteratively fitting a line to it and choosing
    the solution with the lowest standard deviation, which should correspond to the part of the trajectory 
    before the meteor stared to decelerate.
    """


    def __init__(self, jdt_ref, output_dir='.', max_toffset=None, meastype=4, verbose=True, v_init_part=None,\
        v_init_ht=None, estimate_timing_vel=True, monte_carlo=True, mc_runs=None, mc_pick_multiplier=1, \
        mc_noise_std=1.0, geometric_uncert=False, filter_picks=True, calc_orbit=True, show_plots=True, \
        show_jacchia=False, save_results=True, gravity_correction=True, plot_all_spatial_residuals=False, \
        plot_file_type='png', traj_id=None, reject_n_sigma_outliers=3, mc_cores=None):
        """ Init the Ceplecha trajectory solver.

        Arguments:
            jdt_ref: [float] reference Julian date for the measurements. Add provided times should be given
                relative to this number. This is user selectable and can be the time of the first camera, or 
                the first measurement, or some average time for the meteor, but should be close to the time of 
                the meteor. This same reference date/time will be used on all camera measurements for the 
                purposes of computing local sidereal time and making geocentric coordinate transformations, 
                thus it is good that this time corresponds to the beginning of the meteor.

        Keyword arguments:
            output_dir: [str] Path to the output directory where the Trajectory report and 'pickled' object
                will be stored.
            max_toffset: [float] Maximum allowed time offset between cameras in seconds (default 1 second).
            meastype: [float] Flag indicating the type of angle measurements the user is providing for meas1 
                and meas2 below. The following are all in radians:
                        1 = Right Ascension for meas1, declination for meas2, NOTE: epoch of date, NOT J2000!
                        2 = Azimuth +east of due north for meas1, Elevation angle above the horizon for meas2
                        3 = Azimuth +west of due south for meas1, Zenith angle for meas2
                        4 = Azimuth +north of due east for meas1, Zenith angle for meas2
            verbose: [bool] Print out the results and status messages, True by default.
            v_init_part: [float] Fixed part from the beginning of the meteor on which the automated initial
                velocity estimation using the sliding fit will start. Default is 0.25 (25%), but for noisier 
                data this might be bumped up to 0.5.
            v_init_ht: [float] If given, the initial velocity will be estimated as the average velocity
                above the given height in kilometers using data from all stations. None by default, in which
                case the initial velocity will be estimated using the automated siliding fit.
            estimate_timing_vel: [bool/str] Try to estimate the difference in timing and velocity. True by  
                default. A string with the list of fixed time offsets can also be given, e.g. 
                "CA001A":0.42,"CA0005":-0.3.
            monte_carlo: [bool] Runs Monte Carlo estimation of uncertainties. True by default.
            mc_runs: [int] Number of Monte Carlo runs. The default value is the number of observed points.
            mc_pick_multiplier: [int] Number of MC samples that will be taken for every point. 1 by default.
            mc_noise_std: [float] Number of standard deviations of measurement noise to add during Monte
                Carlo estimation.
            geometric_uncert: [bool] If True, all MC runs will be taken to estimate the uncertainty, not just
                the ones with the better cost function value than the pure geometric solution. Use this when
                the lag is not reliable. False by default.
            filter_picks: [bool] If True (default), picks which deviate more than 3 sigma in angular residuals
                will be removed, and the trajectory will be recalculated.
            calc_orbit: [bool] If True, the orbit is calculates as well. True by default
            show_plots: [bool] Show plots of residuals, velocity, lag, meteor position. True by default.
            show_jacchia: [bool] Show the Jacchia fit on the plot with meteor dynamics. False by default.
            save_results: [bool] Save results of trajectory estimation to disk. True by default.
            gravity_correction: [bool] Apply the gravity drop when estimating trajectories. True by default.
            plot_all_spatial_residuals: [bool] Plot all spatial residuals on one plot (one vs. time, and
                the other vs. length). False by default.
            plot_file_type: [str] File extansion of the plot image. 'png' by default, can be 'pdf', 'eps', ...
            traj_id: [str] Trajectory solution identifier. None by default.
            reject_n_sigma_outliers: [float] Reject angular outliers that are n sigma outside the fit.
                This value is 3 (sigma) by default.
            mc_cores: [int] Number of CPU cores to use for Monte Carlo parallell processing. None by default,
                which means that all cores will be used.

        """

        # All time data must be given relative to this Julian date
        self.jdt_ref = jdt_ref

        # Measurement type
        self.meastype = meastype

        # Directory where the trajectory estimation results will be saved
        self.output_dir = output_dir

        # Maximum time offset between cameras
        if max_toffset is None:
            max_toffset = 1.0
        self.max_toffset = max_toffset

        # If verbose True, results and status messages will be printed out, otherwise they will be supressed
        self.verbose = verbose

        # Fixed part from the beginning of the meteor on which the initial velocity estimation using the 
        #   sliding fit will start
        if v_init_part is None:
            v_init_part = 0.25
        self.v_init_part = v_init_part

        # (Optional) Height in kilometers above which points will be taken for estimating the initial
        #   velocity (linear fit)
        self.v_init_ht = v_init_ht

        # Estimating the difference in timing between stations, and the initial velocity if this flag is True
        self.fixed_time_offsets = {}
        if isinstance(estimate_timing_vel, str):

            # If a list of fixed timing offsets was given, parse it into a dictionary
            for entry in estimate_timing_vel.split(','):
                station, offset = entry.split(":")
                self.fixed_time_offsets[station] = float(offset)

            print("Fixed timing given:", self.fixed_time_offsets)

            self.estimate_timing_vel = False

        elif isinstance(estimate_timing_vel, bool):
            self.estimate_timing_vel = estimate_timing_vel
        else:
            self.estimate_timing_vel = True

        # Running Monte Carlo simulations to estimate uncertainties
        self.monte_carlo = monte_carlo

        # Number of Monte Carlo runs
        self.mc_runs = mc_runs

        # Number of MC samples that will be taken for every point
        self.mc_pick_multiplier = mc_pick_multiplier

        # Standard deviatons of measurement noise to add during Monte Carlo runs
        self.mc_noise_std = mc_noise_std

        # If True, pure geometric uncertainties will be computed and culling of solutions based on cost
        #   function value will not be done
        self.geometric_uncert = geometric_uncert

        # Filter bad picks (ones that deviate more than 3 sigma in angular residuals) if this flag is True
        self.filter_picks = filter_picks

        # Calculate orbit if True
        self.calc_orbit = calc_orbit

        # If True, plots will be shown on screen when the trajectory estimation is done
        self.show_plots = show_plots

        # Show Jacchia fit on dynamics plots
        self.show_jacchia = show_jacchia

        # Save results to disk if true
        self.save_results = save_results

        # Apply the correction for gravity when estimating the trajectory
        self.gravity_correction = gravity_correction

        # Plot all spatial residuals on one plot
        self.plot_all_spatial_residuals = plot_all_spatial_residuals

        # Image file type for the plot
        self.plot_file_type = plot_file_type

        # Trajectory solution identifier
        self.traj_id = str(traj_id)

        # n sigma outlier rejection
        self.reject_n_sigma_outliers = reject_n_sigma_outliers

        # Number of CPU cores to be used for MC
        self.mc_cores = mc_cores

        ######################################################################################################


        # Construct a file name for this event
        self.generateFileName()

        # Counts from how many observations are given from the beginning (start from 1)
        # NOTE: This should not the used as the number of observations, use len(traj.observations) instead!
        self.meas_count = 1

        # List of observations
        self.observations = []

        # Minimization status - if True if LoS angle minimization is successfull, False otherwise
        self.los_mini_status = False

        # Index of the station with the reference time
        self.t_ref_station = 0

        # Final estimate of timing offsets between stations
        self.time_diffs_final = None

        # List of plane intersections
        self.intersection_list = None

        # Coordinates of the first point
        self.rbeg_lat = None
        self.rbeg_lon = None
        self.rbeg_ele = None
        self.rbeg_jd = None

        # Coordinates of the end point
        self.rend_lat = None
        self.rend_lon = None
        self.rend_ele = None
        self.rend_jd = None


        # Intersecting planes state vector
        self.state_vect = None

        # Angles (radians) between the trajectory and the station, looking from the state vector determined
        #   by intersecting planes
        self.incident_angles = []

        # Initial state vector (minimization)
        self.state_vect_mini = None

        # Radiant in ECi and equatorial coordinatrs (minimiziation)
        self.radiant_eci_mini = None
        self.radiant_eq_mini = None

        # Calculated initial velocity
        self.v_init = None

        # Calculated average velocity
        self.v_avg = None

        # Status of timing minimization
        self.timing_minimization_successful = False

        # Fit to the best portion of time vs. length
        self.velocity_fit = None

        # Jacchia fit parameters for all observations combined
        self.jacchia_fit = None

        # Cost function value of the time vs. state vector distance fit
        self.timing_res = None

        # Standard deviation of all time differences between individual stations
        self.timing_stddev = -1.0

        # Average position of the meteor
        self.state_vect_avg = None

        # Average JD of the meteor
        self.jd_avg = None

        # Orbit object which contains orbital parameters
        self.orbit = None

        # Uncertainties calculated using Monte Carlo
        self.uncertainties = None
        self.uncertanties = self.uncertainties

        # Orbital covariance matrix (angles in degrees)
        self.orbit_cov = None

        # Initial state vector covariance matrix
        self.state_vect_cov = None



    def generateFileName(self):
        """ Generate a file name for saving results using the reference julian date. """

        self.file_name = jd2Date(self.jdt_ref, dt_obj=True).strftime('%Y%m%d_%H%M%S')


    def infillTrajectory(self, meas1, meas2, time_data, lat, lon, ele, station_id=None, excluded_time=None,
        ignore_list=None, magnitudes=None, fov_beg=None, fov_end=None, obs_id=None, comment=''):
        """ Initialize a set of measurements for a given station. 
    
        Arguments:
            meas1: [list or ndarray] First measurement array (azimuth or R.A., depending on meastype, see 
                meastype documentation for more information). Measurements should be given in radians.
            meas2: [list or ndarray] Second measurement array (altitude, zenith angle or declination, 
                depending on meastype, see meastype documentation for more information), in radians.
            time_data: [list or ndarray] Time in seconds from the reference Julian date.
            lat: [float] WGS84 latitude +N of station in radians.
            lon: [float] WGS84 longitude +E of station in radians.
            ele: [float] EGS96 geoidal elevation of station in meters (not the height above the WGS84 
                ellipsoid!).

        Keyword arguments:
            station_id: [str] Identification of the station. None by default.
            excluded_time: [list] A range of minimum and maximum observation time which should be excluded 
                from the optimization because the measurements are missing in that portion of the time.
            ignore_list: [list or ndarray] A list of 0s and 1s which should be of the equal length as 
                the input data. If a particular data point is to be ignored, number 1 should be put,
                otherwise (if the point should be used) 0 should be used. E.g. the this should could look
                like this: [0, 0, 0, 1, 1, 0, 0], which would mean that the fourth and the fifth points
                will be ignored in trajectory estimation.
            magnitudes: [list] A list of apparent magnitudes of the meteor. None by default.
            fov_beg: [bool] True if the meteor began inside the FOV, False otherwise. None by default.
            fov_end: [bool] True if the meteor ended inside the FOV, False otherwise. None by default.
            obs_id: [int] Unique ID of the observation. This is to differentiate different observations from
                the same station.
            comment: [str] A comment about the observations. May be used to store RMS FF file number on which
                the meteor was observed.
        Return:
            None
        """

        # If station ID was not given, assign it a name
        if station_id is None:
            station_id = self.meas_count

        # If obs_id was not given, assign it
        if obs_id is None:
            obs_id = self.meas_count


        # Convert measuremet lists to numpy arrays
        meas1 = np.array(meas1)
        meas2 = np.array(meas2)
        time_data = np.array(time_data)

        # Add a fixed offset to time data if given
        if str(station_id) in self.fixed_time_offsets:
            time_data += self.fixed_time_offsets[str(station_id)]

        # Skip the observation if all points were ignored
        if ignore_list is not None:
            if np.all(ignore_list):
                print('All points from station {:s} are ignored, not using this station in the solution!'.format(station_id))


        # Init a new structure which will contain the observed data from the given site
        obs = ObservedPoints(self.jdt_ref, meas1, meas2, time_data, lat, lon, ele, station_id=station_id, \
            meastype=self.meastype, excluded_time=excluded_time, ignore_list=ignore_list, \
            magnitudes=magnitudes, fov_beg=fov_beg, fov_end=fov_end, obs_id=obs_id, comment=comment)
            
        # Add observations to the total observations list
        self.observations.append(obs)

        self.meas_count += 1


    def infillWithObs(self, obs, meastype=None):
        """ Infill the trajectory with already initialized ObservedPoints object. 
    
        Arguments:
            obs: [ObservedPoints] Instance of ObservedPoints.

        Keyword arguments:
            meastype: [int] Measurement type. If not given, it will be read from the trajectory object.
        """

        if meastype is None:
            meas1 = obs.meas1
            meas2 = obs.meas2


        # If inputs were RA and Dec
        elif meastype == 1:
            meas1 = obs.ra_data
            meas2 = obs.dec_data

        # If inputs were azimuth +east of due north, and elevation angle
        elif meastype == 2:
            meas1 = obs.azim_data
            meas2 = obs.elev_data


        # If inputs were azimuth +west of due south, and zenith angle
        elif meastype == 3:

            meas1 = (obs.azim_data + np.pi)%(2*np.pi)
            meas2 = np.pi/2.0 - obs.elev_data

        # If input were azimuth +north of due east, and zenith angle
        elif meastype == 4:

            meas1 = (np.pi/2.0 - obs.azim_data)%(2*np.pi)
            meas2 = np.pi/2.0 - obs.elev_data


        ### PRESERVE COMPATBILITY WITH OLD obs OBJECTS ###

        # Check if the observation had any excluded points
        if hasattr(obs, 'excluded_time'):
            excluded_time = obs.excluded_time
        else:
            excluded_time = None

        # Check if it has the ignore list argument
        if hasattr(obs, 'ignore_list'):
            ignore_list = obs.ignore_list

        else:
            ignore_list = np.zeros(len(obs.time_data), dtype=np.uint8)

        # Check for apparent magnitudes
        if hasattr(obs, 'magnitudes'):
            magnitudes = obs.magnitudes

        else:
            magnitudes = None

        # Check if the observation object has FOV beg/end flags
        if not hasattr(obs, 'fov_beg'):
            obs.fov_beg = None
        if not hasattr(obs, 'fov_end'):
            obs.fov_end = None

        # Check if the observation object has obs_id argument
        if not hasattr(obs, 'obs_id'):
            obs.obs_id = None

        # Check if the observation object as the comment entry
        if not hasattr(obs, 'comment'):
            obs.comment = ''

        ### ###


        self.infillTrajectory(meas1, meas2, obs.time_data, obs.lat, obs.lon, obs.ele, \
            station_id=obs.station_id, excluded_time=excluded_time, ignore_list=ignore_list, \
            magnitudes=magnitudes, fov_beg=obs.fov_beg, fov_end=obs.fov_end, obs_id=obs.obs_id, \
            comment=obs.comment)



    def calcAllResiduals(self, state_vect, radiant_eci, observations):
        """ Calculate horizontal and vertical residuals for all observed points. 
            
            The residuals are calculated from the closest point on the line of sight to the point of the 
            radiant line.

        Arguments:
            state_vect: [ndarray] (x, y, z) ECI coordinates of the initial state vector (meters).
            radiant_eci: [ndarray] (x, y, z) components of the unit radiant direction vector.
            observations: [list] A list of ObservationPoints objects which hold measurements from individual
                stations.

        """

        # Go though observations from all stations
        for obs in observations:

            # Init empty lists for residuals
            obs.h_residuals = []
            obs.v_residuals = []

            # Go through all individual position measurement from each site
            for t, jd, stat, meas in zip(obs.time_data, obs.JD_data, obs.stat_eci_los, obs.meas_eci_los):

                # Calculate horizontal and vertical residuals
                hres, vres = calcSpatialResidual(self.jdt_ref, jd, state_vect, radiant_eci, stat, meas, \
                    gravity=self.gravity_correction)

                # Add residuals to the residual list
                obs.h_residuals.append(hres)
                obs.v_residuals.append(vres)

            # Convert residual lists to numpy arrays
            obs.h_residuals = np.array(obs.h_residuals)
            obs.v_residuals = np.array(obs.v_residuals)

            # Calculate RMSD of both residuals
            obs.h_res_rms = RMSD(obs.h_residuals[obs.ignore_list == 0])
            obs.v_res_rms = RMSD(obs.v_residuals[obs.ignore_list == 0])


            # Calculate the angular residuals from the radiant line, with the gravity drop taken care of
            obs.ang_res = angleBetweenSphericalCoords(obs.elev_data, obs.azim_data, obs.model_elev, \
                obs.model_azim)


            # Calculate the standard deviaton of angular residuals in radians, taking the ignored points into
            #   account
            if not obs.ignore_station:
                obs.ang_res_std = RMSD(obs.ang_res[obs.ignore_list == 0])

            else:
                # Compute RMSD for all points if the station is ignored
                obs.ang_res_std = RMSD(obs.ang_res)



    def calcVelocity(self, state_vect, radiant_eci, observations, weights, calc_res=False):
        """ Calculates point to point velocity for the given solution, as well as the average velocity 
            including all previous points up to the given point.


        Arguments:
            state_vect: [ndarray] (x, y, z) ECI coordinates of the initial state vector (meters).
            radiant_eci: [ndarray] (x, y, z) components of the unit radiant direction vector.
            observations: [list] A list of ObservationPoints objects which hold measurements from individual
                stations.
            weights: [list] A list of statistical weights for every station.

        Keyword arguments:
            calc_res: [bool] If True, the cost of lag residuals will be calculated. The timing offsets first 
                need to be calculated for this to work.

        """


        # Go through observations from all stations
        for obs in observations:

            # List of distances from the first trajectory point on the radiant line
            first_pt_distances = []

            # List of distances from the state vector
            state_vect_dist = []

            # Go through all individual position measurement from each site
            for i, (stat, meas) in enumerate(zip(obs.stat_eci_los, obs.meas_eci_los)):

                # Calculate closest points of approach (observed line of sight to radiant line)
                obs_cpa, rad_cpa, d = findClosestPoints(stat, meas, state_vect, radiant_eci)

                # Take the position of the first point as the reference point
                if i == 0:
                    ref_point = np.copy(rad_cpa)

                # Calculate the distance from the first observed point to the projected point on the radiant line
                dist = vectMag(ref_point - rad_cpa)
                
                first_pt_distances.append(dist)

                # Distance from the state vector to the projected point on the radiant line
                state_vect_dist.append(vectMag(state_vect - rad_cpa))


            # Convert the distances (length along the trail) into a numpy array
            obs.length = np.array(first_pt_distances)
            obs.state_vect_dist = np.array(state_vect_dist)


            ### Calculate average velocity including all points up to the given point ###
            velocities_prev_point = []
            for i, (t, l) in enumerate(zip(obs.time_data, obs.length)):

                # For the first 4 points compute the velocity using the first 4 points
                if i < 4:
                    time_part = obs.time_data[:4]
                    len_part = obs.length[:4]

                # Otherwise include all points up to the current point
                else:
                    time_part = obs.time_data[: i+1]
                    len_part = obs.length[: i+1]

                # If there are NaNs or infs, drop them
                filter_mask = np.logical_not(np.isnan(time_part) | np.isinf(time_part) | np.isnan(len_part) \
                    | np.isinf(len_part))

                time_part = time_part[filter_mask]
                len_part = len_part[filter_mask]

                if len(time_part):

                    # Fit a line through time vs. length data
                    popt, _ = scipy.optimize.curve_fit(lineFunc, time_part, len_part)

                    velocities_prev_point.append(popt[0])

                else:

                    # If there are no good points to estimate the velocity on, use NaN
                    velocities_prev_point.append(np.nan)



            obs.velocities_prev_point = np.array(velocities_prev_point)


            ### ###

            ### Length vs. time

            # plt.plot(obs.state_vect_dist, obs.time_data, marker='x', label=str(obs.station_id), zorder=3)

            ##########

            ### Calculate point to point velocities ###

            # Shift the radiant distances one element down (for difference calculation)
            dists_shifted = np.r_[0, obs.length][:-1]

            # Calculate distance differences from point to point (first is always 0)
            dists_diffs = obs.length - dists_shifted

            # Shift the time one element down (for difference calculation)
            time_shifted = np.r_[0, obs.time_data][:-1]

            # Calculate the time differences from point to point
            time_diffs = obs.time_data - time_shifted

            # Replace zeros in time by machine precision value to avoid division by zero errors
            time_diffs[time_diffs == 0] = np.finfo(np.float64).eps

            # Calculate velocity for every point
            obs.velocities = dists_diffs/time_diffs

            ### ###


        # plt.ylabel('Time (s)')
        # plt.xlabel('Distance from state vector (m)')

        # plt.gca().invert_yaxis()

        # plt.legend()
        # plt.grid()
        # plt.savefig('mc_time_offsets.' + self.plot_file_type, dpi=300)
        # plt.show()



        if calc_res:

            # Because the timing offsets have already been applied, the timing offsets are 0
            zero_timing_res = np.zeros(len(self.observations))

            # Calculate the timing offset between the meteor time vs. length
            
            if self.timing_res is None:
                self.timing_res = timingResiduals(zero_timing_res, self.observations, self.t_ref_station, \
                    weights)

            self.timing_stddev = timingResiduals(zero_timing_res, self.observations, self.t_ref_station, \
                weights, ret_stddev=True)


    def calcAvgVelocityAboveHt(self, observations, bottom_ht, weights):
        """ Calculate the average velocity of all points above a given height.

        Arguments:
            observations: [list] A list of ObservationPoints objects which hold measurements from individual
                stations.
            bottom_ht: [float] Height above which points will be used to compute the average velocity (m).
            weights: [list] A list of statistical weights for every station.
    
        Return:
            (v_ht_avg, intercept): 
                v_ht_avg: [float] Average velocity above the given height (m/s).
                intercept: [float] Fit intercept (m).
        """

        # Maker sure weight values are OK
        weights = checkWeights(observations, weights)

        # Construct arrays of times vs. distance from state vector

        all_times = []
        all_state_vect_dists = []
        all_inv_weights = []

        for obs, w in zip(observations, weights):

            # Skip ignored stations
            if obs.ignore_station:
                continue


            # Skip stations with weight 0
            if w <= 0:
                continue


            for t, sv_dist, ht, ignore in zip(obs.time_data, obs.state_vect_dist, obs.model_ht, \
                obs.ignore_list):

                # Skip ignored points
                if ignore:
                    continue

                # Skip heights below the given height
                if ht < bottom_ht:
                    continue

                all_times.append(t)
                all_state_vect_dists.append(sv_dist)
                all_inv_weights.append(1.0/w)


        # If there are less than 4 points, don't estimate the initial velocity this way!
        if len(all_times) < 4:
            print('!!! Error, there are less than 4 points for velocity estimation above the given height of {:.2f} km!'.format(bottom_ht/1000))
            print('Using automated velocity estimation with the sliding fit...')
            return None, None

        # Fit a line through the time vs. state vector distance data
        line_params, _ = scipy.optimize.curve_fit(lineFunc, all_times, all_state_vect_dists, \
            sigma=all_inv_weights)

        return line_params



    def calcLag(self, observations, velocity_fit=None):
        """ Calculate lag by fitting a line to the first part of the points and subtracting the line from the 
            length along the trail.

        Arguments:
            observations: [list] A list of ObservationPoints objects which hold measurements from individual
                stations.

        Keyword arguments:
            velocity_fit: [tuple of float] Initial velocity and fit intercept (m/s and m). None by defualt.

        """

        # Go through observations from all stations
        for obs in observations:


            if velocity_fit is None:

                # Fit a line to the first part of the points
                init_part_size = int(self.v_init_part*len(obs.time_data))

                # If the size is smaller than 4 points, take all point
                if init_part_size < 4:
                    init_part_size = len(obs.time_data)

                # Cut the length and time to the first quarter
                quart_length = obs.length[:init_part_size]
                quart_time = obs.time_data[:init_part_size]

                # Fit a line to the data, estimate the velocity
                obs.lag_line, _ = scipy.optimize.curve_fit(lineFunc, quart_time, quart_length)

                # Calculate lag
                obs.lag = obs.length - lineFunc(obs.time_data, *obs.lag_line)

            else:
                
                obs.lag_line = list(velocity_fit)

                # Calculate lag
                obs.lag = obs.state_vect_dist - lineFunc(obs.time_data, *obs.lag_line)


            # Initial velocity is the slope of the fitted line
            obs.v_init = obs.lag_line[0]            



    def fitJacchiaLag(self, observations):
        """ Fit an exponential model proposed by Jacchia (1955) to the lag. """

        # Go through observations from all stations and do a per station fit
        for obs in observations:

            # Initial parameters
            p0 = np.zeros(2)

            try:
                obs.jacchia_fit, _ = scipy.optimize.curve_fit(jacchiaLagFunc, obs.time_data, obs.lag, p0=p0)

            # If the maximum number of iterations have been reached, skip Jacchia fitting
            except RuntimeError:
                obs.jacchia_fit = p0

            # Force the parameters to be positive
            obs.jacchia_fit = np.abs(obs.jacchia_fit)

            if self.verbose:
                print('Jacchia fit params for station:', obs.station_id, ':', obs.jacchia_fit)


        # Get the time and lag points from all sites
        time_all = np.hstack([obs.time_data[obs.ignore_list == 0] for obs in self.observations \
            if not obs.ignore_station])
        lag_all = np.hstack([obs.lag[obs.ignore_list == 0] for obs in self.observations \
            if not obs.ignore_station])
        time_lag = np.c_[time_all, lag_all]

        # Sort by time
        time_lag = time_lag[time_lag[:, 0].argsort()]

        # Unpack all data points sorted by time
        time_all, lag_all = time_lag.T

        # Do a Jacchia function fit on the collective lag
        p0 = np.zeros(2)

        try:
            jacchia_fit, _ = scipy.optimize.curve_fit(jacchiaLagFunc, time_all, lag_all, p0=p0)

        # If the maximum number of iterations have been reached, skip Jacchia fitting
        except RuntimeError:
            jacchia_fit = p0


        return jacchia_fit



    def estimateTimingAndVelocity(self, observations, weights, estimate_timing_vel=True):
        """ Estimates time offsets between the stations by matching time vs. distance from state vector. 
            The initial velocity is calculated by ineratively fitting a line from the beginning to 20% of the 
            total trajectory, and up to the 80% of the total trajectory. The fit with the lowest standard
            deviation is chosen to represent the initial velocity.

        Arguments:
            observations: [list] A list of ObservationPoints objects which hold measurements from individual
                stations.
            weights: [list] A list of statistical weights for every station.

        Keyword arguments:
            estimate_timing_vel: [bool] If True (default), the time differences and the velocity will be 
                estimated, otherwise the velocity will be estimated as the medial velocity.

        Return:
            (fit_success, velocity_fit, v_init_mini, time_diffs, observations): [tuple]
                fit_success: [bool] True if timing minimization was successful, False otherwise.
                velocity_fit: [tuple] (slope, intercept) tuple of a line fit on the time vs. length data.
                v_init_mini: [float] Estimated initial velocity in m/s.
                time_diffs: [ndarray] Estimated time offsets from individual stations.
                observations: [list] A list of ObservationPoints objects which hold measurements from 
                    individual stations. These objects are modified during timing estimations.

        """

        # Take the initial velocity as the median velocity between all sites
        v_init_list = np.array([obs.v_init for obs in observations if obs.v_init is not None])
        v_init = np.median(v_init_list)

        # Timing differences which will be calculated
        time_diffs = np.zeros(len(observations))

        if not estimate_timing_vel:
            return True, np.zeros(2), v_init, time_diffs, observations

        # Run timing offset estimation if it needs to be done
        if estimate_timing_vel:

            # Initial timing difference between sites is 0 (there are N-1 timing differences, as the time 
            # difference for the reference site is always 0)
            p0 = np.zeros(shape=(len(self.observations) - 1))

            # # Set the time reference station to be the one with the most used points
            # obs_points = [obs.kmeas for obs in self.observations]
            # self.t_ref_station = obs_points.index(max(obs_points))


            if self.verbose:
                print('Initial function evaluation:', timingResiduals(p0, observations, self.t_ref_station, 
                    weights=weights))


            # Set bounds for timing to +/- given maximum time offset
            bounds = []
            for i in range(len(self.observations) - 1):
                bounds.append([-self.max_toffset, self.max_toffset])


            ### Try different methods of optimization until it is successful ##

            #   If there are more than 5 stations, use the advanced L-BFGS-B method by default
            if len(self.observations) >= 5:
                methods = [None]
                maxiter_list = [15000]
            else:
                # If there are less than 5, try faster methods first
                methods = ['SLSQP', 'TNC', None]
                maxiter_list = [1000, None, 15000]

            # Try different methods to minimize timing residuals
            for opt_method, maxiter in zip(methods, maxiter_list):

                # Run the minimization of residuals between all stations
                timing_mini = scipy.optimize.minimize(timingResiduals, p0, args=(observations, \
                    self.t_ref_station, weights), bounds=bounds, method=opt_method, options={'maxiter': maxiter},\
                    tol=1e-12)

                # Stop trying methods if this one was successful
                if timing_mini.success:

                    # Set the final value of the timing residual
                    self.timing_res = timing_mini.fun

                    if self.verbose:
                        print('Successful timing optimization with', opt_method)
                        print("Final function evaluation:", timing_mini.fun)


                    break

                else:
                    print('Unsuccessful timing optimization with', opt_method)

            ### ###


            if not timing_mini.success:

                print('Timing difference and initial velocity minimization failed with the message:')
                print(timing_mini.message)
                print('Try increasing the range of time offsets!')
                v_init_mini = v_init

                velocity_fit = np.zeros(2)
                v_init_mini = 0


        # Check if the velocity should be estimated
        estimate_velocity = False
        timing_minimization_successful = False
        if not estimate_timing_vel:
            estimate_velocity = True
            timing_minimization_successful = True
        else:
            if timing_mini.success:
                estimate_velocity = True
                timing_minimization_successful = True

        # If the minimization was successful, apply the time corrections
        if estimate_velocity:

            stat_count = 0
            for i, obs in enumerate(observations):

                # The timing difference for the reference station is always 0
                if (i == self.t_ref_station) or (not estimate_timing_vel):
                    t_diff = 0

                else:
                    t_diff = timing_mini.x[stat_count]
                    stat_count += 1

                if self.verbose:
                    print('STATION ' + str(obs.station_id) + ' TIME OFFSET = ' + str(t_diff) + ' s')

                # Skip NaN and inf time offsets
                if np.isnan(t_diff) or np.isinf(t_diff):
                    continue

                # Apply the time shift to original time data
                obs.time_data = obs.time_data + t_diff

                # Apply the time shift to the excluded time
                if obs.excluded_time is not None:
                    obs.excluded_time = [ex_time + t_diff for ex_time in obs.excluded_time]

                # Add the final time difference of the site to the list
                time_diffs[i] = t_diff



            # Add in time and distance points, excluding the ignored points
            times = []
            state_vect_dist = []
            weight_list = []
            for obs, wt in zip(observations, weights):

                # Skip ignored stations
                if obs.ignore_station:
                    continue

                times.append(obs.time_data[obs.ignore_list == 0])
                state_vect_dist.append(obs.state_vect_dist[obs.ignore_list == 0])
                weight_list.append(np.zeros_like(obs.time_data[obs.ignore_list == 0]) + wt)

            times = np.concatenate(times).ravel()
            state_vect_dist = np.concatenate(state_vect_dist).ravel()
            weight_list = np.concatenate(weight_list).ravel()

            # Sort points by time
            time_sort_ind = times.argsort()
            times = times[time_sort_ind]
            state_vect_dist = state_vect_dist[time_sort_ind]
            weight_list = weight_list[time_sort_ind]


            stddev_list = []

            # Calculate the velocity on different initial portions of the trajectory

            # Find the best fit by starting from the first few beginning points
            for part_beg in range(4):

                # Find the best fit on different portions of the trajectory
                for part in np.arange(self.v_init_part, 0.8, 0.05):

                    # Get the index of the end of the first portion of points
                    part_end = int(part*len(times))

                    # Make sure there are at least 4 points per every station
                    if (part_end - part_beg) < 4*numStationsNotIgnored(observations):
                        part_end = part_beg + 4*numStationsNotIgnored(observations)


                    # Make sure the end index is not larger than the meteor
                    if part_end >= len(times):
                        part_end = len(times) - 1


                    # Select only the first part of all points
                    times_part = times[part_beg:part_end]
                    state_vect_dist_part = state_vect_dist[part_beg:part_end]
                    weights_list_path = weight_list[part_beg:part_end]

                    # Fit a line to time vs. state_vect_dist
                    velocity_fit = scipy.optimize.least_squares(lineFuncLS, [v_init, 1], args=(times_part, \
                        state_vect_dist_part, weights_list_path), loss='soft_l1')
                    velocity_fit = velocity_fit.x

                    # Calculate the lag and fit a line to it
                    lag_temp = state_vect_dist - lineFunc(times, *velocity_fit)
                    lag_fit = scipy.optimize.least_squares(lineFuncLS, np.ones(2), args=(times, lag_temp, \
                        weight_list), loss='soft_l1')
                    lag_fit = lag_fit.x

                    # Add the point to the considered list only if the lag has a negative trend, or a trend
                    #   that is not *too* positive, about 100 m per second is the limit
                    if lag_fit[0] <= 100:

                        # Calculate the standard deviation of the line fit and add it to the list of solutions
                        line_stddev = RMSD(state_vect_dist_part - lineFunc(times_part, *velocity_fit), \
                            weights=weights_list_path)
                        stddev_list.append([line_stddev, velocity_fit])


            # stddev_arr = np.array([std[0] for std in stddev_list])

            # plt.plot(range(len(stddev_arr)), stddev_arr, label='line')

            # plt.legend()
            # plt.show()



            # If no lags were negative (meaning all fits were bad), use the initially estimated initial 
            # velocity
            if not stddev_list:

                v_init_mini = v_init

                # Redo the lag fit, but with fixed velocity
                vel_intercept, _ = scipy.optimize.curve_fit(lambda x, intercept: lineFunc(x, v_init_mini, \
                    intercept), times, state_vect_dist, p0=[0])

                velocity_fit = [v_init_mini, vel_intercept[0]]


            else:

                # Take the velocity fit with the minimum line standard deviation
                stddev_min_ind = np.argmin([std[0] for std in stddev_list])
                velocity_fit = stddev_list[stddev_min_ind][1]

                # Make sure the velocity is positive
                v_init_mini = np.abs(velocity_fit[0])

                # Calculate the lag for every site
                for obs in observations:
                    obs.lag = obs.state_vect_dist - lineFunc(obs.time_data, *velocity_fit)


                if self.verbose:
                    print('ESTIMATED Vinit:', v_init_mini, 'm/s')

            


        return timing_minimization_successful, velocity_fit, v_init_mini, time_diffs, observations




    def calcLLA(self, state_vect, radiant_eci, observations):
        """ Calculate latitude, longitude and altitude of every point on the observer's line of sight, 
            which is closest to the radiant line.

        Arguments:
            state_vect: [ndarray] (x, y, z) ECI coordinates of the initial state vector (meters).
            radiant_eci: [ndarray] (x, y, z) components of the unit radiant direction vector.
            observations: [list] A list of ObservationPoints objects which hold measurements from individual
                stations.

        """

        ### Compute parameters for gravity drop ###

        # Determine the first time
        t0 = min([obs.time_data[0] for obs in observations])

        # Determine the largest distance from the centre of the Earth and use it as the beginning point
        eci_list = []
        for obs in observations:
        
            # Calculate closest points of approach (observed line of sight to radiant line)
            obs_cpa, rad_cpa, d = findClosestPoints(obs.stat_eci_los[0], obs.meas_eci_los[0], state_vect, \
                radiant_eci)

            eci_list.append(rad_cpa)

        # Find the largest distance from the centre of the Earth
        max_dist_indx = np.argmax([vectMag(r) for r in eci_list])

        # Get ECI coordinates of the largest distance and the distance itself
        eci0 = eci_list[max_dist_indx]
        r0 = vectMag(eci0)

        
        ### Compute the apparent zenith angle ###

        # Compute the apparent radiant
        ra_a, dec_a = eci2RaDec(self.state_vect_mini)

        # Compute alt/az of the apparent radiant
        lat_0, lon_0, _ = cartesian2Geo(self.jdt_ref, *eci0)
        _, alt_a = raDec2AltAz(ra_a, dec_a, self.jdt_ref, lat_0, lon_0)

        # Compute the apparent zenith angle
        zc = np.pi - alt_a

        ####

            
        # Compute the vertical component of the velocity if the orbit was already computed
        v0z = -self.v_init*np.cos(zc)

        ###########################################


        # Go through all observations from all stations
        for obs in observations:

            # Init LLA arrays
            obs.meas_lat = np.zeros_like(obs.time_data)
            obs.meas_lon = np.zeros_like(obs.time_data)
            obs.meas_ht = np.zeros_like(obs.time_data)
            obs.meas_range = np.zeros_like(obs.time_data)

            obs.model_lat = np.zeros_like(obs.time_data)
            obs.model_lon = np.zeros_like(obs.time_data)
            obs.model_ht = np.zeros_like(obs.time_data)
            obs.model_range = np.zeros_like(obs.time_data)


            # Go through all individual position measurement from each site
            for i, (t, stat, meas) in enumerate(zip(obs.time_data, obs.stat_eci_los, obs.meas_eci_los)):

                # Calculate closest points of approach (observed line of sight to radiant line)
                obs_cpa, rad_cpa, d = findClosestPoints(stat, meas, state_vect, radiant_eci)


                ### Take the gravity drop into account ### 
                if self.gravity_correction:

                    # Calculate the time in seconds from the beginning of the meteor
                    t_rel = t - t0

                    # Apply the gravity drop
                    rad_cpa = applyGravityDrop(rad_cpa, t_rel, r0, v0z)
                

                # Calculate the range to the observed CPA
                r_meas = vectMag(obs_cpa - stat)

                # Calculate the coordinates of the observed CPA
                lat_meas, lon_meas, ele_meas = cartesian2Geo(obs.JD_data[i], *obs_cpa)

                obs.meas_lat[i] = lat_meas
                obs.meas_lon[i] = lon_meas
                obs.meas_ht[i] = ele_meas
                obs.meas_range[i] = r_meas


                # Calculate the range to the radiant CPA
                r_model = vectMag(rad_cpa - stat)

                # Calculate the coordinates of the observed CPA
                lat_model, lon_model, ele_model = cartesian2Geo(obs.JD_data[i], *rad_cpa)

                obs.model_lat[i] = lat_model
                obs.model_lon[i] = lon_model
                obs.model_ht[i] = ele_model
                obs.model_range[i] = r_model


            # If the whole station is not ignored
            if not obs.ignore_station:

                # Set the coordinates of the first point on the trajectory, taking the ignored points into account
                obs.rbeg_lat = obs.model_lat[obs.ignore_list == 0][0]
                obs.rbeg_lon = obs.model_lon[obs.ignore_list == 0][0]
                obs.rbeg_ele = obs.model_ht[obs.ignore_list == 0][0]
                obs.rbeg_jd = obs.JD_data[obs.ignore_list == 0][0]

                # Set the coordinates of the last point on the trajectory, taking the ignored points into account
                obs.rend_lat = obs.model_lat[obs.ignore_list == 0][-1]
                obs.rend_lon = obs.model_lon[obs.ignore_list == 0][-1]
                obs.rend_ele = obs.model_ht[obs.ignore_list == 0][-1]
                obs.rend_jd = obs.JD_data[obs.ignore_list == 0][-1]

            # If the station is compltely ignored, compute the coordinates including all points
            else:

                # Set the coordinates of the first point on the trajectory, taking the ignored points into account
                obs.rbeg_lat = obs.model_lat[0]
                obs.rbeg_lon = obs.model_lon[0]
                obs.rbeg_ele = obs.model_ht[0]
                obs.rbeg_jd = obs.JD_data[0]

                # Set the coordinates of the last point on the trajectory, taking the ignored points into account
                obs.rend_lat = obs.model_lat[-1]
                obs.rend_lon = obs.model_lon[-1]
                obs.rend_ele = obs.model_ht[-1]
                obs.rend_jd = obs.JD_data[-1]



        # Make a list of observations without any ignored stations in them
        nonignored_observations = [obs for obs in self.observations if not obs.ignore_station]

        # Find the highest beginning height
        beg_hts = [obs.rbeg_ele for obs in nonignored_observations]
        first_begin = beg_hts.index(max(beg_hts))

        # Set the coordinates of the height point as the first point
        self.rbeg_lat = nonignored_observations[first_begin].rbeg_lat
        self.rbeg_lon = nonignored_observations[first_begin].rbeg_lon
        self.rbeg_ele = nonignored_observations[first_begin].rbeg_ele
        self.rbeg_jd = nonignored_observations[first_begin].rbeg_jd


        # Find the lowest ending height
        end_hts = [obs.rend_ele for obs in nonignored_observations]
        last_end = end_hts.index(min(end_hts))

        # Set coordinates of the lowest point as the last point
        self.rend_lat = nonignored_observations[last_end].rend_lat
        self.rend_lon = nonignored_observations[last_end].rend_lon
        self.rend_ele = nonignored_observations[last_end].rend_ele
        self.rend_jd = nonignored_observations[last_end].rend_jd



    def calcECIEqAltAz(self, state_vect, radiant_eci, observations):
        """ Calculate ECI coordinates of both CPAs (observed and radiant), equatorial and alt-az coordinates 
            of CPA positions on the radiant line. 

        Arguments:
            state_vect: [ndarray] (x, y, z) ECI coordinates of the initial state vector (meters).
            radiant_eci: [ndarray] (x, y, z) components of the unit radiant direction vector.
            observations: [list] A list of ObservationPoints objects which hold measurements from individual
                stations.

        """

        # Determine the first time of observation
        t0 = min([obs.time_data[0] for obs in observations])

        # Go through observations from all stations
        for obs in observations:

            # Init array for modelled ECI positons
            obs.model_eci = []

            # Init array for modelled RA, Dec positions
            obs.model_ra = np.zeros_like(obs.time_data)
            obs.model_dec = np.zeros_like(obs.time_data)

            # Init arrays for modelled alt, az
            obs.model_azim = np.zeros_like(obs.time_data)
            obs.model_elev = np.zeros_like(obs.time_data)


            # Go through all individual position measurement from each site
            for i, (t, jd, stat, meas) in enumerate(zip(obs.time_data, obs.JD_data, obs.stat_eci_los, \
                obs.meas_eci_los)):

                # Calculate closest points of approach (observed line of sight to radiant line)
                obs_cpa, rad_cpa, d = findClosestPoints(stat, meas, state_vect, radiant_eci)


                # # Calculate the time in seconds from the beginning of the meteor
                # t_rel = t - t0

                # # Calculate the gravitational acceleration at the given height
                # g = G*EARTH.MASS/(vectMag(rad_cpa)**2)

                # # Determine the sign of the initial time
                # time_sign = np.sign(t_rel)

                # # Calculate the amount of gravity drop from a straight trajectory (handle the case when the
                # #   time is negative)
                # drop = time_sign*(1/2.0)*g*t_rel**2

                # # Apply gravity drop to ECI coordinates
                # rad_cpa -= drop*vectNorm(rad_cpa)


                # Set the ECI position of the CPA on the radiant line, as seen by this observer
                obs.model_eci.append(rad_cpa)

                # Calculate the right ascension and declination of the modelled point from the observer's 
                # point of view
                stat_rad_eci = rad_cpa - stat
                model_ra, model_dec = eci2RaDec(stat_rad_eci)

                obs.model_ra[i] = model_ra
                obs.model_dec[i] = model_dec

                # Calculate the azimuth and elevation of the modelled point from the observer's point of view
                model_azim, model_elev = raDec2AltAz(model_ra, model_dec, obs.JD_data[i], obs.lat, obs.lon)

                obs.model_azim[i] = model_azim
                obs.model_elev[i] = model_elev


            obs.model_eci = np.array(obs.model_eci)


            ### Assign model_fit1, model_fit2, so they are in the same format as the input meas1, meas2 data
            ######################################################################################################

            # If inputs were RA and Dec
            if self.meastype == 1:

                obs.model_fit1 = obs.model_ra
                obs.model_fit2 = obs.model_dec

            # If inputs were azimuth +east of due north, and elevation angle
            elif self.meastype == 2:

                obs.model_fit1 = obs.model_azim
                obs.model_fit2 = obs.model_elev

            # If inputs were azimuth +west of due south, and zenith angle
            elif self.meastype == 3:

                obs.model_fit1 = (obs.model_azim + np.pi)%(2*np.pi)
                obs.model_fit2 = np.pi/2.0 - obs.model_elev

            # If input were azimuth +north of due east, and zenith angle
            elif self.meastype == 4:

                obs.model_fit1 = (np.pi/2.0 - obs.model_azim)%(2*np.pi)
                obs.model_fit2 = np.pi/2.0 - obs.model_elev

            ######################################################################################################



    def calcAverages(self, observations):
        """ Calculate the average velocity, the average ECI position of the trajectory and the average 
            Julian date of the trajectory.

        Arguments:
            observations: [list] A list of ObservationPoints objects which hold measurements from individual
                stations.

        Return:
            (v_avg, eci_avg, jd_avg): [tuple]
                v_avg: [float] Average velocity of the meteor in m/s.
                eci_avg: [ndarray] (x, y, z) ECI coordinates of the average point on the trajectory (meters).
                jd_avg: [float] Julian date of the average time of the trajectory.

        """

        v_sum = 0
        eci_sum = np.zeros(3)
        jd_min = np.inf
        jd_max = -np.inf
        meas_sum = 0

        count = 0

        # Go through all observations
        for obs in observations:

            # Skip ignored stations
            if obs.ignore_station:
                continue

            # Calculate the average velocity, ignoring ignored points
            meteor_duration = obs.time_data[obs.ignore_list == 0][-1] - obs.time_data[obs.ignore_list == 0][0]
            meteor_length = vectMag(obs.model_eci[obs.ignore_list == 0][-1] \
                - obs.model_eci[obs.ignore_list == 0][0])

            # Calculate the average velocity
            v_avg = meteor_length/meteor_duration

            v_sum += v_avg
            eci_sum += np.sum(obs.model_eci[obs.ignore_list == 0], axis=0)
            
            jd_min = min(jd_min, np.min(obs.JD_data[obs.ignore_list == 0]))
            jd_max = max(jd_max, np.max(obs.JD_data[obs.ignore_list == 0]))

            # Add in the total number of used points
            meas_sum += len(obs.time_data[obs.ignore_list == 0])

            count += 1


        # Average velocity across all stations
        v_avg = v_sum/count

        # Average ECI across all stations
        eci_avg = eci_sum/meas_sum

        # Average Julian date
        jd_avg = (jd_min + jd_max)/2

        
        return v_avg, eci_avg, jd_avg



    def calcAbsMagnitudes(self):
        """ Compute absolute magnitudes (apparent magnitude at 100 km) after trajectory estimation. """

        # Go through observations from all stations
        for obs in self.observations:

            # Check if the apparent magnitudes were given
            if obs.magnitudes is not None:

                abs_magnitudes = []
                for i, app_mag in enumerate(obs.magnitudes):

                    if app_mag is not None:
                        
                        # Compute absolute magntiude (apparent magnitude at 100 km)
                        abs_mag = app_mag + 5*np.log10(100000/obs.model_range[i])

                    else:
                        abs_mag = None

                    abs_magnitudes.append(abs_mag)


                obs.absolute_magnitudes = np.array(abs_magnitudes)


            else:
                obs.absolute_magnitudes = None



    def dumpMeasurements(self, dir_path, file_name):
        """ Writes the initialized measurements in a MATLAB format text file."""

        with open(os.path.join(dir_path, file_name), 'w') as f:

            for i, obs in enumerate(sorted(self.observations, key=lambda x:x.rbeg_ele, reverse=True)):

                # Write site coordinates
                f.write('m->longitude[' + str(i) + '] = ' + str(obs.lon) + ';\n')
                f.write('m->latitude[' + str(i) + '] = ' + str(obs.lat) + ';\n')
                f.write('m->heightkm[' + str(i) + '] = ' + str(obs.ele/1000) + ';\n\n')

                # Construct an measurement matrix (time, elevation, azimuth) - meastype 2
                meas_matr = np.c_[obs.time_data, np.degrees(obs.elev_data), np.degrees(obs.azim_data)]

                f.write('double ' + chr(97 + i) + '[' + str(len(meas_matr)) + '][3] = {\n')
                for j, row in enumerate(meas_matr):
                    suffix = ','
                    if j == len(meas_matr) - 1:
                        suffix = '};\n'
                    
                    f.write(', '.join(row.astype(str)) + suffix + '\n')


            yyyy, MM, DD, hh, mm, ss, ms = jd2Date(self.jdt_ref)
            ss = ss + ms/1000

            date_formatted = ', '.join(map(str, [yyyy, MM, DD, hh, mm, ss]))
            f.write('m->jdt_ref = JulianDate( ' + date_formatted + ');\n')
        
        print('Measurements dumped into ', os.path.join(dir_path, file_name))



    def toJson(self):
        """ Convert the Trajectory object to a JSON string. """

        # Get a list of builtin types
        try :
            import __builtin__
            builtin_types = [t for t in __builtin__.__dict__.itervalues() if isinstance(t, type)]
        except: 
            # Python 3.x
            import builtins
            builtin_types = [getattr(builtins, d) for d in dir(builtins) if isinstance(getattr(builtins, d), type)]
            


        def _convertDict(d):
            """ Convert the given object dictionary to JSON-compatible format. """

            d = copy.deepcopy(d)

            d_new = {}

            for key in d:

                # Set the old value to the new dictionary
                d_new[key] = d[key]

                # Recursively convert all dictionaries
                if isinstance(d[key], dict):
                    d_new[key] = _convertDict(d[key])

                # Recursively convert items in lists
                if isinstance(d[key], list):

                    # Skip empty lists
                    if len(d[key]) == 0:
                        continue

                    # Remove the old list
                    del d_new[key]

                    # Convert the list to a dictionary
                    d_tmp = {i: item for (i, item) in enumerate(d[key])}

                    # Run the convert procedure
                    d_tmp = _convertDict(d_tmp)

                    # Unpack the dictionary to a list
                    index_list = []
                    value_list = []
                    for k in d_tmp:
                        index_list.append(k)
                        value_list.append(d_tmp[k])

                    # Sort value list by index
                    value_list = [x for _, x in sorted(zip(index_list, value_list))]

                    d_new[key] = value_list


                # Skip None types
                elif d[key] is None:
                    continue

                # Convert datetime objects to strings
                elif isinstance(d[key], datetime.datetime):
                    d_new[key] = str(d[key])

                # Convert numpy arrays to lists
                elif isinstance(d[key], np.ndarray):
                    d_new[key] = d[key].tolist()

                # Convert numpy types to float
                elif type(d[key]).__module__ == np.__name__:
                    d_new[key] = float(d[key])


                # Recursively convert all non-builtin types
                elif type(d[key]) not in builtin_types:

                    # Get the name of the class
                    class_name = type(d[key]).__name__
                    key_name = class_name

                    # Handle class-specific things
                    if class_name == "ObservedPoints":
                        key_name += "." + d[key].station_id

                    elif class_name == "PlaneIntersection":
                        key_name += "." + d[key].obs1.station_id + "_" + d[key].obs2.station_id
                        del d[key].obs1
                        del d[key].obs2

                    # Remove a list of trajectoryes in the uncertainties object
                    elif class_name == "MCUncertainties":
                        d[key].mc_traj_list = None


                    # Assign the converted dictionary to the given attribute name
                    del d_new[key]
                    d_new[key] = {key_name: _convertDict(d[key].__dict__)}
                    


            return d_new



        traj = copy.deepcopy(self)

        # Remove noise-added observations
        if hasattr(traj, "obs_noisy"):
            del traj.obs_noisy

        # Delete duplicate misspelt attribute
        if hasattr(traj, "uncertanties"):
            del traj.uncertanties

        # Convert the trajectory object's attributes to JSON-compatible format
        traj_dict = _convertDict(traj.__dict__)

        # Convert the trajectory object to JSON
        out_str = json.dumps(traj_dict, indent=4, sort_keys=False)


        return out_str


    def saveReport(self, dir_path, file_name, uncertainties=None, verbose=True, save_results=True):
        """ Save the trajectory estimation report to file. 
    
        Arguments:
            dir_path: [str] Path to the directory where the report will be saved.
            file_name: [str] Name of the report time.

        Keyword arguments:
            uncertainties: [MCUncertainties object] Object contaning uncertainties of every parameter.
            verbose: [bool] Print the report to the screen. True by default.
            save_results: [bool] If True, the results will be saved to a file.
        """


        def _uncer(str_format, std_name, multi=1.0, deg=False):
            """ Internal function. Returns the formatted uncertanty, if the uncertanty is given. If not,
                it returns nothing. 

            Arguments:
                str_format: [str] String format for the unceertanty.
                std_name: [str] Name of the uncertanty attribute, e.g. if it is 'x', then the uncertanty is 
                    stored in uncertainties.x.
        
            Keyword arguments:
                multi: [float] Uncertanty multiplier. 1.0 by default. This is used to scale the uncertanty to
                    different units (e.g. from m/s to km/s).
                deg: [bool] Converet radians to degrees if True. False by default.
                """

            if deg:
                multi *= np.degrees(1.0)

            if uncertainties is not None:

                # Construct symmetrical 1 sigma uncertainty
                ret_str = " +/- " + str_format.format(getattr(uncertainties, std_name)*multi)

                # Add confidence interval if available
                if hasattr(uncertainties, std_name + "_ci"):
                    ci_l, ci_u = np.array(getattr(uncertainties, std_name + "_ci"))*multi
                    ret_str += ", [{:s}, {:s}]".format(str_format.format(ci_l), str_format.format(ci_u))

                return ret_str

            else:
                return ''


        # Format longitude in the -180 to 180 deg range
        _formatLongitude = lambda x: (x + np.pi)%(2*np.pi) - np.pi

        
        out_str = ''

        out_str += 'Input measurement type: '

        # Write out measurement type
        if self.meastype == 1:
            out_str += 'Right Ascension for meas1, Declination for meas2, epoch of date\n'

        elif self.meastype == 2:
            out_str += 'Azimuth +east of due north for meas1, Elevation angle above the horizon for meas2\n'

        elif self.meastype == 3:
            out_str += 'Azimuth +west of due south for meas1, Zenith angle for meas2\n'

        elif self.meastype == 4:
            out_str += 'Azimuth +north of due east for meas1, Zenith angle for meas2\n'
        
        out_str += "\n"



        # Write the uncertainty type
        if self.geometric_uncert:
            uncert_label = "Purely geometric uncertainties"

        else:
            uncert_label = "Uncertainties computed using MC runs with lower cost function value than the purely geometric solution"

        if self.state_vect_cov is not None:
            out_str += 'Uncertainties type:\n'
            out_str += ' {:s}'.format(uncert_label)
            out_str += '\n\n'



        out_str += "Reference JD: {:20.12f}\n".format(self.jdt_ref)
        out_str += "Time: " + str(jd2Date(self.orbit.jd_ref, dt_obj=True)) + " UTC\n"

        out_str += "\n\n"

        out_str += 'Plane intersections\n'
        out_str += '-------------------\n'

        # Write out all intersecting planes pairs
        for n, plane_intersection in enumerate(self.intersection_list):

            n = n + 1

            out_str += 'Intersection ' + str(n) + ' - Stations: ' + str(plane_intersection.obs1.station_id) +\
                ' and ' + str(plane_intersection.obs2.station_id) + '\n'

            out_str += ' Convergence angle = {:.5f} deg\n'.format(np.degrees(plane_intersection.conv_angle))
            
            ra, dec = plane_intersection.radiant_eq
            out_str += ' R.A. = {:>9.5f}  Dec = {:>+9.5f} deg\n'.format(np.degrees(ra), np.degrees(dec))


        out_str += '\nBest intersection: Stations ' + str(self.best_conv_inter.obs1.station_id) + ' and ' \
            + str(self.best_conv_inter.obs2.station_id) \
            + ' with Qconv = {:.2f} deg\n'.format(np.degrees(self.best_conv_inter.conv_angle))

        out_str += '\n\n'

        out_str += 'Least squares solution\n'
        out_str += '----------------------\n'

        # Calculate the state vector components
        x, y, z = self.state_vect_mini
        vx, vy, vz = self.v_init*self.radiant_eci_mini

        # Write out the state vector
        out_str += "State vector (ECI, epoch of date):\n"
        out_str += " X =  {:s} m\n".format(valueFormat("{:11.2f}", x, '{:7.2f}', uncertainties, 'x'))
        out_str += " Y =  {:s} m\n".format(valueFormat("{:11.2f}", y, '{:7.2f}', uncertainties, 'y'))
        out_str += " Z =  {:s} m\n".format(valueFormat("{:11.2f}", z, '{:7.2f}', uncertainties, 'z'))
        out_str += " Vx = {:s} m/s\n".format(valueFormat("{:11.2f}", vx, '{:7.2f}', uncertainties, 'vx'))
        out_str += " Vy = {:s} m/s\n".format(valueFormat("{:11.2f}", vy, '{:7.2f}', uncertainties, 'vy'))
        out_str += " Vz = {:s} m/s\n".format(valueFormat("{:11.2f}", vz, '{:7.2f}', uncertainties, 'vz'))

        out_str += "\n"


        # Write out the state vector covariance matrix
        if self.state_vect_cov is not None:

            out_str += "State vector covariance matrix (X, Y, Z, Vx, Vy, Vz):\n"

            for line in self.state_vect_cov:
                line_list = []

                for entry in line:
                    line_list.append("{:+.6e}".format(entry))
                
                out_str += ", ".join(line_list) + "\n"

            out_str += "\n"


        out_str += "Timing offsets (from input data):\n"
        for stat_id, t_diff in zip([obs.station_id for obs in self.observations], self.time_diffs_final):
            out_str += "{:>14s}: {:.6f} s\n".format(str(stat_id), t_diff)

        out_str += "\n"

        if self.orbit is not None:
            out_str += "Reference point on the trajectory:\n"
            out_str += "  Time: " + str(jd2Date(self.orbit.jd_ref, dt_obj=True)) + " UTC\n"
            out_str += "  Lat     = {:s} deg\n".format(valueFormat("{:>11.6f}", self.orbit.lat_ref, \
                '{:6.4f}', uncertainties, 'lat_ref', deg=True))
            out_str += "  Lon     = {:s} deg\n".format(valueFormat("{:>11.6f}", self.orbit.lon_ref, \
                '{:6.4f}', uncertainties, 'lon_ref', deg=True, callable_val=_formatLongitude, \
                callable_ci=_formatLongitude))
            out_str += "  Ht      = {:s} m\n".format(valueFormat("{:>11.2f}", self.orbit.ht_ref, \
                '{:6.2f}', uncertainties, 'ht_ref', deg=False))
            out_str += "  Lat geo = {:s} deg\n".format(valueFormat("{:>11.6f}", self.orbit.lat_geocentric, \
                '{:6.4f}', uncertainties, 'lat_geocentric', deg=True))
            out_str += "\n"

            # Write out orbital parameters
            out_str += self.orbit.__repr__(uncertainties=uncertainties, v_init_ht=self.v_init_ht)
            out_str += "\n"


            # Write out the orbital covariance matrix
            if self.state_vect_cov is not None:

                out_str += "Orbit covariance matrix:\n"
                out_str += "             e     ,     q (AU)   ,      Tp (JD) ,   node (deg) ,   peri (deg) ,    i (deg)\n"

                elements_list = ["e   ", "q   ", "Tp  ", "node", "peri", "i   "]

                for elem_name, line in zip(elements_list, self.orbit_cov):
                    line_list = [elem_name]

                    for entry in line:
                        line_list.append("{:+.6e}".format(entry))
                    
                    out_str += ", ".join(line_list) + "\n"

                out_str += "\n"


        out_str += "Jacchia fit on lag = -|a1|*exp(|a2|*t):\n"
        jacchia_fit = self.jacchia_fit
        if jacchia_fit is None:
            jacchia_fit = [0, 0]
        out_str += " a1 = {:.6f}\n".format(jacchia_fit[0])
        out_str += " a2 = {:.6f}\n".format(jacchia_fit[1])
        out_str += "\n"

        if self.estimate_timing_vel is True:
            out_str += "Mean time residuals from time vs. length:\n"
            out_str += "  Station with reference time: {:s}\n".format(
                str(self.observations[self.t_ref_station].station_id))
            out_str += "  Avg. res. = {:.3e} s\n".format(self.timing_res)
            out_str += "  Stddev    = {:.2e} s\n".format(self.timing_stddev)
            out_str += "\n"

        out_str += "Begin point on the trajectory:\n"
        out_str += "  Lat = {:s} deg\n".format(valueFormat("{:>11.6f}", self.rbeg_lat, "{:6.4f}", \
            uncertainties, 'rbeg_lat', deg=True))
        if uncertainties is not None:
            if hasattr(uncertainties, "rbeg_lat_m"):
                out_str += "                    +/- {:6.2f} m\n".format(uncertainties.rbeg_lat_m)
        out_str += "  Lon = {:s} deg\n".format(valueFormat("{:>11.6f}", self.rbeg_lon, "{:6.4f}", \
            uncertainties, 'rbeg_lon', deg=True, callable_val=_formatLongitude, callable_ci=_formatLongitude))
        if uncertainties is not None:
            if hasattr(uncertainties, "rbeg_lon_m"):
                out_str += "                    +/- {:6.2f} m\n".format(uncertainties.rbeg_lon_m)
        out_str += "  Ht  = {:s} m\n".format(valueFormat("{:>11.2f}", self.rbeg_ele, "{:6.2f}", \
            uncertainties, 'rbeg_ele'))

        out_str += "End point on the trajectory:\n"
        out_str += "  Lat = {:s} deg\n".format(valueFormat("{:>11.6f}", self.rend_lat, "{:6.4f}", \
            uncertainties, 'rend_lat', deg=True))
        if uncertainties is not None:
            if hasattr(uncertainties, "rend_lat_m"):
                out_str += "                    +/- {:6.2f} m\n".format(uncertainties.rend_lat_m)
        out_str += "  Lon = {:s} deg\n".format(valueFormat("{:>11.6f}", self.rend_lon, "{:6.4f}", \
            uncertainties, 'rend_lon', deg=True, callable_val=_formatLongitude, callable_ci=_formatLongitude))
        if uncertainties is not None:
            if hasattr(uncertainties, "rend_lon_m"):
                out_str += "                    +/- {:6.2f} m\n".format(uncertainties.rend_lon_m)
        out_str += "  Ht  = {:s} m\n".format(valueFormat("{:>11.2f}", self.rend_ele, "{:6.2f}", \
            uncertainties, 'rend_ele'))
        out_str += "\n"

        ### Write information about stations ###
        ######################################################################################################
        out_str += "Stations\n"
        out_str += "--------\n"

        out_str += "            ID, Ignored, Lon +E (deg), Lat +N (deg),  Ht (m), Jacchia a1, Jacchia a2,  Beg Ht (m),  End Ht (m), +/- Obs ang (deg), +/- V (m), +/- H (m), Persp. angle (deg), Weight, FOV Beg, FOV End, Comment\n"
        
        for obs in self.observations:

            station_info = []
            station_info.append("{:>14s}".format(str(obs.station_id)))
            station_info.append("{:>7s}".format(str(obs.ignore_station)))
            station_info.append("{:>12.6f}".format(np.degrees(obs.lon)))
            station_info.append("{:>12.6f}".format(np.degrees(obs.lat)))
            station_info.append("{:>7.2f}".format(obs.ele))
            jacchia_fit = obs.jacchia_fit
            if jacchia_fit is None:
                jacchia_fit = [0, 0]
            station_info.append("{:>10.6f}".format(jacchia_fit[0]))
            station_info.append("{:>10.6f}".format(jacchia_fit[1]))
            station_info.append("{:>11.2f}".format(obs.rbeg_ele))
            station_info.append("{:>11.2f}".format(obs.rend_ele))
            station_info.append("{:>17.6f}".format(np.degrees(obs.ang_res_std)))
            station_info.append("{:>9.2f}".format(obs.v_res_rms))
            station_info.append("{:>9.2f}".format(obs.h_res_rms))
            station_info.append("{:>18.2f}".format(np.degrees(obs.incident_angle)))

            if obs.weight is not None:
                station_info.append("{:>6.4f}".format(obs.weight))
            else:
                station_info.append("{:>6s}".format('None'))

            station_info.append("{:>7s}".format(str(obs.fov_beg)))
            station_info.append("{:>7s}".format(str(obs.fov_end)))
            station_info.append("{:s}".format(str(obs.comment)))



            out_str += ", ".join(station_info) + "\n"
        
        ######################################################################################################

        out_str += "\n"

        ### Write information about individual points ###
        ######################################################################################################
        out_str += "Points\n"
        out_str += "------\n"


        out_str += " No, "
        out_str += "    Station ID, "
        out_str += " Ignore, "
        out_str += " Time (s), "
        out_str += "                  JD, "
        out_str += "    meas1, "
        out_str += "    meas2, "
        out_str += "Azim +E of due N (deg), "
        out_str += "Alt (deg), "
        out_str += "Azim line (deg), "
        out_str += "Alt line (deg), "
        out_str += "RA obs (deg), "
        out_str += "Dec obs (deg), "
        out_str += "RA line (deg), "
        out_str += "Dec line (deg), "
        out_str += "      X (m), "
        out_str += "      Y (m), "
        out_str += "      Z (m), "
        out_str += "Latitude (deg), "
        out_str += "Longitude (deg), "
        out_str += "Height (m), "
        out_str += " Range (m), "
        out_str += "Length (m), "
        out_str += "State vect dist (m), "
        out_str += "  Lag (m), "
        out_str += "Vel (m/s), "
        out_str += "Vel prev avg (m/s), "
        out_str += "H res (m), "
        out_str += "V res (m), "
        out_str += "Ang res (asec), "
        out_str += "AppMag, "
        out_str += "AbsMag"
        out_str += "\n"

        # Go through observation from all stations
        for obs in self.observations:

            # Go through all observed points
            for i in range(obs.kmeas):

                point_info = []

                point_info.append("{:3d}".format(i))

                point_info.append("{:>14s}".format(str(obs.station_id)))

                point_info.append("{:>7d}".format(obs.ignore_list[i]))
                
                point_info.append("{:9.6f}".format(obs.time_data[i]))
                point_info.append("{:20.12f}".format(obs.JD_data[i]))

                point_info.append("{:9.5f}".format(np.degrees(obs.meas1[i])))
                point_info.append("{:9.5f}".format(np.degrees(obs.meas2[i])))

                point_info.append("{:22.5f}".format(np.degrees(obs.azim_data[i])))
                point_info.append("{:9.5f}".format(np.degrees(obs.elev_data[i])))

                point_info.append("{:15.5f}".format(np.degrees(obs.model_azim[i])))
                point_info.append("{:14.5f}".format(np.degrees(obs.model_elev[i])))

                point_info.append("{:12.5f}".format(np.degrees(obs.ra_data[i])))
                point_info.append("{:+13.5f}".format(np.degrees(obs.dec_data[i])))

                point_info.append("{:13.5f}".format(np.degrees(obs.model_ra[i])))
                point_info.append("{:+14.5f}".format(np.degrees(obs.model_dec[i])))

                point_info.append("{:11.2f}".format(obs.model_eci[i][0]))
                point_info.append("{:11.2f}".format(obs.model_eci[i][1]))
                point_info.append("{:11.2f}".format(obs.model_eci[i][2]))

                point_info.append("{:14.6f}".format(np.degrees(obs.model_lat[i])))
                point_info.append("{:+15.6f}".format(np.degrees(obs.model_lon[i])))
                point_info.append("{:10.2f}".format(obs.model_ht[i]))
                point_info.append("{:10.2f}".format(obs.model_range[i]))

                point_info.append("{:10.2f}".format(obs.length[i]))
                point_info.append("{:19.2f}".format(obs.state_vect_dist[i]))
                point_info.append("{:9.2f}".format(obs.lag[i]))

                point_info.append("{:9.2f}".format(obs.velocities[i]))
                point_info.append("{:18.2f}".format(obs.velocities_prev_point[i]))

                point_info.append("{:9.2f}".format(obs.h_residuals[i]))
                point_info.append("{:9.2f}".format(obs.v_residuals[i]))
                point_info.append("{:14.2f}".format(3600*np.degrees(obs.ang_res[i])))

                if obs.magnitudes is not None:

                    # Write the magnitude
                    if obs.magnitudes[i] is not None:
                        point_info.append("{:+6.2f}".format(obs.magnitudes[i]))
                    else:
                        point_info.append("{:>6s}".format('None'))

                    # Write the magnitude
                    if obs.absolute_magnitudes[i] is not None:
                        point_info.append("{:+6.2f}".format(obs.absolute_magnitudes[i]))
                    else:
                        point_info.append("{:>6s}".format('None'))

                else:
                    point_info.append("{:>6s}".format('None'))
                    point_info.append("{:>6s}".format('None'))





                out_str += ", ".join(point_info) + "\n"


        ######################################################################################################


        out_str += "\n"

        out_str += "Notes\n"
        out_str += "-----\n"
        out_str += "- Points that have not been taken into consideration when computing the trajectory have '1' in the 'Ignore' column.\n"
        out_str += "- The time already has time offsets applied to it.\n"
        out_str += "- 'meas1' and 'meas2' are given input points.\n"
        out_str += "- X, Y, Z are ECI (Earth-Centered Inertial) positions of projected lines of sight on the radiant line.\n"
        out_str += "- Zc is the observed zenith distance of the entry angle, while the Zg is the entry zenith distance corrected for Earth's gravity.\n"
        out_str += "- Latitude (deg) and Longitude (deg) are in WGS84 coordinates, while Height (m) is in the EGM96 datum. There values are coordinates of each point on the radiant line.\n"
        out_str += "- Jacchia (1955) deceleration equation fit was done on the lag.\n"
        out_str += "- Right ascension and declination in the table are given in the epoch of date for the corresponding JD, per every point.\n"
        out_str += "- 'RA and Dec obs' are the right ascension and declination calculated from the observed values, while the 'RA and Dec line' are coordinates of the lines of sight projected on the fitted radiant line. The coordinates are in the epoch of date, and NOT J2000!. 'Azim and alt line' are thus corresponding azimuthal coordinates.\n"
        out_str += "- 'Vel prev avg' is the average velocity including all previous points up to the given point. For the first 4 points this velocity is computed as the average velocity of those 4 points. \n"
        if uncertainties is not None:
            out_str += "- The number after +/- is the 1 sigma uncertainty, and the numbers in square brackets are the 95% confidence interval \n"

        if verbose:
            print(out_str)

        # Save the report to a file
        if save_results:
            
            mkdirP(dir_path)

            with open(os.path.join(dir_path, file_name), 'w') as f:
                f.write(out_str)


        return out_str



    def savePlots(self, output_dir, file_name, show_plots=True, ret_figs=False):
        """ Show plots of the estimated trajectory. 
    
        Arguments:
            output_dir: [str] Path to the output directory.
            file_name: [str] File name which will be used for saving plots.

        Keyword_arguments:
            show_plots: [bools] Show the plots on the screen. True by default.
            ret_figs: [bool] If True, it will return a dictionary of figure handles for every plot. It will
                override the show_plots and set them to False, and it will not save any plots.

        Return:
            fig_pickle_dict: [dict] Dictionary of pickled figure handles for every plot. To unpickle the
                figure objects, run:
                    fig = pickle.loads(fig_pickle_dict[key])
                where key is the dictionary key, e.g. "lags_all".

        """

        if output_dir is None:
            output_dir = '.'

        if file_name is None:
            file_name = 'blank'


        # Dictionary which will hold figure handles for every plot
        fig_pickle_dict = {}

        # Override the status of saving commands if the figures should be returned
        save_results_prev_status = self.save_results
        if ret_figs:
            self.save_results = False
            show_plots = False

            

        # Get the first reference time
        t0 = min([obs.time_data[0] for obs in self.observations])

        # Plot spatial residuals per observing station
        for obs in sorted(self.observations, key=lambda x: x.rbeg_ele, reverse=True):

            ### PLOT SPATIAL RESIDUALS PER STATION ###
            ##################################################################################################

            # Plot vertical residuals
            plt.scatter(obs.time_data, obs.v_residuals, c='r', \
                label='Vertical, RMSD = {:.2f} m'.format(obs.v_res_rms), zorder=3, s=4, marker='o')

            # Plot horizontal residuals
            plt.scatter(obs.time_data, obs.h_residuals, c='b', \
                label='Horizontal, RMSD = {:.2f} m'.format(obs.h_res_rms), zorder=3, s=20, marker='+')

            # Mark ignored points
            if np.any(obs.ignore_list):

                ignored_times = obs.time_data[obs.ignore_list > 0]
                ignored_v_res = obs.v_residuals[obs.ignore_list > 0]
                ignored_h_res = obs.h_residuals[obs.ignore_list > 0]

                plt.scatter(ignored_times, ignored_v_res, facecolors='none', edgecolors='k', marker='o', \
                    zorder=3, s=20, label='Ignored points')
                plt.scatter(ignored_times, ignored_h_res, facecolors='none', edgecolors='k', marker='o', 
                    zorder=3, s=20)


            plt.title('Residuals, station ' + str(obs.station_id))
            plt.xlabel('Time (s)')
            plt.ylabel('Residuals (m)')

            plt.grid()

            plt.legend(prop={'size': LEGEND_TEXT_SIZE})

            # Set the residual limits to +/-10m if they are smaller than that
            if (np.max(np.abs(obs.v_residuals)) < 10) and (np.max(np.abs(obs.h_residuals)) < 10):
                plt.ylim([-10, 10])


            # Pickle the figure
            if ret_figs:
                fig_pickle_dict["spatial_residuals_{:s}".format(str(obs.station_id))] \
                    = pickle.dumps(plt.gcf(), protocol=2)


            if self.save_results:
                savePlot(plt, file_name + '_' + str(obs.station_id) + '_spatial_residuals.' \
                    + self.plot_file_type, output_dir)

            if show_plots:
                plt.show()

            else:
                plt.clf()
                plt.close()

            ##################################################################################################


        # marker type, size multiplier
        markers = [
         ['x', 2 ],
         ['+', 8 ],
         ['o', 1 ],
         ['s', 1 ],
         ['d', 1 ],
         ['v', 1 ],
         ['*', 1.5 ],
         ]
         
        if self.plot_all_spatial_residuals:


            ### PLOT ALL SPATIAL RESIDUALS VS. TIME ###
            ##################################################################################################

            for obs in sorted(self.observations, key=lambda x: x.rbeg_ele, reverse=True):

                # Plot vertical residuals
                vres_plot = plt.scatter(obs.time_data, obs.v_residuals, marker='o', s=4, \
                    label='{:s}, vertical, RMSD = {:.2f} m'.format(str(obs.station_id), obs.v_res_rms), \
                    zorder=3)

                # Plot horizontal residuals
                plt.scatter(obs.time_data, obs.h_residuals, c=vres_plot.get_facecolor(), marker='+', \
                    label='{:s}, horizontal, RMSD = {:.2f} m'.format(str(obs.station_id), obs.h_res_rms), \
                    zorder=3)

                # Mark ignored points
                if np.any(obs.ignore_list):

                    ignored_times = obs.time_data[obs.ignore_list > 0]
                    ignored_v_res = obs.v_residuals[obs.ignore_list > 0]
                    ignored_h_res = obs.h_residuals[obs.ignore_list > 0]

                    plt.scatter(ignored_times, ignored_v_res, facecolors='none', edgecolors='k', marker='o', \
                        zorder=3, s=20)
                    plt.scatter(ignored_times, ignored_h_res, facecolors='none', edgecolors='k', marker='o', 
                        zorder=3, s=20)


            plt.title('All spatial residuals')
            plt.xlabel('Time (s)')
            plt.ylabel('Residuals (m)')

            plt.grid()

            plt.legend(prop={'size': LEGEND_TEXT_SIZE})

            # Set the residual limits to +/-10m if they are smaller than that
            if np.max(np.abs(plt.gca().get_ylim())) < 10:
                plt.ylim([-10, 10])

            # Pickle the figure
            if ret_figs:
                fig_pickle_dict["all_spatial_residuals"] = pickle.dumps(plt.gcf(), protocol=2)

            if self.save_results:
                savePlot(plt, file_name + '_all_spatial_residuals.' + self.plot_file_type, output_dir)

            if show_plots:
                plt.show()

            else:
                plt.clf()
                plt.close()

            ##################################################################################################


            ### PLOT ALL SPATIAL RESIDUALS VS LENGTH ###
            ##################################################################################################

            for obs in sorted(self.observations, key=lambda x: x.rbeg_ele, reverse=True):

                # Plot vertical residuals
                vres_plot = plt.scatter(obs.state_vect_dist/1000, obs.v_residuals, marker='o', s=4, \
                    label='{:s}, vertical, RMSD = {:.2f} m'.format(str(obs.station_id), obs.v_res_rms), \
                    zorder=3)

                # Plot horizontal residuals
                plt.scatter(obs.state_vect_dist/1000, obs.h_residuals, c=vres_plot.get_facecolor(), 
                    marker='+', label='{:s}, horizontal, RMSD = {:.2f} m'.format(str(obs.station_id), \
                        obs.h_res_rms), zorder=3)

                # Mark ignored points
                if np.any(obs.ignore_list):

                    ignored_length = obs.state_vect_dist[obs.ignore_list > 0]
                    ignored_v_res = obs.v_residuals[obs.ignore_list > 0]
                    ignored_h_res = obs.h_residuals[obs.ignore_list > 0]

                    plt.scatter(ignored_length/1000, ignored_v_res, facecolors='none', edgecolors='k', \
                        marker='o', zorder=3, s=20)
                    plt.scatter(ignored_length/1000, ignored_h_res, facecolors='none', edgecolors='k', \
                        marker='o', zorder=3, s=20)


            plt.title('All spatial residuals')
            plt.xlabel('Length (km)')
            plt.ylabel('Residuals (m)')

            plt.grid()

            plt.legend(prop={'size': LEGEND_TEXT_SIZE})

            # Set the residual limits to +/-10m if they are smaller than that
            if np.max(np.abs(plt.gca().get_ylim())) < 10:
                plt.ylim([-10, 10])

            # Pickle the figure
            if ret_figs:
                fig_pickle_dict["all_spatial_residuals_length"] = pickle.dumps(plt.gcf(), protocol=2)

            if self.save_results:
                savePlot(plt, file_name + '_all_spatial_residuals_length.' + self.plot_file_type, output_dir)

            if show_plots:
                plt.show()

            else:
                plt.clf()
                plt.close()


            ##################################################################################################



            ### PLOT TOTAL SPATIAL RESIDUALS VS LENGTH ###
            ##################################################################################################

            for i, obs in enumerate(sorted(self.observations, key=lambda x:x.rbeg_ele, reverse=True)):

                marker, size_multiplier = markers[i%len(markers)]

                # Compute total residuals, take the signs from vertical residuals
                tot_res = np.sign(obs.v_residuals)*np.hypot(obs.v_residuals, obs.h_residuals)

                # Plot total residuals
                plt.scatter(obs.state_vect_dist/1000, tot_res, marker=marker, s=10*size_multiplier, \
                    label='{:s}'.format(str(obs.station_id)), zorder=3)

                # Mark ignored points
                if np.any(obs.ignore_list):

                    ignored_length = obs.state_vect_dist[obs.ignore_list > 0]
                    ignored_tot_res = tot_res[obs.ignore_list > 0]

                    plt.scatter(ignored_length/1000, ignored_tot_res, facecolors='none', edgecolors='k', \
                        marker='o', zorder=3, s=20)


            plt.title('Total spatial residuals')
            plt.xlabel('Length (km)')
            plt.ylabel('Residuals (m), vertical sign')

            plt.grid()

            plt.legend(prop={'size': LEGEND_TEXT_SIZE})

            # Set the residual limits to +/-10m if they are smaller than that
            if np.max(np.abs(plt.gca().get_ylim())) < 10:
                plt.ylim([-10, 10])

            # Pickle the figure
            if ret_figs:
                fig_pickle_dict["total_spatial_residuals_length"] = pickle.dumps(plt.gcf(), protocol=2)

            if self.save_results:
                savePlot(plt, file_name + '_total_spatial_residuals_length.' + self.plot_file_type, \
                    output_dir)

            if show_plots:
                plt.show()

            else:
                plt.clf()
                plt.close()


            ##################################################################################################


            ### PLOT TOTAL SPATIAL RESIDUALS VS LENGTH (with influence of gravity) ###
            ##################################################################################################

            # Plot only with gravity compensation is used
            if self.gravity_correction:

                for i, obs in enumerate(sorted(self.observations, key=lambda x:x.rbeg_ele, reverse=True)):

                    marker, size_multiplier = markers[i%len(markers)]


                    ## Compute residual from gravity corrected point ##

                    res_total_grav_list = []

                    # Go through all individual position measurements from each site
                    #for t, jd, stat, meas in zip(obs.time_data, obs.JD_data, obs.stat_eci_los, obs.meas_eci_los):
                    for jd, tlat, tlon, tht, mlat, mlon, mht in zip(obs.JD_data, obs.model_lat, \
                        obs.model_lon, obs.model_ht, obs.meas_lat, obs.meas_lon, obs.meas_ht):


                        # Compute cartiesian coordinates of trajectory points
                        traj_eci = np.array(geo2Cartesian(tlat, tlon, tht, jd))

                        # Compute cartiesian coordinates of measurement points
                        meas_eci = np.array(geo2Cartesian(mlat, mlon, mht, jd))

                        # Compute the total distance between the points
                        res_total_grav = vectMag(traj_eci - meas_eci)

                        # The sign of the residual is the vertical component (meas higher than trajectory is
                        #   positive)
                        if vectMag(meas_eci) > vectMag(traj_eci):
                            res_total_grav = -res_total_grav


                        res_total_grav_list.append(res_total_grav)
                        

                    res_total_grav_list = np.array(res_total_grav_list)

                    ## ##

                    # Plot total residuals
                    plt.scatter(obs.state_vect_dist/1000, res_total_grav_list, marker=marker, 
                        s=10*size_multiplier, label='{:s}'.format(str(obs.station_id)), zorder=3)

                    # Mark ignored points
                    if np.any(obs.ignore_list):

                        ignored_length = obs.state_vect_dist[obs.ignore_list > 0]
                        ignored_tot_res = res_total_grav_list[obs.ignore_list > 0]

                        plt.scatter(ignored_length/1000, ignored_tot_res, facecolors='none', edgecolors='k', \
                            marker='o', zorder=3, s=20)


                plt.title('Total spatial residuals (gravity corrected)')
                plt.xlabel('Length (km)')
                plt.ylabel('Residuals (m), vertical sign')

                plt.grid()

                plt.legend(prop={'size': LEGEND_TEXT_SIZE})

                # Set the residual limits to +/-10m if they are smaller than that
                if np.max(np.abs(plt.gca().get_ylim())) < 10:
                    plt.ylim([-10, 10])

                # Pickle the figure
                if ret_figs:
                    fig_pickle_dict["total_spatial_residuals_length_grav"] = pickle.dumps(plt.gcf(), \
                        protocol=2)

                if self.save_results:
                    savePlot(plt, file_name + '_total_spatial_residuals_length_grav.' + self.plot_file_type, \
                        output_dir)

                if show_plots:
                    plt.show()

                else:
                    plt.clf()
                    plt.close()


            ##################################################################################################


        ### PLOT ALL TOTAL SPATIAL RESIDUALS VS HEIGHT ###
        ##################################################################################################

        for i, obs in enumerate(sorted(self.observations, key=lambda x:x.rbeg_ele, reverse=True)):

            marker, size_multiplier = markers[i%len(markers)]

            # Calculate root mean square of the total residuals
            total_res_rms = np.sqrt(obs.v_res_rms**2 + obs.h_res_rms**2)

            # Compute total residuals, take the signs from vertical residuals
            tot_res = np.sign(obs.v_residuals)*np.hypot(obs.v_residuals, obs.h_residuals)

            # Plot total residuals
            plt.scatter(tot_res, obs.meas_ht/1000, marker=marker, \
                s=10*size_multiplier, label='{:s}, RMSD = {:.2f} m'.format(str(obs.station_id), \
                total_res_rms), zorder=3)

            # Mark ignored points
            if np.any(obs.ignore_list):

                ignored_ht = obs.model_ht[obs.ignore_list > 0]
                ignored_tot_res = np.sign(obs.v_residuals[obs.ignore_list > 0])\
                    *np.hypot(obs.v_residuals[obs.ignore_list > 0], obs.h_residuals[obs.ignore_list > 0])


                plt.scatter(ignored_tot_res, ignored_ht/1000, facecolors='none', edgecolors='k', \
                    marker='o', zorder=3, s=20)


        plt.title('All spatial residuals')
        plt.xlabel('Total deviation (m)')
        plt.ylabel('Height (km)')

        plt.grid()

        plt.legend(prop={'size': LEGEND_TEXT_SIZE})

        # Set the residual limits to +/-10m if they are smaller than that
        if np.max(np.abs(plt.gca().get_xlim())) < 10:
            plt.gca().set_xlim([-10, 10])

        # Pickle the figure
        if ret_figs:
            fig_pickle_dict["all_spatial_total_residuals_height"] = pickle.dumps(plt.gcf(), protocol=2)

        if self.save_results:
            savePlot(plt, file_name + '_all_spatial_total_residuals_height.' + self.plot_file_type, \
                output_dir)

        if show_plots:
            plt.show()

        else:
            plt.clf()
            plt.close()


        ##################################################################################################




        # # Plot lag per observing station
        # for obs in sorted(self.observations, key=lambda x: x.rbeg_ele, reverse=True):
            
        #     ### PLOT LAG ###
        #     ##################################################################################################

        #     fig, ax1 = plt.subplots()

        #     # Extract lag points that were not ignored
        #     used_times = obs.time_data[obs.ignore_list == 0]
        #     used_lag = obs.lag[obs.ignore_list == 0]

        #     if not obs.ignore_station:

        #         # Plot the lag
        #         ax1.plot(used_lag, used_times, color='r', marker='x', label='Lag', zorder=3)

        #         # Plot the Jacchia fit
        #         ax1.plot(jacchiaLagFunc(obs.time_data, *obs.jacchia_fit), obs.time_data, color='b', 
        #             label='Jacchia fit', zorder=3)


        #     # Plot ignored lag points
        #     if np.any(obs.ignore_list):

        #         ignored_times = obs.time_data[obs.ignore_list > 0]
        #         ignored_lag = obs.lag[obs.ignore_list > 0]

        #         ax1.scatter(ignored_lag, ignored_times, c='k', marker='+', zorder=4, \
        #             label='Lag, ignored points')

            
        #     ax1.legend(prop={'size': LEGEND_TEXT_SIZE})

        #     plt.title('Lag, station ' + str(obs.station_id))
        #     ax1.set_xlabel('Lag (m)')
        #     ax1.set_ylabel('Time (s)')

        #     ax1.set_ylim(min(obs.time_data), max(obs.time_data))

        #     ax1.grid()

        #     ax1.invert_yaxis()

        #     # Set the height axis
        #     ax2 = ax1.twinx()
        #     ax2.set_ylim(min(obs.meas_ht)/1000, max(obs.meas_ht)/1000)
        #     ax2.set_ylabel('Height (km)')

        #     plt.tight_layout()

        #     if self.save_results:
        #         savePlot(plt, file_name + '_' + str(obs.station_id) + '_lag.' + self.plot_file_type, output_dir)

        #     if show_plots:
        #         plt.show()

        #     else:
        #         plt.clf()
        #         plt.close()


        #     ##################################################################################################


        # Generate a list of colors to use for markers
        colors = cm.viridis(np.linspace(0, 0.8, len(self.observations)))

        # Only use one type of markers if there are not a lot of stations
        plot_markers = ['x']

        # Keep colors non-transparent if there are not a lot of stations
        alpha = 1.0


        # If there are more than 5 stations, interleave the colors with another colormap and change up
        #   markers
        if len(self.observations) > 5:
            colors_alt = cm.inferno(np.linspace(0, 1, len(self.observations)))
            for i in range(len(self.observations)):
                if i%2 == 1:
                    colors[i] = colors_alt[i]

            plot_markers.append("+")

            # Add transparency for more stations
            alpha = 0.75


        # Sort observations by first height to preserve color linearity
        obs_ht_sorted = sorted(self.observations, key=lambda x: x.model_ht[0])


        ### PLOT ALL LAGS ###
        ######################################################################################################

        # Plot lags from each station on a single plot
        for i, obs in enumerate(obs_ht_sorted):

            # Extract lag points that were not ignored
            used_times = obs.time_data[obs.ignore_list == 0]
            used_lag = obs.lag[obs.ignore_list == 0]

            # Choose the marker
            marker = plot_markers[i%len(plot_markers)]

            # Plot the lag
            plt_handle = plt.plot(used_lag, used_times, marker=marker, label=str(obs.station_id), 
                zorder=3, markersize=3, color=colors[i], alpha=alpha)


            # Plot ignored lag points
            if np.any(obs.ignore_list):

                ignored_times = obs.time_data[obs.ignore_list > 0]
                ignored_lag = obs.lag[obs.ignore_list > 0]

                plt.scatter(ignored_lag, ignored_times, facecolors='k', edgecolors=plt_handle[0].get_color(), 
                    marker='o', s=8, zorder=4, label='{:s} ignored points'.format(str(obs.station_id)))



        # Plot the Jacchia fit on all observations
        if self.show_jacchia:
            
            time_all = np.sort(np.hstack([obs.time_data for obs in self.observations]))
            time_jacchia = np.linspace(np.min(time_all), np.max(time_all), 1000)
            plt.plot(jacchiaLagFunc(time_jacchia, *self.jacchia_fit), time_jacchia, label='Jacchia fit', 
                zorder=3, color='k', alpha=0.5, linestyle="dashed")


        plt.title('Lags, all stations')

        plt.xlabel('Lag (m)')
        plt.ylabel('Time (s)')

        plt.legend(prop={'size': LEGEND_TEXT_SIZE})
        plt.grid()
        plt.gca().invert_yaxis()

        # Pickle the figure
        if ret_figs:
            fig_pickle_dict["lags_all"] = pickle.dumps(plt.gcf(), protocol=2)

        if self.save_results:
            savePlot(plt, file_name + '_lags_all.' + self.plot_file_type, output_dir)

        if show_plots:
            plt.show()

        else:
            plt.clf()
            plt.close()

        ######################################################################################################



        ### PLOT VELOCITY ###
        ######################################################################################################

        # Possible markers for velocity
        vel_markers = ['x', '+', '.', '2']

        fig, ax1 = plt.subplots()

        vel_max = -np.inf
        vel_min = np.inf

        ht_max = -np.inf
        ht_min = np.inf

        t_max = -np.inf
        t_min = np.inf

        
        first_ignored_plot = True


        # Plot velocities from each observed site
        for i, obs in enumerate(obs_ht_sorted):

            # Mark ignored velocities
            if np.any(obs.ignore_list):

                # Extract data that is not ignored
                ignored_times = obs.time_data[1:][obs.ignore_list[1:] > 0]
                ignored_velocities = obs.velocities[1:][obs.ignore_list[1:] > 0]

                # Set the label only for the first occurence
                if first_ignored_plot:

                    ax1.scatter(ignored_velocities/1000, ignored_times, facecolors='none', edgecolors='k', \
                        zorder=4, s=30, label='Ignored points')

                    first_ignored_plot = False

                else:
                    ax1.scatter(ignored_velocities/1000, ignored_times, facecolors='none', edgecolors='k', \
                        zorder=4, s=30)


            # Plot all point to point velocities
            ax1.scatter(obs.velocities[1:]/1000, obs.time_data[1:], marker=vel_markers[i%len(vel_markers)], 
                c=colors[i].reshape(1,-1), alpha=alpha, label='{:s}'.format(str(obs.station_id)), zorder=3)


            # Determine the max/min velocity and height, as this is needed for plotting both height/time axes
            vel_max = max(np.max(obs.velocities[1:]/1000), vel_max)
            vel_min = min(np.min(obs.velocities[1:]/1000), vel_min)

            ht_max = max(np.max(obs.meas_ht), ht_max)
            ht_min = min(np.min(obs.meas_ht), ht_min)

            t_max = max(np.max(obs.time_data), t_max)
            t_min = min(np.min(obs.time_data), t_min)


        # Plot the velocity calculated from the Jacchia model
        if self.show_jacchia:
            t_vel = np.linspace(t_min, t_max, 1000)
            ax1.plot(jacchiaVelocityFunc(t_vel, self.jacchia_fit[0], self.jacchia_fit[1], self.v_init)/1000, \
                t_vel, label='Jacchia fit', alpha=0.5, color='k')

        plt.title('Velocity')
        ax1.set_xlabel('Velocity (km/s)')
        ax1.set_ylabel('Time (s)')

        ax1.legend(prop={'size': LEGEND_TEXT_SIZE})
        ax1.grid()

        # Set velocity limits to +/- 3 km/s
        ax1.set_xlim([vel_min - 3, vel_max + 3])

        # Set time axis limits
        ax1.set_ylim([t_min, t_max])
        ax1.invert_yaxis()

        # Set the height axis
        ax2 = ax1.twinx()
        ax2.set_ylim(ht_min/1000, ht_max/1000)
        ax2.set_ylabel('Height (km)')

        plt.tight_layout()


        # Pickle the figure
        if ret_figs:
            fig_pickle_dict["velocities"] = pickle.dumps(plt.gcf(), protocol=2)

        if self.save_results:
            savePlot(plt, file_name + '_velocities.' + self.plot_file_type, output_dir)

        if show_plots:
            plt.show()

        else:
            plt.clf()
            plt.close()

        ######################################################################################################


        ### PLOT DISTANCE FROM RADIANT STATE VECTOR POSITION ###
        ######################################################################################################

        fig, ax1 = plt.subplots()

        for i, obs in enumerate(obs_ht_sorted):

            # Extract points that were not ignored
            used_times = obs.time_data[obs.ignore_list == 0]
            used_dists = obs.state_vect_dist[obs.ignore_list == 0]

            # Choose the marker
            marker = plot_markers[i%len(plot_markers)]

            plt_handle = ax1.plot(used_dists/1000, used_times, marker=marker, label=str(obs.station_id), \
                zorder=3, markersize=3, color=colors[i], alpha=alpha)


            # Plot ignored points
            if np.any(obs.ignore_list):
 
                ignored_times = obs.time_data[obs.ignore_list > 0]
                ignored_dists = obs.state_vect_dist[obs.ignore_list > 0]

                ax1.scatter(ignored_dists/1000, ignored_times, facecolors='k', 
                    edgecolors=plt_handle[0].get_color(), marker='o', s=8, zorder=5, \
                    label='{:s} ignored points'.format(str(obs.station_id)))



        # Add the fitted velocity line
        if self.velocity_fit is not None:

            # Get time data range
            t_min = min([np.min(obs.time_data) for obs in self.observations])
            t_max = max([np.max(obs.time_data) for obs in self.observations])

            t_range = np.linspace(t_min, t_max, 100)

            ax1.plot(lineFunc(t_range, *self.velocity_fit)/1000, t_range, label='Velocity fit', \
                linestyle='--', alpha=0.5, zorder=3)


        title = "Distances from state vector"
        if self.estimate_timing_vel:
            stres=self.timing_res
            if stres is None:
                stres=0
            title += ", Time residuals = {:.3e} s".format(stres)

        plt.title(title)

        ax1.set_ylabel('Time (s)')
        ax1.set_xlabel('Distance from state vector (km)')
        
        ax1.legend(prop={'size': LEGEND_TEXT_SIZE})
        ax1.grid()
        
        # Set time axis limits
        ax1.set_ylim([t_min, t_max])
        ax1.invert_yaxis()

        # Set the height axis
        ax2 = ax1.twinx()
        ax2.set_ylim(ht_min/1000, ht_max/1000)
        ax2.set_ylabel('Height (km)')


        # Pickle the figure
        if ret_figs:
            fig_pickle_dict["lengths"] = pickle.dumps(plt.gcf(), protocol=2)

        if self.save_results:
            savePlot(plt, file_name + '_lengths.' + self.plot_file_type, output_dir)


        if show_plots:
            plt.show()

        else:
            plt.clf()
            plt.close()

        ######################################################################################################


        ### Plot lat/lon of the meteor ###
            
        # Calculate mean latitude and longitude of all meteor points
        met_lon_mean = meanAngle([x for x in obs.meas_lon for obs in self.observations])
        met_lat_mean = meanAngle([x for x in obs.meas_lat for obs in self.observations])


        # Put coordinates of all sites and the meteor in the one list
        lat_list = [obs.lat for obs in self.observations]
        lat_list.append(met_lat_mean)
        lon_list = [obs.lon for obs in self.observations]
        lon_list.append(met_lon_mean)

        # Put edge points of the meteor in the list
        lat_list.append(self.rbeg_lat)
        lon_list.append(self.rbeg_lon)
        lat_list.append(self.rend_lat)
        lon_list.append(self.rend_lon)
        lat_list.append(self.orbit.lat_ref)
        lon_list.append(self.orbit.lon_ref)


        # Init the map
        m = GroundMap(lat_list, lon_list, border_size=50, color_scheme='light')


        # Plot locations of all stations and measured positions of the meteor
        for i, obs in enumerate(sorted(self.observations, key=lambda x:x.rbeg_ele, reverse=True)):

            # Extract marker type and size multiplier
            marker, sm = markers[i%len(markers)]

            # Plot stations
            m.scatter(obs.lat, obs.lon, s=sm*10, label=str(obs.station_id), marker=marker)

            # Plot measured points
            m.plot(obs.meas_lat[obs.ignore_list == 0], obs.meas_lon[obs.ignore_list == 0], c='r')

            # Plot ignored points
            if np.any(obs.ignore_list != 0):
                m.scatter(obs.meas_lat[obs.ignore_list != 0], obs.meas_lon[obs.ignore_list != 0], c='k', \
                    marker='x', s=5, alpha=0.5)



        # Plot a point marking the final point of the meteor
        m.scatter(self.rend_lat, self.rend_lon, c='k', marker='+', s=50, alpha=0.75, label='Lowest height')


        # If there are more than 10 observations, make the legend font smaller
        legend_font_size = LEGEND_TEXT_SIZE
        if len(self.observations) >= 10:
            legend_font_size = 5

        plt.legend(loc='upper left', prop={'size': legend_font_size})



        # Pickle the figure
        if ret_figs:
            fig_pickle_dict["ground_track"] = pickle.dumps(plt.gcf(), protocol=2)

        if self.save_results:
            savePlot(plt, file_name + '_ground_track.' + self.plot_file_type, output_dir)

        if show_plots:
            plt.show()

        else:
            plt.clf()
            plt.close()

        ######################################################################################################


        # # Plot angular residuals for every station separately
        # for obs in sorted(self.observations, key=lambda x: x.rbeg_ele, reverse=True):

        #     # Calculate residuals in arcseconds
        #     res = np.degrees(obs.ang_res)*3600

        #     # Mark ignored points
        #     if np.any(obs.ignore_list):

        #         ignored_times = obs.time_data[obs.ignore_list > 0]
        #         ignored_residuals = res[obs.ignore_list > 0]

        #         plt.scatter(ignored_times, ignored_residuals, facecolors='none', edgecolors='k', s=20, \
        #             zorder=4, label='Ignored points')


        #     # Calculate the RMSD of the residuals in arcsec
        #     res_rms = np.degrees(obs.ang_res_std)*3600

        #     # Plot residuals
        #     plt.scatter(obs.time_data, res, label='Angle, RMSD = {:.2f}"'.format(res_rms), s=2, zorder=3)


        #     plt.title('Observed vs. Radiant LoS Residuals, station ' + str(obs.station_id))
        #     plt.ylabel('Angle (arcsec)')
        #     plt.xlabel('Time (s)')

        #     # The lower limit is always at 0
        #     plt.ylim(ymin=0)

        #     plt.grid()
        #     plt.legend(prop={'size': LEGEND_TEXT_SIZE})

        #     if self.save_results:
        #         savePlot(plt, file_name + '_' + str(obs.station_id) + '_angular_residuals.' \
        #             + self.plot_file_type, output_dir)

        #     if show_plots:
        #         plt.show()

        #     else:
        #         plt.clf()
        #         plt.close()


        # Plot angular residuals from all stations
        first_ignored_plot = True
        for i, obs in enumerate(sorted(self.observations, key=lambda x:x.rbeg_ele, reverse=True)):

            # Extract marker type and size multiplier
            marker, sm = markers[i%len(markers)]

            # Calculate residuals in arcseconds
            res = np.degrees(obs.ang_res)*3600

            # Mark ignored points
            if np.any(obs.ignore_list):

                ignored_times = obs.time_data[obs.ignore_list > 0]
                ignored_residuals = res[obs.ignore_list > 0]

                # Plot the label only for the first occurence
                if first_ignored_plot:
                    
                    plt.scatter(ignored_times, ignored_residuals, facecolors='none', edgecolors='k', s=20, \
                        zorder=4, label='Ignored points')

                    first_ignored_plot = False

                else:
                    plt.scatter(ignored_times, ignored_residuals, facecolors='none', edgecolors='k', s=20, \
                        zorder=4)


            # Calculate the RMS of the residuals in arcsec
            res_rms = np.degrees(obs.ang_res_std)*3600

            # Plot residuals
            plt.scatter(obs.time_data, res, s=10*sm, zorder=3, label=str(obs.station_id) + \
                ', RMSD = {:.2f}"'.format(res_rms), marker=marker)


        plt.title('Observed vs. Radiant LoS Residuals, all stations')
        plt.ylabel('Angle (arcsec)')
        plt.xlabel('Time (s)')

        # The lower limit is always at 0
        plt.ylim(ymin=0)

        plt.grid()
        plt.legend(prop={'size': LEGEND_TEXT_SIZE})

        # Pickle the figure
        if ret_figs:
            fig_pickle_dict["all_angular_residuals"] = pickle.dumps(plt.gcf(), protocol=2)

        if self.save_results:
            savePlot(plt, file_name + '_all_angular_residuals.' + self.plot_file_type, output_dir)

        if show_plots:
            plt.show()

        else:
            plt.clf()
            plt.close()



        ######################################################################################################

        ### PLOT ABSOLUTE MAGNITUDES VS TIME, IF ANY ###

        first_ignored_plot = True
        if np.any([obs.absolute_magnitudes is not None for obs in self.observations]):

            # Go through all observations
            for obs in sorted(self.observations, key=lambda x: x.rbeg_ele, reverse=True):

                # Check if the absolute magnitude was given
                if obs.absolute_magnitudes is not None:

                    # Filter out None absolute magnitudes and magnitudes fainter than mag 10
                    filter_mask = np.array([abs_mag is not None for abs_mag in obs.absolute_magnitudes])

                    # Extract data that is not ignored
                    used_times = obs.time_data[filter_mask & (obs.ignore_list == 0)]
                    used_magnitudes = obs.absolute_magnitudes[filter_mask & (obs.ignore_list == 0)]

                    # Filter out magnitudes fainter than mag 10
                    mag_mask = np.array([abs_mag < 10 for abs_mag in used_magnitudes])
                    
                    # avoid crash if no magnitudes exceed the threshold
                    if isinstance(mag_mask, int):
                        used_times = used_times[mag_mask]
                        used_magnitudes = used_magnitudes[mag_mask]


                    plt_handle = plt.plot(used_times, used_magnitudes, marker='x', \
                        label=str(obs.station_id), zorder=3)

                    # Mark ignored absolute magnitudes
                    if np.any(obs.ignore_list):

                        # Extract data that is ignored
                        ignored_times = obs.time_data[filter_mask & (obs.ignore_list > 0)]
                        ignored_magnitudes = obs.absolute_magnitudes[filter_mask & (obs.ignore_list > 0)]

                        plt.scatter(ignored_times, ignored_magnitudes, facecolors='k', \
                            edgecolors=plt_handle[0].get_color(), marker='o', s=8, zorder=4)


            plt.xlabel('Time (s)')
            plt.ylabel('Absolute magnitude')

            plt.gca().invert_yaxis()

            plt.legend(prop={'size': LEGEND_TEXT_SIZE})

            plt.grid()

            # Pickle the figure
            if ret_figs:
                fig_pickle_dict["abs_mag"] = pickle.dumps(plt.gcf(), protocol=2)

            if self.save_results:
                savePlot(plt, file_name + '_abs_mag.' + self.plot_file_type, output_dir)

            if show_plots:
                plt.show()

            else:
                plt.clf()
                plt.close()


        ######################################################################################################


        ### PLOT ABSOLUTE MAGNITUDES VS HEIGHT, IF ANY ###

        first_ignored_plot = True
        if np.any([obs.absolute_magnitudes is not None for obs in self.observations]):

            # Go through all observations
            for obs in sorted(self.observations, key=lambda x: x.rbeg_ele, reverse=True):

                # Check if the absolute magnitude was given
                if obs.absolute_magnitudes is not None:

                    # Filter out None absolute magnitudes
                    filter_mask = np.array([abs_mag is not None for abs_mag in obs.absolute_magnitudes])

                    # Extract data that is not ignored
                    used_heights = obs.model_ht[filter_mask & (obs.ignore_list == 0)]
                    used_magnitudes = obs.absolute_magnitudes[filter_mask & (obs.ignore_list == 0)]

                    plt_handle = plt.plot(used_magnitudes, used_heights/1000, marker='x', \
                        label=str(obs.station_id), zorder=3)

                    # Mark ignored absolute magnitudes
                    if np.any(obs.ignore_list):

                        # Extract data that is ignored
                        ignored_heights = obs.model_ht[filter_mask & (obs.ignore_list > 0)]
                        ignored_magnitudes = obs.absolute_magnitudes[filter_mask & (obs.ignore_list > 0)]

                        plt.scatter(ignored_magnitudes, ignored_heights/1000, facecolors='k', \
                            edgecolors=plt_handle[0].get_color(), marker='o', s=8, zorder=4)


            plt.xlabel('Absolute magnitude')
            plt.ylabel('Height (km)')

            plt.gca().invert_xaxis()

            plt.legend(prop={'size': LEGEND_TEXT_SIZE})

            plt.grid()

            # Pickle the figure
            if ret_figs:
                fig_pickle_dict["abs_mag_ht"] = pickle.dumps(plt.gcf(), protocol=2)

            if self.save_results:
                savePlot(plt, file_name + '_abs_mag_ht.' + self.plot_file_type, output_dir)

            if show_plots:
                plt.show()

            else:
                plt.clf()
                plt.close()


        ######################################################################################################


        # Plot the orbit in 3D
        if self.calc_orbit:

            # Check if the orbit was properly calculated
            if self.orbit.ra_g is not None:

                # Construct a list of orbital elements of the meteor
                orbit_params = np.array([
                    [self.orbit.a, self.orbit.e, np.degrees(self.orbit.i), np.degrees(self.orbit.peri), \
                        np.degrees(self.orbit.node)]
                    ])

                if (output_dir is None) or (file_name is None):
                    plot_path = None
                    save_results = False

                else:
                    plot_path = os.path.join(output_dir, file_name)
                    save_results = self.save_results


                # Run orbit plotting procedure
                plotOrbits(orbit_params, jd2Date(self.jdt_ref, dt_obj=True), save_plots=save_results, \
                    plot_path=plot_path, linewidth=1, color_scheme='light', \
                    plot_file_type=self.plot_file_type)


                plt.tight_layout()

                # Pickle the figure
                if ret_figs:
                    fig_pickle_dict["orbit"] = pickle.dumps(plt.gcf(), protocol=2)


                if show_plots:
                    plt.show()

                else:
                    plt.clf()
                    plt.close()



        # Restore the status of save results scripts and return a dictionary of pickled figure objects
        if ret_figs:
            self.save_results = save_results_prev_status

            return fig_pickle_dict



    def showLoS(self):
        """ Show the stations and the lines of sight solution. """


        # Compute ECI values if they have not been computed
        if self.observations[0].model_eci is None:
            self.calcECIEqAltAz(self.state_vect_mini, self.radiant_eci_mini, self.observations)


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Calculate the position of the state vector (aka. first point on the trajectory)
        traj_point = self.observations[0].model_eci[0]/1000

        # Calculate the length to the last point on the trajectory
        meteor_len = np.sqrt(np.sum((self.observations[0].model_eci[0]/1000 \
            - self.observations[0].model_eci[-1]/1000)**2))

        # Calculate the plot limits
        x_list = [x_stat for obs in self.observations for x_stat in obs.stat_eci_los[:, 0]/1000]
        x_list.append(traj_point[0])
        y_list = [y_stat for obs in self.observations for y_stat in obs.stat_eci_los[:, 1]/1000]
        y_list.append(traj_point[1])
        z_list = [z_stat for obs in self.observations for z_stat in obs.stat_eci_los[:, 2]/1000]
        z_list.append(traj_point[2])

        x_min, x_max = min(x_list), max(x_list)
        y_min, y_max = min(y_list), max(y_list)
        z_min, z_max = min(z_list), max(z_list)


        # Normalize the plot limits so they are rectangular
        delta_x = x_max - x_min
        delta_y = y_max - y_min
        delta_z = z_max - z_min
        delta_max = max([delta_x, delta_y, delta_z])

        x_diff = delta_max - delta_x
        x_min -= x_diff/2
        x_max += x_diff/2

        y_diff = delta_max - delta_y
        y_min -= y_diff/2
        y_max += y_diff/2

        z_diff = delta_max - delta_z
        z_min -= z_diff/2
        z_max += z_diff/2


        # Plot stations and observations
        for obs in self.observations:

            # Station positions
            ax.scatter(obs.stat_eci_los[:, 0]/1000, obs.stat_eci_los[:, 1]/1000, obs.stat_eci_los[:, 2]/1000,\
                s=20)

            # Plot lines of sight
            for i, (stat_eci_los, meas_eci_los) in enumerate(zip(obs.stat_eci_los, obs.meas_eci_los)):

                # Take every other
                if i%2 == 1:
                    continue

                # Calculate the point on the trajectory
                traj_pt, _, _ = findClosestPoints(stat_eci_los, meas_eci_los, self.state_vect_mini, 
                    self.radiant_eci_mini)

                vect_len = np.sqrt(np.sum((stat_eci_los - traj_pt)**2))/1000

                # Lines of sight
                ax.quiver(stat_eci_los[0]/1000, stat_eci_los[1]/1000, stat_eci_los[2]/1000, 
                    meas_eci_los[0]/1000, meas_eci_los[1]/1000, meas_eci_los[2]/1000, 
                    length=vect_len, normalize=True, arrow_length_ratio=0, color='blue', alpha=0.5)



        # Plot the radiant state vector
        rad_x, rad_y, rad_z = -self.radiant_eci_mini/1000
        rst_x, rst_y, rst_z = traj_point
        ax.quiver(rst_x, rst_y, rst_z, rad_x, rad_y, rad_z, length=meteor_len, normalize=True, color='red', 
            arrow_length_ratio=0.1)

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])


        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')

        # Change the size of ticks (make them smaller)
        ax.tick_params(axis='both', which='major', labelsize=8)

        plt.show()



    def calcStationIncidentAngles(self, state_vect, radiant_eci, observations):
        """ Calculate angles between the radiant vector and the vector pointing from a station to the 
            initial state vector. 

        Arguments:
            state_vect: [ndarray] (x, y, z) ECI coordinates of the initial state vector (meters).
            radiant_eci: [ndarray] (x, y, z) components of the unit radiant direction vector.
            observations: [list] A list of ObservationPoints objects which hold measurements from individual
                stations.

        Return:
            return: [list] A list of angles (radians) for every station.
        """

        angles = []

        for obs in observations:

            # Calculate the vector pointing from the station to the state vector
            w = vectNorm(state_vect - obs.stat_eci)

            # Calculate the angle between the pointing vector and the radiant vector
            q_r = np.arccos(np.dot(radiant_eci, w))

            angles.append(q_r)


        return angles




    def run(self, _rerun_timing=False, _rerun_bad_picks=False, _mc_run=False, _orig_obs=None, 
        _prev_toffsets=None):
        """ Estimate the trajectory from the given input points. 
        
        Keyword arguments (internal flags, DO NOT SPECIFY MANUALLY!):
            _rerun_timing: [bool] Internal flag. Is it True when everything is recalculated upon estimating 
                the difference in timings, so it breaks the second trajectory run after updating the values
                of R.A., Dec, velocity, etc.
            _rerun_bad_picks: [bool] Internal flag. Is is True when a second pass of trajectory estimation is
                run with bad picks removed, thus improving the solution.
            _mc_run: [bool] Internal flag. True if the solver is calculating the Carlo Run.
            _orig_obs: [list] Used for Monte Carlo. A list of original observations, with no added noise.
                Used for calculating all other parameters after the trajectory with noise has been estimated.
            _prev_toffsets: [ndarray] Internal variable. Used for keeping the initially estimated timing 
                offsets from the first run of the solver. None by default.


        Return:
            traj_best: [Trajectory object] The best trajectory from all Monte Carlo runs. If no Monte Carlo
                runs were preformed, the pure LoS trajectory will be returned.

        """

        # Make sure there are at least 2 stations
        if numStationsNotIgnored(self.observations) < 2:
            
            print('At least 2 sets of measurements from 2 stations are needed to estimate the trajectory!')

            return None


        ### Recompute the reference JD and all times so that the first time starts at 0 ###

        # Determine the first relative time from reference JD
        t0 = min([obs.time_data[0] for obs in self.observations if (not obs.ignore_station) \
            or (not np.all(obs.ignore_list))])

        # If the first time is not 0, normalize times so that the earliest time is 0
        if t0 != 0.0:

            # Offset all times by t0
            for obs in self.observations:
                obs.time_data -= t0


            # Recompute the reference JD to corresponds with t0
            self.jdt_ref = self.jdt_ref + t0/86400.0


        ###################################################################################


        # Determine which station has the reference time (the first time entry is 0 for that station, but
        # do not take the station which has excluded points)
        for i, obs in enumerate(self.observations):

            # Do not take the station with excluded points as the reference one
            if obs.excluded_indx_range:
                continue
            
            if obs.time_data[0] == 0.0:
                self.t_ref_station = i
                break


        ### INTERSECTING PLANES SOLUTION ###
        ######################################################################################################

        self.intersection_list = []

        # Calculate all plane intersections in between all station pairs, only use non-ignored stations
        nonignored_observations = [obs for obs in self.observations if not obs.ignore_station]
        for i, obs1 in enumerate(nonignored_observations):
            for j, obs2 in enumerate(nonignored_observations[i + 1:]):

                # Perform plane intersection
                plane_intersection = PlaneIntersection(obs1, obs2)

                if self.verbose:
                    print('Convergence angle between stations', obs1.station_id, 'and', obs2.station_id)
                    print(' Q =', np.degrees(plane_intersection.conv_angle), 'deg')
                
                self.intersection_list.append(plane_intersection)


        radiant_sum = np.zeros(shape=3)
        weights_sum = 1e-10

        # Sum all radiants ECI positions and weights
        for plane_intersection in self.intersection_list:

            # Add the calculated radiant to the radiant sum
            radiant_sum += plane_intersection.weight*plane_intersection.radiant_eci

            weights_sum += plane_intersection.weight


        # Calculate the average radiant
        avg_radiant = radiant_sum/weights_sum

        # Normalize the radiant vector to a unit vector
        self.avg_radiant = vectNorm(avg_radiant)

        # Calculate the radiant position in RA and Dec
        self.radiant_eq = eci2RaDec(self.avg_radiant)

        if self.verbose:
            print('Multi-Track Weighted IP radiant:', np.degrees(self.radiant_eq))


        # Choose the intersection with the largest convergence angle as the best solution
        # The reason why the average trajectory determined from plane intersections is not taken as the 'seed'
        # for the LoS method is that the state vector cannot be calculated for the average radiant
        self.best_conv_inter = max(self.intersection_list, key=attrgetter('conv_angle'))

        if self.verbose:
            print('Best Convergence Angle IP radiant:', np.degrees(self.best_conv_inter.radiant_eq))


        # Set the 3D position of the radiant line as the state vector, at the beginning point
        self.state_vect = moveStateVector(self.best_conv_inter.cpa_eci, self.best_conv_inter.radiant_eci,
            self.observations)

        # Calculate incident angles between the trajectory and the station
        self.incident_angles = self.calcStationIncidentAngles(self.state_vect, \
            self.best_conv_inter.radiant_eci, self.observations)

        # Join each observation the calculated incident angle
        for obs, inc_angl in zip(self.observations, self.incident_angles):
            obs.incident_angle = inc_angl



        # If there are more than 2 stations, use weights for fitting
        if numStationsNotIgnored(self.observations) > 2:

            # Calculate minimization weights for LoS minimization as squared sines of incident angles
            weights = [np.sin(w)**2 for w in self.incident_angles]

        else:

            # Use unity weights if there are only two stations
            weights = [1.0]*len(self.observations)


        # Set weights to 0 for stations that are not used
        weights = [w if (self.observations[i].ignore_station == False) else 0 for i, w in enumerate(weights)]

        # Set weights to stations
        for w, obs in zip(weights, self.observations):
                obs.weight = w


        # Print weights
        if self.verbose:
            print('LoS statistical weights:')

            for obs in self.observations:
                print("{:>12s}, {:.3f}".format(obs.station_id, obs.weight))

        ######################################################################################################


        if self.verbose:
            print('Intersecting planes solution:', self.state_vect)
            
            print('Minimizing angle deviations...')


        ### LEAST SQUARES SOLUTION ###
        ######################################################################################################

        # Calculate the initial sum and angles deviating from the radiant line
        angle_sum = angleSumMeasurements2Line(self.observations, self.state_vect, \
             self.best_conv_inter.radiant_eci, weights=weights, \
             gravity=(_rerun_timing and self.gravity_correction))

        if self.verbose:
            print('Initial angle sum:', angle_sum)


        # Set the initial guess for the state vector and the radiant from the intersecting plane solution
        p0 = np.r_[self.state_vect, self.best_conv_inter.radiant_eci]

        # Perform the minimization of angle deviations. The gravity will only be compansated for after the
        #   initial estimate of timing differences
        minimize_solution = scipy.optimize.minimize(minimizeAngleCost, p0, args=(self.observations, weights, 
            (_rerun_timing and self.gravity_correction)), method="Nelder-Mead")

        # NOTE
        # Other minimization methods were tried as well, but all produce higher fit residuals than Nelder-Mead.
        # Tried:
        # - Powell, CS, BFGS - larger residuals
        # - Least Squares - large residuals
        # - Basinhopping with NM seed solution - long time to execute with no improvement


        # If the minimization diverged, bound the solution to +/-10% of state vector
        if np.max(np.abs(minimize_solution.x[:3] - self.state_vect)/self.state_vect) > 0.1:

            print('WARNING! Unbounded state vector optimization failed!')
            print('Trying bounded minimization to +/-10% of state vector position.')

            # Limit the minimization to 10% of original estimation in the state vector
            bounds = []
            for val in self.state_vect:
                bounds.append(sorted([0.9*val, 1.1*val]))

            # Bound the radiant vector to +/- 25% of original vales, per each ECI coordinate
            for val in self.best_conv_inter.radiant_eci:
                bounds.append(sorted([0.75*val, 1.25*val]))

            print('BOUNDS:', bounds)
            print('p0:', p0)
            minimize_solution = scipy.optimize.minimize(minimizeAngleCost, p0, args=(self.observations, \
                weights, (_rerun_timing and self.gravity_correction)), bounds=bounds, method='SLSQP')


        if self.verbose:
            print('Minimization info:')
            print(' Message:', minimize_solution.message)
            print(' Iterations:', minimize_solution.nit)
            print(' Success:', minimize_solution.success)
            print(' Final function value:', minimize_solution.fun)


        # Set the minimization status
        self.los_mini_status = minimize_solution.success

        # If the minimization succeded
        if minimize_solution.success:
        
            # Unpack the solution
            self.state_vect_mini, self.radiant_eci_mini = np.hsplit(minimize_solution.x, 2)

            # Set the state vector to the position of the highest point projected on the radiant line
            self.state_vect_mini = moveStateVector(self.state_vect_mini, self.radiant_eci_mini, 
                self.observations)

            # Normalize radiant direction
            self.radiant_eci_mini = vectNorm(self.radiant_eci_mini)

            # Convert the minimized radiant solution to RA and Dec
            self.radiant_eq_mini = eci2RaDec(self.radiant_eci_mini)

            if self.verbose:
                print('Position and radiant LMS solution:')
                print(' State vector:', self.state_vect_mini)
                print(' Ra', np.degrees(self.radiant_eq_mini[0]), 'Dec:', np.degrees(self.radiant_eq_mini[1]))

        else:

            print('Angle minimization failed altogether!')

            # If the solution did not succeed, set the values to intersecting plates solution
            self.radiant_eci_mini = self.best_conv_inter.radiant_eci

            # Normalize radiant direction
            self.radiant_eci_mini = vectNorm(self.radiant_eci_mini)

            # Convert the minimized radiant solution to RA and Dec
            self.radiant_eq_mini = eci2RaDec(self.radiant_eci_mini)

            # Calculate the state vector
            self.state_vect_mini = moveStateVector(self.state_vect, self.radiant_eci_mini, 
                self.observations)

        ######################################################################################################


        # If running a Monte Carlo run, switch the observations to the original ones, so the noise does not 
        # influence anything except the radiant position
        if (_mc_run or _rerun_timing) and (_orig_obs is not None):
                
            # Store the noisy observations for later
            self.obs_noisy = list(self.observations)

            # Replace the noisy observations with original observations
            self.observations = _orig_obs


            # If this is the run of recalculating the parameters after updating the timing, preserve the
            # timing as well
            if _rerun_timing:
                for obs, obs_noise in zip(self.observations, self.obs_noisy):
                    obs.time_data = np.copy(obs_noise.time_data)


        # Calculate velocity at each point
        self.calcVelocity(self.state_vect_mini, self.radiant_eci_mini, self.observations, weights)


        if self.verbose and self.estimate_timing_vel:
            print('Estimating initial velocity and timing differences...')



        # # Show the pre-time corrected time vs. length
        # if not _rerun_timing:

        #     ### PLOT DISTANCE FROM RADIANT STATE VECTOR POSITION ###
        #     ######################################################################################################
        #     for obs in self.observations:

        #         # Extract points that were not ignored
        #         used_times = obs.time_data[obs.ignore_list == 0]
        #         used_dists = obs.state_vect_dist[obs.ignore_list == 0]

        #         plt_handle = plt.plot(used_dists/1000, used_times, marker='x', label=str(obs.station_id), \
        #             zorder=3)


        #         # Plot ignored points
        #         if np.any(obs.ignore_list):

        #             ignored_times = obs.time_data[obs.ignore_list > 0]
        #             ignored_dists = obs.state_vect_dist[obs.ignore_list > 0]
                        
        #             plt.scatter(ignored_dists/1000, ignored_times, facecolors='k', \
        #                 edgecolors=plt_handle[0].get_color(), marker='o', s=8, zorder=4, \
        #                 label='{:s} ignored points'.format(str(obs.station_id)))


        #     plt.title("Distances from state vector, before time correction")

        #     plt.ylabel('Time (s)')
        #     plt.xlabel('Distance from state vector (km)')
            
        #     plt.legend()
        #     plt.grid()
        #     plt.gca().invert_yaxis()

        #     plt.tight_layout()

        #     plt.show()


        # Calculate the lag ONLY if it was not calculated during timing estimation
        if self.observations[0].lag is None:

            # Calculate lag
            self.calcLag(self.observations)
            

        # Estimate the timing difference between stations and the initial velocity and update the time
        self.timing_minimization_successful, self.velocity_fit, self.v_init, self.time_diffs, \
            self.observations = self.estimateTimingAndVelocity(self.observations, weights, \
                estimate_timing_vel=self.estimate_timing_vel)


        # If estimating the timing failed, skip any further steps
        if not self.timing_minimization_successful:
            print('unable to minimise timing')
            return None


        # Calculate velocity at each point with updated timings
        self.calcVelocity(self.state_vect_mini, self.radiant_eci_mini, self.observations, weights,
            calc_res=_rerun_timing)


        ### RERUN THE TRAJECTORY ESTIMATION WITH UPDATED TIMINGS ###
        ######################################################################################################

        # Runs only in the first pass of trajectory estimation and estimates timing offsets between stations
        if not _rerun_timing:

            # Assign the initial timing differences
            if not _rerun_bad_picks:
                self.time_diffs_final = self.time_diffs

            else:
                # Assign the timing differences after bad pick removal
                self.time_diffs_final += self.time_diffs


            # After the timing has been estimated, everything needs to be recalculated from scratch
            if self.estimate_timing_vel:


                # If doing a Monte Carlo run, switch back to noisy observations
                if _mc_run and (_orig_obs is not None):

                    # Keep the updated timings
                    for obs, obs_noise in zip(self.observations, self.obs_noisy):
                        obs_noise.time_data = np.copy(obs.time_data)

                    # Switch back to noisy observations, but with updated timing
                    self.observations = self.obs_noisy


                # Make a copy of observations
                temp_observations = copy.deepcopy(self.observations)

                
                # Reset the observation points
                self.observations = []

                if self.verbose:
                    print()
                    print("---------------------------------------------------------------------------------")
                    print("Updating the solution after the timing estimation...")
                    print("---------------------------------------------------------------------------------")

                # Reinitialize the observations with proper timing
                for obs in temp_observations:
                    self.infillWithObs(obs)

                
                # Re-run the trajectory estimation with updated timings. This will update all calculated
                # values up to this point
                self.run(_rerun_timing=True, _prev_toffsets=self.time_diffs, _orig_obs=_orig_obs)


        else:

            # In the second pass with updated timings, calculate the final timing offsets
            self.time_diffs_final += self.time_diffs


            return None


        ######################################################################################################


        # If running a Monte Carlo runs, switch the observations to the original ones, so noise does not 
        # infuence anything except the radiant position
        if _mc_run and (_orig_obs is not None):
                
            # Store the noisy observations for later
            self.obs_noisy = list(self.observations)

            # Replace the noisy observations with original observations
            self.observations = _orig_obs

            # Keep the updated timings
            for obs, obs_noise in zip(self.observations, self.obs_noisy):
                obs.time_data = np.copy(obs_noise.time_data)



        # If the stations have no time overlap at all, skip further computations
        if len([obs for obs in self.observations if not obs.ignore_station]) < 2:
            return None


        # Do a Jacchia exponential fit to the lag, per every station
        self.jacchia_fit = self.fitJacchiaLag(self.observations)

        # Calculate latitude, longitude and altitude of each point closest to the radiant line, in WGS84
        self.calcLLA(self.state_vect_mini, self.radiant_eci_mini, self.observations)


        # Compute the initial velocity as the average velocity of all points above the given height
        #   (optional)
        if self.v_init_ht is not None:
            v_ht_avg, intercept_ht_avg = self.calcAvgVelocityAboveHt(self.observations, 1000*self.v_init_ht, \
                weights)

            # Assign this average velocity as the initial velocity if the fit was successful
            if v_ht_avg is not None:
                
                self.v_init = v_ht_avg
                self.velocity_fit = [self.v_init, intercept_ht_avg]

                # Recalculate the lag
                self.calcLag(self.observations, velocity_fit=self.velocity_fit)

                # Refit jacchia lag fit
                self.jacchia_fit = self.fitJacchiaLag(self.observations)


        # Calculate ECI positions of the CPA on the radiant line, RA and Dec of the points on the radiant
        # line as seen by the observers, the corresponding azimuth and elevation, and set arrays model_fit1 
        # and model_fit2 to be of the same type as the input parameters meas1 and meas2
        self.calcECIEqAltAz(self.state_vect_mini, self.radiant_eci_mini, self.observations)


        # Calculate horizontal, vertical and angular residuals from the lines of sight to the radiant line
        self.calcAllResiduals(self.state_vect_mini, self.radiant_eci_mini, self.observations)

        # Calculate absolute magnitudes
        self.calcAbsMagnitudes()


        ### REMOVE BAD PICKS AND RECALCULATE ###
        ######################################################################################################

        if self.filter_picks:
            if (not _rerun_bad_picks):

                picks_rejected = 0

                # Remove all picks which deviate more than N sigma in angular residuals
                for obs in self.observations:

                    # Find the indicies of picks which are within N sigma
                    good_picks = obs.ang_res < (np.mean(obs.ang_res) \
                        + self.reject_n_sigma_outliers*obs.ang_res_std)

                    # If the number of good picks is below 4, do not remove any picks
                    if np.count_nonzero(good_picks) < 4:
                        continue

                    # Check if any picks were removed
                    if np.count_nonzero(good_picks) < len(obs.ang_res):
                        picks_rejected += len(obs.ang_res) - np.count_nonzero(good_picks)

                        # Ignore bad picks
                        obs.ignore_list[~good_picks] = 1


                # Run only if some picks were rejected
                if picks_rejected:

                    # Make a copy of observations
                    temp_observations = copy.deepcopy(self.observations)
                    
                    # Reset the observation points
                    self.observations = []

                    if self.verbose:
                        print()
                        print("---------------------------------------------------------------------------------")
                        print("Updating the solution after rejecting", picks_rejected, "bad picks...")
                        print("---------------------------------------------------------------------------------")

                    # Reinitialize the observations without the bad picks
                    for obs in temp_observations:
                        self.infillWithObs(obs)

                    
                    # Re-run the trajectory estimation with updated timings. This will update all calculated
                    # values up to this point
                    self.run(_rerun_bad_picks=True, _prev_toffsets=self.time_diffs_final)

                else:
                    if self.verbose:
                        print("All picks are within 3 sigma...")


            else:

                # In the second pass, return None
                return None

        ######################################################################################################

        # If the time fit failed, stop further computations
        if not self.timing_minimization_successful:
            return None


        ### CALCULATE ORBIT ###
        ######################################################################################################

        if self.calc_orbit:

            # Calculate average velocity and average ECI position of the trajectory
            self.v_avg, self.state_vect_avg, self.jd_avg = self.calcAverages(self.observations)


            # Calculate the orbit of the meteor
            # If the LoS estimation failed, then the plane intersection solution will be used for the orbit,
            # which needs to have fixed stations and the average velocity should be the reference velocity
            self.orbit = calcOrbit(self.radiant_eci_mini, self.v_init, self.v_avg, self.state_vect_mini, \
                self.rbeg_jd, stations_fixed=(not minimize_solution.success), \
                reference_init=minimize_solution.success)

            if self.verbose:
                print(self.orbit.__repr__(v_init_ht=self.v_init_ht))


        ######################################################################################################


        # Break if doing a Monte Carlo run
        if _mc_run:
            return None


        if self.monte_carlo:

            # Do a Monte Carlo estimate of the uncertainties in all calculated parameters
            traj_best, uncertainties = monteCarloTrajectory(self, mc_runs=self.mc_runs, \
                mc_pick_multiplier=self.mc_pick_multiplier, noise_sigma=self.mc_noise_std, \
                geometric_uncert=self.geometric_uncert, plot_results=self.save_results, \
                mc_cores=self.mc_cores)


            ### Save uncertainties to the trajectory object ###
            if uncertainties is not None:
                traj_uncer = copy.deepcopy(uncertainties)

                # Remove the list of all MC trajectires (it is unecessarily big)
                traj_uncer.mc_traj_list = []

                # Set the uncertainties to the best trajectory (maintain compatibility with older version 
                #   before the typo fix)
                traj_best.uncertainties = traj_uncer
                traj_best.uncertanties = traj_uncer

            ######


            # Copy uncertainties to the geometrical trajectory
            self = copyUncertainties(traj_best, self)


        else:
            uncertainties = None


        #### SAVE RESULTS ###
        ######################################################################################################

        if self.save_results or self.show_plots:

            # Save Monte Carlo results
            if self.monte_carlo:

                traj_best.save_results = self.save_results
                traj_best.show_plots = self.show_plots

                # Monte Carlo output directory
                mc_output_dir = os.path.join(self.output_dir, 'Monte Carlo')
                mc_file_name = self.file_name + "_mc"


                if self.save_results:

                    if self.verbose:
                        print('Saving Monte Carlo results...')

                    # Save the picked trajectory structure with Monte Carlo points
                    savePickle(traj_best, mc_output_dir, mc_file_name + '_trajectory.pickle')
                    
                    # Save the uncertainties
                    savePickle(uncertainties, mc_output_dir, mc_file_name + '_uncertainties.pickle')

                    # Save trajectory report
                    traj_best.saveReport(mc_output_dir, mc_file_name + '_report.txt', \
                        uncertainties=uncertainties, verbose=self.verbose)

                # Save and show plots
                traj_best.savePlots(mc_output_dir, mc_file_name, show_plots=self.show_plots)


            ## Save original picks results
            if self.save_results:

                if self.verbose:
                    print('Saving results with original picks...')

                # Save the picked trajectory structure with original points
                savePickle(self, self.output_dir, self.file_name + '_trajectory.pickle')

                # Save trajectory report with original points
                self.saveReport(self.output_dir, self.file_name + '_report.txt', \
                        uncertainties=uncertainties, verbose = not self.monte_carlo)


            # Save and show plots
            self.savePlots(self.output_dir, self.file_name, \
                show_plots=(self.show_plots and not self.monte_carlo))
            

        ######################################################################################################



        ## SHOW PLANE INTERSECTIONS AND LoS PLOTS ###
        #####################################################################################################

        # # Show the plane intersection
        # if self.show_plots:
        #   self.best_conv_inter.show()


        # # Show lines of sight solution in 3D
        # if self.show_plots:
        #   self.showLoS()


        #####################################################################################################


        # Return the best trajectory
        if self.monte_carlo:
            return traj_best

        else:
            return self






if __name__ == "__main__":

    ### TEST CASE ###
    ##########################################################################################################

    import time
    from wmpl.Utils.TrajConversions import equatorialCoordPrecession_vect, J2000_JD

    ## TEST EVENT
    ###############
    # Reference julian date
    jdt_ref = 2458601.365760937799

    # Inputs are RA/Dec
    meastype = 1

    # Measurements
    station_id1 = "RU0001"
    time1 = np.array([0.401190, 0.441190, 0.481190, 0.521190, 0.561190, 0.601190, 0.641190, 0.681190, 
                      0.721190, 0.761190, 0.801190, 0.841190, 0.881190, 0.921190, 0.961190, 1.001190, 
                      1.041190, 1.081190, 1.121190, 1.161190, 1.201190, 1.241190, 1.281190, 1.321190, 
                      1.361190, 1.401190, 1.441190, 1.561190, 1.601190, 1.641190, 1.721190, 1.761190, 
                      1.841190])
    ra1 = np.array([350.35970, 350.71676, 351.29184, 351.58998, 352.04673, 352.50644, 352.91289, 353.37336, 
                    353.80532, 354.23339, 354.69277, 355.07317, 355.49321, 355.93473, 356.32148, 356.74755, 
                    357.13866, 357.51363, 357.89944, 358.34052, 358.72626, 359.11597, 359.53391, 359.88343, 
                      0.35106,   0.71760,   1.05526,   2.17105,   2.58634,   2.86315,   3.58752,   3.90806, 
                      4.48084])
    dec1 = np.array([+74.03591, +73.94472, +73.80889, +73.73877, +73.59830, +73.46001, +73.35001, +73.22812, 
                     +73.10211, +72.98779, +72.84568, +72.72924, +72.59691, +72.46677, +72.33622, +72.18147, 
                     +72.04381, +71.91015, +71.77648, +71.63370, +71.47512, +71.32664, +71.16185, +71.03236, 
                     +70.84506, +70.67285, +70.54194, +70.01219, +69.80856, +69.69043, +69.38316, +69.23522, 
                     +68.93025])


    station_id2 = "RU0002"
    time2 = np.array([0.000000, 0.040000, 0.080000, 0.120000, 0.160000, 0.200000, 0.240000, 0.280000, 
                      0.320000, 0.360000, 0.400000, 0.440000, 0.480000, 0.520000, 0.560000, 0.600000, 
                      0.640000, 0.680000, 0.720000, 0.760000, 0.800000, 0.840000, 0.880000, 0.920000, 
                      0.960000, 1.000000, 1.040000, 1.080000, 1.120000, 1.160000, 1.200000, 1.240000, 
                      1.280000, 1.320000, 1.360000, 1.400000, 1.440000, 1.480000, 1.520000, 1.560000, 
                      1.600000, 1.640000, 1.680000, 1.720000, 1.760000, 1.800000, 1.840000, 1.880000, 
                      1.920000, 1.960000, 2.000000, 2.040000, 2.080000, 2.120000, 2.160000, 2.200000, 
                      2.240000, 2.280000, 2.320000, 2.360000, 2.400000, 2.440000, 2.480000, 2.520000,])
    ra2 = np.array([ 81.27325, 81.20801, 81.06648, 81.03509, 80.93281, 80.87338, 80.74776, 80.68456, 
                     80.60038, 80.52306, 80.45021, 80.35990, 80.32309, 80.21477, 80.14311, 80.06967, 
                     79.98169, 79.92234, 79.84210, 79.77507, 79.72752, 79.62422, 79.52738, 79.48236, 
                     79.39613, 79.30580, 79.23434, 79.20863, 79.12019, 79.03670, 78.94849, 78.89223, 
                     78.84252, 78.76605, 78.69339, 78.64799, 78.53858, 78.53906, 78.47469, 78.39496, 
                     78.33473, 78.25761, 78.23964, 78.17867, 78.16914, 78.07010, 78.04741, 77.95169, 
                     77.89130, 77.85995, 77.78812, 77.76807, 77.72458, 77.66024, 77.61543, 77.54208, 
                     77.50465, 77.45944, 77.43200, 77.38361, 77.36004, 77.28842, 77.27131, 77.23300])
    dec2 = np.array([+66.78618, +66.66040, +66.43476, +66.21971, +66.01550, +65.86401, +65.63294, +65.43265, 
                     +65.25161, +65.01655, +64.83118, +64.62955, +64.45051, +64.23361, +64.00504, +63.81778, 
                     +63.61334, +63.40714, +63.19009, +62.98101, +62.76420, +62.52019, +62.30266, +62.05585, 
                     +61.84240, +61.60207, +61.40390, +61.22904, +60.93950, +60.74076, +60.53772, +60.25602, 
                     +60.05801, +59.83635, +59.59978, +59.37846, +59.10216, +58.88266, +58.74728, +58.45432, 
                     +58.18503, +57.97737, +57.72030, +57.55891, +57.31933, +56.98481, +56.85845, +56.58652, 
                     +56.36153, +56.15409, +55.88252, +55.66986, +55.46593, +55.20145, +54.91643, +54.69826, 
                     +54.49443, +54.25651, +54.06386, +53.86395, +53.70069, +53.47312, +53.33715, +53.20272])

    # Convert measurement to radians
    ra1 = np.radians(ra1)
    dec1 = np.radians(dec1)

    ra2 = np.radians(ra2)
    dec2 = np.radians(dec2)

    ###

    ### SITES INFO

    lon1 = np.radians(37.315140)
    lat1 = np.radians(44.890740)
    ele1 = 26.00

    lon2 = np.radians(38.583580)
    lat2 = np.radians(44.791620)
    ele2 = 240.00

    ###


    # Init new trajectory solving
    traj_solve = Trajectory(jdt_ref, meastype=meastype, save_results=False, monte_carlo=False, show_plots=False)

    # Set input points for the first site
    traj_solve.infillTrajectory(ra1, dec1, time1, lat1, lon1, ele1, station_id=station_id1)

    # Set input points for the second site
    traj_solve.infillTrajectory(ra2, dec2, time2, lat2, lon2, ele2, station_id=station_id2)

    traj_solve.run()

    ###############


    # TEST
    fig_pickle_dict = traj_solve.savePlots(None, None, show_plots=False, ret_figs=True)

    for key in fig_pickle_dict:
        print(key)
        fig = pickle.loads(fig_pickle_dict[key])
