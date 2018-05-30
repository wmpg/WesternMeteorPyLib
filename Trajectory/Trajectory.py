""" PyLIG trajectory solver

Estimates meteor trajectory from given observed points. 

"""

from __future__ import print_function, division, absolute_import

import time
import copy
import sys
import os
import datetime
from operator import attrgetter

import numpy as np
import scipy.optimize
import scipy.interpolate
import scipy.stats

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from mpl_toolkits.basemap import Basemap


from Trajectory.Orbit import calcOrbit

from Utils.Math import vectNorm, vectMag, meanAngle, findClosestPoints, RMSD, angleBetweenSphericalCoords, \
    checkContinuity
from Utils.OSTools import mkdirP
from Utils.Pickling import savePickle
from Utils.Plotting import savePlot
from Utils.PlotOrbits import plotOrbits
from Utils.PlotCelestial import CelestialPlot
from Utils.PlotMap import GroundMap
from Utils.TrajConversions import EARTH, G, ecef2ENU, enu2ECEF, geo2Cartesian, geo2Cartesian_vect, \
    cartesian2Geo, altAz2RADec_vect, raDec2AltAz, raDec2AltAz_vect, raDec2ECI, eci2RaDec, jd2Date, datetime2JD
from Utils.PyDomainParallelizer import DomainParallelizer




class ObservedPoints(object):
    def __init__(self, jdt_ref, meas1, meas2, time_data, lat, lon, ele, meastype, station_id=None, \
        excluded_time=None, ignore_list=None, ignore_station=False):
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

        ######################################################################################################


        ### CALCULATED DATA ###
        ######################################################################################################

        # Angle between the station, the state vector, and the trajectory
        self.incident_angle = None

        # Residuals from the fit
        self.h_residuals = None
        self.v_residuals = None

        # Calculated velocities (in m/s)
        self.velocities = None

        # Calculated length along the path (meters)
        self.length = None

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

        # Add meteor line of sight positions and station positions to single arays
        x_data = np.append(self.x_eci, 0)
        y_data = np.append(self.y_eci, 0)
        z_data = np.append(self.z_eci, 0)

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

        # If the radiant is closer to the antiradiant, reverse signs
        if np.dot(self.obs1.meas_eci[0], self.radiant_eci) < np.dot(self.obs1.meas_eci[-1], self.radiant_eci):
            self.radiant_eci = -self.radiant_eci

        # Calculate the radiant position in RA and Dec
        self.radiant_eq = eci2RaDec(self.radiant_eci)


        ###### Calculate the closest point of approach (CPA) from the stations to the radiant line,
        ###### that is, a vector pointing from each station to the radiant line, which magnitude
        ###### corresponds to the distance to the radiant line

        ### Calculate the unit vector pointing from the 1st station to the radiant line ###
        ######################################################################################################

        w1 = np.cross(self.radiant_eci, self.obs1.plane_N)

        # Normalize the vector
        w1 = vectNorm(w1)

        # Invert vector orientation if pointing towards the station, not the radiant line
        if np.dot(w1, self.obs1.meas_eci[0]) < 0:
            w1 = -w1
        
        ######################################################################################################


        ### Calculate the unit vector pointing from the 2nd station to the radiant line ###
        ######################################################################################################

        w2 = np.cross(self.radiant_eci, self.obs2.plane_N)

        # Normalize the vector
        w2 = vectNorm(w2)

        # Invert vector orientation if pointing towards the station, not the radiant line
        if np.dot(w2, self.obs2.meas_eci[0]) < 0:
            w2 = -w2
        ######################################################################################################


        ### Calculate the range from stations to the radiant line ###
        ######################################################################################################

        # Calculate the difference in position of the two stations
        stat_diff = self.obs1.stat_eci - self.obs2.stat_eci

        # Calculate the angle between the pointings to the radiant line
        stat_cosangle = np.dot(w1, w2)


        # Calculate the range from the 1st station to the radiant line
        stat_range1 = (stat_cosangle*np.dot(stat_diff, w2) - np.dot(stat_diff, w1))/(1.0 \
            - stat_cosangle**2)

        # Calculate the CPA vector for the 1st station
        self.rcpa_stat1 = stat_range1*w1


        # Calculate the range from the 2nd station to the radiant line
        stat_range2 = (np.dot(stat_diff, w2) - stat_cosangle*np.dot(stat_diff, w1))/(1.0 \
            - stat_cosangle**2)

        # Calculate the CPA vector for the 2nd station
        self.rcpa_stat2 = stat_range2*w2


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


        # Calculate the quiver arrow length
        arrow_len = 0.2*np.sqrt((x_min - x_max)**2 + (y_min - y_max)**2 + (z_min - z_max)**2)

        # Plot stations and observations
        for obs in observations:

            # Station positions
            ax.scatter(obs.x_stat, obs.y_stat, obs.z_stat, s=50)

            # Lines of sight
            ax.quiver(obs.x_stat, obs.y_stat, obs.z_stat, obs.x_eci, obs.y_eci, obs.z_eci, length=arrow_len, \
                normalize=True, arrow_length_ratio=0.1, color='blue')

            d = -np.array([obs.x_stat, obs.y_stat, obs.z_stat]).dot(obs.plane_N)

            # Create x,y
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))

            # Calculate corresponding z
            z = (-obs.plane_N[0]*xx - obs.plane_N[1]*yy - d)*1.0/obs.plane_N[2]

            # Plot plane normal
            ax.quiver(obs.x_stat, obs.y_stat, obs.z_stat, *obs.plane_N, length=arrow_len/2, 
                normalize=True, arrow_length_ratio=0.1, color='green')

            # Plot the plane
            ax.plot_surface(xx, yy, z, alpha=0.25)


        # Plot the radiant state vector
        rad_x, rad_y, rad_z = -self.radiant_eci
        rst_x, rst_y, rst_z = traj_point
        ax.quiver(rst_x, rst_y, rst_z, rad_x, rad_y, rad_z, length=arrow_len, normalize=True, color='red', \
            arrow_length_ratio=0.1)

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

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

                # Calculate the gravitational acceleration at the given height
                g = G*EARTH.MASS/(vectMag(rad_cpa)**2)

                # Determing the sign of the initial time
                time_sign = np.sign(t_rel)

                # Calculate the amount of gravity drop from a straight trajectory (handle the case when the time
                #   can be negative)
                drop = time_sign*(1.0/2)*g*t_rel**2

                # Apply gravity drop to ECI coordinates
                rad_cpa -= drop*vectNorm(rad_cpa)


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




def calcSpatialResidual(jd, state_vect, radiant_eci, stat, meas):
    """ Calculate horizontal and vertical residuals from the radiant line, for the given observed point.

    Arguments:
        jd: [float] Julian date
        t_rel: [float] Time from the beginning of the trajectory
        state_vect: [3 element ndarray] ECI position of the state vector
        radiant_eci: [3 element ndarray] radiant direction vector in ECI
        stat: [3 element ndarray] position of the station in ECI
        meas: [3 element ndarray] line of sight from the station, in ECI

    Return:
        (hres, vres): [tuple of floats] residuals in horitontal and vertical direction from the radiant line

    """


    # Note:
    #   This function has been tested (without the gravity influence part) and it produces good results


    meas = vectNorm(meas)

    # Calculate closest points of approach (observed line of sight to radiant line) from the state vector
    obs_cpa, rad_cpa, d = findClosestPoints(stat, meas, state_vect, radiant_eci)

    # ### STILL IN TESTING !!!
    # # Calculate the gravitational acceleration at the given height
    # g = G*EARTH.MASS/(vectMag(rad_cpa)**2)

    # # Determine the sign of the initial time
    # time_sign = np.sign(t_rel)

    # # Calculate the amount of gravity drop from a straight trajectory (handle the case when the
    # #   time is negative)
    # drop = time_sign*(1/2.0)*g*t_rel**2

    # # Apply gravity drop to ECI coordinates
    # rad_cpa -= drop*vectNorm(rad_cpa)

    # ###########################

    # # Calculate closest points of approach (observed line of sight to radiant line) from the gravity corrected
    # #   point
    # obs_cpa, _, d = findClosestPoints(stat, meas, rad_cpa, radiant_eci)

    # ##!!!!!


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



def lineFunc(x, m, k):
    """ Line defined by slope and intercept. 
    
    Arguments:
        x: [float] independant variable
        m: [float] slope
        k: [float] intercept

    Return:
        [float]: line given by (m, k) evaluated at x

    """

    return m*x + k



def lineFuncLS(params, x, y):
    """ Line defined by slope and intercept. Version for least squares.
    
    Arguments:
        params: [list] Line parameters
        x: [float] Independant variable
        y: [float] Estimated values

    Return:
        [float]: line given by (m, k) evaluated at x

    """

    return lineFunc(x, *params) - y



def jacchiaLagFunc(t, a1, a2):
    """ Jacchia (1955) model for modeling lengths along the trail of meteors, modified to fit the lag (length 
        along the trail minus the linear part, estimated by fitting a line to the first 25% of observations, 
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



def fitLagIntercept(time, length, v_init, initial_intercept=0.0):
    """ Finds the intercept of the line with the given slope. Used for fitting time vs. length along the trail
        data.

    Arguments:
        time: [ndarray] Array containing the time data (seconds).
        length: [ndarray] Array containing the length along the trail data.
        v_init: [float] Fixed slope of the line (i.e. initial velocity).

    Keyword arguments:
        initial_intercept: [float] Initial estimate of the intercept.

    Return:
        (slope, intercept): [tuple of floats] fitted line parameters
    """

    # Fit a line to the first 25% of the points
    quart_size = int(0.25*len(time))

    # If the size is smaller than 4 points, take all point
    if quart_size < 4:
        quart_size = len(time)

    quart_length = length[:quart_size]
    quart_time = time[:quart_size]

    # Redo the lag fit, but with fixed velocity
    lag_intercept, _ = scipy.optimize.curve_fit(lambda x, intercept: lineFunc(x, v_init, intercept), 
        quart_time, quart_length, p0=[initial_intercept])

    return v_init, lag_intercept[0]




# def timingAndVelocityResiduals(params, observations, t_ref_station, ret_stddev=False):
#     """ Calculate the sum of absolute differences between the lag of the reference station and all other 
#         stations, by using the given initial velocity and timing differences between stations. 
    
#     Arguments:
#         params: [ndarray] first element is the initial velocity, all others are timing differences from the 
#             reference station (NOTE: reference station is NOT in this list)
#         observations: [list] a list of ObservedPoints objects
#         t_ref_station: [int] index of the reference station

#     Arguments:
#         ret_stddev: [bool] Returns the standard deviation of lag offsets instead of the cost function.
    
#     Return:
#         [float] sum of absolute differences between the reference and lags of all stations
#     """

#     stat_count = 0

#     # The first parameters is the initial velocity
#     v_init = params[0]

#     lags = []

#     # Go through observations from all stations
#     for i, obs in enumerate(observations):

#         # Time difference is 0 for the reference statins
#         if i == t_ref_station:
#             t_diff = 0

#         else:
#             # Take the estimated time difference for all other stations
#             t_diff = params[stat_count + 1]
#             stat_count += 1


#         # Calculate the shifted time
#         time_shifted = obs.time_data + t_diff

#         # Estimate the intercept of the lag line, with the fixed slope (i.e. initial velocity)
#         lag_line = fitLagIntercept(time_shifted, obs.length, v_init, obs.lag_line[1])

#         # Calculate lag
#         lag = obs.length - lineFunc(time_shifted, *lag_line)

#         # Add lag to lag list
#         lags.append([time_shifted,  np.array(lag)])


#     # Choose the reference lag
#     ref_time, ref_lag = lags[t_ref_station]

#     # Do a monotonic cubic spline fit on the reference lag
#     ref_line_spline = scipy.interpolate.PchipInterpolator(ref_time, ref_lag, extrapolate=True)

#     residual_sum = 0
#     stddev_sum = 0
#     stddev_count = 0

#     # Go through all lags
#     for i, obs in enumerate(observations):

#         # Skip the lag from the reference station
#         if i == t_ref_station:
#             continue

#         time, lag = lags[i]

#         # Take only those points that overlap with the reference station
#         common_points = np.where((time > np.min(ref_time)) & (time < np.max(ref_time)))

#         # Do this is there are at least 4 overlapping points
#         if len(common_points[0]) > 4:
#             time = time[common_points]
#             lag = lag[common_points]

#         # Calculate the residuals in lag from the current lag to the reference lag, using smooth approximation
#         # of L1 (absolute value) cost
#         z = (ref_line_spline(time) - lag)**2
#         residual_sum += np.sum(2*(np.sqrt(1 + z) - 1))

#         # Standard deviation calculation
#         stddev_sum += np.sum(z)
#         stddev_count += len(z)


#     lag_stddev = np.sqrt(stddev_sum/stddev_count)


#     if ret_stddev:

#         # Returned for reporting the goodness of fit
#         return lag_stddev

#     else:

#         # Returned for minimization
#         return residual_sum



def timingResiduals(params, observations, t_ref_station, weights=None, ret_stddev=False, \
    ret_len_residuals=False):
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
        ret_len_residuals: [bool] Returns the length residuals instead of the timing residuals. Used for 
            evaluating the goodness of length matching during the Monte Carlo procedure. False by default.
    
    Return:
        [float] Average absolute difference between the timings from all stations using the length for
            matching.

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



    # Returns the length residuals (NOT USED)
    if ret_len_residuals:

        # Choose the reference station time and distances
        ref_time, ref_dist = state_vect_distances[t_ref_station]

        # Do a monotonic cubic spline fit on the reference state vector distance
        ref_line_spline = scipy.interpolate.PchipInterpolator(ref_time, ref_dist, extrapolate=True)

        residual_sum = 0
        stddev_sum = 0
        stddev_count = 0

        # Go through all state vector distances
        for i, obs in enumerate(observations):

            # Skip the reference station
            if i == t_ref_station:
                continue

            time_data, state_vect_dist = state_vect_distances[i]

            # Take only those points that overlap with the reference station
            common_points = np.where((time_data > np.min(ref_time)) & (time_data < np.max(ref_time)))

            # Do this is there are at least 4 overlapping points
            if len(common_points[0]) > 4:
                time_data = time_data[common_points]
                state_vect_dist = state_vect_dist[common_points]

            # Calculate the residuals in dist from the current dist to the reference dist, using smooth
            # approximation of L1 (absolute value) cost
            z = (ref_line_spline(time_data) - state_vect_dist)**2
            residual_sum += np.sum(2*(np.sqrt(1 + z) - 1))

            # Standard deviation calculation
            stddev_sum += np.sum(z)
            stddev_count += len(z)


        # Calculate the standard deviation of the fit
        dist_stddev = np.sqrt(stddev_sum/stddev_count)


        if ret_stddev:

            # Returned for reporting the goodness of fit
            return dist_stddev

        else:

            # Returned for minimization
            return residual_sum/stddev_count



    # Return the timing residuals (used for determining the timing offset)
    else:

        cost_sum = 0
        cost_point_count = 0
        weights_sum = 1e-10

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

        # Go through all observations from all stations
        for obs in observations:

            # Calculate closest points of approach (observed line of sight to radiant line) of the first point
            # on the trajectory across all stations
            _, rad_cpa, _ = findClosestPoints(obs.stat_eci_los[0], obs.meas_eci_los[0], state_vect, 
                radiant_eci)

            rad_cpa_list.append(rad_cpa)


        # Choose the state vector with the largest height
        rad_cpa_beg = rad_cpa_list[np.argmax([vectMag(rad_cpa_temp) for rad_cpa_temp in rad_cpa_list])]

        return rad_cpa_beg




class MCUncertanties(object):
    def __init__(self, mc_traj_list):
        """ Container for standard deviations of trajectory parameters calculated using Monte Carlo. """

        # A list with all trajectory objects calculated via Monte Carlo
        self.mc_traj_list = mc_traj_list

        # State vector position
        self.state_vect_mini = None
        self.x = None
        self.y = None
        self.z = None

        # Velocity state vector
        self.vx = None
        self.vy = None
        self.z = None

        # Radiant vector
        self.radiant_eci_mini = None

        # Beginning/ending points
        self.rbeg_lon = None
        self.rbeg_lat = None
        self.rbeg_ele = None

        self.rend_lon = None
        self.rend_lat = None
        self.rend_ele = None

        # Apparent radiant position (radians)
        self.ra = None
        self.dec = None

        # Apparent azimuth and altitude
        self.azimuth_apparent = None
        self.elevation_apparent = None

        # Estimated average velocity
        self.v_avg = None

        # Estimated initial velocity
        self.v_init = None

        # Longitude of the reference point on the trajectory (rad)
        self.lon_ref = None

        # Latitude of the reference point on the trajectory (rad)
        self.lat_ref = None

        # Geocentric latitude of the reference point (rad)
        self.lat_geocentric = None

        # Apparent zenith angle (before the correction for Earth's gravity)
        self.zc = None

        # Zenith distance of the geocentric radiant (after the correction for Earth's gravity)
        self.zg = None

        # Velocity at infinity
        self.v_inf = None

        # Geocentric velocity (m/s)
        self.v_g = None

        # Geocentric radiant position (radians)
        self.ra_g = None
        self.dec_g = None

        # Ecliptic coordinates of the radiant (radians)
        self.L_g = None
        self.B_g = None

        # Sun-centered ecliptic rectangular coordinates of the average position on the meteor's trajectory 
        # (in kilometers)
        self.meteor_pos = None

        # Helioventric velocity of the meteor (m/s)
        self.v_h = None

        # Corrected heliocentric velocity vector of the meteoroid using the method of Sato & Watanabe (2014)
        self.v_h_x = None
        self.v_h_y = None
        self.v_h_z = None

        # Corrected ecliptci coordinates of the meteor using the method of Sato & Watanabe (2014)
        self.L_h = None
        self.B_h = None

        # Solar longitude (radians)
        self.la_sun = None

        # Semi-major axis (AU)
        self.a = None

        # Eccentricty
        self.e = None

        # Inclination (radians)
        self.i = None

        # Argument of perihelion (radians)
        self.peri = None

        # Ascending node (radians)
        self.node = None

        # Longitude of perihelion (radians)
        self.pi = None

        # Perihelion distance (AU)
        self.q = None

        # Aphelion distance (AU)
        self.Q = None

        # True anomaly at the moment of contact with Earth (radians)
        self.true_anomaly = None

        # Exxentric anomaly (radians)
        self.eccentric_anomaly = None

        # Mean anomaly (radians)
        self.mean_anomaly = None

        # Calculate the date and time of the last perihelion passage (datetime object)
        self.last_perihelion = None

        # Mean motion in the orbit (rad/day)
        self.n = None

        # Orbital period
        self.T = None

        # Tisserand's parameter with respect to Jupiter
        self.Tj = None





def calcMCUncertanties(traj_list, traj_best):
    """ Takes a list of trajectory objects and returns the standard deviation of every parameter. 

    Arguments:
        traj_list: [list] A list of Trajectory objects, each is the result of an individual Monte Carlo run.
        traj_best: [Trajectory object] Trajectory which is chosen to the be the best of all MC runs.

    Return:
        un: [MCUncertainties object] Object containing the uncertainty of every calculated parameter.
    """


    # Init a new container for uncertanties
    un = MCUncertanties(traj_list)

    # Initial velocity
    un.v_init = np.std([traj.v_init for traj in traj_list])

    # State vector
    un.x = np.std([traj.state_vect_mini[0] for traj in traj_list])
    un.y = np.std([traj.state_vect_mini[1] for traj in traj_list])
    un.z = np.std([traj.state_vect_mini[2] for traj in traj_list])

    un.state_vect_mini = np.array([un.x, un.y, un.z])


    rad_x = np.std([traj.radiant_eci_mini[0] for traj in traj_list])
    rad_y = np.std([traj.radiant_eci_mini[1] for traj in traj_list])
    rad_z = np.std([traj.radiant_eci_mini[2] for traj in traj_list])

    un.radiant_eci_mini = np.array([rad_x, rad_y, rad_z])

    # Velocity state vector
    un.vx = abs(traj_best.v_init*traj_best.radiant_eci_mini[0]*(un.v_init/traj_best.v_init
        + rad_x/traj_best.radiant_eci_mini[0]))
    un.vy = abs(traj_best.v_init*traj_best.radiant_eci_mini[1]*(un.v_init/traj_best.v_init
        + rad_y/traj_best.radiant_eci_mini[1]))
    un.vz = abs(traj_best.v_init*traj_best.radiant_eci_mini[2]*(un.v_init/traj_best.v_init
        + rad_z/traj_best.radiant_eci_mini[2]))


    # Beginning/ending points
    un.rbeg_lon = scipy.stats.circstd([traj.rbeg_lon for traj in traj_list])
    un.rbeg_lat = np.std([traj.rbeg_lat for traj in traj_list])
    un.rbeg_ele = np.std([traj.rbeg_ele for traj in traj_list])

    un.rend_lon = scipy.stats.circstd([traj.rend_lon for traj in traj_list])
    un.rend_lat = np.std([traj.rend_lat for traj in traj_list])
    un.rend_ele = np.std([traj.rend_ele for traj in traj_list])


    if traj_best.orbit is not None:

        # Apparent
        un.ra = scipy.stats.circstd([traj.orbit.ra for traj in traj_list])
        un.dec = np.std([traj.orbit.dec for traj in traj_list])
        un.v_avg = np.std([traj.orbit.v_avg for traj in traj_list])
        un.v_inf = np.std([traj.orbit.v_inf for traj in traj_list])

        un.azimuth_apparent = scipy.stats.circstd([traj.orbit.azimuth_apparent for traj in traj_list])
        un.elevation_apparent = np.std([traj.orbit.elevation_apparent for traj in traj_list])

        # reference point on the meteor trajectory
        un.lon_ref = scipy.stats.circstd([traj.orbit.lon_ref for traj in traj_list])
        un.lat_ref = np.std([traj.orbit.lat_ref for traj in traj_list])
        un.lat_geocentric = np.std([traj.orbit.lat_geocentric for traj in traj_list])

        # Geocentric
        un.ra_g = scipy.stats.circstd([traj.orbit.ra_g for traj in traj_list])
        un.dec_g = np.std([traj.orbit.dec_g for traj in traj_list])
        un.v_g = np.std([traj.orbit.v_g for traj in traj_list])

        # Meteor position in Sun-centred rectangular coordinates
        meteor_pos_x = np.std([traj.orbit.meteor_pos[0] for traj in traj_list])
        meteor_pos_y = np.std([traj.orbit.meteor_pos[1] for traj in traj_list])
        meteor_pos_z = np.std([traj.orbit.meteor_pos[2] for traj in traj_list])

        un.meteor_pos = np.array([meteor_pos_x, meteor_pos_y, meteor_pos_z])

        # Zenith angles
        un.zc = np.std([traj.orbit.zc for traj in traj_list])
        un.zg = np.std([traj.orbit.zg for traj in traj_list])


        # Ecliptic geocentric
        un.L_g = scipy.stats.circstd([traj.orbit.L_g for traj in traj_list])
        un.B_g = np.std([traj.orbit.B_g for traj in traj_list])
        un.v_h = np.std([traj.orbit.v_h for traj in traj_list])

        # Ecliptic heliocentric
        un.L_h = scipy.stats.circstd([traj.orbit.L_h for traj in traj_list])
        un.B_h = np.std([traj.orbit.B_h for traj in traj_list])
        un.v_h_x = np.std([traj.orbit.v_h_x for traj in traj_list])
        un.v_h_y = np.std([traj.orbit.v_h_y for traj in traj_list])
        un.v_h_z = np.std([traj.orbit.v_h_z for traj in traj_list])

        # Orbital elements
        un.la_sun = scipy.stats.circstd([traj.orbit.la_sun for traj in traj_list])
        un.a = np.std([traj.orbit.a for traj in traj_list])
        un.e = np.std([traj.orbit.e for traj in traj_list])
        un.i = np.std([traj.orbit.i for traj in traj_list])
        un.peri = scipy.stats.circstd([traj.orbit.peri for traj in traj_list])
        un.node = scipy.stats.circstd([traj.orbit.node for traj in traj_list])
        un.pi = scipy.stats.circstd([traj.orbit.pi for traj in traj_list])
        un.q = np.std([traj.orbit.q for traj in traj_list])
        un.Q = np.std([traj.orbit.Q for traj in traj_list])
        un.true_anomaly = scipy.stats.circstd([traj.orbit.true_anomaly for traj in traj_list])
        un.eccentric_anomaly = scipy.stats.circstd([traj.orbit.eccentric_anomaly for traj in traj_list])
        un.mean_anomaly = scipy.stats.circstd([traj.orbit.mean_anomaly for traj in traj_list])

        # Last perihelion uncertanty (days)
        un.last_perihelion = np.std([datetime2JD(traj.orbit.last_perihelion) for traj in traj_list if \
            isinstance(traj.orbit.last_perihelion, datetime.datetime)])

        # Mean motion in the orbit (rad/day)
        un.n = np.std([traj.orbit.n for traj in traj_list])

        # Orbital period
        un.T = np.std([traj.orbit.T for traj in traj_list])

        # Tisserand's parameter
        un.Tj = np.std([traj.orbit.Tj for traj in traj_list])
    

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
    node_list = np.array([traj.orbit.node for traj in mc_traj_list])
    peri_list = np.array([traj.orbit.peri for traj in mc_traj_list])
    i_list = np.array([traj.orbit.i for traj in mc_traj_list])
    

    # Calculate the orbital covariance
    orbit_input = np.c_[e_list, q_list, tp_list, node_list, peri_list, i_list].T
    orbit_cov = np.cov(orbit_input, aweights=weights)


    # Extract inital state vectors
    state_vect_list = np.array([traj.state_vect_mini for traj in mc_traj_list])
    initial_vel_vect_list = np.array([traj.v_init*traj.radiant_eci_mini for traj in mc_traj_list])

    # Calculate inital state vector covariance
    state_vect_input = np.hstack([state_vect_list, initial_vel_vect_list]).T
    state_vect_cov = np.cov(state_vect_input, aweights=weights)


    return orbit_cov, state_vect_cov



def _MCTrajSolve(i, traj, observations):
    """ Internal function. Does a Monte Carlo run of the given trajectory object. Used as a function for
        parallelization. 

    Arguments:
        i: [int] Number of MC run to be printed out.
        traj: [Trajectory object] Trajectory object on which the run will be performed.
        observations: [list] A list of observations with no noise.

    Return:
        traj: [Trajectory object] Trajectory object with the MC solution.

    """

    print('Run No.', i + 1)

    traj.run(_mc_run=True, _orig_obs=observations)

    return traj



def monteCarloTrajectory(traj, mc_runs=None, mc_pick_multiplier=1, noise_sigma=1):
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


    print('Doing', mc_runs, ' Monte Carlo runs...')

    # List which holds all trajectory objects with the added noise
    mc_input_list = []

    # Do mc_runs Monte Carlo runs
    for i in range(mc_runs):

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

                zhat = np.zeros(3)

                # Southern Hemisphere
                if(rhat[2] < 0.0):

                    zhat[0] =  0.0
                    zhat[1] =  0.0
                    zhat[2] = +1.0

                    uhat = vectNorm(np.cross(rhat, zhat))
                    vhat = vectNorm(np.cross(uhat, rhat))
                
                # Northern Hemisphere
                else:
                    zhat[0] =  0.0
                    zhat[1] =  0.0
                    zhat[2] = -1.0

                    uhat = vectNorm(np.cross(zhat, rhat))
                    vhat = vectNorm(np.cross(uhat, rhat))
                

                # sqrt(2)/2*noise in each orthogonal dimension
                sigma = noise_sigma*np.abs(obs.ang_res_std)/np.sqrt(2.0)

                meas_eci_noise = np.zeros(3)

                meas_eci_noise[0] = rhat[0] + np.random.normal(0, sigma)*uhat[0] \
                    + np.random.normal(0, sigma)*vhat[0]
                meas_eci_noise[1] = rhat[1] + np.random.normal(0, sigma)*uhat[1] \
                    + np.random.normal(0, sigma)*vhat[1]
                meas_eci_noise[2] = rhat[2] + np.random.normal(0, sigma)*uhat[2] \
                    + np.random.normal(0, sigma)*vhat[2]

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
                ignore_list=obs.ignore_list)

            
        # Do not show plots or perform additional optimizations
        traj_mc.verbose = False
        traj_mc.estimate_timing_vel = True
        traj_mc.filter_picks = False
        traj_mc.show_plots = False
        traj_mc.save_results = False


        # Add the modified trajectory object to the input list for parallelization and the original observations
        mc_input_list.append([i, traj_mc, traj.observations])


    # Run MC trajectory estimation on multiple cores
    mc_results = DomainParallelizer(mc_input_list, _MCTrajSolve)

    # Add the original trajectory in the Monte Carlo results, if it is the one which has the best length match
    mc_results.append(traj)

    ##########################################################################################################

    # TESTING!!!!!!!!!!!
    # # Take only those solutions which have the timing standard deviation <= than the initial solution
    # mc_results = [mc_traj for mc_traj in mc_results if mc_traj.timing_stddev <= traj.timing_stddev]

    # Take only those solutions which have the timing residuals <= than the initial solution
    mc_results = [mc_traj for mc_traj in mc_results if mc_traj.timing_res <= traj.timing_res]

    ##########

    # Reject those solutions for which LoS angle minimization failed
    mc_results = [mc_traj for mc_traj in mc_results if mc_traj.los_mini_status == True]

    # Reject those solutions for which the orbit could not be calculated
    mc_results = [mc_traj for mc_traj in mc_results if (mc_traj.orbit.ra_g is not None) \
        and (mc_traj.orbit.dec_g is not None)]


    # Break the function of there are no trajectories to process
    if len(mc_results) == 0:
        return traj, None



    # Choose the solution with the lowest timing residuals as the best solution - THIS GIVES BETTER RESULTS
    # THAN STDDEV!!!
    timing_res_trajs = [traj_tmp.timing_res for traj_tmp in mc_results]
    best_traj_ind = timing_res_trajs.index(min(timing_res_trajs))

    # # Choose the solution with the lowest length standard deviation as the best solution
    # timing_res_trajs = [traj_tmp.timing_stddev for traj_tmp in mc_results]
    # best_traj_ind = timing_res_trajs.index(min(timing_res_trajs))

    # Choose the best trajectory
    traj_best = mc_results[best_traj_ind]


    # Calculate the standard deviation of every trajectory parameter
    uncertanties = calcMCUncertanties(mc_results, traj_best)


    # Calculate orbital and inital state vector covariance matrices
    traj_best.orbit_cov, traj_best.state_vect_cov = calcCovMatrices(mc_results)


    ### PLOT RADIANT SPREAD (Vg color and length stddev) ###
    ##########################################################################################################

    if traj.orbit is not None:

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
                savePlot(plt, traj.file_name + '_monte_carlo_eq_' + plt_flag + '.png', \
                    output_dir=traj.output_dir)


            if traj.show_plots:
                plt.show()

            else:
                plt.clf()
                plt.close()

    ##########################################################################################################



    ### PLOT ORBITAL ELEMENTS SPREAD ###
    ##########################################################################################################

    if traj.orbit is not None:

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

        # Semimajor axis vs. inclination
        ax1.hist2d(a_list, np.degrees(incl_list))
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
        ax2.hist2d(np.degrees(peri_list), np.degrees(incl_list))
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
        ax3.hist2d(e_list, q_list)
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
        ax4.hist2d(np.degrees(peri_list), q_list)
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
            savePlot(plt, traj.file_name + '_monte_carlo_orbit_elems.png', output_dir=traj.output_dir)


        if traj.show_plots:
            plt.show()

        else:
            plt.clf()
            plt.close()

    ##########################################################################################################


    return traj_best, uncertanties




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


    def __init__(self, jdt_ref, output_dir='.', max_toffset=1.0, meastype=4, verbose=True, \
        estimate_timing_vel=True, monte_carlo=True, mc_runs=None, mc_pick_multiplier=1,  mc_noise_std=1.0, \
        filter_picks=True, calc_orbit=True, show_plots=True, save_results=True, gravity_correction=True,
        plot_all_spatial_residuals=False):
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
            estimate_timing_vel: [bool] Try to estimate the difference in timing and velocity. True by default.
            monte_carlo: [bool] Runs Monte Carlo estimation of uncertanties. True by default.
            mc_runs: [int] Number of Monte Carlo runs. The default value is the number of observed points.
            mc_pick_multiplier: [int] Number of MC samples that will be taken for every point. 1 by default.
            mc_noise_std: [float] Number of standard deviations of measurement noise to add during Monte
                Carlo estimation.
            filter_picks: [bool] If True (default), picks which deviate more than 3 sigma in angular residuals
                will be removed, and the trajectory will be recalculated.
            calc_orbit: [bool] If True, the orbit is calculates as well. True by default
            show_plots: [bool] Show plots of residuals, velocity, lag, meteor position. True by default.
            save_results: [bool] Save results of trajectory estimation to disk. True by default.
            gravity_correction: [bool] Apply the gravity drop when estimating trajectories. True by default.
            plot_all_spatial_residuals: [bool] Plot all spatial residuals on one plot (one vs. time, and
                the other vs. length). False by default.

        """

        # All time data must be given relative to this Julian date
        self.jdt_ref = jdt_ref

        # Measurement type
        self.meastype = meastype

        # Directory where the trajectory estimation results will be saved
        self.output_dir = output_dir

        # Maximum time offset between cameras
        self.max_toffset = max_toffset

        # If verbose True, results and status messages will be printed out, otherwise they will be supressed
        self.verbose = verbose

        # Estimating the difference in timing between stations, and the initial velocity if this flag is True
        self.estimate_timing_vel = estimate_timing_vel

        # Running Monte Carlo simulations to estimate uncertanties
        self.monte_carlo = monte_carlo

        # Number of Monte Carlo runs
        self.mc_runs = mc_runs

        # Number of MC samples that will be taken for every point
        self.mc_pick_multiplier = mc_pick_multiplier

        # Standard deviatons of measurement noise to add during Monte Carlo runs
        self.mc_noise_std = mc_noise_std

        # Filter bad picks (ones that deviate more than 3 sigma in angular residuals) if this flag is True
        self.filter_picks = filter_picks

        # Calculate orbit if True
        self.calc_orbit = calc_orbit

        # If True, plots will be shown on screen when the trajectory estimation is done
        self.show_plots = show_plots

        # Save results to disk if true
        self.save_results = save_results

        # Apply the correction for gravity when estimating the trajectory
        self.gravity_correction = gravity_correction

        # Plot all spatial residuals on one plot
        self.plot_all_spatial_residuals = plot_all_spatial_residuals

        ######################################################################################################


        # Construct a file name for this event
        self.file_name = jd2Date(self.jdt_ref, dt_obj=True).strftime('%Y%m%d_%H%M%S')

        # Counts from how may stations the observations are given (start from 1)
        self.station_count = 1

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

        # Calculated initial velocity
        self.v_init = None

        # Calculated average velocity
        self.v_avg = None

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

        # Uncertanties calculated using Monte Carlo
        self.uncertanties = None

        # Orbital covariance matrix
        self.orbit_cov = None

        # Initial state vector covariance matrix
        self.state_vect_cov = None



    def infillTrajectory(self, meas1, meas2, time_data, lat, lon, ele, station_id=None, excluded_time=None,
        ignore_list=None):
        """ Initialize a set of measurements for a given station. 
    
        Arguments:
            meas1: [list or ndarray] First measurement array (azimuth or R.A., depending on meastype, see 
                meastype documentation for more information). Measurements should be given in radians.
            meas2: [list or ndarray] Second measurement array (altitude, zenith angle or declination, 
                depending on meastype, see meastype documentation for more information), in radians.
            time_data: [list or ndarray] Time in seconds from the reference Julian date.
            lat: [float] Latitude +N of station in radians.
            lon: [float] Longitude +E of station in radians.
            ele: [float] Elevation of station in meters.

        Keyword arguments:
            station_id: [str] Identification of the station. None by default.
            excluded_time: [list] A range of minimum and maximum observation time which should be excluded 
                from the optimization because the measurements are missing in that portion of the time.
            ignore_list: [list or ndarray] A list of 0s and 1s which should be of the equal length as 
                the input data. If a particular data point is to be ignored, number 1 should be put,
                otherwise (if the point should be used) 0 should be used. E.g. the this should could look
                like this: [0, 0, 0, 1, 1, 0, 0], which would mean that the fourth and the fifth points
                will be ignored in trajectory estimation.

        Return:
            None
        """

        # If station ID was not given, assign it a name
        if station_id is None:
            station_id = self.station_count


        # Convert measuremet lists to numpy arrays
        meas1 = np.array(meas1)
        meas2 = np.array(meas2)
        time_data = np.array(time_data)

        # Skip the observation if all points were ignored
        if ignore_list is not None:
            if np.all(ignore_list):
                print('All points from station {:s} are ignored, not using this station in the solution!'.format(station_id))


        # Init a new structure which will contain the observed data from the given site
        obs = ObservedPoints(self.jdt_ref, meas1, meas2, time_data, lat, lon, ele, station_id=station_id, 
            meastype=self.meastype, excluded_time=excluded_time, ignore_list=ignore_list)
            
        # Add observations to the total observations list
        self.observations.append(obs)

        self.station_count += 1



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
                hres, vres = calcSpatialResidual(jd, state_vect, radiant_eci, stat, meas)

                # Add residuals to the residual list
                obs.h_residuals.append(hres)
                obs.v_residuals.append(vres)

            # Convert residual lists to numpy arrays
            obs.h_residuals = np.array(obs.h_residuals)
            obs.v_residuals = np.array(obs.v_residuals)

            # Calculate RMSD of both residuals
            obs.h_res_rms = RMSD(obs.h_residuals)
            obs.v_res_rms = RMSD(obs.v_residuals)


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



    def calcVelocity(self, state_vect, radiant_eci, observations, calc_res=False):
        """ Calculates velocity for the given solution.


        Arguments:
            state_vect: [ndarray] (x, y, z) ECI coordinates of the initial state vector (meters).
            radiant_eci: [ndarray] (x, y, z) components of the unit radiant direction vector.
            observations: [list] A list of ObservationPoints objects which hold measurements from individual
                stations.

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


            ### Length vs. time

            # plt.plot(obs.state_vect_dist, obs.time_data, marker='x', label=str(obs.station_id), zorder=3)

            ##########

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


        # plt.ylabel('Time (s)')
        # plt.xlabel('Distance from state vector (m)')

        # plt.gca().invert_yaxis()

        # plt.legend()
        # plt.grid()
        # plt.savefig('mc_time_offsets.png', dpi=300)
        # plt.show()



        if calc_res and (self.time_diffs_final is not None):

            # Calculate the timing offset between the meteor time vs. length
            self.timing_res = timingResiduals(self.time_diffs_final, self.observations, self.t_ref_station)
            self.timing_stddev = timingResiduals(self.time_diffs_final, self.observations, self.t_ref_station, \
                ret_stddev=True)




    def calcLag(self, observations):
        """ Calculate lag by fitting a line to the first 25% of the points and subtracting the line from the 
            length along the trail.

        Arguments:
            observations: [list] A list of ObservationPoints objects which hold measurements from individual
                stations.
        """

        # Go through observations from all stations
        for obs in observations:

            # Fit a line to the first 25% of the points
            quart_size = int(0.25*len(obs.time_data))

            # If the size is smaller than 4 points, take all point
            if quart_size < 4:
                quart_size = len(obs.time_data)

            # Cut the length and time to the first quarter
            quart_length = obs.length[:quart_size]
            quart_time = obs.time_data[:quart_size]

            # Fit a line to the data
            obs.lag_line, _ = scipy.optimize.curve_fit(lineFunc, quart_time, quart_length)

            # Initial velocity is the slope of the fitted line
            obs.v_init = obs.lag_line[0]

            # Calculate lag
            obs.lag = obs.length - lineFunc(obs.time_data, *obs.lag_line)



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
        time_all = np.hstack([obs.time_data for obs in self.observations])
        lag_all = np.hstack([obs.lag for obs in self.observations])
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



    def estimateTimingAndVelocity(self, observations, estimate_timing_vel=True):
        """ Estimates time offsets between the stations by matching time vs. distance from state vector. 
            The initial velocity is calculated by ineratively fitting a line from the beginning to 20% of the 
            total trajectory, and up to the 80% of the total trajectory. The fit with the lowest standard
            deviation is chosen to represent the initial velocity.

        Arguments:
            observations: [list] A list of ObservationPoints objects which hold measurements from individual
                stations.

        Return:
            (velocity_fit, v_init_mini, time_diffs, observations): [tuple]
                velocity_fit: [tuple] (slope, intercept) tuple of a line fit on the time vs. length data.
                v_init_mini: [float] Estimated initial velocity in m/s.
                time_diffs: [ndarray] Estimated time offsets from individual stations.
                observations: [list] A list of ObservationPoints objects which hold measurements from 
                    individual stations. These objects are modified during timing estimations.

        """

        # Take the initial velocity as the median velocity between all sites
        v_init = np.array([obs.v_init for obs in observations])
        v_init = np.median(v_init)

        # Timing differences which will be calculated
        time_diffs = np.zeros(len(observations))

        # If the timing difference and velocity difference is not desired to be performed, skip the procedure
        if not estimate_timing_vel:
            return v_init, time_diffs


        # Initial timing difference between sites is 0 (there are N-1 timing differences, as the time 
        # difference for the reference site is always 0)
        p0 = np.zeros(shape=(self.station_count - 1))

        # Set the time reference station to be the one with the most used points
        obs_points = [obs.kmeas for obs in self.observations]
        self.t_ref_station = obs_points.index(max(obs_points))


        if self.verbose:
            print('Initial function evaluation:', timingResiduals(p0, observations, self.t_ref_station))


        # Set bounds for timing to +/- given maximum time offset
        bounds = []
        for i in range(self.station_count - 1):
            bounds.append([-self.max_toffset, self.max_toffset])


        # Try different methods of optimization until it is successful
        methods = ['SLSQP', 'TNC', None]
        maxiter_list = [1000, None, 15000]
        for opt_method, maxiter in zip(methods, maxiter_list):

            # Run the minimization of residuals between all stations (set tolerance to 1 ns)
            timing_mini = scipy.optimize.minimize(timingResiduals, p0, args=(observations, \
                self.t_ref_station), bounds=bounds, method=opt_method, options={'maxiter': maxiter}, \
                tol=1e-9)

            # Stop trying methods if this one was successful
            if timing_mini.success:
                if self.verbose:
                    print('Successful timing optimization with', opt_method)

                break

            else:
                print('Unsuccessful timing optimization with', opt_method)


        # If the minimization was successful, apply the time corrections
        if timing_mini.success:

            if self.verbose:
                print("Final function evaluation:", timing_mini.fun)
                

            stat_count = 0
            for i, obs in enumerate(observations):

                # The timing difference for the reference station is always 0
                if i == self.t_ref_station:
                    t_diff = 0

                else:
                    t_diff = timing_mini.x[stat_count]
                    stat_count += 1

                if self.verbose:
                    print('STATION ' + str(obs.station_id) + ' TIME OFFSET = ' + str(t_diff) + ' s')

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
            for obs in observations:

                # Skip ignored stations
                if obs.ignore_station:
                    continue

                times.append(obs.time_data[obs.ignore_list == 0])
                state_vect_dist.append(obs.state_vect_dist[obs.ignore_list == 0])

            times = np.concatenate(times).ravel()
            state_vect_dist = np.concatenate(state_vect_dist).ravel()

            # Sort points by time
            time_sort_ind = times.argsort()
            times = times[time_sort_ind]
            state_vect_dist = state_vect_dist[time_sort_ind]


            stddev_list = []

            # Calculate the velocity on different initial portions of the trajectory

            # Find the best fit by starting from the first few beginning points
            for part_beg in range(4):

                # Find the best fit on different portions of the trajectory
                for part in np.arange(0.25, 0.8, 0.05):

                    # Get the index of the beginning of the first portion of points
                    # part_beg = int(init_point*len(times))

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

                    # Fit a line to time vs. state_vect_dist
                    velocity_fit = scipy.optimize.least_squares(lineFuncLS, [v_init, 1], args=(times_part, \
                        state_vect_dist_part), loss='soft_l1')
                    velocity_fit = velocity_fit.x

                    # Calculate the lag and fit a line to it
                    lag_temp = state_vect_dist - lineFunc(times, *velocity_fit)
                    lag_fit = scipy.optimize.least_squares(lineFuncLS, np.ones(2), args=(times, lag_temp), \
                        loss='soft_l1')
                    lag_fit = lag_fit.x

                    # Add the point to the considered list only if the lag has a negative trend
                    if lag_fit[0] <= 0:

                        # Calculate the standard deviation of the line fit and add it to the list of solutions
                        line_stddev = RMSD(state_vect_dist_part - lineFunc(times_part, *velocity_fit))
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



        else:

            print('Timing difference and initial velocity minimization failed with the message:')
            print(timing_mini.message)
            print('Try increasing the range of time offsets!')
            v_init_mini = v_init

            velocity_fit = np.zeros(2)
            v_init_mini = 0


        return velocity_fit, v_init_mini, time_diffs, observations




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

                # Calculate the time in seconds from the beginning of the meteor
                t_rel = t - t0

                # Apply the gravity drop
                rad_cpa = applyGravityDrop(rad_cpa, t_rel, r0, v0z)


                # # Calculate the gravitational acceleration at the given height
                # g = G*EARTH.MASS/(vectMag(rad_cpa)**2)

                # # Determing the sign of the initial time
                # time_sign = np.sign(t_rel)

                # # Calculate the amount of gravity drop from a straight trajectory (handle the case when the time
                # #   can be negative)
                # drop = time_sign*(1.0/2)*g*t_rel**2

                # # Apply gravity drop to ECI coordinates
                # rad_cpa -= drop*vectNorm(rad_cpa)

                ###
                

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



        # Find the highest beginning height
        beg_hts = [obs.rbeg_ele for obs in self.observations if obs.ignore_station == False]
        first_begin = beg_hts.index(max(beg_hts))

        # Set the coordinates of the height point as the first point
        self.rbeg_lat = self.observations[first_begin].rbeg_lat
        self.rbeg_lon = self.observations[first_begin].rbeg_lon
        self.rbeg_ele = self.observations[first_begin].rbeg_ele
        self.rbeg_jd = self.observations[first_begin].rbeg_jd


        # Find the lowest ending height
        end_hts = [obs.rend_ele for obs in self.observations if obs.ignore_station == False]
        last_end = end_hts.index(min(end_hts))

        # Set coordinates of the lowest point as the last point
        self.rend_lat = self.observations[last_end].rend_lat
        self.rend_lon = self.observations[last_end].rend_lon
        self.rend_ele = self.observations[last_end].rend_ele
        self.rend_jd = self.observations[last_end].rend_jd



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

            # If input were RA and Dec
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



    def dumpMeasurements(self, dir_path, file_name):
        """ Writes the initialized measurements in a MATLAB format text file."""

        with open(os.path.join(dir_path, file_name), 'w') as f:

            for i, obs in enumerate(self.observations):

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



    def saveReport(self, dir_path, file_name, uncertanties=None, verbose=True):
        """ Save the trajectory estimation report to file. 
    
        Arguments:
            dir_path: [str] Path to the directory where the report will be saved.
            file_name: [str] Name of the report time.

        Keyword arguments:
            uncertanties: [MCUncertainties object] Object contaning uncertainties of every parameter.
            verbose: [bool] Print the report to the screen. True by default.
        """


        def _uncer(str_format, std_name, multi=1.0, deg=False):
            """ Internal function. Returns the formatted uncertanty, if the uncertanty is given. If not,
                it returns nothing. 

            Arguments:
                str_format: [str] String format for the unceertanty.
                std_name: [str] Name of the uncertanty attribute, e.g. if it is 'x', then the uncertanty is 
                    stored in uncertanties.x.
        
            Keyword arguments:
                multi: [float] Uncertanty multiplier. 1.0 by default. This is used to scale the uncertanty to
                    different units (e.g. from m/s to km/s).
                deg: [bool] Converet radians to degrees if True. False by defualt.
                """

            if deg:
                multi *= np.degrees(1.0)

            if uncertanties is not None:
                return " +/- " + str_format.format(getattr(uncertanties, std_name)*multi)

            else:
                return ''

        
        out_str = ''

        # out_str += 'reference JD: {:.12f}\n'.format(self.jdt_ref)
        out_str += 'Input measurement type: '

        # Write out measurement type
        if self.meastype == 1:
            out_str += 'Right Ascension for meas1, Declination for meas2, epoch of date'

        elif self.meastype == 2:
            out_str += 'Azimuth +east of due north for meas1, Elevation angle above the horizon for meas2'

        elif self.meastype == 3:
            out_str += 'Azimuth +west of due south for meas1, Zenith angle for meas2'

        elif self.meastype == 4:
            out_str += 'Azimuth +north of due east for meas1, Zenith angle for meas2'

        
        out_str += "\n"

        out_str += "reference JD: {:20.12f}".format(self.jdt_ref)

        out_str += "\n\n"

        out_str += 'Plane intersections\n'
        out_str += '-------------------\n'

        # Print out all intersecting planes pairs
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
        out_str += " X =  {:11.2f}{:s} m\n".format(x, _uncer('{:.2f}', 'x'))
        out_str += " Y =  {:11.2f}{:s} m\n".format(y, _uncer('{:.2f}', 'y'))
        out_str += " Z =  {:11.2f}{:s} m\n".format(z, _uncer('{:.2f}', 'z'))
        out_str += " Vx = {:11.2f}{:s} m/s\n".format(vx, _uncer('{:.2f}', 'vx'))
        out_str += " Vy = {:11.2f}{:s} m/s\n".format(vy, _uncer('{:.2f}', 'vy'))
        out_str += " Vz = {:11.2f}{:s} m/s\n".format(vz, _uncer('{:.2f}', 'vz'))

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
            out_str += "{:>10s}: {:.6f} s\n".format(str(stat_id), t_diff)

        out_str += "\n"

        if self.orbit is not None:
            out_str += "Reference point on the trajectory:\n"
            out_str += "  Time: " + str(jd2Date(self.orbit.jd_ref, dt_obj=True)) + " UTC\n"
            out_str += "  Lon     = {:+>10.6f}{:s} deg\n".format(np.degrees(self.orbit.lon_ref), \
                _uncer('{:.4f}', 'lon_ref', deg=True))
            out_str += "  Lat     = {:+>10.6f}{:s} deg\n".format(np.degrees(self.orbit.lat_ref), \
                _uncer('{:.4f}', 'lat_ref', deg=True))
            out_str += "  Lat geo = {:+>10.6f}{:s} deg\n".format(np.degrees(self.orbit.lat_geocentric), \
                _uncer('{:.4f}', 'lat_geocentric', deg=True))
            out_str += "\n"

            # Write out orbital parameters
            out_str += self.orbit.__repr__(uncertanties=uncertanties)
            out_str += "\n"


            # Write out the orbital covariance matrix
            if self.state_vect_cov is not None:

                out_str += "Orbit covariance matrix:\n"
                out_str += "             e     ,     q (AU)   ,      Tp (JD) ,   node (rad) ,   peri (rad) ,    i (rad)\n"

                elements_list = ["e   ", "q   ", "Tp  ", "node", "peri", "i   "]

                for elem_name, line in zip(elements_list, self.orbit_cov):
                    line_list = [elem_name]

                    for entry in line:
                        line_list.append("{:+.6e}".format(entry))
                    
                    out_str += ", ".join(line_list) + "\n"

                out_str += "\n"


        out_str += "Jacchia fit on lag = -|a1|*exp(|a2|*t):\n"
        out_str += " a1 = {:.6f}\n".format(self.jacchia_fit[0])
        out_str += " a2 = {:.6f}\n".format(self.jacchia_fit[1])
        out_str += "\n"

        out_str += "Mean time residuals from time vs. length:\n"
        out_str += "  Station with reference time: {:s}\n".format(str(self.observations[self.t_ref_station].station_id))
        out_str += "  Avg. res. = {:.3e} s\n".format(self.timing_res)
        out_str += "  Stddev    = {:.2e} s\n".format(self.timing_stddev)
        out_str += "\n"

        out_str += "Begin point on the trajectory:\n"
        out_str += "  Lon = {:>12.6f}{:s} deg\n".format(np.degrees(self.rbeg_lon), _uncer('{:.4f}', 
            'rbeg_lon', deg=True))
        out_str += "  Lat = {:>12.6f}{:s} deg\n".format(np.degrees(self.rbeg_lat), _uncer('{:.4f}', 
            'rbeg_lat', deg=True))
        out_str += "  Ht  = {:>8.2f}{:s} m\n".format(self.rbeg_ele, _uncer('{:.2f}', 'rbeg_ele'))

        out_str += "End point on the trajectory:\n"
        out_str += "  Lon = {:>12.6f}{:s} deg\n".format(np.degrees(self.rend_lon), _uncer('{:.4f}', 
            'rend_lon', deg=True))
        out_str += "  Lat = {:>12.6f}{:s} deg\n".format(np.degrees(self.rend_lat), _uncer('{:.4f}', 
            'rend_lat', deg=True))
        out_str += "  Ht  = {:>8.2f}{:s} m\n".format(self.rend_ele, _uncer('{:.2f}', 'rend_ele'))
        out_str += "\n"

        ### Write information about stations ###
        ######################################################################################################
        out_str += "Stations\n"
        out_str += "--------\n"

        out_str += "        ID, Lon +E (deg), Lat +N (deg), Ele (m), Jacchia a1, Jacchia a2, Beg Ele (m),  End Ht (m), +/- Obs ang (deg), +/- V (m), +/- H (m), Incident angle (deg)\n"
        
        for obs in self.observations:

            station_info = []
            station_info.append("{:>10s}".format(str(obs.station_id)))
            station_info.append("{:>12.6f}".format(np.degrees(obs.lon)))
            station_info.append("{:>12.6f}".format(np.degrees(obs.lat)))
            station_info.append("{:>7.2f}".format(obs.ele))
            station_info.append("{:>10.6f}".format(obs.jacchia_fit[0]))
            station_info.append("{:>10.6f}".format(obs.jacchia_fit[1]))
            station_info.append("{:>11.2f}".format(obs.rbeg_ele))
            station_info.append("{:>11.2f}".format(obs.rend_ele))
            station_info.append("{:>17.6f}".format(np.degrees(obs.ang_res_std)))
            station_info.append("{:>9.2f}".format(obs.v_res_rms))
            station_info.append("{:>9.2f}".format(obs.h_res_rms))
            station_info.append("{:>20.2f}".format(np.degrees(obs.incident_angle)))


            out_str += ", ".join(station_info) + "\n"
        
        ######################################################################################################

        out_str += "\n"

        ### Write information about individual points ###
        ######################################################################################################
        out_str += "Points\n"
        out_str += "------\n"


        out_str += " No, Station ID,  Ignore,  Time (s),                   JD,     meas1,     meas2, Azim +E of due N (deg), Alt (deg), Azim line (deg), Alt line (deg), RA obs (deg), Dec obs (deg), RA line (deg), Dec line (deg),       X (m),       Y (m),       Z (m), Latitude (deg), Longitude (deg), Height (m),  Range (m), Length (m),  Lag (m), Vel (m/s), H res (m), V res (m), Ang res (asec)\n"

        # Go through observation from all stations
        for obs in self.observations:

            # Go through all observed points
            for i in range(obs.kmeas):

                point_info = []

                point_info.append("{:3d}".format(i))

                point_info.append("{:>10s}".format(str(obs.station_id)))

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
                point_info.append("{:8.2f}".format(obs.lag[i]))

                point_info.append("{:9.2f}".format(obs.velocities[i]))

                point_info.append("{:9.2f}".format(obs.h_residuals[i]))
                point_info.append("{:9.2f}".format(obs.v_residuals[i]))
                point_info.append("{:14.2f}".format(3600*np.degrees(obs.ang_res[i])))



                out_str += ", ".join(point_info) + "\n"


        ######################################################################################################


        out_str += "\n"

        out_str += "Notes\n"
        out_str += "-----\n"
        out_str += "- Points that have not been taken into consideratio when computing the trajectory have '1' in the 'Ignore' column.\n"
        out_str += "- The time already has time offsets applied to it.\n"
        out_str += "- 'meas1' and 'meas2' are given input points.\n"
        out_str += "- X, Y, Z are ECI (Earth-Centered Inertial) positions of projected lines of sight on the radiant line.\n"
        out_str += "- Zc is the observed zenith distance of the entry angle, while the Zg is the entry zenith distance corrected for Earth's gravity.\n"
        out_str += "- Latitude (deg), Longitude (deg), Height (m) are WGS84 coordinates of each point on the radiant line.\n"
        out_str += "- Jacchia (1955) deceleration equation fit was done on the lag.\n"
        out_str += "- Right ascension and declination in the table are given for the epoch of date for the corresponding JD, per every point.\n"
        out_str += "- 'RA and Dec obs' are the right ascension and declination calculated from the observed values, while the 'RA and Dec line' are coordinates of the lines of sight projected on the fitted radiant line. The coordinates are in the epoch of date, and NOT J2000!. 'Azim and alt line' are thus corresponding azimuthal coordinates.\n"

        if verbose:
            print(out_str)

        mkdirP(dir_path)

        # Save the report to a file
        with open(os.path.join(dir_path, file_name), 'w') as f:
            f.write(out_str)



    def savePlots(self, output_dir, file_name, show_plots=True):
        """ Show plots of the estimated trajectory. 
    
        Arguments:
            output_dir: [str] Path to the output directory.
            file_name: [str] File name which will be used for saving plots.

        Keyword_arguments:
            show_plots: [bools] Show the plots on the screen. True by default.

        """

        # Get the first reference time
        t0 = min([obs.time_data[0] for obs in self.observations])

        # Plot spatial residuals per observing station
        for obs in self.observations:

            ### PLOT SPATIAL RESIDUALS PER STATION ###
            ##################################################################################################


            # NOTE: It is possible that the gravity drop is not easily visible due to the perspective of the
            #   observer
            # # Calculate the gravity acceleration at every point
            # g = []
            # for eci in obs.model_eci:
            #     g.append(G*EARTH.MASS/(vectMag(eci)**2))

            # g = np.array(g)

            # # Generate gravity drop data
            # grav_drop = -np.sign(obs.time_data - t0)*1/2.0*g*(obs.time_data - t0)**2

            # # Plot the gravity drop
            # plt.plot(obs.time_data, grav_drop, c='red', linestyle='--', linewidth=1.0, label='Gravity drop')

            # Calculate root mean square of the residuals
            v_res_rms = RMSD(obs.v_residuals)
            #v_res_grav_rms = RMSD(obs.v_residuals - grav_drop)
            h_res_rms = RMSD(obs.h_residuals)

            # # Plot vertical residuals
            # plt.scatter(obs.time_data, obs.v_residuals, c='red', \
            #     label='Vertical, RMSD = {:.2f}\nw/ gravity, RMSD = {:.2f}'.format(v_res_rms, v_res_grav_rms), 
            #     zorder=3, s=2)

            # Plot vertical residuals
            plt.scatter(obs.time_data, obs.v_residuals, c='red', \
                label='Vertical, RMSD = {:.2f}'.format(v_res_rms), zorder=3, s=2)

            # Plot horizontal residuals
            plt.scatter(obs.time_data, obs.h_residuals, c='b', \
                label='Horizontal, RMSD = {:.2f}'.format(h_res_rms), zorder=3, s=2)

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

            plt.legend()

            # Set the residual limits to +/-10m if they are smaller than that
            if (np.max(np.abs(obs.v_residuals)) < 10) and (np.max(np.abs(obs.h_residuals)) < 10):
                plt.ylim([-10, 10])


            if self.save_results:
                savePlot(plt, file_name + '_' + str(obs.station_id) + '_spatial_residuals.png', \
                    output_dir)

            if show_plots:
                plt.show()

            else:
                plt.clf()
                plt.close()

            ##################################################################################################



        if self.plot_all_spatial_residuals:


            # Plot all spatial residuals (vs. time)
            for obs in self.observations:

                ### PLOT ALL SPATIAL RESIDUALS ###
                ##################################################################################################

                # Calculate root mean square of the residuals
                v_res_rms = RMSD(obs.v_residuals)
                h_res_rms = RMSD(obs.h_residuals)

                # Plot vertical residuals
                vres_plot = plt.scatter(obs.time_data, obs.v_residuals, marker='o', s=4, \
                    label='{:s}, vertical, RMSD = {:.2f}'.format(str(obs.station_id), v_res_rms), zorder=3)

                # Plot horizontal residuals
                plt.scatter(obs.time_data, obs.h_residuals, c=vres_plot.get_facecolor(), marker='+', \
                    label='{:s}, horizontal, RMSD = {:.2f}'.format(str(obs.station_id), h_res_rms), zorder=3)

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

            plt.legend()

            # Set the residual limits to +/-10m if they are smaller than that
            if np.max(np.abs(plt.gca().get_ylim())) < 10:
                plt.ylim([-10, 10])


            if self.save_results:
                savePlot(plt, file_name + '_all_spatial_residuals.png', output_dir)

            if show_plots:
                plt.show()

            else:
                plt.clf()
                plt.close()


            # Plot all spatial residuals (vs. length)
            for obs in self.observations:

                ### PLOT ALL SPATIAL RESIDUALS VS LENGTH ###
                ##############################################################################################

                # Calculate root mean square of the residuals
                v_res_rms = RMSD(obs.v_residuals)
                h_res_rms = RMSD(obs.h_residuals)

                # Plot vertical residuals
                vres_plot = plt.scatter(obs.state_vect_dist/1000, obs.v_residuals, marker='o', s=4, \
                    label='{:s}, vertical, RMSD = {:.2f}'.format(str(obs.station_id), v_res_rms), zorder=3)

                # Plot horizontal residuals
                plt.scatter(obs.state_vect_dist/1000, obs.h_residuals, c=vres_plot.get_facecolor(), 
                    marker='+', label='{:s}, horizontal, RMSD = {:.2f}'.format(str(obs.station_id), \
                        h_res_rms), zorder=3)

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

            plt.legend()

            # Set the residual limits to +/-10m if they are smaller than that
            if np.max(np.abs(plt.gca().get_ylim())) < 10:
                plt.ylim([-10, 10])


            if self.save_results:
                savePlot(plt, file_name + '_all_spatial_residuals_length.png', output_dir)

            if show_plots:
                plt.show()

            else:
                plt.clf()
                plt.close()


            ##################################################################################################




        # Plot lag per observing station
        for obs in self.observations:
            
            ### PLOT LAG ###
            ##################################################################################################

            fig, ax1 = plt.subplots()

            # Extract lag points that were not ignored
            used_times = obs.time_data[obs.ignore_list == 0]
            used_lag = obs.lag[obs.ignore_list == 0]

            if not obs.ignore_station:

                # Plot the lag
                ax1.plot(used_lag, used_times, color='r', marker='x', label='Lag', zorder=3)

                # Plot the Jacchia fit
                ax1.plot(jacchiaLagFunc(obs.time_data, *obs.jacchia_fit), obs.time_data, color='b', 
                    label='Jacchia fit', zorder=3)


            # Plot ignored lag points
            if np.any(obs.ignore_list):

                ignored_times = obs.time_data[obs.ignore_list > 0]
                ignored_lag = obs.lag[obs.ignore_list > 0]

                ax1.scatter(ignored_lag, ignored_times, c='k', marker='+', zorder=4, \
                    label='Lag, ignored points')

            
            ax1.legend()

            plt.title('Lag, station ' + str(obs.station_id))
            ax1.set_xlabel('Lag (m)')
            ax1.set_ylabel('Time (s)')

            ax1.set_ylim(min(obs.time_data), max(obs.time_data))

            ax1.grid()

            ax1.invert_yaxis()

            # Set the height axis
            ax2 = ax1.twinx()
            ax2.set_ylim(min(obs.meas_ht)/1000, max(obs.meas_ht)/1000)
            ax2.set_ylabel('Height (km)')

            plt.tight_layout()

            if self.save_results:
                savePlot(plt, file_name + '_' + str(obs.station_id) + '_lag.png', output_dir)

            if show_plots:
                plt.show()

            else:
                plt.clf()
                plt.close()


            ##################################################################################################

        ### PLOT ALL LAGS ###
        ######################################################################################################

        # Plot lags from each station on a single plot
        for obs in self.observations:

            # Extract lag points that were not ignored
            used_times = obs.time_data[obs.ignore_list == 0]
            used_lag = obs.lag[obs.ignore_list == 0]

            # Plot the lag
            plt_handle = plt.plot(used_lag, used_times, marker='x', label='Station: ' + str(obs.station_id), 
                zorder=3)


            # Plot ignored lag points
            if np.any(obs.ignore_list):

                ignored_times = obs.time_data[obs.ignore_list > 0]
                ignored_lag = obs.lag[obs.ignore_list > 0]

                plt.scatter(ignored_lag, ignored_times, facecolors='k', edgecolors=plt_handle[0].get_color(), 
                    marker='o', s=8, zorder=4, label='Station: {:s} ignored points'.format(str(obs.station_id)))




        # Plot the Jacchia fit on all observations
        time_all = np.sort(np.hstack([obs.time_data for obs in self.observations]))
        plt.plot(jacchiaLagFunc(time_all, *self.jacchia_fit), time_all, label='Jacchia fit', 
            zorder=3)


        plt.title('Lags, all stations')

        plt.xlabel('Lag (m)')
        plt.ylabel('Time (s)')

        plt.legend()
        plt.grid()
        plt.gca().invert_yaxis()

        if self.save_results:
            savePlot(plt, file_name + '_lags_all.png', output_dir)

        if show_plots:
            plt.show()

        else:
            plt.clf()
            plt.close()

        ######################################################################################################



        ### PLOT DISTANCE FROM RADIANT STATE VECTOR POSITION ###
        ######################################################################################################
        for obs in self.observations:

            # Extract points that were not ignored
            used_times = obs.time_data[obs.ignore_list == 0]
            used_dists = obs.state_vect_dist[obs.ignore_list == 0]

            plt_handle = plt.plot(used_dists/1000, used_times, marker='x', label=str(obs.station_id), \
                zorder=3)


            # Plot ignored points
            if np.any(obs.ignore_list):

                ignored_times = obs.time_data[obs.ignore_list > 0]
                ignored_dists = obs.state_vect_dist[obs.ignore_list > 0]
                    
                plt.scatter(ignored_dists/1000, ignored_times, facecolors='k', 
                    edgecolors=plt_handle[0].get_color(), marker='o', s=8, zorder=4, \
                    label='{:s} ignored points'.format(str(obs.station_id)))



        # Add the fitted velocity line
        if self.velocity_fit is not None:

            # Get time data range
            t_min = min([np.min(obs.time_data) for obs in self.observations])
            t_max = max([np.max(obs.time_data) for obs in self.observations])

            t_range = np.linspace(t_min, t_max, 100)

            plt.plot(lineFunc(t_range, *self.velocity_fit)/1000, t_range, label='Velocity fit', \
                linestyle='--', alpha=0.5, zorder=3)


        plt.title('Distances from state vector, Time residuals = {:.3e} s'.format(self.timing_res))

        plt.ylabel('Time (s)')
        plt.xlabel('Distance from state vector (km)')
        
        plt.legend()
        plt.grid()
        plt.gca().invert_yaxis()

        if self.save_results:
            savePlot(plt, file_name + '_lengths.png', output_dir)


        if show_plots:
            plt.show()

        else:
            plt.clf()
            plt.close()

        ######################################################################################################



        ### PLOT VELOCITY ###
        ######################################################################################################

        # Possible markers for velocity
        markers = ['x', '+', '.', '2']

        # Generate a list of colors to use for markers
        colors = cm.rainbow(np.linspace(0, 1 , len(self.observations)))

        fig, ax1 = plt.subplots()

        vel_max = -np.inf
        vel_min = np.inf

        ht_max = -np.inf
        ht_min = np.inf

        t_max = -np.inf
        t_min = np.inf

        
        first_ignored_plot = True

        # Plot velocities from each observed site
        for i, obs in enumerate(self.observations):

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


            # Plot all velocities
            ax1.scatter(obs.velocities[1:]/1000, obs.time_data[1:], marker=markers[i%len(markers)], 
                c=colors[i], alpha=0.5, label='Station: ' + str(obs.station_id), zorder=3)


            # Determine the max/min velocity and height, as this is needed for plotting both height/time axes
            vel_max = max(np.max(obs.velocities[1:]/1000), vel_max)
            vel_min = min(np.min(obs.velocities[1:]/1000), vel_min)

            ht_max = max(np.max(obs.meas_ht[1:]), ht_max)
            ht_min = min(np.min(obs.meas_ht[1:]), ht_min)

            t_max = max(np.max(obs.time_data[1:]), t_max)
            t_min = min(np.min(obs.time_data[1:]), t_min)


        # Plot the velocity calculated from the Jacchia model
        t_vel = np.linspace(t_min, t_max, 1000)
        ax1.plot(jacchiaVelocityFunc(t_vel, self.jacchia_fit[0], self.jacchia_fit[1], self.v_init)/1000, \
            t_vel, label='Jacchia fit', alpha=0.5)

        plt.title('Velocity')
        ax1.set_xlabel('Velocity (km/s)')
        ax1.set_ylabel('Time (s)')

        ax1.legend()
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

        if self.save_results:
            savePlot(plt, file_name + '_velocities.png', output_dir)

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


        # Put coordinate of all sites and the meteor in the one list
        lat_list = [obs.lat for obs in self.observations]
        lat_list.append(met_lat_mean)
        lon_list = [obs.lon for obs in self.observations]
        lon_list.append(met_lon_mean)


        # Init the map
        m = GroundMap(lat_list, lon_list, border_size=50, color_scheme='light')


        # Plot locations of all stations and measured positions of the meteor
        for obs in self.observations:

            # Plot stations
            m.scatter(obs.lat, obs.lon, s=10, label=str(obs.station_id), marker='x')

            # Plot measured points
            m.plot(obs.meas_lat, obs.meas_lon, c='r')


        # Plot a point marking the final point of the meteor
        m.scatter(self.rend_lat, self.rend_lon, c='y', marker='+', s=50, alpha=0.75, label='Endpoint')


        plt.legend(loc='upper right')


        if self.save_results:
            savePlot(plt, file_name + '_ground_track.png', output_dir)

        if show_plots:
            plt.show()

        else:
            plt.clf()
            plt.close()

        ######################################################################################################


        # Compare original and modeled measurements (residuals in azimuthal coordinates)
        for obs in self.observations:

            # Calculate residuals in arcseconds
            res = np.degrees(obs.ang_res)*3600

            # Mark ignored points
            if np.any(obs.ignore_list):

                ignored_times = obs.time_data[obs.ignore_list > 0]
                ignored_residuals = res[obs.ignore_list > 0]

                plt.scatter(ignored_times, ignored_residuals, facecolors='none', edgecolors='k', s=20, \
                    zorder=4, label='Ignored points')


            # Calculate the RMSD of the residuals in arcsec
            res_rms = np.degrees(obs.ang_res_std)*3600

            # Plot residuals
            plt.scatter(obs.time_data, res, label='Angle, RMSD = {:.2f}"'.format(res_rms), s=2, zorder=3)


            plt.title('Observed vs. Radiant LoS Residuals, station ' + str(obs.station_id))
            plt.ylabel('Angle (arcsec)')
            plt.xlabel('Time (s)')

            # The lower limit is always at 0
            plt.ylim(ymin=0)

            plt.grid()
            plt.legend()

            if self.save_results:
                savePlot(plt, file_name + '_' + str(obs.station_id) + '_angular_residuals.png', \
                output_dir)

            if show_plots:
                plt.show()

            else:
                plt.clf()
                plt.close()


        # Plot angular residuals from all stations
        first_ignored_plot = True
        for obs in self.observations:

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
            plt.scatter(obs.time_data, res, s=2, zorder=3, label='Station ' + str(obs.station_id) + \
                ', RMSD = {:.2f}"'.format(res_rms))


        plt.title('Observed vs. Radiant LoS Residuals, all stations')
        plt.ylabel('Angle (arcsec)')
        plt.xlabel('Time (s)')

        # The lower limit is always at 0
        plt.ylim(ymin=0)

        plt.grid()
        plt.legend()

        if self.save_results:
            savePlot(plt, file_name + '_all_angular_residuals.png', output_dir)

        if show_plots:
            plt.show()

        else:
            plt.clf()
            plt.close()



        # Plot the orbit in 3D
        if self.calc_orbit:

            # Check if the orbit was properly calculated
            if self.orbit.ra_g is not None:

                # Construct a list of orbital elements of the meteor
                orbit_params = np.array([
                    [self.orbit.a, self.orbit.e, np.degrees(self.orbit.i), np.degrees(self.orbit.peri), \
                        np.degrees(self.orbit.node)]
                    ])

                # Run orbit plotting procedure
                plotOrbits(orbit_params, jd2Date(self.jdt_ref, dt_obj=True), save_plots=self.save_results, \
                    plot_path=os.path.join(output_dir, file_name), linewidth=1, color_scheme='light')


                plt.tight_layout()


                if show_plots:
                    plt.show()

                else:
                    plt.clf()
                    plt.close()



    def showLoS(self):
        """ Show the stations and the lines of sight solution. """


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Calculate the position of the state vector (aka. first point on the trajectory)
        traj_point = self.observations[0].model_eci[0]

        # Calculate the length to the last point on the trajectory
        meteor_len = np.sqrt(np.sum((self.observations[0].model_eci[0] - self.observations[0].model_eci[-1])**2))

        # Calculate the plot limits
        x_list = [x_stat for obs in self.observations for x_stat in obs.stat_eci_los[:, 0]]
        x_list.append(traj_point[0])
        y_list = [y_stat for obs in self.observations for y_stat in obs.stat_eci_los[:, 1]]
        y_list.append(traj_point[1])
        z_list = [z_stat for obs in self.observations for z_stat in obs.stat_eci_los[:, 2]]
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
            ax.scatter(obs.stat_eci_los[:, 0], obs.stat_eci_los[:, 1], obs.stat_eci_los[:, 2], s=20)

            # Plot lines of sight
            for stat_eci_los, meas_eci_los in zip(obs.stat_eci_los, obs.meas_eci_los):

                # Calculate the point on the trajectory
                traj_pt, _, _ = findClosestPoints(stat_eci_los, meas_eci_los, self.state_vect_mini, 
                    self.radiant_eci_mini)

                vect_len = np.sqrt(np.sum((stat_eci_los - traj_pt)**2))

                # Lines of sight
                ax.quiver(stat_eci_los[0], stat_eci_los[1], stat_eci_los[2], meas_eci_los[0], meas_eci_los[1], 
                    meas_eci_los[2], length=vect_len, normalize=True, arrow_length_ratio=0, color='blue', 
                    alpha=0.5)



        # Plot the radiant state vector
        rad_x, rad_y, rad_z = -self.radiant_eci_mini
        rst_x, rst_y, rst_z = traj_point
        ax.quiver(rst_x, rst_y, rst_z, rad_x, rad_y, rad_z, length=meteor_len, normalize=True, color='red', 
            arrow_length_ratio=0.1)

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])


        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

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
        t0 = min([obs.time_data[0] for obs in self.observations])

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

        # Calculate all plane intersections in between all station pairs
        for i, obs1 in enumerate(self.observations):
            for j, obs2 in enumerate(self.observations[i + 1:]):

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
        self.incident_angles = self.calcStationIncidentAngles(self.state_vect, self.best_conv_inter.radiant_eci, \
            self.observations)

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


        # Set weights for stations that are not used to 0
        weights = [w if (self.observations[i].ignore_station == False) else 0 for i, w in enumerate(weights)]



        if self.verbose:
            print('LoS statistical weights:')

            for i, obs in enumerate(self.observations):
                print(obs.station_id, weights[i])

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


        # If running a Monte Carlo run, switch the observations to the original ones, so noise does not 
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
        self.calcVelocity(self.state_vect_mini, self.radiant_eci_mini, self.observations, 
            calc_res=_rerun_timing)


        # Calculate the lag if it was not calculated during timing estimation
        if self.observations[0].lag is None:

            # Calculate lag
            self.calcLag(self.observations)


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

        #         plt_handle = plt.plot(used_dists, used_times, marker='x', label=str(obs.station_id), zorder=3)


        #         # Plot ignored points
        #         if np.any(obs.ignore_list):

        #             ignored_times = obs.time_data[obs.ignore_list > 0]
        #             ignored_dists = obs.state_vect_dist[obs.ignore_list > 0]
                        
        #             plt.scatter(ignored_dists, ignored_times, facecolors='k', edgecolors=plt_handle[0].get_color(), 
        #                 marker='o', s=8, zorder=4, label='{:s} ignored points'.format(str(obs.station_id)))


        #     plt.title("Distances from state vector, before time correction")

        #     plt.ylabel('Time (s)')
        #     plt.xlabel('Distance from state vector (m)')
            
        #     plt.legend()
        #     plt.grid()
        #     plt.gca().invert_yaxis()

        #     plt.tight_layout()

        #     plt.show()


        # Estimate the timing difference between stations and the initial velocity
        self.velocity_fit, self.v_init, self.time_diffs, self.observations = \
            self.estimateTimingAndVelocity(self.observations, estimate_timing_vel=self.estimate_timing_vel)

        self.time_diffs_final = self.time_diffs



        # Calculate velocity at each point with updated timings
        self.calcVelocity(self.state_vect_mini, self.radiant_eci_mini, self.observations, 
            calc_res=_rerun_timing)


        ### RERUN THE TRAJECTORY ESTIMATION WITH UPDATED TIMINGS ###
        ######################################################################################################

        # Runs only in the first pass of trajectory estimation and estimates timing offsets between stations
        if not _rerun_timing:

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
                    print("Updating the solution after the timing estimation...")

                # Reinitialize the observations with proper timing
                for obs in temp_observations:
            
                    self.infillTrajectory(obs.meas1, obs.meas2, obs.time_data, obs.lat, obs.lon, obs.ele, \
                        station_id=obs.station_id, excluded_time=obs.excluded_time, \
                        ignore_list=obs.ignore_list)

                
                # Re-run the trajectory estimation with updated timings. This will update all calculated
                # values up to this point
                self.run(_rerun_timing=True, _prev_toffsets=self.time_diffs, _orig_obs=_orig_obs)


        else:

            # In the second pass, calculate the final timing offsets
            if _prev_toffsets is not None:
                self.time_diffs_final = _prev_toffsets + self.time_diffs

            else:
                self.time_diffs_final = self.time_diffs

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



        # Do a Jacchia exponential fit to the lag, per every station
        self.jacchia_fit = self.fitJacchiaLag(self.observations)

        # Calculate latitude, longitude and altitude of each point closest to the radiant line, in WGS84
        self.calcLLA(self.state_vect_mini, self.radiant_eci_mini, self.observations)

        # Calculate ECI positions of the CPA on the radiant line, RA and Dec of the points on the radiant
        # line as seen by the observers, the corresponding azimuth and elevation, and set arrays model_fit1 
        # and model_fit2 to be of the same type as the input parameters meas1 and meas2
        self.calcECIEqAltAz(self.state_vect_mini, self.radiant_eci_mini, self.observations)


        # Calculate horizontal, vertical and angular residuals from the lines of sight to the radiant line
        self.calcAllResiduals(self.state_vect_mini, self.radiant_eci_mini, self.observations)


        ### REMOVE BAD PICKS AND RECALCULATE ###
        ######################################################################################################

        if self.filter_picks:
            if (not _rerun_bad_picks):

                picks_rejected = 0

                # Remove all picks which deviate more than 3 sigma in angular residuals
                for obs in self.observations:

                    # Find the indicies of picks which are within 3 sigma
                    good_picks = np.argwhere(obs.ang_res < (np.mean(obs.ang_res) + 3*obs.ang_res_std)).ravel()

                    # If the number of good picks is below 4, do not remove any picks
                    if len(good_picks) < 4:
                        continue

                    # Check if any picks were removed
                    if len(good_picks) < len(obs.ang_res):
                        picks_rejected += len(obs.ang_res) - len(good_picks)

                        # Take only the good picks
                        obs.time_data = obs.time_data[good_picks]
                        obs.meas1 = obs.meas1[good_picks]
                        obs.meas2 = obs.meas2[good_picks]
                        obs.ignore_list = obs.ignore_list[good_picks]


                # Run only if some picks were rejected
                if picks_rejected:

                    # Make a copy of observations
                    temp_observations = copy.deepcopy(self.observations)
                    
                    # Reset the observation points
                    self.observations = []

                    print("Updating the solution after rejecting", picks_rejected, "bad picks...")

                    # Reinitialize the observations without the bad picks
                    for obs in temp_observations:
                
                        self.infillTrajectory(obs.meas1, obs.meas2, obs.time_data, obs.lat, obs.lon, \
                            obs.ele, station_id=obs.station_id, excluded_time=obs.excluded_time,
                            ignore_list=obs.ignore_list)

                    
                    # Re-run the trajectory estimation with updated timings. This will update all calculated
                    # values up to this point
                    self.run(_rerun_bad_picks=True)

                else:
                    print("All picks are within 3 sigma...")


            else:

                # In the second pass, return None
                return None

        ######################################################################################################


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
                print(self.orbit)


        ######################################################################################################


        # Break if doing a Monte Carlo run
        if _mc_run:
            return None


        if self.monte_carlo:

            # Do a Monte Carlo estimate of the uncertanties in all calculated parameters
            traj_best, uncertanties = monteCarloTrajectory(self, mc_runs=self.mc_runs, 
                mc_pick_multiplier=self.mc_pick_multiplier, noise_sigma=self.mc_noise_std)


            ### Save uncertainties to the trajectory object ###
            if uncertanties is not None:
                traj_uncer = copy.deepcopy(uncertanties)

                # Remove the list of all MC trajectires (it is unecessarily big)
                traj_uncer.mc_traj_list = []

                # Set the uncertanties to the best trajectory
                traj_best.uncertanties = traj_uncer

            ######


        else:
            uncertanties = None


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
                    
                    # Save the uncertanties
                    savePickle(uncertanties, mc_output_dir, mc_file_name + '_uncertanties.pickle')

                    # Save trajectory report
                    traj_best.saveReport(mc_output_dir, mc_file_name + '_report.txt', \
                        uncertanties=uncertanties, verbose=True)

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
                        uncertanties=uncertanties, verbose = not self.monte_carlo)


            # Save and show plots
            self.savePlots(self.output_dir, self.file_name, \
                show_plots=(self.show_plots and not self.monte_carlo))
            

        ######################################################################################################



        ## SHOW PLANE INTERSECTIONS AND LoS PLOTS ###
        #####################################################################################################

        # # Show the plane intersection
        # if self.show_plots:
        #     self.best_conv_inter.show()


        # # Show lines of sight solution in 3D
        # if self.show_plots:
        #     self.showLoS()


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
    from Utils.TrajConversions import equatorialCoordPrecession_vect, J2000_JD

    ## TEST EVENT 1
    ###############
    # reference julian date
    jdt_ref = 2457660.770667

    meastype = 4

    # Measurements
    time1 = np.array([0.057753086090087891, 0.066874027252197266, 0.075989007949829102, 0.085109949111938477, 0.094237089157104492, 0.10335803031921387, 0.11248111724853516, 0.12160706520080566, 0.13072991371154785, 0.1398470401763916, 0.14896798133850098, 0.1580970287322998, 0.16721701622009277, 0.17634010314941406, 0.18546104431152344, 0.19459104537963867, 0.20371103286743164, 0.21282792091369629, 0.2219550609588623, 0.23107600212097168, 0.24019694328308105, 0.24931812286376953, 0.25844597816467285, 0.26756501197814941, 0.27669310569763184, 0.28580904006958008, 0.29493308067321777, 0.30405712127685547, 0.31317901611328125, 0.32230591773986816, 0.33142495155334473, 0.34055089950561523, 0.34967303276062012, 0.35879397392272949, 0.36792206764221191, 0.37704110145568848, 0.38615989685058594, 0.39528894424438477, 0.40440893173217773, 0.41353106498718262, 0.42265510559082031, 0.43178009986877441, 0.44089889526367188, 0.45002102851867676, 0.45915102958679199, 0.46827292442321777, 0.47739696502685547, 0.4865109920501709, 0.4956510066986084, 0.50475692749023438, 0.51387810707092285, 0.52300906181335449, 0.53212499618530273, 0.54124712944030762, 0.55037498474121094, 0.55949711799621582, 0.56861710548400879, 0.57773995399475098, 0.58686208724975586, 0.59599995613098145, 0.60510897636413574, 0.6142280101776123, 0.62335801124572754, 0.6324760913848877])
    phi1 = np.array([55.702480827431032, 55.793824368465614, 55.88753020599011, 55.980570544693705, 56.07327845058068, 56.16663811716176, 56.260021035671755, 56.351956828609609, 56.44505503179294, 56.538332186739993, 56.632552238675849, 56.725387680018272, 56.818000654246454, 56.911201723248155, 57.004115910036212, 57.097832261372453, 57.191412597398575, 57.283977960828018, 57.376746480149919, 57.470141289434302, 57.563543438121414, 57.656628924724671, 57.749937800483188, 57.842688174097987, 57.935406662789603, 58.028832186692952, 58.121575350363329, 58.214494019816144, 58.307473172753141, 58.400174495678399, 58.493023084887334, 58.586384750506248, 58.678949326179911, 58.771357833168224, 58.863850501356033, 58.956860447376158, 59.049171725989119, 59.141616797438303, 59.233887951111122, 59.326106753286169, 59.41819565787236, 59.510388361622056, 59.602333162218116, 59.694591678426015, 59.786106749232736, 59.877491613135611, 59.969101239628593, 60.06002031553944, 60.151680627553716, 60.242091556272477, 60.33297244327273, 60.423181062857637, 60.513186887636216, 60.602894553039164, 60.692293889528756, 60.780956442762005, 60.870177557670779, 60.958943574792976, 61.046938105117917, 61.134351245894756, 61.221159385330608, 61.306529365426798, 61.391267208416664, 61.467270929128503])
    theta1 = np.array([120.26319138247609, 120.22540695789934, 120.18673532370377, 120.14842794035015, 120.11034570859977, 120.0720842731949, 120.03390164839767, 119.99639652474744, 119.95850343282032, 119.92062401795023, 119.88244908946078, 119.84492055599212, 119.80756591352674, 119.77005822821938, 119.73274953746579, 119.69520274362065, 119.65779412056121, 119.6208730070672, 119.58395197967953, 119.54686324407939, 119.50985294580174, 119.47304858913093, 119.43623605212085, 119.39972291168513, 119.3633006386402, 119.32667935949192, 119.2904032683148, 119.25413572335212, 119.21792146205601, 119.1818915371656, 119.14588012264632, 119.10974570224057, 119.07399457396869, 119.03837757153748, 119.00280158839186, 118.96710031996264, 118.93173985985386, 118.89640024180805, 118.86119863355532, 118.82608798178681, 118.79109719190022, 118.75613703041395, 118.72134029918524, 118.6864941253839, 118.65199692326831, 118.61761616580844, 118.5832180249274, 118.54914529743198, 118.51486108335325, 118.48110902498068, 118.44724605387167, 118.41369719617806, 118.38028657342608, 118.34704871121083, 118.31398640380742, 118.28125669697118, 118.24838089786711, 118.21573228793197, 118.18342568415328, 118.15138963972852, 118.11963134239126, 118.08845334007034, 118.05755902182497, 118.02989359016864])

    time2 = np.array([0.0, 0.0091240406036376953, 0.018245935440063477, 0.027379035949707031, 0.036490917205810547, 0.045619010925292969, 0.05474090576171875, 0.063858985900878906, 0.072983026504516602, 0.082114934921264648, 0.091231107711791992, 0.10035109519958496, 0.10947489738464355, 0.11859893798828125, 0.12772512435913086, 0.13684391975402832, 0.14596700668334961, 0.15510010719299316, 0.16421103477478027, 0.17334103584289551, 0.18245911598205566, 0.19158196449279785, 0.20070290565490723, 0.20982694625854492, 0.2189481258392334, 0.22807097434997559, 0.23719310760498047, 0.24631595611572266, 0.25544309616088867, 0.26456212997436523, 0.27368307113647461, 0.28281092643737793, 0.29193806648254395, 0.30105209350585938, 0.31017804145812988, 0.31929898262023926, 0.32842206954956055, 0.33754897117614746, 0.34666705131530762, 0.3557898998260498, 0.36492395401000977, 0.37403392791748047, 0.38316202163696289, 0.39228296279907227, 0.40140891075134277, 0.41053390502929688, 0.41965007781982422, 0.42877292633056641, 0.43789410591125488, 0.44702005386352539, 0.45614290237426758, 0.46526408195495605, 0.47438502311706543, 0.48351693153381348, 0.49264097213745117, 0.50175309181213379, 0.51088404655456543, 0.52000308036804199, 0.52912497520446777, 0.53824996948242188, 0.54737210273742676, 0.55649089813232422, 0.56562089920043945, 0.5747380256652832, 0.58387494087219238, 0.592987060546875, 0.60210895538330078, 0.61122798919677734, 0.62035298347473145, 0.62947607040405273])
    phi2 = np.array([53.277395606543514, 53.378894622674743, 53.479956569118926, 53.581583564486643, 53.684034387407628, 53.785592745520816, 53.888221858788505, 53.989652095989705, 54.091286379162753, 54.193602941001174, 54.295489508972871, 54.39680492261197, 54.498476109777449, 54.600324950506916, 54.701540003200897, 54.803176858628973, 54.905770160432461, 55.007812728006726, 55.109255891578165, 55.210470634952003, 55.311514652098822, 55.413530094031998, 55.515323573286715, 55.616651349503798, 55.718365072619598, 55.81929890161981, 55.920171553847844, 56.021613048812512, 56.122821258097112, 56.224678899349627, 56.325865881424491, 56.426926299896216, 56.52861756575669, 56.629470224659684, 56.730172326265581, 56.831015465257991, 56.932197458064081, 57.033194520779368, 57.133991458819061, 57.234684773453658, 57.334955465097238, 57.435791110725937, 57.536108210586804, 57.63636328763743, 57.736907767451896, 57.837175586955425, 57.937203809457536, 58.036781893703278, 58.136978754564268, 58.236686044195643, 58.336377908906051, 58.43535465814314, 58.534625554011399, 58.6333660935654, 58.732691095623927, 58.831079484821906, 58.928886668384948, 59.026971367888081, 59.124250755486784, 59.221517041956538, 59.318507836373392, 59.414909920684529, 59.512434568208263, 59.608138099768297, 59.703628259049658, 59.799124823225796, 59.891752747734891, 59.983964699509606, 60.075068063440163, 60.161279720285414])
    theta2 = np.array([101.53457826463746, 101.51641844358608, 101.49838475808278, 101.48029817266169, 101.46211330853345, 101.44413445103872, 101.42601386963629, 101.40815190889943, 101.3903005255073, 101.37237603253595, 101.35457316211422, 101.3369156085497, 101.3192413904863, 101.30158154144411, 101.28407617303588, 101.26654229652355, 101.24888830028594, 101.23137351945253, 101.21400528102164, 101.19671926225118, 101.17950508630588, 101.16216841059016, 101.14491224554251, 101.12777721128521, 101.1106189746513, 101.09363369919245, 101.07669966495487, 101.05971116018192, 101.04280246671952, 101.02582610588094, 101.00900182871261, 100.99223844387629, 100.97541036501492, 100.95876039276899, 100.94217411607163, 100.92560326192354, 100.90901536078817, 100.89249613221422, 100.87604760647557, 100.85965363121295, 100.84336561880852, 100.82702299646236, 100.81080116796716, 100.79462576794111, 100.77843999684811, 100.76233476558444, 100.74630362442984, 100.73037973345876, 100.71439203286904, 100.69851722922239, 100.68267935704768, 100.66698899065921, 100.65128571495896, 100.63569963309027, 100.62005459779738, 100.60458982468012, 100.58924850257456, 100.57389559776691, 100.5587001573241, 100.5435378536578, 100.52844926863388, 100.51348253940968, 100.49837205688057, 100.48357341370303, 100.46883689073712, 100.45412830754735, 100.43988903856147, 100.42574044202429, 100.41178798399015, 100.39860839656382])

    # Convert measurement to radians
    theta1 = np.radians(theta1)
    phi1 = np.radians(phi1)

    theta2 = np.radians(theta2)
    phi2 = np.radians(phi2)

    ###

    ### SITES INFO

    lon1 = np.radians(-80.772090)
    lat1 = np.radians(43.264200)
    ele1 = 329.0

    lon2 = np.radians(-81.315650)
    lat2 = np.radians(43.192790)
    ele2 = 324.0

    ###


    # Init new trajectory solving
    traj_solve = Trajectory(jdt_ref, meastype=meastype, save_results=False, monte_carlo=False)

    # Set input points for the first site
    traj_solve.infillTrajectory(theta1, phi1, time1, lat1, lon1, ele1)

    # Set input points for the second site
    traj_solve.infillTrajectory(theta2, phi2, time2, lat2, lon2, ele2)

    traj_solve.run()

    ###############