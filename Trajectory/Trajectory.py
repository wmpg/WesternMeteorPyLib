""" PyLIG trajectory solver

Estimates meteor trajectory from given observed points. 

"""

from __future__ import print_function, division, absolute_import

import copy
import sys
import os
from operator import attrgetter

import numpy as np
import scipy.optimize
import scipy.interpolate

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


from Trajectory.Orbit import calcOrbit

from Utils.Math import vectNorm, vectMag, meanAngle, findClosestPoints
from Utils.OSTools import mkdirP
from Utils.Pickling import savePickle
from Utils.Plotting import savePlot
from Utils.PlotOrbits import plotOrbits
from Utils.TrajConversions import ecef2ENU, enu2ECEF, geo2Cartesian, geo2Cartesian_vect, cartesian2Geo, \
    altAz2RADec_vect, raDec2AltAz, raDec2AltAz_vect, raDec2ECI, eci2RaDec, jd2Date




class ObservedPoints(object):
    """ Structure for containing data of observations from invidiual stations. """

    def __init__(self, jdt_ref, meas1, meas2, time_data, lat, lon, ele, meastype, station_id=None):
        """ Init the container structure for observations. 
        
        Arguments:
            jdt_ref: [float] Reference Julian date/time that the measurements times are provided relative to. 
                    This is user selectable and can be the time of the first camera, or the first measurement, 
                    or some average time for the meteor, but should be close to the time of passage of the 
                    meteor. This same reference date/time will be used on all camera measurements for the 
                    purposes of computing local sidereal time and making  geocentric coordinate 
                    transformations.
            meas1: [list or ndarray] First measurement array (azimuth or R.A., depending on meastype, see 
                meastype documentation for more information). Measurements should be given in radians.
            meas2: [list or ndarray] Second measurement array (altitude, zenith angle or declination, 
                depending on meastype, see meastype documentation for more information), in radians.
            time_data: [list or ndarray] Time in seconds from the referent Julian date.
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
            station_id: [str] Identification of the station. None by default.

        Return:
            None

        """

        ### INPUT DATA ###
        ######################################################################################################

        self.meas1 = meas1
        self.meas2 = meas2

        # Referent Julian date
        self.jdt_ref = jdt_ref

        self.time_data = time_data

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

        # ECI coordinates of observed CPA to the radiant line, with the station moving in time
        self.meas_eci_los = None

        # ECI coordinates of radiant CPA to the observed line of sight
        self.model_eci = None

        # Arrays for geo coordinates of closest points of approach of observed lines of sight to the radiant line
        self.meas_lat = None
        self.meas_lon = None
        self.meas_ht = None
        self.meas_range = None

        # Arrays for geo coordinates of closest points of approach of the radiant line to the observed lines of sight
        self.model_lat = None
        self.model_lon = None
        self.model_ht = None
        self.model_range = None

        # Coordinates of the first point (observed)
        self.rbeg_lat = None
        self.rbeg_lon = None
        self.rbeg_ele = None

        # Coordinates of the last point (observed)
        self.rend_lat = None
        self.rend_lon = None
        self.rend_ele = None

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

        # Calculate position of the station in ECI coordinates (only for the referent JD, used for 
        # intersecting planes solution)
        self.x_stat, self.y_stat, self.z_stat = geo2Cartesian(self.lat, self.lon, self.ele, self.jdt_ref)
        self.stat_eci = np.array([self.x_stat, self.y_stat, self.z_stat])

        # Calculate positions of the station in ECI coordinates, for each JD of individual measurements
        # (used for the lines of sight least squares approach)
        self.stat_eci_los = np.array(geo2Cartesian_vect(self.lat, self.lon, self.ele, self.JD_data)).T

        # Fit a plane through the given points
        self.plane_N = self.planeFit()


        # ### PLOT RESULTS

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # ax.scatter(self.x_stat, self.y_stat, self.z_stat, s=50)
        # ax.scatter(self.x_stat + self.x_eci, self.y_stat + self.y_eci, self.z_stat + self.z_eci, c='red')

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

        # Let the JD data be fixed to the referent time - this is done because initally the azimuthal to 
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
        #self.x_eci_los, self.y_eci_los, self.z_eci_los = self.meas_eci_los.T
        
        ### USED IN GURAL SOLVER FOR ADDING THE NOISE TO ECI COORDINATES
        # for kmeas, eci_coord in enumerate(self.meas_eci):
            
        #     zhat = np.zeros(shape=3)
            
        #     # Southern hemisphere    
        #     if eci_coord[2] < 0:
                
        #         zhat[2] = +1.0
        #         uhat = np.cross(eci_coord, zhat)
        #         uhat = uhat/np.linalg.norm(uhat)

        #     # Northern hemisphere
        #     else:
        #         zhat[2] = -1.0
        #         uhat = np.cross(zhat, eci_coord)
        #         uhat = uhat/np.linalg.norm(uhat)

        #     vhat = np.cross(uhat, eci_coord)
        #     vhat = vhat/np.linalg.norm(vhat)

        #     # Calculate the unit vector pointing from the station camera to the observed point
        #     meas_eci = eci_coord + (GAUSS NOISE???)*uhat + (GAUSS NOISE??)*vhat
        #     meas_eci = meas_eci/np.linalg.norm(meas_eci)

        #     print('Meas ECI:', meas_eci)
        
        



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
    """ Handles intersection of a pair of planes. """

    def __init__(self, obs1, obs2):
        """ Calculate the plane intersection between two stations. 
            
        Arguments:
            obs1
            obs2
        Return:
            None

        """

        self.obs1 = obs1
        self.obs2 = obs2

        # Calculate the observed angular length of the track from the first station
        obsangle1 = np.arccos(np.dot(self.obs1.meas_eci[0], self.obs1.meas_eci[-1]))

        # Calculate the observed angular length of the track from the second station
        obsangle2 = np.arccos(np.dot(self.obs2.meas_eci[0], self.obs2.meas_eci[-1]))


        ### Calculate the angle between the pair of planes (convergence angle)
        
        # Calculate the cosine of the convergence angle
        ang_cos = np.dot(self.obs1.plane_N, self.obs2.plane_N)

        # Make sure the cosine is in the proper range
        self.conv_angle = np.arccos(np.abs(np.clip(ang_cos, -1, 1)))

        ###


        # Calculate the ECI coordinates of the plane intersection
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



def angleSumMeasurements2Line(observations, state_vect, best_conv_radiant_eci):
    """ Sum all angles between the radiant line and measurement lines of sight.

        This function is used as a cost function for the least squares radiant solution of Borovicka et 
        al. (1990). The difference from the original approach is that the distancesfrom the radiant line
        have been replaced with angles.

    Arguments:


    Return:

    """

    angle_sum = 0

    # Go through all observations from all stations
    for obs in observations:
        
        # Go through all measured positions
        for meas_eci, stat_eci in zip(obs.meas_eci_los, obs.stat_eci_los):

            # Get the position of the projection of the measurement line of sight on the radiant line,
            # measured from the state vector
            _, r, _ = findClosestPoints(stat_eci, meas_eci, state_vect, best_conv_radiant_eci)
            r = r - stat_eci
            r = vectNorm(r)

            # Calculate the angle between the measurement line of sight as seen from the station
            cosangle = np.dot(meas_eci, r)

            # Make sure the cosine is within limits and calculate the angle
            angle_sum += np.arccos(np.clip(cosangle, -1, 1))


    return angle_sum



def minimizeAngleCost(params, observations):
    """ A helper function for minimization of angle deviations. """

    state_vect, best_conv_radiant_eci = np.hsplit(params, 2)
    
    return angleSumMeasurements2Line(observations, state_vect, best_conv_radiant_eci)




def calcResidual(jd, state_vect, radiant_eci, stat, meas):
    """ Calculate horizontal and vertical residuals from the radiant line, for the given observed point.

    Arguments:
        jd: [float] Julian date
        state_vect: [3 element ndarray] ECI position of the state vector
        radiant_eci: [3 element ndarray] radiant direction vector in ECI
        stat: [3 element ndarray] position of the station in ECI
        meas: [3 element ndarray] line of sight from the station, in ECI

    Return:
        (hres, vres): [tuple of floats] residuals in horitontal and vertical direction from the radiant line

    """

    meas = vectNorm(meas)

    # Calculate closest points of approach (observed line of sight to radiant line)
    obs_cpa, rad_cpa, d = findClosestPoints(stat, meas, state_vect, radiant_eci)

    # Vector pointing from the point on the trajectory to the point on the line of sight
    p = obs_cpa - rad_cpa

    # Calculate geographical coordinates of the state vector position
    lat, lon, elev = cartesian2Geo(jd, *state_vect)

    # Calculate ENU (East, North, Up) vector in the position of the state vector, and direction of the radiant
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
    vres = -np.sign(ehx)*np.hypot(ehx, ehy)

    # Calculate horizontal residuals
    hres = -np.dot(p, evert)

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



def jacchiaLagFunc(t, a1, a2):
    """ Jacchia (1955) model for modelling lengths along the trail of meteors, modified to fit the lag (length 
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



def jacchiaVelocityFunc(t, a1, a2, v_init):
    """ Derivation of the Jacchia (1955) model, used for calculating velocities from the fitted model. 
    
    Arguments:
        t: [float or ndarrray] time in seconds at which the Jacchia function will be evaluated
        a1: [float] 1st acceleration term
        a2: [float] 2nd acceleration term
        v_init: [float] initial velocity in m/s

    Return:
        [float] velocity at time t

    """

    return v_init + -np.abs(a1*a2)*np.exp(np.abs(a2)*t)



def fitLagIntercept(time, length, v_init, initial_intercept=0.0):
    """ Finds the intercept of the line with the given slope. Used for fitting time vs. length along the trail
        data.

    Arguments:
        time: [ndarray] array containing the time data (seconds)
        length: [ndarray] array containing the length along the trail data
        v_init: [float] fixed slope of the line (i.e. initial velocity)

    Keyword arguments:
        initial_intercept: [float] initial estimate of the intercept

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




def timingAndVelocityResiduals(params, observations, t_ref_station):
    """ Calculate the sum of absolute differences between the lag of the referent station and all other 
        stations, by using the given initial velocity and timing differences between stations. 
    
    Arguments:
        params: [ndarray] first element is the initial velocity, all others are timing differences from the 
            referent station (NOTE: referent station is NOT in this list)
        observations: [list] a list of ObservedPoints objects
        t_ref_station: [int] index of the referent station
    
    Return:
        [float] sum of absolute differences between the referent and lags of all stations
    """

    stat_count = 0

    # The first parameters is the initial velocity
    v_init = params[0]

    lags = []

    # Go through observations from all stations
    for i, obs in enumerate(observations):

        # Time difference is 0 for the referent statins
        if i == t_ref_station:
            t_diff = 0

        else:
            # Take the estimated time difference for all other stations
            t_diff = params[stat_count + 1]
            stat_count += 1


        # Calculate the shifted time
        time_shifted = obs.time_data + t_diff

        # Estimate the intercept of the lag line, with the fixed slope (i.e. initial velocity)
        lag_line = fitLagIntercept(time_shifted, obs.length, v_init, obs.lag_line[1])

        # Calculate lag
        lag = obs.length - lineFunc(time_shifted, *lag_line)

        # Add lag to lag list
        lags.append([time_shifted,  np.array(lag)])


    # Choose the referent lag
    ref_time, ref_lag = lags[t_ref_station]

    # Do a spline fit on the referent lag
    ref_line_spline = scipy.interpolate.CubicSpline(ref_time, ref_lag, extrapolate=True)

    residual_sum = 0

    # Go through all lags
    for i, obs in enumerate(observations):

        # Skip the lag from the referent station
        if i == t_ref_station:
            continue

        time, lag = lags[i]

        # # Take only those points that overlap with the referent station
        # common_points = np.where((time > np.min(ref_time)) & (time < np.max(ref_time)))
        # time = time[common_points]
        # lag = lag[common_points]

        # Calculate the residuals in lag from the current lag to the referent lag, using smooth approximation
        # of L1 (absolute value) cost
        z = (ref_line_spline(time) - lag)**2
        residual_sum += np.sum(2*(np.sqrt(1 + z) - 1))


    return residual_sum





class Trajectory(object):
    """ Meteor trajectory solver designed for the UWO CAMO system.

    The solver makes a first estimate using the Ceplecha (1987) plane intersection approach, then refines the 
    solution my miniming the angles between the observed lines of sight and the radiant line. Furthermore, 
    initial velocity and timing differences between the stations are estimated by matching their lags 
    (deviations from a constant speed trajectory).
    
    """


    def __init__(self, jdt_ref, output_dir='.', max_toffset=1.0, meastype=4, verbose=True, 
        estimate_timing_vel=True, filter_picks=True, calc_orbit=True, show_plots=True, save_results=True):
        """ Init the Ceplecha trajectory solver.

        Arguments:
            jdt_ref: [float] Reference Julian date/time that the measurements times are provided relative to. 
                    This is user selectable and can be the time of the first camera, or the first measurement, 
                    or some average time for the meteor, but should be close to the time of passage of the 
                    meteor. This same reference date/time will be used on all camera measurements for the 
                    purposes of computing local sidereal time and making  geocentric coordinate 
                    transformations.

        Keyword arguments:
            output_dir: [str] path to the output directory where the Trajectory report and 'pickled' object
                will be stored
            max_toffset: [float] Maximum allowed time offset between cameras in seconds (default 1 second)
            meastype: [float] Flag indicating the type of angle measurements the user is providing for meas1 
                and meas2 below. The following are all in radians:
                        1 = Right Ascension for meas1, Declination for meas2.
                        2 = Azimuth +east of due north for meas1, Elevation angle
                            above the horizon for meas2
                        3 = Azimuth +west of due south for meas1, Zenith angle for meas2
                        4 = Azimuth +north of due east for meas1, Zenith angle for meas2
            verbose: [bool] Print out the results and status messages, True by default
            estimate_timing_vel: [bool] Try to estimate the difference in timing and velocity. True by default
            filter_picks: [bool] If True (default), picks which deviate more than 3 sigma in angular residuals
                will be removed, and the trajectory will be recalculated.
            calc_orbit: [bool] If True, the orbit is calculates as well. True by default
            show_plots: [bool] Show plots of residuals, velocity, lag, meteor position. True by default.
            save_results: [bool] Save results of trajectory estimation to disk. True by default.

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

        # Filter bad picks (ones that deviate more than 3 sigma in angular residuals) if this flag is True
        self.filter_picks = filter_picks

        # Calculate orbit if True
        self.calc_orbit = calc_orbit

        # If True, plots are shown
        self.show_plots = show_plots

        # Save results to disk if true
        self.save_results = save_results

        ######################################################################################################

        # Counts from how may stations the observations are given (start from 1)
        self.station_count = 1

        # List of observations
        self.observations = []

        # Index of the station with the referent time
        self.t_ref_station = 0

        # Final estimate of timing offsets between stations
        self.time_diffs_final = None

        # List of plane intersections
        self.intersection_list = None

        # Coordinates of the first point
        self.rbeg_lat = None
        self.rbeg_lon = None
        self.rbeg_ele = None

        # Coordinates of the end point
        self.rend_lat = None
        self.rend_lon = None
        self.rend_ele = None

        # Calculated initial velocity
        self.v_init = None

        # Jacchia fit parameters for all observations combined
        self.jacchia_fit = None

        # Orbit object which contains orbital parameters
        self.orbit = None



    def infillTrajectory(self, meas1, meas2, time_data, lat, lon, ele, station_id=None):
        """ Initialize a set of measurements for a given station. 
    
        Arguments:
            meas1: [list or ndarray] First measurement array (azimuth or R.A., depending on meastype, see 
                meastype documentation for more information). Measurements should be given in radians.
            meas2: [list or ndarray] Second measurement array (altitude, zenith angle or declination, 
                depending on meastype, see meastype documentation for more information), in radians.
            time_data: [list or ndarray] Time in seconds from the referent Julian date.
            lat: [float] Latitude +N of station in radians.
            lon: [float] Longitude +E of station in radians.
            ele: [float] Elevation of station in meters.

        Keyword arguments:
            station_id: [str] Identification of the station. None by default.

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

        # Init a new structure which will contain the observed data from the given site
        obs = ObservedPoints(self.jdt_ref, meas1, meas2, time_data, lat, lon, ele, station_id=station_id, 
            meastype=self.meastype)
            
        # Add observations to the total observations list
        self.observations.append(obs)

        self.station_count += 1



    def calcAllResiduals(self, state_vect, radiant_eci, observations):
        """ Calculate horizontal and vertical residuals for all observed points. 
            
            The residuals are calculated from the closest point on the line of sight to the point of the radiant 
            line.

        """

        # Go though observations from all stations
        for obs in observations:

            # Init empty lists for residuals
            obs.h_residuals = []
            obs.v_residuals = []

            # Go through all individual position measurement from each site
            for jd, stat, meas in zip(obs.JD_data, obs.stat_eci_los, obs.meas_eci_los):

                # Calculate horizontal and vertical residuals
                hres, vres = calcResidual(jd, state_vect, radiant_eci, stat, meas)

                # Add residuals to the residual list
                obs.h_residuals.append(hres)
                obs.v_residuals.append(vres)

            # Convert residual lists to numpy arrays
            obs.h_residuals = np.array(obs.h_residuals)
            obs.v_residuals = np.array(obs.v_residuals)


            # Calculate angular deviations in azimuth and elevation
            elev_res = obs.elev_data - obs.model_elev
            azim_res = (np.abs(obs.azim_data - obs.model_azim)%(2*np.pi))*np.sin(obs.elev_data)

            # Calculate the angular residuals from the radiant line
            obs.ang_res = np.sqrt(elev_res**2 + azim_res**2)



    def calcVelocity(self, state_vect, radiant_eci, observations):
        """ Calculates velocity for the given solution.

        """

        # Go through observations from all stations
        for obs in observations:

            radiant_distances = []

            # Go through all individual position measurement from each site
            for i, (stat, meas) in enumerate(zip(obs.stat_eci_los, obs.meas_eci_los)):

                # Calculate closest points of approach (observed line of sight to radiant line)
                obs_cpa, rad_cpa, d = findClosestPoints(stat, meas, state_vect, radiant_eci)

                # Take the position of the first point as the referent point
                if i == 0:
                    ref_point = np.copy(rad_cpa)

                # Calculate the distance from the first observed point to the projected point on the radiant line
                dist = vectMag(ref_point - rad_cpa)
                
                radiant_distances.append(dist)


            # Convert the distances (length along the trail) into a numpy array
            obs.length = np.array(radiant_distances)

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



    def calcLag(self, observations):
        """ Calculate lag by fitting a line to the first 25% of the points and subtracting the line from the 
            length along the trail.

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

            obs.jacchia_fit, _ = scipy.optimize.curve_fit(jacchiaLagFunc, obs.time_data, obs.lag, p0=p0)

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
        jacchia_fit, _ = scipy.optimize.curve_fit(jacchiaLagFunc, time_all, lag_all, p0=p0)


        return jacchia_fit



    def estimateTimingAndVelocity(self, observations, estimate_timing_vel=True):
        """ Estimate the timing difference between stations and the initial velocity by minimizing the 
            residuals between the lag. 

        """

        # Take the initial velocity as the median velocity between all sites
        v_init = np.array([obs.v_init for obs in observations])
        v_init = np.median(v_init)

        # Timing differences which will be calculated
        time_diffs = np.zeros(len(observations))

        # If the timing difference and velocity difference is not desired to be performed, skip the procedure
        if not estimate_timing_vel:
            return v_init, time_diffs

        if self.verbose:
            print('Median Vinit from all stations:', v_init, 'm/s')

        # Initial timing difference between sites is 0 (there are N-1 timing differences, as the time 
        # difference for the referent site is always 0)
        t_diffs = np.zeros(shape=(self.station_count - 1))

        # Initial parameters are the estimated initial velocity and the time difference (except for the referent station)
        p0 = np.r_[v_init, t_diffs]


        # Set the time referent station to be the one with the most picks
        obs_points = [obs.kmeas for obs in self.observations]
        self.t_ref_station = obs_points.index(max(obs_points))


        if self.verbose:
            print('Initial function evaluation:', timingAndVelocityResiduals(p0, observations, 
                self.t_ref_station))


        ### Perform the minimization of lag differences ###
        ######################################################################################################

        # Define the bounds for the initial velocity to be +/-5%
        bounds = [(0.95*v_init, 1.05*v_init)]

        # Set bounds for timing to +/- given maximum time offset
        for i in range(self.station_count - 1):
            bounds.append([-self.max_toffset, self.max_toffset])


        # Try different methods of optimization until it is successful
        methods = ['SLSQP', 'TNC']
        for opt_method in methods:
            
            # Run the minimization of residuals betwpeen lags of all stations
            vel_timing_mini = scipy.optimize.minimize(timingAndVelocityResiduals, p0, args=(observations, \
                self.t_ref_station), bounds=bounds, method=opt_method)

            # Stop trying methods if this one was successful
            if vel_timing_mini.success:
                break

        ######################################################################################################

        
        # If the minimization was successful, apply the time corrections
        if vel_timing_mini.success:

            v_init_mini, _ = vel_timing_mini.x[:2]

            if self.verbose:
                print("Final function evaluation:", vel_timing_mini.fun)
                print('ESTIMATED Vinit:', v_init_mini, 'm/s')

            stat_count = 0
            for i, obs in enumerate(observations):

                # The timing difference for the referent station is always 0
                if i == self.t_ref_station:
                    t_diff = 0

                else:
                    t_diff = vel_timing_mini.x[1 + stat_count]
                    stat_count += 1

                if self.verbose:
                    print('STATION ' + str(obs.station_id) + ' TIME OFFSET = ' + str(t_diff) + ' s')

                # Apply the time shift to original time data
                obs.time_data = obs.time_data + t_diff

                # Add the final time difference of the site to the list
                time_diffs[i] = t_diff

                # Calculate the new intercept for the length along the trail, with the given initial velocity
                lag_line = fitLagIntercept(obs.time_data, obs.length, v_init_mini, 0)

                # Recalculate the lag
                obs.lag = obs.length - lineFunc(obs.time_data, *lag_line)

        else:

            print('Timing difference and initial velocity minimization failed with the message:')
            print(vel_timing_mini.message)
            v_init_mini = v_init

        # Return the best estimate of the initial velocity
        return v_init_mini, time_diffs



    def calcLLA(self, state_vect, radiant_eci, observations):
        """ Calculate latitude, longitude and altitude of every point on the obesrver line of sight, 
            which is closest to the radiant line.

        """

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
            for i, (stat, meas) in enumerate(zip(obs.stat_eci_los, obs.meas_eci_los)):

                # Calculate closest points of approach (observed line of sight to radiant line)
                obs_cpa, rad_cpa, d = findClosestPoints(stat, meas, state_vect, radiant_eci)

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



            # Set the coordinates of the first point
            obs.rbeg_lat = obs.meas_lat[0]
            obs.rbeg_lon = obs.meas_lon[0]
            obs.rbeg_ele = obs.meas_ht[0]

            # Set the coordinates of the last point
            obs.rend_lat = obs.meas_lat[-1]
            obs.rend_lon = obs.meas_lon[-1]
            obs.rend_ele = obs.meas_ht[-1]


        # Find the heigest beginning height
        beg_hts = [obs.rbeg_ele for obs in self.observations]
        first_begin = beg_hts.index(max(beg_hts))

        # Set the coordinates of the height point as the first point
        self.rbeg_lat = self.observations[first_begin].rbeg_lat
        self.rbeg_lon = self.observations[first_begin].rbeg_lon
        self.rbeg_ele = self.observations[first_begin].rbeg_ele


        # Find the lowest ending height
        end_hts = [obs.rend_ele for obs in self.observations]
        last_end = end_hts.index(min(end_hts))

        # Set coordinates of the lowest point as the last point
        self.rend_lat = self.observations[last_end].rend_lat
        self.rend_lon = self.observations[last_end].rend_lon
        self.rend_ele = self.observations[last_end].rend_ele





    def calcECIEqAltAz(self, state_vect, radiant_eci, observations):
        """ Calculate ECI coordinates of both CPAs (observed and radiant), equatorial and alt-az coordinates 
            of CPA positions on the radiant line. 

        """


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
            for i, (stat, meas) in enumerate(zip(obs.stat_eci_los, obs.meas_eci_los)):

                # Calculate closest points of approach (observed line of sight to radiant line)
                obs_cpa, rad_cpa, d = findClosestPoints(stat, meas, state_vect, radiant_eci)

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

        """

        v_sum = 0
        eci_sum = np.zeros(3)
        jd_min = np.inf
        jd_max = -np.inf
        meas_sum = 0

        # Go through all observations
        for obs in observations:

            # Calculate the average velocity
            meteor_duration = obs.time_data[-1] - obs.time_data[0]
            meteor_length = vectMag(obs.model_eci[-1] - obs.model_eci[0])

            # Calculate the average velocity
            v_avg = meteor_length/meteor_duration

            v_sum += v_avg
            eci_sum += np.sum(obs.model_eci, axis=0)
            
            jd_min = min(jd_min, np.min(obs.JD_data))
            jd_max = max(jd_max, np.max(obs.JD_data))

            meas_sum += obs.kmeas


        # Average velocity across all stations
        v_avg = v_sum/len(observations)

        # Average ECI across all stations
        eci_avg = eci_sum/meas_sum

        # Average Julian date
        #jd_avg = jd_sum/meas_sum
        jd_avg = (jd_min + jd_max)/2

        
        return v_avg, eci_avg, jd_avg



    def dumpMeasurements(self, file_name):
        """ Writes the initialized measurements in a text file. Used for Gural trajectory solver tests."""

        with open(file_name, 'w') as f:

            for i, obs in enumerate(self.observations):

                # Write site coordinates
                f.write('m->longitude[' + str(i) + '] = ' + str(obs.lon) + ';\n')
                f.write('m->latitude[' + str(i) + '] = ' + str(obs.lat) + ';\n')
                f.write('m->heightkm[' + str(i) + '] = ' + str(obs.ele/1000) + ';\n\n')

                # Construct an measurement matrix (time, azimuth, zenith angle)
                meas_matr = np.c_[obs.time_data, np.degrees(obs.meas2), np.degrees(obs.meas1)]

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
        
        print('Measurements dumped into ', file_name)



    def saveReport(self, dir_path, file_name):
        """ Save the trajectory estimation report to file. """
        
        out_str = ''

        # out_str += 'Referent JD: {:.12f}\n'.format(self.jdt_ref)
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

        out_str += "Referent JD: {:20.12f}".format(self.jdt_ref)

        out_str += "\n\n"

        out_str += 'Plane intersections\n'
        out_str += '-------------------\n'

        # Print out all intersecting planes pairs
        for n, plane_intersection in enumerate(self.intersection_list):

            n = n + 1

            out_str += 'Intersection ' + str(n) + ' - Stations: ' + str(plane_intersection.obs1.station_id) + ' and ' + \
                str(plane_intersection.obs2.station_id) + '\n'

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
        out_str += "State vector (ECI):\n"
        out_str += " X =  {:10.2f} m\n".format(x)
        out_str += " Y =  {:10.2f} m\n".format(y)
        out_str += " Z =  {:10.2f} m\n".format(z)
        out_str += " Vx = {:10.2f} m/s\n".format(vx)
        out_str += " Vy = {:10.2f} m/s\n".format(vy)
        out_str += " Vz = {:10.2f} m/s\n".format(vz)

        out_str += "\n"

        out_str += "Timing offsets:\n"
        for stat_id, t_diff in zip([obs.station_id for obs in self.observations], self.time_diffs_final):
            out_str += "{:>10s}: {:.6f} s\n".format(str(stat_id), t_diff)

        out_str += "\n"

        out_str += "Average point on the trajectory:\n"
        out_str += "  Time: " + str(jd2Date(self.orbit.jd_avg, dt_obj=True)) + " UTC\n"
        out_str += "  Lon   = {:>10.6f}  Lat = {:>10.6f} deg\n".format(np.degrees(self.orbit.lon_avg), np.degrees(self.orbit.lat_avg))
        out_str += "\n"

        # Write out orbital parameters
        out_str += self.orbit.__repr__()
        out_str += "\n"

        out_str += "Jacchia fit on lag = -|a1|*exp(|a2|*t):\n"
        out_str += " a1 = {:.6f}\n".format(self.jacchia_fit[0])
        out_str += " a2 = {:.6f}\n".format(self.jacchia_fit[1])
        out_str += "\n"

        out_str += "      Lon +E (deg), Lat + N Deg),  Ele (m)\n"
        out_str += "Begin {:>12.6f}, {:>12.6f}, {:>8.2f}\n".format(np.degrees(self.rbeg_lon), np.degrees(self.rbeg_lat), self.rbeg_ele)
        out_str += "End   {:>12.6f}, {:>12.6f}, {:>8.2f}\n".format(np.degrees(self.rend_lon), np.degrees(self.rend_lat), self.rend_ele)
        out_str += "\n"

        ### Write information about stations ###
        ######################################################################################################
        out_str += "Stations\n"
        out_str += "--------\n"

        out_str += "        ID, Lon +E (deg), Lat +N (deg), Ele (m), Jacchia a1, Jacchia a2, Beg Ele (m), End Ele (m) \n"
        
        for obs in self.observations:
            station_info = [obs.station_id, obs.lat, obs.lon, obs.ele]

            station_info = []
            station_info.append("{:>10s}".format(str(obs.station_id)))
            station_info.append("{:>12.6f}".format(np.degrees(obs.lon)))
            station_info.append("{:>12.6f}".format(np.degrees(obs.lat)))
            station_info.append("{:>7.2f}".format(obs.ele))
            station_info.append("{:>10.6f}".format(obs.jacchia_fit[0]))
            station_info.append("{:>10.6f}".format(obs.jacchia_fit[1]))
            station_info.append("{:>11.2f}".format(obs.rbeg_ele))
            station_info.append("{:>11.2f}".format(obs.rend_ele))


            out_str += ", ".join(station_info) + "\n"
        
        ######################################################################################################

        out_str += "\n"

        ### Write information about individual points ###
        ######################################################################################################
        out_str += "Points\n"
        out_str += "------\n"


        out_str += " No, Station ID,  Time (s),                   JD,     meas1,     meas2, Azim +E of due N (deg), Alt (deg), Azim line (deg), Alt line (deg), RA obs (deg), Dec obs (deg), RA line (deg), Dec line (deg),      X (m),      Y (m),      Z (m), Latitude (deg), Longitude (deg), Height (m), Length (m),  Lag (m), Vel (m/s), H res (m), V res (m), Ang res (asec)\n"

        # Go through observation from all stations
        for obs in self.observations:

            # Go through all observed points
            for i in range(obs.kmeas):

                point_info = []

                point_info.append("{:3d}".format(i))

                point_info.append("{:>10s}".format(str(obs.station_id)))
                
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

                point_info.append("{:10.2f}".format(obs.model_eci[i][0]))
                point_info.append("{:10.2f}".format(obs.model_eci[i][1]))
                point_info.append("{:10.2f}".format(obs.model_eci[i][2]))

                point_info.append("{:14.6f}".format(np.degrees(obs.model_lat[i])))
                point_info.append("{:+15.6f}".format(np.degrees(obs.model_lon[i])))
                point_info.append("{:10.2f}".format(obs.model_ht[i]))

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
        out_str += "- 'meas1' and 'meas2' are given input points.\n"
        out_str += "- X, Y, Z are ECI (Earth-Centered Inertial) positions of projected lines of sight on the radiant line.\n"
        out_str += "- Latitude (deg), Longitude (deg), Height (m) are WGS84 coordinates of each point on the radiant line.\n"
        out_str += "- Jacchia (1955) equation fit was done on the lag (length minus the line fitted on the first 25% of the length).\n"
        out_str += "- Right ascension and declination in the table are given for the epoch of date for the corresponding JD, per every point.\n"
        out_str += "- 'RA and Dec obs' are the right ascension and declination calculated from the observed values, while the 'RA and Dec line' are coordinates of the lines of sight projected on the fitted radiant line. 'Azim and alt line' are thus corresponding azimuthal coordinates.\n"


        print(out_str)

        mkdirP(dir_path)

        # Save the report to a file
        with open(os.path.join(dir_path, file_name), 'w') as f:
            f.write(out_str)



    def showPlots(self, file_name):
        """ Show plots of the estimated trajectory. 
    
        Arguments:
            file_name: [str] file name which will be used for saving plots

        """

        # Plot residuals per observing station
        for obs in self.observations:

            ### PLOT RESIDUALS ###
            ##################################################################################################

            # Calculate root mean square
            v_res_rms = round(np.sqrt(np.mean(obs.v_residuals**2)), 2)
            h_res_rms = round(np.sqrt(np.mean(obs.h_residuals**2)), 2)

            # Plot vertical residuals
            plt.scatter(obs.time_data, obs.v_residuals, c='red', label='Vertical, RMS = '+str(v_res_rms), 
                zorder=3, s=2)

            # Plot horizontal residuals
            plt.scatter(obs.time_data, obs.h_residuals, c='b', label='Horizontal, RMS = '+str(h_res_rms), 
                zorder=3, s=2)

            plt.title('Residuals, station ' + str(obs.station_id))
            plt.xlabel('Time (s)')
            plt.ylabel('Residuals (m)')

            plt.grid()

            plt.legend()

            # Set the residual limits to +/-10m if they are smaller than that
            if (np.max(np.abs(obs.v_residuals)) < 10) and (np.max(np.abs(obs.h_residuals)) < 10):
                plt.ylim([-10, 10])


            savePlot(plt, file_name + '_' + str(obs.station_id) + '_spatial_residuals.png', \
                self.output_dir)

            plt.show()

            ##################################################################################################


        # Plot lag per observing station
        for obs in self.observations:
            
            ### PLOT LAG ###
            ##################################################################################################

            fig, ax1 = plt.subplots()

            ax1.plot(obs.lag, obs.time_data, color='r', marker='x', label='Lag', zorder=3)

            # Plot the Jacchia fit
            ax1.plot(jacchiaLagFunc(obs.time_data, *obs.jacchia_fit), obs.time_data, color='b', 
                label='Jacchia fit', zorder=3)
            
            ax1.legend()

            plt.title('Lag, station ' + str(obs.station_id))
            ax1.set_xlabel('Lag (m)')
            ax1.set_ylabel('Time (s)')

            ax1.set_ylim(min(obs.time_data), max(obs.time_data))

            ax1.grid()

            ax1.invert_yaxis()

            # Set the height axis
            ax2 = ax1.twinx()
            ax2.set_ylim(min(obs.meas_ht), max(obs.meas_ht))
            ax2.set_ylabel('Height (m)')

            plt.tight_layout()

            savePlot(plt, file_name + '_' + str(obs.station_id) + '_lag.png', self.output_dir)

            plt.show()


            ##################################################################################################



        # Plot lags from each station on a single plot
        for obs in self.observations:
            plt.plot(obs.lag, obs.time_data, marker='x', label='Station: ' + str(obs.station_id), 
                zorder=3)

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

        savePlot(plt, file_name + '_lags_all.png', self.output_dir)

        plt.show()



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

        # Plot velocities from each observed site
        for i, obs in enumerate(self.observations):

            ax1.scatter(obs.velocities[1:], obs.time_data[1:], marker=markers[i%len(markers)], 
                c=colors[i], alpha=0.5, label='Station: ' + str(obs.station_id), zorder=3)

            # Determine the max/min velocity and height, as this is needed for plotting both height/time axes
            vel_max = max(np.max(obs.velocities[1:]), vel_max)
            vel_min = min(np.min(obs.velocities[1:]), vel_min)

            ht_max = max(np.max(obs.meas_ht[1:]), ht_max)
            ht_min = min(np.min(obs.meas_ht[1:]), ht_min)

            t_max = max(np.max(obs.time_data[1:]), t_max)
            t_min = min(np.min(obs.time_data[1:]), t_min)


        # Plot the velocity calculated from the Jacchia model
        t_vel = np.linspace(t_min, t_max, 1000)
        ax1.plot(jacchiaVelocityFunc(t_vel, self.jacchia_fit[0], self.jacchia_fit[1], self.v_init), t_vel,
            label='Jacchia fit', alpha=0.5)

        plt.title('Velocity')
        ax1.set_xlabel('Velocity (m/s)')
        ax1.set_ylabel('Time (s)')

        ax1.legend()
        ax1.grid()

        # Set velocity limits to +/- 3 km/s
        ax1.set_xlim([vel_min - 3000, vel_max + 3000])

        # Set time axis limits
        ax1.set_ylim([t_min, t_max])
        ax1.invert_yaxis()

        # Set the height axis
        ax2 = ax1.twinx()
        ax2.set_ylim(ht_min, ht_max)
        ax2.set_ylabel('Height (m)')

        plt.tight_layout()

        savePlot(plt, file_name + '_velocities.png', self.output_dir)

        plt.show()

        ######################################################################################################


        ### Plot lat/lon of the meteor ###
            
        # Calculate mean latitude and longitude of all meteor points
        met_lon_mean = np.degrees(meanAngle([x for x in obs.meas_lon for obs in self.observations]))
        met_lat_mean = np.degrees(meanAngle([x for x in obs.meas_lat for obs in self.observations]))

        # Calculate the mean latitude and longitude by including station positions
        lon_mean = np.degrees(meanAngle([np.radians(met_lon_mean)] + [obs.lon for obs in self.observations]))
        lat_mean = np.degrees(meanAngle([np.radians(met_lat_mean)] + [obs.lat for obs in self.observations]))


        # Put coordinate of all sites and the meteor in the one list
        geo_coords = [[obs.lat, obs.lon] for obs in self.observations]
        geo_coords.append([np.radians(met_lat_mean), np.radians(met_lon_mean)])

        # Find the maximum distance from the center to all stations and meteor points, this is used for 
        # scaling the finalground track plot
        
        max_dist = 0
        
        lat1 = np.radians(lat_mean)
        lon1 = np.radians(lon_mean)

        for lat2, lon2 in geo_coords:

            # Calculate the angular distance between two coordinates
            delta_lat = lat2 - lat1
            delta_lon = lon2 - lon1
            a = np.sin(delta_lat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(delta_lon/2)**2
            c = 2*np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            d = 6371000*c

            # Set the current distance as maximum if it is larger than the previously found max. value
            if d > max_dist:
                max_dist = d


        # Add some buffer to the maximum distance (50 km)
        max_dist += 50000


        from mpl_toolkits.basemap import Basemap
        m = Basemap(projection='gnom', lat_0=lat_mean, lon_0=lon_mean, width=2*max_dist, height=2*max_dist, 
            resolution='i')

        # Draw the coast boundary and fill the oceans with the given color
        m.drawmapboundary(fill_color='0.2')

        # Fill continents, set lake color same as ocean color
        m.fillcontinents(color='black', lake_color='0.2', zorder=0)

        # Draw country borders
        m.drawcountries(color='0.2')
        m.drawstates(color='0.15', linestyle='--')

        # Draw parallels
        parallels = np.arange(-90, 90, 1.)
        m.drawparallels(parallels, labels=[False, True, True, False], color='0.25')

        # Draw meridians
        meridians = np.arange(0, 360, 1.)
        m.drawmeridians(meridians, labels=[True, False, False, True], color='0.25')

        # Plot locations of all stations and measured positions of the meteor
        for obs in self.observations:

            # Plot stations
            x, y = m(np.degrees(obs.lon), np.degrees(obs.lat))
            m.scatter(x, y, s=10, label=str(obs.station_id), marker='x')

            # Plot measured points
            x, y = m(np.degrees(obs.meas_lon), np.degrees(obs.meas_lat))
            m.plot(x, y, c='r')


        # # Plot an arrow showing the direction of the flight path
        # x1, y1 = m(np.degrees(self.rbeg_lon), np.degrees(self.rbeg_lat))
        # x2, y2 = m(np.degrees(self.rend_lon), np.degrees(self.rend_lat))

        # plt.annotate('', xy=(x1, y1),  xycoords='data',
        #         xytext=(x2, y2), textcoords='data',
        #         arrowprops=dict(arrowstyle="->, head_width=0.3, head_length=0.5", color='r'),
        #         )

        # Plot a point marking the final point of the meteor
        x_end, y_end = m(np.degrees(self.rend_lon), np.degrees(self.rend_lat))
        m.scatter(x_end, y_end, c='y', marker='+', s=50, alpha=0.75, label='Endpoint')


        plt.legend()

        savePlot(plt, file_name + '_ground_track.png', self.output_dir)

        plt.show()

        ######################################################################################################


        # Compare original and modeled measurements (residuals in azimuthal coordinates)
        for obs in self.observations:

            # Calculate residuals in arcseconds
            res = np.degrees(obs.ang_res)*3600

            # Calculate the RMS of the residuals
            res_rms = round(np.sqrt(np.mean(res**2)), 2)

            # Plot residuals
            plt.scatter(obs.time_data, res, label='Angle, RMS = {:.2f}'.format(res_rms), s=2, zorder=3)

            plt.title('Observed vs. Radiant LoS Residuals, station ' + str(obs.station_id))
            plt.ylabel('Angle (arcsec)')
            plt.xlabel('Time (s)')

            # The lower limit is always at 0
            plt.ylim(ymin=0)

            plt.grid()
            plt.legend()

            savePlot(plt, file_name + '_' + str(obs.station_id) + '_angular_residuals.png', \
                self.output_dir)

            plt.show()


        # Plot angular residuals from all stations
        for obs in self.observations:

            # Calculate residuals in arcseconds
            res = np.degrees(obs.ang_res)*3600

            # Calculate the RMS of the residuals
            res_rms = round(np.sqrt(np.mean(res**2)), 2)

            # Plot residuals
            plt.scatter(obs.time_data, res, s=2, zorder=3, label='Station ' + str(obs.station_id) + \
                ', RMS = {:.2f}'.format(res_rms))


        plt.title('Observed vs. Radiant LoS Residuals, all stations')
        plt.ylabel('Angle (arcsec)')
        plt.xlabel('Time (s)')

        # The lower limit is always at 0
        plt.ylim(ymin=0)

        plt.grid()
        plt.legend()

        savePlot(plt, file_name + '_all_angular_residuals.png', self.output_dir)

        plt.show()



        # Plot the orbit in 3D
        if self.calc_orbit:

            # Construct a list of orbital elements of the meteor
            orbit_params = np.array([
                [self.orbit.a, self.orbit.e, np.degrees(self.orbit.i), np.degrees(self.orbit.peri), \
                    np.degrees(self.orbit.node)]
                ])

            # Run orbit plotting procedure
            plotOrbits(orbit_params, jd2Date(self.jdt_ref, dt_obj=True))




    def run(self, _rerun_timing=False, _rerun_bad_picks=False, _prev_toffsets=None):
        """ Estimate the trajectory from the given input points. 
        
        Arguments:
            _rerun_timing: [bool] Internal flag. Is it True when everything is recalculated upon estimating 
                the difference in timings, so it breaks the second trajectory run after updating the values
                of R.A., Dec, velocity, etc.
            _rerun_bad_picks: [bool] Internal flag. Is is True when a second pass of trajectory estimation is
                run with bad picks removed, thus improving the solution.
            _prev_toffsets: [ndarray] Internal variable. Used for keeping the initially estimated timing 
                offsets from the first run of the solver. None by default.

        """

        # Determine which station has the referent time (the first time entry is 0 for that station)
        for i, obs in enumerate(self.observations):
            
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
        weights_sum = 0

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


        # # Calculate the initial 3D position state vector in ECI coordinates
        # self.state_vect, _, _ = findClosestPoints(self.best_conv_inter.obs1.stat_eci, 
        #     self.best_conv_inter.obs1.meas_eci[0], self.best_conv_inter.cpa_eci, 
        #     self.best_conv_inter.radiant_eci)

        # Set the 3D position of the radiant line as the state vector
        self.state_vect = self.best_conv_inter.cpa_eci


        ######################################################################################################


        if self.verbose:
            print('Intersecting planes solution:', self.state_vect)
            
            print('Minimizing angle deviations...')


        ### LEAST SQUARES SOLUTION ###
        ######################################################################################################

        # Calculate the initial sum and angles deviating from the radiant line
        angle_sum = angleSumMeasurements2Line(self.observations, self.state_vect, 
             self.best_conv_inter.radiant_eci)

        if self.verbose:
            print('Initial angle sum:', angle_sum)


        # Set the initial guess for the state vector and the radiant from the intersecting plane solution
        p0 = np.r_[self.state_vect, self.best_conv_inter.radiant_eci]

        # Perform the minimization of angle deviations
        minimize_solution = scipy.optimize.minimize(minimizeAngleCost, p0, args=(self.observations), 
            method="Nelder-Mead")


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
                bounds.append([0.75*val, 1.25*val])

            print('BOUNDS:', bounds)
            print('p0:', p0)
            minimize_solution = scipy.optimize.minimize(minimizeAngleCost, p0, args=(self.observations), 
                bounds=bounds, method='SLSQP')


        if self.verbose:
            print('Minimization info:')
            print(' Message:', minimize_solution.message)
            print(' Iterations:', minimize_solution.nit)
            print(' Success:', minimize_solution.success)
            print(' Final function value:', minimize_solution.fun)

        # If the minimization succeded
        if minimize_solution.success:
        
            # Unpack the solution
            self.state_vect_mini, self.radiant_eci_mini = np.hsplit(minimize_solution.x, 2)

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

            # If the solution did not succed, set the values to intersecting plates solution
            self.state_vect_mini = self.state_vect
            self.radiant_eci_mini = self.best_conv_inter.radiant_eci

            # Normalize radiant direction
            self.radiant_eci_mini = vectNorm(self.radiant_eci_mini)

            # Convert the minimized radiant solution to RA and Dec
            self.radiant_eq_mini = eci2RaDec(self.radiant_eci_mini)

        ######################################################################################################


        # Calculate velocity at each point
        self.calcVelocity(self.state_vect_mini, self.radiant_eci_mini, self.observations)

        # Calculate lag
        self.calcLag(self.observations)



        if self.verbose and self.estimate_timing_vel:
                print('Estimating initial velocity and timing differences...')


        # Estimate the timing difference between stations and the initial velocity
        self.v_init, self.time_diffs = self.estimateTimingAndVelocity(self.observations, \
            estimate_timing_vel=self.estimate_timing_vel)

        self.time_diffs_final = self.time_diffs


        ### RERUN THE TRAJECTORY ESTIMATION WITH UPDATED TIMINGS ###
        ######################################################################################################

        # Runs only in the first pass of trajectory estimation and estimates timing offsets between stations
        if not _rerun_timing:

            # After the timing has been estimated, everything needs to be recalculated from scratch
            if self.estimate_timing_vel:

                # Make a copy of observations
                temp_observations = copy.deepcopy(self.observations)

                
                # Reset the observation points
                self.observations = []

                print("Updating the solution after the timing estimation...")

                # Reinitialize the observations with proper timing
                for obs in temp_observations:
            
                    self.infillTrajectory(obs.meas1, obs.meas2, obs.time_data, obs.lat, obs.lon, obs.ele, \
                        obs.station_id)

                
                # Re-run the trajectory estimation with updated timings. This will update all calculated
                # values up to this point
                self.run(_rerun_timing=True, _prev_toffsets=self.time_diffs)


        else:

            # In the second pass, calculate the final timing offsets
            if _prev_toffsets is not None:
                self.time_diffs_final = _prev_toffsets + self.time_diffs

            else:
                self.time_diffs_final = self.time_diffs

            return None

        ######################################################################################################



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
                    good_picks = np.argwhere(obs.ang_res < (np.mean(obs.ang_res) + 3*np.std(obs.ang_res))).ravel()

                    # Check if any picks were removed
                    if len(good_picks) < len(obs.ang_res):
                        picks_rejected += len(obs.ang_res) - len(good_picks)

                        # Take only the good picks
                        obs.time_data = obs.time_data[good_picks]
                        obs.meas1 = obs.meas1[good_picks]
                        obs.meas2 = obs.meas2[good_picks]


                # Run only if some picks were rejected
                if picks_rejected:

                    # Make a copy of observations
                    temp_observations = copy.deepcopy(self.observations)
                    
                    # Reset the observation points
                    self.observations = []

                    print("Updating the solution after rejecting", picks_rejected, "bad picks...")

                    # Reinitialize the observations without the bad picks
                    for obs in temp_observations:
                
                        self.infillTrajectory(obs.meas1, obs.meas2, obs.time_data, obs.lat, obs.lon, obs.ele, \
                            obs.station_id)

                    
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
            v_avg, eci_avg, jd_avg = self.calcAverages(self.observations)


            # Calculate the orbit of the meteor
            self.orbit = calcOrbit(self.radiant_eci_mini, self.v_init, v_avg, eci_avg, jd_avg)

            # Set observed radiant parameters
            self.orbit.ra, self.orbit.dec = self.radiant_eq_mini
            self.orbit.v_avg = v_avg
            self.orbit.v_init = self.v_init

            print(self.orbit)


        ######################################################################################################



        #### SAVE REPORTS ###
        ######################################################################################################

        if self.save_results:

            # Construct a file name for this event
            file_name = jd2Date(self.jdt_ref, dt_obj=True).strftime('%Y%m%d_%H%M%S')

            # Save the picked trajectory structure
            savePickle(self, self.output_dir, file_name + '_trajectory.pickle')

            # Save trajectory report
            self.saveReport(self.output_dir, file_name + '_report.txt')

        ######################################################################################################


        # Show plots if show_plots flag is true
        if self.show_plots:

            self.showPlots(file_name)








if __name__ == "__main__":

    ### TEST CASE ###
    ##########################################################################################################

    import time

    # Referent julian date
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
    traj_solve = Trajectory(jdt_ref, meastype)

    # Set input points for the first site
    traj_solve.infillTrajectory(theta1, phi1, time1, lat1, lon1, ele1)

    # Set input points for the second site
    traj_solve.infillTrajectory(theta2, phi2, time2, lat2, lon2, ele2)


    t1 = time.clock()

    # Solve the trajectory
    traj_solve.run()

    print('Run time:', time.clock() - t1)


    ##########################################################################################################