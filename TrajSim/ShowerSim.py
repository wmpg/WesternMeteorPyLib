""" Given the parameters that describe a meteor shower, this code generates the shower meteors.
"""

from __future__ import print_function, division, absolute_import

import sys
import os
import datetime

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

from Config import config

from TrajSim.TrajSim import geocentricRadiantToApparent

from Trajectory.Trajectory import ObservedPoints, Trajectory
from Trajectory.GuralTrajectory import GuralTrajectory
from Trajectory.Orbit import calcOrbit

from Utils.TrajConversions import J2000_JD, EARTH, altAz2RADec, raDec2ECI, rotatePolar, jd2Date, datetime2JD, \
    jd2DynamicalTimeJD, cartesian2Geo, geo2Cartesian, eci2RaDec, raDec2AltAz, equatorialCoordPrecession
from Utils.Ephem import astronomicalNight
from Utils.SolarLongitude import solLon2jdJPL
from Utils.PlotCelestial import CelestialPlot
from Utils.Math import meanAngle, pointInsideConvexHull, samplePointsFromHull, vectMag, vectNorm, \
    sphericalToCartesian, cartesianToSpherical
from Utils.OSTools import mkdirP
from Utils.Pickling import savePickle
from Utils.Plotting import savePlot


# Try importing Campbell-Brown & Koschny (2004) ablation code 
try:
    from MetSim.MetSim import loadInputs, runSimulation
    METSIM_IMPORT = True
except:
    METSIM_IMPORT = False



class SimStation(object):
    def __init__(self, lat, lon, elev, station_id, fps, t_offset, obs_ang_std, azim_centre, elev_centre, \
        fov_wid, fov_ht, lim_mag=None, P_0m=None):
        """ Simulated station parameters. 

        Arguments:
            lat: [float] Station latitude (radians).
            lon: [float] Station longitude (radians).
            elev: [float] Stations elevtaion (meters).
            station_id: [str] Station name or ID.
            fps: [float] Camera framerate in frames per second.
            t_offset: [float] Time offset from real time (seconds).
            obs_ang_std: [float] Measurement uncertanties (radians).
            azim_centre: [float] Azimuth eastward of due N of the centre of FOV (radians).
            elev_centre: [float] Elevation from horizon of the centre of FOV (radians).
            fov_wid: [float] FOV width in azimuth (radians).
            fov_ht: [float] FOV height in elevation (radians).

        Keyword arguments:
            lim_mag: [float] Limiting magnitude for meteors.
            P_0m: [float] Bolometric power of zero magnitude meteor (Watts).

        """

        # Station geographical coordiantes
        self.lat = lat
        self.lon = lon
        self.elev = elev

        # Station ID
        self.station_id = station_id

        # FPS of the camera
        self.fps = fps

        # Time offset from the real time in seconds
        self.t_offset = t_offset

        # Uncertanties in observations (radians)
        self.obs_ang_std = obs_ang_std

        # Field of view centre (radians)
        self.azim_centre = azim_centre
        self.elev_centre = elev_centre

        # Field of view width/azimuth (radians)
        self.fov_wid = fov_wid

        # Field of view height/altitude (radians)
        self.fov_ht = fov_ht

        # Limiting magnitude
        self.lim_mag = lim_mag

        # Bolometric power of zero magnitude meteor
        self.P_0m = P_0m



    def fovCornersToECI(self, jd, geo_height, geo_height_top=None):
        """ Calculates ECI coordinates of the corners of the FOV at the given height. 
    
        Arguments:
            jd: [float] Julian date for ECI calculation.
            geo_height: [float] Height above sea level (meters) for which the ECI coordinates will be 
                calculated.
            geo_height_top: [float] If one wants to find the FOV box at two different heights, it is possible  
                to define a second height as well.

        Return:
            corners_eci: [list] A list of FOV corner points in ECI (meters).

        """

        # Define corners of FOV
        corners_azim = [-self.fov_wid/2, -self.fov_wid/2, self.fov_wid/2, self.fov_wid/2]
        corners_elev = [ -self.fov_ht/2,   self.fov_ht/2,  self.fov_ht/2, -self.fov_ht/2]

        corners = []

        # Go through all corner points
        for azim, elev in zip(corners_azim, corners_elev):

            # Find the proper azimuth and elevation for the given corner, but centred at (0, 0)
            azim, elev = rotatePolar(0, 0, azim, elev)

            # Find the corners centred at the FOV centre
            azim, elev = rotatePolar(azim, elev, self.azim_centre, self.elev_centre)

            corners.append([azim, elev])


        corners_eci_unit = []

        # Convert corners to ECI unit vectors
        for corn in corners:

            azim, elev = corn

            ra, dec = altAz2RADec(azim, elev, jd, self.lat, self.lon)

            # Convert R.A. and Dec to ECI unit vectors coordinates
            corn_eci_unit = raDec2ECI(ra, dec)

            corners_eci_unit.append(corn_eci_unit)



        # Calculate the ECI coordinates of the station at the given time
        stat_eci = geo2Cartesian(self.lat, self.lon, self.elev, jd)


        
        def _heightResiduals(k, jd, eci_unit, stat_eci, geo_height):
            
            # Calculate the ECI coordinates given the estimated range
            eci_coord = k*eci_unit

            # Calculate the difference between the estimated height and the desired height
            return (cartesian2Geo(jd, *eci_coord + stat_eci)[2] - geo_height)**2


        corners_eci = []

        # Numerically find ECI coordinates of the corners at the given height
        for corn_eci_unit in corners_eci_unit:

            corn_eci_unit = np.array(corn_eci_unit)

            # Estimate the range to the given height
            res = scipy.optimize.minimize(_heightResiduals, x0=[geo_height], args=(jd, corn_eci_unit, \
                stat_eci, geo_height), method='Nelder-Mead')


            if not res.success:
                print('ERROR! Height estimation failed for one of the corners for station', self.station_id)

            # Calculate the ECI coordinate of the corner at the given height
            corn_eci = res.x*corn_eci_unit + stat_eci

            corners_eci.append(corn_eci)



        # If the top height is given for calculating the FOV box, calculate the second set of points
        if geo_height_top is not None:
            corners_eci += self.fovCornersToECI(jd, geo_height_top)


        # from mpl_toolkits.mplot3d import Axes3D
        # import matplotlib.pyplot as plt

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        # for corn_eci in corners_eci:

        #     # Plot corner
        #     ax.scatter(*corn_eci, c='r')

        # ax.scatter(*stat_eci)


        # # ax.set_xlim([-1, 1])
        # # ax.set_ylim([-1, 1])
        # # ax.set_zlim([-1, 1])
        # plt.show()


        return corners_eci



class SimMeteor(object):
    def __init__(self, ra_g, dec_g, v_g, year, month, sol, jdt_ref, beg_height, state_vect):
        """ Container for a simulated meteor. 
    
        Arguments:
            ra_g: [float] Geocentric right ascension (radians).
            dec_g: [float] Geocentric declination (radians).
            v_g: [float] Geocentric velocity (m/s).
            year: [int] Year of the meteor shower.
            month: [int] Month of the meteor shower.
            sol: [float] Solar longitude (radians).
            jdt_ref: [float] Julian date.
            beg_height: [float] Beginning height (meters).
            state_vect: [ndarray] (x, y, z) ECI coordinates of the initial state vector (meters).
        """

        ### SIMULATED VALUES ###
        ######################################################################################################
        
        # Geocentric right ascension (radians)
        self.ra_g = ra_g

        # Geocentric declination (radians)
        self.dec_g = dec_g

        # Geocentric velocity (m/s)
        self.v_g = v_g

        # Solar longitude (radians)
        self.sol = sol

        # Beginning height (meters)
        self.beg_height = beg_height

        # Referent Julian date which corresponds to the solar longitude
        self.jdt_ref = jdt_ref

        # Initial state vector in ECI coordinates
        self.state_vect = state_vect

        # Velocity model
        self.velocity_model = None

        ######################################################################################################

        # Calculate geographic coordinates of the state vector
        self.rbeg_lat, self.rbeg_lon, self.rbeg_ele = cartesian2Geo(jdt_ref, *state_vect)

        # Calculate apparent radiant and the orbit
        self.ra, self.dec, self.v_init, self.orb = geocentricRadiantToApparent(ra_g, dec_g, v_g, state_vect, \
            jdt_ref)


        # Velocity at the beginning heights
        self.v_begin = None

        # Calculated apparent values
        self.radiant_eq = [self.ra, self.dec]
        self.radiant_eci = np.array(raDec2ECI(*self.radiant_eq))


        # Simulated ECI coordinates per every station
        self.model_eci = []

        # Simulated geographical positions per every station
        self.model_lat = []
        self.model_lon = []
        self.model_elev = []

        # List of observations - ObservedPoints objects from Trajectory.Trajectory
        self.observations = []

        # Name of the file where the info will be stored
        self.file_path = None



    def saveInfo(self, output_dir, t_offsets, obs_ang_uncertainties):
        """ Save information about the simulated meteor. """

        out_str =  ""
        out_str += "Referent JD: {:20.12f}".format(self.jdt_ref) + "\n"
        out_str += "\n"


        x, y, z = self.state_vect
        vx, vy, vz = self.v_init*self.radiant_eci

        # Write out the state vector
        out_str += "State vector (ECI):\n"
        out_str += " X =  {:11.2f} m\n".format(x)
        out_str += " Y =  {:11.2f} m\n".format(y)
        out_str += " Z =  {:11.2f} m\n".format(z)
        out_str += " Vx = {:11.2f} m/s\n".format(vx)
        out_str += " Vy = {:11.2f} m/s\n".format(vy)
        out_str += " Vz = {:11.2f} m/s\n".format(vz)

        out_str +=  "\n"

        out_str += "Radiant (apparent):\n"
        out_str += "  R.A.   = {:>9.5f} deg\n".format(np.degrees(self.ra))
        out_str += "  Dec    = {:>+9.5f} deg\n".format(np.degrees(self.dec))
        out_str += "  Vinit  = {:>9.5f} km/s\n".format(self.v_init/1000)

        if self.v_begin is not None:
            out_str += "  Vbegin = {:>9.5f} km/s\n".format(self.v_begin/1000)

        out_str += "Radiant (geocentric):\n"
        out_str += "  R.A.   = {:>9.5f} deg\n".format(np.degrees(self.ra_g))
        out_str += "  Dec    = {:>+9.5f} deg\n".format(np.degrees(self.dec_g))
        out_str += "  Vg     = {:>9.5f} km/s\n".format(self.v_g/1000)
        #out_str +=  "--------------------\n"
        out_str += "  La Sun = {:>10.6f} deg\n".format(np.degrees(self.sol))

        out_str += "Begin:\n"
        out_str += "  Lon = {:>12.6f} deg\n".format(np.degrees(self.rbeg_lon))
        out_str += "  Lat = {:>12.6f} deg\n".format(np.degrees(self.rbeg_lat))
        out_str += "  Ht  = {:>8.2f} m\n".format(self.rbeg_ele)


        if self.velocity_model is not None:

            out_str += "Velocity model:\n"
            out_str += self.velocity_model.__repr__()
            out_str += "\n"


        # Save station data

        out_str += 'Stations\n'
        out_str += "--------\n"

        out_str += "        ID, Lon +E (deg), Lat +N (deg), Ele (m), Time offset (s), Obs. uncertanty (arcsec)\n"
        for obs, t_off, obs_ang_std in zip(self.observations, t_offsets, obs_ang_uncertainties):

            station_info = []
            station_info.append("{:>10s}".format(str(obs.station_id)))
            station_info.append("{:>12.6f}".format(np.degrees(obs.lon)))
            station_info.append("{:>12.6f}".format(np.degrees(obs.lat)))
            station_info.append("{:>7.2f}".format(obs.ele))
            station_info.append("{:>15.6f}".format(t_off))
            station_info.append("{:>24.2f}".format(obs_ang_std))
            
            out_str += ", ".join(station_info) + "\n"


        print(out_str)

        mkdirP(output_dir)

        # File name of the report file
        file_name = str(self.jdt_ref) + "_sim_met_info.txt"

        self.file_path = os.path.join(output_dir, file_name)

        # Save the report to a file
        with open(self.file_path, 'w') as f:
            f.write(out_str)


        # Erase spline fits in the ablation model, as they cannot be pickled
        if self.velocity_model.name == 'ablation':
            self.velocity_model.velocity_model = None
            self.velocity_model.length_model = None
            self.velocity_model.luminosity_model = None

        # Save the SimMeteor object as pickle
        file_name_pickle = str(self.jdt_ref) + "_sim_met.pickle"
        savePickle(self, output_dir, file_name_pickle)


    def saveTrajectoryComparison(self, traj, traj_method):
        """ Saves the comparison of trajectory results, between the original values and the estimated values.
        
        Arguments:
            traj: [Trajectory]
            traj_method: [str] Method of trajectory estimation.

        """

        # Add the info to the simulation file, if it exists
        if self.file_path:

            out_str = '\n'
            out_str += '-----------------------------\n'
            out_str += 'Trajectory estimation method: {:s}\n' .format(traj_method)

            out_str += '\n'

            # Check if the trajectory was sucessfult estimated
            if traj.orbit.ra_g is not None:

                # Compare differences in radiant
                out_str += 'Differences in geocentric radiant: (original minus estimated)\n'
                out_str += '  dR.A.  = {:8.4f} deg\n'.format(np.degrees(np.abs(self.ra_g - traj.orbit.ra_g)%(2*np.pi)))
                out_str += '  dDec   = {:8.4f} deg\n'.format(np.degrees((self.dec_g - traj.orbit.dec_g + np.pi)%(2*np.pi) - np.pi))
                out_str += '  dVg    = {:7.3f} km/s\n'.format((self.v_g - traj.orbit.v_g)/1000)
                out_str += 'Velocity at beginning height (i.e. initial velocity):\n'
                out_str += 'simVbeg  = {:7.3f} km/s\n'.format(self.v_begin/1000)
                out_str += '  dVbeg  = {:7.3f} km/s\n'.format((self.v_begin - traj.orbit.v_init)/1000)

            else:
                out_str += 'Trajectory estimation failed!'


            out_str += '\n'


            with open(self.file_path, 'a') as f:
                f.write(out_str)





class ConstantVelocity(object):
    def __init__(self, duration):
        """ Constant velocity model for generating meteor trajectories. 
        
        Arguments:
            duration: [float] Duration of the meteor in seconds. 
        """

        self.name = 'constant'

        self.duration = duration



    def getTimeData(self, fps):
        """ Returns an array of time data for the meteor. 
        
        Arguments:
            fps: [float] Frames per second of the camera.

        """

        return np.arange(0, self.duration, 1.0/fps)



    def getLength(self, v_init, t):
        """ Calculates a constant velocity length along the track at the given time with the given initial
            velocity. 

        Arguments:
            v_init: [float] Velocity at t = 0. In m/s.
            t: [float] Time at which the length along the track will be evaluated.

        """

        return v_init*t



    def __repr__(self):
        """ Returned upon printing the object. """

        out_str = "Constant velocity, duration: {:.4f}".format(self.duration) + "\n"

        return out_str




class JacchiaVelocity(object):
    def __init__(self, duration, a1, a2):
        """ Exponential velocity model by Whipple & Jacchia for generating meteor trajectories. 
        
        Arguments:
            duration: [float] Duration of the meteor in seconds. 
            a1: [float] First deceleration term.
            a2: [float] Second deceleration term.
        """

        self.name = 'exponential'

        self.duration = duration

        self.a1 = a1
        self.a2 = a2



    def getTimeData(self, fps):
        """ Returns an array of time data for the meteor. 
        
        Arguments:
            fps: [float] Frames per second of the camera.

        """

        return np.arange(0, self.duration, 1.0/fps)



    def getLength(self, v_init, t):
        """ Calculates the length along the track at the given time with the given velocity using the 
            exponential deceleration model.

        Arguments:
            v_init: [float] Velocity at t = 0. In m/s.
            t: [float] Time at which the length along the track will be evaluated.

        """

        return v_init*t - abs(self.a1*np.exp(self.a2*t))



    def __repr__(self):
        """ Returned upon printing the object. """

        out_str  = "Exponential deceleration (Jacchia), duration: {:.4f}".format(self.duration) + "\n"
        out_str += "  a1 = {:.4f}".format(self.a1) + "\n"
        out_str += "  a2 = {:.4f}".format(self.a2) + "\n"

        return out_str




class AblationModelVelocity(object):
    def __init__(self, mass, density, ablation_coeff, Gamma, Lambda, lum_eff):
        """ Velocity calculated from numerical meteor ablation simulation by method of Campbell-Brown & 
            Koschny (2004).
        
        Arguments:
            mass: [float] Meteoroid mass (kg).
            density: [float] Meteoroid density (kg/m^3).
            ablation_coeff: [float] Ablation coefficient (s^2/km^2)
            Gamma: [float] Drag coefficient.
            Lambda: [float] Heat transfer coefficient.
            lum_eff: [float] Luminous efficiency (fraction).

        """

        self.name = 'ablation'

        self.mass = mass
        self.density = density
        self.Gamma = Gamma
        self.Lambda = Lambda
        self.ablation_coeff = ablation_coeff
        self.lum_eff = lum_eff

        self.length_model = None
        self.luminosity_model = None
        self.velocity_model = None
        self.time = None

        self.met = None
        self.consts = None


        if METSIM_IMPORT:

            # Load simulation parameters from a file
            self.met, self.consts = loadInputs(config.met_sim_input_file)

            # Set physical parameters of the meteor
            self.met.m_init = self.mass
            self.met.rho = self.density
            self.met.Gamma = self.Gamma
            self.met.Lambda = self.Lambda

            # Apply the ablation coefficient to the heat of ablation
            self.met.q = self.Lambda/(2*(self.ablation_coeff/10**6)*self.Gamma)

            # Luminous efficiency
            self.met.lum_eff = self.lum_eff

            # Flag for checking if the simulation was run the first time or not
            self.sim_executed = False

        else:
            raise NotImplementedError("""The numerical meteor ablation simulation code (Campbell-Brown and 
                Koschny 2004) is not available in this version of the code. Please use a different velocity 
                model, such as liinear or exponential.""")



    def getTimeData(self, fps):
        """ Returns an array of time data for the meteor. 
        
        Arguments:
            fps: [float] Frames per second of the camera.

        """

        return np.arange(0, np.max(self.time), 1.0/fps)



    def getSimulation(self, v_init, zangle, beg_height):
        """ Runs the meteor ablation simulation. 
        
        Arguments:
            v_init: [float] Velocity at t = -infinity. In m/s.
            zangle: [float] Zenith angle (radians).
            beg_height: [float] Beginning height (meters).
    
        """

        # Set velocity @180km
        self.met.v_init = v_init

        # Set the zenith angle
        self.consts.zr = zangle

        # Run the simulation with the given parameters
        sim_results = runSimulation(self.met, self.consts)

        # Get the results
        sim_results = np.array(sim_results)

        # Unpack results
        time, height, trail, velocity, luminosity = sim_results.T

        # If the meteor is not ablating at all, skip the meteor
        if len(time) == 0:
            return False

        # Set time 0 for the moment when the simulated height matches the given beginning height
        heights_diffs = np.abs(height - beg_height)
        closest_index = np.argwhere(heights_diffs == np.min(heights_diffs))[0][0]

        self.time = time[closest_index:]
        self.height = height[closest_index:]
        self.trail = trail[closest_index:]
        self.velocity = velocity[closest_index:]
        self.luminosity = luminosity[closest_index:]

        # Normalize time and length
        self.time  -= np.min(self.time)
        self.trail -= np.min(self.trail)
        
        # Fit a spline model to time vs. length
        self.length_model = scipy.interpolate.PchipInterpolator(self.time, self.trail, extrapolate=True)

        # Fit a spline model to time vs. luminosity
        self.luminosity_model = scipy.interpolate.PchipInterpolator(self.time, self.luminosity, \
            extrapolate=True)

        # Fit a spline model to time vs. velocity
        self.velocity_model = scipy.interpolate.PchipInterpolator(self.time, self.velocity, \
            extrapolate=True)



    def getLength(self, v_init, t):
        """ Calculates the length along the track at the given time with the given velocity using the 
            numerical ablation model.

        Arguments:
            v_init: [float] Velocity at t = -infinity. In m/s. (NOT USED IN THIS MODEL!)
            t: [float] Time at which the length along the track will be evaluated.

        """

        return self.length_model(t)



    def __repr__(self):
        """ Returned upon printing the object. """

        out_str = ''
        out_str += "Ablation model\n"
        out_str += "Mass      = {:.2E}\n".format(self.mass) + "\n"
        out_str += "Density   = {:.2f}\n".format(self.density) + "\n"
        out_str += "Gamma     = {:.2f}\n".format(self.Gamma) + "\n"
        out_str += "Lambda    = {:.2f}\n".format(self.Lambda) + "\n"
        out_str += "Abl coeff = {:.2f}\n".format(self.ablation_coeff) + "\n"
        out_str += "Lum eff   = {:.2f}\%\n".format(100*self.lum_eff) + "\n"

        return out_str

            


def initStationList(stations_geo, azim_fovs, elev_fovs, fov_widths, fov_heights, t_offsets=None, \
    fps_list=None, obs_ang_uncertainties=None, lim_magnitudes=None, P_0m_list=None):
    """ Given the parameters of stations, initializes SimStation objects and returns a list of them. 
    
    Arguments:
        stations_geo: [list] List of geographical coordinates of stations and their names:
            - lat: [float] Latitude in degrees.
            - lon: [float] Longitude in degrees.
            - elev: [float] Elevation in meters.
            - station_if: [str] Station name or ID.
        azim_fovs: [list] List of azimuth of the centres of the FOV for every station (degrees).
        elev_fovs: [list] List of altitudes of the centres of the FOV for every station (degrees).
        fov_widths: [list] List of widths of the field of views for every station (degrees).
        fov_heights: [list] List of heights of the field of views for every station (degrees).

    Keyword arguments:
        t_offsets: [list] List of time offsets of stations from real time (seconds).
        fps_list: [list] List of frames per second of each camera.
        obs_ang_uncertanties: [list] Observation precision of every station in arcseconds.
        lim_magnitudes: [list] List of limiting magnitudes for meteors per every station. Only needed for 
        ablation simulations. Default is +3 if not given.
        P_0m_list: [list] List of powers of zero-magnitude meteors in Watts. Only needed for ablation 
            simulations. Default is 840 W if not given.

    Return:
        station_list: [list] A list of SimStation objects with initialized values.

    """

    if t_offsets is None:
        t_offsets = [0]*len(stations_geo)

    if fps_list is None:
        fps_list = [0]*len(stations_geo)

    if obs_ang_uncertainties is None:
        obs_ang_uncertainties = [0]*len(stations_geo)

    if lim_magnitudes is None:
        lim_magnitudes = [3]*len(stations_geo)

    if P_0m_list is None:
        P_0m_list = [840]*len(stations_geo)



    station_list = []
    
    # Generate SimStation objects
    for t_offset, stat_geo, fps, obs_ang_std, azim_centre, elev_centre, fov_wid, fov_ht, lim_mag, P_0m in \
        zip(t_offsets, stations_geo, fps_list, obs_ang_uncertainties, azim_fovs, elev_fovs, fov_widths, \
            fov_heights, lim_magnitudes, P_0m_list):

        # Unpack geographical station coords
        lat, lon, elev, station_id = stat_geo

        # Convert geographical coordinates to radians
        lat = np.radians(lat)
        lon = np.radians(lon)

        # Convert angular uncertanties to radians
        obs_ang_std = np.radians(obs_ang_std/3600)

        # Convert FOV to radians
        azim_centre = np.radians(azim_centre)
        elev_centre = np.radians(elev_centre)
        fov_wid = np.radians(fov_wid)
        fov_ht = np.radians(fov_ht)


        # Make a new station object
        stat = SimStation(lat, lon, elev, station_id, fps, t_offset, obs_ang_std, azim_centre, \
            elev_centre, fov_wid, fov_ht, lim_mag=lim_mag, P_0m=P_0m)


        station_list.append(stat)


    return station_list




def plotStationFovs(station_list, jd, min_height, max_height, extra_points=None):
    """ Plots a 3D plot showing FOVs of stations in ECI coordinates at the given time.

    Arguments:
        station_list: [list] A list of SimStation objects containing station info.
        jd: [float] Julian date for calculating ECI coordinates of FOVs. This date can be arbitrary.
        min_height: [float] Bottom height of the FOV polyhedron (meters).
        max_height: [float] Ceiling height of the FOV polyhedron (meters).

    Keyword arguments:
        extra_points: [list] A list of other points that will be plotted. None by default.

    """

    # Calculate ECI coordinates of corners of the FOV volume at the given heights for every station
    fov_eci_corners = []


    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    import scipy.spatial

    from matplotlib.pyplot import cm 



    # Calculate ECI positions of the FOV vectors at given height
    for stat in station_list:

        # Get ECI coordinates of the FOV at the given maximum height
        top_eci_corners = stat.fovCornersToECI(jd, max_height)

        # Add the ECI position of the station to the vertices list
        stat_eci = geo2Cartesian(stat.lat, stat.lon, stat.elev, jd)
        top_eci_corners.append(stat_eci)

        # Get ECI coordinates of the FOV at the given minimum height
        bottom_eci_corners = stat.fovCornersToECI(jd, min_height)

        fov_eci_corners.append(top_eci_corners + bottom_eci_corners)




    # Plot FOVs

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = cm.rainbow(np.linspace(0, 1, len(station_list)))

    # Plot ECI corners
    for verts, color in zip(fov_eci_corners, colors):

        verts = np.array(verts)
        x, y, z = verts.T

        hull = scipy.spatial.ConvexHull(verts)

        # Plot vertices
        ax.plot(verts.T[0], verts.T[1], verts.T[2], c=color)

        # Plot edges
        for simp in hull.simplices:

            # Cycle to the first coordinate
            simp = np.append(simp, simp[0])

            # Plot the edge
            ax.plot(verts[simp, 0], verts[simp, 1], verts[simp, 2], c=color)


    # Plot the extra point if they were given
    if extra_points is not None:

        for point in extra_points:
            ax.scatter(*point)

    
    plt.show()




def plotMeteorTracks(station_list, sim_meteor_list, output_dir='.', save_plots=True):
    """ Plots tracks of simulated meteors on a map. 
    
    Arguments:
        station_list: [list] A list of SimStation objects containing station info.
        sim_meteor_list: [list] A list of SimMeteor objects which contain information about generated
            meteors.

    Keyword arguments:
        output_dir: [str] Directory where the plots will be saved.
        save_plot: [str] Save plots if True.
    """

    from mpl_toolkits.basemap import Basemap


    # Calculate the mean latitude and longitude of all points to be plotted
    lon_mean = np.degrees(meanAngle([sim_met.rbeg_lon for sim_met in sim_meteor_list] + [stat.lon for stat in station_list]))
    lat_mean = np.degrees(meanAngle([sim_met.rbeg_lat for sim_met in sim_meteor_list] + [stat.lat for stat in station_list]))


    # Put coordinate of all sites and the meteor in the one list
    geo_coords = [[stat.lat, stat.lon] for stat in station_list]
    for sim_met in sim_meteor_list:
        geo_coords.append([sim_met.rbeg_lat, sim_met.rbeg_lon])

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

    m = Basemap(projection='gnom', lat_0=lat_mean, lon_0=lon_mean, width=2*max_dist, height=2*max_dist, 
        resolution='i')

    # Draw the coast boundary and fill the oceans with the given color
    m.drawmapboundary(fill_color='0.2')

    # Fill continents, set lake color same as ocean color
    m.fillcontinents(color='black', lake_color='0.2', zorder=1)

    # Draw country borders
    m.drawcountries(color='0.2')
    m.drawstates(color='0.15', linestyle='--')

    # Draw parallels
    parallels = np.arange(-90, 90, 1.)
    m.drawparallels(parallels, labels=[False, True, False, False], color='0.25')

    # Draw meridians
    meridians = np.arange(0, 360, 1.)
    m.drawmeridians(meridians, labels=[False, False, False, True], color='0.25')

    # Plot locations of all stations and measured positions of the meteor
    for stat in station_list:

        # Plot stations
        x, y = m(np.degrees(stat.lon), np.degrees(stat.lat))
        m.scatter(x, y, s=10, label=str(stat.station_id), marker='x', zorder=3)


    for sim_met in sim_meteor_list:
        
        # Plot a point marking the position of initial state vectors of all meteors
        x_end, y_end = m(np.degrees(sim_met.rbeg_lon), np.degrees(sim_met.rbeg_lat))
        m.scatter(x_end, y_end, c='g', marker='+', s=50, alpha=0.75, zorder=3)

        # Plot simulated tracks
        if sim_met.model_lat:

            for model_lat, model_lon in zip(sim_met.model_lat, sim_met.model_lon):
            
                x, y = m(np.degrees(model_lon), np.degrees(model_lat))
                m.plot(x, y, c='r')


    ## Plot the map scale
    
    # Get XY cordinate of the lower left corner
    ll_x, _ = plt.gca().get_xlim()
    ll_y, _ = plt.gca().get_ylim()

    # Move the label to fit in the lower left corner
    ll_x += 0.2*2*max_dist
    ll_y += 0.1*2*max_dist

    # Convert XY to latitude, longitude
    ll_lon, ll_lat = m(ll_x, ll_y, inverse=True)

    # Round to distance to the closest 10 km
    scale_range = round(max_dist/2/1000/10, 0)*10

    # Plot the scale
    m.drawmapscale(ll_lon, ll_lat, lon_mean, lat_mean, scale_range, barstyle='fancy', units='km', 
        fontcolor='0.5', zorder=3)


    plt.legend()


    if save_plots:
        savePlot(plt, 'trajectories', output_dir=output_dir)


    plt.show()




def stationFovOverlap(station_list, jd, height_min, height_max):
    """ Checks if the station FOVs are overlapping at the given range of heights. 

    The algorithm uses a crude collision detection method, where first it is checked is points from one FOV
    convex hull are in other station's FOV convex hull. If not, the algorithm pairs all points in from one 
    convex hull with each other, finds the point that is between those two and checkes if that point is inside
    other convex hulls.

    Arguments:
        station_list: [list] A list of SimStation objects containing station info.
        jd: [float] Julian date for calculating ECI coordinates of FOVs. This date can be arbitrary.
        height_min: [float] Bottom height of the FOV polyhedron (meters).
        height_max: [float] Upper height of the FOV polyhedron (meters).

    Return:
        (overlap_status, fov_eci_corners):
            overlap: [bool] True is there is an overlap between the FOVs, False otherwise.
            fov_eci_corners: [list] A list of ECI coordinates of FOV corners at given heights.

    """

    # Calculate ECI coordinates of corners of the FOV volume at the given heights for every station
    fov_eci_corners = []

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    import scipy.spatial

    from matplotlib.pyplot import cm 




    for stat in station_list:

        # Get ECI coordinates of the FOV at the given maximum height
        top_eci_corners = stat.fovCornersToECI(jd, height_max)

        # Get ECI coordinates of the FOV at the given minimum height
        bottom_eci_corners = stat.fovCornersToECI(jd, height_min)

        fov_eci_corners.append(top_eci_corners + bottom_eci_corners)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = cm.rainbow(np.linspace(0, 1, len(station_list)))

    # Plot ECI corners
    for verts, color in zip(fov_eci_corners, colors):

        verts = np.array(verts)
        x, y, z = verts.T
        ax.scatter(x, y, z)

        hull = scipy.spatial.ConvexHull(verts)

        # Plot defining corner points
        ax.plot(verts.T[0], verts.T[1], verts.T[2], c=color)

        # 12 = 2 * 6 faces are the simplices (2 simplices per square face)
        for s in hull.simplices:
            s = np.append(s, s[0])  # Here we cycle back to the first coordinate
            ax.plot(verts[s, 0], verts[s, 1], verts[s, 2], c=color)



    # ax.set_xlim([-1, 1])
    # ax.set_ylim([-1, 1])
    # ax.set_zlim([-1, 1])
    
    plt.show()




    ### FIRST CHECK - VECTICES FROM ONE CONVEX HULL IN ANOTHER ###
    ##########################################################################################################

    for i, stat1 in enumerate(station_list):
        for j, stat2 in enumerate(station_list):

            # Skip if is the same station
            if i == j:
                continue

            # Check if any vertices from one station's FOV convex hull are in another
            for vert in fov_eci_corners[i]:

                if pointInsideConvexHull(fov_eci_corners[j], vert):

                    # If the point is inside the hull, that means that the FOVs are intersecing
                    return True, fov_eci_corners


    ##########################################################################################################


    ### SECOND CHECK - MIDPOINTS OF VERTEX PAIRS FROM ONE CONVEX HULL IN ANOTHER ###
    ##########################################################################################################

    for i, stat1 in enumerate(station_list):
        for j, stat2 in enumerate(station_list):

            # Skip if is the same station
            if i == j:
                continue

            # Pair all vertices
            for m, vert1 in enumerate(fov_eci_corners[i]):
                for n, vert2 in enumerate(fov_eci_corners[i]):

                    # Skip the same vertex
                    if m == n:
                        continue

                    # Calculate the midpoint between the two vertices
                    midpoint = np.mean([vert1, vert2], axis=0)

                    # Check if the midpoint is inside the FOV of the other station
                    if pointInsideConvexHull(fov_eci_corners[j], midpoint):
                        return True, fov_eci_corners

    ##########################################################################################################
    
    # If no common points were found, there is a large probability (but no 100%!!!) that the FOVs do not 
    # overlap
    return False, fov_eci_corners




def sampleActivityModel(b, sol_max, n_samples=1):
    """ Drawing samples from a probability distribution representing activity of a meteor shower. The sampling
        is done using the Inverse transform sampling method. Activity model taken from: Jenniskens, P. (1994). 
        Meteor stream activity I. The annual streams. Astronomy and Astrophysics, 287., equation 8.

    Arguments:
        sol_max: [float] Solar longitude of the maximum shower activity (radians).
        b: [float] Slope of the activity profile.

    Keyword arguments:
        n_samples: [float] Number of samples to be drawn from the activity profile distribution.

    """

    y = np.random.uniform(0, 1, size=n_samples)

    # Draw samples from the inverted distribution
    samples = np.sign(np.random.uniform(-1, 1, size=n_samples))*np.log10(y)/b + np.degrees(sol_max)

    return np.radians(samples)%(2*np.pi)



def activityGenerator(b, sol_max):
    """ Generator which returns one value of solar longitude upon every call. 
    
    Arguments:
        sol_max: [float] Solar longitude of the maximum shower activity (radians).
        b: [float] Slope of the activity profile.

    """

    while True:
        yield sampleActivityModel(b, sol_max)[0]




def sampleMass(mass_min, mass_max, mass_index, n_samples):
    """ Sample a meteor shower mass distribution using the given mass range and mass index. Used for ablation
        model meteoroids.

    Arguments:
        mass_min: [float] Logarithm of minimum mass in kilograms (e.g. -3 for 1 gram).
        mass_max: [float] Logarithm of maximum mass in kilograms (e.g. 0 for 1 kilogram).
        mass_index: [float] Shower mass index.
        n_samples: [int] Number of samples to take from the distribution.


    Return:
        masses: [list] A list of masses in kilograms.

    """

    def _massIndex(m, alpha):
        return m**(-alpha)

    alpha = mass_index - 1

    # Construct the cumulative distribution function
    mass_range = np.logspace(mass_min, mass_max, 100)
    mass_frequency = _massIndex(mass_range, alpha)

    # Normalize the distribution to (0, 1) range
    mass_frequency -= np.min(mass_frequency)
    mass_frequency = mass_frequency/np.max(mass_frequency)

    # Generate random numbers in the (0, 1) range
    rand_nums = np.random.uniform(0, 1, n_samples)

    # Flip arrays for interpolation (X axis must be increasing)
    mass_frequency = mass_frequency[::-1]
    mass_range = mass_range[::-1]

    # Sample the distribution
    masses = np.interp(rand_nums, mass_frequency, mass_range)


    # print(masses)

    # # PLOT DISTRIBUTION AND DRAWN MASSES
    # n, bins, _ = plt.hist(masses, bins=np.logspace(mass_min, mass_max, np.floor(np.sqrt(n_samples))))

    # # Normalize the mass function to the number of samples drawn
    # plt.semilogx(mass_range, np.max(n)*mass_frequency, label='s = {:.2f}'.format(mass_index))

    # plt.gca().set_xscale("log")

    # plt.xlabel('Mass (kg)')
    # plt.ylabel('Count')

    # plt.legend()

    # plt.show()


    return masses



def sampleDensity(log_rho_mean, log_rho_sigma, n_samples):
    """ Sample a meteor shower density distribution using the given mean density and stddev. Used for ablation
        model meteoroids. The density distribution is samples from a Gaussian distribution on logarithmic 
        values of densities.

    Arguments:
        log_rho_mean: [float] Mean of the logarithmic density distribution.
        log_rho_sigma: [float] Standard deviation of the logarithmic density distribution.
        n_samples: [int] Number of samples to take from the distribution.
    
    Return:
        [ndarray of float] Densities in kg/m^3

    """

    # Draw samples from a normal distribution of densities
    log_densitites = np.random.normal(log_rho_mean, log_rho_sigma, n_samples)

    # Convert the densities to normal values
    return 10**log_densitites



def generateStateVector(station_list, jd, beg_height_min, beg_height_max):
    """ Generate the initial state vector in ECI coordinates for the given meteor. 
    
    Arguments:
        station_list: [list] A list of SimStation objects containing station info.
        jd: [float] Julian date for calculating ECI coordinates of FOVs. This date can be arbitrary.
        beg_height_min: [float] Bottom height of the FOV polyhedron (meters).
        beg_height_max: [float] Upper height of the FOV polyhedron (meters).

    """

    stat_first = station_list[0]

    # Get the FOV box of the first station at the given station
    fov_box_first = stat_first.fovCornersToECI(jd, beg_height_min, beg_height_max)


    # Calculate FOV boxes for all stations with a larger height range, to avoid geometric problems
    # when a point near the heighest/lowest heights is being generated
    fov_boxes_all = []
    for stat in station_list:

        # Get the FOV box of the given station
        fov_box_stat = stat.fovCornersToECI(jd, 2*beg_height_max)

        # Add the station's ECI position to the list of vertices
        fov_box_stat.append(geo2Cartesian(stat.lat, stat.lon, stat.elev, jd))

        fov_boxes_all.append(fov_box_stat)

    ### TEST ###
    #plotStationFovs(station_list, jd, 1000*beg_height)
    ###

    # Find a point at the given beginning height that is inside all FOVs
    while True:
        
        # Sample one point from the FOV box of the first station
        state_vect = samplePointsFromHull(fov_box_first, 1)[0]

        # Project the point on the given beginning height
        pt_lat, pt_lon, pt_ht = cartesian2Geo(jd, *state_vect)
        state_vect = geo2Cartesian(pt_lat, pt_lon, 1000*beg_height, jd)

        # Check if the point is inside FOV boxes of all stations
        for fov_box_stat in fov_boxes_all:

            # # Get the FOV box of the given station
            # fov_box_stat = stat.fovCornersToECI(jd, 1000*min(beg_height_data), \
            #     beg_height_max)

            # Check if the point is inside the FOV box of the given station
            inside_test = pointInsideConvexHull(fov_box_stat, state_vect)

            # If the point is not inside one of the boxes, find another point
            if inside_test == False:
                break

        # If the point is not inside one of the boxes, find another point
        if inside_test == False:
            continue

        ### TEST ###
        # plotStationFovs(station_list, jd, 1000*beg_height, [state_vect])
        ###


        # If the point is inside all boxes, stop the search
        break


    return state_vect




def simulateMeteorShower(station_list, meteor_velocity_models, n_meteors, ra_g, ra_g_sigma, dec_g, 
    dec_g_sigma, d_ra, d_dec, v_g, v_g_sigma, year, month, sol_max, sol_slope, beg_height, beg_height_sigma,
    output_dir='.', save_plots=True):
    """ Given the parameters of a meteor shower, generate simulated meteors and their trajectories.

    Arguments:
        station_list: [list] A list of SimStation objects which defines the location and parameters of 
            stations observing the shower.
        meteor_velocity_models: [list] A list of meteor velocity models (ConstantVeloicty, JacchiaVelocity, 
            etc.)
        n_meteors: [int] Number of simulated meteor radiants to draw.
        ra_g: [float] Right ascension of centre of geocentric radiant (degrees).
        ra_g_sigma: [float] R.A. standard deviation of the radiant (degrees).
        dec_g: [float] Declination of centre of geocentric radiant (degrees).
        dec_g_sigma: [float] Declination standard deviation of the radiant (degrees).
        d_ra: [float] R.A. radiant drift (degrees of R.A. per radian of solar longitude).
        d_dec: [float] Dec radiant drift (degrees of declination per radian of solar longitude).
        v_g: [float] Mean geocentric velocity (km/s).
        v_g_sigma: [float] Standard deviation of the geocentric velocity (km/s).
        year: [int] Year of the meteor shower.
        month: [int] Month of the meteor shower.
        sol_max: [float] Solar longitude of the maximum shower activity (degrees).
        sol_slope: [float] Slope of the activity profile.
        beg_height: [float] Mean of Gaussian beginning height profile (km).
        beg_height_sigma: [float] Standard deviation of beginning height profile (km).


    Keyword arguments:
        output_dir: [str] Directory where the plots will be saved.
        save_plot: [str] Save plots if True.

    Return:
        sim_meteor_list: [list] A list of SimMeteor objects which contain simulated meteors.

    """

    # Convert angles to radians
    ra_g = np.radians(ra_g)
    ra_g_sigma = np.radians(ra_g_sigma)

    dec_g = np.radians(dec_g)
    dec_g_sigma = np.radians(dec_g_sigma)

    sol_max = np.radians(sol_max)

    # Convert velocity to m/s
    v_g = 1000*v_g
    v_g_sigma = 1000*v_g_sigma

    

    ### GENERATE SOLAR LONGITUDES ###
    ##########################################################################################################

    sol_data = []
    jd_data = []


    # Draw solar longitudes for meteors only during the night for all stations
    for sol in activityGenerator(sol_slope, sol_max):

        # Calculate the corresponding Julian date for the drawn solar longitude
        meteor_jd = solLon2jdJPL(year, month, sol)

        night_start_list = []
        night_end_list = []

        # Find the common time of night for all stations (local time)
        for stat in station_list:

            # Calculate the beginning and the end of astronomical night for the given station at the time
            night_start, night_end = astronomicalNight(jd2Date(meteor_jd, dt_obj=True), stat.lat, \
                stat.lon, stat.elev)

            night_start_list.append(night_start)
            night_end_list.append(night_end)

        # Select only the common parts of the night seen from all stations
        night_start = max(night_start_list)
        night_end = min(night_end_list)

        # Check if the given solar longitude is during the common night from all stations
        if (datetime2JD(night_start) < meteor_jd) and (datetime2JD(night_end) > meteor_jd):
            
            # Add the solar longitude to the final list
            sol_data.append(sol)
            jd_data.append(meteor_jd)

        # Check if there are enough solar longitudes generated
        if len(sol_data) == n_meteors:
            break


    sol_data = np.array(sol_data)
    jd_data = np.array(jd_data)

    # Sort the results by Julian date
    sort_temp = np.c_[jd_data, sol_data]
    sort_temp = sort_temp[np.argsort(sort_temp[:, 0])]
    jd_data, sol_data = sort_temp.T

    ##########################################################################################################


    plt.hist(np.degrees(sol_data))
    plt.xlabel('Solar longitude (deg)')
    plt.ylabel('Counts')
    plt.title('Activity profile')

    if save_plots:
        savePlot(plt, 'activity.png', output_dir=output_dir)

    plt.show()


    ### GENERATE GEOCENTRIC RADIANTS ###
    ##########################################################################################################

    # # Draw R.A. and Dec from a bivariate Gaussian centred at (0, 0)
    # mean = (0, 0, v_g)
    # cov = [[ra_g_sigma, 0, 0], [0, dec_g_sigma, 0], [0, 0, v_g_sigma]]
    # ra_g_data, dec_g_data, v_g_data = np.random.multivariate_normal(mean, cov, n_meteors).T

    # Sample radiant positions from a von Mises distribution centred at (0, 0)
    ra_g_data = np.random.vonmises(0, 1.0/(ra_g_sigma**2), n_meteors)
    dec_g_data = np.random.vonmises(0, 1.0/(dec_g_sigma**2), n_meteors)


    ra_rot_list = []
    dec_rot_list = []

    # Go through all generated RA, Dec and project them properly on a sphere
    for ra, dec in zip(ra_g_data, dec_g_data):

        # Rotate R.A., Dec from (0, 0) to generated coordinates, to account for spherical nature of the angles
        # After rotation, (RA, Dec) will still be scattered around (0, 0)
        ra_rot, dec_rot = rotatePolar(0, 0, ra, dec)

        # Rotate all angles scattered around (0, 0) to the given coordinates of the centre of the distribution
        ra_rot, dec_rot = rotatePolar(ra_rot, dec_rot, ra_g, dec_g)

        ra_rot_list.append(ra_rot)
        dec_rot_list.append(dec_rot)


    ra_g_data = np.array(ra_rot_list)
    dec_g_data = np.array(dec_rot_list)


    # Apply the radiant drift
    ra_g_data = ra_g_data + d_ra*(sol_data - sol_max)
    dec_g_data = dec_g_data + d_dec*(sol_data - sol_max)


    ##########################################################################################################


    # Generate geocentric velocities from a Gaussian distribution
    v_g_data = np.random.normal(v_g, v_g_sigma, size=n_meteors)

    plt.hist(v_g_data/1000)
    plt.xlabel('Geocentric velocity (km/s)')
    plt.ylabel('Counts')
    plt.title('Geocentric velocities')

    if save_plots:
        savePlot(plt, 'vg.png', output_dir=output_dir)

    plt.show()


    # Draw beginning heights from a Gaussian distribution
    beg_height_data = np.random.normal(beg_height, beg_height_sigma, size=n_meteors)

    plt.hist(beg_height_data, orientation='horizontal')
    plt.xlabel('Counts')
    plt.ylabel('Beginning height (km)')
    plt.title('Beginning heights')

    if save_plots:
        savePlot(plt, 'beginning_heights.png', output_dir=output_dir)

    plt.show()


    # Plot radiants on the sky
    m = CelestialPlot(ra_g_data, dec_g_data, projection='stere')

    m.scatter(ra_g_data, dec_g_data, c=v_g_data/1000, s=2)

    m.colorbar(label='$V_g$ (km/s)')

    if save_plots:
        savePlot(plt, 'geo_radiants.png', output_dir=output_dir)

    plt.show()



    # Make sure that the range of beginning heights is at least 10km
    if max(beg_height_data) - min(beg_height_data) < 10:

        beg_height_max = 1000*(beg_height + 5)
        beg_height_min = 1000*(beg_height - 5)

    else:
        beg_height_min = 1000*min(beg_height_data)
        beg_height_max = 1000*max(beg_height_data)
    
    

    ### Generate initial state vectors for drawn shower meteors inside the FOVs of given stations ###
    ##########################################################################################################

    # Check if the station FOVs are overlapping at all at given heights
    if not stationFovOverlap(station_list, np.mean(jd_data), beg_height_min, beg_height_max):
        
        print('ERROR! FOVs of stations are not overlapping!')
        sys.exit()



    # If there is an overlap
    else:

        state_vector_data = []

        # Go through all meteors
        for jd, beg_height in zip(jd_data, beg_height_data):

            # Generate state vector
            state_vect = generateStateVector(station_list, jd, beg_height_min, beg_height_max)

            # Put the found state vector in the list
            state_vector_data.append(state_vect)

    state_vector_data = np.array(state_vector_data)


    ##########################################################################################################



    ### GENERATE SimMeteor OBJECTS ###
    ##########################################################################################################

    sim_meteor_list = []

    # Put all simulated meteors into SimMeteor objects
    for ra_g, dec_g, v_g, sol, jd, beg_height, state_vect in zip(ra_g_data, dec_g_data, v_g_data, sol_data,\
        jd_data, beg_height_data, state_vector_data):

        sim_meteor_list.append(SimMeteor(ra_g, dec_g, v_g, year, month, sol, jd, beg_height, state_vect))

    ##########################################################################################################



    ### ADD TRAJECTORIES TO GENERATED METEORS USING THE GIVEN VELOCITY MODELS ###
    ##########################################################################################################

    for i, (sim_met, velocity_model) in enumerate(zip(sim_meteor_list, meteor_velocity_models)):

        # Generate trajectory data for the given meteor
        sim_meteor = generateTrajectoryData(station_list, sim_met, velocity_model)

        sim_meteor_list[i] = sim_meteor


    ##########################################################################################################



    return sim_meteor_list



def calcGravityDrop(eci_coord, t):
    """ Given the ECI position of the meteor and the duration of flight, this function calculates the
        drop caused by gravity and returns ECI coordinates of the meteor corrected for gravity drop.

    Arguments:
        eci_coord: [ndarray] (x, y, z) ECI coordinates of the meteor at the given time t (meters).
        t: [float] Time of meteor since the beginning of the trajectory.

    """

    # Calculate gravitational acceleration at given ECI coordinates
    r_earth = (EARTH.EQUATORIAL_RADIUS + EARTH.POLAR_RADIUS)/2
    g = 9.81*(r_earth/vectMag(eci_coord))**2

    # Calculate the amount of gravity drop from a straight trajectory
    drop = (1.0/2)*g*t**2

    # Apply gravity drop to ECI coordinates
    eci_coord -= drop*vectNorm(eci_coord)

    return eci_coord
    



def generateTrajectoryData(station_list, sim_met, velocity_model):
    """ Calculates trajectory points given constant velocity, i.e. no deceleration. 
    
    Arguments:
        station_list: [list] A list of SimStation objects which defines the location and parameters of 
            stations observing the shower.
        sim_met: [SimMeteor object] Simulated meteor object for which the trajectory data will be generated.
        velocity_model: [object] Velocity model (Constant, Jacchia, etc.) used for generating the simulated
            meteor.

    Return:
        sim_met: [SimMeteor object] Simulated meteor object with generated trajectory data.

    """

    # Assign the velocity model
    sim_met.velocity_model = velocity_model

    # Go through every station
    for stat in station_list:

        # If the velocity model is given by the ablation model, run the model first
        if sim_met.velocity_model.name == 'ablation':
            sim_met.velocity_model.getSimulation(sim_met.v_init, sim_met.orb.zc, sim_met.rbeg_ele)

            # If the simulation did not run, skip the station
            if sim_met.velocity_model.time is None:
                continue


        # Generate time data
        time_data_model = velocity_model.getTimeData(stat.fps)

        azim_data = []
        elev_data = []
        time_data = []

        model_lat_data = []
        model_lon_data = []
        model_elev_data = []

        first_point = True

        # Go through all trajectory points
        for t in time_data_model:

            # Calculate the Julian date of the trajectory point
            jd = sim_met.jdt_ref + t/86400.0

            # Calculate the length along the trail using the given velocity model
            length = velocity_model.getLength(sim_met.v_init, t)

            # Calculate the state vector position at every point in time
            traj_eci = sim_met.state_vect + length*(-sim_met.radiant_eci)

            # Apply gravity drop to calculated ECI coordinates
            traj_eci = calcGravityDrop(traj_eci, t)


            # If the model is given by the ablation code, check that the magnitude of the meteor was above the
            # detection threshold
            if sim_met.velocity_model.name == 'ablation':

                # Calculate the absolute magnitude (magnitude @100km) at this point in time
                abs_mag = -2.5*np.log10(sim_met.velocity_model.luminosity_model(t)/stat.P_0m)

                # Calculate the range to the station
                stat_range = vectMag(geo2Cartesian(stat.lat, stat.lon, stat.elev, jd) \
                    - traj_eci)

                # Calculate the visual magnitude
                magnitude = abs_mag - 5*np.log10(100000.0/stat_range)

                # Skip this point if the meteor is fainter than the limiting magnitude at this point
                if magnitude > stat.lim_mag:
                    continue


                # Set the current velocity at this point
                current_velocity = sim_met.velocity_model.velocity_model(t)

            else:

                current_velocity = sim_met.v_init


            ### Take only those points inside the FOV of the observer ###
            ##################################################################################################

            # Get ECI coordinates of the station FOV polyhedron
            fov_corners = stat.fovCornersToECI(jd, 2*sim_met.rbeg_ele)

            # Calculate the ECI position of the station at the particular point in time
            stat_eci = geo2Cartesian(stat.lat, stat.lon, stat.elev, jd)

            # Add the ECI position of the station to the vertices list
            fov_corners.append(stat_eci)

            # If the point is not inside the FOV, skip it
            if not pointInsideConvexHull(fov_corners, traj_eci):
                continue

            ##################################################################################################


            ### Project modelled points to the perspective of the observer at the given station ###
            ##################################################################################################

            # Calculate the unit direction vector pointing from the station to the trajectory point
            model_eci = vectNorm(traj_eci - stat_eci)

            # Calculate RA, Dec for the given point
            ra, dec = eci2RaDec(model_eci)


            # Calculate azimuth and altitude of this direction vector
            azim, elev = raDec2AltAz(ra, dec, jd, stat.lat, stat.lon)

            ##################################################################################################

            
            # Add Gaussian noise to simulated horizontal coordinates
            azim += np.random.normal(0, stat.obs_ang_std)
            elev += np.random.normal(0, stat.obs_ang_std)


            # Add final values to the results list
            azim_data.append(azim)
            elev_data.append(elev)
            time_data.append(t)

            # Calculate geographical coordinates of every trajectory point
            model_lat, model_lon, model_elev = cartesian2Geo(jd, *traj_eci)

            model_lat_data.append(model_lat)
            model_lon_data.append(model_lon)
            model_elev_data.append(model_elev)

            # Set the velocity at the first detected point
            if first_point:
                sim_met.v_begin = current_velocity
                first_point = False

        

        azim_data = np.array(azim_data)
        elev_data = np.array(elev_data)
        time_data = np.array(time_data)

        model_lat_data = np.array(model_lat_data)
        model_lon_data = np.array(model_lon_data)
        model_elev_data = np.array(model_elev_data)


        # Apply a random offset to time data
        time_data += stat.t_offset

        # Skip the observations if no points were inside the FOV
        if len(time_data) == 0:
            continue

        # Init new ObservedPoints with simulated values
        obs = ObservedPoints(sim_met.jdt_ref, azim_data, elev_data, time_data, stat.lat, stat.lon, stat.elev,
            meastype=2, station_id=stat.station_id)


        # Add the calculated points to meteor observations
        sim_met.observations.append(obs)

        # Add simulated geographical positions of the trajectory
        sim_met.model_lat.append(model_lat_data)
        sim_met.model_lon.append(model_lon_data)
        sim_met.model_elev.append(model_elev_data)


    return sim_met




if __name__ == "__main__":

    # Directory where the files will be saved
    dir_path = os.path.abspath('../SimulatedMeteors')

    ### CAMO STATION PARAMETERS ###
    ##########################################################################################################

    system_name = 'CAMO'

    # Number of stations in total
    n_stations = 2

    # Maximum time offset (seconds)
    t_max_offset = 1

    # Geographical coordinates of stations (lat, lon, elev, station_id) in degrees and meters
    stations_geo = [
        [43.26420, -80.77209, 329.0, 'tavis'], # Tavis
        [43.19279, -81.31565, 324.0, 'elgin']] # Elgin

    # Camera FPS per station
    fps_list = [110, 110]

    # Observation uncertanties per station (arcsec)
    obs_ang_uncertainties = [1, 1]

    # Azimuths of centre of FOVs (degrees)
    azim_fovs = [326.823, 1.891]

    # Elevations of centre of FOVs (degrees)
    elev_fovs = [41.104, 46.344]

    # Cameras FOV widths (degrees)
    fov_widths = [19.22, 19.22]

    # Cameras FOV heights (degrees)
    fov_heights = [25.77, 25.77]

    # Limiting magnitudes (needed only for ablation simulation)
    lim_magnitudes = [+5.5, +5.5]

    # Powers of zero-magnitude meteors (Watts) (needed only for ablation simulation)
    P_0m_list = [840, 840]

    # Define the mass range (log of mass in kg)
    mass_min = -6
    mass_max = -4

    ##########################################################################################################

    # ### SIMULATED MODERATE STATION PARAMETERS ###
    # ##########################################################################################################

    # system_name = 'CAMS'

    # # Number of stations in total
    # n_stations = 3

    # # Maximum time offset (seconds)
    # t_max_offset = 1

    # # Geographical coordinates of stations (lat, lon, elev, station_id) in degrees and meters
    # stations_geo = [
    #     [43.19279, -81.31565, 324.0, 'M1'], # M1 elgin
    #     [43.19055, -80.09913, 212.0, 'M2'],
    #     [43.96324, -80.80952, 383.0, 'M3']]

    # # Camera FPS per station
    # fps_list = [30, 30, 30]

    # # Observation uncertanties per station (arcsec)
    # obs_ang_uncertainties = [60, 60, 60]

    # # Azimuths of centre of FOVs (degrees)
    # azim_fovs = [56.0, 300.0, 174.0]

    # # Elevations of centre of FOVs (degrees)
    # elev_fovs = [65.0, 65.0, 65.0]

    # # Cameras FOV widths (degrees)
    # fov_widths = [64.0, 64.0, 64.0]

    # # Cameras FOV heights (degrees)
    # fov_heights = [48.0, 48.0, 48.0]

    # # Limiting magnitudes (needed only for ablation simulation)
    # lim_magnitudes = [3.75, 3.75, 3.75]

    # # Powers of zero-magnitude meteors (Watts) (needed only for ablation simulation)
    # P_0m_list = [1210, 1210, 1210]

    # # Define the mass range (log of mass in kg)
    # mass_min = -5
    # mass_max = -3

    # ##########################################################################################################


    # Randomly generate station timing offsets, the first station has zero time offset
    t_offsets = np.random.uniform(-t_max_offset, +t_max_offset, size=n_stations)
    t_offsets[0] = 0


    # Init stations data to SimStation objects
    station_list = initStationList(stations_geo, azim_fovs, elev_fovs, fov_widths, fov_heights, t_offsets, \
    fps_list, obs_ang_uncertainties, lim_magnitudes=lim_magnitudes, P_0m_list=P_0m_list)

    # Plot FOVs of stations at ceiling height of 120km
    plotStationFovs(station_list, datetime2JD(datetime.datetime.now()), 70*1000, 120*1000)


    ### METEOR SHOWER PARAMETERS ###
    ##########################################################################################################
    n_meteors = 50


    # ### GEMINIDS

    # shower_name = 'Geminids'

    # # Radiant position and dispersion
    # ra_g = 113.0
    # ra_g_sigma = 0.15

    # dec_g = 32.5
    # dec_g_sigma = 0.15

    # # Radiant drift in degrees per degree of solar longitude
    # d_ra = 1.05
    # d_dec = -0.17

    # # Geocentric velocity in km/s
    # v_g = 33.5
    # v_g_sigma = 0.1

    # year = 2012
    # month = 12

    # # Solar longitude of peak activity in degrees
    # sol_max = 261
    # sol_slope = 0.4

    # # Beginning height in kilometers
    # beg_height = 95
    # beg_height_sigma = 3

    # ###


    ### PERSEIDS

    # Shower name
    shower_name = 'Perseids'

    # Radiant position and dispersion
    ra_g = 48.2
    ra_g_sigma = 0.15

    dec_g = 58.1
    dec_g_sigma = 0.15

    # Radiant drift in degrees per degree of solar longitude
    d_ra = 1.40
    d_dec = 0.26

    # Geocentric velocity in km/s
    v_g = 59.1
    v_g_sigma = 0.1

    year = 2012
    month = 8

    # Solar longitude of peak activity in degrees
    sol_max = 140
    sol_slope = 0.4

    # Beginning height in kilometers
    beg_height = 105
    beg_height_sigma = 3

    ###


    ##########################################################################################################


    ### METEOR VELOCITY MODEL ###
    ##########################################################################################################

    # Set a range of meteor durations
    meteor_durations = np.clip(np.random.normal(0.5, 0.1, n_meteors), 0.2, 1.0)
    #meteor_durations = [1.5]*n_meteors

    # #### Constant velocity model
    # meteor_velocity_models = [ConstantVelocity(duration) for duration in meteor_durations]
    # ####


    # #### Jacchia (exponential deceleration) velocity model
    # a1_list = np.random.uniform(0.08, 0.15, n_meteors)
    # a2_list = np.random.uniform(8, 15, n_meteors)

    # meteor_velocity_models = [JacchiaVelocity(duration, a1, a2) for duration, a1, a2 in zip(meteor_durations,\
    #     a1_list, a2_list)]

    # ####


    #### Velocity model from Campbell-Brown & Koschny (2004) meteor ablation model

    # Make the beginning heights heigher, as the trajectory points will be determined by simulated
    # magnitudes
    beg_height = 120
    beg_height_sigma = 0

    # Luminous efficiency (fraction)
    lum_eff = 0.7/100

    # Ablation coefficient (s^2/km^2)
    ablation_coeff = 0.1

    # Drag coeficient
    Gamma = 1.0

    # Heat transfer coeficient
    Lambda = 0.5

    # Mass index
    mass_index = 2.0


    # Define density distribution (see: Moorhead et al. 2017 "A two-population sporadic meteoroid density 
    #        distribution and its implications for environment models")

    # HTC density distribution (Tj <= 2)
    log_rho_mean = 2.93320
    log_rho_sigma = 0.12714

    # # JFC and asteroidal distribution (Tj > 2) 
    # log_rho_mean = 3.57916
    # log_rho_sigma = 0.09312


    # Sample the masses
    mass_samples = sampleMass(mass_min, mass_max, mass_index, n_meteors)

    # Samples densities
    density_samples = sampleDensity(log_rho_mean, log_rho_sigma, n_meteors)


    # Init velocity models
    meteor_velocity_models = [AblationModelVelocity(mass, density, ablation_coeff, Gamma, Lambda, lum_eff) \
        for mass, density in zip(mass_samples, density_samples)]


    ####

    ##########################################################################################################


    # Make the system directory
    system_dir = os.path.join(dir_path, system_name)
    mkdirP(system_dir)

    # Make the shower directory
    shower_dir = os.path.join(system_dir, shower_name)
    mkdirP(shower_dir)


    
    # Trajectory solver (original or gural)
    traj_solvers = ['planes', 'los', 'monte_carlo', 'gural']



    # Run shower simulation
    sim_meteor_list = simulateMeteorShower(station_list, meteor_velocity_models, n_meteors, ra_g, ra_g_sigma, 
        dec_g, dec_g_sigma, d_ra, d_dec, v_g, v_g_sigma, year, month, sol_max, sol_slope, beg_height, 
        beg_height_sigma, output_dir=shower_dir)



    # Plot meteor positions
    plotMeteorTracks(station_list, sim_meteor_list, output_dir=shower_dir)

    # ## TEST
    # sys.exit()
    # ###
    

    # Solve generated trajectories
    for sim_met in sim_meteor_list:

        # Check that there are at least 2 sets of measurements
        if len(sim_met.observations) < 2:
            print('Skipped meteor at JD =', sim_met.jdt_ref, 'as it was not observable from at least 2 stations!')
            continue

        # Make sure there are at least 4 point from every station
        meas_counts = [len(obs.time_data) for obs in sim_met.observations]
        if meas_counts:
            if np.min(meas_counts) < 4:

                print('Skipped meteor at JD =', sim_met.jdt_ref, 'due to having less than 4 point from any of the stations!')
                continue

        else:
            print('Skipped meteor at JD =', sim_met.jdt_ref, 'due to having less than 4 point from any of the stations!')
            continue


        # Directory where trajectory results will be saved
        output_dir = os.path.join(shower_dir, str(sim_met.jdt_ref))


        # Save info about the simulated meteor
        sim_met.saveInfo(output_dir, t_offsets, obs_ang_uncertainties)


        # Solve the simulated meteor with multiple solvers
        for traj_solver in traj_solvers:


            if (traj_solver == 'los') or (traj_solver == 'planes'):

                # Init the trajectory (LoS or intersecing planes)
                traj = Trajectory(sim_met.jdt_ref, output_dir=output_dir, max_toffset=t_max_offset, \
                    meastype=2, show_plots=False, save_results=False, monte_carlo=False)


            elif traj_solver == 'monte_carlo':
                
                # Init the trajectory
                traj = Trajectory(sim_met.jdt_ref, output_dir=output_dir, max_toffset=t_max_offset, \
                    meastype=2, show_plots=False, mc_runs=250)

            elif traj_solver == 'gural':
                
                # Init the new Gural trajectory solver object
                traj = GuralTrajectory(len(sim_met.observations), sim_met.jdt_ref, output_dir=output_dir, 
                    max_toffset=t_max_offset, meastype=2, velmodel=3, verbose=1, show_plots=False)

            else:
                print(traj_solver, '- unknown trajectory solver!')
                sys.exit()


            # Fill in observations
            for obs in sim_met.observations:

                traj.infillTrajectory(obs.meas1, obs.meas2, obs.time_data, obs.lat, obs.lon, obs.ele, \
                    station_id=obs.station_id)


            # Solve trajectory
            traj = traj.run()


            # Calculate orbit of an intersecting planes solution
            if traj_solver == 'planes':

                # Use the average velocity of the first part of the trajectory for the initial velocity
                time_vel = []
                for obs in traj.observations:
                    for t, v in zip(obs.time_data, obs.velocities):
                        time_vel.append([t, v])

                time_vel = np.array(time_vel)

                # Sort by time
                time_vel = time_vel[time_vel[:, 0].argsort()]

                # Calculate the velocity of the first half of the trajectory
                v_init_fh = np.mean(time_vel[int(len(time_vel)/2), 1])

                # Calculate the orbit
                traj.orbit = calcOrbit(traj.avg_radiant, v_init_fh, traj.v_avg, traj.state_vect_avg, \
                traj.jd_avg, stations_fixed=True, referent_init=False)


            # Save info about the simulation comparison
            sim_met.saveTrajectoryComparison(traj, traj_solver)


            # Dump measurements to a MATLAB-style file
            if traj_solver == 'monte_carlo':
                traj.dumpMeasurements(output_dir, str(sim_met.jdt_ref) + '_meas_dump.txt')


