""" Given the parameters that describe a meteor shower, this code generates the shower meteors.
"""

from __future__ import print_function, division, absolute_import

import sys
import os
import datetime
import random

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

from wmpl.Config import config
from wmpl.TrajSim.TrajSim import geocentricRadiantToApparent
from wmpl.Trajectory.Trajectory import ObservedPoints, Trajectory
from wmpl.Trajectory.GuralTrajectory import GuralTrajectory
from wmpl.Trajectory.Orbit import calcOrbit
from wmpl.Utils.TrajConversions import J2000_JD, G, EARTH, altAz2RADec, raDec2ECI, rotatePolar, jd2Date, datetime2JD, \
    jd2DynamicalTimeJD, jd2LST, cartesian2Geo, geo2Cartesian, eci2RaDec, raDec2AltAz, \
    equatorialCoordPrecession
from wmpl.Utils.Ephem import astronomicalNight
from wmpl.Utils.SolarLongitude import solLon2jdJPL
from wmpl.Utils.PlotMap import GroundMap
from wmpl.Utils.PlotCelestial import CelestialPlot
from wmpl.Utils.Math import meanAngle, pointInsideConvexHull, samplePointsFromHull, vectMag, vectNorm, \
    sphericalToCartesian, cartesianToSpherical, angleBetweenSphericalCoords
from wmpl.Utils.OSTools import mkdirP
from wmpl.Utils.Pickling import savePickle
from wmpl.Utils.Plotting import savePlot


# Try importing Campbell-Brown & Koschny (2004) ablation code 
try:
    from MetSim.MetSim import loadInputs, runSimulation
    METSIM_IMPORT = True
except:
    METSIM_IMPORT = False



class SimStation(object):
    def __init__(self, lat, lon, elev, station_id, fps, t_offset, obs_ang_std, azim_centre, elev_centre, \
        fov_wid, fov_ht, lim_mag=None, P_0m=None, min_ang_vel=None):
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
            min_ang_vel: [float] Minimum angular velocity (deg/s) of a meteor that will be properly detected.

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

        # Minimum angular velocity of a meteor that will be detected
        self.min_ang_vel = min_ang_vel



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
    def __init__(self, ra_g, dec_g, v_g, year, month, sol, jdt_ref, beg_height, state_vect, \
        obs_ang_uncertanties, t_offsets):
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
            obs_ang_uncertanties: [list of floats] A list of observational unceretanties from every station 
                (in arcsecs).
            t_offsets: [list] List of maximum time offsets of stations from real time (seconds).
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

        # reference Julian date which corresponds to the solar longitude
        self.jdt_ref = jdt_ref

        # Initial state vector in ECI coordinates
        self.state_vect = state_vect

        # Velocity model
        self.velocity_model = None

        # Station observational uncertainties
        self.obs_ang_uncertanties = obs_ang_uncertanties

        # Max. time offsets per every station
        self.t_offsets = t_offsets

        # ECI coordinates of the beginning point
        self.begin_eci = None

        # Coordinates of the beginning point on the trajectory
        self.rbeg_lat, self.rbeg_lon, self.rbeg_ele = None, None, None

        ######################################################################################################

        # Calculate geographic coordinates of the state vector
        self.state_vect_lat, self.state_vect_lon, self.state_vect_ele = cartesian2Geo(jdt_ref, *state_vect)

        # Calculate apparent radiant and the orbit
        self.ra, self.dec, self.v_init, self.orbit = geocentricRadiantToApparent(ra_g, dec_g, v_g, \
            state_vect, jdt_ref)


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



    def savePickle(self, output_dir):
        """ Save the MetSim object in a .pickle file. """

        file_name_pickle = str(self.jdt_ref) + "_sim_met.pickle"

        # Save the SimMeteor object as a pickle file
        savePickle(self, output_dir, file_name_pickle)


    def initOutput(self, output_dir):
        """ Prepare everything for saving solver results. """

        mkdirP(output_dir)

        # File name of the report file
        file_name = str(self.jdt_ref) + "_sim_met_info.txt"

        self.file_path = os.path.join(output_dir, file_name)



    def saveInfo(self, output_dir):
        """ Save information about the simulated meteor. """


        self.initOutput(self, output_dir)

        out_str =  ""
        out_str += "reference JD: {:20.12f}".format(self.jdt_ref) + "\n"
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
        out_str += "State vector (geo):\n"
        out_str += " Lon = {:>12.6f} deg\n".format(np.degrees(self.state_vect_lon))
        out_str += " Lat = {:>12.6f} deg\n".format(np.degrees(self.state_vect_lat))
        out_str += " Ht  = {:>8.2f} m\n".format(self.state_vect_ele)

        out_str +=  "\n"

        out_str += "Radiant (apparent, epoch of date):\n"
        out_str += "  R.A.   = {:>9.5f} deg\n".format(np.degrees(self.ra))
        out_str += "  Dec    = {:>+9.5f} deg\n".format(np.degrees(self.dec))
        out_str += "  Vinit  = {:>9.5f} km/s\n".format(self.v_init/1000)

        if self.v_begin is not None:
            out_str += "  Vbegin = {:>9.5f} km/s\n".format(self.v_begin/1000)


        # Calculate the geocentric radiant in the epoch of date
        ra_g_eod, dec_g_eod = equatorialCoordPrecession(J2000_JD.days, self.jdt_ref, self.ra_g, self.dec_g)
        out_str += "Radiant (geocentric, epoch of date):\n"
        out_str += "  R.A.   = {:>9.5f} deg\n".format(np.degrees(ra_g_eod))
        out_str += "  Dec    = {:>+9.5f} deg\n".format(np.degrees(dec_g_eod))


        x_beg, y_beg, z_beg = self.begin_eci

        out_str += "Begin point:\n"
        out_str += " X =  {:11.2f} m\n".format(x_beg)
        out_str += " Y =  {:11.2f} m\n".format(y_beg)
        out_str += " Z =  {:11.2f} m\n".format(z_beg)
        out_str += " Lon = {:>12.6f} deg\n".format(np.degrees(self.rbeg_lon))
        out_str += " Lat = {:>12.6f} deg\n".format(np.degrees(self.rbeg_lat))
        out_str += " Ht  = {:>8.2f} m\n".format(self.rbeg_ele)

        out_str += "\n"

        out_str += "Orbit:\n"
        out_str += "----------------------------------\n"
        out_str += self.orbit.__repr__()

        out_str += "----------------------------------\n"
        out_str += "\n"

        # out_str += "Radiant (geocentric, J2000):\n"
        # out_str += "  R.A.   = {:>9.5f} deg\n".format(np.degrees(self.ra_g))
        # out_str += "  Dec    = {:>+9.5f} deg\n".format(np.degrees(self.dec_g))
        # out_str += "  Vg     = {:>9.5f} km/s\n".format(self.v_g/1000)
        # #out_str +=  "--------------------\n"
        # out_str += "  La Sun = {:>10.6f} deg\n".format(np.degrees(self.sol))


        if self.velocity_model is not None:

            out_str += "Velocity model:\n"
            out_str += self.velocity_model.__repr__()
            out_str += "\n"


        # Save station data

        out_str += 'Stations\n'
        out_str += "--------\n"

        out_str += "        ID, Lon +E (deg), Lat +N (deg), Ele (m), Time offset (s), Obs. uncertanty (arcsec)\n"
        for obs, t_off, obs_ang_std in zip(self.observations, self.t_offsets, self.obs_ang_uncertanties):

            station_info = []
            station_info.append("{:>10s}".format(str(obs.station_id)))
            station_info.append("{:>12.6f}".format(np.degrees(obs.lon)))
            station_info.append("{:>12.6f}".format(np.degrees(obs.lat)))
            station_info.append("{:>7.2f}".format(obs.ele))
            station_info.append("{:>15.6f}".format(t_off))
            station_info.append("{:>24.2f}".format(obs_ang_std))
            
            out_str += ", ".join(station_info) + "\n"


        print(out_str)

        # Save the report to a file
        with open(self.file_path, 'w') as f:
            f.write(out_str)


        # Erase spline fits in the ablation model, as they cannot be pickled
        if self.velocity_model.name == 'ablation':
            self.velocity_model.velocity_model = None
            self.velocity_model.length_model = None
            self.velocity_model.luminosity_model = None

        
        # Save the MetSim object as a pickle file
        self.savePickle(output_dir)




    def saveTrajectoryComparison(self, traj, traj_method, note=''):
        """ Saves the comparison of trajectory results, between the original values and the estimated values.
        
        Arguments:
            traj: [Trajectory]
            traj_method: [str] Method of trajectory estimation.
            note: [str] Extra note to write in the file.

        """

        # Add the info to the simulation file, if it exists
        if os.path.isfile(self.file_path):

            out_str = '\n'
            out_str += '-----------------------------\n'
            out_str += 'Trajectory estimation method: {:s} {:s}\n' .format(traj_method, note)

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

            print(out_str)


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



class LinearDeceleration(object):
    def __init__(self, duration, t0, decel):
        """ Linear deceleration model for generating meteor trajectories. 
        
        Arguments:
            duration: [float] Duration of the meteor in seconds. 
            t0: [float] Beginning time of linear deceleration (seconds).
            decel: [float] Deceleration (m/s^2).
        """

        self.name = 'linear'

        self.duration = duration
        self.t0 = t0
        self.decel = decel



    def getTimeData(self, fps):
        """ Returns an array of time data for the meteor. 
        
        Arguments:
            fps: [float] Frames per second of the camera.

        """

        return np.arange(0, self.duration, 1.0/fps)



    def getLength(self, v_init, t):
        """ Calculates a length along the track at the given time with the given initial velocity and
            constant deceleration. 

        Arguments:
            v_init: [float] Velocity at t = 0. In m/s.
            t: [float] Time at which the length along the track will be evaluated.

        """

        if t < self.t0:
            return v_init*t

        else:
            return v_init*t - ((t - self.t0)**2)*self.decel/2.0



    def __repr__(self):
        """ Returned upon printing the object. """

        out_str = "Linear deceleration, duration: {:.4f}, t0: {:.4f}, decel: {:.2f}".format(self.duration, 
            self.t0, self.decel) + "\n"

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
    def __init__(self, mass_min, mass_max, mass_index, density, ablation_coeff, Gamma, Lambda, lum_eff):
        """ Velocity calculated from numerical meteor ablation simulation by method of Campbell-Brown & 
            Koschny (2004).
        
        Arguments:
            mass_min: [float] Logarithm of minimum mass in kilograms (e.g. -3 for 1 gram).
            mass_max: [float] Logarithm of maximum mass in kilograms (e.g. 0 for 1 kilogram).
            mass_index: [float] Shower mass index.
            density: [float] Meteoroid density (kg/m^3).
            ablation_coeff: [float] Ablation coefficient (s^2/km^2)
            Gamma: [float] Drag coefficient.
            Lambda: [float] Heat transfer coefficient.
            lum_eff: [float] Luminous efficiency (fraction).

        """

        self.name = 'ablation'

        self.mass_min = mass_min
        self.mass_max = mass_max
        self.mass_index = mass_index
        self.density = density
        self.Gamma = Gamma
        self.Lambda = Lambda
        self.ablation_coeff = ablation_coeff
        self.lum_eff = lum_eff

        self.mass = None
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

        # Generate a mass upon the first run
        if self.mass is None:
            self.mass = sampleMass(self.mass_min, self.mass_max, self.mass_index, 1)[0]


        self.met.m_init = self.mass

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
        closest_index = np.argwhere(heights_diffs == np.min(heights_diffs))

        # Check if any heights matched
        if len(closest_index) == 0:
            return False

        closest_index = closest_index[0][0]

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
    fps_list=None, obs_ang_uncertainties=None, lim_magnitudes=None, P_0m_list=None, min_ang_velocities=None):
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
        min_ang_velocities: [list] Minimum angular velocity of the meteor as seen by a given station. 
            The default is 0.0 deg/s, meaning all meteor will be considered detected, regardless of their
            angular velocity.

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

    if min_ang_velocities is None:
        min_ang_velocities = [0.0]*len(stations_geo)



    station_list = []
    
    # Generate SimStation objects
    for t_offset, stat_geo, fps, obs_ang_std, azim_centre, elev_centre, fov_wid, fov_ht, lim_mag, P_0m, \
        min_ang_vel in zip(t_offsets, stations_geo, fps_list, obs_ang_uncertainties, azim_fovs, elev_fovs, \
            fov_widths, fov_heights, lim_magnitudes, P_0m_list, min_ang_velocities):

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
            elev_centre, fov_wid, fov_ht, lim_mag=lim_mag, P_0m=P_0m, min_ang_vel=min_ang_vel)


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

    # from mpl_toolkits.basemap import Basemap


    # Calculate the mean latitude and longitude of all points to be plotted
    lat_list = [sim_met.state_vect_lat for sim_met in sim_meteor_list] + [stat.lat for stat in station_list]
    lon_list = [sim_met.state_vect_lon for sim_met in sim_meteor_list] + [stat.lon for stat in station_list]


    m = GroundMap(lat_list, lon_list, border_size=50, color_scheme='light')


    # Plot locations of all stations and measured positions of the meteor
    for stat in station_list:

        # Plot stations
        m.scatter(stat.lat, stat.lon, s=10, label=str(stat.station_id), marker='x')


    for sim_met in sim_meteor_list:
        
        # Plot a point marking the position of initial state vectors of all meteors
        m.scatter(sim_met.state_vect_lat, sim_met.state_vect_lon, c='g', marker='+', s=50, alpha=0.75)

        # Plot simulated tracks
        if sim_met.model_lat:

            for model_lat, model_lon in zip(sim_met.model_lat, sim_met.model_lon):
                m.plot(model_lat, model_lon, c='r')


    plt.legend(loc='upper right')


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



def sampleDensityMoorhead(log_rho_mean, log_rho_sigma, n_samples):
    """ Sample a meteor shower density distribution using the given mean density and stddev. Used for ablation
        model meteoroids. The density distribution is samples from a Gaussian distribution on logarithmic 
        values of densities. For more details see: Moorhead, Althea V., et al. "A two-population sporadic 
        meteoroid bulk density distribution and its implications for environment models." MNRAS 472.4 (2017): 
        3833-3841.

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



def generateStateVector(station_list, jd, beg_height, beg_height_min, beg_height_max):
    """ Generate the initial state vector in ECI coordinates for the given meteor. 
    
    Arguments:
        station_list: [list] A list of SimStation objects containing station info.
        jd: [float] Julian date for calculating ECI coordinates of FOVs. This date can be arbitrary.
        beg_height: [float] Beginning height (meters).
        beg_height_min: [float] Bottom height of the FOV polyhedron (meters).
        beg_height_max: [float] Upper height of the FOV polyhedron (meters).

    """


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
        
        # Sample one point from the FOV box of a random station
        stat_rand = random.choice(station_list)

        # Get the FOV box of the random station at the given station
        fov_box_rand = stat_rand.fovCornersToECI(jd, beg_height_min, beg_height_max)

        state_vect = samplePointsFromHull(fov_box_rand, 1)[0]

        # Project the point on the given beginning height
        pt_lat, pt_lon, pt_ht = cartesian2Geo(jd, *state_vect)
        state_vect = geo2Cartesian(pt_lat, pt_lon, 1000*beg_height, jd)

        fov_count = 0

        # Check if the point is inside FOV boxes of at least one more station
        for fov_box_stat in fov_boxes_all:

            # # Get the FOV box of the given station
            # fov_box_stat = stat.fovCornersToECI(jd, 1000*min(beg_height_data), \
            #     beg_height_max)

            # Check if the point is inside the FOV box of the given station
            inside_test = pointInsideConvexHull(fov_box_stat, state_vect)

            if inside_test:
                fov_count += 1

            # # If the point is not inside one of the boxes, find another point
            # if inside_test == False:
            #     break

        # # If the point is not inside one of the boxes, find another point
        # if inside_test == False:
        #     continue

        # If the point is not inside the FOV of another station, skip it
        if fov_count < 2:
            continue

        ### TEST ###
        # plotStationFovs(station_list, jd, 1000*beg_height, [state_vect])
        ###


        # If the point is inside all boxes, stop the search
        break


    return state_vect



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

    # Reset the mass if using the ablation modelling (prevent for getting stuck on small masses that produce
    #   too faint meteors)
    if sim_met.velocity_model.name == 'ablation':
        sim_met.velocity_model.mass = None

    # Go through every station
    for stat in station_list:

        # If the velocity model is given by the ablation model, run the model first
        if sim_met.velocity_model.name == 'ablation':
            sim_met.velocity_model.getSimulation(sim_met.v_init, sim_met.orbit.zc, sim_met.state_vect_ele)

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

        length_max = 0


        # Vertical component of the initial velocity (has to be negative as the meteor is going down)
        v0z = -sim_met.v_init*np.cos(sim_met.orbit.zc)

        # Go through all trajectory points
        for t in time_data_model:

            # Calculate the Julian date of the trajectory point
            jd = sim_met.jdt_ref + t/86400.0

            # Calculate the length along the trail using the given velocity model
            length = velocity_model.getLength(sim_met.v_init, t)

            
            # Make sure the meteor does not start going backwards
            if length < length_max:
                break

            length_max = length



            # Calculate the meteor position at every point in time
            traj_eci = sim_met.state_vect + length*(-sim_met.radiant_eci)



            ### Apply gravity drop to calculated ECI coordinates

            # Choose the appropriate initial height of the meteor
            if first_point:
                r0 = vectMag(traj_eci)

            else:
                r0 = vectMag(sim_met.begin_eci)

            
            traj_eci = applyGravityDrop(traj_eci, t, r0, v0z)

            ###


            # If the model is given by the ablation code, check that the magnitude of the meteor was above the
            # detection threshold
            if sim_met.velocity_model.name == 'ablation':

                # Calculate the absolute magnitude (magnitude @100km) at this point in time
                abs_mag = -2.5*np.log10(sim_met.velocity_model.luminosity_model(t)/stat.P_0m)

                # Calculate the range to the station
                stat_range = vectMag(geo2Cartesian(stat.lat, stat.lon, stat.elev, jd) - traj_eci)

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
            fov_corners = stat.fovCornersToECI(jd, 2*sim_met.state_vect_ele)

            # Calculate the ECI position of the station at the particular point in time
            stat_eci = geo2Cartesian(stat.lat, stat.lon, stat.elev, jd)

            # Add the ECI position of the station to the vertices list
            fov_corners.append(stat_eci)

            # Skip the estimation if the trajectory has NaNs
            if np.any(np.isnan(traj_eci)):
                print('Trajectory ECI coordinates is NaN, skipping...')
                continue

            # If the point is not inside the FOV, skip it
            if not pointInsideConvexHull(fov_corners, traj_eci):
                #print('Point not inside the hull!', traj_eci)
                continue

            ##################################################################################################


            ### Project modelled points to the perspective of the observer at the given station ###
            ##################################################################################################

            # Calculate the unit direction vector pointing from the station to the trajectory point
            rhat = vectNorm(traj_eci - stat_eci)


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
            sigma = stat.obs_ang_std/np.sqrt(2.0)

            model_eci = np.zeros(3)

            model_eci[0] = rhat[0] + np.random.normal(0, sigma)*uhat[0] + np.random.normal(0, sigma)*vhat[0]
            model_eci[1] = rhat[1] + np.random.normal(0, sigma)*uhat[1] + np.random.normal(0, sigma)*vhat[1]
            model_eci[2] = rhat[2] + np.random.normal(0, sigma)*uhat[2] + np.random.normal(0, sigma)*vhat[2]

            # Normalize to a unit vector
            model_eci = vectNorm(model_eci)

            ###



            # Calculate RA, Dec for the given point
            ra, dec = eci2RaDec(model_eci)

            # Calculate azimuth and altitude of this direction vector
            azim, elev = raDec2AltAz(ra, dec, jd, stat.lat, stat.lon)

            ##################################################################################################


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

                # ECI coordinates of the beginning point
                sim_met.begin_eci = traj_eci

                # Calculate coordinates of the beginning point on the trajectory
                sim_met.rbeg_lat, sim_met.rbeg_lon, sim_met.rbeg_ele = cartesian2Geo(jd, *traj_eci)

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


        # Calculate the average angular velocity in rad/s
        ang_vel = angleBetweenSphericalCoords(azim_data[0], elev_data[0], azim_data[-1], \
            elev_data[-1])/(np.max(time_data) - np.min(time_data))

        # Skip observations slower than the minimum angular velocity
        if np.degrees(ang_vel) < stat.min_ang_vel:
            print('Angular velocity of meteor {:.6f} from station {:s} too slow: {:.2f}'.format(sim_met.jdt_ref, stat.station_id, np.degrees(ang_vel)))
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





def simulateMeteorShower(station_list, meteor_velocity_models, n_meteors, ra_g, ra_g_sigma, dec_g, 
    dec_g_sigma, d_ra, d_dec, v_g, v_g_sigma, d_vg, year, month, sol_max, sol_slope, beg_height, 
    beg_height_sigma, nighttime_meteors_only=True, output_dir='.', save_plots=True, orbit_limits=None):
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
        d_ra: [float] R.A. radiant drift (degrees of R.A. per degree of solar longitude).
        d_dec: [float] Dec radiant drift (degrees of declination per degree of solar longitude).
        v_g: [float] Mean geocentric velocity (km/s).
        v_g_sigma: [float] Standard deviation of the geocentric velocity (km/s).
        d_vg: [float] Vg drift in km/s per degree of solar longitude.
        year: [int] Year of the meteor shower.
        month: [int] Month of the meteor shower.
        sol_max: [float] Solar longitude of the maximum shower activity (degrees).
        sol_slope: [float] Slope of the activity profile.
        beg_height: [float] Mean of Gaussian beginning height profile (km).
        beg_height_sigma: [float] Standard deviation of beginning height profile (km).


    Keyword arguments:
        nighttime_meteors_only: [bool] If True, only meteors seen during the night will be taken. True by
            default.
        output_dir: [str] Directory where the plots will be saved.
        save_plot: [str] Save plots if True.
        orbit_limits: [list] A list of limits per orbital element. None by defualt. The syntax is the 
            following: ['param_name', param_min, param_max], the fist element is the name of the orbital 
            parameter (it has to be a variable name from the Orbit class), the second is the minimum value 
            of the parameter, and the third element is the maxium value. If angles are being limited, they 
            should be in radians.
            Example: if we want to limit the semimajor axis and the inclination, then
                orbit_limits = [['a', 2.2, 2.23], ['incl', np.radians(2), np.radians(10)]]

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

    

    ### GENERATE METEORS ###
    ##########################################################################################################

    sol_data = []
    jd_data = []

    ra_g_list = []
    dec_g_list = []
    v_g_list = []

    beg_height_list = []
    state_vector_list = []

    sim_meteor_list = []

    # Get a list of uncertanties per every station and the range of timing offsets
    obs_ang_uncertanties = [stat.obs_ang_std for stat in station_list]
    t_offsets = [stat.t_offset for stat in station_list]


    # Make sure that the range of beginning heights is at least 10km (determine that as the -2 and +2 sigma 
    #   difference)
    if 4*beg_height_sigma < 10:

        beg_height_min = 1000*(beg_height - 5)
        beg_height_max = 1000*(beg_height + 5)

    else:
        beg_height_min = 1000*(beg_height - 2*beg_height_sigma)
        beg_height_max = 1000*(beg_height + 2*beg_height_sigma)


    # Check if the station FOVs are overlapping at all at given heights
    if not stationFovOverlap(station_list, solLon2jdJPL(year, month, sol_max), beg_height_min, beg_height_max):
        
        print('ERROR! FOVs of stations are not overlapping!')
        sys.exit()



    meteor_no = 0

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
        if (not nighttime_meteors_only) or ((datetime2JD(night_start) < meteor_jd) \
            and (datetime2JD(night_end) > meteor_jd)):

            print('{:d}/{:d} meteor at JD = {:.6f}, LaSun = {:.6f} deg'.format(meteor_no + 1, n_meteors, \
                meteor_jd, np.degrees(sol)))

        else:
            print('The generated meteor occured during daytime, skipping it!')

            # print('Night start     :', night_start)
            # print('Meteor candidate:', jd2Date(meteor_jd, dt_obj=True))
            # print('Night end       :', night_end)

            # print('-----------------')

            continue


        ### GENERATE GEOCENTRIC RADIANTS ###
        ######################################################################################################

        # Sample radiant positions from a von Mises distribution centred at (0, 0)
        ra = np.random.vonmises(0, 1.0/(ra_g_sigma**2), 1)
        dec = np.random.vonmises(0, 1.0/(dec_g_sigma**2), 1)

        # Rotate R.A., Dec from (0, 0) to generated coordinates, to account for spherical nature of the angles
        # After rotation, (RA, Dec) will still be scattered around (0, 0)
        ra_rot, dec_rot = rotatePolar(0, 0, ra, dec)

        # Rotate all angles scattered around (0, 0) to the given coordinates of the centre of the distribution
        ra_rot, dec_rot = rotatePolar(ra_rot, dec_rot, ra_g, dec_g)


        # Apply the radiant drift
        ra_g_final = np.radians(np.degrees(ra_rot) + d_ra*np.degrees(sol - sol_max))
        dec_g_final = np.radians(np.degrees(dec_rot) + d_dec*np.degrees(sol - sol_max))

        # Generate geocentric velocities from a Gaussian distribution
        v_g_final = np.random.normal(v_g, v_g_sigma, size=1)[0]

        # Apply the velocity drift
        v_g_final = v_g_final + 1000*d_vg*np.degrees(sol - sol_max)


        # Draw beginning heights from a Gaussian distribution
        beg_height_final = np.random.normal(beg_height, beg_height_sigma, size=1)[0]


        # Generate initial state vectors for drawn shower meteors inside the FOVs of given stations
        state_vect = generateStateVector(station_list, meteor_jd, beg_height_final, beg_height_min, \
            beg_height_max)


        # Init the SimMeter object
        sim_meteor = SimMeteor(ra_g_final, dec_g_final, v_g_final, year, month, sol, meteor_jd, \
            beg_height_final, state_vect, obs_ang_uncertanties, t_offsets)


        # Check if the generated orbit is within the set limits
        if orbit_limits is not None:

            # Check if there is only one limit, of if there are more
            if not all(isinstance(elem, list) for elem in orbit_limits):
                orbit_limits = [orbit_limits]

            skip_meteor = False

            print('Orbital limits:')

            # Check every limit
            for entry in orbit_limits:
                arg_name, arg_min, arg_max = entry

                # Get the value of the given parameter
                arg_value = sim_meteor.orbit.__getattribute__(arg_name)

                # The parameter if it is not within the given range
                if (not (arg_value >= arg_min)) or (not (arg_value <= arg_max)):

                    print('Skipping meteor, the orbit parameter {:s} is outside bounds: {:.3f} <= {:.3f} <= {:.3f}'.format(arg_name, arg_min, arg_value, arg_max))

                    skip_meteor = True
                    break

                else:
                    print("{:s} = {:.3f} <= {:.3f} <= {:.3f}".format(arg_name, arg_min, arg_value, arg_max))


            if skip_meteor:
                continue



        # Generate trajectory data for the given meteor
        sim_meteor = generateTrajectoryData(station_list, sim_meteor, meteor_velocity_models[meteor_no])


        # Check that there are at least 2 sets of measurements
        if len(sim_meteor.observations) < 2:
            print('Skipped meteor at JD =', sim_meteor.jdt_ref, 'as it was not observable from at least 2 stations!')
            continue

        # Make sure there are at least 4 point from every station
        meas_counts = [len(obs.time_data) for obs in sim_meteor.observations]
        if meas_counts:
            if np.min(meas_counts) < 4:

                print('Skipped meteor at JD =', sim_meteor.jdt_ref, 'due to having less than 4 point from any of the stations!')
                continue

        else:
            print('Skipped meteor at JD =', sim_meteor.jdt_ref, 'due to having less than 4 point from any of the stations!')
            continue



        # Make sure the angle between the stations is at least 1 degree
        max_angle = 0
        for i, obs1 in enumerate(sim_meteor.observations):
            for j, obs2 in enumerate(sim_meteor.observations):

                # Skip same and already paired stations
                if j <= i:
                    continue

                # Calculate ECI coordinates of stations
                stat1_eci = np.array(geo2Cartesian(obs1.lat, obs1.lon, obs1.ele, sim_meteor.jdt_ref))
                stat2_eci = np.array(geo2Cartesian(obs2.lat, obs2.lon, obs2.ele, sim_meteor.jdt_ref))

                # Calculate vectors pointing from stations to the state vector
                r1 = vectNorm(stat1_eci - np.array(sim_meteor.state_vect))
                r2 = vectNorm(stat2_eci - np.array(sim_meteor.state_vect))

                # Calculate the angle between the stations from the state vector
                stat_angle = np.arccos(np.dot(r1, r2))

                max_angle = max([max_angle, stat_angle])


        # Skip the simulation if the angle is less than 1 degree
        if np.degrees(max_angle) < 1.0:
            print('Skipped meteor at JD =', sim_meteor.jdt_ref, 'due to a convergence angle of less than 1 degree!')
            continue


        ##############################################################

        # Add the solar longitude to the final list
        sol_data.append(sol)
        jd_data.append(meteor_jd)

        # Add the geocentric radiant to the list
        ra_g_list.append(ra_g_final)
        dec_g_list.append(dec_g_final)
        v_g_list.append(v_g_final)

        beg_height_list.append(beg_height_final)

        # Put the found state vector in the list
        state_vector_list.append(state_vect)

        sim_meteor_list.append(sim_meteor)

        # Check if there are enough meteors
        if meteor_no == n_meteors - 1:
            break

        meteor_no += 1


    sol_data = np.array(sol_data)
    jd_data = np.array(jd_data)

    ra_g_data = np.array(ra_g_list)
    dec_g_data = np.array(dec_g_list)
    v_g_data = np.array(v_g_list)
    beg_height_data = np.array(beg_height_list)

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


    ##########################################################################################################


    plt.hist(v_g_data/1000)
    plt.xlabel('Geocentric velocity (km/s)')
    plt.ylabel('Counts')
    plt.title('Geocentric velocities')

    if save_plots:
        savePlot(plt, 'vg.png', output_dir=output_dir)

    plt.show()

    plt.hist(beg_height_data, orientation='horizontal')
    plt.xlabel('Counts')
    plt.ylabel('Beginning height (km)')
    plt.title('Beginning heights')

    if save_plots:
        savePlot(plt, 'beginning_heights.png', output_dir=output_dir)

    plt.show()


    # Plot radiants on the sky
    m = CelestialPlot(ra_g_data, dec_g_data, projection='stere', bgcolor='w')

    m.scatter(ra_g_data, dec_g_data, c=v_g_data/1000, s=2)

    m.colorbar(label='$V_g$ (km/s)')

    if save_plots:
        savePlot(plt, 'geo_radiants.png', output_dir=output_dir)

    plt.show()


    ##########################################################################################################


    # Prepare the solar longitudes for plotting
    sol_data_plt = np.degrees(sol_data)
    
    if (np.max(sol_data_plt) - np.min(sol_data_plt)) > 180:
        sol_data_plt = sol_data_plt[sol_data_plt < 180] + 360

    sol_plt_arr = np.linspace(np.min(sol_data_plt), np.max(sol_data_plt), 100)


    # Plot RA vs. solar longitude
    plt.scatter(sol_data_plt, np.degrees(ra_g_data))

    # Plot the radiant drift in RA
    plt.plot(sol_plt_arr, np.degrees(ra_g) + d_ra*(sol_plt_arr - np.degrees(sol_max)))

    plt.xlabel('Solar longitude (deg)')
    plt.ylabel('Right ascension (deg)')

    if save_plots:
        savePlot(plt, 'ra_g_drift.png', output_dir=output_dir)

    plt.show()



    # Plot Dec vs. solar longitude
    plt.scatter(sol_data_plt, np.degrees(dec_g_data))

    # Plot the radiant drift in Dec
    plt.plot(sol_plt_arr, np.degrees(dec_g) + d_dec*(sol_plt_arr - np.degrees(sol_max)))

    plt.xlabel('Solar longitude (deg)')
    plt.ylabel('Declination (deg)')

    if save_plots:
        savePlot(plt, 'dec_g_drift.png', output_dir=output_dir)

    plt.show()


    # Plot Vg vs. solar longitude
    plt.scatter(sol_data_plt, v_g_data)

    # Plot the drift in Vg
    plt.plot(sol_plt_arr, v_g + 1000*d_vg*(sol_plt_arr - np.degrees(sol_max)))

    plt.xlabel('Solar longitude (deg)')
    plt.ylabel('Geocentric velocity (km/s)')

    if save_plots:
        savePlot(plt, 'v_g_drift.png', output_dir=output_dir)

    plt.show()
    
    
    print('Beginning height range:', beg_height_min/1000, beg_height_max/1000)


    return sim_meteor_list





if __name__ == "__main__":

    # Directory where the files will be saved
    dir_path = os.path.abspath('../SimulatedMeteors')


    # ### EMCCD STATION PARAMETERS ###
    # ##########################################################################################################

    # system_name = 'EMCCD'

    # # Number of stations in total
    # n_stations = 4

    # # Maximum time offset (seconds)
    # t_max_offset = 1

    # # Geographical coordinates of stations (lat, lon, elev, station_id) in degrees and meters
    # stations_geo = [
    #     [43.26420, -80.77209, 329.0, '1F'],
    #     [43.26420, -80.77209, 329.0, '1G'],
    #     [43.19279, -81.31565, 324.0, '2F'],
    #     [43.19279, -81.31565, 324.0, '2G']]

    # # Camera FPS per station
    # fps_list = [16.67, 16.67, 16.67, 16.67]

    # # Observation uncertanties per station (arcsec)
    # obs_ang_uncertainties = [10, 10, 10, 10]

    # # Azimuths of centre of FOVs (degrees)
    # azim_fovs = [301.66, 320.0, 358.54, 18.89]

    # # Elevations of centre of FOVs (degrees)
    # elev_fovs = [64.35, 61.89, 73.5, 65.54]

    # # Cameras FOV widths (degrees)
    # fov_widths = [14.75, 14.75, 14.75, 14.75]

    # # Cameras FOV heights (degrees)
    # fov_heights = [14.75, 14.75, 14.75, 14.75]

    # # Limiting magnitudes (needed only for ablation simulation)
    # lim_magnitudes = [+7.5, +7.5, +7.5, +7.5]

    # # Powers of zero-magnitude meteors (Watts) (needed only for ablation simulation)
    # P_0m_list = [840, 840, 840, 840]

    # # Minimum angular velocity for detection (deg/s)
    # min_ang_velocities = [0.5, 0.5, 0.5, 0.5]

    # ##########################################################################################################

    # ### PERFECT STATIONS (CAMO-based) PARAMETERS ###
    # ##########################################################################################################

    # system_name = 'Perfect_CAMO'

    # # Number of stations in total
    # n_stations = 2

    # # Maximum time offset (seconds)
    # t_max_offset = 1

    # # Geographical coordinates of stations (lat, lon, elev, station_id) in degrees and meters
    # stations_geo = [
    #     [43.26420, -80.77209, 329.0, 'tavis'], # Tavis
    #     [43.19279, -81.31565, 324.0, 'elgin']] # Elgin

    # # Camera FPS per station
    # fps_list = [100, 100]

    # # Observation uncertanties per station (arcsec)
    # obs_ang_uncertainties = [0.001, 0.001]

    # # Azimuths of centre of FOVs (degrees)
    # azim_fovs = [326.823, 1.891]

    # # Elevations of centre of FOVs (degrees)
    # elev_fovs = [41.104, 46.344]

    # # Cameras FOV widths (degrees)
    # fov_widths = [19.22, 19.22]

    # # Cameras FOV heights (degrees)
    # fov_heights = [25.77, 25.77]

    # # Limiting magnitudes (needed only for ablation simulation)
    # lim_magnitudes = [+5.5, +5.5]

    # # Powers of zero-magnitude meteors (Watts) (needed only for ablation simulation)
    # P_0m_list = [840, 840]

    # # Minimum angular velocity for detection (deg/s)
    # min_ang_velocities = [3.0, 3.0]

    # ##########################################################################################################


    # ### CAMO STATION PARAMETERS ###
    # ##########################################################################################################

    # system_name = 'CAMO'

    # # Number of stations in total
    # n_stations = 2

    # # Maximum time offset (seconds)
    # t_max_offset = 1

    # # Geographical coordinates of stations (lat, lon, elev, station_id) in degrees and meters
    # stations_geo = [
    #     [43.26420, -80.77209, 329.0, 'tavis'], # Tavis
    #     [43.19279, -81.31565, 324.0, 'elgin']] # Elgin

    # # Camera FPS per station
    # fps_list = [100, 100]

    # # Observation uncertanties per station (arcsec)
    # obs_ang_uncertainties = [1, 1]

    # # Azimuths of centre of FOVs (degrees)
    # azim_fovs = [326.823, 1.891]

    # # Elevations of centre of FOVs (degrees)
    # elev_fovs = [41.104, 46.344]

    # # Cameras FOV widths (degrees)
    # fov_widths = [30.0, 30.0]

    # # Cameras FOV heights (degrees)
    # fov_heights = [30.0, 30.0]

    # # Limiting magnitudes (needed only for ablation simulation)
    # lim_magnitudes = [+5.5, +5.5]

    # # Powers of zero-magnitude meteors (Watts) (needed only for ablation simulation)
    # P_0m_list = [840, 840]

    # # Minimum angular velocity for detection (deg/s)
    # min_ang_velocities = [1.0, 1.0]

    # ##########################################################################################################

    # ### CABARNET STATION PARAMETERS ###
    # ##########################################################################################################

    # system_name = 'CABERNET'

    # # Number of stations in total
    # n_stations = 3

    # # Maximum time offset (seconds)
    # t_max_offset = 1.0

    # # Geographical coordinates of stations (lat, lon, elev, station_id) in degrees and meters
    # stations_geo = [
    #     [42.936389, 0.142778, 2877.0, 'PicDuMidi'], #PicDuMidi
    #     [42.787761, 1.300037, 1600.0, 'Guzet'], #Guzet
    #     [42.051639, 0.729586, 1569.0, 'Montsec']] #Montsec

    # # Camera FPS per station
    # fps_list = [95, 95, 95]

    # # Observation uncertainties per station (arcsec)
    # obs_ang_uncertainties = [3.24, 3.24, 3.24]

    # # Azimuths of centre of FOVs (degrees)
    # azim_fovs = [126.51, 251.94, 357.67]

    # # Elevations of centre of FOVs (degrees)
    # elev_fovs = [62.67, 62.81, 64.74]

    # # Cameras FOV widths (degrees)
    # fov_widths = [40.72, 40.72, 40.72]

    # # Cameras FOV heights (degrees)
    # fov_heights = [27.21, 27.21, 27.21]

    # # Limiting magnitudes (needed only for ablation simulation)
    # lim_magnitudes = [+5.0, +5.0, +5.0]

    # # Powers of zero-magnitude meteors (Watts) (needed only for ablation simulation)
    # P_0m_list = [840, 840, 840]

    # # Minimum angular velocity for detection (deg/s)
    # min_ang_velocities = [1.0, 1.0, 1.0]

    # ##########################################################################################################

    # ### SIMULATED MODERATE STATION PARAMETERS ###
    # ##########################################################################################################

    # system_name = 'CAMSsim'

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
    # obs_ang_uncertainties = [30, 30, 30]

    # # Azimuths of centre of FOVs (degrees)
    # azim_fovs = [56.0, 300.0, 174.0]

    # # Elevations of centre of FOVs (degrees)
    # elev_fovs = [65.0, 65.0, 65.0]

    # # Cameras FOV widths (degrees)
    # fov_widths = [64.0, 64.0, 64.0]

    # # Cameras FOV heights (degrees)
    # fov_heights = [48.0, 48.0, 48.0]

    # # Limiting magnitudes (needed only for ablation simulation)
    # lim_magnitudes = [5.0, 5.0, 5.0]

    # # Powers of zero-magnitude meteors (Watts) (needed only for ablation simulation)
    # P_0m_list = [1210, 1210, 1210]

    # # Minimum angular velocity for detection (deg/s)
    # min_ang_velocities = [2.0, 2.0, 2.0]

    # ##########################################################################################################

    ### SIMULATED ALL-SKY STATION PARAMETERS ###
    ##########################################################################################################

    system_name = 'SOMN_sim'

    # Number of stations in total
    n_stations = 3

    # Maximum time offset (seconds)
    t_max_offset = 1.0

    # Geographical coordinates of stations (lat, lon, elev, station_id) in degrees and meters
    stations_geo = [
        [43.19279, -81.31565, 324.0, 'A1'],
        [43.19055, -80.09913, 212.0, 'A2'],
        [43.96324, -80.80952, 383.0, 'A3']]

    # Camera FPS per station
    fps_list = [30, 30, 30]

    # Observation uncertanties per station (arcsec)
    obs_ang_uncertainties = [120, 120, 120]

    # Azimuths of centre of FOVs (degrees)
    azim_fovs = [56.0, 300.0, 174.0]

    # Elevations of centre of FOVs (degrees)
    elev_fovs = [90.0, 90.0, 90.0]

    # Cameras FOV widths (degrees)
    fov_widths = [120.0, 120.0, 120.0]

    # Cameras FOV heights (degrees)
    fov_heights = [120.0, 120.0, 120.0]

    # Limiting magnitudes (needed only for ablation simulation)
    lim_magnitudes = [-0.5, -0.5, -0.5]

    # Powers of zero-magnitude meteors (Watts) (needed only for ablation simulation)
    P_0m_list = [1210, 1210, 1210]

    # Minimum angular velocity for detection (deg/s)
    min_ang_velocities = [1.0, 1.0, 1.0]

    ##########################################################################################################


    # ### SIMULATED ALL-SKY PRECISE STATION PARAMETERS ###
    # ##########################################################################################################

    # system_name = 'SOMN_precise_sim'

    # # Number of stations in total
    # n_stations = 3

    # # Maximum time offset (seconds)
    # t_max_offset = 1.0

    # # Geographical coordinates of stations (lat, lon, elev, station_id) in degrees and meters
    # stations_geo = [
    #     [43.19279, -81.31565, 324.0, 'A1'],
    #     [43.19055, -80.09913, 212.0, 'A2'],
    #     [43.96324, -80.80952, 383.0, 'A3']]

    # # Camera FPS per station
    # fps_list = [30, 30, 30]

    # # Observation uncertanties per station (arcsec)
    # obs_ang_uncertainties = [30, 30, 30]

    # # Azimuths of centre of FOVs (degrees)
    # azim_fovs = [56.0, 300.0, 174.0]

    # # Elevations of centre of FOVs (degrees)
    # elev_fovs = [90.0, 90.0, 90.0]

    # # Cameras FOV widths (degrees)
    # fov_widths = [120.0, 120.0, 120.0]

    # # Cameras FOV heights (degrees)
    # fov_heights = [120.0, 120.0, 120.0]

    # # Limiting magnitudes (needed only for ablation simulation)
    # lim_magnitudes = [-0.5, -0.5, -0.5]

    # # Powers of zero-magnitude meteors (Watts) (needed only for ablation simulation)
    # P_0m_list = [1210, 1210, 1210]

    # # Minimum angular velocity for detection (deg/s)
    # min_ang_velocities = [1.0, 1.0, 1.0]

    # ##########################################################################################################


    # Randomly generate station timing offsets, the first station has zero time offset
    t_offsets = np.random.uniform(-t_max_offset, +t_max_offset, size=n_stations)
    t_offsets[0] = 0


    # Init stations data to SimStation objects
    station_list = initStationList(stations_geo, azim_fovs, elev_fovs, fov_widths, fov_heights, t_offsets, \
        fps_list, obs_ang_uncertainties, lim_magnitudes=lim_magnitudes, P_0m_list=P_0m_list, \
        min_ang_velocities=min_ang_velocities)

    # Plot FOVs of stations at ceiling height of 120km
    plotStationFovs(station_list, datetime2JD(datetime.datetime.now()), 70*1000, 120*1000)


    ### METEOR SHOWER PARAMETERS ###
    ##########################################################################################################
    
    n_meteors = 100

    orbit_limits = None

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

    # # Velocity drift
    # d_vg = 0.0

    # year = 2012
    # month = 12

    # # Solar longitude of peak activity in degrees
    # sol_max = 261
    # sol_slope = 0.4

    # # Beginning height in kilometers
    # beg_height = 95
    # beg_height_sigma = 3

    # ###


    # ### URSIDS

    # # Shower name
    # shower_name = '2014Ursids'

    # # Radiant position and dispersion
    # ra_g = 219.9
    # ra_g_sigma = 3.2

    # dec_g = 75.4
    # dec_g_sigma = 1.1

    # # Radiant drift in degrees per degree of solar longitude
    # d_ra = 0.05
    # d_dec = -0.31

    # # Geocentric velocity in km/s
    # v_g = 32.9
    # v_g_sigma = 0.8

    # # Velocity drift
    # d_vg = 0.0

    # year = 2014
    # month = 12

    # # Solar longitude of peak activity in degrees
    # sol_max = 271.0
    # sol_slope = 0.61

    # # Beginning height in kilometers
    # beg_height = 95
    # beg_height_sigma = 3

    # ###


    # ### PERSEIDS

    # # Shower name
    # shower_name = '2012Perseids'

    # # Radiant position and dispersion
    # ra_g = 48.2
    # ra_g_sigma = 0.15

    # dec_g = 58.1
    # dec_g_sigma = 0.15

    # # Radiant drift in degrees per degree of solar longitude
    # d_ra = 1.40
    # d_dec = 0.26

    # # Geocentric velocity in km/s
    # v_g = 59.1
    # v_g_sigma = 0.1

    # # Velocity drift
    # d_vg = 0.0

    # year = 2012
    # month = 8

    # # Solar longitude of peak activity in degrees
    # sol_max = 140.0
    # sol_slope = 0.4

    # # Beginning height in kilometers
    # beg_height = 105
    # beg_height_sigma = 3

    # ###


    # ### 2011 Draconids

    # # Shower name
    # shower_name = '2011Draconids'

    # # Radiant position and dispersion
    # ra_g = 263.387
    # ra_g_sigma = 0.291019

    # dec_g = 55.9181
    # dec_g_sigma = 0.15746

    # # Radiant drift in degrees per degree of solar longitude
    # d_ra = 0.0
    # d_dec = 0.0

    # # Geocentric velocity in km/s
    # v_g = 20.9245
    # v_g_sigma = 0.04191

    # # Velocity drift
    # d_vg = 0.0

    # year = 2011
    # month = 10

    # # Solar longitude of peak activity in degrees
    # sol_max = 195.07 # Paul's sims
    # sol_slope = 17.5 # Koten et al. 2014 observations

    # # Beginning height in kilometers
    # beg_height = 95
    # beg_height_sigma = 3

    # ##


    # ### Long sporadic fireball

    # # Shower name
    # shower_name = 'LongFireball'

    # # Radiant position and dispersion
    # ra_g = 304.67053
    # ra_g_sigma = 0.04

    # dec_g = -7.28225
    # dec_g_sigma = 0.07

    # # Radiant drift in degrees per degree of solar longitude
    # d_ra = 0
    # d_dec = 0

    # # Geocentric velocity in km/s
    # v_g = 11.3
    # v_g_sigma = 0.02

    # # Velocity drift
    # d_vg = 0.0

    # year = 2017
    # month = 9

    # # Solar longitude of peak activity in degrees
    # sol_max = 180.150305
    # sol_slope = 20

    # # Beginning height in kilometers
    # beg_height = 76.4 
    # beg_height_sigma = 3

    # ###


    ### 2015 Taurid outburst

    # Shower name
    shower_name = '2015Taurids'

    # Radiant position and dispersion
    ra_g = 53.059624
    ra_g_sigma = 0.334

    dec_g = 14.65736
    dec_g_sigma = 0.267

    # Radiant drift in degrees per degree of solar longitude
    d_ra = 0.554
    d_dec = 0.06

    # Geocentric velocity in km/s
    v_g = 29.689892
    v_g_sigma = 0.223

    # Velocity drift
    d_vg = -0.293

    year = 2015
    month = 10

    # Solar longitude of peak activity in degrees
    sol_max = 220.956
    sol_slope = 0.15

    # Beginning height in kilometers
    beg_height = 100
    beg_height_sigma = 3


    # Set constraints to the orbit
    orbit_limits = ['a', 2.24, 2.28]

    ###


    ##########################################################################################################


    ### METEOR VELOCITY MODEL ###
    ##########################################################################################################

    # Set a range of meteor durations
    #meteor_durations = np.clip(np.random.normal(0.5, 0.1, n_meteors), 0.2, 1.0)
    meteor_durations = [2.0]*n_meteors

    # #### Constant velocity model
    # meteor_velocity_models = [ConstantVelocity(duration) for duration in meteor_durations]
    # ####


    # #### Linear deceleration model
    
    # # Randomly generate deceleration times t0
    # #t0_rand = np.random.uniform(0.1, 0.7, size=n_meteors) # Ratios of deceleratoin start
    # t0_rand = np.random.uniform(0.3, 0.6, size=n_meteors) # Ratios of deceleratoin start
    # t0_list = meteor_durations*t0_rand

    # # Randomly generate decelerations
    # #decel_list = np.random.uniform(100, 800, size=n_meteors)
    # decel_list = np.random.uniform(4000, 6000, size=n_meteors)

    # meteor_velocity_models = [LinearDeceleration(duration, t0, decel) for duration, t0, decel in \
    #     zip(meteor_durations, t0_list, decel_list)]

    # ####


    # #### Jacchia (exponential deceleration) velocity model
    # a1_list = np.random.uniform(0.08, 0.15, n_meteors)
    # a2_list = np.random.uniform(8, 15, n_meteors)

    # meteor_velocity_models = [JacchiaVelocity(duration, a1, a2) for duration, a1, a2 in zip(meteor_durations,\
    #     a1_list, a2_list)]

    # ####





    # # ## Velocity model from Campbell-Brown & Koschny (2004) meteor ablation model #####

    # ## 2011 Draconids ###
    # # Make the beginning heights heigher, as the trajectory points will be determined by simulated
    # #   magnitudes
    # beg_height = 110
    # beg_height_sigma = 0

    # # Luminous efficiency (fraction)
    # lum_eff = 0.7/100

    # # Ablation coefficient (s^2/km^2)
    # ablation_coeff = 0.21 # Ceplecha et al. 1998, D type

    # # Drag coeficient
    # Gamma = 1.0

    # # Heat transfer coeficient
    # Lambda = 0.5


    # # # Mass range (log of mass in kg) seen by the system (EMCCD, 20 km/s, Draconids)
    # # mass_min = -6.5
    # # mass_max = -4.5

    # # # Mass range (log of mass in kg) seen by the system (CAMO, 20 km/s, Draconids)
    # # mass_min = -6.0
    # # mass_max = -4.0

    # # Mass range (log of mass in kg) seen by the system (allsky, 20 km/s, Draconids)
    # mass_min = -2.5
    # mass_max = 1.5

    # # # Mass range (log of mass in kg) seen by the system (moderate, 20 km/s, Draconids)
    # # mass_min = -4.9
    # # mass_max = -3.0

    # # Mass index
    # mass_index = 1.95 # Koten et al. 2014

    # # Sample densities (Borovicka et al. 2013: Radiants, orbits, spectra, and deceleration of selected 2011 
    # #   Draconids)
    # density_samples = np.random.uniform(100, 400, n_meteors)


    # # \ 2011 Draconids



    # ## Perseids ###

    # # Make the beginning heights heigher, as the trajectory points will be determined by simulated
    # # magnitudes
    # beg_height = 130
    # beg_height_sigma = 0

    # # Luminous efficiency (fraction)
    # lum_eff = 0.7/100

    # # Ablation coefficient (s^2/km^2) (cometary)
    # ablation_coeff = 0.1

    # # Drag coeficient
    # Gamma = 1.0

    # # Heat transfer coeficient
    # Lambda = 0.5

    # # Mass index
    # mass_index = 2.0 # Beech et al. 1999

    # # Mass range (log of mass in kg) seen by the system (allsky, 60 km/s, Perseids)
    # mass_min = -3.5
    # mass_max = -1

    # # # Mass range (log of mass in kg) seen by the system (moderate, 60 km/s, Perseids)
    # # mass_min = -6.0
    # # mass_max = -4.0

    # # # Mass range (log of mass in kg) seen by the system (CAMO, 60 km/s, Perseids)
    # # mass_min = -6.5
    # # mass_max = -4.5


    # # Define density distribution (see: Moorhead et al. 2017 "A two-population sporadic meteoroid density 
    # #        distribution and its implications for environment models")

    # # HTC density distribution (Tj <= 2)
    # log_rho_mean = 2.93320
    # log_rho_sigma = 0.12714

    # # # JFC and asteroidal distribution (Tj > 2) 
    # # log_rho_mean = 3.57916
    # # log_rho_sigma = 0.09312

    # # Samples densities
    # density_samples = sampleDensityMoorhead(log_rho_mean, log_rho_sigma, n_meteors)

    # ## \Perseids



    # ## Ursids ###

    # # Make the beginning heights heigher, as the trajectory points will be determined by simulated
    # # magnitudes
    # beg_height = 120
    # beg_height_sigma = 0

    # # Luminous efficiency (fraction)
    # lum_eff = 0.7/100

    # # Ablation coefficient (s^2/km^2) (cometary)
    # ablation_coeff = 0.1

    # # Drag coeficient
    # Gamma = 1.0

    # # Heat transfer coeficient
    # Lambda = 0.5

    # # Mass index
    # mass_index = 1.58 # compured from population index s = 1 + 2.5*log10(r) from Molau et al. 2015

    # # Mass range (log of mass in kg) seen by the system (allsky, 30 km/s, Ursids)
    # mass_min = -3.0
    # mass_max = 0.0

    # # # Mass range (log of mass in kg) seen by the system (CAMS, 30 km/s, Ursids)
    # # mass_min = -5.5
    # # mass_max = -3.4

    # # # Mass range (log of mass in kg) seen by the system (CAMO, 30 km/s, Ursids)
    # # mass_min = -6.3
    # # mass_max = -4.5


    # # Define density distribution (see: Moorhead et al. 2017 "A two-population sporadic meteoroid density 
    # #        distribution and its implications for environment models")

    # # HTC density distribution (Tj <= 2)
    # log_rho_mean = 2.93320
    # log_rho_sigma = 0.12714

    # # Samples densities
    # density_samples = sampleDensityMoorhead(log_rho_mean, log_rho_sigma, n_meteors)

    # ## \Ursids



    ## Taurids ###

    # Make the beginning heights heigher, as the trajectory points will be determined by simulated
    # magnitudes
    beg_height = 120
    beg_height_sigma = 0

    # Luminous efficiency (fraction)
    lum_eff = 0.7/100

    # Ablation coefficient (s^2/km^2) (cometary)
    ablation_coeff = 0.1

    # Drag coeficient
    Gamma = 1.0

    # Heat transfer coeficient
    Lambda = 0.5

    # Mass index
    mass_index = 1.8

    # Mass range (log of mass in kg) seen by the system (allsky, 30 km/s, Taurids)
    mass_min = -3.0
    mass_max = 0.0


    # Sample densities - around 1400 kg/m3
    # Reference: Brown, P., Marchenko, V., Moser, D. E., Weryk, R., & Cooke, W. (2013). Meteorites from 
    # meteor showers: A case study of the Taurids. Meteoritics & Planetary Science, 48(2), 270-288.
    density_samples = np.random.uniform(1200, 1600, n_meteors)

    ## \Taurids


    # Init velocity models
    meteor_velocity_models = [AblationModelVelocity(mass_min, mass_max, mass_index, density, ablation_coeff, \
        Gamma, Lambda, lum_eff) for density in density_samples]


    # ####################################################################################

    ##########################################################################################################


    # Make the system directory
    system_dir = os.path.join(dir_path, system_name)
    mkdirP(system_dir)

    # Make the shower directory
    shower_dir = os.path.join(system_dir, shower_name)
    mkdirP(shower_dir)



    # Run shower simulation
    sim_meteor_list = simulateMeteorShower(station_list, meteor_velocity_models, n_meteors, ra_g, ra_g_sigma, 
        dec_g, dec_g_sigma, d_ra, d_dec, v_g, v_g_sigma, d_vg, year, month, sol_max, sol_slope, beg_height, 
        beg_height_sigma, output_dir=shower_dir, orbit_limits=orbit_limits, nighttime_meteors_only=False)



    # Plot meteor positions
    plotMeteorTracks(station_list, sim_meteor_list, output_dir=shower_dir)


    # Save simulated meteors to disk
    for met_no, sim_met in enumerate(sim_meteor_list):

        sim_met.saveInfo(shower_dir)

        print('Saved meteor:', sim_met.jdt_ref)

