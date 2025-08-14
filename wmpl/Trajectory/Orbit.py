from __future__ import print_function, division, absolute_import

import os
import sys
import datetime
import argparse

import numpy as np

from jplephem.spk import SPK

import wmpl
from wmpl.Config import config
from wmpl.Utils.Earth import calcEarthRectangularCoordJPL
from wmpl.Utils.ShowerAssociation import associateShower
from wmpl.Utils.SolarLongitude import jd2SolLonJPL
from wmpl.Utils.TrajConversions import J2000_JD, J2000_OBLIQUITY, AU, SUN_MU, SUN_MASS, G, SIDEREAL_YEAR, \
    jd2LST, jd2Date, jd2DynamicalTimeJD, eci2RaDec, altAz2RADec, raDec2AltAz, raDec2Ecliptic, cartesian2Geo,\
    equatorialCoordPrecession, eclipticToRectangularVelocityVect, correctedEclipticCoord, datetime2JD, \
    geo2Cartesian
from wmpl.Utils.Math import vectNorm, vectMag, rotateVector, cartesianToSpherical, sphericalToCartesian
from wmpl.Utils.Misc import valueFormat
from wmpl.Utils.Pickling import loadPickle



class Orbit(object):
    """ Structure for storing the orbit solution of a meteor. """

    def __init__(self):


        ### Apparent radiant in ECI (Earth's rotation is included) ###

        # Apparent radiant position (ECI, radians)
        self.ra = None
        self.dec = None

        # Apparent azimuth and altitude (ECI)
        self.azimuth_apparent = None
        self.elevation_apparent = None

        # Estimated average velocity (ECI)
        self.v_avg = None

        # Estimated initial velocity (ECI)
        self.v_init = None

        # Estimated initial velocity standard deviation from the direct fit (ECI)
        self.v_init_stddev = None

        ### ###



        ### Apparent radiant which includes no Earth's rotation (reference to the ground) ###

        # Apparent radiant position (no Earth's rotation, radians)
        self.ra_norot = None
        self.dec_norot = None

        # Apparent azimuth and altitude (no Earth's rotation)
        self.azimuth_apparent_norot = None
        self.elevation_apparent_norot = None

        # Estimated average velocity (no Earth's rotation)
        self.v_avg_norot = None

        # Estimated initial velocity (no Earth's rotation)
        self.v_init_norot = None

        ### ###



        # Reference Julian date for the trajectory. Can be the time of the first point on the trajectory or the
        # average time of the meteor
        self.jd_ref = None

        # Dynamical Julian date
        self.jd_dyn = None

        # reference Local Sidreal Time of the reference trajectory position
        self.lst_ref = None

        # Longitude of the reference point on the trajectory (rad)
        self.lon_ref = None

        # Latitude of the reference point on the trajectory (rad)
        self.lat_ref = None

        # Height of the reference point on the trajectory (meters)
        self.ht_ref = None
        self.ht_ref_wgs84 = None

        # Geocentric latitude of the reference point on the trajectory (rad)
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

        # Components of the heliocentric velocity vector of the meteoroid
        self.v_h_x = None
        self.v_h_y = None
        self.v_h_z = None

        # Heliocentric ecliptic coordinates of the meteor
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

        # Latitude of perihelion (radians)
        self.b = None

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

        # Tisserand's parameter with respect to Jupiter
        self.Tj = None

        # Orbital period
        self.T = None


    def fixMissingParameters(self):
        """ Some old orbit files might have missing parameters that were not computed. This function computes
            them.
        """

        if (not hasattr(self, 'b')) and (self.v_g is not None):
            self.b = calcLatitudeOfPerihelion(self.peri, self.node, self.i)


    def __repr__(self, uncertainties=None, v_init_ht=None):
        """ String to be printed out when the Orbit object is printed. """

        out_str =  ""
        #out_str +=  "--------------------\n"

        # Check if the orbit was calculated
        if self.ra_g is not None:
            out_str += "  JD dynamic   = {:20.12f} \n".format(self.jd_dyn)
            out_str += "  LST apparent = {:.10f} deg\n".format(np.degrees(self.lst_ref))


        ### Apparent radiant in ECI ###

        out_str += "Radiant (apparent in ECI which includes Earth's rotation, epoch of date):\n"
        out_str += "  R.A.      = {:s} deg\n".format(valueFormat("{:>9.5f}", self.ra, "{:.5f}", \
            uncertainties, 'ra', deg=True))
        out_str += "  Dec       = {:s} deg\n".format(valueFormat("{:>+9.5f}", self.dec, "{:.5f}", \
            uncertainties, 'dec', deg=True))
        out_str += "  Azimuth   = {:s} deg\n".format(valueFormat("{:>9.5f}", self.azimuth_apparent, \
            "{:.5f}", uncertainties, 'azimuth_apparent', deg=True))
        out_str += "  Elevation = {:s} deg\n".format(valueFormat("{:>+9.5f}", self.elevation_apparent, \
            "{:.5f}", uncertainties, 'elevation_apparent', deg=True))
        out_str += "  Vavg      = {:s} km/s\n".format(valueFormat("{:>9.5f}", self.v_avg, "{:.5f}", \
            uncertainties, 'v_avg', multi=1.0/1000))


        if v_init_ht is not None:
            v_init_ht_str = ' (average above {:.2f} km)'.format(v_init_ht)
        else:
            v_init_ht_str = ''

        out_str += "  Vinit     = {:s} km/s{:s}\n".format(valueFormat("{:>9.5f}", self.v_init, "{:.5f}", \
            uncertainties, 'v_init', multi=1.0/1000), v_init_ht_str)


        # Write out the direct fit velocity standard deviation
        if hasattr(self, "v_init_stddev"):
            if self.v_init_stddev is not None:
                out_str += "  Vstddev   = {:>9.5f} km/s (direct fit)\n".format(self.v_init_stddev/1000.0)


        ### ###


        ### Apparent radiant in ECEF (no rotation included) ###

        out_str += "Radiant (apparent ground-fixed, epoch of date):\n"
        out_str += "  R.A.      = {:s} deg\n".format(valueFormat("{:>9.5f}", self.ra_norot, "{:.5f}", \
            uncertainties, 'ra_norot', deg=True))
        out_str += "  Dec       = {:s} deg\n".format(valueFormat("{:>+9.5f}", self.dec_norot, "{:.5f}", \
            uncertainties, 'dec_norot', deg=True))
        out_str += "  Azimuth   = {:s} deg\n".format(valueFormat("{:>9.5f}", self.azimuth_apparent_norot, \
            "{:.5f}", uncertainties, 'azimuth_apparent_norot', deg=True))
        out_str += "  Elevation = {:s} deg\n".format(valueFormat("{:>+9.5f}", self.elevation_apparent_norot, \
            "{:.5f}", uncertainties, 'elevation_apparent_norot', deg=True))
        out_str += "  Vavg      = {:s} km/s\n".format(valueFormat("{:>9.5f}", self.v_avg_norot, "{:.5f}", \
            uncertainties, 'v_avg_norot', multi=1.0/1000))
        out_str += "  Vinit     = {:s} km/s{:s}\n".format(valueFormat("{:>9.5f}", self.v_init_norot, \
            "{:.5f}", uncertainties, 'v_init_norot', multi=1.0/1000), v_init_ht_str)



        ### ###


        # Check if the orbital elements could be calculated, and write them out
        if self.ra_g is not None:

            out_str += "Radiant (geocentric, J2000):\n"
            out_str += "  R.A.   = {:s} deg\n".format(valueFormat("{:>9.5f}", self.ra_g, '{:.5f}', \
                uncertainties, 'ra_g', deg=True))
            out_str += "  Dec    = {:s} deg\n".format(valueFormat("{:>+9.5f}", self.dec_g, '{:.5f}', \
                uncertainties, 'dec_g', deg=True))
            out_str += "  Vg     = {:s} km/s\n".format(valueFormat("{:>9.5f}", self.v_g, '{:.5f}', \
                uncertainties, 'v_g', multi=1.0/1000))
            out_str += "  Vinf   = {:s} km/s\n".format(valueFormat("{:>9.5f}", self.v_inf, '{:.5f}', \
                uncertainties, 'v_inf', multi=1.0/1000))
            out_str += "  Zc     = {:s} deg\n".format(valueFormat("{:>9.5f}", self.zc, '{:.5f}', \
                uncertainties, 'zc', deg=True))
            out_str += "  Zg     = {:s} deg\n".format(valueFormat("{:>9.5f}", self.zg, '{:.5f}', \
                uncertainties, 'zg', deg=True))
            out_str += "Radiant (ecliptic geocentric, J2000):\n"
            out_str += "  Lg     = {:s} deg\n".format(valueFormat("{:>9.5f}", self.L_g, '{:.5f}', \
                uncertainties, 'L_g', deg=True))
            out_str += "  Bg     = {:s} deg\n".format(valueFormat("{:>+9.5f}", self.B_g, '{:.5f}', \
                uncertainties, 'B_g', deg=True))
            out_str += "  Vh     = {:s} km/s\n".format(valueFormat("{:>9.5f}", self.v_h, '{:.5f}', \
                uncertainties, 'v_h', multi=1/1000.0))
            out_str += "Radiant (ecliptic heliocentric, J2000):\n"
            out_str += "  Lh     = {:s} deg\n".format(valueFormat("{:>9.5f}", self.L_h, '{:.5f}', \
                uncertainties, 'L_h', deg=True))
            out_str += "  Bh     = {:s} deg\n".format(valueFormat("{:>+9.5f}", self.B_h, '{:.5f}', \
                uncertainties, 'B_h', deg=True))
            out_str += "  Vh_x   = {:s} km/s\n".format(valueFormat("{:>9.5f}", self.v_h_x, '{:.5f}', \
                uncertainties, 'v_h_x'))
            out_str += "  Vh_y   = {:s} km/s\n".format(valueFormat("{:>9.5f}", self.v_h_y, '{:.5f}', \
                uncertainties, 'v_h_y'))
            out_str += "  Vh_z   = {:s} km/s\n".format(valueFormat("{:>9.5f}", self.v_h_z, '{:.5f}', \
                uncertainties, 'v_h_z'))
            out_str += "Orbit:\n"
            out_str += "  La Sun = {:s} deg\n".format(valueFormat("{:>10.6f}", self.la_sun, '{:.6f}', \
                uncertainties, 'la_sun', deg=True))
            out_str += "  a      = {:s} AU\n".format(valueFormat("{:>10.6f}", self.a, '{:.6f}', \
                uncertainties, 'a'))
            out_str += "  e      = {:s}\n".format(valueFormat("{:>10.6f}", self.e, '{:.6f}', \
                uncertainties, 'e'))
            out_str += "  i      = {:s} deg\n".format(valueFormat("{:>10.6f}", self.i, '{:.6f}', \
                uncertainties, 'i', deg=True))
            out_str += "  peri   = {:s} deg\n".format(valueFormat("{:>10.6f}", self.peri, '{:.6f}', \
                uncertainties, 'peri', deg=True))
            out_str += "  node   = {:s} deg\n".format(valueFormat("{:>10.6f}", self.node, '{:.6f}', \
                uncertainties, 'node', deg=True))
            out_str += "  Pi     = {:s} deg\n".format(valueFormat("{:>10.6f}", self.pi, '{:.6f}', \
                uncertainties, 'pi', deg=True))
            if hasattr(self, 'b'):
                out_str += "  b      = {:s} deg\n".format(valueFormat("{:>10.6f}", self.b, '{:.6f}', \
                uncertainties, 'b', deg=True))
            out_str += "  q      = {:s} AU\n".format(valueFormat("{:>10.6f}", self.q, '{:.6f}', \
                uncertainties, 'q'))
            out_str += "  f      = {:s} deg\n".format(valueFormat("{:>10.6f}", self.true_anomaly, '{:.6f}', \
                uncertainties, 'true_anomaly', deg=True))
            out_str += "  M      = {:s} deg\n".format(valueFormat("{:>10.6f}", self.mean_anomaly, '{:.6f}', \
                uncertainties, 'mean_anomaly', deg=True))
            out_str += "  Q      = {:s} AU\n".format(valueFormat("{:>10.6f}", self.Q, '{:.6f}', \
                uncertainties, 'Q'))
            out_str += "  n      = {:s} deg/day\n".format(valueFormat("{:>10.6f}", self.n, '{:.6f}', \
                uncertainties, 'n', deg=True))
            out_str += "  T      = {:s} years\n".format(valueFormat("{:>10.6f}", self.T, '{:.6f}', \
                uncertainties, 'T'))
            
            if self.last_perihelion is not None:
                out_str += "  Last perihelion JD = {:s}\n".format(valueFormat("{:.4f}", \
                    self.last_perihelion, "{:.4f}", uncertainties, 'last_perihelion', \
                    callable_val=datetime2JD), callable_ci=None)
                out_str += "  Last perihelion dt = {:s}\n".format(valueFormat("{:s}", \
                    self.last_perihelion, "{:.4f} days", uncertainties, 'last_perihelion', \
                    callable_val=lambda x: datetime.datetime.strftime(x, "%Y-%m-%d %H:%M:%S"), \
                    callable_ci=lambda x: datetime.datetime.strftime(jd2Date(x, dt_obj=True), \
                        "%Y-%m-%d %H:%M:%S")))
            else:
                out_str += "  Last perihelion JD = NaN \n"
                out_str += "  Last perihelion dt = NaN \n"

            out_str += "  Tj     = {:s}\n".format(valueFormat("{:>10.6f}", self.Tj, '{:.6f}', \
                uncertainties, 'Tj'))


            out_str += "Shower association:\n"

            # Perform shower association
            shower_obj = associateShower(self.la_sun, self.L_g, self.B_g, self.v_g)
            if shower_obj is None:
                shower_no = -1
                shower_code = '...'
            else:
                shower_no = shower_obj.IAU_no
                shower_code = shower_obj.IAU_code

            out_str += "  IAU No.  = {:>4d}\n".format(shower_no)
            out_str += "  IAU code = {:>4s}\n".format(shower_code)


        return out_str



def calcLatitudeOfPerihelion(peri, node, incl):
    """ Calculate the latitude of perihelion. 
        Source: https://en.wikipedia.org/wiki/Longitude_of_the_periapsis#Derivation_of_ecliptic_longitude_and_latitude_of_perihelion_for_inclined_orbits
    """

    Ap = np.cos(peri)*np.cos(node) - np.sin(peri)*np.sin(node)*np.cos(incl)
    Bp = np.cos(J2000_OBLIQUITY)*(np.cos(peri)*np.sin(node) + np.sin(peri)*np.cos(node)*np.cos(incl)) \
        - np.sin(J2000_OBLIQUITY)*np.sin(peri)*np.sin(incl)
    Cp = np.sin(J2000_OBLIQUITY)*(np.cos(peri)*np.sin(node) + np.sin(peri)*np.cos(node)*np.cos(incl)) \
        + np.cos(J2000_OBLIQUITY)*np.sin(peri)*np.sin(incl)

    # RA/Dec of the direction of perihelion
    ra_p = np.arctan2(Bp, Ap)%(2*np.pi)
    dec_p = np.arcsin(Cp)
    
    # Longitue of perihelion
    # pi_t = np.arctan2(np.sin(ra_p)*np.cos(J2000_OBLIQUITY) + np.tan(dec_p)*np.sin(J2000_OBLIQUITY), np.cos(ra_p))%(2*np.pi)
    
    # Latitude of perihelion
    b = np.arcsin(np.sin(dec_p)*np.cos(J2000_OBLIQUITY) - np.cos(dec_p)*np.sin(J2000_OBLIQUITY)*np.sin(ra_p))

    return b



def calcOrbitalElements(jd_ref, ra_g, dec_g, v_g, eci_ref):
    """ Compute keplerian orbital elements (+ a few other parametrs), given the geocentric radiant and
        a reference ECI location.

    Arguments:
        jd_ref: [float] Reference Julian date of the meteor trajectory.
        eci_ref: [ndarry] X, Y, Z components of ECI coordinates at the reference Julian date (meters) in J2000.
        ra_g: [float] Geocentric radiant, right ascension (J2000, radians).
        dec_g: [float] Geocentric radiant, declination (J2000, radians).
        v_g: [float] Geocentic velocity (m/s).

    Return:
        (tuple): orbital elements
    """

    # Calculate the dynamical JD
    jd_dyn = jd2DynamicalTimeJD(jd_ref)

    # Calculate the ecliptic latitude and longitude of the geocentric radiant (J2000 epoch)
    L_g, B_g = raDec2Ecliptic(J2000_JD.days, ra_g, dec_g)


    # Load the JPL ephemerids data
    jpl_ephem_data = SPK.open(config.jpl_ephem_file)
    
    # Get the position of the Earth (km) and its velocity (km/s) at the given Julian date (J2000 epoch)
    # The position is given in the ecliptic coordinates, origin of the coordinate system is in the centre
    # of the Sun
    earth_pos, earth_vel = calcEarthRectangularCoordJPL(jd_dyn, jpl_ephem_data, sun_centre_origin=True)

    # print('Earth position:')
    # print(earth_pos)
    # print('Earth velocity:')
    # print(earth_vel)

    # Convert the Earth's position to rectangular equatorial coordinates (FK5)
    earth_pos_eq = rotateVector(earth_pos, np.array([1.0, 0.0, 0.0]), J2000_OBLIQUITY)

    # print('Earth position (FK5):')
    # print(earth_pos_eq)

    # print('Meteor ECI:')
    # print(eci_ref)

    # Add the position of the meteor's trajectory to the position of the Earth to calculate the 
    # equatorial coordinates of the meteor (in kilometers)
    meteor_pos = earth_pos_eq + eci_ref/1000


    # print('Meteor position (FK5):')
    # print(meteor_pos)

    # Convert the position of the trajectory from FK5 to heliocentric ecliptic coordinates
    meteor_pos = rotateVector(meteor_pos, np.array([1.0, 0.0, 0.0]), -J2000_OBLIQUITY)

    # print('Meteor position:')
    # print(meteor_pos)


    ##########################################################################################################

    # Calculate components of the heliocentric velocity of the meteor (km/s)
    v_h = np.array(earth_vel) + np.array(eclipticToRectangularVelocityVect(L_g, B_g, v_g/1000))

    # Calculate the heliocentric velocity in km/s
    v_h_mag = vectMag(v_h)


    # Calculate the heliocentric ecliptic coordinates of the meteoroid using the method of 
    # Sato and Watanabe (2014).
    L_h, B_h, met_v_h = correctedEclipticCoord(L_g, B_g, v_g/1000, earth_vel)


    # Calculate the solar longitude
    la_sun = jd2SolLonJPL(jd_dyn)


    # Calculations below done using Dave Clark's Master thesis equations

    # Specific orbital energy
    epsilon = (vectMag(v_h)**2)/2 - SUN_MU/vectMag(meteor_pos)

    # Semi-major axis in AU
    a = -SUN_MU/(2*epsilon*AU)

    # Calculate mean motion in rad/day
    n = np.sqrt(G*SUN_MASS/((np.abs(a)*AU*1000.0)**3))*86400.0


    # Calculate the orbital period in years
    # avoid floating point error if orbit is hyperbolic
    if a > 0: 
        T = 2*np.pi*np.sqrt(((a*AU)**3)/SUN_MU)/(86400*SIDEREAL_YEAR)
    else:
        T = np.nan

    # Calculate the orbit angular momentum
    h_vect = np.cross(meteor_pos, v_h)
    
    # Calculate inclination
    incl = np.arccos(h_vect[2]/vectMag(h_vect))


    # Calculate eccentricity
    e_vect = np.cross(v_h, h_vect)/SUN_MU - vectNorm(meteor_pos)
    eccentricity = vectMag(e_vect)


    # Calculate perihelion distance (source: Jenniskens et al., 2011, CAMS overview paper)
    if eccentricity == 1:
        q = (vectMag(meteor_pos) + np.dot(e_vect, meteor_pos))/(1 + vectMag(e_vect))
    else:
        q = a*(1.0 - eccentricity)

    # Calculate the aphelion distance
    Q = a*(1.0 + eccentricity)


    # Normal vector to the XY reference frame
    k_vect = np.array([0, 0, 1])

    # Vector from the Sun pointing to the ascending node
    n_vect = np.cross(k_vect, h_vect)

    # Calculate node
    if vectMag(n_vect) == 0:
        node = 0
    else:
        node = np.arctan2(n_vect[1], n_vect[0])

    node = node%(2*np.pi)


    # Calculate argument of perihelion
    if vectMag(n_vect) != 0:
        peri = np.arccos(np.dot(n_vect, e_vect)/(vectMag(n_vect)*vectMag(e_vect)))

        if e_vect[2] < 0:
            peri = 2*np.pi - peri

    else:
        peri = np.arccos(e_vect[0]/vectMag(e_vect))

    peri = peri%(2*np.pi)



    # Calculate the longitude of perihelion
    pi = (node + peri)%(2*np.pi)

    # Calculate the latitude of perihelion
    b = calcLatitudeOfPerihelion(peri, node, incl)


    ### Calculate true anomaly
    true_anomaly = np.arccos(np.dot(e_vect, meteor_pos)/(vectMag(e_vect)*vectMag(meteor_pos)))
    if np.dot(meteor_pos, v_h) < 0:
        true_anomaly = 2*np.pi - true_anomaly

    true_anomaly = true_anomaly%(2*np.pi)

    ###


    # Calculate eccentric anomaly
    # not meaningful for eccentricity > 1
    if eccentricity < 1: 
        eccentric_anomaly = np.arctan2(np.sqrt(1 - eccentricity**2)*np.sin(true_anomaly), eccentricity \
            + np.cos(true_anomaly))

        # Calculate mean anomaly
        mean_anomaly = eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly)
        mean_anomaly = mean_anomaly % (2 * np.pi)
    else:
        eccentric_anomaly = np.nan
        mean_anomaly = np.nan

    # Calculate the time in days since the last perihelion passage of the meteoroid
    # not meaningful for non-closed orbits
    if a > 0:
        dt_perihelion = (mean_anomaly*a**(3.0/2))/0.01720209895
    else:
        dt_perihelion = np.nan

    if not np.isnan(dt_perihelion):
        
        # Calculate the date and time of the last perihelion passage
        last_perihelion = jd2Date(jd_dyn - dt_perihelion, dt_obj=True)

    else:
        last_perihelion = None


    # Calculate Tisserand's parameter with respect to Jupiter
    Tj = 2*np.sqrt((1 - eccentricity**2)*a/5.204267)*np.cos(incl) + 5.204267/a

    return (meteor_pos, L_g, B_g , met_v_h, L_h, B_h, v_h_mag*1000, la_sun, a, eccentricity, incl, 
        peri, node, pi, b, q, Q, true_anomaly, eccentric_anomaly, mean_anomaly, last_perihelion, n, T, Tj)



def calcOrbit(radiant_eci, v_init, v_avg, eci_ref, jd_ref, stations_fixed=False, reference_init=True, \
    rotation_correction=False, v_init_stddev_direct=None):
    """ Calculate the meteor's orbit from the given meteor trajectory. The orbit of the meteoroid is defined 
        relative to the centre of the Sun (heliocentric).

    Arguments:
        radiant_eci: [3 element ndarray] Radiant vector in ECI coordinates (meters).
        v_init: [float] Initial velocity (m/s).
        v_avg: [float] Average velocity of a meteor (m/s).
        eci_ref: [float] reference ECI coordinates in the epoch of date (meters, in the epoch of date) of the 
            meteor trajectory. They can be calculated with the geo2Cartesian function. Ceplecha (1987) assumes 
            this to the the average point on the trajectory, while Jennsikens et al. (2011) assume this to be 
            the first point on the trajectory as that point is not influenced by deceleration.
            NOTE: If the stations are not fixed, the reference ECI coordinates should be the ones
            of the initial point on the trajectory, NOT of the average point!
        jd_ref: [float] reference Julian date of the meteor trajectory. Ceplecha (1987) takes this as the 
            average time of the trajectory, while Jenniskens et al. (2011) take this as the the first point
            on the trajectory.
    
    Keyword arguments:
        stations_fixed: [bool] If True, the correction for Earth's rotation will be performed on the radiant,
            but not the velocity. This should be True ONLY in two occasions:
                - if the ECEF coordinate system was used for trajectory estimation
                - if the ECI coordinate system was used for trajectory estimation, BUT the stations were not
                    moved in time, but were kept fixed at one point, regardless of the trajectory estimation
                    method.
            It is necessary to perform this correction for the intersecting planes method, but not for
            the lines of sight method ONLY when the stations are not fixed. Of course, if one is using the 
            lines of sight method with fixed stations, one should perform this correction!
        reference_init: [bool] If True (default), the initial point on the trajectory is given as the reference
            one, i.e. the reference ECI coordinates are the ECI coordinates of the initial point on the
            trajectory, where the meteor has the velocity v_init. If False, then the reference point is the
            average point on the trajectory, and the average velocity will be used to do the corrections.
        rotation_correction: [bool] If True, the correction of the initial velocity for Earth's rotation will
            be performed. False by default. This should ONLY be True if the coordiante system for trajectory
            estimation was ECEF, i.e. did not rotate with the Earth. In all other cases it should be False, 
            even if fixed station coordinates were used in the ECI coordinate system!
        v_init_stddev_direct: [float] Standard deviation of the direct velocity fit derived without Monte
            Carlo. None by default.

    Return:
        orb: [Orbit object] Object containing the calculated orbit.

    """


    ### Correct the velocity vector for the Earth's rotation if the stations are fixed ###
    ##########################################################################################################

    eci_x, eci_y, eci_z = eci_ref
    eci_ref = np.array(eci_ref)

    # Calculate the geocentric latitude (latitude which considers the Earth as an elipsoid) of the reference 
    # trajectory point
    lat_geocentric = np.arctan2(eci_z, np.sqrt(eci_x**2 + eci_y**2))


    # Calculate the dynamical JD
    jd_dyn = jd2DynamicalTimeJD(jd_ref)

    # Calculate the geographical coordinates of the reference trajectory ECI position
    lat_ref, lon_ref, ht_ref = cartesian2Geo(jd_ref, *eci_ref)


    # Initialize a new orbit structure and assign calculated parameters
    orb = Orbit()



    # Calculate the velocity of the Earth rotation at the position of the reference trajectory point (m/s)
    v_e = 2*np.pi*vectMag(eci_ref)*np.cos(lat_geocentric)/86164.09053

    
    # Calculate the equatorial coordinates of east from the reference position on the trajectory
    azimuth_east = np.pi/2
    altitude_east = 0
    ra_east, dec_east = altAz2RADec(azimuth_east, altitude_east, jd_ref, lat_ref, lon_ref)


    # Compute velocity components of the state vector
    if reference_init:

        # If the initial velocity was the reference velocity, use it for the correction
        v_ref_vect = v_init*radiant_eci


    else:
        # Calculate reference velocity vector using the average point on the trajectory and the average
        # velocity
        v_ref_vect = v_avg*radiant_eci



    # Apply the Earth rotation correction if the station coordinates are fixed (a MUST for the 
    # intersecting planes method!)
    if stations_fixed:

        ### Set fixed stations radiant info ###

        # If the stations are fixed, then the input state vector is already fixed to the ground
        orb.ra_norot, orb.dec_norot = eci2RaDec(radiant_eci)

        # Apparent azimuth and altitude (no rotation)
        orb.azimuth_apparent_norot, orb.elevation_apparent_norot = raDec2AltAz(orb.ra_norot, orb.dec_norot, \
            jd_ref, lat_ref, lon_ref)

        # Estimated average velocity (no rotation)
        orb.v_avg_norot = v_avg

        # Estimated initial velocity (no rotation)
        orb.v_init_norot = v_init

        ### ###


        v_ref_corr = np.zeros(3)

        # Calculate the corrected reference velocity vector/radiant
        v_ref_corr[0] = v_ref_vect[0] - v_e*np.cos(ra_east)
        v_ref_corr[1] = v_ref_vect[1] - v_e*np.sin(ra_east)
        v_ref_corr[2] = v_ref_vect[2]



    else:

        # MOVING STATIONS
        # Velocity vector will remain unchanged if the stations were moving
        if reference_init:
            v_ref_corr = v_init*radiant_eci

        else:
            v_ref_corr = v_avg*radiant_eci



        ### ###
        # If the rotation correction does not have to be applied, meaning that the rotation is already
        # included, compute a version of the radiant and the velocity without Earth's rotation
        # (REPORTING PURPOSES ONLY, THESE VALUES ARE NOT USED IN THE CALCULATION)

        v_ref_nocorr = np.zeros(3)

        # Calculate the derotated reference velocity vector/radiant
        v_ref_nocorr[0] = v_ref_vect[0] + v_e*np.cos(ra_east)
        v_ref_nocorr[1] = v_ref_vect[1] + v_e*np.sin(ra_east)
        v_ref_nocorr[2] = v_ref_vect[2]

        # Compute the radiant without Earth's rotation included
        orb.ra_norot, orb.dec_norot = eci2RaDec(vectNorm(v_ref_nocorr))
        orb.azimuth_apparent_norot, orb.elevation_apparent_norot = raDec2AltAz(orb.ra_norot, orb.dec_norot, \
            jd_ref, lat_ref, lon_ref)
        orb.v_init_norot = vectMag(v_ref_nocorr)
        orb.v_avg_norot = orb.v_init_norot - v_init + v_avg

        ### ###


            

    ##########################################################################################################



    ### Correct velocity for Earth's gravity ###
    ##########################################################################################################

    # If the reference velocity is the initial velocity
    if reference_init:

        # Use the corrected velocity for Earth's rotation (when ECEF coordinates are used)
        if rotation_correction:
            v_init_corr = vectMag(v_ref_corr)

        else:
            # IMPORTANT NOTE: The correction in this case is only done on the radiant (even if the stations 
            # were fixed, but NOT on the initial velocity!). Thus, correction from Ceplecha 1987, 
            # equation (35) is not needed. If the initial velocity is determined from time vs. length and in 
            # ECI coordinates, whose coordinates rotate with the Earth, the moving stations play no role in 
            # biasing the velocity.
            v_init_corr = v_init

    else:

        if rotation_correction:

            # Calculate the corrected initial velocity if the reference velocity is the average velocity
            v_init_corr = vectMag(v_ref_corr) + v_init - v_avg
            

        else:
            v_init_corr = v_init



    # Calculate apparent RA and Dec from radiant state vector
    orb.ra, orb.dec = eci2RaDec(radiant_eci)
    orb.v_init = v_init
    orb.v_init_stddev = v_init_stddev_direct
    orb.v_avg = v_avg

    # Calculate the apparent azimuth and altitude (geodetic latitude, because ra/dec are calculated from ECI,
    #   which is calculated from WGS84 coordinates)
    orb.azimuth_apparent, orb.elevation_apparent = raDec2AltAz(orb.ra, orb.dec, jd_ref, lat_ref, lon_ref)

    orb.jd_ref = jd_ref
    orb.lon_ref = lon_ref
    orb.lat_ref = lat_ref
    orb.ht_ref = ht_ref
    orb.ht_ref_wgs84 = wmpl.Utils.GeoidHeightEGM96.mslToWGS84Height(orb.lat_ref, orb.lon_ref, orb.ht_ref)
    orb.lat_geocentric = lat_geocentric

    # Assume that the velocity in infinity is the same as the initial velocity (after rotation correction, if
    # it was needed)
    orb.v_inf = v_init_corr


    # Make sure the velocity of the meteor is larger than the escape velocity
    if v_init_corr**2 > (2*6.67408*5.9722)*1e13/vectMag(eci_ref):

        # Calculate the geocentric velocity (sqrt of squared inital velocity minus the square of the Earth escape 
        # velocity at the height of the trajectory), units are m/s.
        # Square of the escape velocity is: 2GM/r, where G is the 2014 CODATA-recommended value of 
        # 6.67408e-11 m^3/(kg s^2), and the mass of the Earth is M = 5.9722e24 kg
        v_g = np.sqrt(v_init_corr**2 - (2*6.67408*5.9722)*1e13/vectMag(eci_ref))


        # Calculate the radiant corrected for Earth's rotation (ONLY if the stations were fixed, otherwise it
        #   is the same as the apparent radiant)
        ra_corr, dec_corr = eci2RaDec(vectNorm(v_ref_corr))

        # Calculate the Local Sidreal Time of the reference trajectory position
        lst_ref = np.radians(jd2LST(jd_ref, np.degrees(lon_ref))[0])

        # Calculate the apparent zenith angle
        zc = np.arccos(np.sin(dec_corr)*np.sin(lat_geocentric) \
            + np.cos(dec_corr)*np.cos(lat_geocentric)*np.cos(lst_ref - ra_corr))

        # Calculate the zenith attraction correction
        delta_zc = 2*np.arctan2((v_init_corr - v_g)*np.tan(zc/2), v_init_corr + v_g)

        # Zenith distance of the geocentric radiant
        zg = zc + np.abs(delta_zc)

        ##########################################################################################################



        ### Calculate the geocentric radiant ###
        ##########################################################################################################

        # Get the azimuth from the corrected RA and Dec
        azimuth_corr, _ = raDec2AltAz(ra_corr, dec_corr, jd_ref, lat_geocentric, lon_ref)

        # Calculate the geocentric radiant
        ra_g, dec_g = altAz2RADec(azimuth_corr, np.pi/2 - zg, jd_ref, lat_geocentric, lon_ref)
        

        ### Precess ECI coordinates to J2000 ###

        # Convert rectangular to spherical coordiantes
        re, delta_e, alpha_e = cartesianToSpherical(*eci_ref)

        # Precess coordinates to J2000
        alpha_ej, delta_ej = equatorialCoordPrecession(jd_ref, J2000_JD.days, alpha_e, delta_e)

        # Convert coordinates back to rectangular
        eci_ref = sphericalToCartesian(re, delta_ej, alpha_ej)
        eci_ref = np.array(eci_ref)

        ######

        # Precess the geocentric radiant to J2000
        ra_g, dec_g = equatorialCoordPrecession(jd_ref, J2000_JD.days, ra_g, dec_g)


        # Compute the orbital elements
        (
            meteor_pos, L_g, B_g , met_v_h, L_h, B_h, v_h_mag, la_sun, a, eccentricity, incl, 
            peri, node, pi, b, q, Q, true_anomaly, eccentric_anomaly, mean_anomaly, last_perihelion, n, T, Tj
        ) = calcOrbitalElements(jd_ref, ra_g, dec_g, v_g, eci_ref)



        # Assign calculated parameters
        orb.lst_ref = lst_ref
        orb.jd_dyn = jd_dyn
        orb.v_g = v_g
        orb.ra_g = ra_g
        orb.dec_g = dec_g
        orb.zc = zc
        orb.zg = zg


        orb.meteor_pos = meteor_pos
        orb.L_g = L_g
        orb.B_g = B_g

        orb.v_h_x, orb.v_h_y, orb.v_h_z = met_v_h
        orb.L_h = L_h
        orb.B_h = B_h

        orb.v_h = v_h_mag

        orb.la_sun = la_sun

        orb.a = a
        orb.e = eccentricity
        orb.i = incl
        orb.peri = peri
        orb.node = node
        orb.pi = pi
        orb.b = b
        orb.q = q
        orb.Q = Q
        orb.true_anomaly = true_anomaly
        orb.eccentric_anomaly = eccentric_anomaly
        orb.mean_anomaly = mean_anomaly
        orb.last_perihelion = last_perihelion
        orb.n = n
        orb.T = T

        orb.Tj = Tj


    return orb




if __name__ == "__main__":

    from wmpl.Utils.TrajConversions import raDec2ECI

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description=""" Compute the orbit from given trajectory parameters, or recompute the orbit using the given trajectory pickle file and a few modified trajectory values.
        Usage:

        a) Recomputing an orbit using an existing trajectory, but modifying one one of the trajectory parameters, e.g. with the initial velocity of 20.5 km/s:
            python -m wmpl.Trajectory.Orbit trajectory.pickle -v 20.5

        b) Compute the orbit from scratch:
            python -m wmpl.Trajectory.Orbit -r 317.74 -d 31.72 -v 54.9 -t "20180614-072809.3" -a 44.43 -o -81.56 -e 105.8

        c) If the apparent radiant was given in J2000, use the --j2000 option.
        """,
        formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('pickle_file', type=str, nargs='?', help='Path to the trajectory pickle file.')

    arg_parser.add_argument('-r', '--ra', help='Custom right ascention of the apparent radiant (deg) in the epoch of date (use option --j2000 to use the J2000 epoch).', type=float, \
        default=None)

    arg_parser.add_argument('-d', '--dec', help='Custom declination of the apparent radiant (deg) in the epoch of date (use option --j2000 to use the J2000 epoch).', type=float, \
        default=None)

    arg_parser.add_argument('--azim', help='Azimuth (+E of due N) of the apparent radiant (deg) (optional, if equatorial coordinates are not given).', type=float, \
        default=None)

    arg_parser.add_argument('--alt', help='Altitude above the horizon of the apparent radiant (deg) (optional, if equatorial coordinates are not given).', type=float, \
        default=None)

    arg_parser.add_argument('-v', '--vinit', help='Custom initial velocity in km/s.', type=float, \
        default=None)

    arg_parser.add_argument('-w', '--vavg', help='Custom average velocity in km/s.', type=float, \
        default=None)

    arg_parser.add_argument('-t', '--time', help='Reference UTC date and time for which the relative time of the meteor is t = 0. Format: YYYYMMDD-HHMMSS.uuu', \
        type=str, default=None)

    arg_parser.add_argument('-a', '--lat', help='Latitude +N of the reference position on the trajectory (deg).', \
        type=float, default=None)

    arg_parser.add_argument('-o', '--lon', help='Longitude +E of the reference position on the trajectory (deg).', \
        type=float, default=None)

    arg_parser.add_argument('-e', '--ele', help='Height of the reference position on the trajectory (km).', \
        type=float, default=None)

    arg_parser.add_argument('--lat2', help='Latitude +N of the end position on the trajectory (deg). Optional, if the radiant is not given. ', \
        type=float, default=None)

    arg_parser.add_argument('--lon2', help='Longitude +E of the end position on the trajectory (deg). Optional, if the radiant is not given. ', \
        type=float, default=None)

    arg_parser.add_argument('--ele2', help='Height of the end position on the trajectory (km). Optional, if the radiant is not given. ', \
        type=float, default=None)

    arg_parser.add_argument('-j', '--j2000', \
        help="Give the radiant in J2000.", \
        action="store_true")

    arg_parser.add_argument('-k', '--refavg', \
        help="The average position on the trajectory is used as a reference position instead of the initial position (e.g. with MILIG). The correction for Earth's rotation will be applied.", \
        action="store_true")

    arg_parser.add_argument('-c', '--vrotcorr', \
        help="Correct the magnitude of the velocity due to the Earth's rotation.", \
        action="store_true")

    arg_parser.add_argument('-s', '--statfixed', \
        help="Shoud be used if the stations were fixed during trajectory estimation (e.g. with MILIG).", \
        action="store_true")

    arg_parser.add_argument('-m', '--milig', \
        help="MILIG input mode, i.e. the trajectory was estimated with fixed stations and reference average position on the trajectory. This replaces calling both options --refavg and --statfixed.", \
        action="store_true")


    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    ############################


    # Load the pickle file, if given
    if cml_args.pickle_file is not None:
        traj = loadPickle(*os.path.split(cml_args.pickle_file))

    else:
        traj = None



    parameter_missing_message = "To compute the orbit without the existing trajectory file, {:s} must also be provided!"


    if cml_args.vinit is not None:
        v_init = 1000*cml_args.vinit
    elif traj is not None:
        v_init = traj.orbit.v_init
    else:
        print(parameter_missing_message.format('initial velocity'))
        sys.exit()

    if cml_args.vavg is not None:
        v_avg = 1000*cml_args.vavg
    elif traj is not None:
        v_avg = traj.orbit.v_avg
    elif v_init is not None:
        v_avg = v_init
    else:
        print(parameter_missing_message.format('average velocity'))
        sys.exit()

    if cml_args.time is not None:
        dt_ref = datetime.datetime.strptime(cml_args.time, "%Y%m%d-%H%M%S.%f")
        jd_ref = datetime2JD(dt_ref)
    elif traj is not None:
        jd_ref = traj.orbit.jd_ref
    else:
        print(parameter_missing_message.format('reference time'))
        sys.exit()


    # Parse reference location
    if (cml_args.lat is None) and (cml_args.lon is None) and (cml_args.ele is None):

        # Reuse the ECI coordinates from the given trajectory file
        if traj is not None:
            eci_ref = traj.state_vect_mini

        else:
            print(parameter_missing_message.format('lat, lon, ht'))
            sys.exit()


    else:

        # Parse individual location parameters
        if cml_args.lat is not None:
            lat_ref = np.radians(cml_args.lat)
        elif traj is not None:
            lat_ref = traj.orbit.lat_ref
        else:
            print(parameter_missing_message.format('latitude'))
            sys.exit()

        if cml_args.lon is not None:
            lon_ref = np.radians(cml_args.lon)
        elif traj is not None:
            lon_ref = traj.orbit.lon_ref
        else:
            print(parameter_missing_message.format('longitude'))
            sys.exit()

        if cml_args.ele is not None:
            ht_ref = 1000*cml_args.ele
        elif traj is not None:
            ht_ref = traj.orbit.ht_ref
        else:
            print(parameter_missing_message.format('height'))
            sys.exit()


        # Compute the ECI coordinates of the reference point on the trajectory
        eci_ref = geo2Cartesian(lat_ref, lon_ref, ht_ref, jd_ref)



    # Parse the optional end location
    if (cml_args.lat2 is None) and (cml_args.lon2 is None) and (cml_args.ele2 is None):


        # If the optional end location was not given, then compute the ECI radiant from the radiant coords

        # Check if alt/az is given
        if (cml_args.azim is not None) and (cml_args.alt is not None):

            # Compute ra/dec from alt/az
            ra, dec = altAz2RADec(np.radians(cml_args.azim), np.radians(cml_args.alt), jd_ref, lat_ref, \
                lon_ref)

        # Try loading equatorial coordinates if alt/az are not given
        else:

            if cml_args.ra is not None:
                ra = np.radians(cml_args.ra)
            elif traj is not None:
                ra = traj.orbit.ra
            else:
                print(parameter_missing_message.format('RA'))
                sys.exit()

            if cml_args.dec is not None:
                dec = np.radians(cml_args.dec)
            elif traj is not None:
                dec = traj.orbit.dec
            else:
                print(parameter_missing_message.format('Dec'))
                sys.exit()

            # Precess to epoch of date if given in J2000
            if cml_args.j2000:
                ra, dec = equatorialCoordPrecession(J2000_JD.days, jd_ref, ra, dec)

        # Compute the radiant vector in ECI coordinates
        radiant_eci = np.array(raDec2ECI(ra, dec))


    else:

        # Parse individual location parameters
        if cml_args.lat2 is not None:
            lat2 = np.radians(cml_args.lat2)
        elif traj is not None:
            lat2 = traj.rend_lat
        else:
            print(parameter_missing_message.format('lat2'))
            sys.exit()

        if cml_args.lon2 is not None:
            lon2 = np.radians(cml_args.lon2)
        elif traj is not None:
            lon2 = traj.rend_lon
        else:
            print(parameter_missing_message.format('lon2'))
            sys.exit()

        if cml_args.ele2 is not None:
            ht2 = 1000*cml_args.ele2
        elif traj is not None:
            ht2 = traj.rend_ele
        else:
            print(parameter_missing_message.format('ele2'))
            sys.exit()


        ### Compute the radiant vector in ECI ###

        # Get ECI coordinates of the end location
        eci_end = geo2Cartesian(lat2, lon2, ht2, jd_ref)

        # Compute the radiant
        radiant_eci = np.array(eci_ref) - np.array(eci_end)
        radiant_eci /= vectMag(radiant_eci)

        ### ###






    # Set the right flags
    reference_init = (not cml_args.refavg) and (not cml_args.milig)
    rotation_correction = cml_args.vrotcorr or cml_args.milig #or cml_args.statfixed
    stations_fixed = cml_args.statfixed or cml_args.milig


    # # Test values
    # radiant_eci = np.array(raDec2ECI(np.radians(265.16047), np.radians(-18.84373)))
    # v_init      = 16424.81
    # v_avg       = 15768.71
    # eci_ref     =  np.array([3757410.98, -2762153.20, 4463901.73])
    # jd_ref      = 2457955.794670294970

    # Compute the orbit
    orb = calcOrbit(radiant_eci, v_init, v_avg, eci_ref, jd_ref, reference_init=reference_init, \
        rotation_correction=rotation_correction, stations_fixed=stations_fixed)

    # Print the results
    print('Ref JD:', jd_ref)
    print('ECI ref:', *eci_ref)

    print(orb)

