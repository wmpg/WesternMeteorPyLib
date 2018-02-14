from __future__ import print_function, division, absolute_import

import numpy as np

from jplephem.spk import SPK


from Config import config

from Utils.Earth import calcEarthRectangularCoordJPL
from Utils.SolarLongitude import jd2SolLonJPL
from Utils.TrajConversions import J2000_JD, J2000_OBLIQUITY, AU, SUN_MU, SUN_MASS, G, SIDEREAL_YEAR, jd2LST, \
    jd2Date, jd2DynamicalTimeJD, eci2RaDec, altAz2RADec, raDec2AltAz, raDec2Ecliptic, cartesian2Geo, \
    equatorialCoordPrecession, eclipticToRectangularVelocityVect, correctedEclipticCoord, datetime2JD
from Utils.Math import vectNorm, vectMag, rotateVector, cartesianToSpherical, sphericalToCartesian



class Orbit(object):
    """ Structure for storing the orbit solution of a meteor. """

    def __init__(self):

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

        # Referent Julian date for the trajectory. Can be the time of the first point on the trajectory or the
        # average time of the meteor
        self.jd_ref = None

        # Dynamical Julian date
        self.jd_dyn = None

        # Referent Local Sidreal Time of the referent trajectory position
        self.lst_ref = None

        # Longitude of the referent point on the trajectory (rad)
        self.lon_ref = None

        # Latitude of the referent point on the trajectory (rad)
        self.lat_ref = None

        # Geocentric latitude of the referent point on the trajectory (rad)
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

        # Tisserand's parameter with respect to Jupiter
        self.Tj = None

        # Orbital period
        self.T = None


    def __repr__(self, uncertanties=None):
        """ String to be printed out when the Orbit object is printed. """

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


        out_str =  ""
        #out_str +=  "--------------------\n"

        # Check if the orbit was calculated
        if self.ra_g is not None:
            out_str += "  JD dynamic   = {:20.12f} \n".format(self.jd_dyn)
            out_str += "  LST apparent = {:.10f} deg\n".format(np.degrees(self.lst_ref))


        out_str += "Radiant (apparent, epoch of date):\n"
        out_str += "  R.A.      = {:>9.5f}{:s} deg\n".format(np.degrees(self.ra), _uncer('{:.4f}', 'ra', 
            deg=True))
        out_str += "  Dec       = {:>+9.5f}{:s} deg\n".format(np.degrees(self.dec), _uncer('{:.4f}', 'dec', 
            deg=True))
        out_str += "  Azimuth   = {:>+9.5f}{:s} deg\n".format(np.degrees(self.azimuth_apparent), \
            _uncer('{:.4f}', 'azimuth_apparent', deg=True))
        out_str += "  Elevation = {:>+9.5f}{:s} deg\n".format(np.degrees(self.elevation_apparent), \
            _uncer('{:.4f}', 'elevation_apparent', deg=True))
        out_str += "  Vavg      = {:>9.5f}{:s} km/s\n".format(self.v_avg/1000, _uncer('{:.4f}', 'v_avg', 
            multi=1.0/1000))
        out_str += "  Vinit     = {:>9.5f}{:s} km/s\n".format(self.v_init/1000, _uncer('{:.4f}', 'v_init', 
            multi=1.0/1000))


        # Check if the orbital elements could be calculated, and write them out
        if self.ra_g is not None:

            out_str += "Radiant (geocentric, J2000):\n"
            out_str += "  R.A.   = {:>9.5f}{:s} deg\n".format(np.degrees(self.ra_g), _uncer('{:.4f}', 'ra_g', 
                deg=True))
            out_str += "  Dec    = {:>+9.5f}{:s} deg\n".format(np.degrees(self.dec_g), _uncer('{:.4f}', 'dec_g', 
                deg=True))
            out_str += "  Vg     = {:>9.5f}{:s} km/s\n".format(self.v_g/1000, _uncer('{:.4f}', 'v_g', 
                multi=1.0/1000))
            out_str += "  Vinf   = {:>9.5f}{:s} km/s\n".format(self.v_inf/1000, _uncer('{:.4f}', 'v_inf', 
                multi=1.0/1000))
            out_str += "  Zc     = {:>9.5f}{:s} deg\n".format(np.degrees(self.zc), _uncer('{:.4f}', 'zc', 
                deg=True))
            out_str += "  Zg     = {:>9.5f}{:s} deg\n".format(np.degrees(self.zg), _uncer('{:.4f}', 'zg', 
                deg=True))
            out_str += "Radiant (ecliptic geocentric, J2000):\n"
            out_str += "  Lg     = {:>9.5f}{:s} deg\n".format(np.degrees(self.L_g), _uncer('{:.4f}', 'L_g', 
                deg=True))
            out_str += "  Bg     = {:>+9.5f}{:s} deg\n".format(np.degrees(self.B_g), _uncer('{:.4f}', 'B_g', 
                deg=True))
            out_str += "  Vh     = {:>9.5f}{:s} km/s\n".format(self.v_h/1000, _uncer('{:.4f}', 'v_h', 
                multi=1/1000.0))
            out_str += "Radiant (ecliptic heliocentric, J2000):\n"
            out_str += "  Lh     = {:>9.5f}{:s} deg\n".format(np.degrees(self.L_h), _uncer('{:.4f}', 'L_h', 
                deg=True))
            out_str += "  Bh     = {:>+9.5f}{:s} deg\n".format(np.degrees(self.B_h), _uncer('{:.4f}', 'B_h', 
                deg=True))
            out_str += "  Vh_x   = {:>9.5f}{:s} km/s\n".format(self.v_h_x, _uncer('{:.4f}', 'v_h_x'))
            out_str += "  Vh_y   = {:>9.5f}{:s} km/s\n".format(self.v_h_y, _uncer('{:.4f}', 'v_h_y'))
            out_str += "  Vh_z   = {:>9.5f}{:s} km/s\n".format(self.v_h_z, _uncer('{:.4f}', 'v_h_z'))
            out_str += "Orbit:\n"
            out_str += "  La Sun = {:>10.6f}{:s} deg\n".format(np.degrees(self.la_sun), _uncer('{:.4f}', 'la_sun', 
                deg=True))
            out_str += "  a      = {:>10.6f}{:s} AU\n".format(self.a, _uncer('{:.4f}', 'a'))
            out_str += "  e      = {:>10.6f}{:s}\n".format(self.e, _uncer('{:.4f}', 'e'))
            out_str += "  i      = {:>10.6f}{:s} deg\n".format(np.degrees(self.i), _uncer('{:.4f}', 'i', 
                deg=True))
            out_str += "  peri   = {:>10.6f}{:s} deg\n".format(np.degrees(self.peri), _uncer('{:.4f}', 'peri', 
                deg=True))
            out_str += "  node   = {:>10.6f}{:s} deg\n".format(np.degrees(self.node), _uncer('{:.4f}', 'node', 
                deg=True))
            out_str += "  Pi     = {:>10.6f}{:s} deg\n".format(np.degrees(self.pi), _uncer('{:.4f}', 'pi', 
                deg=True))
            out_str += "  q      = {:>10.6f}{:s} AU\n".format(self.q, _uncer('{:.4f}', 'q'))
            out_str += "  f      = {:>10.6f}{:s} deg\n".format(np.degrees(self.true_anomaly), _uncer('{:.4f}', 
                'true_anomaly', deg=True))
            out_str += "  M      = {:>10.6f}{:s} deg\n".format(np.degrees(self.mean_anomaly), _uncer('{:.4f}', 
                'mean_anomaly', deg=True))
            out_str += "  Q      = {:>10.6f}{:s} AU\n".format(self.Q, _uncer('{:.4f}', 'Q'))
            out_str += "  n      = {:>10.6f}{:s} deg/day\n".format(np.degrees(self.n), _uncer('{:.4f}', 'n', 
                deg=True))
            out_str += "  T      = {:>10.6f}{:s} years\n".format(self.T, _uncer('{:.4f}', 'T'))
            
            if self.last_perihelion is not None:
                out_str += "  Last perihelion JD = {:.6f} ".format(datetime2JD(self.last_perihelion)) \
                    + "(" + str(self.last_perihelion) + ")" + _uncer('{:.4f}', 'last_perihelion') \
                    + " days \n"
            else:
                out_str += "  Last perihelion JD = NaN \n"

            out_str += "  Tj     = {:>10.6f}{:s}\n".format(self.Tj, _uncer('{:.4f}', 'Tj'))


        return out_str



def calcOrbit(radiant_eci, v_init, v_avg, eci_ref, jd_ref, stations_fixed=False, referent_init=True, \
    rotation_correction=False):
    """ Calculate the meteor's orbit from the given meteor trajectory. The orbit of the meteoroid is defined 
        relative to the centre of the Sun (heliocentric).

    Arguments:
        radiant_eci: [3 element ndarray] Radiant vector in ECI coordinates (meters).
        v_init: [float] Initial velocity (m/s).
        v_avg: [float] Average velocity of a meteor (m/s).
        eci_ref: [float] Referent ECI coordinates (meters, in the epoch of date) of the meteor trajectory. 
            They can be calculated with the geo2Cartesian function. Ceplecha (1987) assumes this to the the 
            average point on the trajectory, while Jennsikens et al. (2011) assume this to be the first point 
            on the trajectory as that point is not influenced by deceleration.
            NOTE: If the stations are not fixed, the referent ECI coordinates should be the ones
            of the initial point on the trajectory, NOT of the average point!
        jd_ref: [float] Referent Julian date of the meteor trajectory. Ceplecha (1987) takes this as the 
            average time of the trajectory, while Jenniskens et al. (2011) take this as the the first point
            on the trajectory.
    
    Keyword arguments:
        stations_fixed: [bool] If True, the correction for Earth's rotation will be performed on the radiant,
            but not the velocity. This should be True ONLY in two occasions:
                - if the ECEF coordinate system was used for trajectory estimiation
                - if the ECI coordinate system was used for trajectory estimation, BUT the stations were not
                    moved in time, but were kept fixed at one point, regardless of the trajectory estimation
                    method.
            It is necessary to perform this correction for the intersecting planes method, but not for
            the lines of sight method ONLY when the stations are not fixed. Of course, if one is using the 
            lines of sight method with fixed stations, one should perform this correction!
        referent_init: [bool] If True (default), the initial point on the trajectory is given as the referent
            one, i.e. the referent ECI coordinates are the ECI coordinates of the initial point on the
            trajectory, where the meteor has the velocity v_init. If False, then the referent point is the
            average point on the trajectory, and the average velocity will be used to do the corrections.
        rotation_correction: [bool] If True, the correction of the initial velocity for Earth's rotation will
            be performed. False by default. This should ONLY be True if the coordiante system for trajectory
            estimation was ECEF, i.e. did not rotate with the Earth. In all other cases it should be False, 
            even if fixed station coordinates were used in the ECI coordinate system!

    Return:
        orb: [Orbit object] Object containing the calculated orbit.

    """


    ### Correct the velocity vector for the Earth's rotation if the stations are fixed ###
    ##########################################################################################################

    eci_x, eci_y, eci_z = eci_ref

    # Calculate the geocentric latitude (latitude which considers the Earth as an elipsoid) of the referent 
    # trajectory point
    lat_geocentric = np.arctan2(eci_z, np.sqrt(eci_x**2 + eci_y**2))


    # Calculate the dynamical JD
    jd_dyn = jd2DynamicalTimeJD(jd_ref)

    # Calculate the geographical coordinates of the referent trajectory ECI position
    lat_ref, lon_ref, _ = cartesian2Geo(jd_ref, *eci_ref)


    # Apply the Earth rotation correction if the station ECI coordinates are fixed (a MUST for the 
    # intersecting planes method!)
    if stations_fixed:

        # Calculate the velocity of the Earth rotation at the position of the referent trajectory point (m/s)
        v_e = 2*np.pi*vectMag(eci_ref)*np.cos(lat_geocentric)/86164.09053

        
        # Calculate the equatorial coordinates of east from the referent position on the trajectory
        azimuth_east = np.pi/2
        altitude_east = 0
        ra_east, dec_east = altAz2RADec(azimuth_east, altitude_east, jd_ref, lat_ref, lon_ref)


        if referent_init:

            # If the initial velocity was the referent velocity, use it for the correction
            v_ref_vect = v_init*radiant_eci


        else:
            # Calculate referent velocity vector using the average point on the trajectory and the average
            # velocity
            v_ref_vect = v_avg*radiant_eci


        v_ref_corr = np.zeros(3)

        # Calculate the corrected referent velocity
        v_ref_corr[0] = v_ref_vect[0] - v_e*np.cos(ra_east)
        v_ref_corr[1] = v_ref_vect[1] - v_e*np.sin(ra_east)
        v_ref_corr[2] = v_ref_vect[2]



    else:

        # MOVING STATIONS
        # Velocity vector will remain unchanged if the stations were moving
        if referent_init:
            v_ref_corr = v_init*radiant_eci

        else:
            v_ref_corr = v_avg*radiant_eci

            

    ##########################################################################################################



    ### Correct velocity for Earth's gravity ###
    ##########################################################################################################

    # If the referent velocity is the initial velocity
    if referent_init:

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

            # Calculate the corrected initial velocity if the referent velocity is the average velocity
            v_init_corr = vectMag(v_ref_corr) + v_init - v_avg
            

        else:
            v_init_corr = v_init



    # Initialize a new orbit structure and assign calculated parameters
    orb = Orbit()

    # Calculate apparent RA and Dec from radiant state vector
    orb.ra, orb.dec = eci2RaDec(radiant_eci)
    orb.v_init = v_init
    orb.v_avg = v_avg

    # Calculate the apparent azimuth and altitude (geodetic latitude, because ra/dec are calculated from ECI,
    #   which is calculated from WGS84 coordinates)
    orb.azimuth_apparent, orb.elevation_apparent = raDec2AltAz(orb.ra, orb.dec, jd_ref, lat_geocentric, \
        lon_ref)

    orb.jd_ref = jd_ref
    orb.lon_ref = lon_ref
    orb.lat_ref = lat_ref
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

        # Calculate the Local Sidreal Time of the referent trajectory position
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
        earth_pos_eq = rotateVector(earth_pos, np.array([1, 0, 0]), J2000_OBLIQUITY)

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
        meteor_pos = rotateVector(meteor_pos, np.array([1, 0, 0]), -J2000_OBLIQUITY)

        # print('Meteor position:')
        # print(meteor_pos)


        ##########################################################################################################

        # Calculate components of the heliocentric velocity of the meteor (km/s)
        v_h = np.array(earth_vel) + np.array(eclipticToRectangularVelocityVect(L_g, B_g, v_g/1000))

        # Calculate the heliocentric velocity in km/s
        v_h_mag = vectMag(v_h)


        # Calculate the corrected heliocentric ecliptic coordinates of the meteoroid using the method of 
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
        T = 2*np.pi*np.sqrt(((a*AU)**3)/SUN_MU)/(86400*SIDEREAL_YEAR)


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


        ### Calculate true anomaly
        true_anomaly = np.arccos(np.dot(e_vect, meteor_pos)/(vectMag(e_vect)*vectMag(meteor_pos)))
        if np.dot(meteor_pos, v_h) < 0:
            true_anomaly = 2*np.pi - true_anomaly

        true_anomaly = true_anomaly%(2*np.pi)

        ###


        # Calculate eccentric anomaly
        eccentric_anomaly = np.arctan2(np.sqrt(1 - eccentricity**2)*np.sin(true_anomaly), eccentricity \
            + np.cos(true_anomaly))

        # Calculate mean anomaly
        mean_anomaly = eccentric_anomaly - eccentricity*np.sin(eccentric_anomaly)
        mean_anomaly = mean_anomaly%(2*np.pi)

        # Calculate the time in days since the last perihelion passage of the meteoroid
        dt_perihelion = (mean_anomaly*a**(3.0/2))/0.01720209895


        if not np.isnan(dt_perihelion):
            
            # Calculate the date and time of the last perihelion passage
            last_perihelion = jd2Date(jd_dyn - dt_perihelion, dt_obj=True)

        else:
            last_perihelion = None


        # Calculate Tisserand's parameter with respect to Jupiter
        Tj = 2*np.sqrt((1 - eccentricity**2)*a/5.204267)*np.cos(incl) + 5.204267/a



        # Assign calculated parameters
        orb.lst_ref = lst_ref
        orb.jd_dyn = jd_dyn
        orb.v_g = v_g
        orb.ra_g = ra_g
        orb.dec_g = dec_g

        orb.meteor_pos = meteor_pos
        orb.L_g = L_g
        orb.B_g = B_g

        orb.v_h_x, orb.v_h_y, orb.v_h_z = met_v_h
        orb.L_h = L_h
        orb.B_h = B_h

        orb.zc = zc
        orb.zg = zg

        orb.v_h = v_h_mag*1000

        orb.la_sun = la_sun

        orb.a = a
        orb.e = eccentricity
        orb.i = incl
        orb.peri = peri
        orb.node = node
        orb.pi = pi
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

    from Utils.TrajConversions import raDec2ECI

    # Calculate an orbit as a test
    radiant_eci = np.array(raDec2ECI(np.radians(265.16047), np.radians(-18.84373)))
    v_init      = 16424.81
    v_avg       = 15768.71
    eci_ref     =  np.array([3757410.98, -2762153.20, 4463901.73])
    jd_ref      = 2457955.794670294970


    orb = calcOrbit(radiant_eci, v_init, v_avg, eci_ref, jd_ref)

    print(orb)







    