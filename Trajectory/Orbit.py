from __future__ import print_function, division, absolute_import

import numpy as np

from jplephem.spk import SPK


from Config import config

from Utils.Earth import calcEarthRectangularCoordJPL
from Utils.SolarLongitude import jd2SolLonJPL
from Utils.TrajConversions import J2000_JD, J2000_OBLIQUITY, AU, SUN_MU, SUN_MASS, G, jd2LST, jd2Date, \
    eci2RaDec, altAz2RADec, raDec2AltAz, raDec2Ecliptic, cartesian2Geo, equatorialCoordPrecession
from Utils.Math import vectNorm, vectMag, rotateVector



class Orbit(object):
    """ Structure for storing the orbit solution of a meteor. """

    def __init__(self):

        # Apparent radiant position (radians)
        self.ra = None
        self.dec = None

        # Estimated average velocity
        self.v_avg = None

        # Estimated initial velocity
        self.v_init = None

        # Julian date of the average point on the trajectory
        self.jd_avg = None

        # Longitude of the average point on the trajectory (rad)
        self.lon_avg = None

        # Latitude of the average point on the trajectory (rad)
        self.lat_avg = None

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

        # Apparent zenith angle (before the correction for Earth's gravity)
        self.zc = None

        # Zenith distance of the geocentric radiant (after the correction for Earth's gravity)
        self.zg = None

        # Helioventric velocity of the meteor (m/s)
        self.v_h = None

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
        out_str += "Radiant (apparent):\n"
        out_str += "  R.A.   = {:>9.5f}{:s} deg\n".format(np.degrees(self.ra), _uncer('{:.4f}', 'ra', 
            deg=True))
        out_str += "  Dec    = {:>+9.5f}{:s} deg\n".format(np.degrees(self.dec), _uncer('{:.4f}', 'dec', 
            deg=True))
        out_str += "  Vavg   = {:>9.5f}{:s} km/s\n".format(self.v_avg/1000, _uncer('{:.4f}', 'v_avg', 
            multi=1.0/1000))
        out_str += "  Vinit  = {:>9.5f}{:s} km/s\n".format(self.v_init/1000, _uncer('{:.4f}', 'v_init', 
            multi=1.0/1000))
        out_str += "Radiant (geocentric):\n"
        out_str += "  R.A.   = {:>9.5f}{:s} deg\n".format(np.degrees(self.ra_g), _uncer('{:.4f}', 'ra_g', 
            deg=True))
        out_str += "  Dec    = {:>+9.5f}{:s} deg\n".format(np.degrees(self.dec_g), _uncer('{:.4f}', 'dec_g', 
            deg=True))
        out_str += "  Vg     = {:>9.5f}{:s} km/s\n".format(self.v_g/1000, _uncer('{:.4f}', 'v_g', 
            multi=1.0/1000))
        out_str += "  Vinf   = {:>9.5f}{:s} km/s\n".format(self.v_inf/1000, _uncer('{:.4f}', 'v_inf', 
            multi=1.0/1000))
        out_str += "  Zg     = {:>9.5f}{:s} deg\n".format(np.degrees(self.zg), _uncer('{:.4f}', 'zg', 
            deg=True))
        out_str += "Radiant (ecliptic):\n"
        out_str += "  L      = {:>9.5f}{:s} deg\n".format(np.degrees(self.L_g), _uncer('{:.4f}', 'L_g', 
            deg=True))
        out_str += "  B      = {:>+9.5f}{:s} deg\n".format(np.degrees(self.B_g), _uncer('{:.4f}', 'B_g', 
            deg=True))
        out_str += "  Vh     = {:>9.5f}{:s} km/s\n".format(self.v_h/1000, _uncer('{:.4f}', 'v_h', 
            multi=1/1000.0))
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
        out_str += "  Last perihelion: " + str(self.last_perihelion) + _uncer('{:.4f}', 'last_perihelion') \
            + " days \n"
        out_str += "  Tj     = {:>10.6f}{:s}\n".format(self.Tj, _uncer('{:.4f}', 'Tj'))


        return out_str



def calcOrbit(radiant_eci, v_init, v_avg, eci_avg, jd_avg):
    """ Calculate the meteor's orbit from the given meteor trajectory. The orbit of the meteoroid is defined 
        relative to the barycentre of the Solar system.

    Arguments:
        radiant_eci: [3 element ndarray] radiant vector in ECI coordinates (meters)
        v_init: [float] initial velocity (m/s)
        v_avg: [float] average velocity of a meteor (m/s)
        eci_avg: [float] average ECI coordinates of the meteor trajectory (meters)
        jd_avg: [float] average Julian date of the meteor trajectory
    
    Return:
        orb: [Orbit object] object containing the calculated orbit

    """

    ### Correct the velocity vector for the Earth's rotation

    eci_x, eci_y, eci_z = eci_avg

    # Calculate the geocentric latitude (latitude which considers the Earth as an elipsoid) of the average 
    # trajectory point
    lat_geocentric = np.arctan2(eci_z, np.sqrt(eci_x**2 + eci_y**2))

    # Calculate the velocity of the Earth rotation at the position of the average trajectory (m/s)
    v_e = 2*np.pi*vectMag(eci_avg)*np.cos(lat_geocentric)/86164.09053


    # Calculate the geographical coordinates of the average trajectory ECI position
    lat_avg, lon_avg, _ = cartesian2Geo(jd_avg, *eci_avg)
    
    # Calculate the equatorial coordinates of east from the average position on the trajectory
    azimuth_east = np.pi/2
    altitude_east = 0
    ra_east, dec_east = altAz2RADec(azimuth_east, altitude_east, jd_avg, lat_avg, lon_avg)


    # Calculate average velocity vector
    v_avg_vect = v_avg*radiant_eci


    v_avg_corr = np.zeros(3)

    # Calculate the corrected average velocity
    v_avg_corr[0] = v_avg_vect[0] - v_e*np.cos(ra_east)
    v_avg_corr[1] = v_avg_vect[1] - v_e*np.sin(ra_east)
    v_avg_corr[2] = v_avg_vect[2]

    ###



    ### Correct for Earth's gravity

    # Calculate the corrected initial velocity
    v_init_corr = vectMag(v_avg_corr) + v_init - v_avg

    # Calculate the geocentric velocity (sqrt of squared inital velocity minus the square of the Earth escape 
    # velocity at the height of the trajectory), units are m/s.
    # Square of the escape velocity is: 2GM/r, where G is the 2014 CODATA-recommended value of 
    # 6.67408e-11 m^3/(kg s^2), and the mass of the Earth is M = 5.9722e24 kg
    v_g = np.sqrt(v_init_corr**2 - (2*6.67408*5.9722)*1e13/vectMag(eci_avg))


    # Calculate the radiant corrected for Earth's rotation
    ra_corr, dec_corr = eci2RaDec(vectNorm(v_avg_corr))

    # Calculate the Local Sidreal Time of the average trajectory position
    lst_avg = np.radians(jd2LST(jd_avg, np.degrees(lon_avg))[0])

    # Calculate the apparent zenith angle
    zc = np.arccos(np.sin(dec_corr)*np.sin(lat_geocentric) \
        + np.cos(dec_corr)*np.cos(lat_geocentric)*np.cos(lst_avg - ra_corr))

    # Calculate the zenith attraction correction
    delta_zc = 2*np.arctan2((v_init_corr - v_g)*np.tan(zc/2), v_init_corr + v_g)

    # Zenith distance of the geocentric radiant
    zg = zc + np.abs(delta_zc)

    ###



    ### Calculate the geocentric radiant

    # Get the azimuth from the corrected RA and Dec
    azimuth_corr, _ = raDec2AltAz(ra_corr, dec_corr, jd_avg, lat_avg, lon_avg)

    # Calculate the geocentric radiant
    ra_g, dec_g = altAz2RADec(azimuth_corr, np.pi/2 - zg, jd_avg, lat_geocentric, lon_avg)

    # Precess the geocentric radiant to J2000
    ra_g, dec_g = equatorialCoordPrecession(jd_avg, J2000_JD.days, ra_g, dec_g)


    # Calculate the ecliptic latitude and longitude of the geocentric radiant (J2000 epoch)
    L_g, B_g = raDec2Ecliptic(jd_avg, ra_g, dec_g)


    # Load the JPL ephemerids data
    jpl_ephem_data = SPK.open(config.jpl_ephem_file)
    
    # Get the position of the Earth (km) and its velocity (km/s) at the given Julian date (J2000 epoch)
    # The position is given in the ecliptic coordinates, origin of the coordinate system is in the Solar 
    # system barycentre
    earth_pos, earth_vel = calcEarthRectangularCoordJPL(jd_avg, jpl_ephem_data)

    # Convert the Earth's position to rectangular equatorial coordinates (FK5)
    earth_pos_eq = rotateVector(earth_pos, np.array([1, 0, 0]), J2000_OBLIQUITY)

    # Add the average position of the meteor's trajectory to the position of the Earth to calculate the 
    # equatorial coordinates of the meteor (in kilometers)
    meteor_pos = earth_pos_eq + eci_avg/1000

    # Convert the position of the average trajectory from FK5 to Sun-centered ecliptic coordinates
    meteor_pos = rotateVector(meteor_pos, np.array([1, 0, 0]), -J2000_OBLIQUITY)


    # Convert the meteor's velocity to km/s
    v_g = v_g/1000.0

    # Calculate components of the heliocentric velocity of the meteor
    v_h = np.zeros(3)
    v_h[0] = earth_vel[0] - v_g*np.cos(L_g)*np.cos(B_g)
    v_h[1] = earth_vel[1] - v_g*np.sin(L_g)*np.cos(B_g)
    v_h[2] = earth_vel[2] - v_g*np.sin(B_g)

    # Calculate the heliocentric velocity in km/s
    v_h_mag = vectMag(v_h)


    # Calculate the solar longitude
    #la_sun = np.arctan2(earth_pos[1], earth_pos[0]) + np.pi
    #la_sun = la_sun%(2*np.pi)
    la_sun = jd2SolLonJPL(jd_avg)


    ### Calculations done using Dave Clark's Master thesis equations

    # Specific orbital energy
    epsilon = (vectMag(v_h)**2)/2 - SUN_MU/vectMag(meteor_pos)

    # Semi-major axis in AU
    a = -SUN_MU/(2*epsilon*AU)

    # Calculate mean motion in rad/day
    n = np.sqrt(G*SUN_MASS/((np.abs(a)*AU*1000.0)**3))*86400.0


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


    # Calculate true anomaly
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

    # Calculate the date and time of the last perihelion passage
    last_perihelion = jd2Date(jd_avg - dt_perihelion, dt_obj=True)


    # Calculate Tisserand's parameter with respect to Jupiter
    Tj = 2*np.sqrt((1 - eccentricity**2)*a/5.204267)*np.cos(incl) + 5.204267/a


    ### Initialize a new orbit structure and assign calculated parameters
    orb = Orbit()

    orb.jd_avg = jd_avg
    orb.lon_avg = lon_avg
    orb.lat_avg = lat_avg

    # Set orbit parameters to the orbit object
    orb.v_inf = v_init_corr
    orb.v_g = v_g*1000
    orb.ra_g = ra_g
    orb.dec_g = dec_g

    orb.meteor_pos = meteor_pos
    orb.L_g = L_g
    orb.B_g = B_g

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

    orb.Tj = Tj

    ###


    return orb













    