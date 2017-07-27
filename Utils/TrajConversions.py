""" 
A set of tools of working with meteor data. 

Includes:
    - Julian date conversion
    - LST calculation
    - Coordinate transformations
    - RA and Dec precession correction
    - ...
    
"""

# The MIT License

# Copyright (c) 2017 Denis Vida

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from __future__ import print_function, division, absolute_import

import math
import numpy as np
from datetime import datetime, timedelta, MINYEAR

from Utils.Math import vectNorm, vectMag, rotateVector


### CONSTANTS ###

# Astronomical units in kilometers
AU = 149597870.7

# Gravitational constant of the Sun in km^3/s^2
SUN_MU = 1.32712440018e11

# Mass of the Sun in kilograms
SUN_MASS = 1.98855e+30

# Gravitational constant in m^3/kg/s^2
G = 6.67384e-11

# Obliquity of the Earth at J2000.0 epoch
J2000_OBLIQUITY = np.radians(23.4392911111)

# Define Julian epoch
JULIAN_EPOCH = datetime(2000, 1, 1, 12) # J2000.0 noon
J2000_JD = timedelta(2451545) # J2000.0 epoch in julian days

class EARTH_CONSTANTS(object):
    """ Holds Earth's shape parameters. """

    def __init__(self):

        # Earth elipsoid parameters in meters (source: WGS84, the GPS standard)
        self.EQUATORIAL_RADIUS = 6378137.0
        self.POLAR_RADIUS = 6356752.314245
        self.E = math.sqrt(1.0 - self.POLAR_RADIUS**2/self.EQUATORIAL_RADIUS**2)
        self.RATIO = self.EQUATORIAL_RADIUS/self.POLAR_RADIUS
        self.SQR_DIFF = self.EQUATORIAL_RADIUS**2 - self.POLAR_RADIUS**2

# Initialize Earth shape constants object
EARTH = EARTH_CONSTANTS()


#################


### DECORATORS ###

def floatArguments(func):
    """ A decorator that converts all function arguments to float. 
    
    @param func: a function to be decorated

    @return :[funtion object] the decorated function
    """

    def inner_func(*args):
        args = map(float, args)
        return func(*args)

    return inner_func


##################


### Time transformations ###


def unixTime2Date(ts, tu, dt_obj=False):
    """ Convert UNIX time given in ts and tu to date and time. 
    
    Arguments:
        ts: [int] UNIX time, seconds part
        tu: [int] UNIX time, microsecond part

    Kwargs:
        dt_obj: [bool] default False, function returns a datetime object if True

    Return:
        if dt_obj == False (default): [tuple] (year, month, day, hours, minutes, seconds, milliseconds)
        else: [datetime object]

    """

    # Convert the UNIX timestamp to datetime object
    dt = datetime.utcfromtimestamp(float(ts) + float(tu)/1000000)


    if dt_obj:
        return dt

    else:
        return dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, float(tu)/1000



def date2JD(year, month, day, hour, minute, second, millisecond=0, UT_corr=0.0):
    """ Convert date and time to Julian Date with epoch J2000.0. 
    
    Arguments:
        year: [int] year
        month: [int] month
        day: [int] day of the date
        hour: [int] hours
        minute: [int] minutes
        second: [int] seconds

    Kwargs:
        millisecond: [int] milliseconds (optional)
        UT_corr: [float] UT correction in hours (difference from local time to UT)
    
    Return:
        [float] julian date, epoch 2000.0
    """

    # Convert all input arguments to integer (except milliseconds)
    year, month, day, hour, minute, second = map(int, (year, month, day, hour, minute, second))

    # Create datetime object of current time
    dt = datetime(year, month, day, hour, minute, second, int(millisecond*1000))

    # Calculate Julian date
    julian = dt - JULIAN_EPOCH + J2000_JD - timedelta(hours=UT_corr)
    
    # Convert seconds to day fractions
    return julian.days + (julian.seconds + julian.microseconds/1000000.0)/86400.0



def datetime2JD(dt):
    """ Converts a datetime object to Julian date. 

    Arguments:
        dt: [datetime object]

    Return:
        jd: [float] Julian date
    """

    return date2JD(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond/1000.0)



def jd2Date(jd, UT_corr=0, dt_obj=False):
    """ Converts the given Julian date to (year, month, day, hour, minute, second, millisecond) tuple. 

    Arguments:
        jd: [float] Julian date

    Keyword arguments:
        UT_corr: [float] UT correction in hours (difference from local time to UT)
        dt_obj: [bool] returns a datetime object if True. False by default.

    Return:
        (year, month, day, hour, minute, second, millisecond)

    """

    
    dt = timedelta(days=jd)

    try:
        date = dt + JULIAN_EPOCH - J2000_JD + timedelta(hours=UT_corr) 

    # If the date is out of range (i.e. before year 1) use year 1. This is the limitation in the datetime
    # library. Time handling should be switched to astropy.time
    except OverflowError:
        date = datetime(MINYEAR, 1, 1, 0, 0, 0)


    # Return a datetime object if dt_obj == True
    if dt_obj:
        return date

    return date.year, date.month, date.day, date.hour, date.minute, date.second, date.microsecond/1000.0



def unixTime2JD(ts, tu):
    """ Converts UNIX time to Julian date. 
    
    Arguments:
        ts: [int] UNIX time, seconds part
        tu: [int] UNIX time, microsecond part

    Return:
        [float] julian date, epoch 2000.0

    """

    return date2JD(*unixTime2Date(ts, tu))



def jd2LST(julian_date, lon):
    """ Convert Julian date to Local Sidereal Time and Greenwich Sidereal Time. 

    Source: J. Meeus: Astronomical Algorithms

    Arguments:
        julian_date: [float] decimal julian date, epoch J2000.0
        lon: [float] longitude of the observer in degrees
    
    Return:
        (LST, GST): [tuple of floats] a tuple of Local Sidereal Time and Greenwich Sidereal Time
    """

    t = (julian_date - J2000_JD.days)/36525.0

    # Greenwich Sidereal Time
    GST = 280.46061837 + 360.98564736629*(julian_date - J2000_JD.days) + 0.000387933*t**2 - (t**3)/38710000
    GST = (GST + 360)%360

    # Local Sidereal Time
    LST = (GST + lon + 360)%360
    
    return LST, GST



def LST2LongitudeEast(julian_date, LST):
    """ Convert Julian date and Local Sidereal Time to east longitude. 
    
    Arguments:
        julian_date: [float] decimal julian date, epoch J2000.0
        LST: [float] Local Sidereal Time in degrees

    Return:
        lon: [float] longitude of the observer in degrees
    """

    t = (julian_date - J2000_JD.days)/36525.0

    # Greenwich Sidereal Time
    GST = 280.46061837 + 360.98564736629*(julian_date - J2000_JD.days) + 0.000387933*t**2 - (t**3)/38710000
    GST = (GST + 360)%360

    # Calculate longitude
    lon = (LST - GST + 180)%360 - 180

    return lon, GST



def GST2JD(gst):
    """ Convert Greenwich Sidereal Time to Julian date. 
    
    Arguments:
        gst: [float] Greenwich Sidereal Time in degrees

    Return:
        jd: [float] Julian date

    """




############################



def ecef2ENU(phi, lam, x, y, z):
    """ Convert Earth centered - Earth fixed (ECEF) Cartesian coordinates to ENU coordinates (East, North, 
        Up).

        See 'enu2ECEF' function for more details.

    """

    x_enu =             -np.sin(lam)*x +             np.cos(lam)*y
    y_enu = -np.sin(phi)*np.cos(lam)*x - np.sin(phi)*np.sin(lam)*y + np.cos(phi)*z
    z_enu =  np.cos(phi)*np.cos(lam)*x + np.cos(phi)*np.sin(lam)*y + np.sin(phi)*z

    return x_enu, y_enu, z_enu




def enu2ECEF(phi, lam, x, y, z, t=0.0):
    """ Convert ENU local coordinates (East, North, Up) to Earth centered - Earth fixed (ECEF) Cartesian, 
        correcting for Earth rotation if needed.
        
        ENU coordinates can be transformed to ECEF by two rotations:
        1. A clockwise rotation over east-axis by an angle (90 - phi) to align the up-axis with the z-axis.
        2. A clockwise rotation over the z-axis by and angle (90 + lam) to align the east-axis with the x-axis.

        Source: http://www.navipedia.net/index.php/Transformations_between_ECEF_and_ENU_coordinates

    Arguments: 
        phi: [float] east-axis rotation angle
        lam: [float] z-axis rotation angle
        x: [float] ENU x coordinate
        y: [float] ENU y coordinate
        z: [float] ENU z coordinate

    Keyword arguments:
        t: [float] time in seconds, 0 by default

    Return:
        (x_ecef, y_ecef, z_ecef): [tuple of floats] ECEF coordinates

    """

    # Calculate ECEF coordinate from given local coordinates
    x_ecef = -np.sin(lam)*x - np.sin(phi)*np.cos(lam)*y + np.cos(phi)*np.cos(lam)*z
    y_ecef =  np.cos(lam)*x - np.sin(phi)*np.sin(lam)*y + np.cos(phi)*np.sin(lam)*z
    z_ecef =                  np.cos(phi)            *y + np.sin(phi)            *z

    # Calculate time correction (in radians)
    tau = 2*np.pi/(23.0*3600.0 + 56.0*60.0 + 4.09054) # Earth rotation in rad/s
    yaw = -tau*t

    x_temp = x_ecef
    y_temp = y_ecef

    # Apply time correction
    x_ecef =  np.cos(yaw)*x_temp + np.sin(yaw)*y_temp
    y_ecef = -np.sin(yaw)*x_temp + np.cos(yaw)*y_temp

    return x_ecef, y_ecef, z_ecef




def latLonAlt2ECEF(lat, lon, h):
    """ Convert geographical coordinates to Earth centered - Earth fixed coordinates.

    Arguments:
        lat: [float] latitude in radians (+north)
        lon: [float] longitude in radians (+east)
        h: [float] elevation in meters

    Return:
        (x, y, z): [tuple of floats] ECEF coordinates

    """

    # Get distance from Earth centre to the position given by geographical coordinates, in WGS84
    N = EARTH.EQUATORIAL_RADIUS/math.sqrt(1.0 - (EARTH.E**2)*math.sin(lat)**2)

    # Calculate ECEF coordinates
    ecef_x = (N + h)*math.cos(lat)*math.cos(lon)
    ecef_y = (N + h)*math.cos(lat)*math.sin(lon)
    ecef_z = ((1 - EARTH.E**2)*N + h)*math.sin(lat)

    return ecef_x, ecef_y, ecef_z



### Spatial coordinates transformations ###

@floatArguments
def geo2Cartesian(lat_rad, lon_rad, h, julian_date):
    """ Convert geographical Earth coordinates to Cartesian ECI coordinate system (Earth center as origin).
        The Earth is considered as an elipsoid.

    @param lat_rad: [float] latitude of the observer in radians
    @param lon_rad: [float] longitde of the observer in radians
    @param h: [int or float] elevation of the observer in meters
    @param julian_date: [float] decimal julian date, epoch J2000.0

    @return (x, y, z): [tuple of floats] a tuple of X, Y, Z Cartesian ECI coordinates
    """

    lon = np.degrees(lon_rad)

    # Get Local Sidereal Time
    LST_rad = np.radians(jd2LST(julian_date, lon)[0])

    ecef_x, ecef_y, ecef_z = latLonAlt2ECEF(lat_rad, lon_rad, h)

    # Calculate the Earth radius at given latitude
    Rh = math.sqrt(ecef_x**2 + ecef_y**2 + ecef_z**2)

    # Calculate the geocentric latitude (latitude which considers the Earth as an elipsoid)
    lat_geocentric = math.atan2(ecef_z, math.sqrt(ecef_x**2 + ecef_y**2))

    # Calculate Cartesian ECI coordinates (in meters)
    x = Rh*np.cos(lat_geocentric)*np.cos(LST_rad)
    y = Rh*np.cos(lat_geocentric)*np.sin(LST_rad)
    z = Rh*np.sin(lat_geocentric)

    return x, y, z


# Vectorize the geo2Cartesian function, so julian_date can be given as a numpy array
geo2Cartesian_vect = np.vectorize(geo2Cartesian, excluded=['lat_rad', 'lon_rad', 'h'])



def ecef2LatLonAlt(x, y, z):
    """ Convert Earth centered - Earth fixed coordinates to geographical coordinates (latitude, longitude, 
        elevation).

    Arguments:
        x: [float] ECEF x coordinate
        y: [float] ECEF y coordinate
        z: [float] ECEF z coordinate

    Return:
        (lat, lon, alt): [tuple of floats] latitude and longitude in radians, elevation in meters

    """

    # Calculate the polar eccentricity
    ep = np.sqrt((EARTH.EQUATORIAL_RADIUS**2 - EARTH.POLAR_RADIUS**2)/(EARTH.POLAR_RADIUS**2))

    # Calculate the longitude
    lon = np.arctan2(y, x)

    p = np.sqrt(x**2  +  y**2);

    theta = np.arctan2( z*EARTH.EQUATORIAL_RADIUS, p*EARTH.POLAR_RADIUS)

    # Calculate the latitude
    lat = np.arctan2(z + (ep**2)*EARTH.POLAR_RADIUS*np.sin(theta)**3, 
        p - (EARTH.E**2)*EARTH.EQUATORIAL_RADIUS*np.cos(theta)**3)

    # Get distance from Earth centre to the position given by geographical coordinates, in WGS84
    N = EARTH.EQUATORIAL_RADIUS/math.sqrt(1.0 - (EARTH.E**2)*math.sin(lat)**2)

    # Calculate the height in meters
    alt = p/np.cos(lat) - N

    # Handle the case when in the Southern hemisphere
    if( np.abs(x) < 1  and  np.abs(y) < 1):
        alt = np.abs(z) - EARTH.POLAR_RADIUS

    return lat, lon, alt



def cartesian2Geo(julian_date, x, y, z):
    """ Convert Cartesian ECI coordinates of a point (origin in Earth's centre) to geographical coordinates.
    
    Arguments:
        julian_date: [float] decimal julian date
        X: [float] X coordinate of a point in space (meters)
        Y: [float] Y coordinate of a point in space (meters)
        Z: [float] Z coordinate of a point in space (meters)
    
    Return:
        (lon, lat, ele): [tuple of floats]
            lon: longitude of the point in radians
            lat: latitude of the point in radians
            ele: elevation in meters
    """


    # Calculate LLA
    lat, r_LST, ele = ecef2LatLonAlt(x, y, z)

    # Calculate proper longitude from the given JD
    lon, _ = LST2LongitudeEast(julian_date, np.degrees(r_LST))

    # Convert longitude to radians
    lon = np.radians(lon)

    # # Get LST and GST
    # LST, GST = jd2LST(julian_date, lon)

    # # Convert Cartesian coordinates to latitude and longitude
    # lon_p = math.degrees(math.atan2(Yi, Xi) - math.radians(GST))
    # lat_p = math.degrees(math.atan2(math.sqrt(Xi**2 + Yi**2), Zi))

    return lat, lon, ele




def altAz2RADec(azim, elev, jd, lat, lon):
    """ Convert azimuth and altitude in a given time and position on Earth to right ascension and 
        declination. 

    Arguments:
        azim: [float] azimuth (+east of due north) in radians
        elev: [float] elevation above horizon in radians
        jd: [float] Julian date
        lat: [float] latitude of the observer in radians
        lon: [float] longitde of the observer in radians

    Return:
        (RA, dec): [tuple]
            RA: [float] right ascension (radians)
            dec: [float] declination (radians)
    """
    
    # Calculate hour angle
    ha = np.arctan2(-np.sin(azim), np.tan(elev)*np.cos(lat) - np.cos(azim)*np.sin(lat))

    # Calculate Local Sidereal Time
    lst = np.radians(jd2LST(jd, np.degrees(lon))[0])
    
    # Calculate right ascension
    ra = (lst - ha)%(2*np.pi)

    # Calculate declination
    dec = np.arcsin(np.sin(lat)*np.sin(elev) + np.cos(lat)*np.cos(elev)*np.cos(azim))

    return ra, dec

# Vectorize the altAz2RADec function so it can take numpy arrays for: azim, elev, jd
altAz2RADec_vect = np.vectorize(altAz2RADec, excluded=['lat', 'lon'])



def raDec2AltAz(ra, dec, jd, lat, lon):
    """ Convert right ascension and declination to azimuth (+east of sue north) and altitude. 

    Arguments:
        ra: [float] right ascension in radians
        dec: [float] declination in radians
        jd: [float] Julian date
        lat: [float] latitude in radians
        lon: [float] longitude in radians

    Return:
        (azim, elev): [tuple]
            azim: [float] azimuth (+east of due north) in radians
            elev: [float] elevation above horizon in radians

        """

    # Calculate Local Sidereal Time
    lst = np.radians(jd2LST(jd, np.degrees(lon))[0])

    # Calculate the hour angle
    ha = lst - ra

    # Constrain the hour angle to [-pi, pi] range
    ha = (ha + np.pi)%(2*np.pi) - np.pi

    # Calculate the azimuth
    azim = np.pi + np.arctan2(np.sin(ha), np.cos(ha)*np.sin(lat) - np.tan(dec)*np.cos(lat))

    # Calculate the sine of elevation
    sin_elev = np.sin(lat)*np.sin(dec) + np.cos(lat)*np.cos(dec)*np.cos(ha)

    # Wrap the sine of elevation in the [-1, +1] range
    sin_elev = (sin_elev + 1)%2 - 1

    elev = np.arcsin(sin_elev)

    return azim, elev

# Vectorize the raDec2AltAz function so it can take numpy arrays for: ra, dec, jd
raDec2AltAz_vect = np.vectorize(raDec2AltAz, excluded=['lat', 'lon'])



def raDec2ECI(ra, dec):
    """ Convert right ascension and declination to Earth-centered inertial coordinates. 

    Arguments:
        ra: [float] right ascension in radians
        dec: [float] declination in radians

    Return:
        (x, y, z): [tuple of floats] Earth-centered inertial coordinates

    """

    x = np.cos(dec)*np.cos(ra)
    y = np.cos(dec)*np.sin(ra)
    z = np.sin(dec)

    return x, y, z



def eci2RaDec(eci):
    """ Convert Earth-centered intertial coordinates to right ascension and declination. 

    Arguments:
        eci: [3 element ndarray] Earth-centered inertial coordinats

    Return:
        (ra, dec): [tuple of floats] right ascension and declinaton
    """

    # Normalize the ECI coordinates
    eci = vectNorm(eci)

    # Calculate declination
    dec = np.arcsin(eci[2])

    # Calculate right ascension
    ra = np.arctan2(eci[1], eci[0])%(2*np.pi)

    return ra, dec

    

def rotatePolar(azim, dec, azim_rot, elev_rot):
    """ Rotate the given polar coordinates on a sphere by the given azimuth and elevation.
    
    Arguments:
        azim: [float] Azimuth of coordinates to be rotated.
        dec: [float] Elevation of coordinates to be rotated.
        azim_rot: [float] Azimuth for which the input coordinates will be rotated on a sphere.
        elev_rot: [float] Elevation for which the input coordinates will be rotated on a sphere.

    Return:
        (azim, dec): [floats] Rotated coordinates.

    """

    # Convert the angles to cartesian unit vectors
    cartesian_vect = np.array(raDec2ECI(azim, dec))


    # Rotate elevation
    cartesian_vect = rotateVector(cartesian_vect, np.array([0, 1, 0]), -elev_rot)

    # Rotate azimuth
    cartesian_vect = rotateVector(cartesian_vect, np.array([0, 0, 1]), azim_rot)


    # Convert carterian unit vector to polar coordinates
    azim, dec = np.array(eci2RaDec(cartesian_vect))


    return azim, dec




def raDec2Ecliptic(jd, ra, dec):
    """ Convert right ascension and declinatoin to ecliptic longitude and latitude.

    Arguments:
        jd: [float] Julian date of the desired epoch
        ra: [float] right ascension in radians
        dec: [float] declination in radians

    Return:
        L, B: [tuple of floats] ecliptic longitude and latitude in radians

    """

    T = (jd - J2000_JD.days)/36525.0

    eps = (23.0 + 26.0/60.0 + 21.448/3600.0)*np.pi/180.0 - (46.8150*T - 0.00059*(T**2) 
        + 0.001813*(T**3))*np.pi/180.0/3600.0

    # Calculate ecliptic longitude
    L = np.arctan2(np.sin(eps)*np.sin(dec) + np.sin(ra)*np.cos(dec)*np.cos(eps), np.cos(ra)*np.cos(dec))

    # Calculate ecliptic latitude
    B = np.arcsin(np.cos(eps)*np.sin(dec) - np.sin(ra)*np.cos(dec)*np.sin(eps))

    # Wrap the longitude to [0, 2pi] range
    L = L%(2*np.pi)

    # Wrap latitude to [-pi/2, pi/2] range
    B = (B + np.pi/2)%np.pi - np.pi/2

    return L, B



def ecliptic2RectangularCoord(L, B, r_au):
    """ Calculates the rectangular coordinates of the Earth, (x, y, z) components, in Au.

    """


    x = r_au*np.cos(L)*np.cos(B)
    y = r_au*np.sin(L)*np.cos(B)
    z = r_au*np.sin(B)

    return x, y, z



def rectangular2EclipticCoord(x, y, z):
    """ Calculate ecliptic coordinats from given rectangular coordinates. Rectangular coordinates must be in
        the ecliptic reference frame, J2000 equinox, and in kilometers.

    Arguments:
        x
        y
        z

    Return:
        L, B, r_au

    """

    # Calculate the distance from the Sun to the Earth in km
    r = vectMag(np.array([x, y, z]))

    # Calculate the ecliptic latitude
    B = np.arcsin(z/r)

    # Calculate ecliptic longitude
    L = np.arctan2(y, x)

    # Convert the distance to AU
    r_au = r/AU


    return L, B, r_au



def eclipticToRectangularVelocityVect(L, B, v):
    """ Calculate heliocentric velocity component given the ecliptic latitude, longitude and heliocentric 
        velocity.

    Arguments:
        L: [float] Ecliptic longitude (radians).
        B: [float] Ecliptic latitude (radians).
        v: [float] Heliocentric velocity.

    Return:
        (x, y, z): [tuple of floats] Heliocentric velocity components.
    """


    x = -v*np.cos(L)*np.cos(B)
    y = -v*np.sin(L)*np.cos(B)
    z = -v*np.sin(B)

    return x, y, z



def correctedEclipticCoord(L_g, B_g, v_g, earth_vel):
    """ Calculates the corrected ecliptic coordinates using the method of Sato and Watanabe (2014).
    
    Arguments:
        L_g: [float] Geocentric ecliptic longitude (radians).
        B_g: [float] Geocentric ecliptic latitude (radians).
        v_g: [float] Geocentric velocity (km/s).
        earth_vel: [3 element ndarray] Earh velocity vector (km/s)

    Return:
        L_h, B_h, met_v_h:
            L_h: [float] Corrected ecliptic longitude (radians).
            B_h: [float] Corrected ecliptic latitude (radians).
            met_v_h: [3 element ndarray] Heliocentric velocity vector of the meteoroid (km/s).
    """

    # Calculate velocity components of the meteor
    xm, ym, zm = eclipticToRectangularVelocityVect(L_g, B_g, v_g)

    # Calculate the heliocentric velocity vector magnitude
    v_h = vectMag(np.array(earth_vel) + np.array([xm, ym, zm]))

    # Calculate the corrected meteoroid velocity vector
    xm_c = (xm + earth_vel[0])/v_h
    ym_c = (ym + earth_vel[1])/v_h
    zm_c = (zm + earth_vel[2])/v_h

    # Calculate corrected radiant in ecliptic coordinates
    L_h = np.arctan2(ym_c, xm_c)
    B_h = np.arcsin(zm_c)

    # Calculate the heliocentric velocity vector of the meteoroid
    xh, yh, zh = eclipticToRectangularVelocityVect(L_h, B_h, v_h)

    return L_h, B_h, np.array([xh, yh, zh])




###########################################


### Precession ###

def equatorialCoordPrecession(start_epoch, final_epoch, ra, dec):
    """ Corrects Right Ascension and Declination from one epoch to another, taking only precession into 
        account.

        Implemented from: Jean Meeus - Astronomical Algorithms, 2nd edition, pages 134-135
    
    Arguments:
        start_epoch: [float] Julian date of the starting epoch
        final_epoch: [float] Julian date of the final epoch
        ra: [float] non-corrected right ascension in radians
        dec: [float] non-corrected declination in radians
    
    Return:
        (ra, dec): [tuple of floats] precessed equatorial coordinates in degrees

    """

    T = (start_epoch - J2000_JD.days)/36525.0
    t = (final_epoch - start_epoch)/36525.0

    # Calculate correction parameters
    zeta  = ((2306.2181 + 1.39656*T - 0.000139*T**2)*t + (0.30188 - 0.000344*T)*t**2 + 0.017998*t**3)/3600
    z     = ((2306.2181 + 1.39656*T - 0.000139*T**2)*t + (1.09468 + 0.000066*T)*t**2 + 0.018203*t**3)/3600
    theta = ((2004.3109 - 0.85330*T - 0.000217*T**2)*t - (0.42665 + 0.000217*T)*t**2 - 0.041833*t**3)/3600

    # Convert parameters to radians
    zeta, z, theta = map(math.radians, (zeta, z, theta))

    # Calculate the next set of parameters
    A = np.cos(dec)  *np.sin(ra + zeta)
    B = np.cos(theta)*np.cos(dec)*np.cos(ra + zeta) - np.sin(theta)*np.sin(dec)
    C = np.sin(theta)*np.cos(dec)*np.cos(ra + zeta) + np.cos(theta)*np.sin(dec)

    # Calculate right ascension
    ra_corr = np.arctan2(A, B) + z

    # Calculate declination (apply a different equation if close to the pole, closer then 0.5 degrees)
    if (np.pi/2 - np.abs(dec)) < np.radians(0.5):
        dec_corr = np.arccos(np.sqrt(A**2 + B**2))
    else:
        dec_corr = np.arcsin(C)

    # Wrap right ascension to [0, 2*pi] range
    ra_corr = ra_corr%(2*np.pi)

    # Wrap declination to [-pi/2, pi/2] range
    dec_corr = (dec_corr + np.pi/2)%np.pi - np.pi/2

    return ra_corr, dec_corr

# Vectorize the equatorialCoordPrecession, so ra and dec can be passed as numpy arrays
equatorialCoordPrecession_vect = np.vectorize(equatorialCoordPrecession, excluded=['start_epoch'])



def eclipticRectangularPrecession(start_jd, end_jd, x, y, z):
    """ Precess ecliptic rectangular coordinates from one epoch to the other.

        Source: Jean Meeus, Astronomical Algorithms, p. 137
    """

    T = (start_jd - end_jd)/36525.0
    t = (end_jd - start_jd)/36525.0

    eta = np.pi/180.0/3600.0*((47.0029 - 0.06603*T + 0.000598*(T**2))*t + (-0.03302 + 0.000598*T)*(t**2) 
        + 0.000060*(t**3))

    pi  = np.pi/180.0/3600.0*(174.876384*3600.0 + 3289.4789*T + 0.60622*(T**2) - (869.8089 + 0.50491*T)*t 
        + 0.03536*(t**2))

    p   = np.pi/180.0/3600.0*((5029.0966 + 2.22226*T - 0.000042*T*T)*t + (1.11113 - 0.000042*T)*(t**2) 
        - 0.000006*(t**3))

    R = np.hypot(z, np.hypot(y, x))
    B = np.arctan2(z, np.hypot(y, x))
    L = np.arctan2(y, x)

    a = np.cos(eta)*np.cos(B)*np.sin(pi - L) - np.sin(eta)*np.sin(B)
    b = np.cos(B)*np.cos(pi - L)
    c = np.cos(eta)*np.sin(B) + np.sin(eta)*np.cos(B)*np.sin(pi - L)

    L = p + pi - np.arctan2(a, b)
    B = np.arctan2(c, np.hypot(a, b))

    x = R*np.cos(B)*np.cos(L)
    y = R*np.cos(B)*np.sin(L)
    z = R*np.sin(B)

    return x, y, z




##################


if __name__ == "__main__":

    # TEST

    jd = date2JD(2016, 9, 29, 6, 29, 45, millisecond=452.127, UT_corr=2)
    print(jd)

    print(jd2Date(jd, UT_corr=2))

    # Test ECEF funtions
    lat, lon, h = np.radians(18.5), np.radians(45.3), 90
    print(lat, lon, h)
    x, y, z = latLonAlt2ECEF(lat, lon, h)
    print(x, y, z)
    print(ecef2LatLonAlt(x, y, z))


    x, y, z = geo2Cartesian(lat, lon, h, jd)
    print(x, y, z)
    print(cartesian2Geo(jd, x, y, z))


    # SPE 9 meteor burster
    ra = np.radians(47.98)
    dec = np.radians(+39.35)
    jd = date2JD(2016, 9, 9, 23, 06, 59)

    L, B = np.degrees(raDec2Ecliptic(jd, ra, dec))

    print('L:', L)
    print('B:', B)


    jd = date2JD(2011, 2, 4, 23, 20, 42)
    print('JD: {:.12f}'.format(jd))
    print(jd2LST(jd,  18.41805))


    # Precess back and forth test
    print('RA, Dec:', np.degrees(ra), np.degrees(dec))
    ra_prec, dec_prec = equatorialCoordPrecession(jd, J2000_JD.days, ra, dec)
    print('RA, Dec precessed:', np.degrees(ra_prec), np.degrees(dec_prec))
    ra_prec_back, dec_prec_back = equatorialCoordPrecession(J2000_JD.days, jd, ra_prec, dec_prec)
    print('RA, Dec precessed back:', np.degrees(ra_prec_back), np.degrees(dec_prec_back))


    # Ra/Dec -> Alt/az -> Ra/Dec test
    print('Ra/Dec -> Alt/az -> Ra/Dec test')
    print(np.degrees(ra), np.degrees(dec), jd, np.degrees(lat), np.degrees(lon))
    azim, elev = raDec2AltAz(ra, dec, jd, lat, lon)
    print('Azim: ', np.degrees(azim), 'elev:', np.degrees(elev))
    ra_back, dec_back = altAz2RADec(azim, elev, jd, lat, lon)
    print(np.degrees(ra_back), np.degrees(dec_back))