""" Calculating the position of the Earth at given time, using VSOP87 and JPL DE430. """

from __future__ import print_function, division, absolute_import

import os
import numpy as np
import math
from jplephem.spk import SPK

import wmpl.Utils.TrajConversions
from wmpl.Utils.Math import rotateVector



class VSOP87:

    def __init__(self, file_name):

        self.file_name = file_name

        # List of fitted parameters
        self.fit_list = []

        self.loadVSOP87()


    def loadVSOP87(self):
        """ Loads VSOP87 file for the Earth. """


        with open(self.file_name) as f:

            for line in f:
                
                # Skip header lines
                if 'VSOP87' in line:
                    continue

                # Read the p and q indices
                p = int(line[3]) - 1
                q = int(line[4])

                # Read amplitude
                A = float(line[79:97])

                # Read phase
                B = float(line[97:111])

                # Read frequency
                C = float(line[111:131])

                self.fit_list.append([p, q, A, B, C])




def calcEarthEclipticCoordVSOP(jd, vsop_data):
    """ Calculates the ecliptic coordinates of the Earth for the given Julian date.

        The calculations are done using the VSOP87 model, the returned coordinates are heliocentric in the
        epoch of date.

    Arguments:
        jd: [float] Julian date
        vsop_data: [VSOP87 object] loaded VSOP87 data

    Return:
        L, B, r_au: [tuple of floats]
            L - ecliptic longitude in radians
            B - ecliptic latitude in radians
            r_au - distante from the Earth to the Sun in AU
    """

    T = (jd - wmpl.Utils.TrajConversions.J2000_JD.days)/365250.0

    pos = np.zeros(3)
    P = np.zeros(6)

    # Init term degrees
    for n in range(6):
        P[n] = T**n

    # Calculate coordinates
    for fit in vsop_data.fit_list:

        p, q, A, B, C = fit

        pos[p] += P[q]*A*np.cos(B + C*T)


    # Unpack calculated values
    L, B, r_au = pos

    # Wrap the ecliptic longitude to 2 pi
    L = L%(2*np.pi)


    return L, B, r_au




def calcEarthRectangularCoordJPL(jd, jpl_data, sun_centre_origin=False):
    """ Calculate the ecliptic rectangular coordinates of the Earth at the given Julian date. Epoch is J2000,
        the returned units are in kilometers. The coordinate are calculated using DE430 ephemerids. The
        centre of the coordinate system is the Solar system barycentre, unless otherwise specified.

    Arguments:
        jd: [float] Julian date
        jpl_data: [?] SPK loaded with jplephem library

    Keyword arguments:
        sun_centre_origin: [bool] If True, the origin of the coordinate system will be in the Sun centre. 
            If False (default), the origin will be in the Solar system barycentre.

    Return:
        position, velocity: [tuple of ndarrays] position and velocity of Earth in kilometers and km/s, in 
            ecliptic rectangular cordinates, in J2000.0 epoch.

    """

    # Calculate the position and the velocity of the Earth-Moon barycentre system with respect to the Solar System Barycentre
    position_bary, velocity_bary = jpl_data[0, 3].compute_and_differentiate(jd)

    # Calculate the position of the Sun with respect to the Solar System Barycentre
    position_sun, velocity_sun = jpl_data[0, 10].compute_and_differentiate(jd)

    # Calculate the position and the velocity of the Earth with respect to the Earth-Moon barycentre
    position_earth, velocity_earth = jpl_data[3, 399].compute_and_differentiate(jd)


    # Origin in the centre of mass of the Sun
    if sun_centre_origin:

        # Calculate the position of the Earth with respect to the centre of mass of the Sun
        position = position_bary - position_sun + position_earth

        # Calculate the velocity of the Earth in km/s
        velocity = (velocity_bary - velocity_sun + velocity_earth)/86400.0


    # Origin in the Solar system barycentre
    else:

        # Calculate the position of the Earth with respect to the Solar system barycentre
        position = position_bary + position_earth

        # Calculate the velocity of the Earth in km/s
        velocity = (velocity_bary + velocity_earth)/86400.0


    # Rotate the position to the ecliptic reference frame (from the Earth equator reference frame)
    position = rotateVector(position, np.array([1, 0, 0]), -wmpl.Utils.TrajConversions.J2000_OBLIQUITY)

    # Rotate the velocity vector to the ecliptic reference frame (from the Earth equator reference frame)
    velocity = rotateVector(velocity, np.array([1, 0, 0]), -wmpl.Utils.TrajConversions.J2000_OBLIQUITY)

    # Return the position and the velocity of the Earth with respect to the Sun
    return position, velocity




def calcNutationComponents(jd_dyn):
    """ Calculate Earth's nutation components from the given Julian date.

    Source: Meeus (1998) Astronomical algorithms, 2nd edition, chapter 22.
    
    The precision is limited to 0.5" in nutation in longitude and 0.1" in nutation in obliquity. The errata
    for the 2nd edition was used to correct the equation for delta_psi.
    
    Arguments:
        jd_dyn: [float] Dynamical Julian date. See Utils.TrajConversions.jd2DynamicalTimeJD function.

    Return:
        (delta_psi, delta_eps): [tuple of floats] Differences from mean nutation due to the influence of
            the Moon and minor effects (radians).
    """


    T = (jd_dyn - wmpl.Utils.TrajConversions.J2000_JD.days)/36525.0

    # # Mean Elongation of the Moon from the Sun
    # D = 297.85036 + 445267.11148*T - 0.0019142*T**2 + (T**3)/189474

    # # Mean anomaly of the Earth with respect to the Sun
    # M = 357.52772 + 35999.05034*T - 0.0001603*T**2 - (T**3)/300000

    # # Mean anomaly of the Moon
    # Mm = 134.96298 + 477198.867398*T + 0.0086972*T**2 + (T**3)/56250

    # # Argument of latitude of the Moon
    # F = 93.27191  + 483202.017538*T - 0.0036825*T**2 + (T**3)/327270


    # Longitude of the ascending node of the Moon's mean orbit on the ecliptic, measured from the mean equinox
    # of the date
    omega = 125.04452 - 1934.136261*T


    # Mean longitude of the Sun (deg)
    L = 280.4665 + 36000.7698*T

    # Mean longitude of the Moon (deg)
    Ll = 218.3165 + 481267.8813*T


    # Nutation in longitude
    delta_psi = -17.2*math.sin(math.radians(omega)) - 1.32*math.sin(np.radians(2*L)) \
        - 0.23*math.sin(math.radians(2*Ll)) + 0.21*math.sin(math.radians(2*omega))

    # Nutation in obliquity
    delta_eps = 9.2*math.cos(math.radians(omega)) + 0.57*math.cos(math.radians(2*L)) \
        + 0.1*math.cos(math.radians(2*Ll)) - 0.09*math.cos(math.radians(2*omega))


    # Convert to radians
    delta_psi = np.radians(delta_psi/3600)
    delta_eps = np.radians(delta_eps/3600)

    return delta_psi, delta_eps


    ### For higher precision (more than 0.5" in delta psi and 0.1" in delta eps) use the code below:


    # # Longitude of the ascending node of the Moon
    # omega = 125.04452 - 1934.136261*T   + 0.0020708*T**2 + (T**3)/450000


    # # Periodic terms
    # #
    # # Source: http://read.pudn.com/downloads137/sourcecode/others/585305/nutate.c__.htm
    # #
    # # Trigonometric arguments: 
    # # i - Mm
    # # j - M
    # # k - F
    # # l - D
    # # m - omega
    # # a - C cos
    # # b - C cos T
    # # c - C sin
    # # d - C sin T
    # #         i      j   k   l   m        a       b      c     d  
    # pp = [  [ 0,     0,  0,  0,  1,   -171996, -1742,  92025, 89],
    #         [ 0,     0,  0,  0,  2,      2062,    2,    -895,  5],
    #         [-2,     0,  2,  0,  1,      46,      0,     -24,  0],
    #         [ 2,     0, -2,  0,  0,      11,      0,       0,  0],
    #         [-2,     0,  2,  0,  2,     -3,       0,       1,  0],
    #         [ 1,    -1,  0, -1,  0,     -3,       0,       0,  0],
    #         [ 0,    -2,  2, -2,  1,     -2,       0,       1,  0],
    #         [ 2,     0, -2,  0,  1,      1,       0,       0,  0],
    #         [ 0,     0,  2, -2,  2,     -13187, -16,    5736,-31],
    #         [ 0,     1,  0,  0,  0,      1426,  -34,      54, -1],
    #         [ 0,     1,  2, -2,  2,     -517,    12,     224, -6],
    #         [ 0,    -1,  2, -2,  2,      217,    -5,     -95,  3],
    #         [ 0,     0,  2, -2,  1,      129,     1,     -70,  0],
    #         [ 2,     0,  0, -2,  0,      48,      0,       1,  0],
    #         [ 0,     0,  2, -2,  0,     -22,      0,       0,  0],
    #         [ 0,     2,  0,  0,  0,      17,     -1,       0,  0],
    #         [ 0,     1,  0,  0,  1,     -15,      0,       9,  0],
    #         [ 0,     2,  2, -2,  2,     -16,      1,       7,  0],
    #         [ 0,    -1,  0,  0,  1,     -12,      0,       6,  0],
    #         [-2,     0,  0,  2,  1,     -6,       0,       3,  0],
    #         [ 0,    -1,  2, -2,  1,     -5,       0,       3,  0],
    #         [ 2,     0,  0, -2,  1,      4,       0,-      2,  0],
    #         [ 0,     1,  2, -2,  1,      4,       0,-      2,  0],
    #         [ 1,     0,  0, -1,  0,     -4,       0,       0,  0],
    #         [ 2,     1,  0, -2,  0,      1,       0,       0,  0],
    #         [ 0,     0, -2,  2,  1,      1,       0,       0,  0],
    #         [ 0,     1, -2,  2,  0,     -1,       0,       0,  0],
    #         [ 0,     1,  0,  0,  2,      1,       0,       0,  0],
    #         [-1,     0,  0,  1,  1,      1,       0,       0,  0],
    #         [ 0,     1,  2, -2,  0,     -1,       0,       0,  0],
    #         [ 0,     0,  2,  0,  2,     -2274,   -2,     977,- 5],
    #         [ 1,     0,  0,  0,  0,      712,     1,      -7,  0],
    #         [ 0,     0,  2,  0,  1,     -386,    -4,     200,  0],
    #         [ 1,     0,  2,  0,  2,     -301,     0,     129,- 1],
    #         [ 1,     0,  0, -2,  0,     -158,     0,      -1,  0],
    #         [-1,     0,  2,  0,  2,      123,     0,     -53,  0],
    #         [ 0,     0,  0,  2,  0,      63,      0,      -2,  0],
    #         [ 1,     0,  0,  0,  1,      63,      1,     -33,  0],
    #         [-1,     0,  0,  0,  1,     -58,     -1,      32,  0],
    #         [-1,     0,  2,  2,  2,     -59,      0,      26,  0],
    #         [ 1,     0,  2,  0,  1,     -51,      0,      27,  0],
    #         [ 0,     0,  2,  2,  2,     -38,      0,      16,  0],
    #         [ 2,     0,  0,  0,  0,      29,      0,      -1,  0],
    #         [ 1,     0,  2, -2,  2,      29,      0,     -12,  0],
    #         [ 2,     0,  2,  0,  2,     -31,      0,      13,  0],
    #         [ 0,     0,  2,  0,  0,      26,      0,      -1,  0],
    #         [-1,     0,  2,  0,  1,      21,      0,     -10,  0],
    #         [-1,     0,  0,  2,  1,      16,      0,      -8,  0],
    #         [ 1,     0,  0, -2,  1,     -13,      0,       7,  0],
    #         [-1,     0,  2,  2,  1,     -10,      0,       5,  0],
    #         [ 1,     1,  0, -2,  0,      -7,      0,       0,  0],
    #         [ 0,     1,  2,  0,  2,       7,      0,      -3,  0],
    #         [ 0,    -1,  2,  0,  2,      -7,      0,       3,  0],
    #         [ 1,     0,  2,  2,  2,      -8,      0,       3,  0],
    #         [ 1,     0,  0,  2,  0,       6,      0,       0,  0],
    #         [ 2,     0,  2, -2,  2,       6,      0,      -3,  0],
    #         [ 0,     0,  0,  2,  1,      -6,      0,       3,  0],
    #         [ 0,     0,  2,  2,  1,      -7,      0,       3,  0],
    #         [ 1,     0,  2, -2,  1,       6,      0,      -3,  0],
    #         [ 0,     0,  0, -2,  1,      -5,      0,       3,  0],
    #         [ 1,    -1,  0,  0,  0,       5,      0,       0,  0],
    #         [ 2,     0,  2,  0,  1,      -5,      0,       3,  0],
    #         [ 0,     1,  0, -2,  0,      -4,      0,       0,  0],
    #         [ 1,     0, -2,  0,  0,       4,      0,       0,  0],
    #         [ 0,     0,  0,  1,  0,      -4,      0,       0,  0],
    #         [ 1,     1,  0,  0,  0,      -3,      0,       0,  0],
    #         [ 1,     0,  2,  0,  0,       3,      0,       0,  0],
    #         [ 1,    -1,  2,  0,  2,      -3,      0,       1,  0],
    #         [-1,    -1,  2,  2,  2,      -3,      0,       1,  0],
    #         [-2,     0,  0,  0,  1,      -2,      0,       1,  0],
    #         [ 3,     0,  2,  0,  2,      -3,      0,       1,  0],
    #         [ 0,    -1,  2,  2,  2,      -3,      0,       1,  0],
    #         [ 1,     1,  2,  0,  2,       2,      0,      -1,  0],
    #         [-1,     0,  2, -2,  1,      -2,      0,       1,  0],
    #         [ 2,     0,  0,  0,  1,       2,      0,      -1,  0],
    #         [ 1,     0,  0,  0,  2,      -2,      0,       1,  0],
    #         [ 3,     0,  0,  0,  0,       2,      0,       0,  0],
    #         [ 0,     0,  2,  1,  2,       2,      0,      -1,  0],
    #         [-1,     0,  0,  0,  2,       1,      0,      -1,  0],
    #         [ 1,     0,  0, -4,  0,      -1,      0,       0,  0],
    #         [-2,     0,  2,  2,  2,       1,      0,      -1,  0],
    #         [-1,     0,  2,  4,  2,      -2,      0,       1,  0],
    #         [ 2,     0,  0, -4,  0,      -1,      0,       0,  0],
    #         [ 1,     1,  2, -2,  2,       1,      0,      -1,  0],
    #         [ 1,     0,  2,  2,  1,      -1,      0,       1,  0],
    #         [-2,     0,  2,  4,  2,      -1,      0,       1,  0],
    #         [-1,     0,  4,  0,  2,       1,      0,       0,  0],
    #         [ 1,    -1,  0, -2,  0,       1,      0,       0,  0],
    #         [ 2,     0,  2, -2,  1,       1,      0,      -1,  0],
    #         [ 2,     0,  2,  2,  2,      -1,      0,       0,  0],
    #         [ 1,     0,  0,  2,  1,      -1,      0,       0,  0],
    #         [ 0,     0,  4, -2,  2,       1,      0,       0,  0],
    #         [ 3,     0,  2, -2,  2,       1,      0,       0,  0],
    #         [ 1,     0,  2, -2,  0,      -1,      0,       0,  0],
    #         [ 0,     1,  2,  0,  1,       1,      0,       0,  0],
    #         [-1,    -1,  0,  2,  1,       1,      0,       0,  0],
    #         [ 0,     0, -2,  0,  1,      -1,      0,       0,  0],
    #         [ 0,     0,  2, -1,  2,      -1,      0,       0,  0],
    #         [ 0,     1,  0,  2,  0,      -1,      0,       0,  0],
    #         [ 1,     0, -2, -2,  0,      -1,      0,       0,  0],
    #         [ 0,    -1,  2,  0,  1,      -1,      0,       0,  0],
    #         [ 1,     1,  0, -2,  1,      -1,      0,       0,  0],
    #         [ 1,     0, -2,  2,  0,      -1,      0,       0,  0],
    #         [ 2,     0,  0,  2,  0,       1,      0,       0,  0],
    #         [ 0,     0,  2,  4,  2,      -1,      0,       0,  0],
    #         [ 0,     1,  0,  1,  0,       1,      0,       0,  0]]


    # delta_psi = 0
    # delta_eps = 0


    # # Add up the periodic terms
    # for i, j, k, l, m, a, b, c, d in pp:

    #     alpha = math.radians(D*l + M*j + Mm*i + F*k + omega*m)

    #     # Nutation in longitude
    #     delta_psi = (c + d*T/10.0)*math.sin(alpha)

    #     # Nutation in obliquity
    #     delta_eps = (a + b*T/10.0)*math.cos(alpha)


    # # Convert delta psi and eps to radians
    # delta_psi = np.radians(10000*3600*delta_psi)
    # delta_eps = np.radians(10000*3600*delta_eps)


    # return delta_psi, delta_eps




def calcTrueObliquity(jd):
    """ Calculate the true obliquity of the Earth at the given Julian date. 
    
    Arguments:
        jd_dyn: [float] Julian date.

    Return:
        eps: [float] True obliquity eps0 + delta_eps.

    """


    # Calculate the dynamical time JD
    jd_dyn = wmpl.Utils.TrajConversions.jd2DynamicalTimeJD(jd)


    # Calculate Earth's nutation components
    delta_psi, delta_eps = calcNutationComponents(jd_dyn)


    # Calculate the mean obliquity (in arcsec)
    u = (jd_dyn - 2451545.0)/3652500.0
    eps0 = 84381.448 - 4680.93*u - 1.55*u**2 + 1999.25*u**3 - 51.38*u**4 - 249.67*u**5 - 39.05*u**6 \
        + 7.12*u**7 + 27.87*u**8 + 5.79*u**9 + 2.45*u**10


    # Convert to radians
    eps0 /= 3600
    eps0 = np.radians(eps0)

    # Calculate true obliquity
    eps = (eps0 + delta_eps)%(2*np.pi)


    return eps




def calcApparentSiderealEarthRotation(julian_date):
    """ Calculate apparent sidereal rotation GST of the Earth. 
        
        Calculated according to: 
        Clark, D. L. (2010). Searching for fireball pre-detections in sky surveys. The School of Graduate and 
        Postdoctoral Studies. University of Western Ontario, London, Ontario, Canada, MSc Thesis.

    """

    t = (julian_date - wmpl.Utils.TrajConversions.J2000_JD.days)/36525.0

    # Calculate the Mean sidereal rotation of the Earth in radians (Greenwich Sidereal Time)
    GST = 280.46061837 + 360.98564736629*(julian_date - wmpl.Utils.TrajConversions.J2000_JD.days) + 0.000387933*t**2 - (t**3)/38710000
    GST = (GST + 360)%360
    GST = math.radians(GST)

    # print('GST:', np.degrees(GST), 'deg')

    # Calculate the dynamical time JD
    jd_dyn = wmpl.Utils.TrajConversions.jd2DynamicalTimeJD(julian_date)


    # Calculate Earth's nutation components
    delta_psi, delta_eps = calcNutationComponents(jd_dyn)

    # print('Delta Psi:', np.degrees(delta_psi), 'deg')
    # print('Delta Epsilon:', np.degrees(delta_eps), 'deg')


    # Calculate the mean obliquity (in arcsec)
    u = (jd_dyn - 2451545.0)/3652500.0
    eps0 = 84381.448 - 4680.93*u - 1.55*u**2 + 1999.25*u**3 - 51.38*u**4 - 249.67*u**5 - 39.05*u**6 \
        + 7.12*u**7 + 27.87*u**8 + 5.79*u**9 + 2.45*u**10

    # Convert to radians
    eps0 /= 3600
    eps0 = np.radians(eps0)

    # print('Mean obliquity:', np.degrees(eps0), 'deg')

    # Calculate apparent sidereal Earth's rotation
    app_sid_rot = (GST + delta_psi*math.cos(eps0 + delta_eps))%(2*math.pi)

    return app_sid_rot




def greatCircleDistance(lat1, lon1, lat2, lon2):
    """ Calculate the great circle distance in kilometers between two points on the Earth.
        Source: https://gis.stackexchange.com/a/56589/15183

    Arguments:
        lat1: [float] Latitude 1 (radians).
        lon1: [float] Longitude 1 (radians).
        lat2: [float] Latitude 2 (radians).
        lon2: [float] Longitude 2 (radians).

    Return:
        [float]: Distance in kilometers.
    """
    
    # Haversine formula
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))

    # Distance in kilometers.
    dist = 6371*c

    return dist




if __name__ == "__main__":

    from wmpl.Utils.TrajConversions import date2JD

    #jd = date2JD(2016, 9, 29, 6, 29, 45)
    #jd = 2455843.318098123185
    jd = 2455843.318098123

    print("JD:", "{:10.10f}".format(jd))


    vsop_file = os.path.join('share', 'VSOP87D.ear')

    # Load VSOP data
    vsop_data = VSOP87(vsop_file)


    jpl_ephem_file = os.path.join('share', 'de430.bsp')

    # Load JPL ephemerids files
    jpl_data = SPK.open(jpl_ephem_file)


    # Calculate ecliptic coordinates
    L, B, r_au = calcEarthEclipticCoordVSOP(jd, vsop_data)

    print()
    print('VSOP87, J2000:')

    # Convert ecliptic coordinates to Cartesian coordinates
    x, y, z = wmpl.Utils.TrajConversions.ecliptic2RectangularCoord(L, B, r_au)

    # Precess VSOP coordinates to J2000
    x, y, z = wmpl.Utils.TrajConversions.eclipticRectangularPrecession(jd, wmpl.Utils.TrajConversions.J2000_JD.days, \
        x, y, z)

    L, B, r_au = wmpl.Utils.TrajConversions.rectangular2EclipticCoord(x, y, z)


    print('Ecliptic longitude:', np.degrees(L))
    print('Ecliptic latitude:', np.degrees(B))
    print('Distance (AU):', r_au)

    print('X:', x*wmpl.Utils.TrajConversions.AU)
    print('Y:', y*wmpl.Utils.TrajConversions.AU)
    print('Z:', z*wmpl.Utils.TrajConversions.AU)

    print()
    print('JPL DE430, J2000, heliocentric:')
    jpl_pos, jpl_vel = calcEarthRectangularCoordJPL(jd, jpl_data, sun_centre_origin=True)

    # Precess the JPL coordinates to the epoch of date
    #x, y, z = wmpl.Utils.TrajConversions.eclipticRectangularPrecession(wmpl.Utils.TrajConversions.J2000_JD.days, jd, *jpl_pos)


    # Calculate ecliptic coordinates from the JPL rectangular coordinates
    L, B, r_au = wmpl.Utils.TrajConversions.rectangular2EclipticCoord(x, y, z)

    print('Ecliptic longitude:', np.degrees(L))
    print('Ecliptic latitude:', np.degrees(B))
    print('Distance (AU):', r_au)

    print('X:', jpl_pos[0])
    print('Y:', jpl_pos[1])
    print('Z:', jpl_pos[2])

    print('Vx', jpl_vel[0])
    print('Vy', jpl_vel[1])
    print('Vz', jpl_vel[2])


    print()
    print('------------')
    # Test the apparent LST calculation
    print('JD: {:.10f}'.format(jd))
    print('Apparent GST: {:.10f} deg'.format(np.degrees(calcApparentSiderealEarthRotation(jd))))


    # Earth obliquty at J2000, for some reason this does not match the constant value
    print('Calcualted obliquty@J2000:', np.degrees(calcTrueObliquity(wmpl.Utils.TrajConversions.J2000_JD.days)))