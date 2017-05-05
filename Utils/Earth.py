""" Calculating the position of the Earth at given time, using VSOP87 and JPL DE430. """

from __future__ import print_function, division, absolute_import

import os
import numpy as np
from jplephem.spk import SPK

from Utils.TrajConversions import J2000_JD, J2000_OBLIQUITY, AU, eclipticRectangularPrecession, \
    ecliptic2RectangularCoord, rectangular2EclipticCoord

from Utils.Math import rotateVector



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

        The calculations are done using the VSOP87 model.

    Arguments:
        jd: [float] Julian date
        vsop_data: [VSOP87 object] loaded VSOP87 data

    Return:
        L, B, r_au: [tuple of floats]
            L - ecliptic longitude in radians
            B - ecliptic latitude in radians
            r_au - distante from the Earth to the Sun in AU
    """

    T = (jd - J2000_JD.days)/365250.0

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
    position = rotateVector(position, np.array([1, 0, 0]), -J2000_OBLIQUITY)

    # Rotate the velocity vector to the ecliptic reference frame (from the Earth equator reference frame)
    velocity = rotateVector(velocity, np.array([1, 0, 0]), -J2000_OBLIQUITY)

    # Return the position and the velocity of the Earth with respect to the Sun
    return position, velocity






if __name__ == "__main__":

    from Utils.TrajConversions import date2JD

    jd = date2JD(2016, 9, 29, 6, 29, 45)

    print("JD:", "{:10.10f}".format(jd))


    vsop_file = os.path.join('share', 'VSOP87D.ear')

    # Load VSOP data
    vsop_data = VSOP87(vsop_file)


    jpl_ephem_file = os.path.join('share', 'de430.bsp')

    # Load JPL ephemerids files
    jpl_data = SPK.open(jpl_ephem_file)


    # Calculate ecliptic coordinates
    L, B, r_au = calcEarthEclipticCoordVSOP(jd, vsop_data)

    print('Ecliptic longitude:', np.degrees(L))
    print('Ecliptic latitude:', np.degrees(B))
    print('Distance (AU):', r_au)

    # Convert ecliptic coordinates to Cartesian coordinates
    x, y, z = ecliptic2RectangularCoord(L, B, r_au)

    print('X:', x*AU)
    print('Y:', y*AU)
    print('Z:', z*AU)

    jpl_pos, jpl_vel = calcEarthRectangularCoordJPL(jd, jpl_data)
    print('JPL:', jpl_pos, jpl_vel)

    # Precess the coordinates to the epoch of date
    x, y, z = eclipticRectangularPrecession(J2000_JD.days, jd, *jpl_pos)


    # Calculate ecliptic coordinates from the JPL rectangular coordinates
    L, B, r_au = rectangular2EclipticCoord(x, y, z)

    print('Ecliptic longitude:', np.degrees(L))
    print('Ecliptic latitude:', np.degrees(B))
    print('Distance (AU):', r_au)