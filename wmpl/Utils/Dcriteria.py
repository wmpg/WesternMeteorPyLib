
from __future__ import print_function, division, absolute_import

import math

import numpy as np

from wmpl.Utils.OrbitConstants import A_JUPITER, GAUSS_K, GAUSS_K_SQUARED



# Average speed of Earth [km/s]
SPEED_EARTH = 29.7


def calcDSH(q1, e1, i1, O1, w1, q2, e2, i2, O2, w2):
    """ Calculate the Southworth and Hawking meteoroid orbit dissimilarity criterion.

    Arguments:
        q1: [double] perihelion distance of the first orbit
        e1: [double] num. eccentricity of the first orbit
        i1: [double] inclination of the first orbit (rad)
        O1: [double] longitude of ascending node of the first orbit (rad)
        w1: [double] argument of perihelion of the first orbit (rad)
        q2: [double] perihelion distance of the second orbit
        e2: [double] num. eccentricity of the second orbit
        i2: [double] inclination of the second orbit (rad)
        O2: [double] longitude of ascending node of the second orbit (rad)
        w2: [double] argument of perihelion of the second orbit (rad)

    Return:
        [double] D_SH value

    """

    rho = 1

    if (abs(O2 - O1) > math.pi):
        rho = -1


    I21 = math.acos(math.cos(i1)*math.cos(i2) + math.sin(i1)*math.sin(i2)*math.cos(O2 - O1))


    asin_val = math.cos((i2 + i1)/2.0)*math.sin((O2 - O1)/2.0)*(1/math.cos(I21/2.0))

    # Name sure the value going into asin is not beyond the bounds due to numerical reasons
    if abs(asin_val) > 1:
        asin_val = math.copysign(1.0, asin_val)

    pi21 = w2 - w1 + 2*rho*math.asin(asin_val)

    DSH2 = pow((e2 - e1), 2) + pow((q2 - q1), 2) + pow((2 * math.sin(I21/2.0)), 2) + \
        pow((e2 + e1)/2.0, 2)*pow((2 * math.sin(pi21 / 2.0)), 2)


    return math.sqrt(DSH2)




def calcDH(q1, e1, i1, O1, w1, q2, e2, i2, O2, w2):
    """ Calculate the Jopek meteoroid orbit dissimilarity criterion.

    Arguments:
        q1: [double] perihelion distance of the first orbit
        e1: [double] num. eccentricity of the first orbit
        i1: [double] inclination of the first orbit (rad)
        O1: [double] longitude of ascending node of the first orbit (rad)
        w1: [double] argument of perihelion of the first orbit (rad)
        q2: [double] perihelion distance of the second orbit
        e2: [double] num. eccentricity of the second orbit
        i2: [double] inclination of the second orbit (rad)
        O2: [double] longitude of ascending node of the second orbit (rad)
        w2: [double] argument of perihelion of the second orbit (rad)

    Return:
        [double] D_H value

    """

    I21 = math.acos(math.cos(i1)*math.cos(i2) + math.sin(i1)*math.sin(i2)*math.cos(O2 - O1))


    asin_val = math.cos((i2 + i1)/2.0)*math.sin((O2-O1)/2.0)*1/math.cos(I21/2.0)

    # Name sure the value going into asin is not beyond the bounds due to numerical reasons
    if abs(asin_val) > 1:
        asin_val = math.copysign(1.0, asin_val)

    pi21 = w2 - w1 + 2*math.asin(asin_val)

    DH2 = (e2 - e1)**2 + ((q2 - q1)/(q2 + q1))**2 + (2*math.sin(I21/2.0))**2 \
        + ((e2 + e1)/2.0)**2*(2*math.sin(pi21/2.0))**2

    return math.sqrt(DH2)




def calcDD(q1, e1, i1, O1, w1, q2, e2, i2, O2, w2):
    """ Calculate the Drummond (1981) meteoroid orbit dissimilarity criterion.

    Arguments:
        q1: [double] perihelion distance of the first orbit
        e1: [double] num. eccentricity of the first orbit
        i1: [double] inclination of the first orbit (rad)
        O1: [double] longitude of ascending node of the first orbit (rad)
        w1: [double] argument of perihelion of the first orbit (rad)
        q2: [double] perihelion distance of the second orbit
        e2: [double] num. eccentricity of the second orbit
        i2: [double] inclination of the second orbit (rad)
        O2: [double] longitude of ascending node of the second orbit (rad)
        w2: [double] argument of perihelion of the second orbit (rad)

    Return:
        [double] D_H value

    """

    I21 = math.acos(math.cos(i1)*math.cos(i2) + math.sin(i1)*math.sin(i2)*math.cos(O2 - O1))

    lambda1 = O1 + math.atan2(math.cos(i1)*math.sin(w1), math.cos(w1))

    beta1 = math.asin(math.sin(i1)*math.sin(w1))

    lambda2 = O2 + math.atan2(math.cos(i2)*math.sin(w2), math.cos(w2))

    beta2 = math.asin(math.sin(i2)*math.sin(w2))

    theta21 = math.acos(math.sin(beta1)*math.sin(beta2) + math.cos(beta1)*math.cos(beta2)*math.cos(lambda2 \
        - lambda1))

    DD2 = ((e2 - e1)/(e2 + e1))**2 + ((q2 - q1)/(q2 + q1))**2 + (I21/math.pi)**2 \
        + ((e2 + e1)/2.0)**2*(theta21/math.pi)**2

    return math.sqrt(DD2)




def calcVgComponents(ra, dec, sol, vg):
    """ Calculates geocentric velovity (Vg) components relative to Earth velocity. All components are in 
        J2000.0 and all angles are in radians. Used for calculating the Valsecchi D criteria, needed for 
        Opik Vg component calculation.

    Arguments:
        ra: [double] Right ascension (rad)
        dec: [double] Declination (rad)
        sol: [double] Solar longitude (rad)
        vg: [double] Geocentric velocity (km/s)

    Return:
        [list] Geocentric velocity vector components (km/s)
    """

    # Obliquity of Earth orbit for J2000.0 (Boulet)
    earth_EPS = 0.40909280
    sin_EPS = math.sin(earth_EPS)
    cos_EPS = math.cos(earth_EPS)

    # Terrestrial longitude J2000.0
    LE = sol - math.pi
    sin_LE = math.sin(LE)
    cos_LE = math.cos(LE)

    # Calculate Vg components before rotation
    vg_x = -(vg/SPEED_EARTH)*math.cos(dec)*math.cos(ra)
    vg_y = -(vg/SPEED_EARTH)*math.cos(dec)*math.sin(ra)
    vg_z = -(vg/SPEED_EARTH)*math.sin(dec)

    output = [0, 0, 0]

    # Rotate Vg components and add them to an output vector 
    output[0] =  cos_LE*vg_x + sin_LE*cos_EPS*vg_y + sin_LE*sin_EPS*vg_z
    output[1] = -sin_LE*vg_x + cos_LE*cos_EPS*vg_y + cos_LE*sin_EPS*vg_z
    output[2] =                      -sin_EPS*vg_y +        cos_EPS*vg_z

    return output




def calcDN(ra1, dec1, sol1, vg1, ra2, dec2, sol2, vg2, d_max=999.0):
    """ Calculate the Valsecchi D criterion between two orbits. Only parameters used are ra, dec, sol, vg, 
        other are disregarded.

    Arguments:
        point1: [double pointer] container for:
            ra1: [double] right ascension, 1st orbit (radians)
            dec1: [double] declination, 1st orbit (radians)
            sol1: [double] solar longitude, 1st orbit (radians)
            vg1: [double] geocentric velocity, 1st orbit (km/s)
            q1: [double] perihelion distance of the first orbit
            e1: [double] num. eccentricity of the first orbit
            i1: [double] inclination of the first orbit (radians)
            O1: [double] longitude of ascending node of the first orbit (radians)
            w1: [double] argument of perihelion of the first orbit (radians)

        point2: [double pointer] container for:
            ra2: [double] right ascension, 2nd orbit (radians)
            dec2: [double] declination, 2nd orbit (radians)
            sol2: [double] solar longitude, 2nd orbit (radians)
            vg2: [double] geocentric velocity, 2nd orbit (km/s)
            q2: [double] perihelion distance of the second orbit
            e2: [double] num. eccentricity of the second orbit
            i2: [double] inclination of the second orbit (radians)
            O2: [double] longitude of ascending node of the second orbit (radians)
            w2: [double] argument of perihelion of the second orbit (radians)

        d_max: [double] maximum value of the criterion, if larger than that number, 999.0 will be returned
            (used for speeding up the algorithm)

    Return:
        [double] Valsecchi D criterion value
    """

    # Define weights
    w1 = 1.0
    w2 = 1.0
    w3 = 1.0

    # Calculate the Vg components relative to Earth
    vg_x1, vg_y1, vg_z1 = calcVgComponents(ra1, dec1, sol1, vg1)

    vg_x2, vg_y2, vg_z2 = calcVgComponents(ra2, dec2, sol2, vg2)

    # Reaclaulate the speeds to realtive speed to Earth
    vg1 = vg1/SPEED_EARTH
    vg2 = vg2/SPEED_EARTH

    # Primary comparison
    sqr_diff_Vg = (vg2 - vg1)**2

    if sqr_diff_Vg <= d_max:

        # Orbit 1B, choose correct quadrant
        phi1 = math.atan2(vg_x1, vg_z1)
        cos_theta1 = vg_y1/vg1

        # Orbit 2B, choose correct quadrant
        phi2 = math.atan2(vg_x2, vg_z2)
        cos_theta2 = vg_y2/vg2

        # Final comparison
        sqr_diff_cos_theta = w1 * (cos_theta2 - cos_theta1)**2

        # Secondary check against max. dissimilarity
        if (sqr_diff_cos_theta <= d_max):
            
            d_phi_A = 2*math.sin((phi2 - phi1)/2.0)
            d_phi_B = 2*math.sin((math.pi + phi2 - phi1)/2.0)

            d_lambda_A = 2*math.sin((sol2 - sol1)/2.0)
            d_lambda_B = 2*math.sin((math.pi + sol2 - sol1)/2.0)

            d_zeta = min(w2 * d_phi_A**2 + w3 * d_lambda_A**2, 
                w2 * d_phi_B**2 + w3 * d_lambda_B**2)

            dissim = math.sqrt(sqr_diff_Vg + sqr_diff_cos_theta + d_zeta)

        else:
            return d_max

    else:
        dissim = d_max

    if (dissim > d_max):
        dissim = d_max

    return dissim




# def calcDV(Lh1, Bh1, sol1, Vh1, Lh2, Bh2, sol2, Vh2, d_max=999.0):
#     """ D criterion calculated using Vida et al. 2018 (TBP) which uses the corrected heliocentric velocity
#         vector (correction by Sato & Watanabe 2017) and Valsecchi-type D criterion approach of calculating
#         the similarity between orbits.

#     Arguments:
#         Lh1: [float] Corrected Sun-centred ecliptic longitude of meteor A (radians).
#         Bh1: [float] Corrected Sun-centred ecliptic latitude of meteor A (radians).
#         Vh1: [float] Heliocentric velocity of meteor A (km/s).
#         sol1: [float] Solar longitude of meteor A (radians).
#         Lh2: [float] Corrected Sun-centred ecliptic longitude of meteor B (radians).
#         Bh2: [float] Corrected Sun-centred ecliptic latitude of meteor B (radians).
#         Vh2: [float] Heliocentric velocity of meteor B (km/s).
#         sol2: [float] Solar longitude of meteor B (radians).

#     Keyword arguments:
#         d_max: [float] Maximum values of returned D criteria. This is used for speeding up calculations, as
#             some checks can be done before the whole criteria is calculated. Default value is 999.0.
    
#     Return:
#         [float] Value of calculated D criteria.

#     """

#     # Define weights
#     w1 = 1.0
#     w2 = 1.0
#     w3 = 1.0


#     # Convert ecliptic angles to velocity vector
#     Vx1 = -Vh1*math.cos(Lh1)*math.cos(Bh1)
#     Vy1 = -Vh1*math.sin(Lh1)*math.cos(Bh1)
#     Vz1 = -Vh1*math.sin(Bh1)

#     Vx2 = -Vh2*math.cos(Lh2)*math.cos(Bh2)
#     Vy2 = -Vh2*math.sin(Lh2)*math.cos(Bh2)
#     Vz2 = -Vh2*math.sin(Bh2)

#     # Squared difference between the heliocentric velocities
#     sqr_diff_Vh = (Vh2 - Vh1)**2

#     # First cut - check if the difference between the heliocentric velocitites is too large
#     if sqr_diff_Vh <= d_max:


#         # Orbit 1B, choose correct quadrant
#         phi1 = math.atan2(Vx1, Vz1)
#         cos_theta1 = Vy1/Vh1

#         # Orbit 2B, choose correct quadrant
#         phi2 = math.atan2(Vx2, Vz2)
#         cos_theta2 = Vy2/Vh2

#         sqr_diff_cos_theta = w1*(cos_theta2 - cos_theta1)**2

#         # Secondary check against max. dissimilarity
#         if (sqr_diff_cos_theta <= d_max):
            
#             d_phi_A = 2*math.sin((phi2 - phi1)/2.0)
#             d_phi_B = 2*math.sin((math.pi + phi2 - phi1)/2.0)

#             d_lambda_A = 2*math.sin((sol2 - sol1)/2.0)
#             d_lambda_B = 2*math.sin((math.pi + sol2 - sol1)/2.0)

#             d_zeta = min(w2 * d_phi_A**2 + w3 * d_lambda_A**2, 
#                 w2 * d_phi_B**2 + w3 * d_lambda_B**2)

#             dissim = math.sqrt(sqr_diff_Vh + sqr_diff_cos_theta + d_zeta)


#             return dissim

#         else:
#             return d_max

#     else:
#         return d_max



def calcDVuncert(Lh1, Lh1_std, Bh1, Bh1_std, sol1, Vh1, Vh1_std, Lh2, Lh2_std, Bh2, Bh2_std, sol2, Vh2, Vh2_std, 
    d_max=999.0):
    """ D criterion calculated using Vida et al. 2018 (TBP) which uses the corrected heliocentric velocity
        vector (correction by Sato & Watanabe 2017) and Valsecchi-type D criterion approach of calculating
        the similarity between orbits. The uncertainties are included in calculation.

    Arguments:
        Lh1: [float] Corrected Sun-centred ecliptic longitude of meteor A (radians).
        Bh1: [float] Corrected Sun-centred ecliptic latitude of meteor A (radians).
        sol1: [float] Solar longitude of meteor A (radians).
        Vh1: [float] Heliocentric velocity of meteor A (km/s).
        Lh2: [float] Corrected Sun-centred ecliptic longitude of meteor B (radians).
        Bh2: [float] Corrected Sun-centred ecliptic latitude of meteor B (radians).
        sol2: [float] Solar longitude of meteor B (radians).
        Vh2: [float] Heliocentric velocity of meteor B (km/s).

    Keyword arguments:
        d_max: [float] Maximum values of returned D criteria. This is used for speeding up calculations, as
            some checks can be done before the whole criteria is calculated. Default value is 999.0.
    
    Return:
        [float] Value of calculated D criteria.

    """


    def hyp(x, c):
        """ Hyperbola which is approximating uncertainty influence on the total dissimilarity. """

        return ((math.sqrt(2) - 1)/(2 - math.sqrt(2))**2)*(math.sqrt(x**2 + c**2) - c)



    # Calculate total angular uncertainties
    Lh_std = math.sqrt(Lh1_std**2 + Lh2_std**2)
    Bh_std = math.sqrt(Bh1_std**2 + Bh2_std**2)
    ang_std = math.sqrt((math.sin((Bh1 + Bh2)/2)*Lh_std)**2 + Bh_std**2)


    # Calculate velocity uncertainty
    Vh_std = math.sqrt(Vh1_std**2 + Vh2_std**2)/21.05


    # Define weights
    if ang_std > 0:
        
        # Calculate angle weight w.r.t. uncertainty
        w1 = Vh_std/(math.sqrt(1 - math.cos(ang_std)))

        print('w1', w1)

    else:
        w1 = 2.0


    w2 = 1.0


    # Convert ecliptic angles to velocity vector
    Vx1 = -Vh1*math.cos(Lh1)*math.cos(Bh1)
    Vy1 = -Vh1*math.sin(Lh1)*math.cos(Bh1)
    Vz1 = -Vh1*math.sin(Bh1)

    Vx2 = -Vh2*math.cos(Lh2)*math.cos(Bh2)
    Vy2 = -Vh2*math.sin(Lh2)*math.cos(Bh2)
    Vz2 = -Vh2*math.sin(Bh2)


    # Calculate the dot product between the two vectores
    dot = Vx1*Vx2 + Vy1*Vy2 + Vz1*Vz2

    # Calculate the product of vector magnitudes
    mags = math.sqrt(Vx1**2 + Vy1**2 + Vz1**2)*math.sqrt(Vx2**2 + Vy2**2 + Vz2**2)


    # Velocity component
    v_dissim = abs(Vh1 - Vh2)/21.05
    v_dissim = hyp(v_dissim, Vh_std)

    # Angular components
    ang_dissim = hyp(1.0 - dot/mags, 1.0 - math.cos(ang_std))

    # Solar longitude component
    sol_dissmin = 2*math.sin((sol1 - sol2)/2.0)

    # Calculate the total squared dissimularity
    dissim_2 = v_dissim**2 + (w1*2*ang_dissim) + w2*sol_dissmin**2

    if dissim_2 < 0:
        dissim_2 = 0

    return math.sqrt(dissim_2)




def calcDV(Lh1, Bh1, sol1, Vh1, Lh2, Bh2, sol2, Vh2, d_max=999.0):
    """ D criterion calculated using Vida et al. 2017 (TBP) which uses the corrected heliocentric velocity
        vector (correction by Sato & Watanabe 2017) and Valsecchi-type D criterion approach of calculating
        the similarity between orbits.

    Arguments:
        Lh1: [float] Corrected Sun-centred ecliptic longitude of meteor A (radians).
        Bh1: [float] Corrected Sun-centred ecliptic latitude of meteor A (radians).
        sol1: [float] Solar longitude of meteor A (radians).
        Vh1: [float] Heliocentric velocity of meteor A (km/s).
        Lh2: [float] Corrected Sun-centred ecliptic longitude of meteor B (radians).
        Bh2: [float] Corrected Sun-centred ecliptic latitude of meteor B (radians).
        sol2: [float] Solar longitude of meteor B (radians).
        Vh2: [float] Heliocentric velocity of meteor B (km/s).

    Keyword arguments:
        d_max: [float] Maximum values of returned D criteria. This is used for speeding up calculations, as
            some checks can be done before the whole criteria is calculated. Default value is 999.0.
    
    Return:
        [float] Value of calculated D criteria.

    """

    # Define weights
    w1 = 2.0
    w2 = 1.0


    # Convert ecliptic angles to velocity vector
    Vx1 = -Vh1*math.cos(Lh1)*math.cos(Bh1)
    Vy1 = -Vh1*math.sin(Lh1)*math.cos(Bh1)
    Vz1 = -Vh1*math.sin(Bh1)

    Vx2 = -Vh2*math.cos(Lh2)*math.cos(Bh2)
    Vy2 = -Vh2*math.sin(Lh2)*math.cos(Bh2)
    Vz2 = -Vh2*math.sin(Bh2)


    # Calculate the dot product between the two vectores
    dot = Vx1*Vx2 + Vy1*Vy2 + Vz1*Vz2

    # Calculate the product of vector magnitudes
    mags = math.sqrt(Vx1**2 + Vy1**2 + Vz1**2)*math.sqrt(Vx2**2 + Vy2**2 + Vz2**2)

    # print(math.degrees(math.acos(dot/mags)))
    # print(w1*(1 - abs(dot/mags)))


    # Velocity component
    v_dissim = abs(Vh1 - Vh2)/21.05

    # Angular components
    ang_dissim = 1.0 - dot/mags

    # Solar longitude component
    sol_dissmin = 2*math.sin((sol1 - sol2)/2.0)

    # Calculate the total squared dissimularity
    dissim_2 = v_dissim**2 + (w1*2*ang_dissim) + w2*sol_dissmin**2


    #dissim = (Vh1 - Vh2)**2 + w1*(1 - dot/mags) + w2*(2*math.sin((sol1 - sol2)/2.0))**2

    if dissim_2 < 0:
        dissim_2 = 0


    return math.sqrt(dissim_2)



def _mutualInclinationCos(i1, O1, i2, O2):
    """ Cosine of the mutual inclination of two orbital planes.

    Arguments:
        i1: [float] inclination of the first orbit (rad)
        O1: [float] longitude of ascending node of the first orbit (rad)
        i2: [float] inclination of the second orbit (rad)
        O2: [float] longitude of ascending node of the second orbit (rad)

    Return:
        [float] cos I, clipped to [-1, 1]
    """

    cos_I = np.cos(i1)*np.cos(i2) + np.sin(i1)*np.sin(i2)*np.cos(O1 - O2)

    return np.clip(cos_I, -1.0, 1.0)


def _perihelionDirectionCos(i1, O1, w1, i2, O2, w2):
    """ Cosine of the angle between the perihelion directions of two orbits, i.e. between their
        Laplace-Runge-Lenz vectors.

    Arguments:
        i1: [float] inclination of the first orbit (rad)
        O1: [float] longitude of ascending node of the first orbit (rad)
        w1: [float] argument of perihelion of the first orbit (rad)
        i2: [float] inclination of the second orbit (rad)
        O2: [float] longitude of ascending node of the second orbit (rad)
        w2: [float] argument of perihelion of the second orbit (rad)

    Return:
        [float] cos P, clipped to [-1, 1]
    """

    c1, c2 = np.cos(i1), np.cos(i2)
    s1, s2 = np.sin(i1), np.sin(i2)
    delta = O1 - O2

    cos_P = s1*s2*np.sin(w1)*np.sin(w2) \
        + (np.cos(w1)*np.cos(w2) + c1*c2*np.sin(w1)*np.sin(w2))*np.cos(delta) \
        + (c2*np.cos(w1)*np.sin(w2) - c1*np.sin(w1)*np.cos(w2))*np.sin(delta)

    return np.clip(cos_P, -1.0, 1.0)


def calcRho1(q1, e1, i1, O1, w1, q2, e2, i2, O2, w2, L=1.0):
    """ Calculate the Kholshevnikov et al. (2016) rho_1 distance between two orbits.

        rho_1 is a true metric on the space of Keplerian orbits: unlike D_SH, D_D and D_H it
        satisfies the triangle inequality, and it stays well defined for circular orbits. It is
        built from the difference of the angular momentum vectors and of the eccentricity vectors.

        Reference: Kholshevnikov, Kokhirova, Babadzhanov & Khamroev (2016), MNRAS 462, 2275,
        doi:10.1093/mnras/stw1712.

        No published threshold was found for rho_1. One would not transfer from D_SH or D_D in
        any case: the angular momentum difference has units of length and is made dimensionless by
        dividing by L, so the choice of L fixes how it is weighted against the already
        dimensionless eccentricity term, and hence fixes the numerical scale of rho_1.

    Arguments:
        q1: [float] perihelion distance of the first orbit (AU)
        e1: [float] num. eccentricity of the first orbit
        i1: [float] inclination of the first orbit (rad)
        O1: [float] longitude of ascending node of the first orbit (rad)
        w1: [float] argument of perihelion of the first orbit (rad)
        q2: [float] perihelion distance of the second orbit (AU)
        e2: [float] num. eccentricity of the second orbit
        i2: [float] inclination of the second orbit (rad)
        O2: [float] longitude of ascending node of the second orbit (rad)
        w2: [float] argument of perihelion of the second orbit (rad)

    Keyword arguments:
        L: [float] scale length used to make the angular momentum term dimensionless (AU).
            Default 1 AU, as recommended for the Solar system.

    Return:
        [float] rho_1 value
    """

    p1 = q1*(1.0 + e1)
    p2 = q2*(1.0 + e2)

    cos_I = _mutualInclinationCos(i1, O1, i2, O2)
    cos_P = _perihelionDirectionCos(i1, O1, w1, i2, O2, w2)

    rho_sqr = (p1 + p2 - 2*np.sqrt(p1*p2)*cos_I)/L \
        + (e1**2 + e2**2 - 2*e1*e2*cos_P)

    return np.sqrt(np.maximum(rho_sqr, 0.0))


def calcRho2(q1, e1, i1, O1, w1, q2, e2, i2, O2, w2, L=1.0):
    """ Calculate the Kholshevnikov et al. (2016) rho_2 distance between two orbits.

        rho_2 is a true metric on the space of Keplerian orbits, built from the two orthogonal
        vectors u and v with |u| = sqrt(p) and |v| = e*sqrt(p), so both of its terms carry the
        same units. Like rho_1 it satisfies the triangle inequality and admits circular orbits.

        Reference: Kholshevnikov, Kokhirova, Babadzhanov & Khamroev (2016), MNRAS 462, 2275,
        doi:10.1093/mnras/stw1712.

        No published threshold was found for rho_2. With L = 1 AU it is dimensionless, but its
        numerical scale is set by that choice, so D_SH and D_D thresholds do not carry over.

    Arguments:
        q1: [float] perihelion distance of the first orbit (AU)
        e1: [float] num. eccentricity of the first orbit
        i1: [float] inclination of the first orbit (rad)
        O1: [float] longitude of ascending node of the first orbit (rad)
        w1: [float] argument of perihelion of the first orbit (rad)
        q2: [float] perihelion distance of the second orbit (AU)
        e2: [float] num. eccentricity of the second orbit
        i2: [float] inclination of the second orbit (rad)
        O2: [float] longitude of ascending node of the second orbit (rad)
        w2: [float] argument of perihelion of the second orbit (rad)

    Keyword arguments:
        L: [float] scale length used to normalise the metric (AU). Default 1 AU.

    Return:
        [float] rho_2 value
    """

    p1 = q1*(1.0 + e1)
    p2 = q2*(1.0 + e2)

    cos_I = _mutualInclinationCos(i1, O1, i2, O2)
    cos_P = _perihelionDirectionCos(i1, O1, w1, i2, O2, w2)

    rho_sqr = ((1.0 + e1**2)*p1 + (1.0 + e2**2)*p2
        - 2*np.sqrt(p1*p2)*(cos_I + e1*e2*cos_P))/L

    return np.sqrt(np.maximum(rho_sqr, 0.0))


def calcRho5(q1, e1, i1, O1, w1, q2, e2, i2, O2, w2, L=1.0):
    """ Calculate the Kholshevnikov et al. (2016) rho_5 distance between two orbits.

        rho_5 is the minimum of rho_2 over both nodes and both arguments of perihelion, so it is a
        metric on the quotient space in which orbits differing only by a rotation about the
        ecliptic pole and by an apsidal rotation are identified. It measures how close two orbits
        could be brought by precession alone, which is the relevant comparison for streams whose
        nodes and apsides have had time to circulate.

        The node and the argument of perihelion are therefore not used, but they are kept in the
        signature so that rho_5 can be substituted for calcDSH without changing the call.

        Reference: Kholshevnikov, Kokhirova, Babadzhanov & Khamroev (2016), MNRAS 462, 2275,
        doi:10.1093/mnras/stw1712.

        No published threshold was found for rho_5.

    Arguments:
        q1: [float] perihelion distance of the first orbit (AU)
        e1: [float] num. eccentricity of the first orbit
        i1: [float] inclination of the first orbit (rad)
        O1: [float] longitude of ascending node of the first orbit (rad), not used
        w1: [float] argument of perihelion of the first orbit (rad), not used
        q2: [float] perihelion distance of the second orbit (AU)
        e2: [float] num. eccentricity of the second orbit
        i2: [float] inclination of the second orbit (rad)
        O2: [float] longitude of ascending node of the second orbit (rad), not used
        w2: [float] argument of perihelion of the second orbit (rad), not used

    Keyword arguments:
        L: [float] scale length used to normalise the metric (AU). Default 1 AU.

    Return:
        [float] rho_5 value
    """

    p1 = q1*(1.0 + e1)
    p2 = q2*(1.0 + e2)

    rho_sqr = ((1.0 + e1**2)*p1 + (1.0 + e2**2)*p2
        - 2*np.sqrt(p1*p2)*(e1*e2 + np.cos(i1 - i2)))/L

    return np.sqrt(np.maximum(rho_sqr, 0.0))


def calcC(q1, e1, i1, O1, w1, q2, e2, i2, O2, w2):
    """ Calculate the Neslusan (2002) C criterion, the length of the difference of the two orbital
        angular momentum vectors per unit mass.

        The criterion compares orbital planes and sizes only; it carries no information about the
        apsidal orientation, and nothing in it is specific to meteor showers.

        Units are Gaussian, with the solar gravitational parameter taken as unity, so that the
        magnitude of the angular momentum vector is sqrt(p) with p in AU.

        Reference: Neslusan, in Dynamics of Natural and Artificial Celestial Bodies, the
        proceedings of the US/European Celestial Mechanics Workshop held in Poznan in July 2000,
        365. The volume is dated 2002 in some citations and 2001 in others, including the reference
        list of Jopek, Rudawska & Bartczak (2008), which also describes the criterion as the
        difference of the orbital momentum vectors per unit mass, as implemented here.

        That volume was not accessible, so whether it publishes a threshold is unchecked. None is
        supplied here.

    Arguments:
        q1: [float] perihelion distance of the first orbit (AU)
        e1: [float] num. eccentricity of the first orbit
        i1: [float] inclination of the first orbit (rad)
        O1: [float] longitude of ascending node of the first orbit (rad)
        w1: [float] argument of perihelion of the first orbit (rad), not used
        q2: [float] perihelion distance of the second orbit (AU)
        e2: [float] num. eccentricity of the second orbit
        i2: [float] inclination of the second orbit (rad)
        O2: [float] longitude of ascending node of the second orbit (rad)
        w2: [float] argument of perihelion of the second orbit (rad), not used

    Return:
        [float] C value (sqrt(AU))
    """

    # |c| = sqrt(p) rather than k*sqrt(p), i.e. the Gaussian constant is left out and the result
    #   carries units of sqrt(AU). calcDVJopek builds the same vector with the k factor included,
    #   so the two are on scales differing by k = 0.0172. Neslusan's own paper could not be read to
    #   settle which convention it uses; since the criterion is applied with the break-point method
    #   rather than a fixed threshold, the scale does not affect a search, but it does mean a C
    #   value from here cannot be compared against one quoted elsewhere without checking.
    c1 = np.sqrt(q1*(1.0 + e1))
    c2 = np.sqrt(q2*(1.0 + e2))

    cx1, cy1, cz1 = c1*np.sin(i1)*np.sin(O1), -c1*np.sin(i1)*np.cos(O1), c1*np.cos(i1)
    cx2, cy2, cz2 = c2*np.sin(i2)*np.sin(O2), -c2*np.sin(i2)*np.cos(O2), c2*np.cos(i2)

    return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2 + (cz1 - cz2)**2)


def calcDR(ra1, dec1, sol1, vg1, ra2, dec2, sol2, vg2, w1=1.0):
    """ Calculate the reduced Valsecchi et al. (1999) D_R criterion between two orbits.

        D_R keeps only the two terms of D_N that are nearly invariant under the principal secular
        perturbation of meteoroid orbits, the circulation of the argument of perihelion, and drops
        the terms in the angle phi and the solar longitude. It is therefore a necessary but not a
        sufficient condition for membership of the same stream.

        Unlike the other criteria added here this one takes scalars only, because
        calcVgComponents, which it shares with calcDN, is written with the math module.

        Reference: Valsecchi, Jopek & Froeschle (1999), MNRAS 304, 743.

        That paper recommends no threshold for either D_N or D_R. Jenniskens (2008) reports
        D_N < 0.20 as the value at which association was implicated; since D_R <= D_N, that is a
        necessary condition on D_R rather than a threshold for it.

    Arguments:
        ra1: [float] right ascension of the first radiant (rad)
        dec1: [float] declination of the first radiant (rad)
        sol1: [float] solar longitude of the first orbit (rad)
        vg1: [float] geocentric velocity of the first orbit (km/s)
        ra2: [float] right ascension of the second radiant (rad)
        dec2: [float] declination of the second radiant (rad)
        sol2: [float] solar longitude of the second orbit (rad)
        vg2: [float] geocentric velocity of the second orbit (km/s)

    Keyword arguments:
        w1: [float] weight of the cos(theta) term, not an argument of perihelion despite carrying
            the name this module uses for one elsewhere. The paper leaves the weights undefined and
            uses unity throughout its application.

    Return:
        [float] D_R value
    """

    _, vg_y1, _ = calcVgComponents(ra1, dec1, sol1, vg1)
    _, vg_y2, _ = calcVgComponents(ra2, dec2, sol2, vg2)

    # U is the geocentric velocity in units of the Earth's orbital speed
    u1 = vg1/SPEED_EARTH
    u2 = vg2/SPEED_EARTH

    cos_theta1 = vg_y1/u1
    cos_theta2 = vg_y2/u2

    return np.sqrt((u2 - u1)**2 + w1*(cos_theta2 - cos_theta1)**2)


# Dispersions of the three D_B invariants over the sample of likely stream and parent-body pairs
#   of Jenniskens (2008), tables 1 and 2. C3 is an angle [rad]
DB_SIGMA_C1 = 0.13
DB_SIGMA_C2 = 0.06
DB_SIGMA_C3 = np.radians(14.2)


def calcDB(e1, i1, O1, w1, e2, i2, O2, w2):
    """ Calculate the Jenniskens (2008) D_B criterion between two orbits.

        D_B compares three quantities that are near-invariant under the secular perturbations of a
        short-period orbit over one nutation cycle, so it asks whether two orbits could have been
        the same orbit recently, rather than whether they are the same orbit now. C1 follows from
        the z-component of the angular momentum and the energy, C2 is the Lidov (1961, 1962)
        integral of the twice-averaged three-body problem, and C3 is the longitude of perihelion,
        which drifts far more slowly than either angle alone. The three are taken from
        Babadzhanov (1989).

        Each difference is divided by the dispersion of that quantity over the paper's sample of
        likely stream and parent-body pairs, 0.13, 0.06 and 14.2 deg, available here as
        DB_SIGMA_C1, DB_SIGMA_C2 and DB_SIGMA_C3.

        The paper notes, and does not correct for, the fact that C1, C2 and C3 are not orthogonal.

        Published thresholds: D_B < 1.0 together with D_T < 0.3 identifies parent bodies and
        siblings of a stream; D_B < 1.5 with D_T < 0.6 also captures bodies related through an
        earlier fragmentation.

        Reference: Jenniskens (2008), Icarus 194, 13, eq. 18, doi:10.1016/j.icarus.2007.09.016.

    Arguments:
        e1: [float] num. eccentricity of the first orbit
        i1: [float] inclination of the first orbit (rad)
        O1: [float] longitude of ascending node of the first orbit (rad)
        w1: [float] argument of perihelion of the first orbit (rad)
        e2: [float] num. eccentricity of the second orbit
        i2: [float] inclination of the second orbit (rad)
        O2: [float] longitude of ascending node of the second orbit (rad)
        w2: [float] argument of perihelion of the second orbit (rad)

    Return:
        [float] D_B value
    """

    c1_1 = (1.0 - e1**2)*np.cos(i1)**2
    c1_2 = (1.0 - e2**2)*np.cos(i2)**2

    c2_1 = e1**2*(0.4 - np.sin(i1)**2*np.sin(w1)**2)
    c2_2 = e2**2*(0.4 - np.sin(i2)**2*np.sin(w2)**2)

    # C3 is an angle, so the smallest of the two differences around the circle is the relevant one
    d_c3 = (w1 + O1) - (w2 + O2)
    d_c3 = (d_c3 + np.pi)%(2*np.pi) - np.pi

    return np.sqrt(((c1_1 - c1_2)/DB_SIGMA_C1)**2 + ((c2_1 - c2_2)/DB_SIGMA_C2)**2
        + (d_c3/DB_SIGMA_C3)**2)


def calcDT(q1, e1, i1, q2, e2, i2, a_planet=A_JUPITER):
    """ Calculate the Jenniskens (2008) D_T criterion between two orbits, the absolute difference
        of their Tisserand parameters.

        The Tisserand parameter is conserved under the same secular perturbations as the D_B
        invariants, so D_T asks the same question as D_B along a different axis.

        The paper writes the Tisserand parameter in terms of the perihelion distance and the
        eccentricity rather than the semi-major axis, because the observational errors in q and e
        are smaller than those in a. That form is used here. It is algebraically the same quantity
        as calcTisserand returns, and stays finite as the eccentricity approaches 1, where the
        semi-major axis form evaluates to nan.

        Published thresholds: D_T < 0.3 together with D_B < 1.0 identifies parent bodies and
        siblings of a stream; D_T < 0.6 with D_B < 1.5 also captures bodies related through an
        earlier fragmentation.

        Reference: Jenniskens (2008), Icarus 194, 13, eqs 13 and 14,
        doi:10.1016/j.icarus.2007.09.016.

    Arguments:
        q1: [float] perihelion distance of the first orbit (AU)
        e1: [float] num. eccentricity of the first orbit
        i1: [float] inclination of the first orbit (rad)
        q2: [float] perihelion distance of the second orbit (AU)
        e2: [float] num. eccentricity of the second orbit
        i2: [float] inclination of the second orbit (rad)

    Keyword arguments:
        a_planet: [float] semi-major axis of the perturbing planet (AU). Default Jupiter.

    Return:
        [float] D_T value
    """

    def tisserandFromPerihelion(q, e, i):
        return a_planet*(1.0 - e)/q + 2*np.cos(i)*np.sqrt(q*(1.0 + e)/a_planet)

    return np.abs(tisserandFromPerihelion(q1, e1, i1) - tisserandFromPerihelion(q2, e2, i2))


# Weights of the D_X terms, as given by Rudawska et al. (2015). The paper describes them as
#   normalising each term's contribution; measured over pairs drawn within a shower from the
#   dispersions of its table 2, the right ascension term still dominates, so they are reproduced
#   here as published values rather than as a normalisation that can be relied on
DX_W_SOL = 0.17
DX_W_RA = 1.20
DX_W_DEC = 1.20
DX_W_VG = 0.20


def calcDX(ra1, dec1, sol1, vg1, ra2, dec2, sol2, vg2, w_sol=DX_W_SOL, w_ra=DX_W_RA,
    w_dec=DX_W_DEC, w_vg=DX_W_VG):
    """ Calculate the Rudawska et al. (2015) D_X criterion between two orbits.

        D_X compares the geocentric quantities directly, which avoids propagating the velocity
        uncertainty into the semi-major axis. The radiant terms are scaled by the velocity
        difference, so a pair of meteors with similar radiants but different speeds is separated.

        The velocity difference enters the radiant terms additively as |Vg1 - Vg2| + 1, so the
        criterion is not invariant to the unit of the geocentric velocity. The paper tabulates Vg
        in km/s, which is the unit assumed here.

        Note that the criterion is not symmetric: the right ascension term is scaled by the cosine
        of the first declination and the velocity term by the first velocity, so exchanging the two
        orbits changes the result slightly. The paper applies it to a group mean against a group
        mean, where the asymmetry is immaterial.

        The arguments are ordered as in calcDN, calcDR and calcDV rather than as in the paper's
        equation, so that the geocentric criteria in this module can be called interchangeably.

        Published threshold: groups are merged when D_X <= 0.15.

        Reference: Rudawska, Matlovic, Toth & Kornos (2015), P&SS 118, 38, eq. 2,
        doi:10.1016/j.pss.2015.07.011.

    Arguments:
        ra1: [float] right ascension of the first radiant (rad)
        dec1: [float] declination of the first radiant (rad)
        sol1: [float] solar longitude of the first orbit (rad)
        vg1: [float] geocentric velocity of the first orbit (km/s)
        ra2: [float] right ascension of the second radiant (rad)
        dec2: [float] declination of the second radiant (rad)
        sol2: [float] solar longitude of the second orbit (rad)
        vg2: [float] geocentric velocity of the second orbit (km/s)

    Keyword arguments:
        w_sol: [float] weight of the solar longitude term. Default as published.
        w_ra: [float] weight of the right ascension term. Default as published.
        w_dec: [float] weight of the declination term. Default as published.
        w_vg: [float] weight of the geocentric velocity term. Default as published.

    Return:
        [float] D_X value
    """

    d_vg = np.abs(vg1 - vg2)

    term_sol = w_sol*(2*np.sin((sol1 - sol2)/2.0))**2
    # cos(dec1) scales the chord, it is not part of the angle: the paper writes the term as
    #   [2 sin((ra1 - ra2)/2) cos(dec1)]^2. The two forms agree to first order in the radiant
    #   separation, which is why a within-shower comparison cannot tell them apart, but they
    #   differ by 41% at 180 deg of right ascension and 60 deg of declination.
    term_ra = w_ra*(d_vg + 1.0)*(2*np.sin((ra1 - ra2)/2.0)*np.cos(dec1))**2
    term_dec = w_dec*(d_vg + 1.0)*(2*np.sin(np.abs(dec1 - dec2)/2.0))**2
    term_vg = w_vg*(d_vg/vg1)**2

    return np.sqrt(term_sol + term_ra + term_dec + term_vg)


# Standard deviations of the vectorial elements within a stream, table 1 of Jopek, Rudawska &
#   Bartczak (2008), as (angular momentum triple, eccentricity vector triple, energy), in units of
#   AU, day and solar masses. Keyed by the age of the stream in years, plus the sporadic background
#   measured from the IAU 2003 data
DV_DISPERSIONS = {
    0: ((2.5e-5, 2.4e-5, 2.8e-5), (2.8e-3, 2.8e-3, 2.0e-3), 7.1e-7),
    2000: ((3.9e-4, 3.4e-4, 1.7e-4), (6.6e-3, 7.1e-3, 1.3e-2), 7.6e-7),
    4000: ((9.8e-4, 5.9e-4, 4.1e-4), (1.1e-2, 1.6e-2, 2.3e-2), 9.8e-7),
    5000: ((1.3e-3, 6.8e-4, 5.1e-4), (1.4e-2, 2.0e-2, 2.7e-2), 1.1e-6),
    6000: ((1.5e-3, 7.9e-4, 6.2e-4), (1.8e-2, 2.3e-2, 3.1e-2), 1.2e-6),
    'sporadic': ((6.8e-3, 7.8e-3, 1.2e-2), (5.1e-1, 4.9e-1, 3.4e-1), 4.3e-4),
    }

# Age of the stream whose dispersions the paper used for its own search
DV_DEFAULT_EPOCH = 4000

# Thresholds at the 99% reliability level, table 2 of the same paper, keyed by the smallest stream
#   size accepted. The column is headed "D_V x 10^-1", so the tabulated figures are scaled up here
DV_THRESHOLDS = {
    8: 2.414, 9: 2.575, 10: 2.707, 11: 2.815, 12: 2.906, 13: 2.985, 14: 3.057, 15: 3.128,
    }


def calcDVWeights(epoch=DV_DEFAULT_EPOCH):
    """ Calculate the D_V weights from the dispersions of the vectorial elements within a stream.

        A pair of orbits differing by twice the dispersion in a single element contributes exactly
        1 to the sum, which is what sets the scale of the criterion.

        Reference: Jopek, Rudawska & Bartczak (2008), EM&P 102, 73, eq. 5,
        doi:10.1007/s11038-007-9197-8.

    Keyword arguments:
        epoch: [int] age of the stream in years, one of the keys of DV_DISPERSIONS, or the string
            'sporadic' for the background. Default 4000, which the paper used for its own search.

    Return:
        [tuple] the three angular momentum weights, the three eccentricity vector weights, and the
            energy weight
    """

    if epoch not in DV_DISPERSIONS:
        raise ValueError("No dispersions published for epoch {!r}. Available: {!s}.".format(epoch,
            sorted(DV_DISPERSIONS, key=str)))

    sigma_h, sigma_e, sigma_energy = DV_DISPERSIONS[epoch]

    return ([1.0/(2*s)**2 for s in sigma_h], [1.0/(2*s)**2 for s in sigma_e],
        1.0/(2*sigma_energy)**2)


def calcDVJopek(q1, e1, i1, O1, w1, q2, e2, i2, O2, w2, w_h=None, w_e=None, w_E=None):
    """ Calculate the Jopek, Rudawska & Bartczak (2008) D_V criterion between two orbits.

        D_V compares the two vectorial integrals of the two-body problem, the angular momentum and
        the eccentricity vector, together with the orbital energy. Working with the vectors rather
        than with the angles avoids the branch-cut and circular-orbit problems of the D_SH family.

        The weights are the reciprocal squared dispersions of each element within a stream,
        w = (2*sigma)^-2, so a pair differing by twice the dispersion in one element contributes
        exactly 1. They default to the dispersions the paper measured for a stream 4000 years after
        formation, which is the set it used for its own search; calcDVWeights returns the set for
        any of the tabulated ages, or for the sporadic background.

        Units are AU, day and solar masses, so the angular momentum is in AU^2/day and the energy
        in AU^2/day^2. The weights are dimensional, so they are only meaningful in these units.

        This is a different criterion from calcDV in this module, which implements the unpublished
        Vida criterion; calcDV is left untouched.

        The factors of 1.5 on the third angular momentum component and 2 on the energy are
        deliberate: those two are the invariant and semi-invariant parts of the set, so the paper
        weights them up beyond what their dispersions alone would give.

        Published thresholds, at the 99% reliability level, are in DV_THRESHOLDS, keyed by the
        smallest stream size accepted. They run from 2.414 for groups of 8 to 3.128 for groups of
        15.

        Reference: Jopek, Rudawska & Bartczak (2008), EM&P 102, 73, eqs 1 to 5,
        doi:10.1007/s11038-007-9197-8.

    Arguments:
        q1: [float] perihelion distance of the first orbit (AU)
        e1: [float] num. eccentricity of the first orbit
        i1: [float] inclination of the first orbit (rad)
        O1: [float] longitude of ascending node of the first orbit (rad)
        w1: [float] argument of perihelion of the first orbit (rad)
        q2: [float] perihelion distance of the second orbit (AU)
        e2: [float] num. eccentricity of the second orbit
        i2: [float] inclination of the second orbit (rad)
        O2: [float] longitude of ascending node of the second orbit (rad)
        w2: [float] argument of perihelion of the second orbit (rad)

    Keyword arguments:
        w_h: [list] three weights of the angular momentum components. Default as published.
        w_e: [list] three weights of the eccentricity vector components. Default as published.
        w_E: [float] weight of the energy term. Default as published.

    Return:
        [float] D_V value
    """

    if w_h is None or w_e is None or w_E is None:

        default_h, default_e, default_energy = calcDVWeights()

        w_h = default_h if w_h is None else w_h
        w_e = default_e if w_e is None else w_e
        w_E = default_energy if w_E is None else w_E

    p1 = q1*(1.0 + e1)
    p2 = q2*(1.0 + e2)

    h1, h2 = GAUSS_K*np.sqrt(p1), GAUSS_K*np.sqrt(p2)

    hx1, hy1, hz1 = h1*np.sin(i1)*np.sin(O1), -h1*np.sin(i1)*np.cos(O1), h1*np.cos(i1)
    hx2, hy2, hz2 = h2*np.sin(i2)*np.sin(O2), -h2*np.sin(i2)*np.cos(O2), h2*np.cos(i2)

    ex1 = e1*(np.cos(O1)*np.cos(w1) - np.sin(O1)*np.sin(w1)*np.cos(i1))
    ey1 = e1*(np.sin(O1)*np.cos(w1) + np.cos(O1)*np.sin(w1)*np.cos(i1))
    ez1 = e1*np.sin(w1)*np.sin(i1)

    ex2 = e2*(np.cos(O2)*np.cos(w2) - np.sin(O2)*np.sin(w2)*np.cos(i2))
    ey2 = e2*(np.sin(O2)*np.cos(w2) + np.cos(O2)*np.sin(w2)*np.cos(i2))
    ez2 = e2*np.sin(w2)*np.sin(i2)

    # Energy of an orbit with q and e, i.e. -mu/(2a), written so it stays finite as e approaches 1
    en1 = -GAUSS_K_SQUARED*(1.0 - e1)/(2.0*q1)
    en2 = -GAUSS_K_SQUARED*(1.0 - e2)/(2.0*q2)

    return np.sqrt(w_h[0]*(hx1 - hx2)**2 + w_h[1]*(hy1 - hy2)**2 + 1.5*w_h[2]*(hz1 - hz2)**2
        + w_e[0]*(ex1 - ex2)**2 + w_e[1]*(ey1 - ey2)**2 + w_e[2]*(ez1 - ez2)**2
        + 2*w_E*(en1 - en2)**2)


# Reference orbit of the Taurid Complex core used by Asher, Clube & Steel (1993) [AU, -, rad]
TC_REFERENCE_A = 2.1
TC_REFERENCE_E = 0.82
TC_REFERENCE_INCL = np.radians(4.0)

# Perihelion distance of the same reference orbit as quoted by Steel, Asher & Clube (1991) [AU]
TC_REFERENCE_Q = 0.375

# Scale normalising the semi-major axis term of D_ACS [AU]. Asher, Clube & Steel (1993) print it
#   in eq. 2 itself, which reads D^2 = ((a1 - a2)/3)^2 + (e1 - e2)^2 + (2 sin((i1 - i2)/2))^2, so it
#   is a published constant and not one inferred from their table 1
DACS_A_SCALE = 3.0


def calcDACS(a1, e1, i1, a2, e2, i2):
    """ Calculate the Asher, Clube & Steel (1993) D_ACS criterion between two orbits.

        D_ACS compares only the size, shape and inclination of the two orbits. It carries no node
        or longitude term by design: the Taurid Complex has been dispersed in longitude of
        perihelion by Jovian perturbations, so a longitude term appropriate to a narrow stream
        would dominate the sum. Longitude alignment is instead tested separately, after the
        criterion has selected on (a, e, i).

        The criterion is normally evaluated against the Taurid Complex core orbit, available here
        as TC_REFERENCE_A, TC_REFERENCE_E and TC_REFERENCE_INCL.

        Published thresholds: D = 0.15 restricts the selection to the core of the complex, as used
        with the perihelion-distance form in Steel, Asher & Clube (1991); Asher, Clube & Steel
        (1993) suggest D of about 0.2 as the value that best defines Taurid Complex asteroids.

        Note that the paper does not feed observed elements into the criterion. The inclination
        varies by a factor of a few over 10**3 yr, so it is first adjusted by Brouwer (1947) secular
        perturbation theory to the smallest value the orbit ever reaches, and the eccentricity is
        adjusted likewise, which moves D by less than 0.01 in nearly all cases. No such adjustment
        is applied here, so passing observed elements will not reproduce the paper's table.

        The companion calcDSAC is the earlier form of Steel, Asher & Clube (1991), which uses the
        perihelion distance in place of the semi-major axis. That form suits meteoroids, whose q is
        better determined than their a; this one suits asteroids, whose a is well determined.

        Reference: Asher, Clube & Steel (1993), MNRAS 264, 93, eq. 2, doi:10.1093/mnras/264.1.93.

    Arguments:
        a1: [float] semi-major axis of the first orbit (AU)
        e1: [float] num. eccentricity of the first orbit
        i1: [float] inclination of the first orbit (rad)
        a2: [float] semi-major axis of the second orbit (AU)
        e2: [float] num. eccentricity of the second orbit
        i2: [float] inclination of the second orbit (rad)

    Return:
        [float] D_ACS value
    """

    return np.sqrt(((a1 - a2)/DACS_A_SCALE)**2 + (e1 - e2)**2 + (2*np.sin((i1 - i2)/2.0))**2)


def calcDSAC(q1, e1, i1, q2, e2, i2):
    """ Calculate the Steel, Asher & Clube (1991) Taurid Complex criterion between two orbits.

        This is the perihelion-distance form of the criterion, and the earlier of the two. It is
        the one to use for meteoroids, whose perihelion distance is better determined than their
        semi-major axis, since a carries the full weight of the uncertainty in the meteoroid
        velocity. For asteroids, whose a is well determined, use calcDACS instead, which is the
        same expression with a scaled semi-major axis term in place of the perihelion term.

        Like calcDACS it carries no node or longitude term, for the same reason: the Taurid Complex
        is dispersed in longitude of perihelion, so a longitude term appropriate to a narrow stream
        would dominate the sum.

        The reference orbit is the Taurid Complex core, available as TC_REFERENCE_Q,
        TC_REFERENCE_E and TC_REFERENCE_INCL.

        The perihelion term carries no scale factor, so it is in AU while the other two terms are
        dimensionless, and is implicitly divided by 1 AU as in D_SH.

        Published threshold: D = 0.15, which restricts the selection to the core of the complex.

        Unlike calcDACS, the inclination needs no secular adjustment when this form is applied to
        meteoroids: an orbit has to cross the Earth's to produce a meteor, which already constrains
        the inclination to be low.

        Reference: Steel, Asher & Clube (1991), MNRAS 251, 632, as eq. 1 of Asher, Clube & Steel
        (1993), MNRAS 264, 93, doi:10.1093/mnras/264.1.93.

    Arguments:
        q1: [float] perihelion distance of the first orbit (AU)
        e1: [float] num. eccentricity of the first orbit
        i1: [float] inclination of the first orbit (rad)
        q2: [float] perihelion distance of the second orbit (AU)
        e2: [float] num. eccentricity of the second orbit
        i2: [float] inclination of the second orbit (rad)

    Return:
        [float] D_SAC value
    """

    return np.sqrt((q1 - q2)**2 + (e1 - e2)**2 + (2*np.sin((i1 - i2)/2.0))**2)



if __name__ == "__main__":

    import os
    import sys
    import argparse

    import numpy as np

    from wmpl.Utils.OrbitClassification import (calcTisserand, calcKresakK, calcKresakP,
        calcAphelionDistance, isCometaryQi, isCometaryEi, isCometaryKi, isCometaryPi,
        classifyTancrediComet)
    from wmpl.Utils.Pickling import loadPickle


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Compare two trajectory files by computing their D criteria.")

    arg_parser.add_argument('traj_path', metavar='TRAJ_PATH', type=str, \
        help="Path to a trajectory pickle file.")

    arg_parser.add_argument('traj_path2', metavar='TRAJ_PATH2', type=str, nargs='?', \
        default=None, \
        help="Path to an optional second trajectory pickle file. If it's not given, then the orbital parameters need to be specified manually.")

    arg_parser.add_argument('-q', '--q', metavar='PERIHELION_DIST', help="Perihelion distance in AU.", \
        type=float)

    arg_parser.add_argument('-e', '--e', metavar='ECCENTRICITY', help="Eccentricity.", \
        type=float)

    arg_parser.add_argument('-i', '--i', metavar='INCLINATION', help="Inclination (deg).", \
        type=float)

    arg_parser.add_argument('-p', '--peri', metavar='ARG_OF_PERI', help="Argument of perihelion (deg).", \
        type=float)

    arg_parser.add_argument('-n', '--node', metavar='ASCENDING_NODE', help="Ascending node (deg).", \
        type=float)

    arg_parser.add_argument('--ra', metavar='RA_G', type=float, \
        help="Right ascension of the geocentric radiant (deg). Needed together with --dec, --sol \
and --vg for the criteria defined on the radiant.")

    arg_parser.add_argument('--dec', metavar='DEC_G', type=float, \
        help="Declination of the geocentric radiant (deg).")

    arg_parser.add_argument('--sol', metavar='SOLAR_LON', type=float, \
        help="Solar longitude (deg).")

    arg_parser.add_argument('--vg', metavar='V_G', type=float, \
        help="Geocentric velocity (km/s).")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    print("Input files:")
    print("First  =", cml_args.traj_path)
    print("Second =", cml_args.traj_path2)

    def geocentricFromOrbit(orbit):
        """ Pull the radiant, the solar longitude and the geocentric speed out of an orbit, in the
            units the criteria take. The orbit stores speeds in m/s and the criteria expect km/s.

        Arguments:
            orbit: [Orbit] orbit of a trajectory

        Return:
            [tuple] (ra, dec, solar longitude, speed) in radians and km/s, or None if the orbit
                does not carry all four
        """

        values = (orbit.ra_g, orbit.dec_g, orbit.la_sun, orbit.v_g)

        if any(v is None for v in values):
            return None

        return (orbit.ra_g, orbit.dec_g, orbit.la_sun, orbit.v_g/1000.0)


    def heliocentricFromOrbit(orbit):
        """ The same for the corrected heliocentric direction and speed, which the Vida criterion
            takes.

        Arguments:
            orbit: [Orbit] orbit of a trajectory

        Return:
            [tuple] (ecliptic longitude, ecliptic latitude, solar longitude, speed) in radians and
                km/s, or None if the orbit does not carry all four
        """

        values = (orbit.L_h, orbit.B_h, orbit.la_sun, orbit.v_h)

        if any(v is None for v in values):
            return None

        return (orbit.L_h, orbit.B_h, orbit.la_sun, orbit.v_h/1000.0)


    # Load the reference trajectory pickle file
    traj_ref = loadPickle(*os.path.split(cml_args.traj_path))

    # Load orbital elements
    q_ref = traj_ref.orbit.q
    e_ref = traj_ref.orbit.e
    incl_ref = np.degrees(traj_ref.orbit.i)
    peri_ref = np.degrees(traj_ref.orbit.peri)
    node_ref = np.degrees(traj_ref.orbit.node)

    # Quantities the criteria defined on the radiant need
    geo_ref = geocentricFromOrbit(traj_ref.orbit)
    helio_ref = heliocentricFromOrbit(traj_ref.orbit)



    # If the trajectory pickle was given as a second argument, load the orbital elements from it
    if cml_args.traj_path2 is not None:

        # Load the trajectory pickle
        traj = loadPickle(*os.path.split(cml_args.traj_path2))

        # Load orbital elements
        q = traj.orbit.q
        e = traj.orbit.e
        incl = np.degrees(traj.orbit.i)
        peri = np.degrees(traj.orbit.peri)
        node = np.degrees(traj.orbit.node)

        geo_cmp = geocentricFromOrbit(traj.orbit)
        helio_cmp = heliocentricFromOrbit(traj.orbit)


    # Otherwise, load orbital elements from the manual entry
    else:

        # Check that all elements are given
        if (cml_args.q is not None) and (cml_args.e is not None) and (cml_args.i is not None) \
            and (cml_args.peri is not None) and (cml_args.node is not None):

            q = cml_args.q
            e = cml_args.e
            incl = cml_args.i
            peri = cml_args.peri
            node = cml_args.node

        else:
            print("All orbital elements need to be specified: q, e, i, peri, node!")
            sys.exit()

        # The Vida criterion needs a heliocentric direction, which is not a command line option
        helio_cmp = None

        # The radiant quantities are optional, and all four are needed together
        geocentric_args = (cml_args.ra, cml_args.dec, cml_args.sol, cml_args.vg)

        if all(v is not None for v in geocentric_args):
            geo_cmp = (np.radians(cml_args.ra), np.radians(cml_args.dec),
                np.radians(cml_args.sol), cml_args.vg)

        else:
            geo_cmp = None

            if any(v is not None for v in geocentric_args):
                print("--ra, --dec, --sol and --vg have to be given together, ignoring them.")



    # Print reference orbital elements
    print("Reference orbital elements:")
    print("  q = {:.5f} AU".format(q_ref))
    print("  e = {:.5f}".format(e_ref))
    print("  i = {:.5f} deg".format(incl_ref))
    print("  w = {:.5f} deg".format(peri_ref))
    print("  O = {:.5f} deg".format(node_ref))
    print()
    print("Comparison orbital elements:")
    print("  q = {:.5f} AU".format(q))
    print("  e = {:.5f}".format(e))
    print("  i = {:.5f} deg".format(incl))
    print("  w = {:.5f} deg".format(peri))
    print("  O = {:.5f} deg".format(node))
    print()



    # Orbital elements of both orbits, in the units the criteria take
    orbit_ref = (q_ref, e_ref, np.radians(incl_ref), np.radians(node_ref), np.radians(peri_ref))
    orbit_cmp = (q, e, np.radians(incl), np.radians(node), np.radians(peri))

    # The semi-major axis, which two of the criteria and all of the classification take
    a_ref = q_ref/(1.0 - e_ref)
    a_cmp = q/(1.0 - e)

    print()
    print("Criteria on the orbital elements")
    print("--------------------------------")

    for name, value, threshold in (
        ("D_SH", calcDSH(*(orbit_ref + orbit_cmp)), "0.15 for a sample of 359 orbits, Lindblad (1971)"),
        ("D_D", calcDD(*(orbit_ref + orbit_cmp)), "see Dthresholds, scales with the sample size"),
        ("D_H", calcDH(*(orbit_ref + orbit_cmp)), "see Dthresholds, scales with the sample size"),
        ("rho_1", calcRho1(*(orbit_ref + orbit_cmp)), "none published"),
        ("rho_2", calcRho2(*(orbit_ref + orbit_cmp)), "none published"),
        ("rho_5", calcRho5(*(orbit_ref + orbit_cmp)), "none published"),
        ("C", calcC(*(orbit_ref + orbit_cmp)), "none published, used with the break-point method"),
        ("D_V", calcDVJopek(*(orbit_ref + orbit_cmp)), "2.414 to 3.128, Jopek et al. (2008)"),
        ("D_B", calcDB(*(orbit_ref[1:] + orbit_cmp[1:])), "< 1.0 with D_T < 0.3, Jenniskens (2008)"),
        ("D_T", calcDT(q_ref, e_ref, np.radians(incl_ref), q, e, np.radians(incl)), \
            "< 0.3 with D_B < 1.0, Jenniskens (2008)"),
        ("D_ACS", calcDACS(a_ref, e_ref, np.radians(incl_ref), a_cmp, e, np.radians(incl)), \
            "0.15 for the core, about 0.2 defines the complex"),
        ("D_SAC", calcDSAC(q_ref, e_ref, np.radians(incl_ref), q, e, np.radians(incl)), \
            "0.15 for the core of the complex"),
        ):

        print("  {:<6s} = {:9.4f}   threshold: {:s}".format(name, float(value), threshold))

    # The criteria defined on the radiant need the geocentric quantities, which come either from a
    #   second trajectory file or from the command line
    if (geo_ref is not None) and (geo_cmp is not None):

        print()
        print("Criteria on the geocentric radiant and speed")
        print("-------------------------------------------")

        ra_ref, dec_ref, sol_ref, vg_ref = geo_ref
        ra_cmp, dec_cmp, sol_cmp, vg_cmp = geo_cmp

        geo_args = (ra_ref, dec_ref, sol_ref, vg_ref, ra_cmp, dec_cmp, sol_cmp, vg_cmp)

        for name, value, threshold in (
            ("D_N", calcDN(*geo_args), "0.20, as quoted by Jenniskens (2008)"),
            ("D_R", calcDR(*geo_args), "none published, a necessary condition on D_N"),
            ("D_X", calcDX(*geo_args), "0.15 for merging groups, Rudawska et al. (2015)"),
            ):

            print("  {:<6s} = {:9.4f}   threshold: {:s}".format(name, float(value), threshold))

        # The Vida criterion needs the corrected heliocentric direction instead
        if (helio_ref is not None) and (helio_cmp is not None):

            print("  {:<6s} = {:9.4f}   threshold: {:s}".format("D_V*", \
                float(calcDV(*(helio_ref + helio_cmp))), "none published, unpublished criterion"))
            print("  (D_V* is the Vida criterion already in this module, not the Jopek D_V above)")

    else:
        print()
        print("Criteria on the geocentric radiant and speed were skipped: they need the radiant,")
        print("the solar longitude and the geocentric speed of both orbits, from a second")
        print("trajectory file or from --ra, --dec, --sol and --vg.")

    print()
    print("Dynamical classification of each orbit")
    print("--------------------------------------")
    print("  {:<28s} {:>18s} {:>18s}".format("", "first", "second"))

    for label, values in (
        ("Tisserand parameter T_J", (calcTisserand(a_ref, e_ref, np.radians(incl_ref)), \
            calcTisserand(a_cmp, e, np.radians(incl)))),
        ("Kresak K", (calcKresakK(a_ref, e_ref), calcKresakK(a_cmp, e))),
        ("Kresak P (yr)", (calcKresakP(a_ref, e_ref), calcKresakP(a_cmp, e))),
        ("Aphelion Q (AU)", (calcAphelionDistance(a_ref, e_ref), calcAphelionDistance(a_cmp, e))),
        ):

        print("  {:<28s} {:18.4f} {:18.4f}".format(label, float(values[0]), float(values[1])))

    for label, test in (("Q-i", isCometaryQi), ("E-i", isCometaryEi), ("K-i", isCometaryKi),
                        ("P-i", isCometaryPi)):

        first = "cometary" if bool(test(a_ref, e_ref, np.radians(incl_ref))) else "asteroidal"
        second = "cometary" if bool(test(a_cmp, e, np.radians(incl))) else "asteroidal"

        print("  {:<28s} {:>18s} {:>18s}".format("Jopek & Williams " + label, first, second))

    print("  {:<28s} {:>18s} {:>18s}".format("Tancredi class", \
        classifyTancrediComet(a_ref, e_ref, np.radians(incl_ref)), \
        classifyTancrediComet(a_cmp, e, np.radians(incl))))




    sys.exit()

    ##########################################################################################################


    from wmpl.Utils.PlotCelestial import CelestialPlot


    import numpy as np
    import matplotlib.pyplot as plt


    # Lh, Bh, Vh, sol
    meteor_data = [
        [-125.45181, -10.76560, 33.06874, 261.555537],
        [-125.39419, -11.81266, 35.30113, 261.556261],
        [-124.57469, -10.64484, 33.56727, 261.556798],
        [-125.22765, -10.89096, 32.24767, 261.557992],
        [-122.32308,  -8.76510, 34.71396, 261.559377], # BAD GEOMETRY
        [-122.30905, -12.29563, 35.00645, 261.570680]
        ]

    meteor_data = np.array(meteor_data)

    Lh = np.radians(meteor_data[:, 0])
    Bh = np.radians(meteor_data[:, 1])
    Vh = meteor_data[:, 2]
    sol = np.radians(meteor_data[:, 3])


    for i, (Lh1, Bh1, Vh1, sol1) in enumerate(zip(Lh, Bh, Vh, sol)):
        for j, (Lh2, Bh2, Vh2, sol2) in enumerate(zip(Lh, Bh, Vh, sol)[i:]):

            if i == (i + j):
                continue

            print('----------')
            print('Orbit 1:', np.degrees(Lh1), np.degrees(Bh1), Vh1, np.degrees(sol1))
            print('Orbit 2:', np.degrees(Lh2), np.degrees(Bh2), Vh2, np.degrees(sol2))
            print('D_V:', calcDV_TEST(Lh1, Bh1, Vh1, sol1, Lh2, Bh2, Vh2, sol2))



    m = CelestialPlot(Lh, Bh, projection='stere')

    # Plot heliocentric radians
    m.scatter(Lh, Bh, c=Vh)

    m.colorbar()
    plt.show()



    # uncertainties test

    def hypo(x, c):
        return ((np.sqrt(2) - 1)/(2 - np.sqrt(2))**2)*(np.sqrt(x**2 + c**2) - c)


    # Uncertanty (stddev)
    c = 0.5

    x = np.linspace(0, 5, 100)

    plt.plot(x, hypo(x, c))

    plt.plot(np.zeros(100) + np.sqrt(c), np.linspace(0, 10, 100))

    # Plot x=y line
    plt.plot(x, x, linestyle='--')

    plt.show()