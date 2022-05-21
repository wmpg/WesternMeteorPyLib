
from __future__ import print_function, division, absolute_import

import math



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




if __name__ == "__main__":

    import os
    import sys
    import argparse

    import numpy as np

    from wmpl.Utils.Pickling import loadPickle


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Compare two trajectory files by computing their D criteria.")

    arg_parser.add_argument('traj_path', metavar='TRAJ_PATH', type=str, \
        help="Path to a trajectory pickle file.")

    arg_parser.add_argument('traj_path2', metavar='TRAJ_PATH', type=str, \
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

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    print("Input files:")
    print("First  =", cml_args.traj_path)
    print("Second =", cml_args.traj_path2)

    # Load the reference trajectory pickle file
    traj_ref = loadPickle(*os.path.split(cml_args.traj_path))

    # Load orbital elements
    q_ref = traj_ref.orbit.q
    e_ref = traj_ref.orbit.e
    incl_ref = np.degrees(traj_ref.orbit.i)
    peri_ref = np.degrees(traj_ref.orbit.peri)
    node_ref = np.degrees(traj_ref.orbit.node)



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



    # Compute various D criteria
    d_sh = calcDSH(q_ref, e_ref, np.radians(incl_ref), np.radians(node_ref), np.radians(peri_ref), \
        q, e, np.radians(incl), np.radians(node), np.radians(peri))
    d_d = calcDD(q_ref, e_ref, np.radians(incl_ref), np.radians(node_ref), np.radians(peri_ref), \
        q, e, np.radians(incl), np.radians(node), np.radians(peri))
    d_h = calcDH(q_ref, e_ref, np.radians(incl_ref), np.radians(node_ref), np.radians(peri_ref), \
        q, e, np.radians(incl), np.radians(node), np.radians(peri))

    print()
    print("Results:")
    print("D_SH = {:.4f}".format(d_sh))
    print("D_D  = {:.4f}".format(d_d))
    print("D_H  = {:.4f}".format(d_h))




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