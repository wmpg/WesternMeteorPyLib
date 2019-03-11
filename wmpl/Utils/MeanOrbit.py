""" Functions for calculating mean orbital parameters from a given list of orbits. 

    Functions courtesy of Pete Gural and Ivica Skokic.
"""

from __future__ import print_function, division, absolute_import


import numpy as np


# Predefined standard epochs
JDE_B1900 = 2415020.3135
JDE_B1950 = 2433282.4235
JDE_J2000 = 2451545.00
JDE_J2050 = 2469807.50

# Gauss's gravitational constant, sqrt(GM)
GAUSS_GRAVITY_CONST = 0.01720209895
AU = 1.495978707e+11  #m
M_SUN = 1.989e+30     #kg
MU_SUN = GAUSS_GRAVITY_CONST**2


def decomposeOrbElem(orb_elem):

    if np.ndim(orb_elem)<2:
        q,e,i,O,w = np.array([orb_elem]).T

    else:    
        q,e,i,O,w = orb_elem.T

    return q,e,i,O,w




def precessOrbit(orb_elem, initial_jde, final_jde=JDE_J2000):
    """
    Reduction of ecliptical orbital elements from one equinox to another.

    Reference: Meeus, J., 1991, Astronomical Algorithms, Willmann-Bell, Inc., p. 147-150
    
    Arguments:
        orb_elem : [ndarray] q,e,i,O,w - Instead of q, a (semimajor axis) can be safely used because
            q (or a) and e input fields are not used and just copied to the output.
            q: [float] Perihelion distance in AU.
            e: [float] Eccentricity.
            i: [float] Inclincation (radians).
            o: [float] Node (radians).
            w: [float] Argument of perihelion (radians).
        initial_jde: [float] Initial epoch for equinox, in julian days referred to dynamical time.

    Keyword arguments:
        final_jde: [float] Final epoch for equinox, in julian days referred to dynamical time, default value 
            is J2000.0.

    Return:
        (q, e, i, O, w)
    
    """

    # Fix input data dimensionality for numpy methods
    fix_dim = False
    if np.ndim(orb_elem) < 2:
        orb_elem = np.array([orb_elem])
        fix_dim = True

    T = (initial_jde - 2451545.0)/36525.0
    t = (final_jde - initial_jde)/36525.0

    eta = np.radians((t*(47.0029 + T*(-0.06603 + 0.000598*T) + t*(-0.03302 + 0.000598*T + 0.000060*t)))/3600)
    ppi = np.radians(174.876384 + (T*(3289.4789 + 0.60622*T) + t*(-869.8089 - 0.50491*T + 0.03536*t))/3600)
    p = np.radians((t*(5029.0966 + T*(2.22226 - 0.000042*T) + t*(1.11113 - 0.000042*T + 0.000006*t)))/3600)

    psi = ppi + p

    q0, e0, i0, o0, w0 = orb_elem.T

    A = np.sin(i0)*np.sin(o0 - ppi)
    B = -np.sin(eta)*np.cos(i0) + np.cos(eta)*np.sin(i0)*np.cos(o0 - ppi)

    o = psi + np.arctan2(A,B)
    i = np.arcsin(np.sqrt(A*A + B*B))
    w = w0 + np.arctan2(-np.sin(eta)*np.sin(o0 - ppi), np.sin(i0)*np.cos(eta) \
        - np.cos(i0)*np.sin(eta)*np.cos(o0 - ppi))

    # Deal with cases when i0 == 0
    idx = i0 == 0
    o[idx] = psi + np.pi
    i[idx] = eta

    out_elem = np.array([q0, e0, i, o, w])

    if fix_dim: 
        out_elem = np.squeeze(out_elem)

    return out_elem




def kepler2vectorial(orb_elem):
    """ Convert keplerian orbital elements q, e, i, o, w to vector elements h, e, E (angular momentum vector, 
        eccentricty vector, energy).

    Arguments:
        orb_elem: [ndarray] Numpy array with orbital elements
            q: [float] Perihelion distance in AU.
            e: [float] Eccentricity.
            i: [float] Inclincation (radians).
            o: [float] Node (radians).
            w: [float] Argument of perihelion (radians).

    Return:
        (h, ev, E): [tuple of floats]
    """

    # Fix input data dimensionality for numpy methods
    fix_dim = False
    if np.ndim(orb_elem) < 2:
        orb_elem = np.array([orb_elem])
        fix_dim = True

    q, e, i, O, w = decomposeOrbElem(orb_elem)

    n = np.size(i)
    ev = np.zeros([n, 3])
    h = np.zeros([n, 3])

    sinw = np.sin(w)
    cosw = np.cos(w)
    sinO = np.sin(O)
    cosO = np.cos(O)
    sini = np.sin(i)
    cosi = np.cos(i)

    x = q*(cosw*cosO - sinw*sinO*cosi)
    y = q*(cosw*sinO + sinw*cosO*cosi)
    z = q*sinw*sini

    tmp = np.sqrt(MU_SUN*(1 + e)/q)

    x_dot = tmp*(-sinw*cosO - cosw*sinO*cosi)
    y_dot = tmp*(-sinw*sinO + cosw*cosO*cosi)
    z_dot = tmp*cosw*sini

    r = np.sqrt(x*x + y*y + z*z)

    h[:, 0] = y*z_dot - z*y_dot
    h[:, 1] = z*x_dot - x*z_dot
    h[:, 2] = x*y_dot - y*x_dot

    #E = 0.5*(x_dot*x_dot+y_dot*y_dot+z_dot*z_dot)-MU_SUN/r
    E = 0.5*MU_SUN*(e - 1.0)/q  #almost identical (rel. err. 1e-14) to upper but faster

    inv_mu = 1/MU_SUN

    ev[:,0] = inv_mu*(y_dot*h[:,2] - z_dot*h[:,1]) - x/r
    ev[:,1] = inv_mu*(z_dot*h[:,0] - x_dot*h[:,2]) - y/r
    ev[:,2] = inv_mu*(x_dot*h[:,1] - y_dot*h[:,0]) - z/r

    if fix_dim:
        h = np.squeeze(h)
        ev = np.squeeze(ev)
        E = np.squeeze(E)

    return h, ev, E


def vectorial2kepler(h, ev, E):

    # fix input data dimensionality for numpy methods
    fix_dim = False
    if np.ndim(h) < 2:
       h = np.array([h])
       ev = np.array([ev])
       fix_dim = True

    h2 = np.diag(np.dot(h, h.T))
    ev2 = np.diag(np.dot(ev, ev.T))
    e = np.sqrt(ev2)

    # # Comparing this value to the one given above should check orbit elements concistency, if to different, 
    # # h, e, E are ill defined
    # e_check = np.sqrt(1.0 + 2.0*E/(MU_SUN*MU_SUN)*h2) 
    # print(e, (e - e_check)/e2*100)

    q = h2/MU_SUN/(1.0 + e)

    idx = h2 != 0
    i = np.zeros(np.shape(h2))
    i[idx] = np.arctan2(np.sqrt(h[idx, 0]*h[idx, 0] + h[idx, 1]*h[idx, 1]), h[idx, 2])
    idx = np.sin(i) != 0

    O = np.zeros(np.shape(h2))
    O[idx] = np.arctan2(h[idx,0], -h[idx,1])

    idx = (ev2 != 0)*(np.sin(i) !=0)

    w = np.zeros(np.shape(h2))    
    w[idx] = np.arctan2(np.sqrt(h2)*ev[idx, 2], h[idx, 0]*ev[idx, 1] - h[idx, 1]*ev[idx, 0])

    idx = (w < 0)
    w[idx] = w[idx] + 2*np.pi

    idx = (O < 0)
    O[idx] = O[idx] + 2*np.pi

    out = np.array([q, e, i, O, w]).T

    if fix_dim: 
        out = np.squeeze(out)

    return out



def meanOrbitKeplerAvg(orb_elem):
    """ Simple mean of kepler orbital elements. This will generally not give a consistent orbit. 
        Better use LSQ average.
    """

    # fix input data dimensionality for numpy methods
    if np.ndim(orb_elem) < 2:
        orb_elem = np.array([orb_elem])

    out_elem = np.mean(orb_elem, axis=0)

    return out_elem



def meanOrbitVectorAvg(orb_elem):
    """ Simple mean of vector orbital elements h, e, E. This will generally not give a consistent orbit. 
        Better use LSQ average.

    """

    #----- compute the angular momentum vectors, eccentricty vectors, energy
    #                  hvec(:,3),                evec(:,3),           En(:)

    # fix input data dimensionality for numpy methods
    if np.ndim(orb_elem) < 2:
        orb_elem = np.array([orb_elem])

    hvec, evec, En = kepler2vectorial(orb_elem)

    #----- compute the mean of each component
    h = np.mean(hvec, 0)
    e = np.mean(evec, 0)
    Emean = np.mean(En)

    #----- convert back to Keplerian orbital elements
    out_elem = vectorial2kepler(h, e, Emean)

    return out_elem



def meanOrbitVectorLSQ(orb_elem):
    """ Compute the mean orbit of 7 or more meteors using Jopek's constrained least squares formulated 
        solution working in heliospheric vectorial space.

    Reference: Jopek et al. 2006 "Calculation of the Mean Orbit of a Meteoroid Stream", MNRAS 371, 1367-1372

    Arguments:
        orb_elem: [ndarray] Numpy array with orbital elements
            q: [float] Perihelion distance in AU.
            e: [float] Eccentricity.
            i: [float] Inclincation (radians).
            o: [float] Node (radians).
            w: [float] Argument of perihelion (radians).

    Return:
        (q, e, i, o, w): [tuple of floats] Mean orbital elements.
    """

    # Fix input data dimensionality for numpy methods
    if np.ndim(orb_elem) < 2:
        orb_elem = np.array([orb_elem])

    h, ev, E = kepler2vectorial(orb_elem)
    
    N = np.size(ev)

    # Calculate average orbital elements
    evs = np.mean(ev , 0)
    hs = np.mean(h , 0)
    Es = np.mean(E , 0)
    evs_err = np.std(ev, 0, ddof=1)/np.sqrt(N)
    hs_err = np.std(h, 0, ddof=1)/np.sqrt(N)
    Es_err = np.std(E, 0, ddof=1)/np.sqrt(N)

    # There need to be at least 7 orbital elements for this method to work
    if (N < 7):
        return evs, hs, Es, evs_err, hs_err, Es_err

    he = 1
    max_iter = 20
    max_err = 1e-10
    i = 0

    while (i < max_iter) and (abs(he) > max_err):

        he = np.dot(hs, evs.T)
        hs2 = np.dot(hs, hs.T)
        evs2 = np.dot(evs, evs.T)

        # Calculate R matrix (symmetric)
        R = np.zeros([9,9])

        for j in range(0, 7):
            R[j,j] = N

        for j in range(0, 3):
            R[j,   7] =- evs[j]
            R[j+3, 7] =- hs[j]
            R[j,   8] =- 4/(MU_SUN**2)*hs[j]*Es
            R[j+3, 8] =- 2*evs[j]

        R[6,8] = 2*hs2/(MU_SUN**2)

        for k in range(0,9):
            for l in range(k,9):
                R[l, k] = R[k, l]

        # Calculate t vector
        t = np.zeros([9])

        for j in range(0,3):
            t[j]     = np.sum(hs[j] - h[:,j])
            t[j + 3] = np.sum(evs[j] - ev[:,j])

        t[6] = np.sum(Es - E)
        t[7] = he
        t[8] = evs2 - 2*Es/MU_SUN**2*hs2 - 1

        dOs = np.linalg.solve(R, t)

        for j in range(0, 3):
            hs[j] = hs[j] + dOs[j]
            evs[j] = evs[j] + dOs[j + 3]

        Es = Es + dOs[6]

        i += 1


    if (abs(he) > max_err) and (i >= max_iter):
        print('WARNING: iteration did not converge in', max_iter, \
            'steps. Results may not be correct. Try using ordinary average.')

    out_elem = vectorial2kepler(hs, evs, Es)

    return out_elem



if __name__ == "__main__":

    import os
    import argparse


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="""Compute mean orbital elements of orbits in the given file (orbits.csv by default) and output them to a file.""",
        formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('input_file', type=str, nargs='?', default='orbits.csv', help='Path to the input CSV file with orbits. The values should be comma or semicolon separated, and the columns should be: q, e, i, o, w. The angular values are expected in degrees. If no file is given, orbits.csv will be used.')

    arg_parser.add_argument('-o', '--out', metavar='OUT_FILE', help="Output file path. out.txt by default.", type=str, nargs='?')

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    ############################

    file_path = cml_args.input_file

    print('File path:', file_path)


    if os.path.isfile(file_path):

        # Read the CSV file
        with open(file_path) as f:

            orb_elem_list = []

            for line in f:

                # Skip line beginnign with #
                if line.startswith('#'):
                    continue

                line = line.replace('\n', '').replace('\r', '')

                # Determine which delimiter to use
                if line.count(',') > line.count(';'):
                    line = line.split(',')

                else:
                    line = line.split(';')


                # Extract orbital elements
                if len(line) < 5:
                    continue

                q, e, i, o, w = list(map(float, line[:5]))


                orb_elem_list.append([q, e, np.radians(i), np.radians(o), np.radians(w)])



            # Compute mean orbital elements
            mean_orb = meanOrbitVectorLSQ(np.array(orb_elem_list))

            q, e, i, o, w = mean_orb



            if cml_args.out is None:
                out_path = os.path.join(os.path.dirname(os.path.abspath(file_path)), 'out.txt')

            else:
                out_path = cml_args.out



            # Write orbital elements to file
            with open(out_path, 'w') as f:

                f.write('Mean orbit\n\r')
                f.write('----------\n\r')
                f.write('q = {:.6f} AU\n\r'.format(q))
                f.write('e = {:.6f}\n\r'.format(e))
                f.write('i = {:.6f} deg\n\r'.format(np.degrees(i)))
                f.write('node = {:.6f} deg\n\r'.format(np.degrees(o)))
                f.write('peri = {:.6f} deg\n\r'.format(np.degrees(w)))







