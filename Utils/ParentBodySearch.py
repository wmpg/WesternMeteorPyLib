""" Functions for comapring the given orbit with orbits of comets and asteroids. """

from __future__ import print_function, division, absolute_import

import numpy as np

from Utils.Dcriteria import calcDSH, calcDH, calcDD
from Config import config


def loadCometsElements(comets_file):
    """ Loads the orbital elements of comets from the JPL comets element list. """

    comets_list = []

    with open(comets_file) as f:

        for k, line in enumerate(f):

            # Skip the first two lines
            if k < 2:
                continue


            comet_name = line[:44].strip()
            q = float(line[52:63])
            e = float(line[64:74])
            incl = float(line[75:84])
            peri = float(line[85:94])
            node = float(line[95:104])

            comets_list.append([comet_name, q, e, incl, peri, node])

    return comets_list


def loadAsteroidsElements(asteroids_file):
    """ Loads the elements of asteroids in the MPC format. """

    asteroids_list = []

    with open(asteroids_file) as f:

        for i, line in enumerate(f):
            
            # Skip first 2 lines
            if i < 2:
                continue

            asteroid_name = line[:38].strip()
            q = float(line[39:45])
            e = float(line[107:112])
            incl = float(line[101:106])
            peri = float(line[89:94])
            node = float(line[95:100])

            asteroids_list.append([asteroid_name, q, e, incl, peri, node])


    return asteroids_list



# Load the parent body database
comets_elements = loadCometsElements(config.comets_elements_file)
asteroids_elements = loadAsteroidsElements(config.asteroids_amors_file)
asteroids_elements += loadAsteroidsElements(config.asteroids_apollos_file)
asteroids_elements += loadAsteroidsElements(config.asteroids_atens_file)


def findParentBodies(q, e, incl, peri, node, d_crit='dsh', top_n=10):
    """ Compares the given orbit to the orbit of asteroids and comets from the database using the given
        D criterion function and returns top N best maches.
    
    Arguments;
        q: [float] Perihelion distance in AU.
        e: [float] Eccentricity.
        incl: [float] Inclination (radians).
        peri: [float] Argument of perihelion (radians).
        node: [float] Ascending node (radians).

    Keyword arguments:
        d_crit: [str] D criterion function. 'dsh' for Southworth and Hawkins, 'dd' for Drummond, and 'dh' for 
            Jopek.
        top_n: [int] How many objects with the heighest orbit similarity will be returned. 10 is default.
            If -1 is given, the whole list of bodies will be returned.

    Return:
        [list] A list of best maching objects: [Object name, q, e, i, peri, node, D criterion value].
    """


    # Choose the appropriate D criterion function:
    if d_crit == 'dsh':
        dFunc = calcDSH

    elif d_crit == 'dd':
        dFunc = calcDD

    elif d_crit == 'dh':
        dFunc = calcDH

    else:
        print('The given D criteria function not recognized! Using D_SH by default...')
        dFunc = calcDSH


    # Combine comets and asteroids into one parent body list
    parent_body_elements = comets_elements + asteroids_elements

    dcrit_list = []

    # Calculate D criteria for every comet and asteroid
    for k, body in enumerate(parent_body_elements):

        # Extract body orbital elments
        name, q2, e2, incl2, peri2, node2 = body

        # Convert the body elements to radians
        incl2 = np.radians(incl2)
        peri2 = np.radians(peri2)
        node2 = np.radians(node2)

        # Calculate D criterion between the body and the meteoroid
        d_crit = dFunc(q, e, incl, node, peri, q2, e2, incl2, node2, peri2)

        # Add the body index and D criterion to list
        dcrit_list.append([k, d_crit])


    # Sort the bodies by ascending D criterion value
    dcrit_list = np.array(dcrit_list)
    dcrit_list = dcrit_list[np.argsort(dcrit_list[:, 1])]

    # Return the whole list of bodies if top_n is -1
    if top_n == -1:
        pass

    # Return top N most similar bodies
    else:

        # Make sure the list is long enough
        if top_n > len(dcrit_list):
            top_n = len(dcrit_list)

        dcrit_list = dcrit_list[:top_n]

            
    results = []

    for index, d_crit in dcrit_list:
        results.append(parent_body_elements[int(index)] + [d_crit])

    return results








if __name__ == "__main__":

    # Orbital elements of the July 21 meteoroid
    # q      =   0.948562
    # e      =   0.710995
    # incl   =   2.461790
    # peri   =  31.979790
    # node   = 298.845068

    # With velocity correction of 400 m/s
    q      =   0.944491
    e      =   0.744092
    incl   =   2.302671
    peri   =  32.557683
    node   = 298.851272

    # Find parent bodies
    parent_matches = findParentBodies(q, e, np.radians(incl), np.radians(peri), np.radians(node), \
        d_crit='dsh', top_n=5)

    print('Name, q, e, incl, peri, node, D crit')
    for entry in parent_matches:
        print(entry)