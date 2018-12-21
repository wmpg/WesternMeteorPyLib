""" Functions for comapring the given orbit with orbits of comets and asteroids. """

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

import numpy as np

from wmpl.Utils.Dcriteria import calcDSH, calcDH, calcDD
from wmpl.Config import config


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


    ### PROVIDE ORBITAL ELEMENTS MANUALLY ###
    ##########################################################################################################

    # Orbital elements of the July 21 meteoroid
    # q      =   0.948562
    # e      =   0.710995
    # incl   =   2.461790
    # peri   =  31.979790
    # node   = 298.845068

    # # With velocity correction of 400 m/s
    # q      =   0.944491
    # e      =   0.744092
    # incl   =   2.302671
    # peri   =  32.557683
    # node   = 298.851272


    # # 20170923 meteorite dropper
    # q      =   0.978776
    # e      =   0.576897
    # incl   =   2.823131
    # peri   = 202.247996
    # node   = 180.337617


    # # 20180117 Michigan fireball
    # q      =   0.929614
    # e      =   0.645998
    # incl   =   0.862916
    # peri   = 210.803526
    # node   = 296.539520

    # 20180615 RMS low shallow meteor
    q      =   0.052157
    e      =   0.973413
    incl   =   25.531061
    peri   = 22.588003
    node   = 83.919153


    ##########################################################################################################


    d_crit_type = 'dsh'

    # Find parent bodies for the given orbit
    parent_matches = findParentBodies(q, e, np.radians(incl), np.radians(peri), np.radians(node), \
        d_crit=d_crit_type, top_n=5)

    print('Name                ,   q,     e,    incl,   peri,   node,   D crit', d_crit_type)
    for entry in parent_matches:
        print("{:20s}, {:.3f}, {:.3f}, {:5.2f}, {:7.3f}, {:7.3f}, {:.3f}".format(*entry))





    # ##########################################################################################################

    # ### Plot orbits of all JFCs
    # from wmpl.Utils.PlotOrbits import plotOrbits
    # import datetime

    # # Get orbital elements of comets
    # orbit_list = []

    # count = 0

    # for entry in comets_elements:

    #     comet_name, q, e, incl, peri, node = entry

    #     if e >= 1.0:
    #         continue

    #     a = q/(1.0 - e)


    #     # Take only short period comets
    #     if a > 34:
    #         continue

    #     orbit_list.append([a, e, incl, peri, node])

    #     count += 1

    #     # Limit the number of plotted objects
    #     if count > 4000:
    #         break

    # orbit_list = np.array(orbit_list)

    # orb_time = datetime.datetime.now()

    # plt = plotOrbits(orbit_list, orb_time, linewidth=0.2, color_scheme='dark')

    # plt.show()


    # ##########################################################################################################

    # ### Plot orbits of all KBOs
    # from wmpl.Utils.PlotOrbits import plotOrbits
    # import datetime

    # # Load a list of KBOs
    # #file_path = "C:\Users\delorayn1\Desktop\distant_extended.dat"
    # file_path = "/home/dvida/Desktop/distant_extended.dat"
    # with open(file_path) as f:
    #     kbos_elements = []

    #     for line in f:
    #         line = line.split()

    #         peri = float(line[5])
    #         node = float(line[6])
    #         incl = float(line[7])
    #         e = float(line[8])

    #         try:
    #             a = float(line[10])

    #         except:
    #             continue

    #         kbos_elements.append([a, e, incl, peri, node])
    


    # # Get orbital elements of KBOs
    # orbit_list = []

    # count = 0

    # for entry in kbos_elements:

    #     a, e, incl, peri, node = entry

    #     if e >= 1.0:
    #         continue


    #     # Take only TNOs
    #     if (a < 30) or (a > 55):
    #         continue

    #     orbit_list.append([a, e, incl, peri, node])

    #     count += 1

    #     # Limit the number of plotted objects
    #     if count > 5000:
    #         break

    # orbit_list = np.array(orbit_list)

    # orb_time = datetime.datetime.now()

    # # Plot KBOs
    # plt = plotOrbits(orbit_list, orb_time, linewidth=0.1, color_scheme='dark', figsize=(20, 20))




    # # Get orbital elements of Centaurs
    # orbit_list = []

    # count = 0

    # for entry in kbos_elements:

    #     a, e, incl, peri, node = entry

    #     if e >= 1.0:
    #         continue


    #     # Take only Centaurs
    #     if (a > 30):
    #         continue

    #     orbit_list.append([a, e, incl, peri, node])

    #     count += 1

    #     # Limit the number of plotted objects
    #     if count > 5000:
    #         break


    # orbit_list = np.array(orbit_list)

    # orb_time = datetime.datetime.now()

    # # Plot Centaurs
    # plt = plotOrbits(orbit_list, orb_time, linewidth=0.4, color_scheme='dark', orbit_colors=['#2072e3']*len(orbit_list), plt_handle=plt)



    # # Get orbital elements of comets
    # orbit_list = []

    # count = 0

    # for entry in comets_elements:

    #     comet_name, q, e, incl, peri, node = entry

    #     if e >= 1.0:
    #         continue

    #     a = q/(1.0 - e)


    #     # Take only short period comets
    #     if a > 34:
    #         continue

    #     orbit_list.append([a, e, incl, peri, node])

    #     count += 1

    #     # Limit the number of plotted objects
    #     if count > 4000:
    #         break

    # orbit_list = np.array(orbit_list)

    # orb_time = datetime.datetime.now()

    # plt = plotOrbits(orbit_list, orb_time, linewidth=0.05, color_scheme='dark', orbit_colors=['r']*len(orbit_list), plt_handle=plt)




    # plt.show()