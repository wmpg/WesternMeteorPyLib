""" Associate the given meteor trajectory to a meteor shower. """


from __future__ import print_function, division, absolute_import


import os
import sys
import copy

# Preserve Python 2 compatibility for the encoding option in the "open" function
from io import open

import numpy as np


from wmpl.Config import config
from wmpl.Utils.Math import angleBetweenSphericalCoords
from wmpl.Utils.Pickling import loadPickle


class MeteorShower(object):
    def __init__(self, la_sun, L_g, B_g, v_g, IAU_no, dispersion=None):
        """ Container for meteor shower parameters. 
        
        Arguments:
            la_sun: [float] Solar longitude of the shower radiant (radians).
            L_g: [float] Galactic longitude of the shower radiant (radians).
            B_g: [float] Galactic latitude of the shower radiant (radians).
            v_g: [float] Geocentric velocity of the shower radiant (m/s).
            IAU_no: [int] IAU number of the shower.

        Keyword arguments:
            dispersion: [float] Dispersion of the shower radiant (radians). None by default.
        
        """

        self.la_sun = la_sun
        self.L_g = L_g
        self.B_g = B_g
        self.v_g = v_g
        self.dispersion = dispersion
        self.IAU_no = IAU_no


        # Find the shower code and name in the IAU table
        IAU_Shower_line = iau_shower_list[iau_shower_list[:, 1].astype(int) == self.IAU_no][0]
        self.IAU_code = IAU_Shower_line[3]
        self.IAU_name = IAU_Shower_line[4]


    def __repr__(self):

        out_str  = "Shower: {:d} {:s} {:s}\n".format(self.IAU_no, self.IAU_code, self.IAU_name)
        out_str += "    Sol: {:.6f} deg\n".format(np.degrees(self.la_sun))
        out_str += "    L_g: {:.6f} deg\n".format(np.degrees(self.L_g))
        out_str += "    B_g: {:.6f} deg\n".format(np.degrees(self.B_g))
        out_str += "    V_g: {:.3f} km/s\n".format(self.v_g/1000)


        return out_str


# def loadJenniskensShowers(dir_path, file_name):
#     """ Load the showers from the Jenniskens et al. (2018) table and init MeteorShower objects. """


#     jenniskens_shower_list = []

#     with open(os.path.join(dir_path, file_name), encoding='cp1252') as f:

#         data_start = 0
#         for line in f:

#             # Find the beginning of the data table
#             if "====================================" in line:
#                 data_start += 1
#                 continue


#             # Skip all non-data lines
#             if data_start < 2:
#                 continue


#             line = line.replace('\n', '').replace('\r', '')

#             # Skip empty lines
#             if not line:
#                 continue


#             # Stop if the final line was reached
#             if "[FINAL]" in line:
#                 break

#             # Unpack the shower data
#             l0, L_l0, B_g, v_g, IAU_no = line.split()

#             # Convert values to radians and m/s
#             jenniskens_shower_list.append([np.radians(float(l0)), np.radians(float(L_l0)), 
#                 np.radians(float(B_g)), 1000*float(v_g), int(IAU_no)])



#     return np.array(jenniskens_shower_list)



# # Load the Jenniskens table on startup
# if os.path.isfile(config.jenniskens_shower_table_npy):

#     # Load the npy file (faster) if available
#     jenniskens_shower_list = np.load(config.jenniskens_shower_table_npy)
# else:

#     # If not, load the text file and store the npy for faster loading later
#     jenniskens_shower_list = loadJenniskensShowers(*os.path.split(config.jenniskens_shower_table_file))
#     np.save(config.jenniskens_shower_table_npy, jenniskens_shower_list)


def loadGMNShowerTable(dir_path, file_name):
    """ Load the showers from the GMN table and init MeteorShower objects. 
    
    Arguments:
        dir_path: [str] Path to the directory containing the GMN table.
        file_name: [str] Name of the GMN table file.

    Return:
        [list] List of shower activity measurements and radiant positions (in radians and m/s).
    """

    gmn_shower_list = []

    with open(os.path.join(dir_path, file_name), encoding='cp1252') as f:
        for line in f:

            # Skip the header (lines starting with #)
            if line.startswith('#'):
                continue

            line = line.strip()

            line = line.replace('\n', '').replace('\r', '')

            # Skip empty lines
            if not line:
                continue

            # Unpack the shower data
            la_sun, L_g, B_g, v_g, dispersion, IAU_no, IAU_code = line.split()

            # Convert values to radians and m/s
            gmn_shower_list.append([
                np.radians(float(la_sun)), 
                np.radians(float(L_g)),
                np.radians(float(B_g)), 
                1000*float(v_g), 
                np.radians(float(dispersion)), 
                int(IAU_no)
                ]
            )


    return np.array(gmn_shower_list)


# Load the GMN table on startup
if os.path.isfile(config.gmn_shower_table_npy):
    
    # Load the npy file (faster) if available
    gmn_shower_list = np.load(config.gmn_shower_table_npy)

else:

    # If not, load the text file and store the npy for faster loading later
    gmn_shower_list = loadGMNShowerTable(*os.path.split(config.gmn_shower_table_file))
    np.save(config.gmn_shower_table_npy, gmn_shower_list)


# Load the IAU table
if os.path.isfile(config.iau_shower_table_npy):

    # Load the npy file (faster) if available
    iau_shower_list = np.load(config.iau_shower_table_npy)

else:

    # If not, load the text file and store the npy file for faster loading later
    iau_shower_list = np.loadtxt(config.iau_shower_table_file, delimiter="|", usecols=range(20), dtype=str)
    np.save(config.iau_shower_table_npy, iau_shower_list)



def associateShower(la_sun, L_g, B_g, v_g, sol_window=1.0, max_radius=None, \
    max_veldif_percent=10.0):
    """ Given a shower radiant in Sun-centered ecliptic coordinates, associate it to a meteor shower
        using the showers listed in the GMN shower list.

    Arguments:
        la_sun: [float] Solar longitude (radians).
        L_g: [float] Sun-centered ecliptic longitude (i.e. geocentric ecliptic longitude minus the 
            solar longitude) (radians).
        B_g: [float] Sun-centered geocentric ecliptic latitude (radians).
        v_g: [float] Geocentric velocity (m/s).

    Keyword arguments:
        sol_window: [float] Solar longitude window of association (deg).
        max_radius: [float] Maximum angular separation from reference radiant (deg). None by default, 
            which will use the measured dispersion of the shower from the table.
        max_veldif_percent: [float] Maximum velocity difference in percent.

    Return:
        [MeteorShower instance] MeteorShower instance for the closest match, or None for sporadics.
    """

    # Create a working copy of the shower table
    temp_shower_list = copy.deepcopy(gmn_shower_list)


    # Find all showers in the solar longitude window
    la_sun_diffs = np.abs((temp_shower_list[:, 0] - la_sun + np.pi)%(2*np.pi) - np.pi)
    temp_shower_list = temp_shower_list[la_sun_diffs <= np.radians(sol_window)]


    # Check if any associations were found
    if not len(temp_shower_list):
        return None

    
    # Compute the angular distance between the shower radiants and the reference radiant
    radiant_distances = angleBetweenSphericalCoords(temp_shower_list[:, 2], temp_shower_list[:, 1], B_g, \
        (L_g - la_sun)%(2*np.pi))
    
    # Use the measured dispersion if no maximum radius is given
    if max_radius is None:
        
        # Get the maximum radius for each shower
        max_radius = np.degrees(temp_shower_list[:, 4])
    
        # Filter the showers
        filter_mask = radiant_distances <= np.radians(max_radius)
        temp_shower_list = temp_shower_list[filter_mask]
        max_radius = max_radius[filter_mask]

    else:

        # Filter the showers using a fixed radius
        temp_shower_list = temp_shower_list[radiant_distances <= np.radians(max_radius)]


    # Check if any associations were found
    if not len(temp_shower_list):
        return None


    # Find all showers within the maximum velocity difference limit
    velocity_diff_percents = np.abs(100*(temp_shower_list[:, 3] - v_g)/temp_shower_list[:, 3])
    velocity_filter = velocity_diff_percents <= max_veldif_percent
    temp_shower_list = temp_shower_list[velocity_filter]

    
    # Check if any associations were found
    if not len(temp_shower_list):
        return None
    
    # Filter the max radius if it's a numpy array
    if isinstance(max_radius, np.ndarray):
        max_radius = max_radius[velocity_filter]


    ### Choose the best matching shower by the solar longitude, radiant, and velocity closeness ###

    # Compute the closeness parameters as a sum of normalized closeness by every individual parameter
    sol_dist_norm = np.abs(((temp_shower_list[:, 0] - la_sun + np.pi)%(2*np.pi) \
        - np.pi))/np.radians(sol_window)
    rad_dist_norm = angleBetweenSphericalCoords(temp_shower_list[:, 2], temp_shower_list[:, 1], B_g, (L_g \
        - la_sun)%(2*np.pi))/np.radians(max_radius)
    vg_dist_norm = np.abs(100*(temp_shower_list[:, 3] - v_g)/temp_shower_list[:, 3])/max_veldif_percent
    closeness_param = sol_dist_norm + rad_dist_norm + vg_dist_norm

    # Choose the best matching shower
    best_shower = temp_shower_list[np.argmin(closeness_param)]

    ### ###


    # Init a shower object
    l0, L_l0, B_g, v_g, dispersion, IAU_no = best_shower
    shower_obj = MeteorShower(l0, (L_l0 + l0)%(2*np.pi), B_g, v_g, int(round(IAU_no)), dispersion=dispersion)


    return shower_obj




def associateShowerTraj(traj, sol_window=1.0, max_radius=None, \
    max_veldif_percent=10.0):
    """ Given a Trajectory object, associate it to a meteor shower using the GMN shower list.

    Arguments:
        traj: [Trajectory object] A trajectory object.

    Keyword arguments:
        sol_window: [float] Solar longitude window of association (deg).
        max_radius: [float] Maximum angular separation from reference radiant (deg). If None, the
            measured dispersion of the shower from the table will be used.
        max_veldif_percent: [float] Maximum velocity difference in percent.

    Return:
        [MeteorShower instance] MeteorShower instance for the closest match, or None for sporadics.
    """

    if traj.orbit.ra_g is not None:
        return associateShower(traj.orbit.la_sun, traj.orbit.L_g, traj.orbit.B_g, traj.orbit.v_g, \
            sol_window=sol_window, max_radius=max_radius, max_veldif_percent=max_veldif_percent)

    else:
        return None




if __name__ == "__main__":

    import argparse


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="""Associate the given trajectory object to a meteor shower.""",
        formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('traj_path', type=str, help="Path to a trajectory pickle file.")

    arg_parser.add_argument('-s', '--solwindow', metavar='SOL_WINDOW', \
        help="Solar longitude window (deg) for association. Note that the shower table has an entry for the same shower across several solar longitudes, which covers the activity period.", \
        type=float, default=1.0)

    arg_parser.add_argument('-r', '--radius', metavar='RADIUS', \
        help="Maximum distance from reference radiant for association (deg).", \
        type=float, default=None)

    arg_parser.add_argument('-v', '--velperc', metavar='VEL_PERC', \
        help="Maximum difference in geocentric velocity (in percent) from the reference radiant.", \
        type=float, default=10.0)


    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    ##################################


    # Load the given trajectory object
    if os.path.isfile(cml_args.traj_path):

        # Load the trajectory object
        traj = loadPickle(*os.path.split(cml_args.traj_path))

        # Perform shower association
        shower_obj = associateShowerTraj(traj, sol_window=cml_args.solwindow, max_radius=cml_args.radius, \
            max_veldif_percent=cml_args.velperc)


        if shower_obj is None:
            print("Sporadic")
        
        else:
            print(shower_obj)


    else:
        print("The file {:s} does not exist!".format(cml_args.traj_path))



    ##########################################################################################################
    sys.exit()

    ### Test shower association ###

    import time


    # Draconid meteor
    la_sun = np.radians(195.402007)
    L_g = np.radians(247.41825)
    B_g = np.radians(78.95281)
    v_g = 1000*19.83965


    t1 = time.time()
    print(associateShower(la_sun, L_g, B_g, v_g))
    print("Time for association:", time.time() - t1)