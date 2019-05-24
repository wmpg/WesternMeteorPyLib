""" Loading and hadling Mirfit *.met files. """

from __future__ import print_function, absolute_import, division

import os
import sys
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

from wmpl.Formats.Plates import AffPlate, AstPlate, plateExactMap, plateScaleMap
from wmpl.Trajectory.Trajectory import Trajectory
from wmpl.Trajectory.GuralTrajectory import GuralTrajectory
from wmpl.Utils.TrajConversions import unixTime2JD


class Star(object):
    """ Container for info about stars in the vid. """

    def __init__(self, theta, phi, mag, name):

        self.theta = theta
        self.phi = phi
        self.mag = mag
        self.name = name


    def __repr__(self):

        ret_str = ""
        ret_str += self.name + " theta " + str(self.theta) + " phi " + str(self.phi) + " mag " + str(self.mag)

        return ret_str



class PickInfo(object):
    """ Container of information for individual picks. """

    def __init__(self, theta, phi):

        # Pick frame
        self.frame = 0

        # Fragment ID
        self.id = 0

        # Pick time
        self.unix_time = 0

        # Mirror coordinates of the image centre
        self.hx = 0
        self.hy = 0

        # Original picks centroids
        self.cx = 0
        self.cy = 0

        # Original picks sky coordinates
        self.theta = theta
        self.phi = phi

        # Log sum pixel
        self.lsp = 0



class MetStruct(object):
    

    def __init__(self, dir_path, mirfit=False):
        """ Container for Met file info. 
        
        Arguments:
            dir_path: [str] Path to the directory which contains the .met file.

        Keyword arguments:
            mirfit: [bool] Flag which indicates if Mirfit .met file if being loaded (True). If False (by 
            defualt), METAL-style .met file will be loaded. Note: Specifying the wrong format for the .met 
            file may produce an error.
        """


        # Met file location
        self.dir_path = dir_path

        # Mirfit flag
        self.mirfit = mirfit

        # Site geo coordinates
        self.sites_location = {}

        # Init pick dict where picks will be stored as a list
        self.picks = {}

        # Dictionary of lists of pick objects, with full pick info
        self.picks_objs = {}

        # Init mirror positions in time
        self.mirror_pos = {}

        # Init plates
        self.scale_plates = {}
        self.exact_plates = {}

        # Init geographical positions
        self.lat = {}
        self.lon = {}
        self.elev = {}

        # Init star positions
        self.stars = {}

        # Vid files
        self.vids = {}


    def pairFrame2MirPos(self):
        """ Pairs frames to their respective mirror positions. """

        # Do this for both sites
        for site in self.sites:

            # Extract time data and encoder positions
            time_data = np.array(self.mirror_pos[site])[:, 0]
            hx_data = np.array(self.mirror_pos[site])[:, 1]
            hy_data = np.array(self.mirror_pos[site])[:, 2]

            # Linear regression: time vs. hx
            hx_slope, hx_intercept, _, _, _ = scipy.stats.linregress(time_data, hx_data)

            # Linear regression: time vs. hy
            hy_slope, hy_intercept, _, _, _ = scipy.stats.linregress(time_data, hy_data)

            # print 'SITE: ', site

            # Go through all picks
            for pick in self.picks[site]:

                # Extract frame number
                frame = pick[0]

                # Get pick frame time
                ts, tu = pick[11], pick[12]
                frame_time = ts + tu/(10**6)

                # Calculate mirror positions
                hx = hx_slope*frame_time + hx_intercept
                hy = hy_slope*frame_time + hy_intercept

                # print 'fr', frame, 'frtime, {:f}'.format(frame_time), 'hx', hx, 'hy', hy

                # Append mirror positions to the pick
                pick.append(hx)
                pick.append(hy)



    def __repr__(self):

        ret_str = ""

        for site in self.sites:
            ret_str += 'Site '+str(site)+'\n'
            ret_str += 'Location: '+','.join(list(map(str, self.sites_location[site]))) + '\n'
            ret_str += 'PICKS: \n'
            ret_str += str(self.picks[site])
            ret_str += '\n'
            ret_str += 'SCALE: \n'
            ret_str += str(self.scale_plates[site])
            ret_str += '\n'
            ret_str += 'EXACT: \n'
            ret_str += str(self.exact_plates[site])
            ret_str += '\n'
            ret_str += 'Vid file:'+str(self.vids[site])+'\n'
            ret_str += '\n'
            ret_str += 'STARS:\n'

            for star in self.stars[site]:
                ret_str += str(star) + '\n'

            ret_str += '\n\n'


        return ret_str



def coordinatesSkyToImage(theta, phi, exact, scale, hx_centre, hy_centre):
    """ Converts a point from (theta, phi) to image coordinates. """

    # Calculate the pick location from (theta, phi) to mirror encoder coordinates
    hx, hy = plateExactMap(exact, theta, phi, reverse_map=True)

    # Calculate encoder offset from the centre
    hu = hx - hx_centre
    hv = hy - hy_centre

    # Get image offsets from encoder offsets
    nx, ny = plateScaleMap(scale, hu, hv, reverse_map=True)

    # Get image coordinates of the centroid
    mx = scale.wid/2.0 + nx
    my = scale.ht/2.0 - ny

    return mx, my


def coordinatesImageToSky(mx, my, exact, scale, hx_centre, hy_centre):
    """ Converts a point from image coordinates (mx, my) to (theta, phi). """

    # Get image coordinates of the centroid
    cx = mx - scale.wid/2.0
    cy = scale.ht/2.0 - my

    # Get image offsets from encoder offsets
    hu, hv = plateScaleMap(scale, cx, cy)

    # Calculate encoder offset from the centre
    hx = hx_centre + hu
    hy = hy_centre + hv

    # Calculate the pick location from (theta, phi) to mirror encoder coordinates
    theta, phi = plateExactMap(exact, hx, hy)

    return theta, phi




def extractPicks(met, mirfit=False):
    """ Extracts picks from the list and convert them to pick objects. """

    # Go though all sites
    for site in met.sites:

        # If there are no picks for this site, skip it and remove from the list of sites
        if not met.picks[site]:
            met.sites.remove(site)
            continue

        # Extract mirfit picks into mirfit objects
        if mirfit: 

            # Extract frames
            frames = np.array(met.picks[site])[:,0]

            # Fragment ID
            id_data = np.array(met.picks[site])[:,1]

            # Extract original centroids
            cx_data, cy_data = np.hsplit(np.array(met.picks[site])[:,2:4], 2)

            # Extract UNIX time
            ts_data, tu_data = np.hsplit(np.array(met.picks[site])[:,11:13], 2)
            time_data = ts_data + tu_data/1000000

            # Extract log sum pixel
            lsp_data = np.array(met.picks[site])[:,7]


            # Extract mirror positions on each frame
            hx_data, hy_data = np.hsplit(np.array(met.picks[site])[:,22:24], 2)

            # Init theta, phi arrays
            theta = np.zeros_like(cx_data)
            phi = np.zeros_like(cx_data)

            # Calculate (theta, phi) from image coordinates
            for i in range(len(cx_data)):
                theta[i], phi[i] = coordinatesImageToSky(cx_data[i], cy_data[i], met.exact_plates[site], 
                    met.scale_plates[site], hx_data[i], hy_data[i])

            # Init a list of picks for this site
            met.picks_objs[site] = []

            # Generate a list of pick object
            for fr, frag_id, cx, cy, hx, hy, theta_pick, phi_pick, unix_time, lsp in zip(frames, id_data, \
                cx_data, cy_data, hx_data, hy_data, theta, phi, time_data, lsp_data):
                
                # Init a new pick
                pick = PickInfo(theta_pick, phi_pick)

                # Set the pick frame
                pick.frame = int(fr)

                # Set pick UNIX time
                pick.unix_time = unix_time

                # Set fragment ID
                pick.id = frag_id

                # Set original pick centroids
                pick.cx = cx
                pick.cy = cy

                # Set mirror position of frame centre
                pick.hx = hx
                pick.hy = hy

                # Log sum pixel
                pick.lsp = lsp

                # Add the pick to the list of all picks
                met.picks_objs[site].append(pick)


        # Extract METAL picks to pick objects
        else:

            # Extract frames
            frames = np.array(met.picks[site])[:, 2].astype(np.int)

            # Extract original centroids
            cx_data, cy_data = np.hsplit(np.array(met.picks[site])[:, 4:6].astype(np.float64), 2)

            # Extract UNIX time
            ts_data, tu_data = np.hsplit(np.array(met.picks[site])[:, 23:25].astype(np.float64), 2)
            time_data = ts_data + tu_data/1000000

            # Azimuthal coordinates
            theta, phi = np.hsplit(np.radians(np.array(met.picks[site])[:, 12:14].astype(np.float64)), 2)


            # Init a list of picks for this site
            met.picks_objs[site] = []

            # Generate a list of pick object
            for fr, cx, cy, theta_pick, phi_pick, unix_time in zip(frames, cx_data, cy_data, theta, phi, 
                time_data):
                
                # Init a new pick
                pick = PickInfo(theta_pick, phi_pick)

                # Set the pick frame
                pick.frame = int(fr)

                # Set pick UNIX time
                pick.unix_time = unix_time

                # Set original pick centroids
                pick.cx = cx
                pick.cy = cy

                # Add the pick to the list of all picks
                met.picks_objs[site].append(pick)


    return met





def loadMet(dir_path, file_name, mirfit=False):
    """ Loads a *.met file. 
    
    Arguments:
        dir_path: [str] Path to the directory containing the *.met file
        file_name: [str] Name of the *.met file

    Keyword arguments:
        mirfit: [bool] Flag which indicates if Mirfit .met file if being loaded (True). If False (by defualt),
            METAL-style .met file will be loaded. Note: Specifying the wrong format for the .met file may 
            produce an error.

    Return:
        met: [MetStruct object]
    """


    # Init an empty Met structure
    met = MetStruct(dir_path, mirfit=mirfit)

    with open(os.path.join(dir_path, file_name)) as f:

        sites = []

        
        f.seek(0)

        # Mirfit and METAL store site numbers differently
        if mirfit:
            site_prefix = "video ;"
        else:
            site_prefix = "plate ;"
            

        # Find all participating sites that are in the .met file
        for line in f:

            # Find the lines which will contain the site ID
            if site_prefix in line:

                # Extract the site ID
                line = line.strip(site_prefix).split()
                site_id = line[1]

                # If there is no plate for this site, skip it
                if 'NULL' in line[2]:
                    continue


                # Add the site ID to the list of sites
                sites.append(site_id)


                # Extract site geographical coordinates is METAL file was given
                if not mirfit:

                    met.lat[site_id] = float(line[11])
                    met.lon[site_id] = float(line[13])
                    met.elev[site_id] = float(line[15])


        met.sites = sites


        # Go through the sites
        for site in met.sites:

            # Reset file pointer to the beginning
            f.seek(0)

            # Init list of data for individual sites
            met.picks[site] = []
            met.scale_plates[site] = []
            met.exact_plates[site] = []
            met.mirror_pos[site] = []
            met.stars[site] = []


            # Exact plate prefix
            exact_prefix = "exact ; site "+str(site)+" type 'AST'"

            # Scale plate prefix
            scale_prefix = "scale ; site "+str(site)+" type 'AFF'"

            # Video file prefix
            vid_prefix = "video ; site "+str(site)
            vid_prefix_nosite = "video ; site "

            # Mirror positions prefix
            mirror_pos_prefix = "point ;"
            mirror_pos_read = False


            # Star entry prefix
            star_prefix = "mark ; site "+str(site)+" type starpick"

            # Prefixes for meteor picks and star picks (METAL and Mirfit have different styles of .met files)
            if mirfit:

                # Mirfit-style .met file

                # Line must start with this prefix to be taken as a meteor pick
                pick_prefix = "mark ; site "+str(site)+" type meteor "

            else:

                # METAL-style .met file
                pick_prefix = "mark ; tag "


            # Load meteor picks and plates from each site
            for line in f:

                # If loading a METAL-style file, every line needs to be checked it if has the proper site
                if (not mirfit) and (not ('site ' + str(site) in line)):
                    continue


                # Check if scale plate
                if scale_prefix in line:

                    # Extract scale plate data
                    scale_data = line.replace(scale_prefix, '').split()[1::2]
                    scale_data = list(map(float, scale_data[:-1]))

                    # Init a new scale plate
                    scale = AffPlate()

                    # Assign loaded values to the scale plate object
                    scale.sx, scale.sy, scale.phi, scale.tx, scale.ty, scale.wid, scale.ht = scale_data[:7]
                    scale.site = site

                    # Init the converson matrix
                    scale.initM()

                    met.scale_plates[site] = scale


                # Check if exact plate
                if exact_prefix in line:

                    # Extract exact plate data
                    exact_data = line.replace(exact_prefix, '').split()[1::2]

                    # Extract the header
                    exact_header = list(map(float, exact_data[3:12]))

                    # Unpack fit parameters
                    exact_fit = list(map(lambda x: list(map(float, x.split(':'))), exact_data[12:]))

                    # Init a new exact plate
                    exact = AstPlate()

                    # Assign loaded exact plate values to the exact plate object
                    exact.lat, exact.lon, exact.elev, exact.ts, exact.tu, exact.th0, exact.phi0, exact.wid, \
                        exact.ht = exact_header
                    exact.a, exact.da, exact.b, exact.db, exact.c, exact.dc, exact.d, exact.dd = exact_fit
                    exact.site = site

                    # Init the converson matrix
                    exact.initM()
                    
                    met.exact_plates[site] = exact

                    # Init geographical coordinates for the given site
                    met.lon[exact.site] = exact.lon
                    met.lat[exact.site] = exact.lat
                    met.elev[exact.site] = exact.elev


                # Check if star entry
                if (star_prefix in line) and mirfit:

                    # Extract star data
                    star_data = line.replace(star_prefix, '').split()[1::2]

                    # Init new star object
                    cur_star = Star(np.radians(float(star_data[0])), np.radians(float(star_data[1])), 
                        float(star_data[2]), star_data[7].replace("'", ""))

                    # Add star to the star list
                    met.stars[site].append(cur_star)

                
                # Check if meteor position pick
                if pick_prefix in line:

                    if mirfit:

                        # Mirfit-style pick

                        # Extract pick data
                        pick_data = line.replace(pick_prefix, '').split()[1::2]
                        pick_data = list(map(float, pick_data))

                        met.picks[site].append(pick_data)

                    else:

                        # METAL-style pick

                        # Extract pick data
                        pick_data = line.replace(pick_prefix, '').split()[0::2]

                        # Check that it is a meteor pick
                        if 'type meteor' in line:
                            pick_data = list(map(float, pick_data[:29] + pick_data[31:]))

                            met.picks[site].append(pick_data)


                # Check if vid file name
                if vid_prefix in line:

                    # Extract vid file data
                    video_data = line.replace(vid_prefix, '').split()[1::2]

                    # Extract vid file name
                    met.vids[site] = video_data[1].replace("'", "")

                    # Extract site location
                    met.sites_location[site] = list(map(float, video_data[12:15]))

                    # Allow reading mirror positions
                    mirror_pos_read = True


                # Check if mirror position entry
                if mirror_pos_read and (mirror_pos_prefix in line):

                    # Extract mirror position data
                    mpos_data = line.replace(mirror_pos_prefix, '').split()[1::2]
                    mpos_data = list(map(float, mpos_data))

                    # Store mirror position
                    met.mirror_pos[site].append(mpos_data)


                # Break if another site is reached
                if mirror_pos_read and (vid_prefix_nosite in line) and not (vid_prefix in line):
                    mirror_pos_read = False
                    break


    if mirfit:
        
        # Pair frames to mirror positions
        met.pairFrame2MirPos()



    # Extract picks into pick objects
    met = extractPicks(met, mirfit=mirfit)

    return met

          



def solveTrajectoryMet(met, solver='original', velmodel=3, **kwargs):
        """ Runs the trajectory solver on points of the given type. 

        Keyword arguments:
            solver: [str] Trajectory solver to use:
                - 'original' (default) - "in-house" trajectory solver implemented in Python
                - 'gural' - Pete Gural's PSO solver
            velmodel: [int] Velocity propagation model for the Gural solver
                0 = constant   v(t) = vinf
                1 = linear     v(t) = vinf - |acc1| * t
                2 = quadratic  v(t) = vinf - |acc1| * t + acc2 * t^2
                3 = exponent   v(t) = vinf - |acc1| * |acc2| * exp( |acc2| * t ) (default)
        """


        # Check that there are at least two stations present
        if len(met.sites) < 2:
            print('ERROR! The .met file does not contain multistation data!')

            return False



        time_data = {}
        theta_data = {}
        phi_data = {}

        # Go through all sites
        for site in met.sites:

            time_picks = []
            theta_picks = []
            phi_picks = []

            # Go through all picks
            for pick in met.picks_objs[site]:

                # Add the pick to the picks list
                theta_picks.append(pick.theta)
                phi_picks.append(pick.phi)

                # Add the time of the pick to a list
                time_picks.append(pick.unix_time)


            # Add the picks to the list of picks of both sites
            time_data[site] = np.array(time_picks).ravel()
            theta_data[site] = np.array(theta_picks).ravel()
            phi_data[site] = np.array(phi_picks).ravel()


        # Take the earliest time of all sites as the reference time
        ref_unix_time = min([time_data[key][0] for key in time_data.keys()])

        # Normalize all times with respect to the reference times
        for site in met.sites:
            time_data[site] = time_data[site] - ref_unix_time


        # Convert the reference Unix time to Julian date
        ts = int(ref_unix_time)
        tu = (ref_unix_time - ts)*1e6
        ref_JD = unixTime2JD(ts, tu)


        if solver == 'original':

            # Init the new trajectory solver object
            traj_solve = Trajectory(ref_JD, output_dir=met.dir_path, **kwargs)

        elif solver == 'gural':

            # Select extra keyword arguments that are present only for the gural solver
            gural_keys = ['max_toffset', 'nummonte', 'meastype', 'verbose', 'show_plots']
            gural_kwargs = {key: kwargs[key] for key in gural_keys if key in kwargs}

            # Init the new Gural trajectory solver object
            traj_solve = GuralTrajectory(len(met.sites), ref_JD, velmodel, verbose=1, output_dir=met.dir_path, 
                **gural_kwargs)


        # Infill trajectories from each site
        for site in met.sites:

            theta_picks = theta_data[site]
            phi_picks = phi_data[site]
            time_picks = time_data[site]

            lat = met.lat[site]
            lon = met.lon[site]
            elev = met.elev[site]

            traj_solve.infillTrajectory(phi_picks, theta_picks, time_picks, lat, lon, elev)


        print('Filling done!')


        # # Dump measurements to a file
        # traj_solve.dumpMeasurements(self.met.dir_path.split(os.sep)[-1] + '_dump.txt')


        # Solve the trajectory
        traj_solve.run()






if __name__ == "__main__":


    import argparse

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Run the trajectory solver on the given METAL or mirfit .met file. If mirfit is used, make sure to use the --mirfit option.")

    arg_parser.add_argument('met_path', nargs=1, metavar='MET_PATH', type=str, \
        help='Full path to the .met file.')

    arg_parser.add_argument('-m', '--mirfit', \
        help='Mirfit format instead of metal format.', action="store_true")

    arg_parser.add_argument('-s', '--solver', metavar='SOLVER', help="""Trajectory solver to use. \n
        - 'original' - Monte Carlo solver
        - 'gural0' - Gural constant velocity
        - 'gural1' - Gural linear deceleration
        - 'gural2' - Gural quadratic deceleration
        - 'gural3' - Gural exponential deceleration
         """, type=str, nargs='?', default='original')

    arg_parser.add_argument('-t', '--maxtoffset', metavar='MAX_TOFFSET', nargs=1, \
        help='Maximum time offset between the stations.', type=float)

    arg_parser.add_argument('-v', '--vinitht', metavar='V_INIT_HT', nargs=1, \
        help='The initial veloicty will be estimated as the average velocity above this height (in km). If not given, the initial velocity will be estimated using the sliding fit which can be controlled with the --velpart option.', \
        type=float)

    arg_parser.add_argument('-p', '--velpart', metavar='VELOCITY_PART', \
        help='Fixed part from the beginning of the meteor on which the initial velocity estimation using the sliding fit will start. Default is 0.25 (25 percent), but for noisier data this might be bumped up to 0.5.', \
        type=float, default=0.25)

    arg_parser.add_argument('-d', '--disablemc', \
        help='Do not use the Monte Carlo solver, but only run the geometric solution.', action="store_true")
    
    arg_parser.add_argument('-r', '--mcruns', metavar="MC_RUNS", nargs='?', \
        help='Number of Monte Carlo runs.', type=int, default=100)

    arg_parser.add_argument('-u', '--uncertgeom', \
        help='Compute purely geometric uncertainties.', action="store_true")
    
    arg_parser.add_argument('-g', '--disablegravity', \
        help='Disable gravity compensation.', action="store_true")

    arg_parser.add_argument('-l', '--plotallspatial', \
        help='Plot a collection of plots showing the residuals vs. time, lenght and height.', \
        action="store_true")

    arg_parser.add_argument('-i', '--imgformat', metavar='IMG_FORMAT', nargs=1, \
        help="Plot image format. 'png' by default, can be 'pdf', 'eps',... ", type=str, default='png')

    arg_parser.add_argument('-x', '--hideplots', \
        help="Don't show generated plots on the screen, just save them to disk.", action="store_true")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    ### Parse command line arguments ###

    met_path = os.path.abspath(cml_args.met_path[0])

    dir_path = os.path.dirname(met_path)

    # Check if the given directory is OK
    if not os.path.isfile(met_path):
        print('No such file:', met_path)
        sys.exit()



    max_toffset = None
    if cml_args.maxtoffset:
        max_toffset = cml_args.maxtoffset[0]

    velpart = None
    if cml_args.velpart:
        velpart = cml_args.velpart

    vinitht = None
    if cml_args.vinitht:
        vinitht = cml_args.vinitht[0]

    ### ###


    # Load the met file
    met = loadMet(*os.path.split(met_path), mirfit=cml_args.mirfit)


    # Run trajectory solver on the loaded .met file
    solveTrajectoryMet(met, solver=cml_args.solver, max_toffset=max_toffset, \
            monte_carlo=(not cml_args.disablemc), mc_runs=cml_args.mcruns, \
            geometric_uncert=cml_args.uncertgeom, gravity_correction=(not cml_args.disablegravity), 
            plot_all_spatial_residuals=cml_args.plotallspatial, plot_file_type=cml_args.imgformat, \
            show_plots=(not cml_args.hideplots), v_init_part=velpart, v_init_ht=vinitht)


    # print(met.scale_plates.items())

    # print(met.scale_plates['1'].M)
    # print(met.exact_plates['1'].M)

    # ### TEST PLATES
    # from Formats.Plates import plateExactMap, plateScaleMap
    
    # # Test scale plate
    # hu, hv = plateScaleMap(met.scale_plates['1'], 100, 100)

    # # Reverse map
    # print(plateScaleMap(met.scale_plates['1'], hu, hv, reverse_map=True))
    # print(hu, hv)

    # # Test exact plate
    # hx, hy = plateExactMap(met.exact_plates['2'], 31997, 22290)

    # print(plateExactMap(met.exact_plates['2'], hx, hy, reverse_map=True))

    # ############ TEST IMAGE

    # site = '2'

    # #hx_centre = 34852
    # hx_centre = 34847.8046875
    # #hy_centre = 34052
    # hy_centre = 34057.7265625

    # # mx = 10
    # # my = 10

    # # # Get image coordinates of the centroid
    # # cx = mx - 320
    # # cy = 240 - my

    # # # Get image offsets from encoder offsets
    # # hu, hv = plateScaleMap(met.scale_plates[site], cx, cy)

    # # # Calculate encoder offset from the centre
    # # hx = hx_centre + hu
    # # hy = hy_centre + hv

    # # # Calculate the pick location from (theta, phi) to mirror encoder coordinates
    # # theta, phi = plateExactMap(met.exact_plates[site], hx, hy)

    # # print 'pick theta, phi:', map(np.degrees, (theta, phi))


    # # # Image coord test
    # # th_test, phi_test = plateExactMap(met.exact_plates[site], hx_centre, hy_centre)
    # # print 'Centre theta, phi:', map(np.degrees, (th_test, phi_test))


    # ############ TEST IMAGE 2

    # theta = np.radians(43.948964)
    # phi = np.radians(90.955409)

    # # Calculate the pick location from (theta, phi) to mirror encoder coordinates
    # hx, hy = plateExactMap(met.exact_plates[site], theta, phi, reverse_map=True)

    # # Calculate encoder offset from the centre
    # hu = hx - hx_centre
    # hv = hy - hy_centre

    # # Get image offsets from encoder offsets
    # nx, ny = plateScaleMap(met.scale_plates[site], hu, hv, reverse_map=True)

    # # Get image coordinates of the centroid
    # mx = 320 + nx
    # my = 240 - ny

    # print('mx, my:', mx, my)