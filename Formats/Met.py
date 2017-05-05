""" Loading and hadling Mirfit *.met files. """

from __future__ import print_function, absolute_import, division

import os
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

from Formats.Plates import AffPlate, AstPlate


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




class MetStruct(object):
    """ Container for Met file info. """

    def __init__(self, dir_path):

        # Met file location
        self.dir_path = dir_path

        # Site geo coordinates
        self.sites_location = {}

        # Init picks
        self.picks = {}

        # Init mirror positions in time
        self.mirror_pos = {}

        # Init plates
        self.scale_plates = {}
        self.exact_plates = {}

        # Init star positions
        self.stars = {}

        # Vid files
        self.vids = {}


    def pairFrame2MirPos(self):
        """ Pairs frames to their respective mirror positions. """

        # Do this for both sites
        for site in self.sites:

            # Extract time data and encoder positions
            time_data = np.array(self.mirror_pos[site])[:,0]
            hx_data = np.array(self.mirror_pos[site])[:,1]
            hy_data = np.array(self.mirror_pos[site])[:,2]

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
            ret_str += 'Location: '+','.join(map(str, self.sites_location[site]))+'\n'
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
    met = MetStruct(dir_path)

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
                line = line.strip(site_prefix)
                line = line.split()

                # Add the site ID to the list of sites
                sites.append(line[1])

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
                    scale_data = map(float, scale_data[:-1])

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
                    exact_header = map(float, exact_data[3:12])

                    # Unpack fit parameters
                    exact_fit = map(lambda x: map(float, x.split(':')), exact_data[12:])

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
                        pick_data = map(float, pick_data)

                        met.picks[site].append(pick_data)

                    else:

                        # METAL-style pick

                        # Extract pick data
                        pick_data = line.replace(pick_prefix, '').split()[0::2]

                        # Check that it is a meteor pick
                        if 'type meteor' in line:
                            pick_data = map(float, pick_data[:29] + pick_data[31:])

                            met.picks[site].append(pick_data)


                # Check if vid file name
                if vid_prefix in line:

                    # Extract vid file data
                    video_data = line.replace(vid_prefix, '').split()[1::2]

                    # Extract vid file name
                    met.vids[site] = video_data[1].replace("'", "")

                    # Extract site location
                    met.sites_location[site] = map(float, video_data[12:15])

                    # Allow reading mirror positions
                    mirror_pos_read = True


                # Check if mirror position entry
                if mirror_pos_read and (mirror_pos_prefix in line):

                    # Extract mirror position data
                    mpos_data = line.replace(mirror_pos_prefix, '').split()[1::2]
                    mpos_data = map(float, mpos_data)

                    # Store mirror position
                    met.mirror_pos[site].append(mpos_data)


                # Break if another site is reached
                if mirror_pos_read and (vid_prefix_nosite in line) and not (vid_prefix in line):
                    mirror_pos_read = False
                    break


    if mirfit:
        
        # Pair frames to mirror positions
        met.pairFrame2MirPos()


    return met

            

if __name__ == "__main__":

    # Directory where the met file is
    dir_path = "/home/dvida/Dropbox/UWO Master's/Projects/MirfitPrepare/20160927_030027_mir"

    # Name of the met file
    file_name = 'state.met'

    # Load the met file
    met = loadMet(dir_path, file_name)

    print(met.scale_plates[1].M)
    print(met.exact_plates[1].M)

    ### TEST PLATES
    from Plates import plateExactMap, plateScaleMap
    
    # Test scale plate
    hu, hv = plateScaleMap(met.scale_plates[1], 100, 100)

    # Reverse map
    print(plateScaleMap(met.scale_plates[1], hu, hv, reverse_map=True))
    print(hu, hv)

    # Test exact plate
    hx, hy = plateExactMap(met.exact_plates[2], 31997, 22290)

    print(plateExactMap(met.exact_plates[2], hx, hy, reverse_map=True))

    ############ TEST IMAGE

    site = 2

    #hx_centre = 34852
    hx_centre = 34847.8046875
    #hy_centre = 34052
    hy_centre = 34057.7265625

    # mx = 10
    # my = 10

    # # Get image coordinates of the centroid
    # cx = mx - 320
    # cy = 240 - my

    # # Get image offsets from encoder offsets
    # hu, hv = plateScaleMap(met.scale_plates[site], cx, cy)

    # # Calculate encoder offset from the centre
    # hx = hx_centre + hu
    # hy = hy_centre + hv

    # # Calculate the pick location from (theta, phi) to mirror encoder coordinates
    # theta, phi = plateExactMap(met.exact_plates[site], hx, hy)

    # print 'pick theta, phi:', map(np.degrees, (theta, phi))


    # # Image coord test
    # th_test, phi_test = plateExactMap(met.exact_plates[site], hx_centre, hy_centre)
    # print 'Centre theta, phi:', map(np.degrees, (th_test, phi_test))


    ############ TEST IMAGE 2

    theta = np.radians(43.948964)
    phi = np.radians(90.955409)

    # Calculate the pick location from (theta, phi) to mirror encoder coordinates
    hx, hy = plateExactMap(met.exact_plates[site], theta, phi, reverse_map=True)

    # Calculate encoder offset from the centre
    hu = hx - hx_centre
    hv = hy - hy_centre

    # Get image offsets from encoder offsets
    nx, ny = plateScaleMap(met.scale_plates[site], hu, hv, reverse_map=True)

    # Get image coordinates of the centroid
    mx = 320 + nx
    my = 240 - ny

    print('mx, my:', mx, my)

