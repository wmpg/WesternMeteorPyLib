from __future__ import print_function, division, absolute_import

import sys
import os
import copy
import numpy as np
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg
import scipy.optimize
import scipy.ndimage
import scipy.misc

from Formats.Met import loadMet
from Formats.Vid import readVid
from Formats.Plates import plateExactMap, plateScaleMap, loadScale, loadExact

from Utils.Math import polarToCartesian, cartesianToPolar, rotatePoint, pointInsidePolygon
from Utils.GreatCircle import greatCircle, fitGreatCircle
from Utils.TrajConversions import unixTime2JD

from Trajectory.GuralTrajectory import GuralTrajectory
from Trajectory.Trajectory import Trajectory





class TrajPoints(object):
    """ Container for a trajectory solution. """

    def __init__(self):

        # A list of picks used in the solution, for all sites
        self.picks = {}

        self.sites_geoloc = {}

        # Geocentric parameters
        self.ra = 0
        self.dec = 0
        self.vg = 0

        self.vinf = 0

        # Orbital elements
        self.q = 0
        self.e = 0
        self.i = 0
        self.peri = 0
        self.node = 0



class PickInfo(object):
    """ Container of information for individual picks. """

    def __init__(self, theta, phi):

        # Pick frame
        self.frame = 0

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

        # Great circle parameters
        self.theta0 = 0
        self.phi0 = 0

        # Phase of the great circle
        self.phase = 0

        # Coordinates calculated from the great circle pick
        self.theta_gc = 0
        self.phi_gc = 0
        
        self.cx_gc = 0
        self.cy_gc = 0

        self.x_gc = 0
        self.y_gc = 0
        self.z_gc = 0

        # Position after Monte Carlo uncertanties have been applied
        self.x_draw = 0
        self.y_draw = 0
        self.z_draw = 0
        self.theta_draw = 0
        self.phi_draw = 0
        



    def calcProjection(self):
        """ Calculates the projection of the pick to the great circle. """

        # Find the phase angle of the closes point on great circle to the original point
        self.phase = greatCirclePhase(self.theta, self.phi, self.theta0, self.phi0)

        # Calculate the projected point
        self.x_gc, self.y_gc, self.z_gc = greatCircle(self.phase, self.theta0, self.phi0)

        # Convert the point to polar coordinates
        self.theta_gc, self.phi_gc = cartesianToPolar(self.x_gc, self.y_gc, self.z_gc)

        


class ImageRotate(object):
    """ Rotate the given image, and provide methods for mapping points to the rotated image. """

    def __init__(self, img, angle):

        self.angle = angle

        # Rotate image by the calculated angle
        #self.img_rot = scipy.misc.imrotate(img, np.degrees(self.angle))
        self.img_rot = scipy.ndimage.rotate(img, np.degrees(self.angle))

        # Save original image shape
        self.img_ht = img.shape[0]
        self.img_wid = img.shape[1]

        # Calculate image enlargement from rotation
        self.y_diff = (self.img_rot.shape[0] - self.img_ht)/2.0 - 1
        self.x_diff = (self.img_rot.shape[1] - self.img_wid)/2.0


    def rotatePoint(self, x, y):
        """ Rotate the point from non-rotated coordinates to rotated image coordinates. """

        # Rotate the centroid and plot
        x_rot, y_rot = rotatePoint((self.img_wid/2, self.img_ht/2), (x, y), -self.angle)

        return x_rot + self.x_diff, y_rot + self.y_diff


    def reversePoint(self, x, y):
        """ Rotate the point back from the rotated image to non-rotated coordinates. """

        x_rev = x - self.x_diff
        y_rev = y - self.y_diff

        return rotatePoint((self.img_wid/2, self.img.ht/2), (x_rev, y_rev), self.angle)







class MonteCarloPicks(object):
    """ Class for handling Monte carlo picks. """
    
    def __init__(self, met):

        # Met object which contains the picks and calibration plates
        self.met = met

        # Init great circle parameters (both sites)
        self.theta0 = {}
        self.phi0 = {}

        # Great circle normal (both sites)
        self.N = {}

        # Pick uncertanties in transverse and longitudinal directions (both sites)
        self.transverse_std = {} # Uncertanty in GC inclination (theta0)
        self.longitudinal_std = {} # Uncertanty in GC phase

        # Direction of progressing phase (depending if the phase of the GC is positive as the meteor 
        # progresses through time, it can be 1 (positive) or -1 (negative) phase progression)
        self.phase_direction = {}

        # Dictionary of lists of pick objects, with full pick info
        self.picks_full = {}

        # A list of drawn trajectories
        self.trajectories = []

        # Handles to vid file objects
        self.vids = {}

        # Load vid files for both sites
        self.loadVidFiles()



    def loadVidFiles(self):
        """ Loads vid files from both sites. """

        for site in met.sites:

            # Open the vid file from this site
            self.vids[site] = readVid(self.met.dir_path, self.met.vids[site])



    def fitGreatCircle(self):
        """ Fits a great circle to the picks. """

        # Go though all sites
        for site in self.met.sites:

            # Extract frames
            frames = np.array(met.picks[site])[:,0]

            # Extract mirror positions on each frame
            hx_data, hy_data = np.hsplit(np.array(met.picks[site])[:,22:24], 2)

            # Extract original centroids
            cx_data, cy_data = np.hsplit(np.array(met.picks[site])[:,2:4], 2)

            # Extract UNIX time
            ts_data, tu_data = np.hsplit(np.array(met.picks[site])[:,11:13], 2)
            time_data = ts_data + tu_data/1000000

            # Init theta, phi arrays
            theta = np.zeros_like(cx_data)
            phi = np.zeros_like(cx_data)

            # Calculate (theta, phi) from image coordinates
            for i in range(len(cx_data)):
                theta[i], phi[i] = coordinatesImageToSky(cx_data[i], cy_data[i], met.exact_plates[site], 
                    met.scale_plates[site], hx_data[i], hy_data[i])

            # Convert (theta, phi) to Cartesian coordinates
            x, y, z = polarToCartesian(phi, theta)

            # Add (0, 0, 0) to the data (as the great circel should go through the origin)
            x = np.append(x, 0)
            y = np.append(y, 0)
            z = np.append(z, 0)

            # Regular grid covering the domain of the data
            X,Y = np.meshgrid(np.arange(-1.0, 1.0, 0.1), np.arange(-1.0, 1.0, 0.1))

            # Fit the great circle
            C, theta0, phi0 = fitGreatCircle(x, y, z)

            # Store great circle parameters
            self.theta0[site], self.phi0[site] = theta0, phi0

            print('GC params:', theta0, phi0)

            # evaluate it on grid
            Z = C[0]*X + C[1]*Y + C[2]

            # plot points and fitted surface
            fig = plt.figure()
            ax = fig.gca(projection='3d')

            # Plot the original points
            ax.scatter(x, y, z)

            # Plot the best fit plane
            ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)

            # Plot fitted great circle
            t_array = np.arange(0, 2*np.pi, 0.01)
            ax.scatter(*greatCircle(t_array, theta0, phi0), c='b', s=5)

            # Plot the zero of the great circle
            ax.scatter(*greatCircle(0, theta0, phi0), c='r', s=100)

            # Define plane normal
            N = greatCircle(np.pi/2.0, theta0+np.pi/2.0, phi0)

            # Store as great circle normal
            self.N[site] = N

            # Plot the plane normal
            ax.scatter(*N, c='g', s=100)

            print('Normal', N)

            # Calculate STDDEV of angle deviations from picks to the plane
            sin_s = 0
            cos_s = 0

            for pick in np.c_[x, y, z][:-1]:

                dev_angle = vectorAngle(pick, N)-np.pi/2
                # print np.degrees(dev_angle)*3600

                sin_s += np.sin(dev_angle)
                cos_s += np.cos(dev_angle)

            sin_s = sin_s/len(met.picks[site])
            cos_s = cos_s/len(met.picks[site])

            stddev = np.sqrt(-np.log(sin_s**2 + cos_s**2))

            print('GC stddev in arcsec: ', np.degrees(stddev)*3600)

            # Store as stddev in transverse direction
            self.transverse_std[site] = stddev


            #########
            ### TO DO:
            ### DETERMINE LONGITUDINAL UNCERTANTY (20 arcsecs TEMP value)
            self.longitudinal_std[site] = np.radians(20/3600.0)

            ########

            # Init a list of picks for this site
            self.picks_full[site] = []

            # Generate a list of pick object
            for fr, cx, cy, hx, hy, theta_pick, phi_pick, unix_time in zip(frames, cx_data, cy_data, hx_data, hy_data, theta, phi, time_data):
                
                # Init a new pick
                pick = PickInfo(theta_pick, phi_pick)

                # Set the pick frame
                pick.frame = int(fr)

                # Set pick UNIX time
                pick.unix_time = unix_time

                # Set original pick centroids
                pick.cx = cx
                pick.cy = cy

                # Set mirror position of frame centre
                pick.hx = hx
                pick.hy = hy

                # Set the great circle parameters of the pick
                pick.theta0 = self.theta0[site]
                pick.phi0 = self.phi0[site]

                # Calculate the pick projection to the great circle
                pick.calcProjection()

                # Calculate image position of the great circle pick
                pick.cx_gc, pick.cy_gc = coordinatesSkyToImage(pick.theta_gc, pick.phi_gc, 
                    self.met.exact_plates[site], self.met.scale_plates[site], pick.hx, pick.hy)

                # Add the pick to the list of all picks
                self.picks_full[site].append(pick)


            # Plot the projections of original points to the fitted great circle
            for pick in self.picks_full[site]:

                print(pick.theta, pick.phi, pick.phase)
                ax.scatter(pick.x_gc, pick.y_gc, pick.z_gc, c='yellow')



            # ### Plot the projection of the first point on the great circle
            
            # # Get the phase of the great circle for the fist point
            # phase0 = greatCirclePhase(theta[0], phi[0], theta0, phi0)

            # print 'Phase:', phase0

            # ax.scatter(*greatCircle(phase0, theta0, phi0), c='yellow', s=200)
            # ###
            

            plt.xlim([-1, 1])
            plt.ylim([-1, 1])
            ax.set_zlim(-1, 1)

            ax.set_aspect('equal')

            # plt.show()
            plt.clf()
            #plt.close()

            plt.plot(theta, phi)
            gc_theta, gc_phi = cartesianToPolar(*greatCircle(t_array, theta0, phi0))
            plt.plot(gc_theta, gc_phi, color='r')

            # plt.show()
            plt.close()



    def determinePhaseProgression(self):
        """ Determines if the meteor progresses through time with the rising GC phase or not. """

        # Do this for both sites
        for site in self.met.sites:

            # Take the first and the last pick
            fp = self.picks_full[site][0]
            lp = self.picks_full[site][-1]

            # Calculate the angle between picks
            dist = vectorAngle(np.r_[fp.x_gc, fp.y_gc, fp.z_gc], np.r_[lp.x_gc, lp.y_gc, lp.z_gc])

            # Take a small step in phase forward from the first pick
            sp = greatCircle(fp.phase+np.radians(0.5), fp.theta0, fp.phi0)

            # Check if the the rising phase is approaching the last pick or not
            if vectorAngle(np.r_[sp], np.r_[lp.x_gc, lp.y_gc, lp.z_gc]) < dist:

                # If the distance to the last pick is smaller with the positive phase step, set phase as +
                self.phase_direction[site] = 1

            else:

                # Set phase progression as negative
                self.phase_direction[site] = -1




    def samplePicks(self, n):
        """ Sample picks from distributions defined by the transverse and longitudinal uncertanties. """

        # Determine the directon of phase progression of the meteor through time
        self.determinePhaseProgression()

        # Draw n number of trajectories
        for i in range(n):

            # Init a new trajectory on each drawing
            traj = TrajPoints()

            # Draw picks for all sites
            for site in self.met.sites:

                # Init the trajectory picks for this site
                traj.picks[site] = []

                # Go through all picks in this site
                for pick in self.picks_full[site]:

                    # Make a new pick copy
                    new_pick = copy.deepcopy(pick)

                    # Generate a transverse pick uncertanty
                    trans_dev = np.random.normal(0, self.transverse_std[site])

                    # Generate a longitudinal uncertanty (only take the front half of the Gaussian)
                    longi_dev = np.abs(np.random.normal(0, self.longitudinal_std[site]))

                    # Draw a pick from the great circle with the drawn offset
                    draw_x, draw_y, draw_z = greatCircle(new_pick.phase + self.phase_direction[site]*longi_dev,
                        new_pick.theta0+trans_dev, new_pick.phi0)

                    # Set the drawn coordinates to the pick
                    new_pick.draw_x, new_pick.draw_y, new_pick.draw_z = draw_x, draw_y, draw_z

                    # Calculate the polar coordinates of the drawn pick
                    new_pick.theta_draw, new_pick.phi_draw = cartesianToPolar(draw_x, draw_y, draw_z)

                    # Add pick to the trajectory
                    traj.picks[site].append(new_pick)


            # Add the drawn trajectory to the list of trajectories
            self.trajectories.append(traj)



    def extractPicks(self, site, pick_type='original'):
        """ Returns a list of original picks for the given site. 
    
        Arguments: 
            site: [name] name of the site (integer, float, str, whatever is the key name)

        Keyword arguments:
            pick_type: [str] Can be:
                        - 'original' (default) original manual picks
                        - 'gc' original picks projected onto a great circle
                        - 'draw' picks drawn from a probability distribution

        Return:
            time_picks, theta_picks, phi_picks
        """

        time_picks = []
        theta_picks = []
        phi_picks = []

        # Go through all picks
        for pick in self.picks_full[site]:

            # Go though all possible pick types and select the given type of pick
            if pick_type == 'original':
                theta_pt = pick.theta
                phi_pt = pick.phi

            elif pick_type == 'gc':
                theta_pt = pick.theta_gc
                phi_pt = pick.phi_gc

            elif pick_type == 'draw':
                theta_pt = pick.theta_draw
                phi_pt = pick.phi_draw

            else:
                print("Error! No pick of type:", pick_type)
                sys.exit()

            # Add the pick to the picks list
            theta_picks.append(theta_pt)
            phi_picks.append(phi_pt)

            # Add the time of the pick to a list
            time_picks.append(pick.unix_time)


        return time_picks, theta_picks, phi_picks



    def showVidPicks(self, site):
        """ Shows frames, picks and stars of the given site. """
        

        # Go through picks
        for pick in self.picks_full[site]:


            # Print picks info
            print('PICK frame', pick.frame)
            print('Cx, Cy', pick.cx, pick.cy)
            

            # Extract the desired frame
            img = self.vids[site].frames[pick.frame]

            # Show the image
            plt.imshow(img, cmap='inferno')#, vmin=0, vmax=255)

            print('Theta, phi', np.degrees(pick.theta), np.degrees(pick.phi))
            print('GC', np.degrees(pick.theta_gc), np.degrees(pick.phi_gc))

            # # Convert theta, phi back to image coordinates
            # cx_calc, cy_calc = coordinatesSkyToImage(pick.theta, pick.phi, self.met.exact_plates[site], 
            #     self.met.scale_plates[site], pick.hx, pick.hy)
            
            print('centre hx, hy', pick.hx, pick.hy)
            # print 'reverse mapping', cx_calc, cy_calc

            # Plot the current pick location
            # plt.scatter(cx_calc, cy_calc, c='b', marker='x', s=50)

            print('GC pick img:', pick.cx_gc, pick.cy_gc)

            # Plot great circle fitted pick location
            plt.scatter(pick.cx_gc, pick.cy_gc, c='g', marker='x', s=50)

            # Plot original pick locations
            plt.scatter(pick.cx, pick.cy, c='r', marker='x', s=50)



            ###############

            # Get the position of the top long. uncertanty, left transverse position
            phase_tl = pick.phase + self.phase_direction[site]*self.longitudinal_std[site]
            cx_tl, cy_tl = greatCircleToImage(phase_tl, pick.theta0 - 2*self.transverse_std[site], pick.phi0, 
                self.met.exact_plates[site], self.met.scale_plates[site], pick.hx, pick.hy)

            plt.scatter(cx_tl, cy_tl, c='yellow', marker='x', s=50)

            # Get the position of the top long. uncertanty, right transverse position
            phase_tr = pick.phase + self.phase_direction[site]*self.longitudinal_std[site]
            cx_tr, cy_tr = greatCircleToImage(phase_tr, pick.theta0 + 2*self.transverse_std[site], pick.phi0, 
                self.met.exact_plates[site], self.met.scale_plates[site], pick.hx, pick.hy)

            plt.scatter(cx_tr, cy_tr, c='yellow', marker='x', s=50)

            # Get the position of the bottom long. uncertanty, left transverse position
            phase_bl = pick.phase
            cx_bl, cy_bl = greatCircleToImage(phase_bl, pick.theta0 - 2*self.transverse_std[site], pick.phi0, 
                self.met.exact_plates[site], self.met.scale_plates[site], pick.hx, pick.hy)

            plt.scatter(cx_bl, cy_bl, c='yellow', marker='x', s=50)

            # Get the position of the bottom long. uncertanty, right transverse position
            phase_br = pick.phase
            cx_br, cy_br = greatCircleToImage(phase_br, pick.theta0 + 2*self.transverse_std[site], pick.phi0, 
                self.met.exact_plates[site], self.met.scale_plates[site], pick.hx, pick.hy)

            plt.scatter(cx_br, cy_br, c='yellow', marker='x', s=50)

            ###############


            # # Get coordinates of the max. longitudinal uncertanty
            # phase_max = pick.phase + self.phase_direction[site]*self.longitudinal_std[site]
            # x, y, z = greatCircle(phase_max, pick.theta0, pick.phi0)
            # theta_max, phi_max = cartesianToPolar(x, y, z)

            # # Get he image position of the max. longitudinal uncertanty
            # cx_max, cy_max = coordinatesSkyToImage(theta_max, phi_max, self.met.exact_plates[site], 
            # self.met.scale_plates[site], pick.hx, pick.hy)

            # # Calculate the angle of the vector made by the original GC pick and the max. long. pick
            # angle = np.arctan2(pick.cy_gc - cy_max, pick.cx_gc - cx_max)

            # print 'Angle:', np.degrees(angle)

            # # Plot max. longitudinal uncertany position
            # plt.scatter(cx_max, cy_max, c='white', marker='x', s=50)
            

            # Plot star positions which are in the FOV
            for star in self.met.stars[site]:

                # Get the image position of the star
                star_x, star_y = coordinatesSkyToImage(star.theta, star.phi, self.met.exact_plates[site], 
                self.met.scale_plates[site], pick.hx, pick.hy)

                # Check if the star is inside the image
                if (star_x > 0) and (star_x < self.met.scale_plates[site].wid):
                    if (star_y > 0) and (star_y < self.met.scale_plates[site].ht):
                        plt.scatter(star_x, star_y, marker='D', s=50, facecolors='none', edgecolors='purple')

                        plt.text(star_x, star_y+8, star.name, color='purple', size=7, va='top', ha='center')


            plt.title('Frame: '+str(pick.frame))

            # Set plot limits
            plt.xlim(0, self.met.scale_plates[site].wid)
            plt.ylim(self.met.scale_plates[site].ht, 0)

            plt.show()
            plt.clf()


            # # Init the object for image rotation
            # img_rot = ImageRotate(img, angle)

            # # Show rotated image
            # plt.imshow(img_rot.img_rot, cmap='inferno')


            # # Rotate the centroid with the rotated image
            # cx_rot, cy_rot = img_rot.rotatePoint(pick.cx_gc, pick.cy_gc)

            # # Plot the centroid
            # plt.scatter(cx_rot, cy_rot, c='green', marker='x', s=50)

            # # Plot the forward position
            # plt.scatter(*img_rot.rotatePoint(cx_max, cy_max), c='white', marker='x', s=50)

            # # Plot the original centroid
            # plt.scatter(*img_rot.rotatePoint(pick.cx, pick.cy), c='red', marker='x', s=50)            

            # plt.show()

            # plt.clf()



            #sys.exit()
            #break


        def extractVidDistribution(self, site):
            """ Extracts parts of video frames from vid file as distributions for sampling pick locations. 
                Normalized video intensities serve as weights in the distribution.
        
            """

            # Go through picks
            for pick in self.picks_full[site]:


                # Print picks info
                print('PICK frame', pick.frame)
                print('GC: Cx, Cy', pick.cx_gc, pick.cy_gc)
                
                # Get coordinates of the max. longitudinal uncertanty
                phase_max = pick.phase + self.phase_direction[site]*self.longitudinal_std[site]
                x, y, z = greatCircle(phase_max, pick.theta0, pick.phi0)
                theta_max, phi_max = cartesianToPolar(x, y, z)

                # Get the image position of the max. longitudinal uncertanty
                cx_max, cy_max = coordinatesSkyToImage(theta_max, phi_max, self.met.exact_plates[site], 
                self.met.scale_plates[site], pick.hx, pick.hy)

                # Calculate the angle of the vector made by the original GC pick and the max. long. pick
                angle = np.arctan2(pick.cy_gc - cy_max, pick.cx_gc, cx_max)

                print('Angle:', np.degrees(angle))


                # Extract the desired frame
                img = self.vids[site].frames[pick.frame]



                ##### TEMP!!
                break


    def solveTrajectory(self, pick_type='original', velmodel=3, solver='original'):
        """ Runs the trajectory solver on points of the given type. 

        Keyword arguments:
            pick_type: [str] Can be:
                - 'original' (default) original manual picks
                - 'gc' original picks projected onto a great circle
                - 'draw' picks drawn from a probability distribution
            velmodel: [int] Velocity propagation model
                0 = constant   v(t) = vinf
                1 = linear     v(t) = vinf - |acc1| * t
                2 = quadratic  v(t) = vinf - |acc1| * t + acc2 * t^2
                3 = exponent   v(t) = vinf - |acc1| * |acc2| * exp( |acc2| * t ) (default)
            solver: [str] Trajectory solver to use:
                - 'original' (default) - "in-house" trajectory solver implemented in Python
                - 'gural' - Pete Gural's PSO solver
        """

        time_data = {}
        theta_data = {}
        phi_data = {}

        # Go through all sites
        for site in self.met.sites:

            # Extract picks of the given type
            time_picks, theta_picks, phi_picks = self.extractPicks(site, pick_type=pick_type)

            # Add the picks to the list of picks of both sites
            time_data[site] = np.array(time_picks).ravel()
            theta_data[site] = np.array(theta_picks).ravel()
            phi_data[site] = np.array(phi_picks).ravel()


        # Take the earliest time of all sites as the referent time
        ref_unix_time = min([time_data[key][0] for key in time_data.keys()])

        # Normalize all times with respect to the referent times
        for site in self.met.sites:
            time_data[site] = time_data[site] - ref_unix_time


        # Convert the referent Unix time to Julian date
        ts = int(ref_unix_time)
        tu = (ref_unix_time - ts)*1e6
        ref_JD = unixTime2JD(ts, tu)


        if solver == 'original':

            # Init the new trajectory solver object
            traj_solve = Trajectory(ref_JD, show_plots=True, output_dir=self.met.dir_path)

        elif solver == 'gural':

            # Init the new Gural trajectory solver object
            traj_solve = GuralTrajectory(len(self.met.sites), ref_JD, velmodel, verbose=1)


        # Infill trajectories from each site
        for site in self.met.sites:

            theta_picks = theta_data[site]
            phi_picks = phi_data[site]
            time_picks = time_data[site]

            lat = self.met.exact_plates[site].lat
            lon = self.met.exact_plates[site].lon
            elev = self.met.exact_plates[site].elev

            traj_solve.infillTrajectory(phi_picks, theta_picks, time_picks, lat, lon, elev)


        print('Filling done!')


        # # Dump measurements to a file
        # traj_solve.dumpMeasurements(self.met.dir_path.split(os.sep)[-1] + '_dump.txt')


        # Solve the trajectory
        traj_solve.run()
            



            


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


def greatCircleToImage(phase, theta0, phi0, exact, scale, hx_centre, hy_centre):
    """ Take a point on the great circle, and return its coordinates on an image. """

    # Get the great circle position in Cartesian coordinates
    x, y, z = greatCircle(phase, theta0, phi0)

    # Convert the GC cartesian coordinates to polar
    theta, phi = cartesianToPolar(x, y, z)

    # Get the image position of a point on the GC
    cx, cy = coordinatesSkyToImage(theta, phi, exact, scale, hx_centre, hy_centre)

    return cx, cy




def vectorAngle(v1, v2):
    """ Calculates the angle between two vectors. """

    cos_angle = np.dot(v1, v2)
    sin_angle = np.linalg.norm(np.cross(v1, v2))
    return np.arctan2(sin_angle, cos_angle)



def greatCirclePhase(theta, phi, theta0, phi0):
    """ Find the phase angle of the point closest to the given point on the great circle. """

    def _pointDist(x):
        """ Calculates the Cartesian distance from a point defined in polar coordinates, and a point on
            a great circle. """
        
        # Convert the pick to Cartesian coordinates
        point = polarToCartesian(phi, theta)

        # Get the point on the great circle
        circle = greatCircle(x, theta0, phi0)

        # Return the distance from the pick to the great circle
        return np.sqrt((point[0] - circle[0])**2 + (point[1] - circle[1])**2 + (point[2] - circle[2])**2)

    # Find the phase angle on the great circle which corresponds to the pick
    res = scipy.optimize.minimize(_pointDist, 0)

    return res.x



        


if __name__ == "__main__":

    ### Directory where the met file is

    # Long meteor
    # dir_path = os.path.join(os.path.curdir, "..", "MirfitPrepare", "20160929_062945_mir")

    # Short meteor
    
    # dir_path = os.path.join(os.path.curdir, "..", "MirfitPrepare", "20161007_052346_mir")

    # Long meteor
    # dir_path = os.path.join(os.path.curdir, "..", "MirfitPrepare", "20161007_052749_mir")

    # Leading fragment event
    # dir_path = os.path.join(os.path.curdir, "..", "MirfitPrepare", "20161009_064237_mir")

    # Short meteor
    dir_path = os.path.join(os.path.curdir, "..", "MirfitPrepare", "20170303_055055_mir")

    

    ###


    # Name of the met file
    file_name = 'state.met'

    # Load the met file
    met = loadMet(dir_path, file_name, mirfit=True)

    # # Load the plates
    # met.scale_plates[1] = loadScale(dir_path, 'scale_01.aff')
    # met.scale_plates[2] = loadScale(dir_path, 'scale_02.aff')
    # met.exact_plates[1] = loadExact(dir_path, 'exact_01.ast')
    # met.exact_plates[2] = loadExact(dir_path, 'exact_02.ast')
        
    # Init the picks class
    mc_picks = MonteCarloPicks(met)

    # Fit the picks to a great circle
    mc_picks.fitGreatCircle()

    # Run the trajectory solver
    #mc_picks.solveTrajectory(pick_type='original', solver='original')
    mc_picks.solveTrajectory(pick_type='original', solver='original')


    sys.exit()



    ###### Print out picks
    start_time = min(mc_picks.picks_full[1][0].unix_time, mc_picks.picks_full[2][0].unix_time)
    print(datetime.datetime.utcfromtimestamp(start_time))
    print("Reference time:", "{:20.10f}".format(start_time[0]))

    for site_no in [1, 2]:

        times = []
        thetas = []
        phis = []
        for pick in mc_picks.picks_full[site_no]:

            rel_time = pick.unix_time - start_time
            #print rel_time, pick.theta_gc, pick.phi_gc

            times.append(rel_time[0])
            thetas.append(pick.theta_gc[0])
            phis.append(pick.phi_gc[0])

        print('SITE:', site_no)
        print(times)
        print(list(map(np.degrees, thetas)))
        print(list(map(np.degrees, phis)))


    sys.exit()
    #####




    # Monte Carlo the picks and draw trajectories
    mc_picks.samplePicks(2)

    ###
    mc_picks.showVidPicks(2)
    ###

    # Go through all drawn trajectories
    for traj in mc_picks.trajectories:

        # Go through both sites
        for site in mc_picks.met.sites:

            # Plot the picks
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            plt.title('Site: '+str(site))

            # Plot all picks
            for pick in traj.picks[site]:



                # Plot the original pick
                ax.scatter(*polarToCartesian(pick.theta, pick.phi), c='b')

                # Plot the drawn picks
                ax.scatter(*polarToCartesian(pick.theta_draw, pick.phi_draw), c='r')

                #print pick.theta, pick.theta_draw, pick.phi, pick.phi_draw

            plt.show()
            plt.close()
            