""" Given the locations and pointings of camera, optimize their pointings so they have a maximum overlap
    at the given range of heights.
"""

from __future__ import print_function, division, absolute_import

import copy
import datetime

import numpy as np
import matplotlib.pyplot as plt

import scipy.optimize

from TrajSim.ShowerSim import initStationList, plotStationFovs
from Utils.TrajConversions import datetime2JD
from Utils.Math import estimateHullOverlapRatio



# def _volumeRun(pointings, station_list, fixed_cameras, jd, max_height, min_height, niter=1):
#     """ Given the ooptimization parameters, calculate the common volume. The functions returns a negative
#         value for the volume because the minimum is being searched.
#     """

#     # Pointings index
#     k = 0

#     # Set the estimated FOV centre for moving cameras
#     for i, stat in enumerate(station_list):

#         if not fixed_cameras[i]:
            
#             # Set the azimuth
#             stat.azim_centre = pointings[k]
#             k += 1

#             # Set the elevation
#             stat.elev_centre = pointings[k]
#             k += 1


#     volumes = []

#     # Calculate the mean of several common volume runs
#     for i in range(niter):
#         vol = stationsCommonVolume(station_list, fixed_cameras, jd, max_height, min_height)
#         volumes.append(vol)


#     vol_mean = np.mean(volumes)

#     print()
#     print(np.degrees(pointings))
#     print(vol_mean)

    
#     return -vol_mean



def stationsCommonVolume(station_list, fixed_cameras, jd, max_height, min_height):
    """ Calculates the common volume between the stations FOVs. """

    common_volume = 0

    # Go through all fixed stations
    for i, stat_fixed in enumerate(station_list):

        # Check if this is the fixed camera
        if fixed_cameras[i]:

            # Go through all non-fixed stations
            for j, stat_moving in enumerate(station_list):

                # Check if this is the moving camera
                if not fixed_cameras[j]:

                    ### CALCULATE ECI COORDINATES OF FIXED STATION ###
                    ##########################################################################################
                    # Get ECI coordinates of the FOV at the given maximum height
                    fixed_top_eci_corners = stat_fixed.fovCornersToECI(jd, max_height)

                    # Get ECI coordinates of the FOV at the given minimum height
                    fixed_bottom_eci_corners = stat_fixed.fovCornersToECI(jd, min_height)

                    fixed_eci_corners = np.array(fixed_top_eci_corners + fixed_bottom_eci_corners)

                    ##########################################################################################


                    ### CALCULATE ECI COORDINATES OF MOVING STATION ###
                    ##########################################################################################
                    # Get ECI coordinates of the FOV at the given maximum height
                    moving_top_eci_corners = stat_moving.fovCornersToECI(jd, max_height)

                    # Get ECI coordinates of the FOV at the given minimum height
                    moving_bottom_eci_corners = stat_moving.fovCornersToECI(jd, min_height)

                    moving_eci_corners = np.array(moving_top_eci_corners + moving_bottom_eci_corners)

                    ##########################################################################################

                    # Calculate the common volume between the fixed and the moving camera
                    common_v = estimateHullOverlapRatio(fixed_eci_corners, moving_eci_corners, volume=True, niter=1000)

                    common_volume += common_v

    return common_volume



# def optimizePointingsFOV(station_list, fixed_cameras, min_height, max_height):
#     """ Optimize the pointings of the cameras by finding the pointings where the common volume of the sky at
#         given height is maximized. It is assumed that one of the cameras are fixes, while all other can be
#         moved.

#     """

#     # Check that at least one camera is fixed
#     if not (True in fixed_cameras):
#         print("""At least one camera must be fixed! Check the fixed_cameras variable, at least one entry 
#             there should be True.""")

#         return station_list

#     # Check that at least one camera can be moved
#     if not (False in fixed_cameras):
#         print("""At least one camera must be non-fixed! Check the fixed_cameras variable, at least one entry 
#             there should be False.""")

#         return station_list

#     # Use the current Julian date as the referent time (this is just used to convert the coordinated to ECI,
#     # it has no operational importance whatsoever).
#     jd = datetime2JD(datetime.datetime.now())

    
#     # Construct the initial parameters list
#     p0 = np.array([[stat.azim_centre, stat.elev_centre] for (i, stat) in enumerate(station_list) \
#         if not fixed_cameras[i]]).ravel()

#     # Set the bounds for every parameter (azimuth from 0 to 2pi, elevation from 0 to pi)
#     bounds = [[0, np.pi] if i%2 else [0, 2*np.pi] for i in range(len(p0))]

    
#     # Find the pointings with the largest common volume
#     res = scipy.optimize.minimize(_volumeRun, p0, bounds=bounds, args=(station_list, fixed_cameras, jd, max_height, min_height))

#     print(res)
#     print(np.degrees(res.x))




def explorePointings(station_list, fixed_cameras, min_height, max_height, moving_ranges, steps):
    """ Given the list of cameras, a range of heights and a range of possible movements for the camera,
        construct a map of volume overlaps for each camera position. The map will be saves as an image.

    Arguments:
        station_list: [list] A list of SimStation objects.
        fixed_cameras: [list] A list of bools indiciating if the camera is fixed or it can be moved to
            optimize the overlap.
        min_height: [float] Minimum height of the FOV polyhedron (meters).
        max_height: [float] Maximum height of the FOV polyhedron (meters).
        moving_ranges: [list] A list of possible movement for each non-fixed camera (degrees).
        steps: [int] Steps to take inside the moving range. The map will thus have a resolution of
            range/steps.

    Return:
        None
    """

    station_list = copy.deepcopy(station_list)


    k = 0

    # Use the current Julian date as the referent time (this is just used to convert the coordinated to ECI,
    # it has no operational importance whatsoever).
    jd = datetime2JD(datetime.datetime.now())

    # Go through all moving cameras
    for i, stat_moving in enumerate(station_list):

        if not fixed_cameras[i]:

            # Original centre pointing
            azim_centre_orig = stat_moving.azim_centre
            elev_centre_orig = stat_moving.elev_centre

            # Get the range of movements for this camera
            mv_range = np.radians(moving_ranges[k])
            k += 1


            volume_results = []

            d_range = np.linspace(-mv_range/2.0, +mv_range/2, steps)

            # Make a grid of movements
            for d_elev in d_range:
                for d_azim in d_range:

                    # Calculate the azimuth of centre
                    azim = (azim_centre_orig + d_azim)%(2*np.pi)

                    # Calculate the elevation of the centre
                    elev = elev_centre_orig + d_elev

                    if elev > np.pi/2:
                        elev = (np.pi/2 - elev)%(np.pi/2)

                    if elev < 0:
                        elev = np.abs(elev)

                    # Set the new centre to the station
                    stat_moving.azim_centre = azim
                    stat_moving.elev_centre = elev

                    # Assign the changed parameter to the moving camera
                    station_list[i] = stat_moving

                    # Estimate the volume only for this moving camera
                    fix_cam = [True]*len(station_list)
                    fix_cam[i] = False

                    # Estimate the intersection volume
                    vol = stationsCommonVolume(station_list, fix_cam, jd, max_height, min_height)

                    volume_results.append([azim, elev, vol])

                    print('Azim {:.2f} elev {:.2f} vol {:e}'.format(np.degrees(azim), np.degrees(elev), vol))



            volume_results = np.array(volume_results)

            azims = volume_results[:, 0].reshape(len(d_range), len(d_range))
            elevs = volume_results[:, 1].reshape(len(d_range), len(d_range))

            # Select only the volumes
            vols = volume_results[:, 2].reshape(len(d_range), len(d_range))

            # Find the index of the largest volume
            vol_max = np.unravel_index(vols.argmax(), vols.shape)

            # Print the largest overlapping volume
            print('MAX OVERLAP:')
            print('Azim {:.2f} elev {:.2f} vol {:e}'.format(np.degrees(azims[vol_max]), np.degrees(elevs[vol_max]), vols[vol_max]))


            plt.imshow(vols/1e9, extent=np.degrees([azim_centre_orig + np.min(d_range), azim_centre_orig + np.max(d_range), elev_centre_orig + np.max(d_range), \
                elev_centre_orig + np.min(d_range)]))

            plt.gca().invert_yaxis()

            plt.xlabel('Azimuth (deg)')
            plt.ylabel('Elevation (deg)')

            plt.colorbar(label='Common volume (km^3)')

            plt.savefig('fov_map_' + stat_moving.station_id + '_ht_range_' + str(min_height) + '_' + str(max_height) + '.png', dpi=300)

            #plt.show()

            plt.clf()
            plt.close()
            




if __name__ == "__main__":


    ### STATION PARAMETERS ###
    ##########################################################################################################

    # Number of stations in total
    n_stations = 2

    # Geographical coordinates of stations (lat, lon, elev, station_id) in degrees and meters
    stations_geo = [
        [43.26420, -80.77209, 329.0, 'tavis'], # Tavis
        [43.19279, -81.31565, 324.0, 'elgin']  # Elgin
        ]


    ### CAMO WIDE ###
    #################

    # Azimuths of centre of FOVs (degrees)
    azim_fovs = [326.823, 1.891]

    # Elevations of centre of FOVs (degrees)
    elev_fovs = [41.104, 46.344]

    # # Azimuths of centre of FOVs (degrees)
    # azim_fovs = [338.823, 1.891]

    # # Elevations of centre of FOVs (degrees)
    # elev_fovs = [45.104, 46.344]


    # Cameras FOV widths (degrees)
    fov_widths = [19.22, 19.22]

    # Cameras FOV heights (degrees)
    fov_heights = [25.77, 25.77]


    # If the camera FOV is fixed, it should have True at its index, and False if it can be moved to optimize
    # the overlap
    fixed_cameras = [True, False]

    # Height range to optimize for (kilometers)
    min_height = 70
    max_height = 120

    #################



    # ### CAMO MIRRORS ###
    # #################

    # # Azimuths of centre of FOVs (degrees)
    # azim_fovs = [335.29, 358.3]

    # # Elevations of centre of FOVs (degrees)
    # elev_fovs = [43.912, 46.643]


    # # Cameras FOV widths (degrees)
    # fov_widths = [40.0, 39.0]

    # # Cameras FOV heights (degrees)
    # fov_heights = [40.0, 39.0]

    # # If the camera FOV is fixed, it should have True at its index, and False if it can be moved to optimize
    # # the overlap
    # fixed_cameras = [False, True]

    # # Height range to optimize for (kilometers)
    # min_height = 70
    # max_height = 120

    # #################



    # ### EMCCD ###
    # #################

    # # Azimuths of centre of FOVs (degrees)
    # azim_fovs = [315, 16.75]

    # # Elevations of centre of FOVs (degrees)
    # elev_fovs = [63.5, 65.5]


    # # Cameras FOV widths (degrees)
    # fov_widths = [14.5, 14.5]

    # # Cameras FOV heights (degrees)
    # fov_heights = [14.5, 14.5]

    # # If the camera FOV is fixed, it should have True at its index, and False if it can be moved to optimize
    # # the overlap
    # fixed_cameras = [False, True]

    # # Height range to optimize for (kilometers)
    # min_height = 80
    # max_height = 100

    # #################


    # How much each non-fixed camera can be moved on each axis (degrees)
    moving_ranges = [30]

    # Steps of movement to explore
    steps = 21



    ##########################################################################################################

    # Calculate heights in meters
    min_height *= 1000
    max_height *= 1000

    # Init stations data to SimStation objects
    station_list = initStationList(stations_geo, azim_fovs, elev_fovs, fov_widths, fov_heights)


    # Show current FOV overlap
    plotStationFovs(station_list, datetime2JD(datetime.datetime.now()), min_height, max_height)

    # Do an assessment for the whole range of given rights
    explorePointings(station_list, fixed_cameras, min_height, max_height, moving_ranges, steps)

    # # Do the anaysis for ranges of heights

    # # Height step in kilometers
    # height_step = 5

    # height_step *= 1000
    
    # for ht_min in range(min_height, max_height - height_step + 1, height_step):

    #     print(ht_min, ht_min + height_step)

    #     # Make a map of pointings and common volumes for all given steps in the moving range
    #     explorePointings(station_list, fixed_cameras, ht_min, ht_min + height_step, moving_ranges, steps)

