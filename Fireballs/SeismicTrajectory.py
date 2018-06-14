""" Determine the fireball trajectory from seismic data.

Modified method of Pujol et al. (2005).

"""

from __future__ import print_function, division, absolute_import

import sys

import numpy as np
import scipy.signal
import scipy.optimize
import pyswarm
import matplotlib.pyplot as plt


from Formats.CSSseismic import loadCSSseismicData
from Utils.TrajConversions import date2JD, raDec2ECI, geo2Cartesian, cartesian2Geo, raDec2AltAz, eci2RaDec, latLonAlt2ECEF, ecef2ENU, enu2ECEF, ecef2LatLonAlt
from Utils.Math import vectMag, vectNorm, rotateVector, meanAngle
from Utils.Plotting import Arrow3D, set3DEqualAxes
from Utils.PlotMap import GroundMap



def mergeChannels(seismic_data):
    """ Merge seismic data from the same channels which are fragmented into several chunks which start at a
        different time

    """

    merged_data = []
    merged_indices = []

    for i, entry1 in enumerate(seismic_data):

        # Skip the entry if it was already merged
        if i in merged_indices:
            continue

        # Unpack the loaded seismic data
        site1, w1, time_data1, waveform_data1 = entry1

        merged_sites = [site1]
        merged_wfdisc = [w1]
        merged_time = [time_data1]
        merged_waveform = [waveform_data1]


        for j, entry2 in enumerate(seismic_data):

            # Skip the same entries
            if i == j:
                continue

            # Skip the entry if it was already merged
            if j in merged_indices:
                continue

            # Unpack the seismic data
            site2, w2, time_data2, waveform_data2 = entry2


            # Check if the channels are the same
            if (w1.sta == w2.sta) and (w1.chan == w2.chan):

                merged_sites.append(site2)
                merged_wfdisc.append(w2)
                merged_time.append(time_data2)
                merged_waveform.append(waveform_data2)

                merged_indices.append(j)


        # Add all merged data to the list, but keep the first site and wfdisc info as the reference one
        merged_data.append([merged_sites, merged_wfdisc, merged_time, merged_waveform])


    return merged_data



def wrapRaDec(ra, dec):
    """ Wraps RA and Dec into their limits. """

    # Wrap RA into [0, 2pi) range
    ra = ra%(2*np.pi)

    # Wrap Dec into [-pi/2, pi/2]
    dec = (dec + np.pi/2)%np.pi - np.pi/2

    return ra, dec




def timeOfArrival(stat_coord, x0, y0, t0, v, azim, zangle, v_sound):
    """ Calculate the time of arrival at given coordinates in the local coordinate system for the given
        parameters of the fireball.

    Arguments:
        stat_coord: [3 element ndarray] Coordinates of the station in the local coordinate system.
        x0: [float] Intersection with the X axis in the local coordinate system (meters).
        y0: [float] Intersection with the Y axis in the local coordinate system (meters).
        t0: [float] Time when the trajectory intersected the reference XY plane (seconds), offset from 
            some reference time.
        v: [float] Velocity of the fireball (m/s).
        azim: [float] Fireball azimuth (+E of due S).
        zangle: [float] Zenith angle.
        v_sound: [float] Average speed of sound (m/s).

    Return:
        ti: [float] Balistic shock time of arrival to the given coordinates (seconds).

    """

    # Calculate the mach angle
    beta = np.arcsin(v_sound/v)

    # Trajectory vector
    u = np.array([np.cos(azim)*np.sin(zangle), np.sin(azim)*np.cos(zangle), -np.cos(zangle)])

    # Difference from the reference point on the trajectory and the station
    b = stat_coord - np.array([x0, y0, 0])


    # Calculate the distance along the trajectory
    dt = np.abs(np.dot(b, u))

    # Calculate the distance perpendicular to the trajectory
    dp = np.sqrt(vectMag(b)**2 - dt**2)

    # Calculate the time of arrival
    ti = t0 - dt/v + (dp*np.cos(beta))/v_sound


    return ti




def waveReleasePoint(stat_coord, x0, y0, t0, v, azim, zangle, v_sound):
    """ Calculate the point on the trajectory from which the balistic wave was released and heard by the given
        station.

    Arguments:
        stat_coord: [3 element ndarray] Coordinates of the station in the local coordinate system.
        x0: [float] Intersection with the X axis in the local coordinate system (meters).
        y0: [float] Intersection with the Y axis in the local coordinate system (meters).
        t0: [float] Time when the trajectory intersected the reference XY plane (seconds), offset from 
            some reference time.
        v: [float] Velocity of the fireball (m/s).
        azim: [float] Fireball azimuth (+E of due S).
        zangle: [float] Zenith angle.
        v_sound: [float] Average speed of sound (m/s).

    Return:
        traj_point: [3 element ndarray] Location of the release point in the local coordinate system.

    """

    # Calculate the mach angle
    beta = np.arcsin(v_sound/v)

    # Trajectory vector
    u = np.array([np.cos(azim)*np.sin(zangle), np.sin(azim)*np.cos(zangle), -np.cos(zangle)])

    # Difference from the reference point on the trajectory and the station
    b = stat_coord - np.array([x0, y0, 0])


    # Calculate the distance along the trajectory
    dt = np.abs(np.dot(b, u))

    # Calculate the distance perpendicular to the trajectory
    dp = np.sqrt(vectMag(b)**2 - dt**2)


    # Vector from (x0, y0) to the point of wave release
    r = -u*(dt + dp*np.tan(beta))

    # Position of the wave release in the local coordinate system
    traj_point = np.array([x0, y0, 0]) + r


    return traj_point




def timeResidualsAzimuth(params, stat_coord_list, arrival_times, v_sound, azim_off=None, v_fixed=False, \
        print_residuals=False):
    """ Cost function for seismic fireball trajectory optimization. The function uses 

    Arguments:
        params: [list] Estimated parameters: x0, t0, t0, v, azim, elev.
        stat_coord_list: [list of ndarrays] A list of station coordinates (x, y, z) in the reference coordinate system.
        arrival_times: [list] A list of arrival times of the sound wave to the seismic station (in seconds 
            from some reference time).
        v_sound: [float] Average speed of sound (m/s).

    Keyword arguments:
        azim_off: [float] Azimuth around which the given values are centred. If None (default), it is assumed 
            that the azimuth is calculated +E of due South.
        v_fixed: [bool] Use a fixed velocity of 20 km/s

    """

    # Unpack estimated parameters
    x0, y0, t0, v, azim, zangle = params

    # Convert the values from km to m
    x0 *= 1000
    y0 *= 1000
    v  *= 1000

    if v_fixed:

        # Keep the fireball velocity fixed
        v = 20000


    # If the azimuth offset is given, recalculate the proper values
    if azim_off is not None:
        azim += azim_off

    # Wrap azimuth and zenith angle to the allowed range
    azim = azim%(2*np.pi)
    zangle = zangle%(np.pi/2)

    cost_value = 0

    # Go through all arrival times
    for i, (t_obs, stat_coord) in enumerate(zip(arrival_times, stat_coord_list)):

        ### Calculate the difference between the observed and the prediced arrival times ###
        ######################################################################################################

        # Calculate the time of arrival
        ti = timeOfArrival(stat_coord, x0, y0, t0, v, azim, zangle, v_sound)

        # Smooth approximation of l1 (absolute value) loss
        z = (t_obs - ti)**2
        cost = 2*((1 + z)**0.5 - 1)

        cost_value += cost


        if print_residuals:
            print("{:>3d}, {:<.3f}".format(i, t_obs - ti))

        ######################################################################################################


    return cost_value



def latLon2Local(lat0, lon0, elev0, lat, lon, elev):
    """ Convert geographic coordinates into a local coordinate system where the reference coordinates will be
        the origin. The positive direction of the X axis points towards the south, the positive direction of
        the Y axis points towards the east and the positive direction of the Z axis points to the zenith at
        the reference coordinates.

    Arguments:
        lat0: [float] reference latitude +N (radians).
        lon0: [float] reference longtidue +E (radians).
        elev0: [float] reference elevation above sea level (meters).
        lat: [float] Latitude +N (radians).
        lon: [float] Longtidue +E (radians).
        elev: [float] Elevation above sea level (meters).


    Return:
        (x, y, z): [3 element ndarray] (x, y, z) local coordinates.
    """

    # Calculate the ECEF coordinates of the reference position
    x0, y0, z0 = latLonAlt2ECEF(lat0, lon0, elev0)
    ref_ecef = np.array([x0, y0, z0])

    # Convert the geo coordinates of the station into ECEF coordinates
    coord_ecef = latLonAlt2ECEF(lat, lon, elev)


    ### Convert the ECEF coordinates into to local coordinate system ###
    
    local_coord = coord_ecef - ref_ecef

    # Rotate the coordinates so the origin point is tangent to the Earth's surface
    local_coord = np.array(ecef2ENU(lat0, lon0, *local_coord))

    # Rotate the coordinate system so X points towards the south and Y towards the east
    local_coord = rotateVector(local_coord, np.array([0, 0, 1]), np.pi/2)

    ######


    return local_coord




def local2LatLon(lat0, lon0, elev0, local_coord):
    """ Convert local coordinates into geographic coordinates. See latLon2Local for more details.

    Arguments:
        lat0: [float] reference latitude +N (radians).
        lon0: [float] reference longtidue +E (radians).
        elev0: [float] reference elevation above sea level (meters).
        local_coord: [3 element ndarray] (x, y, z):
            - x: [float] Local X coordinate (meters).
            - y: [float] Local Y coordinate (meters).
            - z: [float] Local Z coordinate (meters).

    Return:
        (lat, lon, elev): [3 element ndarray] Geographic coordinates, angles in radians, elevation in meters.
    """


    # Calculate the ECEF coordinates of the reference position
    x0, y0, z0 = latLonAlt2ECEF(lat0, lon0, elev0)
    ref_ecef = np.array([x0, y0, z0])


    # Rotate the coordinate system back to ENU
    local_coord = rotateVector(local_coord, np.array([0, 0, 1]), -np.pi/2)

    # Convert the coordinates back to ECEF
    coord_ecef = np.array(enu2ECEF(lat0, lon0, *local_coord)) + ref_ecef

    # Convert ECEF coordinates back to geo coordinates
    lat, lon, elev = ecef2LatLonAlt(*coord_ecef)


    return lat, lon, elev




def convertStationCoordinates(station_list, ref_indx):
    """ Converts the coordinates of stations into the local coordinate system, the origin being one of the
        given stations.
    
    Arguments:
        station_list: [list] A list of stations and arrival times, each entry is a tuple of:
            (name, lat, lon, elev, arrival_time_jd), where latitude and longitude are in radians, the 
            elevation is in meters and Julian date of the arrival time.
        ref_indx: [int] Index of the reference station which will be in the origin of the local coordinate
            system.
    """


    # Select the location of the reference station
    _, lat0, lon0, elev0, _ = station_list[ref_indx]

    stat_coord_list = []

    # Calculate the local coordinates of stations at the given time
    for i, entry in enumerate(station_list):

        _, lat, lon, elev, _ = entry
        
        # Convert geographical to local coordinates
        stat_coord = latLon2Local(lat0, lon0, elev0, lat, lon, elev)

        ######

        stat_coord_list.append(stat_coord)


    return stat_coord_list



def estimateSeismicTrajectoryAzimuth(station_list, v_sound, p0=None, azim_range=None, elev_range=None, \
        v_fixed=False):
    """ Estimate the trajectory of a fireball from seismic/infrasound data by modelling the arrival times of
        the balistic shock at given stations.

    The method is implemented according to Pujol et al. (2005) and Ishihara et al. (2003).

    Arguments:
        station_list: [list] A list of stations and arrival times, each entry is a tuple of:
            (name, lat, lon, elev, arrival_time_jd), where latitude and longitude are in radians, the 
            elevation is in meters and Julian date of the arrival time.
        v_sound: [float] The average speed of sound in the atmosphere (m/s).

    Keyword arguments:
        p0: [6 element ndarray] Initial parameters for trajectory estimation:
            p0[0]: [float] p0, north-south offset of the trajectory intersection with the ground (in km, +S).
            p0[1]: [float] y0, east-west offset of the trajectory intersection with the ground (in km, +E).
            p0[2]: [float] Time when the trajectory was at point (p0, y0), reference to the reference time 
                (seconds).
            p0[3]: [float] Velocity of the fireball (km/s).
            p0[4]: [float] Initial azimuth (+E of due south) of the fireball (radians).
            p0[5]: [float] Initial zenith angle of the fireball (radians).
        azim_range: [list of floats] (min, max) azimuths for the search, azimuths should be +E of due North
            in degrees. If the range of azmiuths traverses the 0/360 discontinuity, please use negative
            values for azimuths > 180 deg!
        elev_range: [list of floats] (min, max) elevations for the search, in degrees. The elevation is 
            measured from the horizon up.
        v_fixed: [bool] If True, use a fixed fireball velocity of 20 km/s. False by default.

    """

    # Extract all Julian dates
    jd_list = [entry[4] for entry in station_list]

    # Calculate the arrival times as the time in seconds from the earliest JD
    jd_ref = min(jd_list)
    jd_list = np.array(jd_list)
    arrival_times = (jd_list - jd_ref)*86400.0

    # Get the index of the first arrival station
    first_arrival_indx = np.argwhere(jd_list == jd_ref)[0][0]

    # Convert station coordiantes to local coordinates, with the station of the first arrival being the
    # origin of the coordinate system
    stat_coord_list = convertStationCoordinates(station_list, first_arrival_indx)

        
    if p0 is None:

        # Initial parameters:
        p0 = np.zeros(6)

        # Initial parameter
        # Initial point (km)
        # X direction (south positive) from the reference station
        p0[0] = 0
        # Y direction (east positive) from the reference station
        p0[1] = 0

        # Set the time of the wave release to 1 minute (about 20km at 320 m/s)
        p0[2] = -60

        # Set fireball velocity (km/s)
        p0[3] = 20

        # Set initial direction
        # Azim
        p0[4] = np.radians(0)
        # Zangle
        p0[5] = np.radians(45)



    # Find the trajectory by minimization
    # The minimization will probably fail as t0 and the velocity of the fireball are dependant, but the
    # radiant should converge


    # Set azimuth bounds
    if azim_range is not None:
        azim_min, azim_max = azim_range

        # Check if the azimuths traverse more than 180 degrees, meaning they are traversing 0/360
        if abs(max(azim_range) - min(azim_range)) > 180:
            azim_min -= 360

        # Convert the azimuths to +E of due S
        azim_min = np.radians(180 - azim_min)
        azim_max = np.radians(180 - azim_max)

        # Center the search around the mean azimuth
        azim_avg = (azim_min + azim_max)/2

        # Calculate the upper and lower azimuth boundaries from the mean
        azim_down = azim_avg - azim_min
        azim_up = azim_avg - azim_max
        

    else:
        azim_avg = None

        # Use 0 to 4*pi range as the solution might be around 0/2*pi
        azim_up = 4*np.pi
        azim_down = 0



    # Set the elevation bounds
    if elev_range is not None:
        elev_max, elev_min = elev_range

        # Calculate zenith angles
        zangle_min = np.radians(90 - elev_min)
        zangle_max = np.radians(90 - elev_max)

    else:
        zangle_min, zangle_max = 0, np.pi/2



    # Set the bounds for every parameters
    bounds = [
        (-300, +300), # X0
        (-300, +300), # Y0
        (-200, +200), # t0
        (11, 30), # Velocity (km/s)
        (azim_down, azim_up), # Azimuth
        (zangle_min, zangle_max) # Zenith angle
        ]

    print('Bounds:', bounds)

    # Extract lower and upper bounds
    lower_bounds = [bound[0] for bound in bounds]
    upper_bounds = [bound[1] for bound in bounds]

    class MiniResults():
        def __init__(self):
            self.x = None

    # Run PSO several times and choose the best solution
    solutions = []
    
    for i in range(10):

        print('Running PSO, run', i)

        # Use PSO for minimization
        x, fopt = pyswarm.pso(timeResidualsAzimuth, lower_bounds, upper_bounds, args=(stat_coord_list, \
            arrival_times, v_sound, azim_avg, v_fixed), maxiter=2000, swarmsize=2000, phip=1.1, phig=1.0, \
            debug=False, omega=0.5)

        print('Run', i, 'best estimation', fopt)

        solutions.append([x, fopt])


    # Choose the solution with the smallest residuals
    fopt_array = np.array([fopt for x, fopt in solutions])
    best_indx = np.argmin(fopt_array)

    x, fopt = solutions[best_indx]


    res = MiniResults()
    res.x = x

    print("X:", res.x)
    print('Final function value:', fopt)

    # Extract estimated parameters
    x0, y0 = res.x[:2]
    t0 = res.x[2]
    v_est = res.x[3]
    azim, zangle = res.x[4:]

    # Recalculate the proper azimuth if the range was given
    if azim_range:
        azim += azim_avg

    # Wrap azimuth and elevation to the allowed range
    azim = azim%(2*np.pi)
    zangle = zangle%(np.pi/2)

    print('--------------------')
    print('RESULTS:')
    print('x0:', x0, 'km')
    print('y0:', y0, 'km')
    print('t0:', t0, 's')
    print('v:', v_est, 'km/s')
    print('Azim (+E of due N) : ', (180 - np.degrees(azim))%360)
    print('Elev (from horizon):', 90 - np.degrees(zangle))


    # Print the time residuals per every station
    timeResidualsAzimuth(res.x, stat_coord_list, arrival_times, v_sound, azim_off=azim_avg, v_fixed=v_fixed, 
        print_residuals=True)




    # Plot the stations and the estimated trajectory
    plotStationsAndTrajectory(station_list, [1000*x0, 1000*y0, t0, 1000*v_est, azim, zangle], v_sound, 'fireball')




def plotStationsAndTrajectory(station_list, params, v_sound, file_name):
    """ Plots the stations in the local coordinate system and the fitted trajectory. """



    # Unpack estimated parameters
    x0, y0, t0, v, azim, zangle = params


    # Extract all Julian dates
    jd_list = [entry[4] for entry in station_list]

    # Calculate the arrival times as the time in seconds from the earliest JD
    jd_ref = min(jd_list)
    ref_indx = np.argmin(jd_list)
    jd_list = np.array(jd_list)
    stat_obs_times_of_arrival = (jd_list - jd_ref)*86400.0


    # Convert station coordiantes to local coordinates, with the station of the first arrival being the
    # origin of the coordinate system
    stat_coord_list = convertStationCoordinates(station_list, ref_indx)

    # Extract station coordinates
    x_stat, y_stat, z_stat = np.array(stat_coord_list).T

    # Convert station coordinates to km
    x_stat /= 1000
    y_stat /= 1000
    z_stat /= 1000

    x0 /= 1000
    y0 /= 1000


    # Calculate the trajectory vector
    u = np.array([np.cos(azim)*np.sin(zangle), np.sin(azim)*np.cos(zangle), -np.cos(zangle)])



    # Calculate modelled times of arrival and points of wave releasefor every station
    stat_model_times_of_arrival = []
    wave_release_points = []

    for stat_coord in stat_coord_list:

        # Calculate time of arrival
        ti = timeOfArrival(stat_coord, 1000*x0, 1000*y0, t0, v, azim, zangle, v_sound)
        stat_model_times_of_arrival.append(ti)

        # Calculate point of wave release
        traj_point = waveReleasePoint(stat_coord, 1000*x0, 1000*y0, t0, v, azim, zangle, v_sound)/1000
        wave_release_points.append(traj_point)

    stat_model_times_of_arrival = np.array(stat_model_times_of_arrival)
    wave_release_points = np.array(wave_release_points)


    # Get the release points with the highest and the lowest release height
    high_point = np.argmax(wave_release_points[:, 2])
    low_point = np.argmin(wave_release_points[:, 2])

    print('x0:', x0, 'km')
    print('y0:', y0, 'km')
    print('t0:', t0, 's')
    print('v:', v, 'km/s')
    print('Azim (+E of due N) : ', (180 - np.degrees(azim))%360)
    print('Elev (from horizon):', 90 - np.degrees(zangle))
    print('Wave release:')
    print(wave_release_points[:, 2])
    print(' - top:', wave_release_points[high_point, 2], 'km')
    print(' - bottom:', wave_release_points[low_point, 2], 'km')
    print()
    print('Residuals:')
    print('{:>10s}: {:6s}'.format('Station', 'res (s)'))
    
    sqr_res_acc = 0

    # Print the residuals per station
    for stat_entry, toa_obs, toa_model in zip(station_list, stat_obs_times_of_arrival, \
        stat_model_times_of_arrival):

        station_name = stat_entry[0]

        print('{:>10s}: {:+06.2f}s'.format(station_name, toa_obs - toa_model))

        sqr_res_acc += (toa_obs - toa_model)**2


    print('RMS:', np.sqrt(sqr_res_acc/len(station_list)), 's')

    
    # Determine the maximum absolute time of arrival (either positive of negative)
    toa_abs_max = np.max([np.abs(np.min(stat_model_times_of_arrival)), np.max(stat_model_times_of_arrival)])


    # Calcualte absolute observed - calculated residuals
    toa_residuals = np.abs(stat_obs_times_of_arrival - stat_model_times_of_arrival)

    # Determine the maximum absolute residual
    toa_res_max = np.max(toa_residuals)



    ### PLOT 3D ###
    ##########################################################################################################

    # Setup 3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')


    # # Plot the stations except the reference
    # station_mask = np.ones(len(x), dtype=bool)
    # station_mask[ref_indx] = 0
    # ax.scatter(x[station_mask], y[station_mask], z[station_mask], c=stat_model_times_of_arrival[station_mask], \
    #     depthshade=0, cmap='viridis', vmin=0.00, vmax=toa_abs_max, edgecolor='k')

    # # Plot the reference station
    # ax.scatter(x[ref_indx], y[ref_indx], z[ref_indx], c='k', zorder=5)

    # Plot the stations (the colors are observed - calculated residuasl)
    stat_scat = ax.scatter(x_stat, y_stat, z_stat, c=toa_residuals, depthshade=0, cmap='inferno_r', \
        edgecolor='0.5', linewidths=1, vmin=0, vmax=toa_res_max)

    plt.colorbar(stat_scat, label='abs(O - C) (s)')


    # Plot the trajectory intersection with the ground
    ax.scatter(x0, y0, 0, c='g')


    # Plot the lowest and highest release points
    wrph_x, wrph_y, wrph_z = wave_release_points[high_point]
    wrpl_x, wrpl_y, wrpl_z = wave_release_points[low_point]
    ax.scatter(wrph_x, wrph_y, wrph_z)
    ax.scatter(wrpl_x, wrpl_y, wrpl_z)




    print('Trajectory vector:', u)

    # Get the limits of the plot
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    z_min, z_max = ax.get_zlim()

    # Get the maximum range of axes
    x_range = abs(x_max - x_min)
    y_range = abs(y_max - y_min)
    z_range = abs(z_max - z_min)

    traj_len = 1.5*max([x_range, y_range, z_range])


    # Calculate the beginning of the trajectory
    x_beg = x0 - traj_len*u[0]
    y_beg = y0 - traj_len*u[1]
    z_beg = -traj_len*u[2]

    # Plot the trajectory
    ax.plot([x0, x_beg], [y0, y_beg], [0, z_beg], c='k')

    # Plot wave release trajectory segment
    ax.plot([wrph_x, wrpl_x], [wrph_y, wrpl_y], [wrph_z, wrpl_z], color='red', linewidth=2)


    ### Plot the boom corridor ###
    img_dim = 100
    x_data = np.linspace(x_min, x_max, img_dim)
    y_data = np.linspace(y_min, y_max, img_dim)
    xx, yy = np.meshgrid(x_data, y_data)

    # Make an array of all plane coordinates
    plane_coordinates = np.c_[xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())]

    times_of_arrival = np.zeros_like(xx.ravel())
    
    # Calculate times of arrival for each point on the reference plane
    for i, plane_coords in enumerate(plane_coordinates):
        
        ti = timeOfArrival(1000*plane_coords, 1000*x0, 1000*y0, t0, v, azim, zangle, v_sound)

        times_of_arrival[i] = ti


    times_of_arrival = times_of_arrival.reshape(img_dim, img_dim)


    # Determine range and number of contour levels, so they are always centred around 0
    # toa_abs_max = np.max([np.abs(np.min(times_of_arrival)), np.max(times_of_arrival)])
    levels = np.linspace(0.00, toa_abs_max, 25)

    # Plot colorcoded times of arrival on the surface
    toa_conture = ax.contourf(xx, yy, times_of_arrival, levels, zdir='z', offset=np.min(z_stat), \
        cmap='viridis', alpha=1.0)

    # Add a color bar which maps values to colors
    fig.colorbar(toa_conture, label='Time of arrival (s)')


    ######



    # Constrain the plot to the initial bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


    # Set a constant aspect ratio
    #ax.set_aspect('equal', adjustable='datalim')

    # Set a constant and equal aspect ratio
    ax.set_aspect('equal', adjustable='box-forced')
    set3DEqualAxes(ax)


    ax.set_xlabel('X (+ south)')
    ax.set_ylabel('Y (+ east)')
    ax.set_zlabel('Z (+ zenith)')


    plt.savefig(file_name + '_3d.png', dpi=300)

    plt.show()




    ### PLOT THE MAP ###
    ##########################################################################################################

    # Extract coordinates of the reference station
    lat0, lon0, elev0 = station_list[ref_indx][1:4]

    # Calculate the coordinates of the trajectory intersection with the ground
    lat_i, lon_i, elev_i = local2LatLon(lat0, lon0, elev0, 1000*np.array([x0, y0, 0]))

    # Calculate the coordinate of the beginning of the trajectory
    lat_beg, lon_beg, elev_beg = local2LatLon(lat0, lon0, elev0, 1000*np.array([x_beg, y_beg, z_beg]))


    # Calculate coordinates of trajectory release points
    wrph_lat, wrph_lon, wrph_elev = local2LatLon(lat0, lon0, elev0, 1000*np.array([wrph_x, wrph_y, wrph_z]))
    wrpl_lat, wrpl_lon, wrpl_elev = local2LatLon(lat0, lon0, elev0, 1000*np.array([wrpl_x, wrpl_y, wrpl_z]))



    ### Calculate pointing vectors from stations to release points

    wr_points = []
    for wr_point, stat_point in zip(wave_release_points, np.c_[x_stat, y_stat, z_stat]):

        # Calculate vector pointing from station to release point
        wr_point = wr_point - stat_point

        wr_points.append(wr_point)

    
    # Normalize by the longest vector, which will be 1/10 of the trajectory length

    wr_len_max = max([vectMag(wr_point) for wr_point in wr_points])

    wr_vect_x = []
    wr_vect_y = []
    wr_vect_z = []

    for wr_point in wr_points:

        wr_x, wr_y, wr_z = (wr_point/wr_len_max)*traj_len/10

        wr_vect_x.append(wr_x)
        wr_vect_y.append(wr_y)
        wr_vect_z.append(wr_z)

    ###



    # Extract station coordinates
    stat_lat = []
    stat_lon = []
    stat_elev = []

    for entry in station_list:

        lat_t, lon_t, elev_t = entry[1:4]

        stat_lat.append(lat_t)
        stat_lon.append(lon_t)
        stat_elev.append(elev_t)


    stat_lat = np.array(stat_lat)
    stat_lon = np.array(stat_lon)
    stat_elev = np.array(stat_elev)

    # Init the ground map
    m = GroundMap(np.append(stat_lat, lat_i), np.append(stat_lon, lon_i), border_size=20, \
        color_scheme='light')



    # Convert contour local coordinated to geo coordinates
    lat_cont = []
    lon_cont = []
    
    for x_cont, y_cont in zip(xx.ravel(), yy.ravel()):
        
        lat_c, lon_c, _ = local2LatLon(lat0, lon0, elev0, 1000*np.array([x_cont, y_cont, 0]))

        lat_cont.append(lat_c)
        lon_cont.append(lon_c)


    lat_cont = np.array(lat_cont).reshape(img_dim, img_dim)
    lon_cont = np.array(lon_cont).reshape(img_dim, img_dim)


    # Plot the time of arrival contours
    toa_conture = m.m.contourf(np.degrees(lon_cont), np.degrees(lat_cont), times_of_arrival, levels, zorder=3, \
        latlon=True, cmap='viridis')


    # Add a color bar which maps values to colors
    m.m.colorbar(toa_conture, label='Time of arrival (s)')


    # Plot stations
    m.scatter(stat_lat, stat_lon, c=stat_model_times_of_arrival, s=20, marker='o', edgecolor='0.5', \
        linewidths=1, cmap='viridis', vmin=0.00, vmax=toa_abs_max)


    # Plot intersection with the ground
    m.scatter(lat_i, lon_i, s=10, marker='x', c='g')


    # Plot the trajectory
    m.plot([lat_beg, lat_i], [lon_beg, lon_i], c='g')


    # Plot the wave release segment
    m.plot([wrph_lat, wrpl_lat], [wrph_lon, wrpl_lon], c='red', linewidth=2)


    # Plot wave release directions
    for i in range(len(x_stat)):
        
        wr_lat_i, wr_lon_i, wr_elev_i = local2LatLon(lat0, lon0, elev0, 1000*np.array([x_stat[i] + wr_vect_x[i], y_stat[i] + wr_vect_y[i], z_stat[i] + wr_vect_z[i]]))
        #wr_lat_i, wr_lon_i, wr_elev_i = local2LatLon(lat0, lon0, elev0, 1000*np.array([wrp_x[i], wrp_y[i], wrp_z[i]]))

        m.plot([stat_lat[i], wr_lat_i], [stat_lon[i], wr_lon_i], c='k', linewidth=1.0)


    plt.tight_layout()


    plt.savefig(file_name + '_map.png', dpi=300)

    plt.show()



    ##########################################################################################################




if __name__ == "__main__":

    import matplotlib.pyplot as plt


    # lat0 = 15
    # lon0 = 35
    # elev0 = 250
    # local_coords = latLon2Local(lat0, lon0, elev0, np.radians(15.1), np.radians(35.02), 300)
    # lat, lon, elev = local2LatLon(lat0, lon0, elev0, local_coords)

    # print(np.degrees(lat))
    # print(np.degrees(lon))
    # print(elev)


    # sys.exit()


    # Average speed of sound in the atmosphere
    v_sound = 320 # m/s


    ### TEST!!!! ###
    ##########################################################################################################

    # Wayne event
    station_list = [
        ['G58A', np.radians(45.149200), np.radians(-74.054001), 173, date2JD(2013, 11, 27, 00, 50, 27, 150)],
        ['F58A', np.radians(45.866300), np.radians(-73.814500), 239, date2JD(2013, 11, 27, 00, 52, 23, 900)],
        ['E58A', np.radians(46.372100), np.radians(-73.277100), 764, date2JD(2013, 11, 27, 00, 55, 25, 880)],
        ['H59A', np.radians(44.645500), np.radians(-73.690498), 355, date2JD(2013, 11, 27, 00, 52, 48, 800)],
        ['H58A', np.radians(44.417599), np.radians(-74.179802), 537, date2JD(2013, 11, 27, 00, 53, 19, 320)]
    ]
    plotStationsAndTrajectory(station_list, [-10.6255131247*1000,  0.97761117287*1000, -22.39726861, 29.72664835*1000, np.radians(180 - 180.485602168), np.radians(90 - 42.5945238073)], v_sound, 'wayne')
    #estimateSeismicTrajectoryAzimuth(station_list, 320, azim_range=[100, 220], elev_range=[5, 35])


    # # Bolivia 2007
    # station_list = [
    #     ['BBOD ', np.radians(-16.64), np.radians(-68.60), 4235, date2JD(2007, 9, 15, 16, 42, 22)],
    #     ['I08BO', np.radians(-16.21), np.radians(-68.45), 4131, date2JD(2007, 9, 15, 16, 44, 10)],
    #     ['BBOE ', np.radians(-16.81), np.radians(-67.98), 4325, date2JD(2007, 9, 15, 16, 45, 21)],
    #     ['BBOK ', np.radians(-16.58), np.radians(-67.87), 4638, date2JD(2007, 9, 15, 16, 46, 05)],
    #     ['I41PY', np.radians(-26.34), np.radians(-57.31),  164, date2JD(2007, 9, 15, 18,  8, 51)]
    # ]
    # estimateSeismicTrajectoryAzimuth(station_list, 320, elev_range=[5, 80])



    # # Pujol et al. 2005 data
    # arkansas_ref_jd = date2JD(2003, 11, 04, 03, 0, 0)
    # station_list = [
    #     ['BVAR', np.radians(35.443), np.radians(-90.677), 200, arkansas_ref_jd + 0.00/86400],
    #     ['HBAR', np.radians(35.555), np.radians(-90.657), 200, arkansas_ref_jd + 3.32/86400],
    #     ['QUAR', np.radians(35.644), np.radians(-90.649), 200, arkansas_ref_jd + 10.17/86400],
    #     ['TWAR', np.radians(35.361), np.radians(-90.560), 200, arkansas_ref_jd + 26.99/86400],
    #     ['JHAR', np.radians(35.606), np.radians(-90.524), 200, arkansas_ref_jd + 29.56/86400],
    #     ['TMAR', np.radians(35.695), np.radians(-90.489), 200, arkansas_ref_jd + 43.27/86400],
    #     ['BLAR', np.radians(35.369), np.radians(-90.449), 200, arkansas_ref_jd + 45.59/86400],
    #     ['NHAR', np.radians(35.786), np.radians(-90.544), 200, arkansas_ref_jd + 45.91/86400],
    #     ['NFAR', np.radians(35.448), np.radians(-90.393), 200, arkansas_ref_jd + 51.19/86400],
    #     ['TYAR', np.radians(35.509), np.radians(-90.292), 200, arkansas_ref_jd + 68.83/86400],
    #     ['LPAR', np.radians(35.602), np.radians(-90.300), 200, arkansas_ref_jd + 69.86/86400],
    #     ['RVAR', np.radians(35.690), np.radians(-90.286), 200, arkansas_ref_jd + 78.47/86400],
    #     ['CPAR', np.radians(35.556), np.radians(-90.236), 200, arkansas_ref_jd + 79.96/86400],
    #     ['BOAR', np.radians(35.823), np.radians(-90.287), 200, arkansas_ref_jd + 93.97/86400],
    #     ['HTAR', np.radians(35.655), np.radians(-90.185), 200, arkansas_ref_jd + 94.52/86400],
    #     ['MSAR', np.radians(35.784), np.radians(-90.147), 200, arkansas_ref_jd + 113.34/86400],
    #     ['LVAR', np.radians(35.915), np.radians(-90.222), 200, arkansas_ref_jd + 119.65/86400],
    #     ['HOVM', np.radians(36.044), np.radians(-90.067), 200, arkansas_ref_jd + 168.05/86400]
    # ]

    # # Initial parameters (Arkasnas fireball)
    # p0 = np.zeros(6)
    # # Initial point (km)
    # # X direction (south positive) from the reference station
    # p0[0] = -18.1
    # # Y direction (east positive) from the reference station
    # p0[1] = -85.7

    # # Set the time of the wave release to 1 minute (about 20km at 320 m/s)
    # p0[2] = -134

    # # Set fireball velocity (km/s)
    # p0[3] = 20

    # # Set initial direction
    # # Azim (+E of due south)
    # p0[4] = np.radians(-106.5)
    # # Zangle
    # p0[5] = np.radians(37.6)

    # # Arkansas
    # plotStationsAndTrajectory(station_list, [-9.05406157e+00*1000,  -8.22575287e+01*1000,  -1.66322331e+02,   1.84898542e+01*1000, np.radians(180 - 272.303002438),   7.13692280e-01], v_sound, 'arkansas')
    # #estimateSeismicTrajectoryAzimuth(station_list, 320, p0=p0, azim_range=[230, 310], elev_range=[30, 55])


    # # Ishikara 2003 data
    # station_list = [
    #     #["MYK  ", np.radians(39.590), np.radians(141.98), 120, date2JD(1998, 3, 30, 3, 23,  8, 270, UT_corr=9)],
    #     ["KGJ* ", np.radians(39.387), np.radians(141.57), 375, date2JD(1998, 3, 30, 3, 23, 42, 740, UT_corr=9)],
    #     ["JOM* ", np.radians(39.473), np.radians(141.29), 210, date2JD(1998, 3, 30, 3, 24,  7, 830, UT_corr=9)],
    #     ["MNS* ", np.radians(39.355), np.radians(141.20),  65, date2JD(1998, 3, 30, 3, 24, 13, 780, UT_corr=9)],
    #     ["NTM* ", np.radians(39.632), np.radians(141.30), 311, date2JD(1998, 3, 30, 3, 24, 22, 060, UT_corr=9)],
    #     ["THR* ", np.radians(39.118), np.radians(141.26), 165, date2JD(1998, 3, 30, 3, 24, 28, 930, UT_corr=9)],
    #     ["NAM* ", np.radians(39.466), np.radians(141.00), 245, date2JD(1998, 3, 30, 3, 24, 37, 340, UT_corr=9)],
    #     ["HAN* ", np.radians(39.374), np.radians(140.94), 300, date2JD(1998, 3, 30, 3, 24, 38, 350, UT_corr=9)],
    #     ["GTO* ", np.radians(39.237), np.radians(140.91), 610, date2JD(1998, 3, 30, 3, 24, 41, 290, UT_corr=9)],
    #     ["YHB* ", np.radians(39.618), np.radians(141.08), 300, date2JD(1998, 3, 30, 3, 24, 43, 850, UT_corr=9)],
    #     ["SAW* ", np.radians(39.403), np.radians(140.77), 280, date2JD(1998, 3, 30, 3, 24, 57, 550, UT_corr=9)],
    #     ["JMK* ", np.radians(38.952), np.radians(141.22),  70, date2JD(1998, 3, 30, 3, 24, 59, 120, UT_corr=9)],
    #     ["SWU* ", np.radians(39.486), np.radians(140.79), 445, date2JD(1998, 3, 30, 3, 25,  1, 140, UT_corr=9)],
    #     ["HRQ* ", np.radians(38.984), np.radians(141.04), 123, date2JD(1998, 3, 30, 3, 25,  1, 500, UT_corr=9)],
    #     ["OSK* ", np.radians(39.616), np.radians(140.90), 270, date2JD(1998, 3, 30, 3, 25,  3, 980, UT_corr=9)],
    #     ["HMK* ", np.radians(39.848), np.radians(141.24), 650, date2JD(1998, 3, 30, 3, 25,  6, 760, UT_corr=9)],
    #     ["KT44*", np.radians(39.086), np.radians(140.72), 400, date2JD(1998, 3, 30, 3, 25,  7, 380, UT_corr=9)],
    #     ["HRN* ", np.radians(39.256), np.radians(140.63), 170, date2JD(1998, 3, 30, 3, 25,  8, 640, UT_corr=9)],
    #     ["KT43*", np.radians(39.130), np.radians(140.66), 280, date2JD(1998, 3, 30, 3, 25,  9, 760, UT_corr=9)],
    #     ["JRG* ", np.radians(39.396), np.radians(140.63), 200, date2JD(1998, 3, 30, 3, 25, 12,  90, UT_corr=9)],
    #     ["KT48*", np.radians(39.061), np.radians(140.59), 235, date2JD(1998, 3, 30, 3, 25, 20, 420, UT_corr=9)],
    #     ["GNY* ", np.radians(38.857), np.radians(140.72), 440, date2JD(1998, 3, 30, 3, 25, 39, 110, UT_corr=9)] 
    # ]

    # # Initial parameter (Japanese fireball)
    # p0 = np.zeros(6)
    # # Initial point (km)
    # # X direction (south positive) from the reference station
    # p0[0] = -93
    # # Y direction (east positive) from the reference station
    # p0[1] = 207

    # # Set the time of the wave release to 1 minute (about 20km at 320 m/s)
    # p0[2] = -60

    # # Set fireball velocity (km/s)
    # p0[3] = 20

    # # Set initial direction
    # # Azim (+E of due south)
    # p0[4] = np.radians(135)
    # # Zangle
    # p0[5] = np.radians(90 - 18.5)

    # # Japanese
    # # plotStationsAndTrajectory(station_list, -11466.2958037, 175318.786352, 5.00539821,    0.28304974, 'miyako')
    # estimateSeismicTrajectoryAzimuth(station_list, 320, p0=p0, azim_range=[30, 90], elev_range=[10, 40])




    # # Moravka
    # moravka_ref_jd = date2JD(2000, 5, 6, 11, 51, 50)
    # station_list = [
    #     ['CSM ', np.radians(49.8004), np.radians(18.5608),  278, moravka_ref_jd +  90.10/86400],
    #     ['RAJ ', np.radians(49.8514), np.radians(18.5817),  272, moravka_ref_jd +  96.40/86400],
    #     ['LUT ', np.radians(49.8832), np.radians(18.4150),  217, moravka_ref_jd + 100.90/86400],
    #     ['PRS ', np.radians(49.9143), np.radians(18.5528),  205, moravka_ref_jd + 102.80/86400],
    #     ['BMZ ', np.radians(49.8344), np.radians(18.1411),  250, moravka_ref_jd + 129.10/86400],
    #     ['MAJ ', np.radians(49.8237), np.radians(18.4713), -365, moravka_ref_jd +  92.95/86400],
    #     ['CSA ', np.radians(49.8531), np.radians(18.4925), -497, moravka_ref_jd +  95.26/86400],
    #     ['KVE ', np.radians(49.8003), np.radians(18.5007), -141, moravka_ref_jd +  90.40/86400]
    # ]

    
    # # Initial parameters (Moravka)
    # p0 = np.zeros(6)
    # # Initial point (km)
    # # X direction (south positive) from the reference station
    # p0[0] = 100
    # # Y direction (east positive) from the reference station
    # p0[1] = 0

    # # Set the time of the wave release to 1 minute (about 20km at 320 m/s)
    # p0[2] = -60

    # # Set fireball velocity (km/s)
    # p0[3] = 20

    # # Set initial direction
    # # Azim (+E of due south)
    # p0[4] = np.radians(0)
    # # Zangle
    # p0[5] = np.radians(90 - 20)

    # # Moravka
    # plotStationsAndTrajectory(station_list, [1000*7.10714884e+01,  -1000*2.26290796e+00,  -7.02326093e+01,   1000*1.40304477e+01, 6.64257699e-02,   1.24330021e+00], v_sound, 'moravka')
    # # estimateSeismicTrajectoryAzimuth(station_list, 320, p0=p0, azim_range=[140, 220], elev_range=[15, 40], \
    # #     v_fixed=False)




    # # TEST STATIONS
    # arkansas_ref_jd = date2JD(2003, 11, 04, 03, 0, 0)
    # station_list = [
    #     ['1', np.radians(35.443), np.radians(-90.677), 200, arkansas_ref_jd],
    #     ['2', np.radians(35.443), np.radians(-90.677), 300, arkansas_ref_jd],
    #     ['3', np.radians(35.443), np.radians(-90.577), 200, arkansas_ref_jd]
    # ]

    # # TEST convert station coordinates
    # stat_coord_list = convertStationCoordinates(station_list, 0)

    # print(stat_coord_list)


    sys.exit()
    ##########################################################################################################





    ### INPUTS ###
    ##########################################################################################################


    # DATA PATHS
    dir_paths = [
        "/local4/infrasound/Infrasound/Fireball/15-Sep-2007/seismic/UBh2"
        #"/local4/infrasound/Infrasound/Fireball/15-Sep-2011-SW_USA/is56"
        #"/local4/infrasound/Infrasound/Fireball/15-Sep-2011-SW_USA/is57"
        #"/local4/infrasound/Infrasound/Fireball/15-Sep-2007/seismic/LPAZ"
    ]

    site_files = [
        "UBh2.site"
        #"is56.site"
        #"is57.site"
        #"impact.site"
        ]

    wfdisc_files = [
        "UBh2.wfdisc"
        #"is56.wfdisc"
        #"is57.wfdisc"
        #"impact.wfdisc"
        ]

    # Average speed of sound in the atmosphere
    v_sound = 320 # m/s


    ##########################################################################################################

    seismic_data = []

    # Load seismic data from given files
    for dir_path, site_file, wfdisc_file in zip(dir_paths, site_files, wfdisc_files):

        # Load the seismic data from individual file
        file_data = loadCSSseismicData(dir_path, site_file, wfdisc_file)

        # Add all entries to the global list
        for entry in file_data:
            seismic_data.append(entry)


    # Merge all data from the same channels
    merged_data = mergeChannels(seismic_data)

    # Determine the earliest time from all beginning times
    ref_time = min([w.begin_time for _, merged_w, _, _ in merged_data for w in merged_w])

    # Setup the plotting
    f, axes = plt.subplots(nrows=len(merged_data), ncols=1, sharex=True)

    # Go through data from all channels
    for i, entry in enumerate(merged_data):

        # Select the current axis for plotting
        ax = axes[i]

        # Unpack the channels
        merged_sites, merged_wfdisc, merged_time, merged_waveform = entry

        # Go through all individual measurements
        for site, w, time_data, waveform_data in zip(merged_sites, merged_wfdisc, merged_time, merged_waveform):


            # Calculate the difference from the reference time
            t_diff = (w.begin_time - ref_time).total_seconds()

            # Offset the time data to be in accordance with the reference time
            time_data += t_diff

            # Plot the seismic data
            ax.plot(time_data/60, waveform_data, zorder=3, linewidth=0.5)

        # Add the station label
        ax.text(ax.get_xlim()[0], ax.get_ylim()[1], w.sta + " " + w.chan, va='top')

        ax.grid(color='0.9')


    plt.xlabel('Time (min)')

    plt.subplots_adjust(hspace=0)
    plt.show()

