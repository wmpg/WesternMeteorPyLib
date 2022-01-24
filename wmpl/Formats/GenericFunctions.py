""" Function that adds common arguments for all input formats to the argument parser handle. """


import os

import numpy as np

from wmpl.Trajectory.Trajectory import Trajectory
from wmpl.Trajectory.GuralTrajectory import GuralTrajectory
from wmpl.Utils.TrajConversions import J2000_JD, jd2Date, equatorialCoordPrecession_vect, raDec2AltAz_vect



class MeteorObservation(object):
    """ Container for meteor observations. 
        
        The points in arrays are RA and Dec in J2000 epoch, in radians.

        Arguments:
            jdt_ref: [float] Reference Julian date when the relative time is t = 0s.
            station_id: [str] Station ID.
            latitude: [float] Latitude +N in radians.
            longitude: [float] Longitude +E in radians.
            height: [float] Elevation above sea level (MSL) in meters.
            fps: [float] Frames per second.

        Keyword arguments:
            ff_name: [str] Name of the originating FF file.

    """
    def __init__(self, jdt_ref, station_id, latitude, longitude, height, fps, ff_name=None):

        self.jdt_ref = jdt_ref
        self.station_id = station_id
        self.latitude = latitude
        self.longitude = longitude
        self.height = height
        self.fps = fps

        self.ff_name = ff_name

        self.frames = []
        self.time_data = []
        self.x_data = []
        self.y_data = []
        self.azim_data = []
        self.elev_data = []
        self.ra_data = []
        self.dec_data = []
        self.mag_data = []
        self.abs_mag_data = []



    def addPoint(self, frame_n, x, y, azim, elev, ra, dec, mag):
        """ Adds the measurement point to the meteor.

        Arguments:
            frame_n: [flaot] Frame number from the reference time.
            x: [float] X image coordinate.
            y: [float] X image coordinate.
            azim: [float] Azimuth, J2000 in degrees.
            elev: [float] Elevation angle, J2000 in degrees.
            ra: [float] Right ascension, J2000 in degrees.
            dec: [float] Declination, J2000 in degrees.
            mag: [float] Visual magnitude.

        """

        self.frames.append(frame_n)

        # Calculate the time in seconds w.r.t. to the reference JD
        point_time = float(frame_n)/self.fps

        self.time_data.append(point_time)

        self.x_data.append(x)
        self.y_data.append(y)

        # Angular coordinates converted to radians
        self.azim_data.append(np.radians(azim))
        self.elev_data.append(np.radians(elev))
        self.ra_data.append(np.radians(ra))
        self.dec_data.append(np.radians(dec))
        self.mag_data.append(mag)



    def finish(self):
        """ When the initialization is done, convert data lists to numpy arrays. """

        self.frames = np.array(self.frames)
        self.time_data = np.array(self.time_data)
        self.x_data = np.array(self.x_data)
        self.y_data = np.array(self.y_data)
        self.azim_data = np.array(self.azim_data)
        self.elev_data = np.array(self.elev_data)
        self.ra_data = np.array(self.ra_data)
        self.dec_data = np.array(self.dec_data)
        self.mag_data = np.array(self.mag_data)

        # Sort by frame
        temp_arr = np.c_[self.frames, self.time_data, self.x_data, self.y_data, self.azim_data, \
        self.elev_data, self.ra_data, self.dec_data, self.mag_data]
        temp_arr = temp_arr[np.argsort(temp_arr[:, 0])]
        self.frames, self.time_data, self.x_data, self.y_data, self.azim_data, self.elev_data, self.ra_data, \
            self.dec_data, self.mag_data = temp_arr.T




    def __repr__(self):

        out_str = ''

        out_str += 'Station ID = ' + str(self.station_id) + '\n'
        out_str += 'JD ref = {:f}'.format(self.jdt_ref) + '\n'
        out_str += 'DT ref = {:s}'.format(jd2Date(self.jdt_ref, \
            dt_obj=True).strftime("%Y/%m/%d-%H%M%S.%f")) + '\n'
        out_str += 'Lat = {:f}, Lon = {:f}, Ht = {:f} m'.format(np.degrees(self.latitude), 
            np.degrees(self.longitude), self.height) + '\n'
        out_str += 'FPS = {:f}'.format(self.fps) + '\n'

        out_str += 'Points:\n'
        out_str += 'Time, X, Y, azimuth, elevation, RA, Dec, Mag:\n'

        for point_time, x, y, azim, elev, ra, dec, mag in zip(self.time_data, self.x_data, self.y_data, \
            self.azim_data, self.elev_data, self.ra_data, self.dec_data, self.mag_data):

            if mag is None:
                mag = 0

            out_str += '{:.4f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:+.2f}, {:.2f}\n'.format(point_time,\
                x, y, np.degrees(azim), np.degrees(elev), np.degrees(ra), np.degrees(dec), mag)


        return out_str



def prepareObservations(meteor_list):
    """ Takes a list of MeteorObservation objects, normalizes all data points to the same reference Julian 
        date, precesses the observations from J2000 to the epoch of date. 
    
    Arguments:
        meteor_list: [list] List of MeteorObservation objects

    Return:
        (jdt_ref, meteor_list):
            - jdt_ref: [float] reference Julian date for which t = 0
            - meteor_list: [list] A list a MeteorObservations whose time is normalized to jdt_ref, and are
                precessed to the epoch of date

    """

    if meteor_list:

        # The reference meteor is the one with the first time of the first frame
        ref_ind = np.argmin([met.jdt_ref + met.time_data[0]/86400.0 for met in meteor_list])
        tsec_delta = meteor_list[ref_ind].time_data[0] 
        jdt_delta = tsec_delta/86400.0


        ### Normalize all times to the beginning of the first meteor

        # Apply the normalization to the reference meteor
        meteor_list[ref_ind].jdt_ref += jdt_delta
        meteor_list[ref_ind].time_data -= tsec_delta


        meteor_list_tcorr = []

        for i, meteor in enumerate(meteor_list):

            # Only correct non-reference meteors
            if i != ref_ind:

                # Calculate the difference between the reference and the current meteor
                jdt_diff = meteor.jdt_ref - meteor_list[ref_ind].jdt_ref
                tsec_diff = jdt_diff*86400.0

                # Normalize all meteor times to the same reference time
                meteor.jdt_ref -= jdt_diff
                meteor.time_data += tsec_diff

            meteor_list_tcorr.append(meteor)

        ######

        # The reference JD for all meteors is thus the reference JD of the first meteor
        jdt_ref = meteor_list_tcorr[ref_ind].jdt_ref


        ### Precess observations from J2000 to the epoch of date
        meteor_list_epoch_of_date = []
        for meteor in meteor_list_tcorr:

            jdt_ref_vect = np.zeros_like(meteor.ra_data) + jdt_ref

            # Precess from J2000 to the epoch of date
            ra_prec, dec_prec = equatorialCoordPrecession_vect(J2000_JD.days, jdt_ref_vect, meteor.ra_data, 
                meteor.dec_data)

            meteor.ra_data = ra_prec
            meteor.dec_data = dec_prec

            # Convert preccesed Ra, Dec to altitude and azimuth
            meteor.azim_data, meteor.elev_data = raDec2AltAz_vect(meteor.ra_data, meteor.dec_data, jdt_ref,
                meteor.latitude, meteor.longitude)

            meteor_list_epoch_of_date.append(meteor)


        ######


        return jdt_ref, meteor_list_epoch_of_date

    else:
        return None, None



def solveTrajectoryGeneric(jdt_ref, meteor_list, dir_path, solver='original', **kwargs):
    """ Feed the list of meteors in the trajectory solver and run it. 
    
    Arguments:
        jdt_ref: [float] Reference Julian date for all objects in meteor_list.
        meteor_list: [list] A list of MeteorObservation objects.
        dir_path: [str] Path to the data directory.

    Keyword arguments:
        solver: [str] Solver choice:
            - "original" is the Monte Carlo solver
            - "gural" is the Gural solver (through C++ bindings)
        **kwargs: Keyword arguments for the trajectory solver.

    """

    # Create name of output directory
    output_dir = os.path.join(dir_path, jd2Date(jdt_ref, dt_obj=True).strftime("%Y%m%d-%H%M%S.%f"))


    # Init the trajectory solver
    if solver == 'original':
        traj = Trajectory(jdt_ref, output_dir=output_dir, meastype=1, **kwargs)

    elif solver.lower().startswith('gural'):
        velmodel = solver.lower().strip('gural')
        if len(velmodel) == 1:
            velmodel = int(velmodel)
        else:
            velmodel = 0

        traj = GuralTrajectory(len(meteor_list), jdt_ref, velmodel=velmodel, meastype=1, verbose=1, 
            output_dir=output_dir)

    else:
        print('No such solver:', solver)
        return 


    # Add meteor observations to the solver
    for meteor in meteor_list:

        if solver == 'original':

            comment = ''
            if hasattr(meteor, "ff_name"):
                comment = meteor.ff_name

            traj.infillTrajectory(meteor.ra_data, meteor.dec_data, meteor.time_data, meteor.latitude, 
                meteor.longitude, meteor.height, station_id=meteor.station_id, \
                magnitudes=meteor.mag_data, comment=comment)

        elif solver.lower().startswith('gural'):

            # Extract velocity model is given
            try:
                velmodel = int(solver[-1])

            except: 
                # Default to the exponential model
                velmodel = 3

            traj.infillTrajectory(meteor.ra_data, meteor.dec_data, meteor.time_data, meteor.latitude, 
                meteor.longitude, meteor.height)


    # Solve the trajectory
    traj = traj.run()

    return traj



def addSolverOptions(arg_parser, skip_velpart=False):
    """ Adds common arguments for all input formats to the argument parser handle. """

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

    if not skip_velpart:
        arg_parser.add_argument('-p', '--velpart', metavar='VELOCITY_PART', \
            help='Fixed part from the beginning of the meteor on which the initial velocity estimation using the sliding fit will start. Default is 0.25 (25 percent), but for noisier data this might be bumped up to 0.5.', \
            type=float, default=0.25)

    arg_parser.add_argument('-d', '--disablemc', \
        help='Do not use the Monte Carlo solver, but only run the geometric solution.', action="store_true")

    arg_parser.add_argument('-e', '--notimefit', \
        help="Do not estimate timing and velocity together. The times are assumed to be fixed. A list of time offsets per station can be provided, e.g. \"CA001A:0.42,CA0005:-0.3\" for speciflying offsets of 0.42 and -0.3 seconds for stations CA001A and CA0005, respectively. Make sure there are no spaces between the arguments, although spaces are fine in the station name.", \
        type=str, nargs="?", default=True)
    
    arg_parser.add_argument('-r', '--mcruns', metavar="MC_RUNS", \
        help='Number of Monte Carlo runs.', type=int, default=100)

    arg_parser.add_argument('-u', '--uncertgeom', \
        help='Compute purely geometric uncertainties.', action="store_true")

    arg_parser.add_argument('-m', '--mcstd', metavar='MC_STDDEVS', \
        help='Standard deviations of noise to add to measurements during the Monte Carlo procedure. 1.0 by default.', \
        type=float, default=1.0)
    
    arg_parser.add_argument('-g', '--disablegravity', \
        help='Disable gravity compensation.', action="store_true")

    arg_parser.add_argument('-l', '--plotallspatial', \
        help='Plot a collection of plots showing the residuals vs. time, lenght and height.', \
        action="store_true")

    arg_parser.add_argument('-j', '--jacchia', \
        help='Show the Jacchia exponential deceleration fit on plots with the dynamics.', \
        action="store_true")

    arg_parser.add_argument('-i', '--imgformat', metavar='IMG_FORMAT', nargs=1, \
        help="Plot image format. 'png' by default, can be 'pdf', 'eps',... ", type=str, default='png')

    arg_parser.add_argument('-x', '--hideplots', \
        help="Don't show generated plots on the screen, just save them to disk.", action="store_true")


    return arg_parser