""" Solve the trajectory from observations in ECSV files. """

from __future__ import print_function, division, absolute_import

import os
import sys
import glob
import datetime

import numpy as np

from wmpl.Formats.GenericFunctions import addSolverOptions, solveTrajectoryGeneric, MeteorObservation, \
    prepareObservations, writeMiligInputFileMeteorObservation
from wmpl.Utils.Math import vectNorm, vectMag, angleBetweenSphericalCoords
from wmpl.Utils.Pickling import loadPickle
from wmpl.Utils.TrajConversions import J2000_JD, datetime2JD, altAz2RADec_vect, \
    equatorialCoordPrecession_vect, jd2Date



# Use a fixed FPS (not really important in the grand scheme of things, it's just for compatibility with
#   used functions)
FPS = 15


def ecsvFragments(ecsv_paths):
    """ Find the additional fragments described in ECSV files, whose columns have a numeric suffix (e.g. ra1,
        azimuth1) as in Appendix 4 of the GFE standard.

    Arguments:
        ecsv_paths: [list] List of paths to ECSV files.

    Return:
        [list] Sorted numbers of the additional fragments (1, 2, ...), empty if there are none.
    """

    fragments = set()
    for ecsv_file in ecsv_paths:

        with open(ecsv_file) as f:
            header = next((line for line in f if not line.startswith('#')), '')

        for name in header.split(','):
            name = name.strip()
            if name.startswith('azimuth') and name[len('azimuth'):].isdigit() and int(name[len('azimuth'):]):
                fragments.add(int(name[len('azimuth'):]))

    return sorted(fragments)


def loadECSVs(ecsv_paths, no_prepare=False, fragment=0):
    """ Load meteor observations from ECSV files. 
    
    Arguments:
        ecsv_paths: [list] List of paths to ECSV files.

    Keyword arguments:
        no_prepare: [bool] If True, only load the observations, do not prepare them for the solver.
        fragment: [int] Fragment to load. 0 (default) is the main fragment, without the points flagged as not
            to be used in its trajectory (trajectory_use = False). An additional fragment k is read from the
            columns with the suffix k (e.g. azimuth1), as in Appendix 4 of the GFE standard, which makes the
            suffix 0 optional for the main fragment; files without the columns of the fragment are skipped.
    
    """

    # Init meteor objects
    jdt_ref = None
    meteor_list = []
    for ecsv_file in ecsv_paths:

        
        station_lat = None
        station_lon = None
        station_ele = None
        station_id = None
        image_file = ''

        # Load the station information from the ECSV file
        with open(ecsv_file) as f:

            for line in f:
                if line.startswith("#"):
                    line = line.replace('\n', '').replace('\r', '').replace('{', '').replace('}', '')
                    line = line.split(':')
                    if len(line) > 1:

                        if "obs_latitude" in line[0]:
                            station_lat = float(line[1])

                        if "obs_longitude" in line[0]:
                            station_lon = float(line[1])

                        if "obs_elevation" in line[0]:
                            station_ele = float(line[1])

                        if 'image_file' in line[0]:
                            image_file = line[1].strip().strip("'")

                        if "camera_id" in line[0]:
                            station_id = line[1].strip().strip("'")

                        if (station_id is None) and ("dfn_camera_codename" in line[0]):
                            station_id = line[1].strip().strip("'")
                            

            if (station_lat is None) or (station_lon is None) or (station_ele is None) \
                or (station_id is None):

                print("Station info could not be read from file:", ecsv_file, ", skipping...")
                continue


            # Load meteor measurements
            delimiter = ','
            data = np.loadtxt(ecsv_file, comments='#', delimiter=delimiter, dtype=str)

            # Determine the column indices from the header
            header = [name.strip() for name in data[0].tolist()]

            # Suffix of the columns of the fragment, optional for the main one (e.g. azimuth or azimuth0). Skip
            #   the files which do not describe it
            suffix = next((sfx for sfx in ([str(fragment)] if fragment else ['', '0']) 
                if ('azimuth' + sfx) in header), None)
            if suffix is None:
                continue

            dt_indx = header.index('datetime' + suffix)
            azim_indx = header.index('azimuth' + suffix)
            alt_indx = header.index('altitude' + suffix)
            x_indx = header.index('x_image' + suffix)
            y_indx = header.index('y_image' + suffix)

            # Only the main fragment has photometry
            if ('mag_data' in header) and (not fragment):
                mag_indx = header.index('mag_data')
            else:
                mag_indx = None


            # Skip the header
            data = data[1:]

            # Skip the rows without this fragment, e.g. for the main fragment (fragment 0) the frames on which
            #   only other fragments were measured (their columns have a numeric suffix, e.g. azimuth1)
            data = data[np.char.strip(data[:, azim_indx]) != '']

            # Leave out the points of the main fragment not to be used in its trajectory
            if (not fragment) and ('trajectory_use' in header):
                data = data[np.char.lower(np.char.strip(data[:, header.index('trajectory_use')])) != 'false']

            if len(data) == 0:
                print("The station {:s} has no points of fragment {:d}, skipping: {:s}".format(station_id, 
                    fragment, ecsv_file))
                continue

            # Unpack data
            dt_data, azim_data, alt_data, x_data, y_data = data[:, dt_indx], data[:, azim_indx], \
                data[:, alt_indx], data[:, x_indx], data[:, y_indx]

            azim_data = azim_data.astype(np.float64)
            alt_data = alt_data.astype(np.float64)
            x_data = x_data.astype(np.float64)
            y_data = y_data.astype(np.float64)

            # Get magnitude data, if any
            if mag_indx is not None:
                mag_data = data[:, mag_indx].astype(np.float64)
            else:
                mag_data = np.zeros_like(azim_data) + 10.0


            # Convert time to JD
            jd_data = []
            for date in dt_data:
                dt = datetime.datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%f")
                jd_data.append(datetime2JD(dt))


            jd_data = np.array(jd_data)

            # Take the first time as reference time
            jdt_ref = jd_data[0]


            # Compute relative time
            time_data = (jd_data - jdt_ref)*86400


            # Compute RA/Dec
            ra_data, dec_data = altAz2RADec_vect(np.radians(azim_data), np.radians(alt_data), jd_data, \
                np.radians(station_lat), np.radians(station_lon))

            # Precess to J2000
            ra_data, dec_data = equatorialCoordPrecession_vect(jdt_ref, J2000_JD.days, ra_data, dec_data)

            comment = '{"ff_name":' + f'"{image_file}"' +'}'


            # Init the meteor object
            meteor = MeteorObservation(jdt_ref, station_id, np.radians(station_lat), \
                np.radians(station_lon), station_ele, FPS, ff_name=comment)

            # Add data to meteor object
            for t_rel, x_centroid, y_centroid, ra, dec, azim, alt, mag in zip(time_data, x_data, y_data, \
                np.degrees(ra_data), np.degrees(dec_data), azim_data, alt_data, mag_data):

                meteor.addPoint(t_rel*FPS, x_centroid, y_centroid, azim, alt, ra, dec, mag)

            meteor.finish()

            # Keep the file the observation comes from, to match the fragments in it to the main one
            meteor.ecsv_file = ecsv_file


            # Check that the observation has a minimum number of points
            if len(meteor.time_data) < 4:
                print("The station {:s} has too few points (<4), skipping: {:s}".format(station_id, ecsv_file))
                continue


            meteor_list.append(meteor)


    if no_prepare:
        return jdt_ref, meteor_list

    else:
        
        # Normalize all observations to the same JD and precess from J2000 to the epoch of date
        return prepareObservations(meteor_list)



def solverStationIDs(meteors):
    """ Station IDs the trajectory solver gives the observations added in this order: a station already added
        gets the suffix _2, _3, ... (see Trajectory.infillTrajectory()). """

    station_ids = []
    counts = {}
    for meteor in meteors:

        station_id = str(meteor.station_id)
        counts[station_id] = counts.get(station_id, 0) + 1
        station_ids.append(station_id if (counts[station_id] == 1) else "{:s}_{:d}".format(station_id, 
            counts[station_id]))

    return station_ids


def appliedTimeOffsets(traj, input_meteors):
    """ Time offset applied to every observation of a solved trajectory, as the shift of its times from the
        input ones. It includes the offsets fixed in the input, and can differ by about a millisecond from the
        reported timing offsets, which do not account for every iteration of the timing estimation.

    Arguments:
        traj: [Trajectory] Solved trajectory.
        input_meteors: [list] MeteorObservation objects it was solved from, from loadECSVs(), in the order they
            were given to the solver.

    Return:
        [dict] Time offset (s) for the file (ecsv_file) of every input observation matched by a solved one: 
            the station ID the solver gave it (see solverStationIDs()), the number of points and time steps.
    """

    solved = {str(obs.station_id): obs for obs in traj.observations}

    offsets = {}
    for station_id, meteor in zip(solverStationIDs(input_meteors), input_meteors):

        obs = solved.get(station_id)
        input_jd = meteor.jdt_ref + np.array(meteor.time_data)/86400

        if (obs is not None) and (len(input_jd) == len(obs.JD_data)) \
            and np.allclose((input_jd - input_jd[0])*86400, (obs.JD_data - obs.JD_data[0])*86400, atol=1e-3):

            offsets[meteor.ecsv_file] = np.mean(obs.JD_data - input_jd)*86400

    return offsets


def originalPicksTrajectory(traj):
    """ The solution with the original picks of a trajectory solved with the Monte Carlo runs, for which the 
        solver returns the best of the runs instead. It is loaded from the pickle saved with the results.

    Arguments:
        traj: [Trajectory] Solved trajectory, from solveTrajectoryGeneric().

    Return:
        [Trajectory] The solution with the original picks, with the uncertainties of the Monte Carlo runs. The
            given trajectory if it was solved without them, or if its results were not saved.
    """

    if (traj.uncertainties is None) or (not traj.save_results):
        return traj

    traj_path = os.path.join(traj.output_dir, traj.file_name + '_trajectory.pickle')
    if not os.path.isfile(traj_path):
        return traj

    return loadPickle(*os.path.split(traj_path))


def solveFragmentTrajectories(traj, ecsv_paths, reuse_timing=True):
    """ Solve the trajectories of the additional fragments described in ECSV files (see ecsvFragments()),
        after the main one and with the same solver options. Each one is saved in a fragment_k folder of the
        output directory of the main trajectory.

    Arguments:
        traj: [Trajectory] Solved trajectory of the main fragment, from solveTrajectoryGeneric().
        ecsv_paths: [list] List of paths to ECSV files.

    Keyword arguments:
        reuse_timing: [bool] Fix the time offsets of the stations to the ones of the main trajectory (default),
            those of its solution with the original picks, instead of estimating them again from the fewer 
            points of every fragment.

    Return:
        [list] (fragment, trajectory) of every fragment solved, with the solution with the original picks 
            (see originalPicksTrajectory()).
    """

    if not hasattr(traj, 'solver_kwargs'):
        print("The options the main trajectory was solved with are unknown, so the fragments cannot be solved.")
        return []

    kwargs = dict(traj.solver_kwargs)
    solver = kwargs.pop('solver')

    if solver != 'original':
        print("The trajectories of additional fragments can only be solved with the original solver.")
        return []

    _, input_meteors = loadECSVs(ecsv_paths, no_prepare=True)
    main_offsets = appliedTimeOffsets(originalPicksTrajectory(traj), input_meteors)

    fragment_trajs = []
    for fragment in ecsvFragments(ecsv_paths):

        print()
        print("Solving the trajectory of fragment {:d}...".format(fragment))

        _, meteor_list = loadECSVs(ecsv_paths, no_prepare=True, fragment=fragment)
        if len(meteor_list) < 2:
            print("Fragment {:d} has enough points (4 or more) from only {:d} station(s), but 2 are needed, "
                "skipping.".format(fragment, len(meteor_list)))
            continue

        jdt_ref, meteor_list = prepareObservations(meteor_list)

        frag_kwargs = dict(kwargs)
        if reuse_timing:

            # Offsets of the main trajectory for the same files, under the station IDs the solver will use
            station_ids = solverStationIDs(meteor_list)
            missing = [station_id for station_id, meteor in zip(station_ids, meteor_list) 
                if meteor.ecsv_file not in main_offsets]
            if missing:
                print("Stations not in the main trajectory, whose time offsets are estimated:", ", ".join(missing))

            # Fix the offsets of the stations in the main trajectory, which are then kept in the Monte Carlo runs
            #   too. They are given as fixed times with the timing estimation on, so that the timing residuals 
            #   the Monte Carlo runs are checked against are still computed
            fixed = ["{:s}:{:.6f}".format(station_id, main_offsets[meteor.ecsv_file]) 
                for station_id, meteor in zip(station_ids, meteor_list) if meteor.ecsv_file in main_offsets]
            frag_kwargs['estimate_timing_vel'] = True
            frag_kwargs['fixed_times'] = ",".join(fixed) if fixed else None

        frag_traj = solveTrajectoryGeneric(jdt_ref, meteor_list, 
            os.path.join(traj.output_dir, "fragment_{:d}".format(fragment)), solver=solver, **frag_kwargs)

        if frag_traj is None:
            print("The trajectory of fragment {:d} could not be solved.".format(fragment))
            continue

        fragment_trajs.append((fragment, originalPicksTrajectory(frag_traj)))

    return fragment_trajs


def fragmentComparison(traj, fragment_trajs, reuse_timing=True):
    """ Summarize how the trajectories of additional fragments differ from the main one, to judge at a glance
        whether they are consistent with it.

    Arguments:
        traj: [Trajectory] Solved trajectory of the main fragment, the solution with the original picks (see
            originalPicksTrajectory()).
        fragment_trajs: [list] (fragment, trajectory) pairs, from solveFragmentTrajectories().

    Keyword arguments:
        reuse_timing: [bool] Whether the fragments took the time offsets of the main trajectory.

    Return:
        [str] The summary.
    """

    def radiantSigma(t):
        """ 1-sigma of the geocentric radiant (rad) from the Monte Carlo runs, or None without them. """

        un = t.uncertainties
        if (un is None) or (getattr(un, 'ra_g', None) is None) or (getattr(un, 'dec_g', None) is None):
            return None

        return np.hypot(un.ra_g*np.cos(t.orbit.dec_g), un.dec_g)

    def valueSigma(t, name):
        """ 1-sigma of a value from the Monte Carlo runs, or None without them. """

        return getattr(t.uncertainties, name, None) if (t.uncertainties is not None) else None

    def sigmaStr(diff, sig1, sig2):
        """ The difference in units of the combined 1-sigma, if both are known. """

        if (sig1 is None) or (sig2 is None) or (np.hypot(sig1, sig2) == 0):
            return ""

        return " ({:.1f} sigma)".format(abs(diff)/np.hypot(sig1, sig2))

    def modelPoints(t):
        """ ECI positions on the fitted path of trajectory t (with the gravity drop, if corrected), sorted by
            time, with their Julian dates. """

        used = [obs for obs in t.observations if not obs.ignore_station]
        jd = np.concatenate([obs.JD_data[obs.ignore_list == 0] for obs in used])
        eci = np.concatenate([obs.model_eci[obs.ignore_list == 0] for obs in used])
        order = np.argsort(jd)

        return jd[order], eci[order]

    def interpolate(x, xp, fp):
        """ Linear interpolation of fp(xp) at x, extrapolated linearly from the end points. """

        if (x < xp[0]) and (xp[1] > xp[0]):
            return fp[0] + (x - xp[0])*(fp[1] - fp[0])/(xp[1] - xp[0])
        if (x > xp[-1]) and (xp[-1] > xp[-2]):
            return fp[-1] + (x - xp[-1])*(fp[-1] - fp[-2])/(xp[-1] - xp[-2])

        return np.interp(x, xp, fp)

    # Distance travelled along the main path by each of its points
    main_jd, main_eci = modelPoints(traj)
    direction = -vectNorm(traj.radiant_eci_mini)
    main_along = np.dot(main_eci - traj.state_vect_mini, direction)
    along_order = np.argsort(main_along)

    def alongFit(jd, eci):
        """ Distance travelled along the main path as a function of the Julian date, smoothed by a quadratic 
            fit (with the deceleration) over the noise of the individual points. """

        poly = np.polyfit((jd - main_jd[0])*86400, np.dot(eci - traj.state_vect_mini, direction), 
            min(2, len(jd) - 1))

        return lambda t: np.polyval(poly, (t - main_jd[0])*86400)

    main_along_fit = alongFit(main_jd, main_eci)

    def crossDistance(point):
        """ Distance (m) of a point from the main path, as far along it as the point. """

        along = np.dot(point - traj.state_vect_mini, direction)

        # Point of the main path as far along as the given one, with the gravity drop
        main_point = np.array([interpolate(along, main_along[along_order], main_eci[along_order, i]) 
            for i in range(3)])
        diff = point - main_point

        return vectMag(diff - np.dot(diff, direction)*direction)

    def stationsPoints(t):
        return "{:d} / {:d}".format(len(t.observations), sum(len(obs.time_data) for obs in t.observations))

    def velocityMethod(t):
        """ How the initial velocity of a solution was estimated when the drag and ablation fit was asked for, 
            which starts at the first point of each solution, or None if it was not (the straight line). """

        fit = getattr(t, 'v_init_drag_fit', None)
        if fit is not None:
            return "drag and ablation fit to {:d} points, {:.2f} to {:.2f} s, {:.2f} to {:.2f} km".format(
                fit.n_points, fit.t_range[0], fit.t_range[1], fit.ht_range[1]/1000, fit.ht_range[0]/1000)

        if getattr(t, 'v_init_drag', False):
            reason = getattr(t, 'v_init_drag_rejection', None)
            return "straight line, as the drag fit was not used{:s}".format(
                "" if reason is None else ": " + reason)

        return None

    out_str = "\n"
    out_str += "Additional fragments compared with the main fragment (fragment 0)\n"
    out_str += "-----------------------------------------------------------------\n"
    out_str += "Time offsets of the stations: {:s}\n".format("those of the main trajectory" if reuse_timing \
        else "estimated for every fragment")
    if traj.uncertainties is None:
        out_str += "Without the Monte Carlo runs the differences cannot be compared with the uncertainties.\n"
    else:
        out_str += "Solutions with the original picks, with the uncertainties (1 sigma) of the Monte Carlo runs.\n"
    if velocityMethod(traj) is not None:
        out_str += "Initial velocity of the main fragment from the {:s}\n".format(velocityMethod(traj))

    if not fragment_trajs:
        out_str += "\nNone of the additional fragments could be solved.\n"

    for fragment, frag_traj in fragment_trajs:

        out_str += "\n"
        out_str += "Fragment {:d}\n".format(fragment)
        out_str += "  Stations / points     : {:s} (main fragment: {:s})\n".format(stationsPoints(frag_traj), 
            stationsPoints(traj))

        if (frag_traj.orbit is not None) and (traj.orbit is not None) and (frag_traj.orbit.ra_g is not None) \
            and (traj.orbit.ra_g is not None):

            radiant_diff = angleBetweenSphericalCoords(traj.orbit.dec_g, traj.orbit.ra_g, frag_traj.orbit.dec_g, 
                frag_traj.orbit.ra_g)
            out_str += "  Geocentric radiant    : {:.3f} deg from the main one{:s}\n".format(np.degrees(radiant_diff), 
                sigmaStr(radiant_diff, radiantSigma(traj), radiantSigma(frag_traj)))

            v_g_diff = frag_traj.orbit.v_g - traj.orbit.v_g
            out_str += "  Geocentric velocity   : {:.3f} km/s ({:+.3f} km/s{:s})\n".format(frag_traj.orbit.v_g/1000, 
                v_g_diff/1000, sigmaStr(v_g_diff, valueSigma(traj, 'v_g'), valueSigma(frag_traj, 'v_g')))

        # The initial velocity of every solution is the one at its first point, from the method it could use
        frag_method = velocityMethod(frag_traj)
        if (velocityMethod(traj) is not None) or (frag_method is not None):
            out_str += "  Initial velocity from : the {:s}\n".format(frag_method or "straight line")
            if hasattr(traj, 'v_init_drag_fit') and ((traj.v_init_drag_fit is None) 
                != (getattr(frag_traj, 'v_init_drag_fit', None) is None)):
                out_str += "  NOTE: not the method of the main fragment, so the velocities are not directly " \
                    "comparable\n"

        v_init_diff = frag_traj.v_init - traj.v_init
        out_str += "  Initial velocity      : {:.3f} km/s ({:+.3f} km/s{:s})\n".format(frag_traj.v_init/1000, 
            v_init_diff/1000, sigmaStr(v_init_diff, valueSigma(traj, 'v_init'), valueSigma(frag_traj, 'v_init')))

        out_str += "  Begin / end height    : {:.2f} / {:.2f} km (main fragment: {:.2f} / {:.2f} km)\n".format(
            frag_traj.rbeg_ele/1000, frag_traj.rend_ele/1000, traj.rbeg_ele/1000, traj.rend_ele/1000)

        frag_jd, frag_eci = modelPoints(frag_traj)
        frag_along_fit = alongFit(frag_jd, frag_eci)
        out_str += "  Off the main path     : {:.0f} m at the fragment's first point, {:.0f} m at its last\n".format(
            crossDistance(frag_eci[0]), crossDistance(frag_eci[-1]))
        out_str += "  Ahead of the main one : {:+.0f} m at the fragment's first point, {:+.0f} m at its last " \
            "(at the same time)\n".format(*[frag_along_fit(t) - main_along_fit(t) for t in (frag_jd[0], frag_jd[-1])])

        out_str += "  First seen            : {:+.3f} s from the main fragment\n".format(
            (frag_traj.rbeg_jd - traj.rbeg_jd)*86400)

    return out_str


def saveECSV(dir_path, meteor_observations, 
             network_name='RMS', x_res=None, y_res=None, photom_band=None, img_name=None, calib_stars=None, 
             fov_mid_azim=None, fov_mid_elev=None, fov_mid_rot_horiz=None, fov_horiz=None, fov_vert=None):
    """ Save meteor observations to ECSV files. 
    
    Arguments:
        dir_path: [str] Directory path where the ECSV files will be saved.
        meteor_observations: [list] List of MeteorObservation objects to save.

    Keyword arguments:
        network_name: [str] Name of the network, used in the ECSV file name and header. "RMS" by default.
        x_res: [int] Horizontal resolution of the camera in pixels.
        y_res: [int] Vertical resolution of the camera in pixels.
        photom_band: [str] Photometric band of the star catalogue, e.g. "B", "V", etc.
        img_name: [str] Name of the original image or video file, used in the ECSV header.
        calib_stars: [int] Number of stars used in the astrometric calibration, used in the ECSV header.
        fov_mid_azim: [float] Azimuth of the centre of the field of view in decimal degrees. North = 0, 
            increasing to the East.
        fov_mid_elev: [float] Elevation of the centre of the field of view in decimal degrees. Horizon = 0,
            Zenith = 90.
        fov_mid_rot_horiz: [float] Rotation of the field of view from horizontal, decimal degrees. Clockwise 
            is positive.
        fov_horiz: [float] Horizontal extent of the field of view in decimal degrees.
        fov_vert: [float] Vertical extent of the field of view in decimal degrees.
    
    """

    for meteor in meteor_observations:
        
        # Get the reference datetime
        dt_ref = jd2Date(meteor.jdt_ref, dt_obj=True)

        isodate_format_file = "%Y-%m-%dT%H_%M_%S"
        isodate_format_entry = "%Y-%m-%dT%H:%M:%S.%f"

        # Construct the file name
        # E.g. 2025-06-24T07_55_23_RMS_CA003D.ecsv
        ecsv_name = f"{dt_ref.strftime(isodate_format_file)}_{network_name}_{meteor.station_id}.ecsv"
        ecsv_path = os.path.join(dir_path, ecsv_name)

        
        # Prepare the metadata/header
        meta_dict = {
            'obs_latitude': np.degrees(meteor.latitude),   # Decimal signed latitude (-90 S to +90 N)
            'obs_longitude': np.degrees(meteor.longitude), # Decimal signed longitude (-180 W to +180 E)
            'obs_elevation': meteor.height,                # Altitude in metres above MSL. Note not WGS84
            'origin': 'SkyFit2',                           # The software which produced the data file
            'camera_id': meteor.station_id,                # The code name of the camera, likely to be network-specific
            'cx' : x_res,                                  # Horizontal camera resolution in pixels
            'cy' : y_res,                                  # Vertical camera resolution in pixels
            'photometric_band' : photom_band,              # The photometric band of the star catalogue
            'image_file' : img_name,                       # The name of the original image or video
            'isodate_start_obs': str(dt_ref.strftime(isodate_format_entry)), # The date and time of the start of the video or exposure
            'astrometry_number_stars' : calib_stars,       # The number of stars identified and used in the astrometric calibration
            'mag_label': 'mag_data',                       # The label of the Magnitude column in the Point Observation data
            'no_frags': 1,                                 # The number of meteoroid fragments described in this data
            'obs_az': fov_mid_azim,                        # The azimuth of the centre of the field of view in decimal degrees. North = 0, increasing to the East
            'obs_ev': fov_mid_elev,                        # The elevation of the centre of the field of view in decimal degrees. Horizon =0, Zenith = 90
            'obs_rot': fov_mid_rot_horiz,                  # Rotation of the field of view from horizontal, decimal degrees. Clockwise is positive
            'fov_horiz': fov_horiz,                        # Horizontal extent of the field of view, decimal degrees
            'fov_vert': fov_vert,                          # Vertical extent of the field of view, decimal degrees
           }

                # Write the header
        out_str = """# %ECSV 0.9
# ---
# datatype:
# - {name: datetime, datatype: string}
# - {name: ra, unit: deg, datatype: float64}
# - {name: dec, unit: deg, datatype: float64}
# - {name: azimuth, datatype: float64}
# - {name: altitude, datatype: float64}
# - {name: x_image, unit: pix, datatype: float64}
# - {name: y_image, unit: pix, datatype: float64}
# - {name: integrated_pixel_value, datatype: int64}
# - {name: background_pixel_value, datatype: int64}
# - {name: saturated_pixels, datatype: bool}
# - {name: mag_data, datatype: float64}
# - {name: err_minus_mag, datatype: float64}
# - {name: err_plus_mag, datatype: float64}
# - {name: snr, datatype: float64}
# delimiter: ','
# meta: !!omap
"""
        # Add the meta information
        for key in meta_dict:

            value = meta_dict[key]

            if isinstance(value, str):
                value_str = "'{:s}'".format(value)
            else:
                value_str = str(value)

            out_str += "# - {" + "{:s}: {:s}".format(key, value_str) + "}\n"

        
        out_str += "# schema: astropy-2.0\n"
        out_str += "datetime,ra,dec,azimuth,altitude,x_image,y_image,mag_data\n"


        # Go though the meteor points
        for (
            t_rel, 
            x_centroid, y_centroid, 
            azim, alt, ra, dec, 
            mag
            ) in zip(
            meteor.time_data, 
            meteor.x_data, meteor.y_data, 
            meteor.azim_data, meteor.elev_data, meteor.ra_data, meteor.dec_data, 
            meteor.mag_data
            ):

            # Compute the absolute time
            frame_time = dt_ref + datetime.timedelta(seconds=t_rel)


            # Precess RA/Dec to J2000
            ra_J2000, dec_J2000 = equatorialCoordPrecession_vect(meteor.jdt_ref, J2000_JD.days, ra, dec)


            # Add an entry to the ECSV file
            entry = [
                frame_time.strftime(isodate_format_entry),
                "{:10.6f}".format(np.degrees(ra_J2000)), "{:+10.6f}".format(np.degrees(dec_J2000)),
                "{:10.6f}".format(np.degrees(azim)), "{:+10.6f}".format(np.degrees(alt)),
                "{:9.3f}".format(x_centroid), "{:9.3f}".format(y_centroid),
                "{:+7.2f}".format(mag)
                ]

            out_str += ",".join(entry) + "\n"


        # Write file to disk
        with open(ecsv_path, 'w') as f:
            f.write(out_str)


        print("ESCV file saved to:", ecsv_path)
            




if __name__ == "__main__":

    import argparse


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Run the trajectory solver on DFN ECSV files.")

    arg_parser.add_argument('ecsv_files', nargs="+", metavar='ECSV_PATH', type=str, \
        help="Path to 2 of more ECSV files. Wildcards are supported, so e.g. /path/to/*.ecsv also works.")

    # Add other solver options
    arg_parser = addSolverOptions(arg_parser, skip_velpart=True)

    arg_parser.add_argument('-p', '--velpart', metavar='VELOCITY_PART', \
        help="Fixed part from the beginning of the meteor on which the initial velocity estimation using the sliding fit will start. Default is 0.4 (40 percent), but for noisier data this might be bumped up to 0.5.", \
        type=float, default=0.4)

    arg_parser.add_argument('-w', '--walk', \
        help="Recursively find all ECSV files in the given folder and use them for trajectory estimation. If a directory containing the file contains the word 'REJECT', it will be skipped. ", \
        action="store_true")
    
    arg_parser.add_argument('--writemilig', metavar='MILIG_PATH', type=str, \
        help="Write the observations to a MILIG input file and exit. The MILIG_PATH argument is the path to the output file.")

    arg_parser.add_argument('--fragments', \
        help="After the main fragment, also solve the trajectories of the additional fragments in the ECSV files (columns with a numeric suffix, e.g. azimuth1, as in Appendix 4 of the GFE standard), with the same options, and print how they differ from the main one. Each one is saved in a fragment_k folder of the output directory.", \
        action="store_true")

    arg_parser.add_argument('--fragtimefit', \
        help="With --fragments, estimate the time offsets of the stations for every fragment instead of taking the ones of the main trajectory, which are better constrained by its larger number of points.", \
        action="store_true")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    ### Parse command line arguments ###

    ecsv_paths = []
    print('Using ECSV files:')


    # If the recursive walk option is given, find all ECSV files recursively in the given folder
    if cml_args.walk:

        # Take the dir path as the given path
        dir_path = cml_args.ecsv_files[0]

        # Find all manual reduction ECSV files in the given folder
        ecsv_names = []
        for entry in sorted(os.walk(dir_path), key=lambda x: x[0]):

            dir_name, _, file_names = entry

            # Skip all directories with the word "REJECT" in them
            if "REJECT" in dir_name:
                print("Directory {:s} skipped because it contains 'REJECT'.".format(dir_name))
                continue

            # Add all ECSV files with picks to the processing list
            for fn in file_names:
                if fn.lower().endswith(".ecsv"):

                    # Add ECSV file, but skip duplicates
                    if fn not in ecsv_names:
                        ecsv_paths.append(os.path.join(dir_name, fn))
                        ecsv_names.append(fn)

                        print(fn)


    else:
        for ecsv_p in cml_args.ecsv_files:
            for ecsv_full_p in glob.glob(ecsv_p):
                ecsv_full_path = os.path.abspath(ecsv_full_p)

                # Check that the path exists
                if os.path.exists(ecsv_full_path):
                    ecsv_paths.append(ecsv_full_path)
                    print(ecsv_full_path)
                else:
                    print('File not found:', ecsv_full_path)


        # Extract dir path
        dir_path = os.path.dirname(ecsv_paths[0])


    # Load the observations into container objects
    jdt_ref, meteor_list = loadECSVs(ecsv_paths)


    # Write the observations to a MILIG input file and exit
    if cml_args.writemilig:
        writeMiligInputFileMeteorObservation(jdt_ref, meteor_list, cml_args.writemilig)
        print("MILIG input file written to:", cml_args.writemilig)
        print("Exiting...")
        sys.exit()



    # Check that there are more than 2 ECSV files given
    if len(ecsv_paths) < 2:
        print("At least 2 files are needed for trajectory estimation!")
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


    # Solve the trajectory
    traj = solveTrajectoryGeneric(jdt_ref, meteor_list, dir_path, solver=cml_args.solver, \
        max_toffset=max_toffset, monte_carlo=(not cml_args.disablemc), mc_runs=cml_args.mcruns, \
        geometric_uncert=cml_args.uncertgeom, gravity_correction=(not cml_args.disablegravity), 
        gravity_factor=cml_args.gfact,
        plot_all_spatial_residuals=cml_args.plotallspatial, plot_file_type=cml_args.imgformat, \
        show_plots=(not cml_args.hideplots), v_init_part=velpart, v_init_ht=vinitht, \
        show_jacchia=cml_args.jacchia,
        estimate_timing_vel=(False if cml_args.notimefit is None else cml_args.notimefit), \
        fixed_times=cml_args.fixedtimes, mc_noise_std=cml_args.mcstd, enable_OSM_plot=cml_args.enableOSM)


    # Solve the trajectories of the additional fragments, after the main one
    if cml_args.fragments and (traj is not None):

        if not ecsvFragments(ecsv_paths):
            print()
            print("There are no additional fragments in the ECSV files.")

        else:
            fragment_trajs = solveFragmentTrajectories(traj, ecsv_paths, reuse_timing=(not cml_args.fragtimefit))
            print(fragmentComparison(originalPicksTrajectory(traj), fragment_trajs, 
                reuse_timing=(not cml_args.fragtimefit)))
