""" Solve the trajectory from observations in ECSV files. """

from __future__ import print_function, division, absolute_import

import os
import sys
import copy
import glob
import datetime
import itertools

import numpy as np
import scipy.optimize

from wmpl.Formats.GenericFunctions import addSolverOptions, solveTrajectoryGeneric, MeteorObservation, \
    prepareObservations, writeMiligInputFileMeteorObservation
from wmpl.Trajectory.StationSystematics import stationSystematicErrors, systematicErrorSize
from wmpl.Utils.Math import vectNorm, vectMag, angleBetweenSphericalCoords
from wmpl.Utils.AtmosphereDensity import getAtmDensity_vect
from wmpl.Utils.DragFit import DragFit
from wmpl.Trajectory.Trajectory import ObservedPoints
from wmpl.Utils.Pickling import loadPickle
from wmpl.Utils.TrajConversions import J2000_JD, datetime2JD, altAz2RADec_vect, \
    equatorialCoordPrecession_vect, jd2Date



# Use a fixed FPS (not really important in the grand scheme of things, it's just for compatibility with
#   used functions)
FPS = 15


def ecsvFragments(ecsv_paths):
    """ Find the additional fragments described in ECSV files, whose columns have a numeric suffix (e.g.
        ra1, azimuth1) as in Appendix 4 of the GFE standard.

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

            suffix = name.strip()[len('azimuth'):]
            if name.strip().startswith('azimuth') and suffix.isdigit() and int(suffix):
                fragments.add(int(suffix))

    return sorted(fragments)



def loadECSVs(ecsv_paths, no_prepare=False, fragment=0, flares=False):
    """ Load meteor observations from ECSV files. 
    
    Arguments:
        ecsv_paths: [list] List of paths to ECSV files.

    Keyword arguments:
        no_prepare: [bool] If True, only load the observations, do not prepare them for the solver.
        fragment: [int] Fragment to load. 0 (default) is the main fragment, without the points flagged as not
            to be used in its trajectory (trajectory_use = False). An additional fragment k is read from the
            columns with the suffix k (e.g. azimuth1), as in Appendix 4 of the GFE standard, which makes the
            suffix 0 optional for the main fragment; files without the columns of the fragment are skipped.
        flares: [bool] Only load the points of the main fragment flagged as flares (flare = True), also those
            not used in its trajectory, whose flags are kept in the trajectory_use array of every observation.
            The flare_rows array of every observation keeps the position of every flare point among all the
            points of the main fragment in its file, in time.
            The minimum number of points of an observation does not apply.
    
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

            # Suffix of the columns of the fragment, optional for the main one (e.g. azimuth or azimuth0).
            #   Skip the files which do not describe it
            suffix = next((sfx for sfx in ([str(fragment)] if fragment else ['', '0']) \
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

            # Whether the points of the main fragment are to be used in its trajectory
            if (not fragment) and ('trajectory_use' in header):
                trajectory_use = np.char.lower(np.char.strip(data[:, header.index('trajectory_use')])) != 'false'
            else:
                trajectory_use = np.ones(len(data), dtype=bool)

            # Only the flares (used in the trajectory or not), or only the points used in the trajectory
            if flares:
                if (fragment) or ('flare' not in header):
                    continue

                # Position of every point among the points of the main fragment in time, sorting the times of
                #   fixed format as text, to tell apart two flares with a point not flagged between them
                time_order = np.argsort(np.char.strip(data[:, dt_indx]), kind='stable')
                row_order = np.empty(len(data), dtype=int)
                row_order[time_order] = np.arange(len(data))

                flare = np.char.lower(np.char.strip(data[:, header.index('flare')])) == 'true'
                data, trajectory_use, flare_rows = data[flare], trajectory_use[flare], row_order[flare]

            else:
                data = data[trajectory_use]

            if len(data) == 0:
                if not flares:
                    print("The station {:s} has no points of fragment {:d}, skipping: {:s}".format( \
                        station_id, fragment, ecsv_file))
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

                # SkyFit2 sets the integrated pixel value to 1 if no photometry pixels were chosen, which
                #   leaves the magnitude at the photometric offset, so these points get no magnitude
                if 'integrated_pixel_value' in header:
                    no_photometry = data[:, header.index('integrated_pixel_value')].astype(np.float64) <= 1
                    mag_data[no_photometry] = np.nan

                    if np.any(no_photometry):
                        print(("Station {:s}: {:d} of {:d} points were picked without photometry " \
                            + "(integrated pixel value 1), they have no magnitude").format(station_id, \
                            int(np.sum(no_photometry)), len(mag_data)))

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

            # Frame numbers of the points in their order sorted by finish(), for the duration of a frame
            meteor.frame_numbers = None
            if 'frame_number' in header:
                frame_numbers = np.array([float(frame) if frame.strip() else np.nan \
                    for frame in data[:, header.index('frame_number')]])
                meteor.frame_numbers = frame_numbers[np.argsort(time_data, kind='stable')]

            # Flags of the points, in their order sorted by finish()
            if flares:
                meteor.trajectory_use = trajectory_use[np.argsort(time_data, kind='stable')]
                meteor.flare_rows = flare_rows[np.argsort(time_data, kind='stable')]
                meteor_list.append(meteor)
                continue


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
    """ Compute the station IDs the trajectory solver gives the observations when they are added in the
        given order. A station whose ID was already given gets the suffix _2, or the one after the largest
        suffix given to it (see Trajectory.infillTrajectory()).

    Arguments:
        meteors: [list] MeteorObservation objects, in the order they are given to the solver.

    Return:
        [list] Station IDs.
    """

    station_ids = []
    for meteor in meteors:

        station_id = str(meteor.station_id)

        if station_id in station_ids:

            # Suffixes already given to this station
            suffixes = [int(sid.split('_')[-1]) for sid in station_ids \
                if sid.startswith(station_id + '_') and sid.split('_')[-1].isdigit()]

            station_id += '_' + str(max(suffixes) + 1 if suffixes else 2)

        station_ids.append(station_id)

    return station_ids



def parseStationOffsets(offsets_str):
    """ Parse the time offsets given per station on the command line, e.g. "CA001A:0.42,CA0005:-0.3".

    Arguments:
        offsets_str: [str] The offsets. Anything else than a string (e.g. True or None) gives none.

    Return:
        [dict] {station ID: time offset (s)}.
    """

    if not isinstance(offsets_str, str):
        return {}

    offsets = {}
    for entry in offsets_str.split(','):
        if ':' in entry:
            station_id, offset = entry.rsplit(':', 1)
            offsets[station_id] = float(offset)

    return offsets



def formatStationOffsets(offsets):
    """ Format time offsets per station as they are given on the command line (see parseStationOffsets()).

    Arguments:
        offsets: [dict] {station ID: time offset (s)}.

    Return:
        [str] The offsets, None if there are none.
    """

    if not offsets:
        return None

    return ",".join("{:s}:{:.6f}".format(station_id, offset) for station_id, offset in offsets.items())



def fragmentStationOffsets(offsets, main_ids, station_ids, meteors):
    """ Rename the time offsets given per station for the main trajectory to the station IDs of a fragment.
        A station with several files can have other IDs in the fragment, e.g. when the fragment is not in its
        first file.

    Arguments:
        offsets: [dict] {station ID in the main trajectory: time offset (s)}, see parseStationOffsets().
        main_ids: [dict] {ecsv_file: station ID in the main trajectory}.
        station_ids: [list] Station IDs of the observations of the fragment, see solverStationIDs().
        meteors: [list] MeteorObservation objects of the fragment.

    Return:
        [dict] {station ID in the fragment: time offset (s)}.
    """

    frag_offsets = {}
    for station_id, meteor in zip(station_ids, meteors):

        # A file of the main trajectory takes the offset given under its ID there, another file the one given
        #   under its own ID, unless that ID is the one of another file of the main trajectory
        main_id = main_ids.get(meteor.ecsv_file)
        if main_id is None:
            if station_id in main_ids.values():
                continue

            main_id = station_id

        if main_id in offsets:
            frag_offsets[station_id] = offsets[main_id]

    return frag_offsets



def appliedTimeOffsets(traj, input_meteors):
    """ Compute the time offset applied to every observation of a solved trajectory, as the shift of its
        times from the input ones. It includes the offsets fixed in the input, and can differ by about a
        millisecond from the reported timing offsets, which do not account for every iteration of the timing
        estimation.

    Arguments:
        traj: [Trajectory object] Solved trajectory.
        input_meteors: [list] MeteorObservation objects it was solved from, from loadECSVs(), in the order
            they were given to the solver.

    Return:
        [dict] {ecsv_file: time offset (s)} of every input observation matched by a solved one which was not
            ignored, by the station ID the solver gave it (see solverStationIDs()), its number of points and
            its time steps.
    """

    solved = {str(obs.station_id): obs for obs in traj.observations}

    offsets = {}
    for station_id, meteor in zip(solverStationIDs(input_meteors), input_meteors):

        obs = solved.get(station_id)
        input_jd = meteor.jdt_ref + np.array(meteor.time_data)/86400

        # Skip the stations the solver ignored, e.g. without an overlap in time to estimate their offsets
        if (obs is None) or getattr(obs, 'ignore_station', False):
            continue

        # Check that the solved observation has the same points
        if len(input_jd) != len(obs.JD_data):
            continue

        if not np.allclose((input_jd - input_jd[0])*86400, (obs.JD_data - obs.JD_data[0])*86400, atol=1e-3):
            continue

        offsets[meteor.ecsv_file] = np.mean(obs.JD_data - input_jd)*86400

    return offsets



def originalPicksTrajectory(traj):
    """ Load the solution with the original picks of a trajectory solved with the Monte Carlo runs, for
        which the solver returns the best of the runs instead. It is loaded from the pickle saved with the
        results.

    Arguments:
        traj: [Trajectory object] Solved trajectory, from solveTrajectoryGeneric().

    Return:
        [Trajectory object] The solution with the original picks, with the uncertainties of the Monte Carlo
            runs. The given trajectory if it was solved without them, or if its results were not saved.
    """

    if (traj.uncertainties is None) or (not traj.save_results):
        return traj

    traj_path = os.path.join(traj.output_dir, traj.file_name + '_trajectory.pickle')
    if not os.path.isfile(traj_path):
        return traj

    return loadPickle(*os.path.split(traj_path))



def solveFragmentTrajectories(traj, ecsv_paths, solver='original', reuse_timing=True, systematics=True, \
    **kwargs):
    """ Solve the trajectories of the additional fragments described in ECSV files (see ecsvFragments()),
        after the main one and with the same solver options. Each one is saved in a fragment_k folder of the
        output directory of the main trajectory.

    Arguments:
        traj: [Trajectory object] Solved trajectory of the main fragment, from solveTrajectoryGeneric().
        ecsv_paths: [list] List of paths to ECSV files.

    Keyword arguments:
        solver: [str] Solver the main trajectory was solved with, only "original" is supported here.
            "original" by default.
        reuse_timing: [bool] Fix the time offsets of the stations to the ones of the main trajectory, those
            of its solution with the original picks, instead of estimating them again from the fewer points
            of every fragment. Only used when the time offsets are estimated, as otherwise the fragments
            take the fixed ones of the main trajectory anyway. True by default.
        systematics: [bool] Include in the Monte Carlo uncertainties of every fragment the systematic errors
            of the stations, measured on the main trajectory if it has 3 stations or more (see
            StationSystematics). True by default.
        **kwargs: Keyword arguments for the trajectory solver, those the main trajectory was solved with.

    Return:
        [list] (fragment, trajectory) of every fragment solved, with the solution with the original picks
            (see originalPicksTrajectory()).
    """

    if solver != 'original':
        print("The trajectories of additional fragments can only be solved with the original solver.")
        return []

    _, input_meteors = loadECSVs(ecsv_paths, no_prepare=True)
    main_offsets = appliedTimeOffsets(originalPicksTrajectory(traj), input_meteors)

    # Station IDs of the files in the main trajectory, under which the offsets on the command line are given
    main_ids = dict(zip([meteor.ecsv_file for meteor in input_meteors], solverStationIDs(input_meteors)))

    # The time offsets are estimated unless the timing estimation was turned off, or given offsets
    #   for every station (estimate_timing_vel is then False or a string)
    estimate_timing = (kwargs.get('estimate_timing_vel', True) is True)

    # Measure the systematic errors of the stations on the main trajectory, which needs 3 distinct stations
    main_errors = {}
    if systematics and kwargs.get('monte_carlo', True):

        if len(set(str(meteor.station_id) for meteor in input_meteors)) >= 3:
            traj_orig = originalPicksTrajectory(traj)
            main_errors = stationSystematicErrors(traj_orig)

        if not main_errors:
            print("The systematic errors of the stations can only be measured with 3 stations or more in " \
                "the main trajectory, the uncertainties of the fragments do not include them.")

    fragment_trajs = []
    for fragment in ecsvFragments(ecsv_paths):

        print()
        print("Solving the trajectory of fragment {:d}...".format(fragment))

        # A fragment needs two stations, which several files of the same station are not
        _, meteor_list = loadECSVs(ecsv_paths, no_prepare=True, fragment=fragment)
        n_stations = len(set(str(meteor.station_id) for meteor in meteor_list))
        if n_stations < 2:
            print("Fragment {:d} has enough points (4 or more) from only {:d} station(s), but 2 are needed, "
                "skipping.".format(fragment, n_stations))
            continue

        jdt_ref, meteor_list = prepareObservations(meteor_list)

        # Station IDs the solver will give the observations of the fragment
        station_ids = solverStationIDs(meteor_list)

        # Offsets fixed on the command line, under the station IDs of the fragment
        given_fixed = fragmentStationOffsets(parseStationOffsets(kwargs.get('fixed_times')), main_ids, \
            station_ids, meteor_list)

        frag_kwargs = dict(kwargs)
        if not estimate_timing:

            # Without the timing estimation, keep the offsets given for every station, under the IDs of the
            #   fragment
            if isinstance(kwargs.get('estimate_timing_vel'), str):
                given = fragmentStationOffsets(parseStationOffsets(kwargs['estimate_timing_vel']), main_ids, \
                    station_ids, meteor_list)
                frag_kwargs['estimate_timing_vel'] = formatStationOffsets(given) or False

        elif reuse_timing:

            # Fix the offsets of the stations in the main trajectory, which are then kept in the Monte Carlo
            #   runs too, and those fixed on the command line for the other ones. They are given as fixed
            #   times with the timing estimation on, so that the timing residuals the Monte Carlo runs are
            #   checked against are still computed
            fixed = dict(given_fixed)
            for station_id, meteor in zip(station_ids, meteor_list):
                if meteor.ecsv_file in main_offsets:
                    fixed[station_id] = main_offsets[meteor.ecsv_file]

            missing = [station_id for station_id in station_ids if station_id not in fixed]
            if missing:
                print("Stations not in the main trajectory, whose time offsets are estimated:", \
                    ", ".join(missing))

            frag_kwargs['fixed_times'] = formatStationOffsets(fixed)

        elif isinstance(kwargs.get('fixed_times'), str):
            frag_kwargs['fixed_times'] = formatStationOffsets(given_fixed)

        # Add the systematic errors of the stations at the fragment to its Monte Carlo runs
        if main_errors:
            frag_kwargs['mc_systematics'] = systematicErrorSize(traj_orig, main_errors, meteor_list)
            print("Systematic errors of the stations in the Monte Carlo runs: {:.1f} arcsec across the " \
                "trajectory, {:.1f} arcsec/km along it".format( \
                np.degrees(frag_kwargs['mc_systematics']['sigma_c'])*3600, \
                np.degrees(frag_kwargs['mc_systematics']['sigma_g'])*3600*1000))

        frag_traj = solveTrajectoryGeneric(jdt_ref, meteor_list, \
            os.path.join(traj.output_dir, "fragment_{:d}".format(fragment)), solver=solver, **frag_kwargs)

        if frag_traj is None:
            print("The trajectory of fragment {:d} could not be solved.".format(fragment))
            continue

        fragment_trajs.append((fragment, originalPicksTrajectory(frag_traj)))

    return fragment_trajs



def localMotionFit(t, along, window=0.3):
    """ Fit the distance travelled along a path locally, to get it at any time. The deceleration of a
        meteor is not a parabola over the whole flight, and one would be off by tens of metres at the end.

    Arguments:
        t: [ndarray] Times of the points (s).
        along: [ndarray] Distance along the path of every point (m).

    Keyword arguments:
        window: [float] The fit uses the points within this time (s), or those of the 10 closest distinct
            times if there are fewer. Past the ends of the path the window is that of its closest end. 0.3 by
            default.

    Return:
        [function] Distance along the path at a given time.
    """

    times = np.unique(t)

    def motion(time):

        # Past the ends of the path, fit its last part
        centre = np.clip(time, times[0], times[-1])

        # Take enough distinct times, as the points of several stations can share one
        near_times = times[np.abs(times - centre) <= window]
        if len(near_times) < 10:
            near_times = times[np.argsort(np.abs(times - centre))[:10]]

        near = np.isin(t, near_times)

        return np.polyval(np.polyfit(t[near], along[near], min(2, len(near_times) - 1)), time)

    return motion



# Distance off the main path (m) beyond which no fragment of the main went in fireballs simulated with MetSim
FRAGMENT_OFF_PATH_LIMIT = 1500.0

# Least convergence angle of the stations (deg) the trajectory correlator of WMPL takes (see CorrelateEngine)
FRAGMENT_MIN_CONVERGENCE = 3.0



def convergenceAngle(traj):
    """ Get the convergence angle of the best pair of stations of a solved trajectory.

    Arguments:
        traj: [Trajectory object] Solved trajectory.

    Return:
        [float] The angle (deg), None if the solver did not compute it.
    """

    best = getattr(traj, 'best_conv_inter', None)
    if best is None:
        return None

    return np.degrees(best.conv_angle)



def trajectoryModelPoints(traj):
    """ Get the ECI positions on the fitted path of a solved trajectory (with the gravity drop, if corrected),
        sorted by time, with their Julian dates.

    Arguments:
        traj: [Trajectory object] Solved trajectory.

    Return:
        (jd, eci): [tuple of ndarrays] Julian dates and ECI positions (m) of its used points.
    """

    used = [obs for obs in traj.observations if not obs.ignore_station]
    jd = np.concatenate([obs.JD_data[obs.ignore_list == 0] for obs in used])
    eci = np.concatenate([obs.model_eci[obs.ignore_list == 0] for obs in used])
    order = np.argsort(jd)

    return jd[order], eci[order]



def offMainPath(traj, frag_traj):
    """ Compute how far the first and the last point of a fragment are from the main path, each one from the
        point of the main path as far along it.

    Arguments:
        traj: [Trajectory object] Solved trajectory of the main fragment.
        frag_traj: [Trajectory object] Solved trajectory of the fragment.

    Return:
        (off_first, off_last): [tuple of floats] The distances (m).
    """

    def interpolate(x, xp, fp):
        """ Interpolate fp(xp) linearly at x, extrapolating linearly from the end points. """

        if (x < xp[0]) and (xp[1] > xp[0]):
            return fp[0] + (x - xp[0])*(fp[1] - fp[0])/(xp[1] - xp[0])

        if (x > xp[-1]) and (xp[-1] > xp[-2]):
            return fp[-1] + (x - xp[-1])*(fp[-1] - fp[-2])/(xp[-1] - xp[-2])

        return np.interp(x, xp, fp)


    # Distance travelled along the main path by each of its points
    _, main_eci = trajectoryModelPoints(traj)
    direction = -vectNorm(traj.radiant_eci_mini)
    main_along = np.dot(main_eci - traj.state_vect_mini, direction)
    along_order = np.argsort(main_along)

    def crossDistance(point):
        """ Compute the distance (m) of a point from the main path, as far along it as the point. """

        along = np.dot(point - traj.state_vect_mini, direction)

        # Point of the main path as far along as the given one, with the gravity drop
        main_point = np.array([interpolate(along, main_along[along_order], main_eci[along_order, i]) \
            for i in range(3)])
        diff = point - main_point

        return vectMag(diff - np.dot(diff, direction)*direction)


    _, frag_eci = trajectoryModelPoints(frag_traj)

    return crossDistance(frag_eci[0]), crossDistance(frag_eci[-1])



# Largest distance along the path of a fragment between two of its stations at the same times, with the time
#   offsets of the main trajectory (m). In fireballs simulated with MetSim 0.2-0.3 % of the fragments of the
#   main one were further, and 56-70 % of those whose stations measured different fragments
FRAGMENT_TIMING_LIMIT = 1000.0



def projectionMismatch(t_a, along_a, t_b, along_b, deg=2):
    """ Measure how far apart along the path the projections of a fragment from two stations are at the same
        times, as the constant difference of the second station in a joint polynomial fit of their positions
        in time. The points further from the fit than 4 times its robust scatter are dropped, keeping at
        least 2 points of each station.

    Arguments:
        t_a: [ndarray] Times of the points of the first station (s).
        along_a: [ndarray] Their positions along the path (m).
        t_b: [ndarray] Times of the points of the second station (s).
        along_b: [ndarray] Their positions along the path (m).

    Keyword arguments:
        deg: [int] Degree of the polynomial, lowered when there are few points. 2 by default.

    Return:
        [float] Position of the second station minus that of the first (m), None with too few points.
    """

    t = np.concatenate([t_a, t_b])
    along = np.concatenate([along_a, along_b])
    is_b = np.concatenate([np.zeros(len(t_a)), np.ones(len(t_b))])

    deg = min(deg, len(t) - 3)
    if deg < 1:
        return None

    design = np.column_stack([(t - np.mean(t))**p for p in range(deg + 1)] + [is_b])

    # Drop the outliers, with a scatter of at least a metre for points without noise
    good = np.ones(len(t), dtype=bool)
    for _ in range(4):

        coeffs = np.linalg.lstsq(design[good], along[good], rcond=None)[0]
        residuals = along - design.dot(coeffs)
        scatter = 1.4826*np.median(np.abs(residuals[good]))
        new_good = np.abs(residuals) < 4*max(scatter, 1.0)

        if (np.sum(new_good & (is_b == 0)) < 2) or (np.sum(new_good & (is_b == 1)) < 2) \
            or np.array_equal(new_good, good):
            break

        good = new_good

    return np.linalg.lstsq(design[good], along[good], rcond=None)[0][-1]



def stationsApart(traj):
    """ Measure how far apart along the path of a solved trajectory every two of its stations see the meteor
        at the same times, from their points on the path (see projectionMismatch()). Stations which measured
        different fragments under one number give a line on which they are far apart, unless the time
        offsets were estimated for that line, which absorbs it.

    Arguments:
        traj: [Trajectory object] Solved trajectory.

    Return:
        [dict] {(station ID, station ID): distance (m)} for every two stations with 3 used points or more.
    """

    direction = -vectNorm(traj.radiant_eci_mini)

    tracks = {}
    for obs in traj.observations:

        if getattr(obs, 'ignore_station', False):
            continue

        used = np.array(obs.ignore_list) == 0
        if np.sum(used) < 3:
            continue

        tracks[str(obs.station_id)] = ((np.array(obs.JD_data)[used] - traj.jdt_ref)*86400, \
            np.dot(np.array(obs.model_eci)[used] - traj.state_vect_mini, direction))

    apart = {}
    for station_a, station_b in itertools.combinations(sorted(tracks), 2):

        (t_a, along_a), (t_b, along_b) = tracks[station_a], tracks[station_b]
        mismatch = projectionMismatch(t_a, along_a, t_b, along_b)
        if mismatch is not None:
            apart[(station_a, station_b)] = abs(mismatch)

    return apart



def fragmentComparison(traj, fragment_trajs, reuse_timing=True):
    """ Summarize how the trajectories of additional fragments differ from the main one, to judge at a
        glance whether they are consistent with it.

    Arguments:
        traj: [Trajectory object] Solved trajectory of the main fragment, the solution with the original
            picks (see originalPicksTrajectory()).
        fragment_trajs: [list] (fragment, trajectory) pairs, from solveFragmentTrajectories().

    Keyword arguments:
        reuse_timing: [bool] Whether the fragments took the time offsets of the main trajectory. True by
            default.

    Return:
        [str] The summary.
    """

    def radiantSigma(t):
        """ Compute the 1 sigma of the geocentric radiant (rad) from the Monte Carlo runs, None without
            them.
        """

        un = t.uncertainties
        if (un is None) or (getattr(un, 'ra_g', None) is None) or (getattr(un, 'dec_g', None) is None):
            return None

        return np.hypot(un.ra_g*np.cos(t.orbit.dec_g), un.dec_g)


    def valueSigma(t, name):
        """ Get the 1 sigma of a value from the Monte Carlo runs, None without them. """

        if t.uncertainties is None:
            return None

        return getattr(t.uncertainties, name, None)


    def sigmaStr(diff, sig1, sig2):
        """ Format the difference in units of the combined 1 sigma, if both are known. """

        if (sig1 is None) or (sig2 is None) or (np.hypot(sig1, sig2) == 0):
            return ""

        return " ({:.1f} sigma)".format(abs(diff)/np.hypot(sig1, sig2))


    # Distance travelled along the main path by each of its points
    main_jd, main_eci = trajectoryModelPoints(traj)
    direction = -vectNorm(traj.radiant_eci_mini)
    main_along = np.dot(main_eci - traj.state_vect_mini, direction)
    main_motion = localMotionFit((main_jd - main_jd[0])*86400, main_along)


    def stationsPoints(t):
        """ Format the number of stations and points of a trajectory. """

        return "{:d} / {:d}".format(len(t.observations), sum(len(obs.time_data) for obs in t.observations))


    out_str = "\n"
    out_str += "Additional fragments compared with the main fragment (fragment 0)\n"
    out_str += "-----------------------------------------------------------------\n"
    out_str += "Time offsets of the stations: {:s}\n".format("those of the main trajectory" if reuse_timing \
        else "estimated for every fragment")
    if traj.uncertainties is None:
        out_str += "Without the Monte Carlo runs the differences cannot be compared with the uncertainties.\n"
    else:
        out_str += "Solutions with the original picks, with the uncertainties (1 sigma) of the Monte Carlo " \
            "runs.\n"

        if any(getattr(frag_traj, 'mc_systematics', None) is not None for _, frag_traj in fragment_trajs):
            out_str += "The uncertainties of the fragments include the systematic errors of the stations, " \
                "measured on the main trajectory.\n"

    if not fragment_trajs:
        out_str += "\nNone of the additional fragments could be solved.\n"

    for fragment, frag_traj in fragment_trajs:

        out_str += "\n"
        out_str += "Fragment {:d}\n".format(fragment)
        out_str += "  Stations / points     : {:s} (main fragment: {:s})\n".format( \
            stationsPoints(frag_traj), stationsPoints(traj))

        # Warn about stations which hardly converge for the fragment, which leave its line unconstrained
        conv_angle, main_conv_angle = convergenceAngle(frag_traj), convergenceAngle(traj)
        if conv_angle is not None:
            out_str += "  Convergence angle     : {:.1f} deg{:s}\n".format(conv_angle, \
                "" if main_conv_angle is None else " (main fragment: {:.1f} deg)".format(main_conv_angle))
            if conv_angle < FRAGMENT_MIN_CONVERGENCE:
                out_str += "  NOTE: less than {:.0f} deg, the least the trajectory correlator of WMPL " \
                    "takes: its trajectory\n        is poorly constrained\n".format(FRAGMENT_MIN_CONVERGENCE)

        if (frag_traj.orbit is not None) and (traj.orbit is not None) and (frag_traj.orbit.ra_g is not None) \
            and (traj.orbit.ra_g is not None):

            radiant_diff = angleBetweenSphericalCoords(traj.orbit.dec_g, traj.orbit.ra_g, \
                frag_traj.orbit.dec_g, frag_traj.orbit.ra_g)
            out_str += "  Geocentric radiant    : {:.3f} deg from the main one{:s}\n".format( \
                np.degrees(radiant_diff), sigmaStr(radiant_diff, radiantSigma(traj), radiantSigma(frag_traj)))

            v_g_diff = frag_traj.orbit.v_g - traj.orbit.v_g
            out_str += "  Geocentric velocity   : {:.3f} km/s ({:+.3f} km/s{:s})\n".format( \
                frag_traj.orbit.v_g/1000, v_g_diff/1000, \
                sigmaStr(v_g_diff, valueSigma(traj, 'v_g'), valueSigma(frag_traj, 'v_g')))

        v_init_diff = frag_traj.v_init - traj.v_init
        out_str += "  Initial velocity      : {:.3f} km/s ({:+.3f} km/s{:s})\n".format( \
            frag_traj.v_init/1000, v_init_diff/1000, \
            sigmaStr(v_init_diff, valueSigma(traj, 'v_init'), valueSigma(frag_traj, 'v_init')))

        out_str += "  Begin / end height    : {:.2f} / {:.2f} km (main fragment: {:.2f} / {:.2f} " \
            "km)\n".format(frag_traj.rbeg_ele/1000, frag_traj.rend_ele/1000, traj.rbeg_ele/1000, \
            traj.rend_ele/1000)

        frag_jd, frag_eci = trajectoryModelPoints(frag_traj)

        # How far ahead of the main fragment, fitted over the noise of the individual points. The fit is on
        #   the difference between the two, which is much smoother than the motion of either
        frag_t = (frag_jd - main_jd[0])*86400
        ahead = np.dot(frag_eci - traj.state_vect_mini, direction) \
            - np.array([main_motion(t) for t in frag_t])
        ahead_poly = np.polyfit(frag_t, ahead, min(2, len(frag_t) - 1))

        off_first, off_last = offMainPath(traj, frag_traj)
        out_str += "  Off the main path     : {:.0f} m at the fragment's first point, {:.0f} m at its " \
            "last\n".format(off_first, off_last)

        # Warn about a fragment too far from the main path to have come from the main fragment
        if max(off_first, off_last) > FRAGMENT_OFF_PATH_LIMIT:
            out_str += "  NOTE: more than {:.1f} km off the main path, where no fragment of the main one " \
                "was in simulated\n        fireballs: another body, different fragments measured by the " \
                "stations, or a poorly\n        constrained trajectory?\n" \
                .format(FRAGMENT_OFF_PATH_LIMIT/1000)

        # Warn about stations which see the fragment at different places along its path at the same times, as
        #   when they measured different fragments under one number. Time offsets estimated for the fragment
        #   would absorb it
        apart = stationsApart(frag_traj) if (reuse_timing and (len(frag_traj.observations) > 1)) else {}
        if apart:

            pairs = ["{:s}-{:s} {:.0f} m".format(station_a, station_b, dist) \
                for (station_a, station_b), dist in sorted(apart.items())]
            out_str += "  Stations apart        : {:s}\n".format(",\n                          ".join( \
                ", ".join(pairs[i:i + 3]) for i in range(0, len(pairs), 3)))
            out_str += "                          along the path at the same times\n"

            if max(apart.values()) > FRAGMENT_TIMING_LIMIT:
                out_str += "  NOTE: more than {:.1f} km apart, which hardly any fragment of the main one " \
                    "was in simulated\n        fireballs: different fragments measured by the " \
                    "stations?\n".format(FRAGMENT_TIMING_LIMIT/1000)

        out_str += "  Ahead of the main one : {:+.0f} m at the fragment's first point, {:+.0f} m at its " \
            "last (at the same time)\n".format(np.polyval(ahead_poly, frag_t[0]), \
            np.polyval(ahead_poly, frag_t[-1]))

        # Where the main fragment was not seen its motion is extrapolated, which is poor if it was slowing
        #   down
        outside = max(frag_t[-1] - (main_jd[-1] - main_jd[0])*86400, -frag_t[0])
        if outside > 0.05:
            out_str += "  NOTE: the fragment is seen {:.2f} s outside the main one, for which that part is " \
                "extrapolated\n".format(outside)

        out_str += "  First seen            : {:+.3f} s from the main fragment\n".format( \
            (frag_traj.rbeg_jd - traj.rbeg_jd)*86400)

    return out_str



def loadFlares(ecsv_paths):
    """ Load the points of the main fragment flagged as flares in ECSV files, and the observations the main
        trajectory is solved from, which give the time offsets of their stations.

    Arguments:
        ecsv_paths: [list] List of paths to ECSV files.

    Return:
        (flare_meteors, input_meteors): [tuple] Lists of MeteorObservation objects from loadECSVs(), with
            flares=True and without.
    """

    _, flare_meteors = loadECSVs(ecsv_paths, no_prepare=True, flares=True)
    _, input_meteors = loadECSVs(ecsv_paths, no_prepare=True)

    return flare_meteors, input_meteors



def frameDuration(meteor):
    """ Duration of a frame of the camera of an observation, from the time between its points per frame
        between them if its file gives the frame numbers, so that it is the duration of one frame also when
        not every frame was picked, or else from the time between its points.

    Arguments:
        meteor: [MeteorObservation object] Observation from loadECSVs().

    Return:
        [float] Median duration of a frame (s).
    """

    time_steps = np.diff(meteor.time_data)

    # The time per frame between the points with increasing frame numbers
    frame_numbers = getattr(meteor, 'frame_numbers', None)
    if frame_numbers is not None:
        frame_steps = np.diff(frame_numbers)
        valid = np.isfinite(frame_steps) & (frame_steps > 0)
        if np.any(valid):
            return float(np.median(time_steps[valid]/frame_steps[valid]))

    return float(np.median(time_steps))



def projectFlares(traj, flare_meteors, input_meteors, noise_sigma=None, ang_res_std=None, time_diffs=None):
    """ Project the flare points on a solved trajectory as the solver does with its own points, with the
        gravity drop, at their times corrected by the time offset the trajectory applied to their station.

    Arguments:
        traj: [Trajectory] Solved trajectory of the main fragment, or one of its Monte Carlo runs.
        flare_meteors: [list] Flare points, from loadFlares().
        input_meteors: [list] Observations the trajectory was solved from, from loadFlares().

    Keyword arguments:
        noise_sigma: [float] If given, Gaussian noise is added to the lines of sight of the flares as in the
            Monte Carlo runs of the solver: this many times the angular residual of their station.
        ang_res_std: [dict] Angular residual (rad) of every station ID, used with noise_sigma.
        time_diffs: [dict] Time offsets (s) of station IDs to add to the applied ones, e.g. those a Monte Carlo
            run estimated, which it does not apply to the times of its observations.

    Return:
        [dict] For the key (ecsv_file, index) of every flare point of a station in the trajectory: its 
            station_id, jd (corrected time), ht (m above sea level, as the heights of the trajectory), ht_time
            (the height of the trajectory at the time of the point, from its own points, m), used (in the
            trajectory), frame_dt (duration of a frame of the station, s, see frameDuration()), and the ECI
            vectors eci (the point on the trajectory), stat (the station) and los (the line of sight), and the
            lat and lon of the point on the trajectory (radians). Also its ecsv_file and row, its position
            among the points of the main fragment in the file in time (see loadECSVs()), or None if not known.
    """

    offsets = appliedTimeOffsets(traj, input_meteors)
    station_ids = dict(zip([meteor.ecsv_file for meteor in input_meteors], solverStationIDs(input_meteors)))
    frame_dts = {meteor.ecsv_file: frameDuration(meteor) for meteor in input_meteors}

    flare_obs, obs_meteors = [], []
    for meteor in flare_meteors:

        if meteor.ecsv_file not in offsets:
            continue

        # Times from the reference time of the trajectory, with the time offset of the station
        t_rel = (meteor.jdt_ref - traj.jdt_ref)*86400 + np.array(meteor.time_data) + offsets[meteor.ecsv_file]
        if time_diffs is not None:
            t_rel += time_diffs.get(station_ids[meteor.ecsv_file], 0.0)

        # Precess from J2000 to the epoch of date, as the observations given to the solver
        ra, dec = equatorialCoordPrecession_vect(J2000_JD.days, np.zeros_like(meteor.ra_data) + traj.jdt_ref, 
            meteor.ra_data, meteor.dec_data)

        obs = ObservedPoints(traj.jdt_ref, ra, dec, t_rel, meteor.latitude, meteor.longitude, meteor.height, 1, 
            station_id=station_ids[meteor.ecsv_file])

        # Add noise to the lines of sight as the Monte Carlo runs of the solver do
        if noise_sigma is not None:

            sigma = noise_sigma*abs(ang_res_std.get(str(obs.station_id), np.nan))
            if not (sigma >= 0):
                sigma = np.radians(1)

            for i, rhat in enumerate(obs.meas_eci_los):
                rhat = vectNorm(rhat)
                uhat = vectNorm(np.cross(rhat, np.array([0.0, 0.0, 1.0])))
                vhat = vectNorm(np.cross(uhat, rhat))
                obs.meas_eci_los[i] = vectNorm(rhat + np.random.normal(0, sigma)*uhat 
                    + np.random.normal(0, sigma)*vhat)

        flare_obs.append(obs)
        obs_meteors.append(meteor)

    if not flare_obs:
        return {}

    # Project the flares together with the observations of the trajectory, so that the gravity drop has the 
    #   same reference, on a copy of the trajectory which is changed by the projection
    traj_copy = copy.copy(traj)
    observations = copy.deepcopy(traj.observations) + flare_obs
    traj_copy.calcECIEqAltAz(traj.state_vect_mini, traj.radiant_eci_mini, observations)
    traj_copy.calcLLA(traj.state_vect_mini, traj.radiant_eci_mini, observations)

    # Heights of the trajectory in time, from its own points on this solution, with the time offsets added to
    #   the times of the flares
    main_obs = [obs for obs in observations[:len(traj.observations)] if not obs.ignore_station]
    main_t = np.concatenate([(obs.JD_data[obs.ignore_list == 0] - traj.jdt_ref)*86400 \
        + (time_diffs or {}).get(str(obs.station_id), 0.0) for obs in main_obs])
    main_ht = np.concatenate([obs.model_ht[obs.ignore_list == 0] for obs in main_obs])
    heightAt = localMotionFit(main_t, main_ht)

    projected = {}
    for meteor, obs in zip(obs_meteors, flare_obs):
        for i in range(len(obs.time_data)):
            projected[(meteor.ecsv_file, i)] = {'station_id': str(obs.station_id), 'jd': obs.JD_data[i], 
                'ht': obs.model_ht[i], 'ht_time': heightAt((obs.JD_data[i] - traj.jdt_ref)*86400), 
                'used': bool(meteor.trajectory_use[i]), 
                'frame_dt': frame_dts[meteor.ecsv_file], 'eci': np.array(obs.model_eci[i]),
                'stat': np.array(obs.stat_eci_los[i]), 'los': np.array(obs.meas_eci_los[i]),
                'lat': obs.model_lat[i], 'lon': obs.model_lon[i]}

            # The position of the point among the points of the main fragment in its file
            rows = getattr(meteor, 'flare_rows', None)
            projected[(meteor.ecsv_file, i)].update(ecsv_file=meteor.ecsv_file, \
                row=(None if rows is None else int(rows[i])))

    return projected



def runTimeOffsets(mc_traj, traj):
    """ Get the time offsets a Monte Carlo run estimated, which it does not apply to the times of its
        observations, to add to those applied to the solution with the original picks (see projectFlares()).
        The solver adds that solution to its runs, with the offsets it estimated already applied to its times,
        so it has none to add.

    Arguments:
        mc_traj: [Trajectory object] One of the Monte Carlo runs.
        traj: [Trajectory object] The solution with the original picks (see originalPicksTrajectory()).

    Return:
        [dict] {station ID: time offset (s)}.
    """

    # The solution with the original picks itself, by its values, as it may have been loaded from its file
    same = np.array_equal(mc_traj.state_vect_mini, traj.state_vect_mini) \
        and np.array_equal(mc_traj.time_diffs_final, traj.time_diffs_final)

    return {str(obs.station_id): 0.0 if same else t_diff for obs, t_diff \
        in zip(mc_traj.observations, mc_traj.time_diffs_final)}



def groundSpeedFit(traj):
    """ Fit the speed of the meteor along a solved trajectory with the drag fit of wmpl.Utils.DragFit,
        relative to the ground (the air), for the dynamic pressure. The speeds of the fit are inertial, so the
        Earth's rotation along the path is taken off them, which offsets them all alike (orbit.v_init -
        orbit.v_init_norot, as Orbit takes it for v_avg_norot). In fireballs simulated with MetSim, the drag
        fit gave the dynamic pressure at the fragmentations within 0.7 % in 90 % of the cases, and over the
        last quarter of the flight without the bias of about 4 % of an alpha-beta fit.

    Arguments:
        traj: [Trajectory object] Solved trajectory.

    Return:
        (speedAt, speedStdAt):
            - speedAt: [function] The speed relative to the ground (m/s) at the given times from the reference
                time of the trajectory (s), NaN outside the drag fit.
            - speedStdAt: [function] The uncertainty of that speed from the covariance of the fit (m/s, see
                DragFit.motionStdAtTime()), from the noise of the points only.
            Both are None if the drag model could not be fitted.
    """

    try:
        drag_fit = DragFit(traj)

    except ValueError as e:
        print("The speeds and dynamic pressures of the flares are not computed: {:s}".format(str(e)))
        return None, None

    # The Earth's rotation along the path, 0 without the speed relative to the ground
    v_rotation = 0.0
    if (traj.orbit is not None) and (getattr(traj.orbit, 'v_init_norot', None) is not None):
        v_rotation = traj.orbit.v_init - traj.orbit.v_init_norot

    def speedAt(t_rel):
        return drag_fit.velocityAtTime(np.atleast_1d(t_rel)) - v_rotation

    def speedStdAt(t_rel):
        return drag_fit.motionStdAtTime(np.atleast_1d(t_rel))[0]

    return speedAt, speedStdAt



def flareHeights(traj, ecsv_paths, mc_trajs=None):
    """ Times and heights of the points of the main fragment flagged as flares in ECSV files, used in its
        trajectory or not (trajectory_use = False), on a solution of it (see projectFlares()), with their
        uncertainties from its Monte Carlo runs, if given. A point not used in the trajectory is only a time:
        where it was clicked is not a measurement (e.g. a saturated flare in which the meteor cannot be seen),
        so it takes the height of the trajectory at that time, not that of its line of sight.

    Arguments:
        traj: [Trajectory] Solved trajectory of the main fragment, the solution with the original picks (see
            originalPicksTrajectory()).
        ecsv_paths: [list] List of paths to ECSV files it was solved from.

    Keyword arguments:
        mc_trajs: [list] Its Monte Carlo runs. The flares are projected on each one, with the noise of the runs
            on their lines of sight and the time offsets each run estimated (see runTimeOffsets()), and the
            standard deviations of their times and heights are their uncertainties. The noise of the runs
            only moves the radiant, so the speed of the drag fit is shifted in every run within its
            uncertainty from the fit, by one draw per run for all the flare points, as
            Trajectory.sampleMCVelocities() does with the initial velocity. Where the fit leaves the speed
            unconstrained, it is not shifted.

    Return:
        [list] Dictionaries of the flare points sorted by time, with the station_id, jd, t_rel (s from the
            reference time of the trajectory), ht (m above sea level), used, frame_dt (s), ht_rate (rate of
            change of the height of the trajectory at that time, m/s), the uncertainties t_std (s) and ht_std
            (m), and their parts from the Monte Carlo runs only, t_std_mc and ht_std_mc. The uncertainties are
            None without Monte Carlo runs. Also the speed relative to the ground v (m/s, see
            groundSpeedFit()) and the dynamic pressure p = rho*v^2 (Pa) with the NRLMSISE air density at the
            point, with v_std, v_std_mc, p_std and p_std_mc as for the height, or None if the drag model could
            not be fitted.
    """

    flare_meteors, input_meteors = loadFlares(ecsv_paths)
    if not flare_meteors:
        return []

    projected = projectFlares(traj, flare_meteors, input_meteors)

    # Dynamic pressure from the speed of the drag fit of this solution, at the given times (s from the
    #   reference time of the trajectory), heights (m) and places (radians)
    speedAt, speedStdAt = groundSpeedFit(traj) if projected else (None, None)

    def pressureAt(t_rel, ht, lat, lon, speed_shift=0.0):
        """ Compute the dynamic pressure (Pa) with the speed shifted by speed_shift (m/s), NaN without the
            drag fit or outside it. """

        if speedAt is None:
            return np.nan

        rho = getAtmDensity_vect(np.atleast_1d(lat), np.atleast_1d(lon), np.atleast_1d(ht), traj.jdt_ref)

        return float(rho[0]*(speedAt(t_rel)[0] + speed_shift)**2)

    for meteor in flare_meteors:
        if (meteor.ecsv_file, 0) not in projected:
            print("The flares of station {:s} are skipped, as it is not in the main trajectory: {:s}".format(
                str(meteor.station_id), meteor.ecsv_file))

    # Scatter of the times, heights, dynamic pressures and speeds of the flares in the Monte Carlo runs
    mc_values = {key: [] for key in projected}
    if mc_trajs:

        # Uncertainty of the speed of every flare point from the drag fit, not shifted where it is not finite
        speed_std = {key: 0.0 for key in projected}
        if speedAt is not None:
            stds = speedStdAt(np.array([(flare['jd'] - traj.jdt_ref)*86400 for flare in projected.values()]))
            speed_std = {key: (float(std) if np.isfinite(std) else 0.0) for key, std in zip(projected, stds)}

        ang_res_std = {str(obs.station_id): obs.ang_res_std for obs in traj.observations}
        for mc_traj in mc_trajs:

            # Shift the speed of the run within the uncertainty of the drag fit, as its noise does not move it
            speed_draw = np.random.normal(0, 1)

            mc_projected = projectFlares(mc_traj, flare_meteors, input_meteors, noise_sigma=traj.mc_noise_std,
                ang_res_std=ang_res_std, time_diffs=runTimeOffsets(mc_traj, traj))

            for key, flare in mc_projected.items():
                if key in mc_values:
                    ht = flare['ht'] if flare['used'] else flare['ht_time']
                    t_rel = (flare['jd'] - traj.jdt_ref)*86400
                    speed_shift = speed_draw*speed_std[key]
                    speed = (float(speedAt(t_rel)[0]) + speed_shift) if (speedAt is not None) else np.nan
                    mc_values[key].append(((flare['jd'] - projected[key]['jd'])*86400, ht, \
                        pressureAt(t_rel, ht, flare['lat'], flare['lon'], speed_shift), speed))

    # Heights of the trajectory in time, to compare flares seen at different times
    used_obs = [obs for obs in traj.observations if not obs.ignore_station]
    traj_t = np.concatenate([obs.time_data[obs.ignore_list == 0] for obs in used_obs])
    traj_ht = np.concatenate([obs.model_ht[obs.ignore_list == 0] for obs in used_obs])

    flares = []
    for key, flare in projected.items():

        flare['t_rel'] = (flare['jd'] - traj.jdt_ref)*86400
        if not flare['used']:
            flare['ht'] = flare['ht_time']

        # Local rate of change of the height, from the points of the trajectory within 0.3 s
        near = np.abs(traj_t - flare['t_rel']) <= 0.3
        if np.sum(near) < 3:
            near = np.ones_like(traj_t, dtype=bool)
        flare['ht_rate'] = np.polyfit(traj_t[near], traj_ht[near], 1)[0]
        flare['t_std'] = flare['ht_std'] = flare['t_std_mc'] = flare['ht_std_mc'] = None

        # Speed and dynamic pressure, and how fast the pressure changes along the trajectory
        flare['v'] = flare['p'] = flare['p_std'] = flare['p_std_mc'] = None
        flare['v_std'] = flare['v_std_mc'] = None
        if speedAt is not None:
            flare['v'] = float(speedAt(flare['t_rel'])[0])
            flare['p'] = pressureAt(flare['t_rel'], flare['ht'], flare['lat'], flare['lon'])
            p_rate = (pressureAt(flare['t_rel'] + 0.01, flare['ht'] + 0.01*flare['ht_rate'], flare['lat'], \
                flare['lon']) - pressureAt(flare['t_rel'] - 0.01, flare['ht'] - 0.01*flare['ht_rate'], \
                flare['lat'], flare['lon']))/0.02
            v_rate = (speedAt(flare['t_rel'] + 0.01)[0] - speedAt(flare['t_rel'] - 0.01)[0])/0.02

        if len(mc_values[key]) > 1:
            values = np.array(mc_values[key])
            flare['t_std_mc'], flare['ht_std_mc'] = np.std(values[:, :2], axis=0)

            # The flare happened at any time within its frame, so its time is only known to within the
            #   duration of the frame (a uniform distribution), and the height of the meteor at that time to 
            #   within how much it changes in that time
            frame_std = flare['frame_dt']/np.sqrt(12)
            flare['t_std'] = np.hypot(flare['t_std_mc'], frame_std)
            flare['ht_std'] = np.hypot(flare['ht_std_mc'], flare['ht_rate']*frame_std)

            # The same for the dynamic pressure, from the runs in which it could be computed
            pressures = values[:, 2][np.isfinite(values[:, 2])]
            if (flare['p'] is not None) and np.isfinite(flare['p']) and (len(pressures) > 1):
                flare['p_std_mc'] = np.std(pressures)
                flare['p_std'] = np.hypot(flare['p_std_mc'], p_rate*frame_std) if np.isfinite(p_rate) \
                    else flare['p_std_mc']

            # The same for the speed
            speeds = values[:, 3][np.isfinite(values[:, 3])]
            if (flare['v'] is not None) and np.isfinite(flare['v']) and (len(speeds) > 1):
                flare['v_std_mc'] = np.std(speeds)
                flare['v_std'] = np.hypot(flare['v_std_mc'], v_rate*frame_std) if np.isfinite(v_rate) \
                    else flare['v_std_mc']

        if (flare['p'] is not None) and (not np.isfinite(flare['p'])):
            flare['v'] = flare['p'] = flare['v_std'] = flare['v_std_mc'] = None

        flares.append(flare)

    return sorted(flares, key=lambda flare: flare['jd'])


def timeReferenceStations(traj, mc_trajs=None):
    """ Stations whose time offset a solution does not estimate, so that the times of all stations refer to
        their clocks: the reference station (by default the one with the longest track, whose offset is fixed
        to 0), or the stations with fixed time offsets.

    Arguments:
        traj: [Trajectory] Solved trajectory.

    Keyword arguments:
        mc_trajs: [list] Its Monte Carlo runs, which estimate the time offsets even if the solution does not,
            and give the uncertainties of the times.

    Return:
        [list] Station IDs, empty if the time offsets were not estimated.
    """

    timing_traj = mc_trajs[0] if mc_trajs else traj
    stations_time_dict = getattr(timing_traj, 'stations_time_dict', None)
    if (not timing_traj.estimate_timing_vel) or (not stations_time_dict):
        return []

    return [str(station_id) for station_id, status in stations_time_dict.items() 
        if not isinstance(status, bool)]



def timeReferenceStr(ref_stations):
    """ Say which clocks the times of a report refer to.

    Arguments:
        ref_stations: [list] Stations whose clocks the times refer to, from timeReferenceStations(), or None.

    Return:
        [str] The sentence, empty if ref_stations is None.
    """

    if ref_stations is None:
        return ""

    if len(ref_stations) == 1:
        return "Times relative to the clock of the reference station {:s}, whose time offset is not\n" \
            "estimated: their absolute accuracy depends on how well it was synchronised (e.g. NTP, GPS).\n" \
            .format(ref_stations[0])

    if ref_stations:
        return "Times relative to the clocks of the stations with fixed time offsets, {:s}: their\n" \
            "absolute accuracy depends on how well those clocks were synchronised (e.g. NTP, GPS).\n" \
            .format(", ".join(ref_stations))

    return "The time offsets of the stations were not estimated: times from the clock of each station.\n"



# Real error of the time offset the solver estimates between two stations (s). Its fit and Monte Carlo runs
#   give it about 30 times too small (0.24 ms in the median). In fireballs simulated with MetSim and seen by
#   25 fps cameras, 68 % of the errors were below 7 ms, with a median of 4.4 ms and a 99th percentile of 28 ms
FLARE_TIME_OFFSET_ERROR = 0.007



def flareReport(flares, match_window=1.0, ref_stations=None):
    """ Report the flare points from flareHeights(): the flares of every station, the points, and how the
        flares of different stations compare, as they need not be the same: each one against the closest one
        in time of every other station within match_window. The flares of a station are its flare points in
        consecutive rows of its file, among the points of the main fragment, so a point which is not flagged
        separates two flares, whatever the frame rate of the camera and how often the points were picked.
        Points whose rows are not known are taken as one flare if they are at most 1.5 frames apart.

    Arguments:
        flares: [list] Flare points from flareHeights().

    Keyword arguments:
        match_window: [float] Largest time between the flares of two stations compared (s), 1 by default.
        ref_stations: [list] Stations whose clocks the times refer to, from timeReferenceStations(), to say
            so in the report. When the solver estimated the time offsets of the others, the comparison of
            the flares of two stations adds the real error of those offsets (FLARE_TIME_OFFSET_ERROR) to the
            uncertainty of their times, except between two of these stations. None by default, to leave it
            out.

    Return:
        [str] The report.
    """

    def valueStr(value, std, fmt):
        return (fmt + " +/- " + fmt).format(value, std) if (std is not None) else fmt.format(value)

    def timeStr(jd):
        return jd2Date(jd, dt_obj=True).strftime("%H:%M:%S.%f")[:-3]

    def motionStr(flare):
        """ Format the speed and the dynamic pressure of a flare point, if known. """

        if flare.get('p') is None:
            return ""

        return ", v = {:s} km/s, p = {:s} kPa".format(speedStr(flare), pressureStr(flare))

    def speedStr(flare):
        """ Format the speed of a flare point in km/s, empty if not known. """

        if flare.get('v') is None:
            return ""

        return valueStr(flare['v']/1000, None if flare.get('v_std') is None else flare['v_std']/1000, \
            "{:.3f}")

    def pressureStr(flare):
        """ Format the dynamic pressure of a flare point in kPa, empty if not known. """

        if flare.get('p') is None:
            return ""

        return valueStr(flare['p']/1000, None if flare.get('p_std') is None else flare['p_std']/1000, \
            "{:.1f}")

    def sameFlare(prev, flare):
        """ Check whether a flare point is in the row after the previous flare point of its file, or without
            the rows, at most 1.5 frames after it. """

        if (prev.get('row') is None) or (flare.get('row') is None):
            return (flare['jd'] - prev['jd'])*86400 <= 1.5*flare['frame_dt']

        return (flare.get('ecsv_file') == prev.get('ecsv_file')) and (flare['row'] == prev['row'] + 1)

    # Flares of every station, from the flare points of each of its files in consecutive rows
    events = []
    for station_id in sorted(set(flare['station_id'] for flare in flares)):

        station_flares = [flare for flare in flares if flare['station_id'] == station_id]

        station_events = []
        for ecsv_file in list(dict.fromkeys(flare.get('ecsv_file') for flare in station_flares)):

            file_events = []
            for flare in [flare for flare in station_flares if flare.get('ecsv_file') == ecsv_file]:
                if file_events and sameFlare(file_events[-1][-1], flare):
                    file_events[-1].append(flare)
                else:
                    file_events.append([flare])

            station_events += file_events

        events += sorted(station_events, key=lambda event: event[0]['jd'])

    uncertain = any(flare['ht_std'] is not None for flare in flares)

    out_str = "\n"
    out_str += "Flares\n"
    out_str += "------\n"
    out_str += "Points flagged as flares, used in the trajectory or not, at the times of their frames\n"
    out_str += "corrected by the time offsets of their stations. Heights above sea level. A point used in\n"
    out_str += "the trajectory is projected on it. A point not used (trajectory_use = False) is only a\n"
    out_str += "time, as where it was clicked is not a measurement (e.g. a saturated flare in which the\n"
    out_str += "meteor cannot be seen): it takes the height of the trajectory at that time.\n"
    out_str += "The flares of a station are its flare points in consecutive rows of its file: a point not\n"
    out_str += "flagged as a flare between two flagged ones separates two flares.\n"
    out_str += timeReferenceStr(ref_stations)
    if uncertain:
        out_str += "Uncertainties (1 sigma) from projecting them on the Monte Carlo runs, and from the duration\n"
        out_str += "of the frame the flare was seen in, at any time within it (in time, and in height as the\n"
        out_str += "meteor descends during the frame).\n"
    else:
        out_str += "Without the Monte Carlo runs the times and heights have no uncertainties.\n"
    if any(flare.get('p') is not None for flare in flares):
        out_str += "Speeds v relative to the ground from a drag fit of the trajectory (wmpl.Utils.DragFit),\n"
        out_str += "and dynamic pressures p = rho*v^2 with the NRLMSISE air density, whose own error is not\n"
        out_str += "included.\n"
        if uncertain:
            out_str += "The uncertainty of the speed from the noise of the points, from the covariance\n"
            out_str += "of the drag fit, is drawn in every Monte Carlo run. The bias of the fit where the\n"
            out_str += "meteor fragments is not included.\n"
    out_str += "\n"

    out_str += "Flares of every station (t from the reference time of the trajectory):\n"
    for i, event in enumerate(events):
        used = sum(flare['used'] for flare in event)
        out_str += "{:2d}  {:12s} {:s}-{:s} UTC, {:d} point(s), {:s} used in the trajectory\n".format(i + 1, 
            event[0]['station_id'], timeStr(event[0]['jd']), timeStr(event[-1]['jd']), len(event), 
            "all" if used == len(event) else ("none" if used == 0 else "{:d}".format(used)))
        out_str += "      Begin: t = {:s} s, ht = {:s} km{:s}\n".format(valueStr(event[0]['t_rel'], 
            event[0]['t_std'], "{:.4f}"), valueStr(event[0]['ht']/1000, 
            None if event[0]['ht_std'] is None else event[0]['ht_std']/1000, "{:.3f}"), motionStr(event[0]))
        if len(event) > 1:
            out_str += "      End:   t = {:s} s, ht = {:s} km{:s}\n".format(valueStr(event[-1]['t_rel'], 
                event[-1]['t_std'], "{:.4f}"), valueStr(event[-1]['ht']/1000, 
                None if event[-1]['ht_std'] is None else event[-1]['ht_std']/1000, "{:.3f}"), \
                motionStr(event[-1]))

    out_str += "\n"
    # The speed and the dynamic pressure of every point, if any is known
    motion = any(flare.get('p') is not None for flare in flares)

    out_str += "Flare points:\n"
    out_str += "Flare  Station       Time (UTC)      t (s)                  Ht (km)               " \
        + ("{:18s}  {:20s}  ".format("v (km/s)", "p (kPa)") if motion else "") + "Used\n"
    for i, event in enumerate(events):
        for flare in event:
            motion_str = "{:18s}  {:20s}  ".format(speedStr(flare), pressureStr(flare)) if motion else ""
            out_str += "{:5d}  {:12s}  {:s}    {:21s}  {:20s}  {:s}{:s}\n".format(i + 1, flare['station_id'],
                timeStr(flare['jd']), valueStr(flare['t_rel'], flare['t_std'], "{:.4f}"),
                valueStr(flare['ht']/1000, None if flare['ht_std'] is None else flare['ht_std']/1000, \
                "{:.3f}"), motion_str, "yes" if flare['used'] else "no")

    # Compare the flares of different stations, which need not be the same: e.g. a fireball can show two flares
    #   which one camera sees at other times than another
    stations = sorted(set(event[0]['station_id'] for event in events))
    if len(stations) > 1:

        def compare(a, b):
            """ Time between two flares (0 if their times overlap), and the difference of their heights at the
                same time, between their closest points in time, with these points. """

            gap = max(0.0, b[0]['t_rel'] - a[-1]['t_rel'], a[0]['t_rel'] - b[-1]['t_rel'])
            pa, pb = min(((fa, fb) for fa in a for fb in b), key=lambda x: abs(x[1]['t_rel'] - x[0]['t_rel']))

            # Height of b at the time of a, along the trajectory
            dht = pb['ht'] - pa['ht'] - pa['ht_rate']*(pb['t_rel'] - pa['t_rel'])

            return gap, dht, pa, pb

        out_str += "\n"
        out_str += "Flares of different stations compared, each one with the closest one in time of every other\n"
        out_str += "station: the time between them (none if their times overlap; the time uncertainty includes a\n"
        out_str += "frame of each camera) and the difference of their heights at the same time. Between\n"
        out_str += "points not used in the trajectory, whose heights are those of the trajectory, only the\n"
        out_str += "times count. Two cameras can mark the same flare a frame apart, e.g. as it lasts longer\n"
        out_str += "than a frame, so one frame of the slower camera is taken off the time between them\n"
        out_str += "before it is weighed against its uncertainty.\n"
        if ref_stations and any(station_id not in ref_stations for station_id in stations):
            out_str += "The uncertainty of the time between two stations also includes the real error of\n"
            out_str += "the time offsets the solver estimated, {:.0f} ms, measured in simulations, far\n" \
                .format(1000*FLARE_TIME_OFFSET_ERROR)
            out_str += "above the uncertainty of their fit.\n"

        compared = set()
        for i, event in enumerate(events):
            for station_id in stations:

                if station_id == event[0]['station_id']:
                    continue

                others = [j for j, other in enumerate(events) if other[0]['station_id'] == station_id]
                j = min(others, key=lambda j: (compare(event, events[j])[0], 
                    abs(compare(event, events[j])[3]['t_rel'] - compare(event, events[j])[2]['t_rel'])))
                gap, dht, pa, pb = compare(event, events[j])

                if gap > match_window:
                    out_str += "{:2d} {:s}: no flare of {:s} within {:.1f} s\n".format(i + 1, event[0]['station_id'], 
                        station_id, match_window)
                    continue

                if (min(i, j), max(i, j)) in compared:
                    continue
                compared.add((min(i, j), max(i, j)))

                line = "{:2d} {:s} - {:2d} {:s}: {:s}, heights {:+.3f} km apart".format(i + 1, 
                    event[0]['station_id'], j + 1, station_id, 
                    "times overlap" if gap == 0 else "{:.3f} s apart".format(gap), dht/1000)

                if (pa['t_std'] is not None) and (pb['t_std'] is not None):

                    # The time of each flare is known to within its frame, but the heights are compared at the 
                    #   same time along the trajectory, which does not depend on when the flares happened within 
                    #   their frames, only on the Monte Carlo uncertainties
                    # The time offsets the solver estimated err far more than the uncertainties of its fit,
                    #   which moves the time between two stations, and the height of one at the time of the
                    #   other
                    offset_error = 0.0
                    if ref_stations and not ((pa['station_id'] in ref_stations) \
                        and (pb['station_id'] in ref_stations)):
                        offset_error = FLARE_TIME_OFFSET_ERROR

                    t_sig = np.sqrt(pa['t_std']**2 + pb['t_std']**2 + offset_error**2)
                    ht_sig = np.sqrt(pa['ht_std_mc']**2 + pb['ht_std_mc']**2 
                        + (pa['ht_rate']**2)*(pa['t_std_mc']**2 + pb['t_std_mc']**2 + offset_error**2))

                    # Two cameras can mark frames of the same flare up to a frame of the slower one apart. In
                    #   fireballs simulated with MetSim, 4.9 % of the same flares of a 25 and a 13 fps camera
                    #   were further apart than 3 sigma of their frames, and 2.7 % after taking off that frame
                    t_dist = max(gap - max(pa['frame_dt'], pb['frame_dt']), 0.0)

                    consistent = (t_dist <= 3*t_sig) and (abs(dht) <= 3*ht_sig)
                    line += " ({:.1f} and {:.1f} sigma): {:s}".format(t_dist/t_sig, abs(dht)/ht_sig, 
                        "consistent" if consistent else "NOT consistent, different flares?")

                out_str += line + "\n"

    return out_str



def dragFitFlareWarning(traj, flares):
    """ Warn when flare points lie within the part of the trajectory that the drag fit of the initial velocity
        (--vinitdrag) used, at or above its lowest point, as that fit models a single body without
        fragmentation.

    Arguments:
        traj: [Trajectory object] Solved trajectory the flares were projected on.
        flares: [list] Flare points from flareHeights().

    Return:
        [str] The warning, empty if no drag fit was used or no flare lies within it.
    """

    # The drag fit is only there when the solver has the --vinitdrag option and its fit was used
    fit = getattr(traj, 'v_init_drag_fit', None)
    if (fit is None) or (getattr(fit, 'ht_range', None) is None):
        return ""

    ht_lo, ht_hi = fit.ht_range

    # A flare point that is itself the lowest fitted point is within the fit, 1 m for the rounding
    inside = [flare for flare in flares if flare['ht'] >= ht_lo - 1.0]
    if not inside:
        return ""

    # Highest flare point of every station within the fit
    highest = {}
    for flare in inside:
        station_id = str(flare['station_id'])
        if (station_id not in highest) or (flare['ht'] > highest[station_id]['ht']):
            highest[station_id] = flare

    points_str = ", ".join(["{:s} at {:.2f} km".format(station_id, flare['ht']/1000) \
        for station_id, flare in sorted(highest.items(), key=lambda item: -item[1]['ht'])])

    out_str  = "\n"
    out_str += "WARNING: {:d} flare point(s) at or above the lowest point of the --vinitdrag fit\n" \
        .format(len(inside))
    out_str += "of the initial velocity ({:.2f} km), which models a single body without\n".format(ht_lo/1000)
    out_str += "fragmentation. Highest per station: {:s}.\n".format(points_str)

    # The fit should end above the highest flare, if any fitted point is left above it
    ht_flare = max(flare['ht'] for flare in inside)
    if ht_flare < ht_hi:
        out_str += "Fit only the points above it, e.g. with --vinitdraght {:.1f}.\n".format( \
            np.ceil(ht_flare/100)/10)
    else:
        out_str += "It is at or above the first fitted point ({:.2f} km), so no fitted point is free\n" \
            "of it.\n".format(ht_hi/1000)

    return out_str



def oneStationSolution(traj, meteor, input_meteors, time_diffs=None, **noise_kwargs):
    """ Measure the points of a fragment seen from one station against a solution of the main trajectory, on
        which they are projected as the flares are (see projectFlares()).

    Arguments:
        traj: [Trajectory object] Solved trajectory of the main fragment, or one of its Monte Carlo runs.
        meteor: [MeteorObservation object] Points of the fragment from one file, from loadECSVs().
        input_meteors: [list] Observations the trajectory was solved from.

    Keyword arguments:
        time_diffs: [dict] Time offsets (s) of station IDs to add to the applied ones, see projectFlares().
            None by default.
        noise_kwargs: noise_sigma and ang_res_std of projectFlares().

    Return:
        [dict] None if the station is not in the trajectory. Otherwise, for every point sorted by time, in
            arrays: jd, eci (the point on the path), ht (m), off (distance between its line of sight and
            the path, m, signed), ahead (distance along the path from the main fragment at the same time, m),
            and ht_per_m and ahead_per_m (error of its height and of ahead per metre the fragment is off the
            path, in the plane of the path and the line of sight). Also the first and last values of straight
            and quadratic fits over the noise of the points, off_first, off_last, off_rate (m/s), ahead_first
            and ahead_last, and off_noise, the scatter of the points about the fit (m), los_noise, the same as
            an angle (rad), outside (the time the fragment is seen outside the main fragment, s) and
            station_id.
    """

    projected = projectFlares(traj, [meteor], input_meteors, time_diffs=time_diffs, **noise_kwargs)
    if not projected:
        return None

    points = [projected[(meteor.ecsv_file, i)] for i in range(len(meteor.time_data))]
    jd = np.array([point['jd'] for point in points])
    eci = np.array([point['eci'] for point in points])
    stat = np.array([point['stat'] for point in points])
    los = np.array([point['los'] for point in points])
    t = (jd - jd[0])*86400

    direction = -vectNorm(traj.radiant_eci_mini)

    # Distance between the line of sight and the path along the normal to both, all one station sees of it
    normal = np.cross(los, direction)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    off = np.sum((eci - stat)*normal, axis=1)
    off_poly = np.polyfit(t, off, 1)
    off_residuals = off - np.polyval(off_poly, t)

    # Distance along the path of the main fragment at the time of every point, as in this solution
    used = [obs for obs in traj.observations if not obs.ignore_station]
    main_t = np.concatenate([(obs.JD_data[obs.ignore_list == 0] - jd[0])*86400 \
        + (time_diffs or {}).get(str(obs.station_id), 0.0) for obs in used])
    main_along = np.dot(np.concatenate([obs.model_eci[obs.ignore_list == 0] for obs in used]) \
        - traj.state_vect_mini, direction)

    main_motion = localMotionFit(main_t, main_along)

    # Fit the difference with the main fragment, which is much smoother than the motion of either
    ahead = np.dot(eci - traj.state_vect_mini, direction) - np.array([main_motion(time) for time in t])
    ahead_fit = np.polyval(np.polyfit(t, ahead, min(2, len(t) - 1)), t)

    # A fragment off the path in the plane of the path and the line of sight is distance/sin(angle) from
    #   its projection along the line of sight, which moves its height and how far ahead it is
    los_path_angle = np.arccos(np.clip(np.abs(np.dot(los, direction)), 0, 1))
    sin_altitude = np.abs(np.sum(los*eci, axis=1))/np.linalg.norm(eci, axis=1)

    return {'station_id': points[0]['station_id'], 'jd': jd, 'eci': eci, \
        'ht': np.array([point['ht'] for point in points]), 'off': off, 'ahead': ahead, \
        'ht_per_m': sin_altitude/np.sin(los_path_angle), 'ahead_per_m': 1/np.tan(los_path_angle), \
        'off_first': np.polyval(off_poly, t[0]), 'off_last': np.polyval(off_poly, t[-1]), \
        'off_rate': off_poly[0], 'ahead_first': ahead_fit[0], 'ahead_last': ahead_fit[-1], \
        'off_noise': np.std(off_residuals, ddof=2), \
        'outside': max(t[-1] - np.max(main_t), np.min(main_t) - t[0]), \
        'los_noise': np.std(off_residuals/np.linalg.norm(eci - stat, axis=1), ddof=2)}



def poorFragmentTrajectories(traj, fragment_trajs):
    """ Find the solved fragments whose trajectory says little, to project their points on the main
        trajectory too (see unsolvedFragments()): those whose stations converge by less than the trajectory
        correlator of WMPL takes, and those further off the main path than fragments of the main one go (see
        fragmentComparison()).

    Arguments:
        traj: [Trajectory object] Solved trajectory of the main fragment, the solution with the original
            picks (see originalPicksTrajectory()).
        fragment_trajs: [list] (fragment, trajectory) pairs, from solveFragmentTrajectories().

    Return:
        [dict] {fragment: why}, with why saying it in a few words.
    """

    poor = {}
    for fragment, frag_traj in fragment_trajs:

        conv_angle = convergenceAngle(frag_traj)
        off = max(offMainPath(traj, frag_traj))

        if (conv_angle is not None) and (conv_angle < FRAGMENT_MIN_CONVERGENCE):
            poor[fragment] = "its stations converge by only {:.1f} deg".format(conv_angle)

        elif off > FRAGMENT_OFF_PATH_LIMIT:
            poor[fragment] = "its trajectory is {:.1f} km off the main path".format(off/1000)

    return poor



def unsolvedFragments(traj, ecsv_paths, fragments, mc_trajs=None, why=None):
    """ Project on the main trajectory the points of additional fragments in ECSV files which have no
        trajectory of their own, each station on its own (see oneStationSolution()): those seen from one
        station, and those whose trajectory could not be solved, or says little (see
        poorFragmentTrajectories()). This assumes that the fragment is on the main path, which one station
        cannot test.

    Arguments:
        traj: [Trajectory object] Solved trajectory of the main fragment, the solution with the original
            picks (see originalPicksTrajectory()).
        ecsv_paths: [list] List of paths to ECSV files it was solved from.
        fragments: [list] Numbers of the fragments to project.

    Keyword arguments:
        mc_trajs: [list] Its Monte Carlo runs. The points are projected on each one, with the noise of the
            runs on their lines of sight (the scatter of the points of the fragment, as they are not the
            points of the main fragment) and the time offsets each run estimated (see runTimeOffsets()). The
            standard deviations are the uncertainties, which do not include a separation from the main path.
            None by default.
        why: [dict] Why a fragment is projected, {fragment: why}, e.g. from poorFragmentTrajectories().
            For the others it is that it was seen from one station, or that its trajectory could not be
            solved. None by default.

    Return:
        [list] A dictionary for every file of every fragment: the values of oneStationSolution(), with also
            fragment, n_stations (distinct stations of the fragment), why, n_points, t_rel (s from the
            reference time of the trajectory), and in std a dictionary of the standard deviations in the Monte
            Carlo
            runs of the time of every point (t, s), of its height (ht), and of off_first, off_last, off_rate,
            ahead_first and ahead_last. std is None without runs.
    """

    _, input_meteors = loadECSVs(ecsv_paths, no_prepare=True)

    solutions = []
    for fragment in fragments:

        _, meteors = loadECSVs(ecsv_paths, no_prepare=True, fragment=fragment)
        n_stations = len(set(str(meteor.station_id) for meteor in meteors))

        for meteor in meteors:

            meteor.trajectory_use = np.ones(len(meteor.time_data), dtype=bool)

            solution = oneStationSolution(traj, meteor, input_meteors)
            if solution is None:
                print("Fragment {:d} from {:s} is not projected, as the station is not in the main " \
                    "trajectory: {:s}".format(fragment, str(meteor.station_id), meteor.ecsv_file))
                continue

            solution.update(fragment=fragment, n_stations=n_stations, n_points=len(meteor.time_data), \
                std=None, t_rel=(solution['jd'] - traj.jdt_ref)*86400)
            solution['why'] = (why or {}).get(fragment, "seen from one station" if (n_stations == 1) \
                else "its trajectory could not be solved")

            # Project the points on every Monte Carlo run of the main trajectory
            mc_solutions = []
            for mc_traj in (mc_trajs or []):

                mc_solution = oneStationSolution(mc_traj, meteor, input_meteors, \
                    noise_sigma=traj.mc_noise_std, \
                    ang_res_std={solution['station_id']: solution['los_noise']}, \
                    time_diffs=runTimeOffsets(mc_traj, traj))

                if mc_solution is not None:
                    mc_solutions.append(mc_solution)

            if len(mc_solutions) > 1:

                solution['std'] = {'t': np.std([(mc['jd'] - solution['jd'])*86400 for mc in mc_solutions], \
                    axis=0), 'ht': np.std([mc['ht'] for mc in mc_solutions], axis=0)}
                for name in ('off_first', 'off_last', 'off_rate', 'ahead_first', 'ahead_last'):
                    solution['std'][name] = np.std([mc[name] for mc in mc_solutions])

            solutions.append(solution)

    return solutions



def unsolvedFragmentReport(solutions, traj, ref_stations=None):
    """ Report the projections of the fragments without a trajectory, from unsolvedFragments().

    Arguments:
        solutions: [list] Projections from unsolvedFragments().
        traj: [Trajectory object] Solved trajectory of the main fragment they were projected on.

    Keyword arguments:
        ref_stations: [list] Stations whose clocks the times refer to, from timeReferenceStations(), to say
            so in the report. None by default, to leave it out.

    Return:
        [str] The report.
    """

    def valueStr(value, std, fmt, scale=1):
        """ Format a value with its standard deviation, if known. """

        if std is None:
            return fmt.format(value/scale)

        return (fmt + " +/- " + fmt.replace("+", "")).format(value/scale, std/scale)


    def stdOf(frag, name, i=None):
        """ Get a standard deviation of a projection, None without the Monte Carlo runs. """

        if frag['std'] is None:
            return None

        return frag['std'][name] if (i is None) else frag['std'][name][i]


    out_str = "\n"
    out_str += "Additional fragments projected on the main trajectory\n"
    out_str += "-----------------------------------------------------\n"
    out_str += "Fragments seen from one station, whose trajectory could not be solved, or whose\n"
    out_str += "trajectory says little (stations converging by less than {:.0f} deg, or more than\n" \
        .format(FRAGMENT_MIN_CONVERGENCE)
    out_str += "{:.1f} km off the main path), have their points projected on the main trajectory, each\n" \
        .format(FRAGMENT_OFF_PATH_LIMIT/1000)
    out_str += "station on its own, at the times of their frames corrected by the time offset of the\n"
    out_str += "station. Heights above sea level. This assumes that the fragment is on the main path,\n"
    out_str += "which one station cannot test: only the part of a separation across the line of sight is\n"
    out_str += "seen, so the distances off the path and the speeds are lower limits, and the rest moves the\n"
    out_str += "heights, as given below. The heights are those of the main path where the line of sight\n"
    out_str += "passes closest to it.\n"
    out_str += timeReferenceStr(ref_stations)
    if any(frag['std'] is not None for frag in solutions):
        out_str += "Uncertainties (1 sigma) from projecting them on the Monte Carlo runs, with the\n"
        out_str += "scatter of the points of each fragment as their noise. They do not include a\n"
        out_str += "separation from the main path.\n"
    else:
        out_str += "Without the Monte Carlo runs the values have no uncertainties.\n"

    for frag in solutions:

        sign = np.sign(frag['off_last'])
        why = frag.get('why', "seen from one station" if (frag['n_stations'] == 1) \
            else "its trajectory could not be solved")

        out_str += "\n"
        out_str += "Fragment {:d}, from {:s} ({:s})\n".format(frag['fragment'], frag['station_id'], why)
        out_str += "  Points                : {:d}\n".format(frag['n_points'])
        out_str += "  First seen            : {:+.3f} s from the main fragment\n".format( \
            (frag['jd'][0] - traj.rbeg_jd)*86400)
        out_str += "  Begin / end height    : {:s} / {:s} km (main fragment: {:.2f} / {:.2f} km)\n".format( \
            valueStr(frag['ht'][0], stdOf(frag, 'ht', 0), "{:.2f}", 1000), \
            valueStr(frag['ht'][-1], stdOf(frag, 'ht', -1), "{:.2f}", 1000), traj.rbeg_ele/1000, \
            traj.rend_ele/1000)
        out_str += "  Off the main path     : {:s} m at the fragment's first point, {:s} m at its last, " \
            "across the\n                          line of sight only (scatter of the points: {:.0f} " \
            "m)\n".format(valueStr(abs(frag['off_first']), stdOf(frag, 'off_first'), "{:.0f}"), \
            valueStr(abs(frag['off_last']), stdOf(frag, 'off_last'), "{:.0f}"), frag['off_noise'])
        out_str += "  Moving off the path   : {:s} m/s, across the line of sight only\n".format( \
            valueStr(sign*frag['off_rate'], stdOf(frag, 'off_rate'), "{:+.0f}"))
        out_str += "  Ahead of the main one : {:s} m at the fragment's first point, {:s} m at its last " \
            "(at the same\n                          time), as if on the main path\n".format( \
            valueStr(frag['ahead_first'], stdOf(frag, 'ahead_first'), "{:+.0f}"), \
            valueStr(frag['ahead_last'], stdOf(frag, 'ahead_last'), "{:+.0f}"))

        if frag['outside'] > 0.05:
            out_str += "  NOTE: the fragment is seen {:.2f} s outside the main one, for which that part is " \
                "extrapolated\n".format(frag['outside'])

        out_str += "  If 100 m off the path : in the plane of the path and the line of sight, which is not " \
            "seen, the\n                          heights are off by {:.0f} m at the fragment's first " \
            "point and {:.0f} m at its\n                          last, and how far ahead it is by {:.0f} " \
            "and {:.0f} m\n".format(100*frag['ht_per_m'][0], 100*frag['ht_per_m'][-1], \
            100*abs(frag['ahead_per_m'][0]), 100*abs(frag['ahead_per_m'][-1]))

        # Every point of the fragment
        out_str += "\n"
        out_str += "  Time (UTC)      t (s)                  Ht (km)               Off (m)  Ahead (m)\n"
        for i in range(frag['n_points']):
            out_str += "  {:s}    {:21s}  {:20s}  {:7.0f}  {:+9.0f}\n".format( \
                jd2Date(frag['jd'][i], dt_obj=True).strftime("%H:%M:%S.%f")[:-3], \
                valueStr(frag['t_rel'][i], stdOf(frag, 't', i), "{:.4f}"), \
                valueStr(frag['ht'][i], stdOf(frag, 'ht', i), "{:.3f}", 1000), abs(frag['off'][i]), \
                frag['ahead'][i])

    return out_str



# Largest distance along the path between the projections of two stations for their fragments to be matched
#   (m), which 0.2 % of the fragments seen from two stations exceeded in fireballs simulated with MetSim
FRAGMENT_MATCH_LIMIT = 1500.0

# Least distance by which a match must be closer than any other candidate of either fragment not to be
#   ambiguous (m)
FRAGMENT_MATCH_MARGIN = 500.0

# Least time two stations must both see a pair of fragments for to compare them (s)
FRAGMENT_MATCH_OVERLAP = 0.08



def fragmentMatches(traj, ecsv_paths, fragments):
    """ Match the additional fragments seen from every two stations by where they are along the main path at
        the same times, whatever their numbers in the ECSV files. The points of every station are projected
        on the main trajectory on their own (see projectFlares()), and two fragments are a candidate pair
        when their projections are less than FRAGMENT_MATCH_LIMIT apart along the path (see
        projectionMismatch()). The pairs are chosen by the least total distance (Hungarian algorithm), and a
        fragment without a candidate is left without a partner. The main fragment is not matched.

    Arguments:
        traj: [Trajectory object] Solved trajectory of the main fragment, the solution with the original
            picks (see originalPicksTrajectory()).
        ecsv_paths: [list] List of paths to ECSV files it was solved from.
        fragments: [list] Numbers of the fragments to match.

    Return:
        [list] For every two stations of the trajectory with fragments seen at the same times, a dictionary
            with stations (their two station IDs), fragments (the numbers of the fragments of each, two
            lists), distance (the distance along the path of every pair, m, a matrix with a row for every
            fragment of the first station, NaN when they are not seen at the same times) and pairs, the
            matches as (fragment of the first station, fragment of the second, distance (m), distance of the
            closest other candidate of either fragment (m), None if there is none). As the pairs are chosen
            together, that candidate can be closer than the match.
    """

    _, input_meteors = loadECSVs(ecsv_paths, no_prepare=True)
    direction = -vectNorm(traj.radiant_eci_mini)

    # Times and positions along the path of the points of every fragment from every station in the trajectory
    tracks, cameras = {}, {}
    for fragment in fragments:

        _, meteors = loadECSVs(ecsv_paths, no_prepare=True, fragment=fragment)
        for meteor in meteors:
            meteor.trajectory_use = np.ones(len(meteor.time_data), dtype=bool)

        projected = projectFlares(traj, meteors, input_meteors)

        for meteor in meteors:

            points = [projected[(meteor.ecsv_file, i)] for i in range(len(meteor.time_data)) \
                if (meteor.ecsv_file, i) in projected]
            if not points:
                continue

            station_id = points[0]['station_id']
            cameras[station_id] = str(meteor.station_id)
            tracks.setdefault(station_id, {})[fragment] = ( \
                np.array([(point['jd'] - traj.jdt_ref)*86400 for point in points]), \
                np.array([np.dot(point['eci'] - traj.state_vect_mini, direction) for point in points]))

    matches = []
    for station_a, station_b in itertools.combinations(sorted(tracks), 2):

        # Two files of one camera are one station
        if cameras[station_a] == cameras[station_b]:
            continue

        frags_a, frags_b = sorted(tracks[station_a]), sorted(tracks[station_b])

        distance = np.full((len(frags_a), len(frags_b)), np.nan)
        for (i, frag_a), (j, frag_b) in itertools.product(enumerate(frags_a), enumerate(frags_b)):

            (t_a, along_a), (t_b, along_b) = tracks[station_a][frag_a], tracks[station_b][frag_b]
            overlap = min(np.max(t_a), np.max(t_b)) - max(np.min(t_a), np.min(t_b))
            if overlap < FRAGMENT_MATCH_OVERLAP:
                continue

            mismatch = projectionMismatch(t_a, along_a, t_b, along_b)
            if mismatch is not None:
                distance[i, j] = abs(mismatch)

        if np.all(np.isnan(distance)):
            continue

        # Pairs further apart than the limit, or not compared, cannot be matched
        allowed = np.nan_to_num(distance, nan=np.inf) < FRAGMENT_MATCH_LIMIT
        not_allowed = 10*FRAGMENT_MATCH_LIMIT*max(len(frags_a), len(frags_b))
        cost = np.where(allowed, np.nan_to_num(distance), not_allowed)

        pairs = []
        for i, j in zip(*scipy.optimize.linear_sum_assignment(cost)):

            if not allowed[i, j]:
                continue

            # The closest other candidate of either fragment
            others = np.concatenate([np.delete(cost[i, :], j), np.delete(cost[:, j], i)])
            others = others[others < not_allowed]
            closest_other = np.min(others) if len(others) else None

            pairs.append((frags_a[i], frags_b[j], distance[i, j], closest_other))

        matches.append({'stations': (station_a, station_b), 'fragments': (frags_a, frags_b), \
            'distance': distance, 'pairs': pairs})

    return matches



def fragmentMatchReport(matches):
    """ Report the matches of the fragments of every two stations, from fragmentMatches().

    Arguments:
        matches: [list] Matches from fragmentMatches().

    Return:
        [str] The report, empty if there are no matches.
    """

    if not matches:
        return ""

    out_str = "\n"
    out_str += "Additional fragments matched between stations\n"
    out_str += "---------------------------------------------\n"
    out_str += "The fragments of every two stations are matched by where they are along the main path\n"
    out_str += "at the same times, whatever their numbers in the files. The points of every station are\n"
    out_str += "projected on the main trajectory on their own, with the time offsets of the stations, and\n"
    out_str += "the distance is how far apart along the path the two projections are, which also takes\n"
    out_str += "part of a separation from the path. Fragments more than {:.1f} km apart are not matched,\n" \
        .format(FRAGMENT_MATCH_LIMIT/1000)
    out_str += "and a match less than {:.1f} km closer than the closest other candidate of either\n" \
        .format(FRAGMENT_MATCH_MARGIN/1000)
    out_str += "fragment is ambiguous. As the pairs are chosen together, that candidate can be closer\n"
    out_str += "than the match. A match of different numbers which is not ambiguous suggests that the\n"
    out_str += "stations measured the fragments under crossed numbers, and a fragment further than\n"
    out_str += "{:.1f} km from the one with its number at the other station may not be the same\n" \
        .format(FRAGMENT_MATCH_LIMIT/1000)
    out_str += "fragment.\n"

    for match in matches:

        station_a, station_b = match['stations']
        frags_a, frags_b = match['fragments']
        width = max(len(station_a), len(station_b), 8)

        def distanceOf(frag_a, frag_b):
            """ Get the distance of two fragments (m), NaN if they were not compared. """

            if (frag_a not in frags_a) or (frag_b not in frags_b):
                return np.nan

            return match['distance'][frags_a.index(frag_a), frags_b.index(frag_b)]


        def distanceStr(dist):
            """ Format a distance, in km from 1 km on. """

            return "{:.2f} km".format(dist/1000) if (dist >= 1000) else "{:.0f} m".format(dist)


        out_str += "\n"
        out_str += "{:s} and {:s}\n".format(station_a, station_b)
        out_str += "  {:>{w}s}  {:>{w}s}  Distance (m)  Closest other (m)\n".format(station_a, station_b, \
            w=width)

        for frag_a, frag_b, dist, closest_other in match['pairs']:

            notes = []
            if frag_a != frag_b:
                same_dist = distanceOf(frag_a, frag_a)
                notes.append("NOTE: different numbers" + ("" if (frag_a not in frags_b) \
                    else " ({:d} at {:s}: {:s})".format(frag_a, station_b, "not seen at the same times" \
                    if np.isnan(same_dist) else distanceStr(same_dist))))

            if (closest_other is not None) and (closest_other - dist < FRAGMENT_MATCH_MARGIN):
                notes.append("ambiguous")

            line = "  {:{w}d}  {:{w}d}  {:12.0f}  {:>17s}  {:s}".format(frag_a, frag_b, dist, \
                "-" if (closest_other is None) else "{:.0f}".format(closest_other), ", ".join(notes), w=width)
            out_str += line.rstrip() + "\n"

        # The fragments left without a partner, noting those further than the limit from the fragment with
        #   their number at the other station
        partners_a = {pair[0]: pair[1] for pair in match['pairs']}
        partners_b = {pair[1]: pair[0] for pair in match['pairs']}
        unmatched = [(str(frag), "-", frag, station_b, frags_b, partners_b) for frag in frags_a \
            if frag not in partners_a] + [("-", str(frag), frag, station_a, frags_a, partners_a) \
            for frag in frags_b if frag not in partners_b]

        for frag_a_str, frag_b_str, frag, other_station, other_frags, other_partners in unmatched:

            dist = distanceOf(frag, frag)
            if frag not in other_frags:
                note = "not matched"

            elif np.isnan(dist):
                note = "not matched, {:d} at {:s} is not seen at the same times".format(frag, other_station)

            elif dist >= FRAGMENT_MATCH_LIMIT:
                note = "NOTE: not matched, {:d} at {:s} is {:s} away".format(frag, other_station, \
                    distanceStr(dist))

            # The fragment with its number is close, but matched with another one
            else:
                note = "not matched, {:d} at {:s} ({:s} away) is matched with {:d}".format(frag, \
                    other_station, distanceStr(dist), other_partners.get(frag, -1))

            out_str += "  {:>{w}s}  {:>{w}s}  {:>12s}  {:>17s}  {:s}\n".format(frag_a_str, frag_b_str, "-", \
                "-", note, w=width)

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
        help="After the main fragment, also solve the trajectories of the additional fragments in the ECSV " \
            "files (columns with a numeric suffix, e.g. azimuth1, as in Appendix 4 of the GFE standard), " \
            "with the same options, and print how they differ from the main one. Each one is saved in a " \
            "fragment_k folder of the output directory.", \
        action="store_true")

    arg_parser.add_argument('--fragnosys', \
        help="With --fragments, do not include in the Monte Carlo uncertainties of the fragments the " \
            "systematic errors of the stations, measured on the main trajectory if it has 3 stations or " \
            "more.", \
        action="store_true")

    arg_parser.add_argument('--fragtimefit', \
        help="With --fragments, estimate the time offsets of the stations for every fragment instead of " \
            "taking the ones of the main trajectory, which are better constrained by its larger number of " \
            "points.", \
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


    # Options of the trajectory solver, the same for the main fragment and the additional ones
    solver_kwargs = dict(
        max_toffset=max_toffset, monte_carlo=(not cml_args.disablemc), mc_runs=cml_args.mcruns, \
        geometric_uncert=cml_args.uncertgeom, gravity_correction=(not cml_args.disablegravity), 
        gravity_factor=cml_args.gfact,
        plot_all_spatial_residuals=cml_args.plotallspatial, plot_file_type=cml_args.imgformat, \
        show_plots=(not cml_args.hideplots), v_init_part=velpart, v_init_ht=vinitht, \
        show_jacchia=cml_args.jacchia,
        estimate_timing_vel=(False if cml_args.notimefit is None else cml_args.notimefit), \
        fixed_times=cml_args.fixedtimes, mc_noise_std=cml_args.mcstd, enable_OSM_plot=cml_args.enableOSM)

    # Solve the trajectory
    traj = solveTrajectoryGeneric(jdt_ref, meteor_list, dir_path, solver=cml_args.solver, **solver_kwargs)


    # Report the times and heights of the points flagged as flares, if any, also in the saved report
    if (traj is not None) and (cml_args.solver == 'original'):

        # On the solution with the original picks, which the saved report describes, with uncertainties from
        #   the Monte Carlo runs
        traj_orig = originalPicksTrajectory(traj)
        mc_trajs = getattr(traj, 'mc_traj_list', None)
        flares = flareHeights(traj_orig, ecsv_paths, mc_trajs=mc_trajs)
        if flares:

            ref_stations = timeReferenceStations(traj_orig, mc_trajs=mc_trajs)
            flare_report = flareReport(flares, ref_stations=ref_stations)
            flare_report += dragFitFlareWarning(traj_orig, flares)
            print(flare_report)

            report_path = os.path.join(traj.output_dir, traj.file_name + '_report.txt')
            if os.path.isfile(report_path):
                with open(report_path, 'a') as f:
                    f.write(flare_report)


    # Solve the trajectories of the additional fragments, after the main one
    if cml_args.fragments and (traj is not None):

        if cml_args.solver != 'original':
            print()
            print("The trajectories of additional fragments can only be solved with the original solver.")

        elif not ecsvFragments(ecsv_paths):
            print()
            print("There are no additional fragments in the ECSV files.")

        else:
            fragment_trajs = solveFragmentTrajectories(traj, ecsv_paths, solver=cml_args.solver,
                reuse_timing=(not cml_args.fragtimefit), systematics=(not cml_args.fragnosys), \
                **solver_kwargs)
            print(fragmentComparison(originalPicksTrajectory(traj), fragment_trajs, \
                reuse_timing=(not cml_args.fragtimefit)))

            # Project on the main trajectory the fragments without a trajectory, and those whose trajectory
            #   says little, also in its saved report
            traj_orig = originalPicksTrajectory(traj)
            solved = [fragment for fragment, _ in fragment_trajs]
            why = poorFragmentTrajectories(traj_orig, fragment_trajs)
            unsolved = [fragment for fragment in ecsvFragments(ecsv_paths) \
                if (fragment not in solved) or (fragment in why)]

            mc_trajs = getattr(traj, 'mc_traj_list', None)
            projections = unsolvedFragments(traj_orig, ecsv_paths, unsolved, mc_trajs=mc_trajs, why=why)
            if projections:

                unsolved_report = unsolvedFragmentReport(projections, traj_orig, \
                    ref_stations=timeReferenceStations(traj_orig, mc_trajs=mc_trajs))
                print(unsolved_report)

                report_path = os.path.join(traj.output_dir, traj.file_name + '_report.txt')
                if os.path.isfile(report_path):
                    with open(report_path, 'a') as f:
                        f.write(unsolved_report)

            # Match the fragments of every two stations whatever their numbers, also in the saved report
            match_report = fragmentMatchReport(fragmentMatches(traj_orig, ecsv_paths, \
                ecsvFragments(ecsv_paths)))
            if match_report:

                print(match_report)

                report_path = os.path.join(traj.output_dir, traj.file_name + '_report.txt')
                if os.path.isfile(report_path):
                    with open(report_path, 'a') as f:
                        f.write(match_report)
