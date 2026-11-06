""" Tests for the additional fragments of ECSV files without a trajectory of their own, seen from one station
    or whose trajectory could not be solved or says little, which are projected on the main trajectory, each
    station on its own, and for the flares. Noiseless events with a known truth, solved with the real solver.

Run with pytest:
    python -m pytest wmpl/Formats/Tests/test_ECSVUnsolvedFragments.py -v
"""

from __future__ import print_function, division, absolute_import

import copy
import datetime
import os
import shutil
import sys
from types import SimpleNamespace

import numpy as np

from wmpl.Formats.ECSV import loadECSVs, unsolvedFragments, unsolvedFragmentReport
from wmpl.Formats.ECSV import flareHeights, runTimeOffsets
from wmpl.Formats.ECSV import loadFlares, projectFlares
from wmpl.Formats.ECSV import poorFragmentTrajectories
from wmpl.Formats.GenericFunctions import solveTrajectoryGeneric
from wmpl.Utils.TrajConversions import latLonAlt2ECEF, ecef2ENU, ecef2LatLonAlt


# Metadata of a station
META = """# %ECSV 0.9
# ---
# meta: !!omap
# - {{obs_latitude: {lat:.6f}}}
# - {{obs_longitude: {lon:.6f}}}
# - {{obs_elevation: {ele:.1f}}}
# - {{camera_id: '{name}'}}
# - {{image_file: 'FF_{name}.fits'}}
"""

# Columns of the main fragment and of an additional one (with its number as a suffix)
MAIN = ['datetime', 'ra', 'dec', 'azimuth', 'altitude', 'x_image', 'y_image', 'mag_data']
FRAGMENT = ['datetime', 'ra', 'dec', 'azimuth', 'altitude', 'x_image', 'y_image']

# Latitude (deg), longitude (deg) and elevation (m) of the stations
STATIONS = {'XX0001': (45.00, 10.00, 300.0), 'XX0002': (45.40, 10.70, 200.0), 'XX0003': (44.70, 10.80, 500.0),
    'XX0004': (45.70, 10.00, 400.0)}

# Time of the first frame, frame rate (frames per second) and speed (m/s)
T0 = datetime.datetime(2026, 1, 30, 10, 25, 36, 0)
FPS = 25.0
SPEED = 20000.0

# The straight path of the main fragment, and the frame from which fragment 1 is seen
BEGIN = np.array(latLonAlt2ECEF(np.radians(45.3), np.radians(10.1), 90000.0))
END = np.array(latLonAlt2ECEF(np.radians(45.05), np.radians(10.55), 60000.0))
DIRECTION = (END - BEGIN)/np.linalg.norm(END - BEGIN)
N_FRAMES = int(np.linalg.norm(END - BEGIN)/SPEED*FPS)
FRAGMENT_START = int(0.4*N_FRAMES)

# Lag of the main fragment, a1*exp(2*t) in metres
LAG_A1 = 0.0



def station(name):
    """ Compute the ECEF position of a station.

    Arguments:
        name: [str] Station code.

    Return:
        [ndarray] Position (m).
    """

    lat, lon, ele = STATIONS[name]

    return np.array(latLonAlt2ECEF(np.radians(lat), np.radians(lon), ele))



def azAlt(point, name):
    """ Compute the azimuth and altitude of a point seen from a station.

    Arguments:
        point: [ndarray] ECEF position of the point (m).
        name: [str] Station code.

    Return:
        (azim, alt): [tuple of floats] Azimuth, +E of due N, and altitude (deg).
    """

    lat, lon, _ = STATIONS[name]
    east, north, up = ecef2ENU(np.radians(lat), np.radians(lon), *(point - station(name)))

    return np.degrees(np.arctan2(east, north))%360, np.degrees(np.arctan2(up, np.hypot(east, north)))



def mainPoint(i):
    """ Compute the position of the main fragment on a frame.

    Arguments:
        i: [int] Frame number.

    Return:
        [ndarray] ECEF position (m).
    """

    return BEGIN + DIRECTION*(SPEED*i/FPS - LAG_A1*np.exp(2.0*i/FPS))



def writeEvent(dir_path, fragment_stations=('XX0001',), offset=None, \
    main_stations=('XX0001', 'XX0002', 'XX0003'), clocks=None):
    """ Write the ECSV files of an event without noise: the main fragment, and fragment 1 on the last frames,
        at the point of the main one at the same time displaced by offset(i).

    Arguments:
        dir_path: [str] Directory of the files.

    Keyword arguments:
        fragment_stations: [tuple] Stations which see fragment 1. Only XX0001 by default.
        offset: [function] Displacement of fragment 1 from the main fragment on frame i (m). None by
            default, for none.
        main_stations: [tuple] Stations which see the main fragment. XX0001 to XX0003 by default.
        clocks: [dict] Error of the clock of some stations, {station code: error (s)}. None by default.

    Return:
        [list] Paths of the files.
    """

    if offset is None:
        offset = lambda i: np.zeros(3)

    paths = []
    for name in sorted(set(main_stations) | set(fragment_stations)):

        lat, lon, ele = STATIONS[name]
        clock = 0.0 if (clocks is None) else clocks.get(name, 0.0)

        rows = []
        for i in range(N_FRAMES):

            when = (T0 + datetime.timedelta(seconds=i/FPS + clock)).strftime("%Y-%m-%dT%H:%M:%S.%f")
            row = ['']*len(MAIN)
            if name in main_stations:
                azim, alt = azAlt(mainPoint(i), name)
                row = [when, '10.0', '-30.0', '{:.8f}'.format(azim), '{:.8f}'.format(alt), '100', '200', \
                    '2.0']

            row += [str(i), 'False', 'True']

            fragment = ['']*len(FRAGMENT)
            if (name in fragment_stations) and (i >= FRAGMENT_START):
                azim, alt = azAlt(mainPoint(i) + offset(i), name)
                fragment = [when, '10.0', '-30.0', '{:.8f}'.format(azim), '{:.8f}'.format(alt), '100', '210']

            rows.append(",".join(row + fragment))

        header = MAIN + ['frame_number', 'flare', 'trajectory_use'] + [column + '1' for column in FRAGMENT]

        path = os.path.join(str(dir_path), name + '.ecsv')
        with open(path, 'w') as f:
            f.write(META.format(lat=lat, lon=lon, ele=ele, name=name) + ",".join(header) + "\n" \
                + "\n".join(rows) + "\n")

        paths.append(path)

    return paths



def solveMain(paths, dir_path, estimate_timing=False):
    """ Solve the main fragment of an event without the Monte Carlo runs.

    Arguments:
        paths: [list] Paths of the ECSV files.
        dir_path: [str] Output directory.

    Keyword arguments:
        estimate_timing: [bool] Estimate the time offsets of the stations. False by default.

    Return:
        [Trajectory object] The solved trajectory.
    """

    jdt_ref, meteor_list = loadECSVs(paths)

    return solveTrajectoryGeneric(jdt_ref, meteor_list, str(dir_path), solver='original', monte_carlo=False, \
        save_results=False, show_plots=False, verbose=False, calc_orbit=False, gravity_correction=False, \
        estimate_timing_vel=estimate_timing)



def lineOfSightPlane(name, i):
    """ Compute the unit vectors normal to the path in the plane of the line of sight and the path, and across
        the line of sight, at a frame.

    Arguments:
        name: [str] Station code.
        i: [int] Frame number.

    Return:
        (in_plane, across): [tuple of ndarrays] The unit vectors.
    """

    los = mainPoint(i) - station(name)
    across = np.cross(los, DIRECTION)

    return np.cross(DIRECTION, across)/np.linalg.norm(np.cross(DIRECTION, across)), \
        across/np.linalg.norm(across)



def height(point):
    """ Compute the height of a point above the WGS84 ellipsoid.

    Arguments:
        point: [ndarray] ECEF position (m).

    Return:
        [float] Height (m).
    """

    return ecef2LatLonAlt(*point)[2]



def monteCarloRuns(traj, n=6):
    """ Make copies of a solution with a slightly different path, standing in for its Monte Carlo runs.

    Arguments:
        traj: [Trajectory object] Solved trajectory.

    Keyword arguments:
        n: [int] Number of runs. 6 by default.

    Return:
        [list] The runs.
    """

    rng = np.random.default_rng(1)

    runs = []
    for _ in range(n):

        run = copy.copy(traj)
        run.state_vect_mini = traj.state_vect_mini + rng.normal(0, 20.0, 3)
        run.time_diffs_final = np.zeros(len(traj.observations))
        runs.append(run)

    return runs



def testAFragmentOnTheMainPathGetsTheHeightsOfTheMainOne(tmp_path):
    """ Check that a fragment on the main path gets the heights and times of the main fragment. """

    paths = writeEvent(tmp_path)
    traj = solveMain(paths, tmp_path)

    solutions = unsolvedFragments(traj, paths, [1])

    assert [frag['fragment'] for frag in solutions] == [1]

    frag = solutions[0]
    assert frag['station_id'] == 'XX0001'
    assert frag['n_stations'] == 1
    assert frag['n_points'] == N_FRAMES - FRAGMENT_START
    assert frag['std'] is None

    obs = [obs for obs in traj.observations if str(obs.station_id) == 'XX0001'][0]
    assert np.max(np.abs(frag['ht'] - obs.model_ht[FRAGMENT_START:])) < 1.0
    assert np.max(np.abs(frag['jd'] - obs.JD_data[FRAGMENT_START:]))*86400 < 1e-3

    assert np.max(np.abs(frag['off'])) < 5.0
    assert np.max(np.abs(frag['ahead'])) < 20.0



def testASeparationAcrossTheLineOfSightIsSeen(tmp_path):
    """ Check that a fragment 300 m off the path across the line of sight is seen 300 m off. """

    middle = (FRAGMENT_START + N_FRAMES)//2
    across = lineOfSightPlane('XX0001', middle)[1]

    paths = writeEvent(tmp_path, offset=lambda i: 300.0*across)
    frag = unsolvedFragments(solveMain(paths, tmp_path), paths, [1])[0]

    assert abs(abs(frag['off'][middle - FRAGMENT_START]) - 300.0) < 10.0



def testASeparationAlongTheLineOfSightIsNotSeenAndMovesTheHeightAsReported(tmp_path):
    """ Check that a fragment 300 m off the path in the plane of the path and the line of sight is not seen
        off, and that its height and how far ahead it is are off by what the report gives per metre.
    """

    middle = (FRAGMENT_START + N_FRAMES)//2
    in_plane = lineOfSightPlane('XX0001', middle)[0]

    paths = writeEvent(tmp_path, offset=lambda i: 300.0*in_plane)
    traj = solveMain(paths, tmp_path)
    frag = unsolvedFragments(traj, paths, [1])[0]

    k = middle - FRAGMENT_START
    assert abs(frag['off'][k]) < 10.0

    # The heights of the solver are above sea level, so the height of the point on the main path is the
    #   reference
    obs = [obs for obs in traj.observations if str(obs.station_id) == 'XX0001'][0]
    true_height = obs.model_ht[middle] + height(mainPoint(middle) + 300.0*in_plane) \
        - height(mainPoint(middle))

    error = abs(frag['ht'][k] - true_height)
    assert error > 50.0
    assert abs(error - 300.0*frag['ht_per_m'][k]) < 0.05*error

    # How far ahead of the main one it appears to be, while it is at the same distance along the path
    assert abs(abs(frag['ahead'][k]) - 300.0*abs(frag['ahead_per_m'][k])) < 0.05*abs(frag['ahead'][k])



def testAFragmentMovingAwayFromThePathGivesItsSpeedAcrossTheLineOfSight(tmp_path):
    """ Check that a fragment moving away from the path at 100 m/s across the line of sight is seen to. """

    middle = (FRAGMENT_START + N_FRAMES)//2
    across = lineOfSightPlane('XX0001', middle)[1]

    # Move away at 100 m/s from the first frame of the fragment
    paths = writeEvent(tmp_path, offset=lambda i: 100.0*(i - FRAGMENT_START)/FPS*across)
    frag = unsolvedFragments(solveMain(paths, tmp_path), paths, [1])[0]

    assert abs(np.sign(frag['off_last'])*frag['off_rate'] - 100.0) < 5.0
    assert abs(frag['off_first']) < 10.0



def testAFragmentBehindTheMainOneIsReportedBehind(tmp_path):
    """ Check that a fragment slowing down with respect to the main one is reported behind it. """

    # Slow down at 200 m/s with respect to the main one, along the path
    paths = writeEvent(tmp_path, offset=lambda i: -200.0*(i - FRAGMENT_START)/FPS*DIRECTION)
    frag = unsolvedFragments(solveMain(paths, tmp_path), paths, [1])[0]

    duration = (N_FRAMES - 1 - FRAGMENT_START)/FPS

    assert abs(frag['ahead_first']) < 20.0
    assert abs(frag['ahead_last'] + 200.0*duration) < 20.0



def testAMainFragmentSlowingDownIsFollowedLocally(tmp_path, monkeypatch):
    """ Check that a fragment moving with a main fragment that slows down stays with it, as its deceleration
        is not a parabola over the whole flight.
    """

    monkeypatch.setattr(sys.modules[__name__], 'LAG_A1', 5.0)

    paths = writeEvent(tmp_path)
    frag = unsolvedFragments(solveMain(paths, tmp_path), paths, [1])[0]

    assert abs(frag['ahead_last']) < 15.0



def testTheTimeOffsetOfTheStationIsTheOneOfTheMainTrajectory(tmp_path):
    """ Check that a clock 40 ms late at the station of the fragment, which is not the reference station, does
        not make it appear behind the main one, which at this speed would be 800 m.
    """

    paths = writeEvent(tmp_path, fragment_stations=('XX0002',), clocks={'XX0002': 0.040})
    traj = solveMain(paths, tmp_path, estimate_timing=True)

    frag = unsolvedFragments(traj, paths, [1])[0]

    assert frag['station_id'] == 'XX0002'
    assert np.max(np.abs(frag['ahead'])) < 30.0



def testTheSolutionWithTheOriginalPicksAmongTheRunsAddsNoTimeOffset(tmp_path):
    """ Check that the solution with the original picks, which the solver adds to its Monte Carlo runs with
        the time offsets it estimated already applied to its times, does not have them added again: with a
        clock 40 ms late, that run would move a flare by 40 ms and a fragment along the path by 800 m.
    """

    paths = writeEvent(tmp_path, fragment_stations=('XX0002',), clocks={'XX0002': 0.040})

    # Flag a point of the main fragment as a flare
    path = [path for path in paths if path.endswith('XX0002.ecsv')][0]
    with open(path) as f:
        text = f.read()
    with open(path, 'w') as f:
        f.write(text.replace(",{:d},False,True,".format(FRAGMENT_START), \
            ",{:d},True,True,".format(FRAGMENT_START)))

    traj = solveMain(paths, tmp_path, estimate_timing=True)
    runs = monteCarloRuns(traj) + [traj]

    assert np.max(np.abs(traj.time_diffs_final)) > 0.03
    assert all(t_diff == 0 for t_diff in runTimeOffsets(traj, traj).values())

    frag = unsolvedFragments(traj, paths, [1], mc_trajs=runs)[0]

    assert np.max(frag['std']['t']) < 1e-3
    assert frag['std']['ahead_first'] < 50.0

    flares = flareHeights(traj, paths, mc_trajs=runs)

    assert len(flares) == 1
    assert flares[0]['t_std_mc'] < 1e-3



def testAFlareNotUsedInTheTrajectoryTakesTheHeightOfTheTrajectoryAtItsTime(tmp_path):
    """ Check that a flare point not used in the trajectory, clicked where the meteor was 1 s later (as in a
        saturated flare, in which it cannot be seen), takes the height of the trajectory at its time and not
        that of its line of sight, also in the Monte Carlo runs, while a flare used in the trajectory is
        projected on it.
    """

    paths = writeEvent(tmp_path)
    frame_unused, frame_used = FRAGMENT_START, FRAGMENT_START + 5
    clicked = azAlt(mainPoint(frame_unused + int(FPS)), 'XX0002')

    # Flag the two flares at XX0002, the one not used clicked elsewhere
    path = [path for path in paths if path.endswith('XX0002.ecsv')][0]
    with open(path) as f:
        lines = f.read().splitlines()
    for n, line in enumerate(lines):
        fields = line.split(',')
        if line.startswith('#') or (fields[0] == 'datetime'):
            continue
        if int(fields[8]) == frame_unused:
            fields[3], fields[4] = '{:.8f}'.format(clicked[0]), '{:.8f}'.format(clicked[1])
            fields[9], fields[10] = 'True', 'False'
        elif int(fields[8]) == frame_used:
            fields[9] = 'True'
        lines[n] = ','.join(fields)
    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")

    traj = solveMain(paths, tmp_path)
    reference = [obs for obs in traj.observations if obs.station_id == 'XX0001'][0]

    flares = flareHeights(traj, paths)
    unused = [flare for flare in flares if not flare['used']][0]
    used = [flare for flare in flares if flare['used']][0]

    assert abs(unused['ht'] - reference.model_ht[frame_unused]) < 20.0
    assert abs(used['ht'] - reference.model_ht[frame_used]) < 20.0

    # Its line of sight would be kilometres away
    flare_meteors, input_meteors = loadFlares(paths)
    projected = projectFlares(traj, flare_meteors, input_meteors)
    assert min(abs(flare['ht'] - unused['ht']) for flare in projected.values() if not flare['used']) > 1000.0

    # In the runs, given other time offsets of the station than the other stations, it follows the
    #   trajectory at the time each run gives it, while its line of sight would not move
    runs = monteCarloRuns(traj)
    index = [obs.station_id for obs in traj.observations].index('XX0002')
    offsets = 0.01*np.arange(len(runs))
    for run, offset in zip(runs, offsets):
        run.time_diffs_final = np.zeros(len(traj.observations))
        run.time_diffs_final[index] = offset

    unused = [flare for flare in flareHeights(traj, paths, mc_trajs=runs) if not flare['used']][0]

    assert unused['ht_std_mc'] > 0.3*abs(unused['ht_rate'])*np.std(offsets)



def testAFragmentOfTwoStationsWithoutATrajectoryIsProjectedFromEach(tmp_path):
    """ Check that a fragment seen from two stations whose trajectory could not be solved is projected from
        each of them, and that two files of one camera are one station.
    """

    paths = writeEvent(tmp_path, fragment_stations=('XX0001', 'XX0002'))
    traj = solveMain(paths, tmp_path)

    solutions = unsolvedFragments(traj, paths, [1])

    assert sorted(frag['station_id'] for frag in solutions) == ['XX0001', 'XX0002']
    assert all(frag['n_stations'] == 2 for frag in solutions)
    assert "its trajectory could not be solved" in unsolvedFragmentReport(solutions, traj)

    # A second file of the camera of the only station of a fragment
    dir_one = tmp_path/"one"
    dir_one.mkdir()
    paths = writeEvent(dir_one, fragment_stations=('XX0001',))
    second = str(dir_one/"XX0001_second.ecsv")
    shutil.copy(paths[0], second)
    paths.append(second)

    solutions = unsolvedFragments(solveMain(paths, dir_one), paths, [1])

    assert len(solutions) == 2
    assert all(frag['n_stations'] == 1 for frag in solutions)



def testAStationOutOfTheMainTrajectoryIsNotProjected(tmp_path, capsys):
    """ Check that a fragment whose only station is not in the main trajectory, with no time offset, is not
        projected.
    """

    paths = writeEvent(tmp_path, fragment_stations=('XX0004',))

    assert unsolvedFragments(solveMain(paths, tmp_path), paths, [1]) == []
    assert "Fragment 1 from XX0004 is not projected" in capsys.readouterr().out



def testTheReportWithAndWithoutTheMonteCarloRuns(tmp_path):
    """ Check the report of a fragment projected on the main trajectory, with and without the Monte Carlo
        runs, and when it is seen after the main one.
    """

    middle = (FRAGMENT_START + N_FRAMES)//2
    across = lineOfSightPlane('XX0001', middle)[1]

    # Scatter the lines of sight, for the noise of the runs
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 15.0, N_FRAMES)
    paths = writeEvent(tmp_path, offset=lambda i: (100.0 + noise[i])*across)
    traj = solveMain(paths, tmp_path)

    solutions = unsolvedFragments(traj, paths, [1])
    report = unsolvedFragmentReport(solutions, traj, ref_stations=['XX0002'])

    assert "Fragment 1, from XX0001 (seen from one station)" in report
    assert "which one station cannot test" in report
    assert "reference station XX0002" in report
    assert "Without the Monte Carlo runs" in report
    assert "+/-" not in report
    assert "extrapolated" not in report

    # Seen after the last point of the main fragment
    report = unsolvedFragmentReport([dict(solutions[0], outside=0.4)], traj)
    assert "seen 0.40 s outside the main one" in report

    solutions = unsolvedFragments(traj, paths, [1], mc_trajs=monteCarloRuns(traj))
    frag = solutions[0]

    assert np.all(frag['std']['ht'] > 0)
    assert frag['std']['off_last'] > 0

    report = unsolvedFragmentReport(solutions, traj)

    assert "Without the Monte Carlo runs" not in report
    assert "They do not include a\nseparation from the main path" in report
    assert "+/-" in report



def testPoorFragmentTrajectoriesAreFound():
    """ Check that the solved fragments whose stations converge by less than 3 deg, or whose trajectory is
        more than 1.5 km off the main path, are found, with why, and not the others.
    """

    def solution(offset, conv_angle):
        """ Make a solution along a straight path, this far off the main one, with this convergence angle. """

        along = 800.0*np.arange(20)
        obs = SimpleNamespace(JD_data=2461000.5 + np.arange(20)/FPS/86400, ignore_station=False, \
            model_eci=np.array([[6.5e6 - distance, offset, 0.0] for distance in along]), \
            ignore_list=np.zeros(20, dtype=int))

        return SimpleNamespace(observations=[obs], radiant_eci_mini=np.array([1.0, 0.0, 0.0]), \
            state_vect_mini=np.array([6.5e6, 0.0, 0.0]), best_conv_inter=None if (conv_angle is None) \
            else SimpleNamespace(conv_angle=np.radians(conv_angle)))

    main = solution(0.0, 40.0)
    fragment_trajs = [(1, solution(0.0, 2.0)), (2, solution(2000.0, 10.0)), (3, solution(300.0, 10.0)), \
        (4, solution(300.0, None))]

    assert poorFragmentTrajectories(main, fragment_trajs) == {1: "its stations converge by only 2.0 deg", \
        2: "its trajectory is 2.0 km off the main path"}



def testTheReportSaysWhyAFragmentIsProjected(tmp_path):
    """ Check that the report says why a fragment is projected when given, e.g. a solved one whose stations
        hardly converge, and otherwise that it was seen from one station.
    """

    paths = writeEvent(tmp_path)
    traj = solveMain(paths, tmp_path)

    why = {1: "its stations converge by only 2.0 deg"}
    report = unsolvedFragmentReport(unsolvedFragments(traj, paths, [1], why=why), traj)

    assert "Fragment 1, from XX0001 (its stations converge by only 2.0 deg)" in report
    assert "Additional fragments projected on the main trajectory" in report
    assert "(seen from one station)" in unsolvedFragmentReport(unsolvedFragments(traj, paths, [1]), traj)

