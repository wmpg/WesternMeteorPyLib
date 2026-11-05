""" Tests for the additional fragments of ECSV files seen from only one station, which are projected on the main
    trajectory. Noiseless events with a known truth, solved with the real solver.

Run with pytest:
    python -m pytest wmpl/Formats/Tests/test_ECSVOneStationFragments.py -v
"""

from __future__ import print_function, division, absolute_import

import copy
import datetime
import os
import sys

import numpy as np

from wmpl.Formats.ECSV import loadECSVs, oneStationFragments, oneStationFragmentReport
from wmpl.Formats.GenericFunctions import solveTrajectoryGeneric
from wmpl.Utils.TrajConversions import latLonAlt2ECEF, ecef2ENU, ecef2LatLonAlt


META = """# %ECSV 0.9
# ---
# meta: !!omap
# - {{obs_latitude: {lat:.6f}}}
# - {{obs_longitude: {lon:.6f}}}
# - {{obs_elevation: {ele:.1f}}}
# - {{camera_id: '{name}'}}
# - {{image_file: 'FF_{name}.fits'}}
"""

MAIN = ['datetime', 'ra', 'dec', 'azimuth', 'altitude', 'x_image', 'y_image', 'mag_data']
FRAGMENT = ['datetime', 'ra', 'dec', 'azimuth', 'altitude', 'x_image', 'y_image']

STATIONS = {'XX0001': (45.00, 10.00, 300.0), 'XX0002': (45.40, 10.70, 200.0), 'XX0003': (44.70, 10.80, 500.0),
    'XX0004': (45.70, 10.00, 400.0)}

T0 = datetime.datetime(2026, 1, 30, 10, 25, 36, 0)
FPS = 25.0
SPEED = 20000.0
BEGIN = np.array(latLonAlt2ECEF(np.radians(45.3), np.radians(10.1), 90000.0))
END = np.array(latLonAlt2ECEF(np.radians(45.05), np.radians(10.55), 60000.0))
DIRECTION = (END - BEGIN)/np.linalg.norm(END - BEGIN)
N_FRAMES = int(np.linalg.norm(END - BEGIN)/SPEED*FPS)
FRAGMENT_START = int(0.4*N_FRAMES)

# Lag of the main fragment, a1*exp(2*t) in metres
LAG_A1 = 0.0


def station(name):

    lat, lon, ele = STATIONS[name]

    return np.array(latLonAlt2ECEF(np.radians(lat), np.radians(lon), ele))


def azAlt(point, name):
    """ Azimuth and altitude (deg) of a point seen from a station. """

    lat, lon, _ = STATIONS[name]
    east, north, up = ecef2ENU(np.radians(lat), np.radians(lon), *(point - station(name)))

    return np.degrees(np.arctan2(east, north))%360, np.degrees(np.arctan2(up, np.hypot(east, north)))


def mainPoint(i):

    return BEGIN + DIRECTION*(SPEED*i/FPS - LAG_A1*np.exp(2.0*i/FPS))


def writeEvent(dir_path, fragment_stations=('XX0001',), offset=None, main_stations=('XX0001', 'XX0002', 'XX0003'),
    clocks=None):
    """ Write an event without noise: the main fragment, and fragment 1 on the last frames, seen from the given
        stations, at the point of the main one at the same time displaced by offset(i), in metres. """

    if offset is None:
        offset = lambda i: np.zeros(3)

    paths = []
    for name in sorted(set(main_stations) | set(fragment_stations)):

        lat, lon, ele = STATIONS[name]
        clock = 0.0 if clocks is None else clocks.get(name, 0.0)

        rows = []
        for i in range(N_FRAMES):

            when = (T0 + datetime.timedelta(seconds=i/FPS + clock)).strftime("%Y-%m-%dT%H:%M:%S.%f")
            row = ['']*len(MAIN)
            if name in main_stations:
                row = [when, '10.0', '-30.0', '{:.8f}'.format(azAlt(mainPoint(i), name)[0]),
                    '{:.8f}'.format(azAlt(mainPoint(i), name)[1]), '100', '200', '2.0']

            row += [str(i), 'False', 'True']

            fragment = ['']*len(FRAGMENT)
            if (name in fragment_stations) and (i >= FRAGMENT_START):
                azim, alt = azAlt(mainPoint(i) + offset(i), name)
                fragment = [when, '10.0', '-30.0', '{:.8f}'.format(azim), '{:.8f}'.format(alt), '100', '210']

            rows.append(",".join(row + fragment))

        header = MAIN + ['frame_number', 'flare', 'trajectory_use'] + [column + '1' for column in FRAGMENT]

        path = os.path.join(str(dir_path), name + '.ecsv')
        with open(path, 'w') as f:
            f.write(META.format(lat=lat, lon=lon, ele=ele, name=name) + ",".join(header) + "\n" + "\n".join(rows) + "\n")

        paths.append(path)

    return paths


def solveMain(paths, dir_path, estimate_timing=False):

    jdt_ref, meteor_list = loadECSVs(paths)

    return solveTrajectoryGeneric(jdt_ref, meteor_list, str(dir_path), solver='original', monte_carlo=False,
        save_results=False, show_plots=False, verbose=False, calc_orbit=False, gravity_correction=False,
        estimate_timing_vel=estimate_timing)


def lineOfSightPlane(name, i):
    """ Unit vectors normal to the path: in the plane of the line of sight and the path, and across the line of
        sight, at frame i. """

    los = mainPoint(i) - station(name)
    across = np.cross(los, DIRECTION)

    return np.cross(DIRECTION, across)/np.linalg.norm(np.cross(DIRECTION, across)), across/np.linalg.norm(across)


def test_a_fragment_on_the_main_path_gets_the_heights_of_the_main_one(tmp_path):

    paths = writeEvent(tmp_path)
    traj = solveMain(paths, tmp_path)

    fragments = oneStationFragments(traj, paths)

    assert [frag['fragment'] for frag in fragments] == [1]

    frag = fragments[0]
    assert frag['station_id'] == 'XX0001'
    assert frag['n_points'] == N_FRAMES - FRAGMENT_START
    assert frag['std'] is None

    obs = [obs for obs in traj.observations if str(obs.station_id) == 'XX0001'][0]
    assert np.max(np.abs(frag['ht'] - obs.model_ht[FRAGMENT_START:])) < 1.0
    assert np.max(np.abs(frag['jd'] - obs.JD_data[FRAGMENT_START:]))*86400 < 1e-3

    assert np.max(np.abs(frag['off'])) < 5.0
    assert np.max(np.abs(frag['ahead'])) < 20.0


def height(point):

    return ecef2LatLonAlt(*point)[2]


def test_a_separation_across_the_line_of_sight_is_seen(tmp_path):

    middle = (FRAGMENT_START + N_FRAMES)//2
    across = lineOfSightPlane('XX0001', middle)[1]

    paths = writeEvent(tmp_path, offset=lambda i: 300.0*across)
    frag = oneStationFragments(solveMain(paths, tmp_path), paths)[0]

    assert abs(abs(frag['off'][middle - FRAGMENT_START]) - 300.0) < 10.0


def test_a_separation_along_the_line_of_sight_is_not_seen_and_the_height_is_off_as_reported(tmp_path):

    middle = (FRAGMENT_START + N_FRAMES)//2
    in_plane = lineOfSightPlane('XX0001', middle)[0]

    paths = writeEvent(tmp_path, offset=lambda i: 300.0*in_plane)
    traj = solveMain(paths, tmp_path)
    frag = oneStationFragments(traj, paths)[0]

    k = middle - FRAGMENT_START
    assert abs(frag['off'][k]) < 10.0

    # The heights of the solver are above sea level, so the height of the point on the main path is the reference
    obs = [obs for obs in traj.observations if str(obs.station_id) == 'XX0001'][0]
    true_height = obs.model_ht[middle] + height(mainPoint(middle) + 300.0*in_plane) - height(mainPoint(middle))

    error = abs(frag['ht'][k] - true_height)
    assert error > 50.0
    assert abs(error - 300.0*frag['ht_per_m'][k]) < 0.05*error

    # How far ahead of the main one it appears to be, while it is at the same distance along the path
    assert abs(abs(frag['ahead'][k]) - 300.0*abs(frag['ahead_per_m'][k])) < 0.05*abs(frag['ahead'][k])


def test_a_fragment_moving_away_from_the_path_gives_its_speed_across_the_line_of_sight(tmp_path):

    middle = (FRAGMENT_START + N_FRAMES)//2
    across = lineOfSightPlane('XX0001', middle)[1]

    # 100 m/s from the first frame of the fragment
    paths = writeEvent(tmp_path, offset=lambda i: 100.0*(i - FRAGMENT_START)/FPS*across)
    frag = oneStationFragments(solveMain(paths, tmp_path), paths)[0]

    assert abs(np.sign(frag['off_last'])*frag['off_rate'] - 100.0) < 5.0
    assert abs(frag['off_first']) < 10.0


def test_a_fragment_behind_the_main_one_is_reported_behind(tmp_path):

    # Slows down at 200 m/s with respect to the main one, along the path
    paths = writeEvent(tmp_path, offset=lambda i: -200.0*(i - FRAGMENT_START)/FPS*DIRECTION)
    frag = oneStationFragments(solveMain(paths, tmp_path), paths)[0]

    duration = (N_FRAMES - 1 - FRAGMENT_START)/FPS

    assert abs(frag['ahead_first']) < 20.0
    assert abs(frag['ahead_last'] + 200.0*duration) < 20.0


def test_a_main_fragment_slowing_down_is_followed_locally(tmp_path, monkeypatch):
    """ Its deceleration is not a parabola over the whole flight, which would put a fragment moving with it off by 
        tens of metres at the end. """

    monkeypatch.setattr(sys.modules[__name__], "LAG_A1", 5.0)

    paths = writeEvent(tmp_path)
    frag = oneStationFragments(solveMain(paths, tmp_path), paths)[0]

    assert abs(frag['ahead_last']) < 15.0


def test_the_time_offset_of_the_station_is_the_one_of_the_main_trajectory(tmp_path):
    """ A clock 40 ms late at the station of the fragment does not make it appear behind the main one, which at
        this speed would be 800 m. It is not the reference station, whose time offset is 0. """

    paths = writeEvent(tmp_path, fragment_stations=('XX0002',), clocks={'XX0002': 0.040})
    traj = solveMain(paths, tmp_path, estimate_timing=True)

    frag = oneStationFragments(traj, paths)[0]

    assert frag['station_id'] == 'XX0002'
    assert np.max(np.abs(frag['ahead'])) < 30.0


def test_a_fragment_seen_from_two_stations_is_left_to_the_solver(tmp_path):

    paths = writeEvent(tmp_path, fragment_stations=('XX0001', 'XX0002'))

    assert oneStationFragments(solveMain(paths, tmp_path), paths) == []


def test_a_station_out_of_the_main_trajectory_has_no_time_offset_so_it_is_skipped(tmp_path, capsys):

    paths = writeEvent(tmp_path, fragment_stations=('XX0004',))

    assert oneStationFragments(solveMain(paths, tmp_path), paths) == []
    assert "its only station, XX0004, is not in the main trajectory" in capsys.readouterr().out


def monteCarloRuns(traj, n=6):
    """ Copies of a solution with a slightly different path, standing in for its Monte Carlo runs. """

    rng = np.random.default_rng(1)

    runs = []
    for _ in range(n):

        run = copy.copy(traj)
        run.state_vect_mini = traj.state_vect_mini + rng.normal(0, 20.0, 3)
        run.time_diffs_final = np.zeros(len(traj.observations))
        runs.append(run)

    return runs


def test_the_report_with_and_without_the_monte_carlo_runs(tmp_path):

    middle = (FRAGMENT_START + N_FRAMES)//2
    across = lineOfSightPlane('XX0001', middle)[1]

    # With some scatter on the lines of sight, for the noise of the runs
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 15.0, N_FRAMES)
    paths = writeEvent(tmp_path, offset=lambda i: (100.0 + noise[i])*across)
    traj = solveMain(paths, tmp_path)

    fragments = oneStationFragments(traj, paths)
    report = oneStationFragmentReport(fragments, traj, ref_stations=['XX0002'])

    assert "Fragment 1, from XX0001" in report
    assert "which one station cannot test" in report
    assert "reference station XX0002" in report
    assert "Without the Monte Carlo runs" in report
    assert "+/-" not in report
    assert "extrapolated" not in report

    # Seen after the last point of the main fragment
    report = oneStationFragmentReport([dict(fragments[0], outside=0.4)], traj)
    assert "seen 0.40 s outside the main one" in report

    fragments = oneStationFragments(traj, paths, mc_trajs=monteCarloRuns(traj))
    frag = fragments[0]

    assert np.all(frag['std']['ht'] > 0)
    assert frag['std']['off_last'] > 0

    report = oneStationFragmentReport(fragments, traj)

    assert "Without the Monte Carlo runs" not in report
    assert "They do not include a separation from the main path" in report
    assert "+/-" in report
