""" Tests for reading ECSV files with several fragments: the GFE standard (Appendix 4) keeps one row per
    frame, with the additional fragments in columns with a numeric suffix (e.g. azimuth1). Frames on which
    only an additional fragment was measured leave the main fragment columns empty, and main fragment points
    flagged with trajectory_use = False are left out of its trajectory.

Run with pytest:
    python -m pytest wmpl/Formats/Tests/test_ECSV.py -v
"""

from __future__ import print_function, division, absolute_import

from types import SimpleNamespace

import numpy as np

from wmpl.Formats.ECSV import ecsvFragments, loadECSVs, appliedTimeOffsets, solverStationIDs, \
    originalPicksTrajectory, fragmentComparison
from wmpl.Utils.Pickling import savePickle


META = """# %ECSV 0.9
# ---
# meta: !!omap
# - {obs_latitude: -43.6}
# - {obs_longitude: 172.1}
# - {obs_elevation: 100.0}
# - {camera_id: 'STATION'}
# - {image_file: 'FF_STATION.fits'}
"""

MAIN = ['datetime', 'ra', 'dec', 'azimuth', 'altitude', 'x_image', 'y_image', 'mag_data']
FRAGMENT = ['datetime', 'ra', 'dec', 'azimuth', 'altitude', 'x_image', 'y_image']


def point(i, altitude):
    """ Time and position of a point on frame i. """

    return ['2026-01-30T10:25:36.{:06d}'.format(40000*i), '10.0', '-30.0', '{:.4f}'.format(120 + 0.5*i), 
        '{:.4f}'.format(altitude - 0.3*i), '{:.3f}'.format(100 + 10*i), '{:.3f}'.format(200 + 5*i)]


def writeEcsv(path, station, fragments=True):
    """ Six frames of the main fragment, the third flagged as not to be used and the last only measured for
        fragment 1, and fragment 1 on the last five frames. Without fragments, the previous file format. """

    header = MAIN + (['frame_number', 'flare', 'trajectory_use'] + [name + '1' for name in FRAGMENT] 
        if fragments else [])

    rows = []
    for i in range(7):

        main = point(i, 30.0) + ['{:.2f}'.format(2.0 - 0.1*i)] if i < 6 else [''] * len(MAIN)
        if not fragments:
            if i < 6:
                rows.append(main)
            continue

        flags = [str(i), 'False', 'False' if i == 2 else 'True'] if i < 6 else [str(i), '', '']
        fragment = point(i, 29.0) if i >= 2 else [''] * len(FRAGMENT)
        rows.append(main + flags + fragment)

    with open(path, 'w') as f:
        f.write(META.replace('STATION', station) + ",".join(header) + "\n")
        f.write("".join(",".join(row) + "\n" for row in rows))

    return str(path)


def test_fragments_are_found(tmp_path):
    """ The additional fragments are the suffixes of the azimuth columns; files without them have none. """

    assert ecsvFragments([writeEcsv(tmp_path/"a.ecsv", "XX0001")]) == [1]
    assert ecsvFragments([writeEcsv(tmp_path/"b.ecsv", "XX0002", fragments=False)]) == []


def test_main_fragment_skips_empty_and_unused_points(tmp_path):
    """ Of the 7 rows, the frame with only fragment 1 is skipped, and so is the main fragment point flagged
        as not to be used in the trajectory, leaving 5 points. """

    _, meteors = loadECSVs([writeEcsv(tmp_path/"a.ecsv", "XX0001")], no_prepare=True)

    assert len(meteors) == 1
    assert np.allclose(np.degrees(meteors[0].azim_data), [120.0, 120.5, 121.5, 122.0, 122.5])


def test_additional_fragment_columns(tmp_path):
    """ Fragment 1 is read from its own columns, without the photometry of the main fragment. """

    _, meteors = loadECSVs([writeEcsv(tmp_path/"a.ecsv", "XX0001")], no_prepare=True, fragment=1)

    assert len(meteors) == 1
    assert len(meteors[0].time_data) == 5
    assert np.isclose(np.degrees(min(meteors[0].elev_data)), 29.0 - 0.3*6)
    assert set(meteors[0].mag_data) == {10.0}


def test_files_without_the_fragment_are_skipped(tmp_path):
    """ Only the stations whose files describe a fragment are loaded for it. """

    paths = [writeEcsv(tmp_path/"a.ecsv", "XX0001"), writeEcsv(tmp_path/"b.ecsv", "XX0002", fragments=False)]

    _, main = loadECSVs(paths, no_prepare=True)
    _, fragment = loadECSVs(paths, no_prepare=True, fragment=1)

    assert sorted(meteor.station_id for meteor in main) == ['XX0001', 'XX0002']
    assert [meteor.station_id for meteor in fragment] == ['XX0001']


def test_previous_format_is_unchanged(tmp_path):
    """ A file without the annotation columns loads all its points, as before. """

    _, meteors = loadECSVs([writeEcsv(tmp_path/"b.ecsv", "XX0002", fragments=False)], no_prepare=True)

    assert len(meteors[0].time_data) == 6
    assert list(meteors[0].mag_data) == [2.0, 1.9, 1.8, 1.7, 1.6, 1.5]


def test_main_fragment_with_the_zero_suffix(tmp_path):
    """ The GFE standard makes the suffix 0 of the main fragment optional (azimuth0 is azimuth). """

    path = writeEcsv(tmp_path/"b.ecsv", "XX0002", fragments=False)
    with open(path) as f:
        lines = f.read().splitlines()

    header = [i for i, line in enumerate(lines) if not line.startswith('#')][0]
    lines[header] = ",".join(name + '0' if name in FRAGMENT else name for name in lines[header].split(','))
    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")

    assert ecsvFragments([path]) == []

    _, meteors = loadECSVs([path], no_prepare=True)
    assert len(meteors[0].time_data) == 6


def test_station_without_points_is_skipped(tmp_path):
    """ A station whose main fragment points are all flagged as not to be used (in any letter case) is 
        skipped, and so is a set of files without any points. """

    path = writeEcsv(tmp_path/"a.ecsv", "XX0001")
    with open(path) as f:
        text = f.read()
    with open(path, 'w') as f:
        f.write(text.replace(',True,', ',false,'))

    jdt_ref, meteors = loadECSVs([path], no_prepare=True)
    assert (jdt_ref is None) and (meteors == [])

    # Fragment 1 is not flagged, so it still loads
    _, meteors = loadECSVs([path], no_prepare=True, fragment=1)
    assert len(meteors) == 1


def test_solver_station_ids():
    """ A station added again gets the suffix _2, _3, ..., as in Trajectory.infillTrajectory(). """

    meteors = [SimpleNamespace(station_id=station_id) for station_id in ('A', 'B', 'A', 'A', 'B')]

    assert solverStationIDs(meteors) == ['A', 'B', 'A_2', 'A_3', 'B_2']


def test_applied_time_offsets_of_a_station_with_two_files(tmp_path):
    """ The offset applied to every observation is matched to its own input file, also when a station has
        two (e.g. one per video file), which the solver names XX0001 and XX0001_2. """

    path_a = writeEcsv(tmp_path/"a.ecsv", "XX0001", fragments=False)
    path_b = writeEcsv(tmp_path/"b.ecsv", "XX0001", fragments=False)

    # The second file has a point less, at other times
    with open(path_b) as f:
        lines = f.read().splitlines()
    with open(path_b, 'w') as f:
        f.write("\n".join(line.replace('2026-01-30T10:25:36.', '2026-01-30T10:25:37.') for line in lines[:-1]) 
            + "\n")

    _, meteors = loadECSVs([path_a, path_b], no_prepare=True)

    # Solved observations, with the times of the two files shifted by 0.25 s and -0.1 s
    observations = [SimpleNamespace(station_id=station_id, 
        JD_data=meteor.jdt_ref + (np.array(meteor.time_data) + shift)/86400) 
        for station_id, meteor, shift in zip(['XX0001', 'XX0001_2'], meteors, [0.25, -0.1])]

    offsets = appliedTimeOffsets(SimpleNamespace(observations=observations), meteors)

    assert sorted(offsets) == sorted([path_a, path_b])
    assert np.isclose(offsets[path_a], 0.25, atol=1e-4)
    assert np.isclose(offsets[path_b], -0.1, atol=1e-4)


def test_original_picks_trajectory(tmp_path):
    """ With the Monte Carlo runs the solver returns the best run, and the solution with the original picks is 
        the one saved with the results. """

    best = SimpleNamespace(uncertainties=SimpleNamespace(), save_results=True, output_dir=str(tmp_path), 
        file_name="20260130_102535", name="best run")
    savePickle(SimpleNamespace(name="original picks"), str(tmp_path), "20260130_102535_trajectory.pickle")

    assert originalPicksTrajectory(best).name == "original picks"

    # Without the Monte Carlo runs the solver returns the solution with the original picks itself
    no_mc = SimpleNamespace(**dict(best.__dict__, uncertainties=None))
    assert originalPicksTrajectory(no_mc) is no_mc

    # Without the saved results, there is no other solution than the given one
    not_saved = SimpleNamespace(**dict(best.__dict__, save_results=False))
    assert originalPicksTrajectory(not_saved) is not_saved


def test_fragment_comparison_says_how_the_initial_velocities_were_estimated():
    """ With the drag fit of the initial velocity asked for, the comparison says which method every solution 
        used, as the fit starts at the first point of each one and can be rejected for a fragment (e.g. born
        below its height limit), and warns when the main fragment used the other one. """

    def solution(t0, v_init, **kwargs):
        """ A solution along a straight vertical path at 20 km/s, seen from t0 for 1 s. """

        t = t0 + np.arange(0, 1.0, 0.1)
        model_eci = np.array([[6.4e6 + 1e5 - 2e4*ti, 0.0, 0.0] for ti in t])
        obs = SimpleNamespace(JD_data=2461000.5 + t/86400, model_eci=model_eci, time_data=t, 
            ignore_list=np.zeros(len(t), dtype=int), ignore_station=False)

        return SimpleNamespace(observations=[obs], radiant_eci_mini=np.array([1.0, 0.0, 0.0]), 
            state_vect_mini=np.array([6.5e6, 0.0, 0.0]), orbit=None, uncertainties=None, v_init=v_init, 
            rbeg_ele=1e5 - 2e4*t0, rend_ele=1e5 - 2e4*(t0 + 0.9), rbeg_jd=obs.JD_data[0], **kwargs)

    drag_fit = SimpleNamespace(n_points=10, t_range=(0.0, 0.9), ht_range=(82000.0, 100000.0))
    main = solution(0.0, 20500.0, v_init_drag=True, v_init_drag_fit=drag_fit, v_init_drag_rejection=None)
    fragment = solution(0.5, 19000.0, v_init_drag=True, v_init_drag_fit=None, 
        v_init_drag_rejection="no points in the fitted part")

    report = fragmentComparison(main, [(1, fragment)])

    assert "Initial velocity of the main fragment from the drag and ablation fit to 10 points, 0.00 to 0.90 s, " \
        "100.00 to 82.00 km" in report
    assert "Initial velocity from : the straight line, as the drag fit was not used: no points in the fitted " \
        "part" in report
    assert "NOTE: not the method of the main fragment" in report

    # Without the drag fit, nothing is said about the method
    plain = [solution(t0, v) for t0, v in [(0.0, 20500.0), (0.5, 19000.0)]]
    report = fragmentComparison(plain[0], [(1, plain[1])])

    assert ("Initial velocity from" not in report) and ("Initial velocity of the main" not in report)
