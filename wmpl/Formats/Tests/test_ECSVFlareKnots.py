""" Tests for the flares of ECSV files in the drag fit of the solver: the main fragment keeps the time of the
    first point of every flare, also of a point not used in its trajectory, and the solver puts a knot of K
    there.

Run with pytest:
    python -m pytest wmpl/Formats/Tests/test_ECSVFlareKnots.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np

from wmpl.Formats.ECSV import flareHeights, loadECSVs
from wmpl.Formats.GenericFunctions import solveTrajectoryGeneric
from wmpl.Formats.Tests.test_ECSVUnsolvedFragments import FPS, FRAGMENT_START, writeEvent


# Frame of the first flare
FLARE = FRAGMENT_START



def flagFrames(paths, station, flags, unused=()):
    """ Flag frames of a station of a noiseless event as flares, and leave some out of its trajectory.

    Arguments:
        paths: [list] Paths of the ECSV files of the event, from writeEvent().
        station: [str] Code of the station.
        flags: [dict] {frame: used in the trajectory} for every frame flagged as a flare.

    Keyword arguments:
        unused: [tuple] Frames left out of the trajectory without a flare. Empty by default.
    """

    path = [path for path in paths if path.endswith(station + '.ecsv')][0]
    with open(path) as f:
        lines = f.read().splitlines()

    for n, line in enumerate(lines):

        fields = line.split(',')
        if line.startswith('#') or (fields[0] == 'datetime'):
            continue

        frame = int(fields[8])
        if frame in flags:
            fields[9:11] = ['True', str(flags[frame])]
        elif frame in unused:
            fields[10] = 'False'

        lines[n] = ','.join(fields)

    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")



def solve(paths, dir_path, **kwargs):
    """ Solve the main fragment of an event without the Monte Carlo runs.

    Arguments:
        paths: [list] Paths of the ECSV files.
        dir_path: [str] Output directory.

    Keyword arguments:
        kwargs: Options of the solver.

    Return:
        [Trajectory object] The solved trajectory.
    """

    jdt_ref, meteor_list = loadECSVs(paths)

    return solveTrajectoryGeneric(jdt_ref, meteor_list, str(dir_path), solver='original', monte_carlo=False, \
        save_results=False, show_plots=False, verbose=False, calc_orbit=False, gravity_correction=False, \
        estimate_timing_vel=False, **kwargs)



def testTheMainFragmentKeepsTheTimesOfTheFirstPointsOfItsFlares(tmp_path):
    """ Check that a flare on three consecutive points gives the time of its first point, also when that
        point is not used in the trajectory, in the times of the points after they are normalized to the
        first point of the event, which is 3 frames before the first used point of XX0001.
    """

    paths = writeEvent(tmp_path)
    flagFrames(paths, 'XX0002', {FLARE: False, FLARE + 1: True, FLARE + 2: True, FLARE + 8: True})
    flagFrames(paths, 'XX0001', {FLARE + 4: True}, unused=(0, 1, 2))

    _, meteor_list = loadECSVs(paths)
    meteors = {meteor.station_id: meteor for meteor in meteor_list}


    def frameTime(meteor, frame):
        """ Time of a frame of a station, from its points, which are on every frame. """

        k = int(np.argmin(np.abs(meteor.frame_numbers - (frame + 3))))
        return meteor.time_data[k] - (meteor.frame_numbers[k] - frame)/FPS


    assert np.allclose(meteors['XX0002'].flare_times, [frameTime(meteors['XX0002'], FLARE), \
        frameTime(meteors['XX0002'], FLARE + 8)], atol=1e-6)
    assert np.allclose(meteors['XX0001'].flare_times, [frameTime(meteors['XX0001'], FLARE + 4)], atol=1e-6)
    assert meteors['XX0003'].flare_times is None



def testTheSolverPutsAKnotAtTheFlaresOfTheFiles(tmp_path):
    """ Check that the flare of the three stations on the same frame gives the stations one knot of K at the
        height of the flare, and none with flare_knots=False.
    """

    paths = writeEvent(tmp_path)
    for station in ('XX0001', 'XX0002', 'XX0003'):
        flagFrames(paths, station, {FLARE + 4: True})

    traj = solve(paths, tmp_path)
    assert all(len(obs.flare_index) == 1 for obs in traj.observations)

    info = traj.flare_knot_info
    assert info['stations'] == [['XX0001', 'XX0002', 'XX0003']]

    # The knot is at the height of the flare points on the trajectory
    ht_flare = np.mean([flare['ht'] for flare in flareHeights(traj, paths)])
    assert abs(info['hts'][0] - ht_flare) < 20.0

    traj_free = solve(paths, tmp_path, flare_knots=False)
    assert traj_free.flare_knot_info is None
    assert not np.any(traj_free.drag_timing_fit.is_knot_ramp)
