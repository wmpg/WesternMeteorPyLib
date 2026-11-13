""" Tests for the flares of a station in the report of the flares of ECSV files: its flare points in
    consecutive rows of its file, so that a point which is not flagged separates two flares, whatever the
    frame rate of the camera and how often the points were picked.

Run with pytest:
    python -m pytest wmpl/Formats/Tests/test_ECSVFlareRows.py -v
"""

from __future__ import print_function, division, absolute_import

from wmpl.Formats.ECSV import flareHeights, flareReport, loadECSVs
from wmpl.Formats.GenericFunctions import solveTrajectoryGeneric
from wmpl.Formats.Tests.test_ECSVFlarePressure import flarePoint
from wmpl.Formats.Tests.test_ECSVUnsolvedFragments import FRAGMENT_START, writeEvent


# First flagged frame, a multiple of the steps between the picks used below
FIRST = 6*(FRAGMENT_START//6)



def flaggedEvent(dir_path, flagged, step=1, unpicked=(), frame_numbers=True):
    """ Write a noiseless event with flares flagged on frames of station XX0002, keep only every step-th frame
        of that station, and solve it.

    Arguments:
        dir_path: [Path] Directory of the files, which is made.
        flagged: [list] Frames flagged as flares.

    Keyword arguments:
        step: [int] Keep only the frames of XX0002 which are a multiple of it. 1 by default, for all.
        unpicked: [list] Frames of XX0002 left without a point. Empty by default.
        frame_numbers: [bool] If False, the file of XX0002 does not give the frame numbers. True by default.

    Return:
        (paths, traj): [tuple] Paths of the ECSV files, and the solved trajectory.
    """

    dir_path.mkdir()
    paths = writeEvent(dir_path)

    path = [path for path in paths if path.endswith('XX0002.ecsv')][0]
    with open(path) as f:
        lines = f.read().splitlines()

    # Keep the picked frames of XX0002 and flag the flares
    kept = []
    for line in lines:

        fields = line.split(',')
        if line.startswith('#'):
            kept.append(line)
            continue

        # Rename the column of the frame numbers to leave them out
        if fields[0] == 'datetime':
            kept.append(line if frame_numbers else line.replace('frame_number', 'frame_index'))
            continue

        # Keep only the picked frames, with their flags
        frame = int(fields[8])
        if (frame%step) or (frame in unpicked):
            continue

        if frame in flagged:
            fields[9] = 'True'

        kept.append(','.join(fields))

    with open(path, 'w') as f:
        f.write("\n".join(kept) + "\n")

    # Solve the event
    jdt_ref, meteor_list = loadECSVs(paths)
    traj = solveTrajectoryGeneric(jdt_ref, meteor_list, str(dir_path), solver='original', monte_carlo=False, \
        save_results=False, show_plots=False, verbose=False, calc_orbit=True, gravity_correction=False, \
        estimate_timing_vel=False)

    return paths, traj



def flareCount(dir_path, flagged, **kwargs):
    """ Count the flares the report gives for the event of flaggedEvent().

    Arguments:
        dir_path: [Path] Directory of the files, which is made.
        flagged: [list] Frames flagged as flares.

    Keyword arguments:
        kwargs: [dict] Options of flaggedEvent().

    Return:
        [int] Number of flares in the report.
    """

    paths, traj = flaggedEvent(dir_path, flagged, **kwargs)

    return flareReport(flareHeights(traj, paths)).count("point(s)")



def testAPointNotFlaggedSeparatesTwoFlares(tmp_path):
    """ Check that a point which is not flagged between flagged ones of a 25 fps camera separates two flares,
        although they are only 80 ms apart, and that flagged consecutive frames are one flare.
    """

    assert flareCount(tmp_path/"apart", [FIRST, FIRST + 1, FIRST + 3]) == 2
    assert flareCount(tmp_path/"together", [FIRST, FIRST + 1, FIRST + 2]) == 1



def testFlaresOfPointsPickedOnFewerFramesAreOneFlare(tmp_path):
    """ Check that flagged points picked every second or third frame, with no point between them, are one
        flare, although they are 80 and 120 ms apart.
    """

    assert flareCount(tmp_path/"second", [FIRST, FIRST + 2, FIRST + 4], step=2) == 1
    assert flareCount(tmp_path/"third", [FIRST, FIRST + 3, FIRST + 6], step=3) == 1



def testAFrameWithoutAPointInAFlareDoesNotSeparateIt(tmp_path):
    """ Check that two frames left without a point inside a flare, e.g. as it saturated, do not separate it,
        as no point which is not flagged lies between its points, 120 ms apart on a camera picked on every
        frame.
    """

    assert flareCount(tmp_path/"gap", [FIRST, FIRST + 1, FIRST + 4], unpicked=[FIRST + 2, FIRST + 3]) == 1



def testPointsWithoutRowsAreOneFlareWithinOneAndAHalfFrames():
    """ Check that flare points whose rows are not known are one flare when at most 1.5 frames apart, and two
        otherwise.
    """

    assert flareReport([flarePoint('A', 0.50, 0.04), flarePoint('A', 0.54, 0.04)]).count("point(s)") == 1
    assert flareReport([flarePoint('A', 0.50, 0.04), flarePoint('A', 0.58, 0.04)]).count("point(s)") == 2



if __name__ == "__main__":

    testPointsWithoutRowsAreOneFlareWithinOneAndAHalfFrames()

    print("All tests passed.")
