""" Tests for the duration of a frame of the flares of ECSV files, which gives the uncertainty of their times:
    from the frame numbers of the file, so that it is the duration of one frame also when not every frame was
    picked.

Run with pytest:
    python -m pytest wmpl/Formats/Tests/test_ECSVFlareFrameDuration.py -v
"""

from __future__ import print_function, division, absolute_import

from wmpl.Formats.ECSV import flareHeights
from wmpl.Formats.Tests.test_ECSVFlareRows import FIRST, flaggedEvent
from wmpl.Formats.Tests.test_ECSVUnsolvedFragments import FPS



def testTheFrameDurationComesFromTheFrameNumbers(tmp_path):
    """ Check that a camera picked every second frame gets the duration of one frame from the frame numbers,
        not the time between its points, and that without the frame numbers it gets the time between them.
    """

    paths, traj = flaggedEvent(tmp_path/"numbers", [FIRST], step=2)
    assert abs(flareHeights(traj, paths)[0]['frame_dt'] - 1/FPS) < 1e-4

    paths, traj = flaggedEvent(tmp_path/"none", [FIRST], step=2, frame_numbers=False)
    assert abs(flareHeights(traj, paths)[0]['frame_dt'] - 2/FPS) < 1e-4



def testFramesWithoutAPointDoNotChangeTheFrameDuration(tmp_path):
    """ Check that frames left without a point, e.g. in a saturated flare, do not change the duration of a
        frame of a camera picked on every frame.
    """

    paths, traj = flaggedEvent(tmp_path/"gap", [FIRST], unpicked=[FIRST + 2, FIRST + 3, FIRST + 5])
    assert abs(flareHeights(traj, paths)[0]['frame_dt'] - 1/FPS) < 1e-4
