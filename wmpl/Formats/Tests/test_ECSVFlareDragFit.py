""" Tests for the warning given when flares of ECSV files lie within the drag fit of the initial velocity
    (--vinitdrag), which models a single body without fragmentation.

Run with pytest:
    python -m pytest wmpl/Formats/Tests/test_ECSVFlareDragFit.py -v
"""

from __future__ import print_function, division, absolute_import

import types

from wmpl.Formats.ECSV import dragFitFlareWarning



def dragFitTrajectory(ht_lo, ht_hi):
    """ Make a stand-in for a trajectory whose initial velocity came from the drag fit, with only what the
        warning reads from it.

    Arguments:
        ht_lo: [float] Height of the lowest fitted point (m).
        ht_hi: [float] Height of the highest fitted point (m).

    Return:
        [SimpleNamespace] The trajectory.
    """

    fit = types.SimpleNamespace(ht_range=(ht_lo, ht_hi))

    return types.SimpleNamespace(v_init_drag_fit=fit)



def flarePoint(station_id, ht):
    """ Make a flare point as flareHeights() gives it, with only what the warning reads from it.

    Arguments:
        station_id: [str] Station of the point.
        ht: [float] Height of the point (m).

    Return:
        [dict] The flare point.
    """

    return {'station_id': station_id, 'ht': ht}



def testNoWarningWithoutTheDragFit():
    """ Check that there is no warning without the drag fit, as with the straight-line initial velocity or a
        solver without the --vinitdrag option.
    """

    flares = [flarePoint('XX0001', 90000.0)]

    assert dragFitFlareWarning(types.SimpleNamespace(), flares) == ""
    assert dragFitFlareWarning(types.SimpleNamespace(v_init_drag_fit=None), flares) == ""



def testNoWarningForFlaresBelowTheFit():
    """ Check that flares below the lowest fitted point give no warning. """

    traj = dragFitTrajectory(80000.0, 95000.0)
    flares = [flarePoint('XX0001', 79900.0), flarePoint('XX0002', 70000.0)]

    assert dragFitFlareWarning(traj, flares) == ""



def testWarningForFlaresWithinTheFit():
    """ Check that flares above the lowest fitted point are reported with the highest one of every station,
        and that the suggested height keeps only the points above the highest flare.
    """

    traj = dragFitTrajectory(80000.0, 95000.0)
    flares = [flarePoint('XX0001', 84000.0), flarePoint('XX0001', 85020.0), flarePoint('XX0002', 86030.0), \
        flarePoint('XX0002', 70000.0)]

    warning = dragFitFlareWarning(traj, flares)

    assert "WARNING: 3 flare point(s)" in warning
    assert "(80.00 km)" in warning
    assert "XX0002 at 86.03 km, XX0001 at 85.02 km." in warning
    assert "--vinitdraght 86.1" in warning
    assert "84.00" not in warning



def testWarningForAFlareAtTheLowestFittedPoint():
    """ Check that a flare point which is itself the lowest fitted point is within the fit. """

    traj = dragFitTrajectory(80000.0, 95000.0)

    assert "WARNING: 1 flare point(s)" in dragFitFlareWarning(traj, [flarePoint('XX0001', 80000.0)])



def testWarningForAFlareAboveTheWholeFit():
    """ Check that a flare at or above the first fitted point gives no height to fit above it. """

    traj = dragFitTrajectory(80000.0, 95000.0)

    warning = dragFitFlareWarning(traj, [flarePoint('XX0001', 96000.0)])

    assert "no fitted point is free" in warning
    assert "--vinitdraght" not in warning.split("fragmentation.")[1]



if __name__ == "__main__":

    testNoWarningWithoutTheDragFit()
    testNoWarningForFlaresBelowTheFit()
    testWarningForFlaresWithinTheFit()
    testWarningForAFlareAtTheLowestFittedPoint()
    testWarningForAFlareAboveTheWholeFit()

    print("All tests passed.")
