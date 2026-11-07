""" Tests for the uncertainty of the speed of the flares of ECSV files: the Monte Carlo runs only move the
    radiant, so every run shifts the speed of the drag fit within its uncertainty from the fit, and the
    dynamic pressure of the run follows.

Run with pytest:
    python -m pytest wmpl/Formats/Tests/test_ECSVFlareSpeedUncertainty.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np

from wmpl.Formats.ECSV import flareHeights, flareReport
from wmpl.Formats.Tests.test_ECSVFlarePressure import flaggedEvent
from wmpl.Formats.Tests.test_ECSVUnsolvedFragments import FRAGMENT_START, monteCarloRuns
from wmpl.Utils.DragFit import DragFit



def testTheRunsShiftTheSpeedWithinTheUncertaintyOfTheDragFit(tmp_path):
    """ Check that the speed of a flare scatters in the Monte Carlo runs as much as the uncertainty of the
        drag fit, which the runs alone do not change, and that the dynamic pressure follows it.
    """

    paths, traj = flaggedEvent(tmp_path, FRAGMENT_START + 5)

    np.random.seed(0)
    flare = flareHeights(traj, paths, mc_trajs=monteCarloRuns(traj, n=200))[0]

    speed_std = DragFit(traj).motionStdAtTime(flare['t_rel'])[0][0]
    assert np.isfinite(speed_std) and (speed_std > 0)
    assert 0.8 < flare['v_std_mc']/speed_std < 1.2

    # The frame adds to it, and the pressure, proportional to the square of the speed, scatters at least
    #   twice as much in relative terms
    assert flare['v_std'] >= flare['v_std_mc']
    assert flare['p_std_mc']/flare['p'] > 0.8*2*flare['v_std_mc']/flare['v']



def testTheReportGivesTheSpeedAndPressureOfEveryPoint(tmp_path):
    """ Check that the report gives the speed with its uncertainty, and the speed and the pressure of every
        flare point in the table of the points.
    """

    paths, traj = flaggedEvent(tmp_path, FRAGMENT_START + 5)

    np.random.seed(0)
    flare = flareHeights(traj, paths, mc_trajs=monteCarloRuns(traj))[0]
    report = flareReport([flare])

    speed_str = "{:.3f} +/- {:.3f}".format(flare['v']/1000, flare['v_std']/1000)
    pressure_str = "{:.1f} +/- {:.1f}".format(flare['p']/1000, flare['p_std']/1000)

    assert ", v = {:s} km/s, p = {:s} kPa".format(speed_str, pressure_str) in report

    points = report.split("Flare points:")[1]
    assert "v (km/s)" in points and "p (kPa)" in points
    assert (speed_str in points) and (pressure_str in points)



def testWithoutTheMonteCarloRunsTheSpeedHasNoUncertainty(tmp_path):
    """ Check that the speed has no uncertainty without the Monte Carlo runs, as the other values. """

    paths, traj = flaggedEvent(tmp_path, FRAGMENT_START + 5)
    flare = flareHeights(traj, paths)[0]

    assert (flare['v'] is not None) and (flare['v_std'] is None) and (flare['v_std_mc'] is None)
    assert "+/-" not in flareReport([flare]).split("Flare points:")[1]
