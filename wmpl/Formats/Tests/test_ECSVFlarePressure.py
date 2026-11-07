""" Tests for the speed and the dynamic pressure of the flares of ECSV files, from the drag fit of the main
    trajectory, and for the comparison of the flares of two cameras which mark the same flare a frame apart.

Run with pytest:
    python -m pytest wmpl/Formats/Tests/test_ECSVFlarePressure.py -v
"""

from __future__ import print_function, division, absolute_import

import sys

import numpy as np

from wmpl.Formats.ECSV import flareHeights, flareReport, loadECSVs
from wmpl.Formats.GenericFunctions import solveTrajectoryGeneric
from wmpl.Formats.Tests.test_ECSVUnsolvedFragments import FRAGMENT_START, SPEED, monteCarloRuns, writeEvent
from wmpl.Utils.AtmosphereDensity import getAtmDensity_vect



def flaggedEvent(dir_path, frame):
    """ Write a noiseless event with a flare flagged on a frame of station XX0002, and solve it with its orbit,
        which gives the Earth's rotation along the path.

    Arguments:
        dir_path: [str] Directory of the files.
        frame: [int] Frame of the flare.

    Return:
        (paths, traj): [tuple] Paths of the ECSV files, and the solved trajectory.
    """

    paths = writeEvent(dir_path)

    path = [path for path in paths if path.endswith('XX0002.ecsv')][0]
    with open(path) as f:
        lines = f.read().splitlines()
    for n, line in enumerate(lines):
        fields = line.split(',')
        if (not line.startswith('#')) and (fields[0] != 'datetime') and (int(fields[8]) == frame):
            fields[9] = 'True'
            lines[n] = ','.join(fields)
    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")

    jdt_ref, meteor_list = loadECSVs(paths)
    traj = solveTrajectoryGeneric(jdt_ref, meteor_list, str(dir_path), solver='original', monte_carlo=False, \
        save_results=False, show_plots=False, verbose=False, calc_orbit=True, gravity_correction=False, \
        estimate_timing_vel=False)

    return paths, traj



def testAFlareGetsTheSpeedRelativeToTheGroundAndTheDynamicPressure(tmp_path):
    """ Check that a flare of a meteor moving at a constant speed relative to the ground gets that speed, not
        the inertial one, and the dynamic pressure of the air density at its height.
    """

    paths, traj = flaggedEvent(tmp_path, FRAGMENT_START + 5)
    flare = flareHeights(traj, paths)[0]

    assert abs(flare['v'] - SPEED) < 20.0

    rho = getAtmDensity_vect(np.array([flare['lat']]), np.array([flare['lon']]), np.array([flare['ht']]), \
        traj.jdt_ref)[0]
    assert abs(flare['p']/(rho*SPEED**2) - 1) < 0.005

    # Without the Monte Carlo runs the pressure has no uncertainty, and the report gives it
    assert flare['p_std'] is None
    report = flareReport([flare])
    assert ", v = {:.3f} km/s, p = {:.1f} kPa".format(flare['v']/1000, flare['p']/1000) in report
    assert "dynamic pressures p = rho*v^2" in report



def testTheMonteCarloRunsGiveTheUncertaintyOfThePressure(tmp_path):
    """ Check that the dynamic pressure of a flare gets an uncertainty from the heights of the Monte Carlo
        runs and from the duration of its frame.
    """

    paths, traj = flaggedEvent(tmp_path, FRAGMENT_START + 5)
    flare = flareHeights(traj, paths, mc_trajs=monteCarloRuns(traj))[0]

    assert flare['p_std_mc'] > 0
    assert flare['p_std'] > flare['p_std_mc']
    assert flare['p_std'] < 0.2*flare['p']



def testWithoutTheDragFitTheFlaresHaveNoPressure(tmp_path, monkeypatch, capsys):
    """ Check that the flares are still given when the drag model cannot be fitted, without their speeds and
        pressures.
    """

    paths, traj = flaggedEvent(tmp_path, FRAGMENT_START + 5)

    def failingFit(traj):
        raise ValueError("no fit")

    monkeypatch.setattr(sys.modules['wmpl.Formats.ECSV'], 'DragFit', failingFit)

    flare = flareHeights(traj, paths)[0]

    assert (flare['v'] is None) and (flare['p'] is None)
    assert "are not computed: no fit" in capsys.readouterr().out
    assert "kPa" not in flareReport([flare])



def flarePoint(station_id, t, frame_dt):
    """ Make a flare point of a meteor descending at 10 km/s, with uncertainties as from flareHeights().

    Arguments:
        station_id: [str] Station ID.
        t: [float] Time from the reference time (s).
        frame_dt: [float] Duration of the frame of the camera (s).

    Return:
        [dict] The flare point.
    """

    frame_std = frame_dt/np.sqrt(12)

    return {'station_id': station_id, 'jd': 2461070.9 + t/86400, 't_rel': t, 'ht': 90000.0 - 10000.0*t, \
        'used': True, 'frame_dt': frame_dt, 'ht_rate': -10000.0, 't_std_mc': 0.0005, 'ht_std_mc': 50.0, \
        't_std': np.hypot(0.0005, frame_std), 'ht_std': np.hypot(50.0, 10000.0*frame_std)}



def testFlaresOneFrameOfTheSlowerCameraApartAreConsistent():
    """ Check that the same flare marked one frame apart by a 13 fps and a 25 fps camera is consistent, and two
        frames apart not.
    """

    one = flareReport([flarePoint('A', 0.5, 0.075), flarePoint('B', 0.575, 0.04)])
    two = flareReport([flarePoint('A', 0.5, 0.075), flarePoint('B', 0.65, 0.04)])

    assert " 1 A -  2 B: 0.075 s apart" in one
    assert "(0.0 and 0.0 sigma): consistent" in one
    assert "NOT consistent" in two



def testTheRealErrorOfTheTimeOffsetsIsAddedWhenTheSolverEstimatedThem():
    """ Check that the real error of the time offsets is added to the uncertainty of the time between the flares
        of two stations when the solver estimated the offset of one of them, and not when the offsets were not
        estimated or both are fixed: two 25 fps cameras 0.091 s apart are 3.1 sigma apart without it, after
        taking off a frame, and 2.9 sigma with it.
    """

    flares = [flarePoint('A', 0.5, 0.04), flarePoint('B', 0.591, 0.04)]

    estimated = flareReport(flares, ref_stations=['A'])
    assert "(2.9 and 0.0 sigma): consistent" in estimated
    assert "real error of\nthe time offsets the solver estimated, 7 ms" in estimated

    for ref_stations in ([], ['A', 'B']):
        report = flareReport(flares, ref_stations=ref_stations)
        assert "(3.1 and 0.0 sigma): NOT consistent" in report
        assert "real error" not in report

    # The same for the heights compared at the same time, which the error of the offsets also moves: 250 m
    #   apart are 3.5 sigma without it and 2.5 sigma with it
    above = flarePoint('B', 0.5, 0.04)
    above['ht'] += 250.0
    flares = [flarePoint('A', 0.5, 0.04), above]

    assert "(0.0 and 2.5 sigma): consistent" in flareReport(flares, ref_stations=['A'])
    assert "(0.0 and 3.5 sigma): NOT consistent" in flareReport(flares, ref_stations=[])
