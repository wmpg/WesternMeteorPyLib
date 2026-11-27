""" Tests for MetSim GUI integration with the drag-coupled trajectory solver (pre-atmospheric velocity
    seeding, saveUpdatedOrbit report consistency, and drag-fit plot overlays).

Run with pytest:
    python -m pytest wmpl/MetSim/Tests/test_GUIDragFitIntegration.py -v
"""

from __future__ import print_function, division, absolute_import

import contextlib
import io
import os
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

pytest.importorskip('PyQt5')

from wmpl.MetSim.GUI import MetSimGUI
from wmpl.MetSim.MetSimErosion import Constants
from wmpl.Trajectory.Tests.SyntheticTrajectory import DragMeteor, STATIONS, observeMeteor



def _solvedFallTrajectory():
    """ Solve a synthetic meteorite-dropping fireball with a non-zero pre-atmospheric drag correction.

    Return:
        traj: [Trajectory object] Solved trajectory with drag_timing_fit and dv_pre_atm > 0.
    """

    # Start the visible track below 60 km so the gated pre-atm air-column correction is applied
    meteor = DragMeteor(14500.0, 120.0, 45.0, 0.002, 0.004, 5.8, 11.0)
    traj = observeMeteor(meteor, noise=4.0, stations=STATIONS, gravity_correction=False, max_toffset=5.0, \
        calc_orbit=True)
    with contextlib.redirect_stdout(io.StringIO()):
        traj = traj.run()

    return traj



def testVelocityPlotAndCommonFeaturesOverlayDragFit():
    """ Check that updateVelocityPlot() draws the smooth drag-fit curve and updateCommonPlotFeatures() marks
        peak dynamic pressure, detected K changes and unobserved gaps without breaking legacy trajectories.
    """

    traj = _solvedFallTrajectory()
    const = Constants()
    const.erosion_on = False
    const.disruption_on = False
    const.erosion_height_change = 0.0

    # Prepare offscreen matplotlib figures matching the GUI canvas wrapper interface
    fig_vel, ax_vel = plt.subplots()
    fig_mag, ax_mag = plt.subplots()
    ax_mag.set_ylim([20.0, 65.0])
    ax_mag.set_xlim([-12.0, 2.0])
    ax_mag.invert_xaxis()

    vel_canvas = SimpleNamespace(axes=ax_vel, figure=fig_vel, draw=lambda: None)
    gui = SimpleNamespace(
        traj=traj,
        const=const,
        simulation_results=None,
        simulation_results_prev=None,
        plot_beg_ht=60.0,
        plot_end_ht=20.0,
        velocityPlot=SimpleNamespace(canvas=vel_canvas),
    )
    gui.updateCommonPlotFeatures = lambda ax, sr, plot_text=False: \
        MetSimGUI.updateCommonPlotFeatures(gui, ax, sr, plot_text=plot_text)

    # Render velocity plot and verify the drag-fit curve is drawn alongside observed picks
    MetSimGUI.updateVelocityPlot(gui)
    vel_labels = [line.get_label() for line in ax_vel.get_lines()]
    assert 'Drag fit' in vel_labels

    # Render common features on magnitude plot with text labels enabled and check Peak P_dyn annotation
    MetSimGUI.updateCommonPlotFeatures(gui, ax_mag, None, plot_text=True)
    mag_texts = [txt.get_text() for txt in ax_mag.texts]
    assert any('Peak P_dyn' in t for t in mag_texts)

    # Verify legacy trajectories without drag_timing_fit still render cleanly
    traj_legacy = SimpleNamespace(
        observations=traj.observations,
        orbit=traj.orbit,
        rbeg_ele=traj.rbeg_ele,
        rend_ele=traj.rend_ele,
    )
    gui.traj = traj_legacy
    MetSimGUI.updateVelocityPlot(gui)
    vel_labels_legacy = [line.get_label() for line in ax_vel.get_lines()]
    assert 'Drag fit' not in vel_labels_legacy

    plt.close(fig_vel)
    plt.close(fig_mag)



def testSaveUpdatedOrbitClearsPreAtmCorrection(tmp_path):
    """ Check that saveUpdatedOrbit() clears the solver's prior dv_pre_atm and marks pre_atm_info as replaced
        by MetSim so report_sim.txt matches the recomputed orbit.

    Arguments:
        tmp_path: [Path] Temporary directory provided by pytest.
    """

    traj = _solvedFallTrajectory()
    assert traj.dv_pre_atm > 0.0
    assert traj.pre_atm_info['applied']

    # Build a minimal stand-in GUI with a MetSim-fitted entry speed 120 m/s above traj.orbit.v_init
    const = Constants()
    const.v_rotation = 0.0
    const.v_init = traj.orbit.v_init + 120.0

    traj_path = os.path.join(str(tmp_path), 'event_trajectory.pickle')
    gui = SimpleNamespace(traj=traj, const=const, traj_path=traj_path, usg_data=None)

    with contextlib.redirect_stdout(io.StringIO()):
        MetSimGUI.saveUpdatedOrbit(gui)

    # Verify report_sim.txt records that the prior air-column correction was replaced by MetSim
    report_path = os.path.join(str(tmp_path), 'event_report_sim.txt')
    assert os.path.isfile(report_path)
    with open(report_path) as f:
        report_txt = f.read()

    assert 'replaced by MetSim simulation velocity' in report_txt
    assert '(applied to orbit)' not in report_txt
    assert 'pre-atm drag corr' not in report_txt
