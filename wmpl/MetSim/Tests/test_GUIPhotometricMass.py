""" Tests that the MetSim GUI computes the photometric mass with the chosen luminous efficiency, and rescales the
initial mass when the model is changed.

The GUI methods are called on a stand-in object with the attributes they use, so no window is opened. PyQt5 is
still needed to import the GUI module, and the tests are skipped without it.

Run with pytest:
    python -m pytest wmpl/MetSim/Tests/test_GUIPhotometricMass.py -v
"""

from __future__ import print_function, division, absolute_import

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from wmpl.MetSim.GUI import MetSimGUI
from wmpl.MetSim.MetSimErosion import Constants
from wmpl.Utils.Math import mergeClosePoints
from wmpl.Utils.Physics import calcMass


# Synthetic light curve from one station, 1 s at 100 frames per second
TIME = np.linspace(0, 1, 101)
MAG_ABS = -1 + 4*(TIME - 0.5)**2

# Average velocity (m/s)
V_AVG = 18000.0



def _expectedMass(tau):
    """ Compute the photometric mass of the light curve after the points closer than a frame are averaged, as
        the GUI does.

    Arguments:
        tau: [float or int] Constant luminous efficiency (ratio) or the code of the model.

    Return:
        mass: [float] Photometric mass (kg).
    """

    time_arr, mag_arr = mergeClosePoints(TIME, MAG_ABS, np.median(np.diff(TIME)), method='avg')

    return calcMass(np.array(time_arr), np.array(mag_arr), V_AVG, tau=tau, P_0m=1500.0)



def _standIn(lum_eff_type=0, lum_eff=0.7, with_photometry=True):
    """ Make an object with the attributes that calcPhotometricMass() and updateLumEffType() use.

    Keyword arguments:
        lum_eff_type: [int] Code of the luminous efficiency model. 0 (constant) by default.
        lum_eff: [float] Constant luminous efficiency (%). 0.7 by default.
        with_photometry: [bool] Give the observation absolute magnitudes. True by default.

    Return:
        gui: [SimpleNamespace] Stand-in for the GUI.
    """

    obs = SimpleNamespace(time_data=TIME, absolute_magnitudes=(MAG_ABS if with_photometry else None), \
        ignore_list=np.zeros(len(TIME), dtype=int))
    traj = SimpleNamespace(observations=[obs], orbit=SimpleNamespace(v_avg=V_AVG))

    const = Constants()
    const.lum_eff_type = lum_eff_type
    const.lum_eff = lum_eff
    const.P_0m = 1500.0

    gui = SimpleNamespace(met_obs=None, ecsv_obs=None, lc_data=None, traj=traj, const=const)
    gui.calcPhotometricMass = lambda: MetSimGUI.calcPhotometricMass(gui)
    gui.inputLumEff = SimpleNamespace(setDisabled=lambda disabled: None)
    gui.updateInputBoxes = lambda: None

    return gui



def testThePhotometricMassUsesTheChosenModel():
    """ Check that the photometric mass is computed with the constant luminous efficiency for type 0, and with
        the model of lum_eff_type otherwise.
    """

    # Check the constant luminous efficiency
    _, mass_constant = MetSimGUI.calcPhotometricMass(_standIn(lum_eff_type=0, lum_eff=0.7))
    assert np.isclose(mass_constant, _expectedMass(0.007), rtol=1e-12)

    # Check every model
    for lum_eff_type in range(1, 9):
        _, mass_model = MetSimGUI.calcPhotometricMass(_standIn(lum_eff_type=lum_eff_type))
        assert np.isclose(mass_model, _expectedMass(lum_eff_type), rtol=1e-12), lum_eff_type

    # Check that the models change the mass, RC2001 Type III against Type I
    assert MetSimGUI.calcPhotometricMass(_standIn(lum_eff_type=3))[1] \
        > 5*MetSimGUI.calcPhotometricMass(_standIn(lum_eff_type=1))[1]



def testChangingTheModelRescalesTheInitialMass():
    """ Check that choosing another model scales the initial mass by the ratio of the photometric masses, so a
        change the user made to it is kept, as with a new constant luminous efficiency.
    """

    # Set the initial mass to twice the photometric mass
    gui = _standIn(lum_eff_type=0)
    mass_constant = MetSimGUI.calcPhotometricMass(gui)[1]
    gui.const.m_init = 2*mass_constant

    # Choose RC2001 Type I
    gui.lumEffComboBox = SimpleNamespace(currentIndex=lambda: 1)
    MetSimGUI.updateLumEffType(gui)

    assert gui.const.lum_eff_type == 1

    mass_rc2001 = MetSimGUI.calcPhotometricMass(gui)[1]
    assert np.isclose(gui.const.m_init, 2*mass_rc2001, rtol=1e-12)
    assert not np.isclose(mass_rc2001, mass_constant, rtol=1e-2)

    # Check that choosing the same model again changes nothing
    m_init = gui.const.m_init
    MetSimGUI.updateLumEffType(gui)

    assert gui.const.m_init == m_init



def testChangingTheModelWithoutPhotometryKeepsTheInitialMass():
    """ Check that the initial mass does not change with the model when there is no photometry.
    """

    gui = _standIn(lum_eff_type=0, with_photometry=False)
    m_init = gui.const.m_init

    # Choose Borovicka 2013
    gui.lumEffComboBox = SimpleNamespace(currentIndex=lambda: 4)
    MetSimGUI.updateLumEffType(gui)

    assert gui.const.lum_eff_type == 4
    assert gui.const.m_init == m_init
