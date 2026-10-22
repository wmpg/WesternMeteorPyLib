""" Tests that the MetSim GUI computes the photometric mass with the luminous efficiency chosen in it (the constant
value or the lum_eff_type model), and rescales the initial mass when the model is changed.

The GUI methods are called on a stand-in object with the attributes they use, so no window is opened. PyQt5 is
still needed to import the GUI module.

Run under pytest, or directly:

    python -m wmpl.MetSim.Tests.test_GUIPhotometricMass
"""

from types import SimpleNamespace

import numpy as np

try:
    import pytest
    pytest.importorskip("PyQt5")
except ImportError:
    pass

from wmpl.MetSim.GUI import MetSimGUI
from wmpl.MetSim.MetSimErosion import Constants
from wmpl.Utils.Math import mergeClosePoints
from wmpl.Utils.Physics import calcMass


# A synthetic light curve from one station, 1 s at 100 frames per second, at 18 km/s
TIME = np.linspace(0, 1, 101)
MAG_ABS = -1 + 4*(TIME - 0.5)**2
V_AVG = 18000.0


def _expectedMass(tau):
    """ calcMass() on the light curve after the GUI's averaging of the points closer than a frame. """

    time_arr, mag_arr = mergeClosePoints(TIME, MAG_ABS, np.median(np.diff(TIME)), method='avg')

    return calcMass(np.array(time_arr), np.array(mag_arr), V_AVG, tau=tau, P_0m=1500.0)


def _standIn(lum_eff_type=0, lum_eff=0.7, with_photometry=True):
    """ An object with what calcPhotometricMass() and updateLumEffType() use. """

    obs = SimpleNamespace(time_data=TIME, absolute_magnitudes=MAG_ABS if with_photometry else None,
        ignore_list=np.zeros(len(TIME), dtype=int))
    traj = SimpleNamespace(observations=[obs], orbit=SimpleNamespace(v_avg=V_AVG))

    const = Constants()
    const.lum_eff_type, const.lum_eff, const.P_0m = lum_eff_type, lum_eff, 1500.0

    gui = SimpleNamespace(met_obs=None, ecsv_obs=None, lc_data=None, traj=traj, const=const)
    gui.calcPhotometricMass = lambda: MetSimGUI.calcPhotometricMass(gui)
    gui.inputLumEff = SimpleNamespace(setDisabled=lambda disabled: None)
    gui.updateInputBoxes = lambda: None

    return gui


def test_photometric_mass_uses_the_chosen_model():
    """ The constant luminous efficiency for type 0, and the model of lum_eff_type otherwise. """

    _, mass_constant = MetSimGUI.calcPhotometricMass(_standIn(lum_eff_type=0, lum_eff=0.7))
    assert np.isclose(mass_constant, _expectedMass(0.007), rtol=1e-12)

    for lum_eff_type in range(1, 9):
        _, mass_model = MetSimGUI.calcPhotometricMass(_standIn(lum_eff_type=lum_eff_type))
        assert np.isclose(mass_model, _expectedMass(lum_eff_type),
            rtol=1e-12), lum_eff_type

    # The models do change the mass (RC2001 Type I against Type III)
    assert MetSimGUI.calcPhotometricMass(_standIn(lum_eff_type=3))[1] > \
        5*MetSimGUI.calcPhotometricMass(_standIn(lum_eff_type=1))[1]


def test_changing_the_model_rescales_the_initial_mass():
    """ Choosing another model scales the initial mass by the ratio of the photometric masses, keeping any change
        the user made to it, as a new constant luminous efficiency does. """

    gui = _standIn(lum_eff_type=0)
    mass_constant = MetSimGUI.calcPhotometricMass(gui)[1]
    gui.const.m_init = 2*mass_constant

    gui.lumEffComboBox = SimpleNamespace(currentIndex=lambda: 1)
    MetSimGUI.updateLumEffType(gui)

    assert gui.const.lum_eff_type == 1
    mass_rc2001 = MetSimGUI.calcPhotometricMass(gui)[1]
    assert np.isclose(gui.const.m_init, 2*mass_rc2001, rtol=1e-12)
    assert not np.isclose(mass_rc2001, mass_constant, rtol=1e-2)

    # The same model again changes nothing
    m_init = gui.const.m_init
    MetSimGUI.updateLumEffType(gui)
    assert gui.const.m_init == m_init


def test_changing_the_model_without_photometry_keeps_the_initial_mass():

    gui = _standIn(lum_eff_type=0, with_photometry=False)
    m_init = gui.const.m_init

    gui.lumEffComboBox = SimpleNamespace(currentIndex=lambda: 4)
    MetSimGUI.updateLumEffType(gui)

    assert gui.const.lum_eff_type == 4
    assert gui.const.m_init == m_init


if __name__ == "__main__":
    test_photometric_mass_uses_the_chosen_model()
    test_changing_the_model_rescales_the_initial_mass()
    test_changing_the_model_without_photometry_keeps_the_initial_mass()
    print("All GUIPhotometricMass checks passed.")
