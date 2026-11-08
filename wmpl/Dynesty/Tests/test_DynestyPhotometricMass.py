""" Tests the photometric mass that ObservationData() estimates from the luminous efficiency lines of the
    prior file.

With a luminous efficiency model, the mass is computed by calcMass() with the code of the model at the average
speed, iterated to the mass it gives. The tests run ObservationData() on the example trajectory of the
repository, with the luminous efficiency lines of the base prior file replaced.

Run with pytest:
    python -m pytest wmpl/Dynesty/Tests/test_DynestyPhotometricMass.py -v
"""

from __future__ import print_function, division, absolute_import

import os
import tempfile
import warnings

import numpy as np

import wmpl.Dynesty.DynestyMetSim as DynestyMetSim
from wmpl.Utils.Physics import calcMass


# Solved trajectory shipped with the repository
DYNESTY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_PICKLE = os.path.join(DYNESTY_DIR, "examples", "20191023_091225", "20191023_091225_trajectory.pickle")

# Prior file to which the luminous efficiency lines are added
BASE_PRIOR = os.path.join(DYNESTY_DIR, "priors", "stony_meteoroid_SingleBody.prior")



def _photometricMass(lum_eff_lines):
    """ Run ObservationData() on the example trajectory with the given luminous efficiency lines.

    Arguments:
        lum_eff_lines: [list of str] Lines of the prior file that set the luminous efficiency.

    Return:
        (m_init, calls):
            - m_init: [float] Photometric mass estimated by ObservationData() (kg).
            - calls: [list] Arguments, keyword arguments and result of each call of calcMass().
    """

    with open(BASE_PRIOR) as f:
        lines = [line for line in f if not line.startswith("lum_eff")]

    # Record the calls of calcMass()
    calls = []

    def recordedCalcMass(*args, **kwargs):
        mass = calcMass(*args, **kwargs)
        calls.append((args, kwargs, mass))
        return mass

    calc_mass = DynestyMetSim.calcMass
    DynestyMetSim.calcMass = recordedCalcMass

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:

            prior_path = os.path.join(tmp_dir, "test.prior")
            with open(prior_path, "w") as f:
                f.writelines(lines + [line + "\n" for line in lum_eff_lines])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m_init = DynestyMetSim.ObservationData([EXAMPLE_PICKLE], prior_file_path=prior_path).m_init

    finally:
        DynestyMetSim.calcMass = calc_mass

    return m_init, calls



def testTheModelIsEvaluatedAtTheAverageSpeed():
    """ Check that a luminous efficiency model is passed to calcMass() by its code with the average speed, and
        that the mass is the one the model gives at that mass.
    """

    m_init, calls = _photometricMass(["lum_eff_type, 5, fix"])

    # The last call computes the mass with the model
    (time_arr, mag_arr, velocity), kwargs, mass = calls[-1]

    assert kwargs["tau"] == 5
    assert np.ndim(velocity) == 0
    assert mass == m_init

    # Check that the mass is self-consistent
    mass_at_mass = calcMass(time_arr, mag_arr, velocity, tau=5, P_0m=kwargs["P_0m"], lum_eff_mass=mass)

    assert np.isclose(mass_at_mass, mass, rtol=1e-3)



def testTheConstantModelWithoutAValueTakesTheDefault():
    """ Check that the constant model fixed without a lum_eff line takes the default 0.7 %. Before, it failed
        dividing None.
    """

    mass_default, _ = _photometricMass(["lum_eff_type, 0, fix"])
    mass_explicit, _ = _photometricMass(["lum_eff_type, 0, fix", "lum_eff, 0.7, fix"])
    mass_double, _ = _photometricMass(["lum_eff_type, 0, fix", "lum_eff, 1.4, fix"])

    assert mass_default == mass_explicit
    assert np.isclose(mass_double, mass_default/2, rtol=1e-9)
