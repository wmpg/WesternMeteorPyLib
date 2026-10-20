""" Tests for the orbital class the REBOUND report gives the final orbit: the meteoroid orbital class
of Borovicka et al. (2005), imported from wmpl.Utils.OrbitClassification rather than recomputed.

They are skipped when rebound/reboundx are not importable.
"""

import os
import importlib.util

import numpy as np
import pytest

from wmpl.Utils import OrbitClassification


REBOUND_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "REBOUND.py")


@pytest.fixture(scope="module")
def reb():
    """ Load REBOUND.py under an isolated module name, skipping if REBOUND is unavailable. """

    spec = importlib.util.spec_from_file_location("_wmpl_test_rebound_orbit_class", REBOUND_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not module.REBOUND_FOUND:
        pytest.skip("rebound/reboundx not importable")

    return module


def testTheClassifierIsTheOneOfOrbitClassification(reb):
    """ The report uses the classifier of OrbitClassification, not a copy of it. """

    assert reb.classifyBorovicka is OrbitClassification.classifyBorovicka


@pytest.mark.parametrize("name,a,e,inc_deg,expected", [
    ("1 Ceres", 2.7675, 0.0758, 10.59, 'asteroidal-chondritic'),
    ("67P/Churyumov-Gerasimenko", 3.4630, 0.6410, 7.04, 'jupiter-family'),
    ("1P/Halley", 17.834, 0.96714, 162.26, 'halley-type'),
    ("3200 Phaethon", 1.2712, 0.8898, 22.26, 'sun-approaching'),
])
def testClassOfKnownBodies(reb, name, a, e, inc_deg, expected):
    """ The class from (a, e, i), as REBOUND gives the final orbit, is the class from (q, e, i). """

    assert reb.borovickaClass(a, e, np.radians(inc_deg)) == expected, name


def testClassIsDefinedForAnUnboundOrbit(reb):
    """ REBOUND gives a hyperbolic orbit a negative semi-major axis. The Tisserand parameter is not
        defined there, but the class is: a = -3 AU, e = 1.2 is q = 0.6 AU, Halley type.
    """

    assert reb.tisserandParameterJupiter(-3.0, 1.2, np.radians(30.0)) is None
    assert reb.borovickaClass(-3.0, 1.2, np.radians(30.0)) == 'halley-type'

    # The same hyperbola with its perihelion inside 0.2 AU is Sun-approaching
    assert reb.borovickaClass(-0.5, 1.2, np.radians(30.0)) == 'sun-approaching'
