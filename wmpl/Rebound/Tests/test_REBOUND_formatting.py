""" Tests for how the REBOUND close-encounter tables print distances and epochs.

They need neither REBOUND nor REBOUNDx installed. They are in a file of their own, not at the end of
test_REBOUND_numerics.py, where every change that adds tests appends and the changes then conflict.
"""

import os
import importlib.util

import pytest


REBOUND_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "REBOUND.py")


@pytest.fixture(scope="module")
def reb():
    """ Load REBOUND.py under an isolated module name. """

    spec = importlib.util.spec_from_file_location("_wmpl_test_rebound_formatting", REBOUND_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def testEncounterTablesUseCompactDistancesAndTimes(reb):
    """ Distances keep 3 significant figures in AU and km, and epochs use the unit that suits them. """

    assert reb.formatDistance(0.018503) == "0.0185 AU (2.77 Mkm)"
    assert reb.formatDistance(0.0024430) == "0.00244 AU (365,000 km)"
    assert reb.formatDistance(26.799538) == "26.8 AU (4,010 Mkm)"
    assert reb.formatDistance(1.863751) == "1.86 AU (279 Mkm)"
    assert reb.formatDistance(0.0) == "0 AU (0 km)"

    assert reb.formatTimeStamp(-106537.3739) == "-291.7 yr"
    assert reb.formatTimeStamp(-1.0e6) == "-2738 yr"
    assert reb.formatTimeStamp(45.31) == "+45.3 d"
    assert reb.formatTimeStamp(-0.25) == "-6.0 h"
    assert reb.formatTimeStamp(0.0) == "0 d"
