""" Tests for the distance at which a passage by the Sun is listed as a close encounter, and for the
closer one at which it counts in the WHFast warning.

The listing limit is the Sun-approaching limit of Borovicka et al. (2005), q < 0.2 AU. The warning
limit stays at 0.1 AU, inside which WHFast's fixed step was measured to fail. They are skipped when
rebound/reboundx are not importable.
"""

import os
import importlib.util

import pytest


REBOUND_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "REBOUND.py")


@pytest.fixture(scope="module")
def reb():
    """ Load REBOUND.py under an isolated module name, skipping if REBOUND is unavailable. """

    spec = importlib.util.spec_from_file_location("_wmpl_test_rebound_sun_limit", REBOUND_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not module.REBOUND_FOUND:
        pytest.skip("rebound/reboundx not importable")

    return module


def testTheSunIsListedInsideTheSunApproachingLimit(reb):
    """ A Geminid-like passage at q = 0.14 AU is inside 0.2 AU and is listed; one at 0.25 AU is not. """

    assert reb.SUN_ENCOUNTER_AU == 0.2

    encounters = reb.encountersFromMinDistances({"Sun": 0.14}, {"Sun": -400.0})

    assert [e["body"] for e in encounters] == ["Sun"]
    assert encounters[0]["min_dist_au"] == 0.14

    assert reb.encountersFromMinDistances({"Sun": 0.25}, {"Sun": -400.0}) == []


def _sunDiagnostics(q_au, t_from=-50.0, t_passage=-900.0):
    """ Per-particle diagnostics with one Sun passage at q_au, after WHFast took over at t_from. """

    return {"fixed_step_from_days": t_from,
            "min_dist_au": {"Sun": q_au},
            "min_time_days": {"Sun": t_passage},
            "encounters": [{"body": "Sun", "min_dist_au": q_au, "time_days": t_passage,
                            "hill_radius_au": None, "n_hill": None, "index": None}]}


def testWhfastWarningCountsOnlySunPassagesInsideItsOwnLimit(reb):
    """ A Sun passage between 0.1 and 0.2 AU is listed but WHFast handles it, so it raises no
        warning; one inside 0.1 AU does, and the warning quotes that limit.
    """

    assert reb.WHFAST_SUN_WARNING_AU == 0.1 < reb.SUN_ENCOUNTER_AU

    assert reb.whfastEncounterWarning({"obj": _sunDiagnostics(0.14)}) is None
    assert reb.whfastEncounterWarning({"obj": _sunDiagnostics(0.1)}) is None

    warning = reb.whfastEncounterWarning({"obj": _sunDiagnostics(0.07),
                                          "mc_0": _sunDiagnostics(0.14)})

    assert "1 close encounter" in warning
    assert "Sun < 0.10 AU" in warning
