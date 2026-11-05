""" Tests for the Tisserand parameter and the orbital class of the clones at the end of the run, and
for the lines of the report that give them with the nominal's.

Pure numerics, so they run without REBOUND or REBOUNDx installed.
"""

import os
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest


REBOUND_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "REBOUND.py")


@pytest.fixture(scope="module")
def reb():
    """ Load REBOUND.py under an isolated module name. """

    spec = importlib.util.spec_from_file_location("_wmpl_test_rebound_class_summary", REBOUND_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def _orbit(a, e, inc_deg):
    return SimpleNamespace(a=a, e=e, inc=np.radians(inc_deg))


def testSummaryOfTheClones(reb):
    """ T_J of each clone, its 1 sigma from the 95% interval as for the elements, and the fraction of
    the clones in each class, the most frequent first. """

    rng = np.random.default_rng(3)
    orbits = [_orbit(2.2 + 0.05*rng.normal(), 0.62, 0.5) for _ in range(190)]
    orbits += [_orbit(3.6, 0.45, 5.0) for _ in range(10)]   # aphelion beyond 4.5 AU, T_J < 3

    s = reb.tisserandAndClassSummary(orbits)

    t_j = np.array([reb.tisserandParameterJupiter(o.a, o.e, o.inc) for o in orbits])
    lo, hi = np.percentile(t_j, [2.5, 97.5])
    assert s["n"] == 200 and s["n_unbound"] == 0
    assert s["tisserand_95ci"] == pytest.approx([lo, hi])
    assert s["tisserand_std"] == pytest.approx((hi - lo)/2/1.96)
    assert list(s["classes"]) == ["asteroidal-chondritic", "jupiter-family"]
    assert s["classes"]["jupiter-family"] == pytest.approx(0.05)


def testUnboundClonesHaveNoTisserandButAClass(reb):

    orbits = [_orbit(2.2, 0.6, 1.0), _orbit(2.3, 0.6, 1.0), _orbit(-3.0, 1.2, 30.0)]
    s = reb.tisserandAndClassSummary(orbits)

    assert s["n_unbound"] == 1
    assert s["classes"]["halley-type"] == pytest.approx(1/3)


def testReportSaysItIsTheNominalAtTheEndOfTheRun(reb):

    clones = reb.tisserandAndClassSummary([_orbit(2.2 + 0.01*k, 0.62, 0.5) for k in range(20)])
    lines = reb._tisserandAndClassReportLines(3.405940, "asteroidal-chondritic", clones)
    text = "\n".join(lines)

    assert "at the end of the run" in text
    assert "nominal +/- 1 sigma [95% CI] of the 20 clones:" in text
    assert lines[2].startswith("  T_J   = 3.405940 +/- ")
    assert lines[3] == "  class = asteroidal-chondritic (nominal); clones: asteroidal-chondritic 100.0%"


def testReportWithoutClonesOrForAnUnboundNominal(reb):

    lines = reb._tisserandAndClassReportLines(3.4, "asteroidal-chondritic")
    assert lines == ["Tisserand parameter w.r.t. Jupiter and orbital class (Borovicka et al. 2005) at the end "
                     "of the run, nominal:", "  T_J   = 3.400000", "  class = asteroidal-chondritic"]

    clones = reb.tisserandAndClassSummary([_orbit(-3.0, 1.2, 30.0), _orbit(2.2, 0.6, 1.0)])
    text = "\n".join(reb._tisserandAndClassReportLines(None, "halley-type", clones))
    assert "T_J   = undefined (unbound orbit)" in text
    assert "(1 unbound clones have no T_J)" in text
