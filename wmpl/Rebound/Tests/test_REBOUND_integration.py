""" Tests for REBOUND version compatibility and a real end-to-end integration.

The compatibility helpers are tested with stand-in objects, so those tests run without REBOUND. The
remaining tests integrate for real and are skipped when rebound/reboundx, the DE430 ephemeris or the
example trajectory are unavailable. They catch what the pure-numerics tests cannot: REBOUND API
changes between major versions, and a reboundx compiled against a different REBOUND version.
"""

import os
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest


REBOUND_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "REBOUND.py")

EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Dynesty",
                           "examples", "20191023_091225")
EXAMPLE_FILE = "20191023_091225_trajectory.pickle"


@pytest.fixture(scope="module")
def reb():
    """ Load REBOUND.py under an isolated module name. """

    spec = importlib.util.spec_from_file_location("_wmpl_test_rebound_integration", REBOUND_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


@pytest.fixture
def realReb(reb):
    """ The REBOUND module, skipping the test if rebound/reboundx cannot be imported. """

    if not reb.REBOUND_FOUND:
        pytest.skip("rebound/reboundx not importable: {}".format(reb._REBOUND_IMPORT_ERROR))

    return reb


### Version compatibility helpers (no REBOUND needed) ###


class _RecordingSim:
    """ Stand-in simulation recording the arguments passed to add(). """

    def __init__(self):
        self.added = []
        self.particles = []

    @property
    def N(self):
        return len(self.particles)

    def add(self, *args, **kwargs):
        self.added.append((args, kwargs))
        self.particles.append(SimpleNamespace())


@pytest.mark.parametrize("version, args, kwargs, expected_kwargs, named_after", [
    # REBOUND 4 names every particle with 'hash'
    ("4.3.0", ("Geocenter",), {"date": "JD2451545.0"}, {"date": "JD2451545.0", "hash": "Earth"}, False),
    ("4.3.0", (), {"m": 1.0}, {"m": 1.0, "hash": "Earth"}, False),
    # REBOUND 5 uses 'name', except for Horizons queries, which are named after being added
    ("5.1.1", (), {"m": 1.0}, {"m": 1.0, "name": "Earth"}, False),
    ("5.1.1", ("Geocenter",), {"date": "JD2451545.0"}, {"date": "JD2451545.0"}, True),
])
def testParticleIdentifiedWithTheVersionsKeyword(reb, monkeypatch, version, args, kwargs,
                                                 expected_kwargs, named_after):

    monkeypatch.setattr(reb, "rb", SimpleNamespace(__version__=version), raising=False)

    sim = _RecordingSim()
    reb._addNamedParticle(sim, "Earth", *args, **kwargs)

    assert sim.added == [(args, expected_kwargs)]
    assert getattr(sim.particles[0], "name", None) == ("Earth" if named_after else None)


def testHeartbeatSetterUsedWhenItWorks(reb):

    sim = SimpleNamespace()

    def func(sim_pointer):
        pass

    assert reb._setHeartbeat(sim, func) is None
    assert sim.heartbeat is func


def testHeartbeatFallsBackWhenTheSetterIsBroken(reb, monkeypatch):

    # Mimic the REBOUND 5.1.1 setter, which fails because '_hb' is missing from __slots__
    class BrokenSim:
        __slots__ = ["_heartbeat"]

        @property
        def heartbeat(self):
            return None

        @heartbeat.setter
        def heartbeat(self, func):
            self._hb = func

    wrapped = []
    monkeypatch.setattr(reb, "rb", SimpleNamespace(
        simulation=SimpleNamespace(AFF=lambda f: wrapped.append(f) or ("wrapped", f))), raising=False)

    def func(sim_pointer):
        pass

    sim = BrokenSim()
    ref = reb._setHeartbeat(sim, func)

    assert ref == ("wrapped", func)
    assert sim._heartbeat is ref
    assert wrapped == [func]


def testHeartbeatUnrelatedAttributeErrorPropagates(reb):

    class OtherSim:
        @property
        def heartbeat(self):
            return None

        @heartbeat.setter
        def heartbeat(self, func):
            raise AttributeError("something else")

    with pytest.raises(AttributeError, match="something else"):
        reb._setHeartbeat(OtherSim(), lambda p: None)


### Installed REBOUND/REBOUNDx ###


def testReboundxAttachesToTheSimulation(realReb):
    """ Fails when reboundx was compiled against a different REBOUND version than the installed one. """

    sim = realReb.rb.Simulation()
    rebx = realReb.reboundx.Extras(sim)

    realReb._checkReboundxAttached(sim)
    assert rebx is not None


def testNamedParticlesAndHeartbeatWorkOnTheInstalledVersion(realReb):

    rb = realReb.rb

    sim = rb.Simulation()
    realReb._addNamedParticle(sim, "Sun", m=1.0)
    realReb._addNamedParticle(sim, "Earth", m=3e-6, x=1.0, vy=1.0)

    assert sim.particles["Earth"].m == 3e-6
    with pytest.raises(rb.ParticleNotFound):
        sim.particles["Mars"]

    calls = []

    def heartbeat(sim_pointer):
        calls.append(sim_pointer.contents.t)

    heartbeat_ref = realReb._setHeartbeat(sim, heartbeat)
    sim.integrate(0.1)

    assert len(calls) > 0
    del heartbeat_ref


def testEndToEndIntegrationMatchesReference(realReb):
    """ Integrate the example trajectory 60 days back and compare with a reference solution.

    The reference was computed with rebound 4.3.0 and reboundx 4.3.0 (compiled against it) and
    reproduced with rebound 5.1.1 and reboundx 5.1.0 to ~1e-13. The tolerance leaves room for
    floating-point differences between platforms, while catching any real change in the dynamics.
    """

    from wmpl.Config import config
    from wmpl.Utils.Pickling import loadPickle

    if not os.path.isfile(config.jpl_ephem_file):
        pytest.skip("DE430 ephemeris not found: {}".format(config.jpl_ephem_file))

    if not os.path.isfile(os.path.join(EXAMPLE_DIR, EXAMPLE_FILE)):
        pytest.skip("Example trajectory not found")

    traj = loadPickle(EXAMPLE_DIR, EXAMPLE_FILE)

    sim_outputs, sim_outputs_mc, diagnostics = realReb.reboundSimulate(
        None, None, traj=traj, direction="backward", sim_days=60, n_outputs=200, mc_runs=1, n_cpu=1,
        show_progress=False, return_diagnostics=True, random_seed=42)

    assert len(sim_outputs) == 200
    assert sim_outputs_mc == {}

    orbit = sim_outputs[-1][2]
    rtol = 1e-7
    assert orbit.a == pytest.approx(10.984662940935554, rel=rtol)
    assert orbit.e == pytest.approx(0.9469750833346323, rel=rtol)
    assert np.degrees(orbit.inc) == pytest.approx(164.56525202758945, rel=rtol)
    assert np.degrees(orbit.Omega) == pytest.approx(29.396553080685386, rel=rtol)
    assert np.degrees(orbit.omega) == pytest.approx(81.51628083264556, rel=rtol)

    # The Moon's closest approach comes from the heartbeat, so this also checks that it ran
    diag = next(iter(diagnostics.values()))
    assert diag["departed"]
    assert diag["impact"] is None
    assert diag["min_dist_au"]["Luna"] == pytest.approx(0.0018192455341135359, rel=rtol)
    assert diag["min_time_days"]["Luna"] == pytest.approx(-0.0426370612, rel=1e-6)
