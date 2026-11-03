""" Tests for the rescaling of the MEGNO deviation at its real scale. The integrations skip without
REBOUND or REBOUNDx installed.
"""

import os
import importlib.util

import numpy as np
import pytest


REBOUND_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "REBOUND.py")


@pytest.fixture(scope="module")
def reb():
    """ Load REBOUND.py under an isolated module name. """

    spec = importlib.util.spec_from_file_location("_wmpl_test_rebound_lyapunov_clones", REBOUND_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


@pytest.fixture
def realReb(reb):
    """ The REBOUND module, skipping the test if rebound/reboundx cannot be imported. """

    if not reb.REBOUND_FOUND:
        pytest.skip("rebound/reboundx not importable: {}".format(reb._REBOUND_IMPORT_ERROR))

    return reb


def _chaoticTask(reb, span_years, n_outputs):
    """ A particle in the chaotic zone just outside Jupiter's orbit around a Sun with only Jupiter: it
    has a Lyapunov time of tens of years and takes well under a second per century. """

    import rebound as rb

    sim = rb.Simulation()
    sim.add(m=1.0)
    sim.add(primary=sim.particles[0], m=9.5479e-4, a=5.2, e=0.048)
    sim.add(primary=sim.particles[0], a=4.5, e=0.05, inc=0.02, omega=1.0, Omega=2.0, f=3.0)

    states = [[p.x, p.y, p.z, p.vx, p.vy, p.vz] for p in sim.particles]

    return {"planet_names": ["Sun", "Jupiter"], "planet_states": states[:2], "planet_masses": [1.0, 9.5479e-4],
            "particle_states": [states[2]], "direction": "forward",
            "times": [reb.daysToSimTime(t) for t in np.linspace(0, span_years*365.25, n_outputs)]}


def testRescalingAtItsOwnLimitAddsNoJump(realReb, monkeypatch):
    """ At the limit in use, ln|delta| right after the first rescale is that of a run without rescaling.
    IAS15 keeps the round-off of its compensated summation at the old scale, so the test has to reach the
    limit: at a limit of 1e20 the same difference was about 8. """

    limit = realReb.MEGNO_RESCALE_ABOVE
    task = _chaoticTask(realReb, 1500.0, 3000)

    rescaled = np.array(realReb.computeMegno(task)["ln_delta"])

    monkeypatch.setattr(realReb, "MEGNO_RESCALE_ABOVE", 1e300)
    unscaled = np.array(realReb.computeMegno(task)["ln_delta"])

    # The run has to reach the limit for this to test anything
    first = int(np.argmax(unscaled > np.log(limit)))
    assert unscaled[first] > np.log(limit)

    # The orbit itself is the same to round-off until then, so the series can be compared there
    assert abs(rescaled[first + 1] - unscaled[first + 1]) < 1e-5

    # No output-to-output step stands out at the rescales
    assert np.max(np.abs(np.diff(rescaled)[first:first + 5])) < 3.0
