""" Tests of --draw_q: the clones of a pickle solved with Monte Carlo can draw q instead of e. """

import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from wmpl.Rebound import REBOUND as reb
from wmpl.Utils.OrbitClassification import classifyJupiterFamilyDynamicsShober2024, elementUncertainties, \
    formatJupiterFamilyDynamics
from wmpl.Utils.Pickling import loadPickle


EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Dynesty", "examples",
    "20191023_091225")
EXAMPLE_FILE = "20191023_091225_trajectory.pickle"
EXAMPLE = os.path.join(EXAMPLE_DIR, EXAMPLE_FILE)

UNCERTAINTIES = SimpleNamespace(a=0.01, e=0.001, q=0.0002, i=np.radians(0.1), node=np.radians(0.01),
    peri=np.radians(0.01))


def test_the_uncertainty_of_q_is_taken_only_when_asked():

    traj = SimpleNamespace(uncertainties=UNCERTAINTIES)

    with_e, source = elementUncertainties(traj)
    assert with_e['e'] == 0.001 and 'q' not in with_e
    assert source == "Monte Carlo of the trajectory"

    with_q, source = elementUncertainties(traj, draw_q=True)
    assert with_q['q'] == 0.0002 and 'e' not in with_q
    assert source == "Monte Carlo of the trajectory"


def test_a_trajectory_without_monte_carlo_has_no_uncertainty_of_q_to_draw():

    with pytest.raises(ValueError, match="no Monte Carlo"):
        elementUncertainties(SimpleNamespace(uncertainties=None), draw_q=True)


@pytest.mark.skipif(not (reb.REBOUND_FOUND and os.path.isfile(reb.config.jpl_ephem_file)
    and os.path.isfile(EXAMPLE)), reason="rebound, the DE430 ephemeris or the example not available")
def test_a_real_trajectory_integrates_drawing_q_and_the_report_says_so():

    traj = loadPickle(EXAMPLE_DIR, EXAMPLE_FILE)
    traj.uncertainties = UNCERTAINTIES

    result = classifyJupiterFamilyDynamicsShober2024(traj, n_clones=2, years=60.0, seed=1, n_cpu=1, draw_q=True)

    assert result['n_integrated'] == 2 and result['assumptions']['sigma']['q'] == 0.0002
    assert "e = 1 - q/a, unlike the paper" in "\n".join(formatJupiterFamilyDynamics(result))


def _cli(*args):

    result = subprocess.run([sys.executable, "-W", "ignore", "-m", "wmpl.Utils.OrbitClassification"] + list(args),
        capture_output=True, text=True)

    return result.returncode, result.stdout + result.stderr


@pytest.mark.skipif(not os.path.isfile(EXAMPLE), reason="example trajectory not available")
@pytest.mark.parametrize("args, message", [
    ([EXAMPLE, "--dynamics", "shober2024", "--draw_q", "--years", "60", "--max_clones", "2"], "no Monte Carlo"),
    ([EXAMPLE, "--dynamics", "--draw_q"], "--draw_q goes with"),
    ([EXAMPLE, "--dynamics", "shober2024", "--sigma_q", "0.001"], "--draw_q takes the uncertainty of q"),
    ])
def test_what_the_command_line_says_with_a_pickle(args, message):

    code, output = _cli(*args)

    assert code == 1
    assert message in output


def test_the_elements_given_by_hand_are_pointed_to_sigma_q():

    code, output = _cli("-a", "4.3", "-e", "0.79", "-i", "7.1", "-p", "40.95", "-n", "203.945", "--date",
        "2013-04-13 21:43:55.4", "--dynamics", "shober2024", "--draw_q")

    assert code == 1
    assert "use --sigma_q" in output
