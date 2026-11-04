""" Tests of the figure of the Jupiter-family dynamics: the fraction it draws, and that it is made from what
a run returns. The last test integrates through the command line, and needs rebound and the DE430 ephemeris. """

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from wmpl.Rebound import REBOUND as reb
from wmpl.Utils.OrbitClassification import calcBinomialInterval, countPerihelionReach
from wmpl.Utils.OrbitDynamicsPlot import plotJupiterFamilyDynamics, reachFractionOverTime


T = np.linspace(0.0, -1000.0, 11)


def _result(q, a, **kwargs):

    result = dict(dynamics_class='undetermined', p=0.5, p_low=0.2, p_high=0.8, f_q=0.1, f_a=0.1, n_clones=len(q),
        assumptions=dict(mode='shober2024', confidence=0.95, years=1000.0), ensemble=dict(t=T, q=q, a=a))
    result.update(kwargs)

    return result


def test_the_fraction_counts_each_clone_once_and_ends_at_p():

    q = np.full((4, 11), 1.0)
    q[0, 3:] = 3.0          # stays up from the output 3
    q[1, 5] = 3.0           # only at the output 5, and it still counts afterwards
    q[2, :] = 1.0

    p, low, high = reachFractionOverTime(T, q)

    assert p.tolist() == [0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    reached, n_clones = countPerihelionReach(q)
    assert p[-1] == reached/n_clones
    assert (low[-1], high[-1]) == calcBinomialInterval(reached, n_clones)


def test_the_fraction_leaves_out_clones_without_a_state_and_counts_those_that_left():

    q = np.full((3, 11), 1.0)
    q[0] = np.nan
    q[1, 6:] = np.inf
    q[2, 7:] = np.nan       # hit a body

    assert reachFractionOverTime(T, q)[0][-1] == 0.5


def test_no_clone_with_a_state_has_no_fraction():

    with pytest.raises(ValueError, match="No clone"):
        reachFractionOverTime(T, np.full((2, 11), np.nan))


def _ensemble():

    q = np.full((6, 11), 1.0)
    a = np.full((6, 11), 3.5)
    q[0, 4:], a[0, 4:] = 3.0, 8.0
    q[1, 6:], a[1, 6:] = np.inf, np.inf     # left the solar system
    a[2, 8:] = -5.0                         # unbound
    q[3, 7:], a[3, 7:] = np.nan, np.nan     # hit a body
    q[4], a[4] = np.nan, np.nan             # no state at all

    return q, a


@pytest.mark.parametrize("name", ["figure.png", "figure.pdf"])
def test_the_figure_is_saved_from_every_kind_of_clone(tmp_path, name):

    path = tmp_path/name
    plotJupiterFamilyDynamics(_result(*_ensemble()), str(path))

    assert path.stat().st_size > 1000


def test_the_figure_is_saved_with_no_clone_reaching_the_limit(tmp_path):

    path = tmp_path/"figure.png"
    plotJupiterFamilyDynamics(_result(np.full((3, 11), 1.0), np.full((3, 11), 3.5)), str(path))

    assert path.stat().st_size > 1000


def test_more_clones_than_lines_are_drawn(tmp_path):

    q = np.full((50, 11), 3.0)
    path = tmp_path/"figure.png"

    plotJupiterFamilyDynamics(_result(q, np.full((50, 11), 8.0)), str(path), max_lines=5)

    assert path.stat().st_size > 1000


def test_a_result_without_arrays_is_refused(tmp_path):

    with pytest.raises(ValueError, match="return_ensemble"):
        plotJupiterFamilyDynamics(_result(*_ensemble(), ensemble=None), str(tmp_path/"figure.png"))


def test_every_clone_unbound_is_refused(tmp_path):

    nothing = np.full((2, 11), np.nan)

    with pytest.raises(ValueError, match="unbound"):
        plotJupiterFamilyDynamics(_result(nothing, nothing), str(tmp_path/"figure.png"))


@pytest.mark.skipif(not (reb.REBOUND_FOUND and os.path.isfile(reb.config.jpl_ephem_file)),
    reason="rebound or the DE430 ephemeris not available")
def test_the_command_line_saves_the_figure_and_draws_it_again_from_the_checkpoint(tmp_path):

    figure, again, saved, checkpoint = (str(tmp_path/name) for name in ("a.png", "b.png", "r.json", "c.pkl"))

    command = [sys.executable, "-W", "ignore", "-m", "wmpl.Utils.OrbitClassification", "-a", "4.3", "-e", "0.79", "-i",
        "7.1", "-p", "40.95", "-n", "203.945", "--date", "2013-04-13 21:43:55.4", "--dynamics", "shober2024",
        "--max_clones", "2", "--years", "60", "--cores", "1", "--seed", "1", "--dynamics_checkpoint", checkpoint]

    first = subprocess.run(command + ["--dynamics_plot", figure, "--dynamics_save", saved], capture_output=True,
        text=True)
    assert first.returncode == 0, first.stderr
    assert os.path.getsize(figure) > 1000

    # The arrays are for the figure, and the JSON file stays as it was
    assert 'ensemble' not in json.load(open(saved))

    second = subprocess.run(command + ["--dynamics_plot", again], capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    assert "2 clone(s) read back" in second.stdout
    assert os.path.getsize(again) > 1000
