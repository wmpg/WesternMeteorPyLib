""" Tests that loadParentBodies does not mix general relativity models: a state saved with another
one is propagated again. No integration is run. """

import importlib.util
import json
import os

import pytest


REBOUND_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "REBOUND.py")

JD = 2459000.5
SAVED = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
PROPAGATED = [9.0, 9.0, 9.0, 0.9, 0.9, 0.9]


@pytest.fixture(scope="module")
def reb():

    spec = importlib.util.spec_from_file_location("_wmpl_test_rebound_parent_bodies_gr", REBOUND_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


@pytest.fixture
def load(reb, monkeypatch, tmp_path):
    """ Load a file with one body saved at the epoch, and return the state and the models propagated. """

    def _load(at_epoch_extra, **kwargs):

        at_epoch = dict({"jd_tdb": JD, "position": SAVED[:3], "velocity": SAVED[3:]}, **at_epoch_extra)
        path = str(tmp_path/"matches.json")

        with open(path, "w") as f:
            json.dump({"criteria_keys": ["DSH"], "matches": [{"name": "A body", "rank": 1, "q_AU": 1.0, "e": 0.5,
                "incl_deg": 5.0, "peri_deg": 10.0, "node_deg": 20.0, "catalogue_epoch_jd": JD - 100.0,
                "catalogue_mean_anomaly_deg": 30.0, "at_epoch": at_epoch}]}, f)

        models = []

        def propagate(bodies, jd_tdb, ephem_source="local", gr="gr_full"):
            models.append(gr)
            return [{"position": PROPAGATED[:3], "velocity": PROPAGATED[3:]} for _ in bodies]

        monkeypatch.setattr(reb, "propagateBodiesToEpoch", propagate)

        bodies, _, skipped = reb.loadParentBodies(path, JD, **kwargs)
        assert skipped == []

        return bodies[0]["state"], models

    return _load


@pytest.mark.parametrize("model", ["gr", "gr_full", "none"])
def test_a_state_saved_with_the_same_model_is_taken_as_it_is(load, model):

    assert load({"gr": model}, gr=model) == (SAVED, [])


@pytest.mark.parametrize("saved, wanted", [("gr_full", "none"), ("none", "gr_full"), ("gr", "gr_full")])
def test_a_state_saved_with_another_model_is_propagated_with_the_one_asked_for(load, saved, wanted):

    assert load({"gr": saved}, gr=wanted) == (PROPAGATED, [wanted])


def test_a_file_that_does_not_name_its_model_holds_gr_full(load):

    assert load({}, gr="gr_full") == (SAVED, [])
    assert load({}, gr="none") == (PROPAGATED, ["none"])
