""" Tests that a stored trajectory pickle still loads and still carries the attribute names the
    library reads.

An attribute name is part of a pickle's on-disk format: unpickling restores the instance __dict__
by key, so renaming self.coordiantes to self.coordinates makes every archived file raise
AttributeError the next time something reads it. Years of solved trajectories are stored this way,
so a rename is not a refactor, it is a format change.

wmpl.Utils.Pickling.loadPickle already shims one such case. Older files carry the misspelled
"uncertanties" and newer ones carry "uncertainties"; loadPickle back-fills whichever is missing, so
both spellings are always present after a load. That is the pattern to follow if another attribute
ever has to be renamed: correct the spelling, keep the old name as an alias written on load.

These tests exist to make a future spelling sweep fail loudly rather than silently, and are the
reason a repo-wide typo fix can be merged with confidence.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_PickleCompat.py -v

or standalone (no pytest required):
    python -m wmpl.Utils.Tests.test_PickleCompat
"""

from __future__ import print_function, division, absolute_import

import os
from types import SimpleNamespace

from wmpl.Utils.Pickling import loadPickle, savePickle


EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "Dynesty", "examples", "20191023_091225")
EXAMPLE_FILE = "20191023_091225_trajectory.pickle"


# Attributes of a stored Trajectory that the rest of the library reads by name. Renaming any of
#   them breaks every archived pickle, so they are pinned here.
TRAJECTORY_ATTRIBUTES = [
    "jdt_ref",
    "traj_id",
    "orbit",
    "observations",
    "state_vect_mini",
    "radiant_eci_mini",
    "v_init",
    "v_avg",
    "uncertainties",
    "rbeg_ele",
    "rend_ele",
    "gravity_factor",
    ]

# Attributes of a stored ObservedPoints that are read elsewhere
OBSERVATION_ATTRIBUTES = [
    "station_id",
    "jdt_ref",
    "time_data",
    "meas1",
    "meas2",
    "model_ht",
    "state_vect_dist",
    "ignore_list",
    ]


def _loadExample():
    """ The trajectory pickle tracked in the repository, or None if it is not there. """

    path = os.path.join(EXAMPLE_DIR, EXAMPLE_FILE)

    if not os.path.isfile(path):
        return None

    return loadPickle(EXAMPLE_DIR, EXAMPLE_FILE)


def testStoredTrajectoryKeepsItsAttributeNames():
    """ Every attribute the library reads off a Trajectory must survive a round trip from disk. """

    traj = _loadExample()

    if traj is None:
        return

    missing = [name for name in TRAJECTORY_ATTRIBUTES if not hasattr(traj, name)]

    assert not missing, \
        "a stored trajectory no longer carries {!s}; renaming an attribute changes the pickle " \
        "format, so keep the old name as an alias in wmpl.Utils.Pickling.loadPickle".format(missing)


def testStoredObservationsKeepTheirAttributeNames():
    """ The same, for the per-station observations hanging off the trajectory. """

    traj = _loadExample()

    if traj is None:
        return

    assert traj.observations, "the example trajectory carries no observations"

    for obs in traj.observations:

        missing = [name for name in OBSERVATION_ATTRIBUTES if not hasattr(obs, name)]

        assert not missing, \
            "station {!s} no longer carries {!s}".format(getattr(obs, "station_id", "?"), missing)


def _roundTrip(obj, tmp_dir):
    """ Write an object with savePickle and read it back with loadPickle. """

    savePickle(obj, tmp_dir, "alias_test.pickle")

    return loadPickle(tmp_dir, "alias_test.pickle")


def testLoadBackFillsTheLegacyMisspelling(tmp_path):
    """ loadPickle back-fills whichever spelling of "uncertainties" a file is missing.

        This is the alias pattern to copy if another attribute ever has to be renamed: correct the
        spelling in the code, and write the old name on load so archived files keep working.

        The example pickle happens to store both spellings, so it cannot exercise the shim. These
        objects carry one spelling each, which is what an older or newer file actually looks like.
    """

    tmp_dir = str(tmp_path)

    # A file written before the spelling was corrected
    legacy = SimpleNamespace()
    legacy.uncertanties = "sentinel"
    restored = _roundTrip(legacy, tmp_dir)

    assert hasattr(restored, "uncertainties"), \
        "a file carrying only the legacy misspelling must gain the corrected name on load"
    assert restored.uncertainties == "sentinel"

    # A file written after it was corrected
    current = SimpleNamespace()
    current.uncertainties = "sentinel"
    restored = _roundTrip(current, tmp_dir)

    assert hasattr(restored, "uncertanties"), \
        "a file carrying only the corrected name must keep the legacy one, or old scripts break"
    assert restored.uncertanties == "sentinel"


def testBothSpellingsOfUncertaintiesArePresent():
    """ After a load both spellings resolve, whichever the file on disk happened to carry. """

    traj = _loadExample()

    if traj is None:
        return

    assert hasattr(traj, "uncertainties"), "the corrected spelling must be present"
    assert hasattr(traj, "uncertanties"), \
        "the legacy misspelling must still resolve, or old analysis scripts break"

    assert traj.uncertainties is traj.uncertanties, \
        "the two spellings must be the same object, not two copies that can drift apart"


def testOrbitSurvivesTheRoundTrip():
    """ The orbit hangs off the trajectory and is the part most often read from an archived file. """

    traj = _loadExample()

    if traj is None:
        return

    if traj.orbit is None:
        return

    for name in ("a", "e", "i", "q", "Tj", "la_sun", "v_g"):
        assert hasattr(traj.orbit, name), "the stored orbit no longer carries {:s}".format(name)


if __name__ == "__main__":

    import tempfile

    test_functions = [
        testStoredTrajectoryKeepsItsAttributeNames,
        testStoredObservationsKeepTheirAttributeNames,
        testBothSpellingsOfUncertaintiesArePresent,
        testOrbitSurvivesTheRoundTrip,
        testLoadBackFillsTheLegacyMisspelling,
        ]

    tmp_holder = tempfile.mkdtemp()

    failed = 0
    for test_func in test_functions:

        try:
            if test_func is testLoadBackFillsTheLegacyMisspelling:
                test_func(tmp_holder)
            else:
                test_func()
            print("PASS: {:s}".format(test_func.__name__))

        except Exception as e:
            failed += 1
            print("FAIL: {:s}: {:s}".format(test_func.__name__, str(e)))

    print()
    if failed:
        print("{:d}/{:d} tests failed".format(failed, len(test_functions)))
        raise SystemExit(1)

    print("All {:d} tests passed".format(len(test_functions)))
