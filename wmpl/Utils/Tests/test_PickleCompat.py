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
import logging
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


def testSavePickleSurvivesAnyFileName(tmp_path):
    """ savePickle writes through a temporary file, which must not collide with the target name.

    The temporary name used to be file_name.replace('.p', '_x'), a global substring replace. A name
    containing no '.p' produced a temporary name equal to the target, so the just-written file was
    removed and the rename then raised. It must also stay invisible to the correlator, which globs
    for "*.pickle*" to find work, and it must not be left behind.
    """

    # Its own directory, so the listing assertions below hold under pytest and under the standalone
    #   runner, which hands every test the same temporary directory
    tmp_dir = os.path.join(str(tmp_path), "savepickle_names")
    os.makedirs(tmp_dir, exist_ok=True)

    for file_name in ["a_trajectory.pickle", "results.dat", "a.pre.pickle", "noextension"]:

        savePickle({"name": file_name}, tmp_dir, file_name)

        assert loadPickle(tmp_dir, file_name) == {"name": file_name}
        assert os.path.isfile(os.path.join(tmp_dir, file_name))

        # Nothing else is left in the directory, under any name
        assert os.listdir(tmp_dir) == [file_name]

        os.remove(os.path.join(tmp_dir, file_name))

    # Overwriting keeps the new content, and the file is never absent in between
    savePickle({"v": 1}, tmp_dir, "t.pickle")
    savePickle({"v": 2}, tmp_dir, "t.pickle")
    assert loadPickle(tmp_dir, "t.pickle") == {"v": 2}
    assert os.listdir(tmp_dir) == ["t.pickle"]


def testLoadBackFillsTheLbfgsbCutoff(tmp_path):
    """ A trajectory pickled before l_bfgs_b_cutoff existed still loads with a usable value.

    estimateTimingAndVelocity reads self.l_bfgs_b_cutoff, so an older pickle without it would raise
    AttributeError. loadPickle back-fills it, as it does for gravity_factor and v0z.
    """

    tmp_dir = os.path.join(str(tmp_path), "lbfgsb_backfill")
    os.makedirs(tmp_dir, exist_ok=True)

    # The back-fill only applies to objects that look like a trajectory
    legacy = SimpleNamespace(orbit=None, observations=[], gravity_factor=1.0, v0z=0.0)
    assert not hasattr(legacy, "l_bfgs_b_cutoff")

    savePickle(legacy, tmp_dir, "legacy_trajectory.pickle")
    loaded = loadPickle(tmp_dir, "legacy_trajectory.pickle")

    assert hasattr(loaded, "l_bfgs_b_cutoff"), "the cutoff was not back-filled"
    assert loaded.l_bfgs_b_cutoff == 5

    # An explicit value is left alone
    explicit = SimpleNamespace(orbit=None, observations=[], l_bfgs_b_cutoff=7)
    savePickle(explicit, tmp_dir, "explicit_trajectory.pickle")
    assert loadPickle(tmp_dir, "explicit_trajectory.pickle").l_bfgs_b_cutoff == 7


def testLoadBackFillsTheLogger(tmp_path):
    """ A trajectory pickled before the logger moved onto the instance still loads with one.

    Trajectory methods log through self.log, so an older pickle without the attribute would raise
    AttributeError the first time one of them logged. Loggers pickle by name, so newer pickles carry
    theirs already and must keep it.
    """

    tmp_dir = os.path.join(str(tmp_path), "logger_backfill")
    os.makedirs(tmp_dir, exist_ok=True)

    legacy = SimpleNamespace(orbit=None, observations=[])
    assert not hasattr(legacy, "log")

    savePickle(legacy, tmp_dir, "legacy_trajectory.pickle")
    loaded = loadPickle(tmp_dir, "legacy_trajectory.pickle")

    assert hasattr(loaded, "log"), "the logger was not back-filled"
    assert loaded.log.name == "wmpl_logger"
    loaded.log.debug("usable")

    # A logger already on the object is kept
    carried = SimpleNamespace(orbit=None, observations=[], log=logging.getLogger("carried_logger"))
    savePickle(carried, tmp_dir, "carried_trajectory.pickle")
    assert loadPickle(tmp_dir, "carried_trajectory.pickle").log.name == "carried_logger"


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
        testSavePickleSurvivesAnyFileName,
        testLoadBackFillsTheLbfgsbCutoff,
        testLoadBackFillsTheLogger,
        ]

    tmp_holder = tempfile.mkdtemp()

    failed = 0
    for test_func in test_functions:

        try:
            if test_func in (testLoadBackFillsTheLegacyMisspelling, testSavePickleSurvivesAnyFileName,
                             testLoadBackFillsTheLbfgsbCutoff, testLoadBackFillsTheLogger):
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
