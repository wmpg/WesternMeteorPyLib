""" Tests for how Trajectory attaches to a logger.

The correlator hands its own logger to every Trajectory it builds, so that solver output lands in the
correlator's log file. That has to work without affecting anything else in the process: a library
class must not redirect the logging of its own module, and must not accumulate handlers.

Run under pytest, or directly:

    python -m wmpl.Trajectory.Tests.test_TrajectoryLogging
"""

import io
import logging

import pytest

import wmpl.Trajectory.Trajectory as TrajectoryModule
from wmpl.Trajectory.Trajectory import Trajectory


# Arguments that build a Trajectory cheaply, without running anything
TRAJ_KWARGS = dict(jdt_ref=2451545.0, output_dir="/tmp", meastype=2)


@pytest.fixture(autouse=True)
def restoreModuleLogger():
    """ Leave the module logger exactly as it was found, whatever a test does to it. """

    module_logger = TrajectoryModule.log
    handlers = list(module_logger.handlers)

    yield

    TrajectoryModule.log = module_logger
    module_logger.handlers = handlers


def testParentLoggerIsUsedWithoutTouchingTheModule():
    """ A parent logger applies to that trajectory only, and leaves the module logger alone. """

    module_logger = TrajectoryModule.log
    parent = logging.getLogger("test_parent_logger")

    traj = Trajectory(parentlogger=parent, **TRAJ_KWARGS)

    assert traj.log is parent
    assert TrajectoryModule.log is module_logger, "the module logger was rebound by a constructor"

    # A later trajectory without a parent logger is unaffected by the earlier one
    assert Trajectory(**TRAJ_KWARGS).log is module_logger


def testHandlersDoNotAccumulateAcrossTrajectories():
    """ The console handler is attached once, not once per trajectory.

    Attaching one per construction makes every line appear as many times as there are Trajectory
    objects, which in the correlator is once per solved meteor.
    """

    module_logger = TrajectoryModule.log

    # Start from a known state, so the count does not depend on what ran before
    module_logger.handlers = []

    for _ in range(6):
        Trajectory(**TRAJ_KWARGS)

    assert len(module_logger.handlers) <= 1, \
        "{:d} handlers after 6 trajectories; every line would be printed that many times".format(
            len(module_logger.handlers))


def testParentLoggerOutputDoesNotReachTheModuleLogger():
    """ Lines logged by a trajectory with a parent logger stay out of the module's own logger. """

    module_logger = TrajectoryModule.log
    parent = logging.getLogger("test_isolated_parent_logger")
    parent.handlers = []

    traj = Trajectory(parentlogger=parent, **TRAJ_KWARGS)

    module_buffer = io.StringIO()
    module_handler = logging.StreamHandler(module_buffer)
    module_logger.addHandler(module_handler)

    parent_buffer = io.StringIO()
    parent.addHandler(logging.StreamHandler(parent_buffer))
    parent.setLevel(logging.INFO)

    traj.log.info("a line for the parent logger")

    module_logger.removeHandler(module_handler)

    assert "a line for the parent logger" in parent_buffer.getvalue()
    assert module_buffer.getvalue() == "", "the line leaked onto the module logger"


def testTheLoggerIsNotStoredOnTheInstance():
    """ The logger is resolved from a name, so a trajectory stays serialisable.

    A Logger cannot be JSON-encoded and toJson() converts the whole instance dictionary, so holding
    the logger itself there makes toJson raise "cannot pickle '_thread.RLock' object" for every
    trajectory. Only the name is stored.
    """

    import copy
    import json

    traj = Trajectory(**TRAJ_KWARGS)

    assert "log" not in traj.__dict__, "the logger object is in the instance dictionary"
    assert traj.__dict__["logger_name"] == TrajectoryModule.log.name

    # The three things a Trajectory has to survive
    as_json = json.loads(traj.toJson())
    assert "log" not in as_json
    assert as_json["logger_name"] == TrajectoryModule.log.name

    assert copy.deepcopy(traj).log is traj.log

    import pickle
    assert pickle.loads(pickle.dumps(traj, protocol=2)).log is traj.log


def testALegacyTrajectoryWithoutALoggerNameStillLogs():
    """ A trajectory restored from a pickle written before the logger existed falls back cleanly. """

    traj = Trajectory(**TRAJ_KWARGS)
    del traj.__dict__["logger_name"]

    assert traj.log is TrajectoryModule.log
    traj.log.debug("usable")


if __name__ == "__main__":

    import sys
    sys.exit(pytest.main([__file__, "-q"]))
