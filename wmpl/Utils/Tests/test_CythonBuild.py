""" Tests that wmpl compiles its Cython modules with pyximport in a build directory of its own copy
(wmpl.Utils.CythonBuild), not in the ~/.pyxbld shared by every copy on the machine, and that a module
compiled into the source tree before its .pyx was changed is compiled again from the .pyx.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_CythonBuild.py -v
"""

import importlib.machinery
import os
import sys
import tempfile

# Importing any wmpl module installs pyximport
import wmpl.MetSim.MetSimErosion
from wmpl.Utils.CythonBuild import pyximportBuildDir, staleCompiledModules, StaleCompiledModuleFinder



def testBuildDirIsOnePerCopy():
    """ Check that two copies of wmpl get different build directories, that a copy reached through a link
        gets the same one, and that all are under ~/.pyxbld.
    """

    with tempfile.TemporaryDirectory() as tmp_dir:

        copy_a = os.path.join(tmp_dir, "a", "wmpl")
        copy_b = os.path.join(tmp_dir, "b", "wmpl")
        os.makedirs(copy_a)
        os.makedirs(copy_b)

        dir_a, dir_b = pyximportBuildDir(copy_a), pyximportBuildDir(copy_b)

        assert dir_a != dir_b

        # Check the link only where links can be made, which on Windows needs extra privileges
        link_a = os.path.join(tmp_dir, "link_a")
        try:
            os.symlink(os.path.join(tmp_dir, "a"), link_a)
            link_made = True

        except (OSError, NotImplementedError):
            link_made = False

        if link_made:
            assert pyximportBuildDir(os.path.join(link_a, "wmpl")) == dir_a

        for build_dir in (dir_a, dir_b):
            assert os.path.dirname(build_dir) == os.path.join(os.path.expanduser("~"), ".pyxbld")
            assert os.path.basename(build_dir).startswith("wmpl_")

    # The directory of this copy does not change
    assert pyximportBuildDir() == pyximportBuildDir()



def testWmplInstallsPyximportWithTheBuildDirOfItsCopy():
    """ Check that importing wmpl installs pyximport before any of its modules does, so the only pyximport
        importer builds in the directory of this copy.
    """

    finders = [finder for finder in sys.meta_path if type(finder).__name__ == "PyxImportMetaFinder"]

    assert len(finders) == 1
    assert finders[0].pyxbuild_dir == pyximportBuildDir()



def touchFile(file_path, mtime):
    """ Make an empty file with the given modification time.

    Arguments:
        file_path: [str] Path of the file.
        mtime: [float] Modification time (Unix time).

    Return:
        None
    """

    with open(file_path, "w"):
        pass

    os.utime(file_path, (mtime, mtime))



def testStaleCompiledModulesAreFound():
    """ Check that a module compiled into the source tree for this Python before its .pyx was changed is
        found, with the name of the module, and that a module compiled after it is not, nor a file compiled
        for another version of Python or another system.
    """

    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]

    with tempfile.TemporaryDirectory() as tmp_dir:

        wmpl_dir = os.path.join(tmp_dir, "wmpl")
        metsim_dir = os.path.join(wmpl_dir, "MetSim")
        utils_dir = os.path.join(wmpl_dir, "Utils")
        os.makedirs(metsim_dir)
        os.makedirs(utils_dir)

        # Compiled before a git pull changed the .pyx
        touchFile(os.path.join(metsim_dir, "MetSimErosionCyTools" + suffix), 1000.0)
        touchFile(os.path.join(metsim_dir, "MetSimErosionCyTools.pyx"), 2000.0)

        # Compiled after the .pyx was changed, next to an old file of another version of Python
        touchFile(os.path.join(utils_dir, "FastTools.pyx"), 1000.0)
        touchFile(os.path.join(utils_dir, "FastTools" + suffix), 2000.0)
        touchFile(os.path.join(utils_dir, "FastTools.cp27-win_amd64.pyd"), 500.0)

        # Only compiled for another version of Python, which this Python does not import
        touchFile(os.path.join(utils_dir, "OtherTools.pyx"), 1000.0)
        touchFile(os.path.join(utils_dir, "OtherTools.cpython-27-x86_64-linux-gnu.so"), 500.0)

        stale = staleCompiledModules(wmpl_dir)

        assert stale == {"wmpl.MetSim.MetSimErosionCyTools": \
            os.path.join(metsim_dir, "MetSimErosionCyTools" + suffix)}



def testStaleModulesAreTakenFromThePyx():
    """ Check that the finder hands only the stale modules to the finder of pyximport. """

    class PyxFinder(object):
        def find_spec(self, fullname, path, target=None):
            return "spec of " + fullname

    finder = StaleCompiledModuleFinder({"wmpl.MetSim.MetSimErosionCyTools": "old.pyd"}, PyxFinder())

    assert finder.find_spec("wmpl.MetSim.MetSimErosionCyTools", None) \
        == "spec of wmpl.MetSim.MetSimErosionCyTools"
    assert finder.find_spec("wmpl.MetSim.MetSimErosion", None) is None





if __name__ == "__main__":

    # Run all tests
    test_list = [testBuildDirIsOnePerCopy, testWmplInstallsPyximportWithTheBuildDirOfItsCopy, \
        testStaleCompiledModulesAreFound, testStaleModulesAreTakenFromThePyx]

    for test in test_list:
        test()
        print("PASS {:s}".format(test.__name__))
