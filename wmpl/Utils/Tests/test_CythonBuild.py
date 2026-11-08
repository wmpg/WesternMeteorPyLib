""" Tests that wmpl compiles its Cython modules with pyximport in a build directory of its own copy
(wmpl.Utils.CythonBuild), not in the ~/.pyxbld shared by every copy on the machine.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_CythonBuild.py -v
"""

import os
import sys
import tempfile

# Importing any wmpl module installs pyximport
import wmpl.MetSim.MetSimErosion
from wmpl.Utils.CythonBuild import pyximportBuildDir



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




if __name__ == "__main__":

    # Run all tests
    test_list = [testBuildDirIsOnePerCopy, testWmplInstallsPyximportWithTheBuildDirOfItsCopy]

    for test in test_list:
        test()
        print("PASS {:s}".format(test.__name__))
