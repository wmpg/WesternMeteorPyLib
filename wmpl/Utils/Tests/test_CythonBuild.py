""" Tests that wmpl compiles its Cython modules with pyximport in a build directory of its own copy
(wmpl.Utils.CythonBuild), not in the ~/.pyxbld shared by every copy on the machine.

Run under pytest, or directly:

    python -m wmpl.Utils.Tests.test_CythonBuild
"""

import os
import sys
import tempfile

import wmpl.MetSim.MetSimErosion  # noqa: F401 (installs pyximport, as any wmpl module does)
from wmpl.Utils.CythonBuild import pyximportBuildDir


def test_build_dir_is_one_per_copy():
    """ Two copies of wmpl get different build directories, a copy reached through a link the same one, and all
        are under ~/.pyxbld. """

    with tempfile.TemporaryDirectory() as tmp_dir:

        copy_a = os.path.join(tmp_dir, "a", "wmpl")
        copy_b = os.path.join(tmp_dir, "b", "wmpl")
        os.makedirs(copy_a)
        os.makedirs(copy_b)
        link_a = os.path.join(tmp_dir, "link_a")
        os.symlink(os.path.join(tmp_dir, "a"), link_a)

        dir_a, dir_b = pyximportBuildDir(copy_a), pyximportBuildDir(copy_b)

        assert dir_a != dir_b
        assert pyximportBuildDir(os.path.join(link_a, "wmpl")) == dir_a
        for build_dir in (dir_a, dir_b):
            assert os.path.dirname(build_dir) == os.path.join(os.path.expanduser("~"), ".pyxbld")
            assert os.path.basename(build_dir).startswith("wmpl_")

    # This copy's is stable
    assert pyximportBuildDir() == pyximportBuildDir()


def test_wmpl_installs_pyximport_with_the_build_dir_of_its_copy():
    """ Importing wmpl installs pyximport before any of its modules does, so the only pyximport importer builds in
        this copy's directory (pyximport keeps the build directory of its first installation). """

    finders = [finder for finder in sys.meta_path if type(finder).__name__ == "PyxImportMetaFinder"]

    assert len(finders) == 1
    assert finders[0].pyxbuild_dir == pyximportBuildDir()


if __name__ == "__main__":
    test_build_dir_is_one_per_copy()
    test_wmpl_installs_pyximport_with_the_build_dir_of_its_copy()
    print("All CythonBuild checks passed.")
