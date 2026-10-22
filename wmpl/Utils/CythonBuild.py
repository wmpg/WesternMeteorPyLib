""" Compile wmpl's Cython modules with pyximport in a build directory of their own copy of wmpl.

pyximport's default build directory, ~/.pyxbld, is shared by every copy of wmpl on a machine (checkouts, git
worktrees, installations): a module is compiled to the same path whichever copy it comes from, and only rebuilt when
its .pyx is newer than the compiled file there. So once two copies have each compiled a module, the one that did it
first imports the other's compiled module, whatever its own .pyx says, without any warning.
"""

import hashlib
import os

import numpy as np


def pyximportBuildDir(wmpl_dir=None):
    """ The directory pyximport compiles the Cython modules of a copy of wmpl in, under ~/.pyxbld: one per copy,
        named after the path of its wmpl package.

    Keyword arguments:
        wmpl_dir: [str] Path of the wmpl package. None by default, for this copy.

    Return:
        [str] The build directory.
    """

    if wmpl_dir is None:
        wmpl_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # The real path, so that the same copy reached through a link builds in one place
    copy_id = hashlib.sha1(os.path.realpath(wmpl_dir).encode("utf-8")).hexdigest()[:12]

    return os.path.join(os.path.expanduser("~"), ".pyxbld", "wmpl_" + copy_id)


def installPyximport():
    """ Install pyximport for wmpl's Cython modules: with the numpy headers, and compiling in the build directory of
        this copy (pyximportBuildDir()). pyximport keeps the build directory of its first installation in a process,
        so it is installed when wmpl is imported, before any module does it. """

    import pyximport

    return pyximport.install(setup_args={'include_dirs': [np.get_include()]}, build_dir=pyximportBuildDir())
