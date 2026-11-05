""" Compile the Cython modules of wmpl with pyximport in a build directory of their own copy of wmpl.

pyximport compiles into ~/.pyxbld by default, which is shared by every copy of wmpl on a machine (checkouts,
git worktrees, installations). A module is compiled to the same path whichever copy it comes from, and it is
only rebuilt when its .pyx is newer than the compiled file there. So once two copies have each compiled a
module, the one which did it first imports the module of the other copy, without any warning.
"""

import hashlib
import os

import numpy as np



def pyximportBuildDir(wmpl_dir=None):
    """ Return the directory under ~/.pyxbld in which pyximport compiles the Cython modules of a copy of
        wmpl, named after the path of its wmpl package.

    Keyword arguments:
        wmpl_dir: [str] Path of the wmpl package. None by default, in which case this copy is used.

    Return:
        [str] Path of the build directory.
    """

    if wmpl_dir is None:
        wmpl_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Use the real path, so the same copy reached through a link is built in one place
    copy_id = hashlib.sha1(os.path.realpath(wmpl_dir).encode("utf-8")).hexdigest()[:12]

    return os.path.join(os.path.expanduser("~"), ".pyxbld", "wmpl_" + copy_id)



def installPyximport():
    """ Install pyximport with the numpy headers, compiling in the build directory of this copy of wmpl.
        pyximport keeps the build directory of its first installation in a process, so this is called when
        wmpl is imported, before any module installs it.

    Return:
        [tuple] The importers returned by pyximport.install().
    """

    import pyximport

    return pyximport.install(setup_args={'include_dirs': [np.get_include()]}, build_dir=pyximportBuildDir())
