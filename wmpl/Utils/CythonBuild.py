""" Compile the Cython modules of wmpl with pyximport in a build directory of their own copy of wmpl.

pyximport compiles into ~/.pyxbld by default, which is shared by every copy of wmpl on a machine (checkouts,
git worktrees, installations). A module is compiled to the same path whichever copy it comes from, and it is
only rebuilt when its .pyx is newer than the compiled file there. So once two copies have each compiled a
module, the one which did it first imports the module of the other copy, without any warning.

A module can also be compiled into the source tree itself, next to its .pyx, by "pip install -e ." or
"python setup.py build_ext --inplace". Python imports that file before it asks pyximport, also after a git
pull changed the .pyx, so the old module is used, which fails on a missing function or, worse, computes with
the old code. Such a file is skipped if it is older than its .pyx, and the module is compiled from the .pyx.
"""

import hashlib
import importlib.abc
import importlib.machinery
import os
import sys

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



def staleCompiledModules(wmpl_dir=None):
    """ Find the Cython modules of a copy of wmpl which were compiled into its source tree before their .pyx
        was last changed, e.g. by "pip install -e ." before a git pull.

    Keyword arguments:
        wmpl_dir: [str] Path of the wmpl package. None by default, in which case this copy is used.

    Return:
        [dict] Names of the modules (e.g. "wmpl.MetSim.MetSimErosionCyTools") and the paths of their old
            compiled files.
    """

    if wmpl_dir is None:
        wmpl_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    package_root = os.path.dirname(wmpl_dir)

    stale = {}
    for dir_path, dir_names, file_names in os.walk(wmpl_dir):

        # Skip the directories without code
        dir_names[:] = [name for name in dir_names if (not name.startswith(".")) \
            and (name not in ["__pycache__", "share"])]

        for file_name in file_names:

            if not file_name.endswith(".pyx"):
                continue

            pyx_path = os.path.join(dir_path, file_name)
            module_base = file_name[:-len(".pyx")]

            # Take the compiled file which this Python would import, e.g.
            #   MetSimErosionCyTools.cpython-311-x86_64-linux-gnu.so on Linux, in the order Python tries them,
            #   so that files compiled for other versions of Python or other systems are not checked
            for suffix in importlib.machinery.EXTENSION_SUFFIXES:

                compiled_path = os.path.join(dir_path, module_base + suffix)

                if os.path.isfile(compiled_path):

                    if os.path.getmtime(compiled_path) < os.path.getmtime(pyx_path):

                        module_path = os.path.relpath(os.path.join(dir_path, module_base), package_root)
                        stale[".".join(module_path.split(os.sep))] = compiled_path

                    break

    return stale



class StaleCompiledModuleFinder(importlib.abc.MetaPathFinder):
    def __init__(self, stale, pyx_finder):
        """ Import finder which takes the given modules from their .pyx through pyximport, before Python
            finds their old compiled files in the source tree.

        Arguments:
            stale: [dict] Names of the modules and the paths of their old compiled files, from
                staleCompiledModules().
            pyx_finder: [object] Finder of pyximport, which compiles a module from its .pyx.
        """

        self.stale = stale
        self.pyx_finder = pyx_finder



    def find_spec(self, fullname, path, target=None):
        """ Find the module in its .pyx if it is one of the stale modules, otherwise leave it to the other
            finders.
        """

        if fullname not in self.stale:
            return None

        return self.pyx_finder.find_spec(fullname, path, target)



def installPyximport():
    """ Install pyximport with the numpy headers, compiling in the build directory of this copy of wmpl.
        pyximport keeps the build directory of its first installation in a process, so this is called when
        wmpl is imported, before any module installs it. The modules compiled into the source tree before
        their .pyx was last changed are compiled from the .pyx instead (see staleCompiledModules()).

    Return:
        [tuple] The importers returned by pyximport.install().
    """

    import pyximport

    importers = pyximport.install(setup_args={'include_dirs': [np.get_include()]}, \
        build_dir=pyximportBuildDir())

    # Check the compiled files in the source tree once, when pyximport is installed
    pyx_finder = importers[1]
    if pyx_finder is not None:

        stale = staleCompiledModules()

        if stale:

            for compiled_path in stale.values():
                print("NOTE: {:s} was compiled before its .pyx was last changed (e.g. before a git pull), so "
                    "the module is compiled again from the .pyx. Delete the file, or compile it again with "
                    "\"python setup.py build_ext --inplace\", to remove this note.".format(compiled_path))

            # Put the finder before the one of Python, which would find the old compiled files
            sys.meta_path.insert(0, StaleCompiledModuleFinder(stale, pyx_finder))

    return importers
