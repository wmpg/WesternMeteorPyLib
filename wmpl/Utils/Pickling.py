""" Functions for pickling and unpickling Python objects. """

from __future__ import print_function, absolute_import

import io
import os
import sys
import uuid
import pickle
import logging


from wmpl.Utils.OSTools import mkdirP



def savePickle(obj, dir_path, file_name):
    """ Dump the given object into a file using Python 'pickling'. The file can be loaded into Python
        ('unpickled') afterwards for further use.

        The object is written to a temporary file which is then moved onto the target name, so a
        reader watching for the file never sees a partially written one. This matters for the
        correlator, where one process writes trajectories that another polls for.

    Arguments:
    	obj: [object] Object which will be pickled.
        dir_path: [str] Path of the directory where the pickle file will be stored.
        file_name: [str] Name of the file where the object will be stored.

    """

    mkdirP(dir_path)

    # The temporary name must not contain ".pickle", because the correlator globs for "*.pickle*" to
    #   find work, and it must be unique, so that two processes writing the same file name do not
    #   share a temporary file. It carries no extension at all.
    tmp_path = os.path.join(dir_path, "_tmp_" + uuid.uuid4().hex)

    with open(tmp_path, 'wb') as f:
        pickle.dump(obj, f, protocol=2)

    # os.replace is atomic on POSIX and Windows alike, and overwrites the target without a separate
    #   remove, which would otherwise leave a window where the file does not exist at all
    os.replace(tmp_path, os.path.join(dir_path, file_name))



class NPCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        """ Special unpickler to handle numpy._core rename to numpy.core. """
        
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')

        return super().find_class(module, name)


def loadPickle(dir_path, file_name):
    """ Loads pickle file.

    Arguments:
        dir_path: [str] Path of the directory where the pickle file will be stored.
        file_name: [str] Name of the file where the object will be stored.
    """

    path = os.path.join(dir_path, file_name)

    with open(path, 'rb') as f:
        buf = f.read()

    def _loads_compat(b):
        # Python 2
        if sys.version_info[0] < 3:
            return pickle.loads(b)
        # Python 3
        return pickle.loads(b, encoding='latin1')

    # First try normal unpickling
    try:
        p = _loads_compat(buf)

    except ModuleNotFoundError as e:
        # Fallback only for numpy._core rename
        if "numpy._core" not in str(e):
            raise
        
        # Custom unpickler to fix numpy._core rename
        p = NPCompatUnpickler(io.BytesIO(buf)).load()

    # Fix attribute compatibility in trajectory objects with older versions which had a
    # typo "uncertanties"
    if hasattr(p, "uncertainties"):
        p.uncertanties = p.uncertainties

    if hasattr(p, "uncertanties"):
        p.uncertainties = p.uncertanties


    # Check if the pickle file is a trajectory file
    if hasattr(p, 'orbit') and hasattr(p, 'observations'):

        # If there is no orbit object, create one
        if p.orbit is not None:
            p.orbit.fixMissingParameters()

        # If the gravity factor is missing, add it
        if not hasattr(p, 'gravity_factor'):
            p.gravity_factor = 1.0

        # If v0z is missing, add it
        if not hasattr(p, 'v0z'):
            p.v0z = 0.0

        # If the L-BFGS-B station cutoff is missing, add its default. Trajectories pickled before the
        #   cutoff became a parameter do not carry it, and estimateTimingAndVelocity reads it.
        if not hasattr(p, 'l_bfgs_b_cutoff'):
            p.l_bfgs_b_cutoff = 5

        # If the logger is missing, attach the module's own. Trajectories pickled before Trajectory
        #   kept its logger on the instance have no "log" attribute, and every method that logs
        #   reads it. Loggers pickle by name, so newer pickles carry it already.
        if not hasattr(p, 'log'):
            p.log = logging.getLogger("wmpl_logger")

    return p

