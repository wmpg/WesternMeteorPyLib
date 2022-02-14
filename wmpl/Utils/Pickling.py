""" Functions for pickling and unpickling Python objects. """

from __future__ import absolute_import, print_function

import os
import pickle
import sys

from wmpl.Utils.OSTools import mkdirP


def savePickle(obj, dir_path, file_name):
    """ Dump the given object into a file using Python 'pickling'. The file can be loaded into Python
        ('unpickled') afterwards for further use.

    Arguments:
    	obj: [object] Object which will be pickled.
        dir_path: [str] Path of the directory where the pickle file will be stored.
        file_name: [str] Name of the file where the object will be stored.

    """

    mkdirP(dir_path)

    with open(os.path.join(dir_path, file_name), 'wb') as f:
        pickle.dump(obj, f, protocol=2)


def loadPickle(dir_path, file_name):
    """ Loads pickle file.
	
	Arguments:
		dir_path: [str] Path of the directory where the pickle file will be stored.
        file_name: [str] Name of the file where the object will be stored.

    """

    with open(os.path.join(dir_path, file_name), 'rb') as f:

        # Python 2
        if sys.version_info[0] < 3:
            p = pickle.load(f)

        # Python 3
        else:
            p = pickle.load(f, encoding='latin1')

        # Fix attribute compatibility in trajectory objects with older versions which had a
        #   typo "uncertanties"
        if hasattr(p, "uncertainties"):
            p.uncertanties = p.uncertainties

        if hasattr(p, "uncertanties"):
            p.uncertainties = p.uncertanties

        # Check if the pickle file is a trajectory file
        if hasattr(p, 'orbit') and hasattr(p, 'observations'):
            if p.orbit is not None:
                p.orbit.fixMissingParameters()

        return p
