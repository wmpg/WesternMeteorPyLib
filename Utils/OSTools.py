""" OS and file system functions. """


import os
import errno


def mkdirP(path):
    """ Makes a directory and handles all errors.
    """

    # Try to make a directory
    try:
        os.makedirs(path)

    # If it already exist, do nothing
    except OSError, exc:
        if exc.errno == errno.EEXIST:
            pass

    # Raise all other errors
	else: 
		raise



def listDirRecursive(dir_path):
    """ Return a list of all files in the given directory tree.

    Arguments:
        dir_path: [str] Path to the directory.

    Return:
        [list] A list of full paths of all files.
    """


    return [os.path.join(os.path.abspath(dp), f) for dp, dn, fn in os.walk(dir_path) for f in fn]