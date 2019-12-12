""" OS and file system functions. """

from __future__ import print_function, absolute_import, division

import os
import sys
import errno


def mkdirP(path):
    """ Makes a directory and handles all errors.
    """

    # Try to make a directory
    try:
        os.makedirs(path)

    # If it already exist, do nothing
    except OSError as exc:
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



def importBasemap():
    """ Fix basemap import when PROJ_LIB is not defined. """

    try:
        # Try importing Basemap as is
        from mpl_toolkits.basemap import Basemap
    
    except KeyError:

        # Choose which error will be checked for if no "conda" module is found
        if sys.version_info.major < 3:
            import_error = ImportError
        else:
            import_error = ModuleNotFoundError

        try:
            # If PROJ_LIB could not be found, add it to the enviroment
            import os
            import conda

            conda_file_dir = conda.__file__
            conda_dir = conda_file_dir.split('lib')[0]
            proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
            os.environ["PROJ_LIB"] = proj_lib

            try:
                from mpl_toolkits.basemap import Basemap

            except FileNotFoundError:

                # The epsg file is probably missing
                print("The epsg file is probably missing, download it from the following link and put it in the path given above: https://raw.githubusercontent.com/matplotlib/basemap/master/lib/mpl_toolkits/basemap/data/epsg")
                sys.exit()


        except import_error:

            # If no "conda" module is found, give up
            from mpl_toolkits.basemap import Basemap

        except FileNotFoundError:

            # The epsg file is probably missing
            print("The epsg file is probably missing, download it from the following link and put it in the path given above: https://raw.githubusercontent.com/matplotlib/basemap/master/lib/mpl_toolkits/basemap/data/epsg")
            sys.exit()



    return Basemap