""" OS and file system functions. """

from __future__ import print_function, absolute_import, division

import os
import sys
import errno
# not used here but required to force-load Cartopy before basemap. Otherwise
# loadBaseMap() crashes on Windows 10
import cartopy.io.img_tiles as cimgt 

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


def walkDirsToDepth(dir_path, depth=-1):
    """ Mimic os.walk, but define the maximum depth. 
    
    Arguments:
        dir_path: [str] Path to the directory.

    Keyword arguments:
        depth: [int] Maximum depth. Use -1 for no limit, in which case the function behaves the same as
            os.walk.

    Return:
        file_list: [list] List where the elements are:
            - dir_path - path to the directory
            - dir_list - list of directories in the path
            - file_list - list of files in the path
    """
    
    final_list = []
    dir_list = []
    file_list = []

    # Find all files and directories in the given path and sort them accordingly
    for entry in sorted(os.listdir(dir_path)):

        entry_path = os.path.join(dir_path, entry)

        if os.path.isdir(entry_path):
            dir_list.append(entry)

        else:
            file_list.append(entry)


    # Mimic the output of os.walk
    final_list.append([dir_path, dir_list, file_list])


    # Decrement depth for the next recursion
    depth -= 1

    # Stop the recursion if the final depth has been reached
    if depth != 0:

        # Do this recursively for all directories up to a certain depth
        for dir_name in dir_list:

            final_list_rec = walkDirsToDepth(os.path.join(dir_path, dir_name), depth=depth)

            # Add the list to the total list
            final_list += final_list_rec


    return final_list


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

            try:
                import conda

                conda_file_dir = conda.__file__
                conda_dir = conda_file_dir.split('lib')[0]

            except:

                # If no "conda" module is found, try making a path from matplotlib
                import matplotlib

                mpl_path = matplotlib.__file__
                mpl_path_split = mpl_path.split(os.sep)
                conda_dir = os.sep.join(mpl_path_split[:mpl_path_split.index("anaconda3") + 1])


            if sys.platform == "linux":
                proj_lib = os.path.join(conda_dir, "envs", "wmpl", "share", "basemap")
            else:
                proj_lib = os.path.join(conda_dir, "envs", "wmpl", "Library", "share")

            os.environ["PROJ_LIB"] = proj_lib


            try:
                from mpl_toolkits.basemap import Basemap

            except FileNotFoundError:

                # The epsg file is probably missing
                print("The epsg file is probably missing, download it from the following link and put it in the path given above: https://raw.githubusercontent.com/matplotlib/basemap/master/lib/mpl_toolkits/basemap/data/epsg")
                sys.exit()


        except import_error:

            # Throw an error if everything failed
            from mpl_toolkits.basemap import Basemap

        except FileNotFoundError:

            # The epsg file is probably missing
            print("The epsg file is probably missing, download it from the following link and put it in the path given above: https://raw.githubusercontent.com/matplotlib/basemap/master/lib/mpl_toolkits/basemap/data/epsg")
            sys.exit()



    return Basemap