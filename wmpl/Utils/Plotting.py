""" Functions for plotting purposes. """

from __future__ import absolute_import, print_function, division

import os

import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from Config import config
from wmpl.Utils.OSTools import mkdirP




def savePlot(plt_handle, file_path, output_dir='.', kwargs=None):
    """ Saves the plot to the given file path, with the DPI defined in configuration. 

    Arguments:
        plt_handle: [object] handle to the plot to the saved (usually 'plt' in the main program)
        file_path: [string] file name and path to which the plot will be saved

    Keyword arguments:
        kwargs: [dictionary] Extra keyword arguments to be passed to savefig. None by default.

    """


    if kwargs is None:
        kwargs = {}

    # Make the output directory, if it does not exist
    mkdirP(output_dir)

    # Save the plot (remove all surrounding white space)
    plt_handle.savefig(os.path.join(output_dir, file_path), dpi=config.plots_DPI, bbox_inches='tight', 
        **kwargs)




class Arrow3D(FancyArrowPatch):
    """ Arrow in 3D for plotting in matplotlib. 

    Arguments:
        xs: [list of floats] (origin, destination) pair for the X axis
        ys: [list of floats] (origin, destination) pair for the Y axis
        zs: [list of floats] (origin, destination) pair for the Z axis
    
    Source: 
        https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot

    """
    def __init__(self, xs, ys, zs, *args, **kwargs):

        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)

        self._verts3d = xs, ys, zs


    def draw(self, renderer):

        xs3d, ys3d, zs3d = self._verts3d

        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)

        self.set_positions((xs[0], ys[0]),(xs[1], ys[1]))

        FancyArrowPatch.draw(self, renderer)




def set3DEqualAxes(ax):
    """ Make axes of 3D plot have equal scale so that spheres appear as spheres, cubes as cubes, etc. 
        This is one possible solution to Matplotlib's ax.set_aspect('equal') and ax.axis('equal') not working 
        for 3D.

    Source: https://stackoverflow.com/a/31364297

    Arguments:
        ax: [matplotlib axis] Axis handle of a 3D plot.
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity norm, hence I call half the max range 
    # the plot radius
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])




if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Test the arrow function
    a = Arrow3D([0.2, 3], [1, -2], [0, 1], mutation_scale=20, lw=3, arrowstyle="-|>", color="r")

    ax.add_artist(a)

    plt.show()