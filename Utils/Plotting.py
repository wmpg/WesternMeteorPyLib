""" Functions for plotting purposes. """

from __future__ import absolute_import, print_function, division

import os

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from Config import config
from Utils.OSTools import mkdirP




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





if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Test the arrow function
    a = Arrow3D([0.2, 3], [1, -2], [0, 1], mutation_scale=20, lw=3, arrowstyle="-|>", color="r")

    ax.add_artist(a)

    plt.show()