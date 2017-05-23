""" Functions for plotting purposes. """

from __future__ import absolute_import, print_function, division

import os

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

