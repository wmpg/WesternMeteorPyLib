""" Configuration for running the library. It is loaded by several modules to read in the default settings.
"""


from __future__ import print_function, division, absolute_import

import os

class ConfigStruct(object):

	def __init__(self):

		# VSOP87 file location
		self.vsop_file = os.path.join('share', 'VSOP87D.ear')


		# DE430 JPL ephemerids file location
		self.jpl_ephem_file = os.path.join('share', 'de430.bsp')

		# Meteor simulation default parameters file
		self.met_sim_input_file = os.path.join('MetSim', 'Metsim0001_input.txt')

		# DPI of saves plots
		self.plots_DPI = 300





# Init the configuration structure
config = ConfigStruct()