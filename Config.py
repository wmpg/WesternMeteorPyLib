""" Configuration for running the library. It is loaded by several modules to read in the default settings.
"""


from __future__ import print_function, division, absolute_import

import os

class ConfigStruct(object):

	def __init__(self):

		# Find the absolute path of the directory where this file is located
		abs_path = os.path.split(os.path.abspath(__file__))[0]

		### EPHEMERIDS

		# VSOP87 file location
		self.vsop_file = os.path.join(abs_path, 'share', 'VSOP87D.ear')

		# DE430 JPL ephemerids file location
		self.jpl_ephem_file = os.path.join(abs_path, 'share', 'de430.bsp')

		###


		### PARENT BODY ORBITAL ELEMENTS

		# JPL comets elements
		self.comets_elements_file = os.path.join(abs_path, 'share', 'ELEMENTS.COMET')

		# Asteroid elements
		self.asteroids_amors_file = os.path.join(abs_path, 'share', 'Amors.txt')
		self.asteroids_apollos_file = os.path.join(abs_path, 'share', 'Apollos.txt')
		self.asteroids_atens_file = os.path.join(abs_path, 'share', 'Atens.txt')

		###

		# Meteor simulation default parameters file
		self.met_sim_input_file = os.path.join(abs_path, 'MetSim', 'Metsim0001_input.txt')

		# DPI of saves plots
		self.plots_DPI = 300





# Init the configuration structure
config = ConfigStruct()