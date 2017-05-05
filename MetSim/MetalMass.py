""" Loading and calculating masses from METAL .met files. """

from __future__ import print_function, absolute_import, division


import numpy as np
import scipy.interpolate

import matplotlib.pyplot as plt

from Formats.Met import loadMet



def calcMass(time, mag_abs, velocity):
	""" Calculates the mass of a meteoroid from the time and absolute magnitude. 
	
	Arguments:
		time: [ndarray] time of individual magnitude measurement (s)
		mag_abs: [nadrray] absolute magnitudes (i.e. apparent meteor magnitudes @100km)
		velocity: [float or ndarray] average velocity of the meteor, or velocity at every point of the meteor
			in m/s

	Return:
		mass: [float] photometric mass of the meteoroid in kg

	"""

	# Theory:
	# I = P_0m*10^(-0.4*M_abs)
	# M = (2/tau)*integral(I/v^2 dt)

	# Luminous efficiency = 0.7% (Ceplecha & McCrosky, 1976)
	tau = 0.7/100

	# Calculate the intensities from absolute magnitudes
	# The number P_0m = 840W is the power output for a zero absolute magnitude meteor in the R bandpass (we are
	# using stars in the R band for photometry), for T = 4500K.
	# Weryk & Brown, 2013 - "Simultaneous radar and video meteors - II. Photometry and ionisation"
	P_0m = 840.0
	intens = P_0m*10**(-0.4*mag_abs)

	# Interpolate I/v^2
	intens_interpol = scipy.interpolate.CubicSpline(time, intens)

	# x_data = np.linspace(np.min(time), np.max(time), 1000)
	# plt.plot(x_data, intens_interpol(x_data))
	# plt.scatter(time, intens/(velocity**2))
	# plt.show()

	# Integrate the interpolated I/v^2
	intens_int = intens_interpol.integrate(np.min(time), np.max(time))

	# Calculate the mass
	mass = (2.0/(tau*velocity**2))*intens_int

	return mass

	

def loadMetalMags(dir_path, file_name):
	""" Loads time and absolute magnitudes (apparent @100km) from the METAL .met file where the photometry has
		been done on a meteor.

	Arguments:
		dir_path: [str] path to the directory where the METAL .met file is located
		file_name: [str] name of the METAL .met file

	Return:
		[(time1, mag_abs1), (time2, mag_abs2),...]: [list of tuples of ndarrays] Time in seconds and absolute 
			magnitudes
	
	"""

	# Load the METAL-style met file
	met = loadMet(dir_path, file_name)

	time_mags = []

	# Extract the time and absolute magnitudes from all sites
	for site in met.sites:
		
	 	# Extract time, range, and apparent magnitude
	 	data = np.array([[pick[29], pick[31], pick[17]] for pick in met.picks[site]])

	 	# Remove rows with infinite magnitudes
	 	data = data[data[:, 2] != np.inf]

		
	 	# Split into time, range and apparent magnitude
	 	time, r, mag_app = data.T

	 	# Calculate absolute magnitude (apparent magntiude @100km range)
	 	mag_abs = mag_app + 5*np.log10(100.0/r)

	 	# Append the time and magnitudes for this site
	 	time_mags.append((site, time, mag_abs))


	return time_mags




if __name__ == '__main__':

	#dir_path = "/home/dvida/Dropbox/UWO Master's/Projects/MirfitPrepare/20161007_052749_met"
	dir_path = "/home/dvida/Dropbox/UWO Master's/Projects/MetalPrepare/20161007_052346_met"

	file_name = 'state.met'

	time_mags = loadMetalMags(dir_path, file_name)

	# m/s
	#v_avg = 23541.10
	v_avg = 26972.28


	masses = []

	for site, time, mag_abs in time_mags:

	 	plt.plot(time, mag_abs)
	 	plt.gca().invert_yaxis()
		plt.show()


		# Calculate the mass from magnitudes
		mass = calcMass(time, mag_abs, v_avg)

		masses.append(mass)

		print('Mass:', mass)
		print(np.log10(mass))



	print('Median mass:', np.median(masses))



