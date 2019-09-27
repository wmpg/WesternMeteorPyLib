""" Updates asteroid and comet orbit files from MPC and JPL website. """

from __future__ import print_function, division, absolute_import

import os
import sys
import ssl


# Fix certificates error
ssl._create_default_https_context = ssl._create_unverified_context


if sys.version_info.major < 3:
	import urllib as urllibrary

else:
	import urllib.request as urllibrary

def updateOrbitFiles():
	""" Updates asteroid and comet orbit files from MPC and JPL website. """


	# Comets
	comets_url = "https://ssd.jpl.nasa.gov/dat/ELEMENTS.COMET"

	# Amors
	amors_url = "http://cgi.minorplanetcenter.net/cgi-bin/textversion.cgi?f=lists/Amors.html"

	# Apollos
	apollos_url = "http://cgi.minorplanetcenter.net/cgi-bin/textversion.cgi?f=lists/Apollos.html"

	# Atens
	atens_url = "http://cgi.minorplanetcenter.net/cgi-bin/textversion.cgi?f=lists/Atens.html"


	dir_path = os.path.split(os.path.abspath(__file__))[0]
	dir_path = os.path.join(dir_path, 'wmpl', 'share')

	file_names = ['ELEMENTS.COMET', 'Amors.txt', 'Apollos.txt', 'Atens.txt']
	url_list = [comets_url, amors_url, apollos_url, atens_url]

	# Download all orbit files
	for fname, url in zip(file_names, url_list):
		print('Downloading {:s}...'.format(fname), end='')
		urllibrary.urlretrieve(url, os.path.join(dir_path, fname))
		print(' done!')




if __name__ == "__main__":

	updateOrbitFiles()