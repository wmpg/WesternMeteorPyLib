""" Updates asteroid and comet orbit files from MPC and JPL website. """

from __future__ import print_function, division, absolute_import

import os
import sys
import ssl
import socket
import shutil

# Fix certificates error
ssl._create_default_https_context = ssl._create_unverified_context


if sys.version_info.major < 3:
	import urllib as urllibrary

else:
	import urllib.request as urllibrary

def updateOrbitFiles():
	""" Updates asteroid and comet orbit files from MPC and JPL website. """

	# Set a 15 second connection timeout so the compile is not held by a hanging download
	socket.setdefaulttimeout(15)

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

	# Temporary file path
	temp_path = os.path.join(dir_path, "temp.part")

	# Download all orbit files
	for fname, url in zip(file_names, url_list):
		print('Downloading {:s}...'.format(fname))

		# Download the file to a temporary location (as not to overwrite the existing file)
		try:

			# Download the file to a temporary location
			urllibrary.urlretrieve(url, temp_path)

			# Move the downloaded file to a final location
			shutil.move(temp_path, os.path.join(dir_path, fname))

		except Exception as e:
			print("Download failed with:" + repr(e))


		print(' ... done!')




if __name__ == "__main__":

	updateOrbitFiles()