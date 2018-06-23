import numpy as np

from wmpl.Utils.TrajConversions import J2000_JD, equatorialCoordPrecession

class InfData(object):

	def __init__(self, station_id, lat, lon, ele, jd_data, ra_data, dec_data):

		self.station_id = station_id
		self.lat = lat
		self.lon = lon
		self.ele = ele
		self.jd_data = np.array(jd_data)
		self.ra_data = np.array(ra_data)
		self.dec_data = np.array(dec_data)

		self.time_data = None




def loadINF(file_name):
	""" Loads data from the Croatian Meteor Network's INF file. """

	with open(file_name) as f:

		# Skip first 2 lines
		for i in range(2):
			f.readline()

		# Read the station ID
		station_id = f.readline().split()[1]

		# Read longitude
		lon = np.radians(float(f.readline().split()[1]))

		# Read latitude
		lat = np.radians(float(f.readline().split()[1]))

		# Read elevation
		ele = float(f.readline().split()[1])

		jd_data = []
		ra_data = []
		dec_data = []

		# Load JD, Ra and Dec data
		for line in f:

			if not line:
				continue

			line = line.split()

			line = map(float, line)

			# Extract Julian date
			jd = line[0]

			# Extract right ascension
			ra = np.radians(line[1])

			# Extract declination
			dec = np.radians(line[2])

			# Precess ra and dec to epoch of date (from J2000.0 epoch)
			ra, dec = equatorialCoordPrecession(J2000_JD.days, jd, ra, dec)

			jd_data.append(jd)
			ra_data.append(ra)
			dec_data.append(dec)


	inf = InfData(station_id, lat, lon, ele, jd_data, ra_data, dec_data)

	return inf



if __name__ == "__main__":

	pass