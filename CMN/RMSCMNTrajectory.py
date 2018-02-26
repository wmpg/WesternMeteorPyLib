from __future__ import print_function, division, absolute_import

import os

import numpy as np

from CMN.CMNFormats import loadINF
from Formats.CAMS import loadFTPDetectInfo

from Trajectory.Trajectory import Trajectory
from Utils.TrajConversions import equatorialCoordPrecession_vect, J2000_JD


if __name__ == "__main__":


	ftp_stations = {'HR0002': [np.radians(45.34813055555556), np.radians(14.050744444444444), 337]}

	ftpdetectinfo_file = "C:\Users\delorayn1\Desktop\HR0002_20180119_162419_928144_detected\FTPdetectinfo_HR0002_20180119_162419_928144_temporal.txt"

	inf_dir = "C:\Users\delorayn1\Desktop\ImplementCorrection\CMN inf files"
	inf_files = ["M_2018011920DUI0003.inf", "M_2018011920VIB0002.inf"]

	# Load meteor from FTPdetectinfo
	ftp = loadFTPDetectInfo(ftpdetectinfo_file, ftp_stations)

	ftp_meteor = ftp[2]


	# Referent JD
	jdt_ref = ftp_meteor.jdt_ref - (ftp_meteor.time_data[0] - 10.0)/86400
	ftp_meteor.time_data -= ftp_meteor.time_data[0]

	output_dir = "C:\Users\delorayn1\Desktop\ImplementCorrection\FF_HR0002_20180119_210212_876_0414720"

	# Init the new trajectory
	traj = Trajectory(jdt_ref, meastype=1, max_toffset=15.0, output_dir=output_dir, monte_carlo=False)


	# Precess RA/Dec to J2000
	ra_data, dec_data = equatorialCoordPrecession_vect(J2000_JD.days, ftp_meteor.jdt_ref + ftp_meteor.time_data/86400, ftp_meteor.ra_data, ftp_meteor.dec_data)

	ra_data, dec_data = ftp_meteor.ra_data, ftp_meteor.dec_data
	# Infill trajectory with RMS data
	traj.infillTrajectory(ra_data, dec_data, ftp_meteor.time_data, \
		ftp_meteor.latitude, ftp_meteor.longitude, ftp_meteor.height, station_id=ftp_meteor.station_id)


	# Load INF files
	for inf in inf_files:
		
		inf_met = loadINF(os.path.join(inf_dir, inf))

		time_data = (inf_met.jd_data - jdt_ref)*86400

		print('INF time:', time_data)
		print('INF ra, dec:', np.degrees(inf_met.ra_data), np.degrees(inf_met.dec_data))

		# Precess RA/Dec to J2000
		ra_data, dec_data = equatorialCoordPrecession_vect(J2000_JD.days, inf_met.jd_data, inf_met.ra_data, \
			inf_met.dec_data)
		#ra_data, dec_data = inf_met.ra_data, inf_met.dec_data

		traj.infillTrajectory(ra_data, dec_data, time_data, inf_met.lat, inf_met.lon, inf_met.ele, \
			station_id=inf_met.station_id)


	traj.run()

		

