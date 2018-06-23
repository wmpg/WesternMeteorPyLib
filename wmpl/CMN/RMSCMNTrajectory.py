from __future__ import print_function, division, absolute_import

import os

import numpy as np

from wmpl.CMN.CMNFormats import loadINF
from wmpl.Formats.CAMS import loadFTPDetectInfo

from wmpl.Trajectory.Trajectory import Trajectory
from wmpl.Utils.TrajConversions import equatorialCoordPrecession_vect, J2000_JD


if __name__ == "__main__":


	ftp_stations = {'HR0002': [np.radians(45.34813055555556), np.radians(14.050744444444444), 337]}

	#ftpdetectinfo_file = "C:\\Users\\delorayn1\\Desktop\\HR0002_20180119_162419_928144_detected\\FTPdetectinfo_HR0002_20180119_162419_928144_temporal.txt"
	ftpdetectinfo_file = "C:\\Users\\delorayn1\\Desktop\\HR0002_20180119_162419_928144_detected\\FTPdetectinfo_HR0002_20180121_162654_728788_temporal.txt"


	# Load meteor from FTPdetectinfo
	ftp = loadFTPDetectInfo(ftpdetectinfo_file, ftp_stations)

	inf_dir = "C:\\Users\\delorayn1\\Desktop\\ImplementCorrection\\CMN inf files"
		

	## METEOR 1
	#inf_files = ["M_2018011920DUI0003.inf", "M_2018011920VIB0002.inf"]
	#inf_files = ["M_2018011920DUI0003.inf"]
	inf_files = ["M_2018011920VIB0002.inf"]

	ftp_meteor = ftp[2]

	output_dir = "C:\\Users\\delorayn1\\Desktop\\ImplementCorrection\\FF_HR0002_20180119_210212_876_0414720"
	############


	## METEOR 2
	inf_files = ["M_2018011920DUI0012.inf"]

	ftp_meteor = ftp[8]

	output_dir = "C:\\Users\\delorayn1\\Desktop\\ImplementCorrection\\FF_HR0002_20180119_233859_048_0641536.fits"
	############


	## METEOR 3
	inf_files = ["M_2018011920MLA0009.inf"]

	ftp_meteor = ftp[17]

	output_dir = "C:\\Users\\delorayn1\\Desktop\\ImplementCorrection\\FF_HR0002_20180120_014014_887_0817664"
	############


	## METEOR 4
	inf_files = ["M_2018011920CIO0006.inf"]

	ftp_meteor = ftp[29]

	output_dir = "C:\\Users\\delorayn1\\Desktop\\ImplementCorrection\\FF_HR0002_20180120_050021_373_1105920"
	############


	#########

	## METEOR 5
	inf_files = ["M_2018012122VIB0002.inf"]

	ftp_meteor = ftp[7]

	output_dir = "C:\\Users\\delorayn1\\Desktop\\ImplementCorrection\\FF_HR0002_20180121_191911_939_0254720"
	############



	# reference JD
	jdt_ref = ftp_meteor.jdt_ref - (ftp_meteor.time_data[0] - 10.0)/86400
	ftp_meteor.time_data -= ftp_meteor.time_data[0]

	# Init the new trajectory
	traj = Trajectory(jdt_ref, meastype=1, max_toffset=15.0, output_dir=output_dir, monte_carlo=True)


	# Precess RA/Dec from J2000 to epoch of date
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

		# Precess RA/Dec from J2000 to epoch of date
		ra_data, dec_data = equatorialCoordPrecession_vect(J2000_JD.days, inf_met.jd_data, inf_met.ra_data, \
			inf_met.dec_data)
		#ra_data, dec_data = inf_met.ra_data, inf_met.dec_data

		traj.infillTrajectory(ra_data, dec_data, time_data, inf_met.lat, inf_met.lon, inf_met.ele, \
			station_id=inf_met.station_id)


	traj.run()

		

