
import os
import sys

import matplotlib.pyplot as plt

from CMN.CMNFormats import loadINF
from Trajectory.Trajectory import Trajectory
from Trajectory.GuralTrajectory import GuralTrajectory


if __name__ == "__main__":

	dir_path = "CMN" + os.sep + "cmn_fireball_2017030506"
	files = ['M_2017030506APO0001.txt', 'M_2017030506KOP0001.txt']


	observations = []
	
	# Load observed data
	for file_name in files:

		observations.append(loadINF(os.path.join(dir_path, file_name)))


	# Get the reference JD from the first site
	jdt_ref = observations[0].jd_data[0]

	max_first = 0
	
	# Recalculate time data
	for obs in observations:
		obs.time_data = (obs.jd_data - jdt_ref)*86400.0

		# # Normalize all time data so their beginning is at 0
		# obs.time_data -= obs.time_data[0]



	# sys.exit()


	# Init the Trajectory solver
	traj_solve = Trajectory(jdt_ref, max_toffset=1, meastype=1, estimate_timing_vel=False)

	#traj_solve = GuralTrajectory(len(observations), jdt_ref, velmodel=1, max_toffset=1.0, nummonte=1, meastype=1, verbose=1)

	# Infill the observed data
	for obs in observations:
		traj_solve.infillTrajectory(obs.ra_data, obs.dec_data, obs.time_data, obs.lat, obs.lon, obs.ele)


	# Solve the trajectory
	traj_solve.run()
