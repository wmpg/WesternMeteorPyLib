import rebound as rb
import numpy as np
import reboundx
import scipy
from reboundx import constants as rbxConstants
from wmpl.Utils.TrajConversions import (
    J2000_JD,
    equatorialCoordPrecession,
    jd2DynamicalTimeJD,
    eci2RaDec,
    vectMag,
    raDec2ECI
)
from wmpl.Utils.Earth import calcTrueObliquity
from astropy.time import Time


def convertToBarycentric(state_vect, jd, log_file_path=""):
    """ Takes a state vector in ECI coordinates (m and m/s), Julian date and converts from ECI (geocentric) to 
    Solar System barycentric coordiantes. The units are changed to AU and AU/year.

    Arguments:
        state_vect: [list] Position and velocity components in ECI coordinates (epoch of date) in m and m/s, 
        [x, y, z, vx, vy, vz], as given in the WMPL trajectory solution.
        jd: [float] Julian date (decimal).

    Keyword arguments:
        log_file_path: [str] Path to the log file where the output will be written.

    Return:
        [list] Position and velocity components in barycentric coordinates in AU and AU/year, eg. 
            [x, y, z, vx, vy, vz]
    """

    # If a log file is specified, open it
    if len(log_file_path):
        log_file = open(log_file_path, "w")
        
        def log(message):
            log_file.write(message + "\n")

        log(f"Reference Julian date (TDB): {jd}")

    # Create a REBOUND simulation
    sim = rb.Simulation()

    # Uses JPL Horizons to query for J2000 ecliptic SSB, geometric states vector correction
    sim.add("Geocenter", date=f"JD{jd:.6f}", hash="Earth")

    ps = sim.particles

    # Convert the state vector to REBOUND units
    aum = rb.units.lengths_SI["au"]

    # Extract the position and velocity vectors
    eci_pos = np.array(state_vect[:3])
    eci_vel = np.array(state_vect[3:])

    if len(log_file_path):
        log(f"Position vector in equatorial ECI, epoch of date (m): {eci_pos}")
        log(f"Velocity vector in equatorial ECI, epoch of date (m/s): {eci_vel}")

    # Convert rectangular to spherical coordinates
    re = vectMag(eci_pos)
    alpha_e, delta_e = eci2RaDec(eci_pos)

    # Convert the Julian date to dynamical time
    jd_dyn = jd2DynamicalTimeJD(jd)
    alpha_ej, delta_ej = equatorialCoordPrecession(
        jd_dyn, J2000_JD.days, alpha_e, delta_e
    )

    if len(log_file_path):
        log(f"RA precessed to J2000: {alpha_ej}")
        log(f"Declination precessed to J2000: {delta_ej}")

    geo_x, geo_y, geo_z = raDec2ECI(alpha_ej, delta_ej)
    geo_pos = np.array([geo_x*re, geo_y*re, geo_z*re])

    if len(log_file_path):
        log(f"Position vector in equatorial ECI, J2000: {geo_pos}")

    eps = calcTrueObliquity(J2000_JD.days)
    eps = -eps
    rotM = scipy.linalg.expm(
        np.cross(
            np.eye(3),
            np.array([1, 0, 0])/np.linalg.norm(np.array([1, 0, 0]))*eps,
        )
    )
    pos_vec_rot = np.dot(rotM, geo_pos)

    if len(log_file_path):
        log(f"Position vector in ecliptic ECI, J2000: {pos_vec_rot}")

    v_inf = vectMag(eci_vel)
    alpha_e_vel, delta_e_vel = eci2RaDec(eci_vel)
    alpha_ej_vel, delta_ej_vel = equatorialCoordPrecession(
        jd_dyn, J2000_JD.days, alpha_e_vel, delta_e_vel
    )

    if len(log_file_path):
        log(f"RA of velocity direction precessed to J2000: {alpha_ej_vel}")
        log(f"Declination of velocity direction precessed to J2000: {delta_e_vel}")
        log(f"V_inf = {v_inf}")

    geo_vx, geo_vy, geo_vz = raDec2ECI(alpha_ej_vel, delta_ej_vel)
    geo_vel = np.array([-v_inf*geo_vx, -v_inf*geo_vy, -v_inf*geo_vz])

    if len(log_file_path):
        log(f"Velocity vector in equatorial ECI, J2000: {geo_vel}")

    vel_vec_rot = np.dot(rotM, geo_vel)

    if len(log_file_path):
        log(f"Velocity vector in ecliptic ECI, J2000: {vel_vec_rot}")

    # Convert the state vector to AU and AU/year
    state_vect_rot = np.concatenate((pos_vec_rot, vel_vec_rot))
    state_vect_rot = [x/aum for x in state_vect_rot]
    state_vect_rot[3] *= 60*60*24*365.25/(2*np.pi)
    state_vect_rot[4] *= 60*60*24*365.25/(2*np.pi)
    state_vect_rot[5] *= 60*60*24*365.25/(2*np.pi)

    if len(log_file_path):
        log(
            f"Meteor state vector in ecliptic ECI, J2000 (AU, AU/year where year = 2pi): {state_vect_rot}"
        )
        log(
            f"Earth state vector in ecliptic solar system barycentric, J2000 (AU, AU/year where year = 2pi): {[ps['Earth'].x, ps['Earth'].y, ps['Earth'].z, ps['Earth'].vx, ps['Earth'].vy, ps['Earth'].vz]}"
        )

    # Add the Earth's position and velocity to the meteoroid's position and velocity
    state_vect_rot[0] += ps["Earth"].x
    state_vect_rot[1] += ps["Earth"].y
    state_vect_rot[2] += ps["Earth"].z
    state_vect_rot[3] += ps["Earth"].vx
    state_vect_rot[4] += ps["Earth"].vy
    state_vect_rot[5] += ps["Earth"].vz

    if len(log_file_path):
        log(
            f"Meteor state vector in barycentric, J2000 (AU, AU/year where year = 2pi): {state_vect_rot}"
        )
        log_file.close()

    return state_vect_rot


def reboundSimulate(
        julian_date, state_vect, traj=None, 
        direction="forward", tsimend=60, n_outputs=500, obj_name="obj", obj_mass=0.0, 
        state_vect_log=""):
    """ Takes an state vector (or a Trajectory object), runs REBOUND and produces orbital elements for the 
    object at the end of the simulation or at the specified time.

    Arguments:
        julian_date: [float] Reference Julian date (decimal). If None, the data from the Trajectory object 
            will be used.
        state_vect: [list] Position and velocity components in ECI coordinates (epoch of date) in m and m/s,
            [x, y, z, vx, vy, vz], as given in the WMPL trajectory solution. If None, the data from the
            Trajectory object will be used.

    Keyword arguments:
        traj: [Trajectory] Trajectory object with the meteoroid data. If given, the julian_date and state_vect
            arguments will be ignored.
        direction: [str] Direction of the simulation, either "forward" or "backward".
        tsimend: [float] Length of integration in days, default is 60 days.
        n_outputs: [int] Number of outputs (samples along the simulation), default is 500.
        obj_name: [str] Name of the object that's being integrated, default is "obj".
        obj_mass: [float] Mass of object in solar masses if asteroid or larger object, default is 0.0.

    Keyword arguments:
        state_vect_log: [str] Path to a log file where the state vector output will be written. If not given,
            the output will be written to the standard output.

    Return:
        None

    """

    # Set up the simulation
    sim = rb.Simulation()
    aum = rb.units.lengths_SI["au"]  # 1 au in m
    aukm = aum/1e3  # au in km


    # Constants (gravitational harmonics of Earth, Earth radius, time conversion)
    RE_eq = 6378.135/aukm
    J2 = 1.0826157e-3
    J4 = -1.620e-6
    dmin = 4.326e-5  # Earth radius in au
    tsimend = tsimend/365.25  # simulation endtime in years
    year = 2.0*np.pi  # One year in units where G=1


    # Convert from UTC to TDB
    time_utc = Time(julian_date, format='jd', scale='utc')
    time_tdb = time_utc.tdb.jd

    # Set up the number of outputs and the time array
    if direction == "forward":
        sim.dt = 0.001
        times = np.linspace(0, year*tsimend, n_outputs)

    else:
        sim.dt = -0.001
        times = np.linspace(0, -year*tsimend, n_outputs))

    # Add the Sun and the planets
    sim.add("Sun", date=f"JD{time_tdb:.6f}", hash="Sun")
    sim.add("Mercury", date=f"JD{time_tdb:.6f}", hash="Mercury")
    sim.add("Venus", date=f"JD{time_tdb:.6f}", hash="Venus")
    sim.add("Geocenter", date=f"JD{time_tdb:.6f}", hash="Earth")
    sim.add("Luna", date=f"JD{time_tdb:.6f}", hash="Luna")
    sim.add("Mars", date=f"JD{time_tdb:.6f}", hash="Mars")
    sim.add("Jupiter", date=f"JD{time_tdb:.6f}", hash="Jupiter")
    sim.add("Saturn", date=f"JD{time_tdb:.6f}", hash="Saturn")
    sim.add("Uranus", date=f"JD{time_tdb:.6f}", hash="Uranus")
    sim.add("Neptune", date=f"JD{time_tdb:.6f}", hash="Neptune")

    ps = sim.particles

    # To start the meteoroid in the right spot, we need to feed in x,y,z barycentric position in AU
    # The velocity should be in AU/year divided by 2pi
    state_vect_rot = convertToBarycentric(state_vect, time_tdb, state_vect_log)

    # Add the meteoroid to the simulation
    sim.add(
        m=obj_mass,
        x=state_vect_rot[0],
        y=state_vect_rot[1],
        z=state_vect_rot[2],
        vx=state_vect_rot[3],
        vy=state_vect_rot[4],
        vz=state_vect_rot[5],
        hash=obj_name,
    )

    # Add gravitational harmonics of Earth
    rebx = reboundx.Extras(sim)
    gh = rebx.load_force("gravitational_harmonics")
    rebx.add_force(gh)
    ps["Earth"].params["J2"] = J2
    ps["Earth"].params["J4"] = J4
    ps["Earth"].params["R_eq"] = RE_eq
    ps["Earth"].r = dmin  # set size of Earth
    ps["Luna"].r = dmin/4  # set size of Moon


    # gr is the general relativity correction for just the Sun, gr_full is the correction for all bodies
    gr = rebx.load_force("gr_full")
    rebx.add_force(gr)
    gr.params["c"] = rbxConstants.C

    # We always move to the center of momentum frame before an integration
    sim.move_to_com()

    # Specify how simulation should resolve collisions
    sim.collision = "direct"
    sim.collision_resolve = "merge"
    sim.collision_resolve_keep_sorted = 1
    sim.track_energy_offset = 1

    with open(f"{obj_name}.txt", "a") as log_file:

        def log(message):
            log_file.write(message + "\n")

        for i, time in enumerate(times):
            if i % 25 == 0:
                print(f"loop iter = {i}, {time}")
            sim.move_to_com()
            sim.integrate(time)
            sim.move_to_hel()
            if i == 0:
                log(f"Heliocentric J2000 State Vec: {ps[obj_name]}")
        log("Integrated Object Orbital Elements (J2000, Heliocentric, 60 days prior):")
        log(f'a (AU): {ps[obj_name].orbit(primary=ps["Sun"]).a}')
        log(f'e: {ps[obj_name].orbit(primary=ps["Sun"]).e}')
        log(f'i (degrees): {np.rad2deg(ps[obj_name].orbit(primary=ps["Sun"]).inc)}')
        log(
            f'node (degrees): {np.rad2deg(ps[obj_name].orbit(primary=ps["Sun"]).Omega)}'
        )
    return


if __name__ == "__main__":

    import os
    import argparse

    from wmpl.Utils.Pickling import loadPickle


    ###

    parser = argparse.ArgumentParser(description="Run REBOUND simulation for a given trajectory pickle file.")

    parser.add_argument("pickle_path", type=str, help="Path to the pickle file with the trajectory data.")

    args = parser.parse_args()

    ###

    # Load the trajectory data from a pickle file
    traj = loadPickle(*os.path.split(args.pickle_path))


    # import pandas as pd
    # import ast

    # # Read in the test objects
    # df = pd.read_csv("test_objects.csv", header=0)


    # for i in range(len(df)):
    #     reboundSimulate(df["julian_date"].values[i], np.array(ast.literal_eval(df["state_vect"].values[i])), df["obj_name"].values[i])