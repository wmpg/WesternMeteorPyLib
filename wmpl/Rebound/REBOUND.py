""" Functions for running REBOUND simulations on wmpl trajectories. """

import numpy as np
import scipy
import matplotlib.pyplot as plt

try:
    import rebound as rb
    import reboundx
    from reboundx import constants as rbxConstants

    REBOUND_FOUND = True

except ImportError:
    print("REBOUND package not found. Install REBOUND and reboundx packages to use the REBOUND functions.")
    REBOUND_FOUND = False

import astropy.time

from wmpl.Utils.TrajConversions import (
    J2000_JD,
    equatorialCoordPrecession,
    jd2DynamicalTimeJD,
    eci2RaDec,
    vectMag,
    raDec2ECI
)
from wmpl.Utils.Earth import calcTrueObliquity


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

    # Skip if REBOUND is not found
    if not REBOUND_FOUND:
        print("REBOUND package not found. Install REBOUND and reboundx packages to use the REBOUND functions.")
        return None

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
        verbose=False):
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
        verbose: [bool] If True, print out the progress of the simulation.

    Return:
        outputs: [list] List of outputs, each containing the time, state vector and orbital elements at that 
            time.

    """

    # Skip if REBOUND is not found
    if not REBOUND_FOUND:
        print("REBOUND package not found. Install REBOUND and reboundx packages to use the REBOUND functions.")
        return None

    # If the trajectory is given, override the julian_date and state_vect arguments
    if traj is not None:

        # Extract the state vector from the trajectory
        x, y, z = traj.state_vect_mini
        vx, vy, vz = traj.v_init*traj.radiant_eci_mini
        state_vect = [x, y, z, vx, vy, vz]

        # Extract the Julian date from the trajectory
        julian_date = traj.jdt_ref

    # If the state vector is not given, raise an error
    if state_vect is None:
        raise ValueError("The state_vect argument must be given if the traj argument is not given.")
    
    # If the Julian date is not given, raise an error
    if julian_date is None:
        raise ValueError("The julian_date argument must be given if the traj argument is not given.")

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
    time_utc = astropy.time.Time(julian_date, format='jd', scale='utc')
    time_tdb = time_utc.tdb.jd

    # TEST - compare to the already computed dynamical time
    print("astropy dynamical time:", time_tdb)
    print("computed dynamical time:", jd2DynamicalTimeJD(julian_date))

    # Set up the number of outputs and the time array
    if direction == "forward":
        sim.dt = 0.001
        times = np.linspace(0, year*tsimend, n_outputs)

    else:
        sim.dt = -0.001
        times = np.linspace(0, -year*tsimend, n_outputs)

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

    # To start the meteoroid in the right spot, we need to feed in x,y,z barycentric position in AU
    # The velocity should be in AU/year divided by 2pi
    state_vect_rot = convertToBarycentric(state_vect, time_tdb)

    if verbose:
        print("Initial state vector in ECI coordinates:")
        print("[ x,  y,  z] = ", state_vect[:3])
        print("[vx, vy, vz] = ", state_vect[3:])

        print("Initial state vector in barycentric coordinates:")
        print("[ x,  y,  z] = ", state_vect_rot[:3])
        print("[vx, vy, vz] = ", state_vect_rot[3:])

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

    ps = sim.particles

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

    outputs = []

    # Integrate the simulation and save the state vectors and orbital elements in the output list
    # These are not time steps, but the times at which the simulation state is saved
    for i, time in enumerate(times):

        sim.move_to_com()
        sim.integrate(time)
        sim.move_to_hel()

        # If the particle is not in the simulation anymore, break the loop
        if obj_name not in ps:
            break

        # Extract the heliocentric state vector
        state_vect_hel = [ps[obj_name].x, ps[obj_name].y, ps[obj_name].z, ps[obj_name].vx, ps[obj_name].vy, ps[obj_name].vz]

        # Extract the orbital elements
        orb_elem = ps[obj_name].orbit(primary=ps["Sun"])

        if verbose and (i%25 == 0):
            print(f"loop iter = {i}, {time}, a = {orb_elem.a}, e = {orb_elem.e}, inc = {orb_elem.inc}, Omega = {orb_elem.Omega}, omega = {orb_elem.omega}, f = {orb_elem.f}")

        outputs.append([time, state_vect_hel, orb_elem])

    return outputs


if __name__ == "__main__":

    import os
    import argparse

    from wmpl.Utils.Pickling import loadPickle


    ###

    parser = argparse.ArgumentParser(description="Run REBOUND simulation for a given trajectory pickle file.")

    parser.add_argument("pickle_path", type=str, help="Path to the pickle file with the trajectory data.")

    parser.add_argument("--verbose", action="store_true", help="Print out the progress of the simulation.")

    args = parser.parse_args()

    ###

    # Load the trajectory data from a pickle file
    traj = loadPickle(*os.path.split(args.pickle_path))

    
    # Run the simulation -60 and +60 days from the epoch of the trajectory
    sims_60back = reboundSimulate(None, None, traj=traj, direction="backward", tsimend=60, 
                                  obj_name=traj.traj_id, verbose=args.verbose)
    sims_60fwrd = reboundSimulate(None, None, traj=traj, direction="forward",  tsimend=60, 
                                  obj_name=traj.traj_id, verbose=args.verbose)


    # Print the -60 and +60 days simulation orbital elements
    print("Orbital elements -60 days:")
    print("a = ", sims_60back[-1][2].a)
    print("e = ", sims_60back[-1][2].e)
    print("inc = ", sims_60back[-1][2].inc)
    print("Omega = ", sims_60back[-1][2].Omega)
    print("omega = ", sims_60back[-1][2].omega)
    print("f = ", sims_60back[-1][2].f)
    print()
    print("Orbital elements +60 days:")
    print("a = ", sims_60fwrd[-1][2].a)
    print("e = ", sims_60fwrd[-1][2].e)
    print("inc = ", sims_60fwrd[-1][2].inc)
    print("Omega = ", sims_60fwrd[-1][2].Omega)
    print("omega = ", sims_60fwrd[-1][2].omega)
    print("f = ", sims_60fwrd[-1][2].f)

    # Plot the orbital elements of the before and after simulation on the same plot (one subplot for each element)
    fig, axs = plt.subplots(3, 2, figsize=(12, 8))

    for out_arr, name in [[sims_60back, "Backwards"], [sims_60fwrd, "Forwards"]]:
        a = [x[2].a for x in out_arr]
        e = [x[2].e for x in out_arr]
        i = [x[2].inc for x in out_arr]
        Omega = [x[2].Omega for x in out_arr]
        omega = [x[2].omega for x in out_arr]
        f = [x[2].f for x in out_arr]

        axs[0, 0].plot(a, label=name)
        axs[0, 1].plot(e, label=name)
        axs[1, 0].plot(i, label=name)
        axs[1, 1].plot(Omega, label=name)
        axs[2, 0].plot(omega, label=name)
        axs[2, 1].plot(f, label=name)

    axs[0, 0].set_title("Semi-major axis (a)")
    axs[0, 1].set_title("Eccentricity (e)")
    axs[1, 0].set_title("Inclination (i)")
    axs[1, 1].set_title("Longitude of the ascending node (Omega)")
    axs[2, 0].set_title("Argument of pericenter (omega)")
    axs[2, 1].set_title("True anomaly (f)")

    for ax in axs.flatten():
        ax.legend()

    plt.tight_layout()
    plt.show()