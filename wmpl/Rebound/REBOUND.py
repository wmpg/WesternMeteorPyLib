""" Functions for running REBOUND simulations on wmpl trajectories. """

import numpy as np
import scipy
import scipy.stats
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



def extractSimParams(ps, obj_name):
    """ Extracts the state vector, orbital elements and distance from the Earth from a REBOUND simulation.

    Arguments:
        ps: [REBOUND simulation] REBOUND simulation object.
        obj_name: [str] Name of the object in the simulation.

    Return:
        state_vect_hel: [list] Heliocentric state vector of the object in the simulation, [x, y, z, vx, vy, vz].
        orb_elem: [REBOUND orbit] Orbital elements of the object in the simulation.
        earth_dist: [float] Distance between the object and the Earth in AU.

    """

    # Extract the heliocentric state vector
    state_vect_pos = [ps[obj_name].x, ps[obj_name].y, ps[obj_name].z]
    state_vect_vel = [ps[obj_name].vx, ps[obj_name].vy, ps[obj_name].vz]
    state_vect_hel = state_vect_pos + state_vect_vel

    # Get the coordinates of the Earth
    earth_coords = [ps["Earth"].x, ps["Earth"].y, ps["Earth"].z]

    # Compute the distance between the meteoroid and the Earth
    earth_dist = np.linalg.norm(np.array(state_vect_pos) - np.array(earth_coords))

    # Extract the orbital elements
    orb_elem = ps[obj_name].orbit(primary=ps["Sun"])

    return state_vect_hel, orb_elem, earth_dist


def reboundSimulate(
        julian_date, state_vect, traj=None, 
        direction="forward", tsimend=60, n_outputs=500, obj_name="obj", obj_mass=0.0, mc_runs=100,
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
        mc_runs: [int] Number of Monte Carlo simulations to run, default is 100.
        verbose: [bool] If True, print out the progress of the simulation.

    Return:
        outputs: [list] List of outputs, each containing the time, state vector and orbital elements at that 
            time.

    """

    ### HOTFIX: Solve the cert issue
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    ###

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
    


    # If the number of Monte Carlo simulations is given and the trajectory has uncertainties defined,
    # sample the state vector from the uncertainties
    state_vect_realizations = []
    if (mc_runs > 1) and (traj is not None) and (traj.uncertainties is not None):

        # Extract the state vector covariance matrix
        cov = traj.state_vect_cov

        # Sample the state vector from the uncertainties
        for i in range(mc_runs):

            # Sample the state vector from the uncertainties
            sv_realization = np.random.multivariate_normal(state_vect, cov)

            state_vect_realizations.append(sv_realization)

            if verbose:
                print(f"MC realization {i}: {sv_realization}")

        


    # Set up the simulation
    sim = rb.Simulation()
    rebx = reboundx.Extras(sim)
    
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


    # Add the Monte Carlo realizations to the simulation
    mc_realization_names = []
    for i, state_vect_realization in enumerate(state_vect_realizations):

        sv_mc_rot = convertToBarycentric(state_vect_realization, time_tdb)

        mc_name = f"{obj_name}_MC_{i}"

        sim.add(
            m=obj_mass,
            x=sv_mc_rot[0],
            y=sv_mc_rot[1],
            z=sv_mc_rot[2],
            vx=sv_mc_rot[3],
            vy=sv_mc_rot[4],
            vz=sv_mc_rot[5],
            hash=mc_name,
        )

        mc_realization_names.append(mc_name)


    ps = sim.particles

    # Add gravitational harmonics of Earth
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

    # # Specify how simulation should resolve collisions
    # sim.collision = "direct"
    # sim.collision_resolve = "merge"
    # sim.collision_resolve_keep_sorted = 1
    # sim.track_energy_offset = 1
    
    # Disable collision detection
    sim.collision = "none"

    outputs = []
    outputs_mc = {}

    # Integrate the simulation and save the state vectors and orbital elements in the output list
    # These are not time steps, but the times at which the simulation state is saved
    for i, time in enumerate(times):

        sim.move_to_com()
        sim.integrate(time)
        sim.move_to_hel()

        # If the particle is not in the simulation anymore, break the loop
        try:
            ps[obj_name]
        except rb.ParticleNotFound:
            break

        # Extract the state vector and the orbital elements
        state_vect_hel, orb_elem, earth_dist = extractSimParams(ps, obj_name)

        if verbose and (i%25 == 0):
            print(f"{i}: t = {time:.6f} d, a = {orb_elem.a:10.6f}, e = {orb_elem.e:10.6f}, inc = {orb_elem.inc:10.6f}, Omega = {orb_elem.Omega:10.6f}, omega = {orb_elem.omega:10.6f}, f = {orb_elem.f:10.6f}")

        outputs.append([time, state_vect_hel, orb_elem, earth_dist])


        # Extract the state vector and the orbital elements for the Monte Carlo realizations
        for mc_name in mc_realization_names:

            try:
                ps[mc_name]
            except rb.ParticleNotFound:
                continue

            state_vect_hel, orb_elem, earth_dist = extractSimParams(ps, mc_name)

            if mc_name not in outputs_mc:
                outputs_mc[mc_name] = []

            outputs_mc[mc_name].append([time, state_vect_hel, orb_elem, earth_dist])


    return outputs, outputs_mc


if __name__ == "__main__":

    import os
    import argparse

    from wmpl.Utils.Pickling import loadPickle


    ###

    parser = argparse.ArgumentParser(description="Run REBOUND simulation for a given trajectory pickle file. The simulation is run 60 days backwards by default.")

    parser.add_argument("pickle_path", type=str, help="Path to the pickle file with the trajectory data.")

    parser.add_argument("--days", type=float, help="Run the simulation for the given number of days.", default=60)

    parser.add_argument("--forward", type=str, help="Run the simulation forward for the given number of days.")

    parser.add_argument("--mc", type=int, help="Run the simulation for the given number of Monte Carlo simulations."
                        "The default is 0", default=0)
    
    parser.add_argument("--verbose", action="store_true", help="Print out the progress of the simulation.")

    args = parser.parse_args()

    # Extract the number of days from the arguments and the simulation direction
    sim_days = args.days
    direction = "backward" if args.forward is None else "forward"

    ###

    # Load the trajectory data from a pickle file
    traj = loadPickle(*os.path.split(args.pickle_path))

    
    # Run the simulation -60 and +60 days from the epoch of the trajectory
    sim_outputs, sim_outputs_mc = reboundSimulate(
        None, None, traj=traj, direction=direction, tsimend=sim_days,
        obj_name=traj.traj_id, mc_runs=args.mc, verbose=args.verbose
        )

    # Compute the 95% CI for the orbital elements from the Monte Carlo realizations
    a_ci_str = ""
    e_ci_str = ""
    incl_ci_str = ""
    Omega_ci_str = ""
    omega_ci_str = ""
    f_ci_str = ""
    if len(sim_outputs_mc):

        # Extract the orbital elements at the end for each MC realization
        a_mc = []
        e_mc = []
        incl_mc = []
        Omega_mc = []
        omega_mc = []
        f_mc = []

        for mc_name in sim_outputs_mc:
            a_mc.append(sim_outputs_mc[mc_name][-1][2].a)
            e_mc.append(sim_outputs_mc[mc_name][-1][2].e)
            incl_mc.append(sim_outputs_mc[mc_name][-1][2].inc)
            Omega_mc.append(sim_outputs_mc[mc_name][-1][2].Omega)
            omega_mc.append(sim_outputs_mc[mc_name][-1][2].omega)
            f_mc.append(sim_outputs_mc[mc_name][-1][2].f)
        

        a_95ci_low = np.percentile(a_mc, 2.5)
        a_95ci_high = np.percentile(a_mc, 97.5)
        e_95ci_low = np.percentile(e_mc, 2.5)
        e_95ci_high = np.percentile(e_mc, 97.5)
        incl_95ci_low = np.percentile(incl_mc, 2.5)
        incl_95ci_high = np.percentile(incl_mc, 97.5)
        Omega_95ci_low = np.percentile(Omega_mc, 2.5)
        Omega_95ci_high = np.percentile(Omega_mc, 97.5)
        omega_95ci_low = np.percentile(omega_mc, 2.5)
        omega_95ci_high = np.percentile(omega_mc, 97.5)
        f_95ci_low = np.percentile(f_mc, 2.5)
        f_95ci_high = np.percentile(f_mc, 97.5)

        # Compute the standard deviation of the orbital elements
        a_std = np.std(a_mc)
        e_std = np.std(e_mc)
        incl_std = np.std(incl_mc)
        Omega_std = scipy.stats.circstd(Omega_mc)
        omega_std = scipy.stats.circstd(omega_mc)
        f_std = scipy.stats.circstd(f_mc)


        a_ci_str = f" +/- {np.degrees(a_std):.6f} [{a_95ci_low:10.6f}, {a_95ci_high:10.6f}]"
        e_ci_str = f" +/- {np.degrees(e_std):.6f} [{e_95ci_low:10.6f}, {e_95ci_high:10.6f}]"
        incl_ci_str = f" +/- {np.degrees(incl_std):.6f} [{np.degrees(incl_95ci_low):10.6f}, {np.degrees(incl_95ci_high):10.6f}]"
        Omega_ci_str = f" +/- {np.degrees(Omega_std):.6f} [{np.degrees(Omega_95ci_low):10.6f}, {np.degrees(Omega_95ci_high):10.6f}]"
        omega_ci_str = f" +/- {np.degrees(omega_std):.6f} [{np.degrees(omega_95ci_low):10.6f}, {np.degrees(omega_95ci_high):10.6f}]"
        f_ci_str = f" +/- {np.degrees(f_std):.6f} [{np.degrees(f_95ci_low):10.6f}, {np.degrees(f_95ci_high):10.6f}]"





    # Print the -60 and +60 days simulation orbital elements
    print("Orbital elements {:.2f} days from the epoch {:.6f} {:s}".format(sim_days, traj.jdt_ref, direction))
    print("a    = {:>10.6}{:s} AU".format(sim_outputs[-1][2].a, a_ci_str))
    print("e    = {:>10.6}{:s}".format(sim_outputs[-1][2].e, e_ci_str))
    print("i    = {:>10.6}{:s} deg".format(np.degrees(sim_outputs[-1][2].inc), incl_ci_str))
    print("peri = {:>10.6}{:s} deg".format(np.degrees(sim_outputs[-1][2].Omega), Omega_ci_str))
    print("node = {:>10.6}{:s} deg".format(np.degrees(sim_outputs[-1][2].omega), omega_ci_str))
    print("f    = {:>10.6}{:s} deg".format(np.degrees(sim_outputs[-1][2].f), f_ci_str))

    # Plot the orbital elements of the before and after simulation on the same plot (one subplot for each element)
    fig, axs = plt.subplots(3, 3, figsize=(12, 8), sharex=True)

    # Time in days
    t = [x[0] for x in sim_outputs]

    a = [x[2].a for x in sim_outputs]
    e = [x[2].e for x in sim_outputs]
    incl = [x[2].inc for x in sim_outputs]
    Omega = [x[2].Omega for x in sim_outputs]
    omega = [x[2].omega for x in sim_outputs]
    f = [x[2].f for x in sim_outputs]

    # Distance from the Earth
    earth_dist = [x[3] for x in sim_outputs]

    # Find the time when the object exits the Earth's Hill sphere
    earth_hill = 0.01 # AU
    exit_index = None
    for i, dist in enumerate(earth_dist):
        if dist > earth_hill:
            exit_index = i
            break

    axs[0, 0].plot(t, a)
    axs[0, 1].plot(t, e)
    axs[1, 0].plot(t, np.degrees(incl))
    axs[1, 1].plot(t, np.degrees(Omega))
    axs[2, 0].plot(t, np.degrees(omega))
    axs[2, 1].plot(t, np.degrees(f))

    # Create filtered arrays for the exit from the Earth's Hill sphere
    if exit_index is not None:
        t_exit = t[exit_index:]
        a_exit = a[exit_index:]
        e_exit = e[exit_index:]
        incl_exit = incl[exit_index:]
        Omega_exit = Omega[exit_index:]
        omega_exit = omega[exit_index:]
        f_exit = f[exit_index:]

        # Adjust Y axis limits for the exit from the Earth
        axs[0, 0].set_ylim(ymin=min(a_exit), ymax=max(a_exit))
        axs[0, 1].set_ylim(ymin=min(e_exit), ymax=max(e_exit))
        axs[1, 0].set_ylim(ymin=min(np.degrees(incl_exit)), ymax=max(np.degrees(incl_exit)))
        axs[1, 1].set_ylim(ymin=min(np.degrees(Omega_exit)), ymax=max(np.degrees(Omega_exit)))
        axs[2, 0].set_ylim(ymin=min(np.degrees(omega_exit)), ymax=max(np.degrees(omega_exit)))
        axs[2, 1].set_ylim(ymin=min(np.degrees(f_exit)), ymax=max(np.degrees(f_exit)))
                           


    # Plot the distance from the Earth
    axs[0, 2].plot(t, earth_dist)

    # Mark the exit from the Earth's Hill sphere
    if exit_index is not None:
        axs[0, 2].axvline(t[exit_index], color="red", linestyle="--", label="Exit from Earth's Hill sphere")

        # Set the X axis limit so the maximum time is at the exit from the Earth's Hill sphere
        axs[0, 2].set_xlim(xmax=t[exit_index])

    # Set the axis labels
    for ax in axs.flatten():
        ax.set_xlabel("Time [days]")


    # Plot the MC realizations (all in thin alpha=0.5 lines)
    for mc_name in sim_outputs_mc:

        a_mc = [x[2].a for x in sim_outputs_mc[mc_name]]
        e_mc = [x[2].e for x in sim_outputs_mc[mc_name]]
        incl_mc = [x[2].inc for x in sim_outputs_mc[mc_name]]
        Omega_mc = [x[2].Omega for x in sim_outputs_mc[mc_name]]
        omega_mc = [x[2].omega for x in sim_outputs_mc[mc_name]]
        f_mc = [x[2].f for x in sim_outputs_mc[mc_name]]
        earth_dist = [x[3] for x in sim_outputs_mc[mc_name]]

        axs[0, 0].plot(t, a_mc, alpha=0.5, color='k', lw=0.5)
        axs[0, 1].plot(t, e_mc, alpha=0.5, color='k', lw=0.5)
        axs[1, 0].plot(t, np.degrees(incl_mc), alpha=0.5, color='k', lw=0.5)
        axs[1, 1].plot(t, np.degrees(Omega_mc), alpha=0.5, color='k', lw=0.5)
        axs[2, 0].plot(t, np.degrees(omega_mc), alpha=0.5, color='k', lw=0.5)
        axs[2, 1].plot(t, np.degrees(f_mc), alpha=0.5, color='k', lw=0.5)
        axs[0, 2].plot(t, earth_dist, alpha=0.5, color='k', lw=0.5)

    
    
    axs[0, 0].set_ylabel("a [AU]")
    axs[0, 1].set_ylabel("e")
    axs[1, 0].set_ylabel("i [deg]")
    axs[1, 1].set_ylabel("peri [deg]")
    axs[2, 0].set_ylabel("node [deg]")
    axs[2, 1].set_ylabel("f [deg]")
    axs[0, 2].set_ylabel("Earth distance [AU]")

    for ax in axs.flatten():
        ax.legend()

    plt.tight_layout()
    plt.show()