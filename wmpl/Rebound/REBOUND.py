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


def state_vec_convert(state_vec, jd, log_file_path=""):
    """
    Takes a state vector, date and julian date and converts from ECI to barycentric, changes units to
    AU for lengths and AU/code year for velocities, and logs output to a file if log_file_path name given.
    """
    if len(log_file_path):
        log_file = open(log_file_path, "w")
        
        def log(message):
            log_file.write(message + "\n")
        log(f"Reference Julian date (TDB): {jd}")

    sim = rb.Simulation()
    # Uses JPL Horizons to query for J2000 ecliptic SSB, geometric states vector correction
    sim.add("Geocenter", date=f"JD{jd:.6f}", hash="Earth")
    ps = sim.particles
    aum = rb.units.lengths_SI["au"]
    eci_pos = np.array(state_vec[:3])
    eci_vel = np.array(state_vec[3:])

    if len(log_file_path):
        log(f"Position vector in equatorial ECI, epoch of date (m): {eci_pos}")
        log(f"Velocity vector in equatorial ECI, epoch of date (m/s): {eci_vel}")

    # Convert rectangular to spherical coordinates
    re = vectMag(eci_pos)
    alpha_e, delta_e = eci2RaDec(eci_pos)
    
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
            np.array([1, 0, 0]) / np.linalg.norm(np.array([1, 0, 0])) * eps,
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

    state_vec_rot = np.concatenate((pos_vec_rot, vel_vec_rot))
    state_vec_rot = [x / aum for x in state_vec_rot]
    state_vec_rot[3] *= 60 * 60 * 24 * 365.25 / (2 * np.pi)
    state_vec_rot[4] *= 60 * 60 * 24 * 365.25 / (2 * np.pi)
    state_vec_rot[5] *= 60 * 60 * 24 * 365.25 / (2 * np.pi)

    if len(log_file_path):
        log(
            f"Meteor state vector in ecliptic ECI, J2000 (AU, AU/year where year = 2pi): {state_vec_rot}"
        )
        log(
            f"Earth state vector in ecliptic solar system barycentric, J2000 (AU, AU/year where year = 2pi): {[ps['Earth'].x, ps['Earth'].y, ps['Earth'].z, ps['Earth'].vx, ps['Earth'].vy, ps['Earth'].vz]}"
        )

    state_vec_rot[0] += ps["Earth"].x
    state_vec_rot[1] += ps["Earth"].y
    state_vec_rot[2] += ps["Earth"].z
    state_vec_rot[3] += ps["Earth"].vx
    state_vec_rot[4] += ps["Earth"].vy
    state_vec_rot[5] += ps["Earth"].vz

    if len(log_file_path):
        log(
            f"Meteor state vector in barycentric, J2000 (AU, AU/year where year = 2pi): {state_vec_rot}"
        )
        log_file.close()

    return state_vec_rot


def rb_sim(julian_date, state_vec, direction="forward", tsimend=60, obj_name="obj", obj_mass=0.0, state_vec_log=""):
    """
    Takes an object and its parameters, runs rebound and returns a csv with the orbital elements over time period.
    Inputs:
        julian_date: UTC julian date (float)
        state_vec: array with the position and velocity vector components in m and m/s eg. [x, y, z, vx, vy, vz] (np.array)
        direction: default is forward time but if you want reversed time, you can specify direction="backward"
        tsimend: length of integration in days, default is 60 days
        obj_name: name of the object that's being integrated, default is just "obj"
        obj_mass: mass of object in solar masses if asteroid or larger object, default is 0.0
        state_vec_log: name of the log file if you want one generated for the state vector conversion to rebound units
    """
    sim = rb.Simulation()
    aum = rb.units.lengths_SI["au"]  # 1 au in m
    aukm = aum / 1e3  # au in km
    # used for Earth's J2 component
    RE_eq = 6378.135 / aukm
    J2 = 1.0826157e-3
    J4 = -1.620e-6
    dmin = 4.326e-5  # Earth radius in au
    tsimend = tsimend / 365.25  # simulation endtime in years
    Noutputs = 500
    year = 2.0 * np.pi  # One year in units where G=1
    # Convert from UTC to TDB
    time_utc = Time(julian_date, format='jd', scale='utc')
    time_tdb = time_utc.tdb.jd

    if direction == "forward":
        sim.dt = 0.001
        times = np.linspace(0, year * tsimend, Noutputs)
    else:
        sim.dt = -0.001
        times = np.linspace(0, -year * tsimend, Noutputs)

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
    state_vec_rot = state_vec_convert(state_vec, time_tdb, state_vec_log)

    sim.add(
        m=obj_mass,
        x=state_vec_rot[0],
        y=state_vec_rot[1],
        z=state_vec_rot[2],
        vx=state_vec_rot[3],
        vy=state_vec_rot[4],
        vz=state_vec_rot[5],
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
    ps["Luna"].r = dmin / 4  # set size of Moon
    # gr is the general relativity correction for just the Sun, gr_full is the correction for all bodies in the sim
    gr = rebx.load_force("gr_full")
    rebx.add_force(gr)
    gr.params["c"] = rbxConstants.C

    sim.move_to_com()  # We always move to the center of momentum frame before an integration

    # Specify how simulation should resolve collision
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

# To test the above code:
# import pandas as pd
# import ast
# df = pd.read_csv("test_objects.csv", header=0)
# for i in range(len(df)):
#     rb_sim(df["julian_date"].values[i], np.array(ast.literal_eval(df["state_vec"].values[i])), df["obj_name"].values[i])