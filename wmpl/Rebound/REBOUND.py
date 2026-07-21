""" Functions for running REBOUND simulations on wmpl trajectories. """

import os
import re
import sys
import time
import warnings
import concurrent.futures
from types import SimpleNamespace

import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt

from jplephem.spk import SPK

try:
    # Silence the noisy "pkg_resources is deprecated" warning emitted while importing reboundx
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
        import rebound as rb
        import reboundx
        from reboundx import constants as rbxConstants
    import astropy.time

    REBOUND_FOUND = True

except ImportError:
    print("REBOUND package not found. Install REBOUND and reboundx packages to use the REBOUND functions.")
    REBOUND_FOUND = False

from wmpl.Config import config
from wmpl.Utils.TrajConversions import (
    J2000_JD,
    J2000_OBLIQUITY,
    equatorialCoordPrecession,
    jd2DynamicalTimeJD,
    eci2RaDec,
    vectMag,
    raDec2ECI
)
from wmpl.Utils.Math import rotateVector
from wmpl.Utils.Earth import calcTrueObliquity


# Hill-sphere radii in AU used for close-encounter detection. The Moon (Luna) uses its
# Earth-relative Hill radius. The Sun is excluded (it has no Hill sphere in this context).
HILL_RADII_AU = {
    "Mercury": 0.00122,
    "Venus":   0.00673,
    "Earth":   0.00981,
    "Luna":    0.000411,
    "Mars":    0.00658,
    "Jupiter": 0.33820,
    "Saturn":  0.42850,
    "Uranus":  0.46340,
    "Neptune": 0.76890,
}


# NAIF-ID segment paths (center, target) used to build each body's state relative to the Solar
# System barycentre (0) from the local DE430 kernel. States along a path are summed. These are
# chosen to reproduce exactly the bodies that REBOUND's JPL Horizons queries return: planet
# centres for Mercury (199) and Venus (299), Earth (399) and Moon (301) w.r.t. the Earth-Moon
# barycentre (3), and system barycentres for Mars-Neptune (4-8).
_EPHEM_NAIF_PATHS = {
    "Sun":     [(0, 10)],
    "Mercury": [(0, 1), (1, 199)],
    "Venus":   [(0, 2), (2, 299)],
    "Earth":   [(0, 3), (3, 399)],
    "Luna":    [(0, 3), (3, 301)],
    "Mars":    [(0, 4)],
    "Jupiter": [(0, 5)],
    "Saturn":  [(0, 6)],
    "Uranus":  [(0, 7)],
    "Neptune": [(0, 8)],
}

# NAIF ID whose GM (in REBOUND's embedded Horizons mass table) gives each body's mass. These
# match the bodies resolved by REBOUND's Horizons queries so the masses are identical to the
# web path.
_EPHEM_MASS_NAIF = {
    "Sun": 10, "Mercury": 199, "Venus": 299, "Earth": 399, "Luna": 301,
    "Mars": 4, "Jupiter": 5, "Saturn": 6, "Uranus": 7, "Neptune": 8,
}


def reboundBodyMassSolar(naif_id):
    """ Return a body's mass in solar masses, using the same GM table REBOUND uses for its JPL
    Horizons queries (rebound.horizons.HORIZONS_MASS_DATA).

    The mass is computed as the ratio GM_body/GM_Sun, which equals the mass in solar masses and
    is exactly the value REBOUND assigns to bodies added via Horizons (up to the Sun's own mass
    being 1.0 in the G=1, AU, solar-mass, year/2pi unit system used by the simulation).

    Arguments:
        naif_id: [int] NAIF ID of the body (e.g. 5 for the Jupiter system barycentre).

    Return:
        [float] Mass of the body in solar masses.
    """

    import rebound.horizons as rbh

    def _gm(idn):
        match = re.search(
            r"BODY{:d}\_GM .* \( *([\.DE\+\-0-9]+ *)\)".format(int(idn)), rbh.HORIZONS_MASS_DATA
        )
        return float(match.group(1).replace("D+", "E+"))

    return _gm(naif_id)/_gm(10)


def ephemBodyStateRebound(body, jd_tdb, jpl_ephem_data):
    """ Compute a body's state vector and mass in REBOUND simulation units from the local DE430
    ephemeris, reproducing the state that REBOUND's JPL Horizons query would add.

    The returned state is barycentric (Solar System barycentre origin) in the ecliptic J2000
    frame, with positions in AU and velocities in AU/(year/2pi) - i.e. the units of a default
    REBOUND simulation (G=1, length AU, mass solar masses, time year/2pi).

    Arguments:
        body: [str] Body name, a key of _EPHEM_NAIF_PATHS (e.g. "Jupiter", "Earth", "Luna").
        jd_tdb: [float] Julian date in Barycentric Dynamical Time (TDB).
        jpl_ephem_data: [SPK] An opened jplephem SPK kernel (SPK.open(config.jpl_ephem_file)).

    Return:
        state: [list] [x, y, z, vx, vy, vz] in AU and AU/(year/2pi).
        mass: [float] Body mass in solar masses.
    """

    aum = rb.units.lengths_SI["au"]  # 1 au in m
    aukm = aum/1e3                   # 1 au in km

    # Sum the segment states along the path to get the body's state relative to the SSB.
    # compute_and_differentiate returns position in km and velocity in km/day, in the
    # equatorial ICRF/J2000 frame.
    pos_km = np.zeros(3)
    vel_kmday = np.zeros(3)
    for center, target in _EPHEM_NAIF_PATHS[body]:
        position, velocity = jpl_ephem_data[center, target].compute_and_differentiate(jd_tdb)
        pos_km += np.array(position)
        vel_kmday += np.array(velocity)

    # Rotate from the equatorial J2000 frame to the ecliptic J2000 frame (same convention as
    # wmpl.Utils.Earth.calcEarthRectangularCoordJPL)
    pos_km = rotateVector(pos_km, np.array([1.0, 0.0, 0.0]), -J2000_OBLIQUITY)
    vel_kmday = rotateVector(vel_kmday, np.array([1.0, 0.0, 0.0]), -J2000_OBLIQUITY)

    # Convert to REBOUND units: km -> AU, km/day -> AU/(year/2pi)
    pos_au = pos_km/aukm
    vel_reb = vel_kmday/aukm*(365.25/(2*np.pi))

    state = list(pos_au) + list(vel_reb)
    mass = reboundBodyMassSolar(_EPHEM_MASS_NAIF[body])

    return state, mass


def findEarthDepartureIndex(sim_outputs, n_hill=3.0):
    """ Find the first timestep at which the object leaves the Earth's Hill-sphere neighborhood.

    Meteoroid orbits all start at the Earth, so the object begins inside the Earth's Hill sphere.
    This returns the index of the first output at which the object-Earth distance first exceeds
    n_hill times the Earth's Hill radius, i.e. when the object has departed the Earth's
    neighborhood. Encounter detection for all bodies except the Moon is restricted to timesteps
    at or after this index (see detectCloseEncounters).

    Arguments:
        sim_outputs: [list] Per-timestep outputs as returned by reboundSimulate.

    Keyword arguments:
        n_hill: [float] Multiple of the Earth's Hill radius that defines the departure boundary.
            Default is 3.0 (matches the close-encounter threshold).

    Return:
        [int or None] Index of the first output outside n_hill Earth Hill radii, or None if the
            object never leaves that neighborhood during the simulation.
    """

    earth_hill = HILL_RADII_AU["Earth"]

    for i, output in enumerate(sim_outputs):

        try:
            earth_dist = output[3]["Earth"]
        except (KeyError, IndexError, TypeError):
            return None

        if earth_dist > n_hill*earth_hill:
            return i

    return None


def detectCloseEncounters(sim_outputs, n_hill=3.0):
    """ Detect close encounters between the integrated object and the planets (and the Moon).

    A close encounter is flagged when the minimum object-body distance drops below n_hill times
    the body's Hill-sphere radius (see HILL_RADII_AU). The Hill sphere is the standard criterion
    for gravitational close encounters; a small multiple (~3 R_Hill) is commonly used as the
    boundary of significant perturbation.

    Because meteoroid orbits all start at the Earth, the initial period during which the object
    is still inside the Earth's Hill-sphere neighborhood is excluded from detection for every
    body EXCEPT the Moon: detection for all other bodies (including a genuine later Earth
    re-encounter) starts only once the object has left the Earth's neighborhood
    (see findEarthDepartureIndex). The Moon is always checked over the full simulation, so a real
    lunar encounter while the object is departing the Earth is still caught. If the object never
    leaves the Earth's neighborhood, only the Moon is checked.

    Arguments:
        sim_outputs: [list] List of per-timestep outputs, each [time, state_vect_hel, orb_elem,
            planet_dists], as returned by reboundSimulate. planet_dists is a dict mapping body
            name to object-body distance in AU.

    Keyword arguments:
        n_hill: [float] Multiple of the Hill radius used as the close-encounter threshold and as
            the Earth-departure boundary. Default is 3.0.

    Return:
        [list] One dict per detected encounter, sorted by closeness (min_dist/R_Hill ascending):
            {
                "body":           [str]   body name,
                "min_dist_au":    [float] minimum distance during the searched interval in AU,
                "time_days":      [float] time of closest approach in days from the epoch,
                "hill_radius_au": [float] body's Hill radius in AU,
                "n_hill":         [float] min_dist_au/hill_radius_au (closeness in Hill radii),
                "index":          [int]   index into sim_outputs of the closest approach,
            }
    """

    encounters = []

    if not sim_outputs:
        return encounters

    # Index at which the object leaves the Earth's neighborhood. Detection for all bodies except
    # the Moon starts here to avoid flagging the trivial initial encounter with the Earth.
    earth_departure_index = findEarthDepartureIndex(sim_outputs, n_hill=n_hill)

    for body, hill_radius in HILL_RADII_AU.items():

        # Extract the distance series for this body (skip if the body is not tracked)
        try:
            dists = [output[3][body] for output in sim_outputs]
        except (KeyError, IndexError, TypeError):
            continue

        # The Moon is checked over the full simulation; all other bodies only after the object
        # has departed the Earth's neighborhood.
        if body == "Luna":
            start = 0
        else:
            # If the object never leaves the Earth's neighborhood, skip all non-Moon bodies
            if earth_departure_index is None:
                continue
            start = earth_departure_index

        # Find the closest approach within the searched interval
        search = dists[start:]
        if not search:
            continue

        min_index = start + int(np.argmin(search))
        min_dist = dists[min_index]

        # Flag an encounter if the closest approach is within n_hill Hill radii
        if min_dist < n_hill*hill_radius:
            encounters.append({
                "body": body,
                "min_dist_au": min_dist,
                "time_days": sim_outputs[min_index][0]/(2*np.pi)*365.25,
                "hill_radius_au": hill_radius,
                "n_hill": min_dist/hill_radius,
                "index": min_index,
            })

    # Sort by closeness in Hill radii (closest first)
    encounters.sort(key=lambda e: e["n_hill"])

    return encounters


def convertToBarycentric(state_vect, jd, log_file_path="", ephem_source="local", jpl_ephem_data=None,
                         earth_state=None):
    """ Takes a state vector in ECI coordinates (m and m/s), Julian date and converts from ECI (geocentric) to
    Solar System barycentric coordiantes. The units are changed to AU and AU/year.

    Arguments:
        state_vect: [list] Position and velocity components in ECI coordinates (epoch of date) in m and m/s,
        [x, y, z, vx, vy, vz], as given in the WMPL trajectory solution.
        jd: [float] Julian date (decimal), in TDB.

    Keyword arguments:
        log_file_path: [str] Path to the log file where the output will be written.
        ephem_source: [str] Source of the Earth's barycentric state used to shift the meteoroid:
            - "local" (default): the local DE430 ephemeris (fast, offline).
            - "horizons": the JPL Horizons web service.
        jpl_ephem_data: [SPK] An opened jplephem SPK kernel. If None and ephem_source is "local",
            the kernel is opened from config.jpl_ephem_file.
        earth_state: [list] Precomputed Earth barycentric state [x, y, z, vx, vy, vz] in REBOUND units
            (AU, AU/(year/2pi)), ecliptic J2000, at the epoch jd. If given, it is used directly and
            ephem_source/jpl_ephem_data are ignored - this avoids re-querying the Earth's position,
            which is identical across Monte Carlo realizations sharing the same epoch.

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

    # Get the Earth's barycentric state (AU, AU/(year/2pi), ecliptic J2000) used to shift the
    # meteoroid from geocentric to barycentric coordinates. The Earth state at a given epoch is
    # identical for all Monte Carlo realizations, so it can be precomputed once and passed in.
    if earth_state is not None:

        pass

    elif ephem_source == "horizons":

        # Use JPL Horizons to query for the J2000 ecliptic SSB Earth state vector
        sim = rb.Simulation()
        sim.add("Geocenter", date=f"JD{jd:.6f}", hash="Earth")
        ps = sim.particles
        earth_state = [ps["Earth"].x, ps["Earth"].y, ps["Earth"].z,
                       ps["Earth"].vx, ps["Earth"].vy, ps["Earth"].vz]

    else:

        # Use the local DE430 ephemeris
        if jpl_ephem_data is None:
            jpl_ephem_data = SPK.open(config.jpl_ephem_file)
        earth_state, _ = ephemBodyStateRebound("Earth", jd, jpl_ephem_data)

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
            f"Earth state vector in ecliptic solar system barycentric, J2000 (AU, AU/year where year = 2pi): {earth_state}"
        )

    # Add the Earth's position and velocity to the meteoroid's position and velocity
    state_vect_rot[0] += earth_state[0]
    state_vect_rot[1] += earth_state[1]
    state_vect_rot[2] += earth_state[2]
    state_vect_rot[3] += earth_state[3]
    state_vect_rot[4] += earth_state[4]
    state_vect_rot[5] += earth_state[5]

    if len(log_file_path):
        log(
            f"Meteor state vector in barycentric, J2000 (AU, AU/year where year = 2pi): {state_vect_rot}"
        )
        log_file.close()

    return state_vect_rot



def extractSimParams(ps, obj_name, planet_names, reference_frame="heliocentric"):
    """ Extracts the state vector, orbital elements and distance from the Earth from a REBOUND simulation.

    Arguments:
        ps: [REBOUND simulation] REBOUND simulation object.
        obj_name: [str] Name of the object in the simulation.
        planet_names: [list] List of planet names in the simulation.

    Keyword arguments:
        reference_frame: [str] Reference frame to use for the state vector. Options: 
            - "heliocentric"
            - "geocentric"
            Default is "heliocentric".

    Return:
        state_vect_hel: [list] Heliocentric state vector of the object in the simulation, [x, y, z, vx, vy, vz].
        orb_elem: [REBOUND orbit] Orbital elements of the object in the simulation.
        earth_dist: [float] Distance between the object and the Earth in AU.

    """

    # Extract the heliocentric state vector
    state_vect_pos = [ps[obj_name].x, ps[obj_name].y, ps[obj_name].z]
    state_vect_vel = [ps[obj_name].vx, ps[obj_name].vy, ps[obj_name].vz]
    state_vect_hel = state_vect_pos + state_vect_vel

    planet_dists = {}
    for planet in planet_names:

        # Get the coordinates of the planet
        planet_coords = [ps[planet].x, ps[planet].y, ps[planet].z]

        # Compute the distance between the meteoroid and the planet
        planet_dists[planet] = np.linalg.norm(np.array(state_vect_pos) - np.array(planet_coords))

    ref_object = 'Sun'
    if reference_frame == "geocentric":
        ref_object = 'Earth'

    # Extract the orbital elements
    orb_elem = ps[obj_name].orbit(primary=ps[ref_object])

    return state_vect_hel, orb_elem, planet_dists


def _integrateParticles(task):
    """ Build a REBOUND simulation from precomputed planet states and integrate one or more test
    particles through it, returning their per-timestep orbital elements and distances.

    This is a module-level function (picklable) so it can be dispatched to worker processes for
    parallel Monte Carlo integration. Each call builds an independent simulation, so the adaptive
    IAS15 timestep of one particle does not affect any other. Workers receive the planets as raw
    barycentric states and masses and never query the network or the ephemeris kernel.

    Arguments:
        task: [dict] A picklable task description with the keys:
            planet_names:    [list] Massive-body names/hashes, in add order.
            planet_states:   [list] Per-planet [x, y, z, vx, vy, vz] in REBOUND units (AU,
                                 AU/(year/2pi)), barycentric ecliptic J2000.
            planet_masses:   [list] Per-planet mass in solar masses.
            particle_states: [list] Per-particle barycentric [x, y, z, vx, vy, vz] (REBOUND units).
            particle_names:  [list] Per-particle name/hash (parallel to particle_states).
            times:           [list] Output times (REBOUND time units) to integrate to.
            direction:       [str]  "forward" or "backward" (sets the timestep sign).
            reference_frame: [str]  "heliocentric" or "geocentric" (passed to extractSimParams).

    Return:
        [dict] {particle_name: [[time, state_vect_hel, orb_ns, planet_dists], ...]}, where orb_ns
            is a picklable SimpleNamespace with attributes a, e, inc, Omega, omega, f.
    """

    planet_names = task["planet_names"]
    planet_states = task["planet_states"]
    planet_masses = task["planet_masses"]
    particle_states = task["particle_states"]
    particle_names = task["particle_names"]
    times = task["times"]
    direction = task["direction"]
    reference_frame = task["reference_frame"]

    aum = rb.units.lengths_SI["au"]  # 1 au in m
    aukm = aum/1e3  # au in km

    # Constants (gravitational harmonics of Earth, Earth radius)
    RE_eq = 6378.135/aukm
    J2 = 1.0826157e-3
    J4 = -1.620e-6
    dmin = 4.326e-5  # Earth radius in au

    # Set up the simulation
    sim = rb.Simulation()
    rebx = reboundx.Extras(sim)
    sim.dt = 0.001 if direction == "forward" else -0.001

    # Add the massive bodies from the precomputed barycentric states
    for name, state, mass in zip(planet_names, planet_states, planet_masses):
        sim.add(
            m=mass,
            x=state[0], y=state[1], z=state[2],
            vx=state[3], vy=state[4], vz=state[5],
            hash=name,
        )

    # Add the test particles (massless)
    for name, state in zip(particle_names, particle_states):
        sim.add(
            x=state[0], y=state[1], z=state[2],
            vx=state[3], vy=state[4], vz=state[5],
            hash=name,
        )

    ps = sim.particles

    # Add the gravitational harmonics of the Earth
    gh = rebx.load_force("gravitational_harmonics")
    rebx.add_force(gh)
    ps["Earth"].params["J2"] = J2
    ps["Earth"].params["J4"] = J4
    ps["Earth"].params["R_eq"] = RE_eq
    ps["Earth"].r = dmin  # set size of Earth
    ps["Luna"].r = dmin/4  # set size of Moon

    # gr_full is the general relativity correction for all bodies
    gr = rebx.load_force("gr_full")
    rebx.add_force(gr)
    gr.params["c"] = rbxConstants.C

    # Move to the center of momentum frame before integrating
    sim.move_to_com()

    # Disable collision detection and the influence of the massless particles on the planets
    sim.collision = "none"
    sim.N_active = len(planet_names)
    sim.testparticle_type = 0

    outputs = {name: [] for name in particle_names}

    # Integrate the simulation and save the state vectors and orbital elements. These are not time
    # steps, but the times at which the simulation state is saved.
    for time in times:

        sim.move_to_com()
        sim.integrate(time)
        sim.move_to_hel()

        for name in particle_names:

            # Skip the particle if it is no longer in the simulation
            try:
                ps[name]
            except rb.ParticleNotFound:
                continue

            state_vect_hel, orb_elem, planet_dists = extractSimParams(
                ps, name, planet_names, reference_frame=reference_frame)

            # Convert the rebound Orbit to a picklable object holding the attributes used downstream
            orb_ns = SimpleNamespace(
                a=orb_elem.a, e=orb_elem.e, inc=orb_elem.inc,
                Omega=orb_elem.Omega, omega=orb_elem.omega, f=orb_elem.f)

            outputs[name].append([time, state_vect_hel, orb_ns, planet_dists])

    return outputs


def reboundSimulate(
        julian_date, state_vect, traj=None,
        direction="forward", sim_days=60, n_outputs=500, obj_name="obj", obj_mass=0.0, mc_runs=100,
        reference_frame="heliocentric", ephem_source="local", n_cpu=None,
        show_progress=True, verbose=False):
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
        sim_days: [float] Length of integration in days, default is 60 days.
        n_outputs: [int] Number of outputs (samples along the simulation), default is 500.
        obj_name: [str] Name of the object that's being integrated, default is "obj".
        obj_mass: [float] Mass of object in solar masses if asteroid or larger object, default is 0.0.
        mc_runs: [int] Number of Monte Carlo simulations to run, default is 100.
        reference_frame: [str] Reference frame to use for the state vector. Options:
            - "heliocentric" (default)
            - "geocentric"
        ephem_source: [str] Source of the planetary positions used to seed the simulation:
            - "local" (default): the local DE430 ephemeris (fast, offline, no web queries).
            - "horizons": the JPL Horizons web service (slower, requires network).
        n_cpu: [int] Number of parallel processes used to integrate the Monte Carlo realizations.
            If None (default), uses max(1, os.cpu_count() - 1). Each realization is integrated in
            its own independent simulation, so its adaptive timestep does not affect the others.
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

        


    # Simulation end time in years and the length of one year in units where G=1
    tsimend = sim_days/365.25
    year = 2.0*np.pi

    # Convert from UTC to TDB
    time_utc = astropy.time.Time(julian_date, format='jd', scale='utc')
    time_tdb = time_utc.tdb.jd

    # Set up the output time array (not integration steps, but the times at which state is saved)
    if direction == "forward":
        times = np.linspace(0, year*tsimend, n_outputs)
    else:
        times = np.linspace(0, -year*tsimend, n_outputs)

    # Add the Sun and the planets
    planet_names = [
        "Sun",
        "Mercury", "Venus", "Earth", "Luna", "Mars",
        "Jupiter", "Saturn", "Uranus", "Neptune"
    ]

    # Names as resolved by JPL Horizons (used only for the "horizons" ephemeris source)
    horizons_names = {"Earth": "Geocenter"}

    # Seed the planets once in a parent simulation. Their raw barycentric states and masses are then
    # passed to the integration workers, so no worker touches the network or the ephemeris kernel.
    parent_sim = rb.Simulation()

    if ephem_source == "horizons":

        ### HOTFIX: Solve the cert issue for the JPL Horizons web queries
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        ###

        # Seed the planets from the JPL Horizons web service
        for name in planet_names:
            parent_sim.add(horizons_names.get(name, name), date=f"JD{time_tdb:.6f}", hash=name)

    else:

        # Seed the planets from the local DE430 ephemeris (fast, offline)
        jpl_ephem_data = SPK.open(config.jpl_ephem_file)
        for name in planet_names:
            body_state, body_mass = ephemBodyStateRebound(name, time_tdb, jpl_ephem_data)
            parent_sim.add(
                m=body_mass,
                x=body_state[0], y=body_state[1], z=body_state[2],
                vx=body_state[3], vy=body_state[4], vz=body_state[5],
                hash=name,
            )

    # Read the raw barycentric planet states (before move_to_com) and masses to hand to the workers
    pp = parent_sim.particles
    planet_states = [[pp[n].x, pp[n].y, pp[n].z, pp[n].vx, pp[n].vy, pp[n].vz] for n in planet_names]
    planet_masses = [pp[n].m for n in planet_names]

    # The Earth's barycentric state at the epoch is identical for the nominal solution and every
    # Monte Carlo realization, so it is computed once and reused for all of them.
    earth_state_ref = planet_states[planet_names.index("Earth")]

    # To start the meteoroid in the right spot, feed in the barycentric position (AU) and velocity
    # (AU/year divided by 2pi)
    state_vect_rot = convertToBarycentric(state_vect, time_tdb, earth_state=earth_state_ref)

    if verbose:
        print("Initial state vector in ECI coordinates:")
        print("[ x,  y,  z] = ", state_vect[:3])
        print("[vx, vy, vz] = ", state_vect[3:])

        print("Initial state vector in barycentric coordinates:")
        print("[ x,  y,  z] = ", state_vect_rot[:3])
        print("[vx, vy, vz] = ", state_vect_rot[3:])

    # Helper to assemble a picklable task for the integration worker
    def _make_task(particle_states, particle_names):
        return {
            "planet_names": planet_names,
            "planet_states": planet_states,
            "planet_masses": planet_masses,
            "particle_states": particle_states,
            "particle_names": particle_names,
            "times": list(times),
            "direction": direction,
            "reference_frame": reference_frame,
        }

    if verbose:
        print("Running simulation...")
        print(f"Simulation time: {tsimend:.2f} years ({tsimend*365.25:.2f} days)")
        print(f"Number of outputs: {n_outputs}")
        print(f"Direction: {direction}")

    # Nominal solution, integrated in its own simulation (in the main process)
    nominal_result = _integrateParticles(_make_task([state_vect_rot], [obj_name]))
    outputs = nominal_result[obj_name]

    # Monte Carlo realizations, each integrated in its own independent simulation, optionally across
    # multiple processes
    outputs_mc = {}
    if state_vect_realizations:

        # Convert each realization to barycentric coordinates (cheap, done in the parent process)
        mc_names = [f"{obj_name}_MC_{i}" for i in range(len(state_vect_realizations))]
        mc_states = [convertToBarycentric(sv, time_tdb, earth_state=earth_state_ref)
                     for sv in state_vect_realizations]

        # Resolve the number of worker processes
        if n_cpu is None:
            n_cpu = max(1, (os.cpu_count() or 1) - 1)
        n_cpu = max(1, min(n_cpu, len(mc_states)))

        # One task per realization, so each gets its own decoupled adaptive timestep
        tasks = [_make_task([s], [n]) for s, n in zip(mc_states, mc_names)]

        n_total = len(tasks)
        is_tty = show_progress and sys.stdout.isatty()

        # Live in-place progress bar on a terminal; stay quiet otherwise (avoids spamming log files)
        def _report(done, t_start):
            if is_tty:
                elapsed = time.time() - t_start
                print("\r  Monte Carlo: {:d}/{:d} realizations integrated ({:.1f} s)".format(
                    done, n_total, elapsed), end="", flush=True)

        if show_progress:
            print("Integrating {:d} Monte Carlo realizations on {:d} core(s)...".format(n_total, n_cpu))

        t_mc_start = time.time()

        if n_cpu == 1:
            results = []
            for k, t in enumerate(tasks, start=1):
                results.append(_integrateParticles(t))
                _report(k, t_mc_start)

        else:
            results = [None]*n_total
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_cpu) as executor:
                future_to_idx = {executor.submit(_integrateParticles, t): idx
                                 for idx, t in enumerate(tasks)}
                done = 0
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        print("\nWarning: Monte Carlo realization {:s} failed and was skipped: {:s}".format(
                            mc_names[idx], str(e)))
                        results[idx] = None
                    done += 1
                    _report(done, t_mc_start)

        n_ok = sum(1 for r in results if r)
        if show_progress:
            # Finish the in-place line (newline) on a terminal, or print the single summary otherwise
            prefix = "\r" if is_tty else "  "
            print("{:s}Monte Carlo: {:d}/{:d} realizations integrated ({:.1f} s){:s}".format(
                prefix, n_ok, n_total, time.time() - t_mc_start,
                "" if n_ok == n_total else "  [{:d} failed]".format(n_total - n_ok)))

        for res in results:
            if res:
                outputs_mc.update(res)

    return outputs, outputs_mc


if __name__ == "__main__":

    import os
    import argparse

    from wmpl.Utils.Pickling import loadPickle


    # Exit cleanly with a helpful message if REBOUND/REBOUNDx are not installed, instead of
    # crashing later with a cryptic "cannot unpack non-iterable NoneType" error.
    if not REBOUND_FOUND:
        print("")
        print("ERROR: the 'rebound' and 'reboundx' packages are required to run this script, "
              "but they could not be imported.")
        print("")
        print("Install them with:  pip install rebound reboundx")
        print("")
        print("Note: on Windows, 'reboundx' has no prebuilt wheel and does not compile with the "
              "MSVC compiler (it uses C features MSVC lacks). Use one of:")
        print("  - Windows Subsystem for Linux (WSL2, e.g. Ubuntu) - recommended, builds cleanly, or")
        print("  - a Linux or macOS machine.")
        print("'rebound' alone is not enough; 'reboundx' must import successfully too.")
        sys.exit(1)

    ###

    parser = argparse.ArgumentParser(description="Run REBOUND simulation for a given trajectory pickle file. The simulation is run 60 days backwards by default.")

    parser.add_argument("pickle_path", type=str, help="Path to the pickle file with the trajectory data.")

    parser.add_argument("--days", type=float, help="Run the simulation for the given number of days.", default=60)

    parser.add_argument("--forward", type=str, help="Run the simulation forward for the given number of days.")

    parser.add_argument("--mc", type=int, help="Run the simulation for the given number of Monte Carlo simulations."
                        "The default is 0", default=0)
    
    parser.add_argument("--geocentric", action="store_true",
                        help="Run the simulation in geocentric reference frame. Default is heliocentric.")
    
    parser.add_argument("--horizons", action="store_true",
                        help="Use the JPL Horizons web service for planet positions instead of the "
                        "local DE430 ephemeris. Slower and requires network access.")

    parser.add_argument("--cores", type=int, default=None,
                        help="Number of parallel processes used to integrate the Monte Carlo "
                        "realizations. Default: all but one core.")

    parser.add_argument("--verbose", action="store_true", help="Print out the progress of the simulation.")

    args = parser.parse_args()

    # Extract the number of days from the arguments and the simulation direction
    sim_days = args.days
    direction = "backward" if args.forward is None else "forward"

    # Source of the planetary ephemeris
    ephem_source = "horizons" if args.horizons else "local"
    print("Using {:s} for planet positions.".format(
        "JPL Horizons (web)" if args.horizons else "the local DE430 ephemeris"))

    # Number of parallel processes for the Monte Carlo integration
    n_cpu = args.cores if args.cores is not None else max(1, (os.cpu_count() or 1) - 1)
    if args.mc > 1:
        print("Monte Carlo integration will use up to {:d} core(s).".format(n_cpu))

    ###

    # Load the trajectory data from a pickle file
    traj = loadPickle(*os.path.split(args.pickle_path))


    ### Set reference frame settings ###
    reference_frame = "heliocentric"
    if args.geocentric:
        reference_frame = "geocentric"
        print("Running the simulation in geocentric reference frame.")

    
    # Set semi-major axis and periapsis units depending on the reference frame
    a_units = "AU"
    q_units = "AU"
    dist_unit_multiplier = 1.0  # Default is AU
    if reference_frame == "geocentric":
        a_units = "km"
        q_units = "km"
        dist_unit_multiplier = 149597870.7  # Convert AU to km

    ### ###

    
    # Run the simulation for the given number of days from the epoch of the trajectory
    t_run_start = time.time()
    sim_outputs, sim_outputs_mc = reboundSimulate(
        None, None, traj=traj, direction=direction, sim_days=sim_days,
        obj_name=traj.traj_id, mc_runs=args.mc, reference_frame=reference_frame,
        ephem_source=ephem_source, n_cpu=n_cpu, verbose=args.verbose
        )
    sim_wall = time.time() - t_run_start

    # Compute the 95% CI for the orbital elements from the Monte Carlo realizations
    a_ci_str = ""
    q_ci_str = ""
    e_ci_str = ""
    incl_ci_str = ""
    Omega_ci_str = ""
    omega_ci_str = ""
    f_ci_str = ""
    if len(sim_outputs_mc):

        # Extract the orbital elements at the end for each MC realization
        a_mc = []
        q_mc = []
        e_mc = []
        incl_mc = []
        Omega_mc = []
        omega_mc = []
        f_mc = []

        for mc_name in sim_outputs_mc:
            a = sim_outputs_mc[mc_name][-1][2].a*dist_unit_multiplier
            e = sim_outputs_mc[mc_name][-1][2].e
            a_mc.append(a)
            e_mc.append(e)
            q_mc.append((1 - e)*a)
            incl_mc.append(sim_outputs_mc[mc_name][-1][2].inc)
            Omega_mc.append(sim_outputs_mc[mc_name][-1][2].Omega)
            omega_mc.append(sim_outputs_mc[mc_name][-1][2].omega)
            f_mc.append(sim_outputs_mc[mc_name][-1][2].f)
        

        a_95ci_low = np.percentile(a_mc, 2.5)
        a_95ci_high = np.percentile(a_mc, 97.5)
        q_95ci_low = np.percentile(q_mc, 2.5)
        q_95ci_high = np.percentile(q_mc, 97.5)
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
        a_std = ((a_95ci_high - a_95ci_low)/2.0)/1.96 # Use 95% CI to ignore outliers
        q_std = ((q_95ci_high - q_95ci_low)/2.0)/1.96 # Use 95% CI to ignore outliers
        e_std = ((e_95ci_high - e_95ci_low)/2.0)/1.96 # Use 95% CI to ignore outliers
        incl_std = ((incl_95ci_high - incl_95ci_low)/2.0)/1.96 # Use 95% CI to ignore outliers
        Omega_std = scipy.stats.circstd(Omega_mc)
        omega_std = scipy.stats.circstd(omega_mc)
        f_std = scipy.stats.circstd(f_mc)


        a_ci_str = f" +/- {a_std:.6f} [{a_95ci_low:10.6f}, {a_95ci_high:10.6f}]"
        q_ci_str = f" +/- {q_std:.6f} [{q_95ci_low:10.6f}, {q_95ci_high:10.6f}]"
        e_ci_str = f" +/- {e_std:.6f} [{e_95ci_low:10.6f}, {e_95ci_high:10.6f}]"
        incl_ci_str = f" +/- {np.degrees(incl_std):.6f} [{np.degrees(incl_95ci_low):10.6f}, {np.degrees(incl_95ci_high):10.6f}]"
        Omega_ci_str = f" +/- {np.degrees(Omega_std):.6f} [{np.degrees(Omega_95ci_low):10.6f}, {np.degrees(Omega_95ci_high):10.6f}]"
        omega_ci_str = f" +/- {np.degrees(omega_std):.6f} [{np.degrees(omega_95ci_low):10.6f}, {np.degrees(omega_95ci_high):10.6f}]"
        f_ci_str = f" +/- {np.degrees(f_std):.6f} [{np.degrees(f_95ci_low):10.6f}, {np.degrees(f_95ci_high):10.6f}]"


    ### Compute the epoch of the final simulation

    # Extract the final time in days
    final_sim_days = sim_outputs[-1][0]/(2*np.pi)*365.25

    # Compute the final epoch
    final_epoch_jd = traj.jdt_ref + final_sim_days

    # Convert the epoch to UTC
    time_utc = astropy.time.Time(final_epoch_jd, format='jd', scale='utc')

    ###

    # Detect close encounters (Hill-sphere criterion). Needed both for the summary and the report file.
    n_hill = 3.0
    encounters = detectCloseEncounters(sim_outputs, n_hill=n_hill)

    # List of bodies for which the distance is tracked (planet_dists dict keys)
    dist_bodies = list(sim_outputs[0][3].keys())

    # Nominal final orbital elements. Note the mapping to the wmpl convention:
    #   peri (argument of perihelion) = REBOUND omega ; node (ascending node) = REBOUND Omega
    a_val = sim_outputs[-1][2].a*dist_unit_multiplier
    e_val = sim_outputs[-1][2].e
    q_val = (1 - e_val)*a_val
    i_val = np.degrees(sim_outputs[-1][2].inc)
    peri_val = np.degrees(sim_outputs[-1][2].omega)
    node_val = np.degrees(sim_outputs[-1][2].Omega)
    f_val = np.degrees(sim_outputs[-1][2].f)

    # Print a readable summary
    hdr = "=" * 78
    print("\n" + hdr)
    print("  REBOUND orbit integration  |  {:s}".format(str(traj.traj_id)))
    print(hdr)
    print("  Ephemeris    : {:s}".format("JPL Horizons (web)" if args.horizons else "local DE430"))
    print("  Direction    : {:s}, {:.2f} days".format(direction, sim_days))
    print("  Frame        : {:s}".format(reference_frame))
    print("  Start epoch  : {:.6f} JD (TDB)".format(traj.jdt_ref))
    print("  Final epoch  : {:.6f} JD (TDB)  =  {:s} UTC".format(final_epoch_jd, time_utc.iso))
    if len(sim_outputs_mc):
        print("  Monte Carlo  : {:d} realizations on {:d} core(s)".format(len(sim_outputs_mc), n_cpu))
    print("  Runtime      : {:.1f} s".format(sim_wall))
    print("-" * 78)
    if len(sim_outputs_mc):
        print("  Final orbital elements   (nominal  +/- 1 sigma  [95% CI]):")
    else:
        print("  Final orbital elements   (nominal):")
    print("")
    print("    a    = {:>13.6f}{:s}  {:s}".format(a_val, a_ci_str, a_units))
    print("    q    = {:>13.6f}{:s}  {:s}".format(q_val, q_ci_str, q_units))
    print("    e    = {:>13.6f}{:s}".format(e_val, e_ci_str))
    print("    i    = {:>13.6f}{:s}  deg".format(i_val, incl_ci_str))
    print("    peri = {:>13.6f}{:s}  deg".format(peri_val, omega_ci_str))
    print("    node = {:>13.6f}{:s}  deg".format(node_val, Omega_ci_str))
    print("    f    = {:>13.6f}{:s}  deg".format(f_val, f_ci_str))
    print("-" * 78)

    # Close-encounter summary
    if encounters:
        print("  Close encounters (< {:.0f} Hill radii):".format(n_hill))
        for enc in encounters:
            print("    {:<8s} {:12.6f} AU ({:12.1f} km)  at t = {:+9.3f} d   ({:.2f} R_Hill, R_Hill = {:.6f} AU)".format(
                enc["body"], enc["min_dist_au"], enc["min_dist_au"]*149597870.7,
                enc["time_days"], enc["n_hill"], enc["hill_radius_au"]))
    else:
        print("  Close encounters (< {:.0f} Hill radii): none detected".format(n_hill))
    print(hdr)


    # Save the results to a file
    out_dir = os.path.dirname(args.pickle_path)
    results_txt_path = os.path.join(out_dir, "rebound_simulation_results.txt")
    plot_png_path = os.path.join(out_dir, "rebound_simulation.png")
    with open(results_txt_path, "w") as f:

        # Save the nominal orbital elements and the errors
        f.write("Orbital elements {:.2f} days from the epoch {:.6f} {:s}\n".format(sim_days, traj.jdt_ref, direction))
        f.write("a    = {:>10.6f}{:s} {:s}\n".format(sim_outputs[-1][2].a*dist_unit_multiplier, a_ci_str, a_units))
        f.write("q    = {:>10.6f}{:s} {:s}\n".format((1 - sim_outputs[-1][2].e)*sim_outputs[-1][2].a*dist_unit_multiplier, q_ci_str, q_units))
        f.write("e    = {:>10.6f}{:s}\n".format(sim_outputs[-1][2].e, e_ci_str))
        f.write("i    = {:>10.6f}{:s} deg\n".format(np.degrees(sim_outputs[-1][2].inc), incl_ci_str))
        f.write("peri = {:>10.6f}{:s} deg\n".format(np.degrees(sim_outputs[-1][2].omega), omega_ci_str))
        f.write("node = {:>10.6f}{:s} deg\n".format(np.degrees(sim_outputs[-1][2].Omega), Omega_ci_str))
        f.write("f    = {:>10.6f}{:s} deg\n".format(np.degrees(sim_outputs[-1][2].f), f_ci_str))

        # Save the detected close encounters (Hill-sphere criterion)
        f.write("\nClose encounters (< {:.0f} Hill radii):\n".format(n_hill))
        if encounters:
            for enc in encounters:
                f.write("  {:<8s} min dist = {:10.6f} AU ({:12.1f} km) at t = {:10.4f} d, R_Hill = {:.6f} AU ({:.2f} R_Hill)\n".format(
                    enc["body"], enc["min_dist_au"], enc["min_dist_au"]*149597870.7,
                    enc["time_days"], enc["hill_radius_au"], enc["n_hill"]))
        else:
            f.write("  None detected.\n")

        # Save the nominal orbital elements and per-body distances from the initial to end time
        # of the simulation. Distances to each body are always in AU.
        f.write("\nOrbital elements and distances to each body [AU] from the initial to end time of the simulation:\n")
        f.write("(distance columns: " + ", ".join(dist_bodies) + ")\n")
        for i, output in enumerate(sim_outputs):

            # Build the per-body distance columns (in AU)
            dist_str = "".join(", {:s} = {:10.6f}".format(body, output[3][body]) for body in dist_bodies)

            # Save the orbital elements and distances at the given time
            f.write(f"t = {output[0]/(2*np.pi)*365.25:.6f} d, a = {output[2].a*dist_unit_multiplier:10.6f}, e = {output[2].e:10.6f}, i = {np.degrees(output[2].inc):10.6f}, Omega = {np.degrees(output[2].Omega):10.6f}, omega = {np.degrees(output[2].omega):10.6f}, f = {np.degrees(output[2].f):10.6f}{dist_str}\n")

        # If the MC was run, save orbital elements of individual MC runs
        if len(sim_outputs_mc):
            f.write("\nOrbital elements of individual Monte Carlo runs:\n")
            for mc_name in sim_outputs_mc:
                f.write(f"{mc_name}:\n")
                f.write("a    = {:>10.6f} {:s}\n".format(sim_outputs_mc[mc_name][-1][2].a*dist_unit_multiplier, a_units))
                f.write("q    = {:>10.6f} {:s}\n".format((1 - sim_outputs_mc[mc_name][-1][2].e)*sim_outputs_mc[mc_name][-1][2].a*dist_unit_multiplier, q_units))
                f.write("e    = {:>10.6f}\n".format(sim_outputs_mc[mc_name][-1][2].e))
                f.write("i    = {:>10.6f} deg\n".format(np.degrees(sim_outputs_mc[mc_name][-1][2].inc)))
                f.write("peri = {:>10.6f} deg\n".format(np.degrees(sim_outputs_mc[mc_name][-1][2].omega)))
                f.write("node = {:>10.6f} deg\n".format(np.degrees(sim_outputs_mc[mc_name][-1][2].Omega)))
                f.write("f    = {:>10.6f} deg\n".format(np.degrees(sim_outputs_mc[mc_name][-1][2].f)))

    # Plot the orbital elements of the before and after simulation on the same plot (one subplot for each element)
    fig, axs = plt.subplots(3, 3, figsize=(14, 10), sharex=True)

    # Time in days
    t = [x[0]/(2*np.pi)*365.25 for x in sim_outputs]

    a = [x[2].a*dist_unit_multiplier for x in sim_outputs]
    e = [x[2].e for x in sim_outputs]
    incl = [x[2].inc for x in sim_outputs]
    Omega = [x[2].Omega for x in sim_outputs]
    omega = [x[2].omega for x in sim_outputs]
    f = [x[2].f for x in sim_outputs]

    # Distance from the planets
    planet_dists = [x[3] for x in sim_outputs]
    earth_dist = [x["Earth"] for x in planet_dists]

    # Find the time when the object exits the Earth's Hill sphere
    earth_hill = HILL_RADII_AU["Earth"] # AU
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
    axs[0, 2].plot(t, np.array(earth_dist)*dist_unit_multiplier)

    # Mark the exit from the Earth's Hill sphere
    if exit_index is not None:
        axs[0, 2].axvline(t[exit_index], color="red", linestyle="--", label="Exit from Earth's Hill sphere")

        # Set the X axis limit so the maximum time is at the exit from the Earth's Hill sphere
        axs[0, 2].set_xlim(xmax=t[exit_index])



    # Plot the distance from the Sun + inner planets
    inner_planets = ["Sun", "Mercury", "Venus", "Earth", "Luna", "Mars"]
    for planet in inner_planets:
        planet_dist = [x[3][planet] for x in sim_outputs]
        axs[1, 2].plot(t, planet_dist, label=planet)

    axs[1, 2].set_ylabel("Distance [AU]")
    axs[1, 2].legend()

    # Plot the distance from the outer planets
    outer_planets = ["Jupiter", "Saturn", "Uranus", "Neptune"]
    for planet in outer_planets:
        planet_dist = [x[3][planet] for x in sim_outputs]
        axs[2, 2].plot(t, planet_dist, label=planet)

    axs[2, 2].set_ylabel("Distance [AU]")
    axs[2, 2].legend()


    # Mark the detected close encounters at the point of closest approach on the distance subplots
    labeled_axes = set()
    for enc in encounters:
        body = enc["body"]
        t_enc = enc["time_days"]
        d_enc = enc["min_dist_au"]

        # Determine which distance subplot(s) show this body
        marks = []
        if body in inner_planets:
            marks.append((axs[1, 2], d_enc))
        if body in outer_planets:
            marks.append((axs[2, 2], d_enc))
        if body == "Earth":
            # The Earth-distance subplot is scaled by the reference-frame unit multiplier
            marks.append((axs[0, 2], d_enc*dist_unit_multiplier))

        for ax, d_plot in marks:

            # Only add the legend label once per axis
            label = "Close encounter" if ax not in labeled_axes else None
            labeled_axes.add(ax)

            ax.plot(t_enc, d_plot, marker="*", color="red", markersize=14, linestyle="none",
                    zorder=5, label=label)
            ax.annotate(body, (t_enc, d_plot), textcoords="offset points", xytext=(5, 5),
                        color="red", fontsize=8)


    # Set the axis labels
    for ax in axs.flatten():

        ax.set_xlabel("Time [days]")

        # Disable offset and scientific notation
        ax.ticklabel_format(useOffset=False, style='plain')


    # Plot the MC realizations (all in thin alpha=0.5 lines)
    for mc_name in sim_outputs_mc:

        a_mc = [x[2].a*dist_unit_multiplier for x in sim_outputs_mc[mc_name]]
        e_mc = [x[2].e for x in sim_outputs_mc[mc_name]]
        incl_mc = [x[2].inc for x in sim_outputs_mc[mc_name]]
        Omega_mc = [x[2].Omega for x in sim_outputs_mc[mc_name]]
        omega_mc = [x[2].omega for x in sim_outputs_mc[mc_name]]
        f_mc = [x[2].f for x in sim_outputs_mc[mc_name]]
        earth_dist = [x[3]["Earth"] for x in sim_outputs_mc[mc_name]]

        axs[0, 0].plot(t, a_mc, alpha=0.5, color='k', lw=0.5)
        axs[0, 1].plot(t, e_mc, alpha=0.5, color='k', lw=0.5)
        axs[1, 0].plot(t, np.degrees(incl_mc), alpha=0.5, color='k', lw=0.5)
        axs[1, 1].plot(t, np.degrees(Omega_mc), alpha=0.5, color='k', lw=0.5)
        axs[2, 0].plot(t, np.degrees(omega_mc), alpha=0.5, color='k', lw=0.5)
        axs[2, 1].plot(t, np.degrees(f_mc), alpha=0.5, color='k', lw=0.5)
        axs[0, 2].plot(t, np.array(earth_dist)*dist_unit_multiplier, alpha=0.5, color='k', lw=0.5)

    
    
    axs[0, 0].set_ylabel("a [{:s}]".format(a_units))
    axs[0, 1].set_ylabel("e")
    axs[1, 0].set_ylabel("i [deg]")
    # axs[1, 1] plots REBOUND Omega (ascending node); axs[2, 0] plots omega (argument of perihelion)
    axs[1, 1].set_ylabel("node [deg]")
    axs[2, 0].set_ylabel("peri [deg]")
    axs[2, 1].set_ylabel("f [deg]")
    axs[0, 2].set_ylabel("Earth distance [{:s}]".format(a_units))

    # Only draw a legend on axes that actually have labeled artists (avoids the empty-legend warning)
    for ax in axs.flatten():
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend()

    plt.tight_layout()

    # Save the figure
    plt.savefig(plot_png_path)

    # Report the saved outputs
    print("  Saved report : {:s}".format(results_txt_path))
    print("  Saved plot   : {:s}".format(plot_png_path))
    print(hdr)

    plt.show()