""" Functions for running REBOUND simulations on wmpl trajectories.

Requires the 'rebound' and 'reboundx' packages. These work on Linux and macOS. They do NOT
install on native Windows: 'reboundx' has no Windows wheel and does not compile with MSVC
(it uses C99 variable-length arrays MSVC does not support). On Windows, use the Windows
Subsystem for Linux (WSL2) - install Ubuntu, set up the wmpl conda environment there, and run
from that shell. Installing only 'rebound' is not enough; 'reboundx' must import too.
"""

import os
import re
import sys
import json
import time
import warnings
import concurrent.futures
from types import SimpleNamespace

import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt

from jplephem.spk import SPK

REBOUND_FOUND = False
_REBOUND_IMPORT_ERROR = None

try:
    # Silence the noisy "pkg_resources is deprecated" warning emitted while importing reboundx
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
        import rebound as rb
        import reboundx
        from reboundx import constants as rbxConstants
    import astropy.time

    REBOUND_FOUND = True

except (ImportError, OSError) as e:
    # Keep optional-dependency failures quiet during import. Report the actual cause only if a
    #   REBOUND function or the command-line interface is used. OSError covers a reboundx library
    #   that fails to load because it was compiled against a different REBOUND version.
    _REBOUND_IMPORT_ERROR = str(e)

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


def _printReboundUnavailable(include_install_help=False):
    """ Print the captured REBOUND import failure when REBOUND is intentionally used. """

    print("ERROR: the 'rebound' and 'reboundx' packages are required, but they could not be imported.")
    if _REBOUND_IMPORT_ERROR is not None:
        print("The error was: {}".format(_REBOUND_IMPORT_ERROR))

    if include_install_help:
        print("")
        print("Install them with:")
        print("  pip install rebound")
        print("  pip install --no-build-isolation reboundx")
        print("('--no-build-isolation' compiles reboundx against the installed rebound.)")
        print("If reboundx is installed but fails to load, recompile it against the installed rebound:")
        print("  pip install --force-reinstall --no-deps --no-build-isolation --no-binary reboundx reboundx")
        print("(see the README for keeping an older rebound).")
        print("")
        print("Note: on Windows, 'reboundx' has no prebuilt wheel and does not compile with the "
              "MSVC compiler (it uses C features MSVC lacks). Use one of:")
        print("  - Windows Subsystem for Linux (WSL2, e.g. Ubuntu) - recommended, builds cleanly, or")
        print("  - a Linux or macOS machine.")
        print("'rebound' alone is not enough; 'reboundx' must import successfully too.")


def _reboundMajorVersion():
    """ Major version of the installed REBOUND, or 5 if it cannot be read.

    The version decides which keyword names a particle (see _addNamedParticle). A version string
    that does not start with a number is assumed to be a recent one, since the naming keyword only
    changed going forward.
    """

    try:
        return int(rb.__version__.split(".")[0])

    except (AttributeError, IndexError, ValueError):
        return 5


def _addNamedParticle(sim, name, *args, **kwargs):
    """ Add a particle to a REBOUND simulation so that it can be retrieved as sim.particles[name].

    REBOUND 5 replaced particle hashes with names, so the keyword is 'name' there and 'hash' in
    REBOUND 4. Lookup by string, sim.particles[name], is the same in both. In REBOUND 5 a body
    queried from Horizons cannot be named through sim.add (the Horizons query takes 'name' as the
    body to look up), so it is named after it is added.

    Arguments:
        sim: [rebound.Simulation] The simulation.
        name: [str] Name used to identify the particle.
        *args, **kwargs: Passed on to sim.add (e.g. a Horizons name and date, or m, x, ..., vz).
    """

    if _reboundMajorVersion() < 5:
        sim.add(*args, hash=name, **kwargs)

    elif args and isinstance(args[0], str):
        sim.add(*args, **kwargs)
        sim.particles[sim.N - 1].name = name

    else:
        sim.add(*args, name=name, **kwargs)


def _setHeartbeat(sim, func):
    """ Set the heartbeat function of a REBOUND simulation.

    In REBOUND 5.1.1 the 'heartbeat' property setter fails for any function ("'Simulation' object
    has no attribute '_hb'"), because it stores its ctypes wrapper in an attribute missing from
    Simulation.__slots__. In that case the C function pointer is set directly.

    Arguments:
        sim: [rebound.Simulation] The simulation.
        func: [function] Heartbeat function, called with a pointer to the simulation.

    Return:
        [object or None] The ctypes wrapper when the pointer was set directly, else None. The caller
            must keep it alive for the whole integration, or the callback is garbage collected.
    """

    try:
        sim.heartbeat = func
        return None

    except AttributeError as e:

        # Any other AttributeError is a real error in the caller's function, not the REBOUND bug
        if "_hb" not in str(e):
            raise

        # What the working setter does, minus the assignment that fails: wrap the function in the
        #   C callback type and store it in the simulation's function pointer
        heartbeat_ref = rb.simulation.AFF(func)
        sim._heartbeat = heartbeat_ref

        return heartbeat_ref


def checkReboundxUsable():
    """ Raise a clear error if the installed REBOUNDx cannot attach to a REBOUND simulation.

    This is the same check _checkReboundxAttached makes, on a throwaway simulation, so that a
    mis-compiled REBOUNDx is reported before any ephemeris or integration work is done rather than
    once per worker process partway through a Monte Carlo run.
    """

    sim = rb.Simulation()

    # The Extras object has to stay referenced while the simulation is checked, or it detaches
    rebx = reboundx.Extras(sim)
    _checkReboundxAttached(sim)
    del rebx


def _checkReboundxAttached(sim):
    """ Raise a clear error if REBOUNDx failed to attach to the simulation.

    reboundx has no prebuilt wheels, so pip compiles it from source, by default in an isolated
    environment with the newest REBOUND rather than the installed one. If the two versions have a
    different C simulation structure, reboundx writes its pointer outside the installed REBOUND's
    structure (overwriting other memory) and sim.extras stays empty. Setting any REBOUNDx parameter
    then fails with an unhelpful "Need to attach reboundx.Extras instance" error.

    Arguments:
        sim: [rebound.Simulation] The simulation, just after reboundx.Extras(sim) was created.
    """

    if not sim.extras:
        raise RuntimeError(
            "REBOUNDx did not attach to the REBOUND simulation. This happens when reboundx was compiled "
            "against a different REBOUND version than the installed one (rebound {:s}, reboundx {:s}). "
            "Recompile reboundx against the installed REBOUND with:\n"
            "  pip install --force-reinstall --no-deps --no-build-isolation --no-binary reboundx "
            "reboundx\n"
            "This installs the latest reboundx, which needs rebound 5 or newer, so update rebound "
            "first. To keep this rebound, pin the reboundx released with it instead by appending "
            "'==<version>' (e.g. reboundx 4.3.0 for rebound 4.3.0); a newer reboundx does not compile "
            "against an older rebound.".format(rb.__version__, reboundx.__version__))


# Hill-sphere radii in AU used for close-encounter detection.
#
# Convention: r_H = a*(m/(3*M_sun))**(1/3), i.e. evaluated at the semi-major axis with no
# eccentricity factor. The alternative perihelion form, a*(1 - e)*(m/(3*M_sun))**(1/3), gives a
# smaller sphere; the larger value is used here deliberately, because these radii define a
# screening threshold for close encounters and a more inclusive sphere is the safer choice.
# The values are computed from the same GM table the simulation masses come from
# (reboundBodyMassSolar), so the table is self-consistent and reproducible.
#
# The Moon (Luna) uses its Earth-relative Hill radius, a_moon*(m_moon/(3*m_earth))**(1/3).
# The Sun is excluded (it has no Hill sphere in this context).
HILL_RADII_AU = {
    "Mercury": 0.001475,  # a = 0.387098 AU
    "Venus":   0.006759,  # a = 0.723332 AU
    "Earth":   0.010004,  # a = 1.000000 AU
    "Luna":    0.000411,  # Earth-relative, a = 384400 km
    "Mars":    0.007246,  # a = 1.523679 AU
    "Jupiter": 0.355322,  # a = 5.204267 AU
    "Saturn":  0.437671,  # a = 9.582017 AU
    "Uranus":  0.469492,  # a = 19.229411 AU
    "Neptune": 0.776641,  # a = 30.103658 AU
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
    """ Detect close encounters between the integrated object and the planets (and the Moon), by
    scanning the sampled output.

    DEPRECATED for new code, because scanning the output is sampling-limited: near the Earth the
    object can cross a large fraction of the Moon's detection sphere between two output samples, so a
    lunar encounter can be missed outright or its minimum distance overestimated (by ~60000 km at
    n_outputs = 100 on a real trajectory). Prefer encountersFromMinDistances, which uses the minima
    tracked at every internal integrator timestep and refined between steps. This function is kept
    for callers that only have sampled output to work from.

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


def hermiteClosestApproach(t0, r0, v0, t1, r1, v1):
    """ Find the closest approach of a relative trajectory within one integrator step.

    The relative position over the step is approximated by the cubic Hermite interpolant that
    matches the relative positions and velocities at both ends of the step. The interpolant
    reproduces straight-line motion exactly, so for a weakly deflected flyby (the usual case for a
    meteoroid) the closest approach it gives is accurate even when the step is much longer than the
    encounter, where the step-end samples alone can miss the minimum by a large fraction of the step
    length. A strongly bent trajectory would make the interpolant underestimate the minimum if the
    step spanned the whole encounter, but IAS15 shrinks its step in that regime: on slow, deep flybys
    of every planet (hyperbolic eccentricity down to ~1) the refined minimum was within 3e-4 of a
    dense re-integration.

    Arguments:
        t0: [float] Time at the start of the step.
        r0: [ndarray] Relative position (object minus body) at t0.
        v0: [ndarray] Relative velocity at t0.
        t1: [float] Time at the end of the step (may be earlier than t0 for backward integration).
        r1: [ndarray] Relative position at t1.
        v1: [ndarray] Relative velocity at t1.

    Return:
        (d_min, t_min): [tuple of floats] Minimum distance of the interpolant on the step (including
            its ends) and the time at which it occurs.
    """

    h = t1 - t0

    # Power-basis coefficients of the interpolant r(s) = A + B*s + C*s^2 + D*s^3, s = (t - t0)/h
    A = r0
    B = h*v0
    C = -3*r0 - 2*h*v0 + 3*r1 - h*v1
    D = 2*r0 + h*v0 - 2*r1 + h*v1

    # d|r|^2/ds is proportional to the quintic r(s).r'(s), whose real roots in (0, 1) are the
    # extrema of the distance inside the step
    P = np.polynomial.polynomial
    drr = np.zeros(6)
    for k in range(3):
        drr = P.polyadd(drr, P.polymul([A[k], B[k], C[k], D[k]], [B[k], 2*C[k], 3*D[k]]))

    # The step ends are always candidates, so a step with no interior minimum returns the closer
    #   of its two ends. Complex roots and roots outside the step are not minima of this step.
    s_candidates = [0.0, 1.0] + [rt.real for rt in P.polyroots(drr)
                                 if (abs(rt.imag) < 1e-9) and (0.0 < rt.real < 1.0)]

    # The quintic gives every extremum, maxima included, so the candidates are simply compared
    d_min, t_min = np.inf, t0
    for s in s_candidates:
        r = A + B*s + C*s**2 + D*s**3
        d = np.sqrt(r @ r)
        if d < d_min:
            d_min, t_min = d, t0 + s*h

    return d_min, t_min


def _encounterRecord(body, dist_au, time_days):
    """ One close-encounter entry, in the format shared by all the encounter lists.

    Only bodies with a Hill radius can be encountered, which is the same gate the callers apply
    before they get here (the Sun is tracked for its distance but is not an encounter body).
    """

    hill_radius = HILL_RADII_AU.get(body)
    if hill_radius is None:
        raise KeyError("No Hill radius for '{:s}', so it cannot be an encounter body.".format(body))

    return {
        "body": body,
        "min_dist_au": dist_au,
        "time_days": time_days,
        "hill_radius_au": hill_radius,
        "n_hill": dist_au/hill_radius,
        "index": None,
    }


def encountersFromMinDistances(min_dist_au, min_time_days, n_hill=3.0):
    """ Build the close-encounter list from the closest-approach distances measured during the
    integration (see the heartbeat tracking in _integrateParticles).

    This is the preferred alternative to detectCloseEncounters, which scans the sampled output and
    is therefore sampling-limited: near the Earth the object can move a large fraction of the Moon's
    detection sphere between two output samples, so a lunar encounter can be missed entirely or its
    minimum distance badly overestimated. The values used here are tracked at every internal
    integrator timestep and refined between steps instead.

    This gives at most one encounter per body, its deepest approach. The "encounters" diagnostic of
    _integrateParticles lists every passage instead, so repeated encounters with the same body are
    not lost, and that is what the command line report uses. This function remains the way to build
    the list for a caller that only has the closest-approach dictionaries, for example from a
    diagnostics dict saved by an older version.

    Arguments:
        min_dist_au: [dict] {body: closest approach in AU, or None if the body was not tracked}.
        min_time_days: [dict] {body: time of closest approach in days}.

    Keyword arguments:
        n_hill: [float] Multiple of the Hill radius used as the close-encounter threshold.
            Default is 3.0.

    Return:
        [list] Encounter dicts in the same format as detectCloseEncounters, sorted by closeness
            (min_dist/R_Hill ascending).
    """

    encounters = []

    for body, hill_radius in HILL_RADII_AU.items():

        dist = min_dist_au.get(body)
        if dist is None:
            continue

        if dist < n_hill*hill_radius:
            encounters.append(_encounterRecord(body, dist, min_time_days.get(body)))

    encounters.sort(key=lambda e: e["n_hill"])

    return encounters


def cloneEncounterSummary(clone_diag):
    """ Aggregate the close encounters of the Monte Carlo clones, per body.

    A clone can meet the same body more than once, so the number of clones that met it and the
    number of passages they made are counted separately: the first is what a fraction of the
    ensemble is meaningful for, the second says how much of the ensemble's history involved that
    body.

    Arguments:
        clone_diag: [dict] {clone name: diagnostics} from reboundSimulate, each holding an
            "encounters" list as built by _integrateParticles.

    Return:
        [dict] {body: {"count": number of clones with at least one encounter,
                       "n_encounters": total number of passages over all clones,
                       "closest_au": closest approach over all clones}}
    """

    summary = {}

    for diag in clone_diag.values():

        clone_list = diag.get("encounters", [])

        # One increment per clone per body, however many times that clone met it
        for body in {enc["body"] for enc in clone_list}:
            summary.setdefault(body, {"count": 0, "n_encounters": 0, "closest_au": np.inf})
            summary[body]["count"] += 1

        # Every passage counts towards the totals and the closest approach
        for enc in clone_list:
            entry = summary[enc["body"]]
            entry["n_encounters"] += 1
            entry["closest_au"] = min(entry["closest_au"], enc["min_dist_au"])

    return summary


def estimateLyapunovFromMC(sim_outputs, sim_outputs_mc):
    """ Estimate the trajectory divergence timescale from the Monte Carlo ensemble.

    The Monte Carlo realizations are trajectories started from slightly different state vectors
    drawn from the measurement covariance, so their separation from the nominal solution over time
    measures exactly the divergence a Lyapunov analysis looks for - no extra variational particles
    are needed, and this costs nothing beyond an MC run that has already been done.

    The root-mean-square separation delta(t) of the realizations from the nominal solution is fitted
    both as exponential growth (ln delta linear in t, the chaotic case, giving a Lyapunov exponent
    and time) and as linear growth (the regular/non-chaotic case). Whichever describes the data
    better is reported.

    Note that this is a finite-time estimate driven by the actual measurement uncertainty, not a
    renormalised variational Lyapunov exponent: the separations are finite rather than
    infinitesimal, so the result is only meaningful while the ensemble stays compact.

    Arguments:
        sim_outputs: [list] Nominal per-timestep outputs from reboundSimulate.
        sim_outputs_mc: [dict] {mc_name: per-timestep outputs} from reboundSimulate.

    Return:
        [dict or None] None if there are too few realizations or time samples, otherwise:
            {
                "n_realizations":      [int],
                "growth":              [str] "exponential", "linear" or "undetermined",
                "lyapunov_time_days":  [float or None] 1/lambda if the growth is exponential,
                "lambda_per_day":      [float or None] fitted exponential growth rate,
                "r2_exponential":      [float] coefficient of determination of the ln-linear fit,
                "r2_linear":           [float] coefficient of determination of the linear fit,
                "separation_start_au": [float] initial RMS separation,
                "separation_end_au":   [float] final RMS separation,
                "growth_factor":       [float] end/start separation ratio,
                "n_truncated":         [int] realizations that ended early (e.g. impacted),
                "n_samples_used":      [int] samples entering the fits,
                "saturated":           [bool] the ensemble spread past 10% of its heliocentric
                                           distance, so the linearised picture has broken down,
                "heliocentric_distance_au": [float] mean heliocentric distance over the fit,
            }
        "growth" is "exponential" only when the log-linear fit is good in absolute terms and
        clearly better than the linear one; "saturated" when the ensemble is no longer compact;
        "linear" for regular motion; and "undetermined" when neither law describes the data. A
        realization that ends early contributes only over its own length and does not shorten the
        analysis for the rest of the ensemble.
    """

    if (not sim_outputs) or (len(sim_outputs_mc) < 2):
        return None

    # Fit-acceptance thresholds. These are deliberately strict: an exponential law is only claimed
    # when it clearly describes the data and clearly beats the linear alternative. Loose thresholds
    # will happily report a confident Lyapunov time from two equally poor fits.
    r2_min = 0.90
    r2_margin = 0.05

    # Separation at which the ensemble is no longer a compact cloud around the nominal solution.
    # Beyond this the linearised picture underlying a Lyapunov exponent has broken down (the growth
    # is dominated by along-track phase drift), so no exponent is claimed.
    saturation_fraction = 0.10

    n_nominal = len(sim_outputs)

    # Each realization contributes over its own length: a realization truncated early (for example
    # by an impact) must not shorten the analysis for all the others.
    sq_sum = np.zeros(n_nominal)
    counts = np.zeros(n_nominal, dtype=int)
    n_truncated = 0

    nominal_pos = np.array([row[1][:3] for row in sim_outputs])

    for mc_name in sim_outputs_mc:

        mc_rows = sim_outputs_mc[mc_name]
        n_common = min(n_nominal, len(mc_rows))
        if n_common < len(sim_outputs):
            n_truncated += 1
        if n_common < 2:
            continue

        mc_pos = np.array([mc_rows[i][1][:3] for i in range(n_common)])
        sq_sum[:n_common] += np.sum((mc_pos - nominal_pos[:n_common])**2, axis=1)
        counts[:n_common] += 1

    # Keep the samples where at least two realizations still contribute
    valid = counts >= 2
    if np.count_nonzero(valid) < 4:
        return None

    # Truncate at the first sample that falls below two contributing realizations
    last = int(np.argmax(~valid)) if (~valid).any() else n_nominal
    if last < 4:
        return None

    delta = np.sqrt(sq_sum[:last]/counts[:last])
    t_days = np.array([abs(sim_outputs[i][0] - sim_outputs[0][0])/(2*np.pi)*365.25
                       for i in range(last)])

    # Heliocentric distance of the nominal solution at each sample
    helio_r = np.linalg.norm(nominal_pos[:last], axis=1)
    helio_dist = float(np.mean(helio_r))

    # A Lyapunov exponent only means anything while the ensemble is still a compact cloud around the
    # nominal solution. Rather than rejecting a long integration outright, the fit is restricted to
    # the leading window in which the cloud is still compact - the information is already in the run,
    # so there is no need to repeat it with a shorter span. The separation grows monotonically in
    # practice, so the window ends at the first sample that exceeds the compactness limit.
    compact = delta <= saturation_fraction*np.where(helio_r > 0, helio_r, np.inf)
    if not compact.any():
        i_sat = 0
    elif compact.all():
        i_sat = last
    else:
        i_sat = int(np.argmax(~compact))

    # Whether the fit had to stop short of the end of the integration
    truncated_at_saturation = bool(i_sat < last)

    # Fit over the compact window (always keeping at least the first samples for reporting)
    n_win = max(i_sat, 0)
    if n_win >= 4:
        t_win = t_days[:n_win]
        d_win = delta[:n_win]
    else:
        # The ensemble was never compact for long enough to fit anything
        t_win = t_days
        d_win = delta

    # Drop the first sample (t = 0) and any non-positive separations before taking the logarithm
    mask = (t_win > 0) & (d_win > 0)
    if np.count_nonzero(mask) < 3:
        return None

    t_fit = t_win[mask]
    d_fit = d_win[mask]

    # Exponential growth: ln(delta) linear in t -> slope is the Lyapunov exponent
    fit_exp = scipy.stats.linregress(t_fit, np.log(d_fit))

    # Linear growth: the regular, non-chaotic case
    fit_lin = scipy.stats.linregress(t_fit, d_fit)

    r2_exp = fit_exp.rvalue**2
    r2_lin = fit_lin.rvalue**2

    # Saturated means there was not even a usable compact window to fit
    saturated = bool(n_win < 4)

    # Only call it exponential if the ln-linear fit is genuinely good, clearly beats the linear
    # alternative, and has a positive rate
    growth = "undetermined"
    lyap_time = None
    lam = None
    if saturated:
        growth = "saturated"
    elif (fit_exp.slope > 0) and (r2_exp >= r2_min) and (r2_exp >= r2_lin + r2_margin):
        growth = "exponential"
        lam = fit_exp.slope
        lyap_time = 1.0/lam
    elif r2_lin >= r2_min:
        growth = "linear"

    # Divergence of the semi-major axis, which is the more meaningful chaos indicator. The position
    # separation above saturates within a few orbits purely from Keplerian phase drift (slightly
    # different semi-major axes give slightly different periods, so the cloud smears along the
    # track), whether or not the motion is chaotic. Differences in the orbital elements are immune to
    # that: for regular motion the spread in a stays essentially constant at its initial value, while
    # chaotic motion makes it grow, typically in jumps at close encounters.
    a_nom = np.array([row[2].a for row in sim_outputs[:last]])
    a_sq = np.zeros(last)
    a_counts = np.zeros(last, dtype=int)
    for mc_name in sim_outputs_mc:
        mc_rows = sim_outputs_mc[mc_name]
        n_common = min(last, len(mc_rows))
        if n_common < 2:
            continue
        a_mc = np.array([mc_rows[i][2].a for i in range(n_common)])
        a_sq[:n_common] += (a_mc - a_nom[:n_common])**2
        a_counts[:n_common] += 1

    element_divergence = None
    with np.errstate(divide="ignore", invalid="ignore"):
        a_valid = a_counts >= 2
        if np.count_nonzero(a_valid) >= 4:

            sigma_a = np.sqrt(a_sq[:last][a_valid]/a_counts[:last][a_valid])
            t_a = t_days[a_valid]

            a_mask = (t_a > 0) & (sigma_a > 0)
            if np.count_nonzero(a_mask) >= 3:

                t_af = t_a[a_mask]
                s_af = sigma_a[a_mask]

                fit_a_exp = scipy.stats.linregress(t_af, np.log(s_af))
                fit_a_lin = scipy.stats.linregress(t_af, s_af)
                r2_a_exp = fit_a_exp.rvalue**2
                r2_a_lin = fit_a_lin.rvalue**2

                a_growth = "undetermined"
                a_lyap = None
                a_lam = None
                if (fit_a_exp.slope > 0) and (r2_a_exp >= r2_min) and (r2_a_exp >= r2_a_lin + r2_margin):
                    a_growth = "exponential"
                    a_lam = fit_a_exp.slope
                    a_lyap = 1.0/a_lam
                elif s_af[-1]/s_af[0] < 2.0:
                    # The spread in a barely changed, so the motion is regular in this window
                    a_growth = "regular"
                elif r2_a_lin >= r2_min:
                    a_growth = "linear"

                element_divergence = {
                    "growth": a_growth,
                    "lyapunov_time_days": a_lyap,
                    "lambda_per_day": a_lam,
                    "r2_exponential": r2_a_exp,
                    "r2_linear": r2_a_lin,
                    "sigma_a_start_au": float(s_af[0]),
                    "sigma_a_end_au": float(s_af[-1]),
                    "sigma_a_growth_factor": float(s_af[-1]/s_af[0]),
                    "span_days": float(t_af[-1]),
                }

    return {
        "n_realizations": len(sim_outputs_mc),
        "n_truncated": n_truncated,
        "n_samples_used": int(np.count_nonzero(mask)),
        "growth": growth,
        "saturated": saturated,
        "truncated_at_saturation": truncated_at_saturation,
        "fit_window_days": float(t_fit[-1]),
        "total_span_days": float(t_days[-1]),
        "lyapunov_time_days": lyap_time,
        "lambda_per_day": lam,
        "r2_exponential": r2_exp,
        "r2_linear": r2_lin,
        "separation_start_au": float(d_fit[0]),
        "separation_end_au": float(delta[-1]),
        "separation_fit_end_au": float(d_fit[-1]),
        "growth_factor": float(delta[-1]/d_fit[0]),
        "heliocentric_distance_au": helio_dist,
        "element_divergence": element_divergence,
    }


def radiationPressureBeta(radius_m, density_kgm3, q_pr=1.0):
    """ Compute beta, the ratio of the solar radiation pressure force to solar gravity, for a
    spherical grain.

        beta = 3*L_sun*Q_pr/(16*pi*G*M_sun*c*rho*s) = 5.7425e-4*Q_pr/(rho*s)

    with rho in kg/m^3 and s in m. As a sanity check, a 1 micron grain of density 3000 kg/m^3 gives
    beta = 0.19, while a 1 cm meteoroid of the same density gives 1.9e-5, i.e. radiation pressure is
    negligible for fireball-producing bodies but important for small grains.

    Arguments:
        radius_m: [float] Grain radius in metres.
        density_kgm3: [float] Bulk density in kg/m^3.

    Keyword arguments:
        q_pr: [float] Radiation-pressure efficiency factor, ~1 for grains large compared to the
            wavelength of sunlight. Default 1.0.

    Return:
        [float] The dimensionless beta parameter.
    """

    if (radius_m <= 0) or (density_kgm3 <= 0):
        raise ValueError("The radius and the density must both be positive.")

    return 5.7425e-4*q_pr/(density_kgm3*radius_m)


def tisserandParameterJupiter(a, e, inc):
    """ Compute the Tisserand parameter with respect to Jupiter, a quasi-invariant of the encounter
    geometry that is the standard way to classify the dynamical origin of a small body.

    Conventional interpretation:
        T_J > 3        asteroidal orbit (decoupled from Jupiter),
        2 < T_J < 3    Jupiter-family-comet-like orbit,
        T_J < 2        Halley-type / long-period comet-like orbit.

    Arguments:
        a: [float] Semi-major axis in AU (must be positive, i.e. a bound heliocentric orbit).
        e: [float] Eccentricity.
        inc: [float] Inclination in radians (measured from the ecliptic, which is used here as an
            approximation to Jupiter's orbital plane; the two differ by ~1.3 degrees).

    Return:
        [float or None] The Tisserand parameter, or None if the orbit is unbound or degenerate so
            that the parameter is not defined.
    """

    a_jupiter = 5.204267  # AU

    if (a is None) or (a <= 0) or (e is None) or (e >= 1):
        return None

    return a_jupiter/a + 2.0*np.cos(inc)*np.sqrt((a/a_jupiter)*(1.0 - e**2))


def tisserandClass(t_j):
    """ Return the conventional dynamical class implied by a Tisserand parameter (see
    tisserandParameterJupiter).

    Arguments:
        t_j: [float or None] Tisserand parameter with respect to Jupiter.

    Return:
        [str] A short description of the dynamical class.
    """

    if t_j is None:
        return "undefined (unbound orbit)"

    if t_j > 3.0:
        return "asteroidal (T_J > 3)"

    if t_j > 2.0:
        return "Jupiter-family-comet-like (2 < T_J < 3)"

    return "Halley-type/long-period-comet-like (T_J < 2)"


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
        _printReboundUnavailable()
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
        _addNamedParticle(sim, "Earth", "Geocenter", date=f"JD{jd:.6f}")
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
            planet_names:    [list] Massive-body names, in add order.
            planet_states:   [list] Per-planet [x, y, z, vx, vy, vz] in REBOUND units (AU,
                                 AU/(year/2pi)), barycentric ecliptic J2000.
            planet_masses:   [list] Per-planet mass in solar masses.
            particle_states: [list] Per-particle barycentric [x, y, z, vx, vy, vz] (REBOUND units).
            particle_names:  [list] Per-particle names (parallel to particle_states).
            times:           [list] Output times (REBOUND time units) to integrate to.
            direction:       [str]  "forward" or "backward" (sets the timestep sign).
            reference_frame: [str]  "heliocentric" or "geocentric" (passed to extractSimParams).
            n_hill:          [float] Optional. Multiple of the Earth's Hill radius the object must
                                 exceed before encounter/impact detection is armed. Default 3.0.
            body_radii:      [dict] Optional. {body: radius in AU} overriding the default physical
                                 radii used for impact detection (Earth 6371 km, Moon 1737.4 km).

    Return:
        [dict] {
            "outputs": {particle_name: [[time, state_vect_hel, orb_ns, planet_dists], ...]}, where
                orb_ns is a picklable SimpleNamespace with attributes a, e, inc, Omega, omega, f,
            "diagnostics": {particle_name: {
                "min_dist_au":   {body: closest approach in AU (None if never tracked)},
                "min_time_days": {body: time of closest approach in days},
                "encounters":    [list] every close encounter, in the order it happened: one dict
                                 per local minimum of an object-body distance inside n_hill Hill
                                 radii, in the format of encountersFromMinDistances,
                "departed":      [bool] whether the object left the Earth's neighbourhood,
                "impact":        None, or {"body", "time_days", "dist_au"} if the object hit a body,
            }},
        }
        The closest approaches are tracked at every internal integrator timestep (via a heartbeat
        callback), not at the output samples, and refined between steps (see
        hermiteClosestApproach), so they are not limited by the output or the step sampling.
        "min_dist_au" keeps only the deepest approach to each body, while "encounters" keeps each
        passage, so repeated encounters with the same body are all reported.
    """

    planet_names = task["planet_names"]
    planet_states = task["planet_states"]
    planet_masses = task["planet_masses"]
    particle_states = task["particle_states"]
    particle_names = task["particle_names"]
    times = task["times"]
    direction = task["direction"]
    reference_frame = task["reference_frame"]

    n_hill = task.get("n_hill", 3.0)

    aum = rb.units.lengths_SI["au"]  # 1 au in m
    aukm = aum/1e3  # au in km

    # Constants (gravitational harmonics of Earth, Earth radius)
    RE_eq = 6378.135/aukm
    J2 = 1.0826157e-3
    J4 = -1.620e-6

    # Physical body radii used for impact (collision) detection, in AU. The task can override them,
    # e.g. to use an atmospheric-entry cross-section instead of the solid-body radius.
    body_radii = {"Earth": 6371.0/aukm, "Luna": 1737.4/aukm}
    body_radii.update(task.get("body_radii", {}))

    # The object starts at the Earth, so encounter and impact detection for every body except the
    # Moon is only armed once the object has left the Earth's neighbourhood (see the module notes
    # on findEarthDepartureIndex). A lunar encounter can only ever happen while the object is deep
    # inside this radius (the Moon's apogee plus a few of its Hill radii is ~0.004 AU, far below
    # it), so the Moon is tracked over the whole integration.
    departure_radius = n_hill*HILL_RADII_AU["Earth"]

    # Set up the simulation
    sim = rb.Simulation()
    rebx = reboundx.Extras(sim)
    _checkReboundxAttached(sim)
    sim.dt = 0.001 if direction == "forward" else -0.001

    # Add the massive bodies from the precomputed barycentric states
    for name, state, mass in zip(planet_names, planet_states, planet_masses):
        _addNamedParticle(
            sim, name,
            m=mass,
            x=state[0], y=state[1], z=state[2],
            vx=state[3], vy=state[4], vz=state[5],
        )

    # Add the test particles (massless)
    for name, state in zip(particle_names, particle_states):
        _addNamedParticle(
            sim, name,
            x=state[0], y=state[1], z=state[2],
            vx=state[3], vy=state[4], vz=state[5],
        )

    ps = sim.particles

    # Add the gravitational harmonics of the Earth
    gh = rebx.load_force("gravitational_harmonics")
    rebx.add_force(gh)
    ps["Earth"].params["J2"] = J2
    ps["Earth"].params["J4"] = J4
    ps["Earth"].params["R_eq"] = RE_eq
    # Assign the body radii used for impact detection
    for bname, brad in body_radii.items():
        if bname in planet_names:
            ps[bname].r = brad

    # gr_full is the general relativity correction for all bodies
    gr = rebx.load_force("gr_full")
    rebx.add_force(gr)
    gr.params["c"] = rbxConstants.C

    # Optional solar radiation pressure and Poynting-Robertson drag on the integrated particles.
    # Off unless a beta is supplied, because it needs the object's size and density, which are not
    # part of a trajectory solution. REBOUNDx's radiation_forces includes both effects.
    beta = task.get("beta")
    if beta:
        rad = rebx.load_force("radiation_forces")
        rebx.add_force(rad)
        rad.params["c"] = rbxConstants.C
        ps["Sun"].params["radiation_source"] = 1
        for name in particle_names:
            ps[name].params["beta"] = beta

    # Move to the center of momentum frame before integrating
    sim.move_to_com()

    # Collision detection starts disabled (the object starts at the Earth's surface, which would
    # register as an immediate impact) and is armed once the object has departed. "line" mode checks
    # for overlap along the line of travel between timesteps, which is required for fast movers.
    sim.collision = "none"
    sim.collision_resolve = "halt"
    sim.N_active = len(planet_names)
    sim.testparticle_type = 0

    # Stop following a particle that runs away, instead of integrating it forever at growing cost.
    # The bound is well outside Neptune, so it can only be triggered by the integrated object.
    max_dist_au = task.get("max_dist_au", 1000.0)
    sim.exit_max_distance = max_dist_au

    # Total energy at the start, used to report the integrator's energy conservation
    energy_start = sim.energy()

    # Indices of the bodies and the test particles in the simulation
    n_planets = len(planet_names)
    earth_i = planet_names.index("Earth")
    particle_idx = {name: n_planets + k for k, name in enumerate(particle_names)}

    # Per-particle closest-approach tracking, updated at every internal timestep by the heartbeat
    # callback. The output samples are far too coarse near the Earth and the Moon (the object can
    # cross the Moon's whole detection sphere between two samples). The internal steps are not
    # enough on their own either: away from the Earth IAS15 takes steps of 1-2 days, so the object
    # can move millions of km per step past a planet, and the step-end samples can overestimate the
    # minimum by a large fraction of that. The closest approach inside each step is therefore
    # refined on a Hermite interpolant of the relative motion (see hermiteClosestApproach). "prev"
    # holds the relative state of the previous step for each body, and "encounters" collects every
    # refined local minimum that falls inside n_hill Hill radii, as (body, time, distance).
    track = {}
    for name in particle_names:
        track[name] = {
            "min_dist": {b: np.inf for b in planet_names},
            "min_time": {b: np.nan for b in planet_names},
            "departed": False,
            "impact": None,
            "escaped": None,
            "prev": {},
            "encounters": [],
        }

    def heartbeat(sim_pointer):

        s = sim_pointer.contents
        p = s.particles
        t_now = s.t

        for pname, pidx in particle_idx.items():

            # Skip particles that are no longer in the simulation
            if pidx >= s.N:
                continue

            o = p[pidx]
            st = track[pname]

            # Distance to the Earth, used to decide whether the object has left its neighbourhood
            pe = p[earth_i]
            d_earth = ((o.x - pe.x)**2 + (o.y - pe.y)**2 + (o.z - pe.z)**2)**0.5
            if (not st["departed"]) and (d_earth > departure_radius):
                st["departed"] = True

            for bi, bname in enumerate(planet_names):

                # The Moon is tracked over the whole integration; every other body only after the
                # object has left the Earth's neighbourhood, so the trivial initial departure from
                # the Earth is not reported as an encounter.
                if (bname != "Luna") and (not st["departed"]):
                    continue

                # Relative state as plain floats: this runs every step for every body, and building
                # numpy arrays here would slow the whole integration down by ~10%
                b = p[bi]
                rel = (o.x - b.x, o.y - b.y, o.z - b.z, o.vx - b.vx, o.vy - b.vy, o.vz - b.vz)
                d = (rel[0]**2 + rel[1]**2 + rel[2]**2)**0.5
                rv = rel[0]*rel[3] + rel[1]*rel[4] + rel[2]*rel[5]

                if d < st["min_dist"][bname]:
                    st["min_dist"][bname] = d
                    st["min_time"][bname] = t_now

                # The distance has a minimum inside the last step if the approach rate r.v changed
                # sign from approaching to receding (h makes this hold in both time directions).
                # Refine it on the Hermite interpolant of the step.
                prev = st["prev"].get(bname)
                if prev is not None:
                    t_prev, rel_prev, rv_prev = prev
                    h = t_now - t_prev
                    if (h*rv_prev < 0) and (h*rv > 0):
                        rp, rn = np.array(rel_prev), np.array(rel)
                        d_h, t_h = hermiteClosestApproach(t_prev, rp[:3], rp[3:], t_now, rn[:3], rn[3:])
                        if d_h < st["min_dist"][bname]:
                            st["min_dist"][bname] = d_h
                            st["min_time"][bname] = t_h

                        # Every such passage inside the Hill-sphere threshold is an encounter
                        if d_h < n_hill*HILL_RADII_AU.get(bname, 0.0):
                            st["encounters"].append((bname, t_h, d_h))

                st["prev"][bname] = (t_now, rel, rv)

    # Keep the returned reference alive until the integration ends (see _setHeartbeat)
    heartbeat_ref = _setHeartbeat(sim, heartbeat)  # noqa: F841

    outputs = {name: [] for name in particle_names}

    # Integrate the simulation and save the state vectors and orbital elements. These are not time
    # steps, but the times at which the simulation state is saved.
    for t_out in times:

        sim.move_to_com()

        # Arm impact detection once every remaining particle has left the Earth's neighbourhood
        if sim.collision == "none":
            active = [n for n, i in particle_idx.items() if i < sim.N]
            if active and all(track[n]["departed"] for n in active):
                sim.collision = "line"

        try:
            sim.integrate(t_out)

        except rb.Collision:

            # Identify which body was hit (the test particles have no radius of their own)
            t_impact = sim.t/(2*np.pi)*365.25
            hit_particle = False
            for pname, pidx in particle_idx.items():
                if pidx >= sim.N:
                    continue
                o = sim.particles[pidx]
                for bi, bname in enumerate(planet_names):
                    b = sim.particles[bi]
                    d = ((o.x - b.x)**2 + (o.y - b.y)**2 + (o.z - b.z)**2)**0.5
                    if d <= (b.r + o.r):
                        track[pname]["impact"] = {"body": bname, "time_days": t_impact,
                                                  "dist_au": d}
                        hit_particle = True

            if hit_particle:
                # The object hit a body, so the integration is physically over
                break

            # The collision did not involve any of the integrated particles, which means two massive
            # bodies overlap (only possible with unphysically large body_radii). Disable collision
            # detection and finish this step rather than silently truncating the integration.
            warnings.warn(
                "REBOUND reported a collision between two massive bodies at t = {:.4f} d; "
                "impact detection disabled for the rest of this integration. Check body_radii.".format(
                    t_impact), RuntimeWarning)
            sim.collision = "none"
            sim.integrate(t_out)

        except rb.Escape:

            # The object was thrown out of the simulation volume (unbound or ejected). Record it and
            # stop, since integrating a runaway particle only gets more expensive.
            t_esc = sim.t/(2*np.pi)*365.25
            for pname, pidx in particle_idx.items():
                if pidx >= sim.N:
                    continue
                o = sim.particles[pidx]
                d_sun = (o.x**2 + o.y**2 + o.z**2)**0.5
                if d_sun > max_dist_au:
                    track[pname]["escaped"] = {"time_days": t_esc, "dist_au": d_sun}
            break

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

            outputs[name].append([t_out, state_vect_hel, orb_ns, planet_dists])

    # Relative energy drift over the integration, as an integrator-quality diagnostic. The test
    # particles are massless, so this measures the massive subsystem the object moves through.
    # The energy must be evaluated in the same frame as at the start (the loop leaves the simulation
    # in the heliocentric frame, and kinetic energy is frame-dependent).
    sim.move_to_com()
    energy_end = sim.energy()
    if energy_start != 0:
        energy_drift = abs((energy_end - energy_start)/energy_start)
    else:
        energy_drift = None

    # Assemble the picklable per-particle diagnostics (times converted to days)
    diagnostics = {}
    for name in particle_names:
        st = track[name]
        diagnostics[name] = {
            "min_dist_au": {b: (None if not np.isfinite(st["min_dist"][b]) else st["min_dist"][b])
                            for b in planet_names},
            "min_time_days": {b: (None if not np.isfinite(st["min_time"][b])
                                  else st["min_time"][b]/(2*np.pi)*365.25)
                              for b in planet_names},
            "encounters": [_encounterRecord(b, d, t/(2*np.pi)*365.25) for b, t, d in st["encounters"]],
            "departed": st["departed"],
            "impact": st["impact"],
            "escaped": st["escaped"],
            "energy_rel_drift": energy_drift,
        }

    return {"outputs": outputs, "diagnostics": diagnostics}


def reboundSimulate(
        julian_date, state_vect, traj=None,
        direction="forward", sim_days=60, n_outputs=500, obj_name="obj", obj_mass=0.0, mc_runs=100,
        reference_frame="heliocentric", ephem_source="local", n_cpu=None,
        show_progress=True, return_diagnostics=False, random_seed=None, beta=None, verbose=False):
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
        obj_mass: [float] Accepted for backwards compatibility but ignored: the integrated object is
            always treated as a massless test particle, so that it cannot perturb the planets and the
            Monte Carlo realizations cannot perturb each other. Give the object a mass only by
            editing _integrateParticles, and note that N_active would have to change with it.
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
        show_progress: [bool] If True (default), report Monte Carlo integration progress.
        return_diagnostics: [bool] If True, also return the per-particle diagnostics dictionary
            (closest approaches and impacts measured during the integration). Default False,
            which preserves the two-value return signature.
        random_seed: [int] Seed for the Monte Carlo state-vector sampling. Pass a value to make a
            run exactly reproducible; None (default) draws a fresh, unpredictable seed. The sampling
            is done in the parent process, so results do not depend on the number of cores used.
        beta: [float] Ratio of solar radiation pressure to solar gravity for the integrated object.
            If given, radiation pressure and Poynting-Robertson drag are applied to the object (see
            radiationPressureBeta to compute it from a size and density). None (default) leaves the
            integration purely gravitational, since a trajectory solution does not constrain the
            object's size and density.
        verbose: [bool] If True, print out the progress of the simulation.

    Return:
        outputs: [list] List of outputs, each containing the time, state vector and orbital elements at that
            time.
        outputs_mc: [dict] {mc_name: outputs} for the Monte Carlo realizations.
        diagnostics: [dict] Only returned if return_diagnostics is True. Maps each particle name to
            its closest approaches ("min_dist_au", "min_time_days"), every close encounter
            ("encounters"), whether it left the Earth's neighbourhood ("departed"), and any impact
            ("impact"). See _integrateParticles.

    """

    # Skip if REBOUND is not found
    if not REBOUND_FOUND:
        _printReboundUnavailable()
        return None

    # Fail early, before any ephemeris work, if REBOUNDx cannot attach to a simulation
    checkReboundxUsable()

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

        # Draw the realizations from a seeded generator so a run can be reproduced exactly. The
        # sampling happens here in the parent process, so the results do not depend on how many
        # cores the integration is spread over.
        rng = np.random.default_rng(random_seed)

        # Sample the state vector from the uncertainties
        for i in range(mc_runs):

            # Sample the state vector from the uncertainties
            sv_realization = rng.multivariate_normal(state_vect, cov)

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
            _addNamedParticle(parent_sim, name, horizons_names.get(name, name),
                              date=f"JD{time_tdb:.6f}")

    else:

        # Seed the planets from the local DE430 ephemeris (fast, offline)
        jpl_ephem_data = SPK.open(config.jpl_ephem_file)
        for name in planet_names:
            body_state, body_mass = ephemBodyStateRebound(name, time_tdb, jpl_ephem_data)
            _addNamedParticle(
                parent_sim, name,
                m=body_mass,
                x=body_state[0], y=body_state[1], z=body_state[2],
                vx=body_state[3], vy=body_state[4], vz=body_state[5],
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
            "beta": beta,
        }

    if verbose:
        print("Running simulation...")
        print(f"Simulation time: {tsimend:.2f} years ({tsimend*365.25:.2f} days)")
        print(f"Number of outputs: {n_outputs}")
        print(f"Direction: {direction}")

    # Nominal solution, integrated in its own simulation (in the main process)
    nominal_result = _integrateParticles(_make_task([state_vect_rot], [obj_name]))
    outputs = nominal_result["outputs"][obj_name]
    diagnostics = dict(nominal_result["diagnostics"])

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
                outputs_mc.update(res["outputs"])
                diagnostics.update(res["diagnostics"])

    if return_diagnostics:
        return outputs, outputs_mc, diagnostics

    return outputs, outputs_mc


if __name__ == "__main__":

    import os
    import argparse

    from wmpl.Utils.Pickling import loadPickle


    # Exit cleanly with a helpful message if REBOUND/REBOUNDx are not installed, instead of
    # crashing later with a cryptic "cannot unpack non-iterable NoneType" error.
    if not REBOUND_FOUND:
        print("")
        _printReboundUnavailable(include_install_help=True)
        sys.exit(1)

    ###

    parser = argparse.ArgumentParser(description="Run REBOUND simulation for a given trajectory pickle file. The simulation is run 60 days backwards by default.")

    parser.add_argument("pickle_path", type=str, help="Path to the pickle file with the trajectory data.")

    parser.add_argument("--days", type=float, help="Run the simulation for the given number of days.", default=60)

    parser.add_argument("--forward", type=float, nargs="?", const=0.0, default=None,
                        help="Run the simulation forward in time. Optionally give the number of days "
                        "(e.g. --forward 100); if no value is given, --days is used.")

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

    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for the Monte Carlo sampling, making a run exactly "
                        "reproducible. If not given, a seed is drawn and reported.")

    parser.add_argument("--outputs", type=int, default=500,
                        help="Number of times along the integration at which the state is saved. "
                        "Increase it for long integrations, where the default 500 samples are "
                        "coarse. Default: 500.")

    parser.add_argument("--beta", type=float, default=None,
                        help="Include solar radiation pressure and Poynting-Robertson drag with this "
                        "beta (the ratio of radiation pressure to solar gravity). Mutually exclusive "
                        "with --radius/--density, which compute beta instead.")

    parser.add_argument("--radius", type=float, default=None,
                        help="Object radius in metres, used with --density to compute beta and "
                        "include radiation forces. Purely gravitational if not given.")

    parser.add_argument("--density", type=float, default=3000.0,
                        help="Object bulk density in kg/m^3, used with --radius to compute beta. "
                        "Default: 3000.")

    parser.add_argument("--verbose", action="store_true", help="Print out the progress of the simulation.")

    args = parser.parse_args()

    # Extract the number of days from the arguments and the simulation direction. --forward may be
    # given on its own (use --days) or with its own number of days.
    if args.forward is None:
        direction = "backward"
        sim_days = args.days
    else:
        direction = "forward"
        sim_days = args.forward if args.forward > 0 else args.days

    # Seed for the Monte Carlo sampling. If none was given, draw one and report it, so that any run
    # can be reproduced afterwards with --seed.
    random_seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy % (2**32))

    ### Non-gravitational forces (off by default) ###
    if (args.beta is not None) and (args.radius is not None):
        parser.error("Give either --beta or --radius (with --density), not both.")

    beta = args.beta
    if args.radius is not None:
        beta = radiationPressureBeta(args.radius, args.density)
        print("Radiation forces ON: radius {:.4g} m, density {:.0f} kg/m^3 -> beta = {:.4e}".format(
            args.radius, args.density, beta))
    elif beta is not None:
        print("Radiation forces ON: beta = {:.4e}".format(beta))
    ### ###

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

    
    # Set semi-major axis and periapsis units depending on the reference frame. Note that this
    # multiplier applies only to the orbital elements: the object-body distances, the close-encounter
    # distances and the divergence figures are always reported in AU (with km given alongside where
    # it helps), because they are distances between bodies rather than orbit sizes and so do not
    # depend on the frame the elements are expressed in.
    a_units = "AU"
    q_units = "AU"
    dist_unit_multiplier = 1.0  # Default is AU
    if reference_frame == "geocentric":
        a_units = "km"
        q_units = "km"
        dist_unit_multiplier = 149597870.7  # Convert AU to km

    ### ###

    
    # State the integration span up front, so it is visible while the integration is running and
    # not only in the summary printed at the end
    print("Integrating {:.2f} days {:s} from the reference epoch {:.6f} JD (TDB) = {:s} UTC.".format(
        sim_days, direction, traj.jdt_ref,
        astropy.time.Time(traj.jdt_ref, format='jd', scale='utc').iso))

    # Run the simulation for the given number of days from the epoch of the trajectory
    t_run_start = time.time()
    sim_outputs, sim_outputs_mc, sim_diagnostics = reboundSimulate(
        None, None, traj=traj, direction=direction, sim_days=sim_days,
        obj_name=traj.traj_id, mc_runs=args.mc, n_outputs=args.outputs,
        reference_frame=reference_frame,
        ephem_source=ephem_source, n_cpu=n_cpu, return_diagnostics=True,
        random_seed=random_seed, beta=beta, verbose=args.verbose
        )
    sim_wall = time.time() - t_run_start

    ### Work out what happened to each Monte Carlo clone ###

    n_hill = 3.0

    # Diagnostics for the clones only (the nominal solution is keyed by the trajectory ID)
    clone_diag = {name: diag for name, diag in sim_diagnostics.items() if name in sim_outputs_mc}

    clones_impacted = {}
    clones_escaped = []
    clones_survived = []
    for name in sim_outputs_mc:
        diag = clone_diag.get(name, {})
        if diag.get("impact"):
            clones_impacted.setdefault(diag["impact"]["body"], []).append(name)
        elif diag.get("escaped"):
            clones_escaped.append(name)
        else:
            clones_survived.append(name)

    # How many clones had a close encounter with each body, how many encounters they had in total
    # (a clone can meet the same body more than once), and the closest approach over all clones
    clone_encounters = cloneEncounterSummary(clone_diag)

    # Clones truncated by an impact or an ejection end at a different epoch than the rest, so mixing
    # their final elements into the confidence interval would blend different times. Use the clones
    # that completed the full span, unless too few of them survived to say anything.
    ci_names = clones_survived if len(clones_survived) >= 2 else list(sim_outputs_mc)
    ci_uses_survivors_only = (len(clones_survived) >= 2) and (len(clones_survived) < len(sim_outputs_mc))

    ### ###

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

        for mc_name in ci_names:
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

    # Detect close encounters (Hill-sphere criterion) from the closest approaches tracked at every
    # internal integrator timestep and refined between steps, which resolves the fast Earth/Moon
    # regime that the sampled output cannot. Needed both for the summary and the report file. n_hill
    # is set above, where the clone outcomes are classified with the same threshold.
    # Every encounter is listed, including repeated ones with the same body, in the order they
    # happened during the integration (for a backward integration, the latest epoch first).
    nominal_diag = sim_diagnostics.get(traj.traj_id, {})
    encounters = sorted(nominal_diag.get("encounters", []), key=lambda e: abs(e["time_days"]))

    # Impact on a body, if any (detected with REBOUND's collision detection)
    impact = nominal_diag.get("impact")

    # Estimate the divergence/Lyapunov timescale from the Monte Carlo ensemble (free when MC is run)
    lyap = estimateLyapunovFromMC(sim_outputs, sim_outputs_mc)

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

    # Always state how far the integration actually went, and flag it if the integration stopped
    # short of the request (for example because the object impacted a body)
    achieved_days = abs(final_sim_days)
    if abs(achieved_days - sim_days) > 1e-6:
        print("  Integration  : {:.2f} days {:s}  (requested {:.2f} days, stopped early)".format(
            achieved_days, direction, sim_days))
    else:
        print("  Integration  : {:.2f} days {:s}".format(achieved_days, direction))

    print("  Frame        : {:s}".format(reference_frame))
    print("  Forces       : {:s}".format(
        "gravity + GR + Earth J2/J4" if beta is None
        else "gravity + GR + Earth J2/J4 + radiation (beta = {:.3e})".format(beta)))
    print("  Start epoch  : {:.6f} JD (TDB)".format(traj.jdt_ref))
    print("  Final epoch  : {:.6f} JD (TDB)  =  {:s} UTC".format(final_epoch_jd, time_utc.iso))
    if len(sim_outputs_mc):
        print("  Monte Carlo  : {:d} realizations on {:d} core(s), seed {:d}".format(
            len(sim_outputs_mc), n_cpu, random_seed))
    print("  Runtime      : {:.1f} s".format(sim_wall))

    # Integrator quality: relative energy drift of the massive subsystem
    energy_drift = nominal_diag.get("energy_rel_drift")
    if energy_drift is not None:
        print("  Energy drift : {:.2e} (relative)".format(energy_drift))
    print("-" * 78)

    # If the object hit something, say so before anything else: the elements below are the state at
    # the last step before the impact, not a surviving orbit.
    if impact:
        print("  *** IMPACT: the object hit {:s} after {:.4f} days ***".format(
            impact["body"], abs(impact["time_days"])))
        print("  The elements below are the last state before the impact, not a surviving orbit.")
        print("-" * 78)

    if impact:
        header = "  Orbital elements at the moment of impact"
    else:
        header = "  Final orbital elements"

    if len(sim_outputs_mc):
        print(header + "   (nominal  +/- 1 sigma  [95% CI]):")
    else:
        print(header + "   (nominal):")
    print("")
    print("    a    = {:>13.6f}{:s}  {:s}".format(a_val, a_ci_str, a_units))
    print("    q    = {:>13.6f}{:s}  {:s}".format(q_val, q_ci_str, q_units))
    print("    e    = {:>13.6f}{:s}".format(e_val, e_ci_str))
    print("    i    = {:>13.6f}{:s}  deg".format(i_val, incl_ci_str))
    print("    peri = {:>13.6f}{:s}  deg".format(peri_val, omega_ci_str))
    print("    node = {:>13.6f}{:s}  deg".format(node_val, Omega_ci_str))
    print("    f    = {:>13.6f}{:s}  deg".format(f_val, f_ci_str))

    # Tisserand parameter with respect to Jupiter and the dynamical class it implies. Only
    # meaningful for a heliocentric orbit.
    if reference_frame == "heliocentric":
        t_j = tisserandParameterJupiter(sim_outputs[-1][2].a, e_val, sim_outputs[-1][2].inc)
        if t_j is None:
            print("    T_J  =        undefined  ({:s})".format(tisserandClass(t_j)))
        else:
            print("    T_J  = {:>13.6f}       {:s}".format(t_j, tisserandClass(t_j)))

    print("-" * 78)

    # Close-encounter summary (minima tracked every integrator timestep, refined between steps)
    if encounters:
        print("  Close encounters (< {:.0f} Hill radii), in the order they happened:".format(n_hill))
        for enc in encounters:
            print("    {:<8s} {:12.6f} AU ({:12.1f} km)  at t = {:+9.3f} d   ({:.2f} R_Hill, R_Hill = {:.6f} AU)".format(
                enc["body"], enc["min_dist_au"], enc["min_dist_au"]*149597870.7,
                enc["time_days"], enc["n_hill"], enc["hill_radius_au"]))
    else:
        print("  Close encounters (< {:.0f} Hill radii): none detected".format(n_hill))

    # Ejection, if the object ran out of the simulation volume
    escaped = nominal_diag.get("escaped")
    if escaped:
        print("  *** EJECTED: the object left the simulation volume after {:.4f} days "
              "({:.1f} AU from the Sun) ***".format(abs(escaped["time_days"]), escaped["dist_au"]))

    # What happened to the Monte Carlo clones, as a fraction of the whole ensemble
    if len(sim_outputs_mc):

        n_clones = len(sim_outputs_mc)
        print("-" * 78)
        print("  Monte Carlo clone outcomes ({:d} clones):".format(n_clones))

        print("    {:<28s} {:5d}  ({:5.1f}%)".format(
            "Completed the full span", len(clones_survived), 100.0*len(clones_survived)/n_clones))

        for body in sorted(clones_impacted, key=lambda b: -len(clones_impacted[b])):
            n_hit = len(clones_impacted[body])
            print("    {:<28s} {:5d}  ({:5.1f}%)".format(
                "IMPACTED " + body, n_hit, 100.0*n_hit/n_clones))

        if clones_escaped:
            print("    {:<28s} {:5d}  ({:5.1f}%)".format(
                "Left the simulation volume", len(clones_escaped),
                100.0*len(clones_escaped)/n_clones))

        # Close encounters across the ensemble, which is what the clones are really there to measure
        if clone_encounters:
            print("")
            print("  Clones with a close encounter (< {:.0f} Hill radii):".format(n_hill))
            for body in sorted(clone_encounters, key=lambda b: -clone_encounters[b]["count"]):
                entry = clone_encounters[body]
                print("    {:<10s} {:5d}/{:<5d} ({:5.1f}%)   {:5d} encounter(s)   closest over all "
                      "clones: {:.6f} AU ({:.0f} km)".format(
                          body, entry["count"], n_clones, 100.0*entry["count"]/n_clones,
                          entry["n_encounters"], entry["closest_au"],
                          entry["closest_au"]*149597870.7))
        else:
            print("")
            print("  No clone came within {:.0f} Hill radii of any body.".format(n_hill))

        if ci_uses_survivors_only:
            print("")
            print("  Note: the confidence intervals above use the {:d} clones that completed the "
                  "full span;".format(len(clones_survived)))
            print("  clones that impacted or were ejected ended at a different epoch and are "
                  "excluded.")

    # Divergence / Lyapunov timescale estimated from the Monte Carlo spread
    if lyap is not None:
        print("-" * 78)
        print("  Trajectory divergence (from {:d} Monte Carlo realizations):".format(
            lyap["n_realizations"]))
        print("    Separation from nominal: {:.3e} -> {:.3e} AU  (x{:.1f})".format(
            lyap["separation_start_au"], lyap["separation_end_au"], lyap["growth_factor"]))
        if lyap["n_truncated"]:
            print("    {:d} realization(s) ended early and contributed only over their own span.".format(
                lyap["n_truncated"]))
        if lyap["truncated_at_saturation"] and not lyap["saturated"]:
            print("    Fitted over the first {:.0f} d of {:.0f} d, while the ensemble was still "
                  "compact".format(lyap["fit_window_days"], lyap["total_span_days"]))
            print("    (separation {:.3e} AU at the end of that window, {:d} samples).".format(
                lyap["separation_fit_end_au"], lyap["n_samples_used"]))
        if lyap["growth"] == "exponential":
            print("    Growth is exponential: Lyapunov time ~ {:.1f} d  "
                  "(lambda = {:.4f} /d, R2 = {:.3f})".format(
                      lyap["lyapunov_time_days"], lyap["lambda_per_day"], lyap["r2_exponential"]))
            print("    The integration loses predictive value on timescales beyond this.")
        elif lyap["growth"] == "linear":
            print("    Growth is linear (regular motion, R2 = {:.3f}); no exponential divergence "
                  "detected".format(lyap["r2_linear"]))
            print("    over this interval, so no Lyapunov time can be derived from it.")
        elif lyap["growth"] == "saturated":
            print("    The ensemble was already spread over {:.1f}% of its heliocentric distance "
                  "within the".format(100*lyap["separation_end_au"]/lyap["heliocentric_distance_au"]))
            print("    first few samples, leaving no compact window to fit. Use a denser output "
                  "sampling (--outputs)")
            print("    or a shorter integration to resolve the early divergence.")
        else:
            print("    Growth fits neither a clean exponential nor a linear law "
                  "(R2_exp = {:.3f}, R2_lin = {:.3f}),".format(
                      lyap["r2_exponential"], lyap["r2_linear"]))
            print("    so no divergence timescale is claimed.")

        # Semi-major-axis spread: the chaos indicator that is not corrupted by phase drift
        ed = lyap.get("element_divergence")
        if ed is not None:
            print("    Spread in a over {:.0f} d: {:.3e} -> {:.3e} AU (x{:.2f})".format(
                ed["span_days"], ed["sigma_a_start_au"], ed["sigma_a_end_au"],
                ed["sigma_a_growth_factor"]))
            if ed["growth"] == "exponential":
                print("    The spread in a grows exponentially: Lyapunov time ~ {:.1f} d "
                      "(R2 = {:.3f}).".format(ed["lyapunov_time_days"], ed["r2_exponential"]))
                print("    This is the chaos estimate to trust; unlike the position separation it is")
                print("    not corrupted by Keplerian phase drift.")
            elif ed["growth"] == "regular":
                print("    The spread in a is essentially unchanged, so the orbit is regular over "
                      "this span:")
                print("    the position divergence above is phase drift, not chaos.")
            else:
                print("    The spread in a grows but not exponentially "
                      "(R2_exp = {:.3f}, R2_lin = {:.3f}).".format(
                          ed["r2_exponential"], ed["r2_linear"]))
    print(hdr)


    # Save the results to a file
    out_dir = os.path.dirname(args.pickle_path)
    results_txt_path = os.path.join(out_dir, "rebound_simulation_results.txt")
    plot_png_path = os.path.join(out_dir, "rebound_simulation.png")
    with open(results_txt_path, "w") as f:

        # Save the nominal orbital elements and the errors
        # If the object hit something, state it at the very top of the report
        if impact:
            f.write("*** IMPACT: the object hit {:s} after {:.4f} days. The elements below are the "
                    "last\n".format(impact["body"], abs(impact["time_days"])))
            f.write("*** state before the impact, not a surviving orbit.\n\n")

        f.write("Orbital elements {:.2f} days {:s} from the epoch {:.6f} JD (TDB){:s}\n".format(
            achieved_days, direction, traj.jdt_ref,
            "" if abs(achieved_days - sim_days) <= 1e-6
            else " (requested {:.2f} days, stopped early)".format(sim_days)))
        f.write("a    = {:>10.6f}{:s} {:s}\n".format(sim_outputs[-1][2].a*dist_unit_multiplier, a_ci_str, a_units))
        f.write("q    = {:>10.6f}{:s} {:s}\n".format((1 - sim_outputs[-1][2].e)*sim_outputs[-1][2].a*dist_unit_multiplier, q_ci_str, q_units))
        f.write("e    = {:>10.6f}{:s}\n".format(sim_outputs[-1][2].e, e_ci_str))
        f.write("i    = {:>10.6f}{:s} deg\n".format(np.degrees(sim_outputs[-1][2].inc), incl_ci_str))
        f.write("peri = {:>10.6f}{:s} deg\n".format(np.degrees(sim_outputs[-1][2].omega), omega_ci_str))
        f.write("node = {:>10.6f}{:s} deg\n".format(np.degrees(sim_outputs[-1][2].Omega), Omega_ci_str))
        f.write("f    = {:>10.6f}{:s} deg\n".format(np.degrees(sim_outputs[-1][2].f), f_ci_str))

        # Save the detected close encounters (Hill-sphere criterion). The distances are the minima
        # tracked at every internal integrator timestep and refined between steps, not sampled from
        # the output below. Every encounter is listed, in the order it happened.
        f.write("\nClose encounters (< {:.0f} Hill radii), in the order they happened.\n".format(n_hill))
        f.write("A body can appear more than once, if the object passed it more than once:\n")
        if encounters:
            for enc in encounters:
                f.write("  {:<8s} min dist = {:10.6f} AU ({:12.1f} km) at t = {:10.4f} d, R_Hill = {:.6f} AU ({:.2f} R_Hill)\n".format(
                    enc["body"], enc["min_dist_au"], enc["min_dist_au"]*149597870.7,
                    enc["time_days"], enc["hill_radius_au"], enc["n_hill"]))
        else:
            f.write("  None detected.\n")

        # Save the closest approach to every body, whether or not it counts as an encounter
        f.write("\nClosest approach to each body over the integration "
                "(tracked every integrator timestep, refined between steps):\n")
        f.write("  Note: the object starts at the Earth, so for every body except the Moon these\n")
        f.write("  minima are measured only after it left the Earth's neighbourhood ({:.0f} Earth\n".format(n_hill))
        f.write("  Hill radii). The Earth value is therefore ~{:.0f} R_Hill unless the object\n".format(n_hill))
        f.write("  genuinely returned. The Moon is tracked over the whole integration, because a\n")
        f.write("  lunar encounter can only happen while the object is still close to the Earth.\n")
        for body in dist_bodies:
            d_min = nominal_diag.get("min_dist_au", {}).get(body)
            t_min = nominal_diag.get("min_time_days", {}).get(body)
            if d_min is None:
                f.write("  {:<8s} not tracked (object never left the Earth's neighbourhood)\n".format(body))
            else:
                hill = HILL_RADII_AU.get(body)
                hill_str = "" if hill is None else "  ({:.2f} R_Hill)".format(d_min/hill)
                f.write("  {:<8s} {:12.6f} AU ({:14.1f} km) at t = {:10.4f} d{:s}\n".format(
                    body, d_min, d_min*149597870.7, t_min, hill_str))

        # Save what happened to the Monte Carlo clones
        if len(sim_outputs_mc):

            n_clones = len(sim_outputs_mc)
            f.write("\nMonte Carlo clone outcomes ({:d} clones):\n".format(n_clones))
            f.write("  {:<30s} {:5d}  ({:5.1f}%)\n".format(
                "Completed the full span", len(clones_survived),
                100.0*len(clones_survived)/n_clones))

            for body in sorted(clones_impacted, key=lambda b: -len(clones_impacted[b])):
                hit_names = clones_impacted[body]
                f.write("  {:<30s} {:5d}  ({:5.1f}%)\n".format(
                    "IMPACTED " + body, len(hit_names), 100.0*len(hit_names)/n_clones))
                times = [clone_diag[n]["impact"]["time_days"] for n in hit_names]
                f.write("      impact times from {:.4f} to {:.4f} d\n".format(
                    min(times), max(times)))

            if clones_escaped:
                f.write("  {:<30s} {:5d}  ({:5.1f}%)\n".format(
                    "Left the simulation volume", len(clones_escaped),
                    100.0*len(clones_escaped)/n_clones))

            if clone_encounters:
                f.write("\nClones with a close encounter (< {:.0f} Hill radii):\n".format(n_hill))
                for body in sorted(clone_encounters, key=lambda b: -clone_encounters[b]["count"]):
                    entry = clone_encounters[body]
                    f.write("  {:<10s} {:5d}/{:<5d} ({:5.1f}%)   {:5d} encounter(s)   closest over all "
                            "clones {:.6f} AU ({:.1f} km)\n".format(
                                body, entry["count"], n_clones, 100.0*entry["count"]/n_clones,
                                entry["n_encounters"], entry["closest_au"],
                                entry["closest_au"]*149597870.7))
            else:
                f.write("\nNo clone came within {:.0f} Hill radii of any body.\n".format(n_hill))

            if ci_uses_survivors_only:
                f.write("\nNote: the confidence intervals above use the {:d} clones that completed\n".format(
                    len(clones_survived)))
                f.write("the full span. Clones that impacted or were ejected ended at a different\n")
                f.write("epoch, so mixing their final elements in would blend different times.\n")

        # Save the Tisserand parameter and the run's provenance
        if reference_frame == "heliocentric":
            t_j = tisserandParameterJupiter(sim_outputs[-1][2].a, sim_outputs[-1][2].e,
                                            sim_outputs[-1][2].inc)
            if t_j is None:
                f.write("\nTisserand parameter w.r.t. Jupiter: undefined ({:s})\n".format(
                    tisserandClass(t_j)))
            else:
                f.write("\nTisserand parameter w.r.t. Jupiter: T_J = {:.6f}  -> {:s}\n".format(
                    t_j, tisserandClass(t_j)))

        if len(sim_outputs_mc):
            f.write("\nMonte Carlo: {:d} realizations, random seed {:d} "
                    "(pass --seed {:d} to reproduce this run)\n".format(
                        len(sim_outputs_mc), random_seed, random_seed))

        if nominal_diag.get("energy_rel_drift") is not None:
            f.write("Relative energy drift of the massive subsystem: {:.3e}\n".format(
                nominal_diag["energy_rel_drift"]))

        if nominal_diag.get("escaped"):
            f.write("\nEJECTED: the object left the simulation volume at t = {:.4f} d, "
                    "{:.2f} AU from the Sun.\n".format(
                        nominal_diag["escaped"]["time_days"], nominal_diag["escaped"]["dist_au"]))

        # Save any impact detected with REBOUND's collision detection
        if impact:
            f.write("\nIMPACT: the object hit {:s} at t = {:.4f} d "
                    "(centre distance {:.8f} AU)\n".format(
                        impact["body"], impact["time_days"], impact["dist_au"]))

        # Save the divergence/Lyapunov estimate derived from the Monte Carlo ensemble
        if lyap is not None:
            f.write("\nTrajectory divergence (from {:d} Monte Carlo realizations):\n".format(
                lyap["n_realizations"]))
            f.write("  RMS separation from nominal: {:.6e} -> {:.6e} AU (factor {:.2f})\n".format(
                lyap["separation_start_au"], lyap["separation_end_au"], lyap["growth_factor"]))
            f.write("  Fit quality: R2(exponential) = {:.4f}, R2(linear) = {:.4f} "
                    "({:d} samples)\n".format(
                        lyap["r2_exponential"], lyap["r2_linear"], lyap["n_samples_used"]))
            if lyap["n_truncated"]:
                f.write("  {:d} realization(s) ended early (e.g. impacted) and contributed only "
                        "over their own span.\n".format(lyap["n_truncated"]))
            if lyap["truncated_at_saturation"] and not lyap["saturated"]:
                f.write("  Fitted over the first {:.1f} d of {:.1f} d, i.e. the leading window in "
                        "which the\n".format(lyap["fit_window_days"], lyap["total_span_days"]))
                f.write("  ensemble was still a compact cloud (separation {:.6e} AU at the end of "
                        "that window).\n".format(lyap["separation_fit_end_au"]))
            if lyap["growth"] == "exponential":
                f.write("  Growth is exponential: lambda = {:.6f} /day, "
                        "Lyapunov time = {:.2f} days.\n".format(
                            lyap["lambda_per_day"], lyap["lyapunov_time_days"]))
                f.write("  The integration loses predictive value beyond this timescale.\n")
            elif lyap["growth"] == "linear":
                f.write("  Growth is linear, i.e. the motion is regular over this interval and no\n")
                f.write("  exponential divergence (and hence no Lyapunov time) can be derived.\n")
            elif lyap["growth"] == "saturated":
                f.write("  The ensemble was already spread over {:.1f}% of its mean heliocentric\n".format(
                    100*lyap["separation_end_au"]/lyap["heliocentric_distance_au"]))
                f.write("  distance ({:.3f} AU) within the first few samples, so there was no "
                        "compact\n".format(lyap["heliocentric_distance_au"]))
                f.write("  window to fit and no exponent is claimed. Use a denser output sampling\n")
                f.write("  (--outputs) or a shorter integration to resolve the early divergence.\n")
            else:
                f.write("  Growth fits neither a clean exponential nor a linear law, so no\n")
                f.write("  divergence timescale is claimed.\n")
            f.write("  Note: this is a finite-time estimate from the measurement-uncertainty\n")
            f.write("  ensemble, not a renormalised variational Lyapunov exponent.\n")

            # Semi-major-axis spread, which is not corrupted by Keplerian phase drift
            ed = lyap.get("element_divergence")
            if ed is not None:
                f.write("\n  Spread in the semi-major axis over {:.1f} d: {:.6e} -> {:.6e} AU "
                        "(factor {:.3f})\n".format(
                            ed["span_days"], ed["sigma_a_start_au"], ed["sigma_a_end_au"],
                            ed["sigma_a_growth_factor"]))
                f.write("  Fit quality: R2(exponential) = {:.4f}, R2(linear) = {:.4f}\n".format(
                    ed["r2_exponential"], ed["r2_linear"]))
                if ed["growth"] == "exponential":
                    f.write("  The spread in a grows exponentially: lambda = {:.6e} /day, "
                            "Lyapunov time = {:.2f} days.\n".format(
                                ed["lambda_per_day"], ed["lyapunov_time_days"]))
                    f.write("  This is the chaos estimate to prefer: unlike the position\n")
                    f.write("  separation it is not corrupted by Keplerian phase drift, which\n")
                    f.write("  smears the cloud along the orbit whether or not the motion is chaotic.\n")
                elif ed["growth"] == "regular":
                    f.write("  The spread in a is essentially unchanged, so the orbit is regular\n")
                    f.write("  over this span and the position divergence above is phase drift.\n")
                else:
                    f.write("  The spread in a grows, but not as a clean exponential.\n")

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

    # Start the outer-planet distance axis at zero, so the distances are read against the Sun
    axs[2, 2].set_ylim(ymin=0)

    axs[2, 2].legend()


    # Mark the detected close encounters at the point of closest approach on the distance subplots.
    # Every passage gets a star, but only the deepest one per body is named: an object in resonance
    #   with a planet meets it over and over, and one text label per passage makes the panel
    #   unreadable.
    labeled_axes = set()
    deepest_per_body = {}
    for enc in encounters:
        best = deepest_per_body.get(enc["body"])
        if (best is None) or (enc["min_dist_au"] < best["min_dist_au"]):
            deepest_per_body[enc["body"]] = enc

    for enc in encounters:
        body = enc["body"]
        t_enc = enc["time_days"]
        d_enc = enc["min_dist_au"]
        name_it = (deepest_per_body[body] is enc)

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

            if name_it:
                ax.annotate(body, (t_enc, d_plot), textcoords="offset points", xytext=(5, 5),
                            color="red", fontsize=8)


    # Set the axis labels
    for ax in axs.flatten():

        ax.set_xlabel("Time [days]")

        # Disable offset and scientific notation
        ax.ticklabel_format(useOffset=False, style='plain')


    # Plot the MC realizations (all in thin alpha=0.5 lines)
    for mc_name in sim_outputs_mc:

        # Each realization gets its own time axis: a realization truncated early (for example by an
        # impact) is shorter than the nominal solution, and plotting it against the nominal time
        # axis would raise a dimension-mismatch error.
        t_mc = [x[0]/(2*np.pi)*365.25 for x in sim_outputs_mc[mc_name]]

        a_mc = [x[2].a*dist_unit_multiplier for x in sim_outputs_mc[mc_name]]
        e_mc = [x[2].e for x in sim_outputs_mc[mc_name]]
        incl_mc = [x[2].inc for x in sim_outputs_mc[mc_name]]
        Omega_mc = [x[2].Omega for x in sim_outputs_mc[mc_name]]
        omega_mc = [x[2].omega for x in sim_outputs_mc[mc_name]]
        f_mc = [x[2].f for x in sim_outputs_mc[mc_name]]
        earth_dist = [x[3]["Earth"] for x in sim_outputs_mc[mc_name]]

        axs[0, 0].plot(t_mc, a_mc, alpha=0.5, color='k', lw=0.5)
        axs[0, 1].plot(t_mc, e_mc, alpha=0.5, color='k', lw=0.5)
        axs[1, 0].plot(t_mc, np.degrees(incl_mc), alpha=0.5, color='k', lw=0.5)
        axs[1, 1].plot(t_mc, np.degrees(Omega_mc), alpha=0.5, color='k', lw=0.5)
        axs[2, 0].plot(t_mc, np.degrees(omega_mc), alpha=0.5, color='k', lw=0.5)
        axs[2, 1].plot(t_mc, np.degrees(f_mc), alpha=0.5, color='k', lw=0.5)
        axs[0, 2].plot(t_mc, np.array(earth_dist)*dist_unit_multiplier, alpha=0.5, color='k', lw=0.5)

    
    
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
    ### Machine-readable results, so downstream analysis does not have to parse the text report ###

    results_json_path = os.path.join(out_dir, "rebound_simulation_results.json")

    def _elementSeries(rows):
        """ Convert one run's outputs into plain lists of floats. """
        return {
            "time_days": [row[0]/(2*np.pi)*365.25 for row in rows],
            "a_au": [row[2].a for row in rows],
            "e": [row[2].e for row in rows],
            "incl_deg": [np.degrees(row[2].inc) for row in rows],
            "peri_deg": [np.degrees(row[2].omega) for row in rows],
            "node_deg": [np.degrees(row[2].Omega) for row in rows],
            "f_deg": [np.degrees(row[2].f) for row in rows],
            "body_distances_au": {body: [row[3][body] for row in rows] for body in dist_bodies},
        }

    results_json = {
        "traj_id": str(traj.traj_id),
        "run": {
            "integration_days": achieved_days,
            "requested_days": sim_days,
            "direction": direction,
            "reference_frame": reference_frame,
            "ephemeris": "horizons" if args.horizons else "local_de430",
            "n_outputs": args.outputs,
            "beta": beta,
            "start_epoch_jd_tdb": traj.jdt_ref,
            "final_epoch_jd_tdb": final_epoch_jd,
            "final_epoch_utc": time_utc.iso,
            "mc_runs": len(sim_outputs_mc),
            "random_seed": random_seed,
            "runtime_s": sim_wall,
        },
        "final_elements": {
            "a": a_val, "a_units": a_units,
            "q": q_val, "q_units": q_units,
            "e": e_val,
            "incl_deg": i_val, "peri_deg": peri_val, "node_deg": node_val, "f_deg": f_val,
            "tisserand_jupiter": (tisserandParameterJupiter(
                sim_outputs[-1][2].a, e_val, sim_outputs[-1][2].inc)
                if reference_frame == "heliocentric" else None),
        },
        # One entry per passage, ordered by time. A body can appear more than once: before the
        #   encounter minima were refined between steps, only the deepest approach per body was kept.
        "encounters": encounters,
        "closest_approaches_au": nominal_diag.get("min_dist_au"),
        "closest_approach_times_days": nominal_diag.get("min_time_days"),
        "impact": impact,
        "escaped": nominal_diag.get("escaped"),
        "clone_outcomes": {
            "n_clones": len(sim_outputs_mc),
            "n_completed": len(clones_survived),
            "n_escaped": len(clones_escaped),
            "impacted": {body: {"count": len(names),
                                "fraction": len(names)/len(sim_outputs_mc),
                                "times_days": [clone_diag[n]["impact"]["time_days"] for n in names]}
                         for body, names in clones_impacted.items()},
            "close_encounters": {body: {"count": entry["count"],
                                        "fraction": entry["count"]/len(sim_outputs_mc),
                                        "n_encounters": entry["n_encounters"],
                                        "closest_au": entry["closest_au"]}
                                 for body, entry in clone_encounters.items()},
            "ci_uses_survivors_only": ci_uses_survivors_only,
            "n_hill_threshold": n_hill,
        } if len(sim_outputs_mc) else None,
        "clone_closest_approaches_au": {name: diag.get("min_dist_au")
                                        for name, diag in clone_diag.items()},
        "clone_encounters": {name: diag.get("encounters", []) for name, diag in clone_diag.items()},
        "energy_rel_drift": nominal_diag.get("energy_rel_drift"),
        "divergence": lyap,
        "nominal": _elementSeries(sim_outputs),
        "monte_carlo": {name: _elementSeries(rows) for name, rows in sim_outputs_mc.items()},
    }

    with open(results_json_path, "w") as jf:
        json.dump(results_json, jf, indent=1, default=float)

    ### ###

    print("  Saved report : {:s}".format(results_txt_path))
    print("  Saved plot   : {:s}".format(plot_png_path))
    print("  Saved data   : {:s}".format(results_json_path))
    print(hdr)

    plt.show()
