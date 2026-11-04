""" An orbit given by its published elements and the time of the fireball, in the form the Jupiter-family
dynamics of wmpl.Utils.OrbitClassification takes from a trajectory.

Elements lists do not give where on the orbit the meteoroid was, but a fireball is at the Earth, so it was at
the node that the Earth was passing at that moment. Which node, and how far it lies from the Earth, follow from
the elements and the time. The distance is also a check of the elements: published elements with their angles
place the node within about 0.01 AU of the Earth, while a mistyped or inconsistent orbit does not reach it.
"""

from __future__ import print_function, division, absolute_import

from types import SimpleNamespace

import numpy as np


# Farthest the node may lie from the Earth before the elements are refused (AU). The elements of three published
#   fireballs give 0.0004 (Trigo-Rodriguez et al. 2009), 0.009 (Madiedo et al. 2014) and 0.012 AU (Hughes et al.
#   2022), while the a, e and i listed for the first in Shober et al. (2024), with its node and argument of
#   perihelion, give 0.25 AU
ELEMENTS_MAX_NODE_MISS = 0.05


def nodeAtTheEarth(a, e, incl, node, peri, jd_tdb):
    """ The node of an orbit that the Earth is closest to at a given time, and how the orbit passes there.

    Arguments:
        a: [float] Semi-major axis (AU).
        e: [float] Eccentricity.
        incl: [float] Inclination (rad).
        node: [float] Longitude of the ascending node (rad), ecliptic J2000.
        peri: [float] Argument of perihelion (rad).
        jd_tdb: [float] Julian date (TDB) of the fireball.

    Return:
        [dict] 'node': 'ascending' or 'descending', 'true_anomaly' (rad) at that node, 'miss_au': distance
            from the Earth, and 'v_g' (m/s), the speed relative to the Earth there, without its attraction.
    """

    from wmpl.Rebound import REBOUND as reb

    names, states, _ = reb.planetStatesAtEpoch(jd_tdb)
    earth = np.array(states[names.index("Earth")]) - np.array(states[names.index("Sun")])

    best = None

    for label, anomaly in (('ascending', -peri), ('descending', np.pi - peri)):

        sim = reb.rb.Simulation()
        sim.add(m=1.0)
        sim.add(primary=sim.particles[0], a=a, e=e, inc=incl, Omega=node, omega=peri, f=anomaly)
        p = sim.particles[1]

        miss = float(np.linalg.norm(np.array([p.x, p.y, p.z]) - earth[:3]))

        if (best is None) or (miss < best['miss_au']):

            speed = np.linalg.norm(np.array([p.vx, p.vy, p.vz]) - earth[3:])
            best = {'node': label, 'true_anomaly': float(anomaly), 'miss_au': miss,
                'v_g': float(speed*reb.AU_M/(reb.SIM_TIME_DAYS*86400.0))}

    return best


def trajectoryFromElements(a, e, incl, node, peri, jd_utc, sigma=None, max_node_miss=ELEMENTS_MAX_NODE_MISS):
    """ Stand-in for a trajectory with the orbit given by its elements, for
        wmpl.Utils.OrbitClassification.classifyJupiterFamilyDynamicsShober2024.

    Arguments:
        a: [float] Semi-major axis (AU).
        e: [float] Eccentricity.
        incl: [float] Inclination (deg).
        node: [float] Longitude of the ascending node (deg), ecliptic J2000.
        peri: [float] Argument of perihelion (deg).
        jd_utc: [float] Julian date (UTC) of the fireball.

    Keyword arguments:
        sigma: [dict] 1-sigma of 'a' (AU), 'e', 'i', 'node' and 'peri' (deg). None (default) leaves the
            uncertainties out, so the clones take those Shober et al. assumed for the MORP fireballs.
        max_node_miss: [float] Farthest the node may lie from the Earth (AU). Default ELEMENTS_MAX_NODE_MISS.

    Return:
        [SimpleNamespace] With 'orbit', 'jdt_ref', 'uncertainties' and 'node_passage' (see nodeAtTheEarth). A
            ValueError if the orbit is not an ellipse or does not reach the Earth.
    """

    from wmpl.Rebound import REBOUND as reb

    if not reb.REBOUND_FOUND:
        raise ImportError(reb._REBOUND_IMPORT_ERROR)

    if (a <= 0) or not (0 <= e < 1):
        raise ValueError("The orbit has to be an ellipse, got a = {:g} AU and e = {:g}.".format(a, e))

    incl, node, peri = np.radians(incl), np.radians(node), np.radians(peri)

    passage = nodeAtTheEarth(a, e, incl, node, peri, reb.utcToTdb(jd_utc))

    if passage['miss_au'] > max_node_miss:
        raise ValueError("At that time the closest node of this orbit lies {:.3f} AU from the Earth, so it is not the "
            "orbit of a fireball then. Check the elements and the date, or raise --max_node_miss ({:g} AU) to "
            "integrate it anyway.".format(passage['miss_au'], max_node_miss))

    uncertainties = None
    if sigma is not None:
        uncertainties = SimpleNamespace(a=sigma['a'], e=sigma['e'], i=np.radians(sigma['i']),
            node=np.radians(sigma['node']), peri=np.radians(sigma['peri']), source="given with the elements")

    return SimpleNamespace(jdt_ref=jd_utc, uncertainties=uncertainties, node_passage=passage,
        orbit=SimpleNamespace(a=a, e=e, q=a*(1.0 - e), i=incl, node=node, peri=peri,
            true_anomaly=passage['true_anomaly'], v_g=passage['v_g']))
