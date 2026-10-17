""" Minimum orbit intersection distance (MOID) between two orbits, vectorised over many pairs at once.
Both can be elliptical, or one of them parabolic or hyperbolic.

The MOID is the smallest distance between two orbits taken as fixed curves in space, wherever the
bodies are on them. It depends only on the orbital elements, so unlike the distance at a passage it
does not depend on the phasing of the two bodies: an encounter closer than the MOID is impossible, and
one near it is possible whenever the bodies arrive at the right place together.

Method: the first orbit is sampled at n_scan points in eccentric anomaly; for each sample the closest
point of the second ellipse is found with Newton's method on its eccentric anomaly, started from the
direction of the sample in the second orbit's plane (exact for a circle). The local minima of this
distance along the first orbit are the candidates; each one, and its two neighbouring samples, start a
two-dimensional Newton iteration on both eccentric anomalies, which converges to the exact stationary
point. The neighbours handle nearly tangent, nearly coplanar orbits, whose two minima can be closer
together than the sampling step.

This is the sampling-and-refinement family of Wisniowski & Rickman (2013, Acta Astron. 63, 293) and of
the SDG method of Hedo, Ruiz & Pelaez (2018, MNRAS 479, 3288), which also reduce the problem to the
minima of the distance from one orbit to the other. Like them it has no formal guarantee of finding the
global minimum, unlike the algebraic method of Baluev & Mikryukov (2019, Astron. Comput. 27, 11), which
finds every stationary point with error control. It was validated instead against a dense grid on both
orbits refined with a tight minimiser: it agrees to ~1e-15 AU on 600 meteoroid-like orbits against the
Earth (including Earth-crossing orbits inclined less than 2 deg, the hard case) and on 150 pairs of
eccentric orbits (e = 0.3-0.97, prograde and retrograde) in either order, at ~30-40 microseconds per
pair.

An open (parabolic or hyperbolic) orbit has no finite semi-major axis, so calcMOIDFromPerihelion takes
perihelion distances. The elliptical orbit is the one scanned, and on the open one the closest point and
the refinement use the true anomaly, out to MOID_OPEN_MAX_R perihelion distances; the closest point to
each sample is started both from its direction and from the perihelion, as a point can have more than
one stationary distance to an open orbit. Against the same kind of reference (a grid in the true anomaly
of the open orbit) it agrees to ~1e-15 AU on 200 meteoroid-like ellipses against parabolic and
hyperbolic (e up to 1.6) orbits, in either order, at ~120 microseconds per pair. Two open orbits are not
handled.

Usage:
    from wmpl.Utils.MOID import calcMOID, calcMOIDFromPerihelion
    moid = calcMOID(a1, e1, i1, node1, peri1, a2, e2, i2, node2, peri2)              # ellipses
    moid = calcMOIDFromPerihelion(q1, e1, i1, node1, peri1, q2, e2, i2, node2, peri2)  # any e, one < 1

calcMOIDTancredi keeps the previous implementation of wmpl.Utils.OrbitClassification (Powell's method
from four starting points, Tancredi 2014) for comparison; it is scalar, ~700 times slower, and can stop
at a local minimum for two eccentric orbits.

The elements can be scalars or arrays of any broadcastable shape; the angles are in radians, and the
MOID is in the unit of the semi-major axes (or perihelion distances).
"""

from __future__ import print_function, division, absolute_import

import numpy as np
import scipy.optimize


# Samples of the first orbit in the coarse scan
MOID_N_SCAN = 48

# Local minima of the scan refined, per pair (each also from its two neighbours)
MOID_N_CANDIDATES = 3

# Newton iterations: for the closest point of the second orbit to each sample, and for the 2D
#   refinement of the candidates
MOID_N_INNER = 4
MOID_N_NEWTON = 8

# Pairs processed at once, to bound the memory of the scan arrays (~1 kB per pair and sample)
MOID_CHUNK = 20000

# An open (parabolic or hyperbolic) orbit is followed out to this heliocentric distance, in the unit of
#   the perihelion distance. Far beyond any elliptical orbit it is compared with, and short of the
#   asymptote, where the true anomaly stops being a usable parameter.
MOID_OPEN_MAX_R = 1e4


def _orbitFrame(incl, node, peri):
    """ Unit vectors to the perihelion (P) and 90 deg ahead of it in the orbit plane (Q), shape (..., 3). """

    ci, si = np.cos(incl), np.sin(incl)
    cn, sn = np.cos(node), np.sin(node)
    cw, sw = np.cos(peri), np.sin(peri)

    P = np.stack([cn*cw - sn*sw*ci, sn*cw + cn*sw*ci, sw*si], axis=-1)
    Q = np.stack([-cn*sw - sn*cw*ci, -sn*sw + cn*cw*ci, cw*si], axis=-1)

    return P, Q


def _ellipsePoint(a, b, e, P, Q, E):
    """ Position on the ellipse at the eccentric anomaly E, and its first and second derivatives in E. """

    cE, sE = np.cos(E), np.sin(E)

    r = (a*(cE - e))[..., None]*P + (b*sE)[..., None]*Q
    dr = (-a*sE)[..., None]*P + (b*cE)[..., None]*Q
    d2r = (-a*cE)[..., None]*P + (-b*sE)[..., None]*Q

    return r, dr, d2r


def _conicPoint(q, e, P, Q, s):
    """ Position on a conic of perihelion distance q and eccentricity e, and its first and second
    derivatives in the parameter s: the eccentric anomaly for an ellipse (e < 1), the true anomaly for
    an open orbit (e >= 1). """

    is_open = e >= 1

    # Ellipse, in the eccentric anomaly (with harmless values in the rows of open orbits)
    e_ell = np.where(is_open, 0.5, e)
    a = q/(1 - e_ell)
    r_ell, dr_ell, d2r_ell = _ellipsePoint(a, a*np.sqrt(1 - e_ell**2), e_ell, P, Q, s)
    if not np.any(is_open):
        return r_ell, dr_ell, d2r_ell

    # Open orbit, in the true anomaly: r = p/(1 + e cos s), with p = q(1 + e)
    cs, ss = np.cos(s), np.sin(s)
    p = q*(1 + e)
    den = np.maximum(1 + e*cs, 1e-12)
    r = p/den
    r_s = p*e*ss/den**2
    r_ss = p*e*(cs*den + 2*e*ss**2)/den**3
    x, y = r*cs, r*ss
    dx, dy = r_s*cs - r*ss, r_s*ss + r*cs
    d2x, d2y = r_ss*cs - 2*r_s*ss - r*cs, r_ss*ss + 2*r_s*cs - r*ss
    r_open = x[..., None]*P + y[..., None]*Q
    dr_open = dx[..., None]*P + dy[..., None]*Q
    d2r_open = d2x[..., None]*P + d2y[..., None]*Q

    o = is_open[..., None]
    return np.where(o, r_open, r_ell), np.where(o, dr_open, dr_ell), np.where(o, d2r_open, d2r_ell)


def _openLimit(q, e):
    """ Largest true anomaly followed on an open orbit (see MOID_OPEN_MAX_R); pi for an ellipse, whose
    parameter is not limited. """

    with np.errstate(divide="ignore", invalid="ignore"):
        p = q*(1 + e)
        cos_lim = np.clip((p/(MOID_OPEN_MAX_R*q) - 1)/np.where(e > 0, e, 1.0), -1.0, 1.0)
    return np.where(e >= 1, np.arccos(cos_lim), np.pi)


def _dot(u, v):
    return np.einsum("...j,...j->...", u, v)


def _moidChunk(q1, e1, P1, Q1, q2, e2, P2, Q2, n_scan, n_candidates, n_inner, n_newton):
    """ MOID of N pairs of orbits given by their shapes and frames (all arrays of length N). The first
    orbit must be an ellipse; the second can be open. Returns the MOID and the parameters of the closest
    points (see _conicPoint). """

    n = q1.shape[0]
    open2 = e2 >= 1

    # Column views for broadcasting against the samples
    c = lambda x: x[:, None]
    q1c, e1c, P1c, Q1c = c(q1), c(e1), c(P1), c(Q1)
    q2c, e2c, P2c, Q2c = c(q2), c(e2), c(P2), c(Q2)
    open2c = c(open2)
    v_lim = c(_openLimit(q2, e2))

    def clipOpen(v):
        return np.where(open2c, np.clip(v, -v_lim, v_lim), v)

    # 1. Samples of the first orbit
    u_scan = np.broadcast_to(np.linspace(0, 2*np.pi, n_scan, endpoint=False), (n, n_scan))
    r1, _, _ = _conicPoint(q1c, e1c, P1c, Q1c, u_scan)

    # 2. Closest point of the second orbit to each sample, by Newton's method on the derivative of the
    #   squared distance. An ellipse starts from the sample's direction in its plane, measured from its
    #   centre (exact for a circle). An open orbit starts both from the sample's direction measured from
    #   the focus and from the perihelion, as a point can have more than one stationary distance to it,
    #   and keeps the closer.
    def closest(v):
        for _ in range(n_inner):
            r2, dr2, d2r2 = _conicPoint(q2c, e2c, P2c, Q2c, v)
            d = r2 - r1
            g = _dot(d, dr2)
            h = _dot(dr2, dr2) + _dot(d, d2r2)
            v = clipOpen(v - np.where(h > 0, g/np.where(h > 0, h, 1.0), 0.0))
        r2, _, _ = _conicPoint(q2c, e2c, P2c, Q2c, v)
        return v, _dot(r2 - r1, r2 - r1)

    e2_ell = np.where(open2, 0.5, e2)
    a2 = q2/(1 - e2_ell)
    b2 = a2*np.sqrt(1 - e2_ell**2)
    from_centre = r1 + (a2*e2_ell)[:, None, None]*P2c
    v_centre = np.arctan2(_dot(from_centre, Q2c)/c(b2), _dot(from_centre, P2c)/c(a2))
    v_focus = clipOpen(np.arctan2(_dot(r1, Q2c), _dot(r1, P2c)))

    v_scan, dist2_scan = closest(np.where(open2c, v_focus, v_centre))
    if np.any(open2):
        v_peri, dist2_peri = closest(np.where(open2c, 0.0, v_centre))
        better = dist2_peri < dist2_scan
        v_scan, dist2_scan = np.where(better, v_peri, v_scan), np.where(better, dist2_peri, dist2_scan)

    # 3. Candidates: the deepest local minima along the first orbit, each with its two neighbours
    is_min = (dist2_scan <= np.roll(dist2_scan, 1, axis=1)) & (dist2_scan <= np.roll(dist2_scan, -1, axis=1))
    score = np.where(is_min, dist2_scan, np.inf)
    n_candidates = min(n_candidates, n_scan)
    idx = np.argsort(score, axis=1)[:, :n_candidates]
    rows = np.arange(n)[:, None]
    valid = np.isfinite(score[rows, idx])
    idx = np.concatenate([idx, (idx - 1) % n_scan, (idx + 1) % n_scan], axis=1)
    valid = np.concatenate([valid, valid, valid], axis=1)
    u = u_scan[rows, idx]
    v = v_scan[rows, idx]

    # 4. 2D Newton on the squared distance f(u, v) = |r1(u) - r2(v)|^2, stepping only where the Hessian
    #   is positive definite (a minimum)
    for _ in range(n_newton):
        p1, dp1, d2p1 = _conicPoint(q1c, e1c, P1c, Q1c, u)
        p2, dp2, d2p2 = _conicPoint(q2c, e2c, P2c, Q2c, v)
        d = p1 - p2
        fu, fv = _dot(d, dp1), -_dot(d, dp2)
        fuu = _dot(dp1, dp1) + _dot(d, d2p1)
        fvv = _dot(dp2, dp2) - _dot(d, d2p2)
        fuv = -_dot(dp1, dp2)
        det = fuu*fvv - fuv**2
        ok = (det > 0) & (fuu > 0)
        safe = np.where(ok, det, 1.0)
        u = u - np.where(ok, (fvv*fu - fuv*fv)/safe, 0.0)
        v = clipOpen(v - np.where(ok, (fuu*fv - fuv*fu)/safe, 0.0))

    p1, _, _ = _conicPoint(q1c, e1c, P1c, Q1c, u)
    p2, _, _ = _conicPoint(q2c, e2c, P2c, Q2c, v)
    dist2 = np.where(valid, _dot(p1 - p2, p1 - p2), np.inf)

    # The refined minimum, never worse than the scan itself
    k = np.argmin(dist2, axis=1)
    best = dist2[np.arange(n), k]
    k_scan = np.argmin(dist2_scan, axis=1)
    scan_best = dist2_scan[np.arange(n), k_scan]
    use_scan = scan_best < best

    moid = np.sqrt(np.where(use_scan, scan_best, best))
    s1 = np.where(use_scan, u_scan[np.arange(n), k_scan], u[np.arange(n), k]) % (2*np.pi)
    s2 = np.where(use_scan, v_scan[np.arange(n), k_scan], v[np.arange(n), k])
    s2 = np.where(open2, s2, s2 % (2*np.pi))

    return moid, s1, s2


def _moidFromPerihelion(q1, e1, i1, node1, peri1, q2, e2, i2, node2, peri2, return_anomalies, n_scan,
                        n_candidates):
    """ calcMOIDFromPerihelion on validated inputs (see there). """

    arrays = np.broadcast_arrays(*[np.asarray(x, dtype=np.float64)
                                   for x in (q1, e1, i1, node1, peri1, q2, e2, i2, node2, peri2)])
    shape = arrays[0].shape
    q1, e1, i1, node1, peri1, q2, e2, i2, node2, peri2 = [x.ravel().copy() for x in arrays]

    moid = np.full(q1.shape, np.nan)
    s1 = np.full(q1.shape, np.nan)
    s2 = np.full(q1.shape, np.nan)

    good = np.ones(q1.shape, dtype=bool)
    for x in (q1, e1, i1, node1, peri1, q2, e2, i2, node2, peri2):
        good &= np.isfinite(x)
    good &= (q1 > 0) & (q2 > 0) & (e1 >= 0) & (e2 >= 0)

    # The first orbit is the one scanned and must be an ellipse: swap the pairs whose first orbit is
    #   open, and leave out those with two open orbits
    swap = good & (e1 >= 1) & (e2 < 1)
    good &= (e1 < 1) | (e2 < 1)
    first = [q1, e1, i1, node1, peri1]
    second = [q2, e2, i2, node2, peri2]
    for x, y in zip(first, second):
        x[swap], y[swap] = y[swap], x[swap].copy()

    idx = np.flatnonzero(good)
    for start in range(0, len(idx), MOID_CHUNK):
        sel = idx[start:start + MOID_CHUNK]
        P1, Q1 = _orbitFrame(i1[sel], node1[sel], peri1[sel])
        P2, Q2 = _orbitFrame(i2[sel], node2[sel], peri2[sel])
        moid[sel], s1[sel], s2[sel] = _moidChunk(q1[sel], e1[sel], P1, Q1, q2[sel], e2[sel], P2, Q2,
                                                 n_scan, n_candidates, MOID_N_INNER, MOID_N_NEWTON)

    s1[swap], s2[swap] = s2[swap], s1[swap].copy()

    def shaped(x):
        x = x.reshape(shape)
        return float(x) if x.ndim == 0 else x

    if return_anomalies:
        return shaped(moid), shaped(s1), shaped(s2)

    return shaped(moid)


def calcMOIDFromPerihelion(q1, e1, i1, node1, peri1, q2, e2, i2, node2, peri2, return_anomalies=False,
                           n_scan=MOID_N_SCAN, n_candidates=MOID_N_CANDIDATES):
    """ Minimum orbit intersection distance between two orbits given by their perihelion distances,
    which can be elliptical, parabolic or hyperbolic as long as one of the two is elliptical (a
    meteoroid against a long-period comet, for example).

    An open orbit is followed out to MOID_OPEN_MAX_R times its perihelion distance, far beyond any
    elliptical orbit it is compared with.

    Arguments:
        q1, e1, i1, node1, peri1: [float or ndarray] Elements of the first orbit: perihelion distance,
            eccentricity, inclination, longitude of the ascending node and argument of perihelion
            (angles in radians).
        q2, e2, i2, node2, peri2: [float or ndarray] The same for the second orbit, in the same frame.
            All the elements are broadcast together.

    Keyword arguments:
        return_anomalies: [bool] Also return where the closest points are on the two orbits: the
            eccentric anomaly on an ellipse, the true anomaly on an open orbit (radians). Default False.
        n_scan: [int] Samples of the elliptical orbit in the coarse scan. Default MOID_N_SCAN.
        n_candidates: [int] Local minima of the scan refined. Default MOID_N_CANDIDATES.

    Return:
        moid: [float or ndarray] The MOID, in the unit of the perihelion distances, with the broadcast
            shape of the inputs (a float for scalar inputs). NaN where both orbits are open, q <= 0,
            e < 0, or an element is not finite.
        If return_anomalies: (moid, s1, s2).
    """

    return _moidFromPerihelion(q1, e1, i1, node1, peri1, q2, e2, i2, node2, peri2, return_anomalies,
                               n_scan, n_candidates)


def calcMOID(a1, e1, i1, node1, peri1, a2, e2, i2, node2, peri2, return_anomalies=False,
             n_scan=MOID_N_SCAN, n_candidates=MOID_N_CANDIDATES):
    """ Minimum orbit intersection distance between two elliptical orbits. For parabolic or hyperbolic
    orbits, which have no finite positive semi-major axis, use calcMOIDFromPerihelion.

    Arguments:
        a1, e1, i1, node1, peri1: [float or ndarray] Elements of the first orbit: semi-major axis,
            eccentricity, inclination, longitude of the ascending node and argument of perihelion
            (angles in radians).
        a2, e2, i2, node2, peri2: [float or ndarray] The same for the second orbit, in the same frame.
            All the elements are broadcast together.

    Keyword arguments:
        return_anomalies: [bool] Also return the eccentric anomalies (radians) of the closest points
            on the two orbits. Default False.
        n_scan: [int] Samples of the first orbit in the coarse scan. Default MOID_N_SCAN.
        n_candidates: [int] Local minima of the scan refined. Default MOID_N_CANDIDATES.

    Return:
        moid: [float or ndarray] The MOID, in the unit of the semi-major axes, with the broadcast shape
            of the inputs (a float for scalar inputs). NaN where either orbit is not elliptical
            (e >= 1, e < 0 or a <= 0) or an element is not finite.
        If return_anomalies: (moid, E1, E2).
    """

    a1, e1, a2, e2 = [np.asarray(x, dtype=np.float64) for x in (a1, e1, a2, e2)]
    with np.errstate(invalid="ignore"):
        elliptical = (a1 > 0) & (a2 > 0) & (e1 >= 0) & (e1 < 1) & (e2 >= 0) & (e2 < 1)
    nan = np.where(elliptical, 0.0, np.nan)

    return _moidFromPerihelion(a1*(1 - e1) + nan, e1, i1, node1, peri1, a2*(1 - e2) + nan, e2, i2, node2,
                               peri2, return_anomalies, n_scan, n_candidates)


def _orbitPosition(ecc_anomaly, a, e, i, node, peri):
    """ Heliocentric position of a point on an orbit, given its eccentric anomaly.

    Arguments:
        ecc_anomaly: [float] eccentric anomaly (rad)
        a: [float] semi-major axis (AU)
        e: [float] num. eccentricity
        i: [float] inclination (rad)
        node: [float] longitude of ascending node (rad)
        peri: [float] argument of perihelion (rad)

    Return:
        [ndarray] position vector (AU)
    """

    # Position in the orbital plane, with the origin at the focus and x towards perihelion
    x = a*(np.cos(ecc_anomaly) - e)
    y = a*np.sqrt(1.0 - e**2)*np.sin(ecc_anomaly)

    cos_w, sin_w = np.cos(peri), np.sin(peri)
    cos_i, sin_i = np.cos(i), np.sin(i)
    cos_O, sin_O = np.cos(node), np.sin(node)

    # Rotate by the argument of perihelion, then by the inclination and the node, i.e. the
    #   standard 3-1-3 Euler sequence written out
    x_peri = x*cos_w - y*sin_w
    y_peri = x*sin_w + y*cos_w

    return np.array([x_peri*cos_O - y_peri*cos_i*sin_O, x_peri*sin_O + y_peri*cos_i*cos_O,
        y_peri*sin_i])


def calcMOIDTancredi(a1, e1, i1, O1, w1, a2, e2, i2, O2, w2):
    """ Calculate the minimum orbital intersection distance between two orbits, with the method of
        Tancredi (2014), appendix A. Kept for comparison: use calcMOID.

        This was the MOID of wmpl.Utils.OrbitClassification before calcMOID replaced it. It is scalar
        and ~700 times slower (~20 ms per pair against ~0.03 ms), and against the dense-grid reference
        of the tests it agrees with calcMOID on orbits against a planet (0 disagreements in 240,
        including Earth-crossing orbits inclined less than 2 deg), but not always on two eccentric
        orbits: in 1 of 80 pairs it stopped at a local minimum, 0.0260 AU instead of 0.0206 AU (e = 0.94
        and 0.81, i ~ 1 deg; see the tests).

        Note the element order: this function takes the semi-major axis first, where every
        dissimilarity criterion in wmpl.Utils.Dcriteria takes the perihelion distance. The two
        signatures are otherwise the same shape, so a q passed here is silently accepted.

        The distance between a point on each orbit is minimised over the two eccentric anomalies.
        That surface has more than one local minimum, so the search is repeated from the three
        further starting points the paper prescribes and the smallest of the four results is taken.

        Reference: Tancredi (2014), Icarus 234, 66, appendix A, doi:10.1016/j.icarus.2014.02.013.

    Arguments:
        a1: [float] semi-major axis of the first orbit (AU)
        e1: [float] num. eccentricity of the first orbit
        i1: [float] inclination of the first orbit (rad)
        O1: [float] longitude of ascending node of the first orbit (rad)
        w1: [float] argument of perihelion of the first orbit (rad)
        a2: [float] semi-major axis of the second orbit (AU)
        e2: [float] num. eccentricity of the second orbit
        i2: [float] inclination of the second orbit (rad)
        O2: [float] longitude of ascending node of the second orbit (rad)
        w2: [float] argument of perihelion of the second orbit (rad)

    Return:
        [float] minimum orbital intersection distance (AU)
    """

    orbit1 = (a1, e1, i1, O1, w1)
    orbit2 = (a2, e2, i2, O2, w2)

    def separation(anomalies):
        return float(np.linalg.norm(_orbitPosition(anomalies[0], *orbit1)
            - _orbitPosition(anomalies[1], *orbit2)))

    def refine(start):

        # The tolerances are well below the AU-scale distances being compared, so the result is
        #   limited by which basin the search lands in rather than by how finely it converges
        found = np.atleast_1d(scipy.optimize.fmin_powell(separation, np.asarray(start, dtype=float),
            disp=False, xtol=1e-10, ftol=1e-12))
        return found, separation(found)

    # Powell converges to whichever local minimum it starts nearest, and the separation surface
    #   has several. The paper restarts from three points reflected about the first result, which
    #   between them reach the other basins: the opposite side of both orbits, the opposite side of
    #   the second only, and the two reflections combined.
    first, smallest = refine((0.0, 0.0))

    for start in ((first[0] + np.pi, first[1] + np.pi), (first[0], 2*np.pi - first[1]),
                  (first[0] + np.pi, np.pi - first[1])):

        _, distance = refine(start)
        smallest = min(smallest, distance)

    return smallest
