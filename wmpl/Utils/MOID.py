""" Minimum orbit intersection distance (MOID) between two elliptical orbits, vectorised over many
pairs at once.

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

Usage:
    from wmpl.Utils.MOID import calcMOID
    moid = calcMOID(a1, e1, i1, node1, peri1, a2, e2, i2, node2, peri2)

The elements can be scalars or arrays of any broadcastable shape; the angles are in radians, and the
MOID is in the unit of the semi-major axes.
"""

from __future__ import print_function, division, absolute_import

import numpy as np


# Samples of the first orbit in the coarse scan
MOID_N_SCAN = 48

# Local minima of the scan refined, per pair (each also from its two neighbours)
MOID_N_CANDIDATES = 3

# Newton iterations: for the closest point of the second ellipse to each sample, and for the 2D
#   refinement of the candidates
MOID_N_INNER = 4
MOID_N_NEWTON = 8

# Pairs processed at once, to bound the memory of the scan arrays (~1 kB per pair and sample)
MOID_CHUNK = 20000


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


def _dot(u, v):
    return np.einsum("...j,...j->...", u, v)


def _moidChunk(a1, e1, P1, Q1, a2, e2, P2, Q2, n_scan, n_candidates, n_inner, n_newton):
    """ MOID of N pairs of elliptical orbits given by their shapes and frames (all arrays of length N).
    Returns the MOID and the eccentric anomalies of the closest points on both orbits. """

    n = a1.shape[0]
    b1 = a1*np.sqrt(1 - e1**2)
    b2 = a2*np.sqrt(1 - e2**2)

    # Column views for broadcasting against the samples
    c = lambda x: x[:, None]
    a1c, b1c, e1c, P1c, Q1c = c(a1), c(b1), c(e1), c(P1), c(Q1)
    a2c, b2c, e2c, P2c, Q2c = c(a2), c(b2), c(e2), c(P2), c(Q2)

    # 1. Samples of the first orbit
    u_scan = np.broadcast_to(np.linspace(0, 2*np.pi, n_scan, endpoint=False), (n, n_scan))
    r1, _, _ = _ellipsePoint(a1c, b1c, e1c, P1c, Q1c, u_scan)

    # 2. Closest point of the second ellipse to each sample: start from the sample's direction in the
    #   second orbit's plane, measured from the ellipse centre (exact for a circle), then Newton on the
    #   derivative of the squared distance
    from_centre = r1 + (a2*e2)[:, None, None]*P2c
    v_scan = np.arctan2(_dot(from_centre, Q2c)/b2c, _dot(from_centre, P2c)/a2c)
    for _ in range(n_inner):
        r2, dr2, d2r2 = _ellipsePoint(a2c, b2c, e2c, P2c, Q2c, v_scan)
        d = r2 - r1
        g = _dot(d, dr2)
        h = _dot(dr2, dr2) + _dot(d, d2r2)
        v_scan = v_scan - np.where(h > 0, g/np.where(h > 0, h, 1.0), 0.0)
    r2, _, _ = _ellipsePoint(a2c, b2c, e2c, P2c, Q2c, v_scan)
    dist2_scan = _dot(r2 - r1, r2 - r1)

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
        p1, dp1, d2p1 = _ellipsePoint(a1c, b1c, e1c, P1c, Q1c, u)
        p2, dp2, d2p2 = _ellipsePoint(a2c, b2c, e2c, P2c, Q2c, v)
        d = p1 - p2
        fu, fv = _dot(d, dp1), -_dot(d, dp2)
        fuu = _dot(dp1, dp1) + _dot(d, d2p1)
        fvv = _dot(dp2, dp2) - _dot(d, d2p2)
        fuv = -_dot(dp1, dp2)
        det = fuu*fvv - fuv**2
        ok = (det > 0) & (fuu > 0)
        safe = np.where(ok, det, 1.0)
        u = u - np.where(ok, (fvv*fu - fuv*fv)/safe, 0.0)
        v = v - np.where(ok, (fuu*fv - fuv*fu)/safe, 0.0)

    p1, _, _ = _ellipsePoint(a1c, b1c, e1c, P1c, Q1c, u)
    p2, _, _ = _ellipsePoint(a2c, b2c, e2c, P2c, Q2c, v)
    dist2 = np.where(valid, _dot(p1 - p2, p1 - p2), np.inf)

    # The refined minimum, never worse than the scan itself
    k = np.argmin(dist2, axis=1)
    best = dist2[np.arange(n), k]
    k_scan = np.argmin(dist2_scan, axis=1)
    scan_best = dist2_scan[np.arange(n), k_scan]
    use_scan = scan_best < best

    moid = np.sqrt(np.where(use_scan, scan_best, best))
    E1 = np.where(use_scan, u_scan[np.arange(n), k_scan], u[np.arange(n), k]) % (2*np.pi)
    E2 = np.where(use_scan, v_scan[np.arange(n), k_scan], v[np.arange(n), k]) % (2*np.pi)

    return moid, E1, E2


def calcMOID(a1, e1, i1, node1, peri1, a2, e2, i2, node2, peri2, return_anomalies=False,
             n_scan=MOID_N_SCAN, n_candidates=MOID_N_CANDIDATES):
    """ Minimum orbit intersection distance between two elliptical orbits.

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

    arrays = np.broadcast_arrays(*[np.asarray(x, dtype=np.float64)
                                   for x in (a1, e1, i1, node1, peri1, a2, e2, i2, node2, peri2)])
    shape = arrays[0].shape
    a1, e1, i1, node1, peri1, a2, e2, i2, node2, peri2 = [x.ravel() for x in arrays]

    moid = np.full(a1.shape, np.nan)
    E1 = np.full(a1.shape, np.nan)
    E2 = np.full(a1.shape, np.nan)

    good = np.ones(a1.shape, dtype=bool)
    for x in (a1, e1, i1, node1, peri1, a2, e2, i2, node2, peri2):
        good &= np.isfinite(x)
    good &= (a1 > 0) & (a2 > 0) & (e1 >= 0) & (e1 < 1) & (e2 >= 0) & (e2 < 1)

    idx = np.flatnonzero(good)
    for start in range(0, len(idx), MOID_CHUNK):
        sel = idx[start:start + MOID_CHUNK]
        P1, Q1 = _orbitFrame(i1[sel], node1[sel], peri1[sel])
        P2, Q2 = _orbitFrame(i2[sel], node2[sel], peri2[sel])
        moid[sel], E1[sel], E2[sel] = _moidChunk(a1[sel], e1[sel], P1, Q1, a2[sel], e2[sel], P2, Q2,
                                                 n_scan, n_candidates, MOID_N_INNER, MOID_N_NEWTON)

    def shaped(x):
        x = x.reshape(shape)
        return float(x) if x.ndim == 0 else x

    if return_anomalies:
        return shaped(moid), shaped(E1), shaped(E2)

    return shaped(moid)
