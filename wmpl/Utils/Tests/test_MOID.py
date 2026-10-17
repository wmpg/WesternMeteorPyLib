""" Tests for the minimum orbit intersection distance in wmpl.Utils.MOID.

Analytic cases (coplanar circles, inclined circles, a coplanar ellipse touching a circle), the symmetry
between the two orbits, the handling of non-elliptical orbits and of array shapes, and a comparison with
an independent reference (a dense grid on both orbits refined with a tight minimiser) on random
meteoroid-like orbits, including the hard case of Earth-crossing orbits inclined by less than 2 deg.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_MOID.py -v
"""

import numpy as np
import pytest

from wmpl.Utils.MOID import calcMOID, _orbitFrame, _ellipsePoint


DEG = np.pi/180

# The Earth's orbit (J2000 ecliptic): a, e, i, node, argument of perihelion
EARTH = (1.0, 0.0167, 0.0, 0.0, 102.9*DEG)


def _reference(el1, el2, n=1500, n_starts=8):
    """ Independent MOID of one pair: dense grid on both orbits, then Nelder-Mead from the deepest
    distinct grid points. """

    from scipy.optimize import minimize

    def frame(el):
        a, e, i, node, peri = el
        P, Q = _orbitFrame(i, node, peri)
        return a, a*np.sqrt(1 - e**2), e, P, Q

    f1, f2 = frame(el1), frame(el2)
    E = np.linspace(0, 2*np.pi, n, endpoint=False)
    r1 = _ellipsePoint(*f1, E)[0]
    r2 = _ellipsePoint(*f2, E)[0]
    d2 = np.sum((r1[:, None, :] - r2[None, :, :])**2, axis=-1)

    def f(x):
        return np.sum((_ellipsePoint(*f1, x[0])[0] - _ellipsePoint(*f2, x[1])[0])**2)

    best, seen = np.inf, []
    for k in np.argsort(d2, axis=None)[:4000]:
        iu, iv = np.unravel_index(k, d2.shape)
        if any((abs(iu - su) < n//20) and (abs(iv - sv) < n//20) for su, sv in seen):
            continue
        seen.append((iu, iv))
        res = minimize(f, [E[iu], E[iv]], method="Nelder-Mead",
                       options={"xatol": 1e-12, "fatol": 1e-22, "maxiter": 4000})
        best = min(best, res.fun)
        if len(seen) >= n_starts:
            break

    return np.sqrt(best)


def testCoplanarCirclesAreTheirRadiusDifference():

    assert calcMOID(1.0, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0) == pytest.approx(0.5, abs=1e-12)
    assert calcMOID(5.2, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.3, 0.7) == pytest.approx(4.2, abs=1e-12)


def testInclinedCirclesOfOneRadiusIntersect():
    """ Two circles of the same radius in different planes cross at the line of nodes. """

    assert calcMOID(1.0, 0.0, 30*DEG, 40*DEG, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0) == pytest.approx(0.0, abs=1e-10)


def testInclinedCirclesOfDifferentRadii():
    """ Concentric circles of radii r1 < r2 inclined by I: the closest points are on the line of nodes,
    where the distance is r2 - r1. """

    assert calcMOID(1.0, 0.0, 20*DEG, 0.0, 0.0, 1.3, 0.0, 0.0, 0.0, 0.0) == pytest.approx(0.3, abs=1e-10)


def testCoplanarEllipseInsideACircle():
    """ A coplanar ellipse inside a circle of radius R is closest at its aphelion: R - a(1 + e). """

    moid, E1, _ = calcMOID(1.0, 0.5, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, return_anomalies=True)
    assert moid == pytest.approx(2.0 - 1.5, abs=1e-12)
    assert E1 == pytest.approx(np.pi, abs=1e-6)


def testTheMoidIsSymmetric():

    rng = np.random.default_rng(3)
    a, e = rng.uniform(1, 4, 50), rng.uniform(0, 0.9, 50)
    i, node, peri = rng.uniform(0, np.pi/2, 50), rng.uniform(0, 2*np.pi, 50), rng.uniform(0, 2*np.pi, 50)
    m12 = calcMOID(a, e, i, node, peri, *EARTH)
    m21 = calcMOID(*EARTH, a, e, i, node, peri)
    assert m12 == pytest.approx(m21, abs=1e-12)


def testNonEllipticalOrbitsAndShapes():

    # Hyperbolic, parabolic, negative a and NaN give NaN; the others are computed
    e = np.array([0.5, 1.0, 1.2, 0.5, np.nan])
    a = np.array([2.0, 2.0, 2.0, -2.0, 2.0])
    moid = calcMOID(a, e, 0.1, 0.0, 0.0, *EARTH)
    assert np.isfinite(moid[0]) and np.all(np.isnan(moid[1:]))

    # Scalars give a float, arrays keep their broadcast shape
    assert isinstance(calcMOID(2.0, 0.5, 0.1, 0.0, 0.0, *EARTH), float)
    grid = calcMOID(np.full((3, 4), 2.0), 0.5, 0.1, np.linspace(0, 1, 4), 0.0, *EARTH)
    assert grid.shape == (3, 4)


def testChunksGiveTheSameResult(monkeypatch):

    import wmpl.Utils.MOID as moid_module

    rng = np.random.default_rng(4)
    a, e = rng.uniform(1, 4, 37), rng.uniform(0, 0.9, 37)
    i, node, peri = rng.uniform(0, 0.5, 37), rng.uniform(0, 2*np.pi, 37), rng.uniform(0, 2*np.pi, 37)
    whole = calcMOID(a, e, i, node, peri, *EARTH)
    monkeypatch.setattr(moid_module, "MOID_CHUNK", 5)
    assert calcMOID(a, e, i, node, peri, *EARTH) == pytest.approx(whole, abs=0)


@pytest.mark.parametrize("seed, low_inclination", [(5, True), (1, False)])
def testAgreesWithAnIndependentReference(seed, low_inclination):
    """ Random Earth-crossing and near-crossing orbits against the dense-grid reference. With orbits
    inclined by less than 2 deg the minima are narrow and, near tangency, two of them can fall within
    one sampling step: the hard case for a scan. """

    rng = np.random.default_rng(seed)
    n = 40
    a = rng.uniform(1.0, 4.0, n)
    q = rng.uniform(0.3, 1.05, n)
    e = 1 - q/a
    incl = (10**rng.uniform(-2, np.log10(2.0), n) if low_inclination else rng.uniform(0, 90, n))*DEG
    node, peri = rng.uniform(0, 2*np.pi, n), rng.uniform(0, 2*np.pi, n)

    moid = calcMOID(a, e, incl, node, peri, *EARTH)
    ref = np.array([_reference((a[k], e[k], incl[k], node[k], peri[k]), EARTH) for k in range(n)])
    assert moid == pytest.approx(ref, abs=1e-9)


def testTwoEccentricOrbitsInEitherOrder():
    """ Neither orbit nearly circular (as for a meteoroid and a comet), prograde and retrograde: the
    closest point is then solved on an eccentric ellipse, whichever orbit is given second. """

    rng = np.random.default_rng(21)
    n = 30

    def orbits():
        a, e = rng.uniform(0.8, 6.0, n), rng.uniform(0.3, 0.97, n)
        incl = np.where(rng.random(n) < 0.4, rng.uniform(0, 3, n), rng.uniform(0, 170, n))*DEG
        return a, e, incl, rng.uniform(0, 2*np.pi, n), rng.uniform(0, 2*np.pi, n)

    A, B = orbits(), orbits()
    ref = np.array([_reference(tuple(x[k] for x in A), tuple(x[k] for x in B)) for k in range(n)])
    assert calcMOID(*A, *B) == pytest.approx(ref, abs=1e-9)
    assert calcMOID(*B, *A) == pytest.approx(ref, abs=1e-9)


def testNearlyTangentNearlyCoplanarOrbit():
    """ q = 0.9918 AU, on the Earth's orbit, and i = 0.057 deg: two minima closer than a sampling step.
    Refined from the scan minimum alone, without its neighbours, the result is the shallower one
    (0.000796 AU instead of 0.000325 AU). """

    el = (3.0300680554931976, 0.672668349882237, 0.05747273948057828*DEG, 59.521537106473986*DEG,
          339.0181425059184*DEG)
    moid = calcMOID(*el, *EARTH)
    assert moid == pytest.approx(_reference(el, EARTH), abs=1e-9)
    assert moid == pytest.approx(0.000325, abs=1e-6)
