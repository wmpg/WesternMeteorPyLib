""" Tests for the gravity of the WGS84 ellipsoid, wmpl.Utils.TrajConversions.gravity. """

import math

import pytest

from wmpl.Utils.TrajConversions import EARTH, gravity


def potentialGravity(lat, h):
    """ Magnitude of the gradient of the WGS84 gravity potential (GM, J2, J4 and the centrifugal term) at
        the geodetic latitude lat and the height h above the ellipsoid. It is independent of gravity().
    """

    a = EARTH.EQUATORIAL_RADIUS
    j2, j4 = 1.08262668355e-3, -1.6196e-6
    f = 1.0 - EARTH.POLAR_RADIUS/a
    e2 = f*(2.0 - f)

    def potential(x, z):
        r = math.hypot(x, z)
        s = z/r
        p2 = (3*s**2 - 1)/2
        p4 = (35*s**4 - 30*s**2 + 3)/8
        return EARTH.GM/r*(1 - j2*(a/r)**2*p2 - j4*(a/r)**4*p4) + 0.5*EARTH.OMEGA**2*x**2

    n = a/math.sqrt(1 - e2*math.sin(lat)**2)
    x, z = (n + h)*math.cos(lat), (n*(1 - e2) + h)*math.sin(lat)
    d = 10.0

    gx = (potential(x + d, z) - potential(x - d, z))/(2*d)
    gz = (potential(x, z + d) - potential(x, z - d))/(2*d)

    return math.hypot(gx, gz)


def testGravityAtTheEquatorAndThePoles():
    assert gravity(0.0, 0.0) == pytest.approx(EARTH.G_EQUATOR, rel=1e-12)
    assert gravity(math.pi/2, 0.0) == pytest.approx(EARTH.G_POLE, rel=1e-12)
    assert gravity(-math.pi/2, 0.0) == pytest.approx(EARTH.G_POLE, rel=1e-12)


@pytest.mark.parametrize('lat_deg', [0.0, 20.0, 45.0, 70.0, 90.0])
def testSurfaceGravityMatchesTheInternationalGravityFormula(lat_deg):
    lat = math.radians(lat_deg)

    # International Gravity Formula 1980, which is not used by gravity()
    igf = 9.780318456*(1.0 + 0.0053023657*math.sin(lat)**2 - 0.0000058498*math.sin(2.0*lat)**2)

    assert gravity(lat, 0.0) == pytest.approx(igf, abs=5e-5)


def testGravityIsSymmetricInLatitudeAndFallsWithHeight():
    for lat in (0.1, 0.8, 1.4):
        assert gravity(lat, 20000.0) == gravity(-lat, 20000.0)
        assert gravity(lat, 80000.0) < gravity(lat, 20000.0) < gravity(lat, 0.0)

    assert gravity(0.0, 50000.0) < gravity(0.8, 50000.0) < gravity(math.pi/2, 50000.0)


@pytest.mark.parametrize('lat_deg', [0.0, 10.0, 45.0, 60.0, 89.0])
@pytest.mark.parametrize('height', [10000.0, 30000.0, 100000.0])
def testFallOfGravityWithHeightMatchesTheEllipsoidalPotential(lat_deg, height):
    """ The constant 9.81 m/s^2 with an inverse square law and the mean Earth radius is off by up to
        2.4e-4 m/s^2 at 30 km, worst at the equator. The tolerance here is 8 times tighter, and also covers the
        J6 and higher terms the reference leaves out.
    """

    lat = math.radians(lat_deg)

    change = gravity(lat, height) - gravity(lat, 0.0)
    expected = potentialGravity(lat, height) - potentialGravity(lat, 0.0)

    assert change == pytest.approx(expected, abs=3e-5 if height < 50000 else 2e-4)
