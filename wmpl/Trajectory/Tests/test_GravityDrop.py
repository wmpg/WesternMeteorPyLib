""" Tests for the gravity drop the trajectory solver models, wmpl.Trajectory.Trajectory.applyGravityDrop, and
    its copy in the shower simulator, wmpl.TrajSim.ShowerSim.applyGravityDrop.
"""

import math

import numpy as np
import pytest

from wmpl.Trajectory.Trajectory import applyGravityDrop
from wmpl.TrajSim.ShowerSim import applyGravityDrop as applyGravityDropSim
from wmpl.Utils.TrajConversions import latLonAlt2ECEF


def radialGravitation(eci):
    """ Downward radial component of the gravitation of the Earth (WGS84 GM, J2 and J4, no centrifugal term,
        as the solver works in an inertial frame), from a numerical gradient of the potential.
    """

    gm, a, j2, j4 = 3.986004418e14, 6378137.0, 1.08262668355e-3, -1.6196e-6

    def potential(p):
        r = np.linalg.norm(p)
        s = p[2]/r
        p2 = (3*s**2 - 1)/2
        p4 = (35*s**4 - 30*s**2 + 3)/8
        return gm/r*(1 - j2*(a/r)**2*p2 - j4*(a/r)**4*p4)

    n = eci/np.linalg.norm(eci)
    d = 10.0

    return -(potential(eci + d*n) - potential(eci - d*n))/(2*d)


def dropAt(eci, t, vz, func=applyGravityDrop):
    """ Radial drop (m) applied to eci after t seconds, with r0 at the point itself. """

    r0 = np.linalg.norm(eci)

    if func is applyGravityDrop:
        new = func(eci, t, r0, 1.0, vz)
    else:
        new = func(eci, t, r0, vz)

    return np.dot(eci - new, eci/r0)


@pytest.mark.parametrize('lat_deg', [0.0, 30.0, 45.0, 60.0, 90.0, -45.0])
@pytest.mark.parametrize('height', [0.0, 100000.0])
def testDropFollowsTheRadialGravitationOfTheOblateEarth(lat_deg, height):
    """ With a small vertical speed the drop is g t^2/2, so g can be read back from it. GM/r^2 alone is off
        by -0.17% at the equator and +0.32% at the poles (up to 0.03 m/s^2); with the J2 term what is left is
        J4, below 1e-4 m/s^2.
    """

    eci = np.array(latLonAlt2ECEF(math.radians(lat_deg), 0.4, height))
    t = 2.0

    g = dropAt(eci, t, 0.0)/(0.5*t**2)

    assert g == pytest.approx(radialGravitation(eci), abs=1e-4)


def testBothDropModelsAgreeOverAShortTime():
    """ The constant vertical speed model reduces to g t^2/2 when the height barely changes. """

    eci = np.array(latLonAlt2ECEF(math.radians(55.0), 0.4, 90000.0))
    t = 0.05

    assert dropAt(eci, t, -15000.0) == pytest.approx(dropAt(eci, t, 0.0), rel=1e-3)


@pytest.mark.parametrize('vz', [0.0, -15000.0])
def testTheSimulatorAndTheSolverApplyTheSameDrop(vz):
    """ Simulated trajectories are solved with the same gravity they were made with. """

    eci = np.array(latLonAlt2ECEF(math.radians(70.0), 2.0, 95000.0))

    assert dropAt(eci, 3.0, vz, applyGravityDropSim) == pytest.approx(dropAt(eci, 3.0, vz), rel=1e-12)
