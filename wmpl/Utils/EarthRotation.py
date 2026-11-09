""" The difference between UT1, which the rotation of the Earth follows, and UTC, in which the times of meteor
    observations are given, and the conversion between the ECI coordinates of wmpl and the inertial frame.

The ECI coordinates of wmpl (geo2Cartesian, cartesian2Geo, altAz2RADec, raDec2AltAz) turn with the Earth by
the sidereal time of the Julian date they are given, which is UTC. The Earth turns by UT1, so these
coordinates are turned from the inertial frame, that of the stars, by the rotation of the Earth in UT1 - UTC:
up to 13.5 arcsec, as |UT1 - UTC| < 0.9 s. Positions on the Earth are converted exactly in and out of them,
but directions measured against the stars, and the radiant and orbit that are given against them, have to
be turned by that angle (eciToInertial, inertialToECI).

References:
    IERS Conventions (2010), IERS Technical Note 36, chapter 5. The tables of UT1 - UTC are those of the
    IERS (Bulletins A and B), read with astropy.utils.iers.
"""

from __future__ import print_function, division, absolute_import

import os

import numpy as np


# Rotation of the Earth in one second of UT1 (rad), from the Earth rotation angle (IERS Conventions 2010,
#   eq. 5.15)
EARTH_ROTATION_RATE = 2*np.pi*1.00273781191135448/86400.0

# Environment variable which, when set, gives UT1 - UTC in seconds instead of the IERS tables
DUT1_ENV = "WMPL_UT1_UTC"

# UT1 - UTC already looked up, by Julian date rounded to 0.001 days
_DUT1_CACHE = {}



def _lookupIERS(jd):
    """ Look up UT1 - UTC in the IERS tables of astropy.

    The final values of Bulletin B shipped with astropy are used when they cover the date, so that no file
    is downloaded for past events. Later dates are read from Bulletin A, which astropy downloads when its
    copy is older than 30 days, with the predictions of the coming year.

    Arguments:
        jd: [float] Julian date (UTC).

    Return:
        [float] UT1 - UTC (s), or None if the tables do not cover the date or cannot be read.
    """

    try:
        from astropy.utils import iers

        mjd = jd - 2400000.5

        # Final values
        table = iers.IERS_B.open()
        if table['MJD'][0].value <= mjd <= table['MJD'][-1].value:
            return float(table.ut1_utc(jd, 0.0).to_value('s'))

        # Rapid values and predictions
        table = iers.IERS_Auto.open()
        dut1, status = table.ut1_utc(jd, 0.0, return_status=True)
        if status < 0:
            return None

        return float(dut1.to_value('s'))

    except Exception as e:
        print("UT1 - UTC could not be read from the IERS tables: {:s}".format(repr(e)))
        return None



def deltaUT1(jd):
    """ Compute UT1 - UTC at the given time.

    If the environment variable WMPL_UT1_UTC is set, its value (s) is returned instead of the IERS tables,
    e.g. 0 for observations computed with UTC as UT1, or on a computer which cannot download the tables.
    If the tables do not cover the date, 0 is returned, with a message.

    Arguments:
        jd: [float] Julian date (UTC).

    Return:
        [float] UT1 - UTC (s).
    """

    if os.environ.get(DUT1_ENV) is not None:
        return float(os.environ[DUT1_ENV])

    key = round(float(jd), 3)
    if key not in _DUT1_CACHE:

        dut1 = _lookupIERS(float(jd))

        if dut1 is None:
            print("UT1 - UTC is not known at JD {:.3f}, so it is taken as 0. The rotation of the Earth is "
                "then off by up to 13.5 arcsec. Set {:s} to give it.".format(jd, DUT1_ENV))
            dut1 = 0.0

        _DUT1_CACHE[key] = dut1

    return _DUT1_CACHE[key]



def utcRotationAngle(dut1):
    """ Compute the angle by which the Earth turns in UT1 - UTC, i.e. the angle by which the ECI coordinates
        of wmpl, turned by UTC, lag behind the inertial frame.

    Arguments:
        dut1: [float] UT1 - UTC (s).

    Return:
        [float] Angle (rad), positive towards increasing right ascension.
    """

    return EARTH_ROTATION_RATE*dut1



def _rotateAroundPole(vect, angle):
    """ Turn vectors around the Z axis, towards increasing right ascension.

    Arguments:
        vect: [ndarray] A vector (3) or vectors (N x 3).
        angle: [float] Angle (rad).

    Return:
        [ndarray] The turned vectors, in the shape of the input.
    """

    vect = np.array(vect, dtype=np.float64)
    if angle == 0:
        return vect

    cos_a, sin_a = np.cos(angle), np.sin(angle)

    rotated = np.copy(vect)
    rotated[..., 0] = cos_a*vect[..., 0] - sin_a*vect[..., 1]
    rotated[..., 1] = sin_a*vect[..., 0] + cos_a*vect[..., 1]

    return rotated



def eciToInertial(vect, dut1):
    """ Convert vectors from the ECI coordinates of wmpl, which turn with the Earth by UTC, to the inertial
        frame of the equator of date, against which the stars and the radiant are given.

    Arguments:
        vect: [ndarray] A vector (3) or vectors (N x 3) in the ECI coordinates of wmpl.
        dut1: [float] UT1 - UTC (s).

    Return:
        [ndarray] The vectors in the inertial frame.
    """

    return _rotateAroundPole(vect, utcRotationAngle(dut1))



def inertialToECI(vect, dut1):
    """ Convert vectors from the inertial frame of the equator of date, e.g. directions measured against the
        stars, to the ECI coordinates of wmpl, which turn with the Earth by UTC.

    Arguments:
        vect: [ndarray] A vector (3) or vectors (N x 3) in the inertial frame.
        dut1: [float] UT1 - UTC (s).

    Return:
        [ndarray] The vectors in the ECI coordinates of wmpl.
    """

    return _rotateAroundPole(vect, -utcRotationAngle(dut1))
