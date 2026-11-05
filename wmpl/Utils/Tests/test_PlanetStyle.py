""" Tests for the shared planet colours and marker sizes in wmpl.Utils.PlanetStyle.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_PlanetStyle.py -v
"""

import pytest

from wmpl.Utils.PlanetStyle import (PLANET_COLORS, PLANET_RADII_KM, SUN_MAX_SIZE_RATIO, planetColor,
    planetMarkerSize)


def testEveryBodyHasAColourAndARadius():

    assert set(PLANET_COLORS) == set(PLANET_RADII_KM)
    assert len(set(PLANET_COLORS.values())) == len(PLANET_COLORS)


def testLunaIsTheMoonAndUnknownBodiesGetTheDefault():

    assert planetColor("Luna") == planetColor("Moon")
    assert planetMarkerSize("Luna") == planetMarkerSize("Moon")
    assert planetColor("Pluto") == "0.3" and planetColor("Pluto", default="k") == "k"
    assert planetMarkerSize("Pluto", earth_size=5.0) == 5.0


def testSizesFollowTheRadii():
    """ The sizes keep the order of the real radii, the Earth is at earth_size, and the Sun is capped
    above Jupiter. """

    assert planetMarkerSize("Earth", earth_size=5.0) == pytest.approx(5.0)

    planets = [b for b in PLANET_RADII_KM if b != "Sun"]
    by_radius = sorted(planets, key=lambda b: PLANET_RADII_KM[b])
    by_size = sorted(planets, key=planetMarkerSize)
    assert by_radius == by_size

    sun = planetMarkerSize("Sun", earth_size=1.0)
    assert sun == pytest.approx(SUN_MAX_SIZE_RATIO)
    assert sun > planetMarkerSize("Jupiter", earth_size=1.0)
