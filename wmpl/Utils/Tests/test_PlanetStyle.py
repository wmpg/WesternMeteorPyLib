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


def _oklab(hex_color, cvd=None):
    """ OKLab of a colour, optionally as seen with protanopia or deuteranopia (Machado et al. 2009,
    severity 1). """

    import numpy as np

    c = np.array([int(hex_color[i:i + 2], 16)/255.0 for i in (1, 3, 5)])
    rgb = np.where(c <= 0.04045, c/12.92, ((c + 0.055)/1.055)**2.4)
    machado = {"protan": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216],
                          [-0.003882, -0.048116, 1.051998]],
               "deutan": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413],
                          [-0.011820, 0.042940, 0.968881]]}
    if cvd is not None:
        rgb = np.clip(np.array(machado[cvd]) @ rgb, 0.0, 1.0)
    lms = np.cbrt(np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                            [0.2119034982, 0.6806995451, 0.1073969566],
                            [0.0883024619, 0.2817188376, 0.6299787005]]) @ rgb)

    return np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                     [1.9779984951, -2.4285922050, 0.4505937099],
                     [0.0259040371, 0.7827717662, -0.8086757660]]) @ lms


def testTheWarmPlanetsTellApart():
    """ Mars, Jupiter, Saturn and Venus are 15 apart in OKLab (x100) with normal vision, and 7.5 for
    protanopia and deuteranopia. """

    import itertools
    import numpy as np

    bodies = ["Venus", "Mars", "Jupiter", "Saturn"]
    for x, y in itertools.combinations(bodies, 2):
        assert 100*np.linalg.norm(_oklab(PLANET_COLORS[x]) - _oklab(PLANET_COLORS[y])) >= 15.0, (x, y)
        for cvd in ("protan", "deutan"):
            d = 100*np.linalg.norm(_oklab(PLANET_COLORS[x], cvd) - _oklab(PLANET_COLORS[y], cvd))
            assert d >= 7.5, (x, y, cvd)
