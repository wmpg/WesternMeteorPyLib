""" How the Sun, the planets and the Moon are drawn in wmpl figures: one colour per body, close to how
it looks, and a marker size that grows with its real radius. Shared by the orbit plots of the
trajectory solver (wmpl.Utils.PlotOrbits) and the close-encounter panel of wmpl.Rebound.REBOUND, so a
body looks the same in both.
"""

from __future__ import print_function, division, absolute_import


# Colour of each body, close to its appearance: grey Mercury, pale golden Venus, blue Earth, grey
#   Moon, rust Mars, tan Jupiter, pale gold Saturn, pale cyan Uranus and deep blue Neptune
PLANET_COLORS = {
    "Sun": "#ffcc33",
    "Mercury": "#9e9a94",
    "Venus": "#e3c16f",
    "Earth": "#2a7bd6",
    "Moon": "#c4c4c4",
    "Mars": "#c1440e",
    "Jupiter": "#d39c64",
    "Saturn": "#e4cf8c",
    "Uranus": "#9fe0e6",
    "Neptune": "#3b5fe0",
}

# Mean radius in km (IAU)
PLANET_RADII_KM = {
    "Sun": 695700.0,
    "Mercury": 2439.7,
    "Venus": 6051.8,
    "Earth": 6371.0,
    "Moon": 1737.4,
    "Mars": 3389.5,
    "Jupiter": 69911.0,
    "Saturn": 58232.0,
    "Uranus": 25362.0,
    "Neptune": 24622.0,
}

# Thin dark outline that keeps the pale bodies (Moon, Saturn, Uranus, Venus) visible on a light
#   background; on a dark one the face colour is used instead
PLANET_EDGE_COLOR = "black"
PLANET_EDGE_WIDTH = 0.5

# Marker sizes grow as radius**PLANET_SIZE_EXPONENT, a compression of the real ratios (Jupiter is 11
#   Earth radii, the Moon 0.27) that keeps the Moon visible and Jupiter from covering the panel: the
#   order is kept, with Jupiter drawn 2.1 times the Earth, Mars 0.83 and the Moon 0.67 times
PLANET_SIZE_EXPONENT = 0.3

# The Sun is drawn at most this many times the Earth's size, a little above Jupiter
SUN_MAX_SIZE_RATIO = 2.5

# Alternative names of the bodies (REBOUND calls the Moon "Luna")
_ALIASES = {"Luna": "Moon"}


def _canonical(body):
    return _ALIASES.get(body, body)


def planetColor(body, default="0.3"):
    """ Colour of a body by name (e.g. "Earth", "Luna"), or the default for an unknown one. """

    return PLANET_COLORS.get(_canonical(body), default)


def planetMarkerSize(body, earth_size=6.0):
    """ Marker size (diameter, in points) of a body, scaled to its real radius (see
    PLANET_SIZE_EXPONENT), with the Earth at earth_size. For a scatter area use its square.

    Arguments:
        body: [str] Body name, e.g. "Jupiter" or "Luna".

    Keyword arguments:
        earth_size: [float] Size of the Earth. Default 6.

    Return:
        [float] The size, or earth_size for an unknown body.
    """

    radius = PLANET_RADII_KM.get(_canonical(body))
    if radius is None:
        return earth_size

    ratio = (radius/PLANET_RADII_KM["Earth"])**PLANET_SIZE_EXPONENT
    if _canonical(body) == "Sun":
        ratio = min(ratio, SUN_MAX_SIZE_RATIO)

    return earth_size*ratio
