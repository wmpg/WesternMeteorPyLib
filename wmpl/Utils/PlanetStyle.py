""" How the Sun, the planets and the Moon are drawn in wmpl figures: one colour per body, close to how
it looks, and a marker size that grows with its real radius. Shared by the orbit plots of the
trajectory solver (wmpl.Utils.PlotOrbits) and the close-encounter panel of wmpl.Rebound.REBOUND, so a
body looks the same in both.
"""

from __future__ import print_function, division, absolute_import


# Colour of each body, close to its appearance and with distinct hues so the bodies tell apart: yellow
#   Sun, grey Mercury, amber Venus, blue Earth, light grey Moon, red Mars, brown Jupiter, olive gold
#   Saturn, cyan Uranus and violet Neptune. Mars, Jupiter, Saturn and Venus are at least 15 apart in
#   OKLab (x100), and at least 7.9 for protanopia and deuteranopia (Machado et al. 2009), where a
#   burnt orange Jupiter was 10 from the red Mars (2 for deuteranopia) and a sand Saturn 10 from Venus
PLANET_COLORS = {
    "Sun": "#f0dc5a",
    "Mercury": "#7f7f7f",
    "Venus": "#ffb000",
    "Earth": "#1f6fe0",
    "Moon": "#b0b0b0",
    "Mars": "#e0301e",
    "Jupiter": "#7d3f17",
    "Saturn": "#9e9f1e",
    "Uranus": "#4fd0d6",
    "Neptune": "#6a48d8",
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

# Thin outline that keeps the bodies visible against the background: black on a light one, for the
#   pale bodies (Sun, Moon, Uranus), and white on a dark one, for the dark bodies (Jupiter, Neptune,
#   Earth, Mars)
PLANET_EDGE_COLOR = "black"
PLANET_EDGE_COLOR_DARK = "white"
PLANET_EDGE_WIDTH = 0.5

# Marker sizes grow as radius**PLANET_SIZE_EXPONENT, a compression of the real ratios (Jupiter is 11
#   Earth radii, the Moon 0.27) that keeps the Moon visible and Jupiter from covering the panel: the
#   order is kept, with Jupiter drawn 1.8 times the Earth, Mars 0.85 and the Moon 0.72 times
PLANET_SIZE_EXPONENT = 0.25

# The Sun is drawn at most this many times the Earth's size, a little above Jupiter
SUN_MAX_SIZE_RATIO = 1.9

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
