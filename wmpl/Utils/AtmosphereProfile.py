""" Air density from an atmosphere profile file, read and evaluated the same way as OpenDarkflight does.

The three formats OpenDarkflight reads are supported:
    wrf: CSV with a header line and the columns height (m), temperature (K), pressure (Pa), relative humidity,
        wind speed, wind direction, wind east, wind north, wind up and density (kg/m^3). Every model profile
        OpenDarkflight downloads (ERA5, ECMWF, GEFS, HRRR, ...) is written in this format. The density column
        is used as is.
    wyoming: radiosonde sounding from the University of Wyoming website. Only the rows with all 11 values
        are used, as in OpenDarkflight.
    supracenter: whitespace-separated height (m), temperature (C), wind speed, wind direction and pressure
        (hPa), one level per line.

For the wyoming and supracenter formats the density comes from the pressure and temperature with the ideal gas
law and OpenDarkflight's gas constants. Between levels, the logarithm of the density is interpolated linearly,
also as in OpenDarkflight. The horizontal wind is interpolated as OpenDarkflight does too, component by
component with a PCHIP interpolator. Outside the height range of the file no density is returned: OpenDarkflight
extrapolates above the top of a profile, and evaluating it differently here would break the continuity the
profile is meant to give. Use a profile that covers the heights needed instead.
"""

import hashlib

import numpy as np
import scipy.interpolate
import scipy.optimize

from wmpl.Utils.AtmosphereDensity import atmDensPoly


# Gas constant and molar mass of air used by OpenDarkflight
R_GAS = 8.314472 # J/(mol K)
M_AIR = 0.0289644 # kg/mol

PROFILE_TYPES = ['wrf', 'wyoming', 'supracenter']



def densityFromPressure(pressure, temperature):
    """ Air density from pressure and temperature with the ideal gas law.

    Arguments:
        pressure: [float or ndarray] Pressure (hPa).
        temperature: [float or ndarray] Temperature (K).

    Return:
        [float or ndarray] Air density (kg/m^3).
    """

    return 100*np.asarray(pressure)*M_AIR/(R_GAS*np.asarray(temperature))



def _readWrf(path):

    heights, densities, speeds, directions = [], [], [], []

    with open(path) as f:

        # Skip the header
        next(f)

        for line in f:

            values = line.strip().split(',')
            if values == ['']:
                continue

            if len(values) != 10:
                raise ValueError("Expected 10 comma-separated columns in the wrf profile {:s}, got {:d}: {:s}"\
                    .format(path, len(values), line.strip()))

            heights.append(float(values[0]))
            densities.append(float(values[9]))
            speeds.append(float(values[4]))
            directions.append(float(values[5]))

    return heights, densities, speeds, directions



def _readWyoming(path):

    heights, densities, speeds, directions = [], [], [], []
    data_start = False
    speed_mult = 0.514

    with open(path) as f:

        for line in f:

            values = line.split()

            # The data starts at the first row of 11 values that begins with two numbers. That row itself is
            #   skipped, as in OpenDarkflight
            if not data_start:

                # Wind speeds are in knots, unless the header gives them in m/s
                if "SKNT" in values:
                    speed_mult = 0.514
                elif "SPED" in values:
                    speed_mult = 1.0

                if len(values) == 11:
                    try:
                        float(values[0])
                        float(values[1])
                        data_start = True
                    except ValueError:
                        pass

                continue

            # An empty line ends the data
            if not values:
                break

            if len(values) == 11:
                pressure, height, temp = float(values[0]), float(values[1]), float(values[2]) + 273.15
                heights.append(height)
                densities.append(densityFromPressure(pressure, temp))
                speeds.append(float(values[7])*speed_mult)
                directions.append(float(values[6]))

    return heights, densities, speeds, directions



def _readSupracenter(path):

    heights, densities, speeds, directions = [], [], [], []

    with open(path) as f:

        for line in f:

            values = line.split()
            if not values:
                continue

            height, temp, speed, direction, pressure = [float(val) for val in values]
            heights.append(height)
            densities.append(densityFromPressure(pressure, temp + 273.15))
            speeds.append(speed)
            directions.append(direction)

    return heights, densities, speeds, directions



class AtmosphereProfile(object):
    """ Air density from an atmosphere profile file (see the module docstring). """

    def __init__(self, path, profile_type='wrf'):
        """
        Arguments:
            path: [str] Path to the profile file.

        Keyword arguments:
            profile_type: [str] Format of the file: 'wrf' (default), 'wyoming' or 'supracenter'.
        """

        if profile_type not in PROFILE_TYPES:
            raise ValueError("Unknown atmosphere profile type '{:s}', it has to be one of: {:s}".format( \
                profile_type, ", ".join(PROFILE_TYPES)))

        self.path = path
        self.profile_type = profile_type

        reader = {'wrf': _readWrf, 'wyoming': _readWyoming, 'supracenter': _readSupracenter}[profile_type]
        heights, densities, speeds, directions = [np.array(arr) for arr in reader(path)]

        if len(heights) < 2:
            raise ValueError("The atmosphere profile {:s} has fewer than two levels".format(path))

        # Keep the first level of each height, sorted by height, as OpenDarkflight does
        heights, indices = np.unique(heights, return_index=True)
        self.heights = heights
        self.densities = densities[indices]

        # Components of the vector towards where the wind blows from (east, north), interpolated as in
        #   OpenDarkflight
        directions = np.radians(directions[indices])
        self._wind_from = scipy.interpolate.PchipInterpolator(self.heights, \
            np.column_stack([speeds[indices]*np.sin(directions), speeds[indices]*np.cos(directions)]))

        # The wind can be switched off to compare with a still atmosphere
        self.use_winds = True

        self.ht_min = self.heights[0]
        self.ht_max = self.heights[-1]

        with open(path, 'rb') as f:
            self.sha256 = hashlib.sha256(f.read()).hexdigest()


    def checkCoverage(self, ht_min, ht_max):
        """ Raise a ValueError if the profile does not cover the given height range (m). """

        if (ht_min < self.ht_min) or (ht_max > self.ht_max):
            raise ValueError("The atmosphere profile {:s} covers {:.2f} to {:.2f} km, but {:.2f} to {:.2f} km "
                "are needed".format(self.path, self.ht_min/1000, self.ht_max/1000, ht_min/1000, ht_max/1000))


    def density(self, height):
        """ Air density (kg/m^3) at the given height (m), interpolating the logarithm of the density. """

        height = np.asarray(height, dtype=float)
        self.checkCoverage(np.min(height), np.max(height))

        rho = np.exp(np.interp(height, self.heights, np.log(self.densities)))

        return float(rho) if rho.ndim == 0 else rho


    def wind(self, height):
        """ Horizontal wind velocity (m/s), the direction the air moves in, at the given height (m).

        Return:
            [ndarray] East and north components, with a leading axis for an array of heights.
        """

        height = np.asarray(height, dtype=float)
        self.checkCoverage(np.min(height), np.max(height))

        return -self._wind_from(height)


    def fitPoly(self, ht_min, ht_max):
        """ Fit the 7th order polynomial of AtmosphereDensity.atmDensPoly() to the profile, as fitAtmPoly()
            does to the MSIS model, for the ablation simulation.

        Arguments:
            ht_min: [float] Bottom of the height range (m).
            ht_max: [float] Top of the height range (m).

        Return:
            (dens_co, max_rel_err): [tuple]
                dens_co: [ndarray] Polynomial coefficients.
                max_rel_err: [float] Largest relative difference between the polynomial and the profile over
                    the height range.
        """

        self.checkCoverage(ht_min, ht_max)

        # Fit on a dense grid plus the profile's own levels in the range
        height_arr = np.linspace(ht_min, ht_max, 200)
        inside = (self.heights > ht_min) & (self.heights < ht_max)
        height_arr = np.sort(np.concatenate([height_arr, self.heights[inside]]))

        dens_log = np.log10(self.density(height_arr))

        def atmDensPolyLog(height_arr, *dens_co):
            return np.log10(atmDensPoly(height_arr, dens_co))

        dens_co, _ = scipy.optimize.curve_fit(atmDensPolyLog, height_arr, dens_log, p0=np.zeros(7), \
            maxfev=10000)

        max_rel_err = np.max(np.abs(10**(atmDensPolyLog(height_arr, *dens_co) - dens_log) - 1))

        return dens_co, max_rel_err
