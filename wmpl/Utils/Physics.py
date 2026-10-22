""" Implementations of frequently used equations. """

from __future__ import print_function, division, absolute_import

import warnings

import numpy as np
import scipy.integrate

# Import the correct scipy.integrate.simpson function
try:
    from scipy.integrate import simps as simpson
except ImportError:
    from scipy.integrate import simpson as simpson

from wmpl.Utils.AtmosphereDensity import getAtmDensity_vect


# Mapping of luminous efficiency model names to the integer lum_eff_type codes used by
# wmpl.MetSim.MetSimErosionCyTools.luminousEfficiency().
LUM_EFF_MODELS = {
    'constant':          0,
    'rc2001':            1,  # ReVelle & Ceplecha (2001), Type I
    'rc2001_t1':         1,
    'rc2001_t2':         2,
    'rc2001_t3':         3,
    'borovicka2013':     4,  # Borovicka et al. (2013), Kosice
    'kosice':            4,
    'camo':              5,  # CAMO faint meteors (Subasinghe 2018 & Brown 2020)
    'ceplecha_mccrosky': 6,  # Ceplecha & McCrosky (1976)
    'cm1976':            6,
    'borovicka2020':     7,  # Borovicka et al. (2020), Two strengths
    'pecina_ceplecha':   8,  # Pecina & Ceplecha (1983)
    'pc1983':            8,
}


def dynamicPressure(lat, lon, height, jd, velocity, gamma=1.0):
    """ Calculate dynamic pressure at the given point on meteor's trajectory. 

    Either a single set of values can be given (i.e. every argument is a float number), or all arguments 
    must be numpy arrays.
        
    Arguments:
        lat: [float] Latitude of the meteor (radians).
        lon: [float] Longitude of the meteor (radians).
        height: [float] Height of the meteor (meters).
        jd: [float] Julian date of the meteor.
        velocity: [float] Velocity of the meteor (m/s).

    Keyword arguments:
        gamma: [flot] Drag coefficient. 1 by default.


    Return:
        dyn_pressure: [float] Dynamic pressure in Pascals.

    """

    # Get the atmospheric densities at every heights
    atm_dens = getAtmDensity_vect(lat, lon, height, jd)

    # Calculate the dynamic pressure
    dyn_pressure = atm_dens*gamma*velocity**2

    return dyn_pressure


def dynamicMass(bulk_density, lat, lon, height, jd, velocity, decel, gamma=1.0, shape_factor=1.21):
    """ Calculate dynamic mass at the given point on meteor's trajectory. 
    
    Either a single set of values can be given (i.e. every argument is a float number), or all arguments 
    must be numpy arrays.
        
    Arguments:
        bulk_density: [float] Bulk density of the meteoroid in kg/m^3.
        lat: [float] Latitude of the meteor (radians).
        lon: [float] Longitude of the meteor (radians).
        height: [float] Height of the meteor (meters).
        jd: [float] Julian date of the meteor.
        velocity: [float] Velocity of the meteor (m/s).
        decel: [float] Deceleration in m/s^2.

    Keyword arguments:
        gamma: [flot] Drag coefficient. 1 by default.
        shape_factor: [float] Shape factory for the body. 1.21 (sphere) by default. Other values:
            - sphere      = 1.21
            - hemisphere  = 1.92
            - cube        = 1.0
            - brick 2:3:5 = 1.55

    Return:
        dyn_mass: [float] Dynamic mass in kg.


    """

    # Calculate the atmosphere density at the given point
    atm_dens = getAtmDensity_vect(lat, lon, height, jd)

    # Calculate the dynamic mass
    dyn_mass = (1.0/(bulk_density**2))*((gamma*shape_factor*(velocity**2)*atm_dens)/decel)**3

    return dyn_mass


def calcIntensity(mag_abs, P_0m=840.0):
    """ Calculate the radiated power (intensity) from absolute magnitudes.

    Arguments:
        mag_abs: [float or ndarray] Absolute magnitudes.

    Keyword arguments:
        P_0m: [float] Power of a zero-magnitude meteor (W). See calcRadiatedEnergy() for details.

    Return:
        intens: [float or ndarray] Radiated power (W).

    """

    return P_0m*10**(-0.4*mag_abs)


def calcRadiatedEnergy(time, mag_abs, P_0m=840.0):
    """ Calculate the radiated energy given the light curve.

    Arguments:
        time: [ndarray] Time of individual magnitude measurement (s).
        mag_abs: [nadrray] Absolute magnitudes (i.e. apparent meteor magnitudes @100km).

    Keyword arguments:
        P_0m: [float] Power output of a zero absolute magnitude meteor. 840W by default, as that is the R
            bandpass for a T = 4500K black body meteor. See: Weryk & Brown, 2013 - "Simultaneous radar and 
            video meteors - II. Photometry and ionisation" for more details.

    Return:
        initens_int: [float] Total radiated energy (J).

    """


    # Calculate the intensities from absolute magnitudes
    intens = calcIntensity(mag_abs, P_0m=P_0m)

    # # Interpolate I
    # intens_interpol = scipy.interpolate.PchipInterpolator(time, intens)

    # x_data = np.linspace(np.min(time), np.max(time), 1000)
    # plt.plot(x_data, intens_interpol(x_data))
    # plt.scatter(time, intens/(velocity**2))
    # plt.show()

    # Integrate the interpolated I
    #intens_int = intens_interpol.integrate(np.min(time), np.max(time))
    intens_int = simpson(intens, x=time)

    return intens_int


def calcMass(time, mag_abs, velocity, tau=0.007, P_0m=840.0, lum_eff_mass=-1, v_init=None):
    """ Calculates the mass of a meteoroid from the time and absolute magnitude.

    Arguments:
        time: [ndarray] Time of individual magnitude measurement (s).
        mag_abs: [ndarray] Absolute magnitudes (i.e. apparent meteor magnitudes @100km).
        velocity: [float or ndarray] Average velocity of the meteor, or velocity at every point of the meteor
            in m/s.

    Keyword arguments:
        tau: [float or int or str] Luminous efficiency selector.
            - float: constant luminous efficiency as a ratio (not percent!). 0.007 (i.e. 0.7%) by default
              (Ceplecha & McCrosky, 1976).
            - int: velocity-dependent lum_eff_type code passed to
              wmpl.MetSim.MetSimErosionCyTools.luminousEfficiency() (0 - constant, 1/2/3 - ReVelle &
              Ceplecha 2001 Type I/II/III, 4 - Borovicka 2013, 5 - CAMO, 6 - Ceplecha & McCrosky 1976,
              7 - Borovicka 2020, 8 - Pecina & Ceplecha 1983). tau is then computed per velocity point.
            - str: model name mapped to the above codes (see LUM_EFF_MODELS), e.g. 'rc2001'.
        P_0m: [float] Power output of a zero absolute magnitude meteor. 840W by default, as that is the R
            bandpass for a T = 4500K black body meteor. See: Weryk & Brown, 2013 - "Simultaneous radar and
            video meteors - II. Photometry and ionisation" for more details.
            NOTE: the panchromatic models (RC2001, Ceplecha & McCrosky 1976, Borovicka,
            Pecina-Ceplecha) are paired with a zero-magnitude power of P_0m = 1500 W (Borovicka et
            al. 2022; the cm1976 law has 1500 W baked into its unit conversion in
            luminousEfficiency()); set P_0m accordingly when using those models. A warning is
            raised if a different P_0m is used with them.
        lum_eff_mass: [float] Meteoroid mass (kg) used to evaluate the mass-dependent luminous efficiency
            models. Ignored for a float tau. If -1 (default), the mass is iterated to self-consistency
            (compute mass -> recompute tau -> repeat until convergence).
        v_init: [float or None] Pre-atmospheric velocity (m/s) used by the ReVelle & Ceplecha (2001)
            deceleration correction in the luminous efficiency calculation. Only affects the RC2001
            models (tau = 1, 2, 3 or 'rc2001*'). Default is None.
            - None: the deceleration correction is not applied.
            - -1: estimate the pre-atmospheric velocity as the median velocity over the first 25% of
              points (robust to measurement noise). Requires a velocity array.
            - >0: use the supplied value as the pre-atmospheric velocity.
            The correction is held at its dv = 0.1 km/s value for velocities within 0.1 km/s of
            v_init or above it (see luminousEfficiency() for details).
            NOTE: the velocity array should represent a fitted deceleration curve (i.e. a physically
            consistent monotonic velocity evolution) rather than individual noisy measurements.

    Return:
        mass: [float] Photometric mass of the meteoroid in kg.

    """

    # Theory:
    # I = P_0m*10^(-0.4*M_abs)
    #
    # General (velocity and/or tau vary along the trajectory):
    #   M = integral(2*I/(tau*v^2) dt)
    #
    # For constant velocity and luminous efficiency this reduces to:
    #   M = (2/(tau*v^2))*integral(I dt)
    #
    # If tau is constant but velocity varies along the trajectory:
    #   M = (2/tau)*integral(I/v^2 dt)

    if len(time) != len(mag_abs):
        raise ValueError("time and mag_abs must have the same length.")

    # If a velocity array is given, it must have the same length as the photometric measurements
    if np.ndim(velocity) > 0 and len(velocity) != len(time):
        raise ValueError("Velocity array length ({:d}) does not match the number of time/magnitude "
            "measurements ({:d}).".format(len(velocity), len(time)))

    if lum_eff_mass < 0 and lum_eff_mass != -1:
        raise ValueError("lum_eff_mass must be >= 0, or -1 for self-consistent iteration.")

    # Constant luminous efficiency (tau given directly as a ratio).
    if not isinstance(tau, str) and not isinstance(tau, (bool, int)):

        # Constant velocity
        if np.ndim(velocity) == 0:
            intens_int = calcRadiatedEnergy(time, mag_abs, P_0m=P_0m)
            return (2.0/(tau*float(velocity)**2))*intens_int

        # Velocity array
        else:
            intens = calcIntensity(mag_abs, P_0m=P_0m)
            vel_arr = np.asarray(velocity, dtype=float)
            dm_dt = 2.0*intens/(tau*vel_arr**2)
            return simpson(dm_dt, x=time)

    # Velocity-dependent model: resolve the lum_eff_type code.
    if isinstance(tau, str):
        key = tau.strip().lower()
        if key not in LUM_EFF_MODELS:
            raise ValueError("Unknown luminous efficiency model: {!r}".format(tau))
        lum_eff_type = LUM_EFF_MODELS[key]
    else:
        lum_eff_type = int(tau)

    # The panchromatic models are calibrated for a zero-magnitude power of 1500 W (see the P_0m
    # docstring note), so warn if a different value is used with them
    if (lum_eff_type in [1, 2, 3, 6, 7, 8]) and (P_0m != 1500.0):
        warnings.warn("The chosen luminous efficiency model (lum_eff_type={:d}) is calibrated for "
            "a panchromatic zero-magnitude power of P_0m = 1500 W, but P_0m = {:g} W was given. "
            "No correction is applied - the returned mass will be off by a factor of {:.2f} "
            "compared to using the calibrated value.".format(lum_eff_type, P_0m, P_0m/1500.0))

    # Compute the luminous efficiency using an analytical velocity-dependent function.
    import pyximport
    pyximport.install(setup_args={'include_dirs': [np.get_include()]})
    from wmpl.MetSim.MetSimErosionCyTools import luminousEfficiency

    # Radiated power at every measurement point
    intens = calcIntensity(mag_abs, P_0m=P_0m)


    # Convert velocity to an array for the velocity-dependent luminous efficiency calculations.
    vel_arr = np.atleast_1d(np.asarray(velocity, dtype=float))

    # Resolve the pre-atmospheric velocity used by the ReVelle & Ceplecha (2001) deceleration
    # correction (only affects the RC2001 models, types 1-3)
    if v_init is None:

        # Disable the deceleration correction
        v_init_eff = -1.0

    elif v_init == -1:

        # A scalar velocity carries no deceleration information to estimate v_init from
        if np.ndim(velocity) == 0:
            raise ValueError("v_init=-1 requires a velocity array to estimate the pre-atmospheric "
                "velocity.")

        # Estimate the pre-atmospheric velocity as the median velocity over the first 25% of points,
        # which is robust to measurement noise (unlike e.g. the maximum velocity)
        v_init_eff = float(np.median(vel_arr[:max(1, len(vel_arr)//4)]))

    elif v_init > 0:

        v_init_eff = float(v_init)

        # The deceleration correction is floored for velocities within 0.1 km/s of v_init, so warn
        # if the given v_init leaves the fastest points on the floor
        if v_init_eff < np.max(vel_arr) + 100:
            print("WARNING: v_init = {:.1f} m/s is within 100 m/s of the maximum given velocity. "
                "The ReVelle & Ceplecha (2001) deceleration correction will be held at its floor "
                "at the fastest points.".format(v_init_eff))

    else:
        raise ValueError("v_init must be None, -1 (automatic estimate), or > 0, "
            "got {}".format(v_init))

    def _mass_for(mass_assumed):
        """ Compute the photometric mass given an assumed meteoroid mass for the lum. eff. models. """

        # Vectorize the scalar Cython call over the velocity points (lum_eff arg only used for type 0)
        tau_arr = np.array([luminousEfficiency(lum_eff_type, 0.7, v, mass_assumed, v_init_eff) for v in vel_arr])

        # Integrate the instantaneous mass-loss rate dm/dt = 2*I/(tau*v^2)
        dm_dt = 2.0*intens/(tau_arr*vel_arr**2)

        return simpson(dm_dt, x=time)


    # If the mass is given explicitly, evaluate the models at that mass.
    if lum_eff_mass >= 0:
        return _mass_for(lum_eff_mass)

    # Otherwise iterate to a self-consistent mass: the luminous efficiency depends on the mass, which
    # is the quantity being solved for. Convergence is measured as a fraction of the first mass
    # estimate, so it behaves consistently across orders of magnitude (e.g. ~1e-7 kg meteoroids) and
    # does not divide by an evolving (possibly tiny) iterate.
    rel_tol = 1e-4

    # Seed the iteration with the luminous efficiency evaluated at a mass of 1 kg. The mass term of
    # the models is bounded, so this seed lands within a small factor of the true mass at any scale.
    mass = _mass_for(1.0)
    mass_ref = abs(mass)

    for _ in range(50):
        mass_new = _mass_for(mass)
        if abs(mass_new - mass) < rel_tol*mass_ref:
            mass = mass_new
            break
        mass = mass_new
    else:
        print("WARNING: luminous efficiency self-consistency iteration did not converge "
            "after 50 iterations.")

    return mass


def calcKB(lat, lon, ht_beg, jd, v, zenith_angle, gmn_correction=False):
    """ Calculate the Celpecha (1958) KB parameter.
    
    Arguments:
        lat: [float] Latitude of the meteor (radians).
        lon: [float] Longitude of the meteor (radians).
        ht_beg: [float] Beginning height of the meteor (meters).
        jd: [float] Julian date of the meteor.
        v: [float] Velocity of the meteor (m/s).
        zenith_angle: [float] Zenith angle of the meteor (radians).

    Keyword arguments:
        gmn_correction: [bool] Apply the correction for GMN data. False by default.
            The correction is described in:
            Cordonnier et al. (2024) - Meteor persisent trains

    Return:
        kb: [float] KB parameter.
    
    """

    # Get the atmospheric density at the meteor begin height
    rho_a = getAtmDensity_vect(lat, lon, ht_beg, jd)

    # Convert atmospheric density from kg/m^3 to g/cm^3
    rho_a /= 1000.0

    # Convert speed from m/s to cm/s
    v *= 100.0

    # Calculate the KB parameter
    kb = np.log10(rho_a) + 2.5*np.log10(v) - 0.5*np.log10(np.cos(zenith_angle))

    # Apply the GMN correction
    if gmn_correction:
        kb -= 0.1

    return kb


def trajectoryLightCurve(traj, mag_limit=8.0):
    """ The light curve of a solved trajectory, as AggregateAndPlot.computeMass() takes it for the photometric
        mass: the absolute magnitudes of the stations that are not ignored, brighter than mag_limit, sorted by time,
        with the points of different stations closer than a frame merged into the brightest one, which mitigates
        saturation (it assumes the photometric calibration was done well).

    Arguments:
        traj: [Trajectory] Solved trajectory.

    Keyword arguments:
        mag_limit: [float] Fainter magnitudes are left out. +8 by default.

    Return:
        (time_arr, mag_arr, ht_arr): Times (s, in the trajectory's time), absolute magnitudes and heights (m) of the
            light curve, or None if the trajectory has no magnitudes.
    """

    from wmpl.Utils.Math import mergeClosePoints

    time_mag_ht = []
    avg_t_diff_max = 0
    for obs in traj.observations:

        # Skip ignored stations, and stations without magnitudes
        if obs.ignore_station or (obs.absolute_magnitudes is None):
            continue

        # Compute average time difference
        avg_t_diff_max = max(avg_t_diff_max, np.median(obs.time_data[1:] - obs.time_data[:-1]))

        # Exclude the faint magnitudes
        mag_filter = obs.absolute_magnitudes < mag_limit
        if not np.any(mag_filter):
            continue

        for t, mag, ht in zip(obs.time_data[mag_filter], obs.absolute_magnitudes[mag_filter], \
            obs.model_ht[mag_filter]):

            if (mag is not None) and (not np.isnan(mag)):
                time_mag_ht.append([t, mag, ht])

    if len(time_mag_ht) == 0:
        return None

    time_mag_ht = np.array(sorted(time_mag_ht, key=lambda x: x[0]))

    # Take the brightest magnitudes of points closer than a frame
    time_arr, mag_arr = mergeClosePoints(time_mag_ht[:, 0], time_mag_ht[:, 1], avg_t_diff_max, method='min')
    time_arr, mag_arr = np.array(time_arr), np.array(mag_arr)

    # The height at the merged times
    ht_arr = np.interp(time_arr, time_mag_ht[:, 0], time_mag_ht[:, 2])

    return time_arr, mag_arr, ht_arr


def groundAlphaBetaFit(traj):
    """ A robust alpha-beta fit (wmpl.Utils.AlphaBeta.fitAlphaBeta(method='robust')) of the speeds of a solved
        trajectory relative to the ground (the air), on heights rescaled to the exponential atmosphere of the
        alpha-beta model, as the AlphaBeta command line does. The speeds relative to the ground are the solver's
        inertial ones minus the Earth's rotation along the path, which offsets them all alike (orbit.v_init -
        orbit.v_init_norot, as Orbit takes it for v_avg_norot), and the fit starts from orbit.v_init_norot.

    Arguments:
        traj: [Trajectory] Solved trajectory, with its orbit and point velocities.

    Return:
        velocityAt: [function] The speed relative to the ground (m/s) at the given heights (m), latitudes and
            longitudes (radians): velocityAt(hts, lats, lons).
    """

    # AlphaBeta imports this module, so it is imported here
    from wmpl.Utils.AlphaBeta import alphaBetaVelocity, fitAlphaBeta, rescaleHeightToExponentialAtmosphere

    # The Earth's rotation along the path (0 for a trajectory without the speed relative to the ground)
    v_init_norot = getattr(traj.orbit, 'v_init_norot', None)
    if v_init_norot is None:
        v_init_norot = traj.orbit.v_init
    v_rotation = traj.orbit.v_init - v_init_norot

    # The measured points, as the AlphaBeta command line takes them
    ht_data, lat_data, lon_data, vel_data = [], [], [], []
    for obs in traj.observations:

        if obs.ignore_station:
            continue

        filter_mask = (obs.ignore_list == 0) & (obs.velocities != 0)

        ht_data += obs.model_ht[filter_mask].tolist()
        lat_data += obs.model_lat[filter_mask].tolist()
        lon_data += obs.model_lon[filter_mask].tolist()
        vel_data += obs.velocities[filter_mask].tolist()

    ht_rescaled = rescaleHeightToExponentialAtmosphere(np.array(lat_data), np.array(lon_data), np.array(ht_data),
        traj.jdt_ref)

    v_init, alpha, beta = fitAlphaBeta(np.array(vel_data) - v_rotation, ht_rescaled, v_init=v_init_norot,
        method='robust')

    def velocityAt(hts, lats, lons):
        return alphaBetaVelocity(rescaleHeightToExponentialAtmosphere(lats, lons, hts, traj.jdt_ref), alpha, beta,
            v_init)

    return velocityAt


def _trajectoryLatLon(traj, hts):
    """ Latitudes and longitudes (radians) along a trajectory at the given heights, between its beginning and end. """

    frac = (np.asarray(hts) - traj.rbeg_ele)/(traj.rend_ele - traj.rbeg_ele)
    lats = traj.rbeg_lat + frac*(traj.rend_lat - traj.rbeg_lat)
    lons = traj.rbeg_lon + frac*((traj.rend_lon - traj.rbeg_lon + np.pi)%(2*np.pi) - np.pi)

    return lats, lons


def maxDynamicPressure(traj, n_heights=1000, velocity_at=None):
    """ The maximum dynamic pressure p = rho*v^2 on a meteoroid along a solved trajectory, as Borovicka et al.
        (2022, Paper I) give it for the European Fireball Network: over the whole trajectory, from its beginning to
        its end height, with the velocity fit extrapolated where there are no velocity measurements.

    The velocity, relative to the ground (the air), comes from a robust alpha-beta fit of the solver's point speeds
    (groundAlphaBetaFit()). The air density is the NRLMSISE one along the trajectory, at its reference time.

    Arguments:
        traj: [Trajectory] Solved trajectory, with its orbit and point velocities.

    Keyword arguments:
        n_heights: [int] Number of heights the pressure is evaluated at, evenly between the beginning and the end
            height. 1000 by default.
        velocity_at: [function] The fit from groundAlphaBetaFit(traj), to reuse one already made. None by default,
            to make it.

    Return:
        (p_max, ht_p_max, v_p_max): The maximum dynamic pressure (Pa), and the height (m) and the speed relative to
            the ground (m/s) where it is reached.
    """

    if velocity_at is None:
        velocity_at = groundAlphaBetaFit(traj)

    # The whole trajectory, from the beginning to the end height
    hts = np.linspace(traj.rbeg_ele, traj.rend_ele, n_heights)
    lats, lons = _trajectoryLatLon(traj, hts)

    vels = velocity_at(hts, lats, lons)
    pressures = getAtmDensity_vect(lats, lons, hts, traj.jdt_ref)*vels**2

    i_max = np.argmax(pressures)

    return pressures[i_max], hts[i_max], vels[i_max]


def photometricMassBorovicka2022(traj, velocity_at=None):
    """ The initial photometric mass of Borovicka et al. (2022, Paper I, Eqs. 1 and 2), the one their Pf takes, from
        the light curve of a solved trajectory (trajectoryLightCurve()): the luminous efficiency of ReVelle &
        Ceplecha (2001) for type I fireballs without its mass and deceleration terms, with 1500 W for magnitude zero,
        integrated with the speed relative to the ground at each point of the light curve, from the robust alpha-beta
        fit (groundAlphaBetaFit()). In calcMass(): tau='rc2001', lum_eff_mass=1.0 (where the mass term vanishes),
        v_init=None and P_0m=1500. The trajectory's absolute magnitudes are taken as Paper I's (1500 W is for the V
        band); for cameras of another band, compute the mass with their own zero-magnitude power.

    The mass is only as good as the trajectory's magnitudes: the saturated or uncalibrated video magnitudes of a
    bright fireball give one far too small (and a Pf far too high, as m^(-1/3)). Check its light curve, or give the
    mass then.

    Arguments:
        traj: [Trajectory] Solved trajectory, with magnitudes.

    Keyword arguments:
        velocity_at: [function] The fit from groundAlphaBetaFit(traj), to reuse one already made. None by default,
            to make it.

    Return:
        [float] Photometric mass (kg), or None if the trajectory has no magnitudes.
    """

    light_curve = trajectoryLightCurve(traj)
    if light_curve is None:
        return None

    time_arr, mag_arr, ht_arr = light_curve

    if velocity_at is None:
        velocity_at = groundAlphaBetaFit(traj)

    vel_arr = velocity_at(ht_arr, *_trajectoryLatLon(traj, ht_arr))

    return calcMass(time_arr, mag_arr, vel_arr, tau='rc2001', P_0m=1500.0, lum_eff_mass=1.0, v_init=None)


def averageZenithDistance(traj):
    """ The zenith distance of the ground-fixed (apparent) radiant at the average point of a trajectory, halfway in
        height between its beginning and its end, where the Earth's curvature makes it the average one along the
        path (TrajConversions.derotatedRadiantAltAz()).

    Arguments:
        traj: [Trajectory] Solved trajectory.

    Return:
        [float] Zenith distance (radians).
    """

    from wmpl.Utils.TrajConversions import derotatedRadiantAltAz, geo2Cartesian

    lat = (traj.rbeg_lat + traj.rend_lat)/2
    lon = traj.rbeg_lon + ((traj.rend_lon - traj.rbeg_lon + np.pi)%(2*np.pi) - np.pi)/2
    ht = (traj.rbeg_ele + traj.rend_ele)/2

    eci = np.array(geo2Cartesian(lat, lon, ht, traj.jdt_ref))
    _, elev, _ = derotatedRadiantAltAz(traj.v_init*traj.radiant_eci_mini, eci, traj.jdt_ref, lat, lon)

    return np.pi/2 - elev


def calcPf(p_max=None, zangle=None, mass=None, v_0=None, traj=None):
    """ The Borovicka et al. (2022) Pf parameter calculation.

        Pf = 100*p_max/(cos(z)*m_phot^(1/3)*v_inf^(3/2)), p_max in MPa, m_phot in kg, v_inf in km/s

    References:
        Borovicka et al. (2022) - Data on 824 fireballs observed by the digital cameras of the European Fireball
            Network in 2017-2018. I. (catalogue: p_max, photometric mass, v_inf), A&A 667, A157; II. (Pf, its
            classes, Eq. 2), A&A 667, A158.

    NOTE: The mass is the initial photometric mass of Paper I (Eqs. 1 and 2): the luminous efficiency of ReVelle &
    Ceplecha (2001) for type I fireballs without its mass and deceleration terms, with a zero-magnitude power of
    1500 W, integrated over the light curve with the speed of each point. In calcMass() that is tau='rc2001' with
    lum_eff_mass=1.0, where its mass term tanh(0.38 ln m) vanishes (within 0.8% of Paper I's tau at all speeds),
    v_init=None and P_0m=1500: calcMass(time, mag_abs, velocity, tau='rc2001', lum_eff_mass=1.0, P_0m=1500).
    Paper I gives these equations as the ones for a 10 kg meteoroid; with the mass term at 10 kg, tau would be
    2.25 times larger and the mass that much smaller. With a trajectory that has magnitudes, it is computed so
    (photometricMassBorovicka2022()) unless it is given.

    Keyword arguments:
        p_max: [float] Maximum dynamic pressure p = rho*v^2 (Pa). Cannot be given together with traj.
        zangle: [float] Average zenith distance of the radiant (radians). Cannot be given together with traj.
        mass: [float] Initial photometric mass (kg), see the note above. Needed without traj, or with a traj
            that has no magnitudes.
        v_0: [float] Initial (entry) velocity v_inf (m/s), relative to the ground as in Paper I. Cannot be given
            together with traj.
        traj: [Trajectory] Optional solved WMPL Trajectory. If given, p_max, zangle and v_0 are derived from it,
            and the mass too if it is not given (a warning says so, with the values):
                - p_max  <- maxDynamicPressure(traj): a robust alpha-beta fit of the speeds relative to the
                            ground (groundAlphaBetaFit()), extrapolated over the whole trajectory, with the
                            NRLMSISE density
                - zangle <- averageZenithDistance(traj): ground-fixed radiant at the average point
                - v_0    <- traj.orbit.v_init_norot, the initial speed relative to the ground
                - mass   <- photometricMassBorovicka2022(traj): from the trajectory's light curve, with the speeds
                            of the same alpha-beta fit. Only as good as the trajectory's magnitudes: the warning
                            gives the brightest one and the radiated energy, to check them

    Return:
        (pf, pf_category): [tuple]
            - pf: [float] Pf parameter.
            - pf_category: [int] Pf category (1 to 5).

    """

    # Derive p_max, zangle and v_0 from the trajectory if provided
    if traj is not None:

        # Disallow mixing traj with explicit values, as it is ambiguous which should be used
        if (p_max is not None) or (zangle is not None) or (v_0 is not None):
            raise ValueError("Give either `traj`, or explicit p_max/zangle/v_0 values, not both.")

        if getattr(traj, 'orbit', None) is None:
            raise ValueError("The given trajectory does not have a computed orbit. Pass p_max, zangle and v_0 "
                "explicitly instead.")

        # One robust alpha-beta fit for the pressure and the mass
        velocity_at = groundAlphaBetaFit(traj)

        p_max, ht_p_max, v_p_max = maxDynamicPressure(traj, velocity_at=velocity_at)
        zangle = averageZenithDistance(traj)
        v_0 = traj.orbit.v_init_norot if (getattr(traj.orbit, 'v_init_norot', None) is not None) \
            else traj.orbit.v_init

        if mass is None:
            mass = photometricMassBorovicka2022(traj, velocity_at=velocity_at)
            if mass is None:
                raise ValueError("The trajectory has no magnitudes: give the initial photometric mass.")
            time_arr, mag_arr, _ = trajectoryLightCurve(traj)
            mass_text = "; the initial photometric mass, {:.4g} kg, is computed from its light curve ({:d} points, " \
                "brightest {:+.2f} mag, {:.3g} J radiated with 1500 W) with the luminous efficiency of Paper I " \
                "(RC2001 type I without its mass term) and the speeds of the same fit. It is only as good as the " \
                "magnitudes: saturated or uncalibrated ones give a mass far too small and a Pf too high".format(mass,
                len(mag_arr), np.min(mag_arr), calcRadiatedEnergy(time_arr, mag_arr, P_0m=1500.0))
        else:
            mass_text = ""

        warnings.warn("calcPf() with a trajectory: the maximum dynamic pressure, {:.4f} MPa at {:.2f} km "
            "({:.3f} km/s), is estimated with a robust alpha-beta fit (fitAlphaBeta(method='robust')) of the "
            "observed speeds relative to the ground, extrapolated over the whole trajectory; the average zenith "
            "distance of the radiant is {:.2f} deg and v_inf {:.3f} km/s, relative to the ground{:s}.".format(
            p_max/1e6, ht_p_max/1000, v_p_max/1000, np.degrees(zangle), v_0/1000, mass_text))

    elif (p_max is None) or (zangle is None) or (v_0 is None):
        raise ValueError("Provide either `traj`, or all of p_max, zangle and v_0.")

    if mass is None:
        raise ValueError("The initial photometric mass is needed.")

    ### Convert units to match the paper ###

    # Pressure in MPa
    p_max = p_max/1e6

    # Entry speed in km/s
    v_0 = v_0/1e3

    ### ###

    # Calculate the Pf parameter
    pf = 100*p_max/(np.cos(zangle)*mass**(1/3.0)*v_0**(3/2))

    # Determine the Pf category
    if 0.85 < pf:
        pf_category = 1
    elif 0.27 < pf <= 0.85:
        pf_category = 2
    elif 0.085 < pf <= 0.27:
        pf_category = 3
    elif 0.027 < pf <= 0.085:
        pf_category = 4
    else:
        pf_category = 5


    return pf, pf_category


def calcPE(mass, rho_e=None, zangle=None, v_0=None, traj=None):
    """ The Ceplecha & McCrosky (1976) PE end-height structural criterion.

        PE = log10(rho_e) - 0.42*log10(m) + 1.49*log10(v) - 1.29*log10(cos(zangle))

    References:
        Ceplecha & McCrosky (1976) - Fireball end heights: A diagnostic for the structure of meteoric
            material, JGR 81, 6257.
        Ceplecha (1988) - Earth's influx of different populations of sporadic meteoroids from
            photographic and television data, BAICz 39, 221 (Eq. 2, group boundaries in Sec. 2e).

    NOTE: The mass MUST be the initial photometric mass computed with the Ceplecha & McCrosky (1976)
    luminous efficiency law to stay consistent with the original calibration. Compute it via
    calcMass() using tau='cm1976' (i.e. lum_eff_type=6) AND P_0m=1500, as the cm1976 law is defined
    for a panchromatic zero-magnitude power of 1500 W (see luminousEfficiency() in
    MetSimErosionCyTools.pyx).

    Arguments:
        mass: [float] Initial photometric mass m_inf (kg).

    Keyword arguments:
        rho_e: [float] Atmospheric mass density at the end height (kg/m^3). Cannot be given together
            with traj.
        zangle: [float] Radiant zenith angle (radians), measured from the local zenith
            (0 = vertical entry, pi/2 = horizontal entry). Cannot be given together with traj.
        v_0: [float] Initial (entry) velocity V_inf (m/s). Cannot be given together with traj.
        traj: [Trajectory] Optional solved WMPL Trajectory. If given, rho_e, zangle and v_0 are
            derived from it instead of being passed explicitly:
                - rho_e  <- getAtmDensity_vect at the end point (NRLMSISE-00, kg/m^3)
                - zangle <- pi/2 - traj.orbit.elevation_apparent_norot (rad)
                - v_0    <- traj.v_init (m/s)

    Return:
        (pe, pe_group): [tuple]
            - pe: [float] PE criterion value.
            - pe_group: [str] Structural group: 'I', 'II', 'IIIa', 'IIIb'.

    """

    # Derive rho_e, zangle and v_0 from the trajectory if provided
    if traj is not None:

        # Disallow mixing traj with explicit values, as it is ambiguous which should be used
        if (rho_e is not None) or (zangle is not None) or (v_0 is not None):
            raise ValueError("Give either `traj`, or explicit rho_e/zangle/v_0 values, not both.")

        if (getattr(traj, 'orbit', None) is None) \
            or (traj.orbit.elevation_apparent_norot is None):
            raise ValueError("The given trajectory does not have a computed orbit, so the radiant "
                "zenith angle cannot be derived. Pass rho_e, zangle and v_0 explicitly instead.")

        # Atmospheric density at the end point. The original calibration used a 1970s CIRA-type
        # reference atmosphere, but the difference to NRLMSISE-00 at fireball end heights is
        # negligible (< ~0.03 dex) compared to the 0.65 dex PE group widths.
        rho_e  = float(getAtmDensity_vect(traj.rend_lat, traj.rend_lon,
                                          traj.rend_ele, traj.rend_jd))
        zangle = np.pi/2 - traj.orbit.elevation_apparent_norot
        v_0    = traj.v_init

    elif (rho_e is None) or (zangle is None) or (v_0 is None):
        raise ValueError("Provide either `traj`, or all of rho_e, zangle and v_0.")

    # Input validation
    if mass <= 0:
        raise ValueError("mass must be > 0.")
    if rho_e <= 0:
        raise ValueError("rho_e must be > 0.")
    if np.cos(zangle) <= 0:
        raise ValueError("zangle must satisfy cos(zangle) > 0.")
    if v_0 <= 0:
        raise ValueError("v_0 must be > 0.")

    # Least-squares coefficients (Ceplecha & McCrosky 1976)
    A = -0.42
    B =  1.49
    C = -1.29

    ### Convert to Ceplecha & McCrosky (1976) units ###
    rho_e_cgs = rho_e*1e-3   # kg/m^3 -> g/cm^3
    mass_g    = mass*1e3     # kg     -> g
    v_kms     = v_0/1e3      # m/s    -> km/s

    ### ###

    # PE
    pe = np.log10(rho_e_cgs) + A*np.log10(mass_g) + B*np.log10(v_kms) \
        + C*np.log10(np.cos(zangle))

    # Structural group (Ceplecha & McCrosky 1976; boundaries as in Ceplecha 1988, Sec. 2e)
    if pe > -4.60:
        pe_group = 'I'      # ordinary chondrites
    elif pe > -5.25:
        pe_group = 'II'     # early-type carbonaceous chondrites
    elif pe > -5.70:
        pe_group = 'IIIa'   # cometary
    else:
        pe_group = 'IIIb'   # cometary, Draconid-like (most friable)

    return pe, pe_group


if __name__ == "__main__":

    import datetime

    from wmpl.Utils.TrajConversions import datetime2JD

    # Test the dynamic pressure function
    lat = np.radians(43)
    lon = np.radians(-81)
    height = 97000.0 # m 
    jd = datetime2JD(datetime.datetime(2018, 6, 15, 7, 15, 0))
    velocity = 41500 #m/s

    print('Dynamic pressure:', dynamicPressure(lat, lon, height, jd, velocity), 'Pa')



    # Test the KB function using an example GMN meteor (lambda-Sculptorid at  2023-12-12 12:46:23.484238)
    lat = np.radians(-30.904705)
    lon = np.radians(115.970139)
    ht_beg = 92.8401 # km
    jd = 2460291.03221625
    v_init = 14.70719 # km/s
    zenith_angle = np.radians(90 - 75.07381)

    print('KB:', calcKB(lat, lon, ht_beg*1000, jd, v_init*1000, zenith_angle, gmn_correction=True))



    ### Test the Pf function ###

    print()
    print("Borovicka et al. (2022) Pf parameter calculation:")

    # Test cases for Pf calculation
    pf_table = [
        # Label,              Pmax (MPa), Entry angle (deg), mass (kg), v_0 (km/s)
        ["Tagish Lake - C2u",       1.20,             18.00,  56_000.0,     15.80],
        ["Winchcombe  - CM2",       0.59,             41.87,      13.0,     13.61],
        ["Krizevci    -  H6",       3.70,             65.71,      50.0,     18.21],
        ["Novo Mesto  -  L5",      11.25,             47.85,    2500.0,     21.11],
        ["   2023CX1  - L56",       3.80,             49.09,     660.0,     14.04],
        ["   2024BX1  - Aub",       2.17,             75.56,     140.0,     15.20],
    ]

    # Test the Pf function - this value should return 1
    print("Pf test (should return ~1):", calcPf(1e6, 0, 1.0, 21.5e3))

    # Print the Pf parameters for the test cases
    for label, p_max, entry_angle, mass, v_0 in pf_table:

        pf, pf_category = calcPf(p_max*1e6, np.radians(90 - entry_angle), mass, v_0*1e3)

        print("  - {:s}: Pf = {:.2f}, Pf category = {:d}".format(label, pf, pf_category))

    ### ###



    ### Test the PE criterion ###

    print()
    print("Ceplecha & McCrosky (1976) PE criterion calculation:")

    # A 1 kg meteoroid at 15 km/s and a 45 deg zenith angle, ending at different heights - one test
    # case per structural group. The atmospheric density at the end height is computed using the
    # NRLMSISE-00 model. The fixed terms sum to
    # -0.42*log10(1000) + 1.49*log10(15) - 1.29*log10(cos(45 deg)) = +0.687, so the hand-checkable
    # expected value is PE = log10(rho_e in g/cm^3) + 0.687
    lat = np.radians(43)
    lon = np.radians(-81)
    jd = datetime2JD(datetime.datetime(2018, 6, 15, 7, 15, 0))

    pe_table = [
        # End height (m), expected PE, expected group
        [30000.0, -4.03,    'I'],
        [45000.0, -4.97,   'II'],
        [54000.0, -5.43, 'IIIa'],
        [63000.0, -5.92, 'IIIb'],
    ]

    for ht_end, pe_expected, group_expected in pe_table:

        # Compute the atmospheric density at the end height (kg/m^3)
        rho_e = float(getAtmDensity_vect(lat, lon, ht_end, jd))

        pe, pe_group = calcPE(1.0, rho_e=rho_e, zangle=np.radians(45), v_0=15e3)

        print("  - End ht {:2.0f} km: rho_e = {:.2e} kg/m^3, PE = {:.2f} (expected {:.2f}), "
            "group = {:>4s} (expected {:>4s})".format(ht_end/1000, rho_e, pe, pe_expected,
            pe_group, group_expected))

    ### ###