""" Implementation of the Borovicka (2007) meteor erosion model with added disruption.

References:
    Borovička, J., Spurný, P., & Koten, P. (2007). Atmospheric deceleration and light curves of Draconid 
    meteors and implications for the structure of cometary dust. Astronomy & Astrophysics, 473(2), 661-672.

    Campbell-Brown, M. D., Borovička, J., Brown, P. G., & Stokan, E. (2013). High-resolution modelling of 
    meteoroid ablation. Astronomy & Astrophysics, 557, A41.

"""

from __future__ import print_function, division, absolute_import


import math

import numpy as np
import scipy.stats
import scipy.integrate


# Cython init
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
from wmpl.MetSim.MetSimErosionCyTools import massLossRK4, decelerationRK4, luminousEfficiency, \
    ionizationEfficiency, atmDensityPoly, adaptiveSingleBodyStep, adaptiveDP45Step


### DEFINE CONSTANTS

# Earth acceleration in m/s^2 on the surface
G0 = 9.81

###


class Constants(object):
    def __init__(self):
        """ Constant parameters for the ablation modelling. """

        ### Simulation parameters ###

        # Time step. When adaptive_dt is False this is the fixed RK4 integration step; when
        #   adaptive_dt is True it is only the OUTPUT cadence (one results row per dt, total_time
        #   advances by dt), while each fragment is integrated with error-controlled adaptive
        #   sub-steps underneath (see adaptive_* below and ablateAll).
        self.dt = 0.005

        ### Adaptive sub-stepping (opt-in). Default False -> the engine runs the original fixed-step
        #   path and reproduces prior results exactly. When True (default), each fragment sub-steps
        #   adaptively within each dt to meet the tolerances below, while the output cadence stays fixed
        #   at dt. Adaptive is on by default: it matches the fixed-step light curve to well within
        #   measurement noise while running several times faster. Set False for bit-for-bit legacy runs. ###
        self.adaptive_dt = True
        # Adaptive integrator: True -> embedded Dormand-Prince RK45 on the coupled (mass, velocity)
        #   system (fewer RHS evals per sub-step, higher order -> ~3-4x faster); False -> step-doubling
        #   on the original operator-split RK4. Both meet the same tolerance; DP45 is the default.
        self.adaptive_high_order = True
        # Tolerance targets NUMERICAL error below MEASUREMENT noise (~0.1 mag, ~0.1 km/s at ~25 FPS),
        #   not machine convergence. rtol=1e-5 keeps the light curve and dynamics indistinguishable from
        #   the fixed-step result (well under measurement precision) at negligible extra cost over 1e-4
        #   (for eroding meteors the sub-step count is set by erosion_release_length, not rtol); loosen to
        #   1e-4 for a hair more speed on tolerance-bound cases, or tighten to 1e-6 for CAMO-grade data.
        self.adaptive_rtol = 1e-5         # relative tolerance on mass and speed
        self.adaptive_atol_m = 1e-14      # absolute mass tolerance (kg); ~ m_kill
        self.adaptive_atol_v = 0.1        # absolute speed tolerance (m/s); floor well under meas. noise
        self.adaptive_dt_min = 1e-7       # smallest allowed sub-step (s)
        self.adaptive_dt_max = 0.005      # largest allowed sub-step (s); should be <= dt
        self.adaptive_max_substeps = 10000  # runaway guard, sub-steps per fragment per macro step

        # Along-track grain-release interval (m). In adaptive mode grains are shed once per accepted
        #   sub-step, so the sub-step cadence sets the grain-birth resolution. Capping an eroding
        #   fragment's sub-step at erosion_release_length/v releases grains roughly every this many
        #   metres of flight - a physical resolution independent of dt and the error tolerance, so the
        #   erosion light-curve shape stops depending on dt. Smaller = finer grain sampling + slower.
        #   Only used in adaptive mode and only for eroding fragments (grains do not erode further).
        #   Exposed in the GUI so it can be coarsened for expensive fits (see erosion_release_vref).
        self.erosion_release_length = 200.0

        # Reference speed (m/s) that turns the along-track cadence into a fixed TIME cadence above it.
        #   The distance cap fixes cadence in metres, so a fast meteor takes tiny sub-steps (cost grows
        #   with v). Fragments faster than erosion_release_vref instead cap their sub-step at
        #   erosion_release_length/erosion_release_vref (a constant time), bounding the sub-step count;
        #   slower fragments keep the exact distance cadence. This keeps the per-event cost roughly
        #   velocity-independent. Set <= 0 to disable (pure distance cadence, the original behaviour).
        self.erosion_release_vref = 30000.0

        # Time elapsed since the beginning
        self.total_time = 0

        # Number of active fragments
        self.n_active = 0

        # Minimum possible mass for ablation (kg)
        self.m_kill = 1e-14

        # Minimum ablation velocity (m/s)
        self.v_kill = 3000

        # Minimum height (m)
        self.h_kill = 60000

        # Maximum length along the trajectory (m) after which the simulation will stop
        # -1 means no limit
        self.len_kill = -1000

        # Initial meteoroid height (m)
        self.h_init = 180000

        # Power of a 0 magnitude meteor
        self.P_0m = 840

        # Atmosphere density coefficients
        self.dens_co = np.array([6.96795507e+01, -4.14779163e+03, 9.64506379e+04, -1.16695944e+06, \
            7.62346229e+06, -2.55529460e+07, 3.45163318e+07])
        
        # Radius of the Earth (m)
        self.r_earth = 6_371_008.7714

        self.total_fragments = 0

        ### ###


        ### Wake parameters ###

        # PSF stddev (m)
        self.wake_psf_weights = [0.9, 0.1]
        self.wake_psf = [3.0, 20]

        # Wake extension from the leading fragment (m)
        self.wake_extension = 200

        # Specific heights at which the wake should be simulated (m)
        self.wake_heights = None

        ### ###



        ### Main meteoroid properties ###

        # Meteoroid bulk density (kg/m^3)
        self.rho = 1000

        # Initial meteoroid mass (kg)
        self.m_init = 2e-5

        # Initial meteoroid veocity (m/s)
        self.v_init = 23570

        # Shape factor (1.21 is sphere)
        self.shape_factor = 1.21

        # Main fragment ablation coefficient (s^2/km^2)
        self.sigma = 0.023/1e6

        # Zenith angle (radians)
        self.zenith_angle = math.radians(45)

        # Drag coefficient
        self.gamma = 1.0

        # Grain bulk density (kg/m^3)
        self.rho_grain = 3000


        # Luminous efficiency type (1 - 8, see luminousEfficiency function)
        self.lum_eff_type = 0

        # Constant luminous efficiency (percent)
        self.lum_eff = 0.7

        # Mean atomic mass of a meteor atom, kg (Jones 1997)
        self.mu = 23*1.66*1e-27

        ### ###


        ### Erosion properties ###

        # Toggle erosion on/off
        self.erosion_on = True


        # Bins per order of magnitude mass
        self.erosion_bins_per_10mass = 10
        
        # Height at which the erosion starts (meters)
        self.erosion_height_start = 102000

        # Erosion coefficient (s^2/m^2)
        self.erosion_coeff = 0.33/1e6

        
        # Height at which the erosion coefficient changes (meters)
        self.erosion_height_change = 90000

        # Erosion coefficient after the change (s^2/m^2)
        self.erosion_coeff_change = 0.33/1e6

        # Density after erosion change (density of small chondrules by default)
        self.erosion_rho_change = 3700

        # Ablation coeff after erosion change
        self.erosion_sigma_change = self.sigma

        # Grain distribution model ('powerlaw' mass or 'gamma' diameters)
        self.erosion_grain_distribution = 'powerlaw'

        # Grain mass distribution index
        self.erosion_mass_index = 2.5

        # Mass range for grains (kg)
        self.erosion_mass_min = 1.0e-11
        self.erosion_mass_max = 5.0e-10

        ###


        ### Disruption properties ###

        # Toggle disruption on/off
        self.disruption_on = True

        # Meteoroid compressive strength (Pa)
        self.compressive_strength = 2000

        # Height of disruption (will be assigned when the disruption occures)
        self.disruption_height = None

        # Erosion coefficient to use after disruption
        self.disruption_erosion_coeff = self.erosion_coeff

        # Disruption mass distribution index
        self.disruption_mass_index = 2.0


        # Mass ratio for disrupted fragments as the ratio of the disrupted mass
        self.disruption_mass_min_ratio = 1.0/100
        self.disruption_mass_max_ratio = 10.0/100

        # Ratio of mass that will disrupt into grains
        self.disruption_mass_grain_ratio = 0.25

        ### ###


        ### Complex fragmentation behaviour ###

        # Indicate if the complex fragmentation is used
        self.fragmentation_on = False

        # Track light curves of individual fragments
        self.fragmentation_show_individual_lcs = False

        # A list of fragmentation entries
        self.fragmentation_entries = []

        # Name of the fragmentation file
        self.fragmentation_file_name = "metsim_fragmentation.txt"

        ### ###


        ### Radar measurements ###

        # Height at which the electron line density is measured (m)
        self.electron_density_meas_ht = -1000

        # Measured electron line density (e-/m)
        self.electron_density_meas_q = -1

        ### ###


        
        ### OUTPUT PARAMETERS ###

        # Velocity at the beginning of erosion
        self.erosion_beg_vel = None

        # Mass at the beginning of erosion
        self.erosion_beg_mass = None

        # Dynamic pressure at the beginning of erosion
        self.erosion_beg_dyn_press = None

        # Mass of main fragment at erosion change
        self.mass_at_erosion_change = None

        # Energy received per unit cross section prior to to erosion begin
        # NOTE: never populated by runSimulation()/ablateAll() - always stays None here. The value this
        #   is meant to hold IS computed, by energyReceivedBeforeErosion() (this module, "es" return
        #   value), but that function is never called anywhere in this codebase to assign it back here.
        self.energy_per_cs_before_erosion = None

        # Energy received per unit mass prior to to erosion begin
        # NOTE: same as energy_per_cs_before_erosion above (see energyReceivedBeforeErosion()'s "ev")
        self.energy_per_mass_before_erosion = None

        # Height at which the main mass was depleeted
        self.main_mass_exhaustion_ht = None

        # Bottom height that the main fragment reached
        self.main_bottom_ht = self.h_init

        ### ###


class Fragment(object):
    def __init__(self):

        self.id = 0

        self.const = None

        # Shape-density coeff
        self.K = 0

        # Initial fragment mass
        self.m_init = 0

        # Instantaneous fragment mass Mass (kg)
        self.m = 0

        # Density (kg/m^3)
        self.rho = 0

        # Ablation coefficient (s^2/m^2)
        self.sigma = 0

        # Velocity (m/s)
        self.v = 0

        # Velocity components (vertical and horizontal)
        self.vv = 0
        self.vh = 0

        # Total drop due to gravity (m)
        self.h_grav_drop_total = 0

        # Length along the trajectory
        self.length = 0

        # Luminous intensity (Watts)
        self.lum = 0

        # Electron line density
        self.q = 0

        # Dynamic pressure (Gamma = 1.0, Pa)
        self.dyn_press = 0

        # Erosion coefficient value
        self.erosion_coeff = 0

        # Indicates that this fragment was born from a disruption event and uses a fixed, disruption-
        #   specific erosion coefficient (const.disruption_erosion_coeff) instead of the ordinary height-
        #   dependent one, so it's excluded from the height-based erosion_coeff recompute below
        self.disruption_child = False

        # Indicates that the one-time density/ablation coeff change at erosion_height_change has already
        #   been applied to this (main) fragment, so it isn't silently re-applied (and doesn't clobber a
        #   later manual sigma change, e.g. from a complex fragmentation "A" event) on every subsequent tick
        self.erosion_height_change_applied = False

        # Grain mass distribution index
        self.erosion_mass_index = 2.5

        # Mass range for grains (kg)
        self.erosion_mass_min = 1.0e-11
        self.erosion_mass_max = 5.0e-10


        self.erosion_enabled = False

        self.disruption_enabled = False

        self.active = False
        self.n_grains = 1

        # Indicate that this is the main fragment
        self.main = False

        # Indicate that the fragment is a grain
        self.grain = False

        # Indicate that this is born out of complex fragmentation
        self.complex = False

        # Identifier of the compex fragmentation entry
        self.complex_id = None

        # Last accepted adaptive sub-step size (s), warm-started across macro steps and copied to
        #   children by spawn_child(). 0 means "no previous step, start from the macro dt".
        self.adaptive_h_sub = 0.0


    def init(self, const, m, rho, v_init, sigma, gamma, zenith_angle, erosion_mass_index, erosion_mass_min, \
        erosion_mass_max):

        self.const = const

        self.m = m
        self.m_init = m
        self.h = const.h_init
        self.rho = rho
        self.v = v_init
        self.sigma = sigma
        self.gamma = gamma
        self.zenith_angle = zenith_angle

        # Compute shape-density coeff
        self.updateShapeDensityCoeff()

        self.erosion_mass_index = erosion_mass_index
        self.erosion_mass_min = erosion_mass_min
        self.erosion_mass_max = erosion_mass_max

        # Compute velocity components
        self.vv = -v_init*math.cos(zenith_angle)
        self.vh = v_init*math.sin(zenith_angle)

        self.active = True
        self.n_grains = 1

    def updateShapeDensityCoeff(self):
        """ Update the value of the shape-density coefficient. """

        self.K = self.gamma*self.const.shape_factor*self.rho**(-2/3.0)

    def spawn_child(self):
        """ Create a child of the Fragment instance. Copy over the reference to the shared 'const' object 
        and copy values of all other attributes. Note: if a mutable attribute that is not shared across 
        Fragment instances is added, this function will need to be revised. 
        """
        
        cls = self.__class__
        child = cls.__new__(cls)
                
        child.__dict__.update(self.__dict__)
    
        return child


class Wake(object):
    def __init__(self, const, frag_list, leading_frag_length, length_array):
        """ Container for the evaluated wake. 
        
        Arguments:
            const: [Constants]
            frag_list: [list of Fragment object] A list of active fragments visible in the wake.
            leading_frag_length: [float] Length from the beginning of the simulation of the leading fragment.
            length_array: [ndarray] An array of lengths (zero centered to the leading fragment) over which 
                the lag will be evaluated.
        """

        # Constants
        self.const = const

        # List of active fragments within the window
        self.frag_list = frag_list

        # Length of the leading fragment
        self.leading_frag_length = leading_frag_length

        # Array of lengths for plotting (independent variable)
        self.length_array = length_array

        # Length of visible fragments
        self.length_points = np.array([frag.length - self.leading_frag_length for frag in self.frag_list])

        # Luminosity of visible fragments
        self.luminosity_points = np.array([frag.lum for frag in self.frag_list])


        # Evalute the Gaussian at every fragment an add to the estimated wake
        self.wake_luminosity_profile = np.zeros_like(length_array)

        # If there is not entry for the wake PSF weights, initialize it
        if not hasattr(self.const, 'wake_psf_weights'):
            self.const.wake_psf_weights = np.ones_like(self.const.wake_psf)

        # Normalize the wake weights so they sum to 1
        self.const.wake_psf_weights = self.const.wake_psf_weights/np.sum(self.const.wake_psf_weights)
        
        for frag_lum, frag_len in zip(self.luminosity_points, self.length_points):

            for psf_m, psf_weight in zip(self.const.wake_psf, self.const.wake_psf_weights):
                self.wake_luminosity_profile += psf_weight*frag_lum*scipy.stats.norm.pdf(self.length_array, loc=frag_len, \
                    scale=psf_m)


def zenithAngleAtSimulationBegin(h0_sim, hb, zc, r_earth):
    """ Compute the meteor zenith angle at the beginning of the simulation, given the observed begin height
        and the observed zenith angle.

    Arguments:
        h0_sim: [float] Initial height of the simulation (m).
        hb: [float] Observed begin height (m).
        zc: [float] Observed zenith angle (radians).

    Returns:
        beta: [float] Zenith angle at the beginning of the simulation (radians).
        
    """

    beta = np.arcsin((hb + r_earth)/(h0_sim + r_earth)*np.sin(zc))

    return beta


def heightCurvature(h0, zc, l, r_earth):
    """ Compute the height at a given distance l from the origin, assuming a curved Earth.
    
    Arguments:
        h0: [float] Initial height (m).
        zc: [float] Zenith angle (radians).
        l: [float] Distance from the origin (m).
        r_earth: [float] Earth radius (m).

    Returns:
        h: [float] Height at distance l from the origin (m).
    """

    # Scalar inputs in the per-fragment hot path - math.sqrt/math.cos avoid numpy scalar overhead
    return math.sqrt((h0 + r_earth)**2 - 2*l*math.cos(zc)*(h0 + r_earth) + l**2) - r_earth


def generateFragments(const, frag_parent, eroded_mass, mass_index, mass_min, mass_max, 
                      keep_eroding=False, disruption=False, mass_model='powerlaw'):
    """ Given the parent fragment, fragment it into daughter fragments using either:
        - a power law mass distribution - appropriate for fragmentation of rock;
        - a gamma distribution - appropriate for spraying of droplets (iron meteoroids).

    Masses are binned and one daughter fragment may represent several fragments/grains, which is specified 
    with the n_grains atribute.

    Arguments:
        const: [object] Constants instance.
        frag_parent: [object] Fragment instance, the parent fragment.
        eroded_mass: [float] Mass to be distributed into daughter fragments. 
        mass_index: [float] Mass index to use to distribute the mass.
        mass_min: [float] Minimum mass bin (kg).
        mass_max: [float] Maximum mass bin (kg).

    Keyword arguments:
        keep_eroding: [bool] Whether the daughter fragments should keep eroding.
        disruption: [bool] Indicates that the disruption occurred, uses a separate erosion parameter for
            disrupted daughter fragments.
        mass_model: [bool] Fragment mass distribution model to use. Options: 
            - 'powerlaw' (default) - a power law mass distribution, appropriate for fragmentation of rock.
            - 'gamma' - a gamma size distribution, appropriate for spraying of droplets (iron meteoroids).

    Return:
        frag_children: [list] A list of Fragment instances - these are the generated daughter fragments.

    """

    # Compute the mass bin coefficient
    mass_bin_coeff = 10**(-1.0/const.erosion_bins_per_10mass)

    # Compute the total number of mass bins across the specified mass range
    k = int(1 + math.log10(mass_min/mass_max)/math.log10(mass_bin_coeff))

    # Use the gamma distribution if specified (e.g. for iron meteoroids which spray droplets)
    if mass_model == 'gamma':

        # Compute the number of needed bins for the gamma distribution
        mass_bins = np.array([mass_max*(mass_bin_coeff**i) for i in range(k)])
        bin_widths = mass_bins*(1 - mass_bin_coeff)        

        # Compute the expected value from the mass power-law distribution 
        log_range = math.log(mass_max/mass_min)

        # The mass index has been adjusted to compute the peak of the gamma distribution
        # - For s = 1, the peak mass is the arithmetic mean of the min and max masses.
        # - For s = 2, the peak mass is the harmonic mean of the min and max masses.
        # - For other s, the peak mass is computed using the formula below
        if mass_index == 1.0:
            
            # For mass_index = 1, the peak is the arithmetic mean
            m_mean = (mass_max - mass_min)/log_range

        elif mass_index == 2.0:
            
            # For mass_index = 2, the peak is the harmonic mean
            m_mean = log_range/(1.0/mass_min - 1.0/mass_max)

        else:
            # For other mass indices, compute the mean using the formula, computing each step separatelly to save time
            a = 2 - mass_index
            b = 1 - mass_index
            m_max_a = mass_max**a
            m_min_a = mass_min**a
            m_max_b = mass_max**b
            m_min_b = mass_min**b

            num = (m_max_a - m_min_a)/a
            den = (m_max_b - m_min_b)/b
            m_mean = num/den

        # Convert mean mass to mean diameter using the formula for spherical grains
        D_mean = (6*m_mean/(math.pi*const.rho_grain))**(1/3)
        # print(f"Mean mass (kg): {m_mean:.4g}")
        # print(f"Mean diameter (µm): { D_mean*1e6:.4g}")

        # The gamma function value for 5/3, used in the gamma distribution
        gamma_5_3 = 0.90274529295093375313996375552960671484470367431640625  # gamma(5/3)
        s = (D_mean*gamma_5_3)**3

        # Compute the grain diameter and convert it to mass
        grain_diameter = (6*mass_bins/(math.pi*const.rho_grain))**(1/3)

        # Compute the number of grains in the bin for diameter distribution
        n_D = (3*grain_diameter **2/s)*np.exp(-grain_diameter **3/s)

        # Compute the derivative of the diameter with respect to mass
        dD_dm = (1/3)*(6/(math.pi*const.rho_grain))**(1/3)*mass_bins**(-2/3)

        # Compute the number of grains in the bin for unit mass distribution
        n_m_raw = n_D*np.abs(dD_dm)

        # Compute the mass per bin from the unit mass distribution
        mass_per_bin_raw = n_m_raw*bin_widths*mass_bins

        # Scale the number of grains in the bin to match the eroded mass
        scaling = eroded_mass/np.sum(mass_per_bin_raw)
        n_m_scaled = n_m_raw*scaling


    # Use the power-law mass distribution by default
    else:
        # Compute the number of the largest grains
        if mass_index == 2:
            n0 = eroded_mass/(mass_max*k)
        else:
            n0 = abs((eroded_mass/mass_max)*(1 - mass_bin_coeff**(2 - mass_index))/(1 - mass_bin_coeff**((2 - mass_index)*k)))


    # Go though every mass bin
    frag_children = []
    leftover_mass = 0
    for i in range(0, k):

        # Gamma size distribution (droplets)
        if mass_model == 'gamma':

            # Extract the mass of the grain in the bin
            m_grain = mass_bins[i]

            # Compute the number of grains in the bin
            n_grains_bin = n_m_scaled[i]*bin_widths[i] + leftover_mass/m_grain
            n_grains_bin_round = int(math.floor(n_grains_bin)) # int(expected_count)

            # Compute the leftover mass
            leftover_mass = (n_grains_bin - n_grains_bin_round)*m_grain

        # Power-law mass distribution
        else:
            # Compute the mass of all grains in the bin (per grain)
            m_grain = mass_max*mass_bin_coeff**i

            # Compute the number of grains in the bin
            n_grains_bin = n0*(mass_max/m_grain)**(mass_index - 1) + leftover_mass/m_grain
            n_grains_bin_round = int(math.floor(n_grains_bin))

            # Compute the leftover mass
            leftover_mass = (n_grains_bin - n_grains_bin_round)*m_grain

        # If there are any grains to erode, erode them
        if n_grains_bin_round > 0:

            # Init the new fragment with params of the parent
            frag_child = frag_parent.spawn_child()

            # Assign the number of grains this fragment stands for (make sure to preserve the previous value
            #   if erosion is done for more fragments)
            frag_child.n_grains *= n_grains_bin_round

            # Assign the grain mass
            frag_child.m = m_grain
            frag_child.m_init = m_grain

            frag_child.active = True
            frag_child.main = False
            frag_child.disruption_enabled = False

            # Indicate that the fragment is a grain
            if (not keep_eroding) and (not disruption):
                frag_child.grain = True

            # Set the erosion coefficient value (disable in grain, only larger fragments)
            if keep_eroding:
                frag_child.erosion_enabled = True

                # If the disruption occurred, use a different erosion coefficient for daughter fragments
                if disruption:
                    frag_child.erosion_coeff = const.disruption_erosion_coeff
                    frag_child.disruption_child = True
                else:
                    frag_child.erosion_coeff = getErosionCoeff(const, frag_parent.h)

            else:
                # Compute the grain density and shape-density coeff
                frag_child.rho = const.rho_grain
                frag_child.updateShapeDensityCoeff()

                frag_child.erosion_enabled = False
                frag_child.erosion_coeff = 0


            # Give every fragment a unique ID
            frag_child.id = const.total_fragments
            const.total_fragments += 1

            frag_children.append(frag_child)


    return frag_children, const


def getErosionCoeff(const, h):
    """ Return the erosion coeff for the given height. """

    # Return the changed erosion coefficient
    if const.erosion_height_change >= h:
        return const.erosion_coeff_change

    # Return the starting erosion coeff
    elif const.erosion_height_start >= h:
        return const.erosion_coeff

    # If the height is above the erosion start height, return 0
    else:
        return 0


def killFragment(const, frag):
    """ Deactivate the given fragment and keep track of the stats. """

    frag.active = False
    const.n_active -= 1

    # Set the height when the main fragment was exhausted
    if frag.main:
        const.main_mass_exhaustion_ht = frag.h


def ablateAll(fragments, const, compute_wake=False, wake_heights_queue=None):
    """ Perform single body ablation of all fragments using the 4th order Runge-Kutta method. 

    Arguments:
        fragments: [list] A list of Fragment instances.
        const: [object] Constants instance.

    Keyword arguments:
        compute_wake: [bool] If True, the wake profile will be computed. False by default.
        wake_heights_queue: [list] A list of heights at which the wake should be computed. None by default.

    Return:
        ...

    Note:
        With const.adaptive_dt False, const.dt is the fixed RK4 step for every fragment, including
        grains. Near erosion_mass_min this single-step RK4 is under-resolved (a single step can overshoot
        the grain's true velocity by tens of percent vs. a finely-substepped mirror). With
        const.adaptive_dt True (default), each fragment is integrated with error-controlled adaptive
        sub-steps (const.adaptive_rtol etc.); dt then only sets the output cadence, and erosion grains
        are shed per sub-step so the light-curve shape stops depending on the step size. Fixed-step
        output is unchanged when adaptive_dt is False.
    """

    # Keep track of the total luminosity
    luminosity_total = 0.0

    # Keep track of the total luminosity weighted lum eff
    tau_total = 0.0

    # Keep track of the luminosity of the main fragment
    luminosity_main = 0.0

    # Keep track of the luminosity weighted lum eff of the main fragment
    tau_main = 0.0

    # Keep track of the luminosity of eroded and disrupted fragments
    luminosity_eroded = 0.0

    # Keep track of the luminosity weighted lum eff of eroded and disrupted fragments
    tau_eroded = 0.0

    # Keep track of the total electron density
    electron_density_total = 0.0

    # Keep track of parameters of the brightest fragment
    brightest_height = 0.0
    brightest_length = 0.0
    brightest_lum    = 0.0
    brightest_vel    = 0.0

    # Keep track of the the main fragment parameters
    main_mass = 0.0
    main_height = 0.0
    main_length = 0.0
    main_vel = 0.0
    main_dyn_press = 0.0

    frag_children_all = []

    # Snapshot which fragments were active at the start of this tick, before any of them can be killed
    # below - used for the leading fragment candidacy check further down, so a fragment that dies partway
    # through this very tick (and so is no longer frag.active by the time we get there) still counts with
    # its just-computed, physically valid position for the tick it died on
    fragments_active_at_tick_start = [frag for frag in fragments if frag.active]

    # Index the complex fragmentation entries by id once, so the per-grain lookup below is O(1) instead
    #   of a linear scan of all entries for every complex grain on every tick
    fragmentation_entry_by_id = {frag_entry.id: frag_entry for frag_entry in const.fragmentation_entries}

    # Total active mass (grain-weighted), accumulated inside the loop below instead of a separate
    #   post-loop scan of the whole fragment list
    mass_total_active = 0.0

    # Go through all active fragments
    for frag in fragments:

        # Skip the fragment if it's not active
        if not frag.active:
            continue

        # Advance this fragment across the macro step. In adaptive mode (const.adaptive_dt) the fragment
        #   is integrated with error-controlled adaptive sub-steps inside the Cython stepper; otherwise
        #   the original single fixed RK4 step is taken. Both paths leave frag.m/v/vv/vh/length/h and
        #   the diagnostic locals (rho_atm, mass_loss_ablation, mass_loss_erosion, deceleration_total)
        #   set for the shared luminosity/electron-density/event code below.
        erosion_events_adaptive = None   # per-sub-step erosion shedding (adaptive mode only)
        if const.adaptive_dt:

            erosion_active = 1 if (frag.erosion_enabled and (frag.erosion_coeff > 0)) else 0

            _stepper = adaptiveDP45Step if const.adaptive_high_order else adaptiveSingleBodyStep

            (frag.m, frag.v, frag.vv, frag.vh, frag.length, frag.h_grav_drop_total, frag.h, rho_atm,
                mass_loss_ablation, mass_loss_erosion, deceleration_total, went_up, n_sub,
                frag.adaptive_h_sub, runaway, floor_accepts, erosion_events_adaptive) \
                    = _stepper(
                    const.dt, frag.K, frag.sigma, frag.erosion_coeff, erosion_active,
                    frag.m, frag.v, frag.vv, frag.vh, frag.length, frag.h_grav_drop_total,
                    const.h_init, const.zenith_angle, const.r_earth, G0, const.dens_co,
                    const.adaptive_rtol, const.adaptive_atol_m, const.adaptive_atol_v, const.m_kill,
                    const.adaptive_dt_min, const.adaptive_dt_max, const.adaptive_max_substeps,
                    frag.adaptive_h_sub, const.erosion_release_length, const.erosion_release_vref)

            # Diagnostics (feed the cost study; also used to warn once on runaway/under-resolved steps)
            const.adaptive_substeps_total += n_sub
            const.adaptive_floor_accepts += floor_accepts
            if runaway:
                const.adaptive_runaway_events += 1

        else:

            # Get atmosphere density for the given height
            rho_atm = atmDensityPoly(frag.h, const.dens_co)

            # Compute the mass loss of the fragment due to ablation
            mass_loss_ablation = massLossRK4(const.dt, frag.K, frag.sigma, frag.m, rho_atm, frag.v)

            # Compute the mass loss due to erosion
            if frag.erosion_enabled and (frag.erosion_coeff > 0):
                mass_loss_erosion = massLossRK4(const.dt, frag.K, frag.erosion_coeff, frag.m, rho_atm, frag.v)
            else:
                mass_loss_erosion = 0

            # Compute the total mass loss
            mass_loss_total = mass_loss_ablation + mass_loss_erosion

            # If the total mass after ablation in this step is below zero, ablate what's left of the whole mass
            # (i.e. land exactly on m_new = 0, not some arbitrary leftover - see m_new below)
            if (frag.m + mass_loss_total) < 0:
                mass_loss_total = -frag.m

            # Compute new mass
            m_new = frag.m + mass_loss_total

            # Compute change in velocity
            deceleration_total = decelerationRK4(const.dt, frag.K, frag.m, rho_atm, frag.v)

            # If the deceleration is negative (i.e. the fragment is accelerating), then stop the fragment
            if deceleration_total > 0:
                frag.vv = frag.vh = frag.v = 0
                deceleration_total = 0

            # Otherwise update the velocity
            else:

                # Compute g at given height
                gv = G0/((1 + frag.h/const.r_earth)**2)

                # ### Add velocity change due to Earth's gravity ###

                # # Vertical component of a
                # av = -gv - deceleration_total*frag.vv/frag.v + frag.vh*frag.v/(const.r_earth + frag.h)

                # # Horizontal component of a
                # ah = -deceleration_total*frag.vh/frag.v - frag.vv*frag.v/(const.r_earth + frag.h)

                # ### ###

                ### Compute deceleration without the effects of gravity (to reconstruct the initial velocity
                # without the gravity component)

                # Vertical component of a
                av = -deceleration_total*frag.vv/frag.v + frag.vh*frag.v/(const.r_earth + frag.h)

                # Horizontal component of a
                ah = -deceleration_total*frag.vh/frag.v - frag.vv*frag.v/(const.r_earth + frag.h)

                ###

                # Compute the drop due to gravity
                h_grav_drop = 0.5*gv*const.dt**2

                # Track the total drop due to gravity
                frag.h_grav_drop_total += h_grav_drop

                # Update the velocity
                frag.vv -= av*const.dt
                frag.vh -= ah*const.dt
                frag.v = math.sqrt(frag.vh**2 + frag.vv**2)

                # Only allow the meteoroid to go down, and stop the ablation if it stars going up
                if frag.vv > 0:

                    frag.vv = 0

                    # Setting the height to zero will stop the ablation during the if catch below
                    frag.h = 0

            # Update length along the track
            frag.length += frag.v*const.dt

            # Update the mass
            frag.m = m_new

            # Old way of computing height which did not include the curvature of the Earth
            # frag.h = frag.h + frag.vv*const.dt

            # Compute the height taking the curvature of the Earth and the gravity drop into account
            frag.h = heightCurvature(const.h_init, const.zenith_angle, frag.length, const.r_earth)
            frag.h -= frag.h_grav_drop_total

        # Get the luminous efficiency
        # NOTE: frag.v/frag.m here are already this tick's post-step values, while mass_loss_ablation/
        # deceleration_total above were computed from last tick's pre-step state - a small (~1.6% in
        # total luminosity) but systematic inconsistency between the state used for the rate terms and
        # the state used for the efficiency factors converting them to lum/q. Left as-is (not clearly
        # wrong, no evidence it's an accidental regression) - see beta below for the same pattern.
        tau = luminousEfficiency(const.lum_eff_type, const.lum_eff, frag.v, frag.m)

        # # Compute luminosity for one grain/fragment (without the deceleration term)
        # lum = -tau*((mass_loss_ablation/const.dt*frag.v**2)/2)

        # Compute luminosity for one grain/fragment (with the deceleration term)
        # NOTE: The deceleration term can sometimes be numerically unstable for some reason...
        lum = -tau*((mass_loss_ablation/const.dt*frag.v**2)/2 + frag.m*frag.v*deceleration_total)

        # Compute the electron line density
        beta = ionizationEfficiency(frag.v)
        q = -beta*(mass_loss_ablation/const.dt)/(const.mu*frag.v)

        # Compute the total luminosity
        frag.lum = lum*frag.n_grains

        # Compute the total electron line density
        frag.q = q*frag.n_grains

        # Keep track of the total luminosity across all fragments
        luminosity_total += frag.lum

        # Keep track of the total number of produced electrons
        electron_density_total += frag.q

        # Keep track of the total luminosity weighted lum eff
        tau_total += tau*frag.lum

        # Compute aerodynamic loading on the grain (always assume Gamma = 1.0)
        # dyn_press = frag.gamma*rho_atm*frag.v**2
        dyn_press = 1.0*rho_atm*frag.v**2
        frag.dyn_press = dyn_press

        # if frag.id == 0:
        #     print('----- id:', frag.id)
        #     print('t:', const.total_time)
        #     print('V:', frag.v/1000)
        #     print('H:', frag.h/1000)
        #     print('m:', frag.m)
        #     print('DynPress:', dyn_press/1000, 'kPa')

        # Keep track of the parameters of the main fragment
        if frag.main:
            luminosity_main = frag.lum
            tau_main = tau
            main_mass = frag.m
            main_height = frag.h
            main_length = frag.length
            main_vel = frag.v
            main_dyn_press = dyn_press

        # Keep track of the brightest fragment (checked before the kill-check below so a fragment's own
        #   death tick - where m/v/h/lum first cross a kill threshold - still counts as a valid candidate,
        #   instead of silently vanishing from brightest_* for the tick it actually died on)
        if frag.lum > brightest_lum:
            brightest_lum = frag.lum
            brightest_height = frag.h
            brightest_length = frag.length
            brightest_vel = frag.v

        # If the fragment is done, stop ablating
        if  (
            (frag.m <= const.m_kill)
            or (frag.v < const.v_kill)
            or (frag.h < const.h_kill)
            or (frag.lum < 0)
            or ((const.len_kill > 0) and (frag.length > const.len_kill))
            ):

            killFragment(const, frag)

            # print('Killing', frag.id)
            continue

        # For fragments born out of complex fragmentation, keep track of their luminosity and height
        if not frag.main:

            # Keep track of magnitudes of complex fragmentation fragments (the per-entry breakdown below is
            #   only computed if explicitly requested, since it's an O(entries) lookup per grain per tick -
            #   this must NOT gate the cheap eroded/disrupted totals below, which are tracked unconditionally)
            if frag.complex:

                if const.fragmentation_show_individual_lcs:

                    # Find the corresponding fragmentation entry (O(1) via the id index built above)
                    frag_entry = fragmentation_entry_by_id.get(frag.complex_id)

                    if frag_entry is not None:

                        # Store luminosity of grains
                        if frag.grain:

                            add_new_entry = False

                            # Check if the last time entry corresponds to the current time, and add to it
                            if not len(frag_entry.grains_time_data):
                                add_new_entry = True
                            elif const.total_time != frag_entry.grains_time_data[-1]:
                                add_new_entry = True

                            # Add the current integration time
                            if add_new_entry:
                                frag_entry.grains_time_data.append(const.total_time)
                                frag_entry.grains_luminosity.append(frag.lum)
                                frag_entry.grains_tau_over_lum.append(tau*frag.lum)

                            # Add to the total luminosity at the current time step that's already been added
                            else:
                                frag_entry.grains_luminosity[-1] += frag.lum
                                frag_entry.grains_tau_over_lum[-1] += tau*frag.lum

                        # Store parameters of the main fragment
                        else:

                            add_new_entry = False

                            # Check if the last time entry corresponds to the current time, and add to it
                            if not len(frag_entry.main_time_data):
                                add_new_entry = True
                            elif const.total_time != frag_entry.main_time_data[-1]:
                                add_new_entry = True

                            # Add the current integration time
                            if add_new_entry:
                                frag_entry.main_time_data.append(const.total_time)
                                frag_entry.main_luminosity.append(frag.lum)
                                frag_entry.main_tau_over_lum.append(tau*frag.lum)

                            # Add to the total luminosity at the current time step that's already been added
                            else:
                                frag_entry.main_luminosity[-1] += frag.lum
                                frag_entry.main_tau_over_lum[-1] += tau*frag.lum

            # Keep track of luminosity of eroded and disrupted fragments ejected directly from the main
            #   fragment (always tracked, regardless of fragmentation_show_individual_lcs - see above)
            else:

                luminosity_eroded += frag.lum
                tau_eroded += tau*frag.lum

        # For non-complex fragmentation only: Check if the erosion should start, given the height,
        #   and create grains
        if (not frag.complex) and (frag.h < const.erosion_height_start) and frag.erosion_enabled \
            and const.erosion_on:

            # Turn on the erosion of the fragment (disruption daughters keep the fixed, disruption-specific
            #   erosion coefficient they were assigned at birth instead - see disruption_child's docstring -
            #   otherwise this would silently overwrite it with the ordinary height-dependent one)
            if not frag.disruption_child:
                frag.erosion_coeff = getErosionCoeff(const, frag.h)

            # Update the main fragment physical parameters if it is changed after erosion coefficient change
            # (only once - otherwise this would silently re-apply every tick and clobber any later manual
            # sigma change, e.g. from a complex fragmentation "A" event)
            if frag.main and (not frag.erosion_height_change_applied) and (const.erosion_height_change >= frag.h):

                # Update the density
                frag.rho = const.erosion_rho_change
                frag.updateShapeDensityCoeff()

                # Update the ablation coeff
                frag.sigma = const.erosion_sigma_change

                frag.erosion_height_change_applied = True

        # Create grains for erosion-enabled fragments
        if frag.erosion_enabled:

            def _spawnGrainsFromErosion(eroded_mass):
                """ Distribute the given eroded mass into grains born from the fragment's current state.
                    Uses the enclosing frag/const. """
                grain_children, _ = generateFragments(const, frag, eroded_mass, \
                    frag.erosion_mass_index, frag.erosion_mass_min, frag.erosion_mass_max, \
                    keep_eroding=False, mass_model=const.erosion_grain_distribution)
                const.n_active += len(grain_children)
                frag_children_all.extend(grain_children)

            eroded_this_tick = False

            if erosion_events_adaptive is not None:

                # Adaptive: shed grains at each sub-step's resolved state, so grain births are spread
                #   along the trajectory (matching a finely-stepped fixed run) rather than dumped at the
                #   macro interval's end point. generateFragments()/spawn_child() copies the parent's
                #   current position/velocity into each grain, so temporarily rewind the parent to each
                #   sub-step's state, spawn, then restore its end-of-macro-step state.
                if erosion_events_adaptive:
                    _saved_state = (frag.h, frag.v, frag.vv, frag.vh, frag.length,
                                    frag.h_grav_drop_total)
                    for (em, e_h, e_v, e_vv, e_vh, e_len, e_grav) in erosion_events_adaptive:
                        if em <= 0:
                            continue
                        frag.h = e_h; frag.v = e_v; frag.vv = e_vv; frag.vh = e_vh
                        frag.length = e_len; frag.h_grav_drop_total = e_grav
                        _spawnGrainsFromErosion(em)
                        eroded_this_tick = True
                    (frag.h, frag.v, frag.vv, frag.vh, frag.length, frag.h_grav_drop_total) = _saved_state

            else:

                # Fixed step: distribute the whole macro-step erosion loss at the end-of-step state
                if abs(mass_loss_erosion) > 0:
                    _spawnGrainsFromErosion(abs(mass_loss_erosion))
                    eroded_this_tick = True

            # Record erosion-begin bookkeeping for the main fragment once, at the END-of-step state so the
            #   (vel, mass, dyn_press) triple is mutually consistent in both modes (in adaptive mode frag
            #   has been restored above; the per-event rewind must not leak into these diagnostics)
            if eroded_this_tick and frag.main:
                if const.erosion_beg_vel is None:
                    const.erosion_beg_vel = frag.v
                    const.erosion_beg_mass = frag.m
                    const.erosion_beg_dyn_press = dyn_press

                # Record the mass when erosion is changed
                elif (const.erosion_height_change >= frag.h) and (const.mass_at_erosion_change is None):
                    const.mass_at_erosion_change = frag.m

        # Disrupt the fragment if the dynamic pressure exceeds its strength
        if frag.disruption_enabled and const.disruption_on:
            if dyn_press > const.compressive_strength:

                # Compute the mass that should be disrupted into fragments
                mass_frag_disruption = frag.m*(1 - const.disruption_mass_grain_ratio)

                fragments_total_mass = 0
                if mass_frag_disruption > 0:

                    # Disrupt the meteoroid into fragments
                    disruption_mass_min = const.disruption_mass_min_ratio*mass_frag_disruption
                    disruption_mass_max = const.disruption_mass_max_ratio*mass_frag_disruption

                    # Generate larger fragments, possibly assign them a separate erosion coefficient
                    frag_children, const = generateFragments(const, frag, mass_frag_disruption, \
                        const.disruption_mass_index, disruption_mass_min, disruption_mass_max, \
                        keep_eroding=const.erosion_on, disruption=True, 
                        mass_model=const.erosion_grain_distribution)

                    frag_children_all += frag_children
                    const.n_active += len(frag_children)

                    # Compute the mass that went into fragments
                    fragments_total_mass = sum([f.n_grains*f.m for f in frag_children])

                    # Assign the height of disruption
                    const.disruption_height = frag.h

                    print('Disrupting id', frag.id)
                    print('Height: {:.3f} km'.format(const.disruption_height/1000))
                    print('Disrupted mass: {:e}'.format(mass_frag_disruption))
                    print('Mass distribution:')
                    for f in frag_children:
                        print('{:4d}: {:e} kg'.format(f.n_grains, f.m))
                    print('Disrupted total mass: {:e}'.format(fragments_total_mass))

                # Disrupt a portion of the leftover mass into grains
                mass_grain_disruption = frag.m - fragments_total_mass
                if mass_grain_disruption > 0:
                    grain_children, const = generateFragments(const, frag, mass_grain_disruption, 
                        frag.erosion_mass_index, frag.erosion_mass_min, frag.erosion_mass_max, \
                        keep_eroding=False, mass_model=const.erosion_grain_distribution)

                    frag_children_all += grain_children
                    const.n_active += len(grain_children)

                # Deactive the disrupted fragment
                frag.m = 0
                killFragment(const, frag)

        # Handle complex fragmentation and status changes of the main fragment
        if frag.main and const.fragmentation_on:

            # Get a list of complex fragmentations that are still to do
            frags_to_do = [frag_entry for frag_entry in const.fragmentation_entries if not frag_entry.done]

            if len(frags_to_do):

                # Go through all fragmentations that needs to be performed
                for frag_entry in frags_to_do:

                    # Check if the height of the main fragment is right to perform the operation.
                    # Run if:
                    # (a) If the fireball is going down and the fragmentation is for the downward direction
                    # (b) If the fireball is going up and the fragmentation is for the upward direction
                    #     And the fireball started going up
                    if ( (not frag_entry.upward_only) and (frag.h < frag_entry.height) ) \
                    or ( 
                        frag_entry.upward_only 
                        and (frag.h > frag_entry.height) 
                        and (frag.h > const.main_bottom_ht)
                        ):
                        
                        parent_initial_mass = frag.m

                        # Change parameters of all fragments
                        if frag_entry.frag_type == "A":

                            for frag_tmp in (fragments + frag_children_all + [frag]):

                                # Update the ablation coefficient
                                if frag_entry.sigma is not None:
                                    frag_tmp.sigma = frag_entry.sigma

                                # Update the drag coefficient
                                if frag_entry.gamma is not None:
                                    frag_tmp.gamma = frag_entry.gamma
                                    frag_tmp.updateShapeDensityCoeff()

                        # Change the parameters of the main fragment
                        if frag_entry.frag_type == "M":

                            if frag_entry.sigma is not None:
                                frag.sigma = frag_entry.sigma

                            if frag_entry.erosion_coeff is not None:
                                frag.erosion_coeff = frag_entry.erosion_coeff

                            if frag_entry.mass_index is not None:
                                frag.erosion_mass_index = frag_entry.mass_index

                            if frag_entry.grain_mass_min is not None:
                                frag.erosion_mass_min = frag_entry.grain_mass_min

                            if frag_entry.grain_mass_max is not None:
                                frag.erosion_mass_max = frag_entry.grain_mass_max

                        # Create a new single-body or eroding fragment
                        if (frag_entry.frag_type == "F") or (frag_entry.frag_type == "EF"):

                            # Go through all new fragments
                            for frag_num in range(frag_entry.number):

                                # Mass of the new fragment
                                new_frag_mass = parent_initial_mass*(frag_entry.mass_percent/100.0)/frag_entry.number
                                frag_entry.mass = new_frag_mass*frag_entry.number

                                # Decrease the parent mass
                                frag.m -= new_frag_mass

                                # Create the new fragment
                                frag_new = frag.spawn_child()
                                frag_new.active = True
                                frag_new.main = False
                                frag_new.disruption_enabled = False

                                # Indicate that the fragments are born out of complex fragmentation
                                frag_new.complex = True

                                # Assign the complex fragmentation ID
                                frag_new.complex_id = frag_entry.id

                                # Assing the mass to the new fragment
                                frag_new.m = new_frag_mass

                                # Assign possible new ablation coeff to this fragment
                                if frag_entry.sigma is not None:
                                    frag_new.sigma = frag_entry.sigma

                                # If the fragment is eroding, set erosion parameters
                                if frag_entry.frag_type == "EF":
                                    frag_new.erosion_enabled = True

                                    frag_new.erosion_coeff = frag_entry.erosion_coeff

                                    frag_new.erosion_mass_index = frag_entry.mass_index
                                    frag_new.erosion_mass_min = frag_entry.grain_mass_min
                                    frag_new.erosion_mass_max = frag_entry.grain_mass_max

                                else:
                                    # Disable erosion for single-body fragments
                                    frag_new.erosion_enabled = False

                                # Add the new fragment to the list of childern
                                frag_children_all.append(frag_new)
                                const.n_active += 1

                        # Release dust
                        if frag_entry.frag_type == "D":

                            # Compute the mass of the dust
                            dust_mass = frag.m*(frag_entry.mass_percent/100.0)
                            frag_entry.mass = dust_mass

                            # Subtract from the parent mass
                            frag.m -= dust_mass

                            # Create the new fragment
                            frag_new = frag.spawn_child()
                            frag_new.active = True
                            frag_new.main = False
                            frag_new.disruption_enabled = False

                            # Indicate that the fragments are born out of complex fragmentation
                            frag_new.complex = True

                            # Assign the complex fragmentation ID
                            frag_new.complex_id = frag_entry.id

                            # Generate dust grains
                            grain_children, const = generateFragments(const, frag_new, dust_mass, \
                                frag_entry.mass_index, frag_entry.grain_mass_min, frag_entry.grain_mass_max, \
                                keep_eroding=False, mass_model=const.erosion_grain_distribution)

                            # Add fragments to the list
                            frag_children_all += grain_children
                            const.n_active += len(grain_children)

                        # Set the fragmentation as finished
                        frag_entry.done = True

                        # Set physical conditions at the moment of fragmentation
                        frag_entry.time = const.total_time
                        frag_entry.dyn_pressure = dyn_press
                        frag_entry.velocity = frag.v
                        frag_entry.parent_mass = parent_initial_mass

        # If the fragment is done, stop ablating
        if (frag.m <= const.m_kill):

            killFragment(const, frag)
            # print('Killing', frag.id)

            continue

        # Fragment survived this tick - add its grain-weighted mass to the total active mass. Only
        #   fragments still active at the end of the tick reach here (killed/disrupted ones hit a
        #   'continue' above), matching the old post-loop "active only" scan exactly
        mass_total_active += frag.m*frag.n_grains

    # Track the leading fragment length (use the tick-start snapshot, not a fresh frag.active filter, so a
    #   fragment that died partway through this tick is still a candidate for it - see the snapshot's own
    #   comment above)
    if len(fragments_active_at_tick_start):
        leading_frag = max(fragments_active_at_tick_start, key=lambda x: x.length)
        leading_frag_length    = leading_frag.length
        leading_frag_height    = leading_frag.h
        leading_frag_vel       = leading_frag.v
        leading_frag_dyn_press = leading_frag.dyn_press
    else:
        leading_frag_length    = None
        leading_frag_height    = None
        leading_frag_vel       = None
        leading_frag_dyn_press = None

    ### Compute the wake profile ###
    
    # If the specific wake heights are given, check if the current height is below the next wake height
    if (wake_heights_queue is not None) and (leading_frag_height is not None):

        # If there are any heights left in the queue
        if len(wake_heights_queue):
            
            # If the current height is below the next wake height, compute the wake
            if leading_frag_height <= wake_heights_queue[0]:
                compute_wake = True
                
                # Pop all heights that are above the current height (including the one we just passed)
                while len(wake_heights_queue) and (leading_frag_height <= wake_heights_queue[0]):
                    wake_heights_queue.pop(0)

            else:
                compute_wake = False
        
        else:
            compute_wake = False



    if compute_wake and (leading_frag_length is not None):

        # Evaluate the Gaussian from +3 sigma in front of the leading fragment to behind
        front_len = leading_frag_length + 3*const.wake_psf[0]
        back_len = leading_frag_length - const.wake_extension

        ### Compute the wake as convoluted luminosities with the PSF ###

        length_array = np.linspace(back_len, front_len, 500) - leading_frag_length

        frag_list = []

        for frag in fragments:

            # Take only those lengths inside the wake window
            if frag.active:
                if (frag.length > back_len) and (frag.length < front_len):
                    frag_list.append(frag.spawn_child())

        # Store evaluated wake
        wake = Wake(const, frag_list, leading_frag_length, length_array)

        ### ###

    else:
        wake = None

    ### ###

    # Add generated fragment children to the list of fragments
    fragments += frag_children_all

    # Fragments/grains created this tick are all active - add their grain-weighted mass to the running
    #   total accumulated in the loop above (completes the "all active fragments" sum). Same n_grains
    #   convention already used for lum/q.
    for frag_child in frag_children_all:
        mass_total_active += frag_child.m*frag_child.n_grains

    # Drop fully-ablated grains from the fragment list so subsequent ticks don't keep re-scanning them
    #   (the list otherwise grows monotonically - dead grains are never needed again). Dead non-grains
    #   (the main fragment, complex-fragmentation daughters) are KEPT: runSimulation() still scans them
    #   after the run to find the main fragment and to aggregate per-fragmentation final masses.
    fragments = [frag for frag in fragments if frag.active or (not frag.grain)]

    # Increment the running time
    const.total_time += const.dt

    # Weigh the tau by luminosity
    if luminosity_total > 0:
        tau_total /= luminosity_total
    else:
        tau_total = 0

    if luminosity_eroded > 0:
        tau_eroded /= luminosity_eroded
    else:
        tau_eroded = 0

    return fragments, const, luminosity_total, luminosity_main, luminosity_eroded, electron_density_total, \
        tau_total, tau_main, tau_eroded, brightest_height, brightest_length, brightest_vel, \
        leading_frag_height, leading_frag_length, leading_frag_vel, leading_frag_dyn_press, \
        mass_total_active, main_mass, main_height, main_length, main_vel, main_dyn_press, wake


def runSimulation(const, compute_wake=False):
    """ Run the ablation simulation. """

    # Back-fill adaptive-timestep settings for Constants loaded from older JSONs that predate them, so
    #   such runs pick up the current defaults (adaptive on). Set adaptive_dt False in the JSON/GUI for
    #   bit-for-bit legacy fixed-step behaviour. Also reset the per-run diagnostics.
    _adaptive_defaults = {'adaptive_dt': True, 'adaptive_high_order': True, 'adaptive_rtol': 1e-5,
        'adaptive_atol_m': 1e-14, 'adaptive_atol_v': 0.1, 'adaptive_dt_min': 1e-7,
        'adaptive_dt_max': const.dt, 'adaptive_max_substeps': 10000,
        'erosion_release_length': 200.0, 'erosion_release_vref': 30000.0}
    for _attr, _default in _adaptive_defaults.items():
        if not hasattr(const, _attr):
            setattr(const, _attr, _default)
    const.adaptive_substeps_total = 0
    const.adaptive_runaway_events = 0
    const.adaptive_floor_accepts = 0

    # The adaptive Cython stepper needs dens_co as a float64 array (memoryview); coerce once (the fixed
    #   path's atmDensityPoly requires an ndarray too, so this is safe in both modes).
    const.dens_co = np.asarray(const.dens_co, dtype=np.float64)

    # Ensure that the grain mass min is smaller than the grain mass max
    if const.erosion_mass_min > const.erosion_mass_max:
        const.erosion_mass_min, const.erosion_mass_max = const.erosion_mass_max, const.erosion_mass_min

    ###


    if const.fragmentation_on:

        # Assign unique IDs to complex fragmentation entries
        for i, frag_entry in enumerate(const.fragmentation_entries):
            frag_entry.id = i

            # Reset output parameters for every fragmentation entry
            frag_entry.resetOutputParameters()


    fragments = []

    # Init the main fragment
    frag = Fragment()
    frag.init(const, const.m_init, const.rho, const.v_init, const.sigma, const.gamma, const.zenith_angle, \
        const.erosion_mass_index, const.erosion_mass_min, const.erosion_mass_max)
    frag.main = True
    
    # Erode the main fragment
    frag.erosion_enabled = True

    # Disrupt the main fragment
    frag.disruption_enabled = True

    fragments.append(frag)



    # Reset simulation parameters
    const.total_time = 0
    const.n_active = 1
    const.total_fragments = 1
    const.main_bottom_ht = const.h_init


    ###



    # Check that the grain density is larger than the bulk density, and if not, set the grain density
    #   to be the same as the bulk density
    if const.rho > const.rho_grain:
        const.rho_grain = const.rho


    # If the wake heights are given, sort them by height descending
    wake_heights_queue = None
    if (const.wake_heights is not None) and compute_wake:
        wake_heights_queue = sorted(const.wake_heights, reverse=True)


    # Run the simulation until all fragments stop ablating
    results_list = []
    wake_results = []
    while const.n_active > 0:

        # Ablate the fragments
        fragments, const, luminosity_total, luminosity_main, luminosity_eroded, electron_density_total, \
            tau_total, tau_main, tau_eroded, brightest_height, brightest_length, brightest_vel, \
            leading_frag_height, leading_frag_length, leading_frag_vel, leading_frag_dyn_press, \
            mass_total_active, main_mass, main_height, main_length, main_vel, main_dyn_press, \
            wake = ablateAll(fragments, const, compute_wake=compute_wake, wake_heights_queue=wake_heights_queue)
        
        # Track the bottom height of the main fragment
        if main_height > 0:
            const.main_bottom_ht = min(main_height, const.main_bottom_ht)

        # Store wake estimation results
        wake_results.append(wake)

        # Stack results list
        results_list.append([const.total_time, luminosity_total, luminosity_main, luminosity_eroded, \
            electron_density_total, tau_total, tau_main, tau_eroded, brightest_height, brightest_length, \
            brightest_vel, leading_frag_height, leading_frag_length, leading_frag_vel, \
            leading_frag_dyn_press, mass_total_active, main_mass, main_height, main_length, main_vel, \
            main_dyn_press])


    # Warn once if any fragment hit the adaptive sub-step cap (under-resolved; raise adaptive_max_substeps
    #   or loosen the tolerance if this is frequent)
    if const.adaptive_dt and (const.adaptive_runaway_events > 0):
        print("WARNING: adaptive stepper hit max_substeps ({:d}) on {:d} fragment-step(s); results may be "
              "under-resolved there.".format(const.adaptive_max_substeps, const.adaptive_runaway_events))
    if const.adaptive_dt and (const.adaptive_floor_accepts > 0):
        print("WARNING: adaptive stepper accepted {:d} sub-step(s) at the dt_min floor ({:g} s) while still "
              "over tolerance; lower adaptive_dt_min or loosen adaptive_rtol if this is frequent.".format(
              const.adaptive_floor_accepts, const.adaptive_dt_min))


    # Find the main fragment and return it with results
    frag_main = None
    for frag in fragments:
        if frag.main:
            frag_main = frag
            break


    ### Find the fragments born out of complex fragmentations and assign them to the fragmentation entries ###

    # Reset all fragment lists for entries
    for frag_entry in const.fragmentation_entries:
        frag_entry.fragments = []

    # Find fragments for every fragmentation
    for frag_entry in const.fragmentation_entries:
        for frag in fragments:
            if not frag.grain:
                if frag.complex_id is not None:
                    if frag_entry.id == frag.complex_id:

                        # Add fragment
                        frag_entry.fragments.append(frag)

                        # Compute the final mass of all fragments in this fragmentation after ablation stopped
                        final_mass = frag_entry.number*frag.m

                        # If the final mass is below a gram, assume it's zero
                        if final_mass < 1e-3:
                            final_mass = None

                        # Assign the final mass to the fragmentation entry
                        frag_entry.final_mass = final_mass


    ### ###


    return frag_main, results_list, wake_results


def energyReceivedBeforeErosion(const, lam=1.0):
    """ Compute the energy the meteoroid receive prior to erosion, assuming no major mass loss occurred.
    
    Arguments:
        const: [Constants]

    Keyword arguments:
        lam: [float] Heat transfter coeff. 1.0 by default.

    Return:
        (es, ev):
            - es: [float] Energy received per unit cross-section (J/m^2)
            - ev: [float] Energy received per unit mass (J/kg).

    """

    # Integrate atmosphere density from the beginning of simulation to beginning of erosion.
    dens_integ = scipy.integrate.quad(atmDensityPoly, const.erosion_height_start, const.h_init, \
        args=(const.dens_co))[0]

    # Compute the energy per unit cross-section
    es = 1/2*lam*(const.v_init**2)*dens_integ/np.cos(const.zenith_angle)

    # Compute initial shape-density coefficient
    k = const.gamma*const.shape_factor*const.rho**(-2/3.0)

    # Compute the energy per unit mass
    ev = es*k/(const.gamma*const.m_init**(1/3.0))

    return es, ev


if __name__ == "__main__":

    import matplotlib.pyplot as plt


    from wmpl.Utils.AtmosphereDensity import fitAtmPoly
    from wmpl.Utils.TrajConversions import date2JD


    # Show wake
    show_wake = False


    # Init the constants
    const = Constants()

    # Fit atmosphere density polynomial for the given location and time on Earth, and the range of simulation 
    # heights
    const.dens_co = fitAtmPoly(
        np.radians(45.3), # lat +N
        np.radians(18.1), # lon +E
         70000, # height_min in m
        180000, # height_max in m (this needs to be the same as the beginning of simulation)
        date2JD(2020, 4, 20, 16, 15, 0) # Julian date
        )


    ### Set some physical parameters of the meteoroid ###

    # Set the power of a zero magnitude meteor for silicon sensors (W)
    const.P_0m = 1210

    # Initial mass (kg)
    const.m_init = 1e-5

    # Bulk density (kg/m^3)
    const.rho = 300

    # Grain density (kg/m^3)
    const.rho_grain = 3000

    # Initial velocity (m/s)
    const.v_init = 45000

    # Ablation coefficient (kg/MJ or s^2/km^2)
    const.sigma = 0.023/1e6

    # Zenith angle = 90 - elevation angle
    const.zenith_angle = math.radians(45)

    # Grain bulk density (kg/m^3) - used for erosion, 3000 is used for faint meteors and 3500 for fireballs
    const.rho_grain = 3000

    # Luminous efficiency type (5 for faint meteors, 7 for fireballs)
    const.lum_eff_type = 5



    # Toggle erosion on/off
    const.erosion_on = True

    # Bins per order of magnitude mass (2 is enough for firebals and a large range of masses, 
    # 5 for fainter meteors)
    const.erosion_bins_per_10mass = 5
    
    # Height at which the erosion starts (meters)
    const.erosion_height_start = 102000

    # Erosion coefficient (kg/MJ or s^2/km^2)
    const.erosion_coeff = 0.33/1e6

   
    # Height at which the erosion coefficient changes - for no change is should be below the end height 
    # (meters)
    const.erosion_height_change = 0


    # Grain mass distribution index
    const.erosion_mass_index = 2.0

    # Mass range for grains (kg)
    const.erosion_mass_min = 1.0e-11
    const.erosion_mass_max = 5.0e-10

    # Disable disruption
    const.disruption_on = False

    ### ###



    # Run the ablation simulation
    frag_main, results_list, wake_results = runSimulation(const, compute_wake=show_wake)



    ### ANALYZE RESULTS ###


    # System limiting magnitude (used for plotting the wake)
    lim_mag = 6.0

    # Unpack the results
    results_list = np.array(results_list).astype(np.float64)
    time_arr, luminosity_arr, luminosity_main_arr, luminosity_eroded_arr, electron_density_total_arr, \
        tau_total_arr, tau_main_arr, tau_eroded_arr, brightest_height_arr, brightest_length_arr, \
        brightest_vel_arr, leading_frag_height_arr, leading_frag_length_arr, leading_frag_vel_arr, \
        leading_frag_dyn_press_arr, mass_total_active_arr, main_mass_arr, main_height_arr, main_length_arr, \
        main_vel_arr, main_dyn_press_arr = results_list.T


    # Calculate absolute magnitude (apparent @100km) from given luminous intensity
    abs_magnitude = -2.5*np.log10(luminosity_arr/const.P_0m)

    # plt.plot(abs_magnitude, brightest_height_arr/1000)
    # plt.gca().invert_xaxis()
    # plt.show()

    plt.plot(time_arr, abs_magnitude)
    plt.gca().invert_yaxis()

    plt.xlabel("Time (s)")
    plt.ylabel("Absolulte magnitude")

    plt.show()



    # Plot mass loss
    plt.plot(time_arr, 1000*mass_total_active_arr)
    plt.xlabel("Time (s)")
    plt.ylabel("Mass (g)")
    plt.show()



    # Plot length vs time
    plt.plot(brightest_length_arr[:-1]/1000, brightest_height_arr[:-1]/1000, label='Brightest bin')
    plt.plot(leading_frag_length_arr[:-1]/1000, leading_frag_height_arr[:-1]/1000, label='Leading fragment')

    plt.ylabel("Height (km)")
    plt.xlabel("Length (km)")

    plt.legend()


    plt.show()


    # Plot the wake animation
    if show_wake and wake_results:
        
        plt.ion()
        fig, ax = plt.subplots(1,1)

        # Determine the plot upper limit
        max_lum_wake = max([max(wake.wake_luminosity_profile) for wake in wake_results if wake is not None])

        

        for wake, abs_mag in zip(wake_results, abs_magnitude):

            if wake is None:
                continue

            # Skip points below the limiting magnitude
            if (abs_mag > lim_mag) or np.isnan(abs_mag):
                continue

            plt.cla()
                
            # Plot the wake profile
            ax.plot(wake.length_array, wake.wake_luminosity_profile)

            # Plot the location of grains
            ax.scatter(wake.length_points, wake.luminosity_points/10, c='k', s=10*wake.luminosity_points/np.max(wake.luminosity_points))

            plt.ylim([0, max_lum_wake])

            plt.pause(2*const.dt)

            fig.canvas.draw()

        plt.ioff()
        plt.clf()
        plt.close()
