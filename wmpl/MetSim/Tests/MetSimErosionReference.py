""" Python implementation of the time step of the MetSim erosion model, the reference for the Cython loop over
    the fragments.

This module has the Fragment class and the ablateAll() and runSimulation() functions of MetSimErosion as a
loop over the fragments in Python. The tests in test_MetSimLoopReference.py run the same constants through
this module and through MetSimErosion and check that every result is the same to the last bit. The functions
outside of the loop (the grains, the atmosphere, the wake) are taken from MetSimErosion, so that only the
loop is compared. A change of the physics of the loop in MetSimErosion has to be made here as well.
"""

from __future__ import print_function, division, absolute_import

import math

import numpy as np

from wmpl.MetSim.MetSimErosion import G0, Wake, heightCurvature, erodeIntoGrains, generateFragments, \
    getErosionCoeff, erosionBinMass, killFragment, prepareAtmDensityTable, atmDensityTableCy
from wmpl.MetSim.MetSimErosionCyTools import luminousEfficiency, atmDensityPoly, ablationStepRK4Substep



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

        # Speed of the drop due to gravity across the path (m/s)
        self.v_grav_drop = 0

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

        # Fractional numbers of grains which the fragment eroded into each mass bin and did not release yet,
        #   and the parameters of the bins (see erodeIntoGrains)
        self.erosion_bin_counts = None
        self.erosion_bin_params = None

        # Indicate that a fragmentation entry set the erosion coefficient, so the height does not change it
        self.erosion_coeff_fixed = False

        # Indicate that this is the main fragment
        self.main = False

        # Indicate that the fragment is a grain
        self.grain = False

        # Indicate that this is born out of complex fragmentation
        self.complex = False

        # Identifier of the complex fragmentation entry
        self.complex_id = None

        # Last substep of the Runge-Kutta integration which resolved the motion, from which the next time
        #   step starts (0 to start from the full time step)
        self.rk4_substep = 0


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

        # The child starts without grains of its own to release, and does not share the list of the parent
        child.erosion_bin_counts = None
        child.erosion_bin_params = None

        return child



def ablateAll(fragments, const, compute_wake=False, wake_heights_queue=None, atm_table=None):
    """ Perform single body ablation of all fragments using the 4th order Runge-Kutta method.

    Arguments:
        fragments: [list] A list of Fragment instances.
        const: [object] Constants instance.

    Keyword arguments:
        compute_wake: [bool] If True, the wake profile will be computed. False by default.
        wake_heights_queue: [list] A list of heights at which the wake should be computed. None by default.
        atm_table: [AtmDensityTableCy object] Atmosphere density table from atmDensityTableCy(), to avoid
            converting the table on every time step. None by default, in which case it is made from the
            constants.

    Return:
        ...

    Note:
        const.dt is fixed for every fragment, including grains. A small grain deep in the atmosphere whose
        mass or speed changes within a fraction of a step is integrated in substeps (see ablationStepRK4).
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

    # Take the atmosphere density table once, not for every fragment
    if atm_table is None:
        atm_table = atmDensityTableCy(const)

    # Distance of the beginning from the centre of the Earth times the cosine of the zenith angle, from which
    #   the local zenith angle of the path is computed
    r_cos_zc = (const.h_init + const.r_earth)*math.cos(const.zenith_angle)

    # Take the density polynomial as a contiguous array of its 7 coefficients, used without the table
    if atm_table is None:
        dens_co = np.ascontiguousarray(const.dens_co, dtype=np.float64)
        if len(dens_co) != 7:
            raise ValueError("The density polynomial needs 7 coefficients, got {:d}".format(len(dens_co)))
    else:
        dens_co = np.zeros(7)

    # Go through all active fragments
    for frag in fragments:

        # Skip the fragment if it's not active
        if not frag.active:
            continue

        # Get atmosphere density for the given height
        if atm_table is None:
            rho_atm = atmDensityPoly(frag.h, const.dens_co)
        else:
            rho_atm = atm_table.density(frag.h)

        # Take the erosion coefficient of the fragment if it erodes
        if frag.erosion_enabled and (frag.erosion_coeff > 0):
            erosion_coeff = frag.erosion_coeff
        else:
            erosion_coeff = 0.0

        # Keep the mass and the speed at the beginning of the step for the luminous efficiency
        m_start = frag.m
        v_start = frag.v

        # Integrate the mass, the speed and the length over the step together, with the air density at the
        #   height of every stage, as the density changes by about 1% in a step of a fast meteor, and add up
        #   the radiated energy and the ionization over the step, starting from the substep which resolved the
        #   motion in the previous step
        mass_loss_total, velocity_change, length_change, energy_step, electrons_step, frag.rk4_substep \
            = ablationStepRK4Substep(const.dt, frag.K, frag.sigma + erosion_coeff, frag.m, frag.v, \
            frag.length, const.h_init, const.zenith_angle, const.r_earth, frag.h_grav_drop_total, atm_table, \
            dens_co, frag.sigma/(frag.sigma + erosion_coeff) if (frag.sigma + erosion_coeff) > 0 else 1.0, \
            const.lum_eff_type, const.lum_eff, const.mu, const.v_kill, frag.rk4_substep)

        # Split the mass loss into erosion and ablation, which have the same form, so that a fragment which
        #   does not erode loses no mass to erosion
        if erosion_coeff > 0:
            mass_loss_erosion = mass_loss_total*erosion_coeff/(frag.sigma + erosion_coeff)
        else:
            mass_loss_erosion = 0.0

        # Compute new mass
        m_new = frag.m + mass_loss_total

        # Take the average deceleration by the air over the step
        deceleration_total = velocity_change/const.dt

        # Compute g at given height
        gv = G0/((1 + frag.h/const.r_earth)**2)

        # Compute the local zenith angle of the straight path, at the distance of the point on the path
        #   from the centre of the Earth without the drop
        cos_z = (r_cos_zc - frag.length)/(frag.h + frag.h_grav_drop_total + const.r_earth)
        sin_z_sq = 1.0 - cos_z*cos_z
        sin_z = math.sqrt(sin_z_sq) if (sin_z_sq > 0) else 0.0

        # Gravity along the path changes the speed, and the drag is only the deceleration by the air
        accel_along = deceleration_total + gv*cos_z

        # Vertical component of a
        av = -accel_along*frag.vv/frag.v + frag.vh*frag.v/(const.r_earth + frag.h)

        # Horizontal component of a
        ah = -accel_along*frag.vh/frag.v - frag.vv*frag.v/(const.r_earth + frag.h)

        # Gravity across the path drops the meteoroid below it, and the drag slows the speed of the drop
        #   as it slows the meteoroid
        frag.v_grav_drop += (gv*sin_z + deceleration_total*frag.v_grav_drop/frag.v)*const.dt

        # Track the total drop of the height, the vertical part of the drop across the path
        frag.h_grav_drop_total += frag.v_grav_drop*sin_z*const.dt

        # Add the length gained from gravity along the path in this step
        length_change += 0.5*gv*cos_z*const.dt**2

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
        frag.length += length_change

        # Update the mass
        frag.m = m_new

        # Compute the height taking the curvature of the Earth and the gravity drop into account
        frag.h = heightCurvature(const.h_init, const.zenith_angle, frag.length, const.r_earth)
        frag.h -= frag.h_grav_drop_total

        # Take the luminous efficiency in the middle of the step, used to weight the efficiencies of the
        #   fragments in the output
        tau = luminousEfficiency(const.lum_eff_type, const.lum_eff, (v_start + frag.v)/2, \
            (m_start + frag.m)/2)

        # Compute the luminosity for one grain/fragment, from the ablated mass and the deceleration by the
        #   air, as the energy radiated over the step (from its substeps) divided by the step
        lum = energy_step/const.dt

        # Compute the electron line density
        q = electrons_step/const.dt

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
        dyn_press = 1.0*rho_atm*frag.v**2
        frag.dyn_press = dyn_press

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

            # Grains released at the speed and the height of a fragment which is too slow, too low or too far
            #   would be killed in their first step, which they would take outside of the range of the
            #   luminous efficiency, so they are released only by a fragment which has run out of mass
            release_grains = (frag.v >= const.v_kill) and (frag.h >= const.h_kill) \
                and not ((const.len_kill > 0) and (frag.length > const.len_kill))

            # Release the mass eroded in this step, as the grains are made below only for living fragments
            if release_grains and frag.erosion_enabled and (abs(mass_loss_erosion) > 0):
                grain_children = erodeIntoGrains(const, frag, abs(mass_loss_erosion))
                const.n_active += len(grain_children)
                frag_children_all += grain_children

            frag_children_all += killFragment(const, frag, release_grains=release_grains)

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
            if (not frag.disruption_child) and (not frag.erosion_coeff_fixed):
                frag.erosion_coeff = getErosionCoeff(const, frag.h)

            # Update the main fragment physical parameters if it is changed after erosion coefficient change
            # (only once - otherwise this would silently re-apply every tick and clobber any later manual
            # sigma change, e.g. from a complex fragmentation "A" event)
            if frag.main and (not frag.erosion_height_change_applied) \
                and (const.erosion_height_change >= frag.h):

                # Update the density
                frag.rho = const.erosion_rho_change
                frag.updateShapeDensityCoeff()

                # Update the ablation coeff
                frag.sigma = const.erosion_sigma_change

                frag.erosion_height_change_applied = True

        # Create grains for erosion-enabled fragments
        if frag.erosion_enabled:

            # Generate new grains if there is some mass to distribute
            if abs(mass_loss_erosion) > 0:

                grain_children = erodeIntoGrains(const, frag, abs(mass_loss_erosion))

                const.n_active += len(grain_children)
                frag_children_all += grain_children

                # Record physical parameters at the beginning of erosion for the main fragment
                if frag.main:
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

                # Deactivate the disrupted fragment
                frag.m = 0
                frag_children_all += killFragment(const, frag)

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

                            # Keep the erosion coefficient of the entry, which the height no longer changes
                            if frag_entry.erosion_coeff is not None:
                                frag.erosion_coeff = frag_entry.erosion_coeff
                                frag.erosion_coeff_fixed = True

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
                                new_frag_mass = parent_initial_mass*(frag_entry.mass_percent/100.0) \
                                    /frag_entry.number
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

                                # Assign the mass to the new fragment
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

                                # Add the new fragment to the list of children
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

            frag_children_all += killFragment(const, frag)

            continue

        # Fragment survived this tick - add its grain-weighted mass to the total active mass. Only
        #   fragments still active at the end of the tick reach here (killed/disrupted ones hit a
        #   'continue' above), matching the old post-loop "active only" scan exactly
        mass_total_active += frag.m*frag.n_grains

        # Add the mass the fragment eroded into its mass bins and did not release as grains yet
        mass_total_active += erosionBinMass(frag)

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

    # Ensure that the grain mass min is smaller than the grain mass max
    if const.erosion_mass_min > const.erosion_mass_max:
        const.erosion_mass_min, const.erosion_mass_max = const.erosion_mass_max, const.erosion_mass_min

    # Check the atmosphere density table, and convert it once to the arrays of the Cython interpolator
    prepareAtmDensityTable(const)
    atm_table = atmDensityTableCy(const)

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
            wake = ablateAll(fragments, const, compute_wake=compute_wake, \
            wake_heights_queue=wake_heights_queue, atm_table=atm_table)

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
