""" Model of the sporadic background. """

from __future__ import print_function, division, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

from wmpl.TrajSim.SporadicSourcesModel import RadiantSample, extractRadiantSampleParameters, \
    initSporadicModel, SporadicModel
from wmpl.Utils.TrajConversions import date2JD, jd2Date, rotatePolar, raDec2Ecliptic
from wmpl.Utils.SolarLongitude import solLon2jdJPL, jd2SolLonJPL, jd2SolLonJPL_vect
from wmpl.Utils.PlotCelestial import CelestialPlot


def showerActivityModel(sol, flux_max, b, sol_max):
    """ Activity model taken from: Jenniskens, P. (1994).  Meteor stream activity I. The annual streams. 
        Astronomy and Astrophysics, 287., equation 8.

    Arguments: 
        sol: [float] Solar longitude for which the activity is computed (radians).
        flux_max: [float] Peak relative flux.
        b: [float] Slope of the shower.
        sol_max: [float] Solar longitude of the peak of the shower (radians).
    """

    # Compute the flux at given solar longitude
    flux = flux_max*10**(-b*np.degrees(np.abs(sol - sol_max)))

    return flux




def sampleActivityModel(b, sol_max, n_samples=1):
    """ Drawing samples from a probability distribution representing activity of a meteor shower. The sampling
        is done using the Inverse transform sampling method. Activity model taken from: Jenniskens, P. (1994). 
        Meteor stream activity I. The annual streams. Astronomy and Astrophysics, 287., equation 8.

    Arguments:
        sol_max: [float] Solar longitude of the maximum shower activity (radians).
        b: [float] Slope of the activity profile.

    Keyword arguments:
        n_samples: [float] Number of samples to be drawn from the activity profile distribution.

    """

    y = np.random.uniform(0, 1, size=n_samples)

    # Draw samples from the inverted distribution
    samples = np.sign(np.random.uniform(-1, 1, size=n_samples))*np.log10(y)/b + np.degrees(sol_max)

    return np.radians(samples)%(2*np.pi)



class MeteorShower(object):
    def __init__(self, ra_g, ra_g_sigma, dec_g, dec_g_sigma, d_ra, d_dec, v_g, v_g_sigma, d_vg, year, month, \
        sol_max, sol_slope):
        """ Meteor shower model.

        Arguments:
            ra_g: [float] Right ascension of centre of geocentric radiant (degrees).
            ra_g_sigma: [float] R.A. standard deviation of the radiant (degrees).
            dec_g: [float] Declination of centre of geocentric radiant (degrees).
            dec_g_sigma: [float] Declination standard deviation of the radiant (degrees).
            d_ra: [float] R.A. radiant drift (degrees of R.A. per degree of solar longitude).
            d_dec: [float] Dec radiant drift (degrees of declination per degree of solar longitude).
            v_g: [float] Mean geocentric velocity (km/s).
            v_g_sigma: [float] Standard deviation of the geocentric velocity (km/s).
            d_vg: [float] Vg drift in km/s per degree of solar longitude.
            year: [int] Year of the meteor shower peak.
            month: [int] Month of the meteor shower peak.
            sol_max: [float] Solar longitude of the maximum shower activity (degrees).
            sol_slope: [float] Slope of the activity profile.

        """

        # Convert angles to radiants
        self.ra_g = np.radians(ra_g)
        self.ra_g_sigma = np.radians(ra_g_sigma)
        self.dec_g = np.radians(dec_g)
        self.dec_g_sigma = np.radians(dec_g_sigma)
        self.d_ra = d_ra
        self.d_dec = d_dec

        self.v_g = 1000*v_g
        self.v_g_sigma = 1000*v_g_sigma
        self.d_vg = d_vg

        self.year = year
        self.month = month

        self.sol_max = np.radians(sol_max)
        self.sol_slope = sol_slope



    def sample(self, n_samples=1, jd_input=None):
        """ Sample the meteor shower.

        Keyword arguments:
            n_samples: [int] Number of samples to draw from the model. 1 by default.
            jd_input: [float] Julian date of the event. If not given, it will be drawn from the activity 
                profile.

        Return:
            samples: [list] A list of RadiantSample objects.

        """


        samples = []

        # Generate radiant samples
        for i in range(n_samples):

            if jd_input is None:

                # Sample the solar longitude
                la_sun = sampleActivityModel(self.sol_slope, self.sol_max)[0]

                # Calculate the corresponding Julian date for the drawn solar longitude
                jd = solLon2jdJPL(self.year, self.month, la_sun)

            else:

                jd = jd_input

                # Compute the solar longitude from the given julian date
                la_sun = jd2SolLonJPL(jd)



            # Sample radiant positions from a von Mises distribution centred at (0, 0)
            ra = np.random.vonmises(0, 1.0/(self.ra_g_sigma**2), 1)
            dec = np.random.vonmises(0, 1.0/(self.dec_g_sigma**2), 1)

            # Rotate R.A., Dec from (0, 0) to generated coordinates, to account for spherical nature of the angles
            # After rotation, (RA, Dec) will still be scattered around (0, 0)
            ra_rot, dec_rot = rotatePolar(0, 0, ra, dec)

            # Rotate all angles scattered around (0, 0) to the given coordinates of the centre of the distribution
            ra_rot, dec_rot = rotatePolar(ra_rot, dec_rot, self.ra_g, self.dec_g)


            # Apply the radiant drift
            ra_g_final = np.radians(np.degrees(ra_rot) + self.d_ra*np.degrees(la_sun - self.sol_max))
            dec_g_final = np.radians(np.degrees(dec_rot) + self.d_dec*np.degrees(la_sun - self.sol_max))

            # Compute ecliptic coordinates
            lam, bet = raDec2Ecliptic(jd, ra_g_final, dec_g_final)

            # Generate geocentric velocities from a Gaussian distribution
            v_g_final = np.random.normal(self.v_g, self.v_g_sigma, size=1)[0]

            # Apply the velocity drift
            v_g_final = v_g_final + 1000*self.d_vg*np.degrees(la_sun - self.sol_max)


            # Init the sample object
            sample = RadiantSample()
            sample.jd = jd
            sample.la_sun = la_sun
            sample.ra_g = ra_g_final
            sample.dec_g = dec_g_final
            sample.vg = v_g_final
            sample.lam = lam
            sample.bet = bet

            samples.append(sample)


        return samples


    def generate(self, jd_input=None):
        """ Draw a sample from meteor shower object. 
    
        Keyword arguments:
            jd_input: [float] Julian date of the event. If not given, it will be drawn from the activity 
                profile.

        Return:
            sample: [RadiantSample objects]

        """
        while True:
            yield self.sample()[0]




class CombinedSources(object):
    def __init__(self, source_models, rel_fluxes, start_jd, end_jd):
        """ Given a list which combines shower objects and/or the sporadic sources object, sample the
            combined model with the given relative fluxes.

        Arguments:
            source_models: [list] A list of MeteorShower or SporadicModel objects.
            ref_fluxes: [list] A list of floats representing relative fluxes.
            start_jd: [float] Julian date of the beginning of the sampling period.
            end_jd: [float] Julian date of the end of the sampling period.
        """

        self.source_models = source_models
        self.rel_fluxes = rel_fluxes


        ### Compute the activity profile of combined sources with the resolution of 30 minutes

        # Construct an array of Julian dates with a 30 min time step
        jd_arr = np.arange(start_jd, end_jd, 30.0/(24*60))

        # Compute solar longitude for every Julian date
        la_sun_arr = jd2SolLonJPL_vect(jd_arr)


        # Compute the total activity for every solar longitude and every source
        total_flux = np.zeros_like(la_sun_arr)
        self.source_flux_interp = []

        for i, (rel_flux, source) in enumerate(zip(rel_fluxes, source_models)):

            # If the source is sporadic, assume a constant flux
            if isinstance(source, SporadicModel):

                src_flux = np.zeros_like(la_sun_arr) + rel_flux

            # If the source is a shower, compute the flux from the activity profile
            else:

                src_flux = showerActivityModel(la_sun_arr, rel_flux, source.sol_slope, source.sol_max)

            # Add the source flux to the total flux
            total_flux += src_flux

            # Interpolate the source flux and save the interpolation handle
            src_flux_interp = scipy.interpolate.interp1d(jd_arr, src_flux)
            self.source_flux_interp.append(src_flux_interp)


        # Compute the cumulative sum of the total flux
        total_flux_cumsum = np.cumsum(total_flux)

        # Interpolate the cumulative as and use it as an inverse density function
        self.flux_inverse_density_func = scipy.interpolate.interp1d(total_flux_cumsum, jd_arr)

        # Save values of drawing random samples from the activity
        self.flux_cumsum_min = total_flux_cumsum[0]
        self.flux_cumsum_max = total_flux_cumsum[-1]



        #### Plot the total flux

        # Handle the case when the solar longitude wraps back to 0
        if np.max(la_sun_arr) - np.min(la_sun_arr) > np.radians(180):
            la_sun_arr[la_sun_arr < np.radians(180)] += 2*np.pi

        plt.plot(np.degrees(la_sun_arr), total_flux)

        plt.gcf().canvas.draw()
        
        # Wrap all xticks to 0-360 range
        locs, labels = plt.xticks()
        lbl_precision = len(labels[0].get_text().split('.')[-1])
        modified_xticks = [[loc, str(round(float(lbl.get_text())%360, lbl_precision))] for loc, lbl in \
            zip(locs, labels) if lbl.get_text()]
        locs = [loc for loc, lbl in modified_xticks]
        labels = [lbl for loc, lbl in modified_xticks]
        plt.xticks(locs, labels)

        plt.gca().set_ylim(ymin=0)

        plt.xlabel('Solar longitude (deg)')
        plt.ylabel('Total flux')

        plt.show()

        ######





    def sample(self, n_samples=1):
        """ Sample the combined source.

        Keyword arguments:
            n_samples: [int] Number of samples to draw from the model. 1 by default.
            

        Return:
            samples: [list] A list of RadiantSample objects.
        """

        samples = []

        # Draw n samples from the model
        for i in range(n_samples):

            # Draw a Julian date from the activity model
            u = np.random.uniform(self.flux_cumsum_min, self.flux_cumsum_max)
            jd = float(self.flux_inverse_density_func(u))

            # Compute the probabilities of choosing a source for the given Julian date
            source_probabilities = [src_flux_interp(jd) for src_flux_interp in self.source_flux_interp]

            # Normalize the relative fluxes
            source_probabilities = np.array(source_probabilities)
            source_probabilities /= np.sum(source_probabilities)

            # Choose a source by weighing it using its relative flux
            source = np.random.choice(self.source_models, p=source_probabilities)

            # Sample the source
            sample = source.sample(n_samples=1, jd_input=jd)[0]

            samples.append(sample)


        return samples




if __name__ == "__main__":

    # Number of samples to draw
    n_samples = 500


    ### METEOR SHOWER ###

    # Shower name
    shower_name = '2012Perseids'

    # Radiant position and dispersion
    ra_g = 48.2
    ra_g_sigma = 0.15

    dec_g = 58.1
    dec_g_sigma = 0.15

    # Radiant drift in degrees per degree of solar longitude
    d_ra = 1.40
    d_dec = 0.26

    # Geocentric velocity in km/s
    v_g = 59.1
    v_g_sigma = 0.1

    # Velocity drift
    d_vg = 0.0

    year = 2012
    month = 8

    # Solar longitude of peak activity in degrees
    sol_max = 140.0
    sol_slope = 0.4

    ###


    # Init the meteor shower model
    met_shower_model = MeteorShower(ra_g, ra_g_sigma, dec_g, dec_g_sigma, d_ra, d_dec, v_g, v_g_sigma, d_vg, \
        year, month, sol_max, sol_slope)


    ####################



    ### SPORADIC BACKGROUND MODEL ###

    # Generate Julian data data
    start_jd = date2JD(2012, 8,  1, 0, 0, 0)
    end_jd   = date2JD(2012, 8, 31,  0, 0, 0)

    # Init the sporadic background
    sporadic_model = initSporadicModel(start_jd, end_jd)


    ####################



    ### COMBINE THE SHOWERS AND THE SPORADIC BACKGROUND ###

    ## NOTE: The shower and the sporadic background may have different time spans, make sure they are the
    ##  the same when using the simulator!
    ##  The shower activity is defined by the peak solar longitude and the slope, but the sporadic background
    ##  range of activity is defined by the range of Julian dates.

    # Combine the shower and the sporadic background
    source_list = [met_shower_model, sporadic_model]

    # Refine relative fluxes betweent the shower and the sporadic background
    relative_fluxes = [100.0, 8.0]

    combined_model = CombinedSources(source_list, relative_fluxes, start_jd, end_jd)

    ####################



    # Sample the combined model
    samples = combined_model.sample(n_samples)

    # Extract parameters from the model
    samples_params = extractRadiantSampleParameters(samples)
    jd_list, la_sun_list, rag_list, decg_list, vg_list, lam_list, bet_list = map(np.array, samples_params)

    # Take out the solar longitude from source ecliptic longitude
    lam_list -= la_sun_list
    lam_list = lam_list%(2*np.pi)

    # Convert Vg to km/s
    vg_list /= 1000


    # Plot the radiants in ecliptic coordinates
    cp = CelestialPlot(lam_list, bet_list, bgcolor='w', lon_0=270)

    cp.scatter(lam_list, bet_list, s=1, c=vg_list)
    cp.colorbar(label='Vg (km/s)')

    plt.title('Sun-centred ecliptic')
    plt.show()


    # Plot the radiants in geocentric equatorial coordinates
    cp = CelestialPlot(rag_list, decg_list, bgcolor='w')

    cp.scatter(rag_list, decg_list, s=1, c=vg_list)
    cp.colorbar(label='Vg (km/s)')

    plt.title('Geocentric equatorial')
    plt.show()


    # Plot the velocity histogram
    plt.hist(vg_list)

    plt.xlabel('Vg (km/s)')
    plt.ylabel('Count')

    plt.show()