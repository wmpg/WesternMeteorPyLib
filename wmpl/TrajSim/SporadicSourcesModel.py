""" Model of the sporadic background. """

from __future__ import print_function, division, absolute_import

import numpy as np
import matplotlib.pyplot as plt

from wmpl.Utils.TrajConversions import rotatePolar, ecliptic2RaDec, date2JD
from wmpl.Utils.PlotCelestial import CelestialPlot


class SporadicSource(object):
    def __init__(self, lam, lam_sig, bet, bet_sig, vg, vg_sig, rel_flux):
        """ Container for sporadic source parameters. 
    
        Arguments:
            lam: [float] Ecliptic longitude of the source (deg).
            lam_sig: [float] Standard deviation of ecliptic longitude (deg).
            bet: [float] Ecliptic latitude of the source (deg).
            bet_sig: [float] Standard deviation of ecliptic latitude (deg).
            vg: [float] Mean geocentric velocity (km/s).
            vg_sig: [float] Standard deviation of the geocentric velocity (km/s).
            rel_flux: [float] Relative flux (no units).
        """

        # Convert angular values to radiants
        self.lam = np.radians(lam)
        self.lam_sig = np.radians(lam_sig)
        self.bet = np.radians(bet)
        self.bet_sig = np.radians(bet_sig)

        self.vg = vg
        self.vg_sig = vg_sig

        self.rel_flux = np.radians(rel_flux)




class SporadicModel(object):
    def __init__(self, start_jd, end_jd):
        """ Adjustable model of sporadic sources. The sources are modeled as 2D von Mises distributions in
            ecliptic coordinates. The velocity is drawn from a Gaussian distribution, the Julian dates are 
            drawn from a uniform distribution between start_jd and end_jd.

        Arguments:
            start_jd: [float] Julian date of the beginning of the sampling period.
            end_jd: [float] Julian date of the end of the sampling period.
        """

        # Init the sporadic model variables
        self.reset()

        self.start_jd = start_jd
        self.end_jd = end_jd


    def addSource(self, lam, lam_sig, bet, bet_sig, vg, vg_sig, rel_flux):
        """ Add a souradic source to the model. 
    
        Arguments:
            lam: [float] Ecliptic longitude of the source (deg).
            lam_sig: [float]
            bet: [float] Ecliptic latitude of the source (deg).
            bet_sig: [float]
            rel_flux: [float] Relative flux (no units).
        """

        self.sources.append(SporadicSource(lam, lam_sig, bet, bet_sig, vg, vg_sig, rel_flux))
        self.fluxes.append(rel_flux)


    def reset(self):
        """ Remove all source from the model. """

        self.sources = []
        self.fluxes = []



    def sample(self, n_samples=1):
        """ Sample the sporadic model.

        Keyword arguments:
            n_samples: [int] Number of samples to draw from the model. 1 by default.
            

        Return:
            samples: [list] A list of lambda, beta pairs (degrees).

        """

        # Normalize fluxes to 0 to 1 range
        flux_norm = np.array(self.fluxes)
        flux_norm /= np.sum(flux_norm)

        samples = []

        # Draw n samples from the model
        for i in range(n_samples):

            # Generate a Julian date
            jd = np.random.uniform(self.start_jd, self.end_jd)


            # Choose a source by weighing it using its relative flux
            source = np.random.choice(self.sources, p=flux_norm)


            # Sample radiant positions from a von Mises distribution centred at (0, 0)
            lam = np.random.vonmises(0, 1.0/(source.lam_sig**2), 1)
            bet = np.random.vonmises(0, 1.0/(source.bet_sig**2), 1)

            # Rotate angles from (0, 0) to generated coordinates, to account for spherical nature of the angles
            # After rotation, the angles will still be scattered around (0, 0)
            lam_rot, bet_rot = rotatePolar(0, 0, lam, bet)

            # Rotate all angles scattered around (0, 0) to the given coordinates of the centre of the distribution
            lam_rot, bet_rot = rotatePolar(lam_rot, bet_rot, source.lam, source.bet)


            # Draw the geocentric velocity
            vg = np.random.normal(source.vg, source.vg_sig)

            # Limit Vg from 11 to 71 km/s
            if vg < 11:
                vg = 11

            if vg > 71:
                vg = 71


            # Compute geocentric RA and Dec
            ra_g, dec_g = ecliptic2RaDec(jd, lam_rot, bet_rot)


            samples.append([np.degrees(lam_rot), np.degrees(bet_rot), vg, np.degrees(ra_g), \
                np.degrees(dec_g)])


        return samples




def initSporadicModel(start_jd, end_jd):
    """ Initialize the sporadic source model using the values from: 
        Jones, J., & Brown, P. (1993). Sporadic meteor radiant distributions: orbital survey results. 
        MNRAS, 265(3), 524-532.


    Arguments:
        start_jd: [float] Julian date of the beginning of the sampling period.
        end_jd: [float] Julian date of the end of the sampling period.

    """

    ### Init the sporadic model ###

    spor_model = SporadicModel(start_jd, end_jd)

    # Values from: 
    #                    lam lam_sig bet bet_sig  vg  vg_sig flux
    
    spor_model.addSource(  71,     5,   0,    5,  35, 10.0,  0.35) # Helion source
    spor_model.addSource( 290,     5,   0,    5,  35, 10.0,  0.58) # Antihelion source
    spor_model.addSource(   0,     5,  15,    5,  60,  5.0,  0.46) # North apex source
    spor_model.addSource(   0,     5, -15,    5,  60,  5.0,  0.46) # South apex source
    spor_model.addSource(   0,    10,  60,    5,  35,  5.0,  1.0) # North toroidal source
    spor_model.addSource(   0,    10, -60,    5,  35,  5.0,  1.0) # North toroidal source
    spor_model.addSource( 180,    50,   0,   20,  15, 10.0,  0.5) # Antapex source (no reference)

    ###############


    return spor_model




if __name__ == '__main__':

    n_samples = 1000


    # Generate Julian data data
    start_jd = date2JD(2018, 7, 27, 0, 0, 0)
    end_jd = date2JD(2018, 7, 28, 0, 0, 0)

    
    # Init the sporadic model
    spor_model = initSporadicModel(start_jd, end_jd)

    # Sample the model
    samples = spor_model.sample(n_samples)

    lam_list, bet_list, vg_list, rag_list, decg_list = np.array(samples).T

    lam_list = np.radians(lam_list)
    bet_list = np.radians(bet_list)

    rag_list = np.radians(rag_list)
    decg_list = np.radians(decg_list)

    # Rotate lambda to be Sun-centred
    lam_list = lam_list%(2*np.pi)


    # Plot the radiants in ecliptic coordinates
    cp = CelestialPlot(lam_list, bet_list, bgcolor='w')

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
