from __future__ import print_function, division, absolute_import

import numpy as np
import matplotlib.pyplot as plt

from wmpl.Utils.TrajConversions import rotatePolar
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
    def __init__(self):
        """ Adjustable model of sporadic sources. The sources are modeled as 2D von Mises distributions in
            ecliptic coordinates. The velocity is drawn from a Gaussian distribution.
        """

        # Init the sporadic model variables
        self.reset()


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



    def sample(self, n=1):
        """ Sample the sporadic model.

        Keyword argument:
            n: [int] Number of samples to return. 1 by default.

        Return:
            samples: [list] A list of lambda, beta pairs (degrees).

        """

        # Normalize fluxes to 0 to 1 range
        flux_norm = np.array(self.fluxes)
        flux_norm /= np.sum(flux_norm)

        samples = []

        # Draw n samples from the model
        for i in range(n):

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


            samples.append([np.degrees(lam_rot), np.degrees(bet_rot), vg])


        return samples





if __name__ == '__main__':

    ### Init the sporadic model ###

    spor_model = SporadicModel()

    # Values from: Jones, J., & Brown, P. (1993). Sporadic meteor radiant distributions: orbital survey 
    #   results. Monthly Notices of the Royal Astronomical Society, 265(3), 524-532.
    #                    lam lam_sig bet bet_sig  vg  vg_sig flux
    
    spor_model.addSource(  71,     5,   0,    5,  25,  7.0,  0.35) # Helion source
    spor_model.addSource( 290,     5,   0,    5,  25,  7.0,  0.58) # Antihelion source
    spor_model.addSource(   0,     5,  15,    5,  60,  5.0,  0.46) # North apex source
    spor_model.addSource(   0,     5, -15,    5,  60,  5.0,  0.46) # South apex source
    spor_model.addSource(   0,    10,  60,    5,  30,  5.0,  1.0) # North toroidal source
    spor_model.addSource( 180,    50,   0,   20,  20, 10.0,  0.5) # Antapex source (no reference)

    ###############


    # Sample the model
    samples = spor_model.sample(1000)

    lam_list, bet_list, vg_list = np.array(samples).T

    lam_list = np.radians(lam_list)
    bet_list = np.radians(bet_list)

    # Rotate lambda to be Sun-centred
    lam_list = lam_list%(2*np.pi)


    # Plot the radiants
    cp = CelestialPlot(lam_list, bet_list)

    cp.scatter(lam_list, bet_list, s=1)

    #plt.scatter(lam_list, bet_list, s=1)

    #plt.ylim([-90, 90])
    #plt.xlim([0, 360])

    plt.show()


    # Plot the velocity histogram
    plt.hist(vg_list)

    plt.xlabel('Vg (km/s)')
    plt.ylabel('Count')

    plt.show()
