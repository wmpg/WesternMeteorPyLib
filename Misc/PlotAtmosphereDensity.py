""" Plots NRL MSISE atmosphere density model. """

import numpy as np
import matplotlib.pyplot as plt


from Config import config
from MetSim.MetSim import loadInputs, atmDensity


if __name__ == "__main__":

	
	# Load input meteor data
    met, consts = loadInputs(config.met_sim_input_file)


    heights = np.linspace(70, 120, 100)

    atmDensity_vect = np.vectorize(atmDensity, excluded=['consts'])

    atm_densities = atmDensity_vect(heights*1000, consts)


    plt.plot(atm_densities, heights, zorder=3)

    plt.xlabel('Density (g/cm^3)')
    plt.ylabel('Height (km)')

    plt.xlim(xmin=0)

    plt.grid()

    plt.title('NRLMSISE-00')

    #plt.gca().invert_yaxis()

    plt.savefig('atm_dens.png', dpi=300)

    plt.show()