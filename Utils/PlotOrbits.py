""" A script used for plotting asteroid, meteoroid and comet orbits in the Solar System. """


# The MIT License

# Copyright (c) 2016 Denis Vida

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


# The source of equations used:
# http://farside.ph.utexas.edu/teaching/celestial/Celestialhtml/node34.html

from __future__ import print_function, division, absolute_import

import os
from datetime import datetime
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

from Utils.Plotting import savePlot, Arrow3D, set3DEqualAxes

# Define Julian epoch
J2000_EPOCH = datetime(2000, 1, 1, 12) # At the noon of 2000/01/01 UTC



class Planet(object):
    """ Defines a planet in the Solar System. """

    def __init__(self, a, lambda0, e, I, lon_of_peri, node, T, color='blue', size=20):
        """
        Arguments: 
            a: [float] semi-major axis (AU)
            lambda0: [float] mean longitude at epoch (degrees)
            e: [float] eccentricity
            I: [float]: inclination (degrees)
            lon_of_peri: [float] longitude of perihelion (degrees)
            node: [float] longitude of ascending node (degrees)
            T: [float]: orbital period (years)

        Keyword arguments:
            color: [str] planet color
            size: [float] planet plot size

        """

        self.a = a
        self.lambda0 = lambda0
        self.e = e
        self.I = I
        self.lon_of_peri = lon_of_peri
        self.node = node
        self.T = T
        self.color = color
        self.size = size

        # Mean orbital angular velocity, in radians per year
        self.n = 2*np.pi/self.T



    def getPosition(self, t):
        """ Returns the planet's position in a given time. 

        Arguments:
            t: [datetime] a point in time of the planet's orbit

        Return:
            (x, y, z): [tuple of floats] Cartesian coordinates of the position in the orbit

        """

        # Find the eccentric anomaly
        E = self.solveForE(t)

        # Get the position of the planet in its orbit
        x, y, z = orbitalElements2Cartesian(self.a, self.e, self.I, self.lon_of_peri - self.node, self.node, E)

        return x, y, z



    def solveForE(self, t, E=0, n=15):
        """ Find the eccentric anomaly via the iterative method.

        Arguments:
            t: [float] a point in time of the planet's orbit (years from epoch)

        Keyword arguments:
            E: [float] initial value of the eccentric anomaly for iteration
            n: [int] number of iterations

        Return:
            E: [float] eccentric anomaly
        """

        # Time of perihelion passage
        tau = np.radians(self.lon_of_peri - self.lambda0)/self.n

        # Mean anomaly
        M = self.n*(t - tau)


        def f(E, e, M):
            return M + e*np.sin(E)


        # Solve for eccentric anomaly using the iterative method
        for i in range(n):
            E = f(E, self.e, M)

        return E



    def plotPlanet(self, ax, time):
        """ Plot the planet and its orbit on a 3D plot. 

        Arguments:
            ax: [matplotlib object] 3D plot axis handle
            time: [float] years from J2000.0 epoch

        """

        # Eccentric anomaly (all ranges)
        E = np.linspace(-np.pi, np.pi, 100)

        # Plot the planet
        x, y, z = self.getPosition(time)
        ax.scatter(x, y, z, c=self.color, s=self.size, edgecolors='face')

        # Plot planet's orbit
        x, y, z = orbitalElements2Cartesian(self.a, self.e, self.I, self.lon_of_peri - self.node, 
            self.node, E)

        ax.plot(x, y, z, color=self.color, linestyle='-', linewidth=0.5)




def orbitalElements2Cartesian(a, e, I, peri, node, E):
    """ Convert orbital elements to Sun-centered Cartesian coordinates in the Solar System.

    Arguments:
        a: [float] semi-major axis (AU)
        e: [float] eccentricity
        I: [float] inclination (degrees)
        peri: [float] longitude of perihelion (degrees)
        node: [float] longitude of ascending node (degrees)
        E: [float/ndarray] eccentric anomaly (radians)

    Return:
        (x, y, z): [tuple of floats/ndarrays] Sun-centered Cartesian coordinates of the orbit

    """

    # Check if the orbit is parabolic or hyperbolic, if it is, set it to a very high eccentricity
    if e >=1:
        e = 0.99999999


    # Convert degrees to radians
    I, peri, node = map(np.radians, [I, peri, node])

    # True anomaly
    theta = 2*np.arctan(np.sqrt((1.0 + e)/(1.0 - e))*np.tan(E/2.0))

    # Distance from the Sun to the point on orbit
    r = a*(1.0 - e*np.cos(E))

    # Cartesian coordinates
    x = r*(np.cos(node)*np.cos(peri + theta) - np.sin(node)*np.sin(peri + theta)*np.cos(I))
    y = r*(np.sin(node)*np.cos(peri + theta) + np.cos(node)*np.sin(peri + theta)*np.cos(I))
    z = r*np.sin(peri + theta)*np.sin(I)

    return x, y, z



def plotPlanets(ax, time):
    """ Plots the Solar system planets. 

    Arguments:
        ax: [matplotlib object] 3D plot axis handle
        time: [float] years from J2000.0 epoch

    """

    # Define orbital elements of planets (J2000.0 epoch)
    mercury = Planet(0.3871, 252.25, 0.20564, 7.006, 77.46, 48.34, 0.241, color='#ecd67e', size=10)
    venus = Planet(0.7233, 181.98, 0.00676, 3.398, 131.77, 76.67, 0.615, color='#e7d520', size=30)
    earth = Planet(1.0000, 100.47, 0.01673, 0.000, 102.93, 0, 1.000, color='#1c7ff2', size=30)
    mars = Planet(1.5237, 355.43, 0.09337, 1.852, 336.08, 49.71, 1.881, color='#cc1e2c', size=20)
    jupiter = Planet(5.2025, 34.33, 0.04854, 1.299, 14.27, 100.29, 11.87, color='#D8CA9D', size=55)
    saturn = Planet(9.5415, 50.08, 0.05551, 2.494, 92.86, 113.64, 29.47, color='#ead6b8', size=45)
    uranus = Planet(19.188, 314.20, 0.04686, 0.773, 172.43, 73.96, 84.05, color='#287290', size=40)
    neptune = Planet(30.070, 304.22, 0.00895, 1.770, 46.68, 131.79, 164.9, color='#70B7BA', size=40)

    planets = [mercury, venus, earth, mars, jupiter, saturn, uranus, neptune]

    # Plot the Sun
    ax.scatter(0, 0, 0, c='yellow', s=100)

    # Plot planets
    for planet in planets:
        planet.plotPlanet(ax, time)



class OrbitPlotColorScheme(object):
    def __init__(self):
        """ Color scheme of the orbit plot. """

        self.dark()


    def dark(self):
        """ Dark orbit plot color scheme. """

        self.background = 'black'
        self.vernal_eq = 'w'
        self.planet_default = '#32CD32'


    def light(self):
        """ Light orbit plot color scheme. """

        self.background = '0.99'
        self.vernal_eq = 'black'
        self.planet_default = '#32CD32'



def plotOrbits(orb_elements, time, orbit_colors=None, plot_planets=True, plot_equinox=True, save_plots=False, 
    plot_path=None, plt_handle=None, color_scheme='dark', **kwargs):
    """ Plot the given orbits in the Solar System. 

    Arguments:
        orb_elements: [ndarray of floats]: 2D numpy array with orbits to plot, each entry contains:
            a - Semimajor axis (AU)
            e - Eccentricity
            I - Inclination (degrees)
            peri - Argument of perihelion (degrees)
            node - Ascending node (degrees)
        time: [float] datetime object of the moment of planet positions.

    Keyword Arguments:
        orbit_colors: [list] A list of size orb_elements.shape[0] containing color strings for every planet 
            orbit.
        plot_planets: [bool] If True, planets will be plotted. True by default.
        plot_equinox: [bool] Plots an arrow pointing to the vernal equinox if True. True by default.
        save_plots: [bool] If True, plots will be saved to the given path under plot_path. False by default.
        plot_path: [bool] File name and the full path where the plots will be saved if save_plots == True.
        plt_handle: [matplotlib plt handle] Pass the plot handle if some orbits need to be added to the plot.
        color_scheme: [str] 'dark' or 'light'. Dark by default.
        **kwargs: [dict] Extra keyword arguments which will be passes to the orbit plotter.
        
    """

    cs = OrbitPlotColorScheme()

    if color_scheme == 'light':
        cs.light()

    else:
        cs.dark()


    orb_elements = np.array(orb_elements)

    # Check the shape of given orbital elements array
    if len(orb_elements.shape) < 2:
        orb_elements = np.array([orb_elements])


    # Calculate the time difference from epoch to the given time (in years)
    julian = (time - J2000_EPOCH)
    years_diff = (julian.days + (julian.seconds + julian.microseconds/1000000.0)/86400.0)/365.2425

    if plt_handle is None:

        # Setup the plot
        fig = plt.figure()
        ax = fig.gca(projection='3d', axisbg=cs.background)

        # Set a constant aspect ratio
        ax.set_aspect('equal', adjustable='box-forced')

        # Hide the axes
        ax.set_axis_off()
        ax.grid(b=False)

    else:
        fig = plt_handle.gcf()
        ax = plt_handle.gca()

    # Plot the solar system planets
    if plot_planets:
        plotPlanets(ax, years_diff)

    # Plot the arrow pointing towards the vernal equinox
    if plot_equinox:

        # Plot the arrow
        a = Arrow3D([0, -4], [0, 0], [0, 0], mutation_scale=10, lw=1, arrowstyle="-|>", \
            color=cs.vernal_eq, alpha=0.5)
        ax.add_artist(a)

        # Plot the vernal equinox symbol
        ax.text(-4.1, 0, 0, u'\u2648', fontsize=8, color=cs.vernal_eq, alpha=0.5, \
            horizontalalignment='center', verticalalignment='center')


    # Eccentric anomaly (full range)
    E = np.linspace(-np.pi, np.pi, 100)

    # Plot the given orbits
    for i, orbit in enumerate(orb_elements):
        a, e, I, peri, node = orbit

        # Take extra steps in E if the orbit is very large
        if a > 50:
            E = np.linspace(-np.pi, np.pi, (a/20.0)*100)

        # Get the orbit in cartesian space
        x, y, z = orbitalElements2Cartesian(a, e, I, peri, node, E)

        # Check if the colors orbit are provided
        if orbit_colors:
            color = orbit_colors[i]
        else:
            # Set to default
            color = cs.planet_default

        # Plot orbits
        ax.plot(x, y, z, c=color, **kwargs)

    ax.legend()

    # Add limits (in AU)
    ax.set_xlim3d(-6, 6)
    ax.set_ylim3d(-6, 6)
    ax.set_zlim3d(-6, 6)

    # Set equal aspect ratio
    set3DEqualAxes(ax)

    # Save plots to disk (top and side views)
    if save_plots:

        # Check if the file names is given
        if plot_path is None:
            plot_file_name = 'orbit'
            plot_dir = '.'

        else:
            plot_dir, plot_file_name = os.path.split(plot_path)


        # Save top view
        ax.view_init(elev=90, azim=90)
        savePlot(plt, plot_file_name + '_orbit_top.png', plot_dir)

        # Save side view
        ax.view_init(elev=0, azim=90)
        savePlot(plt, plot_file_name + '_orbit_side.png', plot_dir)


    return plt




if __name__ == '__main__':

    # Time now
    #time = datetime.now()

    # Vernal equinox 2017
    time = datetime(2017, 3, 20, 6, 28)


    # Define orbits to plot
    # a, e, incl, peri, node
    orb_elements = np.array([
        [2.363, 0.515, 4.0, 205.0, 346.1],
        [0.989, 0.089, 3.1, 55.6, 21.2],
        [0.898, 0.460, 1.3, 77.1, 331.2],
        [184.585332285, 0.994914, 89.3950, 130.8767, 282.4633]
        ])

    # Plot orbits
    plotOrbits(orb_elements, time, color_scheme='light')

    plt.show()
