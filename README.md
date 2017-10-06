# WesternMeteorPyLib

A library of common functions used for meteor physics, developed by the [Western Meteor Physics Group](http://meteor.uwo.ca/).

List of features:

 * I/O functions for common formats of meteor data
 * Trajectory estimation methods
   * Gural at al. (2012) Multi-parameter fit method with PSO
   * WMPG Monte Carlo metdhos
 * Orbit estimation
 * Meteor shower/trajectory simulaton
 * Obtaining atmosphere densities using the NRLMSISE-00 model
 * D criteria functions
 * Parent body search
 * Coordinate system transforms
 * Solar longitude calculation (forward and inverse)
 * Orbit visualizations
 * Python 2 and 3 compatible



## Installation

To clone this repository locally, run:

```
git clone --recursive git@github.com:dvida/WesternMeteorPyLib.git
```

After cloning/downloading this library, navigate into it with a terminal and run:

```
python setup.py install
```

This should install most of the libraries you need. You may have issues with basemap, in that case run:

```
pip install https://github.com/matplotlib/basemap/archive/v1.1.0.tar.gz
```


### Data files

JPL DE430 ephemerids are not a part of the library and have to be downloaded separately and put into the **`shared`** directory:

 * [JPL DE430 ephemerids](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de430.bsp)


If you want to use the most recent lists of comets and asteroids, download these as well:

 * [JPL comets elements](https://ssd.jpl.nasa.gov/dat/ELEMENTS.COMET)
 * [MPC Amors](http://cgi.minorplanetcenter.net/cgi-bin/textversion.cgi?f=lists/Amors.html)
 * [MPC Apollos](http://cgi.minorplanetcenter.net/cgi-bin/textversion.cgi?f=lists/Apollos.html)
 * [MPC Athens](http://cgi.minorplanetcenter.net/cgi-bin/textversion.cgi?f=lists/Atens.html)


## Usage

Module interfaces are not 100% complete yet, but individual functions are well documented. To run individual modules, e.g. to demonstrate how the Monte Carlo trajectory solver works, navigate into the WesternMeteorPyLib directory and run

```
python -m Trajectory.Trajectory
```