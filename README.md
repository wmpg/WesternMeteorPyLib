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

The two sections below describe how to install the library on both Linux and Windows. Make sure to read the "Data files" section regardless of the 

### Linux

These are installation instructions for Linux. You might want to install this in a separate [virtual environment](https://www.dabapps.com/blog/introduction-to-pip-and-virtualenv-python/) in Python - in that case you should omit "sudo" in front of "pip" commands.


First, let's install all prerequisites:
```
sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get install -y libblas* liblapack-dev python-pip python-dev python-tk libgeos-3* libxml2-dev libxslt-dev python-dev lib32z1-dev
sudo pip install setuptools --upgrade
sudo pip install numpy --upgrade
sudo pip install matplotlib --upgrade
sudo pip install scipy
sudo pip install https://github.com/matplotlib/basemap/archive/v1.1.0.tar.gz
```


To clone this repository locally, run:

```
git clone --recursive https://github.com/dvida/WesternMeteorPyLib.git
```

After cloning/downloading this library, navigate into it with a terminal and run:

```
sudo python setup.py install
```

Finally, follow the section "Data files" below to complete the installation.


### Windows

On windows, you might not have to install library packages, but the installation might differ. I recommend installing Anaconda, which should install most of the packages you will need. Contact me for more details about Windows installation if you are stuck.


1) Install [Anaconda Python 3.*](https://www.anaconda.com/download/), IMPORTANT: during the installation, make sure to select the following:

	a) Check the checkbox which tells you to add anaconda path to system path: "Add Anaconda to my PATH envorinment variable."

	b) Install only for "me" (single user)


2) Open Anaconda prompt and run:
	```
	conda update anaconda
	conda install numpy scipy matplotlib
	conda install -c conda-forge basemap basemap-data-hires jplephem pyephem
	conda install -c obspy obspy
	```

3) Download and install git: [https://git-scm.com/downloads](https://git-scm.com/downloads)


4) Open git bash, navigate to where you want to pull the code and run:
	```
	git clone --recursive https://github.com/dvida/WesternMeteorPyLib.git
	```
	You will probably have to log in with your GitHub account to do that.


5) From Anaconda prompt navigate to the the cloned WesternMeteorPyLib directory and inside run:
	```
	python setup.py install
	```

6) Finally, follow the section "Data files" below to complete the installation.


### Data files

JPL DE430 ephemerids are not a part of the library and have to be downloaded separately and put into the **`shared`** directory:

 * [JPL DE430 ephemerids](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de430.bsp)


If you want to use the most recent lists of comets and asteroids, download these as well:

 * [JPL comets elements](https://ssd.jpl.nasa.gov/dat/ELEMENTS.COMET)
 * [MPC Amors](http://cgi.minorplanetcenter.net/cgi-bin/textversion.cgi?f=lists/Amors.html)
 * [MPC Apollos](http://cgi.minorplanetcenter.net/cgi-bin/textversion.cgi?f=lists/Apollos.html)
 * [MPC Athens](http://cgi.minorplanetcenter.net/cgi-bin/textversion.cgi?f=lists/Atens.html)


## Usage

Module interfaces are not 100% complete yet, but individual functions are well documented. To run individual modules, e.g. to demonstrate how the Monte Carlo trajectory solver works, navigate into the WesternMeteorPyLib directory and run:

```
python -m Trajectory.Trajectory
```