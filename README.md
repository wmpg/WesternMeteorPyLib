

## Cloning

Because this repository contains submodules, you will have to clone it by running:

```
git clone --recursive https://github.com/wmpg/WesternMeteorPyLib.git
```

Running a normal git clone **will not work.**



## Features

A library of common functions used for meteor physics, developed by the [Western Meteor Physics Group](http://meteor.uwo.ca/).

List of features:

 * I/O functions for common formats of meteor data
 * Trajectory estimation methods
   * Gural at al. (2012) Multi-parameter fit method with PSO
   * Monte Carlo method
 * Orbit computation
 * Meteor shower/trajectory simulaton
 * Obtaining atmosphere densities using the NRLMSISE-00 model
 * D criteria functions
 * Parent body search
 * Coordinate system transforms
 * Solar longitude calculation (forward and inverse)
 * Orbit visualizations
 * Python 2 and 3 compatible



## Installation

The two sections below describe how to install the library on both Linux and Windows.

### Linux

These are installation instructions for Linux, assuming you have [Anaconda](https://www.anaconda.com/distribution/) installed. You might want to install this in a separate [virtual environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) in Anaconda. I recommend creating a separate environment called ```wmpl``` for this code:

```
conda create -y --name wmpl python=3.11
```

Every time you run the code, you **will have to activate the environment by typing:**

```
conda activate wmpl
```

on some systems this may not work, so you will have to write ```source activate wmpl``` instead.

We will now install all needed libraries. With the environment activated as described above, run this in the terminal:

```
conda install -y -c conda-forge numpy scipy matplotlib cython pytz pyqt pandas
conda install -y -c conda-forge jplephem ephem
conda install -y -c conda-forge watchdog h5py
conda install -y -c conda-forge numba
conda install -y -c conda-forge basemap basemap-data-hires
conda install -y -c conda-forge cartopy paramiko
```


Next, navigate to a folder where you will be keeping the code, e.g. ```~/source/```. Create this folder if it doesn't exist.


Then clone this repository:

```
git clone --recursive https://github.com/wmpg/WesternMeteorPyLib.git
```

After cloning/downloading this library, navigate into it with a terminal and run:

```
python setup.py install
```

this step might take a while because it will download the DE430 ephemerids.

Note that every time after closing the terminal or Anaconda prompt window, you will have to activate the ```wmpl``` enviroment by:
```
conda activate wmpl
```

If you experience any issues, please see the "Troubleshooting" section below.



### Windows

The installation might differ on Windows. I recommend installing Anaconda, which should install most of the packages you will need. Contact me for more details about Windows installation if you are stuck.


1) Install [Anaconda Python 3.*](https://www.anaconda.com/download/), IMPORTANT: during the installation, make sure to select the following:

	a) Check the checkbox which tells you to add anaconda path to system path: "Add Anaconda to my PATH envorinment variable."

	b) Install only for "me" (single user)


2) Open the Anaconda powershell and run:
	```
	conda update anaconda
	conda create -y --name wmpl python=3.11
	conda activate wmpl
	conda install -y -c conda-forge numpy scipy matplotlib cython pytz pyqt pandas
	conda install -y -c conda-forge jplephem ephem
	conda install -y -c conda-forge watchdog h5py
	conda install -y -c conda-forge numba
	conda install -y -c conda-forge basemap basemap-data-hires
 	conda install -y -c conda-forge cartopy paramiko
	```

3) Download and install git: [https://git-scm.com/downloads](https://git-scm.com/downloads)


4) Open git bash, navigate to where you want to pull the code and run:
	```
	git clone --recursive https://github.com/wmpg/WesternMeteorPyLib.git
	```
	


5) From the Anaconda powershell prompt navigate to the the cloned WesternMeteorPyLib directory and inside run:
	```
	python setup.py install
	```

Note that every time after closing the terminal or Anaconda prompt window, you will have to activate the ```wmpl``` enviroment by:
```
conda activate wmpl
```

If you experience any issues, please see the "Troubleshooting" section below.


#### Troubleshooting

If you are getting the following error on Windows: ```Unable to find vcvarsall.bat```, that means you need to install [Visual C++ Build Tools 2015](http://go.microsoft.com/fwlink/?LinkId=691126) or 2022. 

If you are getting this error when running the setup: ```ModuleNotFoundError: No module named 'wmpl.PythonNRLMSISE00.nrlmsise_00_header'```, it means that you haven't cloned the repository as per instructions. Please read this README file more carefully (hint: the answer is at the top of the file).

##### ```KeyError: 'PROJ_LIB'```
The basemap conda package is terribly broken and no one seems to care to fix it, so we have to do a little bit of "hacking". 

First, find where your anaconda is installed. Under Windows, it is probably in ```C:\Users\<YOUR_USERNAME>\AppData\Local\Continuum\anaconda3\``` or ```C:\Users\<YOUR_USERNAME>\Anaconda3\```, where you should replace <YOUR_USERNAME> with your username (duh!). From now on I will refer to this path as ```<ANACONDA_DIR>```. 

In this folder, look in the ```pkgs``` folder. You should find a folder named something like ```proj-9.5.1-h4f671f6_0```. The exact details vary and you may have more than one version installed. Make a note of the name of the newest one. I will refer to this as ```<PROJDIR>```

Now, create a new Environment Variable. Note the differences between Windows and Linux.

Windows: ```<ANACONDA_DIR>\pkgs\<PROJDIR>\Library\share\proj\```
Linux:   ```<ANACONDA_DIR>/pkgs/<PROJDIR>/share/proj/```

For example on my Windows PC, the location is 
```C:\Users\Mark\miniconda3\pkgs\proj-9.5.1-h4f671f6_0\Library\share\proj\```
and on my Linux server it is 
```$HOME/miniconda3/pkgs/proj-9.3.1-he5811b7_0/share/proj```

Restart any terminal, CMD or Powershell windows, and you should find that the error has been resolved.  If not, try one of the other versions of proj thats probably in your pkgs folder. 

##### Missing epsg file
If you get this error:

```FileNotFoundError: [Errno 2] No such file or directory: '<ANACONDA_DIR>/envs/wmpl/Library/share\\epsg'```

then you have to manually download the ```epsg``` file ([LINK](https://raw.githubusercontent.com/matplotlib/basemap/master/lib/mpl_toolkits/basemap/data/epsg), choose "Save as...", remove the ".txt" extension if necessary) into the ```<ANACONDA_DIR>/envs/wmpl/Library/share``` directory. Please don't hesitate to contact us if you have trouble getting this to work, as we share the same frustrations with the ```basemap``` library and the lack of any real alternative.


#### ```ImportError: cannot import name 'dedent' from 'matplotlib.cbook'``` error

In the recent matplotlib versions, the maintainters really droped the ball and started shipping buggy versions. Fortunately, we can revert to an old matplotlib version to fix this:

```
conda install -c conda-forge matplotlib=3.1.2
conda upgrade matplotlib
```

If it's still not working, try this:
```
conda install -c conda-forge matplotlib=2.2.4
```

#### ```ImportError: No module named PyQt5.QtWidgets``` error

Run:
```
pip install PyQt5
```

### Manually downloading data files

JPL DE430 ephemerids are not a part of the library, but they **will** be downloaded automatically on install. The file can be downloaded separately and put into the **`shared`** directory:

 * [JPL DE430 ephemerids](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de430.bsp)


If you want to use the most recent lists of comets and asteroids, download these as well, or run the ```UpdateOrbitFiles.py``` script:

 * [JPL comets elements](https://ssd.jpl.nasa.gov/dat/ELEMENTS.COMET)
 * [MPC Amors](http://cgi.minorplanetcenter.net/cgi-bin/textversion.cgi?f=lists/Amors.html)
 * [MPC Apollos](http://cgi.minorplanetcenter.net/cgi-bin/textversion.cgi?f=lists/Apollos.html)
 * [MPC Atens](http://cgi.minorplanetcenter.net/cgi-bin/textversion.cgi?f=lists/Atens.html)


## Usage

Module interfaces are not 100% complete yet, but individual functions are well documented. To run individual modules, e.g. to demonstrate how the trajectory solver works, run:

```
python -m wmpl.Trajectory.Trajectory
```

or, you can use functions from the library in other scripts. E.g. if you want to run a particular function from the library, you can create a new .py file and do:

```
import datetime
import math

# Import modules from WMPL
import wmpl.Utils.TrajConversions as trajconv
import wmpl.Utils.SolarLongitude as sollon

# Compute the Julian date of the current time
jd_now = trajconv.datetime2JD(datetime.datetime.now(datetime.timezone.utc))

# Get the solar longitude of the current time (in radians)
lasun = sollon.jd2SolLonJPL(jd_now)

print(math.degrees(lasun), ' deg')
```


## Citations

For academic use, please cite the paper:
>[Vida, D., Gural, P. S., Brown, P. G., Campbell-Brown, M., & Wiegert, P. (2019). Estimating trajectories of meteors: an observational Monte Carlo approach–I. Theory. Monthly Notices of the Royal Astronomical Society, 491(2), 2688-2705.](https://academic.oup.com/mnras/article-abstract/491/2/2688/5626361)
