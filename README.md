

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
conda create --name wmpl python=3.7
```

Answer yes to all questions.

Every time you run the code, you **will have to activate the environment by typing:**

```
conda activate wmpl
```

on some systems this may not work, so you will have to write ```source activate wmpl``` instead.

We will now install all needed libraries. With the environment activated as described above, run this in the terminal:

```
conda install -y -c conda-forge numpy scipy matplotlib cython pytz
conda install -y -c conda-forge jplephem pyephem
conda install -y -c conda-forge basemap basemap-data-hires
pip install PyQt5
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
	conda create -y --name wmpl python=3.7
	conda activate wmpl
	conda install -y -c conda-forge numpy scipy matplotlib cython pytz
	conda install -y -c conda-forge jplephem pyephem statsmodels
	conda install -y -c conda-forge basemap basemap-data-hires
	pip install PyQt5
	```

3) Download and install git: [https://git-scm.com/downloads](https://git-scm.com/downloads)


4) Open git bash, navigate to where you want to pull the code and run:
	```
	git clone --recursive https://github.com/wmpg/WesternMeteorPyLib.git
	```
	You will probably have to log in with your GitHub account to do that.


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

If you are getting the following error on Windows: ```Unable to find vcvarsall.bat```, that means you need to install [Visual C++ Build Tools 2015](http://go.microsoft.com/fwlink/?LinkId=691126).

If you are getting this error when running the setup: ```ModuleNotFoundError: No module named 'wmpl.PythonNRLMSISE00.nrlmsise_00_header'```, it means that you haven't cloned the repository as per instructions. Please read this README file more carefully (hint: the answer is at the top of the file).

##### ```KeyError: 'PROJ_LIB'``` on Windows
The basemap conda package is terribly broken and no one seems to care to fix it, so we have to do a little bit of "hacking". First, find where your anaconda is installed. Under Windows, it is probably in ```C:\Users\<YOUR_USERNAME>\AppData\Local\Continuum\anaconda3\``` or ```C:\Users\<YOUR_USERNAME>\Anaconda3\```, where you should replace <YOUR_USERNAME> with your username (duh!). From now on I will refer to this path as ```<ANACONDA_DIR>```.
Open the following file in a text editor: ```<ANACONDA_DIR>\envs\wmpl\Lib\site-packages\mpl_toolkits\basemap\__init__.py```. 

Find the line ```pyproj_datadir = os.environ['PROJ_LIB']```, and comment it out by putting a # in front of it. Right below that command, add the following line:
```
pyproj_datadir = "<ANACONDA_DIR>/envs/wmpl/Library/share"
```
Just make sure to replace <ANACONDA_DIR> with the full path. Also, make sure to replace all backslashes ```\``` with forward slashes ```/``` in the path.

Save the file. Enjoy.

If after this you get this error:

```FileNotFoundError: [Errno 2] No such file or directory: '<ANACONDA_DIR>/envs/wmpl/Library/share\\epsg'```

then you have to manually download the ```epsg``` file ([LINK](https://raw.githubusercontent.com/matplotlib/basemap/master/lib/mpl_toolkits/basemap/data/epsg), choose "Save as...", remove the ".txt" extension if necessary) into the ```<ANACONDA_DIR>/envs/wmpl/Library/share``` directory. Please don't hesitate to contact us if you have trouble getting this to work, as we share the same frustrations with the ```basemap``` library and the lack of any real alternative.


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
jd_now = trajconv.datetime2JD(datetime.datetime.now())

# Get the solar longitude of the current time (in radians)
lasun = sollon.jd2SolLonJPL(jd_now)

print(math.degrees(lasun), ' deg')
```


## Citations

For academic use, please cite the paper:
>Vida, D., Gural, P., Brown, P., Campbell-Brown, M., Wiegert, P. (2019). *Estimating trajectories of meteors: an observational Monte Carlo approach*. **MNRAS**, submitted
