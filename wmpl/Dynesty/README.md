# DynestyMetSim

**DynestyMetSim** is a tool designed to automate fitting of the erosion model given in MetSimErosion to observational meteor data, namely meteor light curves and dynamics.

## Table of Contents
- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
  - [Basic Command](#basic-command)
  - [Monitoring Progress](#monitoring-progress)
  - [Resuming Runs](#resuming-runs)
- [Configuration](#configuration)
  - [The Prior File](#the-prior-file)
  - [Advanced Configuration](#advanced-configuration)
- [Output Description](#output-description)
- [Programmatic Data Access](#programmatic-data-access)

---

## Overview

The tool searches a specified input directory for `.pickle` data files. If multiple matching pickle files are found for the same input folder or any other subfolders, unite all the data together in a single dataset and run dynamic nested sampling (implemented via dynesty https://dynesty.readthedocs.io/en/v3.0.0/), each event found in the input directory is one by one proces and saved in a separate folder.

**Key Features:**
* **Automated Data Fusion:** Combines data streams automatically based on timestamps.
* **Robust Execution:** Designed as a "run and forget" tool. If a run fails, it logs the error and proceeds to the next solution without halting (the log file will be called log_error_ ).
* **Resume Capability:** If interrupted, the code can resume from the existing `.dynesty` file without overwriting previous progress. Note: .dynesty files can be finicky and may fail to load if they were created on a different machine or under a slightly different conda environment.
* **MetSim Compatibility:** Supports MetSim JSON data as inputs for model validation, the code will introduce noise (if requested) to test how the posteriory distribution is affectd by noise.

---

## System Requirements

* **Operating System:** **Linux is strongly recommended**.
    * *Note:* Running on Windows is possible but discouraged, as the process may hang or freeze during long simulations.
* **Hardware:** These simulations are computationally intensive.
    * Typical runtime: **1 to 2 days** on 96 cores per meteor.
    * Complex cases (bright fireballs or bad data): Up to **6 days** per event.

## installation--setup

Install Dynesty in wmpl conda enviroment.
```text
pip install dynesty
```
specific error with dynsty installation please check : https://dynesty.readthedocs.io/en/v3.0.0/index.html

---

## Usage

### Basic Command
To run the simulation, provide the path to the input folder containing your data and the path to the configuration files (if the configuration files are already in the input folder no need to specify).

```
python -m wmpl.Dynesty.DynestyMetSim "PATH_TO_INPUT_FOLDER" --output_dir "PATH_TO_OUTPUT_FOLDER" --prior "PATH_TO_PRIOR_FILE"
```

* **Input Folder:** Can contain either a single pickle trajectory file, or a folder containing multiple pickle trajectory files for batch processing. Multiple paths can be separated by a comma.
* **Output Folder:** The folder where it will be generated a new folder named after each event found in the input folder. If unspecified, results are saved in each of the input directory found with pickle data.
* **Prior File:** Bayesian priors for the model parameters. Select one of the templates in wmpl/Dynesty/priors, as appropriate for your case, of create a custom file. If unspecified, the tool looks for a `.prior` file in the input directory. If none is found, it defaults to a modified single fragmentation model (~stony_meteoroid_1ErosFrag.prior).

**Example with multiple inputs:**
```
python -m wmpl.Dynesty.DynestyMetSim "PATH_TO_INPUT_FOLDER_1, PATH_TO_INPUT_FOLDER_2" --output_dir "PATH_TO_OUTPUT_FOLDER" --prior "PATH_TO_PRIOR_FILE"
```
remember if you put both PATH_TO_INPUT_FOLDER_1 and PATH_TO_INPUT_FOLDER_2 in the same folder and run it from the folder where both are stored the code will do the exact same as separating them with a comma, as long as they have all different events (or for the events that share the same name it will try to merge camera data).

### Command Line Arguments

| Flag | Description |
| :--- | :--- |
| `--output_dir` | Specifies the directory where results and logs will be saved. |
| `--prior` | Path to the `.prior` configuration file. |
| `--extraprior` | Path to an `.extraprior` file for advanced tuning (e.g., dust release, erosion heights). |
| `--pick_pos` | Adjusts the pick position in the meteor frame (0 to 1). <br>• `0`: Leading edge (default).<br>• `0.5`: Centroid (recommended for fireballs). |
| `--cores` | Specify the number of CPU cores to use. Default uses all available cores. |
| `-new` | Forces a new simulation in the output folder does not continue the dynesty simulation if interrupted. Prevents mixing data if a `.dynesty` file already exists (though separate folders are recommended). |
| `-all` | Merges all available data from multiple cameras, only necesary if using EMCCD and CAMO narrow-field, wide-field data! By default the code will take lightcurve from EMCCD if not present it will combine CAMO narrow-field with wide-field data, while for the decelaration (lag) the code will take first CAMO narrow-field, then if not present EMCCD and if neither are present it will use CAMO wide-field. |
| `-plot` | Generates plots based on the current state of the `.dynesty` file without running the simulation. Useful to make sure everything is loaded correctly and when .dynesty file is created to check on progress. |
| `-NoBackup` | Skips the generation of the `posterior_backup.pkl.gz` file (gets ovewritten at the end of a run, it must save the backup). Saves ~5-20 minutes if extended data is not needed (if `posterior_backup.pkl.gz` is already present it's not going to generate a new file in any case, only if a run has finished). |

### Monitoring Progress
When the simulation is running, the terminal will display a status line (Dynesty progress):

```text
328433it [59:38:25,  7.78s/it, batch: 3 | bound: 4 | nc: 97 | ncall: 2878565 | eff(%):  0.971 | loglstar: -437.933 < -436.032 < -432.656 | logz: -472.176 +/-  0.219 | stop:  1.179]
```

What each part means:

* **`328433it`** — Total number of accepted **iterations** so far.
* **`[59:38:25, 7.78s/it, ...]`**

  * `59:38:25` = **elapsed wall time**
  * `7.78s/it` = **average time per iteration**
* **`batch: 3`** — **Dynamic nested sampling batch** index (Dynesty may add batches of live points to refine the solution).
* **`bound: 4`** — Current **bounding region** used to draw proposals (higher values usually mean later/updated bounds, often tighter regions around high-likelihood parts of parameter space).
* **`nc: 97`** — Number of **proposal attempts / likelihood calls needed to accept the last sample** (higher = harder to find a valid point under the current likelihood constraint).
* **`ncall: 2878565`** — **Total number of log-likelihood evaluations** so far (main driver of runtime).
* **`eff(%): 0.971`** — **Sampling efficiency** (accepted samples per likelihood call, in percent; low efficiency means many rejected proposals).
* **`loglstar: -437.933 < -436.032 < -432.656`** — **Log-likelihood constraint / bounds** summary for the current stage:

  * middle value is the current **likelihood threshold** (new samples must exceed this),
  * outer values summarize the current lower/upper context reported by Dynesty (early on these may show `-inf` and `inf`).
* **`logz: -472.176 +/- 0.219`** — Current estimate of **log-evidence** (`logZ`) and its **uncertainty**.
* **`stop: 1.179`** — **Stopping metric**: how close Dynesty is to termination (a “remaining evidence / stopping” diagnostic). When this drops below the configured stopping threshold, Dynesty stops.

more in: https://dynesty.readthedocs.io/en/v3.0.0/quickstart.html

**To visualize intermediate results:**
You can generate plots of the current progress without interrupting the run:
```
python -m wmpl.Dynesty.DynestyMetSim "PATH_TO_INPUT_FOLDER" -plot
```
* This generates images in the output folder showing loaded data (e.g., `LumLag_pot`).
* It updates the `log_` file with current stats.

### Resuming Runs
If a simulation is interrupted, simply run the command again pointing to the same output folder. The code will detect the `.dynesty` file and resume the calculation.

---

## Configuration

### The Prior File
The `.prior` file defines the range and distribution of parameters for the simulation. You can edit this file to adjust the model.

**Format:**
`Variable_Name, Min/Sigma/Alpha, Max/Mean/Mode, Option`

* Lines starting with `#` are comments.
* Numpy functions (e.g., `np.pi`) are supported in expressions, and you can also reference observation-based helper variables in expressions: h_beg (begin height, default=max(height)), h_end (end height, default=min(height)), h_peak (peak luminosity height), plus v_0/m_0/zc_0 (initial speed/mass/zenith-angle guesses) and n_lag0/n_lum0 (initial noise guesses for lag/lum)

**Distribution Options:**
* **(default)**: Uniform distribution between Min and Max.
* `nan`: Sets default values or estimates the most likely value (e.g., for `v_init`, `zenith_angle`).
* `norm`: Normal distribution. Interprets inputs as `Mean` and `Sigma`. (Recommended for velocity or zenith angle).
* `invgamma`: Inverse Gamma distribution. Interprets inputs as `Alpha` and `Mode`. (Recommended for noise uncertainty; peaks at mode with a long tail).
* `log`: Takes the log10 of the range. Use for priors spanning multiple orders of magnitude.
* `fix`: Fixes the parameter to the first value provided.

**Example Configuration:**

```properties
# name var, min/sigma/alpha, max/mean/mode, options

v_init, 500, nan, norm              # Velocity [m/s] (Gaussian)
zenith_angle, nan, fix              # Zenith angle [rad] (Fixed)
m_init, nan, nan                    # Initial mass [kg]
rho, 10, 4000                       # Density [kg/m^3]
sigma, 0.001/1e6, 0.05/1e6          # Ablation coeff [kg/J]
erosion_height_start,nan,nan     	  # erosion_height_start [m]
erosion_coeff,0.0,1e-6,log          # erosion_coeff [kg/J] (np.log10 applied)
erosion_mass_index,1,3          		# erosion_mass_index [-]
erosion_mass_min,5e-12,1e-9,log     # erosion_mass_min [kg] (np.log10 applied)
erosion_mass_max,1e-10,1e-7,log    	# erosion_mass_max [kg] (np.log10 applied)
noise_lag, 10, nan, invgamma        # Lag noise [m] (Inverse Gamma)
noise_lum, 5, nan, invgamma         # Luminosity noise [J/s]
```

The same can be express with helper variables as :
```properties
# name var, min/sigma/alpha, max/mean/mode, options

v_init, 500, v_0, norm              # Velocity [m/s] (Gaussian)
zenith_angle, zc_0, fix             # Zenith angle [rad] (Fixed)
m_init, 10**(np.floor(np.log10(m_0)-1)), 2*10**(np.floor(np.log10(m_0)+1))  # Initial mass [kg]
rho, 10, 4000, log                  # Density [kg/m^3] (Log10 applied)
sigma, 0.001/1e6, 0.05/1e6          # Ablation coeff [kg/J]
erosion_height_start,h_beg-100-(h_beg - h_peak)/2,h_beg + 100+(h_beg - h_peak)/2 # erosion_height_start [m]
erosion_coeff,0.0,1e-6,log          # erosion_coeff [kg/J] (np.log10 applied)
erosion_mass_index,1,3          		# erosion_mass_index [-]
erosion_mass_min,5e-12,1e-9,log     # erosion_mass_min [kg] (np.log10 applied)
erosion_mass_max,1e-10,1e-7,log    	# erosion_mass_max [kg] (np.log10 applied)
noise_lag, 10, n_lag0, invgamma     # Lag noise [m] (Inverse Gamma)
noise_lum, 5, n_lum0, invgamma      # Luminosity noise [J/s] (Inverse Gamma)
```

**Notes on Specific Settings:**
* **Fragmentation:** To disable double fragmentation, remove the second fragmentation parameters from the prior file (it will be set at 1 km effectivelly disabling it).
* **Camera Settings:** If using non-standard cameras, fix the Zero Magnitude Power and FPS (these will be the same values share across all the cameras if more than one camera type are present):
    ```properties
    P_0m, 840, fix
    fps, 20, fix
    ```

### Advanced Configuration
For greater flexibility, use the `--extraprior` flag with a specialized file. This allows tuning of over 30 variables, including dust release and specific erosion height changes (usefull for fireballs or to introduce a third fragmentation).
* *Warning:* Adding too many variables (>30) may prevent Dynesty from converging on a solution.

---

## Output Description

### Initial Output
Upon starting, the tool creates the output directory containing:
1.  Copies of input pickle and prior files.
2.  `report.txt`: Trajectory solution details.
3.  `log_[timestamp]_combined.txt`: Contains prior ranges and initial status.
4.  Initial data plots (to verify data loading).

### Final Results
When finished, a `_results` folder is generated (e.g., `20191023_091225_results`).

**Key Files:**
* **`fit_plots/`**: Folder containing best-fit model images.
* **`*.json`**: Solution files compatible with the MetSim GUI.
* **`*_correlation_plot.png`**: Parameter correlations.
* **`*_posterior_bands_vs_height.png`**: Shows the range where most solutions lie (generated only if the backup file is created).
* **`*_rho_mass_weighted_distribution.png`**: Useful for multiple fragmentations; shows mass-weighted density.
* **`*_tau_distribution.png`**: Luminous efficiency ($\tau$) derived from photometric mass ($\tau = 2 E_{rad} / (m_{init} v_0^2)$).
* **`posterior_backup.pkl.gz`**: A compressed archive containing comprehensive simulation data (samples, weights, best guesses, erosion dynamics).

---

## Programmatic Data Access

The `posterior_backup.pkl.gz` file contains detailed simulation data stored in a dictionary structure. This way if the .dynesty cannot be open in an other machine is always possible to open the `posterior_backup.pkl.gz` file and load the results. All plots can be recreated from the backup file except for 2 plots from dynesty that require to open the oiginal .dynesty file (i.e. _dynesty_runplot.png and _trace_plot.png).

### Structure of Backup Data
```python
backup_small = {
    "dynesty": {
        "file_name": file_name,  # event/run id

        "samples": [...],              # dynesty samples (nsamp, ndim)
        "importance_weights": [...],   # dynesty weights for samples
        "weights": [...],              # pipeline weights for samples

        "logl": [...],                 # log-likelihood per sample
        "logwt": [...],                # log-weights per sample
        "logz": [...],                 # evidence history (final = logz[-1])
        "logzerr": [...],              # evidence err history (or None)

        "niter": ...,                  # total iterations
        "ncall": ...,                  # total likelihood calls
        "eff": ...,                    # sampling efficiency (%)
        "summary": "...",              # summary text block

        "variables": [...],            # parameter names (matches samples columns)
        "flags_dict": {...},           # prior flags
        "fixed_values": {...},         # fixed params from .prior

        "median": [...],               # posterior median
        "mean": [...],                 # posterior mean
        "approx_modes": [...],         # approx posterior modes
        "95_CI_lower": [...],          # 95% CI lower
        "95_CI_upper": [...],          # 95% CI upper

        "rho_array": [...],            # rho per sample/sim
        "rho_mass_weighted_estimate": {...},  # rho stats [median][low95][high95]
        "const_backups": {...}         # extra physics diagnostics
    },
    "best_guess": {
            'luminosity': ..., "abs_magnitude": ..., "velocity": ..., "lag": ...
       }, # Best fit points
    "bands": {                        # Posterior bands for plotting
        "lum": ..., "mag": ..., "vel": ..., "lag": ...
    },
}
```

### Python Example: Loading Data
Use the following snippet to load and access the backup file:

```python
import gzip
import pickle
import os
import numpy as np

# Assuming 'name' is your filename and 'output_dir' is your path
backup_file = "20191023_091225_posterior_backup.pkl.gz"
file_path = os.path.join(output_dir, backup_file)

print(f"Using backup file: {backup_file}")

with gzip.open(file_path, "rb") as f:
    backup_small = pickle.load(f)

# Example: Accessing Mass Weighted Density (Rho)
rho_data = backup_small['dynesty']['rho_mass_weighted_estimate']
rho_median = rho_data['median']
rho_low = rho_data['low95']
rho_high = rho_data['high95']

print(f"Rho: {rho_median} (+{rho_high} / -{rho_low})")
```
