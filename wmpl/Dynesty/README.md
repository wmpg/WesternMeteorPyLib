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

The tool searches a specified input directory for `.pickle` data files. [cite_start]If multiple matching pickle files are found for the same event, e.g. one containing the light curve and another containing the decleration, it creates a combined dataset[cite: 2].

**Key Features:**
* [cite_start]**Automated Data Fusion:** Combines data streams automatically based on timestamps[cite: 2].
* **Robust Execution:** Designed as a "run and forget" tool. [cite_start]If a run fails, it logs the error and proceeds to the next solution without halting[cite: 3, 4].
* [cite_start]**Resume Capability:** If interrupted, the code can resume from the existing `.dynesty` file without overwriting previous progress[cite: 6].
* [cite_start]**MetSim Compatibility:** Supports MetSim JSON data inputs[cite: 11].

---

## System Requirements

* [cite_start]**Operating System:** **Linux is strongly recommended**[cite: 5].
    * [cite_start]*Note:* Running on Windows is possible but discouraged, as the process may hang or freeze during long simulations[cite: 5].
* [cite_start]**Hardware:** These simulations are computationally intensive[cite: 4].
    * Typical runtime: **1 to 2 days** on 96 cores per meteor.
    * Complex cases (bright fireballs or bad data): Up to **6 days** per event.
---

## Usage

### Basic Command
[cite_start]To run the simulation, provide the path to the input folder containing your data and the path to the configuration files[cite: 7, 17].

```
python -m wmpl.Dynesty.DynestyMetSim "PATH_TO_INPUT_FOLDER" --output_dir "PATH_TO_OUTPUT_FOLDER" --prior "PATH_TO_PRIOR_FILE"
```

* **Input Folder:** Can contain either a single pickle trajectory file, or a folder containing multiple pickle trajectory files for batch processing. [cite_start]Multiple paths can be separated by a comma[cite: 14].
* [cite_start]**Output Folder:** If unspecified, results are saved in the input directory[cite: 2, 16].
* **Prior File:** Bayesian priors for the model parameters. Select one of the templates in wmpl/Dynesty/priors, as appropriate for your case, of create a custom file.If unspecified, the tool looks for a `.prior` file in the input directory. If none is found, it defaults to a single fragmentation model (stony_meteoroid_1ErosFrag.prior).

**Example with multiple inputs:**
```
python -m wmpl.Dynesty.DynestyMetSim "PATH_TO_INPUT_FOLDER_1, PATH_TO_INPUT_FOLDER_2" --output_dir "PATH_TO_OUTPUT_FOLDER" --prior "PATH_TO_PRIOR_FILE"
```

### Command Line Arguments

| Flag | Description |
| :--- | :--- |
| `--output_dir` | [cite_start]Specifies the directory where results and logs will be saved[cite: 7]. |
| `--prior` | [cite_start]Path to the `.prior` configuration file[cite: 7]. |
| `--extraprior` | [cite_start]Path to an `.extraprior` file for advanced tuning (e.g., dust release, erosion heights)[cite: 55]. |
| `--pick_pos` | [cite_start]Adjusts the pick position in the meteor frame (0 to 1)[cite: 63]. <br>• `0`: Leading edge (default).<br>• `0.5`: Centroid (recommended for fireballs). |
| `--cores` | Specify the number of CPU cores to use. [cite_start]Default uses all available cores[cite: 65]. |
| `-new` | Forces a new simulation in the output folder. [cite_start]Prevents mixing data if a `.dynesty` file already exists (though separate folders are recommended)[cite: 58]. |
| `-all` | [cite_start]Uses all available data (ensure magnitudes overlap and declarations are computed from the same pickle file)[cite: 60]. |
| `-plot` | Generates plots based on the current state of the `.dynesty` file without running the simulation. [cite_start]Useful for checking progress[cite: 10]. |
| `-NoBackup` | Skips the generation of the `posterior_backup.pkl.gz` file. [cite_start]Saves ~10-20 minutes if extended data is not needed[cite: 67, 68]. |

### Monitoring Progress
When the simulation is running, the terminal will display a status line (Dynesty progress):

[cite_start]`[Iteration count] [Time elapsed] ... [Log Likelihood Stats] ... [Evidence Estimates]` [cite: 8]

**To visualize intermediate results:**
[cite_start]You can generate plots of the current progress without interrupting the run[cite: 9]:
```
python -m wmpl.Dynesty.DynestyMetSim "PATH_TO_INPUT_FOLDER" -plot
```
* [cite_start]This generates images in the output folder showing loaded data (e.g., `LumLag_pot`)[cite: 73].
* [cite_start]It updates the `log_` file with current stats[cite: 52].

### Resuming Runs
If a simulation is interrupted, simply run the command again pointing to the same output folder. [cite_start]The code will detect the `.dynesty` file and resume the calculation[cite: 6].

---

## Configuration

### The Prior File
The `.prior` file defines the range and distribution of parameters for the simulation. [cite_start]You can edit this file to adjust the model[cite: 18].

**Format:**
[cite_start]`Variable_Name, Min/Sigma/Alpha, Max/Mean/Mode, Option` [cite: 29]

* [cite_start]Lines starting with `#` are comments[cite: 18].
* [cite_start]Numpy functions (e.g., `np.pi`) are supported in expressions[cite: 20].

**Distribution Options:**
* [cite_start]**(default)**: Uniform distribution between Min and Max[cite: 21].
* [cite_start]`nan`: Sets default values or estimates the most likely value (e.g., for `v_init`, `zenith_angle`)[cite: 23].
* `norm`: Normal distribution. Interprets inputs as `Mean` and `Sigma`. (Recommended for velocity or zenith angle) [cite_start][cite: 24].
* `invgamma`: Inverse Gamma distribution. Interprets inputs as `Alpha` and `Mode`. (Recommended for noise uncertainty; peaks at mode with a long tail) [cite_start][cite: 25].
* `log`: Takes the log10 of the range. [cite_start]Use for priors spanning multiple orders of magnitude[cite: 26].
* [cite_start]`fix`: Fixes the parameter to the first value provided[cite: 27].

**Example Configuration:**

```properties
# name var, min/sigma/alpha, max/mean/mode, options

v_init, 500, nan, norm              # Velocity [m/s] (Gaussian)
zenith_angle, nan, fix              # Zenith angle [rad] (Fixed)
m_init, nan, nan                    # Initial mass [kg]
rho, 10, 4000, log                  # Density [kg/m^3] (Log10 applied)
sigma, 0.001/1e6, 0.05/1e6          # Ablation coeff [kg/J]
erosion_mass_min, 5e-12, 1e-9, log  # Min erosion mass [kg]
noise_lag, 10, nan, invgamma        # Lag noise [m] (Inverse Gamma)
noise_lum, 5, nan, invgamma         # Luminosity noise [J/s]
```

**Notes on Specific Settings:**
* [cite_start]**Fragmentation:** To disable double fragmentation, remove the second fragmentation parameters from the prior file[cite: 46].
* [cite_start]**Camera Settings:** If using non-standard cameras, fix the Zero Magnitude Power and FPS[cite: 48]:
    ```properties
    P_0m, 840, fix
    fps, 20, fix
    ```

### Advanced Configuration
For greater flexibility, use the `--extraprior` flag with a specialized file. [cite_start]This allows tuning of over 30 variables, including dust release and specific erosion height changes[cite: 55].
* [cite_start]*Warning:* Adding too many variables (>30) may prevent Dynesty from converging on a solution[cite: 56].

---

## Output Description

### Initial Output
Upon starting, the tool creates the output directory containing:
1.  [cite_start]Copies of input pickle and prior files[cite: 51].
2.  [cite_start]`report.txt`: Trajectory solution details[cite: 51].
3.  [cite_start]`log_[timestamp]_combined.txt`: Contains prior ranges and initial status[cite: 52].
4.  [cite_start]Initial data plots (to verify data loading)[cite: 52].

### Final Results
[cite_start]When finished, a `_results` folder is generated (e.g., `20191023_091225_results`)[cite: 74, 76].

**Key Files:**
* [cite_start]**`fit_plots/`**: Folder containing best-fit model images[cite: 81].
* [cite_start]**`*.json`**: Solution files compatible with the MetSim GUI[cite: 81].
* [cite_start]**`*_correlation_plot.png`**: Parameter correlations[cite: 76].
* [cite_start]**`*_posterior_bands_vs_height.png`**: Shows the range where most solutions lie[cite: 78].
* [cite_start]**`*_rho_mass_weighted_distribution.png`**: Useful for multiple fragmentations; shows mass-weighted density[cite: 78].
* [cite_start]**`*_tau_distribution.png`**: Luminous efficiency ($\tau$) derived from photometric mass ($\tau = 2 E_{rad} / (m_{init} v_0^2)$)[cite: 78, 79].
* [cite_start]**`posterior_backup.pkl.gz`**: A compressed archive containing comprehensive simulation data (samples, weights, best guesses, erosion dynamics)[cite: 83].

---

## Programmatic Data Access

[cite_start]The `posterior_backup.pkl.gz` file contains detailed simulation data stored in a dictionary structure[cite: 83, 85].

### Structure of Backup Data
```python
backup_small = {
    "dynesty": {
        "samples_eq": [...],          # Equal-weighted samples
        "weights": [...],             # Weights for samples
        "rho_mass_weighted_estimate": { ... } # Median, low95, high95
    },
    "best_guess": { ... },            # Best fit parameter values
    "bands": {                        # Posterior bands for plotting
        "lum": ..., "mag": ..., "vel": ..., "lag": ...
    },
    "const_backups": {                # Physics constants per simulation
        "rho_mass_weighted": ...,
        "erosion_beg_dyn_press": ...,
        "energy_per_mass_before_erosion": ...,
        # ... other dynamic variables
    }
}
```

### Python Example: Loading Data
[cite_start]Use the following snippet to load and access the backup file [cite: 123-131]:

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