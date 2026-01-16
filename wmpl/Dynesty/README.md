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

The tool searches a specified input directory for `.pickle` data files. If multiple matching pickle files are found for the same event, e.g. one containing the light curve and another containing the decleration, it creates a combined dataset.

**Key Features:**
* **Automated Data Fusion:** Combines data streams automatically based on timestamps.
* **Robust Execution:** Designed as a "run and forget" tool. If a run fails, it logs the error and proceeds to the next solution without halting.
* **Resume Capability:** If interrupted, the code can resume from the existing `.dynesty` file without overwriting previous progress.
* **MetSim Compatibility:** Supports MetSim JSON data inputs.

---

## System Requirements

* **Operating System:** **Linux is strongly recommended**.
    * *Note:* Running on Windows is possible but discouraged, as the process may hang or freeze during long simulations.
* **Hardware:** These simulations are computationally intensive.
    * Typical runtime: **1 to 2 days** on 96 cores per meteor.
    * Complex cases (bright fireballs or bad data): Up to **6 days** per event.
---

## Usage

### Basic Command
To run the simulation, provide the path to the input folder containing your data and the path to the configuration files.

```
python -m wmpl.Dynesty.DynestyMetSim "PATH_TO_INPUT_FOLDER" --output_dir "PATH_TO_OUTPUT_FOLDER" --prior "PATH_TO_PRIOR_FILE"
```

* **Input Folder:** Can contain either a single pickle trajectory file, or a folder containing multiple pickle trajectory files for batch processing. Multiple paths can be separated by a comma.
* **Output Folder:** If unspecified, results are saved in the input directory.
* **Prior File:** Bayesian priors for the model parameters. Select one of the templates in wmpl/Dynesty/priors, as appropriate for your case, of create a custom file.If unspecified, the tool looks for a `.prior` file in the input directory. If none is found, it defaults to a single fragmentation model (stony_meteoroid_1ErosFrag.prior).

**Example with multiple inputs:**
```
python -m wmpl.Dynesty.DynestyMetSim "PATH_TO_INPUT_FOLDER_1, PATH_TO_INPUT_FOLDER_2" --output_dir "PATH_TO_OUTPUT_FOLDER" --prior "PATH_TO_PRIOR_FILE"
```

### Command Line Arguments

| Flag | Description |
| :--- | :--- |
| `--output_dir` | Specifies the directory where results and logs will be saved. |
| `--prior` | Path to the `.prior` configuration file. |
| `--extraprior` | Path to an `.extraprior` file for advanced tuning (e.g., dust release, erosion heights). |
| `--pick_pos` | Adjusts the pick position in the meteor frame (0 to 1). <br>• `0`: Leading edge (default).<br>• `0.5`: Centroid (recommended for fireballs). |
| `--cores` | Specify the number of CPU cores to use. Default uses all available cores. |
| `-new` | Forces a new simulation in the output folder. Prevents mixing data if a `.dynesty` file already exists (though separate folders are recommended). |
| `-all` | Uses all available data (ensure magnitudes overlap and declarations are computed from the same pickle file). |
| `-plot` | Generates plots based on the current state of the `.dynesty` file without running the simulation. Useful for checking progress. |
| `-NoBackup` | Skips the generation of the `posterior_backup.pkl.gz` file. Saves ~10-20 minutes if extended data is not needed. |

### Monitoring Progress
When the simulation is running, the terminal will display a status line (Dynesty progress):

`[Iteration count] [Time elapsed] ... [Log Likelihood Stats] ... [Evidence Estimates]`

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
The `.prior` file defines the range and distribution of parameters for the simulation. You can edit this file to adjust the model   .

**Format:**
`Variable_Name, Min/Sigma/Alpha, Max/Mean/Mode, Option`

* Lines starting with `#` are comments.
* Numpy functions (e.g., `np.pi`) are supported in expressions.

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
rho, 10, 4000, log                  # Density [kg/m^3] (Log10 applied)
sigma, 0.001/1e6, 0.05/1e6          # Ablation coeff [kg/J]
erosion_mass_min, 5e-12, 1e-9, log  # Min erosion mass [kg]
noise_lag, 10, nan, invgamma        # Lag noise [m] (Inverse Gamma)
noise_lum, 5, nan, invgamma         # Luminosity noise [J/s]
```

**Notes on Specific Settings:**
* **Fragmentation:** To disable double fragmentation, remove the second fragmentation parameters from the prior file.
* **Camera Settings:** If using non-standard cameras, fix the Zero Magnitude Power and FPS:
    ```properties
    P_0m, 840, fix
    fps, 20, fix
    ```

### Advanced Configuration
For greater flexibility, use the `--extraprior` flag with a specialized file. This allows tuning of over 30 variables, including dust release and specific erosion height changes.
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
* **`*_posterior_bands_vs_height.png`**: Shows the range where most solutions lie.
* **`*_rho_mass_weighted_distribution.png`**: Useful for multiple fragmentations; shows mass-weighted density.
* **`*_tau_distribution.png`**: Luminous efficiency ($\tau$) derived from photometric mass ($\tau = 2 E_{rad} / (m_{init} v_0^2)$).
* **`posterior_backup.pkl.gz`**: A compressed archive containing comprehensive simulation data (samples, weights, best guesses, erosion dynamics).

---

## Programmatic Data Access

The `posterior_backup.pkl.gz` file contains detailed simulation data stored in a dictionary structure.

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