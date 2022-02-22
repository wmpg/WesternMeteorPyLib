import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy
from wmpl.Formats.Met import loadMet
from wmpl.MetSim.GUI import MetObservations, SimulationResults, collectPaths, loadConstants
from wmpl.MetSim.MetSimErosion import Constants, runSimulation
from wmpl.Utils.AtmosphereDensity import fitAtmPoly
from wmpl.Utils.Math import meanAngle, mergeDatasets, movingAverage, movingOperation
from wmpl.Utils.Pickling import loadPickle, savePickle


def loadPickleFilesFromDir(folder):
    """
    Return:
        [tuple] traj, met_obs, const
            - traj: [ObservedPoints]
            - met_obs: [MetObservations or None]
            - const: [Constants]
    """
    traj_pickle_file, met_file, _, load_file = collectPaths(folder, '', True, '.', get_latest=True)
    traj = loadPickle(*os.path.split(traj_pickle_file))

    # Load a METAL .met file if given
    met_obs = None
    if met_file is not None:
        if os.path.isfile(met_file):
            met = loadMet(*os.path.split(os.path.abspath(met_file)))
            met_obs = MetObservations(met, traj)
        else:
            print('The .met file does not exist:', met_file)
            sys.exit()

    # get const file
    if load_file is None:
        return None, None, None

    # Load the constants from the JSON files
    const, _ = loadConstants(load_file)
    const.dens_co = np.array(const.dens_co)

    # adjust density coefficient
    lat_mean = np.mean([traj.rbeg_lat, traj.rend_lat])
    lon_mean = meanAngle([traj.rbeg_lon, traj.rend_lon])

    const.dens_co = fitAtmPoly(
        lat_mean, lon_mean, const.h_init, max(traj.rend_ele - 5000, 15000), traj.jdt_ref
    )

    return traj, met_obs, const


def displayResiduals(traj, met_obs, const, sr):
    ##################################
    # Magnitude

    fig, ax = plt.subplots(2, sharey=True)
    # simulation magnitude
    mag_sim = scipy.interpolate.interp1d(
        sr.leading_frag_height_arr / 1000, sr.abs_magnitude, bounds_error=False, fill_value=0
    )

    for frag_entry in const.fragmentation_entries:
        if len(frag_entry.main_height_data):
            ax[0].plot(frag_entry.main_abs_mag, frag_entry.main_height_data / 1000, label='frag main')
        # Plot magnitude of the grains
        if len(frag_entry.grains_height_data):
            ax[0].plot(frag_entry.grains_abs_mag, frag_entry.grains_height_data / 1000, label='frag grains')

    ax[0].plot(sr.abs_magnitude, sr.leading_frag_height_arr / 1000, label='mag')
    ax[0].plot(sr.abs_magnitude_eroded, sr.leading_frag_height_arr / 1000, label='mag eroded')
    ax[0].legend()

    # collect data from objects
    for obs in traj.observations:
        # Skip instances when no magnitudes are present

        if obs.absolute_magnitudes is None:
            continue

        # Extract data
        abs_mag_data = obs.absolute_magnitudes[obs.ignore_list == 0]
        height_data = obs.model_ht[obs.ignore_list == 0] / 1000

        # Don't plot magnitudes fainter than 8
        mag_filter = abs_mag_data < 8
        height_data = height_data[mag_filter]
        abs_mag_data = abs_mag_data[mag_filter]

        ax[0].plot(abs_mag_data, height_data)
        # ax[1].scatter(abs_mag_data - mag_sim(height_data), height_data)
        ax[1].errorbar(
            movingAverage(abs_mag_data - mag_sim(height_data), 5, stride=2),
            movingAverage(height_data, 5, stride=2),
            xerr=movingOperation(abs_mag_data - mag_sim(height_data), np.std, 5, stride=2),
            capsize=3,
            marker='x',
            linestyle='none',
        )

    if met_obs is not None:
        # Plot additional magnitudes for all sites
        for site in met_obs.sites:

            # Extract data
            abs_mag_data = met_obs.abs_mag_data[site]
            height_data = met_obs.height_data[site] / 1000
            ax[0].plot(abs_mag_data, height_data, label='mag')
            # ax[1].scatter(abs_mag_data - mag_sim(height_data), height_data)
            ax[1].errorbar(
                movingAverage(abs_mag_data - mag_sim(height_data), 5, stride=2),
                movingAverage(height_data, 5, stride=2),
                xerr=movingOperation(abs_mag_data - mag_sim(height_data), np.std, 5, stride=2),
                capsize=3,
                marker='x',
                linestyle='none',
            )
    plt.show()

    #####################
    #####################
    # Velocity
    fig, ax = plt.subplots(2, sharey=True)
    vel_sim = scipy.interpolate.interp1d(
        sr.leading_frag_height_arr / 1000, sr.leading_frag_vel_arr / 1000, bounds_error=False, fill_value=0
    )

    for obs in traj.observations:
        # Extract data
        vel_data = obs.velocities[obs.ignore_list == 0][1:] / 1000
        height_data = obs.model_ht[obs.ignore_list == 0][1:] / 1000
        ax[0].plot(vel_data, height_data)
        # ax[1].scatter(vel_data - vel_sim(height_data), height_data)
        windowed_data = movingOperation(vel_data - vel_sim(height_data), n=8, stride=2, ret_arr=True)
        med = np.median(windowed_data, axis=1)
        ax[1].errorbar(
            med,
            movingAverage(height_data, n=8, stride=2),
            xerr=[
                med - np.percentile(windowed_data, 10, axis=1),
                np.percentile(windowed_data, 90, axis=1) - med,
            ],
            capsize=3,
            marker='x',
            linestyle='none',
        )

    ax[0].plot(sr.brightest_vel_arr / 1000, sr.brightest_height_arr / 1000, label='brightest')
    ax[0].plot(sr.leading_frag_vel_arr / 1000, sr.leading_frag_height_arr / 1000, label='leading frag')
    ax[0].legend()
    plt.show()

    ####################
    ####################
    # Lag
    fig, ax = plt.subplots(2, sharey=True)
    temp_arr = np.c_[sr.brightest_height_arr, sr.brightest_length_arr]
    temp_arr = temp_arr[
        (sr.brightest_height_arr <= traj.rbeg_ele)
        # & (sr.brightest_height_arr >= plot_end_ht)
    ]
    brightest_ht_arr, brightest_len_arr = temp_arr.T

    temp_arr = np.c_[sr.leading_frag_height_arr, sr.leading_frag_length_arr]
    temp_arr = temp_arr[
        (sr.leading_frag_height_arr <= traj.rbeg_ele)
        # & (sr.leading_frag_height_arr >= plot_end_ht)
    ]
    leading_ht_arr, leading_frag_len_arr = temp_arr.T

    leading_lag_sim = (
        leading_frag_len_arr
        - leading_frag_len_arr[0]
        - traj.orbit.v_init
        * np.arange(0, const.dt * len(leading_frag_len_arr), const.dt)[: len(leading_frag_len_arr)]
    )

    if len(brightest_len_arr):

        # Compute the simulated lag using the observed velocity
        brightest_lag_sim = (
            brightest_len_arr
            - brightest_len_arr[0]
            - traj.orbit.v_init
            * np.arange(0, const.dt * len(brightest_len_arr), const.dt)[: len(brightest_len_arr)]
        )
        brightest_interp = scipy.interpolate.interp1d(
            brightest_ht_arr, brightest_lag_sim, bounds_error=False, fill_value=0
        )

    for obs in traj.observations:

        # Get observed heights
        height_data = obs.model_ht[obs.ignore_list == 0]

        obs.lag[obs.ignore_list == 0], height_data / 1000

        obs_height_indices = height_data > np.min(brightest_ht_arr)
        obs_hts = height_data[obs_height_indices]
        brightest_residuals = obs.lag[obs.ignore_list == 0][obs_height_indices] - brightest_interp(obs_hts)

        ax[0].plot(obs.lag[obs.ignore_list == 0][obs_height_indices], obs_hts / 1000)
        # ax[1].scatter(brightest_residuals, obs_hts / 1000)
        ax[1].errorbar(
            movingAverage(brightest_residuals, 20, stride=4),
            movingAverage(obs_hts / 1000, 20, stride=4),
            xerr=movingOperation(brightest_residuals, np.std, 20, stride=4),
            capsize=3,
            marker='x',
            linestyle='none',
        )

    if met_obs is not None:
        for site in met_obs.sites:
            height_data = met_obs.height_data[site] / 1000

            # ax[0].plot(met_obs.lag_data[site], height_data, label='met_obs')

    ax[0].plot(
        brightest_lag_sim[: len(brightest_ht_arr)],
        (brightest_ht_arr / 1000)[: len(brightest_lag_sim)],
        label='brightest',
    )
    ax[0].plot(
        leading_lag_sim[: len(leading_ht_arr)],
        (leading_ht_arr / 1000)[: len(leading_lag_sim)],
        label='leading',
    )
    ax[0].legend()
    plt.show()


def compileFolderData(folder):
    """
    Collects all data from folder, decides whether it's good or not, then merges all data from different
    cameras then takes moving averages over them
    
    """
    ########################################
    ### collect all data for the folder ####
    traj, met_obs, const = loadPickleFilesFromDir(folder)
    if not const:
        return None

    frag_main, results_list, wake_results = runSimulation(const)
    sr = SimulationResults(const, frag_main, results_list, wake_results)

    # displayResiduals(traj, met_obs, const, sr)

    kept_data = {'mag': [], 'vel': [], 'lag': []}
    discard_traj_obs_list = []
    discard_met_obs_list = []
    data_type_window_size = {'mag': 5, 'vel': 5, 'lag': 20}

    def getGoodDataset(station, discard_list, height, val, val_sim, data_type='mag'):
        """ Add data to kept_data based on whether it agrees with simulation 
        Returns True if it was added """

        n = data_type_window_size[data_type]

        filter_index = np.isfinite(val)
        val = val[filter_index]
        height = height[filter_index]
        residuals = val - val_sim(height)

        windowed_residuals = movingOperation(residuals, n=n, ret_arr=True)
        mean_residuals = np.mean(windowed_residuals, axis=1)
        std_residuals = np.std(windowed_residuals, axis=1)
        print(
            data_type,
            station,
            'height',
            np.count_nonzero(~np.isnan(height)) / len(height),
            'val',
            np.count_nonzero(~np.isnan(val)) / len(val),
            np.mean(np.abs(mean_residuals) < std_residuals / 2),
            np.mean(np.abs(mean_residuals) < std_residuals),
            np.mean(np.abs(mean_residuals)),
            np.mean(std_residuals),
        )
        if (
            np.isnan(np.mean(mean_residuals))
            or np.isinf(np.mean(mean_residuals))
            or np.mean(mean_residuals) > 1e100
        ):
            print(height, val, residuals)

        if np.mean(np.abs(mean_residuals) < std_residuals) > 0.5:
            kept_data[data_type].append((height, val, residuals))

    mag_sim = scipy.interpolate.interp1d(
        sr.leading_frag_height_arr / 1000, sr.abs_magnitude, bounds_error=False, fill_value=0
    )

    #### Extract mag data ###
    for obs in traj.observations:
        if obs.absolute_magnitudes is None:
            continue

        abs_mag_data = obs.absolute_magnitudes[obs.ignore_list == 0]
        height_data = obs.model_ht[obs.ignore_list == 0] / 1000
        mag_filter = abs_mag_data < 8
        height_data = height_data[mag_filter]
        abs_mag_data = abs_mag_data[mag_filter]

        getGoodDataset(obs.station_id, discard_traj_obs_list, height_data, abs_mag_data, mag_sim, 'mag')

    if met_obs is not None:
        # Plot additional magnitudes for all sites
        for site in met_obs.sites:
            # Extract data
            abs_mag_data = met_obs.abs_mag_data[site]
            height_data = met_obs.height_data[site] / 1000

            getGoodDataset(site, discard_met_obs_list, height_data, abs_mag_data, mag_sim, 'mag')

    ### extract velocity data ###
    vel_sim = scipy.interpolate.interp1d(
        sr.brightest_height_arr / 1000, sr.brightest_vel_arr / 1000, bounds_error=False, fill_value=0
    )

    for obs in traj.observations:
        # Extract data
        vel_data = obs.velocities[obs.ignore_list == 0][1:] / 1000
        height_data = obs.model_ht[obs.ignore_list == 0][1:] / 1000

        getGoodDataset(obs.station_id, discard_traj_obs_list, height_data, vel_data, vel_sim, 'vel')

    ### extract lag ###

    # using the simulated brightest point of the meteor to decide its position (rather than largest)
    temp_arr = np.c_[sr.brightest_height_arr, sr.brightest_length_arr]
    temp_arr = temp_arr[
        (sr.brightest_height_arr <= traj.rbeg_ele)
        # & (sr.brightest_height_arr >= plot_end_ht)
    ]
    brightest_ht_arr, brightest_len_arr = temp_arr.T

    brightest_lag_sim = (
        brightest_len_arr
        - brightest_len_arr[0]
        - traj.orbit.v_init
        * np.arange(0, const.dt * len(brightest_len_arr), const.dt)[: len(brightest_len_arr)]
    )
    brightest_interp = scipy.interpolate.interp1d(
        brightest_ht_arr / 1000, brightest_lag_sim, bounds_error=False, fill_value=0
    )

    for obs in traj.observations:

        # Get observed heights
        height_data = obs.model_ht[obs.ignore_list == 0]
        obs_height_indices = height_data > np.min(brightest_ht_arr)
        obs_hts = height_data[obs_height_indices] / 1000

        getGoodDataset(
            obs.station_id,
            discard_traj_obs_list,
            obs_hts,
            obs.lag[obs.ignore_list == 0][obs_height_indices],
            brightest_interp,
            'lag',
        )

    if met_obs is not None:
        for site in met_obs.sites:
            height_data = met_obs.height_data[site] / 1000
            getGoodDataset(
                site, discard_met_obs_list, height_data, met_obs.lag_data[site], brightest_interp, 'lag'
            )

    ###################################
    # processing data that will be kept. This will compute the moving averages and moving std on
    # the data after merging them into single arrays
    processed_data = {
        'mag': tuple([None] * 5),
        'vel': tuple([None] * 5),
        'lag': tuple([None] * 5),
    }
    for key, lst in kept_data.items():
        if not len(lst):  # if part of the data isn't good, then
            print('Filling with nan')
            continue
        else:
            # if data is good, merge the data so that you can take moving averages with all data
            val = lst[0]
            for val2 in lst[1:]:
                val = mergeDatasets(val, val2, ascending=False)

        windowed_residuals = movingOperation(val[2], n=5, ret_arr=True)
        windowed_val = movingOperation(val[1], n=5, ret_arr=True)
        average_height = movingAverage(val[0], n=5)

        processed_data[key] = (
            average_height,
            np.mean(windowed_val, axis=1),
            np.std(windowed_val, axis=1),
            np.mean(windowed_residuals, axis=1),
            np.std(windowed_residuals, axis=1),
        )
    # if too much data is bad, don't use dataset
    if processed_data['mag'][0] is None or processed_data['vel'][0] is None:
        print('bad fit')
        return None

    ###########################################
    # process data again so that all data corresponds to since height values. This makes it so that
    # values can properly be compared with each other, as they occur at the same height.

    # using a height range that exists for all variables
    selected_var = 'mag'  # variable to choose ht values for

    ht_values = processed_data[selected_var][0]  # select height values that doesn't contain min_ht or max_ht
    processed_data2 = [*processed_data[selected_var]]
    for key, (ht, mean_val, std_val, mean_res, std_res) in processed_data.items():
        if key == selected_var:
            continue
        if ht is None:
            processed_data2.extend(
                [
                    np.full(ht_values.shape, np.nan),
                    np.full(ht_values.shape, np.nan),
                    np.full(ht_values.shape, np.nan),
                    np.full(ht_values.shape, np.nan),
                ]
            )
        else:
            print(np.min(ht), np.max(ht), np.min(ht_values), np.max(ht_values))
            processed_data2.extend(
                [
                    scipy.interpolate.interp1d(
                        -ht, mean_val, assume_sorted=True, bounds_error=False, fill_value=np.nan
                    )(-ht_values),
                    scipy.interpolate.interp1d(
                        -ht, std_val, assume_sorted=True, bounds_error=False, fill_value=np.nan
                    )(-ht_values),
                    scipy.interpolate.interp1d(
                        -ht, mean_res, assume_sorted=True, bounds_error=False, fill_value=np.nan
                    )(-ht_values),
                    scipy.interpolate.interp1d(
                        -ht, std_res, assume_sorted=True, bounds_error=False, fill_value=np.nan
                    )(-ht_values),
                ]
            )

    return np.array(processed_data2)


def main(cml_args):
    path = cml_args.dir_file
    with open(path) as f:
        path_list = filter(lambda x: not x.startswith('#'), f.read().splitlines())

    all_kept_data = []
    for _path in path_list:
        print(_path)
        data = compileFolderData(_path)
        if data is not None:
            all_kept_data.append(data)
        print('-------------')

    if not all_kept_data:
        print('No data to work with')
        return
    print([h.shape for h in all_kept_data])
    all_kept_data = np.hstack(all_kept_data).T

    np.set_printoptions(threshold=np.inf)

    min_mag, max_mag = np.nanmin(all_kept_data[:, 1]), np.nanmax(all_kept_data[:, 1])
    min_vel, max_vel = np.nanmin(all_kept_data[:, 1 + 4]), np.nanmax(all_kept_data[:, 1 + 4])
    vbins = 20
    magbins = 20
    mag_to_idx = lambda mag: int((mag - min_mag) / (max_mag - min_mag) * (magbins - 1))
    vel_to_idx = lambda vel: int((vel - min_vel) / (max_vel - min_vel) * (vbins - 1))
    idx_to_mag = lambda i: i / (magbins - 1) * (max_mag - min_mag) + min_mag
    idx_to_vel = lambda i: i / (vbins - 1) * (max_vel - min_vel) + min_vel
    bin_array = np.full((magbins, vbins, all_kept_data.shape[0]), 0, dtype=bool)
    for i, data in enumerate(all_kept_data):
        if np.isfinite(data[1]) and np.isfinite(data[1 + 4]):
            bin_array[mag_to_idx(data[1]), vel_to_idx(data[1 + 4]), i] = 1

    most_common_mag_idx = np.argmax(np.sum(bin_array, axis=(1, 2)))
    most_common_vel_idx = np.argmax(np.sum(bin_array, axis=(0, 2)))

    plt.figure()
    for mag_idx in range(magbins):
        for i, cond in enumerate(bin_array[mag_idx, most_common_vel_idx, :]):
            if cond:
                plt.scatter(all_kept_data[i][1], all_kept_data[i][1 + 3])
    plt.title(
        f"Velocity range: [{idx_to_vel(most_common_vel_idx):.2f}, {idx_to_vel(most_common_vel_idx+1):.2f}] km/s"
    )
    plt.ylabel('Lag residual standard deviation')
    plt.xlabel('Magnitude')
    plt.show()

    plt.figure()
    for vel_idx in range(vbins):
        for i, cond in enumerate(bin_array[most_common_mag_idx, vel_idx, :]):
            if cond:
                plt.scatter(all_kept_data[i][1 + 4], all_kept_data[i][1 + 3])
    plt.title(
        f"Magnitude range: [{idx_to_mag(most_common_mag_idx):.3f}, {idx_to_mag(most_common_mag_idx+1):.3f}]"
    )
    plt.ylabel('Lag residual standard deviation')
    plt.xlabel('Velocity (km/s)')
    plt.show()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Displays the error characterization")
    arg_parser.add_argument('dir_file', metavar='dir', type=str, help="Path to the directory.")
    args = arg_parser.parse_args()
    main(args)
