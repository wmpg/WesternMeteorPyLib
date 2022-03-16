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
from wmpl.Utils.Math import meanAngle, mergeDatasets, movingAverage, movingOperation, stdOfStd
from wmpl.Utils.Pickling import loadPickle, savePickle


def distModel(t, x0, v, a, b):
    return x0 + v * t - np.abs(a) * np.exp(np.abs(b) * t)


def velModel(t, v, a, b):
    return v - np.abs(a * b) * np.exp(np.abs(b) * t)


def decelModel(t, a, b):
    return -np.abs(a * b ** 2) * np.exp(np.abs(b) * t)


def fitModelToHeight(model, t, h, y, bounds=None):
    """ Converts a model that is a function of time into a function of height returning function that takes a
    height """
    try:
        popt, _ = scipy.optimize.curve_fit(model, t, y, bounds=bounds)
        t_func = scipy.interpolate.interp1d(h, t, bounds_error=False)
        return lambda _h: model(t_func(_h), *popt)
    except RuntimeError:
        return None


def loadMeteorDataFromDir(folder):
    """
    Return:
        [tuple] traj, met_obs, const
            - traj: [ObservedPoints]
            - met_obs: [MetObservations or None]
            - const: [Constants]
    """
    traj_pickle_file, met_file, _, load_file = collectPaths(folder, '', True, '.', get_latest=True)

    if load_file is None:
        return None, None, None

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


def displayResiduals(traj, met_obs, const):
    ##################################
    # Magnitude

    fig, ax = plt.subplots(2, sharey=True)
    # simulation magnitude

    for frag_entry in const.fragmentation_entries:
        if len(frag_entry.main_height_data):
            ax[0].plot(frag_entry.main_abs_mag, frag_entry.main_height_data / 1000, label='frag main')
        # Plot magnitude of the grains
        if len(frag_entry.grains_height_data):
            ax[0].plot(frag_entry.grains_abs_mag, frag_entry.grains_height_data / 1000, label='frag grains')

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
        nan_filter = np.isfinite(abs_mag_data)
        # ax[1].scatter(abs_mag_data - mag_sim(height_data), height_data)
        fit = np.poly1d(np.polyfit(height_data[nan_filter], abs_mag_data[nan_filter], 8))
        ax[0].plot(fit(height_data), height_data)

        ax[1].scatter(fit(height_data) - abs_mag_data, height_data)

    if met_obs is not None:
        # Plot additional magnitudes for all sites
        for site in met_obs.sites:

            # Extract data
            abs_mag_data = met_obs.abs_mag_data[site]
            height_data = met_obs.height_data[site] / 1000
            ax[0].plot(abs_mag_data, height_data, label='mag')
            # ax[1].scatter(abs_mag_data - mag_sim(height_data), height_data)
            nan_filter = np.isfinite(abs_mag_data)
            fit = np.poly1d(np.polyfit(height_data[nan_filter], abs_mag_data[nan_filter], 8))
            ax[0].plot(fit(height_data), height_data)

            ax[1].scatter(fit(height_data) - abs_mag_data, height_data)

    ax[0].legend()
    plt.show()

    #####################
    #####################
    # Velocity
    fig, ax = plt.subplots(2, sharey=True)
    for obs in traj.observations:
        # Extract data
        time_data = obs.time_data[obs.ignore_list == 0][1:]
        vel_data = obs.velocities[obs.ignore_list == 0][1:] / 1000
        height_data = obs.model_ht[obs.ignore_list == 0][1:] / 1000
        ax[0].plot(vel_data, height_data)
        fit = fitModelToHeight(
            velModel, time_data, height_data, vel_data, bounds=([-np.inf, 0, 0], [np.inf] * 3)
        )
        if fit:
            ax[0].plot(fit(height_data), height_data)
            ax[1].scatter(fit(height_data) - vel_data, height_data)

    plt.show()
    ####################
    ####################
    # Lag
    fig, ax = plt.subplots(2, sharey=True)

    for obs in traj.observations:
        # Get observed heights
        lag_data = obs.lag[obs.ignore_list == 0]
        height_data = obs.model_ht[obs.ignore_list == 0] / 1000
        time_data = obs.time_data[obs.ignore_list == 0]
        ax[0].plot(lag_data, height_data)

        nan_filter = np.isfinite(lag_data)
        fit = np.poly1d(np.polyfit(height_data[nan_filter], lag_data[nan_filter], 8))
        if fit:
            ax[0].plot(fit(height_data), height_data)
            ax[1].scatter(fit(height_data) - lag_data, height_data)

    if met_obs is not None:
        for site in met_obs.sites:
            height_data = met_obs.height_data[site] / 1000

            # ax[0].plot(met_obs.lag_data[site], height_data, label='met_obs')

    ax[0].legend()
    plt.show()


def compileFolderData(folder):
    """
    Collects all data from folder, decides whether it's good or not, then merges all data from different
    cameras then takes moving averages over them
    
    """
    ########################################
    ### collect all data for the folder ####
    traj, met_obs, const = loadMeteorDataFromDir(folder)
    if not const:
        return None

    # displayResiduals(traj, met_obs, const)

    kept_data = {'mag': {}, 'vel': {}, 'lag': {}}

    #### Extract mag data ###
    for obs in traj.observations:
        if obs.absolute_magnitudes is None:
            continue

        abs_mag_data = obs.absolute_magnitudes[obs.ignore_list == 0]
        height_data = obs.model_ht[obs.ignore_list == 0] / 1000
        mag_filter = abs_mag_data < 8
        height_data = height_data[mag_filter]
        abs_mag_data = abs_mag_data[mag_filter]

        nan_filter = np.isfinite(abs_mag_data)
        fit = np.poly1d(np.polyfit(height_data[nan_filter], abs_mag_data[nan_filter], 8))
        kept_data['mag'][obs.station_id] = [abs_mag_data - fit(height_data), height_data, fit]

    if met_obs is not None:
        # Plot additional magnitudes for all sites
        for site in met_obs.sites:
            # Extract data
            abs_mag_data = met_obs.abs_mag_data[site]
            height_data = met_obs.height_data[site] / 1000

            nan_filter = np.isfinite(abs_mag_data)
            fit = np.poly1d(np.polyfit(height_data[nan_filter], abs_mag_data[nan_filter], 8))
            kept_data['mag'][site] = [abs_mag_data - fit(height_data), height_data, fit]

    ### extract velocity data ###

    for obs in traj.observations:
        # Extract data
        vel_data = obs.velocities[obs.ignore_list == 0][1:] / 1000
        height_data = obs.model_ht[obs.ignore_list == 0][1:] / 1000
        time_data = obs.time_data[obs.ignore_list == 0][1:]
        fit = fitModelToHeight(
            velModel, time_data, height_data, vel_data, bounds=([-np.inf, 0, 0], [np.inf] * 3)
        )
        if fit:
            kept_data['vel'][obs.station_id] = [vel_data - fit(height_data), height_data, fit]

    ### extract lag ###

    # using the simulated brightest point of the meteor to decide its position (rather than largest)

    for obs in traj.observations:

        # Get observed heights
        height_data = obs.model_ht[obs.ignore_list == 0] / 1000
        lag_data = obs.lag[obs.ignore_list == 0]
        time_data = obs.time_data[obs.ignore_list == 0]

        nan_filter = np.isfinite(lag_data)
        fit = np.poly1d(np.polyfit(height_data[nan_filter], lag_data[nan_filter], 4))
        kept_data['lag'][obs.station_id] = [lag_data - fit(height_data), height_data, fit]
        # fit = fitModelToHeight(
        #     distModel, time_data, height_data, lag_data, bounds=([-np.inf, -np.inf, 0, 0], [np.inf] * 4)
        # )
        # if fit:
        #     kept_data['lag'][obs.station_id] = [lag_data - fit(height_data), height_data, fit]

    # if met_obs is not None:
    #     for site in met_obs.sites:
    #         height_data = met_obs.height_data[site] / 1000

    ###################################
    # processing data that will be kept.

    # {'mag':[(ht, res, mag_fit, vel_fit, lag_fit), ...]}
    processed_data = {'mag': [], 'vel': [], 'lag': []}
    for key, dic in kept_data.items():
        # different stations should be kept in different entries
        for station in kept_data[key]:
            if dic.get(station):
                res, ht, fit = dic[station]
                new_entry = [ht, res]

                for dic2 in kept_data.values():
                    if dic2.get(station):
                        _, _, fit = dic2[station]
                        new_entry.append(fit(ht))  # ht may be extrapolating
                    else:
                        new_entry.append(ht * np.nan)
                processed_data[key].append(new_entry)

    # ignore dataset if is no information in one of the fields
    for val in processed_data.values():
        if not val:
            return None

    processed_data = {key: np.concatenate(val, axis=1) for key, val in processed_data.items()}
    return processed_data


def plotErrorCharacterization(all_kept_data, plot_variable, title, magbins=20, vbins=20):
    ### plotting magnitude residuals vs velocity and magnitude ###

    min_mag, max_mag = (
        np.nanmin(all_kept_data[plot_variable][:, 2]),
        np.nanmax(all_kept_data[plot_variable][:, 2]),
    )
    min_vel, max_vel = (
        np.nanmin(all_kept_data[plot_variable][:, 3]),
        np.nanmax(all_kept_data[plot_variable][:, 3]),
    )

    mag_to_idx = lambda mag: min(int((mag - min_mag) / (max_mag - min_mag) * magbins), magbins - 1)
    vel_to_idx = lambda vel: min(int((vel - min_vel) / (max_vel - min_vel) * vbins), vbins - 1)
    idx_to_mag = lambda i: i * (max_mag - min_mag) / magbins + min_mag
    idx_to_vel = lambda i: i * (max_vel - min_vel) / vbins + min_vel

    res_arr = np.full((magbins, vbins, all_kept_data[plot_variable].shape[0]), np.nan, dtype=np.float64)
    mag_arr = np.copy(res_arr)
    vel_arr = np.copy(res_arr)

    for i, data in enumerate(all_kept_data[plot_variable]):
        if np.isfinite(data[2]) and np.isfinite(data[3]):
            res_arr[mag_to_idx(data[2]), vel_to_idx(data[3]), i] = data[1]
            mag_arr[mag_to_idx(data[2]), vel_to_idx(data[3]), i] = data[2]
            vel_arr[mag_to_idx(data[2]), vel_to_idx(data[3]), i] = data[3]

    nan_arr = np.sum(np.isfinite(res_arr), axis=2)
    std_arr = np.nanstd(res_arr, axis=2, ddof=1)
    std_std_arr = stdOfStd(std_arr, nan_arr)

    v_values = np.linspace(min_vel, max_vel, vbins + 1)
    mag_values = np.linspace(min_mag, max_mag, magbins + 1)
    fit_2d = lambda x, a, b, c: a * x[0] + b * x[1] + c
    v_mag_values = np.stack(
        np.meshgrid((v_values[1:] + v_values[:-1]) / 2, (mag_values[1:] + mag_values[:-1]) / 2)
    )  # dimensions (2, len(mag_values)-1, len(v_values)-1)

    # print(np.concatenate(v_mag_values)[np.isfinite(std_arr.flatten())].shape)
    _filter = np.isfinite(std_arr.flatten())
    data = np.concatenate(v_mag_values.T)[_filter].T
    print(data, data.shape)
    popt, pcov = scipy.optimize.curve_fit(
        fit_2d,
        data,
        std_arr.flatten()[_filter],
        sigma=std_std_arr.flatten()[_filter]
        # bounds=([0, 0, -np.inf, 0, -np.inf, -np.inf], [np.inf] * 6),
    )
    plt.subplot(1, 2, 1)
    plt.scatter(data[0], std_arr.flatten()[_filter])
    plt.scatter(data[0], fit_2d(data, *popt), c='r')
    plt.subplot(1, 2, 2)
    plt.scatter(data[1], std_arr.flatten()[_filter])
    plt.scatter(data[1], fit_2d(data, *popt), c='r')

    # plt.plot(
    #     data, fit_2d(data, *popt),
    # )
    plt.show()
    # print(fit_2d(30, *popt))
    print(popt)
    print('fit errors', np.sqrt(np.diag(pcov)))

    plt.pcolormesh(
        v_values, mag_values, np.abs((std_arr - fit_2d(v_mag_values, *popt)) / std_std_arr), cmap='coolwarm'
    )  # - fit_2d(v_mag_values, *popt),
    for i in range(magbins):
        for j in range(vbins):
            if nan_arr[i, j] > 1:
                plt.text(
                    idx_to_vel(j + 0.5),
                    idx_to_mag(i + 0.5),
                    f"{std_std_arr[i, j]:.3f}",
                    color="k",
                    ha="center",
                    va="center",
                    transform=plt.gca().transData,
                )

    plt.colorbar()
    plt.ylabel('Magnitude')
    plt.xlabel('Velocity (km/s)')
    plt.title(title)

    plt.xlim(
        [
            idx_to_vel(np.min(np.nonzero(np.nansum(std_arr, axis=0)))),
            idx_to_vel(np.max(np.nonzero(np.nansum(std_arr, axis=0))) + 1),
        ]
    )
    plt.ylim(
        [
            idx_to_mag(np.min(np.nonzero(np.nansum(std_arr, axis=1)))),
            idx_to_mag(np.max(np.nonzero(np.nansum(std_arr, axis=1))) + 1),
        ]
    )

    plt.show()

    plt.title(np.sum(nan_arr))
    data = res_arr.flatten()[np.isfinite(res_arr.flatten())]
    # hist, bin_edges = np.histogram(data, bins=100, normed=True)
    ret = scipy.stats.t.fit(res_arr.flatten()[np.isfinite(res_arr.flatten())])
    plt.hist(data, bins=100, density=True)
    x = np.linspace(np.min(data), np.max(data), 300)
    plt.plot(x, scipy.stats.t.pdf(x, ret[0], loc=ret[1], scale=ret[2]))
    print(ret)
    plt.show()

    # for i in range(magbins):
    #     for j in range(vbins):
    #         if nan_arr[i, j] > 4:
    #             plt.title(np.sum(nan_arr[i, j]))
    #             data = res_arr[i, j][np.isfinite(res_arr[i, j])]
    #             # hist, bin_edges = np.histogram(data, bins=100, normed=True)
    #             ret = scipy.stats.norm.fit(data)
    #             plt.hist(data, bins=30, density=True)
    #             x = np.linspace(np.min(data), np.max(data), 50)
    #             plt.plot(x, scipy.stats.norm.pdf(x, loc=ret[0], scale=ret[1]))
    #             print(ret)
    #             plt.show()


def main(cml_args):
    path = cml_args.dir_file
    with open(path) as f:
        path_list = filter(lambda x: not x.startswith('#'), f.read().splitlines())

    all_kept_data = {'mag': [], 'vel': [], 'lag': []}
    for _path in path_list:
        print(_path)
        data = compileFolderData(_path)
        if data is not None:
            for key in all_kept_data:
                all_kept_data[key].append(data[key])
        print('-------------')

    for key in all_kept_data:
        all_kept_data[key] = np.concatenate(all_kept_data[key], axis=1).T

    if not all_kept_data:
        print('No data to work with')
        return

    # np.set_printoptions(threshold=np.inf)

    vbins = 20
    magbins = 20
    plotErrorCharacterization(
        all_kept_data, 'mag', 'Magnitude residual standard deviation', vbins=vbins, magbins=magbins
    )
    plotErrorCharacterization(
        all_kept_data, 'lag', 'Lag residual standard deviation (m)', vbins=vbins, magbins=120
    )


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Displays the error characterization")
    arg_parser.add_argument('dir_file', metavar='dir', type=str, help="Path to the directory.")
    args = arg_parser.parse_args()
    main(args)
