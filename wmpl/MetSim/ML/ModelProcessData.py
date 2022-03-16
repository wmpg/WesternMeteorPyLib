import matplotlib.pyplot as plt
import numpy as np
from wmpl.MetSim.GUI import SimulationResults
from wmpl.MetSim.MetSimErosion import runSimulation
from wmpl.MetSim.ML.ErrorCharacterization import loadMeteorDataFromDir
from wmpl.MetSim.ML.FitErosion import loadModel
from wmpl.MetSim.ML.GenerateSimulations import (
    SIM_CLASSES_DICT,
    SIM_CLASSES_NAMES,
    PhysicalParameters,
    normalizeSimulations,
)
from wmpl.Utils.Math import mergeClosePoints2, mergeDatasets


def collectAllData(traj, met_obs):
    """
    Takes trajectory and met observations and extracts all relevant information from it
    
    Arguments:
        traj: [ObservedPoints]
        met_obs: [MetObservations]
    
    Returns:
        kept_data: [dict] Stores var:d
            var: [str] 'mag' or 'vel' or 'length'
            d: [dict] Stores station:data
                station: [str] Station id
                data: [list] Contains three elements, time, height, var_data
                    time: [array] 
                    height: [array]
                    var_data: [array] Corresponds to the quantity specified with var (magnitude data for 'mag')

    """
    kept_data = {'mag': {}, 'vel': {}, 'len': {}}

    #### Extract mag data ###
    for obs in traj.observations:
        if obs.absolute_magnitudes is None:
            continue

        time_data = obs.time_data[obs.ignore_list == 0]
        abs_mag_data = obs.absolute_magnitudes[obs.ignore_list == 0]
        height_data = obs.model_ht[obs.ignore_list == 0] / 1000
        mag_filter = abs_mag_data < 8
        time_data = time_data[mag_filter]
        height_data = height_data[mag_filter]
        abs_mag_data = abs_mag_data[mag_filter]

        kept_data['mag'][obs.station_id] = [time_data, height_data, abs_mag_data]

    if met_obs is not None:
        # Plot additional magnitudes for all sites
        for site in met_obs.sites:
            # Extract data
            time_data = met_obs.time_data[site]
            abs_mag_data = met_obs.abs_mag_data[site]
            height_data = met_obs.height_data[site] / 1000

            kept_data['mag'][site] = [time_data, height_data, abs_mag_data]

    ### extract velocity data ###

    for obs in traj.observations:
        # Extract data
        time_data = obs.time_data[obs.ignore_list == 0][1:]
        vel_data = obs.velocities[obs.ignore_list == 0][1:] / 1000
        height_data = obs.model_ht[obs.ignore_list == 0][1:] / 1000

        kept_data['vel'][obs.station_id] = [time_data, height_data, vel_data]

    ### extract lag ###

    # using the simulated brightest point of the meteor to decide its position (rather than largest)

    for obs in traj.observations:

        # Get observed heights
        time_data = obs.time_data[obs.ignore_list == 0]
        height_data = obs.model_ht[obs.ignore_list == 0] / 1000
        length_data = obs.length[obs.ignore_list == 0]

        kept_data['len'][obs.station_id] = [time_data, height_data, length_data]

    return kept_data


def normalizeData(data, fps, param_obj):
    """
    Takes data outputted from collectAllData and combines, normalizes, truncates/pads and structures 
    it so that it can be directly inputted into a ML model.
    
    """
    new_data = {'mag': [], 'len': []}
    xy = (np.array([]), np.array([]), np.array([]), np.array([]))
    for i, key in enumerate(['mag', 'len']):
        for station in data[key]:
            xy = mergeDatasets(xy, np.insert(np.array(data[key][station]), 3 - i, np.nan, axis=0),)

    data = mergeClosePoints2(xy, 1 / fps)
    print(data.shape)
    ## remove data points with nan HERE ##
    norm_data = normalizeSimulations(PhysicalParameters(), param_obj, *data.T)
    # the data length is too small (~35 vs 256)
    raise Exception("stop")
    return np.array(norm_data).T


def main(args):
    folder = args.data_folder
    model_folder = args.model_folder
    station_list = args.station
    model_file = f"{args.modelname}.json"
    weights_file = f"{args.modelname}.h5"

    traj, met_obs, const = loadMeteorDataFromDir(folder)
    if traj is None:
        print("Data doesn't have a fit json file")
        return

    data = collectAllData(traj, met_obs)

    fps = len(traj.observations[0].time_data) / (
        traj.observations[0].time_data[-1] - traj.observations[0].time_data[0]
    )
    print(fps)
    # filter data to only include specified stations
    if station_list:
        data = {key: {k: station for k, station in d if station in station_list} for key, d in data.items()}

    param_obj = SIM_CLASSES_DICT[args.classname]()
    input_data = normalizeData(data, fps, param_obj)

    model = loadModel(model_folder, model_file, weights_file)
    output = model.predict(input_data)[0]
    param_values = param_obj.getDenormalizedInputs(output[0])

    # update const with denormalized param values
    for param_name, val in zip(param_obj.param_list, param_values):
        setattr(const, param_name, val)

    simulation_results = SimulationResults(const, *runSimulation(const))

    # visualizing neural network results
    fig, ax = plt.subplots(2, sharey=True)

    ax[0].plot(
        simulation_results.abs_magnitude,
        simulation_results.brightest_height_arr / 1000,
        label='ML predicted output',
    )
    ax[1].plot(
        simulation_results.brightest_length_arr / 1000,
        simulation_results.brightest_height_arr / 1000,
        label='ML predicted output',
    )
    for i, key in enumerate(['mag', 'len']):
        for station in data[key]:
            ax[i].scatter(data[key][station][2], data[key][station][1], label=f'station: {station}')

    plt.show()


if __name__ == '__main__':
    import argparse

    ### COMMAND LINE ARGUMENTS
    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(
        description="Fit the ML model using the files listed in the given text file (file names only, one file per row)."
    )

    arg_parser.add_argument(
        'data_folder', metavar='DATAFOLDER', type=str, help="Data folder containing all data to be loaded",
    )
    arg_parser.add_argument(
        'model_folder', metavar='MODELFOLDER', type=str, help="Folder of model",
    )
    arg_parser.add_argument(
        '-mn',
        '--modelname',
        metavar="MODEL_NAME",
        type=str,
        help='Name of model (without extension) to be saved.',
        default='model',
    )
    arg_parser.add_argument(
        '-cn',
        '--classname',
        metavar='CLASS_NAME',
        type=str,
        help=f"Use simulation parameters from the given class. Options: {', '.join(SIM_CLASSES_NAMES):s}",
        default='ErosionSimParametersCAMO',
    )
    arg_parser.add_argument('--station', type=str, help='Which stations to use', nargs='+')
    main(arg_parser.parse_args())
