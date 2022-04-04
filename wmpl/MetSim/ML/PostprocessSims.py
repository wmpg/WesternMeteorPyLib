""" Preprocess the simulations before feeding them into the neural network. """

from __future__ import absolute_import, division, print_function, unicode_literals

import multiprocessing
import os
import random
import time
from typing import Optional

import h5py
import numpy as np
from wmpl.MetSim.ML.GenerateSimulations import (
    DATA_LENGTH,
    SIM_CLASSES_NAMES,
    ErosionSimContainer,
    ErosionSimParametersCAMO,
    MetParam,
    PhysicalParameters,
    dataFunction,
    extractSimData,
    getFileList,
)
from wmpl.Utils.Pickling import loadPickle
from wmpl.Utils.PyDomainParallelizer import domainParallelizer


def saveProcessedData(
    data_path: str,
    output_path: str,
    filename: str,
    param_class_name: Optional[str] = None,
    multiprocess=True,
):
    """ Save a list of pickle files which passes postprocessing criteria to disk.

    Arguments:
        data_path: [str] Path to directory with simulation pickle files.
        param_class_name: [str] Name of the parameter class used for postprocessing.

    """

    # Load one simulation to get simulation parameters
    file_list = getFileList(data_path)
    with h5py.File(f'{os.path.join(output_path, filename)}.h5', 'w') as h5file:
        sim_dataset = h5file.create_dataset(
            'simulation', shape=(len(file_list), DATA_LENGTH, 4), chunks=True, dtype=np.float32
        )
        param_dataset = h5file.create_dataset(
            'parameters', shape=(len(file_list), 10), chunks=True, dtype=np.float32
        )

        t1 = time.perf_counter()
        valid = 0
        if not multiprocess:
            for i, filepath in enumerate(file_list):
                if i % 1000 == 0:
                    print(i / len(file_list) * 100, time.perf_counter() - t1)

                sim = loadPickle(data_path, filepath)
                ret = extractSimData(sim, param_class_name=param_class_name)
                if ret is None:
                    continue

                _, sim_data, param_data = ret
                sim_dataset[valid] = sim_data
                param_dataset[valid] = param_data
                valid += 1
        else:
            tasks = multiprocessing.cpu_count() * 200
            loops = int(len(file_list) / tasks) + 1
            for i in range(loops):
                print(i / loops * 100, time.perf_counter() - t1)
                domain = [[file] for file in file_list[i * tasks : (i + 1) * tasks]]

                output = domainParallelizer(
                    domain, dataFunction, kwarg_dict={'param_class_name': param_class_name}
                )

                # this is the bottleneck, but it can't be improved on because h5py object can't be pickled
                for sim, extractdata in output:
                    if sim is None:
                        continue

                    _, sim_data, param_data = extractdata
                    sim_dataset[valid] = sim_data
                    param_dataset[valid] = param_data
                    valid += 1

        sim_dataset.resize((valid, DATA_LENGTH, 4))
        param_dataset.resize((valid, 10))


def loadProcessedData(h5path: str, batchsize: int, validation=False, validation_fraction=0.2):
    """ Generator for loading h5py datasets without loading everything into memory """
    with h5py.File(h5path, 'r') as h5file:
        i = 0

        index_list = list(range(int(len(h5file['simulation']) / batchsize)))
        if validation:
            index_list = index_list[-int(len(index_list) * validation_fraction) :]
        else:
            index_list = index_list[: -int(len(index_list) * validation_fraction)]

        while True:
            batch_sim = h5file['simulation'][
                index_list[i] * batchsize : (index_list[i] + 1) * batchsize, ..., 1:
            ]
            batch_param = h5file['parameters'][index_list[i] * batchsize : (index_list[i] + 1) * batchsize]

            i += 1
            if len(batch_sim) < batchsize or i == len(index_list):
                # semi shuffle, since h5py can't handle indexing as complex as numpy (it's also slower)
                np.random.shuffle(index_list)
                i = 0
                continue

            yield batch_sim, batch_param


def loadh5pyData(path):
    """ 
    Loads all h5py data from dataset into memory. 
    WARNING: For larger datasets, this can be on the order of gigabytes. Use only with excess of memory
    """
    with h5py.File(path, 'r') as f:
        input_train = f['simulation'][..., 1:]
        label_train = f['parameters'][...]

    return input_train, label_train


def validateSimulation(dir_path, file_name, param_class_name, min_frames_visible):

    # Load the pickle file
    sim = loadPickle(dir_path, file_name)

    # Extract simulation data
    res = extractSimData(
        sim, min_frames_visible=min_frames_visible, check_only=True, param_class_name=param_class_name
    )

    # If the simulation didn't satisfy the filters, skip it
    if res is None:
        return None

    print("Good:", file_name)

    return os.path.join(dir_path, file_name), res


# def postprocessSims(data_path, param_class_name=None, min_frames_visible=10):
#     """ Preprocess simulations generated by the ablation model to prepare them for training.

#     From all simulations, make fake observations by taking only data above the limiting magnitude and
#     add noise to simulated data.

#     Arguments:
#         data_path: [str] Path to directory with simulations.

#     Keyword arguments:
#         param_class_name: [str] Name of the class used for postprocessing parameters. None by default, in
#             which case the original parameters will be used.
#         min_frames_visible: [int] Minimum number of frames the meteor was visible.

#     """

#     # Go through all simulations and create a list for processing
#     processing_list = []
#     for entry in os.walk(data_path):

#         dir_path, _, file_list = entry

#         for file_name in file_list:

#             file_path = os.path.join(dir_path, file_name)

#             # Check if the given file is a pickle file
#             if os.path.isfile(file_path) and file_name.endswith(".pickle"):

#                 processing_list.append([dir_path, file_name, param_class_name, min_frames_visible])

#     # Validate simulation (parallelized)
#     print("Starting postprocessing in parallel...")
#     results_list = domainParallelizer(processing_list, validateSimulation)

#     # Randomize the list
#     random.shuffle(results_list)

#     # Save the list of post-processed pickle files to disk
#     saveProcessedList(data_path, results_list, param_class_name, min_frames_visible)


if __name__ == "__main__":

    import argparse

    ### COMMAND LINE ARGUMENTS
    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(
        description="Check that the simulations in the given directory satisfy the given conditions and create a file with the list of simulation to use for training."
    )

    arg_parser.add_argument('data_path', type=str, help="Path to the directory with simulation pickle files.")

    arg_parser.add_argument('output_path', type=str, help="Path to save processed data to")
    arg_parser.add_argument('-f', '--filename', type=str, help="Name of file for processed data")
    arg_parser.add_argument(
        '-p',
        '--params',
        metavar='PARAM_CLASS',
        type=str,
        help="Override the postprocessing parameters by using parameters from the given class. Options: {:s}".format(
            ", ".join(SIM_CLASSES_NAMES)
        ),
    )

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    saveProcessedData(cml_args.data_path, cml_args.output_path, cml_args.filename, multiprocess=False)
