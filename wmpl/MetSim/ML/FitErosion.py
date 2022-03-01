""" Fit the erosion model using machine learning. """

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import os
import random
from re import A

import keras
import matplotlib.pyplot as plt
import numpy as np
from wmpl.MetSim.GUI import SimulationResults
from wmpl.MetSim.MetSimErosion import runSimulation
from wmpl.MetSim.ML.GenerateSimulations import (
    DATA_LENGTH,
    SIM_CLASSES,
    SIM_CLASSES_DICT,
    SIM_CLASSES_NAMES,
    ErosionSimContainer,
    ErosionSimParameters,
    ErosionSimParametersCAMO,
    ErosionSimParametersCAMOWide,
    MetParam,
    PhysicalParameters,
    extractSimData,
)
from wmpl.Utils.Pickling import loadPickle
from wmpl.Utils.PyDomainParallelizer import domainParallelizer


def dataFunction(file_path, param_class_name):

    # Load the pickle file
    sim = loadPickle(*os.path.split(file_path))

    # Extract model inputs and outputs
    return extractSimData(sim, param_class_name=param_class_name)


class DataGenerator(object):
    def __init__(
        self,
        data_list,
        batch_size,
        steps_per_epoch,
        param_class_name=None,
        validation=False,
        validation_portion=0.2,
    ):
        """ Generate meteor data for the ML fit function. 
    
        Arguments:
            data_list: [list] A list of files that will be used as inputs.
            batch_size: [int] Number of inputs in every step.
            steps_per_epoch: [int] Number of steps in every epoch (iteration) with batch_size inputs each.

        Keyword arguments:
            param_class_name: [str] Override the simulation parameters object with an instance of the given
                class. An exact name of the class needs to be given.
            validation: [bool] Generate validation data. False by default.
            validation_portion: [float] Portion of input files to be used for validation.
        """

        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.data_list = data_list

        intial_len = len(self.data_list)

        self.param_class_name = param_class_name

        self.validation = validation

        # Compute the number of files in each epoch
        data_per_epoch = self.batch_size * self.steps_per_epoch

        # Cut the processing list to the steps multiplier
        self.data_list = self.data_list[: -int(len(self.data_list) % data_per_epoch)]

        # Compute the number of total epochs
        total_epochs = int(len(self.data_list) // data_per_epoch)
        if total_epochs == 0:
            raise Exception(
                'Total epochs is zero. Batch size or steps per epoch are too high. '
                f'At least {data_per_epoch} samples expected with {intial_len} '
                'given.'
            )

        # Compute the number of epochs for the fit
        self.fit_epochs = int((1.0 - validation_portion) * total_epochs)

        # Number of validation epochs
        self.validation_epochs = total_epochs - self.fit_epochs

        # Make sure that there is a minimum of one validation epoch
        if self.validation_epochs < 1:
            self.validation_epochs = 1
            self.fit_epochs -= 1

        # Current data index
        self.data_index_start = 0

        # Current validation index
        self.validation_index_start = self.fit_epochs * data_per_epoch - 1

    def __iter__(self):
        data_length = DATA_LENGTH  # default value
        # Select valirable depending on if the validation data is used or not
        if self.validation:
            curr_index = self.validation_index_start
            epochs = self.validation_epochs

        else:
            curr_index = self.data_index_start
            epochs = self.fit_epochs

        # Generate data for every epoch
        for step in range(epochs * self.steps_per_epoch):
            param_list = []
            result_list = []

            # Get a portion of files to load
            beg_index = curr_index
            file_list = self.data_list[beg_index : (beg_index + self.batch_size)]

            # Load pickle files and postprocess in parallel
            domain = []
            for file_path in file_list:
                domain.append([file_path, self.param_class_name])

            # Postprocess the data in parallel
            res_list = domainParallelizer(domain, dataFunction)

            # Postprocess results
            filtered_res = []
            for res in res_list:

                curr_index += 1

                # Skip simulation which did not satisfy filters
                if res is None:
                    # print("Skipped:", file_path)
                    continue

                filtered_res.append(res)

            res_list = filtered_res

            # Load more results using one core until the proper length is achieved
            while len(res_list) < self.batch_size:

                file_path = self.data_list[curr_index]

                # Extract model inputs and outputs from the pickle file
                res = dataFunction(file_path, self.param_class_name)

                curr_index += 1

                # Skip simulation which did not satisfy filters
                if res is None:
                    # print("Skipped:", file_path)
                    continue

                res_list.append(res)

            # Split results to input/output list
            for res in res_list:

                # Extract results
                param_dict, input_data_normed, simulated_data_normed = res
                data_length = param_dict['camera'].data_length

                # Add data to model input and output lists
                param_list.append(input_data_normed)
                result_list.append(simulated_data_normed)

            param_list = np.array(param_list)
            result_list = np.array(result_list)

            height_data_normed_list, mag_data_normed_list, length_data_normed_list = np.split(
                result_list, 3, axis=1
            )

            yield [
                height_data_normed_list.reshape(-1, data_length, 1),
                length_data_normed_list.reshape(-1, data_length, 1),
                mag_data_normed_list.reshape(-1, data_length, 1),
            ], param_list


class ReportFitGoodness(keras.callbacks.Callback):
    """ Report the fit goodness at the every epoch end.
  """

    def __init__(self, validation_gen):
        self.validation_gen = validation_gen

    def on_epoch_end(self, epoch, logs=None):

        # Evaluate model accuracy using validation data
        percent_errors = evaluateFit(self.model, self.validation_gen, ret_perc=True)

        print()
        print("Epoch {:d} errors".format(epoch + 1))
        param_name_list = ["M0", "V0", "ZC", "DENS", "ABL", "ERHT", "ERCO", "ER_S", "ERMm", "ERMM"]
        print(" ".join(["{:>6s}".format(param_name) for param_name in param_name_list]))
        print(str(len(percent_errors) * "{:5.2f}% ").format(*percent_errors))
        print()
        print()


def loadModel(file_path, model_file='model.json', weights_file='model.h5'):
    with open(os.path.join(file_path, model_file), 'r') as json_file:

        # load json and create model
        loaded_model_json = json_file.read()
        loaded_model = keras.models.model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights(os.path.join(file_path, weights_file))
        print("Loaded model from disk")

        return loaded_model


def evaluateFit(model, validation_gen, ret_perc=False):
    validation_gen = copy.deepcopy(validation_gen)

    # Generate test data
    test_data = next(iter(validation_gen))
    test_outputs, test_inputs = test_data

    # Predict data
    pred_norm_params = model.predict(test_outputs)

    # Compute mean absolute percentage error for every model parameter
    percent_errors = 100 * np.mean(np.abs(pred_norm_params - test_inputs), axis=0)

    if ret_perc:
        return percent_errors

    else:

        print("Mean absolute percentage error per parameter:")
        param_name_list = ["M0", "V0", "ZC", "DENS", "ABL", "ERHT", "ERCO", "ER_S", "ERMm", "ERMM"]
        print(" ".join(["{:>6s}".format(param_name) for param_name in param_name_list]))
        print(str(len(percent_errors) * "{:5.2f}% ").format(*percent_errors))


def evaluateFit2(model, file_path, param_class_name=None):
    """ Evaluates model by visually comparing expected simulation values to the simulation values 
    given from the prediction """

    sim = loadPickle(*os.path.split(file_path))

    ret = extractSimData(sim, param_class_name=param_class_name)
    if ret is None:
        print('Dataset is invalid')
        return

    input_param_dict, norm_input_param_vals, norm_sim_data = ret

    data_length = input_param_dict['camera'].data_length
    normalized_output_param_vals = model.predict(tuple(i.reshape(-1, data_length, 1) for i in norm_sim_data))
    phys_params = copy.deepcopy(input_param_dict['physical'])
    phys_params.setParamValues(phys_params.getDenormalizedInputs(normalized_output_param_vals))
    const = phys_params.getConst()
    simulation_results = SimulationResults(const, *runSimulation(const))

    print('predicted', phys_params.getInputs())
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

    ax[0].plot(
        sim.simulation_results.abs_magnitude,
        sim.simulation_results.brightest_height_arr / 1000,
        label='Correct simulated output',
        c='k',
    )
    ax[1].plot(
        sim.simulation_results.brightest_length_arr / 1000,
        sim.simulation_results.brightest_height_arr / 1000,
        label='Correct simulated output',
        c='k',
    )

    ax[0].set_ylabel('Height (km)')
    ax[0].set_xlabel("Magnitude")
    ax[0].legend()
    ax[1].set_ylabel('Height (km)')
    ax[1].set_xlabel("Length (km)")
    ax[1].legend()

    print('correct', sim.params.getInputs())

    plt.show()


def fitCNNMultiHeaded(data_gen, validation_gen, output_dir, model_file, weights_file):
    # https://machinelearningmastery.com/how-to-develop-convolutional-neural-network-models-for-time-series-forecasting/
    # Height input model
    checkpoint_filepath = os.path.join(output_dir, 'modelcheckpoint.h5')
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath, mode='min', verbose=1, save_weights_only=True
    )

    visible1 = keras.engine.input_layer.Input(shape=(DATA_LENGTH, 1))
    cnn1 = keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu')(visible1)
    cnn1 = keras.layers.MaxPooling1D(pool_size=2)(cnn1)
    cnn1 = keras.layers.Flatten()(cnn1)
    cnn1 = keras.layers.Dense(128, kernel_initializer='normal', activation='relu')(cnn1)
    cnn1 = keras.layers.Dense(128, kernel_initializer='normal', activation='relu')(cnn1)

    # Length input model
    visible2 = keras.engine.input_layer.Input(shape=(DATA_LENGTH, 1))
    cnn2 = keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu')(visible2)
    cnn2 = keras.layers.MaxPooling1D(pool_size=2)(cnn2)
    cnn2 = keras.layers.Flatten()(cnn2)
    cnn2 = keras.layers.Dense(128, kernel_initializer='normal', activation='relu')(cnn2)
    cnn2 = keras.layers.Dense(128, kernel_initializer='normal', activation='relu')(cnn2)

    # Magnitude input model
    visible3 = keras.engine.input_layer.Input(shape=(DATA_LENGTH, 1))
    cnn3 = keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu')(visible3)
    cnn3 = keras.layers.MaxPooling1D(pool_size=2)(cnn3)
    cnn3 = keras.layers.Flatten()(cnn3)
    cnn3 = keras.layers.Dense(128, kernel_initializer='normal', activation='relu')(cnn3)
    cnn3 = keras.layers.Dense(128, kernel_initializer='normal', activation='relu')(cnn3)

    # merge input models
    merge = keras.layers.Concatenate()([cnn1, cnn2, cnn3])
    dense = keras.layers.Dense(256, kernel_initializer='normal', activation='relu')(merge)
    dense = keras.layers.Dense(256, kernel_initializer='normal', activation='relu')(dense)
    dense = keras.layers.Dense(256, kernel_initializer='normal', activation='relu')(dense)
    dense = keras.layers.Dense(256, kernel_initializer='normal', activation='relu')(dense)
    output = keras.layers.Dense(10, kernel_initializer='normal', activation="linear", batch_size=batch_size)(
        dense
    )

    # Tie inputs together
    model = keras.models.Model(inputs=[visible1, visible2, visible3], outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # fit model
    model.fit(
        x=iter(data_gen),
        steps_per_epoch=data_gen.steps_per_epoch,
        epochs=data_gen.fit_epochs,
        callbacks=[ReportFitGoodness(validation_gen), model_checkpoint_callback],
        workers=0,
        max_queue_size=1,
    )

    # Save the model to disk
    model_json = model.to_json()

    model_file = os.path.join(output_dir, model_file)
    weights_file = os.path.join(output_dir, weights_file)
    with open(model_file, "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(weights_file)
    print("Saved model to disk")

    # Evaluate fit quality
    evaluateFit(model, validation_gen)


def getFileList(folder):
    """ Get list of all files contained in folder, found recursively. """
    # path gives should be files/clean_ML_trainingset
    file_list = []
    for root, _, files in os.walk(folder):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


if __name__ == "__main__":

    import argparse

    ### COMMAND LINE ARGUMENTS
    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(
        description="Fit the ML model using the files listed in the given text file (file names only, one file per row)."
    )

    arg_parser.add_argument(
        'data_folder',
        metavar='DATAFOLDER',
        type=str,
        help="Data folder containing all data to be loaded for training and testing. Path may contain subfolders.",
    )
    arg_parser.add_argument(
        'output_dir', metavar='OUTPUTDIR', type=str, help='Data folder to save model data to'
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
    arg_parser.add_argument(
        '-e',
        '--evaluate',
        action='store_true',
        help='Inputting this parameter will not train the model, but instead evaluate the model by visually '
        'showing what it predicts compared to the simulation.',
    )
    arg_parser.add_argument(
        '--grouping',
        type=int,
        nargs=2,
        help='Allows for specifying the batch size and steps per epoch, respectively.',
    )
    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    ### INPUTS ###
    if cml_args.grouping:
        batch_size, steps_per_epoch = cml_args.grouping
    else:
        batch_size = DATA_LENGTH
        steps_per_epoch = 80

    # Model file names
    model_file = f"{cml_args.modelname}.json"
    weights_file = f"{cml_args.modelname}.h5"

    ### ###

    # Load the list of files from the given input file and randomly drawn limiting magnitude and length
    #   measurement delay used to generate the data
    data_list = []
    data_list = getFileList(cml_args.data_folder)

    # randomize the order of the dataset to remove bias
    random.Random(0).shuffle(data_list)

    print("{:d} inputs used for training/testing...".format(len(data_list)))

    if cml_args.evaluate:
        model = loadModel(cml_args.output_dir, model_file, weights_file)
        evaluateFit2(model, data_list[2])  # , cml_args.classname)
    else:
        # Init the data generator
        data_gen = DataGenerator(
            data_list, batch_size, steps_per_epoch, param_class_name=cml_args.classname, validation=False
        )

        # Init the validation generator
        validation_gen = DataGenerator(
            data_list, batch_size, steps_per_epoch, param_class_name=cml_args.classname, validation=True
        )

        # ## TEST DATA GEN ###
        # for epoch in range(data_gen.fit_epochs):
        #     for step in range(data_gen.steps_per_epoch):
        #         print(epoch, "/", step)
        #         result_list, param_list = next(iter(data_gen))

        #         print(len(param_list))

        # Fit the model
        fitCNNMultiHeaded(data_gen, validation_gen, cml_args.output_dir, model_file, weights_file)

