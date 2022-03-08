""" Fit the erosion model using machine learning. """

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import datetime
import os
import random
from re import A

import keras
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
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
        random_state=None,
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

        self.random_state = random_state
        if self.random_state is None:
            self.random_state = random.Random()

        intial_len = len(self.data_list)

        self.param_class_name = param_class_name

        self.validation = validation

        # Compute the number of files in each epoch
        data_per_epoch = self.batch_size * self.steps_per_epoch

        self.training_list = self.data_list[: int(intial_len * (1 - validation_portion))]
        self.validation_list = self.data_list[int(intial_len * (1 - validation_portion)) :]

        # Compute the number of total epochs
        self.total_epochs = int(len(self.data_list) // data_per_epoch)
        self.validation_epochs = int(len(self.validation_list) // data_per_epoch)
        self.training_epochs = int(len(self.training_list) // data_per_epoch)
        if self.total_epochs == 0:
            raise Exception(
                'Total epochs is zero. Batch size or steps per epoch are too high. '
                f'At least {data_per_epoch} samples expected with {intial_len} '
                'given.'
            )

    @property
    def epochs(self):
        if self.validation:
            return self.validation_epochs
        return self.training_epochs

    def __iter__(self):
        # Select file list depending on whether validation or training is being done
        if self.validation:
            data_list = self.validation_list
        else:
            data_list = self.training_list

        res_list = []
        curr_index = 0
        to_delete = []
        # Generate data for every epoch
        while True:
            # Get a portion of files to load
            file_list = data_list[curr_index : curr_index + batch_size]

            # Load pickle files and postprocess in parallel
            domain = [[file_path, self.param_class_name] for file_path in file_list]

            # Postprocess the data in parallel
            res_list += domainParallelizer(domain, dataFunction)

            filtered_res_list = []
            # discard bad results
            for i, res in enumerate(res_list[-len(file_list) :]):
                if res is None:
                    to_delete.append(curr_index + i)
                else:
                    filtered_res_list.append(res)

            res_list = res_list[: -self.batch_size] + filtered_res_list
            curr_index += self.batch_size

            # if you fully loop data, shuffle it
            if curr_index >= len(data_list):
                data_list = np.delete(data_list, to_delete)
                to_delete = []
                curr_index = 0
                self.random_state.shuffle(data_list)
                if len(data_list) == 0:
                    raise Exception("No valid data")

            # if there aren't enough results to fill the batch, collect another batch and fill the gaps with it
            # where extras will be used in subsequent iterations
            if len(res_list) < self.batch_size:
                continue

            next_res_list = res_list[self.batch_size :]
            res_list = res_list[: self.batch_size]

            # Split results to input/output list
            param_dict_list = []
            param_list = []
            result_list = []
            for res in res_list:
                # Extract results
                param_dict, input_data_normed, simulated_data_normed = res

                # Add data to model input and output lists
                param_dict_list.append(param_dict)
                param_list.append(input_data_normed)
                result_list.append(simulated_data_normed)

            res_list = next_res_list

            param_list = np.array(param_list)
            result_list = np.array(result_list)

            # yield dimenions [(batch_size, data_length, 1), ...]
            if self.validation:
                yield param_dict_list, np.moveaxis(result_list, 1, 2), param_list
            else:
                yield np.moveaxis(result_list, 1, 2), param_list


class ReportFitGoodness(keras.callbacks.Callback):
    """ Report the fit goodness at the every epoch end.
  """

    def __init__(self, validation_gen):
        self.validation_gen = iter(validation_gen)

    def on_epoch_end(self, epoch, logs=None):

        # Evaluate model accuracy using validation data
        print()
        print("Epoch {:d} errors".format(epoch + 1))
        evaluateFit(self.model, self.validation_gen, output=True)


def loadModel(file_path, model_file='model.json', weights_file='model.h5'):
    with open(os.path.join(file_path, model_file), 'r') as json_file:

        # load json and create model
        loaded_model_json = json_file.read()
        loaded_model = keras.models.model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights(os.path.join(file_path, weights_file))
        print("Loaded model from disk")

        return loaded_model


def evaluateFit(model, validation_gen, output=False, display=False):
    param_name_list = ["M0", "V0", "ZC", "DENS", "ABL", "ERHT", "ERCO", "ER_S", "ERMm", "ERMM"]

    # Generate test data
    test_data = next(validation_gen)
    param_dict_list, test_outputs, test_inputs = test_data
    camera_param = param_dict_list[0]['physical']

    # print([i.shape for i in test_outputs])
    # Predict data
    pred_norm_params = model.predict(test_outputs)
    norm_errors = np.abs(pred_norm_params - test_inputs)
    correct_output = np.array(camera_param.getDenormalizedInputs(test_inputs.T)).T
    pred_output = np.array(camera_param.getDenormalizedInputs(pred_norm_params.T)).T
    denorm_errors = np.abs(pred_output - correct_output)
    denorm_perc_errors = denorm_errors / correct_output
    # print(pred_output)
    # print(correct_output)
    if display:
        fig, ax = plt.subplots(2, sharey=True, sharex=True)
        print(test_inputs.shape)
        ax[0].scatter(*test_inputs[:, [0, -1]].T, label='correct')
        ax[0].set_yscale('log')
        ax[0].set_xscale('log')
        ax[0].set_xlabel('M0')
        ax[0].set_ylabel('ERMM')
        ax[0].legend()
        ax[1].scatter(*pred_norm_params[:, [0, -1]].T, label='predicted')
        ax[1].set_yscale('log')
        ax[1].set_xscale('log')
        ax[1].legend()
        ax[1].set_xlabel('M0')
        ax[1].set_ylabel('ERMM')
        plt.show()

        fig, ax = plt.subplots(len(denorm_errors.T), sharex=True, sharey=True)
        for a, values, label in zip(ax, norm_errors.T, param_name_list):
            a.hist(values, bins='auto', label=label)
            a.legend(loc='upper right')
        # a.set_yscale('log')
        a.set_xlim([0, 1])
        plt.show()

    # Compute mean absolute percentage error for every model parameter
    percent_norm_errors = 100 * np.mean(norm_errors, axis=0)
    denorm_errors_av = np.mean(denorm_errors, axis=0)
    denorm_perc_errors_av = 100 * np.mean(denorm_perc_errors, axis=0)

    if output:
        print("Mean absolute percentage error and mean absolute error per parameter:")
        print(" ".join(["{:>9s}".format(param_name) for param_name in param_name_list]))
        print(str(len(percent_norm_errors) * "{:8.2f}% ").format(*percent_norm_errors))
        print(str(len(denorm_perc_errors_av) * "{:8.2f}% ").format(*denorm_perc_errors_av))
        print(str(len(denorm_errors_av) * "{:9.2E} ").format(*denorm_errors_av))

    return percent_norm_errors, denorm_errors_av


def evaluateFit2(model, file_path, validation_gen, param_class_name=None):
    """ Evaluates model by visually comparing expected simulation values to the simulation values 
    given from the prediction """
    evaluateFit(model, iter(validation_gen), display=True, output=True)
    print()

    sim = loadPickle(*os.path.split(file_path))

    ret = extractSimData(sim, param_class_name=param_class_name)
    if ret is None:
        print('Dataset is invalid')
        return

    input_param_dict, norm_input_param_vals, norm_sim_data = ret

    data_length = input_param_dict['camera'].data_length

    normalized_output_param_vals = model.predict(norm_sim_data.T[None])  # dimensions (1, 256, 4)
    print()
    print('correct norm', norm_input_param_vals)
    print('pred norm', list(normalized_output_param_vals[0]))
    print()
    print('perc error', list(np.abs(normalized_output_param_vals[0] - np.array(norm_input_param_vals)) * 100))
    print()
    phys_params = copy.deepcopy(input_param_dict['physical'])
    phys_params.setParamValues(phys_params.getDenormalizedInputs(normalized_output_param_vals[0]))
    const = phys_params.getConst()
    simulation_results = SimulationResults(const, *runSimulation(const))

    starting_height = (
        simulation_results.brightest_height_arr[np.argmax(simulation_results.abs_magnitude < 8)] / 1000
    )
    ending_height = (
        simulation_results.brightest_height_arr[-np.argmax(simulation_results.abs_magnitude[::-1] < 8) - 1]
        / 1000
        - 10
    )
    print('predicted', phys_params.getInputs())

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(norm_sim_data[3], norm_sim_data[1])
    ax[0, 0].set_xlabel('Mag')
    ax[0, 0].set_ylabel('Ht')

    ax[1, 0].scatter(np.diff(norm_sim_data[2]), norm_sim_data[1][:-1], marker='o', s=2)
    ax[1, 0].set_xlabel('Velocity')
    ax[1, 0].set_ylabel('Ht')

    ax[0, 1].plot(sim.simulation_results.abs_magnitude, sim.simulation_results.brightest_height_arr)
    ax[0, 1].set_xlabel('Mag')
    ax[0, 1].set_ylabel('Ht')

    ax[1, 1].scatter(
        np.diff(sim.simulation_results.brightest_length_arr / 1000) / sim.const.dt,
        sim.simulation_results.brightest_height_arr[:-1],
        marker='o',
        s=2,
    )
    ax[1, 1].set_xlabel('Velocity (km/s)')
    ax[1, 1].set_ylabel('Ht')

    plt.show()

    fig, ax = plt.subplots(2, sharey=True)

    ax[0].plot(
        simulation_results.abs_magnitude[:-1],
        simulation_results.brightest_height_arr[:-1] / 1000,
        label='ML predicted output',
    )
    ax[1].scatter(
        np.diff(simulation_results.brightest_length_arr / 1000)[:-1] / sim.const.dt,
        simulation_results.brightest_height_arr[:-2] / 1000,
        marker='o',
        s=2,
        label='ML predicted output',
    )

    ax[0].plot(
        sim.simulation_results.abs_magnitude[:-1],
        sim.simulation_results.brightest_height_arr[:-1] / 1000,
        label='Correct simulated output',
        c='k',
    )
    ax[1].scatter(
        np.diff(sim.simulation_results.brightest_length_arr / 1000)[:-1] / sim.const.dt,
        sim.simulation_results.brightest_height_arr[:-2] / 1000,
        marker='o',
        c='k',
        s=2,
        label='Correct simulated output',
    )

    ax[0].set_ylabel('Height (km)')
    ax[0].set_xlabel("Magnitude")
    ax[0].legend()
    ax[1].set_ylabel('Height (km)')
    ax[1].set_xlabel("Velocity (km/s)")
    ax[1].legend()

    ax[0].set_ylim([ending_height, starting_height])
    ax[1].set_ylim([ending_height, starting_height])
    ax[0].set_xlim(right=8)
    print('correct', sim.params.getInputs())

    plt.show()


def fitCNNMultiHeaded(data_gen, validation_gen, output_dir, model_file, weights_file, fit_param=None):
    # https://machinelearningmastery.com/how-to-develop-convolutional-neural-network-models-for-time-series-forecasting/
    # Height input model
    model_title = weights_file[:-3]
    checkpoint_filepath = os.path.join(output_dir, f'{model_title}_checkpoint.h5')
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath, mode='min', verbose=1, save_weights_only=True
    )

    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor='loss', patience=5, min_delta=0, verbose=1
    )

    # visible0 = keras.engine.input_layer.Input(shape=(DATA_LENGTH, 1))
    # cnn0 = keras.layers.Conv1D(filters=64, kernel_size=10, activation='relu')(visible0)
    # cnn0 = keras.layers.MaxPooling1D(pool_size=2)(cnn0)
    # cnn0 = keras.layers.Flatten()(cnn0)
    # cnn0 = keras.layers.Dense(256, kernel_initializer='normal', activation='relu')(cnn0)
    # cnn0 = keras.layers.Dense(256, kernel_initializer='normal', activation='relu')(cnn0)

    # visible1 = keras.engine.input_layer.Input(shape=(DATA_LENGTH, 1))
    # cnn1 = keras.layers.Conv1D(filters=64, kernel_size=10, activation='relu')(visible1)
    # cnn1 = keras.layers.MaxPooling1D(pool_size=2)(cnn1)
    # cnn1 = keras.layers.Flatten()(cnn1)
    # cnn1 = keras.layers.Dense(256, kernel_initializer='normal', activation='relu')(cnn1)
    # cnn1 = keras.layers.Dense(256, kernel_initializer='normal', activation='relu')(cnn1)

    # # Length input model
    # visible2 = keras.engine.input_layer.Input(shape=(DATA_LENGTH, 1))
    # cnn2 = keras.layers.Conv1D(filters=64, kernel_size=10, activation='relu')(visible2)
    # cnn2 = keras.layers.MaxPooling1D(pool_size=2)(cnn2)
    # cnn2 = keras.layers.Flatten()(cnn2)
    # cnn2 = keras.layers.Dense(256, kernel_initializer='normal', activation='relu')(cnn2)
    # cnn2 = keras.layers.Dense(256, kernel_initializer='normal', activation='relu')(cnn2)

    # # Magnitude input model
    # visible3 = keras.engine.input_layer.Input(shape=(DATA_LENGTH, 1))
    # cnn3 = keras.layers.Conv1D(filters=64, kernel_size=10, activation='relu')(visible3)
    # cnn3 = keras.layers.MaxPooling1D(pool_size=2)(cnn3)
    # cnn3 = keras.layers.Flatten()(cnn3)
    # cnn3 = keras.layers.Dense(256, kernel_initializer='normal', activation='relu')(cnn3)
    # cnn3 = keras.layers.Dense(256, kernel_initializer='normal', activation='relu')(cnn3)

    input = keras.engine.input_layer.Input(shape=(DATA_LENGTH, 4))
    cnn = keras.layers.Conv1D(filters=64, kernel_size=10, activation='relu')(input)
    cnn = keras.layers.MaxPooling1D(pool_size=2)(cnn)
    cnn = keras.layers.Flatten()(cnn)

    # merge input models
    # merge = keras.layers.Concatenate()([cnn0, cnn1, cnn2, cnn3])
    dense = keras.layers.Dense(256, kernel_initializer='normal', activation='relu')(cnn)
    dense = keras.layers.Dense(256, kernel_initializer='normal', activation='relu')(dense)
    dense = keras.layers.Dense(256, kernel_initializer='normal', activation='relu')(dense)
    dense = keras.layers.Dense(256, kernel_initializer='normal', activation='relu')(dense)
    output = keras.layers.Dense(
        10,
        kernel_initializer='normal',
        activation="linear",
        batch_size=batch_size,
        # activity_regularizer=keras.regularizers.l1(0.01),
    )(dense)

    # Tie inputs together
    model = keras.models.Model(inputs=input, outputs=output)

    def loss_fn(y_true, y_pred):
        if fit_param:
            weights = tf.one_hot(fit_param, 10, dtype=tf.float32)
        else:
            weights = tf.Tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=tf.float32)
        # print(K.sum(K.square(y_true - y_pred) * weights))
        # raise Exception('hey')
        return K.sum(K.square(y_true - y_pred) * weights / K.sum(weights), axis=-1)

    # Compile the model
    model.compile(optimizer='adam', loss=loss_fn)

    # Save the model to disk BEFORE fitting, so that it plus the checkpoint will have all information
    model_json = model.to_json()

    model_file = os.path.join(output_dir, model_file)
    weights_file = os.path.join(output_dir, weights_file)
    with open(model_file, "w") as json_file:
        json_file.write(model_json)

    # fit model
    history = model.fit(
        x=iter(data_gen),
        steps_per_epoch=data_gen.steps_per_epoch,
        epochs=data_gen.epochs,
        callbacks=[ReportFitGoodness(validation_gen), model_checkpoint_callback, early_stopping_callback],
        workers=0,
        max_queue_size=1,
    )

    # serialize weights to HDF5
    model.save_weights(weights_file)
    print("Saved model to disk")

    new_file = not os.path.exists(os.path.join(output_dir, 'log.csv'))
    with open(os.path.join(output_dir, 'log.csv'), 'a+') as f:
        if new_file:
            f.write('Date,model name,parameters,batch size,step per epoch,epochs,training data,final loss\n')
        f.write(
            f"{datetime.datetime.now()},{model_title},{model.count_params()},{data_gen.batch_size},"
            f"{data_gen.steps_per_epoch},{data_gen.epochs},{len(data_gen.training_list)},"
            f"{history.history['loss'][-1]}\n"
        )

    # Evaluate fit quality
    evaluateFit(model, iter(validation_gen))


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
        type=int,
        help='Inputting this parameter will not train the model, but instead evaluate the model by visually '
        'showing what it predicts compared to the simulation.',
    )
    arg_parser.add_argument(
        '--grouping',
        type=int,
        nargs=2,
        help='Allows for specifying the batch size and steps per epoch, respectively.',
    )
    arg_parser.add_argument('--fitparam', type=int)
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

    if cml_args.evaluate is not None:
        # Init the validation generator
        data_gen = DataGenerator(
            data_list, batch_size, steps_per_epoch, param_class_name=cml_args.classname, validation=True
        )
        model = loadModel(cml_args.output_dir, model_file, weights_file)
        evaluateFit2(model, data_list[cml_args.evaluate], data_gen, cml_args.classname)
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
        fitCNNMultiHeaded(
            data_gen,
            validation_gen,
            cml_args.output_dir,
            model_file,
            weights_file,
            fit_param=cml_args.fitparam,
        )

