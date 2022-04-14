""" Fit the erosion model using machine learning. """

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import datetime
import os
import random
import time
from multiprocessing import parent_process
from re import A
from typing import List, Optional, Union

import h5py
import keras
import keras.backend as K
import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import scipy
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
    dataFunction,
    extractSimData,
    getFileList,
)
from wmpl.MetSim.ML.PostprocessSims import getProcessedDataLength, loadProcessedData, saveData
from wmpl.Utils.Pickling import loadPickle
from wmpl.Utils.PyDomainParallelizer import domainParallelizer


def calculatePredictedSimulation(phys_params, normalized_output_param_val):
    phys_params = copy.deepcopy(phys_params)
    phys_params.setParamValues(phys_params.getDenormalizedInputs(normalized_output_param_val))
    const = phys_params.getConst()
    simulation_results = SimulationResults(const, *runSimulation(const))
    return phys_params, simulation_results


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
        random.Random(0).shuffle(self.data_list)

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
            file_list = data_list[curr_index : curr_index + self.batch_size]

            # Load pickle files and postprocess in parallel
            domain = [[file_path, self.param_class_name] for file_path in file_list]

            # Postprocess the data in parallel
            new_res = domainParallelizer(domain, dataFunction)

            filtered_res_list = []
            # discard bad results
            for i, res in enumerate(new_res):
                if res[0] is None:
                    to_delete.append(res[1])
                else:
                    filtered_res_list.append(res)

            res_list += filtered_res_list
            curr_index += self.batch_size

            # if you fully loop data, shuffle it
            if curr_index >= len(data_list):
                # delete specific files since threads don't preserve order
                data_list = np.setdiff1d(data_list, to_delete)
                if self.validation:
                    self.validation_list = data_list
                else:
                    self.training_list = data_list

                if len(data_list) == 0:
                    raise Exception("No valid data")
                to_delete = []
                curr_index = 0
                self.random_state.shuffle(data_list)

            # if there aren't enough results to fill the batch, collect another batch and fill the gaps with it
            # where extras will be used in subsequent iterations
            if len(res_list) < self.batch_size:
                continue

            next_res_list = res_list[self.batch_size :]
            res_list = res_list[: self.batch_size]

            # Split results to input/output list
            sim, extract_data = zip(*res_list)
            _, result_list, param_list = zip(*extract_data)

            res_list = next_res_list

            param_list = np.array(param_list)
            result_list = np.array(result_list)

            # yield dimenions [(batch_size, data_length, 4), (batch_size, 10)]
            if self.validation:
                yield sim, result_list, param_list
            else:
                yield result_list, param_list


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


def correlationPlot(X, Y, log, param_name_list, param_unit, param_pretext):
    fig, ax = plt.subplots(10, 10, sharex='col', sharey='row')
    for i in range(10):
        for j in range(10):
            # ax[j, i].set_title(param_name_list[i])
            x = X[:, i]
            y = Y[:, j]
            # making the diagonals histograms is difficult because the y axis would be different
            if log[j]:
                ax[j, i].set_yscale('log')
                ybins = np.logspace(np.log10(np.min(y[y > 0])), np.log10(np.max(y)), 50)
            else:
                ybins = np.linspace(np.min(y), np.max(y), 50)

            if log[i]:
                ax[j, i].set_xscale('log')
                xbins = np.logspace(np.log10(np.min(x[x > 0])), np.log10(np.max(x)), 50)
            else:
                xbins = np.linspace(np.min(x), np.max(x), 50)

            counts, _, _ = np.histogram2d(x, y, bins=(xbins, ybins))
            ax[j, i].pcolormesh(xbins, ybins, counts.T)
            # ax[j, i].plot(
            #     [
            #         getattr(camera_param, camera_param.param_list[i]).min * param_scaling[i],
            #         getattr(camera_param, camera_param.param_list[i]).max * param_scaling[i],
            #     ],
            #     [
            #         getattr(camera_param, camera_param.param_list[j]).min * param_scaling[j],
            #         getattr(camera_param, camera_param.param_list[j]).max * param_scaling[j],
            #     ],
            # )
            if i == 0:
                ax[j, i].set_ylabel(f'{param_pretext[1]} {param_name_list[j]} ' + param_unit[j])
            if j == len(ax) - 1:
                ax[j, i].set_xlabel(f'{param_pretext[0]} {param_name_list[i]} ' + param_unit[i])

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def evaluateFit(model, validation_gen, output=False, display=False, log=None):
    """
    Evaluate fit via comparing parameters
    
    Arguments:
        model: []
        validation_gen: [DataGenerator]
        output: [bool] Whether to print evaluation info
        display: [bool] Whether to display evaluation info. Will stop program until plot is closed
        log: [list of bool] List of 10 boolena elements. If true, when display=True, the axis
            corresponding to the index will use log scale
    """
    param_name_list = ["M0", "V0", "ZC", "DENS", "ABL", "ERHT", "ERCO", "ER_S", "ERMm", "ERMM"]
    param_unit = ['(kg)', '(km/s)', '(deg)', '(kg/m3)', '(kg/MJ)', '(km)', '(kg/MJ)', '', '(kg)', '(kg)']
    param_scaling = np.array([1, 1 / 1000, 180 / np.pi, 1, 1e6, 1 / 1000, 1e6, 1, 1, 1])

    # Generate validation data
    # sim_list, validation_outputs, validation_inputs = next(validation_gen)
    # phys_param = sim_list[0].params
    validation_outputs, validation_inputs = next(validation_gen)
    phys_param = PhysicalParameters()

    # Predict data
    pred_norm_params = model.predict(validation_outputs)

    # noramlized data
    norm_errors = np.abs(pred_norm_params - validation_inputs)
    param_corr = np.corrcoef(pred_norm_params, validation_inputs, rowvar=False)
    param_corr = param_corr.flatten()[
        int(param_corr.shape[0] / 2) : int(param_corr.size / 2) : param_corr.shape[0] + 1
    ]

    # unnormalized data
    correct_output = np.array(phys_param.getDenormalizedInputs(validation_inputs.T)).T
    pred_output = np.array(phys_param.getDenormalizedInputs(pred_norm_params.T)).T
    denorm_errors = np.abs(pred_output - correct_output)
    denorm_perc_errors = denorm_errors / correct_output

    if display:
        # meteor_err = np.sqrt(np.sum(norm_errors[:, :5] ** 2, axis=1))
        # rel_error = meteor_err / np.max(meteor_err)

        # filter = (correct_output[:, 3] < 500) & (correct_output[:, 4] < 2e-7)
        # filter2 = (
        #     (correct_output[:, 3] > 1800) & (correct_output[:, 3] < 2400) & (correct_output[:, 4] < 4e-7)
        # ) & (rel_error > 0.5)

        # if np.sum(filter):
        #     data = validation_outputs[filter]
        #     fig, ax = plt.subplots(2, sharey=True)
        #     scat1 = ax[0].scatter(
        #         data[:, :, 3].T,
        #         data[:, :, 1].T,
        #         c=np.stack((rel_error[filter],) * data.shape[1]),
        #         s=1,
        #         alpha=0.4,
        #     )  # magnitude
        #     scat2 = ax[1].scatter(
        #         np.diff(data[:, :, 2].T, axis=0),
        #         data[:, :, 1].T[:-1],
        #         c=np.stack((rel_error[filter],) * data.shape[1])[:-1],
        #         s=1,
        #         alpha=0.3,
        #     )
        #     ax[0].set_ylabel('Height (km)')
        #     ax[0].set_xlabel('Mag')
        #     plt.colorbar(scat1, ax=ax[0])
        #     plt.colorbar(scat2, ax=ax[1])
        #     ax[1].set_ylabel('Height (km)')
        #     ax[1].set_xlabel('velocity')
        #     ax[1].legend()
        #     plt.show()

        # if np.sum(filter2):
        #     data = validation_outputs[filter2]
        #     fig, ax = plt.subplots(2, sharey=True)
        #     scat1 = ax[0].scatter(
        #         data[:, :, 3].T,
        #         data[:, :, 1].T,
        #         c=np.stack((rel_error,) * data.shape[1])[:, filter2],
        #         s=1,
        #         alpha=0.3,
        #     )  # magnitude
        #     scat2 = ax[1].scatter(
        #         np.diff(data[:, :, 2].T, axis=0),
        #         data[:, :, 1].T[:-1],
        #         c=np.stack((rel_error,) * data.shape[1])[:, filter2][:-1],
        #         s=1,
        #         alpha=0.3,
        #     )
        #     ax[0].set_ylabel('Height (km)')
        #     ax[0].set_xlabel('Mag')
        #     plt.colorbar(scat1, ax=ax[0])
        #     plt.colorbar(scat2, ax=ax[1])
        #     ax[1].set_ylabel('Height (km)')
        #     ax[1].set_xlabel('velocity')
        #     ax[1].legend()
        #     plt.show()

        # (100, 10) (100, 256, 4)
        # print(pred_norm_params.shape, validation_outputs.shape)
        # dist_mat = scipy.spatial.distance.cdist(pred_norm_params, pred_norm_params)  # (100, 100)
        # print(np.sum(dist_mat == 0))
        # dist_mat = dist_mat[~np.eye(dist_mat.shape[0], dtype=bool)].reshape(dist_mat.shape[0], -1)
        # print(dist_mat)
        # closest_points = np.argpartition(dist_mat, 10, axis=1)[:, :10]  # (100, 9)
        # print(closest_points)
        # min_dist = dist_mat[np.arange(dist_mat.shape[0])[:, None], closest_points]  # (100, 9)
        # print(min_dist)
        # closest_output_points = validation_outputs[closest_points.T]
        # # (9, 100, 256, 4) - (100, 256, 4)
        # min_dist_error = (
        #     (np.max(closest_output_points[..., 2], axis=-1) - np.max(validation_outputs[..., 2], axis=-1))
        #     ** 2
        #     + (
        #         closest_output_points[
        #             np.arange(closest_output_points.shape[0])[:, None],
        #             np.arange(closest_output_points.shape[1])[None, :],
        #             np.argmax(closest_output_points[..., 2], axis=-1),
        #             1,
        #         ]
        #         - validation_outputs[
        #             np.arange(validation_outputs.shape[0]), np.argmax(validation_outputs[..., 2], axis=-1), 1,
        #         ]
        #     )
        #     ** 2
        # ).T
        # peak_height = validation_outputs[
        #     np.arange(validation_outputs.shape[0]), np.argmax(validation_outputs[..., 2], axis=-1), 1,
        # ].T
        # peak_mag = np.max(validation_outputs[..., 2], axis=-1)
        # plt.scatter(
        #     peak_height, peak_mag,
        # )
        # plt.xlabel('Peak height (km)')
        # plt.ylabel('Peak magnitude')
        # plt.show()
        # dist_mat[np.arange(dist_mat.shape[0])[:,None],

        # plt.scatter(np.sum(validation_outputs[:, :, 2] > 0, axis=1), rel_error)
        # plt.xlabel('Number of magnitude points')
        # plt.ylabel('Error')
        # plt.show()

        # fig, ax = plt.subplots(2, sharey=True, sharex=True)
        # ax[0].scatter(*validation_inputs[:, [1, 7]].T, label='correct')
        # # ax[0].set_yscale('log')
        # # ax[0].set_xscale('log')
        # ax[0].set_xlabel('M0')
        # ax[0].set_ylabel('ERMM')
        # ax[0].legend()
        # ax[1].scatter(*pred_norm_params[:, [1, 7]].T, label='predicted')
        # # ax[1].set_yscale('log')
        # # ax[1].set_xscale('log')
        # ax[1].legend()
        # ax[1].set_xlabel('M0')
        # ax[1].set_ylabel('ERMM')
        # plt.show()

        if log is None:
            log = [True, False, False, False, False, False, False, False, True, True]
        scaled_corr = correct_output * param_scaling
        scaled_pred = pred_output * param_scaling

        correlationPlot(scaled_corr, scaled_pred, log, param_name_list, param_unit, ['Correct', 'Predicted'])
        correlationPlot(scaled_pred, scaled_pred, log, param_name_list, param_unit, ['', ''])
        # correlationPlot(scaled_corr, scaled_corr, log, param_name_list, param_unit, ['', ''])

        # plt.scatter(
        #     scaled_corr[:, 3], scaled_corr[:, 4], c=rel_error,
        # )
        # plt.xlabel('Density (kg/m3)')
        # plt.ylabel('Ablation coeffient (kg/MJ)')
        # plt.colorbar()
        # plt.show()

        # fig, ax = plt.subplots(len(denorm_errors.T), sharex=True, sharey=True)
        # for a, values, label in zip(ax, norm_errors.T, param_name_list):
        #     a.hist(values, bins='auto', label=label)
        #     a.legend(loc='upper right')
        # # a.set_yscale('log')
        # a.set_xlim([0, 1])
        # plt.show()

    # Compute mean absolute percentage error for every model parameter
    percent_norm_errors = 100 * np.mean(norm_errors, axis=0)
    denorm_perc_errors_av = 100 * np.mean(denorm_perc_errors, axis=0)
    # mean minimum distance between a correct set of parameters and predicted set. The correct set
    # is the goal "uniformity", so the closest to that the predicted parameters are, the better.
    dens_abl_corr = np.corrcoef(*pred_norm_params[:, 3:5].T)[0, 1]

    if output:
        print("Mean absolute percentage error and mean absolute error per parameter:")
        print(" ".join(["{:>9s}".format(param_name) for param_name in param_name_list]))
        print(str(len(percent_norm_errors) * "{:8.2f}% ").format(*percent_norm_errors))
        print(str(len(denorm_perc_errors_av) * "{:8.2f}% ").format(*denorm_perc_errors_av))
        print(str(len(param_corr) * "{:9.4f} ").format(*param_corr))
        print(f'Density-ablation correlation: {dens_abl_corr}')

    return percent_norm_errors, denorm_perc_errors_av, param_corr


def evaluateFit2(model, validation_gen, mode=1, noerosion=False, param_class_name=None):
    """ Evaluates model by visually comparing expected simulation values to the simulation values 
    given from the prediction 
    
    Arguments:
        model: [Model] Trained model to evaluate
    
    keyword arguments:
        mode: [int] 1 for analysis on all meteors in batch, 2 for a random meteor in batch
    """

    sim_list, norm_sim_data, norm_input_param_vals = next(validation_gen)

    # apply neural network on normalized data to get a set of parameters
    normalized_output_param_vals = model.predict(norm_sim_data)  # dimensions (batch_size, 256, 4)

    # if there is no erosion, set the erosion height to 0 for the simulation
    # normalized_output_param_vals[:, 1:] = norm_input_param_vals[:, 1:]
    if noerosion:
        # we already know the zenith angle
        normalized_output_param_vals[:, 2] = norm_input_param_vals[:, 2]

        # normalized_output_param_vals[:, :3] = norm_input_param_vals[:, :3]
        # normalized_output_param_vals[:, 3:5] = norm_input_param_vals[:, 3:5]
        normalized_output_param_vals[:, 5] = -1
        # see what happens with correct values

    if mode == 2:
        print()
        print('correct norm', list(norm_input_param_vals[0]))
        print('pred norm', list(normalized_output_param_vals[0]))
        print()
        print(
            'perc error',
            list(np.abs(normalized_output_param_vals[0] - np.array(norm_input_param_vals[0])) * 100),
        )
        print()

    camera_param = SIM_CLASSES_DICT.get(param_class_name, ErosionSimParameters)()
    example_phys_params = sim_list[0].params

    # run the simulation for each set of parameters in the batch and compare them to what they should be
    domain = [
        [example_phys_params, normalized_output_param_val]
        for normalized_output_param_val in normalized_output_param_vals
    ]

    # Postprocess the data in parallel
    t1 = time.perf_counter()
    ret = domainParallelizer(domain, calculatePredictedSimulation)
    phys_param_list, pred_simulation_result_list = zip(*ret)
    print(time.perf_counter() - t1)

    if mode == 2:
        i = np.random.randint(len(sim_list))

        sim = sim_list[i]
        correct_phys_params = sim.params
        pred_simulation_results = pred_simulation_result_list[i]
        phys_params = phys_param_list[i]

        print('correct', correct_phys_params.getInputs())
        print('predicted', phys_params.getInputs())

        fig, ax = plt.subplots(2, 2, sharey='col')
        ax[0, 0].plot(norm_sim_data[i, :, 3], norm_sim_data[i, :, 1])
        ax[0, 0].set_xlabel('Mag')
        ax[0, 0].set_ylabel('Ht')

        ax[1, 0].scatter(np.diff(norm_sim_data[i, :, 2]), norm_sim_data[i, :-1, 1], marker='o', s=2)
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
            pred_simulation_results.abs_magnitude[:-1],
            pred_simulation_results.brightest_height_arr[:-1] / 1000,
            label='ML predicted output',
        )
        ax[1].scatter(
            np.diff(pred_simulation_results.brightest_length_arr / 1000)[:-1]
            / np.diff(pred_simulation_results.time_arr)[:-1],
            pred_simulation_results.brightest_height_arr[:-2] / 1000,
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

        starting_height = (
            pred_simulation_results.brightest_height_arr[np.argmax(pred_simulation_results.abs_magnitude < 8)]
            / 1000
        )
        ending_height = (
            pred_simulation_results.brightest_height_arr[
                -np.argmax(pred_simulation_results.abs_magnitude[::-1] < 8) - 1
            ]
            / 1000
            - 10
        )

        ax[0].set_ylim([ending_height, starting_height])
        ax[1].set_ylim([ending_height, starting_height])
        ax[0].set_xlim(right=8)

        plt.show()
    else:
        mag_err_arr = np.array([])
        vel_err_arr = np.array([])

        dh_arr = np.array([])
        dm_arr = np.array([])
        dv_arr = np.array([])
        dv2_arr = np.array([])

        for i, (pred_simulation_result, sim) in enumerate(zip(pred_simulation_result_list, sim_list)):
            pred_mag_func = scipy.interpolate.interp1d(
                pred_simulation_result.brightest_height_arr, pred_simulation_result.abs_magnitude
            )

            pred_vel_func = scipy.interpolate.interp1d(
                pred_simulation_result.brightest_height_arr, pred_simulation_result.brightest_vel_arr,
            )

            filter = (sim.simulation_results.brightest_height_arr > camera_param.ht_min) & (
                sim.simulation_results.brightest_height_arr < camera_param.ht_max
            )

            cor_i = np.argmin(sim.simulation_results.abs_magnitude)
            sim_i = np.argmin(pred_simulation_result.abs_magnitude)
            h1 = sim.simulation_results.brightest_height_arr[cor_i]
            h2 = pred_simulation_result.brightest_height_arr[sim_i]
            acc_v1 = sim.simulation_results.brightest_vel_arr[cor_i]
            acc_v2 = pred_simulation_result.brightest_vel_arr[sim_i]
            m1 = sim.simulation_results.abs_magnitude[cor_i]
            m2 = pred_mag_func(h1)

            mag_err = (
                pred_mag_func(sim.simulation_results.brightest_height_arr[filter])
                - sim.simulation_results.abs_magnitude[filter]
            )
            vel_err = (
                pred_vel_func(sim.simulation_results.brightest_height_arr[filter])
                - sim.simulation_results.brightest_vel_arr[filter]
            )

            mag_err_arr = np.append(mag_err_arr, mag_err)
            vel_err_arr = np.append(vel_err_arr, vel_err)
            dm_arr = np.append(dm_arr, m2 - m1)
            dh_arr = np.append(dh_arr, h2 - h1)
            dv_arr = np.append(dv_arr, vel_err[0])
            dv2_arr = np.append(dv2_arr, acc_v2 - acc_v1 - vel_err[0])

        mag_filter = (mag_err_arr > -2) & (mag_err_arr < 2)
        vel_filter = (vel_err_arr > -10_000) & (vel_err_arr < 10_000)
        plt.subplot(1, 2, 1)
        plt.hist(mag_err_arr[mag_filter], bins=200, density=True)
        plt.xlabel('Magnitude error')
        plt.ylabel("Probability density")
        plt.title(
            rf'$\sigma$ = {np.std(mag_err_arr[mag_filter]):.3f},  $\mu$ = {np.mean(mag_err_arr[mag_filter]):.3f}'
        )
        # print(list(length_err_arr))

        plt.subplot(1, 2, 2)
        plt.hist(vel_err_arr[vel_filter] / 1000, bins=200, density=True)
        plt.xlabel('Velocity error (km/s)')
        plt.ylabel("Probability density")
        plt.title(
            rf'$\sigma$ = {np.std(vel_err_arr[vel_filter])/1000:.3f},  $\mu$ = {np.mean(vel_err_arr[vel_filter])/1000:.3f}'
        )

        plt.show()

        plt.hist2d(
            mag_err_arr[mag_filter & vel_filter],
            vel_err_arr[mag_filter & vel_filter] / 1000,
            bins=[100, 100],
        )
        plt.xlabel('Magnitude error')
        plt.ylabel('Length error (km)')
        plt.show()

        plt.subplot(2, 2, 1)
        plt.hist(dm_arr, bins='auto', density=True)
        plt.xlabel('Magnitude difference at expected peak height [pred-correct]')
        plt.ylabel("Probability density")
        plt.title(rf'$\sigma$ = {np.std(dm_arr):.3f},  $\mu$ = {np.mean(dm_arr):.3f}')

        plt.subplot(2, 2, 2)
        plt.hist(dh_arr / 1000, bins='auto', density=True)
        plt.xlabel('Peak height difference [pred-correct] (km)')
        plt.ylabel("Probability density")
        plt.title(rf'$\sigma$ = {np.std(dh_arr / 1000):.3f},  $\mu$ = {np.mean(dh_arr / 1000):.3f}')

        plt.subplot(2, 2, 3)
        plt.hist(dv_arr / 1000, bins='auto', density=True)
        plt.xlabel('Initial Velocity error (km/s)')
        plt.ylabel("Probability density")
        plt.title(rf'$\sigma$ = {np.std(dv_arr)/1000:.3f},  $\mu$ = {np.mean(dv_arr)/1000:.3f}')

        plt.subplot(2, 2, 4)
        plt.hist(dv2_arr / 1000, bins='auto', density=True)
        plt.xlabel('Peak magnitude velocity decrease error (km/s)')
        plt.ylabel("Probability density")
        plt.title(rf'$\sigma$ = {np.std(dv2_arr)/1000:.3f},  $\mu$ = {np.mean(dv2_arr)/1000:.3f}')

        plt.show()

        # plt.hist2d(dm_arr, dh_arr, bins=[40, 40], density=True)
        plt.scatter(dm_arr, dh_arr / 1000, s=1)
        plt.xlabel('Magnitude difference at expected peak height [pred-correct]')
        plt.ylabel('Peak height difference [pred-correct] (km)')
        # plt.colorbar()
        plt.show()


def testFitModels(data_gen, validation_gen, batch_size, steps_per_epoch, epochs):
    def generateModel(hp):
        model = keras.Sequential()
        conv_layers = hp.Int("conv layers", min_value=1, max_value=4, step=1)
        filters = hp.Int('filters', min_value=15, max_value=30, step=3)
        kernels = hp.Int('kernels', min_value=4, max_value=20, step=2)
        # pooling = hp.Boolean('pooling')
        # pooling_size = hp.Int('pooling size', min_value=2, max_value=10, step=2)
        units = hp.Int('units', min_value=400, max_value=800, step=20)
        dense_layers = hp.Int('dense layers', min_value=3, max_value=6, step=1)

        model.add(keras.layers.Input(shape=(DATA_LENGTH, 3)))
        for i in range(conv_layers):
            model.add(
                keras.layers.Conv1D(filters=filters, kernel_size=kernels, activation='relu', padding='same')
            )
            # if pooling:
            #     model.add(keras.layers.MaxPooling1D(pool_size=pooling_size, padding='same'))
        model.add(keras.layers.Flatten())

        for i in range(dense_layers):
            model.add(keras.layers.Dense(units, activation='relu', kernel_initializer='normal'))

        model.add(keras.layers.Dense(10, kernel_initializer='normal'))
        model.compile(optimizer='adam', loss='mse')

        return model

    tuner = kt.RandomSearch(generateModel, 'loss', max_trials=500)
    tuner.search(
        x=iter(data_gen),
        # validation_data=iter(validation_gen),
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        epochs=1,
    )

    model = tuner.get_best_models()[0]
    model.fit(
        x=iter(data_gen),
        validation_data=iter(validation_gen),
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        epochs=epochs,
    )

    print(tuner.get_best_hyperparameters()[0].values)


def fit_data(yi):
    try:
        filter = yi[:, 2] > 0
        dist_func = lambda t, h0, v, a, b: h0 + v * t - np.abs(a) * np.exp(np.abs(b) * t)
        t0 = yi[0, 0]
        popt1, pcov = scipy.optimize.curve_fit(dist_func, yi[filter, 0] - t0, yi[filter, 2], maxfev=100_000)

        filter = yi[:, 3] > 0
        mag_func = lambda t, a, b, c, loc, scale: c * scipy.stats.beta.pdf(t, a, b, loc, scale)
        popt2, pcov = scipy.optimize.curve_fit(
            mag_func,
            yi[filter, 1],
            yi[filter, 3],
            maxfev=10_000,
            bounds=([1, 1, 0, 0, 0], [np.inf, np.inf, np.inf, 1, 2]),
        )

        return np.append(popt1, popt2)
    except RuntimeError:
        return np.array([np.nan] * 9)


def fitCurveInverse(
    data_gen,
    validation_gen,
    output_dir: str,
    model_file: str,
    weights_file: str,
    batch_size: Optional[int] = None,
    steps_per_epoch: Optional[int] = None,
    epochs: Optional[int] = None,
):
    model_name = weights_file[:-3]

    def feature_engineer(gen):
        for y, x in gen:
            new_y = np.array(domainParallelizer(y[:, None], fit_data))
            # print(new_y.shape)
            # dist_func = lambda t, a, b, c, loc, scale: c * scipy.stats.beta.pdf(t, a, b, loc, scale)

            # for i in range(10):
            #     plt.plot(y[i, :, 0] - y[i, 0, 0], func(y[i, :, 0] - y[i, 0, 0], *new_y[i]) - y[i, :, 2])
            # print(new_y[0])
            # for i in range(10):
            #     plt.plot(y[i, :, 1], dist_func(y[i, :, 1], *new_y[i]))
            #     plt.plot(y[i, :, 1], y[i, :, 3])
            #     plt.show()
            # raise Exception('hey')
            yield new_y, x

    data_gen = saveData(feature_engineer(data_gen), output_dir, 'fit_dataset', 'y', 'x')

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(9,)))
    for i in range(10):
        model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(10))

    model.compile(loss='mse', optimizer='adam')
    model.fit(x=data_gen, steps_per_epoch=steps_per_epoch, epochs=epochs)
    keras.models.save_model(
        model, os.path.join(output_dir, model_name + '_fit_inverse_solver.hdf5'), save_format='h5'
    )


def fitForwardProblem(
    data_gen,
    validation_gen,
    output_dir: str,
    model_file: str,
    weights_file: str,
    batch_size: Optional[int] = None,
    steps_per_epoch: Optional[int] = None,
    epochs: Optional[int] = None,
):
    model_name = weights_file[:-3]

    # restructuring b=(batchsize, nparams) and a=(batchsize, DATA_LENGTH, 3) ->
    #   (batchsize*DATA_LENGTH, 1+nparams) and (batchsize*DATA_LENGTH, 2)

    data_gen = (
        (
            np.concatenate((np.concatenate(a[..., 1:2]), np.repeat(b, DATA_LENGTH, axis=0)), axis=1),
            np.concatenate(a[..., 3:4]),
        )
        for (a, b) in data_gen
    )

    def feature_engineer(gen):
        for x, y in gen:
            x = np.concatenate((x, x[:, 0:1] < x[:, 6:7]), axis=1)
            yield x, y

    validation_gen = (
        (
            np.concatenate((np.concatenate(a[..., 1:2]), np.repeat(b, DATA_LENGTH, axis=0)), axis=1),
            np.concatenate(a[..., 3:4]),
        )
        for (a, b) in validation_gen
    )

    data_gen = feature_engineer(data_gen)
    # def build_model(hp):
    #     # relu, adam, 190, 6
    #     units = hp.Int('units', min_value=10, max_value=150, step=10)
    #     hidden_layers = hp.Int('layers', min_value=1, max_value=10, step=1)

    #     model = keras.Sequential()
    #     model.add(keras.layers.Input(shape=(11,)))
    #     for i in range(hidden_layers):
    #         model.add(keras.layers.Dense(units, activation='relu'))

    #     model.add(keras.layers.Dense(2))
    #     model.compile(optimizer='adam', loss='mse')
    #     return model

    # tuner = kt.RandomSearch(build_model, objective='loss', max_trials=100)
    # tuner.search(
    #     data_gen,
    #     # validation_data=validation_gen,
    #     epochs=2,
    #     batch_size=batch_size,
    #     steps_per_epoch=steps_per_epoch,
    # )
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(1 + 10 + 1,)))
    for i in range(7):
        model.add(keras.layers.Dense(110, activation='relu'))
    model.add(keras.layers.Dense(1))

    def loss_fn(y_true, y_pred):
        # penalty = K.mean(
        #     keras.activations.relu(y_pred[:, 1:, 0] - y_pred[:, :-1, 0])
        #     * K.cast(y_true[:, :-1, 0] > 0, 'float32'),
        #     axis=1,
        # )
        # require that heights decrease
        err = K.square(y_true - y_pred)  # (batchsize, 2)
        return err[:, 0] * (K.cast(y_true[:, 0] > 0, 'float32'))  # + err[:, 1] * (
        #     K.cast(y_true[:, 1] > 0, 'float32')
        # )

    model.compile(optimizer='adam', loss=loss_fn)
    model.fit(x=data_gen, steps_per_epoch=steps_per_epoch, epochs=epochs)
    keras.models.save_model(
        model, os.path.join(output_dir, model_name + '_forward_problem_vel.hdf5'), save_format='h5'
    )


def fitInverseProblem(
    data_gen,
    validation_gen,
    output_dir: str,
    model_file: str,
    weights_file: str,
    batch_size: Optional[int] = None,
    steps_per_epoch: Optional[int] = None,
    epochs: Optional[int] = None,
    load: bool = False,
    fit_param: Optional[List[int]] = None,
    extra_information: Optional[dict] = None,
):
    """
    Trains and saves neural network
    
    Arguments:
        data_gen: [generator] Generator which will be iterated over to get data
        validation_gen: [generator] Generator which will iterated over to get validation data
        model_file: [str] Name of model file to save/load (does include .json extension)
        weights_file: [str] Name of weights file to save/load (does include .h5 extension)

    
    keyword arguments:
        batch_size: [int] Batch size. If not given, will read from data_gen
        steps_per_epoch: [int] Number of steps per epoch
        epochs: [int] Number of epochs
        load: [bool] Whether to load function
        fit_param: [list[int]] A list of 10 values that are either 0 or 1. If 1, the corresponding parameter
            will not be considered in loss function
        extra_information: [dict] Dictionary of extra information that will be saved to a file upon completion
    
    """
    # define epochs and steps per epoch
    if steps_per_epoch is None or epochs is None:
        batch_size = data_gen.batch_size
        steps_per_epoch = data_gen.steps_per_epoch
        epochs = data_gen.epochs

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

    input = keras.engine.input_layer.Input(shape=(DATA_LENGTH, 3))
    cnn = keras.layers.Conv1D(filters=22, kernel_size=10, activation='relu')(input)
    cnn = keras.layers.Conv1D(filters=22, kernel_size=10, activation='relu')(cnn)
    cnn = keras.layers.Flatten()(cnn)

    dense = keras.layers.Dense(500, kernel_initializer='normal', activation='relu')(cnn)
    for i in range(5):
        dense = keras.layers.Dense(500, kernel_initializer='normal', activation='relu')(dense)
    output = keras.layers.Dense(
        10,
        kernel_initializer='normal',
        activation="linear",
        batch_size=batch_size,
        # activity_regularizer=keras.regularizers.l1(0.01),
    )(dense)

    if os.path.exists(os.path.join(output_dir, model_file)):
        # load pre-existing model
        if load:
            model = loadModel(output_dir, model_file, weights_file)
            # rename copy to not override
            i = 1
            model_file = f"{model_title}({i}).json"
            weights_file = f"{model_title}({i}).h5"
            while os.path.exists(os.path.join(output_dir, model_file)):
                i += 1
                model_file = f"{model_title}({i}).json"
                weights_file = f"{model_title}({i}).h5"
        else:
            model = keras.models.Model(inputs=input, outputs=output)
            # raise Exception('Model file already exists! Stopping so it won\'t be overrided')
    else:
        # Tie inputs together
        model = keras.models.Model(inputs=input, outputs=output)
    # model.summary()
    # raise Exception('hey')

    def loss_fn(y_true, y_pred):
        if fit_param:
            weights = tf.one_hot(fit_param, 10, dtype=tf.float32)
        else:
            weights = tf.constant([0, 0, 0, 1, 1, 0, 0, 0, 0, 0], dtype=tf.float32)

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
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[ReportFitGoodness(validation_gen), model_checkpoint_callback, early_stopping_callback],
        workers=0,
        max_queue_size=1,
    )

    # serialize weights to HDF5
    model.save_weights(weights_file)
    print("Saved model to disk")

    # Evaluate fit quality
    percent_norm_errors, denorm_perc_errors_av, param_corr = evaluateFit(
        model, iter(validation_gen), output=False
    )

    new_file = not os.path.exists(os.path.join(output_dir, 'fitlog.csv'))
    with open(os.path.join(output_dir, 'fitlog.csv'), 'a+') as f:
        data = {
            'Date': datetime.datetime.now(),
            'model name': model_title,
            'dataset': extra_information['cml_args'].data_folder,
            'model parameters': model.count_params(),
            'batch size': batch_size,
            'step per epoch': steps_per_epoch,
            'epochs': epochs,
            'training data': extra_information['length'],
            'final loss': history.history['loss'][-1],
            'percent normalized errors': list(percent_norm_errors),
            'denormalized percent errors': list(denorm_perc_errors_av),
            'parameter correlation': list(param_corr),
        }

        if new_file:
            f.write('|'.join([str(x) for x in data.keys()]) + '\n')

        f.write('|'.join([str(x) for x in data.values()]) + '\n')


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
        help='0 evaluates based on parameters. 1 evaluates based on simulated data. 2 Evaluates based on '
        'simulated data but acts on a single meteor and will show extra plots.',
    )
    arg_parser.add_argument(
        '--grouping',
        type=int,
        nargs=3,
        help='Allows for specifying the batch size, steps per epoch and epochs, respectively.',
    )
    arg_parser.add_argument(
        '--load',
        action='store_true',
        help='Whether  to allow a pre-existing model to be loaded if it exists, and train it.',
    )
    arg_parser.add_argument(
        '--logplot',
        type=int,
        nargs=10,
        help='Providing 1 will specify whether a plot should be a log plot, 0 otherwise. Must be given '
        'when in evaluation mode.',
    )
    arg_parser.add_argument(
        '--pickle',
        action='store_true',
        help='Whether or not to load a dataset which contains pickled simulations',
    )
    arg_parser.add_argument('--noerosion', action='store_true')
    arg_parser.add_argument('--fitparam', type=int)
    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    ### INPUTS ###
    if cml_args.grouping:
        batch_size, steps_per_epoch, epochs = cml_args.grouping
    else:
        batch_size = 500
        steps_per_epoch = 300
        epochs = 50

    # Model file names
    model_file = f"{cml_args.modelname}.json"
    weights_file = f"{cml_args.modelname}.h5"

    ### ###
    # load data
    if cml_args.pickle:
        # Load the list of files from the given input file and randomly drawn limiting magnitude and length
        #   measurement delay used to generate the data
        data_list = []
        data_list = getFileList(cml_args.data_folder)

        # Init the data generator
        data_gen = DataGenerator(
            data_list, batch_size, steps_per_epoch, param_class_name=cml_args.classname, validation=False
        )

        # Init the validation generator
        validation_gen = DataGenerator(
            data_list, batch_size, steps_per_epoch, param_class_name=cml_args.classname, validation=True
        )

        # randomize the order of the dataset to remove bias
        length = len(data_list)

    else:
        data_gen = loadProcessedData(cml_args.data_folder, batch_size, validation_split=0.2)
        validation_gen = loadProcessedData(
            cml_args.data_folder, batch_size, validation_split=0.2, validation=True
        )
        length = getProcessedDataLength(cml_args.data_folder)

    print("{:d} inputs used for training/testing...".format(length))

    # train neural network or evaluate it
    if cml_args.evaluate is not None:
        data_gen = loadProcessedData(cml_args.data_folder, batch_size, validation_split=0)
        model = loadModel(cml_args.output_dir, model_file, weights_file)

        if cml_args.evaluate == 0:
            evaluateFit(model, iter(data_gen), display=True, output=True, log=cml_args.logplot)
        else:
            evaluateFit2(model, iter(data_gen), mode=cml_args.evaluate, noerosion=cml_args.noerosion)
    else:
        # Fit the model
        # testFitModels(data_gen, validation_gen, batch_size, steps_per_epoch, epochs)
        # fitForwardProblem(
        #     data_gen,
        #     validation_gen,
        #     cml_args.output_dir,
        #     model_file,
        #     weights_file,
        #     steps_per_epoch=steps_per_epoch,
        #     epochs=epochs,
        #     batch_size=batch_size,
        # )
        fitCurveInverse(
            data_gen,
            validation_gen,
            cml_args.output_dir,
            model_file,
            weights_file,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            batch_size=batch_size,
        )
        # fitInverseProblem(
        #     data_gen,
        #     validation_gen,
        #     cml_args.output_dir,
        #     model_file,
        #     weights_file,
        #     batch_size=batch_size,
        #     steps_per_epoch=steps_per_epoch,
        #     epochs=epochs,
        #     fit_param=cml_args.fitparam,
        #     load=cml_args.load,
        #     extra_information={'cml_args': cml_args, 'length': length},
        # )

