""" Fit the erosion model using machine learning. """

from __future__ import print_function, division, absolute_import, unicode_literals

import os
import importlib


import numpy as np
import keras

from wmpl.Utils.Pickling import loadPickle



class DataGenerator(object):
    def __init__(self, data_path, batch_size, steps_per_epoch, validation_portion=0.2):
        """ Generate meteor data. """


        self.data_path = data_path
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch

        # Compute the number of files in each epoch
        self.data_per_epoch = self.batch_size*self.steps_per_epoch


        # Load the list of files from disk
        self.data_list = []
        for file_name in os.listdir(data_path):

            file_path = os.path.join(data_path, file_name)

            # Check if the given file is a pickle file
            if os.path.isfile(file_path) and file_name.endswith(".pickle"):

                # Add the pickle file to the processing list
                self.data_list.append(file_name)


        # Cut the processing list to the steps multiplier
        self.data_list = self.data_list[:-int(len(self.data_list)%self.data_per_epoch)]

        # Compute the number of total epochs
        total_epochs = int(len(self.data_list)//self.data_per_epoch)

        # Compute the number of epochs for the fit
        self.fit_epochs = int((1.0 - validation_portion)*total_epochs)

        # Number of validation epochs
        self.validation_epochs = total_epochs - self.fit_epochs


        # Current data index
        self.curr_index = 0

        # Current validation index
        self.curr_validation_index = self.fit_epochs*self.data_per_epoch - 1



    def generate(self):

        # Generate data for every epoch
        for epoch in self.epochs:


            param_list = []
            result_list = []

            pickle_counter = 0

            # Get a portion of files to load
            file_list = self.data_list[self.curr_index:(self.curr_index + batch_size)]

            # Load all pickle files in the data path directory
            for file_path in os.listdir(data_path):

                file_path = os.path.join(data_path, file_name)

                # Check if the given file is a pickle file
                if os.path.isfile(file_path) and file_name.endswith(".pickle"):

                    # Load the pickle file
                    sim = loadPickle(data_path, file_name)


                    pickle_counter += 1



                    # Add data to list


                    # Stop loading 




            for i in range(batch_size):

                # Load the simulation from disk

                # Generate a random duration
                duration = np.random.uniform(DURATION_MIN, DURATION_MAX)

                # Generate a random peak magnitude (allow 3 magnitude of room at top which will be variable by
                #   velocity and deceleration)
                peak_mag = np.random.uniform(MAG_BRIGHTEST + 3, MAG_FAINTEST)

                # Generate a random F number
                f_num = np.random.uniform(0.0, 1.0)

                # Generate a random deceleration
                decel = np.random.uniform(DECEL_MIN, DECEL_MAX)

                # Generate a random velocity
                velocity = np.random.uniform(V_MIN, V_MAX)

                # Generate a random zenith angle
                zangle = np.random.uniform(ZANGLE_MIN, ZANGLE_MAX)


                # Run the model to create outputs
                model_output = meteorModel(duration, peak_mag, f_num, decel, velocity, zangle)
                result_list.append(model_output)

                # Map inputs to to 0-1 range
                model_inputs = inputNormalize(duration, peak_mag, f_num, decel, velocity, zangle)
                param_list.append(model_inputs)


            param_list = np.array(param_list)
            result_list = np.array(result_list)

            if split_results:

                height_data_normed_list, mag_data_normed_list, \
                    length_data_normed_list = np.split(result_list, 3, axis=1)

                yield [height_data_normed_list.reshape(-1, DATA_LENGTH, 1), \
                        length_data_normed_list.reshape(-1, DATA_LENGTH, 1), \
                        mag_data_normed_list.reshape(-1, DATA_LENGTH, 1)], param_list

            else:
            
                result_list = result_list.reshape(batch_size, 3, DATA_LENGTH, 1)

                # Return a model run
                yield (result_list, param_list)