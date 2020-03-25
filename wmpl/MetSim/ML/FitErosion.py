""" Fit the erosion model using machine learning. """

from __future__ import print_function, division, absolute_import, unicode_literals

import os
import copy

import numpy as np
import keras


from wmpl.Utils.Pickling import loadPickle
from wmpl.Utils.PyDomainParallelizer import domainParallelizer
from wmpl.MetSim.ML.GenerateSimulations import DATA_LENGTH, MetParam, ErosionSimContainer, \
    ErosionSimParametersCAMO, extractSimData



def dataFunction(data_path, file_name, postprocess_params):

    # Load the pickle file
    sim = loadPickle(data_path, file_name)

    # Extract model inputs and outputs
    return extractSimData(sim, postprocess_params=postprocess_params)


class DataGenerator(object):
    def __init__(self, data_path, data_list, batch_size, steps_per_epoch, validation=False, \
            validation_portion=0.2):
        """ Generate meteor data for the ML fit function. 
    
        Arguments:
            data_path: [str] Path to the data files.
            data_list: [list] A list of files that will be used as inputs.
            batch_size: [int] Number of inputs in every step.
            steps_per_epoch: [int] Number of steps in every epoch (iteration) with batch_size inputs each.

        Keyword arguments:
            validation: [bool] Generate validation data. False by default.
            validation_portion: [float] Portion of input files to be used for validation.
        """


        self.data_path = data_path
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.data_list = data_list

        self.validation = validation

        # Compute the number of files in each epoch
        self.data_per_epoch = self.batch_size*self.steps_per_epoch

        # Cut the processing list to the steps multiplier
        self.data_list = self.data_list[:-int(len(self.data_list)%self.data_per_epoch)]

        # Compute the number of total epochs
        total_epochs = int(len(self.data_list)//self.data_per_epoch)

        # Compute the number of epochs for the fit
        self.fit_epochs = int((1.0 - validation_portion)*total_epochs)

        # Number of validation epochs
        self.validation_epochs = total_epochs - self.fit_epochs

        # Make sure that there is a minimum of one validation epoch
        if self.validation_epochs < 1:
            self.validation_epochs = 1
            self.fit_epochs -= 1


        # Current data index
        self.data_index_start = 0

        # Current validation index
        self.validation_index_start = self.fit_epochs*self.data_per_epoch - 1


    def __iter__(self):

        # Select valirable depending on if the validation data is used or not
        if self.validation:
            curr_index = self.validation_index_start
            epochs = self.validation_epochs

        else:
            curr_index = self.data_index_start
            epochs = self.fit_epochs


        # Generate data for every epoch
        for step in range(epochs*self.steps_per_epoch):

            param_list = []
            result_list = []

            # Get a portion of files to load
            beg_index = curr_index
            file_list = self.data_list[beg_index:(beg_index + self.batch_size)]

            # Load pickle files and postprocess in parallel
            domain = []
            for entry in file_list:

                file_name, postprocess_params = entry

                domain.append([self.data_path, file_name, postprocess_params])


            # Postprocess the data in parallel
            res_list = domainParallelizer(domain, dataFunction)

            # Postprocess results
            filtered_res = []
            for res in res_list:

                curr_index += 1

                # Skip simulation which did not satisfy filters
                if res is None:
                    print("Skipped:", file_name)
                    continue

                filtered_res.append(res)

            res_list = filtered_res


            # Load more results using one core until the proper length is achieved
            while len(res_list) < self.batch_size:

                file_name, postprocess_params = self.data_list[curr_index]

                # Extract model inputs and outputs from the pickle file
                res = dataFunction(self.data_path, file_name, postprocess_params)

                curr_index += 1

                # Skip simulation which did not satisfy filters
                if res is None:
                    print("Skipped:", file_name)
                    continue

                res_list.append(res)


            # Split results to input/output list
            for res in res_list:

                # Extract results
                model_params, input_data_normed, simulated_data_normed = res

                # Add data to model input and output lists
                param_list.append(input_data_normed)
                result_list.append(simulated_data_normed)


            # # Skip simulation which did not satisfy filters
            #     if res is None:
            #         print("Skipped:", file_name)
            #         file_list.append(self.data_list[beg_index + self.batch_size + skipped_files + 1])
            #         skipped_files += 1
            #         continue


            param_list = np.array(param_list)
            result_list = np.array(result_list)


            height_data_normed_list, mag_data_normed_list, \
                length_data_normed_list = np.split(result_list, 3, axis=1)

            yield [height_data_normed_list.reshape(-1, model_params.data_length, 1), \
                    length_data_normed_list.reshape(-1, model_params.data_length, 1), \
                    mag_data_normed_list.reshape(-1, model_params.data_length, 1)], param_list





class ReportFitGoodness(keras.callbacks.Callback):
  """ Report the fit goodness at the every epoch end.
  """

  def __init__(self, validation_gen):
    self.validation_gen = validation_gen

  def on_epoch_end(self, epoch, logs=None):

    # Evaluate model accuracy using validation data
    percent_errors = evaluteFit(self.model, self.validation_gen, ret_perc=True)

    print()
    print("Epoch {:d} errors".format(epoch + 1))
    param_name_list = ["M0", "V0", "ZC", "DENS", "ABL", "ERHT", "ERCO", "ER_S", "ERMm", "ERMM"]
    print(" ".join(["{:>6s}".format(param_name) for param_name in param_name_list]))
    print(str(len(percent_errors)*"{:5.2f}% ").format(*percent_errors))

    pass




def loadModel(model_file, weights_file):

    
    with open('model.json', 'r') as json_file:

        # load json and create model    
        loaded_model_json = json_file.read()
        loaded_model = keras.models.model_from_json(loaded_model_json)
        
        # load weights into new model
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")

        return loaded_model




def evaluteFit(model, validation_gen, ret_perc=False):

    # Create a copy of the generator
    validation_gen = copy.deepcopy(validation_gen)

    # Generate test data
    test_data = next(iter(validation_gen))
    test_outputs, test_inputs = test_data

    # Predict data
    probabilities = model.predict(test_outputs)

    # Compute mean absolute percentage error for every model parameter
    percent_errors = 100*np.mean(np.abs(probabilities - test_inputs), axis=0)

    if ret_perc:
        return percent_errors

    else:

        print("Mean absolute percentage error per parameter:")
        param_name_list = ["M0", "V0", "ZC", "DENS", "ABL", "ERHT", "ERCO", "ER_S", "ERMm", "ERMM"]
        print(" ".join(["{:>6s}".format(param_name) for param_name in param_name_list]))
        print(str(len(percent_errors)*"{:5.2f}% ").format(*percent_errors))



def fitCNNMultiHeaded(data_gen, validation_gen, output_dir, model_file, weights_file):


    # Height input model
    visible1 = keras.engine.input_layer.Input(shape=(DATA_LENGTH, 1))
    cnn1 = keras.layers.Conv1D(filters=32, kernel_size=4, activation='relu')(visible1)
    cnn1 = keras.layers.MaxPooling1D(pool_size=2)(cnn1)
    cnn1 = keras.layers.Flatten()(cnn1)
    cnn1 = keras.layers.Dense(128, kernel_initializer='normal', activation='relu')(cnn1)

    # Length input model
    visible2 = keras.engine.input_layer.Input(shape=(DATA_LENGTH, 1))
    cnn2 = keras.layers.Conv1D(filters=32, kernel_size=4, activation='relu')(visible2)
    cnn2 = keras.layers.MaxPooling1D(pool_size=2)(cnn2)
    cnn2 = keras.layers.Flatten()(cnn2)
    cnn2 = keras.layers.Dense(128, kernel_initializer='normal', activation='relu')(cnn2)

    # Magnitude input model
    visible3 = keras.engine.input_layer.Input(shape=(DATA_LENGTH, 1))
    cnn3 = keras.layers.Conv1D(filters=32, kernel_size=4, activation='relu')(visible3)
    cnn3 = keras.layers.MaxPooling1D(pool_size=2)(cnn3)
    cnn3 = keras.layers.Flatten()(cnn3)
    cnn3 = keras.layers.Dense(128, kernel_initializer='normal', activation='relu')(cnn3)


    # merge input models
    merge = keras.layers.merge.concatenate([cnn1, cnn2, cnn3])
    dense1 = keras.layers.Dense(128, kernel_initializer='normal', activation='relu')(merge)
    dense2 = keras.layers.Dense(64, kernel_initializer='normal', activation='relu')(dense1)
    output = keras.layers.Dense(10, kernel_initializer='normal', activation="linear", \
        batch_size=batch_size)(dense2)

    # Tie inputs together
    model = keras.models.Model(inputs=[visible1, visible2, visible3], outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='mse')


    # fit model
    model.fit_generator(generator=iter(data_gen), 
                        steps_per_epoch=data_gen.steps_per_epoch, 
                        epochs=data_gen.fit_epochs,
                        callbacks=[ReportFitGoodness(validation_gen)],
                        workers=0,
                        max_queue_size=1
                        )


    # Save the model to disk
    model_json = model.to_json()
    with open(model_file, "w") as json_file:
        json_file.write(model_json)
        
    # serialize weights to HDF5
    model.save_weights(weights_file)
    print("Saved model to disk")


    # Evaluate fit quality
    evaluteFit(model, validation_gen)





if __name__ == "__main__":


    import argparse


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Fit the ML model using the files listed in the given text file (file names only, one file per row).")

    arg_parser.add_argument('input_list_path', metavar='INPUT_LIST_PATH', type=str, \
        help="Path to file which holds the list of input files which should be in the same directory.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################


    ### INPUTS ###

    batch_size = 256

    steps_per_epoch = 50

    # Model file names
    model_file = "model.json"
    weights_file = "model.h5"

    ### ###


    # Extract directory path and the input file name
    dir_path, input_file = os.path.split(cml_args.input_list_path)

    # Load the list of files from the given input file and randomly drawn limiting magnitude and length 
    #   measurement delay used to generate the data
    data_list = []
    with open(cml_args.input_list_path) as f:
        for entry in f:

            # Skip comment lines
            if entry.startswith("#"):
                continue

            entry = entry.replace('\n', '').replace('\r', '')

            if not entry:
                continue

            # Split the data into the file name and postprocessing parameters
            file_name, lim_mag, lim_mag_length, len_delay = entry.split(",")
            lim_mag = float(lim_mag)
            lim_mag_length = float(lim_mag_length)
            len_delay = float(len_delay)

            file_path = os.path.join(dir_path, file_name)

            # Check if the given file exists
            if os.path.isfile(file_path):

                # Add the file to the processing list
                data_list.append([file_name, [lim_mag, lim_mag_length, len_delay]])



    print("{:d} inputs used for training...".format(len(data_list)))

    # Init the data generator
    data_gen = DataGenerator(dir_path, data_list, batch_size, steps_per_epoch, validation=False)

    # Init the validation generator
    validation_gen = DataGenerator(dir_path, data_list, batch_size, steps_per_epoch, validation=True)


    # ## TEST DATA GEN ###
    # for epoch in range(data_gen.fit_epochs):
    #     for step in range(data_gen.steps_per_epoch):
    #         print(epoch, "/", step)
    #         result_list, param_list = next(iter(data_gen))

    #         print(len(param_list))

    # Fit the model
    fitCNNMultiHeaded(data_gen, validation_gen, dir_path, model_file, weights_file)