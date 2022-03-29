import argparse
import os
import time

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from wmpl.MetSim.ML.FitErosion import loadModel
from wmpl.MetSim.ML.GenerateSimulations import DATA_LENGTH
from wmpl.MetSim.ML.PostprocessSims import loadh5pyData


class Autoencoder(keras.models.Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = keras.Sequential(
            [
                keras.layers.Input(shape=(DATA_LENGTH, 3)),
                keras.layers.Conv1D(16, 20, activation='relu', padding='same'),
                keras.layers.MaxPooling1D(5, padding='same'),
                keras.layers.Conv1D(8, 20, activation='relu', padding='same'),
                keras.layers.MaxPooling1D(5, padding='same'),
                keras.layers.Flatten(),
                keras.layers.Dense(100),
                keras.layers.Dense(latent_dim),
            ]
        )

        self.decoder = keras.Sequential(
            [
                keras.layers.Dense(DATA_LENGTH),
                keras.layers.Reshape((DATA_LENGTH, 1)),
                keras.layers.Conv1D(16, 20, activation='relu', padding='same'),
                keras.layers.Conv1D(16, 20, activation='relu', padding='same'),
                keras.layers.Conv1D(3, 20, padding='same'),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def trainAutoencoder(
    data_path: str,
    output_dir: str,
    epochs: int,
    steps_per_epoch: int,
    batchsize: int,
    model_name: str = 'model',
):
    # early_stopping_callback = keras.callbacks.EarlyStopping(
    #     monitor='loss', patience=5, min_delta=0, verbose=1
    # )
    latent_dim = 10
    autoencoder = Autoencoder(latent_dim)
    autoencoder.compile(optimizer='adam', loss=keras.losses.MeanSquaredError())

    input_train, label_train = loadh5pyData(data_path)

    history = autoencoder.fit(
        x=input_train,
        y=input_train,
        batch_size=batchsize,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_split=0.2,
    )

    pca = PCA(n_components=latent_dim)
    pca.fit(autoencoder.encoder.predict(input_train[:5000]))
    print(list(pca.explained_variance_ratio_ * 100))

    # save model
    # model_json = autoencoder.to_json()
    # model_file = os.path.join(output_dir, f'{model_name}.h5')
    # weights_file = os.path.join(output_dir, f'{model_name}.json')
    # with open(model_file, "w") as json_file:
    #     json_file.write(model_json)
    # autoencoder.save_weights(weights_file)
    keras.models.save_model(
        autoencoder, os.path.join(output_dir, model_name + '_encoder.tf'), save_format='tf'
    )


def trainParamPrediction(
    data_path: str,
    output_dir: str,
    model_path: str,
    epochs: int,
    steps_per_epoch: int,
    batchsize: int,
    model_name: str = 'model',
):
    encoder = keras.models.load_model(model_path)

    input_train, label_train = loadh5pyData(data_path)
    print(np.mean(label_train, axis=0))
    raise Exception('hey')
    model2 = keras.Sequential(
        [
            keras.layers.Input(shape=(10,)),
            keras.layers.Dense(50, activation='relu'),
            keras.layers.Dense(50, activation='relu'),
            keras.layers.Dense(50, activation='relu'),
            keras.layers.Dense(10),
        ]
    )

    model2.compile(optimizer='adam', loss='mse')
    model2.fit(
        x=encoder.predict(input_train),
        y=label_train,
        batch_size=batchsize,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_split=0.2,
    )

    keras.models.save_model(
        model2, os.path.join(output_dir, model_name + '_encoder_translator.hdf5'), save_format='h5'
    )


def main():
    # trainAutoencoder(
    #     r'D:\datasets\meteor\norestrictions_dataset.h5',
    #     r'D:\datasets\meteor\saturn\trained_models\trained2',
    #     30,
    #     50,
    #     500,
    # )

    trainParamPrediction(
        r'D:\datasets\meteor\noerosion_dataset.h5',
        r'D:\datasets\meteor\saturn\trained_models\trained2',
        r'D:\datasets\meteor\saturn\trained_models\trained2\model_encoder.hdf5',
        30,
        50,
        500,
    )


if __name__ == '__main__':
    main()
