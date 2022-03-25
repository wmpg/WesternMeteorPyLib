import keras
from wmpl.MetSim.ML.FitErosion import loadModel
from wmpl.MetSim.ML.GenerateSimulations import DATA_LENGTH


class Autoencoder(keras.models.Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = keras.Sequential(
            [
                keras.layers.Input(shape=(DATA_LENGTH, 3)),
                keras.layers.Conv1D(16, 20, activation='relu', padding='same', strides=2),
                keras.layers.Conv1D(8, 20, activation='relu', padding='same', strides=2),
                keras.layers.Flatten(),
                keras.layers.Dense(100),
                keras.layers.Dense(latent_dim),
            ]
        )

        self.decoder = keras.Sequential(
            [
                keras.layers.Dense(100),
                keras.layers.Conv1DTranspose(8, kernel_size=20, strides=2, activation='relu', padding='same'),
                keras.layers.Conv1DTranspose(
                    16, kernel_size=20, strides=2, activation='relu', padding='same'
                ),
                keras.layers.Conv1D(1, kernel_size=20, activation='sigmoid', padding='same'),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def trainAutoencoder():
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor='loss', patience=5, min_delta=0, verbose=1
    )

    autoencoder = Autoencoder(20)
    autoencoder.compile(optimizer='adam', loss=keras.losses.MeanSquaredError())

    autoencoder.fit()
    autoencoder.decoder.summary()


def main():
    trainAutoencoder()


if __name__ == '__main__':
    main()
