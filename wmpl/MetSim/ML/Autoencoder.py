import argparse
import os
import sys
import time
from functools import partial
from typing import Optional

import keras
import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
import tensorflow as tf
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtGui import QTransform
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.uic import loadUi
from pyqtgraph import PlotWidget, plot
from sklearn.decomposition import PCA
from wmpl.MetSim.GUITools import MatplotlibPopupWindow
from wmpl.MetSim.ML.FitErosion import correlationPlot
from wmpl.MetSim.ML.GenerateSimulations import DATA_LENGTH, PhysicalParameters
from wmpl.MetSim.ML.PostprocessSims import loadh5pyData


class ClickableImageItem(pg.ImageItem):
    sigClicked = pyqtSignal(object, object)
    sigHover = pyqtSignal(object, object)
    sigMouseExit = pyqtSignal(object, object)

    def __init__(self, image=None, **kargs):
        super().__init__(image, **kargs)

        self.mouse_in = False

    def mouseClickEvent(self, event):
        self.sigClicked.emit(self, event)

    def hoverEvent(self, event):
        if event.isEnter():
            self.mouse_in = True
        elif event.isExit():
            self.mouse_in = False
            self.sigMouseExit.emit(self, event)

        if self.mouse_in:
            self.sigHover.emit(self, event)


class LatentDistanceGUI(QMainWindow):
    def __init__(self, model):
        QMainWindow.__init__(self)
        loadUi(os.path.join(os.path.dirname(__file__), "LatentDistanceGUI.ui"), self)
        self.setWindowTitle("LatentDistance")

        ## variables used for computation ##
        self.param_obj = PhysicalParameters()
        self.model = model
        self.params = np.zeros((10,), dtype=np.float64)
        self.variable_params = np.full((10,), False, dtype=bool)
        self.variable_params[3:5] = True
        self.latent_space = np.zeros((self.n * self.n, 20), dtype=np.float64)
        self.parameter_names = np.array(
            ["M0", "V0", "ZC", "DENS", "ABL", "ERHT", "ERCO", "ER_S", "ERMm", "ERMM"]
        )
        self.parameter_units = np.array(
            ['kg', 'km/s', 'deg', 'kg/m3', 'kg/MJ', 'km', 'kg/MJ', '', 'kg', 'kg']
        )
        self.parameter_scaling = np.array([1, 1 / 1000, 180 / np.pi, 1, 1e6, 1 / 1000, 1e6, 1, 1, 1])

        # grid coordinates
        self.n = 100
        self.coords = (
            np.concatenate(np.meshgrid(np.linspace(0, 1, self.n), np.linspace(0, 1, self.n)))
            .reshape(2, self.n * self.n)
            .T[:, ::-1]
        )  # (n*n, 2)
        self.current_index = 0
        self.set_index = 0

        ## informative label ##
        self.computation_enabled = False
        self.ComputationEnabledLabel.setText('Recomputation Enabled')
        self.ComputationEnabledLabel.setStyleSheet("color: #006325")

        ## image plot ##
        # self.graphWidget.setLimits(xMin=0, xMax=1, yMin=0, yMax=1)
        self.graphWidget.setMouseEnabled(x=False, y=False)

        cm = pg.ColorMap(
            [0, 0.03, 0.1, 0.2, 1], [(0, 0, 0), (255, 255, 0), (255, 0, 0), (255, 255, 255), (0, 0, 255)]
        )
        self.image = ClickableImageItem(np.eye(self.n), levels=(0, 3))
        self.CoordinatesLabel.setText('(0, 0)')
        self.setImageTransform()
        self.image.setColorMap(cm)
        self.graphWidget.addItem(self.image)
        self.graphWidget.addColorBar(self.image, colorMap=cm, interactive=False)

        # labels
        text = self.parameter_names[self.variable_params]
        units = self.parameter_units[self.variable_params]
        self.graphWidget.setLabel('bottom', text=f'{text[0]} ({units[0]})')
        self.graphWidget.setLabel('left', text=f'{text[1]} ({units[1]})')

        # signals
        self.image.sigHover.connect(self.onMoved)
        self.image.sigClicked.connect(self.onClicked)
        self.image.sigMouseExit.connect(self.computeDistances)

        ## Parameter inputs ##
        self.parameter_checkboxes = [
            self.InitialMassCheckbox,
            self.InitialVelocityCheckbox,
            self.ZenithCheckbox,
            self.DensityCheckbox,
            self.AblationCoefficientCheckbox,
            self.ErosionHeightCheckbox,
            self.ErosionCoefficientCheckbox,
            self.MassIndexCheckbox,
            self.ErosionMassMinCheckbox,
            self.ErosionMassMaxCheckbox,
        ]
        self.parameter_inputs = [
            self.InitialMassInput,
            self.InitialVelocityInput,
            self.ZenithInput,
            self.DensityInput,
            self.AblationCoefficientInput,
            self.ErosionHeightInput,
            self.ErosionCoefficientInput,
            self.MassIndexInput,
            self.ErosionMassMinInput,
            self.ErosionMassMaxInput,
        ]
        self.parameter_values = [
            self.InitialMassValue,
            self.InitialVelocityValue,
            self.ZenithValue,
            self.DensityValue,
            self.AblationCoefficientValue,
            self.ErosionHeightValue,
            self.ErosionCoefficientValue,
            self.MassIndexValue,
            self.ErosionMassMinValue,
            self.ErosionMassMaxValue,
        ]

        for i, (checkbox, input) in enumerate(zip(self.parameter_checkboxes, self.parameter_inputs)):
            checkbox.stateChanged.connect(partial(self.checkboxStateChange, i))
            checkbox.setChecked(self.variable_params[i])

            input.setText(str(self.params[i]))
            input.editingFinished.connect(partial(self.entryChanged, i))

        self.updateParameterValue()

        # after initialization, allow for user interaction
        self.computation_enabled = True
        self.computeLatentSpace()

    ## signal functions ##
    def checkboxStateChange(self, i):
        checkbox = self.parameter_checkboxes[i]
        input = self.parameter_inputs[i]

        checked = checkbox.checkState()
        self.variable_params[i] = checked > 0
        input.setDisabled(checked)

        if np.sum([checkbox.checkState() > 0 for checkbox in self.parameter_checkboxes]) != 2:
            self.computation_enabled = False
            self.ComputationEnabledLabel.setText('Recomputation Disabled')
            self.ComputationEnabledLabel.setStyleSheet("color: #b40000")
        else:
            self.computation_enabled = True
            self.ComputationEnabledLabel.setText('Recomputation Enabled')
            self.ComputationEnabledLabel.setStyleSheet("color: #006325")

            text = self.parameter_names[self.variable_params]
            units = self.parameter_units[self.variable_params]
            self.graphWidget.setLabel('bottom', text=f'{text[0]} ({units[0]})')
            self.graphWidget.setLabel('left', text=f'{text[1]} ({units[1]})')

        if self.computation_enabled:
            self.computeLatentSpace()
            self.setImageTransform()

    def setImageTransform(self):
        xy_range = [[], []]

        for i, (param, scaling) in enumerate(
            zip(
                np.array(self.param_obj.param_list)[self.variable_params],
                self.parameter_scaling[self.variable_params],
            )
        ):
            xy_range[i] = [
                getattr(self.param_obj, param).min * scaling,
                getattr(self.param_obj, param).max * scaling,
            ]

        self.transform = QTransform()
        self.transform.translate(xy_range[0][0], xy_range[1][0])
        self.transform.scale(
            1 / self.n * (xy_range[0][1] - xy_range[0][0]), 1 / self.n * (xy_range[1][1] - xy_range[1][0])
        )

        self.image.setTransform(self.transform)

        self.graphWidget.setRange(xRange=xy_range[0], yRange=xy_range[1], padding=0)

    def entryChanged(self, i):
        changed = False
        try:
            new_value = float(self.parameter_inputs[i].text())
            changed = new_value != self.params[i]
            self.params[i] = new_value
        except ValueError:
            self.parameter_inputs[i].setText(str(self.params[i]))

        if changed:
            self.computeLatentSpace()
            self.updateParameterValue(i)

    def onClicked(self, image, ev):
        pos = np.array([ev.pos().x(), ev.pos().y()])
        self.set_index = np.ravel_multi_index(np.floor(pos).astype(int), (self.n, self.n))
        # no need to compute distance

    def onMoved(self, image, ev):
        pos = np.array([ev.pos().x(), ev.pos().y()])
        self.current_index = np.ravel_multi_index(np.floor(pos).astype(int), (self.n, self.n))
        self.computeDistances()

    ## other functions ##
    def computeLatentSpace(self):
        """
        When the fit parameters are changed in any way, the latent space must be recomputed
        """
        if not self.computation_enabled:
            return

        input_values = np.stack((self.params,) * self.n * self.n, axis=0)  # (n*n, 10)
        input_values[:, self.variable_params] = self.coords
        self.latent_space = self.model.predict(input_values)

        self.computeDistances()

    def computeDistances(self):
        """
        When latent space or current index is updated, this should be called
        """
        if self.image.mouse_in:
            i = self.current_index
        else:
            i = self.set_index

        clicked_coord = self.latent_space[i : i + 1]  # keep dim of (1, n*n)
        dist = np.sqrt(np.sum((self.latent_space - clicked_coord) ** 2, axis=1))
        self.image.setImage(dist.reshape(self.n, self.n), levels=(0, 3))

        transformed_pos = pg.transformCoordinates(self.transform, self.coords[i] * self.n)
        self.CoordinatesLabel.setText(f'({transformed_pos[0]:.4e}, {transformed_pos[1]:.4e})')

    def updateParameterValue(self, i: Optional[int] = None):
        values = self.param_obj.getDenormalizedInputs(self.params)
        if i is None:
            for i, value in enumerate(values):
                self.parameter_values[i].setText(
                    f'{value*self.parameter_scaling[i]:.4e} {self.parameter_units[i]}'
                )
        else:
            self.parameter_values[i].setText(
                f'{values[i]*self.parameter_scaling[i]:.4e} {self.parameter_units[i]}'
            )


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
    latent_dim = 20
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
    keras.models.save_model(autoencoder, os.path.join(output_dir, model_name + '_encoder'), save_format='tf')


def trainParamPrediction(
    data_path: str,
    output_dir: str,
    autoencoder_path: str,
    epochs: int,
    steps_per_epoch: int,
    batchsize: int,
    model_name: str = 'model',
):
    autoencoder = keras.models.load_model(autoencoder_path)

    input_train, label_train = loadh5pyData(data_path)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(20,)),
            keras.layers.Dense(50, activation='relu'),
            keras.layers.Dense(50, activation='relu'),
            keras.layers.Dense(50, activation='relu'),
            keras.layers.Dense(50, activation='relu'),
            keras.layers.Dense(50, activation='relu'),
            keras.layers.Dense(10),
        ]
    )

    model.compile(optimizer='adam', loss='mse')
    model.fit(
        x=autoencoder.encoder.predict(input_train),
        y=label_train,
        batch_size=batchsize,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_split=0.2,
    )

    keras.models.save_model(
        model, os.path.join(output_dir, model_name + '_encoder_translator.hdf5'), save_format='h5'
    )


def trainLatentSpaceFinder(
    data_path: str,
    output_dir: str,
    autoencoder_path: str,
    epochs: int,
    steps_per_epoch: int,
    batchsize: int,
    model_name: str = 'model',
):

    """
    Solves the inverse problem to the param prediction model, it transforms physical parameters to parameters
    in the pca latent space
    """
    autoencoder = keras.models.load_model(autoencoder_path)

    input_train, label_train = loadh5pyData(data_path)

    pca = PCA(n_components=20)
    pca.fit(autoencoder.encoder.predict(input_train[:5000]))

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(10,)),
            keras.layers.Dense(50, activation='relu'),
            keras.layers.Dense(50, activation='relu'),
            keras.layers.Dense(50, activation='relu'),
            keras.layers.Dense(50, activation='relu'),
            keras.layers.Dense(50, activation='relu'),
            keras.layers.Dense(20),
        ]
    )

    model.compile(optimizer='adam', loss='mse')
    model.fit(
        x=label_train,
        y=pca.transform(autoencoder.encoder.predict(input_train)),
        batch_size=batchsize,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_split=0.2,
    )

    keras.models.save_model(
        model, os.path.join(output_dir, model_name + '_latentspace_finder.hdf5'), save_format='h5'
    )


def visualizeLatentSpace(autoencoder_path: str, data_path: str):
    """
    Displays an interactive visualization where you can use sliders to adjust the pca latent space
    to understand what the pca latent space respresents
    
    Arguments:
        autoencoder_path: [str] Path to autoencoder model file
        data_path: [str] path to .hdf5 dataset
    
    """
    autoencoder = keras.models.load_model(autoencoder_path)
    input_train, label_train = loadh5pyData(data_path)

    latent_space = autoencoder.encoder.predict(input_train[:5000])
    pca = PCA(n_components=20)
    pca.fit(latent_space)

    pca_init = np.mean(pca.transform(latent_space), axis=0)
    pca_range = np.std(pca.transform(latent_space), axis=0)
    sliders = [
        Slider(
            ax=plt.axes([0.25, 0.95 - i * 0.05, 0.65, 0.03]),
            label=f'{pca.explained_variance_ratio_[i]:.3f}',
            valmin=-std,
            valmax=std,
            valinit=val,
        )
        for i, (val, std) in enumerate(zip(pca_init, pca_range))
    ]

    input_data = np.array([slider.val for slider in sliders])[None]
    data = autoencoder.decoder.predict(pca.inverse_transform(input_data))[0]

    fig, ax = plt.subplots(2)
    (line1,) = ax[0].plot(data[:, 2])
    (line2,) = ax[1].plot(data[:, 1])
    ax[0].set_ylabel('Normalized Magnitude')
    ax[1].set_ylabel('Normalized length')

    def update(*args):
        input_data = np.array([slider.val for slider in sliders])[None]
        data = autoencoder.decoder.predict(pca.inverse_transform(input_data))[0]

        # line1.set_ydata(data[:, 0])
        line1.set_ydata(data[:, 2])
        # line2.set_ydata(data[:, 0])
        line2.set_ydata(data[:, 1])
        fig.canvas.draw_idle()

    for slider in sliders:
        slider.on_changed(update)

    plt.show()


def visualizeInverseSolving(
    autoencoder_path: str, param_model_path: str, data_path: str, forward: bool = False
):
    """
    Visualize how effectively autoencoder can solve the inverse problem
    """
    autoencoder = keras.models.load_model(autoencoder_path)
    model = keras.models.load_model(param_model_path)
    input_train, label_train = loadh5pyData(data_path)

    if forward:
        pca = PCA(n_components=20)
        pca.fit(autoencoder.encoder.predict(input_train[:5000]))

        correlationPlot(
            model.predict(label_train),
            pca.transform(autoencoder.encoder.predict(input_train)),
            [False] * 10,
            [''] * 10,
            [''] * 10,
            ['', ''],
        )
    else:
        correlationPlot(
            label_train,
            model.predict(autoencoder.encoder.predict(input_train)),
            [False] * 10,
            [''] * 10,
            [''] * 10,
            ['', ''],
        )


def visualizeLatentSpaceDistance(model_path):
    model = keras.models.load_model(model_path)

    app = QApplication([])
    main_window = LatentDistanceGUI(model)
    main_window.show()
    sys.exit(app.exec_())


def main():
    data_path = r'D:\datasets\meteor\norestrictions_dataset.h5'
    model_path = r'D:\datasets\meteor\saturn\trained_models\trained2'
    model_name = 'inverseproblem'
    autoencoder_path = rf'D:\datasets\meteor\saturn\trained_models\trained2\model_encoder'
    translator_path = (
        rf'D:\datasets\meteor\saturn\trained_models\trained2\{model_name}_encoder_translator.hdf5'
    )
    latentspacefinder_path = (
        rf'D:\datasets\meteor\saturn\trained_models\trained2\{model_name}_latentspace_finder.hdf5'
    )

    # trainAutoencoder(data_path, model_path, 30, 50, 500, model_name=model_name)

    # trainParamPrediction(data_path, model_path, autoencoder_path, 30, 300, 500, model_name=model_name)
    # trainLatentSpaceFinder(data_path, model_path, autoencoder_path, 30, 300, 500, model_name=model_name)

    # visualizeLatentSpace(autoencoder_path, data_path)

    # visualizeInverseSolving(autoencoder_path, latentspacefinder_path, data_path, forward=True)

    visualizeLatentSpaceDistance(latentspacefinder_path)


if __name__ == '__main__':
    main()
