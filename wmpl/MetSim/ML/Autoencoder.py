import argparse
import os
import sys
import time
from functools import partial
from select import select
from tkinter import Y
from typing import Optional

import keras
import keras.backend as K
import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
import pyswarms as ps
import tensorflow as tf
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QTransform
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.uic import loadUi
from pyqtgraph import PlotWidget, plot
from scipy.stats import gaussian_kde
from scipy.stats.stats import pearsonr
from sklearn.decomposition import PCA
from wmpl.MetSim.GUITools import MatplotlibPopupWindow
from wmpl.MetSim.MetSimErosion import runSimulation
from wmpl.MetSim.ML.FitErosion import correlationPlot
from wmpl.MetSim.ML.GenerateSimulations import (
    DATA_LENGTH,
    ErosionSimParameters,
    PhysicalParameters,
    SimulationResults,
    extractSimData,
)
from wmpl.MetSim.ML.PostprocessSims import loadh5pyData, loadProcessedData


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

        self.sim_popup = None
        self.computation_enabled = False
        ## variables used for computation ##

        # grid coordinates
        self.n = 100
        self.coords = (
            np.concatenate(np.meshgrid(np.linspace(0, 1, self.n), np.linspace(0, 1, self.n)))
            .reshape(2, self.n * self.n)
            .T[:, ::-1]
        )  # (n*n, 2)

        self.param_obj = PhysicalParameters()
        self.pca_forward_model = model
        self.forward_model = keras.models.load_model(
            r'D:\datasets\meteor\trained_models\trained2\noersion_model_ls_forward.hdf5'
        )
        self.inverse_model = keras.models.load_model(
            r'D:\datasets\meteor\trained_models\trained2\noersion_model_encoder_translator.hdf5'
        )
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

        self.current_index = 0
        self.set_index = 0
        self.set_latent_space = self.latent_space[0]
        self.var_error_index = 0

        ## simulation mode ##
        self.SensitivityModeCheckbox.stateChanged.connect(self.recomputeDisplay)

        ## image plot ##
        # self.graphWidget.setLimits(xMin=0, xMax=1, yMin=0, yMax=1)
        self.graphWidget.setMouseEnabled(x=False, y=False)

        cm = pg.ColorMap(
            [0, 0.03, 0.1, 0.2, 1], [(255, 255, 255), (255, 0, 0), (255, 255, 0), (0, 255, 0), (0, 0, 255)]
        )
        self.cmap_max = 3
        self.image = ClickableImageItem(np.eye(self.n), levels=(0, self.cmap_max))
        self.CoordinatesLabel.setText('(0, 0)')
        self.CoordinatesLabel2.setText('')
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
        self.image.sigMouseExit.connect(
            lambda: self.computeDistances() if not self.getSensitivityMode() else None
        )

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

            input.setMinimum(0)
            input.setMaximum(100)
            # input.setTickInterval(50)
            # input.setValue(0)
            input.sliderReleased.connect(partial(self.sliderReleased, i))
            input.valueChanged.connect(partial(self.valueChanged, i))

        self.updateParameterValue()

        # after initialization, allow for user interaction
        self.computation_enabled = True
        self.computeLatentSpace()

    def getSimulationMode(self):
        return self.SimulationModeCheckbox.checkState() > 0

    def getSensitivityMode(self):
        return self.SensitivityModeCheckbox.checkState() > 0

    ## signal functions ##
    def recomputeDisplay(self):
        if self.getSensitivityMode():
            self.computeSensitivity()
        else:
            self.computeLatentSpace()

    def checkboxStateChange(self, i):
        checkbox = self.parameter_checkboxes[i]
        input = self.parameter_inputs[i]

        checked = checkbox.checkState()
        self.variable_params[i] = checked > 0
        input.setDisabled(checked)

        if np.sum([checkbox.checkState() > 0 for checkbox in self.parameter_checkboxes]) != 2:
            self.computation_enabled = False
            for i, checkbox in enumerate(self.parameter_checkboxes):
                if self.variable_params[i]:
                    checkbox.setStyleSheet("color: #b40000; font-weight: bold")
                else:
                    checkbox.setStyleSheet('')
        else:
            self.computation_enabled = True
            for checkbox in self.parameter_checkboxes:
                checkbox.setStyleSheet("")

            text = self.parameter_names[self.variable_params]
            units = self.parameter_units[self.variable_params]
            self.graphWidget.setLabel('bottom', text=f'{text[0]} ({units[0]})')
            self.graphWidget.setLabel('left', text=f'{text[1]} ({units[1]})')

        if self.computation_enabled:
            self.recomputeDisplay()
            self.setImageTransform()

    def sliderReleased(self, i):
        self.recomputeDisplay()

    def valueChanged(self, i):
        self.params[i] = self.parameter_inputs[i].value() / self.parameter_inputs[i].maximum()
        self.updateParameterValue(i)

    def onClicked(self, image, ev):
        pos = np.array([ev.pos().x(), ev.pos().y()])
        if not self.getSimulationMode() and not self.getSensitivityMode():
            self.set_index = np.ravel_multi_index(np.floor(pos).astype(int), (self.n, self.n))
            self.set_latent_space = self.latent_space[self.set_index]
        elif self.getSensitivityMode():
            self.var_error_index += 1
            self.var_error_index %= 5
            self.computeSensitivity()
        else:
            self.simulateData()
        # no need to compute distance

    def onMoved(self, image, ev):
        pos = np.array([ev.pos().x(), ev.pos().y()])
        self.current_index = np.ravel_multi_index(np.floor(pos).astype(int), (self.n, self.n))

        transformed_pos = pg.transformCoordinates(self.transform, pos)

        if not self.getSimulationMode():
            if not self.getSensitivityMode():
                self.computeDistances()
            self.CoordinatesLabel.setText(f'({transformed_pos[0]:.4e}, {transformed_pos[1]:.4e})')
            self.CoordinatesLabel2.setText('')
        else:
            self.CoordinatesLabel2.setText(f'({transformed_pos[0]:.4e}, {transformed_pos[1]:.4e})')

    ## other functions ##
    def computeSensitivity(self):
        # automatic differentiation
        # with tf.GradientTape() as tape:
        #     input_values = np.stack((self.params,) * self.n * self.n, axis=0)  # (n*n, 10)
        #     input_values[:, self.variable_params] = self.coords
        #     input_values = tf.convert_to_tensor(input_values)
        #     tape.watch(input_values)
        #     self.latent_space = self.pca_forward_model(input_values)
        #     jacobian = tape.batch_jacobian(self.latent_space, input_values)
        #     sensitivity = tf.sqrt(K.sum(jacobian ** 2, axis=(1, 2)))

        # self.image.setImage(sensitivity.numpy().reshape(self.n, self.n), levels=(0, 100))
        input_values = np.stack((self.params,) * self.n * self.n, axis=0)  # (n*n, 10)
        input_values[:, self.variable_params] = self.coords
        latent_space = self.forward_model.predict(input_values)
        output_values = self.inverse_model.predict(latent_space)
        dist = np.sqrt(((input_values - output_values) ** 2)[:, self.var_error_index])
        print(np.mean(dist), np.std(dist), np.min(dist), np.max(dist))
        self.image.setImage(dist.reshape(self.n, self.n), levels=(0, 0.5))

    def simulateData(self):
        values = self.params.copy()
        values[self.variable_params] = self.coords[self.current_index]
        # print(np.max(self.image.image.flatten()[self.current_index]))
        color = self.image.getColorMap().map(
            self.image.image.flatten()[self.current_index] / self.cmap_max, mode='qcolor'
        )

        if self.sim_popup is None:
            self.sim_popup = SimulationPopup()

        self.sim_popup.runSimulation(values, color)
        self.sim_popup.show()

    def computeLatentSpace(self):
        """
        When the fit parameters are changed in any way, the latent space must be recomputed
        """
        if not self.computation_enabled:
            return

        input_values = np.stack((self.params,) * self.n * self.n, axis=0)  # (n*n, 10)
        input_values[:, self.variable_params] = self.coords
        self.latent_space = self.pca_forward_model.predict(input_values)

        self.computeDistances()

    def computeDistances(self):
        """
        When latent space or current index is updated, this should be called
        """
        if self.image.mouse_in:
            i = self.current_index
            clicked_coord = self.latent_space[i : i + 1]  # keep dim of (1, n*n)
        else:
            clicked_coord = self.set_latent_space

        dist = np.sqrt(np.sum((self.latent_space - clicked_coord) ** 2, axis=1))
        self.image.setImage(dist.reshape(self.n, self.n), levels=(0, self.cmap_max))

        if not self.image.mouse_in:
            transformed_pos = pg.transformCoordinates(self.transform, self.coords[self.set_index] * self.n)
            self.CoordinatesLabel.setText(f'({transformed_pos[0]:.4e}, {transformed_pos[1]:.4e})')

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


class SimulationPopup(pg.GraphicsView):
    def __init__(self, *args, **kwargs):
        super(SimulationPopup, self).__init__(*args, **kwargs)
        self.resize(800, 600)
        layout = pg.GraphicsLayout()
        self.setCentralItem(layout)

        self.phys_param = None
        self.forward_model = keras.models.load_model(
            r'D:\datasets\meteor\trained_models\trained2\model_forward_problem_mag.hdf5', compile=False
        )
        self.generator = loadProcessedData(r'D:\datasets\meteor\noerosion_dataset.h5', 1, validation_split=0)
        self.optimizer = ps.single.GlobalBestPSO(
            n_particles=3000,
            dimensions=10,
            options={'c1': 1.5, 'c2': 3, 'w': 0.5},
            bounds=(np.full(10, 0), np.full(10, 1.0)),
        )

        self.magnitude_plot = layout.addPlot(0, 0)
        self.magnitude_plot.showGrid(x=True, y=True, alpha=0.3)
        self.magnitude_plot.setLabel('bottom', 'Magnitude')
        self.magnitude_plot.setLabel('left', 'Height (km)')
        self.magnitude_plot.setXRange(8, -2)

        self.velocity_plot = layout.addPlot(0, 1)
        self.velocity_plot.showGrid(x=True, y=True, alpha=0.3)
        self.velocity_plot.setLabel('bottom', 'Velocity (km/s)')
        self.velocity_plot.setLabel('left', 'Height (km)')
        self.velocity_plot.setXRange(20, 80)
        self.velocity_plot.setYRange(70, 140)

        self.norm_magnitude_plot = layout.addPlot(1, 0)
        self.norm_magnitude_plot.showGrid(x=True, y=True, alpha=0.3)
        self.norm_magnitude_plot.setLabel('bottom', 'Magnitude')
        self.norm_magnitude_plot.setLabel('left', 'Height')

        self.norm_velocity_plot = layout.addPlot(1, 1)
        self.norm_velocity_plot.showGrid(x=True, y=True, alpha=0.3)
        self.norm_velocity_plot.setLabel('bottom', 'Velocity')
        self.norm_velocity_plot.setLabel('left', 'Height')

        self.magnitude_plot.setYLink(self.velocity_plot)
        self.norm_magnitude_plot.setYLink(self.norm_velocity_plot)

    def function(self, X, heights=None, goal=None):
        """ pyswarms optimize function """
        input_data = np.concatenate(
            (np.tile(heights, X.shape[0])[:, None], np.repeat(X, heights.shape[0], axis=0)), axis=1
        )
        return np.mean(
            (self.forward_model.predict(input_data).reshape(X.shape[0], heights.shape[0], 2) - goal[None])
            ** 2,
            axis=(1, 2),
        )

    def runSimulation(self, norm_values, color):
        y, x = next(self.generator)
        norm_values = x[0]
        self.norm_magnitude_plot.plot(y[0, :, 3], y[0, :, 1])
        self.norm_velocity_plot.plot(y[0, :, 2], y[0, :, 1])

        self.phys_param = PhysicalParameters()
        denorm_values = self.phys_param.getDenormalizedInputs(norm_values)
        self.phys_param.setParamValues(denorm_values)
        const = self.phys_param.getConst()

        simulation_results = SimulationResults(const, *runSimulation(const, compute_wake=False))
        self.plot(simulation_results, color)

        heights = np.linspace(0, 1, 256)
        input_data = np.concatenate(
            (heights[:, None], np.repeat(norm_values[None], heights.shape[0], axis=0)), axis=1
        )
        # feature engineering
        input_data = np.concatenate(
            (input_data, input_data[:, 0:1] < input_data[:, 6:7]), axis=1
        )  # add boolean erosion parameter
        input_data = np.concatenate(
            (input_data, input_data[:, 6:11] * (input_data[:, 0:1] < input_data[:, 6:7])), axis=1
        )

        prediction = self.forward_model(input_data, training=False)
        self.norm_magnitude_plot.plot(prediction[:, 0], heights)
        # self.norm_velocity_plot.plot(prediction[:, 0], heights)

        # cost, joint_vars = self.optimizer.optimize(self.function, iters=10, heights=heights, goal=prediction)
        # print(joint_vars)
        # heights = np.linspace(0, 1, 256)
        # input_data = np.concatenate(
        #     (heights[:, None], np.repeat(joint_vars[None], heights.shape[0], axis=0)), axis=1
        # )

        # prediction = self.forward_model(input_data, training=False)
        # self.norm_magnitude_plot.plot(prediction[:, 1], heights)
        # self.norm_velocity_plot.plot(prediction[:, 0], heights)

    def plot(self, sim: SimulationResults, color):
        ret = extractSimData(
            sim_results=sim, phys_params=self.phys_param, camera_params=ErosionSimParameters()
        )
        if ret is not None:
            _, sim_data_normed, _ = ret
            self.norm_magnitude_plot.plot(sim_data_normed[:, 3], sim_data_normed[:, 1], pen=color)
            self.norm_velocity_plot.plot(sim_data_normed[:, 2], sim_data_normed[:, 1], pen=color)

            # cost, joint_vars = self.optimizer.optimize(
            #     self.function, iters=10, heights=sim_data_normed[:, 1], goal=sim_data_normed[:, 2:4]
            # )
            # print(joint_vars)
            # self.computed = True

        self.magnitude_plot.plot(sim.abs_magnitude, sim.brightest_height_arr / 1000, pen=color)
        self.velocity_plot.plot(sim.brightest_vel_arr / 1000, sim.brightest_height_arr / 1000, pen=color)

    def closeEvent(self, event):
        self.norm_magnitude_plot.clear()
        self.norm_velocity_plot.clear()

        self.magnitude_plot.clear()
        self.velocity_plot.clear()
        super().closeEvent(event)


class Autoencoder(keras.models.Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = keras.Sequential(
            [
                keras.layers.Input(shape=(DATA_LENGTH, 2)),
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
                keras.layers.Conv1D(1, 20, padding='same'),
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
        x=input_train[..., 2:4],
        y=input_train[..., 2:4],
        batch_size=batchsize,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_split=0.2,
    )

    pca = PCA(n_components=latent_dim)
    pca.fit(autoencoder.encoder.predict(input_train[:5000, :, 2:4]))
    print(list(pca.explained_variance_ratio_ * 100))

    plt.plot(autoencoder.decoder.predict(pca.inverse_transform(K.one_hot(0, 20)[None]))[0])
    plt.show()
    # # save model
    # # model_json = autoencoder.to_json()
    # # model_file = os.path.join(output_dir, f'{model_name}.h5')
    # # weights_file = os.path.join(output_dir, f'{model_name}.json')
    # # with open(model_file, "w") as json_file:
    # #     json_file.write(model_json)
    # # autoencoder.save_weights(weights_file)
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
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(9,)),
            keras.layers.Dense(100, activation='relu',),  # kernel_regularizer=keras.regularizers.L2(1e-3)
            keras.layers.Dense(100, activation='relu',),  # kernel_regularizer=keras.regularizers.L2(1e-3)
            keras.layers.Dense(100, activation='relu',),  # kernel_regularizer=keras.regularizers.L2(1e-3)
            keras.layers.Dense(100, activation='relu',),  # kernel_regularizer=keras.regularizers.L2(1e-3)
            keras.layers.Dense(100, activation='relu',),  # kernel_regularizer=keras.regularizers.L2(1e-3)
            keras.layers.Dense(5),
        ]
    )

    x, y = loadh5pyData(data_path, 'x', 'y')

    model.compile(loss='mse', optimizer='adam')
    model.fit(
        (y - y.mean(axis=0)) / y.std(axis=0),
        (x[:, :5] - x[:, :5].mean(axis=0)) / x[:, :5].std(axis=0),
        batch_size=batchsize,
        epochs=epochs,
    )

    correlationPlot(
        model.predict((y - y.mean(axis=0)) / y.std(axis=0)),
        (x[:, :5] - x[:, :5].mean(axis=0)) / x[:, :5].std(axis=0),
    )


def trainParamPredictionRegions(
    data_path: str,
    output_dir: str,
    autoencoder_path: str,
    epochs: int,
    steps_per_epoch: int,
    batchsize: int,
    model_name: str = 'model',
):
    autoencoder = keras.models.load_model(autoencoder_path)
    forward = keras.models.load_model(
        r'D:\datasets\meteor\trained_models\trained2\noersion_model_ls_forward.hdf5'
    )
    # to_pca = keras.models.load_model(
    #     rf'D:\datasets\meteor\trained_models\trained2\tuned_latentspace_finder.hdf5'
    # )

    y, x = loadh5pyData(data_path)
    # y = autoencoder.encoder.predict(y[..., 1:])
    # generator = loadProcessedData(data_path, batchsize, validation_split=0.2)
    # generator = ((autoencoder.encoder.predict(sim[..., 1:]), param) for (sim, param) in generator)

    # pca = PCA(n_components=20)
    # pca.fit(y[:5000])
    # pca_weights = pca.explained_variance_ratio_
    # print(np.ceil(pca_weights * 100))
    # print(np.prod(np.ceil(pca_weights * 100)))
    # print(np.prod(np.ceil(pca_weights * 100)[:7]))
    # H, edges = np.histogramdd(pca.transform(y)[:, :7], bins=np.ceil(pca_weights * 100).astype(int)[:7])
    # p = gaussian_kde(y[:2000].T)(y.T)
    # p = np.where(p > 0, 1 / p, 0)
    # p /= np.sum(p)

    # selected_indices = np.random.choice(np.arange(len(p)), size=(len(p) * 10,), p=p)

    H, edges = np.histogramdd(x[:, :5], bins=4)
    error_arr = np.zeros(H.shape)
    print(error_arr.shape)
    print(H.min(), H.max(), H.mean(), H.std(), np.sum(H < 30))
    # raise Exception('hey')
    index_list = np.arange(H.size)
    np.random.shuffle(index_list)
    for bin_index_ in index_list:
        gen_x = np.random.uniform(low=0, high=1, size=x.shape)
        # bin_index_ = 100
        bin_index = np.unravel_index(bin_index_, H.shape)
        for dim in range(len(bin_index)):
            gen_x[:, dim] = (
                gen_x[:, dim] * (edges[dim][bin_index[dim] + 1] - edges[dim][bin_index[dim]])
                + edges[dim][bin_index[dim]]
            )
        gen_y = forward.predict(gen_x)

        model = keras.Sequential(
            [
                keras.layers.Input(shape=(20,)),
                keras.layers.Dense(200, activation='relu',),  # kernel_regularizer=keras.regularizers.L2(1e-3)
                keras.layers.Dense(200, activation='relu',),  # kernel_regularizer=keras.regularizers.L2(1e-3)
                keras.layers.Dense(200, activation='relu',),  # kernel_regularizer=keras.regularizers.L2(1e-3)
                keras.layers.Dense(200, activation='relu',),  # kernel_regularizer=keras.regularizers.L2(1e-3)
                keras.layers.Dense(200, activation='relu',),  # kernel_regularizer=keras.regularizers.L2(1e-3)
                keras.layers.Dense(5),
            ]
        )
        model.compile(optimizer='adam', loss='mse')
        i = slice(0, 5)
        history = model.fit(
            x=(gen_y - gen_y.mean(axis=0)) / gen_y.std(axis=0),
            y=((gen_x - gen_x.mean(axis=0)) / gen_x.std(axis=0))[:, i],
            batch_size=1000,
            epochs=20,
            verbose=0,
        )

        # correlationPlot(
        #     ((gen_x - gen_x.mean(axis=0)) / gen_x.std(axis=0))[:, i],
        #     model.predict((gen_y - gen_y.mean(axis=0)) / gen_y.std(axis=0)),
        # )
        print(bin_index_, history.history['loss'][-1])
        # print()
        error_arr[bin_index] = history.history['loss'][-1]
        # raise Exception('hey')

    # keras.models.save_model(
    #     model, os.path.join(output_dir, model_name + '_encoder_translator_sampled.hdf5'), save_format='h5'
    # )
    np.save('file.npy', error_arr)
    np.savez('file2.npz', *edges)


def trainAutoencoderFindLatentSpace(
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

    # label_train[:, 3] >

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(10,)),
            keras.layers.Dense(190, activation='relu'),
            keras.layers.Dense(190, activation='relu'),
            keras.layers.Dense(190, activation='relu'),
            keras.layers.Dense(190, activation='relu'),
            keras.layers.Dense(190, activation='relu'),
            keras.layers.Dense(190, activation='relu'),
            keras.layers.Dense(20),
        ]
    )
    model.compile(optimizer='adam', loss='mse')

    model.fit(
        x=label_train,
        y=autoencoder.encoder.predict(input_train[..., 1:]),
        batch_size=batchsize,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_split=0.2,
    )
    keras.models.save_model(
        model, os.path.join(output_dir, f'{model_name}_ls_forward_.hdf5'), save_format='h5'
    )
    # prediction = model.predict(autoencoder.encoder.predict(input_train))
    # print(pearsonr(prediction[:, 3], label_train[:, 3])[0], pearsonr(prediction[:, 4], label_train[:, 4])[0])


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

    def build_model(hp):
        # relu, adam, 190, 10
        activation = hp.Choice("activation", ["relu", "tanh"])
        optimizer = hp.Choice('optimizer', ['adagrad', 'adam', 'adamax', 'nadam', 'sgd', 'rmsprop'])
        units = hp.Int('units', min_value=20, max_value=200, step=10)
        hidden_layers = hp.Int('layers', min_value=1, max_value=10, step=1)
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(10,)))
        for i in range(hidden_layers):
            model.add(keras.layers.Dense(units, activation=activation))

        model.add(keras.layers.Dense(20))
        model.compile(optimizer=optimizer, loss='mse')
        return model

    tuner = kt.RandomSearch(build_model, objective='val_loss', max_trials=100)
    tuner.search(
        label_train,
        pca.transform(autoencoder.encoder.predict(input_train)),
        epochs=1,
        validation_split=0.2,
        batch_size=batchsize,
    )
    model = tuner.get_best_models()[0]

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

    latent_space = autoencoder.encoder.predict(input_train[:20000, :, 2:3])
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
    # (line1,) = ax[0].plot(data[:, 2])
    # (line2,) = ax[1].plot(data[:, 1])
    (line3,) = ax[0].plot(data[:, 0])
    ax[0].set_ylabel('Normalized Magnitude')
    ax[1].set_ylabel('Normalized length')

    def update(*args):
        input_data = np.array([slider.val for slider in sliders])[None]
        data = autoencoder.decoder.predict(pca.inverse_transform(input_data))[0]

        # line1.set_ydata(data[:, 0])
        # line1.set_ydata(data[:, 2])
        # line2.set_ydata(data[:, 0])
        # line2.set_ydata(data[:, 1])
        line3.set_ydata(data[:, 0])
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
        # pca = PCA(n_components=20)
        # pca.fit(autoencoder.encoder.predict(input_train[:5000]))

        # correlationPlot(
        #     model.predict(label_train),
        #     pca.transform(autoencoder.encoder.predict(input_train)),
        #     [False] * 10,
        #     [''] * 10,
        #     [''] * 10,
        #     ['', ''],
        # )
        correlationPlot(model.predict(label_train), autoencoder.encoder.predict(input_train[..., 1:]))
    else:
        correlationPlot(
            label_train,
            model.predict(autoencoder.encoder.predict(input_train[..., 1:])),
            # model.predict(autoencoder.encoder.predict(input_train)),
        )


def visualizeLatentSpaceDistance(model_path):
    model = keras.models.load_model(model_path)

    app = QApplication([])
    main_window = LatentDistanceGUI(model)
    main_window.show()
    sys.exit(app.exec_())


def main():
    data_path = r'D:\datasets\meteor\very_restricted_dataset.h5'
    data_path2 = r'D:\datasets\meteor\norestrictions2_dataset.h5'
    data_path3 = r'D:\datasets\meteor\noerosion_dataset.h5'
    fit_data_path = r'D:\datasets\meteor\fit_dataset2.h5'
    model_path = r'D:\datasets\meteor\trained_models\trained2'
    model_name = 'noersion_model'
    autoencoder_path = rf'D:\datasets\meteor\trained_models\trained2\model_encoder'
    translator_path = rf'D:\datasets\meteor\trained_models\trained2\tuned_encoder_translator.hdf5'
    latentspacefinder_path = rf'D:\datasets\meteor\trained_models\trained2\tuned_latentspace_finder.hdf5'

    # trainAutoencoder(data_path3, model_path, 30, 50, 500, model_name=model_name)

    # trainParamPrediction(fit_data_path, model_path, autoencoder_path, 5, 300, 500, model_name=model_name)
    # trainParamPredictionRegions(data_path3, model_path, autoencoder_path, 5, 300, 500, model_name=model_name)
    # trainAutoencoderFindLatentSpace(
    #     data_path3, model_path, autoencoder_path, 30, 300, 500, model_name=model_name
    # )

    # visualizeLatentSpace(rf'D:\datasets\meteor\trained_models\trained2\test2_encoder', data_path2)

    # visualizeInverseSolving(
    #     autoencoder_path,
    #     r'D:\datasets\meteor\trained_models\trained2\noersion_model_encoder_translator_sampled.hdf5',
    #     data_path3,
    #     forward=False,
    # )

    visualizeLatentSpaceDistance(latentspacefinder_path)


if __name__ == '__main__':
    main()
