""" Tests that the comparison plot of the GUI does not change the observations from a .met file, which the
automated fit and the time offset of the simulation use afterwards, and that its legend names the renamed
CAMO sites.

Run with pytest:
    python -m pytest wmpl/MetSim/Tests/test_GUIMetObservations.py -v
"""

from __future__ import print_function, division, absolute_import

import copy
import tempfile
from types import SimpleNamespace
from unittest import mock

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from wmpl.MetSim.GUI import MetSimGUI, plotObsAndSimComparison
from wmpl.MetSim.Tests.DisplacedFirstPoint import simulate, observe



def testComparisonPlotKeepsTheMetSites():
    """ Check that CAMO observations from the .met sites 2 and 4, which the plot shows as 1 and 2, keep their
        sites and data after the plot, so the input data of the fit can still be made from them, and that the
        legend names both wide-field sites.
    """

    sr = simulate()

    # Observations of the two CAMO stations, with magnitudes
    traj = observe(sr, 0.0)
    traj.jdt_ref = 2459000.5
    for obs, station_id in zip(traj.observations, ["1", "2"]):
        obs.station_id = station_id
        obs.absolute_magnitudes = np.zeros(len(obs.time_data))

    # Additional magnitudes from the .met file of the sites 2 and 4
    met_obs = SimpleNamespace(sites=["2", "4"], height_data={}, abs_mag_data={}, time_data={})
    for site, obs in zip(met_obs.sites, traj.observations):
        met_obs.height_data[site] = obs.model_ht
        met_obs.abs_mag_data[site] = obs.absolute_magnitudes + 1.0
        met_obs.time_data[site] = obs.time_data

    met_obs_before = copy.deepcopy(met_obs)

    # Keep the figure, which the plot closes after saving it
    figures = []
    close = plt.close
    def keepFigure(fig=None):
        """ Keep the figure and close it. """

        figures.append(fig)
        close(fig)

    with tempfile.TemporaryDirectory() as plot_dir, mock.patch.object(plt, 'close', keepFigure):
        plotObsAndSimComparison(traj, sr, None, met_obs, None, plot_dir, plot_lag=True, camo=True)

    # The legend has both wide-field sites under their CAMO names
    labels = [text.get_text() for ax in figures[0].axes if ax.get_legend() is not None \
        for text in ax.get_legend().get_texts()]
    assert "WF - Tavistock" in labels
    assert "WF - Elginfield" in labels

    assert met_obs.sites == met_obs_before.sites
    for key in ['height_data', 'abs_mag_data', 'time_data']:
        assert sorted(getattr(met_obs, key)) == sorted(getattr(met_obs_before, key))

    # The input data of the fit has all points of the trajectory and of the .met file
    fit_input_data = MetSimGUI.fitInputData(SimpleNamespace(traj=traj, ecsv_obs=None, met_obs=met_obs))
    assert len(fit_input_data) == 2*sum(len(obs.time_data) for obs in traj.observations)




if __name__ == "__main__":

    # Run all tests
    test_list = [testComparisonPlotKeepsTheMetSites]

    for test in test_list:
        test()
        print("PASS {:s}".format(test.__name__))
