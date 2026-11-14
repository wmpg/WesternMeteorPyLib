""" Tests that calcMCUncertainties() takes the uncertainties of Vx, Vy and Vz from the spread of the Monte
    Carlo runs, also for the negative components of the radiant.

Run with pytest:
    python -m pytest wmpl/Trajectory/Tests/test_MCVelocityUncertainty.py -v
"""

from __future__ import print_function, division, absolute_import

import numpy as np

import matplotlib
matplotlib.use('Agg')

from wmpl.Trajectory.Trajectory import calcMCUncertainties
from wmpl.Utils.Math import vectNorm


class FakeRun(object):
    """ Stand-in for a Monte Carlo run with the attributes calcMCUncertainties() reads, without an orbit. """

    def __init__(self, v_init, radiant_eci):
        """ Make the run.

        Arguments:
            v_init: [float] Initial velocity (m/s).
            radiant_eci: [ndarray] Unit radiant vector in ECI coordinates.
        """

        self.v_init = v_init
        self.radiant_eci_mini = radiant_eci
        self.state_vect_mini = np.array([4.0e6, 3.0e6, -3.5e6])

        # Same coordinates for the beginning, end and lowest point
        for point in ['rbeg', 'rend', 'htmin']:
            setattr(self, point + '_lat', np.radians(45.0))
            setattr(self, point + '_lon', np.radians(15.0))
            setattr(self, point + '_ele', 80000.0)
            setattr(self, point + '_ele_wgs84', 80045.0)

        self.orbit = None



def makeRuns(radiant_eci, n_runs=400, v_sigma=50.0, radiant_sigma=2e-4, seed=1):
    """ Make Monte Carlo runs scattered around the given radiant and an initial velocity of 30 km/s.

    Arguments:
        radiant_eci: [ndarray] Unit radiant vector in ECI coordinates.

    Keyword arguments:
        n_runs: [int] Number of runs. 400 by default.
        v_sigma: [float] Spread of the initial velocity (m/s). 50 by default.
        radiant_sigma: [float] Spread of each component of the radiant before it is normalized. 2e-4 by
            default.
        seed: [int] Seed of the random numbers. 1 by default.

    Return:
        [list] A list of FakeRun objects.
    """

    rng = np.random.default_rng(seed)

    return [FakeRun(30000.0 + rng.normal(0, v_sigma), \
        vectNorm(radiant_eci + rng.normal(0, radiant_sigma, 3))) for _ in range(n_runs)]



def testVelocityVectorSigmaIsTheSpreadOfTheRuns():
    """ Check that the uncertainties of Vx, Vy, Vz are the standard deviations of the velocity vectors of the
        runs, for a radiant with positive and negative components.
    """

    runs = makeRuns(vectNorm(np.array([0.25, 0.30, -0.92])))
    un = calcMCUncertainties(runs, runs[0])

    velocity_vectors = np.array([run.v_init*run.radiant_eci_mini for run in runs])

    assert np.allclose([un.vx, un.vy, un.vz], np.std(velocity_vectors, axis=0), rtol=1e-12, atol=0)



def testNegativeComponentIsNotUnderestimated():
    """ Check that the spread of the initial velocity does not reduce the uncertainty of a negative component
        of the velocity vector.
    """

    # Negative Z component dominated by the spread of the initial velocity
    runs = makeRuns(vectNorm(np.array([0.25, 0.30, -0.92])), v_sigma=100.0, radiant_sigma=1e-5)
    un = calcMCUncertainties(runs, runs[0])

    vz = np.array([run.v_init*run.radiant_eci_mini[2] for run in runs])

    # The spread of Vz is about 0.92*100 m/s
    assert un.vz > 0.8*92.0
    assert abs(un.vz - np.std(vz)) < 1e-9



if __name__ == "__main__":

    testVelocityVectorSigmaIsTheSpreadOfTheRuns()
    testNegativeComponentIsNotUnderestimated()

    print("All tests passed.")
