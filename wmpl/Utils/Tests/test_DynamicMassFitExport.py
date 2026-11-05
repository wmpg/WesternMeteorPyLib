""" Tests for the DynamicMassFit pickle export (wmpl.Utils.DynamicMassFitExport).

Run with: python -m pytest wmpl/Utils/Tests/test_DynamicMassFitExport.py
"""

import datetime
import pickle

import numpy as np
import pytest

from wmpl.Utils.DynamicMassFitExport import ejectionState, jdToRefTime, buildDynMassFitOutput, \
    saveDynMassFitPickle, EJECTION_KEYS
from wmpl.Utils.TrajConversions import datetime2JD


JD_REF = datetime2JD(datetime.datetime(2021, 2, 28, 21, 54, 15, 882964))

FIT = {'dyn_mass': 0.11, 'decel': 3.2}


def _model(density_sigma=0.0, v_kill_sigma=0.0):
    return {'gamma_a': 0.55, 'density': 3500.0, 'density_sigma': density_sigma, 'v_kill': 3.0, \
        'v_kill_sigma': v_kill_sigma}


def _mcResults(n=4):
    """ Monte Carlo results as returned by DynamicMassFit.runMonteCarloDynMass(), with the last realization's
        final simulation not run. """

    rng = np.random.default_rng(0)

    res = {
        'dyn_mass': rng.uniform(0.05, 0.2, n), 'dyn_mass_geom': np.full(n, 0.1), 'decel': rng.uniform(2e3, 4e3, n),
        'vel_eval': np.full(n, 6300.0), 'ht_eval': np.full(n, 30e3), 'time_eval': np.full(n, 7.4),
        'density': rng.uniform(3000, 4000, n), 'traj_index': np.arange(n, dtype=float),

        # The reference time of every realization is 1 ms later than the nominal one
        'jdt_ref': np.full(n, JD_REF + 0.001/86400),

        'final_mass': rng.uniform(0.05, 0.2, n), 'final_lat': np.full(n, 51.94), 'final_lon': np.full(n, -2.08),
        'final_ele': np.full(n, 26.6), 'final_azim': np.full(n, 264.2), 'final_elev': np.full(n, 41.4),
        'final_decel': np.full(n, -1700.0), 'final_vel': np.full(n, 2990.0), 'final_time': np.full(n, 8.5),
        'v_kill': np.full(n, 3000.0),
    }

    for key in ['final_mass', 'final_lat', 'final_lon', 'final_ele', 'final_azim', 'final_elev', 'final_decel', \
        'final_vel', 'final_time']:
        res[key][-1] = np.nan

    return res


def _builtinsOnly(value):
    """ True if the structure holds only built-in Python types. """

    if isinstance(value, dict):
        return all(isinstance(key, str) and _builtinsOnly(val) for key, val in value.items())

    if isinstance(value, list):
        return all(_builtinsOnly(val) for val in value)

    return isinstance(value, (str, float, int, bool, type(None)))


def testRefTimeUsesTheOpenDarkflightFormat():

    ref_time = jdToRefTime(JD_REF)

    parsed = datetime.datetime.strptime(ref_time, '%Y-%m-%d %H:%M:%S.%f')

    assert parsed == pytest.approx(datetime.datetime(2021, 2, 28, 21, 54, 15, 882964), \
        abs=datetime.timedelta(microseconds=100))


def testOutputHoldsOnlyBuiltinTypesAndSurvivesAPickleRoundTrip(tmp_path):

    nominal = ejectionState(np.float64(51.94), -2.08, 26.6, 2.99, 264.2, 41.4, 8.5, 0.103, 3500)

    out = buildDynMassFitOutput(JD_REF, FIT, _model(), nominal=nominal, mc_results=_mcResults(), \
        mc_realizations=4, dmf_args={'mc': True})

    assert _builtinsOnly(out)

    file_path = str(tmp_path/"out.pickle")
    saveDynMassFitPickle(file_path, out)

    with open(file_path, 'rb') as f:
        assert pickle.load(f) == out

    assert list(out['nominal'].keys()) == EJECTION_KEYS


def testSamplesSkipUnsimulatedRealizationsAndUseTheNominalReferenceTime():

    out = buildDynMassFitOutput(JD_REF, FIT, _model(), mc_results=_mcResults(), mc_realizations=5)

    samples = out['samples']

    # The last realization was fitted but not simulated
    assert len(samples['mass']) == 3
    assert out['mc_counts'] == {'in_solver_file': 5, 'fitted': 4, 'simulated': 3}
    assert len(out['mc_fit']['dyn_mass']) == 4

    # Times are shifted to the nominal reference time: 1 ms offset + 8.5 s after the realization's own. A Julian
    #   date of ~2.46e6 in double precision resolves only ~40 us
    assert samples['ref_time_ds'] == pytest.approx([8.501]*3, abs=1e-4)

    # Speeds are in km/s and indices are integers
    assert samples['vel'] == pytest.approx([2.99]*3)
    assert samples['v_kill'] == pytest.approx([3.0]*3)
    assert samples['traj_index'] == [0, 1, 2]


def testUncertaintiesIncludedFollowTheSigmas():

    out = buildDynMassFitOutput(JD_REF, FIT, _model(), mc_results=_mcResults())
    assert out['uncertainties_included'] == ['trajectory_geometry', 'velocity_fit']

    out = buildDynMassFitOutput(JD_REF, FIT, _model(density_sigma=300, v_kill_sigma=0.5), \
        mc_results=_mcResults())
    assert out['uncertainties_included'] == ['trajectory_geometry', 'velocity_fit', 'density', 'v_kill']

    # Without Monte Carlo there are no samples and nothing to declare
    out = buildDynMassFitOutput(JD_REF, FIT, _model())
    assert out['samples'] is None
    assert out['uncertainties_included'] is None
    assert out['mc_counts'] is None



if __name__ == "__main__":

    import sys

    sys.exit(pytest.main([__file__, "-q"]))
