""" Save the results of DynamicMassFit in a pickle that a dark flight code can read directly.

The pickle holds a single dict made only of built-in Python types (dict, list, str, float, int, bool and None),
so it can be loaded with pickle.load() in any Python 3 environment, without wmpl or numpy. Ejection states use
the units of the OpenDarkflight input file: degrees, km, km/s, kg and kg/m^3. Times are in seconds after
'ref_time', a UTC string in the '%Y-%m-%d %H:%M:%S.%f' format.

Contents of the dict:
    format, version: [str, int] Identify this layout.
    planet: [str] Body the trajectory is on.
    ref_time: [str] UTC reference time of the nominal trajectory.
    units: [dict] Units of every quantity in the file.
    nominal, minus_2sigma, plus_2sigma: [dict or None] Ejection state at the end of the ablation simulation for
        the nominal dynamic mass and the analytic -/+2 sigma masses (deceleration uncertainty only). None if the
        simulation was not run.
    samples: [dict of lists or None] Ejection state of every Monte Carlo realization whose simulation ran, with
        the same keys as the nominal state, plus v_kill, dyn_mass and traj_index. The realizations are joint
        draws, so their parameters must be used together, not resampled independently.
    uncertainties_included: [list of str] Sources of uncertainty the samples already carry.
    mc_fit: [dict of lists or None] Dynamic mass of every Monte Carlo realization with a velocity fit.
    mc_counts: [dict or None] Number of Monte Carlo realizations in the solver file, fitted and simulated.
    fit: [dict] Dynamic mass fit on the nominal trajectory.
    model: [dict] Physical assumptions of the dynamic mass and the ablation simulation. msis_version and
        msis_time give the MSIS model (--atm, e.g. "00" or "2.1") and the time it was evaluated at if --atmtime
        set one. Its atm_profile entry describes the atmosphere profile file used instead of the MSIS model (path,
        format, sha256, height range in km, and whether its winds were used), or is None, so a dark flight code
        can check it uses the same atmosphere.
    provenance: [dict] Input files, DynamicMassFit options, wmpl commit and creation time.
"""

import datetime
import os
import pickle
import subprocess

import numpy as np

from wmpl.Utils.TrajConversions import jd2Date


FORMAT_NAME = 'wmpl_dynmassfit'
FORMAT_VERSION = 1

# Keys of an ejection state
EJECTION_KEYS = ['lat', 'lon', 'ht', 'vel', 'az', 'alt', 'ref_time_ds', 'mass', 'density']

UNITS = {
    'lat': 'deg, geodetic, +N',
    'lon': 'deg, +E',
    'ht': 'km above the geoid (EGM96)',
    'vel': 'km/s, relative to the ground',
    'az': 'deg, azimuth of the ground-fixed radiant, +E of due N',
    'alt': 'deg, elevation of the ground-fixed radiant',
    'ref_time_ds': 's after ref_time',
    'mass': 'kg',
    'density': 'kg/m^3',
    'v_kill': 'km/s, relative to the air',
    'dyn_mass': 'kg',
    'decel': 'km/s^2',
    'ht_eval': 'km',
    'vel_eval': 'km/s, relative to the ground',
    'time_eval': 's after ref_time',
    'sim_ablation_coeff': 's^2/km^2',
}



def ejectionState(lat, lon, ht, vel, az, alt, ref_time_ds, mass, density):
    """ Ejection state for the dark flight, in the units of the OpenDarkflight input file.

    Arguments:
        lat: [float] Latitude (deg).
        lon: [float] Longitude (deg).
        ht: [float] Height above the geoid (km).
        vel: [float] Speed (km/s).
        az: [float] Azimuth of the ground-fixed radiant (deg, +E of due N).
        alt: [float] Elevation of the ground-fixed radiant (deg).
        ref_time_ds: [float] Time after the reference time (s).
        mass: [float] Mass (kg).
        density: [float] Bulk density (kg/m^3).

    Return:
        [dict] The ejection state.
    """

    return dict(zip(EJECTION_KEYS, [float(val) for val in [lat, lon, ht, vel, az, alt, ref_time_ds, mass, \
        density]]))



def jdToRefTime(jd):
    """ Convert a Julian date to the UTC time string used by the OpenDarkflight input file. """

    return jd2Date(jd, dt_obj=True).strftime('%Y-%m-%d %H:%M:%S.%f')



def _plain(value):
    """ Convert numpy values in a nested structure to built-in Python types. """

    if isinstance(value, dict):
        return {str(key): _plain(val) for key, val in value.items()}

    if isinstance(value, (list, tuple)):
        return [_plain(val) for val in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    return value



def _wmplCommit():
    """ Return the git commit of the wmpl checkout, or None if it is not a git repository. """

    try:
        res = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(os.path.abspath(__file__)), \
            capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None

    if res.returncode != 0:
        return None

    return res.stdout.strip()



def _mcBlocks(mc_results, jd_ref):
    """ Build the samples and mc_fit blocks from the results of DynamicMassFit.runMonteCarloDynMass(). """

    mc_fit = {
        'dyn_mass': mc_results['dyn_mass'],
        'dyn_mass_geom': mc_results['dyn_mass_geom'],
        'decel': mc_results['decel']/1000,
        'vel_eval': mc_results['vel_eval']/1000,
        'ht_eval': mc_results['ht_eval']/1000,
        'density': mc_results['density'],
        'traj_index': mc_results['traj_index'].astype(int),
    }

    if 'final_mass' not in mc_results:
        return None, mc_fit

    sim = ~np.isnan(mc_results['final_mass'])

    if not np.any(sim):
        return None, mc_fit

    samples = {
        'lat': mc_results['final_lat'][sim],
        'lon': mc_results['final_lon'][sim],
        'ht': mc_results['final_ele'][sim],
        'vel': mc_results['final_vel'][sim]/1000,
        'az': mc_results['final_azim'][sim],
        'alt': mc_results['final_elev'][sim],

        # Each realization has its own reference time, which differs from the nominal one by a few ms
        'ref_time_ds': (mc_results['jdt_ref'][sim] - jd_ref)*86400 + mc_results['final_time'][sim],

        'mass': mc_results['final_mass'][sim],
        'density': mc_results['density'][sim],
        'v_kill': mc_results['v_kill'][sim]/1000,
        'dyn_mass': mc_results['dyn_mass'][sim],
        'traj_index': mc_results['traj_index'][sim].astype(int),
    }

    return samples, mc_fit



def buildDynMassFitOutput(jd_ref, fit, model, nominal=None, minus_2sigma=None, plus_2sigma=None, \
    mc_results=None, mc_realizations=None, traj_path=None, mc_path=None, dmf_args=None, planet='earth'):
    """ Assemble the DynamicMassFit results into a dict of built-in Python types (see the module docstring).

    Arguments:
        jd_ref: [float] Reference Julian date of the nominal trajectory.
        fit: [dict] Dynamic mass fit on the nominal trajectory.
        model: [dict] Physical assumptions. It needs 'density_sigma' (kg/m^3) and 'v_kill_sigma' (km/s).

    Keyword arguments:
        nominal, minus_2sigma, plus_2sigma: [dict or None] Ejection states (see ejectionState()).
        mc_results: [dict of ndarray or None] Output of DynamicMassFit.runMonteCarloDynMass().
        mc_realizations: [int or None] Number of Monte Carlo realizations in the solver file.
        traj_path: [str or None] Path of the trajectory pickle.
        mc_path: [str or None] Path of the solver's Monte Carlo uncertainties pickle.
        dmf_args: [dict or None] DynamicMassFit command line options.
        planet: [str] Body the trajectory is on.

    Return:
        [dict] The output.
    """

    samples = mc_fit = mc_counts = None

    if mc_results is not None:

        samples, mc_fit = _mcBlocks(mc_results, jd_ref)

        mc_counts = {
            'in_solver_file': mc_realizations,
            'fitted': len(mc_results['dyn_mass']),
            'simulated': 0 if samples is None else len(samples['mass']),
        }

    # The samples always carry the trajectory geometry and velocity fit uncertainties
    uncertainties = None
    if samples is not None:
        uncertainties = ['trajectory_geometry', 'velocity_fit']
        if model['density_sigma'] > 0:
            uncertainties.append('density')
        if model['v_kill_sigma'] > 0:
            uncertainties.append('v_kill')

    output = {
        'format': FORMAT_NAME,
        'version': FORMAT_VERSION,
        'planet': planet,
        'ref_time': jdToRefTime(jd_ref),
        'units': UNITS,
        'nominal': nominal,
        'minus_2sigma': minus_2sigma,
        'plus_2sigma': plus_2sigma,
        'samples': samples,
        'uncertainties_included': uncertainties,
        'mc_fit': mc_fit,
        'mc_counts': mc_counts,
        'fit': fit,
        'model': model,
        'provenance': {
            'traj_path': None if traj_path is None else os.path.abspath(traj_path),
            'mc_uncertainties_path': None if mc_path is None else os.path.abspath(mc_path),
            'dmf_args': dmf_args,
            'wmpl_commit': _wmplCommit(),
            'created_utc': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        },
    }

    return _plain(output)



def saveDynMassFitPickle(file_path, output):
    """ Save the output of buildDynMassFitOutput() to a pickle readable by Python 3.4 or newer. """

    with open(file_path, 'wb') as f:
        pickle.dump(output, f, protocol=4)



def printDynMassFitPickle(file_path):
    """ Print a summary of a pickle saved by saveDynMassFitPickle(). """

    with open(file_path, 'rb') as f:
        out = pickle.load(f)

    print("{:s} v{:d}, planet: {:s}, ref_time: {:s} UTC".format(out['format'], out['version'], out['planet'], \
        out['ref_time']))

    for name in ['minus_2sigma', 'nominal', 'plus_2sigma']:
        state = out[name]
        if state is None:
            print("{:12s} not simulated".format(name))
            continue

        print("{:12s} lat {lat:.5f}, lon {lon:.5f}, ht {ht:.3f} km, vel {vel:.3f} km/s, az {az:.3f}, alt {alt:.3f}, "
            "t {ref_time_ds:.3f} s, mass {mass:.4f} kg, density {density:.0f} kg/m^3".format(name, **state))

    if out['samples'] is None:
        print("samples      none")
    else:
        print("samples      {:d} Monte Carlo realizations, carrying: {:s}".format(len(out['samples']['mass']), \
            ", ".join(out['uncertainties_included'])))

    print("created      {:s} UTC, wmpl {}".format(out['provenance']['created_utc'], \
        out['provenance']['wmpl_commit']))



if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Print a summary of a pickle saved by DynamicMassFit "
        "with --save_pickle.")

    arg_parser.add_argument('file_path', metavar='FILE_PATH', type=str, help="Path to the pickle.")

    cml_args = arg_parser.parse_args()

    printDynMassFitPickle(cml_args.file_path)
