""" Tests that the atmosphere of the drag fit of --vinitdrag is chosen with the environment variables of
    wmpl.Utils.AtmosphereDensity, as the solver's command lines have no atmosphere options.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_DragInitialVelocityAtmosphere.py -v
"""

from __future__ import print_function, division, absolute_import

import os
import subprocess
import sys


# Variables of wmpl.Utils.AtmosphereDensity which select the MSIS model of a new process
MSIS_VARIABLES = ("WMPL_MSIS_VERSION", "WMPL_MSIS_SPACE_WEATHER", "WMPL_MSIS_JD")



def _atmosphereInNewProcess(**variables):
    """ Start a new process with the given MSIS variables and return the atmosphere of its drag fit.

    Keyword arguments:
        variables: [str] Values of the MSIS variables, by name. The others are left unset.

    Return:
        [str] The description of atmosphereDescription() in that process.
    """

    env = {key: value for key, value in os.environ.items() if key not in MSIS_VARIABLES}
    env.update(variables)

    code = "from wmpl.Utils.DragInitialVelocity import atmosphereDescription; print(atmosphereDescription())"
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)

    return out.stdout.strip().splitlines()[-1]



def testTheEnvironmentChoosesTheAtmosphereOfTheDragFit():
    """ Without the variables the fit takes NRLMSIS 2.1 with the measured solar activity, and with them the
        model and the activity they name.
    """

    assert _atmosphereInNewProcess() \
        == "NRLMSIS 2.1 with the measured solar activity, at the trajectory's time"

    assert _atmosphereInNewProcess(WMPL_MSIS_VERSION="00", WMPL_MSIS_SPACE_WEATHER="constant") \
        == "NRLMSISE-00 with constant solar activity, at the trajectory's time"




if __name__ == "__main__":

    testTheEnvironmentChoosesTheAtmosphereOfTheDragFit()

    print("All tests passed.")
