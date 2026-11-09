""" Tests that importing wmpl imports each of its modules once, under its full name.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_PackageImport.py -v
"""

from __future__ import print_function, division, absolute_import

import os
import subprocess
import sys



def testModulesAreImportedOnceUnderTheirFullNames():
    """ Check in a fresh process that after importing wmpl, no module is also loaded without the wmpl prefix,
        and that a module reached through the package attributes is the one in sys.modules.
    """

    code = "import sys, wmpl; import wmpl.Utils.AtmosphereDensity as module; " \
        "print(module is sys.modules['wmpl.Utils.AtmosphereDensity']); " \
        "print(sorted(name for name in sys.modules if name.split('.')[0] in ('Utils', 'Trajectory', " \
        "'MetSim', 'Formats')))"

    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, \
        env=dict(os.environ, MPLBACKEND='Agg'))

    lines = result.stdout.strip().splitlines()

    assert result.returncode == 0, result.stderr
    assert lines[-2] == 'True'
    assert lines[-1] == '[]'
