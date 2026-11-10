""" Tests that importing wmpl imports each of its modules once, under its full name.

Run with pytest:
    python -m pytest wmpl/Utils/Tests/test_PackageImport.py -v
"""

from __future__ import print_function, division, absolute_import

import os
import pickle
import subprocess
import sys

from wmpl.Utils.Pickling import loadPickle


# Root of the repository which holds this copy of wmpl
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))



def testModulesAreImportedOnceUnderTheirFullNames():
    """ Check in a fresh process that after importing wmpl, no module is also loaded without the wmpl prefix,
        and that a module reached through the package attributes is the one in sys.modules.
    """

    # List the modules loaded under the names of the top-level packages of wmpl, without the prefix
    code = "import pkgutil, sys, wmpl; import wmpl.Utils.AtmosphereDensity as module; " \
        "print(module is sys.modules['wmpl.Utils.AtmosphereDensity']); " \
        "packages = [info.name for info in pkgutil.iter_modules(wmpl.__path__)]; " \
        "print(sorted(name for name in sys.modules if name.split('.')[0] in packages))"

    # Run from the root of the repository, so the copy of wmpl under test is imported
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=REPO_ROOT, \
        env=dict(os.environ, MPLBACKEND='Agg'))

    lines = result.stdout.strip().splitlines()

    assert result.returncode == 0, result.stderr
    assert lines[-2] == 'True'
    assert lines[-1] == '[]'



def testPickleWithModulesWithoutThePrefixLoads(tmp_path):
    """ Check that a pickle which names a wmpl module without the wmpl prefix, as older versions of wmpl could
        save them, still loads.
    """

    from wmpl.Utils.TrajConversions import EARTH

    # Save the object under the name of its module without the prefix
    data = pickle.dumps(EARTH, protocol=2).replace(b'wmpl.Utils.TrajConversions', b'Utils.TrajConversions')
    assert b'wmpl.' not in data
    (tmp_path / 'old.pickle').write_bytes(data)

    earth = loadPickle(str(tmp_path), 'old.pickle')

    assert type(earth) is type(EARTH)
    assert earth.EQUATORIAL_RADIUS == EARTH.EQUATORIAL_RADIUS
