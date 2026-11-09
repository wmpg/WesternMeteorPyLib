""" Import all submodules. """


import importlib
import pkgutil

# Install pyximport before any submodule does, so the Cython modules are compiled in the build directory of
#   this copy of wmpl (see wmpl.Utils.CythonBuild), not in one shared with other copies
try:
    from wmpl.Utils.CythonBuild import installPyximport
    installPyximport()
except ImportError:
    pass

# Excluded packages
# Note: "Tests" is excluded because importing the test modules on `import wmpl` executes their
#   top-level code - e.g. wmpl/Utils/Tests/test_AlphaBeta.py calls matplotlib.use("Agg"), which
#   would switch the whole process to the non-interactive Agg backend and break plt.show() in any
#   script that imports wmpl (e.g. `python -m wmpl.Utils.AlphaBeta`).
exclude = ["MetSim.ML", "GUI", "Tests"]

__all__ = []
for module_info in pkgutil.walk_packages(__path__, prefix=__name__ + '.'):

    # Name of the module inside wmpl, e.g. Utils.Physics
    module_name = module_info.name[len(__name__) + 1:]

    # Skip the config module
    if 'Config' in module_name:
        continue

    # Skip Supracenter
    if 'Supracenter' in module_name:
        continue


    ### Skip excluded packages ###
    skip_package = False
    for exclude_test in exclude:
        if exclude_test in module_name:
            skip_package = True
            break

    if skip_package:
        continue

    ### ###


    # Import the module under its full name, which also makes it an attribute of its package, so that it is
    #   not loaded a second time under the name without wmpl
    __all__.append(module_name)
    importlib.import_module(module_info.name)
