""" Import all submodules. """


import pkgutil
import sys

# Excluded packages
# "Tests" excludes every Tests subpackage (e.g. Utils.Tests, MetSim.Tests) - test modules may call
# matplotlib.use(...) or similar process-wide side effects at import time, which must not fire just
# because something did `import wmpl`
exclude = ["MetSim.ML", "GUI", "Tests"]

__all__ = []
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    
    # Skip the config module
    if 'Config' in module_name:
        continue

    # Skip Supracenter
    if 'Supracenter' in module_name:
        continue


    ### Skip exluded packages ###
    skip_package = False
    for exclude_test in exclude:
        if exclude_test in module_name:
            skip_package = True
            break

    if skip_package:
        continue

    ### ###


    __all__.append(module_name)
    if sys.version_info.major <= 3 and sys.version_info.minor <= 11:
        module = loader.find_module(module_name).load_module(module_name)
    else:
        module = loader.find_spec(module_name).loader.load_module(module_name)
    exec('%s = module' % module_name)
