""" Import all submodules. """


import pkgutil

# Excluded packages
exclude = ["MetSim.ML", "GUI", "MetSim"]

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
    module = loader.find_module(module_name).load_module(module_name)
    exec('%s = module' % module_name)
