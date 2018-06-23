
import pkgutil

# Import all submodules

__all__ = []
for loader, module_name, is_pkg in  pkgutil.walk_packages(__path__):
    
    # Skip the config module
    if 'Config' in module_name:
        continue

    __all__.append(module_name)
    module = loader.find_module(module_name).load_module(module_name)
    exec('%s = module' % module_name)