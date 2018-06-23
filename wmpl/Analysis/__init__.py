import os


# Get the path of the current folder
dir_path = os.path.split(os.path.abspath(__file__))[0]

# Import all modules
__all__ = [name for name in os.listdir(dir_path) if ((name != '__init__.py') and ('.py' in name)) or os.path.isdir(name)]