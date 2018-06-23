import os
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


packages = []

dir_path = os.path.split(os.path.abspath(__file__))[0]

# Get all folders with Python packages
for dir_name in os.listdir(dir_path):

    local_path = os.path.join(dir_path, dir_name)

    if os.path.isdir(local_path):

        # Check if there is an __init__.py file in those folders
        if "__init__.py" in os.listdir(local_path):

            packages.append(dir_name)


# Read requirements file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()


# Get all data files in 'share'
share_files = [os.path.join('wmpl', 'share', file_name) for file_name in os.listdir(os.path.join(dir_path, 'wmpl', 'share'))]


setup(
    name = "westernmeteorpylib",
    version = "1.0",
    author = "Denis Vida",
    author_email = "denis.vida@gmail.com",
    description = ("Python code developed for the Western Meteor Physics Group."),
    license = "MIT",
    keywords = "meteors",
    packages=find_packages(),
    data_files=[(os.path.join('wmpl', 'share'), share_files)],
    include_package_data=True,
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
    ],
    setup_requires=["numpy"],
    install_requires=requirements
)