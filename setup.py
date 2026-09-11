import os

import setuptools

# One repo, two trees. Default install is 1.1. For 2.0: pip install -e navsim2.0
script_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_folder)

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="navsim",
    version="1.1.0",
    description="ReCogDrive: NAVSIM 1.1 (default) and 2.0 live in navsim1.1/ and navsim2.0/",
    python_requires=">=3.9",
    packages=setuptools.find_packages("navsim1.1"),
    package_dir={"": "navsim1.1"},
    license="apache-2.0",
    install_requires=requirements,
)
