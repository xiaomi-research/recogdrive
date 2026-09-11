import os

import setuptools

script_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_folder)

req = os.path.join(script_folder, "requirements.txt")
if not os.path.isfile(req):
    req = os.path.join(os.path.dirname(script_folder), "requirements.txt")
with open(req) as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="navsim",
    version="1.1.0",
    author="University of Tuebingen",
    author_email="kashyap.chitta@uni-tuebingen.de",
    description="NAVSIM 1.1 + ReCogDrive",
    url="https://github.com/autonomousvision/navsim",
    python_requires=">=3.9",
    packages=setuptools.find_packages(script_folder),
    package_dir={"": "."},
    license="apache-2.0",
    install_requires=requirements,
)
