import os

import setuptools

# Install either NAVSIM tree together with the shared ReCogDrive implementation.
script_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_folder)
tree = os.environ.get("NAVSIM_TREE", "1.1")
if tree not in ("1.1", "2.0"):
    raise ValueError("NAVSIM_TREE must be 1.1 or 2.0")
tree_path = "navsim" + tree

with open(os.path.join(tree_path, "requirements.txt")) as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="navsim",
    version=tree + ".0",
    description="ReCogDrive: NAVSIM 1.1 (default) and 2.0 live in navsim1.1/ and navsim2.0/",
    python_requires=">=3.9",
    packages=setuptools.find_packages(tree_path) + setuptools.find_packages(include=["recogdrive", "recogdrive.*"]),
    package_dir={"": tree_path, "recogdrive": "recogdrive"},
    package_data={
        "navsim": ["planning/script/config/**/*.yaml"],
        "recogdrive.eval.vqa.drivebench": ["system_prompt.txt", "visual_description.json"],
    },
    license="apache-2.0",
    install_requires=requirements,
    extras_require={"lora": ["peft"], "vqa": ["lmdeploy"], "closedloop": ["grpcio"],
                    "test": ["pytest>=8,<9", "wheel"]},
)
