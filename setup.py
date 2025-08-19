#!/usr/bin/env python

import os
from setuptools import find_packages, setup
from typing import List

# Package meta-data.
REPO_NAME = "flight_price_prediction-ml-project"
DESCRIPTION = "An AI/ML project for Flight Price Price Prediction system"
URL = "https://github.com/candobettercode/gems_prediction-ml-project.git"
SRC_REPO = "src" # Name of local package
EMAIL = "siddhesh1199@gmail.com"
AUTHOR = "condonettercode"
REQUIRES_PYTHON = ">=3.8"
VERSION = "0.1.0"

# Requirements helper
def get_requirements(file_path: str) -> List[str]:
    """
    Load dependencies from requirements.txt.
    Ignores empty lines and comments.
    """
    requirements = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if line == "-e .":
                    continue
                requirements.append(line)
    return requirements

# What packages are required for this module to be executed?
REQUIRED = get_requirements("requirements.txt")

# Optional dependencies
EXTRAS = {
    "dev": ["black", "flake8", "isort", "pytest", "pytest-cov"],
    "docs": ["sphinx", "sphinx_rtd_theme"],
}

here = os.path.abspath(os.path.dirname(__file__))

# Setup function
setup(
    name=SRC_REPO,
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description=open(os.path.join(here, "README.md"), encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url=URL,
    packages=find_packages(exclude=("tests", "docs")),
    python_requires=REQUIRES_PYTHON,
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="machine learning, deep learning, AI, data science, EDA",
)
