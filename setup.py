"""Setup configuration for fallout_sim package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="fallout_sim",
    version="0.1.0",
    author="fallout_sim contributors",
    description="Python Lagrangian fallout simulator compatible with OPEN-RISOP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lwillard/fallout_sim",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
    ],
    entry_points={
        "console_scripts": [
            "fallout-sim=fallout_sim.cli:main",
        ],
    },
)
