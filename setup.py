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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for Fallout Simulator

A GPU-accelerated Lagrangian particle dispersion model for atmospheric fallout transport.
"""

from setuptools import setup, find_packages
import os
import re

# Read version from the main script
def get_version():
    """Extract version from the main script file."""
    version_file = os.path.join(os.path.dirname(__file__), 'fallout_sim.py')
    if os.path.exists(version_file):
        with open(version_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Look for version patterns in comments or docstrings
            version_match = re.search(r'[Vv]ersion\s*[:=]\s*(["\']?)([0-9]+\.[0-9]+(?:\.[0-9]+)?)\1', content)
            if version_match:
                return version_match.group(2)
    return "1.0.0"

# Read the README file
def get_long_description():
    """Read README.md for long description."""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "GPU-accelerated fallout dispersion simulator with vertical wind interpolation."

# Define package requirements
REQUIRED_PACKAGES = [
    # Core scientific computing
    'numpy>=1.20.0',
    'pandas>=1.3.0',
    'xarray>=0.19.0',
    
    # Visualization and mapping
    'matplotlib>=3.4.0',
    'cartopy>=0.20.0',
    
    # Optional: contour smoothing (used in plot generation)
    'scipy>=1.7.0',
]

# Optional dependencies for enhanced functionality
OPTIONAL_PACKAGES = {
    'gpu': [
        'cupy-cuda11x>=9.0.0; platform_system!="Darwin"',  # CUDA 11.x support
        'cupy-cuda12x>=12.0.0; platform_system!="Darwin"',  # CUDA 12.x support (alternative)
    ],
    'geospatial': [
        'geopandas>=0.10.0',
        'shapely>=1.8.0',
        'fiona>=1.8.0',
    ],
    'performance': [
        'numba>=0.55.0',  # JIT compilation for performance-critical functions
    ],
    'dev': [
        'pytest>=6.0.0',
        'pytest-cov>=2.12.0',
        'black>=21.0.0',
        'flake8>=3.9.0',
        'mypy>=0.910',
    ]
}

# All optional dependencies combined
OPTIONAL_PACKAGES['all'] = [
    package for group in OPTIONAL_PACKAGES.values() 
    for package in group
]

setup(
    # Basic package information
    name='fallout-simulator',
    version=get_version(),
    description='GPU-accelerated fallout dispersion simulator with vertical wind interpolation',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    
    # Author and contact information
    author='Fallout Simulator Development Team',
    author_email='contact@fallout-sim.org',  # Replace with actual contact
    url='https://github.com/username/fallout-simulator',  # Replace with actual repository
    
    # License and classification
    license='MIT',  # Update based on your chosen license
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',  # Update to match license
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
        'Environment :: GPU :: NVIDIA CUDA',
    ],
    
    # Package structure
    py_modules=['fallout_sim'],  # Single module package
    python_requires='>=3.8',
    
    # Dependencies
    install_requires=REQUIRED_PACKAGES,
    extras_require=OPTIONAL_PACKAGES,
    
    # Entry points for command-line usage
    entry_points={
        'console_scripts': [
            'fallout-sim=fallout_sim:main',
            'fallout-simulator=fallout_sim:main',
        ],
    },
    
    # Package data and additional files
    include_package_data=True,
    package_data={
        '': ['README.md', 'LICENSE*', '*.txt'],
    },
    
    # Keywords for PyPI search
    keywords=[
        'atmospheric dispersion',
        'fallout simulation',
        'lagrangian particle model',
        'gpu acceleration',
        'meteorology',
        'nuclear fallout',
        'atmospheric transport',
        'cupy',
        'scientific computing',
    ],
    
    # Project URLs
    project_urls={
        'Documentation': 'https://github.com/username/fallout-simulator/wiki',  # Replace with actual docs
        'Bug Reports': 'https://github.com/username/fallout-simulator/issues',  # Replace with actual issues
        'Source': 'https://github.com/username/fallout-simulator',  # Replace with actual repository
        'Funding': 'https://github.com/sponsors/username',  # Replace if applicable
    },
    
    # Additional metadata
    zip_safe=False,  # Package contains data files that need to be accessible
    platforms=['any'],
    
    # Optional: specify minimum versions for critical dependencies
    setup_requires=[
        'setuptools>=45.0',
        'wheel>=0.36.0',
    ],
)

# Post-installation message
def print_installation_info():
    """Print information about optional dependencies after installation."""
    print("\\n" + "="*60)
    print("Fallout Simulator Installation Complete!")
    print("="*60)
    print("\\nRequired dependencies installed successfully.")
    print("\\nOptional enhancements:")
    print("  • GPU acceleration:  pip install fallout-simulator[gpu]")
    print("  • GIS/Shapefile export: pip install fallout-simulator[geospatial]")
    print("  • Performance boost:  pip install fallout-simulator[performance]")
    print("  • All enhancements:  pip install fallout-simulator[all]")
    print("\\nData requirements:")
    print("  • Download NCEP/NCAR reanalysis wind data:")
    print("    https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/pressure/")
    print("\\nUsage:")
    print("  fallout-sim laydown.csv --out results --uwnd uwnd.nc --vwnd vwnd.nc")
    print("\\nDocumentation: See README.md for complete usage guide")
    print("="*60 + "\\n")

if __name__ == '__main__':
    # This will run if setup.py is executed directly
    print_installation_info()
