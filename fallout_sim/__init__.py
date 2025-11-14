"""
Fallout Simulator - A Python Lagrangian fallout simulator compatible with OPEN-RISOP.

This package provides tools for simulating nuclear fallout using Lagrangian particle tracking.
"""

__version__ = "0.1.0"
__author__ = "fallout_sim contributors"

from .simulator import FalloutSimulator
from .particle import Particle, ParticleCloud
from .atmosphere import AtmosphericModel
from .output import RasterOutput

__all__ = [
    "FalloutSimulator",
    "Particle",
    "ParticleCloud",
    "AtmosphericModel",
    "RasterOutput",
]
