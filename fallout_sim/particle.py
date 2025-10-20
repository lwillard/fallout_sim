"""
Particle module for Lagrangian fallout simulation.

Defines individual particles and particle clouds for tracking fallout.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Particle:
    """Represents a single fallout particle in the Lagrangian simulation."""
    
    # Position (lat, lon, altitude in meters)
    lat: float
    lon: float
    altitude: float
    
    # Particle properties
    diameter: float  # meters
    density: float   # kg/m^3
    activity: float  # Becquerels
    
    # Particle state
    deposited: bool = False
    deposition_time: float = 0.0
    
    def __post_init__(self):
        """Calculate derived properties."""
        # Calculate terminal velocity (Stokes' law approximation)
        self.terminal_velocity = self._calculate_terminal_velocity()
    
    def _calculate_terminal_velocity(self) -> float:
        """
        Calculate terminal velocity using Stokes' law.
        
        Returns:
            Terminal velocity in m/s (positive downward)
        """
        # Air properties at sea level
        air_density = 1.225  # kg/m^3
        air_viscosity = 1.81e-5  # Pa·s
        g = 9.81  # m/s^2
        
        # Stokes' law for small particles
        v_term = (self.density * g * self.diameter**2) / (18 * air_viscosity)
        
        # Adjust for altitude (simplified)
        altitude_factor = np.exp(-self.altitude / 10000)
        return v_term * altitude_factor


class ParticleCloud:
    """Manages a collection of particles for fallout simulation."""
    
    def __init__(self):
        self.particles: List[Particle] = []
    
    def add_particle(self, particle: Particle):
        """Add a particle to the cloud."""
        self.particles.append(particle)
    
    def create_particles(
        self,
        source_lat: float,
        source_lon: float,
        source_altitude: float,
        num_particles: int,
        size_distribution: str = "log-normal",
        mean_diameter: float = 1e-5,
        std_diameter: float = 2.0,
        activity_per_particle: float = 1e9
    ):
        """
        Create a cloud of particles from a nuclear source.
        
        Args:
            source_lat: Source latitude (degrees)
            source_lon: Source longitude (degrees)
            source_altitude: Source altitude (meters)
            num_particles: Number of particles to create
            size_distribution: "log-normal" or "uniform"
            mean_diameter: Mean particle diameter (meters)
            std_diameter: Standard deviation for log-normal (dimensionless)
            activity_per_particle: Radioactivity per particle (Bq)
        """
        if size_distribution == "log-normal":
            # Log-normal distribution for particle sizes
            diameters = np.random.lognormal(
                mean=np.log(mean_diameter),
                sigma=np.log(std_diameter),
                size=num_particles
            )
        else:
            # Uniform distribution
            diameters = np.random.uniform(
                mean_diameter * 0.5,
                mean_diameter * 1.5,
                num_particles
            )
        
        # Create particles with initial positions spread around source
        # Add some initial dispersion
        for diameter in diameters:
            # Small random offset from source
            lat_offset = np.random.normal(0, 0.001)
            lon_offset = np.random.normal(0, 0.001)
            alt_offset = np.random.normal(0, 100)
            
            particle = Particle(
                lat=source_lat + lat_offset,
                lon=source_lon + lon_offset,
                altitude=max(0, source_altitude + alt_offset),
                diameter=diameter,
                density=2500,  # typical soil/debris density kg/m^3
                activity=activity_per_particle
            )
            self.add_particle(particle)
    
    def get_active_particles(self) -> List[Particle]:
        """Return list of particles that haven't been deposited."""
        return [p for p in self.particles if not p.deposited]
    
    def get_deposited_particles(self) -> List[Particle]:
        """Return list of deposited particles."""
        return [p for p in self.particles if p.deposited]
    
    def get_positions(self) -> np.ndarray:
        """
        Get positions of all active particles.
        
        Returns:
            Array of shape (n, 3) with [lat, lon, altitude]
        """
        active = self.get_active_particles()
        if not active:
            return np.array([]).reshape(0, 3)
        return np.array([[p.lat, p.lon, p.altitude] for p in active])
    
    def get_deposition_map(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get deposition data for all deposited particles.
        
        Returns:
            Tuple of (latitudes, longitudes, activities)
        """
        deposited = self.get_deposited_particles()
        if not deposited:
            return np.array([]), np.array([]), np.array([])
        
        lats = np.array([p.lat for p in deposited])
        lons = np.array([p.lon for p in deposited])
        activities = np.array([p.activity for p in deposited])
        
        return lats, lons, activities
