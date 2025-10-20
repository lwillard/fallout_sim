"""
Main simulator module integrating all components.

Implements the Lagrangian fallout simulation.
"""

import numpy as np
from typing import Optional, Tuple
import time as pytime

from .particle import ParticleCloud, Particle
from .atmosphere import AtmosphericModel
from .output import RasterOutput


class FalloutSimulator:
    """
    Main Lagrangian fallout simulator.
    
    Simulates nuclear fallout by tracking individual particles through
    the atmosphere using Lagrangian methods.
    """
    
    def __init__(
        self,
        source_lat: float,
        source_lon: float,
        source_altitude: float = 1000.0,
        num_particles: int = 10000,
        total_activity: float = 1e15,
        atmospheric_model: Optional[AtmosphericModel] = None
    ):
        """
        Initialize fallout simulator.
        
        Args:
            source_lat: Source latitude (degrees)
            source_lon: Source longitude (degrees)
            source_altitude: Initial cloud altitude (meters)
            num_particles: Number of particles to simulate
            total_activity: Total radioactivity (Becquerels)
            atmospheric_model: Atmospheric model for transport
        """
        self.source_lat = source_lat
        self.source_lon = source_lon
        self.source_altitude = source_altitude
        self.num_particles = num_particles
        self.total_activity = total_activity
        
        # Initialize particle cloud
        self.particle_cloud = ParticleCloud()
        self.particle_cloud.create_particles(
            source_lat=source_lat,
            source_lon=source_lon,
            source_altitude=source_altitude,
            num_particles=num_particles,
            activity_per_particle=total_activity / num_particles
        )
        
        # Initialize atmospheric model
        self.atmosphere = atmospheric_model or AtmosphericModel()
        
        # Simulation state
        self.current_time = 0.0
        self.time_step = 60.0  # seconds
    
    def step(self, dt: Optional[float] = None):
        """
        Advance simulation by one time step.
        
        Args:
            dt: Time step in seconds (uses default if None)
        """
        if dt is None:
            dt = self.time_step
        
        active_particles = self.particle_cloud.get_active_particles()
        
        for particle in active_particles:
            if particle.deposited:
                continue
            
            # Get atmospheric conditions
            u, v, w = self.atmosphere.get_wind_velocity(
                particle.lat, particle.lon, particle.altitude, self.current_time
            )
            
            # Add turbulent dispersion
            du, dv, dw = self.atmosphere.get_turbulent_velocity(
                particle.altitude, dt
            )
            
            # Total velocity
            u_total = u + du
            v_total = v + dv
            w_total = w + dw - particle.terminal_velocity
            
            # Convert velocities to position changes
            # Approximate: 1 degree latitude ≈ 111 km
            # 1 degree longitude ≈ 111 km * cos(lat)
            meters_per_degree_lat = 111000.0
            meters_per_degree_lon = 111000.0 * np.cos(np.radians(particle.lat))
            
            dlat = v_total * dt / meters_per_degree_lat
            dlon = u_total * dt / meters_per_degree_lon
            dalt = w_total * dt
            
            # Update particle position
            particle.lat += dlat
            particle.lon += dlon
            particle.altitude += dalt
            
            # Check for ground deposition
            if particle.altitude <= 0:
                particle.altitude = 0
                particle.deposited = True
                particle.deposition_time = self.current_time
            else:
                # Check for dry/wet deposition
                dry_rate, wet_rate = self.atmosphere.get_deposition_rate(
                    particle.diameter,
                    particle.terminal_velocity,
                    particle.altitude
                )
                
                # Probabilistic deposition
                deposition_probability = (dry_rate + wet_rate) * dt
                if np.random.random() < deposition_probability:
                    particle.deposited = True
                    particle.deposition_time = self.current_time
        
        self.current_time += dt
    
    def run(
        self,
        duration: float,
        progress_callback: Optional[callable] = None
    ):
        """
        Run simulation for specified duration.
        
        Args:
            duration: Simulation duration in seconds
            progress_callback: Optional callback function(time, active_count)
        """
        steps = int(duration / self.time_step)
        
        for step_num in range(steps):
            self.step()
            
            if progress_callback and (step_num % 10 == 0):
                active_count = len(self.particle_cloud.get_active_particles())
                progress_callback(self.current_time, active_count)
            
            # Early termination if all particles deposited
            if len(self.particle_cloud.get_active_particles()) == 0:
                break
    
    def generate_output(
        self,
        filename: str,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        resolution: float = 0.01
    ) -> RasterOutput:
        """
        Generate georeferenced raster output.
        
        Args:
            filename: Output filename (without extension)
            bounds: Optional (min_lon, min_lat, max_lon, max_lat)
            resolution: Grid resolution in degrees
        
        Returns:
            RasterOutput object with deposition data
        """
        # Get deposition data
        lats, lons, activities = self.particle_cloud.get_deposition_map()
        
        # Determine bounds if not provided
        if bounds is None:
            if len(lats) > 0:
                # Add margin around deposited particles
                margin = 0.5  # degrees
                bounds = (
                    np.min(lons) - margin,
                    np.min(lats) - margin,
                    np.max(lons) + margin,
                    np.max(lats) + margin
                )
            else:
                # Default bounds around source
                margin = 1.0
                bounds = (
                    self.source_lon - margin,
                    self.source_lat - margin,
                    self.source_lon + margin,
                    self.source_lat + margin
                )
        
        # Create raster output
        raster = RasterOutput(bounds=bounds, resolution=resolution)
        raster.add_deposition_data(lats, lons, activities)
        raster.save_raster(filename)
        
        return raster
    
    def get_statistics(self) -> dict:
        """
        Get simulation statistics.
        
        Returns:
            Dictionary with statistics
        """
        active = self.particle_cloud.get_active_particles()
        deposited = self.particle_cloud.get_deposited_particles()
        
        return {
            "total_particles": len(self.particle_cloud.particles),
            "active_particles": len(active),
            "deposited_particles": len(deposited),
            "simulation_time": self.current_time,
            "fraction_deposited": len(deposited) / len(self.particle_cloud.particles)
        }
