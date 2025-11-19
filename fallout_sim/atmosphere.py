"""
Atmospheric model for Lagrangian particle transport.

Provides wind fields and atmospheric conditions for particle advection.
"""

import numpy as np
from typing import Tuple, Optional, Callable


class AtmosphericModel:
    """
    Simple atmospheric model for particle transport.
    
    Provides wind velocity fields and atmospheric conditions.
    """
    
    def __init__(
        self,
        wind_speed: float = 5.0,
        wind_direction: float = 0.0,
        wind_speed_aloft: Optional[float] = None,
        wind_direction_aloft: Optional[float] = None,
        altitude_transition: float = 2000.0,
        custom_wind_field: Optional[Callable] = None
    ):
        """
        Initialize atmospheric model.
        
        Args:
            wind_speed: Surface wind speed (m/s)
            wind_direction: Surface wind direction (degrees, 0=North, 90=East)
            wind_speed_aloft: Wind speed at altitude (m/s), defaults to wind_speed * 2
            wind_direction_aloft: Wind direction at altitude (degrees)
            altitude_transition: Altitude for transition between surface and aloft (m)
            custom_wind_field: Optional custom wind field function(lat, lon, alt, time)
        """
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.wind_speed_aloft = wind_speed_aloft or (wind_speed * 2)
        self.wind_direction_aloft = wind_direction_aloft or wind_direction
        self.altitude_transition = altitude_transition
        self.custom_wind_field = custom_wind_field
    
    def get_wind_velocity(
        self,
        lat: float,
        lon: float,
        altitude: float,
        time: float = 0.0
    ) -> Tuple[float, float, float]:
        """
        Get wind velocity components at a given position and time.
        
        Args:
            lat: Latitude (degrees)
            lon: Longitude (degrees)
            altitude: Altitude (meters)
            time: Simulation time (seconds)
        
        Returns:
            Tuple of (u, v, w) wind components in m/s
            u: eastward component
            v: northward component
            w: vertical component (usually ~0)
        """
        if self.custom_wind_field:
            return self.custom_wind_field(lat, lon, altitude, time)
        
        # Linear interpolation between surface and aloft
        if altitude < 0:
            altitude = 0
        
        blend = min(altitude / self.altitude_transition, 1.0)
        
        # Interpolate wind speed and direction
        speed = self.wind_speed * (1 - blend) + self.wind_speed_aloft * blend
        direction = self.wind_direction * (1 - blend) + self.wind_direction_aloft * blend
        
        # Convert from meteorological convention (direction FROM which wind blows)
        # to velocity components
        direction_rad = np.radians(direction)
        
        # u (eastward), v (northward)
        u = -speed * np.sin(direction_rad)
        v = -speed * np.cos(direction_rad)
        w = 0.0  # No mean vertical wind
        
        return u, v, w
    
    def get_turbulent_velocity(
        self,
        altitude: float,
        dt: float
    ) -> Tuple[float, float, float]:
        """
        Get turbulent velocity components (random walk).
        
        Args:
            altitude: Altitude (meters)
            dt: Time step (seconds)
        
        Returns:
            Tuple of (du, dv, dw) turbulent velocity components
        """
        # Turbulence intensity decreases with altitude
        sigma_h = 1.0 * np.exp(-altitude / 1000)  # horizontal turbulence (m/s)
        sigma_v = 0.5 * np.exp(-altitude / 1000)  # vertical turbulence (m/s)
        
        # Random walk with appropriate time scaling
        du = np.random.normal(0, sigma_h * np.sqrt(dt))
        dv = np.random.normal(0, sigma_h * np.sqrt(dt))
        dw = np.random.normal(0, sigma_v * np.sqrt(dt))
        
        return du, dv, dw
    
    def get_deposition_rate(
        self,
        particle_diameter: float,
        terminal_velocity: float,
        altitude: float
    ) -> Tuple[float, float]:
        """
        Calculate dry and wet deposition rates.
        
        Args:
            particle_diameter: Particle diameter (meters)
            terminal_velocity: Terminal fall velocity (m/s)
            altitude: Current altitude (meters)
        
        Returns:
            Tuple of (dry_deposition_rate, wet_deposition_rate) in 1/s
        """
        # Dry deposition - particles settling out
        # Rate increases as particle approaches ground
        if altitude < 100:
            # Deposition layer
            dry_rate = terminal_velocity / max(altitude, 1.0)
        else:
            dry_rate = 0.0
        
        # Wet deposition (washout by precipitation)
        # Simplified model - constant rate when present
        # In reality, would depend on precipitation rate
        wet_rate = 0.0  # No precipitation by default
        
        return dry_rate, wet_rate
