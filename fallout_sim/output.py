"""
Output module for generating georeferenced raster graphics.

Compatible with OPEN-RISOP format (PNG + PGW world files).
"""

import numpy as np
from typing import Tuple, Optional
import warnings


class RasterOutput:
    """
    Generate georeferenced raster output compatible with OPEN-RISOP.
    
    Produces PNG images with PGW world files for GIS compatibility.
    """
    
    def __init__(
        self,
        bounds: Tuple[float, float, float, float],
        resolution: float = 0.01
    ):
        """
        Initialize raster output generator.
        
        Args:
            bounds: (min_lon, min_lat, max_lon, max_lat) in degrees
            resolution: Grid resolution in degrees
        """
        self.min_lon, self.min_lat, self.max_lon, self.max_lat = bounds
        self.resolution = resolution
        
        # Calculate grid dimensions
        self.nx = int((self.max_lon - self.min_lon) / resolution) + 1
        self.ny = int((self.max_lat - self.min_lat) / resolution) + 1
        
        # Initialize deposition grid
        self.grid = np.zeros((self.ny, self.nx))
    
    def add_deposition_data(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        activities: np.ndarray
    ):
        """
        Add deposition data to the raster grid.
        
        Args:
            lats: Array of particle latitudes
            lons: Array of particle longitudes
            activities: Array of particle activities (Bq)
        """
        # Convert positions to grid indices
        i_indices = ((lats - self.min_lat) / self.resolution).astype(int)
        j_indices = ((lons - self.min_lon) / self.resolution).astype(int)
        
        # Filter out-of-bounds particles
        valid = (
            (i_indices >= 0) & (i_indices < self.ny) &
            (j_indices >= 0) & (j_indices < self.nx)
        )
        
        i_indices = i_indices[valid]
        j_indices = j_indices[valid]
        activities = activities[valid]
        
        # Accumulate activities in grid cells
        for i, j, activity in zip(i_indices, j_indices, activities):
            self.grid[i, j] += activity
    
    def save_raster(
        self,
        filename: str,
        dose_rate_conversion: float = 1.0,
        colormap: str = "hot"
    ):
        """
        Save raster as PNG with PGW world file.
        
        Args:
            filename: Output filename (without extension)
            dose_rate_conversion: Conversion factor from Bq to dose rate
            colormap: Matplotlib colormap name
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm
        except ImportError:
            warnings.warn("matplotlib not available, cannot save raster")
            return
        
        # Convert to dose rate (mSv/hr or similar)
        dose_grid = self.grid * dose_rate_conversion
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot with logarithmic scale if there's significant range
        vmin = np.min(dose_grid[dose_grid > 0]) if np.any(dose_grid > 0) else 1e-10
        vmax = np.max(dose_grid) if np.any(dose_grid > 0) else 1.0
        
        if vmax / vmin > 100:
            # Use log scale
            im = ax.imshow(
                np.flipud(dose_grid),
                extent=[self.min_lon, self.max_lon, self.min_lat, self.max_lat],
                cmap=colormap,
                norm=LogNorm(vmin=max(vmin, 1e-10), vmax=vmax),
                interpolation='bilinear'
            )
        else:
            # Linear scale
            im = ax.imshow(
                np.flipud(dose_grid),
                extent=[self.min_lon, self.max_lon, self.min_lat, self.max_lat],
                cmap=colormap,
                vmin=0,
                vmax=vmax,
                interpolation='bilinear'
            )
        
        plt.colorbar(im, ax=ax, label='Dose Rate')
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        ax.set_title('Fallout Deposition Pattern')
        
        # Save PNG
        plt.savefig(f"{filename}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create PGW world file
        self._save_world_file(filename)
    
    def _save_world_file(self, filename: str):
        """
        Save PGW world file for georeferencing.
        
        The world file format has 6 lines:
        1. x-scale (pixel size in x direction)
        2. rotation about y-axis (usually 0)
        3. rotation about x-axis (usually 0)
        4. y-scale (negative pixel size in y direction)
        5. x-coordinate of upper-left pixel center
        6. y-coordinate of upper-left pixel center
        """
        with open(f"{filename}.pgw", 'w') as f:
            # Pixel size in degrees
            f.write(f"{self.resolution}\n")
            f.write("0\n")
            f.write("0\n")
            # Negative because y increases downward in image
            f.write(f"{-self.resolution}\n")
            # Upper-left corner coordinates
            f.write(f"{self.min_lon}\n")
            f.write(f"{self.max_lat}\n")
    
    def get_grid_statistics(self) -> dict:
        """
        Get statistics about the deposition grid.
        
        Returns:
            Dictionary with statistics
        """
        total_activity = np.sum(self.grid)
        max_activity = np.max(self.grid)
        affected_cells = np.sum(self.grid > 0)
        
        return {
            "total_activity_bq": total_activity,
            "max_activity_bq": max_activity,
            "affected_cells": affected_cells,
            "total_cells": self.nx * self.ny,
            "affected_area_km2": affected_cells * (self.resolution * 111.0) ** 2
        }
