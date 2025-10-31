
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fallout_sim.py — GPU-accelerated fallout simulator with vertical wind interpolation.

This version is converted from NumPy to CuPy for GPU acceleration.
Core computation arrays are kept on GPU while I/O and plotting use CPU arrays.

This version uses NCEP/NCAR Reanalysis-1 pressure-level winds (uwnd/vwnd NetCDF)
and **interpolates winds with particle altitude** automatically. No ARL needed.

Wind time is taken from the simulation clock, which starts at the earliest
`start_time` in your laydown.csv, and advances in minutes. At each step we use
the **nearest 6-hourly analysis** to the current time (UTC). Each source in the
CSV spawns at its own datetime, or at a single override datetime if specified.

C:\\Dev>python fallout_sim.py test_laydown.csv --uwnd uwnd.2025.nc --vwnd vwnd.2025.nc --hours 24 --extent world --out v6_3

OPEN-RISOP Example:
C:\\Dev>python fallout_sim.py "OPEN-RISOP 1.00 MIXED COUNTERFORCE+COUNTERVALUE ATTACK.xlsx" --uwnd uwnd.2021.nc --vwnd vwnd.2021.nc --hours 48 --extent world --out RISOP_Mixed_Attack --override-datetime 2021-03-15T06:00:00

Inputs (download from PSL):
  https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/pressure/uwnd.YYYY.nc
  https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/pressure/vwnd.YYYY.nc

Run examples:
  python fallout_sim.py laydown.csv --out OUTDIR ^
    --uwnd uwnd.1997.nc --vwnd vwnd.1997.nc --hours 24 --extent world
  
  # Override all source datetimes with a single time:
  python fallout_sim.py laydown.csv --out OUTDIR ^
    --uwnd uwnd.1997.nc --vwnd vwnd.1997.nc --hours 24 --extent world ^
    --override-datetime 1997-06-02T12:00:00
  
  # Use OPEN-RISOP Excel format:
  python fallout_sim.py "OPEN-RISOP 1.00 MIXED ATTACK.xlsx" --out OUTDIR ^
    --uwnd uwnd.1997.nc --vwnd vwnd.1997.nc --hours 24 --extent world ^
    --override-datetime 1997-06-02T12:00:00

Notes:
- Winds vary with particle altitude z via a **standard atmosphere** mapping from
  height (m) -> pressure (hPa), then linear blend between the two bounding
  pressure levels. This is fast and avoids needing temperature/height files.
- You can still adjust the mild power-law scaling with height via WIND_ALPHA.
"""
from __future__ import annotations
import os, sys, math, glob, logging, argparse, time
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from datetime import datetime, timedelta, timezone
from collections import Counter
from scipy.optimize import fsolve

# GPU fallback mechanism
try:
    import cupy as np
    import numpy as numpy_original  # Keep original numpy for datetime64 and other unsupported functions
    HAS_CUPY = True
    GPU_DEVICE_INFO = f"GPU: {np.cuda.runtime.getDeviceCount()} device(s) available"
except ImportError as e:
    import numpy as np
    import numpy as numpy_original  # Both point to numpy when no GPU
    HAS_CUPY = False
    GPU_DEVICE_INFO = f"GPU: Not available ({str(e)})"
except Exception as e:
    import numpy as np
    import numpy as numpy_original
    HAS_CUPY = False
    GPU_DEVICE_INFO = f"GPU: Error during initialization ({str(e)})"

# Global flag to force CPU-only mode via command line
FORCE_CPU_ONLY = False
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# Optional Excel support
try:
    import openpyxl
    HAS_EXCEL_SUPPORT = True
except ImportError:
    HAS_EXCEL_SUPPORT = False
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib as mpl
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.ticker import LogFormatter
import warnings

# Helper functions for numpy/cupy conversion
def to_cupy(arr):
    """Convert numpy array to cupy if needed and GPU is enabled"""
    if FORCE_CPU_ONLY:
        # When CPU-only mode is forced, ensure we return numpy arrays
        if hasattr(arr, 'get'):  # CuPy array
            return arr.get()
        return numpy_original.asarray(arr)
    
    if isinstance(arr, numpy_original.ndarray):
        return np.asarray(arr)
    return arr

def to_numpy(arr):
    """Convert cupy array to numpy if needed"""
    if hasattr(arr, 'get'):  # CuPy array
        return arr.get()
    return arr

warnings.filterwarnings(
    "ignore",
    category=matplotlib.MatplotlibDeprecationWarning
)

# ========================== CONFIGURATION CONSTANTS ==========================

# Metadata storage for output file tracking and processing history
FOOTER_META: Dict[str, dict] = {}

# ==================== GEOGRAPHICAL EXTENTS ====================
# Coordinate bounds for different mapping regions (lon_west, lon_east, lat_south, lat_north)

CONUS_EXTENT = (-125.0, -66.5, 24.0, 49.0)  # Continental United States bounds
WORLD_EXTENT = (-180.0, 180.0, -90.0, 90.0)  # Global extent (full Earth coverage)

# ==================== GRID AND PLOTTING CONFIGURATION ====================

# Hierarchical grid system to balance memory usage and detail
# Coarse global grid: 48x24 cells covering the world
# Fine grid: 4000x4000 cells per coarse cell stored as temporary files
# Total effective resolution: 192,000 x 96,000 = ~18.4 billion grid points
# Memory usage: Only active grids in memory (~64MB for 4000x4000 float32 array)
COARSE_GRID_NXY = (48, 24)  # Global coarse grid (nx, ny)
FINE_GRID_NXY = (4000, 4000)  # Fine grid per coarse cell (nx, ny)
# Effective total resolution: 192,000 x 96,000 points (~0.00188° per fine cell)

# Default plotting extent for all outputs (can be overridden via --extent command line)
PLOT_EXTENT = WORLD_EXTENT  # default plotting extent

# ==================== PHYSICAL CONSTANTS ====================
# Standard atmospheric and Earth constants used in physics calculations

G = 9.80665          # Standard gravitational acceleration [m/s²] - used for settling velocity
R_EARTH = 6371000.0  # Earth radius [m] - used for lat/lon to distance conversions
RHO_AIR = 1.225      # Air density at sea level, 15°C [kg/m³] - used in Stokes settling law
MU_AIR = 1.8e-5      # Dynamic viscosity of air at 15°C [Pa·s] - used in Stokes drag calculation
RHO_PARTICLE = 1500.0  # Particle density [kg/m³] - typical for fallout/ash particles (glass-like)

# ==================== WIND SCALING PARAMETERS ====================

# Power-law wind profile exponent for vertical scaling between surface and release height
# Smaller value (0.06) accounts for vertical level blending already present in reanalysis data
# Formula: wind_at_height = wind_10m * (height/Z_REF)^WIND_ALPHA
# Typical values: 0.1-0.2 for neutral stability, 0.06 accounts for reanalysis smoothing
WIND_ALPHA = 0.06  # smaller than before since we now do level blending

# Reference height for wind power-law scaling [m]
# Standard meteorological reference height for surface wind measurements
Z_REF = 10.0  # m (reference for power-law scaling)

# ==================== TIME STEPPING CONFIGURATION ====================

# Adaptive time stepping strategy: fine resolution early, coarser later for efficiency
# Early phase uses small steps to capture initial rapid dispersion accurately
STEP_SEC_INITIAL = 5   # Time step [seconds] for first 5 minutes (ultra-fine for dispersion)
STEP_MIN_EARLY = 1     # Time step [minutes] from 5 min to first hour (high temporal resolution)
STEP_MIN_LATE  = 6     # Time step [minutes] after first hour (computational efficiency)
STEP_LATE_START_H = 1  # Hour when simulation switches from early to late time stepping
STEP_INITIAL_MIN = 5.0 / 60.0  # 5 minutes in hours - when to switch from seconds to minutes

# ==================== TRANSPORT AND DIFFUSION PARAMETERS ====================

# Horizontal advection scaling factor - accounts for sub-grid turbulence and model limitations
# Value > 1.0 enhances horizontal transport to compensate for coarse reanalysis resolution
# Typical range: 2.0-5.0, calibrated against observations/high-res models
HORIZ_ADVECTION_SCALE = 3.3

# Random walk diffusion strength as fraction of time step displacement
# Simulates sub-grid turbulent diffusion not resolved by reanalysis winds
# Higher values = more spreading, lower values = more concentrated plumes
# Typical range: 0.05-0.15 for atmospheric dispersion
RAND_FRACTION = 0.02  # Increased from 0.06 for more turbulent dispersion

# ==================== BOUNDARY CONDITIONS ====================
# Policy for handling particles that reach domain edges during wind interpolation
# 'clamp': Use nearest valid wind value (conservative, prevents particle loss)
# 'zero': Set wind to zero at edges (can cause artificial accumulation)
EDGE_WIND_POLICY = 'clamp'  # 'clamp' or 'zero'

# ==================== OUTPUT VISUALIZATION PARAMETERS ====================
# Number of Chaikin smoothing iterations for contour polygon generation
# More iterations = smoother polygons but exponentially more vertices
# 0 = no smoothing, 1-2 = moderate smoothing, 3+ = very smooth but slow
POLY_SMOOTH_ITER = 2

# Contour levels for fallout concentration visualization [arbitrary units]
# These values define the boundaries for contour lines and filled regions
# Values should be in ascending order and represent meaningful thresholds
# Note: Values below 100 will increase simulation run time significantly
# particularly for generating shapefiles due to more complex geometries
CONTOUR_LEVELS = numpy_original.array([100, 500, 1000, 3000, 5000], dtype=float)

# ==================== PARTICLE COUNT CONFIGURATION ====================

# Static particle count functions - determine computational particles per source
# Higher counts = better statistical representation but more computation
# These are deliberately simple functions - could be made yield-dependent for realism

def cloud_particle_count(yield_kt: float) -> int: 
    """Number of particles for cloud stem (main fallout mass)
    Args: yield_kt - weapon yield in kilotons (currently unused for simplicity)
    Returns: Fixed particle count for computational efficiency
    """
    return 6000  # Sufficient for statistical accuracy while maintaining performance

def stem_particle_count(yield_kt: float) -> int:
    """Number of particles for stem component (early fallout)
    Args: yield_kt - weapon yield in kilotons (currently unused for simplicity)  
    Returns: Fixed particle count, smaller than cloud due to less mass
    """
    return 1200   # Proportionally smaller than cloud component

# ==================== PARTICLE SIZE DISTRIBUTION PARAMETERS ====================

# Height-dependent particle size distributions for different burst types
# Note: Size distributions depend solely on height of detonation classification, not yield
# This is an intentional model simplification - larger detonations should produce smaller
# particles due to higher energy fluxes, but we're fitting to historical fallout patterns
# rather than creating a rigorous physics model for computational efficiency

# Particle diameter bins [millimeters] - defines discrete size classes for simulation
# Range: 0.03-10.0 mm covers most relevant fallout particles
# Smaller particles (<0.03 mm) remain airborne too long to be computationally practical
# At 24 hours, ~35% of 0.05 mm particles will still be lofted
SIZE_BINS_MM = [ 0.03, 0.05, 0.10, 0.15, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 3.50, 4.00, 4.50, 5.00, 7.50, 10.0 ]

# Probability distributions for each size bin by burst type [fraction, must sum to 1.0]

# Ground burst: Emphasizes larger particles due to surface material incorporation
# Peak at 1.0-1.5 mm with significant contribution from 0.5-3.0 mm range
GROUND_BURST_PROBS = [ 0.00, 0.00, 0.00, 0.00, 0.00, 0.15, 0.15, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05 ]
#GROUND_BURST_PROBS = [ 0.00, 0.00, 0.00,  0.00, 0.00, 0.00, 0.00, 0.00, 0.34, 0.33, 0.33, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00 ]

# Low air burst: Intermediate size distribution between ground and high air bursts
# Peak at 0.15 mm with emphasis on 0.05-0.25 mm range, minimal large particles
LOW_AIR_BURST_PROBS = [ 0.00, 0.00, 0.10, 0.15, 0.20, 0.15, 0.10, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00 ]

# High air burst: Dominated by very fine particles from complete vaporization
# Concentrated in 0.05-0.15 mm range, no particles larger than 0.25 mm
AIR_BURST_PROBS = [ 0.00, 0.25, 0.25, 0.25, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00 ]

# ----------------------------- Data models -----------------------------
@dataclass
class Source:
    ID: str; start_time: datetime; lat: float; lon: float
    yield_kt: float; h_release_km: float; frac: float

@dataclass
class Particle:
    lon: float; lat: float; z: float
    vx: float = 0.0; vy: float = 0.0
    size_m: float = 1e-4
    mass: float = 1.0
    fallout_mass: float = 0.0
    src_id: str = ""
    deposited: bool = False
    w_settle: float = 0.0
    created_at: datetime = None
    landed_at: Optional[datetime] = None
    pol_factor: float = 1.0
    yield_kt: float = 1.0  # Source yield for yield-scaled deposition

# ------------------------- Utility functions -------------------------
def entrain_km(Y): return 0.5 * (Y ** 0.4)

def compute_geometry(src: Source):
    Y = src.yield_kt
    H_TOP = 1.2 * (Y ** 0.40)
    W_MAX = 1.9 * (Y ** 0.40) 
    H_BOTTOM = src.h_release_km + 0.2 * H_TOP
    src.H_TOP_km = H_TOP; src.W_MAX_km = W_MAX
    src.H_BOTTOM_km = H_BOTTOM; src.H_MID_km = 0.5 * H_TOP

    # the altitude where there will be entrainment of fallout debris;
    # note that most models assume pure airbusts will produce negligible 
    # "militarily significant" fallout; we are assuming a low airburst
    # will irradiate ground debris with neutron activation, which will
    # create a radioactive stem or plume which will settle to the ground
    # as fallout
    src.entrain_km = entrain_km(Y); src.stem_radius_km = src.entrain_km
    src.has_stem = src.h_release_km <= src.entrain_km

# distributes the particles randomly in a spheroid representing the stabilized
# mushroom cloud
def random_points_in_spheroid(n, cx_lon, cy_lat, z_center_m, a_vert_m, b_horiz_m, rng):
    if n <= 0: return np.zeros(0), np.zeros(0), np.zeros(0)
    
    # Use Gaussian distribution for concentrated spheroid distribution
    # Increase sigma for more spread - using 0.25 instead of 0.125 for wider distribution
    sigma_horiz = 0.25 * b_horiz_m  # Horizontal standard deviation
    sigma_vert = 0.25 * a_vert_m    # Vertical standard deviation
    
    # Generate points using Gaussian distribution in 3D with better precision
    # Use numpy_original to ensure we get NumPy arrays, then convert to appropriate type
    x_gauss = numpy_original.array(rng.normal(0, sigma_horiz, size=n), dtype=numpy_original.float64)
    y_gauss = numpy_original.array(rng.normal(0, sigma_horiz, size=n), dtype=numpy_original.float64)
    z_gauss = numpy_original.array(rng.normal(0, sigma_vert, size=n), dtype=numpy_original.float64)
    
    # Convert to appropriate array type (np might be cupy or numpy)
    x_gauss = np.array(x_gauss, dtype=np.float64)
    y_gauss = np.array(y_gauss, dtype=np.float64)
    z_gauss = np.array(z_gauss, dtype=np.float64)
    
    # Clip to spheroid boundaries to maintain shape
    x = np.clip(x_gauss, -b_horiz_m, b_horiz_m)
    y = np.clip(y_gauss, -b_horiz_m, b_horiz_m)
    z = np.clip(z_gauss, -a_vert_m, a_vert_m) + z_center_m
    
    # Convert to lat/lon coordinates with full precision
    dlon = (x / (R_EARTH * np.cos(np.deg2rad(cy_lat)))) * 180.0/np.pi
    dlat = (y / R_EARTH) * 180.0/np.pi
    lon = cx_lon + dlon; lat = cy_lat + dlat
    return lon, lat, z

# distributes the particles randomly in a cylinder representing the stem of the
# mushroom cloud
def random_points_in_cylinder(n, base_lon, base_lat, z_bottom_m, height_m, radius_m, rng):
    if n <= 0: return np.zeros(0), np.zeros(0), np.zeros(0)
    
    # Use Gaussian distribution for radial positioning
    # Increase sigma for more spread - using 0.25 instead of 0.125 for wider distribution
    sigma_r = 0.25 * radius_m  # Standard deviation for Gaussian radial distribution
    
    # Generate random values using numpy_original, then convert to appropriate type
    theta_np = numpy_original.array(rng.uniform(0, 2*numpy_original.pi, size=n), dtype=numpy_original.float64)
    r_gauss_np = numpy_original.abs(numpy_original.array(rng.normal(0, sigma_r, size=n), dtype=numpy_original.float64))
    z_rand_np = numpy_original.array(rng.random(size=n), dtype=numpy_original.float64)
    
    # Convert to appropriate array type (np might be cupy or numpy)
    theta = np.array(theta_np, dtype=np.float64)
    r_gauss = np.array(r_gauss_np, dtype=np.float64)
    z_rand = np.array(z_rand_np, dtype=np.float64)
    
    # Clip radial distance to cylinder boundary
    r = np.clip(r_gauss, 0, radius_m)
    
    # Generate heights uniformly within cylinder height
    z = z_bottom_m + z_rand * height_m
    
    # Convert to Cartesian coordinates
    x = r * np.cos(theta); y = r * np.sin(theta)
    
    # Convert to lat/lon offsets with full precision
    dlon = (x / (R_EARTH * np.cos(np.deg2rad(base_lat)))) * 180.0/np.pi
    dlat = (y / R_EARTH) * 180.0/np.pi
    
    lon = base_lon + dlon; lat = base_lat + dlat
    return lon, lat, z

# Cache for settling velocity calculations
_settling_velocity_cache = {}

# Lightweight GPU array cache for frequently used sizes (only when GPU available)
_gpu_array_cache = {}

def _get_gpu_array(size: int, dtype=np.float32) -> np.ndarray:
    """Get a reusable GPU array of given size (CPU fallback if no GPU)."""
    if not HAS_CUPY:
        return np.zeros(size, dtype=dtype)  # CPU array when no GPU
    key = (size, dtype)
    if key not in _gpu_array_cache:
        _gpu_array_cache[key] = np.zeros(size, dtype=dtype)
    return _gpu_array_cache[key]

# determines how fast the particles settle out of the air
def settling_velocity(size_m: float, rho_p: float = RHO_PARTICLE) -> float:
    # Check cache first
    cache_key = (round(size_m * 1e6), rho_p)  # Round to micrometer precision
    if cache_key in _settling_velocity_cache:
        return _settling_velocity_cache[cache_key]
    
    d = max(size_m, 1e-6); w = 0.01
    for _ in range(20):
        Re = max(1e-9, RHO_AIR * w * d / MU_AIR)
        Cd = 24.0/Re + 6.0/(1.0 + math.sqrt(Re)) + 0.4
        w_new = math.sqrt((4.0/3.0) * d * (rho_p - RHO_AIR) * G / (RHO_AIR * Cd))
        if abs(w_new - w) < 1e-4: break
        w = 0.5*w + 0.5*w_new
    
    # Cache the result
    _settling_velocity_cache[cache_key] = w
    return w

def aloft_decay_multiplier(elapsed_min: float) -> float:
    return 1.0
    #t = float(elapsed_min)
    #if t < 10.0:   return 1.0
    #if t < 17.8:   return 0.85 ** ((t - 10.0) / 7.8)
    #if t < 31.7:   return 0.65 * (0.5 ** ((t - 17.8) / 13.9))
    #if t < 60.0:   return 0.45 * (0.5 ** ((t - 31.7) / 28.3))
    #if t < 120.0:  return 0.225
    #if t < 180.0:  return 0.125
    #return 0.03125

def normalize_lonlat(lon: float, lat: float):
    lon_wrapped = ((lon + 180.0) % 360.0) - 180.0
    lat_clamped = max(min(lat, 89.9), -89.9)
    return lon_wrapped, lat_clamped

# ---------------------------- Wind (Reanalysis) ----------------------------
@dataclass
class ReanalGrid3D:
    time: numpy_original.datetime64
    levels_desc: np.ndarray  # hPa, descending (e.g., 1000..100)
    lons: np.ndarray         # (nx,)
    lats: np.ndarray         # (ny, ascending)
    U: np.ndarray            # (nlev, ny, nx) [m/s]
    V: np.ndarray            # (nlev, ny, nx) [m/s]
    dlon: float; dlat: float; lon0: float; lat0: float

class ReanalAccessor:
    def __init__(self, uwnd_path: str, vwnd_path: str):
        self.u_ds = xr.open_dataset(uwnd_path)     # 'uwnd' [m/s], dims (time, level, lat, lon)
        self.v_ds = xr.open_dataset(vwnd_path)     # 'vwnd' [m/s]
        self._levels = self.u_ds['level'].values.astype(float)
        # coordinates
        lats = self.u_ds['lat'].values.astype(float)
        lons = self.u_ds['lon'].values.astype(float)
        # normalize longitude to [-180, 180) - use numpy for coordinate processing
        if numpy_original.nanmax(lons) > 180:
            lons = ((lons + 180.0) % 360.0) - 180.0
            lon_order = numpy_original.argsort(lons)
            self._lons180 = lons[lon_order]
            self._lon_order = lon_order
        else:
            self._lons180 = lons
            self._lon_order = None
        # ensure lat ascending
        if lats[0] > lats[-1]:
            self._lats = lats[::-1]
            self._lat_flip = True
        else:
            self._lats = lats
            self._lat_flip = False
        self._dlon = float(abs(self._lons180[1] - self._lons180[0]))
        self._dlat = float(abs(self._lats[1] - self._lats[0]))
        self.cache: Dict[numpy_original.datetime64, ReanalGrid3D] = {}

    @staticmethod
    def _z_to_pressure_hpa(z_m: np.ndarray) -> np.ndarray:
        """Standard atmosphere (piecewise) approximate z->p (hPa)."""
        if FORCE_CPU_ONLY:
            z = to_numpy(z_m).astype(numpy_original.float64)
            array_lib = numpy_original
        else:
            z = np.asarray(z_m, dtype=np.float64)
            array_lib = np
        
        p0 = 1013.25  # hPa
        T0 = 288.15   # K
        L  = 0.0065   # K/m
        gM_over_R = 34.163195  # K/km exponent helper (g*M/R*1e-3)
        
        # Clip altitude to prevent invalid values in power function
        # Max valid altitude for troposphere formula: z < T0/L ≈ 44.3 km
        z_clipped = array_lib.clip(z, 0.0, 43000.0)
        
        # troposphere (z <= 11 km)
        z_km = z_clipped / 1000.0
        temp_ratio = 1.0 - (L * z_clipped) / T0
        # Ensure temp_ratio stays positive for power function
        temp_ratio = array_lib.maximum(temp_ratio, 1e-10)
        
        p = array_lib.where(
            z_km <= 11.0,
            p0 * array_lib.power(temp_ratio, gM_over_R / L * 1e-3),
            # lower stratosphere 11-20 km (isothermal approx T=216.65K)
            p0 * array_lib.power(1.0 - (L * 11000.0) / T0, gM_over_R / L * 1e-3) *
            array_lib.exp(-(z_clipped - 11000.0) * 9.80665 / (287.05 * 216.65))
        )
        return p.astype(array_lib.float32)

    
    def _nearest_time_key(self, t: datetime) -> numpy_original.datetime64:
        """Return dataset time (numpy_original.datetime64) nearest to UTC timestamp t."""
        # dataset times are numpy datetime64[ns] (tz-naive but UTC-based)
        times = self.u_ds['time'].values  # numpy datetime64 array
        # convert incoming datetime (aware or naive) to UTC numpy datetime64[ns]
        if getattr(t, "tzinfo", None) is not None:
            ts64 = numpy_original.datetime64(pd.Timestamp(t).tz_convert("UTC").to_datetime64())
        else:
            ts64 = numpy_original.datetime64(pd.Timestamp(t).tz_localize("UTC").to_datetime64())
        # choose nearest by absolute difference
        idx = int(numpy_original.argmin(numpy_original.abs(times - ts64)))
        return numpy_original.datetime64(times[idx])

    def grid3d_for_time(self, t: datetime) -> ReanalGrid3D:
        key = self._nearest_time_key(t)
        if key in self.cache: return self.cache[key]
        # load as (level, lat, lon) - xarray returns numpy arrays
        u = self.u_ds['uwnd'].sel(time=key).transpose('level', 'lat', 'lon').values
        v = self.v_ds['vwnd'].sel(time=key).transpose('level', 'lat', 'lon').values
        # reorder lon
        if self._lon_order is not None:
            u = u[:, :, self._lon_order]
            v = v[:, :, self._lon_order]
        # flip lat to ascending
        if self._lat_flip:
            u = u[:, ::-1, :]
            v = v[:, ::-1, :]
        # convert wind data to cupy for GPU acceleration
        u_gpu = np.asarray(u.astype(numpy_original.float32))
        v_gpu = np.asarray(v.astype(numpy_original.float32))
        grid = ReanalGrid3D(
            time=key, levels_desc=numpy_original.array(self._levels.astype(float)), 
            lons=numpy_original.array(self._lons180.copy()), lats=numpy_original.array(self._lats.copy()),
            U=u_gpu, V=v_gpu,
            dlon=self._dlon, dlat=self._dlat, lon0=float(self._lons180[0]), lat0=float(self._lats[0])
        )
        self.cache[key] = grid
        return grid

    def interp_uv_at_alt(self, t: datetime, lons_deg: np.ndarray, lats_deg: np.ndarray, z_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate U,V at particle altitudes z (m):
        1) convert z -> pressure p (hPa) via standard atmosphere
        2) find bracketing pressure levels in the dataset
        3) bilinear interp horizontally at each level and blend vertically
        """
        grid = self.grid3d_for_time(t)
        # 0..N-1 in ascending pressure (for searchsorted)
        lev_desc = grid.levels_desc  # e.g., [1000,925,...,100] - numpy array
        lev_asc = lev_desc[::-1]
        p = self._z_to_pressure_hpa(z_m)  # returns cupy array
        # searchsorted needs both arrays to be same type
        p_cpu = to_numpy(p)
        pos = numpy_original.searchsorted(lev_asc, p_cpu)  # first index with lev_asc[idx] >= p
        pos = numpy_original.clip(pos, 1, len(lev_asc)-1)
        pos = to_cupy(pos)  # convert back to cupy for further processing
        # indices in ascending; convert to descending
        idx_lo_asc = pos      # higher pressure (lower altitude)
        idx_hi_asc = pos - 1  # lower pressure (higher altitude)
        idx_lo = (len(lev_desc) - 1) - idx_lo_asc
        idx_hi = (len(lev_desc) - 1) - idx_hi_asc
        # convert indices to cpu for numpy array access
        idx_lo_cpu = to_numpy(idx_lo)
        idx_hi_cpu = to_numpy(idx_hi)
        
        if FORCE_CPU_ONLY:
            # In CPU-only mode, keep everything as NumPy arrays
            lev_lo = numpy_original.array(lev_desc[idx_lo_cpu], dtype=numpy_original.float32)
            lev_hi = numpy_original.array(lev_desc[idx_hi_cpu], dtype=numpy_original.float32)
            p_for_calc = to_numpy(p)
            w = (p_for_calc - lev_hi) / numpy_original.maximum(lev_lo - lev_hi, 1e-6)
            w = numpy_original.clip(w, 0.0, 1.0).astype(numpy_original.float32)
        else:
            # GPU mode - use CuPy arrays
            lev_lo = to_cupy(lev_desc[idx_lo_cpu])
            lev_hi = to_cupy(lev_desc[idx_hi_cpu])
            w = (p - lev_hi) / np.maximum(lev_lo - lev_hi, 1e-6)
            w = np.clip(w, 0.0, 1.0).astype(np.float32)

        # horizontal bilinear at the two levels
        def bilinear_at_level(Lidx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            # wrap lon into [-180,180)
            if FORCE_CPU_ONLY:
                # Use NumPy for all operations in CPU-only mode
                lons = ((to_numpy(lons_deg) + 180.0) % 360.0) - 180.0
                lats = to_numpy(lats_deg)
                nx = grid.lons.size; ny = grid.lats.size
                fi = (lons - grid.lon0) / grid.dlon
                fj = (lats - grid.lat0) / grid.dlat
                i0 = numpy_original.floor(fi).astype(int); j0 = numpy_original.floor(fj).astype(int)
                di = fi - i0; dj = fj - j0
                i0 = numpy_original.mod(i0, nx); i1 = numpy_original.mod(i0 + 1, nx)
                j0 = numpy_original.clip(j0, 0, ny - 2); j1 = j0 + 1
                # gather corners; use advanced indexing for selected level per particle
                U = grid.U; V = grid.V
                # shape helpers for broadcasting
                Lidx_cpu = to_numpy(Lidx)
                idx = (Lidx_cpu, j0, i0); idx_r = (Lidx_cpu, j0, i1); idx_d = (Lidx_cpu, j1, i0); idx_rd = (Lidx_cpu, j1, i1)
                u00 = to_numpy(U[idx]); u10 = to_numpy(U[idx_r]); u01 = to_numpy(U[idx_d]); u11 = to_numpy(U[idx_rd])
                v00 = to_numpy(V[idx]); v10 = to_numpy(V[idx_r]); v01 = to_numpy(V[idx_d]); v11 = to_numpy(V[idx_rd])
                u = (1-di)*(1-dj)*u00 + di*(1-dj)*u10 + (1-di)*dj*u01 + di*dj*u11
                v = (1-di)*(1-dj)*v00 + di*(1-dj)*v10 + (1-di)*dj*v01 + di*dj*v11
                return u.astype(numpy_original.float32), v.astype(numpy_original.float32)
            else:
                # GPU mode - use CuPy
                lons = ((lons_deg + 180.0) % 360.0) - 180.0
                lats = lats_deg
                nx = grid.lons.size; ny = grid.lats.size
                fi = (lons - grid.lon0) / grid.dlon
                fj = (lats - grid.lat0) / grid.dlat
                i0 = np.floor(fi).astype(int); j0 = np.floor(fj).astype(int)
                di = fi - i0; dj = fj - j0
                i0 = np.mod(i0, nx); i1 = np.mod(i0 + 1, nx)
                j0 = np.clip(j0, 0, ny - 2); j1 = j0 + 1
                # gather corners; use advanced indexing for selected level per particle
                U = grid.U; V = grid.V
                # shape helpers for broadcasting
                idx = (Lidx, j0, i0); idx_r = (Lidx, j0, i1); idx_d = (Lidx, j1, i0); idx_rd = (Lidx, j1, i1)
                u00 = U[idx]; u10 = U[idx_r]; u01 = U[idx_d]; u11 = U[idx_rd]
                v00 = V[idx]; v10 = V[idx_r]; v01 = V[idx_d]; v11 = V[idx_rd]
                u = (1-di)*(1-dj)*u00 + di*(1-dj)*u10 + (1-di)*dj*u01 + di*dj*u11
                v = (1-di)*(1-dj)*v00 + di*(1-dj)*v10 + (1-di)*dj*v01 + di*dj*v11
                return u.astype(np.float32), v.astype(np.float32)

        u_lo, v_lo = bilinear_at_level(idx_lo)
        u_hi, v_hi = bilinear_at_level(idx_hi)
        
        if FORCE_CPU_ONLY:
            # Ensure all arrays are NumPy for consistent operations
            w_cpu = to_numpy(w) if hasattr(w, 'get') else w
            u = w_cpu*u_lo + (1.0 - w_cpu)*u_hi
            v = w_cpu*v_lo + (1.0 - w_cpu)*v_hi
        else:
            # GPU mode
            u = w*u_lo + (1.0 - w)*u_hi
            v = w*v_lo + (1.0 - w)*v_hi
        
        return u, v

# ---------------------------- Size / init ----------------------------
# Cache for burst type logging to prevent duplicate messages
_burst_type_logged = set()

# ---------------------------- Size / init ----------------------------
def build_height_skewed_probs(src: 'Source', for_stem: bool) -> numpy_original.ndarray:
    """
    Choose base size distribution from the new arrays and optionally skew it by height.
    Rules:
      - If src.has_stem == True           -> GROUND_BURST_PROBS
      - Else if h_release_km < 2*entrain  -> LOW_AIR_BURST_PROBS
      - Else                              -> AIR_BURST_PROBS
    Then apply the existing low-release skew toward larger sizes (same logic as before).
    """
    # pick base distribution - use lists for speed, convert to numpy at end
    if getattr(src, "has_stem", False):
        base = GROUND_BURST_PROBS.copy()  # already a list
        # Only log once per source ID to prevent duplicate messages
        if src.ID not in _burst_type_logged:
            logging.info("ID: %s is a ground burst", src.ID)
            _burst_type_logged.add(src.ID)
    elif src.h_release_km < 2.0 * max(entrain_km(src.yield_kt), 1e-6):
        base = LOW_AIR_BURST_PROBS.copy()
        if src.ID not in _burst_type_logged:
            logging.info("ID: %s is a low air burst", src.ID)
            _burst_type_logged.add(src.ID)
    else:
        base = AIR_BURST_PROBS.copy()
        if src.ID not in _burst_type_logged:
            logging.info("ID: %s is an air burst", src.ID)
            _burst_type_logged.add(src.ID)

    # normalize safely - use scalar operations
    s = sum(base)
    if s <= 0:
        # fallback to uniform if misconfigured
        base = [1.0/len(base)] * len(base)
    else:
        base = [x/s for x in base]

    # For ground bursts, use the distribution as-is without height skewing
    # Height skewing was originally intended for air bursts, not ground bursts
    if getattr(src, "has_stem", False):
        return numpy_original.array(base, dtype=float)

    # keep the prior height-based skew (nudges distribution toward larger sizes if low release)
    # Only apply this to air bursts where height matters for particle size
    entr = max(entrain_km(src.yield_kt), 1e-6)
    thresh = 0.5 * entr
    h = max(src.h_release_km, 0.0)
    if h >= thresh:
        return numpy_original.array(base, dtype=float)

    # same gamma formulation as before - use scalar operations where possible
    skew01 = float((thresh - h) / max(thresh, 1e-9))
    gamma = 2.0 * skew01

    # Only use numpy for the final calculation that needs array operations
    sizes_mm = SIZE_BINS_MM  # already a list
    sz_min = min(sizes_mm)
    sz_range = max(sizes_mm) - sz_min + 1e-12
    
    # Vectorized calculation only for the power operation
    sz01 = [(sz - sz_min) / sz_range for sz in sizes_mm]
    weights = [base[i] * ((1e-6 + sz01[i]) ** gamma) for i in range(len(base))]
    
    # normalize weights
    w_sum = sum(weights)
    if w_sum > 0:
        weights = [w/w_sum for w in weights]
    
    return numpy_original.array(weights, dtype=float)

def sample_sizes(src: 'Source', n: int, for_stem: bool) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=float)
    
    probs = build_height_skewed_probs(src, for_stem=for_stem)
    
    # Batch sampling with single choice call to select which bin
    idx = numpy_original.random.choice(len(SIZE_BINS_MM), size=n, p=probs)
    
    # For continuous distribution, sample uniformly within each bin's range
    size_bins_array = numpy_original.array(SIZE_BINS_MM, dtype=float)
    
    # Create bin ranges - from current bin to next bin (or to current bin for last bin)
    sampled_sizes = numpy_original.zeros(n, dtype=float)
    for i in range(len(SIZE_BINS_MM)):
        mask = idx == i
        count = numpy_original.sum(mask)
        if count > 0:
            if i < len(SIZE_BINS_MM) - 1:
                # Sample uniformly between this bin and next bin
                bin_min = SIZE_BINS_MM[i]
                bin_max = SIZE_BINS_MM[i + 1]
                sampled_sizes[mask] = numpy_original.random.uniform(bin_min, bin_max, size=count)
            else:
                # Last bin - use the bin value with small random variation
                bin_val = SIZE_BINS_MM[i]
                sampled_sizes[mask] = bin_val * numpy_original.random.uniform(0.95, 1.05, size=count)
    
    # Convert to appropriate array type and units (mm -> meters)
    sampled_sizes = np.array(sampled_sizes, dtype=float) * 1e-3
    
    return sampled_sizes

def eddy_K(z_m: float) -> float:
    return max(1.0, 10.0 * (1.0 + min(z_m, 2000.0) / 500.0))

def total_prompt_dose(R, W):
    """
    Returns total prompt radiation dose (gamma + neutron) in rads.
    
    Parameters:
        R (float): Slant range in meters
        W (float): Yield in kilotons (kt TNT)
    
    Returns:
        float: Total dose in rad (tissue)
    """
    if R <= 0:
        raise ValueError("Distance R must be > 0")
    
    # Internal separation of components
    D_gamma   = 5.0e5 * W * math.exp(-R / 400.0) / (R ** 2)
    D_neutron = 2.0e5 * W * math.exp(-R / 200.0) / (R ** 2)
    
    return D_gamma + D_neutron

# Attenuation coefficients for air at sea level
# These are linear attenuation coefficients in m^-1
mu_gamma = 0.000035  # m⁻¹  (~1.5 MeV gamma in air)
mu_neut  = 0.000095  # m⁻¹  (fast neutrons in air)

# Dose constants calibrated to give ~4170 rad at 500m for 28kt
# These apply to the reference yield (1 kt equivalent)
# Gamma:neutron ratio approximately 5:2
C_gamma = 2.66e7     # rad·m²/kt (gamma radiation)
C_neut  = 1.06e7     # rad·m²/kt (neutron radiation, tissue dose)

def yield_scaling_factor(W):
    """
    Yield scaling for prompt radiation.
    
    For large yields, the effective radiation output doesn't scale linearly
    because the fireball becomes optically thick and radiation is emitted
    from a larger surface area rather than volume.
    
    Based on empirical data from megaton-range tests:
    - Small yields (< 20 kt): approximately linear
    - Medium yields (20-1000 kt): scales as W^0.33
    - Large yields (> 1000 kt): scales as W^0.20 (very aggressive suppression)
    
    This accounts for:
    - Fireball opacity (radiation from surface only)
    - Atmospheric scattering/absorption over large volumes
    - Geometric dilution effects
    - X-ray re-radiation losses in megaton-range fireballs
    
    Returns effective yield for dose calculations.
    """
    if W <= 20.0:
        return W
    elif W <= 1000.0:
        # Medium yields: W^0.33 scaling
        return 20.0 * (W / 20.0) ** 0.33
    else:
        # Large yields (megaton range): W^0.20 scaling (very suppressed)
        # First scale to 1000 kt, then continue with aggressive scaling
        W_at_1000 = 20.0 * (1000.0 / 20.0) ** 0.33
        return W_at_1000 * (W / 1000.0) ** 0.20

def prompt_dose(R, W, target_dose=None, hob_m=0.0):
    """
    Compute total prompt dose or difference from target.
    
    Parameters
    ----------
    R : float
        Slant range (m)
    W : float
        Yield (kt)
    target_dose : float, optional
        If provided, returns (dose - target); else returns dose.
    hob_m : float, optional
        Height of burst in meters (default: 0.0 for ground burst)
    
    Returns
    -------
    float
    """
    if R <= 0:
        return np.inf if target_dose is None else np.inf
    
    # Use yield scaling for large weapons
    W_eff = yield_scaling_factor(W)
    
    att_gamma = math.exp(-mu_gamma * R)
    att_neut  = math.exp(-mu_neut  * R)
    
    # Apply gamma correction factor for ground bursts
    # Ground bursts have enhanced gamma radiation due to ground reflection/scattering
    gamma_factor = gamma_correction_factor(W) if hob_m == 0.0 else 1.0
    
    D_gamma   = C_gamma * W_eff * att_gamma * gamma_factor / (R**2)
    D_neutron = C_neut  * W_eff * att_neut  / (R**2)
    D_total   = D_gamma + D_neutron
    
    return D_total if target_dose is None else D_total - target_dose

def slant_range_for_dose(W, target_dose, hob_m=0.0, tol=1.0, use_bisect=False):
    """
    Find slant range where total prompt dose = target_dose (rad).
    
    Parameters
    ----------
    W : float
        Yield in kt
    target_dose : float
        Desired dose in rad (e.g. 10, 100, 500)
    hob_m : float, optional
        Height of burst in meters (default: 0.0 for ground burst)
    tol : float
        Distance tolerance in meters (for bisection)
    use_bisect : bool
        Use pure bisection (no scipy) if True
    
    Returns
    -------
    float
        Slant range in meters; 0.0 if target dose not achievable
    """
    if W <= 0 or target_dose <= 0:
        return 0.0

    # Estimate upper bound using effective yield for scaling
    # Account for gamma correction factor in the estimate
    W_eff = yield_scaling_factor(W)
    gamma_factor = gamma_correction_factor(W) if hob_m == 0.0 else 1.0
    R_max_guess = math.sqrt((C_gamma * gamma_factor + C_neut) * W_eff / target_dose) * 5.0
    
    logging.info("Calculating slant range for yield=%.1f kt, target dose=%.1f rad, hob=%.1fm", W, target_dose, hob_m)
    logging.info("Initial bounds: lo=1.0m, hi=%.1fm", R_max_guess)
    
    if use_bisect:
        # --- Bisection method (no scipy) ---
        lo, hi = 1.0, R_max_guess
        for iteration in range(80):  # converges in <80 steps
            mid = (lo + hi) / 2
            actual_dose_at_mid = prompt_dose(mid, W, hob_m=hob_m)
            dose_diff = actual_dose_at_mid - target_dose
            
            if iteration < 5 or iteration % 10 == 0:  # Log first few and every 10th iteration
                logging.debug("Iteration %d: mid=%.1fm, dose=%.1f rad, diff=%.1f rad", 
                             iteration, mid, actual_dose_at_mid, dose_diff)
            
            # If dose_diff > 0: actual > target, we're too close, need to go farther out
            # If dose_diff < 0: actual < target, we're too far, need to come closer in
            if dose_diff > 0:
                lo = mid  # Move lower bound out (search farther ranges)
            else:
                hi = mid  # Move upper bound in (search closer ranges)
            if hi - lo < tol:
                break
        R = lo
        # Debug logging
        actual_dose = prompt_dose(R, W, hob_m=hob_m)
        logging.info("Bisection converged in %d iterations: R=%.1fm, dose=%.1f rad (target=%.1f rad)", 
                     iteration + 1, R, actual_dose, target_dose)
    else:
        # --- fsolve (scipy) ---
        try:
            R, = fsolve(lambda R: prompt_dose(R[0], W, target_dose, hob_m=hob_m), x0=R_max_guess/2)
        except:
            return 0.0

    # Final validation - check if dose is within reasonable range of target
    actual_dose = prompt_dose(R, W, hob_m=hob_m)
    error_pct = abs(actual_dose - target_dose) / target_dose * 100
    # Accept if within 10% of target (either direction)
    if abs(actual_dose - target_dose) <= target_dose * 0.10:
        logging.info("Range validation PASSED: R=%.1fm gives dose=%.1f rad (target=%.1f rad, error=%.1f%%)", 
                     R, actual_dose, target_dose, error_pct)
        return R
    else:
        logging.warning("Range validation FAILED: R=%.1fm gives dose=%.1f rad (target=%.1f rad, error=%.1f%%)", 
                       R, actual_dose, target_dose, error_pct)
        return 0.0

def gamma_correction_factor(yield_kt: float) -> float:
    Fmax = 3.16
    Y0 = 1762.914118095948
    n = 1.4916666666666667
    Y = max(float(yield_kt), 0.0)
    F = 1.0 + (Fmax - 1.0) * (Y**n) / (Y**n + Y0**n)
    return F

def init_particles_for_source(src: Source, rng: numpy_original.random.Generator, gpu_rng: Optional[object] = None) -> List['Particle']:
    compute_geometry(src)
    parts: List[Particle] = []
    cp = cloud_particle_count(src.yield_kt)
    base_stem = stem_particle_count(src.yield_kt)
    extra_stem = base_stem if src.has_stem else 0

    a_vert_m = max((src.H_TOP_km - src.H_BOTTOM_km) * 0.5 * 1000.0, 1.0)
    z_center_m = (src.H_TOP_km + src.H_BOTTOM_km) * 0.5 * 1000.0
    b_horiz_m = (src.W_MAX_km * 0.0675) * 1000.0 # scaling factor to reduce horizontal spread
    # Use GPU RNG for position sampling where possible so we get cupy arrays when needed
    cloud_lons, cloud_lats, cloud_z = random_points_in_spheroid(cp, src.lon, src.lat, z_center_m, a_vert_m, b_horiz_m, gpu_rng or rng)
    cloud_sizes = sample_sizes(src, cp, for_stem=False)

    stem_lons = stem_lats = stem_z = stem_sizes = np.array([])
    if extra_stem > 0:
        radius_m = max(src.stem_radius_km * 1000.0, 1.0)
        z_bottom_m = 0.0
        # Extend stem height to top of spheroid for full overlap
        height_m = max(src.H_TOP_km * 1000.0, 1.0)
        stem_lons, stem_lats, stem_z = random_points_in_cylinder(extra_stem, src.lon, src.lat, z_bottom_m, height_m, radius_m, gpu_rng or rng)
        stem_sizes = sample_sizes(src, extra_stem, for_stem=True)

    if src.H_BOTTOM_km < src.entrain_km and cp > 0:
        k = int(0.25 * cp)
        radius_m = max(src.stem_radius_km * 1000.0, 1.0)
        z_bottom_m = 0.0; height_m = max(src.H_TOP_km * 1000.0, 1.0)
        llon, llat, lz = random_points_in_cylinder(k, src.lon, src.lat, z_bottom_m, height_m, radius_m, gpu_rng or rng)
        cloud_lons[:k], cloud_lats[:k], cloud_z[:k] = llon, llat, lz

    total_particles = int(cp + int(extra_stem))
    total_fallout_mass = 5650.0 * (float(src.yield_kt) ** 0.8)
    mass_per_particle = (total_fallout_mass / max(1, total_particles))

    parts = []
    
    # Batch process all particles at once with vectorized operations
    all_lons = []
    all_lats = []
    all_zs = []
    all_sizes = []
    
    # Collect cloud particles
    if cp > 0:
        cloud_lons_cpu = to_numpy(cloud_lons) if hasattr(cloud_lons, 'get') else cloud_lons
        cloud_lats_cpu = to_numpy(cloud_lats) if hasattr(cloud_lats, 'get') else cloud_lats  
        cloud_z_cpu = to_numpy(cloud_z) if hasattr(cloud_z, 'get') else cloud_z
        cloud_sizes_cpu = to_numpy(cloud_sizes) if hasattr(cloud_sizes, 'get') else cloud_sizes
        
        all_lons.extend(cloud_lons_cpu)
        all_lats.extend(cloud_lats_cpu)
        all_zs.extend(cloud_z_cpu)
        all_sizes.extend(cloud_sizes_cpu)
    
    # Collect stem particles
    if extra_stem > 0:
        stem_lons_cpu = to_numpy(stem_lons) if hasattr(stem_lons, 'get') else stem_lons
        stem_lats_cpu = to_numpy(stem_lats) if hasattr(stem_lats, 'get') else stem_lats
        stem_z_cpu = to_numpy(stem_z) if hasattr(stem_z, 'get') else stem_z
        stem_sizes_cpu = to_numpy(stem_sizes) if hasattr(stem_sizes, 'get') else stem_sizes
        
        all_lons.extend(stem_lons_cpu)
        all_lats.extend(stem_lats_cpu)
        all_zs.extend(stem_z_cpu)
        all_sizes.extend(stem_sizes_cpu)
    
    # Vectorized settling velocity calculation for all particles at once
    if all_sizes:
        w_settle_batch = [settling_velocity(float(d)) for d in all_sizes]
        
        # Batch particle creation with pre-computed values
        for i in range(len(all_lons)):
            parts.append(Particle(
                lon=float(all_lons[i]), lat=float(all_lats[i]), z=float(all_zs[i]),
                size_m=float(all_sizes[i]), mass=src.frac, fallout_mass=mass_per_particle,
                src_id=src.ID, w_settle=w_settle_batch[i], created_at=src.start_time,
                yield_kt=float(src.yield_kt)))  # Store source yield for deposition scaling
    
    return parts

# ---------------------------- Advection step ----------------------------
# Wind interpolation cache for repeated time steps
_wind_interp_cache = {}

def compute_wind_displacements_for_particles_reanal(particles: List[Particle], winds: ReanalAccessor, when_t: datetime, dt_s: float) -> Tuple[np.ndarray, np.ndarray]:
    n = len(particles)
    if n == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    # Use GPU only for large particle counts AND when GPU is available AND not forced to CPU
    use_gpu = HAS_CUPY and not FORCE_CPU_ONLY and n > 1000
    
    # Log computation path (debug level to avoid spam)
    if n > 0:
        if use_gpu:
            logging.debug("Using GPU acceleration for %d particles", n)
        elif FORCE_CPU_ONLY:
            logging.debug("Using CPU path for %d particles (forced by --no-gpu)", n)
        elif not HAS_CUPY:
            logging.debug("Using CPU path for %d particles (no GPU available)", n)
        else:
            logging.debug("Using CPU path for %d particles (below GPU threshold)", n)
    
    if use_gpu:
        # GPU path for large arrays with caching
        if n <= 10000:  # Use cache for moderate sizes
            lons = _get_gpu_array(n, np.float32)[:n]
            lats = _get_gpu_array(n, np.float32)[:n] 
            zs = _get_gpu_array(n, np.float32)[:n]
            
            # Fill arrays manually for small-medium arrays
            for i, p in enumerate(particles):
                lons[i] = p.lon
                lats[i] = p.lat
                zs[i] = p.z
        else:
            # Direct array creation for very large arrays
            lons = np.array([p.lon for p in particles], dtype=np.float32)
            lats = np.array([p.lat for p in particles], dtype=np.float32)
            zs = np.array([p.z for p in particles], dtype=np.float32)
        
        u, v = winds.interp_uv_at_alt(when_t, lons, lats, zs)
        
        # Optional mild height scaling + global scaling
        s = np.power(np.maximum(zs, Z_REF) / Z_REF, WIND_ALPHA, dtype=np.float32)
        u = u * s * HORIZ_ADVECTION_SCALE
        v = v * s * HORIZ_ADVECTION_SCALE
        
        dx_w = u * dt_s
        dy_w = v * dt_s
    else:
        # CPU path for small arrays (less overhead)
        import numpy as cpu_np
        lons = cpu_np.array([p.lon for p in particles], dtype=cpu_np.float32)
        lats = cpu_np.array([p.lat for p in particles], dtype=cpu_np.float32)
        zs = cpu_np.array([p.z for p in particles], dtype=cpu_np.float32)
        
        # Wind interpolation still happens on GPU (wind data is large)
        u, v = winds.interp_uv_at_alt(when_t, to_cupy(lons), to_cupy(lats), to_cupy(zs))
        u = to_numpy(u); v = to_numpy(v)  # Convert back to CPU
        
        # CPU-based scaling calculations
        s = cpu_np.power(cpu_np.maximum(zs, Z_REF) / Z_REF, WIND_ALPHA).astype(cpu_np.float32)
        u = u * s * HORIZ_ADVECTION_SCALE
        v = v * s * HORIZ_ADVECTION_SCALE
        
        dx_w = u * dt_s
        dy_w = v * dt_s
        
        # Convert back to CuPy for compatibility
        dx_w = to_cupy(dx_w)
        dy_w = to_cupy(dy_w)
    
    return dx_w, dy_w

# ---------------------------- Output helpers ----------------------------
def _mm(size_m: float) -> float: return round(1000.0 * float(size_m), 3)

def _size_bucket(size_m: float) -> str:
    """Assign a particle size to a bucket based on SIZE_BINS_MM ranges"""
    size_mm = size_m * 1000.0
    for i in range(len(SIZE_BINS_MM) - 1):
        if size_mm < SIZE_BINS_MM[i + 1]:
            return f"{SIZE_BINS_MM[i]:.2f}-{SIZE_BINS_MM[i+1]:.2f} mm"
    # Last bucket or beyond
    return f"{SIZE_BINS_MM[-1]:.2f}+ mm"

def _log_initial_size_hist(initial_hist: Counter, total_init: int):
    if total_init <= 0:
        logging.info("Initial lofted particles by size: none"); return
    lines = ["Initial lofted particles by size buckets (mm):"]
    for sz_bucket, cnt in sorted(initial_hist.items()):
        lines.append(f"  {sz_bucket} : {cnt}")
    lines.append(f"TOTAL initial lofted: {total_init}")
    logging.info("\n".join(lines))

def _log_remaining_size_hist(active_particles: List['Particle'], initial_hist: Counter):
    rem_hist = Counter(_size_bucket(p.size_m) for p in active_particles if not p.deposited)
    total_rem = sum(rem_hist.values())
    lines = ["Remaining lofted particles by size buckets (mm):"]
    total_init = sum(initial_hist.values())
    for sz_bucket, init_cnt in sorted(initial_hist.items()):
        rem_cnt = rem_hist.get(sz_bucket, 0)
        pct = (100.0 * rem_cnt / init_cnt) if init_cnt > 0 else 0.0
        lines.append(f"  {sz_bucket} : {rem_cnt}  ({pct:.1f}% of initial)")
    lines.append(f"TOTAL remaining lofted: {total_rem} ({(100.0*total_rem/max(1,total_init)):.1f}% of initial)")
    logging.info("\n".join(lines))

def _make_map_ax(extent):
    is_world = (extent == WORLD_EXTENT)
    if is_world:
        proj = ccrs.Robinson()
        fig = plt.figure(figsize=(40, 22), dpi=100)
        ax = plt.axes(projection=proj)
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.6)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.5)
        ax.gridlines(draw_labels=False, linewidth=0.25, linestyle=':')
    else:
        proj = ccrs.LambertConformal(central_longitude=-96, central_latitude=39, standard_parallels=(33, 45))
        fig = plt.figure(figsize=(40, 40), dpi=100)
        ax = plt.axes(projection=proj)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.7)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.7)
        ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.4)
        ax.gridlines(draw_labels=False, linewidth=0.3, linestyle=':')
    return fig, ax, is_world

# ==================== HIERARCHICAL GRID MANAGEMENT ====================

class HierarchicalGrid:
    """Manages hierarchical grid system with coarse global grid and fine tile grids"""
    
    def __init__(self, outdir: str, extent=WORLD_EXTENT):
        self.outdir = outdir
        self.extent = extent
        self.coarse_nx, self.coarse_ny = COARSE_GRID_NXY
        self.fine_nx, self.fine_ny = FINE_GRID_NXY
        self.temp_dir = os.path.join(outdir, "temp_grids")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Calculate coarse grid cell boundaries
        lon_w, lon_e, lat_s, lat_n = extent
        self.coarse_dlon = (lon_e - lon_w) / self.coarse_nx
        self.coarse_dlat = (lat_n - lat_s) / self.coarse_ny
        self.lon_w, self.lon_e = lon_w, lon_e
        self.lat_s, self.lat_n = lat_s, lat_n
        
        # Cache for loaded fine grids
        self._grid_cache = {}
        self._max_cache_size = 16  # Maximum number of fine grids to keep in memory
    
    def get_coarse_indices(self, lon: float, lat: float) -> tuple:
        """Get coarse grid indices for given lon/lat"""
        fi = (lon - self.lon_w) / (self.lon_e - self.lon_w) * self.coarse_nx
        fj = (lat - self.lat_s) / (self.lat_n - self.lat_s) * self.coarse_ny
        ci = int(np.clip(np.floor(fi), 0, self.coarse_nx - 1))
        cj = int(np.clip(np.floor(fj), 0, self.coarse_ny - 1))
        return ci, cj
    
    def get_fine_grid_path(self, ci: int, cj: int) -> str:
        """Get file path for fine grid at coarse indices (ci, cj)"""
        return os.path.join(self.temp_dir, f"{ci:02d}x{cj:02d}y.tmp")
    
    def get_coarse_cell_extent(self, ci: int, cj: int) -> tuple:
        """Get geographic extent of coarse cell (ci, cj)"""
        lon_w = self.lon_w + ci * self.coarse_dlon
        lon_e = self.lon_w + (ci + 1) * self.coarse_dlon
        lat_s = self.lat_s + cj * self.coarse_dlat
        lat_n = self.lat_s + (cj + 1) * self.coarse_dlat
        return lon_w, lon_e, lat_s, lat_n
    
    def load_fine_grid(self, ci: int, cj: int) -> np.ndarray:
        """Load or create fine grid for coarse cell (ci, cj)"""
        key = (ci, cj)
        
        # Check cache first
        if key in self._grid_cache:
            return self._grid_cache[key]
        
        # Manage cache size
        if len(self._grid_cache) >= self._max_cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._grid_cache))
            self._save_fine_grid(oldest_key[0], oldest_key[1], self._grid_cache[oldest_key])
            del self._grid_cache[oldest_key]
        
        # Try to load from disk (numpy.save adds .npy extension)
        grid_path = self.get_fine_grid_path(ci, cj)
        npy_path = grid_path + '.npy'
        if os.path.exists(npy_path):
            try:
                grid_data = numpy_original.load(npy_path)
                grid = to_cupy(grid_data)  # Convert to appropriate array type
                if grid.shape != (self.fine_nx, self.fine_ny):
                    # Invalid shape, create new
                    grid = np.zeros((self.fine_nx, self.fine_ny), dtype=np.float32)
            except Exception:
                # Corrupted file, create new
                grid = np.zeros((self.fine_nx, self.fine_ny), dtype=np.float32)
        else:
            # Create new grid
            grid = np.zeros((self.fine_nx, self.fine_ny), dtype=np.float32)
        
        # Store in cache
        self._grid_cache[key] = grid
        return grid
    
    def _save_fine_grid(self, ci: int, cj: int, grid: np.ndarray):
        """Save fine grid to disk"""
        grid_path = self.get_fine_grid_path(ci, cj)
        grid_cpu = to_numpy(grid)  # Ensure it's a CPU array
        numpy_original.save(grid_path, grid_cpu)
    
    def flush_cache(self):
        """Save all cached grids to disk and clear cache"""
        for (ci, cj), grid in self._grid_cache.items():
            self._save_fine_grid(ci, cj, grid)
        self._grid_cache.clear()
    
    def deposit_particles(self, deposited: List[Particle]):
        """Deposit particles into appropriate fine grids"""
        if not deposited:
            return
        
        # Group particles by coarse grid cell
        particles_by_cell = {}
        for p in deposited:
            ci, cj = self.get_coarse_indices(p.lon, p.lat)
            if (ci, cj) not in particles_by_cell:
                particles_by_cell[(ci, cj)] = []
            particles_by_cell[(ci, cj)].append(p)
        
        # Process each coarse cell
        for (ci, cj), particles in particles_by_cell.items():
            fine_grid = self.load_fine_grid(ci, cj)
            cell_extent = self.get_coarse_cell_extent(ci, cj)
            
            # Deposit particles into fine grid
            self._deposit_to_fine_grid(particles, fine_grid, cell_extent)
    
    def _deposit_to_fine_grid(self, particles: List[Particle], grid: np.ndarray, extent: tuple):
        """Deposit particles into a single fine grid using yield-scaled spatial distribution"""
        if not particles:
            return
            
        lon_w, lon_e, lat_s, lat_n = extent
        nx, ny = grid.shape
        
        lons = np.array([normalize_lonlat(p.lon, p.lat)[0] for p in particles], dtype=np.float64)
        lats = np.array([normalize_lonlat(p.lon, p.lat)[1] for p in particles], dtype=np.float64)
        masses = 24 * np.array([getattr(p, 'fallout_mass', 0.0) * getattr(p, 'pol_factor', 1.0) for p in particles], dtype=np.float32)
        
        # Get yields for each particle (from source)
        yields = np.array([getattr(p, 'yield_kt', 1.0) for p in particles], dtype=np.float32)
        
        # Calculate latitude correction factor: 1/cos(lat)
        # At higher latitudes, longitude degrees represent shorter physical distances,
        # so we need more grid cells to cover the same physical area
        # Cap at 85° latitude to avoid extreme values near poles
        lats_capped = np.clip(lats, -85.0, 85.0)
        lat_correction = 1.0 / np.maximum(np.cos(np.radians(lats_capped)), 0.1)
        
        # Calculate deposit square size based on yield^0.4 and latitude correction
        # This gives the half-width in grid cells to spread deposition
        # Minimum 1 cell, scales up with yield and latitude
        base_half_width = yields ** 0.4
        deposit_half_widths = np.maximum(1, np.round(base_half_width * lat_correction).astype(np.int32))
        
        # Calculate continuous grid coordinates for particle centers
        fi = (lons - lon_w) / (lon_e - lon_w) * (nx - 1)
        fj = (lats - lat_s) / (lat_n - lat_s) * (ny - 1)
        
        # Get center cell coordinates
        i_center = np.round(fi).astype(np.int64)
        j_center = np.round(fj).astype(np.int64)
        
        # Clip centers to valid range
        i_center = np.clip(i_center, 0, nx - 1)
        j_center = np.clip(j_center, 0, ny - 1)
        
        # Convert everything to CPU numpy for efficient processing
        i_center_cpu = to_numpy(i_center)
        j_center_cpu = to_numpy(j_center)
        deposit_hw_cpu = to_numpy(deposit_half_widths)
        masses_cpu = to_numpy(masses)
        
        # Handle grid conversion
        if hasattr(grid, 'get'):
            grid_cpu = grid.get()
            use_gpu = True
        else:
            grid_cpu = numpy_original.asarray(grid)
            use_gpu = False
        
        # Vectorized deposition using numpy operations
        # Group particles by their half-width for efficient batch processing
        unique_hws = numpy_original.unique(deposit_hw_cpu)
        
        for hw in unique_hws:
            # Find all particles with this half-width
            mask = deposit_hw_cpu == hw
            if not numpy_original.any(mask):
                continue
            
            ic_batch = i_center_cpu[mask]
            jc_batch = j_center_cpu[mask]
            mass_batch = masses_cpu[mask]
            
            # Number of cells in square: (2*hw + 1)^2
            n_cells = (2 * hw + 1) ** 2
            mass_per_cell_batch = mass_batch / n_cells
            
            # For each offset in the square, deposit all particles at once
            for di in range(-hw, hw + 1):
                for dj in range(-hw, hw + 1):
                    i_deposit = numpy_original.clip(ic_batch + di, 0, nx - 1)
                    j_deposit = numpy_original.clip(jc_batch + dj, 0, ny - 1)
                    
                    # Vectorized deposit using add.at
                    numpy_original.add.at(grid_cpu, (i_deposit, j_deposit), mass_per_cell_batch)
        
        # Copy back to GPU if needed
        if use_gpu:
            import cupy as cp
            grid[:] = cp.asarray(grid_cpu)
    
    def add_prompt_radiation(self, source: 'Source'):
        """Add prompt radiation dose to grid cells near ground zero
        
        Args:
            source: Source object with yield_kt, lon, lat, h_release_km
        """
        # Get minimum contour level to determine radius
        min_dose = float(numpy_original.min(CONTOUR_LEVELS))
        hob_m = source.h_release_km * 1000.0  # Height of burst in meters
        
        # Calculate maximum radius where prompt dose >= min_dose
        max_radius_m = slant_range_for_dose(source.yield_kt, min_dose, hob_m=hob_m, use_bisect=True)
        
        logging.info("slant_range_for_dose returned: %.1f m for yield=%.1f kt, min_dose=%.1f rad, hob=%.1fm",
                     max_radius_m, source.yield_kt, min_dose, hob_m)
        
        # Debug: verify the calculation
        if max_radius_m > 0:
            test_dose = prompt_dose(max_radius_m, source.yield_kt, hob_m=hob_m)
            logging.info("Verification: dose at %.1fm = %.1f rad (target was %.1f rad)",
                         max_radius_m, test_dose, min_dose)
        
        if max_radius_m <= 0:
            logging.debug("No prompt radiation effects for yield %.1f kt at dose threshold %.1f rad", 
                         source.yield_kt, min_dose)
            return
        
        logging.info("Adding prompt radiation for source %s (yield=%.1f kt, max_radius=%.0f m, min_dose=%.1f rad)",
                     source.ID, source.yield_kt, max_radius_m, min_dose)
        
        gz_lon, gz_lat = source.lon, source.lat
        earth_r = R_EARTH
        
        # Calculate geographic extent of affected area
        angular_radius = max_radius_m / earth_r
        dlat = numpy_original.degrees(angular_radius)
        dlon = numpy_original.degrees(angular_radius / numpy_original.cos(numpy_original.radians(gz_lat)))
        
        affected_lon_w = max(gz_lon - dlon, self.lon_w)
        affected_lon_e = min(gz_lon + dlon, self.lon_e)
        affected_lat_s = max(gz_lat - dlat, self.lat_s)
        affected_lat_n = min(gz_lat + dlat, self.lat_n)
        
        # Find affected coarse cells
        ci_min = max(0, min(int(numpy_original.floor((affected_lon_w - self.lon_w) / self.coarse_dlon)), self.coarse_nx - 1))
        ci_max = max(0, min(int(numpy_original.ceil((affected_lon_e - self.lon_w) / self.coarse_dlon)), self.coarse_nx))
        cj_min = max(0, min(int(numpy_original.floor((affected_lat_s - self.lat_s) / self.coarse_dlat)), self.coarse_ny - 1))
        cj_max = max(0, min(int(numpy_original.ceil((affected_lat_n - self.lat_s) / self.coarse_dlat)), self.coarse_ny))
        
        total_cells_modified = 0
        
        # Pre-calculate constants for dose calculation
        gz_lat_rad = numpy_original.radians(gz_lat)
        cos_gz_lat = numpy_original.cos(gz_lat_rad)
        
        # Pre-compute yield scaling and gamma correction (outside loops!)
        W_eff = yield_scaling_factor(source.yield_kt)
        gamma_factor = gamma_correction_factor(source.yield_kt) if hob_m == 0.0 else 1.0
        C_gamma_eff = C_gamma * W_eff * gamma_factor
        C_neut_eff = C_neut * W_eff
        
        # Debug: calculate approximate fine cell size
        coarse_cell_width_deg = self.coarse_dlon
        coarse_cell_width_m = coarse_cell_width_deg * 111320 * numpy_original.cos(gz_lat_rad)  # approx meters
        fine_cell_width_m = coarse_cell_width_m / self.fine_nx
        logging.debug("Fine cell size: ~%.1f m (max_radius: %.1f m)", fine_cell_width_m, max_radius_m)
        
        # Process each affected coarse cell with vectorized operations
        for ci in range(ci_min, ci_max):
            for cj in range(cj_min, cj_max):
                # Quick check: is this coarse cell center too far away?
                # If yes, skip loading the fine grid entirely
                coarse_cell_lon = self.lon_w + (ci + 0.5) * self.coarse_dlon
                coarse_cell_lat = self.lat_s + (cj + 0.5) * self.coarse_dlat
                
                # Rough distance check (cheaper than loading grid)
                dlat_check = numpy_original.radians(coarse_cell_lat - gz_lat)
                dlon_check = numpy_original.radians(coarse_cell_lon - gz_lon)
                a_check = (numpy_original.sin(dlat_check / 2) ** 2 + 
                          cos_gz_lat * numpy_original.cos(numpy_original.radians(coarse_cell_lat)) * 
                          numpy_original.sin(dlon_check / 2) ** 2)
                c_check = 2 * numpy_original.arctan2(numpy_original.sqrt(a_check), numpy_original.sqrt(1 - a_check))
                coarse_dist = earth_r * c_check
                
                # If coarse cell center is more than (radius + diagonal of coarse cell) away, skip
                coarse_diagonal = coarse_cell_width_m * 1.5  # Conservative estimate
                if coarse_dist > max_radius_m + coarse_diagonal:
                    continue
                
                fine_grid = self.load_fine_grid(ci, cj)
                cell_extent = self.get_coarse_cell_extent(ci, cj)
                cell_lon_w, cell_lon_e, cell_lat_s, cell_lat_n = cell_extent
                
                # Calculate fine cell dimensions
                fine_dlon = (cell_lon_e - cell_lon_w) / self.fine_nx
                fine_dlat = (cell_lat_n - cell_lat_s) / self.fine_ny
                
                # Use float32 for speed (sufficient precision for radiation calculations)
                # Vectorized: create coordinate arrays for all cells in this coarse grid
                fi_array = numpy_original.arange(self.fine_nx, dtype=numpy_original.float32)
                fj_array = numpy_original.arange(self.fine_ny, dtype=numpy_original.float32)
                fi_grid, fj_grid = numpy_original.meshgrid(fi_array, fj_array, indexing='ij')
                
                # Calculate all cell center coordinates at once
                cell_lons = numpy_original.float32(cell_lon_w) + (fi_grid + 0.5) * numpy_original.float32(fine_dlon)
                cell_lats = numpy_original.float32(cell_lat_s) + (fj_grid + 0.5) * numpy_original.float32(fine_dlat)
                
                # Simplified flat-earth approximation for small areas (much faster than Haversine)
                # Valid for distances < 100 km (typical prompt radiation ranges)
                # Convert to meters using approximate degree-to-meter conversion
                dx_m = (cell_lons - gz_lon) * 111320.0 * cos_gz_lat  # meters east-west
                dy_m = (cell_lats - gz_lat) * 111320.0  # meters north-south
                ground_dist_m = numpy_original.sqrt(dx_m**2 + dy_m**2)
                
                # Calculate slant ranges for all cells
                slant_ranges = numpy_original.sqrt(ground_dist_m ** 2 + hob_m ** 2)
                
                # Create mask for cells within max radius
                within_radius = slant_ranges <= max_radius_m
                
                if not numpy_original.any(within_radius):
                    continue  # No cells in this coarse grid are affected
                
                # Vectorized dose calculation for cells within radius
                # Calculate attenuation factors (avoid division by zero)
                slant_ranges_safe = numpy_original.maximum(slant_ranges, 1.0)
                att_gamma = numpy_original.exp(-mu_gamma * slant_ranges_safe)
                att_neut = numpy_original.exp(-mu_neut * slant_ranges_safe)
                
                # Calculate dose components using pre-computed effective constants
                inv_r_sq = numpy_original.float32(1.0) / (slant_ranges_safe ** 2)
                D_gamma = C_gamma_eff * att_gamma * inv_r_sq
                D_neutron = C_neut_eff * att_neut * inv_r_sq
                dose_grid = D_gamma + D_neutron
                
                # Only apply dose where within radius AND dose >= min_dose
                # (no point adding negligible doses)
                significant_dose = (within_radius) & (dose_grid >= min_dose)
                dose_grid = numpy_original.where(significant_dose, dose_grid, 0.0)
                
                # Add to grid (handle NumPy vs CuPy)
                if hasattr(fine_grid, 'get'):  # CuPy array
                    fine_grid_cpu = fine_grid.get()
                    fine_grid_cpu += dose_grid.astype(numpy_original.float32)
                    # Copy back
                    fine_grid[:] = np.array(fine_grid_cpu)
                else:  # NumPy array
                    fine_grid += dose_grid.astype(numpy_original.float32)
                
                total_cells_modified += int(numpy_original.sum(significant_dose))
        
        logging.info("Prompt radiation added to %d fine grid cells (min dose: %.1f rad)", 
                     int(total_cells_modified), min_dose)
    
    def assemble_full_grid_for_extent(self, target_extent=None, max_output_size=8000) -> np.ndarray:
        """Assemble full grid from fine grids for given extent with edge handling and smart downsampling"""
        if target_extent is None:
            target_extent = self.extent
        
        # Determine which coarse cells intersect with target extent
        target_cells = self._get_intersecting_cells(target_extent)
        
        if not target_cells:
            # No data in target extent
            return numpy_original.zeros((100, 100), dtype=numpy_original.float32)
        
        # Calculate output grid dimensions based on target extent and fine grid resolution
        lon_w, lon_e, lat_s, lat_n = target_extent
        
        # Calculate effective resolution per fine cell
        fine_dlon = self.coarse_dlon / self.fine_nx
        fine_dlat = self.coarse_dlat / self.fine_ny
        
        # Calculate initial output size
        out_nx = int(numpy_original.ceil((lon_e - lon_w) / fine_dlon))
        out_ny = int(numpy_original.ceil((lat_n - lat_s) / fine_dlat))
        
        # Aggressive downsampling for performance - prioritize speed over resolution for large areas
        if out_nx > max_output_size or out_ny > max_output_size:
            scale_factor = max(out_nx / max_output_size, out_ny / max_output_size)
            out_nx = int(out_nx / scale_factor)
            out_ny = int(out_ny / scale_factor) 
            fine_dlon *= scale_factor
            fine_dlat *= scale_factor
            
        # Additional safety cap
        out_nx = min(out_nx, max_output_size)
        out_ny = min(out_ny, max_output_size)
        
        output_grid = numpy_original.zeros((out_nx, out_ny), dtype=numpy_original.float32)
        
        # Flush cache to save memory for assembly
        self.flush_cache()
        
        # Pre-check which cells actually have data to avoid loading empty grids
        cells_with_data = []
        for ci, cj in target_cells:
            grid_path = self.get_fine_grid_path(ci, cj)
            npy_path = grid_path + '.npy'
            if os.path.exists(npy_path):
                try:
                    # Quick check: if file is very small, it's likely empty (compressed zeros)
                    file_size = os.path.getsize(npy_path)
                    # Minimum size threshold - NumPy saves even empty arrays with some overhead
                    # For 4000x4000 float32, we expect ~64MB for full data, but even sparse data 
                    # with compression should be > 10KB if it contains meaningful values
                    if file_size > 10000:  # More than 10KB suggests actual data
                        cells_with_data.append((ci, cj))
                except:
                    pass  # If we can't check, include it to be safe
            
        if not cells_with_data:
            logging.debug("No data found in any grid tiles for extent %s", target_extent)
            return output_grid  # All grids are empty
            
        # Calculate downsampling factor for fine grids if needed
        downsample_factor = max(1, int(scale_factor)) if 'scale_factor' in locals() else 1
        
        logging.debug("Loading %d grid tiles with downsample factor %d", len(cells_with_data), downsample_factor)
        
        for idx, (ci, cj) in enumerate(cells_with_data):
            if len(cells_with_data) > 10 and idx % max(1, len(cells_with_data) // 10) == 0:
                logging.debug("Processing grid tile %d/%d", idx + 1, len(cells_with_data))
            fine_grid = self.load_fine_grid(ci, cj)
            fine_grid_cpu = to_numpy(fine_grid)
            
            # Downsample fine grid if needed for performance
            if downsample_factor > 1:
                step = downsample_factor
                fine_grid_cpu = fine_grid_cpu[::step, ::step]
            
            # Map fine grid to output grid coordinates
            cell_extent = self.get_coarse_cell_extent(ci, cj)
            cell_lon_w, cell_lon_e, cell_lat_s, cell_lat_n = cell_extent
            
            # Calculate overlap region
            overlap_lon_w = max(lon_w, cell_lon_w)
            overlap_lon_e = min(lon_e, cell_lon_e)
            overlap_lat_s = max(lat_s, cell_lat_s)
            overlap_lat_n = min(lat_n, cell_lat_n)
            
            if overlap_lon_w >= overlap_lon_e or overlap_lat_s >= overlap_lat_n:
                continue
            
            # Map to output grid indices using numpy for consistency
            out_i_start = int(numpy_original.floor((overlap_lon_w - lon_w) / fine_dlon))
            out_i_end = int(numpy_original.ceil((overlap_lon_e - lon_w) / fine_dlon))
            out_j_start = int(numpy_original.floor((overlap_lat_s - lat_s) / fine_dlat))
            out_j_end = int(numpy_original.ceil((overlap_lat_n - lat_s) / fine_dlat))
            
            # Map to fine grid indices (adjusted for downsampling)
            fine_nx_effective = fine_grid_cpu.shape[0]
            fine_ny_effective = fine_grid_cpu.shape[1]
            
            fine_i_start = int(numpy_original.floor((overlap_lon_w - cell_lon_w) / self.coarse_dlon * fine_nx_effective))
            fine_i_end = int(numpy_original.ceil((overlap_lon_e - cell_lon_w) / self.coarse_dlon * fine_nx_effective))
            fine_j_start = int(numpy_original.floor((overlap_lat_s - cell_lat_s) / self.coarse_dlat * fine_ny_effective))
            fine_j_end = int(numpy_original.ceil((overlap_lat_n - cell_lat_s) / self.coarse_dlat * fine_ny_effective))
            
            # Clamp indices
            out_i_start = max(0, min(out_i_start, out_nx))
            out_i_end = max(0, min(out_i_end, out_nx))
            out_j_start = max(0, min(out_j_start, out_ny))
            out_j_end = max(0, min(out_j_end, out_ny))
            
            fine_i_start = max(0, min(fine_i_start, fine_nx_effective))
            fine_i_end = max(0, min(fine_i_end, fine_nx_effective))
            fine_j_start = max(0, min(fine_j_start, fine_ny_effective))
            fine_j_end = max(0, min(fine_j_end, fine_ny_effective))
            
            # Copy overlapping region - ensure both arrays are numpy arrays
            if (out_i_end > out_i_start and out_j_end > out_j_start and
                fine_i_end > fine_i_start and fine_j_end > fine_j_start):
                
                # Handle size mismatches by using minimum dimensions
                copy_nx = min(out_i_end - out_i_start, fine_i_end - fine_i_start)
                copy_ny = min(out_j_end - out_j_start, fine_j_end - fine_j_start)
                
                if copy_nx > 0 and copy_ny > 0:
                    output_grid[out_i_start:out_i_start+copy_nx, out_j_start:out_j_start+copy_ny] += \
                        fine_grid_cpu[fine_i_start:fine_i_start+copy_nx, fine_j_start:fine_j_start+copy_ny]
        
        return output_grid
    
    def _get_intersecting_cells(self, extent: tuple) -> list:
        """Get list of coarse cell indices that intersect with given extent"""
        lon_w, lon_e, lat_s, lat_n = extent
        
        # Add padding to ensure edge cells are included
        padding_cells = 1
        
        # Find cell ranges
        ci_start = max(0, int(np.floor((lon_w - self.lon_w) / self.coarse_dlon)) - padding_cells)
        ci_end = min(self.coarse_nx - 1, int(np.ceil((lon_e - self.lon_w) / self.coarse_dlon)) + padding_cells)
        cj_start = max(0, int(np.floor((lat_s - self.lat_s) / self.coarse_dlat)) - padding_cells)
        cj_end = min(self.coarse_ny - 1, int(np.ceil((lat_n - self.lat_s) / self.coarse_dlat)) + padding_cells)
        
        cells = []
        for ci in range(ci_start, ci_end + 1):
            for cj in range(cj_start, cj_end + 1):
                cells.append((ci, cj))
        
        return cells
    
    def export_nonzero_cells_shapefile(self, output_path: str, precision: int = 1):
        """Export all fine grid cells with concentration > 0.0 as a shapefile
        
        Args:
            output_path: Path for output shapefile
            precision: Decimal precision for RAD values (default: 1)
        """
        try:
            import geopandas as gpd
            from shapely.geometry import Point
        except ImportError:
            logging.error("geopandas and shapely are required for shapefile export")
            return False
        
        # Flush cache to ensure all data is saved to disk
        self.flush_cache()
        
        # Find all grid files that exist (numpy.save adds .npy extension)
        grid_files = []
        for ci in range(self.coarse_nx):
            for cj in range(self.coarse_ny):
                grid_path = self.get_fine_grid_path(ci, cj)
                # numpy.save automatically adds .npy extension
                npy_path = grid_path + '.npy'
                if os.path.exists(npy_path):
                    # Quick check for non-empty files
                    file_size = os.path.getsize(npy_path)
                    if file_size > 10000:  # Skip likely empty files
                        grid_files.append((ci, cj, npy_path))
        
        if not grid_files:
            logging.warning("No grid files with data found")
            return False
        
        logging.info("Processing %d grid files for shapefile export", len(grid_files))
        
        # Collect all non-zero cells
        cell_records = []
        
        for idx, (ci, cj, grid_path) in enumerate(grid_files):
            if idx % max(1, len(grid_files) // 10) == 0:
                logging.info("Processing grid file %d/%d", idx + 1, len(grid_files))
            
            try:
                # Load grid
                grid_data = numpy_original.load(grid_path)
                
                # Find non-zero cells
                nonzero_indices = numpy_original.where(grid_data > 0.0)
                if len(nonzero_indices[0]) == 0:
                    continue  # No data in this grid
                
                # Get coarse cell extent
                cell_extent = self.get_coarse_cell_extent(ci, cj)
                cell_lon_w, cell_lon_e, cell_lat_s, cell_lat_n = cell_extent
                
                # Calculate fine cell dimensions
                fine_dlon = (cell_lon_e - cell_lon_w) / self.fine_nx
                fine_dlat = (cell_lat_n - cell_lat_s) / self.fine_ny
                
                # Create points for each non-zero cell (at cell center)
                for fi, fj in zip(nonzero_indices[0], nonzero_indices[1]):
                    concentration = float(grid_data[fi, fj])
                    
                    # Calculate cell center coordinates
                    lon_center = cell_lon_w + (fi + 0.5) * fine_dlon
                    lat_center = cell_lat_s + (fj + 0.5) * fine_dlat
                    
                    # Create point at the cell center
                    cell_point = Point(lon_center, lat_center)
                    
                    # Round concentration to specified precision
                    rad_value = round(concentration, precision)
                    
                    cell_records.append({
                        'geometry': cell_point,
                        'RAD': rad_value
                    })
                
            except Exception as e:
                logging.warning("Error processing grid file %s: %s", grid_path, e)
                continue
        
        if not cell_records:
            logging.warning("No cells with concentration > 0.0 found")
            return False
        
        logging.info("Creating shapefile with %d cells", len(cell_records))
        
        # Create GeoDataFrame and save
        try:
            gdf = gpd.GeoDataFrame(cell_records, crs="EPSG:4326")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            
            # Save shapefile
            gdf.to_file(output_path)
            
            logging.info("Successfully exported shapefile: %s", output_path)
            logging.info("Total cells exported: %d", len(gdf))
            logging.info("RAD value range: %.1f to %.1f", gdf['RAD'].min(), gdf['RAD'].max())
            
            return True
            
        except Exception as e:
            logging.error("Error saving shapefile: %s", e)
            return False

    def cleanup(self):
        """Clean up temporary files"""
        self.flush_cache()
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass

def export_deposited_particles_shapefile(particles: List[Particle], output_path: str, precision: int = 1):
    """Export deposited particle locations as a point shapefile
    
    Args:
        particles: List of deposited particles
        output_path: Path for output shapefile
        precision: Decimal precision for RAD values (default: 1)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError:
        logging.error("geopandas and shapely are required for shapefile export")
        return False
    
    if not particles:
        logging.warning("No deposited particles to export")
        return False
    
    logging.info("Processing %d deposited particles for shapefile export", len(particles))
    
    # Collect particle data
    particle_records = []
    for p in particles:
        if not p.deposited:
            continue
        
        # Normalize coordinates
        lon, lat = normalize_lonlat(p.lon, p.lat)
        
        # Create point at particle location
        point = Point(lon, lat)
        
        # Calculate concentration value (RAD) from particle
        rad_value = round(24 * p.fallout_mass * p.pol_factor, precision)
        
        particle_records.append({
            'geometry': point,
            'RAD': rad_value,
            'SIZE_MM': round(p.size_m * 1000, 3),
            'SRC_ID': getattr(p, 'src_id', '')
        })
    
    if not particle_records:
        logging.warning("No deposited particles found")
        return False
    
    logging.info("Creating shapefile with %d particles", len(particle_records))
    
    # Create GeoDataFrame and save
    try:
        gdf = gpd.GeoDataFrame(particle_records, crs="EPSG:4326")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        # Save shapefile
        gdf.to_file(output_path)
        
        logging.info("Successfully exported particle shapefile: %s", output_path)
        logging.info("Total particles exported: %d", len(gdf))
        if len(gdf) > 0:
            logging.info("RAD value range: %.1f to %.1f", gdf['RAD'].min(), gdf['RAD'].max())
            logging.info("Size range: %.3f to %.3f mm", gdf['SIZE_MM'].min(), gdf['SIZE_MM'].max())
        
        return True
        
    except Exception as e:
        logging.error("Error saving particle shapefile: %s", e)
        return False

def subset_grid_for_extent(grid: np.ndarray, extent, world_extent=WORLD_EXTENT):
    lon_w, lon_e, lat_s, lat_n = extent
    w_lon_w, w_lon_e, w_lat_s, w_lat_n = world_extent
    tol = 1e-9
    if (abs(lon_w - w_lon_w) < tol and abs(lon_e - w_lon_e) < tol and
        abs(lat_s - w_lat_s) < tol and abs(lat_n - w_lat_n) < tol):
        return grid
    nx, ny = grid.shape
    def lon_to_i(lon: float) -> int:
        span = w_lon_e - w_lon_w
        lon_norm = ((lon - w_lon_w) % span)
        return int(np.clip(round(lon_norm / span * (nx - 1)), 0, nx - 1))
    def lat_to_j(lat: float) -> int:
        frac = (lat - w_lat_s) / (w_lat_n - w_lat_s)
        return int(np.clip(round(frac * (ny - 1)), 0, ny - 1))
    i0 = lon_to_i(lon_w); i1 = lon_to_i(lon_e)
    j0 = lat_to_j(lat_s); j1 = lat_to_j(lat_n)
    if i0 <= i1:
        sub = grid[i0:i1+1, j0:j1+1]
    else:
        sub = np.concatenate((grid[i0:nx, j0:j1+1], grid[0:i1+1, j0:j1+1]), axis=0)
    if sub.shape[0] < 2 or sub.shape[1] < 2:
        # Convert to numpy for padding, then back to cupy
        sub_np = to_numpy(sub)
        sub_np = numpy_original.pad(sub_np, ((0, max(0, 2 - sub_np.shape[0])), (0, max(0, 2 - sub_np.shape[1]))), mode='edge')
        sub = to_cupy(sub_np)
    return sub

def generate_filename(laydown_name: str, timestamp: str, file_type: str, hour: int, extension: str) -> str:
    """Generate filename in format: laydown_timestamp_type_hourH.ext or laydown_timestamp_hourH.ext for shapefiles"""
    if file_type in ['cont', 'conc', 'loft', 'cells']:
        return f"{laydown_name}_{timestamp}_{file_type}_{hour}H.{extension}"
    else:  # shapefile - skip type
        return f"{laydown_name}_{timestamp}_{hour}H.{extension}"

def next_index_for(prefix: str, suffix: str, outdir: str) -> int:
    pattern = os.path.join(outdir, f"{prefix}*{suffix}")
    nums = []
    for p in glob.glob(pattern):
        base = os.path.basename(p)
        try: nums.append(int(base.split("_")[1]))
        except Exception: pass
    return (max(nums) + 1) if nums else 0

def plot_lofted(parts: List[Particle], hour: int, outdir: str, extent=PLOT_EXTENT, laydown_name: str = "", run_timestamp: str = "") -> str:
    if laydown_name and run_timestamp:
        filename = generate_filename(laydown_name, run_timestamp, 'loft', hour, 'png')
        path = os.path.join(outdir, filename)
    else:
        # Fallback to old naming scheme
        idx = next_index_for('fallout_loft_', f'_{hour}H.png', outdir)
        path = os.path.join(outdir, f'fallout_loft_{idx}_{hour}H.png')

    fig, ax, _ = _make_map_ax(extent)

    # Extract particle attributes
    lons = [p.lon for p in parts if not p.deposited]
    lats = [p.lat for p in parts if not p.deposited]
    z_m  = [p.z   for p in parts if not p.deposited]

    if lons:
        # Convert to numpy for matplotlib
        lons = numpy_original.array(lons)
        lats = numpy_original.array(lats)
        z_m = numpy_original.array(z_m)
        
        # Normalize altitude values
        norm = mpl.colors.Normalize(vmin=min(z_m), vmax=max(z_m))

        # Base colormap
        base_cmap = plt.cm.viridis

        # Build RGBA with altitude-dependent alpha (low = transparent, high = opaque)
        cmap_with_alpha = base_cmap(norm(z_m))
        alphas = norm(z_m)  # 0 (low z) → transparent, 1 (high z) → opaque
        cmap_with_alpha[:, -1] = alphas  # replace alpha channel

        # Scatter with RGBA colors
        sc = ax.scatter(lons, lats, c=cmap_with_alpha, s=2, transform=ccrs.PlateCarree())

        # Colorbar still shows altitude scale (using base cmap, not the transparent RGBA array)
        sm = plt.cm.ScalarMappable(cmap=base_cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax, shrink=0.84, pad=0.02)
        cbar.set_label('Lofted particle altitude (m)')

    ax.set_title(f'fallout plumes D+{hour}H')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path

def deposit_to_grid(deposited: List[Particle], grid: np.ndarray, extent=WORLD_EXTENT):
    if not deposited: return
    lon_w, lon_e, lat_s, lat_n = extent
    nx, ny = grid.shape
    lons = np.array([normalize_lonlat(p.lon, p.lat)[0] for p in deposited], dtype=np.float64)
    lats = np.array([normalize_lonlat(p.lon, p.lat)[1] for p in deposited], dtype=np.float64)
    masses = 24*np.array([getattr(p, 'fallout_mass', 0.0) * getattr(p, 'pol_factor', 1.0) for p in deposited], dtype=np.float32)
    fi = (lons - lon_w) / (lon_e - lon_w)
    fj = (lats - lat_s) / (lat_n - lat_s)
    ii = (np.floor(fi * (nx - 1)).astype(np.int64)) % nx
    jj = np.clip(np.floor(fj * (ny - 1)).astype(np.int64), 0, ny - 1)
    # Use cupy's add.at equivalent
    np.add.at(grid, (ii, jj), masses)

def plot_concentration_hierarchical(hierarchical_grid: HierarchicalGrid, hour: int, outdir: str, extent=PLOT_EXTENT, laydown_name: str = "", run_timestamp: str = "") -> str:
    """Plot concentration using hierarchical grid system"""
    if laydown_name and run_timestamp:
        filename = generate_filename(laydown_name, run_timestamp, 'conc', hour, 'png')
        path = os.path.join(outdir, filename)
    else:
        # Fallback to old naming scheme
        idx = next_index_for('fallout_conc_', f'_{hour}H.png', outdir)
        path = os.path.join(outdir, f'fallout_conc_{idx}_{hour}H.png')
    
    # Assemble grid for the target extent with reasonable size for visualization
    max_viz_size = 2000 if extent == WORLD_EXTENT else 4000 
    grid = hierarchical_grid.assemble_full_grid_for_extent(extent, max_output_size=max_viz_size)
    
    fig, ax, _ = _make_map_ax(extent)
    Z = to_numpy(grid.T)  # Convert to numpy for matplotlib
    eps = max(numpy_original.percentile(Z[Z>0] if numpy_original.any(Z>0) else numpy_original.array([1.0]), 5), 1e-6)
    red_alpha = LinearSegmentedColormap.from_list('red_alpha', [(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 1.0)], N=256)
    im = ax.imshow(Z, origin='lower', extent=extent, transform=ccrs.PlateCarree(),
                   cmap=red_alpha, norm=LogNorm(vmin=eps, vmax=max(Z.max(), eps*10)))
    cbar = fig.colorbar(im, ax=ax, shrink=0.84, pad=0.02, format=LogFormatter(10))
    cbar.set_label('Deposited fallout (arb. units)')
    ax.set_title(f'settled fallout D+{hour}H')
    fig.savefig(path, bbox_inches='tight'); plt.close(fig)
    return path

def plot_concentration(grid: np.ndarray, hour: int, outdir: str, extent=PLOT_EXTENT, laydown_name: str = "", run_timestamp: str = "") -> str:
    if laydown_name and run_timestamp:
        filename = generate_filename(laydown_name, run_timestamp, 'conc', hour, 'png')
        path = os.path.join(outdir, filename)
    else:
        # Fallback to old naming scheme
        idx = next_index_for('fallout_conc_', f'_{hour}H.png', outdir)
        path = os.path.join(outdir, f'fallout_conc_{idx}_{hour}H.png')
    Zsub = subset_grid_for_extent(grid, extent)
    fig, ax, _ = _make_map_ax(extent)
    Z = to_numpy(Zsub.T)  # Convert to numpy for matplotlib
    eps = max(numpy_original.percentile(Z[Z>0] if numpy_original.any(Z>0) else numpy_original.array([1.0]), 5), 1e-6)
    red_alpha = LinearSegmentedColormap.from_list('red_alpha', [(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 1.0)], N=256)
    im = ax.imshow(Z, origin='lower', extent=extent, transform=ccrs.PlateCarree(),
                   cmap=red_alpha, norm=LogNorm(vmin=eps, vmax=max(Z.max(), eps*10)))
    cbar = fig.colorbar(im, ax=ax, shrink=0.84, pad=0.02, format=LogFormatter(10))
    cbar.set_label('Deposited fallout (arb. units)')
    ax.set_title(f'settled fallout D+{hour}H')
    fig.savefig(path, bbox_inches='tight'); plt.close(fig)
    return path

def plot_contours_and_shp_hierarchical(hierarchical_grid: HierarchicalGrid, hour: int, outdir: str, extent=PLOT_EXTENT, laydown_name: str = "", run_timestamp: str = ""):
    """Plot contours and generate shapefile using hierarchical grid system
    
    For PNG generation: uses downsampled grid for performance
    For shapefile generation: uses full-resolution fine grids without downsampling
    """
    start = datetime.now()
    
    if laydown_name and run_timestamp:
        png_filename = generate_filename(laydown_name, run_timestamp, 'cont', hour, 'png')
        png_path = os.path.join(outdir, png_filename)
        shp_filename = generate_filename(laydown_name, run_timestamp, 'shp', hour, 'shp')
        shp_path = os.path.join(outdir, shp_filename)
    else:
        # Fallback to old naming scheme
        idx_png = next_index_for('fallout_cont_', f'_{hour}H.png', outdir)
        png_path = os.path.join(outdir, f'fallout_cont_{idx_png}_{hour}H.png')
        shp_idx = next_index_for('fallout_cont_', f'_{hour}H.shp', outdir)
        shp_path = os.path.join(outdir, f'fallout_cont_{shp_idx}_{hour}H.shp')

    # For PNG: Use downsampled grid for visualization performance
    max_contour_size = 4000 if extent == WORLD_EXTENT else 8000
    grid = hierarchical_grid.assemble_full_grid_for_extent(extent, max_output_size=max_contour_size)
    nx_sub, ny_sub = grid.shape
    
    # Log grid size for performance monitoring
    grid_max = float(to_numpy(grid).max())
    grid_nonzero = int((to_numpy(grid) > 0).sum())
    logging.info("Contour PNG grid: %dx%d (%d points), max=%.1f rad, nonzero cells=%d", 
                 nx_sub, ny_sub, nx_sub * ny_sub, grid_max, grid_nonzero)
    
    # Optimization 1: Early exit if grid is empty or too small
    Z = to_numpy(grid.T)  # Convert to numpy for matplotlib
    if Z.max() <= 0 or nx_sub < 10 or ny_sub < 10:
        # Create empty outputs and return quickly
        fig, ax, _ = _make_map_ax(extent)
        ax.set_title(f"settled fallout contours D+{hour}H (no data)")
        fig.savefig(png_path, bbox_inches='tight'); plt.close(fig)
        return png_path, None
    
    xs = numpy_original.linspace(extent[0], extent[1], nx_sub)
    ys = numpy_original.linspace(extent[2], extent[3], ny_sub)
    X, Y = numpy_original.meshgrid(xs, ys, indexing='xy')

    fig, ax, _ = _make_map_ax(extent)
    levels = numpy_original.unique(CONTOUR_LEVELS.astype(float))  # Use numpy for matplotlib
    cs = ax.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.7, transform=ccrs.PlateCarree())
    ax.clabel(cs, inline=True, fontsize=6)
    ax.set_title(f"settled fallout contours D+{hour}H")
    fig.savefig(png_path, bbox_inches='tight'); plt.close(fig)

    shp_out: Optional[str] = None
    try:
        import geopandas as gpd
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        from scipy import ndimage

        # Generate high-resolution shapefile directly from fine grids (no downsampling)
        logging.info("Generating high-resolution contour shapefile from fine grids...")
        
        levels = numpy_original.unique(CONTOUR_LEVELS.astype(float))
        min_level = float(levels[0]) if len(levels) > 0 else 0
        
        # Get cells that intersect the target extent
        target_cells = hierarchical_grid._get_intersecting_cells(extent)
        
        if not target_cells:
            logging.info("No grid cells in target extent")
            return png_path, None
        
        # Fast pre-filter: identify which grids have data without loading them
        cells_with_data = []
        for ci, cj in target_cells:
            grid_path = hierarchical_grid.get_fine_grid_path(ci, cj)
            npy_path = grid_path + '.npy'
            if os.path.exists(npy_path):
                try:
                    # Quick file size check - compressed empty grids are very small
                    file_size = os.path.getsize(npy_path)
                    # For 4000x4000 float32, meaningful data produces files > 100KB
                    # Empty/sparse grids compress to < 20KB typically
                    if file_size > 100000:  # 100KB threshold
                        cells_with_data.append((ci, cj))
                except:
                    # If we can't check, include it to be safe
                    cells_with_data.append((ci, cj))
        
        if not cells_with_data:
            logging.info("No fine grids with significant data found")
            return png_path, None
        
        logging.info("Found %d/%d grid tiles with data (skipped %d empty tiles)", 
                     len(cells_with_data), len(target_cells), len(target_cells) - len(cells_with_data))
        
        # Group adjacent grids that have data at their borders for seamless contours
        def find_connected_groups(cells_list):
            """Group adjacent cells that should be processed together"""
            if not cells_list:
                return []
            
            cells_set = set(cells_list)
            visited = set()
            groups = []
            
            def get_neighbors(ci, cj):
                """Get adjacent cells (4-connected: N, S, E, W)"""
                return [(ci+1, cj), (ci-1, cj), (ci, cj+1), (ci, cj-1)]
            
            def flood_fill(start_cell):
                """Find all connected cells starting from start_cell"""
                group = []
                stack = [start_cell]
                
                while stack:
                    cell = stack.pop()
                    if cell in visited or cell not in cells_set:
                        continue
                    
                    visited.add(cell)
                    group.append(cell)
                    
                    # Add unvisited neighbors to stack
                    for neighbor in get_neighbors(*cell):
                        if neighbor in cells_set and neighbor not in visited:
                            stack.append(neighbor)
                
                return group
            
            # Find all connected groups
            for cell in cells_list:
                if cell not in visited:
                    group = flood_fill(cell)
                    if group:
                        groups.append(sorted(group))
            
            return groups
        
        # Group cells by connectivity
        cell_groups = find_connected_groups(cells_with_data)
        logging.info("Grouped %d grid tiles into %d connected regions", len(cells_with_data), len(cell_groups))
        
        # Helper function to extract polygons from matplotlib contour object
        def _extract_polygons_from_contour(csf_obj, levels_array, records_list):
            """Extract polygon geometries from matplotlib contourf object"""
            from shapely.geometry import Point as ShapelyPoint
            
            def iter_level_rings(csf):
                if hasattr(csf, "collections") and csf.collections:
                    for i, coll in enumerate(csf.collections):
                        try:
                            paths = coll.get_paths()
                        except Exception:
                            paths = []
                        rings = [numpy_original.asarray(r) for p in paths for r in p.to_polygons() if len(r) >= 3]
                        yield i, rings
                elif hasattr(csf, "allsegs") and csf.allsegs:
                    for i, segs in enumerate(csf.allsegs):
                        rings = [numpy_original.asarray(seg) for seg in segs if seg is not None and len(seg) >= 3]
                        yield i, rings
            
            for i, rings in iter_level_rings(csf_obj):
                if i < len(levels_array) - 1:
                    lev_min = float(levels_array[i])
                    lev_max = float(levels_array[i+1])
                elif i == len(levels_array) - 1:
                    lev_min = float(levels_array[-1])
                    lev_max = float('inf')
                else:
                    continue
                
                if not rings:
                    continue
                
                # Classify rings as outers or holes
                outers, holes = [], []
                for ring in rings:
                    if len(ring) < 3:
                        continue
                    x, y = ring[:, 0], ring[:, 1]
                    area2 = numpy_original.sum(x[:-1]*y[1:] - x[1:]*y[:-1]) + (x[-1]*y[0] - x[0]*y[-1])
                    is_outer = area2 > 0
                    (outers if is_outer else holes).append(ring)
                
                # Create polygons
                for outer in outers:
                    try:
                        x, y = outer[:, 0], outer[:, 1]
                        area_estimate = abs(numpy_original.sum(x[:-1]*y[1:] - x[1:]*y[:-1])) * 0.5
                        if area_estimate < 1e-10:
                            continue
                        
                        outer_poly = Polygon(outer)
                        if outer_poly.area < 1e-10:
                            continue
                        
                        if not outer_poly.is_valid:
                            outer_poly = outer_poly.buffer(0)
                            if outer_poly.is_empty:
                                continue
                        
                        # Match holes to this outer polygon
                        outer_bounds = outer_poly.bounds
                        inner_rings = []
                        for hole in holes:
                            hole_x, hole_y = hole[:, 0], hole[:, 1]
                            if (hole_x.min() >= outer_bounds[0] and hole_x.max() <= outer_bounds[2] and
                                hole_y.min() >= outer_bounds[1] and hole_y.max() <= outer_bounds[3]):
                                try:
                                    hole_center_x, hole_center_y = hole_x.mean(), hole_y.mean()
                                    if outer_poly.contains(ShapelyPoint(hole_center_x, hole_center_y)):
                                        inner_rings.append(hole)
                                except Exception:
                                    continue
                        
                        poly = Polygon(outer, holes=inner_rings or None)
                        if poly.area > 1e-10:
                            records_list.append({
                                "geometry": poly,
                                "level_min": float(lev_min),
                                "level_max": float(lev_max if numpy_original.isfinite(lev_max) else -1.0),
                                "rad": int(round(lev_min)),
                            })
                    except Exception:
                        continue
        
        # Safety limit: max cells in merged grid before fallback to individual processing
        MAX_MERGED_CELLS = 64_000_000  # ~256MB for float32, conservative limit
        
        # Process each group of connected grids
        from collections import defaultdict
        all_records = []
        
        for group_idx, group_cells in enumerate(cell_groups):
            if len(cell_groups) > 1:
                logging.info("  Processing region %d/%d (%d grids)...", group_idx + 1, len(cell_groups), len(group_cells))
            
            # Determine bounding box for this group
            ci_min = min(ci for ci, cj in group_cells)
            ci_max = max(ci for ci, cj in group_cells)
            cj_min = min(cj for ci, cj in group_cells)
            cj_max = max(cj for ci, cj in group_cells)
            
            # Calculate dimensions for merged grid
            n_cols = ci_max - ci_min + 1
            n_rows = cj_max - cj_min + 1
            fine_nx = hierarchical_grid.fine_nx
            fine_ny = hierarchical_grid.fine_ny
            
            merged_nx = n_cols * fine_nx
            merged_ny = n_rows * fine_ny
            total_cells = merged_nx * merged_ny
            
            # Check if merged grid would be too large
            if total_cells > MAX_MERGED_CELLS:
                logging.warning("  Region too large to merge (%d cells, %.1f MB) - processing grids individually", 
                               total_cells, total_cells * 4 / 1_000_000)
                # Fall back to processing each grid individually
                for ci, cj in group_cells:
                    fine_grid = hierarchical_grid.load_fine_grid(ci, cj)
                    fine_grid_cpu = to_numpy(fine_grid)
                    
                    if fine_grid_cpu.max() < min_level:
                        continue
                    
                    # Process individual grid without merging
                    # (contour code will be duplicated below for both individual and merged cases)
                    cell_extent = hierarchical_grid.get_coarse_cell_extent(ci, cj)
                    indiv_lon_w, indiv_lon_e, indiv_lat_s, indiv_lat_n = cell_extent
                    indiv_nx, indiv_ny = fine_grid_cpu.shape
                    
                    xs_indiv = numpy_original.linspace(indiv_lon_w, indiv_lon_e, indiv_nx)
                    ys_indiv = numpy_original.linspace(indiv_lat_s, indiv_lat_n, indiv_ny)
                    X_indiv, Y_indiv = numpy_original.meshgrid(xs_indiv, ys_indiv, indexing='xy')
                    Z_indiv = fine_grid_cpu.T
                    
                    # Smoothing
                    smoothing_kernel = numpy_original.array([[0.0625, 0.125, 0.0625],
                                                           [0.125,  0.25,  0.125],
                                                           [0.0625, 0.125, 0.0625]], dtype=numpy_original.float32)
                    Zs_indiv = ndimage.convolve(Z_indiv, smoothing_kernel, mode='nearest')
                    
                    # Generate contours
                    fig, ax = plt.subplots()
                    try:
                        csf_indiv = ax.contourf(X_indiv, Y_indiv, Zs_indiv, levels=levels, extend='max')
                        plt.close(fig)
                        
                        # Extract polygons (using shared helper function below)
                        _extract_polygons_from_contour(csf_indiv, levels, all_records)
                    except Exception as e:
                        plt.close(fig)
                        logging.debug("Contourf failed for grid (%d,%d): %s", ci, cj, e)
                continue  # Skip merged processing for this group
            
            # Safe to merge - create merged grid
            merged_grid = numpy_original.zeros((merged_nx, merged_ny), dtype=numpy_original.float32)
            
            # Get geographic extent of merged region
            first_cell_extent = hierarchical_grid.get_coarse_cell_extent(ci_min, cj_min)
            last_cell_extent = hierarchical_grid.get_coarse_cell_extent(ci_max, cj_max)
            
            merged_lon_w = first_cell_extent[0]
            merged_lon_e = last_cell_extent[1]
            merged_lat_s = first_cell_extent[2]
            merged_lat_n = last_cell_extent[3]
            
            # Load and place each grid in the merged array
            for ci, cj in group_cells:
                fine_grid = hierarchical_grid.load_fine_grid(ci, cj)
                fine_grid_cpu = to_numpy(fine_grid)
                
                # Calculate position in merged grid
                i_offset = (ci - ci_min) * fine_nx
                j_offset = (cj - cj_min) * fine_ny
                
                # Place this fine grid in the merged grid
                merged_grid[i_offset:i_offset+fine_nx, j_offset:j_offset+fine_ny] = fine_grid_cpu
            
            # Skip if merged grid is below threshold
            if merged_grid.max() < min_level:
                continue
            
            # Create coordinate arrays for merged grid
            xs = numpy_original.linspace(merged_lon_w, merged_lon_e, merged_nx)
            ys = numpy_original.linspace(merged_lat_s, merged_lat_n, merged_ny)
            X, Y = numpy_original.meshgrid(xs, ys, indexing='xy')
            
            # Transpose for matplotlib (expects [ny, nx] for contourf)
            Z = merged_grid.T
            
            # Light smoothing to reduce jaggedness
            smoothing_kernel = numpy_original.array([[0.0625, 0.125, 0.0625],
                                                   [0.125,  0.25,  0.125],
                                                   [0.0625, 0.125, 0.0625]], dtype=numpy_original.float32)
            Zs = ndimage.convolve(Z, smoothing_kernel, mode='nearest')
            
            # Generate filled contours for this grid
            fig, ax = plt.subplots()
            try:
                csf = ax.contourf(X, Y, Zs, levels=levels, extend='max')
                plt.close(fig)
                
                # Extract polygons using helper function
                _extract_polygons_from_contour(csf, levels, all_records)
            except Exception as e:
                plt.close(fig)
                logging.debug("Contourf failed for merged region: %s", e)
                continue
        
        logging.info("Generated %d polygon records from fine grids", len(all_records))
        
        if all_records:
            gdf = gpd.GeoDataFrame(all_records, crs="EPSG:4326")
            # Ensure the layer will be Polygon only
            if not all(g.geom_type == 'Polygon' for g in gdf.geometry):
                gdf = gdf.explode(index_parts=False, ignore_index=True)
                gdf = gdf[gdf.geometry.geom_type == 'Polygon']
            os.makedirs(os.path.dirname(shp_path) or ".", exist_ok=True)
            gdf.to_file(shp_path)
            shp_out = shp_path
            logging.info("Created %d records", len(gdf))
        else:
            logging.warning("No contour polygons found; writing empty shapefile.")
            try:
                gdf = gpd.GeoDataFrame({"geometry": [], "level_min": [], "level_max": [], "rad": []}, crs="EPSG:4326")
                gdf.to_file(shp_path)
                shp_out = shp_path
            except Exception as e2:
                logging.warning("Failed to write empty shapefile: %s", e2)
                shp_out = None
    except Exception as e:
        logging.warning("Shapefile export failed: %s", e)
        import traceback
        logging.debug(traceback.format_exc())
        shp_out = None

    end = datetime.now()
    delta = end - start        
    print("Shapefile generation time:", delta)

    return png_path, shp_out

def plot_contours_and_shp(grid: np.ndarray, hour: int, outdir: str, extent=PLOT_EXTENT, laydown_name: str = "", run_timestamp: str = ""):
    start = datetime.now()
    
    if laydown_name and run_timestamp:
        png_filename = generate_filename(laydown_name, run_timestamp, 'cont', hour, 'png')
        png_path = os.path.join(outdir, png_filename)
        shp_filename = generate_filename(laydown_name, run_timestamp, 'shp', hour, 'shp')
        shp_path = os.path.join(outdir, shp_filename)
    else:
        # Fallback to old naming scheme
        idx_png = next_index_for('fallout_cont_', f'_{hour}H.png', outdir)
        png_path = os.path.join(outdir, f'fallout_cont_{idx_png}_{hour}H.png')
        shp_idx = next_index_for('fallout_cont_', f'_{hour}H.shp', outdir)
        shp_path = os.path.join(outdir, f'fallout_cont_{shp_idx}_{hour}H.shp')

    Zsub = subset_grid_for_extent(grid, extent)
    nx_sub, ny_sub = Zsub.shape
    
    # Optimization 1: Early exit if grid is empty or too small
    Z = to_numpy(Zsub.T)  # Convert to numpy for matplotlib
    if Z.max() <= 0 or nx_sub < 10 or ny_sub < 10:
        # Create empty outputs and return quickly
        fig, ax, _ = _make_map_ax(extent)
        ax.set_title(f"settled fallout contours D+{hour}H (no data)")
        fig.savefig(png_path, bbox_inches='tight'); plt.close(fig)
        return png_path, None
    
    xs = numpy_original.linspace(extent[0], extent[1], nx_sub)
    ys = numpy_original.linspace(extent[2], extent[3], ny_sub)
    X, Y = numpy_original.meshgrid(xs, ys, indexing='xy')

    fig, ax, _ = _make_map_ax(extent)
    levels = numpy_original.unique(CONTOUR_LEVELS.astype(float))  # Use numpy for matplotlib
    cs = ax.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.7, transform=ccrs.PlateCarree())
    ax.clabel(cs, inline=True, fontsize=6)
    ax.set_title(f"settled fallout contours D+{hour}H")
    fig.savefig(png_path, bbox_inches='tight'); plt.close(fig)

    shp_out: Optional[str] = None
    try:
        import geopandas as gpd
        from shapely.geometry import Polygon, LinearRing, GeometryCollection, Point
        from shapely.ops import unary_union

        # Optimization 2: Skip shapefile generation if no significant contours
        max_val = float(Z.max())
        min_level = float(levels[0]) if len(levels) > 0 else 0
        if max_val < min_level:
            return png_path, None

        # Optimization 3: Use optimized smoothing - single pass kernel convolution
        fig2, ax2 = plt.subplots()
        # Fast 3x3 smoothing kernel (replaces expensive padding + manual convolution)
        from scipy import ndimage
        smoothing_kernel = numpy_original.array([[0.0625, 0.125, 0.0625],
                                               [0.125,  0.25,  0.125],
                                               [0.0625, 0.125, 0.0625]], dtype=numpy_original.float32)
        Zs = ndimage.convolve(Z, smoothing_kernel, mode='nearest')
        csf = ax2.contourf(X, Y, Zs, levels=levels, extend='max')
        plt.close(fig2)

        # --- Optimized Chaikin smoothing for contour rings ---
        def _chaikin_ring_optimized(coords, iters):
            pts = numpy_original.asarray(coords, dtype=numpy_original.float32)  # Use float32 throughout
            if pts.shape[0] < 3:
                return pts
                
            # Optimization 4: Adaptive smoothing - reduce iterations for small rings
            n_pts = len(pts)
            if n_pts < 20:
                actual_iters = max(0, min(1, iters or POLY_SMOOTH_ITER))  # 1 iteration for small rings
            elif n_pts < 100:
                actual_iters = max(0, min(2, iters or POLY_SMOOTH_ITER))  # 2 iterations for medium rings
            else:
                actual_iters = iters or POLY_SMOOTH_ITER  # Full iterations only for large rings
                
            if numpy_original.allclose(pts[0], pts[-1], atol=1e-6):  # Relaxed tolerance
                pts = pts[:-1]
                
            # Optimization 5: Vectorized Chaikin with pre-allocated arrays
            for _ in range(int(actual_iters)):
                nxt = numpy_original.roll(pts, -1, axis=0)
                new_size = 2 * len(pts)
                out = numpy_original.empty((new_size, 2), dtype=numpy_original.float32)
                out[0::2] = 0.75*pts + 0.25*nxt  # Q points
                out[1::2] = 0.25*pts + 0.75*nxt  # R points
                pts = out
                
            # Close the ring
            return numpy_original.vstack([pts, pts[0:1]])

        def iter_level_rings(csf_obj):
            if hasattr(csf_obj, "collections") and csf_obj.collections:
                for i, coll in enumerate(csf_obj.collections):
                    try:
                        paths = coll.get_paths()
                    except Exception:
                        paths = []
                    rings = [numpy_original.asarray(r) for p in paths for r in p.to_polygons() if len(r) >= 3]
                    yield i, rings
            elif hasattr(csf_obj, "allsegs") and csf_obj.allsegs:
                for i, segs in enumerate(csf_obj.allsegs):
                    rings = [numpy_original.asarray(seg) for seg in segs if seg is not None and len(seg) >= 3]
                    yield i, rings

        from collections import defaultdict
        per_band = defaultdict(list)

        for i, rings in iter_level_rings(csf):
            # Optimization 6: Batch process rings and use optimized smoothing
            if POLY_SMOOTH_ITER and POLY_SMOOTH_ITER > 0:
                rings = [_chaikin_ring_optimized(r, POLY_SMOOTH_ITER) for r in rings if r is not None and len(r) >= 3]

            if i < len(levels) - 1:
                lev_min = float(levels[i]); lev_max = float(levels[i+1])
            elif i == len(levels) - 1:
                lev_min = float(levels[-1]); lev_max = float('inf')
            else:
                continue

            if not rings:
                continue

            # Optimization 7: Vectorized ring classification  
            outers, holes = [], []
            for ring in rings:
                if len(ring) < 3:
                    continue
                # Fast shoelace formula for orientation (vectorized)
                x, y = ring[:, 0], ring[:, 1]
                area2 = numpy_original.sum(x[:-1]*y[1:] - x[1:]*y[:-1]) + (x[-1]*y[0] - x[0]*y[-1])
                is_outer = area2 > 0
                (outers if is_outer else holes).append(ring)

            # Optimization 8: Streamlined polygon creation with reduced validation
            for outer in outers:
                try:
                    # Skip tiny polygons early
                    x, y = outer[:, 0], outer[:, 1]
                    area_estimate = abs(numpy_original.sum(x[:-1]*y[1:] - x[1:]*y[:-1])) * 0.5
                    if area_estimate < 1e-10:  # Skip microscopic polygons
                        continue
                        
                    outer_poly = Polygon(outer)
                    if outer_poly.area < 1e-10:  # Quick area check before validation
                        continue
                        
                    # Only validate/buffer if necessary  
                    if not outer_poly.is_valid:
                        outer_poly = outer_poly.buffer(0)
                        if outer_poly.is_empty:
                            continue
                    
                    # Optimization 9: Simplified hole matching (only check containment for nearby holes)
                    outer_bounds = outer_poly.bounds
                    inner_rings = []
                    for hole in holes:
                        hole_x, hole_y = hole[:, 0], hole[:, 1]
                        # Quick bounds check before expensive containment test
                        if (hole_x.min() >= outer_bounds[0] and hole_x.max() <= outer_bounds[2] and
                            hole_y.min() >= outer_bounds[1] and hole_y.max() <= outer_bounds[3]):
                            try:
                                hole_center_x, hole_center_y = hole_x.mean(), hole_y.mean()
                                if outer_poly.contains(Point(hole_center_x, hole_center_y)):
                                    inner_rings.append(hole)
                            except Exception:
                                continue
                                
                    poly = Polygon(outer, holes=inner_rings or None)
                    if poly.area > 1e-10:  # Only add polygons with meaningful area
                        per_band[(lev_min, lev_max)].append(poly)
                except Exception:
                    continue

        # Optimization 10: Streamlined polygon union and record building
        records = []
        for (lev_min, lev_max), plist in per_band.items():
            if not plist:
                continue
                
            # Filter out invalid/empty polygons upfront
            valid_geoms = [p for p in plist if p.area > 1e-10]
            if not valid_geoms:
                continue

            # Smart union strategy: skip union for single polygons, batch small numbers
            if len(valid_geoms) == 1:
                polys = valid_geoms
            elif len(valid_geoms) <= 5:
                # Small number - union directly
                try:
                    u = unary_union(valid_geoms)
                    polys = [u] if u.geom_type == 'Polygon' else (list(u.geoms) if hasattr(u, 'geoms') else [])
                except Exception:
                    polys = valid_geoms  # Fallback to individual polygons
            else:
                # Large number - cascade union in chunks to avoid memory issues
                try:
                    chunk_size = 10
                    chunks = [valid_geoms[i:i+chunk_size] for i in range(0, len(valid_geoms), chunk_size)]
                    chunk_unions = [unary_union(chunk) for chunk in chunks]
                    u = unary_union(chunk_unions)
                    polys = [u] if u.geom_type == 'Polygon' else (list(u.geoms) if hasattr(u, 'geoms') else [])
                except Exception:
                    polys = valid_geoms  # Fallback

            # Build records with minimal validation
            for poly in polys:
                if hasattr(poly, 'area') and poly.area > 1e-10:
                    records.append({
                        "geometry": poly,
                        "level_min": float(lev_min),
                        "level_max": float(lev_max if numpy_original.isfinite(lev_max) else -1.0),
                        "rad": int(round(lev_min)),
                    })

        if records:
            gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
            # Ensure the layer will be Polygon only
            if not all(g.geom_type == 'Polygon' for g in gdf.geometry):
                gdf = gdf.explode(index_parts=False, ignore_index=True)
                gdf = gdf[gdf.geometry.geom_type == 'Polygon']
            os.makedirs(os.path.dirname(shp_path) or ".", exist_ok=True)
            gdf.to_file(shp_path)
            shp_out = shp_path
#            logging.info("Created %d Polygon features", len(gdf))
        else:
            logging.warning("No filled-contour Polygons found; writing empty shapefile.")
            try:
                gdf = gpd.GeoDataFrame({"geometry": [], "level_min": [], "level_max": [], "rad": []}, crs="EPSG:4326")
                gdf.to_file(shp_path); shp_out = shp_path
            except Exception as e2:
                logging.warning("Failed to write empty shapefile: %s", e2); shp_out = None
    except Exception as e:
        logging.warning("Shapefile export skipped: %s", e); shp_out = None

    end = datetime.now()

    # Compute difference
    delta = end - start        
    print("Shapefile generation time:", delta)
    #print("Milliseconds:", delta.total_seconds() * 1000)

    return png_path, shp_out


# ------------------------------ Simulation ------------------------------
def _log_footer(active_particles: List[Particle], initial_hist: Counter):
    lines = [f"rand_fraction={RAND_FRACTION:.2f} | horiz_advection_scale={HORIZ_ADVECTION_SCALE:g}"]
    total_init = sum(initial_hist.values())
    lines.append(f"active_remaining={sum(1 for p in active_particles if not p.deposited)} | total_init={total_init}")
    return "\n".join(lines)

def read_laydown(csv_path: str, override_datetime: Optional[str] = None) -> List[Source]:
    """
    Read laydown CSV or Excel file and parse source locations.
    Supports both simple CSV format and OPEN-RISOP Excel format.
    
    Args:
        csv_path: Path to laydown CSV or Excel file (.csv, .xlsx, .xls)
        override_datetime: Optional datetime string (ISO format) to override all individual 
                          source start times. When provided, the datetime column becomes optional.
                          Format: YYYY-MM-DDTHH:MM:SS (e.g., '1997-06-02T12:00:00')
    
    Returns:
        List of Source objects with parsed data
    """
    def norm(s: str) -> str: return ''.join(ch for ch in s.lower() if ch.isalnum())
    
    # Determine file type and read accordingly
    file_ext = os.path.splitext(csv_path.lower())[1]
    
    if file_ext in ['.xlsx', '.xls']:
        if not HAS_EXCEL_SUPPORT:
            raise ValueError(f"Excel file support not available. Install openpyxl: pip install openpyxl")
        
        logging.info("📊 Reading Excel file: %s", csv_path)
        try:
            # Try to read Excel file - handle multiple sheets if needed
            xl_file = pd.ExcelFile(csv_path)
            sheet_names = xl_file.sheet_names
            logging.debug("Found Excel sheets: %s", sheet_names)
            
            # Try to find the main data sheet (usually first sheet or one with "target" in name)
            target_sheet = sheet_names[0]  # Default to first sheet
            for sheet in sheet_names:
                if any(keyword in sheet.lower() for keyword in ['target', 'laydown', 'data', 'main']):
                    target_sheet = sheet
                    break
            
            logging.info("📋 Using Excel sheet: '%s'", target_sheet)
            df = pd.read_excel(csv_path, sheet_name=target_sheet, dtype=str, keep_default_na=False)
            
            # Handle Excel files that might have metadata rows at the top
            # Look for the actual header row (contains target/coordinate data)
            header_row = 0
            for i in range(min(10, len(df))):  # Check first 10 rows max
                row_str = ' '.join(str(val).lower() for val in df.iloc[i] if pd.notna(val))
                if any(keyword in row_str for keyword in ['lat', 'lon', 'target', 'yield', 'id', 'dmpi']):
                    if i > 0:
                        logging.info("📍 Found header row at line %d, skipping %d metadata rows", i+1, i)
                        df = pd.read_excel(csv_path, sheet_name=target_sheet, header=i, dtype=str, keep_default_na=False)
                    break
            
        except Exception as e:
            logging.error("❌ Failed to read Excel file: %s", e)
            raise ValueError(f"Could not read Excel file {csv_path}: {e}")
    else:
        logging.info("📄 Reading CSV file: %s", csv_path)
        # Read CSV with better handling of mixed types
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    
    # Remove completely empty rows
    df = df.dropna(how='all')
    
    # Convert empty strings back to NaN for proper detection
    df = df.replace('', pd.NA)
    
    aliases = {
        'id': {'id', 'target_id', 'targetid', 'target', 'dmpi', 'dmpino', 'dmpi_no', 'weapon_id', 'weaponid', 'name'},
        'datetime': {'datetime','datetimestamp','timestamp','datetimeutc','date','time','datetimez','datetimeyyymmddhhmm','date_time','date/time',
                    'toa', 'time_on_target', 'timeontarget', 'impact_time', 'impacttime', 'detonation_time', 'detonationtime'},
        'latitude': {'latitude','lat', 'target_lat', 'targetlat', 'lat_dd', 'latdd', 'target_latitude', 'targetlatitude'},
        'longitude': {'longitude','lon','lng', 'target_lon', 'targetlon', 'lon_dd', 'londd', 'target_longitude', 'targetlongitude'},
        'yield': {'yield','yieldkt','ktyield', 'weapon_yield', 'weaponyield', 'kt_yield', 'ktyield', 'yield_kt', 'warhead_yield', 'warheadyield', 'yieldkt'},
        'height_of_release_km': {'heightofreleasekm','heightofrelease','height','releaseheight','heightkm', 'hob', 'hobm', 'hob(m)', 
                                'height_of_burst', 'heightofburst', 'burst_height', 'burstheight', 'altitude', 
                                'detonation_height', 'detonationheight'},
        'fallout fraction': {'pollutionfraction','pollution_frac','pollutionfrac','falloutfraction','fallout_frac','falloutfrac','fraction','frac','massfraction',
                           'fallout_fraction', 'surface_fraction', 'surfacefraction'},
        'pollution fraction': {'pollutionfraction','pollution_frac','pollutionfrac','falloutfraction','fallout_frac','falloutfrac','fraction','frac','massfraction',
                             'fallout_fraction', 'surface_fraction', 'surfacefraction'}
    }
    raw_to_norm = {c: norm(c) for c in df.columns}
    def find_col(key: str, required=True) -> Optional[str]:
        wanted = aliases[key]
        for raw, n in raw_to_norm.items():
            if n in wanted: return raw
        if required:
            raise ValueError(f"Missing required column (or alias) for '{key}'. Found: {list(df.columns)}")
        return None
    col_id   = find_col('id')
    # datetime column is optional when override_datetime is provided OR for OPEN-RISOP format
    has_hob_field = any('hob' in norm(col) for col in df.columns)
    datetime_required = (override_datetime is None) and (not has_hob_field)
    col_dt  = find_col('datetime', required=datetime_required)
    col_lat  = find_col('latitude');    col_lon = find_col('longitude')
    col_yld  = find_col('yield');       col_h   = find_col('height_of_release_km')
    # fallout fraction is optional - OPEN-RISOP files typically don't have this
    col_frac = find_col('fallout fraction', required=False)

    # Parse override datetime if provided
    override_dt = None
    if override_datetime:
        try:
            override_dt = pd.to_datetime(override_datetime, utc=True)
            logging.info("🕒 DateTime override enabled: %s UTC will be used for all sources", override_dt.isoformat())
        except Exception as e:
            raise ValueError(f"Invalid override datetime format '{override_datetime}': {e}")
    else:
        logging.debug("Using individual datetime from each source row")

    sources: List[Source] = []
    for idx, r in df.iterrows():
        # Skip rows with missing critical data
        if pd.isna(r[col_id]) or r[col_id] == '':
            logging.warning("Skipping row %d: missing ID", idx)
            continue
        # Skip datetime check if we have override datetime OR if this is OPEN-RISOP format
        if not override_datetime and not has_hob_field and col_dt and (pd.isna(r[col_dt]) or r[col_dt] == ''):
            logging.warning("Skipping row %d (ID=%s): missing datetime", idx, r[col_id])
            continue
        if pd.isna(r[col_lat]) or pd.isna(r[col_lon]):
            logging.warning("Skipping row %d (ID=%s): missing coordinates", idx, r[col_id])
            continue
        if pd.isna(r[col_yld]):
            logging.warning("Skipping row %d (ID=%s): missing yield", idx, r[col_id])
            continue
            
        # Parse datetime (use override if provided, or default for OPEN-RISOP)
        if override_dt is not None:
            t = override_dt
        elif has_hob_field and (not col_dt or pd.isna(r[col_dt]) or r[col_dt] == ''):
            # OPEN-RISOP format without datetime - use a default
            t = pd.to_datetime('2000-01-01 12:00:00', utc=True)
            if idx == 0:  # Log once for first row
                logging.info("🕐 OPEN-RISOP file has no datetime - using default: %s UTC", t.isoformat())
        else:
            # Try to parse datetime with multiple formats for OPEN-RISOP compatibility
            datetime_str = r[col_dt]
            t = None
            
            # Try common formats
            datetime_formats = [
                None,  # Let pandas auto-detect first
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d',
                '%m/%d/%Y %H:%M:%S',
                '%m/%d/%Y',
                '%d/%m/%Y %H:%M:%S',
                '%d/%m/%Y',
                '%Y%m%d%H%M',
                '%Y%m%d'
            ]
            
            for fmt in datetime_formats:
                try:
                    if fmt is None:
                        t = pd.to_datetime(datetime_str, utc=True, errors='coerce')
                    else:
                        t = pd.to_datetime(datetime_str, format=fmt, utc=True, errors='coerce')
                    if pd.notna(t):
                        break
                except:
                    continue
                    
            if pd.isna(t): 
                logging.warning("Skipping row %d (ID=%s): invalid datetime '%s'", idx, r[col_id], datetime_str)
                continue
            
        # Parse numeric values with error handling
        try:
            lat = float(r[col_lat])
            lon = float(r[col_lon])
            yield_kt = float(r[col_yld])
            
            # Handle height of burst (HOB) - OPEN-RISOP format uses meters
            if not pd.isna(r[col_h]):
                h_val = str(r[col_h]).lower().strip()
                if h_val in ['surface', 'ground', 'contact', '0', '']:
                    h_release_km = 0.0
                elif h_val in ['airburst', 'air', 'optimal']:
                    # Estimate optimal airburst height based on yield (rough approximation)
                    h_release_km = min(2.0, 0.1 + 0.3 * (yield_kt ** 0.4))  # Rough scaling
                else:
                    h_release_km = float(h_val)
                    # Check if this looks like the HOB (m) field from OPEN-RISOP
                    col_h_norm = norm(col_h) if col_h else ''
                    if 'hobm' in col_h_norm or 'hob(m)' in col_h_norm or 'hob' in col_h_norm:
                        # OPEN-RISOP HOB field is in meters, convert to km
                        h_release_km = h_release_km / 1000.0
                        logging.debug("Converted HOB from %s m to %.3f km for ID=%s", h_val, h_release_km, r[col_id])
                    else:
                        # For other formats, apply smart unit detection
                        if h_release_km > 1000:  # Assume feet
                            h_release_km = h_release_km * 0.0003048  # feet to km
                        elif h_release_km > 100:  # Assume meters
                            h_release_km = h_release_km / 1000.0  # meters to km
                        # Values < 100 assumed to already be in km
            else:
                h_release_km = 0.0
                
            # Handle fallout fraction - OPEN-RISOP files typically don't have this field
            if col_frac and not pd.isna(r[col_frac]):
                frac = float(r[col_frac])
            elif has_hob_field:
                # OPEN-RISOP format detected - use 0.5 as default fallout fraction
                frac = 0.5
                if idx == 0:  # Log once for first row
                    logging.info("🌊 OPEN-RISOP file missing fallout fraction - using default: 0.5")
            else:
                # Standard format - use 1.0 as default (legacy behavior)
                frac = 1.0
        except (ValueError, TypeError) as e:
            logging.warning("Skipping row %d (ID=%s): numeric conversion error - %s", idx, r[col_id], e)
            continue
            
        sources.append(Source(
            ID=str(r[col_id]), start_time=t.to_pydatetime(),
            lat=lat, lon=lon, yield_kt=yield_kt, 
            h_release_km=h_release_km, frac=frac
        ))
    
    if not sources:
        available_columns = list(df.columns)
        logging.error("Available columns: %s", available_columns)
        if file_ext in ['.xlsx', '.xls']:
            raise ValueError(f"No valid sources found in Excel file {csv_path}. "
                           f"Check sheet structure and column names. Available columns: {available_columns}")
        else:
            raise ValueError(f"No valid sources found in CSV file {csv_path}. "
                           f"Check data format and column names. Available columns: {available_columns}")
    
    # Detect if this looks like OPEN-RISOP format
    is_open_risop = False
    if col_h and ('hob' in norm(col_h) and any(keyword in norm(col) for col in df.columns for keyword in ['yield', 'latitude', 'longitude'])):
        is_open_risop = True
        logging.info("🎯 OPEN-RISOP format detected (HOB field found)")
    
    # Log source summary
    yield_range = f"{min(s.yield_kt for s in sources):.1f}-{max(s.yield_kt for s in sources):.1f} kt" if sources else "N/A"
    height_range = f"{min(s.h_release_km for s in sources):.3f}-{max(s.h_release_km for s in sources):.3f} km" if sources else "N/A"
    frac_range = f"{min(s.frac for s in sources):.1f}-{max(s.frac for s in sources):.1f}" if sources else "N/A"
    
    format_info = "OPEN-RISOP format" if is_open_risop else "Standard format"
    
    if override_datetime:
        logging.info("✅ Successfully loaded %d sources from %s (%s, all using override datetime %s UTC)", 
                    len(sources), csv_path, format_info, override_dt.isoformat())
    else:
        time_range = f"{min(s.start_time for s in sources)} to {max(s.start_time for s in sources)}" if len(sources) > 1 else str(sources[0].start_time)
        logging.info("✅ Successfully loaded %d sources from %s (%s, time range: %s)", 
                    len(sources), csv_path, format_info, time_range)
    
    logging.info("📊 Source summary: yields %s, heights %s, fallout fractions %s", yield_range, height_range, frac_range)
    return sources

def simulate(csv_path: str, outdir: str, uwnd_path: str, vwnd_path: str,
             hours: int = 24, seed: int = 42, loglevel: str = "INFO", extent: tuple = PLOT_EXTENT,
             output_all_hours: bool = False, force_cpu: bool = False, override_datetime: Optional[str] = None):
    global FORCE_CPU_ONLY
    FORCE_CPU_ONLY = force_cpu  # Set global flag for use in other functions
    
    os.makedirs(outdir, exist_ok=True)
    logging.basicConfig(level=getattr(logging, loglevel.upper(), logging.INFO), format="%(asctime)s | %(levelname)s | %(message)s")
    
    # Log GPU acceleration status at startup
    effective_gpu_available = HAS_CUPY and not force_cpu
    if force_cpu:
        logging.info("💻 GPU acceleration DISABLED by --no-gpu flag")
    elif HAS_CUPY:
        try:
            device_name = np.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')
            logging.info("🚀 GPU acceleration ENABLED - Using %s", device_name)
        except:
            logging.info("🚀 GPU acceleration ENABLED - %s", GPU_DEVICE_INFO)
    else:
        logging.info("💻 GPU acceleration DISABLED - %s", GPU_DEVICE_INFO)
        logging.info("💻 Simulation will run on CPU (suitable for small to medium workloads)")
    
    rng = numpy_original.random.default_rng(seed)
    # Set random seed for GPU arrays (if available) or CPU arrays (fallback)
    if HAS_CUPY:
        np.random.seed(seed + 12345)  # CuPy random seed
    else:
        np.random.seed(seed + 12345)  # NumPy random seed (same as CuPy would use)
    t_start = time.perf_counter()
    
    # Extract laydown filename and create timestamp for file naming
    laydown_name = os.path.splitext(os.path.basename(csv_path))[0]
    run_timestamp = datetime.now().strftime("%H%M%S")

    sources = read_laydown(csv_path, override_datetime=override_datetime)
    if not sources:
        logging.error("No sources found in laydown file."); return

    t0 = min(s.start_time for s in sources)
    t_end = t0 + timedelta(hours=hours)
    logging.info("Simulation window: %s → %s UTC", t0.isoformat(), t_end.isoformat())
    
    if output_all_hours:
        logging.info("Output mode: Generating files for all logged hours")
    else:
        logging.info("Output mode: Generating files only for final hour (use --output-all-hours to change)")

    winds = ReanalAccessor(uwnd_path, vwnd_path)

    active_particles: List[Particle] = []
    pending_by_time: Dict[datetime, List[Source]] = {}
    for s in sources:
        pending_by_time.setdefault(s.start_time, []).append(s)

    # Initialize hierarchical grid system
    hierarchical_grid = HierarchicalGrid(outdir, extent)
    
    # Add prompt radiation for all sources before particle deposition
    logging.info("Adding prompt radiation effects for %d sources", len(sources))
    for src in sources:
        hierarchical_grid.add_prompt_radiation(src)

    initial_size_hist: Counter = Counter()
    initial_total = 0
    initial_hist_printed = False
    
    # Track all deposited particles for point shapefile export
    all_deposited_particles: List[Particle] = []

    t = t0
    last_output_hour = -1
    step_idx = 0

    # Pre-sort sources by time for more efficient spawning
    sorted_pending = sorted([(ts, srcs) for ts, srcs in pending_by_time.items()])
    pending_idx = 0  # Index into sorted_pending
    
    # Cache expensive time calculations
    t_iso_cache = {}
    t_timestamp_cache = {}
    
    # Pre-cache t0 and t_end to avoid KeyError
    t_timestamp_cache[t0] = t0.timestamp()
    t_timestamp_cache[t_end] = t_end.timestamp()
    t_iso_cache[t0] = t0.isoformat()
    t_iso_cache[t_end] = t_end.isoformat()

    while t <= t_end:
        # Cache time calculations
        if t not in t_timestamp_cache:
            t_timestamp_cache[t] = t.timestamp()
            t_iso_cache[t] = t.isoformat()
        
        hours_since_start = (t_timestamp_cache[t] - t_timestamp_cache[t0]) / 3600.0
        
        # Three-tier adaptive time stepping
        if hours_since_start < STEP_INITIAL_MIN:
            # Ultra-fine steps for first 5 minutes (in seconds)
            step_sec = STEP_SEC_INITIAL
            step_min = step_sec / 60.0
        elif hours_since_start < STEP_LATE_START_H:
            # Fine steps for remainder of first hour
            step_min_target = STEP_MIN_EARLY
            mins_to_hour = (60 - (t.minute % 60)) % 60
            step_min = step_min_target if (mins_to_hour == 0 or step_min_target <= mins_to_hour) else mins_to_hour
            rem_min = max(0, int(math.ceil((t_timestamp_cache[t_end] - t_timestamp_cache[t]) / 60.0)))
            step_min = max(1, min(step_min, rem_min))
        else:
            # Coarser steps after first hour
            step_min_target = STEP_MIN_LATE
            mins_to_hour = (60 - (t.minute % 60)) % 60
            step_min = step_min_target if (mins_to_hour == 0 or step_min_target <= mins_to_hour) else mins_to_hour
            rem_min = max(0, int(math.ceil((t_timestamp_cache[t_end] - t_timestamp_cache[t]) / 60.0)))
            step_min = max(1, min(step_min, rem_min))

        # Optimized source spawning - process sources in chronological order
        new_srcs = []
        while pending_idx < len(sorted_pending) and sorted_pending[pending_idx][0] <= t:
            _, srcs = sorted_pending[pending_idx]
            new_srcs.extend(srcs)
            pending_idx += 1
            
        # Optimized source spawning - batch operations for better performance
        if new_srcs:
            # Log batch spawning
            total_yield = sum(s.yield_kt for s in new_srcs)
            src_names = ", ".join(s.ID for s in new_srcs)
            logging.info("t=%s | Spawning particles for %d sources: %s (total yield=%.2f kt)", 
                        t_iso_cache[t], len(new_srcs), src_names, total_yield)
            
            # Batch wind sampling for all sources at once
            if len(new_srcs) > 1:
                # Prepare batch coordinates
                batch_lons = np.array([s.lon for s in new_srcs], dtype=np.float32)
                batch_lats = np.array([s.lat for s in new_srcs], dtype=np.float32)
                
                # Compute geometry for all sources
                for s in new_srcs:
                    compute_geometry(s)
                
                # Batch height sampling at plume mid-heights
                batch_heights = np.array([((s.H_TOP_km + s.H_BOTTOM_km)*0.5)*1000.0 for s in new_srcs], dtype=np.float32)
                
                # Single batch wind interpolation call
                batch_u, batch_v = winds.interp_uv_at_alt(t, batch_lons, batch_lats, batch_heights)
            
            # Process each source with pre-computed wind data
            all_spawned = []
            all_sizes_mm = []
            
            for i, s in enumerate(new_srcs):
                if len(new_srcs) == 1:
                    # Single source - compute geometry and wind individually
                    compute_geometry(s)
                    zsamp = np.array([((s.H_TOP_km + s.H_BOTTOM_km)*0.5)*1000.0], dtype=np.float32)
                    u0, v0 = winds.interp_uv_at_alt(t, np.array([s.lon]), np.array([s.lat]), zsamp)
                else:
                    # Use pre-computed batch wind data
                    u0 = np.array([batch_u[i]])
                    v0 = np.array([batch_v[i]])
                
                # Spawn particles for this source
                spawned = init_particles_for_source(s, rng, gpu_rng=None)
                all_spawned.extend(spawned)
                
                # Batch collect sizes for histogram update - bucket by size ranges
                size_buckets = [_size_bucket(p.size_m) for p in spawned]
                all_sizes_mm.extend(size_buckets)
            
            # Single batch update of particles and histogram
            if all_spawned:
                active_particles.extend(all_spawned)
                # Batch histogram update using Counter.update() - much faster
                size_counts = Counter(all_sizes_mm)
                initial_size_hist.update(size_counts)
                initial_total += len(all_spawned)
            
            # Batch wind reporting for all sources
            if len(new_srcs) > 1:
                # Vectorized wind speed calculations - use appropriate array library
                if FORCE_CPU_ONLY:
                    wind_speeds = numpy_original.hypot(batch_u, batch_v)
                    wind_bearings = (numpy_original.degrees(numpy_original.arctan2(batch_u, batch_v)) + 360.0) % 360.0
                else:
                    wind_speeds = np.hypot(batch_u, batch_v)
                    wind_bearings = (np.degrees(np.arctan2(batch_u, batch_v)) + 360.0) % 360.0
                
                for i, s in enumerate(new_srcs):
                    spd = float(wind_speeds[i])
                    bearing_to = float(wind_bearings[i])
                    logging.info("ID=%s | Wind @ ~plume mid-height: %.2f m/s, dir_to=%.1f°", s.ID, spd, bearing_to)
            else:
                # Single source wind reporting
                for i, s in enumerate(new_srcs):
                    u0, v0 = winds.interp_uv_at_alt(t, np.array([s.lon]), np.array([s.lat]), 
                                                   np.array([((s.H_TOP_km + s.H_BOTTOM_km)*0.5)*1000.0]))
                    spd = float(np.hypot(u0[0], v0[0]))
                    bearing_to = (math.degrees(math.atan2(u0[0], v0[0])) + 360.0) % 360.0
                    logging.info("ID=%s | Wind @ ~plume mid-height: %.2f m/s, dir_to=%.1f°", s.ID, spd, bearing_to)

        if (not initial_hist_printed) and initial_total > 0:
            _log_initial_size_hist(initial_size_hist, initial_total); initial_hist_printed = True

        dt_s = step_min * 60.0
        if active_particles:
            dx_w, dy_w = compute_wind_displacements_for_particles_reanal(active_particles, winds, t, dt_s)
            moved, deposited_now = _step_chunk((active_particles, dx_w, dy_w, dt_s, RAND_FRACTION, int(seed + step_idx*9973)))
        else:
            moved, deposited_now = [], []

        if deposited_now:
            # Vectorized deposited particle processing
            current_timestamp = t_timestamp_cache[t]
            
            # Batch time calculations
            elapsed_times = []
            for p in deposited_now:
                try:
                    created_timestamp = getattr(p, "created_at", t).timestamp() if hasattr(getattr(p, "created_at", t), 'timestamp') else current_timestamp
                    elapsed_min = max(0.0, (current_timestamp - created_timestamp) / 60.0)
                except Exception:
                    elapsed_min = 0.0
                elapsed_times.append(elapsed_min)
            
            # Vectorized decay factor calculation
            decay_factors = [aloft_decay_multiplier(elapsed_min) for elapsed_min in elapsed_times]
            
            # Batch update particle properties
            for i, p in enumerate(deposited_now):
                p.pol_factor = decay_factors[i]
                p.landed_at = t
            
            hierarchical_grid.deposit_particles(deposited_now)
            
            # Store deposited particles for point shapefile export
            all_deposited_particles.extend(deposited_now)

        active_particles = moved

        allowed_hours = {0, 1, 2, 5, 11, 17, 23, 29, 36, 48, 60, 72, 84, 96, 112, 124, 136}
        if t.minute == 0 and (t.hour in allowed_hours) and t.hour != last_output_hour:
            hour_since_start = int((t_timestamp_cache[t] - t_timestamp_cache[t0]) // 3600)
            # Pre-compute expensive operations
            active_count = len([p for p in active_particles if not p.deposited])
            
            logging.info("=== Hour %d | lofted=%d ===",
                         hour_since_start, active_count)
            
            # Generate intermediate output files only if requested
            if output_all_hours:
                loft_png = plot_lofted(active_particles, hour_since_start, outdir, extent, laydown_name, run_timestamp)
                conc_png = plot_concentration_hierarchical(hierarchical_grid, hour_since_start, outdir, extent, laydown_name, run_timestamp)

                if t.hour == 23 or t.hour == 48:
                    cont_png, shp_path = plot_contours_and_shp_hierarchical(hierarchical_grid, hour_since_start, outdir, extent, laydown_name, run_timestamp)
                    logging.info("Generated intermediate files: loft, conc, contours, and shapefile") 
                else:
                    logging.info("Generated intermediate files: loft and conc plots")
            else:
                # Skip file generation during simulation
                logging.debug("Skipping intermediate output for hour %d (use --output-all-hours to enable)", hour_since_start)
            
            last_output_hour = t.hour

        # Increment time using seconds for initial period, minutes thereafter
        if hours_since_start < STEP_INITIAL_MIN:
            t += timedelta(seconds=step_sec)
        else:
            t += timedelta(minutes=step_min)
        step_idx += 1

    # Always generate final output files at the end of simulation
    final_hour = int((t_timestamp_cache[t_end] - t_timestamp_cache[t0]) // 3600)
    logging.info("=== Final Hour %d | Generating output files ===", final_hour)
    
    # Export fine grid cells with concentration > 0.0 first
    final_hour = int((t_timestamp_cache[t_end] - t_timestamp_cache[t0]) // 3600)
    if laydown_name and run_timestamp:
        cells_filename = generate_filename(laydown_name, run_timestamp, 'cells', final_hour, 'shp')
        cells_shp_path = os.path.join(outdir, cells_filename)
        particles_filename = generate_filename(laydown_name, run_timestamp, 'particles', final_hour, 'shp')
        particles_shp_path = os.path.join(outdir, particles_filename)
    else:
        cells_idx = next_index_for('fallout_cells_', f'_{final_hour}H.shp', outdir)
        cells_shp_path = os.path.join(outdir, f'fallout_cells_{cells_idx}_{final_hour}H.shp')
        particles_idx = next_index_for('fallout_particles_', f'_{final_hour}H.shp', outdir)
        particles_shp_path = os.path.join(outdir, f'fallout_particles_{particles_idx}_{final_hour}H.shp')
    
    logging.info("Exporting fine grid cells to shapefile...")
    success = hierarchical_grid.export_nonzero_cells_shapefile(cells_shp_path, precision=1)
    if success:
        logging.info("Fine grid cells exported: %s", os.path.basename(cells_shp_path))
    
    # Export deposited particle locations
    logging.info("Exporting deposited particle locations to shapefile...")
    particles_success = export_deposited_particles_shapefile(all_deposited_particles, particles_shp_path, precision=1)
    if particles_success:
        logging.info("Deposited particles exported: %s", os.path.basename(particles_shp_path))
    
    # Generate final output files
    loft_png = plot_lofted(active_particles, final_hour, outdir, extent, laydown_name, run_timestamp)
    conc_png = plot_concentration_hierarchical(hierarchical_grid, final_hour, outdir, extent, laydown_name, run_timestamp)
    cont_png, shp_path = plot_contours_and_shp_hierarchical(hierarchical_grid, final_hour, outdir, extent, laydown_name, run_timestamp)
    
    logging.info("Generated final output files: loft, conc, contours, and shapefiles")
    logging.info("Final files: %s | %s | %s | %s | %s | %s", 
                 os.path.basename(loft_png), os.path.basename(conc_png), 
                 os.path.basename(cont_png), os.path.basename(shp_path) if shp_path else "no_shp",
                 os.path.basename(cells_shp_path) if success else "no_cells_shp",
                 os.path.basename(particles_shp_path) if particles_success else "no_particles_shp")

    _log_remaining_size_hist(active_particles, initial_hist=initial_size_hist)
    
    # Clean up hierarchical grid
    hierarchical_grid.cleanup()
    
    logging.info("Simulation complete.")
    elapsed = time.perf_counter() - t_start
    logging.info("Elapsed wall time: %.3f s (%.2f min)", elapsed, elapsed/60.0)
    
    # Log final GPU acceleration status
    if FORCE_CPU_ONLY:
        logging.info("💻 Simulation completed using CPU-only processing (forced by --no-gpu)")
    elif HAS_CUPY:
        logging.info("🚀 GPU acceleration was AVAILABLE during this simulation")
    else:
        logging.info("💻 Simulation completed using CPU-only processing")

# ------------------------- Random-walk step core -------------------------
def _step_chunk(args):
    particles, dx_w, dy_w, dt_s, rand_fraction, seed = args
    n = len(particles)
    if n == 0: return [], []
    
    # For small particle counts, use CPU arrays (faster due to less overhead)
    use_gpu = n > 1000  # Only use GPU for large particle counts
    
    if use_gpu:
        # Set random seed for CuPy
        np.random.seed(seed)
        
        # Vectorized attribute extraction for GPU
        lon_list = [p.lon for p in particles]
        lat_list = [p.lat for p in particles]
        z_list = [p.z for p in particles]
        w_list = [p.w_settle for p in particles]
        
        lon = np.array(lon_list, dtype=np.float32)
        lat = np.array(lat_list, dtype=np.float32)
        z = np.array(z_list, dtype=np.float32)
        w = np.array(w_list, dtype=np.float32)
        
        dx_w = np.asarray(dx_w, dtype=np.float32)
        dy_w = np.asarray(dy_w, dtype=np.float32)
    else:
        # Use regular NumPy for small arrays (less overhead)
        import numpy as cpu_np
        cpu_np.random.seed(seed)
        
        lon = cpu_np.array([p.lon for p in particles], dtype=cpu_np.float32)
        lat = cpu_np.array([p.lat for p in particles], dtype=cpu_np.float32)
        z   = cpu_np.array([p.z   for p in particles], dtype=cpu_np.float32)
        w   = cpu_np.array([p.w_settle for p in particles], dtype=cpu_np.float32)
        
        # Ensure dx_w and dy_w are CPU NumPy arrays
        if hasattr(dx_w, 'get'):  # CuPy array
            dx_w = dx_w.get().astype(cpu_np.float32)
        else:
            dx_w = cpu_np.array(dx_w, dtype=cpu_np.float32)
            
        if hasattr(dy_w, 'get'):  # CuPy array
            dy_w = dy_w.get().astype(cpu_np.float32)
        else:
            dy_w = cpu_np.array(dy_w, dtype=cpu_np.float32)

    dt_s = float(dt_s)
    
    # Choose array library based on whether we're using GPU
    array_lib = np if use_gpu else cpu_np

    K = array_lib.maximum(1.0, 10.0 * (1.0 + array_lib.minimum(z, 2000.0) / 500.0)).astype(array_lib.float32)
    sigma = array_lib.sqrt(2.0 * K * dt_s).astype(array_lib.float32)
    
    # Generate random numbers
    if use_gpu:
        dx_r = np.random.normal(0.0, 1.0, size=n).astype(np.float32) * sigma
        dy_r = np.random.normal(0.0, 1.0, size=n).astype(np.float32) * sigma
    else:
        dx_r = cpu_np.random.normal(0.0, 1.0, size=n).astype(cpu_np.float32) * sigma
        dy_r = cpu_np.random.normal(0.0, 1.0, size=n).astype(cpu_np.float32) * sigma

    wind_mag = array_lib.hypot(dx_w, dy_w).astype(array_lib.float32)
    rand_mag = array_lib.hypot(dx_r, dy_r).astype(array_lib.float32) + 1e-6
    target_ratio = rand_fraction / max(1e-6, 1.0 - rand_fraction)
    scale = (target_ratio * wind_mag) / rand_mag
    dx_r *= scale; dy_r *= scale

    dx = (dx_w + dx_r).astype(array_lib.float32); dy = (dy_w + dy_r).astype(array_lib.float32)

    lat_calc = array_lib.clip(lat, -89.9, 89.9)
    dlon = (dx / (R_EARTH * array_lib.cos(array_lib.deg2rad(lat_calc)))).astype(array_lib.float32) * (180.0 / np.pi)
    dlat = (dy / R_EARTH).astype(array_lib.float32) * (180.0 / np.pi)
    
    # Update positions
    lon = ((lon + dlon + 180.0) % 360.0) - 180.0
    lat = array_lib.clip(lat + dlat, -89.9, 89.9)
    z = array_lib.maximum(0.0, z - w * dt_s).astype(array_lib.float32)

    dep_mask = (z <= 0.0)
    moved_idx = array_lib.nonzero(~dep_mask)[0]; dep_idx = array_lib.nonzero(dep_mask)[0]

    # Convert to CPU for particle updates (always needed)
    if use_gpu:
        moved_idx_cpu = to_numpy(moved_idx)
        dep_idx_cpu = to_numpy(dep_idx)
        lon_cpu = to_numpy(lon)
        lat_cpu = to_numpy(lat)
        z_cpu = to_numpy(z)
    else:
        moved_idx_cpu = moved_idx
        dep_idx_cpu = dep_idx
        lon_cpu = lon
        lat_cpu = lat
        z_cpu = z
    
    moved = []
    deposited_now = []
    
    # Batch particle updates with reduced indexing overhead
    for i in moved_idx_cpu:
        p = particles[i]
        p.lon = float(lon_cpu[i])
        p.lat = float(lat_cpu[i])
        p.z = float(z_cpu[i])
        moved.append(p)
    
    for i in dep_idx_cpu:
        p = particles[i]
        p.lon = float(lon_cpu[i])
        p.lat = float(lat_cpu[i])
        p.z = 0.0
        p.deposited = True
        deposited_now.append(p)
    
    return moved, deposited_now

# ------------------------------ CLI ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Fallout Lagrangian Simulation (Reanalysis winds, vertical interpolation)")
    ap.add_argument("laydown_csv", help="Path to laydown CSV or Excel file (.csv, .xlsx, .xls). Supports OPEN-RISOP Excel format.")
    ap.add_argument("--out", dest="outdir", required=True, help="Output directory for PNG/shapefile outputs")
    ap.add_argument("--uwnd", required=True, help="Path to PSL uwnd.<year>.nc (pressure levels)")
    ap.add_argument("--vwnd", required=True, help="Path to PSL vwnd.<year>.nc (pressure levels)")
    ap.add_argument("--hours", type=int, default=24, help="Simulation length in hours (default: 24)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    ap.add_argument("--log", dest="loglevel", default="INFO", help="Logging level (DEBUG/INFO/WARN)")
    ap.add_argument("--extent", choices=["world","conus"], default="world", help="Map extent for outputs")
    ap.add_argument("--output-all-hours", action="store_true", 
                    help="Generate output files for all hours (default: only final hour)")
    ap.add_argument("--no-gpu", action="store_true", 
                    help="Force CPU-only simulation (disable GPU acceleration)")
    ap.add_argument("--override-datetime", type=str, metavar="YYYY-MM-DDTHH:MM:SS",
                    help="Override all source start times with this single datetime (UTC). "
                         "Format: YYYY-MM-DDTHH:MM:SS (e.g., 1997-06-02T12:00:00)")
    args = ap.parse_args()
    extent = WORLD_EXTENT if args.extent == "world" else CONUS_EXTENT
    simulate(args.laydown_csv, args.outdir, args.uwnd, args.vwnd,
             hours=args.hours, seed=args.seed, loglevel=args.loglevel, extent=extent,
             output_all_hours=args.output_all_hours, force_cpu=args.no_gpu, 
             override_datetime=args.override_datetime)

if __name__ == "__main__":
    main()
