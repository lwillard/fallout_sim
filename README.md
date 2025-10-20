# Fallout Simulator

GPU-accelerated fallout dispersion simulator with vertical wind interpolation using reanalysis data.

## Overview

This Lagrangian particle simulation models atmospheric fallout transport and deposition from nuclear detonations. The simulator uses NCEP/NCAR reanalysis wind data with vertical interpolation and supports both GPU (CuPy) and CPU (NumPy) execution modes. It can process both standard CSV files and OPEN-RISOP Excel target databases.

## Features

- **Multi-Format Input**: Supports CSV files and OPEN-RISOP Excel formats (.xlsx, .xls)
- **Hybrid GPU/CPU Processing**: Automatically uses GPU for large particle counts (>1000), CPU for smaller workloads
- **DateTime Override**: Synchronize all detonations to a single datetime for scenario analysis
- **OPEN-RISOP Integration**: Native support for nuclear target databases
- **Adaptive Time Stepping**: Fine resolution (1-minute steps) for first hour, coarser (6-minute steps) thereafter
- **Realistic Physics**: Stokes settling law, power-law wind profiles, turbulent diffusion
- **Multiple Burst Types**: Ground burst, low air burst, and high air burst with appropriate size distributions
- **Comprehensive Output**: Particle plots, concentration grids, contour maps, and shapefiles
- **Optimized Performance**: Smart batching, GPU array caching, and streamlined polygon generation

## Requirements

### Required Dependencies
```bash
# Core scientific computing
numpy>=1.20.0
pandas>=1.3.0
xarray>=0.19.0
matplotlib>=3.4.0
cartopy>=0.20.0
scipy>=1.7.0

# Optional Excel support (for OPEN-RISOP files)
openpyxl>=3.0.0

# Optional GPU acceleration (recommended for large simulations)
cupy-cuda11x>=9.0.0  # or cupy-cuda12x depending on your CUDA version

# Optional geospatial output
geopandas>=0.10.0
shapely>=1.8.0
```

### Data Requirements
Download NCEP/NCAR reanalysis pressure-level wind data:
- **U-wind**: https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/pressure/uwnd.YYYY.nc
- **V-wind**: https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/pressure/vwnd.YYYY.nc

## Usage

### Basic Command
```bash
# CSV format
python fallout_sim.py laydown.csv --out results --uwnd uwnd.2021.nc --vwnd vwnd.2021.nc

# OPEN-RISOP Excel format with datetime override
python fallout_sim.py "OPEN-RISOP 1.00 MIXED ATTACK.xlsx" \
  --out results --uwnd uwnd.2021.nc --vwnd vwnd.2021.nc \
  --override-datetime 2021-03-15T06:00:00
```

### Complete Syntax
```bash
python fallout_sim.py LAYDOWN_FILE [OPTIONS]
```

### Required Arguments
- `LAYDOWN_FILE`: Path to CSV or Excel file (.csv, .xlsx, .xls) containing detonation scenarios

### Optional Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--out OUTDIR` | Output directory for results | **Required** |
| `--uwnd FILE` | Path to U-wind reanalysis file | **Required** |
| `--vwnd FILE` | Path to V-wind reanalysis file | **Required** |
| `--hours N` | Simulation duration in hours | 24 |
| `--seed N` | Random seed for reproducibility | 42 |
| `--log LEVEL` | Logging level (DEBUG/INFO/WARN) | INFO |
| `--extent REGION` | Map extent (world/conus) | world |
| `--output-all-hours` | Generate files for all hours | Only final hour |
| `--no-gpu` | Force CPU-only execution | Auto-detect GPU |
| `--override-datetime` | Override all source times with single datetime | Individual times |

## Input File Formats

The simulator supports multiple input formats for maximum compatibility:

### CSV Format

The input CSV file must contain the following columns (case-insensitive):

#### Required Columns

| Column | Aliases | Description | Example |
|--------|---------|-------------|---------|
| `id` | `id`, `name` | Unique identifier for each detonation | `"DET001"` |
| `datetime` | `timestamp`, `date`, `datetimeutc` | Detonation time (ISO format) | `"2021-06-15T12:00:00Z"` |
| `latitude` | `lat` | Latitude in decimal degrees | `40.7128` |
| `longitude` | `lon`, `lng` | Longitude in decimal degrees | `-74.0060` |
| `yield` | `yieldkt`, `ktyield` | Weapon yield in kilotons | `100` |
| `height_of_release_km` | `height`, `releaseheight` | Release height in kilometers | `0.5` |
| `fallout_fraction` | `pollution_fraction`, `fraction`, `frac` | Fraction producing fallout | `0.77` |

#### Example CSV
```csv
id,datetime,latitude,longitude,yield,height_of_release_km,fallout_fraction
DET001,2021-06-15T12:00:00Z,40.7128,-74.0060,100,0.5,0.77
DET002,2021-06-15T13:30:00Z,41.2033,-77.1945,50,2.0,0.65
```

### OPEN-RISOP Excel Format

Professional nuclear target database format used by strategic planning systems.

#### OPEN-RISOP Columns

| Column | Description | Units | Example |
|--------|-------------|-------|---------|
| `Latitude` | Target latitude | Decimal degrees | `40.7128` |
| `Longitude` | Target longitude | Decimal degrees | `-74.0060` |
| `Name` | Target identifier/name | Text | `"NYC_Target_01"` |
| `State/Territory` | Geographic region | Text | `"NY"` |
| `Yield (kt)` | Weapon yield | Kilotons | `100` |
| `HOB (m)` | Height of burst | Meters | `500` |

#### OPEN-RISOP Features
- **No DateTime Required**: Uses `--override-datetime` or defaults to 2000-01-01 12:00:00 UTC
- **Automatic Unit Conversion**: HOB (m) automatically converted to kilometers
- **Default Fallout Fraction**: Uses 0.5 when column missing (typical for mixed weapons)
- **Smart Sheet Detection**: Automatically finds data sheet in multi-sheet Excel files
- **Header Detection**: Skips metadata rows and finds actual column headers

#### Example OPEN-RISOP Command
```bash
python fallout_sim.py "OPEN-RISOP 1.00 MIXED COUNTERFORCE+COUNTERVALUE ATTACK.xlsx" \
  --uwnd uwnd.2021.nc --vwnd vwnd.2021.nc \
  --hours 48 --extent world --out RISOP_Mixed_Attack \
  --override-datetime 2021-03-15T06:00:00
```

### Burst Type Classification
Based on `height_of_release_km` and 'yield':
- **ground burst**: where the fireball makes contact with the ground
- **low airbust**: where the fireball does not make contact with the ground but is close enough for neutron bombardment
to irradiate the ground immediately under the fireball and for the ground to be heated sufficiently to form a vapor stem
of neutron irradiated material
- **high airburst**: where there is no neutron or fireball interaction with the ground from the nuclear explosion

## Output Files

The simulator generates multiple output formats with timestamps for organization:

### File Naming Convention
```
{laydown_name}_{HHMMSS}_{type}_{hour}H.{ext}
```
Example: `scenario1_143052_loft_24H.png`

### Generated Files

| Type | Description | Format |
|------|-------------|---------|
| **Loft Plot** | Active airborne particles | PNG |
| **Concentration Plot** | Ground deposition density | PNG |
| **Contour Plot** | Fallout concentration contours | PNG |
| **Shapefile** | GIS-compatible contour polygons | SHP + supporting files |

### Output Control
- **Default**: Only generates files for the final simulation hour
- **All Hours**: Use `--output-all-hours` to generate files for every hour (slower)

## Configuration Parameters

The script includes extensively documented constants that can be modified:

### Physics Parameters
```python
G = 9.80665           # Gravitational acceleration [m/s²]
RHO_AIR = 1.225       # Air density [kg/m³]
RHO_PARTICLE = 1500.0 # Particle density [kg/m³]
```

### Transport Parameters
```python
HORIZ_ADVECTION_SCALE = 3.9  # Horizontal transport enhancement factor
RAND_FRACTION = 0.11         # Random walk diffusion strength
WIND_ALPHA = 0.06           # Wind profile power law exponent
```

### Computational Parameters
```python
STEP_MIN_EARLY = 1   # Time step for first hour [minutes]
STEP_MIN_LATE = 6    # Time step after first hour [minutes]
GRID_NXY = (12000, 6000)  # Global grid resolution
```

## Performance Optimization

### GPU Acceleration
- **Automatic**: Uses GPU for >1000 particles, CPU for smaller counts
- **Manual Control**: Use `--no-gpu` to force CPU-only execution
- **Requirements**: CUDA-compatible GPU + CuPy installation

### Memory Usage
- **Grid Memory**: ~288MB for global 12k×6k grid (float32)
- **Particle Memory**: Scales with particle count and simulation duration
- **Large Simulations**: Consider using `--extent conus` for regional simulations

### Optimization Tips
1. **Start Small**: Test with short durations (`--hours 6`) first
2. **GPU Check**: Verify GPU acceleration in startup logs
3. **Output Control**: Use default output mode unless you need intermediate files
4. **Regional Focus**: Use `--extent conus` for US-focused scenarios

## Example Workflows

### Basic Regional Simulation
```bash
python fallout_sim.py scenario.csv \
  --out results \
  --uwnd uwnd.2021.nc \
  --vwnd vwnd.2021.nc \
  --hours 48 \
  --extent conus
```

### OPEN-RISOP Strategic Analysis
```bash
python fallout_sim.py "OPEN-RISOP 1.00 MIXED COUNTERFORCE+COUNTERVALUE ATTACK.xlsx" \
  --out RISOP_Analysis \
  --uwnd uwnd.2021.nc \
  --vwnd vwnd.2021.nc \
  --hours 72 \
  --extent world \
  --override-datetime 2021-03-15T06:00:00 \
  --output-all-hours
```

### DateTime Override Scenario Testing
```bash
# Test same targets at different times
python fallout_sim.py targets.csv \
  --out winter_scenario \
  --uwnd uwnd.2021.nc \
  --vwnd vwnd.2021.nc \
  --hours 48 \
  --override-datetime 2021-01-15T12:00:00

python fallout_sim.py targets.csv \
  --out summer_scenario \
  --uwnd uwnd.2021.nc \
  --vwnd vwnd.2021.nc \
  --hours 48 \
  --override-datetime 2021-07-15T12:00:00
```

### CPU-Only Testing
```bash
python fallout_sim.py test_scenario.csv \
  --out test_results \
  --uwnd uwnd.2021.nc \
  --vwnd vwnd.2021.nc \
  --hours 6 \
  --no-gpu \
  --log DEBUG
```

## Troubleshooting

### Common Issues

**Excel Files Not Supported**
- Install openpyxl: `pip install openpyxl`
- Verify Excel file is not corrupted
- Check that Excel file contains target data sheets

**DateTime Override Issues**
- Use ISO format: `YYYY-MM-DDTHH:MM:SS`
- Example: `2021-03-15T06:00:00` (UTC assumed)
- Quotes not needed for command line arguments

**OPEN-RISOP Format Issues**
- Verify Excel file has `Latitude`, `Longitude`, `Yield (kt)`, `HOB (m)` columns
- Check that numeric columns contain valid numbers
- Use `--log DEBUG` to see column detection process

**GPU Not Detected**
- Install CuPy: `pip install cupy-cuda11x`
- Check CUDA installation: `nvidia-smi`
- Use `--no-gpu` flag to run on CPU

**Memory Errors**
- Reduce simulation hours: `--hours 12`
- Use regional extent: `--extent conus`
- Close other GPU applications

**Wind Data Issues**
- Verify file paths exist and are readable
- Check that datetime range in CSV overlaps with wind data
- Ensure wind files cover the geographic extent of your scenarios

**Empty Output**
- Check file format matches requirements
- Verify datetime strings are ISO format with timezone (if using CSV)
- Confirm latitude/longitude coordinates are in decimal degrees

### Log Levels
- `--log DEBUG`: Detailed execution information
- `--log INFO`: Standard progress messages (default)
- `--log WARN`: Warnings and errors only

## Technical Details

### Physics Model
- **Lagrangian Particle Tracking**: Each particle follows individual trajectory
- **Stokes Settling**: Realistic terminal velocity based on particle size
- **Power-Law Wind Profile**: Vertical wind scaling with height
- **Turbulent Diffusion**: Random walk component for sub-grid mixing

### Computational Approach
- **Hybrid CPU/GPU**: Automatic selection based on workload size
- **Adaptive Time Stepping**: Balance accuracy and performance
- **Vectorized Operations**: Efficient batch processing of particles
- **Smart Caching**: GPU memory reuse for repeated array sizes

### Coordinate Systems
- **Input**: WGS84 decimal degrees (latitude/longitude)
- **Computation**: Equirectangular projection for efficiency
- **Output**: WGS84 for GIS compatibility

## Citation

If you use this simulator in research, it would be nice if you cite:

```
Fallout Simulator v2025.2
Lane Willard

```

## License

MIT License

## Support

For issues, questions, or contributions:
- Report bugs via [your issue tracker]
- Documentation: [your documentation site]

- Contact: [your contact information]

