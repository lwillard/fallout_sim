# Fallout Simulator

A Python Lagrangian fallout simulator compatible with OPEN-RISOP.

## Overview

This package provides a Lagrangian particle tracking simulator for modeling nuclear fallout dispersion. It simulates the atmospheric transport and deposition of radioactive particles following a nuclear event, producing georeferenced raster outputs compatible with OPEN-RISOP and standard GIS software.

## Features

- **Lagrangian Particle Tracking**: Simulates individual particle trajectories through the atmosphere
- **Atmospheric Transport**: Includes wind advection, turbulent dispersion, and gravitational settling
- **Deposition Models**: Implements dry and wet deposition processes
- **OPEN-RISOP Compatible**: Generates PNG raster images with PGW world files for GIS integration
- **Flexible Configuration**: Supports both command-line and JSON configuration
- **Particle Size Distribution**: Log-normal particle size distribution for realistic fallout modeling

## Installation

```bash
# Clone the repository
git clone https://github.com/lwillard/fallout_sim.git
cd fallout_sim

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### Command Line Interface

Run a simulation with basic parameters:

```bash
fallout-sim --lat 40.0 --lon -100.0 --altitude 1000 \
            --particles 10000 --activity 1e15 \
            --duration 86400 --wind-speed 5 --wind-direction 90 \
            --output my_simulation
```

### Using a Configuration File

Create a JSON configuration file (e.g., `config.json`):

```json
{
    "source_lat": 40.0,
    "source_lon": -100.0,
    "source_altitude": 1000.0,
    "num_particles": 50000,
    "total_activity": 1e16,
    "duration": 172800,
    "wind_speed": 10.0,
    "wind_direction": 90.0,
    "resolution": 0.01,
    "output": "fallout_output"
}
```

Run with configuration:

```bash
fallout-sim --config config.json
```

### Python API

```python
from fallout_sim import FalloutSimulator, AtmosphericModel

# Create atmospheric model
atmosphere = AtmosphericModel(
    wind_speed=5.0,        # m/s
    wind_direction=90.0    # degrees (0=North, 90=East)
)

# Create simulator
sim = FalloutSimulator(
    source_lat=40.0,
    source_lon=-100.0,
    source_altitude=1000.0,
    num_particles=10000,
    total_activity=1e15,
    atmospheric_model=atmosphere
)

# Run simulation
sim.run(duration=86400)  # 24 hours

# Generate output
raster = sim.generate_output("fallout_output", resolution=0.01)

# Get statistics
stats = sim.get_statistics()
print(f"Deposited: {stats['deposited_particles']} particles")
```

## Output Format

The simulator generates two files compatible with OPEN-RISOP and GIS software:

1. **PNG file** (`filename.png`): Raster image showing deposition pattern
2. **PGW file** (`filename.pgw`): World file containing georeferencing information

The PGW world file format follows the standard specification with 6 lines:
- Line 1: x-scale (pixel size in longitude)
- Line 2: rotation about y-axis (0)
- Line 3: rotation about x-axis (0)
- Line 4: y-scale (negative pixel size in latitude)
- Line 5: x-coordinate of upper-left pixel center
- Line 6: y-coordinate of upper-left pixel center

These files can be directly loaded into GIS software like QGIS, ArcGIS, or used with OPEN-RISOP tools.

## Parameters

### Source Parameters
- `source_lat`: Latitude of source location (degrees)
- `source_lon`: Longitude of source location (degrees)
- `source_altitude`: Initial cloud altitude (meters)

### Particle Parameters
- `num_particles`: Number of particles to simulate (more = better resolution, slower)
- `total_activity`: Total radioactivity in Becquerels (Bq)

### Atmospheric Parameters
- `wind_speed`: Wind speed at surface (m/s)
- `wind_direction`: Wind direction (degrees, 0=North, 90=East, meteorological convention)
- `wind_speed_aloft`: Wind speed at altitude (defaults to 2x surface)
- `wind_direction_aloft`: Wind direction at altitude (defaults to surface direction)

### Simulation Parameters
- `duration`: Simulation duration (seconds)
- `resolution`: Output grid resolution (degrees, default: 0.01 ≈ 1 km)

## Physics Models

### Particle Transport
- **Advection**: Mean wind transport at all altitudes
- **Turbulent Dispersion**: Random walk model with altitude-dependent variance
- **Gravitational Settling**: Terminal velocity based on Stokes' law

### Deposition
- **Dry Deposition**: Gravitational settling and surface layer deposition
- **Wet Deposition**: Optional precipitation scavenging (configurable)

### Particle Properties
- **Size Distribution**: Log-normal distribution (realistic for nuclear debris)
- **Density**: Typical soil/debris density (2500 kg/m³)
- **Terminal Velocity**: Calculated from particle size and density

## Example Scenarios

See the `examples/` directory for sample configurations:

```bash
# Run example scenario
fallout-sim --config examples/example_config.json
```

## Integration with OPEN-RISOP

The output files are directly compatible with OPEN-RISOP:

1. Load the PNG and PGW files into your GIS software
2. The georeferencing allows overlay with OPEN-RISOP target databases
3. Combine multiple scenarios to assess cumulative fallout patterns
4. Export to other formats as needed for consequence analysis

## Technical Notes

### Coordinate System
- Uses WGS84 geographic coordinates (latitude/longitude)
- Altitude is meters above ground level
- Output grid uses equirectangular projection

### Numerical Methods
- Lagrangian particle tracking with Eulerian grid output
- Forward Euler integration for particle positions
- Probabilistic deposition based on deposition rates

### Performance
- Simulation speed scales with number of particles
- 10,000 particles: ~1-2 minutes for 24-hour simulation
- 100,000 particles: ~10-20 minutes for 24-hour simulation

## Limitations

- Simplified atmospheric model (uniform horizontal wind fields)
- No terrain effects or boundary layer parameterization
- Simplified deposition models
- No radioactive decay or daughter products
- No plume rise model (fixed initial altitude)

For more sophisticated modeling, consider coupling with:
- HYSPLIT for meteorological fields
- WRF for high-resolution atmospheric modeling
- FLEXPART for advanced dispersion modeling

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

MIT License - see LICENSE file for details.

## References

- OPEN-RISOP: https://github.com/davidteter/OPEN-RISOP
- Lagrangian particle modeling techniques
- Nuclear fallout physics and atmospheric dispersion

## Acknowledgments

This simulator is designed to be compatible with the OPEN-RISOP project for open-source strategic operations planning and consequence assessment.