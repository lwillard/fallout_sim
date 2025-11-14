"""
Command-line interface for fallout simulator.
"""

import argparse
import json
import sys
from pathlib import Path

from .simulator import FalloutSimulator
from .atmosphere import AtmosphericModel


def load_config(config_file: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_file, 'r') as f:
        return json.load(f)


def progress_callback(time: float, active_count: int):
    """Print simulation progress."""
    print(f"Time: {time/3600:.2f} hours, Active particles: {active_count}")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Lagrangian fallout simulator compatible with OPEN-RISOP"
    )
    
    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Configuration file (JSON)"
    )
    
    parser.add_argument(
        "--lat",
        type=float,
        help="Source latitude (degrees)"
    )
    
    parser.add_argument(
        "--lon",
        type=float,
        help="Source longitude (degrees)"
    )
    
    parser.add_argument(
        "--altitude",
        type=float,
        default=1000.0,
        help="Source altitude (meters, default: 1000)"
    )
    
    parser.add_argument(
        "--particles",
        type=int,
        default=10000,
        help="Number of particles (default: 10000)"
    )
    
    parser.add_argument(
        "--activity",
        type=float,
        default=1e15,
        help="Total activity in Bq (default: 1e15)"
    )
    
    parser.add_argument(
        "--duration",
        type=float,
        default=86400,
        help="Simulation duration in seconds (default: 86400 = 24 hours)"
    )
    
    parser.add_argument(
        "--wind-speed",
        type=float,
        default=5.0,
        help="Wind speed in m/s (default: 5.0)"
    )
    
    parser.add_argument(
        "--wind-direction",
        type=float,
        default=0.0,
        help="Wind direction in degrees, 0=North (default: 0)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="fallout_output",
        help="Output filename prefix (default: fallout_output)"
    )
    
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.01,
        help="Output grid resolution in degrees (default: 0.01)"
    )
    
    args = parser.parse_args()
    
    # Load configuration if provided
    if args.config:
        config = load_config(args.config)
        source_lat = config.get("source_lat", args.lat)
        source_lon = config.get("source_lon", args.lon)
        source_altitude = config.get("source_altitude", args.altitude)
        num_particles = config.get("num_particles", args.particles)
        total_activity = config.get("total_activity", args.activity)
        duration = config.get("duration", args.duration)
        wind_speed = config.get("wind_speed", args.wind_speed)
        wind_direction = config.get("wind_direction", args.wind_direction)
        resolution = config.get("resolution", args.resolution)
        output = config.get("output", args.output)
    else:
        # Use command-line arguments
        if args.lat is None or args.lon is None:
            print("Error: --lat and --lon are required (or use --config)")
            sys.exit(1)
        
        source_lat = args.lat
        source_lon = args.lon
        source_altitude = args.altitude
        num_particles = args.particles
        total_activity = args.activity
        duration = args.duration
        wind_speed = args.wind_speed
        wind_direction = args.wind_direction
        resolution = args.resolution
        output = args.output
    
    print("=" * 60)
    print("Lagrangian Fallout Simulator")
    print("=" * 60)
    print(f"Source: ({source_lat:.4f}°, {source_lon:.4f}°) at {source_altitude}m")
    print(f"Particles: {num_particles}")
    print(f"Total activity: {total_activity:.2e} Bq")
    print(f"Duration: {duration/3600:.2f} hours")
    print(f"Wind: {wind_speed} m/s from {wind_direction}°")
    print("=" * 60)
    
    # Create atmospheric model
    atmosphere = AtmosphericModel(
        wind_speed=wind_speed,
        wind_direction=wind_direction
    )
    
    # Create simulator
    simulator = FalloutSimulator(
        source_lat=source_lat,
        source_lon=source_lon,
        source_altitude=source_altitude,
        num_particles=num_particles,
        total_activity=total_activity,
        atmospheric_model=atmosphere
    )
    
    # Run simulation
    print("\nRunning simulation...")
    simulator.run(duration=duration, progress_callback=progress_callback)
    
    # Get statistics
    stats = simulator.get_statistics()
    print("\nSimulation complete!")
    print(f"Deposited particles: {stats['deposited_particles']} "
          f"({stats['fraction_deposited']*100:.1f}%)")
    
    # Generate output
    print(f"\nGenerating output: {output}.png, {output}.pgw")
    raster = simulator.generate_output(
        filename=output,
        resolution=resolution
    )
    
    # Output statistics
    grid_stats = raster.get_grid_statistics()
    print("\nDeposition statistics:")
    print(f"  Total activity deposited: {grid_stats['total_activity_bq']:.2e} Bq")
    print(f"  Maximum cell activity: {grid_stats['max_activity_bq']:.2e} Bq")
    print(f"  Affected area: {grid_stats['affected_area_km2']:.2f} km²")
    print(f"  Affected cells: {grid_stats['affected_cells']} / {grid_stats['total_cells']}")
    
    print("\nOutput files created successfully!")
    print("Files are compatible with OPEN-RISOP and GIS software.")


if __name__ == "__main__":
    main()
