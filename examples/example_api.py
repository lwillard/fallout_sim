"""
Example script demonstrating the Python API.
"""

from fallout_sim import FalloutSimulator, AtmosphericModel


def main():
    """Run example simulation."""
    print("Creating atmospheric model...")
    atmosphere = AtmosphericModel(
        wind_speed=10.0,
        wind_direction=90.0,  # Eastward wind
        wind_speed_aloft=20.0
    )
    
    print("Initializing simulator...")
    simulator = FalloutSimulator(
        source_lat=40.0,
        source_lon=-100.0,
        source_altitude=1000.0,
        num_particles=5000,
        total_activity=1e15,
        atmospheric_model=atmosphere
    )
    
    print("Running simulation...")
    
    def progress(time, active):
        print(f"  Time: {time/3600:.1f}h, Active: {active}")
    
    simulator.run(duration=24*3600, progress_callback=progress)
    
    print("\nGenerating output...")
    raster = simulator.generate_output("example_api_output", resolution=0.02)
    
    print("\nSimulation statistics:")
    stats = simulator.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nGrid statistics:")
    grid_stats = raster.get_grid_statistics()
    for key, value in grid_stats.items():
        print(f"  {key}: {value}")
    
    print("\nDone! Check example_api_output.png and example_api_output.pgw")


if __name__ == "__main__":
    main()
