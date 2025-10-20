"""
Basic tests for fallout simulator.
"""

import numpy as np
import pytest
from fallout_sim import FalloutSimulator, AtmosphericModel, Particle, ParticleCloud, RasterOutput


def test_particle_creation():
    """Test basic particle creation."""
    particle = Particle(
        lat=40.0,
        lon=-100.0,
        altitude=1000.0,
        diameter=1e-5,
        density=2500.0,
        activity=1e9
    )
    
    assert particle.lat == 40.0
    assert particle.lon == -100.0
    assert particle.altitude == 1000.0
    assert not particle.deposited
    assert particle.terminal_velocity > 0


def test_particle_cloud():
    """Test particle cloud creation."""
    cloud = ParticleCloud()
    cloud.create_particles(
        source_lat=40.0,
        source_lon=-100.0,
        source_altitude=1000.0,
        num_particles=100,
        activity_per_particle=1e9
    )
    
    assert len(cloud.particles) == 100
    assert len(cloud.get_active_particles()) == 100
    assert len(cloud.get_deposited_particles()) == 0


def test_atmospheric_model():
    """Test atmospheric model wind calculations."""
    atm = AtmosphericModel(
        wind_speed=10.0,
        wind_direction=90.0  # East wind
    )
    
    u, v, w = atm.get_wind_velocity(40.0, -100.0, 100.0, 0.0)
    
    # East wind means negative u (wind FROM east)
    assert u < 0
    assert abs(u) > 0
    assert w == 0  # No vertical wind


def test_simulator_initialization():
    """Test simulator initialization."""
    sim = FalloutSimulator(
        source_lat=40.0,
        source_lon=-100.0,
        source_altitude=1000.0,
        num_particles=100,
        total_activity=1e15
    )
    
    assert sim.source_lat == 40.0
    assert sim.source_lon == -100.0
    assert len(sim.particle_cloud.particles) == 100


def test_simulator_step():
    """Test single simulation step."""
    sim = FalloutSimulator(
        source_lat=40.0,
        source_lon=-100.0,
        source_altitude=500.0,
        num_particles=10,
        total_activity=1e12
    )
    
    initial_positions = sim.particle_cloud.get_positions()
    sim.step(dt=60.0)
    new_positions = sim.particle_cloud.get_positions()
    
    # Particles should have moved
    if len(new_positions) > 0 and len(initial_positions) > 0:
        assert not np.allclose(initial_positions, new_positions)


def test_simulator_run():
    """Test full simulation run."""
    atm = AtmosphericModel(wind_speed=5.0, wind_direction=90.0)
    
    sim = FalloutSimulator(
        source_lat=40.0,
        source_lon=-100.0,
        source_altitude=100.0,  # Low altitude for quick deposition
        num_particles=50,
        total_activity=1e12,
        atmospheric_model=atm
    )
    
    sim.run(duration=3600)  # 1 hour
    
    stats = sim.get_statistics()
    assert stats['total_particles'] == 50
    assert stats['deposited_particles'] > 0  # Some should have deposited


def test_raster_output():
    """Test raster output creation."""
    raster = RasterOutput(
        bounds=(-101.0, 39.0, -99.0, 41.0),
        resolution=0.1
    )
    
    # Add some deposition data
    lats = np.array([40.0, 40.1, 40.2])
    lons = np.array([-100.0, -100.1, -100.2])
    activities = np.array([1e9, 2e9, 3e9])
    
    raster.add_deposition_data(lats, lons, activities)
    
    assert np.sum(raster.grid) > 0
    
    stats = raster.get_grid_statistics()
    assert stats['total_activity_bq'] > 0
    assert stats['affected_cells'] > 0


def test_deposition():
    """Test particle deposition."""
    sim = FalloutSimulator(
        source_lat=40.0,
        source_lon=-100.0,
        source_altitude=10.0,  # Very low altitude
        num_particles=20,
        total_activity=1e12
    )
    
    # Run for sufficient time for all to deposit
    sim.run(duration=7200)
    
    stats = sim.get_statistics()
    # At low altitude, most particles should deposit
    assert stats['deposited_particles'] > 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
