#!/usr/bin/env python3
"""
Performance test for multi-threaded vs single-threaded deposition
"""
import time
import numpy as np
from fallout_sim import HierarchicalGrid, Particle

def create_test_particles(n_particles=10000):
    """Create test particles for benchmarking"""
    particles = []
    for i in range(n_particles):
        p = Particle(
            lon=-120.0 + np.random.random() * 2.0,  # Random lon in range
            lat=35.0 + np.random.random() * 2.0,    # Random lat in range
            z=np.random.random() * 1000.0           # Random height
        )
        p.mass = 1.0
        particles.append(p)
    return particles

def benchmark_deposition():
    """Benchmark single-threaded vs multi-threaded deposition"""
    print("Creating test particles...")
    particles = create_test_particles(10000)

    print("Creating hierarchical grid...")
    grid = HierarchicalGrid(
        min_lon=-122.0, max_lon=-118.0,
        min_lat=33.0, max_lat=37.0,
        coarse_cell_size_km=10.0,
        fine_cell_size_m=100.0
    )

    # Test single-threaded deposition
    print("\nTesting single-threaded deposition...")
    start_time = time.time()
    grid.deposit(particles, max_workers=1)
    single_time = time.time() - start_time
    print(".2f")

    # Reset grid for fair comparison
    grid = HierarchicalGrid(
        min_lon=-122.0, max_lon=-118.0,
        min_lat=33.0, max_lat=37.0,
        coarse_cell_size_km=10.0,
        fine_cell_size_m=100.0
    )

    # Test multi-threaded deposition
    print("Testing multi-threaded deposition (4 workers)...")
    start_time = time.time()
    grid.deposit(particles, max_workers=4)
    multi_time = time.time() - start_time
    print(".2f")

    # Calculate speedup
    speedup = single_time / multi_time if multi_time > 0 else 0
    print(".2f")

    if speedup > 1.0:
        print("Multi-threading provides a performance benefit!")
    else:
        print("Multi-threading introduces overhead and slows things down.")

    return single_time, multi_time, speedup

if __name__ == "__main__":
    benchmark_deposition()
