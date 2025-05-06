#!/usr/bin/env python3
"""
Matrix operations profiling example using JAX's built-in profiling tools.

This example demonstrates:
1. How to profile JAX matrix operations
2. How to capture both compilation and execution time
3. How to generate profile traces for visualization
"""

import os
import jax
import jax.numpy as jnp
from jax.profiler import start_trace, stop_trace
import time
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run JAX profiling on matrix operations")
    parser.add_argument("--size", type=int, default=1024, 
                        help="Size of matrices (default: 1024)")
    parser.add_argument("--output-dir", type=str, default="./profiles/matrix_operations",
                        help="Directory to save profiling data (default: ./profiles/matrix_operations)")
    parser.add_argument("--iterations", type=int, default=2,
                        help="Number of iterations after compilation (default: 2)")
    return parser.parse_args()

def matrix_operations(size):
    """Perform various matrix operations to profile."""
    # Create some data
    x = jnp.ones((size, size))
    y = jnp.ones((size, size))
    
    # Perform matrix operations
    z1 = jnp.dot(x, y)
    z2 = x + y
    z3 = jnp.transpose(x)
    z4 = jnp.sin(x) + jnp.cos(y)
    z5 = jnp.sum(x, axis=0)
    
    return z1, z2, z3, z4, z5

def main():
    args = parse_args()
    
    # Create a directory for the profile
    profile_dir = Path(args.output_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    print(f"JAX devices: {jax.devices()}")
    print(f"Profile directory: {os.path.abspath(profile_dir)}")

    # JIT-compile the function
    jit_matrix_operations = jax.jit(lambda: matrix_operations(args.size))

    # Start profiling session
    print("\nStarting profiling session...")
    start_trace(str(profile_dir))

    # First run (compilation)
    print("Compiling and running...")
    results = jit_matrix_operations()
    # Force execution by blocking on all results
    for r in results:
        r.block_until_ready()
    print("Completed first run (compilation)")

    # Additional runs (execution only)
    for i in range(args.iterations):
        print(f"Running iteration {i+1}/{args.iterations}...")
        results = jit_matrix_operations()
        # Force execution by blocking on all results
        for r in results:
            r.block_until_ready()
        print(f"Completed iteration {i+1}")

    # Stop profiling
    stop_trace()

    print(f"\nProfile saved to: {profile_dir}")
    print("\nTo view the profile, you can:")
    print("1. Use TensorBoard:")
    print(f"   tensorboard --logdir={profile_dir}")
    print("   Then open your browser to http://localhost:6006")
    print("   Click on the 'Profile' tab in the navigation bar")
    print("\n2. Use Perfetto UI:")
    print("   Go to https://ui.perfetto.dev")
    print("   Click 'Open trace file' and select the .trace.json.gz file in the profile directory")
    print("\n3. Use Chrome Tracing:")
    print("   Open Chrome and go to chrome://tracing")
    print("   Click 'Load' and select the .trace.json.gz file")

if __name__ == "__main__":
    main() 