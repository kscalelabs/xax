#!/usr/bin/env python3
"""
Comprehensive JAX profiling example.

This example demonstrates:
1. How to profile JAX operations with detailed annotations
2. How to capture both compilation and execution time
3. How to view results in both TensorBoard and Perfetto UI
"""

import jax
import jax.numpy as jnp
from jax.profiler import start_trace, stop_trace
import numpy as np
import os
import time
import argparse
from pathlib import Path
import contextlib

def parse_args():
    parser = argparse.ArgumentParser(description="Run JAX profiling demo")
    parser.add_argument("--size", type=int, default=2000, 
                        help="Size of matrices (default: 2000)")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Number of iterations to profile (default: 3)")
    parser.add_argument("--output-dir", type=str, default="./profiles/comprehensive_profile",
                        help="Directory to save profiling data (default: ./profiles/comprehensive_profile)")
    return parser.parse_args()

# Context manager for annotating sections of the profile
@contextlib.contextmanager
def trace_context(name):
    """Context manager for profiling annotations."""
    try:
        yield
    finally:
        pass  # No-op in this implementation, but useful for documentation

def annotated_matrix_operations(size):
    """Perform various matrix operations with annotations for profiling."""
    # Create some data
    with trace_context("create_matrices"):
        x = jnp.ones((size, size))
        y = jnp.ones((size, size))
    
    # Matrix multiplication
    with trace_context("matrix_multiply"):
        z1 = jnp.dot(x, y)
    
    # Element-wise addition
    with trace_context("matrix_add"):
        z2 = x + y
    
    # Matrix transpose
    with trace_context("matrix_transpose"):
        z3 = jnp.transpose(x)
    
    # Element-wise operations
    with trace_context("matrix_elementwise"):
        z4 = jnp.sin(x) + jnp.cos(y)
    
    # Reduction operation
    with trace_context("matrix_reduce"):
        z5 = jnp.sum(x, axis=0)
    
    return z1, z2, z3, z4, z5

def simple_matrix_operations(size):
    """Perform matrix operations without individual annotations."""
    # Create data
    x = jnp.ones((size, size))
    y = jnp.ones((size, size))
    
    # Perform operations
    z1 = jnp.dot(x, y)
    z2 = x + y
    z3 = jnp.transpose(x)
    z4 = jnp.sin(x) + jnp.cos(y)
    z5 = jnp.sum(x, axis=0)
    
    return z1, z2, z3, z4, z5

def main():
    args = parse_args()
    
    # Create profile directory
    profile_dir = Path(args.output_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"JAX devices: {jax.devices()}")
    print(f"Matrix size: {args.size}x{args.size}")
    print(f"Profile directory: {os.path.abspath(profile_dir)}")
    
    # Create JIT-compiled versions of both functions
    jitted_annotated = jax.jit(lambda: annotated_matrix_operations(args.size))
    jitted_simple = jax.jit(lambda: simple_matrix_operations(args.size))
    
    # Warmup (compile the functions)
    print("\nCompiling functions (warmup)...")
    annotated_results = jitted_annotated()
    # Force execution by blocking on all results
    for r in annotated_results:
        r.block_until_ready()
    
    simple_results = jitted_simple()
    # Force execution by blocking on all results
    for r in simple_results:
        r.block_until_ready()
    print("Compilation completed")
    
    # First profile: Annotated matrix operations
    annotated_profile_dir = profile_dir / "annotated"
    annotated_profile_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nStarting profile for annotated matrix operations...")
    start_trace(str(annotated_profile_dir))
    
    for i in range(args.iterations):
        print(f"Running annotated iteration {i+1}/{args.iterations}...")
        results = jitted_annotated()
        # Force execution by blocking on all results
        for r in results:
            r.block_until_ready()
    
    stop_trace()
    print("Annotated profiling completed!")
    
    # Second profile: Simple matrix operations
    simple_profile_dir = profile_dir / "simple"
    simple_profile_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nStarting profile for simple matrix operations...")
    start_trace(str(simple_profile_dir))
    
    for i in range(args.iterations):
        print(f"Running simple iteration {i+1}/{args.iterations}...")
        results = jitted_simple()
        # Force execution by blocking on all results
        for r in results:
            r.block_until_ready()
    
    stop_trace()
    print("Simple profiling completed!")
    
    # Print viewing instructions
    print(f"\nProfiles saved to: {profile_dir}")
    print("\nTo view the profiles, you can use either:")
    
    print("\n1. Our combined viewer (TensorBoard + Perfetto):")
    print(f"   python -m xax.examples.profiling.view_profile --profile-dir {profile_dir}")
    
    print("\n2. TensorBoard only (with fixed Python 3.13 compatibility):")
    print(f"   python -m xax.examples.profiling.view_tensorboard --profile-dir {profile_dir}")
    print("   Then open http://localhost:6006 in your browser")
    print("   Click on the 'Profile' tab in the navigation bar")
    
    print("\n3. Manually with Perfetto UI:")
    print("   Go to https://ui.perfetto.dev")
    print("   Click 'Open trace file' and select one of the trace files in:")
    print(f"   - {annotated_profile_dir}/plugins/profile/*/traces/*.trace.json.gz")
    print(f"   - {simple_profile_dir}/plugins/profile/*/traces/*.trace.json.gz")

if __name__ == "__main__":
    main() 