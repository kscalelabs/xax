#!/usr/bin/env python3
"""
Comprehensive JAX profiling example.

This example demonstrates:
1. How to profile JAX operations with detailed annotations
2. How to capture both compilation and execution time
3. How to profile matrix operations of different sizes
4. How to view results in both TensorBoard and Perfetto UI
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
    parser.add_argument("--small-size", type=int, default=1024,
                        help="Size for smaller matrix operations (default: 1024)")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Number of iterations to profile (default: 3)")
    parser.add_argument("--output-dir", type=str, default="./profiles/comprehensive_profile",
                        help="Directory to save profiling data (default: ./profiles/comprehensive_profile)")
    parser.add_argument("--mode", type=str, choices=["all", "annotated", "simple", "small"], 
                        default="all", help="Which profiling mode to run")
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

def run_profile_session(func, profile_dir, iterations, name):
    """Run a profiling session for the given function."""
    print(f"\nStarting profile for {name}...")
    
    profile_dir.mkdir(parents=True, exist_ok=True)
    start_trace(str(profile_dir))
    
    for i in range(iterations):
        print(f"Running {name} iteration {i+1}/{iterations}...")
        results = func()
        # Force execution by blocking on all results
        for r in results:
            r.block_until_ready()
    
    stop_trace()
    print(f"{name} profiling completed!")
    return profile_dir

def main():
    args = parse_args()
    
    # Create profile directory
    profile_dir = Path(args.output_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"JAX devices: {jax.devices()}")
    print(f"Large matrix size: {args.size}x{args.size}")
    print(f"Small matrix size: {args.small_size}x{args.small_size}")
    print(f"Profile directory: {os.path.abspath(profile_dir)}")
    
    # Create JIT-compiled versions of all functions
    jitted_annotated = jax.jit(lambda: annotated_matrix_operations(args.size))
    jitted_simple = jax.jit(lambda: simple_matrix_operations(args.size))
    jitted_small = jax.jit(lambda: simple_matrix_operations(args.small_size))
    
    # Warmup (compile the functions)
    print("\nCompiling functions (warmup)...")
    if args.mode in ["all", "annotated"]:
        annotated_results = jitted_annotated()
        for r in annotated_results:
            r.block_until_ready()
    
    if args.mode in ["all", "simple"]:
        simple_results = jitted_simple()
        for r in simple_results:
            r.block_until_ready()
    
    if args.mode in ["all", "small"]:
        small_results = jitted_small()
        for r in small_results:
            r.block_until_ready()
            
    print("Compilation completed")
    
    # Run the selected profile sessions
    profile_dirs = []
    
    if args.mode in ["all", "annotated"]:
        # Profile: Annotated matrix operations
        annotated_profile_dir = profile_dir / "annotated"
        profile_dirs.append(run_profile_session(
            jitted_annotated, 
            annotated_profile_dir, 
            args.iterations, 
            "annotated matrix operations"
        ))
    
    if args.mode in ["all", "simple"]:
        # Profile: Simple large matrix operations
        simple_profile_dir = profile_dir / "simple"
        profile_dirs.append(run_profile_session(
            jitted_simple,
            simple_profile_dir,
            args.iterations,
            "simple large matrix operations"
        ))
    
    if args.mode in ["all", "small"]:
        # Profile: Small matrix operations (from matrix_profile_demo.py)
        small_profile_dir = profile_dir / "small"
        profile_dirs.append(run_profile_session(
            jitted_small,
            small_profile_dir,
            args.iterations,
            "small matrix operations"
        ))
    
    # Print viewing instructions
    print(f"\nProfiles saved to: {profile_dir}")
    print("\nTo view the profiles, use our unified viewer:")
    print(f"   python -m xax.examples.profiling.view_profile --profile-dir {profile_dir}")
    
    # Print additional options for viewing specific profiles
    if profile_dirs:
        print("\nOr view specific profile sessions:")
        for i, dir_path in enumerate(profile_dirs, 1):
            print(f"   {i}. {dir_path.name}: python -m xax.examples.profiling.view_profile --profile-dir {dir_path}")
    
    print("\nOptions for the viewer:")
    print("   --ui=tensorboard     View in TensorBoard only")
    print("   --ui=perfetto        View in Perfetto UI only")
    print("   --extract            Extract and decompress trace files for Perfetto UI")
    print("   --list-only          Just list available trace files")

if __name__ == "__main__":
    main() 