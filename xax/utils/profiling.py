"""Utilities for analyzing JAX profiling results."""

import os
import functools
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

import jax
from jax.profiler import annotate_function

def annotate(name: str) -> Callable:
    """Decorator to annotate a function in the profiling output.
    
    Args:
        name: Name to give this function in the profile.
        
    Returns:
        A decorator that can be applied to a function.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with jax.profiler.TraceAnnotation(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def trace_function(fn: Callable, name: Optional[str] = None) -> Callable:
    """Wrap a function to trace its execution in profiling.
    
    Args:
        fn: The function to trace.
        name: Optional name for the traced function. If None, uses the function name.
        
    Returns:
        A wrapped version of the function that will be annotated in profiling.
    """
    name = name or fn.__name__
    
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with jax.profiler.TraceAnnotation(name):
            return fn(*args, **kwargs)
    
    return wrapped

def open_latest_profile(profile_dir: Optional[str] = None) -> None:
    """Open the latest profile in TensorBoard.
    
    Args:
        profile_dir: Directory containing profiling data. If None, uses the default.
    """
    import subprocess
    import tempfile
    
    if profile_dir is None:
        profile_dir = os.environ.get("XAX_PROFILE_DIR", None)
        if profile_dir is None:
            raise ValueError("No profile directory specified and XAX_PROFILE_DIR not set")
    
    profile_dir = Path(profile_dir)
    if not profile_dir.exists():
        raise ValueError(f"Profile directory {profile_dir} does not exist")
    
    # Find the most recent profile
    profiles = list(profile_dir.glob("*"))
    if not profiles:
        raise ValueError(f"No profiles found in {profile_dir}")
    
    latest_profile = max(profiles, key=lambda p: p.stat().st_mtime)
    
    # Launch TensorBoard
    subprocess.run(["tensorboard", "--logdir", str(latest_profile)]) 