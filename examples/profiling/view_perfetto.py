#!/usr/bin/env python3
"""
Script to view JAX profiling results in Perfetto UI.
This script helps find and open trace files in the browser.
"""

import os
import sys
import glob
import webbrowser
import argparse
from pathlib import Path
import gzip
import json
import tempfile
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="View JAX profile traces in Perfetto UI")
    parser.add_argument("--profile-dir", type=str, default=None,
                      help="Directory containing profiling data (default: auto-detect latest)")
    parser.add_argument("--extract", action="store_true", default=False,
                      help="Extract and decompress trace files for easier viewing")
    parser.add_argument("--list-only", action="store_true", default=False,
                      help="Only list trace files without opening browser")
    return parser.parse_args()

def find_latest_profile_dir():
    """Find the most recent profile directory."""
    # Common profile locations
    search_paths = [
        "./profiles/*/",
        "./profiles/*/*/",
        "./*/profiles/*/",
    ]
    
    all_profile_dirs = []
    for pattern in search_paths:
        all_profile_dirs.extend(glob.glob(pattern))
    
    if not all_profile_dirs:
        return None
    
    # Sort by modification time (newest last)
    return sorted(all_profile_dirs, key=lambda p: os.path.getmtime(p))[-1]

def find_trace_files(profile_dir):
    """Find all trace files in the profile directory and subdirectories."""
    # Search patterns for trace files
    trace_patterns = [
        os.path.join(profile_dir, "**", "*.trace.json.gz"),
        os.path.join(profile_dir, "**", "*.trace.json"),
    ]
    
    trace_files = []
    for pattern in trace_patterns:
        trace_files.extend(glob.glob(pattern, recursive=True))
    
    return trace_files

def extract_trace_file(trace_file, output_dir=None):
    """Extract and decompress a trace file to make it easier to load in Perfetto."""
    if not output_dir:
        output_dir = tempfile.mkdtemp(prefix="jax_profile_")
    
    # Create a more readable filename
    base_name = os.path.basename(trace_file)
    if base_name.endswith('.gz'):
        output_name = base_name[:-3]  # Remove .gz extension
    else:
        output_name = base_name
        
    # Add a timestamp to avoid overwrites
    timestamp = os.path.getmtime(trace_file)
    parts = output_name.split('.')
    if len(parts) > 1:
        output_name = f"{'.'.join(parts[:-1])}_{int(timestamp)}.{parts[-1]}"
    else:
        output_name = f"{output_name}_{int(timestamp)}"
    
    output_path = os.path.join(output_dir, output_name)
    
    # Extract content
    if trace_file.endswith('.gz'):
        with gzip.open(trace_file, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        shutil.copy(trace_file, output_path)
    
    return output_path

def open_perfetto_ui(trace_files, extract=False):
    """Open Perfetto UI and guide user to open trace files."""
    if not trace_files:
        print("No trace files found")
        return False
    
    # Process trace files
    processed_files = []
    if extract:
        extract_dir = tempfile.mkdtemp(prefix="jax_traces_")
        print(f"\nExtracting trace files to: {extract_dir}")
        
        for trace_file in trace_files:
            extracted = extract_trace_file(trace_file, extract_dir)
            processed_files.append(extracted)
            
        print(f"Extracted {len(processed_files)} trace files")
        files_to_show = processed_files
    else:
        files_to_show = [os.path.abspath(f) for f in trace_files]
    
    # Print instructions for viewing
    print("\nTo view profiles in Perfetto UI:")
    print("1. Opening https://ui.perfetto.dev in your browser")
    print("2. Click 'Open trace file' and select one of these files:")
    for i, trace_file in enumerate(files_to_show, 1):
        file_size = os.path.getsize(trace_file) / 1024  # Convert to KB
        print(f"   {i}. {trace_file} ({file_size:.1f} KB)")
    
    # Open Perfetto UI in browser
    webbrowser.open("https://ui.perfetto.dev")
    
    return True

def main():
    args = parse_args()
    
    # Find profile directory
    profile_dir = args.profile_dir
    if not profile_dir:
        profile_dir = find_latest_profile_dir()
        if not profile_dir:
            print("No profile directories found. Run a profiling example first.")
            return
    
    print(f"Using profile directory: {profile_dir}")
    
    # Find trace files
    trace_files = find_trace_files(profile_dir)
    
    if not trace_files:
        print(f"No trace files found in {profile_dir}")
        return
    
    print(f"Found {len(trace_files)} trace file(s)")
    
    # Open in Perfetto UI if not list-only mode
    if not args.list_only:
        open_perfetto_ui(trace_files, extract=args.extract)
    else:
        # Just list the files
        for i, trace_file in enumerate(trace_files, 1):
            file_size = os.path.getsize(trace_file) / 1024  # KB
            print(f"{i}. {trace_file} ({file_size:.1f} KB)")

if __name__ == "__main__":
    main() 