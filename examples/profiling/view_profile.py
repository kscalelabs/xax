#!/usr/bin/env python3
"""
Unified script to view JAX profiling results in TensorBoard and/or Perfetto UI.
This script handles the missing imghdr module issue in Python 3.13 and provides
extraction capabilities for trace files.
"""

import os
import sys
import subprocess
import argparse
import importlib.util
import webbrowser
from pathlib import Path
import glob
import tempfile
import json
import gzip
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="View JAX profiling results")
    parser.add_argument("--profile-dir", type=str, default=None,
                      help="Directory containing profiling data (default: auto-detect latest)")
    parser.add_argument("--port", type=int, default=6006,
                      help="Port to run TensorBoard on (default: 6006)")
    parser.add_argument("--bind-all", action="store_true", default=False,
                      help="Bind to all network interfaces (default: False)")
    parser.add_argument("--ui", type=str, choices=["tensorboard", "perfetto", "both"], default="both",
                      help="UI to open (tensorboard, perfetto, or both)")
    parser.add_argument("--extract", action="store_true", default=False,
                      help="Extract and decompress trace files for easier viewing in Perfetto")
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

def create_imghdr_patch_script():
    """Create a script that patches sys.modules with a fake imghdr."""
    patch_script = """
import sys

class FakeImghdr:
    def what(self, file, h=None):
        # Basic image type detection based on file extension
        if isinstance(file, str):
            if file.lower().endswith(('.png')):
                return 'png'
            elif file.lower().endswith(('.jpg', '.jpeg')):
                return 'jpeg'
            elif file.lower().endswith(('.gif')):
                return 'gif'
        return None

# Create test functions that TensorBoard might use
def test_jpeg(h, f):
    return h[0:2] == b'\\xff\\xd8' if h else False

def test_png(h, f):
    return h[0:8] == b'\\x89PNG\\r\\n\\x1a\\n' if h else False

def test_gif(h, f):
    return h[0:6] in (b'GIF87a', b'GIF89a') if h else False

# Create the module
fake_imghdr = FakeImghdr()
fake_imghdr.test_jpeg = test_jpeg
fake_imghdr.test_png = test_png
fake_imghdr.test_gif = test_gif
fake_imghdr.tests = [test_jpeg, test_png, test_gif]

# Add it to sys.modules before any imports happen
sys.modules["imghdr"] = fake_imghdr

# Now import and run tensorboard
import tensorboard.main
tensorboard.main.run_main()
"""
    
    # Create a temporary file
    fd, path = tempfile.mkstemp(suffix='.py', prefix='tb_imghdr_patch_')
    with os.fdopen(fd, 'w') as f:
        f.write(patch_script)
    
    return path

def check_tensorboard():
    """Check if TensorBoard is installed."""
    if importlib.util.find_spec("tensorboard") is None:
        print("TensorBoard not found. Please install it with: pip install tensorboard tensorboard-plugin-profile")
        return False
    return True

def run_tensorboard(logdir, port=6006, bind_all=False):
    """Run TensorBoard with the given logdir."""
    if not check_tensorboard():
        return False
    
    print(f"Starting TensorBoard with logdir: {logdir}")
    print(f"Access the interface at: http://localhost:{port}")
    
    # Create a patched script that handles the imghdr issue
    patch_script_path = create_imghdr_patch_script()
    
    cmd = [
        sys.executable,
        patch_script_path,
        "--logdir", str(logdir),
        "--port", str(port),
    ]
    
    if bind_all:
        cmd.append("--bind_all")
    
    try:
        process = subprocess.Popen(cmd)
        print("TensorBoard is running. Press Ctrl+C to stop.")
        return process
    except Exception as e:
        print(f"Error running TensorBoard: {e}")
        return None

def open_perfetto_ui(profile_dir, extract=False, list_only=False):
    """Open Perfetto UI and guide the user to select trace files."""
    trace_files = find_trace_files(profile_dir)
    
    if not trace_files:
        print(f"No trace files found in {profile_dir}")
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
    
    # Open Perfetto UI in browser if not list-only mode
    if not list_only:
        webbrowser.open("https://ui.perfetto.dev")
    
    return True

def main():
    args = parse_args()
    
    # Find the profile directory
    profile_dir = args.profile_dir
    if not profile_dir:
        profile_dir = find_latest_profile_dir()
        if not profile_dir:
            print("No profile directories found. Run a profiling example first.")
            return
    
    print(f"Using profile directory: {profile_dir}")
    
    # Just list trace files if list-only mode
    if args.list_only:
        trace_files = find_trace_files(profile_dir)
        if not trace_files:
            print(f"No trace files found in {profile_dir}")
            return
            
        print(f"Found {len(trace_files)} trace file(s):")
        for i, trace_file in enumerate(trace_files, 1):
            file_size = os.path.getsize(trace_file) / 1024  # KB
            print(f"{i}. {trace_file} ({file_size:.1f} KB)")
        return
    
    tb_process = None
    
    # Run TensorBoard if requested
    if args.ui in ["tensorboard", "both"]:
        tb_process = run_tensorboard(profile_dir, port=args.port, bind_all=args.bind_all)
    
    # Open Perfetto UI if requested
    if args.ui in ["perfetto", "both"]:
        open_perfetto_ui(profile_dir, extract=args.extract, list_only=args.list_only)
    
    # Wait for TensorBoard to exit if it was started
    if tb_process:
        try:
            tb_process.wait()
        except KeyboardInterrupt:
            print("\nStopping TensorBoard...")
            tb_process.terminate()
            tb_process.wait()
            print("TensorBoard stopped.")

if __name__ == "__main__":
    main() 