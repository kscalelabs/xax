#!/usr/bin/env python3
"""
Script to view JAX profiling results in either TensorBoard or Perfetto UI.
Handles the missing imghdr module issue in Python 3.13.
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
    """Find trace files in the profile directory."""
    trace_pattern = os.path.join(profile_dir, "plugins", "profile", "*", "*.trace.json.gz")
    return glob.glob(trace_pattern)

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

def open_perfetto_ui(profile_dir):
    """Open Perfetto UI and guide the user to select trace files."""
    trace_files = find_trace_files(profile_dir)
    
    if not trace_files:
        print(f"No trace files found in {profile_dir}")
        return False
    
    print("\nTo view profiles in Perfetto UI:")
    print("1. Opening https://ui.perfetto.dev in your browser")
    print("2. Click 'Open trace file' and select one of these files:")
    for trace_file in trace_files:
        print(f"   - {os.path.abspath(trace_file)}")
    
    # Open Perfetto UI in the browser
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
    
    tb_process = None
    
    # Run TensorBoard if requested
    if args.ui in ["tensorboard", "both"]:
        tb_process = run_tensorboard(profile_dir, port=args.port, bind_all=args.bind_all)
    
    # Open Perfetto UI if requested
    if args.ui in ["perfetto", "both"]:
        open_perfetto_ui(profile_dir)
    
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