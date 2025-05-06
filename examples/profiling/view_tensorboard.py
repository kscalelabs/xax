#!/usr/bin/env python3
"""
Script to view JAX profiling results in TensorBoard.
This properly handles the issue with missing imghdr module in Python 3.13.
"""

import os
import sys
import subprocess
import argparse
import importlib.util
from pathlib import Path
import glob
import tempfile

def parse_args():
    parser = argparse.ArgumentParser(description="View JAX profiling results in TensorBoard")
    parser.add_argument("--profile-dir", type=str, default=None,
                      help="Directory containing profiling data (default: auto-detect latest)")
    parser.add_argument("--port", type=int, default=6006,
                      help="Port to run TensorBoard on (default: 6006)")
    parser.add_argument("--bind-all", action="store_true", default=False,
                      help="Bind to all network interfaces (default: False)")
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

def create_tensorboard_launcher():
    """
    Create a launcher script that injects a fake imghdr module before importing TensorBoard.
    This approach ensures the fake module is in place before any imports happen.
    """
    launcher_code = """
import sys

# Create and inject fake imghdr module BEFORE any imports
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
print("Fake imghdr module injected successfully")

# Now import and run tensorboard
import tensorboard.main
tensorboard.main.run_main()
"""
    
    # Create a temporary file with the launcher code
    fd, path = tempfile.mkstemp(suffix='.py', prefix='tb_launcher_')
    with os.fdopen(fd, 'w') as f:
        f.write(launcher_code)
    
    return path

def check_tensorboard():
    """Check if TensorBoard is installed."""
    if importlib.util.find_spec("tensorboard") is None:
        print("TensorBoard not found. Please install it with: pip install tensorboard tensorboard-plugin-profile")
        return False
    return True

def run_tensorboard(logdir, port=6006, bind_all=False):
    """Run TensorBoard with the given logdir using the launcher script."""
    if not check_tensorboard():
        return False
    
    print(f"Starting TensorBoard with logdir: {logdir}")
    print(f"Access the interface at: http://localhost:{port}")
    
    # Create a launcher script that handles the imghdr issue
    launcher_path = create_tensorboard_launcher()
    
    # Build command line arguments to pass to the launcher
    cmd = [
        sys.executable,
        launcher_path,
        "--logdir", str(logdir),
        "--port", str(port),
    ]
    
    if bind_all:
        cmd.append("--bind_all")
    
    try:
        print("Starting TensorBoard with Python 3.13 compatibility...")
        process = subprocess.Popen(cmd)
        print("TensorBoard is running. Press Ctrl+C to stop.")
        process.wait()
    except KeyboardInterrupt:
        print("\nStopping TensorBoard...")
        process.terminate()
        process.wait()
        print("TensorBoard stopped.")
    except Exception as e:
        print(f"Error running TensorBoard: {e}")
        return False
    finally:
        # Clean up the temporary launcher script
        try:
            os.unlink(launcher_path)
        except:
            pass
    
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
    
    # Run TensorBoard
    run_tensorboard(profile_dir, port=args.port, bind_all=args.bind_all)

if __name__ == "__main__":
    main() 