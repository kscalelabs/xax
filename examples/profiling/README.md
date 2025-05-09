# XAX Profiling Examples

This directory contains examples demonstrating the JAX profiling capabilities in the XAX framework.

## Examples

### MNIST Profiling

`mnist_profiling.py` demonstrates:
- Setting up profiling configuration
- Using the profiling context manager
- Annotating functions for detailed profiling
- Profiling both training and validation steps

### Matrix Operations Profiling

`matrix_profile_demo.py` is a simpler example that demonstrates:
- How to profile JAX matrix operations
- How to capture both compilation and execution time
- How to generate profile traces for visualization

### Comprehensive Profiling Demo

`comprehensive_profile_demo.py` shows advanced profiling capabilities:
- Detailed operation annotations
- Multiple profiling sessions
- Comparison of annotated vs. non-annotated profiles
- Robust support for both TensorBoard and Perfetto UI

## Running Examples

### MNIST Profiling Example

```bash
python -m xax.examples.profiling.mnist_profiling
```

### Matrix Operations Example

```bash
python -m xax.examples.profiling.matrix_profile_demo
```

### Comprehensive Profiling Demo

```bash
python -m xax.examples.profiling.comprehensive_profile_demo
```

You can customize the comprehensive profiling example with these options:
- `--size N` - Size of matrices (default: 2000)
- `--output-dir DIR` - Directory to save profiling data (default: ./profiles/comprehensive_profile)
- `--iterations N` - Number of iterations to profile (default: 3)

## Viewing Profiling Results

After running, the profiling results will be saved to the experiment directory under the `profiles` subdirectory. We provide multiple ways to view these results:

### Option 1: Combined Viewer (TensorBoard + Perfetto)

For the easiest viewing experience, use our combined viewer which handles both TensorBoard and Perfetto UI:

```bash
python -m xax.examples.profiling.view_profile --profile-dir path/to/profiles
```

This will:
1. Start TensorBoard with Python 3.13 compatibility fixes
2. Open Perfetto UI in your browser with instructions for loading trace files

### Option 2: TensorBoard Only (with Python 3.13 Fix)

If you prefer to use TensorBoard's interface:

```bash
python -m xax.examples.profiling.view_tensorboard --profile-dir path/to/profiles
```

This script:
1. Handles the missing `imghdr` module issue in Python 3.13
2. Starts TensorBoard with the profile data
3. Opens the interface at http://localhost:6006

Then, navigate to the 'Profile' tab to see the visualization.

### Option 3: Perfetto UI Only (Recommended for Advanced Analysis)

For the most detailed profile analysis, use our dedicated Perfetto UI viewer:

```bash
python -m xax.examples.profiling.view_perfetto --profile-dir path/to/profiles --extract
```

This will:
1. Find and extract all trace files from the profile directory
2. Open Perfetto UI in your browser
3. Guide you to select the extracted trace files

The `--extract` flag decompresses the trace files for easier loading. You can also use `--list-only` to just see available trace files without opening the browser.

## Understanding Profiling Results

The profile visualization shows:

1. **Timeline View** - Shows when each operation executed
2. **Execution Time** - Length of bars indicates duration
3. **Compilation vs. Execution** - Different phases of JAX's operation
4. **Memory Operations** - Shows memory allocations and transfers

Look for:
- Long-running operations (potential bottlenecks)
- Unexpected compilation during execution
- Excessive memory transfers
- Repeated operations that could be optimized

## Troubleshooting

### TensorBoard Profile Issues

If you encounter issues with TensorBoard not showing profiles:

1. Try using the `view_tensorboard.py` script which handles Python 3.13 compatibility
2. Ensure you have `tensorboard-plugin-profile` installed: `pip install tensorboard-plugin-profile`
3. If problems persist, use the Perfetto UI viewer instead (`view_perfetto.py`), which is more reliable

### Perfetto UI Tips

When using Perfetto UI:
1. After loading a trace, use the WASD keys to navigate (W/S to zoom, A/D to pan)
2. Use the '?' key to show keyboard shortcuts
3. Search for specific events using the search box at the top
4. Use the 'M' key to mark areas of interest

## Implementation Details

The profiling implementation uses JAX's built-in profiling capabilities:
- `start_trace()` and `stop_trace()` functions capture performance data
- Profiling is non-intrusive and works with JIT compilation
- Results are saved as standard trace files compatible with Chrome Tracing Protocol
- Multiple visualization options are provided for flexibility
- Profiling has minimal overhead when not active 