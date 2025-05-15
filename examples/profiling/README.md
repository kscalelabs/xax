# XAX Profiling Examples

This directory contains examples demonstrating the JAX profiling capabilities in the XAX framework.

## Examples

### MNIST Profiling

`mnist_profiling.py` demonstrates:
- Setting up profiling configuration
- Using the profiling context manager
- Annotating functions for detailed profiling
- Profiling both training and validation steps

### Standalone MNIST Profiling

`standalone_mnist_profiling.py` provides:
- A self-contained example of JAX profiling
- No dependencies on other XAX modules
- Clear visualization of forward and backward passes

### Comprehensive Profiling Demo

`comprehensive_profile_demo.py` shows advanced profiling capabilities:
- Detailed operation annotations
- Multiple profiling sessions with different matrix sizes
- Comparison of annotated vs. non-annotated profiles
- Robust support for both TensorBoard and Perfetto UI

## Running Examples

### MNIST Profiling Example

```bash
python -m xax.examples.profiling.mnist_profiling
```

### Standalone MNIST Example

```bash
python -m xax.examples.profiling.standalone_mnist_profiling
```

### Comprehensive Profiling Demo

```bash
python -m xax.examples.profiling.comprehensive_profile_demo
```

You can customize the comprehensive profiling example with these options:
- `--size N` - Size of large matrices (default: 2000)
- `--small-size N` - Size of small matrices (default: 1024)
- `--output-dir DIR` - Directory to save profiling data (default: ./profiles/comprehensive_profile)
- `--iterations N` - Number of iterations to profile (default: 3)
- `--mode MODE` - Which profiling mode to run (all, annotated, simple, small)

## Viewing Profiling Results

After running, the profiling results will be saved to the experiment directory under the `profiles` subdirectory. We provide a unified viewer that supports both TensorBoard and Perfetto UI:

```bash
python -m xax.examples.profiling.view_profile --profile-dir path/to/profiles
```

### Viewer Options

The unified viewer supports several options:

```bash
# View in TensorBoard only
python -m xax.examples.profiling.view_profile --profile-dir path/to/profiles --ui=tensorboard

# View in Perfetto UI only
python -m xax.examples.profiling.view_profile --profile-dir path/to/profiles --ui=perfetto

# Extract and decompress trace files for easier loading in Perfetto UI
python -m xax.examples.profiling.view_profile --profile-dir path/to/profiles --extract

# Just list available trace files without opening a browser
python -m xax.examples.profiling.view_profile --profile-dir path/to/profiles --list-only

# Use a specific port for TensorBoard
python -m xax.examples.profiling.view_profile --profile-dir path/to/profiles --port 8080

# Bind TensorBoard to all network interfaces
python -m xax.examples.profiling.view_profile --profile-dir path/to/profiles --bind-all
```

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

1. Ensure you have `tensorboard-plugin-profile` installed: `pip install tensorboard-plugin-profile`
2. Try using the `--ui=perfetto` option to use Perfetto UI instead
3. Use `--extract` to decompress trace files for easier loading

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
- Python 3.13 compatibility ensures it works with the latest Python release
- Profiling has minimal overhead when not active 