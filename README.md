# JAX Profiling in XAX

This extension adds profiling capabilities to the XAX framework, allowing users to identify performance bottlenecks in their JAX code.

## Getting Started

### 1. Enable profiling in your configuration

Add the following to your task configuration:

```python
@dataclass
class MyTaskConfig(TrainConfig):  # or ScriptConfig
    # Enable profiling
    enable_profiling: bool = True
    
    # Directory to save profiling results (optional)
    profiling_dir: Optional[str] = None
    
    # How often to run profiling (in steps)
    profile_every_n_steps: int = 100
    
    # Duration of each profiling session in milliseconds
    profile_duration_ms: int = 3000
    
    # Maximum number of profiling sessions to save
    max_profile_count: int = 5
```

### 2. Use the profiling context in your code

You can use the profiling context to profile specific sections of code:

```python
# In a task method
with self.profile_context(state, name="my_operation"):
    # Code to profile
    result = my_operation(...)
```

### 3. Annotate functions for more detailed profiling

Annotate important functions to see them in the profiling output:

```python
from xax.utils.profiling import annotate

@annotate("compute_loss")
def compute_loss(self, model, x, y):
    # Function implementation
    ...
```

### 4. View profiling results

Profiling results are saved to the experiment directory under the `profiles` subdirectory.
We provide multiple viewing utilities with full Python 3.13 compatibility:

```bash
# Combined TensorBoard and Perfetto UI viewer (recommended)
python -m xax.examples.profiling.view_profile --profile-dir=path/to/profiles

# TensorBoard only with Python 3.13 compatibility fix
python -m xax.examples.profiling.view_tensorboard --profile-dir=path/to/profiles

# Perfetto UI only (extracts and decompresses trace files)
python -m xax.examples.profiling.view_perfetto --profile-dir=path/to/profiles --extract
```

Or programmatically:

```python
from xax.utils.profiling import open_profile_viewer

# Opens the latest profile in the combined viewer
open_profile_viewer("path/to/experiment/profiles")
```

## Examples

See the `examples/profiling/` directory for complete examples:
- `mnist_profiling.py` - MNIST training with profiling
- `matrix_profile_demo.py` - Simple matrix operations profiling
- `comprehensive_profile_demo.py` - Advanced profiling capabilities

For more detailed documentation, see `examples/profiling/README.md`.

## Advanced Usage

### Tracing specific functions

You can trace specific functions without modifying them:

```python
from xax.utils.profiling import trace_function

# Wrap an existing function
original_fn = model.forward
model.forward = trace_function(original_fn, name="model_forward")
```

### Getting detailed GPU execution traces

Enable detailed GPU traces in your configuration:

```python
@dataclass
class MyTaskConfig(TrainConfig):
    enable_profiling: bool = True
    
    # Enable detailed GPU traces
    profile_gpu_trace: bool = True
```

This will capture more detailed information about GPU kernel execution times.

### Troubleshooting

If you encounter issues with TensorBoard profile visualization:
1. Ensure you have `tensorboard-plugin-profile` installed
2. Try using the dedicated `view_perfetto.py` script instead
3. See the troubleshooting section in `examples/profiling/README.md` 