# JAX Profiling in XAX

This extension adds comprehensive profiling capabilities to the XAX framework, allowing users to identify and eliminate performance bottlenecks in their JAX computation graphs.

## Key Features

- **Seamless Integration**: Profiling is available through a mixin that can be added to any task
- **Low Overhead**: Selective profiling only when needed to minimize performance impact
- **Detailed Analysis**: Captures XLA operation times, function call hierarchies, and compilation phases
- **Multiple Viewers**: Support for both TensorBoard and Perfetto UI with Python 3.13 compatibility
- **Easy Extraction**: Tools to extract and process trace files for easier analysis

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

We provide a unified viewer that supports both TensorBoard and Perfetto UI with full Python 3.13 compatibility:

```bash
# Unified viewer (TensorBoard + Perfetto)
python -m xax.examples.profiling.view_profile --profile-dir=path/to/profiles

# Additional options
python -m xax.examples.profiling.view_profile --profile-dir=path/to/profiles --ui=tensorboard  # TensorBoard only
python -m xax.examples.profiling.view_profile --profile-dir=path/to/profiles --ui=perfetto     # Perfetto UI only
python -m xax.examples.profiling.view_profile --profile-dir=path/to/profiles --extract         # Extract trace files
python -m xax.examples.profiling.view_profile --profile-dir=path/to/profiles --list-only       # List trace files
```

You can also programmatically open profiles:

```python
from xax.utils.profiling import open_latest_profile

# Opens the latest profile in TensorBoard
open_latest_profile("path/to/experiment/profiles")
```

## Examples

See the following example files for demonstrations of the profiling functionality:

- `examples/profiling/mnist_profiling.py` - Basic MNIST profiling example
- `examples/profiling/standalone_mnist_profiling.py` - Standalone MNIST profiling
- `examples/profiling/comprehensive_profile_demo.py` - Comprehensive profiling demo showing various use cases

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

## References

- [JAX Profiler Documentation](https://docs.jax.dev/en/latest/jax.profiler.html)
- [TensorBoard Profiling Guide](https://tensorflow.org/guide/profiler)
- [Perfetto UI Documentation](https://perfetto.dev/docs/visualization/perfetto-ui) 