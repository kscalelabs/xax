# XAX: JAX Framework with Profiling Support

XAX is an experimentation framework for JAX with comprehensive profiling capabilities, allowing users to identify performance bottlenecks in their JAX code.

## Features

- **Profiling**: Capture detailed profiles of JAX computations for optimization
- **TensorBoard & Perfetto Integration**: Visualize profiles using industry-standard tools
- **Python 3.13 Compatible**: Full compatibility with the latest Python release

## Getting Started with Profiling

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
We provide a unified viewing utility with full Python 3.13 compatibility:

```bash
# Unified profiling viewer (supports both TensorBoard and Perfetto UI)
python -m xax.examples.profiling.view_profile --profile-dir=path/to/profiles

# Additional options
python -m xax.examples.profiling.view_profile --ui=tensorboard     # TensorBoard only
python -m xax.examples.profiling.view_profile --ui=perfetto        # Perfetto UI only
python -m xax.examples.profiling.view_profile --extract            # Extract trace files
python -m xax.examples.profiling.view_profile --list-only          # List trace files
```

## Examples

See the `examples/profiling/` directory for complete examples:
- `mnist_profiling.py` - MNIST training with profiling
- `standalone_mnist_profiling.py` - Standalone MNIST profiling 
- `comprehensive_profile_demo.py` - Advanced profiling capabilities for various matrix operations

## Documentation

For comprehensive documentation about the profiling feature, see:
- `docs/profiling.md` - Complete usage guide and advanced features
- `examples/profiling/README.md` - Example-specific documentation

## Advanced Troubleshooting

If you encounter issues with profile visualization:
1. Ensure you have `tensorboard-plugin-profile` installed
2. Try using the `--extract` option to extract trace files for Perfetto UI
3. Use the `--list-only` option to verify that trace files are being generated 