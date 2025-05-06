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
You can view them using TensorBoard:

```bash
tensorboard --logdir=path/to/experiment/profiles
```

Or programmatically:

```python
from xax.utils.profiling import open_latest_profile

# Opens the latest profile in TensorBoard
open_latest_profile("path/to/experiment/profiles")
```

## Example

See the `examples/mnist_profiling.py` file for a complete example of using the profiling functionality.

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