# JAX Profiling Feature for XAX

This feature adds comprehensive profiling capabilities to the XAX framework, allowing users to identify and eliminate performance bottlenecks in their JAX computation graphs.

## Key Features

- **Seamless Integration**: Profiling is available through a mixin that can be added to any task
- **Low Overhead**: Selective profiling only when needed to minimize performance impact
- **Detailed Analysis**: Captures XLA operation times, function call hierarchies, and compilation phases
- **Easy Visualization**: Integration with TensorBoard for intuitive timeline displays

## Documentation

Comprehensive documentation is available in the following locations:
- [Profiling Guide](docs/profiling.md) - Complete usage documentation
- [Example Code](examples/profiling/) - MNIST example with profiling
- [PR Description](docs/PR_DESCRIPTION.md) - Technical details of implementation

## Getting Started

```python
@dataclass
class MyTaskConfig(TrainConfig):
    # Enable profiling
    enable_profiling: bool = True
    profile_every_n_steps: int = 100

# In your code
with self.profile_context(state, name="computation_name"):
    # Code to profile
    result = expensive_computation()
```

## References

- [JAX Profiler Documentation](https://docs.jax.dev/en/latest/jax.profiler.html)
- [TensorBoard Profiling Guide](https://tensorflow.org/guide/profiler) 