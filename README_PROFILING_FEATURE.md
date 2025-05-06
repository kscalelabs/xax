# JAX Profiling Feature for XAX

This feature adds comprehensive profiling capabilities to the XAX framework, allowing users to identify and eliminate performance bottlenecks in their JAX computation graphs.

## Key Features

- **Seamless Integration**: Profiling is available through a mixin that can be added to any task
- **Low Overhead**: Selective profiling only when needed to minimize performance impact
- **Detailed Analysis**: Captures XLA operation times, function call hierarchies, and compilation phases
- **Multiple Viewers**: Support for both TensorBoard and Perfetto UI with Python 3.13 compatibility
- **Easy Extraction**: Tools to extract and process trace files for easier analysis

## Documentation

Comprehensive documentation is available in the following locations:
- [Profiling Guide](docs/profiling.md) - Complete usage documentation
- [Example Code](examples/profiling/) - MNIST and comprehensive profiling examples
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

## Viewing Profile Results

We provide multiple tools for viewing profiles with full Python 3.13 compatibility:

```bash
# Combined viewer (TensorBoard + Perfetto)
python -m xax.examples.profiling.view_profile --profile-dir=path/to/profiles

# TensorBoard only with Python 3.13 fix
python -m xax.examples.profiling.view_tensorboard --profile-dir=path/to/profiles

# Perfetto UI with extraction (recommended for detailed analysis)
python -m xax.examples.profiling.view_perfetto --profile-dir=path/to/profiles --extract
```

## References

- [JAX Profiler Documentation](https://docs.jax.dev/en/latest/jax.profiler.html)
- [TensorBoard Profiling Guide](https://tensorflow.org/guide/profiler)
- [Perfetto UI Documentation](https://perfetto.dev/docs/visualization/perfetto-ui) 