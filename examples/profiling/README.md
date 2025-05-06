# XAX Profiling Examples

This directory contains examples demonstrating the JAX profiling capabilities in the XAX framework.

## Examples

### MNIST Profiling

`mnist_profiling.py` demonstrates:
- Setting up profiling configuration
- Using the profiling context manager
- Annotating functions for detailed profiling
- Profiling both training and validation steps

## Running Examples

To run the MNIST profiling example:

```bash
python -m xax.examples.profiling.mnist_profiling
```

After running, the profiling results will be saved to the experiment directory under the `profiles` subdirectory. You can view them using TensorBoard:

```bash
tensorboard --logdir=path/to/experiment/profiles
``` 