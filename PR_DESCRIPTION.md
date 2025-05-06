# Mixed Precision Training Integration

This PR adds comprehensive mixed precision training support to XAX, enabling faster training and reduced memory usage while maintaining model accuracy.

## Features

- **Mixed Precision Mixin**: Adds a reusable `MixedPrecisionMixin` to handle mixed precision training logic
- **Flexible Precision Policies**: Support for custom data type policies for parameters, computation, and outputs
- **Loss Scaling**: Implements static and dynamic loss scaling to prevent gradient underflow
- **Integrated with Training Loop**: Transparently works with existing `TrainMixin`
- **Examples & Benchmarks**: Includes MNIST example and benchmarking tools to demonstrate usage and performance
- **Hardware Awareness**: Automatically selects the optimal half-precision format (float16 for GPU, bfloat16 for TPU)
- **Documentation**: Comprehensive documentation in `docs/mixed_precision.md`

## Implementation Details

- Added mixed precision utilities in `xax/utils/mixed_precision.py`
- Integrated mixed precision support into `TrainMixin` 
- Created configuration options in `TrainConfig` for easy enabling/disabling
- Added MNIST example to showcase usage in real training scenarios
- Created benchmarking scripts to measure performance benefits
- Added comprehensive unit tests to ensure correctness

## Testing

- **Unit Tests**: 
  - `test_mixed_precision.py`: Tests the core mixed precision utilities (policies, loss scaling)
  - `test_mixed_precision_train.py`: Tests that training with mixed precision produces correct results

- **Benchmarks**:
  - `examples/simple_benchmark.py`: Simple benchmark to test functionality
  - `examples/mixed_precision_benchmark.py`: More comprehensive benchmark to measure performance gains
  - `examples/benchmark_mixed_precision.py`: Full benchmarking script with visualization

- **Verified Results**:
  - Tested on MNIST dataset with various precision policies
  - Verified models trained with mixed precision achieve similar accuracy to full precision models
  - Confirmed performance gains on GPU hardware
  - Ensured compatibility with existing training code

## How to Test

The PR includes comprehensive tests for all components of the mixed precision implementation. Follow these steps to verify:

```bash
# Install testing dependencies if needed
pip install pytest

# Run all mixed precision utility tests (verifies core functionality)
python -m pytest xax/tests/test_mixed_precision.py -v

# Run specific test classes from the mixed precision utilities
python -m pytest xax/tests/test_mixed_precision.py::TestPolicies -v  # Test precision policies
python -m pytest xax/tests/test_mixed_precision.py::TestLossScaling -v  # Test loss scaling mechanisms
python -m pytest xax/tests/test_mixed_precision.py::TestMixedPrecisionMixin -v  # Test mixin functionality

# Run mixed precision training integration tests
python -m pytest xax/tests/test_mixed_precision_train.py -v  

# Run specific training test methods
python -m pytest xax/tests/test_mixed_precision_train.py::TestMixedPrecisionTraining::test_precision_policy -v
python -m pytest xax/tests/test_mixed_precision_train.py::TestMixedPrecisionTraining::test_loss_scaling -v
python -m pytest xax/tests/test_mixed_precision_train.py::TestMixedPrecisionTraining::test_training_comparison -v
```

These tests verify:
- Proper creation and application of precision policies 
- Correct behavior of static, dynamic, and no-op loss scaling mechanisms
- Appropriate integration with the existing training loop
- Preservation of model quality and training convergence with mixed precision
- Tree-based dtype transformation utilities
- Hardware-specific optimizations (float16 for GPU, bfloat16 for TPU)

For performance benchmarking, run:
```bash
python xax/examples/mixed_precision_benchmark.py
```

This will compare training speed and memory usage between standard and mixed precision training.

## Usage

To enable mixed precision training, simply update your task configuration:

```python
@dataclass
class MyConfig(TrainConfig):
    enable_mixed_precision: bool = True
    precision_policy: str = "mixed"  # or "float16", "default", etc.
    loss_scaling: str = "dynamic"
    loss_scale_value: float = 2**15
```

Then ensure your task inherits from both `TrainMixin` and `MixedPrecisionMixin`:

```python
class MyTask(TrainMixin[MyConfig], MixedPrecisionMixin[MyConfig]):
    # Standard task implementation
    # Mixed precision handled automatically
```

See the documentation and examples for more detailed usage instructions.

## Future Work

- Add more extensive benchmarks comparing mixed precision vs full precision performance
- Extend support to more hardware platforms
- Optimize for specific model architectures
- Explore automatic mixed precision techniques for dynamic precision selection 