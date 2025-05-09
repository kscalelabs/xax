# Mixed Precision Training Integration

This PR adds comprehensive mixed precision training support to XAX, enabling faster training and reduced memory usage while maintaining model accuracy.

## Features

- **Mixed Precision Mixin**: Adds a reusable `MixedPrecisionMixin` to handle mixed precision training logic
- **Flexible Precision Policies**: Support for custom data type policies for parameters, computation, and outputs
- **Loss Scaling**: Implements static and dynamic loss scaling to prevent gradient underflow
- **Integrated with Training Loop**: Transparently works with existing `TrainMixin`
- **Hardware Awareness**: Automatically selects the optimal half-precision format (float16 for GPU, bfloat16 for TPU)
- **Complete Benchmarking**: Includes benchmark tool with visualization for performance comparison
- **Comprehensive Documentation**: Detailed usage instructions and examples

## Implementation Details

- **Core Utilities** (`xax/utils/mixed_precision.py`):
  - Precision Policy: Controls type conversion for parameters, computation, and outputs
  - Loss Scale: Prevents underflow with NoOp, Static, and Dynamic implementations
  - Tree Utilities: Efficiently maps operations across parameter trees

- **Task Integration** (`xax/task/mixins/mixed_precision.py`):
  - `MixedPrecisionMixin`: Seamlessly integrates with the XAX training pipeline
  - Configurable via `MixedPrecisionConfig` with sensible defaults
  
- **Testing & Benchmarks**:
  - Unit tests for all core components 
  - Integration tests for training correctness
  - Benchmark with visualization for performance analysis

## Testing

All tests are passing, verifying the implementation's correctness:

### Core Functionality Tests
```bash
python -m pytest xax/tests/test_mixed_precision.py -v
```
- **TestPolicies**: Verifies policy creation, type conversion, and casting operations
- **TestLossScaling**: Validates static, dynamic, and no-op loss scaling behavior
- **TestMixedPrecisionMixin**: Confirms configuration handling and integration

### Training Integration Tests
```bash
python -m pytest xax/tests/test_mixed_precision_train.py -v
```
- **TestMixedPrecisionTraining**:
  - `test_precision_policy`: Confirms correct policy application
  - `test_loss_scaling`: Verifies gradient scaling behavior
  - `test_training_comparison`: Ensures similar convergence between standard and mixed precision

### Benchmark
```bash
python xax/examples/mixed_precision_benchmark.py
```
The benchmark demonstrates the implementation in action with:
- Comparative analysis of speed between full and mixed precision
- Memory usage tracking
- Loss convergence visualization
- Final model quality comparison

Results show expected behavior with almost identical final loss values, confirming training stability with mixed precision.

## How to Use

To enable mixed precision training, simply update your task configuration:

```python
@dataclass
class MyConfig(TrainConfig, MixedPrecisionConfig):
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

The implementation offers several configuration options:

1. **Precision Policies**:
   - `"default"`: float32 for all operations
   - `"mixed"`: float32 for parameters, half precision for computation, float32 for output
   - `"float16"`: float16 for all operations
   - Custom: Specify with format `"params=float32,compute=float16,output=float32"`

2. **Loss Scaling Strategies**:
   - `"none"`: No loss scaling (use with caution)
   - `"static"`: Fixed loss scale value
   - `"dynamic"`: Automatically adjusts based on gradient overflow detection

## Future Work

- Extend benchmark to support larger models and datasets
- Add profiling for more detailed performance analysis
- Implement automatic mixed precision with dynamic type selection
- Optimize for specific model architectures 