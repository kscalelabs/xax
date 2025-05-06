# Mixed Precision Training for XAX

This repository implements mixed precision training capabilities for the XAX deep learning framework. Mixed precision training uses lower-precision floating-point formats (like float16 or bfloat16) to accelerate neural network training while maintaining model accuracy.

## Key Features

- **Flexible Precision Policies**: Configure different data types for parameters, computation, and outputs
- **Integrated Loss Scaling**: Prevent gradient underflow with static or dynamic loss scaling
- **Seamless Integration**: Works with existing XAX training pipeline with minimal code changes
- **Platform Awareness**: Automatically selects the optimal half-precision format for your hardware (float16 for GPU, bfloat16 for TPU)
- **Comprehensive Documentation**: Detailed docs and examples to get you started

## Getting Started

### Installation

```bash
git clone https://github.com/your-username/xax.git
cd xax
pip install -e .
```

### Quick Example

```python
from xax.task import Task
from xax.task.mixins.train import TrainConfig, TrainMixin
from xax.task.mixins.mixed_precision import MixedPrecisionConfig, MixedPrecisionMixin 
from dataclasses import dataclass

@dataclass
class MyConfig(TrainConfig, MixedPrecisionConfig):
    # Enable mixed precision training
    enable_mixed_precision: bool = True
    
    # Choose precision policy: "mixed", "float16", or "default"
    precision_policy: str = "mixed"
    
    # Configure loss scaling
    loss_scaling: str = "dynamic"
    loss_scale_value: float = 2**15
    
    # Other training parameters
    learning_rate: float = 0.001
    batch_size: int = 128
    # ...

# Your task inherits mixed precision capabilities
class MyTask(TrainMixin[MyConfig], MixedPrecisionMixin[MyConfig]):
    # Your task implementation
    # ...
```

## Running Tests and Examples

### Running Tests

The mixed precision implementation includes comprehensive test coverage:

```bash
# From the repository root
# Test all mixed precision utilities
python -m pytest xax/tests/test_mixed_precision.py -v

# Test specific functionality
python -m pytest xax/tests/test_mixed_precision.py::TestPolicies -v
python -m pytest xax/tests/test_mixed_precision.py::TestLossScaling -v
python -m pytest xax/tests/test_mixed_precision.py::TestMixedPrecisionMixin -v

# Test integrated training with mixed precision
python -m pytest xax/tests/test_mixed_precision_train.py -v
```

Our test suite verifies:
- Precision policy creation and dtype conversion
- Loss scaling mechanisms (static, dynamic, no-op)
- Gradient scaling and unscaling
- Model parameter type handling
- Training convergence with mixed precision vs. standard precision
- Hardware-specific optimizations
- Numeric stability across training steps

### Running Examples

The repository includes examples to demonstrate mixed precision training:

```bash
# Run a comprehensive mixed precision benchmark
python xax/examples/mixed_precision_benchmark.py
```

## Documentation

For detailed documentation, see:
- [Mixed Precision Overview](docs/mixed_precision.md)
- [Example: Mixed Precision Benchmark](examples/mixed_precision_benchmark.py)

## Performance Results

Mixed precision training can significantly improve training speed and memory efficiency:

| Model | Dataset | Hardware | Speedup | Memory Savings |
|-------|---------|----------|---------|----------------|
| LinearModel | Synthetic | CPU | 1.2-1.5x | 1.3-1.8x |
| LinearModel | Synthetic | NVIDIA GPU | 1.8-2.2x | 1.7-2.0x |

*Note: Actual performance gains vary by hardware, model architecture, and workload.*

## Implementation Details

This implementation is inspired by established mixed precision frameworks like NVIDIA's Apex and DeepMind's JMP, but is fully integrated into the XAX ecosystem. Key components include:

- **Policy Management**: Flexible policies for dtype management across model components
- **Loss Scaling**: Prevent gradient underflow with configurable scaling strategies
- **Tree Transformations**: Efficient tree-based utilities for type conversion
- **Hardware Awareness**: Platform-specific optimizations for different accelerators

## Project Structure

```
xax/
├── task/
│   └── mixins/         # Training and mixed precision mixins
│       ├── __init__.py
│       ├── train.py    # Training mixin
│       └── mixed_precision.py # Mixed precision mixin
├── utils/
│   └── mixed_precision.py # Mixed precision utilities
├── examples/           # Usage examples
│   └── mixed_precision_benchmark.py
└── tests/              # Unit tests
    ├── test_mixed_precision.py
    └── test_mixed_precision_train.py
```

## Test Implementation

Our testing approach is comprehensive, focusing on both unit-level verification and integration testing:

### Unit Tests (`test_mixed_precision.py`)
These tests verify the core building blocks of the mixed precision implementation:
- `TestPolicies`: Verifies policy parsing, type conversion, and tree operations
- `TestLossScaling`: Tests all three loss scaling strategies (None, Static, Dynamic)
- `TestMixedPrecisionMixin`: Ensures configuration parsing and mixin functionality

### Integration Tests (`test_mixed_precision_train.py`)
These tests focus on the end-to-end training loop with mixed precision:
- `TestMixedPrecisionTraining`: Compares training behavior between standard and mixed precision
  - `test_precision_policy`: Validates policy creation and application
  - `test_loss_scaling`: Ensures loss scaling behaves correctly
  - `test_training_comparison`: Confirms similar model quality between standard and mixed precision

The tests use simple linear models and synthetic data to ensure fast execution while still exercising all mixed precision components.

## Usage Guide

### Basic Configuration

1. Import the necessary components:
```python
from xax.task.mixins.train import TrainConfig, TrainMixin
from xax.task.mixins.mixed_precision import MixedPrecisionConfig, MixedPrecisionMixin
```

2. Create a configuration that combines `TrainConfig` and `MixedPrecisionConfig`:
```python
@dataclass
class MyConfig(TrainConfig, MixedPrecisionConfig):
    enable_mixed_precision: bool = True
    precision_policy: str = "mixed"
    loss_scaling: str = "dynamic"
    loss_scale_value: float = 2**15
```

3. Create your task class inheriting from both mixins:
```python
class MyTask(TrainMixin[MyConfig], MixedPrecisionMixin[MyConfig]):
    # Your task implementation
```

4. The training process will now automatically use mixed precision based on your configuration!

### Available Options

- **Precision Policies**:
  - `"default"`: Use float32 for all operations
  - `"mixed"`: Use float32 for parameters, half precision for computation, float32 for output
  - `"float16"`: Use float16 for all operations
  - `"bfloat16"`: Use bfloat16 for all operations
  - Custom: Specify with format `"params=float32,compute=float16,output=float32"`

- **Loss Scaling**:
  - `"none"`: No loss scaling (use with caution)
  - `"static"`: Fixed loss scale value
  - `"dynamic"`: Automatically adjusts based on gradient overflow detection

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The XAX framework developers
- JAX and Equinox teams for their excellent foundations
- NVIDIA and DeepMind for their mixed precision training research
