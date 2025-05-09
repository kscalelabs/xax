# Mixed Precision Training for XAX

This repository implements mixed precision training capabilities for the XAX deep learning framework. Mixed precision training uses lower-precision floating-point formats (like float16 or bfloat16) to accelerate neural network training while maintaining model accuracy.

## Key Features

- **Flexible Precision Policies**: Configure different data types for parameters, computation, and outputs
- **Integrated Loss Scaling**: Prevent gradient underflow with static or dynamic loss scaling
- **Seamless Integration**: Works with existing XAX training pipeline with minimal code changes
- **Platform Awareness**: Automatically selects the optimal half-precision format for your hardware (float16 for GPU, bfloat16 for TPU)
- **Visualization & Benchmarking**: Includes tools to measure and visualize performance benefits

## Getting Started

### Installation

```bash
git clone https://github.com/xerk-dot/xax.git
cd xax
pip install -e .
```

### Quick Example

```python
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

## Implementation Details

### Core Components

The implementation consists of three main components:

1. **Precision Policies** (`Policy` class):
   - Controls data types for parameters, computation, and outputs
   - Provides efficient type conversion utilities
   - Supports predefined and custom policies

2. **Loss Scaling** (`LossScale` classes):
   - `NoOpLossScale`: No scaling (for full precision training)
   - `StaticLossScale`: Fixed loss scaling value
   - `DynamicLossScale`: Automatically adjusts based on gradient overflow

3. **MixedPrecisionMixin**:
   - Integrates with existing training loop
   - Handles parameter casting, loss scaling, and gradient unscaling
   - Manages gradient finiteness checking and recovery

### Project Structure

```
xax/
├── task/
│   └── mixins/
│       ├── mixed_precision.py # Mixed precision mixin
│       └── train.py           # Training mixin
├── utils/
│   └── mixed_precision.py     # Core utilities
├── examples/
│   └── mixed_precision_benchmark.py  # Benchmark tool
└── tests/
    ├── test_mixed_precision.py       # Utility tests
    └── test_mixed_precision_train.py # Training tests
```

## Running Tests and Benchmarks

### Testing

All tests are passing, verifying the mixed precision implementation:

```bash
# Test the core utilities
python -m pytest xax/tests/test_mixed_precision.py -v

# Test training integration
python -m pytest xax/tests/test_mixed_precision_train.py -v
```

Our test suite verifies:
- Precision policy creation and dtype conversion
- Loss scaling mechanisms (static, dynamic, no-op)
- Gradient scaling and unscaling
- Model parameter type handling
- Training convergence with mixed precision vs. standard precision
- Hardware-specific optimizations

### Benchmarking

The repository includes a comprehensive benchmark tool:

```bash
# Run the mixed precision benchmark
python xax/examples/mixed_precision_benchmark.py
```

The benchmark:
- Compares training speed between full and mixed precision
- Measures memory usage differences
- Tracks training loss curves
- Generates visualizations of results
- Validates model correctness

## Performance Results

The benchmark shows the implementation is working correctly, with nearly identical final loss values between full and mixed precision training. On appropriate hardware with specialized tensor cores (like NVIDIA GPUs or Google TPUs), mixed precision training typically provides significant performance benefits:

| Test Environment | Results |
|------------------|---------|
| CPU (tested) | Mixed precision achieves similar training quality to full precision |
| CPU (tested) | Both approaches converge to similar final loss values |
| CPU (tested) | Mixed precision maintains numerical stability with dynamic loss scaling |

**Note:** While our tests on CPU show similar performance between full and mixed precision (which is expected), the true performance benefits would be observed on GPU/TPU hardware with dedicated half-precision acceleration. In those environments, literature suggests 1.5-3x speed improvements and memory savings of 1.5-2x are typical.

## Usage Guide

### Precision Policies

You can choose from predefined policies or create custom ones:

```python
# Use predefined policies
config.precision_policy = "default"  # All float32
config.precision_policy = "mixed"    # float32 params, half-precision compute, float32 output
config.precision_policy = "float16"  # All float16 (or bfloat16 on TPU)

# Or create a custom policy (params=float32, compute=bfloat16, output=float32)
config.precision_policy = "params=float32,compute=bfloat16,output=float32"
```

### Loss Scaling

Configure loss scaling to prevent gradient underflow:

```python
# No loss scaling (use with caution in mixed precision)
config.loss_scaling = "none"

# Static loss scaling with fixed value
config.loss_scaling = "static"
config.loss_scale_value = 128.0

# Dynamic loss scaling (recommended)
config.loss_scaling = "dynamic"
config.loss_scale_value = 2**15        # Initial scale
config.loss_scale_growth_interval = 2000  # Steps between scale increases
config.loss_scale_growth_factor = 2.0     # Scale multiplier on increase
config.loss_scale_backoff_factor = 0.5    # Scale multiplier on overflow
```

## Implementation Notes

- The implementation ensures backward compatibility with existing code
- Hardware detection automatically selects the appropriate half-precision format
- Tree-based utilities efficiently handle nested parameter structures
- Gradient finiteness checking prevents training instability
- The solution draws inspiration from established approaches like NVIDIA Apex and TensorFlow mixed precision

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The XAX framework developers
- JAX and Equinox teams for their excellent foundations
- NVIDIA and DeepMind for their mixed precision training research
