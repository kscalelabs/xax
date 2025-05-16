# Mixed Precision Training in XAX

This document explains how to use the mixed precision training feature in XAX. Mixed precision training is a technique that uses lower precision data types (like float16 or bfloat16) during computation to speed up training, while maintaining model weights in higher precision (like float32) to preserve accuracy.

## Running the Mixed Precision Examples

The mixed precision examples demonstrate key concepts and provide benchmarks:

```bash
# Run exploration of mixed precision concepts
python3 xax/examples/mixed_precision.py explore

# Run benchmark comparing full vs. mixed precision
python3 xax/examples/mixed_precision.py benchmark 

# Run both examples
python3 xax/examples/mixed_precision.py all

# Show help
python3 xax/examples/mixed_precision.py --help
```

The examples demonstrate:
1. Different precision policies (default, mixed, float16, bfloat16)
2. Loss scaling techniques (static, dynamic)
3. Performance comparison between full and mixed precision training

## Benefits of Mixed Precision Training

- **Faster training**: Lower precision computations are faster, especially on GPUs with hardware support for these operations.
- **Less memory usage**: Lower precision data types require less memory, allowing for larger batch sizes or models.
- **More efficient hardware utilization**: Modern GPUs have specialized hardware for half-precision operations that can be 2-8x faster than full precision.

## Getting Started

### 1. Enable mixed precision in your configuration

Add the following to your task configuration:

```python
@dataclass
class MyTaskConfig(TrainConfig):
    # Enable mixed precision
    enable_mixed_precision: bool = True
    
    # Mixed precision policy options: "mixed", "float16", "default"
    # or custom like "params=float32,compute=float16,output=float32"
    precision_policy: str = "mixed"
    
    # Loss scaling approach: "none", "static", or "dynamic"
    loss_scaling: str = "dynamic"
    
    # Initial scale for loss scaling
    loss_scale_value: float = 2**15
```

### 2. Use MixedPrecisionMixin in your task

If you're creating a custom task, make sure it inherits from `MixedPrecisionMixin`:

```python
class MyTask(TrainMixin[MyTaskConfig], MixedPrecisionMixin[MyTaskConfig]):
    # Your task implementation
    ...
```

If you're using the standard `TrainMixin`, it already includes `MixedPrecisionMixin`, so you don't need to add it explicitly.

## Mixed Precision Training Process

The mixed precision training process follows these steps:

1. **Store parameters in high precision** (typically float32)
2. **Cast to low precision for computation** (float16/bfloat16) 
3. **Scale the loss** to prevent gradient underflow
4. **Compute gradients** in low precision
5. **Unscale gradients** back to their original magnitude
6. **Check for non-finite gradients** (NaN/Inf)
7. **Update parameters** using the optimizer
8. **Cast back to high precision** for storage

This process is demonstrated in detail in the `mixed_precision.py` example.

## Precision Policies

A precision policy defines which data types to use for different parts of your model and computation:

- **param_dtype**: The data type used for storing model parameters (typically float32)
- **compute_dtype**: The data type used during computation (typically float16 or bfloat16)
- **output_dtype**: The data type for returning results (varies)

XAX provides several predefined policies:

- **"default"**: Everything in float32 (no mixed precision)
- **"mixed"**: Parameters in float32, computation in half precision (float16 on GPU, bfloat16 on TPU)
- **"float16"** or **"half"**: Everything in half precision

You can also define a custom policy using a string:

```python
# Custom policy with specific dtypes for each part
precision_policy = "params=float32,compute=float16,output=float32"
```

## Loss Scaling

When training with reduced precision (especially float16), gradients can underflow to zero. Loss scaling helps prevent this by scaling the loss before backpropagation and then unscaling the gradients.

XAX provides three loss scaling strategies:

### No Loss Scaling

```python
loss_scaling = "none"
```

No scaling is applied. This is fine for models that don't have gradient underflow issues or when using bfloat16.

### Static Loss Scaling

```python
loss_scaling = "static"
loss_scale_value = 2**15  # Fixed scale value (32768)
```

Applies a fixed scale factor to the loss. This is simple but requires manual tuning.

### Dynamic Loss Scaling

```python
loss_scaling = "dynamic"
loss_scale_value = 2**15  # Initial scale value
loss_scale_growth_interval = 2000  # Steps before increasing scale
loss_scale_growth_factor = 2.0  # Factor to increase scale
loss_scale_backoff_factor = 0.5  # Factor to decrease scale
```

Automatically adjusts the scale during training:
- Increases by `growth_factor` after `growth_interval` consecutive steps with finite gradients
- Decreases by `backoff_factor` when non-finite gradients are encountered

## Handling Non-Finite Gradients

When using mixed precision, you may occasionally encounter NaN or Inf gradients. XAX can automatically handle these:

```python
skip_nonfinite_updates = True  # Skip updates with non-finite gradients
```

When enabled, updates with non-finite gradients are skipped, and the loss scale is adjusted if using dynamic scaling.

## Advanced Usage

### Manual Control in Custom Training Loops

If you're writing a custom training loop, you can use these methods from the `MixedPrecisionMixin`:

```python
# Cast parameters to computation precision
model = self.cast_params_to_compute(model)

# Scale loss before computing gradients
loss = self.scale_loss(loss)

# Unscale gradients after gradient computation
grads = self.unscale_grads(grads)

# Check if gradients are finite
grads_finite = self.check_grads_finite(grads)

# Apply update with mixed precision handling
model, opt_state, new_loss_scale = self.mixed_precision_update(
    model, grads, optimizer_update, opt_state
)

# Update loss scale for next iteration
self.set_loss_scale(new_loss_scale)

# Cast parameters back to storage precision
model = self.cast_params_to_storage(model)
```

### Direct Use of Mixed Precision Utilities

You can directly use the utilities in `xax.utils.mixed_precision`:

```python
from xax.utils.mixed_precision import Policy, StaticLossScale, tree_map_dtype

# Create a custom policy
policy = Policy(
    param_dtype=jnp.float32,
    compute_dtype=jnp.float16,
    output_dtype=jnp.float32
)

# Cast parameters to computation precision
params_compute = policy.cast_to_compute(params)

# Create a loss scale
loss_scale = StaticLossScale(2**15)

# Scale loss
scaled_loss = loss_scale.scale(loss)

# Unscale gradients
unscaled_grads = loss_scale.unscale(grads)
```

## Example

See the `xax/examples/mixed_precision.py` file for a complete example of mixed precision training. It includes:

1. **Exploration of mixed precision concepts**
2. **Benchmarking** full vs. mixed precision performance
3. **Detailed documentation** of the mixed precision training process

## Platform-Specific Considerations

### GPU

On NVIDIA GPUs, float16 is the preferred half-precision format for mixed precision training. It offers significant speedups, especially on newer GPUs with Tensor Cores (Volta, Turing, Ampere, etc.).

### TPU

On TPUs, bfloat16 is the preferred half-precision format. XAX automatically selects bfloat16 when running on TPUs. The wider exponent range of bfloat16 makes loss scaling less critical, but it's still recommended for numerical stability.

## Performance Tips

1. **Batch size**: Try increasing your batch size when using mixed precision since it uses less memory.
2. **Loss scaling**: Start with dynamic loss scaling for best results.
3. **Custom operations**: If implementing custom operations, ensure they handle reduced precision properly.
4. **Check for underflow/overflow**: Monitor training closely for any signs of numerical instability.

## Implementation Details

XAX's mixed precision training is inspired by the [JMP library](https://github.com/google-deepmind/jmp) from DeepMind but is integrated directly into the XAX framework for a seamless experience. 