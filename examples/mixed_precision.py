"""Mixed precision examples demonstrating JAX mixed precision training.

This file provides examples of mixed precision training with JAX:
1. Simple benchmark comparing training speed and memory usage
2. Demonstration of mixed precision policies
3. Exploration of loss scaling behavior

Run the example with:
    # Run exploration of mixed precision concepts
    python3 xax/examples/mixed_precision.py explore
    
    # Run benchmark comparison between full and mixed precision
    python3 xax/examples/mixed_precision.py benchmark
    
    # Run both
    python3 xax/examples/mixed_precision.py all
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import time
import matplotlib.pyplot as plt
import argparse
import sys
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List, Union, NamedTuple

from xax.utils.mixed_precision import (
    Policy, get_policy, NoOpLossScale, StaticLossScale, DynamicLossScale, 
    get_default_half_dtype, all_finite
)

#------------------------------------------------------------------------------
# MODELS
#------------------------------------------------------------------------------

class MLP(eqx.Module):
    """A simple MLP model with one hidden layer for benchmarking."""
    layer1_weight: jnp.ndarray
    layer1_bias: jnp.ndarray
    layer2_weight: jnp.ndarray
    layer2_bias: jnp.ndarray
    
    def __init__(self, in_dim, hidden_dim, out_dim, key):
        key1, key2 = jax.random.split(key)
        w1_key, b1_key = jax.random.split(key1)
        w2_key, b2_key = jax.random.split(key2)
        
        # Initialize with standard distribution / sqrt(fan_in)
        self.layer1_weight = jax.random.normal(w1_key, (in_dim, hidden_dim)) / jnp.sqrt(in_dim)
        self.layer1_bias = jax.random.normal(b1_key, (hidden_dim,))
        self.layer2_weight = jax.random.normal(w2_key, (hidden_dim, out_dim)) / jnp.sqrt(hidden_dim)
        self.layer2_bias = jax.random.normal(b2_key, (out_dim,))
    
    def __call__(self, x):
        # First layer with ReLU activation
        h = jnp.dot(x, self.layer1_weight) + self.layer1_bias
        h = jnp.maximum(0, h)  # ReLU
        # Output layer
        return jnp.dot(h, self.layer2_weight) + self.layer2_bias


# Custom LossScale implementation that works with JAX tracing
class CustomLossScaleState(NamedTuple):
    """State for tracking dynamic loss scale."""
    loss_scale: jnp.ndarray
    steps_since_finite: jnp.ndarray
    growth_interval: jnp.ndarray
    growth_factor: jnp.ndarray
    backoff_factor: jnp.ndarray
    max_scale: jnp.ndarray


@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision training."""
    enable_mixed_precision: bool = False
    precision_policy: str = "mixed"  # "default", "mixed", "float16", "bfloat16"
    loss_scaling: str = "dynamic"  # "none", "static", "dynamic"
    loss_scale_value: float = 2**15
    loss_scale_growth_interval: int = 2000
    loss_scale_growth_factor: float = 2.0
    loss_scale_backoff_factor: float = 0.5
    loss_scale_max_value: Optional[float] = None
    skip_nonfinite_updates: bool = True  # Skip updates that would produce non-finite gradients


@dataclass
class BenchmarkConfig(MixedPrecisionConfig):
    """Configuration for benchmark tasks."""
    input_dim: int = 512
    hidden_dim: int = 512
    output_dim: int = 10
    batch_size: int = 128
    num_iterations: int = 100
    warmup_steps: int = 3
    seed: int = 42


class MixedPrecisionTrainer:
    """A simple trainer that supports mixed precision.
    
    This class demonstrates the core components of mixed precision training:
    1. Precision policies for different parts of the model
    2. Loss scaling to prevent gradient underflow
    3. Gradient unscaling and finiteness checks
    4. Dynamic loss scale adjustment
    
    A typical mixed precision training flow involves:
    - Store model parameters in high precision (float32)
    - Cast to lower precision (float16/bfloat16) for computation
    - Scale the loss to prevent gradient underflow
    - Unscale gradients before optimizer update
    - Check for non-finite gradients and adjust loss scale
    - Update parameters and cast back to high precision for storage
    """
    
    def __init__(self, config: BenchmarkConfig):
        """Initialize the trainer with the given configuration.
        
        Args:
            config: Configuration containing mixed precision settings
        """
        self.config = config
        key = jax.random.PRNGKey(config.seed)
        
        # Create model and optimizer
        model_key, data_key = jax.random.split(key)
        self.model = MLP(
            config.input_dim, 
            config.hidden_dim, 
            config.output_dim, 
            model_key
        )
        self.optimizer = optax.adam(1e-3)
        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))
        
        # Generate synthetic data
        self._generate_data(data_key)
        
        # Set up mixed precision utilities
        self.policy = get_policy(config.precision_policy)
        
        # Set up loss scaling state
        if not config.enable_mixed_precision or config.loss_scaling == "none":
            # No loss scaling needed for full precision or when explicitly disabled
            self.loss_scale_value = jnp.array(1.0, dtype=jnp.float32)
            self.use_dynamic_scaling = False
            self.dynamic_scale_state = None
        elif config.loss_scaling == "static":
            # Static loss scaling uses a fixed scale factor
            self.loss_scale_value = jnp.array(config.loss_scale_value, dtype=jnp.float32)
            self.use_dynamic_scaling = False
            self.dynamic_scale_state = None
        elif config.loss_scaling == "dynamic":
            # Dynamic loss scaling automatically adjusts based on gradient behavior
            self.use_dynamic_scaling = True
            self.dynamic_scale_state = CustomLossScaleState(
                loss_scale=jnp.array(config.loss_scale_value, dtype=jnp.float32),
                steps_since_finite=jnp.array(0, dtype=jnp.int32),
                growth_interval=jnp.array(config.loss_scale_growth_interval, dtype=jnp.int32),
                growth_factor=jnp.array(config.loss_scale_growth_factor, dtype=jnp.float32),
                backoff_factor=jnp.array(config.loss_scale_backoff_factor, dtype=jnp.float32),
                max_scale=jnp.array(config.loss_scale_max_value 
                                   if config.loss_scale_max_value is not None 
                                   else float('inf'), dtype=jnp.float32)
            )
        else:
            raise ValueError(f"Unknown loss scaling approach: {config.loss_scaling}")
    
    def _generate_data(self, key):
        """Generate synthetic data for training."""
        x_key, y_key = jax.random.split(key)
        self.images = jax.random.normal(x_key, (self.config.batch_size, self.config.input_dim))
        y_indices = jax.random.randint(y_key, (self.config.batch_size,), 0, self.config.output_dim)
        self.labels = jax.nn.one_hot(y_indices, self.config.output_dim)
    
    def get_batch(self):
        """Return a batch of data."""
        return (self.images, self.labels)
    
    def scale_loss(self, loss):
        """Scale the loss value."""
        if self.use_dynamic_scaling:
            return loss * self.dynamic_scale_state.loss_scale
        else:
            return loss * self.loss_scale_value
    
    def unscale_grads(self, grads):
        """Unscale the gradients."""
        if self.use_dynamic_scaling:
            scale = self.dynamic_scale_state.loss_scale
        else:
            scale = self.loss_scale_value
        
        return jax.tree_util.tree_map(lambda g: g / scale, grads)
    
    def adjust_loss_scale(self, grads_finite, state):
        """Adjust loss scale using JAX's functional style."""
        if not self.use_dynamic_scaling:
            return state
        
        # Helper functions for dynamic scaling
        def increase_scale():
            new_scale = jnp.minimum(
                state.loss_scale * state.growth_factor,
                state.max_scale
            )
            return CustomLossScaleState(
                loss_scale=new_scale,
                steps_since_finite=jnp.array(0, dtype=jnp.int32),
                growth_interval=state.growth_interval,
                growth_factor=state.growth_factor,
                backoff_factor=state.backoff_factor,
                max_scale=state.max_scale
            )
        
        def continue_counter():
            return CustomLossScaleState(
                loss_scale=state.loss_scale,
                steps_since_finite=state.steps_since_finite + 1,
                growth_interval=state.growth_interval,
                growth_factor=state.growth_factor,
                backoff_factor=state.backoff_factor,
                max_scale=state.max_scale
            )
        
        def decrease_scale():
            return CustomLossScaleState(
                loss_scale=state.loss_scale * state.backoff_factor,
                steps_since_finite=jnp.array(0, dtype=jnp.int32),
                growth_interval=state.growth_interval,
                growth_factor=state.growth_factor,
                backoff_factor=state.backoff_factor,
                max_scale=state.max_scale
            )
        
        # Use JAX's functional control flow
        should_increase = (state.steps_since_finite >= state.growth_interval)
        
        # First branch: if gradients are finite
        # Second branch: if gradients are not finite
        return jax.lax.cond(
            grads_finite,
            # If grads are finite, either increase scale or continue counter
            lambda _: jax.lax.cond(
                should_increase,
                lambda _: increase_scale(),
                lambda _: continue_counter(),
                None
            ),
            # If grads are not finite, decrease scale
            lambda _: decrease_scale(),
            None
        )
    
    def train_step(self, model, batch):
        """Execute one training step with mixed precision support.
        
        This method demonstrates the complete mixed precision training process:
        1. Cast model parameters to compute precision
        2. Forward pass in reduced precision
        3. Scale loss to prevent gradient underflow
        4. Compute gradients
        5. Unscale gradients back to original range
        6. Check for gradient finiteness
        7. Skip updates with non-finite gradients if configured
        8. Apply optimizer update
        9. Cast parameters back to storage precision
        
        Args:
            model: The model to train
            batch: Tuple of (inputs, targets)
            
        Returns:
            Tuple of (loss, updated_model)
        """
        x, y = batch
        
        # Define loss function
        def loss_fn(model):
            # Step 1: Cast model to compute precision if using mixed precision
            # This converts parameters from high precision (e.g., float32) to lower
            # precision (e.g., float16/bfloat16) for faster computation
            if self.config.enable_mixed_precision:
                model = self.policy.cast_to_compute(model)
            
            # Step 2: Forward pass
            # This runs in reduced precision when mixed precision is enabled
            logits = jax.vmap(model)(x)
            loss = optax.softmax_cross_entropy(logits, y).mean()
            
            # Step 3: Scale loss if using mixed precision
            # This prevents gradient underflow in half precision
            if self.config.enable_mixed_precision:
                loss = self.scale_loss(loss)
            
            return loss
        
        # Step 4: Compute gradients
        # JAX automatically handles backpropagation through the precision casting
        loss, grads = jax.value_and_grad(loss_fn)(model)
        
        # Define update functions
        def apply_update(model, grads, opt_state):
            """Apply optimizer update."""
            updates, new_opt_state = self.optimizer.update(grads, opt_state, model)
            new_model = eqx.apply_updates(model, updates)
            return new_model, new_opt_state, loss
        
        def skip_update(model, grads, opt_state):
            """Skip optimizer update."""
            return model, opt_state, loss
        
        # Handle mixed precision
        if self.config.enable_mixed_precision:
            # Step 5: Unscale gradients
            # This returns gradients to their original magnitude
            grads = self.unscale_grads(grads)
            
            # Step 6: Check gradient finiteness
            # Detect NaN or Inf values that can occur with numerical instability
            grads_finite = all_finite(grads)
            
            # Step 7: Update loss scale state for next iteration
            # This adjusts the loss scale based on gradient behavior
            new_scale_state = self.adjust_loss_scale(grads_finite, self.dynamic_scale_state)
            
            # Step 8: Apply updates conditionally
            # Skip updates with non-finite gradients if configured
            update_condition = jnp.logical_or(
                grads_finite, 
                jnp.logical_not(self.config.skip_nonfinite_updates)
            )
            model, self.opt_state, loss = jax.lax.cond(
                update_condition,
                lambda args: apply_update(*args),
                lambda args: skip_update(*args),
                (model, grads, self.opt_state)
            )
            
            # Step 9: Cast back to storage precision
            # This ensures parameters are stored in high precision
            model = self.policy.cast_to_param(model)
            
            # Update loss scale state (outside JIT region)
            self.dynamic_scale_state = new_scale_state
        else:
            # Standard update for full precision
            model, self.opt_state, loss = apply_update(model, grads, self.opt_state)
        
        return loss, model
    
    def run_benchmark(self) -> Dict[str, float]:
        """Run benchmark and return timing results."""
        results = {}
        
        # JIT compile the training step
        jit_train_step = jax.jit(self.train_step)
        
        # Warmup JIT compilation
        for _ in range(self.config.warmup_steps):
            _, self.model = jit_train_step(self.model, self.get_batch())
        
        # Time training steps
        start_time = time.time()
        for _ in range(self.config.num_iterations):
            _, self.model = jit_train_step(self.model, self.get_batch())
        end_time = time.time()
        
        # Record metrics
        total_time = end_time - start_time
        results["total_time"] = total_time
        results["time_per_step"] = total_time / self.config.num_iterations
        results["steps_per_second"] = self.config.num_iterations / total_time
        
        # Measure memory usage
        try:
            mem_usage = jax.live_arrays()
            total_bytes = sum(x.nbytes for x in mem_usage)
            results["memory_usage_bytes"] = total_bytes
            results["memory_usage_mb"] = total_bytes / (1024 * 1024)
        except:
            # May not be available in all JAX versions
            results["memory_usage_bytes"] = None
            results["memory_usage_mb"] = None
        
        return results


def run_benchmark_comparison():
    """Run comparison between full precision and mixed precision.
    
    This function performs a benchmark comparing:
    1. Standard float32 precision training
    2. Mixed precision training (float32 parameters, float16/bfloat16 computation)
    
    It measures training speed and memory usage differences between the two approaches.
    
    Returns:
        Dict with benchmark results for both precision modes
    """
    results = {}
    
    try:
        print("Running full precision benchmark...")
        fp_config = BenchmarkConfig(
            enable_mixed_precision=False,
            precision_policy="default",
            seed=42
        )
        fp_task = MixedPrecisionTrainer(fp_config)
        results["full_precision"] = fp_task.run_benchmark()
        print(f"Full precision: {results['full_precision']['time_per_step']:.4f} seconds/step")
        
        print("\nRunning mixed precision benchmark...")
        # Use appropriate half precision for the platform
        half_dtype = get_default_half_dtype()
        mp_config = BenchmarkConfig(
            enable_mixed_precision=True,
            precision_policy="mixed",
            loss_scaling="dynamic",
            loss_scale_value=2**15,
            seed=42
        )
        mp_task = MixedPrecisionTrainer(mp_config)
        results["mixed_precision"] = mp_task.run_benchmark()
        print(f"Mixed precision: {results['mixed_precision']['time_per_step']:.4f} seconds/step")
        
        # Calculate and print speedup
        speedup = results["full_precision"]["time_per_step"] / results["mixed_precision"]["time_per_step"]
        print(f"\nSpeedup: {speedup:.2f}x")
        
        # Calculate and print memory savings if available
        if (results["full_precision"]["memory_usage_mb"] is not None and 
            results["mixed_precision"]["memory_usage_mb"] is not None):
            memory_savings = (results["full_precision"]["memory_usage_mb"] / 
                            results["mixed_precision"]["memory_usage_mb"])
            print(f"Memory savings: {memory_savings:.2f}x")
        
        # Plot the results
        plot_benchmark_results(results)
        
    except Exception as e:
        print(f"\nError during benchmark: {e}")
        print("This might be due to device constraints or limitations in the current JAX environment.")
        print("Try running on a platform with better mixed precision support, like a GPU or TPU.")
    
    return results


def plot_benchmark_results(results):
    """Plot comparison results between full and mixed precision.
    
    This function creates and displays plots comparing:
    1. Training speed (time per step)
    2. Memory usage
    
    Args:
        results: Dictionary containing benchmark results for different precision modes
    """
    if "full_precision" not in results or "mixed_precision" not in results:
        print("Missing data for comparison plot")
        return
    
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot time per step
        times = [
            results["full_precision"]["time_per_step"],
            results["mixed_precision"]["time_per_step"]
        ]
        ax1.bar(["Full Precision", "Mixed Precision"], times, color=["blue", "orange"])
        ax1.set_ylabel("Time per step (seconds)")
        ax1.set_title("Training Speed Comparison")
        
        # Format y-axis to show smaller numbers more clearly
        ax1.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
        
        # Add speedup text
        speedup = results["full_precision"]["time_per_step"] / results["mixed_precision"]["time_per_step"]
        ax1.text(1, times[1], f"{speedup:.2f}x faster", ha="center", va="bottom")
        
        # Plot memory usage if available
        if (results["full_precision"]["memory_usage_mb"] is not None and 
            results["mixed_precision"]["memory_usage_mb"] is not None):
            memory = [
                results["full_precision"]["memory_usage_mb"],
                results["mixed_precision"]["memory_usage_mb"]
            ]
            ax2.bar(["Full Precision", "Mixed Precision"], memory, color=["blue", "orange"])
            ax2.set_ylabel("Memory Usage (MB)")
            ax2.set_title("Memory Usage Comparison")
            
            # Add memory savings text
            memory_savings = results["full_precision"]["memory_usage_mb"] / results["mixed_precision"]["memory_usage_mb"]
            ax2.text(1, memory[1], f"{memory_savings:.2f}x less memory", ha="center", va="bottom")
        else:
            ax2.text(0.5, 0.5, "Memory usage data not available", ha="center", va="center", transform=ax2.transAxes)
        
        # Add a title for the whole figure
        fig.suptitle(f"Mixed Precision vs Full Precision Training on {jax.devices()[0].platform.upper()}", 
                    fontsize=14)
        
        # Add a note about platform considerations
        platform_note = (
            "Note: Performance benefits of mixed precision are typically\n"
            "much more significant on GPUs with Tensor Cores or TPUs"
        )
        fig.text(0.5, 0.01, platform_note, ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig("mixed_precision_benchmark_results.png")
        print(f"Saved plot to mixed_precision_benchmark_results.png")
        
        # Display the plot if in an interactive environment
        try:
            plt.show()
        except:
            print("Plot display not available in this environment")
            
    except Exception as e:
        print(f"Error creating plot: {e}")
        print("This might be due to issues with the matplotlib environment")


def explore_mixed_precision():
    """Explore mixed precision behavior with different policies.
    
    This function demonstrates the key components of mixed precision training:
    1. Different precision policies (default, mixed, float16, bfloat16)
    2. Loss scaling techniques (static, dynamic)
    3. Dynamic loss scale adjustment behavior
    """
    print("JAX Mixed Precision Exploration")
    print("-" * 50)
    
    # Print device information
    print(f"JAX devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
    
    try:
        # Get default half precision dtype
        half_dtype = get_default_half_dtype()
        print(f"Default half precision dtype: {half_dtype}")
        print("Note: bfloat16 is preferred for TPUs, float16 for GPUs")
        
        # Explore different policies
        print("\n1. Available Precision Policies:")
        print("-" * 30)
        
        policies = ["default", "mixed", "float16", "bfloat16", 
                "params=float32,compute=float16,output=float32"]
        
        for policy_name in policies:
            policy = get_policy(policy_name)
            print(f"\n{policy_name}:")
            print(f"  - Param dtype:   {policy.param_dtype}")
            print(f"  - Compute dtype: {policy.compute_dtype}")
            print(f"  - Output dtype:  {policy.output_dtype}")
        
        # Demonstrate loss scaling
        print("\n2. Loss Scaling Methods:")
        print("-" * 30)
        print("Loss scaling prevents gradient underflow in half precision training")
        
        # No loss scaling
        no_op = NoOpLossScale()
        print(f"NoOpLossScale: {no_op.loss_scale}")
        print("  - No scaling applied, useful with bfloat16 or when underflow isn't an issue")
        
        # Static loss scaling
        static = StaticLossScale(128.0)
        print(f"StaticLossScale: {static.loss_scale}")
        print("  - Fixed scale factor, simple but requires manual tuning")
        
        # Dynamic loss scaling
        dynamic = DynamicLossScale(initial_scale=16.0)
        print(f"DynamicLossScale: {dynamic.loss_scale}")
        print("  - Automatically adjusts based on gradient behavior")
        
        # Simulate a series of updates with the dynamic loss scale
        print("\n3. Dynamic Loss Scale Behavior:")
        print("-" * 30)
        print("Demonstration of how dynamic loss scaling adjusts during training:")
        
        # Simulate steps with the dynamic loss scale
        loss_scale = DynamicLossScale(initial_scale=16.0, growth_interval=10)
        print(f"Initial scale: {loss_scale.loss_scale}")
        
        # Simulate 10 steps with finite gradients
        for i in range(1, 11):
            grads_finite = jnp.array(True)
            loss_scale = loss_scale.adjust(grads_finite)
            print(f"Step {i}: Grads finite=True, Scale={loss_scale.loss_scale}")
        
        # Simulate a step with non-finite gradients
        grads_finite = jnp.array(False)
        loss_scale = loss_scale.adjust(grads_finite)
        print(f"Step 11: Grads finite=False, Scale={loss_scale.loss_scale} (reduced due to non-finite gradients)")
        
        # Resume with finite gradients
        for i in range(12, 15):
            grads_finite = jnp.array(True)
            loss_scale = loss_scale.adjust(grads_finite)
            print(f"Step {i}: Grads finite=True, Scale={loss_scale.loss_scale}")
    
    except Exception as e:
        print(f"\nError during exploration: {e}")
        print("This might be due to limitations in the current JAX environment.")


def main():
    """Main entry point for running examples."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="JAX Mixed Precision Examples")
    parser.add_argument("example", nargs="?", default="explore", 
                        choices=["explore", "benchmark", "all"],
                        help="Which example to run: explore, benchmark, or all")
    args = parser.parse_args()
    
    print("JAX Mixed Precision Examples")
    print("=" * 50)
    print("JAX devices:", jax.devices())
    
    # Run examples based on command-line argument
    if args.example in ["explore", "all"]:
        print("\n1. Exploring Mixed Precision Concepts")
        print("=" * 50)
        explore_mixed_precision()
    
    if args.example in ["benchmark", "all"]:
        print("\n2. Running Mixed Precision Benchmark")
        print("=" * 50)
        run_benchmark_comparison()


if __name__ == "__main__":
    main() 