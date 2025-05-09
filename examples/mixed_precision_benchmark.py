"""Mixed precision benchmark to demonstrate the implementation."""

import jax
import jax.numpy as jnp
import jax.tree_util
import equinox as eqx
import optax
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List

# Import directly from the module files instead of using __init__.py
# This avoids dependencies on other components like ArtifactConfig
import sys
import os

# Add the root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the mixed precision utilities directly
from utils.mixed_precision import (
    Policy, get_policy, NoOpLossScale, StaticLossScale, DynamicLossScale,
    get_default_half_dtype, all_finite, select_tree
)


@dataclass
class TrainConfig:
    """Training configuration."""
    learning_rate: float = 0.001
    seed: int = 42
    

@dataclass
class MixedPrecisionConfig:
    """Mixed precision configuration."""
    enable_mixed_precision: bool = False
    precision_policy: str = "default"
    loss_scaling: str = "none"
    loss_scale_value: float = 1.0
    loss_scale_growth_interval: int = 2000
    loss_scale_growth_factor: float = 2.0
    loss_scale_backoff_factor: float = 0.5


@dataclass
class BenchmarkConfig(TrainConfig, MixedPrecisionConfig):
    """Combined configuration for the benchmark."""
    input_dim: int = 512
    hidden_dim: int = 512
    output_dim: int = 10
    batch_size: int = 32
    num_steps: int = 20
    warmup_steps: int = 3


class MLPModel(eqx.Module):
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


class BenchmarkTask:
    """Simplified task for benchmarking mixed precision training."""
    
    def __init__(self, config: BenchmarkConfig):
        """Initialize the benchmark task."""
        self.config = config
        key = jax.random.PRNGKey(config.seed)
        
        # Create model and optimizer
        model_key, data_key = jax.random.split(key)
        self.model = MLPModel(
            config.input_dim, 
            config.hidden_dim, 
            config.output_dim, 
            model_key
        )
        self.optimizer = optax.adam(config.learning_rate)
        
        # Initialize optimizer state
        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))
        
        # Generate synthetic data
        self.x, self.y = self._generate_data(data_key)
        
        # For mixed precision
        if config.enable_mixed_precision:
            self.policy = self._create_policy()
            self.loss_scale = self._create_loss_scale()
        else:
            self.policy = get_policy("default")
            self.loss_scale = NoOpLossScale()
        
        # Track loss scale state
        self.loss_scale_value = jnp.array(self.loss_scale.loss_scale)
        self.steps_since_finite = jnp.array(getattr(self.loss_scale, "_steps_since_finite", 0))
        
        # JIT compile training step
        self.train_step_jit = jax.jit(self.train_step)
        
        # Track losses for plotting
        self.losses = []
    
    def _create_policy(self):
        """Create the precision policy based on configuration."""
        if not self.config.enable_mixed_precision:
            return get_policy("default")
        return get_policy(self.config.precision_policy)
    
    def _create_loss_scale(self):
        """Create the loss scale based on configuration."""
        if not self.config.enable_mixed_precision:
            return NoOpLossScale()
            
        if self.config.loss_scaling == "none":
            return NoOpLossScale()
        elif self.config.loss_scaling == "static":
            return StaticLossScale(self.config.loss_scale_value)
        elif self.config.loss_scaling == "dynamic":
            return DynamicLossScale(
                initial_scale=self.config.loss_scale_value,
                growth_interval=self.config.loss_scale_growth_interval,
                growth_factor=self.config.loss_scale_growth_factor,
                backoff_factor=self.config.loss_scale_backoff_factor
            )
        else:
            raise ValueError(f"Unknown loss scaling: {self.config.loss_scaling}")
    
    def _generate_data(self, key):
        """Generate synthetic data for training."""
        x_key, y_key = jax.random.split(key)
        x = jax.random.normal(x_key, (self.config.batch_size, self.config.input_dim))
        # For classification-style targets with noise to ensure non-zero loss
        y_indices = jax.random.randint(y_key, (self.config.batch_size,), 0, self.config.output_dim)
        y_onehot = jax.nn.one_hot(y_indices, self.config.output_dim)
        # Add a small amount of noise to make the problem non-trivial
        noise_key = jax.random.fold_in(y_key, 1)
        noise = jax.random.normal(noise_key, y_onehot.shape) * 0.1
        y = y_onehot + noise
        return x, y
    
    def scale_loss(self, loss):
        """Scale the loss if using mixed precision."""
        return loss * self.loss_scale_value
        
    def unscale_grads(self, grads):
        """Unscale gradients if using mixed precision."""
        return jax.tree_util.tree_map(lambda g: g / self.loss_scale_value, grads)
        
    def update_loss_scale(self, grads_finite):
        """Update the loss scale value based on gradient finiteness."""
        if isinstance(self.loss_scale, StaticLossScale):
            # Static loss scale doesn't change
            return self.loss_scale_value, self.steps_since_finite
        
        elif isinstance(self.loss_scale, DynamicLossScale):
            # Update steps counter
            new_steps = jnp.where(grads_finite, 
                                 self.steps_since_finite + 1, 
                                 jnp.array(0))
            
            # Check if we should increase the scale
            should_increase = (new_steps >= self.loss_scale._growth_interval)
            
            # Update scale based on both conditions
            new_scale = jnp.where(
                ~grads_finite,
                # If not finite, reduce the scale
                self.loss_scale_value * self.loss_scale._backoff_factor,
                # If finite, maybe increase if we hit the growth interval
                jnp.where(
                    should_increase,
                    self.loss_scale_value * self.loss_scale._growth_factor,
                    self.loss_scale_value
                )
            )
            
            # Reset counter if we increased the scale or had non-finite gradients
            new_steps = jnp.where(should_increase | ~grads_finite, jnp.array(0), new_steps)
            
            return new_scale, new_steps
        else:
            # NoOpLossScale doesn't change
            return self.loss_scale_value, self.steps_since_finite
    
    def cast_params_to_compute(self, model):
        """Cast model parameters to compute dtype if using mixed precision."""
        if not self.config.enable_mixed_precision:
            return model
        return self.policy.cast_to_compute(model)
    
    def get_output_and_loss(self, model, x, y):
        """Compute model output and loss."""
        # Apply mixed precision policy to model and inputs if enabled
        if self.config.enable_mixed_precision:
            # Use the policy to cast parameters and inputs
            model = self.policy.cast_to_compute(model)
            x = self.policy.cast_to_compute(x)
        
        # Forward pass
        logits = jax.vmap(model)(x)
        
        # Cast output if needed
        if self.config.enable_mixed_precision:
            logits = self.policy.cast_to_output(logits)
        
        # Compute cross-entropy loss
        log_softmax = jax.nn.log_softmax(logits)
        loss = -jnp.mean(jnp.sum(y * log_softmax, axis=1))
        
        # Add small regularization to prevent loss from becoming too small
        # This helps with training and plotting stability
        l2_reg = 1e-5 * sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
        loss = loss + l2_reg
        
        return loss, logits
    
    def train_step(self, model):
        """Perform a single training step."""
        x, y = self.x, self.y
        
        # Define the loss function for computing gradients
        def loss_fn(m):
            loss, _ = self.get_output_and_loss(m, x, y)
            # Scale loss for mixed precision
            if self.config.enable_mixed_precision:
                loss = self.scale_loss(loss)
            return loss
        
        # Compute loss and gradients
        loss, grads = jax.value_and_grad(loss_fn)(model)
        
        # Process for mixed precision
        if self.config.enable_mixed_precision:
            # Unscale gradients
            grads = self.unscale_grads(grads)
            
            # Check if gradients are finite
            grads_finite = all_finite(grads)
            
            # Update the loss scale based on whether gradients are finite
            new_scale, new_steps = self.update_loss_scale(grads_finite)
            
            # Update optimizer state and model parameters
            updates, new_opt_state = self.optimizer.update(grads, self.opt_state, model)
            new_model = eqx.apply_updates(model, updates)
            
            # Select between new and old based on gradient finiteness
            model = select_tree(grads_finite, new_model, model)
            opt_state = select_tree(grads_finite, new_opt_state, self.opt_state)
            
            # Update internal state
            self.loss_scale_value = new_scale
            self.steps_since_finite = new_steps
            self.opt_state = opt_state
        else:
            # Standard update
            updates, new_opt_state = self.optimizer.update(grads, self.opt_state, model)
            model = eqx.apply_updates(model, updates)
            self.opt_state = new_opt_state
        
        # Compute true loss (unscaled) for reporting
        true_loss, _ = self.get_output_and_loss(model, x, y)
        
        return true_loss, self.opt_state, model
    
    def run_benchmark(self):
        """Run the benchmark and return timing results."""
        print(f"Running benchmark with mixed precision {'enabled' if self.config.enable_mixed_precision else 'disabled'}...")
        if self.config.enable_mixed_precision:
            print(f"  Policy: {self.policy}")
            print(f"  Loss scaling: starting at {self.loss_scale_value}")
        print(f"  Model: MLP with {self.config.input_dim} input, {self.config.hidden_dim} hidden, {self.config.output_dim} output")
        print(f"  Batch size: {self.config.batch_size}")
        
        # Measure model memory size
        model_size_bytes = sum(x.nbytes for x in jax.tree_util.tree_leaves(eqx.filter(self.model, eqx.is_array)))
        model_size_mb = model_size_bytes / (1024 * 1024)
        print(f"  Model size: {model_size_mb:.2f} MB")
        
        # Warm up JIT compilation with specified warmup steps
        print("  Warming up JIT compilation...")
        model = self.model
        for _ in range(self.config.warmup_steps):
            _, self.opt_state, model = self.train_step_jit(model)
        
        # Reset model for the actual benchmark
        model = self.model
        self.opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))
        self.losses = []
        
        # Run the benchmark
        print(f"  Running {self.config.num_steps} training steps...")
        start_time = time.time()
        loss = 0
        for i in range(self.config.num_steps):
            loss, self.opt_state, model = self.train_step_jit(model)
            self.losses.append(float(loss))
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        time_per_step = total_time / self.config.num_steps
        
        # Get memory usage if available
        try:
            mem_usage = jax.live_arrays()
            total_bytes = sum(x.nbytes for x in mem_usage)
            memory_mb = total_bytes / (1024 * 1024)
        except:
            memory_mb = model_size_mb * 2  # Rough approximation
        
        # Print results
        print(f"  Total time: {total_time:.4f} seconds")
        print(f"  Time per step: {time_per_step*1000:.2f} ms")
        print(f"  Final loss: {loss:.6f}")
        print(f"  Model layer1 weight dtype: {model.layer1_weight.dtype}")
        print(f"  Memory usage: {memory_mb:.2f} MB")
        
        # Return results as dictionary
        results = {
            "total_time": total_time,
            "time_per_step": time_per_step,
            "final_loss": float(loss),
            "model_dtype": str(model.layer1_weight.dtype),
            "memory_mb": memory_mb,
            "losses": self.losses
        }
        return results


def run_comparison():
    """Run the benchmark with and without mixed precision."""
    half_dtype = get_default_half_dtype()
    print(f"Default half precision dtype: {half_dtype}")
    
    # Common configuration - use smaller values suitable for CPU
    common_config = {
        "input_dim": 512,
        "hidden_dim": 512,
        "output_dim": 10,
        "batch_size": 32,
        "num_steps": 20,
        "warmup_steps": 3,
        "seed": 42,
    }
    
    # Run with full precision
    fp_config = BenchmarkConfig(
        enable_mixed_precision=False,
        precision_policy="default",
        **common_config
    )
    print("\n" + "="*60)
    print("FULL PRECISION BENCHMARK")
    print("="*60)
    fp_task = BenchmarkTask(fp_config)
    fp_results = fp_task.run_benchmark()
    
    # Run with mixed precision
    mp_config = BenchmarkConfig(
        enable_mixed_precision=True,
        precision_policy="mixed",  # params=float32, compute=half, output=float32
        loss_scaling="dynamic",
        loss_scale_value=2**15,
        **common_config
    )
    print("\n" + "="*60)
    print("MIXED PRECISION BENCHMARK")
    print("="*60)
    mp_task = BenchmarkTask(mp_config)
    mp_results = mp_task.run_benchmark()
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    if fp_results["time_per_step"] > 0 and mp_results["time_per_step"] > 0:
        speedup = fp_results["time_per_step"] / mp_results["time_per_step"]
        print(f"Speed comparison: mixed precision is {speedup:.2f}x faster")
        print(f"  - Full precision: {fp_results['time_per_step']*1000:.2f} ms/step")
        print(f"  - Mixed precision: {mp_results['time_per_step']*1000:.2f} ms/step")
    
    if fp_results["memory_mb"] and mp_results["memory_mb"]:
        memory_ratio = fp_results["memory_mb"] / mp_results["memory_mb"]
        print(f"Memory comparison: mixed precision uses {1/memory_ratio:.2f}x less memory")
        print(f"  - Full precision: {fp_results['memory_mb']:.2f} MB")
        print(f"  - Mixed precision: {mp_results['memory_mb']:.2f} MB")
    
    print(f"Loss comparison:")
    print(f"  - Full precision final loss: {fp_results['final_loss']:.6f}")
    print(f"  - Mixed precision final loss: {mp_results['final_loss']:.6f}")
    
    return {
        "full_precision": fp_results,
        "mixed_precision": mp_results,
    }


def plot_results(results):
    """Plot the benchmark results."""
    fp = results["full_precision"]
    mp = results["mixed_precision"]
    
    # Create plots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Time comparison
    methods = ["Full Precision", "Mixed Precision"]
    times = [fp["time_per_step"]*1000, mp["time_per_step"]*1000]  # Convert to ms
    axs[0, 0].bar(methods, times, color=["blue", "orange"])
    axs[0, 0].set_ylabel("Time per step (ms)")
    axs[0, 0].set_title("Training Speed Comparison")
    
    # Speedup annotation
    speedup = fp["time_per_step"] / mp["time_per_step"]
    speedup_text = f"{speedup:.2f}x faster" if speedup > 1 else f"{1/speedup:.2f}x slower"
    axs[0, 0].text(1, times[1] * 1.1, speedup_text, 
            ha="center", va="bottom", fontweight="bold")
    
    # 2. Memory comparison
    if fp["memory_mb"] and mp["memory_mb"]:
        memories = [fp["memory_mb"], mp["memory_mb"]]
        axs[0, 1].bar(methods, memories, color=["blue", "orange"])
        axs[0, 1].set_ylabel("Memory Usage (MB)")
        axs[0, 1].set_title("Memory Usage Comparison")
        
        # Memory savings annotation
        memory_ratio = fp["memory_mb"] / mp["memory_mb"]
        if memory_ratio > 1:
            mem_text = f"{memory_ratio:.2f}x less memory"
        else:
            mem_text = f"{1/memory_ratio:.2f}x more memory"
        axs[0, 1].text(1, memories[1] * 1.1, mem_text, 
                ha="center", va="bottom", fontweight="bold")
    else:
        axs[0, 1].text(0.5, 0.5, "Memory usage data not available", 
                ha="center", va="center", transform=axs[0, 1].transAxes)
    
    # 3. Training loss curve
    if "losses" in fp and "losses" in mp:
        steps = range(1, len(fp["losses"]) + 1)
        axs[1, 0].plot(steps, fp["losses"], label="Full Precision")
        axs[1, 0].plot(steps, mp["losses"], label="Mixed Precision")
        axs[1, 0].set_xlabel("Training Steps")
        axs[1, 0].set_ylabel("Loss")
        axs[1, 0].set_title("Training Loss Comparison")
        axs[1, 0].legend()
        axs[1, 0].grid(True, alpha=0.3)
    
    # 4. Final loss comparison
    final_losses = [fp["final_loss"], mp["final_loss"]]
    axs[1, 1].bar(methods, final_losses, color=["blue", "orange"])
    axs[1, 1].set_ylabel("Final Loss")
    axs[1, 1].set_title("Final Loss Comparison")
    
    # Add loss difference annotation
    loss_diff = abs(fp["final_loss"] - mp["final_loss"])
    # Avoid division by zero
    if abs(fp["final_loss"]) > 1e-10:
        relative_diff = loss_diff / abs(fp["final_loss"]) * 100
        diff_text = f"Difference: {relative_diff:.2f}%"
    else:
        diff_text = f"Absolute diff: {loss_diff:.6f}"
    axs[1, 1].text(0.5, 0.9, diff_text, 
            ha="center", transform=axs[1, 1].transAxes, fontweight="bold")
    
    fig.tight_layout()
    
    # Add a title for the whole figure
    fig.suptitle("Mixed Precision vs Full Precision Training Benchmark", fontsize=16)
    fig.subplots_adjust(top=0.92)
    
    # Save the figure
    plt.savefig("mixed_precision_benchmark_results.png", dpi=120)
    print("Results saved to mixed_precision_benchmark_results.png")


if __name__ == "__main__":
    print("Mixed Precision Benchmark")
    print("========================")
    print("JAX devices:", jax.devices())
    print(f"Default backend: {jax.default_backend()}")
    
    results = run_comparison()
    
    try:
        plot_results(results)
    except Exception as e:
        print(f"Could not create plot: {str(e)}") 