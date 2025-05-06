"""Mixed precision benchmark to demonstrate the implementation."""

import jax
import jax.numpy as jnp
import jax.tree_util
import equinox as eqx
import optax
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Protocol

from xax.task.mixins.train import TrainConfig, TrainMixin
from xax.task.mixins.mixed_precision import MixedPrecisionMixin, MixedPrecisionConfig
from xax.utils.mixed_precision import (
    Policy, get_policy, NoOpLossScale, StaticLossScale, DynamicLossScale,
    get_default_half_dtype, all_finite, select_tree
)


@dataclass
class TestConfig(TrainConfig, MixedPrecisionConfig):
    """Combined configuration for the benchmark."""
    input_dim: int = 784
    output_dim: int = 10
    batch_size: int = 32
    num_steps: int = 100


class LinearModel(eqx.Module):
    """A simple linear model for testing."""
    weight: jnp.ndarray
    bias: jnp.ndarray
    
    def __init__(self, in_dim, out_dim, key):
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.normal(wkey, (in_dim, out_dim))
        self.bias = jax.random.normal(bkey, (out_dim,))
    
    def __call__(self, x):
        return x @ self.weight + self.bias


class BenchmarkTask(TrainMixin[TestConfig], MixedPrecisionMixin[TestConfig]):
    """Task that combines TrainMixin and MixedPrecisionMixin for benchmarking."""
    
    def __init__(self, config: TestConfig):
        """Initialize the benchmark task."""
        self.config = config
        key = jax.random.PRNGKey(config.seed)
        
        # Create model and optimizer
        model_key, data_key = jax.random.split(key)
        self.model = LinearModel(config.input_dim, config.output_dim, model_key)
        self.optimizer = optax.adam(config.learning_rate)
        
        # Initialize optimizer state
        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))
        
        # Generate synthetic data
        self.x, self.y = self._generate_data(data_key)
        
        # For mixed precision
        if config.enable_mixed_precision:
            self.policy = self._create_policy()
            self.loss_scale = self._create_loss_scale()
        
        # JIT compile training step
        self.train_step_jit = jax.jit(self.train_step_with_data)
    
    def _create_policy(self):
        """Create the precision policy based on configuration."""
        return self.policy
    
    def _create_loss_scale(self):
        """Create the loss scale based on configuration."""
        return self.loss_scale
    
    def _generate_data(self, key):
        """Generate synthetic data for training."""
        x_key, y_key = jax.random.split(key)
        x = jax.random.normal(x_key, (self.config.batch_size, self.config.input_dim))
        y = jax.random.normal(y_key, (self.config.batch_size, self.config.output_dim))
        return x, y
    
    def get_output_and_loss(self, model, batch, train=True):
        """Compute model output and loss."""
        x, y = batch
        
        # Apply mixed precision policy to model and inputs if enabled
        if self.config.enable_mixed_precision:
            # Use the policy to cast parameters and inputs
            model = self.policy.cast_to_compute(model)
            x = self.policy.cast_to_compute(x)
        
        # Forward pass
        pred = jax.vmap(model)(x)
        
        # Compute loss
        loss = jnp.mean((pred - y) ** 2)
        
        # Scale loss if using mixed precision
        if self.config.enable_mixed_precision:
            loss = self.scale_loss(loss)
        
        return loss, pred
    
    def train_step_with_data(self, model, batch=None):
        """Training step that uses the stored data if no batch is provided."""
        if batch is None:
            batch = (self.x, self.y)
        return self.train_step(model, batch)
    
    def run_benchmark(self):
        """Run the benchmark and return timing results."""
        print(f"Running benchmark with mixed precision {'enabled' if self.config.enable_mixed_precision else 'disabled'}...")
        if self.config.enable_mixed_precision:
            print(f"  Policy: {self.policy}")
            print(f"  Loss scaling: {self.loss_scale}")
        
        # Warm up JIT compilation
        _, _, _ = self.train_step_jit(self.model)
        
        # Run the benchmark
        start_time = time.time()
        loss = 0
        model = self.model
        for _ in range(self.config.num_steps):
            loss, self.opt_state, model = self.train_step_jit(model)
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
            memory_mb = None
        
        # Print results
        print(f"  Total time: {total_time:.4f} seconds")
        print(f"  Time per step: {time_per_step:.6f} seconds")
        print(f"  Final loss: {loss}")
        print(f"  Model weight dtype: {model.weight.dtype}")
        if memory_mb is not None:
            print(f"  Memory usage: {memory_mb:.2f} MB")
        
        # Return results as dictionary
        results = {
            "total_time": total_time,
            "time_per_step": time_per_step,
            "final_loss": float(loss),
            "model_dtype": str(model.weight.dtype),
            "memory_mb": memory_mb,
        }
        return results


def run_comparison():
    """Run the benchmark with and without mixed precision."""
    half_dtype = get_default_half_dtype()
    print(f"Default half precision dtype: {half_dtype}")
    
    # Run with full precision
    fp_config = TestConfig(
        enable_mixed_precision=False,
        precision_policy="default",
        seed=42,
    )
    fp_task = BenchmarkTask(fp_config)
    fp_results = fp_task.run_benchmark()
    
    # Run with mixed precision
    mp_config = TestConfig(
        enable_mixed_precision=True,
        precision_policy="mixed",  # params=float32, compute=half, output=float32
        loss_scaling="dynamic",
        loss_scale_value=2**15,
        seed=42,
    )
    mp_task = BenchmarkTask(mp_config)
    mp_results = mp_task.run_benchmark()
    
    # Compare results
    if fp_results["time_per_step"] > 0 and mp_results["time_per_step"] > 0:
        speedup = fp_results["time_per_step"] / mp_results["time_per_step"]
        print(f"\nSpeed comparison: mixed precision is {speedup:.2f}x faster")
    
    if fp_results["memory_mb"] and mp_results["memory_mb"]:
        memory_ratio = fp_results["memory_mb"] / mp_results["memory_mb"]
        print(f"Memory comparison: mixed precision uses {1/memory_ratio:.2f}x less memory")
    
    return {
        "full_precision": fp_results,
        "mixed_precision": mp_results,
    }


if __name__ == "__main__":
    print("JAX devices:", jax.devices())
    print(f"Default backend: {jax.default_backend()}")
    
    results = run_comparison() 