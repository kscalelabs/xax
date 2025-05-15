"""Mixed precision examples demonstrating JAX mixed precision training.

This file provides examples of mixed precision training with JAX:
1. Simple benchmark comparing training speed and memory usage
2. MNIST training example with mixed precision
3. Utility for exploring mixed precision behavior

Run one of the examples with:
    python -m xax.examples.mixed_precision --example benchmark
    python -m xax.examples.mixed_precision --example mnist
    python -m xax.examples.mixed_precision --example explore

Each example demonstrates how to use the mixed precision utilities in xax.
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import time
import argparse
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List, Union

from xax.core.state import State
from xax.task.script import Script, ScriptConfig
from xax.task.mixins.train import TrainConfig, TrainMixin
from xax.task.mixins.mixed_precision import MixedPrecisionConfig, MixedPrecisionMixin
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


class CNN(eqx.Module):
    """Simple CNN model for MNIST."""
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    
    def __init__(self, *, key):
        conv1_key, conv2_key, linear1_key, linear2_key = jax.random.split(key, 4)
        # Equinox Conv2d expects input shape [H, W, C_in]
        self.conv1 = eqx.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, key=conv1_key)
        self.conv2 = eqx.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, key=conv2_key)
        self.linear1 = eqx.nn.Linear(9216, 128, key=linear1_key)
        self.linear2 = eqx.nn.Linear(128, 10, key=linear2_key)
    
    def __call__(self, x):
        # Input shape: [H, W, C_in]
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.max_pool(x, (2, 2), strides=(2, 2))
        x = jax.nn.relu(self.conv2(x))
        x = jax.nn.max_pool(x, (2, 2), strides=(2, 2))
        # Flatten for the linear layer
        x = x.reshape(-1)
        x = jax.nn.relu(self.linear1(x))
        return self.linear2(x)


#------------------------------------------------------------------------------
# BENCHMARK EXAMPLE
#------------------------------------------------------------------------------

@dataclass
class BenchmarkConfig(TrainConfig, MixedPrecisionConfig):
    """Configuration for benchmark tasks."""
    input_dim: int = 512
    hidden_dim: int = 512
    output_dim: int = 10
    batch_size: int = 128
    num_iterations: int = 100
    warmup_steps: int = 3
    

class BenchmarkTask(TrainMixin[BenchmarkConfig], MixedPrecisionMixin[BenchmarkConfig]):
    """Task for benchmarking mixed precision vs regular precision."""
    
    def __init__(self, config: BenchmarkConfig):
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
        self.optimizer = optax.adam(config.learning_rate)
        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))
        
        # Generate synthetic data
        self._generate_data(data_key)
        
        # Mixed precision setup
        self.policy = self._create_policy()
        self.loss_scale = self._create_loss_scale()
        
        # JIT compile the training step
        self.jit_train_step = jax.jit(self.train_step)
    
    def _generate_data(self, key):
        """Generate synthetic data for training."""
        x_key, y_key = jax.random.split(key)
        self.images = jax.random.normal(x_key, (self.config.batch_size, self.config.input_dim))
        y_indices = jax.random.randint(y_key, (self.config.batch_size,), 0, self.config.output_dim)
        self.labels = jax.nn.one_hot(y_indices, self.config.output_dim)
    
    def get_batch(self):
        """Return a batch of data."""
        return (self.images, self.labels)
    
    def get_output_and_loss(self, model, batch):
        """Compute model output and loss."""
        x, y = batch
        logits = jax.vmap(model)(x)
        loss = optax.softmax_cross_entropy(logits, y).mean()
        return logits, loss
    
    def run_benchmark(self) -> Dict[str, float]:
        """Run benchmark and return timing results."""
        results = {}
        
        # Warmup JIT compilation
        for _ in range(self.config.warmup_steps):
            _, _, self.model = self.jit_train_step(self.model, self.get_batch())
        
        # Time training steps
        start_time = time.time()
        for _ in range(self.config.num_iterations):
            _, _, self.model = self.jit_train_step(self.model, self.get_batch())
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
    """Run comparison between full precision and mixed precision."""
    results = {}
    
    print("Running full precision benchmark...")
    fp_config = BenchmarkConfig(
        enable_mixed_precision=False,
        precision_policy="default",
        seed=42
    )
    fp_task = BenchmarkTask(fp_config)
    results["full_precision"] = fp_task.run_benchmark()
    print(f"Full precision: {results['full_precision']['time_per_step']:.4f} seconds/step")
    
    print("\nRunning mixed precision benchmark...")
    mp_config = BenchmarkConfig(
        enable_mixed_precision=True,
        precision_policy="mixed",
        loss_scaling="dynamic",
        loss_scale_value=2**15,
        seed=42
    )
    mp_task = BenchmarkTask(mp_config)
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
    _plot_benchmark_results(results)
    
    return results


def _plot_benchmark_results(results):
    """Plot comparison results between full and mixed precision."""
    if "full_precision" not in results or "mixed_precision" not in results:
        print("Missing data for comparison plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot time per step
    times = [
        results["full_precision"]["time_per_step"],
        results["mixed_precision"]["time_per_step"]
    ]
    ax1.bar(["Full Precision", "Mixed Precision"], times, color=["blue", "orange"])
    ax1.set_ylabel("Time per step (seconds)")
    ax1.set_title("Training Speed Comparison")
    
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
    
    plt.tight_layout()
    plt.savefig("mixed_precision_benchmark_results.png")
    plt.show()


#------------------------------------------------------------------------------
# MNIST EXAMPLE
#------------------------------------------------------------------------------

@dataclass
class MNISTConfig(ScriptConfig, MixedPrecisionConfig):
    """Configuration for MNIST training with mixed precision."""
    batch_size: int = 64
    learning_rate: float = 0.001
    num_epochs: int = 5
    random_seed: int = 42


class MNISTTask(Script[MNISTConfig], MixedPrecisionMixin[MNISTConfig]):
    """Task for training MNIST with mixed precision."""
    
    def __init__(self, config: MNISTConfig):
        super().__init__(config)
        self.model = None
        self.optimizer = None
    
    def setup(self):
        # Initialize the model and optimizer
        key = jax.random.PRNGKey(self.config.random_seed)
        self.model = CNN(key=key)
        self.optimizer = optax.adam(self.config.learning_rate)
        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))
        
        # Set up mixed precision utilities
        self.policy = self._create_policy()
        self.loss_scale = self._create_loss_scale()
        
        # Print configuration
        print(f"Mixed precision: {'Enabled' if self.config.enable_mixed_precision else 'Disabled'}")
        if self.config.enable_mixed_precision:
            print(f"- Policy: {self.config.precision_policy}")
            print(f"- Loss scaling: {self.config.loss_scaling} (initial scale: {self.loss_scale.loss_scale})")
            print(f"- Compute dtype: {self.policy.compute_dtype}")
            print(f"- Parameter dtype: {self.policy.param_dtype}")
            print(f"- Output dtype: {self.policy.output_dtype}")
    
    def load_data(self):
        """Create synthetic MNIST-like data for demonstration."""
        print("Creating synthetic MNIST data...")
        key = jax.random.PRNGKey(self.config.random_seed)
        key1, key2 = jax.random.split(key)
        
        # Create training data: 6000 images of size 28x28
        x_train = jax.random.normal(key1, (6000, 28, 28, 1))
        y_train = jax.random.randint(key1, (6000,), 0, 10)
        
        # Create test data: 1000 images of size 28x28
        x_test = jax.random.normal(key2, (1000, 28, 28, 1))
        y_test = jax.random.randint(key2, (1000,), 0, 10)
        
        # Convert to numpy for easier handling
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        
        return (x_train, y_train), (x_test, y_test)
    
    def compute_loss(self, model, x, y):
        """Compute loss and accuracy."""
        # Forward pass
        logits = jax.vmap(model)(x)
        one_hot = jax.nn.one_hot(y, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        accuracy = jnp.mean(jnp.argmax(logits, axis=1) == y)
        return loss, accuracy
    
    def train_step(self, model, opt_state, x, y):
        """Execute one training step with mixed precision support."""
        # Define loss function
        def loss_fn(model):
            # Cast model to compute precision
            if self.config.enable_mixed_precision:
                model = self.cast_params_to_compute(model)
            
            # Compute loss
            loss, accuracy = self.compute_loss(model, x, y)
            
            # Scale loss for mixed precision training
            if self.config.enable_mixed_precision:
                loss = self.scale_loss(loss)
                
            return loss, (loss, accuracy)
        
        # Compute gradients
        (_, (loss, accuracy)), grads = jax.value_and_grad(loss_fn, has_aux=True)(model)
        
        # Handle mixed precision updates
        if self.config.enable_mixed_precision:
            # Unscale gradients
            grads = self.unscale_grads(grads)
            
            # Check if gradients are finite
            grads_finite = self.check_grads_finite(grads)
            
            # Define optimizer update function
            def optimizer_update(params, opt_state):
                updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
                new_params = eqx.apply_updates(params, updates)
                return new_params, new_opt_state
            
            # Apply mixed precision update
            model, opt_state, new_loss_scale = self.mixed_precision_update(
                model, grads, optimizer_update, opt_state
            )
            
            # Update loss scale
            self.set_loss_scale(new_loss_scale)
        else:
            # Standard update
            updates, new_opt_state = self.optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            opt_state = new_opt_state
        
        # Cast model back to parameter precision if using mixed precision
        if self.config.enable_mixed_precision:
            model = self.cast_params_to_storage(model)
            
        return model, opt_state, loss, accuracy
    
    def validation_step(self, model, x, y):
        """Evaluate model on validation data."""
        # Cast model to compute precision for validation
        if self.config.enable_mixed_precision:
            model = self.cast_params_to_compute(model)
            
        loss, accuracy = self.compute_loss(model, x, y)
        return loss, accuracy
    
    def run(self):
        """Run MNIST training."""
        self.setup()
        (x_train, y_train), (x_test, y_test) = self.load_data()
        
        # Create JIT-compiled versions of steps
        jit_train_step = jax.jit(self.train_step)
        jit_validation_step = jax.jit(self.validation_step)
        
        # Training loop
        step = 0
        
        for epoch in range(self.config.num_epochs):
            # Shuffle training data
            perm = np.random.permutation(len(x_train))
            x_train_shuffled = x_train[perm]
            y_train_shuffled = y_train[perm]
            
            # Training
            epoch_losses = []
            epoch_accuracies = []
            
            for i in range(0, len(x_train), self.config.batch_size):
                batch_end = min(i + self.config.batch_size, len(x_train))
                x_batch = x_train_shuffled[i:batch_end]
                y_batch = y_train_shuffled[i:batch_end]
                
                self.model, self.opt_state, loss, accuracy = jit_train_step(
                    self.model, self.opt_state, x_batch, y_batch
                )
                
                epoch_losses.append(loss)
                epoch_accuracies.append(accuracy)
                
                # Log training progress
                if i % (10 * self.config.batch_size) == 0:
                    loss_scale_value = float(self.loss_scale.loss_scale) if self.config.enable_mixed_precision else 1.0
                    print(f"Epoch {epoch+1}/{self.config.num_epochs}, Step {step}, "
                          f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, "
                          f"Loss scale: {loss_scale_value:.1f}")
                
                step += 1
            
            # Calculate epoch averages
            avg_train_loss = np.mean(epoch_losses)
            avg_train_accuracy = np.mean(epoch_accuracies)
            print(f"Training - Epoch {epoch+1}/{self.config.num_epochs}, "
                  f"Avg Loss: {avg_train_loss:.4f}, Avg Accuracy: {avg_train_accuracy:.4f}")
            
            # Validation
            val_losses, val_accuracies = [], []
            for i in range(0, len(x_test), self.config.batch_size):
                batch_end = min(i + self.config.batch_size, len(x_test))
                x_batch = x_test[i:batch_end]
                y_batch = y_test[i:batch_end]
                
                loss, accuracy = jit_validation_step(self.model, x_batch, y_batch)
                
                val_losses.append(loss)
                val_accuracies.append(accuracy)
            
            # Log validation results
            avg_val_loss = np.mean(val_losses)
            avg_val_accuracy = np.mean(val_accuracies)
            print(f"Validation - Epoch {epoch+1}/{self.config.num_epochs}, "
                  f"Avg Loss: {avg_val_loss:.4f}, Avg Accuracy: {avg_val_accuracy:.4f}")
        
        print("Training complete!")
        
        # Print model parameter info
        if self.config.enable_mixed_precision:
            print("\nModel parameter dtypes:")
            _print_model_dtype_info(self.model)


def run_mnist_example():
    """Run the MNIST example with mixed precision."""
    # Run with mixed precision enabled
    config = MNISTConfig(
        enable_mixed_precision=True,
        precision_policy="mixed",
        loss_scaling="dynamic",
        loss_scale_value=2**15,
        batch_size=64,
        num_epochs=3
    )
    task = MNISTTask(config)
    task.run()


#------------------------------------------------------------------------------
# EXPLORATION UTILITIES
#------------------------------------------------------------------------------

def _print_model_dtype_info(model):
    """Print information about model parameter dtypes."""
    def print_dtypes(x):
        if hasattr(x, 'dtype'):
            return f"{x.shape}, {x.dtype}"
        return None
    
    params = jax.tree_util.tree_map(print_dtypes, model)
    flat_params = jax.tree_util.tree_leaves(params)
    non_none_params = [p for p in flat_params if p is not None]
    
    # Group by dtype
    dtype_groups = {}
    for param_info in non_none_params:
        dtype = param_info.split(', ')[1]
        if dtype not in dtype_groups:
            dtype_groups[dtype] = []
        dtype_groups[dtype].append(param_info)
    
    # Print summary
    print(f"Total parameters: {len(non_none_params)}")
    for dtype, params in dtype_groups.items():
        print(f"  {dtype}: {len(params)} parameters")
        # Print a few examples
        for i, param in enumerate(params[:3]):
            print(f"    - {param}")
        if len(params) > 3:
            print(f"    - ... and {len(params) - 3} more")


def explore_mixed_precision():
    """Explore mixed precision behavior with different policies."""
    print("JAX Mixed Precision Exploration")
    print("-" * 50)
    
    # Print device information
    print(f"JAX devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
    
    # Get default half precision dtype
    half_dtype = get_default_half_dtype()
    print(f"Default half precision dtype: {half_dtype}")
    
    # Explore different policies
    policies = ["default", "mixed", "float16", "bfloat16", 
               "params=float32,compute=float16,output=float32"]
    
    print("\nAvailable Precision Policies:")
    for policy_name in policies:
        policy = get_policy(policy_name)
        print(f"\n{policy_name}:")
        print(f"  - Param dtype:   {policy.param_dtype}")
        print(f"  - Compute dtype: {policy.compute_dtype}")
        print(f"  - Output dtype:  {policy.output_dtype}")
    
    # Demonstrate loss scaling
    print("\nLoss Scaling Methods:")
    
    # No loss scaling
    no_op = NoOpLossScale()
    print(f"NoOpLossScale: {no_op.loss_scale}")
    
    # Static loss scaling
    static = StaticLossScale(128.0)
    print(f"StaticLossScale: {static.loss_scale}")
    
    # Dynamic loss scaling
    dynamic = DynamicLossScale(initial_scale=16.0)
    print(f"DynamicLossScale: {dynamic.loss_scale}")
    
    # Simulate a series of updates with the dynamic loss scale
    print("\nDynamic Loss Scale Behavior:")
    scale = dynamic.loss_scale
    print(f"Initial scale: {scale}")
    
    # Simulate 10 steps with finite gradients (should increase after growth_interval)
    for i in range(1, 11):
        grads_finite = True
        scale, _ = dynamic.update(scale, grads_finite)
        print(f"Step {i}: Grads finite={grads_finite}, Scale={scale}")
    
    # Simulate non-finite gradients (should decrease immediately)
    grads_finite = False
    scale, _ = dynamic.update(scale, grads_finite)
    print(f"Step 11: Grads finite={grads_finite}, Scale={scale}")
    
    # Resume with finite gradients
    for i in range(12, 15):
        grads_finite = True
        scale, _ = dynamic.update(scale, grads_finite)
        print(f"Step {i}: Grads finite={grads_finite}, Scale={scale}")


#------------------------------------------------------------------------------
# MAIN ENTRY POINT
#------------------------------------------------------------------------------

def main():
    """Main entry point for running examples."""
    parser = argparse.ArgumentParser(description="Mixed Precision Examples")
    parser.add_argument("--example", type=str, default="benchmark",
                        choices=["benchmark", "mnist", "explore"],
                        help="Which example to run")
    args = parser.parse_args()
    
    print(f"Running {args.example} example")
    print("JAX devices:", jax.devices())
    
    if args.example == "benchmark":
        run_benchmark_comparison()
    elif args.example == "mnist":
        run_mnist_example()
    elif args.example == "explore":
        explore_mixed_precision()
    else:
        print(f"Unknown example: {args.example}")


if __name__ == "__main__":
    main() 