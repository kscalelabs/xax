"""Benchmark script to compare mixed precision vs regular precision training."""

import jax
import jax.numpy as jnp
import numpy as np
import time
import equinox as eqx
import optax
from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional
import argparse

from xax.task.mixins.train import TrainConfig, TrainMixin
from xax.task.mixins.mixed_precision import MixedPrecisionConfig, MixedPrecisionMixin


class CNN(eqx.Module):
    """Simple CNN model for benchmarking."""
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    dense1: eqx.nn.Linear
    dense2: eqx.nn.Linear
    
    def __init__(self, key):
        keys = jax.random.split(key, 4)
        # Make the model larger to better demonstrate performance differences
        self.conv1 = eqx.nn.Conv2d(1, 64, kernel_size=3, key=keys[0])
        self.conv2 = eqx.nn.Conv2d(64, 128, kernel_size=3, key=keys[1])
        self.dense1 = eqx.nn.Linear(128 * 5 * 5, 512, key=keys[2])
        self.dense2 = eqx.nn.Linear(512, 10, key=keys[3])
    
    def __call__(self, x):
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.max_pool(x, (2, 2), strides=(2, 2))
        x = jax.nn.relu(self.conv2(x))
        x = jax.nn.max_pool(x, (2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = jax.nn.relu(self.dense1(x))
        x = self.dense2(x)
        return x


@dataclass
class BenchmarkConfig(TrainConfig, MixedPrecisionConfig):
    """Configuration for benchmark tasks."""
    batch_size: int = 128
    learning_rate: float = 0.001
    num_iterations: int = 100
    model_size_factor: int = 1  # Scale model size for more realistic benchmarks
    

class BenchmarkTask(TrainMixin[BenchmarkConfig], MixedPrecisionMixin[BenchmarkConfig]):
    """Task for benchmarking mixed precision vs regular precision."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        key = jax.random.PRNGKey(config.seed)
        self.model = CNN(key)
        self.optimizer = optax.adam(config.learning_rate)
        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))
        self.loss_scale = self._create_loss_scale()
        self.policy = self._create_policy()
        
        # Generate synthetic data for benchmarking
        self._generate_data()
        
        # Compile the training step for better performance
        self.jit_train_step = jax.jit(self.train_step)
    
    def _generate_data(self):
        """Generate synthetic data for training."""
        key = jax.random.PRNGKey(self.config.seed)
        key1, key2 = jax.random.split(key)
        # Create random training data
        self.images = jax.random.normal(key1, (self.config.batch_size, 1, 28, 28))
        self.labels = jax.random.randint(key2, (self.config.batch_size,), 0, 10)
        # One-hot encode labels
        self.one_hot_labels = jax.nn.one_hot(self.labels, 10)
    
    def get_batch(self):
        """Return a batch of data."""
        return (self.images, self.one_hot_labels)
    
    def get_output_and_loss(self, model, batch, train=True):
        """Compute model output and loss."""
        x, y = batch
        logits = model(x)
        loss = optax.softmax_cross_entropy(logits, y).mean()
        return logits, loss
    
    def run_benchmark(self) -> Dict[str, float]:
        """Run benchmark and return timing results."""
        results = {}
        
        # Warm up JIT compilation
        _ = self.jit_train_step(self.model, self.get_batch())
        
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


def run_comparison(full_precision=True, mixed_precision=True, num_iterations=100, batch_size=128):
    """Run comparison between full precision and mixed precision."""
    results = {}
    
    if full_precision:
        print("Running full precision benchmark...")
        config = BenchmarkConfig(
            enable_mixed_precision=False,
            precision_policy="default",
            batch_size=batch_size,
            num_iterations=num_iterations,
            seed=42
        )
        task = BenchmarkTask(config)
        results["full_precision"] = task.run_benchmark()
        print(f"Full precision: {results['full_precision']['time_per_step']:.4f} seconds/step")
    
    if mixed_precision:
        print("Running mixed precision benchmark...")
        config = BenchmarkConfig(
            enable_mixed_precision=True,
            precision_policy="mixed",
            loss_scaling="dynamic",
            loss_scale_value=2**15,
            batch_size=batch_size,
            num_iterations=num_iterations,
            seed=42
        )
        task = BenchmarkTask(config)
        results["mixed_precision"] = task.run_benchmark()
        print(f"Mixed precision: {results['mixed_precision']['time_per_step']:.4f} seconds/step")
    
    if full_precision and mixed_precision:
        speedup = results["full_precision"]["time_per_step"] / results["mixed_precision"]["time_per_step"]
        print(f"Speedup: {speedup:.2f}x")
        
        if results["full_precision"]["memory_usage_mb"] and results["mixed_precision"]["memory_usage_mb"]:
            memory_savings = results["full_precision"]["memory_usage_mb"] / results["mixed_precision"]["memory_usage_mb"]
            print(f"Memory savings: {memory_savings:.2f}x")
    
    return results


def plot_results(results):
    """Plot comparison results."""
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
    if results["full_precision"]["memory_usage_mb"] and results["mixed_precision"]["memory_usage_mb"]:
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


def benchmark_batch_sizes():
    """Run benchmarks for different batch sizes to show scaling benefits."""
    batch_sizes = [32, 64, 128, 256, 512]
    fp_times = []
    mp_times = []
    
    for bs in batch_sizes:
        print(f"\nBenchmarking batch size {bs}")
        results = run_comparison(num_iterations=20, batch_size=bs)
        fp_times.append(results["full_precision"]["time_per_step"])
        mp_times.append(results["mixed_precision"]["time_per_step"])
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, fp_times, 'o-', label="Full Precision")
    plt.plot(batch_sizes, mp_times, 'o-', label="Mixed Precision")
    plt.xlabel("Batch Size")
    plt.ylabel("Time per Step (seconds)")
    plt.title("Training Time vs Batch Size")
    plt.legend()
    plt.grid(True)
    plt.savefig("batch_size_scaling.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark mixed precision training")
    parser.add_argument('--no-fp', dest='full_precision', action='store_false', 
                        help='Skip full precision benchmark')
    parser.add_argument('--no-mp', dest='mixed_precision', action='store_false',
                        help='Skip mixed precision benchmark')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of training iterations')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--batch-test', action='store_true',
                        help='Run benchmark for various batch sizes')
    
    args = parser.parse_args()
    
    print("JAX devices:", jax.devices())
    print(f"Default backend: {jax.default_backend()}")
    
    if args.batch_test:
        benchmark_batch_sizes()
    else:
        results = run_comparison(
            full_precision=args.full_precision,
            mixed_precision=args.mixed_precision,
            num_iterations=args.iterations,
            batch_size=args.batch_size
        )
        
        if args.full_precision and args.mixed_precision:
            plot_results(results) 