"""Simple benchmark to verify mixed precision implementation."""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import time
from dataclasses import dataclass

@dataclass
class Config:
    """A simple configuration for the benchmark."""
    enable_mixed_precision: bool = False
    seed: int = 42


class SimpleLinear(eqx.Module):
    """A simple linear model for testing."""
    weight: jnp.ndarray
    bias: jnp.ndarray
    
    def __init__(self, in_dim, out_dim, key):
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.normal(wkey, (in_dim, out_dim))
        self.bias = jax.random.normal(bkey, (out_dim,))
    
    def __call__(self, x):
        return x @ self.weight + self.bias


def main():
    print("JAX devices:", jax.devices())
    
    # Create a simple model
    key = jax.random.PRNGKey(42)
    model = SimpleLinear(784, 10, key)
    
    # Generate some fake data
    batch_size = 32
    x = jax.random.normal(key, (batch_size, 784))
    y = jax.random.normal(key, (batch_size, 10))
    
    # Define a simple loss function
    def loss_fn(model, x, y):
        pred = jax.vmap(model)(x)
        return jnp.mean((pred - y)**2)
    
    # Create optimizer
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # Define a training step
    @jax.jit
    def train_step(model, x, y, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(model, x, y)
        updates, new_opt_state = optimizer.update(grads, opt_state, model)
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, loss
    
    # Run benchmark
    print("Running training benchmark...")
    n_steps = 100
    
    # Warm up JIT
    model, opt_state, _ = train_step(model, x, y, opt_state)
    
    # Measure time
    start_time = time.time()
    for _ in range(n_steps):
        model, opt_state, loss = train_step(model, x, y, opt_state)
    end_time = time.time()
    
    print(f"Total time: {end_time - start_time:.4f} seconds")
    print(f"Time per step: {(end_time - start_time) / n_steps:.4f} seconds")
    print(f"Final loss: {loss}")
    print(f"Model dtype: {model.weight.dtype}")
    
    # Print memory usage if available
    try:
        mem_usage = jax.live_arrays()
        total_bytes = sum(x.nbytes for x in mem_usage)
        print(f"Memory usage: {total_bytes / (1024 * 1024):.2f} MB")
    except:
        print("Memory usage information not available")


if __name__ == "__main__":
    main() 