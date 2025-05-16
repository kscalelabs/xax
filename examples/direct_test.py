"""Minimal test file for JAX."""

import jax
import jax.numpy as jnp
import numpy as np
import time

# Print JAX information
print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
print("Default backend:", jax.default_backend())

# Simple test of JAX operations
print("\nRunning basic JAX operations...")
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([4.0, 5.0, 6.0])
print("x + y =", x + y)
print("x * y =", x * y)

# Test JIT compilation
print("\nTesting JIT compilation...")
@jax.jit
def add(a, b):
    return a + b

result = add(x, y)
print("JIT add result:", result)

# Test different dtypes
for dtype in [jnp.float32, jnp.float16]:
    print(f"\nTesting {dtype}:")
    a = jnp.ones((10, 10), dtype=dtype)
    b = jnp.ones((10, 10), dtype=dtype)
    c = jnp.matmul(a, b)
    print(f"Matrix product shape: {c.shape}, dtype: {c.dtype}")

print("\nTest complete!") 