"""Tests for the export module.

Note: The jax2tf.convert with native_serialization=False has been deprecated since July 2024,
but we're still using it for compatibility reasons. This will generate deprecation warnings
when running the tests.
"""

import os
import shutil
import tempfile
from typing import Generator

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
import tensorflow as tf
from jaxtyping import Array, Float

from xax.nn.export import export


class SumModel(eqx.Module):
    """Sum module for testing export."""

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return jnp.sum(x, axis=-1, keepdims=True)


class MultiplyModel(eqx.Module):
    """Multiply module for testing export."""

    factor: float

    def __init__(self, factor: float = 2.0) -> None:
        self.factor = factor

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return x * self.factor


class MLP(eqx.Module):
    """MLP for testing export."""

    layers: list[eqx.nn.Linear]

    def __init__(self, in_features: int, hidden_features: int, out_features: int, key: jax.Array) -> None:
        key1, key2 = jax.random.split(key)
        self.layers = [
            eqx.nn.Linear(in_features, hidden_features, key=key1),
            eqx.nn.Linear(hidden_features, out_features, key=key2),
        ]

    def __call__(self, x: Array) -> Array:
        """Forward pass through the MLP for a single example (no batch dimension)."""
        x = self.layers[0](x)
        x = jax.nn.relu(x)
        return self.layers[1](x)


@pytest.fixture
def temp_export_dir() -> Generator[str, None, None]:
    """Create a temporary directory for exports."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_export_sum_model(temp_export_dir: str) -> None:
    """Test exporting a simple sum model."""
    # Define the model and input shape
    model = SumModel()
    input_shape = (3,)  # 3 dimensions as input

    test_input_jax = jnp.array([1.0, 2.0, 3.0])
    jax_result = model(test_input_jax)
    assert jnp.allclose(jax_result, jnp.array([6.0]))

    # Export the model with explicit batch size
    export(
        model=model.__call__,
        input_shape=input_shape,
        output_dir=temp_export_dir,
        batch_dim=2,  # Specify a concrete batch dimension
    )

    assert os.path.exists(os.path.join(temp_export_dir, "saved_model.pb"))

    loaded_model = tf.saved_model.load(temp_export_dir)

    # Test with a batch of inputs
    test_input = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=tf.float32)
    result = loaded_model.infer(test_input)

    # Expected results: [1+2+3=6, 4+5+6=15]
    expected = tf.constant([[6.0], [15.0]], dtype=tf.float32)
    tf.debugging.assert_near(result, expected, rtol=1e-5)


def test_export_multiply_model(temp_export_dir: str) -> None:
    """Test exporting a model with parameters."""
    model = MultiplyModel(factor=3.0)
    input_shape = (3,)

    test_input_jax = jnp.array([1.0, 2.0, 3.0])
    jax_result = model(test_input_jax)
    assert jnp.allclose(jax_result, jnp.array([3.0, 6.0, 9.0]))

    # Export the model with explicit batch size
    export(
        model=model.__call__,
        input_shape=input_shape,
        output_dir=temp_export_dir,
        batch_dim=1,
    )

    assert os.path.exists(os.path.join(temp_export_dir, "saved_model.pb"))

    loaded_model = tf.saved_model.load(temp_export_dir)

    # Test with a single batch
    test_input = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)
    result = loaded_model.infer(test_input)

    # Expected results: [1*3, 2*3, 3*3] = [3, 6, 9]
    expected = tf.constant([[3.0, 6.0, 9.0]], dtype=tf.float32)
    tf.debugging.assert_near(result, expected, rtol=1e-5)


def test_export_mlp_model(temp_export_dir: str) -> None:
    """Test exporting a 2-layer MLP model."""
    # Define the model architecture
    in_features = 4
    hidden_features = 8
    out_features = 2

    # Initialize the model with a fixed random key for reproducibility
    key = jax.random.PRNGKey(42)
    model = MLP(in_features, hidden_features, out_features, key)

    # Create test input data - single example, no batch dimension
    test_input_jax = jnp.array([0.5, 1.0, 1.5, 2.0])

    # Compute the expected output with JAX model for a single example
    jax_result = model(test_input_jax)

    # Create a batched version of the model function for export
    def batched_model(x: Array) -> Array:
        return jax.vmap(model)(x)

    export(
        model=batched_model,
        input_shape=(in_features,),
        output_dir=temp_export_dir,
        batch_dim=2,  # Support for batching with 2 examples
    )

    assert os.path.exists(os.path.join(temp_export_dir, "saved_model.pb"))

    loaded_model = tf.saved_model.load(temp_export_dir)

    # Create a batched input for the TF model
    batched_input = tf.constant([[0.5, 1.0, 1.5, 2.0], [0.5, 1.0, 1.5, 2.0]], dtype=tf.float32)

    tf_result = loaded_model.infer(batched_input)

    # Create a batched version of the original output for comparison
    batched_jax_result = jnp.stack([jax_result, jax_result])

    # Convert to tensorflow tensor for comparison
    expected_result = tf.convert_to_tensor(batched_jax_result, dtype=tf.float32)

    # Compare the TF model output with the expected JAX output
    tf.debugging.assert_near(tf_result, expected_result, rtol=1e-5)
