"""Tests for the export module.

Note: The jax2tf.convert with native_serialization=False has been deprecated since July 2024,
but we're still using it for compatibility reasons. This will generate deprecation warnings
when running the tests.
"""

from pathlib import Path

import equinox as eqx
import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest
import tensorflow as tf
from jax.experimental import jax2tf
from jaxtyping import Array, Float

import xax


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


@pytest.mark.parametrize(
    "tf_input, expected_output",
    [([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[6.0], [15.0]]), ([[10.0, 20.0, 30.0]], [[60.0]])],
)
@pytest.mark.slow
def test_export_sum_model_parametric(
    tmp_path: Path,
    tf_input: list[list[float]],
    expected_output: list[list[float]],
) -> None:
    model = SumModel()
    xax.export(model=model.__call__, input_shape=(3,), output_dir=tmp_path, batch_size=len(tf_input))
    loaded_model = tf.saved_model.load(tmp_path)
    result = loaded_model.infer(tf.constant(tf_input, dtype=tf.float32))
    tf.debugging.assert_near(result, tf.constant(expected_output, dtype=tf.float32), rtol=1e-5)


@pytest.mark.parametrize(
    "tf_input, expected_output",
    [([[1.0, 2.0, 3.0]], [[3.0, 6.0, 9.0]]), ([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], [[3.0, 3.0, 3.0], [6.0, 6.0, 6.0]])],
)
@pytest.mark.slow
def test_export_multiply_model_parametric(
    tmp_path: Path,
    tf_input: list[list[float]],
    expected_output: list[list[float]],
) -> None:
    model = MultiplyModel(factor=3.0)
    xax.export(model=model.__call__, input_shape=(3,), output_dir=tmp_path, batch_size=len(tf_input))
    loaded_model = tf.saved_model.load(tmp_path)
    result = loaded_model.infer(tf.constant(tf_input, dtype=tf.float32))
    tf.debugging.assert_near(result, tf.constant(expected_output, dtype=tf.float32), rtol=1e-5)


@pytest.mark.parametrize("batch_size, rand_key", [(1, 42), (2, 43), (3, 44), (5, 45)])
@pytest.mark.slow
def test_export_mlp_model_fixed(tmp_path: Path, batch_size: int, rand_key: int) -> None:
    key = jax.random.PRNGKey(rand_key)

    in_features = jax.random.randint(key, (1,), 1, 10).item()
    hidden_features = jax.random.randint(key, (1,), 1, 10).item()
    out_features = jax.random.randint(key, (1,), 1, 10).item()
    model = MLP(in_features, hidden_features, out_features, key)
    test_input = jax.random.uniform(key, (in_features,)).tolist()
    jax_out = model(jnp.array(test_input))

    batched_expected = tf.convert_to_tensor(jnp.stack([jax_out] * batch_size).tolist(), dtype=tf.float32)

    def batched_model(x: Array) -> Array:
        return jax.vmap(model)(x)

    xax.export(
        model=batched_model,
        input_shape=(in_features,),
        output_dir=tmp_path,
        batch_size=batch_size,
    )
    loaded_model = tf.saved_model.load(tmp_path)

    tf_input = tf.constant([test_input] * batch_size, dtype=tf.float32)
    tf_result = loaded_model.infer(tf_input)
    tf.debugging.assert_near(tf_result, batched_expected, rtol=1e-5)


@pytest.mark.parametrize(
    "tf_input, expected_output",
    [
        # Batch size 1
        ([[1.0, 2.0, 3.0]], [[2.0, 4.0, 6.0]]),
        # Batch size 3
        (
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[2.0, 4.0, 6.0], [8.0, 10.0, 12.0], [14.0, 16.0, 18.0]],
        ),
        # Batch size 5
        (
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0]],
            [[2.0, 4.0, 6.0], [8.0, 10.0, 12.0], [14.0, 16.0, 18.0], [20.0, 22.0, 24.0], [26.0, 28.0, 30.0]],
        ),
    ],
)
@pytest.mark.slow
def test_export_multiply_model_poly(
    tmp_path: Path,
    tf_input: list[list[float]],
    expected_output: list[list[float]],
) -> None:
    model = MultiplyModel(factor=2.0)
    # Export with polymorphic batch size (batch_size=None)
    xax.export(model=jax.vmap(model), input_shape=(3,), output_dir=tmp_path, batch_size=None)
    loaded_model = tf.saved_model.load(tmp_path)
    result = loaded_model.infer(tf.constant(tf_input, dtype=tf.float32))
    tf.debugging.assert_near(result, tf.constant(expected_output, dtype=tf.float32), rtol=1e-5)


@pytest.mark.parametrize(
    "tf_input",
    [
        # Batch size 1
        [[0.5, 1.0, 1.5, 2.0]],
        # Batch size 3 (all same input)
        [[0.5, 1.0, 1.5, 2.0]] * 3,
        # Batch size 5 with varied inputs
        [
            [0.5, 1.0, 1.5, 2.0],
            [1.0, 1.5, 2.0, 2.5],
            [0.1, 0.2, 0.3, 0.4],
            [2.0, 2.0, 2.0, 2.0],
            [-0.5, -1.0, -1.5, -2.0],
        ],
    ],
)
@pytest.mark.slow
def test_export_mlp_model_poly(tmp_path: Path, tf_input: list[list[float]]) -> None:
    in_features = 4
    hidden_features = 8
    out_features = 2
    key = jax.random.PRNGKey(42)
    model = MLP(in_features, hidden_features, out_features, key)

    def batched_model(x: Array) -> Array:
        return jax.vmap(model)(x)

    xax.export(model=batched_model, input_shape=(in_features,), output_dir=tmp_path, batch_size=None)

    loaded_model = tf.saved_model.load(tmp_path)
    tf_input_tensor = tf.constant(tf_input, dtype=tf.float32)

    # Compute expected output using jax.vmap
    jax_output = jax.vmap(model)(jnp.array(tf_input))
    expected = tf.convert_to_tensor(jax_output.tolist(), dtype=tf.float32)
    tf_result = loaded_model.infer(tf_input_tensor)
    tf.debugging.assert_near(tf_result, expected, rtol=1e-5)


class FlaxMLP(nn.Module):
    """Flax MLP for testing export."""

    in_features: int
    hidden_features: list[int]
    out_features: int

    @nn.compact
    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        """Forward pass through the MLP with multiple hidden layers."""
        x = nn.Dense(features=self.hidden_features[0])(x)
        x = nn.relu(x)

        for features in self.hidden_features[1:]:
            x = nn.Dense(features=features)(x)
            x = nn.relu(x)

        x = nn.Dense(features=self.out_features)(x)
        return x


@pytest.mark.parametrize(
    "in_features, hidden_features, out_features, batch_size",
    [
        (784, [128, 64], 10, 2),
        (100, [50], 5, 1),
        (3, [10, 10, 10], 2, 5),
        (512, [256, 128, 64], 32, 3),
    ],
)
@pytest.mark.slow
def test_export_flax_mlp(
    tmp_path: Path,
    in_features: int,
    hidden_features: list[int],
    out_features: int,
    batch_size: int,
) -> None:
    """Test exporting a FlaxMLP model with different configurations."""
    model = FlaxMLP(in_features, hidden_features, out_features)

    # Initialize the model parameters (very important for flax models)
    key = jax.random.PRNGKey(42)
    params = model.init(key, jnp.ones((1, in_features)))

    test_input = jax.random.normal(key, (batch_size, in_features))

    expected_output = model.apply(params, test_input)

    xax.export_flax(model=model, params=params, input_shape=(in_features,), output_dir=tmp_path)

    loaded_model = tf.saved_model.load(tmp_path)

    tf_input = tf.convert_to_tensor(test_input.tolist(), dtype=tf.float32)
    result = loaded_model.signatures["serving_default"](inputs=tf_input)["outputs"]

    tf.debugging.assert_near(result, expected_output, rtol=1e-5)


@pytest.mark.slow
def test_export_with_params_basic_multiply(tmp_path: Path) -> None:
    """Test exporting a basic multiply function with parameters."""
    factor = 3.0

    def multiply_fn(params: Float[Array, "..."], x: Float[Array, "..."]) -> Float[Array, "..."]:
        return x * params[0]

    # Create a parameter array with the factor
    params = jnp.array([factor])

    tf_input = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)
    expected_output = tf_input * factor

    xax.export_with_params(
        model=multiply_fn,
        params=params,
        input_shape=(3,),
        output_dir=tmp_path,
        batch_dim=len(tf_input),
    )

    loaded_model = tf.saved_model.load(tmp_path)

    result = loaded_model.infer(tf.convert_to_tensor(tf_input, dtype=tf.float32))

    tf.debugging.assert_near(result, expected_output, rtol=1e-5)


@pytest.mark.parametrize(
    "tf_input, expected_factor",
    [
        ([[1.0, 2.0, 3.0]], 3.0),
        ([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], 3.0),
        ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], 3.0),
    ],
)
@pytest.mark.slow
def test_export_with_params_poly_batch(tmp_path: Path, tf_input: list[list[float]], expected_factor: float) -> None:
    """Test exporting a model with polymorphic batch dimensions using export_with_params."""
    factor = jnp.array(expected_factor)

    def multiply_model(x: Float[Array, "..."]) -> Float[Array, "..."]:
        return x * factor

    converted_model = jax2tf.convert(multiply_model, polymorphic_shapes=["(b, ...)"])

    tf_module = tf.Module()

    # Define the TF function that accepts polymorphic batch dimensions
    @tf.function(input_signature=[tf.TensorSpec([None, 3], tf.float32)])
    def infer(inputs: tf.Tensor) -> tf.Tensor:
        return converted_model(inputs)

    tf_module.infer = infer  # type: ignore [attr-defined]

    tf.saved_model.save(tf_module, tmp_path)
    loaded_model = tf.saved_model.load(tmp_path)
    expected_output = [[x * expected_factor for x in batch] for batch in tf_input]
    result = loaded_model.infer(tf.constant(tf_input, dtype=tf.float32))
    tf.debugging.assert_near(result, tf.constant(expected_output, dtype=tf.float32), rtol=1e-5)
