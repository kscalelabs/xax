"""Tests for equinox utilities."""

from pathlib import Path
from typing import Callable, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from xax.nn.equinox import (
    DTYPE,
    ActivationFunction,
    MLPHyperParams,
    _infer_activation,
    export_eqx_mlp,
    load_eqx_mlp,
    make_eqx_mlp,
)


@pytest.fixture
def default_hyperparams() -> MLPHyperParams:
    """Create default hyperparameters for testing."""
    return {
        "in_size": 2,
        "out_size": 1,
        "width_size": 32,
        "depth": 2,
        "activation": "relu",
        "final_activation": "identity",
        "use_bias": True,
        "use_final_bias": True,
        "dtype": "float32",
    }


@pytest.fixture
def test_model(default_hyperparams: MLPHyperParams) -> eqx.nn.MLP:
    """Create a model for testing."""
    return make_eqx_mlp(default_hyperparams, jax.random.PRNGKey(42))


class TestInferActivation:
    """Tests for _infer_activation function."""

    @pytest.mark.parametrize(
        "activation_name,expected_function",
        [
            ("relu", jax.nn.relu),
            ("tanh", jax.nn.tanh),
            ("sigmoid", jax.nn.sigmoid),
            ("gelu", jax.nn.gelu),
        ],
    )
    def test_jax_activations(self, activation_name: str, expected_function: callable) -> None:
        """Test jax.nn activation functions."""
        activation_name = cast(ActivationFunction, activation_name)
        activation = _infer_activation(activation_name)
        assert activation is expected_function

    def test_identity_activation(self) -> None:
        """Test identity activation."""
        activation = _infer_activation("identity")
        assert activation(5.0) == 5.0
        assert activation(-3.0) == -3.0

    @pytest.mark.parametrize("invalid_activation", ["invalid_activation", "unknown", "not_a_function"])
    def test_invalid_activation(self, invalid_activation: str) -> None:
        """Test invalid activation raises ValueError."""
        with pytest.raises(ValueError, match="Activation function .* not found"):
            # We intentionally pass an invalid activation name to test error handling
            _infer_activation(cast(ActivationFunction, invalid_activation))


class TestMakeEqxMLP:
    """Tests for make_eqx_mlp function."""

    def test_make_mlp(self, default_hyperparams: MLPHyperParams) -> None:
        """Test making an MLP with default parameters."""
        model = make_eqx_mlp(default_hyperparams, jax.random.PRNGKey(42))

        # Test model properties match hyperparameters
        assert model.in_size == default_hyperparams["in_size"]
        assert model.out_size == default_hyperparams["out_size"]
        assert model.width_size == default_hyperparams["width_size"]
        assert model.depth == default_hyperparams["depth"]
        assert model.use_bias == default_hyperparams["use_bias"]
        assert model.use_final_bias == default_hyperparams["use_final_bias"]

        shape = (default_hyperparams["in_size"],) if default_hyperparams["in_size"] != "scalar" else ()
        x = jax.random.normal(jax.random.PRNGKey(42), shape=shape)
        y = model(x)
        assert y.shape == (default_hyperparams["out_size"],)

    @pytest.mark.parametrize("activation", ["tanh", "sigmoid", "relu", "gelu"])
    def test_make_mlp_different_activations(
        self, default_hyperparams: MLPHyperParams, activation: ActivationFunction
    ) -> None:
        """Test making MLPs with different activation functions."""
        params = default_hyperparams.copy()
        params["activation"] = activation
        model = make_eqx_mlp(params, jax.random.PRNGKey(42))
        assert model.activation is getattr(jax.nn, activation)

    @pytest.mark.parametrize(
        "hyperparams_config",
        [
            {
                "in_size": "scalar",
                "out_size": "scalar",
                "width_size": 16,
                "depth": 1,
                "activation": "relu",
                "final_activation": "identity",
                "use_bias": True,
                "use_final_bias": True,
                "dtype": "float32",
            },
            {
                "in_size": 5,
                "out_size": 3,
                "width_size": 64,
                "depth": 3,
                "activation": "tanh",
                "final_activation": "sigmoid",
                "use_bias": False,
                "use_final_bias": True,
                "dtype": "float32",
            },
        ],
    )
    def test_make_mlp_different_configs(self, hyperparams_config: MLPHyperParams) -> None:
        """Test making MLPs with different configurations."""
        model = make_eqx_mlp(hyperparams_config, jax.random.PRNGKey(42))

        # Test model properties match hyperparameters
        assert model.in_size == hyperparams_config["in_size"]
        assert model.out_size == hyperparams_config["out_size"]
        assert model.width_size == hyperparams_config["width_size"]
        assert model.depth == hyperparams_config["depth"]
        assert model.use_bias == hyperparams_config["use_bias"]
        assert model.use_final_bias == hyperparams_config["use_final_bias"]

        # Test forward pass
        shape = (hyperparams_config["in_size"],) if hyperparams_config["in_size"] != "scalar" else ()
        x = jnp.ones(shape)
        y = model(x)

        # Check output shape
        expected_shape = (hyperparams_config["out_size"],) if hyperparams_config["out_size"] != "scalar" else ()
        assert y.shape == expected_shape


class TestExportLoadEqxMLP:
    """Tests for export_eqx_mlp and load_eqx_mlp functions."""

    @pytest.mark.parametrize(
        "input_data",
        [
            lambda shape: jnp.ones(shape),
            lambda shape: jnp.zeros(shape),
            lambda shape: jax.random.normal(jax.random.PRNGKey(42), shape=shape),
        ],
        ids=["ones", "zeros", "random"],
    )
    def test_export_load_roundtrip(
        self, test_model: eqx.nn.MLP, tmpdir: Path, input_data: Callable[[tuple[int, ...]], jax.Array]
    ) -> None:
        """Test exporting and loading a model preserves properties."""
        model_path = tmpdir / "model.eqx"

        export_eqx_mlp(test_model, model_path)

        assert model_path.exists()

        loaded_model = load_eqx_mlp(model_path)

        # Check model properties are preserved
        assert loaded_model.in_size == test_model.in_size
        assert loaded_model.out_size == test_model.out_size
        assert loaded_model.width_size == test_model.width_size
        assert loaded_model.depth == test_model.depth
        assert loaded_model.use_bias == test_model.use_bias
        assert loaded_model.use_final_bias == test_model.use_final_bias

        shape = (test_model.in_size,) if test_model.in_size != "scalar" else ()
        x = input_data(shape)
        assert jnp.allclose(test_model(x), loaded_model(x))

    @pytest.mark.parametrize("filename", ["model_path_obj.eqx", "model_with_special_chars!@#.eqx", "model.tmp.eqx"])
    def test_export_with_path_object(self, test_model: eqx.nn.MLP, tmpdir: Path, filename: str) -> None:
        """Test export_eqx_mlp works with different Path objects."""
        from pathlib import Path

        model_path = Path(tmpdir) / filename

        export_eqx_mlp(test_model, model_path)

        loaded_model = load_eqx_mlp(model_path)

        assert loaded_model.in_size == test_model.in_size
        assert loaded_model.out_size == test_model.out_size

        x = jnp.ones((test_model.in_size,))
        assert jnp.allclose(test_model(x), loaded_model(x))

    @pytest.mark.parametrize("dtype", ["float32"])
    def test_export_load_preserves_dtype(self, default_hyperparams: MLPHyperParams, tmpdir: Path, dtype: str) -> None:
        """Test exporting and loading preserves the dtype."""
        params = default_hyperparams.copy()
        params["dtype"] = cast(DTYPE, dtype)
        model = make_eqx_mlp(params, jax.random.PRNGKey(42))
        model_path = tmpdir / f"model_{dtype}.eqx"

        export_eqx_mlp(model, model_path)

        loaded_model = load_eqx_mlp(model_path)

        shape = (model.in_size,) if model.in_size != "scalar" else ()
        x = jax.random.normal(jax.random.PRNGKey(42), shape=shape)

        original_output = model(x)
        loaded_output = loaded_model(x)

        assert jnp.allclose(original_output, loaded_output)

        assert loaded_model.in_size == model.in_size
        assert loaded_model.out_size == model.out_size
        assert loaded_model.width_size == model.width_size
        assert loaded_model.depth == model.depth
