"""Equinox utilities."""

import json
import logging
from pathlib import Path
from typing import Callable, Literal, TypedDict, cast

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray

logger = logging.getLogger(__name__)

ActivationFunction = Literal[
    "relu",
    "tanh",
    "celu",
    "elu",
    "gelu",
    "glu",
    "hard_sigmoid",
    "hard_silu",
    "hard_swish",
    "hard_tanh",
    "leaky_relu",
    "log_sigmoid",
    "log_softmax",
    "logsumexp",
    "relu6",
    "selu",
    "sigmoid",
    "soft_sign",
    "softmax",
    "softplus",
    "sparse_plus",
    "sparse_sigmoid",
    "silu",
    "swish",
    "squareplus",
    "mish",
    "identity",
]

DTYPE = Literal["float32", "float64"]

DTYPE_MAP: dict[DTYPE, jax.numpy.dtype] = {
    "float32": jax.numpy.float32,
    "float64": jax.numpy.float64,
}


class MLPHyperParams(TypedDict):
    """Hyperparameters of an Equinox MLP."""

    in_size: int | Literal["scalar"]
    out_size: int | Literal["scalar"]
    width_size: int
    depth: int
    activation: ActivationFunction
    final_activation: ActivationFunction
    use_bias: bool
    use_final_bias: bool
    dtype: DTYPE


def _infer_activation(activation: ActivationFunction) -> Callable:
    if activation == "identity":
        return lambda x: x
    try:
        return getattr(jax.nn, activation)
    except AttributeError:
        raise ValueError(f"Activation function `{activation}` not found in `jax.nn`")


def make_eqx_mlp(hyperparams: MLPHyperParams, *, key: PRNGKeyArray) -> eqx.nn.MLP:
    """Create an Equinox MLP from a set of hyperparameters.

    Args:
        hyperparams: The hyperparameters of the MLP.
        key: The PRNG key to use for the MLP.
    """
    activation = _infer_activation(hyperparams["activation"])
    final_activation = _infer_activation(hyperparams["final_activation"])
    dtype = DTYPE_MAP[hyperparams["dtype"]]

    return eqx.nn.MLP(
        in_size=hyperparams["in_size"],
        out_size=hyperparams["out_size"],
        width_size=hyperparams["width_size"],
        depth=hyperparams["depth"],
        activation=activation,
        final_activation=final_activation,
        use_bias=hyperparams["use_bias"],
        use_final_bias=hyperparams["use_final_bias"],
        dtype=dtype,
        key=key,
    )


def export_eqx_mlp(
    model: eqx.nn.MLP,
    output_path: str | Path,
    dtype: jax.numpy.dtype = eqx._misc.default_floating_dtype(),
) -> None:
    """Serialize an Equinox MLP to a .eqx file.

    Args:
        model: The JAX MLP to export.
        output_path: The path to save the exported model.
        dtype: The dtype of the model.
    """
    activation = model.activation.__name__
    final_activation = model.final_activation.__name__

    if final_activation == "<lambda>":
        logger.warning("Final activation is a lambda function. Assuming identity.")
        final_activation = "identity"

    # cast strings to ActivationFunction for type checking
    activation = cast(ActivationFunction, activation)
    final_activation = cast(ActivationFunction, final_activation)

    if dtype not in DTYPE_MAP.values():
        raise ValueError(f"Invalid dtype: {dtype}. Must be one of {DTYPE_MAP.values()}")

    dtype = {v: k for k, v in DTYPE_MAP.items()}[dtype]

    hyperparams: MLPHyperParams = {
        "in_size": model.in_size,
        "out_size": model.out_size,
        "width_size": model.width_size,
        "depth": model.depth,
        "activation": activation,
        "final_activation": final_activation,
        "use_bias": model.use_bias,
        "use_final_bias": model.use_final_bias,
        "dtype": dtype,
    }

    with open(output_path, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode(encoding="utf-8"))
        eqx.tree_serialise_leaves(f, model)


def save_eqx(
    model: eqx.Module,
    output_path: str | Path,
) -> None:
    """Serialize an Equinox module to a .eqx file.

    Args:
        model: The Equinox module to export.
        output_path: The path to save the exported model.
    """
    with open(output_path, "wb") as f:
        eqx.tree_serialise_leaves(f, model)


def load_eqx(
    model: eqx.Module,
    eqx_file: str | Path,
) -> eqx.Module:
    """Deserialize an Equinox module from a .eqx file.

    Args:
        model: The Equinox module to load into.
        eqx_file: The path to the .eqx file to load.
    """
    with open(eqx_file, "rb") as f:
        return eqx.tree_deserialise_leaves(f, model)


def load_eqx_mlp(
    eqx_file: str | Path,
) -> eqx.nn.MLP:
    with open(eqx_file, "rb") as f:
        hyperparams = json.loads(f.readline().decode(encoding="utf-8"))
        model = make_eqx_mlp(hyperparams=hyperparams, key=jax.random.PRNGKey(0))
        return eqx.tree_deserialise_leaves(f, model)
