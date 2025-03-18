"""Export JAX functions to TensorFlow SavedModel format."""

import logging
from pathlib import Path
from typing import Callable

import flax
import tensorflow as tf
from jax.experimental import jax2tf
from jaxtyping import Array, PyTree
from orbax.export import ExportManager, JaxModule, ServingConfig
import jax

logger = logging.getLogger(__name__)


def export(
    model: Callable,
    input_shape: tuple[int, ...],
    output_dir: str | Path = "export",
    batch_dim: int | None = None,
) -> None:
    tf_module = tf.Module()
    tf_module.infer = tf.function(
        jax2tf.convert(
            model,
            polymorphic_shapes=[
                "(b, ...)" if batch_dim is not None else "(None, ...)",
            ],
            # setting this to False will allow the model to run on platforms other than the one that exports the model
            # https://github.com/jax-ml/jax/blob/051687dc4c899df3d95c30b812ade401d8b31166/jax/experimental/jax2tf/README.md?plain=1#L1342
            # generally though I think native_serialization is recommended
            native_serialization=False,
            with_gradient=False,
        ),
        autograph=False,
        input_signature=[tf.TensorSpec([batch_dim] + list(input_shape), tf.float32)],
    )

    logger.info("Exporting SavedModel to %s", output_dir)
    tf.saved_model.save(
        tf_module,
        output_dir,
    )


def export_with_params(
    model: Callable,
    params: PyTree,
    input_shape: tuple[int, ...],
    output_dir: str | Path = "export",
    batch_dim: int | None = None,
) -> None:
    """Export a JAX function that takes parameters to TensorFlow SavedModel.

    Args:
        model: The JAX function to export. Should take parameters as first argument.
        params: The parameters to use for the model.
        input_shape: The shape of the input tensor, excluding batch dimension.
        output_dir: Directory to save the exported model.
        batch_dim: Optional batch dimension. If None, a polymorphic batch dimension is used.
    """
    pass


def export_flax(
    model: flax.linen.Module,
    params: PyTree,
    input_shape: tuple[int, ...],
    preprocessor: Callable | None = None,
    postprocessor: Callable | None = None,
    input_name: str = "inputs",
    output_name: str = "outputs",
    output_dir: str | Path = "export",
) -> None:
    jax_module = JaxModule(
        params, model.apply, trainable=False, input_polymorphic_shape="(b, ...)"
    )  # if you want to use a batch dimension

    # to avoid mapping sequences to ambiguous mappings
    if postprocessor is None:

        def postprocessor(x: PyTree) -> PyTree:
            return {output_name: x}

    export_manager = ExportManager(
        jax_module,
        [
            ServingConfig(
                "serving_default",
                input_signature=tf.TensorSpec([None] + list(input_shape), tf.float32, name=input_name),
                tf_preprocessor=preprocessor,
                tf_postprocessor=postprocessor,
            )
        ],
    )

    logger.info("Exporting model to %s", output_dir)
    export_manager.save(output_dir)
