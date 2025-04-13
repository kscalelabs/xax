"""Export JAX functions to TensorFlow SavedModel format."""

import logging
from pathlib import Path
from typing import Callable

import jax
from jaxtyping import Array, PyTree

try:
    import flax
    import tensorflow as tf
    from jax.experimental import jax2tf
    from orbax.export import ExportManager, JaxModule, ServingConfig
except ImportError as e:
    raise ImportError(
        "In order to export models, please install Xax with exportable dependencies, "
        "using 'xax[exportable]` to install the required dependencies."
    ) from e

logger = logging.getLogger(__name__)


def _run_infer(tf_module: tf.Module, input_shapes: list[tuple[int, ...]], batch_size: int | None) -> tf.Tensor:
    """Warm up the model by running it once."""
    if batch_size is not None:
        test_inputs = [
            jax.random.normal(jax.random.PRNGKey(42), (batch_size, *input_shape)) for input_shape in input_shapes
        ]
    else:
        test_inputs = [jax.random.normal(jax.random.PRNGKey(42), (1, *input_shape)) for input_shape in input_shapes]
    if not hasattr(tf_module, "infer"):
        raise ValueError("Model does not have an infer method")
    return tf_module.infer(*test_inputs)


def export(
    model: Callable,
    input_shapes: list[tuple[int, ...]],
    output_dir: str | Path = "export",
    batch_size: int | None = None,
) -> None:
    """Export a JAX function to TensorFlow SavedModel.

    Note: Tensorflow GraphDef can't be larger than 2GB - https://github.com/tensorflow/tensorflow/issues/51870
    You can avoid this by saving model parameters as non-constants.

    Args:
        model: The JAX function to export.
        input_shapes: The shape of the input tensors, excluding batch dimension.
        output_dir: Directory to save the exported model.
        batch_size: Optional batch dimension. If None, a polymorphic batch dimension is used.
    """
    tf_module = tf.Module()
    # Create a polymorphic shape specification for each input
    poly_spec = "(b, ...)" if batch_size is not None else "(None, ...)"
    polymorphic_shapes = [poly_spec] * len(input_shapes)
    tf_module.infer = tf.function(  # type: ignore [attr-defined]
        jax2tf.convert(
            model,
            polymorphic_shapes=polymorphic_shapes,
            # setting this to False will allow the model to run on platforms other than the one that exports the model
            # https://github.com/jax-ml/jax/blob/051687dc4c899df3d95c30b812ade401d8b31166/jax/experimental/jax2tf/README.md?plain=1#L1342
            # generally though I think native_serialization is recommended
            native_serialization=False,
            with_gradient=False,
        ),
        autograph=False,
        input_signature=[tf.TensorSpec([batch_size] + list(input_shape), tf.float32) for input_shape in input_shapes],
    )

    # warm up the model
    _run_infer(tf_module, input_shapes, batch_size)

    logger.info("Exporting SavedModel to %s", output_dir)
    tf.saved_model.save(
        tf_module,
        output_dir,
    )


def export_with_params(
    model: Callable,
    params: PyTree,
    input_shapes: list[tuple[int, ...]],
    output_dir: str | Path = "export",
    batch_dim: int | None = None,
) -> None:
    """Export a JAX function that takes parameters to TensorFlow SavedModel.

    Args:
        model: The JAX function to export. Should take parameters as first argument.
        params: The parameters to use for the model.
        input_shapes: The shape of the input tensors, excluding batch dimension.
        output_dir: Directory to save the exported model.
        batch_dim: Optional batch dimension. If None, a polymorphic batch dimension is used.
    """
    param_vars = tf.nest.map_structure(tf.Variable, params)

    converted_model = jax2tf.convert(model)

    def model_fn(*inputs: PyTree) -> Array:
        return converted_model(param_vars, *inputs)

    tf_module = tf.Module()
    tf_module._variables = tf.nest.flatten(param_vars)  # type: ignore [attr-defined]
    tf_module.infer = tf.function(  # type: ignore [attr-defined]
        model_fn,
        jit_compile=True,
        autograph=False,
        input_signature=[tf.TensorSpec([batch_dim] + list(input_shape), tf.float32) for input_shape in input_shapes],
    )

    # warm up the model
    _run_infer(tf_module, input_shapes, batch_dim)

    logger.info("Exporting SavedModel to %s", output_dir)
    tf.saved_model.save(tf_module, output_dir)


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
                input_signature=[tf.TensorSpec([None] + list(input_shape), tf.float32, name=input_name)],
                tf_preprocessor=preprocessor,
                tf_postprocessor=postprocessor,
            )
        ],
    )

    logger.info("Exporting model to %s", output_dir)
    export_manager.save(output_dir)
