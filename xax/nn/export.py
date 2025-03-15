from typing import Any, Callable, Generic, ParamSpec, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from jaxtyping import PyTree
from orbax.export import ExportManager, JaxModule, ServingConfig

P = ParamSpec("P")  # function parameters
R = TypeVar("R")  # function return type


class ExportableModel(Generic[P, R]):
    def __init__(
        self,
        model: Callable[P, R],
        params: PyTree,
        preprocess_fn: Callable[[P], P] | None = None,
        postprocess_fn: Callable[[R], dict[str, R]] | None = None,
    ) -> None:
        """Initializes a wrapper around a JAX model with parameters.

        Args:
            model: The JAX model to wrap. The first argument should be the parameters.
            params: The parameters of the model.
            export_dir: The directory to export the model to.
            preprocess_fn: A tf function to preprocess the input arguments to the model.
            postprocess_fn: A tf function to postprocess the output of the model.
        """
        self.model = model
        self.params = params

        if postprocess_fn is None:
            def postprocess_fn(x: R) -> dict[str, R]:
                return {"output": x}

        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn

        self.jax_module = JaxModule(params, model)
        self.export_manager = ExportManager(
            self.jax_module,
            [
                ServingConfig(
                    'serving_default',
                    input_signature=P.args + P.kwargs,
                    preprocess_fn=self.preprocess_fn,
                    postprocess_fn=self.postprocess_fn,
                )
            ],
        )

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Call the model with the given parameters and arguments."""
        return self.model(self.params, *args, **kwargs)

    def export(self, export_dir: str) -> None:
        """Export the model to the given directory."""
        self.export_manager.save(export_dir)

"""
Example usage:

model = flax.linen.Dense(10)
params = model.init(rng, x)

exportable_model = ExportableModel(model, params, "export_dir")

"""
