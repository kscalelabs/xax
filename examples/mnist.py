"""Trains a simple convolutional neural network on the MNIST dataset.

Run this example with `python -m examples.mnist`.
"""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from dpshdl.impl.mnist import MNIST
from jaxtyping import Array, Float, Int

import xax

Batch = tuple[Float[Array, "1 28 28"], Int[Array, "10"]]
Output = Float[Array, "10"]


def cross_entropy(y: Int[Array, " batch"], pred_y: Float[Array, "batch 10"]) -> Float[Array, ""]:
    pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
    return -jnp.mean(pred_y)


@dataclass
class Config(xax.Config):
    in_dim: int = xax.field(1, help="Number of input dimensions")


class MnistClassification(xax.Task[Config]):
    layers: list

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        # Split the PRNG key into four keys for the four layers.
        key1, key2, key3, key4 = jax.random.split(self.prng_key, 4)

        self.layers = [
            eqx.nn.Conv2d(1, 3, kernel_size=4, key=key1),
            eqx.nn.MaxPool2d(kernel_size=2),
            jax.nn.relu,
            jnp.ravel,
            eqx.nn.Linear(1728, 512, key=key2),
            jax.nn.sigmoid,
            eqx.nn.Linear(512, 64, key=key3),
            jax.nn.relu,
            eqx.nn.Linear(64, 10, key=key4),
            jax.nn.log_softmax,
        ]

    def get_optimizer(self) -> optax.GradientTransformation:
        return optax.adam(1e-3)

    def forward(self, x: Float[Array, "1 28 28"]) -> Output:
        for layer in self.layers:
            x = layer(x)
        return x

    def __call__(self, batch: Batch) -> Output:
        x, y = batch
        pred_y = jax.vmap(self.forward)(x[:, None])
        return cross_entropy(y, pred_y)

    def get_dataset(self, phase: xax.Phase) -> MNIST:
        return MNIST(
            train=phase == "train",
            root_dir=xax.get_data_dir() / "mnist",
            dtype="float32",
        )


if __name__ == "__main__":
    # python -m examples.mnist
    MnistClassification.launch(Config(batch_size=16, num_dataloader_workers=0))
