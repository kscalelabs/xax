"""Trains a simple convolutional neural network on the MNIST dataset.

Run this example with `python -m examples.mnist`.
"""

from dataclasses import dataclass
from typing import Iterator

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from dpshdl.impl.mnist import MNIST
from jaxtyping import Array, PRNGKeyArray

import xax


def cross_entropy(y: Array, pred_y: Array) -> Array:
    pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
    return -jnp.mean(pred_y)


@jax.tree_util.register_dataclass
@dataclass
class Config(xax.Config):
    batch_size: int = xax.field(256, help="The size of a minibatch")


class Model(eqx.Module):
    layers: list

    def __init__(self, rng_key: PRNGKeyArray) -> None:
        super().__init__()

        # Split the PRNG key into four keys for the four layers.
        key1, key2, key3, key4 = jax.random.split(rng_key, 4)

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

    def __call__(self, x: Array) -> Array:
        x = x[None]  # Add channel dimension.
        for layer in self.layers:
            x = layer(x)
        return x


class MnistClassification(xax.Task[Config]):
    def get_model(self, key: PRNGKeyArray) -> Model:
        return Model(key)

    def get_optimizer(self) -> optax.GradientTransformation:
        return optax.adam(1e-3)

    def get_output(self, model: Model, batch: tuple[Array, Array]) -> Array:
        x, _ = batch
        y_hat = jax.vmap(model)(x)
        return y_hat

    def compute_loss(self, model: Model, batch: tuple[Array, Array], output: Array) -> Array:
        (_, y), yhat = batch, output
        return cross_entropy(y, yhat)

    def log_train_step(self, model: Model, batch: tuple[Array, Array], output: Array, state: xax.State) -> None:
        (_, y), yhat = batch, output.argmax(axis=1)
        self.logger.log_scalar("acc", (yhat == y).astype(float).mean())

    def log_valid_step(self, model: Model, batch: tuple[Array, Array], output: Array, state: xax.State) -> None:
        max_images = 16
        batch = jax.tree_map(lambda x: jax.device_get(x[:max_images]), batch)
        (x, y), yhat = batch, output.argmax(axis=1)
        labels = [f"pred: {p}\ntrue: {t}" for p, t in zip(yhat[:max_images], y[:max_images])]
        self.logger.log_labeled_images("predictions", (x, labels), max_images=max_images)

    def get_iterator(self, phase: xax.Phase) -> Iterator[tuple[Array, Array]]:
        ds = MNIST(train=phase == "train", root_dir=xax.get_data_dir() / "mnist", dtype="float32")

        key = jax.random.PRNGKey(0)
        images, labels = jax.device_put((ds.images, ds.labels))

        while True:
            key, ind_key = jax.random.split(key)
            indices = jax.random.randint(ind_key, (self.config.batch_size,), 0, images.shape[0])
            yield (images[indices], labels[indices])


if __name__ == "__main__":
    # python -m examples.mnist
    MnistClassification.launch(Config())
