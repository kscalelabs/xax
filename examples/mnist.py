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


@dataclass
class Config(xax.Config):
    batch_size: int = xax.field(128, help="The size of a minibatch")
    learning_rate: float = xax.field(1e-3, help="The learning rate")
    hidden_dim: int = xax.field(512, help="Hidden layer dimension")
    num_hidden_layers: int = xax.field(2, help="Number of hidden layers")


class Model(eqx.Module):
    config: Config
    layers: list

    def __init__(self, config: Config, *, key: PRNGKeyArray) -> None:
        super().__init__()

        self.config = config

        # Input and output dimensions
        input_dim = 28 * 28
        output_dim = 10

        # Split the PRNG key for all layers
        keys = jax.random.split(key, self.config.num_hidden_layers + 1)

        # Build layers list
        self.layers = []
        current_dim = input_dim

        # Add hidden layers
        for i in range(self.config.num_hidden_layers):
            self.layers.extend([eqx.nn.Linear(current_dim, self.config.hidden_dim, key=keys[i]), jax.nn.relu])
            current_dim = self.config.hidden_dim

        # Add output layer
        self.layers.extend([eqx.nn.Linear(current_dim, output_dim, key=keys[-1]), jax.nn.log_softmax])

    def __call__(self, x: Array) -> Array:
        x = x.reshape(28 * 28)
        for layer in self.layers:
            x = layer(x)
        return x


class MnistClassification(xax.Task[Config]):
    def get_model(self, key: PRNGKeyArray) -> Model:
        return Model(self.config, key=key)

    def get_optimizer(self) -> optax.GradientTransformation:
        return optax.adam(self.config.learning_rate)

    def get_output(self, model: Model, batch: tuple[Array, Array]) -> Array:
        x, _ = batch
        return jax.vmap(model)(x)

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

    def get_data_iterator(self, phase: xax.Phase) -> Iterator:
        ds = MNIST(train=phase == "train", root_dir=xax.get_data_dir() / "mnist", dtype="float32")

        key = jax.random.PRNGKey(0)
        images, labels = jax.device_put((ds.images, ds.labels))

        while True:
            key, ind_key = jax.random.split(key)
            indices = jax.random.randint(ind_key, (self.config.batch_size,), 0, images.shape[0])
            yield images[indices], labels[indices]


if __name__ == "__main__":
    # python -m examples.mnist
    MnistClassification.launch(Config())
