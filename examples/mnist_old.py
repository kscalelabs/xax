"""MNIST example in Jax."""

import itertools
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from dpshdl.dataloader import Dataloader
from dpshdl.impl.mnist import MNIST
from jaxtyping import Array, Float, Int

import xax


class Model(eqx.Module):
    layers: list

    def __init__(self, prng_key: Array) -> None:
        super().__init__()

        # Split the PRNG key into four keys for the four layers.
        key1, key2, key3, key4 = jax.random.split(prng_key, 4)

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

    def __call__(self, x: Float[Array, "1 28 28"]) -> Float[Array, "10"]:
        for layer in self.layers:
            x = layer(x)
        return x


def main() -> None:
    learning_rate = 0.001
    batch_size = 128

    # Split the PRNG key into four keys for the four layers.
    prng_key = jax.random.PRNGKey(1337)
    model = Model(prng_key)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    def cross_entropy(y: Int[Array, " batch"], pred_y: Float[Array, "batch 10"]) -> Float[Array, ""]:
        pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
        return -jnp.mean(pred_y)

    def loss(model: Model, batch: tuple[Float[Array, "batch 1 28 28"], Int[Array, " batch"]]) -> Float[Array, ""]:
        inputs, targets = batch
        preds = jax.vmap(model)(inputs)
        return cross_entropy(targets, preds)

    def accuracy(model: Model, batch: tuple[Float[Array, "batch 1 28 28"], Int[Array, " batch"]]) -> Float[Array, ""]:
        inputs, targets = batch
        preds = jax.vmap(model)(inputs)
        return jnp.mean(jnp.argmax(preds, axis=1) == targets)

    @eqx.filter_jit
    def update(
        model: Model,
        opt_state: optax.OptState,
        batch: tuple[Float[Array, "batch 1 28 28"], Int[Array, " batch"]],
    ) -> tuple[Model, optax.OptState, Float[Array, ""]]:
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, batch)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, new_opt_state, loss_value

    print("\nStarting training...")
    root_dir = xax.get_data_dir() / "mnist"
    with (
        Dataloader(MNIST(True, root_dir, dtype="float32"), batch_size=batch_size) as train_dl,
        Dataloader(MNIST(False, root_dir, dtype="float32"), num_workers=1, batch_size=batch_size) as test_dl,
    ):
        print("Initialized dataloaders")
        for epoch in range(10):
            start_time = time.time()
            n = 0
            for batch in itertools.islice(train_dl, 100):
                images, labels = batch
                images = images[:, None]
                n += images.shape[0]
                model, opt_state, _ = update(model, opt_state, (images, labels))
            epoch_time = time.time() - start_time

            test_acc, test_count = 0.0, 0
            for batch in itertools.islice(test_dl, 10):
                test_images, test_labels = batch
                test_images = test_images[:, None]
                acc = accuracy(model, (test_images, test_labels))
                test_acc += acc * test_images.shape[0]
                test_count += test_images.shape[0]
            test_acc = test_acc / test_count
            print(f"Epoch {epoch} in {epoch_time:0.2f} sec")
            print(f"Test set accuracy {test_acc}")


if __name__ == "__main__":
    # python -m examples.mnist_old
    main()
