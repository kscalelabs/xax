"""MNIST example in Jax."""

import itertools
import time

import jax.numpy as jnp
import numpy as np
from jax import grad, jit, random
from jax.example_libraries import optimizers, stax
from jax.example_libraries.optimizers import OptimizerState
from jax.example_libraries.stax import Dense, LogSoftmax, Relu
from jaxtyping import ArrayLike

import xax


def collate(batch: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    images, labels = zip(*batch)
    return np.stack(images), np.stack(labels)


def loss(params: tuple[ArrayLike, ArrayLike], batch: tuple[ArrayLike, ArrayLike]) -> ArrayLike:
    inputs, targets = batch
    preds = predict(params, inputs)
    return -jnp.mean(jnp.sum(preds * targets, axis=1))


def accuracy(params: tuple[ArrayLike, ArrayLike], batch: tuple[ArrayLike, ArrayLike]) -> ArrayLike:
    inputs, targets = batch
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(predict(params, inputs), axis=1)
    return jnp.mean(predicted_class == target_class)


init_random_params, predict = stax.serial(Dense(1024), Relu, Dense(1024), Relu, Dense(10), LogSoftmax)


def main() -> None:
    rng = random.PRNGKey(0)

    step_size = 0.001
    num_epochs = 10
    batch_size = 128
    momentum_mass = 0.9

    opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=momentum_mass)

    @jit
    def update(i: int, opt_state: OptimizerState, batch: tuple[ArrayLike, ArrayLike]) -> OptimizerState:
        params = get_params(opt_state)
        return opt_update(i, grad(loss)(params, batch), opt_state)

    _, init_params = init_random_params(rng, (-1, 28 * 28))
    opt_state = opt_init(init_params)
    itercount = itertools.count()

    print("\nStarting training...")
    with (
        xax.Dataloader(xax.MNIST(train=True), num_workers=32, batch_size=batch_size) as train_dl,
        xax.Dataloader(xax.MNIST(train=False), num_workers=1, batch_size=batch_size) as test_dl,
    ):
        for batch in train_dl:
            start_time = time.time()
            n = 0
            for batch in train_dl:
                collated_batch = collate(batch)
                n += collated_batch[0].shape[0]
                print(n)
                opt_state = update(next(itercount), opt_state, collated_batch)
            epoch_time = time.time() - start_time

            breakpoint()

            params = get_params(opt_state)
            test_acc, test_count = 0.0, 0
            for batch in test_dl:
                test_images, test_labels = collate(batch)
                breakpoint()
                acc = accuracy(params, (test_images, test_labels))
                test_count += test_images.shape[0]
                print(f"Epoch {epoch} in {epoch_time:0.2f} sec")
            print(f"Test set accuracy {test_acc}")


if __name__ == "__main__":
    # python -m examples.mnist_old
    main()
