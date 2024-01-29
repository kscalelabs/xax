"""MNIST example in Jax."""

import array
import gzip
import itertools
import os
import struct
import time
import urllib.request
from typing import Iterator

import jax.numpy as jnp
import numpy as np
import numpy.random as npr
from jax import grad, jit, random
from jax.example_libraries import optimizers, stax
from jax.example_libraries.optimizers import OptimizerState
from jax.example_libraries.stax import Dense, LogSoftmax, Relu
from jaxtyping import ArrayLike

_DATA = "/tmp/jax_example_data/"


def _download(url: str, filename: str) -> None:
    if not os.path.exists(_DATA):
        os.makedirs(_DATA)
    out_file = os.path.join(_DATA, filename)
    if not os.path.isfile(out_file):
        urllib.request.urlretrieve(url, out_file)
        print(f"downloaded {url} to {_DATA}")


def _partial_flatten(x: np.ndarray) -> np.ndarray:
    return np.reshape(x, (x.shape[0], -1))


def _one_hot(x: np.ndarray, k: int, dtype: type = np.float32) -> np.ndarray:
    return np.array(x[:, None] == np.arange(k), dtype)


def mnist_raw() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

    def parse_labels(filename: str) -> np.ndarray:
        with gzip.open(filename, "rb") as fh:
            _ = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename: str) -> np.ndarray:
        with gzip.open(filename, "rb") as fh:
            _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

    for filename in [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]:
        _download(base_url + filename, filename)

    train_images = parse_images(os.path.join(_DATA, "train-images-idx3-ubyte.gz"))
    train_labels = parse_labels(os.path.join(_DATA, "train-labels-idx1-ubyte.gz"))
    test_images = parse_images(os.path.join(_DATA, "t10k-images-idx3-ubyte.gz"))
    test_labels = parse_labels(os.path.join(_DATA, "t10k-labels-idx1-ubyte.gz"))

    return train_images, train_labels, test_images, test_labels


def mnist(permute_train: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_images, train_labels, test_images, test_labels = mnist_raw()

    train_images = _partial_flatten(train_images) / np.float32(255.0)
    test_images = _partial_flatten(test_images) / np.float32(255.0)
    train_labels = _one_hot(train_labels, 10)
    test_labels = _one_hot(test_labels, 10)

    if permute_train:
        perm = np.random.RandomState(0).permutation(train_images.shape[0])
        train_images = train_images[perm]
        train_labels = train_labels[perm]

    return train_images, train_labels, test_images, test_labels


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

if __name__ == "__main__":
    # python -m examples.mnist
    rng = random.PRNGKey(0)

    step_size = 0.001
    num_epochs = 10
    batch_size = 128
    momentum_mass = 0.9

    train_images, train_labels, test_images, test_labels = mnist()
    num_train = train_images.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    def data_stream() -> Iterator[tuple[np.ndarray, np.ndarray]]:
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                yield train_images[batch_idx], train_labels[batch_idx]

    batches = data_stream()

    opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=momentum_mass)

    @jit
    def update(i: int, opt_state: OptimizerState, batch: tuple[ArrayLike, ArrayLike]) -> OptimizerState:
        params = get_params(opt_state)
        return opt_update(i, grad(loss)(params, batch), opt_state)

    _, init_params = init_random_params(rng, (-1, 28 * 28))
    opt_state = opt_init(init_params)
    itercount = itertools.count()

    print("\nStarting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        for _ in range(num_batches):
            opt_state = update(next(itercount), opt_state, next(batches))
        epoch_time = time.time() - start_time

        params = get_params(opt_state)
        train_acc = accuracy(params, (train_images, train_labels))
        test_acc = accuracy(params, (test_images, test_labels))
        print(f"Epoch {epoch} in {epoch_time:0.2f} sec")
        print(f"Training set accuracy {train_acc}")
        print(f"Test set accuracy {test_acc}")
