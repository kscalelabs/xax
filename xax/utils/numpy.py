"""Defines some Numpy utility functions."""

import numpy as np


def partial_flatten(x: np.ndarray) -> np.ndarray:
    """Flattens all but the first dimension of an array.

    Args:
        x: The array to flatten.

    Returns:
        The flattened array.
    """
    return np.reshape(x, (x.shape[0], -1))


def one_hot(x: np.ndarray, k: int, dtype: type = np.float32) -> np.ndarray:
    """Converts an array of labels to a one-hot representation.

    Args:
        x: The array of labels.
        k: The number of classes.
        dtype: The dtype of the returned array.

    Returns:
        The one-hot representation of the labels.
    """
    return np.array(x[:, None] == np.arange(k), dtype)


def worker_chunk(x: np.ndarray, worker_id: int, num_workers: int, dim: int = 0) -> np.ndarray:
    """Chunks an array into `num_workers` chunks.

    Args:
        x: The array to chunk.
        worker_id: The worker ID.
        num_workers: The number of workers.
        dim: The dimension to chunk along.

    Returns:
        The chunked array.
    """
    chunk_size = x.shape[dim] // num_workers
    start = worker_id * chunk_size
    end = start + chunk_size
    return x[start:end]
