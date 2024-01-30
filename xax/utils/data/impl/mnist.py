"""Implements the MNIST dataset."""

import array
import gzip
import logging
import struct
from typing import Literal

import numpy as np

from xax.utils.data.dataset import Dataset
from xax.utils.experiments import DataDownloader
from xax.utils.numpy import one_hot as one_hot_fn, partial_flatten, worker_chunk

logger = logging.getLogger(__name__)

MnistDtype = Literal["int8", "float32"]


class MNIST(Dataset[tuple[np.ndarray, np.ndarray]]):
    def __init__(self, train: bool, dtype: MnistDtype = "int8", one_hot: bool = False) -> None:
        super().__init__()

        self.train = train
        self.dtype = dtype
        self.one_hot = one_hot

        base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
        images_name = "train-images-idx3-ubyte.gz" if train else "t10k-images-idx3-ubyte.gz"
        labels_name = "train-labels-idx1-ubyte.gz" if train else "t10k-labels-idx1-ubyte.gz"
        images_path = DataDownloader(base_url + images_name, "mnist", images_name).ensure_downloaded()
        labels_path = DataDownloader(base_url + labels_name, "mnist", labels_name).ensure_downloaded()

        with gzip.open(labels_path, "rb") as fh:
            _ = struct.unpack(">II", fh.read(8))
            labels = np.array(array.array("B", fh.read()), dtype=np.uint8)

        with gzip.open(images_path, "rb") as fh:
            _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            images = np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

        self.labels = one_hot_fn(labels, 10) if one_hot else labels
        self.images = partial_flatten(images) / np.float32(255.0)

        self.rand = np.random.RandomState(0)

    def as_dtype(self, images: np.ndarray) -> np.ndarray:
        if self.dtype == "int8":
            return images
        elif self.dtype == "float32":
            return images.astype(np.float32) / 255.0
        else:
            raise ValueError(f"Unknown dtype: {self.dtype}")

    def worker_init(self, worker_id: int, num_workers: int) -> None:
        self.images = self.images[:5000]
        self.labels = self.labels[:5000]

        self.images = worker_chunk(self.images, worker_id, num_workers)
        self.labels = worker_chunk(self.labels, worker_id, num_workers)

    def next(self) -> tuple[np.ndarray, np.ndarray]:
        index = self.rand.randint(0, len(self.images))
        image = self.images[index]
        label = self.labels[index]
        return self.as_dtype(image), label


if __name__ == "__main__":
    # python -m xax.utils.data.impl.mnist
    MNIST(True, "float32").test(max_samples=1000)
