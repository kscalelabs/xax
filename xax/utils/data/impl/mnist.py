"""Implements the MNIST dataset."""

import array
import gzip
import logging
import struct
from typing import Literal

import numpy as np

from xax.utils.data.dataset import Dataset
from xax.utils.experiments import DataDownloader

logger = logging.getLogger(__name__)

MnistDtype = Literal["int8", "float32"]


class MNIST(Dataset[tuple[np.ndarray, np.ndarray]]):
    def __init__(self, train: bool, dtype: MnistDtype = "int8") -> None:
        super().__init__()

        self.train = train
        self.dtype = dtype

        base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
        images_name = "train-images-idx3-ubyte.gz" if train else "t10k-images-idx3-ubyte.gz"
        labels_name = "train-labels-idx1-ubyte.gz" if train else "t10k-labels-idx1-ubyte.gz"
        images_path = DataDownloader(base_url + images_name, "mnist", images_name).ensure_downloaded()
        labels_path = DataDownloader(base_url + labels_name, "mnist", labels_name).ensure_downloaded()

        with gzip.open(labels_path, "rb") as fh:
            _ = struct.unpack(">II", fh.read(8))
            self.labels = np.array(array.array("B", fh.read()), dtype=np.uint8)

        with gzip.open(images_path, "rb") as fh:
            _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            self.images = np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

        self.index = 0

    def as_dtype(self, images: np.ndarray) -> np.ndarray:
        if self.dtype == "int8":
            return images
        elif self.dtype == "float32":
            return images.astype(np.float32) / 255.0
        else:
            raise ValueError(f"Unknown dtype: {self.dtype}")

    def worker_init(self, worker_id: int, num_workers: int) -> None:
        self.images = self.images[worker_id::num_workers]
        self.labels = self.labels[worker_id::num_workers]

    def start(self) -> None:
        self.index = 0
        if self.train:
            perm = np.random.RandomState(0).permutation(self.images.shape[0])
            self.images = self.images[perm]
            self.labels = self.labels[perm]

    def next(self) -> tuple[np.ndarray, np.ndarray]:
        if self.index >= len(self.images):
            raise StopIteration
        image = self.images[self.index]
        label = self.labels[self.index]
        self.index += 1
        return self.as_dtype(image), label


if __name__ == "__main__":
    # python -m xax.utils.data.impl.mnist
    MNIST(True, "float32").test(max_samples=1000)
