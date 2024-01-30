"""Tests the dataset classes."""

import itertools
import random

import pytest

import xax


class DummyDataset(xax.Dataset[int]):
    def __init__(self, max_len: int) -> None:
        super().__init__()

        self.max_len = max_len
        self.i = 0

    def start(self) -> None:
        self.i = 0

    def next(self) -> int:
        if self.i >= self.max_len:
            raise StopIteration
        self.i += 1
        return random.randint(0, 5)


def test_dataset_simple() -> None:
    ds = DummyDataset(10)
    for sample in itertools.islice(ds, 10):
        assert isinstance(sample, int)


@pytest.mark.parametrize("stop_on_first", [True, False])
def test_round_robin_dataset(stop_on_first: bool) -> None:
    dss = [DummyDataset(i) for i in range(1, 5)]
    ds = xax.RoundRobinDataset(dss, stop_on_first=stop_on_first)
    num_samples = sum(1 for _ in ds)
    assert num_samples == (4 if stop_on_first else sum(range(1, 5)))


@pytest.mark.parametrize("stop_on_first", [True, False])
def test_random_dataset(stop_on_first: bool) -> None:
    dss = [DummyDataset(i) for i in range(1, 5)]
    ds = xax.RandomDataset(dss, stop_on_first=stop_on_first)
    num_samples = sum(1 for _ in ds)
    if not stop_on_first:
        assert num_samples == sum(range(1, 5))


if __name__ == "__main__":
    # python -m tests.utils.test_dataset
    test_round_robin_dataset(False)
