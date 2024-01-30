"""Tests the dataset classes."""

import itertools
from typing import Iterator

import xax


class DummyDataset(xax.Dataset[int]):
    def __init__(self, value: int) -> None:
        super().__init__()

        self.value = value

    def next(self) -> int:
        return self.value


def test_dataset_simple() -> None:
    ds = DummyDataset(1)
    for sample in itertools.islice(ds, 10):
        assert sample == 1


def test_round_robin_dataset() -> None:
    dss = [DummyDataset(i) for i in range(5)]
    ds = xax.RoundRobinDataset(dss)
    assert list(itertools.islice(ds, 10)) == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]


def test_random_dataset() -> None:
    dss = [DummyDataset(i) for i in range(5)]
    ds = xax.RandomDataset(dss)
    assert all(0 <= i < 5 for i in itertools.islice(ds, 10))


class DummyChunkedDataset(xax.ChunkedDataset[int]):
    def __init__(self, value: int) -> None:
        super().__init__()

        self.value = value

    def next_chunk(self) -> Iterator[int]:
        for i in range(self.value):
            yield i


def test_chunked_dataset() -> None:
    ds = DummyChunkedDataset(5)
    assert list(itertools.islice(ds, 10)) == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]


if __name__ == "__main__":
    # python -m tests.utils.test_dataset
    test_round_robin_dataset()
