"""Tests the dataloader."""

import itertools
import random

import pytest

import xax


class DummyDataset(xax.Dataset[int]):
    def start(self) -> None:
        pass

    def next(self) -> int:
        return random.randint(0, 5)


@pytest.mark.parametrize("num_workers", [0, 1, 2])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_dataloader(num_workers: int, batch_size: int) -> None:
    ds = DummyDataset()
    with xax.Dataloader(ds, num_workers=num_workers, batch_size=batch_size) as dl:
        for sample in itertools.islice(dl, 10):
            assert len(sample) == batch_size


if __name__ == "__main__":
    # python -m tests.utils.test_dataloader
    test_dataloader(0, 1)
