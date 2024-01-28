"""Defines PyTest configuration for the project."""

import functools
import random

import jax
import numpy as np
import pytest
from _pytest.python import Function


@pytest.fixture(autouse=True)
def set_random_seed() -> None:
    random.seed(1337)
    np.random.seed(1337)


@functools.lru_cache()
def has_gpu() -> bool:
    try:
        jax.devices("gpu")
        return True
    except RuntimeError:
        return False


@functools.lru_cache()
def has_mps() -> bool:
    try:
        jax.devices("xla_mps")
        return True
    except RuntimeError:
        return False


def pytest_runtest_setup(item: Function) -> None:
    for mark in item.iter_markers():
        if mark.name == "has_gpu" and not has_gpu():
            pytest.skip("Skipping because this test requires a GPU and none is available")


def pytest_collection_modifyitems(items: list[Function]) -> None:
    items.sort(key=lambda x: x.get_closest_marker("slow") is not None)
