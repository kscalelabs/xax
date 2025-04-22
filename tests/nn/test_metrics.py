"""Tests metrics functions."""

import jax.numpy as jnp

import xax


def test_dtw_identical() -> None:
    series_1 = jnp.array([1, 2, 3, 4, 5])
    series_2 = jnp.array([1, 2, 3, 4, 5])

    dtw_distance = xax.dtw(series_1, series_2)
    assert dtw_distance.item() == 0.0


def test_dtw_different() -> None:
    series_1 = jnp.array([1, 2, 3, 4, 5])
    series_2 = jnp.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])

    dtw_distance = xax.dtw(series_1, series_2)
    assert dtw_distance.item() == -1.0
