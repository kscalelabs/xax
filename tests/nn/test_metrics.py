"""Tests metrics functions."""

import jax.numpy as jnp

import xax


def test_dtw() -> None:
    t = jnp.linspace(0, 2 * jnp.pi, 32)
    timeseries_1 = jnp.sin(t)
    timeseries_2 = jnp.cos(t)
    series = [timeseries_1, timeseries_2]

    dtw_distance = xax.dtw(series[0], series[1])
    assert dtw_distance.shape == ()
    assert dtw_distance.dtype == jnp.float32
    assert dtw_distance.item() == 0.0
