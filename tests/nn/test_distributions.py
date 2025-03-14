"""Tests for the distributions module."""

import jax
import jax.numpy as jnp
import pytest

import xax


def test_gaussian_distribution() -> None:
    """Test the Gaussian distribution."""
    distribution = xax.GaussianDistribution(action_dim=2)
    parameters = jnp.array([0.0, 1.0, 0.0, 1.0])
    actions = jnp.array([0.0, 0.0])
    assert jnp.allclose(
        distribution.log_prob(parameters, actions),
        -0.5 * jnp.square((actions - parameters[:2]) / parameters[2:]),
        atol=1e-6,
    )
