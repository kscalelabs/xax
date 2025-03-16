"""Tests for the pytree utils."""

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import PyTree

import xax


@pytest.fixture
def shuffle_test_data() -> PyTree:
    """Fixture providing consistent test data for all tests."""
    return {
        "observations": jnp.array(
            [
                # env 0
                [
                    [1, 2],  # t0 - feature_dim=2
                    [3, 4],  # t1
                    [5, 6],  # t2
                ],
                # env 1
                [
                    [7, 8],
                    [9, 10],
                    [11, 12],
                ],
            ]
        ),
        "actions": jnp.array(
            [
                # env 0
                [
                    [101, 102],
                    [201, 202],
                    [301, 302],
                ],
                # env 1
                [
                    [401, 402],
                    [501, 502],
                    [601, 602],
                ],
            ]
        ),
    }


@pytest.fixture
def key_42() -> jax.Array:
    """Fixture providing a consistent PRNG key with seed 42."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def key_43() -> jax.Array:
    """Fixture providing a consistent PRNG key with seed 43."""
    return jax.random.PRNGKey(43)


@pytest.fixture
def key_44() -> jax.Array:
    """Fixture providing a consistent PRNG key with seed 44."""
    return jax.random.PRNGKey(44)


@pytest.fixture
def key_45() -> jax.Array:
    """Fixture providing a consistent PRNG key with seed 45."""
    return jax.random.PRNGKey(45)


@pytest.fixture
def key_46() -> jax.Array:
    """Fixture providing a consistent PRNG key with seed 46."""
    return jax.random.PRNGKey(46)


@pytest.mark.parametrize(
    "pytree, expected",
    [
        ({"a": jnp.array([1, 2, 3]), "b": jnp.array([4, 5, 6])}, False),
        ({"a": jnp.array([1, jnp.nan, 3]), "b": jnp.array([4, 5, 6])}, True),
        ({"a": jnp.array([1, 2, 3]), "b": jnp.array([4, jnp.nan, 6])}, True),
        ({"a": jnp.array([1, 2, 3]), "b": {"c": jnp.array([4, 5, 6])}}, False),
        ({"a": jnp.array([1, 2, 3]), "b": {"c": jnp.array([4, jnp.nan, 6])}}, True),
        ({"a": jnp.array([]), "b": jnp.array([])}, False),
        ({"a": [1, 2, 3], "b": (4, 5, 6)}, False),
        (
            {
                "conv1": {
                    "kernel": jnp.ones((3, 3, 1, 16)),
                    "bias": jnp.zeros(16),
                },
                "conv2": {
                    "kernel": jnp.ones((3, 3, 16, 32)),
                    "bias": jnp.zeros(32),
                },
                "dense": {
                    "kernel": jnp.ones((32, 10)),
                    "bias": jnp.zeros(10),
                },
            },
            False,
        ),
        (
            {
                "conv1": {
                    "kernel": jnp.ones((3, 3, 1, 16)),
                    "bias": jnp.zeros(16),
                },
                "conv2": {
                    "kernel": jnp.ones((3, 3, 16, 32)),
                    "bias": jnp.array([0.0, jnp.nan, 0.0, 0.0] + [0.0] * 28),
                },
                "dense": {
                    "kernel": jnp.ones((32, 10)),
                    "bias": jnp.zeros(10),
                },
            },
            True,
        ),
        (
            {
                "embedding": jnp.ones((1000, 128)),
                "transformer_blocks": [
                    {
                        "attention": {
                            "query": {"kernel": jnp.ones((128, 64)), "bias": jnp.zeros(64)},
                            "key": {"kernel": jnp.ones((128, 64)), "bias": jnp.zeros(64)},
                            "value": {"kernel": jnp.ones((128, 64)), "bias": jnp.zeros(64)},
                            "output": {"kernel": jnp.ones((64, 128)), "bias": jnp.zeros(128)},
                        },
                        "ffn": {
                            "linear1": {"kernel": jnp.ones((128, 256)), "bias": jnp.zeros(256)},
                            "linear2": {"kernel": jnp.ones((256, 128)), "bias": jnp.zeros(128)},
                        },
                    },
                    {
                        "attention": {
                            "query": {"kernel": jnp.ones((128, 64)), "bias": jnp.zeros(64)},
                            "key": {"kernel": jnp.ones((128, 64)), "bias": jnp.zeros(64)},
                            "value": {"kernel": jnp.ones((128, 64)), "bias": jnp.zeros(64)},
                            "output": {"kernel": jnp.ones((64, 128)), "bias": jnp.zeros(128)},
                        },
                        "ffn": {
                            "linear1": {"kernel": jnp.ones((128, 256)), "bias": jnp.zeros(256)},
                            "linear2": {"kernel": jnp.array([[jnp.nan]] + [[1.0]] * 255), "bias": jnp.zeros(128)},
                        },
                    },
                ],
                "layernorm": {"scale": jnp.ones(128), "bias": jnp.zeros(128)},
            },
            True,
        ),
        (
            {
                "params": {"w": jnp.ones((10, 5)), "b": jnp.zeros(5)},
                "state": {"mean": jnp.zeros(5), "var": jnp.ones(5)},
                "hyperparams": {"lr": 0.01, "dropout_rate": 0.5},
                "metrics": {"loss": jnp.array([0.1, 0.09, 0.08]), "accuracy": jnp.array([0.85, 0.86, 0.87])},
            },
            False,
        ),
        (
            {
                "grads": {
                    "layer1": {"w": jnp.ones((5, 10)), "b": jnp.zeros(10)},
                    "layer2": {"w": jnp.ones((10, 1)), "b": jnp.array([jnp.nan])},
                },
                "optimizer_state": {
                    "m": {
                        "layer1": {"w": jnp.zeros((5, 10)), "b": jnp.zeros(10)},
                        "layer2": {"w": jnp.zeros((10, 1)), "b": jnp.zeros(1)},
                    },
                    "v": {
                        "layer1": {"w": jnp.zeros((5, 10)), "b": jnp.zeros(10)},
                        "layer2": {"w": jnp.zeros((10, 1)), "b": jnp.zeros(1)},
                    },
                },
            },
            True,
        ),
    ],
)
def test_pytree_has_nans(pytree: PyTree, expected: bool) -> None:
    assert xax.pytree_has_nans(pytree) == expected


def test_reshuffle_pytree(shuffle_test_data: PyTree, key_42: jax.Array, key_43: jax.Array) -> None:
    """Test reshuffle_pytree with fixed PRNG key."""
    # Test reshuffling along the first dimension (num_envs=2)
    reshuffled_data = xax.reshuffle_pytree(shuffle_test_data, (2,), key_42)

    # Pre-computed expected values for PRNGKey(42) and dimension size 2
    # Based on actual output
    expected_obs = jnp.array(
        [
            [
                [1, 2],
                [3, 4],
                [5, 6],
            ],
            [
                [7, 8],
                [9, 10],
                [11, 12],
            ],
        ]
    )

    expected_actions = jnp.array(
        [
            [
                [101, 102],
                [201, 202],
                [301, 302],
            ],
            [
                [401, 402],
                [501, 502],
                [601, 602],
            ],
        ]
    )

    # Compare with expected values
    assert jnp.array_equal(reshuffled_data["observations"], expected_obs)
    assert jnp.array_equal(reshuffled_data["actions"], expected_actions)

    # Test reshuffling along the first two dimensions (num_envs=2, num_timesteps=3)
    reshuffled_data = xax.reshuffle_pytree(shuffle_test_data, (2, 3), key_43)

    # Pre-computed expected values for PRNGKey(43) and flattened dimension size 6
    # Based on actual output
    expected_obs = jnp.array(
        [
            [
                [1, 2],  # Original (0, 0) data
                [11, 12],  # Original (1, 2) data
                [9, 10],  # Original (1, 1) data
            ],
            [
                [7, 8],  # Original (1, 0) data
                [3, 4],  # Original (0, 1) data
                [5, 6],  # Original (0, 2) data
            ],
        ]
    )

    expected_actions = jnp.array(
        [
            [
                [101, 102],  # Original (0, 0) data
                [601, 602],  # Original (1, 2) data
                [501, 502],  # Original (1, 1) data
            ],
            [
                [401, 402],  # Original (1, 0) data
                [201, 202],  # Original (0, 1) data
                [301, 302],  # Original (0, 2) data
            ],
        ]
    )

    # Compare with expected values
    assert jnp.array_equal(reshuffled_data["observations"], expected_obs)
    assert jnp.array_equal(reshuffled_data["actions"], expected_actions)


def test_reshuffle_pytree_independently(shuffle_test_data: PyTree, key_44: jax.Array) -> None:
    """Test reshuffle_pytree_independently with fixed PRNG key."""
    # Test reshuffling along the first two dimensions independently
    reshuffled_data = xax.reshuffle_pytree_independently(shuffle_test_data, (2, 3), key_44)

    # Pre-computed expected values for PRNGKey(44)
    # Based on actual output
    expected_obs = jnp.array(
        [
            [
                [9, 10],  # Original (1, 1) data
                [7, 8],  # Original (1, 0) data
                [11, 12],  # Original (1, 2) data
            ],
            [
                [3, 4],  # Original (0, 1) data
                [1, 2],  # Original (0, 0) data
                [5, 6],  # Original (0, 2) data
            ],
        ]
    )

    expected_actions = jnp.array(
        [
            [
                [501, 502],  # Original (1, 1) data
                [401, 402],  # Original (1, 0) data
                [601, 602],  # Original (1, 2) data
            ],
            [
                [201, 202],  # Original (0, 1) data
                [101, 102],  # Original (0, 0) data
                [301, 302],  # Original (0, 2) data
            ],
        ]
    )

    # Compare with expected values
    assert jnp.array_equal(reshuffled_data["observations"], expected_obs)
    assert jnp.array_equal(reshuffled_data["actions"], expected_actions)

    # Verify that the reshuffling is done independently for each dimension

    # Check that the same permutation is applied to both environments
    # Find the permutation for dimension 1 (timesteps)
    # For env 0 in the reshuffled data (which is env 1 in the original data)
    env0_perm = []
    for t in range(3):
        for orig_t in range(3):
            if jnp.array_equal(reshuffled_data["observations"][0, t], shuffle_test_data["observations"][1, orig_t]):
                env0_perm.append(orig_t)
                break

    # For env 1 in the reshuffled data (which is env 0 in the original data)
    env1_perm = []
    for t in range(3):
        for orig_t in range(3):
            if jnp.array_equal(reshuffled_data["observations"][1, t], shuffle_test_data["observations"][0, orig_t]):
                env1_perm.append(orig_t)
                break

    # The permutation should be the same for both environments
    assert env0_perm == env1_perm
    assert env0_perm == [1, 0, 2]


def test_reshuffle_pytree_along_dims(shuffle_test_data: PyTree, key_45: jax.Array, key_46: jax.Array) -> None:
    """Test reshuffle_pytree_along_dims with fixed PRNG key."""
    # Test reshuffling along the time dimension only (dimension 1)
    time_reshuffled_data = xax.reshuffle_pytree_along_dims(shuffle_test_data, (1,), (3,), key_45)

    # Pre-computed expected values for PRNGKey(45)
    # Based on actual output
    expected_obs = jnp.array(
        [
            [
                [1, 2],  # Original (0, 0) data
                [3, 4],  # Original (0, 1) data
                [5, 6],  # Original (0, 2) data
            ],
            [
                [7, 8],  # Original (1, 0) data
                [9, 10],  # Original (1, 1) data
                [11, 12],  # Original (1, 2) data
            ],
        ]
    )

    expected_actions = jnp.array(
        [
            [
                [101, 102],  # Original (0, 0) data
                [201, 202],  # Original (0, 1) data
                [301, 302],  # Original (0, 2) data
            ],
            [
                [401, 402],  # Original (1, 0) data
                [501, 502],  # Original (1, 1) data
                [601, 602],  # Original (1, 2) data
            ],
        ]
    )

    assert jnp.array_equal(time_reshuffled_data["observations"], expected_obs)
    assert jnp.array_equal(time_reshuffled_data["actions"], expected_actions)

    # Test reshuffling along both env and time dimensions
    env_time_reshuffled_data = xax.reshuffle_pytree_along_dims(shuffle_test_data, (0, 1), (2, 3), key_46)

    # Pre-computed expected values for PRNGKey(46)
    # Based on actual output
    expected_env_time_obs = jnp.array(
        [
            [
                [1, 2],  # Original (0, 0) data
                [9, 10],  # Original (1, 1) data
                [7, 8],  # Original (1, 0) data
            ],
            [
                [5, 6],  # Original (0, 2) data
                [11, 12],  # Original (1, 2) data
                [3, 4],  # Original (0, 1) data
            ],
        ]
    )

    expected_env_time_actions = jnp.array(
        [
            [
                [101, 102],  # Original (0, 0) data
                [501, 502],  # Original (1, 1) data
                [401, 402],  # Original (1, 0) data
            ],
            [
                [301, 302],  # Original (0, 2) data
                [601, 602],  # Original (1, 2) data
                [201, 202],  # Original (0, 1) data
            ],
        ]
    )

    # Compare with expected values
    assert jnp.array_equal(env_time_reshuffled_data["observations"], expected_env_time_obs)
    assert jnp.array_equal(env_time_reshuffled_data["actions"], expected_env_time_actions)


def test_compare_reshuffle_methods(shuffle_test_data: PyTree, key_42: jax.Array) -> None:
    """Compare the behavior of different reshuffle methods."""
    # Reshuffle using reshuffle_pytree
    reshuffled1 = xax.reshuffle_pytree(shuffle_test_data, (2, 3), key_42)

    # Pre-computed expected values for PRNGKey(42) with reshuffle_pytree
    # Based on actual output
    expected_reshuffled1 = jnp.array(
        [
            [
                [9, 10],  # Original (1, 1) data
                [5, 6],  # Original (0, 2) data
                [11, 12],  # Original (1, 2) data
            ],
            [
                [7, 8],  # Original (1, 0) data
                [1, 2],  # Original (0, 0) data
                [3, 4],  # Original (0, 1) data
            ],
        ]
    )

    assert jnp.array_equal(reshuffled1["observations"], expected_reshuffled1)

    reshuffled2 = xax.reshuffle_pytree_along_dims(shuffle_test_data, (0, 1), (2, 3), key_42)

    # Verify that both methods produce the same result when reshuffling the same dimensions
    # with the same key (since reshuffle_pytree_along_dims uses reshuffle_pytree internally)
    assert jnp.array_equal(reshuffled1["observations"], reshuffled2["observations"])

    # Compare with reshuffle_pytree_independently which should produce different results
    # since it uses a different reshuffling approach
    reshuffled3 = xax.reshuffle_pytree_independently(shuffle_test_data, (2, 3), key_42)

    # Pre-computed expected values for PRNGKey(42) with reshuffle_pytree_independently
    # Based on actual output
    expected_reshuffled3 = jnp.array(
        [
            [
                [11, 12],  # Original (1, 2) data
                [7, 8],  # Original (1, 0) data
                [9, 10],  # Original (1, 1) data
            ],
            [
                [5, 6],  # Original (0, 2) data
                [1, 2],  # Original (0, 0) data
                [3, 4],  # Original (0, 1) data
            ],
        ]
    )

    assert jnp.array_equal(reshuffled3["observations"], expected_reshuffled3)
    assert not jnp.array_equal(reshuffled1["observations"], reshuffled3["observations"])

    # Verify that reshuffle_pytree_along_dims with only the time dimension
    # produces different results than reshuffling both dimensions
    reshuffled4 = xax.reshuffle_pytree_along_dims(shuffle_test_data, (1,), (3,), key_42)

    # Pre-computed expected values for PRNGKey(42) with reshuffle_pytree_along_dims on dim 1
    # Based on actual output
    expected_reshuffled4 = jnp.array(
        [
            [
                [5, 6],  # Original (0, 2) data
                [1, 2],  # Original (0, 0) data
                [3, 4],  # Original (0, 1) data
            ],
            [
                [11, 12],  # Original (1, 2) data
                [7, 8],  # Original (1, 0) data
                [9, 10],  # Original (1, 1) data
            ],
        ]
    )

    assert jnp.array_equal(reshuffled4["observations"], expected_reshuffled4)

    assert not jnp.array_equal(reshuffled2["observations"], reshuffled4["observations"])

    # Verify that when reshuffling only the time dimension, the env dimension is preserved
    # For each environment, the set of timesteps should be the same before and after
    for env in range(2):
        orig_timesteps = jnp.sort(shuffle_test_data["observations"][env, :, 0])
        reshuffled_timesteps = jnp.sort(reshuffled4["observations"][env, :, 0])
        assert jnp.array_equal(orig_timesteps, reshuffled_timesteps)
