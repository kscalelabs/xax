"""Tests for the distributions module."""

import jax
import jax.numpy as jnp
import pytest

import xax

EPSILON = 1e-6


# Helper function to construct parameter vector from mean and std.
def construct_params(mean: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([mean, std])


@pytest.mark.parametrize(
    "mean, std, actions",
    [
        (jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]), jnp.array([0.0, 0.0])),
        (jnp.array([1.0, -1.0]), jnp.array([0.5, 2.0]), jnp.array([1.0, 0.0])),
        (jnp.array([2.0, 3.0, 1.0]), jnp.array([1.5, 0.2, 2.0]), jnp.array([2.0, 3.0, 1.0])),
        (jnp.array([0.0]), jnp.array([1e-5]), jnp.array([0.0])),
        (jnp.array([1e6, -1e6]), jnp.array([10.0, 5.0]), jnp.array([1e6, -1e6])),
        (jnp.array([0.0, 0.0]), jnp.array([1e-10, 1e-10]), jnp.array([1e-10, -1e-10])),
    ],
)
def test_gaussian_log_prob_values(mean: jnp.ndarray, std: jnp.ndarray, actions: jnp.ndarray) -> None:
    """Test that Gaussian log_prob returns the expected value for various means and stds."""
    parameters = construct_params(mean, std)
    distribution = xax.GaussianDistribution(action_dim=mean.shape[0])
    expected = -0.5 * jnp.square((actions - mean) / std) - jnp.log(std) - 0.5 * jnp.log(2 * jnp.pi)
    expected_log_prob = expected
    computed_log_prob = distribution.log_prob(parameters, actions)
    assert jnp.allclose(computed_log_prob, expected_log_prob, atol=EPSILON)


@pytest.mark.parametrize(
    "mean, std",
    [
        (jnp.array([1.0, 2.0, 3.0]), jnp.array([1.0, 1.0, 1.0])),
        (jnp.array([-1.0, 0.0]), jnp.array([0.5, 2.0])),
        (jnp.array([1e9, -1e9]), jnp.array([1.0, 1.0])),
        (jnp.array([0.0, 0.0, 0.0]), jnp.array([1e-8, 1e-8, 1e-8])),
    ],
)
def test_gaussian_mode(mean: jnp.ndarray, std: jnp.ndarray) -> None:
    """Test that Gaussian mode returns the mean for various parameter settings."""
    parameters = construct_params(mean, std)
    distribution = xax.GaussianDistribution(action_dim=mean.shape[0])
    computed_mode = distribution.mode(parameters)
    expected_mode = mean
    assert jnp.allclose(computed_mode, expected_mode, atol=EPSILON)


@pytest.mark.parametrize(
    "mean, std",
    [
        (jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0])),
        (jnp.array([1.0, 2.0, 3.0, 4.0]), jnp.array([0.5, 1.0, 1.5, 2.0])),
        (jnp.zeros(10), jnp.ones(10)),
        (jnp.array([1.0]), jnp.array([0.1])),
    ],
)
def test_gaussian_sample_shape(mean: jnp.ndarray, std: jnp.ndarray) -> None:
    """Test that Gaussian sample returns the correct shape for different dimensions."""
    parameters = construct_params(mean, std)
    distribution = xax.GaussianDistribution(action_dim=mean.shape[0])
    rng = jax.random.PRNGKey(42)
    sample = distribution.sample(parameters, rng)
    assert sample.shape == mean.shape


@pytest.mark.parametrize(
    "mean, std",
    [
        (jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0])),
        (jnp.array([1.0, -1.0]), jnp.array([0.5, 2.0])),
        (jnp.array([2.0, 3.0]), jnp.array([1.5, 0.2])),
        (jnp.array([0.0, 0.0]), jnp.array([1e-5, 1e-5])),
        (jnp.array([10.0, -10.0]), jnp.array([100.0, 100.0])),
    ],
)
def test_gaussian_entropy(mean: jnp.ndarray, std: jnp.ndarray) -> None:
    """Test that Gaussian entropy returns the expected value for various parameters."""
    parameters = construct_params(mean, std)
    distribution = xax.GaussianDistribution(action_dim=mean.shape[0])
    expected_entropy = 0.5 + 0.5 * jnp.log(2 * jnp.pi) + jnp.log(std)
    rng = jax.random.PRNGKey(123)
    computed_entropy = distribution.entropy(parameters, rng)
    assert jnp.allclose(computed_entropy, expected_entropy, atol=EPSILON)


def test_gaussian_invalid_parameters() -> None:
    """Test that an invalid parameters shape raises a ValueError."""
    distribution = xax.GaussianDistribution(action_dim=3)
    # For action_dim=3, we expect parameters to have 6 elements.
    parameters = jnp.zeros(5)
    with pytest.raises(AssertionError):
        distribution.get_mean_std(parameters)


def test_gaussian_negative_std() -> None:
    """Test that negative std values in GaussianDistribution produce NaN values in computations."""
    distribution = xax.GaussianDistribution(action_dim=2)
    neg_std = jnp.array([-1.0, -0.5])
    parameters = construct_params(jnp.array([0.0, 0.0]), neg_std)
    mean, std = distribution.get_mean_std(parameters)
    assert jnp.allclose(mean, jnp.array([0.0, 0.0]), atol=EPSILON)
    assert jnp.allclose(std, neg_std, atol=EPSILON)

    # Check that computations with negative std produce NaN values
    actions = jnp.array([0.1, 0.2])
    log_prob = distribution.log_prob(parameters, actions)
    assert jnp.any(jnp.isnan(log_prob))

    rng = jax.random.PRNGKey(42)
    entropy = distribution.entropy(parameters, rng)
    assert jnp.any(jnp.isnan(entropy))


@pytest.mark.parametrize(
    "mean, std",
    [
        (jnp.array([0.5, -0.5]), jnp.array([1.0, 1.0])),
        (jnp.array([1.0, 2.0]), jnp.array([0.5, 2.0])),
        (jnp.array([10.0, -10.0]), jnp.array([0.1, 0.1])),
        (jnp.array([100.0, 100.0]), jnp.array([1.0, 1.0])),
        (jnp.array([0.0, 0.0, 0.0]), jnp.array([0.01, 0.01, 0.01])),
    ],
)
def test_tanh_gaussian_mode(mean: jnp.ndarray, std: jnp.ndarray) -> None:
    """Test that the mode of TanhGaussianDistribution is tanh(mean) for various parameters."""
    parameters = construct_params(mean, std)
    distribution = xax.TanhGaussianDistribution(action_dim=mean.shape[0])
    computed_mode = distribution.mode(parameters)
    expected_mode = jnp.tanh(mean)
    assert jnp.allclose(computed_mode, expected_mode, atol=EPSILON)


@pytest.mark.parametrize(
    "mean, std",
    [
        (jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0])),
        (jnp.array([1.0, -1.0, 0.5]), jnp.array([0.5, 2.0, 1.0])),
        (jnp.array([1000.0, -1000.0]), jnp.array([1.0, 1.0])),
        (jnp.array([0.0, 0.0]), jnp.array([0.01, 100.0])),
    ],
)
def test_tanh_gaussian_sample_range(mean: jnp.ndarray, std: jnp.ndarray) -> None:
    """Test that samples from TanhGaussianDistribution lie in (-1, 1) for various parameters."""
    parameters = construct_params(mean, std)
    distribution = xax.TanhGaussianDistribution(action_dim=mean.shape[0])
    rng = jax.random.PRNGKey(0)
    samples = [distribution.sample(parameters, rng) for _ in range(100)]
    samples_array = jnp.array(samples)
    # All outputs from tanh should be in (-1, 1)
    assert jnp.all(samples_array <= 1.0)
    assert jnp.all(samples_array >= -1.0)


@pytest.mark.parametrize(
    "mean, std, pre_tanh",
    [
        (jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]), jnp.array([0.2, -0.2])),
        (jnp.array([1.0, -1.0]), jnp.array([0.5, 2.0]), jnp.array([0.1, -0.3])),
        (jnp.array([10.0, -10.0]), jnp.array([0.1, 0.1]), jnp.array([9.9, -9.9])),
        (jnp.array([0.0]), jnp.array([0.01]), jnp.array([0.0])),
        (jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]), jnp.array([10.0, -10.0])),
    ],
)
def test_tanh_gaussian_log_prob(mean: jnp.ndarray, std: jnp.ndarray, pre_tanh: jnp.ndarray) -> None:
    """Test that TanhGaussian log_prob returns the expected value for various parameters."""
    parameters = construct_params(mean, std)
    distribution = xax.TanhGaussianDistribution(action_dim=mean.shape[0])
    actions = jnp.tanh(pre_tanh)
    # Mimic the clipping performed in the implementation.
    clipped_actions = jnp.clip(actions, -1 + EPSILON, 1 - EPSILON)
    pre_tanh_computed = jnp.arctanh(clipped_actions)
    # No longer applying softplus to std
    expected_base = -0.5 * jnp.square((pre_tanh_computed - mean) / std) - jnp.log(std) - 0.5 * jnp.log(2 * jnp.pi)
    expected_base_log_prob = expected_base
    jacobian_correction = jnp.log(1 - jnp.square(actions) + EPSILON)
    expected_log_prob = expected_base_log_prob - jacobian_correction
    computed_log_prob = distribution.log_prob(parameters, actions)
    assert jnp.allclose(computed_log_prob, expected_log_prob, atol=EPSILON)


@pytest.mark.parametrize(
    "mean, std",
    [
        (jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0])),
        (jnp.array([1.0, -1.0]), jnp.array([0.5, 2.0])),
        (jnp.array([0.0]), jnp.array([0.01])),
        (jnp.array([100.0, -100.0]), jnp.array([10.0, 10.0])),
    ],
)
def test_tanh_gaussian_entropy_manual(mean: jnp.ndarray, std: jnp.ndarray) -> None:
    """Test that TanhGaussianDistribution entropy matches a manual calculation for various parameters."""
    parameters = construct_params(mean, std)
    distribution = xax.TanhGaussianDistribution(action_dim=mean.shape[0])
    # Use the underlying Gaussian for a baseline calculation.
    gaussian = xax.GaussianDistribution(action_dim=mean.shape[0])
    rng = jax.random.PRNGKey(456)
    base_entropy = jnp.sum(gaussian.entropy(parameters, rng))
    pre_tanh_sample = gaussian.sample(parameters, rng)
    # Manually compute the Jacobian correction used in the tanh transformation:
    jacobian_correction = jnp.sum(
        2.0 * (jnp.log(2.0) - pre_tanh_sample - jax.nn.softplus(-2.0 * pre_tanh_sample)), axis=-1
    )
    expected_entropy = base_entropy + jacobian_correction
    computed_entropy = jnp.sum(distribution.entropy(parameters, rng))
    assert jnp.allclose(computed_entropy, expected_entropy, atol=EPSILON)


def test_tanh_gaussian_negative_std() -> None:
    """Test that negative std values in TanhGaussianDistribution produce NaN values in computations."""
    distribution = xax.TanhGaussianDistribution(action_dim=2)
    neg_std = jnp.array([-1.0, -0.5])
    parameters = construct_params(jnp.array([0.5, -0.5]), neg_std)
    mean, std = distribution.get_mean_std(parameters)

    assert jnp.allclose(mean, jnp.array([0.5, -0.5]), atol=EPSILON)
    assert jnp.allclose(std, neg_std, atol=EPSILON)

    # Check that log_prob produces NaN values with negative std
    actions = jnp.tanh(jnp.array([0.1, -0.1]))
    log_prob = distribution.log_prob(parameters, actions)
    assert jnp.any(jnp.isnan(log_prob))

    # Check that entropy produces NaN values with negative std
    rng = jax.random.PRNGKey(42)
    entropy = distribution.entropy(parameters, rng)
    assert jnp.any(jnp.isnan(entropy))


@pytest.mark.parametrize(
    "logits",
    [
        jnp.array([0.1, 0.5, 0.3, 0.2]),
        jnp.array([-1.0, 0.0, 1.0]),
        jnp.array([1.0, 1.0, 1.0, 1.0]),
        jnp.array([0.0, 0.0]),
        jnp.array([1e6, 0.0, 0.0, 0.0, 0.0]),
    ],
)
def test_categorical_mode(logits: jnp.ndarray) -> None:
    """Test that Categorical mode returns the index of the highest logit."""
    distribution = xax.CategoricalDistribution(action_dim=logits.shape[0])
    computed_mode = distribution.mode(logits)
    expected_mode = jnp.argmax(logits)
    assert computed_mode == expected_mode


@pytest.mark.parametrize(
    "logits, action_index",
    [
        (jnp.array([1.0, 2.0, 3.0]), 2),
        (jnp.array([0.5, 0.2, 0.1]), 0),
        (jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]), 9),
        (jnp.array([10.0, -10.0]), 0),
        (jnp.zeros(5), 0),
    ],
)
def test_categorical_log_prob(logits: jnp.ndarray, action_index: int) -> None:
    """Test that Categorical log_prob computes the correct log probability."""
    distribution = xax.CategoricalDistribution(action_dim=logits.shape[0])
    actions = jnp.array(action_index)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    expected_log_prob = log_probs[action_index]
    computed_log_prob = jnp.sum(distribution.log_prob(logits, actions))
    assert jnp.allclose(computed_log_prob, expected_log_prob, atol=EPSILON)


@pytest.mark.parametrize(
    "logits",
    [
        jnp.array([0.1, 0.2, 0.3, 0.4, 0.0]),
        jnp.array([1.0, 0.0, -1.0]),
        jnp.array([1e10, 0.0, 0.0, 0.0]),
        jnp.array([-1e10, -1e10, 0.0]),
        jnp.zeros(20),
    ],
)
def test_categorical_sample(logits: jnp.ndarray) -> None:
    """Test that Categorical sample returns a valid index for various logits."""
    distribution = xax.CategoricalDistribution(action_dim=logits.shape[0])
    rng = jax.random.PRNGKey(789)
    sample = distribution.sample(logits, rng)
    # Check that the sample is an integer index within the valid range.
    assert sample.dtype in [jnp.int32, jnp.int64]
    assert (sample >= 0) and (sample < logits.shape[0])


@pytest.mark.parametrize(
    "logits",
    [
        jnp.array([1.0, 2.0, 3.0]),
        jnp.array([0.5, 0.5, 0.5]),
        jnp.zeros(10),
        jnp.array([100.0, 0.0, 0.0, 0.0]),
        jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]),
    ],
)
def test_categorical_entropy(logits: jnp.ndarray) -> None:
    """Test that Categorical entropy returns the expected value for various logits."""
    distribution = xax.CategoricalDistribution(action_dim=logits.shape[0])
    rng = jax.random.PRNGKey(101112)
    computed_entropy = jnp.sum(distribution.entropy(logits, rng))
    p = jax.nn.softmax(logits, axis=-1)
    log_p = jax.nn.log_softmax(logits, axis=-1)
    expected_entropy = -jnp.sum(p * log_p)
    assert jnp.allclose(computed_entropy, expected_entropy, atol=EPSILON)


def test_jittability() -> None:
    """Test that the distributions are jittable."""
    categorical = xax.CategoricalDistribution(action_dim=10)
    gaussian = xax.GaussianDistribution(action_dim=10)
    tanh_gaussian = xax.TanhGaussianDistribution(action_dim=10)

    rng = jax.random.PRNGKey(101112)

    # jit the entropy function
    jax.jit(categorical.entropy)(jnp.zeros(10), rng)
    jax.jit(gaussian.entropy)(jnp.zeros(20), rng)
    jax.jit(tanh_gaussian.entropy)(jnp.zeros(20), rng)

    jax.jit(categorical.log_prob)(jnp.zeros(10), 0)
    jax.jit(gaussian.log_prob)(jnp.zeros(20), jnp.zeros(10))
    jax.jit(tanh_gaussian.log_prob)(jnp.zeros(20), jnp.zeros(10))

    jax.jit(categorical.sample)(jnp.zeros(10), rng)
    jax.jit(gaussian.sample)(jnp.zeros(20), rng)
    jax.jit(tanh_gaussian.sample)(jnp.zeros(20), rng)

    jax.jit(categorical.mode)(jnp.zeros(10))
    jax.jit(gaussian.mode)(jnp.zeros(20))
    jax.jit(tanh_gaussian.mode)(jnp.zeros(20))

    jax.jit(gaussian.get_mean_std)(jnp.zeros(20))
    jax.jit(tanh_gaussian.get_mean_std)(jnp.zeros(20))
