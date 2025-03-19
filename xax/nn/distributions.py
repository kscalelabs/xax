"""Probability distributions implemented using the stateless class interface."""

from abc import ABC, abstractmethod

import attrs
import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


@attrs.define(kw_only=True, frozen=True)
class Distribution(ABC):
    """Abstract class for parametrized action distribution."""

    @property
    @abstractmethod
    def num_params(self) -> int:
        """Number of parameters of the distribution. Function of action_dim."""

    @abstractmethod
    def sample(self, rng: Array) -> Array:
        """Returns a sample from the postprocessed distribution."""

    @abstractmethod
    def mode(self) -> Array:
        """Returns the mode of the postprocessed distribution."""

    @abstractmethod
    def log_prob(self, actions: Array) -> Array:
        """Compute the log probability of actions."""

    @abstractmethod
    def entropy(self, rng: Array) -> Array:
        """Return the entropy of the given distribution.

        Note that we pass in rng because some distributions may require
        sampling to compute the entropy.
        """


@attrs.define(kw_only=True, frozen=True)
class GaussianDistribution(Distribution):
    """Normal distribution."""

    mean: Array = attrs.field()
    std: Array = attrs.field()

    @property
    def num_params(self) -> int:
        return self.mean.shape[-1]

    def entropy(self, rng: Array) -> Array:
        """Return the entropy of the normal distribution.

        Args:
            rng: A random number generator.

        Returns:
            The entropy of the distribution, shape (*, action_dim).
        """
        log_normalization = 0.5 * jnp.log(2 * jnp.pi) + jnp.log(self.std)
        entropies = 0.5 + log_normalization
        return entropies

    def sample(self, rng: Array) -> Array:
        """Sample from the normal distribution.

        Args:
            parameters: The parameters of the distribution, shape (*, 2 * action_dim).
            rng: A random number generator.

        Returns:
            The sampled actions, shape (*, action_dim).
        """
        return jax.random.normal(rng, shape=self.mean.shape) * self.std + self.mean

    def mode(self) -> Array:
        """Returns the mode of the normal distribution.

        Returns:
            The mode of the distribution, shape (*, action_dim).
        """
        return self.mean

    def log_prob(self, actions: Array) -> Array:
        """Compute the log probability of actions.

        Args:
            actions: The actions to compute the log probability of, shape (*, action_dim).

        Returns:
            The log probability of the actions, shape (*, action_dim).
        """
        log_probs = -0.5 * jnp.square((actions - self.mean) / self.std) - jnp.log(self.std) - 0.5 * jnp.log(2 * jnp.pi)
        chex.assert_shape(log_probs, actions.shape)
        return log_probs


@attrs.define(kw_only=True, frozen=True)
class TanhGaussianDistribution(GaussianDistribution):
    """Normal distribution followed by tanh."""

    def _log_det_jacobian(self, actions: Array) -> Array:
        """Compute the log determinant of the jacobian of the tanh transform.

        $p(x) = p(y) * |dy/dx| = p(tanh(x)) * |1 - tanh(x)^2|$

        Args:
            actions: The actions to compute the log determinant of the
                Jacobian of the tanh transform, shape (*, action_dim). Should be
                pre-tanh actions when used for change of variables.

        Returns:
            The log determinant of the jacobian of the tanh transform, shape (*).
        """
        return 2.0 * (jnp.log(2.0) - actions - jax.nn.softplus(-2.0 * actions))

    def sample(self, rng: Array) -> Array:
        """Sample from the normal distribution and apply tanh.

        Args:
            rng: A random number generator.

        Returns:
            The sampled actions, shape (*, action_dim).
        """
        normal_sample = super().sample(rng)
        return jnp.tanh(normal_sample)

    def mode(self) -> Array:
        """Returns the mode of the normal-tanh distribution.

        For the normal distribution, the mode is the mean.
        After applying tanh, the mode is tanh(mean).

        Returns:
            The mode of the distribution, shape (*, action_dim).
        """
        return jnp.tanh(self.mean)

    def log_prob(self, actions: Array, eps: float = 1e-6) -> Array:
        """Compute the log probability of actions.

        This formulation computes the Gaussian log density on the pre-tanh
        values and then subtracts the Jacobian correction computed directly
        from the final actions.

        Args:
            actions: The actions to compute the log probability of, shape (*, action_dim).
            eps: A small epsilon value to avoid division by zero.

        Returns:
            The log probability of the actions, shape (*, action_dim).
        """
        # Compute the pre-tanh values from the actions (with clipping for stability)
        pre_tanh = jnp.arctanh(jnp.clip(actions, -1 + eps, 1 - eps))

        # Compute the base log probability from the Gaussian density (pre-tanh)
        base_log_prob = super().log_prob(pre_tanh)

        # Compute the log-determinant of the Jacobian for the tanh transformation
        # uses post-tanh actions (y vs x)
        jacobian_correction = jnp.log(1 - jnp.square(actions) + eps)

        log_probs = base_log_prob - jacobian_correction
        chex.assert_shape(log_probs, actions.shape)
        return log_probs

    def entropy(self, rng: PRNGKeyArray) -> Array:
        """Return the entropy of the normal-tanh distribution.

        Approximates entropy using sampling since there is no closed-form solution.

        The entropy of the transformed distribution is given by:
            H(Y) = H(X) + E[log|d tanh(x)/dx|],
        where H(X) is the Gaussian entropy and the Jacobian term is computed from the pre-tanh sample.

        Args:
            rng: A random number generator.

        Returns:
            The entropy of the distribution, shape (*, action_dim).
        """
        # Base Gaussian entropy.
        normal_entropy = super().entropy(rng)

        # Get the pre-tanh sample.
        normal_sample = super().sample(rng)

        # Compute log determinant of the Jacobian of tanh transformation.
        log_det_jacobian = self._log_det_jacobian(normal_sample)
        entropies = normal_entropy + log_det_jacobian
        chex.assert_shape(entropies, (..., self.num_params))
        return entropies


@attrs.define(kw_only=True, frozen=True)
class CategoricalDistribution(Distribution):
    """Categorical distribution."""

    logits: Array = attrs.field()

    @property
    def num_params(self) -> int:
        """Number of parameters of the distribution. Function of action_dim."""
        return self.logits.shape[-1]

    def sample(self, rng: PRNGKeyArray) -> Array:
        """Sample from the categorical distribution. Parameters are logits.

        Args:
            parameters: The parameters of the distribution, shape (*, num_actions).
            rng: A random number generator.

        Returns:
            The sampled actions, shape (*).
        """
        return jax.random.categorical(rng, self.logits)

    def mode(self) -> Array:
        """Returns the mode of the categorical distribution.

        Returns:
            The mode of the distribution, shape (*).
        """
        return jnp.argmax(self.logits, axis=-1)

    def log_prob(self, actions: Array) -> Array:
        """Compute the log probability of actions.

        Args:
            actions: The actions to compute the log probability of, shape (*).

        Returns:
            The log probability of the actions, shape (*).
        """
        chex.assert_type(actions, jnp.int32)
        log_probs = jax.nn.log_softmax(self.logits, axis=-1)
        action_log_probs = log_probs[actions]
        return action_log_probs

    def entropy(self, rng: PRNGKeyArray) -> Array:
        """Return the entropy of the categorical distribution.

        Args:
            rng: A random number generator.

        Returns:
            The entropy of the distribution, shape (*, num_actions).
        """
        log_probs = jax.nn.log_softmax(self.logits, axis=-1)
        entropies = -log_probs * jnp.exp(log_probs)
        chex.assert_shape(entropies, self.logits.shape)
        return entropies
