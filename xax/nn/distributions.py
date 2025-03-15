"""Probability distributions implemented using the stateless class interface."""

from abc import ABC, abstractmethod

import attrs
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


@attrs.define(kw_only=True, frozen=True)
class ActionDistribution(ABC):
    """Abstract class for parametrized action distribution."""

    action_dim: int = attrs.field()
    """Shape of the distribution's output vector."""

    @property
    @abstractmethod
    def num_params(self) -> int:
        """Number of parameters of the distribution. Function of action_dim."""
        ...

    @abstractmethod
    def sample(self, parameters: Array, rng: Array) -> Array:
        """Returns a sample from the postprocessed distribution."""
        ...

    @abstractmethod
    def mode(self, parameters: Array) -> Array:
        """Returns the mode of the postprocessed distribution."""
        ...

    @abstractmethod
    def log_prob(self, parameters: Array, actions: Array) -> Array:
        """Compute the log probability of actions."""
        ...

    @abstractmethod
    def entropy(self, parameters: Array, rng: Array) -> Array:
        """Return the entropy of the given distribution.

        Note that we pass in rng because some distributions may require
        sampling to compute the entropy.
        """
        ...


@attrs.define(kw_only=True, frozen=True)
class GaussianDistribution(ActionDistribution):
    """Normal distribution."""

    def get_mean_std(self, parameters: Array) -> tuple[Array, Array]:
        """Split the parameters into the mean and standard deviation.

        Applies softplus to ensure positive std.

        Args:
            parameters: The parameters of the distribution, shape (*, 2 * action_dim).

        Returns:
            The mean and standard deviation, shape (*, action_dim).
        """
        # Validate that parameters has the expected shape
        if parameters.shape[-1] != 2 * self.action_dim:
            raise ValueError(
                f"Expected parameters with last dimension of size {2 * self.action_dim}, "
                f"but got {parameters.shape[-1]}. Make sure the parameters match the "
                f"distribution's num_params ({self.num_params})."
            )

        mean, std = jnp.split(parameters, 2, axis=-1)
        std = jax.nn.softplus(std)
        return mean, std

    @property
    def num_params(self) -> int:
        """Number of parameters of the distribution. Function of action_dim."""
        return 2 * self.action_dim

    def sample(self, parameters: Array, rng: Array) -> Array:
        """Sample from the normal distribution.

        Parameters should be the concatenation of the mean and standard
        deviation. As such, it should have shape (..., 2 * action_dim).

        Args:
            parameters: The parameters of the distribution, shape (*, 2 * action_dim).
            rng: A random number generator.

        Returns:
            The sampled actions, shape (*, action_dim).
        """
        mean, std = self.get_mean_std(parameters)
        return jax.random.normal(rng, shape=mean.shape) * std + mean

    def mode(self, parameters: Array) -> Array:
        """Returns the mode of the normal distribution.

        Parameters should be the concatenation of the mean and standard
        deviation. As such, it should have shape (..., 2 * action_dim).

        Args:
            parameters: The parameters of the distribution, shape (*, 2 * action_dim).

        Returns:
            The mode of the distribution, shape (*, action_dim).
        """
        mean, _ = self.get_mean_std(parameters)
        return mean

    def log_prob(self, parameters: Array, actions: Array) -> Array:
        """Compute the log probability of actions.

        Args:
            parameters: The parameters of the distribution, shape (*, 2 * action_dim).
            actions: The actions to compute the log probability of, shape (*, action_dim).

        Returns:
            The log probability of the actions, shape (*, action_dim).
        """
        mean, std = self.get_mean_std(parameters)
        log_probs = -0.5 * jnp.square((actions - mean) / std) - jnp.log(std) - 0.5 * jnp.log(2 * jnp.pi)
        if log_probs.shape != actions.shape:
            raise ValueError(
                f"Expected log_probs with shape {actions.shape}, but got {log_probs.shape}."
            )
        return log_probs

    def entropy(self, parameters: Array, rng: Array) -> Array:
        """Return the entropy of the normal distribution.

        Args:
            parameters: The parameters of the distribution, shape (*, 2 * action_dim).
            rng: A random number generator.

        Returns:
            The entropy of the distribution, shape (*, action_dim).
        """
        _, std = self.get_mean_std(parameters)
        entropies = 0.5 + 0.5 * jnp.log(2 * jnp.pi) + jnp.log(std)
        
        if entropies.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected entropies with last dimension {self.action_dim}, but got {entropies.shape[-1]}."
            )
        return entropies


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

    def sample(self, parameters: Array, rng: Array) -> Array:
        """Sample from the normal distribution and apply tanh.

        Parameters should be the concatenation of the mean and standard
        deviation parameters. As such, it should have shape (..., 2 * action_dim).

        Args:
            parameters: The parameters of the distribution, shape (*, 2 * action_dim).
            rng: A random number generator.

        Returns:
            The sampled actions, shape (*, action_dim).
        """
        normal_sample = super().sample(parameters, rng)
        return jnp.tanh(normal_sample)

    def mode(self, parameters: Array) -> Array:
        """Returns the mode of the normal-tanh distribution.

        For the normal distribution, the mode is the mean.
        After applying tanh, the mode is tanh(mean).

        Args:
            parameters: The parameters of the distribution, shape (*, 2 * action_dim).

        Returns:
            The mode of the distribution, shape (*, action_dim).
        """
        return jnp.tanh(super().mode(parameters))

    def log_prob(self, parameters: Array, actions: Array, eps: float = 1e-6) -> Array:
        """Compute the log probability of actions.

        This formulation computes the Gaussian log density on the pre-tanh
        values and then subtracts the Jacobian correction computed directly
        from the final actions.

        Args:
            parameters: The parameters of the distribution, shape (*, 2 * action_dim).
            actions: The actions to compute the log probability of, shape (*, action_dim).
            eps: A small epsilon value to avoid division by zero.

        Returns:
            The log probability of the actions, shape (*, action_dim).
        """
        mean, std = self.get_mean_std(parameters)

        # Compute the pre-tanh values from the actions (with clipping for stability)
        pre_tanh = jnp.arctanh(jnp.clip(actions, -1 + eps, 1 - eps))

        # Compute the base log probability from the Gaussian density (pre-tanh)
        base_log_prob = super().log_prob(parameters, pre_tanh)

        # Compute the log-determinant of the Jacobian for the tanh transformation
        # uses post-tanh actions (y vs x)
        jacobian_correction = jnp.log(1 - jnp.square(actions) + eps)

        log_probs = base_log_prob - jacobian_correction
        if log_probs.shape != actions.shape:
            raise ValueError(
                f"Expected log_probs with shape {actions.shape}, but got {log_probs.shape}."
            )
        return log_probs

    def entropy(self, parameters: Array, rng: PRNGKeyArray) -> Array:
        """Return the entropy of the normal-tanh distribution.

        Approximates entropy using sampling since there is no closed-form solution.

        The entropy of the transformed distribution is given by:
            H(Y) = H(X) + E[log|d tanh(x)/dx|],
        where H(X) is the Gaussian entropy and the Jacobian term is computed from the pre-tanh sample.

        Args:
            parameters: The parameters of the distribution, shape (*, 2 * action_dim).
            rng: A random number generator.

        Returns:
            The entropy of the distribution, shape (*, action_dim).
        """
        # base gaussian entropy
        normal_entropy = super().entropy(parameters, rng)

        # get pre-tanh sample
        normal_sample = super().sample(parameters, rng)

        # compute log det of the jacobian of tanh transformation...
        log_det_jacobian = self._log_det_jacobian(normal_sample)

        entropies = normal_entropy + log_det_jacobian

        if entropies.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected entropies with last dimension {self.action_dim}, but got {entropies.shape[-1]}."
            )
        return entropies


@attrs.define(kw_only=True, frozen=True)
class CategoricalDistribution(ActionDistribution):
    """Categorical distribution."""

    @property
    def num_params(self) -> int:
        """Number of parameters of the distribution. Function of action_dim."""
        return self.action_dim

    def sample(self, parameters: Array, rng: PRNGKeyArray) -> Array:
        """Sample from the categorical distribution. Parameters are logits.

        Parameters should have shape (..., num_actions).

        Args:
            parameters: The parameters of the distribution, shape (*, num_actions).
            rng: A random number generator.

        Returns:
            The sampled actions, shape (*).
        """
        return jax.random.categorical(rng, parameters)

    def mode(self, parameters: Array) -> Array:
        """Returns the mode of the categorical distribution.

        Parameters should have shape (..., num_actions).

        Args:
            parameters: The parameters of the distribution, shape (*, num_actions).

        Returns:
            The mode of the distribution, shape (*).
        """
        return jnp.argmax(parameters, axis=-1)

    def log_prob(self, parameters: Array, actions: Array) -> Array:
        """Compute the log probability of actions.

        Args:
            parameters: The parameters of the distribution, shape (*, num_actions).
            actions: The actions to compute the log probability of, shape (*).

        Returns:
            The log probability of the actions, shape (*).
        """
        logits = parameters
        log_probs = jax.nn.log_softmax(logits, axis=-1)

        batch_shape = actions.shape
        flat_log_probs = log_probs.reshape(-1, log_probs.shape[-1])
        flat_actions = actions.reshape(-1)
        flat_action_log_prob = flat_log_probs[jnp.arange(flat_log_probs.shape[0]), flat_actions]
        action_log_prob = flat_action_log_prob.reshape(batch_shape)

        if action_log_prob.shape != batch_shape:
            raise ValueError(
                f"Expected action_log_prob with shape {batch_shape}, but got {action_log_prob.shape}."
            )
        return action_log_prob

    def entropy(self, parameters: Array, rng: PRNGKeyArray) -> Array:
        """Return the entropy of the categorical distribution.

        Args:
            parameters: The parameters of the distribution, shape (*, num_actions).
            rng: A random number generator.

        Returns:
            The entropy of the distribution, shape (*, num_actions).
        """
        logits = parameters
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        entropies = -log_probs * jnp.exp(log_probs)

        if entropies.shape != parameters.shape:
            raise ValueError(
                f"Expected entropies with shape {parameters.shape}, but got {entropies.shape}."
            )
        return entropies
