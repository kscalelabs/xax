"""Defines some probability distribution helper functions.

In general, it is preferrable to use Distrax or another library, but we wanted
to have a simple interface of our own so that we can quickly upgrade Jax
versions (since Distrax is tied pretty closely to Tensorflow).
"""

__all__ = [
    "Distribution",
    "Categorical",
    "Normal",
    "MixtureOfGaussians",
]

import math
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


class Distribution(ABC):
    @abstractmethod
    def log_prob(self, x: Array) -> Array: ...

    @abstractmethod
    def sample(self, key: PRNGKeyArray) -> Array: ...

    @abstractmethod
    def mode(self) -> Array: ...

    @abstractmethod
    def entropy(self) -> Array: ...


class Categorical(Distribution):
    def __init__(self, logits_n: Array) -> None:
        self.logits_n = logits_n

    @property
    def num_categories(self) -> int:
        return self.logits_n.shape[-1]

    def log_prob(self, x: Array) -> Array:
        """Compute log probability for specific categories.

        Args:
            x: Array of category indices

        Returns:
            Log probabilities for the given categories
        """
        log_probs = jax.nn.log_softmax(self.logits_n, axis=-1)
        # Use advanced indexing to get the log probabilities for the given categories
        return log_probs[x]

    def sample(self, key: PRNGKeyArray) -> Array:
        return jax.random.categorical(key, self.logits_n, axis=-1)

    def mode(self) -> Array:
        return self.logits_n.argmax(axis=-1)

    def entropy(self) -> Array:
        """Compute entropy of the categorical distribution."""
        probs = jax.nn.softmax(self.logits_n, axis=-1)
        log_probs = jax.nn.log_softmax(self.logits_n, axis=-1)
        return -jnp.sum(probs * log_probs, axis=-1)


class Normal(Distribution):
    def __init__(self, loc: Array, scale: Array) -> None:
        self.loc = loc
        self.scale = scale

    def log_prob(self, x: Array) -> Array:
        return -0.5 * jnp.log(2 * jnp.pi) - jnp.log(self.scale) - (x - self.loc) ** 2 / (2 * self.scale**2)

    def sample(self, key: PRNGKeyArray) -> Array:
        return self.loc + self.scale * jax.random.normal(key, self.loc.shape)

    def mode(self) -> Array:
        return self.loc

    def entropy(self) -> Array:
        return jnp.log(2 * jnp.pi * jnp.e) + jnp.log(self.scale)


class MixtureOfGaussians(Distribution):
    def __init__(self, means_nm: Array, stds_nm: Array, logits_nm: Array) -> None:
        """Initialize a mixture of Gaussians.

        Args:
            means_nm: Array of shape (..., n_components) containing means
            stds_nm: Array of shape (..., n_components) containing standard deviations
            logits_nm: Array of shape (..., n_components) containing mixing logits
        """
        self.means_nm = means_nm
        self.stds_nm = jnp.clip(stds_nm, min=1e-6)
        self.logits_nm = jnp.clip(logits_nm, -math.log(1e4), math.log(1e4))

    def log_prob(self, x: Array) -> Array:
        """Compute log probability of the mixture.

        Args:
            x: Array of shape (...,) containing values to evaluate

        Returns:
            Log probabilities of shape (...,)
        """
        # Expand x to match component dimensions
        x_expanded = x[..., None]  # Shape: (..., 1)

        # Compute log probabilities for each component
        component_log_probs = (
            -0.5 * jnp.log(2 * jnp.pi)
            - jnp.log(self.stds_nm)
            - (x_expanded - self.means_nm) ** 2 / (2 * self.stds_nm**2)
        )

        # Compute mixing weights
        mixing_logits = jax.nn.log_softmax(self.logits_nm, axis=-1)

        # Combine using log-sum-exp trick for numerical stability
        return jax.scipy.special.logsumexp(component_log_probs + mixing_logits, axis=-1)

    def sample(self, key: PRNGKeyArray) -> Array:
        """Sample from the mixture of Gaussians.

        Args:
            key: PRNG key

        Returns:
            Samples of shape (...,) where ... are the batch dimensions
        """
        # Sample component indices
        component_key, sample_key = jax.random.split(key)
        component_indices = jax.random.categorical(component_key, self.logits_nm, axis=-1)

        # Sample from selected components using advanced indexing
        # We need to handle the case where we have batch dimensions
        batch_shape = self.means_nm.shape[:-1]  # All dimensions except the last (components)

        # Reshape for easier indexing
        means_flat = self.means_nm.reshape(-1, self.means_nm.shape[-1])
        stds_flat = self.stds_nm.reshape(-1, self.stds_nm.shape[-1])
        indices_flat = component_indices.reshape(-1)

        # Get selected means and stds
        selected_means = means_flat[jnp.arange(len(indices_flat)), indices_flat]
        selected_stds = stds_flat[jnp.arange(len(indices_flat)), indices_flat]

        # Generate random noise
        noise = jax.random.normal(sample_key, selected_means.shape)

        # Reshape back to original batch shape
        samples = selected_means + selected_stds * noise
        return samples.reshape(batch_shape)

    def mode(self) -> Array:
        """Return the mode of the mixture (approximate - returns mean of highest weight component)."""
        mixing_weights = jax.nn.softmax(self.logits_nm, axis=-1)
        max_weight_idx = jnp.argmax(mixing_weights, axis=-1)

        # Use advanced indexing to get the means of the highest weight components
        batch_shape = self.means_nm.shape[:-1]
        means_flat = self.means_nm.reshape(-1, self.means_nm.shape[-1])
        indices_flat = max_weight_idx.reshape(-1)

        selected_means = means_flat[jnp.arange(len(indices_flat)), indices_flat]
        return selected_means.reshape(batch_shape)

    def entropy(self) -> Array:
        """Compute entropy of the mixture (approximate)."""
        mixing_weights = jax.nn.softmax(self.logits_nm, axis=-1)
        component_entropies = jnp.log(2 * jnp.pi * jnp.e) + jnp.log(self.stds_nm)

        # Weighted sum of component entropies plus mixing entropy
        weighted_entropies = jnp.sum(mixing_weights * component_entropies, axis=-1)
        mixing_entropy = -jnp.sum(mixing_weights * jnp.log(mixing_weights + 1e-8), axis=-1)

        return weighted_entropies + mixing_entropy
