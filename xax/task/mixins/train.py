"""Training mixin for XAX tasks."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Optional, Protocol, Tuple, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import optax


@dataclass
class TrainConfig:
    """Configuration options for training."""
    seed: int = 42
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 10


T = TypeVar("T", bound=TrainConfig)


class TrainMixin(Generic[T]):
    """A mixin that adds training capabilities to a task."""
    
    config: T
    model: Any
    optimizer: optax.GradientTransformation
    opt_state: optax.OptState
    
    def train_step(self, model: Any, batch: Any) -> Tuple[Any, optax.OptState, Any]:
        """Perform a single training step.
        
        Args:
            model: The model to train
            batch: The batch of data to use for training
            
        Returns:
            A tuple of (loss, new_opt_state, new_model)
        """
        # Get the loss and gradients
        (loss, aux), grads = self._loss_and_grad(model, batch)
        
        # Update the model parameters using the optimizer
        updates, new_opt_state = self.optimizer.update(grads, self.opt_state)
        new_model = eqx.apply_updates(model, updates)
        
        return loss, new_opt_state, new_model
    
    def _loss_and_grad(self, model: Any, batch: Any) -> Tuple[Tuple[Any, Any], Any]:
        """Compute the loss and gradients for the given model and batch.
        
        Args:
            model: The model to compute gradients for
            batch: The batch of data to use
            
        Returns:
            A tuple of ((loss, aux), gradients)
        """
        loss_fn = lambda m: self.get_output_and_loss(m, batch)
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(model)
        return (loss, aux), grads
    
    def get_output_and_loss(self, model: Any, batch: Any, train: bool = True) -> Tuple[Any, Any]:
        """Get the output and loss for the given model and batch.
        
        Args:
            model: The model to compute output for
            batch: The batch of data to use
            train: Whether we're in training mode
            
        Returns:
            A tuple of (loss, aux)
        """
        raise NotImplementedError("Subclasses must implement get_output_and_loss") 