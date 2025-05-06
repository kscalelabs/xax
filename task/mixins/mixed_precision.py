"""Defines a mixin for mixed precision training."""

import logging
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

import jax
import jax.numpy as jnp
import optax

from xax.core.state import State
from xax.task.base import BaseTask
from xax.utils.mixed_precision import (
    DynamicLossScale,
    LossScale,
    NoOpLossScale,
    Policy,
    StaticLossScale,
    all_finite,
    get_default_half_dtype,
    get_policy,
    select_tree,
)
from xax.utils.pytree import PyTree

logger = logging.getLogger(__name__)


@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision training."""
    
    # Whether to enable mixed precision training
    enable_mixed_precision: bool = False
    
    # Mixed precision policy, can be a predefined string like "mixed", "float16", 
    # or a custom policy string like "params=float32,compute=float16,output=float32"
    precision_policy: str = "mixed"
    
    # Loss scaling approach: "none", "static", or "dynamic"
    loss_scaling: str = "dynamic"
    
    # Initial scale for loss scaling
    loss_scale_value: float = 2**15
    
    # For dynamic loss scaling: number of consecutive steps with finite gradients
    # before increasing the loss scale
    loss_scale_growth_interval: int = 2000
    
    # For dynamic loss scaling: factor by which to increase the loss scale
    loss_scale_growth_factor: float = 2.0
    
    # For dynamic loss scaling: factor by which to decrease the loss scale
    loss_scale_backoff_factor: float = 0.5
    
    # For dynamic loss scaling: maximum allowed loss scale value
    loss_scale_max_value: Optional[float] = None
    
    # Whether to skip updates that would produce non-finite gradients
    skip_nonfinite_updates: bool = True


Config = TypeVar("Config", bound=MixedPrecisionConfig)


class MixedPrecisionMixin(BaseTask[Config], Generic[Config], ABC):
    """Mixin for mixed precision training.
    
    This mixin adds mixed precision capabilities to a task, allowing models to
    train faster by using lower precision for computations while maintaining
    higher precision for parameters.
    """
    
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._policy: Optional[Policy] = None
        self._loss_scale: Optional[LossScale] = None
    
    @property
    def policy(self) -> Policy:
        """Get the mixed precision policy."""
        if self._policy is None:
            self._policy = get_policy(self.config.precision_policy)
        return self._policy
    
    @property
    def loss_scale(self) -> LossScale:
        """Get the loss scale."""
        if self._loss_scale is None:
            if not self.config.enable_mixed_precision or self.config.loss_scaling == "none":
                self._loss_scale = NoOpLossScale()
            elif self.config.loss_scaling == "static":
                self._loss_scale = StaticLossScale(self.config.loss_scale_value)
            elif self.config.loss_scaling == "dynamic":
                half_dtype = get_default_half_dtype()
                self._loss_scale = DynamicLossScale(
                    initial_scale=self.config.loss_scale_value,
                    growth_interval=self.config.loss_scale_growth_interval,
                    growth_factor=self.config.loss_scale_growth_factor,
                    backoff_factor=self.config.loss_scale_backoff_factor,
                    max_scale=self.config.loss_scale_max_value,
                    dtype=jnp.float32,  # Always use float32 for the scale itself
                )
            else:
                raise ValueError(f"Unknown loss scaling approach: {self.config.loss_scaling}")
        return self._loss_scale
    
    def set_loss_scale(self, loss_scale: LossScale) -> None:
        """Set the loss scale."""
        self._loss_scale = loss_scale
    
    def scale_loss(self, loss: jnp.ndarray) -> jnp.ndarray:
        """Scale the loss by the loss scale factor.
        
        Should be applied before computing gradients.
        
        Args:
            loss: The loss to scale.
            
        Returns:
            The scaled loss.
        """
        if not self.config.enable_mixed_precision:
            return loss
        return self.loss_scale.scale(loss)
    
    def unscale_grads(self, grads: PyTree) -> PyTree:
        """Unscale the gradients by the loss scale factor.
        
        Should be applied after computing gradients and before applying updates.
        
        Args:
            grads: The gradients to unscale.
            
        Returns:
            The unscaled gradients.
        """
        if not self.config.enable_mixed_precision:
            return grads
        return self.loss_scale.unscale(grads)
    
    def adjust_loss_scale(self, grads_finite: jnp.ndarray) -> None:
        """Adjust the loss scale based on whether gradients are finite.
        
        Args:
            grads_finite: A boolean indicating whether all gradients are finite.
        """
        if not self.config.enable_mixed_precision:
            return
        self._loss_scale = self.loss_scale.adjust(grads_finite)
    
    def check_grads_finite(self, grads: PyTree) -> jnp.ndarray:
        """Check if all gradients are finite.
        
        Args:
            grads: The gradients to check.
            
        Returns:
            A boolean indicating whether all gradients are finite.
        """
        return all_finite(grads)
    
    def should_skip_nonfinite_update(self) -> bool:
        """Check if updates with non-finite gradients should be skipped."""
        return (
            self.config.enable_mixed_precision
            and self.config.skip_nonfinite_updates
        )
    
    def cast_params_to_compute(self, params: PyTree) -> PyTree:
        """Cast parameters to computation dtype.
        
        Args:
            params: The parameters to cast.
            
        Returns:
            The cast parameters.
        """
        if not self.config.enable_mixed_precision:
            return params
        return self.policy.cast_to_compute(params)
    
    def cast_params_to_storage(self, params: PyTree) -> PyTree:
        """Cast parameters to storage dtype.
        
        Args:
            params: The parameters to cast.
            
        Returns:
            The cast parameters.
        """
        if not self.config.enable_mixed_precision:
            return params
        return self.policy.cast_to_param(params)
    
    def mixed_precision_update(
        self,
        params: PyTree,
        grads: PyTree,
        optimizer_update: Callable[[PyTree, optax.OptState], tuple[PyTree, optax.OptState]],
        opt_state: optax.OptState,
    ) -> tuple[PyTree, optax.OptState, LossScale]:
        """Apply an update with mixed precision considerations.
        
        This method handles gradient unscaling, checking for finite values,
        and conditionally applying updates.
        
        Args:
            params: The current parameters.
            grads: The computed gradients.
            optimizer_update: A function that applies optimizer updates.
            opt_state: The current optimizer state.
            
        Returns:
            A tuple containing the updated parameters, optimizer state, and loss scale.
        """
        if not self.config.enable_mixed_precision:
            # Simply apply the update
            new_params, new_opt_state = optimizer_update(params, opt_state)
            return new_params, new_opt_state, self.loss_scale
        
        # Unscale gradients
        unscaled_grads = self.unscale_grads(grads)
        
        # Check if gradients are finite
        grads_finite = self.check_grads_finite(unscaled_grads)
        
        # Adjust loss scale
        new_loss_scale = self.loss_scale.adjust(grads_finite)
        
        if self.should_skip_nonfinite_update():
            # Conditionally apply the update
            new_params, new_opt_state = optimizer_update(params, opt_state)
            selected_params = select_tree(grads_finite, new_params, params)
            return selected_params, opt_state, new_loss_scale
        else:
            # Always apply the update
            new_params, new_opt_state = optimizer_update(params, opt_state)
            return new_params, new_opt_state, new_loss_scale 