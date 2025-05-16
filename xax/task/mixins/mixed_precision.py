"""Defines a mixin for mixed precision training."""

import logging
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

import jax
import jax.numpy as jnp
import jmp  # Import DeepMind's JMP library
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
    compute_gradient_stats,
    get_default_half_dtype,
    get_loss_scale_metrics,
    get_policy,
    select_tree,
    should_warn_about_precision,
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
    
    # Whether to use JMP library instead of custom implementation
    use_jmp: bool = True
    
    # Whether to log mixed precision statistics
    log_mixed_precision_stats: bool = True


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
        self._jmp_policy: Optional[jmp.Policy] = None
        self._loss_scale: Optional[LossScale] = None
        self._jmp_loss_scale: Optional[jmp.LossScale] = None
    
    @property
    def policy(self) -> Policy:
        """Get the mixed precision policy."""
        if self._policy is None:
            self._policy = get_policy(self.config.precision_policy)
            
            # Initialize JMP policy if enabled
            if self.config.use_jmp:
                self._jmp_policy = jmp.get_policy(self.config.precision_policy)
                logger.info(f"Using JMP policy with param_dtype={self._jmp_policy.param_dtype}, "
                           f"compute_dtype={self._jmp_policy.compute_dtype}, "
                           f"output_dtype={self._jmp_policy.output_dtype}")
            
        return self._policy
    
    @property
    def jmp_policy(self) -> Optional[jmp.Policy]:
        """Get the JMP policy if enabled."""
        if self._jmp_policy is None and self.config.use_jmp:
            _ = self.policy  # Initialize both policies
        return self._jmp_policy
    
    @property
    def loss_scale(self) -> LossScale:
        """Get the loss scale."""
        if self._loss_scale is None:
            if not self.config.enable_mixed_precision or self.config.loss_scaling == "none":
                self._loss_scale = NoOpLossScale()
                if self.config.use_jmp:
                    self._jmp_loss_scale = jmp.NoOpLossScale()
            elif self.config.loss_scaling == "static":
                self._loss_scale = StaticLossScale(self.config.loss_scale_value)
                if self.config.use_jmp:
                    self._jmp_loss_scale = jmp.StaticLossScale(self.config.loss_scale_value)
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
                if self.config.use_jmp:
                    # Use the correct parameter names for jmp.DynamicLossScale
                    # loss_scale: the initial scale value
                    # period: steps before increasing scale (growth_interval)
                    # factor: growth and backoff factor 
                    # min_loss_scale: minimum scale value (no direct equivalent in our API)
                    self._jmp_loss_scale = jmp.DynamicLossScale(
                        loss_scale=jnp.array(self.config.loss_scale_value, dtype=jnp.float32),
                        period=self.config.loss_scale_growth_interval,
                        factor=self.config.loss_scale_growth_factor,
                        min_loss_scale=jnp.array(1.0, dtype=jnp.float32)  # Default minimum
                    )
            else:
                raise ValueError(f"Unknown loss scaling approach: {self.config.loss_scaling}")
        return self._loss_scale
    
    @property
    def jmp_loss_scale(self) -> Optional[jmp.LossScale]:
        """Get the JMP loss scale if enabled."""
        if self._jmp_loss_scale is None and self.config.use_jmp:
            _ = self.loss_scale  # Initialize both loss scales
        return self._jmp_loss_scale
    
    def set_loss_scale(self, loss_scale: LossScale) -> None:
        """Set the loss scale.
        
        Args:
            loss_scale: The new loss scale to use.
        """
        self._loss_scale = loss_scale
    
    def set_jmp_loss_scale(self, loss_scale: jmp.LossScale) -> None:
        """Set the JMP loss scale.
        
        Args:
            loss_scale: The new JMP loss scale to use.
        """
        if self.config.use_jmp:
            self._jmp_loss_scale = loss_scale
    
    def scale_loss(self, loss: jnp.ndarray) -> jnp.ndarray:
        """Scale the loss for mixed precision training.
        
        Args:
            loss: The loss to scale.
            
        Returns:
            The scaled loss.
        """
        if not self.config.enable_mixed_precision:
            return loss
            
        if self.config.use_jmp and self.jmp_loss_scale is not None:
            return self.jmp_loss_scale.scale(loss)
        return self.loss_scale.scale(loss)
    
    def unscale_grads(self, grads: PyTree) -> PyTree:
        """Unscale gradients for mixed precision training.
        
        Args:
            grads: The gradients to unscale.
            
        Returns:
            The unscaled gradients.
        """
        if not self.config.enable_mixed_precision:
            return grads
            
        if self.config.use_jmp and self.jmp_loss_scale is not None:
            return self.jmp_loss_scale.unscale(grads)
        return self.loss_scale.unscale(grads)
    
    def check_grads_finite(self, grads: PyTree) -> jnp.ndarray:
        """Check if all gradients are finite.
        
        Args:
            grads: The gradients to check.
            
        Returns:
            Boolean indicating if all gradients are finite.
        """
        if self.config.use_jmp:
            return jmp.all_finite(grads)
        return all_finite(grads)
    
    def mixed_precision_update(
        self,
        params: PyTree,
        grads: PyTree,
        optimizer_update: Callable[[PyTree, Any], tuple[PyTree, Any]],
        optimizer_state: Any,
    ) -> tuple[PyTree, Any, Any]:
        """Update parameters with mixed precision handling.
        
        This handles skipping non-finite updates and updating the loss scale.
        
        Args:
            params: The current parameters.
            grads: The gradients to apply.
            optimizer_update: Function that applies optimizer updates.
            optimizer_state: The current optimizer state.
            
        Returns:
            Tuple of (updated_params, updated_optimizer_state, updated_loss_scale).
        """
        if not self.config.enable_mixed_precision:
            new_params, new_optimizer_state = optimizer_update(params, optimizer_state)
            return new_params, new_optimizer_state, self.loss_scale
        
        # Check if gradients are finite
        grads_finite = self.check_grads_finite(grads)
        
        # Collect gradient statistics for logging if enabled
        if self.config.log_mixed_precision_stats:
            grad_stats = compute_gradient_stats(grads)
            loss_scale_metrics = get_loss_scale_metrics(self.loss_scale)
            should_warn, warning_msg = should_warn_about_precision(
                grad_stats, 
                self.loss_scale.loss_scale, 
                self.policy.compute_dtype
            )
            
            if should_warn:
                logger.warning(f"Mixed precision warning: {warning_msg}")
                
            # Log metrics (you can integrate with your own logging system)
            if hasattr(self, 'log_metrics'):
                metrics_to_log = {}
                # Only log non-zero values to avoid cluttering logs
                for k, v in grad_stats.items():
                    if v != 0 and v != False:
                        metrics_to_log[f"mp/{k}"] = v
                for k, v in loss_scale_metrics.items():
                    metrics_to_log[f"mp/{k}"] = v
                
                # Ensure the metrics method exists and accepts these arguments
                try:
                    self.log_metrics(metrics_to_log)
                except (AttributeError, TypeError):
                    # Fallback to basic logging if the method doesn't exist or has different signature
                    for k, v in metrics_to_log.items():
                        logger.debug(f"{k}: {v}")
        
        # Adjust loss scale based on whether gradients are finite
        if self.config.use_jmp and self.jmp_loss_scale is not None:
            new_jmp_loss_scale = self.jmp_loss_scale.adjust(grads_finite)
            self.set_jmp_loss_scale(new_jmp_loss_scale)
            new_loss_scale = self.loss_scale
        else:
            new_loss_scale = self.loss_scale.adjust(grads_finite)
            self.set_loss_scale(new_loss_scale)
        
        # Skip update if the gradients are not finite
        if self.config.skip_nonfinite_updates:
            new_params, new_optimizer_state = select_tree(
                grads_finite,
                optimizer_update(params, optimizer_state),
                (params, optimizer_state),
            )
        else:
            new_params, new_optimizer_state = optimizer_update(params, optimizer_state)
        
        return new_params, new_optimizer_state, new_loss_scale
    
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
            
        if self.config.use_jmp and self.jmp_policy is not None:
            return self.jmp_policy.cast_to_compute(params)
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
            
        if self.config.use_jmp and self.jmp_policy is not None:
            return self.jmp_policy.cast_to_param(params)
        return self.policy.cast_to_param(params) 