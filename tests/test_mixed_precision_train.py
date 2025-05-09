"""Test mixed precision training functionality."""

import jax
import jax.numpy as jnp
import jax.tree_util
import equinox as eqx
import optax
import pytest
import time
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any

from xax.utils.mixed_precision import (
    Policy, get_policy, NoOpLossScale, StaticLossScale, DynamicLossScale,
    get_default_half_dtype, all_finite, select_tree
)


class LinearModel(eqx.Module):
    """A simple linear model for testing."""
    weight: jnp.ndarray
    bias: jnp.ndarray
    
    def __init__(self, in_dim, out_dim, key):
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.normal(wkey, (in_dim, out_dim))
        self.bias = jax.random.normal(bkey, (out_dim,))
    
    def __call__(self, x):
        return x @ self.weight + self.bias


class TestMixedPrecisionTraining:
    """Tests for mixed precision training functionality."""
    
    @pytest.fixture
    def model_and_data(self):
        """Create a model and synthetic data for testing."""
        key = jax.random.PRNGKey(42)
        model_key, data_key = jax.random.split(key)
        
        # Create model
        in_dim, out_dim = 784, 10
        model = LinearModel(in_dim, out_dim, model_key)
        
        # Generate synthetic data
        batch_size = 64
        x_key, y_key = jax.random.split(data_key)
        x = jax.random.normal(x_key, (batch_size, in_dim))
        y = jax.random.normal(y_key, (batch_size, out_dim))
        
        return model, x, y
    
    def train_standard(self, model, x, y, n_steps=10):
        """Train with standard precision."""
        # Create optimizer
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        
        # Define loss function
        def loss_fn(model, x, y):
            pred = jax.vmap(model)(x)
            return jnp.mean((pred - y)**2)
        
        # Define training step
        @jax.jit
        def train_step(model, x, y, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(model, x, y)
            updates, new_opt_state = optimizer.update(grads, opt_state, model)
            new_model = eqx.apply_updates(model, updates)
            return new_model, new_opt_state, loss
        
        # Train
        for i in range(n_steps):
            model, opt_state, loss = train_step(model, x, y, opt_state)
        
        return model, loss
    
    def train_mixed_precision(self, model, x, y, n_steps=10):
        """Train with mixed precision."""
        # Create policy and loss scale
        policy = get_policy("mixed")  # params=float32, compute=half, output=float32
        loss_scale_value = jnp.array(2**15, dtype=jnp.float32)
        
        # Create optimizer
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        
        # Define loss function with mixed precision
        def loss_fn(model, x, y, scale_value):
            # Cast model and inputs to compute dtype
            mp_model = policy.cast_to_compute(model)
            mp_x = policy.cast_to_compute(x)
            
            # Forward pass in reduced precision
            pred = jax.vmap(mp_model)(mp_x)
            
            # Cast output back to full precision if needed
            pred = policy.cast_to_output(pred)
            
            # Compute loss
            loss = jnp.mean((pred - y)**2)
            
            # Scale loss
            scaled_loss = loss * scale_value
            
            return scaled_loss
        
        # Define training step with mixed precision
        @jax.jit
        def train_step(model, x, y, opt_state, scale_value, step_count):
            loss, grads = jax.value_and_grad(loss_fn)(model, x, y, scale_value)
            
            # Unscale gradients
            grads = jax.tree_util.tree_map(lambda g: g / scale_value, grads)
            
            # Check if gradients are finite
            grads_finite = all_finite(grads)
            
            # Update loss scale value (dynamic loss scaling logic)
            new_scale_value = jnp.where(
                ~grads_finite,  # If grads are not finite, decrease scale
                scale_value * 0.5,
                scale_value
            )
            
            # Only update when gradients are finite
            updates, new_opt_state = optimizer.update(grads, opt_state, model)
            new_model = eqx.apply_updates(model, updates)
            model = select_tree(grads_finite, new_model, model)
            opt_state = select_tree(grads_finite, new_opt_state, opt_state)
            
            # Compute true loss for reporting
            true_loss = jnp.mean((jax.vmap(model)(x) - y)**2)
            
            return model, opt_state, new_scale_value, true_loss, grads_finite
        
        # Train
        step_count = 0
        
        for i in range(n_steps):
            model, opt_state, loss_scale_value, loss, grads_finite = train_step(
                model, x, y, opt_state, loss_scale_value, jnp.array(step_count)
            )
            step_count += int(grads_finite)  # Only increment the counter for finite gradient steps
        
        return model, loss
    
    def test_training_comparison(self, model_and_data):
        """Test that standard and mixed precision training produce similar results."""
        model, x, y = model_and_data
        
        # Train with standard precision
        # Create fresh models for each run
        key = jax.random.PRNGKey(42)
        in_dim, out_dim = 784, 10
        model_std_copy = LinearModel(in_dim, out_dim, key)
        model_std, loss_std = self.train_standard(model_std_copy, x, y)
        
        # Train with mixed precision
        key = jax.random.PRNGKey(42)  # Same key for reproducibility
        model_mp_copy = LinearModel(in_dim, out_dim, key)
        model_mp, loss_mp = self.train_mixed_precision(model_mp_copy, x, y)
        
        # Test that the losses are similar (within 5%)
        relative_diff = abs(float(loss_std) - float(loss_mp)) / float(loss_std)
        assert relative_diff < 0.05, f"Losses differ too much: standard={loss_std}, mixed={loss_mp}"
        
        # Test predictions on new data
        key = jax.random.PRNGKey(99)
        test_x = jax.random.normal(key, (10, 784))
        
        # Get predictions from both models
        pred_std = jax.vmap(model_std)(test_x)
        pred_mp = jax.vmap(model_mp)(test_x)
        
        # Test that predictions are similar
        pred_diff = jnp.mean(jnp.abs(pred_std - pred_mp))
        assert pred_diff < 0.1, f"Predictions differ too much: diff={pred_diff}"
    
    def test_loss_scaling(self):
        """Test that loss scaling works correctly."""
        # Create a loss scale
        static_scale = StaticLossScale(2.0)
        dynamic_scale = DynamicLossScale(
            initial_scale=2**15,
            growth_interval=2000,
            growth_factor=2.0,
            backoff_factor=0.5,
        )
        
        # Test scaling
        loss = jnp.array(1.0, dtype=jnp.float32)
        
        # Static scaling
        scaled_loss = static_scale.scale(loss)
        assert scaled_loss == 2.0, f"Expected 2.0, got {scaled_loss}"
        
        # Dynamic scaling
        scaled_loss = dynamic_scale.scale(loss)
        assert scaled_loss == 2**15, f"Expected {2**15}, got {scaled_loss}"
        
        # Test unscaling
        grads = {"w": jnp.array(4.0), "b": jnp.array(2.0)}
        unscaled_grads = static_scale.unscale(grads)
        assert unscaled_grads["w"] == 2.0, f"Expected 2.0, got {unscaled_grads['w']}"
        assert unscaled_grads["b"] == 1.0, f"Expected 1.0, got {unscaled_grads['b']}"
        
        # Test adjustment
        grads_finite = jnp.array(True)
        grads_not_finite = jnp.array(False)
        
        # Dynamic scale should stay the same with finite grads (until growth interval)
        new_scale = dynamic_scale.adjust(grads_finite)
        assert new_scale.loss_scale == dynamic_scale.loss_scale
        
        # Dynamic scale should decrease with non-finite grads
        new_scale = dynamic_scale.adjust(grads_not_finite)
        assert new_scale.loss_scale == dynamic_scale.loss_scale * 0.5
    
    def test_precision_policy(self):
        """Test that precision policies work correctly."""
        # Create policies
        mixed_policy = get_policy("mixed")
        assert mixed_policy.param_dtype == jnp.float32
        assert mixed_policy.compute_dtype == get_default_half_dtype()
        assert mixed_policy.output_dtype == jnp.float32
        
        # Test casting
        x = jnp.ones((10, 10), dtype=jnp.float32)
        x_compute = mixed_policy.cast_to_compute(x)
        assert x_compute.dtype == get_default_half_dtype()
        
        # Create a model and test casting
        key = jax.random.PRNGKey(42)
        model = LinearModel(10, 5, key)
        
        # Cast model to compute dtype
        mp_model = mixed_policy.cast_to_compute(model)
        assert mp_model.weight.dtype == get_default_half_dtype()
        assert mp_model.bias.dtype == get_default_half_dtype()
        
        # Cast back to param dtype
        param_model = mixed_policy.cast_to_param(mp_model)
        assert param_model.weight.dtype == jnp.float32
        assert param_model.bias.dtype == jnp.float32 