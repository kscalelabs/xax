"""Tests for the mixed precision training utilities."""

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

from xax.utils.mixed_precision import (
    Policy,
    get_policy as parse_policy_str,
    LossScale,
    StaticLossScale,
    DynamicLossScale,
    NoOpLossScale as NoLossScale,
    tree_map_dtype,
    all_finite,
    select_tree,
)
from xax.task.mixins.mixed_precision import MixedPrecisionConfig, MixedPrecisionMixin
from xax.task.mixins.train import TrainConfig, TrainMixin
import optax


# Simple linear model for testing
class LinearModel(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, in_size, out_size, key):
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.normal(wkey, (out_size, in_size))
        self.bias = jax.random.normal(bkey, (out_size,))

    def __call__(self, x):
        return self.weight @ x + self.bias


@dataclass
class TestConfig(TrainConfig, MixedPrecisionConfig):
    """Test configuration with mixed precision settings."""
    pass


class TestTask(TrainMixin[TestConfig], MixedPrecisionMixin[TestConfig]):
    """Test task with mixed precision support."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        key = jax.random.PRNGKey(config.seed)
        self.model = LinearModel(10, 1, key)
        self._policy = None
        self._loss_scale = None
        self.optimizer = self.get_optimizer()
        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))
    
    def get_output_and_loss(self, model, batch, train=True):
        x, y = batch
        output = model(x)
        loss = jnp.mean((output - y) ** 2)
        return loss, output

    def _loss_and_grad(self, model: Any, batch: Any) -> Tuple[Tuple[Any, Any], Any]:
        """Override to properly pass train parameter."""
        loss_fn = lambda m: self.get_output_and_loss(m, batch, train=True)
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(model)
        return (loss, aux), grads

    def get_batch(self):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (10,))
        y = jnp.array([1.0])
        return (x, y)
    
    def get_optimizer(self):
        return optax.sgd(self.config.learning_rate)
    
    def get_opt_state(self, model):
        return self.opt_state


class TestPolicies:
    """Tests for precision policies."""

    def test_parse_policy_str(self):
        # Test predefined policies
        assert parse_policy_str("default") == Policy(jnp.float32, jnp.float32, jnp.float32)
        assert parse_policy_str("mixed") == Policy(jnp.float32, 
                                                  jnp.bfloat16 if jax.default_backend() == "tpu" else jnp.float16, 
                                                  jnp.float32)
        assert parse_policy_str("float16") == Policy(jnp.float16, jnp.float16, jnp.float16)
        
        # Test custom policy
        custom = parse_policy_str("params=float32,compute=float16,output=bfloat16")
        assert custom.param_dtype == jnp.float32
        assert custom.compute_dtype == jnp.float16
        assert custom.output_dtype == jnp.bfloat16

    def test_tree_map_dtype(self):
        # Create a parameter tree
        params = {
            "layer1": {
                "weight": jnp.ones((2, 2), dtype=jnp.float32),
                "bias": jnp.zeros((2,), dtype=jnp.float32)
            },
            "layer2": jnp.ones((2, 2), dtype=jnp.float32)
        }
        
        # Convert to float16
        params_f16 = tree_map_dtype(jnp.float16, params)
        
        # Check dtypes
        assert params["layer1"]["weight"].dtype == jnp.float32
        assert params_f16["layer1"]["weight"].dtype == jnp.float16
        assert params_f16["layer1"]["bias"].dtype == jnp.float16
        assert params_f16["layer2"].dtype == jnp.float16

    def test_policy_casting(self):
        policy = Policy(jnp.float32, jnp.float16, jnp.bfloat16)
        
        # Create parameters
        params = {
            "weight": jnp.ones((2, 2), dtype=jnp.float32),
            "bias": jnp.zeros((2,), dtype=jnp.float32)
        }
        
        # Test casting to compute
        params_compute = policy.cast_to_compute(params)
        assert params_compute["weight"].dtype == jnp.float16
        assert params_compute["bias"].dtype == jnp.float16
        
        # Test casting to param
        params_mixed = {
            "weight": jnp.ones((2, 2), dtype=jnp.float16),
            "bias": jnp.zeros((2,), dtype=jnp.float16)
        }
        params_restored = policy.cast_to_param(params_mixed)
        assert params_restored["weight"].dtype == jnp.float32
        assert params_restored["bias"].dtype == jnp.float32
        
        # Test casting to output
        output = jnp.ones((2, 2), dtype=jnp.float16)
        output_cast = policy.cast_to_output(output)
        assert output_cast.dtype == jnp.bfloat16


class TestLossScaling:
    """Tests for loss scaling utilities."""

    def test_no_loss_scale(self):
        scale = NoLossScale()
        loss = 2.0
        
        # Test scaling (should be identity)
        scaled_loss = scale.scale(loss)
        assert scaled_loss == loss
        
        # Test unscaling (should be identity)
        grads = {"weight": jnp.ones((2, 2)), "bias": jnp.zeros(2)}
        unscaled_grads = scale.unscale(grads)
        assert jnp.array_equal(unscaled_grads["weight"], grads["weight"])
        
        # Test adjust (should be identity)
        new_scale = scale.adjust(True)
        assert new_scale == scale

    def test_static_loss_scale(self):
        scale_value = 128.0
        scale = StaticLossScale(scale_value)
        loss = 2.0
        
        # Test scaling
        scaled_loss = scale.scale(loss)
        assert scaled_loss == loss * scale_value
        
        # Test unscaling
        grads = {"weight": jnp.ones((2, 2)) * 128.0, "bias": jnp.zeros(2)}
        unscaled_grads = scale.unscale(grads)
        assert jnp.array_equal(unscaled_grads["weight"], jnp.ones((2, 2)))
        
        # Test adjust (should be identity for static)
        new_scale = scale.adjust(True)
        assert new_scale.loss_scale == scale.loss_scale

    def test_dynamic_loss_scale(self):
        scale = DynamicLossScale(
            initial_scale=4.0,
            growth_interval=2,
            growth_factor=2.0,
            backoff_factor=0.5
        )
        
        # Test initial value
        assert scale.loss_scale == 4.0
        assert scale._steps_since_finite == 0
        
        # Test adjust with finite gradients
        new_scale1 = scale.adjust(True)
        assert new_scale1.loss_scale == 4.0
        assert new_scale1._steps_since_finite == 1
        
        # Test growth after interval
        new_scale2 = new_scale1.adjust(True)
        assert new_scale2.loss_scale == 8.0  # Doubled after 2 steps
        assert new_scale2._steps_since_finite == 0  # Reset counter
        
        # Test backoff with non-finite gradients
        new_scale3 = new_scale2.adjust(False)
        assert new_scale3.loss_scale == 4.0  # Halved
        assert new_scale3._steps_since_finite == 0  # Reset counter


class TestMixedPrecisionMixin:
    """Tests for MixedPrecisionMixin."""

    def test_create_policy(self):
        # Test default policy
        config = TestConfig(
            enable_mixed_precision=True,
            precision_policy="default",
            loss_scaling="none",
            loss_scale_value=1.0
        )
        task = TestTask(config)
        policy = task.policy
        assert policy.param_dtype == jnp.float32
        assert policy.compute_dtype == jnp.float32
        assert policy.output_dtype == jnp.float32
        
        # Test mixed policy
        config = TestConfig(
            enable_mixed_precision=True,
            precision_policy="mixed",
            loss_scaling="none",
            loss_scale_value=1.0
        )
        task = TestTask(config)
        policy = task.policy
        assert policy.param_dtype == jnp.float32
        assert policy.compute_dtype == (jnp.bfloat16 if jax.default_backend() == "tpu" else jnp.float16)
        assert policy.output_dtype == jnp.float32
        
        # Test with disabled mixed precision
        config = TestConfig(
            enable_mixed_precision=False,
            precision_policy="mixed",
            loss_scaling="none",
            loss_scale_value=1.0
        )
        task = TestTask(config)
        policy = task.policy
        # Policy is still created but not used when disabled
        assert policy.param_dtype == jnp.float32
        assert policy.compute_dtype == (jnp.bfloat16 if jax.default_backend() == "tpu" else jnp.float16)
        assert policy.output_dtype == jnp.float32

    def test_create_loss_scale(self):
        # Test no loss scaling
        config = TestConfig(
            enable_mixed_precision=True,
            precision_policy="mixed",
            loss_scaling="none",
            loss_scale_value=1.0
        )
        task = TestTask(config)
        assert isinstance(task.loss_scale, NoLossScale)
        
        # Test static loss scaling
        config = TestConfig(
            enable_mixed_precision=True,
            precision_policy="mixed",
            loss_scaling="static",
            loss_scale_value=128.0
        )
        task = TestTask(config)
        assert isinstance(task.loss_scale, StaticLossScale)
        assert task.loss_scale.loss_scale == 128.0
        
        # Test dynamic loss scaling
        config = TestConfig(
            enable_mixed_precision=True,
            precision_policy="mixed",
            loss_scaling="dynamic",
            loss_scale_value=16.0,
            loss_scale_growth_interval=1000,
            loss_scale_growth_factor=2.0,
            loss_scale_backoff_factor=0.5
        )
        task = TestTask(config)
        assert isinstance(task.loss_scale, DynamicLossScale)
        assert task.loss_scale.loss_scale == 16.0
        assert task.loss_scale._growth_interval == 1000

    def test_mixed_precision_methods(self):
        config = TestConfig(
            enable_mixed_precision=True,
            precision_policy="mixed",
            loss_scaling="static",
            loss_scale_value=2.0
        )
        task = TestTask(config)
        
        # Test parameter casting
        model = task.model
        compute_model = task.cast_params_to_compute(model)
        
        # Model weights should be in compute dtype
        half_dtype = jnp.bfloat16 if jax.default_backend() == "tpu" else jnp.float16
        assert eqx.filter_jit(lambda m: m.weight.dtype)(compute_model) == half_dtype
        
        # Test loss scaling
        loss = 1.0
        scaled_loss = task.scale_loss(loss)
        assert scaled_loss == 2.0
        
        # Test gradient unscaling
        grads = {"weight": jnp.ones((1, 10)) * 2.0, "bias": jnp.ones(1) * 2.0}
        unscaled_grads = task.unscale_grads(grads)
        assert jnp.array_equal(unscaled_grads["weight"], jnp.ones((1, 10)))

    def test_train_with_mixed_precision(self):
        config = TestConfig(
            enable_mixed_precision=True,
            precision_policy="mixed",
            loss_scaling="static",
            loss_scale_value=2.0,
            learning_rate=0.01,
            seed=42
        )
        task = TestTask(config)
        
        # Initial model should be in param dtype
        assert eqx.filter_jit(lambda m: m.weight.dtype)(task.model) == jnp.float32
        
        # Run a training step
        batch = task.get_batch()
        loss, metrics, model = task.train_step(task.model, batch)
        
        # Model should still be in param dtype after training
        assert eqx.filter_jit(lambda m: m.weight.dtype)(model) == jnp.float32
        
        # Loss and metrics should be finite
        assert jnp.isfinite(loss)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 