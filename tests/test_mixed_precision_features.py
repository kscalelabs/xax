"""Tests for the enhanced mixed precision features in XAX."""

import jax
import jax.numpy as jnp
import os
import pytest
from typing import Dict, Any
import logging

from xax.utils.mixed_precision import (
    tree_map_dtype, 
    compute_gradient_stats, 
    get_loss_scale_metrics,
    should_warn_about_precision,
    Policy,
    DynamicLossScale,
    StaticLossScale,
    NoOpLossScale
)
from xax.utils.pytree import PyTree

# Check if JMP is available
try:
    import jmp
    HAS_JMP = True
except ImportError:
    HAS_JMP = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_xla_flags():
    """Test that XLA flags are properly set for mixed precision optimization."""
    xla_flags = os.environ.get("XLA_FLAGS", "")
    
    # Check that at least one of our optimization flags is set
    expected_flags = [
        "xla_gpu_enable_fast_min_max",
        "xla_gpu_enable_cublaslt",
    ]
    
    # Test passes if at least one flag is present
    assert any(flag in xla_flags for flag in expected_flags), \
        f"None of the expected XLA optimization flags found in: {xla_flags}"
    
    print("✅ XLA optimization flags are properly set")

def test_gradient_stats_calculation():
    """Test the gradient statistics calculation functionality."""
    # Create a sample gradient tree
    grads = {
        "layer1": {
            "weight": jnp.ones((10, 10), dtype=jnp.float32),
            "bias": jnp.zeros((10,), dtype=jnp.float32) 
        },
        "layer2": {
            "weight": 2 * jnp.ones((5, 5), dtype=jnp.float32),
            "bias": jnp.array([1e-10, 1e-8, 1e-6, 1e-4], dtype=jnp.float32)
        }
    }
    
    # Calculate gradient statistics
    stats = compute_gradient_stats(grads)
    
    # Check that all expected keys are present
    expected_keys = ["grad_norm", "max_abs_grad", "min_abs_grad_nonzero", 
                     "has_nan", "has_inf", "finite_ratio"]
    for key in expected_keys:
        assert key in stats, f"Missing key {key} in gradient statistics"
    
    # Basic validation of values
    assert stats["has_nan"] == False
    assert stats["has_inf"] == False
    assert stats["finite_ratio"] == 1.0
    assert stats["max_abs_grad"] == 2.0
    assert stats["grad_norm"] > 0
    
    # Test with some NaN/Inf values
    grads_with_nan = {
        "layer1": {
            "weight": jnp.ones((10, 10), dtype=jnp.float32),
            "bias": jnp.array([jnp.nan, 0.0, 1.0], dtype=jnp.float32)
        }
    }
    
    stats_nan = compute_gradient_stats(grads_with_nan)
    assert stats_nan["has_nan"] == True
    assert stats_nan["finite_ratio"] < 1.0
    
    print("✅ Gradient statistics calculation works correctly")

def test_loss_scale_metrics():
    """Test the collection of loss scale metrics."""
    # Test static loss scale
    static_scale = StaticLossScale(128.0)
    static_metrics = get_loss_scale_metrics(static_scale)
    
    assert "loss_scale_value" in static_metrics
    assert static_metrics["loss_scale_value"] == 128.0
    
    # Test dynamic loss scale
    dynamic_scale = DynamicLossScale(
        initial_scale=1024.0,
        growth_interval=2000,
        growth_factor=2.0,
        backoff_factor=0.5
    )
    
    dynamic_metrics = get_loss_scale_metrics(dynamic_scale)
    assert "loss_scale_value" in dynamic_metrics
    assert "loss_scale_growth_interval" in dynamic_metrics
    assert "loss_scale_growth_factor" in dynamic_metrics
    assert "loss_scale_backoff_factor" in dynamic_metrics
    assert "loss_scale_steps" in dynamic_metrics
    
    assert dynamic_metrics["loss_scale_value"] == 1024.0
    assert dynamic_metrics["loss_scale_growth_interval"] == 2000
    assert dynamic_metrics["loss_scale_growth_factor"] == 2.0
    assert dynamic_metrics["loss_scale_backoff_factor"] == 0.5
    
    print("✅ Loss scale metrics collection works correctly")

def test_precision_warnings():
    """Test the precision warning system."""
    # Create gradient statistics that should trigger a warning
    stats_overflow = {
        "grad_norm": jnp.array(1000.0),
        "max_abs_grad": jnp.array(1000.0),
        "min_abs_grad_nonzero": jnp.array(0.001),
        "has_nan": False,
        "has_inf": False,
        "finite_ratio": jnp.array(1.0)
    }
    
    # Check overflow warning
    should_warn, msg = should_warn_about_precision(
        stats_overflow, jnp.array(2**15), jnp.float16
    )
    assert should_warn == True
    assert "might be too high" in msg
    
    # Create gradient statistics that should trigger an underflow warning
    stats_underflow = {
        "grad_norm": jnp.array(0.1),
        "max_abs_grad": jnp.array(0.1),
        "min_abs_grad_nonzero": jnp.array(1e-8),
        "has_nan": False,
        "has_inf": False,
        "finite_ratio": jnp.array(1.0)
    }
    
    # Check underflow warning
    should_warn, msg = should_warn_about_precision(
        stats_underflow, jnp.array(128.0), jnp.float16
    )
    assert should_warn == True
    assert "Very small gradient values detected" in msg
    
    # Create normal gradient statistics that shouldn't trigger a warning
    stats_normal = {
        "grad_norm": jnp.array(1.0),
        "max_abs_grad": jnp.array(1.0),
        "min_abs_grad_nonzero": jnp.array(0.001),
        "has_nan": False,
        "has_inf": False,
        "finite_ratio": jnp.array(1.0)
    }
    
    # Check no warning
    should_warn, msg = should_warn_about_precision(
        stats_normal, jnp.array(128.0), jnp.float16
    )
    assert should_warn == False
    assert msg == ""
    
    print("✅ Precision warning system works correctly")

@pytest.mark.skipif(not HAS_JMP, reason="JMP not installed")
def test_jmp_integration():
    """Test the integration with Google DeepMind's JMP library."""
    if not HAS_JMP:
        print("⚠️ Skipping JMP integration test as JMP is not installed")
        return
    
    # Create a JMP policy
    policy = jmp.Policy(
        param_dtype=jnp.float32,
        compute_dtype=jnp.bfloat16,
        output_dtype=jnp.float32
    )
    
    # Create a simple model
    model = {
        "weight": jnp.ones((32, 32), dtype=jnp.float32),
        "bias": jnp.zeros((32,), dtype=jnp.float32)
    }
    
    # Cast to compute dtype
    compute_model = policy.cast_to_compute(model)
    
    # Check that weights are cast to bfloat16
    assert compute_model["weight"].dtype == jnp.bfloat16
    assert compute_model["bias"].dtype == jnp.bfloat16
    
    # Cast back to param dtype
    param_model = policy.cast_to_param(compute_model)
    
    # Check that weights are cast back to float32
    assert param_model["weight"].dtype == jnp.float32
    assert param_model["bias"].dtype == jnp.float32
    
    # Test loss scaling with JMP
    loss = jnp.array(2.0, dtype=jnp.float32)
    
    # Static loss scale
    loss_scale = jmp.StaticLossScale(128.0)
    scaled_loss = loss_scale.scale(loss)
    assert scaled_loss == 256.0  # 2.0 * 128.0
    
    # Create gradients and test unscaling
    grads = {"weight": jnp.ones((32, 32)) * 128.0}
    unscaled_grads = loss_scale.unscale(grads)
    assert jnp.allclose(unscaled_grads["weight"], jnp.ones((32, 32)))
    
    # Test dynamic loss scale - use the correct parameter names for JMP
    dynamic_loss_scale = jmp.DynamicLossScale(
        loss_scale=jnp.array(2**10, dtype=jnp.float32),
        period=2000,
        factor=2
    )
    
    # Test scaling
    scaled_loss = dynamic_loss_scale.scale(loss)
    
    # Test all_finite check
    all_finite = jmp.all_finite(grads)
    assert all_finite == True
    
    # Test string-based policy creation (similar to XAX's get_policy)
    string_policy = jmp.get_policy("params=float32,compute=bfloat16,output=float32")
    assert string_policy.param_dtype == jnp.float32
    assert string_policy.compute_dtype == jnp.bfloat16
    assert string_policy.output_dtype == jnp.float32
    
    # JMP doesn't support predefined "mixed" policy, so we use a basic one
    basic_policy = jmp.get_policy("float32")
    assert basic_policy.param_dtype == jnp.float32
    
    print("✅ JMP integration works correctly")

def test_custom_vs_jmp_api_compatibility():
    """Test that our custom API is compatible with JMP's API."""
    if not HAS_JMP:
        print("⚠️ Skipping API compatibility test as JMP is not installed")
        return
    
    # Create policies with both APIs
    custom_mixed_policy = Policy(
        param_dtype=jnp.float32,
        compute_dtype=jnp.bfloat16,
        output_dtype=jnp.float32
    )
    
    jmp_mixed_policy = jmp.Policy(
        param_dtype=jnp.float32,
        compute_dtype=jnp.bfloat16,
        output_dtype=jnp.float32
    )
    
    # Check policy attributes
    assert custom_mixed_policy.param_dtype == jmp_mixed_policy.param_dtype
    assert custom_mixed_policy.compute_dtype == jmp_mixed_policy.compute_dtype
    assert custom_mixed_policy.output_dtype == jmp_mixed_policy.output_dtype
    
    # Create a simple model for testing
    model = {
        "weight": jnp.ones((32, 32), dtype=jnp.float32),
        "bias": jnp.zeros((32,), dtype=jnp.float32)
    }
    
    # Cast to compute dtype with both APIs
    custom_compute_model = custom_mixed_policy.cast_to_compute(model)
    jmp_compute_model = jmp_mixed_policy.cast_to_compute(model)
    
    # Check results are consistent
    assert custom_compute_model["weight"].dtype == jmp_compute_model["weight"].dtype
    assert custom_compute_model["bias"].dtype == jmp_compute_model["bias"].dtype
    
    # Cast back to param dtype with both APIs
    custom_param_model = custom_mixed_policy.cast_to_param(custom_compute_model)
    jmp_param_model = jmp_mixed_policy.cast_to_param(jmp_compute_model)
    
    # Check results are consistent
    assert custom_param_model["weight"].dtype == jmp_param_model["weight"].dtype
    assert custom_param_model["bias"].dtype == jmp_param_model["bias"].dtype
    
    # Test loss scaling with both APIs
    loss = jnp.array(2.0, dtype=jnp.float32)
    
    # Static loss scale
    custom_loss_scale = StaticLossScale(128.0)
    jmp_loss_scale = jmp.StaticLossScale(128.0)
    
    custom_scaled_loss = custom_loss_scale.scale(loss)
    jmp_scaled_loss = jmp_loss_scale.scale(loss)
    
    assert custom_scaled_loss == jmp_scaled_loss
    
    # Create gradients and test unscaling
    grads = {"weight": jnp.ones((32, 32)) * 128.0}
    custom_unscaled_grads = custom_loss_scale.unscale(grads)
    jmp_unscaled_grads = jmp_loss_scale.unscale(grads)
    
    assert jnp.allclose(custom_unscaled_grads["weight"], jmp_unscaled_grads["weight"])
    
    # Note: The APIs for DynamicLossScale are different, but we verify basic functionality
    custom_dynamic = DynamicLossScale(initial_scale=128.0, growth_interval=2000)
    jmp_dynamic = jmp.DynamicLossScale(loss_scale=jnp.array(128.0, dtype=jnp.float32), period=2000)
    
    # Test that loss scaling works the same
    custom_dyn_scaled = custom_dynamic.scale(loss)
    jmp_dyn_scaled = jmp_dynamic.scale(loss)
    assert jnp.allclose(custom_dyn_scaled, jmp_dyn_scaled)
    
    print("✅ Custom API is compatible with JMP API")

if __name__ == "__main__":
    print("Running tests for enhanced mixed precision features...")
    
    test_xla_flags()
    test_gradient_stats_calculation()
    test_loss_scale_metrics()
    test_precision_warnings()
    
    if HAS_JMP:
        test_jmp_integration()
        test_custom_vs_jmp_api_compatibility()
    else:
        print("⚠️ Skipping JMP tests as JMP is not installed (run: pip install git+https://github.com/google-deepmind/jmp)")
    
    print("\n✅ All tests passed!") 