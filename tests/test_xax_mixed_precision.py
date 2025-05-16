"""Tests for the PyTree type and integration with JAX tree operations."""

import jax
import jax.numpy as jnp
import os
import pytest
from typing import Dict, List, Tuple, Any, cast
import logging

from xax.utils.pytree import PyTree
from xax.utils.mixed_precision import (
    tree_map_dtype, 
    compute_gradient_stats, 
    get_loss_scale_metrics,
    should_warn_about_precision
)

# Check if JMP is available
try:
    import jmp
    HAS_JMP = True
except ImportError:
    HAS_JMP = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pytree_type_compatibility():
    """Test that the PyTree type is compatible with various JAX tree structures."""
    # Create various pytree structures
    tree_examples = [
        # Dictionary
        {"a": jnp.ones(3), "b": jnp.zeros(2)},
        
        # Nested dictionary
        {"outer": {"inner1": jnp.ones(3), "inner2": jnp.zeros(2)}},
        
        # List
        [jnp.ones(3), jnp.zeros(2)],
        
        # Tuple
        (jnp.ones(3), jnp.zeros(2)),
        
        # Mixed nested structure
        {"a": [jnp.ones(3), jnp.zeros(2)], "b": (jnp.ones(1), {"c": jnp.zeros(4)})},
        
        # Scalars
        {"a": 1, "b": 2.0, "c": True, "d": "string"},
        
        # With None values
        {"a": None, "b": jnp.ones(2)},
    ]
    
    # Test with jax.tree_util.tree_map
    for tree in tree_examples:
        # Explicitly cast to PyTree type to verify type compatibility
        pytree: PyTree = cast(PyTree, tree)
        
        # Test standard tree_map with a doubling function
        doubled_tree = jax.tree_util.tree_map(lambda x: x * 2 if isinstance(x, (int, float, jnp.ndarray)) else x, pytree)
        
        # Verify the structure is maintained
        jax.tree_util.tree_structure(doubled_tree)
        
        print(f"✓ JAX tree operations work with {type(tree).__name__} structure")
    
    print("✅ PyTree type is compatible with JAX tree operations")

def test_pytree_in_mixed_precision():
    """Test that PyTree is used correctly in mixed precision operations."""
    # Create a sample model-like structure
    model_tree = {
        "layer1": {
            "weight": jnp.ones((2, 2), dtype=jnp.float32),
            "bias": jnp.zeros((2,), dtype=jnp.float32)
        },
        "layer2": {
            "weight": jnp.ones((2, 2), dtype=jnp.float32),
            "bias": jnp.zeros((2,), dtype=jnp.float32)
        },
        "non_array": "metadata"
    }
    
    # Cast to PyTree type
    model: PyTree = cast(PyTree, model_tree)
    
    # Test tree_map_dtype function which uses PyTree
    fp16_model = tree_map_dtype(jnp.float16, model)
    
    # Verify dtypes were changed for arrays but structure maintained
    assert fp16_model["layer1"]["weight"].dtype == jnp.float16
    assert fp16_model["layer1"]["bias"].dtype == jnp.float16
    assert fp16_model["layer2"]["weight"].dtype == jnp.float16
    assert fp16_model["layer2"]["bias"].dtype == jnp.float16
    assert fp16_model["non_array"] == "metadata"  # Non-arrays unchanged
    
    # Test with different structures that should all be valid PyTrees
    different_pytrees = [
        # Simple array
        jnp.ones((3, 3)),
        
        # List of arrays with different shapes
        [jnp.ones((2, 2)), jnp.zeros((3, 3))],
        
        # Dictionary with arrays and scalars
        {"a": jnp.ones((2, 2)), "b": 5.0},
        
        # Deeply nested structure
        {"a": {"b": {"c": jnp.ones((2, 2))}}},
        
        # Mixed types
        {"a": jnp.ones((2, 2)), "b": [jnp.zeros(3), (jnp.ones(1), 5.0)]}
    ]
    
    for tree in different_pytrees:
        # Cast to PyTree
        test_tree: PyTree = cast(PyTree, tree)
        
        # Should work with tree_map_dtype
        result = tree_map_dtype(jnp.float16, test_tree)
        
        # Structure should be maintained
        assert jax.tree_util.tree_structure(result) == jax.tree_util.tree_structure(test_tree)
        
        print(f"✓ tree_map_dtype works with {type(tree).__name__} structure")
    
    print("✅ PyTree type works correctly with mixed precision operations")

def test_large_pytree_structures():
    """Test that PyTree works with large tree structures that might be used in ML models."""
    # Create a large nested structure representing a complex model
    large_model = {}
    
    # Create 10 layers with weights and biases
    for i in range(10):
        large_model[f"layer_{i}"] = {
            "weights": jnp.ones((8, 8), dtype=jnp.float32),
            "bias": jnp.zeros((8,), dtype=jnp.float32),
            "activation": "relu" if i % 2 == 0 else "tanh",
            "norm": {
                "scale": jnp.ones((8,), dtype=jnp.float32),
                "offset": jnp.zeros((8,), dtype=jnp.float32)
            }
        }
    
    # Add some configuration
    large_model["config"] = {
        "learning_rate": 0.001,
        "optimizer": "adam",
        "parameters": {
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8
        },
        "mixed_precision": True
    }
    
    # Cast to PyTree
    model: PyTree = cast(PyTree, large_model)
    
    # Should work with tree_map_dtype
    fp16_model = tree_map_dtype(jnp.float16, model)
    
    # Check a few values
    assert fp16_model["layer_0"]["weights"].dtype == jnp.float16
    assert fp16_model["layer_5"]["norm"]["scale"].dtype == jnp.float16
    assert fp16_model["layer_9"]["bias"].dtype == jnp.float16
    
    # Non-array values should be unchanged
    assert fp16_model["layer_0"]["activation"] == "relu"
    assert fp16_model["config"]["optimizer"] == "adam"
    assert fp16_model["config"]["mixed_precision"] is True
    
    print("✅ PyTree type works with large model structures")

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
    
    # Test warning generation
    should_warn, msg = should_warn_about_precision(
        stats, jnp.array(2**15), jnp.float16
    )
    assert isinstance(should_warn, bool)
    
    print("✅ Gradient statistics calculation works correctly")

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
    
    # Test dynamic loss scale
    dynamic_loss_scale = jmp.DynamicLossScale(
        loss_scale=jnp.array(2**10, dtype=jnp.float32),
        period=2000,
        factor=2
    )
    
    # Test scaling and adjustment
    scaled_loss = dynamic_loss_scale.scale(loss)
    
    # Test all_finite check
    all_finite = jmp.all_finite(grads)
    assert all_finite == True
    
    print("✅ JMP integration works correctly")

def test_xla_flags():
    """Test that XLA flags are properly set for mixed precision optimization."""
    xla_flags = os.environ.get("XLA_FLAGS", "")
    
    # Check that at least one of our optimization flags is set
    expected_flags = [
        "xla_gpu_enable_fast_min_max",
        "xla_gpu_enable_cublaslt",
        "xla_tpu_enable_bfloat16_mmt",
        "xla_gpu_enable_triton_softmax_fusion"
    ]
    
    # Test passes if at least one flag is present
    assert any(flag in xla_flags for flag in expected_flags), \
        f"None of the expected XLA optimization flags found in: {xla_flags}"
    
    print("✅ XLA optimization flags are properly set")

if __name__ == "__main__":
    print("Running tests for XAX PyTree and mixed precision module...")
    
    test_pytree_type_compatibility()
    test_pytree_in_mixed_precision()
    test_large_pytree_structures()
    test_gradient_stats_calculation()
    
    if HAS_JMP:
        test_jmp_integration()
    else:
        print("⚠️ Skipping JMP integration test as JMP is not installed")
    
    test_xla_flags()
    
    print("\n✅ All tests passed!") 