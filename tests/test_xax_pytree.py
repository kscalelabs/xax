"""Tests for the PyTree type and integration with JAX tree operations."""

import jax
import jax.numpy as jnp
import os
import pytest
from typing import Dict, List, Tuple, Any, cast

from xax.utils.mixed_precision import tree_map_dtype
from xax.utils.pytree import PyTree

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

if __name__ == "__main__":
    print("Running tests for XAX PyTree module...")
    
    test_pytree_type_compatibility()
    test_pytree_in_mixed_precision()
    test_large_pytree_structures()
    
    print("\n✅ All PyTree tests passed!") 