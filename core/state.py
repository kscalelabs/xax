"""State management for XAX."""

from typing import Any, Dict, Optional

import jax.numpy as jnp


class State:
    """Represents the state of a task during execution."""
    
    def __init__(self):
        """Initialize an empty state."""
        self._data: Dict[str, Any] = {}
    
    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Get a value from the state.
        
        Args:
            key: The key to retrieve
            default: The default value to return if the key is not found
            
        Returns:
            The value associated with the key, or the default if not found
        """
        return self._data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in the state.
        
        Args:
            key: The key to set
            value: The value to store
        """
        self._data[key] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple values in the state.
        
        Args:
            updates: A dictionary of key-value pairs to update
        """
        self._data.update(updates)
    
    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the state.
        
        Args:
            key: The key to check
            
        Returns:
            True if the key exists, False otherwise
        """
        return key in self._data 