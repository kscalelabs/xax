"""Utilities for working with JAX pytrees."""

from typing import Any, Dict, List, Tuple, Union

# Type alias for a PyTree
PyTree = Union[
    Dict[str, Any], 
    List[Any], 
    Tuple[Any, ...], 
    None, 
    int, 
    float, 
    bool, 
    str,
    Any,
] 