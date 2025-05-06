"""Base task classes for XAX."""

from abc import ABC
from typing import Generic, TypeVar

import jax

from xax.core.state import State


Config = TypeVar("Config")


class BaseTask(Generic[Config], ABC):
    """Base class for all XAX tasks."""
    
    config: Config
    
    def __init__(self, config: Config) -> None:
        """Initialize the task.
        
        Args:
            config: The configuration for the task
        """
        self.config = config 