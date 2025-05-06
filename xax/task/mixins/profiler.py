"""Defines a mixin for profiling JAX computations."""

import functools
import logging
import os
import tempfile
import contextlib
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generic, Literal, Optional, TypeVar, cast

import jax
from jax.profiler import start_trace, stop_trace

from xax.core.conf import is_missing
from xax.core.state import State
from xax.task.base import BaseTask
from xax.task.mixins.artifacts import ArtifactsMixin
from xax.task.mixins.logger import LoggerMixin
from xax.utils.logging import LOG_STATUS

logger = logging.getLogger(__name__)

@dataclass
class ProfilerConfig:
    """Configuration for profiling JAX computations."""
    
    # Whether to enable profiling
    enable_profiling: bool = False
    
    # Directory to save profiling results; if None, uses the experiment directory
    profiling_dir: Optional[str] = None
    
    # How often to run profiling (in steps)
    profile_every_n_steps: int = 100
    
    # Duration of each profiling session in milliseconds
    profile_duration_ms: int = 3000
    
    # Maximum number of profiling sessions to save
    max_profile_count: int = 5
    
    # Profiling backend to use
    backend: Literal["tensorboard", "perfetto"] = "tensorboard"


Config = TypeVar("Config", bound=ProfilerConfig)


class ProfilerMixin(BaseTask[Config], Generic[Config], ABC):
    """Mixin for profiling JAX computations.
    
    This mixin adds profiling capabilities to a task. It will periodically collect
    traces of JAX operations and save them to the experiment directory.
    """
    
    _profile_count: int
    
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._profile_count = 0
    
    @property
    def should_profile(self) -> bool:
        """Returns whether profiling is enabled."""
        return self.config.enable_profiling
    
    def get_profiling_dir(self) -> Path:
        """Returns the directory where profiling results will be saved."""
        if not hasattr(self, "_profiling_dir"):
            if self.config.profiling_dir is not None:
                self._profiling_dir = Path(self.config.profiling_dir).expanduser().resolve()
            elif isinstance(self, ArtifactsMixin):
                self._profiling_dir = self.exp_dir / "profiles"
            else:
                # Create a temporary directory if we don't have a better place
                self._profiling_dir = Path(tempfile.mkdtemp()) / "profiles"
            
            self._profiling_dir.mkdir(parents=True, exist_ok=True)
            logger.log(LOG_STATUS, f"Profiling directory: {self._profiling_dir}")
            
        return self._profiling_dir
    
    def should_run_profiler(self, state: State) -> bool:
        """Check if we should run the profiler at the current step."""
        if not self.should_profile:
            return False
        
        if self._profile_count >= self.config.max_profile_count:
            return False
        
        if state.num_steps % self.config.profile_every_n_steps == 0:
            return True
        
        return False
    
    @contextlib.contextmanager
    def profile_context(self, state: State, name: str = "profile"):
        """A context manager for profiling a section of code.
        
        Args:
            state: The current training state.
            name: A name for this profiling session.
        """
        if not self.should_run_profiler(state):
            yield
            return
        
        try:
            profile_name = f"{name}_step_{state.num_steps}"
            logger.info(f"Starting profiling session: {profile_name}")
            
            output_path = self.get_profiling_dir() / profile_name
            start_trace(output_path, create_perfetto_link=True)
            
            yield
            
            # Log profiling information to the logger if available
            if isinstance(self, LoggerMixin):
                self.logger.log_text(
                    f"Profiling session: {profile_name}",
                    f"Profile saved to {output_path}",
                    namespace="profiling",
                )
            
            self._profile_count += 1
        finally:
            stop_trace()
            logger.info(f"Finished profiling session: {profile_name}")
    
    def profile_fn(self, fn: Callable, state: State, name: str = "profile") -> Any:
        """Profile a function execution.
        
        Args:
            fn: The function to profile.
            state: The current training state.
            name: A name for this profiling session.
            
        Returns:
            The result of the function.
        """
        if not self.should_run_profiler(state):
            return fn()
        
        with self.profile_context(state, name):
            return fn() 