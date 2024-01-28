"""A task mixin to support ``torch.compile``."""

import logging
from dataclasses import dataclass
from typing import Callable, Generic, ParamSpec, TypeVar, cast

import torch
from mlfab.core.conf import field
from mlfab.task.mixins.device import DeviceConfig, DeviceMixin
from omegaconf import II
from torch import nn

logger = logging.getLogger(__name__)

T = TypeVar("T")
P = ParamSpec("P")

Model = TypeVar("Model", bound=nn.Module)


@dataclass
class TorchCompileOptions:
    model: bool = field(II("oc.env:COMPILE_MODEL,0"), help="Enable Torch compilation for the model")
    func: bool = field(II("oc.env:COMPILE_FUNC,0"), help="Enable Torch compilation for functions")
    fullgraph: bool = field(False, help="Whether it is OK to break the model into subgraphs")
    dynamic: bool = field(False, help="Whether to use dynamic shape tracing")
    backend: str = field("auto", help="The backend to use")
    model_mode: str | None = field("max-autotune", help="Either 'default', 'reduce-overhead' or 'max-autotune'")
    func_mode: str | None = field("reduce-overhead", help="Either 'default', 'reduce-overhead' or 'max-autotune'")


@dataclass
class CompileConfig(DeviceConfig):
    compiler: TorchCompileOptions = field(TorchCompileOptions(), help="Torch compile config")


Config = TypeVar("Config", bound=CompileConfig)


class CompileMixin(DeviceMixin[Config], Generic[Config]):
    """Defines a mixin for calling `torch.compile` on models and functions."""

    def get_compiler_backend(self) -> str | Callable:
        backend: str | Callable = self.config.compiler.backend
        if backend == "auto":
            backend = self.device.get_torch_compile_backend()
            logger.info("Using torch-compile backend [%s]", backend)
        return backend

    def compile_model(self, model: Model) -> Model:
        if self.config.compiler.model:
            model = cast(
                Model,
                torch.compile(
                    model,
                    fullgraph=self.config.compiler.fullgraph,
                    dynamic=self.config.compiler.dynamic,
                    backend=self.get_compiler_backend(),
                    mode=self.config.compiler.model_mode,
                    disable=not self.config.compiler.model,
                ),
            )

        return model

    def compile_function(self, func: Callable[P, T]) -> Callable[P, T]:
        if self.config.compiler.func:
            func = torch.compile(
                func,
                fullgraph=self.config.compiler.fullgraph,
                dynamic=self.config.compiler.dynamic,
                backend=self.get_compiler_backend(),
                mode=self.config.compiler.func_mode,
                disable=not self.config.compiler.func,
            )

        return func
