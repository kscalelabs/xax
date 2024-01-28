"""Defines a mixin for abstracting the PyTorch tensor device."""

import functools
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
from mlfab.core.conf import Device as BaseDeviceConfig, field, parse_dtype
from mlfab.nn.device.auto import detect_device
from mlfab.nn.device.base import base_device
from mlfab.task.base import BaseConfig, BaseTask


@dataclass
class DeviceConfig(BaseConfig):
    device: BaseDeviceConfig = field(BaseDeviceConfig(), help="Device configuration")


Config = TypeVar("Config", bound=DeviceConfig)


class DeviceMixin(BaseTask[Config], Generic[Config]):
    @functools.cached_property
    def device(self) -> base_device:
        return detect_device()

    @functools.cached_property
    def torch_device(self) -> torch.device:
        return self.device.device

    @functools.cached_property
    def torch_dtype(self) -> torch.dtype:
        if (dtype := parse_dtype(self.config.device)) is not None:
            return dtype
        return self.device.dtype
