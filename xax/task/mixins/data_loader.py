"""Defines a mixin for instantiating dataloaders."""

import logging
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import jax
from dpshdl.dataloader import CollatedDataloaderItem, Dataloader
from dpshdl.dataset import Dataset, ErrorHandlingDataset
from dpshdl.prefetcher import Prefetcher
from jaxtyping import Array
from omegaconf import II, MISSING

from xax.core.conf import field, is_missing
from xax.core.state import Phase
from xax.task.base import BaseConfig, BaseTask
from xax.task.mixins.process import ProcessConfig, ProcessMixin
from xax.utils.data.collate import CollateMode, collate

logger = logging.getLogger(__name__)

T = TypeVar("T")
Tc_co = TypeVar("Tc_co", covariant=True)


@dataclass
class DataLoaderConfig:
    batch_size: int = field(MISSING, help="Size of each batch")
    batch_size_multiplier: float = field(MISSING, help="Batch size multiplier")
    num_workers: int | None = field(MISSING, help="Number of workers for loading samples")
    prefetch_factor: int = field(2, help="Number of items to pre-fetch on each worker")


@dataclass
class DataLoadersConfig(ProcessConfig, BaseConfig):
    batch_size: int = field(MISSING, help="Size of each batch")
    raise_dataloader_errors: bool = field(False, help="If set, raise dataloader errors inside the worker processes")
    num_dataloader_workers: int | None = field(None, help="Default number of dataloader workers")
    train_dl: DataLoaderConfig = field(
        DataLoaderConfig(
            batch_size=II("batch_size"),
            batch_size_multiplier=1.0,
            num_workers=II("num_dataloader_workers"),
        ),
        help="Train dataloader config",
    )
    test_dl: DataLoaderConfig = field(
        DataLoaderConfig(
            batch_size=II("batch_size"),
            batch_size_multiplier=II("train_dl.batch_size_multiplier"),
            num_workers=1,
        ),
        help="Valid dataloader config",
    )
    debug_dataloader: bool = field(False, help="Debug dataloaders")


Config = TypeVar("Config", bound=DataLoadersConfig)


class DataLoadersMixin(ProcessMixin[Config], BaseTask[Config], Generic[Config]):
    def __init__(self, config: Config) -> None:
        if is_missing(config, "batch_size") and (
            is_missing(config.train_dl, "batch_size") or is_missing(config.test_dl, "batch_size")
        ):
            config.batch_size = self.get_batch_size()

        super().__init__(config)

    def get_batch_size(self) -> int:
        raise NotImplementedError(
            "When `batch_size` is not specified in your training config, you should override the `get_batch_size` "
            "method to return the desired training batch size."
        )

    def dataloader_config(self, phase: Phase) -> DataLoaderConfig:
        match phase:
            case "train":
                return self.config.train_dl
            case "valid":
                return self.config.test_dl
            case "test":
                return self.config.test_dl
            case _:
                raise KeyError(f"Unknown phase: {phase}")

    def get_dataset(self, phase: Phase) -> Dataset[T, Tc_co]:
        """Returns the dataset for the given phase.

        Args:
            phase: The phase for the dataset to return.

        Raises:
            NotImplementedError: If this method is not overridden
        """
        raise NotImplementedError("The task should implement `get_dataset`")

    def get_dataloader(self, dataset: Dataset[T, Tc_co], phase: Phase) -> Dataloader[T, Tc_co]:
        debugging = self.config.debug_dataloader
        if debugging:
            logger.warning("Parallel dataloaders disabled in debugging mode")

        cfg = self.dataloader_config(phase)

        # Wraps the dataset to handle errors.
        dataset = ErrorHandlingDataset(dataset)

        return Dataloader(
            dataset=dataset,
            batch_size=round(cfg.batch_size * cfg.batch_size_multiplier),
            num_workers=0 if debugging else cfg.num_workers,
            prefetch_factor=cfg.prefetch_factor,
            ctx=self.multiprocessing_context,
            dataloader_worker_init_fn=self.dataloader_worker_init_fn,
            collate_worker_init_fn=self.collate_worker_init_fn,
            item_callback=self.dataloader_item_callback,
            raise_errs=self.config.raise_dataloader_errors,
        )

    def get_prefetcher(self, dataloader: Dataloader[T, Tc_co]) -> Prefetcher[T, Tc_co]:
        return Prefetcher(dataloader=dataloader, to_device_fn=self.to_device_fn)

    @classmethod
    def to_device_fn(cls, arr: Array) -> Array:
        return jax.device_put(arr)

    @classmethod
    def collate_fn(cls, items: list[Any], *, mode: CollateMode = "stack") -> Any | None:  # noqa: ANN401
        return collate(items, mode=mode)

    def dataloader_worker_init_fn(self, worker_id: int, num_workers: int) -> None:
        pass

    def collate_worker_init_fn(self) -> None:
        pass

    def dataloader_item_callback(self, item: CollatedDataloaderItem[Tc_co]) -> None:
        pass
