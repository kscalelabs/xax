"""Defines a mixin for instantiating dataloaders."""

import logging
from abc import ABC
from dataclasses import dataclass
from typing import Generic, Iterator, TypeVar

import jax
from dpshdl.dataloader import CollatedDataloaderItem, Dataloader
from dpshdl.dataset import Dataset, ErrorHandlingDataset
from dpshdl.prefetcher import Prefetcher
from jaxtyping import PRNGKeyArray
from omegaconf import II, MISSING

from xax.core.conf import field, is_missing
from xax.core.state import Phase
from xax.nn.functions import set_random_seed
from xax.task.base import BaseConfig, BaseTask
from xax.task.mixins.process import ProcessConfig, ProcessMixin
from xax.utils.logging import LOG_ERROR_SUMMARY, configure_logging

logger = logging.getLogger(__name__)

T = TypeVar("T")
Tc_co = TypeVar("Tc_co", covariant=True)


@jax.tree_util.register_dataclass
@dataclass
class DataloaderErrorConfig:
    sleep_backoff: float = field(0.1, help="The initial sleep time after an exception")
    sleep_backoff_power: float = field(2.0, help="Power to raise the sleep time by after each consecutive exception")
    maximum_exceptions: int = field(10, help="The maximum number of consecutive exceptions before raising an error")
    backoff_after: int = field(5, help="The number of consecutive exceptions before starting to backoff")
    traceback_depth: int = field(5, help="The depth of the traceback to print when an exception occurs")
    flush_every_n_steps: int | None = field(None, help="Flush the error summary after this many steps")
    flush_every_n_seconds: float | None = field(10.0, help="Flush the error summary after this many seconds")
    log_exceptions_all_workers: bool = field(False, help="If set, log exceptions from all workers")


@jax.tree_util.register_dataclass
@dataclass
class DataloaderConfig:
    num_workers: int | None = field(MISSING, help="Number of workers for loading samples")
    prefetch_factor: int = field(2, help="Number of items to pre-fetch on each worker")
    error: DataloaderErrorConfig = field(DataloaderErrorConfig(), help="Dataloader error configuration")


@jax.tree_util.register_dataclass
@dataclass
class DataloadersConfig(ProcessConfig, BaseConfig):
    batch_size: int = field(MISSING, help="Size of each batch")
    raise_dataloader_errors: bool = field(False, help="If set, raise dataloader errors inside the worker processes")
    train_dl: DataloaderConfig = field(
        DataloaderConfig(num_workers=II("mlfab.num_workers:-1")),
        help="Train dataloader config",
    )
    valid_dl: DataloaderConfig = field(
        DataloaderConfig(num_workers=1),
        help="Valid dataloader config",
    )
    debug_dataloader: bool = field(False, help="Debug dataloaders")


Config = TypeVar("Config", bound=DataloadersConfig)


class DataloadersMixin(ProcessMixin[Config], BaseTask[Config], Generic[Config], ABC):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def get_batch_size(self) -> int:
        raise NotImplementedError(
            "When `batch_size` is not specified in your training config, you should override the `get_batch_size` "
            "method to return the desired training batch size."
        )

    @property
    def batch_size(self) -> int:
        if is_missing(self.config, "batch_size"):
            self.config.batch_size = self.get_batch_size()
        return self.config.batch_size

    def dataloader_config(self, phase: Phase) -> DataloaderConfig:
        match phase:
            case "train":
                return self.config.train_dl
            case "valid":
                return self.config.valid_dl
            case _:
                raise KeyError(f"Unknown phase: {phase}")

    def get_dataset(self, phase: Phase) -> Dataset:
        """Returns the dataset for the given phase.

        Args:
            phase: The phase for the dataset to return.

        Returns:
            The dataset for the given phase.
        """
        raise NotImplementedError(
            "You must implement either the `get_dataset` method to return the dataset for the given phase, "
            "or `get_data_iterator` to return an iterator for the given dataset."
        )

    def get_data_iterator(self, phase: Phase, key: PRNGKeyArray) -> Iterator:
        raise NotImplementedError(
            "You must implement either the `get_dataset` method to return the dataset for the given phase, "
            "or `get_data_iterator` to return an iterator for the given dataset."
        )

    def get_dataloader(self, dataset: Dataset[T, Tc_co], phase: Phase) -> Dataloader[T, Tc_co]:
        debugging = self.config.debug_dataloader
        if debugging:
            logger.warning("Parallel dataloaders disabled in debugging mode")

        cfg = self.dataloader_config(phase)

        # Wraps the dataset to handle errors.
        dataset = ErrorHandlingDataset(
            dataset=dataset,
            sleep_backoff=cfg.error.sleep_backoff,
            sleep_backoff_power=cfg.error.sleep_backoff_power,
            maximum_exceptions=cfg.error.maximum_exceptions,
            backoff_after=cfg.error.backoff_after,
            traceback_depth=cfg.error.traceback_depth,
            flush_every_n_steps=cfg.error.flush_every_n_steps,
            flush_every_n_seconds=cfg.error.flush_every_n_seconds,
            log_exceptions_all_workers=cfg.error.log_exceptions_all_workers,
            log_level=LOG_ERROR_SUMMARY,
        )

        return Dataloader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            num_workers=0 if debugging else cfg.num_workers,
            prefetch_factor=cfg.prefetch_factor,
            mp_manager=self.multiprocessing_manager,
            dataloader_worker_init_fn=self.dataloader_worker_init_fn,
            collate_worker_init_fn=self.collate_worker_init_fn,
            item_callback=self.dataloader_item_callback,
            raise_errs=self.config.raise_dataloader_errors,
        )

    def get_prefetcher(self, dataloader: Dataloader[T, Tc_co]) -> Prefetcher[Tc_co, Tc_co]:
        return Prefetcher(to_device_func=jax.device_put, dataloader=dataloader)

    @classmethod
    def dataloader_worker_init_fn(cls, worker_id: int, num_workers: int) -> None:
        configure_logging(prefix=f"{worker_id}")
        set_random_seed(offset=worker_id + 1)

    @classmethod
    def collate_worker_init_fn(cls) -> None:
        configure_logging(prefix="collate")
        set_random_seed(offset=-1)

    @classmethod
    def dataloader_item_callback(cls, item: CollatedDataloaderItem) -> None:
        pass
