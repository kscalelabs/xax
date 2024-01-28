"""Defines a mixin for instantiating dataloaders."""

import logging
from dataclasses import dataclass
from typing import Any, Generic, Sized, TypeVar

from mlfab.core.conf import field, is_missing
from mlfab.core.state import Phase
from mlfab.nn.functions import set_random_seed
from mlfab.task.base import BaseConfig, BaseTask
from mlfab.task.mixins.process import ProcessConfig, ProcessMixin
from mlfab.utils.data.collate import CollateMode, collate
from mlfab.utils.data.error_handling import error_handling_dataset
from omegaconf import II, MISSING
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.datapipes.datapipe import IterDataPipe, MapDataPipe
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler

logger = logging.getLogger(__name__)

DataPipeT = TypeVar("DataPipeT", bound=IterDataPipe | MapDataPipe)


@dataclass
class DataLoaderConfig:
    batch_size: int = field(MISSING, help="Size of each batch")
    batch_size_multiplier: float = field(MISSING, help="Batch size multiplier")
    shuffle: bool = field(MISSING, help="Should the batches be shuffled on each iteration")
    num_workers: int = field(MISSING, help="Number of workers for loading samples")
    pin_memory: bool = field(MISSING, help="Should memory be pinned to it's GPU location")
    drop_last: bool = field(MISSING, help="Should the last batch be dropped if not full")
    timeout: float = field(0, help="How long to wait for a sample to be ready")
    prefetch_factor: int | None = field(None, help="Number of items to pre-fetch on each worker")
    persistent_workers: bool = field(False, help="Persist worker processes between epochs")
    seed: int = field(1337, help="Dataloader random seed")


@dataclass
class DataLoadersConfig(ProcessConfig, BaseConfig):
    batch_size: int = field(MISSING, help="Size of each batch")
    num_dataloader_workers: int = field(II("mlfab.num_workers:-1"), help="Default number of dataloader workers")
    train_dl: DataLoaderConfig = field(
        DataLoaderConfig(
            batch_size=II("batch_size"),
            batch_size_multiplier=1.0,
            shuffle=True,
            num_workers=II("num_dataloader_workers"),
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        ),
        help="Train dataloader config",
    )
    test_dl: DataLoaderConfig = field(
        DataLoaderConfig(
            batch_size=II("batch_size"),
            batch_size_multiplier=II("train_dl.batch_size_multiplier"),
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
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

    def apply_datapipe_transformations(self, datapipe: DataPipeT, phase: Phase) -> DataPipeT:
        """Applies transformations to the datapipe.

        Args:
            datapipe: The datapipe to transform
            phase: The dataset's phase

        Returns:
            The transformed datapipe
        """
        cfg = self.dataloader_config(phase)

        # Wraps the dataset in an error-handling dataset.
        datapipe = error_handling_dataset(datapipe)

        datapipe = datapipe.shuffle() if phase == "train" else datapipe
        datapipe = datapipe.sharding_filter()
        datapipe = datapipe.batch(round(cfg.batch_size * cfg.batch_size_multiplier), drop_last=cfg.drop_last)
        datapipe = datapipe.collate(collate_fn=self.collate_fn)

        return datapipe

    def get_datapipe_dataloader(self, datapipe: MapDataPipe | IterDataPipe, phase: Phase) -> DataLoader:
        debugging = self.config.debug_dataloader
        if debugging:
            logger.warning("Parallel dataloaders disabled in debugging mode")

        cfg = self.dataloader_config(phase)

        datapipe = self.apply_datapipe_transformations(datapipe, phase)

        return DataLoader(
            datapipe,
            num_workers=0 if debugging else cfg.num_workers,
            pin_memory=cfg.pin_memory,
            timeout=0 if debugging else cfg.timeout,
            worker_init_fn=self.worker_init_fn,
            multiprocessing_context=None if debugging or cfg.num_workers <= 0 else self.multiprocessing_context,
            generator=None,
            prefetch_factor=None if debugging or cfg.num_workers == 0 else cfg.prefetch_factor,
            persistent_workers=False if debugging or cfg.num_workers == 0 else cfg.persistent_workers,
        )

    def get_dataset(self, phase: Phase) -> Dataset:
        """Returns the dataset for the given phase.

        Args:
            phase: The phase for the dataset to return.

        Raises:
            NotImplementedError: If this method is not overridden
        """
        raise NotImplementedError("The task should implement `get_dataset`")

    def get_sampler(self, dataset: Dataset, cfg: DataLoaderConfig, phase: Phase) -> Sampler[int]:
        """Returns a dataset sampler to use instead of random sampling.

        The default behavior for a non-iterable dataset is to use a
        RandomSampler for all the elements from the dataset. The sampler
        should yield integer indices into the dataset.

        Args:
            dataset: The dataset to sample from
            cfg: The associated dataloader config
            phase: The dataset's phase

        Raises:
            NotImplementedError: If this method is not overridden
        """
        raise NotImplementedError("`get_sampler` should be implemented for the specific task")

    def get_batch_sampler(self, sampler: Sampler, cfg: DataLoaderConfig, phase: Phase) -> Sampler[list[int]]:
        """Returns a dataset batch sampler to use instead fo sequential sampling.

        The batch sampler should yield lists of integer indices, which
        are the samples that are passed to the dataset.

        Args:
            sampler: The underlying sampler
            cfg: The associated dataloader config
            phase: The dataset's phase

        Raises:
            NotImplementedError: If this method is not overridden
        """
        raise NotImplementedError("`get_sampler` should be implemented for the specific task")

    def get_dataloader(self, dataset: Dataset, phase: Phase) -> DataLoader:
        if isinstance(dataset, (MapDataPipe, IterDataPipe)):
            return self.get_datapipe_dataloader(dataset, phase)

        debugging = self.config.debug_dataloader
        if debugging:
            logger.warning("Parallel dataloaders disabled in debugging mode")

        cfg = self.dataloader_config(phase)

        # Wraps the dataset to handle errors.
        dataset = error_handling_dataset(dataset)

        # Arguments shared by all dataloaders.
        common_kwargs = {
            "num_workers": 0 if debugging else cfg.num_workers,
            "collate_fn": self.collate_fn,
            "pin_memory": cfg.pin_memory,
            "timeout": 0 if debugging else cfg.timeout,
            "worker_init_fn": self.worker_init_fn,
            "multiprocessing_context": None,
            "generator": None,
            "prefetch_factor": None if debugging or cfg.num_workers == 0 else cfg.prefetch_factor,
            "persistent_workers": False if debugging or cfg.num_workers == 0 else cfg.persistent_workers,
        }

        try:
            sampler = self.get_sampler(dataset, cfg, phase)
        except NotImplementedError:
            return DataLoader(
                dataset=dataset,
                batch_size=round(cfg.batch_size * cfg.batch_size_multiplier),
                drop_last=cfg.drop_last,
                shuffle=cfg.shuffle if isinstance(dataset, Sized) else False,
                **common_kwargs,  # type: ignore[arg-type]
            )

        try:
            batch_sampler = self.get_batch_sampler(sampler, cfg, phase)
        except NotImplementedError:
            return DataLoader(
                dataset=dataset,
                sampler=sampler,
                batch_size=round(cfg.batch_size * cfg.batch_size_multiplier),
                drop_last=cfg.drop_last,
                **common_kwargs,  # type: ignore[arg-type]
            )

        return DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            **common_kwargs,  # type: ignore[arg-type]
        )

    @classmethod
    def worker_init_fn(cls, worker_id: int) -> None:
        set_random_seed(offset=worker_id)

    @classmethod
    def collate_fn(cls, items: list[Any], *, mode: CollateMode = "stack") -> Any | None:  # noqa: ANN401
        return collate(items, mode=mode)
