"""Defines a class for loading data from a dataset.

This is a greatly simplified dataloader implementation. We start a pool of
worker processes which continually load data from the dataset and put it on a
queue. The main process then reads from the queue and yields batches of data
as lists. The user is responsible for performing any additional downstream
tasks like collation.
"""

import multiprocessing as mp
import os
import threading
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from queue import Queue
from threading import Event
from types import TracebackType
from typing import Generic, Self, TypeVar

from xax.utils.data.dataset import Dataset

T = TypeVar("T")


@dataclass(frozen=True)
class DataLoaderItem(Generic[T]):
    item: T | None
    exception: BaseException | None
    worker_id: int


def dataloader_worker(
    dataset: Dataset[T],
    queue: "Queue[DataLoaderItem[T]]",
    stop_event: Event,
    worker_id: int,
    num_workers: int,
) -> None:
    dataset.worker_init(worker_id, num_workers)
    while True:
        if stop_event.is_set():
            break
        try:
            queue.put(DataLoaderItem(dataset.next(), None, worker_id))
        except BaseException as e:
            queue.put(DataLoaderItem(None, e, worker_id))
            raise


def default_num_workers(default: int) -> int:
    if hasattr(os, "sched_getaffinity"):
        try:
            return len(os.sched_getaffinity(0))
        except Exception:
            pass
    if (cpu_count := os.cpu_count()) is not None:
        return cpu_count
    return default


class Dataloader(Generic[T]):
    """Defines a dataloader for loading data from a dataset.

    Usage:

    .. code-block:: python

        with Dataloader(dataset, num_workers=4, batch_size=4) as dataloader:
            for batch in dataloader:
                # Do something with batch

    Parameters:
        dataset: The dataset to load data from.
        num_workers: The number of workers to use. If 0, the dataset will be
            loaded on the main process. Otherwise, the dataset will be loaded
            on a pool of workers. If not provided, it will default to the
            number of CPUs on the system.
        batch_size: The batch size to use.
        preload_factor: The number of batches to pre-load from the dataset.
        ctx: The multiprocessing context to use. If not provided, the default
            context will be used.
    """

    def __init__(
        self,
        dataset: Dataset[T],
        num_workers: int | None = None,
        batch_size: int = 1,
        preload_factor: int = 2,
        ctx: BaseContext | None = None,
    ) -> None:
        super().__init__()

        if num_workers is None:
            num_workers = default_num_workers(0)
        if num_workers < 0:
            raise ValueError("`num_workers` must be greater than or equal to 0")
        if batch_size < 1:
            raise ValueError("`batch_size` must be greater than or equal to 1")
        if preload_factor < 1:
            raise ValueError("`preload_factor` must be greater than or equal to 1")

        self.dataset = dataset
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.ctx = mp.get_context() if ctx is None else ctx
        self.manager = self.ctx.Manager()

        self.processes: list[mp.Process] | None = None
        self.single_process_thread: threading.Thread | None = None
        self.queue: Queue[DataLoaderItem[T]] = self.manager.Queue(maxsize=batch_size * preload_factor)
        self.stop_event: Event = self.manager.Event()

    def __iter__(self) -> "Dataloader[T]":
        return self

    def __next__(self) -> list[T]:
        if self.processes is None and self.single_process_thread is None:
            raise RuntimeError("Dataloader is not running")
        items: list[T] = []
        while len(items) < self.batch_size:
            item = self.queue.get()
            if item.exception is not None:
                raise item.exception
            assert item.item is not None, "`item` should not be `None` unless there was an exception"
            items.append(item.item)
        return items

    def __enter__(self) -> Self:
        if self.processes is None and self.single_process_thread is None:
            if self.num_workers > 0:
                self.processes = []
                for i in range(self.num_workers):
                    process = mp.Process(
                        target=dataloader_worker,
                        args=(self.dataset, self.queue, self.stop_event, i, self.num_workers),
                        daemon=True,
                        name=f"xax-dataloader-worker-{i}",
                    )
                    process.start()
                    self.processes.append(process)
            else:
                self.single_process_thread = threading.Thread(
                    target=dataloader_worker,
                    args=(self.dataset, self.queue, self.stop_event, 0, 1),
                )
                self.single_process_thread.start()
        else:
            raise RuntimeError("Dataloader is already running")
        return self

    def __exit__(self, _t: type[BaseException] | None, _e: BaseException | None, _tr: TracebackType | None) -> None:
        if self.processes is None and self.single_process_thread is None:
            raise RuntimeError("DataLoader is not running")

        # Stops the dataloader workers by setting the stop event and clearing
        # the queue. Clearing the queue is important because otherwise the
        # workers will block while trying to put items on the queue.
        self.stop_event.set()
        while not self.queue.empty():
            self.queue.get()

        # Joins the dataloader workers.
        if self.processes is not None:
            for process in self.processes:
                process.terminate()
            self.processes = None
        elif self.single_process_thread is not None:
            self.single_process_thread.join()
            self.single_process_thread = None
        else:
            raise RuntimeError("Unexpected state")
