"""Defines an interface for loading data."""

import bdb
import logging
import random
import sys
import time
from abc import ABC, abstractmethod
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Generic, Iterator, Sequence, TypeVar

from xax.utils.text import TextBlock, render_text_blocks

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Dataset(ABC, Iterator[T], Generic[T]):
    """Defines the Xax dataset interface.

    Xax datasets are analogous to PyTorch's iterable datasets.
    """

    @abstractmethod
    def start(self) -> None:
        """Starts the dataset.

        This method is called before the first call to `next`, signifies the
        beginning of a new epoch.
        """

    @abstractmethod
    def next(self) -> T:
        """Returns the next item in the dataset.

        Returns:
            The next item in the dataset.

        Raises:
            StopIteration: If the dataset has been exhausted.
        """

    def worker_init(self, worker_id: int, num_workers: int) -> None:
        """Initializes the dataset worker.

        This method is called once per worker when the dataset is used in a
        dataloader.

        Args:
            worker_id: The ID of the worker.
            num_workers: The number of workers in the worker pool.
        """

    def __iter__(self) -> "Dataset[T]":
        # Don't override this! Use `start` instead.
        self.start()
        return self

    def __next__(self) -> T:
        # Don't override this! Use `next` instead.
        return self.next()


class RoundRobinDataset(Dataset[T], Generic[T]):
    """Defines a dataset that yields samples in round robin fashion.

    Parameters:
        datasets: The datasets to sample from.
        stop_on_first: Whether to stop after the first dataset is exhausted,
            or to continue sampling until all datasets are exhausted. The
            former case ensures that your samples are balanced across datasets,
            while the latter case ensures that you use all of your data.
    """

    def __init__(
        self,
        datasets: Sequence[Dataset[T]],
        stop_on_first: bool = False,
    ) -> None:
        super().__init__()

        self.datasets = datasets
        self.stop_on_first = stop_on_first
        self.datasets_queue: Deque[Dataset[T]] = deque()

    def start(self) -> None:
        self.datasets_queue.clear()
        for dataset in self.datasets:
            dataset.start()
            self.datasets_queue.append(dataset)

    def next(self) -> T:
        while True:
            if len(self.datasets_queue) == 0:
                raise StopIteration
            dataset = self.datasets_queue.popleft()
            try:
                next_item = dataset.next()
                self.datasets_queue.append(dataset)
                return next_item
            except StopIteration:
                if self.stop_on_first:
                    raise
                continue


class RandomDataset(Dataset[T], Generic[T]):
    """Defines a dataset that randomly samples from a list of datasets.

    Parameters:
        datasets: The datasets to sample from.
        stop_on_first: Whether to stop after the first dataset is exhausted,
            or to continue sampling until all datasets are exhausted. The
            former case ensures that your samples are balanced across datasets,
            while the latter case ensures that you use all of your data.
    """

    def __init__(
        self,
        datasets: Sequence[Dataset[T]],
        stop_on_first: bool = False,
    ) -> None:
        super().__init__()

        self.datasets = datasets
        self.stop_on_first = stop_on_first
        self.dataset_list: list[Dataset[T]] = []

    def start(self) -> None:
        self.dataset_list.clear()
        for dataset in self.datasets:
            dataset.start()
            self.dataset_list.append(dataset)

    def next(self) -> T:
        while True:
            if len(self.dataset_list) == 0:
                raise StopIteration
            dataset = random.choice(self.dataset_list)
            try:
                return dataset.next()
            except StopIteration:
                if self.stop_on_first:
                    raise
                self.dataset_list.remove(dataset)
                continue


def get_loc(num_excs: int = 1) -> str:
    _, _, exc_tb = sys.exc_info()
    if exc_tb is None or (exc_tb := exc_tb.tb_next) is None:
        return "unknown"
    exc_strs: list[str] = []
    for _ in range(num_excs):
        exc_strs += [f"{exc_tb.tb_frame.f_code.co_filename}:{exc_tb.tb_lineno}"]
        if (exc_tb := exc_tb.tb_next) is None:
            break
    return "\n".join(exc_strs)


@dataclass(frozen=True)
class ExceptionSummary:
    top_exception_messages: list[tuple[str, int]]
    top_exception_types: list[tuple[str, int]]
    top_exception_locations: list[tuple[str, int]]
    last_exception: Exception | None

    def __str__(self) -> str:
        blocks: list[list[TextBlock]] = []

        blocks += [
            [
                TextBlock("Error Summary", color="red", bold=True, width=60, center=True),
                TextBlock("Count", color="yellow", bold=False, width=10, center=True),
            ],
        ]

        def get_header(s: str) -> list[list[TextBlock]]:
            return [
                [
                    TextBlock(s, color="yellow", bold=True, width=60),
                    TextBlock("", width=10),
                ],
            ]

        def get_line(ks: str, v: int) -> list[list[TextBlock]]:
            line = [
                TextBlock(ks, width=60, no_sep=True),
                TextBlock(f"{v}", width=10, no_sep=True),
            ]
            return [line]

        # Logs unique exception strings.
        blocks += get_header("Exceptions")
        for k, v in self.top_exception_messages:
            blocks += get_line(k, v)

        # Logs the individual exception classes.
        blocks += get_header("Types")
        for k, v in self.top_exception_types:
            blocks += get_line(k, v)

        # Logs by line number.
        blocks += get_header("Locations")
        for k, v in self.top_exception_locations:
            blocks += get_line(k, v)

        return render_text_blocks(blocks)


class ExceptionSummaryWriter:
    """Defines a utility class for storing and logging exceptions.

    Parameters:
        max_exceptions: The maximum number of unique exceptions to log.
    """

    def __init__(self, max_exceptions: int = 10) -> None:
        super().__init__()

        self.max_exceptions = max_exceptions

        self.exceptions: Counter[str] = Counter()
        self.exception_classes: Counter[str] = Counter()
        self.exception_locs: Counter[str] = Counter()

        self.last_exception: Exception | None = None
        self.num_steps = 0
        self.step_has_error = False
        self.total_exceptions = 0

    def start(self) -> None:
        self.num_steps = 0
        self.step_has_error = False
        self.total_exceptions = 0

    def next(self) -> None:
        self.num_steps += 1
        self.step_has_error = False

    def add_exception(self, exc: Exception, loc: str) -> None:
        self.last_exception = exc
        self.exceptions[f"{exc.__class__.__name__}: {exc}"] += 1
        self.exception_classes[exc.__class__.__name__] += 1
        self.exception_locs[loc] += 1
        if not self.step_has_error:
            self.step_has_error = True
            self.total_exceptions += 1

    def summary(self) -> ExceptionSummary:
        return ExceptionSummary(
            top_exception_messages=self.exceptions.most_common(self.max_exceptions),
            top_exception_types=self.exception_classes.most_common(self.max_exceptions),
            top_exception_locations=self.exception_locs.most_common(self.max_exceptions),
            last_exception=self.last_exception,
        )

    def clear(self) -> None:
        self.exceptions.clear()
        self.exception_classes.clear()
        self.exception_locs.clear()

    def __str__(self) -> str:
        return str(self.summary())


class ErrorHandlingDataset(Dataset[T]):
    """Defines a wrapper for safely handling errors in iterable datasets.

    Parameters:
        dataset: The dataset to wrap.
        sleep_backoff: The initial sleep time after an exception.
        sleep_backoff_power: The power to raise the sleep time by after
            each consecutive exception.
        maximum_exceptions: The maximum number of consecutive exceptions
            to allow before raising an error.
        backoff_after: The number of exceptions to allow before backing
            off (i.e. increasing the sleep time).
        traceback_depth: The number of stack frames to include in the
            exception traceback.
    """

    def __init__(
        self,
        dataset: Dataset[T],
        sleep_backoff: float = 0.1,
        sleep_backoff_power: float = 2.0,
        maximum_exceptions: int = 10,
        backoff_after: int = 5,
        traceback_depth: int = 3,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.sleep_backoff = sleep_backoff
        self.sleep_backoff_power = sleep_backoff_power
        self.maximum_exceptions = maximum_exceptions
        self.backoff_after = backoff_after
        self.traceback_depth = traceback_depth

        self.exc_summary = ExceptionSummaryWriter()

    def start(self) -> None:
        self.dataset.start()
        self.exc_summary.start()

    def next(self) -> T:
        num_exceptions = 0
        backoff_time = self.sleep_backoff
        self.exc_summary.next()

        while num_exceptions < self.maximum_exceptions:
            try:
                return self.dataset.next()
            except (bdb.BdbQuit, KeyboardInterrupt, StopIteration):
                raise
            except Exception as e:
                self.exc_summary.add_exception(e, get_loc(self.traceback_depth))
            num_exceptions += 1
            if num_exceptions > self.backoff_after:
                logger.error("Encountered %d exceptions, backing off for %f seconds", num_exceptions, backoff_time)
                time.sleep(backoff_time)
                backoff_time *= self.sleep_backoff_power
        raise RuntimeError(f"Reached max exceptions {self.maximum_exceptions}\n{self.exc_summary.summary()}")
