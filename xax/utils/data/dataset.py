"""Defines an interface for loading data."""

import bdb
import itertools
import logging
import random
import re
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from queue import Queue
from typing import Generic, Iterator, Sequence, TypeVar

from xax.utils.logging import configure_logging
from xax.utils.text import TextBlock, render_text_blocks

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Dataset(Iterator[T], Generic[T], ABC):
    """Defines the Xax dataset interface.

    Xax datasets are analogous to a PyTorch iterable datasets that iterates
    forever. This means that there is no concept of an epoch or dataset size.
    """

    def worker_init(self, worker_id: int, num_workers: int) -> None:
        """Initializes the dataset worker.

        This method is called once per worker when the dataset is used in a
        dataloader.

        Args:
            worker_id: The ID of the worker.
            num_workers: The number of workers in the worker pool.
        """

    @abstractmethod
    def next(self) -> T:
        """Returns the next item in the dataset.

        Returns:
            The next item in the dataset.
        """

    def __iter__(self) -> "Dataset[T]":
        # Don't override this! Use `worker_init` instead.
        return self

    def __next__(self) -> T:
        # Don't override this! Use `next` instead.
        return self.next()

    def test(
        self,
        max_samples: int = 10,
        handle_errors: bool = False,
        log_interval: int | None = 1,
        truncate: int | None = 80,
        replace_whitespace: bool = True,
    ) -> None:
        """Defines a function for doing adhoc testing of the dataset.

        Args:
            max_samples: The maximum number of samples to test.
            handle_errors: If set, wraps the dataset in an error handling
                wrapper that will catch and log exceptions.
            log_interval: How often to log a sample. If None, don't log any
                samples.
            truncate: The maximum number of characters to show in a sample.
                If None, shows the entire sample.
            replace_whitespace: If set, replaces whitespace characters with
                spaces.
        """
        configure_logging()
        ds = ErrorHandlingDataset(self, flush_every_n=max_samples) if handle_errors else self
        start_time = time.time()
        ws_regex = re.compile(r"\s+") if replace_whitespace else None
        for i, sample in enumerate(itertools.islice(ds, max_samples)):
            if log_interval is not None and i % log_interval == 0:
                sample_str = str(sample)
                if ws_regex is not None:
                    sample_str = ws_regex.sub(" ", sample_str)
                if truncate is not None and len(sample_str) > truncate:
                    sample_str = sample_str[: truncate - 3] + "..."
                logger.info("Sample %d: %s", i, sample_str)
        elapsed_time = time.time() - start_time
        samples_per_second = i / elapsed_time
        logger.info("Tested %d samples in %f seconds (%f samples per second)", i + 1, elapsed_time, samples_per_second)


class ChunkedDataset(Dataset[T], Generic[T], ABC):
    """Defines a dataset that yields chunks of samples in a separate thread.

    This dataset is useful for implementing IO bound datasets such as
    datasets that read from disk or download data from the internet. Subclasses
    should implement the `next_chunk` method, which iterates over the next
    chunk of data. For example, a dataset that reads from a spinning disk
    might implement `next_chunk` as follows:

    .. code-block:: python

        def next_chunk(self) -> Iterator[T]:
            path = self.paths[self.current_path]
            self.current_path = (self.current_path + 1) % len(self.paths)
            with open(path, "rb", buffering=BIG_BUFFERING_SIZE) as fh:
                for line in fh:
                    yield line

    In the above example, we use a large buffer size to reduce the number of
    IO operations. Since this happens in a separate thread, we don't need to
    worry about creating a new file pointer for each chunk.

    Parameters:
        max_queue_size: The maximum number of samples to keep in the
            queue at a time.
    """

    def __init__(self, max_queue_size: int = 32) -> None:
        super().__init__()

        self.next_chunk_queue: Queue[T] = Queue(maxsize=max_queue_size)
        self.error_queue: Queue[Exception] = Queue()
        self.next_chunk_event = threading.Event()
        self.next_chunk_thread: threading.Thread | None = None

    @abstractmethod
    def next_chunk(self) -> Iterator[T]:
        """Returns the next chunk of data.

        Returns:
            The next chunk of data.
        """

    def chunked_dataset_thread(self) -> None:
        while True:
            try:
                for sample in self.next_chunk():
                    if self.next_chunk_queue.full():
                        self.next_chunk_event.clear()
                        self.next_chunk_event.wait()
                    self.next_chunk_queue.put(sample)
            except (bdb.BdbQuit, KeyboardInterrupt, StopIteration):
                raise
            except Exception as e:
                self.error_queue.put(e)

    def next(self) -> T:
        if self.next_chunk_thread is None:
            self.next_chunk_thread = threading.Thread(target=self.chunked_dataset_thread, daemon=True)
            self.next_chunk_thread.start()

        # If there are any errors in the error queue, raise them.
        if not self.error_queue.empty():
            raise self.error_queue.get()

        # If the thread is blocking but we're out of samples in the queue,
        # signal the thread to start adding samples again.
        if not self.next_chunk_event.is_set() and self.next_chunk_queue.empty():
            self.next_chunk_event.set()

        return self.next_chunk_queue.get()


class RoundRobinDataset(Dataset[T], Generic[T]):
    """Defines a dataset that yields samples in round robin fashion.

    Parameters:
        datasets: The datasets to sample from.
    """

    def __init__(self, datasets: Sequence[Dataset[T]]) -> None:
        super().__init__()

        self.datasets = datasets
        self.i = 0

    def worker_init(self, worker_id: int, num_workers: int) -> None:
        for dataset in self.datasets:
            dataset.worker_init(worker_id, num_workers)

    def next(self) -> T:
        next_item = self.datasets[self.i].next()
        self.i = (self.i + 1) % len(self.datasets)
        return next_item


class RandomDataset(Dataset[T], Generic[T]):
    """Defines a dataset that randomly samples from a list of datasets.

    Parameters:
        datasets: The datasets to sample from.
    """

    def __init__(
        self,
        datasets: Sequence[Dataset[T]],
        stop_on_first: bool = False,
    ) -> None:
        super().__init__()

        self.datasets = datasets
        self.stop_on_first = stop_on_first

    def worker_init(self, worker_id: int, num_workers: int) -> None:
        for dataset in self.datasets:
            dataset.worker_init(worker_id, num_workers)

    def next(self) -> T:
        return random.choice(self.datasets).next()


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
    num_steps: int
    num_exceptions: int
    top_exception_messages: list[tuple[str, int]]
    top_exception_types: list[tuple[str, int]]
    top_exception_locations: list[tuple[str, int]]
    last_exception: Exception | None

    def __str__(self) -> str:
        blocks: list[list[TextBlock]] = []

        blocks += [
            [
                TextBlock(f"Error Summary ({self.num_steps} steps)", color="red", bold=True, width=60, center=True),
                TextBlock("Count", color="yellow", bold=False, width=10, center=True),
                TextBlock("Percent", color="yellow", bold=False, width=10, center=True),
            ],
        ]

        def get_header(s: str) -> list[list[TextBlock]]:
            return [
                [
                    TextBlock(s, color="yellow", bold=True, width=60),
                    TextBlock("", width=10),
                    TextBlock("", width=10),
                ],
            ]

        def get_line(ks: str, v: int) -> list[list[TextBlock]]:
            line = [
                TextBlock(ks, width=60, no_sep=True),
                TextBlock(f"{v}", width=10, no_sep=True),
                TextBlock(f"{v / self.num_steps * 100:.2f}%", width=10, no_sep=True),
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

        # Logs the total number of exceptions.
        blocks += get_header("Total")
        blocks += get_line("Total", self.num_exceptions)

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

    def next(self) -> None:
        self.num_steps += 1
        self.step_has_error = False

    def __len__(self) -> int:
        return len(self.exceptions)

    def __bool__(self) -> bool:
        return len(self.exceptions) > 0

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
            num_steps=self.num_steps,
            num_exceptions=self.total_exceptions,
            top_exception_messages=self.exceptions.most_common(self.max_exceptions),
            top_exception_types=self.exception_classes.most_common(self.max_exceptions),
            top_exception_locations=self.exception_locs.most_common(self.max_exceptions),
            last_exception=self.last_exception,
        )

    def clear(self) -> None:
        self.exceptions.clear()
        self.exception_classes.clear()
        self.exception_locs.clear()

        self.num_steps = 0
        self.step_has_error = False
        self.total_exceptions = 0

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
        flush_every_n: Flush the exception summary every N steps.
    """

    def __init__(
        self,
        dataset: Dataset[T],
        sleep_backoff: float = 0.1,
        sleep_backoff_power: float = 2.0,
        maximum_exceptions: int = 10,
        backoff_after: int = 5,
        traceback_depth: int = 3,
        flush_every_n: int = 10000,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.sleep_backoff = sleep_backoff
        self.sleep_backoff_power = sleep_backoff_power
        self.maximum_exceptions = maximum_exceptions
        self.backoff_after = backoff_after
        self.traceback_depth = traceback_depth
        self.flush_every_n = flush_every_n
        self.log_exceptions = True

        self.exc_summary = ExceptionSummaryWriter()

    def worker_init(self, worker_id: int, num_workers: int) -> None:
        self.dataset.worker_init(worker_id, num_workers)
        if worker_id != 0:
            self.log_exceptions = False

    def next(self) -> T:
        num_exceptions = 0
        backoff_time = self.sleep_backoff
        self.exc_summary.next()

        if self.exc_summary.num_steps >= self.flush_every_n:
            if self.log_exceptions and self.exc_summary:
                logger.info("Exception summary:\n%s", self.exc_summary.summary())
            self.exc_summary.clear()

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
