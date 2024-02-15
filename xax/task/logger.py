"""Defines the core logger.

A common problem when quickly prototyping ML models is nicely logging images,
videos, audio, or other data. Additionally, logging on every step can be
overwhelming. This logger implements a number of convenience functions to
take heterogeneous input data and put it into a standard format, which can
then be used by downstream loggers to actually log the data. For example, this
logger will automatically tile multiple images into a single image, add
captions to images, and so on.
"""

import functools
import logging
import math
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from types import TracebackType
from typing import Callable, Iterator, Literal, Self, Sequence, TypeVar, get_args

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFont
from PIL.Image import Image as PILImage

from xax.core.state import Phase, State
from xax.utils.experiments import IntervalTicker
from xax.utils.logging import LOG_ERROR_SUMMARY, LOG_PING, LOG_STATUS

logger = logging.getLogger(__name__)

T = TypeVar("T")
LogT = TypeVar("LogT")
Number = int | float | Array | np.ndarray

ChannelSelectMode = Literal["first", "last", "mean"]

DEFAULT_NAMESPACE = "value"

NAMESPACE_STACK: list[str] = []


def standardize_text(text: str, max_line_length: int | None = None, remove_non_ascii: bool = False) -> list[str]:
    """Standardizes a text string to a list of lines.

    Args:
        text: The text to standardize
        max_line_length: If set, truncate lines to this length
        remove_non_ascii: Remove non-ASCII characters if present

    Returns:
        The standardized text lines
    """

    def _chunk_lines(text: str, max_length: int) -> Iterator[str]:
        for i in range(0, len(text), max_length):
            yield text[i : i + max_length]

    if remove_non_ascii:
        text = "".join(char for char in text if ord(char) < 128)
    lines = [re.sub(r"\s+", " ", line) for line in re.split(r"[\n\r]+", text.strip())]
    if max_line_length is not None:
        lines = [subline for line in lines for subline in _chunk_lines(line, max_line_length)]
    return lines


def make_human_viewable_resolution(
    image: PILImage,
    interpolation: Image.Resampling = Image.Resampling.LANCZOS,
    trg_res: tuple[int, int] = (512, 512),
) -> PILImage:
    """Resizes image to human-viewable resolution.

    Args:
        image: The image to resize, with shape (C, H, W)
        interpolation: Interpolation mode to use for image resizing
        trg_res: The target image resolution; the image will be reshaped to
            have approximately the same area as an image with this resolution

    Returns:
        The resized image
    """
    width, height = image.size
    trg_height, trg_width = trg_res
    factor = math.sqrt((trg_height * trg_width) / (height * width))
    new_height, new_width = int(height * factor), int(width * factor)
    return image.resize((new_width, new_height), interpolation)


def image_with_text(
    image: PILImage,
    text: list[str],
    max_num_lines: int | None,
    line_spacing: int,
    centered: bool,
) -> PILImage:
    """Adds a text label to an image.

    Args:
        image: The image to label, with shape (C, H, W)
        text: The text label for the image
        max_num_lines: The number of lines of spacing to add to the bottom
            of the image
        line_spacing: The spacing between adjacent lines
        centered: If set, center the text labels, otherwise align to the left

    Returns:
        The image with a text label
    """
    if not text:
        return image
    if max_num_lines is None:
        max_num_lines = len(text)
    else:
        text = text[:max_num_lines]
    width, height = image.size
    font: ImageFont.ImageFont = ImageFont.load_default()
    _, _, _, line_height = font.getbbox(text[0])
    new_width, new_height = width, height + line_spacing + max_num_lines * (line_height + line_spacing)
    padded_image = Image.new(image.mode, (new_width, new_height), 255)
    padded_image.paste(image, (0, 0))
    drawer = ImageDraw.Draw(padded_image)
    for i, text_line in enumerate(text):
        text_line_top = height + line_spacing + i * (line_height + line_spacing)
        if centered:
            _, _, line_width, _ = font.getbbox(text_line)
            text_line_left = (width - line_width) / 2
            drawer.text((text_line_left, text_line_top), text_line, font=font, fill=0)
        else:
            drawer.text((line_spacing, text_line_top), text_line, font=font, fill=0)
    return padded_image


class namespace_context:  # noqa: N801
    def __init__(self, name: str | None) -> None:
        self._name = name
        self._prev_stack: list[str] | None = None

    def __enter__(self) -> None:
        if self._name is None:
            self._prev_stack = NAMESPACE_STACK[:]
            NAMESPACE_STACK.clear()
        else:
            NAMESPACE_STACK.append(self._name)

    def __exit__(self, _t: type[BaseException] | None, _e: BaseException | None, _tr: TracebackType | None) -> None:
        if self._prev_stack is not None:
            NAMESPACE_STACK[:] = self._prev_stack
        else:
            NAMESPACE_STACK.pop()


def normalize(x: np.ndarray) -> np.ndarray:
    return (x - x.min()) / (x.max() - x.min())


def ternary_search_optimal_side_counts(height: int, width: int, count: int) -> tuple[int, int]:
    min_factors = [i for i in range(1, math.ceil(math.sqrt(count)) + 1) if count % i == 0]
    max_factors = [i for i in min_factors[::-1] if i * i != count]
    factors = [(i, count // i) for i in min_factors] + [(count // i, i) for i in max_factors]

    lo, hi = 0, len(factors) - 1

    def penalty(i: int) -> float:
        hval, wval = factors[i]
        h, w = hval * height, wval * width
        return -(min(h, w) ** 2)

    # Runs ternary search to minimize penalty.
    while lo < hi - 2:
        lmid, rmid = (lo * 2 + hi) // 3, (lo + hi * 2) // 3
        if penalty(lmid) > penalty(rmid):
            lo = lmid
        else:
            hi = rmid

    # Returns the lowest-penalty configuration.
    mid = (lo + hi) // 2
    plo, pmid, phi = penalty(lo), penalty(mid), penalty(hi)

    if pmid <= plo and pmid <= phi:
        return factors[mid]
    elif plo <= phi:
        return factors[lo]
    else:
        return factors[hi]


def tile_images_different_sizes(images: list[PILImage], sep: int) -> PILImage:
    """Tiles a list of images into a single image, even if they have different sizes.

    Args:
        images: The images to tile.
        sep: The separation between adjacent images.

    Returns:
        The tiled image.
    """
    total_width, max_height = sum(image.width for image in images), max(image.height for image in images)
    tiled = Image.new("RGB", (total_width + (len(images) - 1) * sep, max_height))
    x = 0
    for image in images:
        tiled.paste(image, (x, 0))
        x += image.width + sep
    return tiled


def tile_images(images: list[PILImage], sep: int = 0) -> PILImage:
    """Tiles a list of images into a single image.

    Args:
        images: The images to tile.
        sep: The separation between adjacent images.

    Returns:
        The tiled image.
    """
    if not images:
        return Image.new("RGB", (0, 0))

    # Gets the optimal side counts.
    height, width = images[0].height, images[0].width
    if not all(image.size == images[0].size for image in images):
        return tile_images_different_sizes(images, sep)

    hside, wside = ternary_search_optimal_side_counts(height, width, len(images))

    # Tiles the images.
    tiled = Image.new("RGB", (wside * width + (wside - 1) * sep, hside * height + (hside - 1) * sep))
    for i, image in enumerate(images):
        x, y = i % wside, i // wside
        tiled.paste(image, (x * (width + sep), y * (height + sep)))

    return tiled


def as_numpy(array: Array) -> np.ndarray:
    array = jax.device_get(array)
    if jax.dtypes.issubdtype(array.dtype, jnp.floating):
        array = array.astype(jnp.float32)
    elif jax.dtypes.issubdtype(array.dtype, jnp.integer):
        array = array.astype(jnp.int32)
    elif jax.dtypes.issubdtype(array.dtype, jnp.bool_):
        array = array.astype(jnp.bool_)
    return np.array(array)


def get_image(image: np.ndarray | Array | PILImage, target_resolution: tuple[int, int] | None = None) -> PILImage:
    if not isinstance(image, (np.ndarray, Array, PILImage)):
        raise ValueError(f"Unsupported image type: {type(image)}")
    if isinstance(image, Array):
        image = as_numpy(image)
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        if image.ndim != 3:
            raise RuntimeError(f"Expected image to have shape HW, HWC, or CHW, got {image.shape}")

        # Normalizes the image and converts to integer.
        if np.issubdtype(image.dtype, np.floating):
            image = (normalize(image) * 255).round().astype(np.uint8)
        elif image.dtype == np.uint8:
            pass
        else:
            raise ValueError(f"Unsupported image dtype: {image.dtype}")

        # Converts to a PIL image.
        if image.shape[-1] == 1:
            image = Image.fromarray(image[..., 0])
        elif image.shape[-1] == 3:
            image = Image.fromarray(image)
        elif image.shape[0] == 1:
            image = Image.fromarray(image[0])
        elif image.shape[0] == 3:
            image = Image.fromarray(image.transpose(1, 2, 0))
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

    if target_resolution is not None:
        image = make_human_viewable_resolution(image, trg_res=target_resolution)
    return image


@dataclass
class LogImage:
    image: PILImage


@dataclass
class LogLine:
    state: State
    scalars: dict[str, dict[str, Number]]
    strings: dict[str, dict[str, str]]
    images: dict[str, dict[str, LogImage]]


@dataclass
class LogErrorSummary:
    message: str


@dataclass
class LogError:
    message: str
    location: str | None = None

    @property
    def message_with_location(self) -> str:
        message = self.message
        if self.location is not None:
            message += f" ({self.location})"
        return message


@dataclass
class LogStatus:
    message: str
    created: float
    filename: str | None = None
    lineno: int | None = None


@dataclass
class LogPing:
    message: str
    created: float
    filename: str | None = None
    lineno: int | None = None


class LoggerImpl(ABC):
    def __init__(self, log_interval_seconds: float = 1.0) -> None:
        """Defines some default behavior for loggers.

        Every logger needs to implement the ``write`` function, which handles
        actually writing the logs to wherever they needs to go. The basic
        class implements a simple interval-based logging scheme to avoid
        writing too many log lines.

        Args:
            log_interval_seconds: The interval between successive log lines.
        """
        super().__init__()

        self.tickers = {phase: IntervalTicker(log_interval_seconds) for phase in get_args(Phase)}

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    @abstractmethod
    def write(self, line: LogLine) -> None:
        """Handles writing the current log line.

        Args:
            line: The line to write.
        """

    def write_error_summary(self, error_summary: LogErrorSummary) -> None:
        """Handles writing an error summary.

        Args:
            error_summary: The error summary to write.
        """

    def write_error(self, error: LogError) -> None:
        """Handles writing an error line.

        Args:
            error: The error information to write.
        """

    def write_status(self, status: LogStatus) -> None:
        """Handles writing a status line.

        Args:
            status: The status to write.
        """

    def write_ping(self, ping: LogPing) -> None:
        """Handles writing a ping line.

        Args:
            ping: The ping to write.
        """

    def log_git_state(self, git_state: str) -> None:
        """Logs Git state for the current run.

        Args:
            git_state: The Git state, as text blocks.
        """

    def log_training_code(self, training_code: str) -> None:
        """Logs the training script code.

        Args:
            training_code: The training script code.
        """

    def log_config(self, config: DictConfig) -> None:
        """Logs the configuration for the current run.

        Args:
            config: The configuration, as a DictConfig.
        """

    def should_log(self, state: State) -> bool:
        """Function that determines if the logger should log the current step.

        Args:
            state: The current step's state.

        Returns:
            If the logger should log the current step.
        """
        return self.tickers[state.phase].tick(state.elapsed_time_s)


class ToastHandler(logging.Handler):
    def __init__(self, logger: "Logger") -> None:
        super().__init__()

        self.logger = logger

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if record.levelno == LOG_ERROR_SUMMARY:
                self.logger.write_error_summary(record.getMessage())
            elif record.levelno == LOG_STATUS:
                self.logger.write_status(record.getMessage(), record.filename, record.lineno)
            elif record.levelno in (LOG_PING, logging.WARNING):
                self.logger.write_ping(record.getMessage(), record.filename, record.lineno)
            elif record.levelno in (logging.ERROR, logging.CRITICAL, logging.WARNING):
                self.logger.write_error(record.getMessage(), f"{record.filename}:{record.lineno}")
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)

    def add_for_logger(self, logger: logging.Logger) -> None:
        # Removes existing ToastHandler.
        handlers_to_remove = []
        for handler in logger.handlers:
            if isinstance(handler, ToastHandler):
                handlers_to_remove.append(handler)
        for handler in handlers_to_remove:
            logger.removeHandler(handler)

        # Adds the new ToastHandler.
        logger.addHandler(self)


class Logger:
    """Defines an intermediate container which holds values to log somewhere else."""

    def __init__(self, default_namespace: str = DEFAULT_NAMESPACE) -> None:
        self.scalars: dict[str, dict[str, Callable[[], Number]]] = defaultdict(dict)
        self.strings: dict[str, dict[str, Callable[[], str]]] = defaultdict(dict)
        self.images: dict[str, dict[str, Callable[[], PILImage]]] = defaultdict(dict)
        self.default_namespace = default_namespace
        self.loggers: list[LoggerImpl] = []

        # Registers a logging handler to route log messages to the logger.
        root_logger = logging.getLogger()
        ToastHandler(self).add_for_logger(root_logger)

        # Flag when the logger is active.
        self.active = False

    def add_logger(self, *logger: LoggerImpl) -> None:
        """Add the logger, so that it gets called when `write` is called.

        Args:
            logger: The logger to add.
        """
        self.loggers.extend(logger)

    def pack(self, state: State) -> LogLine:
        return LogLine(
            state=state,
            scalars={k: {kk: v() for kk, v in v.items()} for k, v in self.scalars.items()},
            strings={k: {kk: v() for kk, v in v.items()} for k, v in self.strings.items()},
            images={k: {kk: LogImage(v()) for kk, v in v.items()} for k, v in self.images.items()},
        )

    def clear(self) -> None:
        self.scalars.clear()
        self.strings.clear()
        self.images.clear()

    def write(self, state: State) -> None:
        """Writes the current step's logging information.

        Args:
            state: The current step's state.
        """
        should_log = [logger.should_log(state) for logger in self.loggers]
        if not any(should_log):
            self.clear()
            return
        line = self.pack(state)
        self.clear()
        for logger in (logger for logger, should_log in zip(self.loggers, should_log) if should_log):
            logger.write(line)

    def write_error_summary(self, error_summary: str) -> None:
        for logger in self.loggers:
            logger.write_error_summary(LogErrorSummary(error_summary))

    def write_error(self, message: str, location: str | None = None) -> None:
        for logger in self.loggers:
            logger.write_error(LogError(message, location))

    def write_status(
        self,
        message: str,
        filename: str | None = None,
        lineno: int | None = None,
        created: float | None = None,
    ) -> None:
        status = LogStatus(message, time.time() if created is None else created, filename, lineno)
        for logger in self.loggers:
            logger.write_status(status)

    def write_ping(
        self,
        message: str,
        filename: str | None = None,
        lineno: int | None = None,
        created: float | None = None,
    ) -> None:
        ping = LogPing(message, time.time() if created is None else created, filename, lineno)
        for logger in self.loggers:
            logger.write_ping(ping)

    def resolve_namespace(self, namespace: str | None = None) -> str:
        return "_".join([self.default_namespace if namespace is None else namespace] + NAMESPACE_STACK)

    def log_scalar(self, key: str, value: Callable[[], Number] | Number, *, namespace: str | None = None) -> None:
        """Logs a scalar value.

        Args:
            key: The key being logged
            value: The scalar value being logged
            namespace: An optional logging namespace
        """
        if not self.active:
            raise RuntimeError("The logger is not active")
        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache(maxsize=None)
        def scalar_future() -> Number:
            return value() if callable(value) else value

        self.scalars[namespace][key] = scalar_future

    def log_string(self, key: str, value: Callable[[], str] | str, *, namespace: str | None = None) -> None:
        """Logs a string value.

        Args:
            key: The key being logged
            value: The string value being logged
            namespace: An optional logging namespace
        """
        if not self.active:
            raise RuntimeError("The logger is not active")
        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache(maxsize=None)
        def value_future() -> str:
            return value() if callable(value) else value

        self.strings[namespace][key] = value_future

    def log_image(
        self,
        key: str,
        value: Callable[[], np.ndarray | Array | PILImage] | np.ndarray | Array | PILImage,
        *,
        namespace: str | None = None,
        target_resolution: tuple[int, int] | None = (512, 512),
    ) -> None:
        """Logs an image.

        Args:
            key: The key being logged
            value: The image being logged
            namespace: An optional logging namespace
            target_resolution: The target resolution for each image; if None,
                don't resample the images
        """
        if not self.active:
            raise RuntimeError("The logger is not active")
        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache(maxsize=None)
        def image_future() -> PILImage:
            return get_image(value() if callable(value) else value, target_resolution)

        self.images[namespace][key] = image_future

    def log_labeled_image(
        self,
        key: str,
        value: Callable[[], tuple[np.ndarray | Array | PILImage, str]] | tuple[np.ndarray | Array | PILImage, str],
        *,
        namespace: str | None = None,
        max_line_length: int | None = None,
        max_num_lines: int | None = None,
        target_resolution: tuple[int, int] | None = (512, 512),
        line_spacing: int = 2,
        centered: bool = True,
    ) -> None:
        """Logs an image with a label.

        Args:
            key: The key being logged
            value: The image and label being logged
            namespace: An optional logging namespace
            max_line_length: The maximum line length for the label
            max_num_lines: The number of lines of spacing to add to the bottom
                of the image
            target_resolution: The target resolution for each image; if None,
                don't resample the images
            line_spacing: The spacing between adjacent lines
            centered: If set, center the text labels, otherwise align to the left
        """
        if not self.active:
            raise RuntimeError("The logger is not active")
        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache(maxsize=None)
        def image_future() -> PILImage:
            image, label = value() if callable(value) else value
            image = get_image(image, target_resolution)
            return image_with_text(
                image,
                standardize_text(label, max_line_length),
                max_num_lines=max_num_lines,
                line_spacing=line_spacing,
                centered=centered,
            )

        self.images[namespace][key] = image_future

    def log_images(
        self,
        key: str,
        value: (
            Callable[[], Sequence[np.ndarray | Array | PILImage] | np.ndarray | Array]
            | Sequence[np.ndarray | Array | PILImage]
            | np.ndarray
            | Array
        ),
        *,
        namespace: str | None = None,
        max_images: int | None = None,
        target_resolution: tuple[int, int] | None = (256, 256),
        sep: int = 0,
    ) -> None:
        """Logs a set of images.

        The images are tiled to be nearly-square.

        Args:
            key: The key being logged
            value: The images being logged
            namespace: An optional logging namespace
            max_images: The maximum number of images to show; extra images
                are clipped
            target_resolution: The target resolution for each image; if None,
                don't resample the images
            sep: An optional separation amount between adjacent images
        """
        if not self.active:
            raise RuntimeError("The logger is not active")
        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache(maxsize=None)
        def images_future() -> PILImage:
            images = value() if callable(value) else value
            if max_images is not None:
                images = images[:max_images]
            if isinstance(images, Array):
                images = as_numpy(images)
            if isinstance(images, Sequence):
                images = list(images)
            images = [get_image(image, target_resolution) for image in images]
            return tile_images(images, sep)

        self.images[namespace][key] = images_future

    def log_labeled_images(
        self,
        key: str,
        value: (
            Callable[[], tuple[Sequence[np.ndarray | Array | PILImage] | np.ndarray | Array, Sequence[str]]]
            | tuple[Sequence[np.ndarray | Array | PILImage] | np.ndarray | Array, Sequence[str]]
        ),
        *,
        namespace: str | None = None,
        max_images: int | None = None,
        max_line_length: int | None = None,
        max_num_lines: int | None = None,
        target_resolution: tuple[int, int] | None = (256, 256),
        line_spacing: int = 2,
        centered: bool = True,
        sep: int = 0,
    ) -> None:
        """Logs a set of images with labels.

        The images are tiled to be nearly-square.

        Args:
            key: The key being logged
            value: The images and labels being logged
            namespace: An optional logging namespace
            max_images: The maximum number of images to show; extra images
                are clipped
            max_line_length: The maximum line length for the label
            max_num_lines: The number of lines of spacing to add to the bottom
                of the image
            target_resolution: The target resolution for each image; if None,
                don't resample the images
            line_spacing: The spacing between adjacent lines
            centered: If set, center the text labels, otherwise align to the left
            sep: An optional separation amount between adjacent images
        """
        if not self.active:
            raise RuntimeError("The logger is not active")
        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache(maxsize=None)
        def images_future() -> PILImage:
            images, labels = value() if callable(value) else value
            if max_images is not None:
                images = images[:max_images]
                labels = labels[:max_images]
            images = [get_image(image, target_resolution) for image in images]
            images = [
                image_with_text(
                    image,
                    standardize_text(label, max_line_length),
                    max_num_lines=max_num_lines,
                    line_spacing=line_spacing,
                    centered=centered,
                )
                for image, label in zip(images, labels)
            ]
            return tile_images(images, sep)

        self.images[namespace][key] = images_future

    def log_git_state(self, git_state: str) -> None:
        for logger in self.loggers:
            logger.log_git_state(git_state)

    def log_training_code(self, training_code: str) -> None:
        for logger in self.loggers:
            logger.log_training_code(training_code)

    def log_config(self, config: DictConfig) -> None:
        for logger in self.loggers:
            logger.log_config(config)

    def __enter__(self) -> Self:
        self.active = True
        for logger in self.loggers:
            logger.start()
        return self

    def __exit__(self, _t: type[BaseException] | None, _e: BaseException | None, _tr: TracebackType | None) -> None:
        for logger in self.loggers:
            logger.stop()
        self.active = False
