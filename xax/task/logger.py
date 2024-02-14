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
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from types import TracebackType
from typing import Callable, Literal, Self, Sequence, TypeVar, get_args

import numpy as np
from jaxtyping import Array
from omegaconf import DictConfig

from xax.core.state import Phase, State
from xax.utils.experiments import IntervalTicker
from xax.utils.logging import LOG_ERROR_SUMMARY, LOG_PING, LOG_STATUS

logger = logging.getLogger(__name__)

T = TypeVar("T")
LogT = TypeVar("LogT")
Number = int | float | Array | np.ndarray

ChannelSelectMode = Literal["first", "last", "mean"]

VALID_VIDEO_CHANNEL_COUNTS = {1, 3}
VALID_AUDIO_CHANNEL_COUNTS = {1, 2}
TARGET_FPS = 12
DEFAULT_NAMESPACE = "value"


NAMESPACE_STACK: list[str] = []


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


@dataclass
class LogImage:
    pixels: Array


@dataclass
class LogAudio:
    frames: Array
    sample_rate: int


@dataclass
class LogVideo:
    frames: Array


@dataclass
class LogPointCloud:
    xyz: Array
    colors: Array | None


@dataclass
class LogLine:
    state: State
    scalars: dict[str, dict[str, Number]]
    strings: dict[str, dict[str, str]]
    images: dict[str, dict[str, LogImage]]
    audios: dict[str, dict[str, LogAudio]]
    videos: dict[str, dict[str, LogVideo]]
    point_cloud: dict[str, dict[str, LogPointCloud]]


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
        self.images: dict[str, dict[str, Callable[[], Array]]] = defaultdict(dict)
        self.audio: dict[str, dict[str, Callable[[], tuple[Array, int]]]] = defaultdict(dict)
        self.videos: dict[str, dict[str, Callable[[], Array]]] = defaultdict(dict)
        self.histograms: dict[str, dict[str, Callable[[], Array]]] = defaultdict(dict)
        self.point_clouds: dict[str, dict[str, Callable[[], tuple[Array, Array | None]]]] = defaultdict(dict)
        self.default_namespace = default_namespace
        self.loggers: list[LoggerImpl] = []

        # Registers a logging handler to route log messages to the logger.
        root_logger = logging.getLogger()
        ToastHandler(self).add_for_logger(root_logger)

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
            audios={k: {kk: LogAudio(*v()) for kk, v in v.items()} for k, v in self.audio.items()},
            videos={k: {kk: LogVideo(v()) for kk, v in v.items()} for k, v in self.videos.items()},
            point_cloud={k: {kk: LogPointCloud(*v()) for kk, v in v.items()} for k, v in self.point_clouds.items()},
        )

    def clear(self) -> None:
        self.scalars.clear()
        self.strings.clear()
        self.images.clear()
        self.audio.clear()
        self.videos.clear()
        self.histograms.clear()
        self.point_clouds.clear()

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
        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache(maxsize=None)
        def value_future() -> str:
            return value() if callable(value) else value

        self.strings[namespace][key] = value_future

    def log_image(
        self,
        key: str,
        value: Callable[[], Array] | Array,
        *,
        namespace: str | None = None,
        keep_resolution: bool = False,
    ) -> None:
        """Logs an image.

        Args:
            key: The key being logged
            value: The image being logged; can be (C, H, W), (H, W, C) or (H, W)
                as an RGB (3 channel) or grayscale (1 channel) image
            namespace: An optional logging namespace
            keep_resolution: If set, keep the image resolution the same,
                otherwise upscale or downscale the image to a standard
                resolution
        """
        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache(maxsize=None)
        def image_future() -> Array:
            raise NotImplementedError

        self.images[namespace][key] = image_future

    def log_labeled_image(
        self,
        key: str,
        value: Callable[[], tuple[Array, str]] | tuple[Array, str],
        *,
        namespace: str | None = None,
        max_line_length: int | None = None,
        keep_resolution: bool = False,
        centered: bool = True,
    ) -> None:
        """Logs an image with a label.

        Args:
            key: The key being logged
            value: The image and label being logged; the image can be (C, H, W),
                (H, W, C) or (H, W) as an RGB (3 channel) or grayscale
                (1 channel) image
            namespace: An optional logging namespace
            max_line_length: Labels longer than this length are wrapped around
            keep_resolution: If set, keep the image resolution the same,
                otherwise upscale or downscale the image to a standard
                resolution
            centered: If set, center the text labels, otherwise align to the
                left
        """
        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache(maxsize=None)
        def labeled_image_future() -> Array:
            raise NotImplementedError

        self.images[namespace][key] = labeled_image_future

    def log_images(
        self,
        key: str,
        value: Callable[[], Array] | Array,
        *,
        namespace: str | None = None,
        keep_resolution: bool = False,
        max_images: int | None = None,
        sep: int = 0,
    ) -> None:
        """Logs a set of images.

        The images are tiled to be nearly-square.

        Args:
            key: The key being logged
            value: The images being logged; can be (B, C, H, W), (B, H, W, C)
                or (B H, W) as an RGB (3 channel) or grayscale (1 channel) image
            namespace: An optional logging namespace
            keep_resolution: If set, keep the image resolution the same,
                otherwise upscale or downscale the image to a standard
                resolution
            max_images: The maximum number of images to show; extra images
                are clipped
            sep: An optional separation amount between adjacent images
        """
        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache(maxsize=None)
        def images_future() -> Array:
            raise NotImplementedError

        self.images[namespace][key] = images_future

    def log_labeled_images(
        self,
        key: str,
        value: Callable[[], tuple[Array, Sequence[str]]] | tuple[Array, Sequence[str]],
        *,
        namespace: str | None = None,
        max_line_length: int | None = None,
        keep_resolution: bool = False,
        max_images: int | None = None,
        sep: int = 0,
        centered: bool = True,
    ) -> None:
        """Logs a set of images with labels.

        The images are tiled to be nearly-square.

        Args:
            key: The key being logged
            value: The images and labels being logged; images can be
                (B, C, H, W), (B, H, W, C) or (B, H, W) as an RGB (3 channel)
                or grayscale (1 channel) image, with exactly B labels
            namespace: An optional logging namespace
            max_line_length: Labels longer than this length are wrapped around
            keep_resolution: If set, keep the image resolution the same,
                otherwise upscale or downscale the image to a standard
                resolution
            max_images: The maximum number of images to show; extra images
                are clipped
            sep: An optional separation amount between adjacent images
            centered: If set, center the text labels, otherwise align to the
                left
        """
        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache(maxsize=None)
        def labeled_images_future() -> Array:
            raise NotImplementedError

        self.images[namespace][key] = labeled_images_future

    def log_audio(
        self,
        key: str,
        value: Callable[[], Array] | Array,
        *,
        namespace: str | None = None,
        sample_rate: int = 44100,
        log_spec: bool = False,
        n_fft_ms: float = 32.0,
        hop_length_ms: float | None = None,
        channel_select_mode: ChannelSelectMode = "first",
        keep_resolution: bool = False,
    ) -> None:
        """Logs an audio clip.

        Args:
            key: The key being logged
            value: The audio clip being logged; can be (C, T) or (T) as
                a mono (1 channel) or stereo (2 channel) audio clip
            namespace: An optional logging namespace
            sample_rate: The sample rate of the audio clip
            log_spec: If set, also log the spectrogram
            n_fft_ms: FFT size, in milliseconds
            hop_length_ms: The FFT hop length, in milliseconds
            channel_select_mode: How to select the channel if the audio is
                stereo; can be "first", "last", or "mean"; this is only used
                for the spectrogram
            keep_resolution: If set, keep the resolution of the
                spectrogram; otherwise, make human-viewable
        """
        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache(maxsize=None)
        def raw_audio_future() -> Array:
            raise NotImplementedError

        @functools.lru_cache(maxsize=None)
        def audio_future() -> tuple[Array, int]:
            raise NotImplementedError

        self.audio[namespace][key] = audio_future

        if log_spec:
            # Using a unique key for the spectrogram is very important because
            # otherwise Tensorboard will have some issues.
            self.log_spectrogram(
                key=f"{key}_spec",
                value=raw_audio_future,
                namespace=namespace,
                sample_rate=sample_rate,
                n_fft_ms=n_fft_ms,
                hop_length_ms=hop_length_ms,
                channel_select_mode=channel_select_mode,
                keep_resolution=keep_resolution,
            )

    def log_audios(
        self,
        key: str,
        value: Callable[[], Array] | Array,
        *,
        namespace: str | None = None,
        sep_ms: float = 0.0,
        max_audios: int | None = None,
        sample_rate: int = 44100,
        log_spec: bool = False,
        n_fft_ms: float = 32.0,
        hop_length_ms: float | None = None,
        channel_select_mode: ChannelSelectMode = "first",
        spec_sep: int = 0,
        keep_resolution: bool = False,
    ) -> None:
        """Logs multiple audio clips.

        Args:
            key: The key being logged
            value: The audio clip being logged; can be (B, C, T) or (B, T) as
                a mono (1 channel) or stereo (2 channel) audio clip, with
                exactly B clips
            namespace: An optional logging namespace
            sep_ms: An optional separation amount between adjacent audio clips
            max_audios: An optional maximum number of audio clips to log
            sample_rate: The sample rate of the audio clip
            log_spec: If set, also log the spectrogram
            n_fft_ms: FFT size, in milliseconds
            hop_length_ms: The FFT hop length, in milliseconds
            channel_select_mode: How to select the channel if the audio is
                stereo; can be "first", "last", or "mean"; this is only used
                for the spectrogram
            spec_sep: An optional separation amount between adjacent
                spectrograms
            keep_resolution: If set, keep the resolution of the
                spectrogram; otherwise, make human-viewable
        """
        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache(maxsize=None)
        def raw_audio_future() -> Array:
            raise NotImplementedError

        @functools.lru_cache(maxsize=None)
        def audio_future() -> tuple[Array, int]:
            raise NotImplementedError

        self.audio[namespace][key] = audio_future

        if log_spec:
            # Using a unique key for the spectrogram is very important because
            # otherwise Tensorboard will have some issues.
            self.log_spectrograms(
                key=f"{key}_spec",
                value=raw_audio_future,
                namespace=namespace,
                max_audios=max_audios,
                sample_rate=sample_rate,
                n_fft_ms=n_fft_ms,
                hop_length_ms=hop_length_ms,
                channel_select_mode=channel_select_mode,
                spec_sep=spec_sep,
                keep_resolution=keep_resolution,
            )

    def log_spectrogram(
        self,
        key: str,
        value: Callable[[], Array] | Array,
        *,
        namespace: str | None = None,
        sample_rate: int = 44100,
        n_fft_ms: float = 32.0,
        hop_length_ms: float | None = None,
        channel_select_mode: ChannelSelectMode = "first",
        keep_resolution: bool = False,
    ) -> None:
        """Logs spectrograms of an audio clip.

        Args:
            key: The key being logged
            value: The audio clip being logged; can be (C, T) or (T) as
                a mono (1 channel) or stereo (2 channel) audio clip
            namespace: An optional logging namespace
            sample_rate: The sample rate of the audio clip
            n_fft_ms: FFT size, in milliseconds
            hop_length_ms: The FFT hop length, in milliseconds
            channel_select_mode: How to select the channel if the audio is
                stereo; can be "first", "last", or "mean"; this is only used
                for the spectrogram
            keep_resolution: If set, keep the resolution of the
                spectrogram; otherwise, make human-viewable
        """
        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache(maxsize=None)
        def spec_future() -> Array:
            raise NotImplementedError

        self.images[namespace][key] = spec_future

    def log_spectrograms(
        self,
        key: str,
        value: Callable[[], Array] | Array,
        *,
        namespace: str | None = None,
        max_audios: int | None = None,
        sample_rate: int = 44100,
        n_fft_ms: float = 32.0,
        hop_length_ms: float | None = None,
        channel_select_mode: ChannelSelectMode = "first",
        spec_sep: int = 0,
        keep_resolution: bool = False,
    ) -> None:
        """Logs spectrograms of audio clips.

        Args:
            key: The key being logged
            value: The audio clip being logged; can be (B, C, T) or (B, T) as
                a mono (1 channel) or stereo (2 channel) audio clip, with
                exactly B clips
            namespace: An optional logging namespace
            max_audios: An optional maximum number of audio clips to log
            sample_rate: The sample rate of the audio clip
            n_fft_ms: FFT size, in milliseconds
            hop_length_ms: The FFT hop length, in milliseconds
            channel_select_mode: How to select the channel if the audio is
                stereo; can be "first", "last", or "mean"; this is only used
                for the spectrogram
            spec_sep: An optional separation amount between adjacent
                spectrograms
            keep_resolution: If set, keep the resolution of the
                spectrogram; otherwise, make human-viewable
        """
        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache(maxsize=None)
        def spec_future() -> Array:
            raise NotImplementedError

        self.images[namespace][key] = spec_future

    def log_video(
        self,
        key: str,
        value: Callable[[], Array] | Array,
        *,
        namespace: str | None = None,
        fps: int | None = None,
        length: float | None = None,
    ) -> None:
        """Logs a video.

        Args:
            key: The key being logged
            value: The video being logged; the video can be (T, C, H, W),
                (T, H, W, C) or (T, H, W) as an RGB (3 channel) or grayscale
                (1 channel) video
            namespace: An optional logging namespace
            fps: The video frames per second
            length: The desired video length, in seconds, at the target FPS
        """
        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache(maxsize=None)
        def video_future() -> Array:
            raise NotImplementedError

        self.videos[namespace][key] = video_future

    def log_videos(
        self,
        key: str,
        value: Callable[[], Array | list[Array]] | Array | list[Array],
        *,
        namespace: str | None = None,
        max_videos: int | None = None,
        sep: int = 0,
        fps: int | None = None,
        length: int | None = None,
    ) -> None:
        """Logs a set of video.

        Args:
            key: The key being logged
            value: The videos being logged; the video can be (B, T, C, H, W),
                (B, T, H, W, C) or (B T, H, W) as an RGB (3 channel) or
                grayscale (1 channel) video
            namespace: An optional logging namespace
            max_videos: The maximum number of videos to show; extra images
                are clipped
            sep: An optional separation amount between adjacent videos
            fps: The video frames per second
            length: The desired video length, in seconds, at the target FPS
        """
        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache(maxsize=None)
        def videos_future() -> Array:
            raise NotImplementedError

        self.videos[namespace][key] = videos_future

    def log_histogram(
        self,
        key: str,
        value: Callable[[], Array] | Array,
        *,
        namespace: str | None = None,
    ) -> None:
        """Logs a histogram.

        Args:
            key: The key being logged
            value: The values to create a histogram from, with arbitrary shape
            namespace: An optional logging namespace
        """
        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache(maxsize=None)
        def histogram_future() -> Array:
            raise NotImplementedError

        self.histograms[namespace][key] = histogram_future

    def log_point_cloud(
        self,
        key: str,
        value: Callable[[], Array] | Array,
        *,
        namespace: str | None = None,
        max_points: int = 1000,
        colors: Callable[[], Array] | Array | None = None,
    ) -> None:
        """Logs a point cloud.

        Args:
            key: The key being logged
            value: The point cloud values, with shape (N, 3) or (B, ..., 3);
                can pass multiple batches in order to show multiple point
                clouds
            namespace: An optional logging namespace
            max_points: An optional maximum number of points in the point cloud
            colors: An optional color for each point, with the same shape as
                the point cloud
        """
        namespace = self.resolve_namespace(namespace)

        @functools.lru_cache(maxsize=None)
        def point_cloud_future() -> tuple[Array, Array | None]:
            raise NotImplementedError

        self.point_clouds[namespace][key] = point_cloud_future

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
        for logger in self.loggers:
            logger.start()
        return self

    def __exit__(self, _t: type[BaseException] | None, _e: BaseException | None, _tr: TracebackType | None) -> None:
        for logger in self.loggers:
            logger.stop()
