"""Defines utility functions for interfacing with Tensorboard."""

import functools
import io
import os
import tempfile
import time
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import PIL.Image
from PIL.Image import Image as PILImage
from tensorboard.compat.proto.config_pb2 import RunMetadata
from tensorboard.compat.proto.event_pb2 import Event, TaggedRunMetadata
from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.proto.summary_pb2 import Summary, SummaryMetadata
from tensorboard.compat.proto.tensor_pb2 import TensorProto
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from tensorboard.plugins.text.plugin_data_pb2 import TextPluginData
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from xax.core.state import Phase

ImageShape = Literal["HWC", "CHW", "HW", "NHWC", "NCHW", "NHW"]


class TensorboardProtobufWriter:
    def __init__(
        self,
        log_directory: str | Path,
        max_queue_size: int = 10,
        flush_seconds: float = 120.0,
        filename_suffix: str = "",
    ) -> None:
        super().__init__()

        self.log_directory = Path(log_directory)
        self.max_queue_size = max_queue_size
        self.flush_seconds = flush_seconds
        self.filename_suffix = filename_suffix

    @functools.cached_property
    def event_writer(self) -> EventFileWriter:
        return EventFileWriter(
            logdir=str(self.log_directory),
            max_queue_size=self.max_queue_size,
            flush_secs=self.flush_seconds,
            filename_suffix=self.filename_suffix,
        )

    def add_event(
        self,
        event: Event,
        step: int | None = None,
        walltime: float | None = None,
    ) -> None:
        event.wall_time = time.time() if walltime is None else walltime
        if step is not None:
            event.step = int(step)
        self.event_writer.add_event(event)

    def add_summary(
        self,
        summary: Summary,
        global_step: int | None = None,
        walltime: float | None = None,
    ) -> None:
        event = Event(summary=summary)
        self.add_event(event, step=global_step, walltime=walltime)

    def add_graph(
        self,
        graph: GraphDef,
        run_metadata: RunMetadata | None = None,
        walltime: float | None = None,
    ) -> None:
        event = Event(graph_def=graph.SerializeToString())
        self.add_event(event, walltime=walltime)
        if run_metadata is not None:
            trm = TaggedRunMetadata(tag="step1", run_metadata=run_metadata.SerializeToString())
            event = Event(tagged_run_metadata=trm)
            self.add_event(event, walltime=walltime)

    def flush(self) -> None:
        self.event_writer.flush()

    def close(self) -> None:
        self.event_writer.close()


class TensorboardWriter:
    """Defines a class for writing artifacts to Tensorboard.

    Parameters:
        log_directory: The directory to write logs to.
        max_queue_size: The maximum queue size.
        flush_seconds: How often to flush logs.
        filename_suffix: The filename suffix to use.
    """

    def __init__(
        self,
        log_directory: str | Path,
        max_queue_size: int = 10,
        flush_seconds: float = 120.0,
        filename_suffix: str = "",
    ) -> None:
        super().__init__()

        self.pb_writer = TensorboardProtobufWriter(
            log_directory=log_directory,
            max_queue_size=max_queue_size,
            flush_seconds=flush_seconds,
            filename_suffix=filename_suffix,
        )

    def add_scalar(
        self,
        tag: str,
        value: float,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = True,
        double_precision: bool = False,
    ) -> None:
        if new_style:
            self.pb_writer.add_summary(
                Summary(
                    value=[
                        Summary.Value(
                            tag=tag,
                            tensor=(
                                TensorProto(double_val=[value], dtype="DT_DOUBLE")
                                if double_precision
                                else TensorProto(float_val=[value], dtype="DT_FLOAT")
                            ),
                            metadata=SummaryMetadata(
                                plugin_data=SummaryMetadata.PluginData(
                                    plugin_name="scalars",
                                ),
                            ),
                        )
                    ],
                ),
                global_step=global_step,
                walltime=walltime,
            )
        else:
            self.pb_writer.add_summary(
                Summary(
                    value=[
                        Summary.Value(
                            tag=tag,
                            simple_value=value,
                        ),
                    ],
                ),
                global_step=global_step,
                walltime=walltime,
            )

    def add_image(
        self,
        tag: str,
        value: PILImage,
        global_step: int | None = None,
        walltime: float | None = None,
    ) -> None:
        output = io.BytesIO()
        value.convert("RGB").save(output, format="PNG")
        image_string = output.getvalue()
        output.close()

        self.pb_writer.add_summary(
            Summary(
                value=[
                    Summary.Value(
                        tag=tag,
                        image=Summary.Image(
                            height=value.height,
                            width=value.width,
                            colorspace=3,  # RGB
                            encoded_image_string=image_string,
                        ),
                    ),
                ],
            ),
            global_step=global_step,
            walltime=walltime,
        )

    def add_video(
        self,
        tag: str,
        value: np.ndarray,
        global_step: int | None = None,
        walltime: float | None = None,
        fps: int = 30,
    ) -> None:
        assert value.ndim == 4, "Video must be 4D array (T, H, W, C)"
        images = [PIL.Image.fromarray(frame) for frame in value]

        # Create temporary file for GIF
        temp_file = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
        try:
            images[0].save(temp_file.name, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0)
            with open(temp_file.name, "rb") as f:
                video_string = f.read()

        finally:
            # Clean up temporary file
            try:
                os.remove(temp_file.name)
            except OSError:
                pass

        # Add to summary
        self.pb_writer.add_summary(
            Summary(
                value=[
                    Summary.Value(
                        tag=tag,
                        image=Summary.Image(
                            height=value.shape[1],
                            width=value.shape[2],
                            colorspace=value.shape[3],
                            encoded_image_string=video_string,
                        ),
                    ),
                ],
            ),
            global_step=global_step,
            walltime=walltime,
        )

    def add_text(
        self,
        tag: str,
        value: str,
        global_step: int | None = None,
        walltime: float | None = None,
    ) -> None:
        self.pb_writer.add_summary(
            Summary(
                value=[
                    Summary.Value(
                        tag=tag + "/text_summary",
                        metadata=SummaryMetadata(
                            plugin_data=SummaryMetadata.PluginData(
                                plugin_name="text", content=TextPluginData(version=0).SerializeToString()
                            ),
                        ),
                        tensor=TensorProto(
                            dtype="DT_STRING",
                            string_val=[value.encode(encoding="utf_8")],
                            tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]),
                        ),
                    ),
                ],
            ),
            global_step=global_step,
            walltime=walltime,
        )


class TensorboardWriterKwargs(TypedDict):
    max_queue_size: int
    flush_seconds: float
    filename_suffix: str


class TensorboardWriters:
    def __init__(
        self,
        log_directory: str | Path,
        max_queue_size: int = 10,
        flush_seconds: float = 120.0,
        filename_suffix: str = "",
    ) -> None:
        super().__init__()

        self.log_directory = Path(log_directory)

        self.kwargs: TensorboardWriterKwargs = {
            "max_queue_size": max_queue_size,
            "flush_seconds": flush_seconds,
            "filename_suffix": filename_suffix,
        }

    @functools.cached_property
    def train_writer(self) -> TensorboardWriter:
        return TensorboardWriter(self.log_directory / "train", **self.kwargs)

    @functools.cached_property
    def valid_writer(self) -> TensorboardWriter:
        return TensorboardWriter(self.log_directory / "valid", **self.kwargs)

    def writer(self, phase: Phase) -> TensorboardWriter:
        match phase:
            case "train":
                return self.train_writer
            case "valid":
                return self.valid_writer
            case _:
                raise NotImplementedError(f"Unexpected phase: {phase}")
