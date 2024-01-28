"""Logging utilities."""

import logging
import math
import sys

from xax.core.conf import load_user_config
from xax.utils.text import Color, color_parts, colored

# Logging level to show on all ranks.
INFOALL: int = logging.INFO + 1
DEBUGALL: int = logging.DEBUG + 1


class RankFilter(logging.Filter):
    def __init__(self, *, rank: int | None = None) -> None:
        """Logging filter which filters out INFO logs on non-zero ranks.

        Args:
            rank: The current rank
        """
        super().__init__()

        self.rank = rank

        # Log using INFOALL to show on all ranks.
        logging.addLevelName(INFOALL, "INFOALL")
        logging.addLevelName(DEBUGALL, "DEBUGALL")
        levels_to_log_all_ranks = (DEBUGALL, INFOALL, logging.CRITICAL, logging.ERROR, logging.WARNING)
        self.log_all_ranks = {logging.getLevelName(level) for level in levels_to_log_all_ranks}

    def filter(self, record: logging.LogRecord) -> bool:
        if self.rank is None or self.rank == 0:
            return True
        if record.levelname in self.log_all_ranks:
            return True
        return False


class ColoredFormatter(logging.Formatter):
    """Defines a custom formatter for displaying logs."""

    RESET_SEQ = "\033[0m"
    COLOR_SEQ = "\033[1;%dm"
    BOLD_SEQ = "\033[1m"

    COLORS: dict[str, Color] = {
        "WARNING": "yellow",
        "INFOALL": "magenta",
        "INFO": "cyan",
        "DEBUGALL": "grey",
        "DEBUG": "grey",
        "CRITICAL": "yellow",
        "FATAL": "red",
        "ERROR": "red",
    }

    def __init__(
        self,
        *,
        prefix: str | None = None,
        rank: int | None = None,
        world_size: int | None = None,
        use_color: bool = True,
    ) -> None:
        asc_start, asc_end = color_parts("grey")
        message = "{levelname:^19s} " + asc_start + "{asctime}" + asc_end + " [{name}] {message}"
        if prefix is not None:
            message = colored(prefix, "white") + " " + message
        if rank is not None or world_size is not None:
            assert rank is not None and world_size is not None
            digits = int(math.log10(world_size) + 1)
            message = "[" + colored(f"{rank:>{digits}}", "blue", bold=True) + "] " + message
        super().__init__(message, style="{", datefmt="%Y-%m-%d %H:%M:%S")

        self.rank = rank
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname

        match levelname:
            case "DEBUG":
                record.levelname = ""
            case "INFOALL":
                record.levelname = "INFO"
            case "DEBUGALL":
                record.levelname = "DEBUG"

        if record.levelname and self.use_color and levelname in self.COLORS:
            record.levelname = colored(record.levelname, self.COLORS[levelname], bold=True)
        return logging.Formatter.format(self, record)


def configure_logging(*, rank: int | None = None, world_size: int | None = None) -> None:
    """Instantiates logging.

    This captures logs and reroutes them to the Toasts module, which is
    pretty similar to Python logging except that the API is a lot easier to
    interact with.

    Args:
        prefix: An optional prefix to add to the logger
        rank: The current rank, or None if not using multiprocessing
        world_size: The total world size, or None if not using multiprocessing
    """
    if rank is not None or world_size is not None:
        assert rank is not None and world_size is not None
    root_logger = logging.getLogger()

    config = load_user_config().logging

    # Captures warnings from the warnings module.
    logging.captureWarnings(True)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(ColoredFormatter(rank=rank, world_size=world_size))
    stream_handler.addFilter(RankFilter(rank=rank))

    root_logger.addHandler(stream_handler)
    root_logger.setLevel(logging._nameToLevel[config.log_level])

    # Avoid junk logs from other libraries.
    if config.hide_third_party_logs:
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)
        logging.getLogger("torch").setLevel(logging.WARNING)
