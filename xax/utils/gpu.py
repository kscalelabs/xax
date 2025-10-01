"""Defines some utility functions for GPU detection and management."""

import functools
import logging
import subprocess

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=None)
def get_num_gpus() -> int:
    command = "nvidia-smi --query-gpu=index --format=csv --format=csv,noheader"

    try:
        with subprocess.Popen(command.split(), stdout=subprocess.PIPE, universal_newlines=True) as proc:
            stdout = proc.stdout
            assert stdout is not None
            rows = iter(stdout.readline, "")
            return len(list(rows))

    except Exception:
        logger.exception("Caught exception while trying to query `nvidia-smi`")
        return 0