"""Defines a launcher to train a model using multiple processes for multi-GPU training."""

import multiprocessing as mp
import os
from typing import TYPE_CHECKING

import jax

from xax.task.base import RawConfigType
from xax.task.launchers.base import BaseLauncher
from xax.task.mixins.gpu_stats import get_num_gpus
from xax.utils.logging import configure_logging

if TYPE_CHECKING:
    from xax.task.mixins.runnable import Config, RunnableMixin


def _worker_process(
    task: "type[RunnableMixin[Config]]",
    cfgs: tuple[RawConfigType, ...],
    use_cli: bool | list[str],
    gpu_id: int,
) -> None:
    """Worker process for a specific GPU."""
    # Set CUDA_VISIBLE_DEVICES to only show this GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Configure JAX to use the specific device
    jax.config.update("jax_default_device", jax.devices("gpu")[0])

    # Create and run the task for this GPU
    task_obj = task.get_task(*cfgs, use_cli=use_cli)
    task_obj.add_logger_handlers(configure_logging(prefix=f"GPU-{gpu_id}"))
    task_obj.run()


def run_multi_process_training(
    task: "type[RunnableMixin[Config]]",
    *cfgs: RawConfigType,
    use_cli: bool | list[str] = True,
) -> None:
    """Runs training using multiple processes for multi-GPU support."""
    num_gpus = get_num_gpus()

    if num_gpus <= 1:
        # Fall back to single process if no multiple GPUs available
        from xax.task.launchers.single_process import run_single_process_training

        run_single_process_training(task, *cfgs, use_cli=use_cli)
        return

    logger = configure_logging()
    logger.info("Starting multi-process training with %d GPUs", num_gpus)

    # Set up multiprocessing context
    ctx = mp.get_context("spawn")

    # Create processes for each GPU
    processes = []

    for gpu_id in range(num_gpus):
        # Start the process
        process = ctx.Process(
            target=_worker_process,
            args=(task, cfgs, use_cli, gpu_id),
        )
        process.start()
        processes.append(process)

    # Wait for all processes to complete
    for process in processes:
        process.join()

    logger.info("Multi-process training completed")


class MultiProcessLauncher(BaseLauncher):
    """Launcher for multi-process training across multiple GPUs."""

    def launch(
        self,
        task: "type[RunnableMixin[Config]]",
        *cfgs: RawConfigType,
        use_cli: bool | list[str] = True,
    ) -> None:
        run_multi_process_training(task, *cfgs, use_cli=use_cli)
