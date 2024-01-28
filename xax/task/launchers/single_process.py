"""Defines a launcher to train a model locally, in a single process."""

from typing import TYPE_CHECKING

from xax.task.base import RawConfigType
from xax.task.launchers.base import BaseLauncher
from xax.utils.logging import configure_logging

if TYPE_CHECKING:
    from xax.task.mixins.runnable import Config, RunnableMixin


def run_single_process_training(
    task: "type[RunnableMixin[Config]]",
    *cfgs: RawConfigType,
    use_cli: bool | list[str] = True,
) -> None:
    configure_logging()
    task_obj = task.get_task(*cfgs, use_cli=use_cli)
    task_obj.run()


class SingleProcessLauncher(BaseLauncher):
    def launch(
        self,
        task: "type[RunnableMixin[Config]]",
        *cfgs: RawConfigType,
        use_cli: bool | list[str] = True,
    ) -> None:
        run_single_process_training(task, *cfgs, use_cli=use_cli)
