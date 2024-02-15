"""Defines a logger that calls a callback function with the log line."""

from typing import Callable

from omegaconf import DictConfig

from xax.task.logger import LogError, LogErrorSummary, LoggerImpl, LogLine, LogPing, LogStatus


class CallbackLogger(LoggerImpl):
    def __init__(
        self,
        *,
        callback: Callable[[LogLine], None] = lambda x: None,
        error_summary_callback: Callable[[LogErrorSummary], None] = lambda x: None,
        error_callback: Callable[[LogError], None] = lambda x: None,
        status_callback: Callable[[LogStatus], None] = lambda x: None,
        ping_callback: Callable[[LogPing], None] = lambda x: None,
        git_state_callback: Callable[[str], None] = lambda x: None,
        training_code_callback: Callable[[str], None] = lambda x: None,
        config_callback: Callable[[DictConfig], None] = lambda x: None,
    ) -> None:
        super().__init__()

        self.callback = callback
        self.error_summary_callback = error_summary_callback
        self.error_callback = error_callback
        self.status_callback = status_callback
        self.ping_callback = ping_callback
        self.git_state_callback = git_state_callback
        self.training_code_callback = training_code_callback
        self.config_callback = config_callback

    def write(self, line: LogLine) -> None:
        self.callback(line)

    def write_error_summary(self, error_summary: LogErrorSummary) -> None:
        self.error_summary_callback(error_summary)

    def write_error(self, error: LogError) -> None:
        self.error_callback(error)

    def write_status(self, status: LogStatus) -> None:
        self.status_callback(status)

    def write_ping(self, ping: LogPing) -> None:
        self.ping_callback(ping)

    def log_git_state(self, git_state: str) -> None:
        self.git_state_callback(git_state)

    def log_training_code(self, training_code: str) -> None:
        self.training_code_callback(training_code)

    def log_config(self, config: DictConfig) -> None:
        self.config_callback(config)
