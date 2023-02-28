from dataclasses import dataclass


@dataclass
class LoggingConfig:
    """
    Configuration for the logging.
    """

    log_period: int
    save_checkpoints_period: int
    metrics_subsample: int
