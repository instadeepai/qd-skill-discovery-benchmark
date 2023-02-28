from typing import Dict

import numpy as np
from qdax.utils.metrics import CSVLogger


def log_accumulated_metrics(
    metrics: Dict[str, np.ndarray],
    metric_logger: CSVLogger,
    current_step: int,
    last_step: int,
) -> None:
    """Logs uniformly metrics on the interval [current_step, last_step). Useful when
    metrics are aggregated during a jax.lax.scan and we want to display every points.

    Args:
        metrics: Metrics.
        metric_loggers: CSVLogger.
        current_step: The current step (ie. the step number of the last logged metric).
        last_step: The last step (ie. the step number of the first logged metric).
    """

    for metric_name, metric_value in metrics.items():
        x_values = np.flip(
            np.linspace(
                current_step,
                last_step,
                len(list(metrics.values())[0]),
                endpoint=False,
            )
        )
        for i in range(len(metric_value)):
            metric = {
                "metric_name": metric_name,
                "step": x_values[i],
                "value": int(metric_value[i])
                if "int" in str(metric_value[i].dtype)
                else float(metric_value[i]),
            }

            metric_logger.log(metric)
