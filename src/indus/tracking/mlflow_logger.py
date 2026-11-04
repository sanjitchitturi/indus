"""MLflow experiment tracking wrapper for Indus training and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
from omegaconf import DictConfig, OmegaConf

from indus.logging_utils import get_logger

logger = get_logger(__name__)


class MLflowLogger:
    """Structured MLflow logging for training runs and evaluations.

    Handles experiment creation, parameter logging, step-level metrics,
    and artifact uploads in a consistent interface.

    Args:
        cfg: Full Hydra configuration (logged as params).
        tracking_uri: MLflow tracking server URI or local path.
        experiment_name: Name of the MLflow experiment.
        run_name: Optional run display name.
    """

    def __init__(
        self,
        cfg: DictConfig,
        tracking_uri: str = "./mlruns",
        experiment_name: str = "indus-qlora",
        run_name: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        self._run: mlflow.ActiveRun | None = None

    def start_run(self) -> None:
        """Start a new MLflow run and log all configuration parameters."""
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self._run = mlflow.start_run(run_name=self.run_name)

        flat_cfg = _flatten_config(OmegaConf.to_container(self.cfg, resolve=True))
        for key, value in flat_cfg.items():
            try:
                mlflow.log_param(key, value)
            except mlflow.exceptions.MlflowException:
                logger.debug("Skipping param '%s' (already logged or too long)", key)

        logger.info(
            "MLflow run started: experiment='%s', run_id='%s'",
            self.experiment_name,
            self._run.info.run_id,
        )

    def log_step(self, step: int, loss: float, grad_norm: float, lr: float) -> None:
        """Log training step metrics."""
        mlflow.log_metrics(
            {"train/loss": loss, "train/grad_norm": grad_norm, "train/learning_rate": lr},
            step=step,
        )

    def log_val(self, step: int, val_loss: float) -> None:
        """Log validation metrics."""
        mlflow.log_metric("val/loss", val_loss, step=step)

    def log_eval_metrics(self, metrics: dict[str, float], prefix: str = "eval") -> None:
        """Log evaluation metrics (ROUGE, BLEU, BERTScore)."""
        for key, value in metrics.items():
            mlflow.log_metric(f"{prefix}/{key}", value)
        logger.info("Logged %d evaluation metrics with prefix '%s'", len(metrics), prefix)

    def log_memory_report(self, report: dict[str, Any]) -> None:
        """Log memory profiling data as metrics."""
        if "peak_gpu_gb" in report:
            mlflow.log_metric("system/peak_gpu_gb", report["peak_gpu_gb"])
        if "delta_gpu_allocated_gb" in report:
            mlflow.log_metric("system/delta_gpu_gb", report["delta_gpu_allocated_gb"])

    def log_artifact(self, local_path: str | Path, artifact_subdir: str | None = None) -> None:
        """Upload a local file or directory as an MLflow artifact."""
        path = Path(local_path)
        if not path.exists():
            logger.warning("Artifact path does not exist: %s", path)
            return
        if path.is_dir():
            mlflow.log_artifacts(str(path), artifact_path=artifact_subdir)
        else:
            mlflow.log_artifact(str(path), artifact_path=artifact_subdir)
        logger.info("Artifact logged: %s", path)

    def log_model_info(self, trainable_params: int, total_params: int, peak_gpu_gb: float) -> None:
        """Log model architecture and resource information."""
        mlflow.log_params(
            {
                "model/trainable_params": trainable_params,
                "model/total_params": total_params,
                "model/trainable_pct": round(100.0 * trainable_params / total_params, 4),
            }
        )
        mlflow.log_metric("system/peak_gpu_gb", peak_gpu_gb)

    def end_run(self) -> None:
        """End the current MLflow run."""
        if self._run is not None:
            mlflow.end_run()
            logger.info("MLflow run ended: %s", self._run.info.run_id)
            self._run = None

    @property
    def run_id(self) -> str | None:
        """Return the active run ID, or None."""
        return self._run.info.run_id if self._run else None


def _flatten_config(
    d: dict[str, Any] | Any,
    parent_key: str = "",
    sep: str = ".",
) -> dict[str, Any]:
    """Recursively flatten a nested dictionary for MLflow param logging."""
    items: list[tuple[str, Any]] = []
    if not isinstance(d, dict):
        return {parent_key: d} if parent_key else {}

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_config(v, new_key, sep).items())
        elif isinstance(v, (list, tuple)):
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))

    return dict(items)
