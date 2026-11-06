"""Run a sweep of experiment configurations for systematic comparison.

This script orchestrates multiple training and evaluation runs across
different hyperparameter settings. Results are logged to MLflow.

Usage:
    python scripts/run_sweep.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from indus.logging_utils import get_logger, setup_logging

SWEEP_CONFIGS: list[dict[str, str]] = [
    {"training": "qlora_r8", "training.learning_rate": "2e-4", "mlflow.run_name": "r8_lr2e4"},
    {"training": "qlora_r16", "training.learning_rate": "2e-4", "mlflow.run_name": "r16_lr2e4"},
    {"training": "qlora_r32", "training.learning_rate": "2e-4", "mlflow.run_name": "r32_lr2e4"},
    {
        "training": "qlora_r16",
        "training.lora.r": "64",
        "training.lora.lora_alpha": "128",
        "training.batch_size": "2",
        "training.gradient_accumulation_steps": "8",
        "training.learning_rate": "2e-4",
        "mlflow.run_name": "r64_lr2e4",
    },
    {"training": "qlora_r16", "training.learning_rate": "1e-4", "mlflow.run_name": "r16_lr1e4"},
    {"training": "qlora_r16", "training.learning_rate": "5e-4", "mlflow.run_name": "r16_lr5e4"},
    {"training": "qlora_r16", "training.learning_rate": "1e-3", "mlflow.run_name": "r16_lr1e3"},
    {
        "training": "qlora_r16",
        "training.quantization.bnb_4bit_quant_type": "fp4",
        "mlflow.run_name": "r16_fp4",
    },
    {
        "training": "qlora_r16",
        "training.quantization.bnb_4bit_use_double_quant": "false",
        "mlflow.run_name": "r16_no_double_quant",
    },
    {
        "training": "qlora_r16",
        "training.lora.lora_dropout": "0.1",
        "mlflow.run_name": "r16_drop0.1",
    },
    {
        "training": "qlora_r16",
        "training.lora.lora_alpha": "16",
        "mlflow.run_name": "r16_alpha16",
    },
    {"training": "qlora_r16", "training.num_epochs": "5", "mlflow.run_name": "r16_5ep"},
]


def main() -> None:
    """Execute the full sweep of experiments."""
    setup_logging(level="INFO")
    logger = get_logger("sweep")
    logger.info("Starting experiment sweep with %d configurations", len(SWEEP_CONFIGS))

    results: list[dict[str, str | int]] = []
    train_script = str(Path(__file__).resolve().parent / "train.py")

    for i, overrides in enumerate(SWEEP_CONFIGS):
        run_name = overrides.get("mlflow.run_name", f"sweep_{i}")
        logger.info("--- Experiment %d/%d: %s ---", i + 1, len(SWEEP_CONFIGS), run_name)

        cmd = [sys.executable, train_script]
        for key, value in overrides.items():
            cmd.append(f"{key}={value}")
        cmd.append(f"output_dir=./outputs/sweep/{run_name}")

        logger.info("Command: %s", " ".join(cmd))

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            if result.returncode == 0:
                logger.info("Experiment '%s' completed successfully", run_name)
                results.append({"name": run_name, "status": "success", "returncode": 0})
            else:
                logger.error(
                    "Experiment '%s' failed (rc=%d):\n%s",
                    run_name,
                    result.returncode,
                    result.stderr[-500:] if result.stderr else "no stderr",
                )
                results.append(
                    {"name": run_name, "status": "failed", "returncode": result.returncode}
                )
        except subprocess.TimeoutExpired:
            logger.error("Experiment '%s' timed out", run_name)
            results.append({"name": run_name, "status": "timeout", "returncode": -1})

    success = sum(1 for r in results if r["status"] == "success")
    logger.info(
        "Sweep complete: %d/%d succeeded, %d failed", success, len(results), len(results) - success
    )
    for r in results:
        logger.info("  %s: %s", r["name"], r["status"])
    logger.info("View results in MLflow: mlflow ui --port 5000")


if __name__ == "__main__":
    main()
