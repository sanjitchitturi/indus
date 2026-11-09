"""Evaluation script for fine-tuned model on held-out test data.

Usage:
    python scripts/evaluate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indus.data import load_dataset_splits
from indus.evaluation import Evaluator
from indus.logging_utils import get_logger, setup_logging
from indus.model import load_base_model, load_tokenizer
from indus.seed import set_seed
from indus.tracking import MLflowLogger


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run evaluation on the test split and log metrics to MLflow."""
    setup_logging(level=cfg.logging.level)
    logger = get_logger("evaluate")
    logger.info("Evaluation config:\n%s", OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    mlflow_logger = MLflowLogger(
        cfg=cfg,
        tracking_uri=cfg.mlflow.tracking_uri,
        experiment_name=cfg.mlflow.experiment_name,
        run_name=f"eval-{cfg.mlflow.get('run_name', 'default')}",
    )
    mlflow_logger.start_run()

    try:
        data_kwargs = {
            "source_column": cfg.data.source_column,
            "target_column": cfg.data.target_column,
            "id_column": cfg.data.id_column,
            "prompt_template": cfg.data.prompt_template,
        }
        splits = load_dataset_splits(
            train_path=cfg.data.train_path,
            val_path=cfg.data.get("val_path"),
            test_path=cfg.data.get("test_path"),
            val_fraction=cfg.data.val_fraction,
            test_fraction=cfg.data.test_fraction,
            seed=cfg.seed,
            **data_kwargs,
        )

        tokenizer = load_tokenizer(cfg.model)
        device = cfg.get("device", "auto")
        model = load_base_model(cfg.model, training_cfg=cfg.training, device=device)

        adapter_path = Path(cfg.output_dir) / "adapter"
        if adapter_path.exists():
            from peft import PeftModel

            logger.info("Loading adapter from %s", adapter_path)
            model = PeftModel.from_pretrained(model, str(adapter_path))

        evaluator = Evaluator(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=cfg.data.max_target_length,
        )

        output_path = Path(cfg.output_dir) / "predictions.jsonl"
        result = evaluator.evaluate(
            dataset=splits["test"],
            batch_size=cfg.training.batch_size,
            output_path=output_path,
        )

        logger.info("Evaluation results: %s", json.dumps(result["metrics"], indent=2))
        mlflow_logger.log_eval_metrics(result["metrics"], prefix="eval")

        metrics_path = Path(cfg.output_dir) / "eval_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(result, f, indent=2)

        mlflow_logger.log_artifact(metrics_path, artifact_subdir="metrics")
        if output_path.exists():
            mlflow_logger.log_artifact(output_path, artifact_subdir="predictions")

    except Exception:
        logger.exception("Evaluation failed")
        raise
    finally:
        mlflow_logger.end_run()


if __name__ == "__main__":
    main()
