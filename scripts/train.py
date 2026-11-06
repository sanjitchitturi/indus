"""Main training script for QLoRA fine-tuning.

Usage:
    python scripts/train.py
    python scripts/train.py training=qlora_r8
    python scripts/train.py training.learning_rate=5e-4 seed=123
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indus.data import load_dataset_splits, write_split_manifest
from indus.logging_utils import get_logger, setup_logging
from indus.memory import build_memory_report, log_memory, reset_peak_memory
from indus.model import apply_lora_adapter, count_parameters, load_base_model, load_tokenizer
from indus.seed import set_seed
from indus.tracking import MLflowLogger
from indus.training import IndusTrainer
from indus.visualization import generate_all_plots


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run QLoRA fine-tuning with the given Hydra configuration."""
    setup_logging(level=cfg.logging.level, log_file=cfg.logging.get("file"))
    logger = get_logger("train")
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    mlflow_logger = MLflowLogger(
        cfg=cfg,
        tracking_uri=cfg.mlflow.tracking_uri,
        experiment_name=cfg.mlflow.experiment_name,
        run_name=cfg.mlflow.get("run_name"),
    )
    mlflow_logger.start_run()

    try:
        logger.info("Loading dataset...")
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
        write_split_manifest(
            splits,
            Path(cfg.output_dir) / "split_manifest.json",
            seed=cfg.seed,
            source_paths={
                "train": cfg.data.train_path,
                "val": cfg.data.get("val_path"),
                "test": cfg.data.get("test_path"),
            },
        )

        logger.info("Loading tokenizer...")
        tokenizer = load_tokenizer(cfg.model)

        logger.info("Loading base model with quantization...")
        reset_peak_memory()
        mem_before = log_memory("before-model-load")

        device = cfg.get("device", "auto")
        model = load_base_model(cfg.model, training_cfg=cfg.training, device=device)

        logger.info("Applying LoRA adapter...")
        is_quantized = cfg.training.get("method") == "qlora"
        model = apply_lora_adapter(model, cfg.training.lora, is_quantized=is_quantized)

        mem_after = log_memory("after-adapter")
        trainable, total = count_parameters(model)
        mlflow_logger.log_model_info(trainable, total, mem_after.gpu_peak_allocated_gb)
        mlflow_logger.log_memory_report(build_memory_report(mem_before, mem_after))

        logger.info("Starting training...")
        trainer = IndusTrainer(
            model=model,
            tokenizer=tokenizer,
            training_cfg=cfg.training,
            data_cfg=cfg.data,
            train_dataset=splits["train"],
            val_dataset=splits["val"],
            mlflow_logger=mlflow_logger,
            checkpoint_dir=Path(cfg.output_dir) / "checkpoints",
        )
        train_state = trainer.train()

        adapter_dir = Path(cfg.output_dir) / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        best_adapter = Path(cfg.output_dir) / "checkpoints" / "best_adapter"
        if best_adapter.exists():
            shutil.rmtree(adapter_dir)
            shutil.copytree(best_adapter, adapter_dir)
            logger.info("Best validation adapter copied to %s", adapter_dir)
        else:
            model.save_pretrained(str(adapter_dir))
            tokenizer.save_pretrained(str(adapter_dir))
        logger.info("Adapter saved to %s", adapter_dir)

        logger.info("Generating analysis plots...")
        figure_dir = Path(cfg.output_dir) / "figures"
        plot_paths = generate_all_plots(train_state, output_dir=figure_dir)

        for plot_path in plot_paths:
            mlflow_logger.log_artifact(plot_path, artifact_subdir="figures")
        mlflow_logger.log_artifact(adapter_dir, artifact_subdir="adapter")
        logger.info("Training pipeline complete.")

    except Exception:
        logger.exception("Training failed")
        raise
    finally:
        mlflow_logger.end_run()


if __name__ == "__main__":
    main()
