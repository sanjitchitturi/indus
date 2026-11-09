"""Custom training loop with gradient norm tracking and metric logging."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from indus.data.preprocessing import collate_fn, tokenize_example
from indus.exceptions import TrainingError
from indus.logging_utils import get_logger
from indus.memory import log_memory, reset_peak_memory

logger = get_logger(__name__)


@dataclass
class TrainState:
    """Tracks training progress and accumulated metrics."""

    global_step: int = 0
    epoch: int = 0
    best_val_loss: float = float("inf")
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    gradient_norms: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)


class IndusTrainer:
    """Custom trainer for QLoRA fine-tuning with gradient norm tracking.

    This trainer provides explicit control over the training loop, enabling
    fine-grained logging of gradient norms, adapter weight statistics, and
    per-step memory usage.

    Args:
        model: The model with LoRA adapters applied.
        tokenizer: Tokenizer for collation.
        training_cfg: Training hyperparameter configuration.
        data_cfg: Data configuration for preprocessing.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        mlflow_logger: Optional MLflow logger callback.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        training_cfg: DictConfig,
        data_cfg: DictConfig,
        train_dataset: Any,
        val_dataset: Any,
        mlflow_logger: Any = None,
        checkpoint_dir: str | Path | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.training_cfg = training_cfg
        self.data_cfg = data_cfg
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.mlflow_logger = mlflow_logger
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.state = TrainState()
        self.device = next(model.parameters()).device

        for key in ("batch_size", "gradient_accumulation_steps", "num_epochs", "logging_steps"):
            if self.training_cfg[key] <= 0:
                raise TrainingError(f"{key} must be positive.")
        if not 0 <= self.training_cfg.warmup_ratio < 1:
            raise TrainingError("warmup_ratio must be in [0, 1).")
        self._setup_optimizer()
        self._setup_dataloaders()

    def _setup_optimizer(self) -> None:
        """Initialize optimizer and learning rate scheduler."""
        self.optimizer = torch.optim.AdamW(
            (parameter for parameter in self.model.parameters() if parameter.requires_grad),
            lr=self.training_cfg.learning_rate,
            weight_decay=self.training_cfg.weight_decay,
        )

        batches = math.ceil(len(self.train_dataset) / self.training_cfg.batch_size)
        total_steps = (
            math.ceil(batches / self.training_cfg.gradient_accumulation_steps)
            * self.training_cfg.num_epochs
        )
        warmup_steps = int(total_steps * self.training_cfg.warmup_ratio)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(total_steps - warmup_steps, 1),
        )
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        logger.info(
            "Optimizer: AdamW (lr=%.2e, wd=%.4f), scheduler: cosine, " "total_steps=%d, warmup=%d",
            self.training_cfg.learning_rate,
            self.training_cfg.weight_decay,
            total_steps,
            warmup_steps,
        )

    def _setup_dataloaders(self) -> None:
        """Build training and validation DataLoaders."""
        _collate = partial(
            collate_fn,
            tokenizer=self.tokenizer,
            max_length=(
                self.data_cfg.get("max_source_length", 768)
                + self.data_cfg.get("max_target_length", 256)
            ),
        )

        self._train_examples = self._tokenize_dataset(self.train_dataset)
        self._val_examples = self._tokenize_dataset(self.val_dataset)

        self.train_loader = DataLoader(
            self._train_examples,
            batch_size=self.training_cfg.batch_size,
            shuffle=True,
            collate_fn=_collate,
            num_workers=0,
            pin_memory=False,
        )
        self.val_loader = DataLoader(
            self._val_examples,
            batch_size=self.training_cfg.batch_size,
            shuffle=False,
            collate_fn=_collate,
            num_workers=0,
            pin_memory=False,
        )

    def _tokenize_dataset(self, dataset: Any) -> list[dict[str, list[int]]]:
        """Tokenize all examples in a dataset."""
        examples = []
        for i in range(len(dataset)):
            item = dataset[i]
            tok = tokenize_example(
                prompt=item["prompt"],
                target=item["target"],
                tokenizer=self.tokenizer,
                max_source_length=self.data_cfg.get("max_source_length", 768),
                max_target_length=self.data_cfg.get("max_target_length", 256),
            )
            examples.append(tok)
        return examples

    def _compute_gradient_norm(self) -> float:
        """Compute the L2 norm of all trainable gradients."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.requires_grad and p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm**0.5

    def _warmup_lr(self) -> None:
        """Apply linear warmup to the learning rate."""
        if self.state.global_step < self.warmup_steps and self.warmup_steps > 0:
            scale = (self.state.global_step + 1) / self.warmup_steps
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.training_cfg.learning_rate * scale

    def train(self) -> TrainState:
        """Run the full training loop.

        Returns:
            Final TrainState with all accumulated metrics.

        Raises:
            TrainingError: If an unrecoverable error occurs during training.
        """
        logger.info("Starting training for %d epochs", self.training_cfg.num_epochs)
        reset_peak_memory()
        start_time = time.time()
        self.model.train()

        try:
            for epoch in range(self.training_cfg.num_epochs):
                self.state.epoch = epoch
                epoch_loss = self._train_epoch(epoch)
                val_loss = self._validate()

                logger.info(
                    "Epoch %d/%d -- train_loss=%.4f, val_loss=%.4f",
                    epoch + 1,
                    self.training_cfg.num_epochs,
                    epoch_loss,
                    val_loss,
                )

                if val_loss < self.state.best_val_loss:
                    self.state.best_val_loss = val_loss
                    self._save_best_checkpoint()
                    logger.info("New best validation loss: %.4f", val_loss)

        except Exception as exc:
            raise TrainingError(f"Training failed at step {self.state.global_step}: {exc}") from exc

        elapsed = time.time() - start_time
        snap = log_memory("training-complete")
        logger.info(
            "Training complete in %.1f seconds. Peak GPU memory: %.2f GB",
            elapsed,
            snap.gpu_peak_allocated_gb,
        )
        return self.state

    def _save_best_checkpoint(self) -> None:
        """Persist the best adapter and optimizer state after validation."""
        if self.checkpoint_dir is None:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir = self.checkpoint_dir / "best_adapter"
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(str(adapter_dir))
            self.tokenizer.save_pretrained(str(adapter_dir))
        else:
            adapter_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), adapter_dir / "model.pt")
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "state": self.state,
            },
            self.checkpoint_dir / "best_training_state.pt",
        )
        (self.checkpoint_dir / "best_metrics.json").write_text(
            json.dumps(
                {"best_val_loss": self.state.best_val_loss, "global_step": self.state.global_step},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def _train_epoch(self, epoch: int) -> float:
        """Run one training epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        self.optimizer.zero_grad()

        accumulation = self.training_cfg.gradient_accumulation_steps
        window_loss = 0.0
        for step, batch in enumerate(self.train_loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            # Normalize the final partial window by its actual microbatch count.
            window_start = (step // accumulation) * accumulation
            window_size = min(accumulation, len(self.train_loader) - window_start)
            loss = outputs.loss / window_size
            loss.backward()
            raw_loss = outputs.loss.item()
            epoch_loss += raw_loss
            window_loss += raw_loss
            num_batches += 1

            if (step + 1) % accumulation == 0 or step + 1 == len(self.train_loader):
                grad_norm = self._compute_gradient_norm()
                self.state.gradient_norms.append(grad_norm)
                if self.training_cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_cfg.max_grad_norm,
                    )

                self._warmup_lr()
                self.optimizer.step()
                if self.state.global_step >= self.warmup_steps:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                self.state.global_step += 1

                current_lr = self.optimizer.param_groups[0]["lr"]
                self.state.learning_rates.append(current_lr)

                step_loss = window_loss / window_size
                window_loss = 0.0
                self.state.train_losses.append(step_loss)

                if self.state.global_step % self.training_cfg.logging_steps == 0:
                    logger.info(
                        "Step %d -- loss=%.4f, grad_norm=%.4f, lr=%.2e",
                        self.state.global_step,
                        step_loss,
                        grad_norm,
                        current_lr,
                    )
                    if self.mlflow_logger is not None:
                        self.mlflow_logger.log_step(
                            step=self.state.global_step,
                            loss=step_loss,
                            grad_norm=grad_norm,
                            lr=current_lr,
                        )

        return epoch_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(self) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            total_loss += outputs.loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        self.state.val_losses.append(avg_loss)

        if self.mlflow_logger is not None:
            self.mlflow_logger.log_val(step=self.state.global_step, val_loss=avg_loss)

        self.model.train()
        return avg_loss
