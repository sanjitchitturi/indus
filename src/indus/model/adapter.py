"""LoRA adapter configuration and application."""

from __future__ import annotations

from omegaconf import DictConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import PreTrainedModel

from indus.logging_utils import get_logger
from indus.memory import log_memory

logger = get_logger(__name__)

_TASK_TYPE_MAP = {
    "CAUSAL_LM": TaskType.CAUSAL_LM,
    "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
}


def build_lora_config(lora_cfg: DictConfig) -> LoraConfig:
    """Build a PEFT LoraConfig from the Hydra configuration.

    Args:
        lora_cfg: LoRA configuration block with r, lora_alpha, etc.

    Returns:
        Configured LoraConfig instance.
    """
    task_type = _TASK_TYPE_MAP.get(
        lora_cfg.get("task_type", "CAUSAL_LM"),
        TaskType.CAUSAL_LM,
    )
    target_modules = list(lora_cfg.get("target_modules", ["q_proj", "v_proj"]))

    config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=target_modules,
        bias=lora_cfg.get("bias", "none"),
        task_type=task_type,
    )

    logger.info(
        "LoRA config: r=%d, alpha=%d, dropout=%.2f, targets=%s",
        config.r,
        config.lora_alpha,
        config.lora_dropout,
        target_modules,
    )
    return config


def apply_lora_adapter(
    model: PreTrainedModel,
    lora_cfg: DictConfig,
    is_quantized: bool = True,
) -> PreTrainedModel:
    """Prepare a model for LoRA fine-tuning.

    If the model is quantized, it is first prepared for k-bit training
    (gradient checkpointing, input casting). Then the LoRA adapter is applied.

    Args:
        model: Base language model.
        lora_cfg: LoRA configuration from Hydra.
        is_quantized: Whether the model was loaded with quantization.

    Returns:
        Model with LoRA adapter layers applied.
    """
    if is_quantized:
        model = prepare_model_for_kbit_training(model)
        logger.info("Model prepared for k-bit training")

    lora_config = build_lora_config(lora_cfg)
    model = get_peft_model(model, lora_config)

    log_memory("post-adapter-apply")

    trainable, total = count_parameters(model)
    logger.info(
        "Adapter applied: trainable=%s (%.2f%% of %s total)",
        _format_params(trainable),
        100.0 * trainable / total if total > 0 else 0.0,
        _format_params(total),
    )
    return model


def count_parameters(model: PreTrainedModel) -> tuple[int, int]:
    """Count trainable and total parameters.

    Args:
        model: PyTorch model (may have PEFT adapters).

    Returns:
        Tuple of (trainable_parameters, total_parameters).
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _format_params(n: int) -> str:
    """Format parameter count as a human-readable string."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)
