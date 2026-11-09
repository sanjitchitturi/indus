"""Model and tokenizer loading with quantization support."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from indus.exceptions import ModelError
from indus.logging_utils import get_logger
from indus.memory import log_memory

logger = get_logger(__name__)


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    """Convert a string dtype to a torch.dtype."""
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_str not in mapping:
        raise ModelError(f"Unsupported dtype: {dtype_str}. Choose from {list(mapping.keys())}.")
    return mapping[dtype_str]


def build_quantization_config(quant_cfg: DictConfig) -> BitsAndBytesConfig:
    """Build a BitsAndBytesConfig from a Hydra quantization config block.

    Args:
        quant_cfg: Quantization configuration with load_in_4bit, bnb_4bit_quant_type, etc.

    Returns:
        Configured BitsAndBytesConfig instance.
    """
    compute_dtype = _resolve_dtype(quant_cfg.get("bnb_4bit_compute_dtype", "float16"))

    return BitsAndBytesConfig(
        load_in_4bit=quant_cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
    )


def load_tokenizer(model_cfg: DictConfig) -> PreTrainedTokenizerBase:
    """Load and configure the tokenizer.

    Args:
        model_cfg: Model configuration block.

    Returns:
        Configured tokenizer with pad token set.
    """
    tokenizer_name = model_cfg.get("tokenizer_name_or_path") or model_cfg.model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("Set pad_token to eos_token ('%s')", tokenizer.pad_token)

    logger.info("Tokenizer loaded: %s (vocab size: %d)", tokenizer_name, len(tokenizer))
    return tokenizer


def load_base_model(
    model_cfg: DictConfig,
    training_cfg: DictConfig | None = None,
    device: str = "auto",
) -> PreTrainedModel:
    """Load the base causal language model, optionally with quantization.

    Args:
        model_cfg: Model configuration block.
        training_cfg: Training configuration block (contains quantization settings).
        device: Device map strategy. Use 'auto' for automatic placement.

    Returns:
        Loaded model, potentially quantized.

    Raises:
        ModelError: If model loading fails.
    """
    log_memory("pre-model-load")

    model_kwargs: dict[str, Any] = {
        "pretrained_model_name_or_path": model_cfg.model_name_or_path,
        "trust_remote_code": model_cfg.get("trust_remote_code", False),
        "device_map": device,
    }

    torch_dtype = _resolve_dtype(model_cfg.get("torch_dtype", "float16"))
    model_kwargs["torch_dtype"] = torch_dtype

    if training_cfg is not None and training_cfg.get("method") == "qlora":
        quant_config = build_quantization_config(training_cfg.quantization)
        model_kwargs["quantization_config"] = quant_config
        logger.info(
            "Quantization enabled: 4-bit=%s, type=%s, double_quant=%s",
            training_cfg.quantization.load_in_4bit,
            training_cfg.quantization.bnb_4bit_quant_type,
            training_cfg.quantization.bnb_4bit_use_double_quant,
        )

    try:
        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    except Exception as exc:
        raise ModelError(f"Failed to load model '{model_cfg.model_name_or_path}': {exc}") from exc

    log_memory("post-model-load")
    logger.info(
        "Model loaded: %s (dtype: %s, device_map: %s)",
        model_cfg.model_name_or_path,
        torch_dtype,
        device,
    )
    return model
