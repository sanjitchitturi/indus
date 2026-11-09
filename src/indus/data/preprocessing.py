"""Text preprocessing and tokenization for causal LM summarization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

from indus.logging_utils import get_logger

logger = get_logger(__name__)


def build_prompt(
    source: str,
    template: str = "Summarize the following engineering document:\n\n{source}\n\nSummary:",
) -> str:
    """Format a source document into a summarization prompt.

    Args:
        source: Raw document text.
        template: Prompt template with a ``{source}`` placeholder.

    Returns:
        Formatted prompt string.
    """
    return template.format(source=source)


def tokenize_example(
    prompt: str,
    target: str,
    tokenizer: PreTrainedTokenizerBase,
    max_source_length: int = 768,
    max_target_length: int = 256,
) -> dict[str, Any]:
    """Tokenize a single prompt-target pair for causal LM training.

    The input_ids are the concatenation of the prompt tokens and target tokens.
    Labels are set to -100 for the prompt portion so that the loss is computed
    only on the target tokens.

    Args:
        prompt: The formatted prompt string.
        target: The reference summary.
        tokenizer: Hugging Face tokenizer instance.
        max_source_length: Maximum token length for the prompt.
        max_target_length: Maximum token length for the target.

    Returns:
        Dictionary with input_ids, attention_mask, and labels tensors.
    """
    prompt_ids = tokenizer(
        prompt,
        truncation=True,
        max_length=max_source_length,
        add_special_tokens=True,
    )["input_ids"]

    if max_source_length <= 0 or max_target_length <= 0:
        raise ValueError("Token limits must be positive.")
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id.")
    # Reserve the last target position for EOS even when the target is truncated.
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    target_ids = target_ids[: max_target_length - 1] + [tokenizer.eos_token_id]

    input_ids = prompt_ids + target_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + target_ids

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def collate_fn(
    batch: list[dict[str, list[int]]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 1024,
) -> dict[str, Any]:
    """Collate tokenized examples into a padded batch.

    Args:
        batch: List of tokenized examples from tokenize_example.
        tokenizer: Tokenizer for pad token id.
        max_length: Maximum sequence length.

    Returns:
        Padded batch dictionary with PyTorch tensors.
    """
    import torch

    if not batch or max_length <= 0:
        raise ValueError("Batch must be non-empty and max_length positive.")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    max_len = min(max(len(ex["input_ids"]) for ex in batch), max_length)

    input_ids = []
    attention_mask = []
    labels = []

    for ex in batch:
        pad_len = max_len - len(ex["input_ids"])

        if pad_len < 0:
            input_ids.append(ex["input_ids"][-max_len:])
            attention_mask.append(ex["attention_mask"][-max_len:])
            labels.append(ex["labels"][-max_len:])
        else:
            input_ids.append(ex["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(ex["attention_mask"] + [0] * pad_len)
            labels.append(ex["labels"] + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
