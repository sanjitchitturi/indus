"""End-to-end evaluation pipeline: generation followed by metric computation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from indus.evaluation.metrics import compute_metrics
from indus.logging_utils import get_logger
from indus.memory import log_memory

logger = get_logger(__name__)


class Evaluator:
    """Generate summaries from a model and compute evaluation metrics.

    Args:
        model: Language model (base or fine-tuned with adapter).
        tokenizer: Corresponding tokenizer.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (0 for greedy decoding).
        top_p: Nucleus sampling threshold.
        bertscore_model: Model identifier for BERTScore computation.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        bertscore_model: str = "microsoft/deberta-xlarge-mnli",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.bertscore_model = bertscore_model
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def generate(self, prompts: list[str], batch_size: int = 4) -> list[str]:
        """Generate summaries for a list of prompts.

        Args:
            prompts: List of formatted prompt strings.
            batch_size: Number of prompts to process in parallel.

        Returns:
            List of generated summary strings.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        # Decoder-only batched generation must left-pad variable-length prompts.
        self.tokenizer.padding_side = "left"
        self.model.eval()
        all_outputs: list[str] = []

        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
            batch_prompts = prompts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=768,
            ).to(self.device)

            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": self.max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
            }

            if self.temperature > 0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = self.temperature
                gen_kwargs["top_p"] = self.top_p
            else:
                gen_kwargs["do_sample"] = False

            output_ids = self.model.generate(**inputs, **gen_kwargs)

            for j, out in enumerate(output_ids):
                prompt_len = inputs["input_ids"][j].shape[0]
                generated = self.tokenizer.decode(
                    out[prompt_len:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                all_outputs.append(generated.strip())

        logger.info("Generated %d summaries", len(all_outputs))
        return all_outputs

    def evaluate(
        self,
        dataset: Any,
        batch_size: int = 4,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run full evaluation: generate summaries and compute metrics.

        Args:
            dataset: SummarizationDataset with prompt and target fields.
            batch_size: Generation batch size.
            output_path: Optional path to save predictions as JSONL.

        Returns:
            Dictionary with metrics and generation metadata.
        """
        log_memory("eval-start")
        start_time = time.time()

        prompts = [dataset[i]["prompt"] for i in range(len(dataset))]
        references = [dataset[i]["target"] for i in range(len(dataset))]
        doc_ids = [dataset[i]["id"] for i in range(len(dataset))]

        predictions = self.generate(prompts, batch_size=batch_size)
        metrics = compute_metrics(
            predictions=predictions,
            references=references,
            bertscore_model=self.bertscore_model,
        )

        elapsed = time.time() - start_time
        snap = log_memory("eval-complete")

        result = {
            "metrics": metrics,
            "num_examples": len(predictions),
            "elapsed_seconds": round(elapsed, 2),
            "gpu_peak_gb": snap.gpu_peak_allocated_gb,
        }

        if output_path is not None:
            self._save_predictions(output_path, doc_ids, prompts, predictions, references)
            result["predictions_path"] = str(output_path)

        return result

    @staticmethod
    def _save_predictions(
        path: str | Path,
        doc_ids: list[str],
        prompts: list[str],
        predictions: list[str],
        references: list[str],
    ) -> None:
        """Save predictions to a JSONL file for qualitative review."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for did, prompt, pred, ref in zip(
                doc_ids, prompts, predictions, references, strict=True
            ):
                record = {
                    "document_id": did,
                    "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
                    "prediction": pred,
                    "reference": ref,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info("Predictions saved to %s", path)
