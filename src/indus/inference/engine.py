"""Inference engine with memory profiling and optimization reporting."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from indus.logging_utils import get_logger
from indus.memory import MemorySnapshot, log_memory, reset_peak_memory
from indus.model.loader import build_quantization_config

logger = get_logger(__name__)


@dataclass
class InferenceResult:
    """Result of a single inference call with profiling data."""

    text: str
    tokens_generated: int
    latency_seconds: float
    tokens_per_second: float
    memory_snapshot: MemorySnapshot


class InferenceEngine:
    """Optimized inference engine with adapter merging and benchmarking.

    This engine supports loading a base model with optional quantization,
    merging LoRA adapters, and running inference with detailed profiling.

    Args:
        model: Loaded language model.
        tokenizer: Corresponding tokenizer.
        max_new_tokens: Default maximum generation length.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        max_new_tokens: int = 256,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.device = next(model.parameters()).device

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        adapter_path: str | None = None,
        load_in_4bit: bool = True,
        max_new_tokens: int = 256,
    ) -> InferenceEngine:
        """Load model and optional adapter for inference.

        Args:
            model_name_or_path: Base model identifier or path.
            adapter_path: Optional path to a saved LoRA adapter.
            load_in_4bit: Whether to use 4-bit quantization.
            max_new_tokens: Default generation length.

        Returns:
            Configured InferenceEngine instance.
        """
        from omegaconf import OmegaConf

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }

        if load_in_4bit:
            quant_cfg = OmegaConf.create(
                {
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": "float16",
                    "bnb_4bit_use_double_quant": True,
                }
            )
            model_kwargs["quantization_config"] = build_quantization_config(quant_cfg)

        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

        if adapter_path is not None:
            logger.info("Loading LoRA adapter from %s", adapter_path)
            model = PeftModel.from_pretrained(model, adapter_path)

        model.eval()
        return cls(model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> InferenceResult:
        """Generate text with profiling.

        Args:
            prompt: Input prompt string.
            max_new_tokens: Override default max generation length.
            temperature: Sampling temperature (0 for greedy).
            top_p: Nucleus sampling threshold.

        Returns:
            InferenceResult with generated text and profiling data.
        """
        max_tokens = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        if max_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")
        self.model.eval()
        reset_peak_memory()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        else:
            gen_kwargs["do_sample"] = False

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        output_ids = self.model.generate(**inputs, **gen_kwargs)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start

        generated_ids = output_ids[0][prompt_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        n_tokens = len(generated_ids)
        snap = log_memory("inference")

        return InferenceResult(
            text=text,
            tokens_generated=n_tokens,
            latency_seconds=round(elapsed, 4),
            tokens_per_second=round(n_tokens / elapsed, 2) if elapsed > 0 else 0.0,
            memory_snapshot=snap,
        )

    def benchmark(
        self,
        prompts: list[str],
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Run inference on multiple prompts and report aggregate statistics.

        Args:
            prompts: List of prompt strings.
            max_new_tokens: Override max generation length.

        Returns:
            Aggregate benchmark results with latency and throughput stats.
        """
        if not prompts:
            raise ValueError("Provide at least one prompt to benchmark.")
        results = []
        for prompt in prompts:
            result = self.generate(prompt, max_new_tokens=max_new_tokens)
            results.append(result)

        latencies = [r.latency_seconds for r in results]
        throughputs = [r.tokens_per_second for r in results]
        total_tokens = sum(r.tokens_generated for r in results)

        return {
            "num_prompts": len(prompts),
            "total_tokens_generated": total_tokens,
            "mean_latency_s": round(sum(latencies) / len(latencies), 4),
            "p50_latency_s": round(sorted(latencies)[len(latencies) // 2], 4),
            "p95_latency_s": round(sorted(latencies)[int(len(latencies) * 0.95)], 4),
            "mean_tokens_per_second": round(sum(throughputs) / len(throughputs), 2),
            "peak_gpu_gb": max(r.memory_snapshot.gpu_peak_allocated_gb for r in results),
        }
