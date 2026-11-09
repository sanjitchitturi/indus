"""Interactive inference script with memory profiling.

Usage:
    python scripts/infer.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indus.inference import InferenceEngine
from indus.logging_utils import get_logger, setup_logging
from indus.memory import log_memory, reset_peak_memory
from indus.seed import set_seed

DEMO_PROMPTS = [
    (
        "Summarize the following engineering document:\n\n"
        "The bridge inspection report for Structure BR-4412 identified moderate "
        "corrosion on the steel girder bearings at Piers 3 and 5. Section loss was "
        "measured at approximately 15 percent of the original flange thickness. The "
        "concrete deck showed minor map cracking with no active reinforcement corrosion "
        "detected by half-cell potential survey. Load rating analysis indicates the "
        "bridge can carry HL-93 loading at an inventory rating factor of 1.12 and an "
        "operating rating factor of 1.45.\n\nSummary:"
    ),
    (
        "Summarize the following engineering document:\n\n"
        "The commissioning report for the solar photovoltaic array at Facility 9 "
        "documents performance verification of a 2.4 MW ground-mounted system using "
        "bifacial monocrystalline modules. Array DC capacity measured 2,412 kW under "
        "standard test conditions. The AC conversion efficiency through the string "
        "inverters averaged 97.3 percent. First-year energy production is projected at "
        "4,100 MWh based on TMY3 solar resource data.\n\nSummary:"
    ),
]


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run inference on demo prompts with profiling."""
    setup_logging(level=cfg.logging.level)
    logger = get_logger("infer")
    set_seed(cfg.seed)
    reset_peak_memory()

    adapter_path = Path(cfg.output_dir) / "adapter"
    has_adapter = adapter_path.exists()

    logger.info("Loading inference engine...")
    engine = InferenceEngine.from_pretrained(
        model_name_or_path=cfg.model.model_name_or_path,
        adapter_path=str(adapter_path) if has_adapter else None,
        load_in_4bit=True,
        max_new_tokens=cfg.data.max_target_length,
    )
    log_memory("model-loaded")

    logger.info("Running inference on %d demo prompts...", len(DEMO_PROMPTS))
    for i, prompt in enumerate(DEMO_PROMPTS):
        result = engine.generate(prompt)
        logger.info(
            "Prompt %d: %d tokens in %.2fs (%.1f tok/s)",
            i + 1,
            result.tokens_generated,
            result.latency_seconds,
            result.tokens_per_second,
        )
        print(f"\n{'=' * 60}")
        print(f"Prompt {i + 1} (truncated): {prompt[:120]}...")
        print(f"Generated summary: {result.text}")
        print(f"Tokens: {result.tokens_generated}, Latency: {result.latency_seconds}s")
        print(f"Peak GPU: {result.memory_snapshot.gpu_peak_allocated_gb} GB")

    logger.info("Running benchmark...")
    bench = engine.benchmark(DEMO_PROMPTS)
    print(f"\n{'=' * 60}")
    print("Benchmark Results:")
    print(json.dumps(bench, indent=2))

    bench_path = Path(cfg.output_dir) / "inference_benchmark.json"
    bench_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bench_path, "w") as f:
        json.dump(bench, f, indent=2)
    logger.info("Benchmark saved to %s", bench_path)


if __name__ == "__main__":
    main()
