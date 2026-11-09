"""Tests for configuration loading and utility modules."""

from __future__ import annotations

from omegaconf import OmegaConf

from indus.memory import MemorySnapshot, build_memory_report
from indus.seed import set_seed


class TestSeed:
    def test_seed_is_deterministic(self) -> None:
        import random

        import numpy as np

        set_seed(42)
        r1 = random.random()
        n1 = np.random.random()

        set_seed(42)
        r2 = random.random()
        n2 = np.random.random()

        assert r1 == r2
        assert n1 == n2


class TestMemory:
    def test_snapshot_to_dict(self) -> None:
        snap = MemorySnapshot(
            gpu_allocated_gb=1.0,
            gpu_reserved_gb=2.0,
            gpu_peak_allocated_gb=1.5,
            system_used_gb=8.0,
            system_total_gb=32.0,
        )
        d = snap.to_dict()
        assert d["gpu_allocated_gb"] == 1.0
        assert d["system_total_gb"] == 32.0

    def test_build_memory_report(self) -> None:
        before = MemorySnapshot(
            gpu_allocated_gb=0.0,
            gpu_reserved_gb=0.0,
            gpu_peak_allocated_gb=0.0,
            system_used_gb=8.0,
            system_total_gb=32.0,
        )
        after = MemorySnapshot(
            gpu_allocated_gb=4.0,
            gpu_reserved_gb=5.0,
            gpu_peak_allocated_gb=4.5,
            system_used_gb=10.0,
            system_total_gb=32.0,
        )
        report = build_memory_report(before, after)
        assert report["delta_gpu_allocated_gb"] == 4.0
        assert report["peak_gpu_gb"] == 4.5


class TestConfigSanity:
    def test_main_config(self) -> None:
        cfg = OmegaConf.load("configs/config.yaml")
        assert "seed" in cfg
        assert "mlflow" in cfg
        assert "output_dir" in cfg

    def test_model_config(self) -> None:
        cfg = OmegaConf.load("configs/model/mistral_7b.yaml")
        assert "model_name_or_path" in cfg
        assert "max_length" in cfg

    def test_training_config(self) -> None:
        cfg = OmegaConf.load("configs/training/qlora_r16.yaml")
        assert "method" in cfg
        assert cfg.method == "qlora"
        assert "quantization" in cfg
        assert "lora" in cfg
        assert cfg.lora.r == 16

    def test_data_config(self) -> None:
        cfg = OmegaConf.load("configs/data/eng_summarization.yaml")
        assert "source_column" in cfg
        assert "target_column" in cfg
        assert "prompt_template" in cfg

    def test_training_configs_have_consistent_schema(self) -> None:
        required_keys = {
            "method",
            "quantization",
            "lora",
            "num_epochs",
            "batch_size",
            "learning_rate",
        }
        for name in ["qlora_r8", "qlora_r16", "qlora_r32"]:
            cfg = OmegaConf.load(f"configs/training/{name}.yaml")
            for key in required_keys:
                assert key in cfg, f"Missing '{key}' in {name}.yaml"
