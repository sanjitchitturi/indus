"""Regression tests for data integrity, target masking, and accumulation boundaries."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from indus.data.dataset import SummarizationDataset, load_dataset_splits
from indus.data.preprocessing import tokenize_example
from indus.exceptions import DataError


@pytest.mark.parametrize(
    "bad",
    [{}, {"source_text": "", "target_summary": "x"}, {"source_text": 3, "target_summary": "x"}, []],
)
def test_validates_every_record(sample_records, bad):
    with pytest.raises(DataError):
        SummarizationDataset(sample_records + [bad])


@pytest.mark.parametrize("fraction", [-0.1, 0, 1, float("nan")])
def test_invalid_split_fraction(sample_jsonl_path, fraction):
    with pytest.raises(DataError, match="between"):
        load_dataset_splits(sample_jsonl_path, val_fraction=fraction)


def test_tiny_split_has_actionable_error(sample_jsonl_path):
    with pytest.raises(DataError, match="too small"):
        load_dataset_splits(sample_jsonl_path)


def test_truncated_target_keeps_eos_and_masks_prompt():
    class Tokenizer:
        eos_token_id = 99

        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2, 3, 4][: kwargs.get("max_length", 4)]}

    result = tokenize_example("prompt", "target", Tokenizer(), 2, 2)
    assert result["input_ids"] == [1, 2, 1, 99]
    assert result["labels"] == [-100, -100, 1, 99]


def test_partial_accumulation_updates_and_matches_reference():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from omegaconf import OmegaConf

    from indus.training.trainer import IndusTrainer

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, x):
            return SimpleNamespace(loss=(self.weight - x.float()).square().mean())

    trainer = IndusTrainer.__new__(IndusTrainer)
    trainer.model = TinyModel()
    trainer.device = torch.device("cpu")
    trainer.training_cfg = OmegaConf.create(
        {
            "gradient_accumulation_steps": 2,
            "max_grad_norm": 0,
            "logging_steps": 1,
            "learning_rate": 0.1,
        }
    )
    trainer.train_loader = [{"x": torch.tensor([v])} for v in (1.0, 3.0, 5.0)]
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1)
    trainer.warmup_steps = 0
    from indus.training.trainer import TrainState

    trainer.state = TrainState()
    trainer.mlflow_logger = None
    loss = trainer._train_epoch(0)
    # First update uses mean targets (1, 3): w=0.4; final singleton updates to 1.32.
    assert trainer.model.weight.item() == pytest.approx(1.32)
    assert trainer.state.global_step == 2
    assert loss == pytest.approx((1 + 9 + 4.6**2) / 3)
    assert trainer.state.gradient_norms == pytest.approx([4, 9.2])


def test_optimizer_only_contains_trainable_parameters():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from omegaconf import OmegaConf

    from indus.training.trainer import IndusTrainer

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.frozen = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
            self.trainable = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(loss=(self.trainable - input_ids.float()).square().mean())

    cfg = OmegaConf.create(
        {
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "num_epochs": 1,
            "logging_steps": 1,
            "warmup_ratio": 0,
            "learning_rate": 0.1,
            "weight_decay": 0,
            "max_grad_norm": 0,
        }
    )
    trainer = IndusTrainer.__new__(IndusTrainer)
    trainer.model = TinyModel()
    trainer.training_cfg = cfg
    trainer.train_dataset = [0]
    trainer._setup_optimizer()
    params = list(trainer.optimizer.param_groups[0]["params"])
    assert params == [trainer.model.trainable]
