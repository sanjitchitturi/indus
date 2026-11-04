"""CPU tests for decoder padding and inference argument guards."""

from __future__ import annotations

import pytest


def test_evaluator_left_pads_and_decodes_only_generated_tokens():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from indus.evaluation.evaluator import Evaluator

    class Inputs(dict):
        def to(self, device):
            return self

    class Tokenizer:
        padding_side = "right"
        pad_token_id = 0

        def __call__(self, prompts, **kwargs):
            assert self.padding_side == "left"
            return Inputs(
                input_ids=torch.tensor([[0, 1, 2], [3, 4, 5]]),
                attention_mask=torch.tensor([[0, 1, 1], [1, 1, 1]]),
            )

        def decode(self, tokens, **kwargs):
            return " ".join(str(value) for value in tokens.tolist())

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def generate(self, input_ids, **kwargs):
            return torch.cat([input_ids, torch.tensor([[7], [8]])], dim=1)

    evaluator = Evaluator(Model(), Tokenizer())
    assert evaluator.generate(["short", "a longer prompt"], 2) == ["7", "8"]
    with pytest.raises(ValueError, match="positive"):
        evaluator.generate(["a"], 0)


def test_inference_rejects_empty_benchmark_and_zero_token_budget():
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from indus.inference.engine import InferenceEngine

    model = torch.nn.Linear(1, 1)
    engine = InferenceEngine(model, None)
    with pytest.raises(ValueError, match="at least one"):
        engine.benchmark([])
    with pytest.raises(ValueError, match="positive"):
        engine.generate("prompt", max_new_tokens=0)
