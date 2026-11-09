"""Tests for evaluation metric computation."""

from __future__ import annotations

import pytest

from indus.evaluation.metrics import (
    compute_bleu,
    compute_metrics,
    compute_numeric_fact_consistency,
    compute_rouge,
)
from indus.exceptions import EvaluationError


class TestComputeRouge:
    def test_basic_scores(self, predictions: list[str], references: list[str]) -> None:
        result = compute_rouge(predictions, references)
        assert "rougeL_fmeasure" in result
        assert "rougeL_precision" in result
        assert "rougeL_recall" in result
        assert 0.0 <= result["rougeL_fmeasure"] <= 1.0

    def test_identical_inputs(self) -> None:
        texts = ["This is a test sentence."]
        result = compute_rouge(texts, texts)
        assert result["rougeL_fmeasure"] == 1.0

    def test_empty_prediction(self) -> None:
        result = compute_rouge([""], ["Some reference text."])
        assert result["rougeL_fmeasure"] == 0.0


class TestComputeBleu:
    def test_basic_scores(self, predictions: list[str], references: list[str]) -> None:
        result = compute_bleu(predictions, references)
        assert "bleu" in result
        assert "bleu_bp" in result
        assert 0.0 <= result["bleu"] <= 1.0

    def test_identical_inputs(self) -> None:
        texts = ["This is a test sentence for BLEU scoring."]
        result = compute_bleu(texts, texts)
        assert result["bleu"] == 1.0

    def test_completely_different(self) -> None:
        result = compute_bleu(
            ["alpha beta gamma delta"],
            ["one two three four five six seven"],
        )
        assert result["bleu"] < 0.1


class TestComputeMetrics:
    def test_mismatched_lengths(self) -> None:
        with pytest.raises(EvaluationError, match="does not match"):
            compute_metrics(["a", "b"], ["a"])

    def test_empty_inputs(self) -> None:
        with pytest.raises(EvaluationError, match="empty"):
            compute_metrics([], [])

    def test_numeric_fact_consistency(self) -> None:
        result = compute_numeric_fact_consistency(
            ["Load was 50 psf and wind speed was 115 mph."],
            ["Load was 50 psf and wind speed was 100 mph."],
        )
        assert result["numeric_fact_precision"] == pytest.approx(0.5)
        assert result["numeric_fact_recall"] == pytest.approx(0.5)
        assert result["numeric_fact_f1"] == pytest.approx(0.5)
