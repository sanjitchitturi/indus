"""Tests for text preprocessing and visualization."""

from __future__ import annotations

from indus.data.preprocessing import build_prompt


class TestBuildPrompt:
    def test_default_template(self) -> None:
        result = build_prompt("This is the document text.")
        assert "Summarize" in result
        assert "This is the document text." in result
        assert result.endswith("Summary:")

    def test_custom_template(self) -> None:
        template = "TL;DR: {source}\n\nAnswer:"
        result = build_prompt("Some content.", template=template)
        assert result == "TL;DR: Some content.\n\nAnswer:"

    def test_empty_source(self) -> None:
        result = build_prompt("")
        assert "Summary:" in result


class TestVisualization:
    def test_plot_training_loss(self, tmp_path) -> None:
        from indus.visualization.plots import plot_training_loss

        losses = [2.0, 1.8, 1.5, 1.3, 1.1, 1.0, 0.9, 0.85, 0.8, 0.78]
        path = plot_training_loss(losses, output_path=tmp_path / "loss.png")
        assert path.exists()

    def test_plot_gradient_norms(self, tmp_path) -> None:
        from indus.visualization.plots import plot_gradient_norms

        norms = [5.0, 4.5, 3.8, 3.2, 2.8, 2.5, 2.3, 2.1, 2.0, 1.9]
        path = plot_gradient_norms(norms, output_path=tmp_path / "gnorm.png")
        assert path.exists()

    def test_plot_eval_comparison(self, tmp_path) -> None:
        from indus.visualization.plots import plot_eval_comparison

        baseline = {"rougeL_fmeasure": 0.29, "bleu": 0.11, "bertscore_f1": 0.81}
        finetuned = {"rougeL_fmeasure": 0.41, "bleu": 0.18, "bertscore_f1": 0.87}
        path = plot_eval_comparison(baseline, finetuned, output_path=tmp_path / "comp.png")
        assert path.exists()

    def test_plot_adapter_convergence(self, tmp_path) -> None:
        from indus.visualization.plots import plot_adapter_convergence

        losses = [2.0, 1.5, 1.2, 1.0, 0.9, 0.85]
        norms = [5.0, 3.5, 2.5, 2.0, 1.8, 1.7]
        path = plot_adapter_convergence(losses, norms, output_path=tmp_path / "conv.png")
        assert path.exists()
