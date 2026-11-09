"""Verify deterministic baseline evidence and safe report rendering."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from indus.baseline import rouge_l, sentences, summarize
from indus.demo import run_demo


def test_sentence_boundaries_keep_decimal_measurements():
    assert sentences("Cracks measured 0.2 mm. Repairs scheduled tomorrow.") == [
        "Cracks measured 0.2 mm.",
        "Repairs scheduled tomorrow.",
    ]


def test_frequency_ranking_is_deterministic_and_extractive():
    text = "Weather is mild. Pump pressure is stable. Pump pressure was stable."
    assert summarize(text, "frequency", 1) == "Pump pressure is stable."
    assert summarize(text, "lead", 1) == "Weather is mild."


def test_rouge_l_known_subsequence():
    assert rouge_l("a b c", "a c") == pytest.approx(0.8)
    assert rouge_l("", "a") == 0
    assert rouge_l("Pump OK", "pump ok") == 1


def test_demo_artifacts_match_predictions_and_escape_html(sample_jsonl_path, tmp_path):
    records = [json.loads(line) for line in sample_jsonl_path.read_text().splitlines()]
    records[0]["source_text"] += " <script>alert(1)</script>"
    sample_jsonl_path.write_text("\n".join(json.dumps(row) for row in records))
    output = tmp_path / "report"
    result = run_demo(sample_jsonl_path, output)
    predictions = [
        json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()
    ]
    assert result["record_count"] == len(predictions) == 3
    assert len(result["dataset_sha256"]) == 64
    assert result["methods"]["lead"]["rouge_l_f1"] == pytest.approx(
        sum(rouge_l(row["lead"]["summary"], row["target"]) for row in predictions) / 3, abs=1e-6
    )
    report = (output / "report.html").read_text()
    assert "<script>" not in report
    assert "&lt;script&gt;" in report
    assert "no fine-tuned" in result["experiment"]


def test_cli_rejects_invalid_input(sample_jsonl_path, tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "indus",
            "demo",
            "--data",
            str(sample_jsonl_path),
            "--output",
            str(tmp_path / "out"),
            "--sentences",
            "0",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "positive" in result.stderr
