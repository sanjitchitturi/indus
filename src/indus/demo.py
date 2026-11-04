"""Reproducible CPU experiment with provenance and a standalone HTML report."""

from __future__ import annotations

import hashlib
import html
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

from indus import __version__
from indus.baseline import rouge_l, summarize
from indus.data.dataset import SummarizationDataset, load_records


def run_demo(data: Path, output: Path, sentence_count: int = 2) -> dict[str, Any]:
    records = load_records(data)
    dataset = SummarizationDataset(records)
    examples = []
    measurements: dict[str, list[float]] = {"lead": [], "frequency": []}
    scores: dict[str, list[float]] = {"lead": [], "frequency": []}
    for index in range(len(dataset)):
        record = dataset[index]
        item: dict[str, Any] = dict(record)
        for method in measurements:
            start = time.perf_counter()
            prediction = summarize(record["source"], method, sentence_count)
            measurements[method].append((time.perf_counter() - start) * 1000)
            score = rouge_l(prediction, record["target"])
            scores[method].append(score)
            item[method] = {"summary": prediction, "rouge_l_f1": round(score, 6)}
        examples.append(item)
    result = {
        "schema_version": 1,
        "indus_version": __version__,
        "experiment": "CPU extractive baselines — no fine-tuned model",
        "dataset_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "record_count": len(dataset),
        "sentence_count": sentence_count,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "metric": "macro token ROUGE-L F1; lowercase Unicode words; no stemming; scale 0–1",
        "limitations": "Synthetic template-based data; repeated topics and references; not evidence of LLM quality or generalization. Timing excludes loading and metrics; one pass, no warmup.",
        "methods": {
            method: {
                "rouge_l_f1": round(statistics.mean(scores[method]), 6),
                "mean_latency_ms": round(statistics.mean(measurements[method]), 6),
                "p95_latency_ms": round(
                    sorted(measurements[method])[max(0, math_ceil_95(len(dataset)) - 1)], 6
                ),
            }
            for method in measurements
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in examples), encoding="utf-8"
    )
    (output / "report.html").write_text(render_report(result, examples), encoding="utf-8")
    return result


def math_ceil_95(n: int) -> int:
    return (95 * n + 99) // 100


def render_report(result: dict[str, Any], examples: list[dict[str, Any]]) -> str:
    esc = html.escape
    cards = "".join(
        f'<article><span>{esc(method.upper())} / ROUGE-L F1</span><strong>{values["rouge_l_f1"]:.3f}</strong><p>{values["mean_latency_ms"]:.3f} ms / document</p></article>'
        for method, values in result["methods"].items()
    )
    rows = "".join(
        f'<details><summary>{esc(str(row["id"]))} <span>Inspect source & summaries ↗</span></summary><h3>Source document</h3><p>{esc(row["source"])}</p><h3>Reference summary</h3><p>{esc(row["target"])}</p>'
        + "".join(
            f'<h3>{esc(method.title())} · F1 {row[method]["rouge_l_f1"]:.3f}</h3><p>{esc(row[method]["summary"])}</p>'
            for method in result["methods"]
        )
        + "</details>"
        for row in examples
    )
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Indus | Experiment report</title>
<style>
:root{{color-scheme:dark;font-family:system-ui,sans-serif;background:#101b1a;color:#e9eee7}}*{{box-sizing:border-box}}body{{max-width:1120px;margin:auto;padding:40px 24px}}header{{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #344642;padding-bottom:24px}}header b{{letter-spacing:.22em;font-size:24px}}.tag{{color:#bddb9a;border:1px solid #52745b;border-radius:30px;padding:7px 12px;font-size:12px}}h1{{font-size:clamp(36px,6vw,70px);line-height:1.05;font-weight:500;letter-spacing:-.05em;max-width:850px;margin:54px 0 24px}}.intro{{max-width:720px;font-size:18px;color:#b6c7bf;line-height:1.6}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:36px 0}}article{{padding:24px;border:1px solid #344642;border-radius:12px;background:#172724}}article span{{font-size:11px;letter-spacing:.13em;color:#b7c8bd}}strong{{display:block;font-size:48px;font-weight:400;color:#c0e69a;margin:14px 0}}article p{{font-size:13px;color:#b7c8bd}}.note{{border-left:2px solid #c0e69a;padding:12px 20px;background:#1b2b25;line-height:1.6;color:#c9d5c8}}h2{{font-size:24px;margin-top:46px;font-weight:500}}details{{border-bottom:1px solid #344642;padding:20px 0}}summary{{cursor:pointer;padding:4px 0}}summary span{{float:right;font-size:12px;color:#bddb9a}}details p{{color:#b6c7bf;line-height:1.7;max-width:900px}}h3{{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:#c0e69a;margin-top:24px}}footer{{margin-top:48px;font-size:12px;color:#91a69a;overflow-wrap:anywhere;line-height:1.8}}a{{color:#c0e69a}}@media(max-width:640px){{.cards{{grid-template-columns:1fr}}summary span{{float:none;display:block;margin-top:10px}}body{{padding:24px 18px}}}}
</style><header><b>INDUS</b><span class="tag">REPRODUCIBLE ML / CPU DEMO</span></header>
<h1>Engineering documents.<br>Measurable summaries.</h1><p class="intro">An inspectable experiment in document summarization. Compare a lead-sentence baseline with frequency-ranked extraction, then inspect every prediction against its source.</p>
<div class="cards"><article><span>DOCUMENTS EVALUATED</span><strong>{result["record_count"]}</strong><p>{result["sentence_count"]} sentences per summary · CPU</p></article>{cards}</div>
<p class="note">{esc(result["limitations"])} These are measured extractive baseline results. QLoRA training, GPU memory savings, and LLM quality have not been measured in this report.</p>
<h2>Document explorer</h2><p class="intro">Open any record to inspect the evidence behind the aggregate score.</p>{rows}
<footer>Metric: {esc(result["metric"])}<br>Dataset SHA-256: {result["dataset_sha256"]}<br>Indus {esc(result["indus_version"])} · Python {esc(result["python"])} · {esc(result["platform"])}<br><a href="https://github.com/sanjitchitturi/indus">View source on GitHub</a> · metrics.json and predictions.jsonl contain the machine-readable results.</footer></html>"""
