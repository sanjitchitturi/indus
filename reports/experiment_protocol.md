# Results and evidence

The former report contained illustrative numbers for training quality, GPU memory, and
inference throughput. They were not measurements and have been removed to avoid
confusion with experimental evidence.

Run `python -m indus demo` for measured CPU extractive baseline scores. The command
writes an inspectable `report.html`, machine-readable `metrics.json`, and per-document
`predictions.jsonl`, including the input dataset SHA-256 and runtime environment.
See [the checked-in CPU run](cpu-baseline/metrics.json) for an actual execution.

These baselines are not fine-tuned language models. The sample corpus is synthetic,
with repeated templates and target summaries. Its scores do not establish generalization.
The demo evaluates the complete sample file; it does not train or tune on the references.

## Recording a QLoRA experiment

No measured QLoRA quality improvement, GPU memory reduction, or LLM throughput result is
currently claimed. To establish one, run the untuned baseline and adapter evaluation on
the same held-out documents and record:

- Input hashes, split membership, seed, resolved Hydra configuration, and commit SHA.
- GPU model, driver, CUDA and dependency versions, peak allocation, and batch size.
- Baseline and adapter ROUGE-L, BLEU, BERTScore, and per-document predictions.
- Numeric-fact precision, recall, and F1, interpreted as exact token overlap only.
- Warmup, generation lengths, sampling settings, and timing methodology.
- Checks for duplicate documents and near-duplicate templates across splits.

Use real documents with topic/project-disjoint splits before making resume claims about
quality gains. Never use the example corpus to substantiate production readiness.
