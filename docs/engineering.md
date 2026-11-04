# Engineering notes

## Why a CPU baseline exists

A reviewer should be able to inspect an experiment without downloading a 7B model.
Lead extraction establishes a simple baseline; document-frequency ranking provides a
second deterministic method. Neither learns from the reference summaries. The report
keeps each prediction beside its source and reference so an aggregate score is auditable.

Frequency ranking sums non-stopword document term counts for each sentence, divides by
the square root of sentence length, selects the requested number, and restores source
order. Ties use source order. It is a heuristic, not a trained model or semantic retriever.
The stopword set and sentence-boundary rule are intentionally small and inspectable.

## Correctness decisions

- Validate all records, not just the first; report malformed JSON with line numbers.
- Reject split fractions that produce empty datasets instead of failing downstream.
- Include the last accumulation window even when it contains fewer microbatches.
  Normalize that window by its actual microbatch count. The loop averages microbatch
  losses; it does not implement token-count-weighted loss across uneven batches.
- Record gradient norm before clipping so unstable gradients remain visible.
- Start warmup with a nonzero learning rate; count optimizer updates with ceiling division.
- Mask prompt/padding labels and reserve the target's last token for EOS.
- Left-pad batched decoder-only generation and synchronize CUDA before/after timing.
- Publish input hashes and per-example outputs alongside measured CPU aggregate scores.
- Keep training dependencies optional so the installed CPU demo remains lightweight.
- Filter the optimizer to trainable parameters and save the best validation adapter,
  optimizer/scheduler state, and split manifest after each epoch.
- Report exact numeric-fact overlap as a diagnostic; it is a consistency signal, not
  a proof that a generated measurement is semantically or physically correct.

## Current boundaries

Single-process custom training; no distributed training, automatic mixed precision scaler,
checkpoint resume command, mid-epoch checkpointing, or production serving. Best-checkpoint
artifacts are persisted, but resume orchestration is still a follow-up task.
GPU profiling reports the current CUDA device; multi-device models need per-device
aggregation. Training artifacts are saved at the end of a successful run. Model/evaluator
APIs remain in the Transformers 4.x family through dependency bounds.

The sample corpus repeats a small number of synthetic topics and summaries. Random
splitting cannot establish generalization. Audit near-duplicates and split by project or
template family on a licensed real corpus before reporting model quality improvements.

## Next experiments

1. Establish a topic/project-disjoint corpus and persist split membership and environment lockfiles.
2. Run untuned and QLoRA adapters on identical held-out documents with paired confidence intervals.
3. Add numerical-fact consistency checks, since overlap metrics can miss incorrect measurements.
4. Add checkpoint/resume and best-validation adapter selection before long GPU experiments.
5. Measure GPU memory and throughput with warmup, repeated trials, and fixed output budgets.

## Resume wording

Use claims you can demonstrate from this repository, for example:

> Built a configurable QLoRA experimentation pipeline for engineering document summarization
> with target-masked training, MLflow tracking, automated CPU regression tests, and
> reproducible extractive baseline reports with per-document evidence.

Do not claim measured quality improvements or GPU memory savings until you have executed
and documented the corresponding experiments.
