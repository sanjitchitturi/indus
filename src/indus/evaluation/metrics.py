"""Metric computation for summarization evaluation.

Implements ROUGE-L, BLEU (via sacrebleu), and BERTScore F1.
"""

from __future__ import annotations

import re
from typing import Any

from indus.exceptions import EvaluationError
from indus.logging_utils import get_logger

logger = get_logger(__name__)

_NUMERIC_FACT = re.compile(r"(?<!\w)(?:\d+(?:\.\d+)?|\.\d+)(?:\s*[%°]|\s*[a-zA-Z]{1,8})?")


def _validate_pairs(predictions: list[str], references: list[str]) -> None:
    if len(predictions) != len(references):
        raise EvaluationError("Prediction count does not match reference count.")
    if not predictions:
        raise EvaluationError("Cannot compute metrics on empty inputs.")
    if any(not isinstance(text, str) for text in predictions + references):
        raise EvaluationError("Predictions and references must contain strings.")


def _numeric_facts(text: str) -> set[str]:
    """Return normalized numeric tokens with an optional unit."""
    return {re.sub(r"\s+", "", match.lower()) for match in _NUMERIC_FACT.findall(text)}


def compute_numeric_fact_consistency(
    predictions: list[str], references: list[str]
) -> dict[str, float]:
    """Measure exact numeric-fact overlap; this does not assess semantic correctness."""
    _validate_pairs(predictions, references)
    precision = recall = f1 = 0.0
    for prediction, reference in zip(predictions, references, strict=True):
        predicted = _numeric_facts(prediction)
        expected = _numeric_facts(reference)
        overlap = len(predicted & expected)
        p = overlap / len(predicted) if predicted else (1.0 if not expected else 0.0)
        r = overlap / len(expected) if expected else (1.0 if not predicted else 0.0)
        precision += p
        recall += r
        f1 += 2 * p * r / (p + r) if p + r else 0.0
    count = len(predictions)
    return {
        "numeric_fact_precision": round(precision / count, 4),
        "numeric_fact_recall": round(recall / count, 4),
        "numeric_fact_f1": round(f1 / count, 4),
    }


def compute_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute ROUGE-L score using the rouge_score library.

    Args:
        predictions: Generated summaries.
        references: Reference summaries.

    Returns:
        Dictionary with rougeL precision, recall, and fmeasure.
    """
    _validate_pairs(predictions, references)
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = {"rougeL_precision": 0.0, "rougeL_recall": 0.0, "rougeL_fmeasure": 0.0}

    for pred, ref in zip(predictions, references, strict=True):
        result = scorer.score(ref, pred)
        scores["rougeL_precision"] += result["rougeL"].precision
        scores["rougeL_recall"] += result["rougeL"].recall
        scores["rougeL_fmeasure"] += result["rougeL"].fmeasure

    n = len(predictions)
    return {k: round(v / n, 4) for k, v in scores.items()}


def compute_bleu(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute corpus-level BLEU score using sacrebleu.

    sacrebleu is preferred for reproducible BLEU computation because it uses
    a standardized tokenization and avoids common pitfalls of NLTK BLEU.

    Args:
        predictions: Generated summaries.
        references: Reference summaries.

    Returns:
        Dictionary with bleu score and brevity penalty.
    """
    _validate_pairs(predictions, references)
    import sacrebleu

    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return {
        "bleu": round(bleu.score / 100.0, 4),
        "bleu_bp": round(bleu.bp, 4),
    }


def compute_bertscore(
    predictions: list[str],
    references: list[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    batch_size: int = 16,
) -> dict[str, float]:
    """Compute BERTScore F1 using the bert_score library.

    Args:
        predictions: Generated summaries.
        references: Reference summaries.
        model_type: Pretrained model for BERTScore embedding.
        batch_size: Batch size for BERTScore computation.

    Returns:
        Dictionary with bertscore_precision, bertscore_recall, bertscore_f1.
    """
    _validate_pairs(predictions, references)
    from bert_score import score as bert_score_fn

    try:
        precision, recall, f1 = bert_score_fn(
            predictions,
            references,
            model_type=model_type,
            batch_size=batch_size,
            verbose=False,
        )
    except Exception as exc:
        raise EvaluationError(f"BERTScore computation failed: {exc}") from exc

    return {
        "bertscore_precision": round(precision.mean().item(), 4),
        "bertscore_recall": round(recall.mean().item(), 4),
        "bertscore_f1": round(f1.mean().item(), 4),
    }


def compute_metrics(
    predictions: list[str],
    references: list[str],
    bertscore_model: str = "microsoft/deberta-xlarge-mnli",
) -> dict[str, Any]:
    """Compute all evaluation metrics for a set of predictions.

    This is the primary entry point for evaluation. It computes ROUGE-L,
    BLEU (sacrebleu), and BERTScore F1 in a single call.

    Args:
        predictions: Generated summaries.
        references: Reference summaries.
        bertscore_model: Model identifier for BERTScore.

    Returns:
        Combined dictionary of all metric results.

    Raises:
        EvaluationError: If inputs are invalid or metric computation fails.
    """
    if len(predictions) != len(references):
        raise EvaluationError(
            f"Prediction count ({len(predictions)}) does not match "
            f"reference count ({len(references)})."
        )
    if not predictions:
        raise EvaluationError("Cannot compute metrics on empty inputs.")

    logger.info("Computing metrics on %d examples", len(predictions))
    results: dict[str, Any] = {}

    logger.info("Computing ROUGE-L...")
    results.update(compute_rouge(predictions, references))

    logger.info("Computing BLEU (sacrebleu)...")
    results.update(compute_bleu(predictions, references))

    logger.info("Computing numeric-fact consistency...")
    results.update(compute_numeric_fact_consistency(predictions, references))

    logger.info("Computing BERTScore...")
    results.update(compute_bertscore(predictions, references, model_type=bertscore_model))

    logger.info("Metrics: %s", results)
    return results
