"""Evaluation metrics and pipelines for summarization."""


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name in (
        "compute_metrics",
        "compute_rouge",
        "compute_bleu",
        "compute_bertscore",
        "compute_numeric_fact_consistency",
    ):
        from indus.evaluation import metrics as _m

        return getattr(_m, name)
    if name == "Evaluator":
        from indus.evaluation.evaluator import Evaluator

        return Evaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "compute_metrics",
    "compute_rouge",
    "compute_bleu",
    "compute_bertscore",
    "compute_numeric_fact_consistency",
    "Evaluator",
]
