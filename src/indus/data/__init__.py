"""Data loading, preprocessing, and dataset management."""


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name in ("SummarizationDataset", "load_dataset_splits", "write_split_manifest"):
        from indus.data.dataset import (
            SummarizationDataset,
            load_dataset_splits,
            write_split_manifest,
        )

        mapping = {
            "SummarizationDataset": SummarizationDataset,
            "load_dataset_splits": load_dataset_splits,
            "write_split_manifest": write_split_manifest,
        }
        return mapping[name]
    if name in ("build_prompt", "tokenize_example"):
        from indus.data.preprocessing import build_prompt, tokenize_example

        mapping = {"build_prompt": build_prompt, "tokenize_example": tokenize_example}
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SummarizationDataset",
    "load_dataset_splits",
    "write_split_manifest",
    "build_prompt",
    "tokenize_example",
]
