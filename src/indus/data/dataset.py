"""Dataset loading and splitting for summarization tasks."""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any

from indus.exceptions import DataError
from indus.logging_utils import get_logger

logger = get_logger(__name__)


class SummarizationDataset:
    """PyTorch dataset wrapper for document summarization records.

    Each record must contain a source text and target summary. The dataset
    applies a prompt template and returns tokenized examples ready for
    causal language model training.

    Args:
        records: List of dictionaries with source and target fields.
        source_column: Key for the input document text.
        target_column: Key for the reference summary.
        id_column: Key for the document identifier.
        prompt_template: Format string with a ``{source}`` placeholder.
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        source_column: str = "source_text",
        target_column: str = "target_summary",
        id_column: str = "document_id",
        prompt_template: str = (
            "Summarize the following engineering document:\n\n{source}\n\nSummary:"
        ),
    ) -> None:
        self.records = records
        self.source_column = source_column
        self.target_column = target_column
        self.id_column = id_column
        self.prompt_template = prompt_template
        self._validate()

    def _validate(self) -> None:
        if not self.records:
            raise DataError("Dataset is empty. Provide at least one record.")
        for index, sample in enumerate(self.records, start=1):
            if not isinstance(sample, dict):
                raise DataError(f"Record {index} must be an object.")
            for col in [self.source_column, self.target_column]:
                if col not in sample:
                    raise DataError(f"Required column '{col}' missing in record {index}.")
                if not isinstance(sample[col], str) or not sample[col].strip():
                    raise DataError(f"Record {index}: '{col}' must be non-empty text.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, str]:
        record = self.records[idx]
        prompt = self.prompt_template.format(source=record[self.source_column])
        return {
            "id": record.get(self.id_column, str(idx)),
            "prompt": prompt,
            "source": record[self.source_column],
            "target": record[self.target_column],
        }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load records from a JSONL file."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise DataError(f"Invalid JSON on line {line_num} of {path}: {exc}") from exc
    return records


def _load_csv(path: Path) -> list[dict[str, Any]]:
    """Load records from a CSV file."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load records from a JSONL or CSV file.

    Args:
        path: Path to the data file.

    Returns:
        List of record dictionaries.

    Raises:
        DataError: If the file format is unsupported or the file is not found.
    """
    path = Path(path)
    if not path.exists():
        raise DataError(f"Data file not found: {path}")

    if path.suffix == ".jsonl":
        records = _load_jsonl(path)
    elif path.suffix == ".csv":
        records = _load_csv(path)
    else:
        raise DataError(f"Unsupported file format: {path.suffix}. Use .jsonl or .csv.")

    logger.info("Loaded %d records from %s", len(records), path)
    return records


def load_dataset_splits(
    train_path: str | Path,
    val_path: str | Path | None = None,
    test_path: str | Path | None = None,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
    **dataset_kwargs: Any,
) -> dict[str, SummarizationDataset]:
    """Load and optionally split data into train, validation, and test sets.

    If val_path or test_path is None, the corresponding split is carved
    from the training data using the specified fractions.

    Args:
        train_path: Path to the training data file.
        val_path: Optional path to a separate validation file.
        test_path: Optional path to a separate test file.
        val_fraction: Fraction of training data for validation if val_path is None.
        test_fraction: Fraction of training data for test if test_path is None.
        seed: Random seed for splitting.
        **dataset_kwargs: Forwarded to SummarizationDataset.

    Returns:
        Dictionary mapping split names to SummarizationDataset instances.
    """
    for name, fraction in [("val_fraction", val_fraction), ("test_fraction", test_fraction)]:
        if not math.isfinite(fraction) or not 0 < fraction < 1:
            raise DataError(f"{name} must be between 0 and 1 (exclusive).")
    requested = (val_fraction if val_path is None else 0) + (
        test_fraction if test_path is None else 0
    )
    if requested >= 1:
        raise DataError("Validation and test fractions must leave a non-empty training split.")
    all_records = load_records(train_path)
    splits: dict[str, SummarizationDataset] = {}

    if val_path is not None:
        val_records = load_records(val_path)
    else:
        val_records = None

    if test_path is not None:
        test_records = load_records(test_path)
    else:
        test_records = None

    if val_records is None or test_records is None:
        rng = random.Random(seed)
        rng.shuffle(all_records)

        n = len(all_records)
        n_test = int(n * test_fraction) if test_records is None else 0
        n_val = int(n * val_fraction) if val_records is None else 0
        if (test_records is None and n_test == 0) or (val_records is None and n_val == 0):
            raise DataError(
                "Dataset too small for requested splits; add records or supply separate files."
            )

        if test_records is None:
            test_records = all_records[:n_test]
            all_records = all_records[n_test:]

        if val_records is None:
            val_records = all_records[:n_val]
            all_records = all_records[n_val:]

    splits["train"] = SummarizationDataset(all_records, **dataset_kwargs)
    splits["val"] = SummarizationDataset(val_records, **dataset_kwargs)
    splits["test"] = SummarizationDataset(test_records, **dataset_kwargs)

    for name, ds in splits.items():
        logger.info("Split '%s': %d examples", name, len(ds))

    return splits


def write_split_manifest(
    splits: dict[str, SummarizationDataset],
    path: str | Path,
    *,
    seed: int,
    source_paths: dict[str, str | Path | None] | None = None,
) -> Path:
    """Persist split membership and input provenance for reproducible runs."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "source_paths": {
            key: Path(value).name if value is not None else None
            for key, value in (source_paths or {}).items()
        },
        "splits": {
            name: [
                str(example.get(dataset.id_column, index))
                for index, example in enumerate(dataset.records)
            ]
            for name, dataset in splits.items()
        },
    }
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output
