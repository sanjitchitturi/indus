"""Tests for data loading and dataset construction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from indus.data.dataset import (
    SummarizationDataset,
    load_dataset_splits,
    load_records,
    write_split_manifest,
)
from indus.exceptions import DataError


class TestLoadRecords:
    def test_load_jsonl(self, sample_jsonl_path: Path) -> None:
        records = load_records(sample_jsonl_path)
        assert len(records) == 3
        assert records[0]["document_id"] == "DOC-0001"

    def test_load_csv(self, sample_csv_path: Path) -> None:
        records = load_records(sample_csv_path)
        assert len(records) == 3
        assert records[0]["document_id"] == "DOC-0001"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DataError, match="not found"):
            load_records(tmp_path / "nonexistent.jsonl")

    def test_unsupported_format_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "data.parquet"
        path.touch()
        with pytest.raises(DataError, match="Unsupported file format"):
            load_records(path)

    def test_invalid_jsonl_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text("not valid json\n")
        with pytest.raises(DataError, match="Invalid JSON"):
            load_records(path)


class TestSummarizationDataset:
    def test_length(self, sample_records: list[dict]) -> None:
        ds = SummarizationDataset(sample_records)
        assert len(ds) == 3

    def test_getitem_keys(self, sample_records: list[dict]) -> None:
        ds = SummarizationDataset(sample_records)
        item = ds[0]
        assert "id" in item
        assert "prompt" in item
        assert "source" in item
        assert "target" in item

    def test_prompt_contains_source(self, sample_records: list[dict]) -> None:
        ds = SummarizationDataset(sample_records)
        item = ds[0]
        assert sample_records[0]["source_text"] in item["prompt"]

    def test_custom_columns(self) -> None:
        records = [{"text": "hello", "summary": "hi", "uid": "1"}]
        ds = SummarizationDataset(
            records, source_column="text", target_column="summary", id_column="uid"
        )
        assert len(ds) == 1
        assert ds[0]["target"] == "hi"

    def test_empty_records_raises(self) -> None:
        with pytest.raises(DataError, match="empty"):
            SummarizationDataset([])

    def test_missing_column_raises(self) -> None:
        with pytest.raises(DataError, match="Required column"):
            SummarizationDataset([{"wrong_field": "value"}])


class TestLoadDatasetSplits:
    def test_auto_split(self, sample_jsonl_path: Path) -> None:
        splits = load_dataset_splits(
            train_path=sample_jsonl_path,
            val_fraction=0.34,
            test_fraction=0.34,
            seed=42,
        )
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == 3
        assert len(splits["test"]) >= 1
        assert len(splits["val"]) >= 1

    def test_rejects_overlapping_separate_files(
        self, sample_jsonl_path: Path, sample_csv_path: Path
    ) -> None:
        with pytest.raises(DataError, match="occurs in both"):
            load_dataset_splits(
                train_path=sample_jsonl_path,
                val_path=sample_csv_path,
                test_path=sample_csv_path,
            )

    def test_separate_files(self, sample_records: list[dict], tmp_path: Path) -> None:
        paths = []
        for name, record in zip(("train", "val", "test"), sample_records, strict=True):
            path = tmp_path / f"{name}.jsonl"
            path.write_text(json.dumps(record) + "\n")
            paths.append(path)
        splits = load_dataset_splits(
            train_path=paths[0],
            val_path=paths[1],
            test_path=paths[2],
        )
        assert all(len(dataset) == 1 for dataset in splits.values())

    def test_rejects_same_source_with_different_ids(
        self, sample_records: list[dict], tmp_path: Path
    ) -> None:
        paths = []
        rows = [sample_records[0], {**sample_records[0], "document_id": "other"}, sample_records[2]]
        for name, row in zip(("train", "val", "test"), rows, strict=True):
            path = tmp_path / f"{name}.jsonl"
            path.write_text(json.dumps(row) + "\n")
            paths.append(path)
        with pytest.raises(DataError, match="Identical source text"):
            load_dataset_splits(paths[0], paths[1], paths[2])

    def test_writes_split_manifest(self, sample_jsonl_path: Path, tmp_path: Path) -> None:
        splits = load_dataset_splits(
            train_path=sample_jsonl_path,
            val_fraction=0.34,
            test_fraction=0.34,
            seed=7,
        )
        path = write_split_manifest(splits, tmp_path / "manifest.json", seed=7)
        payload = __import__("json").loads(path.read_text())
        assert payload["seed"] == 7
        assert sorted(payload["splits"]) == ["test", "train", "val"]
        assert sum(len(ids) for ids in payload["splits"].values()) == 3
