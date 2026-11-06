"""Shared test fixtures for Indus tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


@pytest.fixture
def sample_records() -> list[dict[str, str]]:
    """Minimal set of summarization records for testing."""
    return [
        {
            "document_id": "DOC-0001",
            "source_text": (
                "The structural inspection of the north wing revealed minor cracking "
                "in the concrete shear walls at levels 3 and 4. Crack widths were "
                "measured at 0.2 mm maximum."
            ),
            "target_summary": (
                "Minor cracking found in north wing concrete shear walls at levels 3 "
                "and 4 with maximum width of 0.2 mm."
            ),
        },
        {
            "document_id": "DOC-0002",
            "source_text": (
                "The HVAC system achieved a coefficient of performance of 5.9 under "
                "full load test conditions. Supply air temperature was stable at 55 "
                "degrees Fahrenheit."
            ),
            "target_summary": (
                "HVAC system COP measured at 5.9 under full load. Supply air stable "
                "at 55 degrees F."
            ),
        },
        {
            "document_id": "DOC-0003",
            "source_text": (
                "Soil borings at the proposed bridge site encountered bedrock at a "
                "depth of 40 feet. Groundwater was found at 18 feet below grade."
            ),
            "target_summary": "Bedrock encountered at 40 feet depth. Groundwater at 18 feet below grade.",
        },
    ]


@pytest.fixture
def sample_jsonl_path(sample_records: list[dict], tmp_path: Path) -> Path:
    """Write sample records to a temporary JSONL file."""
    path = tmp_path / "test_data.jsonl"
    with open(path, "w") as f:
        for record in sample_records:
            f.write(json.dumps(record) + "\n")
    return path


@pytest.fixture
def sample_csv_path(sample_records: list[dict], tmp_path: Path) -> Path:
    """Write sample records to a temporary CSV file."""
    path = tmp_path / "test_data.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sample_records[0].keys())
        writer.writeheader()
        writer.writerows(sample_records)
    return path


@pytest.fixture
def predictions() -> list[str]:
    """Sample predictions for metric testing."""
    return [
        "Minor cracking in shear walls at levels 3 and 4.",
        "HVAC COP of 5.9 under full load. Air temperature at 55 F.",
        "Bedrock at 40 feet. Groundwater at 18 feet.",
    ]


@pytest.fixture
def references() -> list[str]:
    """Sample references for metric testing."""
    return [
        "Minor cracking found in north wing concrete shear walls at levels 3 and 4.",
        "HVAC system COP measured at 5.9 under full load. Supply air stable at 55 degrees F.",
        "Bedrock encountered at 40 feet depth. Groundwater at 18 feet below grade.",
    ]
