"""Installed CLI for CPU experiments and dataset inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from indus import __version__
from indus.exceptions import DataError


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Indus: engineering document summarization experiments"
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser(
        "demo", help="Compare CPU extractive baselines and build an HTML report"
    )
    demo.add_argument(
        "--data", type=Path, default=Path("data/sample/eng_summarization_sample.jsonl")
    )
    demo.add_argument("--output", type=Path, default=Path("outputs/demo"))
    demo.add_argument("--sentences", type=int, default=2)
    validate = commands.add_parser(
        "validate", help="Validate every dataset record and inspect duplicates"
    )
    validate.add_argument("data", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "demo":
            from indus.demo import run_demo

            result = run_demo(args.data, args.output, args.sentences)
            print(json.dumps(result, indent=2))
            print(f"\nReport: {(args.output / 'report.html').resolve()}")
        else:
            from collections import Counter

            from indus.data.dataset import SummarizationDataset, load_records

            records = load_records(args.data)
            dataset = SummarizationDataset(records)
            sources = Counter(row["source_text"] for row in records)
            targets = Counter(row["target_summary"] for row in records)
            print(
                json.dumps(
                    {
                        "records": len(dataset),
                        "unique_sources": len(sources),
                        "unique_targets": len(targets),
                        "duplicate_source_rows": sum(n - 1 for n in sources.values()),
                    },
                    indent=2,
                )
            )
    except (DataError, OSError, ValueError) as exc:
        parser.exit(2, f"indus: {exc}\n")
