#!/usr/bin/env python3
"""
Utility to add a `sentence` column to `sampled_1000_data.csv`.

The new column contains a randomly chosen caption string from the
`umls_json_info` field, which stores a JSON object with a `caption`
list per row.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Optional

import pandas as pd


def _load_umls(value: Any) -> Optional[dict]:
    """Robustly parse the umls_json_info entry into a dict."""
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # As a fallback, replace doubled quotes that occasionally survive CSV parsing.
        repaired = text.replace('""', '"')
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            return None


def _pick_sentence(value: Any, rng: random.Random) -> Optional[str]:
    """Pick a random caption entry from the UMLs JSON payload."""
    payload = _load_umls(value)
    if not payload:
        return None

    captions = payload.get("caption")
    if isinstance(captions, list) and captions:
        return rng.choice(captions)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add a random caption sentence column to sampled_1000_data.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/storage/homefs/hl24f166/homework/project/NLP_project_mimiccxr/sampled_1000_data.csv"),
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/storage/homefs/hl24f166/homework/project/NLP_project_mimiccxr/sampled_1000_data_with_sentence.csv"),
        help="Path to write the updated CSV with the new column.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    df = pd.read_csv(args.input)
    if "umls_json_info" not in df.columns:
        raise ValueError("Input CSV does not contain an `umls_json_info` column.")

    df["sentence"] = df["umls_json_info"].apply(lambda value: _pick_sentence(value, rng))
    df.to_csv(args.output, index=False)
    print(f"Wrote updated data with `sentence` column to {args.output}")


if __name__ == "__main__":
    main()


