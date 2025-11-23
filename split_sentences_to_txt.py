#!/usr/bin/env python3
"""
Split the first 100 `sentence` values from `sampled_1000_data_with_sentence.csv`
into four text files, 25 lines each, preserving the original row order.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

DEFAULT_INPUT = Path(
    "/storage/homefs/hl24f166/homework/project/NLP_project_mimiccxr/sampled_1000_data_with_sentence.csv"
)
DEFAULT_OUTPUT_DIR = Path("/storage/homefs/hl24f166/homework/project/NLP_project_mimiccxr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk the first 100 sentences into four TXT files with 25 lines each."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Source CSV path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the TXT files into.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="sentence_chunk",
        help="Filename prefix (suffixes 1-4.txt are appended).",
    )
    return parser.parse_args()


def write_chunk(sentences: List[str], output_path: Path) -> None:
    output_path.write_text("\n".join(sentences), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    if "sentence" not in df.columns:
        raise ValueError("Input CSV must contain a `sentence` column.")

    first_hundred = df["sentence"].head(100)
    if len(first_hundred) < 100:
        raise ValueError("Input CSV contains fewer than 100 rows.")

    chunk_size = 25
    for chunk_idx in range(4):
        start = chunk_idx * chunk_size
        end = start + chunk_size
        chunk = first_hundred.iloc[start:end].fillna("").astype(str).tolist()
        output_file = args.output_dir / f"{args.prefix}_{chunk_idx + 1}.txt"
        write_chunk(chunk, output_file)
        print(f"Wrote lines {start + 1}-{end} to {output_file}")


if __name__ == "__main__":
    main()



