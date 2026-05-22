"""Blend two DACON submission files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--other", required=True)
    parser.add_argument("--base-weight", type=float, default=0.7)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.base_weight <= 1.0:
        raise ValueError("--base-weight must be in [0, 1]")

    base = pd.read_csv(args.base)
    other = pd.read_csv(args.other)
    merged = base.merge(other, on="ID", suffixes=("_base", "_other"), validate="one_to_one")
    pred = (
        args.base_weight * merged["clicked_base"]
        + (1.0 - args.base_weight) * merged["clicked_other"]
    )
    submission = pd.DataFrame({"ID": merged["ID"], "clicked": pred.clip(1e-7, 1.0 - 1e-7)})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f"Wrote {output} with shape {submission.shape}")
    print(submission["clicked"].describe().to_string())


if __name__ == "__main__":
    main()
