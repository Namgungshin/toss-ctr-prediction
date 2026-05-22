"""Calibrate a LightGBM submission using the training prior or validation labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def prior_correct(pred: np.ndarray, train_ctr: float) -> np.ndarray:
    """Undo class-balanced training prior shift.

    With equal total class weight, the model's probability is close to a
    balanced-prior posterior. Convert those odds back to the observed CTR prior.
    """
    pred = np.clip(pred, 1e-15, 1.0 - 1e-15)
    odds = pred / (1.0 - pred)
    prior_odds = train_ctr / (1.0 - train_ctr)
    corrected_odds = odds * prior_odds
    return corrected_odds / (1.0 + corrected_odds)


def temperature_scale(pred: np.ndarray, temperature: float) -> np.ndarray:
    pred = np.clip(pred, 1e-15, 1.0 - 1e-15)
    logits = np.log(pred / (1.0 - pred)) / temperature
    return 1.0 / (1.0 + np.exp(-logits))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="submissions/baseline_lgbm.csv")
    parser.add_argument("--output", default="submissions/baseline_lgbm_prior_calibrated.csv")
    parser.add_argument("--train-path", default="train.parquet")
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    con = duckdb.connect()
    train_ctr = con.execute(
        f"select avg(clicked) from read_parquet('{args.train_path}')"
    ).fetchone()[0]

    submission = pd.read_csv(args.input)
    pred = submission["clicked"].to_numpy(dtype=np.float64)
    pred = prior_correct(pred, train_ctr)
    if args.temperature != 1.0:
        pred = temperature_scale(pred, args.temperature)
    submission["clicked"] = np.clip(pred, 1e-7, 1.0 - 1e-7)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f"train_ctr={train_ctr:.8f}")
    print(f"Wrote {output} with shape {submission.shape}")
    print(submission["clicked"].describe().to_string())


if __name__ == "__main__":
    main()
