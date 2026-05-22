"""토스 CTR 대회용 첫 LightGBM 베이스라인을 학습한다.

기본 설정은 빠른 반복을 위해 큰 parquet 파일을 샘플링한다. 파이프라인이 안정화되면
`--train-sample-frac`와 `--valid-sample-frac`를 늘려 실험한다.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

from metrics import competition_metrics


EXCLUDE_COLUMNS = {"ID", "seq", "clicked"}
SEQ_FEATURE_EXPRESSIONS = {
    "seq_char_len": "length(seq)",
    "seq_len": "array_length(string_split(seq, ','))",
    "seq_first": "cast(list_extract(string_split(seq, ','), 1) as float)",
    "seq_last": "cast(list_extract(string_split(seq, ','), array_length(string_split(seq, ','))) as float)",
    "seq_unique": "list_unique(string_split(seq, ','))",
}


def get_feature_columns(train_path: str, use_seq_features: bool) -> list[str]:
    schema = pq.ParquetFile(train_path).schema_arrow
    columns = [field.name for field in schema if field.name not in EXCLUDE_COLUMNS]
    if use_seq_features:
        columns += list(SEQ_FEATURE_EXPRESSIONS)
    return columns


def select_sql(columns: list[str], include_target: bool = True) -> str:
    selected = [
        f"{SEQ_FEATURE_EXPRESSIONS[col]} as {col}" if col in SEQ_FEATURE_EXPRESSIONS else col
        for col in columns
    ]
    if include_target:
        selected.append("clicked")
    return ", ".join(selected)


def read_duckdb_frame(
    con: duckdb.DuckDBPyConnection,
    path: str,
    columns: list[str],
    where: str | None = None,
    sample_frac: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    con.execute(f"select setseed({(seed % 10_000) / 10_000.0})")
    clauses = [f"select {select_sql(columns)} from read_parquet('{path}')"]
    filters = []
    if where:
        filters.append(where)
    if sample_frac < 1.0:
        filters.append(f"random() < {sample_frac}")
    if filters:
        clauses.append("where " + " and ".join(f"({f})" for f in filters))
    query = "\n".join(clauses)
    print(f"{path} 로딩: columns={len(columns) + 1}, where={where}, sample_frac={sample_frac}")
    return con.execute(query).fetchdf()


def coerce_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    for col in feature_columns:
        if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype("float32")
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype("float32")
    return df


def balanced_sample_weight(y: pd.Series) -> np.ndarray:
    y_arr = y.to_numpy(dtype=np.int8)
    n_pos = max(int((y_arr == 1).sum()), 1)
    n_neg = max(int((y_arr == 0).sum()), 1)
    weights = np.where(y_arr == 1, 0.5 / n_pos, 0.5 / n_neg)
    return weights / weights.mean()


def make_day7_split(
    con: duckdb.DuckDBPyConnection,
    train_path: str,
    columns: list[str],
    train_sample_frac: float,
    valid_sample_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = read_duckdb_frame(
        con,
        train_path,
        columns,
        where="day_of_week <> '7'",
        sample_frac=train_sample_frac,
        seed=seed,
    )
    valid_df = read_duckdb_frame(
        con,
        train_path,
        columns,
        where="day_of_week = '7'",
        sample_frac=valid_sample_frac,
        seed=seed + 1,
    )
    return train_df, valid_df


def make_random_split(
    con: duckdb.DuckDBPyConnection,
    train_path: str,
    columns: list[str],
    sample_frac: float,
    valid_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = read_duckdb_frame(
        con,
        train_path,
        columns,
        sample_frac=sample_frac,
        seed=seed,
    )
    train_idx, valid_idx = train_test_split(
        df.index,
        test_size=valid_size,
        stratify=df["clicked"],
        random_state=seed,
    )
    return df.loc[train_idx].copy(), df.loc[valid_idx].copy()


def train_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
) -> tuple[lgb.LGBMClassifier, dict[str, float], np.ndarray]:
    train_df = coerce_features(train_df, feature_columns)
    valid_df = coerce_features(valid_df, feature_columns)

    x_train = train_df[feature_columns]
    y_train = train_df["clicked"].astype("int8")
    x_valid = valid_df[feature_columns]
    y_valid = valid_df["clicked"].astype("int8")

    params = {
        "objective": "binary",
        "learning_rate": 0.03,
        "n_estimators": 2000,
        "num_leaves": 96,
        "min_child_samples": 200,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(
        x_train,
        y_train,
        sample_weight=balanced_sample_weight(y_train),
        eval_set=[(x_valid, y_valid)],
        eval_sample_weight=[balanced_sample_weight(y_valid)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)],
    )

    valid_pred = model.predict_proba(x_valid, num_iteration=model.best_iteration_)[:, 1]
    metrics = competition_metrics(y_valid, valid_pred)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return model, metrics, valid_pred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="train.parquet")
    parser.add_argument("--output-dir", default="models/baseline_lgbm")
    parser.add_argument("--validation", choices=["day7", "random"], default="day7")
    parser.add_argument("--train-sample-frac", type=float, default=0.10)
    parser.add_argument("--valid-sample-frac", type=float, default=0.30)
    parser.add_argument("--random-valid-size", type=float, default=0.20)
    parser.add_argument("--use-seq-features", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = get_feature_columns(args.train_path, args.use_seq_features)
    con = duckdb.connect()
    if args.validation == "day7":
        train_df, valid_df = make_day7_split(
            con,
            args.train_path,
            feature_columns,
            args.train_sample_frac,
            args.valid_sample_frac,
            args.seed,
        )
    else:
        train_df, valid_df = make_random_split(
            con,
            args.train_path,
            feature_columns,
            args.train_sample_frac,
            args.random_valid_size,
            args.seed,
        )

    print(f"train 크기: {train_df.shape}, valid 크기: {valid_df.shape}")
    print(f"train ctr: {train_df['clicked'].mean():.6f}, valid ctr: {valid_df['clicked'].mean():.6f}")

    model, metrics, valid_pred = train_model(train_df, valid_df, feature_columns, args.seed)
    model_path = output_dir / "model.txt"
    metadata_path = output_dir / "metadata.json"
    valid_pred_path = output_dir / "valid_predictions.csv"
    model.booster_.save_model(str(model_path), num_iteration=model.best_iteration_)
    metadata_path.write_text(
        json.dumps(
            {
                "feature_columns": feature_columns,
                "validation": args.validation,
                "train_sample_frac": args.train_sample_frac,
                "valid_sample_frac": args.valid_sample_frac,
                "use_seq_features": args.use_seq_features,
                "seed": args.seed,
                "best_iteration": model.best_iteration_,
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "clicked": valid_df["clicked"].astype("int8").to_numpy(),
            "prediction": valid_pred,
        }
    ).to_csv(valid_pred_path, index=False)
    print(f"모델 저장: {model_path}")
    print(f"메타데이터 저장: {metadata_path}")
    print(f"검증 예측 저장: {valid_pred_path}")


if __name__ == "__main__":
    main()
