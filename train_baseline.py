"""토스 CTR 대회용 첫 LightGBM 베이스라인을 학습한다.

기본 설정은 빠른 반복을 위해 큰 parquet 파일을 샘플링한다. 파이프라인이 안정화되면
`--train-sample-frac`와 `--valid-sample-frac`를 늘려 실험한다.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

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
COUNT_FEATURE_GROUP_SETS = {
    "minimal": [
        ("inventory_id", ["inventory_id"]),
        ("hour", ["hour"]),
        ("l_feat_14", ["l_feat_14"]),
        ("inventory_hour", ["inventory_id", "hour"]),
    ],
    "stable": [
        ("gender", ["gender"]),
        ("age_group", ["age_group"]),
        ("inventory_id", ["inventory_id"]),
        ("hour", ["hour"]),
        ("l_feat_14", ["l_feat_14"]),
        ("inventory_hour", ["inventory_id", "hour"]),
        ("inventory_age", ["inventory_id", "age_group"]),
        ("gender_age_inventory", ["gender", "age_group", "inventory_id"]),
    ],
    "full": [
        ("gender", ["gender"]),
        ("age_group", ["age_group"]),
        ("inventory_id", ["inventory_id"]),
        ("hour", ["hour"]),
        ("l_feat_14", ["l_feat_14"]),
        ("feat_b_1", ["feat_b_1"]),
        ("inventory_hour", ["inventory_id", "hour"]),
        ("inventory_age", ["inventory_id", "age_group"]),
        ("inventory_lfeat14", ["inventory_id", "l_feat_14"]),
        ("lfeat14_hour", ["l_feat_14", "hour"]),
        ("gender_age_inventory", ["gender", "age_group", "inventory_id"]),
    ],
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


def count_key(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    parts = [df[col].astype("string").fillna("__NA__") for col in columns]
    key = parts[0]
    for part in parts[1:]:
        key = key + "||" + part
    return key


def add_count_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    groups: list[tuple[str, list[str]]],
    include_freq: bool,
) -> tuple[list[str], dict[str, Any]]:
    feature_names: list[str] = []
    count_maps: dict[str, Any] = {}
    n_train = len(train_df)

    for name, columns in groups:
        missing = [col for col in columns if col not in train_df.columns]
        if missing:
            print(f"count feature {name} 건너뜀: 없는 컬럼 {missing}")
            continue

        train_key = count_key(train_df, columns)
        valid_key = count_key(valid_df, columns)
        counts = train_key.value_counts(dropna=False)
        count_map = {str(key): int(value) for key, value in counts.items()}

        count_col = f"cnt_{name}_log"
        train_count = train_key.map(count_map).fillna(0).astype("float32")
        valid_count = valid_key.map(count_map).fillna(0).astype("float32")
        train_df[count_col] = np.log1p(train_count).astype("float32")
        valid_df[count_col] = np.log1p(valid_count).astype("float32")

        info = {
            "columns": columns,
            "count_col": count_col,
            "n_train": n_train,
            "counts": count_map,
        }
        feature_names.append(count_col)
        if include_freq:
            freq_col = f"cnt_{name}_freq"
            train_df[freq_col] = (train_count / max(n_train, 1)).astype("float32")
            valid_df[freq_col] = (valid_count / max(n_train, 1)).astype("float32")
            feature_names.append(freq_col)
            info["freq_col"] = freq_col

        count_maps[name] = {
            **info,
        }
        print(f"count feature 추가: {name}, unique={len(count_map):,}")

    return feature_names, count_maps


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
    parser.add_argument("--use-count-features", action="store_true")
    parser.add_argument(
        "--count-feature-set",
        choices=sorted(COUNT_FEATURE_GROUP_SETS),
        default="stable",
    )
    parser.add_argument("--count-include-freq", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_feature_columns = get_feature_columns(args.train_path, args.use_seq_features)
    feature_columns = list(raw_feature_columns)
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

    count_maps: dict[str, Any] = {}
    count_feature_columns: list[str] = []
    if args.use_count_features:
        count_feature_columns, count_maps = add_count_features(
            train_df,
            valid_df,
            COUNT_FEATURE_GROUP_SETS[args.count_feature_set],
            args.count_include_freq,
        )
        feature_columns += count_feature_columns
        print(
            f"count feature set={args.count_feature_set}, "
            f"include_freq={args.count_include_freq}, 총 {len(count_feature_columns)}개 추가"
        )

    model, metrics, valid_pred = train_model(train_df, valid_df, feature_columns, args.seed)
    model_path = output_dir / "model.txt"
    metadata_path = output_dir / "metadata.json"
    count_maps_path = output_dir / "count_maps.json"
    valid_pred_path = output_dir / "valid_predictions.csv"
    model.booster_.save_model(str(model_path), num_iteration=model.best_iteration_)
    if args.use_count_features:
        count_maps_path.write_text(
            json.dumps(count_maps, ensure_ascii=False),
            encoding="utf-8",
        )
    metadata_path.write_text(
        json.dumps(
            {
                "feature_columns": feature_columns,
                "raw_feature_columns": raw_feature_columns,
                "count_feature_columns": count_feature_columns,
                "validation": args.validation,
                "train_sample_frac": args.train_sample_frac,
                "valid_sample_frac": args.valid_sample_frac,
                "use_seq_features": args.use_seq_features,
                "use_count_features": args.use_count_features,
                "count_feature_set": args.count_feature_set,
                "count_include_freq": args.count_include_freq,
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
