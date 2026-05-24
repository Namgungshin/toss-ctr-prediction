"""저장된 LightGBM 베이스라인 모델로 제출 파일을 생성한다."""

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


SEQ_FEATURE_EXPRESSIONS = {
    "seq_char_len": "length(seq)",
    "seq_len": "array_length(string_split(seq, ','))",
    "seq_first": "cast(list_extract(string_split(seq, ','), 1) as float)",
    "seq_last": "cast(list_extract(string_split(seq, ','), array_length(string_split(seq, ','))) as float)",
    "seq_unique": "list_unique(string_split(seq, ','))",
}


def select_expr(column: str) -> str:
    if column in SEQ_FEATURE_EXPRESSIONS:
        return f"{SEQ_FEATURE_EXPRESSIONS[column]} as {column}"
    return column


def coerce_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    for col in feature_columns:
        if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if pd.api.types.is_float_dtype(df[col]) or pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype("float32")
    return df


def count_key(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    parts = [df[col].astype("string").fillna("__NA__") for col in columns]
    key = parts[0]
    for part in parts[1:]:
        key = key + "||" + part
    return key


def add_count_features(df: pd.DataFrame, count_maps: dict[str, Any]) -> None:
    for name, info in count_maps.items():
        columns = info["columns"]
        count_col = info["count_col"]
        n_train = max(int(info["n_train"]), 1)
        counts = info["counts"]
        key = count_key(df, columns)
        count = key.map(counts).fillna(0).astype("float32")
        df[count_col] = np.log1p(count).astype("float32")
        if "freq_col" in info:
            df[info["freq_col"]] = (count / n_train).astype("float32")
        print(f"count feature 적용: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-path", default="test.parquet")
    parser.add_argument("--sample-submission", default="sample_submission.csv")
    parser.add_argument("--model-dir", default="models/baseline_lgbm")
    parser.add_argument("--output", default="submissions/baseline_lgbm.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    feature_columns = metadata["feature_columns"]
    raw_feature_columns = metadata.get("raw_feature_columns", feature_columns)

    selected = ", ".join(["ID"] + [select_expr(col) for col in raw_feature_columns])
    con = duckdb.connect()
    test_df = con.execute(
        f"select {selected} from read_parquet('{args.test_path}') order by ID"
    ).fetchdf()
    ids = test_df["ID"].copy()
    count_maps_path = model_dir / "count_maps.json"
    if metadata.get("use_count_features") and count_maps_path.exists():
        count_maps = json.loads(count_maps_path.read_text(encoding="utf-8"))
        add_count_features(test_df, count_maps)
    test_df = coerce_features(test_df, feature_columns)

    model = lgb.Booster(model_file=str(model_dir / "model.txt"))
    pred = model.predict(test_df[feature_columns])

    sample = pd.read_csv(args.sample_submission)
    submission = pd.DataFrame({"ID": ids, "clicked": pred})
    submission = sample[["ID"]].merge(submission, on="ID", how="left")
    if submission["clicked"].isna().any():
        raise ValueError("sample_submission.csv와 정렬한 뒤 누락된 예측값이 있습니다.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f"제출 파일 저장: {output}, shape={submission.shape}")
    print(submission.head().to_string(index=False))


if __name__ == "__main__":
    main()
