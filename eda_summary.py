"""토스 CTR parquet 파일의 간단한 EDA 요약을 생성한다."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pyarrow.parquet as pq


DEFAULT_CARDINALITY_COLS = [
    "gender",
    "age_group",
    "inventory_id",
    "day_of_week",
    "hour",
    "seq",
    "l_feat_1",
    "l_feat_14",
    "feat_a_1",
    "feat_b_1",
    "feat_c_1",
    "feat_d_1",
    "feat_e_1",
    "history_a_1",
    "history_b_1",
]


def parquet_overview(path: str) -> list[str]:
    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    type_counts: dict[str, int] = {}
    for field in schema:
        type_counts[str(field.type)] = type_counts.get(str(field.type), 0) + 1

    lines = [
        f"### {path}",
        "",
        f"- 행 수: {pf.metadata.num_rows:,}",
        f"- 열 수: {pf.metadata.num_columns:,}",
        f"- row group 개수: {pf.metadata.num_row_groups:,}",
        f"- 타입 분포: {type_counts}",
        "",
        "| 인덱스 | 컬럼 | 타입 |",
        "|---:|---|---|",
    ]
    for idx, field in enumerate(schema):
        lines.append(f"| {idx} | `{field.name}` | `{field.type}` |")
    lines.append("")
    return lines


def df_to_markdown(con: duckdb.DuckDBPyConnection, query: str) -> str:
    df = con.execute(query).fetchdf()
    headers = [str(col) for col in df.columns]
    rows = [[str(value) for value in row] for row in df.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_summary(train_path: str, test_path: str, cardinality_cols: list[str]) -> str:
    con = duckdb.connect()
    lines = ["# 토스 CTR EDA 요약", ""]
    lines += parquet_overview(train_path)
    lines += parquet_overview(test_path)

    lines += ["## 타깃 분포", ""]
    lines.append(
        df_to_markdown(
            con,
            f"""
            select
                count(*) as n,
                sum(clicked) as clicks,
                avg(clicked) as ctr
            from read_parquet('{train_path}')
            """,
        )
    )
    lines.append("")

    lines += ["## 요일/시간 분포", "", "### Train 요일별 분포", ""]
    lines.append(
        df_to_markdown(
            con,
            f"""
            select day_of_week, count(*) as n, avg(clicked) as ctr
            from read_parquet('{train_path}')
            group by day_of_week
            order by day_of_week
            """,
        )
    )
    lines += ["", "### Test 요일별 분포", ""]
    lines.append(
        df_to_markdown(
            con,
            f"""
            select day_of_week, count(*) as n
            from read_parquet('{test_path}')
            group by day_of_week
            order by day_of_week
            """,
        )
    )
    lines += ["", "### Train 시간별 분포", ""]
    lines.append(
        df_to_markdown(
            con,
            f"""
            select hour, count(*) as n, avg(clicked) as ctr
            from read_parquet('{train_path}')
            group by hour
            order by hour
            """,
        )
    )
    lines += ["", "### Test 시간별 분포", ""]
    lines.append(
        df_to_markdown(
            con,
            f"""
            select hour, count(*) as n
            from read_parquet('{test_path}')
            group by hour
            order by hour
            """,
        )
    )
    lines.append("")

    available = set(pq.ParquetFile(train_path).schema_arrow.names)
    selected = [col for col in cardinality_cols if col in available]
    union_parts = [
        f"""
        select
            '{col}' as column_name,
            approx_count_distinct({col}) as approx_nunique,
            count(*) - count({col}) as nulls
        from read_parquet('{train_path}')
        """
        for col in selected
    ]
    lines += ["## 주요 컬럼 cardinality", ""]
    lines.append(df_to_markdown(con, " union all ".join(union_parts)))
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="train.parquet")
    parser.add_argument("--test-path", default="test.parquet")
    parser.add_argument("--output", default="reports/eda_summary.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        build_summary(args.train_path, args.test_path, DEFAULT_CARDINALITY_COLS),
        encoding="utf-8",
    )
    print(f"EDA 요약 저장: {output}")


if __name__ == "__main__":
    main()
