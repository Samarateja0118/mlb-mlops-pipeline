"""Append-only output tables: Delta on Databricks, Parquet locally.

Tables written by the pipeline:
- projections: one row per player, stat and target season, from the champion at the time
- monitoring:  one row per stat and season when realized results arrive
- pipeline_runs: one row per stat per pipeline run, with the retrain decision and reason
"""
from __future__ import annotations

import os

import pandas as pd

from .config import Settings


def append(settings: Settings, name: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    if settings.databricks:
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        # Spark can't infer a type for an all-null column (e.g. champion_before on the first run).
        df = df.astype({c: "string" for c in df.columns if df[c].dtype == object})
        spark.createDataFrame(df).write.mode("append").option("mergeSchema", "true").saveAsTable(
            f"{settings.table_root}.{name}"
        )
        return
    os.makedirs(settings.table_root, exist_ok=True)
    path = os.path.join(settings.table_root, f"{name}.parquet")
    if os.path.exists(path):
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df.to_parquet(path, index=False)


def read(settings: Settings, name: str) -> pd.DataFrame:
    if settings.databricks:
        from pyspark.sql import SparkSession

        return SparkSession.builder.getOrCreate().table(f"{settings.table_root}.{name}").toPandas()
    path = os.path.join(settings.table_root, f"{name}.parquet")
    return pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()
