"""Runtime settings. The same code runs locally, in GitHub Actions and on Databricks.

On Databricks (detected by DATABRICKS_RUNTIME_VERSION) runs are tracked in the
workspace's MLflow, models go to the Unity Catalog registry, and tables are Delta
tables. Locally, MLflow uses a SQLite file and tables are Parquet files.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def on_databricks() -> bool:
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


@dataclass(frozen=True)
class Thresholds:
    psi_alert: float = 0.2  # population stability index of player rates vs the training window
    league_shift_alert: float = 0.05  # relative change in league rate, season over season
    volume_shift_alert: float = 0.25  # relative change in median opportunities per qualified player
    edge_retained_alert: float = 0.5  # share of the holdout edge over league average kept in production


@dataclass(frozen=True)
class Settings:
    databricks: bool
    tracking_uri: str
    registry_uri: str
    experiment: str
    model_prefix: str  # registered model name = prefix + stat
    table_root: str  # local directory, or catalog.schema on Databricks
    thresholds: Thresholds = field(default_factory=Thresholds)

    def model_name(self, stat: str) -> str:
        return f"{self.model_prefix}{stat}"


def load() -> Settings:
    if on_databricks():
        catalog = os.environ.get("MLBOPS_CATALOG", "workspace")
        schema = os.environ.get("MLBOPS_SCHEMA", "mlb_mlops")
        return Settings(
            databricks=True,
            tracking_uri="databricks",
            registry_uri="databricks-uc",
            experiment=os.environ.get("MLBOPS_EXPERIMENT", "/Shared/mlb-mlops"),
            model_prefix=f"{catalog}.{schema}.mlb_",
            table_root=f"{catalog}.{schema}",
        )
    home = os.environ.get("MLBOPS_HOME", os.path.abspath(".mlbops"))
    os.makedirs(home, exist_ok=True)
    uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{home}/mlflow.db")
    return Settings(
        databricks=False,
        tracking_uri=uri,
        registry_uri=uri,
        experiment=os.environ.get("MLBOPS_EXPERIMENT", "mlb-mlops"),
        model_prefix="mlb_",
        table_root=os.path.join(home, "tables"),
    )
