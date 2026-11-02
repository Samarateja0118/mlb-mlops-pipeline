"""MLflow model packaging and the champion alias.

Each stat has its own registered model (mlb_bat_k, ...). The version serving
projections carries the alias `champion`; the one it replaced keeps `previous`,
so a rollback is one alias move (`mlbops rollback --stat bat_k`).
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from .models import RateModel

CHAMPION, PREVIOUS = "champion", "previous"
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


class RatePyfunc(mlflow.pyfunc.PythonModel):
    """Input: season lines (playerID, yearID, age, k, n). Output: next-season projected rate per player."""

    def load_context(self, context):
        with open(context.artifacts["params"]) as f:
            p = json.load(f)
        p["weights"] = tuple(p["weights"])
        self.model = RateModel(**p)

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        return self.model.predict(model_input)


def log_model(model: RateModel, example: pd.DataFrame) -> str:
    """Log inside the active run; returns the model URI."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "params.json")
        with open(path, "w") as f:
            json.dump(model.params(), f)
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=RatePyfunc(),
            artifacts={"params": path},
            code_paths=[PACKAGE_DIR],
            signature=infer_signature(example, model.predict(example)),
            input_example=example.head(5),
            pip_requirements=["pandas", "numpy", "scipy", f"mlflow=={mlflow.__version__}"],
        )
    return info.model_uri


@dataclass
class Champion:
    version: str
    tags: dict

    @property
    def kind(self) -> str:
        return self.tags["kind"]

    @property
    def trained_through(self) -> int:
        return int(self.tags["trained_through"])

    @property
    def holdout_skill(self) -> float:
        return float(self.tags["holdout_skill"])


def champion(name: str) -> Champion | None:
    try:
        mv = MlflowClient().get_model_version_by_alias(name, CHAMPION)
    except MlflowException:
        return None
    return Champion(mv.version, dict(mv.tags))


def load(name: str, alias: str = CHAMPION):
    return mlflow.pyfunc.load_model(f"models:/{name}@{alias}")


def register(name: str, model_uri: str, tags: dict) -> str:
    client = MlflowClient()
    mv = mlflow.register_model(model_uri, name)
    for k, v in tags.items():
        client.set_model_version_tag(name, mv.version, k, str(v))
    return mv.version


def promote(name: str, version: str) -> None:
    """Point `champion` at `version`; the old champion keeps `previous` for rollback."""
    client = MlflowClient()
    old = champion(name)
    if old and old.version != str(version):
        client.set_registered_model_alias(name, PREVIOUS, old.version)
    client.set_registered_model_alias(name, CHAMPION, version)


def rollback(name: str) -> str:
    client = MlflowClient()
    prev = client.get_model_version_by_alias(name, PREVIOUS)
    cur = client.get_model_version_by_alias(name, CHAMPION)
    client.set_registered_model_alias(name, CHAMPION, prev.version)
    client.set_registered_model_alias(name, PREVIOUS, cur.version)
    return prev.version
