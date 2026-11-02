# Plan

## Why this project

Recent job fit checks (BMS Analytical AI, Exelixis AI Agentics, Oracle ML Engineer, Glean ML Engineer) kept flagging the same gaps: Databricks or Azure, MLflow, and running models in production (scheduled retraining, a model registry, monitoring). This repo closes those with working code on a real model, not a tutorial dataset.

## Milestones

| # | Milestone | Status |
|---|---|---|
| 1 | Pipeline core: data versioning, two candidates, MLflow tracking and registry, champion/challenger gate, aliases, rollback | Done |
| 2 | Monitoring: drift (PSI, league and volume shift), live skill vs holdout skill, `check-alerts` | Done |
| 3 | 2012-2025 walk-forward replay with results committed | Done |
| 4 | CI (ruff + pytest), weekly GitHub Actions pipeline run | Done |
| 5 | Databricks Asset Bundle: schema, experiment, weekly retrain job, backfill job, failure email | Written, not yet deployed |
| 6 | Deploy to Databricks Free Edition and run the backfill there; add screenshots of the MLflow UI and job runs | Next (needs a Free Edition account) |
| 7 | Add the PyMC hierarchical model from mlb-player-projections as a third challenger (fewer draws; runs on serverless) | Planned |
| 8 | Serving: a small FastAPI endpoint that loads `models:/mlb_<stat>@champion` | Planned |
| 9 | Optional Azure version: Azure ML managed MLflow + scheduled pipeline job, for Azure-specific postings | Planned |

## Design decisions

- **Cheap models first.** The pipeline is the point. Marcel and an empirical-Bayes beta-binomial fit in milliseconds, so a 14-season replay takes a minute and CI can run the whole thing.
- **The gate is conservative on purpose.** A challenger replaces the champion's model type only when a paired bootstrap over players puts its 95% interval below zero. A routine refresh of the same type always goes through, but nothing is promoted if it fails to beat league average on the holdout.
- **Skip when nothing changed.** The weekly job hashes the data and stops if it matches the champion's training data. MLB data lands once a year, so this is the normal case, and it keeps the schedule cheap.
- **One codebase, two runtimes.** `config.py` detects Databricks and switches to the workspace MLflow, Unity Catalog registry, and Delta tables. Locally it uses SQLite and Parquet.
- **Alerts that block vs alerts that inform.** Performance decay and rate drift fail the job. League and volume shifts are logged but don't, because HR rate moves more than 5% in most seasons.

## Later projects in the same series

1. This pipeline.
2. A Go service on Kubernetes with CI/CD.
3. A streaming data project (Kafka or Kinesis into Postgres).
