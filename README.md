# MLB Projections MLOps Pipeline

An end-to-end MLOps pipeline for MLB player rate projections: data versioning, MLflow experiment tracking and model registry, a champion/challenger promotion gate, a scheduled retrain, drift and performance monitoring, and one-command rollback. It runs on **Databricks Free Edition** (serverless jobs, Unity Catalog, managed MLflow) through a Databricks Asset Bundle, and the same code runs locally and in GitHub Actions with a SQLite MLflow store.

The models come from [mlb-player-projections](https://github.com/Samarateja0118/mlb-player-projections), which compares a hierarchical Bayesian model against Marcel-style baselines in a one-off backtest. This repo is about operating projections over time: retraining as each season lands, deciding when a new model is good enough to replace the current one, and noticing when the world changes under it.

## What runs

```
                       weekly schedule (Databricks job / GitHub Actions)
                                         │
  Lahman data ──► version data ──► same as champion's? ──yes──► skip (most weeks)
                                         │ no: a new season arrived
                                         ▼
                 monitor the champion on the season that just finished
                 (drift vs training window, live skill vs holdout skill)
                                         │
                 train candidates on T-3..T-1, score on T (holdout)
                                         │
                 gate: switch model type only if a paired bootstrap
                 says the challenger is better; must beat league average
                                         │
                 refit winner on T-2..T ─► register version ─► alias `champion`
                                         │                    (old one keeps `previous`)
                 batch-score next season's projections ─► projections table
                                         │
                 check-alerts task fails the job (email) on performance decay or rate drift
```

| Piece | Where |
|---|---|
| Data ingest, as-of filtering, content hash | [`src/mlbops/data.py`](src/mlbops/data.py) |
| Candidate models (Marcel, empirical-Bayes beta-binomial) | [`src/mlbops/models.py`](src/mlbops/models.py) |
| MLflow pyfunc packaging, registry, `champion`/`previous` aliases, rollback | [`src/mlbops/registry.py`](src/mlbops/registry.py) |
| Drift (PSI, league shift, volume shift) and live performance | [`src/mlbops/monitor.py`](src/mlbops/monitor.py) |
| The pipeline step, gate, and replay | [`src/mlbops/pipeline.py`](src/mlbops/pipeline.py) |
| Output tables (Delta on Databricks, Parquet locally) | [`src/mlbops/store.py`](src/mlbops/store.py) |
| Databricks jobs, schema, experiment | [`databricks.yml`](databricks.yml), [`resources/mlb_mlops.yml`](resources/mlb_mlops.yml) |
| CI, weekly pipeline run, bundle deploy | [`.github/workflows/`](.github/workflows) |

### Models

Both candidates project next season's rate as a 5/4/3 weighted average of the last three seasons, shrunk toward the league rate by `m` opportunities of league-average performance:

- **marcel**: `m` fixed at 300 (the classic Marcel regression, rescaled).
- **eb**: `m` fitted per stat each retrain by maximum likelihood under a beta-binomial model, so the amount of regression is learned from the data.

They are deliberately cheap (a full 14-season replay of four stats runs in about a minute), so the pipeline mechanics are the focus. The PyMC model from the source repo is the next challenger to add (see [docs/PLAN.md](docs/PLAN.md)).

### Monitoring

When a season's results arrive, the champion that projected it is scored:

- **Drift (no labels needed):** PSI of qualified players' rates against the three training seasons, relative change in the league rate, and relative change in median playing time.
- **Performance:** skill = champion log loss minus league-average log loss on the same players, compared with the skill it showed on its holdout when promoted. `edge_retained` = live skill / holdout skill (1.0 = kept its whole edge).

Rows go to MLflow (one `monitor-<stat>-<season>` run each) and to the `monitoring` table. `mlbops check-alerts` exits non-zero on the blocking alerts (`performance_decay`, `rate_drift` by default), which fails the Databricks job and sends the failure email.

## Results: replaying 2012-2025

`mlbops replay --start 2012 --end 2025` walks forward one season at a time as if each had just been released: 56 retrains, 56 registered versions, 52 monitoring rows. Full output: [`results/replay_pipeline_runs.csv`](results/replay_pipeline_runs.csv), [`results/replay_monitoring.csv`](results/replay_monitoring.csv).

What the pipeline picked up on its own:

- **The 60-game 2020 season.** Median playing time fell 51% for hitters and 39% for pitchers, and `volume_shift` fired on all four stats. On the 2020 holdout, Marcel beat the empirical-Bayes model on hitter strikeout rate (bootstrap 95% CI [-0.00082, -0.00006]), so the gate switched that model to Marcel. After a full season returned (2021 playing time +89%), the 2021 holdout was still not conclusive, so Marcel stayed. The 2022 holdout favored empirical Bayes (CI [-0.00079, -0.00014]) and the gate switched back.
- **League-wide home run swings.** `league_shift` fired on HR rate every season from 2013 to 2019 and from 2021 to 2024 (for example +17% in 2015, +20% in 2019, -12% in 2022, +11% in 2023). HR rate moves enough year to year that this alert is informational, not blocking.
- **Model choice by stat.** Empirical Bayes held hitter strikeout rate for 12 of 14 seasons. Marcel's fixed regression won walk rate, home run rate, and pitcher strikeout rate, the same split the Bayesian backtest in the source repo found.
- **No performance-decay alert fired.** The lowest `edge_retained` was 0.68 (HR rate, 2022), so the champions always kept most of their edge over league average. The alert threshold is 0.5.

Rows where the gate kept the incumbent are the common case: most challengers were not significantly better, and the gate is designed not to flip models on noise.

## Run it locally

```bash
pip install -e ".[dev]"
pytest -q                                  # 9 tests, including an end-to-end run on a temp MLflow store
mlbops replay --start 2012 --end 2025      # ~1 minute; writes .mlbops/mlflow.db and .mlbops/tables/
mlbops run                                 # a scheduled run: "skip: no new data" until 2026 lands
mlbops report                              # pipeline decisions and monitoring table
mlbops rollback --stat bat_k               # move `champion` back to the previous version
mlflow ui --backend-store-uri sqlite:///.mlbops/mlflow.db
```

## Deploy to Databricks Free Edition

1. Sign up for [Databricks Free Edition](https://www.databricks.com/learn/free-edition) and create a personal access token (Settings → Developer).
2. Install the [Databricks CLI](https://docs.databricks.com/aws/en/dev-tools/cli/install) and [uv](https://docs.astral.sh/uv/), then `databricks configure`.
3. `databricks bundle deploy -t dev`, then `databricks bundle run -t dev mlb_backfill` once to build the model history.
4. The `mlb-retrain-and-monitor` job then runs every Monday at 06:00 UTC.

Models register in Unity Catalog as `workspace.mlb_mlops.mlb_<stat>`. Projections, monitoring rows, and run decisions land in Delta tables in the same schema. To deploy from GitHub instead, add `DATABRICKS_HOST` and `DATABRICKS_TOKEN` as repository secrets; [`deploy.yml`](.github/workflows/deploy.yml) deploys on every push to `main`.

**Status:** the pipeline, tests, and the local replay above have been run. The bundle has not yet been deployed to a live workspace.

## Data

The Lahman Baseball Database through 2025 via [`pylahman`](https://pypi.org/project/pylahman/) (SABR, CC BY-SA 3.0), which ships the data inside the package, so serverless jobs need only PyPI access. Lahman has season totals only, which is why new data arrives once a year and most scheduled runs skip.
