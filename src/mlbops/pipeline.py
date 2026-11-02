"""The pipeline step, run once per stat each time it is scheduled.

For data available through season T:

1. Version the data. If the champion was trained on this exact data, stop:
   nothing new has arrived (most scheduled runs end here, since MLB data lands
   once a year).
2. Monitor. If the champion projected season T, score it now that T is in,
   and compute drift of T against the training window.
3. Evaluate candidates on a holdout: each is trained on T-3..T-1 and scored on T.
4. Gate. The best candidate replaces the champion's model type only if a
   paired bootstrap says it is better (95% interval below zero). Whatever wins
   must beat league average on the holdout, or nothing is promoted.
5. Refit the winner on T-2..T, register it, move the `champion` alias.
6. Batch-score projections for T+1 with the new champion.

Every step is logged to MLflow, and each run's decision goes to the
pipeline_runs table.
"""
from __future__ import annotations

import datetime as dt
import logging

import mlflow
import numpy as np
import pandas as pd

from . import config, data, metrics, monitor, registry, store
from .models import CANDIDATES, train

log = logging.getLogger("mlbops")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def evaluate_candidates(lines: pd.DataFrame, stat: str, holdout_year: int) -> tuple[dict, pd.DataFrame]:
    """Train each candidate on the seasons before holdout_year and score it on holdout_year."""
    sp = data.split(lines, holdout_year)
    t = sp.target.set_index("playerID")
    results, losses = {}, {}
    for kind in CANDIDATES:
        m = train(kind, stat, sp.history, sp.league_rate)
        p = m.predict(sp.history).set_index("playerID").proj.reindex(t.index).to_numpy()
        results[kind] = {**metrics.score(p, t.k, t.n), "m": m.m}
        losses[kind] = metrics.per_player_log_loss(p, t.k, t.n)
    ll_league = metrics.log_loss(np.full(len(t), sp.league_rate), t.k, t.n)
    for kind in CANDIDATES:
        results[kind]["skill"] = results[kind]["log_loss"] - ll_league
    loss_df = pd.DataFrame(losses).assign(n=t.n.to_numpy())
    return results, loss_df


def gate(results: dict, losses: pd.DataFrame, incumbent: str | None) -> tuple[str | None, str, dict]:
    """Pick the model type to promote. Returns (kind or None, reason, bootstrap stats)."""
    best = min(results, key=lambda k: results[k]["log_loss"])
    boot = {}
    if incumbent is None:
        choice, reason = best, "no champion yet"
    elif best == incumbent:
        choice, reason = incumbent, "incumbent still best"
    else:
        d, lo, hi = metrics.paired_bootstrap(losses[best], losses[incumbent], losses.n)
        boot = {"boot_diff": d, "boot_lo": lo, "boot_hi": hi}
        if hi < 0:
            choice, reason = best, f"{best} beats {incumbent}, 95% CI [{lo:.5f}, {hi:.5f}]"
        else:
            choice, reason = incumbent, f"{best} not significantly better than {incumbent}"
    if results[choice]["skill"] >= 0:
        return None, f"{choice} does not beat league average on holdout; not promoted", boot
    return choice, reason, boot


def run_stat(settings: config.Settings, stat: str, lines_all: pd.DataFrame, season: int,
             force: bool = False) -> dict:
    lines = data.as_of(lines_all, season)
    version = data.data_version(lines)
    name = settings.model_name(stat)
    champ = registry.champion(name)
    row = {"run_at": _now(), "stat": stat, "as_of": season, "data_version": version,
           "champion_before": champ.version if champ else None}

    if champ and champ.tags.get("data_version") == version and not force:
        store.append(settings, "pipeline_runs", pd.DataFrame([{**row, "action": "skip",
                                                               "reason": "no new data"}]))
        return {**row, "action": "skip", "reason": "no new data"}

    alert_list: list[str] = []
    if champ and champ.trained_through == season - 1:
        with mlflow.start_run(run_name=f"monitor-{stat}-{season}"):
            mon = {**monitor.drift(lines, season),
                   **monitor.performance(registry.load(name), data.split(lines, season), champ.holdout_skill)}
            alert_list = monitor.alerts(mon, settings.thresholds)
            mlflow.set_tags({"stage": "monitor", "stat": stat, "season": season,
                             "champion_version": champ.version, "alerts": ",".join(alert_list) or "none"})
            mlflow.log_metrics({k: v for k, v in mon.items() if not pd.isna(v)})
        store.append(settings, "monitoring", pd.DataFrame([{
            "run_at": row["run_at"], "stat": stat, "season": season, "champion_version": champ.version,
            "champion_kind": champ.kind, **mon, "alerts": ",".join(alert_list)}]))

    with mlflow.start_run(run_name=f"train-{stat}-{season}"):
        results, losses = evaluate_candidates(lines, stat, season)
        kind, reason, boot = gate(results, losses, champ.kind if champ else None)
        mlflow.set_tags({"stage": "train", "stat": stat, "as_of": season, "data_version": version,
                         "decision": kind or "none", "reason": reason,
                         "trigger_alerts": ",".join(alert_list) or "none"})
        for k, r in results.items():
            mlflow.log_metrics({f"holdout_{k}_{m}": v for m, v in r.items()})
        if boot:
            mlflow.log_metrics(boot)
        if kind is None:
            out = {**row, "action": "hold", "reason": reason, "alerts": ",".join(alert_list)}
            store.append(settings, "pipeline_runs", pd.DataFrame([out]))
            return out

        hist = data.history_window(lines, season + 1)
        model = train(kind, stat, hist, data.league_rate(lines, season))
        mlflow.log_params({"kind": kind, "m": round(model.m, 2), "league_rate": round(model.league_rate, 5),
                           "trained_through": season, "history_seasons": f"{season - 2}-{season}",
                           "players": hist.playerID.nunique()})
        uri = registry.log_model(model, hist)
        version_id = registry.register(name, uri, {
            "kind": kind, "trained_through": season, "data_version": version,
            "holdout_season": season, "holdout_log_loss": results[kind]["log_loss"],
            "holdout_skill": results[kind]["skill"], "promotion_reason": reason})
        registry.promote(name, version_id)

    proj = registry.load(name).predict(hist).assign(
        stat=stat, model_version=version_id, model_kind=kind, scored_at=row["run_at"])
    store.append(settings, "projections", proj)
    out = {**row, "action": "promote", "reason": reason, "champion_after": version_id,
           "kind": kind, "alerts": ",".join(alert_list), "projected_players": len(proj)}
    store.append(settings, "pipeline_runs", pd.DataFrame([out]))
    return out


def setup(settings: config.Settings) -> None:
    mlflow.set_tracking_uri(settings.tracking_uri)
    mlflow.set_registry_uri(settings.registry_uri)
    mlflow.set_experiment(settings.experiment)


def run(season: int | None = None, stats=None, force: bool = False, lines: dict | None = None) -> list[dict]:
    settings = config.load()
    setup(settings)
    lines = lines or data.load_lahman()
    stats = stats or list(data.STATS)
    out = []
    for stat in stats:
        s = season or int(lines[stat].yearID.max())
        res = run_stat(settings, stat, lines[stat], s, force)
        log.info("%s as_of=%s %s: %s", stat, s, res["action"], res["reason"])
        out.append(res)
    return out


def replay(start: int, end: int, stats=None) -> list[dict]:
    """Walk forward one season at a time, as if each season's data had just arrived."""
    lines = data.load_lahman()
    out = []
    for season in range(start, end + 1):
        out += run(season, stats, lines=lines)
    return out
