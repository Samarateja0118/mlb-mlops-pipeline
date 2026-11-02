"""Monitoring for a season once its results are in.

Two kinds of signal:

- Data drift: how the new season's player rates and playing time compare with
  the three seasons the model was trained on (PSI of player rates, league rate
  shift, change in median opportunities). No labels needed.
- Performance: how the champion's projections for that season actually scored,
  as "skill" (its log loss minus league average's on the same players), compared
  with the skill it showed on its holdout season when it was promoted.

Example from the replay: the 60-game 2020 season trips the volume alert, and
the 2023 rule changes (pitch clock, shift limits) shift league rates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data, metrics
from .config import Thresholds

MIN_N = 100  # qualified players for drift stats


def psi(reference, current, bins: int = 10) -> float:
    """Population stability index with bins from reference quantiles."""
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(current, edges)[0] / len(current)
    ref, cur = np.clip(ref, 1e-4, None), np.clip(cur, 1e-4, None)
    return float(((cur - ref) * np.log(cur / ref)).sum())


def drift(lines: pd.DataFrame, season: int, window: int = 3) -> dict:
    ref = lines[(lines.yearID >= season - window) & (lines.yearID < season) & (lines.n >= MIN_N)]
    cur = lines[(lines.yearID == season) & (lines.n >= MIN_N)]
    prev = lines[(lines.yearID == season - 1) & (lines.n >= MIN_N)]
    lg_now, lg_prev = data.league_rate(lines, season), data.league_rate(lines, season - 1)
    return {
        "psi_rate": psi(ref.k / ref.n, cur.k / cur.n),
        "league_rate": lg_now,
        "league_shift": lg_now / lg_prev - 1,
        "median_n": float(cur.n.median()),
        "volume_shift": float(cur.n.median() / prev.n.median() - 1),
        "qualified_players": len(cur),
    }


def performance(champion_model, split: data.Split, expected_skill: float) -> dict:
    proj = champion_model.predict(split.history).set_index("playerID").proj
    t = split.target.set_index("playerID")
    p = proj.reindex(t.index).to_numpy()
    s = metrics.score(p, t.k, t.n)
    ll_league = metrics.log_loss(np.full(len(t), split.league_rate), t.k, t.n)
    skill = s["log_loss"] - ll_league
    return {
        **{f"live_{k}": v for k, v in s.items()},
        "live_skill": skill,
        "expected_skill": expected_skill,
        # 1.0 = kept all of its holdout edge over league average, 0 = no better than league
        "edge_retained": skill / expected_skill if expected_skill < 0 else float("nan"),
        "scored_players": len(t),
    }


def alerts(row: dict, th: Thresholds) -> list[str]:
    out = []
    if row.get("psi_rate", 0) > th.psi_alert:
        out.append("rate_drift")
    if abs(row.get("league_shift", 0)) > th.league_shift_alert:
        out.append("league_shift")
    if abs(row.get("volume_shift", 0)) > th.volume_shift_alert:
        out.append("volume_shift")
    er = row.get("edge_retained", float("nan"))
    if not np.isnan(er) and er < th.edge_retained_alert:
        out.append("performance_decay")
    return out
