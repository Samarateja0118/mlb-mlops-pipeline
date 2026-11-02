"""Scoring for rate projections against realized counts.

Adapted from mlb-player-projections (projections/metrics.py). Each plate
appearance (or batter faced) is a binary event, so a projected rate p for a
player with k events in n chances is scored on all n outcomes.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-9


def log_loss(p, k, n) -> float:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    k, n = np.asarray(k, float), np.asarray(n, float)
    return float(-(k * np.log(p) + (n - k) * np.log1p(-p)).sum() / n.sum())


def brier(p, k, n) -> float:
    p, k, n = (np.asarray(x, float) for x in (p, k, n))
    return float((k * (1 - p) ** 2 + (n - k) * p**2).sum() / n.sum())


def weighted_rmse(p, k, n) -> float:
    """RMSE of projected vs realized rate, weighted by opportunities."""
    p, k, n = (np.asarray(x, float) for x in (p, k, n))
    return float(np.sqrt((n * (p - k / n) ** 2).sum() / n.sum()))


def per_player_log_loss(p, k, n) -> np.ndarray:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    k, n = np.asarray(k, float), np.asarray(n, float)
    return -(k * np.log(p) + (n - k) * np.log1p(-p))


def paired_bootstrap(loss_a, loss_b, n_opp, reps: int = 2000, seed: int = 0):
    """Bootstrap players to get a 95% interval for mean per-chance loss(a) - loss(b).
    Negative means a is better."""
    rng = np.random.default_rng(seed)
    loss_a, loss_b, n_opp = (np.asarray(x, float) for x in (loss_a, loss_b, n_opp))
    idx = rng.integers(0, len(loss_a), size=(reps, len(loss_a)))
    diffs = (loss_a[idx].sum(1) - loss_b[idx].sum(1)) / n_opp[idx].sum(1)
    point = (loss_a.sum() - loss_b.sum()) / n_opp.sum()
    return float(point), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def score(p, k, n) -> dict:
    return {"log_loss": log_loss(p, k, n), "brier": brier(p, k, n), "rmse": weighted_rmse(p, k, n)}
