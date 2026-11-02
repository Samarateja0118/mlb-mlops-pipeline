"""Candidate projection models.

Both candidates share one form: a weighted average of a player's last three
seasons, shrunk toward the league rate by `m` opportunities of league-average
performance,

    proj = (sum_s w_s k_s / w_bar + m * league) / (sum_s w_s n_s / w_bar + m)

with weights 5/4/3 (most recent first) and w_bar their mean. They differ in how
`m` is chosen:

- marcel: m is fixed at 300 (the classic Marcel 1,200 against weights that sum
  to 12, rescaled to weights that average 1).
- eb: m is fitted per stat by maximum likelihood under a beta-binomial model
  of the three-season totals (empirical Bayes). This learns how much a stat
  needs to be regressed, which the fixed Marcel constant cannot.

Fitted models are plain parameter sets, wrapped as MLflow pyfunc models in
`registry.py` so they load the same way locally and on Databricks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats

WEIGHTS = (5, 4, 3)
MARCEL_M = 300.0
CANDIDATES = ("marcel", "eb")


@dataclass(frozen=True)
class RateModel:
    kind: str
    stat: str
    m: float  # shrinkage strength, in opportunities
    league_rate: float  # rate the projection is shrunk toward
    trained_through: int  # last season in the training data
    weights: tuple = WEIGHTS

    def params(self) -> dict:
        d = asdict(self)
        d["weights"] = list(self.weights)
        return d

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        """Project the season after the last one in `history` for every player in it."""
        last = int(history.yearID.max())
        w = pd.Series({last - i: wt for i, wt in enumerate(self.weights)}) / np.mean(self.weights)
        h = history.assign(w=history.yearID.map(w).fillna(0.0))
        agg = h.assign(wk=h.w * h.k, wn=h.w * h.n).groupby("playerID")[["wk", "wn"]].sum()
        proj = (agg.wk + self.m * self.league_rate) / (agg.wn + self.m)
        return pd.DataFrame({"playerID": proj.index, "target_year": last + 1, "proj": proj.values})


def fit_beta_binomial(k: np.ndarray, n: np.ndarray) -> tuple[float, float]:
    """MLE of (mean, strength) of a beta prior over player rates. strength = alpha + beta."""
    k, n = np.asarray(k), np.asarray(n)
    mu0 = k.sum() / n.sum()

    def nll(x):
        mu, m = 1 / (1 + np.exp(-x[0])), np.exp(x[1])
        return -stats.betabinom.logpmf(k, n, mu * m, (1 - mu) * m).sum()

    res = optimize.minimize(nll, x0=[np.log(mu0 / (1 - mu0)), np.log(300.0)], method="Nelder-Mead",
                            options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000})
    return float(1 / (1 + np.exp(-res.x[0]))), float(np.exp(res.x[1]))


def train(kind: str, stat: str, history: pd.DataFrame, league_rate: float) -> RateModel:
    """Fit a candidate on a three-season history window. `league_rate` is the last season's rate."""
    last = int(history.yearID.max())
    if kind == "marcel":
        return RateModel("marcel", stat, MARCEL_M, league_rate, last)
    if kind == "eb":
        tot = history.groupby("playerID")[["k", "n"]].sum()
        _, m = fit_beta_binomial(tot.k.to_numpy(), tot.n.to_numpy())
        # The fit is on three-season totals; projection counts are rescaled to weights that average 1,
        # so a player's effective sample is comparable. Keep m on that scale.
        return RateModel("eb", stat, m, league_rate, last)
    raise ValueError(f"unknown model kind {kind!r}; expected one of {CANDIDATES}")
