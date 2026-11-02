"""Ingest Lahman season lines and shape them into per-player, per-season rate data.

Adapted from mlb-player-projections (projections/data.py). Rates are per plate
appearance (hitters) or per batter faced (pitchers). Age is age on July 1.

The pipeline treats data as arriving one season at a time: `as_of(lines, T)`
returns only what was known after season T ended, so a replay over past
seasons never sees the future.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd

# stat name -> (table, event column, opportunity column)
STATS = {
    "bat_k": ("batting", "SO", "PA"),
    "bat_bb": ("batting", "BB", "PA"),
    "bat_hr": ("batting", "HR", "PA"),
    "pit_k": ("pitching", "SO", "BFP"),
}
LINE_COLUMNS = ["playerID", "yearID", "age", "k", "n"]


def age_on_july_1(birth_year: pd.Series, birth_month: pd.Series, season: pd.Series) -> pd.Series:
    return season - birth_year - (birth_month >= 7).astype(int)


def batting_lines(batting: pd.DataFrame) -> pd.DataFrame:
    cols = ["AB", "BB", "HBP", "SH", "SF", "SO", "HR"]
    df = batting.groupby(["playerID", "yearID"], as_index=False)[cols].sum(min_count=1).fillna(0)
    df["PA"] = df["AB"] + df["BB"] + df["HBP"] + df["SH"] + df["SF"]
    return df


def pitching_lines(pitching: pd.DataFrame) -> pd.DataFrame:
    return pitching.groupby(["playerID", "yearID"], as_index=False)[["SO", "BB", "HR", "BFP"]].sum(
        min_count=1
    ).fillna(0)


def build(stat: str, batting: pd.DataFrame, pitching: pd.DataFrame, people: pd.DataFrame) -> pd.DataFrame:
    """Season lines for one stat: columns playerID, yearID, age, k, n."""
    table, event, opp = STATS[stat]
    lines = batting_lines(batting) if table == "batting" else pitching_lines(pitching)
    lines = lines[lines[opp] > 0]
    ppl = people[["playerID", "birthYear", "birthMonth"]].dropna()
    lines = lines.merge(ppl, on="playerID", how="inner")
    lines["age"] = age_on_july_1(lines["birthYear"].astype(int), lines["birthMonth"].astype(int), lines["yearID"])
    out = lines.rename(columns={event: "k", opp: "n"})[LINE_COLUMNS]
    return out.astype({"k": int, "n": int, "age": int, "yearID": int}).reset_index(drop=True)


def load_lahman() -> dict[str, pd.DataFrame]:
    """All four stats from the Lahman database (pylahman bundles the data)."""
    import pylahman

    batting, pitching, people = pylahman.Batting(), pylahman.Pitching(), pylahman.People()
    return {s: build(s, batting, pitching, people) for s in STATS}


def as_of(lines: pd.DataFrame, season: int) -> pd.DataFrame:
    """Only the seasons that had finished by the end of `season`."""
    return lines[lines.yearID <= season].reset_index(drop=True)


def data_version(lines: pd.DataFrame) -> str:
    """Content hash of the season lines. A retrain is skipped when it hasn't changed."""
    canon = lines.sort_values(["playerID", "yearID"])[LINE_COLUMNS].to_csv(index=False)
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Split:
    """History seasons (strictly before target) and the target-season outcomes to score."""

    target_year: int
    history: pd.DataFrame
    target: pd.DataFrame
    league_rate: float


def history_window(lines: pd.DataFrame, target_year: int, years: int = 3, min_hist: int = 100) -> pd.DataFrame:
    hist = lines[(lines.yearID >= target_year - years) & (lines.yearID < target_year)]
    tot = hist.groupby("playerID")["n"].sum()
    return hist[hist.playerID.isin(tot[tot >= min_hist].index)].reset_index(drop=True)


def league_rate(lines: pd.DataFrame, season: int) -> float:
    s = lines[lines.yearID == season]
    return float(s.k.sum() / s.n.sum())


def split(lines: pd.DataFrame, target_year: int, years: int = 3, min_hist: int = 100,
          min_target: int = 100) -> Split:
    """Players with >= min_hist opportunities in the `years` seasons before target_year and
    >= min_target in target_year. Nothing from target_year enters history."""
    hist = history_window(lines, target_year, years, min_hist)
    tgt = lines[(lines.yearID == target_year) & (lines.n >= min_target) & lines.playerID.isin(set(hist.playerID))]
    hist = hist[hist.playerID.isin(set(tgt.playerID))].reset_index(drop=True)
    return Split(target_year, hist, tgt.reset_index(drop=True), league_rate(lines, target_year - 1))
