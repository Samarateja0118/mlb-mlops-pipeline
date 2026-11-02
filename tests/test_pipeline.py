"""Unit tests on synthetic data plus one end-to-end run against a throwaway MLflow store."""
import numpy as np
import pandas as pd
import pytest

from mlbops import data, models, monitor, pipeline, registry


def synthetic_lines(seasons=range(2010, 2016), players=300, seed=0, shift=None):
    """Players with true rates from Beta(20, 80); `shift` = {season: multiplier} to fake drift."""
    rng = np.random.default_rng(seed)
    true = rng.beta(20, 80, players)
    rows = []
    for y in seasons:
        mult = (shift or {}).get(y, 1.0)
        n = rng.integers(150, 650, players)
        k = rng.binomial(n, np.clip(true * mult, 0, 1))
        rows += [{"playerID": f"p{i}", "yearID": y, "age": 25 + (y - 2010), "k": int(k[i]), "n": int(n[i])}
                 for i in range(players)]
    return pd.DataFrame(rows)


def test_as_of_drops_future_seasons():
    lines = synthetic_lines()
    assert data.as_of(lines, 2012).yearID.max() == 2012


def test_data_version_changes_only_with_content():
    lines = synthetic_lines()
    assert data.data_version(lines) == data.data_version(lines.sample(frac=1, random_state=1))
    assert data.data_version(lines) != data.data_version(data.as_of(lines, 2014))


def test_split_has_no_target_season_in_history():
    sp = data.split(synthetic_lines(), 2014)
    assert sp.history.yearID.max() == 2013 and set(sp.target.yearID) == {2014}


def test_marcel_matches_closed_form():
    h = pd.DataFrame({"playerID": ["a"] * 3, "yearID": [2011, 2012, 2013], "age": [25, 26, 27],
                      "k": [10, 20, 30], "n": [100, 100, 100]})
    m = models.RateModel("marcel", "bat_k", 300.0, 0.2, 2013)
    w = np.array([3, 4, 5]) / 4
    expect = ((w * [10, 20, 30]).sum() + 300 * 0.2) / ((w * 100).sum() + 300)
    assert m.predict(h).proj.iloc[0] == pytest.approx(expect)
    assert m.predict(h).target_year.iloc[0] == 2014


def test_beta_binomial_recovers_prior_strength():
    rng = np.random.default_rng(1)
    p = rng.beta(30, 70, 4000)  # strength 100
    n = rng.integers(300, 1500, 4000)
    mu, strength = models.fit_beta_binomial(rng.binomial(n, p), n)
    assert mu == pytest.approx(0.3, abs=0.01)
    assert strength == pytest.approx(100, rel=0.2)


def test_psi_flags_shift_and_not_noise():
    rng = np.random.default_rng(0)
    ref = rng.normal(0.2, 0.05, 5000)
    assert monitor.psi(ref, rng.normal(0.2, 0.05, 5000)) < 0.05
    assert monitor.psi(ref, rng.normal(0.26, 0.05, 5000)) > 0.2


def test_gate_keeps_incumbent_without_significant_win():
    rng = np.random.default_rng(0)
    base = rng.uniform(100, 200, 500)
    losses = pd.DataFrame({"marcel": base, "eb": base - rng.normal(0, 1, 500), "n": 400})
    res = {"marcel": {"log_loss": 0.5001, "skill": -0.001}, "eb": {"log_loss": 0.5000, "skill": -0.0011}}
    kind, reason, _ = pipeline.gate(res, losses, "marcel")
    assert kind == "marcel" and "not significantly" in reason


def test_gate_switches_on_clear_win_and_refuses_no_skill():
    base = np.full(500, 150.0)
    losses = pd.DataFrame({"marcel": base, "eb": base - 1.0, "n": 400})
    res = {"marcel": {"log_loss": 0.51, "skill": -0.001}, "eb": {"log_loss": 0.50, "skill": -0.002}}
    assert pipeline.gate(res, losses, "marcel")[0] == "eb"
    res_bad = {"marcel": {"log_loss": 0.51, "skill": 0.001}, "eb": {"log_loss": 0.52, "skill": 0.002}}
    assert pipeline.gate(res_bad, losses, None)[0] is None


def test_end_to_end_promote_skip_monitor_rollback(tmp_path, monkeypatch):
    monkeypatch.setenv("MLBOPS_HOME", str(tmp_path))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    lines = {"bat_k": synthetic_lines(shift={2015: 1.3})}

    first = pipeline.run(2013, ["bat_k"], lines=lines)[0]
    assert first["action"] == "promote" and first["champion_before"] is None

    again = pipeline.run(2013, ["bat_k"], lines=lines)[0]
    assert again["action"] == "skip"

    pipeline.run(2014, ["bat_k"], lines=lines)
    drifted = pipeline.run(2015, ["bat_k"], lines=lines)[0]
    assert "rate_drift" in drifted["alerts"] and "league_shift" in drifted["alerts"]

    from mlbops import config, store
    s = config.load()
    mon = store.read(s, "monitoring")
    assert list(mon.season) == [2014, 2015]
    proj = store.read(s, "projections")
    assert set(proj.target_year) == {2014, 2015, 2016}

    name = s.model_name("bat_k")
    before = registry.champion(name).version
    registry.rollback(name)
    assert registry.champion(name).version != before
    assert len(registry.load(name).predict(data.history_window(lines["bat_k"], 2016))) > 0
