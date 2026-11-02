"""Command line entry point: `mlbops run | replay | rollback | report`."""
from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd

from . import config, data, pipeline, registry, store


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="mlbops")
    ap.add_argument("--experiment", help="MLflow experiment path (env MLBOPS_EXPERIMENT)")
    ap.add_argument("--catalog", help="Unity Catalog catalog on Databricks (env MLBOPS_CATALOG)")
    ap.add_argument("--schema", help="Unity Catalog schema on Databricks (env MLBOPS_SCHEMA)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="one scheduled pipeline run")
    r.add_argument("--as-of", type=int, help="last finished season (default: latest in the data)")
    r.add_argument("--stat", action="append", choices=list(data.STATS))
    r.add_argument("--force", action="store_true", help="retrain even if the data has not changed")
    rp = sub.add_parser("replay", help="walk forward through past seasons")
    rp.add_argument("--start", type=int, default=2012)
    rp.add_argument("--end", type=int, default=2025)
    rp.add_argument("--stat", action="append", choices=list(data.STATS))
    rb = sub.add_parser("rollback", help="move champion back to the previous version")
    rb.add_argument("--stat", required=True, choices=list(data.STATS))
    sub.add_parser("report", help="print the pipeline_runs and monitoring tables")
    ca = sub.add_parser("check-alerts", help="exit 1 if the latest monitoring rows have blocking alerts")
    ca.add_argument("--blocking", default="performance_decay,rate_drift")
    a = ap.parse_args(argv)
    for flag, env in (("experiment", "MLBOPS_EXPERIMENT"), ("catalog", "MLBOPS_CATALOG"), ("schema", "MLBOPS_SCHEMA")):
        if getattr(a, flag):
            os.environ[env] = getattr(a, flag)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("mlflow", "alembic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if a.cmd == "run":
        pipeline.run(a.as_of, a.stat, a.force)
    elif a.cmd == "replay":
        pipeline.replay(a.start, a.end, a.stat)
    elif a.cmd == "rollback":
        s = config.load()
        pipeline.setup(s)
        print(f"champion is now version {registry.rollback(s.model_name(a.stat))}")
    elif a.cmd == "report":
        s = config.load()
        pd.set_option("display.width", 200)
        pd.set_option("display.max_columns", 20)
        runs = store.read(s, "pipeline_runs")
        mon = store.read(s, "monitoring")
        if not runs.empty:
            print(runs[["stat", "as_of", "action", "kind", "champion_after", "alerts", "reason"]].to_string(index=False))
        if not mon.empty:
            cols = ["stat", "season", "champion_kind", "psi_rate", "league_shift", "volume_shift",
                    "live_skill", "expected_skill", "edge_retained", "alerts"]
            print(mon[cols].round(4).to_string(index=False))
    elif a.cmd == "check-alerts":
        mon = store.read(config.load(), "monitoring")
        if mon.empty:
            print("no monitoring rows yet")
            return
        latest = mon[mon.season == mon.season.max()]
        blocking = set(a.blocking.split(","))
        hits = [(r.stat, r.season, r.alerts) for r in latest.itertuples()
                if blocking & set(str(r.alerts).split(","))]
        for h in hits:
            print("ALERT", *h)
        if hits:
            sys.exit(1)
        print(f"no blocking alerts for season {latest.season.max()}")


if __name__ == "__main__":
    main()
