"""Command line entry point: python -m bessrank.run {data,qa,smoke,features,validate,test,explore,report}."""
import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata

import pandas as pd

from bessrank import config

PACKAGES = ["pandas", "numpy", "pyarrow", "requests", "holidays", "scipy", "ortools", "xgboost",
            "scikit-learn", "torch", "matplotlib", "mlflow", "pytest"]


def git_commit():
    """Current commit hash, with '-dirty' if there are uncommitted changes."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True, text=True).stdout.strip()
    commit = git("rev-parse", "HEAD")
    return commit + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")


def package_versions():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def update_provenance(section, info):
    """Record a stage's details in results/provenance.json with versions and the git commit."""
    path = config.PROVENANCE_JSON
    prov = json.loads(path.read_text()) if path.exists() else {}
    prov[section] = {
        **info,
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "packages": package_versions(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prov, indent=2) + "\n")


def cmd_data(args):
    from bessrank import data
    meta = data.build_data(refresh=args.refresh)
    keep = ["source", "smard_download_utc_first", "smard_download_utc_last", "first_day", "last_day", "rows"]
    update_provenance("data", {k: meta[k] for k in keep})


def cmd_qa(args):
    from bessrank import data
    data.write_qa_report()


def cmd_features(args):
    """Build the feature table (train + validation; the test period stays locked) and record
    the missing-value counts of PLAN.md §2."""
    from bessrank import data, features
    feats = features.build_features(data.load_hourly())
    feats.to_parquet(config.FEATURES_PARQUET, index=False)

    dropped = features.incomplete_days(feats, config.TRAIN_START, config.TRAIN_END)
    val = feats[(feats["delivery_day"] >= config.VAL_START) & (feats["delivery_day"] <= config.VAL_END)]
    val_nan = val[features.FEATURE_COLUMNS].isna()
    summary = {
        "rows": int(len(feats)),
        "features": len(features.FEATURE_COLUMNS),
        "features_from_tso_generation_forecasts": len(features.GEN_FORECAST_FEATURES),
        "train_start": str(config.TRAIN_START),
        "days_removed_by_train_start": (config.TRAIN_START - config.DATA_START).days,
        "train_days": len(pd.date_range(config.TRAIN_START, config.TRAIN_END, freq="D")),
        "train_days_dropped": len(dropped),
        "train_days_dropped_list": [str(d) for d in dropped],
        "validation_days_with_missing_features": int(val.loc[val_nan.any(axis=1), "delivery_day"].nunique()),
        "validation_missing_feature_values": int(val_nan.sum().sum()),
    }
    out = config.RESULTS_DIR / "features_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    update_provenance("features", {k: summary[k] for k in ["rows", "features", "train_start", "train_days_dropped"]})
    print(f"Built {len(feats)} rows x {len(features.FEATURE_COLUMNS)} features; training days dropped: "
          f"{len(dropped)}; validation days with a missing feature: {summary['validation_days_with_missing_features']}.")


def cmd_validate(args):
    """Validation year: S-perfect and the naive strategies (S1). Models are added in S2."""
    from bessrank import data, evaluate, strategies
    vectors = strategies.baseline_price_vectors(data.load_hourly())
    val = vectors[(vectors["delivery_day"] >= config.VAL_START) & (vectors["delivery_day"] <= config.VAL_END)]
    names = ["S-perfect", "S-naive-1d", "S-naive-7d"]
    # Every day is solved by HiGHS and re-checked by SCIP; perfect foresight must be the best.
    daily = evaluate.daily_profits(val, names)
    table = pd.DataFrame([{**evaluate.value_summary(daily, s), **evaluate.forecast_summary(val, s)} for s in names])
    table.to_csv(config.RESULTS_DIR / "validation_strategies.csv", index=False)
    daily.to_csv(config.RESULTS_DIR / "validation_daily_profit.csv", index=False)
    update_provenance("validate", {"strategies": names, "days": int(len(daily)),
                                   "solver_cross_checks": int(len(daily) * len(names))})
    show = table.set_index("strategy")[["profit_eur_per_mw_year", "capture_rate", "var5_eur_per_day",
                                        "es5_eur_per_day", "losing_day_share", "rmse_eur_mwh", "spearman_rho_mean"]]
    print(show.round(3).to_string())  # rounded for display only


def cmd_smoke(args):
    from bessrank import databricks
    ok = databricks.smoke_test()
    sys.exit(0 if ok else 1)


def not_yet(args):
    sys.exit(f"`{args.command}` is not implemented yet (PLAN.md §11, later sessions).")


def main():
    parser = argparse.ArgumentParser(prog="python -m bessrank.run")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("data", help="rebuild data/ from the Databricks volume or SMARD")
    p.add_argument("--refresh", action="store_true", help="skip the volume and re-download from SMARD")
    p.set_defaults(func=cmd_data)
    sub.add_parser("qa", help="write results/qa_data.md").set_defaults(func=cmd_qa)
    sub.add_parser("smoke", help="Databricks smoke test").set_defaults(func=cmd_smoke)
    sub.add_parser("features", help="build the feature table").set_defaults(func=cmd_features)
    sub.add_parser("validate", help="validation-year strategies").set_defaults(func=cmd_validate)
    for name in ["test", "explore", "report"]:
        sub.add_parser(name).set_defaults(func=not_yet)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
