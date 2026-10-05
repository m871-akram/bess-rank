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


def log(message):
    print(f"{datetime.now(timezone.utc):%H:%M:%S} {message}", flush=True)


def load_features():
    """Feature table of train + validation (cmd_features never includes the test period)."""
    feats = pd.read_parquet(config.FEATURES_PARQUET)
    config.check_days_allowed(feats["delivery_day"].unique())
    return feats


def cmd_tune(args):
    """Random search of one model on the validation year (PLAN.md §7). Writes the table of
    every fit to results/ and the validation forecasts to data/predictions/."""
    from bessrank import models
    if args.model.startswith("xgb"):
        configs, fit, n_finalists = models.sample_configs(), models.fit_predict, 3
    else:
        from bessrank import lstm
        configs, fit, n_finalists = lstm.CONFIGS, lstm.fit_predict, 1
    table, preds, selected = models.search(args.model, load_features(), configs, fit, n_finalists, log=log)

    table.to_csv(config.RESULTS_DIR / f"tuning_{args.model}.csv", index=False)
    (config.RESULTS_DIR / f"selected_{args.model}.json").write_text(json.dumps(selected, indent=2) + "\n")
    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(config.PREDICTIONS_DIR / f"{args.model}.parquet", index=False)
    update_provenance(f"tune_{args.model}", {"fits": int((table["seed"] != "avg").sum()),
                                             "selected_config": selected["config"],
                                             "threads": models.N_THREADS})


MODELS = ["xgb-reg", "xgb-rank", "lstm-reg", "lstm-rank"]
STRATEGIES = ["S-perfect", "S-naive-1d", "S-naive-7d", "S-xgb-reg", "S-xgb-rank", "S-lstm-reg", "S-lstm-rank"]
PAIRS = [("S-xgb-rank", "S-xgb-reg"), ("S-lstm-rank", "S-lstm-reg")]
# Display labels: VaR and ES are profit levels, negative = loss (Akram, 2026-10-05).
DISPLAY = {
    "profit_eur_per_mw_year": "Profit (EUR/MW/year)", "capture_rate": "Capture",
    "rmse_eur_mwh": "RMSE (EUR/MWh)", "mae_eur_mwh": "MAE (EUR/MWh)", "spearman_rho_mean": "Spearman rho",
    "hit_rate_cheapest2": "Hit 2 cheapest", "hit_rate_dearest2": "Hit 2 dearest",
    "var5_p5_of_daily_profit_eur": "VaR 5%: P5 of daily profit (EUR)",
    "es5_mean_of_worst5pct_days_eur": "ES 5%: mean of the worst 5% of days (EUR)",
    "losing_day_share": "Losing days", "max_drawdown_eur": "Max drawdown (EUR)",
}


def _profits_worker(job):
    from bessrank import evaluate
    vectors, names = job
    return evaluate.daily_profits(vectors, names)


def parallel_daily_profits(vectors, names, workers=4):
    """evaluate.daily_profits for many price vectors, split over processes. Every day is still
    solved by HiGHS and re-checked by SCIP (CLAUDE.md rule 7)."""
    from concurrent.futures import ProcessPoolExecutor
    chunks = [names[i::workers] for i in range(workers) if names[i::workers]]
    keys = ["ts_utc", "delivery_day", "price"]
    jobs = [(vectors[keys + chunk], chunk) for chunk in chunks]
    with ProcessPoolExecutor(len(jobs)) as pool:
        parts = list(pool.map(_profits_worker, jobs))
    daily = parts[0]
    for part in parts[1:]:
        daily = daily.merge(part, on=["delivery_day", "n_hours"])
    return daily[["delivery_day", "n_hours"] + names]


def validation_vectors():
    """Hourly price vectors of the 7 strategies and of every fitted configuration on the
    validation year. Config columns are named "<model>:<fit>", e.g. "xgb-rank:c03_s0"."""
    from bessrank import data, strategies
    vectors = strategies.baseline_price_vectors(data.load_hourly())
    vectors = vectors[(vectors["delivery_day"] >= config.VAL_START)
                      & (vectors["delivery_day"] <= config.VAL_END)].reset_index(drop=True)
    preds = {m: pd.read_parquet(config.PREDICTIONS_DIR / f"{m}.parquet") for m in MODELS}
    for m, frame in preds.items():
        if not frame["ts_utc"].reset_index(drop=True).equals(vectors["ts_utc"]):
            raise ValueError(f"{m} predictions are not aligned with the validation hours")
        fits = [c for c in frame.columns if c.startswith("c")]
        vectors = pd.concat([vectors, frame[fits + ["final"]].add_prefix(f"{m}:")], axis=1)

    vectors["S-xgb-reg"] = vectors["xgb-reg:final"]
    vectors["S-lstm-reg"] = vectors["lstm-reg:final"]
    # "-rank": the price model's values for the day, reassigned in the ranker's order (§5).
    for family in ["xgb", "lstm"]:
        vectors[f"S-{family}-rank"] = strategies.rank_reassign_frame(
            vectors, f"{family}-reg:final", f"{family}-rank:final")
        for col in [c for c in vectors.columns if c.startswith(f"{family}-rank:c")]:
            vectors[col] = strategies.rank_reassign_frame(vectors, f"{family}-reg:final", col)
    return vectors


def config_table(vectors, daily):
    """Every fitted configuration (PLAN.md §7 output, used in §8): RMSE, Spearman rho and
    validation profit. A ranker's row uses its "-rank" vector (the selected price model's
    values in the ranker's order), so its RMSE is that vector's RMSE."""
    from bessrank import evaluate
    rows = []
    for m in MODELS:
        tuning = pd.read_csv(config.RESULTS_DIR / f"tuning_{m}.csv", dtype={"seed": str})
        for _, fit in tuning.iterrows():
            col = f"{m}:c{int(fit['config']):02d}_{'avg' if fit['seed'] == 'avg' else 's' + fit['seed']}"
            f = evaluate.forecast_summary(vectors, col)
            rows.append({**fit.to_dict(), "vector_rmse_eur_mwh": f["rmse_eur_mwh"],
                         "vector_mae_eur_mwh": f["mae_eur_mwh"],
                         "vector_spearman_rho_mean": f["spearman_rho_mean"],
                         "profit_eur_per_mw_year": float(daily[col].mean() * 365 / config.POWER_MW),
                         "capture_rate": float(daily[col].sum() / daily["S-perfect"].sum())})
    return pd.DataFrame(rows)


def paired_differences(vectors, daily):
    """Exploratory on validation: daily profit and RMSE differences, rank minus reg, with
    moving-block bootstrap CIs (the statistics of H1-H3, PLAN.md §6)."""
    from bessrank import evaluate
    rows = []
    sse = {s: ((vectors[s] - vectors["price"]) ** 2).groupby(vectors["delivery_day"]).sum() for s in STRATEGIES}
    for a, b in PAIRS:
        diff = daily[a] - daily[b]
        mean, lo, hi = evaluate.block_bootstrap_ci(diff)
        rmse, rlo, rhi = evaluate.rmse_difference_ci(sse[a].to_numpy(), sse[b].to_numpy(), daily["n_hours"].to_numpy())
        same_vector = (vectors[a] == vectors[b]).groupby(vectors["delivery_day"]).all()
        rows.append({"comparison": f"{a} - {b}", "days": int(len(diff)),
                     "mean_daily_profit_diff_eur": mean, "ci95_low_eur": lo, "ci95_high_eur": hi,
                     "annual_profit_diff_eur_per_mw": mean * 365 / config.POWER_MW,
                     "days_rank_better": int((diff > 1e-9).sum()), "days_reg_better": int((diff < -1e-9).sum()),
                     "days_identical_price_vector": int(same_vector.sum()),
                     "rmse_diff_eur_mwh": rmse, "rmse_diff_ci95_low": rlo, "rmse_diff_ci95_high": rhi})
    return pd.DataFrame(rows)


def cmd_validate(args):
    """Validation year (PLAN.md §7): the 7 strategies, the paired rank - reg differences and
    the table of every configuration. The test period stays locked."""
    from bessrank import evaluate
    vectors = validation_vectors()
    # Every fitted configuration; "<ranker>:final" holds scores, not prices, so it is left out.
    config_cols = [c for c in vectors.columns if ":" in c and not c.endswith(":final")]
    log(f"Solving {len(STRATEGIES) + len(config_cols)} price vectors x 365 days with HiGHS + SCIP ...")
    daily = parallel_daily_profits(vectors, STRATEGIES + config_cols)
    evaluate.check_perfect_is_upper_bound(daily, [c for c in daily.columns if c not in ("delivery_day", "n_hours", "S-perfect")])

    table = pd.DataFrame([{**evaluate.value_summary(daily, s), **evaluate.forecast_summary(vectors, s)}
                          for s in STRATEGIES])
    paired = paired_differences(vectors, daily)
    configs = config_table(vectors, daily)
    table.to_csv(config.RESULTS_DIR / "validation_strategies.csv", index=False)
    daily[["delivery_day", "n_hours"] + STRATEGIES].to_csv(config.RESULTS_DIR / "validation_daily_profit.csv", index=False)
    paired.to_csv(config.RESULTS_DIR / "validation_paired.csv", index=False)
    configs.to_csv(config.RESULTS_DIR / "validation_configs.csv", index=False)
    update_provenance("validate", {"strategies": STRATEGIES, "days": int(len(daily)),
                                   "config_vectors": len(config_cols),
                                   "solver_cross_checks": int(len(daily) * (len(STRATEGIES) + len(config_cols)))})

    write_validation_summary(table, paired, configs)
    show = table.set_index("strategy")[list(DISPLAY)].rename(columns=DISPLAY)
    print(show.round(3).T.to_string())  # rounded for display only
    print(paired.round(3).T.to_string())


def write_validation_summary(table, paired, configs):
    """results/validation_summary.md: the tables of the S2 report, rounded for display only."""
    from bessrank.data import _markdown_table
    show = table[["strategy"] + list(DISPLAY)].rename(columns=DISPLAY).copy()
    show["Profit (EUR/MW/year)"] = show["Profit (EUR/MW/year)"].round(0).astype(int)
    for col in ["Capture", "Losing days", "Hit 2 cheapest", "Hit 2 dearest"]:
        show[col] = (100 * show[col]).round(1).astype(str) + "%"
    show = show.round(3)

    pairs = paired[["comparison", "mean_daily_profit_diff_eur", "ci95_low_eur", "ci95_high_eur",
                    "annual_profit_diff_eur_per_mw", "days_rank_better", "days_reg_better",
                    "days_identical_price_vector", "rmse_diff_eur_mwh", "rmse_diff_ci95_low",
                    "rmse_diff_ci95_high"]].round(2)
    pairs.columns = ["Comparison", "Mean daily diff (EUR)", "95% CI low", "95% CI high",
                     "Annual diff (EUR/MW)", "Days rank better", "Days reg better",
                     "Days same vector", "RMSE diff (EUR/MWh)", "RMSE CI low", "RMSE CI high"]

    cfg = configs[["model", "config", "seed", "max_depth", "learning_rate", "hidden", "layers", "n_iter",
                   "vector_rmse_eur_mwh", "spearman_rho_mean", "profit_eur_per_mw_year", "selected"]].copy()
    cfg["learning_rate"] = cfg["learning_rate"].round(4)
    cfg["vector_rmse_eur_mwh"] = cfg["vector_rmse_eur_mwh"].round(2)
    cfg["spearman_rho_mean"] = cfg["spearman_rho_mean"].round(4)
    cfg["profit_eur_per_mw_year"] = cfg["profit_eur_per_mw_year"].round(0).astype(int)
    cfg = cfg.astype(object).where(cfg.notna(), "")
    cfg.columns = ["Model", "Config", "Seed", "max_depth", "learning_rate", "hidden", "layers",
                   "Trees/epochs", "RMSE of price vector (EUR/MWh)", "Spearman rho (model output)",
                   "Validation profit (EUR/MW/year)", "Selected"]
    text = "\n".join([
        f"# Validation year {config.VAL_START} to {config.VAL_END} (365 delivery days)", "",
        "Simulated 1 MW / 2 MWh battery, day-ahead market only, price-taker. Rounded for display; "
        "the CSV files next to this one hold the raw numbers.", "",
        "VaR 5% is the 5th percentile (P5) of daily profit and ES 5% the mean daily profit of the "
        "worst 5% of days (19 of 365). Both are profit levels: a negative value is a loss.", "",
        "## Strategies", "", _markdown_table(show), "",
        "## Paired daily differences, rank minus reg (exploratory, validation only)", "",
        "Moving-block bootstrap over days: 7-day blocks, 10,000 resamples, seed 20261004.", "",
        _markdown_table(pairs), "",
        "## Every configuration (PLAN.md §7, used in §8)", "",
        "Seed `avg` is the 3-seed average. For rankers, the RMSE and the profit are those of the "
        "\"-rank\" price vector (the selected price model's values in the ranker's order); the "
        "Spearman rho is that of the model's own output (scores for rankers).", "",
        _markdown_table(cfg), ""])
    (config.RESULTS_DIR / "validation_summary.md").write_text(text)


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
    p = sub.add_parser("tune", help="random search of one model on the validation year")
    p.add_argument("model", choices=["xgb-reg", "xgb-rank", "lstm-reg", "lstm-rank"])
    p.set_defaults(func=cmd_tune)
    sub.add_parser("validate", help="validation-year strategies").set_defaults(func=cmd_validate)
    for name in ["test", "explore", "report"]:
        sub.add_parser(name).set_defaults(func=not_yet)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
