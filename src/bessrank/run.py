"""Command line entry point: python -m bessrank.run {data,qa,smoke,features,tune,validate,test,explore,report,pipeline}."""
import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd

from bessrank import config
from bessrank.config import MODELS, STRATEGIES

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


def upload_run_outputs(run_type, files, commit=None):
    """Copy a run's forecasts and results/provenance.json to the Databricks volume, in
    runs/<run_type>-<commit>/, and check each file's size there.

    Data files stay out of git (RULES.md rule 9), so the volume is where a later session finds
    them (S4 had to refit the S3 forecasts). A failed upload never stops the run: the local
    files are kept and a warning says which uploads failed. Returns True if every file is there.
    """
    from bessrank import databricks
    commit = commit or git_commit()
    dirty = "-dirty" if commit.endswith("-dirty") else ""
    folder = f"runs/{run_type}-{commit.removesuffix('-dirty')[:12]}{dirty}"
    paths = [Path(f) for f in files if Path(f).exists()] + [config.PROVENANCE_JSON]
    failed = []
    for path in paths:
        remote = f"{folder}/{path.name}"
        try:
            ok = databricks.upload_file(path, remote) and databricks.remote_file_size(remote) == path.stat().st_size
        except Exception as exc:  # no DATABRICKS_HOST, network or authentication error
            log(f"upload of {path.name} raised {type(exc).__name__}")
            ok = False
        if not ok:
            failed.append(path.name)
    if failed:
        log(f"WARNING: upload to the Databricks volume failed for {', '.join(failed)}. The run's local "
            f"files are kept (data/predictions/, results/); upload them before this VM is reclaimed.")
    else:
        log(f"Uploaded {len(paths)} files to {config.DBX_VOLUME_PATH}/{folder}/ (sizes checked).")
    return not failed


# Display labels: VaR and ES are profit levels, negative = loss (decided on 2026-10-05).
DISPLAY = {
    "profit_eur_per_mw_year": "Profit (EUR/MW/year)", "capture_rate": "Capture",
    "rmse_eur_mwh": "RMSE (EUR/MWh)", "mae_eur_mwh": "MAE (EUR/MWh)", "spearman_rho_mean": "Spearman rho",
    "hit_rate_cheapest2": "Hit 2 cheapest", "hit_rate_dearest2": "Hit 2 dearest",
    "var5_p5_of_daily_profit_eur": "VaR 5%: P5 of daily profit (EUR)",
    "es5_mean_of_worst5pct_days_eur": "ES 5%: mean of the worst 5% of days (EUR)",
    "losing_day_share": "Losing days", "max_drawdown_eur": "Max drawdown (EUR)",
}


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


def cmd_validate(args):
    """Validation year (PLAN.md §7): the 7 strategies, the paired rank - reg differences and
    the table of every configuration. The test period stays locked."""
    from bessrank import evaluate
    vectors = validation_vectors()
    # Every fitted configuration; "<ranker>:final" holds scores, not prices, so it is left out.
    config_cols = [c for c in vectors.columns if ":" in c and not c.endswith(":final")]
    log(f"Solving {len(STRATEGIES) + len(config_cols)} price vectors x 365 days with HiGHS + SCIP ...")
    daily = evaluate.parallel_daily_profits(vectors, STRATEGIES + config_cols)
    evaluate.check_perfect_is_upper_bound(daily, [c for c in daily.columns if c not in ("delivery_day", "n_hours", "S-perfect")])

    table = pd.DataFrame([{**evaluate.value_summary(daily, s), **evaluate.forecast_summary(vectors, s)}
                          for s in STRATEGIES])
    paired = evaluate.paired_differences(vectors, daily)
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


# --- Test run (PLAN.md §6) --------------------------------------------------------------------
# Files the test run depends on. They must equal the pre-registration commit (PLAN.md §6).
FROZEN_PATHS = ["src", "requirements.txt", "pyproject.toml", "results/selected_xgb-reg.json",
                "results/selected_xgb-rank.json", "results/selected_lstm-reg.json",
                "results/selected_lstm-rank.json"]


def code_changes_since(commit):
    """Frozen files that differ from `commit` in the working tree, including new files."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True, text=True, check=True).stdout.split()
    return sorted(set(git("diff", "--name-only", commit, "--", *FROZEN_PATHS))
                  | set(git("ls-files", "--others", "--exclude-standard", "--", *FROZEN_PATHS)))


def cmd_test(args):
    """The test run of PLAN.md §6, once. With --rehearsal, the same pipeline on the validation
    year (quarterly refits from 2024-10-01), which needs no test data."""
    from bessrank import backtest, data
    if args.rehearsal:
        prefix, starts, last_day = "rehearsal", backtest.REHEARSAL_QUARTERS, config.VAL_END
        hourly = data.load_hourly()
        lock_commit, changed = None, []
    else:
        config.require_test_unlocked()
        lock_commit = config.LOCK_FILE.read_text().strip()
        changed = code_changes_since(lock_commit)
        prefix = "test_amended" if args.amendment else "test"
        if changed and not args.amendment:
            sys.exit("The test run uses the code of the pre-registration commit "
                     f"{lock_commit[:12]} (PLAN.md §6). Changed: {', '.join(changed)}. A re-run after "
                     "a fix is a dated §12 amendment: run with --amendment.")
        if (config.RESULTS_DIR / f"{prefix}_hypotheses.csv").exists():
            sys.exit(f"results/{prefix}_hypotheses.csv exists: the test runs once (PLAN.md §6).")
        starts, last_day = backtest.TEST_QUARTERS, config.TEST_END
        hourly = data.load_hourly(include_test=True)

    quarter_list = backtest.quarters(starts, last_day)
    log(f"{prefix}: {quarter_list[0][0]} to {last_day}, refits at {', '.join(str(s) for s in starts)}")
    out = backtest.run(hourly, quarter_list, log=log)

    for name in ["strategies", "hypotheses", "quarterly", "quarterly_pairs", "daily_profit", "refits"]:
        out[name].to_csv(config.RESULTS_DIR / f"{prefix}_{name}.csv", index=False)
    # Hourly forecasts stay in data/ (they sit next to SMARD prices, never committed).
    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out["forecasts"].to_parquet(config.PREDICTIONS_DIR / f"{prefix}_forecasts.parquet", index=False)
    update_provenance(prefix, {"first_day": str(quarter_list[0][0]), "last_day": str(last_day),
                               "refit_days": [str(s) for s in starts], "days": int(len(out["daily_profit"])),
                               "seeds": list(config.SEEDS), "strategies": STRATEGIES,
                               "solver_cross_checks": out["solver_cross_checks"],
                               "preregistration_commit": lock_commit,
                               "frozen_files_changed_since_preregistration": changed,
                               "runtime_seconds": {k: round(v, 1) for k, v in out["timings"].items()}})
    write_test_summary(prefix, out, quarter_list)
    upload_run_outputs(prefix, [config.PREDICTIONS_DIR / f"{prefix}_forecasts.parquet"])
    show = out["hypotheses"].set_index("hypothesis")[["estimate", "ci95_low", "ci95_high", "margin", "verdict"]]
    print(show.round(3).to_string())  # rounded for display only
    log(f"done in {out['timings']['total_s'] / 60:.1f} min")


def write_test_summary(prefix, out, quarter_list):
    """results/<prefix>_summary.md: hypotheses, strategies and quarters, rounded for display."""
    from bessrank.data import _markdown_table
    hyp = out["hypotheses"][["hypothesis", "role", "comparison", "statistic", "estimate", "ci95_low",
                             "ci95_high", "margin", "verdict"]].copy()
    for col in ["estimate", "ci95_low", "ci95_high", "margin"]:
        hyp[col] = hyp[col].round(3)
    hyp = hyp.astype(object).where(hyp.notna(), "")
    h1 = out["hypotheses"].set_index("hypothesis").loc[["H1", "H3"]]
    extra = h1[["annual_diff_eur_per_mw", "relative_to_reg_profit", "days_rank_better",
                "days_reg_better", "days_identical_price_vector"]].reset_index()
    extra["annual_diff_eur_per_mw"] = extra["annual_diff_eur_per_mw"].round(0)
    extra["relative_to_reg_profit"] = (100 * extra["relative_to_reg_profit"].astype(float)).round(2).astype(str) + "%"
    extra.columns = ["Hypothesis", "Annual diff (EUR/MW/year)", "Diff / reg profit", "Days rank better",
                     "Days reg better", "Days same vector"]

    table = out["strategies"]
    show = table[["strategy"] + list(DISPLAY)].rename(columns=DISPLAY).copy()
    show["Profit (EUR/MW/year)"] = show["Profit (EUR/MW/year)"].round(0).astype(int)
    for col in ["Capture", "Losing days", "Hit 2 cheapest", "Hit 2 dearest"]:
        show[col] = (100 * show[col]).round(1).astype(str) + "%"
    show = show.round(3)

    q = out["quarterly"].copy()
    q["profit_eur_per_mw"] = q["profit_eur_per_mw"].round(0).astype(int)
    q["capture_rate"] = (100 * q["capture_rate"]).round(1).astype(str) + "%"
    q = q[["quarter_start", "strategy", "days", "profit_eur_per_mw", "capture_rate", "rmse_eur_mwh",
           "spearman_rho_mean"]].round(3)
    q.columns = ["Quarter from", "Strategy", "Days", "Profit (EUR/MW)", "Capture", "RMSE (EUR/MWh)", "Spearman rho"]
    qp = out["quarterly_pairs"].round(3)
    qp.columns = ["Quarter from", "Comparison", "Days", "Mean daily diff (EUR)", "Days rank better",
                  "Days reg better", "RMSE diff (EUR/MWh)"]

    refits = out["refits"].groupby(["quarter_start", "model"]).agg(
        train_days=("train_days", "first"), dropped=("train_days_dropped", "first"),
        n_iter=("n_iter", lambda x: ", ".join(str(v) for v in x)), fit_s=("fit_seconds", "sum")).reset_index()
    refits["fit_s"] = refits["fit_s"].round(0).astype(int)
    refits.columns = ["Refit (quarter from)", "Model", "Training days", "Days dropped (§2)",
                      "Trees/epochs per seed", "Fit time, 3 seeds (s)"]
    t = out["timings"]
    label = ("Rehearsal on the validation year (exploratory): the §6 test pipeline with refits at the "
             "start of each validation quarter." if prefix == "rehearsal" else
             "Test year, pre-registered run (PLAN.md §6). H1 is the only confirmatory test; H2 and H3 "
             "are secondary, without multiplicity adjustment.")
    text = "\n".join([
        f"# {prefix.replace('_', ' ').capitalize()}: {quarter_list[0][0]} to {quarter_list[-1][1]} "
        f"({len(out['daily_profit'])} delivery days)", "", label, "",
        "Simulated 1 MW / 2 MWh battery, day-ahead market only, price-taker. Rounded for display; "
        f"the `{prefix}_*.csv` files next to this one hold the raw numbers.", "",
        "## Hypotheses", "",
        "Moving-block bootstrap over days: 7-day blocks, 10,000 resamples, seed 20261004 (same "
        "resampled days for every row). Margin = 1% of the reg strategy's RMSE over the period.", "",
        _markdown_table(hyp), "", _markdown_table(extra), "",
        "## Strategies", "",
        "VaR 5% is the 5th percentile (P5) of daily profit and ES 5% the mean daily profit of the "
        "worst 5% of days. Both are profit levels: a negative value is a loss.", "",
        _markdown_table(show), "",
        "## By quarter (descriptive)", "", _markdown_table(q), "", _markdown_table(qp), "",
        "## Refits", "", _markdown_table(refits), "",
        f"Runtime: {t['total_s'] / 60:.1f} min in total (features {t['features_s']:.0f} s, refits "
        f"{t['refits_s'] / 60:.1f} min, {out['solver_cross_checks']} day solves with HiGHS and SCIP "
        f"{t['solve_s']:.0f} s).", ""])
    (config.RESULTS_DIR / f"{prefix}_summary.md").write_text(text)


def cmd_explore(args):
    """Exploratory analyses after the test run (PLAN.md §8, S4). Outputs go to
    results/explore/; the official test results are never recomputed or overwritten."""
    from bessrank import explore
    config.require_test_unlocked()
    explore.EXPLORE_DIR.mkdir(parents=True, exist_ok=True)
    steps = {"forecasts": explore.regenerate_forecasts, "xgb-variants": explore.run_xgb_variants,
             "analyses": explore.analyses, "conformal-cvar": explore.conformal_cvar}
    steps[args.step](log=log)
    update_provenance(f"explore_{args.step}", {"label": "exploratory (PLAN.md §8)"})
    forecasts = {"forecasts": [explore.FORECASTS_PARQUET], "xgb-variants": [explore.VARIANTS_PARQUET],
                 "conformal-cvar": [explore.QUANTILES_PARQUET], "analyses": []}
    upload_run_outputs(f"explore-{args.step}", forecasts[args.step])


def cmd_report(args):
    """The three README figures. The example day plots test-year prices, so the lock must be open."""
    from bessrank import explore
    config.require_test_unlocked()
    explore.readme_figures(log=log)


def cmd_pipeline(args):
    """The official XGBoost test pipeline on Databricks serverless (PLAN.md §9): bundle the
    committed code, upload it, import the notebook, create or update the job
    `bess-rank-pipeline`, run it once and save its summary to results/databricks_pipeline.json."""
    import tempfile
    from bessrank import databricks
    commit = git_commit()
    if commit.endswith("-dirty"):
        sys.exit("Commit and push first: every gold table and MLflow run records the commit.")
    user = databricks.current_user()
    with tempfile.TemporaryDirectory() as tmp:
        bundle_local = Path(tmp) / f"bess-rank-{commit[:12]}.tar"
        databricks.build_bundle(commit, bundle_local)
        bundle = f"pipeline/{bundle_local.name}"
        if not databricks.upload_file(bundle_local, bundle):
            sys.exit("Upload of the code bundle failed.")
    notebook_path = f"/Users/{user}/bess-rank-notebooks/01_pipeline"
    databricks.import_notebook(databricks.PIPELINE_NOTEBOOK, notebook_path)
    job_id = databricks.ensure_job(databricks.job_settings(notebook_path, bundle))
    run_id = databricks.run_job_now(job_id)
    log(f"job {databricks.JOB_NAME} ({job_id}): run {run_id} started, bundle {bundle}")
    result, detail = databricks.wait_for_run(run_id, timeout_s=5400, poll_s=30)
    log(f"run {run_id}: {result}")
    summary = databricks.read_text(f"runs/databricks-{commit[:12]}/summary.json")
    out = {"job_name": databricks.JOB_NAME, "job_id": job_id, "run_id": run_id, "result": result,
           "bundle": bundle, "summary": json.loads(summary) if summary else None,
           "error": None if result == "SUCCESS" else str(detail)[:2000]}
    (config.RESULTS_DIR / "databricks_pipeline.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    update_provenance("databricks_pipeline", {"job": databricks.JOB_NAME, "run_id": run_id, "result": result,
                                              "pipeline_commit": commit})
    if result != "SUCCESS":
        sys.exit(f"Databricks run failed: {str(detail)[:500]}")


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
    sub.add_parser("pipeline", help="official XGBoost test pipeline as the Databricks job (PLAN.md §9)"
                   ).set_defaults(func=cmd_pipeline)
    sub.add_parser("features", help="build the feature table").set_defaults(func=cmd_features)
    p = sub.add_parser("tune", help="random search of one model on the validation year")
    p.add_argument("model", choices=["xgb-reg", "xgb-rank", "lstm-reg", "lstm-rank"])
    p.set_defaults(func=cmd_tune)
    sub.add_parser("validate", help="validation-year strategies").set_defaults(func=cmd_validate)
    p = sub.add_parser("test", help="the pre-registered test run (PLAN.md §6), once")
    p.add_argument("--rehearsal", action="store_true",
                   help="same pipeline on the validation year; needs no test data")
    p.add_argument("--amendment", action="store_true",
                   help="re-run after a dated §12 amendment; writes test_amended_* files")
    p.set_defaults(func=cmd_test)
    p = sub.add_parser("explore", help="exploratory analyses after the test run (PLAN.md §8)")
    p.add_argument("step", choices=["forecasts", "xgb-variants", "analyses", "conformal-cvar"])
    p.set_defaults(func=cmd_explore)
    sub.add_parser("report", help="the README figures (results/figures/)").set_defaults(func=cmd_report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
