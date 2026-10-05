"""The official XGBoost test pipeline on Databricks serverless (PLAN.md §9), called by
notebooks/01_pipeline.py.

It reruns the pre-registered test steps for the XGBoost pair with the S3 code paths: quarterly
refits (backtest.quarters, models.training_rows, models.fit_predict with the frozen
hyperparameters, seeds 0-2), the "-rank" reassignment, and every day solved by HiGHS and
re-checked by SCIP (CLAUDE.md rule 7). The LSTM pair stays in the VM (torch is not installed
on serverless), so the five non-LSTM strategies are computed here.

Outputs: the Delta tables workspace.bess.gold_* (each row carries the git commit), one MLflow
run per fit plus a summary run, and a parity check against the S3 VM results. It is a
test-type run: data.load_hourly(include_test=True) goes through the lock (BESS_UNLOCK_TEST=1
and PREREGISTRATION.lock), with no bypass.
"""
import json
import os
import time
from datetime import timedelta

import numpy as np
import pandas as pd

from bessrank import backtest, battery, config, data, evaluate, features, models, strategies

XGB_KINDS = ["xgb-reg", "xgb-rank"]
DBX_STRATEGIES = ["S-perfect", "S-naive-1d", "S-naive-7d", "S-xgb-reg", "S-xgb-rank"]
TABLES = ["gold_features", "gold_predictions", "gold_schedules", "gold_daily_profit"]


def snake(name):
    """Column names that every Delta table accepts: 'S-xgb-reg' -> 's_xgb_reg', 'xgb-reg:s0' -> 'xgb_reg_s0'."""
    return name.replace("-", "_").replace(":", "_").lower()


def quarter_start_of(days, quarter_list):
    """Start of the test quarter of each delivery day."""
    starts = [q[0] for q in quarter_list]
    idx = np.searchsorted(np.array(starts, dtype="datetime64[D]"), np.array(list(days), dtype="datetime64[D]"),
                          side="right") - 1
    return np.array(starts, dtype=object)[idx]


def local_hour(frame):
    return pd.to_datetime(frame["ts_utc"], utc=True).dt.tz_convert(config.MARKET_TZ).dt.hour.to_numpy()


# --- 1-2. Refits and forecasts (XGBoost only, one fit at a time) -------------------------------
def refit_and_forecast_xgb(feats, quarter_list, on_fit=None, log=print):
    """backtest.refit_and_forecast for the XGBoost pair, run sequentially with the same 2 threads
    per fit (models.N_THREADS). `on_fit(meta)` is called after each fit (MLflow logging)."""
    frames = []
    for first, last in quarter_list:
        train = models.training_rows(feats, config.TRAIN_START, first - timedelta(days=1))
        predict = models.forecast_rows(feats, first, last)
        frame = predict[models.KEYS].copy()
        for kind in XGB_KINDS:
            params = backtest.selected_params(kind)
            for seed in config.SEEDS:
                started = time.time()
                n_iter, pred = models.fit_predict(kind, params, train, predict, seed)
                frame[f"{kind}:s{seed}"] = pred
                if on_fit:
                    on_fit({"model": kind, "seed": seed, "quarter_start": first, "quarter_end": last,
                            "params": params, "n_iter": int(n_iter), "fit_seconds": time.time() - started,
                            "train_days": int(train["delivery_day"].nunique()),
                            "train_last_day": train["delivery_day"].max(), "pred": pred, "frame": predict})
            # PLAN.md §4: mean prediction for price models, mean score for rankers.
            frame[kind] = frame[[f"{kind}:s{s}" for s in config.SEEDS]].mean(axis=1)
            log(f"{first} {kind}: done")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values("ts_utc").reset_index(drop=True)


# --- 3. Strategies, schedules and settlement ---------------------------------------------------
def xgb_strategy_vectors(hourly, forecasts, first_day, last_day):
    """The five non-LSTM price vectors (as backtest.strategy_vectors, without the LSTM pair)."""
    vectors = strategies.baseline_price_vectors(hourly)
    day = vectors["delivery_day"]
    vectors = vectors[(day >= first_day) & (day <= last_day)].sort_values("ts_utc").reset_index(drop=True)
    if not forecasts["ts_utc"].reset_index(drop=True).equals(vectors["ts_utc"]):
        raise ValueError("forecasts are not aligned with the hours of the period")
    seed_cols = [f"{k}:s{s}" for k in XGB_KINDS for s in config.SEEDS]
    vectors = pd.concat([vectors, forecasts[XGB_KINDS + seed_cols]], axis=1)
    vectors["S-xgb-reg"] = vectors["xgb-reg"]
    vectors["S-xgb-rank"] = strategies.rank_reassign_frame(vectors, "xgb-reg", "xgb-rank")
    if vectors[DBX_STRATEGIES].isna().any().any():
        raise ValueError("a strategy's price vector has a missing value")
    return vectors


def schedules_and_profits(vectors, names=DBX_STRATEGIES):
    """Solve every day for every strategy (HiGHS, re-checked by SCIP) and settle at the actual
    prices. Returns (hourly schedules, daily profit), both long format. Stops on a solver
    mismatch or if a strategy beats perfect foresight (CLAUDE.md rule 7)."""
    bat = battery.DEFAULT_BATTERY
    hourly_rows, daily_rows = [], []
    for day, hours in vectors.groupby("delivery_day", sort=True):
        actual = hours["price"].to_numpy()
        for name in names:
            sched = battery.solve_day(hours[name].to_numpy(), bat, cross_check=True)
            cash = actual * (sched.discharge - sched.charge) - bat.degradation_eur_per_mwh * sched.discharge
            hourly_rows.append(pd.DataFrame({
                "ts_utc": hours["ts_utc"].to_numpy(), "delivery_day": day, "strategy": name,
                "price_vector_eur_mwh": hours[name].to_numpy(), "actual_price_eur_mwh": actual,
                "charge_mwh": sched.charge, "discharge_mwh": sched.discharge,
                "net_sold_mwh": sched.discharge - sched.charge, "soc_end_of_hour_mwh": sched.soc,
                "cash_flow_eur": cash}))
            daily_rows.append({"delivery_day": day, "n_hours": len(hours), "strategy": name,
                               "profit_eur": battery.settle(sched, actual, bat)})
    daily = pd.DataFrame(daily_rows)
    wide = daily.pivot(index="delivery_day", columns="strategy", values="profit_eur").reset_index()
    evaluate.check_perfect_is_upper_bound(wide, [s for s in names if s != "S-perfect"])
    return pd.concat(hourly_rows, ignore_index=True), daily


# --- Parity with the S3 VM run -----------------------------------------------------------------
def parity(daily, forecasts, official_daily, vm_forecasts):
    """Daily profit of the five strategies against results/test_daily_profit.csv (relative rule
    of CLAUDE.md rule 7: |a - b| <= 1e-6 max(1, |a|, |b|)), and the forecasts against the VM's."""
    wide = daily.pivot(index="delivery_day", columns="strategy", values="profit_eur").reset_index()
    official = official_daily.copy()
    official["delivery_day"] = pd.to_datetime(official["delivery_day"]).dt.date
    merged = wide.merge(official, on="delivery_day", suffixes=("", "_vm"))
    out = {"days_compared": int(len(merged)), "profit": {}, "forecasts": {}}
    for s in DBX_STRATEGIES:
        a, b = merged[s].to_numpy(), merged[f"{s}_vm"].to_numpy()
        rel = np.abs(a - b) / np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))
        out["profit"][s] = {"max_abs_diff_eur": float(np.abs(a - b).max()), "max_rel_diff": float(rel.max()),
                            "days_over_1e-6": int((rel > 1e-6).sum())}
    vm = vm_forecasts.set_index("ts_utc")
    here = forecasts.set_index("ts_utc")
    for col in XGB_KINDS + [f"{k}:s{s}" for k in XGB_KINDS for s in config.SEEDS]:
        out["forecasts"][col] = float((here[col] - vm.loc[here.index, col]).abs().max())
    out["all_within_1e-6"] = all(v["days_over_1e-6"] == 0 for v in out["profit"].values())
    return out


# --- Driver (called from the notebook) ---------------------------------------------------------
def run_databricks(spark, commit, experiment, volume, log=print):
    """Steps 1-4 on serverless, the gold tables, MLflow and the parity check. Returns a summary."""
    import mlflow

    timings, started = {}, time.time()
    hourly = data.load_hourly(include_test=True)  # the test lock applies here
    feats = features.build_features(hourly)
    quarter_list = backtest.quarters(backtest.TEST_QUARTERS, config.TEST_END)
    first_day, last_day = quarter_list[0][0], quarter_list[-1][1]

    # Serverless blocks reading spark.mlflow.modelRegistryUri; setting it avoids the lookup (S0).
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(experiment)
    selected = {k: json.loads((config.RESULTS_DIR / f"selected_{k}.json").read_text()) for k in XGB_KINDS}

    def log_fit(meta):
        """One MLflow run per fit: hyperparameters, tree count, validation metric, commit."""
        kind, frame = meta["model"], meta["frame"]
        metric = models.forecast_metrics(kind, meta["pred"], frame)
        with mlflow.start_run(run_name=f"{kind} {meta['quarter_start']} seed {meta['seed']}"):
            mlflow.set_tags({"commit": commit, "stage": "test refit (official XGBoost pipeline)"})
            mlflow.log_params({**meta["params"], "model": kind, "seed": meta["seed"],
                               "quarter_start": str(meta["quarter_start"]), "quarter_end": str(meta["quarter_end"]),
                               "train_first_day": str(config.TRAIN_START), "train_last_day": str(meta["train_last_day"]),
                               "train_days": meta["train_days"], "commit": commit})
            mlflow.log_metrics({
                "n_trees": meta["n_iter"], "fit_seconds": meta["fit_seconds"],
                # the §7 selection metric of this configuration on the validation year (3-seed average)
                f"validation_{selected[kind]['selection_metric']}": selected[kind]["validation_value"],
                # this fit's forecast of its own test quarter
                "test_quarter_spearman_rho_mean": metric["spearman_rho_mean"],
                **({"test_quarter_rmse_eur_mwh": metric["rmse_eur_mwh"]} if "rmse_eur_mwh" in metric else {})})

    t = time.time()
    forecasts = refit_and_forecast_xgb(feats, quarter_list, on_fit=log_fit, log=log)
    timings["refits_s"] = time.time() - t
    t = time.time()
    vectors = xgb_strategy_vectors(hourly, forecasts, first_day, last_day)
    schedules, daily = schedules_and_profits(vectors)
    timings["solve_s"] = time.time() - t

    # Parity with the S3 VM run and H1 recomputed from these numbers.
    official = pd.read_csv(config.RESULTS_DIR / "test_daily_profit.csv")
    vm_forecasts = pd.read_parquet(f"{volume}/predictions/test_forecasts_regenerated.parquet")
    par = parity(daily, forecasts, official, vm_forecasts)
    wide = daily.pivot(index="delivery_day", columns="strategy", values="profit_eur").reset_index()
    h1, lo, hi = evaluate.block_bootstrap_ci(wide["S-xgb-rank"] - wide["S-xgb-reg"])
    summary = {"commit": commit, "h1_mean_daily_diff_eur": h1, "h1_ci95": [lo, hi],
               "h1_verdict_rule": evaluate.verdict_superiority(lo, hi), "parity": par,
               "profit_eur_per_mw_year": {s: float(wide[s].mean() * 365) for s in DBX_STRATEGIES}}

    # Gold tables: written as parquet to the volume, then read by Spark, so pandas never goes
    # through pyspark's own conversion. Every row carries the commit.
    t = time.time()
    test_feats = models.forecast_rows(feats, first_day, last_day)
    gold = {
        "gold_features": test_feats[models.KEYS[:3] + features.FEATURE_COLUMNS],
        "gold_predictions": vectors.drop(columns=["n_hours"]),
        "gold_schedules": schedules,
        "gold_daily_profit": daily.assign(cumulative_profit_eur=daily.groupby("strategy")["profit_eur"].cumsum()),
    }
    folder = f"{volume}/runs/databricks-{commit[:12]}/gold"
    os.makedirs(folder, exist_ok=True)
    for name, frame in gold.items():
        frame = frame.copy()
        frame.columns = [snake(c) for c in frame.columns]
        frame["quarter_start"] = quarter_start_of(frame["delivery_day"], quarter_list)
        if "ts_utc" in frame:
            frame["local_hour"] = local_hour(frame)
        frame["commit"] = commit
        path = f"{folder}/{name}.parquet"
        frame.to_parquet(path, index=False, coerce_timestamps="us", allow_truncated_timestamps=True)
        (spark.read.parquet(path).write.mode("overwrite").option("overwriteSchema", "true")
         .saveAsTable(f"{config.DBX_CATALOG}.{config.DBX_SCHEMA}.{name}"))
        log(f"{name}: {len(frame)} rows")
    timings["tables_s"] = time.time() - t
    timings["total_s"] = time.time() - started
    summary["timings_s"] = timings
    summary["tables"] = {n: f"{config.DBX_CATALOG}.{config.DBX_SCHEMA}.{n}" for n in TABLES}

    with mlflow.start_run(run_name=f"test summary {commit[:12]}"):
        mlflow.set_tags({"commit": commit, "stage": "test summary (official XGBoost pipeline, Databricks)"})
        mlflow.log_metrics({
            **{f"profit_{snake(s)}_eur_per_mw_year": v for s, v in summary["profit_eur_per_mw_year"].items()},
            "h1_mean_daily_diff_eur": h1, "h1_ci95_low_eur": lo, "h1_ci95_high_eur": hi,
            "parity_max_abs_profit_diff_eur": max(v["max_abs_diff_eur"] for v in par["profit"].values()),
            "parity_max_abs_forecast_diff": max(par["forecasts"].values()),
            "runtime_s": timings["total_s"]})
        mlflow.log_dict(summary, "summary.json")
    with open(f"{volume}/runs/databricks-{commit[:12]}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary
