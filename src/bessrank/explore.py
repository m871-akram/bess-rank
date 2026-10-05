"""Exploratory analysis after the test run (PLAN.md §8, session S4).

Everything here is exploratory. The pre-registered H1-H3 verdicts are those of
results/test_*.csv; this module never recomputes or overwrites them. Its outputs go to
results/explore/ (raw numbers in CSV/JSON, figures in PNG).

Steps, each a function called from `python -m bessrank.run explore <step>`:
- forecasts: regenerate the test-year forecasts with the frozen code and check that they give
  the official daily profits (the S3 forecasts stayed in that session's VM);
- xgb-variants: the XGBoost pair without TSO generation forecasts (§8 robustness) and with
  tree counts fixed from validation (Akram's decision 2 on the S3 report);
- the analyses (sensitivity, stability, accuracy vs value, where profit is lost, figures).
"""
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import timedelta

import numpy as np
import pandas as pd
import xgboost as xgb

from bessrank import backtest, battery, config, data, evaluate, features, models, risk
from bessrank.config import STRATEGIES

EXPLORE_DIR = config.RESULTS_DIR / "explore"
FORECASTS_PARQUET = config.PREDICTIONS_DIR / "test_forecasts_regenerated.parquet"
VARIANTS_PARQUET = config.PREDICTIONS_DIR / "test_xgb_variants.parquet"
XGB_KINDS = ["xgb-reg", "xgb-rank"]


def write_json(name, obj):
    EXPLORE_DIR.mkdir(parents=True, exist_ok=True)
    (EXPLORE_DIR / name).write_text(json.dumps(obj, indent=2, default=str) + "\n")


def test_quarters():
    return backtest.quarters(backtest.TEST_QUARTERS, config.TEST_END)


def load_test_inputs():
    """Hourly table and features including the test year (the lock was opened in S3)."""
    hourly = data.load_hourly(include_test=True)
    return hourly, features.build_features(hourly)


# --- Test-year forecasts ----------------------------------------------------------------------
def regenerate_forecasts(log=print):
    """Re-run steps 1-2 of the test pipeline (backtest.refit_and_forecast, unchanged since the
    lock commit) and check the result against the official run: same trees/epochs per fit and
    the same daily profit for every strategy on every day."""
    hourly, feats = load_test_inputs()
    quarter_list = test_quarters()
    forecasts, refits = backtest.refit_and_forecast(feats, quarter_list, log=log)
    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(FORECASTS_PARQUET, index=False)

    vectors = backtest.strategy_vectors(hourly, forecasts, quarter_list[0][0], quarter_list[-1][1])
    daily = evaluate.parallel_daily_profits(vectors, STRATEGIES)
    official = pd.read_csv(config.RESULTS_DIR / "test_daily_profit.csv")
    official_refits = pd.read_csv(config.RESULTS_DIR / "test_refits.csv")
    same_iters = (refits.sort_values(["quarter_start", "model", "seed"])["n_iter"].to_numpy()
                  == official_refits.sort_values(["quarter_start", "model", "seed"])["n_iter"].to_numpy())
    check = {
        "what": "test-year forecasts regenerated with the frozen backtest code; compared with "
                "results/test_daily_profit.csv and results/test_refits.csv",
        "fits_with_same_n_iter": int(same_iters.sum()), "fits": int(len(same_iters)),
        "max_abs_daily_profit_diff_eur": {s: float((daily[s] - official[s]).abs().max()) for s in STRATEGIES},
    }
    write_json("regenerated_forecasts_check.json", check)
    log(json.dumps(check, indent=2))
    return check


def load_forecasts():
    """Regenerated test-year forecasts (or the official file if a session still has it)."""
    return pd.read_parquet(FORECASTS_PARQUET)


def test_vectors(hourly, forecasts):
    first, last = test_quarters()[0][0], test_quarters()[-1][1]
    return backtest.strategy_vectors(hourly, forecasts, first, last)


# --- XGBoost variants (robustness and the tree-count check) -----------------------------------
def _dmatrix(kind, frame, columns):
    """As models._dmatrix, with a chosen list of feature columns."""
    X = frame[columns]
    if kind == "xgb-reg":
        return xgb.DMatrix(X, label=frame["price"].to_numpy())
    qid = pd.factorize(frame["delivery_day"])[0]
    return xgb.DMatrix(X, label=models.rank_labels(frame), qid=qid)


def fit_predict_variant(kind, params, train, predict, seed, columns, n_trees=None):
    """models.fit_predict with two changes: a chosen feature list, and an optional fixed
    number of trees (then no early stopping, the whole window is fitted directly)."""
    p = models.booster_params(kind, params, seed)
    if n_trees is None:
        inner, es = models.split_early_stopping(train)
        stop = {"early_stopping_rounds": models.EARLY_STOPPING_ROUNDS, "verbose_eval": False}
        if kind == "xgb-rank":
            stop.update(custom_metric=models._spearman_metric(es), maximize=True)
        booster = xgb.train(p, _dmatrix(kind, inner, columns), models.MAX_TREES,
                            evals=[(_dmatrix(kind, es, columns), "es")], **stop)
        n_trees = booster.best_iteration + 1
    final = xgb.train(p, _dmatrix(kind, train, columns), n_trees)
    return n_trees, final.predict(xgb.DMatrix(predict[columns]))


def _variant_job(job):
    kind, params, train, predict, columns, n_trees_per_seed = job
    out = []
    for i, seed in enumerate(config.SEEDS):
        n_fixed = None if n_trees_per_seed is None else n_trees_per_seed[i]
        out.append((seed, *fit_predict_variant(kind, params, train, predict, seed, columns, n_fixed)))
    return out


def xgb_variant_forecasts(feats, name, columns, fixed_trees=False, log=print):
    """Quarterly refits of the XGBoost pair (3 seeds, seed-averaged) with a feature list and,
    if fixed_trees, the validation tree counts of results/selected_*.json for every refit."""
    jobs, meta, frames = [], [], {}
    for first, last in test_quarters():
        train = models.training_rows(feats, config.TRAIN_START, first - timedelta(days=1))
        predict = models.forecast_rows(feats, first, last)
        frames[first] = predict[models.KEYS].copy()
        for kind in XGB_KINDS:
            selected = json.loads((config.RESULTS_DIR / f"selected_{kind}.json").read_text())
            trees = selected["n_iter_per_seed"] if fixed_trees else None
            jobs.append((kind, selected["params"], train, predict, columns, trees))
            meta.append((first, kind))
    refits = []
    with ProcessPoolExecutor(backtest.FIT_WORKERS) as pool:
        for (first, kind), result in zip(meta, pool.map(_variant_job, jobs)):
            for seed, n_trees, pred in result:
                frames[first][f"{kind}:s{seed}"] = pred
                refits.append({"variant": name, "quarter_start": first, "model": kind,
                               "seed": seed, "n_iter": int(n_trees)})
            frames[first][kind] = frames[first][[f"{kind}:s{s}" for s in config.SEEDS]].mean(axis=1)
            log(f"{name} {first} {kind}: trees {[r[1] for r in result]}")
    forecasts = pd.concat(frames.values(), ignore_index=True).sort_values("ts_utc").reset_index(drop=True)
    return forecasts, pd.DataFrame(refits)


def run_xgb_variants(log=print):
    """Both XGBoost variants on the test year. Forecasts go to data/ (they sit next to SMARD
    prices), the fits to results/explore/xgb_variant_refits.csv."""
    hourly, feats = load_test_inputs()
    no_gen = [c for c in features.FEATURE_COLUMNS if c not in features.GEN_FORECAST_FEATURES]
    log(f"no-TSO-generation variant: {len(no_gen)} of {len(features.FEATURE_COLUMNS)} features")
    frames, refits = [], []
    for name, columns, fixed in [("no_gen_forecasts", no_gen, False),
                                 ("fixed_trees", features.FEATURE_COLUMNS, True)]:
        started = time.time()
        f, r = xgb_variant_forecasts(feats, name, columns, fixed_trees=fixed, log=log)
        log(f"{name}: {(time.time() - started) / 60:.1f} min")
        frames.append(f.drop(columns=["price", "hour"]).set_index(["ts_utc", "delivery_day"]).add_prefix(f"{name}|"))
        refits.append(r)
    out = pd.concat(frames, axis=1).reset_index()
    out.to_parquet(VARIANTS_PARQUET, index=False)
    pd.concat(refits).to_csv(EXPLORE_DIR / "xgb_variant_refits.csv", index=False)


# --- Solving price vectors with a chosen battery -----------------------------------------------
def _solve_job(job):
    vectors, names, bat = job
    return evaluate.daily_profits(vectors, names, bat)


def solve_profits(vectors, names, bat=battery.DEFAULT_BATTERY, workers=4):
    """evaluate.parallel_daily_profits with a chosen battery. Every day is solved by HiGHS and
    re-checked by SCIP, and perfect foresight must be >= every vector (CLAUDE.md rule 7)."""
    names = ["S-perfect"] + [n for n in names if n != "S-perfect"]
    chunks = [names[i::workers] for i in range(workers) if names[i::workers]]
    keys = ["ts_utc", "delivery_day", "price"]
    with ProcessPoolExecutor(len(chunks)) as pool:
        parts = list(pool.map(_solve_job, [(vectors[keys + c], c, bat) for c in chunks]))
    daily = parts[0]
    for part in parts[1:]:
        daily = daily.merge(part, on=["delivery_day", "n_hours"])
    daily = daily[["delivery_day", "n_hours"] + names]
    evaluate.check_perfect_is_upper_bound(daily, names[1:])
    return daily


def pair_statistic(daily, a, b):
    """Mean daily profit difference a - b with the §6 moving-block bootstrap CI (same seed and
    resampled days as the official run), and the share of b's gap to perfect foresight that a
    closes. Exploratory: no verdict."""
    mean, lo, hi = evaluate.block_bootstrap_ci(daily[a] - daily[b])
    gap = (daily["S-perfect"] - daily[b]).mean()
    return {"comparison": f"{a} - {b}", "mean_daily_diff_eur": mean, "ci95_low_eur": lo,
            "ci95_high_eur": hi, "annual_diff_eur_per_mw": mean * 365 / config.POWER_MW,
            "share_of_gap_to_perfect_closed": mean / gap if gap > 0 else np.nan}


def quarter_of(days):
    """Start of the test quarter of each delivery day (an array of dates)."""
    starts = np.array(backtest.TEST_QUARTERS, dtype="datetime64[D]")
    idx = np.searchsorted(starts, np.array(list(days), dtype="datetime64[D]"), side="right") - 1
    return np.array(backtest.TEST_QUARTERS, dtype=object)[idx]


# --- A. Sensitivity: re-dispatch the test forecasts with other batteries ------------------------
DEFAULT = battery.DEFAULT_BATTERY
# Other durations keep 1 MW, start and end at 50% charge and allow one full cycle per day.
BATTERIES = {
    "base: 2 h, 10 EUR/MWh": DEFAULT,
    "degradation 0 EUR/MWh": replace(DEFAULT, degradation_eur_per_mwh=0.0),
    "degradation 20 EUR/MWh": replace(DEFAULT, degradation_eur_per_mwh=20.0),
    "1 h (1 MW / 1 MWh)": replace(DEFAULT, capacity_mwh=1.0, soc_start_mwh=0.5, max_discharge_mwh=1.0),
    "4 h (1 MW / 4 MWh)": replace(DEFAULT, capacity_mwh=4.0, soc_start_mwh=2.0, max_discharge_mwh=4.0),
}


def sensitivity(vectors, log=print):
    """Every strategy and both rank - reg pairs under each battery setting. Same forecasts,
    no retraining. The base row must reproduce the official daily profits."""
    rows, pairs = [], []
    official = pd.read_csv(config.RESULTS_DIR / "test_daily_profit.csv")
    for name, bat in BATTERIES.items():
        daily = solve_profits(vectors, STRATEGIES, bat)
        if name.startswith("base"):
            diff = max(float((daily[s] - official[s]).abs().max()) for s in STRATEGIES)
            log(f"sensitivity base vs official daily profits: max |diff| {diff:.2e} EUR")
            write_json("sensitivity_base_check.json", {"max_abs_daily_profit_diff_eur": diff})
        for s in STRATEGIES:
            rows.append({"setting": name, **evaluate.value_summary(daily, s)})
        for a, b in config.PAIRS:
            pairs.append({"setting": name, **pair_statistic(daily, a, b)})
        log(f"sensitivity {name}: done")
    pd.DataFrame(rows).to_csv(EXPLORE_DIR / "sensitivity_strategies.csv", index=False)
    pd.DataFrame(pairs).to_csv(EXPLORE_DIR / "sensitivity_pairs.csv", index=False)


# --- A. Robustness (no TSO generation forecasts) and D. the tree-count check --------------------
def variant_analysis(hourly, vectors, official_daily, log=print):
    """H1's statistic for each XGBoost variant next to the official one, overall and by
    quarter, with the variant's RMSE and Spearman rho."""
    from bessrank import strategies
    var = pd.read_parquet(VARIANTS_PARQUET)
    if not var["ts_utc"].reset_index(drop=True).equals(vectors["ts_utc"]):
        raise ValueError("variant forecasts are not aligned with the test hours")
    refits = pd.read_csv(EXPLORE_DIR / "xgb_variant_refits.csv")
    official_refits = pd.read_csv(config.RESULTS_DIR / "test_refits.csv")
    rows, by_q, metrics = [], [], []
    for name in ["official", "no_gen_forecasts", "fixed_trees"]:
        v = vectors[["ts_utc", "delivery_day", "price"]].copy()
        if name == "official":
            v["S-xgb-reg"], v["S-xgb-rank"] = vectors["S-xgb-reg"], vectors["S-xgb-rank"]
            daily, fits = official_daily, official_refits
        else:
            v["xgb-reg"], v["xgb-rank"] = var[f"{name}|xgb-reg"], var[f"{name}|xgb-rank"]
            v["S-xgb-reg"] = v["xgb-reg"]
            v["S-xgb-rank"] = strategies.rank_reassign_frame(v, "xgb-reg", "xgb-rank")
            v["S-perfect"] = v["price"]
            daily = solve_profits(v, ["S-xgb-reg", "S-xgb-rank"])
            fits = refits[refits["variant"] == name]
        rows.append({"variant": name, **pair_statistic(daily, "S-xgb-rank", "S-xgb-reg"),
                     "profit_xgb_reg_eur_per_mw_year": float(daily["S-xgb-reg"].mean() * 365),
                     "profit_xgb_rank_eur_per_mw_year": float(daily["S-xgb-rank"].mean() * 365)})
        for s in ["S-xgb-reg", "S-xgb-rank"]:
            metrics.append({"variant": name, "strategy": s, **evaluate.forecast_summary(v, s)})
        q = quarter_of(daily["delivery_day"])
        for first in backtest.TEST_QUARTERS:
            d = daily[q == first]
            trees = {m: "/".join(str(n) for n in fits[(fits["quarter_start"].astype(str) == str(first))
                                                       & (fits["model"] == m)].sort_values("seed")["n_iter"])
                     for m in XGB_KINDS}
            by_q.append({"variant": name, "quarter_start": first, "days": len(d),
                         "mean_daily_diff_eur": float((d["S-xgb-rank"] - d["S-xgb-reg"]).mean()),
                         "trees_xgb_reg": trees["xgb-reg"], "trees_xgb_rank": trees["xgb-rank"]})
        log(f"variant {name}: H1 statistic {rows[-1]['mean_daily_diff_eur']:.3f} EUR/day")
    pd.DataFrame(rows).to_csv(EXPLORE_DIR / "xgb_variants_h1.csv", index=False)
    pd.DataFrame(by_q).to_csv(EXPLORE_DIR / "xgb_variants_h1_by_quarter.csv", index=False)
    pd.DataFrame(metrics).to_csv(EXPLORE_DIR / "xgb_variants_forecast_metrics.csv", index=False)


# --- A. Stability: PSI of the key features, every metric by quarter and month -------------------
def training_boosters(feats):
    """The selected XGB-reg and XGB-rank (seed 0, validation tree count) fitted on the training
    period 2019-01-01 to 2024-09-30; used for feature importance, PSI and SHAP."""
    train = models.training_rows(feats, config.TRAIN_START, config.TRAIN_END)
    boosters = {}
    for kind in XGB_KINDS:
        selected = json.loads((config.RESULTS_DIR / f"selected_{kind}.json").read_text())
        p = models.booster_params(kind, selected["params"], seed=0, n_threads=4)
        boosters[kind] = xgb.train(p, _dmatrix(kind, train, features.FEATURE_COLUMNS), selected["n_iter_per_seed"][0])
    return boosters


def feature_importance(boosters):
    """Total-gain importance of each feature; each model's gains sum to 1 and the two models
    are averaged."""
    out = pd.DataFrame(index=features.FEATURE_COLUMNS)
    for kind, booster in boosters.items():
        gain = pd.Series(booster.get_score(importance_type="total_gain"))
        out[kind] = (gain / gain.sum()).reindex(out.index).fillna(0.0)
    out["mean"] = out[XGB_KINDS].mean(axis=1)
    return out.sort_values("mean", ascending=False)


def shap_top_features(boosters, feats, log=print):
    """Mean |SHAP value| of each feature on the test year (XGBoost's TreeSHAP, pred_contribs),
    for the training-period XGB-reg (EUR/MWh) and XGB-rank (score units). Exploratory."""
    test = models.forecast_rows(feats, config.TEST_START, config.TEST_END)
    rows = []
    for kind, booster in boosters.items():
        contribs = booster.predict(xgb.DMatrix(test[features.FEATURE_COLUMNS]), pred_contribs=True)[:, :-1]
        mean_abs = pd.Series(np.abs(contribs).mean(axis=0), index=features.FEATURE_COLUMNS)
        for rank, (f, v) in enumerate(mean_abs.sort_values(ascending=False).head(10).items(), start=1):
            rows.append({"model": kind, "rank": rank, "feature": f, "mean_abs_shap": v,
                         "share_of_total": v / mean_abs.sum(),
                         "uses_tso_generation_forecast": f in features.GEN_FORECAST_FEATURES})
    pd.DataFrame(rows).to_csv(EXPLORE_DIR / "shap_top10.csv", index=False)
    log("SHAP: written")


def psi_report(feats, log=print):
    """PSI of the 10 most important features, test vs train (validation vs train for
    reference)."""
    boosters = training_boosters(feats)
    shap_top_features(boosters, feats, log=log)
    imp = feature_importance(boosters)
    imp.to_csv(EXPLORE_DIR / "feature_importance.csv", index_label="feature")
    train = models.training_rows(feats, config.TRAIN_START, config.TRAIN_END)
    val = models.forecast_rows(feats, config.VAL_START, config.VAL_END)
    test = models.forecast_rows(feats, config.TEST_START, config.TEST_END)
    rows = []
    for rank, f in enumerate(imp.index[:10], start=1):
        rows.append({"importance_rank": rank, "feature": f, "importance": imp.loc[f, "mean"],
                     "uses_tso_generation_forecast": f in features.GEN_FORECAST_FEATURES,
                     "psi_test_vs_train": risk.psi(train[f], test[f]), "psi_validation_vs_train": risk.psi(train[f], val[f])})
    pd.DataFrame(rows).to_csv(EXPLORE_DIR / "psi_top10.csv", index=False)
    log("stability: PSI written")


def metrics_by_period(vectors, daily, log=print):
    """Every metric of every strategy by test quarter, and profit and capture by month."""
    q = quarter_of(daily["delivery_day"])
    vq = quarter_of(vectors["delivery_day"])
    by_q = []
    for first in backtest.TEST_QUARTERS:
        d, v = daily[q == first].reset_index(drop=True), vectors[vq == first]
        for s in STRATEGIES:
            by_q.append({"quarter_start": first, **evaluate.value_summary(d, s), **evaluate.forecast_summary(v, s)})
    pd.DataFrame(by_q).to_csv(EXPLORE_DIR / "metrics_by_quarter.csv", index=False)

    month = pd.to_datetime(daily["delivery_day"]).dt.strftime("%Y-%m")
    by_m = daily.groupby(month)[STRATEGIES].sum()
    out = by_m.add_suffix(" profit (EUR)")
    for s in STRATEGIES[1:]:
        out[f"{s} capture"] = by_m[s] / by_m["S-perfect"]
    out.to_csv(EXPLORE_DIR / "profit_by_month.csv", index_label="month")

    # Distribution of daily profit (EUR per day) over the test year.
    levels = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    dist = daily[STRATEGIES].quantile(levels).T
    dist.columns = ["min" if x == 0 else "max" if x == 1 else f"P{round(100 * x)}" for x in levels]
    dist.to_csv(EXPLORE_DIR / "daily_profit_distribution.csv", index_label="strategy")
    log("stability: quarterly and monthly tables written")


# --- B. Accuracy vs value (validation configuration table) -------------------------------------
def accuracy_vs_value():
    """Rank correlation (Spearman) of RMSE and of within-day Spearman rho with validation
    profit, across every single-seed fit of the S2 search, by model and pooled."""
    from scipy.stats import spearmanr
    cfg = pd.read_csv(config.RESULTS_DIR / "validation_configs.csv", dtype={"seed": str})
    fits = cfg[cfg["seed"] != "avg"]
    rows = []
    for group, g in [("all fits", fits)] + list(fits.groupby("model")):
        for metric in ["vector_rmse_eur_mwh", "vector_spearman_rho_mean"]:
            r, p = spearmanr(g[metric], g["profit_eur_per_mw_year"])
            rows.append({"group": group, "metric": metric, "fits": len(g), "rank_corr_with_profit": r, "p_value": p})
    out = pd.DataFrame(rows)
    out.to_csv(EXPLORE_DIR / "accuracy_vs_value.csv", index=False)
    return fits, out


# --- C. Where profit is lost: oracle decomposition ----------------------------------------------
ORACLES = {
    "true order, model values": ("{f}-reg", "price"),
    "reg order, true values": ("price", "{f}-reg"),
    "rank order, true values": ("price", "{f}-rank"),
}


def oracle_vectors(vectors):
    """For each family, the price model's values put in the true order, and the true prices put
    in the price model's or the ranker's order (strategies.rank_reassign)."""
    from bessrank import strategies
    out = vectors.copy()
    for fam in ["xgb", "lstm"]:
        for name, (values, scores) in ORACLES.items():
            out[f"{fam}: {name}"] = strategies.rank_reassign_frame(
                vectors, values.format(f=fam), scores.format(f=fam))
    return out


def where_profit_is_lost(vectors, log=print):
    """Profit of the oracle vectors by quarter, and the gap of each "-reg"/"-rank" strategy to
    perfect foresight split into an order part (true values, model order), a value part (model
    values, true order) and the rest (interaction). Also forecast metrics by quarter."""
    ov = oracle_vectors(vectors)
    names = [c for c in ov.columns if ": " in c]
    daily = solve_profits(ov, STRATEGIES[3:] + names)
    daily.to_csv(EXPLORE_DIR / "oracle_daily_profit.csv", index=False)
    q = quarter_of(daily["delivery_day"])
    rows = []
    for label, mask in [("test year", np.ones(len(daily), bool))] + [(str(f), q == f) for f in backtest.TEST_QUARTERS]:
        d = daily[mask]
        perfect = d["S-perfect"].mean()
        for fam in ["xgb", "lstm"]:
            value_loss = perfect - d[f"{fam}: true order, model values"].mean()
            for obj in ["reg", "rank"]:
                total = perfect - d[f"S-{fam}-{obj}"].mean()
                order_loss = perfect - d[f"{fam}: {obj} order, true values"].mean()
                rows.append({"period": label, "strategy": f"S-{fam}-{obj}", "days": int(mask.sum()),
                             "perfect_eur_day": perfect, "strategy_eur_day": d[f"S-{fam}-{obj}"].mean(),
                             "gap_eur_day": total, "order_only_gap_eur_day": order_loss,
                             "values_only_gap_eur_day": value_loss,
                             "interaction_eur_day": total - order_loss - value_loss})
    pd.DataFrame(rows).to_csv(EXPLORE_DIR / "oracle_decomposition.csv", index=False)

    vq = quarter_of(vectors["delivery_day"])
    fm = []
    for label, mask in [("test year", np.ones(len(vectors), bool))] + [(str(f), vq == f) for f in backtest.TEST_QUARTERS]:
        for s in STRATEGIES[3:]:
            fm.append({"period": label, "strategy": s, **evaluate.forecast_summary(vectors[mask], s)})
    pd.DataFrame(fm).to_csv(EXPLORE_DIR / "forecast_metrics_by_quarter.csv", index=False)

    # Does the BiLSTM price model already order the hours as well as the rankers? Paired
    # differences of the daily within-day Spearman rho, with the same block bootstrap.
    rho = pd.DataFrame({s: evaluate.within_day_spearman(vectors[s], vectors["price"], vectors["delivery_day"])
                        for s in STRATEGIES[3:]}).sort_index()
    rows = []
    for a, b in [("S-xgb-rank", "S-xgb-reg"), ("S-lstm-reg", "S-xgb-reg"), ("S-xgb-rank", "S-lstm-reg"),
                 ("S-lstm-rank", "S-lstm-reg")]:
        diff = (rho[a] - rho[b]).dropna()
        mean, lo, hi = evaluate.block_bootstrap_ci(diff)
        rows.append({"comparison": f"{a} - {b}", "days": len(diff), "mean_daily_spearman_diff": mean,
                     "ci95_low": lo, "ci95_high": hi})
    pd.DataFrame(rows).to_csv(EXPLORE_DIR / "spearman_differences.csv", index=False)

    # LSTM, Jul-Sep 2026: how concentrated is the rank - reg loss?
    d = daily[q == backtest.TEST_QUARTERS[-1]].copy()
    d["diff"] = d["S-lstm-rank"] - d["S-lstm-reg"]
    d["order_diff_true_values"] = d["lstm: rank order, true values"] - d["lstm: reg order, true values"]
    worst = d.nsmallest(10, "diff")[["delivery_day", "S-perfect", "S-lstm-reg", "S-lstm-rank", "diff",
                                     "order_diff_true_values"]]
    worst.to_csv(EXPLORE_DIR / "lstm_q3_worst_days.csv", index=False)
    log("where profit is lost: written")
    return daily


# --- E. Figures ---------------------------------------------------------------------------------
# Light surface, ink and series colours of one validated categorical palette (fixed order).
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
STRATEGY_STYLE = {
    "S-perfect": dict(color=INK, linestyle="--"), "S-naive-1d": dict(color="#8a8985"),
    "S-naive-7d": dict(color="#8a8985", linestyle=":"), "S-xgb-reg": dict(color=BLUE),
    "S-xgb-rank": dict(color=ORANGE), "S-lstm-reg": dict(color=AQUA), "S-lstm-rank": dict(color=YELLOW),
}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _figure(ncols=1, nrows=1, size=(10, 4.2), **kw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows, ncols, figsize=size, facecolor=SURFACE, **kw)
    for ax in np.atleast_1d(axes).ravel():
        _style(ax)
    return plt, fig, axes


def fig_cumulative_profit():
    """Cumulative test-year profit of every strategy (official daily profits), and the
    cumulative rank - reg difference of each family."""
    import matplotlib.dates as mdates
    daily = pd.read_csv(config.RESULTS_DIR / "test_daily_profit.csv", parse_dates=["delivery_day"])
    plt, fig, (ax, ax2) = _figure(2, size=(11, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
    for s in STRATEGIES:
        ax.plot(daily["delivery_day"], daily[s].cumsum() / 1000, linewidth=1.6 if "xgb" in s or "lstm" in s else 1.2,
                label=s, **STRATEGY_STYLE[s])
    ax.set_ylabel("Cumulative profit (kEUR per MW)", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    ax.set_title("Test year: cumulative profit by strategy", color=INK, fontsize=10, loc="left")
    for (a, b), color in zip(config.PAIRS, [ORANGE, YELLOW]):
        cum = (daily[a] - daily[b]).cumsum()
        ax2.plot(daily["delivery_day"], cum, color=color, linewidth=2)
        ax2.annotate(f"{a.replace('S-', '')} - {b.replace('S-', '')}\n{cum.iloc[-1]:+.0f} EUR",
                     (daily["delivery_day"].iloc[-1], cum.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                     color=INK, fontsize=8, va="center")
    ax2.axhline(0, color=INK_2, linewidth=0.8)
    ax2.set_ylabel("Cumulative difference (EUR per MW)", color=INK_2, fontsize=9)
    ax2.set_title("Same values, ranker's order: rank - reg", color=INK, fontsize=10, loc="left")
    ax2.set_xlim(right=daily["delivery_day"].iloc[-1] + pd.Timedelta(days=90))
    for a in (ax, ax2):
        for q in backtest.TEST_QUARTERS[1:]:
            a.axvline(pd.Timestamp(q), color=GRID, linewidth=1.2, zorder=0)
        a.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    fig.text(0.01, 0.01, "Simulated 1 MW / 2 MWh battery, day-ahead market only, price-taker. "
             "Delivery days 2025-10-01 to 2026-09-30. Data: Bundesnetzagentur | SMARD.de",
             color=INK_2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(EXPLORE_DIR / "fig_cumulative_profit.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_accuracy_vs_value(fits, corr, out_path=None):
    """Validation profit against RMSE and against within-day Spearman rho, every single-seed
    fit of the S2 search. Colour = family, marker = objective."""
    plt, fig, axes = _figure(2, size=(11, 4.4), sharey=True)
    marks = {"xgb-reg": (BLUE, "o"), "xgb-rank": (BLUE, "^"), "lstm-reg": (ORANGE, "o"), "lstm-rank": (ORANGE, "^")}
    pooled = corr[corr["group"] == "all fits"].set_index("metric")["rank_corr_with_profit"]
    for ax, metric, label in [(axes[0], "vector_rmse_eur_mwh", "RMSE of the price vector (EUR/MWh)"),
                              (axes[1], "vector_spearman_rho_mean", "Mean within-day Spearman rho")]:
        for m, (color, marker) in marks.items():
            g = fits[fits["model"] == m]
            ax.scatter(g[metric], g["profit_eur_per_mw_year"] / 1000, color=color, marker=marker, s=46,
                       edgecolors=SURFACE, linewidths=1.5, label=m, zorder=3)
        ax.set_xlabel(label, color=INK_2, fontsize=9)
        ax.set_title(f"Rank correlation with profit: {pooled[metric]:+.2f} (58 fits)", color=INK,
                     fontsize=10, loc="left")
    axes[0].set_ylabel("Validation profit (kEUR per MW per year)", color=INK_2, fontsize=9)
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right",
                   title="triangle = ranker (reg values in its order)", title_fontsize=7.5)
    fig.text(0.01, 0.01, "Validation year 2024-10-01 to 2025-09-30, every configuration and seed of the "
             "S2 search. Data: Bundesnetzagentur | SMARD.de", color=INK_2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path or EXPLORE_DIR / "fig_accuracy_vs_value.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_example_day(vectors, daily, out_path=None):
    """The test day where S-xgb-rank and S-xgb-reg differ most in profit: the actual prices,
    both price vectors, and both schedules (net energy sold per hour)."""
    diff = daily["S-xgb-rank"] - daily["S-xgb-reg"]
    i = int(diff.abs().idxmax())
    day = daily.loc[i, "delivery_day"]
    v = vectors[vectors["delivery_day"] == day].sort_values("ts_utc")
    hours = np.arange(len(v))
    plt, fig, (ax, ax2) = _figure(1, 2, size=(9, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
    for col, color, lw, label in [("price", INK, 2, "actual price"), ("S-xgb-reg", BLUE, 1.8, "S-xgb-reg vector"),
                                  ("S-xgb-rank", ORANGE, 1.8, "S-xgb-rank vector (same values, ranker's order)")]:
        ax.step(hours, v[col], where="mid", color=color, linewidth=lw, label=label)
    ax.set_ylabel("EUR/MWh", color=INK_2, fontsize=9)
    top = v[["price", "S-xgb-reg", "S-xgb-rank"]].to_numpy().max()
    ax.set_ylim(top=top + 0.25 * (top - ax.get_ylim()[0]))  # headroom for the legend
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper center", ncol=3)
    ax.set_title(f"{day}: S-xgb-rank {daily.loc[i, 'S-xgb-rank']:.0f} EUR vs S-xgb-reg "
                 f"{daily.loc[i, 'S-xgb-reg']:.0f} EUR (perfect foresight {daily.loc[i, 'S-perfect']:.0f} EUR)",
                 color=INK, fontsize=10, loc="left")
    width = 0.38
    for k, (s, color) in enumerate([("S-xgb-reg", BLUE), ("S-xgb-rank", ORANGE)]):
        sched = battery.solve_day(v[s].to_numpy())
        ax2.bar(hours + (k - 0.5) * width, sched.discharge - sched.charge, width=width - 0.04, color=color, label=s)
    ax2.axhline(0, color=INK_2, linewidth=0.8)
    ax2.set_ylabel("Net energy sold (MWh)\n< 0: charging", color=INK_2, fontsize=9)
    ax2.set_xlabel("Hour of the delivery day (Europe/Berlin)", color=INK_2, fontsize=9)
    ax2.set_xticks(hours[::2])
    ax2.set_xticklabels(local_hour(v)[::2])
    ax2.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.text(0.01, 0.01, "Simulated 1 MW / 2 MWh battery, day-ahead market only, price-taker. "
             "Data: Bundesnetzagentur | SMARD.de", color=INK_2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path or EXPLORE_DIR / "fig_example_day.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return {"day": str(day), "xgb_rank_eur": float(daily.loc[i, "S-xgb-rank"]),
            "xgb_reg_eur": float(daily.loc[i, "S-xgb-reg"]), "perfect_eur": float(daily.loc[i, "S-perfect"])}


FIGURES_DIR = config.RESULTS_DIR / "figures"  # the three README figures


def fig_hero(out_path):
    """README Fig. 1: cumulative rank - reg profit difference over the test year, both families,
    from the official daily profits. Differences, not levels: at this scale the level lines of
    the four model strategies lie on top of each other."""
    import matplotlib.dates as mdates
    daily = pd.read_csv(config.RESULTS_DIR / "test_daily_profit.csv", parse_dates=["delivery_day"])
    plt, fig, ax = _figure(size=(9.5, 4.8))
    last = daily["delivery_day"].iloc[-1]
    for (a, b), color, name in zip(config.PAIRS, [BLUE, ORANGE], ["XGBoost", "BiLSTM"]):
        cum = (daily[a] - daily[b]).cumsum()
        ax.plot(daily["delivery_day"], cum, color=color, linewidth=2.2)
        ax.annotate(f"{name}\n{cum.iloc[-1]:+,.0f} EUR/MW", (last, cum.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", color=INK, fontsize=9, va="center")
    ax.axhline(0, color=INK_2, linewidth=0.8)
    for q in backtest.TEST_QUARTERS[1:]:
        ax.axvline(pd.Timestamp(q), color=GRID, linewidth=1.2, zorder=0)
    ax.set_xlim(right=last + pd.Timedelta(days=75))  # room for the end labels
    ax.set_xticks([t for t in ax.get_xticks() if t <= mdates.date2num(last)])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.set_ylabel("Cumulative profit, rank minus price model (EUR per MW)", color=INK_2, fontsize=9)
    ax.set_title("Training to rank the hours vs training to predict prices: cumulative profit difference",
                 color=INK, fontsize=10.5, loc="left")
    fig.text(0.01, 0.01, "Test year 2025-10-01 to 2026-09-30, pre-registered and run once. Simulated 1 MW / 2 MWh "
             "battery, German day-ahead market, price-taker.\nData: Bundesnetzagentur | SMARD.de",
             color=INK_2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def readme_figures(log=print):
    """The three README figures in results/figures/ (python -m bessrank.run report)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_hero(FIGURES_DIR / "fig1_cumulative_difference.png")
    hourly = data.load_hourly(include_test=True)
    vectors = test_vectors(hourly, load_forecasts())
    daily = pd.read_csv(config.RESULTS_DIR / "test_daily_profit.csv")
    daily["delivery_day"] = pd.to_datetime(daily["delivery_day"]).dt.date
    fig_example_day(vectors, daily, FIGURES_DIR / "fig2_example_day.png")
    fits, corr = accuracy_vs_value()
    fig_accuracy_vs_value(fits, corr, FIGURES_DIR / "fig3_accuracy_vs_value.png")
    log(f"README figures written to {FIGURES_DIR.relative_to(config.ROOT)}/")


# --- Driver -------------------------------------------------------------------------------------
def analyses(log=print):
    """Every analysis after `forecasts` and `xgb-variants` have run (their files are in data/)."""
    hourly, feats = load_test_inputs()
    vectors = test_vectors(hourly, load_forecasts())
    daily = solve_profits(vectors, STRATEGIES)
    sensitivity(vectors, log=log)
    variant_analysis(hourly, vectors, daily, log=log)
    psi_report(feats, log=log)
    metrics_by_period(vectors, daily, log=log)
    fits, corr = accuracy_vs_value()
    where_profit_is_lost(vectors, log=log)
    fig_cumulative_profit()
    fig_accuracy_vs_value(fits, corr)
    example = fig_example_day(vectors, daily)
    write_json("example_day.json", example)
    log(f"figures written; example day {example}")
    write_summary()


# --- Summary (display only) -----------------------------------------------------------------
def _pct(x):
    return (100 * x).round(1).astype(str) + "%"


def write_summary():
    """results/explore/summary.md: every exploratory table, rounded for display only. The raw
    numbers are in the CSV files next to it."""
    from bessrank.data import _markdown_table as md

    def r(name):
        return pd.read_csv(EXPLORE_DIR / name)
    out = ["# Exploratory analyses after the test run (PLAN.md §8, session S4)", "",
           "Exploratory: these are not tests. The pre-registered verdicts are in "
           "`results/test_summary.md` and are not changed by anything here. Simulated 1 MW / 2 MWh "
           "battery (unless stated), day-ahead market only, price-taker. CIs: the §6 moving-block "
           "bootstrap (7-day blocks, 10,000 resamples, seed 20261004). Rounded for display; the "
           "CSV files next to this one hold the raw numbers.", ""]

    chk = json.loads((EXPLORE_DIR / "regenerated_forecasts_check.json").read_text())
    out += ["## Test-year forecasts used here", "",
            f"Regenerated with the frozen backtest code: {chk['fits_with_same_n_iter']} of {chk['fits']} "
            "fits found the same number of trees or epochs as the official run; largest absolute "
            "difference in daily profit against `results/test_daily_profit.csv`, over all strategies "
            f"and days: {max(chk['max_abs_daily_profit_diff_eur'].values()):.2e} EUR.", ""]

    s = r("sensitivity_strategies.csv")
    s = s[s["strategy"] != "S-perfect"].assign(
        profit=lambda d: d["profit_eur_per_mw_year"].round(0).astype(int), capture=lambda d: _pct(d["capture_rate"]))
    p = r("sensitivity_pairs.csv")
    perfect = r("sensitivity_strategies.csv").query("strategy == 'S-perfect'").set_index("setting")
    p["perfect"] = p["setting"].map(perfect["profit_eur_per_mw_year"]).round(0).astype(int)
    p = p.assign(diff=p["mean_daily_diff_eur"].round(2), ci=p["ci95_low_eur"].round(2).astype(str) + " to "
                 + p["ci95_high_eur"].round(2).astype(str), annual=p["annual_diff_eur_per_mw"].round(0).astype(int),
                 gap=_pct(p["share_of_gap_to_perfect_closed"]))
    out += ["## A. Sensitivity: the same test forecasts re-dispatched (no retraining)", "",
            "Other durations keep 1 MW, start and end each day at 50% charge and allow one full cycle per day.", "",
            md(s.pivot(index="setting", columns="strategy", values="profit").reset_index()
               .rename(columns={"setting": "Setting (EUR/MW/year)"})), "",
            md(s.pivot(index="setting", columns="strategy", values="capture").reset_index()
               .rename(columns={"setting": "Setting (capture)"})), "",
            md(p[["setting", "comparison", "perfect", "diff", "ci", "annual", "gap"]].rename(columns={
                "setting": "Setting", "comparison": "Comparison", "perfect": "S-perfect (EUR/MW/year)",
                "diff": "Mean daily diff (EUR)", "ci": "95% CI", "annual": "Annual diff (EUR/MW)",
                "gap": "Share of reg's gap to perfect closed"})), ""]

    v = r("xgb_variants_h1.csv")
    v = v.assign(diff=v["mean_daily_diff_eur"].round(2), ci=v["ci95_low_eur"].round(2).astype(str) + " to "
                 + v["ci95_high_eur"].round(2).astype(str), annual=v["annual_diff_eur_per_mw"].round(0).astype(int),
                 reg=v["profit_xgb_reg_eur_per_mw_year"].round(0).astype(int),
                 rank=v["profit_xgb_rank_eur_per_mw_year"].round(0).astype(int), gap=_pct(v["share_of_gap_to_perfect_closed"]))
    fm = r("xgb_variants_forecast_metrics.csv")
    fm = fm.assign(rmse=fm["rmse_eur_mwh"].round(2), rho=fm["spearman_rho_mean"].round(3),
                   lo=_pct(fm["hit_rate_cheapest2"]), hi=_pct(fm["hit_rate_dearest2"]))
    vq = r("xgb_variants_h1_by_quarter.csv")
    vq["mean_daily_diff_eur"] = vq["mean_daily_diff_eur"].round(2)
    out += ["## A. Robustness (no TSO generation forecasts) and D. tree-count check: H1's statistic", "",
            "`official` is the pre-registered run (verdict: supported). `no_gen_forecasts` drops the 15 "
            "features built from TSO wind and PV forecasts (the load forecast stays) and repeats the §6 "
            "fit procedure. `fixed_trees` uses every feature and the validation tree counts of "
            "`results/selected_*.json` at every refit (XGB-reg 467/223/571, XGB-rank 83/88/151 for "
            "seeds 0/1/2), without early stopping. Exploratory; no verdict.", "",
            md(v[["variant", "reg", "rank", "diff", "ci", "annual", "gap"]].rename(columns={
                "variant": "Variant", "reg": "S-xgb-reg (EUR/MW/year)", "rank": "S-xgb-rank (EUR/MW/year)",
                "diff": "Mean daily diff (EUR)", "ci": "95% CI", "annual": "Annual diff (EUR/MW)",
                "gap": "Share of gap closed"})), "",
            md(fm[["variant", "strategy", "rmse", "rho", "lo", "hi"]].rename(columns={
                "variant": "Variant", "strategy": "Strategy", "rmse": "RMSE (EUR/MWh)", "rho": "Spearman rho",
                "lo": "Hit 2 cheapest", "hi": "Hit 2 dearest"})), "",
            md(vq[["variant", "quarter_start", "mean_daily_diff_eur", "trees_xgb_reg", "trees_xgb_rank"]].rename(columns={
                "variant": "Variant", "quarter_start": "Quarter from", "mean_daily_diff_eur": "Mean daily diff (EUR)",
                "trees_xgb_reg": "XGB-reg trees (s0/s1/s2)", "trees_xgb_rank": "XGB-rank trees (s0/s1/s2)"})), ""]

    psi_t = r("psi_top10.csv").round(3)
    out += ["## A. Stability: PSI of the 10 most important features", "",
            "Importance: total gain of the selected XGB-reg and XGB-rank (seed 0) fitted on the training "
            "period, each normalised to 1, averaged. PSI bins: training deciles plus a missing bin. "
            "Usual reading: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 large shift.", "",
            md(psi_t[["importance_rank", "feature", "importance", "uses_tso_generation_forecast",
                      "psi_test_vs_train", "psi_validation_vs_train"]].rename(columns={
                "importance_rank": "Rank", "feature": "Feature", "importance": "Importance",
                "uses_tso_generation_forecast": "TSO generation forecast", "psi_test_vs_train": "PSI test vs train",
                "psi_validation_vs_train": "PSI validation vs train"})), ""]

    q = r("metrics_by_quarter.csv")
    q = q.assign(profit=(q["mean_daily_profit_eur"] * q["days"]).round(0).astype(int), capture=_pct(q["capture_rate"]),
                 lo=_pct(q["hit_rate_cheapest2"]), hi=_pct(q["hit_rate_dearest2"]), losing=_pct(q["losing_day_share"]))
    q = q.round(3)
    out += ["## A. Every metric by quarter (test year)", "",
            "VaR 5% = P5 of daily profit, ES 5% = mean of the worst 5% of the quarter's days (5 days); "
            "both are profit levels, negative = loss.", "",
            md(q[["quarter_start", "strategy", "profit", "capture", "rmse_eur_mwh", "mae_eur_mwh", "spearman_rho_mean",
                  "lo", "hi", "var5_p5_of_daily_profit_eur", "es5_mean_of_worst5pct_days_eur", "losing",
                  "max_drawdown_eur"]].rename(columns={
                "quarter_start": "Quarter from", "strategy": "Strategy", "profit": "Profit (EUR/MW)",
                "capture": "Capture", "rmse_eur_mwh": "RMSE", "mae_eur_mwh": "MAE", "spearman_rho_mean": "Spearman rho",
                "lo": "Hit 2 cheapest", "hi": "Hit 2 dearest", "var5_p5_of_daily_profit_eur": "VaR 5%: P5 (EUR)",
                "es5_mean_of_worst5pct_days_eur": "ES 5%: worst 5% (EUR)", "losing": "Losing days",
                "max_drawdown_eur": "Max drawdown (EUR)"})), ""]

    m = r("profit_by_month.csv")
    cap = m[["month"] + [c for c in m.columns if c.endswith("capture")]].copy()
    for c in cap.columns[1:]:
        cap[c] = _pct(cap[c])
    cap.columns = [c.replace(" capture", "") for c in cap.columns]
    cap.insert(1, "S-perfect (EUR/MW)", m["S-perfect profit (EUR)"].round(0).astype(int))
    out += ["## A. Capture rate by month (test year)", "", md(cap), ""]
    dist = r("daily_profit_distribution.csv").round(1)
    out += ["## Distribution of daily profit (test year, EUR per day)", "", md(dist), ""]

    a = r("accuracy_vs_value.csv").round(3)
    out += ["## B. Accuracy vs value (validation year, every single-seed fit of the S2 search)", "",
            "Rank correlation (Spearman) between each accuracy metric and validation profit. A ranker's "
            "RMSE is that of its \"-rank\" vector (the selected price model's values in its order).", "",
            md(a.rename(columns={"group": "Fits", "metric": "Metric", "fits": "n",
                                 "rank_corr_with_profit": "Rank corr. with profit", "p_value": "p-value"})),
            "", "![accuracy vs value](fig_accuracy_vs_value.png)", ""]

    o = r("oracle_decomposition.csv")
    o = o.assign(**{c: o[c].round(2) for c in ["strategy_eur_day", "gap_eur_day", "order_only_gap_eur_day",
                                                "values_only_gap_eur_day", "interaction_eur_day"]})
    f = r("forecast_metrics_by_quarter.csv")
    f = f.assign(rmse=f["rmse_eur_mwh"].round(2), rho=f["spearman_rho_mean"].round(3),
                 lo=_pct(f["hit_rate_cheapest2"]), hi=_pct(f["hit_rate_dearest2"]))
    w = r("lstm_q3_worst_days.csv").round(2)
    out += ["## C. Where profit is lost", "",
            "Gap = perfect-foresight profit minus the strategy's, in EUR per day. Order only: the true "
            "prices put in the model's order (price model for \"-reg\", ranker for \"-rank\"). Values "
            "only: the price model's values put in the true order (the same for reg and rank). "
            "Interaction = gap - order only - values only.", "",
            md(o[["period", "strategy", "strategy_eur_day", "gap_eur_day", "order_only_gap_eur_day",
                  "values_only_gap_eur_day", "interaction_eur_day"]].rename(columns={
                "period": "Period", "strategy": "Strategy", "strategy_eur_day": "Profit (EUR/day)",
                "gap_eur_day": "Gap to perfect", "order_only_gap_eur_day": "Order only",
                "values_only_gap_eur_day": "Values only", "interaction_eur_day": "Interaction"})), "",
            "Forecast metrics of the price vectors by quarter:", "",
            md(f[["period", "strategy", "rmse", "rho", "lo", "hi"]].rename(columns={
                "period": "Period", "strategy": "Strategy", "rmse": "RMSE (EUR/MWh)", "rho": "Spearman rho",
                "lo": "Hit 2 cheapest", "hi": "Hit 2 dearest"})), "",
            "Paired differences of the daily within-day Spearman rho (test year, block bootstrap):", "",
            md(r("spearman_differences.csv").round(4)), "",
            "LSTM, Jul-Sep 2026: the 10 days where S-lstm-rank loses most against S-lstm-reg (EUR). "
            "`order_diff_true_values` = (true prices in the ranker's order) - (true prices in LSTM-reg's order).", "",
            md(w), ""]

    ex = json.loads((EXPLORE_DIR / "example_day.json").read_text())
    out += ["## E. Figures", "", "![cumulative profit](fig_cumulative_profit.png)", "",
            f"Day with the largest |S-xgb-rank - S-xgb-reg| profit difference: {ex['day']}.", "",
            "![example day](fig_example_day.png)", ""]
    if (EXPLORE_DIR / "cvar_frontier.csv").exists():
        cov = r("conformal_coverage.csv").round(3)
        fr = r("cvar_frontier.csv")
        fr = fr.assign(profit=fr["profit_eur_per_mw_year"].round(0).astype(int), capture=_pct(fr["capture_rate"]),
                       losing=_pct(fr["losing_day_share"])).round(2)
        out += ["## F. Conformal quantiles and the CVaR frontier (stretch)", "",
                "XGB-quantile (19 levels, XGB-reg's hyperparameters, seed 0) fitted once on the training "
                "period, calibrated by CQR on the validation year, applied to the test year without refits. "
                "200 scenarios per day: Gaussian copula with the within-day residual correlation of the "
                "training period, marginals from the calibrated quantiles (linear tails beyond 5% and 95%). "
                "Schedule: maximise mean scenario profit - lambda x CVaR 95% of the loss (Rockafellar-Uryasev), "
                "OR-Tools SCIP, re-solved with HiGHS; settled at the actual prices.", "",
                md(cov), "",
                md(fr[["strategy", "profit", "capture", "mean_daily_profit_eur", "var5_p5_of_daily_profit_eur",
                       "es5_mean_of_worst5pct_days_eur", "losing", "max_drawdown_eur"]].rename(columns={
                    "strategy": "Schedule", "profit": "Profit (EUR/MW/year)", "capture": "Capture",
                    "mean_daily_profit_eur": "Mean daily profit (EUR)", "var5_p5_of_daily_profit_eur": "VaR 5%: P5 (EUR)",
                    "es5_mean_of_worst5pct_days_eur": "ES 5%: worst 5% (EUR)", "losing": "Losing days",
                    "max_drawdown_eur": "Max drawdown (EUR)"})), "", "![CVaR frontier](fig_cvar_frontier.png)", ""]
    if (EXPLORE_DIR / "shap_top10.csv").exists():
        sh = r("shap_top10.csv")
        sh = sh.assign(share=_pct(sh["share_of_total"]), mean_abs_shap=sh["mean_abs_shap"].round(3))
        out += ["## F. SHAP: top 10 features on the test year (stretch)", "",
                "Mean |SHAP| (TreeSHAP) of the training-period XGB-reg (EUR/MWh) and XGB-rank (score units).", "",
                md(sh[["model", "rank", "feature", "mean_abs_shap", "share", "uses_tso_generation_forecast"]].rename(
                    columns={"model": "Model", "rank": "Rank", "feature": "Feature", "mean_abs_shap": "Mean |SHAP|",
                             "share": "Share of total", "uses_tso_generation_forecast": "TSO generation forecast"})), ""]
    (EXPLORE_DIR / "summary.md").write_text("\n".join(out))


# --- F. Stretch: conformal quantiles and the CVaR frontier ---------------------------------------
LAMBDAS = [0.0, 0.5, 1.0, 2.0, 5.0]
N_SCENARIOS = 200
SCENARIO_SEED = 20261005
QUANTILES_PARQUET = config.PREDICTIONS_DIR / "test_quantiles.parquet"


def local_hour(frame):
    """Clock hour in Europe/Berlin (0-23); a 25-hour day has 02:00 twice, a 23-hour day none."""
    return pd.to_datetime(frame["ts_utc"], utc=True).dt.tz_convert(config.MARKET_TZ).dt.hour.to_numpy()


def residual_correlation(train, median):
    """Within-day correlation of the median-forecast residuals between local hours (0-23),
    from the 24-hour training days; each hour's residuals are standardised first."""
    r = pd.DataFrame({"day": train["delivery_day"].to_numpy(), "hour": local_hour(train),
                      "res": train["price"].to_numpy() - median})
    r = r[r.groupby("day")["hour"].transform("size") == 24]
    wide = r.pivot(index="day", columns="hour", values="res")
    return np.corrcoef(((wide - wide.mean()) / wide.std()).to_numpy(), rowvar=False)


def quantile_function(q, u, levels=risk.QUANTILE_LEVELS):
    """Price at probability u from one hour's 19 sorted quantiles: linear between levels, and
    linear extrapolation with the outermost slopes below 0.05 and above 0.95."""
    lo_slope = (q[1] - q[0]) / (levels[1] - levels[0])
    hi_slope = (q[-1] - q[-2]) / (levels[-1] - levels[-2])
    out = np.interp(u, levels, q)
    out = np.where(u < levels[0], q[0] - (levels[0] - u) * lo_slope, out)
    return np.where(u > levels[-1], q[-1] + (u - levels[-1]) * hi_slope, out)


def day_scenarios(qday, hours, corr, rng, n=N_SCENARIOS):
    """n price scenarios for one day: Gaussian copula with the hours' residual correlation
    (a 25-hour day repeats the 02:00 row, a 23-hour day skips it), marginals from the
    calibrated quantiles."""
    from scipy.stats import norm
    c = corr[np.ix_(hours, hours)]
    w, V = np.linalg.eigh(c)
    zs = rng.standard_normal((n, len(hours))) @ (V * np.sqrt(np.clip(w, 0, None))).T
    u = norm.cdf(zs)
    return np.column_stack([quantile_function(qday[h], u[:, h]) for h in range(len(hours))])


def _cvar_job(job):
    days = []
    for day, qday, hours, actual, corr in job:
        rng = np.random.default_rng([SCENARIO_SEED, day.toordinal()])
        scen = day_scenarios(qday, hours, corr, rng)
        row = {"delivery_day": day}
        for lam in LAMBDAS:
            sched = battery.solve_day_cvar(scen, lam)
            row[f"lambda {lam:g}"] = battery.settle(sched, actual)
        days.append(row)
    return days


def conformal_cvar(log=print):
    """XGB-quantile fitted on the training period, calibrated by CQR on the validation year and
    applied to the test year (one fit, no quarterly refits). Then 200 scenarios per test day and
    the CVaR-constrained schedule for each lambda, settled at the actual prices."""
    hourly, feats = load_test_inputs()
    train = models.training_rows(feats, config.TRAIN_START, config.TRAIN_END)
    val = models.forecast_rows(feats, config.VAL_START, config.VAL_END)
    test = models.forecast_rows(feats, config.TEST_START, config.TEST_END)
    params = json.loads((config.RESULTS_DIR / "selected_xgb-reg.json").read_text())["params"]
    started = time.time()
    n_trees, (q_train, q_val, q_test) = models.fit_predict_quantiles(params, train, [train, val, test])
    log(f"XGB-quantile: {n_trees} trees, {time.time() - started:.0f} s")
    q_val, q_test = np.sort(q_val, axis=1), np.sort(q_test, axis=1)
    corrections = risk.cqr_corrections(q_val, val["price"])
    q_test_c = risk.apply_cqr(q_test, corrections)
    rows = []
    for label, q, y in [("validation, raw", q_val, val["price"]), ("test, raw", q_test, test["price"]),
                        ("test, CQR-calibrated", q_test_c, test["price"])]:
        rows.append({"set": label, "pinball_loss_eur_mwh": risk.pinball_loss(q, y),
                     **{f"coverage {k:.0%}": v for k, v in risk.interval_coverage(q, y).items()}})
    pd.DataFrame(rows).to_csv(EXPLORE_DIR / "conformal_coverage.csv", index=False)
    write_json("conformal_corrections.json", {"n_trees": n_trees, "corrections_eur_mwh": corrections})
    out = test[models.KEYS].copy()
    out[[f"q{a:.2f}" for a in risk.QUANTILE_LEVELS]] = q_test_c
    out.to_parquet(QUANTILES_PARQUET, index=False)

    corr = residual_correlation(train, np.median(np.sort(q_train, axis=1), axis=1))
    jobs = [[] for _ in range(4)]
    for k, (day, g) in enumerate(test.assign(row=np.arange(len(test))).groupby("delivery_day", sort=True)):
        jobs[k % 4].append((day, q_test_c[g["row"].to_numpy()], local_hour(g), g["price"].to_numpy(), corr))
    started = time.time()
    with ProcessPoolExecutor(4) as pool:
        daily = pd.DataFrame([r for part in pool.map(_cvar_job, jobs) for r in part])
    daily = daily.sort_values("delivery_day").reset_index(drop=True)
    log(f"CVaR schedules: {len(daily)} days x {len(LAMBDAS)} lambdas, SCIP + HiGHS, {time.time() - started:.0f} s")
    official = pd.read_csv(config.RESULTS_DIR / "test_daily_profit.csv")
    daily["S-perfect"] = official["S-perfect"].to_numpy()
    daily["S-xgb-reg"] = official["S-xgb-reg"].to_numpy()
    if not (pd.to_datetime(official["delivery_day"]).dt.date.to_numpy() == daily["delivery_day"].to_numpy()).all():
        raise ValueError("CVaR days are not aligned with the official test days")
    evaluate.check_perfect_is_upper_bound(daily, [f"lambda {lam:g}" for lam in LAMBDAS])
    daily.to_csv(EXPLORE_DIR / "cvar_daily_profit.csv", index=False)
    frontier = pd.DataFrame([{"strategy": c, **evaluate.value_summary(daily, c)}
                             for c in [f"lambda {lam:g}" for lam in LAMBDAS] + ["S-xgb-reg"]])
    frontier.to_csv(EXPLORE_DIR / "cvar_frontier.csv", index=False)
    fig_cvar_frontier(frontier)
    log("conformal + CVaR: written")


def fig_cvar_frontier(frontier):
    """Realized mean daily profit against realized ES 5% on the test year, one point per lambda."""
    plt, fig, ax = _figure(size=(6.4, 4.4))
    lam = frontier[frontier["strategy"].str.startswith("lambda")]
    ax.plot(lam["es5_mean_of_worst5pct_days_eur"], lam["mean_daily_profit_eur"], color=BLUE, linewidth=2,
            marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
    for _, r in lam.iterrows():
        ax.annotate(r["strategy"].replace("lambda", "λ ="), (r["es5_mean_of_worst5pct_days_eur"], r["mean_daily_profit_eur"]),
                    xytext=(6, -10), textcoords="offset points", color=INK, fontsize=8)
    ref = frontier[frontier["strategy"] == "S-xgb-reg"].iloc[0]
    ax.scatter([ref["es5_mean_of_worst5pct_days_eur"]], [ref["mean_daily_profit_eur"]], color=ORANGE, marker="D", s=46,
               zorder=3, edgecolors=SURFACE, linewidths=1.5)
    ax.annotate("S-xgb-reg (point forecast)", (ref["es5_mean_of_worst5pct_days_eur"], ref["mean_daily_profit_eur"]),
                xytext=(6, 4), textcoords="offset points", color=INK, fontsize=8)
    ax.set_xlabel("ES 5%: mean profit of the worst 5% of days (EUR)", color=INK_2, fontsize=9)
    ax.set_ylabel("Mean daily profit (EUR)", color=INK_2, fontsize=9)
    ax.set_title("Test year: CVaR-constrained schedules from conformal scenarios", color=INK, fontsize=10, loc="left")
    fig.text(0.01, 0.01, "Simulated 1 MW / 2 MWh battery, day-ahead only, price-taker. Exploratory. "
             "Data: Bundesnetzagentur | SMARD.de", color=INK_2, fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(EXPLORE_DIR / "fig_cvar_frontier.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
