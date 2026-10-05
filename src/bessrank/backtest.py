"""The pre-registered test run (PLAN.md §6) and its rehearsal on the validation year.

1. At the start of each quarter, refit every model on every training day before it (from
   2019-01-01, with the §2 rule for dropping days), with the frozen hyperparameters of
   results/selected_*.json and the §7 fit procedure (early stopping on the last 3 months of
   that window, then a refit on the whole window), for seeds 0, 1 and 2.
2. Forecast the quarter's days with that quarter's models only, averaging the 3 seeds.
3. Build the 7 strategies' price vectors, solve every day with HiGHS and SCIP, settle.
4. H1-H3 verdicts with moving-block bootstrap CIs, the results table, the quarterly
   breakdown.

The rehearsal runs the same code on the validation year (quarters from 2024-10-01), so the
pipeline runs end to end before the test lock is opened. Its numbers are exploratory.
"""
import json
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta

import numpy as np
import pandas as pd

from bessrank import config, evaluate, features, models, strategies
from bessrank.config import MODELS, PAIRS, STRATEGIES

TEST_QUARTERS = [date(2025, 10, 1), date(2026, 1, 1), date(2026, 4, 1), date(2026, 7, 1)]
REHEARSAL_QUARTERS = [date(2024, 10, 1), date(2025, 1, 1), date(2025, 4, 1), date(2025, 7, 1)]
H2_MARGIN = 0.01  # H2: 1% of S-xgb-reg's RMSE over the year (PLAN.md §6)
# Two fits side by side, each with the 2 threads used in the S2 search (models.N_THREADS,
# lstm.N_THREADS), so every fit runs exactly as it did on validation.
FIT_WORKERS = 2


def quarters(starts, last_day):
    """(first day, last day) of each quarter."""
    ends = [s - timedelta(days=1) for s in starts[1:]] + [last_day]
    return list(zip(starts, ends))


def selected_params(kind):
    """Frozen hyperparameters from the validation search (PLAN.md §6)."""
    return json.loads((config.RESULTS_DIR / f"selected_{kind}.json").read_text())["params"]


# --- 1-2. Quarterly refits and forecasts ------------------------------------------------------
def _fit_job(job):
    """Fit one model with every seed on one training window and forecast one quarter."""
    kind, params, train, predict, seeds = job
    if kind.startswith("xgb"):
        fit = models.fit_predict
    else:
        from bessrank import lstm  # torch is imported only in the processes that need it
        fit = lstm.fit_predict
    out = []
    for seed in seeds:
        started = time.time()
        n_iter, pred = fit(kind, params, train, predict, seed)
        out.append((seed, n_iter, pred, time.time() - started))
    return out


def refit_and_forecast(feats, quarter_list, log=print):
    """Steps 1-2 for every quarter and model.

    Returns (forecasts, refits): one row per forecast hour with the seed-averaged forecast of
    each model (prices for "-reg", scores for "-rank") and each seed's forecast
    ("<model>:s<seed>"); one row per fit with the number of trees or epochs found.
    """
    params = {kind: selected_params(kind) for kind in MODELS}
    jobs, info, frames = [], [], {}
    for first, last in quarter_list:
        train = models.training_rows(feats, config.TRAIN_START, first - timedelta(days=1))
        predict = models.forecast_rows(feats, first, last)
        frames[first] = predict[models.KEYS].copy()
        n_dropped = len(features.incomplete_days(feats, config.TRAIN_START, first - timedelta(days=1)))
        # LSTM jobs first: they are the longest, so the two workers end at about the same time.
        for kind in sorted(MODELS, key=lambda k: not k.startswith("lstm")):
            jobs.append((kind, params[kind], train, predict, config.SEEDS))
            info.append({"quarter_start": first, "quarter_end": last, "model": kind,
                         "train_first_day": train["delivery_day"].min(),
                         "train_last_day": train["delivery_day"].max(),
                         "train_days": int(train["delivery_day"].nunique()),
                         "train_days_dropped": n_dropped,
                         "forecast_days": int(predict["delivery_day"].nunique())})

    log(f"{len(jobs)} jobs ({len(quarter_list)} quarters x {len(MODELS)} models, "
        f"{len(config.SEEDS)} seeds each) on {FIT_WORKERS} workers ...")
    refits = []
    with ProcessPoolExecutor(FIT_WORKERS) as pool:
        for meta, result in zip(info, pool.map(_fit_job, jobs)):
            frame, kind = frames[meta["quarter_start"]], meta["model"]
            for seed, n_iter, pred, seconds in result:
                frame[f"{kind}:s{seed}"] = pred  # same row order as `predict`
                refits.append({**meta, "seed": seed, "n_iter": int(n_iter), "fit_seconds": seconds})
            # PLAN.md §4: mean prediction for price models, mean score for rankers.
            frame[kind] = frame[[f"{kind}:s{s}" for s in config.SEEDS]].mean(axis=1)
            log(f"{meta['quarter_start']} {kind}: n_iter {[n for _, n, _, _ in result]}, "
                f"{sum(sec for *_, sec in result):.0f} s")
    forecasts = pd.concat([frames[first] for first, _ in quarter_list], ignore_index=True)
    columns = models.KEYS + MODELS + [f"{m}:s{s}" for m in MODELS for s in config.SEEDS]
    return forecasts[columns].sort_values("ts_utc").reset_index(drop=True), pd.DataFrame(refits)


# --- 3. Strategies ----------------------------------------------------------------------------
def strategy_vectors(hourly, forecasts, first_day, last_day):
    """Hourly price vectors of the 7 strategies over [first_day, last_day] (PLAN.md §5)."""
    vectors = strategies.baseline_price_vectors(hourly)
    day = vectors["delivery_day"]
    vectors = vectors[(day >= first_day) & (day <= last_day)].sort_values("ts_utc").reset_index(drop=True)
    if not forecasts["ts_utc"].reset_index(drop=True).equals(vectors["ts_utc"]):
        raise ValueError("forecasts are not aligned with the hours of the period")
    vectors = pd.concat([vectors, forecasts[MODELS]], axis=1)
    vectors["S-xgb-reg"] = vectors["xgb-reg"]
    vectors["S-lstm-reg"] = vectors["lstm-reg"]
    # "-rank": the price model's values for the day, reassigned in the ranker's order (§5).
    vectors["S-xgb-rank"] = strategies.rank_reassign_frame(vectors, "xgb-reg", "xgb-rank")
    vectors["S-lstm-rank"] = strategies.rank_reassign_frame(vectors, "lstm-reg", "lstm-rank")
    if vectors[STRATEGIES].isna().any().any():
        raise ValueError("a strategy's price vector has a missing value")
    return vectors


# --- 4. Results -------------------------------------------------------------------------------
def strategy_table(vectors, daily):
    """Every reported metric of every strategy over the whole period (PLAN.md §6)."""
    return pd.DataFrame([{**evaluate.value_summary(daily, s), **evaluate.forecast_summary(vectors, s)}
                         for s in STRATEGIES])


def hypotheses(vectors, daily):
    """H1 (confirmatory), H2 and H3 (secondary) with their verdicts, and the LSTM RMSE
    difference (descriptive, no verdict). Statistics and rules: PLAN.md §6."""
    paired = evaluate.paired_differences(vectors, daily).set_index("comparison")
    xgb, lstm = (f"{a} - {b}" for a, b in PAIRS)
    rmse_reg = {s: float(np.sqrt(((vectors[s] - vectors["price"]) ** 2).mean())) for s in ("S-xgb-reg", "S-lstm-reg")}

    def profit_row(name, role, comparison, reg):
        p = paired.loc[comparison]
        lo, hi = p["ci95_low_eur"], p["ci95_high_eur"]
        return {"hypothesis": name, "role": role, "comparison": comparison,
                "statistic": "mean daily profit difference (EUR/day)",
                "estimate": p["mean_daily_profit_diff_eur"], "ci95_low": lo, "ci95_high": hi,
                "margin": np.nan, "verdict": evaluate.verdict_superiority(lo, hi),
                "annual_diff_eur_per_mw": p["annual_profit_diff_eur_per_mw"],
                "relative_to_reg_profit": p["mean_daily_profit_diff_eur"] / daily[reg].mean(),
                "days_rank_better": p["days_rank_better"], "days_reg_better": p["days_reg_better"],
                "days_identical_price_vector": p["days_identical_price_vector"]}

    def rmse_row(name, role, comparison, reg, test):
        p = paired.loc[comparison]
        lo, hi = p["rmse_diff_ci95_low"], p["rmse_diff_ci95_high"]
        # The margin is fixed once from the whole year, not recomputed on each resample.
        margin = H2_MARGIN * rmse_reg[reg]
        return {"hypothesis": name, "role": role, "comparison": comparison,
                "statistic": "RMSE difference over all hours (EUR/MWh)",
                "estimate": p["rmse_diff_eur_mwh"], "ci95_low": lo, "ci95_high": hi, "margin": margin,
                "verdict": evaluate.verdict_equivalence(lo, hi, margin) if test else "descriptive (no test)",
                "reg_rmse_eur_mwh": rmse_reg[reg]}

    rows = [profit_row("H1", "confirmatory", xgb, "S-xgb-reg"),
            rmse_row("H2", "secondary", xgb, "S-xgb-reg", test=True),
            profit_row("H3", "secondary", lstm, "S-lstm-reg"),
            rmse_row("LSTM RMSE", "descriptive", lstm, "S-lstm-reg", test=False)]
    return pd.DataFrame(rows)


def quarterly_tables(vectors, daily, quarter_list):
    """Quarterly breakdown: every strategy, and the rank - reg differences (descriptive)."""
    rows, pair_rows = [], []
    for first, last in quarter_list:
        in_q = (daily["delivery_day"] >= first) & (daily["delivery_day"] <= last)
        d = daily[in_q].reset_index(drop=True)
        v = vectors[(vectors["delivery_day"] >= first) & (vectors["delivery_day"] <= last)]
        for s in STRATEGIES:
            f = evaluate.forecast_summary(v, s)
            rows.append({"quarter_start": first, "strategy": s, "days": int(len(d)),
                         "profit_eur_per_mw": float(d[s].sum() / config.POWER_MW),
                         "mean_daily_profit_eur": float(d[s].mean()),
                         "capture_rate": float(d[s].sum() / d["S-perfect"].sum()),
                         "rmse_eur_mwh": f["rmse_eur_mwh"], "mae_eur_mwh": f["mae_eur_mwh"],
                         "spearman_rho_mean": f["spearman_rho_mean"]})
        for a, b in PAIRS:
            diff = d[a] - d[b]
            ra, rb = (float(np.sqrt(((v[s] - v["price"]) ** 2).mean())) for s in (a, b))
            pair_rows.append({"quarter_start": first, "comparison": f"{a} - {b}", "days": int(len(d)),
                              "mean_daily_profit_diff_eur": float(diff.mean()),
                              "days_rank_better": int((diff > 1e-9).sum()),
                              "days_reg_better": int((diff < -1e-9).sum()),
                              "rmse_diff_eur_mwh": ra - rb})
    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


def run(hourly, quarter_list, log=print):
    """Steps 1-4. Returns a dict of result tables, the hourly vectors and the timings."""
    timings = {}
    started = time.time()
    feats = features.build_features(hourly)
    timings["features_s"] = time.time() - started

    t = time.time()
    forecasts, refits = refit_and_forecast(feats, quarter_list, log=log)
    timings["refits_s"] = time.time() - t

    t = time.time()
    first_day, last_day = quarter_list[0][0], quarter_list[-1][1]
    vectors = strategy_vectors(hourly, forecasts, first_day, last_day)
    log(f"Solving {len(STRATEGIES)} price vectors x {vectors['delivery_day'].nunique()} days with HiGHS + SCIP ...")
    daily = evaluate.parallel_daily_profits(vectors, STRATEGIES)
    timings["solve_s"] = time.time() - t

    table = strategy_table(vectors, daily)
    hyp = hypotheses(vectors, daily)
    quarterly, quarterly_pairs = quarterly_tables(vectors, daily, quarter_list)
    timings["total_s"] = time.time() - started
    return {"strategies": table, "hypotheses": hyp, "quarterly": quarterly,
            "quarterly_pairs": quarterly_pairs, "daily_profit": daily, "refits": refits,
            "forecasts": forecasts, "timings": timings,
            "solver_cross_checks": int(len(daily) * len(STRATEGIES))}
