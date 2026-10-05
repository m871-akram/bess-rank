"""Profit, capture rate, forecast metrics and the moving-block bootstrap (PLAN.md §5, §6)."""
import numpy as np
import pandas as pd

from bessrank import battery, config, risk

PERFECT = "S-perfect"


def daily_profits(vectors, strategies, bat=battery.DEFAULT_BATTERY, cross_check=True):
    """Optimise each day on each strategy's price vector and settle at the actual prices.

    `vectors` has one row per delivery hour with `price` (actual) and one column per strategy.
    Returns one row per delivery day and one profit column (EUR) per strategy. Stops the run if
    the solvers disagree or a strategy beats perfect foresight on any day (CLAUDE.md rule 7).
    """
    rows = []
    for day, hours in vectors.sort_values("ts_utc").groupby("delivery_day", sort=True):
        actual = hours["price"].to_numpy()
        row = {"delivery_day": day, "n_hours": len(hours)}
        for name in strategies:
            schedule = battery.solve_day(hours[name].to_numpy(), bat, cross_check=cross_check)
            row[name] = battery.settle(schedule, actual, bat)
        rows.append(row)
    daily = pd.DataFrame(rows)
    check_perfect_is_upper_bound(daily, [s for s in strategies if s != PERFECT])
    return daily


def check_perfect_is_upper_bound(daily, others, rtol=battery.AGREEMENT_RTOL):
    """Perfect foresight must earn at least as much as every other strategy on every day."""
    if PERFECT not in daily:
        return
    for name in others:
        slack = rtol * np.maximum(1.0, daily[PERFECT].abs())
        bad = daily[name] > daily[PERFECT] + slack
        if bad.any():
            raise AssertionError(f"{name} beats {PERFECT} on {int(bad.sum())} day(s), "
                                 f"first {daily.loc[bad, 'delivery_day'].iloc[0]}")


def value_summary(daily, strategy):
    """Annual profit (EUR per MW per year), capture rate and risk metrics of one strategy."""
    profit = daily[strategy].to_numpy()
    n_days = len(profit)
    out = {
        "strategy": strategy,
        "days": n_days,
        # Mean daily profit x 365; for the 365-day validation and test years this is the sum.
        "profit_eur_per_mw_year": float(profit.mean() * 365 / config.POWER_MW),
        "mean_daily_profit_eur": float(profit.mean()),
    }
    if PERFECT in daily:
        out["capture_rate"] = float(profit.sum() / daily[PERFECT].sum())
    out.update(risk.risk_metrics(profit))
    return out


def _extreme_hours(values, k, cheapest):
    """Positions of the k cheapest (or most expensive) hours; ties broken by time."""
    order = np.lexsort((np.arange(len(values)), values if cheapest else -values))
    return set(order[:k])


def within_day_spearman(pred, actual, days):
    """Spearman rho between `pred` and `actual` for each delivery day (a Series by day).

    Spearman rho is the Pearson correlation of the ranks (average ranks for ties, as in
    scipy.stats.spearmanr). A day where either vector is flat has no rho (NaN).
    """
    df = pd.DataFrame({"day": days, "pred": np.asarray(pred, dtype=float),
                       "actual": np.asarray(actual, dtype=float)})
    by_day = df.groupby("day", sort=False)
    rp = by_day["pred"].rank()
    ra = by_day["actual"].rank()
    rp = rp - rp.groupby(df["day"]).transform("mean")
    ra = ra - ra.groupby(df["day"]).transform("mean")
    num = (rp * ra).groupby(df["day"], sort=False).sum()
    den = np.sqrt((rp ** 2).groupby(df["day"], sort=False).sum() * (ra ** 2).groupby(df["day"], sort=False).sum())
    return (num / den).where(den > 0)


def mean_within_day_spearman(pred, actual, days):
    """Mean of the daily Spearman rho, over the days where it is defined."""
    return float(within_day_spearman(pred, actual, days).mean())


def forecast_summary(vectors, strategy, k=2):
    """Price RMSE and MAE (EUR/MWh), mean within-day Spearman rho, and hit rates for the k
    cheapest and k most expensive hours."""
    err = vectors[strategy] - vectors["price"]
    rhos = within_day_spearman(vectors[strategy], vectors["price"], vectors["delivery_day"])
    hit_low, hit_high = [], []
    for _, day in vectors.groupby("delivery_day", sort=False):
        pred, actual = day[strategy].to_numpy(), day["price"].to_numpy()
        hit_low.append(len(_extreme_hours(pred, k, True) & _extreme_hours(actual, k, True)) / k)
        hit_high.append(len(_extreme_hours(pred, k, False) & _extreme_hours(actual, k, False)) / k)
    return {
        "rmse_eur_mwh": float(np.sqrt((err ** 2).mean())),
        "mae_eur_mwh": float(err.abs().mean()),
        "spearman_rho_mean": float(rhos.mean()),
        "spearman_days": int(rhos.notna().sum()),
        f"hit_rate_cheapest{k}": float(np.mean(hit_low)),
        f"hit_rate_dearest{k}": float(np.mean(hit_high)),
    }


def block_indices(n, block=config.BOOTSTRAP_BLOCK_DAYS, n_resamples=config.BOOTSTRAP_RESAMPLES,
                  seed=config.BOOTSTRAP_SEED):
    """Day indices of each moving-block resample, shape (n_resamples, n).

    Each resample joins random blocks of 7 consecutive days (never single hours), which keeps
    the serial dependence between neighbouring days, and is cut to n days.
    """
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_resamples, n_blocks))
    return (starts[:, :, None] + np.arange(block)).reshape(n_resamples, -1)[:, :n]


def block_bootstrap_ci(daily_diff, block=config.BOOTSTRAP_BLOCK_DAYS,
                       n_resamples=config.BOOTSTRAP_RESAMPLES, seed=config.BOOTSTRAP_SEED, level=0.95):
    """Mean daily difference and its moving-block bootstrap CI (PLAN.md §6).

    Returns (mean, lower, upper), percentile interval.
    """
    x = np.asarray(daily_diff, dtype=float)
    means = x[block_indices(len(x), block, n_resamples, seed)].mean(axis=1)
    tail = (1 - level) / 2
    lower, upper = np.quantile(means, [tail, 1 - tail])
    return float(x.mean()), float(lower), float(upper)


def rmse_difference_ci(sse_a, sse_b, n_hours, block=config.BOOTSTRAP_BLOCK_DAYS,
                       n_resamples=config.BOOTSTRAP_RESAMPLES, seed=config.BOOTSTRAP_SEED, level=0.95):
    """RMSE(a) - RMSE(b) over the period (EUR/MWh) and its moving-block bootstrap CI (H2).

    Inputs are per day: the sums of squared errors of a and b and the number of hours. Each
    resample recomputes both RMSEs over its days, so long days weigh by their hours.
    """
    sse_a, sse_b, n_hours = (np.asarray(v, dtype=float) for v in (sse_a, sse_b, n_hours))
    idx = block_indices(len(n_hours), block, n_resamples, seed)
    hours = n_hours[idx].sum(axis=1)
    diffs = np.sqrt(sse_a[idx].sum(axis=1) / hours) - np.sqrt(sse_b[idx].sum(axis=1) / hours)
    point = np.sqrt(sse_a.sum() / n_hours.sum()) - np.sqrt(sse_b.sum() / n_hours.sum())
    tail = (1 - level) / 2
    lower, upper = np.quantile(diffs, [tail, 1 - tail])
    return float(point), float(lower), float(upper)


# --- Paired comparisons and pre-registered verdicts (PLAN.md §6) ------------------------------
def paired_differences(vectors, daily, pairs=config.PAIRS):
    """Daily profit and RMSE differences, rank minus reg, with moving-block bootstrap CIs
    (the statistics of H1-H3, PLAN.md §6). Every row uses the same resampled days."""
    rows = []
    sse = (vectors[[s for pair in pairs for s in pair]].sub(vectors["price"], axis=0) ** 2
           ).groupby(vectors["delivery_day"]).sum()
    for a, b in pairs:
        diff = daily[a] - daily[b]
        mean, lo, hi = block_bootstrap_ci(diff)
        rmse, rlo, rhi = rmse_difference_ci(sse[a].to_numpy(), sse[b].to_numpy(), daily["n_hours"].to_numpy())
        same_vector = (vectors[a] == vectors[b]).groupby(vectors["delivery_day"]).all()
        rows.append({"comparison": f"{a} - {b}", "days": int(len(diff)),
                     "mean_daily_profit_diff_eur": mean, "ci95_low_eur": lo, "ci95_high_eur": hi,
                     "annual_profit_diff_eur_per_mw": mean * 365 / config.POWER_MW,
                     "days_rank_better": int((diff > 1e-9).sum()), "days_reg_better": int((diff < -1e-9).sum()),
                     "days_identical_price_vector": int(same_vector.sum()),
                     "rmse_diff_eur_mwh": rmse, "rmse_diff_ci95_low": rlo, "rmse_diff_ci95_high": rhi})
    return pd.DataFrame(rows)


def verdict_superiority(lower, upper):
    """H1 and H3: supported if the CI's lower bound > 0, contradicted if its upper bound < 0,
    inconclusive otherwise."""
    if lower > 0:
        return "supported"
    if upper < 0:
        return "contradicted"
    return "inconclusive"


def verdict_equivalence(lower, upper, margin):
    """H2: supported if the whole CI lies inside (-margin, +margin), contradicted if the whole
    CI lies outside it (lower bound > margin, or upper bound < -margin), inconclusive otherwise."""
    if -margin < lower and upper < margin:
        return "supported"
    if lower > margin or upper < -margin:
        return "contradicted"
    return "inconclusive"


# --- Solving many price vectors in parallel ----------------------------------------------------
def _profits_worker(job):
    vectors, names = job
    return daily_profits(vectors, names)


def parallel_daily_profits(vectors, names, workers=4):
    """daily_profits for many price vectors, split over processes. Every day is still
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
    daily = daily[["delivery_day", "n_hours"] + names]
    # Perfect foresight is checked against every other vector, also across worker chunks.
    check_perfect_is_upper_bound(daily, [c for c in names if c != PERFECT])
    return daily
