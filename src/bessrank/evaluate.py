"""Profit, capture rate, forecast metrics and the moving-block bootstrap (PLAN.md §5, §6)."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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


def forecast_summary(vectors, strategy, k=2):
    """Price RMSE and MAE (EUR/MWh), mean within-day Spearman rho, and hit rates for the k
    cheapest and k most expensive hours."""
    err = vectors[strategy] - vectors["price"]
    rhos, hit_low, hit_high = [], [], []
    for _, day in vectors.groupby("delivery_day", sort=False):
        pred, actual = day[strategy].to_numpy(), day["price"].to_numpy()
        if np.ptp(pred) > 0 and np.ptp(actual) > 0:  # rho is undefined for a flat day
            rhos.append(spearmanr(pred, actual).statistic)
        hit_low.append(len(_extreme_hours(pred, k, True) & _extreme_hours(actual, k, True)) / k)
        hit_high.append(len(_extreme_hours(pred, k, False) & _extreme_hours(actual, k, False)) / k)
    return {
        "rmse_eur_mwh": float(np.sqrt((err ** 2).mean())),
        "mae_eur_mwh": float(err.abs().mean()),
        "spearman_rho_mean": float(np.mean(rhos)),
        "spearman_days": len(rhos),
        f"hit_rate_cheapest{k}": float(np.mean(hit_low)),
        f"hit_rate_dearest{k}": float(np.mean(hit_high)),
    }


def block_bootstrap_ci(daily_diff, block=config.BOOTSTRAP_BLOCK_DAYS,
                       n_resamples=config.BOOTSTRAP_RESAMPLES, seed=config.BOOTSTRAP_SEED, level=0.95):
    """Mean daily difference and its moving-block bootstrap CI (PLAN.md §6).

    Blocks of 7 consecutive days are resampled (never single hours), which keeps the serial
    dependence between neighbouring days. Returns (mean, lower, upper).
    """
    x = np.asarray(daily_diff, dtype=float)
    n = len(x)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_resamples, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_resamples, -1)[:, :n]
    means = x[idx].mean(axis=1)
    tail = (1 - level) / 2
    lower, upper = np.quantile(means, [tail, 1 - tail])
    return float(x.mean()), float(lower), float(upper)
