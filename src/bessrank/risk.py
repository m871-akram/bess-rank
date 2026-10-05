"""Risk metrics of daily profit (PLAN.md §5); PSI and conformal calibration (§8, S4).

All inputs are daily profits in EUR (per MW, the battery is 1 MW). VaR and ES are reported
as profit levels, not as losses: a negative value is a loss. Column names and table headers
say so ("P5 of daily profit", "mean of the worst 5% of days") so a risk reader does not
misread the sign (decided on 2026-10-05).
"""
import numpy as np

ALPHA = 0.05


def value_at_risk(daily_profit, alpha=ALPHA):
    """VaR 5%: the 5th percentile of daily profit (linear interpolation between days)."""
    return float(np.quantile(np.asarray(daily_profit, dtype=float), alpha))


def expected_shortfall(daily_profit, alpha=ALPHA):
    """ES 5%: mean profit of the worst ceil(5% x n) days (19 of 365)."""
    x = np.sort(np.asarray(daily_profit, dtype=float))
    k = int(np.ceil(alpha * len(x)))
    return float(x[:k].mean())


def max_drawdown(daily_profit):
    """Largest fall of cumulative profit from an earlier peak (EUR, >= 0). The start (0 EUR)
    counts as a peak, so early losses are included."""
    cumulative = np.concatenate([[0.0], np.cumsum(np.asarray(daily_profit, dtype=float))])
    return float((np.maximum.accumulate(cumulative) - cumulative).max())


def losing_day_share(daily_profit):
    """Share of days with a negative profit."""
    return float((np.asarray(daily_profit, dtype=float) < 0).mean())


def risk_metrics(daily_profit, alpha=ALPHA):
    return {
        "var5_p5_of_daily_profit_eur": value_at_risk(daily_profit, alpha),
        "es5_mean_of_worst5pct_days_eur": expected_shortfall(daily_profit, alpha),
        "max_drawdown_eur": max_drawdown(daily_profit),
        "losing_day_share": losing_day_share(daily_profit),
    }


# --- Stability and conformal calibration (PLAN.md §8, exploratory) ------------------------
def psi(expected, actual, bins=10):
    """Population Stability Index of `actual` (test) against `expected` (train).

    Bins are the deciles of `expected` (merged where values repeat), plus a bin for missing
    values; empty bins get a floor of 1e-4 so the log stays finite. Usual reading: < 0.1
    stable, 0.1-0.25 moderate shift, > 0.25 large shift.
    """
    expected, actual = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
    edges = np.unique(np.nanquantile(expected, np.linspace(0, 1, bins + 1)))[1:-1]

    def shares(x):
        idx = np.where(np.isnan(x), len(edges) + 1, np.searchsorted(edges, x, side="right"))
        return np.maximum(np.bincount(idx, minlength=len(edges) + 2) / len(x), 1e-4)
    p, q = shares(expected), shares(actual)
    return float(np.sum((q - p) * np.log(q / p)))


QUANTILE_LEVELS = np.round(np.arange(0.05, 0.951, 0.05), 2)  # 19 levels (PLAN.md §4)


def cqr_corrections(q_cal, y_cal, levels=QUANTILE_LEVELS):
    """Conformalized quantile regression (Romano, Patterson & Candes 2019) on a calibration set.

    For each symmetric pair of levels (a, 1 - a) the conformity score of a calibration hour is
    max(q_a - y, y - q_{1-a}); the correction is its ceil((n + 1)(1 - 2a)) / n empirical
    quantile, which widens (or narrows, if negative) the interval so that it covers about
    1 - 2a of the hours. Returns {a: correction}. The median gets no correction.
    """
    q_cal, y_cal = np.asarray(q_cal, dtype=float), np.asarray(y_cal, dtype=float)
    n, out = len(y_cal), {}
    for i, a in enumerate(levels):
        j = len(levels) - 1 - i
        if a >= 0.5:
            break
        scores = np.maximum(q_cal[:, i] - y_cal, y_cal - q_cal[:, j])
        level = min(1.0, np.ceil((n + 1) * (1 - 2 * a)) / n)
        out[float(a)] = float(np.quantile(scores, level, method="higher"))
    return out


def apply_cqr(q, corrections, levels=QUANTILE_LEVELS):
    """Move each pair (a, 1 - a) apart by its correction; sort each row so quantiles never cross."""
    q = np.array(q, dtype=float, copy=True)
    for i, a in enumerate(levels):
        if float(a) in corrections:
            j = len(levels) - 1 - i
            q[:, i] -= corrections[float(a)]
            q[:, j] += corrections[float(a)]
    return np.sort(q, axis=1)


def pinball_loss(q, y, levels=QUANTILE_LEVELS):
    """Mean pinball (quantile) loss over every hour and level (EUR/MWh)."""
    diff = np.asarray(y, dtype=float)[:, None] - np.asarray(q, dtype=float)
    return float(np.mean(np.maximum(levels * diff, (levels - 1) * diff)))


def interval_coverage(q, y, levels=QUANTILE_LEVELS):
    """Share of hours inside each central interval [q_a, q_{1-a}]: {nominal coverage: share}."""
    y = np.asarray(y, dtype=float)
    out = {}
    for i, a in enumerate(levels):
        if a >= 0.5:
            break
        j = len(levels) - 1 - i
        out[round(1 - 2 * float(a), 2)] = float(np.mean((q[:, i] <= y) & (y <= q[:, j])))
    return out
