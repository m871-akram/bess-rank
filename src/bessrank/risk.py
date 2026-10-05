"""Risk metrics of daily profit (PLAN.md §5). PSI and conformal calibration come in S4.

All inputs are daily profits in EUR (per MW, the battery is 1 MW). VaR and ES are reported
as profit levels: a negative value is a loss.
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
        "var5_eur_per_day": value_at_risk(daily_profit, alpha),
        "es5_eur_per_day": expected_shortfall(daily_profit, alpha),
        "max_drawdown_eur": max_drawdown(daily_profit),
        "losing_day_share": losing_day_share(daily_profit),
    }
