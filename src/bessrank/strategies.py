"""Price vectors per strategy, including the "-rank" reassignment (PLAN.md §5).

Every strategy produces one price vector per delivery day. The battery schedule is optimised
on that vector and settled at the actual prices, so strategies differ only in the vector.
"""
import numpy as np
import pandas as pd

from bessrank import features


def rank_reassign(values, scores):
    """Keep the price model's values for the day and reassign them in the ranker's order.

    1. Sort the forecast values: v(1) <= ... <= v(n).
    2. Order the hours by ranker score, cheapest first; ties are broken by the price
       forecast, then by time.
    3. Give the k-th hour in that order the k-th smallest value.

    If the ranker orders the hours like the price model, the output equals `values`.
    """
    v = np.asarray(values, dtype=float)
    s = np.asarray(scores, dtype=float)
    if v.shape != s.shape:
        raise ValueError("values and scores must have the same length")
    hours = np.arange(len(v))
    # np.lexsort sorts by the last key first: score, then value, then hour.
    order = np.lexsort((hours, v, s))
    out = np.empty_like(v)
    out[order] = np.sort(v)
    return out


def rank_reassign_frame(frame, value_col, score_col):
    """Apply rank_reassign day by day. `frame` has one row per delivery hour."""
    out = pd.Series(np.nan, index=frame.index)
    for _, day in frame.groupby("delivery_day", sort=False):
        out[day.index] = rank_reassign(day[value_col].to_numpy(), day[score_col].to_numpy())
    return out


def baseline_price_vectors(hourly):
    """Price vectors of S-perfect, S-naive-1d and S-naive-7d for every hour of `hourly`.

    Naive strategies use the price at the same wall-clock hour on D-1 or D-7 (the 02:00 hours
    are averaged or interpolated on DST days, see features.lagged_hourly), which is known at
    the decision time.
    """
    out = hourly[["ts_utc", "delivery_day", "hour", "n_hours", "price"]].copy()
    out["S-perfect"] = hourly["price"]
    out["S-naive-1d"] = features.lagged_hourly(hourly, "price", 1)
    out["S-naive-7d"] = features.lagged_hourly(hourly, "price", 7)
    return out
