"""Features for each hour h of delivery day D, known at 11:00 Europe/Berlin on D-1 (PLAN.md §3).

Every feature declares an availability rule in FEATURES. `visible_data(hourly, D)` blanks
every value that is not known at the decision time for D; tests/test_features.py recomputes
the features from that view and checks that the values for D do not change.

Lagged hourly values are looked up by wall-clock hour: hour h of D uses the same local hour
of D-k. This works across days of 23, 24 and 25 hours (see daily_profile).
"""
from datetime import timedelta

import holidays
import numpy as np
import pandas as pd

from bessrank import config

# --- Availability rules ---------------------------------------------------------------------
# Decision for delivery day D at 11:00 on D-1: series s is known up to the end of delivery day
# D - AVAILABLE_UNTIL[s] (in days).
AVAILABLE_UNTIL = {
    "price": 1,  # prices for D-1 were published around 12:45 on D-2
    "price_smard_hourly": 1,
    "load_actual": 2,  # actuals up to the end of D-2: conservative, as SMARD lags a little
    "residual_load_actual": 2,
    "load_fc": 0,  # TSO load forecast for D, due by 10:00 on D-1 (EU 543/2013 Art. 6(2)(b))
    # TSO wind and PV forecasts for D are only due at 18:00 on D-1 (Art. 14(1)(c)-(d)). They
    # stand in for the vendor weather forecasts traders buy in the morning (PLAN.md §3 caveat;
    # tested in §8 by a run without them).
    "wind_onshore_fc": 0,
    "wind_offshore_fc": 0,
    "pv_fc": 0,
}

RULES = {
    "calendar": "Known in advance: date, wall-clock hour, DST, German national holidays.",
    "load_fc_D": "TSO load forecast (411) for the hours of D; due by 10:00 on D-1.",
    "gen_fc_D": "TSO wind/PV forecasts (123, 3791, 125) for D; proxy for morning vendor forecasts.",
    "price_D-1": "Day-ahead prices of delivery days D-28 .. D-1 (published by 12:45 on D-2).",
    "actual_D-2": "Actual load (410) and residual load (4359) of delivery days D-7 .. D-2.",
}

# Every feature and its rule. Features using any TSO generation forecast have rule gen_fc_D;
# they are the ones dropped in the §8 robustness run.
FEATURES = {
    "local_hour": "calendar", "weekday": "calendar", "month": "calendar",
    "doy_sin": "calendar", "doy_cos": "calendar", "holiday": "calendar",
    "bridge_day": "calendar", "dst": "calendar", "n_hours": "calendar",
    "load_fc": "load_fc_D",
    "wind_onshore_fc": "gen_fc_D", "wind_offshore_fc": "gen_fc_D", "pv_fc": "gen_fc_D",
    "wind_pv_fc": "gen_fc_D", "wind_pv_share": "gen_fc_D", "residual_load_fc": "gen_fc_D",
    "pv_norm": "gen_fc_D", "wind_pv_norm": "gen_fc_D",
    "residual_load_fc_dev": "gen_fc_D", "residual_load_fc_rank": "gen_fc_D",
    "residual_load_fc_day_mean": "gen_fc_D", "residual_load_fc_day_min": "gen_fc_D",
    "residual_load_fc_day_max": "gen_fc_D", "pv_fc_day_energy": "gen_fc_D",
    "wind_fc_day_energy": "gen_fc_D",
    "price_lag1": "price_D-1", "price_lag7": "price_D-1",
    "price_lag1_day_mean": "price_D-1", "price_lag1_day_min": "price_D-1",
    "price_lag1_day_max": "price_D-1", "price_lag1_day_spread": "price_D-1",
    "price_lag7_day_mean": "price_D-1", "price_lag7_day_min": "price_D-1",
    "price_lag7_day_max": "price_D-1", "price_lag7_day_spread": "price_D-1",
    "price_lag1_rank": "price_D-1", "price_lag7_rank": "price_D-1",
    "price_profile28": "price_D-1",
    "load_actual_lag2": "actual_D-2", "load_actual_lag7": "actual_D-2",
    "residual_load_actual_lag2": "actual_D-2", "residual_load_actual_lag7": "actual_D-2",
}
FEATURE_COLUMNS = list(FEATURES)
GEN_FORECAST_FEATURES = [name for name, rule in FEATURES.items() if rule == "gen_fc_D"]
KEY_COLUMNS = ["ts_utc", "delivery_day", "hour"]
LABEL = "price"  # EUR/MWh at (D, h); a label, never a feature
PROFILE_DAYS = 28


def visible_data(hourly, day):
    """The hourly table as it is known at 11:00 on day-1, when deciding for delivery day `day`.

    Rows after `day` are removed, and each series is blanked after its availability limit.
    """
    view = hourly[hourly["delivery_day"] <= day].copy()
    for series, lag in AVAILABLE_UNTIL.items():
        if series in view:
            view.loc[view["delivery_day"] > day - timedelta(days=lag), series] = np.nan
    return view


# --- Wall-clock profiles --------------------------------------------------------------------
def _day_index(hourly):
    return pd.DatetimeIndex(pd.to_datetime(hourly["delivery_day"])).as_unit("ns")


def daily_profile(hourly, column):
    """Day x wall-clock hour (0..23) matrix of one series, on a complete calendar of days.

    25-hour days: the two 02:00 values are averaged. 23-hour days have no 02:00; it is set to
    the mean of 01:00 and 03:00. So D can look up the same local hour on D-k even when the two
    days have different lengths.
    """
    day = _day_index(hourly)
    prof = (pd.DataFrame({"day": day, "local_hour": hourly["local_hour"].to_numpy(),
                          "value": hourly[column].to_numpy()})
            .groupby(["day", "local_hour"])["value"].mean().unstack("local_hour"))
    calendar = pd.date_range(day.min(), day.max(), freq="D").as_unit("ns")
    prof = prof.reindex(index=calendar, columns=range(24))
    n_hours = pd.Series(hourly["n_hours"].to_numpy(), index=day).groupby(level=0).first()
    short_days = n_hours.index[n_hours == 23]
    prof.loc[short_days, 2] = (prof.loc[short_days, 1] + prof.loc[short_days, 3]) / 2
    return prof


def _shift_days(frame, days):
    """Value of day D taken from day D - days, on the same calendar index."""
    return frame.shift(days, freq="D").reindex(frame.index)


def _lookup(day_by_hour, hourly):
    """Pick, for each row of `hourly`, the value at (its delivery day, its local hour)."""
    rows = day_by_hour.index.get_indexer(_day_index(hourly))
    values = day_by_hour.to_numpy()[rows, hourly["local_hour"].to_numpy()]
    values[rows < 0] = np.nan
    return pd.Series(values, index=hourly.index)


def _lookup_daily(by_day, hourly):
    """Broadcast a per-day Series (calendar index) to the rows of `hourly`."""
    return pd.Series(by_day.reindex(_day_index(hourly)).to_numpy(), index=hourly.index)


def lagged_hourly(hourly, column, lag_days):
    """Value of `column` at the same wall-clock hour on D - lag_days."""
    return _lookup(_shift_days(daily_profile(hourly, column), lag_days), hourly)


# --- Calendar -------------------------------------------------------------------------------
def _calendar(hourly):
    """Calendar features. Holidays are German national holidays (no regional ones)."""
    local = hourly["ts_utc"].dt.tz_convert(config.MARKET_TZ)
    day = _day_index(hourly)
    years = range(day.min().year - 1, day.max().year + 2)
    de = holidays.Germany(years=years)

    def is_holiday(days):
        return np.array([d.date() in de for d in days])

    unique_days = day.unique()
    holiday = pd.Series(is_holiday(unique_days), index=unique_days)
    # Bridge day: a working day squeezed between a holiday and the weekend, when many take
    # the day off (Monday before a Tuesday holiday, Friday after a Thursday holiday).
    weekday = unique_days.weekday
    next_is_holiday = is_holiday(unique_days + pd.Timedelta(days=1))
    prev_is_holiday = is_holiday(unique_days - pd.Timedelta(days=1))
    bridge = pd.Series(~holiday.to_numpy() & (((weekday == 0) & next_is_holiday) | ((weekday == 4) & prev_is_holiday)),
                       index=unique_days)

    doy = day.dayofyear.to_numpy()
    utc_offset = local.dt.tz_localize(None) - hourly["ts_utc"].dt.tz_localize(None)
    return pd.DataFrame({
        "local_hour": hourly["local_hour"].to_numpy(),
        "weekday": day.weekday,
        "month": day.month,
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "holiday": holiday.reindex(day).to_numpy().astype(int),
        "bridge_day": bridge.reindex(day).to_numpy().astype(int),
        "dst": (utc_offset == pd.Timedelta(hours=2)).to_numpy().astype(int),  # CEST
        "n_hours": hourly["n_hours"].to_numpy(),
    }, index=hourly.index)


# --- Forecasts for D ------------------------------------------------------------------------
def _forecasts(hourly):
    """TSO forecasts for the hours of D and within-day transformations of them."""
    f = pd.DataFrame(index=hourly.index)
    for name in ["load_fc", "wind_onshore_fc", "wind_offshore_fc", "pv_fc"]:
        f[name] = hourly[name]
    wind = hourly["wind_onshore_fc"] + hourly["wind_offshore_fc"]
    f["wind_pv_fc"] = wind + hourly["pv_fc"]  # equals SMARD 5097 (checked in S0)
    f["wind_pv_share"] = f["wind_pv_fc"] / hourly["load_fc"]
    f["residual_load_fc"] = hourly["load_fc"] - f["wind_pv_fc"]  # equals SMARD 4362

    by_day = f.groupby(hourly["delivery_day"])
    # Shape within the day matters for the order of the hours, so normalise per day.
    pv_max = by_day["pv_fc"].transform("max")
    f["pv_norm"] = (f["pv_fc"] / pv_max).where(pv_max > 0, 0.0)
    f["wind_pv_norm"] = f["wind_pv_fc"] / by_day["wind_pv_fc"].transform("max")
    f["residual_load_fc_dev"] = f["residual_load_fc"] - by_day["residual_load_fc"].transform("mean")
    # Rank scaled to 0 (lowest) .. 1 (highest), so 23- and 25-hour days are comparable.
    f["residual_load_fc_rank"] = (by_day["residual_load_fc"].rank(method="average") - 1) / (
        hourly["n_hours"] - 1)

    f["residual_load_fc_day_mean"] = by_day["residual_load_fc"].transform("mean")
    f["residual_load_fc_day_min"] = by_day["residual_load_fc"].transform("min")
    f["residual_load_fc_day_max"] = by_day["residual_load_fc"].transform("max")
    # Daily energy (MWh). skipna=False: a day with a long gap gets NaN, not a smaller sum.
    f["pv_fc_day_energy"] = by_day["pv_fc"].transform(lambda s: s.sum(skipna=False))
    f["wind_fc_day_energy"] = wind.groupby(hourly["delivery_day"]).transform(lambda s: s.sum(skipna=False))
    return f


# --- Lagged prices and actuals --------------------------------------------------------------
def _daily_stats(hourly, column):
    """Mean, min and max of `column` per delivery day, on a complete calendar of days.

    A day with any missing hour gets NaN, so a gap never silently changes the statistic.
    """
    day = _day_index(hourly)
    values = pd.Series(hourly[column].to_numpy(), index=day)
    g = values.groupby(level=0)
    stats = pd.DataFrame({"mean": g.mean(), "min": g.min(), "max": g.max()})
    has_gap = values.isna().groupby(level=0).any()
    stats.loc[has_gap[has_gap].index] = np.nan
    return stats.reindex(pd.date_range(day.min(), day.max(), freq="D").as_unit("ns"))


def _lagged_prices(hourly):
    f = pd.DataFrame(index=hourly.index)
    prof = daily_profile(hourly, "price")
    stats = _daily_stats(hourly, "price")
    # Rank of each wall-clock hour within the day, scaled to 0 (cheapest) .. 1 (most expensive).
    rank = (prof.rank(axis=1, method="average") - 1) / 23
    for lag in (1, 7):
        f[f"price_lag{lag}"] = _lookup(_shift_days(prof, lag), hourly)
        lagged_stats = _shift_days(stats, lag)
        for stat in ("mean", "min", "max"):
            f[f"price_lag{lag}_day_{stat}"] = _lookup_daily(lagged_stats[stat], hourly)
        f[f"price_lag{lag}_day_spread"] = f[f"price_lag{lag}_day_max"] - f[f"price_lag{lag}_day_min"]
        f[f"price_lag{lag}_rank"] = _lookup(_shift_days(rank, lag), hourly)

    # Typical shape of the day: each day's prices standardised within the day, then averaged
    # over the 28 previous days. A flat day (std 0) contributes zeros.
    std = prof.std(axis=1)
    z = prof.sub(prof.mean(axis=1), axis=0).div(std, axis=0)
    z.loc[std == 0] = 0.0
    z = z.where(prof.notna())
    profile = sum(_shift_days(z, k) for k in range(1, PROFILE_DAYS + 1)) / PROFILE_DAYS
    f["price_profile28"] = _lookup(profile, hourly)
    return f


def _lagged_actuals(hourly):
    f = pd.DataFrame(index=hourly.index)
    for column in ["load_actual", "residual_load_actual"]:
        prof = daily_profile(hourly, column)
        for lag in (2, 7):
            f[f"{column}_lag{lag}"] = _lookup(_shift_days(prof, lag), hourly)
    return f


def build_features(hourly):
    """One row per delivery hour: keys, the label `price`, and FEATURE_COLUMNS.

    `hourly` is the hourly table after short gaps are filled (data.load_hourly()). Rows with
    a remaining NaN come from gaps longer than 2 hours or from the first days of the data.
    """
    hourly = hourly.sort_values("ts_utc").reset_index(drop=True)
    out = pd.concat([hourly[KEY_COLUMNS + [LABEL]], _calendar(hourly), _forecasts(hourly),
                     _lagged_prices(hourly), _lagged_actuals(hourly)], axis=1)
    return out[KEY_COLUMNS + [LABEL] + FEATURE_COLUMNS]


# --- Missing values (PLAN.md §2) ------------------------------------------------------------
def incomplete_days(feats, first_day, last_day):
    """Delivery days in [first_day, last_day] with a missing feature or label on any hour.

    Training drops these days. Validation and test days are never dropped: XGBoost gets the
    NaN, the LSTM the training-window median for that hour (S2).
    """
    window = feats[(feats["delivery_day"] >= first_day) & (feats["delivery_day"] <= last_day)]
    has_nan = window[FEATURE_COLUMNS + [LABEL]].isna().any(axis=1)
    return sorted(window.loc[has_nan, "delivery_day"].unique())
