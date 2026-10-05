"""No look-ahead (CLAUDE.md rule 2): features for day D only use data known at 11:00 on D-1.

For a set of delivery days D, every value that is not available at the decision time is
deleted (features.visible_data), the gaps are filled and the features recomputed; the rows of
D must be identical to those computed from the full table. Synthetic tests run everywhere;
the test on real data runs when data/processed/hourly.parquet exists.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from bessrank import config, data, features

HISTORY_DAYS = 40  # more than the longest lag (28-day profile) plus a margin


def synthetic_hourly(first_day, last_day, seed=0):
    """Random hourly table with the processed columns. No real prices are involved."""
    grid = data.hourly_grid(first_day, last_day)
    table = data.add_time_columns(pd.DataFrame(index=grid)).reset_index()
    rng = np.random.default_rng(seed)
    for name in data.SERIES:
        table[name] = rng.normal(100, 30, len(table))
    table["pv_fc"] = table["pv_fc"].clip(lower=0)
    table["price_smard_hourly"] = table["price"]
    table["n_quarter_hours"] = 0
    return table


def assert_no_lookahead(raw, day):
    """Features for `day` from the full table equal those from the data visible on day-1 11:00."""
    window = raw[raw["delivery_day"] >= day - timedelta(days=HISTORY_DAYS)]
    full = features.build_features(data.fill_short_gaps(window))
    seen = features.build_features(data.fill_short_gaps(features.visible_data(window, day)))
    a = full.loc[full["delivery_day"] == day, features.FEATURE_COLUMNS].reset_index(drop=True)
    b = seen.loc[seen["delivery_day"] == day, features.FEATURE_COLUMNS].reset_index(drop=True)
    assert len(a) == len(b) > 0
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_every_feature_declares_a_known_rule():
    assert set(features.FEATURES.values()) <= set(features.RULES)
    assert len(features.FEATURE_COLUMNS) == len(set(features.FEATURE_COLUMNS))
    assert features.LABEL not in features.FEATURE_COLUMNS
    assert set(features.AVAILABLE_UNTIL) >= set(data.SERIES)


def test_visible_data_blanks_unknown_values():
    raw = synthetic_hourly(date(2024, 3, 1), date(2024, 3, 20))
    day = date(2024, 3, 15)
    view = features.visible_data(raw, day)
    assert view["delivery_day"].max() == day
    assert view.loc[view["delivery_day"] >= day, "price"].isna().all()
    assert view.loc[view["delivery_day"] < day, "price"].notna().all()
    assert view.loc[view["delivery_day"] >= day - timedelta(days=1), "load_actual"].isna().all()
    assert view.loc[view["delivery_day"] < day - timedelta(days=1), "load_actual"].notna().all()
    assert view.loc[view["delivery_day"] == day, "load_fc"].notna().all()


@pytest.mark.parametrize("day", [date(2024, 3, 15), date(2024, 3, 31), date(2024, 4, 1),
                                 date(2024, 4, 7), date(2024, 10, 27), date(2024, 10, 28)])
def test_no_lookahead_synthetic(day):
    raw = synthetic_hourly(date(2024, 1, 1), date(2024, 11, 30))
    assert_no_lookahead(raw, day)


def test_lookahead_is_detected():
    """The check itself works: a feature that peeks at D's price is caught."""
    raw = synthetic_hourly(date(2024, 1, 1), date(2024, 3, 31))
    day = date(2024, 3, 15)
    original = features._lagged_prices

    def leaky(hourly):
        out = original(hourly)
        out["price_lag1"] = hourly["price"]  # the price of D itself: look-ahead
        return out

    features._lagged_prices = leaky
    try:
        with pytest.raises(AssertionError):
            assert_no_lookahead(raw, day)
    finally:
        features._lagged_prices = original


def test_lags_use_the_same_wall_clock_hour_across_dst():
    raw = synthetic_hourly(date(2024, 10, 20), date(2024, 10, 29))
    feats = features.build_features(raw)
    by_day = {d: g.reset_index(drop=True) for d, g in raw.groupby("delivery_day")}

    # D = 2024-10-28 (24 h) after the 25-hour day: 02:00 takes the mean of the two 02:00 hours.
    d28 = feats[feats["delivery_day"] == date(2024, 10, 28)].reset_index(drop=True)
    d27 = by_day[date(2024, 10, 27)]
    assert d28.loc[2, "price_lag1"] == pytest.approx(d27.loc[d27["local_hour"] == 2, "price"].mean())
    assert d28.loc[5, "price_lag1"] == d27.loc[6, "price"]  # 05:00 is position 6 on the 25-h day

    # D = 2024-10-27 (25 h): both 02:00 hours take 02:00 of D-1.
    d27f = feats[feats["delivery_day"] == date(2024, 10, 27)].reset_index(drop=True)
    assert len(d27f) == 25
    assert d27f.loc[2, "price_lag1"] == d27f.loc[3, "price_lag1"] == by_day[date(2024, 10, 26)].loc[2, "price"]


def test_23_hour_day_lag_interpolates_0200():
    raw = synthetic_hourly(date(2024, 3, 28), date(2024, 4, 2))
    feats = features.build_features(raw)
    d31 = raw[raw["delivery_day"] == date(2024, 3, 31)].reset_index(drop=True)
    apr1 = feats[feats["delivery_day"] == date(2024, 4, 1)].reset_index(drop=True)
    assert len(d31) == 23
    expected = (d31.loc[d31["local_hour"] == 1, "price"].iloc[0] + d31.loc[d31["local_hour"] == 3, "price"].iloc[0]) / 2
    assert apr1.loc[2, "price_lag1"] == pytest.approx(expected)


def test_short_gaps_filled_long_gaps_kept():
    raw = synthetic_hourly(date(2024, 3, 1), date(2024, 3, 5))
    raw.loc[10, "load_fc"] = np.nan  # 1-hour gap
    raw.loc[[30, 31], "pv_fc"] = np.nan  # 2-hour gap
    raw.loc[50:52, "wind_onshore_fc"] = np.nan  # 3-hour gap: too long
    filled = data.fill_short_gaps(raw)
    assert filled.loc[10, "load_fc"] == pytest.approx((raw.loc[9, "load_fc"] + raw.loc[11, "load_fc"]) / 2)
    step = (raw.loc[32, "pv_fc"] - raw.loc[29, "pv_fc"]) / 3
    assert filled.loc[30, "pv_fc"] == pytest.approx(raw.loc[29, "pv_fc"] + step)
    assert filled.loc[31, "pv_fc"] == pytest.approx(raw.loc[29, "pv_fc"] + 2 * step)
    assert filled.loc[50:52, "wind_onshore_fc"].isna().all()
    assert filled["load_fc_filled"].sum() == 1 and filled["pv_fc_filled"].sum() == 2
    assert filled["wind_onshore_fc_filled"].sum() == 0


def test_first_hour_of_25_hour_day_is_filled_in_utc_time():
    raw = synthetic_hourly(date(2024, 10, 26), date(2024, 10, 28))
    first = raw.index[(raw["delivery_day"] == date(2024, 10, 27)) & (raw["hour"] == 0)][0]
    raw.loc[first, "pv_fc"] = np.nan
    filled = data.fill_short_gaps(raw)
    assert filled.loc[first, "pv_fc"] == pytest.approx((raw.loc[first - 1, "pv_fc"] + raw.loc[first + 1, "pv_fc"]) / 2)


def test_incomplete_days_flags_long_gaps():
    raw = synthetic_hourly(date(2024, 1, 1), date(2024, 3, 31))
    raw.loc[raw["delivery_day"] == date(2024, 3, 10), "load_fc"] = np.nan
    feats = features.build_features(data.fill_short_gaps(raw))
    assert features.incomplete_days(feats, date(2024, 3, 1), date(2024, 3, 31)) == [date(2024, 3, 10)]


@pytest.mark.skipif(not config.HOURLY_PARQUET.exists(), reason="processed data not built")
def test_no_lookahead_real_data():
    """Real training/validation data: DST days, a day after a 411 gap, the first validation
    day, the last validation day, and 30 random days."""
    raw = data.load_hourly(fill_gaps=False)  # the test period stays locked
    rng = np.random.default_rng(0)
    days = pd.date_range(config.TRAIN_START, config.VAL_END, freq="D").date
    sample = list(rng.choice(days, size=30, replace=False))
    sample += [date(2024, 3, 31), date(2024, 10, 27), date(2023, 10, 29), date(2022, 2, 23),
               config.TRAIN_START, config.VAL_START, config.VAL_END]
    for day in sample:
        assert_no_lookahead(raw, day)
