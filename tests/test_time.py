"""Delivery days are Europe/Berlin calendar days with 23, 24 or 25 hours; timestamps are UTC."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from bessrank import config, data


def day_table(first_day, last_day):
    """Hourly grid with the delivery-day columns, as in the processed table."""
    return data.add_time_columns(pd.DataFrame(index=data.hourly_grid(first_day, last_day)))


def test_2024_03_31_has_23_hours():
    t = day_table(date(2024, 3, 31), date(2024, 3, 31))
    assert len(t) == 23
    assert list(t["hour"]) == list(range(23))
    assert (t["n_hours"] == 23).all()
    assert 2 not in set(t["local_hour"])  # 02:00 local time does not exist that night
    assert t.index[0] == pd.Timestamp("2024-03-30 23:00", tz="UTC")  # 00:00 CET
    assert t.index[-1] == pd.Timestamp("2024-03-31 21:00", tz="UTC")  # 23:00 CEST


def test_2024_10_27_has_25_hours():
    t = day_table(date(2024, 10, 27), date(2024, 10, 27))
    assert len(t) == 25
    assert list(t["hour"]) == list(range(25))
    assert (t["n_hours"] == 25).all()
    assert (t["local_hour"] == 2).sum() == 2  # 02:00 local time occurs twice
    assert t.index[0] == pd.Timestamp("2024-10-26 22:00", tz="UTC")  # 00:00 CEST
    assert t.index[-1] == pd.Timestamp("2024-10-27 22:00", tz="UTC")  # 23:00 CET


def test_neighbouring_days_have_24_hours():
    for day in [date(2024, 3, 30), date(2024, 4, 1), date(2024, 10, 26), date(2024, 10, 28)]:
        assert len(day_table(day, day)) == 24


def test_grid_is_utc_and_hourly_across_dst():
    t = day_table(date(2024, 3, 25), date(2024, 11, 3))
    assert str(t.index.tz) == "UTC"
    assert (np.diff(t.index.asi8) == 3_600_000_000_000).all()  # exactly one hour apart
    sizes = t.groupby("delivery_day").size()
    assert sizes[date(2024, 3, 31)] == 23 and sizes[date(2024, 10, 27)] == 25
    assert set(sizes.drop([date(2024, 3, 31), date(2024, 10, 27)])) == {24}


def test_quarter_hour_mean_on_a_25_hour_day():
    start = pd.Timestamp("2024-10-26 22:00", tz="UTC")
    qh = pd.Series(np.arange(100, dtype=float), index=pd.date_range(start, periods=100, freq="15min"))
    hourly = data.quarter_hours_to_hourly(qh)
    assert len(hourly) == 25
    assert (hourly["n_quarter_hours"] == 4).all()
    np.testing.assert_allclose(hourly["price_qh_mean"], np.arange(25) * 4 + 1.5)


def _points(index, value):
    return [[int(ts.value // 1_000_000), value] for ts in index]


def test_build_uses_quarter_hour_mean_from_quarter_hour_start():
    # Synthetic data around QUARTER_HOUR_START: no real prices are read.
    first, last = date(2025, 9, 30), date(2025, 10, 1)
    grid = data.hourly_grid(first, last)
    raw = {name: _points(grid, 10.0) for name in data.SERIES}
    raw["price"] = _points(grid, 50.0)
    qh_index = pd.date_range(data.hourly_grid(last, last)[0], periods=96, freq="15min")
    raw["price_qh"] = [[ms, float(i)] for i, (ms, _) in enumerate(_points(qh_index, 0.0))]
    raw["price_qh"][-1][1] = None  # one missing quarter-hour in the last hour

    table, raw_checks = data.build_hourly_table(raw, first, last)
    before = table[table["delivery_day"] == first]
    after = table[table["delivery_day"] == last]
    assert len(before) == 24 and len(after) == 24
    assert (before["price"] == 50.0).all() and (before["n_quarter_hours"] == 0).all()
    np.testing.assert_allclose(after["price"].iloc[:-1], np.arange(23) * 4 + 1.5)
    assert np.isnan(after["price"].iloc[-1])  # incomplete hour is left missing, not averaged
    assert (table["price_smard_hourly"] == 50.0).all()
    assert str(table["ts_utc"].dt.tz) == "UTC"
    assert all(check["duplicates"] == 0 for check in raw_checks.values())


@pytest.mark.skipif(not config.HOURLY_PARQUET.exists(), reason="processed data not built")
def test_processed_table_has_dst_days():
    table = data.load_hourly()
    sizes = table.groupby("delivery_day").size()
    assert sizes[date(2024, 3, 31)] == 23
    assert sizes[date(2024, 10, 27)] == 25
    assert set(sizes.unique()) <= {23, 24, 25}
