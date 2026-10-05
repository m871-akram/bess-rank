"""The "-rank" reassignment keeps the price model's values and only changes their order."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from bessrank import battery, data, evaluate, risk, strategies


def test_identical_when_orders_agree():
    rng = np.random.default_rng(0)
    for n in (23, 24, 25):
        values = rng.normal(60, 40, n)
        for scores in (values, 3 * values - 7, np.exp(values / 50)):  # same order as values
            np.testing.assert_array_equal(strategies.rank_reassign(values, scores), values)


def test_identical_with_tied_forecast_values():
    values = np.array([50.0, 30.0, 50.0, 10.0, 30.0, 70.0])
    np.testing.assert_array_equal(strategies.rank_reassign(values, values), values)
    # Ranker ties among hours with different values are broken by the price forecast.
    scores = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 2.0])
    np.testing.assert_array_equal(strategies.rank_reassign(values, scores), values)


def test_same_schedule_and_profit_when_orders_agree():
    rng = np.random.default_rng(1)
    values = rng.normal(60, 40, 24)
    actual = rng.normal(60, 40, 24)
    reassigned = strategies.rank_reassign(values, 2 * values + 1)
    a, b = battery.solve_day(values), battery.solve_day(reassigned)
    assert battery.settle(a, actual) == battery.settle(b, actual)


def test_values_kept_order_changed():
    values = np.array([10.0, 20.0, 30.0, 40.0])
    scores = np.array([4.0, 3.0, 2.0, 1.0])  # ranker reverses the order
    out = strategies.rank_reassign(values, scores)
    np.testing.assert_array_equal(out, [40.0, 30.0, 20.0, 10.0])
    np.testing.assert_array_equal(np.sort(out), np.sort(values))


def test_ties_in_scores_fall_back_to_forecast_then_time():
    values = np.array([30.0, 10.0, 10.0, 20.0])
    scores = np.zeros(4)  # the ranker cannot separate any hours
    # Order by value, then by time: hours 1, 2, 3, 0 get 10, 10, 20, 30.
    np.testing.assert_array_equal(strategies.rank_reassign(values, scores), values)


def test_rank_reassign_frame_by_day():
    frame = pd.DataFrame({
        "delivery_day": [1, 1, 1, 2, 2],
        "v": [10.0, 20.0, 30.0, 5.0, 6.0],
        "s": [3.0, 2.0, 1.0, 0.0, 1.0],
    })
    out = strategies.rank_reassign_frame(frame, "v", "s")
    np.testing.assert_array_equal(out.to_numpy(), [30.0, 20.0, 10.0, 5.0, 6.0])


def test_naive_vectors_use_previous_days():
    grid = data.hourly_grid(date(2024, 3, 1), date(2024, 3, 10))
    hourly = data.add_time_columns(pd.DataFrame(index=grid)).reset_index()
    hourly["price"] = np.arange(len(hourly), dtype=float)  # synthetic
    v = strategies.baseline_price_vectors(hourly)
    d = v[v["delivery_day"] == date(2024, 3, 10)]
    np.testing.assert_array_equal(d["S-naive-1d"], d["price"] - 24)
    np.testing.assert_array_equal(d["S-naive-7d"], d["price"] - 7 * 24)
    np.testing.assert_array_equal(d["S-perfect"], d["price"])


def test_risk_metrics_on_known_numbers():
    profit = np.arange(1, 101, dtype=float) - 10  # -9 .. 90
    assert risk.value_at_risk(profit) == pytest.approx(np.quantile(profit, 0.05))
    assert risk.expected_shortfall(profit) == pytest.approx(np.mean(profit[:5]))
    assert risk.losing_day_share(profit) == pytest.approx(0.09)
    assert risk.max_drawdown([5, -3, -4, 10, -1]) == pytest.approx(7)
    assert risk.max_drawdown([-2, 1]) == pytest.approx(2)  # loss from the starting point


def test_block_bootstrap_is_reproducible_and_centred():
    rng = np.random.default_rng(3)
    x = rng.normal(2.0, 1.0, 365)
    m1, lo1, hi1 = evaluate.block_bootstrap_ci(x, n_resamples=2000)
    m2, lo2, hi2 = evaluate.block_bootstrap_ci(x, n_resamples=2000)
    assert (m1, lo1, hi1) == (m2, lo2, hi2)
    assert lo1 < m1 < hi1 and lo1 > 1.5 and hi1 < 2.5
