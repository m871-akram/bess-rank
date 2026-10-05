"""Exploratory helpers (PLAN.md §8): PSI, quarter lookup and the oracle vectors."""
from datetime import date

import numpy as np
import pandas as pd

from bessrank import explore, risk


def test_psi_is_zero_for_the_same_sample_and_large_for_a_shift():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert risk.psi(x, x) == 0.0
    assert risk.psi(x, rng.normal(size=5000)) < 0.02
    assert risk.psi(x, x + 1.0) > 0.25


def test_psi_counts_missing_values_in_their_own_bin():
    x = np.arange(1000, dtype=float)
    y = x.copy()
    y[:100] = np.nan  # 10% of the test values missing, none in train
    assert risk.psi(x, y) > 0.1


def test_quarter_of_maps_days_to_the_test_quarter_start():
    days = [date(2025, 10, 1), date(2025, 12, 31), date(2026, 1, 1), date(2026, 6, 30), date(2026, 9, 30)]
    assert list(explore.quarter_of(days)) == [date(2025, 10, 1), date(2025, 10, 1), date(2026, 1, 1),
                                              date(2026, 4, 1), date(2026, 7, 1)]


def test_oracle_vectors_reassign_values_and_order():
    # One day of 4 hours. The price model has the right values in the wrong order.
    v = pd.DataFrame({"delivery_day": [date(2026, 1, 1)] * 4, "price": [10.0, 40.0, 20.0, 30.0]})
    v["xgb-reg"] = [40.0, 10.0, 20.0, 30.0]
    v["xgb-rank"] = [0.0, 3.0, 1.0, 2.0]  # the true order
    v["lstm-reg"], v["lstm-rank"] = v["price"], v["price"]
    out = explore.oracle_vectors(v)
    # model values in the true order = the true prices here
    assert out["xgb: true order, model values"].tolist() == [10.0, 40.0, 20.0, 30.0]
    # true prices in the price model's order
    assert out["xgb: reg order, true values"].tolist() == [40.0, 10.0, 20.0, 30.0]
    # true prices in the ranker's (correct) order
    assert out["xgb: rank order, true values"].tolist() == [10.0, 40.0, 20.0, 30.0]


def test_cqr_restores_coverage_of_too_narrow_quantiles():
    rng = np.random.default_rng(1)
    levels = risk.QUANTILE_LEVELS
    y_cal, y_new = rng.normal(size=4000), rng.normal(size=4000)
    # quantiles of N(0, 0.5^2): far too narrow for N(0, 1) data
    from scipy.stats import norm
    q = np.tile(norm.ppf(levels, scale=0.5), (4000, 1))
    assert risk.interval_coverage(q, y_new)[0.9] < 0.7
    calibrated = risk.apply_cqr(q, risk.cqr_corrections(q, y_cal))
    assert abs(risk.interval_coverage(calibrated, y_new)[0.9] - 0.9) < 0.02
    assert abs(risk.interval_coverage(calibrated, y_new)[0.5] - 0.5) < 0.03
    assert np.all(np.diff(calibrated, axis=1) >= 0)


def test_quantile_function_interpolates_and_extrapolates():
    q = np.arange(19, dtype=float)  # q at level 0.05 k + 0.05 equals k
    assert np.allclose(explore.quantile_function(q, np.array([0.05, 0.5, 0.95])), [0.0, 9.0, 18.0])
    assert np.allclose(explore.quantile_function(q, np.array([0.0, 1.0])), [-1.0, 19.0])
