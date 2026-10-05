"""XGBoost model checks: the ranking objective, labels, training rows and the search space."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from scipy.stats import spearmanr

from bessrank import config, evaluate, features, models


def test_rank_objective_is_the_all_pairs_logistic_loss():
    """XGB-rank's settings give the same trees as a hand-written objective: sum over every pair
    with label_i > label_j of log(1 + exp(-(s_i - s_j))). XGBoost uses twice the textbook
    hessian, which the reference copies."""
    rng = np.random.default_rng(1)
    n_days, n = 60, 24
    X = rng.normal(size=(n_days * n, 5))
    qid = np.repeat(np.arange(n_days), n)
    price = np.round(X[:, 0] + 0.5 * rng.normal(size=n_days * n), 1)  # rounding creates ties
    labels = np.concatenate([np.unique(price[qid == q], return_inverse=True)[1] for q in range(n_days)])
    dtrain = xgb.DMatrix(X, label=labels.astype(float), qid=qid)
    base = {"tree_method": "hist", "max_depth": 3, "eta": 0.3, "seed": 0, "nthread": 1, "base_score": 0.0}

    def all_pairs(preds, _):
        grad, hess = np.zeros_like(preds), np.zeros_like(preds)
        for q in range(n_days):
            idx = np.where(qid == q)[0]
            s, lab = preds[idx], labels[idx]
            higher = lab[:, None] > lab[None, :]
            sig = 1 / (1 + np.exp(-(s[:, None] - s[None, :])))
            g = -(1 - sig) * higher
            h = sig * (1 - sig) * higher
            grad[idx] += g.sum(axis=1) - g.sum(axis=0)
            hess[idx] += h.sum(axis=1) + h.sum(axis=0)
        return grad, 2 * hess

    reference = xgb.train({**base, "disable_default_eval_metric": 1}, dtrain, 5, obj=all_pairs).predict(dtrain)
    ours = xgb.train({**base, **models.OBJECTIVES["xgb-rank"]}, dtrain, 5).predict(dtrain)
    np.testing.assert_allclose(ours, reference, atol=1e-6)


def test_rank_labels_dense_with_ties():
    frame = pd.DataFrame({"delivery_day": [date(2024, 1, 1)] * 4 + [date(2024, 1, 2)] * 3,
                          "price": [30.0, -5.0, 30.0, 80.0, 1.0, 2.0, 1.0]})
    assert models.rank_labels(frame).tolist() == [1, 0, 1, 2, 0, 1, 0]


def test_early_stopping_start():
    assert models.es_start(date(2024, 9, 30)) == date(2024, 7, 1)
    assert models.es_start(date(2025, 12, 31)) == date(2025, 10, 1)
    assert models.es_start(date(2026, 3, 31)) == date(2026, 1, 1)


def _frame(first_day, n_days):
    rows = []
    for k in range(n_days):
        day = first_day + timedelta(days=k)
        ts = pd.Timestamp(day, tz="UTC") + pd.to_timedelta(np.arange(24), unit="h")
        rows.append(pd.DataFrame({"ts_utc": ts, "delivery_day": day, "hour": range(24), "price": 1.0}))
    frame = pd.concat(rows, ignore_index=True)
    for col in features.FEATURE_COLUMNS:
        frame[col] = 0.0
    return frame.sample(frac=1, random_state=0)  # shuffled


def test_training_rows_sorted_and_incomplete_days_dropped():
    frame = _frame(date(2024, 1, 1), 5)
    frame.loc[frame["delivery_day"] == date(2024, 1, 3), "pv_fc"] = np.nan
    rows = models.training_rows(frame, date(2024, 1, 1), date(2024, 1, 5))
    assert date(2024, 1, 3) not in set(rows["delivery_day"])
    assert rows["ts_utc"].is_monotonic_increasing
    # forecast days are never dropped
    assert models.forecast_rows(frame, date(2024, 1, 1), date(2024, 1, 5))["delivery_day"].nunique() == 5


def test_training_rows_refuse_test_period(monkeypatch):
    monkeypatch.delenv("BESS_UNLOCK_TEST", raising=False)
    frame = _frame(config.TEST_START - timedelta(days=2), 4)
    with pytest.raises(config.LockedPeriodError):
        models.training_rows(frame, config.VAL_START, config.TEST_START)
    with pytest.raises(config.LockedPeriodError):
        models.forecast_rows(frame, config.VAL_START, config.TEST_START)


def test_within_day_spearman_matches_scipy():
    rng = np.random.default_rng(0)
    lengths = [23, 24, 25, 24]
    days = np.repeat(np.arange(4), lengths)
    actual = np.round(rng.normal(size=days.size), 1)
    pred = rng.normal(size=days.size)
    pred[days == 3] = 1.0  # flat forecast: no rho for that day
    rho = evaluate.within_day_spearman(pred, actual, days)
    for d in range(3):
        assert rho[d] == pytest.approx(spearmanr(pred[days == d], actual[days == d]).statistic, abs=1e-12)
    assert np.isnan(rho[3])
    assert evaluate.mean_within_day_spearman(pred, actual, days) == pytest.approx(rho[:3].mean())


def test_search_space():
    configs = models.sample_configs()
    assert len(configs) == 15 and configs == models.sample_configs()
    for c in configs:
        assert 3 <= c["max_depth"] <= 10 and 0.01 <= c["learning_rate"] <= 0.2
        assert 1 <= c["min_child_weight"] <= 20 and 0.6 <= c["subsample"] <= 1
        assert 0.5 <= c["colsample_bytree"] <= 1 and 0.1 <= c["reg_lambda"] <= 10
