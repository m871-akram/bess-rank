"""Test-run pieces that can be checked without test data: verdict rules, quarters, the lock."""
from datetime import date
from types import SimpleNamespace

import pytest

from bessrank import backtest, config, evaluate, run


@pytest.mark.parametrize("lower, upper, expected", [
    (0.1, 2.0, "supported"), (-2.0, -0.1, "contradicted"), (-0.1, 2.0, "inconclusive"),
    (0.0, 2.0, "inconclusive"),  # the lower bound must be strictly above 0
])
def test_superiority_verdict(lower, upper, expected):
    assert evaluate.verdict_superiority(lower, upper) == expected


@pytest.mark.parametrize("lower, upper, expected", [
    (-0.2, 0.2, "supported"),          # whole CI inside (-0.3, 0.3)
    (0.4, 0.9, "contradicted"),        # whole CI above +margin
    (-0.9, -0.4, "contradicted"),      # whole CI below -margin
    (-0.2, 0.5, "inconclusive"),       # crosses +margin
    (-0.5, 0.5, "inconclusive"),       # contains the whole margin
])
def test_equivalence_verdict(lower, upper, expected):
    assert evaluate.verdict_equivalence(lower, upper, margin=0.3) == expected


def test_test_quarters_cover_the_test_year():
    q = backtest.quarters(backtest.TEST_QUARTERS, config.TEST_END)
    assert q[0] == (date(2025, 10, 1), date(2025, 12, 31))
    assert q[-1] == (date(2026, 7, 1), date(2026, 9, 30))
    assert sum((last - first).days + 1 for first, last in q) == 365
    assert all(q[i][1].toordinal() + 1 == q[i + 1][0].toordinal() for i in range(3))


def test_rehearsal_quarters_cover_the_validation_year():
    q = backtest.quarters(backtest.REHEARSAL_QUARTERS, config.VAL_END)
    assert q[0][0] == config.VAL_START and q[-1][1] == config.VAL_END
    assert sum((last - first).days + 1 for first, last in q) == 365


def test_test_run_refuses_to_start_while_locked(monkeypatch):
    monkeypatch.delenv("BESS_UNLOCK_TEST", raising=False)
    with pytest.raises(config.LockedPeriodError):
        run.cmd_test(SimpleNamespace(rehearsal=False, amendment=False))
