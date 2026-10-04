"""The test period stays locked unless BESS_UNLOCK_TEST=1 AND PREREGISTRATION.lock holds a commit hash."""
from datetime import date

import pandas as pd
import pytest

from bessrank import config, data

COMMIT = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def lock(tmp_path):
    return tmp_path / "PREREGISTRATION.lock"


def test_locked_by_default(monkeypatch, lock):
    monkeypatch.delenv("BESS_UNLOCK_TEST", raising=False)
    assert not config.is_test_unlocked(lock)
    with pytest.raises(config.LockedPeriodError):
        config.require_test_unlocked(lock)


def test_env_variable_alone_is_not_enough(monkeypatch, lock):
    monkeypatch.setenv("BESS_UNLOCK_TEST", "1")
    assert not config.is_test_unlocked(lock)


def test_lock_file_alone_is_not_enough(monkeypatch, lock):
    monkeypatch.delenv("BESS_UNLOCK_TEST", raising=False)
    lock.write_text(COMMIT)
    assert not config.is_test_unlocked(lock)


@pytest.mark.parametrize("content", ["", "UNLOCK TEST", COMMIT[:7], COMMIT.upper()])
def test_lock_file_must_hold_a_full_commit_hash(monkeypatch, lock, content):
    monkeypatch.setenv("BESS_UNLOCK_TEST", "1")
    lock.write_text(content)
    assert not config.is_test_unlocked(lock)


@pytest.mark.parametrize("value", ["0", "true", "yes", ""])
def test_env_variable_must_be_exactly_1(monkeypatch, lock, value):
    monkeypatch.setenv("BESS_UNLOCK_TEST", value)
    lock.write_text(COMMIT)
    assert not config.is_test_unlocked(lock)


def test_env_variable_and_lock_file_unlock(monkeypatch, lock):
    monkeypatch.setenv("BESS_UNLOCK_TEST", "1")
    lock.write_text(COMMIT + "\n")
    assert config.is_test_unlocked(lock)
    config.require_test_unlocked(lock)  # does not raise


def test_check_days_allowed(monkeypatch, lock):
    monkeypatch.delenv("BESS_UNLOCK_TEST", raising=False)
    config.check_days_allowed([date(2024, 10, 1), config.VAL_END], lock)  # validation is fine
    with pytest.raises(config.LockedPeriodError):
        config.check_days_allowed([config.VAL_END, config.TEST_START], lock)


def test_load_hourly_drops_test_days_until_unlocked(monkeypatch, tmp_path, lock):
    # Synthetic table: one validation day and one test day (no real prices involved).
    days = [date(2025, 9, 30), date(2025, 10, 1)]
    table = pd.DataFrame({"delivery_day": days, "price": [1.0, 2.0]})
    parquet = tmp_path / "hourly.parquet"
    table.to_parquet(parquet, index=False)
    monkeypatch.setattr(config, "HOURLY_PARQUET", parquet)
    monkeypatch.setattr(config, "LOCK_FILE", lock)
    monkeypatch.delenv("BESS_UNLOCK_TEST", raising=False)

    loaded = data.load_hourly()
    assert list(loaded["delivery_day"]) == [date(2025, 9, 30)]
    with pytest.raises(config.LockedPeriodError):
        data.load_hourly(include_test=True)

    monkeypatch.setenv("BESS_UNLOCK_TEST", "1")
    lock.write_text(COMMIT)
    assert list(data.load_hourly(include_test=True)["delivery_day"]) == days
