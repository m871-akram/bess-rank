"""Project constants (periods, battery, paths) and the test-period lock.

Every module takes its dates and parameters from here, so a change is made in one place.
"""
import os
import re
from datetime import date
from pathlib import Path

# --- Market and time zones -------------------------------------------------------------
# Timestamps are stored in UTC; delivery days are German calendar days (23, 24 or 25 hours).
MARKET_TZ = "Europe/Berlin"
REGION = "DE-LU"  # bidding zone Germany-Luxembourg

# Decision for delivery day D is taken at 11:00 Europe/Berlin on D-1 (one hour before the
# 12:00 day-ahead auction gate closure).
DECISION_HOUR = 11

# --- Periods (delivery days, both ends inclusive) ---------------------------------------
DATA_START = date(2018, 10, 1)  # every SMARD series used starts here for DE-LU
# Training starts after the 2018 run-in of the TSO load forecast (411), which has 21 gaps of
# 1-4 days between 2018-10-02 and 2018-12-31 (PLAN.md §2, §12 2026-10-05). The earlier days
# still feed the lagged features of January 2019.
TRAIN_START = date(2019, 1, 1)
TRAIN_END = date(2024, 9, 30)
VAL_START = date(2024, 10, 1)
VAL_END = date(2025, 9, 30)
TEST_START = date(2025, 10, 1)
TEST_END = date(2026, 9, 30)
DATA_END = TEST_END

# From this delivery day the day-ahead auction clears in 15-minute slots. The study stays
# hourly: the hourly price is the mean of the four quarter-hour prices.
QUARTER_HOUR_START = date(2025, 10, 1)

# --- Battery (PLAN.md §5) ---------------------------------------------------------------
POWER_MW = 1.0
CAPACITY_MWH = 2.0
ROUND_TRIP_EFFICIENCY = 0.88
ETA_CHARGE = ROUND_TRIP_EFFICIENCY ** 0.5  # split evenly between charging and discharging
ETA_DISCHARGE = ROUND_TRIP_EFFICIENCY ** 0.5
SOC_START_MWH = 1.0  # 50%, also the required end-of-day level
MAX_DISCHARGE_MWH_PER_DAY = 2.0  # at most one equivalent full cycle per day
DEGRADATION_EUR_PER_MWH = 10.0  # charged on energy discharged

# --- Models and strategies (PLAN.md §4, §5) ----------------------------------------------
MODELS = ["xgb-reg", "xgb-rank", "lstm-reg", "lstm-rank"]
STRATEGIES = ["S-perfect", "S-naive-1d", "S-naive-7d", "S-xgb-reg", "S-xgb-rank", "S-lstm-reg", "S-lstm-rank"]
# Controlled comparisons: within each family only the within-day order differs (§5).
PAIRS = [("S-xgb-rank", "S-xgb-reg"), ("S-lstm-rank", "S-lstm-reg")]

# --- Randomness and uncertainty ---------------------------------------------------------
SEEDS = (0, 1, 2)
BOOTSTRAP_SEED = 20261004
BOOTSTRAP_BLOCK_DAYS = 7
BOOTSTRAP_RESAMPLES = 10_000

# --- Paths ------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # one JSON file per SMARD chunk
PROCESSED_DIR = DATA_DIR / "processed"
HOURLY_PARQUET = PROCESSED_DIR / "hourly.parquet"
HOURLY_META = PROCESSED_DIR / "hourly_meta.json"  # download time, raw duplicate counts
FEATURES_PARQUET = PROCESSED_DIR / "features.parquet"
PREDICTIONS_DIR = DATA_DIR / "predictions"  # validation forecasts of every fitted model (S2)
RESULTS_DIR = ROOT / "results"
PROVENANCE_JSON = RESULTS_DIR / "provenance.json"
LOCK_FILE = ROOT / "PREREGISTRATION.lock"

# --- Databricks (Free Edition) ----------------------------------------------------------
DBX_CATALOG = "workspace"
DBX_SCHEMA = "bess"
DBX_VOLUME = "raw"
DBX_VOLUME_PATH = f"/Volumes/{DBX_CATALOG}/{DBX_SCHEMA}/{DBX_VOLUME}"


# --- Test-period lock (RULES.md rule 1) ------------------------------------------------
class LockedPeriodError(RuntimeError):
    """Raised when code asks for test-period data while the lock is closed."""


def is_test_unlocked(lock_file=None):
    """True only if BESS_UNLOCK_TEST=1 is set AND the lock file holds a git commit hash.

    The lock file is created after the pre-registration is merged (PLAN.md §6), so the
    environment variable alone is never enough.
    """
    lock_file = Path(lock_file) if lock_file is not None else LOCK_FILE
    if os.environ.get("BESS_UNLOCK_TEST") != "1":
        return False
    if not lock_file.is_file():
        return False
    content = lock_file.read_text().strip()
    return re.fullmatch(r"[0-9a-f]{40}", content) is not None


def require_test_unlocked(lock_file=None):
    """Stop with a clear message unless the test period has been unlocked."""
    if not is_test_unlocked(lock_file):
        raise LockedPeriodError(
            f"Delivery days from {TEST_START} onward are the locked test set. They can be "
            "loaded only when BESS_UNLOCK_TEST=1 is set and PREREGISTRATION.lock holds the "
            "commit hash of the merged pre-registration."
        )


def check_days_allowed(days, lock_file=None):
    """Raise if any delivery day is in the test period and the lock is closed.

    `days` is an iterable of datetime.date, e.g. the `delivery_day` column of the hourly table.
    """
    if any(day >= TEST_START for day in days):
        require_test_unlocked(lock_file)
