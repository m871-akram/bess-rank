"""Data stage: SMARD download, Databricks cache, hourly table and data-quality report.

`python -m bessrank.run data` calls `build_data()`; `python -m bessrank.run qa` calls
`write_qa_report()`. Everything else reads the table through `load_hourly()`, which keeps the
test period locked (RULES.md rule 1).
"""
import itertools
import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from bessrank import config, databricks

SMARD_URL = "https://www.smard.de/app/chart_data"

# SMARD filter ids for region DE-LU. Volumes are MWh per hour, i.e. average MW over the hour.
SERIES = {
    "price": 4169,  # day-ahead price, EUR/MWh
    "wind_onshore_fc": 123,  # TSO day-ahead generation forecasts
    "wind_offshore_fc": 3791,
    "pv_fc": 125,
    "load_fc": 411,  # TSO day-ahead load forecast
    "load_actual": 410,
    "residual_load_actual": 4359,
}
# Ready-made SMARD sums, downloaded for one week only to check our derived series (PLAN.md §2).
CHECK_SERIES = {"wind_pv_fc_smard": 5097, "residual_load_fc_smard": 4362}
ID_CHECK_WEEK = date(2024, 6, 10)  # a Monday in the training period

MAX_CONCURRENT = 4  # at most 4 requests in flight, to stay polite to SMARD
MAX_REQUESTS_PER_SECOND = 10  # and at most ~10 requests started per second (all threads)
MAX_ATTEMPTS = 5  # waits of 1, 2, 4, 8 s between attempts

# Harmonised day-ahead clearing price limits (SDAC), used as a format check only.
PRICE_MIN, PRICE_MAX = -500.0, 4000.0

_thread_local = threading.local()
_download_counter = itertools.count(1)
_rate_lock = threading.Lock()
_next_request_time = 0.0


# --- SMARD download -----------------------------------------------------------------------
def _session():
    """One requests.Session per worker thread (sessions are not shared across threads)."""
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


def _wait_for_rate_limit():
    """Space request starts at least 1/MAX_REQUESTS_PER_SECOND apart, across all threads."""
    global _next_request_time
    with _rate_lock:
        now = time.monotonic()
        start = max(now, _next_request_time)
        _next_request_time = start + 1.0 / MAX_REQUESTS_PER_SECOND
    time.sleep(max(0.0, start - now))


def fetch_json(url):
    """GET a SMARD JSON file, retrying with exponential backoff."""
    error = None
    for attempt in range(MAX_ATTEMPTS):
        _wait_for_rate_limit()
        try:
            resp = _session().get(url, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            error = f"HTTP {resp.status_code}"
        except (requests.RequestException, ValueError) as exc:
            error = repr(exc)
        if attempt < MAX_ATTEMPTS - 1:
            time.sleep(2**attempt)
    raise RuntimeError(f"SMARD request failed after {MAX_ATTEMPTS} attempts: {url} ({error})")


def berlin_midnight_ms(day):
    """Epoch milliseconds of 00:00 Europe/Berlin on `day` (SMARD's timestamp format)."""
    local = datetime(day.year, day.month, day.day, tzinfo=ZoneInfo(config.MARKET_TZ))
    return int(local.timestamp() * 1000)


def chunk_timestamps(filter_id, resolution, first_day, last_day):
    """Start times of the weekly SMARD chunks that overlap delivery days first_day..last_day."""
    url = f"{SMARD_URL}/{filter_id}/{config.REGION}/index_{resolution}.json"
    stamps = sorted(fetch_json(url)["timestamps"])
    start_ms = berlin_midnight_ms(first_day)
    end_ms = berlin_midnight_ms(last_day + timedelta(days=1))
    # A chunk runs from its own timestamp to the next one in the index.
    ends = stamps[1:] + [float("inf")]
    return [t for t, t_end in zip(stamps, ends) if t < end_ms and t_end > start_ms]


def fetch_chunk(filter_id, resolution, ts):
    """Return one weekly chunk, from data/raw if already downloaded."""
    name = f"{filter_id}_{config.REGION}_{resolution}_{ts}.json"
    path = config.RAW_DIR / name
    if path.exists():
        return json.loads(path.read_text())
    chunk = fetch_json(f"{SMARD_URL}/{filter_id}/{config.REGION}/{name}")
    path.write_text(json.dumps(chunk))
    n = next(_download_counter)
    if n % 250 == 0:
        print(f"  {n} chunks downloaded", flush=True)
    return chunk


def download_raw(first_day=config.DATA_START, last_day=config.DATA_END):
    """Download the 7 hourly series and the quarter-hour prices; return {name: [[ms, value]]}.

    Quarter-hour prices are only needed from QUARTER_HOUR_START, when the auction moved to
    15-minute products.
    """
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []  # (series name, filter id, resolution, chunk timestamp)
    for name, filter_id in SERIES.items():
        for ts in chunk_timestamps(filter_id, "hour", first_day, last_day):
            jobs.append((name, filter_id, "hour", ts))
    qh_first = max(first_day, config.QUARTER_HOUR_START)
    for ts in chunk_timestamps(SERIES["price"], "quarterhour", qh_first, last_day):
        jobs.append(("price_qh", SERIES["price"], "quarterhour", ts))
    print(f"  {len(jobs)} chunks needed", flush=True)

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
        chunks = list(pool.map(lambda job: fetch_chunk(*job[1:]), jobs))

    raw = {}
    for job, chunk in zip(jobs, chunks):
        raw.setdefault(job[0], []).extend(chunk["series"])
    return raw


# --- Hourly table -------------------------------------------------------------------------
def points_to_series(points, name):
    """SMARD [epoch_ms, value] pairs -> float Series on a UTC index, plus the duplicate count."""
    frame = pd.DataFrame(points, columns=["ms", name])
    n_duplicates = int(frame.duplicated("ms").sum())
    frame = frame.drop_duplicates("ms", keep="last").sort_values("ms")
    index = pd.DatetimeIndex(pd.to_datetime(frame["ms"], unit="ms", utc=True), name="ts_utc")
    return pd.Series(frame[name].astype(float).to_numpy(), index=index, name=name), n_duplicates


def hourly_grid(first_day, last_day):
    """UTC start time of every hour from first_day 00:00 to last_day 24:00 Europe/Berlin."""
    start = pd.Timestamp(first_day).tz_localize(config.MARKET_TZ).tz_convert("UTC")
    end = pd.Timestamp(last_day + timedelta(days=1)).tz_localize(config.MARKET_TZ).tz_convert("UTC")
    return pd.date_range(start, end, freq="h", inclusive="left", name="ts_utc")


def add_time_columns(table):
    """Add delivery-day columns derived from the UTC index. Days have 23, 24 or 25 hours."""
    local = table.index.tz_convert(config.MARKET_TZ)
    table = table.copy()
    table.insert(0, "delivery_day", local.date)
    # `hour` is the position within the delivery day (0..n-1), so it is unique even on the
    # 25-hour day, where the wall-clock hour 02:00 occurs twice.
    table.insert(1, "hour", table.groupby("delivery_day").cumcount())
    table.insert(2, "local_hour", local.hour)
    table.insert(3, "n_hours", table.groupby("delivery_day")["hour"].transform("size"))
    return table


def quarter_hours_to_hourly(qh):
    """Mean quarter-hour price per hour and the number of quarter-hours present.

    Germany's UTC offset is a whole number of hours, so flooring to the hour in UTC gives
    the same hours as flooring in local time.
    """
    grouped = qh.groupby(qh.index.floor("h"))
    return pd.DataFrame({"price_qh_mean": grouped.mean(), "n_quarter_hours": grouped.count()})


def build_hourly_table(raw, first_day=config.DATA_START, last_day=config.DATA_END):
    """Put every series on one hourly UTC grid and add the delivery-day columns.

    Returns the table (one row per delivery hour, `ts_utc` as a column) and per-series raw
    checks (duplicate timestamps, timestamps off the hourly grid) for the QA report.
    """
    grid = hourly_grid(first_day, last_day)
    table = pd.DataFrame(index=grid)
    raw_checks = {}
    for name in SERIES:
        series, n_dup = points_to_series(raw[name], name)
        in_range = series.index[(series.index >= grid[0]) & (series.index <= grid[-1])]
        raw_checks[name] = {"duplicates": n_dup, "off_grid": int(in_range.difference(grid).size)}
        table[name] = series.reindex(grid)

    table = add_time_columns(table)

    # Market choice: from QUARTER_HOUR_START the study price is the mean of the four
    # quarter-hour prices, which is exact for a battery holding constant power within the
    # hour. SMARD's own hourly series is kept for the QA comparison.
    qh, n_dup = points_to_series(raw["price_qh"], "price_qh")
    raw_checks["price_qh"] = {"duplicates": n_dup}
    hourly_qh = quarter_hours_to_hourly(qh).reindex(grid)
    table["price_smard_hourly"] = table["price"]
    table["n_quarter_hours"] = hourly_qh["n_quarter_hours"].fillna(0).astype(int)
    in_qh_period = table["delivery_day"] >= config.QUARTER_HOUR_START
    qh_price = hourly_qh["price_qh_mean"].where(table["n_quarter_hours"] == 4)
    table.loc[in_qh_period, "price"] = qh_price[in_qh_period]
    return table.reset_index(), raw_checks


# --- ID checks (PLAN.md §2) ---------------------------------------------------------------
def _week_frame(filter_ids, first_day, last_day):
    """Hourly values of several SMARD filters for a few days, on the hourly grid."""
    frame = {}
    for name, filter_id in filter_ids.items():
        stamps = chunk_timestamps(filter_id, "hour", first_day, last_day)
        points = [p for ts in stamps for p in fetch_chunk(filter_id, "hour", ts)["series"]]
        frame[name] = points_to_series(points, name)[0]
    return pd.DataFrame(frame).reindex(hourly_grid(first_day, last_day))


def check_series_ids(week_start=ID_CHECK_WEEK):
    """Check, on one training-period week, that 411 is a load forecast and that our derived
    series equal SMARD's 5097 (wind + PV forecast) and 4362 (forecast residual load)."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    last_day = week_start + timedelta(days=6)
    names = ["wind_onshore_fc", "wind_offshore_fc", "pv_fc", "load_fc", "load_actual"]
    week = _week_frame({**{n: SERIES[n] for n in names}, **CHECK_SERIES}, week_start, last_day)
    wind_pv = week["wind_onshore_fc"] + week["wind_offshore_fc"] + week["pv_fc"]
    residual = week["load_fc"] - wind_pv
    load_error = (week["load_fc"] - week["load_actual"]).abs() / week["load_actual"]

    # A forecast is published before delivery, so 411 should have values for hours where
    # actual load (410) is not yet known.
    lead_hours = (_last_published_ms(SERIES["load_fc"]) - _last_published_ms(SERIES["load_actual"])) // 3_600_000

    return {
        "week": f"{week_start} to {last_day}",
        "hours": int(len(week)),
        "missing_values": int(week.isna().sum().sum()),
        "max_abs_diff_windpv_vs_5097_MWh": float((wind_pv - week["wind_pv_fc_smard"]).abs().max()),
        "max_abs_diff_residual_vs_4362_MWh": float((residual - week["residual_load_fc_smard"]).abs().max()),
        "load_fc_vs_410_mape_pct": float(load_error.mean() * 100),
        "load_fc_vs_410_corr": float(week["load_fc"].corr(week["load_actual"])),
        "load_fc_equal_to_410_hours": int((week["load_fc"] == week["load_actual"]).sum()),
        "load_fc_hours_published_beyond_410": int(lead_hours),
    }


def _last_published_ms(filter_id):
    """Timestamp of the latest non-missing value of a series. Only the timestamp is used,
    never the values, and the two newest chunks are not cached because they still grow."""
    stamps = sorted(fetch_json(f"{SMARD_URL}/{filter_id}/{config.REGION}/index_hour.json")["timestamps"])
    points = []
    for ts in stamps[-2:]:
        points += fetch_json(f"{SMARD_URL}/{filter_id}/{config.REGION}/{filter_id}_{config.REGION}_hour_{ts}.json")["series"]
    return max(ms for ms, value in points if value is not None)


# --- Databricks cache ---------------------------------------------------------------------
VOLUME_PARQUET = "processed/hourly.parquet"
VOLUME_META = "processed/hourly_meta.json"


def pull_from_databricks():
    """Fetch the processed parquet and its metadata from the volume. True on success."""
    if not databricks.available():
        return False
    return databricks.download_file(VOLUME_PARQUET, config.HOURLY_PARQUET) and databricks.download_file(
        VOLUME_META, config.HOURLY_META
    )


def push_to_databricks():
    """Upload the processed parquet and its metadata to the volume. True on success."""
    if not databricks.available():
        return False
    return databricks.upload_file(config.HOURLY_PARQUET, VOLUME_PARQUET) and databricks.upload_file(
        config.HOURLY_META, VOLUME_META
    )


def build_data(refresh=False):
    """Rebuild data/processed: from the Databricks volume if possible, otherwise from SMARD.

    refresh=True skips the volume and clears the raw cache, forcing a fresh SMARD download.
    """
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if not refresh and pull_from_databricks():
        print("Pulled the hourly table from the Databricks volume.")
        return json.loads(config.HOURLY_META.read_text())

    if refresh and config.RAW_DIR.exists():
        shutil.rmtree(config.RAW_DIR)
    print("Downloading from SMARD ...", flush=True)
    started = time.time()
    raw = download_raw()
    table, raw_checks = build_hourly_table(raw)
    table.to_parquet(config.HOURLY_PARQUET, index=False)

    # Download time = when the raw chunks were fetched (they may come from an earlier run).
    mtimes = [p.stat().st_mtime for p in config.RAW_DIR.glob("*.json")]
    meta = {
        "source": "SMARD chart API (Bundesnetzagentur | SMARD.de)",
        "smard_download_utc_first": datetime.fromtimestamp(min(mtimes), timezone.utc).isoformat(timespec="seconds"),
        "smard_download_utc_last": datetime.fromtimestamp(max(mtimes), timezone.utc).isoformat(timespec="seconds"),
        "series_filters": SERIES,
        "first_day": str(config.DATA_START),
        "last_day": str(config.DATA_END),
        "rows": int(len(table)),
        "raw_checks": raw_checks,
        "id_check": check_series_ids(),
    }
    config.HOURLY_META.write_text(json.dumps(meta, indent=2))
    print(f"Built {config.HOURLY_PARQUET.name}: {len(table)} hourly rows in {time.time() - started:.0f} s.")

    if push_to_databricks():
        print("Uploaded the hourly table to the Databricks volume.")
    else:
        print("Databricks volume not reachable; the table stays local only.")
    return meta


def load_hourly(include_test=False, fill_gaps=True):
    """Read the hourly table. Test-period days are dropped unless the lock is open.

    Short gaps are filled after the test days are dropped, so a validation hour is never
    interpolated from a test-period value while the lock is closed.
    """
    table = pd.read_parquet(config.HOURLY_PARQUET)
    if include_test:
        config.require_test_unlocked()
    else:
        table = table[table["delivery_day"] < config.TEST_START].reset_index(drop=True)
    return fill_short_gaps(table) if fill_gaps else table


# --- Missing values (PLAN.md §2) ----------------------------------------------------------
MAX_FILL_GAP_HOURS = 2  # gaps of 1-2 hours are interpolated; longer gaps stay missing


def gap_lengths(missing):
    """For each row, the length of the run of missing hours it belongs to (0 if not missing)."""
    runs = missing.ne(missing.shift()).cumsum()
    return missing.groupby(runs).transform("sum").where(missing, 0).astype(int)


def fill_short_gaps(table, columns=tuple(SERIES)):
    """Fill gaps of 1-2 hours by linear interpolation in UTC time from the neighbouring hours
    of the same series. Longer gaps, and gaps at either end of the table, stay NaN.

    This covers the 00:00 hour that SMARD misses for 123, 125 and 411 on 25-hour days.
    Adds a boolean `<series>_filled` column per series so every filled value can be traced.
    """
    table = table.sort_values("ts_utc").reset_index(drop=True)
    steps = table["ts_utc"].diff().dropna()
    if not (steps == pd.Timedelta(hours=1)).all():
        raise ValueError("fill_short_gaps needs a contiguous hourly UTC grid")
    for name in columns:
        values = table[name]
        missing = values.isna()
        has_both_neighbours = values.ffill().notna() & values.bfill().notna()
        short = missing & has_both_neighbours & (gap_lengths(missing) <= MAX_FILL_GAP_HOURS)
        # The grid is evenly spaced in UTC, so interpolating by position is linear in time.
        interpolated = values.interpolate(method="linear", limit_area="inside")
        table[name] = values.where(~short, interpolated)
        table[f"{name}_filled"] = short
    return table


def long_gaps(table, column):
    """Runs of more than MAX_FILL_GAP_HOURS missing hours: first/last UTC hour and length."""
    missing = table[column].isna()
    runs = missing.ne(missing.shift()).cumsum()
    gaps = table[missing].groupby(runs[missing]).agg(
        first_utc=("ts_utc", "first"), last_utc=("ts_utc", "last"),
        first_day=("delivery_day", "first"), last_day=("delivery_day", "last"),
        hours=("ts_utc", "size"))
    return gaps[gaps["hours"] > MAX_FILL_GAP_HOURS].reset_index(drop=True)


# --- Data-quality report ------------------------------------------------------------------
def _longest_gap(missing):
    """Length of the longest run of consecutive True values."""
    runs = missing.ne(missing.shift()).cumsum()
    lengths = missing.groupby(runs).sum()
    return int(lengths.max()) if len(lengths) else 0


def _markdown_table(frame):
    """Render a small DataFrame as a markdown table (display only)."""
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "---|" * len(frame.columns)
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def _test_period_checks(test, test_filled):
    """Format checks on the test period. Returns counts and pass/fail only, never values.

    `test_filled` is the same period after fill_short_gaps; only its NaN counts are used.
    """
    days = pd.date_range(config.TEST_START, config.TEST_END, freq="D").date
    expected_hours = len(hourly_grid(config.TEST_START, config.TEST_END))
    n_hours = test.groupby("delivery_day")["n_hours"].first()
    checks = [
        ("Delivery days present", f"{test['delivery_day'].nunique()} of {len(days)}",
         test["delivery_day"].nunique() == len(days)),
        ("Hourly rows", f"{len(test)} of {expected_hours}", len(test) == expected_hours),
        ("Duplicate timestamps", int(test["ts_utc"].duplicated().sum()), not test["ts_utc"].duplicated().any()),
        ("2025-10-26 has 25 hours", "", n_hours.get(date(2025, 10, 26)) == 25),
        ("2026-03-29 has 23 hours", "", n_hours.get(date(2026, 3, 29)) == 23),
    ]
    # Missing hours before and after filling gaps of 1-2 hours (PLAN.md §2). A raw gap that
    # the rule fills is reported as "filled", not as a failure.
    for name in SERIES:
        n_missing = int(test[name].isna().sum())
        n_left = int(test_filled[name].isna().sum())
        checks.append((f"Missing hours: {name} (raw / after filling 1-2 h gaps)", f"{n_missing} / {n_left}",
                       True if n_missing == 0 else ("filled" if n_left == 0 else False)))
    days_left = int(test_filled.loc[test_filled[list(SERIES)].isna().any(axis=1), "delivery_day"].nunique())
    checks.append(("Days with a gap > 2 h after filling (kept; NaN for XGBoost, median for the LSTM)",
                   days_left, days_left == 0))
    bad_qh = int((test["n_quarter_hours"] != 4).sum())
    checks.append(("Hours without exactly 4 quarter-hour prices", bad_qh, bad_qh == 0))
    out_of_range = int(((test["price"] < PRICE_MIN) | (test["price"] > PRICE_MAX)).sum())
    checks.append((f"Prices outside [{PRICE_MIN:.0f}, {PRICE_MAX:.0f}] EUR/MWh", out_of_range, out_of_range == 0))
    both = test["price"].notna() & test["price_smard_hourly"].notna()
    mismatch = int(((test["price"] - test["price_smard_hourly"]).abs() > 0.01)[both].sum())
    checks.append(("SMARD hourly price missing (hours)", int(test["price_smard_hourly"].isna().sum()),
                   test["price_smard_hourly"].notna().all()))
    checks.append((f"SMARD hourly price differs from quarter-hour mean by > 0.01 EUR/MWh "
                   f"(of {int(both.sum())} hours)", mismatch, mismatch == 0))
    return checks


def missing_value_counts(table, first_day, last_day):
    """Counts for the PLAN.md §2 missing-value rules in one period: hours filled by
    interpolation, hours still missing (gaps > 2 h), and days with a remaining gap."""
    window = table[(table["delivery_day"] >= first_day) & (table["delivery_day"] <= last_day)]
    remaining = window[list(SERIES)].isna()
    return {
        "filled_hours": {n: int(window[f"{n}_filled"].sum()) for n in SERIES},
        "missing_hours": {n: int(remaining[n].sum()) for n in SERIES},
        "days_with_remaining_gap": int(window.loc[remaining.any(axis=1), "delivery_day"].nunique()),
        "days": int(window["delivery_day"].nunique()),
    }


def _missing_value_section(pre):
    """QA lines for the missing-value rules on the training and validation periods."""
    filled = fill_short_gaps(pre)  # as load_hourly() does: test days excluded
    lines = [
        "### Missing values after the PLAN.md §2 rules",
        "",
        f"Gaps of 1-{MAX_FILL_GAP_HOURS} hours are filled by linear interpolation in UTC time. "
        "Training days with a longer gap are dropped; validation and test days are kept (NaN for "
        "XGBoost, training median for the LSTM).",
        "",
    ]
    gaps_411 = long_gaps(pre, "load_fc")
    run_in = gaps_411[gaps_411["first_day"] < config.TRAIN_START]
    later = gaps_411[gaps_411["first_day"] >= config.TRAIN_START]
    removed = (config.TRAIN_START - config.DATA_START).days
    lines += [
        f"- Load forecast 411: {len(gaps_411)} gaps longer than {MAX_FILL_GAP_HOURS} h. "
        f"{len(run_in)} ({int(run_in['hours'].sum())} h) fall in the run-in from "
        f"{run_in['first_day'].min()} to {run_in['last_day'].max()}; training starts on "
        f"{config.TRAIN_START} instead of {config.DATA_START}, which removes {removed} days. "
        f"The {len(later)} later gaps ({int(later['hours'].sum())} h) are whole days: "
        + ", ".join(f"{a}" + (f" to {b}" if b != a else "") for a, b in zip(later["first_day"], later["last_day"]))
        + ".",
        "",
    ]
    rows = []
    for period, first, last in [("train", config.TRAIN_START, config.TRAIN_END),
                                ("validation", config.VAL_START, config.VAL_END)]:
        counts = missing_value_counts(filled, first, last)
        rows.append({"period": f"{period} ({first} to {last})",
                     "filled hours": _per_series(counts["filled_hours"]),
                     "hours still missing": _per_series(counts["missing_hours"]),
                     "days with a gap > 2 h": f"{counts['days_with_remaining_gap']} of {counts['days']}"})
    lines += [_markdown_table(pd.DataFrame(rows)), ""]
    return lines


def _per_series(counts):
    """'name n, ...' for the non-zero counts, or 0."""
    nonzero = [f"{name} {n}" for name, n in counts.items() if n]
    return ", ".join(nonzero) if nonzero else "0"


def write_qa_report(path=None):
    """Write results/qa_data.md (PLAN.md §2). The test period gets counts and pass/fail only."""
    path = path or config.RESULTS_DIR / "qa_data.md"
    # The full table is read here only to count test-period rows; no test value is reported.
    table = pd.read_parquet(config.HOURLY_PARQUET)
    meta = json.loads(config.HOURLY_META.read_text())
    pre = table[table["delivery_day"] < config.TEST_START]
    test = table[table["delivery_day"] >= config.TEST_START]
    year = pd.to_datetime(pre["delivery_day"]).dt.year

    lines = [
        "# Data quality report",
        "",
        f"Source: {meta['source']}. Region {config.REGION}. Raw chunks downloaded "
        f"{meta['smard_download_utc_first']} to {meta['smard_download_utc_last']} (UTC).",
        f"Hourly grid {meta['first_day']} to {meta['last_day']} (delivery days, Europe/Berlin): {meta['rows']} rows.",
        "",
        f"## Train and validation periods ({config.DATA_START} to {config.VAL_END})",
        "",
        "### Delivery days and DST",
        "",
    ]
    n_hours = pre.groupby("delivery_day")["n_hours"].first()
    lines += [
        f"- Days: {len(n_hours)}; hours: {len(pre)}; days with 23 / 24 / 25 hours: "
        f"{(n_hours == 23).sum()} / {(n_hours == 24).sum()} / {(n_hours == 25).sum()}.",
        f"- 23-hour days: {', '.join(str(d) for d in n_hours[n_hours == 23].index)}.",
        f"- 25-hour days: {', '.join(str(d) for d in n_hours[n_hours == 25].index)}.",
        f"- Duplicate UTC timestamps in the table: {int(pre['ts_utc'].duplicated().sum())}.",
        "",
        "### Missing hours per series and year (2018 is Oct-Dec, 2025 is Jan-Sep)",
        "",
    ]
    missing = pre[list(SERIES)].isna().groupby(year).sum()
    missing.insert(0, "hours", year.value_counts().sort_index())
    lines += [_markdown_table(missing.reset_index(names="year")), ""]

    gaps = pd.DataFrame({
        "series": list(SERIES),
        "missing hours": [int(pre[n].isna().sum()) for n in SERIES],
        "longest gap (h)": [_longest_gap(pre[n].isna()) for n in SERIES],
        "raw duplicate timestamps (all years)": [meta["raw_checks"][n]["duplicates"] for n in SERIES],
        "raw timestamps off the hourly grid": [meta["raw_checks"][n]["off_grid"] for n in SERIES],
    })
    lines += ["### Gaps and duplicates", "", _markdown_table(gaps), ""]
    lines += _missing_value_section(pre)

    price = pre["price"]
    neg = (price < 0).groupby(year).sum()
    out_of_range = int(((price < PRICE_MIN) | (price > PRICE_MAX)).sum())
    lines += [
        "### Day-ahead price",
        "",
        f"- Hours outside [{PRICE_MIN:.0f}, {PRICE_MAX:.0f}] EUR/MWh: {out_of_range}.",
        "- Negative-price hours per calendar year (2025: Jan-Sep only): "
        + ", ".join(f"{y}: {int(n)}" for y, n in neg.items()) + ".",
        "",
    ]

    idc = meta["id_check"]
    lines += [
        f"### Series ID checks (week {idc['week']}, training period)",
        "",
        f"- Hours: {idc['hours']}; missing values: {idc['missing_values']}.",
        f"- max |(123 + 3791 + 125) - 5097|: {idc['max_abs_diff_windpv_vs_5097_MWh']:.6g} MWh.",
        f"- max |(411 - wind - PV) - 4362|: {idc['max_abs_diff_residual_vs_4362_MWh']:.6g} MWh.",
        f"- 411 vs actual load 410: MAPE {idc['load_fc_vs_410_mape_pct']:.2f}%, correlation "
        f"{idc['load_fc_vs_410_corr']:.4f}, identical hours {idc['load_fc_equal_to_410_hours']}.",
        f"- 411 has values {idc['load_fc_hours_published_beyond_410']} hours beyond the last 410 value "
        "(it is published before delivery).",
        "",
        f"## Test period ({config.TEST_START} to {config.TEST_END}): counts and pass/fail only",
        "",
        "| check | count | result |",
        "|---|---|---|",
    ]
    filled_all = fill_short_gaps(table)  # counts only; nothing from the test period is shown
    test_filled = filled_all[filled_all["delivery_day"] >= config.TEST_START]
    checks = _test_period_checks(test, test_filled)
    label = {True: "pass", False: "FAIL", "filled": "filled"}
    lines += [f"| {name} | {count} | {label[ok]} |" for name, count, ok in checks]
    lines += [""]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    n_fail = sum(ok is False for _, _, ok in checks)
    print(f"Wrote {path.relative_to(config.ROOT)}; test-period checks failing: {n_fail}.")
    return checks
