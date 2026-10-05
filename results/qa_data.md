# Data quality report

Source: SMARD chart API (Bundesnetzagentur | SMARD.de). Region DE-LU. Raw chunks downloaded 2026-10-05T06:37:33+00:00 to 2026-10-05T06:42:30+00:00 (UTC).
Hourly grid 2018-10-01 to 2026-09-30 (delivery days, Europe/Berlin): 70128 rows.

## Train and validation periods (2018-10-01 to 2025-09-30)

### Delivery days and DST

- Days: 2557; hours: 61368; days with 23 / 24 / 25 hours: 7 / 2543 / 7.
- 23-hour days: 2019-03-31, 2020-03-29, 2021-03-28, 2022-03-27, 2023-03-26, 2024-03-31, 2025-03-30.
- 25-hour days: 2018-10-28, 2019-10-27, 2020-10-25, 2021-10-31, 2022-10-30, 2023-10-29, 2024-10-27.
- Duplicate UTC timestamps in the table: 0.

### Missing hours per series and year (2018 is Oct-Dec, 2025 is Jan-Sep)

| year | hours | price | wind_onshore_fc | wind_offshore_fc | pv_fc | load_fc | load_actual | residual_load_actual |
|---|---|---|---|---|---|---|---|---|
| 2018 | 2209 | 0 | 0 | 0 | 0 | 842 | 4 | 4 |
| 2019 | 8760 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2020 | 8784 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2021 | 8760 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2022 | 8760 | 0 | 0 | 0 | 0 | 144 | 0 | 0 |
| 2023 | 8760 | 0 | 1 | 0 | 1 | 49 | 0 | 0 |
| 2024 | 8784 | 0 | 1 | 0 | 1 | 1 | 0 | 0 |
| 2025 | 6551 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### Gaps and duplicates

| series | missing hours | longest gap (h) | raw duplicate timestamps (all years) | raw timestamps off the hourly grid |
|---|---|---|---|---|
| price | 0 | 0 | 0 | 0 |
| wind_onshore_fc | 2 | 1 | 0 | 0 |
| wind_offshore_fc | 0 | 0 | 0 | 0 |
| pv_fc | 2 | 1 | 0 | 0 |
| load_fc | 1036 | 96 | 0 | 0 |
| load_actual | 4 | 1 | 0 | 0 |
| residual_load_actual | 4 | 1 | 0 | 0 |

### Missing values after the PLAN.md §2 rules

Gaps of 1-2 hours are filled by linear interpolation in UTC time. Training days with a longer gap are dropped; validation and test days are kept (NaN for XGBoost, training median for the LSTM).

- Load forecast 411: 27 gaps longer than 2 h. 21 (840 h) fall in the run-in from 2018-10-02 to 2018-12-31; training starts on 2019-01-01 instead of 2018-10-01, which removes 92 days. The 6 later gaps (192 h) are whole days: 2022-02-22, 2022-03-24, 2022-07-20 to 2022-07-21, 2022-12-21 to 2022-12-22, 2023-03-13, 2023-08-31.

| period | filled hours | hours still missing | days with a gap > 2 h |
|---|---|---|---|
| train (2019-01-01 to 2024-09-30) | wind_onshore_fc 1, pv_fc 1, load_fc 1 | load_fc 192 | 8 of 2100 |
| validation (2024-10-01 to 2025-09-30) | wind_onshore_fc 1, pv_fc 1, load_fc 1 | 0 | 0 of 365 |

### Day-ahead price

- Hours outside [-500, 4000] EUR/MWh: 0.
- Negative-price hours per calendar year (2025: Jan-Sep only): 2018: 27, 2019: 211, 2020: 298, 2021: 139, 2022: 69, 2023: 301, 2024: 457, 2025: 525.

### Series ID checks (week 2024-06-10 to 2024-06-16, training period)

- Hours: 168; missing values: 0.
- max |(123 + 3791 + 125) - 5097|: 0 MWh.
- max |(411 - wind - PV) - 4362|: 0 MWh.
- 411 vs actual load 410: MAPE 3.86%, correlation 0.9886, identical hours 0.
- 411 has values 16 hours beyond the last 410 value (it is published before delivery).

## Test period (2025-10-01 to 2026-09-30): counts and pass/fail only

| check | count | result |
|---|---|---|
| Delivery days present | 365 of 365 | pass |
| Hourly rows | 8760 of 8760 | pass |
| Duplicate timestamps | 0 | pass |
| 2025-10-26 has 25 hours |  | pass |
| 2026-03-29 has 23 hours |  | pass |
| Missing hours: price (raw / after filling 1-2 h gaps) | 0 / 0 | pass |
| Missing hours: wind_onshore_fc (raw / after filling 1-2 h gaps) | 1 / 0 | filled |
| Missing hours: wind_offshore_fc (raw / after filling 1-2 h gaps) | 0 / 0 | pass |
| Missing hours: pv_fc (raw / after filling 1-2 h gaps) | 1 / 0 | filled |
| Missing hours: load_fc (raw / after filling 1-2 h gaps) | 0 / 0 | pass |
| Missing hours: load_actual (raw / after filling 1-2 h gaps) | 0 / 0 | pass |
| Missing hours: residual_load_actual (raw / after filling 1-2 h gaps) | 0 / 0 | pass |
| Days with a gap > 2 h after filling (kept; NaN for XGBoost, median for the LSTM) | 0 | pass |
| Hours without exactly 4 quarter-hour prices | 0 | pass |
| Prices outside [-500, 4000] EUR/MWh | 0 | pass |
| SMARD hourly price missing (hours) | 0 | pass |
| SMARD hourly price differs from quarter-hour mean by > 0.01 EUR/MWh (of 8760 hours) | 0 | pass |
