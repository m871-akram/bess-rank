# Model validation report

Written in session S4 (PLAN.md §8), in the style of a bank's model-risk review. The subject is a
research pipeline, not a trading system: a simulated 1 MW / 2 MWh battery trades the German
day-ahead market (DE-LU) only, as a price-taker.

Everything after the pre-registered test run is **exploratory**. The verdicts on H1–H3 come from
the single pre-registered run (`results/test_summary.md`) and are quoted here as they are. The
exploratory numbers are in `results/explore/` (`summary.md` holds every table, the CSV files the
raw numbers).

## Summary

- **H1 supported: +1.83 €/day (95% CI 0.78 to 2.84), +667 €/MW/year (+0.97%).** S-xgb-rank
  closes 16.4% of S-xgb-reg's gap to perfect foresight (667 of 4,074 €/MW/year).
- **The ranked vector's RMSE was lower, not equal (H2 inconclusive):** −0.25 €/MWh, 95% CI
  −0.37 to −0.14, against a ±0.30 €/MWh equivalence margin.
- **No difference for the LSTM (H3 inconclusive):** −0.08 €/day, 95% CI −1.19 to 1.03.
<!-- SUMMARY-EXPLORE -->

## 1. Purpose and scope

- **Question.** Does a model trained to rank the hours of each delivery day earn more simulated
  battery profit than the same model family trained to predict prices, when both feed the same
  daily battery program? The design is a 2×2: XGBoost or bidirectional LSTM, price or rank
  objective (PLAN.md §1, §4).
- **Controlled comparison.** A "-rank" strategy keeps the price model's forecast values for the
  day and reassigns them to hours in the ranker's order (PLAN.md §5). Within a family only the
  within-day order differs; forecast values, battery and settlement are shared.
- **Intended use.** Research evidence on training objectives for storage arbitrage. Out of
  scope: intraday and balancing markets, bidding strategy and market impact, fees and grid
  charges, battery ageing beyond a linear cost per MWh discharged.
- **Model outputs reviewed.** Hourly price forecasts (XGB-reg, LSTM-reg), within-day scores
  (XGB-rank, LSTM-rank), the price vectors of seven strategies, the daily schedules of a
  mixed-integer program, and daily profit settled at the actual prices.

## 2. Data and quality

- **Source.** SMARD chart API (Bundesnetzagentur | SMARD.de): day-ahead price (4169), TSO
  day-ahead forecasts of load (411), onshore wind (123), offshore wind (3791) and PV (125), actual
  load (410) and actual residual load (4359), hourly, DE-LU. Download time and versions are in
  `results/provenance.json`.
- **Periods.** Train 2019-01-01 → 2024-09-30 (2,092 days after dropping 8 gap days), validation
  2024-10-01 → 2025-09-30, test 2025-10-01 → 2026-09-30 (365 days, locked until the
  pre-registration was merged; PREREGISTRATION.lock = 517478e).
- **Checks** (`results/qa_data.md`). 0 duplicate or off-grid timestamps. Gaps of 1–2 hours are
  interpolated and flagged (the 00:00 hour SMARD misses on 25-hour days); longer gaps drop the
  training day only. The test period passes every format check: 365 days and 8,760 hours, DST
  days with 23 and 25 hours, 2 hours filled (wind onshore and PV on 2025-10-26), 0 days with a
  missing value after filling, every hour with 4 quarter-hour prices, and SMARD's hourly price
  equal to the quarter-hour mean within 0.01 €/MWh on all 8,760 hours.
- **The 15-minute market** (from 2025-10-01). The study stays hourly: the hourly price is the
  mean of the four quarter-hour prices, which is exact for a battery holding constant power within
  each hour. A quarter-hour battery could earn more; that extension was cut (PLAN.md §8).
- **Identity checks.** Wind + PV forecast = SMARD's 5097 and forecast residual load = 4362, both
  with a maximum difference of 0.0 MWh on a training week. 411 behaves like a day-ahead load
  forecast (MAPE 3.86% against actual load); the label is not exposed by the API.

## 3. Conceptual soundness and assumptions

- **Decision time and look-ahead.** The decision for day D is taken at 11:00 Europe/Berlin on
  D-1. Every feature declares its availability rule in `features.py`; `tests/test_features.py`
  deletes everything after the cut-off, recomputes the features for 37 real and 6 synthetic days,
  and asserts identical values.
- **TSO generation forecasts (caveat).** EU Regulation 543/2013 only requires wind and solar
  forecasts for D by 18:00 on D-1, after the 12:00 auction. They are used as a proxy for the
  morning vendor forecasts a trader would buy (as in the epftoolbox literature). Section 7 tests
  the pipeline without them.
- **Battery program.** One MILP per day (23, 24 or 25 hours): 88% round trip, 50% charge at the
  start and end of each day, at most one full cycle per day, 10 €/MWh degradation, and a binary
  per hour that forbids charging and discharging at once (with negative prices an LP would burn
  energy). HiGHS (scipy) and SCIP (OR-Tools) solve every day to a relative gap of 0 and must agree
  within 1e-6, and perfect foresight must be ≥ every strategy on every day; both checks held on
  every solve in S1–S4.
- **Settlement.** The schedule is computed on the strategy's price vector and settled at the
  actual prices. Price-taker: a 1 MW battery does not move a market of about 50–60 GW.
- **Models.** XGBoost 3.2.0 (`reg:squarederror`; `rank:pairwise` over every within-day pair of
  hours, checked against a hand-written gradient) and a 2-direction LSTM over each day's hours
  (masked MSE or the same pairwise logistic loss). Hyperparameters were chosen on the validation
  year by RMSE or within-day Spearman ρ, never by profit (PLAN.md §7).
- **Uncertainty.** Every CI is a moving-block bootstrap over days (7-day blocks, 10,000
  resamples, seed 20261004), never over hours, so serial dependence between days is kept.
<!-- NUMERIC-SECTIONS -->
