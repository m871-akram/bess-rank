# Model validation report

Banks have a model-risk team whose job is to try to break a model before anyone relies on it:
check the data, question the assumptions, look at where it fails, and say what to monitor. I
wrote this report in that spirit, for my own pipeline. The subject is a research study, not a
trading system: a simulated 1 MW / 2 MWh battery trading the German day-ahead market (DE-LU)
only, as a price-taker.

One rule runs through the whole report. The H1–H3 verdicts come from the single pre-registered
test run (`results/test_summary.md`) and are quoted as they are. Everything I did after that run
is **exploratory**: it helps explain the result, but it does not test anything and cannot change
a verdict. The exploratory numbers are in `results/explore/` (`summary.md` has every table, the
CSV files the raw numbers).

## Summary

- **H1 supported: +1.83 €/day (95% CI 0.78 to 2.84), +667 €/MW/year (+0.97%).** S-xgb-rank
  closes 16.4% of S-xgb-reg's gap to perfect foresight (667 of 4,074 €/MW/year).
- **The ranked vector's RMSE was lower, not equal (H2 inconclusive):** −0.25 €/MWh, 95% CI
  −0.37 to −0.14, against a ±0.30 €/MWh equivalence margin.
- **No detectable difference for the BiLSTM (H3 inconclusive):** −0.08 €/day, 95% CI −1.19 to
  1.03.
- **Exploratory checks** (sections 5–7): the XGBoost gain keeps a CI above 0 for every battery I
  tried (+1.16 to +2.73 €/day), without the wind and solar forecasts (+4.38 €/day) and with fixed
  tree counts (+1.30 €/day). Order explains two-thirds of the gap to perfect foresight, and the
  BiLSTM already orders the hours almost as well as the ranker. Risk-averse (CVaR) schedules did
  not pay off.

## 1. What is being validated

- **The question.** Does a model trained to rank the hours of each day earn more for the battery
  than the same model trained to predict prices, when both feed the same daily battery program?
  I test it in a 2×2 design: XGBoost or BiLSTM (bidirectional LSTM), price or rank objective
  (PLAN.md §1, §4).
- **Why the comparison is fair.** A ranker outputs an order, not prices. So the "-rank" strategy
  takes the price model's forecast values for the day and reassigns them to the hours in the
  ranker's order (PLAN.md §5). Within a family, the forecast values, the battery and the
  settlement are identical; only the within-day order differs.
- **What it is for.** Evidence on which training objective suits storage arbitrage. Not covered:
  intraday and balancing markets, bidding and market impact, fees, detailed battery ageing.

## 2. Data and data quality

- **Source.** SMARD chart API (Bundesnetzagentur | SMARD.de), hourly, DE-LU: the day-ahead
  price (4169), the grid operators' day-ahead forecasts of load (411), onshore wind (123),
  offshore wind (3791) and solar (125), actual load (410) and actual residual load (4359).
- **Periods.** Training 2019-01-01 → 2024-09-30 (2,092 days after dropping 8 days with long
  gaps), validation 2024-10-01 → 2025-09-30, test 2025-10-01 → 2026-09-30 (365 days, locked
  until the pre-registration was merged; `PREREGISTRATION.lock` = 517478e).
- **Checks** (`results/qa_data.md`). No duplicate or off-grid timestamps. Gaps of 1–2 hours
  (for example the 00:00 hour SMARD misses on 25-hour days) are interpolated and flagged; longer
  gaps drop that training day only. The test year passes every format check: 365 days and 8,760
  hours, daylight-saving days with 23 and 25 hours, only 2 hours filled, and SMARD's hourly price
  equal to the mean of its four quarter-hour prices within 0.01 €/MWh on every hour.
- **The 15-minute market.** From 2025-10-01 the auction clears in quarter-hours. I use the
  hourly mean, which is exact for a battery holding a constant power within each hour.

## 3. Is the design sound?

- **No look-ahead.** The decision for day D is taken at 11:00 Europe/Berlin on D-1. Every
  feature declares when its data become available (`features.py`), and
  `tests/test_features.py` rebuilds the features of 37 real and 6 synthetic days from only the
  data visible at that time and checks that the values are identical.
- **A caveat on wind and solar forecasts.** EU Regulation 543/2013 only requires the grid
  operators to publish them by 18:00 on D-1, after the 12:00 auction. I use them as a stand-in
  for the morning vendor forecasts a trader would buy (as the price-forecasting literature does),
  and section 5.2 measures what happens without them.
- **The battery program.** One small mixed-integer program per day (23, 24 or 25 hours): 88%
  round trip, 50% charge at the start and end of each day, at most one full cycle per day,
  10 €/MWh degradation, and one binary per hour that forbids charging and discharging at once
  (with negative prices a plain LP would do both to burn energy). Two independent solvers,
  HiGHS (scipy) and SCIP (OR-Tools), solve every day and must agree within 1e-6, and perfect
  foresight must beat or equal every strategy on every day. Both checks held on every solve.
- **Settlement.** The plan is made on the strategy's price vector and paid at the actual prices.
  A 1 MW battery does not move a market of about 50–60 GW, so it is treated as a price-taker.
- **Models.** XGBoost 3.2.0 (`reg:squarederror`, or `rank:pairwise` over every pair of hours of
  a day, checked against a hand-written gradient) and a BiLSTM over each day's hours (masked MSE
  or the same pairwise loss). I chose hyperparameters on the validation year by RMSE or
  within-day Spearman ρ, never by profit (PLAN.md §7).
- **Error bars.** Every CI is a moving-block bootstrap over days (7-day blocks, 10,000 resamples,
  seed 20261004). Resampling whole weeks keeps the correlation between neighbouring days, which
  resampling single hours or days would ignore.

## 4. Main results (test year, pre-registered)

**The verdicts.** H1 is the only confirmatory test; H2 and H3 are secondary, without
multiplicity adjustment.

| Hypothesis | Statistic | Estimate | 95% CI | Verdict |
|---|---|---|---|---|
| H1 (confirmatory) | S-xgb-rank − S-xgb-reg, mean daily profit (€/day) | +1.83 | 0.78 to 2.84 | supported |
| H2 (secondary) | RMSE(S-xgb-rank) − RMSE(S-xgb-reg) (€/MWh), margin ±0.30 | −0.25 | −0.37 to −0.14 | inconclusive |
| H3 (secondary) | S-lstm-rank − S-lstm-reg, mean daily profit (€/day) | −0.08 | −1.19 to 1.03 | inconclusive |
| BiLSTM RMSE (descriptive) | RMSE(S-lstm-rank) − RMSE(S-lstm-reg) (€/MWh), margin ±0.29 | −0.12 | −0.18 to −0.04 | no test |

In words: ranking the hours made the XGBoost battery +667 €/MW/year (+0.97%) richer and closed
16.4% of its gap to perfect foresight; the ranked strategy won on 178 days and lost on 116. Its
RMSE was lower, not equal: the H2 interval lies below 0 and crosses −0.30, so it cannot rule out
a drop of more than 1% (H2 inconclusive). For the BiLSTM there is no detectable difference (H3
inconclusive). The headline I had planned in PLAN.md §1 needed H1 and H2 both supported, so I do
not use it.

**Every strategy** (VaR 5% = 5th percentile of daily profit, ES 5% = mean of the worst 5% of
days, 19 of 365; both are profit levels, so a negative value is a loss):

| Strategy | Profit (€/MW/year) | Capture | RMSE (€/MWh) | Spearman ρ | VaR 5% (€) | ES 5% (€) | Losing days | Max drawdown (€) |
|---|---|---|---|---|---|---|---|---|
| S-perfect | 73,181 | 100.0% | 0 | 1 | 21.73 | 14.73 | 0.0% | 0 |
| S-naive-1d | 60,390 | 82.5% | 47.34 | 0.781 | −7.57 | −23.79 | 7.9% | 61.4 |
| S-naive-7d | 57,936 | 79.2% | 57.46 | 0.794 | −2.58 | −25.55 | 6.6% | 222.3 |
| S-xgb-reg | 69,107 | 94.4% | 30.26 | 0.936 | 14.82 | 5.68 | 1.1% | 16.0 |
| S-xgb-rank | 69,774 | 95.3% | 30.01 | 0.952 | 15.17 | 6.26 | 0.8% | 13.1 |
| S-lstm-reg | 69,864 | 95.5% | 28.50 | 0.945 | 14.78 | 6.90 | 0.5% | 11.3 |
| S-lstm-rank | 69,834 | 95.4% | 28.38 | 0.954 | 15.27 | 7.70 | 0.8% | 11.3 |

- **By quarter.** S-xgb-rank − S-xgb-reg is positive in all four quarters (+2.60, +0.76, +1.93,
  +2.01 €/day). S-lstm-rank − S-lstm-reg is +0.39, +0.22 and +0.53 €/day in the first three and
  −1.45 €/day in Jul–Sep 2026 (section 7.2 explains why). By month, S-xgb-rank's capture is above
  S-xgb-reg's in 11 of 12 months.
- **Against the naive benchmarks.** Every model strategy earns 8,700–9,500 €/MW/year more than
  S-naive-1d, with a 36–40% lower RMSE and 0.5–1.1% losing days against 7.9%.
- **The BiLSTM as challenger** (exploratory). Alone, the BiLSTM price model earns about as much
  as the XGBoost ranker: S-lstm-reg − S-xgb-reg = +2.07 €/day (95% CI 0.71 to 3.45), S-lstm-reg −
  S-xgb-rank = +0.25 €/day (−1.14 to 1.61).
- **Exploratory forecasts.** The test-run forecasts were not kept, so I regenerated them with the
  frozen code: 48 of 48 fits match the official run and the daily profits agree within
  2.3e-13 € (`results/explore/regenerated_forecasts_check.json`).

## 5. Does it survive changes?

### 5.1 Other batteries

I re-planned the same test forecasts with other battery settings, without retraining
(`results/explore/sensitivity_*.csv`). Other durations keep 1 MW, start and end each day at 50%
charge and allow one full cycle per day.

| Setting | S-perfect (€/MW/year) | S-xgb-reg | S-xgb-rank | XGB rank − reg (€/day, 95% CI) | Share of gap closed | BiLSTM rank − reg (€/day, 95% CI) |
|---|---|---|---|---|---|---|
| Base: 2 h, 10 €/MWh | 73,181 | 69,107 | 69,774 | +1.83 (0.78 to 2.84) | 16.4% | −0.08 (−1.19 to 1.03) |
| Degradation 0 €/MWh | 80,319 | 76,276 | 77,021 | +2.04 (0.99 to 3.06) | 18.4% | −0.12 (−1.22 to 0.98) |
| Degradation 20 €/MWh | 66,351 | 62,020 | 62,736 | +1.96 (0.93 to 2.99) | 16.5% | −0.10 (−1.21 to 1.01) |
| 1 h (1 MW / 1 MWh) | 37,919 | 35,416 | 35,838 | +1.16 (0.55 to 1.73) | 16.8% | −0.05 (−1.30 to 0.99) |
| 4 h (1 MW / 4 MWh) | 128,419 | 123,716 | 124,712 | +2.73 (1.05 to 4.64) | 21.2% | +0.66 (−0.40 to 1.59) |

The XGBoost result does not depend on these settings: the gain keeps a CI above 0 everywhere and
closes 16–21% of the gap to perfect foresight. The BiLSTM shows no detectable difference in any
of them.

### 5.2 Without the wind and solar forecasts

Because those forecasts may arrive after the auction (section 3), I removed the 15 features built
from them (27 of 42 kept, the load forecast included) and refitted the XGBoost pair with the same
procedure: quarterly refits, 3 seeds, early stopping at each refit.

| Variant | S-xgb-reg (€/MW/year) | S-xgb-rank | RMSE reg / rank (€/MWh) | Spearman ρ reg / rank | rank − reg (€/day, 95% CI) | Share of gap closed |
|---|---|---|---|---|---|---|
| Official (42 features) | 69,107 | 69,774 | 30.26 / 30.01 | 0.936 / 0.952 | +1.83 (0.78 to 2.84) | 16.4% |
| No wind/solar forecasts | 64,659 | 66,256 | 38.36 / 38.01 | 0.858 / 0.887 | +4.38 (2.20 to 6.51) | 18.7% |

Both strategies lose money (6.4% and 5.0% of their profit), which shows how much the results lean
on these forecasts. But the gain from ranking gets bigger (+4.38 €/day) and stays positive in
every quarter (+4.74, +1.88, +5.65, +5.18 €/day): with weaker inputs, the ranking objective
recovers more of the order.

## 6. Is it stable over time?

**Feature drift (PSI)** for the 10 most important features, test year against the training
period (bins = training deciles; below 0.1 is usually read as stable, 0.1–0.25 as moderate drift,
above 0.25 as large drift):

| Rank | Feature | From wind/solar forecasts | PSI test vs train | PSI validation vs train |
|---|---|---|---|---|
| 1 | price_lag1 | no | 1.05 | 0.97 |
| 2 | residual_load_fc_dev | yes | 0.17 | 0.06 |
| 3 | residual_load_fc_rank | yes | 0.00 | 0.00 |
| 4 | price_lag1_day_mean | no | 1.55 | 0.93 |
| 5 | price_lag7 | no | 1.06 | 0.95 |
| 6 | price_profile28 | no | 0.07 | 0.04 |
| 7 | residual_load_fc | yes | 0.15 | 0.07 |
| 8 | price_lag1_day_max | no | 1.72 | 1.83 |
| 9 | wind_pv_share | yes | 0.15 | 0.06 |
| 10 | price_lag7_rank | no | 0.00 | 0.00 |

- **Price levels moved a lot** (PSI 1.05–1.72): the training years run from the cheap prices of
  2019–2020 through the 2022 energy crisis. The shift was already there in validation
  (0.93–1.83), so model selection saw it.
- **Residual-load features drifted moderately** (0.15–0.17, against 0.06–0.07 in validation),
  consistent with more wind and solar. This includes the ranker's top feature, the within-day
  residual-load deviation (0.17).
- **The within-day rank and profile features stayed stable** (0.00–0.07).
- **The ranker's inputs drifted much less than the price model's** (section 7.5): XGB-rank relies
  on the within-day deviation and rank (PSI 0.17 and 0.00), XGB-reg on levels such as the prices
  on D-1 and D-7 (PSI 1.05 and 1.06).

**Performance by quarter** (`results/explore/metrics_by_quarter.csv`). Capture rises from
88–90% in Oct–Dec 2025 to 97–99% in Jul–Sep 2026, while RMSE rises from 22–23 to 35–38 €/MWh as
the daily price spreads widen: RMSE and profit move in opposite directions. Oct–Dec 2025 is the
only quarter with a negative ES 5% (−3.4 to −1.0 €) and 2–3% losing days.

**What I would monitor.** Monthly capture against the naive benchmark, the PSI of the price-level
and residual-load features, and XGB-reg's early-stopping tree count at each refit (section 7.3).

## 7. Deeper dives

### 7.1 Is accuracy the same as value?

![accuracy vs value](results/explore/fig_accuracy_vs_value.png)

Over the 58 single-seed fits of the validation-year search, the rank correlation with profit is
−0.59 for RMSE and +0.87 for within-day Spearman ρ. Part of that is simply the rankers (all of
them above every price model) standing apart from the price models. But it also holds within the
21 XGB-reg fits: ρ +0.77 (p < 0.001) against RMSE −0.36 (p = 0.11). Within the rankers and within
the 8-fit BiLSTM groups neither metric sorts the fits reliably
(`results/explore/accuracy_vs_value.csv`). The lesson: picking a price model by RMSE is a weak
guide to how much money it makes for a battery.

### 7.2 Where is the money lost?

I split each strategy's gap to perfect foresight (in €/day) into two parts with two "oracle"
experiments: the **order** part is what is lost when the battery gets the true prices but in the
model's order; the **values** part is what is lost when it gets the model's values but in the
true order. What is left over is the interaction.

| Strategy | Profit (€/day) | Gap | Order only | Values only | Interaction |
|---|---|---|---|---|---|
| S-xgb-reg | 189.33 | 11.16 | 7.61 | 3.88 | −0.32 |
| S-xgb-rank | 191.16 | 9.33 | 5.73 | 3.88 | −0.28 |
| S-lstm-reg | 191.41 | 9.09 | 6.19 | 3.57 | −0.67 |
| S-lstm-rank | 191.33 | 9.17 | 5.85 | 3.57 | −0.25 |

- **Most of the loss is order.** Two-thirds of S-xgb-reg's gap is order (7.61 of 11.16 €/day).
  The XGBoost ranker's order removes 1.88 €/day of it, almost exactly H1's +1.83 €/day. The
  values part dominates only in Apr–Jun 2026 (8.8–9.6 of 12.5–15.0 €/day)
  (`results/explore/oracle_decomposition.csv`).
- **The BiLSTM already orders well.** Within-day Spearman ρ, paired by day: XGB-rank − XGB-reg
  +0.016 (0.013 to 0.019), LSTM-reg − XGB-reg +0.009 (0.005 to 0.012), XGB-rank − LSTM-reg +0.007
  (0.004 to 0.010). Hit rates for the 2 cheapest / 2 dearest hours: XGB-reg 69.3% / 71.5%,
  XGB-rank 73.7% / 74.9%, LSTM-reg 72.5% / 73.0%. In profit, LSTM-reg's order loses 0.46 €/day
  more than XGB-rank's, but its values lose 0.31 less and its interaction term is 0.39 more
  favourable, so the two end level (191.41 against 191.16 €/day). The BiLSTM ranker improves ρ
  again (+0.009, 0.006 to 0.012), but that is worth only 0.34 €/day of order and the interaction
  takes it back: hence H3 inconclusive.
- **The BiLSTM's bad summer (Jul–Sep 2026, −1.45 €/day).** With true prices the ranker's order
  is actually *better* that quarter (order-only gap 3.17 against 3.61 €/day); the loss is mostly
  one day. On 2026-09-14 (perfect foresight 741 €), S-lstm-rank earned 626 € against 734 €:
  −108 € of the quarter's −134 €. The ranker swapped two neighbours, ranking 07:00 (317 € in
  reality) just above 18:00 (400 €), so 07:00 received LSTM-reg's third-highest value, 319 €.
  With one cycle a day, the battery then sold 0.9 MWh at 07:00 instead of at 20:00 (526 €). In
  true prices that swap costs only 3.8 €; with forecast values it cost −108 €. Without that day
  the quarter's difference is −0.28 €/day. The weak spot: the "-rank" construction is fragile
  when a near-tie in the ranker's order meets a big gap between the price model's values.

### 7.3 Does the result depend on early stopping?

XGB-reg's early stopping picked very different tree counts from one refit to the next (39 to
858). So I refitted the XGBoost pair on the test year with the validation tree counts fixed at
every refit (XGB-reg 467/223/571, XGB-rank 83/88/151 trees for seeds 0/1/2), without early
stopping.

| | S-xgb-reg (€/MW/year) | S-xgb-rank | RMSE reg (€/MWh) | rank − reg (€/day, 95% CI) | Share of gap closed |
|---|---|---|---|---|---|
| Official (early stopping at each refit) | 69,107 | 69,774 | 30.26 | +1.83 (0.78 to 2.84) | 16.4% |
| Validation tree counts, fixed | 69,358 | 69,830 | 29.20 | +1.30 (0.36 to 2.36) | 12.4% |

| Quarter from | Official rank − reg (€/day) | Trees reg / rank (official) | Fixed trees rank − reg (€/day) |
|---|---|---|---|
| 2025-10-01 | +2.60 | 616/858/810 · 69/70/188 | +2.32 |
| 2026-01-01 | +0.76 | 39/48/43 · 231/124/174 | +0.10 |
| 2026-04-01 | +1.93 | 67/56/64 · 171/176/146 | +1.52 |
| 2026-07-01 | +2.01 | 718/778/427 · 182/251/92 | +1.21 |

With fixed trees XGB-reg gets better (+0.69 €/day) while XGB-rank barely moves (+0.15), so the
gain shrinks by 0.53 €/day, but its CI stays above 0 and it stays positive in every quarter. So
about 0.5 €/day of the official effect is early-stopping noise at the refits, mostly in the price
model, and most of the effect is not. The official verdict stands; this is a check, not a
re-test.

### 7.4 Risk-aware schedules: conformal quantiles and CVaR

Could the battery avoid its worst days by planning for uncertainty? I fitted a quantile version
of XGBoost (19 levels, seed 0, 201 trees) on the training period and calibrated it on the
validation year with conformalized quantile regression (CQR), which widens or narrows the
intervals until they cover the right share of past hours. On the test year the 90% interval
covers 85.0% of hours before calibration and 88.6% after (50% interval: 41.6% and 45.1%; mean
pinball loss 7.80 → 7.77 €/MWh): better, but still a little short, because prices kept drifting
(section 6).

From these quantiles I drew 200 price scenarios per day (a Gaussian copula with the
training-period correlation between hours) and planned the battery to maximise the mean scenario
profit minus λ × CVaR 95% of the loss (Rockafellar–Uryasev). SCIP and HiGHS agree on all 1,825
day-λ solves. The plans are paid at the actual prices:

| Schedule | Profit (€/MW/year) | Capture | VaR 5% (€) | ES 5% (€) | Losing days | Max drawdown (€) |
|---|---|---|---|---|---|---|
| λ = 0 | 68,821 | 94.0% | 13.95 | 5.94 | 0.5% | 22.0 |
| λ = 0.5 | 65,524 | 89.5% | 10.85 | 1.96 | 0.8% | 17.8 |
| λ = 1 | 63,180 | 86.3% | 8.68 | 1.58 | 0.5% | 13.6 |
| λ = 2 | 59,414 | 81.2% | 0.15 | −0.58 | 0.5% | 8.5 |
| λ = 5 | 55,989 | 76.5% | 0.00 | −0.26 | 0.3% | 5.0 |
| S-xgb-reg (reference) | 69,107 | 94.4% | 14.82 | 5.68 | 1.1% | 16.0 |

![CVaR frontier](results/explore/fig_cvar_frontier.png)

A negative result: being risk-averse bought no tail protection on this test year. Mean profit
falls by 5% (λ = 0.5) to 19% (λ = 5) and the realized ES falls with it (5.9 → −0.3 €); only the
maximum drawdown improves (22 → 5 €). Losses are rare (0.3–0.8% of days): the worst days are days
with small price spreads (perfect foresight earns only 14.7 € on them), not big losses, and a
risk-averse plan gives up spread on every day.

### 7.5 What do the models look at? (SHAP)

Mean |SHAP| on the test year for the training-period models (`results/explore/shap_top10.csv`):
XGB-rank draws 62% of its attribution from the within-day residual-load deviation and rank, and
another 17% from the lagged rank and 28-day profile features. XGB-reg's top three are level
features: the residual-load forecast (18%) and the prices on D-1 (13%) and D-7 (11%). This fits
the story: the ranker leans on within-day features, which drifted much less than the price
model's level features (section 6).

## 8. Findings and recommendations

| # | Severity | Finding | Recommendation |
|---|---|---|---|
| 1 | Medium | XGB-reg's early stopping is unstable across refits (39 to 858 trees); with fixed tree counts H1's statistic falls from +1.83 to +1.30 €/day (CI still above 0). | Report the check next to H1; in future, fix tree counts or average several early-stopping windows. |
| 2 | Medium | The results rely on wind and solar forecasts published after the auction; without them profits fall by 5–6%. | Say so next to every profit number; replace them with forecasts available at 11:00 on D-1 before any operational use. |
| 3 | Low | The "-rank" reassignment is fragile when a near-tie in the ranker's order meets a large gap in the price model's values (2026-09-14: −108 €). | Report per-day contributions; a hybrid rule is a future study. |
| 4 | Low | The original test-run forecasts were not kept and had to be regenerated (reproduced exactly). | Done: every `test` and `explore` run now uploads its forecasts to the Databricks volume. |
| 5 | Low | Hourly study of a market that clears in quarter-hours since 2025-10-01. | Stated limitation; future work. |
| 6 | Info | RMSE is a weak proxy for value (section 7.1). | Report an order metric next to RMSE when choosing a price model for a battery. |

**Conclusion.** For what it is meant to do (a controlled comparison of training objectives for
a simulated battery), the pipeline holds up. The data checks pass, there is no look-ahead (it is
tested), the two solvers agree on every day, the verdicts were pre-registered and reported as
they came out, and the main result survives other battery settings, the removal of the wind and
solar forecasts, and fixed tree counts. Findings 1 and 2 change the size of the effect, not its
sign.

## 9. Limitations

- **It is a simulation.** A 1 MW / 2 MWh price-taker on the day-ahead market only, with no fees,
  no intraday or balancing revenue, no bidding risk, linear degradation and a daily 50% charge
  boundary.
- **Hourly study of a 15-minute market**, and **wind and solar forecasts** that are published
  after the auction (section 5.2 measures what they add).
- **No fuel or carbon prices**; lagged prices carry the price level (section 6).
- **One test year, one market.** H1's CI half-width is about 1 €/day, so smaller effects cannot be
  seen, and the result may not carry over to other bidding zones or years.
- **Exploratory means exploratory.** Sections 5–7 were not pre-registered; they describe, they do
  not test, and none of them changes a verdict.
- **Single fits.** The quantile model and the SHAP and importance models are one fit with seed 0,
  and the CVaR scenarios use one copula estimated on training residuals.

## 10. Reproducibility and governance

- **Pre-registration.** PLAN.md §6 was merged in 517478e, and `PREREGISTRATION.lock` holds that
  hash. The test pipeline refuses to start while locked, if any frozen file differs from the lock
  commit, or if its results already exist. It ran once, with no crash, no failed sanity check and
  no setting changed, and no amendment was needed.
- **Code after the lock.** Later work mostly added code (the quantile model, the CVaR program,
  PSI and CQR in `models.py`, `battery.py` and `risk.py`, plus `explore.py` and `pipeline.py`).
  The test command only gained the automatic upload at its end, and the exact code that produced
  the results is kept at tag `confirmatory-run`.
- **Versions.** Python 3.11.15, pandas 3.0.6, numpy 2.4.6, scipy 1.17.1 (HiGHS), OR-Tools 9.15
  (SCIP), XGBoost 3.2.0, torch 2.14.1+cpu; seeds 0/1/2, bootstrap seed 20261004. The SMARD
  download time and the git commit of every stage are in `results/provenance.json`.
- **Data.** SMARD data are never committed; `python -m bessrank.run data` rebuilds them from the
  Databricks volume or from SMARD. Every `test` and `explore` run uploads its forecasts and
  `provenance.json` to `runs/<run type>-<commit>/` on the volume and checks their size there.
- **Tests.** `pytest` covers the lock, daylight-saving days, feature availability (look-ahead),
  both solvers, the "-rank" reassignment, the XGBoost ranking gradient, BiLSTM padding, the
  verdict rules and the exploratory helpers (PSI, CQR, quarter lookup, oracle vectors).
- **Databricks.** The job `bess-rank-pipeline` ran the official XGBoost test pipeline once on
  serverless (commit 4cd8d8a, run 967734030312010). It reproduced all 24 fits' forecasts exactly
  and the daily profits within 2.3e-13 € on all 365 days (H1: +1.83 €/day, 95% CI 0.78 to 2.84),
  with both solvers agreeing on every day. It wrote four Delta tables
  (`workspace.bess.gold_*`) and logged every fit to the MLflow experiment `bess-rank`. Details
  are in the README and `results/databricks_pipeline.json`.
