# PLAN: Rank, don't forecast. XGBoost learning-to-rank for battery arbitrage

*A battery doesn't need the price, it needs the order.*

Status: plan v1, 2026-10-04. Sections §1–§5 and §7–§11 may change until the pre-registration (§6) is committed. After that, changes go only into dated amendments in §12.

---

## 1. Question and claim

**Question.** A battery trading the German day-ahead market must decide which hours of tomorrow to charge in and which to discharge in. Two options:
- train XGBoost to predict each hour's price, or
- train it to rank the hours of the day.

Which one makes the battery more money?

**Why now**
- Germany recorded 457 hours of negative day-ahead prices in 2024 and 573 in 2025 (SMARD), as solar floods midday.
- Battery fleets are growing fast. In July 2026, Masdar and EWEC reached financial close on a 5.2 GW solar + 19 GWh battery project in Abu Dhabi.
- Since delivery day 2025-10-01, the European day-ahead auction clears in 15-minute slots.

**Gap**
- Price forecasts are trained and judged on RMSE.
- Maciejowska, Lipiecki & Uniejewski (arXiv:2511.13616, 2025) evaluate 192 price forecasts by battery-arbitrage profit. They argue that RMSE and MAE often fail to reflect that economic value.
- Decision-focused learning trains forecasts on the decision's outcome, but usually needs differentiable optimisation and neural networks:
  - Elmachtoub & Grigas, "Smart Predict, then Optimize", *Management Science* 2022;
  - for storage, arXiv:2305.00362.

**Idea.** For one day, a battery's schedule depends mostly on the order of the hours and on the size of the spread. XGBoost has a learning-to-rank objective, so we train a ranker with one query group per delivery day.

The comparison is controlled:
- the ranked strategy keeps the price regressor's forecast values for the day;
- it only reassigns those values to hours in the ranker's order.

Any profit difference then comes from the order alone.

**Possible headline** (only if supported by §6): "Same forecast values, different order: the ranked strategy earns X% more on the test year, despite a higher RMSE."

---

## 2. Data

**Source.** The SMARD chart API (Bundesnetzagentur). A community OpenAPI spec is at github.com/bundesAPI/smard-api.
- Index of chunk timestamps: `https://www.smard.de/app/chart_data/{filter}/{region}/index_{resolution}.json`
- Series chunk: `https://www.smard.de/app/chart_data/{filter}/{region}/{filter}_{region}_{resolution}_{timestamp}.json`
- Resolutions: `hour`, `quarterhour`, `day`, … Timestamps are epoch milliseconds; check what they mark (interval start) on DST days.

**Format** (probed on 2024-06-10, a training-period day)
- Chunks are weekly. Each starts at Monday 00:00 Europe/Berlin, as epoch milliseconds.
- `series` is a list of `[timestamp_ms, value]`, with `null` for missing values.
- Every series below starts on 2018-10-01 for region DE-LU.

**Series.** IDs marked "probed" were checked on that day. The others must be verified in S0 on a sample week, by comparing with the SMARD website labels and running sanity checks: PV ≈ 0 at night, offshore < onshore, parts add up. The community spec has errors: it lists 126 as the PV forecast, but 126 is −(wind + PV).

| Series | Filter | Region | Use | Status |
|---|---|---|---|---|
| Day-ahead price DE-LU | 4169 | DE-LU | label, lagged features, settlement | probed |
| Forecast: wind onshore | 123 | DE-LU | feature | probed |
| Forecast: wind offshore | 3791 | DE-LU | feature | probed |
| Forecast: PV | 125 | DE-LU | feature | probed (0 at night, about 30 GW at noon) |
| Forecast: wind + PV | 5097 | DE-LU | feature | probed (= 123 + 3791 + 125 exactly) |
| Forecast: other generation | 715 | DE-LU | feature | probed |
| Forecast: total generation | 122 | DE-LU | feature | probed |
| Forecast: load | 411 | DE-LU | feature | probed. Likely "Prognostizierter Stromverbrauch"; confirm the label |
| Forecast: residual load | 4362 | DE-LU | feature | probed (= 411 − 5097 exactly) |
| Actual load (Netzlast) | 410 | DE-LU | lagged feature | probed |
| Actual residual load | 4359 | DE-LU | lagged feature | probed |
| Neighbour prices (FR 254, NL 256, AT 4170) | – | – | optional lagged features | to check |

**Periods**
- The DE-LU bidding zone starts on 2018-10-01. Older DE-AT-LU data is not used.
- **Train:** 2018-10-01 → 2024-09-30.
- **Validation:** 2024-10-01 → 2025-09-30.
- **Test (locked):** 2025-10-01 → 2026-09-30.

**The 15-minute market**
- From 2025-10-01 the auction yields quarter-hour prices. The study works at hourly resolution throughout.
- For the test year, the hourly price is the mean of the four quarter-hour prices. This is exact for a battery that holds constant power within each hour: revenue = power × hourly mean price × 1 h.
- Check whether SMARD's `hour` series equals that mean. If not, compute the mean from `quarterhour` data.

**Data quality report** (`results/qa_data.md`)
- Coverage per series and year; gaps; duplicates; DST days (23 or 25 hours).
- Negative-price hours per year, up to 2025-09-30 only.
- For the test period: counts and pass/fail format checks only.

---

## 3. Decision time and features

**Decision time:** 11:00 Europe/Berlin on D-1, for all hours of delivery day D.

**What is known at that time**
- Day-ahead prices for every hour up to the end of D-1 (published about 12:45 on D-2).
- Actual load and generation up to the end of D-2. This is conservative, so D-1 actuals are not used.
- **TSO day-ahead load forecast for D (411): available.** EU Regulation 543/2013 Art. 6(2)(b) requires it at least 2 hours before gate closure, so by 10:00.
- **TSO day-ahead generation forecasts for D: a caveat.** This covers wind, PV, other and total (123, 3791, 125, 5097, 715, 122) and 4362, which includes them. Art. 14(1)(c)–(d) only requires them by 18:00 Brussels time on D-1, which is after the 12:00 auction.
  - Following the price-forecasting literature (Lago et al. 2021, *Applied Energy*, epftoolbox), we use them as a proxy for the morning forecasts traders buy from weather vendors.
  - This assumption goes in the README's limitations, and §8 tests it with a run without these forecasts.

**Features for each hour h of day D**

Each feature declares its availability rule in `features.py`:
- **Calendar:** hour of day, weekday, month, day of year (sin/cos), national public holiday (`holidays`, DE), bridge day, DST flag, number of hours in the day.
- **TSO forecasts for (D, h):**
  - load (411), wind onshore, offshore, PV, wind + PV, other, total, and the wind+PV share of the load;
  - residual-load forecast (4362 = 411 − 5097);
  - within-day normalised versions, e.g. PV at h ÷ the day's maximum PV, and the within-day rank of the residual-load forecast. These help ordering.
- **Daily aggregates of the forecasts for D:** mean, min and max residual load; total PV energy; total wind energy.
- **Lagged prices:**
  - the price at (D-1, h) and (D-7, h);
  - the mean, min, max and spread of D-1 and D-7;
  - the within-day rank of h on D-1 and D-7;
  - the 28-day mean normalised price profile by hour.
- **Lagged actuals:** load and residual load at (D-2, h) and (D-7, h).
- The run without TSO generation forecasts (§8) drops every generation-forecast feature, including 4362 and the features derived from it, and keeps the load forecast (411).

Not used: fuel and carbon prices (no free, clean source). Lagged prices carry the price level. Stated as a limitation.

---

## 4. Models

Both models use the same features, the same training windows, XGBoost with `tree_method="hist"` on CPU, and 3 seeds per configuration. Each final forecast averages the 3 seeds: mean prediction for the regressor, mean score for the ranker.

- **M-reg (price regressor).**
  - Target: hourly price (€/MWh). Objective `reg:squarederror`.
  - Output: p̂(D, h).
- **M-rank (within-day ranker).**
  - One query group per delivery day (`qid` = day; rows sorted by day, then hour).
  - Label: dense within-day rank of the actual price, with 0 for the cheapest. Tied prices get the same label, which matters on days with many hours at 0 or −0.01 €/MWh.
  - Objective `rank:pairwise`. Check XGBoost's learning-to-rank docs for label rules, and set `lambdarank_pair_method` / `lambdarank_num_pair_per_sample` so that every pair of hours in a day is used (276 pairs for a 24-hour day).
  - Output: scores whose within-day order is the predicted order.

---

## 5. Strategies, battery and settlement

**Battery**
- 1 MW / 2 MWh; round-trip efficiency 88% (η_c = η_d = √0.88).
- State of charge starts and ends every day at 50% (1 MWh).
- At most one equivalent full cycle per day: total discharge ≤ 2 MWh.
- Degradation cost: 10 €/MWh discharged.
- No simultaneous charging and discharging. With negative prices, a plain linear program would charge and discharge at once to burn energy, so a binary per hour prevents it. This makes it a small mixed-integer program, solved with `scipy.optimize.milp` (HiGHS).

**Program for one day with n hours (23, 24 or 25):**
maximise Σ_h p_h (d_h − c_h) − 10 Σ_h d_h, subject to:
- 0 ≤ c_h ≤ u_h and 0 ≤ d_h ≤ 1 − u_h, with u_h binary;
- soc_{h+1} = soc_h + η_c c_h − d_h / η_d;
- 0 ≤ soc ≤ 2, and soc_0 = soc_n = 1;
- Σ_h d_h ≤ 2.

**Settlement**
- The schedule is computed from a strategy's price vector, then settled at the actual hourly prices.
- Price-taker; no fees, grid charges, intraday or balancing revenue; perfect execution.

**Strategies:** each produces a price vector for day D, which goes through the program above.

| Strategy | Price vector |
|---|---|
| S-perfect | actual prices (upper bound) |
| S-naive-1d | prices of D-1 |
| S-naive-7d | prices of D-7 |
| S-reg | p̂ from M-reg, as forecast |
| S-rank | the same values as p̂, reassigned to hours in M-rank's order |

**How S-rank is built.** Sort the day's regressor forecasts in increasing order: v₍₁₎ ≤ … ≤ v₍ₙ₎. Order the hours by ranker score, cheapest first: h₍₁₎ … h₍ₙ₎; break ties by the regressor's forecast, then by time. Set p̃(h₍ₖ₎) = v₍ₖ₎.
- If the two models agree on the order, S-rank and S-reg are identical.
- S-reg is the special case where the order comes from the regressor itself.

**Metrics**
- Profit (€ per MW per year) and daily profit.
- Capture rate = profit / S-perfect profit.
- Price RMSE and MAE of each strategy's price vector against actual prices.
- Within-day Spearman ρ, and the hit rate for the 2 cheapest and the 2 most expensive hours.

---

## 6. Pre-registration (draft; finalise after validation, before unlocking the test)

**H1 (primary).** S-rank earns more than S-reg on the test year.
- Statistic: mean daily profit difference, S-rank − S-reg.
- 95% CI by moving-block bootstrap: 7-day blocks, 10,000 resamples, seed 20261004.
- **Supported** if the CI's lower bound > 0; **contradicted** if the upper bound < 0; **inconclusive** otherwise.

**H2 (secondary; makes the "worse RMSE, more profit" claim testable).** S-rank's price vector has a higher RMSE than S-reg's, with the same bootstrap CI above 0. The headline needs both H1 and H2 supported.

**Reported for every strategy:** annual profit, capture rate, daily-profit distribution, RMSE and MAE, within-day Spearman ρ, hit rates.

**Test protocol**
- Hyperparameters are fixed from validation.
- Both models are refitted on the first day of each test month, on all data up to the end of the previous month.
- 3 seeds, averaged. Run once.

**Lock procedure**
1. Akram commits this section.
2. `PREREGISTRATION.lock` is created with that commit hash.
3. Only then may `BESS_UNLOCK_TEST=1` be used.

Anything decided after the lock is a dated amendment in §12 and is labelled exploratory.

---

## 7. Validation protocol (2024-10-01 → 2025-09-30)

- Fit on train and predict the validation year. Also do one check with monthly refits, to see how much refitting matters and what it costs.
- Use the last 3 months of train as the inner early-stopping set (at most 3,000 trees), then refit on the full training window with the chosen number of trees.
- **Random search:** 30 configurations per model, 3 seeds each. Each model is tuned on its own objective's metric:
  - M-reg on RMSE;
  - M-rank on mean within-day Spearman ρ.

  They are not tuned on profit, so the comparison stays about the training objective. Profit-tuned versions are exploratory (§8).
- **Search space:**
  - max_depth 3–10;
  - learning_rate 0.01–0.2 (log scale);
  - min_child_weight 1–20;
  - subsample 0.6–1, colsample_bytree 0.5–1;
  - reg_lambda 0.1–10 (log scale).
- **Outputs:**
  - validation results for every strategy;
  - a table of all configurations with RMSE, Spearman ρ and validation profit (used later in §8);
  - the proposed final text of §6.
- Validation results are exploratory and may change the design before the lock. Record every change in §12.

---

## 8. Exploratory analyses (after the test run)

1. **Where profit is lost:**
   - true order with the regressor's values;
   - the regressor's order with the true values.
2. **Robustness:**
   - no TSO generation forecasts, keeping the load forecast (§3 caveat);
   - degradation cost 0 and 20 €/MWh;
   - 1 h and 4 h batteries;
   - two cycles per day.
3. **Accuracy vs value:** across all validation configurations, how well RMSE and Spearman ρ each track profit.
4. **Profit-tuned versions** of both models.
5. **When the ranker helps:** by month and season; negative-price days; the 5 days where S-rank and S-reg differ most (plots).
6. **Drivers:** SHAP top features for M-rank and M-reg.
7. **15-minute extension:** schedule the test year in 96 slots per day.

---

## 9. Databricks (Free Edition)

- **Storage:** catalog `workspace`, schema `bess`, volume `raw`. Upload the raw JSON from the Mac with `scripts/upload_to_databricks.sh` (Databricks CLI), or through the UI if the CLI cannot authenticate.
- **Delta tables:**
  - `bronze_smard` (raw series as downloaded);
  - `silver_hourly` (clean hourly table with QA flags);
  - `gold_features`, `gold_predictions`, `gold_schedules`, `gold_daily_profit`.
- **MLflow:** experiment `bess-rank`, one run per model fit (parameters, metrics, model). Register the final models in Unity Catalog if Free Edition allows it.
- **Job:** one multi-task job that reproduces the official test run (ingest → features → train and predict → dispatch → evaluate). Its results must match the Mac run to a relative difference of 1e-6, or the difference is explained.
- **Dashboard:** one AI/BI dashboard with cumulative profit by strategy, capture rate by month, and example days (prices vs schedule).
- **Optional live mode:** a local script downloads yesterday's SMARD data, uploads it and triggers the job, which writes tomorrow's schedule.

---

## 10. Repo layout

```
bess-rank/
  CLAUDE.md  PLAN.md  README.md  requirements.txt  .gitignore  .claude/settings.json
  src/bessrank/
    config.py      dates, battery parameters, paths, test lock
    data.py        SMARD download, parsing, hourly table, QA report
    features.py    features with availability rules
    models.py      XGBoost regressor and ranker, tuning, seed averaging
    battery.py     daily MILP dispatch and settlement
    strategies.py  price vectors per strategy (including the S-rank reassignment)
    evaluate.py    profit, capture, forecast metrics, block bootstrap
    run.py         python -m bessrank.run {download,qa,features,validate,test,explore}
  scripts/         download_smard.py, upload_to_databricks.sh
  notebooks/       Databricks wrappers (01_ingest … 05_report); local analysis notebooks
  tests/           test_lock, test_time, test_features, test_battery, test_strategies
  results/         small CSV / JSON / PNG outputs (committed)
  data/            raw and processed data (never committed)
```

---

## 11. Sessions, budget and risks

| Session | Work | Stop and report with |
|---|---|---|
| S0 | Repo skeleton; config and test lock; SMARD downloader; ID verification; hourly table; QA report; Databricks smoke test | QA report, what works on Free Edition, missing series, decisions |
| S1 | Features with availability tests; battery program with tests (toy days, DST days, perfect foresight ≥ every strategy); naive strategies on validation; Databricks bronze/silver/gold notebooks | Validation profit of S-perfect and the naive strategies; tests passing |
| S2 | M-reg and M-rank; random search; validation results for all strategies | Validation table; proposed final §6 |
| (Akram) | Review, commit §6, type "UNLOCK TEST" | – |
| S3 | Test run, once | H1 and H2 verdicts with CIs; full results table |
| S4 | §8 analyses; figures | Figures, short findings |
| S5 | Databricks job (parity with the Mac), MLflow, dashboard; README | Final README draft; screenshots |

**README must contain:**
- the question;
- the controlled design, in one picture: same values, different order;
- the results table with CIs;
- 3 figures: cumulative profit; accuracy vs value; an example day;
- limitations (TSO forecast timing, price-taker, day-ahead only, no fuel prices, simulated battery);
- how to run, on the Mac and on Databricks;
- the data credit.

**Budget:** Databricks Free Edition at no cost; Claude usage of $100 at most in total (aim for about $60 over 6 sessions); about 6–7 working days.

**Risks**

| Risk | Mitigation |
|---|---|
| SMARD IDs or format differ from the spec | Verify in S0; fall back to the download centre CSVs (Akram downloads them) |
| Filter 411 turns out not to be the load forecast | Confirm the label in S0; otherwise use the actual load at (D-7, h) as a proxy and state it |
| Free Edition limits (CLI, MLflow, jobs) | Find out in the S0 smoke test; fall back to the UI and notebooks run by hand |
| Look-ahead leakage | Availability rules plus `test_features.py`; the TSO caveat is tested in §8 |
| DST and time-zone bugs | UTC storage; DST tests |
| H1 not supported | Report it as pre-registered; the §8 analysis shows where profit is lost |
| Overfitting validation | Fixed tuning budget; test run once |

---

## 12. Lab notebook

- **2026-10-04.** Plan v1 written. Test period locked from 2025-10-01. The primary comparison is S-rank vs S-reg, where only the order differs.
- **2026-10-04.** SMARD API probed on a training-period day only (2024-06-10); no test-period values were read.
  - Weekly chunks start at Monday 00:00 Europe/Berlin; all series start on 2018-10-01.
  - 125 is the PV forecast. 126 and 4361 are −(wind + PV), not PV.
  - 411 looks like the day-ahead load forecast, and 4362 = 411 − 5097 exactly (forecast residual load).
  - `index_quarterhour` exists for 4169.
