# PLAN: Rank, don't forecast. Learning-to-rank for battery arbitrage

*A battery doesn't need the price, it needs the order.*

Status: plan v2, 2026-10-04. It is a 3-day sprint with Claude Code running in the cloud. Sections §1–§5 and §7–§11 may change until the pre-registration (§6) is merged. After that, changes go only into dated amendments in §12.

---

## 1. Question and claims

**Question.** A battery trading the German day-ahead market must decide which hours of tomorrow to charge in and which to discharge in. Two options:
- train a model to predict each hour's price, or
- train it to rank the hours of the day.

Which one makes the battery more money?

**Design: 2×2.**

| Model | Price objective | Rank objective |
|---|---|---|
| XGBoost | XGB-reg | XGB-rank (primary test) |
| Bidirectional LSTM | LSTM-reg | LSTM-rank (second test) |

**Why now**
- Germany recorded 457 hours of negative day-ahead prices in 2024 and 573 in 2025 (SMARD).
- Battery fleets are growing fast. In July 2026, Masdar and EWEC reached financial close on a 5.2 GW solar + 19 GWh battery project in Abu Dhabi.
- Since delivery day 2025-10-01, the auction clears in 15-minute slots.

**Gap**
- Price forecasts are trained and judged on RMSE.
- Maciejowska, Lipiecki & Uniejewski (arXiv:2511.13616, 2025) evaluate 192 price forecasts by battery-arbitrage profit. They argue that RMSE and MAE often fail to reflect that economic value.
- Decision-focused learning trains forecasts on the decision's outcome, but usually needs differentiable optimisation:
  - Elmachtoub & Grigas, "Smart Predict, then Optimize", *Management Science* 2022;
  - for storage, arXiv:2305.00362.

**Idea and controlled design**
- For one day, a battery's schedule depends mostly on the order of the hours. So train with a ranking objective: XGBoost's `rank:pairwise` with one query group per day, and a pairwise loss for the LSTM.
- In each family, the ranked strategy keeps the price model's forecast values for the day and only reassigns them to hours in the ranker's order. A profit difference then comes from the order alone.

**Risk layer.** This is the language of trading desks and bank risk teams:
- VaR and Expected Shortfall of daily profit;
- a bank-style model validation report;
- if time allows, conformal quantile forecasts feeding a CVaR-constrained schedule solved with OR-Tools.

**Possible headline** (only if supported by §6): "Same forecast values, different order: the ranked strategy earns X% more on the test year, despite a higher RMSE."

---

## 2. Data

**Source.** The SMARD chart API (Bundesnetzagentur). A community OpenAPI spec is at github.com/bundesAPI/smard-api.
- Index of chunk timestamps: `https://www.smard.de/app/chart_data/{filter}/{region}/index_{resolution}.json`
- Series chunk: `https://www.smard.de/app/chart_data/{filter}/{region}/{filter}_{region}_{resolution}_{timestamp}.json`

**Format** (probed on 2024-06-10)
- Chunks are weekly. Each starts at Monday 00:00 Europe/Berlin, as epoch milliseconds.
- `series` is a list of `[timestamp_ms, value]`, with `null` for missing values.
- Every series below starts on 2018-10-01 for region DE-LU.

**Series to download** (region DE-LU, hourly; prices also quarter-hourly from 2025-10-01)

| Series | Filter | Use | Status |
|---|---|---|---|
| Day-ahead price | 4169 | label, lagged features, settlement | probed |
| Forecast: wind onshore | 123 | feature | probed |
| Forecast: wind offshore | 3791 | feature | probed |
| Forecast: PV | 125 | feature | probed (0 at night). The community spec's 126 is −(wind + PV) |
| Forecast: load | 411 | feature | probed. Likely "Prognostizierter Stromverbrauch"; confirm the label |
| Actual load | 410 | lagged feature | probed |
| Actual residual load | 4359 | lagged feature | probed |

Neighbour prices (FR 254, NL 256, AT 4170) are optional; add them only if Day 1 runs ahead.

**Derived series**
- Wind + PV forecast = 123 + 3791 + 125. Check once that it equals filter 5097 on a sample week.
- Forecast residual load = 411 − (wind + PV). Check once that it equals 4362.

**Periods**
- **Train:** 2019-01-01 → 2024-09-30, without the 8 gap days listed under Missing values. Data from 2018-10-01 to 2018-12-31 feed only the lagged features of the first training days.
- **Validation:** 2024-10-01 → 2025-09-30.
- **Test (locked):** 2025-10-01 → 2026-09-30.

**The 15-minute market**
- The study is hourly throughout. For the test year, the hourly price is the mean of the four quarter-hour prices. This is exact for a battery that holds constant power within each hour.
- Check whether SMARD's `hour` series equals that mean. If not, compute it.

**Persistence.** Cloud VMs are fresh each session. `python -m bessrank.run data`:
1. pulls `processed/hourly.parquet` from the Databricks volume if it is reachable;
2. otherwise downloads from SMARD (at most 4 concurrent requests and about 10 requests/s, retries with backoff, about 3,000 requests) and builds the parquet;
3. then uploads the parquet to the volume when the API works.

**Missing values** (decided by Akram, 2026-10-05)
- Gaps of 1–2 hours in any series: linear interpolation in UTC time from the neighbouring hours of the same series. This covers the 00:00 hour that SMARD misses on 25-hour days (123, 125, 411), including 2025-10-26. Every filled value is flagged.
- Training starts on **2019-01-01**, after the 2018 run-in of the load forecast 411 (21 gaps of 1–4 days, 2018-10-02 to 2018-12-31). It does not move later because of the gaps after that date.
- From 2019-01-01 on, a training day with a gap longer than 2 hours in any series used by a feature or label (so a value still missing after the 1–2-hour filling) is dropped from training, and only that day. With the data downloaded in S0 these are 8 days: 2022-02-22, 2022-03-24, 2022-07-20, 2022-07-21, 2022-12-21, 2022-12-22, 2023-03-13 and 2023-08-31 (2,092 training days remain of 2,100). Test-period refits apply the same rule to their windows.
- Validation and test days are never dropped. XGBoost gets NaN (native missing-value handling); the LSTM gets the training-window median for that hour.
- The counts are in `results/qa_data.md` and `results/features_summary.json` (test period: counts only).

**Data quality report** (`results/qa_data.md`)
- Coverage, gaps and duplicates per series and year; DST days.
- Negative-price hours per year, up to 2025-09-30 only.
- Test period: counts and pass/fail format checks only.

---

## 3. Decision time and features

**Decision time:** 11:00 Europe/Berlin on D-1, for all hours of delivery day D.

**What is known at that time**
- Day-ahead prices up to the end of D-1 (published about 12:45 on D-2).
- Actual load up to the end of D-2. This is conservative.
- **TSO load forecast for D (411): available.** EU Regulation 543/2013 Art. 6(2)(b) requires it at least 2 hours before gate closure, so by 10:00.
- **TSO generation forecasts for D (123, 3791, 125, and everything derived from them): a caveat.** Art. 14(1)(c)–(d) only requires them by 18:00 Brussels time on D-1, after the 12:00 auction.
  - Following the price-forecasting literature (Lago et al. 2021, *Applied Energy*, epftoolbox), they are used as a proxy for the morning forecasts traders buy from weather vendors.
  - This is stated in the README limitations and tested in §8 with a run without them (the load forecast is kept).

**Features for each hour h of day D**

Each feature declares its availability rule in `features.py`:
- **Calendar:** hour, weekday, month, day of year (sin/cos), national holiday (`holidays`, DE), bridge day, DST flag, number of hours in the day.
- **Forecasts for (D, h):**
  - load, onshore, offshore, PV, wind + PV, its share of the load, residual load;
  - within-day normalised versions, e.g. PV ÷ the day's maximum PV, and the within-day rank of the residual load.
- **Daily aggregates for D:** mean, min and max residual load; total PV energy; total wind energy.
- **Lagged prices:**
  - the price at (D-1, h) and (D-7, h);
  - the mean, min, max and spread of D-1 and D-7;
  - the within-day rank of h on D-1 and D-7;
  - the 28-day mean normalised profile by hour.
- **Lagged actuals:** load and residual load at (D-2, h) and (D-7, h).

Not used: fuel and carbon prices (no free, clean source). Lagged prices carry the level. Stated as a limitation.

---

## 4. Models

All models use the same features and training windows, and 3 seeds per final configuration. Each final forecast averages the 3 seeds: mean prediction for price models, mean score for rankers.

- **Version:** XGBoost 3.2.0, the newest the VM's package index offers (the 3.4.1 prototype check in §12 ran elsewhere). All settings below are checked on 3.2.0.
- **XGB-reg.**
  - Target: hourly price (€/MWh). Objective `reg:squarederror`, `tree_method="hist"`.
- **XGB-rank.**
  - One query group per delivery day (`qid` = day; rows sorted by day, then hour).
  - Label: dense within-day rank of the actual price, 0 for the cheapest; tied prices share a label.
  - Objective `rank:pairwise` with every pair of hours of a day that have different labels, and the plain logistic loss log(1 + exp(−(s_i − s_j))) summed over those pairs (the LSTM-rank loss).
  - In XGBoost 3.2.0 this is `lambdarank_pair_method="topk"` with `lambdarank_num_pair_per_sample=25` (every hour is in the top 25, so all pairs are built), `lambdarank_normalization=False` and `lambdarank_score_normalization=False`. `tests/test_models.py` checks it against a hand-written gradient. (S2 finding: `"mean"` samples pairs at random, so it cannot guarantee every pair; the v2 text asked for `"mean"`.)
- **LSTM-reg and LSTM-rank.**
  - **Input:** each sample is one delivery day, a sequence of up to 25 hourly feature vectors with the same features as XGBoost.
    - Features are standardised with training-window statistics only.
    - Days are padded to 25 hours, with a mask.
  - **Architecture:** bidirectional LSTM (default 2 layers, 64 units, dropout 0.1 between layers and before the head), then a linear head that outputs one value per hour. Bidirectional means each hour's output sees the whole day. Sequences are packed, so padding never reaches a real hour (`tests/test_lstm.py`).
  - **Losses:**
    - LSTM-reg: masked MSE on the standardised price.
    - LSTM-rank: pairwise logistic loss over every pair of hours with different labels, log(1 + exp(−(s_i − s_j))), the analogue of XGBoost's objective.
  - **Training:** Adam (lr 1e-3), 32 days per batch. Early stopping on the last 3 months of the training window (patience 10, at most 100 epochs) on the selection metric: RMSE for LSTM-reg, mean within-day Spearman ρ for LSTM-rank. CPU.
- **XGB-quantile** (stretch, §8).
  - Objective `reg:quantileerror`, with 19 quantiles from 0.05 to 0.95.
  - Calibrated by conformalized quantile regression (CQR) on the validation year.

---

## 5. Strategies, battery and settlement

**Battery**
- 1 MW / 2 MWh; round-trip efficiency 88% (η_c = η_d = √0.88).
- State of charge starts and ends every day at 50% (1 MWh).
- At most one equivalent full cycle per day: total discharge ≤ 2 MWh.
- Degradation cost: 10 €/MWh discharged.
- No simultaneous charging and discharging. This needs a binary per hour, because with negative prices a plain linear program would charge and discharge at once to burn energy.

**Program for one day with n hours (23, 24 or 25):**
maximise Σ_h p_h (d_h − c_h) − 10 Σ_h d_h, subject to:
- 0 ≤ c_h ≤ u_h and 0 ≤ d_h ≤ 1 − u_h, with u_h binary;
- soc_{h+1} = soc_h + η_c c_h − d_h / η_d;
- 0 ≤ soc ≤ 2, and soc_0 = soc_n = 1;
- Σ_h d_h ≤ 2.

**Solvers**
- Primary: `scipy.optimize.milp` (HiGHS). A prototype solves one day in about 15 ms.
- Check: the same program in OR-Tools (`pywraplp` with SCIP) on every day solved. Objective values must agree within 1e-6 relative, otherwise the run stops.

**Settlement**
- The schedule comes from a strategy's price vector and is settled at the actual hourly prices.
- Price-taker; no fees, grid charges, intraday or balancing revenue.

**Strategies:** each produces a price vector for day D.

| Strategy | Price vector |
|---|---|
| S-perfect | actual prices (upper bound) |
| S-naive-1d | prices of D-1 |
| S-naive-7d | prices of D-7 |
| S-xgb-reg | XGB-reg forecasts |
| S-xgb-rank | XGB-reg's values, reassigned in XGB-rank's order |
| S-lstm-reg | LSTM-reg forecasts |
| S-lstm-rank | LSTM-reg's values, reassigned in LSTM-rank's order |
| S-cvar(λ) (stretch) | risk-aware schedule from quantile scenarios (§8) |

**How a "-rank" strategy is built.**
1. Sort the day's price-model forecasts in increasing order: v₍₁₎ ≤ … ≤ v₍ₙ₎.
2. Order the hours by ranker score, cheapest first: h₍₁₎ … h₍ₙ₎. Break ties by the price forecast, then by time.
3. Set p̃(h₍ₖ₎) = v₍ₖ₎.

If the ranker agrees with the price model's order, the two strategies are identical.

**Metrics**
- **Value:** profit (€ per MW per year), daily profit, capture rate (profit ÷ S-perfect profit).
- **Forecast:** price RMSE and MAE; within-day Spearman ρ; hit rate for the 2 cheapest and the 2 most expensive hours.
- **Risk** (every strategy):
  - VaR 5%, reported as "P5 of daily profit": the 5th percentile of daily profit;
  - Expected Shortfall 5%, reported as "mean of the worst 5% of days": the mean daily profit of the worst ⌈5% × n⌉ days (19 of 365);
  - both are profit levels, not losses: a negative value is a loss. Every table labels them this way;
  - maximum drawdown of cumulative profit;
  - share of losing days.

---

## 6. Pre-registration (draft; finalise after validation, before unlocking the test)

**Tests.** Each uses the mean daily profit difference, with a 95% CI from a moving-block bootstrap: 7-day blocks, 10,000 resamples, seed 20261004. Each is **supported** if the CI's lower bound > 0, **contradicted** if the upper bound < 0, and **inconclusive** otherwise.
- **H1 (primary).** S-xgb-rank earns more than S-xgb-reg on the test year.
- **H2.** S-xgb-rank's price vector has a higher RMSE than S-xgb-reg's, with the same bootstrap CI above 0. The headline "worse RMSE, more profit" needs both H1 and H2.
- **H3.** S-lstm-rank earns more than S-lstm-reg.

**Reported for every strategy:** annual profit, capture rate, RMSE and MAE, Spearman ρ, hit rates, and the risk metrics.

**Test protocol**
- Hyperparameters are fixed from validation.
- Every model is refitted at the start of each test quarter (2025-10-01, 2026-01-01, 2026-04-01, 2026-07-01), on all data up to the day before.
- 3 seeds, averaged. Run once.

**Lock procedure**
1. Akram merges the PR containing the final version of this section.
2. `PREREGISTRATION.lock` is created with that commit hash.
3. Only then may `BESS_UNLOCK_TEST=1` be used.

Anything decided after the lock is a dated amendment in §12 and is labelled exploratory.

---

## 7. Validation protocol (2024-10-01 → 2025-09-30; budget sized for the sprint)

- Fit on train, predict the validation year.
- Early stopping on the last 3 months of the training window, then a refit on the full window with the chosen number of trees or epochs. XGBoost: at most 5,000 trees, stop after 50 trees without improvement, on RMSE for XGB-reg and mean within-day Spearman ρ for XGB-rank.
- **XGBoost:** random search over 15 configurations per model, seed 0. The best 3 are re-run with 3 seeds, and the best is kept.
  - Search space:
    - max_depth 3–10;
    - learning_rate 0.01–0.2 (log scale);
    - min_child_weight 1–20;
    - subsample 0.6–1, colsample_bytree 0.5–1;
    - reg_lambda 0.1–10 (log scale).
  - Selection: RMSE for XGB-reg, mean within-day Spearman ρ for XGB-rank.
- **LSTM:** 6 configurations (hidden 32, 64 or 128 × 1 or 2 layers), seed 0. The best is re-run with 3 seeds. Same selection metrics.
- **Not tuned on profit.** The comparison stays about the training objective.
- **Outputs:**
  - validation results for every strategy;
  - a table of all configurations with RMSE, Spearman ρ and validation profit (used in §8);
  - the proposed final §6.

---

## 8. After the test (exploratory), in priority order

**Must (Day 3)**
1. Risk metrics for every strategy (§5).
2. `VALIDATION.md`, a model validation report in the style of a bank's model-risk team:
   - purpose and scope; data and quality; conceptual soundness and assumptions;
   - outcome analysis: backtests by quarter and month, profit distribution, VaR and ES;
   - benchmarking: the LSTM as challenger, the naive strategies;
   - sensitivity: degradation cost 0 and 20 €/MWh; 1 h and 4 h batteries;
   - robustness: no TSO generation forecasts;
   - stability and monitoring: Population Stability Index (PSI) of the key features, test vs train, and performance drift by quarter;
   - limitations; governance (versions, provenance, MLflow runs).

**Should**

3. Accuracy vs value: RMSE and Spearman ρ against validation profit, across all configurations.
4. Where profit is lost:
   - true order with the price model's values;
   - the price model's order with the true values.

**Stretch**

5. Conformal quantile forecasts (coverage, pinball loss), then a CVaR-constrained schedule:
   - scenarios: 200 per day from the calibrated quantiles, with a Gaussian copula fitted on within-day residual correlations (training data);
   - objective: maximise mean profit − λ · CVaR₉₅% of loss, using the Rockafellar–Uryasev formulation, solved in OR-Tools (SCIP);
   - λ ∈ {0, 0.5, 1, 2, 5};
   - plot realized mean profit against realized ES on the test year.
6. SHAP top features for XGB-rank and XGB-reg.

**Cut:** the 15-minute extension; profit-tuned models.

---

## 9. Databricks (Free Edition): minimal path

1. **Storage:** catalog `workspace`, schema `bess`, volume `raw`. The processed data lives in the volume and doubles as the session cache (§2).
2. **Notebook:** `notebooks/01_pipeline.py` runs the official XGBoost test pipeline. It writes Delta tables (`gold_features`, `gold_predictions`, `gold_schedules`, `gold_daily_profit`) and logs each model fit to the MLflow experiment `bess-rank`.
3. **Job:** one job that wraps the notebook. Its numbers must match the VM run within 1e-6 relative, or the difference is explained.
4. **Dashboard (if time):** cumulative profit by strategy, capture rate by quarter, and an example day.

- **Remote loop from the cloud session (REST API):**
  1. update the Databricks Git folder to the working branch (Repos API);
  2. submit a serverless run (Jobs API `runs/submit`) and poll it;
  3. read the outputs from the volume (Files API).
- **If the API is unreachable or authentication fails:** Akram runs the notebook and the job in the web interface.
- **LSTM runs stay in the VM**, to avoid installing torch on serverless; documented in the README.

---

## 10. Repo layout

```
bess-rank/
  CLAUDE.md  PLAN.md  README.md  VALIDATION.md  requirements.txt  pyproject.toml  .gitignore  .claude/settings.json
  src/bessrank/
    config.py      dates, battery parameters, paths, test lock
    data.py        SMARD download, Databricks cache, hourly table, QA report
    features.py    features with availability rules
    models.py      XGBoost price, ranking and quantile models; tuning; seed averaging
    lstm.py        BiLSTM, both losses, training loop
    battery.py     daily MILP (scipy) + OR-Tools cross-check + CVaR program
    strategies.py  price vectors per strategy (including the "-rank" reassignment)
    evaluate.py    profit, capture, forecast metrics, block bootstrap
    risk.py        VaR, ES, drawdown, PSI, conformal calibration
    databricks.py  minimal REST helpers (Files, Repos, Jobs)
    run.py         python -m bessrank.run {data,qa,features,validate,test,explore,report}
  notebooks/       01_pipeline.py (Databricks); analysis notebooks reading results/
  tests/           test_lock, test_time, test_features, test_battery, test_strategies
  results/         small CSV / JSON / PNG outputs (committed)
  data/            never committed
```

---

## 11. Three-day schedule, gates and cut list

| When | Session | Work | Gate before moving on |
|---|---|---|---|
| Day 1 morning | S0 | Skeleton; config and test lock; data command (download, Databricks cache); ID checks; hourly table; DST tests; QA report; Databricks smoke test | QA report clean; Databricks status known |
| Day 1 afternoon | S1 | Features with availability tests; battery program with OR-Tools cross-check and tests; perfect-foresight and naive strategies on validation; risk metric functions | All tests pass; plausible validation profits |
| Day 2 morning | S2 | XGBoost and LSTM, both objectives; tuning per §7; validation table; proposed final §6 | Validation table sent to Akram |
| Day 2 midday | Akram | Read the validation results; merge §6; type "UNLOCK TEST" | Lock file committed |
| Day 2 afternoon | S3 | Test run, once; H1–H3 verdicts with CIs; risk metrics | Results table |
| Day 3 morning | S4 | §8 Must and Should; `VALIDATION.md`; Stretch only if ahead | Report drafted |
| Day 3 afternoon | S5 | Databricks pipeline, MLflow and job (dashboard if time); README with 3 figures; final checks | README ready to make public |

**Cut list.** If behind, cut in this order:
1. SHAP;
2. conformal + CVaR;
3. dashboard;
4. the "where profit is lost" analysis;
5. accuracy vs value;
6. LSTM tuning reduced to 2 configurations.

**Never cut:** the test lock and pre-registration, the tests, H1–H3, the risk metrics, `VALIDATION.md`, the README.

**Akram's review:** 30–60 minutes at the end of each day reading the code and numbers. In interviews he will be asked to explain any line.

**Budget:** Claude cloud usage of $100 at most; Databricks Free Edition at no cost.

**Risks**

| Risk | Mitigation |
|---|---|
| SMARD format or IDs differ | Verify in S0; fall back to the download-centre CSVs (Akram downloads them) |
| Free Edition offers no token or API access | Akram runs the notebook and job in the web interface |
| Look-ahead leakage | Availability rules plus `test_features.py`; the TSO caveat is tested in §8 |
| DST and time-zone bugs | UTC storage; DST tests |
| H1 or H3 not supported | Report it as pre-registered; the §8 analysis shows where profit is lost |
| Sprint pressure | Gates and cut list; no step skipped silently |

---

## 12. Lab notebook

- **2026-10-04.** Plan v1 written. Test period locked from 2025-10-01. The primary comparison is S-rank vs S-reg, where only the order differs.
- **2026-10-04.** SMARD API probed on a training-period day only (2024-06-10); no test-period values were read.
  - Weekly chunks start at Monday 00:00 Europe/Berlin; all series start on 2018-10-01.
  - 125 is the PV forecast. 126 and 4361 are −(wind + PV), not PV.
  - 411 looks like the day-ahead load forecast, and 4362 = 411 − 5097 exactly (forecast residual load).
  - `index_quarterhour` exists for 4169.
- **2026-10-04.** Plan v2, decided with Akram:
  - 3-day sprint, run in Claude Code cloud sessions.
  - 2×2 design with a BiLSTM as second model family (H3 added).
  - Quarterly test refits instead of monthly.
  - OR-Tools added: a second solver checks every day, and it runs the CVaR program.
  - Risk layer added: VaR, ES, drawdown and `VALIDATION.md`; conformal + CVaR as stretch.
  - RL rejected: the daily schedule is solved by an exact optimisation, so an RL agent could only approximate it.
  - Fairness and EU AI Act left to a separate project: this system makes no decisions about people.
- **2026-10-04.** Prototype checks on a synthetic 24-hour day:
  - `scipy.optimize.milp` (scipy 1.18) and OR-Tools 9.15 with SCIP both give 215.0312 €, in about 10–15 ms each.
  - XGBoost 3.4.1 accepts `XGBRanker(objective="rank:pairwise", lambdarank_pair_method="mean")` with dense integer labels and `qid`.
- **2026-10-04 (S0).** Setup, SMARD checks and Databricks status. No test-period value was printed; test-period results below are counts and pass/fail only.
  - Blocker: the cloud environment's network policy denies `pypi.org` and `files.pythonhosted.org`, so pandas, pyarrow, xgboost, pytest etc. could not be installed. The package code (`config`, `data`, `databricks`, `run`) and tests are written; the lock logic was exercised with plain Python, but the pandas code paths, `tests/`, `results/qa_data.md` and the processed parquet have not run yet. The numbers below come from a throwaway stdlib script on the raw chunks and must be reproduced by `python -m bessrank.run data` / `qa`.
  - SMARD download: 2,979 weekly chunks (7 hourly series from 2018-10-01, quarter-hour 4169 from the chunk of 2025-09-29) in 105 s with 4 concurrent requests; 0 failures, 0 duplicate timestamps.
  - ID checks, week 2024-06-10 to 2024-06-16 (168 h, training): 123 + 3791 + 125 = 5097 and 411 − (123 + 3791 + 125) = 4362, both with max |difference| 0.0 MWh. 411 vs actual load 410: MAPE 3.86%, correlation 0.989, no identical hours, mean 52,341 vs 50,512 MWh (411 about 3.6% above 410 that week); 411 has values 25 hours beyond the last 410 value. Consistent with a day-ahead load forecast. The label itself is not exposed by the chart API, so it is confirmed by behaviour, not by name. No mismatch with §2.
  - Missing hours, 2018-10-01 to 2025-09-30 (61,368 h): 4169 0; 3791 0; 123 2; 125 2; 411 1,036 (2018: 842, 2022: 144, 2023: 49, 2024: 1; longest gap 96 h from 2018-10-02); 410 4 and 4359 4 (all in 2018). 123 and 125 miss only the first hour (00:00 CEST) of the 25-hour days 2023-10-29 and 2024-10-27, and 411 the one of 2024-10-27; the values around them are not shifted (PV peaks at 11:00 UTC on 26, 27 and 28 Oct 2024).
  - DST: 2024-03-31 has 23 hours and 2024-10-27 has 25 in the raw price series; 7 days of 23 h and 7 of 25 h before the test period.
  - Negative-price hours: 2018 (Oct–Dec) 27, 2019 211, 2020 298, 2021 139, 2022 69, 2023 301, 2024 457 (equals SMARD's figure in §1), 2025 (Jan–Sep) 525.
  - Test period (8,760 h), counts and pass/fail: 4169, 3791, 411, 410, 4359 complete (pass); 123 and 125 miss 1 hour each (fail; the first hour of the 25-hour day 2025-10-26, same DST pattern); every hour has 4 quarter-hour prices (pass); SMARD's hourly price equals the quarter-hour mean within 0.01 €/MWh on every hour (pass).
  - Databricks Free Edition: REST authentication works through the session proxy; schema `workspace.bess` and managed volume `raw` created; Files API upload and download work; notebooks import; serverless `runs/submit` works, `%pip install xgboost` works on serverless, and training and reading the volume ran. MLflow failed at first because serverless blocks reading `spark.mlflow.modelRegistryUri`, which `mlflow.set_registry_uri("databricks-uc")` fixes. The second run then failed on our own path clash (notebook folder = experiment path), now fixed. Stopped after two failed runs (CLAUDE.md), so MLflow logging and the Delta write are still unverified.
- **2026-10-05 (S0 finished, S1).** No test-period value was printed; test-period results are counts and pass/fail only.
  - Environment: PyPI now reachable; pandas 3.0.6, scipy 1.17.1, OR-Tools 9.15, XGBoost 3.2.0 (the newest PyPI offers here; `XGBRanker(objective="rank:pairwise", lambdarank_pair_method="mean")` checked on 3.2.0). **torch is not installed:** `download.pytorch.org` redirects to `download-r2.pytorch.org`, which the network policy denies. Needed in S2 for the LSTM.
  - S0 numbers reproduced by `python -m bessrank.run data` / `qa`: 2,979 chunks in 303 s (4 in flight, ≤ 10 requests/s), 0 failures, 0 duplicate or off-grid timestamps; 5097 and 4362 checks max |difference| 0.0 MWh; 411 vs 410 MAPE 3.86%, correlation 0.9886; missing hours per series and year as in the S0 entry; negative-price hours as in the S0 entry (2024: 457). The "hours of 411 beyond the last 410 value" is 16 now vs 25 in S0: it depends on the time of day of the check.
  - The processed parquet is on the volume (`processed/hourly.parquet`); a rebuild from the volume gives byte-identical files.
  - Databricks smoke test passed end to end: serverless run in 95 s, MLflow run `s0-smoke-test` logged in experiment `/Users/<user>/bess-rank` (holdout RMSE 24.83 EUR/MWh, train-period holdout Jul–Sep 2024), Delta table `workspace.bess.smoke_predictions` written (managed, Delta).
  - Missing values (Akram's rules, §2): 411 has 27 gaps > 2 h: 21 in the 2018 run-in and 6 whole-day gaps in 2022–2023. Training start moved to **2019-01-01** (removes 92 days, 2018-10-01 to 2018-12-31). Read literally, "the first day after which 411 has no gap > 2 h" would be 2023-09-01; 2019-01-01 follows the expected "early 2019" and the separate rule for later gap days (to confirm). 8 training days dropped (2022-02-22, 2022-03-24, 2022-07-20, 2022-07-21, 2022-12-21, 2022-12-22, 2023-03-13, 2023-08-31) of 2,100. Filled 1-hour gaps: the 00:00 hour of 2023-10-29 (train) and 2024-10-27 (validation) in 123, 125 and 411. Validation: 0 days with a missing feature. Test: 2 hours filled (123 and 125 on 2025-10-26), 0 hours missing after filling, 0 days affected.
  - Features (S1): 42, of which 15 use TSO generation forecasts. Choices: lags look up the same wall-clock hour on D-k (25-hour days average their two 02:00 hours; 23-hour days get 02:00 = mean of 01:00 and 03:00); S-naive-1d/7d use the same mapping. The 28-day profile is the mean of each day's within-day z-scored prices over D-28..D-1. Daily lagged statistics are NaN if any hour is missing. Holidays are German national holidays only; a bridge day is a Monday before a Tuesday holiday or a Friday after a Thursday holiday. `test_features.py` recomputes features from the data visible at 11:00 on D-1 for 37 real days (DST days, gap neighbours, period edges, 30 random) and 6 synthetic days: identical values, exact equality.
  - Battery: both solvers run with a relative MIP gap of 0 (HiGHS defaults to 1e-4). Agreement check: |a − b| ≤ 1e-6 · max(1 EUR, |a|, |b|).
  - Validation year (2024-10-01 to 2025-09-30, 365 days; 1,095 solves, HiGHS and SCIP agreed on all; perfect ≥ every strategy on every day): S-perfect 67,519 EUR/MW/year, VaR5 35.72 and ES5 25.84 EUR/day; S-naive-1d 55,115 EUR/MW/year, capture 81.6%, VaR5 −2.58, ES5 −23.79 EUR/day, 5.5% losing days, RMSE 47.08 EUR/MWh, Spearman ρ 0.760; S-naive-7d 54,106 EUR/MW/year, capture 80.1%, VaR5 0.66, ES5 −19.93 EUR/day, 4.9% losing days, RMSE 60.45 EUR/MWh, ρ 0.781. Exploratory: naive-1d − naive-7d = 2.76 EUR/day, block-bootstrap 95% CI [−4.50, 9.81].
