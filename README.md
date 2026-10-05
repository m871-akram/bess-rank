# bess-rank: rank, don't forecast

*A battery doesn't need the price, it needs the order.*

- **Question.** Does training a model to rank the hours of each day, instead of predicting their
  prices, earn more money for a simulated 1 MW / 2 MWh battery, German day-ahead market,
  price-taker?
- **Result.** Training XGBoost to rank the hours earned +667 €/MW/yr more than training it to
  predict prices (+1.0%, 95% CI +285 to +1,036), closing 16% of the gap to perfect foresight.
  Pre-registered, tested once on a locked year. For the BiLSTM, ranking made no difference.
- **Accuracy is not value (exploratory).** Across 58 validation fits, how well a model orders the hours of each
  day predicts its profit far better than its RMSE does (rank correlation with profit +0.87 vs
  −0.59).

![Cumulative profit difference, rank minus price model, test year](results/figures/fig1_cumulative_difference.png)

*Fig. 1. Cumulative profit of the ranked strategy minus the price-model strategy over the locked
test year (2025-10-01 to 2026-09-30), per MW. Both strategies of a family use the same forecast
values; only their order within the day differs. XGBoost ends at +667 €/MW, the BiLSTM at
−30 €/MW.*

## Results (test year, pre-registered)

| Strategy | Price vector the battery schedules on | Profit (€/MW/yr) | Capture | RMSE (€/MWh) |
|---|---|---|---|---|
| S-perfect | actual prices (upper bound) | 73,181 | 100.0% | 0 |
| S-naive-1d | prices of the day before | 60,390 | 82.5% | 47.34 |
| S-naive-7d | prices of a week before | 57,936 | 79.2% | 57.46 |
| S-xgb-reg | XGBoost price forecasts | 69,107 | 94.4% | 30.26 |
| S-xgb-rank | the same values, in XGBoost-ranker order | 69,774 | 95.3% | 30.01 |
| S-lstm-reg | BiLSTM price forecasts | 69,864 | 95.5% | 28.50 |
| S-lstm-rank | the same values, in BiLSTM-ranker order | 69,834 | 95.4% | 28.38 |

Capture = profit ÷ perfect-foresight profit. Full table with VaR, ES, drawdown and hit rates:
[`results/test_summary.md`](results/test_summary.md).

**Hypotheses** (95% CIs from a moving-block bootstrap over days; verdicts as pre-registered):

- **H1 (confirmatory): supported.** S-xgb-rank − S-xgb-reg = +1.83 €/day (95% CI 0.78 to 2.84),
  +667 €/MW/year (+0.97%).
  *Exploratory checks next to H1, not tests:*
  - by quarter: +2.60, +0.76, +1.93 and +2.01 €/day;
  - with both models' validation tree counts fixed at every refit (no early stopping):
    +1.30 €/day (0.36 to 2.36). About 0.5 €/day of the official effect came from early stopping
    at the refits, mostly XGB-reg's: fixing the trees raises S-xgb-reg by 0.69 €/day and
    S-xgb-rank by 0.15;
  - TSO caveat: the TSO wind and solar forecasts used as features are only required by 18:00 on
    D-1, after the 12:00 auction. Without them both XGBoost strategies earn 5–6% less, and the
    difference is +4.38 €/day (2.20 to 6.51).
- **H2 (secondary): inconclusive.** The ranked vector's RMSE was lower, not equal: −0.25 €/MWh
  (95% CI −0.37 to −0.14) against a ±0.30 €/MWh equivalence margin.
- **H3 (secondary): inconclusive.** No difference for the LSTM: S-lstm-rank − S-lstm-reg =
  −0.08 €/day (95% CI −1.19 to 1.03).
- *Exploratory:* S-xgb-rank − S-lstm-reg = −0.25 €/day (95% CI −1.61 to 1.14). The BiLSTM price
  model alone earns about as much as the XGBoost ranker.

![Example test day: prices and both schedules](results/figures/fig2_example_day.png)

*Fig. 2. 2025-10-15, chosen because it is the test day on which S-xgb-rank and S-xgb-reg differ
most in profit (+90 €). It is the extreme case, not a typical day: the average difference is
+1.83 €/day. The price model put its highest values in the evening; the ranker put them on the
morning peak, which was the real one, and the battery sold there. Over the test year the two
XGBoost schedules differ on 294 of 365 days: the ranked one earns more on 178 of them and less
on 116.*

![Accuracy vs value: RMSE and within-day Spearman rho against profit](results/figures/fig3_accuracy_vs_value.png)

*Fig. 3. Validation year, 58 fits (every configuration and seed of the hyperparameter search).
Profit against RMSE (left) and against the mean within-day Spearman ρ between forecast and
actual prices (right). Triangles are rankers, shown through their reordered price vectors.*

## Method

- **Data.** SMARD (Bundesnetzagentur), bidding zone DE-LU, hourly: day-ahead prices, TSO
  day-ahead forecasts of load, wind and solar, actual load and actual residual load. Train 2019-01-01 to 2024-09-30,
  validation 2024-10-01 to 2025-09-30, test 2025-10-01 to 2026-09-30 (365 days). From
  2025-10-01 the hourly price is the mean of the four quarter-hour prices.
- **Decision time.** The schedule for day D is fixed at 11:00 Europe/Berlin on D-1. Each of the
  42 features declares when its data become available. A test rebuilds the features of 43 days
  (37 real, 6 synthetic) from only the data visible at 11:00 on D-1 under those rules and checks
  that the values are identical. The TSO wind and solar forecasts count as visible (see the
  TSO caveat and Limitations).
- **Models (2×2).** XGBoost and a bidirectional LSTM, each trained either to predict prices
  (squared error) or to rank the hours of each day (pairwise logistic loss over every pair of
  hours). Hyperparameters were chosen on the validation year by RMSE or within-day Spearman ρ,
  never by profit. Each model was refitted at the start of each test quarter, with 3 seeds.
- **Controlled comparison.** A "-rank" strategy keeps the price model's forecast values for the
  day and only reassigns them to hours in the ranker's order. Toy example with 4 hours:

  | Hour | Actual price | Price model | Ranker's order (1 = cheapest) | Ranked vector |
  |---|---|---|---|---|
  | 1 | 20 | 30 | 1 | 30 |
  | 2 | 60 | 45 | 3 | 55 |
  | 3 | 40 | 55 | 2 | 45 |
  | 4 | 90 | 80 | 4 | 80 |

  Both vectors hold the same four values (30, 45, 55, 80), so any profit difference comes from
  the order alone.
- **Battery.** For each day, a mixed-integer program (scipy/HiGHS) schedules on the strategy's
  price vector: 88% round trip, at most one full cycle per day, 50% charge at the start and end
  of each day, 10 €/MWh degradation, no charging and discharging in the same hour. The schedule
  is settled at the actual prices. OR-Tools/SCIP re-solves every day, and the run stops if the
  two objectives differ by more than 1e-6 relative or a strategy beats perfect foresight on any
  day.
- **Pre-registration.** The hypotheses, statistics, decision rules and every setting were fixed
  in [PLAN.md §6](PLAN.md) and merged before any test-period value was seen (tag `prereg`);
  until then the data checks on the test period reported only counts and pass/fail. Test data
  load only when `BESS_UNLOCK_TEST=1` is set and `PREREGISTRATION.lock` holds a commit hash (the
  committed file holds the `prereg` commit); the test run also refuses to start if the code
  differs from that commit. The test ran once, at tag `confirmatory-run`, and the code refuses
  to overwrite its results.
- **Uncertainty.** Every CI is the 95% percentile interval of a moving-block bootstrap over the
  365 test days (7-day blocks, 10,000 resamples), never over hours.

## Exploratory findings

Not pre-registered; details in [VALIDATION.md](VALIDATION.md), a model validation report in the
style of a bank's model-risk review.

- **Where profit is lost:** the order of the hours accounts for 7.6 of S-xgb-reg's 11.2 €/day
  gap to perfect foresight; the BiLSTM price model already orders the hours almost as well as
  the XGBoost ranker ([§9.2](VALIDATION.md#92-where-profit-is-lost-test-year)).
- **The LSTM's Jul–Sep 2026 loss is mostly one day:** 2026-09-14 accounts for −108 € of the
  quarter's −134 € ([§9.2](VALIDATION.md#92-where-profit-is-lost-test-year)).
- **Drift (PSI):** the price-level features shift strongly between training and test (PSI
  1.05–1.72); the within-day rank and profile features stay stable (0.00–0.07), and the
  ranker's top feature, the within-day residual-load deviation, shifts moderately (0.17)
  ([§8](VALIDATION.md#8-stability-and-monitoring-exploratory)).
- **Conformal quantiles:** after calibration on the validation year, the 90% interval covers
  88.6% of test hours (85.0% before)
  ([§9.4](VALIDATION.md#94-conformal-quantiles-and-the-cvar-frontier-stretch)).
- **CVaR schedules, a negative result:** risk-averse schedules lower mean profit by 5–19% and do
  not improve the realized expected shortfall; only the drawdown falls
  ([§9.4](VALIDATION.md#94-conformal-quantiles-and-the-cvar-frontier-stretch)).

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m bessrank.run data      # rebuild data/ from the Databricks volume, or download from SMARD
python -m bessrank.run qa        # data quality report (test period: counts and pass/fail only)
python -m bessrank.run features  # feature table (train + validation; the test year stays locked)
python -m bessrank.run tune xgb-reg   # validation-year search; also xgb-rank, lstm-reg, lstm-rank
python -m bessrank.run validate  # validation-year strategies and the table of every configuration
pytest
```

The test year is locked: test data load only when `BESS_UNLOCK_TEST=1` is set and
`PREREGISTRATION.lock` holds a commit hash (the committed one is the pre-registration merge,
517478e).

```bash
BESS_UNLOCK_TEST=1 python -m bessrank.run explore forecasts       # refit all 48 test models, compare
BESS_UNLOCK_TEST=1 python -m bessrank.run explore xgb-variants    # no-TSO and fixed-tree refits
BESS_UNLOCK_TEST=1 python -m bessrank.run explore analyses        # the VALIDATION.md analyses
BESS_UNLOCK_TEST=1 python -m bessrank.run explore conformal-cvar  # conformal quantiles and CVaR
BESS_UNLOCK_TEST=1 python -m bessrank.run report                  # the three figures above
```

`explore forecasts` refits every test model with the frozen code and compares the result with
the official run. A fresh VM reproduced all 48 test fits and the daily profits to within
2.3e-13 €. To re-run the confirmatory test itself, check out tag `confirmatory-run` in a
separate clone (it predates the committed results) and run
`BESS_UNLOCK_TEST=1 python -m bessrank.run test`; on later commits the command refuses because
`results/test_hypotheses.csv` exists. Package versions, the SMARD download time and the commit
of every stage are in [`results/provenance.json`](results/provenance.json).

## Databricks

The official XGBoost test pipeline also runs as a Databricks job on serverless compute (Free
Edition).

- **What runs there.** `python -m bessrank.run pipeline` packs one commit's code (a `git
  archive` of `src/`, `PREREGISTRATION.lock` and the frozen hyperparameters) and uploads it to
  the Unity Catalog volume `workspace.bess.raw`. It then imports
  [`notebooks/01_pipeline.py`](notebooks/01_pipeline.py) and runs the job `bess-rank-pipeline`
  ([definition](notebooks/bess-rank-pipeline.job.json); serverless, no schedule). The notebook
  is a thin wrapper around [`bessrank.pipeline`](src/bessrank/pipeline.py), which does the
  following:
  - refits the XGBoost pair every quarter with the frozen hyperparameters and 3 seeds;
  - builds the five non-LSTM strategies;
  - solves every day with HiGHS and re-checks it with SCIP, with the same stop rules as the VM.

  It is a test-type run: the job passes `BESS_UNLOCK_TEST=1` and the bundle carries the lock
  file. Package versions are pinned to those of the S3 run, and no Git credentials are stored
  in Databricks.
- **Delta tables** (`workspace.bess`; every row carries the commit and the test quarter):

  | Table | Rows | Content |
  |---|---|---|
  | `gold_features` | 8,760 | the 42 features for every test hour |
  | `gold_predictions` | 8,760 | actual price, XGBoost forecasts (per seed and averaged), the five strategies' price vectors |
  | `gold_schedules` | 43,800 | per strategy and hour: charge, discharge, state of charge, cash flow |
  | `gold_daily_profit` | 1,825 | per strategy and day: profit and cumulative profit |

- **MLflow experiment `bess-rank`.** 24 runs, one per fit (hyperparameters, tree count, the
  validation-year selection metric, the fit's test-quarter RMSE or Spearman ρ, commit), plus one
  summary run with the test metrics.
- **Parity with the VM run.** Run 967734030312010 (commit 4cd8d8a; 5.9 min in the notebook,
  about 7 min end to end) reproduced all 24 fits' forecasts exactly (largest difference 0.0). The daily profits of the five strategies
  match the S3 results within 2.3e-13 € on all 365 days. H1 recomputed from the Databricks
  numbers gives +1.83 €/day (95% CI 0.78 to 2.84). Details:
  [`results/databricks_pipeline.json`](results/databricks_pipeline.json).
- **Why the LSTM stays in the VM.** Running it on serverless would mean installing torch there.
  The LSTM refits also took most of the test run's 19 minutes, while Free Edition compute has
  daily quotas. The LSTM pair is reproduced in the VM instead (`explore forecasts` above).

## Limitations

- A simulated battery: 1 MW / 2 MWh, a linear degradation cost, one cycle per day.
- Price-taker: the battery's bids do not move the price, and every schedule is assumed to clear.
- Day-ahead market only: no intraday, balancing or ancillary revenue; no fees or grid charges.
- TSO wind and solar forecasts are only required by 18:00 on D-1, after the 12:00 day-ahead
  auction, so they may not be available at the decision time. They stand in for the vendor
  forecasts a trader would buy; without them the two XGBoost strategies earn 5–6% less (the
  BiLSTM was not re-tested).
- No fuel or carbon prices; lagged prices carry the price level.
- From 2025-10-01 the auction clears in 15-minute slots; this study averages them to hours.
- One market and one test year.

## Repository

- [`PLAN.md`](PLAN.md): design, pre-registration (§6) and lab notebook (§12).
- [`VALIDATION.md`](VALIDATION.md): model validation report and every exploratory analysis.
- [`src/bessrank/`](src/bessrank/): one module per stage; [`tests/`](tests/): the test suite.
- [`results/`](results/): committed results (no SMARD data); [`results/explore/`](results/explore/)
  holds the exploratory tables and figures.

---

Data: Bundesnetzagentur | SMARD.de

Implemented with Claude Code under the rules in [CLAUDE.md](CLAUDE.md). Question, design,
pre-registration and decisions: Akram M. Lrhorfi.

Code: [MIT License](LICENSE).
