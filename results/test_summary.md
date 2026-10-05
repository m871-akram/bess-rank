# Test: 2025-10-01 to 2026-09-30 (365 delivery days)

Test year, pre-registered run (PLAN.md §6). H1 is the only confirmatory test; H2 and H3 are secondary, without multiplicity adjustment.

Simulated 1 MW / 2 MWh battery, day-ahead market only, price-taker. Rounded for display; the `test_*.csv` files next to this one hold the raw numbers.

## Hypotheses

Moving-block bootstrap over days: 7-day blocks, 10,000 resamples, seed 20261004 (same resampled days for every row). Margin = 1% of the reg strategy's RMSE over the period.

| hypothesis | role | comparison | statistic | estimate | ci95_low | ci95_high | margin | verdict |
|---|---|---|---|---|---|---|---|---|
| H1 | confirmatory | S-xgb-rank - S-xgb-reg | mean daily profit difference (EUR/day) | 1.828 | 0.779 | 2.84 |  | supported |
| H2 | secondary | S-xgb-rank - S-xgb-reg | RMSE difference over all hours (EUR/MWh) | -0.251 | -0.371 | -0.143 | 0.303 | inconclusive |
| H3 | secondary | S-lstm-rank - S-lstm-reg | mean daily profit difference (EUR/day) | -0.081 | -1.193 | 1.034 |  | inconclusive |
| LSTM RMSE | descriptive | S-lstm-rank - S-lstm-reg | RMSE difference over all hours (EUR/MWh) | -0.121 | -0.182 | -0.04 | 0.285 | descriptive (no test) |

| Hypothesis | Annual diff (EUR/MW/year) | Diff / reg profit | Days rank better | Days reg better | Days same vector |
|---|---|---|---|---|---|
| H1 | 667.0 | 0.97% | 178.0 | 116.0 | 0.0 |
| H3 | -30.0 | -0.04% | 154.0 | 117.0 | 0.0 |

## Strategies

VaR 5% is the 5th percentile (P5) of daily profit and ES 5% the mean daily profit of the worst 5% of days. Both are profit levels: a negative value is a loss.

| strategy | Profit (EUR/MW/year) | Capture | RMSE (EUR/MWh) | MAE (EUR/MWh) | Spearman rho | Hit 2 cheapest | Hit 2 dearest | VaR 5%: P5 of daily profit (EUR) | ES 5%: mean of the worst 5% of days (EUR) | Losing days | Max drawdown (EUR) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S-perfect | 73181 | 100.0% | 0.0 | 0.0 | 1.0 | 100.0% | 100.0% | 21.733 | 14.726 | 0.0% | 0.0 |
| S-naive-1d | 60390 | 82.5% | 47.339 | 29.823 | 0.781 | 53.6% | 54.2% | -7.566 | -23.786 | 7.9% | 61.394 |
| S-naive-7d | 57936 | 79.2% | 57.456 | 37.38 | 0.794 | 52.3% | 53.0% | -2.58 | -25.551 | 6.6% | 222.304 |
| S-xgb-reg | 69107 | 94.4% | 30.262 | 19.215 | 0.936 | 69.3% | 71.5% | 14.815 | 5.676 | 1.1% | 15.97 |
| S-xgb-rank | 69774 | 95.3% | 30.012 | 19.07 | 0.952 | 73.7% | 74.9% | 15.174 | 6.257 | 0.8% | 13.144 |
| S-lstm-reg | 69864 | 95.5% | 28.501 | 18.818 | 0.945 | 72.5% | 73.0% | 14.781 | 6.901 | 0.5% | 11.306 |
| S-lstm-rank | 69834 | 95.4% | 28.38 | 18.744 | 0.954 | 73.7% | 75.6% | 15.273 | 7.704 | 0.8% | 11.265 |

## By quarter (descriptive)

| Quarter from | Strategy | Days | Profit (EUR/MW) | Capture | RMSE (EUR/MWh) | Spearman rho |
|---|---|---|---|---|---|---|
| 2025-10-01 | S-perfect | 92 | 9404 | 100.0% | 0.0 | 1.0 |
| 2025-10-01 | S-naive-1d | 92 | 5741 | 61.1% | 40.215 | 0.685 |
| 2025-10-01 | S-naive-7d | 92 | 4936 | 52.5% | 51.472 | 0.697 |
| 2025-10-01 | S-xgb-reg | 92 | 8253 | 87.8% | 22.914 | 0.908 |
| 2025-10-01 | S-xgb-rank | 92 | 8492 | 90.3% | 22.554 | 0.921 |
| 2025-10-01 | S-lstm-reg | 92 | 8322 | 88.5% | 21.912 | 0.918 |
| 2025-10-01 | S-lstm-rank | 92 | 8357 | 88.9% | 21.83 | 0.924 |
| 2026-01-01 | S-perfect | 90 | 11436 | 100.0% | 0.0 | 1.0 |
| 2026-01-01 | S-naive-1d | 90 | 8185 | 71.6% | 39.656 | 0.691 |
| 2026-01-01 | S-naive-7d | 90 | 7183 | 62.8% | 45.485 | 0.71 |
| 2026-01-01 | S-xgb-reg | 90 | 10567 | 92.4% | 20.085 | 0.916 |
| 2026-01-01 | S-xgb-rank | 90 | 10635 | 93.0% | 19.866 | 0.941 |
| 2026-01-01 | S-lstm-reg | 90 | 10767 | 94.2% | 18.217 | 0.928 |
| 2026-01-01 | S-lstm-rank | 90 | 10787 | 94.3% | 18.08 | 0.946 |
| 2026-04-01 | S-perfect | 91 | 26152 | 100.0% | 0.0 | 1.0 |
| 2026-04-01 | S-naive-1d | 91 | 22870 | 87.4% | 54.912 | 0.859 |
| 2026-04-01 | S-naive-7d | 91 | 22479 | 86.0% | 65.648 | 0.853 |
| 2026-04-01 | S-xgb-reg | 91 | 24787 | 94.8% | 35.722 | 0.965 |
| 2026-04-01 | S-xgb-rank | 91 | 24963 | 95.5% | 35.565 | 0.975 |
| 2026-04-01 | S-lstm-reg | 91 | 24966 | 95.5% | 34.097 | 0.973 |
| 2026-04-01 | S-lstm-rank | 91 | 25015 | 95.6% | 34.191 | 0.974 |
| 2026-07-01 | S-perfect | 92 | 26188 | 100.0% | 0.0 | 1.0 |
| 2026-07-01 | S-naive-1d | 92 | 23594 | 90.1% | 52.465 | 0.89 |
| 2026-07-01 | S-naive-7d | 92 | 23338 | 89.1% | 64.503 | 0.914 |
| 2026-07-01 | S-xgb-reg | 92 | 25499 | 97.4% | 38.098 | 0.956 |
| 2026-07-01 | S-xgb-rank | 92 | 25684 | 98.1% | 37.782 | 0.972 |
| 2026-07-01 | S-lstm-reg | 92 | 25809 | 98.5% | 35.607 | 0.961 |
| 2026-07-01 | S-lstm-rank | 92 | 25675 | 98.0% | 35.253 | 0.973 |

| Quarter from | Comparison | Days | Mean daily diff (EUR) | Days rank better | Days reg better | RMSE diff (EUR/MWh) |
|---|---|---|---|---|---|---|
| 2025-10-01 | S-xgb-rank - S-xgb-reg | 92 | 2.599 | 55 | 24 | -0.36 |
| 2025-10-01 | S-lstm-rank - S-lstm-reg | 92 | 0.386 | 37 | 39 | -0.082 |
| 2026-01-01 | S-xgb-rank - S-xgb-reg | 90 | 0.757 | 42 | 39 | -0.219 |
| 2026-01-01 | S-lstm-rank - S-lstm-reg | 90 | 0.224 | 46 | 24 | -0.137 |
| 2026-04-01 | S-xgb-rank - S-xgb-reg | 91 | 1.926 | 43 | 24 | -0.157 |
| 2026-04-01 | S-lstm-rank - S-lstm-reg | 91 | 0.532 | 34 | 24 | 0.094 |
| 2026-07-01 | S-xgb-rank - S-xgb-reg | 92 | 2.009 | 38 | 29 | -0.316 |
| 2026-07-01 | S-lstm-rank - S-lstm-reg | 92 | -1.453 | 37 | 30 | -0.354 |

## Refits

| Refit (quarter from) | Model | Training days | Days dropped (§2) | Trees/epochs per seed | Fit time, 3 seeds (s) |
|---|---|---|---|---|---|
| 2025-10-01 | lstm-rank | 2457 | 8 | 9, 6, 8 | 239 |
| 2025-10-01 | lstm-reg | 2457 | 8 | 39, 28, 8 | 240 |
| 2025-10-01 | xgb-rank | 2457 | 8 | 69, 70, 188 | 37 |
| 2025-10-01 | xgb-reg | 2457 | 8 | 616, 858, 810 | 43 |
| 2026-01-01 | lstm-rank | 2549 | 8 | 13, 7, 6 | 271 |
| 2026-01-01 | lstm-reg | 2549 | 8 | 14, 18, 9 | 157 |
| 2026-01-01 | xgb-rank | 2549 | 8 | 231, 124, 174 | 58 |
| 2026-01-01 | xgb-reg | 2549 | 8 | 39, 48, 43 | 6 |
| 2026-04-01 | lstm-rank | 2639 | 8 | 6, 10, 6 | 244 |
| 2026-04-01 | lstm-reg | 2639 | 8 | 30, 35, 14 | 275 |
| 2026-04-01 | xgb-rank | 2639 | 8 | 171, 176, 146 | 52 |
| 2026-04-01 | xgb-reg | 2639 | 8 | 67, 56, 64 | 7 |
| 2026-07-01 | lstm-rank | 2730 | 8 | 17, 9, 7 | 328 |
| 2026-07-01 | lstm-reg | 2730 | 8 | 15, 12, 18 | 182 |
| 2026-07-01 | xgb-rank | 2730 | 8 | 182, 251, 92 | 55 |
| 2026-07-01 | xgb-reg | 2730 | 8 | 718, 778, 427 | 40 |

Runtime: 19.8 min in total (features 1 s, refits 19.4 min, 2555 day solves with HiGHS and SCIP 22 s).
