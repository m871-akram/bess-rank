# Rehearsal: 2024-10-01 to 2025-09-30 (365 delivery days)

Rehearsal on the validation year (exploratory): the §6 test pipeline with refits at the start of each validation quarter.

Simulated 1 MW / 2 MWh battery, day-ahead market only, price-taker. Rounded for display; the `rehearsal_*.csv` files next to this one hold the raw numbers.

## Hypotheses

Moving-block bootstrap over days: 7-day blocks, 10,000 resamples, seed 20261004 (same resampled days for every row). Margin = 1% of the reg strategy's RMSE over the period.

| hypothesis | role | comparison | statistic | estimate | ci95_low | ci95_high | margin | verdict |
|---|---|---|---|---|---|---|---|---|
| H1 | confirmatory | S-xgb-rank - S-xgb-reg | mean daily profit difference (EUR/day) | 2.851 | 1.186 | 4.58 |  | supported |
| H2 | secondary | S-xgb-rank - S-xgb-reg | RMSE difference over all hours (EUR/MWh) | -0.087 | -0.283 | 0.176 | 0.303 | supported |
| H3 | secondary | S-lstm-rank - S-lstm-reg | mean daily profit difference (EUR/day) | 1.458 | 0.323 | 2.687 |  | supported |
| LSTM RMSE | descriptive | S-lstm-rank - S-lstm-reg | RMSE difference over all hours (EUR/MWh) | -0.087 | -0.219 | 0.03 | 0.293 | descriptive (no test) |

| Hypothesis | Annual diff (EUR/MW/year) | Diff / reg profit | Days rank better | Days reg better | Days same vector |
|---|---|---|---|---|---|
| H1 | 1040.0 | 1.65% | 175.0 | 111.0 | 0.0 |
| H3 | 532.0 | 0.84% | 162.0 | 127.0 | 0.0 |

## Strategies

VaR 5% is the 5th percentile (P5) of daily profit and ES 5% the mean daily profit of the worst 5% of days. Both are profit levels: a negative value is a loss.

| strategy | Profit (EUR/MW/year) | Capture | RMSE (EUR/MWh) | MAE (EUR/MWh) | Spearman rho | Hit 2 cheapest | Hit 2 dearest | VaR 5%: P5 of daily profit (EUR) | ES 5%: mean of the worst 5% of days (EUR) | Losing days | Max drawdown (EUR) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S-perfect | 67519 | 100.0% | 0.0 | 0.0 | 1.0 | 100.0% | 100.0% | 35.722 | 25.836 | 0.0% | 0.0 |
| S-naive-1d | 55115 | 81.6% | 47.082 | 28.989 | 0.76 | 57.4% | 55.9% | -2.584 | -23.79 | 5.5% | 135.611 |
| S-naive-7d | 54106 | 80.1% | 60.445 | 35.98 | 0.781 | 54.0% | 54.2% | 0.655 | -19.933 | 4.9% | 93.676 |
| S-xgb-reg | 63195 | 93.6% | 30.285 | 16.998 | 0.927 | 71.1% | 75.1% | 25.519 | 13.482 | 0.5% | 9.919 |
| S-xgb-rank | 64236 | 95.1% | 30.198 | 16.919 | 0.945 | 75.9% | 77.5% | 27.068 | 18.189 | 0.3% | 8.376 |
| S-lstm-reg | 63654 | 94.3% | 29.294 | 16.669 | 0.935 | 71.5% | 76.6% | 28.471 | 15.353 | 0.5% | 9.876 |
| S-lstm-rank | 64186 | 95.1% | 29.207 | 16.55 | 0.948 | 76.3% | 77.4% | 26.126 | 15.057 | 0.3% | 11.107 |

## By quarter (descriptive)

| Quarter from | Strategy | Days | Profit (EUR/MW) | Capture | RMSE (EUR/MWh) | Spearman rho |
|---|---|---|---|---|---|---|
| 2024-10-01 | S-perfect | 92 | 14076 | 100.0% | 0.0 | 1.0 |
| 2024-10-01 | S-naive-1d | 92 | 10171 | 72.3% | 62.605 | 0.654 |
| 2024-10-01 | S-naive-7d | 92 | 9820 | 69.8% | 86.421 | 0.666 |
| 2024-10-01 | S-xgb-reg | 92 | 12702 | 90.2% | 41.715 | 0.895 |
| 2024-10-01 | S-xgb-rank | 92 | 12814 | 91.0% | 41.826 | 0.92 |
| 2024-10-01 | S-lstm-reg | 92 | 12738 | 90.5% | 41.588 | 0.905 |
| 2024-10-01 | S-lstm-rank | 92 | 12932 | 91.9% | 41.53 | 0.924 |
| 2025-01-01 | S-perfect | 90 | 12648 | 100.0% | 0.0 | 1.0 |
| 2025-01-01 | S-naive-1d | 90 | 8508 | 67.3% | 45.746 | 0.677 |
| 2025-01-01 | S-naive-7d | 90 | 7970 | 63.0% | 58.837 | 0.683 |
| 2025-01-01 | S-xgb-reg | 90 | 11416 | 90.3% | 30.713 | 0.915 |
| 2025-01-01 | S-xgb-rank | 90 | 11739 | 92.8% | 30.411 | 0.934 |
| 2025-01-01 | S-lstm-reg | 90 | 11700 | 92.5% | 29.153 | 0.927 |
| 2025-01-01 | S-lstm-rank | 90 | 11733 | 92.8% | 29.186 | 0.938 |
| 2025-04-01 | S-perfect | 91 | 22349 | 100.0% | 0.0 | 1.0 |
| 2025-04-01 | S-naive-1d | 91 | 20148 | 90.2% | 37.163 | 0.847 |
| 2025-04-01 | S-naive-7d | 91 | 20189 | 90.3% | 40.413 | 0.885 |
| 2025-04-01 | S-xgb-reg | 91 | 21548 | 96.4% | 24.133 | 0.953 |
| 2025-04-01 | S-xgb-rank | 91 | 21929 | 98.1% | 24.118 | 0.967 |
| 2025-04-01 | S-lstm-reg | 91 | 21820 | 97.6% | 20.261 | 0.96 |
| 2025-04-01 | S-lstm-rank | 91 | 22068 | 98.7% | 20.121 | 0.969 |
| 2025-07-01 | S-perfect | 92 | 18446 | 100.0% | 0.0 | 1.0 |
| 2025-07-01 | S-naive-1d | 92 | 16288 | 88.3% | 38.226 | 0.859 |
| 2025-07-01 | S-naive-7d | 92 | 16126 | 87.4% | 44.977 | 0.889 |
| 2025-07-01 | S-xgb-reg | 92 | 17529 | 95.0% | 19.987 | 0.946 |
| 2025-07-01 | S-xgb-rank | 92 | 17753 | 96.2% | 19.696 | 0.959 |
| 2025-07-01 | S-lstm-reg | 92 | 17396 | 94.3% | 20.907 | 0.949 |
| 2025-07-01 | S-lstm-rank | 92 | 17454 | 94.6% | 20.627 | 0.961 |

| Quarter from | Comparison | Days | Mean daily diff (EUR) | Days rank better | Days reg better | RMSE diff (EUR/MWh) |
|---|---|---|---|---|---|---|
| 2024-10-01 | S-xgb-rank - S-xgb-reg | 92 | 1.226 | 47 | 29 | 0.111 |
| 2024-10-01 | S-lstm-rank - S-lstm-reg | 92 | 2.108 | 45 | 31 | -0.058 |
| 2025-01-01 | S-xgb-rank - S-xgb-reg | 90 | 3.59 | 40 | 37 | -0.301 |
| 2025-01-01 | S-lstm-rank - S-lstm-reg | 90 | 0.369 | 41 | 40 | 0.033 |
| 2025-04-01 | S-xgb-rank - S-xgb-reg | 91 | 4.179 | 48 | 24 | -0.015 |
| 2025-04-01 | S-lstm-rank - S-lstm-reg | 91 | 2.724 | 43 | 24 | -0.14 |
| 2025-07-01 | S-xgb-rank - S-xgb-reg | 92 | 2.438 | 40 | 21 | -0.291 |
| 2025-07-01 | S-lstm-rank - S-lstm-reg | 92 | 0.622 | 33 | 32 | -0.28 |

## Refits

| Refit (quarter from) | Model | Training days | Days dropped (§2) | Trees/epochs per seed | Fit time, 3 seeds (s) |
|---|---|---|---|---|---|
| 2024-10-01 | lstm-rank | 2092 | 8 | 4, 4, 6 | 155 |
| 2024-10-01 | lstm-reg | 2092 | 8 | 14, 18, 11 | 128 |
| 2024-10-01 | xgb-rank | 2092 | 8 | 83, 88, 151 | 35 |
| 2024-10-01 | xgb-reg | 2092 | 8 | 467, 223, 571 | 25 |
| 2025-01-01 | lstm-rank | 2184 | 8 | 5, 3, 8 | 169 |
| 2025-01-01 | lstm-reg | 2184 | 8 | 7, 16, 11 | 119 |
| 2025-01-01 | xgb-rank | 2184 | 8 | 135, 97, 73 | 33 |
| 2025-01-01 | xgb-reg | 2184 | 8 | 71, 750, 48 | 18 |
| 2025-04-01 | lstm-rank | 2274 | 8 | 6, 5, 15 | 225 |
| 2025-04-01 | lstm-reg | 2274 | 8 | 19, 28, 17 | 191 |
| 2025-04-01 | xgb-rank | 2274 | 8 | 332, 152, 126 | 56 |
| 2025-04-01 | xgb-reg | 2274 | 8 | 44, 40, 50 | 6 |
| 2025-07-01 | lstm-rank | 2365 | 8 | 8, 10, 9 | 239 |
| 2025-07-01 | lstm-reg | 2365 | 8 | 8, 17, 8 | 116 |
| 2025-07-01 | xgb-rank | 2365 | 8 | 132, 381, 13 | 51 |
| 2025-07-01 | xgb-reg | 2365 | 8 | 195, 293, 252 | 18 |

Runtime: 14.1 min in total (features 1 s, refits 13.7 min, 2555 day solves with HiGHS and SCIP 22 s).
