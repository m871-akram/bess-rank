# Validation year 2024-10-01 to 2025-09-30 (365 delivery days)

Simulated 1 MW / 2 MWh battery, day-ahead market only, price-taker. Rounded for display; the CSV files next to this one hold the raw numbers.

VaR 5% is the 5th percentile (P5) of daily profit and ES 5% the mean daily profit of the worst 5% of days (19 of 365). Both are profit levels: a negative value is a loss.

## Strategies

| strategy | Profit (EUR/MW/year) | Capture | RMSE (EUR/MWh) | MAE (EUR/MWh) | Spearman rho | Hit 2 cheapest | Hit 2 dearest | VaR 5%: P5 of daily profit (EUR) | ES 5%: mean of the worst 5% of days (EUR) | Losing days | Max drawdown (EUR) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S-perfect | 67519 | 100.0% | 0.0 | 0.0 | 1.0 | 100.0% | 100.0% | 35.722 | 25.836 | 0.0% | 0.0 |
| S-naive-1d | 55115 | 81.6% | 47.082 | 28.989 | 0.76 | 57.4% | 55.9% | -2.584 | -23.79 | 5.5% | 135.611 |
| S-naive-7d | 54106 | 80.1% | 60.445 | 35.98 | 0.781 | 54.0% | 54.2% | 0.655 | -19.933 | 4.9% | 93.676 |
| S-xgb-reg | 63470 | 94.0% | 30.641 | 17.59 | 0.928 | 71.1% | 75.2% | 26.671 | 13.769 | 0.5% | 18.121 |
| S-xgb-rank | 64223 | 95.1% | 30.59 | 17.524 | 0.945 | 74.5% | 77.4% | 27.068 | 18.218 | 0.3% | 8.424 |
| S-lstm-reg | 63771 | 94.4% | 29.38 | 17.117 | 0.934 | 69.5% | 75.2% | 26.815 | 14.893 | 0.5% | 11.107 |
| S-lstm-rank | 64399 | 95.4% | 29.267 | 16.972 | 0.948 | 76.8% | 76.6% | 29.701 | 15.903 | 0.3% | 8.376 |

## Paired daily differences, rank minus reg (exploratory, validation only)

Moving-block bootstrap over days: 7-day blocks, 10,000 resamples, seed 20261004.

| Comparison | Mean daily diff (EUR) | 95% CI low | 95% CI high | Annual diff (EUR/MW) | Days rank better | Days reg better | Days same vector | RMSE diff (EUR/MWh) | RMSE CI low | RMSE CI high |
|---|---|---|---|---|---|---|---|---|---|---|
| S-xgb-rank - S-xgb-reg | 2.06 | 0.65 | 3.45 | 752.96 | 165 | 114 | 0 | -0.05 | -0.22 | 0.2 |
| S-lstm-rank - S-lstm-reg | 1.72 | 0.71 | 2.87 | 627.86 | 170 | 118 | 0 | -0.11 | -0.23 | -0.01 |

## Every configuration (PLAN.md §7, used in §8)

Seed `avg` is the 3-seed average. For rankers, the RMSE and the profit are those of the "-rank" price vector (the selected price model's values in the ranker's order); the Spearman rho is that of the model's own output (scores for rankers).

| Model | Config | Seed | max_depth | learning_rate | hidden | layers | Trees/epochs | RMSE of price vector (EUR/MWh) | Spearman rho (model output) | Validation profit (EUR/MW/year) | Selected |
|---|---|---|---|---|---|---|---|---|---|---|---|
| xgb-reg | 0 | 0 | 9.0 | 0.0224 |  |  | 491.0 | 31.12 | 0.9331 | 63357 | False |
| xgb-reg | 1 | 0 | 8.0 | 0.0616 |  |  | 313.0 | 31.23 | 0.9238 | 62658 | False |
| xgb-reg | 2 | 0 | 5.0 | 0.0101 |  |  | 755.0 | 31.3 | 0.9298 | 63233 | False |
| xgb-reg | 3 | 0 | 9.0 | 0.1327 |  |  | 29.0 | 31.27 | 0.9107 | 61694 | False |
| xgb-reg | 4 | 0 | 6.0 | 0.0145 |  |  | 455.0 | 30.96 | 0.9258 | 63251 | False |
| xgb-reg | 5 | 0 | 9.0 | 0.1983 |  |  | 51.0 | 31.96 | 0.898 | 60752 | False |
| xgb-reg | 6 | 0 | 9.0 | 0.0321 |  |  | 325.0 | 30.95 | 0.9223 | 62919 | False |
| xgb-reg | 7 | 0 | 6.0 | 0.0429 |  |  | 387.0 | 30.81 | 0.9235 | 63052 | False |
| xgb-reg | 8 | 0 | 8.0 | 0.0262 |  |  | 502.0 | 30.65 | 0.9279 | 63211 | False |
| xgb-reg | 9 | 0 | 5.0 | 0.0197 |  |  | 649.0 | 31.17 | 0.9294 | 63221 | False |
| xgb-reg | 10 | 0 | 6.0 | 0.0205 |  |  | 736.0 | 30.84 | 0.9297 | 63290 | False |
| xgb-reg | 11 | 0 | 7.0 | 0.0385 |  |  | 467.0 | 30.59 | 0.9251 | 63256 | False |
| xgb-reg | 12 | 0 | 7.0 | 0.0181 |  |  | 450.0 | 30.93 | 0.9259 | 63328 | False |
| xgb-reg | 13 | 0 | 10.0 | 0.0182 |  |  | 656.0 | 30.92 | 0.9284 | 63217 | False |
| xgb-reg | 14 | 0 | 7.0 | 0.1608 |  |  | 70.0 | 32.04 | 0.9132 | 61061 | False |
| xgb-reg | 11 | 1 | 7.0 | 0.0385 |  |  | 223.0 | 30.69 | 0.9253 | 63216 | False |
| xgb-reg | 11 | 2 | 7.0 | 0.0385 |  |  | 571.0 | 31.09 | 0.9283 | 63523 | False |
| xgb-reg | 11 | avg | 7.0 | 0.0385 |  |  |  | 30.64 | 0.9282 | 63470 | True |
| xgb-reg | 8 | 1 | 8.0 | 0.0262 |  |  | 337.0 | 31.15 | 0.9264 | 63158 | False |
| xgb-reg | 8 | 2 | 8.0 | 0.0262 |  |  | 396.0 | 30.98 | 0.9254 | 63135 | False |
| xgb-reg | 8 | avg | 8.0 | 0.0262 |  |  |  | 30.85 | 0.928 | 63273 | False |
| xgb-reg | 7 | 1 | 6.0 | 0.0429 |  |  | 155.0 | 31.02 | 0.925 | 63105 | False |
| xgb-reg | 7 | 2 | 6.0 | 0.0429 |  |  | 151.0 | 31.07 | 0.9237 | 63229 | False |
| xgb-reg | 7 | avg | 6.0 | 0.0429 |  |  |  | 30.85 | 0.9259 | 63274 | False |
| xgb-rank | 0 | 0 | 9.0 | 0.0224 |  |  | 72.0 | 30.63 | 0.9412 | 64096 | False |
| xgb-rank | 1 | 0 | 8.0 | 0.0616 |  |  | 74.0 | 30.63 | 0.9415 | 64104 | False |
| xgb-rank | 2 | 0 | 5.0 | 0.0101 |  |  | 87.0 | 30.7 | 0.938 | 64050 | False |
| xgb-rank | 3 | 0 | 9.0 | 0.1327 |  |  | 83.0 | 30.61 | 0.943 | 64207 | False |
| xgb-rank | 4 | 0 | 6.0 | 0.0145 |  |  | 353.0 | 30.64 | 0.9397 | 64146 | False |
| xgb-rank | 5 | 0 | 9.0 | 0.1983 |  |  | 202.0 | 30.6 | 0.9431 | 64178 | False |
| xgb-rank | 6 | 0 | 9.0 | 0.0321 |  |  | 284.0 | 30.61 | 0.9433 | 64091 | False |
| xgb-rank | 7 | 0 | 6.0 | 0.0429 |  |  | 359.0 | 30.64 | 0.9425 | 64175 | False |
| xgb-rank | 8 | 0 | 8.0 | 0.0262 |  |  | 138.0 | 30.62 | 0.9419 | 64081 | False |
| xgb-rank | 9 | 0 | 5.0 | 0.0197 |  |  | 430.0 | 30.61 | 0.9407 | 64188 | False |
| xgb-rank | 10 | 0 | 6.0 | 0.0205 |  |  | 114.0 | 30.65 | 0.9394 | 64066 | False |
| xgb-rank | 11 | 0 | 7.0 | 0.0385 |  |  | 75.0 | 30.61 | 0.9405 | 64233 | False |
| xgb-rank | 12 | 0 | 7.0 | 0.0181 |  |  | 330.0 | 30.62 | 0.9417 | 64233 | False |
| xgb-rank | 13 | 0 | 10.0 | 0.0182 |  |  | 134.0 | 30.61 | 0.9421 | 64122 | False |
| xgb-rank | 14 | 0 | 7.0 | 0.1608 |  |  | 64.0 | 30.64 | 0.9428 | 64008 | False |
| xgb-rank | 6 | 1 | 9.0 | 0.0321 |  |  | 194.0 | 30.63 | 0.943 | 64170 | False |
| xgb-rank | 6 | 2 | 9.0 | 0.0321 |  |  | 270.0 | 30.62 | 0.9433 | 64125 | False |
| xgb-rank | 6 | avg | 9.0 | 0.0321 |  |  |  | 30.62 | 0.9434 | 64074 | False |
| xgb-rank | 5 | 1 | 9.0 | 0.1983 |  |  | 148.0 | 30.59 | 0.9439 | 64354 | False |
| xgb-rank | 5 | 2 | 9.0 | 0.1983 |  |  | 65.0 | 30.61 | 0.9431 | 64106 | False |
| xgb-rank | 5 | avg | 9.0 | 0.1983 |  |  |  | 30.6 | 0.944 | 64291 | False |
| xgb-rank | 3 | 1 | 9.0 | 0.1327 |  |  | 88.0 | 30.6 | 0.9438 | 64093 | False |
| xgb-rank | 3 | 2 | 9.0 | 0.1327 |  |  | 151.0 | 30.58 | 0.9441 | 64072 | False |
| xgb-rank | 3 | avg | 9.0 | 0.1327 |  |  |  | 30.59 | 0.9445 | 64223 | True |
| lstm-reg | 0 | 0 |  |  | 32.0 | 1.0 | 22.0 | 29.84 | 0.9133 | 62989 | False |
| lstm-reg | 1 | 0 |  |  | 32.0 | 2.0 | 13.0 | 31.34 | 0.9149 | 62755 | False |
| lstm-reg | 2 | 0 |  |  | 64.0 | 1.0 | 14.0 | 30.46 | 0.9228 | 63211 | False |
| lstm-reg | 3 | 0 |  |  | 64.0 | 2.0 | 15.0 | 30.32 | 0.9275 | 63244 | False |
| lstm-reg | 4 | 0 |  |  | 128.0 | 1.0 | 14.0 | 29.71 | 0.9286 | 63445 | False |
| lstm-reg | 5 | 0 |  |  | 128.0 | 2.0 | 15.0 | 30.85 | 0.9256 | 63488 | False |
| lstm-reg | 4 | 1 |  |  | 128.0 | 1.0 | 18.0 | 30.19 | 0.9262 | 63045 | False |
| lstm-reg | 4 | 2 |  |  | 128.0 | 1.0 | 11.0 | 30.15 | 0.9317 | 63200 | False |
| lstm-reg | 4 | avg |  |  | 128.0 | 1.0 |  | 29.38 | 0.9343 | 63771 | True |
| lstm-rank | 0 | 0 |  |  | 32.0 | 1.0 | 21.0 | 29.26 | 0.9456 | 64479 | False |
| lstm-rank | 1 | 0 |  |  | 32.0 | 2.0 | 14.0 | 29.35 | 0.9443 | 64152 | False |
| lstm-rank | 2 | 0 |  |  | 64.0 | 1.0 | 15.0 | 29.33 | 0.9454 | 63916 | False |
| lstm-rank | 3 | 0 |  |  | 64.0 | 2.0 | 13.0 | 29.37 | 0.9429 | 63827 | False |
| lstm-rank | 4 | 0 |  |  | 128.0 | 1.0 | 10.0 | 29.26 | 0.9458 | 64248 | False |
| lstm-rank | 5 | 0 |  |  | 128.0 | 2.0 | 4.0 | 29.3 | 0.9466 | 64285 | False |
| lstm-rank | 5 | 1 |  |  | 128.0 | 2.0 | 4.0 | 29.31 | 0.9464 | 64090 | False |
| lstm-rank | 5 | 2 |  |  | 128.0 | 2.0 | 6.0 | 29.31 | 0.9466 | 64354 | False |
| lstm-rank | 5 | avg |  |  | 128.0 | 2.0 |  | 29.27 | 0.9479 | 64399 | True |
