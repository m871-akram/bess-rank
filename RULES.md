# RULES.md: bess-rank

## The project in one paragraph
A battery trading on the German day-ahead market (bidding zone DE-LU) earns money by charging in the cheapest hours of a day and discharging in the most expensive ones. Price forecasts are usually trained and judged on RMSE, but the battery's decision depends mostly on the order of the hours within the day. We test whether models trained to rank the hours of each day earn more realized profit than models trained to predict prices, when both feed the same battery optimisation.

The design is a 2×2: XGBoost or bidirectional LSTM, each trained to predict prices or to rank hours. Each comparison is controlled: the ranked strategy keeps the price model's forecast values for the day and only reassigns them to hours in the ranker's order.

A risk layer reports VaR and Expected Shortfall of daily profit and a bank-style validation report. If time allows, it adds conformal quantile forecasts and a CVaR-constrained schedule. Data: SMARD (Bundesnetzagentur). Optimisation: `scipy.optimize.milp` and OR-Tools.

## Hard rules
1. **The test period is locked.** Delivery days from 2025-10-01 onward are the test set.
   - Code must refuse to load prices, labels or features for those days unless two things are true: `BESS_UNLOCK_TEST=1` is set, and `PREREGISTRATION.lock` exists. That file holds the commit hash of the pre-registration and is created only after I merge §6 and type "UNLOCK TEST".
   - Until then, never print, plot or summarise test-period prices. Data-quality checks on the test period output only counts and pass/fail results.
2. **No look-ahead.** The decision for delivery day D is taken at 11:00 Europe/Berlin on D-1.
   - Every feature declares its availability rule in `features.py`.
   - `tests/test_features.py` deletes all data after the cut-off, recomputes the features, and asserts identical values.
3. **Time zones.** Store timestamps in UTC and define delivery days in Europe/Berlin. Days have 23, 24 or 25 hours, so never assume 24 rows per day. Tests cover 2024-03-31 and 2024-10-27.
4. **Controlled comparisons.** Within each model family, only the within-day order may differ between the "-reg" and "-rank" strategies. Forecast values, battery model and settlement are shared (PLAN.md §5).
5. **Store raw numbers.** Never round stored tables; round only for display.
6. **Uncertainty.** Compute confidence intervals with a moving-block bootstrap over days (7-day blocks, 10,000 resamples), never by resampling hours.
7. **Sanity assertions.** These stop the run when violated:
   - on every day, perfect-foresight profit ≥ every other strategy's profit;
   - scipy (HiGHS) and OR-Tools (SCIP) give the same objective value on every day solved, within 1e-6 relative.
8. **Reproducibility.** Use fixed seeds. Record package versions, the data download time and the git commit in `results/provenance.json`.
9. **Data licensing.** Never commit raw or processed SMARD data; commit the download code. Credit "Bundesnetzagentur | SMARD.de" in the README.
10. **Git.** Work on branches and pull requests; never push to `main` or force-push.
11. **Secrets.** Never print, log or commit the Databricks token. Never write it to a file.
12. **Wording.** Results describe a simulated 1 MW / 2 MWh battery trading the day-ahead market only, as a price-taker. Never write "optimal", "real profit" or "deployed".

## Code style
- Write plain, readable code with short docstrings that say why. No clever abstractions.
- A comment where a modelling or market choice is made.
- One module per stage in `src/bessrank/`.
- Databricks notebooks in `notebooks/` are thin wrappers around the package.
