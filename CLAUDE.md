# CLAUDE.md: bess-rank

Read this file and PLAN.md in full at the start of every session. PLAN.md is the source of truth for the study design and holds the lab notebook (§12).

## The project in one paragraph
A battery trading on the German day-ahead market (bidding zone DE-LU) earns money by charging in the cheapest hours of a day and discharging in the most expensive ones. Price forecasts are usually trained and judged on RMSE, but the battery's decision depends mostly on the order of the hours within the day. We test whether XGBoost trained to rank the hours of each day (learning-to-rank, one query group per delivery day) earns more realized profit than XGBoost trained to predict prices (squared error), when both feed the same battery optimisation. The comparison is controlled: the ranked strategy keeps the regressor's forecast values for the day and only reassigns them to hours in the ranker's order. Data: SMARD (Bundesnetzagentur). Compute: the Mac for development and tests; Databricks Free Edition for the official pipeline (Delta tables, MLflow, one job, one dashboard).

## Roles
- Akram owns every decision. You implement, test, run and report.
- Stop and ask Akram before you:
  - unlock the test period;
  - change PLAN.md §6 once the pre-registration is committed;
  - delete data or results;
  - install system-wide software;
  - create anything outside the Databricks Free Edition workspace, or spend money.
- When a choice isn't covered by PLAN.md, give 1–2 options with your recommendation in the report. Don't pick silently.

## Hard rules
1. **The test period is locked.** Delivery days from 2025-10-01 onward are the test set.
   - Code must refuse to load prices, labels or features for those days unless two things are true: the environment variable `BESS_UNLOCK_TEST=1` is set, and a file `PREREGISTRATION.lock` exists. That file holds the commit hash of the pre-registration and is created only after Akram commits §6 and types "UNLOCK TEST".
   - Until then, never print, plot or summarise test-period prices. Data-quality checks on the test period may output only counts and pass/fail results.
2. **No look-ahead.** The decision for delivery day D is taken at 11:00 Europe/Berlin on D-1, one hour before the 12:00 auction gate closure.
   - Every feature must be computable from data available at that time, and declares its availability rule in `features.py`.
   - `tests/test_features.py` checks this: recompute the features with all data after the cut-off deleted, and assert the values are identical.
3. **Time zones.** Store timestamps in UTC and define delivery days in Europe/Berlin. Daylight-saving days have 23 or 25 hours, so never assume 24 rows per day. Tests cover 2024-03-31 and 2024-10-27.
4. **One controlled comparison.** In the primary comparison, only the within-day order may differ between the two strategies. Forecast values, battery model and settlement are shared (PLAN.md §4–§6).
5. **Store raw numbers.** Never round stored tables or results; round only for display.
6. **Uncertainty.** Compute confidence intervals with a moving-block bootstrap over days (7-day blocks, 10,000 resamples), never by resampling hours.
7. **Sanity assertions.** For every day, perfect-foresight profit must be at least the profit of every other strategy. A violation stops the run.
8. **Reproducibility.** Use fixed seeds. Record package versions, the data download date and the git commit in `results/provenance.json`.
9. **Data licensing.** Never commit raw or processed SMARD data; commit the download script. Credit "Bundesnetzagentur | SMARD.de" in the README.
10. **Attribution.** Never add Co-Authored-By trailers, "Generated with Claude Code" lines or similar to commits, PRs or files; Akram is the only author. At the end of a session, give Akram the exact `git add` / `git commit` commands. He runs them.
11. **Wording.** Results describe a simulated 1 MW / 2 MWh battery trading the day-ahead market only, as a price-taker. Never write "optimal" (time-limited or approximate solves), "real profit" or "deployed".

## Environment
- **Local:** macOS (Apple M1), Python 3.11+ in `.venv`, `pip install -r requirements.txt`.
- **Databricks Free Edition:**
  - Serverless compute only.
  - Outbound internet is limited to a few trusted domains, so download data locally and upload it to a Unity Catalog volume.
  - Daily compute quotas apply; going over shuts compute down for the rest of the day. Keep runs small and do the hyperparameter searches locally.
- **Packages:**
  - pandas, numpy, xgboost (2.x or later), scipy (`scipy.optimize.milp` with HiGHS for the battery), scikit-learn;
  - holidays, requests, pyarrow, matplotlib, pytest;
  - mlflow (Databricks), shap (optional).
- **Budget:** about $100 of Claude usage for the whole project. Avoid re-reading large files, print summaries rather than full tables, and run long jobs in the background with logs.

## Code style
- Akram reads every file and knows pandas, scikit-learn and PyTorch. Write plain, readable code with short docstrings that say why. No clever abstractions.
- One module per stage in `src/bessrank/`.
- Databricks notebooks in `notebooks/` are thin wrappers: they call the package and read or write Delta tables.
- Analysis notebooks read only `results/`.

## Reporting
End every session with a report of about 15 lines at most:
- what ran, and where (Mac or Databricks);
- numbers, with units and the split they come from (train / validation / test);
- problems or surprises;
- decisions needed from Akram, numbered, each with your recommendation;
- the git commands to run.

Add a dated entry to PLAN.md §12 for anything that changes numbers or decisions. Never rewrite past entries; add corrections as new entries.
