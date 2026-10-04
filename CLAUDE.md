# CLAUDE.md: bess-rank

Read this file and PLAN.md in full at the start of every session. PLAN.md is the source of truth for the design, the 3-day schedule (§11) and the lab notebook (§12).

## The project in one paragraph
A battery trading on the German day-ahead market (bidding zone DE-LU) earns money by charging in the cheapest hours of a day and discharging in the most expensive ones. Price forecasts are usually trained and judged on RMSE, but the battery's decision depends mostly on the order of the hours within the day. We test whether models trained to rank the hours of each day earn more realized profit than models trained to predict prices, when both feed the same battery optimisation.

The design is a 2×2: XGBoost or bidirectional LSTM, each trained to predict prices or to rank hours. Each comparison is controlled: the ranked strategy keeps the price model's forecast values for the day and only reassigns them to hours in the ranker's order.

A risk layer reports VaR and Expected Shortfall of daily profit and a bank-style validation report. If time allows, it adds conformal quantile forecasts and a CVaR-constrained schedule. Data: SMARD (Bundesnetzagentur). Optimisation: `scipy.optimize.milp` and OR-Tools.

## Roles
- Akram owns every decision. You implement, test, run and report.
- Stop and ask Akram before you:
  - unlock the test period;
  - change PLAN.md §6 once the pre-registration is committed;
  - delete results;
  - create anything outside the Databricks Free Edition workspace, or spend money.
- When a choice isn't covered by PLAN.md, give 1–2 options with your recommendation in the report. Don't pick silently.
- The schedule is 3 days. When you fall behind, follow the cut list in PLAN.md §11 instead of rushing a step.

## Hard rules
1. **The test period is locked.** Delivery days from 2025-10-01 onward are the test set.
   - Code must refuse to load prices, labels or features for those days unless two things are true: `BESS_UNLOCK_TEST=1` is set, and `PREREGISTRATION.lock` exists. That file holds the commit hash of the pre-registration and is created only after Akram merges §6 and types "UNLOCK TEST".
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
10. **Git and authorship.** Akram must be the only author.
    - Before your first commit in a session, run `git var GIT_AUTHOR_IDENT` and `git var GIT_COMMITTER_IDENT`. Both must show Akram. If either shows "Claude", stop and tell him to set the `GIT_*` environment variables in the cloud environment.
    - Never add Co-Authored-By trailers, session links or "Generated with" lines (`.claude/settings.json` disables them).
    - Work on a branch, push, and open a pull request for Akram to merge. Never push to `main` or force-push.
11. **Secrets.** Never print, log or commit the Databricks token. Never write it to a file.
12. **Wording.** Results describe a simulated 1 MW / 2 MWh battery trading the day-ahead market only, as a price-taker. Never write "optimal", "real profit" or "deployed".

## Environment (Claude Code cloud session)
- **The machine:** a fresh Ubuntu x86_64 VM per session, with 4 vCPUs, 16 GB RAM and 30 GB disk. Uncommitted files are lost when the VM is reclaimed.
  - So `python -m bessrank.run data` must rebuild `data/` from scratch. It first tries to pull the processed parquet from the Databricks volume; otherwise it re-downloads from SMARD.
- **Network:** an allowlist set by Akram: package registries, `www.smard.de`, `download.pytorch.org` and his Databricks host. If something is blocked, report it rather than looking for a workaround.
- **Databricks Free Edition:**
  - Serverless compute only, no outbound internet except trusted domains, and daily quotas.
  - Call its REST API from the session with `requests` on `DATABRICKS_HOST`. The token is attached by the session's proxy, or comes from `DATABRICKS_TOKEN` if Akram set it.
  - Keep Databricks runs small; do the hyperparameter searches in the VM.
- **Packages:**
  - pandas, numpy, xgboost (3.x), scipy (`milp`/HiGHS), ortools (SCIP), torch (CPU wheel), scikit-learn;
  - holidays, requests, pyarrow, matplotlib, pytest;
  - mlflow (for Databricks).
- **Long jobs:** run them in the background with a log file and check progress, rather than blocking.
- **Budget:** about $100 of Claude cloud usage for the whole project. Avoid re-reading large files, print summaries rather than tables, and don't loop on failing commands. After two failures, stop and report.

## Code style
- Akram must be able to explain every line in an interview; he knows pandas, scikit-learn and PyTorch.
  - Write plain, readable code with short docstrings that say why. No clever abstractions.
  - A comment where a modelling or market choice is made.
- One module per stage in `src/bessrank/`.
- Databricks notebooks in `notebooks/` are thin wrappers around the package.

## Reporting
End every session with a report of about 15 lines at most:
- what ran;
- numbers, with units and the split they come from (train / validation / test);
- problems or surprises;
- decisions needed from Akram, numbered, each with your recommendation;
- the PR link;
- where you are against the 3-day schedule.

Add a dated entry to PLAN.md §12 for anything that changes numbers or decisions. Never rewrite past entries; add corrections as new entries.
