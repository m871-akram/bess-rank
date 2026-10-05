# Session scripts

Scripts that produced committed results outside `python -m bessrank.run`. They are kept verbatim,
so their sha256 matches the one recorded in `results/provenance.json`.

| Script | What it ran | Equivalent command today |
|---|---|---|
| `s4_session_jobs.py` | S4 (2026-10-05): `explore.regenerate_forecasts`, then `explore.run_xgb_variants`, each followed by an upload of its forecasts to the Databricks volume (`predictions/`). Run from the repository root with `BESS_UNLOCK_TEST=1`. | `python -m bessrank.run explore forecasts` and `python -m bessrank.run explore xgb-variants` (same functions; since S5 the upload is automatic) |
