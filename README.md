# bess-rank: rank, don't forecast

*A battery doesn't need the price, it needs the order.*

A simulated 1 MW / 2 MWh battery trades the German day-ahead market (bidding zone DE-LU) as a
price-taker: it charges in the cheapest hours of each delivery day and discharges in the most
expensive ones. Price forecasts are usually trained and judged on RMSE, but the battery's
schedule depends mostly on the order of the hours within the day. This project tests whether
models trained to rank the hours of each day (XGBoost `rank:pairwise`, a bidirectional LSTM with
a pairwise loss) earn more simulated profit than the same models trained to predict prices, when
both feed the same daily battery optimisation. Each comparison keeps the price model's forecast
values and only reassigns them to hours in the ranker's order, so any profit difference comes
from the order alone. The hypotheses are pre-registered before a locked test year
(2025-10-01 to 2026-09-30) is opened. See [PLAN.md](PLAN.md) for the full design.

Status: work in progress (3-day sprint). Results appear here after the pre-registered test.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m bessrank.run data   # rebuild data/ from the Databricks volume or SMARD
python -m bessrank.run qa     # write results/qa_data.md
pytest
```

## Data

Day-ahead prices, TSO day-ahead forecasts (load, wind, solar) and actual load for DE-LU, from the
SMARD chart API. The data are not stored in this repository; `python -m bessrank.run data`
downloads them.

Data source: **Bundesnetzagentur | SMARD.de**
