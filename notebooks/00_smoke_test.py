# Databricks notebook source
# MAGIC %md
# MAGIC # S0 smoke test
# MAGIC Checks the Databricks path end to end: read the processed hourly table from the volume,
# MAGIC train a small XGBoost on **training-period days only**, log the run to MLflow and write a
# MAGIC Delta table. Test-period rows (delivery days from 2025-10-01) are dropped right after
# MAGIC loading and never used or displayed (RULES.md rule 1).

# COMMAND ----------

# MAGIC %pip install --quiet xgboost

# COMMAND ----------

import json
from datetime import date

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb

VOLUME = "/Volumes/workspace/bess/raw"
TEST_START = date(2025, 10, 1)

table = pd.read_parquet(f"{VOLUME}/processed/hourly.parquet")
table = table[table["delivery_day"] < TEST_START]  # test period stays locked

# Small, fast split inside the training period: fit on 9 months, score the next 3.
fit = table[(table["delivery_day"] >= date(2023, 10, 1)) & (table["delivery_day"] <= date(2024, 6, 30))].dropna()
holdout = table[(table["delivery_day"] >= date(2024, 7, 1)) & (table["delivery_day"] <= date(2024, 9, 30))].dropna()
features = ["hour", "load_fc", "wind_onshore_fc", "wind_offshore_fc", "pv_fc"]

params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1, "random_state": 0}
model = xgb.XGBRegressor(**params)
model.fit(fit[features], fit["price"])
holdout = holdout.assign(prediction=model.predict(holdout[features]))
rmse = float(np.sqrt(((holdout["prediction"] - holdout["price"]) ** 2).mean()))

# COMMAND ----------

user = spark.sql("SELECT current_user()").first()[0]
# Serverless blocks reading spark.mlflow.modelRegistryUri, which MLflow does when no registry
# URI is set; setting it explicitly avoids that lookup.
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Users/{user}/bess-rank")
with mlflow.start_run(run_name="s0-smoke-test"):
    mlflow.log_params(params)
    mlflow.log_param("features", ",".join(features))
    mlflow.log_metric("rmse_holdout_eur_mwh", rmse)

out = holdout[["delivery_day", "hour", "price", "prediction"]]
spark.createDataFrame(out).write.mode("overwrite").saveAsTable("workspace.bess.smoke_predictions")

# COMMAND ----------

dbutils.notebook.exit(json.dumps({
    "rows_fit": len(fit),
    "rows_holdout": len(holdout),
    "rmse_holdout_eur_mwh": rmse,
    "xgboost": xgb.__version__,
    "delta_table": "workspace.bess.smoke_predictions",
}))
