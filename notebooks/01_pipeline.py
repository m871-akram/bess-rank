# Databricks notebook source
# MAGIC %md
# MAGIC # bess-rank: official XGBoost test pipeline (PLAN.md §9)
# MAGIC Thin wrapper around `bessrank.pipeline.run_databricks`. It reruns the pre-registered test
# MAGIC steps for the XGBoost pair on serverless: quarterly refits with the frozen hyperparameters,
# MAGIC the five non-LSTM strategies, every day solved by HiGHS and re-checked by SCIP. It writes
# MAGIC the `workspace.bess.gold_*` Delta tables, logs every fit to the MLflow experiment
# MAGIC `bess-rank`, and checks parity with the S3 VM run.
# MAGIC
# MAGIC The test year is locked. The job passes `BESS_UNLOCK_TEST=1` and the code bundle carries
# MAGIC `PREREGISTRATION.lock`; without both, loading the data stops. The bundle is a `git archive`
# MAGIC of one commit, uploaded to the volume by `python -m bessrank.run pipeline` (no Git
# MAGIC credentials in Databricks).

# COMMAND ----------

# MAGIC %pip install --quiet xgboost==3.2.0 scipy==1.17.1 ortools==9.15.6755 numpy==2.4.6 pandas==3.0.6 holidays==0.105

# COMMAND ----------

import json
import os
import shutil
import sys
import tarfile

dbutils.widgets.text("bundle", "")
dbutils.widgets.text("BESS_UNLOCK_TEST", "0")
VOLUME = "/Volumes/workspace/bess/raw"
ROOT = "/tmp/bess-rank"  # the bundle's root: src/, PREREGISTRATION.lock, results/selected_*.json

shutil.rmtree(ROOT, ignore_errors=True)
with tarfile.open(f"{VOLUME}/{dbutils.widgets.get('bundle')}") as tar:
    tar.extractall(ROOT, filter="data")
os.makedirs(f"{ROOT}/data/processed", exist_ok=True)
shutil.copy(f"{VOLUME}/processed/hourly.parquet", f"{ROOT}/data/processed/hourly.parquet")
sys.path.insert(0, f"{ROOT}/src")
commit = open(f"{ROOT}/COMMIT").read().strip()

# One half of the test lock; PREREGISTRATION.lock in the bundle is the other.
os.environ["BESS_UNLOCK_TEST"] = dbutils.widgets.get("BESS_UNLOCK_TEST")
spark.conf.set("spark.sql.session.timeZone", "UTC")

# COMMAND ----------

from bessrank import pipeline

user = spark.sql("SELECT current_user()").first()[0]
summary = pipeline.run_databricks(spark, commit, experiment=f"/Users/{user}/bess-rank", volume=VOLUME)

# COMMAND ----------

dbutils.notebook.exit(json.dumps(summary, default=str))
