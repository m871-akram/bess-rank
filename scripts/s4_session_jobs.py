import time
from datetime import datetime
from bessrank import explore, databricks, config
def log(msg): print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)
t = time.time()
explore.regenerate_forecasts(log=log)
log(f"regeneration done in {(time.time()-t)/60:.1f} min")
ok = databricks.upload_file(explore.FORECASTS_PARQUET, "predictions/test_forecasts_regenerated.parquet")
log(f"uploaded forecasts to volume: {ok}")
t = time.time()
explore.run_xgb_variants(log=log)
log(f"variants done in {(time.time()-t)/60:.1f} min")
ok = databricks.upload_file(explore.VARIANTS_PARQUET, "predictions/test_xgb_variants.parquet")
log(f"uploaded variants to volume: {ok}")
