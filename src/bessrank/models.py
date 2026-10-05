"""XGBoost price and ranking models, the random search and seed averaging (PLAN.md §4, §7).

Every configuration and seed is fitted the same way (PLAN.md §7):
1. fit on the training window without its last 3 months, early-stopping on those 3 months;
2. refit on the full window with the number of trees (or epochs) found in step 1;
3. predict the days to forecast (the validation year in S2).

The search driver `search` is shared with the LSTM (lstm.fit_predict has the same signature).
"""
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from bessrank import config, evaluate, features

N_CONFIGS = 15  # random-search size per XGBoost model (PLAN.md §7)
SEARCH_SEED = 0  # draws the configurations; the models of the search use seed 0 too
ES_MONTHS = 3  # early stopping on the last 3 months of the training window
MAX_TREES = 5000
EARLY_STOPPING_ROUNDS = 50  # trees without improvement on the early-stopping months
N_THREADS = 2  # two searches run side by side on the 4-vCPU VM
KEYS = ["ts_utc", "delivery_day", "hour", "price"]

FIXED_PARAMS = {"tree_method": "hist"}
OBJECTIVES = {
    "xgb-reg": {"objective": "reg:squarederror", "eval_metric": "rmse"},
    # Pairwise logistic loss over every pair of hours of a day with different labels,
    # sum of log(1 + exp(-(s_i - s_j))): the LSTM-rank loss. In XGBoost 3.x this needs
    # pair_method "topk" with k >= 25 (every hour is in the "top k", so all pairs are built;
    # "mean" samples pairs at random) and both normalisations off (tests/test_models.py
    # checks the gradient against a numpy reference).
    "xgb-rank": {"objective": "rank:pairwise", "lambdarank_pair_method": "topk",
                 "lambdarank_num_pair_per_sample": 25, "lambdarank_normalization": False,
                 "lambdarank_score_normalization": False, "disable_default_eval_metric": 1},
}


# --- Rows used for fitting and forecasting --------------------------------------------------
def es_start(last_train_day):
    """First day of the early-stopping period: the last 3 months of the training window."""
    start = pd.Timestamp(last_train_day) - pd.DateOffset(months=ES_MONTHS) + pd.Timedelta(days=1)
    return start.date()


def training_rows(feats, first_day, last_day):
    """Complete days in [first_day, last_day], sorted by time (so by day, then hour).

    Days with a missing feature or label are dropped from training (PLAN.md §2).
    """
    config.check_days_allowed([last_day])
    dropped = features.incomplete_days(feats, first_day, last_day)
    day = feats["delivery_day"]
    keep = (day >= first_day) & (day <= last_day) & ~day.isin(dropped)
    return feats[keep].sort_values("ts_utc").reset_index(drop=True)


def forecast_rows(feats, first_day, last_day):
    """Every day in [first_day, last_day]; forecast days are never dropped (PLAN.md §2)."""
    config.check_days_allowed([last_day])
    day = feats["delivery_day"]
    return feats[(day >= first_day) & (day <= last_day)].sort_values("ts_utc").reset_index(drop=True)


def split_early_stopping(train):
    """(inner, es): the training window without and with only its last 3 months."""
    first_es_day = es_start(train["delivery_day"].max())
    inner = train[train["delivery_day"] < first_es_day].reset_index(drop=True)
    es = train[train["delivery_day"] >= first_es_day].reset_index(drop=True)
    return inner, es


def rank_labels(frame):
    """Dense within-day rank of the actual price: 0 for the cheapest hour, ties share a label."""
    return (frame.groupby("delivery_day")["price"].rank(method="dense") - 1).astype(int).to_numpy()


# --- XGBoost ----------------------------------------------------------------------------------
def _dmatrix(kind, frame):
    X = frame[features.FEATURE_COLUMNS]
    if kind == "xgb-reg":
        return xgb.DMatrix(X, label=frame["price"].to_numpy())
    # One query group per delivery day; rows are sorted by day, so qid is non-decreasing.
    qid = pd.factorize(frame["delivery_day"])[0]
    return xgb.DMatrix(X, label=rank_labels(frame), qid=qid)


def _spearman_metric(frame):
    """Early-stopping metric for the ranker: mean within-day Spearman rho on `frame`."""
    days, price = frame["delivery_day"].to_numpy(), frame["price"].to_numpy()

    def metric(predt, dmatrix):
        return "spearman", evaluate.mean_within_day_spearman(predt, price, days)
    return metric


def booster_params(kind, params, seed, n_threads=N_THREADS):
    return {**FIXED_PARAMS, **OBJECTIVES[kind], **params, "seed": seed, "nthread": n_threads}


def fit_predict(kind, params, train, predict, seed, n_threads=N_THREADS):
    """Fit steps 1-2 (see the module docstring) and predict `predict`.

    Returns (number of trees, predictions): prices in EUR/MWh for xgb-reg, scores (higher =
    more expensive) for xgb-rank.
    """
    p = booster_params(kind, params, seed, n_threads)
    inner, es = split_early_stopping(train)
    stop = {"early_stopping_rounds": EARLY_STOPPING_ROUNDS, "verbose_eval": False}
    if kind == "xgb-rank":
        # The ranker stops on the metric it is selected on (within-day Spearman rho).
        stop.update(custom_metric=_spearman_metric(es), maximize=True)
    booster = xgb.train(p, _dmatrix(kind, inner), MAX_TREES, evals=[(_dmatrix(kind, es), "es")], **stop)
    n_trees = booster.best_iteration + 1

    final = xgb.train(p, _dmatrix(kind, train), n_trees)
    return n_trees, final.predict(xgb.DMatrix(predict[features.FEATURE_COLUMNS]))


def sample_configs(n=N_CONFIGS, seed=SEARCH_SEED):
    """Random search space of PLAN.md §7. The same seed gives the same configurations to
    XGB-reg and XGB-rank."""
    rng = np.random.default_rng(seed)
    configs = []
    for _ in range(n):
        configs.append({
            "max_depth": int(rng.integers(3, 11)),
            "learning_rate": float(np.exp(rng.uniform(np.log(0.01), np.log(0.2)))),
            "min_child_weight": int(rng.integers(1, 21)),
            "subsample": float(rng.uniform(0.6, 1.0)),
            "colsample_bytree": float(rng.uniform(0.5, 1.0)),
            "reg_lambda": float(np.exp(rng.uniform(np.log(0.1), np.log(10.0)))),
        })
    return configs


# --- Search driver (XGBoost and LSTM) ---------------------------------------------------------
def selection_metric(kind):
    """(column, higher_is_better): RMSE for price models, Spearman rho for rankers (§7)."""
    return ("spearman_rho_mean", True) if kind.endswith("-rank") else ("rmse_eur_mwh", False)


def forecast_metrics(kind, pred, frame):
    """Metrics of one prediction on `frame`. A ranker's scores are not prices, so its RMSE is
    computed later on the "-rank" price vector (evaluate step)."""
    out = {"spearman_rho_mean": evaluate.mean_within_day_spearman(
        pred, frame["price"].to_numpy(), frame["delivery_day"].to_numpy())}
    if kind.endswith("-reg"):
        err = pred - frame["price"].to_numpy()
        out["rmse_eur_mwh"] = float(np.sqrt(np.mean(err ** 2)))
        out["mae_eur_mwh"] = float(np.mean(np.abs(err)))
    return out


def search(kind, feats, configs, fit, n_finalists, seeds=config.SEEDS, log=print):
    """Random search on the validation year (PLAN.md §7).

    Every configuration is fitted with seeds[0]; the n_finalists best are re-run with the
    other seeds, and the configuration whose seed-averaged forecast scores best is kept
    (mean prediction for price models, mean score for rankers, PLAN.md §4).

    Returns (table, predictions, selected): one table row per fitted (configuration, seed)
    plus one per seed average; validation predictions with one column per fit
    (c{config}_s{seed}, c{config}_avg) and `final`; the selected configuration.
    """
    train = training_rows(feats, config.TRAIN_START, config.TRAIN_END)
    val = forecast_rows(feats, config.VAL_START, config.VAL_END)
    preds = val[KEYS].copy()
    rows = []
    metric, higher_is_better = selection_metric(kind)

    def run(i, seed):
        started = time.time()
        n_iter, pred = fit(kind, configs[i], train, val, seed)
        preds[f"c{i:02d}_s{seed}"] = pred
        row = {"model": kind, "config": i, "seed": str(seed), **configs[i], "n_iter": n_iter,
               **forecast_metrics(kind, pred, val), "fit_seconds": time.time() - started}
        rows.append(row)
        log(f"{kind} config {i:2d} seed {seed}: n_iter {n_iter}, {metric} {row[metric]:.4f}, "
            f"{row['fit_seconds']:.0f} s")

    for i in range(len(configs)):
        run(i, seeds[0])
    first = pd.DataFrame(rows)
    finalists = first.sort_values(metric, ascending=not higher_is_better)["config"].head(n_finalists).tolist()
    log(f"{kind} finalists: {finalists}")

    for i in finalists:
        for seed in seeds[1:]:
            run(i, seed)
        avg = preds[[f"c{i:02d}_s{s}" for s in seeds]].mean(axis=1).to_numpy()
        preds[f"c{i:02d}_avg"] = avg
        rows.append({"model": kind, "config": i, "seed": "avg", **configs[i],
                     **forecast_metrics(kind, avg, val)})

    table = pd.DataFrame(rows)
    averaged = table[table["seed"] == "avg"]
    best = averaged.sort_values(metric, ascending=not higher_is_better).iloc[0]
    chosen = int(best["config"])
    preds["final"] = preds[f"c{chosen:02d}_avg"]
    table["selected"] = (table["config"] == chosen) & (table["seed"] == "avg")
    selected = {"model": kind, "config": chosen, "params": configs[chosen], "seeds": list(seeds),
                "selection_metric": metric, "validation_value": float(best[metric]),
                "n_iter_per_seed": [int(n) for n in table.loc[(table["config"] == chosen)
                                                              & (table["seed"] != "avg"), "n_iter"]]}
    log(f"{kind} selected config {chosen}: {metric} {best[metric]:.4f} (3-seed average)")
    return table, preds, selected
