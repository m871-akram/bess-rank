"""Bidirectional LSTM with the price (masked MSE) and pairwise ranking losses (PLAN.md §4).

Each sample is one delivery day: a sequence of up to 25 hourly feature vectors (the XGBoost
features), padded to 25 hours with a mask. The LSTM runs on packed sequences, so on a
23-hour day the backward direction starts at the last real hour, not at the padding, and
padded hours never change the output of real hours (tests/test_lstm.py).

fit_predict follows the same fit procedure as models.fit_predict: early stopping on the last
3 months of the training window, then a refit on the full window for the chosen number of
epochs.
"""
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from bessrank import evaluate, features, models

MAX_HOURS = 25
DROPOUT = 0.1
LEARNING_RATE = 1e-3
BATCH_DAYS = 32
MAX_EPOCHS = 100
PATIENCE = 10  # epochs without improvement on the early-stopping months
N_THREADS = 2  # runs next to another search on the 4-vCPU VM

# PLAN.md §7: hidden 32, 64 or 128 units x 1 or 2 layers.
CONFIGS = [{"hidden": h, "layers": n} for h in (32, 64, 128) for n in (1, 2)]


# --- Data: one padded sequence per day ------------------------------------------------------
class Scaler:
    """Standardisation and missing-value filling, fitted on training rows only."""

    def __init__(self, train):
        X = train[features.FEATURE_COLUMNS]
        self.mean = X.mean()
        self.std = X.std().replace(0.0, 1.0)  # a constant feature stays 0 after scaling
        # Forecast days are never dropped; a missing value gets the training median of that
        # feature at the same wall-clock hour (PLAN.md §2).
        self.hour_median = X.groupby(train["local_hour"]).median()
        self.price_mean = float(train["price"].mean())
        self.price_std = float(train["price"].std())

    def features(self, frame):
        X = frame[features.FEATURE_COLUMNS]
        fill = self.hour_median.reindex(frame["local_hour"]).set_index(X.index)
        return ((X.fillna(fill) - self.mean) / self.std).to_numpy(dtype=np.float32)

    def price(self, frame):
        return ((frame["price"] - self.price_mean) / self.price_std).to_numpy(dtype=np.float32)


def to_days(frame, scaler):
    """Padded day tensors from hourly rows sorted by time.

    Returns x [days, 25, features], y [days, 25] (standardised price; 0 on padding),
    mask [days, 25] (True for real hours) and lengths [days].
    """
    day = np.unique(frame["delivery_day"].to_numpy(), return_inverse=True)[1]
    pos = frame.groupby("delivery_day").cumcount().to_numpy()  # hour index within the day
    n_days = day.max() + 1
    x = np.zeros((n_days, MAX_HOURS, len(features.FEATURE_COLUMNS)), dtype=np.float32)
    y = np.zeros((n_days, MAX_HOURS), dtype=np.float32)
    mask = np.zeros((n_days, MAX_HOURS), dtype=bool)
    x[day, pos] = scaler.features(frame)
    y[day, pos] = scaler.price(frame)
    mask[day, pos] = True
    return {"x": torch.from_numpy(x), "y": torch.from_numpy(y), "mask": torch.from_numpy(mask),
            "lengths": torch.from_numpy(mask.sum(axis=1))}


# --- Model and losses -----------------------------------------------------------------------
class BiLSTM(nn.Module):
    """Bidirectional LSTM, then a linear head giving one value per hour. Each hour's output
    sees the whole day (both directions)."""

    def __init__(self, n_features, hidden, layers, dropout=DROPOUT):
        super().__init__()
        # PyTorch applies `dropout` only between stacked layers; the extra Dropout before
        # the head regularises the 1-layer configurations too.
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        return self.head(self.dropout(out)).squeeze(-1)


def mse_loss(out, y, mask):
    """LSTM-reg: mean squared error on the standardised price, over real hours only."""
    return ((out - y) ** 2)[mask].mean()


def pairwise_loss(out, y, mask):
    """LSTM-rank: log(1 + exp(-(s_i - s_j))) over every pair of real hours with
    price_i > price_j (the XGB-rank objective). Averaged within each day, then over days, so
    every day weighs the same; days with a flat price have no pair and are skipped."""
    diff = out[:, :, None] - out[:, None, :]
    pairs = (y[:, :, None] > y[:, None, :]) & mask[:, :, None] & mask[:, None, :]
    per_pair = nn.functional.softplus(-diff) * pairs
    n_pairs = pairs.sum(dim=(1, 2))
    has_pairs = n_pairs > 0
    return (per_pair.sum(dim=(1, 2))[has_pairs] / n_pairs[has_pairs]).mean()


LOSSES = {"lstm-reg": mse_loss, "lstm-rank": pairwise_loss}


# --- Training -------------------------------------------------------------------------------
def predict_hours(model, days):
    """Model output for the real hours, in the row order of the frame given to to_days."""
    model.eval()
    with torch.no_grad():
        out = model(days["x"], days["lengths"])
    return out[days["mask"]].numpy().astype(float)


def train_model(kind, params, days, seed, epochs, es=None):
    """Adam on batches of 32 days for `epochs` epochs.

    With `es` = (day tensors, frame, scaler), stops after PATIENCE epochs without improvement
    of the early-stopping score (RMSE in EUR/MWh for lstm-reg, within-day Spearman rho for
    lstm-rank) and returns the best epoch count. Returns (model, epochs, loss per epoch).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = BiLSTM(days["x"].shape[2], params["hidden"], params["layers"])
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = LOSSES[kind]
    n_days = days["x"].shape[0]
    best_score, best_epoch, history = None, epochs, []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        order = rng.permutation(n_days)  # new shuffle of the days every epoch
        for start in range(0, n_days, BATCH_DAYS):
            idx = torch.from_numpy(order[start:start + BATCH_DAYS])
            loss = loss_fn(model(days["x"][idx], days["lengths"][idx]), days["y"][idx], days["mask"][idx])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item() * len(idx)
        history.append(total / n_days)
        if es is None:
            continue
        score = early_stopping_score(kind, model, *es)
        if best_score is None or score > best_score:
            best_score, best_epoch = score, epoch
        elif epoch - best_epoch >= PATIENCE:
            break
    return model, best_epoch, history


def early_stopping_score(kind, model, es_days, es_frame, scaler):
    """Higher is better: -RMSE (EUR/MWh) for lstm-reg, mean Spearman rho for lstm-rank."""
    pred = predict_hours(model, es_days)
    price = es_frame["price"].to_numpy()
    if kind == "lstm-reg":
        return -float(np.sqrt(np.mean((pred * scaler.price_std + scaler.price_mean - price) ** 2)))
    return evaluate.mean_within_day_spearman(pred, price, es_frame["delivery_day"].to_numpy())


def fit_predict(kind, params, train, predict, seed, n_threads=N_THREADS):
    """Early stopping on the last 3 months, refit on the full window, predict `predict`.

    Returns (epochs, predictions): prices in EUR/MWh for lstm-reg, scores (higher = more
    expensive) for lstm-rank.
    """
    torch.set_num_threads(n_threads)
    inner, es = models.split_early_stopping(train)
    scaler = Scaler(inner)
    _, epochs, _ = train_model(kind, params, to_days(inner, scaler), seed, MAX_EPOCHS,
                               es=(to_days(es, scaler), es, scaler))

    scaler = Scaler(train)
    model, _, _ = train_model(kind, params, to_days(train, scaler), seed, epochs)
    pred = predict_hours(model, to_days(predict, scaler))
    if kind == "lstm-reg":
        pred = pred * scaler.price_std + scaler.price_mean
    return epochs, pred
