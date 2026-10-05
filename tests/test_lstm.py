"""LSTM checks: padding and the mask on 23-, 24- and 25-hour days, and both losses."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
import torch

from bessrank import features, lstm


def synthetic_frame(lengths, seed=0):
    """Hourly rows for consecutive days of the given lengths, sorted by time. Price depends
    on two features so that a model can learn it."""
    rng = np.random.default_rng(seed)
    rows = []
    for k, n in enumerate(lengths):
        for h in range(n):
            rows.append({"delivery_day": date(2024, 1, 1) + timedelta(days=k), "local_hour": min(h, 23)})
    frame = pd.DataFrame(rows)
    for col in features.FEATURE_COLUMNS:
        if col != "local_hour":
            frame[col] = rng.normal(size=len(frame))
    frame["price"] = 50 + 20 * frame["load_fc"] - 15 * frame["pv_fc"] + rng.normal(size=len(frame))
    return frame


def test_to_days_pads_dst_days():
    frame = synthetic_frame([23, 24, 25])
    days = lstm.to_days(frame, lstm.Scaler(frame))
    assert days["x"].shape == (3, 25, len(features.FEATURE_COLUMNS))
    assert days["lengths"].tolist() == [23, 24, 25]
    assert days["mask"].sum(dim=1).tolist() == [23, 24, 25]
    assert not days["mask"][0, 23:].any() and days["mask"][2].all()
    assert (days["x"][0, 23:] == 0).all() and (days["y"][0, 23:] == 0).all()


@pytest.mark.parametrize("layers", [1, 2])
def test_padding_does_not_change_real_hours(layers):
    """Real hours of a 23- or 25-hour day get the same output alone, in a padded batch, and
    with garbage in the padding: the backward direction must start at the last real hour."""
    torch.manual_seed(0)
    model = lstm.BiLSTM(n_features=5, hidden=8, layers=layers).eval()
    lengths = torch.tensor([23, 25, 24])
    x = torch.randn(3, 25, 5)
    with torch.no_grad():
        batch = model(x, lengths)
        garbage = x.clone()
        garbage[0, 23:] = 1e3
        garbage[2, 24:] = -1e3
        batch_garbage = model(garbage, lengths)
        for i, n in enumerate(lengths.tolist()):
            alone = model(x[i:i + 1, :n], torch.tensor([n]))[0]
            assert torch.allclose(batch[i, :n], alone, atol=1e-6)
            assert torch.allclose(batch_garbage[i, :n], alone, atol=1e-6)


def test_losses_ignore_padding():
    torch.manual_seed(0)
    out, y = torch.randn(2, 25), torch.randn(2, 25)
    mask = torch.ones(2, 25, dtype=torch.bool)
    mask[0, 23:] = False
    out2, y2 = out.clone(), y.clone()
    out2[0, 23:], y2[0, 23:] = 100.0, -100.0
    for loss in (lstm.mse_loss, lstm.pairwise_loss):
        assert torch.isclose(loss(out, y, mask), loss(out2, y2, mask))


def test_pairwise_loss_formula():
    """One 3-hour day: mean over the pairs with different prices of log(1 + exp(-(s_i - s_j)))."""
    out = torch.tensor([[0.5, -1.0, 2.0]])
    y = torch.tensor([[1.0, 1.0, 3.0]])  # hours 0 and 1 tie, so only (2, 0) and (2, 1) count
    mask = torch.ones(1, 3, dtype=torch.bool)
    expected = (np.log1p(np.exp(-(2.0 - 0.5))) + np.log1p(np.exp(-(2.0 + 1.0)))) / 2
    assert lstm.pairwise_loss(out, y, mask).item() == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("kind", ["lstm-reg", "lstm-rank"])
def test_both_losses_decrease(kind):
    lengths = [24] * 40 + [23, 25] * 4
    frame = synthetic_frame(lengths)
    days = lstm.to_days(frame, lstm.Scaler(frame))
    _, _, history = lstm.train_model(kind, {"hidden": 16, "layers": 1}, days, seed=0, epochs=40)
    assert history[-1] < 0.9 * history[0]
    assert np.mean(history[-5:]) < np.mean(history[:5])
