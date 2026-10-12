"""
Encoder-decoder LSTM over features/sequences.py samples.

The encoder reads the last 168 known hours; its final state initialises a
decoder that runs over the target hours with their forecast weather and
calendar, emitting load / level per hour. Loss is masked MSE so padded DST
hours and missing labels don't count. CPU is enough at this size.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from shef.features.sequences import DEC_FEATURES, ENC_FEATURES, Sequences

SEED = 42


class Seq2Seq(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.LSTM(len(ENC_FEATURES), hidden, layers, batch_first=True,
                               dropout=dropout if layers > 1 else 0.0)
        self.decoder = nn.LSTM(len(DEC_FEATURES), hidden, layers, batch_first=True,
                               dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, enc, dec):
        _, state = self.encoder(enc)
        out, _ = self.decoder(dec, state)
        return 1.0 + self.head(out).squeeze(-1)   # predict around a ratio of 1


@dataclass
class Scaler:
    """Standardise the weather columns; everything else is already O(1)."""
    enc_mu: float = 0.0
    enc_sd: float = 1.0

    def fit(self, s: Sequences) -> "Scaler":
        t = s.enc[:, :, ENC_FEATURES.index("temp_c")]
        self.enc_mu, self.enc_sd = float(np.nanmean(t)), float(np.nanstd(t) or 1.0)
        return self

    def apply(self, s: Sequences) -> tuple[np.ndarray, np.ndarray]:
        enc, dec = s.enc.copy(), s.dec.copy()
        i, j = ENC_FEATURES.index("temp_c"), DEC_FEATURES.index("temp_c")
        enc[:, :, i] = (enc[:, :, i] - self.enc_mu) / self.enc_sd
        dec[:, :, j] = (dec[:, :, j] - self.enc_mu) / self.enc_sd
        return np.nan_to_num(enc), np.nan_to_num(dec)


def _tensors(s: Sequences, scaler: Scaler):
    enc, dec = scaler.apply(s)
    f = lambda a: torch.tensor(a, dtype=torch.float32)
    return f(enc), f(dec), f(s.y), f(s.mask)


def _masked_mse(pred, y, mask):
    return ((pred - y) ** 2 * mask).sum() / mask.sum().clamp(min=1)


def fit(train: Sequences, hidden: int = 64, layers: int = 1, dropout: float = 0.1, lr: float = 2e-3,
        epochs: int = 60, batch: int = 64, patience: int = 8, val_frac: float = 0.1, verbose: bool = False):
    """Early-stops on the most recent `val_frac` of training samples (time-ordered)."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)
    scaler = Scaler().fit(train)
    enc, dec, y, mask = _tensors(train, scaler)
    n_val = max(1, int(len(train) * val_frac))
    tr, va = slice(0, len(train) - n_val), slice(len(train) - n_val, len(train))

    model = Seq2Seq(hidden, layers, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, bad, history = np.inf, None, 0, []
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(tr.stop)
        for k in range(0, tr.stop, batch):
            b = order[k:k + batch]
            opt.zero_grad()
            loss = _masked_mse(model(enc[b], dec[b]), y[b], mask[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(_masked_mse(model(enc[va], dec[va]), y[va], mask[va]))
        history.append(v)
        if verbose:
            print(f"    epoch {epoch:2d}  val mse {v:.5f}")
        if v < best - 1e-6:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, scaler, {"epochs": len(history), "best_val_mse": best, "seconds": time.time() - t0}


def predict(model: Seq2Seq, scaler: Scaler, s: Sequences) -> np.ndarray:
    """MW per (sample, target hour); NaN where padded."""
    enc, dec, _, _ = _tensors(s, scaler)
    model.eval()
    with torch.no_grad():
        ratio = model(enc, dec).numpy()
    mw = ratio * s.level[:, None]
    mw[np.isnat(s.target_ts)] = np.nan
    return mw


def subset(s: Sequences, keep: np.ndarray) -> Sequences:
    return Sequences(*(getattr(s, f)[keep] for f in ("enc", "dec", "y", "mask", "level", "target_ts", "issue")))


def target_days(s: Sequences) -> "pd.DataFrame":
    """One row per sample, ts_utc = first target hour, so folds.* masks apply."""
    import pandas as pd

    return pd.DataFrame({"ts_utc": pd.DatetimeIndex(s.target_ts[:, 0]).tz_localize("UTC")})


def to_rows(s: Sequences, pred_mw: np.ndarray, fold) -> "pd.DataFrame":
    """Flatten to scored rows (same columns as cv.Result.scored), real target hours only."""
    import pandas as pd

    real = s.mask.astype(bool)
    ts = pd.DatetimeIndex(s.target_ts[real]).tz_localize("UTC")
    issue = pd.DatetimeIndex(np.repeat(s.issue, real.sum(axis=1))).tz_localize("UTC")
    local = ts.tz_convert("America/Chicago")
    return pd.DataFrame({
        "ts_utc": ts, "issue_time_utc": issue, "lead_hours": ((ts - issue) / pd.Timedelta(hours=1)).astype(int),
        "hour": local.hour, "month": local.month, "y": (s.y * s.level[:, None])[real], "pred": pred_mw[real],
        "fold": fold,
    })
