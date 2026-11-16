"""LSTM sequence model trainer (PyTorch; run on Colab/Kaggle) with ONNX export and registry entry.

Target: log return over `horizon` bars. Loss: Gaussian NLL so the model learns its own uncertainty
(second head = log_std). Walk-forward: train on the first 80% of time, early-stop on the next 10%,
report on the last 10%. Exported ONNX runs on the laptop via onnxruntime (no torch needed).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ptaa.mlmodels.registry import ModelRegistry
from ptaa.mlmodels.seq import CHANNELS, WINDOW, make_windows
from ptaa.training.dataset import CandleSource

log = logging.getLogger(__name__)


def build_seq_dataset(
    symbols: list[str], candles: CandleSource, horizon: int, window: int = WINDOW
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, ts = [], [], []
    for sym in symbols:
        df = candles(sym)
        if df is None or len(df) < window + horizon + 10:
            continue
        w = make_windows(df, window)  # window i ends at bar i+window-1
        c = df["close"].to_numpy(dtype=float)
        end_idx = np.arange(len(w)) + window - 1
        valid = end_idx + horizon < len(c)
        y = np.log(c[end_idx[valid] + horizon] / c[end_idx[valid]])
        xs.append(w[valid])
        ys.append(y.astype(np.float32))
        ts.append(df.index.to_numpy()[end_idx[valid]])
    if not xs:
        raise ValueError("no sequence data")
    X, y, t = np.concatenate(xs), np.concatenate(ys), np.concatenate(ts)
    order = np.argsort(t, kind="stable")
    return X[order], y[order], t[order]


def train_seq(
    symbols: list[str],
    candles: CandleSource,
    registry: ModelRegistry,
    horizon: int = 5,
    universe: str = "NIFTY50",
    epochs: int = 30,
    hidden: int = 64,
    dropout: float = 0.2,
    lr: float = 1e-3,
    batch: int = 256,
    seed: int = 42,
) -> dict[str, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as e:  # pragma: no cover - torch is optional on the laptop
        raise RuntimeError("train_seq needs PyTorch; run the Colab/Kaggle notebook") from e

    torch.manual_seed(seed)
    X, y, t = build_seq_dataset(symbols, candles, horizon)
    n = len(X)
    i_tr, i_va = int(n * 0.8), int(n * 0.9)
    mu, sd = X[:i_tr].mean(axis=(0, 1)), X[:i_tr].std(axis=(0, 1)) + 1e-6
    Xn = (X - mu) / sd

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(CHANNELS, hidden, num_layers=2, batch_first=True, dropout=dropout)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(hidden, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            return self.head(self.drop(out[:, -1]))

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    nll = nn.GaussianNLLLoss()
    Xt, yt = torch.tensor(Xn), torch.tensor(y)

    def run_epoch(lo: int, hi: int, train: bool) -> float:
        net.train(train)
        tot, cnt = 0.0, 0
        for s in range(lo, hi, batch):
            xb, yb = Xt[s : min(s + batch, hi)], yt[s : min(s + batch, hi)]
            out = net(xb)
            loss = nll(out[:, 0], yb, torch.exp(out[:, 1]) ** 2)
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            tot += float(loss) * len(xb)
            cnt += len(xb)
        return tot / max(cnt, 1)

    best, best_state, patience = float("inf"), None, 0
    for ep in range(epochs):
        tr = run_epoch(0, i_tr, True)
        with torch.no_grad():
            va = run_epoch(i_tr, i_va, False)
        log.info("epoch %d train %.4f val %.4f", ep, tr, va)
        if va < best - 1e-5:
            best, best_state, patience = va, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 5:
                break
    if best_state:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        out = net(Xt[i_va:]).numpy()
    pred, truth = out[:, 0], y[i_va:]
    metrics = {
        "test_mae": float(np.mean(np.abs(pred - truth))),
        "test_sign_acc": float(np.mean(np.sign(pred) == np.sign(truth))),
        "test_corr": float(np.corrcoef(pred, truth)[0, 1]) if len(pred) > 2 else 0.0,
        "val_nll": float(best),
        "n_train": float(i_tr),
        "n_test": float(n - i_va),
    }

    # export: fold normalisation into the graph so inference takes raw windows
    class Wrapped(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = net
            self.register_buffer("mu", torch.tensor(mu))
            self.register_buffer("sd", torch.tensor(sd))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net((x - self.mu) / self.sd)

    model_dir = registry.root / "seq"
    model_dir.mkdir(parents=True, exist_ok=True)
    version = registry.next_version("seq_lstm", f"{horizon}d")
    file = model_dir / f"seq_lstm-{horizon}d-v{version}.onnx"
    torch.onnx.export(
        Wrapped().eval(),
        torch.zeros(1, WINDOW, CHANNELS),
        str(file),
        input_names=["window"],
        output_names=["out"],
        dynamic_axes={"window": {0: "batch"}, "out": {0: "batch"}},
        opset_version=17,
    )
    (model_dir / f"seq_lstm-{horizon}d-v{version}.meta.json").write_text(
        json.dumps({"window": WINDOW, "channels": CHANNELS, "horizon": horizon}), encoding="utf-8"
    )
    rec = registry.register(
        kind="seq_lstm",
        horizon=f"{horizon}d",
        file=file.relative_to(registry.root),
        feature_names=[f"seq_ch{i}" for i in range(CHANNELS)],
        train_start=pd.Timestamp(t[0]).date(),
        train_end=pd.Timestamp(t[i_tr - 1]).date(),
        universe=universe,
        n_samples=n,
        metrics=metrics,
        params={
            "window": WINDOW,
            "hidden": hidden,
            "dropout": dropout,
            "epochs": epochs,
            "horizon_bars": horizon,
        },
        notes=f"LSTM Gaussian-NLL, exported {datetime.now().isoformat()}",
    )
    return {"model_id": rec.model_id, "metrics": metrics}
