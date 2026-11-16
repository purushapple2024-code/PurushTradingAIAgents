from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import pandas as pd
import pytest
from onnx import TensorProto, helper, numpy_helper

from ptaa.graph.graph import build_graph, run_once
from ptaa.graph.nodes.data import market_snapshot
from ptaa.mlmodels.seq import CHANNELS, WINDOW, SeqModel, last_window, make_windows
from ptaa.settings import AppConfig
from ptaa.training.pipeline import synthetic_candles

from .test_graph import _ctx


def _tiny_onnx(path: Path, mean: float, log_std: float) -> None:
    """Constant model: ReduceMean over the window + bias -> [mean, log_std]. Exercises the I/O contract."""
    x = helper.make_tensor_value_info("window", TensorProto.FLOAT, ["batch", WINDOW, CHANNELS])
    y = helper.make_tensor_value_info("out", TensorProto.FLOAT, ["batch", 2])
    w = numpy_helper.from_array(np.zeros((CHANNELS, 2), dtype=np.float32), "W")
    b = numpy_helper.from_array(np.array([mean, log_std], dtype=np.float32), "B")
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), "axes")
    nodes = [
        helper.make_node("ReduceMean", ["window", "axes"], ["pooled"], keepdims=0),
        helper.make_node("MatMul", ["pooled", "W"], ["lin"]),
        helper.make_node("Add", ["lin", "B"], ["out"]),
    ]
    graph = helper.make_graph(nodes, "tiny", [x], [y], initializer=[w, b, axes])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def test_make_windows_shape_and_alignment() -> None:
    df = synthetic_candles("AAA", n=300)
    w = make_windows(df)
    assert w.shape == (300 - WINDOW + 1, WINDOW, CHANNELS) and w.dtype == np.float32
    c = df["close"].to_numpy()
    assert abs(w[-1, -1, 0] - np.log(c[-1] / c[-2])) < 1e-5
    lw = last_window(df)
    assert lw is not None and lw.shape == (1, WINDOW, CHANNELS) and np.allclose(lw[0], w[-1])
    assert last_window(df.head(50)) is None


def test_seq_model_inference_and_probabilities(tmp_path: Path) -> None:
    f = tmp_path / "m.onnx"
    _tiny_onnx(f, mean=0.02, log_std=float(np.log(0.01)))
    m = SeqModel(f, horizon_bars=5)
    mean, std = m.predict(np.zeros((1, WINDOW, CHANNELS), dtype=np.float32))
    assert abs(mean - 0.02) < 1e-5 and abs(std - 0.01) < 1e-5
    p_up, p_down = m.to_probabilities(mean, std)
    assert p_up > 0.95 and p_down < 0.02
    p_up2, p_down2 = m.to_probabilities(0.0, 0.01)
    assert abs(p_up2 - p_down2) < 1e-9 and p_up2 < 0.5


def test_sequence_analyst_in_graph(tmp_path: Path, cfg: AppConfig) -> None:
    ctx = _ctx(tmp_path, cfg, p_up=0.55)
    f = tmp_path / "models" / "seq" / "s.onnx"
    f.parent.mkdir(parents=True)
    _tiny_onnx(f, mean=0.03, log_std=float(np.log(0.01)))
    rec = ctx.registry.register(
        "seq_lstm",
        "5d",
        f,
        [f"seq_ch{i}" for i in range(CHANNELS)],
        pd.Timestamp("2020-01-01").date(),
        datetime.now(tz=timezone.utc).date(),
        "NIFTY50",
        1000,
        {"test_mae": 0.01},
        params={"horizon_bars": 5},
    )
    assert ctx.registry.verify(rec.model_id)[0]
    snap = market_snapshot(
        synthetic_candles("RELIANCE", n=600), synthetic_candles("NIFTY", n=600)["close"]
    )
    assert snap["seq_window"] is not None and len(snap["seq_window"]) == WINDOW
    snap["index_returns"] = [0.003] * 60
    out = run_once(build_graph(ctx), ctx, "RELIANCE", "swing", snap)
    assert out["errors"] == [], out["errors"]
    seq = [s for s in out["verified_signals"] if s.analyst == "sequence"]
    assert len(seq) == 1 and seq[0].p_up > 0.9
    assert seq[0].expected_move_pct == pytest.approx(3.0, abs=0.01)
    assert out["ensemble"].weights.get("sequence", 0) > 0
    # tamper with the onnx file -> registry verification fails -> no sequence signal next run
    f.write_bytes(f.read_bytes() + b"x")
    ctx.models.clear()
    out2 = run_once(build_graph(ctx), ctx, "TCS", "swing", {**snap})
    assert all(s.analyst != "sequence" for s in out2.get("verified_signals", []))
