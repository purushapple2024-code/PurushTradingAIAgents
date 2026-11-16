from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ptaa.graph.graph import build_graph, run_once
from ptaa.mlmodels.ppo import PPOSizer, build_obs
from ptaa.settings import AppConfig
from ptaa.training.paper_env import OBS_DIM, Episode, PaperSizingEnv
from ptaa.training.train_ppo import evaluate_policy

from .test_graph import _ctx, _snapshot


def _episode(n: int = 400, seed: int = 0) -> Episode:
    rng = np.random.default_rng(seed)
    score = rng.uniform(0.5, 0.9, n).astype(np.float32)
    # informative episode: higher score -> higher chance of a positive return
    ret = np.where(rng.uniform(size=n) < score, 0.03, -0.012).astype(np.float32)
    return Episode(
        score=score,
        uncertainty=np.full(n, 0.012, np.float32),
        regime=rng.integers(0, 4, n),
        ret=ret,
        entry=np.full(n, 1000.0, np.float32),
        stop_frac=np.full(n, 0.024, np.float32),
    )


def test_env_contract_and_rules_cap(cfg: AppConfig) -> None:
    env = PaperSizingEnv(_episode(), cfg.risk["swing"], cfg.fees)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (OBS_DIM,) and env.observation_space.contains(obs)
    obs, r, term, trunc, info = env.step(np.array([1.0], dtype=np.float32))
    # fixed full risk: qty = min(1% risk / 24 per share, 5% position / 1000) = min(416, 50) = 50
    assert info["qty"] == 50
    obs, r, term, trunc, info = env.step(np.array([0.0], dtype=np.float32))
    assert info["qty"] == 0 and info["pnl"] == 0.0 and r == 0.0
    m_full = evaluate_policy(PaperSizingEnv(_episode(), cfg.risk["swing"], cfg.fees), lambda o: 1.0)
    m_zero = evaluate_policy(PaperSizingEnv(_episode(), cfg.risk["swing"], cfg.fees), lambda o: 0.0)
    assert m_zero["final_equity"] == 1_000_000.0 and m_full["steps"] == 400
    # a score-aware heuristic sizer should beat a flat one on an informative episode
    m_smart = evaluate_policy(
        PaperSizingEnv(_episode(), cfg.risk["swing"], cfg.fees), lambda o: float(o[0] > 0.7)
    )
    assert m_smart["final_equity"] > m_zero["final_equity"]


def _const_policy(path: Path, value: float) -> None:
    x = helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["batch", OBS_DIM])
    y = helper.make_tensor_value_info("action", TensorProto.FLOAT, ["batch", 1])
    w = numpy_helper.from_array(np.zeros((OBS_DIM, 1), dtype=np.float32), "W")
    b = numpy_helper.from_array(np.array([value], dtype=np.float32), "B")
    g = helper.make_graph(
        [
            helper.make_node("MatMul", ["obs", "W"], ["l"]),
            helper.make_node("Add", ["l", "B"], ["action"]),
        ],
        "p",
        [x],
        [y],
        initializer=[w, b],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    onnx.save(m, str(path))


def test_ppo_sizer_used_only_when_it_beat_fixed(tmp_path: Path, cfg: AppConfig) -> None:
    ctx = _ctx(tmp_path, cfg, p_up=0.85)
    f = tmp_path / "models" / "ppo" / "p.onnx"
    f.parent.mkdir(parents=True)
    _const_policy(f, 0.5)
    obs = build_obs(0.8, 0.012, "trend_up", -0.01, 0.6, 0.024, 0.02, 10)
    assert obs.shape == (1, OBS_DIM) and PPOSizer(f).risk_fraction(obs) == 0.5
    # registered but did NOT beat fixed sizing -> trader ignores it
    ctx.registry.register(
        "ppo",
        "sizer",
        f,
        [f"obs{i}" for i in range(OBS_DIM)],
        date(2020, 1, 1),
        date.today(),
        "NIFTY50",
        400,
        {"beats_fixed": 0.0},
    )
    g = build_graph(ctx)
    snap = _snapshot("TCS")
    out = run_once(g, ctx, "TCS", "swing", snap)
    assert "fixed-fraction" in out["proposal"].rationale
    q_fixed = out["proposal"].quantity
    ctx.broker.close_position("TCS", snap["price"], out["as_of"], "manual")  # free the symbol
    # new version that beat fixed -> trader uses the 0.5 fraction -> about half the quantity
    ctx.registry.register(
        "ppo",
        "sizer",
        f,
        [f"obs{i}" for i in range(OBS_DIM)],
        date(2020, 1, 1),
        date.today(),
        "NIFTY50",
        400,
        {"beats_fixed": 1.0},
    )
    out2 = run_once(g, ctx, "TCS", "swing", snap)
    assert "PPO size 0.50" in out2["proposal"].rationale
    assert 0 < out2["proposal"].quantity <= q_fixed
    assert (
        abs(out2["proposal"].quantity - q_fixed / 2) <= 1 or out2["proposal"].quantity == q_fixed
    )  # position cap may bind
