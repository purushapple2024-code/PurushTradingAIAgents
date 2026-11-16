"""LLM-backed narrative: debate, thesis text, risk persona views, PM report.

Guards applied to every LLM output before it enters state:
- `strip_uncited_numbers`: a sentence containing a number that does not appear in the verified
  input set is removed (the model may not introduce figures).
- `cited_ids`: only signal ids that exist are kept; sentences citing unknown ids are removed.
- any failure -> deterministic template, so the run always completes.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from ptaa.graph.state import DebateTurn, Signal
from ptaa.llm.client import LLMClient, fence_untrusted, load_prompt

_NUM = re.compile(r"(?<![A-Za-z])[-+]?\d[\d,]*\.?\d*%?")
_SENT = re.compile(r"(?<=[.!?])\s+")
_CITE = re.compile(r"\[(S\d+)\]")


class Argument(BaseModel):
    text: str = Field(min_length=10, max_length=1200)
    cited: list[str] = Field(default_factory=list)


class Narrative(BaseModel):
    narrative: str = Field(min_length=10, max_length=1200)
    key_risks: list[str] = Field(default_factory=list, max_length=5)
    cited: list[str] = Field(default_factory=list)


class PersonaView(BaseModel):
    view: str = Field(min_length=10, max_length=600)


class Report(BaseModel):
    report: str = Field(min_length=10, max_length=1500)


def allowed_numbers(signals: list[Signal], extra: dict[str, Any] | None = None) -> set[str]:
    """Numbers the model is allowed to mention: anything in the signals' text and the extra context."""
    blob = " ".join(
        f"{s.p_up:.2f} {s.p_down:.2f} {s.summary} "
        + " ".join(e.claim + " " + (e.quote or "") for e in s.evidence)
        for s in signals
    )
    if extra:
        blob += " " + " ".join(str(v) for v in extra.values())
    return {_canon(n) for n in _NUM.findall(blob)}


def _canon(n: str) -> str:
    n = n.replace(",", "").rstrip("%").lstrip("+")
    try:
        return f"{float(n):g}"
    except ValueError:
        return n


def strip_uncited_numbers(text: str, allowed: set[str], known_ids: set[str]) -> tuple[str, int]:
    """Remove sentences with numbers not in `allowed` or citing unknown signal ids. Returns (text, removed)."""
    kept, removed = [], 0
    for sent in _SENT.split(text.strip()):
        nums = {_canon(n) for n in _NUM.findall(_CITE.sub("", sent))}
        ids = set(_CITE.findall(sent))
        if (nums - allowed) or (ids - known_ids):
            removed += 1
            continue
        kept.append(sent)
    return " ".join(kept).strip(), removed


def _fmt_signals(signals: list[Signal]) -> str:
    lines = []
    for s in signals:
        parts = []
        for e in s.evidence[:4]:
            item = e.claim
            if e.quote:  # article text is untrusted: fence it so the model treats it as data
                item += " " + fence_untrusted(e.quote[:200], label="quote")
            parts.append(item)
        lines.append(
            f"[{s.signal_id}] {s.analyst} ({s.horizon}): P(up)={s.p_up:.2f} P(down)={s.p_down:.2f}. "
            f"{s.summary}. Evidence: " + " | ".join(parts)
        )
    return "\n".join(lines)


def run_debate(
    llm: LLMClient,
    symbol: str,
    module: str,
    regime: str,
    ensemble_score: float,
    signals: list[Signal],
    rounds: int = 2,
) -> list[DebateTurn]:
    system_v, system = load_prompt("system")
    known = {s.signal_id for s in signals}
    allowed = allowed_numbers(signals, {"score": f"{ensemble_score:.2f}"})
    turns: list[DebateTurn] = []
    for r in range(1, rounds + 1):
        for side in ("bull", "bear"):
            _, tmpl = load_prompt(side)
            prior = "\n".join(f"{t.side.upper()} r{t.round}: {t.text}" for t in turns)
            user = tmpl.format(
                symbol=symbol,
                regime=regime,
                ensemble_score=f"{ensemble_score:.2f}",
                module=module,
                signals=_fmt_signals(signals),
                prior_debate=("Prior debate:\n" + prior) if prior else "",
            )
            out = llm.complete_json(Argument, system, user, prompt_name=f"{side}_v{system_v}")
            if out is None:
                text = _template_argument(side, signals)
            else:
                text, _ = strip_uncited_numbers(out.text, allowed, known)
                text = text or _template_argument(side, signals)
            turns.append(
                DebateTurn(side=side, round=r, text=text, cited=sorted(set(_CITE.findall(text))))
            )  # type: ignore[arg-type]
    return turns


def _template_argument(side: str, signals: list[Signal]) -> str:
    if side == "bull":
        parts = [
            f"[{s.signal_id}] {s.analyst} P(up)={s.p_up:.2f}" for s in signals if s.p_up >= s.p_down
        ]
        return "Supporting: " + "; ".join(parts) if parts else "No supporting signals."
    parts = [
        f"[{s.signal_id}] {s.analyst} P(down)={s.p_down:.2f}" for s in signals if s.p_down > s.p_up
    ]
    return "Opposing: " + "; ".join(parts) if parts else "No opposing signals."


def write_narrative(
    llm: LLMClient,
    symbol: str,
    direction: str,
    regime: str,
    debate: list[DebateTurn],
    signals: list[Signal],
    fallback: str,
) -> tuple[str, list[str]]:
    _, system = load_prompt("system")
    _, tmpl = load_prompt("manager")
    known = {s.signal_id for s in signals}
    allowed = allowed_numbers(signals)
    user = tmpl.format(
        symbol=symbol,
        direction=direction,
        regime=regime,
        debate="\n".join(f"{t.side.upper()} r{t.round}: {t.text}" for t in debate),
    )
    out = llm.complete_json(Narrative, system, user, prompt_name="manager")
    if out is None:
        return fallback, []
    text, _ = strip_uncited_numbers(out.narrative, allowed, known)
    risks = [r for r in out.key_risks if not (set(map(_canon, _NUM.findall(r))) - allowed)]
    return (text or fallback), risks[:5]


def persona_views(llm: LLMClient, proposal: str, rules: str, risks: list[str]) -> list[str]:
    _, system = load_prompt("system")
    _, tmpl = load_prompt("risk_persona")
    views = []
    for persona in ("aggressive", "neutral", "conservative"):
        out = llm.complete_json(
            PersonaView,
            system,
            tmpl.format(
                persona=persona, proposal=proposal, rules=rules, risks=", ".join(risks) or "none"
            ),
            prompt_name=f"persona_{persona}",
        )
        views.append(f"{persona}: {out.view}" if out else f"{persona}: (no view; rules decided)")
    return views


def write_report(
    llm: LLMClient,
    symbol: str,
    decision: str,
    thesis: str,
    verdict: str,
    personas: list[str],
    fallback: str,
    allowed: set[str],
    known: set[str],
) -> str:
    _, system = load_prompt("system")
    _, tmpl = load_prompt("report")
    out = llm.complete_json(
        Report,
        system,
        tmpl.format(
            symbol=symbol,
            decision=decision,
            thesis=thesis,
            verdict=verdict,
            personas="\n".join(personas),
        ),
        prompt_name="report",
    )
    if out is None:
        return fallback
    text, _ = strip_uncited_numbers(
        out.report, allowed | {_canon(n) for n in _NUM.findall(decision + verdict + thesis)}, known
    )
    return text or fallback
