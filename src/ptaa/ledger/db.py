"""Paper ledger + PaperBroker: SQLite via SQLModel. Fills apply the configured Indian fee model."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func
from sqlmodel import Session, SQLModel, create_engine, select

from ptaa.data.universe import sector_of
from ptaa.execution.broker import Fill, OrderRequest
from ptaa.graph.state import AuditEvent
from ptaa.ledger.models import AuditRow, EquityRow, OrderRow, PositionRow, RunRow
from ptaa.models import Module
from ptaa.risk.fees import compute_fees
from ptaa.risk.rules import OpenPosition, PortfolioState
from ptaa.settings import FeeConfig


class Ledger:
    def __init__(self, path: Path, fees: FeeConfig, starting_cash: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
        SQLModel.metadata.create_all(self.engine)
        self.fees = fees
        self.starting_cash = starting_cash

    # -- runs and audit ------------------------------------------------------
    def record_run(self, row: RunRow) -> None:
        with Session(self.engine) as s:
            s.merge(row)
            s.commit()

    def record_audit(self, run_id: str, events: list[AuditEvent]) -> None:
        with Session(self.engine) as s:
            for e in events:
                s.add(AuditRow(run_id=run_id, **e.model_dump()))
            s.commit()

    def audit_for(self, run_id: str) -> list[AuditRow]:
        with Session(self.engine) as s:
            return list(
                s.exec(select(AuditRow).where(AuditRow.run_id == run_id).order_by(AuditRow.at))
            )

    def runs(self, limit: int = 50) -> list[RunRow]:
        with Session(self.engine) as s:
            return list(s.exec(select(RunRow).order_by(RunRow.created_at.desc()).limit(limit)))  # type: ignore[attr-defined]

    # -- portfolio -----------------------------------------------------------
    def positions(self) -> list[PositionRow]:
        with Session(self.engine) as s:
            return list(s.exec(select(PositionRow)))

    def cash(self) -> float:
        with Session(self.engine) as s:
            buys = s.exec(
                select(func.coalesce(func.sum(OrderRow.price * OrderRow.quantity), 0.0)).where(
                    OrderRow.side == "BUY"
                )
            ).one()
            sells = s.exec(
                select(func.coalesce(func.sum(OrderRow.price * OrderRow.quantity), 0.0)).where(
                    OrderRow.side == "SELL"
                )
            ).one()
            fees = s.exec(select(func.coalesce(func.sum(OrderRow.fees), 0.0))).one()
        return float(self.starting_cash - buys + sells - fees)

    def fees_paid(self) -> float:
        with Session(self.engine) as s:
            return float(s.exec(select(func.coalesce(func.sum(OrderRow.fees), 0.0))).one())

    def equity(self, prices: dict[str, float] | None = None) -> tuple[float, float, float]:
        """(cash, positions_value, equity) using given prices else last marked prices."""
        cash = self.cash()
        value = 0.0
        for p in self.positions():
            px = (prices or {}).get(p.symbol, p.last_price or p.avg_price)
            value += px * p.quantity * (1 if p.side == "BUY" else -1) + (
                2 * p.avg_price * p.quantity if p.side == "SELL" else 0
            )
        return cash, value, cash + value

    def peak_equity(self) -> float:
        with Session(self.engine) as s:
            peak = s.exec(select(func.max(EquityRow.equity))).one()
        _, _, eq = self.equity()
        return float(max(peak or 0.0, eq, self.starting_cash))

    def snapshot(self, prices: dict[str, float] | None = None) -> EquityRow:
        cash, value, eq = self.equity(prices)
        realised = eq - self.starting_cash - value  # cash change ex positions
        row = EquityRow(
            at=datetime.now(tz=timezone.utc),
            cash=cash,
            positions_value=value,
            equity=eq,
            realised_pnl=realised,
            fees_paid=self.fees_paid(),
        )
        with Session(self.engine) as s:
            s.add(row)
            s.commit()
        return row

    def day_pnl(self, today: datetime | None = None) -> float:
        today = (today or datetime.now(tz=timezone.utc)).date()
        with Session(self.engine) as s:
            rows = list(
                s.exec(
                    select(EquityRow)
                    .where(func.date(EquityRow.at) == today.isoformat())
                    .order_by(EquityRow.at)
                )
            )
        if len(rows) < 2:
            return 0.0
        return float(rows[-1].equity - rows[0].equity)

    def portfolio_state(self, prices: dict[str, float] | None = None) -> PortfolioState:
        cash, _, eq = self.equity(prices)
        open_pos = [
            OpenPosition(
                symbol=p.symbol,
                sector=sector_of(p.symbol),
                value=(prices or {}).get(p.symbol, p.last_price or p.avg_price) * p.quantity,
            )
            for p in self.positions()
        ]
        return PortfolioState(
            equity=eq,
            peak_equity=self.peak_equity(),
            cash=cash,
            open_positions=open_pos,
            day_pnl=self.day_pnl(),
        )

    def mark(self, prices: dict[str, float]) -> None:
        with Session(self.engine) as s:
            for p in s.exec(select(PositionRow)):
                if p.symbol in prices:
                    p.last_price = prices[p.symbol]
                    s.add(p)
            s.commit()

    # -- fills ---------------------------------------------------------------
    def fill(self, req: OrderRequest, price: float, at: datetime, module: Module) -> Fill:
        fb = compute_fees(self.fees, req.side, module, price, req.quantity)
        order_id = uuid.uuid4().hex[:12]
        with Session(self.engine) as s:
            s.add(
                OrderRow(
                    run_id=req.run_id,
                    symbol=req.symbol,
                    module=module,
                    side=req.side,
                    quantity=req.quantity,
                    price=price,
                    fees=fb.total,
                    reason=req.reason,
                    strategy=req.strategy,
                    filled_at=at,
                )
            )
            pos = s.get(PositionRow, req.symbol)
            if req.reason == "entry":
                if pos is None:
                    s.add(
                        PositionRow(
                            symbol=req.symbol,
                            module=module,
                            side=req.side,
                            quantity=req.quantity,
                            avg_price=price,
                            stop_loss=req.stop_loss,
                            target=req.target,
                            strategy=req.strategy,
                            opened_at=at,
                            last_price=price,
                        )
                    )
                else:  # add to position, same side only
                    total = pos.quantity + req.quantity
                    pos.avg_price = (pos.avg_price * pos.quantity + price * req.quantity) / total
                    pos.quantity = total
                    s.add(pos)
            elif pos is not None:  # exit
                pos.quantity -= req.quantity
                if pos.quantity <= 0:
                    s.delete(pos)
                else:
                    s.add(pos)
            s.commit()
        return Fill(
            symbol=req.symbol,
            side=req.side,
            quantity=req.quantity,
            price=price,
            fees=fb.total,
            filled_at=at,
            order_id=order_id,
        )


class PaperBroker:
    name = "paper"
    is_live = False

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    def place(self, req: OrderRequest, price: float, at: datetime) -> Fill:
        return self.ledger.fill(req, price, at, req.module)

    def close_position(self, symbol: str, price: float, at: datetime, reason: str) -> Fill | None:
        pos = next((p for p in self.ledger.positions() if p.symbol == symbol), None)
        if pos is None:
            return None
        side = "SELL" if pos.side == "BUY" else "BUY"
        req = OrderRequest(
            symbol=symbol,
            module=pos.module,
            side=side,
            quantity=pos.quantity,
            stop_loss=pos.stop_loss,
            target=pos.target,
            strategy=pos.strategy,
            reason=reason,
        )  # type: ignore[arg-type]
        return self.ledger.fill(req, price, at, pos.module)  # type: ignore[arg-type]
