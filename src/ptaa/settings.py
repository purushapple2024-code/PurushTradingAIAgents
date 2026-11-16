"""Secrets and environment settings (pydantic-settings) plus the YAML runtime config.

Secrets come only from the environment / .env and are never logged: `Settings.safe_dict()`
is the only representation that should ever be printed.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ExecutionAdapterName = Literal["paper", "angelone", "upstox", "flattrade"]
LLMBackendName = Literal["ollama", "inprocess", "hosted_free", "none"]


class Settings(BaseSettings):
    """Environment-backed settings. Prefix: PTAA_."""

    model_config = SettingsConfigDict(
        env_prefix="PTAA_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Execution safety
    live_trading_enabled: bool = False
    execution_adapter: ExecutionAdapterName = "paper"

    # Angel One
    angel_api_key: SecretStr | None = None
    angel_client_code: str | None = None
    angel_pin: SecretStr | None = None
    angel_totp_secret: SecretStr | None = None

    # Upstox
    upstox_api_key: SecretStr | None = None
    upstox_api_secret: SecretStr | None = None
    upstox_redirect_uri: str = "http://127.0.0.1:8765/callback"
    upstox_access_token: SecretStr | None = None

    # Flattrade
    flattrade_api_key: SecretStr | None = None
    flattrade_api_secret: SecretStr | None = None
    flattrade_user_id: str | None = None

    # LLM
    llm_backend: LLMBackendName = "ollama"
    llm_model: str = "qwen3:4b"
    ollama_host: str = "http://127.0.0.1:11434"
    hosted_free_base_url: str | None = None
    hosted_free_api_key: SecretStr | None = None

    # Alerts / inputs
    tradingview_secret: SecretStr | None = None
    telegram_bot_token: SecretStr | None = None
    telegram_chat_id: str | None = None

    # Storage
    data_dir: Path = Path("./data")

    @property
    def has_angel(self) -> bool:
        return all(
            [self.angel_api_key, self.angel_client_code, self.angel_pin, self.angel_totp_secret]
        )

    @property
    def has_upstox(self) -> bool:
        return bool(self.upstox_api_key and self.upstox_api_secret)

    @property
    def has_flattrade(self) -> bool:
        return bool(self.flattrade_api_key and self.flattrade_api_secret and self.flattrade_user_id)

    def safe_dict(self) -> dict[str, Any]:
        """Settings with every secret masked; the only form that may be logged or displayed."""
        out: dict[str, Any] = {}
        for name, value in self.model_dump().items():
            field = type(self).model_fields[name]
            if field.annotation is not None and "SecretStr" in str(field.annotation):
                out[name] = "••••" if value else None
            else:
                out[name] = str(value) if isinstance(value, Path) else value
        return out


# --------------------------------------------------------------------------- YAML config


class RiskLimits(BaseModel):
    max_risk_per_trade_pct: float = Field(gt=0, le=5)
    max_position_pct: float = Field(gt=0, le=100)
    max_sector_pct: float = Field(gt=0, le=100)
    max_open_positions: int = Field(ge=1, le=50)
    max_gross_exposure_pct: float = Field(gt=0, le=200)
    daily_loss_limit_pct: float | None = Field(default=None, gt=0, le=20)
    drawdown_halt_pct: float = Field(gt=0, le=50)
    stop_atr_min: float = Field(gt=0)
    stop_atr_max: float = Field(gt=0)
    min_reward_risk: float = Field(gt=0)
    min_avg_turnover_cr: float = Field(ge=0)
    results_blackout_days: int = Field(ge=0)
    ensemble_floor: float = Field(ge=0, le=1)
    model_max_age_days: int = Field(ge=1)


class FeeConfig(BaseModel):
    brokerage_per_order_inr: float = 0.0
    brokerage_intraday_pct: float = 0.03
    brokerage_intraday_cap_inr: float = 20.0
    stt_delivery_pct: float = 0.1
    stt_intraday_sell_pct: float = 0.025
    exchange_txn_pct: float = 0.00297
    sebi_fee_pct: float = 0.0001
    gst_pct: float = 18.0
    stamp_duty_buy_delivery_pct: float = 0.015
    stamp_duty_buy_intraday_pct: float = 0.003


class SwingModule(BaseModel):
    enabled: bool = True
    decision_bar: str = "1d"
    decision_time_ist: str = "15:35"
    fill_at: Literal["next_open", "next_tick"] = "next_open"


class IntradayModule(BaseModel):
    enabled: bool = True
    decision_bar: str = "5m"
    entries_window_ist: tuple[str, str] = ("09:30", "14:45")
    square_off_ist: str = "15:10"
    fill_at: Literal["next_open", "next_tick"] = "next_tick"
    llm_top_n_per_bar: int = 3


class Modules(BaseModel):
    swing: SwingModule = SwingModule()
    intraday: IntradayModule = IntradayModule()


class FeedsConfig(BaseModel):
    primary: Literal["angelone", "upstox", "yfinance"] = "angelone"
    secondary: Literal["angelone", "upstox", "yfinance"] = "upstox"
    fallback: Literal["angelone", "upstox", "yfinance"] = "yfinance"
    failover_after_ms: int = 1500
    archive_ticks: bool = True


class UniverseConfig(BaseModel):
    swing: str = "NIFTY50"
    intraday: str = "NIFTY50_LIQUID15"
    custom: list[str] = []


class LLMConfig(BaseModel):
    backend: LLMBackendName = "ollama"
    model: str = "qwen3:4b"
    fallback_model: str = "llama3.2:3b"
    temperature: float = 0.0
    debate_rounds: int = 2


class AlertsConfig(BaseModel):
    telegram: bool = True
    desktop: bool = True
    daily_report_email: bool = False


class ResearchConfig(BaseModel):
    allowed_domains: list[str] = []
    fetch_timeout_s: int = 10
    fetch_max_bytes: int = 2_000_000


class AppConfig(BaseModel):
    paper_book_inr: float = 1_000_000
    universe: UniverseConfig = UniverseConfig()
    modules: Modules = Modules()
    feeds: FeedsConfig = FeedsConfig()
    bars: dict[str, list[str]] = {"intervals": ["1m", "5m", "1h", "1d"]}
    risk: dict[Literal["swing", "intraday"], RiskLimits]
    fees: FeeConfig = FeeConfig()
    llm: LLMConfig = LLMConfig()
    alerts: AlertsConfig = AlertsConfig()
    research: ResearchConfig = ResearchConfig()


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Load and validate the YAML config. Raises on any invalid value."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return AppConfig.model_validate(raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
