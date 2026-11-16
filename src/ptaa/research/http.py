"""Guarded HTTP client: domain allow-list, timeout, size cap, cache. The only way out to the web.

Fetched text is untrusted data. Callers wrap it in delimiters before any LLM sees it, and the
Verifier checks quotes against the cached text, never against what a model says it read.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)

UA = "ptaa-research/0.1 (+paper-trading research bot; respects robots and rate limits)"


class DomainNotAllowedError(ValueError):
    pass


@dataclass
class Fetched:
    url: str
    status: int
    text: str
    fetched_at: float
    from_cache: bool = False


def host_allowed(url: str, allowed: list[str]) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in allowed)


class GuardedHTTP:
    def __init__(
        self,
        allowed_domains: list[str],
        cache_dir: Path,
        timeout_s: int = 10,
        max_bytes: int = 2_000_000,
        min_gap_s: float = 1.0,
        ttl_s: int = 6 * 3600,
    ):
        self.allowed = [d.lower() for d in allowed_domains]
        self.cache_dir = Path(cache_dir) / "pages"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s, self.max_bytes, self.min_gap_s, self.ttl_s = (
            timeout_s,
            max_bytes,
            min_gap_s,
            ttl_s,
        )
        self._s = requests.Session()
        self._s.headers.update(
            {
                "User-Agent": UA,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
        )
        self._last: dict[str, float] = {}

    def _key(self, url: str) -> Path:
        return self.cache_dir / (hashlib.sha256(url.encode()).hexdigest()[:32] + ".json")

    def cached(self, url: str) -> Fetched | None:
        p = self._key(url)
        if not p.exists():
            return None
        d = json.loads(p.read_text(encoding="utf-8"))
        return Fetched(
            url=url, status=d["status"], text=d["text"], fetched_at=d["fetched_at"], from_cache=True
        )

    def get(self, url: str, force: bool = False) -> Fetched:
        if not host_allowed(url, self.allowed):
            raise DomainNotAllowedError(f"{urlparse(url).hostname} is not on the allow-list")
        if not force:
            hit = self.cached(url)
            if hit and time.time() - hit.fetched_at < self.ttl_s:
                return hit
        host = urlparse(url).hostname or ""
        gap = self.min_gap_s - (time.time() - self._last.get(host, 0.0))
        if gap > 0:
            time.sleep(gap)
        r = self._s.get(url, timeout=self.timeout_s, stream=True, allow_redirects=True)
        self._last[host] = time.time()
        if r.url and not host_allowed(r.url, self.allowed):
            r.close()
            raise DomainNotAllowedError(
                f"redirected to {urlparse(r.url).hostname}, not on the allow-list"
            )
        buf = bytearray()
        for chunk in r.iter_content(65536):
            buf.extend(chunk)
            if len(buf) > self.max_bytes:
                r.close()
                break
        text = buf.decode(r.encoding or "utf-8", errors="replace")
        f = Fetched(url=url, status=r.status_code, text=text, fetched_at=time.time())
        self._key(url).write_text(
            json.dumps({"status": f.status, "text": f.text, "fetched_at": f.fetched_at}),
            encoding="utf-8",
        )
        return f
