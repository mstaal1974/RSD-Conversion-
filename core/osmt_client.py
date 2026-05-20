"""
core/osmt_client.py

Thin OSMT (Open Skills Management Tool) REST client.

OSMT is the reference implementation of Rich Skill Descriptors (RSDs) maintained
by the Open Skills Network — https://github.com/opensalt/osmt. We use its
standard JWT-based auth and the public skills endpoints.

Required env vars (or st.secrets):
  OSMT_BASE_URL       — e.g. https://osmt.example.org
  OSMT_API_TOKEN      — long-lived API token (preferred), OR
  OSMT_USERNAME +     — username/password fallback (uses /api/auth/login)
  OSMT_PASSWORD

Usage:
    client = OSMTClient.from_env()
    client.health()                              # liveness check
    summary = client.bulk_upsert_skills(rsd_jsonl_iter)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Iterable, Iterator
from urllib.parse import urljoin

import urllib.request
import urllib.error


class OSMTError(RuntimeError):
    """Any non-2xx response or transport error from OSMT."""


@dataclass
class OSMTClient:
    base_url: str
    token: str = ""
    username: str = ""
    password: str = ""
    timeout: int = 30
    _headers: dict[str, str] = field(default_factory=dict)

    # ───────────────────────────────────────────────────────────────────
    # Construction
    # ───────────────────────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> "OSMTClient":
        base = os.getenv("OSMT_BASE_URL", "").rstrip("/")
        if not base:
            raise OSMTError("OSMT_BASE_URL not set")
        client = cls(
            base_url=base,
            token=os.getenv("OSMT_API_TOKEN", ""),
            username=os.getenv("OSMT_USERNAME", ""),
            password=os.getenv("OSMT_PASSWORD", ""),
        )
        client._authenticate()
        return client

    def _authenticate(self) -> None:
        if self.token:
            self._headers["Authorization"] = f"Bearer {self.token}"
            return
        if self.username and self.password:
            body = json.dumps({"username": self.username, "password": self.password}).encode()
            resp = self._raw_request("POST", "/api/auth/login", body=body,
                                     headers={"Content-Type": "application/json"})
            tok = resp.get("token") or resp.get("access_token")
            if not tok:
                raise OSMTError(f"OSMT login returned no token: {resp}")
            self.token = tok
            self._headers["Authorization"] = f"Bearer {tok}"
            return
        raise OSMTError(
            "OSMT credentials not configured. Set OSMT_API_TOKEN or "
            "OSMT_USERNAME + OSMT_PASSWORD."
        )

    # ───────────────────────────────────────────────────────────────────
    # Low-level HTTP
    # ───────────────────────────────────────────────────────────────────

    def _raw_request(self, method: str, path: str, *,
                     body: bytes | None = None,
                     headers: dict[str, str] | None = None) -> dict | list:
        url = urljoin(self.base_url + "/", path.lstrip("/"))
        merged = {"Accept": "application/json", **self._headers, **(headers or {})}
        req = urllib.request.Request(url, data=body, method=method, headers=merged)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read().decode("utf-8") or "null"
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            raise OSMTError(f"{method} {url} → {e.code} {e.reason}: {detail}") from e
        except urllib.error.URLError as e:
            raise OSMTError(f"{method} {url} → transport error: {e}") from e

    # ───────────────────────────────────────────────────────────────────
    # Public API
    # ───────────────────────────────────────────────────────────────────

    def health(self) -> dict:
        """Round-trip the auth + a small endpoint to confirm the client works."""
        return self._raw_request("GET", "/api/skills?size=1")

    def create_skill(self, rsd: dict) -> dict:
        """Create one RSD on the OSMT instance. Returns the created skill record."""
        body = json.dumps([rsd]).encode()
        resp = self._raw_request(
            "POST", "/api/skills",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        return resp[0] if isinstance(resp, list) and resp else resp

    def bulk_upsert_skills(
        self,
        records: Iterable[dict],
        batch_size: int = 50,
        progress_callback=None,
    ) -> dict:
        """
        Push many RSDs to OSMT, batched. Returns summary counts.

        OSMT accepts a JSON array of skills per /api/skills POST. We chunk
        the iterator to keep request bodies reasonable.
        """
        ok = err = 0
        errors: list[str] = []
        batch: list[dict] = []

        def flush(b: list[dict]) -> None:
            nonlocal ok, err
            if not b:
                return
            try:
                body = json.dumps(b).encode()
                self._raw_request(
                    "POST", "/api/skills",
                    body=body,
                    headers={"Content-Type": "application/json"},
                )
                ok += len(b)
            except OSMTError as e:
                err += len(b)
                errors.append(str(e)[:300])

        for i, rec in enumerate(records):
            batch.append(rec)
            if len(batch) >= batch_size:
                flush(batch)
                batch = []
                if progress_callback:
                    progress_callback((ok + err), f"Pushed {ok}, {err} errors")
        flush(batch)
        if progress_callback:
            progress_callback((ok + err), f"Done — pushed {ok}, {err} errors")
        return {"pushed": ok, "errors": err, "error_messages": errors}


# ───────────────────────────────────────────────────────────────────────
# Convenience: stream a JSONL file as Python dicts
# ───────────────────────────────────────────────────────────────────────

def iter_jsonl(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
